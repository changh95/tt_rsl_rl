# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.6 device tests, part 1 (single-chip semantics: whole-batch references): the CNN encoder of `rsl_rl_tt/cnn.py` against `rsl_rl.modules.CNN` - the forward pass and the hand-written gradients of every
parameter against autograd for the supported layer types (stride, zero padding, activations, max pool, global average pool) and the exactness of the row gathers it is built from.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8c -- python -m pytest -m tt tests/tt/test_tt_cnn.py -s -q -p no:cacheprovider
"""

import json
import torch
from pathlib import Path

import pytest

from rsl_rl.modules import CNN
from rsl_rl_tt.cnn import TTCNN
from rsl_rl_tt.testing.parity import compare
from rsl_rl_tt.tt_common import pad_matrix

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
MEASURED: dict = {}
B = 32


@pytest.fixture(scope="module", autouse=True)
def _dump():
    yield
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase8_6_cnn.json").write_text(json.dumps(MEASURED, indent=2, default=str))


def rec(key, value):
    MEASURED[key] = value
    print(f"[measure] {key}: {value}")
    return value


CONFIGS = {
    "one-conv": dict(input_dim=(8, 8), input_channels=1, output_channels=[4], kernel_size=3, activation="elu"),
    "stride-2-padding": dict(input_dim=(8, 8), input_channels=2, output_channels=[6], kernel_size=3, stride=2, padding="zeros", activation="relu"),
    "two-convs-tanh": dict(input_dim=(8, 8), input_channels=1, output_channels=[4, 6], kernel_size=[3, 3], stride=[1, 2], padding="zeros", activation="tanh"),
    "max-pool": dict(input_dim=(8, 8), input_channels=1, output_channels=[4, 4], kernel_size=3, padding="zeros", max_pool=[True, False], activation="elu"),
    "global-avg-pool": dict(input_dim=(8, 8), input_channels=2, output_channels=[5], kernel_size=3, padding="zeros", global_pool="avg", activation="elu"),
    "rectangular-5x5": dict(input_dim=(10, 6), input_channels=1, output_channels=[3], kernel_size=5, padding="zeros", activation="elu"),
}


def build(ctx, cfg, seed=0):
    torch.manual_seed(seed)
    cnn = CNN(**cfg)
    cnn.init_weights()
    for m in cnn:
        if isinstance(m, torch.nn.Conv2d):
            m.bias.data.normal_(0.0, 0.1)
    return cnn, TTCNN(ctx, cnn, cfg["input_dim"], cfg["input_channels"], [B])


def flat_images(ctx, x):
    return ctx.upload(pad_matrix(x.reshape(x.shape[0], -1), x.shape[0], -(-x.shape[1] * x.shape[2] * x.shape[3] // 32) * 32), op="H-1", batch_sharded=True)


def latent_to_torch(rnn, latent_host, batch):
    """[B, H W C_p] device latent -> torch's flatten order [B, C H W]."""
    return latent_host[:, rnn.latent_rows()]


@pytest.mark.parametrize("name", list(CONFIGS))
def test_forward_and_gradients_match_torch(tt_ctx, name):
    ctx = tt_ctx
    cfg = CONFIGS[name]
    cnn, rnn = build(ctx, cfg)
    g = torch.Generator().manual_seed(1)
    h, w = cfg["input_dim"]
    x = torch.randn(B, cfg["input_channels"], h, w, generator=g)
    want = cnn(x)
    latent, cache = rnn.forward(flat_images(ctx, x), B, train=True)
    got = latent_to_torch(rnn, ctx.download(latent, op="H-6"), B)
    rep = compare(got, want.detach(), "mlp_forward", f"{name} forward")
    assert rep.passed, str(rep)
    # the padded channels of the latent are exactly 0
    lat = ctx.download(latent, op="H-6").reshape(B, -1, rnn.out_c_p)
    assert float(lat[:, :, rnn.out_channels :].abs().max()) == 0.0
    # gradients of a nonlinear functional of the latent
    proj = torch.randn(want.shape[1], generator=g)
    cnn.zero_grad()
    (want * proj).pow(2).sum().backward()
    dl = torch.zeros(B, latent.shape[1])
    dl[:, rnn.latent_rows()] = 2.0 * want.detach() * proj * proj  # d (want proj)^2 / d want
    grads = rnn.backward(cache, ctx.upload(dl.contiguous(), op="H-1", batch_sharded=True), B)
    report = {}
    for (p, idx, layer, kind), grad in zip(rnn.param_slots(), grads):
        got_g = rnn.weight_from_host(layer, kind, ctx.download(grad, op="H-6", batch_sharded=False))
        r = compare(got_g, p.grad, "mlp_backward", f"{name} layer {layer} {kind}")
        report[f"{layer}.{kind}"] = r.rel_fro_eff
        assert r.passed, str(r)
    rec(f"cnn {name}", {"forward_rel_fro": rep.rel_fro, "grad_rel_fro_eff": report})


def test_the_row_gathers_are_bit_exact_and_padding_rows_are_what_they_should_be(tt_ctx):
    """A convolution's patches are gathered with constant indices: every patch element is the exact input value (six decades of dynamic range), or the padding."""
    ctx = tt_ctx
    ttnn = ctx.ttnn
    cfg = CONFIGS["stride-2-padding"]
    cnn, rnn = build(ctx, cfg)
    h, w = cfg["input_dim"]
    g = torch.Generator().manual_seed(4)
    x = torch.randn(B, 2, h, w, generator=g) * torch.pow(10.0, torch.randint(-3, 3, (B, 2, h, w), generator=g).float())
    rows = rnn.to_rows(flat_images(ctx, x), B)
    op = rnn.convs[0]
    geom = rnn.geom[(B, op["hw_in"], op["k"], op["s"], op["p"])]
    patches = rnn._patches(geom, rows, 0.0)
    want_rows = x.permute(0, 2, 3, 1).reshape(B * h * w, 2)  # [B H W, C]
    for k, p in enumerate(patches):
        got = ctx.download(p, op="H-6")[:, :2]
        ky, kx = divmod(k, op["k"])
        want = torch.zeros(B * geom.ho * geom.wo, 2)
        for b in range(B):
            for yo in range(geom.ho):
                for xo in range(geom.wo):
                    iy, ix = yo * op["s"] + ky - op["p"][0], xo * op["s"] + kx - op["p"][1]
                    if 0 <= iy < h and 0 <= ix < w:
                        want[(b * geom.ho + yo) * geom.wo + xo] = want_rows[b * h * w + iy * w + ix]
        assert torch.equal(got, want), f"offset {k}: the gathered patch differs from the input"
    # the max-pool padding row is -3e38 (finite, below every activation)
    mp = CONFIGS["max-pool"]
    cnn2, rnn2 = build(ctx, mp)
    x2 = torch.randn(B, 1, 8, 8, generator=g) - 5.0  # all negative: a zero padding would win the max
    latent, _ = rnn2.forward(flat_images(ctx, x2), B, train=False)
    assert compare(latent_to_torch(rnn2, ctx.download(latent, op="H-6"), B), cnn2(x2).detach(), "mlp_forward", "max pool of negative inputs").passed
