# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.5 device tests, part 1 (single-chip semantics: whole-batch references): the GRU / LSTM of `rsl_rl_tt/rnn.py` against `torch.nn.GRU` / `torch.nn.LSTM` - one step, a whole sequence with
resets after dones (the state is zeroed after a done step) and the hand-written back-propagation through time against autograd, for 1 and 2 layers.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8n -- python -m pytest -m tt tests/tt/test_tt_rnn.py -s -q -p no:cacheprovider
"""

import json
import torch
from pathlib import Path

import pytest

from rsl_rl.modules import RNN
from rsl_rl_tt.rnn import TTRNN
from rsl_rl_tt.testing.parity import compare
from rsl_rl_tt.tt_common import pad_matrix

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
MEASURED: dict = {}
IN, HIDDEN, E, T = 20, 48, 64, 6  # padded to 32 / 64 on the device


@pytest.fixture(scope="module", autouse=True)
def _dump():
    yield
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase8_5_rnn.json").write_text(json.dumps(MEASURED, indent=2, default=str))


def rec(key, value):
    MEASURED[key] = value
    print(f"[measure] {key}: {value}")
    return value


def build(ctx, kind, layers, seed=0):
    torch.manual_seed(seed)
    mirror = RNN(IN, HIDDEN, layers, kind)
    return mirror, TTRNN(ctx, mirror)


def up(ctx, x, cols):
    return ctx.upload(pad_matrix(x, x.shape[0], cols), op="H-1", batch_sharded=True)


def dev_state(ctx, rnn, state_host):
    """Host state tensors (torch layout [layers, E, H] per component) -> the device's flat list."""
    comps = state_host if isinstance(state_host, tuple) else (state_host,)
    out = []
    for layer in range(rnn.num_layers):
        for comp in comps:
            out.append(up(ctx, comp[layer], rnn.hidden_p))
    return out


def host_state(ctx, rnn, dev):
    n = 2 if rnn.lstm else 1
    comps = [torch.stack([ctx.download(dev[layer * n + k], op="H-6")[:, :HIDDEN] for layer in range(rnn.num_layers)]) for k in range(n)]
    return tuple(comps) if rnn.lstm else comps[0]


def rand_state(rnn, g):
    if rnn.lstm:
        return (torch.randn(rnn.num_layers, E, HIDDEN, generator=g) * 0.5, torch.randn(rnn.num_layers, E, HIDDEN, generator=g) * 0.5)
    return torch.randn(rnn.num_layers, E, HIDDEN, generator=g) * 0.5


@pytest.mark.parametrize("kind", ["gru", "lstm"])
@pytest.mark.parametrize("layers", [1, 2])
def test_one_step_matches_torch(tt_ctx, kind, layers):
    ctx = tt_ctx
    mirror, rnn = build(ctx, kind, layers)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(E, IN, generator=g)
    state = rand_state(rnn, g)
    with torch.no_grad():
        out, new = mirror.rnn(x.unsqueeze(0), state)
    h_top, new_dev = rnn.step(up(ctx, x, rnn.in_p[0]), dev_state(ctx, rnn, state))
    rep = compare(ctx.download(h_top, op="H-6")[:, :HIDDEN], out[0], "mlp_forward", f"{kind}x{layers} output")
    assert rep.passed, str(rep)
    got = host_state(ctx, rnn, new_dev)
    for a, b in zip(got if rnn.lstm else (got,), new if rnn.lstm else (new,)):
        r = compare(a, b, "mlp_forward", f"{kind}x{layers} state")
        assert r.passed, str(r)
    # the padded columns of the state stay exactly zero
    assert float(ctx.download(h_top, op="H-6")[:, HIDDEN:].abs().max()) == 0.0
    rec(f"step {kind}x{layers}", {"output_rel_fro": rep.rel_fro})


@pytest.mark.parametrize("kind", ["gru", "lstm"])
@pytest.mark.parametrize("layers", [1, 2])
def test_sequence_and_bptt_gradients_match_autograd(tt_ctx, kind, layers):
    ctx = tt_ctx
    mirror, rnn = build(ctx, kind, layers, seed=3)
    g = torch.Generator().manual_seed(2)
    xs = torch.randn(T, E, IN, generator=g)
    dones = (torch.rand(T, E, 1, generator=g) < 0.25).float()
    state0 = rand_state(rnn, g)
    proj = torch.randn(HIDDEN, generator=g)  # a linear functional of the outputs: loss = sum_t,e,h out * proj
    # torch reference: the scan with resets, differentiated by autograd
    state = tuple(s.clone() for s in state0) if rnn.lstm else state0.clone()
    outs = []
    for t in range(T):
        out, state = mirror.rnn(xs[t].unsqueeze(0), state)
        outs.append(out[0])
        keep = 1.0 - dones[t]
        state = tuple(s * keep for s in state) if rnn.lstm else state * keep
    loss = sum((o * proj).sum() for o in outs)
    mirror.rnn.zero_grad()
    loss.backward()
    # the device
    keeps = [ctx.upload((1.0 - dones[t]).contiguous(), op="H-1", batch_sharded=True) for t in range(T)]
    outs_dev, cache = rnn.forward_sequence([up(ctx, xs[t], rnn.in_p[0]) for t in range(T)], dev_state(ctx, rnn, state0), keeps)
    worst = 0.0
    for t in range(T):
        r = compare(ctx.download(outs_dev[t], op="H-6")[:, :HIDDEN], outs[t].detach(), "mlp_forward", f"{kind}x{layers} step {t} output")
        worst = max(worst, r.rel_fro)
        assert r.passed, str(r)
    d_out = ctx.upload(pad_matrix(proj.reshape(1, -1).expand(E, -1).contiguous(), E, rnn.hidden_p), op="H-1", batch_sharded=True)
    grads, d_inputs = rnn.backward_sequence(cache, [d_out] * T, keeps)
    report = {}
    for (p, idx, layer, kind_), grad in zip(rnn.param_slots(), grads):
        got = rnn.weight_from_host(layer, kind_, ctx.download(grad, op="H-6", batch_sharded=False))
        r = compare(got, p.grad, "mlp_backward", f"{kind}x{layers} layer {layer} {kind_}")
        report[f"{layer}.{kind_}"] = r.rel_fro_eff
        assert r.passed, str(r)
    # the gradient with respect to the inputs of every step (padded columns are zero)
    xs_t = xs.clone().requires_grad_(True)
    state = tuple(s.clone() for s in state0) if rnn.lstm else state0.clone()
    loss2 = 0
    for t in range(T):
        out, state = mirror.rnn(xs_t[t].unsqueeze(0), state)
        loss2 = loss2 + (out[0] * proj).sum()
        keep = 1.0 - dones[t]
        state = tuple(s * keep for s in state) if rnn.lstm else state * keep
    loss2.backward()
    for t in range(T):
        r = compare(ctx.download(d_inputs[t], op="H-6")[:, :IN], xs_t.grad[t], "mlp_backward", f"{kind}x{layers} step {t} d input")
        assert r.passed, str(r)
        assert float(ctx.download(d_inputs[t], op="H-6")[:, IN:].abs().max()) == 0.0
    rec(f"sequence {kind}x{layers}", {"output_worst_rel_fro": worst, "grad_rel_fro_eff": report})
