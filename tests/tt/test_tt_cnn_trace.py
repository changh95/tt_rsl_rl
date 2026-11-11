# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.6 device tests: the traced path with image encoders (the encoder's gathers, matmuls and its hand-written backward are part of the captured act / mini-batch steps) is BIT-identical to the eager
path, and `load()` keeps the captured traces valid.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8ct -- python -m pytest -m tt tests/tt/test_tt_cnn_trace.py -s -q -p no:cacheprovider

Like `test_tt_trace.py` this file opens its own mesh with a trace region (run it on its own); also runs on 1x2 / 2x4 with `TT_TEST_MESH`.
"""

import torch

import pytest

from tests.tt.test_tt_cnn_update import cnn_spec
from tests.tt.test_tt_trace import assert_state_equal, build, host_state, release, run_iteration, traced_mesh  # noqa: F401  (traced_mesh is a fixture)

pytestmark = pytest.mark.tt


def full_state(alg):
    """The MLP-side state of `host_state` plus every encoder parameter."""
    ctx = alg.ctx
    out = host_state(alg)
    for tag, model in (("actor", alg.tt_actor), ("critic", alg.tt_critic)):
        if model.encoder is not None:
            for k, p in enumerate(model.encoder.parameters()):
                out[f"{tag}.enc.{k}"] = ctx.download(p, op="H-6", batch_sharded=False)
        out[f"{tag}.head0"] = ctx.download(model.mlp.layers[0].w, op="H-6", batch_sharded=False)  # (the first layer has the encoder's latent layout; `host_state` reads it via the torch layout)
    return out


def test_traced_cnn_training_is_bit_identical_to_eager(traced_mesh, tmp_path):  # noqa: F811
    spec = cnn_spec()
    eager, _ = build("cartpole_img", tmp_path / "eager", trace=False, spec=spec)
    traced, _ = build("cartpole_img", tmp_path / "traced", trace=True, spec=spec)
    try:
        assert traced.alg.tt_actor.cnn is not None
        torch.manual_seed(4242)
        eager.learn(num_learning_iterations=3, init_at_random_ep_len=True)
        torch.manual_seed(4242)
        traced.learn(num_learning_iterations=3, init_at_random_ep_len=True)
        assert set(traced.alg._traces) >= {"act", "gae", "update.step.0", "update.step.3"}, "the traces were not captured"
        ea, ta = (tmp_path / "eager" / "metrics.csv").read_text().splitlines(), (tmp_path / "traced" / "metrics.csv").read_text().splitlines()
        cols = ea[0].split(",")
        assert cols == ta[0].split(",")
        differ, checked = {}, 0
        for col in ("Loss/value", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate", "Policy/mean_std", "Train/mean_reward"):
            if col not in cols:
                continue
            k = cols.index(col)
            va, vb = [r.split(",")[k] for r in ea[1:]], [r.split(",")[k] for r in ta[1:]]
            if va != vb:
                differ[col] = (va, vb)
            checked += 1
        assert not differ, "logged scalars differ between eager and traced: " + "; ".join(f"{c}: eager {a} traced {b}" for c, (a, b) in differ.items())
        assert checked >= 4
        assert_state_equal(full_state(eager.alg), full_state(traced.alg), "eager vs traced")
    finally:
        release(traced)


def test_traced_cnn_path_survives_load(traced_mesh, tmp_path):  # noqa: F811
    spec = cnn_spec()
    a, env_a = build("cartpole_img", tmp_path / "a", trace=True, seed=1, spec=spec)
    b, _ = build("cartpole_img", tmp_path / "b", trace=True, seed=2, spec=spec)
    try:
        a.learn(num_learning_iterations=2)
        ckpt = tmp_path / "a.pt"
        a.save(str(ckpt))
        b.learn(num_learning_iterations=1)
        trace_ids = dict(b.alg._traces)
        b.load(str(ckpt), map_location="cpu")
        assert b.alg._traces == trace_ids
        assert_state_equal(full_state(a.alg), full_state(b.alg), "after load")
        steps = a.alg.storage.num_transitions_per_env
        for it in range(2):
            la, recorded = run_iteration(a.alg, env_a, steps, seed=500 + 100 * it)
            lb, _ = run_iteration(b.alg, None, steps, seed=500 + 100 * it, recorded=recorded)
            assert la == lb, f"iteration {it}: losses differ after load: {la} vs {lb}"
            assert_state_equal(full_state(a.alg), full_state(b.alg), f"iteration {it} after load")
    finally:
        release(a, b)
