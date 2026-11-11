# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.5 device tests: the traced path with recurrent models (the act step steps the hidden state IN PLACE inside the trace, the mini-batch step unrolls the sequence and its BPTT) is BIT-identical to
the eager path, hidden states included, and `load()` keeps the captured traces valid.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8nt -- python -m pytest -m tt tests/tt/test_tt_rnn_trace.py -s -q -p no:cacheprovider

Like `test_tt_trace.py` this file opens its own mesh with a trace region (run it on its own); also runs on 1x2 / 2x4 with `TT_TEST_MESH`.
"""

import torch

import pytest

from tests.tt.test_tt_rnn_update import rnn_spec
from tests.tt.test_tt_trace import assert_state_equal, build, host_state, release, run_iteration, traced_mesh  # noqa: F401  (traced_mesh is a fixture)

pytestmark = pytest.mark.tt


def full_state(alg):
    """The MLP-side state of `host_state` plus every recurrent parameter and the live hidden states."""
    ctx = alg.ctx
    out = host_state(alg)
    for tag, model in (("actor", alg.tt_actor), ("critic", alg.tt_critic)):
        if model.recurrent:
            for k, p in enumerate(model.rnn.parameters()):
                out[f"{tag}.rnn.{k}"] = ctx.download(p, op="H-6", batch_sharded=False)
            for k, s in enumerate(model.state):
                out[f"{tag}.state.{k}"] = ctx.download(s, op="H-6")
    return out


@pytest.mark.parametrize("case", [dict(rnn_type="gru", layers=1), dict(rnn_type="lstm", layers=2), dict(rnn_type="gru", layers=1, critic=False)], ids=["gru", "lstm-2-layers", "gru-actor-only"])
def test_traced_recurrent_training_is_bit_identical_to_eager(traced_mesh, case, tmp_path):  # noqa: F811
    spec = rnn_spec(**case)
    eager, _ = build("cartpole_po", tmp_path / "eager", trace=False, spec=spec)
    traced, _ = build("cartpole_po", tmp_path / "traced", trace=True, spec=spec)
    try:
        assert traced.alg.recurrent
        torch.manual_seed(4242)
        eager.learn(num_learning_iterations=4, init_at_random_ep_len=True)
        torch.manual_seed(4242)
        traced.learn(num_learning_iterations=4, init_at_random_ep_len=True)
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


def test_traced_recurrent_path_survives_load(traced_mesh, tmp_path):  # noqa: F811
    spec = rnn_spec("lstm", 1)
    a, env_a = build("cartpole_po", tmp_path / "a", trace=True, seed=1, spec=spec)
    b, _ = build("cartpole_po", tmp_path / "b", trace=True, seed=2, spec=spec)
    try:
        a.learn(num_learning_iterations=2)
        ckpt = tmp_path / "a.pt"
        a.save(str(ckpt))
        b.learn(num_learning_iterations=1)
        trace_ids = dict(b.alg._traces)
        b.load(str(ckpt), map_location="cpu")
        assert b.alg._traces == trace_ids
        sa, sb = full_state(a.alg), full_state(b.alg)
        for k in sa:
            if ".state." not in k:  # the hidden state is not part of a checkpoint (upstream does not save it either)
                assert torch.equal(sa[k], sb[k]), f"after load: {k} differs"
        for model_a, model_b in ((a.alg.tt_actor, b.alg.tt_actor), (a.alg.tt_critic, b.alg.tt_critic)):  # ... so a continued run starts from the same state
            model_b.state = [b.alg.ctx.put(s, t) for s, t in zip(model_b.state, model_a.state)]
        steps = a.alg.storage.num_transitions_per_env
        for it in range(2):
            la, recorded = run_iteration(a.alg, env_a, steps, seed=500 + 100 * it)
            lb, _ = run_iteration(b.alg, None, steps, seed=500 + 100 * it, recorded=recorded)
            assert la == lb, f"iteration {it}: losses differ after load: {la} vs {lb}"
            assert_state_equal(full_state(a.alg), full_state(b.alg), f"iteration {it} after load")
    finally:
        release(a, b)
