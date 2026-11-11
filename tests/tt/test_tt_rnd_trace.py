# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.2 device tests: the traced path with RND (the predictor's mini-batch step is part of the captured step; the reward path runs eagerly per environment step on persistent
state) is BIT-identical to the eager path, and `load()` keeps the captured traces valid.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8rt -- python -m pytest -m tt tests/tt/test_tt_rnd_trace.py -s -q -p no:cacheprovider

Like `test_tt_trace.py` this file opens its own mesh with a trace region (run it on its own); also runs on 1x2 / 2x4 with `TT_TEST_MESH`.
"""

import torch

import pytest

from tests.tt.test_tt_rnd import rnd_spec
from tests.tt.test_tt_trace import assert_state_equal, build, host_state, release, run_iteration, traced_mesh  # noqa: F401  (traced_mesh is a fixture)

pytestmark = pytest.mark.tt


def rnd_state(alg):
    """Everything RND keeps on the device: predictor / target, Adam moments and step, normalizer statistics, the reward accumulator, the step counter."""
    ctx = alg.ctx
    down = lambda t: ctx.download(t, op="H-6", batch_sharded=False)
    r = alg.rnd_tt
    out = {}
    for tag, net in (("predictor", r.predictor), ("target", r.target)):
        for i, layer in enumerate(net.layers):
            out[f"rnd.{tag}.{i}.w"], out[f"rnd.{tag}.{i}.b"] = down(layer.w), down(layer.b)
    for k, (m, v) in enumerate(zip(r.adam.m, r.adam.v)):
        out[f"rnd.adam.m.{k}"], out[f"rnd.adam.v.{k}"] = down(m), down(v)
    out["rnd.adam.t"] = torch.tensor(float(r.adam.t))
    if r.state_norm is not None:
        stats = r.state_norm.host_stats()
        out["rnd.state.mean"], out["rnd.state.var"], out["rnd.state.count"] = stats["mean"], stats["var"], torch.tensor(float(stats["count"]))
    if r.reward_norm is not None:
        rn = r.reward_norm
        out["rnd.reward.mean"], out["rnd.reward.var"], out["rnd.reward.std"] = down(rn.mean), down(rn.var), down(rn.std)
        out["rnd.reward.count"] = torch.tensor(float(rn.count))
        out["rnd.reward.avg"] = ctx.download(rn.avg, op="H-6")
    out["rnd.counter"] = torch.tensor(float(alg.rnd.update_counter))
    return out


def full_state(alg):
    return {**host_state(alg), **rnd_state(alg)}


@pytest.mark.parametrize("case", ["cartpole-actor", "tracking12-own"])
def test_traced_training_with_rnd_is_bit_identical_to_eager(traced_mesh, case, tmp_path):  # noqa: F811
    """Two runners from the same seed, one eager and one traced, run 4 iterations: every logged scalar (incl. the RND loss and the intrinsic reward statistics) and every piece of device
    state (incl. the predictor, its Adam moments, the normalizers and the discounted-reward accumulator) must be EQUAL."""
    name, spec = rnd_spec(case)
    eager, _ = build(name, tmp_path / "eager", trace=False, spec=spec)
    traced, _ = build(name, tmp_path / "traced", trace=True, spec=spec)
    try:
        assert eager.alg.rnd_tt is not None and traced.alg.rnd_tt is not None
        torch.manual_seed(4242)
        eager.learn(num_learning_iterations=4, init_at_random_ep_len=True)
        torch.manual_seed(4242)
        traced.learn(num_learning_iterations=4, init_at_random_ep_len=True)
        assert set(traced.alg._traces) >= {"act", "gae", "update.step.0", "update.step.3"}, "the traces were not captured"
        ea, ta = (tmp_path / "eager" / "metrics.csv").read_text().splitlines(), (tmp_path / "traced" / "metrics.csv").read_text().splitlines()
        cols = ea[0].split(",")
        assert cols == ta[0].split(",")
        checked, differ = 0, {}
        for col in ("Loss/value", "Loss/surrogate", "Loss/entropy", "Loss/rnd", "Loss/learning_rate", "Policy/mean_std", "Train/mean_reward", "Rnd/mean_intrinsic_reward", "Rnd/mean_extrinsic_reward"):
            if col not in cols:
                continue
            k = cols.index(col)
            va, vb = [r.split(",")[k] for r in ea[1:]], [r.split(",")[k] for r in ta[1:]]
            if va != vb:
                differ[col] = (va, vb)
            checked += 1
        assert not differ, "logged scalars differ between eager and traced: " + "; ".join(f"{c}: eager {a} traced {b}" for c, (a, b) in differ.items())
        assert checked >= 5 and "Loss/rnd" in cols
        assert torch.equal(eager.alg.intrinsic_rewards, traced.alg.intrinsic_rewards)
        assert_state_equal(full_state(eager.alg), full_state(traced.alg), "eager vs traced")
    finally:
        release(traced)


def test_traced_path_with_rnd_survives_load(traced_mesh, tmp_path):  # noqa: F811
    """`load()` rewrites weights, normalizers, Adam moments, the learning rate AND the RND state in place: b follows a bit for bit after loading a's checkpoint."""
    name, spec = rnd_spec("cartpole-actor")
    a, env_a = build(name, tmp_path / "a", trace=True, seed=1, spec=spec)
    b, _ = build(name, tmp_path / "b", trace=True, seed=2, spec=spec)
    try:
        a.learn(num_learning_iterations=2)
        ckpt = tmp_path / "a.pt"
        a.save(str(ckpt))
        b.learn(num_learning_iterations=1)
        trace_ids = dict(b.alg._traces)
        assert not torch.equal(rnd_state(b.alg)["rnd.predictor.0.w"], rnd_state(a.alg)["rnd.predictor.0.w"])
        b.load(str(ckpt), map_location="cpu")
        assert b.alg._traces == trace_ids
        sa, sb = full_state(a.alg), full_state(b.alg)
        for k in sa:
            if k == "rnd.reward.avg":  # upstream does not checkpoint the discounted-reward accumulator (it restarts from the first reward): not part of a checkpoint
                continue
            assert torch.equal(sa[k], sb[k]), f"after load: {k} differs"
        b.alg.rnd_tt.reward_norm.avg = b.alg.ctx.put(b.alg.rnd_tt.reward_norm.avg, a.alg.rnd_tt.reward_norm.avg)  # ... so a continued run starts from the same accumulator
        steps = a.alg.storage.num_transitions_per_env
        for it in range(2):
            la, recorded = run_iteration(a.alg, env_a, steps, seed=500 + 100 * it)
            lb, _ = run_iteration(b.alg, None, steps, seed=500 + 100 * it, recorded=recorded)
            assert la == lb and "rnd" in la, f"iteration {it}: losses differ after load: {la} vs {lb}"
            assert_state_equal(full_state(a.alg), full_state(b.alg), f"iteration {it} after load")
    finally:
        release(a, b)
