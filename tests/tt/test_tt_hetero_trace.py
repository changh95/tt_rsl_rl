# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.1 device tests: the traced path with a heteroscedastic actor (the std is stored per sample, the std head is trained through the MLP) is BIT-identical to the
eager path, and `load()` keeps the captured traces valid.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8t -- python -m pytest -m tt tests/tt/test_tt_hetero_trace.py -s -q -p no:cacheprovider

Like `test_tt_trace.py` this file opens its own mesh with a trace region (run it on its own); also runs on 1x2 / 2x4 with `TT_TEST_MESH`.
"""

import torch

import pytest

from tests.tt.test_tt_rollout import small_spec
from tests.tt.test_tt_trace import N_CHIPS, assert_state_equal, build, host_state, release, run_iteration, traced_mesh  # noqa: F401  (traced_mesh is a fixture)

pytestmark = pytest.mark.tt

HETERO = "HeteroscedasticGaussianDistribution"


def spec_for(name, std_type):
    return small_spec(name, num_envs=max(128, 32 * N_CHIPS), distribution=HETERO, std_type=std_type, init_std=0.8)


@pytest.mark.parametrize("name, std_type", [("cartpole", "scalar"), ("tracking12", "log")])
def test_traced_training_is_bit_identical_to_eager(traced_mesh, name, std_type, tmp_path):  # noqa: F811
    """Two runners from the same seed, one eager and one traced, run 4 iterations: every logged scalar and every piece of device state must be EQUAL."""
    eager, _ = build(name, tmp_path / "eager", trace=False, spec=spec_for(name, std_type))
    traced, _ = build(name, tmp_path / "traced", trace=True, spec=spec_for(name, std_type))
    try:
        assert eager.alg.hetero and traced.alg.hetero and traced.alg.storage.store_std
        torch.manual_seed(4242)
        eager.learn(num_learning_iterations=4)
        torch.manual_seed(4242)
        traced.learn(num_learning_iterations=4)
        assert set(traced.alg._traces) >= {"act", "gae", "update.step.0", "update.step.3"}, "the traces were not captured"
        ea, ta = (tmp_path / "eager" / "metrics.csv").read_text().splitlines(), (tmp_path / "traced" / "metrics.csv").read_text().splitlines()
        cols = ea[0].split(",")
        assert cols == ta[0].split(",")
        checked = 0
        for col in ("Loss/value", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate", "Policy/mean_std", "Train/mean_reward"):
            if col not in cols:
                continue
            k = cols.index(col)
            va, vb = [r.split(",")[k] for r in ea[1:]], [r.split(",")[k] for r in ta[1:]]
            assert va == vb, f"{col} differs between eager and traced: eager {va} traced {vb}"
            checked += 1
        assert checked >= 4
        assert_state_equal(host_state(eager.alg), host_state(traced.alg), "eager vs traced")
        # the logged std is a real, state-dependent quantity (not the placeholder of the constructor)
        mean_std = [float(r.split(",")[cols.index("Policy/mean_std")]) for r in ta[1:]]
        assert all(0.0 < s < 10.0 for s in mean_std) and len(set(mean_std)) > 1
    finally:
        release(traced)


def test_traced_path_survives_load(traced_mesh, tmp_path):  # noqa: F811
    """`load()` rewrites weights (incl. the std head), normalizers, Adam moments and the learning rate IN PLACE: b follows a bit for bit after loading a's checkpoint."""
    spec = spec_for("cartpole", "scalar")
    a, env_a = build("cartpole", tmp_path / "a", trace=True, seed=1, spec=spec)
    b, _ = build("cartpole", tmp_path / "b", trace=True, seed=2, spec=spec)
    try:
        a.learn(num_learning_iterations=2)
        ckpt = tmp_path / "a.pt"
        a.save(str(ckpt))
        b.learn(num_learning_iterations=1)
        trace_ids = dict(b.alg._traces)
        before = host_state(b.alg)
        assert not torch.equal(before["param.actor.mlp.2.weight"], host_state(a.alg)["param.actor.mlp.2.weight"])
        b.load(str(ckpt), map_location="cpu")
        assert b.alg._traces == trace_ids
        assert_state_equal(host_state(a.alg), host_state(b.alg), "after load")
        steps = a.alg.storage.num_transitions_per_env
        for it in range(2):
            la, recorded = run_iteration(a.alg, env_a, steps, seed=500 + 100 * it)
            lb, _ = run_iteration(b.alg, None, steps, seed=500 + 100 * it, recorded=recorded)
            assert la == lb, f"iteration {it}: losses differ after load: {la} vs {lb}"
            assert_state_equal(host_state(a.alg), host_state(b.alg), f"iteration {it} after load")
    finally:
        release(a, b)
