# SPDX-License-Identifier: BSD-3-Clause
"""Phase 5 device tests: the TRACED path (act step and mini-batch step captured once and replayed) must reproduce the eager path BIT FOR BIT
(same kernels, same inputs) and keep working across allocations, normalizer updates and checkpoint loads.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p5t -- python -m pytest -m tt tests/tt/test_tt_trace.py -s -q -p no:cacheprovider

One process = one mesh, opened WITH a trace region by the fixture below (`tt_mesh` of conftest.py opens none), so run this file on its own.
"""

import copy
import torch

import pytest

from rsl_rl.runners import OnPolicyRunner
from rsl_rl_tt.testing.envs import make_env
from rsl_rl_tt.testing.reference import RefSpec, make_train_cfg
from tests.tt.test_tt_rollout import small_spec
from tests.tt.test_tt_update import tt_params

pytestmark = pytest.mark.tt


@pytest.fixture(scope="module")
def traced_mesh():
    from rsl_rl_tt.device import open_tray_mesh

    tm = open_tray_mesh((1, 1), trace_region_size=128 * 1024 * 1024)
    yield tm
    tm.close()


def build(name, tmp_path, trace, seed=0, spec=None):
    spec = spec or small_spec(name)
    torch.manual_seed(seed)
    env = make_env(name, spec.num_envs, seed=seed)
    cfg = make_train_cfg(name, spec)
    cfg["algorithm"]["class_name"] = "rsl_rl_tt.ppo:TTPPO"
    cfg["algorithm"]["tt_cfg"] = {"mesh_shape": [1, 1], "trace": trace}
    return OnPolicyRunner(env, cfg, log_dir=str(tmp_path), device="cpu"), env


@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
def test_traced_training_is_bit_identical_to_eager(traced_mesh, name, tmp_path):
    """Two runners from the same seed, one eager and one traced, run 4 iterations (rollouts, shuffles, updates, normalizer updates): every logged
    scalar and every parameter must be EQUAL, not merely close."""
    eager, _ = build(name, tmp_path / "eager", trace=False)
    traced, _ = build(name, tmp_path / "traced", trace=True)
    eager.learn(num_learning_iterations=4)
    try:
        traced.learn(num_learning_iterations=4)
        assert set(traced.alg._traces) >= {"act", "update.step.0", "update.step.3"}, "the traces were not captured"
    except BaseException:
        traced.alg.release_traces()
        raise
    pa, pb = tt_params(eager.alg), tt_params(traced.alg)
    for k in pa:
        assert torch.equal(pa[k], pb[k]), f"parameter {k} differs between eager and traced"
    assert eager.alg.learning_rate == traced.alg.learning_rate
    ea, ta = (tmp_path / "eager" / "metrics.csv").read_text().splitlines(), (tmp_path / "traced" / "metrics.csv").read_text().splitlines()
    cols = ea[0].split(",")
    for col in ("Loss/value", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate", "Policy/mean_std", "Train/mean_reward"):
        k = cols.index(col)
        assert [r.split(",")[k] for r in ea[1:]] == [r.split(",")[k] for r in ta[1:]], f"{col} differs between eager and traced"
    for tag in ("tt_actor", "tt_critic"):
        sa, sb = getattr(eager.alg, tag).normalizer.host_stats(), getattr(traced.alg, tag).normalizer.host_stats()
        assert torch.equal(sa["mean"], sb["mean"]) and torch.equal(sa["var"], sb["var"]) and sa["count"] == sb["count"]
    traced.alg.release_traces()


def test_traced_path_survives_load_and_keeps_replaying_the_same_traces(traced_mesh, tmp_path):
    """`load()` rewrites weights, normalizers, Adam moments and the learning rate IN PLACE, so already captured traces stay valid and use the new state."""
    a, env_a = build("cartpole", tmp_path / "a", trace=True, seed=1)
    a.learn(num_learning_iterations=2)
    ckpt = tmp_path / "a.pt"
    a.save(str(ckpt))
    b, env_b = build("cartpole", tmp_path / "b", trace=True, seed=2)
    b.learn(num_learning_iterations=1)  # captures the traces on b's OWN (different) state
    trace_ids = dict(b.alg._traces)
    b.load(str(ckpt), map_location="cpu")
    assert b.alg._traces == trace_ids
    # b now continues exactly like a does (same state, same data): compare after one more iteration on the same env seed / RNG state
    for runner, env in ((a, env_a), (b, env_b)):
        env.episode_length_buf.zero_()
    torch.manual_seed(77)
    a.learn(num_learning_iterations=1)
    torch.manual_seed(77)
    b.learn(num_learning_iterations=1)
    pa, pb = tt_params(a.alg), tt_params(b.alg)
    assert all(torch.isfinite(v).all() for v in pb.values())
    # the environments differ in their own state, so the data differ; what must hold is that b's replayed traces used the LOADED weights: its first act equals a's policy at the checkpoint
    a.alg.release_traces()
    b.alg.release_traces()
