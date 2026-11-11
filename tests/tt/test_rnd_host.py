# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests of the RND support code (no device): the reference configuration of the end-to-end runs, the upstream RND path on the test environments (the oracle of the device tests),
and the layout rules of the storage block of the RND state."""

import pytest
import torch

from rsl_rl.runners import OnPolicyRunner
from rsl_rl_tt.testing.envs import ENV_NAMES, make_env
from rsl_rl_tt.testing.reference import REF_SPECS, RND_E2E_WEIGHT, make_train_cfg, with_rnd


def test_every_test_env_has_a_step_dt_and_a_reference_rnd_weight():
    assert set(RND_E2E_WEIGHT) == set(ENV_NAMES)
    for name in ENV_NAMES:
        env = make_env(name, 8)
        assert env.unwrapped is env and env.unwrapped.step_dt > 0
        spec = with_rnd(name, REF_SPECS[name])
        assert spec.algorithm["rnd_cfg"]["weight"] == RND_E2E_WEIGHT[name] and "rnd_cfg" not in REF_SPECS[name].algorithm  # the registry spec is not modified


@pytest.mark.parametrize("name", ENV_NAMES)
def test_the_upstream_runner_trains_with_rnd_on_every_test_env(name, tmp_path):
    """The oracle: upstream PPO + RND on the CPU test envs. The weight is scaled by the env's step_dt, the intrinsic reward is logged, the RND state normalizer learns."""
    import dataclasses

    spec = dataclasses.replace(with_rnd(name, REF_SPECS[name]), num_envs=64, num_steps=8, iterations=2)
    torch.manual_seed(0)
    env = make_env(name, spec.num_envs, seed=0)
    runner = OnPolicyRunner(env, make_train_cfg(name, spec), log_dir=str(tmp_path), device="cpu")
    rnd = runner.alg.rnd
    assert rnd is not None and rnd.initial_weight == pytest.approx(RND_E2E_WEIGHT[name] * env.step_dt)
    assert rnd.obs_groups["rnd_state"] == ["policy"]  # no explicit "rnd_state" set: upstream falls back to the policy group
    runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    assert rnd.update_counter == 16 and int(rnd.state_normalizer.count) == 2 * 8 * 64
    assert runner.alg.intrinsic_rewards.shape == (64,) and (runner.alg.intrinsic_rewards >= 0).all()
    header = (tmp_path / "metrics.csv").read_text().splitlines()[0].split(",")
    assert "Loss/rnd" in header


def test_reward_normalization_follows_the_documented_recurrence():
    """The statement the device implementation relies on: `avg = gamma * avg + r` starts from the first reward (None -> clone), is never reset, the statistics learn from `avg`, and the output is `r / std`."""
    from rsl_rl.modules import EmpiricalDiscountedVariationNormalization

    norm = EmpiricalDiscountedVariationNormalization(shape=[], until=int(1e8))
    norm.train()
    g = torch.Generator().manual_seed(0)
    avg = None
    for step in range(4):
        r = torch.rand(32, generator=g)
        avg = r.clone() if avg is None else 0.99 * avg + r
        out = norm(r)
        assert torch.equal(norm.disc_avg.avg, avg)
        assert torch.allclose(out, r / norm.emp_norm._std)
    assert int(norm.emp_norm.count) == 4 * 32
