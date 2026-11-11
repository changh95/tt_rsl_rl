# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests of the MuJoCo humanoid demo environment (`rsl_rl_tt/demos/mujoco_humanoid.py`): rsl_rl `VecEnv` conventions of Isaac Lab, determinism, the reward
formula, agreement of the multithreaded `rollout` stepping with plain `mj_step`, termination / time-out / auto-reset, the private RNG, and a short training run."""

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from rsl_rl_tt.demos.mujoco_humanoid import MujocoHumanoidVecEnv  # noqa: E402


@pytest.fixture
def env():
    e = MujocoHumanoidVecEnv(16, seed=3, num_threads=4)
    yield e
    e.close()


def test_conventions_of_isaac_lab(env):
    obs = env.get_observations()
    assert set(obs.keys()) == {"policy"} and obs["policy"].shape == (16, 45) and obs["policy"].dtype == torch.float32
    assert env.num_envs == 16 and env.num_actions == 17 and env.max_episode_length == 1000 and env.episode_length_buf.dtype == torch.long
    obs2, rewards, dones, extras = env.step(torch.zeros(16, 17))
    assert obs2["policy"].shape == (16, 45) and rewards.shape == (16,) and rewards.dtype == torch.float32
    assert dones.shape == (16,) and dones.dtype == torch.long and extras["time_outs"].dtype == torch.bool and extras["time_outs"].shape == (16,)
    assert torch.isfinite(obs2["policy"]).all() and torch.isfinite(rewards).all()


def test_same_seed_same_trajectory_different_seed_different(env):
    other, third = MujocoHumanoidVecEnv(16, seed=3, num_threads=2), MujocoHumanoidVecEnv(16, seed=4, num_threads=2)
    try:
        g = torch.Generator().manual_seed(0)
        diverged = False
        for _ in range(40):
            a = torch.randn(16, 17, generator=g)
            o1, r1, d1, _ = env.step(a)
            o2, r2, d2, _ = other.step(a)
            o3, r3, _, _ = third.step(a)
            assert torch.equal(o1["policy"], o2["policy"]) and torch.equal(r1, r2) and torch.equal(d1, d2)  # thread count does not matter
            diverged |= not torch.equal(r1, r3)
        assert diverged, "another seed must give another trajectory"
    finally:
        other.close(), third.close()


def test_reward_is_the_documented_formula(env):
    g = torch.Generator().manual_seed(1)
    for _ in range(10):
        a = torch.randn(16, 17, generator=g)
        before = env._state.copy()
        _, rewards, dones, _ = env.step(a)
        if dones.any():
            continue  # after a reset the state is a fresh one
        ctrl = 0.4 * np.clip(a.numpy().astype(np.float64), -1, 1)
        want = env.reward_scale * (1.25 * (env._state[:, 1] - before[:, 1]) / env.dt + 5.0 - 0.1 * np.square(ctrl).sum(-1))
        assert np.allclose(rewards.numpy(), want, atol=1e-6)


def test_multithreaded_rollout_matches_plain_mj_step():
    """Stepping through `mujoco.rollout` (state in, state out, no warm start between calls) is the same physics as Gymnasium's continuous `mj_step`."""
    import mujoco

    e = MujocoHumanoidVecEnv(4, seed=1, num_threads=2)
    try:
        d = mujoco.MjData(e.model)
        mujoco.mj_setState(e.model, d, e._state[0].copy(), mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
        g = np.random.default_rng(0)
        worst, compared = 0.0, 0
        for _ in range(40):
            a = g.uniform(-1, 1, (4, 17)).astype(np.float32)
            d.ctrl[:] = 0.4 * np.clip(a[0].astype(np.float64), -1, 1)
            mujoco.mj_step(e.model, d, nstep=5)
            e.step(torch.from_numpy(a))
            if e._steps[0] == 0:
                break  # the environment reset: stop comparing
            worst, compared = max(worst, float(np.abs(e._state[0, 1 : 1 + e.nq] - d.qpos).max())), compared + 1
        assert compared >= 15
        assert worst < 1e-6, f"max |qpos difference| {worst:.2e} (measured 5e-8 over 23 steps)"
    finally:
        e.close()


def test_a_fall_terminates_resets_and_is_not_a_time_out(env):
    env._state[2, 3] = 0.5  # torso height below 1.0 m
    obs, rewards, dones, extras = env.step(torch.zeros(16, 17))
    assert dones[2] == 1 and not extras["time_outs"][2]
    assert extras["log"]["Episode_Termination/fell"] > 0 and "Metrics/forward_speed" in extras["log"]
    assert env.episode_length_buf[2] == 0 and abs(float(obs["policy"][2, 0]) - 1.4) < 0.05  # reset: standing again, post-reset observation returned


def test_time_limit_is_a_truncation_and_episode_length_buf_is_settable(env):
    env.episode_length_buf = torch.full((16,), env.max_episode_length - 1, dtype=torch.long)  # what OnPolicyRunner.learn(init_at_random_ep_len=True) does
    _, _, dones, extras = env.step(torch.zeros(16, 17))
    assert dones.sum() >= 1 and torch.equal(extras["time_outs"], dones.bool() & extras["time_outs"]) and extras["time_outs"].any()
    assert (env.episode_length_buf[dones.bool()] == 0).all()


def test_the_global_random_generator_is_untouched():
    torch.manual_seed(5)
    state = torch.get_rng_state()
    e = MujocoHumanoidVecEnv(8, seed=0, num_threads=2)
    try:
        for _ in range(5):
            e.step(torch.randn(8, 17, generator=torch.Generator().manual_seed(0)))
        assert torch.equal(torch.get_rng_state(), state)
    finally:
        e.close()


def test_upstream_ppo_trains_on_it_for_a_few_iterations(tmp_path):
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl_tt.demos.humanoid_cfg import humanoid_train_cfg

    e = MujocoHumanoidVecEnv(64, seed=0, num_threads=4)
    try:
        cfg = humanoid_train_cfg(num_steps=8, hidden=(64, 64), num_mini_batches=2, num_epochs=2)
        torch.manual_seed(0)
        runner = OnPolicyRunner(e, cfg, log_dir=str(tmp_path), device="cpu")
        runner.learn(num_learning_iterations=3, init_at_random_ep_len=True)
        rows = (tmp_path / "metrics.csv").read_text().splitlines()
        assert len(rows) == 4 and "nan" not in rows[-1].lower()
    finally:
        e.close()
