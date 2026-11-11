# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests of the Unitree G1 demo environment (`rsl_rl_tt/demos/mujoco_g1.py`): conventions of Isaac Lab, determinism, the observation geometry, agreement of the
multithreaded `rollout` stepping with plain `mj_step` for position control, fall / time-out / auto-reset, commands, the private RNG and a short upstream-PPO run with the
asymmetric critic. Skipped when the model is not fetched (`scripts/fetch_g1_model.sh`)."""

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

from rsl_rl_tt.demos.mujoco_g1 import MujocoG1VecEnv, g1_scene_path, quat_to_rotation  # noqa: E402

try:
    g1_scene_path()
except FileNotFoundError:
    pytest.skip("the G1 model is not fetched: scripts/fetch_g1_model.sh", allow_module_level=True)


@pytest.fixture
def env():
    e = MujocoG1VecEnv(16, seed=3, num_threads=4)
    yield e
    e.close()


def test_conventions_of_isaac_lab(env):
    obs = env.get_observations()
    assert set(obs.keys()) == {"policy", "privileged"} and obs["policy"].shape == (16, 47) and obs["privileged"].shape == (16, 8) and obs["policy"].dtype == torch.float32
    assert env.num_envs == 16 and env.num_actions == 12 and env.max_episode_length == 1000 and env.episode_length_buf.dtype == torch.long
    obs2, rewards, dones, extras = env.step(torch.zeros(16, 12))
    assert rewards.shape == (16,) and rewards.dtype == torch.float32 and dones.dtype == torch.long and extras["time_outs"].dtype == torch.bool
    assert all(torch.isfinite(v).all() for v in obs2.values()) and torch.isfinite(rewards).all()


def test_same_seed_same_trajectory_whatever_the_thread_count(env):
    other = MujocoG1VecEnv(16, seed=3, num_threads=2)
    third = MujocoG1VecEnv(16, seed=4, num_threads=2)
    try:
        g = torch.Generator().manual_seed(0)
        diverged = False
        for _ in range(60):
            a = torch.randn(16, 12, generator=g)
            o1, r1, d1, _ = env.step(a)
            o2, r2, d2, _ = other.step(a)
            o3, r3, _, _ = third.step(a)
            assert torch.equal(o1["policy"], o2["policy"]) and torch.equal(r1, r2) and torch.equal(d1, d2)
            diverged |= not torch.equal(r1, r3)
        assert diverged
    finally:
        other.close(), third.close()


def test_projected_gravity_and_rotation_geometry():
    q = np.array([[1.0, 0, 0, 0], [np.cos(0.3), 0, 0, np.sin(0.3)], [np.cos(0.4), np.sin(0.4), 0, 0]])  # identity, a yaw, a roll
    r = quat_to_rotation(q)
    assert np.allclose(np.einsum("nij,nkj->nik", r, r), np.eye(3)) and np.allclose(np.linalg.det(r), 1.0)
    quat = q
    g = -np.stack([2 * (quat[:, 1] * quat[:, 3] - quat[:, 0] * quat[:, 2]), 2 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1]), 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)], axis=1)
    assert np.allclose(g, np.einsum("nji,j->ni", r, np.array([0, 0, -1.0])))  # R^T (0, 0, -1)
    assert np.allclose(g[0], [0, 0, -1]) and np.allclose(g[1], [0, 0, -1])  # an upright body, yawed or not, sees gravity straight down


def test_multithreaded_rollout_matches_plain_mj_step():
    """(a) With the continuous run's solver warm start passed in, `mujoco.rollout` reproduces plain `mj_step` EXACTLY (the position-control mapping and the state handling are right);
    (b) the environment re-starts the constraint solver from zero at every control step (`rollout` does not return its warm start) and the MJX variant of the G1 uses only 5 solver
    iterations, so the result differs by the solver's convergence, an error that contact dynamics amplify with the horizon: measured <= 2.6e-3 rad over 40 steps (8.7e-4 after 2),
    bound 1e-2; it is the same physics for training and for rendering, and (a) shows that the mapping itself is exact."""
    import mujoco
    from mujoco import rollout

    e = MujocoG1VecEnv(4, seed=1, num_threads=2)
    try:
        m = e.model
        d, dr = mujoco.MjData(m), mujoco.MjData(m)
        mujoco.mj_setState(m, d, e._state[0].copy(), e._spec)
        st_warm, st_env = e._state[0].copy()[None], None
        g = np.random.default_rng(0)
        worst_warm, worst_env, compared = 0.0, 0.0, 0
        warm = np.zeros((1, m.nv))
        with rollout.Rollout(nthread=1) as ro:
            for _ in range(40):
                a = g.normal(0, 0.3, (4, 12)).astype(np.float32)
                target = e.default_q.copy()
                target[:12] += e.ACTION_SCALE * np.clip(a[0].astype(np.float64), -e.ACTION_CLIP, e.ACTION_CLIP)
                ctrl = np.clip(target, e.ctrl_low, e.ctrl_high)
                d.ctrl[:] = ctrl
                mujoco.mj_step(m, d, nstep=e.FRAME_SKIP)
                out, _ = ro.rollout(m, [dr], st_warm, np.repeat(ctrl[None, None, :], e.FRAME_SKIP, axis=1), initial_warmstart=warm)
                st_warm, warm = out[:, -1].copy(), d.qacc_warmstart[None].copy()
                e.step(torch.from_numpy(a))
                if e._steps[0] == 0:
                    break  # the environment reset (a fall): stop comparing
                worst_warm = max(worst_warm, float(np.abs(st_warm[0, 1 : 1 + e.nq] - d.qpos).max()))
                worst_env, compared = max(worst_env, float(np.abs(e._state[0, 1 : 1 + e.nq] - d.qpos).max())), compared + 1
        assert compared >= 15
        assert worst_warm < 1e-9, f"rollout with the warm start differs from mj_step by {worst_warm:.2e}"
        assert worst_env < 1e-2, f"the environment differs from mj_step by {worst_env:.2e} rad"
    finally:
        e.close()


def test_a_fall_terminates_resets_and_costs_the_penalty(env):
    env.TERMINATE_HEIGHT = 0.9  # an instance override: every pelvis (0.755 m at rest) counts as fallen
    obs, rewards, dones, extras = env.step(torch.zeros(16, 12))
    assert (dones == 1).all() and not extras["time_outs"].any() and (rewards < -1.5).all()  # the -2 penalty is not scaled by dt
    assert extras["log"]["Episode_Termination/fell"] == 1.0 and (env.episode_length_buf == 0).all()
    assert abs(float(env._state[2, 3]) - 0.755) < 0.1  # reset to the knees-bent stance


def test_time_limit_is_a_truncation_and_episode_length_buf_is_settable(env):
    env.episode_length_buf = torch.full((16,), env.max_episode_length - 1, dtype=torch.long)
    _, _, dones, extras = env.step(torch.zeros(16, 12))
    assert extras["time_outs"].any() and torch.equal(extras["time_outs"], extras["time_outs"] & dones.bool())


def test_commands_are_sampled_in_range_and_can_be_held(env):
    c = env._cmd
    assert (c[:, 0] >= 0.2).all() and (c[:, 0] <= 0.8).all() and (np.abs(c[:, 1]) <= 0.2).all() and (np.abs(c[:, 2]) <= 0.4).all()
    env.set_command((0.5, 0.0, 0.3))
    env.episode_length_buf = torch.full((16,), env.COMMAND_RESAMPLE_STEPS - 1, dtype=torch.long)
    env.step(torch.zeros(16, 12))
    assert np.allclose(env._cmd, [0.5, 0.0, 0.3])  # resampling keeps the held command
    assert np.allclose(env.get_observations()["policy"][:, 6:9].numpy(), np.array([1.0, 0.0, 0.075]), atol=1e-6)  # command x (2, 2, 0.25)


def test_the_global_random_generator_is_untouched():
    torch.manual_seed(5)
    state = torch.get_rng_state()
    e = MujocoG1VecEnv(8, seed=0, num_threads=2)
    try:
        for _ in range(5):
            e.step(torch.randn(8, 12, generator=torch.Generator().manual_seed(0)))
        assert torch.equal(torch.get_rng_state(), state)
    finally:
        e.close()


def test_upstream_ppo_trains_on_it_with_the_asymmetric_critic(tmp_path):
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl_tt.demos.g1_cfg import g1_train_cfg

    e = MujocoG1VecEnv(64, seed=0, num_threads=4)
    try:
        cfg = g1_train_cfg(num_steps=8, hidden=(64, 64), num_mini_batches=2, num_epochs=2)
        torch.manual_seed(0)
        runner = OnPolicyRunner(e, cfg, log_dir=str(tmp_path), device="cpu")
        runner.learn(num_learning_iterations=3, init_at_random_ep_len=True)
        rows = (tmp_path / "metrics.csv").read_text().splitlines()
        assert len(rows) == 4 and "nan" not in rows[-1].lower()
    finally:
        e.close()
