# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests of the CPU test environments (BRIEF Phase 1)."""

import math
import time
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.env import VecEnv
from rsl_rl_tt.testing.envs import ENV_CLASSES, ENV_NAMES, BatchedCartPole, BatchedPendulum, Tracking12, make_env


def _rollout(env, steps, seed=123):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(steps):
        a = torch.randn(env.num_envs, env.num_actions, generator=g)
        obs, r, d, ex = env.step(a)
        out.append((obs.clone(), r.clone(), d.clone(), ex["time_outs"].clone()))
    return out


@pytest.mark.parametrize("name", ENV_NAMES)
def test_vecenv_contract_shapes_dtypes(name):
    env = make_env(name, 64, seed=0)
    assert isinstance(env, VecEnv)
    assert env.num_envs == 64 and env.max_episode_length == type(env).time_out_steps
    assert env.episode_length_buf.shape == (64,) and env.episode_length_buf.dtype == torch.long
    obs = env.get_observations()
    assert isinstance(obs, TensorDict) and set(obs.keys()) == set(type(env).obs_group_dims)
    for key, dim in type(env).obs_group_dims.items():
        assert obs[key].shape == (64, dim) and obs[key].dtype == torch.float32
    for obs, r, d, to in _rollout(env, 30):
        assert r.shape == (64,) and r.dtype == torch.float32 and torch.isfinite(r).all()
        assert d.shape == (64,) and d.dtype == torch.long and set(d.unique().tolist()) <= {0, 1}  # Isaac Lab wrapper convention
        assert to.dtype == torch.bool and bool((d >= to.long()).all())  # a time-out is always a done
        assert all(torch.isfinite(obs[k]).all() for k in obs.keys())


@pytest.mark.parametrize("name", ENV_NAMES)
def test_deterministic_given_seed_and_actions(name):
    a, b, c = (make_env(name, 32, seed=s) for s in (5, 5, 6))
    ra, rb, rc = _rollout(a, 40), _rollout(b, 40), _rollout(c, 40)
    for (oa, rwa, da, ta), (ob, rwb, db, tb) in zip(ra, rb):
        assert torch.equal(oa["policy"], ob["policy"]) and torch.equal(rwa, rwb) and torch.equal(da, db) and torch.equal(ta, tb)
    assert not torch.equal(ra[-1][0]["policy"], rc[-1][0]["policy"])


@pytest.mark.parametrize("name", ENV_NAMES)
def test_global_rng_is_untouched(name):
    """PPO's action noise and mini-batch shuffle use the global generator; the env must not consume it."""
    env = make_env(name, 32, seed=1)
    before = torch.get_rng_state()
    _rollout(env, 20)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("name", ENV_NAMES)
def test_observations_stay_valid_after_the_next_step(name):
    """VecEnv docstring: the returned observations must not change in place when the env is stepped again."""
    env = make_env(name, 32, seed=2)
    obs, *_ = env.step(torch.zeros(32, env.num_actions))
    snapshot = obs.clone()
    env.step(torch.ones(32, env.num_actions))
    for k in snapshot.keys():
        assert torch.equal(obs[k], snapshot[k])


def test_pendulum_fixed_length_time_out_exactly_at_200():
    env = BatchedPendulum(16, seed=0)
    for t in range(1, 201):
        _, _, d, ex = env.step(torch.zeros(16, 1))
        expect = 1 if t == 200 else 0
        assert d.tolist() == [expect] * 16 and ex["time_outs"].long().tolist() == [expect] * 16, t
    assert env.episode_length_buf.tolist() == [0] * 16
    assert "log" in ex and ex["log"]["Episode_Termination/time_out"] == 1.0  # the final step is a reset step


def test_pendulum_dynamics_and_reward_at_the_upright_equilibrium():
    env = BatchedPendulum(4, seed=0)
    env._s = torch.zeros(4, 2)
    _, r, _, _ = env.step(torch.zeros(4, 1))
    assert torch.allclose(r, torch.zeros(4), atol=1e-7)  # th = 0, w = 0, u = 0 -> zero cost, and the state stays there
    assert torch.allclose(env._s, torch.zeros(4, 2), atol=1e-7)
    env._s = torch.tensor([[math.pi, 0.0]] * 4)
    _, r, _, _ = env.step(torch.zeros(4, 1))
    assert torch.allclose(r, torch.full((4,), -(math.pi**2)), atol=1e-5)  # hanging down: cost = pi^2


def test_cartpole_terminates_early_and_resets():
    env = BatchedCartPole(256, seed=0)
    lengths, terminated_seen = [], False
    for _ in range(120):
        before = env.episode_length_buf.clone()
        obs, r, d, ex = env.step(torch.full((256, 1), 1.0))  # constant push -> the pole falls
        fin = d.bool()
        if fin.any():
            terminated_seen = True
            assert not ex["time_outs"][fin].any()  # falling is a termination, not a time-out
            lengths += (before[fin] + 1).tolist()
            assert obs["policy"][fin].abs().max() <= 0.05 + 1e-6  # the returned obs is the post-reset state
            assert (env.episode_length_buf[fin] == 0).all()
            assert "log" in ex and ex["log"]["Episode_Termination/terminated"] == 1.0
    assert terminated_seen and max(lengths) < 120


def test_tracking12_groups_privileged_and_episode_log():
    env = Tracking12(512, seed=3)
    assert Tracking12.obs_groups_cfg == {"actor": ["policy"], "critic": ["policy", "privileged"]}  # privileged is critic-only
    priv0 = env.get_observations()["privileged"].clone()
    assert priv0.shape == (512, 48)
    for lo, hi, block in ((0.7, 1.3, priv0[:, :12]), (0.5, 1.5, priv0[:, 12:24]), (0.8, 1.2, priv0[:, 24:36]), (-0.5, 0.5, priv0[:, 36:])):
        assert lo <= block.min() and block.max() <= hi  # mass, damping, gain, bias
    saw_log = saw_term = saw_timeout = False
    g = torch.Generator().manual_seed(9)
    for _ in range(320):
        priv_before = env.get_observations()["privileged"].clone()
        obs, r, d, ex = env.step(torch.randn(512, 12, generator=g))
        fin = d.bool()
        keep = ~fin
        assert torch.equal(obs["privileged"][keep], priv_before[keep])  # constant within an episode
        if fin.any():
            assert not torch.equal(obs["privileged"][fin], priv_before[fin])  # redrawn at reset
            log = ex["log"]
            saw_log = True
            assert {"Episode_Reward/tracking", "Episode_Reward/action_rate", "Episode_Reward/effort", "Episode_Reward/termination", "Episode_Termination/terminated", "Episode_Termination/time_out"} <= set(log)
            assert all(k.count("/") == 1 for k in log)
            term = (fin & ~ex["time_outs"]).any()
            saw_term |= bool(term)
            saw_timeout |= bool(ex["time_outs"].any())
        else:
            assert "log" not in ex  # Isaac Lab sends episode statistics only on reset steps
    assert saw_log and saw_term and saw_timeout  # both early termination and time-outs occur


def test_tracking12_dynamics_one_step_matches_the_formula():
    env = Tracking12(2, seed=0)
    for name, val in (("mass", 1.0), ("damping", 0.0), ("gain", 1.0), ("bias", 0.0), ("q", 0.0), ("qd", 0.0)):
        getattr(env, name).fill_(val)
    env.q_cmd.fill_(0.0)
    env.step(torch.ones(2, 12))
    assert torch.allclose(env.qd, torch.full((2, 12), 4.0 * 0.02))  # qdd = tau_max * a / m
    assert torch.allclose(env.q, torch.full((2, 12), 4.0 * 0.02 * 0.02))  # semi-implicit Euler


@pytest.mark.parametrize("name", ENV_NAMES)
def test_random_initial_episode_lengths_are_respected(name):
    """The runner's init_at_random_ep_len writes episode_length_buf; the env must read it for its time-outs."""
    env = make_env(name, 64, seed=0)
    env.episode_length_buf = torch.full((64,), env.max_episode_length - 1, dtype=torch.long)
    _, _, d, ex = env.step(torch.zeros(64, env.num_actions))
    assert d.all()  # every env finishes at the step where its (randomised) episode counter reaches the limit


@pytest.mark.parametrize("name", ENV_NAMES)
def test_speed_is_sufficient_for_thousands_of_envs(name):
    env = make_env(name, 4096, seed=0)
    a = torch.zeros(4096, env.num_actions)
    env.step(a)
    t0 = time.time()
    for _ in range(10):
        env.step(a)
    ms = (time.time() - t0) / 10 * 1e3
    assert ms < 150, f"{name}: {ms:.1f} ms/step at 4096 envs (measured ~0.3-2 ms on an idle 64-core host)"


def test_make_env_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown test env"):
        make_env("humanoid", 8)
    assert set(ENV_CLASSES) == {"cartpole", "pendulum", "tracking12"}
