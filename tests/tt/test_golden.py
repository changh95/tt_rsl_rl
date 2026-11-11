# SPDX-License-Identifier: BSD-3-Clause
"""Host-only: the committed golden baselines exist, are complete, and show that upstream rsl_rl learns each test env in reasonable time
(BRIEF Phase 1 completion criterion). Retrain with `python scripts/train_reference.py`."""

import math
import statistics

import pytest

from rsl_rl_tt.testing.envs import ENV_NAMES
from rsl_rl_tt.testing.golden import GOLDEN_DIR, final_reward, load_curve, load_summary, phase4_threshold, tt_run_passes
from rsl_rl_tt.testing.reference import REF_SPECS

SEEDS = (0, 1, 2)
# "learns": every seed's final reward clears this bar. Measured random-policy mean episode returns for comparison (2048 envs, N(0,1) actions):
# cartpole 25.5, pendulum -1224, tracking12 14.9 (max return: cartpole 500, tracking12 ~300).
LEARNED = {"cartpole": 450.0, "pendulum": -250.0, "tracking12": 150.0}
MAX_WALL_S = 30 * 60  # "reasonable time": a run must finish within 30 minutes on this host
REQUIRED_COLUMNS = {"iteration", "wall_s", "Train/mean_reward", "Train/mean_episode_length", "Loss/value", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate", "Policy/mean_std"}


def test_summary_covers_every_env_and_seed():
    s = load_summary()
    assert s["rsl_rl_tag"] == "v5.5.2" and set(s["envs"]) == set(ENV_NAMES)
    for env in ENV_NAMES:
        assert set(s["envs"][env]["seeds"]) == {str(x) for x in SEEDS}
        assert s["envs"][env]["num_envs"] == REF_SPECS[env].num_envs and s["envs"][env]["num_steps_per_env"] == REF_SPECS[env].num_steps


@pytest.mark.parametrize("env", ENV_NAMES)
@pytest.mark.parametrize("seed", SEEDS)
def test_curve_is_complete_finite_and_learned(env, seed):
    rows = load_curve(env, seed)
    assert len(rows) == REF_SPECS[env].iterations and [int(r["iteration"]) for r in rows] == list(range(len(rows)))
    assert REQUIRED_COLUMNS <= set(rows[0])
    for key in ("Loss/value", "Loss/surrogate", "Policy/mean_std", "Loss/learning_rate"):
        assert all(r[key] is not None and math.isfinite(r[key]) for r in rows), key
    rewards = [r["Train/mean_reward"] for r in rows if r["Train/mean_reward"] is not None]
    assert rewards and all(math.isfinite(x) for x in rewards)
    assert final_reward(rows) >= LEARNED[env], f"{env} seed {seed}: final {final_reward(rows):.1f} < {LEARNED[env]}"
    assert rows[-1]["wall_s"] < MAX_WALL_S


@pytest.mark.parametrize("env", ENV_NAMES)
def test_summary_numbers_match_the_csv_files(env):
    s = load_summary()["envs"][env]
    finals = [final_reward(load_curve(env, seed)) for seed in SEEDS]
    assert s["final_reward_mean"] == pytest.approx(statistics.mean(finals), rel=1e-9)
    assert s["final_reward_std"] == pytest.approx(statistics.pstdev(finals), rel=1e-9)
    for seed in SEEDS:
        assert s["seeds"][str(seed)]["wall_s"] < MAX_WALL_S
    assert (GOLDEN_DIR / f"{env}_curves.png").stat().st_size > 1000


@pytest.mark.parametrize("env", ENV_NAMES)
def test_phase4_acceptance_rule(env):
    s = load_summary()["envs"][env]
    thr = phase4_threshold(env)
    assert thr == pytest.approx(s["final_reward_mean"] - max(s["final_reward_std"], 0.05 * abs(s["final_reward_mean"])))
    assert tt_run_passes(env, [s["final_reward_mean"]] * 3)  # the reference itself passes
    assert not tt_run_passes(env, [thr - 1e-3 * max(1.0, abs(thr))] * 3)
    assert not tt_run_passes(env, [float("nan")] * 3) and not tt_run_passes(env, [])
