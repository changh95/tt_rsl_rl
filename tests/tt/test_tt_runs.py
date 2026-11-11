# SPDX-License-Identifier: BSD-3-Clause
"""Host-only: the committed TT training runs (docs/porting/tt_runs, produced by scripts/train_tt.py on one chip) are complete, finite and satisfy
the BRIEF Phase 4 criterion against the golden baselines: TT final mean reward >= reference mean - max(reference seed std, 5 % of |mean|)."""

import json
import math

import pytest

from rsl_rl_tt.testing.envs import ENV_NAMES
from rsl_rl_tt.testing.golden import TT_RUNS_DIR, final_reward, load_tt_curve, tt_vs_golden
from rsl_rl_tt.testing.reference import REF_SPECS

SEEDS = (0, 1, 2)


def test_summary_exists_and_covers_every_env():
    s = json.loads((TT_RUNS_DIR / "summary.json").read_text())
    assert set(s["envs"]) == set(ENV_NAMES) and "one Blackhole chip" in s["backend"]
    for env in ENV_NAMES:
        assert set(s["envs"][env]["seeds"]) == {str(x) for x in SEEDS}


@pytest.mark.parametrize("env", ENV_NAMES)
@pytest.mark.parametrize("seed", SEEDS)
def test_tt_run_is_complete_and_finite(env, seed):
    rows = load_tt_curve(env, seed)
    assert len(rows) == REF_SPECS[env].iterations and [int(r["iteration"]) for r in rows] == list(range(len(rows)))
    for r in rows:
        assert all(v is None or math.isfinite(v) for v in r.values()), f"{env} seed {seed} iteration {r['iteration']}: non-finite value"
    assert final_reward(rows) == final_reward(rows)  # not NaN


@pytest.mark.parametrize("env", ENV_NAMES)
def test_tt_matches_the_golden_baseline_by_the_phase4_criterion(env):
    rep = tt_vs_golden(env)
    assert rep["all_finite"]
    assert rep["passes"], f"{env}: TT final reward {rep['tt_mean']:.2f} < threshold {rep['threshold']:.2f} (reference {rep['reference_mean']:.2f} +- {rep['reference_std']:.2f}); seeds {rep['tt_final_rewards']}"
