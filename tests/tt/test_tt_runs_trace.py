# SPDX-License-Identifier: BSD-3-Clause
"""Host-only: the committed TRACED TT training runs (docs/porting/tt_runs_trace, produced by `scripts/train_tt.py --trace` on one chip, Phase 5) are complete,
finite and satisfy the same Phase 4 criterion against the golden baselines as the eager runs: the traced path trains as well as upstream."""

import json
import math

import pytest

from rsl_rl_tt.testing.envs import ENV_NAMES
from rsl_rl_tt.testing.golden import GOLDEN_DIR, TT_RUNS_DIR, final_reward, load_tt_curve, tt_vs_golden
from rsl_rl_tt.testing.reference import REF_SPECS

TRACE_DIR = TT_RUNS_DIR.parent / "tt_runs_trace"
SEEDS = (0, 1, 2)


def test_summary_says_traced_hifi4_and_covers_every_env():
    s = json.loads((TRACE_DIR / "summary.json").read_text())
    assert "traced" in s["backend"] and "HiFi4" in s["backend"] and "one Blackhole chip" in s["backend"]
    assert set(s["envs"]) == set(ENV_NAMES)
    for env in ENV_NAMES:
        assert set(s["envs"][env]["seeds"]) == {str(x) for x in SEEDS}


@pytest.mark.parametrize("env", ENV_NAMES)
@pytest.mark.parametrize("seed", SEEDS)
def test_traced_run_is_complete_and_finite(env, seed):
    rows = load_tt_curve(env, seed, TRACE_DIR)
    assert len(rows) == REF_SPECS[env].iterations and [int(r["iteration"]) for r in rows] == list(range(len(rows)))
    for r in rows:
        assert all(v is None or math.isfinite(v) for v in r.values()), f"{env} seed {seed} iteration {r['iteration']}: non-finite value"


@pytest.mark.parametrize("env", ENV_NAMES)
def test_traced_matches_the_golden_baseline_by_the_phase4_criterion(env):
    rep = tt_vs_golden(env, SEEDS, TRACE_DIR, GOLDEN_DIR)
    assert rep["all_finite"]
    assert rep["passes"], f"{env}: traced TT final reward {rep['tt_mean']:.2f} < threshold {rep['threshold']:.2f} (reference {rep['reference_mean']:.2f}); seeds {rep['tt_final_rewards']}"
