# SPDX-License-Identifier: BSD-3-Clause
"""Host-only: the committed MULTI-CHIP TT training runs (docs/porting/tt_runs_mesh_1x2 and tt_runs_mesh_2x4, produced by `scripts/train_tt.py --trace --mesh RxC`,
Phase 6) are complete, finite and satisfy the same Phase 4 criterion against the golden baselines as the single-chip runs: data parallel training learns as well
as upstream. (The late-collapse statistics of cartpole are in docs/porting/measurements/stability_cartpole_*.json and PERF.md.)"""

import json
import math

import pytest

from rsl_rl_tt.testing.envs import ENV_NAMES
from rsl_rl_tt.testing.golden import GOLDEN_DIR, TT_RUNS_DIR, final_reward, load_tt_curve, tt_vs_golden
from rsl_rl_tt.testing.reference import REF_SPECS

SEEDS = (0, 1, 2)
MESHES = {"1x2": 2, "2x4": 8}  # directory suffix -> chips


def run_dir(mesh: str):
    return TT_RUNS_DIR.parent / f"tt_runs_mesh_{mesh}"


@pytest.mark.parametrize("mesh", MESHES)
def test_summary_names_the_mesh_and_covers_every_env(mesh):
    s = json.loads((run_dir(mesh) / "summary.json").read_text())
    assert f"a {mesh} mesh of {MESHES[mesh]} Blackhole chips" in s["backend"] and "traced" in s["backend"] and "HiFi4" in s["backend"]
    assert set(s["envs"]) == set(ENV_NAMES)
    for env in ENV_NAMES:
        assert set(s["envs"][env]["seeds"]) == {str(x) for x in SEEDS}


@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("env", ENV_NAMES)
@pytest.mark.parametrize("seed", SEEDS)
def test_mesh_run_is_complete_and_finite(mesh, env, seed):
    rows = load_tt_curve(env, seed, run_dir(mesh))
    assert len(rows) == REF_SPECS[env].iterations and [int(r["iteration"]) for r in rows] == list(range(len(rows)))
    for r in rows:
        assert all(v is None or math.isfinite(v) for v in r.values()), f"{mesh} {env} seed {seed} iteration {r['iteration']}: non-finite value"
    assert final_reward(rows) == final_reward(rows)  # not NaN


@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("env", ENV_NAMES)
def test_mesh_matches_the_golden_baseline_by_the_phase4_criterion(mesh, env):
    rep = tt_vs_golden(env, SEEDS, run_dir(mesh), GOLDEN_DIR)
    assert rep["all_finite"]
    assert rep["passes"], f"{mesh} {env}: TT final reward {rep['tt_mean']:.2f} < threshold {rep['threshold']:.2f} (reference {rep['reference_mean']:.2f}); seeds {rep['tt_final_rewards']}"
