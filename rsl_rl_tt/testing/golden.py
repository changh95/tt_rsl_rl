# SPDX-License-Identifier: BSD-3-Clause
"""Access to the committed golden baselines (docs/porting/golden) and the Phase 4 acceptance rule derived from them."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "docs" / "porting" / "golden"
FINAL_WINDOW = 10  # must equal scripts/train_reference.py


def load_summary(golden_dir: Path = GOLDEN_DIR) -> dict:
    return json.loads((golden_dir / "summary.json").read_text())


def load_curve(env_name: str, seed: int, golden_dir: Path = GOLDEN_DIR) -> list[dict[str, float | None]]:
    with open(golden_dir / env_name / f"seed{seed}.csv") as f:
        return [{k: (float(v) if v != "" else None) for k, v in row.items()} for row in csv.DictReader(f)]


def final_reward(rows: list[dict]) -> float:
    vals = [r["Train/mean_reward"] for r in rows[-FINAL_WINDOW:] if r.get("Train/mean_reward") is not None]
    return statistics.mean(vals)


def phase4_threshold(env_name: str, golden_dir: Path = GOLDEN_DIR) -> float:
    """BRIEF Phase 4 initial criterion: TT final mean reward >= reference mean - max(std over reference seeds, 5 % of |mean|)."""
    env = load_summary(golden_dir)["envs"][env_name]
    return env["final_reward_mean"] - max(env["final_reward_std"], 0.05 * abs(env["final_reward_mean"]))


def tt_run_passes(env_name: str, tt_final_rewards: list[float], golden_dir: Path = GOLDEN_DIR) -> bool:
    """Apply the Phase 4 criterion to the final rewards of the TT seeds (their mean is compared); NaN/Inf fail."""
    if not tt_final_rewards or any(not math.isfinite(x) for x in tt_final_rewards):
        return False
    return statistics.mean(tt_final_rewards) >= phase4_threshold(env_name, golden_dir)
