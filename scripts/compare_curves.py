# SPDX-License-Identifier: BSD-3-Clause
"""Compare the learning curves of a reference run (e.g. Isaac Lab on a GPU) with those of the TT run: the Phase 4 criterion plus a figure.

    python scripts/compare_curves.py --ref gpu_seed0.csv gpu_seed1.csv gpu_seed2.csv --tt tt_seed0/metrics.csv tt_seed1/metrics.csv --out docs/porting/isaac_cartpole

Criterion (BRIEF Phase 4, the same one the CPU golden baselines are held to): mean over the TT seeds of the final reward (mean of `Train/mean_reward` over the last
`FINAL_WINDOW` iterations) >= mean over the reference seeds - max(std over the reference seeds, 5 % of |mean|); every value finite. With ONE reference seed the std is 0 and
the 5 % margin applies. Writes `<out>.json` and `<out>.png`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.testing.golden import final_reward  # noqa: E402


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [{k: (float(v) if v != "" else None) for k, v in row.items()} for row in csv.DictReader(f)]


def compare(ref_paths: list[str], tt_paths: list[str], metric: str = "Train/mean_reward") -> dict:
    ref_rows, tt_rows = [load(p) for p in ref_paths], [load(p) for p in tt_paths]
    finite = all(v is None or math.isfinite(v) for rows in tt_rows for r in rows for v in r.values())
    ref_finals, tt_finals = [final_reward(r) for r in ref_rows], [final_reward(r) for r in tt_rows]
    ref_mean = statistics.mean(ref_finals)
    ref_std = statistics.pstdev(ref_finals) if len(ref_finals) > 1 else 0.0
    threshold = ref_mean - max(ref_std, 0.05 * abs(ref_mean))
    tt_mean = statistics.mean(tt_finals)
    return {
        "metric": metric, "reference_finals": ref_finals, "reference_mean": ref_mean, "reference_std": ref_std, "tt_finals": tt_finals, "tt_mean": tt_mean,
        "threshold": threshold, "all_finite": bool(finite), "passes": bool(finite and tt_mean >= threshold), "reference_iterations": [len(r) for r in ref_rows], "tt_iterations": [len(r) for r in tt_rows],
    }


def plot(ref_paths: list[str], tt_paths: list[str], out_png: Path, metric: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, p in enumerate(ref_paths):
        rows = [r for r in load(p) if r.get(metric) is not None]
        ax.plot([r["iteration"] for r in rows], [r[metric] for r in rows], color="0.6", linewidth=1.0, label="reference" if i == 0 else None)
    for i, p in enumerate(tt_paths):
        rows = [r for r in load(p) if r.get(metric) is not None]
        ax.plot([r["iteration"] for r in rows], [r[metric] for r in rows], linewidth=1.2, label=f"TT, run {i}")
    ax.set_xlabel("iteration")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", nargs="+", required=True, help="CSV(s) of the reference run(s)")
    ap.add_argument("--tt", nargs="+", required=True, help="CSV(s) of the TT run(s) (<log-dir>/metrics.csv)")
    ap.add_argument("--metric", default="Train/mean_reward")
    ap.add_argument("--title", default="TT vs reference")
    ap.add_argument("--out", required=True, help="output prefix: <out>.json and <out>.png")
    args = ap.parse_args()
    report = compare(args.ref, args.tt, args.metric)
    Path(f"{args.out}.json").write_text(json.dumps(report, indent=2))
    plot(args.ref, args.tt, Path(f"{args.out}.png"), args.metric, args.title)
    print(json.dumps({k: report[k] for k in ("reference_mean", "reference_std", "tt_mean", "threshold", "all_finite", "passes")}, indent=1))
    sys.exit(0 if report["passes"] else 1)


if __name__ == "__main__":
    main()
