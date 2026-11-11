# SPDX-License-Identifier: BSD-3-Clause
"""Learning curves of the humanoid demo runs: mean reward, mean episode length and forward speed against the iteration and the reward against the wall-clock time.

    python scripts/plot_humanoid_curves.py --run "TT, 1 chip=docs/porting/demo_humanoid/tt_1chip" --run "upstream PPO, CPU=docs/porting/demo_humanoid/cpu_150it" --out docs/porting/demo_humanoid/curves.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(run_dir: Path) -> tuple[list[dict], dict]:
    rows = [{k: (float(v) if v != "" else None) for k, v in r.items()} for r in csv.DictReader(open(run_dir / "metrics.csv"))]
    return rows, json.loads((run_dir / "summary.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='"label=run directory" (metrics.csv and summary.json inside); repeat for several runs')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for label, path in (r.split("=", 1) for r in args.run):
        rows, summary = load(Path(path))
        its = [r["iteration"] for r in rows]
        sec = summary["seconds_per_iteration"]
        for ax, key, title in zip(axes[:3], ("Train/mean_reward", "Train/mean_episode_length", "Metrics/forward_speed"), ("mean episode return (reward scaled by 0.1)", "mean episode length (steps, max 1000)", "forward speed (m/s)")):
            pts = [(r["iteration"], r[key]) for r in rows if r.get(key) is not None]
            ax.plot(*zip(*pts), label=f"{label} ({sec:.2f} s/iteration)")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("iteration")
        pts = [(r["iteration"] * sec / 60, r["Train/mean_reward"]) for r in rows if r.get("Train/mean_reward") is not None]
        axes[3].plot(*zip(*pts), label=label)
        axes[3].set_title("mean reward against wall-clock time", fontsize=9)
        axes[3].set_xlabel("minutes")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
