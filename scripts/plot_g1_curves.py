# SPDX-License-Identifier: BSD-3-Clause
"""Learning curves of the Unitree G1 demo runs: episode return, episode length, forward speed against the commanded speed, and the gait-shaping reward terms.

    python scripts/plot_g1_curves.py --run "TT, 8 chips=docs/porting/demo_g1/tt_8chips" --out docs/porting/demo_g1/curves.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='"label=run directory" (metrics.csv and summary.json inside)')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for label, path in (r.split("=", 1) for r in args.run):
        rows = [{k: (float(v) if v != "" else None) for k, v in r.items()} for r in csv.DictReader(open(Path(path) / "metrics.csv"))]
        summary = json.loads((Path(path) / "summary.json").read_text())
        series = lambda key: [(r["iteration"], r[key]) for r in rows if r.get(key) is not None]
        axes[0].plot(*zip(*series("Train/mean_reward")), label=f"{label} ({summary['seconds_per_iteration']:.2f} s/iteration)")
        axes[1].plot(*zip(*series("Train/mean_episode_length")))
        axes[2].plot(*zip(*series("Metrics/forward_speed")), label="achieved forward speed")
        axes[2].plot(*zip(*series("Metrics/command_vx")), linestyle="--", color="0.5", label="commanded")
        axes[3].plot(*zip(*series("Episode_Reward/feet_phase")), label="foot height follows the gait phase")
        axes[3].plot(*zip(*series("Episode_Reward/track_lin_vel")), label="velocity tracking")
    for ax, title in zip(axes, ("mean episode return (reward x dt)", "mean episode length (steps, max 1000)", "forward speed (m/s, finished episodes)", "per-episode reward terms")):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    axes[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
