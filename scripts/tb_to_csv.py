# SPDX-License-Identifier: BSD-3-Clause
"""TensorBoard event files -> CSV with the same layout as the CSVs of this repository (`iteration`, then one column per scalar tag, e.g. `Train/mean_reward`).

The rsl_rl logger of an Isaac Lab GPU run writes TensorBoard scalars under the tags the CSV writer uses, so a GPU reference run (`logs/rsl_rl/<experiment>/<run>/`) becomes
comparable to a TT run (`<log-dir>/metrics.csv`) with `scripts/compare_curves.py`. Needs `tensorboard` (installed with Isaac Lab / rsl_rl).

    python scripts/tb_to_csv.py logs/rsl_rl/cartpole/2026-11-12_10-00-00 --out gpu_seed0.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def convert(run_dir: str | Path, out: str | Path) -> int:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})  # 0 = keep every event
    acc.Reload()
    tags = acc.Tags()["scalars"]
    if not tags:
        raise SystemExit(f"no scalar events under {run_dir}")
    table: dict[int, dict[str, float]] = {}
    for tag in tags:
        for event in acc.Scalars(tag):
            table.setdefault(int(event.step), {})[tag] = float(event.value)
    columns = ["iteration", *tags]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for step in sorted(table):
            w.writerow([step] + [table[step].get(t, "") for t in tags])
    return len(table)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = convert(args.run_dir, args.out)
    print(f"wrote {n} iterations to {args.out}")


if __name__ == "__main__":
    main()
