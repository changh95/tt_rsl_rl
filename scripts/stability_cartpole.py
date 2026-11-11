# SPDX-License-Identifier: BSD-3-Clause
"""Phase 6: how often does cartpole training collapse after reaching the maximum reward, on 1 chip vs a mesh? (Seeds beyond the golden ones; no golden needed.)

For every seed: the iteration at which the mean reward first reaches >= 499, the final reward and the minimum of the curve after that. A "collapse" = the
minimum after first reaching 499 is below 450.

    scripts/tt_run.sh --chips 16,17 --timeout 900 --tag stab -- python scripts/stability_cartpole.py --mesh 1x2 --seeds 3 4 5 6 7 8
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.testing.reference import train_reference  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="1x1")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows, cols = (int(v) for v in args.mesh.split("x"))
    tt_cfg = {"trace": True, "mesh_shape": [rows, cols], **({"replica_check_interval": 10} if rows * cols > 1 else {})}
    results = {}
    for seed in args.seeds:
        log_dir = REPO / ".run" / "stability" / args.mesh / f"seed{seed}"
        shutil.rmtree(log_dir, ignore_errors=True)
        train_reference("cartpole", seed, str(log_dir), num_threads=4, tt=True, tt_cfg=tt_cfg)
        curve = [(int(r["iteration"]), float(r["Train/mean_reward"])) for r in csv.DictReader(open(log_dir / "metrics.csv")) if r.get("Train/mean_reward") not in (None, "")]
        first = next((i for i, v in curve if v >= 499.0), None)
        after = [v for i, v in curve if first is not None and i >= first]
        results[seed] = {"first_499": first, "final": curve[-1][1], "min_after": min(after) if after else None, "collapsed": bool(after) and min(after) < 450.0}
        print(f"mesh {args.mesh} seed {seed}: {results[seed]}", flush=True)
    n_collapsed = sum(r["collapsed"] for r in results.values())
    print(f"mesh {args.mesh}: {n_collapsed} of {len(results)} seeds collapsed after reaching 499")
    if args.out:
        Path(args.out).write_text(json.dumps({"mesh": args.mesh, "seeds": results, "collapsed": n_collapsed}, indent=1))


if __name__ == "__main__":
    main()
