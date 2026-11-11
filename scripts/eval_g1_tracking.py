# SPDX-License-Identifier: BSD-3-Clause
"""Velocity tracking error of the saved checkpoints of the Unitree G1 demo (host only, no device): 128 robots follow the scripted commands of the video for 20 s with the deterministic policy.

    MUJOCO_GL=egl python scripts/eval_g1_tracking.py --dir .run/g1/tt_8chips --out docs/porting/demo_g1/tracking_error.json

The checkpoints are `random` (the untrained network, iteration 1) and every `model_<it>.pt` of `--dir` except `model_0.pt` (which has one update and would share the iteration 1 of the untrained network).
`scripts/render_g1_demo.py --tracking <this json>` draws the curve under the video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.demos.g1_render import tracking_curve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory of the model_<it>.pt checkpoints")
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--robots", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    result = tracking_curve(args.dir, tuple(args.hidden), args.robots, args.seed, log=lambda m: print(m, flush=True))
    Path(args.out).write_text(json.dumps(result, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
