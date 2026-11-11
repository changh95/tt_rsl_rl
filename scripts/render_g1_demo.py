# SPDX-License-Identifier: BSD-3-Clause
"""Render what trained Unitree G1 policies do: a video / GIF (several checkpoints side by side, following a scripted sequence of velocity commands) and an evaluation of each
(speed tracking and falls at held commands). Thin command line over `rsl_rl_tt/demos/g1_render.py`; see `examples/g1_walk/play.py` for the short version.

    MUJOCO_GL=egl python scripts/render_g1_demo.py --checkpoints random .run/g1/tt_8chips/model_100.pt .run/g1/tt_8chips/model_1499.pt \
        --metrics docs/porting/demo_g1/tt_8chips/metrics.csv --tracking docs/porting/demo_g1/tracking_error.json --video docs/porting/demo_g1/g1.mp4 --gif docs/porting/demo_g1/g1.gif --eval-out docs/porting/demo_g1/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.demos.g1_render import label_of, load_policy, measure, render_comparison  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="`random` and/or model_<it>.pt files")
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", type=int, nargs=2, default=[352, 480], metavar=("H", "W"), help="per panel; multiples of 16 for the video")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-envs", type=int, default=256)
    ap.add_argument("--eval-steps", type=int, default=600)
    ap.add_argument("--commands", type=float, nargs="+", default=[0.4, 0.8], help="forward speeds (m/s) of the evaluation; one more run turns (vx 0.5, yaw rate 0.4)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--video", default=None)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--metrics", default=None, help="metrics.csv of the training run: adds the live return under every panel and, under all panels, the value loss and the reward terms against the iteration (needs --tracking)")
    ap.add_argument("--tracking", default=None, help="velocity tracking error of the checkpoints of the run (json of scripts/eval_g1_tracking.py), the first plot of the strip under the panels")
    ap.add_argument("--eval-out", default=None, help="evaluation JSON")
    args = ap.parse_args()
    hidden = tuple(args.hidden)

    if not args.no_eval:
        report = []
        for ck in args.checkpoints:
            policy = load_policy(ck, hidden)
            runs = [(v, 0.0, 0.0) for v in args.commands] + [(0.5, 0.0, 0.4)]
            rep = {"label": label_of(ck), "checkpoint": ck, "by_command": [measure(policy, c, args.eval_steps, args.eval_envs, args.seed) for c in runs]}
            report.append(rep)
            print(json.dumps(rep), flush=True)
        if args.eval_out:
            Path(args.eval_out).write_text(json.dumps(report, indent=2))
    if args.video or args.gif:
        if (args.metrics is None) != (args.tracking is None):
            ap.error("--metrics and --tracking go together (make the json with scripts/eval_g1_tracking.py)")
        render_comparison(args.checkpoints, hidden=hidden, fps=args.fps, size=tuple(args.size), seed=args.seed, video=args.video, gif=args.gif, metrics=args.metrics, tracking=args.tracking)


if __name__ == "__main__":
    main()
