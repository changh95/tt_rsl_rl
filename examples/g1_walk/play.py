# SPDX-License-Identifier: BSD-3-Clause
"""Example: watch a trained Unitree G1 walk (and measure it). No device needed: the policy is a small MLP, evaluated on the CPU, in the unmodified upstream `OnPolicyRunner`.

    scripts/fetch_g1_model.sh                                            # once, if not done for the training
    # one checkpoint: speed tracking and falls at held commands, and a video of the scripted command sequence
    MUJOCO_GL=egl python examples/g1_walk/play.py --checkpoint .run/examples/g1_walk/model_1499.pt --video g1.mp4
    # several checkpoints side by side (`random` = untrained), e.g. the three of the README, with a GIF too
    MUJOCO_GL=egl python examples/g1_walk/play.py --checkpoint random .run/examples/g1_walk/model_100.pt .run/examples/g1_walk/model_1499.pt --video g1.mp4 --gif g1.gif

The robot follows 0.4 m/s, 0.8 m/s, then 0.5 m/s while turning left and right (5 s each). The label of a panel names the iteration of the checkpoint (`model_100.pt` is the state after
101 iterations); the counter shows how often the robot fell (it then starts again, like in training). Under every panel the REWARD of that robot is plotted live (the return of its current
episode, one scale for all panels, red ticks = falls) and under all panels three plots against the training iteration, with the iteration of every panel marked: the VELOCITY TRACKING ERROR of the
saved checkpoints, the critic's VALUE LOSS and the REWARD TERMS (rewards above zero, penalties below). The loss and the reward terms are read from the `metrics.csv` that `train.py` writes next to the
checkpoints; the tracking error is measured on every `model_<it>.pt` there (about 5 s each, once: `tracking_error.json` next to them). `MUJOCO_GL=egl` renders headless; the video needs imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.demos.g1_render import label_of, load_policy, measure, render_comparison, tracking_curve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="+", required=True, help="`random` and/or model_<it>.pt files (upstream format)")
    ap.add_argument("--video", default=None, help="MP4 of the checkpoints side by side")
    ap.add_argument("--gif", default=None, help="a small GIF of the same")
    ap.add_argument("--metrics", default=None, help="metrics.csv of the training run, for the strips under the video (default: the metrics.csv next to the checkpoints)")
    ap.add_argument("--no-curves", action="store_true", help="a video of the robots only: no return strips, no training plots")
    ap.add_argument("--no-eval", action="store_true", help="skip the measurement")
    ap.add_argument("--eval-envs", type=int, default=64, help="robots of the measurement (each holds one command for 12 s)")
    args = ap.parse_args()

    if not args.no_eval:
        for ck in args.checkpoint:
            policy = load_policy(ck)
            print(label_of(ck))
            for cmd in ((0.4, 0.0, 0.0), (0.8, 0.0, 0.0), (0.5, 0.0, 0.4)):
                m = measure(policy, cmd, steps=600, envs=args.eval_envs)
                print(f"  command vx {cmd[0]:.1f} m/s, yaw rate {cmd[2]:+.1f} rad/s -> vx {m['mean_vx']:.2f} m/s, yaw rate {m['mean_yaw_rate']:+.2f} rad/s, "
                      f"{100 * m['fell_fraction']:.0f} % of the finished episodes were falls")
    if args.video or args.gif:
        metrics = args.metrics
        if metrics is None and not args.no_curves:
            beside = [Path(ck).parent / "metrics.csv" for ck in args.checkpoint if ck != "random"]
            metrics = next((str(m) for m in beside if m.exists()), None)
        tracking = None
        if metrics is not None and not args.no_curves:
            tracking = Path(metrics).parent / "tracking_error.json"
            if not tracking.exists():  # once per run: every saved checkpoint, the commands of the video, 128 robots each
                print(f"measuring the tracking error of the checkpoints in {tracking.parent} ...")
                tracking.write_text(json.dumps(tracking_curve(str(tracking.parent)), indent=1))
        render_comparison(args.checkpoint, video=args.video, gif=args.gif, metrics=None if args.no_curves else metrics, tracking=None if args.no_curves else (str(tracking) if tracking else None))


if __name__ == "__main__":
    main()
