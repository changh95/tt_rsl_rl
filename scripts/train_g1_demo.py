# SPDX-License-Identifier: BSD-3-Clause
"""Demo: teach a Unitree G1 to walk with PPO. The physics (MuJoCo, the G1 model of MuJoCo Menagerie) runs on the CPU, thousands of robots at once; the policy is trained on the
Tenstorrent tray, by default on ALL EIGHT chips (`--mesh 2x4`, data parallel), through the unmodified upstream `OnPolicyRunner` and `rsl_rl_tt.ppo:TTPPO`.

    scripts/fetch_g1_model.sh                                          # once: the G1 model (38 MB, git-ignored)
    # all eight chips (device job: only through the wrapper); num_envs must be a multiple of 8 x 32 = 256
    scripts/tt_run.sh --timeout 14400 --tag g1 -- python scripts/train_g1_demo.py --mesh 2x4 --envs 8192 --iterations 1500 --out docs/porting/demo_g1/tt_8chips
    # one chip, or upstream PPO on the CPU for comparison
    scripts/tt_run.sh --chips 16 --timeout 14400 --tag g1 -- python scripts/train_g1_demo.py --mesh 1x1 --envs 4096 --iterations 300 --out docs/porting/demo_g1/tt_1chip
    python scripts/train_g1_demo.py --backend cpu --envs 4096 --iterations 100 --out docs/porting/demo_g1/cpu

Writes `<out>/metrics.csv` (every scalar of the upstream logger, incl. `Metrics/forward_speed`, `Metrics/command_vx`, `Episode_Termination/fell`, every reward term), `summary.json` (time
breakdown: CPU simulation vs the rest) and the checkpoints `model_<it>.pt` (upstream format) in `--runs-dir`. Then `scripts/render_g1_demo.py` makes the video.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl_tt.bridge.config import tt_agent_cfg  # noqa: E402
from rsl_rl_tt.demos.g1_cfg import g1_train_cfg  # noqa: E402
from rsl_rl_tt.demos.mujoco_g1 import MujocoG1VecEnv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("tt", "cpu"), default="tt")
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=24, help="steps per environment per iteration")
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--mesh", default="2x4", help="TT: mesh shape RxC of the chips of this job (2x4 = all eight; 1x2 with --chips 16,17; 1x1 with --chips 16)")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--init-std", type=float, default=0.8)
    ap.add_argument("--entropy", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-threads", type=int, default=60, help="threads of the multithreaded MuJoCo rollout")
    ap.add_argument("--threads", type=int, default=8, help="torch threads of the host-side work")
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--out", required=True, help="directory of metrics.csv and summary.json")
    ap.add_argument("--runs-dir", default=None, help="checkpoint directory (default .run/g1/<name of --out>, not committed)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = Path(args.runs_dir or REPO / ".run" / "g1" / out.name)
    shutil.rmtree(runs, ignore_errors=True)
    runs.mkdir(parents=True)

    env = MujocoG1VecEnv(args.envs, seed=args.seed, num_threads=args.env_threads)
    cfg = g1_train_cfg(num_steps=args.steps, hidden=tuple(args.hidden), init_std=args.init_std, learning_rate=args.lr, entropy_coef=args.entropy, save_interval=args.save_interval)
    rows, cols = (int(v) for v in args.mesh.split("x"))
    if args.backend == "tt":
        cfg = tt_agent_cfg(cfg, mesh_shape=(rows, cols), trace=not args.no_trace)
    torch.manual_seed(args.seed)
    runner = OnPolicyRunner(env, cfg, log_dir=str(runs), device="cpu")
    print(f"{args.backend}: {args.envs} environments x {args.steps} steps = {args.envs * args.steps} samples per iteration, {args.iterations} iterations", flush=True)

    t0 = time.perf_counter()
    try:
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    finally:
        wall = time.perf_counter() - t0
        if hasattr(runner.alg, "release_traces"):
            runner.alg.release_traces()
    shutil.copyfile(runs / "metrics.csv", out / "metrics.csv")

    rows_ = list(csv.DictReader(open(out / "metrics.csv")))
    last = rows_[-10:]
    mean = lambda col: sum(float(r[col]) for r in last if r.get(col) not in (None, "")) / max(1, sum(1 for r in last if r.get(col) not in (None, "")))
    final_cols = ("Train/mean_reward", "Train/mean_episode_length", "Metrics/forward_speed", "Metrics/command_vx", "Episode_Termination/fell")
    summary = {
        "backend": (f"tt {args.mesh} ({'traced' if not args.no_trace else 'eager'}, {rows * cols} chips)" if args.backend == "tt" else "cpu (upstream PPO)"),
        "num_envs": args.envs, "steps_per_iteration": args.steps, "iterations": args.iterations, "samples": args.envs * args.steps * args.iterations,
        "hidden": args.hidden, "seed": args.seed, "wall_s": round(wall, 1), "seconds_per_iteration": round(wall / args.iterations, 3),
        "cpu_simulation_s": round(env.step_seconds, 1), "algorithm_s": round(wall - env.step_seconds, 1),
        "final": {c: round(mean(c), 3) for c in final_cols if c in rows_[-1]}, "env_threads": env.num_threads,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
