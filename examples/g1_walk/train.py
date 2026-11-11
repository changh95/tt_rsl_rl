# SPDX-License-Identifier: BSD-3-Clause
"""Example: train a Unitree G1 to walk on command. MuJoCo simulates the robots on the CPU, PPO runs on the Tenstorrent tray through the unmodified upstream `OnPolicyRunner`.

    scripts/fetch_g1_model.sh                      # once: the G1 model of MuJoCo Menagerie (38 MB, git-ignored)

    # all eight chips of the tray, 28 minutes (device job: only through the wrapper); checkpoints land in .run/examples/g1_walk
    scripts/tt_run.sh --timeout 14400 --tag g1 -- python examples/g1_walk/train.py --mesh 2x4 --envs 4096 --iterations 1500
    # one chip (--mesh must match --chips)
    scripts/tt_run.sh --chips 16 --timeout 14400 --tag g1 -- python examples/g1_walk/train.py --mesh 1x1 --envs 4096 --iterations 1500
    # upstream PPO on the CPU (no device), for comparison
    python examples/g1_walk/train.py --backend cpu --envs 4096 --iterations 100

The whole point of the example is how little there is to it: the environment is a rsl_rl `VecEnv` (`MujocoG1VecEnv`), the configuration is an ordinary rsl_rl 5.x training config, and
`tt_agent_cfg` only switches the algorithm class to `rsl_rl_tt.ppo:TTPPO` and adds the device options (mesh shape, trace). Everything else is `OnPolicyRunner(env, cfg, log_dir).learn(...)`.
Checkpoints (`model_<iteration>.pt`) are written in the upstream format; `play.py` replays them. `scripts/train_g1_demo.py` is the same thing with the bookkeeping of the documented run
(`docs/porting/demo_g1/`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl_tt.bridge.config import tt_agent_cfg  # noqa: E402
from rsl_rl_tt.demos.g1_cfg import g1_train_cfg  # noqa: E402
from rsl_rl_tt.demos.mujoco_g1 import MujocoG1VecEnv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("tt", "cpu"), default="tt", help="tt: TTPPO on the tray; cpu: upstream PPO")
    ap.add_argument("--mesh", default="2x4", help="TT: mesh shape RxC of the chips of this job (2x4 = all eight; 1x2 with --chips 16,17; 1x1 with --chips 16)")
    ap.add_argument("--envs", type=int, default=4096, help="robots simulated in parallel (TT: a multiple of 32 x number of chips)")
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=24, help="control steps (20 ms) per robot per iteration")
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--no-trace", action="store_true", help="TT: eager instead of traced (slower)")
    ap.add_argument("--env-threads", type=int, default=60, help="threads of the multithreaded MuJoCo rollout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", default=str(REPO / ".run" / "examples" / "g1_walk"), help="checkpoints and metrics.csv (not committed)")
    args = ap.parse_args()

    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = MujocoG1VecEnv(args.envs, seed=args.seed, num_threads=args.env_threads)          # a rsl_rl VecEnv: 47 observations + 8 privileged for the critic, 12 leg joints
    cfg = g1_train_cfg(num_steps=args.steps, save_interval=args.save_interval)           # the rsl_rl 5.x config: actor / critic MLP [512, 256, 128], PPO, empirical normalisation
    if args.backend == "tt":
        rows, cols = (int(v) for v in args.mesh.split("x"))
        cfg = tt_agent_cfg(cfg, mesh_shape=(rows, cols), trace=not args.no_trace)         # algorithm class -> rsl_rl_tt.ppo:TTPPO; the rest of the config is unchanged
    runner = OnPolicyRunner(env, cfg, log_dir=str(log_dir), device="cpu")

    samples = args.envs * args.steps * args.iterations
    print(f"{args.backend}: {args.envs} robots x {args.steps} steps x {args.iterations} iterations = {samples / 1e6:.1f} M samples; checkpoints in {log_dir}", flush=True)
    t0 = time.perf_counter()
    try:
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    finally:
        if hasattr(runner.alg, "release_traces"):
            runner.alg.release_traces()  # TT: free the captured traces before the mesh closes
        env.close()
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min; replay: python examples/g1_walk/play.py --checkpoint {log_dir}/model_{args.iterations - 1}.pt", flush=True)


if __name__ == "__main__":
    main()
