# SPDX-License-Identifier: BSD-3-Clause
"""Train on the Tenstorrent tray with the environment on ANOTHER host (Isaac Lab on a GPU machine) or in another process (Phase 7, ADR-0003).

The unmodified upstream `OnPolicyRunner` runs here with `rsl_rl_tt.ppo:TTPPO` and a `RemoteVecEnv` connected to the environment server
(`scripts/isaac_env_server.py`). The agent configuration comes from the server (`--agent-cfg` overrides it with a saved JSON / YAML, e.g. the `params/agent.yaml`
of a GPU reference run); only the algorithm class, its `tt_cfg`, the host-side device and the log writer are changed (`rsl_rl_tt/bridge/config.py`).

Device job: run ONLY through the wrapper, e.g. one chip, or all eight with `--mesh 2x4`:

    scripts/tt_run.sh --chips 16 --timeout 7200 --tag isaac -- python scripts/train_isaac.py --endpoint tcp://GPU_HOST:5555 --iterations 300
    scripts/tt_run.sh --timeout 7200 --tag isaac8 -- python scripts/train_isaac.py --endpoint tcp://GPU_HOST:5555 --mesh 2x4 --log-dir logs/isaac_cartpole_8chips

Writes `<log-dir>/metrics.csv` (every scalar the upstream logger emits, columns as in docs/porting/golden), `config.json` and the usual `model_<it>.pt` checkpoints, which
are upstream-format checkpoints (Isaac Lab's `play.py --checkpoint` loads them).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl_tt.bridge.client import RemoteVecEnv  # noqa: E402
from rsl_rl_tt.bridge.config import load_agent_cfg, tt_agent_cfg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True, help="ZeroMQ endpoint of the environment server, e.g. tcp://gpu-host:5555")
    ap.add_argument("--agent-cfg", default=None, help="saved agent configuration (JSON or YAML) instead of the one the server sends")
    ap.add_argument("--iterations", type=int, default=None, help="learning iterations (default: max_iterations of the agent configuration)")
    ap.add_argument("--mesh", default="1x1", help="mesh shape RxC of the chips of this job: 1x1, 1x2 (--chips 16,17), 2x4 (all eight)")
    ap.add_argument("--no-trace", action="store_true", help="eager path instead of the traced one (debugging)")
    ap.add_argument("--log-dir", default=None, help="run directory (default logs/isaac/<timestamp>)")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for the environment server's reply before reporting a hang")
    ap.add_argument("--threads", type=int, default=4, help="torch threads for the host-side work")
    ap.add_argument("--seed", type=int, default=None, help="override the seed of the agent configuration")
    ap.add_argument("--keep-server", action="store_true", help="do not ask the server to stop when training ends")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    env = RemoteVecEnv(args.endpoint, timeout_s=args.timeout)
    print(f"connected to {args.endpoint}: {env.num_envs} environments, {env.num_actions} actions, observation groups {env.obs_shapes}", flush=True)
    base = load_agent_cfg(args.agent_cfg) if args.agent_cfg else env.agent_cfg
    if base is None:
        raise SystemExit("the server did not send an agent configuration and --agent-cfg was not given")
    rows, cols = (int(v) for v in args.mesh.split("x"))
    cfg = tt_agent_cfg(base, mesh_shape=(rows, cols), trace=not args.no_trace)
    if args.seed is not None:
        cfg["seed"] = args.seed
    torch.manual_seed(int(cfg.get("seed", 0)))
    log_dir = Path(args.log_dir or REPO / "logs" / "isaac" / time.strftime("%Y-%m-%d_%H-%M-%S"))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "agent_cfg_used.json").write_text(json.dumps(cfg, indent=1, default=repr))

    runner = OnPolicyRunner(env, cfg, log_dir=str(log_dir), device="cpu")
    iterations = args.iterations if args.iterations is not None else int(cfg.get("max_iterations", 100))
    t0 = time.time()
    try:
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=bool(cfg.get("init_at_random_ep_len", True)))
    finally:
        wall = time.time() - t0
        s = env.stats
        steps = s.calls.get("step", 0)
        print(f"{iterations} iterations in {wall:.0f} s; bridge: {steps} steps, mean round trip {1e3 * sum(s.step_rtt_s) / max(steps, 1):.2f} ms, "
              f"{s.sent_bytes / 1e6:.0f} MB sent, {s.received_bytes / 1e6:.0f} MB received", flush=True)
        if hasattr(runner.alg, "release_traces"):
            runner.alg.release_traces()
        if not args.keep_server:
            env.close()


if __name__ == "__main__":
    main()
