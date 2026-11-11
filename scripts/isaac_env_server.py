# SPDX-License-Identifier: BSD-3-Clause
"""Serve a vectorized environment to the TT host over ZeroMQ (Phase 7, ADR-0003): run this on the ENVIRONMENT host.

Two modes:

  --local-env cartpole|pendulum|tracking12   the CPU test environments of this repository (loopback validation, no Isaac Lab needed). VERIFIED in this repository
                                             (tests/tt/test_bridge.py, docs/porting/PERF.md section "Environment bridge").
  --task Isaac-Cartpole-v0 ...               an Isaac Lab task on a GPU host. WRITTEN AGAINST Isaac Lab v2.3.0 `scripts/reinforcement_learning/rsl_rl/train.py` and
                                             `isaaclab_rl.rsl_rl.RslRlVecEnvWrapper` (public sources, BSD-3-Clause) but NOT RUN: this host has no Isaac Lab / GPU.
                                             Validate it first with the runbook (docs/porting/ISAAC_RUNBOOK.md); if your Isaac Lab differs, keep the three lines that
                                             matter (`env = RslRlVecEnvWrapper(...)`, `serve(env, endpoint, agent_cfg=agent_cfg.to_dict())`) in your own launcher.

    # environment host (Isaac Lab python):
    python scripts/isaac_env_server.py --task Isaac-Cartpole-v0 --num_envs 4096 --headless --endpoint tcp://*:5555
    # loopback test on the TT host:
    python scripts/isaac_env_server.py --local-env tracking12 --num_envs 2048 --endpoint tcp://127.0.0.1:5555

The server sends `agent_cfg.to_dict()` to the client, so the TT run uses exactly the configuration this environment was set up for. It needs only numpy, pyzmq and torch
(`pip install pyzmq` in the Isaac Lab python); it does not import rsl_rl or ttnn.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def run_local(args) -> None:
    from rsl_rl_tt.bridge.server import serve
    from rsl_rl_tt.testing.envs import make_env
    from rsl_rl_tt.testing.reference import REF_SPECS, RefSpec, make_train_cfg

    base = REF_SPECS[args.local_env]
    spec = RefSpec(num_envs=args.num_envs or base.num_envs, num_steps=base.num_steps, iterations=base.iterations, hidden=base.hidden, algorithm=base.algorithm)
    env = make_env(args.local_env, spec.num_envs, seed=args.seed or 0)
    cfg = make_train_cfg(args.local_env, spec, logger=None)
    if args.dump_agent_cfg:
        Path(args.dump_agent_cfg).write_text(json.dumps(cfg, indent=1, default=repr))
    serve(env, args.endpoint, agent_cfg=cfg)


def run_isaac(args, hydra_args) -> None:
    """Isaac Lab mode: the structure of Isaac Lab v2.3.0 `train.py`, with `serve(...)` where the runner would be created."""
    # the simulator must be launched before any other Isaac Lab import (same order as train.py)
    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]] + hydra_args  # what is left is for Hydra
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.io import dump_yaml

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config

    from rsl_rl_tt.bridge.server import serve

    @hydra_task_config(args.task, args.agent)
    def main(env_cfg, agent_cfg):
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        if args.seed is not None:
            agent_cfg.seed = args.seed
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args.device if getattr(args, "device", None) is not None else env_cfg.sim.device
        env = gym.make(args.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # resets the environment, as the rsl_rl runner never does
        if args.dump_agent_cfg:
            dump_yaml(args.dump_agent_cfg, agent_cfg)
        try:
            serve(env, args.endpoint, agent_cfg=agent_cfg.to_dict())
        finally:
            env.close()

    main()
    simulation_app.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task", help="Isaac Lab task id, e.g. Isaac-Cartpole-v0 (needs the Isaac Lab python)")
    mode.add_argument("--local-env", choices=("cartpole", "pendulum", "tracking12"), help="CPU test environment of this repository (loopback validation)")
    ap.add_argument("--endpoint", default="tcp://*:5555", help="ZeroMQ endpoint to bind: tcp://*:5555 (all interfaces), tcp://127.0.0.1:5555, ipc:///tmp/env.sock")
    ap.add_argument("--num_envs", type=int, default=None, help="number of environments (default: the task's / the reference spec's)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="Isaac Lab: name of the agent configuration entry point")
    ap.add_argument("--dump-agent-cfg", default=None, help="also write the agent configuration to this file (Isaac Lab: YAML like params/agent.yaml; local: JSON)")
    if "--task" in sys.argv:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(ap)
        args, hydra_args = ap.parse_known_args()
        run_isaac(args, hydra_args)
    else:
        run_local(ap.parse_args())


if __name__ == "__main__":
    main()
