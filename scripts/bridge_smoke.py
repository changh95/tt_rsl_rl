# SPDX-License-Identifier: BSD-3-Clause
"""Connection check of the environment bridge from the TT host (no training, no device): shapes, dtypes, finiteness, conventions and the round trip of a few steps.

Run it against the real Isaac Lab server BEFORE the first training run (docs/porting/ISAAC_RUNBOOK.md):

    python scripts/bridge_smoke.py --endpoint tcp://GPU_HOST:5555 --steps 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.bridge.client import RemoteVecEnv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    env = RemoteVecEnv(args.endpoint, timeout_s=args.timeout)
    n, a = env.num_envs, env.num_actions
    print(f"environments {n}, actions {a}, max_episode_length {env.max_episode_length if isinstance(env.max_episode_length, int) else 'per-env tensor'}")
    print(f"observation groups (per environment): {env.obs_shapes}")
    cfg = env.agent_cfg
    print("agent configuration received:", "yes, keys " + str(sorted(cfg)) if cfg else "NO (pass --agent-cfg to train_isaac.py)")
    for key in ("actor", "critic", "algorithm", "obs_groups"):
        if cfg and key not in cfg:
            print(f"WARNING: the agent configuration has no {key!r}: the TT backend needs the rsl_rl v5 format (actor / critic / algorithm / obs_groups)")
    obs = env.get_observations()
    assert all(v.dtype == torch.float32 and v.shape[0] == n and torch.isfinite(v).all() for v in obs.values()), "observations must be finite float32 [num_envs, ...]"
    saw_timeouts, saw_log, dones_total = False, False, 0
    for _ in range(args.steps):
        obs, rewards, dones, extras = env.step(torch.randn(n, a).clamp(-1, 1))
        assert rewards.shape == (n,) and torch.isfinite(rewards).all(), "rewards must be finite [num_envs]"
        assert dones.shape == (n,) and dones.dtype == torch.long, "dones must be long [num_envs]"
        assert all(torch.isfinite(v).all() for v in obs.values()), "non-finite observation"
        saw_timeouts |= "time_outs" in extras
        saw_log |= bool(extras.get("log"))
        dones_total += int(dones.sum())
    s = env.stats
    rtt = sorted(s.step_rtt_s)
    print(f"{args.steps} steps: median round trip {1e3 * rtt[len(rtt) // 2]:.2f} ms, p95 {1e3 * rtt[int(0.95 * (len(rtt) - 1))]:.2f} ms, "
          f"{(s.sent_bytes + s.received_bytes) / args.steps / 1e6:.2f} MB per step; episodes ended: {dones_total}; time_outs sent: {saw_timeouts}; episode log sent: {saw_log}")
    print("OK")
    if "--keep" not in sys.argv:
        pass
    env._sock.close(linger=0)  # leave the server running (a `close` request would stop it)


if __name__ == "__main__":
    main()
