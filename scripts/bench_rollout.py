# SPDX-License-Identifier: BSD-3-Clause
"""Full-size rollouts with TT inference (TTPPO) vs the upstream CPU PPO on the Phase 1 test envs: per-step time breakdown and host-op counts.

Device job: run ONLY through the wrapper.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag bench -- python scripts/bench_rollout.py [--envs cartpole ...] [--iters 5] [--out FILE.json]

The rollout is the inner loop of `OnPolicyRunner.learn` (act -> env.step -> process_env_step, then compute_returns); the update is Phase 3.
In Phase 2 the TT numbers are dominated by the temporary host storage downloads (H-7) and untraced per-op dispatch; they are the baseline for Phase 5.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl_tt.testing.envs import ENV_NAMES, make_env  # noqa: E402
from rsl_rl_tt.testing.reference import REF_SPECS, make_train_cfg  # noqa: E402


def build(name: str, tt: bool, tmp: str):
    spec = REF_SPECS[name]
    torch.manual_seed(0)
    env = make_env(name, spec.num_envs, seed=0)
    cfg = make_train_cfg(name, spec, logger=None)
    if tt:
        cfg["algorithm"]["class_name"] = "rsl_rl_tt.ppo:TTPPO"
        cfg["algorithm"]["tt_cfg"] = {"mesh_shape": [1, 1]}
    return OnPolicyRunner(env, cfg, log_dir=None, device="cpu"), env, spec


def timed_rollouts(alg, env, spec, iters: int, warmup: int = 2) -> dict:
    alg.train_mode()
    obs = env.get_observations()
    rows = []
    for it in range(warmup + iters):
        t_act = t_env = t_proc = 0.0
        with torch.inference_mode():
            for _ in range(spec.num_steps):
                t0 = time.perf_counter()
                actions = alg.act(obs)
                t1 = time.perf_counter()
                obs, rewards, dones, extras = env.step(actions)
                t2 = time.perf_counter()
                alg.process_env_step(obs, rewards, dones, extras)
                t3 = time.perf_counter()
                t_act, t_env, t_proc = t_act + (t1 - t0), t_env + (t2 - t1), t_proc + (t3 - t2)
            t4 = time.perf_counter()
            alg.compute_returns(obs)
            t_ret = time.perf_counter() - t4
        alg.storage.clear()  # the update (Phase 3) would do this
        if it >= warmup:
            rows.append((t_act, t_env, t_proc, t_ret))
    med = [statistics.median(c) for c in zip(*rows)]
    n = spec.num_steps
    return {
        "act_ms_per_step": med[0] / n * 1e3,
        "env_ms_per_step": med[1] / n * 1e3,
        "process_ms_per_step": med[2] / n * 1e3,
        "compute_returns_ms": med[3] * 1e3,
        "rollout_ms": sum(med) * 1e3,
        "env_steps_per_s": spec.num_envs * n / sum(med),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=list(ENV_NAMES), choices=ENV_NAMES)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(8)
    results: dict = {}
    for name in args.envs:
        spec = REF_SPECS[name]
        res: dict = {"num_envs": spec.num_envs, "num_steps": spec.num_steps, "hidden": list(spec.hidden)}
        runner, env, spec = build(name, tt=False, tmp="")
        res["cpu_upstream"] = timed_rollouts(runner.alg, env, spec, args.iters)
        runner, env, spec = build(name, tt=True, tmp="")
        res["tt_1chip"] = timed_rollouts(runner.alg, env, spec, args.iters)
        res["tt_host_ops_total"] = runner.alg.ctx.host_ops.snapshot()
        results[name] = res
        c, t = res["cpu_upstream"], res["tt_1chip"]
        print(
            f"{name:11s} envs={spec.num_envs:5d} steps={spec.num_steps:3d} | CPU act {c['act_ms_per_step']:7.2f} ms/step, rollout {c['rollout_ms']:8.1f} ms | "
            f"TT act {t['act_ms_per_step']:7.2f} ms/step, rollout {t['rollout_ms']:8.1f} ms ({t['env_steps_per_s']:.0f} env-steps/s)",
            flush=True,
        )
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
