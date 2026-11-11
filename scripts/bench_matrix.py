# SPDX-License-Identifier: BSD-3-Clause
"""Phase 5 benchmark: algorithm cost per training iteration (rollout + update) for a given num_envs x hidden-size configuration, upstream CPU vs TTPPO.

The environment is a NULL env (pre-generated random observations, constant dones) so that only the algorithm is measured; the cost of the real
CPU environments is reported by `--env-cost` (tracking12 step time at that size). Shapes follow the tracking12 task (actor obs 36, critic obs 36 + 48
privileged, 12 actions); 24 steps per env, 4 mini-batches, 5 epochs (upstream defaults).

Device job (backend tt): run ONLY through the wrapper:

    scripts/tt_run.sh --chips 16 --timeout 3000 --tag bench -- python scripts/bench_matrix.py --backend tt --envs 4096 --hidden 512 256 128 [--profile] [--trace ...]
    python scripts/bench_matrix.py --backend cpu --envs 4096 --hidden 512 256 128          # host only
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from tensordict import TensorDict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl.env import VecEnv  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl_tt.testing.reference import CSV_LOGGER, make_train_cfg, RefSpec, REF_SPECS  # noqa: E402


class NullEnv(VecEnv):
    num_actions = 12

    def __init__(self, num_envs: int, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.num_envs, self.device, self.cfg = num_envs, "cpu", {}
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self._obs = TensorDict({"policy": torch.randn(num_envs, 36, generator=g), "privileged": torch.randn(num_envs, 48, generator=g)}, batch_size=[num_envs])
        self._rewards = torch.randn(num_envs, generator=g) * 0.1
        self._dones = torch.zeros(num_envs, dtype=torch.long)
        self._extras = {"time_outs": torch.zeros(num_envs, dtype=torch.bool)}

    def get_observations(self):
        return self._obs

    def step(self, actions):
        return self._obs, self._rewards, self._dones, dict(self._extras)


def build(backend: str, num_envs: int, hidden: list[int], steps: int = 24, trace: bool = False):
    spec = RefSpec(num_envs=num_envs, num_steps=steps, iterations=1, hidden=tuple(hidden), algorithm=REF_SPECS["tracking12"].algorithm)
    torch.manual_seed(0)
    env = NullEnv(num_envs)
    cfg = make_train_cfg("tracking12", spec, logger=None)
    cfg["check_for_nan"] = False  # the real runner checks env tensors on the host; not part of the algorithm cost
    if backend == "tt":
        cfg["algorithm"]["class_name"] = "rsl_rl_tt.ppo:TTPPO"
        cfg["algorithm"]["tt_cfg"] = {"mesh_shape": [1, 1], **({"trace": True} if trace else {})}
    return OnPolicyRunner(env, cfg, log_dir=None, device="cpu"), env, spec


def run_iterations(alg, env, spec, iters: int, warmup: int) -> dict:
    alg.train_mode()
    obs = env.get_observations()
    rows = []
    timer = getattr(alg, "timer", None)
    for it in range(warmup + iters):
        if timer is not None and it == warmup:
            timer.reset()
        t0 = time.perf_counter()
        t_act = t_env = t_proc = 0.0
        with torch.inference_mode():
            for _ in range(spec.num_steps):
                a0 = time.perf_counter()
                actions = alg.act(obs)
                a1 = time.perf_counter()
                obs, rewards, dones, extras = env.step(actions)
                a2 = time.perf_counter()
                alg.process_env_step(obs, rewards, dones, extras)
                a3 = time.perf_counter()
                t_act, t_env, t_proc = t_act + (a1 - a0), t_env + (a2 - a1), t_proc + (a3 - a2)
            r0 = time.perf_counter()
            alg.compute_returns(obs)
            t_ret = time.perf_counter() - r0
        u0 = time.perf_counter()
        alg.update()
        t_upd = time.perf_counter() - u0
        rows.append({"act": t_act, "env": t_env, "process": t_proc, "returns": t_ret, "update": t_upd, "total": time.perf_counter() - t0})
        print(f"  iteration {it}{' (warm-up)' if it < warmup else ''}: " + ", ".join(f"{k} {v * 1e3:.0f} ms" for k, v in rows[-1].items()), flush=True)
    med = {k: statistics.median(r[k] for r in rows[warmup:]) * 1e3 for k in rows[0]}
    out = {k + "_ms": v for k, v in med.items()}
    out["env_steps_per_s"] = spec.num_envs * spec.num_steps / (med["total"] / 1e3)
    if timer is not None and timer.enabled:
        out["sections"] = timer.report()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("cpu", "tt"), required=True)
    ap.add_argument("--envs", type=int, required=True)
    ap.add_argument("--hidden", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--profile", action="store_true", help="TT only: time the sections of the TT path (adds device syncs; use for the breakdown, not for headline numbers)")
    ap.add_argument("--trace", action="store_true", help="TT only: use the traced path (Phase 5)")
    ap.add_argument("--threads", type=int, default=16, help="torch threads (CPU backend and host-side work)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    runner, env, spec = build(args.backend, args.envs, args.hidden, args.steps, args.trace)
    if args.backend == "tt" and args.profile:
        runner.alg.timer.enabled = True
    print(f"backend={args.backend} envs={args.envs} hidden={args.hidden} steps={args.steps} trace={args.trace} profile={args.profile}", flush=True)
    res = run_iterations(runner.alg, env, spec, args.iters, args.warmup)
    res.update({"backend": args.backend, "num_envs": args.envs, "hidden": args.hidden, "num_steps": args.steps, "trace": args.trace, "profile": args.profile})
    if args.backend == "tt":
        res["host_ops"] = runner.alg.ctx.host_ops.snapshot()
    print(json.dumps({k: v for k, v in res.items() if k != "sections"}, indent=1))
    for name, v in (res.get("sections") or {}).items():
        print(f"  {name:32s} {v['ms_per_call']:9.2f} ms/call x {v['calls']:4d} = {v['total_ms']:9.1f} ms")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
