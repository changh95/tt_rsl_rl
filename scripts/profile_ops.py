# SPDX-License-Identifier: BSD-3-Clause
"""Phase 5 op profile of ONE eager mini-batch step of TTPPO.update (BRIEF Phase 5: "op profile").

Every ttnn op called inside `_minibatch_step` is wrapped with a device synchronisation before and after, so each time is that op's wall time
(device time plus its host dispatch latency, ~0.1 ms, negligible at 65536 envs where a step takes ~250 ms). Totals are aggregated per op name.
The hardware profiler is NOT used (its environment variables are owned by scripts/tray_env.sh).

    scripts/tt_run.sh --chips 16 --timeout 900 --tag opprof -- python scripts/profile_ops.py --envs 65536 --hidden 512 256 128 --out docs/porting/measurements/opprofile_65536_512-256-128.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.bench_matrix import build  # noqa: E402

OPS = (
    "matmul", "linear", "add", "subtract", "multiply", "divide", "exp", "log", "elu", "elu_bw", "relu", "tanh", "sqrt", "square", "reciprocal", "neg",
    "sum", "mean", "maximum", "minimum", "where", "gt", "lt", "ge", "le", "eq", "clamp", "copy", "concat", "typecast", "to_layout", "embedding",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=65536)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--mini-batches", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(16)
    runner, env, spec = build("tt", args.envs, args.hidden, 24, trace=False, mini_batches=args.mini_batches)
    alg = runner.alg
    ctx = alg.ctx
    ttnn, mesh = ctx.ttnn, ctx.mesh
    sync = lambda: ttnn.synchronize_device(mesh)

    # one rollout (eager), then the update with the first mini-batch step profiled
    alg.train_mode()
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(spec.num_steps):
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
        alg.compute_returns(obs)

    stats: dict = defaultdict(lambda: [0, 0.0])
    state = {"active": False, "done": False, "step_s": 0.0}

    def wrap(name, fn):
        def wrapped(*a, **k):
            if not state["active"]:
                return fn(*a, **k)
            sync()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            sync()
            stats[name][0] += 1
            stats[name][1] += time.perf_counter() - t0
            return out

        return wrapped

    for name in OPS:
        if hasattr(ttnn, name):
            setattr(ttnn, name, wrap(name, getattr(ttnn, name)))

    original = alg._minibatch_step

    def profiled_step(i, rows):
        # the FIRST step of an update only warms the program cache (new shapes: JIT / binary loading would dominate the per-op times); profile the second
        if state["done"] or i == 0:
            return original(i, rows)
        sync()
        t0 = time.perf_counter()
        state["active"] = True
        original(i, rows)
        state["active"] = False
        sync()
        state["step_s"] = time.perf_counter() - t0
        state["done"] = True

    alg._minibatch_step = profiled_step
    alg.update()
    # an unprofiled eager step for reference (same shapes, programs cached)
    sync()
    t0 = time.perf_counter()
    original(0, alg.storage.batch_size // alg.num_mini_batches)
    sync()
    eager_step_s = time.perf_counter() - t0

    total_ops = sum(v[1] for v in stats.values())
    rows = sorted(stats.items(), key=lambda kv: -kv[1][1])
    print(f"envs={args.envs} hidden={args.hidden} mini-batch rows={alg.storage.batch_size // alg.num_mini_batches}")
    print(f"profiled step (op times + syncs) {state['step_s'] * 1e3:.1f} ms; sum of op times {total_ops * 1e3:.1f} ms; plain eager step {eager_step_s * 1e3:.1f} ms")
    for name, (n, t) in rows:
        print(f"  {name:12s} x{n:4d}  {t * 1e3:9.2f} ms  {100 * t / total_ops:5.1f} %  ({t / n * 1e3:.3f} ms/op)")
    if args.out:
        Path(args.out).write_text(json.dumps({"envs": args.envs, "hidden": args.hidden, "mini_batch_rows": alg.storage.batch_size // alg.num_mini_batches, "profiled_step_ms": state["step_s"] * 1e3, "sum_op_ms": total_ops * 1e3, "plain_eager_step_ms": eager_step_s * 1e3,
                                                  "ops": {k: {"calls": v[0], "total_ms": v[1] * 1e3} for k, v in rows}}, indent=1))
    alg.release_traces()


if __name__ == "__main__":
    main()
