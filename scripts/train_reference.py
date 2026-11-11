# SPDX-License-Identifier: BSD-3-Clause
"""Train the UNMODIFIED upstream rsl_rl (CPU torch) on the test environments and store the golden baselines (BRIEF Phase 1).

    python scripts/train_reference.py                         # all envs, seeds 0 1 2, 8 threads per run, 3 runs in parallel
    python scripts/train_reference.py --envs cartpole --seeds 0 --iterations 20 --out /tmp/x --runs-dir /tmp/x/runs

Writes, per env and seed, `<out>/<env>/seed<k>.csv` (one row per iteration: every scalar the upstream Logger emits plus the wall time), and
`<out>/summary.json` (per-run wall time, final reward statistics, machine info) and `<out>/<env>_curves.png`. The full run directories
(config.json, model_*.pt, git diff of rsl_rl/) go to `--runs-dir` (default `.run/golden_runs`, not committed).
Host only: this script never imports ttnn and never opens a device, so it does not go through scripts/tt_run.sh.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.testing.envs import ENV_NAMES, RECURRENT_ENV_CLASSES  # noqa: E402

FINAL_WINDOW = 10  # "final reward" = mean of Train/mean_reward over the last 10 iterations


def _run_one(env_name: str, seed: int, runs_dir: str, iterations: int | None, threads: int, distribution: str = "GaussianDistribution", std_type: str = "scalar", rnd: bool = False, symmetry: bool = False, teacher: str | None = None, rnn: str | None = None) -> dict:
    # imported here so that each worker process builds torch state itself
    import dataclasses

    from rsl_rl_tt.testing.reference import REF_SPECS, train_reference, with_rnd, with_symmetry

    spec = dataclasses.replace(REF_SPECS[env_name], distribution=distribution, std_type=std_type)
    if rnn:  # recurrent models (Phase 8.5): the partially observable cart-pole with a GRU / LSTM
        from rsl_rl_tt.testing.reference import recurrent_e2e_spec

        spec = recurrent_e2e_spec(env_name, rnn)
    if rnd:
        spec = with_rnd(env_name, spec)
    if symmetry:
        spec = with_symmetry(env_name, spec)

    log_dir = os.path.join(runs_dir, env_name, f"seed{seed}")
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)  # our own output directory under --runs-dir, recreated for every run
    if teacher is not None:  # distillation (Phase 8.4): the upstream `Distillation` learns from the teacher checkpoint
        from rsl_rl_tt.testing.reference import DISTILL_ITERATIONS, train_distillation

        info = train_distillation(env_name, seed, log_dir, teacher, iterations=iterations if iterations is not None else DISTILL_ITERATIONS[env_name], spec=spec, num_threads=threads)
        info["log_dir"] = log_dir
        return info
    info = train_reference(env_name, seed, log_dir, iterations=iterations, spec=spec, num_threads=threads)
    info["log_dir"] = log_dir
    return info


def _make_teacher(env_name: str, path: str) -> str:
    from rsl_rl_tt.testing.reference import REF_SPECS, TEACHER_ITERATIONS, make_teacher_checkpoint

    if not os.path.exists(path):
        make_teacher_checkpoint(env_name, path, seed=0, iterations=TEACHER_ITERATIONS[env_name], num_envs=REF_SPECS[env_name].num_envs)
    return path


def _read_metrics(path: str) -> list[dict]:
    with open(path) as f:
        return [{k: (float(v) if v != "" else None) for k, v in row.items()} for row in csv.DictReader(f)]


def _final_reward(rows: list[dict]) -> float:
    vals = [r["Train/mean_reward"] for r in rows[-FINAL_WINDOW:] if r.get("Train/mean_reward") is not None]
    return statistics.mean(vals)


def _plot(env_name: str, curves: dict[int, list[dict]], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for seed, rows in sorted(curves.items()):
        xs = [r["iteration"] for r in rows if r.get("Train/mean_reward") is not None]
        ys = [r["Train/mean_reward"] for r in rows if r.get("Train/mean_reward") is not None]
        ax.plot(xs, ys, label=f"seed {seed}", linewidth=1.2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("Train/mean_reward")
    ax.set_title(f"upstream rsl_rl (CPU) golden curves: {env_name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="+", default=None, choices=ENV_NAMES + tuple(RECURRENT_ENV_CLASSES), help="default: the three standard envs (with --rnn: the partially observable cart-pole)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--iterations", type=int, default=None, help="override the per-env default (reference.REF_SPECS)")
    ap.add_argument("--threads", type=int, default=8, help="torch threads per run")
    ap.add_argument("--parallel", type=int, default=3, help="runs executed at the same time")
    ap.add_argument("--distribution", default="GaussianDistribution", help='action distribution of the actor, e.g. "HeteroscedasticGaussianDistribution" (Phase 8.1)')
    ap.add_argument("--std-type", default="scalar", choices=("scalar", "log"))
    ap.add_argument("--rnd", action="store_true", help="Random Network Distillation (Phase 8.2: reference.with_rnd)")
    ap.add_argument("--symmetry", action="store_true", help="symmetry augmentation + mirror loss (Phase 8.3: reference.with_symmetry)")
    ap.add_argument("--rnn", choices=("gru", "lstm"), default=None, help="recurrent actor and critic (Phase 8.5): the partially observable cart-pole with a GRU / LSTM")
    ap.add_argument("--distill", action="store_true", help="distillation (Phase 8.4): a student learns from a teacher; the teachers are upstream PPO runs made on first use in --teachers")
    ap.add_argument("--teachers", default=str(REPO / ".run" / "teachers"), help="directory of the teacher checkpoints <env>.pt (--distill)")
    ap.add_argument("--out", default=str(REPO / "docs" / "porting" / "golden"))
    ap.add_argument("--runs-dir", default=str(REPO / ".run" / "golden_runs"))
    args = ap.parse_args()

    args.envs = args.envs or (["cartpole_po"] if args.rnn else list(ENV_NAMES))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(e, s) for e in args.envs for s in args.seeds]
    teachers = {}
    if args.distill:
        os.makedirs(args.teachers, exist_ok=True)
        with cf.ProcessPoolExecutor(max_workers=args.parallel, mp_context=__import__("multiprocessing").get_context("spawn")) as pool:
            for e, fut in {e: pool.submit(_make_teacher, e, os.path.join(args.teachers, f"{e}.pt")) for e in args.envs}.items():
                teachers[e] = fut.result()
                print(f"teacher {e}: {teachers[e]}", flush=True)
    infos: dict[tuple[str, int], dict] = {}
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.parallel, mp_context=__import__("multiprocessing").get_context("spawn")) as pool:
        futs = {pool.submit(_run_one, e, s, args.runs_dir, args.iterations, args.threads, args.distribution, args.std_type, args.rnd, args.symmetry, teachers.get(e), args.rnn): (e, s) for e, s in jobs}
        for fut in cf.as_completed(futs):
            key = futs[fut]
            infos[key] = fut.result()
            print(f"done {key[0]} seed {key[1]}: {infos[key]['wall_s']:.1f} s", flush=True)

    import torch

    summary: dict = {
        "rsl_rl_tag": "v5.5.2",
        "distribution": args.distribution,
        "std_type": args.std_type,
        "rnd": args.rnd,
        "symmetry": args.symmetry,
        "distillation": args.distill,
        "rnn": args.rnn,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": {"cpu_count": os.cpu_count(), "threads_per_run": args.threads, "parallel_runs": args.parallel, "platform": platform.platform()},
        "final_window_iterations": FINAL_WINDOW,
        "total_wall_s": round(time.time() - t0, 1),
        "envs": {},
    }
    for env_name in args.envs:
        curves, per_seed = {}, {}
        (out / env_name).mkdir(exist_ok=True)
        for seed in args.seeds:
            info = infos[(env_name, seed)]
            src = os.path.join(info["log_dir"], "metrics.csv")
            shutil.copyfile(src, out / env_name / f"seed{seed}.csv")
            rows = _read_metrics(src)
            curves[seed] = rows
            per_seed[str(seed)] = {"iterations": info["iterations"], "wall_s": round(info["wall_s"], 1), "final_reward": _final_reward(rows), "first_reward": rows[0].get("Train/mean_reward")}
        finals = [v["final_reward"] for v in per_seed.values()]
        summary["envs"][env_name] = {
            "num_envs": infos[(env_name, args.seeds[0])]["num_envs"],
            "num_steps_per_env": infos[(env_name, args.seeds[0])]["num_steps"],
            "seeds": per_seed,
            "final_reward_mean": statistics.mean(finals),
            "final_reward_std": statistics.pstdev(finals),
        }
        _plot(env_name, curves, out / f"{env_name}_curves.png")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({e: (round(v["final_reward_mean"], 2), round(v["final_reward_std"], 2)) for e, v in summary["envs"].items()}))


if __name__ == "__main__":
    main()
