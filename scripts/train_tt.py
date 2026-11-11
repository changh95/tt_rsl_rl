# SPDX-License-Identifier: BSD-3-Clause
"""Train the test envs with the UNMODIFIED upstream OnPolicyRunner + `rsl_rl_tt.ppo:TTPPO` on one chip and compare with the golden baselines (BRIEF Phase 4).

Device job: run ONLY through the wrapper (one process, runs executed one after the other on the open 1x1 mesh):

    scripts/tt_run.sh --chips 16 --timeout 3600 --tag train -- python scripts/train_tt.py [--envs cartpole ...] [--seeds 0 1 2] [--iterations N]

Writes `<out>/<env>/seed<k>.csv` (same columns as the golden CSVs), `<out>/summary.json` (wall times, final rewards, the Phase 4 criterion per env) and
`<out>/<env>_vs_golden.png`. Full run directories (config.json, checkpoints) go to `--runs-dir` (.run/tt_runs, not committed).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.testing.envs import ENV_NAMES  # noqa: E402
from rsl_rl_tt.testing.golden import GOLDEN_DIR, final_reward, load_curve, tt_vs_golden  # noqa: E402
from rsl_rl_tt.testing.reference import REF_SPECS, train_reference, with_rnd  # noqa: E402


TT_LABEL = "1 chip"


def plot(env_name: str, seeds, tt_dir: Path, out_png: Path, golden_dir: Path = GOLDEN_DIR) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for seed in seeds:
        g = load_curve(env_name, seed, golden_dir)
        ax.plot([r["iteration"] for r in g if r.get("Train/mean_reward") is not None], [r["Train/mean_reward"] for r in g if r.get("Train/mean_reward") is not None], color="0.65", linewidth=1.0, label="upstream CPU (golden)" if seed == seeds[0] else None)
        t = load_curve(env_name, seed, tt_dir)
        ax.plot([r["iteration"] for r in t if r.get("Train/mean_reward") is not None], [r["Train/mean_reward"] for r in t if r.get("Train/mean_reward") is not None], linewidth=1.2, label=f"TT {TT_LABEL}, seed {seed}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Train/mean_reward")
    ax.set_title(f"{env_name}: TT ({TT_LABEL}) vs upstream CPU golden curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="+", default=list(ENV_NAMES), choices=ENV_NAMES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4, help="torch threads for the host-side work (env, noise, permutation)")
    ap.add_argument("--trace", action="store_true", help="use the traced path (Phase 5: act step and mini-batch step captured once and replayed)")
    ap.add_argument("--fidelity", default="HiFi4", help="matmul/reduction math fidelity; anything but HiFi4 is a Phase 5 precision experiment")
    ap.add_argument("--mesh", default="1x1", help="mesh shape RxC of the chips of this job (Phase 6 data parallel): 1x1, 1x2 (--chips 16,17), 2x4 (all eight)")
    ap.add_argument("--distribution", default="GaussianDistribution", help='action distribution of the actor, e.g. "HeteroscedasticGaussianDistribution" (Phase 8.1)')
    ap.add_argument("--std-type", default="scalar", choices=("scalar", "log"))
    ap.add_argument("--rnd", action="store_true", help="Random Network Distillation (Phase 8.2: reference.with_rnd; the golden curves must have been made with the same flag)")
    ap.add_argument("--golden", default=str(GOLDEN_DIR), help="directory of the upstream reference curves to compare with (must have been produced with the same --distribution / --std-type)")
    ap.add_argument("--out", default=str(REPO / "docs" / "porting" / "tt_runs"))
    ap.add_argument("--runs-dir", default=str(REPO / ".run" / "tt_runs"))
    args = ap.parse_args()
    rows, cols = (int(v) for v in args.mesh.split("x"))
    chips = rows * cols
    tt_cfg = {"trace": args.trace, "math_fidelity": args.fidelity, "mesh_shape": [rows, cols], **({"replica_check_interval": 10} if chips > 1 else {})}

    global TT_LABEL
    TT_LABEL = "1 chip" if chips == 1 else f"{chips} chips ({args.mesh})"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "backend": f"TTPPO on {'one Blackhole chip (tray 3, node 16)' if chips == 1 else f'a {args.mesh} mesh of {chips} Blackhole chips (tray 3), data parallel'}, fp32 policy, math fidelity {args.fidelity}, {'traced' if args.trace else 'eager'}, unmodified OnPolicyRunner",
        "envs": {},
    }
    t_all = time.time()
    for env_name in args.envs:
        (out / env_name).mkdir(exist_ok=True)
        per_seed = {}
        for seed in args.seeds:
            log_dir = os.path.join(args.runs_dir, env_name, f"seed{seed}")
            if os.path.isdir(log_dir):
                shutil.rmtree(log_dir)  # our own output directory under --runs-dir
            spec = dataclasses.replace(REF_SPECS[env_name], distribution=args.distribution, std_type=args.std_type)
            if args.rnd:
                spec = with_rnd(env_name, spec)
            info = train_reference(env_name, seed, log_dir, iterations=args.iterations, spec=spec, num_threads=args.threads, tt=True, tt_cfg=tt_cfg)
            shutil.copyfile(os.path.join(log_dir, "metrics.csv"), out / env_name / f"seed{seed}.csv")
            rows = load_curve(env_name, seed, out)
            per_seed[str(seed)] = {"iterations": info["iterations"], "wall_s": round(info["wall_s"], 1), "final_reward": final_reward(rows)}
            print(f"done {env_name} seed {seed}: {info['wall_s']:.0f} s, final reward {per_seed[str(seed)]['final_reward']:.2f}", flush=True)
        entry = {"seeds": per_seed, "num_envs": REF_SPECS[env_name].num_envs, "num_steps_per_env": REF_SPECS[env_name].num_steps}
        entry["criterion"] = tt_vs_golden(env_name, args.seeds, out, Path(args.golden))
        summary["envs"][env_name] = entry
        plot(env_name, args.seeds, out, out / f"{env_name}_vs_golden.png", Path(args.golden))
        print(env_name, json.dumps(entry["criterion"]), flush=True)
    summary["total_wall_s"] = round(time.time() - t_all, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
