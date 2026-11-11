#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Phase 5 benchmark matrix, upstream CPU side (HOST ONLY, no device): the same num_envs x hidden configurations as run_bench_matrix.sh with the
# unmodified rsl_rl PPO, plus the cost of the real CPU tracking12 environment at each size. Nothing else CPU-heavy may run meanwhile.
#
#     scripts/run_bench_cpu.sh
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=docs/porting/measurements
THREADS=32   # physical cores of the host (AMD EPYC 9354P)
mkdir -p "$OUT" .run

for envs in 4096 16384 65536; do
  echo "== env cost $envs  $(date +%T)"
  python scripts/bench_matrix.py --backend cpu --threads "$THREADS" --envs "$envs" --hidden 512 256 128 --env-cost --out "$OUT/bench_env_cost_${envs}.json" > ".run/bench_env_cost_${envs}.log" 2>&1 || { echo "env cost failed"; exit 1; }
done
for envs in 4096 16384 65536; do
  for hid in "512 256 128" "1024 1024 512"; do
    name="bench_cpu_${envs}_${hid// /-}"
    iters=3; [ "$envs" -ge 65536 ] && iters=2
    echo "== $name  iters=$iters  $(date +%T)"
    # shellcheck disable=SC2086
    python scripts/bench_matrix.py --backend cpu --threads "$THREADS" --envs "$envs" --hidden $hid --iters "$iters" --warmup 1 --out "$OUT/${name}.json" > ".run/${name}.log" 2>&1 || { echo "FAILED $name (log .run/${name}.log)"; exit 1; }
  done
done
echo "cpu matrix done $(date +%T)"
