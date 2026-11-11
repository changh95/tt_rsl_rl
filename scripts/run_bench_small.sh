#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Phase 5: the UNFAVOURABLE regimes. Small batches / small networks, where the fixed per-step costs of the device (one act round trip per environment step,
# ~600 ops per replayed update step) are not amortised: TT traced vs upstream CPU, plus the eager TT path at 4096 envs (what the trace buys).
# Device jobs strictly one after the other (BRIEF 2-3); a timeout (exit 124) stops everything (BRIEF 2-6). The CPU runs are host only and run
# between the device jobs, never concurrently with them (timing).
#
#     scripts/run_bench_small.sh
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=docs/porting/measurements
mkdir -p "$OUT" .run

for envs in 256 1024; do
  for hid in "64 64" "512 256 128"; do
    name="${envs}_${hid// /-}"
    echo "== TT traced $name  $(date +%T)"
    # shellcheck disable=SC2086
    scripts/tt_run.sh --chips 16 --timeout 600 --tag "small_tt_$name" -- python scripts/bench_matrix.py --backend tt --trace --envs "$envs" --hidden $hid --iters 5 --warmup 2 \
      --out "$OUT/bench_tt_trace_${name}_mb4.json" > ".run/small_tt_$name.log" 2>&1
    rc=$?; if [ "$rc" -eq 124 ]; then echo "TIMEOUT (BRIEF 2-6): stopping"; exit 124; fi; [ "$rc" -eq 0 ] || { echo "FAILED rc=$rc $name"; exit "$rc"; }
    echo "== CPU $name  $(date +%T)"
    python scripts/bench_matrix.py --backend cpu --threads 32 --envs "$envs" --hidden $hid --iters 5 --warmup 2 --out "$OUT/bench_cpu_${name}.json" > ".run/small_cpu_$name.log" 2>&1 || { echo "FAILED cpu $name"; exit 1; }
  done
done
for hid in "512 256 128" "1024 1024 512"; do
  name="4096_${hid// /-}"
  echo "== TT eager $name  $(date +%T)"
  # shellcheck disable=SC2086
  scripts/tt_run.sh --chips 16 --timeout 900 --tag "eager_$name" -- python scripts/bench_matrix.py --backend tt --envs 4096 --hidden $hid --iters 3 --warmup 2 \
    --out "$OUT/bench_tt_eager_${name}.json" > ".run/eager_$name.log" 2>&1
  rc=$?; if [ "$rc" -eq 124 ]; then echo "TIMEOUT (BRIEF 2-6): stopping"; exit 124; fi; [ "$rc" -eq 0 ] || { echo "FAILED rc=$rc eager $name"; exit "$rc"; }
done
echo "small-regime runs done $(date +%T)"
