#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Phase 6 scaling benchmark: chips {1, 2, 8} x num_envs {4096, 16384, 65536} (TOTAL over the chips) x hidden {[512,256,128], [1024,1024,512]}, traced path, null env.
# The 1-chip rows are the Phase 5 matrix (`bench_tt_trace_*_mb4.json`); one of them is repeated here to confirm that the single-chip path did not change.
# Device jobs strictly one after the other (BRIEF 2-3); the first failure or timeout (exit 124, BRIEF 2-6) stops everything: no retries, no resets.
#
#     scripts/run_bench_scaling.sh
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=docs/porting/measurements
mkdir -p "$OUT" .run

stop() { echo "STOP: $1 failed (rc=$2)$([ "$2" -eq 124 ] && echo ', a TIMEOUT: BRIEF 2-6')"; exit "$2"; }

echo "== 1 chip, repeat of 4096 x [512,256,128]  $(date +%T)"
scripts/tt_run.sh --chips 16 --timeout 600 --tag scale_1_repeat -- python scripts/bench_matrix.py --backend tt --trace --envs 4096 --hidden 512 256 128 --iters 3 --warmup 2 \
  --out "$OUT/bench_scaling_1chip_4096_512-256-128_repeat.json" > .run/scale_1_repeat.log 2>&1 || stop "1 chip repeat" $?

for envs in 4096 16384 65536; do
  for hid in "512 256 128" "1024 1024 512"; do
    name="${envs}_${hid// /-}"
    echo "== 2 chips (1x2) $name  $(date +%T)"
    # shellcheck disable=SC2086
    scripts/tt_run.sh --chips 16,17 --timeout 1500 --tag "scale_2_$name" -- python scripts/bench_matrix.py --backend tt --trace --mesh 1x2 --envs "$envs" --hidden $hid --iters 3 --warmup 2 \
      --out "$OUT/bench_scaling_2chips_${name}.json" > ".run/scale_2_$name.log" 2>&1 || stop "2 chips $name" $?
    echo "== 8 chips (2x4) $name  $(date +%T)"
    # shellcheck disable=SC2086
    scripts/tt_run.sh --timeout 1500 --tag "scale_8_$name" -- python scripts/bench_matrix.py --backend tt --trace --mesh 2x4 --envs "$envs" --hidden $hid --iters 3 --warmup 2 \
      --out "$OUT/bench_scaling_8chips_${name}.json" > ".run/scale_8_$name.log" 2>&1 || stop "8 chips $name" $?
  done
done
echo "scaling benchmark done $(date +%T)"
