#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Phase 5 benchmark matrix on ONE chip (TT, traced path): num_envs {4096, 16384, 65536} x hidden {[512,256,128], [1024,1024,512]}.
#
# Device jobs run strictly one after the other through scripts/tt_run.sh (BRIEF 2-3). A job that hits its timeout (exit 124) STOPS the whole run
# (BRIEF 2-6: no retries, no resets). A job that fails because the device memory is exhausted is retried once with 16 mini-batches (upstream default 4);
# any other failure stops the run.
#
#     scripts/run_bench_matrix.sh [extra bench_matrix.py args, e.g. --iters 3]
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=docs/porting/measurements
mkdir -p "$OUT" .run

for envs in 4096 16384 65536; do
  for hid in "512 256 128" "1024 1024 512"; do
    name="bench_tt_trace_${envs}_${hid// /-}"
    for mb in 4 16; do
      echo "== $name  mini-batches=$mb  $(date +%T)"
      # shellcheck disable=SC2086
      scripts/tt_run.sh --chips 16 --timeout 1500 --tag "$name" -- python scripts/bench_matrix.py --backend tt --trace --envs "$envs" --hidden $hid \
        --iters 3 --warmup 2 --mini-batches "$mb" --out "$OUT/${name}_mb${mb}.json" "$@" > ".run/${name}.mb${mb}.log" 2>&1
      rc=$?
      if [ "$rc" -eq 124 ]; then echo "TIMEOUT in $name (BRIEF 2-6): stopping, nothing else is started"; exit 124; fi
      if [ "$rc" -eq 0 ]; then echo "   ok"; break; fi
      if grep -q -i "out of memory" ".run/${name}.mb${mb}.log" && [ "$mb" -eq 4 ]; then echo "   out of device memory with 4 mini-batches: retrying with 16"; continue; fi
      echo "FAILED rc=$rc in $name (log .run/${name}.mb${mb}.log): stopping"; exit "$rc"
    done
  done
done
echo "matrix done $(date +%T)"
