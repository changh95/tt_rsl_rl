#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Phase 5 final measurement chain on the final code. Device jobs run strictly one after the other (BRIEF 2-3) and the chain STOPS at the first step that
# fails or times out (BRIEF 2-6: no retries, no resets).
#
#     scripts/run_phase5_final.sh
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=docs/porting/measurements

step() { # name command...
  local name=$1; shift
  echo "== $name  $(date +%T)"
  "$@"; local rc=$?
  if [ "$rc" -ne 0 ]; then echo "STOP: step '$name' failed (rc=$rc)$([ "$rc" -eq 124 ] && echo ', a TIMEOUT: BRIEF 2-6')"; exit "$rc"; fi
}

step matrix scripts/run_bench_matrix.sh
step small scripts/run_bench_small.sh
for envs in 4096 65536; do
  step "profile $envs" scripts/tt_run.sh --chips 16 --timeout 900 --tag "prof_final_$envs" -- python scripts/bench_matrix.py --backend tt --trace --profile --envs "$envs" \
    --hidden 512 256 128 --iters 3 --warmup 2 --out "$OUT/bench_tt_trace_profile_${envs}_512-256-128.json"
done
step "traced training (3 envs x 3 seeds)" scripts/tt_run.sh --chips 16 --timeout 3600 --tag train_trace_final -- python scripts/train_tt.py --trace \
  --out docs/porting/tt_runs_trace --runs-dir .run/tt_runs_trace
step "phase 2-4 device suites" scripts/tt_run.sh --chips 16 --timeout 900 --tag p5reg_final -- python -m pytest -m tt tests/tt/test_tt_modules.py tests/tt/test_tt_rollout.py \
  tests/tt/test_tt_backward.py tests/tt/test_tt_update.py tests/tt/test_tt_checkpoint.py -q -p no:cacheprovider
step "trace tests" scripts/tt_run.sh --chips 16 --timeout 600 --tag p5trace_final -- python -m pytest -m tt tests/tt/test_tt_trace.py -q -p no:cacheprovider
echo "phase 5 final chain done $(date +%T)"
