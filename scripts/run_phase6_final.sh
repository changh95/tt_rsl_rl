#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Phase 6 verification chain on the final code: the device suites on ONE chip, on a 1x2 mesh and on the full 2x4 tray mesh. Device jobs strictly one after the other
# (BRIEF 2-3); the first failure or timeout (exit 124, BRIEF 2-6) stops the chain: no retries, no resets.
#
#     scripts/run_phase6_final.sh
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

step() { # name command...
  local name=$1; shift
  echo "== $name  $(date +%T)"
  "$@" > ".run/p6final_${name// /_}.log" 2>&1; local rc=$?
  tail -3 ".run/p6final_${name// /_}.log" | grep -E "passed|failed" || true
  if [ "$rc" -ne 0 ]; then echo "STOP: '$name' failed (rc=$rc)$([ "$rc" -eq 124 ] && echo ', a TIMEOUT: BRIEF 2-6')"; exit "$rc"; fi
}

SUITES="tests/tt/test_tt_modules.py tests/tt/test_tt_rollout.py tests/tt/test_tt_backward.py tests/tt/test_tt_update.py tests/tt/test_tt_checkpoint.py"

# shellcheck disable=SC2086
step "1 chip: phase 2-4 suites" scripts/tt_run.sh --chips 16 --timeout 900 --tag p6f_1_suites -- python -m pytest -m tt $SUITES -q -p no:cacheprovider
step "1 chip: trace tests" scripts/tt_run.sh --chips 16 --timeout 600 --tag p6f_1_trace -- python -m pytest -m tt tests/tt/test_tt_trace.py -q -p no:cacheprovider
# shellcheck disable=SC2086
step "1x2: chip-count independent suites" scripts/tt_run.sh --chips 16,17 --timeout 900 --tag p6f_2_suites -- env TT_TEST_MESH=1x2 python -m pytest -m tt $SUITES -q -p no:cacheprovider
step "1x2: multi-chip tests" scripts/tt_run.sh --chips 16,17 --timeout 900 --tag p6f_2_multi -- env TT_TEST_MESH=1x2 python -m pytest -m tt tests/tt/test_tt_multichip.py -q -p no:cacheprovider
step "1x2: trace tests" scripts/tt_run.sh --chips 16,17 --timeout 600 --tag p6f_2_trace -- env TT_TEST_MESH=1x2 python -m pytest -m tt tests/tt/test_tt_trace.py -q -p no:cacheprovider
# shellcheck disable=SC2086
step "2x4: chip-count independent suites" scripts/tt_run.sh --timeout 900 --tag p6f_8_suites -- env TT_TEST_MESH=2x4 python -m pytest -m tt $SUITES -q -p no:cacheprovider
step "2x4: multi-chip tests" scripts/tt_run.sh --timeout 900 --tag p6f_8_multi -- env TT_TEST_MESH=2x4 python -m pytest -m tt tests/tt/test_tt_multichip.py -q -p no:cacheprovider
step "2x4: trace tests" scripts/tt_run.sh --timeout 600 --tag p6f_8_trace -- env TT_TEST_MESH=2x4 python -m pytest -m tt tests/tt/test_tt_trace.py -q -p no:cacheprovider
echo "phase 6 verification chain done $(date +%T)"
