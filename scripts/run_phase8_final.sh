#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Phase 8 verification chain (v1 suites + the Phase 8.x suites: 8.1 heteroscedastic Gaussian, 8.2 RND) on the final code: ONE chip, a 1x2 mesh and the full 2x4 tray mesh. Device jobs strictly one after the
# other (BRIEF 2-3); the first failure or timeout (exit 124, BRIEF 2-6) stops the chain: no retries, no resets.
#
#     scripts/run_phase8_final.sh
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

step() { # name command...
  local name=$1; shift
  echo "== $name  $(date +%T)"
  "$@" > ".run/p8final_${name//[^A-Za-z0-9]/_}.log" 2>&1; local rc=$?
  tail -3 ".run/p8final_${name//[^A-Za-z0-9]/_}.log" | grep -E "passed|failed" || true
  if [ "$rc" -ne 0 ]; then echo "STOP: '$name' failed (rc=$rc)$([ "$rc" -eq 124 ] && echo ', a TIMEOUT: BRIEF 2-6')"; exit "$rc"; fi
}

# one process per mesh: they share the mesh fixture (the whole-batch suites skip themselves on several chips, see conftest.py)
SUITES="tests/tt/test_tt_modules.py tests/tt/test_tt_rollout.py tests/tt/test_tt_backward.py tests/tt/test_tt_update.py tests/tt/test_tt_checkpoint.py tests/tt/test_tt_hetero_update.py tests/tt/test_tt_hetero_checkpoint.py tests/tt/test_tt_rnd.py tests/tt/test_tt_rnd_update.py tests/tt/test_tt_rnd_checkpoint.py"

# shellcheck disable=SC2086
step "1 chip: suites" scripts/tt_run.sh --chips 16 --timeout 1500 --tag p8f_1_suites -- python -m pytest -m tt $SUITES -q -p no:cacheprovider
step "1 chip: trace tests" scripts/tt_run.sh --chips 16 --timeout 600 --tag p8f_1_trace -- python -m pytest -m tt tests/tt/test_tt_trace.py -q -p no:cacheprovider
step "1 chip: heteroscedastic trace tests" scripts/tt_run.sh --chips 16 --timeout 600 --tag p8f_1_htrace -- python -m pytest -m tt tests/tt/test_tt_hetero_trace.py -q -p no:cacheprovider
step "1 chip: RND trace tests" scripts/tt_run.sh --chips 16 --timeout 600 --tag p8f_1_rtrace -- python -m pytest -m tt tests/tt/test_tt_rnd_trace.py -q -p no:cacheprovider
# shellcheck disable=SC2086
step "1x2: suites" scripts/tt_run.sh --chips 16,17 --timeout 1500 --tag p8f_2_suites -- env TT_TEST_MESH=1x2 python -m pytest -m tt $SUITES -q -p no:cacheprovider
step "1x2: multi-chip tests" scripts/tt_run.sh --chips 16,17 --timeout 1200 --tag p8f_2_multi -- env TT_TEST_MESH=1x2 python -m pytest -m tt tests/tt/test_tt_multichip.py tests/tt/test_tt_hetero_multichip.py tests/tt/test_tt_rnd_multichip.py -q -p no:cacheprovider
step "1x2: trace tests" scripts/tt_run.sh --chips 16,17 --timeout 600 --tag p8f_2_trace -- env TT_TEST_MESH=1x2 python -m pytest -m tt tests/tt/test_tt_trace.py -q -p no:cacheprovider
step "1x2: heteroscedastic trace tests" scripts/tt_run.sh --chips 16,17 --timeout 600 --tag p8f_2_htrace -- env TT_TEST_MESH=1x2 python -m pytest -m tt tests/tt/test_tt_hetero_trace.py -q -p no:cacheprovider
step "1x2: RND trace tests" scripts/tt_run.sh --chips 16,17 --timeout 600 --tag p8f_2_rtrace -- env TT_TEST_MESH=1x2 python -m pytest -m tt tests/tt/test_tt_rnd_trace.py -q -p no:cacheprovider
# shellcheck disable=SC2086
step "2x4: suites" scripts/tt_run.sh --timeout 1500 --tag p8f_8_suites -- env TT_TEST_MESH=2x4 python -m pytest -m tt $SUITES -q -p no:cacheprovider
step "2x4: multi-chip tests" scripts/tt_run.sh --timeout 1200 --tag p8f_8_multi -- env TT_TEST_MESH=2x4 python -m pytest -m tt tests/tt/test_tt_multichip.py tests/tt/test_tt_hetero_multichip.py tests/tt/test_tt_rnd_multichip.py -q -p no:cacheprovider
step "2x4: trace tests" scripts/tt_run.sh --timeout 600 --tag p8f_8_trace -- env TT_TEST_MESH=2x4 python -m pytest -m tt tests/tt/test_tt_trace.py -q -p no:cacheprovider
step "2x4: heteroscedastic trace tests" scripts/tt_run.sh --timeout 600 --tag p8f_8_htrace -- env TT_TEST_MESH=2x4 python -m pytest -m tt tests/tt/test_tt_hetero_trace.py -q -p no:cacheprovider
step "2x4: RND trace tests" scripts/tt_run.sh --timeout 600 --tag p8f_8_rtrace -- env TT_TEST_MESH=2x4 python -m pytest -m tt tests/tt/test_tt_rnd_trace.py -q -p no:cacheprovider
echo "phase 8 verification chain done $(date +%T)"
