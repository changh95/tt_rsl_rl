#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# kill_my_jobs.sh -- stop ONLY the jobs that scripts/tt_run.sh recorded in .run/pids (BRIEF section 2-5).
# Never matches by process name. An entry is honoured only if /proc/PID still has the recorded start time
# (guards against PID reuse), and only that PID's own descendants are touched.
# Escalation: SIGINT to the job (timeout forwards it, Python closes the mesh) -> SIGTERM -> SIGKILL to the job and its
# descendants, with waits in between. It never resets or touches any device; after a hang, collect logs and report.
#
#   scripts/kill_my_jobs.sh [--wait SEC]     (SEC = wait per escalation step, default 30)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDS="$REPO_ROOT/.run/pids"
STEP_WAIT=30
if [[ "${1:-}" == "--wait" ]]; then STEP_WAIT="${2:?--wait needs seconds}"; fi

proc_start() { sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'; }
alive() { [[ -r "/proc/$1/stat" && "$(proc_start "$1")" == "$2" ]]; }
descendants() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do echo "$c"; descendants "$c"; done; }
wait_dead() {   # wait_dead PID START SECONDS
    local i
    for ((i = 0; i < $3; i++)); do alive "$1" "$2" || return 0; sleep 1; done
    alive "$1" "$2" && return 1 || return 0
}

[[ -s "$PIDS" ]] || { echo "kill_my_jobs: no recorded jobs in $PIDS"; exit 0; }

found=0
while read -r pid start tag _; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    alive "$pid" "$start" || { echo "kill_my_jobs: $pid ($tag) already gone/stale, skipping"; continue; }
    found=1
    for sig in INT TERM KILL; do
        echo "kill_my_jobs: SIG$sig -> job $pid ($tag)"
        if [[ "$sig" == "KILL" ]]; then
            kill -KILL $(descendants "$pid") "$pid" 2>/dev/null || true   # whole tree of OUR job, collected right now
        else
            kill "-$sig" "$pid" 2>/dev/null || true
        fi
        if wait_dead "$pid" "$start" "$STEP_WAIT"; then echo "kill_my_jobs: job $pid exited"; break; fi
    done
    if alive "$pid" "$start"; then echo "kill_my_jobs: job $pid STILL ALIVE after SIGKILL -- stop and report; do NOT reset devices" >&2; fi
done < "$PIDS"
[[ "$found" -eq 1 ]] || echo "kill_my_jobs: nothing running"
