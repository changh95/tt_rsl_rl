#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# tt_run.sh -- the ONLY way to run anything that may open Tenstorrent devices (BRIEF section 2-3).
#
#   scripts/tt_run.sh [--timeout SEC] [--grace SEC] [--wait SEC] [--tag NAME] -- CMD [ARGS...]
#
# What it does, in order:
#   1. source scripts/tray_env.sh (the single place that sets device-related env vars)
#   2. take the per-tray lock with flock (non-blocking unless --wait is given): chips are exclusive
#      per process, so only one device job runs at a time
#   3. record this job's PID in .run/pids (only PIDs listed there may ever be signalled by us)
#   4. exec `timeout` (SIGINT first so Python can close the mesh in `finally`/atexit, then SIGKILL after
#      the grace period). It never resets anything: on a hang/device error, collect logs and stop (rule 2-6).
#
# Exit codes: 75 = tray busy / lock not acquired, 2 = usage/setup error, otherwise the job's own code
# (124 = timed out, 137 = had to be SIGKILLed).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMEOUT=900
GRACE=60
WAIT=0
TAG=job

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --grace)   GRACE="$2"; shift 2 ;;
        --wait)    WAIT="$2"; shift 2 ;;
        --tag)     TAG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --)        shift; break ;;
        *)         break ;;
    esac
done
if [[ $# -eq 0 ]]; then
    echo "tt_run: no command given" >&2
    usage >&2
    exit 2
fi
for v in "$TIMEOUT" "$GRACE" "$WAIT"; do
    [[ "$v" =~ ^[0-9]+$ ]] || { echo "tt_run: --timeout/--grace/--wait must be integers (seconds)" >&2; exit 2; }
done

# 1. device env (sets TT_TRAY_ID, TT_TRAY_LOCK_FILE, PYTHON_ENV, PATH, ...)
# shellcheck source=scripts/tray_env.sh
source "$REPO_ROOT/scripts/tray_env.sh"

if [[ ! -x "$PYTHON_ENV/bin/python" ]]; then
    echo "tt_run: PYTHON_ENV '$PYTHON_ENV' has no bin/python (create the dedicated venv first)" >&2
    exit 2
fi

# 2. per-tray lock (fd 9 stays open in the exec'd job and its children => held for the job's lifetime)
exec 9>"$TT_TRAY_LOCK_FILE"
if [[ "$WAIT" -gt 0 ]]; then
    flock -w "$WAIT" 9 || { echo "tt_run: tray $TT_TRAY_ID still busy after ${WAIT}s (lock: $TT_TRAY_LOCK_FILE)" >&2; exit 75; }
else
    flock -n 9 || { echo "tt_run: tray $TT_TRAY_ID is busy (lock: $TT_TRAY_LOCK_FILE); not starting" >&2; exit 75; }
fi
export RSL_RL_TT_LOCK_HELD="$TT_TRAY_ID"   # device.py refuses to open devices unless this equals the tray id

# 3. PID bookkeeping: lines are "PID STARTTICKS TAG CMD...". We hold the tray lock here, so editing is race-free.
mkdir -p "$REPO_ROOT/.run"
PIDS="$REPO_ROOT/.run/pids"
touch "$PIDS"
proc_start() { sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'; }
tmp="$(mktemp "$REPO_ROOT/.run/pids.XXXXXX")"
while read -r pid start rest; do   # prune dead/stale entries
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    [[ -r "/proc/$pid/stat" && "$(proc_start "$pid")" == "$start" ]] && echo "$pid $start $rest" >> "$tmp"
done < "$PIDS"
echo "$$ $(proc_start $$) $TAG $*" >> "$tmp"   # after `exec timeout` this PID IS the timeout process
mv "$tmp" "$PIDS"

# 4. run. `exec` keeps the PID recorded above and keeps SIGINT deliverable (a backgrounded child of a
#    non-interactive bash would start with SIGINT ignored, which would break Python's KeyboardInterrupt cleanup).
echo "tt_run: tray=$TT_TRAY_ID timeout=${TIMEOUT}s grace=${GRACE}s tag=$TAG pid=$$ :: $*" >&2
exec timeout --signal=INT --kill-after="${GRACE}s" "${TIMEOUT}s" "$@"
