#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Bash PreToolUse hook entry point (BRIEF section 2-4/2-5/2-10). The logic lives in tray_guard.py.
# Fails CLOSED: if the guard itself errors, the command is blocked (exit 2) instead of silently allowed.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
input="$(cat)"
printf '%s' "$input" | /usr/bin/python3 -I "$here/tray_guard.py"
code=$?
if [[ $code -eq 0 || $code -eq 2 ]]; then
    exit "$code"
fi
echo "tray_guard: internal error (exit $code); blocking to be safe. Ask the owner to look at .claude/hooks/tray_guard.py" >&2
exit 2
