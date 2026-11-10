# shellcheck shell=bash
# SPDX-License-Identifier: BSD-3-Clause
#
# tray_env.sh -- the ONLY place that sets device-selection environment variables (BRIEF 2-2, Phase 0 step 2).
#
#   * SOURCE it (scripts/tt_run.sh does); never execute it. rsl_rl_tt/device.py re-evaluates this file in a clean
#     sub-shell and refuses to open anything if the live environment differs from it.
#   * OWNER-CONTROLLED: do not edit without the owner's approval (permissions.deny + tray_guard hook enforce this).
#
# Tray 3 = /dev/tenstorrent/16 .. 23 (8 chips). Nothing else may ever be used (BRIEF 2-1).
#
# Why BDF tokens: an integer in TT_VISIBLE_DEVICES is NOT a /dev/tenstorrent/N number; it is an index into the nodes sorted by
# PCI BDF (see docs/porting/TTNN_OP_SURVEY.md item 11). Full BDFs select exactly one chip and a typo fails closed.
# The node -> BDF map below was read on 2026-11-10 from /sys/class/tenstorrent/tenstorrent!N/device (symlink read only, no
# device open); device.py re-verifies it on every open.

_tray_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_RSL_RL_ROOT="$(cd "$_tray_env_dir/.." && pwd)"

export TT_TRAY_ID=3
export TT_TRAY_NODE_BDFS="16=0000:c1:00.0 17=0000:c2:00.0 18=0000:c3:00.0 19=0000:c4:00.0 20=0000:c5:00.0 21=0000:c6:00.0 22=0000:c7:00.0 23=0000:c8:00.0"
export TT_ALLOWED_DEV_NODES="/dev/tenstorrent/16 /dev/tenstorrent/17 /dev/tenstorrent/18 /dev/tenstorrent/19 /dev/tenstorrent/20 /dev/tenstorrent/21 /dev/tenstorrent/22 /dev/tenstorrent/23"

# Which of the tray's chips this run may use (space/comma separated /dev node numbers). Default: all 8.
# tt_run.sh --chips 16 (or 16,17) selects a subset for 1x1 / 1x2 bring-up; anything outside 16..23 is rejected.
_chips="${TT_TRAY_CHIPS-16 17 18 19 20 21 22 23}"   # '-' (not ':-'): an explicitly EMPTY selection must be refused, not widened
_chips="${_chips//,/ }"
_visible=""
_seen=" "
for _c in $_chips; do
    _bdf=""
    for _pair in $TT_TRAY_NODE_BDFS; do
        if [ "${_pair%%=*}" = "$_c" ]; then _bdf="${_pair#*=}"; fi
    done
    if [ -z "$_bdf" ]; then
        echo "tray_env: chip '$_c' is not one of the tray-$TT_TRAY_ID nodes (16..23); refusing" >&2
        return 1 2>/dev/null || exit 1
    fi
    case "$_seen" in *" $_c "*) continue ;; esac
    _seen="$_seen$_c "
    _visible="${_visible:+$_visible,}$_bdf"
done
if [ -z "$_visible" ]; then
    echo "tray_env: no chips selected; refusing (an empty TT_VISIBLE_DEVICES would be ambiguous)" >&2
    return 1 2>/dev/null || exit 1
fi
export TT_TRAY_CHIPS="${_seen# }"
export TT_TRAY_CHIPS="${TT_TRAY_CHIPS% }"
export TT_VISIBLE_DEVICES="$_visible"
unset _chips _visible _seen _c _bdf _pair

# These must stay unset: the single-process, single-tray path relies on metal's auto-discovery over the visible chips.
# TT_METAL_VISIBLE_DEVICES has no effect in this tt-metal version (dead variable) and would only mislead.
unset TT_MESH_ID TT_MESH_HOST_RANK TT_MESH_GRAPH_DESC_PATH TT_METAL_VISIBLE_DEVICES TT_METAL_MOCK_CLUSTER_DESC_PATH

# One lock per tray, shared by every clone of this project (chips are exclusive per process; BRIEF 2-3).
export TT_TRAY_LOCK_FILE="/tmp/tt_rsl_rl_tray${TT_TRAY_ID}.lock"

# Dedicated Python environment (BRIEF 2-7): packages are installed only here.
export PYTHON_ENV="$TT_RSL_RL_ROOT/.venv"
export VIRTUAL_ENV="$PYTHON_ENV"
export PATH="$PYTHON_ENV/bin:$PATH"

# Software-side env (how `import ttnn` is provided, private caches). Not device selection; may change without touching the above.
if [ -f "$_tray_env_dir/ttnn_env.sh" ]; then
    # shellcheck source=scripts/ttnn_env.sh
    . "$_tray_env_dir/ttnn_env.sh"
fi
unset _tray_env_dir
