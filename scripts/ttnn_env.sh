# shellcheck shell=bash
# SPDX-License-Identifier: BSD-3-Clause
#
# ttnn_env.sh -- software-side environment for running ttnn (NOT device selection; that is tray_env.sh). Sourced by tray_env.sh.
#
# STATUS: the way `import ttnn` is provided is NOT decided yet (docs/porting/DECISIONS.md ADR-0006):
#   - the existing tt-metal tree was built with WITH_PYTHON_BINDINGS=OFF (no _ttnn.so), so it cannot be imported;
#   - candidates: PyPI wheel ttnn==0.81.1 into the dedicated venv, or a separate bindings build.
# Until the owner decides, this file only sets hygiene variables that are correct for every route.

# Private JIT-kernel cache and logs: never share or delete the common caches (BRIEF 2-5), and never write into a tt-metal tree.
export TT_METAL_CACHE="$TT_RSL_RL_ROOT/.run/tt_metal_cache"
export TT_METAL_LOGS_PATH="$TT_RSL_RL_ROOT/.run/tt_logs"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TT_METAL_CACHE" "$TT_METAL_LOGS_PATH" 2>/dev/null || true
