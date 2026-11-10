# shellcheck shell=bash
# SPDX-License-Identifier: BSD-3-Clause
#
# ttnn_env.sh -- software-side environment for running ttnn (NOT device selection; that is tray_env.sh). Sourced by tray_env.sh.
#
# ttnn is the PyPI wheel `ttnn==0.81.1` installed in the dedicated venv, with the sfpi 7.89.0 toolchain extracted next to it
# (docs/porting/DECISIONS.md ADR-0006). The wheel locates its own runtime root; nothing about it is configured here.
# This file only sets hygiene variables.

# Private JIT-kernel cache and logs: never share or delete the common caches (BRIEF 2-5), and never write into a tt-metal tree.
export TT_METAL_CACHE="$TT_RSL_RL_ROOT/.run/tt_metal_cache"
export TT_METAL_LOGS_PATH="$TT_RSL_RL_ROOT/.run/tt_logs"
# ttnn's own default cache/tmp dirs are ~/.cache/ttnn and /tmp/ttnn, shared with every other ttnn user on this host: keep ours private.
export TTNN_CONFIG_OVERRIDES="{\"cache_path\": \"$TT_RSL_RL_ROOT/.run/ttnn_cache\", \"model_cache_path\": \"$TT_RSL_RL_ROOT/.run/ttnn_cache/models\", \"tmp_dir\": \"$TT_RSL_RL_ROOT/.run/ttnn_tmp\", \"root_report_path\": \"$TT_RSL_RL_ROOT/.run/ttnn_reports\"}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TT_METAL_CACHE" "$TT_METAL_LOGS_PATH" "$TT_RSL_RL_ROOT/.run/ttnn_cache/models" "$TT_RSL_RL_ROOT/.run/ttnn_tmp" 2>/dev/null || true
