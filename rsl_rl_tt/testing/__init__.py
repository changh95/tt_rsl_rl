# SPDX-License-Identifier: BSD-3-Clause
"""Host-only test harness: CPU environments, golden-curve logging and parity utilities (BRIEF Phase 1). Never imports ttnn."""

from rsl_rl_tt.testing.envs import ENV_NAMES, make_env

__all__ = ["ENV_NAMES", "make_env"]
