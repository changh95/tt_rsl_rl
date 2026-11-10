# SPDX-License-Identifier: BSD-3-Clause
"""Print what a job started by scripts/tt_run.sh can see (ADR-0005 option II). Opens NO device; run it only through tt_run.sh.

    scripts/tt_run.sh --chips 16 --timeout 60 -- python scripts/ns_check.py
"""

import os
from pathlib import Path

from rsl_rl_tt import device as dev

expected = dev.load_expected_env(os.environ.get("TT_TRAY_CHIPS"))
cfg = dev.check_env(expected)
visible = dev.list_visible_dev_nodes()
print("uid:", os.getuid(), "uid_map:", Path("/proc/self/uid_map").read_text().split())
print("chips selected:", cfg.chips)
print("TT_VISIBLE_DEVICES:", os.environ.get("TT_VISIBLE_DEVICES"))
print("RSL_RL_TT_LOCK_HELD:", os.environ.get("RSL_RL_TT_LOCK_HELD"))
print("nodes visible to this job:", sorted(visible))
print("char devices:", sorted(n for n in visible if os.path.exists(n) and Path(n).is_char_device()))
print("fds holding a tenstorrent node:", sorted(dev.list_open_tt_nodes()))
dev.verify_sysfs_bdfs(cfg)
dev.verify_dev_view(cfg, visible)
print("sysfs BDF check OK, dev-view check OK")
