# SPDX-License-Identifier: BSD-3-Clause
"""Host-only repository invariants (BRIEF 3.1 and 2-2).

1. `rsl_rl/` is byte-identical to the pinned upstream tag (the plugin never patches upstream).
2. `rsl_rl_tt/device.py` is the ONLY place that opens a Tenstorrent device.
"""

import importlib.metadata as md
import re
import subprocess
from pathlib import Path

from rsl_rl_tt.version_pin import RSL_RL_LIB_VERSION, RSL_RL_TAG

REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / "rsl_rl_tt" / "device.py"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60)


def test_upstream_dir_matches_pinned_tag():
    assert _git("rev-parse", "--verify", f"{RSL_RL_TAG}^{{commit}}").returncode == 0, f"tag {RSL_RL_TAG} missing: fetch upstream tags"
    diff = _git("diff", "--stat", RSL_RL_TAG, "--", "rsl_rl/")  # working tree (incl. uncommitted) vs the tag
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.strip() == "", f"rsl_rl/ must stay identical to {RSL_RL_TAG}:\n{diff.stdout}"
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", "rsl_rl/")
    assert untracked.stdout.strip() == "", f"untracked files inside rsl_rl/:\n{untracked.stdout}"


def test_installed_lib_version_matches_pin():
    assert md.version("rsl-rl-lib") == RSL_RL_LIB_VERSION


# built from pieces so that this file does not match its own scan
_CALL = re.compile(r"\b(open_" + r"device|open_mesh_" + r"device|Create" + r"Device|Create" + r"Devices|MeshDevice" + r"\s*\()")
_SCANNED = [REPO / "rsl_rl_tt", REPO / "scripts", REPO / "tests"]
_EXEMPT = {ENTRY.resolve(), Path(__file__).resolve(), (REPO / "tests" / "tt" / "test_tray_guard.py").resolve()}


def test_only_device_py_opens_devices():
    offenders = []
    for root in _SCANNED:
        for path in sorted(root.rglob("*.py")):
            if path.resolve() in _EXEMPT:
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if _CALL.search(line) and not line.lstrip().startswith("#"):
                    offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, "devices may only be opened in rsl_rl_tt/device.py:\n" + "\n".join(offenders)
