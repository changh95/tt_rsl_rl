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


# Anything on the ttnn/ttml/tt_lib namespaces that opens a device OR builds the cluster (ttnn.get_num_devices constructs the
# MetalContext, i.e. opens every visible chip: TTNN_OP_SURVEY.md item 10). Fake classes in tests that merely define methods with
# these names do not match because the regex requires the `ttnn.` namespace prefix.
_NAMES = r"(open_device|open_mesh_device|create_mesh_device|CreateDevices?|get_num_devices|get_device_ids|get_pcie_device_ids|SystemMeshDescriptor|MeshDevice)"
_NS = r"\b(?:ttnn|_ttnn|ttl|tt_lib|ttml)\b"
_ATTR = re.compile(_NS + r"[\w.\s]*?\.\s*" + _NAMES + r"\b")
_FROM_IMPORT = re.compile(r"from\s+(?:ttnn|ttl|tt_lib|ttml)[\w.]*\s+import\s+[^#\n]*\b" + _NAMES + r"\b")
_SCANNED = [REPO / "rsl_rl_tt", REPO / "scripts", REPO / "tests"]
# test_tray_guard.py holds shell command STRINGS that must be blocked (data, not calls)
_EXEMPT = {ENTRY.resolve(), Path(__file__).resolve(), (REPO / "tests" / "tt" / "test_tray_guard.py").resolve()}


def _offending(line: str) -> bool:
    return not line.lstrip().startswith("#") and bool(_ATTR.search(line) or _FROM_IMPORT.search(line))


def test_only_device_py_opens_devices():
    offenders = []
    for root in _SCANNED:
        for path in sorted(root.rglob("*.py")):
            if path.resolve() in _EXEMPT:
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if _offending(line):
                    offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, "devices may only be opened in rsl_rl_tt/device.py:\n" + "\n".join(offenders)


def test_scanner_catches_the_ways_a_device_could_be_opened():
    bad = [
        "mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))",
        "dev = ttnn.open_device(device_id=0)",
        "with ttnn.create_mesh_device(shape) as m: pass",
        "n = ttnn.get_num_devices()",
        "ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()",
        "d = ttl.device.CreateDevice(0)",
        "from ttnn import open_mesh_device",
        "from ttnn.distributed import create_mesh_device, close_mesh_device",
        "x = ttnn . open_mesh_device(s)",
    ]
    good = [
        "# ttnn.open_mesh_device(s) is only mentioned in a comment",
        "def open_mesh_device(self, **kw):",
        'calls.append(("open_mesh_device", shape))',
        "from rsl_rl_tt.device import open_tray_mesh",
        "ttnn.close_mesh_device(mesh)",
        "t = ttnn.from_torch(x, device=mesh)",
    ]
    assert all(_offending(s) for s in bad), [s for s in bad if not _offending(s)]
    assert not any(_offending(s) for s in good), [s for s in good if _offending(s)]
