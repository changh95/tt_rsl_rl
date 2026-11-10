# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests of scripts/ns_mask_dev.sh (DECISIONS ADR-0005 option II) with FAKE placeholder files.

No Tenstorrent node is touched: the "dev dir" is a scratch directory with regular files (and /dev/null bind-mounted over some of
them for the character-device checks). The mounts live in a private user+mount namespace and vanish with it.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MASK = REPO / "scripts" / "ns_mask_dev.sh"
TT_RUN = REPO / "scripts" / "tt_run.sh"

UNSHARE = ["unshare", "--user", "--map-root-user", "--mount", "--"]


def _userns_works() -> bool:
    if shutil.which("unshare") is None:
        return False
    return subprocess.run([*UNSHARE, "true"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not _userns_works(), reason="unprivileged user+mount namespaces are not available here")


def _fake_dev(tmp_path: Path, names) -> Path:
    dev = tmp_path / "dev"
    dev.mkdir()
    for n in names:
        (dev / str(n)).write_text("")
    return dev


def _mask(tmp_path, dev, keep, *cmd, extra=()):
    return subprocess.run(
        [*UNSHARE, str(MASK), "--dev-dir", str(dev), "--keep", keep, "--stage", str(tmp_path / "stage"), *extra, "--", *cmd],
        capture_output=True,
        text=True,
    )


def test_job_sees_only_the_kept_nodes_and_the_host_view_is_untouched(tmp_path):
    dev = _fake_dev(tmp_path, [3, 16, 17, 18, 23, 24, 99])
    res = _mask(tmp_path, dev, "16 17", "ls", "-A", str(dev))
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == ["16", "17"]
    assert sorted(os.listdir(dev), key=int) == ["3", "16", "17", "18", "23", "24", "99"]  # outside the namespace nothing changed


def test_comma_separated_keep_list_and_single_chip(tmp_path):
    dev = _fake_dev(tmp_path, range(16, 24))
    assert _mask(tmp_path, dev, "20,21", "ls", "-A", str(dev)).stdout.split() == ["20", "21"]
    assert _mask(tmp_path, dev, "16", "ls", "-A", str(dev)).stdout.split() == ["16"]


def test_kept_nodes_are_the_real_files_not_copies(tmp_path):
    dev = _fake_dev(tmp_path, [16, 17])
    (dev / "16").write_text("original")
    res = _mask(tmp_path, dev, "16", "cat", str(dev / "16"))
    assert res.stdout == "original"


def test_character_devices_survive_the_mask_and_open_read_write(tmp_path):
    dev = _fake_dev(tmp_path, [16, 17, 30])
    script = (
        f"set -e; for n in 16 17 30; do mount --bind /dev/null {dev}/$n; done; "
        f'{MASK} --dev-dir {dev} --keep "16 17" --stage {tmp_path}/stage --require-chardev -- '
        f"python3 -c \"import os; [os.close(os.open('{dev}/%d' % n, os.O_RDWR)) for n in (16, 17)]; print(sorted(os.listdir('{dev}')))\""
    )
    res = subprocess.run([*UNSHARE, "bash", "-c", script], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "['16', '17']"


def test_require_chardev_refuses_regular_files(tmp_path):
    dev = _fake_dev(tmp_path, [16])
    res = _mask(tmp_path, dev, "16", "true", extra=["--require-chardev"])
    assert res.returncode == 2 and "not a character device" in res.stderr


@pytest.mark.parametrize("keep", ["99", "16 99", "x", ""])
def test_missing_or_bad_kept_node_refuses_and_does_not_run_the_command(tmp_path, keep):
    dev = _fake_dev(tmp_path, [16])
    marker = tmp_path / "ran"
    res = _mask(tmp_path, dev, keep, "touch", str(marker))
    assert res.returncode == 2 and not marker.exists()


def test_refuses_outside_a_user_namespace(tmp_path):
    dev = _fake_dev(tmp_path, [16])
    res = subprocess.run(
        [str(MASK), "--dev-dir", str(dev), "--keep", "16", "--stage", str(tmp_path / "stage"), "--", "true"], capture_output=True, text=True
    )
    if Path("/proc/self/uid_map").read_text().split()[:3] != ["0", "0", "4294967295"]:
        pytest.skip("the test runner itself already lives in a user namespace")
    assert res.returncode == 2 and "not inside a user namespace" in res.stderr


def test_exec_keeps_the_pid_so_recorded_pids_stay_valid(tmp_path):
    dev = _fake_dev(tmp_path, [16])
    res = _mask(tmp_path, dev, "16", "sh", "-c", "echo $$")
    assert res.returncode == 0 and res.stdout.strip().isdigit()


# ---- tt_run.sh must always go through the mask (static check, nothing is run) ------------------------------------------------------
def test_tt_run_always_enters_the_namespace_mask():
    text = TT_RUN.read_text()
    last = text[text.rindex("\nexec ") :]
    assert "unshare --user --map-root-user --mount" in last
    assert "ns_mask_dev.sh" in last and '--keep "$TT_TRAY_CHIPS"' in last and "--require-chardev" in last
    assert "timeout --foreground" in last
    assert text.count("\nexec timeout") == 0, "a code path execs the job without the namespace mask"
