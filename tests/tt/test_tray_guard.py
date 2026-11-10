# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests for the Bash PreToolUse guard (.claude/hooks/tray_guard.py), BRIEF section 2-4/2-5/2-10.

Both directions matter: forbidden commands must be blocked, and plain reading of sources that merely MENTION forbidden
tools (grep/cat/git log/commit messages/heredocs) must NOT be blocked.
"""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOK_PY = REPO / ".claude" / "hooks" / "tray_guard.py"
HOOK_SH = REPO / ".claude" / "hooks" / "tray_guard.sh"

spec = importlib.util.spec_from_file_location("tray_guard", HOOK_PY)
guard = importlib.util.module_from_spec(spec)
sys.modules["tray_guard"] = guard  # dataclasses (with postponed annotations) need the module to be registered
spec.loader.exec_module(guard)


def verdict(cmd: str, cwd: Path | str = REPO) -> list[str]:
    ctx = guard.Ctx(cwd=os.path.realpath(str(cwd)))
    guard.analyze(cmd, ctx)
    return ctx.reasons


BLOCKED = [
    # -- hardware tools (BRIEF 2-4), in every position a command can hide
    "tt-smi",
    "tt-smi -r",
    "tt-smi -ls",
    "/usr/local/bin/tt-smi -glx_reset",
    "tt_smi",
    "python -m tt_smi",
    "tt-flash flash fw.bin",
    "tt-topology -l mesh",
    "ipmitool chassis power cycle",
    "sudo ls",
    "echo hi && sudo reboot",
    "reboot",
    "bash -c 'tt-smi -r'",
    "sh -lc 'sudo ls'",
    "timeout 5 tt-smi",
    "env FOO=1 tt-smi",
    "FOO=1 tt-smi",
    "nohup tt-smi -r &",
    "cd /tmp && tt-smi",
    "echo $(tt-smi -ls)",
    'echo "$(tt-smi -ls)"',
    "echo `tt-smi`",
    'eval "tt-smi -r"',
    "ls | xargs tt-smi",
    "scripts/tt_run.sh -- tt-smi -r",
    "bash <<EOF\ntt-smi -r\nEOF",
    "python scripts/hw_health_check.py",
    "bash tools/glx_reset.sh",
    "ttrl doctor",
    "python -m ttrl train",
    "modprobe tenstorrent",
    "sysctl -w vm.nr_hugepages=1",
    "echo 1 > /proc/sys/vm/nr_hugepages",
    "systemctl restart foo",
    "python -c \"import subprocess; subprocess.run(['tt-smi'])\"",
    "echo 'unterminated; tt-smi -r",
    # -- other chips (BRIEF 2-1)
    "cat /dev/tenstorrent/0",
    "ls /dev/tenstorrent",
    "ls /dev/tenstorrent/*",
    "stat /dev/tenstorrent/24",
    "stat /dev/tenstorrent/1",
    "stat /dev/tenstorrent/15",
    "for i in 16 17; do stat /dev/tenstorrent/$i; done",
    "echo x > /dev/tenstorrent/16",
    "fuser -k /dev/tenstorrent/16",
    # -- name-based killing (BRIEF 2-5)
    "pkill python",
    "killall -9 python3",
    "kill 1",
    "kill -9 99999999",
    "kill $(pgrep python)",
    "pgrep -f foo | xargs kill",
    # -- deleting other people's files / shared caches (BRIEF 2-5)
    "rm -rf /dev/shm/*",
    "rm -rf /tmp/foo",
    "rm -rf /tmp/*",
    "rm -rf ~/.cache/tt-metal-cache",
    "rm -rf $SOMEDIR",
    "find /tmp -delete",
    "find /dev/shm -name 'x' -exec rm {} +",
    "mv /tmp/x /tmp/y",
    "rm -rf ~",
    # -- owner-controlled files + sibling trees (BRIEF 2, Phase 0-2; 2-7)
    "sed -i s/a/b/ .claude/settings.json",
    "echo x > .claude/hooks/tray_guard.py",
    "echo x >> scripts/tray_env.sh",
    "echo x | tee scripts/tray_env.sh",
    "cp foo .claude/settings.json",
    "chmod +x .claude/hooks/tray_guard.sh",
    "rm .claude/settings.json",
    "cat <<EOF > .claude/settings.json\n{}\nEOF",
    "mv a ~/experiments/tt-metal/x",
    "sed -i 's/a/b/' ~/experiments/tt-metal/foo.py",
    "echo hi > ~/experiments/pi05/x.txt",
    "cd ~/experiments/tt-metal && ./build_metal.sh",
    "~/experiments/tt-metal/build_metal.sh",
    "git -C ~/experiments/tt-metal checkout main",
    "cd ~/experiments/tt-metal && git pull",
    "cd ~/experiments/tt-metal && cmake --build build",
    "bash ~/experiments/tt-metal/create_venv.sh",
    "~/experiments/tt-metal/build/test/tt_metal/unit_tests",
    # -- commands hidden behind find -exec / launchers / git path operations
    "find . -exec tt-smi {} \\;",
    "find . -name x -execdir sudo ls \\;",
    "find /tmp/claude-1000 -exec pkill python {} +",
    "ssh localhost tt-smi -r",
    "tmux new -d 'tt-smi -r'",
    "screen -dmS x tt-smi",
    "git checkout -- .claude/settings.json",
    "git restore scripts/tray_env.sh",
    "git checkout main -- .claude",
    # -- packages only in PYTHON_ENV (BRIEF 2-7)
    "pip install foo",
    "pip3 install --user x",
    "python3 -m pip install x",
    ".venv/bin/python -m pip install --target /tmp/x foo",
    ".venv/bin/pip install -e ~/experiments/tt-metal",
    "python -m venv /tmp/v",
    "uv pip install x",
    "conda install x",
    # -- git (BRIEF 2-10)
    "git push upstream main",
    "git push upstream",
    "git push",
    "git push origin main --force",
    "git push -f origin main",
    "git push -uf origin x",
    "git push origin +main",
    "git push origin :main",
    "git push --force-with-lease origin main",
    "git push origin --delete foo",
    "git push https://github.com/leggedrobotics/rsl_rl main",
    "git push --mirror origin",
    "git remote add foo https://example.com/x.git",
    "git remote set-url origin https://example.com/x.git",
    "git branch -D foo",
    "git branch -d foo",
    "git tag -d v5.5.2",
    "git rebase -i HEAD~2",
    "git filter-branch --all",
    "git config remote.origin.url https://example.com/x.git",
    "gh repo edit --visibility private",
    "gh repo delete changh95/tt_rsl_rl --yes",
    "gh api -X DELETE repos/changh95/tt_rsl_rl",
    "gh api repos/changh95/tt_rsl_rl -f description=x",
    "gh pr create --repo leggedrobotics/rsl_rl",
    "gh auth token",
    # -- device tests only through tt_run.sh (BRIEF 2-3)
    "pytest -m tt tests/tt",
    "python -m pytest -m tt tests/tt",
    "pytest -mtt tests/tt",
    'pytest -m "tt or smoke" tests/tt',
    "pytest -o addopts= tests/tt",
    'python -c "import ttnn; ttnn.open_device(device_id=0)"',
    "python - <<EOF\nimport ttnn\nttnn.open_mesh_device(ttnn.MeshShape(1, 1))\nEOF",
]

ALLOWED = [
    # -- reading sources that merely mention forbidden things must never be blocked
    "ls",
    "git status",
    "git log --oneline -5",
    "git diff v5.5.2 -- rsl_rl/",
    "grep -rn tt-smi docs/",
    'rg "tt-smi" .',
    "cat docs/porting/BRIEF.md",
    'grep -n "tt-smi -r" CLAUDE.md',
    'grep -rn "/dev/tenstorrent" rsl_rl_tt/',
    'rg -n "pkill|killall" scripts',
    "sed -n '1,5p' .claude/settings.json",
    "cat .claude/hooks/tray_guard.py",
    'echo "do not run tt-smi"',
    "printf 'sudo\\n'",
    'git commit -m "docs: mention that tt-smi is banned"',
    "git commit -m \"$(cat <<'EOF'\nAdd hook that blocks tt-smi\n\nIt's the safety net; \"quoted\" text (with parens) is fine.\nEOF\n)\"",
    "cat > docs/x.md <<'EOF'\nnever run tt-smi or sudo\nEOF",
    "grep -c tt-smi README.md # comment mentioning sudo",
    "ls ~/experiments/tt-metal",
    "grep -rn TT_VISIBLE_DEVICES ~/experiments/tt-metal --include=*.py -l",
    "cat ~/experiments/tt-metal/README.md",
    "git -C ~/experiments/tt-metal log -1",
    "git -C ~/experiments/tt-metal describe --tags",
    "cd ~/experiments/tt-metal && git log -1",
    "find ~/experiments/tt-metal -name '*.so'",
    "command -v tt-smi",
    "which tt-smi",
    "kill -l",
    "echo x > /dev/null",
    "ls 2>&1 | head",
    "for f in a b; do echo $f; done",
    "if [ -f x ]; then echo y; fi",
    "timeout 10 ls",
    "cd rsl_rl_tt && ls",
    'python -c "print(1)"',
    "cat .claude/settings.json > /tmp/claude-1000/out.json",
    "find . -name '*.py' -exec grep -l tt-smi {} +",
    "find rsl_rl_tt -name '*.pyc' -delete",
    "git checkout -- docs/porting/PROGRESS.md",
    "git add .claude scripts && git commit -m 'update safety net'",
    "git diff -- .claude/settings.json",
    "git stash list",
    # -- the 8 allowed nodes
    "stat /dev/tenstorrent/16",
    "ls -l /dev/tenstorrent/16 /dev/tenstorrent/23",
    "for n in /dev/tenstorrent/16 /dev/tenstorrent/17; do stat $n; done",
    # -- git on origin
    "git push origin tt/phase-0",
    "git push -u origin tt/phase-0",
    "git push origin main --tags",
    "git push origin HEAD:refs/heads/tt/x",
    "git fetch upstream",
    "git fetch upstream --tags",
    "git log upstream/main..HEAD",
    "git branch -a",
    "git branch --list 'tt/*'",
    "git branch --show-current",
    "git tag -l",
    "git tag phase-0-done",
    "git tag -a phase-0-done -m 'done'",
    "git config user.name",
    "git config --get remote.origin.url",
    "git remote -v",
    "git checkout -b tt/feature",
    "gh repo view changh95/tt_rsl_rl --json name",
    "gh api repos/changh95/tt_rsl_rl",
    "gh api -X GET repos/changh95/tt_rsl_rl",
    "gh pr list",
    "gh auth status",
    # -- device jobs through the wrapper
    "scripts/tt_run.sh --timeout 60 -- python -m pytest -m tt tests/tt/test_x.py",
    "scripts/tt_run.sh -- python scripts/foo.py",
    "bash scripts/tt_run.sh --tag smoke -- python scripts/smoke.py",
    # -- host tests / packaging inside PYTHON_ENV / cleanup inside the repo or the scratchpad
    "pytest tests/tt -q",
    '.venv/bin/python -m pytest -m "not tt" tests/tt',
    "python -m pytest tests/tt/test_tray_guard.py",
    ".venv/bin/python -m pip install pytest",
    ".venv/bin/pip install -e .",
    "source .venv/bin/activate && pip install numpy",
    "VIRTUAL_ENV=.venv pip --version",
    "python3 -m venv .venv",
    "rm -rf .venv",
    "rm -f .run/pids",
    "rm -rf /tmp/claude-1000/foo/bar",
    "mkdir -p /tmp/claude-1000/x",
    "sed -i 's/a/b/' rsl_rl_tt/device.py",
    "echo hi > docs/porting/x.md",
    "cp a.txt docs/porting/",
]


@pytest.mark.parametrize("cmd", BLOCKED, ids=lambda c: c.replace("\n", "\\n")[:70])
def test_blocked(cmd):
    assert verdict(cmd), f"expected BLOCK but was allowed: {cmd!r}"


@pytest.mark.parametrize("cmd", ALLOWED, ids=lambda c: c.replace("\n", "\\n")[:70])
def test_allowed(cmd):
    assert not verdict(cmd), f"expected ALLOW but was blocked: {cmd!r} -> {verdict(cmd)}"


def test_kill_allowed_only_for_recorded_live_pids(tmp_path, monkeypatch):
    proc = subprocess.Popen(["sleep", "30"])
    try:
        stat = Path(f"/proc/{proc.pid}/stat").read_text()
        start = stat.rsplit(") ", 1)[1].split()[19]
        pids = tmp_path / "pids"
        monkeypatch.setattr(guard, "PIDS_FILE", pids)
        assert verdict(f"kill {proc.pid}")  # not recorded yet
        pids.write_text(f"{proc.pid} {start} job python x.py\n")
        assert not verdict(f"kill {proc.pid}")
        assert not verdict(f"kill -INT {proc.pid}")
        assert verdict(f"kill {proc.pid} 1")
        pids.write_text(f"{proc.pid} 123456789 job python x.py\n")  # stale start time (PID reuse)
        assert verdict(f"kill {proc.pid}")
    finally:
        proc.kill()
        proc.wait()
        time.sleep(0)


def test_cd_tracking_resolves_relative_paths():
    assert verdict("cd ~/experiments/tt-metal && rm -rf build")
    assert verdict("cd .claude && rm settings.json")
    assert not verdict("cd docs && rm -f porting/x.md")


def _run_hook(command: str, tool: str = "Bash") -> subprocess.CompletedProcess:
    payload = json.dumps({"tool_name": tool, "tool_input": {"command": command}, "cwd": str(REPO)})
    return subprocess.run(["bash", str(HOOK_SH)], input=payload, capture_output=True, text=True, timeout=30)


def test_hook_entrypoint_exit_codes():
    ok = _run_hook("git status")
    assert ok.returncode == 0, ok.stderr
    bad = _run_hook("tt-smi -r")
    assert bad.returncode == 2 and "BLOCKED" in bad.stderr
    assert _run_hook("tt-smi -r", tool="Read").returncode == 0  # only Bash is guarded
    assert _run_hook("").returncode == 0


def test_hook_fails_closed_on_garbage_input():
    res = subprocess.run(["bash", str(HOOK_SH)], input="not json", capture_output=True, text=True, timeout=30)
    assert res.returncode == 2


def test_hook_scripts_are_executable():
    assert os.access(HOOK_SH, os.X_OK)
    assert sys.version_info >= (3, 9)
