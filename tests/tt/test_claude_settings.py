# SPDX-License-Identifier: BSD-3-Clause
"""Host-only: the project's Claude Code settings register the guard hook and lock the owner-controlled files (BRIEF Phase 0-2)."""

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETTINGS = REPO / ".claude" / "settings.json"


def _settings() -> dict:
    return json.loads(SETTINGS.read_text())


def test_bash_pretooluse_hook_points_at_the_guard():
    entries = _settings()["hooks"]["PreToolUse"]
    bash = [e for e in entries if e.get("matcher") == "Bash"]
    assert len(bash) == 1
    commands = [h["command"] for h in bash[0]["hooks"] if h.get("type") == "command"]
    assert any(c.endswith(".claude/hooks/tray_guard.sh") or c.endswith('.claude/hooks/tray_guard.sh"') for c in commands), commands
    assert os.access(REPO / ".claude" / "hooks" / "tray_guard.sh", os.X_OK)


def test_owner_controlled_files_are_denied_for_every_edit_tool():
    deny = set(_settings()["permissions"]["deny"])
    for tool in ("Edit", "Write"):
        assert any(rule.startswith(f"{tool}(") and ".claude/**" in rule for rule in deny), f"{tool} on .claude/** not denied: {sorted(deny)}"
        assert any(rule.startswith(f"{tool}(") and "scripts/tray_env.sh" in rule for rule in deny), f"{tool} on tray_env.sh not denied"
    assert any(rule.startswith("NotebookEdit(") and ".claude/**" in rule for rule in deny)


def test_deny_rules_cover_project_root_relative_and_absolute_forms():
    deny = " ".join(_settings()["permissions"]["deny"])
    assert f"//{REPO.as_posix().lstrip('/')}/.claude/**" in deny  # absolute form, independent of cwd
    assert "(/.claude/**)" in deny  # project-root-relative form
