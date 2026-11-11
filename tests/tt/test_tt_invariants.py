# SPDX-License-Identifier: BSD-3-Clause
"""Host-only invariants of the TT path (BRIEF 2-8: no silent host fallbacks; the allow-list is the one in DECISIONS.md)."""

import re
from pathlib import Path

from rsl_rl_tt.tt_common import HOST_OPS

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "rsl_rl_tt"


def _code_lines(path: Path):
    """(line number, code) of a module with docstrings and comments removed (so prose mentioning an API is not an offence)."""
    import ast
    import io
    import tokenize

    text = path.read_text()
    skip = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
                skip.update(range(first.lineno, first.end_lineno + 1))
    out = []
    for no, line in enumerate(text.splitlines(), 1):
        if no in skip:
            continue
        try:
            toks = [t for t in tokenize.generate_tokens(io.StringIO(line).readline) if t.type != tokenize.COMMENT]
            code = tokenize.untokenize(toks).strip() if toks else ""
        except (tokenize.TokenError, IndentationError):
            code = line.split("#")[0]
        out.append((no, code))
    return out


def _tt_path_files():
    """Every module of the TT path: all of rsl_rl_tt except the host-only harness (testing/) and the guarded device entry point."""
    return [p for p in PKG.rglob("*.py") if "testing" not in p.parts and p.name not in {"device.py", "version_pin.py"}]


def test_host_transfers_only_through_ttcontext():
    """`ttnn.from_torch` / `ttnn.to_torch` / `copy_host_to_device_tensor` may appear only in tt_common.py, where every call is counted."""
    pattern = re.compile(r"\b(from_torch|to_torch|copy_host_to_device_tensor|copy_device_to_host_tensor)\b")
    offenders = []
    for path in _tt_path_files():
        if path.name == "tt_common.py":
            continue
        for no, line in _code_lines(path):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{no}: {line.strip()}")
    assert not offenders, "host<->device transfers must go through TTContext.upload/download (counted, allow-listed):\n" + "\n".join(offenders)


def test_device_model_code_contains_no_torch():
    """modules.py is the device-side model code: no torch import, hence no torch compute that could be a silent host fallback."""
    code = "\n".join(line for _, line in _code_lines(PKG / "modules.py"))
    assert not re.search(r"^\s*(import torch|from torch)", code, re.M)
    assert "torch." not in code


def test_every_transfer_names_an_allow_list_id():
    """Every `op="H-n"` literal in the TT path exists in HOST_OPS, and HOST_OPS equals the table in docs/porting/DECISIONS.md."""
    used = set()
    for path in _tt_path_files():
        used |= set(re.findall(r'op="(H-\d+)"', path.read_text()))
        used |= set(re.findall(r'host_ops\.count\("(H-\d+)"', path.read_text()))
    assert used and used <= set(HOST_OPS), sorted(used - set(HOST_OPS))
    doc = (REPO / "docs" / "porting" / "DECISIONS.md").read_text()
    doc_ids = set(re.findall(r"^\| (H-\d+) \|", doc, re.M))
    assert doc_ids == set(HOST_OPS), f"DECISIONS.md lists {sorted(doc_ids)}, code lists {sorted(HOST_OPS)}"


def test_ttppo_is_importable_without_a_device():
    """Importing the plugin must not open anything or import ttnn (ttnn is imported lazily when a context is created)."""
    import sys

    import rsl_rl_tt.ppo  # noqa: F401

    assert "ttnn" not in sys.modules
