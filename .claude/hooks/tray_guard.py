#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Bash PreToolUse guard for the tt_rsl_rl port (BRIEF section 2: tray isolation + shared-server hygiene).

Protocol: Claude Code pipes the hook JSON on stdin. Exit code 2 (+ message on stderr) blocks the command.
This is an AUXILIARY safety net: the rules in CLAUDE.md stay binding even where this does not catch something.

Matching is by COMMAND POSITION (a small shell tokenizer: quotes, heredocs, $(...), backticks, ; && || |, wrappers such as
env/timeout/xargs/bash -c), not by substring, so reading sources (grep/cat/rg/git log ... mentioning `tt-smi`) is allowed.
Owner-controlled file: do not edit without approval (also enforced by permissions.deny and by this guard).
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOME = Path(os.path.expanduser("~"))
EXPERIMENTS = REPO.parent  # sibling trees (tt-metal builds, other people's experiments) are read-only for us
SCRATCH = Path(f"/tmp/claude-{os.getuid()}")
PIDS_FILE = REPO / ".run" / "pids"
VENV = REPO / ".venv"
ALLOWED_NODES = {f"/dev/tenstorrent/{i}" for i in range(16, 24)}  # TRAY_ID=3
MAX_DEPTH = 6


# ====================================================================================================================
# tokenizer
# ====================================================================================================================
HEREDOC_RE = re.compile(r"<<(-?)[ \t]*(?:'([^']*)'|\"([^\"]*)\"|\\?([A-Za-z0-9_]+))")
ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=")
SEPARATORS = {";", ";;", "&&", "||", "|", "|&", "&", "(", ")"}
REDIRECTS = {">", ">>", "<", "&>", "&>>", ">&", ">|", "<&", "<<<"}
GLOB_CHARS = "*?[{"


def _heredoc_delim(m: re.Match):
    return m.group(2) or m.group(3) or m.group(4), m.group(1) == "-", bool(m.group(2) or m.group(3))


def _skip_heredoc_body(s: str, i: int, delim: str, strip: bool):
    """i points just after the newline that ends the heredoc's start line. Returns (body, index after terminator)."""
    body = []
    while i < len(s):
        j = s.find("\n", i)
        line = s[i:] if j < 0 else s[i:j]
        i = len(s) if j < 0 else j + 1
        if (line.lstrip("\t") if strip else line) == delim:
            return "\n".join(body), i
        body.append(line)
    return "\n".join(body), i


def _take_paren(s: str, i: int):
    """s[i:] follows an opening '(' ; return (inner, index after the matching ')'). Heredoc bodies are skipped."""
    depth, j = 1, i
    pending = []
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "'":
            k = s.find("'", j + 1)
            if k < 0:
                raise ValueError("unterminated '")
            j = k + 1
            continue
        if c == '"':
            j += 1
            while j < len(s) and s[j] != '"':
                if s[j] == "\\":
                    j += 2
                elif s.startswith("$(", j):
                    _, j = _take_paren(s, j + 2)
                else:
                    j += 1
            j += 1
            continue
        if c == "<" and s.startswith("<<", j) and not s.startswith("<<<", j):
            m = HEREDOC_RE.match(s, j)
            if m:
                pending.append(_heredoc_delim(m)[:2])
                j = m.end()
                continue
        if c == "\n" and pending:
            j += 1
            for delim, strip in pending:
                _, j = _skip_heredoc_body(s, j, delim, strip)
            pending = []
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return s[i:j], j + 1
        j += 1
    raise ValueError("unterminated $(")


def _take_backtick(s: str, i: int):
    j = i
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == "`":
            return s[i:j], j + 1
        j += 1
    raise ValueError("unterminated backtick")


def preprocess(command: str):
    """Newlines -> ';', drop comments, cut heredoc bodies. Returns (code, [(owner, body, quoted_delim)])."""
    out: list[str] = []
    heredocs = []
    pending = []
    i, n = 0, len(command)
    sq = dq = False

    def last_char() -> str:
        return out[-1][-1] if out and out[-1] else ""

    while i < n:
        c = command[i]
        if sq:
            out.append(c)
            sq = c != "'"
            i += 1
            continue
        if dq:
            if c == "\\" and i + 1 < n:
                out.append(command[i : i + 2])
                i += 2
            elif c == "$" and command.startswith("$(", i):
                _, j = _take_paren(command, i + 2)
                out.append(command[i:j])
                i = j
            else:
                dq = c != '"'
                out.append(c)
                i += 1
            continue
        if c == "\\":
            if i + 1 < n and command[i + 1] == "\n":
                i += 2
            else:
                out.append(command[i : i + 2])
                i += 2
            continue
        if c == "$" and command.startswith("$(", i):
            _, j = _take_paren(command, i + 2)
            out.append(command[i:j])
            i = j
            continue
        if c == "'":
            sq = True
        elif c == '"':
            dq = True
        elif c == "#" and (not out or last_char() in ("", " ", "\t", ";", "&", "|", "(", "\n")):
            j = command.find("\n", i)
            i = n if j < 0 else j
            continue
        elif c == "\n":
            out.append(" ; ")
            i += 1
            for delim, strip, quoted, owner in pending:
                body, i = _skip_heredoc_body(command, i, delim, strip)
                heredocs.append((owner, body, quoted))
            pending = []
            continue
        elif c == "<" and command.startswith("<<", i) and not command.startswith("<<<", i):
            m = HEREDOC_RE.match(command, i)
            if m:
                delim, strip, quoted = _heredoc_delim(m)
                owner = re.split(r"[;&|(]", "".join(out))[-1].strip()
                pending.append((delim, strip, quoted, owner))
                out.append(" ")
                i = m.end()
                continue
        out.append(c)
        i += 1
    for _delim, _strip, quoted, owner in pending:  # heredoc started on the last line, no body present
        heredocs.append((owner, "", quoted))
    return "".join(out), heredocs


def tokenize(code: str):
    """Return (tokens, substitutions); tokens are ('w', text) or ('op', text)."""
    toks: list[tuple[str, str]] = []
    subs: list[str] = []
    buf: list[str] = []
    has = False
    i, n = 0, len(code)

    def flush():
        nonlocal buf, has
        if has:
            toks.append(("w", "".join(buf)))
        buf, has = [], False

    while i < n:
        c = code[i]
        if c in " \t":
            flush()
            i += 1
        elif c == "\\":
            if i + 1 < n:
                buf.append(code[i + 1])
                has = True
            i += 2
        elif c == "'":
            j = code.find("'", i + 1)
            if j < 0:
                raise ValueError("unterminated '")
            buf.append(code[i + 1 : j])
            has = True
            i = j + 1
        elif c == '"':
            has = True
            i += 1
            while True:
                if i >= n:
                    raise ValueError('unterminated "')
                d = code[i]
                if d == '"':
                    i += 1
                    break
                if d == "\\" and i + 1 < n:
                    nxt = code[i + 1]
                    buf.append(nxt if nxt in '"\\$`' else d + nxt)
                    i += 2
                elif d == "$" and code.startswith("$(", i):
                    inner, i = _take_paren(code, i + 2)
                    subs.append(inner)
                    buf.append("$SUBST")
                elif d == "`":
                    inner, i = _take_backtick(code, i + 1)
                    subs.append(inner)
                    buf.append("$SUBST")
                else:
                    buf.append(d)
                    i += 1
        elif c == "$" and code.startswith("$(", i):
            inner, i = _take_paren(code, i + 2)
            subs.append(inner)
            buf.append("$SUBST")
            has = True
        elif c == "`":
            inner, i = _take_backtick(code, i + 1)
            subs.append(inner)
            buf.append("$SUBST")
            has = True
        elif c in ";&|()<>":
            op = next((o for o in ("&>>", "<<<", "&&", "||", "|&", ">>", "&>", ">&", ">|", "<&", ";;") if code.startswith(o, i)), c)
            if op in ("<", ">") and code.startswith("(", i + 1):  # process substitution <(cmd) / >(cmd)
                flush()
                i += 1
                continue
            if (op[0] in "<>" or op.startswith("&>")) and has and "".join(buf).isdigit():
                buf, has = [], False  # fd prefix such as the 2 in 2>
            flush()
            toks.append(("op", op))
            i += len(op)
        else:
            buf.append(c)
            has = True
            i += 1
    flush()
    return toks, subs


def split_segments(toks):
    """Yield (words, redirects) per simple command."""
    words: list[str] = []
    redirs: list[tuple[str, str]] = []
    i = 0
    while i < len(toks):
        kind, text = toks[i]
        if kind == "op" and text in SEPARATORS:
            if words or redirs:
                yield words, redirs
            words, redirs = [], []
        elif kind == "op" and text in REDIRECTS:
            if i + 1 < len(toks) and toks[i + 1][0] == "w":
                if text != "<<<":
                    redirs.append((text, toks[i + 1][1]))
                i += 1
        elif kind == "w":
            words.append(text)
        i += 1
    if words or redirs:
        yield words, redirs


# ====================================================================================================================
# context + path helpers
# ====================================================================================================================
@dataclass
class Ctx:
    cwd: str | None
    reasons: list[str] = field(default_factory=list)  # shared by all copies made with dataclasses.replace
    venv_active: bool = False
    under_tt_run: bool = False
    via_xargs: bool = False
    depth: int = 0

    def block(self, why: str) -> None:
        if why not in self.reasons:
            self.reasons.append(why)


def under(p: str, base) -> bool:
    base = str(base).rstrip("/")
    return p == base or p.startswith(base + "/")


def expand(arg: str, ctx: Ctx) -> str | None:
    a = arg
    if a == "~" or a.startswith("~/"):
        a = str(HOME) + a[1:]
    a = a.replace("${HOME}", str(HOME)).replace("$HOME", str(HOME))
    if ctx.cwd:
        a = a.replace("${PWD}", ctx.cwd).replace("$PWD", ctx.cwd)
    return None if ("$" in a or "`" in a) else a


def norm(arg: str, ctx: Ctx, resolve: bool = True):
    """(absolute normalized path | None if not statically resolvable, has_glob)."""
    a = expand(arg, ctx)
    if a is None:
        return None, False
    glob = any(ch in a for ch in GLOB_CHARS)
    if not a.startswith("/"):
        if ctx.cwd is None:
            return None, glob
        a = os.path.join(ctx.cwd, a)
    a = os.path.normpath(a)
    return (os.path.realpath(a) if resolve and not glob else a), glob


def literal_prefix(p: str, glob: bool) -> str:
    return re.split(r"[*?\[{]", p, maxsplit=1)[0] if glob else p


def hits(p: str, glob: bool, base) -> bool:
    """Does path p (possibly a glob) lie under / possibly expand into base?"""
    base = str(base).rstrip("/")
    if not glob:
        return under(p, base)
    pre = literal_prefix(p, True)
    return base.startswith(pre) or pre.startswith(base + "/")


def protected_write_reason(p: str, glob: bool):
    if hits(p, glob, REPO / ".claude"):
        return "`.claude/**` is owner-controlled (BRIEF 2, Phase 0-2)"
    if hits(p, glob, REPO / "scripts" / "tray_env.sh"):
        return "`scripts/tray_env.sh` is owner-controlled (BRIEF 2, Phase 0-2)"
    if hits(p, glob, EXPERIMENTS) and not under(literal_prefix(p, glob), REPO):
        return "trees next to this repo (tt-metal builds, other experiments) must not be modified (BRIEF 2-5, 2-7)"
    if re.match(r"^/(proc|sys)(/|$)", p):
        return "writing under /proc or /sys can change hugepages/drivers (BRIEF 2-4)"
    if hits(p, glob, "/dev/tenstorrent"):
        return "device nodes are only opened via rsl_rl_tt/device.py (BRIEF 2-2)"
    return None


def shared_delete_reason(p: str, glob: bool):
    if under(p, SCRATCH):
        return None
    if p == "/tmp" or p.startswith("/tmp/") or hits(p, glob, "/dev/shm"):
        return "other people's files may live in /tmp or /dev/shm; only delete inside your own scratchpad (BRIEF 2-5)"
    if hits(p, glob, HOME / ".cache") and re.search(r"tt|metal|ttnn|kernel|triton", p, re.I):
        return "shared tt-metal/ttnn caches must not be deleted (BRIEF 2-5)"
    if p in ("/", "/home", str(HOME)) or (glob and literal_prefix(p, glob).rstrip("/") in ("", "/home", str(HOME))):
        return "refusing to delete / or the home directory"
    return None


def check_path_write(arg: str, kind: str, ctx: Ctx, via: str) -> None:
    p, glob = norm(arg, ctx)
    if p is None:
        if kind == "delete":
            ctx.block(f"cannot verify the target of `{via}` ('{arg}': unresolved variable/cwd); use a literal path (BRIEF 2-5)")
        return
    why = protected_write_reason(p, glob) or (shared_delete_reason(p, glob) if kind == "delete" else None)
    if why:
        ctx.block(f"`{via}` on '{arg}': {why}")


def my_pids() -> set[int]:
    """Live PIDs recorded by scripts/tt_run.sh (verified by /proc start time to defeat PID reuse)."""
    out = set()
    try:
        for line in PIDS_FILE.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    stat = Path(f"/proc/{parts[0]}/stat").read_text()
                    if stat.rsplit(") ", 1)[1].split()[19] == parts[1]:
                        out.add(int(parts[0]))
                except (OSError, IndexError):
                    pass
    except OSError:
        pass
    return out


# ====================================================================================================================
# rule tables
# ====================================================================================================================
HW_TOOL_RE = re.compile(r"^tt[-_](smi|flash|topology|burnin|triage|kmd|installer|tools|diag|health|reset|status|update)", re.I)
DIAG_NAME_RE = re.compile(
    r"(health[_-]?check|hw[_-]?diag|glx[_-]?reset|board[_-]?reset|chip[_-]?reset|device[_-]?reset|tray[_-]?reset|galaxy[_-]?reset"
    r"|reset[_-]?(board|chip|device|tray|galaxy|glx))",
    re.I,
)
FORBIDDEN_TOKENS_RE = re.compile(r"(?<![\w.-])(tt[-_]smi|tt[-_]flash|tt[-_]topology|ipmitool|pkill|killall|glx_reset)(?![\w-])", re.I)
HARD_BLOCK = {
    "ipmitool": "BRIEF 2-4: ipmitool is forbidden",
    "sudo": "BRIEF 2-4: sudo is forbidden",
    "su": "BRIEF 2-4: su is forbidden",
    "doas": "BRIEF 2-4: privilege escalation is forbidden",
    "pkexec": "BRIEF 2-4: privilege escalation is forbidden",
    "reboot": "BRIEF 2-4: reboot is forbidden",
    "shutdown": "BRIEF 2-4: shutdown is forbidden",
    "poweroff": "BRIEF 2-4: poweroff is forbidden",
    "halt": "BRIEF 2-4: halt is forbidden",
    "telinit": "BRIEF 2-4: runlevel changes are forbidden",
    "kexec": "BRIEF 2-4: kexec is forbidden",
    "modprobe": "BRIEF 2-4: driver changes are forbidden",
    "rmmod": "BRIEF 2-4: driver changes are forbidden",
    "insmod": "BRIEF 2-4: driver changes are forbidden",
    "dkms": "BRIEF 2-4: driver changes are forbidden",
    "mount": "BRIEF 2-4: mount/hugepages changes are forbidden",
    "umount": "BRIEF 2-4: mount/hugepages changes are forbidden",
    "hugeadm": "BRIEF 2-4: hugepages changes are forbidden",
    "pkill": "BRIEF 2-5: name-based pkill is forbidden; use scripts/kill_my_jobs.sh",
    "killall": "BRIEF 2-5: name-based killall is forbidden; use scripts/kill_my_jobs.sh",
    "killall5": "BRIEF 2-5: killall is forbidden",
    "skill": "BRIEF 2-5: name-based skill is forbidden",
    "ttrl": "BRIEF 2-4: tt-rl's CLI (auto device picking) must never be run",
}
SERVICE_MUTATING = {"reboot", "poweroff", "halt", "kexec", "restart", "stop", "start", "reload", "disable", "enable", "mask", "unmask",
                    "isolate", "daemon-reload", "kill", "suspend", "hibernate"}
SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
# name -> (options that take an argument, number of positional args before the wrapped command)
WRAPPERS = {
    "env": ({"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}, 0),
    "nohup": (set(), 0),
    "setsid": (set(), 0),
    "nice": ({"-n", "--adjustment"}, 0),
    "ionice": ({"-c", "-n", "-p", "-P", "--class", "--classdata"}, 0),
    "time": ({"-f", "-o", "--format", "--output"}, 0),
    "timeout": ({"-s", "--signal", "-k", "--kill-after"}, 1),
    "exec": ({"-a"}, 0),
    "stdbuf": ({"-i", "-o", "-e", "--input", "--output", "--error"}, 0),
    "command": (set(), 0),
    "builtin": (set(), 0),
    "unbuffer": (set(), 0),
    "watch": ({"-n", "--interval"}, 0),
    "xargs": ({"-a", "-d", "-E", "-I", "-L", "-n", "-P", "-s", "--arg-file", "--delimiter", "--max-args", "--max-procs", "--max-lines",
               "--max-chars"}, 0),
    "taskset": ({"-c", "--cpu-list"}, 1),
    "flock": ({"-w", "--timeout", "-E", "--conflict-exit-code"}, 1),
    "chrt": (set(), 1),
}
WRITE_ALL = {"tee", "touch", "chmod", "chown", "chgrp", "mkdir", "patch", "sponge"}
DELETE_ALL = {"rm", "rmdir", "unlink", "shred", "truncate", "mv"}
WRITE_DEST_LAST = {"cp", "install", "rsync", "scp", "ln"}
SEARCH_CMDS = {"grep", "egrep", "fgrep", "rg", "ag", "ack", "sed", "awk", "gawk"}  # first positional is a pattern/script
GIT_MUTATING = {"add", "commit", "checkout", "switch", "restore", "reset", "clean", "stash", "pull", "fetch", "merge", "rebase", "cherry-pick",
                "revert", "apply", "am", "rm", "mv", "submodule", "worktree", "gc", "tag", "branch", "push", "config", "remote", "update-index",
                "init"}
PYTHON_RE = re.compile(r"python[\d.]*|pypy[\d.]*|ipython[\d.]*")
OPAQUE_LAUNCHERS = {"ssh", "sshpass", "tmux", "screen", "at", "batch", "nsenter", "chroot", "docker", "podman", "script", "parallel", "crontab"}
GIT_PATH_MUTATORS = {"checkout", "restore", "rm", "mv", "clean", "apply", "reset", "stash"}


def positional(args):
    return [a for a in args if not a.startswith("-") or a == "-"]


def option_values(args):
    """Candidate paths hidden in `--opt=value` / dd-style `of=value` args."""
    vals = []
    for a in args:
        if a.startswith("--") and "=" in a:
            vals.append(a.split("=", 1)[1])
        elif re.match(r"^(of|if)=", a):
            vals.append(a.split("=", 1)[1])
    return vals


def is_python_code_arg(args, i) -> bool:
    return i > 0 and args[i - 1] == "-c"


# ====================================================================================================================
# analysis
# ====================================================================================================================
def analyze(command: str, ctx: Ctx) -> None:
    if ctx.depth > MAX_DEPTH:
        ctx.block("command nesting too deep to verify")
        return
    try:
        code, heredocs = preprocess(command)
        toks, subs = tokenize(code)
    except ValueError:
        coarse_scan(command, ctx, "unparsable command")
        return
    child = replace(ctx, depth=ctx.depth + 1)
    for owner, body, quoted in heredocs:
        words = [w for w in re.split(r"\s+", owner) if w]
        owner_name = Path(words[0]).name if words else ""
        if owner_name in SHELLS or owner_name == "source":
            analyze(body, child)
        elif PYTHON_RE.fullmatch(owner_name):
            scan_python_code(body, ctx)
        elif not quoted:  # unquoted heredoc bodies still undergo $(...) / `...` expansion
            for m in re.finditer(r"\$\(([^()]*)\)|`([^`]*)`", body):
                analyze(m.group(1) or m.group(2) or "", child)
    for s in subs:
        analyze(s, child)
    for words, redirs in split_segments(toks):
        segment(words, redirs, ctx)


def coarse_scan(text: str, ctx: Ctx, what: str) -> None:
    m = FORBIDDEN_TOKENS_RE.search(text)
    if m:
        ctx.block(f"{what} mentions forbidden tool '{m.group(1)}' (BRIEF 2-4/2-5) and cannot be verified to be mere text")


def scan_python_code(code: str, ctx: Ctx) -> None:
    coarse_scan(code, ctx, "python code")
    if re.search(r"open_(mesh_)?device|MeshDevice\(|open_tray|ttnn\.open", code) and not ctx.under_tt_run:
        ctx.block("BRIEF 2-2/2-3: opening devices from ad-hoc python is forbidden; use rsl_rl_tt/device.py via scripts/tt_run.sh")


def segment(words, redirs, ctx: Ctx) -> None:
    for op, tgt in redirs:
        if op in (">", ">>", "&>", "&>>", ">|") or (op == ">&" and not re.fullmatch(r"\d+|-", tgt)):
            check_path_write(tgt, "write", ctx, "redirect >")
        check_device_args("redirect", [tgt], ctx)
    local = replace(ctx)
    i = 0
    while i < len(words):
        w = words[i]
        if ASSIGN_RE.match(w):
            if w.startswith("VIRTUAL_ENV="):
                p, _ = norm(w.split("=", 1)[1], ctx, resolve=False)
                if p and under(p, VENV):
                    local.venv_active = True
            i += 1
        elif w in ("then", "do", "else", "elif", "if", "while", "until", "!", "{", "}", "time", "function"):
            i += 1
        elif w in ("for", "case", "select"):
            return
        else:
            break
    if i < len(words):
        dispatch(words[i], words[i + 1 :], local, ctx)


def unwrap(name: str, args):
    opts, npos = WRAPPERS[name]
    if name == "taskset" and any(a in ("-c", "--cpu-list") for a in args):
        npos = 0
    i = pos = 0
    while i < len(args):
        a = args[i]
        if name == "env" and ASSIGN_RE.match(a):
            i += 1
        elif a == "--":
            i += 1
            break
        elif a.startswith("-") and len(a) > 1:
            i += 2 if a in opts else 1
        elif pos < npos:
            pos += 1
            i += 1
        else:
            break
    return args[i:]


def parse_tt_run(args):
    """Arguments after tt_run.sh's own options (the wrapped device job)."""
    i = 0
    while i < len(args):
        if args[i] == "--":
            return args[i + 1 :]
        if args[i] in ("--timeout", "--grace", "--wait", "--tag"):
            i += 2
        elif args[i].startswith("-"):
            i += 1
        else:
            break
    return args[i:]


def dispatch(cmd: str, args, ctx: Ctx, persistent: Ctx) -> None:
    exp = expand(cmd, ctx)
    if exp is None:
        ctx.block(f"command name '{cmd}' comes from a variable/substitution and cannot be verified; write it literally")
        return
    name = Path(exp).name
    child = replace(ctx, depth=ctx.depth + 1)

    check_device_args(name, args, ctx)
    check_exec_path(exp, ctx)

    if name == "tt_run.sh":
        inner = parse_tt_run(args)
        if inner:
            dispatch(inner[0], inner[1:], replace(child, under_tt_run=True), persistent)
        return
    if name in WRAPPERS:
        if name == "command" and any(a in ("-v", "-V") for a in args):
            return
        if name == "flock" and any(a in ("-c", "--command") for a in args):
            j = max(i for i, a in enumerate(args) if a in ("-c", "--command"))
            if j + 1 < len(args):
                analyze(args[j + 1], child)
            return
        inner = unwrap(name, args)
        if inner:
            dispatch(inner[0], inner[1:], replace(child, via_xargs=ctx.via_xargs or name == "xargs"), persistent)
        return
    if name in SHELLS:
        run_shell(args, ctx, child, persistent)
        return
    if name == "eval":
        analyze(" ".join(args), child)
        return
    if name in ("source", "."):
        if args:
            p, _ = norm(args[0], ctx, resolve=False)
            if p and Path(p).name == "activate" and under(p, VENV):
                persistent.venv_active = True
            elif p and Path(p).name != "tray_env.sh" and under(p, EXPERIMENTS) and not under(p, REPO):
                ctx.block("sourcing scripts from other trees is forbidden (BRIEF 2-7)")
        return
    if name == "cd":
        target = args[-1] if args else None
        persistent.cwd = norm(target, ctx)[0] if target and target != "-" else None
        return
    check_command(name, exp, args, ctx)


def run_shell(args, ctx, child, persistent) -> None:
    cluster = [i for i, a in enumerate(args) if re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", a)]
    if cluster:
        rest = [a for a in args[cluster[0] + 1 :] if not a.startswith("-")]
        if rest:
            analyze(rest[0], child)
        return
    pos = positional(args)
    if not pos:
        return
    script = Path(expand(pos[0], ctx) or pos[0])
    if script.name == "tt_run.sh":
        inner = parse_tt_run(args[args.index(pos[0]) + 1 :])
        if inner:
            dispatch(inner[0], inner[1:], replace(child, under_tt_run=True), persistent)
        return
    check_exec_path(str(script), ctx, script_arg=True)
    if DIAG_NAME_RE.search(script.name) or HW_TOOL_RE.match(script.name):
        ctx.block(f"BRIEF 2-4: running hardware health/diag/reset script '{script.name}' is forbidden")


def check_exec_path(exp: str, ctx: Ctx, script_arg: bool = False) -> None:
    """Programs/scripts living in sibling trees (tt-metal builds, other experiments) may open ALL chips or modify those trees."""
    if "/" not in exp and not script_arg:
        return
    p, _ = norm(exp, ctx, resolve=False)
    if p and under(p, EXPERIMENTS) and not under(p, REPO) and not PYTHON_RE.fullmatch(Path(p).name):
        ctx.block(f"executing '{exp}' from another tree could open every chip or modify that tree (BRIEF 2-1, 2-7)")


def check_device_args(name: str, args, ctx: Ctx) -> None:
    """Any /dev/tenstorrent path argument must be exactly one of the 8 allowed nodes (patterns of search tools are exempt)."""
    low = name.lower()
    if low in ("echo", "printf", "git", "gh"):
        return
    pos_seen = 0
    for i, a in enumerate(args):
        if PYTHON_RE.fullmatch(low) and is_python_code_arg(args, i):
            continue
        if low in SEARCH_CMDS and not a.startswith("-"):
            pos_seen += 1
            if pos_seen == 1:  # pattern / script, not a file
                continue
        for m in re.findall(r"/dev/tenstorrent[^\s,:;'\"]*", a):
            if m not in ALLOWED_NODES:
                ctx.block(f"'{m}' is not one of the 8 allowed nodes /dev/tenstorrent/16..23 (BRIEF 2-1: other chips: no open/reset/query)")


def check_command(name: str, exp: str, args, ctx: Ctx) -> None:
    low = name.lower()
    pos = positional(args)

    if low in HARD_BLOCK:
        ctx.block(HARD_BLOCK[low])
        return
    if HW_TOOL_RE.match(low) or DIAG_NAME_RE.search(low):
        ctx.block(f"BRIEF 2-4: '{name}' (tt-smi/flash/topology/health/diag/reset family) is forbidden, including read-only queries")
        return
    if any(re.match(r"^-{1,2}glx[_-]?reset", a) for a in args):
        ctx.block("BRIEF 2-4: Galaxy reset flags are forbidden")
    if low == "sysctl" and (any("=" in a for a in args) or "-w" in args):
        ctx.block("BRIEF 2-4: sysctl writes (hugepages etc.) are forbidden")
    if low in ("systemctl", "service") and any(a in SERVICE_MUTATING for a in args):
        ctx.block("BRIEF 2-4: changing services / rebooting is forbidden")
    if low == "init" and any(a in ("0", "6") for a in args):
        ctx.block("BRIEF 2-4: reboot/shutdown is forbidden")
    if low == "fuser" and any(re.match(r"^-[A-Za-z]*k", a) for a in args):
        ctx.block("BRIEF 2-5: `fuser -k` kills by resource; only PIDs in .run/pids may be signalled")
    if low == "kill":
        check_kill(args, ctx)
    if PYTHON_RE.fullmatch(low) or low in ("pytest", "py.test"):
        check_python(low, exp, args, ctx)
    if low in ("pip", "pip3", "uv", "pipx", "conda", "mamba", "poetry") or re.fullmatch(r"pip[\d.]+", low):
        check_pip(low, exp, args, ctx)
    if low in ("make", "ninja", "cmake", "ctest", "scons"):
        cands = option_values(args) + [a for i, a in enumerate(args) if i and args[i - 1] in ("-C", "--build", "-S", "-B", "--directory")]
        for a in cands + ([ctx.cwd] if ctx.cwd else []):
            p, _ = norm(a, ctx)
            if p and under(p, EXPERIMENTS) and not under(p, REPO):
                ctx.block(f"'{name}' in/at '{a}' would modify tt-metal / other experiments (BRIEF 2-7)")
                break
    if low == "git":
        check_git(args, ctx)
    if low == "gh":
        check_gh(args, ctx)
    if low in OPAQUE_LAUNCHERS:  # remote/detached/container launchers: cannot verify, so refuse to carry forbidden tools
        coarse_scan(" ".join(args), ctx, f"`{low}` command line")
    if low == "find":  # commands run by find -exec/-execdir/-ok/-okdir are real commands
        for i, a in enumerate(args):
            if a in ("-exec", "-execdir", "-ok", "-okdir") and i + 1 < len(args):
                end = next((j for j in range(i + 1, len(args)) if args[j] in (";", "+")), len(args))
                inner = args[i + 1 : end]
                if inner:
                    dispatch(inner[0], inner[1:], replace(ctx, depth=ctx.depth + 1), ctx)
    check_fs(low, args, pos, ctx)


def check_kill(args, ctx: Ctx) -> None:
    if any(a in ("-l", "-L", "-ll") for a in args):
        return
    i = 0
    if args and args[0] in ("-s", "-n", "--signal"):
        i = 2
    elif args and args[0].startswith("-") and args[0] != "--":
        i = 1
    if i < len(args) and args[i] == "--":
        i += 1
    pids = args[i:]
    if ctx.via_xargs or not pids:
        ctx.block("BRIEF 2-5: `kill` needs explicit PIDs recorded in .run/pids (xargs/substitution targets cannot be verified)")
        return
    mine = my_pids()
    for p in pids:
        if not re.fullmatch(r"\d+", p) or int(p) <= 1 or int(p) not in mine:
            ctx.block(f"BRIEF 2-5: refusing `kill {p}`: only live PIDs recorded by tt_run.sh in .run/pids may be signalled "
                      "(use scripts/kill_my_jobs.sh)")


def marker_selects_tt(expr: str) -> bool:
    toks = re.findall(r"\w+", expr)
    return any(t == "tt" and (i == 0 or toks[i - 1] != "not") for i, t in enumerate(toks))


def check_python(low: str, exp: str, args, ctx: Ctx) -> None:
    module = code = script = None
    rest = args
    if low in ("pytest", "py.test"):
        module = "pytest"
    else:
        i = 0
        while i < len(args):
            a = args[i]
            if a == "-m" and i + 1 < len(args):
                module, rest = args[i + 1], args[i + 2 :]
                break
            if a == "-c" and i + 1 < len(args):
                code, rest = args[i + 1], args[i + 2 :]
                break
            if a.startswith("-"):
                i += 2 if a in ("-W", "-X") else 1
                continue
            script, rest = a, args[i + 1 :]
            break
    if module == "pytest" or (script and Path(script).name in ("pytest", "py.test")):
        for i, a in enumerate(rest):
            expr = rest[i + 1] if a == "-m" and i + 1 < len(rest) else a[2:] if re.fullmatch(r"-m.+", a) and not a.startswith("--") else None
            if expr is not None and marker_selects_tt(expr) and not ctx.under_tt_run:
                ctx.block("BRIEF 2-3: device tests (`-m tt`) must run via scripts/tt_run.sh -- python -m pytest -m tt ...")
            if "addopts" in a and not ctx.under_tt_run:
                ctx.block("BRIEF 2-3: overriding pytest addopts (which deselects device tests) is only allowed under scripts/tt_run.sh")
    elif module:
        if module.split(".")[0] in ("ttrl", "tt_smi", "tt_flash", "tt_topology", "tt_burnin"):
            ctx.block(f"BRIEF 2-4: `python -m {module}` is forbidden")
        elif module == "pip":
            check_pip("pip", exp, rest, ctx, via_python=exp)
        elif module == "venv":
            for t in positional(rest):
                p, _ = norm(t, ctx, resolve=False)
                if p != str(VENV):
                    ctx.block("BRIEF 2-7: the only venv we create is <repo>/.venv")
    if code is not None:
        scan_python_code(code, ctx)
    if script and (DIAG_NAME_RE.search(Path(script).name) or HW_TOOL_RE.match(Path(script).name) or Path(script).name.startswith("ttrl")):
        ctx.block(f"BRIEF 2-4: running '{script}' (hardware health/diag/reset script or tt-rl CLI) is forbidden")


def check_pip(low: str, exp: str, args, ctx: Ctx, via_python: str | None = None) -> None:
    sub = next((a for a in args if not a.startswith("-")), "")
    if low == "uv" and sub == "venv":
        for x in positional(args[args.index("venv") + 1 :]) or [".venv"]:
            if norm(x, ctx, resolve=False)[0] != str(VENV):
                ctx.block("BRIEF 2-7: the only venv we create is <repo>/.venv")
        return
    if low == "uv" and sub == "pip":
        args = args[args.index("pip") + 1 :]
        sub = next((a for a in args if not a.startswith("-")), "")
    if low in ("conda", "mamba", "pipx", "poetry"):
        if sub in ("install", "remove", "uninstall", "update", "add", "upgrade"):
            ctx.block(f"BRIEF 2-7: '{low} {sub}' would install outside PYTHON_ENV")
        return
    if sub not in ("install", "uninstall"):
        return

    def in_venv(x: str) -> bool:
        p, _ = norm(x, ctx, resolve=False)
        return bool(p and under(p, VENV))

    pyopt = [args[i + 1] for i, a in enumerate(args[:-1]) if a == "--python"] + [a.split("=", 1)[1] for a in args if a.startswith("--python=")]
    if not (ctx.venv_active or in_venv(exp) or (via_python and in_venv(via_python)) or any(in_venv(x) for x in pyopt)):
        ctx.block("BRIEF 2-7: packages may only be installed into PYTHON_ENV (<repo>/.venv): use `.venv/bin/python -m pip ...` "
                  "or activate it in the same command")
    for a in args:
        if a in ("--user", "--break-system-packages", "--target", "--prefix", "--root") or a.startswith(("--target=", "--prefix=", "--root=")):
            ctx.block(f"BRIEF 2-7: pip option '{a}' installs outside PYTHON_ENV")
    for a in args:
        if not a.startswith("-"):
            p, _ = norm(a, ctx)
            if p and under(p, EXPERIMENTS) and not under(p, REPO):
                ctx.block(f"BRIEF 2-7: pip install of '{a}' would write into another tree (egg-info/build); do not")


def check_git(args, ctx: Ctx) -> None:
    gdir = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"):
            if a == "-C" and i + 1 < len(args):
                gdir = args[i + 1]
            i += 2
        elif a.startswith("-"):
            i += 1
        else:
            break
    if i >= len(args):
        return
    sub, rest = args[i], args[i + 1 :]
    cwd = norm(gdir, ctx)[0] if gdir else ctx.cwd
    if sub in GIT_MUTATING and cwd and under(cwd, EXPERIMENTS) and not under(cwd, REPO):
        ctx.block(f"`git {sub}` in {cwd}: trees next to this repo must not be modified (BRIEF 2-7)")
    if sub in GIT_PATH_MUTATORS:  # `git checkout -- .claude/x`, `git restore scripts/tray_env.sh` ... bypass the edit protection
        for a in rest:
            p, glob = norm(a, replace(ctx, cwd=cwd)) if not a.startswith("-") else (None, False)
            if p and (hits(p, glob, REPO / ".claude") or hits(p, glob, REPO / "scripts" / "tray_env.sh")):
                ctx.block(f"`git {sub}` on '{a}': `.claude/**` and `scripts/tray_env.sh` are owner-controlled (BRIEF 2, Phase 0-2)")
    if sub == "push":
        check_git_push(rest, ctx)
    elif sub == "remote" and rest and rest[0] in ("add", "remove", "rm", "rename", "set-url", "set-head", "set-branches", "prune", "update"):
        ctx.block("BRIEF 2-10: remotes are fixed (origin = changh95/tt_rsl_rl, upstream = leggedrobotics/rsl_rl); ask the owner")
    elif sub == "branch" and any(a in ("-d", "-D", "--delete") or re.fullmatch(r"-[a-zA-Z]*[dD][a-zA-Z]*", a) for a in rest):
        ctx.block("BRIEF 2-10: deleting branches is forbidden")
    elif sub == "tag" and any(a in ("-d", "--delete") for a in rest):
        ctx.block("BRIEF 2-10: deleting tags is forbidden")
    elif sub in ("filter-branch", "filter-repo", "rebase", "replace"):
        ctx.block(f"BRIEF 2-10: `git {sub}` rewrites history and is forbidden")
    elif sub == "update-ref" and "-d" in rest:
        ctx.block("BRIEF 2-10: deleting refs is forbidden")
    elif sub == "reflog" and rest and rest[0] in ("expire", "delete"):
        ctx.block("BRIEF 2-10: history rewriting is forbidden")
    elif sub == "config":
        reads = {"--get", "--get-all", "--get-regexp", "--list", "-l", "--show-origin", "--show-scope", "--get-urlmatch", "--name-only"}
        keys = [a for a in rest if not a.startswith("-")]
        if keys and re.match(r"^(remote|url|branch)\.", keys[0]) and not any(a in reads for a in rest):
            ctx.block("BRIEF 2-10: changing remote/url/branch config is forbidden")


def check_git_push(rest, ctx: Ctx) -> None:
    for a in rest:
        if a in ("--mirror", "--delete", "-d", "--prune") or a.startswith("--force"):
            ctx.block(f"BRIEF 2-10: `git push {a}` (force/delete/mirror) is forbidden")
        elif re.fullmatch(r"-[a-zA-Z]*f[a-zA-Z]*", a):
            ctx.block(f"BRIEF 2-10: `git push {a}` (force) is forbidden")
        elif a.startswith("--repo"):
            ctx.block("BRIEF 2-10: `git push --repo` is forbidden; push to origin explicitly")
    pos, skip = [], False
    for a in rest:
        if skip:
            skip = False
        elif a in ("-o", "--push-option", "--receive-pack", "--exec"):
            skip = True
        elif not a.startswith("-"):
            pos.append(a)
    if not pos:
        ctx.block("BRIEF 2-10: bare `git push` is ambiguous; use `git push origin <branch>` (never upstream)")
        return
    remote, refspecs = pos[0], pos[1:]
    if remote != "origin" and "changh95/tt_rsl_rl" not in remote:
        ctx.block(f"BRIEF 2-10: pushing to '{remote}' is forbidden; push only to origin (changh95/tt_rsl_rl), never upstream")
    for r in refspecs:
        if r.startswith("+"):
            ctx.block(f"BRIEF 2-10: refspec '{r}' forces the push (+) and is forbidden")
        elif r.startswith(":"):
            ctx.block(f"BRIEF 2-10: refspec '{r}' deletes a remote ref and is forbidden")


def check_gh(args, ctx: Ctx) -> None:
    pos = [a for a in args if not a.startswith("-")]
    grp = pos[0] if pos else ""
    sub = pos[1] if len(pos) > 1 else ""
    if grp == "repo" and sub in ("edit", "delete", "rename", "archive", "unarchive", "transfer", "create", "fork", "sync", "deploy-key", "set-default"):
        ctx.block(f"BRIEF 2-10: `gh repo {sub}` changes repository settings/state and is forbidden")
    elif grp == "api":
        methods = [args[i + 1].upper() for i, a in enumerate(args[:-1]) if a in ("-X", "--method")]
        methods += [a.split("=", 1)[1].upper() for a in args if a.startswith("--method=")]
        if any(m != "GET" for m in methods) or any(a in ("-f", "-F", "--field", "--raw-field", "--input") for a in args):
            ctx.block("BRIEF 2-10: `gh api` with a non-GET method / payload could change GitHub state; forbidden")
    elif grp == "pr" and sub in ("create", "merge", "close", "edit", "reopen", "ready", "review", "comment", "update-branch"):
        ctx.block("BRIEF 2-10: no pull requests (never to upstream); merge locally at phase end")
    elif grp in ("secret", "variable", "workflow", "ssh-key", "gpg-key", "ruleset", "codespace", "extension", "alias"):
        ctx.block(f"BRIEF 2-10: `gh {grp}` is outside the allowed scope")
    elif grp == "auth" and sub != "status":
        ctx.block("BRIEF 2-10: `gh auth` (other than status) is forbidden")


def check_fs(low: str, args, pos, ctx: Ctx) -> None:
    ovals = option_values(args)
    if low in DELETE_ALL | WRITE_ALL and ctx.via_xargs:
        ctx.block(f"BRIEF 2-5: `xargs {low}` targets cannot be verified; use explicit literal paths")
    if low in DELETE_ALL:
        for a in pos + ovals:
            check_path_write(a, "delete", ctx, low)
    elif low in WRITE_ALL:
        for a in pos + ovals:
            check_path_write(a, "write", ctx, low)
    elif low in WRITE_DEST_LAST:
        targets = [args[i + 1] for i, a in enumerate(args[:-1]) if a in ("-t", "--target-directory")] or pos[-1:]
        for a in targets + ovals:
            if ":" in a and not a.startswith("/"):
                continue  # remote destination
            check_path_write(a, "write", ctx, low)
    elif low == "dd":
        for a in args:
            if a.startswith("of="):
                check_path_write(a[3:], "write", ctx, "dd")
    elif low in ("sed", "perl", "ruby") and any(re.match(r"^-[a-zA-Z]*i", a) or a == "--in-place" or a.startswith("--in-place=") for a in args):
        for a in pos:
            check_path_write(a, "write", ctx, f"{low} -i")
    elif low == "find":
        paths = []
        for a in args:
            if a.startswith("-") or a in ("(", "!"):
                break
            paths.append(a)
        deleting = "-delete" in args or any(
            a in ("-exec", "-execdir", "-ok", "-okdir") and i + 1 < len(args) and Path(args[i + 1]).name in ("rm", "mv", "shred", "unlink", "rmdir")
            for i, a in enumerate(args)
        )
        if deleting:
            for a in paths or ["."]:
                check_path_write(a, "delete", ctx, "find -delete/-exec rm")


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception as exc:  # fail closed: we cannot verify what we cannot read
        print(f"tray_guard: cannot parse hook input ({exc}); blocking.", file=sys.stderr)
        return 2
    if data.get("tool_name") != "Bash":
        return 0
    command = (data.get("tool_input") or {}).get("command", "")
    if not command.strip():
        return 0
    ctx = Ctx(cwd=os.path.realpath(data.get("cwd") or str(REPO)))
    analyze(command, ctx)
    if ctx.reasons:
        print(
            "tray_guard: BLOCKED\n  - " + "\n  - ".join(ctx.reasons)
            + "\nThese are owner-controlled safety rules (CLAUDE.md, 'ABSOLUTE RULES'). Stop and ask the owner if you really need this.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
