# SPDX-License-Identifier: BSD-3-Clause
"""The ONLY place in this project that opens Tenstorrent devices (BRIEF 2-2).

`open_tray_mesh()` is the single entry point. Before anything is opened it checks, in this order:

  (a) the device-related environment equals what `scripts/tray_env.sh` produces (the file is re-evaluated in a clean sub-shell, so it is
      the single source of truth; nothing about the tray is hard-coded here);
  (b) the job runs under `scripts/tt_run.sh` (tray lock held, `RSL_RL_TT_LOCK_HELD == TT_TRAY_ID`);
  (c) sysfs maps every selected `/dev/tenstorrent/N` to the BDF that `TT_VISIBLE_DEVICES` names (a symlink read, no device open);
  (d) the requested mesh shape matches the number of selected chips.
Only then `ttnn` is imported and the mesh opened. Right after opening it reads `/proc/self/fd`: if this process holds any
`/dev/tenstorrent/*` node outside the selected chips, the mesh is closed immediately and `TrayGuardError` is raised.

Known limit (docs/porting/TTNN_OP_SURVEY.md 11.3): UMD enumeration briefly open()s and closes EVERY node on the host before it
filters by TT_VISIBLE_DEVICES. The fd check cannot see those closed probes; hard isolation needs OS-level help (DECISIONS ADR-0005).

Closing is guaranteed by `TrayMesh.close()` (context manager / `finally`) and by an `atexit` hook. No other module may call
`ttnn.open_*` / `Create*Device` (tests/tt/test_repo_invariants.py enforces this).
"""

from __future__ import annotations

import atexit
import importlib
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAY_ENV = REPO_ROOT / "scripts" / "tray_env.sh"

# Every device-selection variable that tray_env.sh owns. A guarded variable that is unset in tray_env.sh must be unset here too.
GUARDED_ENV = (
    "TT_TRAY_ID",
    "TT_TRAY_NODE_BDFS",
    "TT_ALLOWED_DEV_NODES",
    "TT_TRAY_CHIPS",
    "TT_VISIBLE_DEVICES",
    "TT_MESH_ID",
    "TT_MESH_HOST_RANK",
    "TT_MESH_GRAPH_DESC_PATH",
    "TT_METAL_VISIBLE_DEVICES",
    "TT_METAL_MOCK_CLUSTER_DESC_PATH",
)
SYSFS_NODE_DEVICE = "/sys/class/tenstorrent/tenstorrent!{node}/device"
DEV_PREFIX = "/dev/tenstorrent/"


class TrayGuardError(RuntimeError):
    """A tray-isolation check failed; nothing (or nothing more) was opened."""


@dataclass(frozen=True)
class TrayConfig:
    tray_id: int
    node_bdfs: Mapping[int, str]  # all allowed nodes of the tray -> BDF
    chips: tuple[int, ...]  # the selected nodes for this run
    visible_devices: str

    @property
    def allowed_nodes(self) -> frozenset[str]:
        return frozenset(f"{DEV_PREFIX}{n}" for n in self.node_bdfs)

    @property
    def selected_nodes(self) -> frozenset[str]:
        return frozenset(f"{DEV_PREFIX}{n}" for n in self.chips)


# ----------------------------------------------------------------------------------------------------------------------
# (a) environment
# ----------------------------------------------------------------------------------------------------------------------
def load_expected_env(chips: str | None = None, tray_env: Path = TRAY_ENV) -> dict[str, str | None]:
    """Evaluate `scripts/tray_env.sh` in a clean sub-shell and return the guarded variables (None = must be unset)."""
    script = (
        'source "$1" || exit 3; shift; '
        'for v in "$@"; do if [ -n "${!v+x}" ]; then printf "%s=%s\\0" "$v" "${!v}"; else printf "!%s\\0" "$v"; fi; done'
    )
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/")}
    if chips is not None:
        env["TT_TRAY_CHIPS"] = chips
    proc = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", script, "_", str(tray_env), *GUARDED_ENV],
        env=env,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise TrayGuardError(f"scripts/tray_env.sh rejected this configuration: {proc.stderr.decode(errors='replace').strip() or proc.returncode}")
    expected: dict[str, str | None] = {}
    for item in proc.stdout.split(b"\0"):
        if not item:
            continue
        text = item.decode()
        if text.startswith("!"):
            expected[text[1:]] = None
        else:
            key, _, value = text.partition("=")
            expected[key] = value
    return expected


def check_env(expected: Mapping[str, str | None], environ: Mapping[str, str] | None = None) -> TrayConfig:
    """Compare the live environment with tray_env.sh and build a validated TrayConfig."""
    environ = os.environ if environ is None else environ
    problems = []
    for name in GUARDED_ENV:
        want, have = expected.get(name), environ.get(name)
        if want is None and have is not None:
            problems.append(f"{name} must be unset (found {have!r})")
        elif want is not None and have != want:
            problems.append(f"{name} is {have!r} but scripts/tray_env.sh gives {want!r}")
    if problems:
        raise TrayGuardError("device environment differs from scripts/tray_env.sh (start the job with scripts/tt_run.sh):\n  - " + "\n  - ".join(problems))

    try:
        tray_id = int(expected["TT_TRAY_ID"] or "")
        node_bdfs = {int(k): v for k, v in (pair.split("=", 1) for pair in (expected["TT_TRAY_NODE_BDFS"] or "").split())}
        chips = tuple(int(c) for c in (expected["TT_TRAY_CHIPS"] or "").split())
        visible = expected["TT_VISIBLE_DEVICES"] or ""
    except (KeyError, ValueError) as exc:
        raise TrayGuardError(f"scripts/tray_env.sh produced an unparsable tray description: {exc}") from exc

    cfg = TrayConfig(tray_id=tray_id, node_bdfs=node_bdfs, chips=chips, visible_devices=visible)
    if not cfg.chips or not set(cfg.chips) <= set(cfg.node_bdfs):
        raise TrayGuardError(f"selected chips {cfg.chips} are not a non-empty subset of the tray nodes {sorted(cfg.node_bdfs)}")
    if len(set(cfg.node_bdfs.values())) != len(cfg.node_bdfs):
        raise TrayGuardError("tray_env.sh maps two nodes to the same BDF")
    if sorted(visible.split(",")) != sorted(cfg.node_bdfs[c] for c in cfg.chips):
        raise TrayGuardError(f"TT_VISIBLE_DEVICES {visible!r} does not equal the BDFs of the selected chips {cfg.chips}")
    if set((expected["TT_ALLOWED_DEV_NODES"] or "").split()) != set(cfg.allowed_nodes):
        raise TrayGuardError("TT_ALLOWED_DEV_NODES and TT_TRAY_NODE_BDFS disagree in tray_env.sh")
    return cfg


# ----------------------------------------------------------------------------------------------------------------------
# (c) sysfs node -> BDF
# ----------------------------------------------------------------------------------------------------------------------
def read_bdf_from_sysfs(node: int) -> str | None:
    """BDF of /dev/tenstorrent/<node> from the sysfs symlink; does NOT open the device."""
    path = SYSFS_NODE_DEVICE.format(node=node)
    return os.path.basename(os.path.realpath(path)) if os.path.exists(path) else None


def verify_sysfs_bdfs(cfg: TrayConfig, read_bdf: Callable[[int], str | None] = read_bdf_from_sysfs) -> None:
    for node in cfg.chips:
        got, want = read_bdf(node), cfg.node_bdfs[node]
        if got != want:
            raise TrayGuardError(
                f"/dev/tenstorrent/{node} has BDF {got!r} in sysfs but scripts/tray_env.sh expects {want!r}; the node numbering changed "
                "(reboot/reset?). Refusing to open anything. Ask the owner."
            )


# ----------------------------------------------------------------------------------------------------------------------
# open-fd inspection
# ----------------------------------------------------------------------------------------------------------------------
def list_open_tt_nodes(fd_dir: str = "/proc/self/fd") -> set[str]:
    """`/dev/tenstorrent/*` nodes this process currently holds open."""
    nodes = set()
    for name in os.listdir(fd_dir):
        try:
            target = os.readlink(os.path.join(fd_dir, name))
        except OSError:  # fd closed meanwhile (including the directory fd used by listdir)
            continue
        if target == DEV_PREFIX.rstrip("/") or target.startswith(DEV_PREFIX):
            nodes.add(target)
    return nodes


def verify_open_nodes(cfg: TrayConfig, opened: set[str]) -> None:
    outside_tray = sorted(n for n in opened if n not in cfg.allowed_nodes)
    outside_selection = sorted(n for n in opened if n in cfg.allowed_nodes and n not in cfg.selected_nodes)
    if outside_tray:
        raise TrayGuardError(f"process holds device nodes OUTSIDE tray {cfg.tray_id}: {outside_tray}")
    if outside_selection:
        raise TrayGuardError(f"process holds tray nodes outside the selected chips {cfg.chips}: {outside_selection}")


# ----------------------------------------------------------------------------------------------------------------------
# the single entry point
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Deps:
    """Injection points so the guard logic is testable on the host without hardware."""

    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    load_expected_env: Callable[[str | None], dict[str, str | None]] = field(default_factory=lambda: load_expected_env)
    read_bdf: Callable[[int], str | None] = field(default_factory=lambda: read_bdf_from_sysfs)
    list_open_nodes: Callable[[], set[str]] = field(default_factory=lambda: list_open_tt_nodes)
    import_ttnn: Callable[[], Any] = field(default_factory=lambda: (lambda: importlib.import_module("ttnn")))


_ACTIVE: TrayMesh | None = None


class TrayMesh:
    """An open mesh on (a subset of) the tray. Use as a context manager; `close()` is idempotent and also runs at interpreter exit."""

    def __init__(self, ttnn: Any, mesh_device: Any, cfg: TrayConfig, fabric_set: bool):
        self._ttnn, self.mesh_device, self.cfg, self._fabric_set, self._closed = ttnn, mesh_device, cfg, fabric_set, False

    def close(self) -> None:
        global _ACTIVE
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        try:
            self._ttnn.close_mesh_device(self.mesh_device)
        except BaseException as exc:  # still reset fabric and bookkeeping below
            error = exc
        try:
            if self._fabric_set:
                self._ttnn.set_fabric_config(self._ttnn.FabricConfig.DISABLED)
        except BaseException as exc:
            error = error or exc
        if _ACTIVE is self:
            _ACTIVE = None
        atexit.unregister(self.close)
        if error is not None:
            raise error

    def __enter__(self) -> TrayMesh:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _best_effort_close(ttnn: Any, mesh: Any, fabric_set: bool) -> None:
    for what, action in (
        ("close_mesh_device", lambda: mesh is not None and ttnn.close_mesh_device(mesh)),
        ("set_fabric_config(DISABLED)", lambda: fabric_set and ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)),
    ):
        try:
            action()
        except BaseException as exc:
            print(f"rsl_rl_tt.device: {what} failed during cleanup: {exc!r}", file=sys.stderr)


def open_tray_mesh(
    mesh_shape: tuple[int, int],
    *,
    fabric_config: str | None = None,
    l1_small_size: int = 24576,
    trace_region_size: int | None = None,
    deps: Deps | None = None,
) -> TrayMesh:
    """Open a MeshDevice on the selected chips of tray 3, guarded as described in the module docstring.

    mesh_shape      (rows, cols); rows*cols must equal the number of selected chips (`scripts/tt_run.sh --chips`).
    fabric_config   name of a `ttnn.FabricConfig` member (e.g. "FABRIC_1D") needed for CCL; None leaves the fabric disabled.
    l1_small_size   must be > 0 for CCL ops (tt-metal survey item 9).
    trace_region_size  bytes for trace buffers; None keeps the ttnn default (size it after measuring, BRIEF 3.3).
    """
    global _ACTIVE
    deps = deps or Deps()
    if _ACTIVE is not None:
        raise TrayGuardError("a tray mesh is already open in this process; close it first (one open per process)")

    expected = deps.load_expected_env(deps.environ.get("TT_TRAY_CHIPS"))
    cfg = check_env(expected, deps.environ)
    if deps.environ.get("RSL_RL_TT_LOCK_HELD") != str(cfg.tray_id):
        raise TrayGuardError("tray lock not held: device jobs must be started with scripts/tt_run.sh (BRIEF 2-3)")
    verify_sysfs_bdfs(cfg, deps.read_bdf)
    rows, cols = mesh_shape
    if rows * cols != len(cfg.chips):
        raise TrayGuardError(f"mesh shape {rows}x{cols} needs {rows * cols} chips but {len(cfg.chips)} are selected: {cfg.chips}")

    ttnn = deps.import_ttnn()  # imported only after every check passed
    fabric_set, mesh = False, None
    try:
        if fabric_config is not None:
            ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fabric_config))
            fabric_set = True
        kwargs: dict[str, Any] = {"mesh_shape": ttnn.MeshShape(rows, cols), "l1_small_size": l1_small_size, "num_command_queues": 1}
        if trace_region_size is not None:
            kwargs["trace_region_size"] = trace_region_size
        mesh = ttnn.open_mesh_device(**kwargs)
        verify_open_nodes(cfg, deps.list_open_nodes())
    except BaseException:
        _best_effort_close(ttnn, mesh, fabric_set)
        raise

    tray_mesh = TrayMesh(ttnn, mesh, cfg, fabric_set)
    _ACTIVE = tray_mesh
    atexit.register(tray_mesh.close)
    return tray_mesh
