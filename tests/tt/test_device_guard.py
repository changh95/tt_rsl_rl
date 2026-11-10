# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests of rsl_rl_tt/device.py (BRIEF 2-2): no hardware is touched.

The guard logic is exercised with an injected fake `ttnn`. Two tests read real host facts that need no device open:
the output of scripts/tray_env.sh and the sysfs BDF symlinks of the 8 ALLOWED nodes only.
"""

import os
import subprocess
from pathlib import Path

import pytest

from rsl_rl_tt import device as dev

REPO = Path(__file__).resolve().parents[2]
TRAY_ENV = REPO / "scripts" / "tray_env.sh"
NODES = list(range(16, 24))
BDFS = {n: f"0000:c{n - 15}:00.0" for n in NODES}  # mirrors tray_env.sh (cross-checked by test_tray_env_matches_this_test_table)


def make_env(chips=NODES, **overrides) -> dict:
    env = {
        "TT_TRAY_ID": "3",
        "TT_TRAY_NODE_BDFS": " ".join(f"{n}={BDFS[n]}" for n in NODES),
        "TT_ALLOWED_DEV_NODES": " ".join(f"/dev/tenstorrent/{n}" for n in NODES),
        "TT_TRAY_CHIPS": " ".join(str(c) for c in chips),
        "TT_VISIBLE_DEVICES": ",".join(BDFS[c] for c in chips),
        "RSL_RL_TT_LOCK_HELD": "3",
    }
    env.update(overrides)
    return {k: v for k, v in env.items() if v is not None}


def expected_for(env: dict) -> dict:
    out = {name: env.get(name) for name in dev.GUARDED_ENV}
    return out


class FakeFabricConfig:
    DISABLED = "DISABLED"
    FABRIC_1D = "FABRIC_1D"


class FakeTtnn:
    FabricConfig = FakeFabricConfig

    class MeshShape:
        def __init__(self, r, c):
            self.r, self.c = r, c

    def __init__(self, fail_open=False):
        self.calls, self.fail_open = [], fail_open

    def set_fabric_config(self, cfg):
        self.calls.append(("set_fabric_config", cfg))

    def open_mesh_device(self, **kw):
        self.calls.append(("open_mesh_device", (kw["mesh_shape"].r, kw["mesh_shape"].c), kw["l1_small_size"]))
        if self.fail_open:
            raise RuntimeError("fabric bring-up failed")
        return "MESH"

    def close_mesh_device(self, mesh):
        self.calls.append(("close_mesh_device", mesh))


def make_deps(env, ttnn=None, opened=None, bdf=None, importer=None, visible=None):
    ttnn = ttnn or FakeTtnn()
    return dev.Deps(
        environ=env,
        load_expected_env=lambda chips: expected_for(make_env(chips=[int(c) for c in chips.split()]) if chips else make_env()),
        read_bdf=(lambda n: BDFS[n]) if bdf is None else bdf,
        list_open_nodes=lambda: set(opened if opened is not None else {f"/dev/tenstorrent/{n}" for n in NODES}),
        list_visible_nodes=lambda: set(
            visible if visible is not None else {f"/dev/tenstorrent/{c}" for c in env["TT_TRAY_CHIPS"].split()}
        ),
        import_ttnn=importer or (lambda: ttnn),
    ), ttnn


@pytest.fixture(autouse=True)
def _no_leftover_mesh():
    yield
    if dev._ACTIVE is not None:
        dev._ACTIVE.close()


# ---- happy path ------------------------------------------------------------------------------------------------------
def test_happy_path_order_and_cleanup():
    deps, ttnn = make_deps(make_env())
    with dev.open_tray_mesh((2, 4), fabric_config="FABRIC_1D", deps=deps) as tm:
        assert tm.mesh_device == "MESH" and tm.cfg.chips == tuple(NODES)
        assert [c[0] for c in ttnn.calls] == ["set_fabric_config", "open_mesh_device"]  # fabric BEFORE open
        assert ttnn.calls[1] == ("open_mesh_device", (2, 4), 24576)
    assert [c[0] for c in ttnn.calls] == ["set_fabric_config", "open_mesh_device", "close_mesh_device", "set_fabric_config"]
    assert ttnn.calls[-1] == ("set_fabric_config", "DISABLED")  # fabric disabled AFTER close
    assert dev._ACTIVE is None


def test_single_chip_subset():
    deps, ttnn = make_deps(make_env(chips=[16]), opened={"/dev/tenstorrent/16"})
    with dev.open_tray_mesh((1, 1), deps=deps):
        pass
    assert [c[0] for c in ttnn.calls] == ["open_mesh_device", "close_mesh_device"]  # no fabric requested -> none touched


def test_close_is_idempotent():
    deps, ttnn = make_deps(make_env())
    tm = dev.open_tray_mesh((2, 4), deps=deps)
    tm.close()
    tm.close()
    assert [c[0] for c in ttnn.calls].count("close_mesh_device") == 1


def test_second_open_is_refused():
    deps, _ = make_deps(make_env())
    dev.open_tray_mesh((2, 4), deps=deps)
    with pytest.raises(dev.TrayGuardError, match="already open"):
        dev.open_tray_mesh((2, 4), deps=deps)


# ---- refusals happen BEFORE ttnn is even imported -------------------------------------------------------------------
def _never_import():
    raise AssertionError("ttnn must not be imported when a pre-check fails")


@pytest.mark.parametrize(
    "env, match",
    [
        (make_env(TT_VISIBLE_DEVICES=None), "TT_VISIBLE_DEVICES"),  # unset would expose EVERY chip of the host
        (make_env(TT_VISIBLE_DEVICES="0000:c1:00.0"), "TT_VISIBLE_DEVICES"),
        (make_env(TT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"), "TT_VISIBLE_DEVICES"),  # bare integers are BDF-sorted indices, not /dev numbers
        (make_env(TT_TRAY_ID="2"), "TT_TRAY_ID"),
        (make_env(TT_MESH_ID="0"), "must be unset"),
        (make_env(TT_MESH_GRAPH_DESC_PATH="/tmp/x.textproto"), "must be unset"),
        (make_env(TT_METAL_VISIBLE_DEVICES="16"), "must be unset"),
        (make_env(RSL_RL_TT_LOCK_HELD=None), "tray lock not held"),
        (make_env(RSL_RL_TT_LOCK_HELD="1"), "tray lock not held"),
    ],
    ids=["visible-unset", "visible-one-chip", "visible-ints", "tray-id", "mesh-id", "mgd-path", "dead-var", "no-lock", "wrong-lock"],
)
def test_environment_refusals(env, match):
    deps, ttnn = make_deps(env, importer=_never_import)
    with pytest.raises(dev.TrayGuardError, match=match):
        dev.open_tray_mesh((2, 4), deps=deps)
    assert ttnn.calls == []


def test_sysfs_mismatch_refuses_before_import():
    deps, _ = make_deps(make_env(), bdf=lambda n: "0000:99:00.0" if n == 19 else BDFS[n], importer=_never_import)
    with pytest.raises(dev.TrayGuardError, match="node numbering changed"):
        dev.open_tray_mesh((2, 4), deps=deps)


def test_sysfs_node_missing_refuses():
    deps, _ = make_deps(make_env(), bdf=lambda n: None, importer=_never_import)
    with pytest.raises(dev.TrayGuardError, match="node numbering changed"):
        dev.open_tray_mesh((2, 4), deps=deps)


def test_mesh_shape_must_match_selected_chips():
    deps, _ = make_deps(make_env(chips=[16, 17]), importer=_never_import)
    with pytest.raises(dev.TrayGuardError, match="needs 8 chips but 2 are selected"):
        dev.open_tray_mesh((2, 4), deps=deps)


def test_chip_outside_tray_is_rejected_by_tray_env_and_by_check_env():
    bad = make_env()
    bad_expected = expected_for(bad)
    bad_expected["TT_TRAY_CHIPS"] = "24"
    bad["TT_TRAY_CHIPS"] = "24"
    with pytest.raises(dev.TrayGuardError):
        dev.check_env(bad_expected, bad)


# ---- fd inspection after open -----------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "opened, match",
    [
        ({"/dev/tenstorrent/16", "/dev/tenstorrent/3"}, "OUTSIDE tray 3"),
        ({"/dev/tenstorrent/24"}, "OUTSIDE tray 3"),
        ({"/dev/tenstorrent/15"}, "OUTSIDE tray 3"),
        ({"/dev/tenstorrent"}, "OUTSIDE tray 3"),
    ],
)
def test_foreign_open_node_closes_mesh_and_raises(opened, match):
    deps, ttnn = make_deps(make_env(), opened=opened)
    with pytest.raises(dev.TrayGuardError, match=match):
        dev.open_tray_mesh((2, 4), fabric_config="FABRIC_1D", deps=deps)
    assert [c[0] for c in ttnn.calls] == ["set_fabric_config", "open_mesh_device", "close_mesh_device", "set_fabric_config"]
    assert dev._ACTIVE is None


def test_node_outside_selection_but_inside_tray_is_also_refused():
    deps, ttnn = make_deps(make_env(chips=[16]), opened={"/dev/tenstorrent/16", "/dev/tenstorrent/17"})
    with pytest.raises(dev.TrayGuardError, match="outside the selected chips"):
        dev.open_tray_mesh((1, 1), deps=deps)
    assert ttnn.calls[-1][0] == "close_mesh_device"


def test_failed_open_cleans_up_fabric():
    deps, ttnn = make_deps(make_env(), ttnn=FakeTtnn(fail_open=True))
    with pytest.raises(RuntimeError, match="fabric bring-up failed"):
        dev.open_tray_mesh((2, 4), fabric_config="FABRIC_1D", deps=deps)
    assert ttnn.calls[-1] == ("set_fabric_config", "DISABLED")
    assert dev._ACTIVE is None


# ---- the namespace mask (ADR-0005 option II) --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "visible, match",
    [
        ({f"/dev/tenstorrent/{n}" for n in NODES} | {"/dev/tenstorrent/3"}, "may not use"),  # another tray's node is visible
        ({f"/dev/tenstorrent/{n}" for n in range(0, 24)}, "may not use"),  # the unmasked host view
        ({f"/dev/tenstorrent/{n}" for n in NODES[:-1]}, "not visible"),  # a selected chip is missing
        (set(), "not visible"),
    ],
    ids=["extra-node", "unmasked-host", "missing-selected", "empty"],
)
def test_unmasked_or_incomplete_dev_view_refuses_before_import(visible, match):
    deps, ttnn = make_deps(make_env(), visible=visible, importer=_never_import)
    with pytest.raises(dev.TrayGuardError, match=match):
        dev.open_tray_mesh((2, 4), deps=deps)
    assert ttnn.calls == []


def test_subset_selection_must_not_see_the_other_chips_of_the_tray_either():
    deps, _ = make_deps(make_env(chips=[16]), visible={"/dev/tenstorrent/16", "/dev/tenstorrent/17"}, importer=_never_import)
    with pytest.raises(dev.TrayGuardError, match="may not use"):
        dev.open_tray_mesh((1, 1), deps=deps)


def test_list_visible_dev_nodes_lists_names_only(tmp_path):
    (tmp_path / "16").write_text("")
    (tmp_path / "17").write_text("")
    assert dev.list_visible_dev_nodes(str(tmp_path)) == {f"{tmp_path}/16", f"{tmp_path}/17"}
    assert dev.list_visible_dev_nodes(str(tmp_path / "missing")) == set()


# ---- fd listing -------------------------------------------------------------------------------------------------------
def test_list_open_tt_nodes_reads_symlink_targets(tmp_path):
    fd_dir = tmp_path / "fd"
    fd_dir.mkdir()
    (fd_dir / "3").symlink_to("/dev/tenstorrent/17")
    (fd_dir / "4").symlink_to("/dev/tenstorrent/2")
    (fd_dir / "5").symlink_to("/etc/hostname")
    (fd_dir / "6").symlink_to("/dev/null")
    assert dev.list_open_tt_nodes(str(fd_dir)) == {"/dev/tenstorrent/17", "/dev/tenstorrent/2"}


def test_list_open_tt_nodes_on_this_process_is_empty():
    assert dev.list_open_tt_nodes() == set()  # a plain pytest process holds no device


# ---- real tray_env.sh / real sysfs (no device open) ------------------------------------------------------------------
def test_tray_env_default_is_all_eight_chips():
    exp = dev.load_expected_env()
    assert exp["TT_TRAY_ID"] == "3"
    assert exp["TT_TRAY_CHIPS"] == " ".join(map(str, NODES))
    assert exp["TT_VISIBLE_DEVICES"].split(",") == [BDFS[n] for n in NODES]
    for name in ("TT_MESH_ID", "TT_MESH_HOST_RANK", "TT_MESH_GRAPH_DESC_PATH", "TT_METAL_VISIBLE_DEVICES", "TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        assert exp[name] is None
    cfg = dev.check_env(exp, {k: v for k, v in exp.items() if v is not None})
    assert cfg.chips == tuple(NODES)


def test_tray_env_matches_this_test_table():
    assert dev.check_env(expected_for(make_env()), make_env()).node_bdfs == dev.check_env(
        dev.load_expected_env(), {k: v for k, v in dev.load_expected_env().items() if v is not None}
    ).node_bdfs


@pytest.mark.parametrize("chips, expect", [("16", [16]), ("16,17", [16, 17]), ("23 22", [23, 22]), ("16 16 17", [16, 17])])
def test_tray_env_chip_selection(chips, expect):
    exp = dev.load_expected_env(chips)
    assert exp["TT_TRAY_CHIPS"].split() == [str(c) for c in expect]
    assert exp["TT_VISIBLE_DEVICES"].split(",") == [BDFS[c] for c in expect]


@pytest.mark.parametrize("chips", ["15", "24", "0", "16 24", "abc", "", "16,,x"])
def test_tray_env_rejects_other_chips(chips):
    with pytest.raises(dev.TrayGuardError):
        dev.load_expected_env(chips)


def test_tray_env_refuses_when_executed_via_source_with_foreign_chip():
    res = subprocess.run(["bash", "-c", f'TT_TRAY_CHIPS="3" source {TRAY_ENV}'], capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
    assert res.returncode != 0 and "not one of the tray-3 nodes" in res.stderr


@pytest.mark.skipif(not os.path.isdir("/sys/class/tenstorrent"), reason="no Tenstorrent sysfs class on this host")
def test_sysfs_bdfs_of_allowed_nodes_match_tray_env():
    """Reads only the sysfs symlinks of nodes 16..23 (no device open)."""
    cfg = dev.check_env(dev.load_expected_env(), {k: v for k, v in dev.load_expected_env().items() if v is not None})
    dev.verify_sysfs_bdfs(cfg)  # raises if /dev/tenstorrent/16..23 no longer map to the BDFs in tray_env.sh
