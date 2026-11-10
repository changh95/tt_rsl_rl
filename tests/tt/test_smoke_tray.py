# SPDX-License-Identifier: BSD-3-Clause
"""Phase 0 smoke test (b): all 8 chips of tray 3 as one 2x4 mesh, first WITHOUT fabric, then with FABRIC_1D + all_reduce.

Device tests: run ONLY through the wrapper, ONE test per process (one open mesh per process; the first run showed that the node stays
held until process exit), and in this order:

    scripts/tt_run.sh --timeout 900 --tag smoke8a -- python -m pytest -m tt tests/tt/test_smoke_tray.py -s -q -k no_fabric
    scripts/tt_run.sh --timeout 900 --tag smoke8b -- python -m pytest -m tt tests/tt/test_smoke_tray.py -s -q -k all_reduce

These measure; they set no tolerance (docs/porting/TOLERANCES.md is filled from the numbers, BRIEF rule 2-9). The asserts are "the guard held
exactly the 8 selected nodes", "results are finite / not garbage", "every chip computed its own shard".
"""

import json
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
ALL = {f"/dev/tenstorrent/{n}" for n in range(16, 24)}


def _rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.double() - ref.double()).norm() / ref.double().norm())


def _save(name: str, report: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(report, indent=2))


def test_eight_chip_mesh_open_no_fabric():
    from rsl_rl_tt.device import list_open_tt_nodes, open_tray_mesh

    report: dict = {}
    t0 = time.time()
    with open_tray_mesh((2, 4)) as tm:
        report["t_open_s"] = round(time.time() - t0, 2)
        import ttnn

        mesh = tm.mesh_device
        assert tm.cfg.chips == tuple(range(16, 24))
        assert list_open_tt_nodes() == ALL, "exactly the 8 tray-3 nodes must be held"
        report.update(shape=str(mesh.shape), num_devices=mesh.get_num_devices(), device_ids=list(mesh.get_device_ids()), grid=str(mesh.compute_with_storage_grid_size()))
        print("\nMESH:", report)
        assert report["num_devices"] == 8

        # every chip gets its own shard (upload), computes its own matmul, and sends it back
        torch.manual_seed(2)
        a, b = torch.randn(8, 1, 64, 128), torch.randn(1, 1, 128, 96)
        cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        ta = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        compose = ttnn.ConcatMeshToTensor(mesh, dim=0)
        assert torch.equal(ttnn.to_torch(ta, mesh_composer=compose), a), "shard upload/download must be bit exact"
        out = ttnn.to_torch(ttnn.matmul(ta, tb, compute_kernel_config=cfg), mesh_composer=compose)
        ref = a.double() @ b.double()
        per_chip = [_rel(out[i], ref[i]) for i in range(8)]
        report["matmul_rel_fro_per_chip"] = per_chip
        print("per-chip matmul rel_fro:", ["%.2e" % r for r in per_chip])
        assert max(per_chip) < 0.1

    report["fds_held_after_close"] = sorted(list_open_tt_nodes())
    assert set(report["fds_held_after_close"]) <= ALL
    _save("smoke8a_no_fabric.json", report)


def test_eight_chip_all_reduce_fabric_1d():
    from rsl_rl_tt.device import list_open_tt_nodes, open_tray_mesh

    report: dict = {"all_reduce": []}
    t0 = time.time()
    with open_tray_mesh((2, 4), fabric_config="FABRIC_1D") as tm:
        report["t_open_s"] = round(time.time() - t0, 2)
        import ttnn

        mesh = tm.mesh_device
        assert list_open_tt_nodes() == ALL, "exactly the 8 tray-3 nodes must be held"
        print("\nMESH:", mesh.shape, list(mesh.get_device_ids()))
        compose = ttnn.ConcatMeshToTensor(mesh, dim=0)

        for n in (256, 4096, 65536):  # floats per row of 32 -> 8K .. 2M values per chip, the scale of RL-MLP gradients
            torch.manual_seed(3)
            shards = torch.randn(8, 1, 32, n)
            t = ttnn.from_torch(shards, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
            r = ttnn.all_reduce(t)
            out = ttnn.to_torch(r, mesh_composer=compose)
            ref = shards.double().sum(0)
            per_chip_rel = [_rel(out[i], ref) for i in range(8)]
            chip_spread = float((out - out[0:1]).abs().max())
            ref32 = shards.sum(0)  # torch fp32 sum, for scale
            t_start = time.time()
            for _ in range(20):
                r = ttnn.all_reduce(t)
            ttnn.synchronize_device(mesh)
            ms = (time.time() - t_start) / 20 * 1e3
            row = {"values_per_chip": 32 * n, "rel_fro_vs_fp64_max_over_chips": max(per_chip_rel), "torch_fp32_sum_rel": _rel(ref32, ref), "max_abs_spread_between_chips": chip_spread, "ms_per_all_reduce": round(ms, 3)}
            report["all_reduce"].append(row)
            print("all_reduce", row)
            assert max(per_chip_rel) < 0.1 and torch.isfinite(out).all()

    report["fds_held_after_close"] = sorted(list_open_tt_nodes())
    assert set(report["fds_held_after_close"]) <= ALL
    _save("smoke8b_all_reduce.json", report)
