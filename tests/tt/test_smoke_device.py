# SPDX-License-Identifier: BSD-3-Clause
"""Phase 0 smoke test (a): ONE chip through the guarded entry point, small fp32 matmuls vs torch.

Device test: run ONLY through the wrapper (the default pytest.ini deselects `-m tt`):

    scripts/tt_run.sh --chips 16 --timeout 600 --tag smoke1 -- python -m pytest -m tt tests/tt/test_smoke_device.py -s -q

It measures; it sets no tolerance (docs/porting/TOLERANCES.md is filled from these numbers, BRIEF rule 2-9). The only asserts are
"the guard held exactly the selected node" and "results are finite and not garbage".
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"


def _rel(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    diff = (out.double() - ref.double()).abs()
    return float(diff.max()), float(diff.norm() / ref.double().norm())


def test_one_chip_guarded_open_roundtrip_and_matmul():
    from rsl_rl_tt.device import list_open_tt_nodes, open_tray_mesh

    report: dict = {"t_open_s": None, "matmul": []}
    t0 = time.time()
    with open_tray_mesh((1, 1)) as tm:
        report["t_open_s"] = round(time.time() - t0, 2)
        import ttnn

        mesh = tm.mesh_device
        assert tm.cfg.chips == (16,)
        assert list_open_tt_nodes() == {"/dev/tenstorrent/16"}, "exactly the selected node must be held"
        report["arch"] = str(mesh.arch())
        report["grid"] = str(mesh.compute_with_storage_grid_size())
        report["dram_grid"] = str(mesh.dram_grid_size())
        report["device_ids"] = list(mesh.get_device_ids())
        report["mesh_shape"] = str(mesh.shape)
        print("\nDEVICE:", report)

        def up(t, dtype=ttnn.float32):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh)

        def down(t):
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

        # 1. fp32 TILE round trip must be bit exact (no precision loss on upload/download)
        x = torch.randn(1, 1, 96, 160)
        back = down(up(x))
        report["roundtrip_bit_exact"] = bool(torch.equal(back, x))
        print("roundtrip bit exact:", report["roundtrip_bit_exact"], "max abs diff", float((back - x).abs().max()))

        # 2. fp32 matmul vs a float64 reference, per math fidelity (measurement only)
        for (m, k, n) in [(64, 128, 96), (2048, 512, 256)]:
            torch.manual_seed(1)
            a, b = torch.randn(1, 1, m, k), torch.randn(1, 1, k, n)
            ref64 = a.double() @ b.double()
            base_max, base_rel = _rel(a @ b, ref64)  # torch fp32 vs fp64: the best a fp32 matmul can do
            for fid in (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4):
                cfg = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
                )
                out = down(ttnn.matmul(up(a), up(b), compute_kernel_config=cfg))
                assert out.shape == ref64.shape and torch.isfinite(out).all()
                mx, rel = _rel(out, ref64)
                row = {"mkn": [m, k, n], "fidelity": str(fid), "max_abs": mx, "rel_fro": rel, "torch_fp32_rel_fro": base_rel}
                report["matmul"].append(row)
                print(f"matmul {m}x{k}x{n} {str(fid):>18}: rel_fro={rel:.3e} max_abs={mx:.3e}  (torch fp32: {base_rel:.3e})")
                assert rel < 0.1, "garbage result (this is a sanity bound, not a tolerance)"

    assert list_open_tt_nodes() == set(), "all device fds must be gone after close"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "smoke1_chip16.json").write_text(json.dumps(report, indent=2))
    print("pid", os.getpid(), "done")
