# SPDX-License-Identifier: BSD-3-Clause
"""Phase 0 probe on all 8 chips (survey Q15): how exact can a cross-chip fp32 gradient sum be? MEASURES only; no tolerance is set here.

    scripts/tt_run.sh --timeout 900 --tag noise_mesh -- python -m pytest -m tt tests/tt/test_noise_floor_mesh.py -s -q -p no:cacheprovider

Smoke (b) showed `ttnn.all_reduce` on fp32 has rel_fro ~6.4e-4 vs a float64 sum (torch fp32 sum: 5.6e-8). Candidates compared here:
  all_reduce                   : the stock op
  all_gather(2 axes) + local sum: every chip gathers all 8 fp32 shards (a copy, no arithmetic) and sums them with `ttnn.sum`
  all_gather + sequential adds : the same, summed as ((s0+s1)+s2)+... with `ttnn.add` (a fixed order, bit-identical on every chip)
"""

import json
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
ALL = {f"/dev/tenstorrent/{n}" for n in range(16, 24)}


def rel(out, ref):
    return float((out.double() - ref.double()).norm() / ref.double().norm())


def test_cross_chip_sum_variants():
    from rsl_rl_tt.device import list_open_tt_nodes, open_tray_mesh

    report = {}
    with open_tray_mesh((2, 4), fabric_config="FABRIC_1D") as tm:
        import ttnn

        mesh = tm.mesh_device
        assert list_open_tt_nodes() == ALL
        compose = ttnn.ConcatMeshToTensor(mesh, dim=0)
        shard = ttnn.ShardTensorToMesh(mesh, dim=0)

        def timed(fn, reps=20):
            fn()
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(reps):
                r = fn()
            ttnn.synchronize_device(mesh)
            return r, round((time.time() - t0) / reps * 1e3, 3)

        for n in (256, 4096, 32768):  # 8K, 131K, 1M values per chip
            torch.manual_seed(21)
            shards = torch.randn(8, 1, 32, n) * torch.logspace(-3, 1, 8).reshape(8, 1, 1, 1)  # chips with different gradient scales
            ref = shards.double().sum(0)
            t = ttnn.from_torch(shards, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)
            row = {"values_per_chip": 32 * n, "torch_fp32_sum": rel(shards.sum(0), ref)}

            def check(name, fn):
                try:
                    r, ms = timed(fn)
                    out = ttnn.to_torch(r, mesh_composer=compose)
                    out = out.reshape(8, -1, 32, n)[:, :1]  # one [1,32,n] block per chip
                    row[name] = {
                        "rel_fro": max(rel(out[i, 0], ref[0]) for i in range(8)),
                        "chips_bitwise_identical": bool(all(torch.equal(out[0], out[i]) for i in range(1, 8))),
                        "ms": ms,
                    }
                except Exception as exc:
                    row[name] = {"unsupported": f"{type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"}
                print(f"  n={n} {name}: {row[name]}")

            def gather_all():  # 2x4 mesh on FABRIC_1D: all_gather needs a cluster_axis, so gather along axis 1 (4 chips) then axis 0 (2 chips)
                return ttnn.all_gather(ttnn.all_gather(t, dim=0, cluster_axis=1), dim=0, cluster_axis=0)

            check("all_reduce", lambda: ttnn.all_reduce(t))
            check("all_reduce(axis1)+all_reduce(axis0)", lambda: ttnn.all_reduce(ttnn.all_reduce(t, cluster_axis=1), cluster_axis=0))
            check("all_gather(2 axes)+ttnn.sum", lambda: ttnn.sum(gather_all(), dim=0, keepdim=True))

            def gather_adds():
                g = gather_all()
                parts = [g[i : i + 1] for i in range(8)]
                acc = parts[0]
                for p in parts[1:]:
                    acc = ttnn.add(acc, p)
                return acc

            check("all_gather+sequential_add", gather_adds)
            report[f"n={n}"] = row

        report["fds_held"] = sorted(list_open_tt_nodes())
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "noise_floor_mesh.json").write_text(json.dumps(report, indent=2, default=str))
