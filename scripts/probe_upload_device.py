# SPDX-License-Identifier: BSD-3-Clause
"""Phase 5 device probe: cost of the per-step host <-> device transfers of the act step at 4096 / 65536 envs, TILE (current) vs ROW_MAJOR, padded vs real width.

    scripts/tt_run.sh --chips 16 --timeout 300 --tag probe_up -- python scripts/probe_upload_device.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.device import open_tray_mesh  # noqa: E402
from rsl_rl_tt.tt_common import TTContext  # noqa: E402


def main() -> None:
    results = {}
    with open_tray_mesh((1, 1)) as tm:
        ctx = TTContext.create(tm.mesh_device)
        ttnn, mesh = ctx.ttnn, ctx.mesh
        sync = lambda: ttnn.synchronize_device(mesh)

        def med(fn, repeats: int = 7) -> float:
            fn()
            sync()
            t = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn()
                sync()
                t.append(time.perf_counter() - t0)
            return statistics.median(t) * 1e3

        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        for n in (4096, 65536):
            obs, noise32, noise12 = torch.randn(n, 64), torch.randn(n, 32), torch.randn(n, 12)
            row = {}
            # current path: TILE layout host tensor -> persistent TILE device tensor
            dev_obs, dev_noise = ctx.upload(obs, op="H-6", batch_sharded=True), ctx.upload(noise32, op="H-6", batch_sharded=True)
            row["tile: upload_into obs [N,64]"] = med(lambda: ctx.upload_into(dev_obs, obs, op="H-6", batch_sharded=True))
            row["tile: upload_into noise [N,32]"] = med(lambda: ctx.upload_into(dev_noise, noise32, op="H-6", batch_sharded=True))
            # ROW_MAJOR host tensors into persistent ROW_MAJOR device tensors (tilize afterwards on the device)
            rm = lambda host: ttnn.from_torch(host, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper)
            dev_obs_rm = ttnn.from_torch(obs, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=mapper)
            dev_n32_rm = ttnn.from_torch(noise32, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=mapper)
            dev_n12_rm = ttnn.from_torch(noise12, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=mapper)
            row["rm: copy_host_to_device obs [N,64]"] = med(lambda: ttnn.copy_host_to_device_tensor(rm(obs), dev_obs_rm))
            row["rm: copy_host_to_device noise [N,32]"] = med(lambda: ttnn.copy_host_to_device_tensor(rm(noise32), dev_n32_rm))
            row["rm: copy_host_to_device noise [N,12]"] = med(lambda: ttnn.copy_host_to_device_tensor(rm(noise12), dev_n12_rm))
            row["rm: device tilize obs [N,64]"] = med(lambda: ttnn.to_layout(dev_obs_rm, ttnn.TILE_LAYOUT))
            # the building blocks alone: host tensor creation vs the device write
            row["host only: from_torch RM obs (mapper)"] = med(lambda: rm(obs))
            host_rm = rm(obs)
            row["device write only: RM obs"] = med(lambda: ttnn.copy_host_to_device_tensor(host_rm, dev_obs_rm))
            host_tile = ttnn.from_torch(obs, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            row["host only: from_torch TILE obs (mapper)"] = med(lambda: ttnn.from_torch(obs, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper))
            row["device write only: TILE obs"] = med(lambda: ttnn.copy_host_to_device_tensor(host_tile, dev_obs))
            # downloads: actions [N, 32] TILE (current) vs ROW_MAJOR [N, 12]
            act_tile = ctx.upload(noise32, op="H-6", batch_sharded=True)
            row["tile: download actions [N,32]"] = med(lambda: ttnn.to_torch(act_tile, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)))
            act_rm = ttnn.to_layout(act_tile[:, 0:12], ttnn.ROW_MAJOR_LAYOUT)
            row["rm: device slice+untilize actions [N,12]"] = med(lambda: ttnn.to_layout(act_tile[:, 0:12], ttnn.ROW_MAJOR_LAYOUT))
            row["rm: download actions [N,12]"] = med(lambda: ttnn.to_torch(act_rm, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)))
            results[n] = row
            print(f"N={n}")
            for k, v in row.items():
                print(f"  {k:44s} {v:9.2f} ms")
        out = REPO / "docs/porting/measurements/probe_upload_device.json"
        out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
