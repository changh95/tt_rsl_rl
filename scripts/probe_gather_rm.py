# SPDX-License-Identifier: BSD-3-Clause
"""Phase 3 probe: kernel-only timing and exactness of the row-major tosa_gather (table/index uploaded once). MEASURES only.

    scripts/tt_run.sh --chips 16 --timeout 600 --tag gather_rm -- python scripts/probe_gather_rm.py
"""

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.device import open_tray_mesh  # noqa: E402


def main() -> None:
    out = {}
    with open_tray_mesh((1, 1)) as tm:
        import ttnn

        mesh = tm.mesh_device
        for n, m, c in [(24576, 6144, 96), (98304, 24576, 192), (98304, 24576, 32), (98304, 24576, 256)]:
            torch.manual_seed(0)
            table = torch.randn(n, c)
            perm = torch.randperm(n)[:m]
            ref = table[perm]
            row = {}
            for layout_name, layout in (("TILE", ttnn.TILE_LAYOUT), ("ROW_MAJOR", ttnn.ROW_MAJOR_LAYOUT)):
                try:
                    t = ttnn.from_torch(table.reshape(1, n, c), dtype=ttnn.float32, layout=layout, device=mesh)
                    i = ttnn.from_torch(perm.reshape(1, m), dtype=ttnn.uint32, layout=layout, device=mesh)
                    o = ttnn.tosa_gather(t, i)
                    ttnn.synchronize_device(mesh)
                    got = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(-1, c)
                    mism = int((got != ref).sum())
                    t0 = time.time()
                    for _ in range(10):
                        o = ttnn.tosa_gather(t, i)
                    ttnn.synchronize_device(mesh)
                    ms = (time.time() - t0) / 10 * 1e3
                    row[layout_name] = {"mismatched_elements": mism, "kernel_ms": round(ms, 3), "out_shape": list(got.shape)}
                except Exception as exc:
                    row[layout_name] = f"FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
            out[f"N={n} M={m} C={c}"] = row
            print(f"N={n} M={m} C={c}:", json.dumps(row), flush=True)
    (REPO / "docs" / "porting" / "measurements" / "gather_rm_probe.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
