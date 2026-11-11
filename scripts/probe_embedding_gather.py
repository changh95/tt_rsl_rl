# SPDX-License-Identifier: BSD-3-Clause
"""Phase 5 probe: exact fp32 row gather through bf16 pieces + ttnn.embedding. MEASURES only.

An fp32 value is split EXACTLY into three bf16 values (h1 + h2 + h3 == x: h1 = bf16(x), r1 = x - h1, h2 = bf16(r1), r2 = r1 - h2, h3 = bf16(r2); the
subtractions are exact in fp32), the 3C bf16 columns are gathered with `ttnn.embedding` (a page copy), and the pieces are added back. Checks bit-exactness
and times split / gather / combine separately.

    scripts/tt_run.sh --chips 16 --timeout 600 --tag emb -- python scripts/probe_embedding_gather.py
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
        sync = lambda: ttnn.synchronize_device(mesh)
        for n, m, c in [(98304, 24576, 256), (393216, 98304, 256)]:
            torch.manual_seed(0)
            table = torch.randn(n, c) * torch.logspace(-3, 2, c)  # wide dynamic range per column
            perm = torch.randperm(n)[:m]
            ref = table[perm]
            row = {}
            try:
                x = ttnn.from_torch(table, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
                idx = ttnn.from_torch(perm.to(torch.int64).reshape(1, m), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

                def split(x):
                    h1 = ttnn.typecast(x, ttnn.bfloat16)
                    r1 = ttnn.subtract(x, ttnn.typecast(h1, ttnn.float32))
                    h2 = ttnn.typecast(r1, ttnn.bfloat16)
                    r2 = ttnn.subtract(r1, ttnn.typecast(h2, ttnn.float32))
                    h3 = ttnn.typecast(r2, ttnn.bfloat16)
                    return ttnn.concat([h1, h2, h3], dim=1)

                def combine(g):  # g: [m, 3c] bf16
                    g = ttnn.to_layout(g, ttnn.TILE_LAYOUT) if g.layout != ttnn.TILE_LAYOUT else g
                    parts = [ttnn.typecast(g[:, k * c : (k + 1) * c], ttnn.float32) for k in range(3)]
                    return ttnn.add(ttnn.add(parts[0], parts[1]), parts[2])

                s = split(x)
                sync()
                t0 = time.time()
                s = split(x)
                sync()
                row["split_ms"] = round((time.time() - t0) * 1e3, 2)
                s_rm = ttnn.to_layout(s, ttnn.ROW_MAJOR_LAYOUT)
                sync()
                t0 = time.time()
                s_rm = ttnn.to_layout(s, ttnn.ROW_MAJOR_LAYOUT)
                sync()
                row["to_row_major_ms"] = round((time.time() - t0) * 1e3, 2)
                t0 = time.time()
                g = ttnn.embedding(idx, s_rm, layout=ttnn.ROW_MAJOR_LAYOUT)
                sync()
                row["embedding_ms"] = round((time.time() - t0) * 1e3, 2)
                t0 = time.time()
                g = ttnn.embedding(idx, s_rm, layout=ttnn.ROW_MAJOR_LAYOUT)
                sync()
                row["embedding_ms_second_call"] = round((time.time() - t0) * 1e3, 2)
                g = ttnn.reshape(g, (m, 3 * c))
                y = combine(g)  # warm-up (kernel compile)
                sync()
                t0 = time.time()
                y = combine(g)
                sync()
                row["combine_ms_warm"] = round((time.time() - t0) * 1e3, 2)
                t0 = time.time()
                full = combine(ttnn.reshape(ttnn.embedding(idx, ttnn.to_layout(split(x), ttnn.ROW_MAJOR_LAYOUT), layout=ttnn.ROW_MAJOR_LAYOUT), (m, 3 * c)))
                sync()
                row["whole_gather_pipeline_ms_warm"] = round((time.time() - t0) * 1e3, 2)
                got = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(m, c)
                row["bit_exact"] = bool(torch.equal(got, ref))
                row["mismatched"] = int((got != ref).sum())
                row["max_abs_err"] = float((got - ref).abs().max())
            except Exception as exc:
                row["FAILED"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:220] if str(exc) else ''}"
            out[f"N={n} M={m} C={c}"] = row
            print(f"N={n} M={m} C={c}:", json.dumps(row), flush=True)
    (REPO / "docs" / "porting" / "measurements" / "embedding_gather_probe.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
