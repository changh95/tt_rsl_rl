# SPDX-License-Identifier: BSD-3-Clause
"""Phase 3 probe (survey Q14): device row gather for mini-batch selection. Which op gathers fp32 rows bit-exactly, how fast, up to what size?

    scripts/tt_run.sh --chips 16 --timeout 600 --tag gather -- python scripts/probe_gather.py

Candidates: ttnn.tosa_gather(table[1,N,C], index[1,M]) and ttnn.gather(table[N,C], 0, index[M,C]) with UINT32 / UINT16 indices.
Results go to stdout and docs/porting/measurements/gather_probe.json. MEASURES only.
"""

import json
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.device import open_tray_mesh  # noqa: E402

CASES = [(24576, 6144, 96), (98304, 24576, 192)]


def main() -> None:
    results = {}
    with open_tray_mesh((1, 1)) as tm:
        import ttnn

        mesh = tm.mesh_device

        def up(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh)

        def down(t):
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

        for n, m, c in CASES:
            torch.manual_seed(0)
            table = torch.randn(n, c)
            perm = torch.randperm(n)[:m]
            ref = table[perm]
            table_dev = up(table, ttnn.float32)
            row = {}
            for idx_name, idx_dtype, torch_dtype in (("uint32", ttnn.uint32, torch.int64), ("uint16", ttnn.uint16, torch.int32)):
                if idx_name == "uint16" and n > 65535:
                    row[f"tosa_gather/{idx_name}"] = row[f"gather/{idx_name}"] = "skipped (N > 65535)"
                    continue
                idx_t = perm.to(torch_dtype)

                def tosa():
                    idx_dev = up(idx_t.reshape(1, m), idx_dtype)
                    return ttnn.tosa_gather(ttnn.reshape(table_dev, (1, n, c)), idx_dev)

                def plain():
                    idx_dev = up(idx_t.reshape(m, 1).expand(m, c).contiguous(), idx_dtype)
                    return ttnn.gather(table_dev, 0, idx_dev)

                def tosa_rm():
                    t_rm = ttnn.from_torch(table.reshape(1, n, c), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
                    i_rm = ttnn.from_torch(idx_t.reshape(1, m), dtype=idx_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
                    return ttnn.tosa_gather(t_rm, i_rm)

                def embed():
                    w_rm = ttnn.from_torch(table, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
                    i_rm = ttnn.from_torch(idx_t.reshape(1, m), dtype=idx_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
                    return ttnn.embedding(i_rm, w_rm, layout=ttnn.ROW_MAJOR_LAYOUT)

                variants = [(f"tosa_gather/{idx_name}", tosa), (f"gather/{idx_name}", plain)]
                if idx_name == "uint32":
                    variants += [("tosa_gather_RM/uint32", tosa_rm), ("embedding_RM_fp32/uint32", embed)]
                for name, fn in variants:
                    try:
                        out = fn()
                        ttnn.synchronize_device(mesh)
                        got = down(out).reshape(m, c)
                        exact = bool(torch.equal(got, ref))
                        t0 = time.time()
                        for _ in range(10):
                            out = fn()
                        ttnn.synchronize_device(mesh)
                        row[name] = {"bit_exact": exact, "ms_incl_index_upload": round((time.time() - t0) / 10 * 1e3, 3)}
                    except Exception as exc:
                        row[name] = f"FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
                        traceback.print_exc(limit=1)
            results[f"N={n} M={m} C={c}"] = row
            print(f"N={n} M={m} C={c}:", json.dumps(row), flush=True)
    out = REPO / "docs" / "porting" / "measurements" / "gather_probe.json"
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
