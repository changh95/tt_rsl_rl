# SPDX-License-Identifier: BSD-3-Clause
"""Phase 6 probe: the cross-chip mean (`TTContext.mean_over_chips`: all_gather along every mesh axis + fp32 `ttnn.sum`, ADR-0004 A-1) on a 1xN / 2x4 mesh.

Checks, for gradient-like shapes: (1) the result equals the torch mean over the chips' (different) shards, and is bit-identical on every chip; (2) the same
holds when the op chain is CAPTURED as a trace and replayed with new data (CCL inside a trace: the traced multi-chip path depends on it); (3) eager vs replay cost.

    scripts/tt_run.sh --chips 16,17 --timeout 300 --tag probe_ccl -- python scripts/probe_ccl_trace.py --shape 1x2
    scripts/tt_run.sh --timeout 300 --tag probe_ccl8 -- python scripts/probe_ccl_trace.py --shape 2x4
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.device import open_tray_mesh  # noqa: E402
from rsl_rl_tt.tt_common import TTContext  # noqa: E402

SHAPES = {"matrix 512x256": (512, 256), "matrix 1024x1024": (1024, 1024), "row [1, 256]": (1, 256), "scalar [1, 1]": (1, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1x2")
    args = ap.parse_args()
    rows, cols = (int(v) for v in args.shape.split("x"))
    n = rows * cols
    with open_tray_mesh((rows, cols), fabric_config="FABRIC_1D", trace_region_size=64 * 1024 * 1024) as tm:
        ctx = TTContext.create(tm.mesh_device, mesh_shape=(rows, cols))
        ttnn, mesh = ctx.ttnn, ctx.mesh
        sync = lambda: ttnn.synchronize_device(mesh)
        print(f"mesh {args.shape}: {ctx.n_chips} chips, nodes {sorted(tm.cfg.chips)}")
        torch.manual_seed(0)
        ok = True
        for name, (r, c) in SHAPES.items():
            r_pad, c_pad = max(r, 32), max(c, 32)
            host = torch.randn(n * r_pad, c_pad) * torch.logspace(-2, 1, n).repeat_interleave(r_pad).reshape(-1, 1)  # chips with different gradient scales
            shards = host.reshape(n, r_pad, c_pad)
            ref = shards[:, :r, :c].double().mean(0)

            def local(h):
                x = ctx.upload(h, op="H-6", batch_sharded=True)  # chip i holds rows i*r_pad .. (i+1)*r_pad
                return x[0:r, 0:c] if (r, c) != (r_pad, c_pad) else x

            x = local(host)
            out = ctx.mean_over_chips(x)
            per_chip = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(n, -1, c)[:, :r, :c] if r > 1 else ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(n, -1)[:, :c].reshape(n, 1, c)
            err = float((per_chip[0].double() - ref).abs().max() / ref.abs().max())
            same = all(torch.equal(per_chip[0], per_chip[i]) for i in range(1, n))
            print(f"[eager] {name:18s} max rel err vs torch {err:.2e}; bit-identical on all {n} chips: {same}")
            ok &= err < 1e-6 and same

            # the same chain as a trace: persistent input / output, captured once, replayed with new data
            inp = ctx.upload(host, op="H-6", batch_sharded=True)
            view = (lambda t: t[0:r, 0:c]) if (r, c) != (r_pad, c_pad) else (lambda t: t)
            out_buf = ctx.put(None, ctx.mean_over_chips(view(inp)))  # warm-up 1: adopts the output buffer
            ctx.put(out_buf, ctx.mean_over_chips(view(inp)))  # warm-up 2: programs into the cache
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                ctx.put(out_buf, ctx.mean_over_chips(view(inp)))
            finally:
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            host2 = torch.randn(n * r_pad, c_pad) * 3.0
            ctx.upload_into(inp, host2, op="H-6", batch_sharded=True)
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            got = ttnn.to_torch(out_buf, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            got = got.reshape(n, -1, c)[:, :r, :c] if r > 1 else got.reshape(n, -1)[:, :c].reshape(n, 1, c)
            ref2 = host2.reshape(n, r_pad, c_pad)[:, :r, :c].double().mean(0)
            err2 = float((got[0].double() - ref2).abs().max() / ref2.abs().max())
            same2 = all(torch.equal(got[0], got[i]) for i in range(1, n))
            print(f"[trace] {name:18s} replay with NEW data: max rel err {err2:.2e}; bit-identical on all chips: {same2}")
            ok &= err2 < 1e-6 and same2

            def med(fn, reps=20):
                fn()
                sync()
                t = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    fn()
                    sync()
                    t.append(time.perf_counter() - t0)
                return statistics.median(t) * 1e3

            t_eager = med(lambda: ctx.put(out_buf, ctx.mean_over_chips(view(inp))))
            t_replay = med(lambda: ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False))
            print(f"        {name:18s} eager {t_eager:.3f} ms, replay {t_replay:.3f} ms")
            ttnn.release_trace(mesh, trace_id)
        print("PROBE", "OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
