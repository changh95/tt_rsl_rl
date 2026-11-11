# SPDX-License-Identifier: BSD-3-Clause
"""Phase 5 probe: how trace capture/replay behaves in ttnn 0.81.1 for the patterns the traced PPO needs. MEASURES / explores only.

    scripts/tt_run.sh --chips 16 --timeout 600 --tag trace -- python scripts/probe_trace.py

Checks: (1) persistent input updated by copy_host_to_device_tensor, output written in place with ttnn.copy; replay == eager; (2) allocations and eager work
between replays do not corrupt the trace; (3) in-place update of a persistent parameter (param += delta) accumulates correctly over replays; (4) scalar device
tensors updated per replay (Adam-style); (5) timing: eager vs replay of a ~120-op chain.
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
    res = {}
    with open_tray_mesh((1, 1), trace_region_size=64 * 1024 * 1024) as tm:
        import ttnn

        mesh = tm.mesh_device
        cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        host_t = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)  # a HOST tensor (no device)
        dn = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        torch.manual_seed(0)
        rows, k = 2048, 64
        w_t = torch.randn(k, k) * 0.2
        w = up(w_t)
        x = up(torch.zeros(rows, k))  # persistent input
        out = up(torch.zeros(rows, k))  # persistent output
        scale = up(torch.ones(1, 1))  # persistent scalar input
        param = up(torch.zeros(k, k))  # persistent parameter updated in place

        def step():
            y = ttnn.elu(ttnn.linear(x, w, compute_kernel_config=cfg), alpha=1.0)
            z = ttnn.multiply(y, scale)
            ttnn.copy(z, out)  # in-place write of the output buffer
            delta = ttnn.matmul(x, z, transpose_a=True, compute_kernel_config=cfg)  # [k, k]
            ttnn.copy(ttnn.add(param, ttnn.multiply(delta, 1e-3)), param)  # in-place parameter update

        # eager reference on the host
        def torch_step(xt, s, p):
            z = torch.nn.functional.elu(xt @ w_t) * s
            return z, p + 1e-3 * (xt.t() @ z)

        def rel(a, b):
            return float((a.double() - b.double()).norm() / b.double().norm())

        # warm-up (compiles the kernels) on zeros, then reset the state it changed
        step()
        ttnn.synchronize_device(mesh)
        ttnn.copy(up(torch.zeros(k, k)), param)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        step()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh)

        p_ref = torch.zeros(k, k)
        errs = []
        for it in range(5):
            xt, s = torch.randn(rows, k), 1.0 + 0.1 * it
            ttnn.copy_host_to_device_tensor(host_t(xt), x)
            ttnn.copy_host_to_device_tensor(host_t(torch.full((1, 1), s)), scale)
            if it == 2:  # (2) allocate / free tensors and run unrelated eager ops between replays
                junk = [up(torch.randn(rows, 512)) for _ in range(8)]
                _ = ttnn.add(junk[0], junk[1])
                del junk
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            z_ref, p_ref = torch_step(xt, s, p_ref)
            errs.append({"it": it, "out_rel": rel(dn(out), z_ref), "param_rel": rel(dn(param), p_ref)})
        res["correctness"] = errs
        # (5) timing: a chain of ~120 small ops, eager vs replay
        a = up(torch.randn(rows, k))
        buf = up(torch.zeros(rows, k))

        def chain():
            t = a
            for _ in range(40):
                t = ttnn.multiply(ttnn.add(ttnn.elu(t, alpha=1.0), 0.01), 0.99)
            ttnn.copy(t, buf)

        chain()
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        for _ in range(5):
            chain()
        ttnn.synchronize_device(mesh)
        eager_ms = (time.time() - t0) / 5 * 1e3
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        chain()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        for _ in range(5):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        replay_ms = (time.time() - t0) / 5 * 1e3
        res["timing_120_ops"] = {"eager_ms": round(eager_ms, 2), "replay_ms": round(replay_ms, 2)}
        ttnn.release_trace(mesh, tid)
        ttnn.release_trace(mesh, trace_id)
    print(json.dumps(res, indent=1))
    (REPO / "docs" / "porting" / "measurements" / "trace_probe.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
