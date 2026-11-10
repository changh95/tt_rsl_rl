# SPDX-License-Identifier: BSD-3-Clause
"""Phase 0 follow-up probe (ONE chip): how many mantissa bits does each fp32 matmul operand really keep, and what multi-pass split
recovers fp32 accuracy? Also: composed variance vs `ttnn.var` (EmpiricalNormalization). MEASURES only; no tolerance is set here.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag noise2 -- python -m pytest -m tt tests/tt/test_noise_floor_split.py -s -q -p no:cacheprovider

First run (test_noise_floor.py): HiFi3/HiFi4 give rel_fro ~1.2e-3 vs a float64 reference, torch fp32 gives 2e-7, and a 3-pass split at 10 bits
only reached 5.7e-4 => the operands are cut shorter than the 10 TF32 mantissa bits. Here the cut is located per operand.
"""

import json
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
REPORT: dict = {}


@pytest.fixture(scope="module")
def mesh():
    from rsl_rl_tt.device import open_tray_mesh

    tm = open_tray_mesh((1, 1))
    yield tm.mesh_device
    tm.close()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "noise_floor_split.json").write_text(json.dumps(REPORT, indent=2, default=str))


def _tt():
    import ttnn

    return ttnn


def up(mesh, t):
    ttnn = _tt()
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)


def down(mesh, t):
    ttnn = _tt()
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def round_bits(x: torch.Tensor, n: int) -> torch.Tensor:
    """Round-to-nearest to n explicit mantissa bits (n <= 22), as an fp32 tensor."""
    if n >= 23:
        return x.clone()
    drop = 23 - n
    bits = x.contiguous().view(torch.int32)
    return ((bits + (1 << (drop - 1))) & ~((1 << drop) - 1)).view(torch.float32)


def pieces(x: torch.Tensor, bits: int, count: int):
    out, rest = [], x.clone()
    for _ in range(count):
        p = round_bits(rest, bits)
        out.append(p)
        rest = rest - p
    return out


def rel(out, ref):
    return float((out.double() - ref.double()).norm() / ref.double().norm())


def cfg():
    ttnn = _tt()
    return ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)


def mm(mesh, a, b):
    ttnn = _tt()
    return down(mesh, ttnn.matmul(up(mesh, a), up(mesh, b), compute_kernel_config=cfg()))


def rec(key, value):
    REPORT[key] = value
    print(f"{key}: {value}")


def test_operand_precision_scan(mesh):
    """Round ONE operand to n bits and the other to 3 bits (exact at any precision >= 3): the error then shows only the tested side."""
    torch.manual_seed(7)
    m, k, n = 1024, 512, 256
    base_a, base_b = torch.randn(m, k), torch.randn(k, n)
    scan = {}
    for side in ("in0", "in1"):
        for bits in (4, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 23):
            if side == "in0":
                a, b = round_bits(base_a, bits), round_bits(base_b, 3)
            else:
                a, b = round_bits(base_a, 3), round_bits(base_b, bits)
            ref = a.double() @ b.double()
            scan[f"{side}@{bits}bits"] = rel(mm(mesh, a, b), ref)
    rec("operand_scan_rel_fro", scan)


def test_split_matmul_recovers_fp32(mesh):
    ttnn = _tt()
    scan = REPORT.get("operand_scan_rel_fro")
    torch.manual_seed(8)
    m, k, n = 4096, 512, 256
    a, b = torch.randn(m, k), torch.randn(k, n)
    ref = a.double() @ b.double()
    rec("split/torch_fp32", rel(a @ b, ref))
    rec("split/plain_HiFi4", rel(mm(mesh, a, b), ref))
    results = {}
    for label, (e0, e1, p0, p1, limit) in {
        "3x8 / 3x8, i+j<=4 (6 matmuls)": (8, 8, 3, 3, 4),
        "3x8 / 2x12, shift<24 (5 matmuls)": (8, 12, 3, 2, None),
        "2x12 / 3x8, shift<24 (5 matmuls)": (12, 8, 2, 3, None),
        "2x12 / 2x12, all 4 (4 matmuls)": (12, 12, 2, 2, None),
        "2x11 / 2x11, i+j<=3 (3 matmuls)": (11, 11, 2, 2, 3),
        "3x7 / 3x7, i+j<=4 (6 matmuls)": (7, 7, 3, 3, 4),
        "4x6 / 4x6, i+j<=5 (10 matmuls)": (6, 6, 4, 4, 5),
    }.items():
        pa, pb = pieces(a, e0, p0), pieces(b, e1, p1)
        total, used = None, 0
        for i, ai in enumerate(pa):
            for j, bj in enumerate(pb):
                keep = (i + j + 2 <= limit) if limit is not None else ((i * e0 + j * e1) < 24)
                if not keep:
                    continue
                term = ttnn.matmul(up(mesh, ai), up(mesh, bj), compute_kernel_config=cfg())
                total = term if total is None else ttnn.add(total, term)
                used += 1
        results[label] = {"rel_fro": rel(down(mesh, total), ref), "matmuls": used}
        print(label, results[label])
    rec("split/results", results)
    del scan


def test_variance_variants(mesh):
    ttnn = _tt()
    out = {}
    for n in (4096, 65536):
        for name, gen in {
            "normal": lambda s: torch.randn(*s),
            "offset100": lambda s: torch.randn(*s) + 100.0,  # a normalizer sees non-zero-mean observations
            "scale1e-3": lambda s: torch.randn(*s) * 1e-3,
        }.items():
            torch.manual_seed(9)
            x = gen((n, 64))
            ref = x.double().var(0, unbiased=False)
            t = up(mesh, x)
            row = {}
            row["ttnn.var"] = rel(down(mesh, ttnn.var(t, dim=0)).reshape(-1)[:64], ref)
            mean = ttnn.mean(t, dim=0, keepdim=True)
            two_pass = ttnn.mean(ttnn.square(ttnn.subtract(t, mean)), dim=0)
            row["two_pass(mean((x-mean)^2))"] = rel(down(mesh, two_pass).reshape(-1)[:64], ref)
            one_pass = ttnn.subtract(ttnn.mean(ttnn.square(t), dim=0), ttnn.square(ttnn.mean(t, dim=0)))
            row["one_pass(E[x^2]-E[x]^2)"] = rel(down(mesh, one_pass).reshape(-1)[:64], ref)
            row["torch_fp32"] = rel(x.var(0, unbiased=False), ref)
            out[f"{n}x64/{name}"] = row
            print(f"var {n}x64/{name}:", row)
    rec("variance", out)


def mm_cfg(mesh, a, b, *, fp32_acc=True, packer=True):
    ttnn = _tt()
    c = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=packer)
    return down(mesh, ttnn.matmul(up(mesh, a), up(mesh, b), compute_kernel_config=c))


def test_accumulation_floor(mesh):
    """With both operands exact at 4 bits the product terms are exact; any remaining error is accumulation / pack / reload."""
    ttnn = _tt()
    out = {}
    for k in (32, 64, 128, 256, 512, 2048):
        torch.manual_seed(11)
        a, b = round_bits(torch.randn(256, k), 4), round_bits(torch.randn(k, 256), 4)
        ref = a.double() @ b.double()
        row = {}
        for acc, pk in ((True, True), (True, False), (False, True)):
            row[f"fp32acc={acc},packer_l1_acc={pk}"] = rel(mm_cfg(mesh, a, b, fp32_acc=acc, packer=pk), ref)
        # K chunked into 32-wide slices, partial products added with ttnn.add (fp32)
        total = None
        for s in range(0, k, 32):
            term = ttnn.matmul(up(mesh, a[:, s : s + 32].contiguous()), up(mesh, b[s : s + 32, :].contiguous()), compute_kernel_config=cfg())
            total = term if total is None else ttnn.add(total, term)
        row["K-chunked(32)+ttnn.add"] = rel(down(mesh, total), ref)
        out[f"K={k}"] = row
        print(f"floor K={k}:", row)
    rec("accumulation_floor", out)


def test_chunked_split_full_precision_operands(mesh):
    ttnn = _tt()
    torch.manual_seed(12)
    m, k, n = 2048, 512, 256
    a, b = torch.randn(m, k), torch.randn(k, n)
    ref = a.double() @ b.double()
    res = {"plain_HiFi4": rel(mm(mesh, a, b), ref)}
    for label, (e0, e1, p0, p1, limit) in {"3x8/3x8 (6 mm/chunk)": (8, 8, 3, 3, 4), "2x10/2x10 all (4 mm/chunk)": (10, 10, 2, 2, 4), "2x10/3x7 (6 mm/chunk)": (10, 7, 2, 3, 5)}.items():
        for chunk in (32, 128):
            pa, pb = pieces(a, e0, p0), pieces(b, e1, p1)
            total, used = None, 0
            for s in range(0, k, chunk):
                for i, ai in enumerate(pa):
                    for j, bj in enumerate(pb):
                        if i + j + 2 > limit:
                            continue
                        term = ttnn.matmul(up(mesh, ai[:, s : s + chunk].contiguous()), up(mesh, bj[s : s + chunk, :].contiguous()), compute_kernel_config=cfg())
                        total = term if total is None else ttnn.add(total, term)
                        used += 1
            res[f"{label} chunk={chunk}"] = {"rel_fro": rel(down(mesh, total), ref), "matmuls_total": used}
            print(label, chunk, res[f"{label} chunk={chunk}"])
    rec("chunked_split", res)
