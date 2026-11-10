# SPDX-License-Identifier: BSD-3-Clause
"""Phase 0 noise-floor measurements on ONE chip (TTNN_OP_SURVEY.md Q9, Q10, Q11, Q12). They MEASURE; they set no tolerance.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag noise -- nice -n 10 python -m pytest -m tt tests/tt/test_noise_floor.py -s -q

All numbers go to `.run/results/noise_floor.json` and from there into docs/porting/TOLERANCES.md (BRIEF rule 2-9: tolerances come from
measured noise floors with evidence, never from "make the test pass"). An op that raises is recorded as unsupported, not hidden.
The only asserts are "finite" sanity bounds.
"""

import json
import traceback
from pathlib import Path

import numpy as np
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
    (RESULTS / "noise_floor.json").write_text(json.dumps(REPORT, indent=2, default=str))


def _tt():
    import ttnn

    return ttnn


def up(mesh, t):
    ttnn = _tt()
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)


def down(mesh, t):
    ttnn = _tt()
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def ulps(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Max error in units of the fp32 spacing of the (float64-computed, fp32-rounded) reference."""
    r = ref.float().numpy()
    return float(np.max(np.abs(out.double().numpy() - ref.double().numpy()) / np.spacing(np.abs(r)).astype(np.float64).clip(min=1e-45)))


def stats(out: torch.Tensor, ref: torch.Tensor) -> dict:
    d = (out.double() - ref.double()).abs()
    denom = ref.double().abs().clamp(min=1e-30)
    return {
        "max_abs": float(d.max()),
        "rel_fro": float(d.norm() / ref.double().norm().clamp(min=1e-30)),
        "max_rel": float((d / denom).max()),
        "max_ulp": ulps(out, ref),
    }


def record(section: str, key: str, value) -> None:
    REPORT.setdefault(section, {})[key] = value
    print(f"[{section}] {key}: {value}")


def attempt(section: str, key: str, fn) -> None:
    try:
        record(section, key, fn())
    except Exception as exc:  # unsupported op / arg: record, do not hide, do not fail the run
        record(section, key, {"unsupported": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"})
        traceback.print_exc(limit=2)


# ---------------------------------------------------------------------------------------------------------------------------------
# Q10: matmul
# ---------------------------------------------------------------------------------------------------------------------------------
def _cfg(fid, fp32_acc=True, approx=False, packer=True):
    ttnn = _tt()
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32_acc, packer_l1_acc=packer)


def _tf32_split(x: torch.Tensor):
    """hi = x rounded to 10 mantissa bits (round-to-nearest-even on the integer view), lo = x - hi (exact in fp32)."""
    bits = x.contiguous().view(torch.int32)
    rounded = (bits + 0x1000) & ~0x1FFF
    hi = rounded.view(torch.float32)
    return hi, x - hi


def test_matmul_noise_floor(mesh):
    ttnn = _tt()
    shapes = [(512, 512, 512), (4096, 48, 512), (4096, 512, 256), (16384, 256, 128), (4096, 1024, 512)]
    gens = {
        "normal": lambda *s: torch.randn(*s),
        "heavy_tail": lambda *s: torch.distributions.StudentT(3.0).sample(s).clamp(-50, 50),
    }
    fids = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi3": ttnn.MathFidelity.HiFi3, "HiFi4": ttnn.MathFidelity.HiFi4}
    for (m, k, n) in shapes:
        for gname, gen in gens.items():
            torch.manual_seed(10)
            a, b = gen(m, k), gen(k, n)
            ref = a.double() @ b.double()
            tag = f"{m}x{k}x{n}/{gname}"
            record("matmul", f"{tag}/torch_fp32", stats(a @ b, ref))
            ta, tb = up(mesh, a), up(mesh, b)
            attempt("matmul", f"{tag}/default_cfg", lambda: stats(down(mesh, ttnn.matmul(ta, tb)), ref))
            for fname, fid in fids.items():
                attempt("matmul", f"{tag}/{fname}+fp32acc", lambda fid=fid: stats(down(mesh, ttnn.matmul(ta, tb, compute_kernel_config=_cfg(fid))), ref))
            attempt("matmul", f"{tag}/HiFi4_no_fp32acc", lambda: stats(down(mesh, ttnn.matmul(ta, tb, compute_kernel_config=_cfg(ttnn.MathFidelity.HiFi4, fp32_acc=False))), ref))

            # contingency: 3-pass hi/lo split (a_hi b_hi + a_hi b_lo + a_lo b_hi), each pass a plain HiFi4 fp32 matmul
            def split3():
                ah, al = _tf32_split(a)
                bh, bl = _tf32_split(b)
                c = _cfg(ttnn.MathFidelity.HiFi4)
                t_ahbh = ttnn.matmul(up(mesh, ah), up(mesh, bh), compute_kernel_config=c)
                t_ahbl = ttnn.matmul(up(mesh, ah), up(mesh, bl), compute_kernel_config=c)
                t_albh = ttnn.matmul(up(mesh, al), up(mesh, bh), compute_kernel_config=c)
                total = ttnn.add(ttnn.add(t_ahbh, t_ahbl), t_albh)
                return stats(down(mesh, total), ref)

            attempt("matmul", f"{tag}/split3_HiFi4", split3)


# ---------------------------------------------------------------------------------------------------------------------------------
# Q11 / Q12: elementwise
# ---------------------------------------------------------------------------------------------------------------------------------
def _grid(lo, hi, n=64 * 1024):
    return torch.linspace(lo, hi, n, dtype=torch.float32).reshape(1, 1, 256, -1)


def test_eltwise_noise_floor(mesh):
    ttnn = _tt()
    cases = {
        # name: (ttnn fn(x), torch fp64 fn, input range)
        "exp[-20,20]": (lambda x: ttnn.exp(x), torch.exp, (-20.0, 20.0)),
        "exp[-5,2](log_std)": (lambda x: ttnn.exp(x), torch.exp, (-5.0, 2.0)),
        "exp[-1,1](ratio)": (lambda x: ttnn.exp(x), torch.exp, (-1.0, 1.0)),
        "log[1e-6,1e3]": (lambda x: ttnn.log(x), torch.log, (1e-6, 1e3)),
        "sqrt[1e-6,1e3]": (lambda x: ttnn.sqrt(x), torch.sqrt, (1e-6, 1e3)),
        "rsqrt[1e-6,1e3]": (lambda x: ttnn.rsqrt(x), torch.rsqrt, (1e-6, 1e3)),
        "reciprocal[0.01,100]": (lambda x: ttnn.reciprocal(x), torch.reciprocal, (0.01, 100.0)),
        "tanh[-10,10]": (lambda x: ttnn.tanh(x), torch.tanh, (-10.0, 10.0)),
        "softplus[-20,20]": (lambda x: ttnn.softplus(x), torch.nn.functional.softplus, (-20.0, 20.0)),
        "elu[-20,20]": (lambda x: ttnn.elu(x, alpha=1.0), lambda t: torch.nn.functional.elu(t, alpha=1.0), (-20.0, 20.0)),
        "elu_composed[-20,20]": (
            lambda x: ttnn.where(ttnn.gtz(x), x, ttnn.multiply(ttnn.expm1(x), 1.0)) if hasattr(ttnn, "expm1") else ttnn.where(ttnn.gtz(x), x, ttnn.subtract(ttnn.exp(x), 1.0)),
            lambda t: torch.nn.functional.elu(t, alpha=1.0),
            (-20.0, 20.0),
        ),
    }
    for name, (fn, ref_fn, (lo, hi)) in cases.items():
        x = _grid(lo, hi)
        ref = ref_fn(x.double())
        attempt("eltwise", name, lambda fn=fn: stats(down(mesh, fn(up(mesh, x))), ref))
    # exp at and beyond the fp32 overflow edge, large negative values
    for name, vals in {"exp@88": [87.0, 88.0, 88.7, 89.0], "exp@-87": [-80.0, -87.0, -88.0, -104.0]}.items():
        x = torch.tensor(vals, dtype=torch.float32).reshape(1, 1, 1, -1).repeat(1, 1, 32, 8)[..., : len(vals) * 8][..., : 32]
        attempt("eltwise", name, lambda x=x: down(mesh, ttnn.exp(up(mesh, x)))[0, 0, 0, : len(vals)].tolist())
    for name in ("exp", "tanh", "softplus", "elu"):
        fn = getattr(ttnn, name)
        x = _grid(-20.0, 20.0)
        ref = getattr(torch, name, None)
        ref_t = ref(x.double()) if ref is not None else torch.nn.functional.__dict__[name](x.double())

        def approx(fn=fn, x=x, ref_t=ref_t):
            return stats(down(mesh, fn(up(mesh, x), fast_and_approximate_mode=True)), ref_t)

        attempt("eltwise", f"{name}[-20,20]/fast_and_approximate_mode=True", approx)


def test_elu_backward_noise_floor(mesh):
    ttnn = _tt()
    x = torch.cat([torch.linspace(-20, 20, 64 * 1024 - 2), torch.tensor([0.0, -0.0])]).reshape(1, 1, 256, -1).float()
    g = torch.randn_like(x)
    xr = x.clone().requires_grad_(True)
    torch.nn.functional.elu(xr, alpha=1.0).backward(g)
    ref = xr.grad.detach()
    attempt("elu_bw", "ttnn.elu_bw", lambda: stats(down(mesh, ttnn.elu_bw(up(mesh, g), up(mesh, x), alpha=1.0)[0]), ref))
    comp = lambda: stats(down(mesh, ttnn.where(ttnn.gtz(up(mesh, x)), up(mesh, g), ttnn.multiply(up(mesh, g), ttnn.exp(up(mesh, x))))), ref)
    attempt("elu_bw", "composed(gtz ? g : g*exp(x))", comp)
    # at x == 0 exactly: torch gives 1 (x > 0 is False -> alpha*exp(0) = 1) for alpha = 1
    x0 = torch.zeros(1, 1, 32, 32)
    g0 = torch.ones_like(x0)
    attempt("elu_bw", "x==0 ttnn.elu_bw (torch: 1.0)", lambda: float(down(mesh, ttnn.elu_bw(up(mesh, g0), up(mesh, x0), alpha=1.0)[0]).mean()))


# ---------------------------------------------------------------------------------------------------------------------------------
# Q9: reductions
# ---------------------------------------------------------------------------------------------------------------------------------
def test_reduction_noise_floor(mesh):
    ttnn = _tt()
    for n in (4096, 4000, 4099, 65536):
        torch.manual_seed(5)
        x = torch.randn(n, 64)
        ref = x.double().sum(0)
        t = up(mesh, x)
        attempt("reduce", f"sum_dim0[{n},64]/default", lambda: stats(down(mesh, ttnn.sum(t, dim=0)).reshape(-1)[:64], ref))
        attempt("reduce", f"mean_dim0[{n},64]/default", lambda: stats(down(mesh, ttnn.mean(t, dim=0)).reshape(-1)[:64], x.double().mean(0)))
        attempt("reduce", f"var_dim0[{n},64]/default", lambda: stats(down(mesh, ttnn.var(t, dim=0)).reshape(-1)[:64], x.double().var(0, unbiased=False)))
        attempt("reduce", f"max_dim0[{n},64]/default", lambda: stats(down(mesh, ttnn.max(t, dim=0)).reshape(-1)[:64], x.double().max(0).values))
    # global gradient norm: sqrt(sum(g^2)) over a flat buffer
    for size in (50_000, 500_000, 3_000_000):
        torch.manual_seed(6)
        g = torch.randn(size // 1024 * 1024).reshape(-1, 1024) * 1e-2
        ref = g.double().pow(2).sum().sqrt()
        tg = up(mesh, g)

        def norm_default():
            s = ttnn.sum(ttnn.square(tg))
            return stats(down(mesh, ttnn.sqrt(s)).reshape(-1)[:1], ref.reshape(1))

        attempt("reduce", f"global_norm[{g.numel()}]/sum(square)->sqrt", norm_default)
