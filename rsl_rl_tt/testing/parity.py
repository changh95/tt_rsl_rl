# SPDX-License-Identifier: BSD-3-Clause
"""Parity utilities: compare a TT result with the upstream torch reference and, on failure, report the worst element.

Everything is evaluated in float64. A `Tol` lists the criteria that apply to a quantity; the registry `TOLERANCES` mirrors the table in
`docs/porting/TOLERANCES.md` (a test checks that every number below appears there). Per BRIEF 2-9 these values are never loosened to
make a test pass: a change needs a root-cause analysis recorded in `docs/porting/DECISIONS.md` first.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tol:
    """Acceptance criteria; a criterion left at None is not checked. `exact` means bitwise equality."""

    name: str
    rel_fro: float | None = None  # ||out-ref||_F / ||ref||_F
    max_rel: float | None = None  # max |out-ref| / max(|ref|, rel_floor * max|ref|)
    max_abs: float | None = None
    max_ulp: float | None = None  # fp32 ULPs of the reference
    pcc_min: float | None = None
    exact: bool = False
    rel_floor: float = 1e-3
    doc_tokens: tuple[str, ...] = ()  # strings that must appear in docs/porting/TOLERANCES.md (checked by a test)


# Mirrors docs/porting/TOLERANCES.md (measured noise floors x ~3, Phase 0).
TOLERANCES: dict[str, Tol] = {
    "transfer_fp32": Tol("fp32 host<->device transfer", exact=True, doc_tokens=("torch.equal",)),
    "matmul_fp32": Tol("fp32 matmul/linear (HiFi4 + fp32 acc)", rel_fro=4e-3, doc_tokens=("**4e-3**",)),
    "eltwise_fp32": Tol("exp/log/sqrt/rsqrt/reciprocal/tanh/elu/elu_bw", max_rel=5e-7, max_ulp=6, doc_tokens=("**5e-7**",)),
    "row_sum_fp32": Tol("row sum / mean", rel_fro=1e-6, doc_tokens=("**1e-6**",)),
    "variance_fp32": Tol("variance (two-pass)", rel_fro=2e-7, doc_tokens=("**2e-7**",)),
    "global_norm_fp32": Tol("global gradient norm", rel_fro=1e-7, doc_tokens=("**1e-7**",)),
    "max_exact": Tol("max over rows", exact=True),
    "cross_chip_sum_gather": Tol("cross-chip sum, gather + ttnn.sum", rel_fro=2e-7, doc_tokens=("**2e-7**",)),
    "cross_chip_sum_all_reduce": Tol("cross-chip ttnn.all_reduce (fallback)", rel_fro=2e-3, doc_tokens=("**2e-3**",)),
}


@dataclass
class ParityReport:
    name: str
    shape: tuple[int, ...]
    max_abs: float
    max_rel: float
    rel_fro: float
    pcc: float
    max_ulp: float
    n_nonfinite: int
    worst_index: tuple[int, ...] | None
    worst_out: float | None
    worst_ref: float | None
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def __str__(self) -> str:
        head = f"parity[{self.name}] {'PASS' if self.passed else 'FAIL'} shape={self.shape}"
        nums = (
            f"max_abs={self.max_abs:.3e} max_rel={self.max_rel:.3e} rel_fro={self.rel_fro:.3e} pcc={self.pcc:.8f} "
            f"max_ulp={self.max_ulp:.3g} nonfinite={self.n_nonfinite}"
        )
        lines = [head, "  " + nums]
        if self.worst_index is not None:
            lines.append(f"  worst element {self.worst_index}: out={self.worst_out!r} ref={self.worst_ref!r}")
        lines += [f"  violated: {f}" for f in self.failures]
        return "\n".join(lines)


def _to64(x) -> torch.Tensor:
    return torch.as_tensor(x).detach().to("cpu").to(torch.float64)


def pcc(out: torch.Tensor, ref: torch.Tensor) -> float:
    o, r = out.flatten(), ref.flatten()
    if o.numel() < 2:
        return 1.0 if torch.equal(o, r) else 0.0
    o, r = o - o.mean(), r - r.mean()
    denom = o.norm() * r.norm()
    if denom == 0:
        return 1.0 if torch.equal(out, ref) else 0.0
    return float((o @ r) / denom)


def compare(out, ref, tol: Tol | str, name: str = "") -> ParityReport:
    """Compare `out` (TT result brought to the host) with `ref` (upstream torch) under `tol` (a `Tol` or a key of TOLERANCES)."""
    tol = TOLERANCES[tol] if isinstance(tol, str) else tol
    o, r = _to64(out), _to64(ref)
    name = name or tol.name
    if o.shape != r.shape:
        return ParityReport(name, tuple(o.shape), float("inf"), float("inf"), float("inf"), 0.0, float("inf"), 0, None, None, None, [f"shape {tuple(o.shape)} != reference {tuple(r.shape)}"])

    finite_o, finite_r = torch.isfinite(o), torch.isfinite(r)
    # a non-finite value on either side is only acceptable where both sides hold the same one (inf == inf, nan with nan)
    same_nonfinite = (~finite_r) & ((o == r) | (torch.isnan(o) & torch.isnan(r)))
    bad = ((~finite_o) | (~finite_r)) & ~same_nonfinite
    n_nonfinite = int(bad.sum())
    ok = finite_o & finite_r
    diff = torch.where(ok, (o - r).abs(), torch.zeros_like(o))
    ref_abs = torch.where(ok, r.abs(), torch.zeros_like(r))
    scale = float(ref_abs.max()) if ref_abs.numel() else 0.0
    denom = torch.maximum(ref_abs, torch.full_like(ref_abs, tol.rel_floor * scale if scale > 0 else 1.0))
    rel_elem = diff / denom
    fro_ref = float(torch.linalg.vector_norm(torch.where(ok, r, torch.zeros_like(r))))
    rel_fro = float(torch.linalg.vector_norm(diff)) / fro_ref if fro_ref > 0 else (0.0 if float(diff.max() if diff.numel() else 0) == 0 else float("inf"))
    spacing = torch.from_numpy(np.spacing(np.abs(r.float().numpy())).astype(np.float64))
    ulp = diff / spacing.clamp(min=1e-45)
    worst = int(torch.argmax(diff.flatten())) if diff.numel() else 0
    worst_index = tuple(int(i) for i in np.unravel_index(worst, tuple(o.shape))) if o.dim() > 0 else ()
    rep = ParityReport(
        name=name,
        shape=tuple(o.shape),
        max_abs=float(diff.max()) if diff.numel() else 0.0,
        max_rel=float(rel_elem.max()) if rel_elem.numel() else 0.0,
        rel_fro=rel_fro,
        pcc=pcc(torch.where(ok, o, torch.zeros_like(o)), torch.where(ok, r, torch.zeros_like(r))),
        max_ulp=float(ulp.max()) if ulp.numel() else 0.0,
        n_nonfinite=n_nonfinite,
        worst_index=worst_index,
        worst_out=float(o.flatten()[worst]) if o.numel() else None,
        worst_ref=float(r.flatten()[worst]) if r.numel() else None,
    )
    if n_nonfinite:
        rep.failures.append(f"{n_nonfinite} element(s) with a non-finite value that the other side does not share")
    if tol.exact and not torch.equal(o, r):
        rep.failures.append("not bitwise equal")
    for label, got, limit in (("rel_fro", rep.rel_fro, tol.rel_fro), ("max_rel", rep.max_rel, tol.max_rel), ("max_abs", rep.max_abs, tol.max_abs), ("max_ulp", rep.max_ulp, tol.max_ulp)):
        if limit is not None and not got <= limit:
            rep.failures.append(f"{label} {got:.3e} > {limit:.3e}")
    if tol.pcc_min is not None and not rep.pcc >= tol.pcc_min:
        rep.failures.append(f"pcc {rep.pcc:.8f} < {tol.pcc_min}")
    return rep


def assert_parity(out, ref, tol: Tol | str, name: str = "") -> ParityReport:
    rep = compare(out, ref, tol, name)
    if not rep.passed:
        raise AssertionError("\n" + str(rep))
    return rep
