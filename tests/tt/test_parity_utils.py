# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests of rsl_rl_tt/testing/parity.py (BRIEF Phase 1: max abs/rel error, PCC, worst-element report)."""

import re
import torch
from pathlib import Path

import pytest

from rsl_rl_tt.testing.parity import TOLERANCES, Tol, assert_parity, compare, pcc

TOLERANCES_MD = Path(__file__).resolve().parents[2] / "docs" / "porting" / "TOLERANCES.md"


def test_identical_tensors_pass_every_tolerance():
    x = torch.randn(64, 32)
    for key in TOLERANCES:
        rep = compare(x, x.clone(), key)
        assert rep.passed and rep.max_abs == 0 and rep.rel_fro == 0 and rep.pcc == pytest.approx(1.0) and rep.n_nonfinite == 0, str(rep)


def test_metrics_are_computed_in_float64_against_the_reference():
    ref = torch.tensor([1.0, 2.0, 4.0, 8.0])
    out = ref * (1 + 1e-3)
    rep = compare(out, ref, Tol("t", rel_fro=2e-3))
    assert rep.rel_fro == pytest.approx(1e-3, rel=1e-4) and rep.max_rel == pytest.approx(1e-3, rel=1e-4)
    assert rep.max_abs == pytest.approx(8e-3, rel=1e-3) and rep.worst_index == (3,) and rep.passed


def test_failure_reports_the_worst_element_and_every_violated_criterion():
    ref = torch.ones(8, 16)
    out = ref.clone()
    out[5, 11] = 1.5
    rep = compare(out, ref, "matmul_fp32", name="layer0")
    assert not rep.passed and rep.worst_index == (5, 11) and rep.worst_out == 1.5 and rep.worst_ref == 1.0
    text = str(rep)
    assert "FAIL" in text and "layer0" in text and "worst element (5, 11)" in text and "rel_fro" in text
    with pytest.raises(AssertionError, match=r"worst element \(5, 11\)"):
        assert_parity(out, ref, "matmul_fp32", name="layer0")


def test_matmul_tolerance_accepts_the_measured_noise_floor_and_rejects_order_one_bugs():
    torch.manual_seed(0)
    ref = torch.randn(256, 256, dtype=torch.float64)
    floor = ref * (1 + 1.2e-3 * torch.randn_like(ref))  # about the measured fp32 matmul noise (1.17e-3 rel_fro)
    assert compare(floor, ref, "matmul_fp32").passed
    assert not compare(ref.t().contiguous(), ref, "matmul_fp32").passed  # wrong layout
    assert not compare(ref * 1.05, ref, "matmul_fp32").passed  # 5 % scale error
    assert not compare(ref[:, torch.randperm(256)], ref, "matmul_fp32").passed


def test_eltwise_tolerance_is_tight_in_ulps():
    x = torch.linspace(-5, 5, 4096)
    ref = torch.exp(x.double())
    ok = ref.float()
    ok_plus_3ulp = torch.nextafter(torch.nextafter(torch.nextafter(ok, torch.tensor(float("inf"))), torch.tensor(float("inf"))), torch.tensor(float("inf")))
    assert compare(ok_plus_3ulp, ref, "eltwise_fp32").passed
    assert not compare(ref.float() * (1 + 1e-5), ref, "eltwise_fp32").passed


def test_exact_means_bitwise():
    x = torch.randn(10)
    assert compare(x, x.clone(), "transfer_fp32").passed
    y = x.clone()
    y[3] = torch.nextafter(y[3], torch.tensor(float("inf")))
    rep = compare(y, x, "transfer_fp32")
    assert not rep.passed and "not bitwise equal" in str(rep)


def test_shape_mismatch_and_nonfinite_are_failures():
    assert "shape" in str(compare(torch.zeros(2, 3), torch.zeros(3, 2), "matmul_fp32"))
    ref = torch.ones(4)
    out = ref.clone()
    out[1] = float("nan")
    rep = compare(out, ref, "matmul_fp32")
    assert not rep.passed and rep.n_nonfinite == 1
    inf_ref = torch.tensor([1.0, float("inf")])
    assert compare(inf_ref.clone(), inf_ref, "matmul_fp32").passed  # exp overflow: inf where the reference is inf is fine
    assert not compare(torch.tensor([1.0, 5.0]), inf_ref, "matmul_fp32").passed  # finite where the reference overflowed


def test_pcc_edge_cases():
    a = torch.arange(10.0)
    assert pcc(a, 3 * a + 1) == pytest.approx(1.0)
    assert pcc(a, -a) == pytest.approx(-1.0)
    assert pcc(torch.ones(5), torch.ones(5)) == 1.0 and pcc(torch.ones(5), torch.zeros(5)) == 0.0
    assert pcc(torch.tensor([1.0]), torch.tensor([1.0])) == 1.0


def test_pcc_criterion():
    ref = torch.randn(1000)
    assert compare(ref + 0.01 * torch.randn(1000), ref, Tol("p", pcc_min=0.99)).passed
    assert not compare(torch.randn(1000), ref, Tol("p", pcc_min=0.99)).passed


def test_registry_numbers_appear_in_tolerances_md():
    text = TOLERANCES_MD.read_text()
    for key, tol in TOLERANCES.items():
        for token in tol.doc_tokens:
            assert token in text, f"{key}: {token!r} not found in docs/porting/TOLERANCES.md (the registry and the table must agree)"
    # and the numeric limits themselves are the ones the table prints
    nums = {key: re.findall(r"\*\*([0-9.e-]+)\*\*", " ".join(t for t in tol.doc_tokens)) for key, tol in TOLERANCES.items()}
    assert nums["matmul_fp32"] == ["4e-3"] and TOLERANCES["matmul_fp32"].rel_fro == 4e-3
    assert TOLERANCES["eltwise_fp32"].max_rel == 5e-7 and TOLERANCES["cross_chip_sum_gather"].rel_fro == 2e-7
