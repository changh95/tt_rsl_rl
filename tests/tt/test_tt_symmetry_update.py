# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.3 device tests, part 2 (single-chip semantics: whole-batch references): a whole `TTPPO.update()` with the symmetry extension against upstream `PPO.update()` on identical data,
in every mode of the extension and with 2 and 4 copies of the batch: the four losses, the actor / critic in function space (against the upstream algorithm with matmul-sized gradient noise),
and the case where the extension must change NOTHING (logging only).

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8s -- python -m pytest -m tt tests/tt/test_tt_symmetry.py tests/tt/test_tt_symmetry_update.py -s -q -p no:cacheprovider
"""

import torch

import pytest

from rsl_rl_tt.testing.parity import TOLERANCES, UPDATE_TOLERANCES, compare
from rsl_rl_tt.testing.reference import symmetry_cfg
from tests.tt.test_tt_symmetry import CASES, rec
from tests.tt.test_tt_update import SMOOTH, VARIANTS, run_update_pair

pytestmark = pytest.mark.tt

# the mirror loss is a mean of squares of a difference of two actor outputs, each with the matmul floor
SYM_LOSS = TOLERANCES["update_symmetry_loss"]

# with augmentation half (K = 2) or three quarters (K = 4) of a mini-batch are MIRRORED copies, whose ratios (the policy is not symmetric yet, the old log-probability is the original's) spread far
# from 1: many samples sit near the clip boundaries, where the forward noise flips the gate. The exactness of the gradient formulas is therefore tested on the smooth objective (no gates), the
# clipped objective with the standard whole-update limit.
SYM_VARIANTS = {"one-step-smooth": dict(num_learning_epochs=1, num_mini_batches=1, schedule="fixed", **SMOOTH), "full-clipped": VARIANTS["full-clipped"]}

MODES = {
    "augmentation": dict(data_augmentation=True, mirror_loss=False),
    "mirror-loss": dict(data_augmentation=False, mirror_loss=True),
    "both": dict(data_augmentation=True, mirror_loss=True),
    "logging-only": dict(data_augmentation=False, mirror_loss=False),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("variant", list(SYM_VARIANTS))
def test_update_with_symmetry_matches_upstream_from_identical_state(tt_mesh, case, mode, variant, tmp_path):
    name, func = CASES[case]
    over = {**SYM_VARIANTS[variant], "symmetry_cfg": symmetry_cfg(name, func=func, coeff=0.5, **MODES[mode])}
    r = run_update_pair(name, tmp_path, over, warm_iterations=2)
    tt, ref = r["tt_loss"], r["ref_loss"]
    assert set(tt) == set(ref) and "symmetry" in tt
    for key, tol in (("value", "update_value_loss"), ("entropy", "update_entropy"), ("surrogate", "update_surrogate")):
        rep = compare(torch.tensor([tt[key]]), torch.tensor([ref[key]]), TOLERANCES[tol], f"{case}/{mode}/{variant} loss.{key}")
        assert rep.passed, str(rep)
    rep = compare(torch.tensor([tt["symmetry"]]), torch.tensor([ref["symmetry"]]), SYM_LOSS, f"{case}/{mode}/{variant} loss.symmetry")
    rec(f"update {case}/{mode}/{variant} loss.symmetry", {"tt": tt["symmetry"], "ref": ref["symmetry"], "rel": rep.rel_fro})
    assert rep.passed, str(rep)
    assert r["lr"]["tt"] == pytest.approx(r["lr"]["ref"], rel=1e-6)
    for tag in ("actor", "critic"):
        got, noise = r["function"]["tt"][tag], r["function"]["noisy_ref"][tag]
        limit = UPDATE_TOLERANCES["full_update_clipped"] if variant == "full-clipped" else UPDATE_TOLERANCES["noise_factor"] * noise + UPDATE_TOLERANCES["noise_abs"]
        rec(f"update {case}/{mode}/{variant} function_deviation.{tag}", {"tt": got, "noisy_ref": noise})
        assert got <= limit, f"{case}/{mode}/{variant} {tag}: {got:.3e} of the update differs (limit {limit:.3e}; noise baseline {noise:.3e})"


@pytest.mark.parametrize("case", ["cartpole-K2", "tracking12-K4"])
def test_the_mirror_loss_pulls_the_policy_toward_symmetry_and_logging_changes_nothing(tt_mesh, case, tmp_path):
    """Three identical runs from the same state and data: without the extension, with it for logging only (the parameters must be BIT-identical to the first), and with the mirror loss (the
    logged symmetry loss of the next update is lower than without it: the policy became more symmetric)."""
    from tests.tt.test_tt_rollout import build_tt_runner, small_spec
    from tests.tt.test_tt_symmetry import sym_spec
    from tests.tt.test_tt_update import tt_params

    name, _ = CASES[case]
    states, losses = {}, {}
    for label, kwargs in (("off", None), ("logging", MODES["logging-only"]), ("loss", dict(data_augmentation=False, mirror_loss=True, coeff=20.0))):
        spec = small_spec(name) if kwargs is None else sym_spec(case, **kwargs)[1]
        runner, env, cfg = build_tt_runner(name, tmp_path / label, spec=spec, seed=3)
        torch.manual_seed(77)
        runner.learn(num_learning_iterations=6)
        states[label] = tt_params(runner.alg)
        if kwargs is not None:
            rows = (tmp_path / label / "metrics.csv").read_text().splitlines()
            cols = rows[0].split(",")
            losses[label] = [float(r.split(",")[cols.index("Loss/symmetry")]) for r in rows[1:]]
    for k in states["off"]:
        assert torch.equal(states["off"][k], states["logging"][k]), f"the extension used for logging only changed {k}"
    assert losses["loss"][-1] < losses["logging"][-1], f"the mirror loss did not make the policy more symmetric: {losses}"
    rec(f"mirror loss effect {case}", losses)
