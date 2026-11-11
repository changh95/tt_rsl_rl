# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.2 device tests, part 2 (single-chip semantics: whole-batch references): the RND predictor's loss, gradients and Adam steps against upstream autograd, and a whole
`TTPPO.update()` with RND against `PPO.update()` on identical data (actor, critic AND predictor in function space; the RND loss; the state normalizer).

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8r -- python -m pytest -m tt tests/tt/test_tt_rnd.py tests/tt/test_tt_rnd_update.py -s -q -p no:cacheprovider
"""

import copy
import torch

import pytest

from rsl_rl_tt.testing.parity import TOLERANCES, UPDATE_TOLERANCES, compare
from tests.tt.test_tt_rnd import CASES, rec, rnd_cfg, rnd_spec, step_obs
from tests.tt.test_tt_update import VARIANTS, make_pair, ref_rnd_params, rnd_function_deviation, run_update_pair, tt_rnd_params

pytestmark = pytest.mark.tt

RND_LOSS = TOLERANCES["rnd_predictor_loss"]


@pytest.mark.parametrize("case", list(CASES))
def test_predictor_loss_and_gradients_match_autograd(tt_mesh, case, tmp_path):
    name, spec = rnd_spec(case)
    runner, env, cfg, ref_alg = make_pair(name, tmp_path, spec=spec)
    alg = runner.alg
    alg.train_mode()
    ref_alg.train_mode()
    tt, ref = alg.rnd_tt, ref_alg.rnd
    ctx = alg.ctx
    rows = env.num_envs
    torch.manual_seed(3)
    batches = step_obs(env, 4)
    for obs in batches[:2]:  # non-trivial normalizer statistics, the same on both sides
        ref.update_normalization(obs)
        tt.update_normalization(tt.obs_to_device(obs), rows)
    obs = batches[3]
    loss = ref.compute_loss(obs)
    ref.optimizer.zero_grad()
    loss.backward()
    got_loss, got_grads = tt.loss_and_grads(tt.obs_to_device(obs), rows)
    rep = compare(ctx.download(got_loss, op="H-5", batch_sharded=False)[0, :1], loss.detach().reshape(1), RND_LOSS, f"{case} loss")
    assert rep.passed, str(rep)
    worst = {}
    for i, (layer, lin) in enumerate(zip(tt.predictor.layers, [m for m in ref.predictor if type(m).__name__ == "Linear"])):
        dw = layer.weight_from_host(ctx.download(got_grads[2 * i], op="H-6", batch_sharded=False))
        db = layer.bias_from_host(ctx.download(got_grads[2 * i + 1], op="H-6", batch_sharded=False))
        for got, want, what in ((dw, lin.weight.grad, "weight"), (db, lin.bias.grad, "bias")):
            r = compare(got, want, "mlp_backward", f"{case} layer {i} d{what}")
            worst[f"{i}.{what}"] = r.rel_fro_eff
            assert r.passed, str(r)
    rec(f"predictor grads {case}", {"loss_rel_fro": rep.rel_fro, "grad_rel_fro_eff": worst})


@pytest.mark.parametrize("case", ["cartpole-actor", "tracking12-own"])
def test_predictor_adam_steps_match_upstream_in_function_space(tt_mesh, case, tmp_path):
    """Five consecutive steps (a different mini-batch each) of the predictor: the loss of every step, and the embedding the predictor ends up with, against upstream and against upstream with
    gradient noise of the measured matmul size injected (the baseline of the v1 update tests: Adam turns the sign flips of near-zero gradient components into full-size steps)."""
    name, spec = rnd_spec(case)
    runner, env, cfg, ref_alg = make_pair(name, tmp_path, spec=spec)
    alg = runner.alg
    alg.train_mode()
    ref_alg.train_mode()
    tt, ref = alg.rnd_tt, ref_alg.rnd
    ctx = alg.ctx
    rows = env.num_envs
    torch.manual_seed(5)
    batches = step_obs(env, 8)
    for obs in batches[:2]:
        ref.update_normalization(obs)
        tt.update_normalization(tt.obs_to_device(obs), rows)
    noisy = copy.deepcopy(ref)  # same weights, same normalizer, its own Adam: only its gradients get noise
    g = torch.Generator().manual_seed(0)
    for p in noisy.predictor.parameters():
        p.register_hook(lambda grad: grad + 4e-3 * grad.pow(2).mean().sqrt() * torch.randn(grad.shape, generator=g))
    init = ref_rnd_params(ref_alg)
    losses = []
    for obs in batches[2:7]:
        for module in (ref, noisy):
            loss = module.compute_loss(obs)
            if module is ref:
                ref_loss = float(loss.detach())
            module.optimizer.zero_grad()
            loss.backward()
            module.optimizer.step()
        got, grads = tt.loss_and_grads(tt.obs_to_device(obs), rows)
        tt.adam.prepare_step()
        tt.apply(grads)
        losses.append((float(ctx.download(got, op="H-5", batch_sharded=False)[0, 0]), ref_loss))
    assert tt.adam.t == 5
    for k, (got, want) in enumerate(losses):
        assert got == pytest.approx(want, rel=RND_LOSS.rel_fro), f"step {k}: loss {got} vs {want}"
    obs_eval = batches[7]
    p_ref, p_tt = ref_rnd_params(ref_alg), tt_rnd_params(alg)
    lins = [m for m in noisy.predictor if type(m).__name__ == "Linear"]
    p_noisy = {f"rnd.{i}.{k}": getattr(lin, k).detach().clone() for i, lin in enumerate(lins) for k in ("weight", "bias")}
    dev_tt = rnd_function_deviation(p_tt, p_ref, init, ref_alg, obs_eval)
    dev_noisy = rnd_function_deviation(p_noisy, p_ref, init, ref_alg, obs_eval)
    rec(f"predictor adam {case}", {"function_deviation_tt": dev_tt, "function_deviation_noisy_ref": dev_noisy, "losses": losses})
    limit = UPDATE_TOLERANCES["noise_factor"] * dev_noisy + UPDATE_TOLERANCES["noise_abs"]
    assert dev_tt <= limit, f"predictor: {dev_tt:.3e} of the update differs (limit {limit:.3e}; noise baseline {dev_noisy:.3e})"


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("variant", ["one-step", "full-clipped"])
def test_update_with_rnd_matches_upstream_from_identical_state(tt_mesh, case, variant, tmp_path):
    """One `TTPPO.update()` vs one `PPO.update()` with RND from the SAME weights / normalizers / Adam states on the SAME data (the stored rewards include the TT intrinsic reward) and RNG
    state: the three losses and the RND loss, the actor / critic / PREDICTOR in function space, the RND state normalizer's statistics."""
    name, spec = rnd_spec(case)
    r = run_update_pair(name, tmp_path, {**VARIANTS[variant], "rnd_cfg": rnd_cfg()}, warm_iterations=2, spec_over={"obs_groups": CASES[case][1]})
    tt, ref = r["tt_loss"], r["ref_loss"]
    assert set(tt) == set(ref) and "rnd" in tt
    for key, tol in (("value", "update_value_loss"), ("entropy", "update_entropy"), ("surrogate", "update_surrogate")):
        rep = compare(torch.tensor([tt[key]]), torch.tensor([ref[key]]), TOLERANCES[tol], f"{case}/{variant} loss.{key}")
        assert rep.passed, str(rep)
    rep = compare(torch.tensor([tt["rnd"]]), torch.tensor([ref["rnd"]]), RND_LOSS, f"{case}/{variant} loss.rnd")
    rec(f"update {case}/{variant} loss.rnd", {"tt": tt["rnd"], "ref": ref["rnd"], "rel": rep.rel_fro})
    assert rep.passed, str(rep)
    for tag in ("actor", "critic"):
        got, noise = r["function"]["tt"][tag], r["function"]["noisy_ref"][tag]
        limit = UPDATE_TOLERANCES["full_update_clipped"] if variant == "full-clipped" else UPDATE_TOLERANCES["noise_factor"] * noise + UPDATE_TOLERANCES["noise_abs"]
        rec(f"update {case}/{variant} function_deviation.{tag}", {"tt": got, "noisy_ref": noise})
        assert got <= limit, f"{case}/{variant} {tag}: {got:.3e} of the update differs (limit {limit:.3e}; noise baseline {noise:.3e})"
    got, noise = r["rnd_function"]["tt"], r["rnd_function"]["noisy_ref"]
    limit = UPDATE_TOLERANCES["noise_factor"] * noise + UPDATE_TOLERANCES["noise_abs"]
    rec(f"update {case}/{variant} function_deviation.predictor", {"tt": got, "noisy_ref": noise})
    assert got <= limit, f"{case}/{variant} predictor: {got:.3e} of the update differs (limit {limit:.3e}; noise baseline {noise:.3e})"
    alg, ref_alg = r["alg"], r["ref_alg"]
    if alg.rnd_tt.state_norm is not None:
        s = alg.rnd_tt.state_norm.host_stats()
        e = ref_alg.rnd.state_normalizer
        assert compare(s["mean"], e.mean, TOLERANCES["normalizer_update"]).passed and compare(s["var"], e._var.reshape(-1), TOLERANCES["normalizer_update"]).passed
        assert s["count"] == int(e.count)
