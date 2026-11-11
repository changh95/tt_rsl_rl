# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.1 device tests (single-chip semantics: whole-batch references): the heteroscedastic Gaussian (state-dependent std) against upstream autograd / upstream `PPO.update`.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8a -- python -m pytest -m tt tests/tt/test_tt_hetero_update.py -s -q -p no:cacheprovider

Same oracles and tolerances as the v1 suites (`test_tt_backward.py`, `test_tt_update.py`): nothing is loosened for the new distribution.
"""

import copy
import json
import torch
from pathlib import Path

import pytest

from rsl_rl_tt.testing.parity import TOLERANCES, UPDATE_TOLERANCES, compare
from rsl_rl_tt.tt_common import pad_matrix, pad_up
from tests.tt.test_tt_backward import CLIP, C_E, C_V, dev, make_batch, reference_losses
from tests.tt.test_tt_modules import feed_normalizer, make_model, make_obs
from tests.tt.test_tt_update import STEPS_PER_ITER, VARIANTS, rec as rec_update, run_update_pair

pytestmark = pytest.mark.tt

HETERO = "HeteroscedasticGaussianDistribution"
RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
MEASURED: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _dump():
    yield
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase8_hetero_update.json").write_text(json.dumps(MEASURED, indent=2, default=str))


def rec(key, rep):
    MEASURED[key] = {"rel_fro": rep.rel_fro, "rel_fro_eff": rep.rel_fro_eff, "err_rms": rep.err_rms, "ref_rms": rep.ref_rms, "max_abs": rep.max_abs}
    print(f"[measure] {key}: rel_fro={rep.rel_fro:.3e} eff={rep.rel_fro_eff:.3e} err_rms={rep.err_rms:.3e} ref_rms={rep.ref_rms:.3e} max_abs={rep.max_abs:.3e}")
    return rep


# groups, actor out dim, hidden, activation, normalisation, std type, std range, clipped value loss
CASES = [
    ({"policy": 4}, 1, [64, 64], "elu", True, "scalar", (0.05, 4.0), True),
    ({"policy": 36}, 12, [128, 128, 64], "elu", True, "scalar", (0.05, 4.0), True),
    ({"policy": 36}, 12, [128, 128, 64], "elu", True, "log", (0.05, 4.0), False),
    ({"policy": 17}, 5, [33, 33], "tanh", True, "scalar", (0.05, 4.0), True),
    ({"policy": 48}, 29, [50, 70], "elu", False, "log", (0.05, 4.0), False),
    ({"policy": 36}, 12, [128, 64], "elu", True, "scalar", (0.55, 1.1), True),  # narrow clamp range: a good part of the std outputs sit OUTSIDE it (gradient gate = 0)
    ({"policy": 36}, 12, [128, 64], "elu", True, "log", (0.55, 1.1), True),
]


def make_hetero(case, seed=3):
    groups, out_dim, hidden, act, norm, std_type, std_range, _ = CASES[case]
    actor = make_model(groups, list(groups), out_dim, hidden, act, norm, seed=seed, std_type=std_type, init_std=0.8, std_range=std_range, distribution=HETERO)
    critic = make_model(groups, list(groups), 1, hidden, act, norm, dist=False, seed=seed + 1)
    if norm:
        for m in (actor, critic):
            feed_normalizer(m, groups)
    # the std head starts at zero weights (constant std): give it weights so that the std differs from row to row. The wide ranges keep the std well inside the
    # clamp (std ~ 0.8 +- 0.25 in a range of 0.05..4), so that no sample is near a clamp boundary: the gradient is discontinuous there (the gate), and a sample within the
    # matmul noise of a boundary can take the other branch on the device. The narrow ranges push part of the outputs outside the clamp on purpose; their formulas
    # are tested given IDENTICAL network outputs (so the gates agree), not end to end.
    scale = 4.0 if std_range[0] > 0.1 else 1.0
    with torch.no_grad():
        last = actor.mlp[-2]
        g = torch.Generator().manual_seed(11)
        last.weight[out_dim:] = 0.35 * torch.randn(last.weight[out_dim:].shape, generator=g) / last.weight.shape[1] ** 0.5 * scale
        last.bias[out_dim:] += 0.1 * torch.randn(out_dim, generator=g)
    return actor, critic


@pytest.mark.parametrize("case", [0, 1, 2, 3, 4], ids=[f"case{i}" for i in range(5)])
@pytest.mark.parametrize("rows", [64, 512])
def test_loss_and_gradient_parity_with_autograd(tt_ctx, case, rows):
    from rsl_rl_tt.losses import PPOLosses
    from rsl_rl_tt.modules import TTMLPModel

    groups, out_dim, hidden, act, norm, std_type, std_range, clipped_value = CASES[case]
    actor, critic = make_hetero(case)
    obs = make_obs(rows, groups, seed=8, scale=1.5, offset=0.4)
    batch = make_batch(actor, critic, obs, rows, seed=9)

    ref_actor, ref_critic = copy.deepcopy(actor), copy.deepcopy(critic)
    ref = reference_losses(ref_actor, ref_critic, obs, batch, clipped_value=clipped_value)
    ref["loss"].backward()

    tt_a, tt_c = TTMLPModel(tt_ctx, actor), TTMLPModel(tt_ctx, critic)
    losses = PPOLosses(tt_ctx, clip_param=CLIP, value_loss_coef=C_V, entropy_coef=C_E, use_clipped_value_loss=clipped_value)
    mean_dev, dist, cache_a = tt_a.policy_forward_train(tt_a.obs_to_device(obs))
    la = losses.actor(dist, mean_dev, dev(tt_ctx, batch["actions"], rows), dev(tt_ctx, batch["old_logp"], rows, 1), dev(tt_ctx, batch["adv"], rows, 1), rows)
    d_out = tt_ctx.ttnn.concat([la["d_mean"], la["d_raw"]], dim=1)
    grads_a = tt_a.mlp.backward(cache_a, d_out)
    val_dev, cache_c = tt_c.forward_train(tt_c.obs_to_device(obs))
    lc = losses.critic(val_dev[:, 0:1], dev(tt_ctx, batch["old_values"], rows, 1), dev(tt_ctx, batch["returns"], rows, 1), rows)
    col0 = tt_ctx.upload(torch.cat([torch.ones(1, 1), torch.zeros(1, 31)], dim=1), op="H-6", batch_sharded=False)
    grads_c = tt_c.mlp.backward(cache_c, tt_ctx.ttnn.multiply(lc["d_value"], col0))

    host = lambda t, op="H-5", sharded=False: tt_ctx.download(t, op=op, batch_sharded=sharded)
    for name, tt_val, ref_val in (("surrogate", la["surrogate_loss"], ref["surrogate"]), ("value", lc["value_loss"], ref["value"]), ("entropy", dist.entropy_dev, ref["entropy"])):
        rep = rec(f"case{case} rows={rows} loss.{name}", compare(host(tt_val)[0, 0].reshape(1), ref_val.detach().reshape(1), TOLERANCES["ppo_gradients_e2e"]))
        assert rep.passed, str(rep)
    ref_layers_a = [m for m in ref_actor.mlp if type(m).__name__ == "Linear"]
    ref_layers_c = [m for m in ref_critic.mlp if type(m).__name__ == "Linear"]
    for tag, model, grads, ref_layers in (("actor", tt_a, grads_a, ref_layers_a), ("critic", tt_c, grads_c, ref_layers_c)):
        for i, ((dw, db), lin, layer) in enumerate(zip(grads, ref_layers, model.mlp.layers)):
            gw, gb = layer.weight_from_host(host(dw)), layer.bias_from_host(host(db))
            rw = rec(f"case{case} rows={rows} {tag}.layer{i}.weight", compare(gw, lin.weight.grad, TOLERANCES["ppo_gradients_e2e"]))
            rb = rec(f"case{case} rows={rows} {tag}.layer{i}.bias", compare(gb, lin.bias.grad, TOLERANCES["ppo_gradients_e2e"]))
            assert rw.passed and rb.passed, str(rw) + str(rb)
            # padding stays inert: everything of the device layout that is not a real element of a real head is exactly zero
            back = layer.weight_to_host(layer.weight_from_host(host(dw)))
            assert torch.equal(back, host(dw)), f"{tag}.layer{i}: nonzero gradient in a padded row / column"
            assert torch.equal(layer.bias_to_host(layer.bias_from_host(host(db))), host(db))


@pytest.mark.parametrize("case", range(len(CASES)), ids=[f"case{i}" for i in range(len(CASES))])
@pytest.mark.parametrize("rows", [64, 512])
def test_ppo_loss_formulas_match_autograd_given_the_same_network_outputs(tt_ctx, case, rows):
    """Isolates the formulas (log-prob / entropy / std chain rule through the clamp, per row) from the matmul noise: autograd runs on the DEVICE's own output of
    the two-head layer, so both sides see identical inputs."""
    from rsl_rl_tt.losses import PPOLosses
    from rsl_rl_tt.modules import TTMLPModel

    groups, out_dim, hidden, act, norm, std_type, std_range, clipped_value = CASES[case]
    actor, critic = make_hetero(case)
    obs = make_obs(rows, groups, seed=8, scale=1.5, offset=0.4)
    batch = make_batch(actor, critic, obs, rows, seed=9)
    tt_a, tt_c = TTMLPModel(tt_ctx, actor), TTMLPModel(tt_ctx, critic)
    losses = PPOLosses(tt_ctx, clip_param=CLIP, value_loss_coef=C_V, entropy_coef=C_E, use_clipped_value_loss=clipped_value)
    out_dev = tt_a.forward(tt_a.obs_to_device(obs))
    a_p = tt_a.distribution.dim_p
    out_host = tt_ctx.download(out_dev, op="H-2", batch_sharded=True)
    out_tt = torch.stack([out_host[:, :out_dim], out_host[:, a_p : a_p + out_dim]], dim=1).clone().requires_grad_(True)  # upstream layout [rows, 2, A]
    val_dev = tt_c.forward(tt_c.obs_to_device(obs))
    val_tt = tt_ctx.download(val_dev[:, 0:1], op="H-2", batch_sharded=True).clone().requires_grad_(True)

    ref_actor = copy.deepcopy(actor)
    ref_actor.distribution.update(out_tt)
    logp = ref_actor.distribution.log_prob(batch["actions"])
    ratio = torch.exp(logp - batch["old_logp"].squeeze(-1))
    adv = batch["adv"].squeeze(-1)
    surrogate_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)).mean()
    if clipped_value:
        vc = batch["old_values"] + (val_tt - batch["old_values"]).clamp(-CLIP, CLIP)
        value_loss = torch.max((val_tt - batch["returns"]).pow(2), (vc - batch["returns"]).pow(2)).mean()
    else:
        value_loss = (batch["returns"] - val_tt).pow(2).mean()
    entropy = ref_actor.distribution.entropy.mean()
    (surrogate_loss + C_V * value_loss - C_E * entropy).backward()

    mean_dev, dist = tt_a.distribution.from_output(out_dev)
    la = losses.actor(dist, mean_dev, dev(tt_ctx, batch["actions"], rows), dev(tt_ctx, batch["old_logp"], rows, 1), dev(tt_ctx, batch["adv"], rows, 1), rows)
    lc = losses.critic(val_dev[:, 0:1], dev(tt_ctx, batch["old_values"], rows, 1), dev(tt_ctx, batch["returns"], rows, 1), rows)
    h = lambda t, op="H-2", sharded=True: tt_ctx.download(t, op=op, batch_sharded=sharded)
    tight = TOLERANCES["ppo_loss_formulas"]
    checks = [
        ("surrogate_loss", h(la["surrogate_loss"], "H-5", False)[0, 0].reshape(1), surrogate_loss.detach().reshape(1)),
        ("value_loss", h(lc["value_loss"], "H-5", False)[0, 0].reshape(1), value_loss.detach().reshape(1)),
        ("entropy", h(dist.entropy_dev, "H-5", False)[0, 0].reshape(1), entropy.detach().reshape(1)),
        ("log_prob", h(la["logp"])[:, 0], logp.detach()),
        ("d_mean", h(la["d_mean"])[:, :out_dim], out_tt.grad[:, 0, :]),
        ("d_raw_std_head", h(la["d_raw"])[:, :out_dim], out_tt.grad[:, 1, :]),
        ("d_value", h(lc["d_value"])[:, :1], val_tt.grad),
    ]
    for name, got, want in checks:
        rep = rec(f"formulas case{case} rows={rows} {name}", compare(got, want, tight))
        assert rep.passed, str(rep)
    assert torch.count_nonzero(h(la["d_raw"])[:, out_dim:]) == 0 and torch.count_nonzero(h(la["d_mean"])[:, out_dim:]) == 0  # padded columns get no gradient


@pytest.mark.parametrize("case", [0, 1, 2, 4, 5])
def test_kl_with_the_stored_per_sample_std_matches_upstream(tt_ctx, case):
    from rsl_rl_tt.losses import PPOLosses
    from rsl_rl_tt.modules import TTMLPModel

    groups, out_dim, hidden, act, norm, std_type, std_range, clipped_value = CASES[case]
    rows = 256
    actor, critic = make_hetero(case)
    obs = make_obs(rows, groups, seed=8, scale=1.5, offset=0.4)
    batch = make_batch(actor, critic, obs, rows, seed=9)
    tt_a = TTMLPModel(tt_ctx, actor)
    losses = PPOLosses(tt_ctx, clip_param=CLIP, value_loss_coef=C_V, entropy_coef=C_E, use_clipped_value_loss=clipped_value)
    mean_dev, dist = tt_a.policy_forward(tt_a.obs_to_device(obs))
    ttnn = tt_ctx.ttnn
    old_std = tt_ctx.upload(pad_matrix(batch["old_std"], rows, dist.dist.dim_p, fill=1.0), op="H-1", batch_sharded=True)  # padded columns of a stored std are 1
    old_mean = dev(tt_ctx, batch["old_mean"], rows)
    kl = losses.kl_mean(dist, old_mean, ttnn.square(old_std), ttnn.log(old_std), mean_dev)
    # reference: upstream KL between the stored (old) parameters and the new ones, where "new" is the device's own output (so the matmul noise is out of the way)
    out_host = tt_ctx.download(tt_a.forward(tt_a.obs_to_device(obs)), op="H-2", batch_sharded=True)
    a_p = dist.dist.dim_p
    ref_actor = copy.deepcopy(actor)
    ref_actor.distribution.update(torch.stack([out_host[:, :out_dim], out_host[:, a_p : a_p + out_dim]], dim=1))
    ref_kl = ref_actor.get_kl_divergence((batch["old_mean"], batch["old_std"]), ref_actor.output_distribution_params).mean()
    rep = compare(tt_ctx.download(kl, op="H-5", batch_sharded=False)[0, 0].reshape(1), ref_kl.reshape(1), TOLERANCES["row_sum_fp32"], f"kl case{case}")
    rec(f"kl case{case}", rep)
    assert rep.passed, str(rep)


# ---- the full update --------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
@pytest.mark.parametrize("std_type", ["scalar", "log"])
@pytest.mark.parametrize("variant", ["one-step", "full-clipped", "full-smooth"])
def test_update_matches_upstream_from_identical_state(tt_mesh, name, std_type, variant, tmp_path):
    """One TTPPO.update() vs one upstream PPO.update() with a heteroscedastic actor, from the SAME weights / normalizer / Adam state on the SAME data and RNG state
    (the scaffolding and the criteria of the v1 test: losses directly, parameters in function space against upstream with injected matmul-sized gradient noise).
    The function space includes the std head: the whole output of the actor MLP is compared."""
    r = run_update_pair(name, tmp_path, VARIANTS[variant], warm_iterations=2, spec_over={"distribution": HETERO, "std_type": std_type, "init_std": 0.8})
    tt, ref = r["tt_loss"], r["ref_loss"]
    for key, tol in (("value", "update_value_loss"), ("entropy", "update_entropy_hetero"), ("surrogate", "update_surrogate")):
        rep = compare(torch.tensor([tt[key]]), torch.tensor([ref[key]]), TOLERANCES[tol], f"{name}/{std_type}/{variant} loss.{key}")
        rec_update(f"hetero {name}/{std_type}/{variant} loss.{key}", {"tt": tt[key], "ref": ref[key], "abs_diff": abs(tt[key] - ref[key])})
        assert rep.passed, str(rep)
    assert r["lr"]["tt"] == pytest.approx(r["lr"]["ref"], rel=1e-6)
    for tag in ("actor", "critic"):
        got, noise = r["function"]["tt"][tag], r["function"]["noisy_ref"][tag]
        rec_update(f"hetero {name}/{std_type}/{variant} function_deviation.{tag}", {"tt": got, "noisy_ref": noise})
        limit = UPDATE_TOLERANCES["full_update_clipped"] if variant == "full-clipped" else UPDATE_TOLERANCES["noise_factor"] * noise + UPDATE_TOLERANCES["noise_abs"]
        assert got <= limit, f"{name}/{std_type}/{variant} {tag}: {got:.3e} of the update differs (limit {limit:.3e}; noise baseline {noise:.3e})"
