# SPDX-License-Identifier: BSD-3-Clause
"""Phase 3 device tests, part 1: hand-written backward, PPO loss gradients, Adam, grad-norm clipping, adaptive learning rate.
The oracle is upstream torch autograd / torch.optim.Adam / clip_grad_norm_ on the same numbers (BRIEF 3.2: gradient parity per parameter).

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p3a -- python -m pytest -m tt tests/tt/test_tt_backward.py -s -q -p no:cacheprovider
"""

import copy
import json
import torch
from pathlib import Path

import pytest

from rsl_rl_tt.testing.parity import TOLERANCES, Tol, compare
from rsl_rl_tt.tt_common import pad_matrix, pad_up
from tests.tt.test_tt_modules import feed_normalizer, make_model, make_obs

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
MEASURED: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _dump():
    yield
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase3_backward.json").write_text(json.dumps(MEASURED, indent=2, default=str))


def rec(key, rep):
    MEASURED[key] = {"rel_fro": rep.rel_fro, "rel_fro_eff": rep.rel_fro_eff, "err_rms": rep.err_rms, "ref_rms": rep.ref_rms, "max_abs": rep.max_abs}
    print(f"[measure] {key}: rel_fro={rep.rel_fro:.3e} eff={rep.rel_fro_eff:.3e} err_rms={rep.err_rms:.3e} ref_rms={rep.ref_rms:.3e} max_abs={rep.max_abs:.3e}")
    return rep



CLIP, C_V, C_E = 0.2, 1.0, 0.01

CASES = [
    # groups (actor), actor out, hidden, activation, norm, std_type, clipped value loss
    ({"policy": 4}, 1, [64, 64], "elu", True, "scalar", True),
    ({"policy": 36}, 12, [128, 128, 64], "elu", True, "scalar", True),
    ({"policy": 36}, 12, [128, 128, 64], "elu", True, "log", False),
    ({"policy": 17}, 5, [33, 33], "tanh", True, "scalar", True),
    ({"policy": 9}, 3, [96], "relu", False, "log", True),
    ({"policy": 48}, 29, [50, 70], "elu", False, "scalar", False),
]


def reference_losses(actor, critic, obs, batch, *, clipped_value=True):
    """The loss block of rsl_rl PPO.update (pinned tag), run on torch tensors with autograd; returns (loss parts dict)."""
    actor(obs, stochastic_output=True)
    logp = actor.get_output_log_prob(batch["actions"])
    values = critic(obs)
    entropy = actor.output_entropy
    ratio = torch.exp(logp - batch["old_logp"].squeeze(-1))
    adv = batch["adv"].squeeze(-1)
    surrogate = -adv * ratio
    surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)
    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
    if clipped_value:
        value_clipped = batch["old_values"] + (values - batch["old_values"]).clamp(-CLIP, CLIP)
        value_losses = (values - batch["returns"]).pow(2)
        value_losses_clipped = (value_clipped - batch["returns"]).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
    else:
        value_loss = (batch["returns"] - values).pow(2).mean()
    loss = surrogate_loss + C_V * value_loss - C_E * entropy.mean()
    return {"loss": loss, "surrogate": surrogate_loss, "value": value_loss, "entropy": entropy.mean(), "logp": logp, "values": values}


def make_batch(actor, critic, obs, rows, seed):
    """Random but plausible PPO data; log-ratios and value deltas are wide enough that BOTH clip branches are active."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        actor(obs, stochastic_output=True)
        mean, std = actor.output_mean, actor.output_std
        actions = mean + std * torch.randn(mean.shape, generator=g) * 1.3
        logp = actor.get_output_log_prob(actions)
        values = critic(obs)
    # The clipped objectives are DISCONTINUOUS in their gradient at the clip boundaries, so a sample within ~1e-3 of a boundary can take the other
    # branch on the device (the logp / value differ by the matmul noise floor). The parity data therefore keeps every sample >= `margin` away from
    # the boundaries; the boundary behaviour itself is tested separately (test_gate_flips_are_confined_to_samples_at_the_clip_boundary).
    margin = 0.02
    d_logp = 0.35 * torch.randn(logp.shape, generator=g)  # old_logp = logp + d_logp  =>  ratio = exp(-d_logp)
    for boundary in (-torch.log(torch.tensor(1.0 + CLIP)), -torch.log(torch.tensor(1.0 - CLIP))):
        near = (d_logp - boundary).abs() < margin
        d_logp = torch.where(near, d_logp + 2.5 * margin * torch.sign(d_logp - boundary + 1e-9), d_logp)
    d_val = 0.3 * torch.randn(values.shape, generator=g)  # old_values = values + d_val  =>  |value - old_value| = |d_val| against the +-CLIP clamp
    d_val = torch.where((d_val.abs() - CLIP).abs() < margin, d_val + 2.5 * margin * torch.sign(d_val), d_val)
    return {
        "actions": actions,
        "old_logp": (logp + d_logp).unsqueeze(-1),
        "adv": torch.randn(rows, 1, generator=g),
        "returns": values + 0.6 * torch.randn(values.shape, generator=g),
        "old_values": values + d_val,
        "old_mean": mean + 0.1 * torch.randn(mean.shape, generator=g),
        "old_std": std * (1 + 0.15 * torch.rand(std.shape, generator=g)),
    }


def dev(ctx, t, rows, cols=None, op="H-1"):
    """Upload per-sample data: [rows, 1] scalars keep their logical column shape, wider data is zero-padded to a multiple of 32."""
    if t.shape[1] == 1:
        return ctx.upload(t.contiguous(), op=op, batch_sharded=True)
    return ctx.upload(pad_matrix(t, rows, pad_up(t.shape[1])), op=op, batch_sharded=True)


@pytest.mark.parametrize("case", range(len(CASES)), ids=[f"case{i}" for i in range(len(CASES))])
@pytest.mark.parametrize("rows", [64, 512])
def test_loss_and_gradient_parity_with_autograd(tt_ctx, case, rows):
    from rsl_rl_tt.losses import PPOLosses
    from rsl_rl_tt.modules import TTMLPModel

    groups, out_dim, hidden, act, norm, std_type, clipped_value = CASES[case]
    actor = make_model(groups, list(groups), out_dim, hidden, act, norm, seed=3, std_type=std_type, init_std=0.8, std_range=(0.05, 4.0))
    critic = make_model(groups, list(groups), 1, hidden, act, norm, dist=False, seed=4)
    if norm:
        for m in (actor, critic):
            feed_normalizer(m, groups)
    with torch.no_grad():
        p = actor.distribution.std_param if std_type == "scalar" else actor.distribution.log_std_param
        p.add_(0.15 * torch.randn(p.shape, generator=torch.Generator().manual_seed(1)))
    obs = make_obs(rows, groups, seed=8, scale=1.5, offset=0.4)
    batch = make_batch(actor, critic, obs, rows, seed=9)

    # --- upstream autograd ---
    ref_actor, ref_critic = copy.deepcopy(actor), copy.deepcopy(critic)
    ref = reference_losses(ref_actor, ref_critic, obs, batch, clipped_value=clipped_value)
    ref["loss"].backward()

    # --- device ---
    tt_a, tt_c = TTMLPModel(tt_ctx, actor), TTMLPModel(tt_ctx, critic)
    losses = PPOLosses(tt_ctx, clip_param=CLIP, value_loss_coef=C_V, entropy_coef=C_E, use_clipped_value_loss=clipped_value)
    obs_dev = tt_a.obs_to_device(obs)
    mean_dev, cache_a = tt_a.forward_train(obs_dev)
    dist = tt_a.distribution
    la = losses.actor(dist, mean_dev, dev(tt_ctx, batch["actions"], rows), dev(tt_ctx, batch["old_logp"], rows, 1), dev(tt_ctx, batch["adv"], rows, 1), rows)
    grads_a = tt_a.mlp.backward(cache_a, la["d_mean"])
    val_dev, cache_c = tt_c.forward_train(tt_c.obs_to_device(obs))
    lc = losses.critic(val_dev[:, 0:1], dev(tt_ctx, batch["old_values"], rows, 1), dev(tt_ctx, batch["returns"], rows, 1), rows)
    col0 = tt_ctx.upload(torch.cat([torch.ones(1, 1), torch.zeros(1, 31)], dim=1), op="H-6", batch_sharded=False)
    grads_c = tt_c.mlp.backward(cache_c, tt_ctx.ttnn.multiply(lc["d_value"], col0))

    def host(t, op="H-5", sharded=False):
        return tt_ctx.download(t, op=op, batch_sharded=sharded)

    # scalar losses
    for name, tt_val, ref_val in (("surrogate", la["surrogate_loss"], ref["surrogate"]), ("value", lc["value_loss"], ref["value"]), ("entropy", dist.entropy_dev, ref["entropy"])):
        rep = rec(f"case{case} rows={rows} loss.{name}", compare(host(tt_val)[0, 0].reshape(1), ref_val.detach().reshape(1), TOLERANCES["ppo_gradients_e2e"]))
        assert rep.passed, str(rep)
    # per-parameter gradients
    ref_layers_a = [m for m in ref_actor.mlp if type(m).__name__ == "Linear"]
    ref_layers_c = [m for m in ref_critic.mlp if type(m).__name__ == "Linear"]
    for tag, grads, ref_layers in (("actor", grads_a, ref_layers_a), ("critic", grads_c, ref_layers_c)):
        for i, ((dw, db), lin) in enumerate(zip(grads, ref_layers)):
            n_out, n_in = lin.weight.shape
            gw, gb = host(dw)[:n_in, :n_out].t(), host(db)[0, :n_out]
            rw = rec(f"case{case} rows={rows} {tag}.layer{i}.weight", compare(gw, lin.weight.grad, TOLERANCES["ppo_gradients_e2e"]))
            rb = rec(f"case{case} rows={rows} {tag}.layer{i}.bias", compare(gb, lin.bias.grad, TOLERANCES["ppo_gradients_e2e"]))
            assert rw.passed and rb.passed, str(rw) + str(rb)
            # padded rows / columns of the gradient are exactly zero (padding stays inert)
            assert torch.count_nonzero(host(dw)[n_in:, :]) == 0 and torch.count_nonzero(host(dw)[:, n_out:]) == 0 and torch.count_nonzero(host(db)[0, n_out:]) == 0
    ref_param = ref_actor.distribution.std_param if std_type == "scalar" else ref_actor.distribution.log_std_param
    rs = rec(f"case{case} rows={rows} actor.{std_type}_std_param", compare(host(la["d_param"])[0, : out_dim], ref_param.grad, TOLERANCES["ppo_gradients_e2e"]))
    assert rs.passed, str(rs)
    assert torch.count_nonzero(host(la["d_param"])[0, out_dim:]) == 0


# ---- the two halves of the e2e gradient check, each with a tight tolerance ---------------------------------------------------------------
def _host(ctx, t, op="H-5", sharded=False):
    return ctx.download(t, op=op, batch_sharded=sharded)


@pytest.mark.parametrize("case", range(len(CASES)), ids=[f"case{i}" for i in range(len(CASES))])
@pytest.mark.parametrize("rows", [64, 512])
def test_ppo_loss_formulas_match_autograd_given_the_same_network_outputs(tt_ctx, case, rows):
    """Isolates the loss/gradient formulas (clipped surrogate and value loss with torch's tie rules, entropy bonus, std chain rule) from the
    matmul noise: autograd is run on the DEVICE's own mean / value outputs, so both sides see identical inputs."""
    from rsl_rl_tt.losses import PPOLosses
    from rsl_rl_tt.modules import TTMLPModel

    groups, out_dim, hidden, act, norm, std_type, clipped_value = CASES[case]
    actor = make_model(groups, list(groups), out_dim, hidden, act, norm, seed=3, std_type=std_type, init_std=0.8, std_range=(0.05, 4.0))
    critic = make_model(groups, list(groups), 1, hidden, act, norm, dist=False, seed=4)
    if norm:
        for m in (actor, critic):
            feed_normalizer(m, groups)
    obs = make_obs(rows, groups, seed=8, scale=1.5, offset=0.4)
    batch = make_batch(actor, critic, obs, rows, seed=9)
    tt_a, tt_c = TTMLPModel(tt_ctx, actor), TTMLPModel(tt_ctx, critic)
    losses = PPOLosses(tt_ctx, clip_param=CLIP, value_loss_coef=C_V, entropy_coef=C_E, use_clipped_value_loss=clipped_value)
    mean_dev = tt_a.forward(tt_a.obs_to_device(obs))
    val_dev = tt_c.forward(tt_c.obs_to_device(obs))
    mean_tt = _host(tt_ctx, mean_dev, "H-2", True)[:, :out_dim].clone().requires_grad_(True)
    val_tt = _host(tt_ctx, val_dev[:, 0:1], "H-2", True).clone().requires_grad_(True)
    # reference: torch autograd through the loss only
    ref_actor = copy.deepcopy(actor)
    ref_actor.distribution.update(mean_tt)
    logp = ref_actor.distribution.log_prob(batch["actions"])
    ratio = torch.exp(logp - batch["old_logp"].squeeze(-1))
    adv = batch["adv"].squeeze(-1)
    surrogate_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)).mean()
    if clipped_value:
        vc = batch["old_values"] + (val_tt - batch["old_values"]).clamp(-CLIP, CLIP)
        value_loss = torch.max((val_tt - batch["returns"]).pow(2), (vc - batch["returns"]).pow(2)).mean()
    else:
        value_loss = (batch["returns"] - val_tt).pow(2).mean()
    (surrogate_loss + C_V * value_loss - C_E * ref_actor.distribution.entropy.mean()).backward()
    ref_param = ref_actor.distribution.std_param if std_type == "scalar" else ref_actor.distribution.log_std_param

    dist = tt_a.distribution
    la = losses.actor(dist, mean_dev, dev(tt_ctx, batch["actions"], rows), dev(tt_ctx, batch["old_logp"], rows, 1), dev(tt_ctx, batch["adv"], rows, 1), rows)
    lc = losses.critic(val_dev[:, 0:1], dev(tt_ctx, batch["old_values"], rows, 1), dev(tt_ctx, batch["returns"], rows, 1), rows)
    tight = TOLERANCES["ppo_loss_formulas"]
    checks = [
        ("surrogate_loss", _host(tt_ctx, la["surrogate_loss"])[0, 0].reshape(1), surrogate_loss.detach().reshape(1)),
        ("value_loss", _host(tt_ctx, lc["value_loss"])[0, 0].reshape(1), value_loss.detach().reshape(1)),
        ("log_prob", _host(tt_ctx, la["logp"], "H-2", True)[:, 0], logp.detach()),
        ("d_mean", _host(tt_ctx, la["d_mean"], "H-2", True)[:, :out_dim], mean_tt.grad),
        ("d_value", _host(tt_ctx, lc["d_value"], "H-2", True)[:, :1], val_tt.grad),
        (f"d_{std_type}_std_param", _host(tt_ctx, la["d_param"])[0, :out_dim], ref_param.grad),
    ]
    for name, got, want in checks:
        rep = rec(f"formulas case{case} rows={rows} {name}", compare(got, want, tight))
        assert rep.passed, str(rep)


@pytest.mark.parametrize("case", range(len(CASES)), ids=[f"case{i}" for i in range(len(CASES))])
@pytest.mark.parametrize("rows", [64, 512])
def test_mlp_backward_matches_autograd_for_a_given_output_gradient(tt_ctx, case, rows):
    """Isolates the layer-by-layer VJP (matmul transposes, bias row sums, ELU/tanh/ReLU derivatives, padding) from the loss: both sides
    backpropagate the SAME output gradient; the only difference left is the forward activations' matmul noise."""
    from rsl_rl_tt.modules import TTMLPModel

    groups, out_dim, hidden, act, norm, *_ = CASES[case]
    for name, dist_on, out in (("actor", True, out_dim), ("critic", False, 1)):
        model = make_model(groups, list(groups), out, hidden, act, norm, dist=dist_on, seed=5)
        if norm:
            feed_normalizer(model, groups)
        obs = make_obs(rows, groups, seed=8, scale=1.5, offset=0.4)
        grad_out = torch.randn(rows, out, generator=torch.Generator().manual_seed(21)) * 0.05
        ref = copy.deepcopy(model)
        ref_out = ref.mlp(ref.get_latent(obs))
        ref_out.backward(grad_out)
        tt = TTMLPModel(tt_ctx, model)
        _, cache = tt.forward_train(tt.obs_to_device(obs))
        grads = tt.mlp.backward(cache, dev(tt_ctx, grad_out, rows) if out > 1 else tt_ctx.ttnn.multiply(dev(tt_ctx, grad_out, rows, 1), tt_ctx.upload(torch.cat([torch.ones(1, 1), torch.zeros(1, 31)], dim=1), op="H-6", batch_sharded=False)))
        for i, ((dw, db), lin) in enumerate(zip(grads, [m for m in ref.mlp if type(m).__name__ == "Linear"])):
            n_out, n_in = lin.weight.shape
            rw = rec(f"mlp_backward case{case} rows={rows} {name}.layer{i}.weight", compare(_host(tt_ctx, dw)[:n_in, :n_out].t(), lin.weight.grad, TOLERANCES["mlp_backward_relu" if act == "relu" else "mlp_backward"]))
            rb = rec(f"mlp_backward case{case} rows={rows} {name}.layer{i}.bias", compare(_host(tt_ctx, db)[0, :n_out], lin.bias.grad, TOLERANCES["mlp_backward_relu" if act == "relu" else "mlp_backward"]))
            assert rw.passed and rb.passed, str(rw) + str(rb)


# ---- Adam ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("shapes", [[(32, 64), (1, 64)], [(64, 128), (1, 128), (128, 32), (1, 32), (1, 32)]])
def test_adam_matches_torch_over_many_steps_with_changing_lr(tt_ctx, shapes):
    from rsl_rl_tt.optim import TTAdam

    g = torch.Generator().manual_seed(7)
    init = [torch.randn(*s, generator=g) * 0.3 for s in shapes]
    tparams = [p.clone().requires_grad_(True) for p in init]
    opt = torch.optim.Adam(tparams, lr=1e-3)
    ctx = tt_ctx
    params = [ctx.upload(p, op="H-6", batch_sharded=False) for p in init]
    tt = TTAdam(ctx, params)
    lrs = [1e-3, 1e-3, 6.6e-4, 9.9e-4, 1.5e-3, 1e-2, 1e-5, 3e-4]
    for step, lr in enumerate(lrs):
        grads = [torch.randn(*s, generator=g) * (10 ** (-2 + step % 3)) for s in shapes]
        for p, gr in zip(tparams, grads):
            p.grad = gr.clone()
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        params = tt.step(params, [ctx.upload(gr, op="H-6", batch_sharded=False) for gr in grads], ctx.upload_scalar(lr, op="H-8"))
        for k, (p_tt, p_ref) in enumerate(zip(params, tparams)):
            rep = rec(f"adam shapes={len(shapes)} step={step} param{k}", compare(ctx.download(p_tt, op="H-6", batch_sharded=False), p_ref.detach(), TOLERANCES["adam_step"]))
            assert rep.passed, str(rep)


# ---- clip_grad_norm -------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("scale, max_norm", [(0.01, 1.0), (1.0, 1.0), (30.0, 1.0), (30.0, 0.5), (5.0, 100.0)])
def test_clip_grad_norm_matches_torch(tt_ctx, scale, max_norm):
    from rsl_rl_tt.optim import clip_grad_norm

    g = torch.Generator().manual_seed(11)
    grads = [torch.randn(64, 128, generator=g) * scale, torch.randn(1, 128, generator=g) * scale, torch.randn(128, 32, generator=g) * scale]
    tp = [torch.nn.Parameter(torch.zeros_like(x)) for x in grads]
    for p, x in zip(tp, grads):
        p.grad = x.clone()
    ref_norm = torch.nn.utils.clip_grad_norm_(tp, max_norm)
    out, total = clip_grad_norm(tt_ctx, [tt_ctx.upload(x, op="H-6", batch_sharded=False) for x in grads], max_norm)
    rn = rec(f"clip norm scale={scale} max={max_norm}", compare(tt_ctx.download(total, op="H-5", batch_sharded=False)[0, 0].reshape(1), ref_norm.reshape(1), TOLERANCES["clip_grad_norm"]))
    assert rn.passed, str(rn)
    for k, (o, p) in enumerate(zip(out, tp)):
        rep = rec(f"clip grad scale={scale} max={max_norm} g{k}", compare(tt_ctx.download(o, op="H-5", batch_sharded=False), p.grad, TOLERANCES["clip_grad_norm"]))
        assert rep.passed, str(rep)


# ---- adaptive learning rate ----------------------------------------------------------------------------------------------------------
def upstream_lr_rule(lr, kl, desired_kl):
    if kl > desired_kl * 2.0:
        return max(1e-5, lr / 1.5)
    if kl < desired_kl / 2.0 and kl > 0.0:
        return min(1e-2, lr * 1.5)
    return lr


def test_adaptive_lr_rule_matches_upstream_for_every_branch_and_the_clamps(tt_ctx):
    from rsl_rl_tt.optim import TTAdaptiveLR

    desired = 0.01
    sched = TTAdaptiveLR(tt_ctx, 1e-3, desired, "adaptive")
    lr = 1e-3
    kls = [0.05, 0.05, 0.002, 0.002, 0.01, 0.0, 0.0049, 0.0051, 0.0200001, 0.02, 1e-9] + [0.5] * 12 + [1e-6] * 30
    for kl in kls:
        sched.update(tt_ctx.upload_scalar(kl, op="H-8"))
        lr = upstream_lr_rule(lr, kl, desired)
        got = float(tt_ctx.download(sched.lr, op="H-5", batch_sharded=False)[0, 0])
        assert got == pytest.approx(lr, rel=1e-6, abs=0), (kl, got, lr)  # fp32 vs python float64 over up to 30 multiplications by 1.5
    assert lr == 1e-2  # the 1e-6 tail ends at the upper clamp (the 0.5 run before it visited the lower one)


def test_fixed_schedule_never_changes_the_learning_rate(tt_ctx):
    from rsl_rl_tt.optim import TTAdaptiveLR

    sched = TTAdaptiveLR(tt_ctx, 3e-4, 0.01, "fixed")
    sched.update(tt_ctx.upload_scalar(1.0, op="H-8"))
    assert float(tt_ctx.download(sched.lr, op="H-5", batch_sharded=False)[0, 0]) == pytest.approx(3e-4, rel=1e-7)
    sched = TTAdaptiveLR(tt_ctx, 3e-4, None, "adaptive")
    assert not sched.adaptive
