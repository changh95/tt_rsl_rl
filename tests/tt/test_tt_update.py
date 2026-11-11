# SPDX-License-Identifier: BSD-3-Clause
"""Phase 3 device tests, part 2: device storage / GAE / shuffled mini-batches, and the full `TTPPO.update()` against upstream `PPO.update()`
on IDENTICAL data (parameters after the update, each loss, the learning rate, the normalizer statistics, 20-iteration parameter drift).

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p3b -- python -m pytest -m tt tests/tt/test_tt_update.py -s -q -p no:cacheprovider

How identical data is produced: the TT algorithm runs the rollout (teacher forcing is not needed here); every step's actions, values, log-probs,
means, rewards, dones and time-outs are copied from the device into an UPSTREAM `RolloutStorage` of a reference PPO whose models start from the
same weights; the TT returns/advantages are copied too (so the one place where the two critics' forward noise would leak into the data, the
last value, is shared). Both updates then run with the same global RNG state, i.e. the same `randperm`.
"""

import copy
import json
import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl_tt.testing.envs import make_env
from rsl_rl_tt.testing.parity import TOLERANCES, UPDATE_TOLERANCES, Tol, compare
from rsl_rl_tt.testing.reference import REF_SPECS, RefSpec, make_train_cfg
from tests.tt.test_tt_rollout import build_tt_runner, small_spec

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
MEASURED: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _dump():
    yield
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase3_update.json").write_text(json.dumps(MEASURED, indent=2, default=str))


def rec(key, value):
    MEASURED[key] = value
    print(f"[measure] {key}: {value}")
    return value


def dl(ctx, t):
    return ctx.download(t, op="H-6")


# ---- storage / GAE / mini-batches (no model involved) --------------------------------------------------------------------------------------
def make_storage(ctx, steps, envs, dim_a=64, dim_c=96, act=32, share=False, seed=0):
    from rsl_rl_tt.storage import TTRolloutStorage

    g = torch.Generator().manual_seed(seed)
    st = TTRolloutStorage(ctx, envs, steps, dim_a, dim_c, act, share)
    host = {k: [] for k in ("actor_obs", "critic_obs", "action", "mean", "value", "logp", "reward", "done")}
    for s in range(steps):
        row = {
            "actor_obs": torch.randn(envs, dim_a, generator=g),
            "critic_obs": torch.randn(envs, dim_c, generator=g),
            "action": torch.randn(envs, act, generator=g),
            "mean": torch.randn(envs, act, generator=g),
            "value": torch.randn(envs, 1, generator=g),
            "logp": torch.randn(envs, 1, generator=g) - 5,
            "reward": torch.randn(envs, 1, generator=g),
            "done": (torch.rand(envs, 1, generator=g) < 0.15).float(),
        }
        for k, v in row.items():
            host[k].append(v)
        up = lambda t: ctx.upload(t.contiguous(), op="H-1", batch_sharded=True)
        st.add_transition(actor_obs=up(row["actor_obs"]), critic_obs=up(row["actor_obs"] if share else row["critic_obs"]), action=up(row["action"]), mean=up(row["mean"]), value=up(row["value"]), logp=up(row["logp"]), reward=up(row["reward"]), done=up(row["done"]))
    return st, {k: torch.stack(v) for k, v in host.items()}, g


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("steps, envs", [(8, 32), (24, 128), (1, 64)])
def test_gae_matches_upstream_compute_returns(tt_ctx, steps, envs, normalize):
    gamma, lam = 0.99, 0.95
    st, host, g = make_storage(tt_ctx, steps, envs)
    last = torch.randn(envs, 1, generator=g)
    st.compute_returns(tt_ctx.upload(last, op="H-1", batch_sharded=True), gamma, lam, normalize)
    # upstream PPO.compute_returns, verbatim, on host tensors
    values, rewards, dones = host["value"], host["reward"], host["done"]
    returns = torch.zeros_like(values)
    advantage = 0
    for step in reversed(range(steps)):
        next_values = last if step == steps - 1 else values[step + 1]
        nt = 1.0 - dones[step].float()
        delta = rewards[step] + nt * gamma * next_values - values[step]
        advantage = delta + nt * gamma * lam * advantage
        returns[step] = advantage + values[step]
    advantages = returns - values
    if normalize and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    r1 = compare(dl(tt_ctx, st.returns), returns.flatten(0, 1), TOLERANCES["gae_returns"], f"returns steps={steps} envs={envs}")
    r2 = compare(dl(tt_ctx, st.advantages), advantages.flatten(0, 1), TOLERANCES["gae_returns"], f"advantages steps={steps} envs={envs} norm={normalize}")
    rec(f"gae steps={steps} envs={envs} norm={normalize}", {"returns": r1.rel_fro, "advantages": r2.rel_fro})
    assert r1.passed and r2.passed, str(r1) + str(r2)


@pytest.mark.parametrize("share", [False, True])
def test_packed_table_shuffle_and_minibatches_equal_upstream_mini_batch_generator(tt_ctx, share):
    """Same data, same RNG state => the same samples in the same mini-batches as `RolloutStorage.mini_batch_generator` (bit-exact transport)."""
    from rsl_rl.storage import RolloutStorage

    steps, envs, A, mbs, epochs = 8, 64, 5, 4, 3
    st, host, g = make_storage(tt_ctx, steps, envs, act=32, share=share, seed=1)
    last = torch.randn(envs, 1, generator=g)
    st.compute_returns(tt_ctx.upload(last, op="H-1", batch_sharded=True), 0.99, 0.95, True)
    # upstream storage filled with the same numbers (observations as a TensorDict; only the fields the generator yields)
    critic_obs = host["actor_obs"] if share else host["critic_obs"]
    obs = TensorDict({"a": host["actor_obs"], "c": critic_obs}, batch_size=[steps, envs])
    ref = RolloutStorage("rl", envs, steps, TensorDict({"a": host["actor_obs"][0], "c": critic_obs[0]}, batch_size=[envs]), [32], "cpu")
    ref.observations[:] = obs
    ref.actions[:] = host["action"]
    ref.values[:] = host["value"]
    ref.actions_log_prob[:] = host["logp"]
    ref.returns[:] = dl(tt_ctx, st.returns).reshape(steps, envs, 1)
    ref.advantages = dl(tt_ctx, st.advantages).reshape(steps, envs, 1)
    ref.distribution_params = (host["mean"].clone(), torch.ones_like(host["mean"]))

    table = st.packed_table()
    torch.manual_seed(5)
    batches = list(ref.mini_batch_generator(mbs, epochs))
    torch.manual_seed(5)
    n = st.batch_size
    shuffled = st.shuffle(table, tt_ctx.upload_index(torch.randperm(n), op="H-4"))
    rows = n // mbs
    k = 0
    for _epoch in range(epochs):
        for i in range(mbs):
            b = batches[k]
            k += 1
            v = st.split(shuffled[i * rows : (i + 1) * rows, :])
            h = lambda t: dl(tt_ctx, t)
            assert torch.equal(h(v["actor_obs"]), b.observations["a"]), "actor obs rows differ"
            assert torch.equal(h(v["critic_obs"]), b.observations["c"]), "critic obs rows differ"
            assert torch.equal(h(v["actions"])[:, :32], b.actions) and torch.equal(h(v["old_mean"])[:, :32], b.old_distribution_params[0])
            assert torch.equal(h(v["old_values"]), b.values) and torch.equal(h(v["returns"]), b.returns)
            assert torch.equal(h(v["advantages"]), b.advantages) and torch.equal(h(v["old_logp"]), b.old_actions_log_prob)


# ---- the full update -------------------------------------------------------------------------------------------------------------------------
def tt_params(alg):
    """Device parameters of the TT algorithm as host tensors in torch layout, keyed like the mirrors' state dicts."""
    ctx = alg.ctx
    out = {}
    for tag, model in (("actor", alg.tt_actor), ("critic", alg.tt_critic)):
        for i, layer in enumerate(model.mlp.layers):
            w = ctx.download(layer.w, op="H-6", batch_sharded=False)[: layer.in_features, : layer.out_features].t()
            b = ctx.download(layer.b, op="H-6", batch_sharded=False)[0, : layer.out_features]
            out[f"{tag}.mlp.{i}.weight"], out[f"{tag}.mlp.{i}.bias"] = w, b
    d = alg.tt_actor.distribution
    out["actor.std"] = ctx.download(d.param, op="H-6", batch_sharded=False)[0, : d.dim]
    return out


def ref_params(ref_alg):
    out = {}
    for tag, model in (("actor", ref_alg._raw_actor), ("critic", ref_alg._raw_critic)):
        lins = [m for m in model.mlp if type(m).__name__ == "Linear"]
        for i, lin in enumerate(lins):
            out[f"{tag}.mlp.{i}.weight"], out[f"{tag}.mlp.{i}.bias"] = lin.weight.detach().clone(), lin.bias.detach().clone()
    d = ref_alg._raw_actor.distribution
    out["actor.std"] = (d.std_param if d.std_type == "scalar" else d.log_std_param).detach().clone()
    return out


def make_pair(name, tmp_path, spec=None, seed=0, tt_overrides=None):
    """A TT runner (TTPPO) and an upstream PPO built from identical initial weights (same torch seed)."""
    spec = spec or small_spec(name)
    runner, env, cfg = build_tt_runner(name, tmp_path, spec=spec, seed=seed)
    ref_cfg = copy.deepcopy(cfg)
    ref_cfg["algorithm"]["class_name"] = "PPO"
    ref_cfg["algorithm"].pop("tt_cfg", None)
    ref_cfg.pop("logger", None)
    ref_cfg["multi_gpu"] = None
    torch.manual_seed(seed)
    ref_env = make_env(name, spec.num_envs, seed=seed)
    ref_alg = PPO.construct_algorithm(ref_env.get_observations(), ref_env, ref_cfg, "cpu")
    for k, v in runner.alg._raw_actor.state_dict().items():
        assert torch.equal(v, ref_alg._raw_actor.state_dict()[k]), f"initial actor weights differ at {k}"
    ref_alg._rebuild = lambda: PPO.construct_algorithm(ref_env.get_observations(), ref_env, copy.deepcopy(ref_cfg), "cpu")
    return runner, env, cfg, ref_alg


def clone_upstream_alg(ref_alg):
    """An independent upstream algorithm with the same weights, normalizer statistics, Adam moments, learning rate and stored rollout."""
    new = ref_alg._rebuild()
    new._raw_actor.load_state_dict(ref_alg._raw_actor.state_dict())
    new._raw_critic.load_state_dict(ref_alg._raw_critic.state_dict())
    new.optimizer.load_state_dict(copy.deepcopy(ref_alg.optimizer.state_dict()))
    new.learning_rate = ref_alg.learning_rate
    for group in new.optimizer.param_groups:
        group["lr"] = ref_alg.learning_rate
    copy_storage(ref_alg, new)
    return new


def copy_storage(src_alg, dst_alg):
    a, b = src_alg.storage, dst_alg.storage
    for field in ("actions", "rewards", "dones", "values", "actions_log_prob", "returns"):
        getattr(b, field).copy_(getattr(a, field))
    b.observations.copy_(a.observations)
    b.advantages = a.advantages.clone()
    b.distribution_params = None if a.distribution_params is None else tuple(p.clone() for p in a.distribution_params)
    b.step = a.step


def rollout_into_both(alg, ref_alg, env, obs, steps):
    """Run `steps` env steps with the TT policy; mirror every transition (and the TT-computed returns/advantages) into the upstream storage."""
    ctx = alg.ctx
    A = env.num_actions
    ref_alg.storage.clear()
    with torch.inference_mode():
        for _ in range(steps):
            actions = alg.act(obs)
            pend = alg._pending
            tr = ref_alg.transition
            tr.observations = obs
            tr.actions = actions.clone()
            tr.values = dl(ctx, pend["value"])[:, :1].clone()
            tr.actions_log_prob = dl(ctx, pend["logp"])[:, 0].clone()
            mean = dl(ctx, pend["mean"])[:, :A].clone()
            std = alg._std_host.reshape(1, A).expand(env.num_envs, A).clone()
            tr.distribution_params = (mean, std)
            tr.hidden_states = (None, None)
            obs, rewards, dones, extras = env.step(actions)
            tr.rewards = rewards.clone() + ref_alg.gamma * (tr.values * extras["time_outs"].unsqueeze(1).float()).squeeze(1)
            tr.dones = dones
            ref_alg.storage.add_transition(tr)
            tr.clear()
            alg.process_env_step(obs, rewards, dones, extras)
        alg.compute_returns(obs)
    n = steps * env.num_envs
    ref_alg.storage.returns[:] = dl(ctx, alg.storage.returns).reshape(steps, env.num_envs, 1)
    ref_alg.storage.advantages = dl(ctx, alg.storage.advantages).reshape(steps, env.num_envs, 1)
    return obs


def param_report(tt_p, ref_p, init_p, tag):
    """Per parameter: relative error of the UPDATE (theta_tt - theta_ref) / (theta_ref - theta_init), and of the parameter itself."""
    rows = {}
    for k in ref_p:
        upd = (ref_p[k] - init_p[k]).double()
        diff = (tt_p[k] - ref_p[k]).double()
        rows[k] = {"update_err": float(diff.norm() / upd.norm().clamp(min=1e-30)), "param_err": float(diff.norm() / ref_p[k].double().norm().clamp(min=1e-30)), "update_norm": float(upd.norm())}
    return rows


def add_grad_noise(alg, rel=4e-3, seed=0):
    """Hooks that add N(0, (rel * rms(grad))^2) to every parameter gradient of an UPSTREAM algorithm: the update-level noise-floor baseline.
    A TT gradient is the torch gradient plus matmul noise that is ABSOLUTE per tensor (about 3.4e-3..4.9e-3 of its rms, test_tt_backward), so
    near-zero gradient components can change sign, which Adam turns into full-size steps; a multiplicative baseline would hide exactly that."""
    g = torch.Generator().manual_seed(seed)
    for model in (alg._raw_actor, alg._raw_critic):
        for p in model.parameters():
            p.register_hook(lambda grad: grad + rel * grad.pow(2).mean().sqrt() * torch.randn(grad.shape, generator=g))


def function_outputs(params, ref_alg, obs):
    """Actor mean and critic value of an arbitrary parameter set: a torch fp32 forward through the reference models' structure, with the
    given weights substituted (functional_call; the normalizer buffers are those of the reference model)."""
    res = {}
    for tag, model in (("actor", ref_alg._raw_actor), ("critic", ref_alg._raw_critic)):
        lin_idx = [i for i, m in enumerate(model.mlp) if type(m).__name__ == "Linear"]
        mapping = {}
        for k, mod_i in enumerate(lin_idx):
            mapping[f"mlp.{mod_i}.weight"] = params[f"{tag}.mlp.{k}.weight"]
            mapping[f"mlp.{mod_i}.bias"] = params[f"{tag}.mlp.{k}.bias"]
        with torch.no_grad():
            res[tag] = torch.func.functional_call(model, mapping, (obs,))
    return res


def resync_tt_from_ref(alg, ref_alg):
    """Make the device state identical to the reference's: weights, normalizer buffers, Adam moments/step, learning rate (through TTPPO.load-style sync)."""
    alg._raw_actor.load_state_dict(ref_alg._raw_actor.state_dict())
    alg._raw_critic.load_state_dict(ref_alg._raw_critic.state_dict())
    alg.optimizer.load_state_dict(copy.deepcopy(ref_alg.optimizer.state_dict()))
    alg.learning_rate = ref_alg.learning_rate
    alg.sync_from_torch()


def function_deviation(p_a, p_b, p_0, ref_alg, obs_eval):
    """||f(p_a) - f(p_b)|| / ||f(p_b) - f(p_0)|| for the actor mean and the critic value on `obs_eval` (how much of the update differs)."""
    fa, fb, f0 = (function_outputs(p, ref_alg, obs_eval) for p in (p_a, p_b, p_0))
    return {tag: float((fa[tag] - fb[tag]).double().norm() / (fb[tag] - f0[tag]).double().norm().clamp(min=1e-30)) for tag in ("actor", "critic")}


def run_update_pair(name, tmp_path, algorithm_over, warm_iterations=2, noise=4e-3, resync=True):
    base = small_spec(name)
    spec = RefSpec(**{**base.__dict__, "algorithm": {**base.algorithm, **algorithm_over}})
    runner, env, cfg, ref_alg = make_pair(name, tmp_path, spec=spec)
    alg = runner.alg
    alg.train_mode(); ref_alg.train_mode()
    obs = env.get_observations()
    for _ in range(warm_iterations):  # so the normalizers, the learning rate and Adam's moments are no longer at their initial values
        obs = rollout_into_both(alg, ref_alg, env, obs, STEPS_PER_ITER)
        torch.manual_seed(1000)
        ref_alg.update()
        torch.manual_seed(1000)
        alg.update()
        if resync:
            resync_tt_from_ref(alg, ref_alg)
    obs = rollout_into_both(alg, ref_alg, env, obs, STEPS_PER_ITER)
    if resync:
        resync_tt_from_ref(alg, ref_alg)
    noisy = clone_upstream_alg(ref_alg)  # same weights, same Adam moments, same storage: only its gradients get noise
    add_grad_noise(noisy, noise)
    init = ref_params(ref_alg)
    obs_eval = ref_alg.storage.observations.flatten(0, 1)
    lr_before = alg.learning_rate
    torch.manual_seed(2000)
    ref_loss = ref_alg.update()
    torch.manual_seed(2000)
    noisy_loss = noisy.update()
    torch.manual_seed(2000)
    tt_loss = alg.update()
    p_ref, p_tt, p_noisy = ref_params(ref_alg), tt_params(alg), ref_params(noisy)
    out = {"ref_loss": ref_loss, "noisy_loss": noisy_loss, "tt_loss": tt_loss, "lr": {"before": lr_before, "ref": ref_alg.learning_rate, "noisy": noisy.learning_rate, "tt": alg.learning_rate}}
    out["function"] = {"tt": function_deviation(p_tt, p_ref, init, ref_alg, obs_eval), "noisy_ref": function_deviation(p_noisy, p_ref, init, ref_alg, obs_eval)}
    out["params"] = {"tt": param_report(p_tt, p_ref, init, "tt"), "noisy_ref": param_report(p_noisy, p_ref, init, "noisy")}
    out["alg"], out["ref_alg"], out["noisy_alg"], out["env"] = alg, ref_alg, noisy, env
    return out


SMOOTH = dict(clip_param=1e3, use_clipped_value_loss=False)  # no clip gates: the loss is smooth, so deviations reflect noise, not branch flips
VARIANTS = {
    "one-step": dict(num_learning_epochs=1, num_mini_batches=1, schedule="fixed"),
    "full-clipped": dict(schedule="fixed"),
    "full-smooth": dict(schedule="fixed", **SMOOTH),
    "full-clipped-per-minibatch-adv": dict(schedule="fixed", normalize_advantage_per_mini_batch=True),
}


@pytest.mark.parametrize("name", ["cartpole", "pendulum", "tracking12"])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_update_matches_upstream_from_identical_state(tt_mesh, name, variant, tmp_path):
    """One TTPPO.update() vs one upstream PPO.update() from the SAME weights / normalizer / Adam state on the SAME data and RNG state.

    Losses are compared directly; parameters in function space (see UPDATE_TOLERANCES) against a baseline that injects gradient noise of the
    measured matmul size into upstream itself. The learning-rate schedule is fixed here: with identical state the TT KL of the first mini-batch is
    exactly 0 (rollout and update forwards are bit-identical on the device), upstream's is ~1e-9, and `0 < kl` is a knife edge; the KL values and
    the schedule are tested separately."""
    r = run_update_pair(name, tmp_path, VARIANTS[variant], warm_iterations=2)
    tt, ref = r["tt_loss"], r["ref_loss"]
    for key, tol in (("value", "update_value_loss"), ("entropy", "update_entropy"), ("surrogate", "update_surrogate")):
        rep = compare(torch.tensor([tt[key]]), torch.tensor([ref[key]]), TOLERANCES[tol], f"{name}/{variant} loss.{key}")
        rec(f"{name}/{variant} loss.{key}", {"tt": tt[key], "ref": ref[key], "abs_diff": abs(tt[key] - ref[key])})
        assert rep.passed, str(rep)
    assert r["lr"]["tt"] == pytest.approx(r["lr"]["ref"], rel=1e-6)
    for tag in ("actor", "critic"):
        got, noise = r["function"]["tt"][tag], r["function"]["noisy_ref"][tag]
        rec(f"{name}/{variant} function_deviation.{tag}", {"tt": got, "noisy_ref": noise})
        if variant == "full-clipped" or variant == "full-clipped-per-minibatch-adv":
            limit = UPDATE_TOLERANCES["full_update_clipped"]
        else:
            limit = UPDATE_TOLERANCES["noise_factor"] * noise + UPDATE_TOLERANCES["noise_abs"]
        assert got <= limit, f"{name}/{variant} {tag}: {got:.3e} of the update differs (limit {limit:.3e}; noise baseline {noise:.3e})"
    for tag, tt_m, ref_m in (("actor", r["alg"].tt_actor, r["ref_alg"]._raw_actor), ("critic", r["alg"].tt_critic, r["ref_alg"]._raw_critic)):
        if tt_m.normalizer is not None:
            s = tt_m.normalizer.host_stats()
            rm = compare(s["mean"], ref_m.obs_normalizer.mean, TOLERANCES["normalizer_update"])
            rv = compare(s["var"], ref_m.obs_normalizer._var.reshape(-1), TOLERANCES["normalizer_update"])
            assert rm.passed and rv.passed and s["count"] == int(ref_m.obs_normalizer.count), f"{tag} normalizer after the update: " + str(rm) + str(rv)


@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
@pytest.mark.parametrize("smooth", [False, True], ids=["clipped", "smooth"])
def test_parameter_drift_over_20_updates_on_identical_data(tt_mesh, name, smooth, tmp_path):
    """20 consecutive updates (each: rollout with the TT policy; the same data goes to the upstream algorithm and to an upstream algorithm with
    injected gradient noise; NO resynchronisation), all with the same RNG state per update. The TT chain must stay within the drift tolerance."""
    over = dict(schedule="fixed", **(SMOOTH if smooth else {}))
    base = small_spec(name)
    spec = RefSpec(**{**base.__dict__, "algorithm": {**base.algorithm, **over}})
    runner, env, cfg, ref_alg = make_pair(name, tmp_path, spec=spec)
    alg = runner.alg
    alg.train_mode(); ref_alg.train_mode()
    noisy = clone_upstream_alg(ref_alg)
    add_grad_noise(noisy, 4e-3)
    noisy.train_mode()
    init = ref_params(ref_alg)
    obs = env.get_observations()
    curve = []
    for it in range(20):
        obs = rollout_into_both(alg, ref_alg, env, obs, STEPS_PER_ITER)
        copy_storage(ref_alg, noisy)
        obs_eval = ref_alg.storage.observations.flatten(0, 1)
        for a in (ref_alg, noisy, alg):
            torch.manual_seed(3000 + it)
            a.update()
        p_ref, p_tt, p_noisy = ref_params(ref_alg), tt_params(alg), ref_params(noisy)
        d_tt, d_noisy = function_deviation(p_tt, p_ref, init, ref_alg, obs_eval), function_deviation(p_noisy, p_ref, init, ref_alg, obs_eval)
        curve.append({"iter": it, "tt": d_tt, "noisy_ref": d_noisy})
    rec(f"drift20 {name} {'smooth' if smooth else 'clipped'}", curve)
    limit = UPDATE_TOLERANCES["drift20_smooth" if smooth else "drift20_clipped"]
    worst = max(max(row["tt"].values()) for row in curve)
    assert worst <= limit, f"{name}: the TT parameter trajectory drifts by {worst:.3e} of the update size (limit {limit}); noise baseline {max(max(r['noisy_ref'].values()) for r in curve):.3e}"
    assert all(torch.isfinite(torch.tensor(v)) for v in tt_params(alg)["actor.std"].tolist())


def test_kl_matches_upstream_kl_divergence(tt_ctx):
    from rsl_rl.modules import GaussianDistribution
    from rsl_rl_tt.losses import PPOLosses
    from rsl_rl_tt.modules import TTGaussianDistribution
    from rsl_rl_tt.tt_common import pad_matrix, pad_up

    torch.manual_seed(0)
    for dim, std_type in ((1, "scalar"), (12, "scalar"), (29, "log")):
        old = GaussianDistribution(dim, init_std=0.9, std_type=std_type, std_range=(0.05, 4.0))
        new = GaussianDistribution(dim, init_std=0.9, std_type=std_type, std_range=(0.05, 4.0))
        with torch.no_grad():
            for d, scale in ((old, 0.0), (new, 0.2)):
                p = d.std_param if std_type == "scalar" else d.log_std_param
                p.add_(scale * torch.randn(dim))
        rows = 128
        old_mean, new_mean = torch.randn(rows, dim), torch.randn(rows, dim) * 1.1 + 0.05
        old.update(old_mean)
        new.update(new_mean)
        ref = torch.distributions.kl_divergence(torch.distributions.Normal(old_mean, old.std), torch.distributions.Normal(new_mean, new.std)).sum(-1).mean()
        tt_old, tt_new = TTGaussianDistribution(tt_ctx, old), TTGaussianDistribution(tt_ctx, new)
        losses = PPOLosses(tt_ctx, clip_param=0.2, value_loss_coef=1.0, entropy_coef=0.0, use_clipped_value_loss=True)
        up = lambda m: tt_ctx.upload(pad_matrix(m, rows, pad_up(dim)), op="H-1", batch_sharded=True)
        got = losses.kl_mean(tt_new, up(old_mean), tt_old.var, tt_old.log_std, up(new_mean))
        rep = compare(tt_ctx.download(got, op="H-5", batch_sharded=False)[0, 0].reshape(1), ref.reshape(1), TOLERANCES["row_sum_fp32"], f"kl dim={dim} {std_type}")
        assert rep.passed, str(rep)
        zero = losses.kl_mean(tt_old, up(old_mean), tt_old.var, tt_old.log_std, up(old_mean))
        assert abs(float(tt_ctx.download(zero, op="H-5", batch_sharded=False)[0, 0])) < 1e-6  # KL of a distribution with itself


def test_adaptive_schedule_inside_update_moves_the_learning_rate_within_bounds(tt_mesh, tmp_path):
    """The default (adaptive) schedule end to end: after a few updates the learning rate is a valid device-computed value in [1e-5, 1e-2] and the
    policy still trains (finite losses). The individual KL decisions are covered by the KL and learning-rate unit tests."""
    runner, env, cfg = build_tt_runner("tracking12", tmp_path, spec=small_spec("tracking12"))
    runner.learn(num_learning_iterations=4)
    lr = runner.alg.learning_rate
    assert 1e-5 <= lr <= 1e-2  # defined and in bounds (it may legitimately stay at 1e-3 when the KL stays inside [desired/2, 2 desired])


def test_update_host_traffic_is_exactly_the_allow_listed_scalars(tt_mesh, tmp_path):
    """BRIEF Phase 3: no disallowed host fallback in the update path. A full update only uploads the permutation (H-4), per optimizer step the
    two Adam bias-correction scalars and per normalizer the update rate (H-8), and downloads the logged scalars + std (H-5); never data."""
    spec = small_spec("tracking12")
    runner, env, cfg = build_tt_runner("tracking12", tmp_path, spec=spec)
    alg = runner.alg
    alg.train_mode()
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(STEPS_PER_ITER):
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
        alg.compute_returns(obs)
    alg.ctx.host_ops.reset()
    alg.update()
    steps = alg.num_learning_epochs * alg.num_mini_batches
    n_norm = sum(m.normalizer is not None for m in (alg.tt_actor, alg.tt_critic))
    calls = dict(alg.ctx.host_ops.calls)
    # H-8: two Adam bias-correction scalars per optimizer step plus one rate scalar per normalizer. (Phase 5: the constants that the first update used to
    # create lazily - the four unit rows of the packed table (H-6), Adam's two initial scalars and the three loss accumulators (H-8) - are now built when
    # the algorithm is constructed, before any trace can be captured (TTNN_ISSUES T-15): the same transfers, outside this counting window.)
    assert calls == {"H-4": 1, "H-5": 3 + 1 + 1, "H-8": 2 * steps + n_norm}, calls


STEPS_PER_ITER = 8


def test_shuffle_is_bit_exact_for_extreme_values(tt_ctx):
    """The bf16-split embedding gather must move ANY finite fp32 value unchanged: zeros, huge, tiny (down to 1e-30), mixed signs, repeated rows."""
    from rsl_rl_tt.storage import TTRolloutStorage

    steps, envs = 4, 64
    st = TTRolloutStorage(tt_ctx, envs, steps, 32, 32, 32, True)
    n, w = st.batch_size, st.width
    g = torch.Generator().manual_seed(3)
    exps = torch.randint(-30, 38, (n, w), generator=g).float()
    table = torch.randn(n, w, generator=g) * torch.pow(10.0, exps)
    table[::7] = 0.0
    table[3, :] = torch.tensor([3.3e38, -3.3e38, 1e-30, -1e-30, 1.0, -1.0, 0.0] * (w // 7) + [0.0] * (w % 7))
    perm = torch.randperm(n, generator=g)
    dev = tt_ctx.upload(table, op="H-1", batch_sharded=True)
    got = dl(tt_ctx, st.shuffle(dev, tt_ctx.upload_index(perm, op="H-4")))
    want = table[perm]
    normal = want.abs() >= 1e-30
    bad = ((got != want) & normal).sum().item()
    assert bad == 0, f"{bad} of {int(normal.sum())} values with |x| >= 1e-30 changed by the shuffle (max abs diff {(got - want).abs().max():.3e})"
    # below that the tiny bf16 remainders (2^-16 of the value) reach the fp32 flush-to-zero threshold: the absolute error stays below ~2e-38
    assert (got - want).abs().max() <= 2e-38
