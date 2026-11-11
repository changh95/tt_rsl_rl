# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.4 device tests: `TTDistillation` against upstream `Distillation` through the UNMODIFIED `DistillationRunner` - the rollout step by step (student actions, teacher actions), a whole
`update()` from identical state on identical data (every loss type, window length and epoch count, incl. the leftover steps that are never applied), the runner behaviour (a missing teacher
is refused), and checkpoints in both directions (the RL checkpoint the teacher comes from, the distillation checkpoint).

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8d -- python -m pytest -m tt tests/tt/test_tt_distillation.py -s -q -p no:cacheprovider

Single-chip semantics for the whole-batch references (the update parity); the rollout / runner / checkpoint tests run on every mesh.
"""

import copy
import dataclasses
import json
import torch
from pathlib import Path

import pytest

from rsl_rl.runners import DistillationRunner
from rsl_rl_tt.testing.envs import make_env
from rsl_rl_tt.testing.parity import TOLERANCES, UPDATE_TOLERANCES, compare
from rsl_rl_tt.testing.reference import REF_SPECS, build_distillation_runner, make_distill_cfg, make_teacher_checkpoint, test_mesh_shape
from tests.tt.test_tt_checkpoint import shape_tree

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
MEASURED: dict = {}
STEPS = 8
_N = test_mesh_shape()[0] * test_mesh_shape()[1]


@pytest.fixture(scope="module", autouse=True)
def _dump():
    yield
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase8_4_distillation.json").write_text(json.dumps(MEASURED, indent=2, default=str))


def rec(key, value):
    MEASURED[key] = value
    print(f"[measure] {key}: {value}")
    return value


@pytest.fixture(scope="module")
def teachers(tmp_path_factory):
    """Teachers trained for a few iterations by upstream PPO on the CPU: the RL checkpoints a distillation starts from (cartpole: its actor sees `policy`; tracking12: `policy` + `privileged`)."""
    d = tmp_path_factory.mktemp("teachers")
    return {name: make_teacher_checkpoint(name, str(d / f"{name}.pt"), seed=0, iterations=8, num_envs=128) for name in ("cartpole", "tracking12")}


def spec_of(name):
    return dataclasses.replace(REF_SPECS[name], num_envs=max(128, 32 * _N), num_steps=STEPS, iterations=1)


def build(name, tmp_path, teachers, tt=True, algorithm=None, tt_cfg=None, seed=0):
    spec = spec_of(name)
    return build_distillation_runner(name, seed, str(tmp_path), teachers[name], tt=tt, tt_cfg=tt_cfg, algorithm=algorithm, spec=spec)


def dl(ctx, t):
    return ctx.download(t, op="H-6")


# ---- the rollout ---------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
def test_rollout_matches_the_upstream_models_step_by_step(tt_mesh, name, teachers, tmp_path):
    runner, env, spec = build(name, tmp_path, teachers)
    alg = runner.alg
    assert type(alg).__name__ == "TTDistillation" and type(runner).__name__ == "DistillationRunner" and alg.teacher_loaded
    ref_student, ref_teacher = copy.deepcopy(alg._raw_student), copy.deepcopy(alg._raw_teacher)
    ctx = alg.ctx
    alg.train_mode()
    obs = env.get_observations()
    A = env.num_actions
    with torch.inference_mode():
        for step in range(STEPS):
            rng = torch.get_rng_state()
            actions = alg.act(obs)
            teacher_dev = dl(ctx, alg._pending["teacher"])[:, :A]
            torch.set_rng_state(rng)
            ref_actions = ref_student(obs, stochastic_output=True)
            ref_t = ref_teacher(obs)
            assert actions.shape == (env.num_envs, A) and actions.dtype == torch.float32
            rep_a = compare(actions, ref_actions, TOLERANCES["mlp_forward"], f"{name} step {step} student actions")
            rep_t = compare(teacher_dev, ref_t, TOLERANCES["mlp_forward"], f"{name} step {step} teacher actions")
            assert rep_a.passed and rep_t.passed, str(rep_a) + str(rep_t)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
    assert alg.storage.step == STEPS
    rec(f"rollout {name}", {"student_rel_fro": rep_a.rel_fro, "teacher_rel_fro": rep_t.rel_fro})


def test_the_runner_refuses_to_learn_without_a_teacher_and_logs_the_behavior_loss(tt_mesh, teachers, tmp_path):
    name = "cartpole"
    spec = spec_of(name)
    torch.manual_seed(0)
    env = make_env(name, spec.num_envs, seed=0)
    cfg = make_distill_cfg(name, spec)
    cfg["algorithm"]["class_name"] = "rsl_rl_tt.distillation:TTDistillation"
    cfg["algorithm"]["tt_cfg"] = {"mesh_shape": test_mesh_shape()}
    runner = DistillationRunner(env, cfg, log_dir=str(tmp_path), device="cpu")
    with pytest.raises(ValueError, match="Teacher model parameters not loaded"):
        runner.learn(num_learning_iterations=1)
    runner.load(teachers[name], map_location="cpu")
    runner.learn(num_learning_iterations=4)
    rows = (tmp_path / "metrics.csv").read_text().splitlines()
    cols = rows[0].split(",")
    losses = [float(r.split(",")[cols.index("Loss/behavior")]) for r in rows[1:]]
    assert len(losses) == 4 and all(0.0 < v < 1e3 for v in losses) and losses[-1] < losses[0], losses
    assert "Policy/mean_std" in cols  # the student's std (a parameter that distillation never trains) is readable
    ctx = runner.alg.ctx
    host = ctx.host_ops.snapshot()["calls"]
    assert host.get("H-9") is None and set(host) <= {"H-1", "H-2", "H-3", "H-4", "H-5", "H-6", "H-8"}, host


# ---- the update ------------------------------------------------------------------------------------------------------------------------------------
def tt_student_params(alg):
    ctx = alg.ctx
    out = {}
    for i, layer in enumerate(alg.tt_student.mlp.layers):
        out[f"mlp.{i}.weight"] = layer.weight_from_host(ctx.download(layer.w, op="H-6", batch_sharded=False))
        out[f"mlp.{i}.bias"] = layer.bias_from_host(ctx.download(layer.b, op="H-6", batch_sharded=False))
    return out


def ref_student_params(model):
    lins = [m for m in model.mlp if type(m).__name__ == "Linear"]
    out = {}
    for i, lin in enumerate(lins):
        out[f"mlp.{i}.weight"], out[f"mlp.{i}.bias"] = lin.weight.detach().clone(), lin.bias.detach().clone()
    return out


def student_function(params, model, obs):
    """The student's mean action for an arbitrary parameter set (torch fp32, the reference model's normalizer)."""
    lin_idx = [i for i, m in enumerate(model.mlp) if type(m).__name__ == "Linear"]
    mapping = {}
    for k, mod_i in enumerate(lin_idx):
        mapping[f"mlp.{mod_i}.weight"], mapping[f"mlp.{mod_i}.bias"] = params[f"mlp.{k}.weight"], params[f"mlp.{k}.bias"]
    with torch.no_grad():
        return torch.func.functional_call(model, mapping, (obs,))


def rollout_both(alg, ref_alg, env, obs):
    """One rollout of the TT algorithm; every transition is mirrored into the upstream storage (observations, the student's actions, the teacher's actions from the DEVICE, rewards, dones)."""
    ctx = alg.ctx
    A = env.num_actions
    ref_alg.storage.clear()
    with torch.inference_mode():
        for _ in range(STEPS):
            actions = alg.act(obs)
            tr = ref_alg.transition
            tr.observations = obs
            tr.actions = actions.clone()
            tr.privileged_actions = dl(ctx, alg._pending["teacher"])[:, :A].clone()
            obs, rewards, dones, extras = env.step(actions)
            tr.rewards = rewards.clone()
            tr.dones = dones
            ref_alg.storage.add_transition(tr)
            tr.clear()
            alg.process_env_step(obs, rewards, dones, extras)
    return obs


def resync(alg, ref_alg):
    alg._raw_student.load_state_dict(ref_alg._raw_student.state_dict())
    alg.optimizer.load_state_dict(copy.deepcopy(ref_alg.optimizer.state_dict()))
    alg.sync_from_torch()


VARIANTS = {
    "one-window": dict(num_learning_epochs=1, gradient_length=8, loss_type="mse", max_grad_norm=1.0),
    "four-windows": dict(num_learning_epochs=2, gradient_length=4, loss_type="mse", max_grad_norm=None),
    "leftover-steps": dict(num_learning_epochs=2, gradient_length=6, loss_type="mse", max_grad_norm=1.0),  # 16 steps: 2 windows of 6, 4 steps accumulated and never applied
    "every-step-huber": dict(num_learning_epochs=3, gradient_length=1, loss_type="huber", max_grad_norm=0.5),
    "huber-outside-the-quadratic-zone": dict(num_learning_epochs=1, gradient_length=2, loss_type="huber", max_grad_norm=None, learning_rate=2e-2),
}


@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_update_matches_upstream_from_identical_state(tt_mesh, name, variant, teachers, tmp_path):
    """One `TTDistillation.update()` vs one `Distillation.update()` from the SAME student / normalizer / Adam state on the SAME data: the behaviour loss and the student's parameters in
    function space (against upstream with matmul-sized gradient noise injected, the baseline of the PPO update tests), and the student's normalizer statistics."""
    algorithm = VARIANTS[variant]
    runner, env, spec = build(name, tmp_path / "tt", teachers, algorithm=algorithm)
    ref_runner, _, _ = build(name, tmp_path / "ref", teachers, tt=False, algorithm=algorithm)
    alg, ref_alg = runner.alg, ref_runner.alg
    for k, v in alg._raw_student.state_dict().items():
        assert torch.equal(v, ref_alg._raw_student.state_dict()[k]), f"initial student differs at {k}"
    alg.train_mode()
    ref_alg.train_mode()
    obs = env.get_observations()
    for it in range(2):  # the normalizer, Adam's moments are no longer at their initial values
        obs = rollout_both(alg, ref_alg, env, obs)
        torch.manual_seed(1000)
        ref_alg.update()
        torch.manual_seed(1000)
        alg.update()
        resync(alg, ref_alg)
    obs = rollout_both(alg, ref_alg, env, obs)
    resync(alg, ref_alg)
    noisy = copy.deepcopy(ref_alg)  # same student, same Adam state, same stored rollout: only its linear layers get noise
    # The baseline of the update tests: matmul-sized noise (4e-3 of the rms of a layer's output, the measured floor of `matmul_fp32`) at the OUTPUT OF EVERY LINEAR LAYER of the student, so that the
    # backward pass sees noisy activations too. (The PPO tests inject it into the gradients; here the gradient is the difference between the student's output and the teacher's, which shrinks as the
    # student learns while the forward noise does not, so the noise that matters is the forward one.)
    g = torch.Generator().manual_seed(0)
    for m in noisy.student.mlp:
        if type(m).__name__ == "Linear":
            m.register_forward_hook(lambda mod, inp, out: out + 4e-3 * out.detach().pow(2).mean().sqrt() * torch.randn(out.shape, generator=g))
    init = ref_student_params(ref_alg._raw_student)
    obs_eval = ref_alg.storage.observations.flatten(0, 1).clone()
    ref_loss, noisy_loss, tt_loss = ref_alg.update(), noisy.update(), alg.update()
    rep = compare(torch.tensor([tt_loss["behavior"]]), torch.tensor([ref_loss["behavior"]]), TOLERANCES["update_behavior_loss"], f"{name}/{variant} loss.behavior")
    rec(f"update {name}/{variant} loss", {"tt": tt_loss["behavior"], "ref": ref_loss["behavior"], "rel": rep.rel_fro})
    if alg.ctx.n_chips == 1:  # on several chips the logged loss is chip 0's local mean (like rank 0 of upstream's multi-GPU run), not the mean over all environments
        assert rep.passed, str(rep)
    p_ref, p_tt, p_noisy = ref_student_params(ref_alg._raw_student), tt_student_params(alg), ref_student_params(noisy._raw_student)
    f = lambda p: student_function(p, ref_alg._raw_student, obs_eval)
    f_ref, f_tt, f_noisy, f_0 = f(p_ref), f(p_tt), f(p_noisy), f(init)
    dev = lambda a: float((a - f_ref).double().norm() / (f_ref - f_0).double().norm().clamp(min=1e-30))
    got, noise = dev(f_tt), dev(f_noisy)
    rec(f"update {name}/{variant} function_deviation", {"tt": got, "noisy_ref": noise})
    applies = (alg.num_learning_epochs * STEPS) // alg.gradient_length
    # one or two Adam steps: the noise baseline (10x + 1e-3) as in the PPO update tests. Three or more on this consistent, smooth regression loss: the sign flips of the near-zero-gradient components
    # that the first step creates stay, and later steps move them on in the (consistent) wrong direction: the whole-update bound of the many-step tests (TOLERANCES.md, Phase 8.4)
    limit = UPDATE_TOLERANCES["noise_factor"] * noise + UPDATE_TOLERANCES["noise_abs"] if applies <= 2 else UPDATE_TOLERANCES["distill_many_steps"]
    assert got <= limit, f"{name}/{variant}: {got:.3e} of the update differs (limit {limit:.3e}; noise baseline {noise:.3e}; {applies} Adam steps)"
    s, e = alg.tt_student.normalizer.host_stats(), ref_alg._raw_student.obs_normalizer
    assert compare(s["mean"], e.mean, TOLERANCES["normalizer_update"]).passed and compare(s["var"], e._var.reshape(-1), TOLERANCES["normalizer_update"]).passed and s["count"] == int(e.count)
    assert not any(torch.equal(init[k], p_tt[k]) for k in init), "the student did not change"


@pytest.mark.skipif(_N > 1, reason="reads the chip-local accumulators of a single chip")
def test_the_gradient_is_the_sum_of_the_per_step_gradients_not_their_mean(tt_mesh, teachers, tmp_path):
    """Upstream sums the LOSSES of a window before one backward pass: with gradient_length = 4 and Adam disabled (lr 0 is no test), compare the accumulated gradient with torch's: sum_t grad L_t."""
    name = "cartpole"
    runner, env, spec = build(name, tmp_path, teachers, algorithm=dict(num_learning_epochs=1, gradient_length=STEPS, loss_type="mse", max_grad_norm=None))
    alg = runner.alg
    ctx = alg.ctx
    alg.train_mode()
    ref = copy.deepcopy(alg._raw_student)
    obs = env.get_observations()
    observed, teacher = [], []
    with torch.no_grad():  # (not inference mode: the observations are used by autograd below)
        for _ in range(STEPS):
            observed.append(obs.clone())
            actions = alg.act(obs)
            teacher.append(dl(ctx, alg._pending["teacher"])[:, : env.num_actions].clone())
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
    for t in range(STEPS):  # eager steps accumulate; the apply is not run
        alg._step_graph(t)
    loss = 0
    for o, tgt in zip(observed, teacher):
        loss = loss + torch.nn.functional.mse_loss(ref(o), tgt)
    loss.backward()
    lins = [m for m in ref.mlp if type(m).__name__ == "Linear"]
    for k, (lin, layer) in enumerate(zip(lins, alg.tt_student.mlp.layers)):
        dw = layer.weight_from_host(ctx.download(alg._acc_grads[2 * k], op="H-6", batch_sharded=False))
        db = layer.bias_from_host(ctx.download(alg._acc_grads[2 * k + 1], op="H-6", batch_sharded=False))
        assert compare(dw, lin.weight.grad, "mlp_backward", f"layer {k} dW").passed and compare(db, lin.bias.grad, "mlp_backward", f"layer {k} db").passed


# ---- checkpoints -----------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
def test_tt_checkpoint_loads_into_upstream_distillation_with_identical_structure(tt_mesh, name, teachers, tmp_path):
    runner, env, spec = build(name, tmp_path / "tt", teachers)
    runner.learn(num_learning_iterations=3)
    ckpt = tmp_path / "tt.pt"
    runner.save(str(ckpt))
    saved = torch.load(ckpt, weights_only=False)
    assert {"student_state_dict", "teacher_state_dict", "optimizer_state_dict"} <= set(saved)

    up, _, _ = build(name, tmp_path / "up", teachers, tt=False)  # the upstream run loads the teacher from the RL checkpoint, then the distillation checkpoint
    up.load(str(ckpt), map_location="cpu")
    obs = env.get_observations()
    with torch.no_grad():
        assert torch.equal(up.get_inference_policy()(obs), runner.get_inference_policy()(obs)), "the loaded upstream student must equal the TT mirror bit for bit"
    ref, ref_env, _ = build(name, tmp_path / "ref", teachers, tt=False)
    ref.learn(num_learning_iterations=3)
    ref.save(str(tmp_path / "ref.pt"))
    other = torch.load(tmp_path / "ref.pt", weights_only=False)
    assert set(saved) == set(other)
    for key in ("student_state_dict", "teacher_state_dict", "optimizer_state_dict"):
        assert shape_tree(saved[key]) == shape_tree(other[key]), f"{key} differs structurally from an upstream distillation checkpoint"
    teacher_ck = torch.load(teachers[name], weights_only=False)["actor_state_dict"]
    for k, v in saved["teacher_state_dict"].items():
        assert torch.equal(v, teacher_ck[k]), f"the teacher changed during the distillation: {k}"
    assert all(torch.isfinite(s["exp_avg"]).all() and (s["exp_avg_sq"] >= 0).all() for s in saved["optimizer_state_dict"]["state"].values())
    # a TT run resumes from it
    runner.load(str(ckpt), map_location="cpu")
    runner.learn(num_learning_iterations=1)


@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
def test_upstream_distillation_checkpoint_resumes_on_tt_with_identical_device_state(tt_mesh, name, teachers, tmp_path):
    up, _, _ = build(name, tmp_path / "up", teachers, tt=False)
    up.learn(num_learning_iterations=3)
    ckpt = tmp_path / "up.pt"
    up.save(str(ckpt))
    saved = torch.load(ckpt, weights_only=False)

    runner, env, _ = build(name, tmp_path / "tt", teachers, seed=7)  # a DIFFERENT initialisation: everything must come from the checkpoint
    runner.load(str(ckpt), map_location="cpu")
    alg = runner.alg
    ctx = alg.ctx
    for tag, model, sd in (("student", alg.tt_student, saved["student_state_dict"]), ("teacher", alg.tt_teacher, saved["teacher_state_dict"])):
        lin_names = sorted({k.rsplit(".", 1)[0] for k in sd if k.startswith("mlp.") and k.endswith("weight")}, key=lambda s: int(s.split(".")[1]))
        for layer, base in zip(model.mlp.layers, lin_names):
            w = layer.weight_from_host(ctx.download(layer.w, op="H-6", batch_sharded=False))
            b = layer.bias_from_host(ctx.download(layer.b, op="H-6", batch_sharded=False))
            assert torch.equal(w, sd[f"{base}.weight"]) and torch.equal(b, sd[f"{base}.bias"]), f"{tag} {base}"
        stats = model.normalizer.host_stats()
        assert torch.equal(stats["mean"], sd["obs_normalizer._mean"].reshape(-1)) and stats["count"] == int(sd["obs_normalizer.count"]), f"{tag} normalizer"
    state = saved["optimizer_state_dict"]["state"]
    assert alg.adam.t == int(next(iter(state.values()))["step"])
    for slot_p, idx, kind, layer in alg._param_slots():
        opt_idx = next(i for i, q in enumerate(alg.optimizer.param_groups[0]["params"]) if q is slot_p)
        for store, key in ((alg.adam.m, "exp_avg"), (alg.adam.v, "exp_avg_sq")):
            h = ctx.download(store[idx], op="H-6", batch_sharded=False)
            got = layer.weight_from_host(h) if kind == "weight" else layer.bias_from_host(h)
            assert torch.equal(got, state[opt_idx][key]), f"{key} of optimizer slot {opt_idx} ({kind})"
    before = copy.deepcopy(tt_student_params(alg))
    runner.learn(num_learning_iterations=2)
    after = tt_student_params(alg)
    assert any(not torch.equal(before[k], after[k]) for k in before) and all(torch.isfinite(v).all() for v in after.values())


def test_the_teacher_is_loaded_from_an_rl_checkpoint_and_nothing_else_changes(tt_mesh, teachers, tmp_path):
    """Upstream's rule for a checkpoint with an `actor_state_dict`: only the teacher is loaded (not the student, not the optimizer, not the iteration)."""
    runner, env, spec = build("cartpole", tmp_path, teachers, seed=3)
    alg = runner.alg
    ctx = alg.ctx
    fresh = build("cartpole", tmp_path / "fresh", teachers, tt=False, seed=3)[0].alg
    saved = torch.load(teachers["cartpole"], weights_only=False)["actor_state_dict"]
    for layer, k in zip(alg.tt_teacher.mlp.layers, (0, 2, 4)):
        w = layer.weight_from_host(ctx.download(layer.w, op="H-6", batch_sharded=False))
        assert torch.equal(w, saved[f"mlp.{k}.weight"])
    for k, v in alg._raw_student.state_dict().items():
        assert torch.equal(v, fresh._raw_student.state_dict()[k]), f"the student changed by loading the teacher: {k}"
    assert runner.current_learning_iteration == 0
