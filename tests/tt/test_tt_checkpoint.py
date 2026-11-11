# SPDX-License-Identifier: BSD-3-Clause
"""Phase 4 device tests: checkpoint compatibility in both directions and policy export, all through the UNMODIFIED upstream OnPolicyRunner.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p4 -- python -m pytest -m tt tests/tt/test_tt_checkpoint.py -s -q -p no:cacheprovider

  * TT save -> upstream `load()` -> identical inference (and the same optimizer-state STRUCTURE as an upstream checkpoint)
  * upstream save -> TT `load()` -> identical device state (weights, normalizer, Adam moments, learning rate) -> training continues
  * TT-trained policy -> upstream JIT and ONNX export -> torch.jit / onnxruntime outputs match
"""

import copy
import numpy as np
import torch

import pytest

from rsl_rl.runners import OnPolicyRunner
from rsl_rl_tt.testing.envs import make_env
from rsl_rl_tt.testing.parity import TOLERANCES, compare
from rsl_rl_tt.testing.reference import make_train_cfg
from tests.tt.test_tt_rollout import build_tt_runner, small_spec

pytestmark = pytest.mark.tt


def build_upstream_runner(name, tmp_path, spec=None, seed=0):
    spec = spec or small_spec(name)
    torch.manual_seed(seed)
    env = make_env(name, spec.num_envs, seed=seed)
    cfg = make_train_cfg(name, spec)
    return OnPolicyRunner(env, cfg, log_dir=str(tmp_path), device="cpu"), env


def shape_tree(obj):
    """Nested keys with (shape, dtype) leaves for tensors and the type name for everything else: the STRUCTURE of a checkpoint entry, values excluded."""
    if isinstance(obj, torch.Tensor):
        return ("tensor", tuple(obj.shape), str(obj.dtype))
    if isinstance(obj, dict):
        return {k: shape_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [shape_tree(v) for v in obj]
    return type(obj).__name__


@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
def test_tt_checkpoint_loads_into_upstream_with_identical_inference_and_structure(tt_mesh, name, tmp_path):
    tt_runner, tt_env, _ = build_tt_runner(name, tmp_path / "tt")
    tt_runner.learn(num_learning_iterations=3)
    ckpt = tmp_path / "model_tt.pt"
    tt_runner.save(str(ckpt), infos={"note": "from the TT runner"})

    up_runner, up_env = build_upstream_runner(name, tmp_path / "up")
    infos = up_runner.load(str(ckpt), map_location="cpu")
    assert infos == {"note": "from the TT runner"} and up_runner.current_learning_iteration == tt_runner.current_learning_iteration

    obs = tt_env.get_observations()
    with torch.no_grad():
        out_up = up_runner.get_inference_policy()(obs)
        out_tt_mirror = tt_runner.get_inference_policy()(obs)
    assert torch.equal(out_up, out_tt_mirror), "the loaded upstream policy must equal the TT mirror bit for bit"
    # the device forward of the TT algorithm agrees with it within the measured composite tolerance
    alg = tt_runner.alg
    dev_out = alg.ctx.download(alg.tt_actor.forward(alg.tt_actor.obs_to_device(obs)), op="H-6")[:, : tt_env.num_actions]
    rep = compare(dev_out, out_up, TOLERANCES["mlp_forward"])
    assert rep.passed, str(rep)
    # normalizer statistics travelled with the state dict
    for tag, tt_m, up_m in (("actor", alg.tt_actor, up_runner.alg._raw_actor), ("critic", alg.tt_critic, up_runner.alg._raw_critic)):
        if tt_m.normalizer is not None:
            s = tt_m.normalizer.host_stats()
            assert s["count"] == int(up_m.obs_normalizer.count) > 0
            assert compare(s["mean"], up_m.obs_normalizer.mean, TOLERANCES["normalizer_update"]).passed

    # --- structure: a TT checkpoint has exactly the keys/shapes/dtypes of an upstream checkpoint of the same configuration ---
    ref_runner, _ = build_upstream_runner(name, tmp_path / "ref")
    ref_runner.learn(num_learning_iterations=3)
    ref_ckpt = tmp_path / "model_up.pt"
    ref_runner.save(str(ref_ckpt), infos={"note": "from the TT runner"})
    a, b = torch.load(ckpt, weights_only=False), torch.load(ref_ckpt, weights_only=False)
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        assert shape_tree(a[key]) == shape_tree(b[key]), f"{key} differs structurally from an upstream checkpoint"
    assert set(a) == set(b) and a["iter"] == b["iter"]
    opt_a, opt_b = a["optimizer_state_dict"], b["optimizer_state_dict"]
    n_params = len(opt_b["param_groups"][0]["params"])
    assert sorted(opt_a["state"]) == list(range(n_params)) and opt_a["param_groups"][0]["params"] == opt_b["param_groups"][0]["params"]
    for k, v in opt_b["param_groups"][0].items():
        if k not in ("lr", "params"):
            assert opt_a["param_groups"][0][k] == v, k  # betas, eps, weight_decay, amsgrad, ... identical
    steps = {int(s["step"]) for s in opt_a["state"].values()}
    assert steps == {int(next(iter(opt_b["state"].values()))["step"])}, f"optimizer step counters: TT {steps}"
    assert all(torch.isfinite(s["exp_avg"]).all() and (s["exp_avg_sq"] >= 0).all() for s in opt_a["state"].values())
    assert opt_a["param_groups"][0]["lr"] == pytest.approx(alg.learning_rate)


@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
def test_upstream_checkpoint_resumes_on_tt_with_identical_device_state(tt_mesh, name, tmp_path):
    up_runner, up_env = build_upstream_runner(name, tmp_path / "up")
    up_runner.learn(num_learning_iterations=3)
    ckpt = tmp_path / "model_up.pt"
    up_runner.save(str(ckpt))
    saved = torch.load(ckpt, weights_only=False)

    tt_runner, tt_env, _ = build_tt_runner(name, tmp_path / "tt", seed=7)  # a DIFFERENT initialisation: everything must come from the checkpoint
    tt_runner.load(str(ckpt), map_location="cpu")
    alg = tt_runner.alg
    ctx = alg.ctx
    assert tt_runner.current_learning_iteration == saved["iter"] and alg.learning_rate == saved["optimizer_state_dict"]["param_groups"][0]["lr"]
    assert float(ctx.download(alg.lr_sched.lr, op="H-6", batch_sharded=False)[0, 0]) == pytest.approx(alg.learning_rate, rel=1e-6)
    # device weights == checkpoint weights (fp32 transfers are exact)
    for tag, model, sd in (("actor", alg.tt_actor, saved["actor_state_dict"]), ("critic", alg.tt_critic, saved["critic_state_dict"])):
        lin_names = sorted({k.rsplit(".", 1)[0] for k in sd if k.startswith("mlp.") and k.endswith("weight")}, key=lambda s: int(s.split(".")[1]))
        for layer, base in zip(model.mlp.layers, lin_names):
            w = ctx.download(layer.w, op="H-6", batch_sharded=False)[: layer.in_features, : layer.out_features].t()
            b = ctx.download(layer.b, op="H-6", batch_sharded=False)[0, : layer.out_features]
            assert torch.equal(w, sd[f"{base}.weight"]) and torch.equal(b, sd[f"{base}.bias"]), f"{tag} {base}"
        if model.normalizer is not None:
            assert torch.equal(model.normalizer.host_stats()["mean"], sd["obs_normalizer._mean"].reshape(-1))
            assert model.normalizer.count == int(sd["obs_normalizer.count"])
    # Adam moments and the step counter came across: compare with the checkpoint, parameter by parameter
    assert alg.adam is not None and alg.adam.t == int(next(iter(saved["optimizer_state_dict"]["state"].values()))["step"])
    state = saved["optimizer_state_dict"]["state"]
    order = alg.optimizer.param_groups[0]["params"]
    for idx_in_opt, p in enumerate(order):
        for slot_p, dev_idx, kind, owner in alg._param_slots():
            if slot_p is p:
                m = ctx.download(alg.adam.m[dev_idx], op="H-6", batch_sharded=False)
                want = state[idx_in_opt]["exp_avg"]
                got = m[: owner.in_features, : owner.out_features].t() if kind == "weight" else (m[0, : owner.out_features] if kind == "bias" else m[0, : owner.dim])
                assert torch.equal(got, want), f"exp_avg of optimizer slot {idx_in_opt} ({kind})"
    # the policy is the same function; TT keeps training from here
    obs = tt_env.get_observations()
    with torch.no_grad():
        ref_out = up_runner.get_inference_policy()(obs)
    dev_out = ctx.download(alg.tt_actor.forward(alg.tt_actor.obs_to_device(obs)), op="H-6")[:, : tt_env.num_actions]
    assert compare(dev_out, ref_out, TOLERANCES["mlp_forward"]).passed
    before = copy.deepcopy(alg.get_policy().state_dict())
    tt_runner.learn(num_learning_iterations=2)
    after = alg.get_policy().state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before if k.endswith("weight")), "training after load did not change the weights"
    assert all(torch.isfinite(v).all() for v in after.values() if v.is_floating_point())
    assert tt_runner.current_learning_iteration == saved["iter"] + 1  # upstream convention: the saved iteration is re-run first, then it advances


def test_partial_load_keeps_the_current_optimizer_state_like_upstream(tt_mesh, tmp_path):
    up_runner, _ = build_upstream_runner("cartpole", tmp_path / "up")
    up_runner.learn(num_learning_iterations=2)
    ckpt = tmp_path / "up.pt"
    up_runner.save(str(ckpt))
    tt_runner, _, _ = build_tt_runner("cartpole", tmp_path / "tt", seed=3)
    tt_runner.learn(num_learning_iterations=2)
    alg = tt_runner.alg
    t_before = alg.adam.t
    m_before = alg.ctx.download(alg.adam.m[0], op="H-6", batch_sharded=False).clone()
    tt_runner.load(str(ckpt), load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False}, map_location="cpu")
    assert alg.adam is not None and alg.adam.t == t_before and torch.equal(alg.ctx.download(alg.adam.m[0], op="H-6", batch_sharded=False), m_before)


@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
def test_tt_trained_policy_exports_to_jit_and_onnx_with_matching_outputs(tt_mesh, name, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    tt_runner, tt_env, _ = build_tt_runner(name, tmp_path / "tt")
    tt_runner.learn(num_learning_iterations=2)
    export_dir = str(tmp_path / "export")
    tt_runner.export_policy_to_onnx(export_dir)
    tt_runner.export_policy_to_jit(export_dir)
    obs = tt_env.get_observations()
    policy = tt_runner.alg.get_policy()
    x = torch.cat([obs[g] for g in policy.obs_groups], dim=-1)
    with torch.no_grad():
        want = policy(obs)  # the mirror's deterministic forward == the exported function
        jit_out = torch.jit.load(f"{export_dir}/policy.pt")(x)
    assert torch.allclose(jit_out, want, rtol=1e-6, atol=1e-7), "TorchScript export of the TT-trained policy differs from the mirror"
    sess = ort.InferenceSession(f"{export_dir}/policy.onnx", providers=["CPUExecutionProvider"])
    name_in = sess.get_inputs()[0].name
    assert list(sess.get_inputs()[0].shape)[0] == 1, "upstream exports ONNX with a fixed batch of 1 (traced with get_dummy_inputs)"
    rows = 16
    onnx_out = np.concatenate([sess.run(None, {name_in: x[i : i + 1].numpy()})[0] for i in range(rows)])
    np.testing.assert_allclose(onnx_out, want[:rows].numpy(), rtol=1e-5, atol=1e-6)
    # and the DEVICE forward of the TT algorithm agrees with the exported model within the composite tolerance
    alg = tt_runner.alg
    dev_out = alg.ctx.download(alg.tt_actor.forward(alg.tt_actor.obs_to_device(obs)), op="H-6")[:rows, : tt_env.num_actions]
    assert compare(dev_out, torch.from_numpy(onnx_out), TOLERANCES["mlp_forward"]).passed
