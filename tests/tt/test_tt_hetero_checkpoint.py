# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.1 device tests: checkpoint compatibility in both directions and policy export with a heteroscedastic actor (the two-head output layer and its Adam moments),
through the UNMODIFIED upstream OnPolicyRunner. The v1 versions are in `test_tt_checkpoint.py`.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8c -- python -m pytest -m tt tests/tt/test_tt_hetero_checkpoint.py -s -q -p no:cacheprovider
"""

import copy
import numpy as np
import torch

import pytest

from rsl_rl_tt.testing.parity import TOLERANCES, compare
from tests.tt.test_tt_checkpoint import build_upstream_runner, shape_tree
from tests.tt.test_tt_rollout import build_tt_runner, small_spec

pytestmark = pytest.mark.tt

HETERO = "HeteroscedasticGaussianDistribution"


def spec_for(name, std_type):
    return small_spec(name, distribution=HETERO, std_type=std_type, init_std=0.8)


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_tt_checkpoint_loads_into_upstream_with_identical_inference_and_structure(tt_mesh, std_type, tmp_path):
    name = "tracking12"
    spec = spec_for(name, std_type)
    tt_runner, tt_env, _ = build_tt_runner(name, tmp_path / "tt", spec=spec)
    tt_runner.learn(num_learning_iterations=3)
    ckpt = tmp_path / "model_tt.pt"
    tt_runner.save(str(ckpt), infos={"note": "from the TT runner"})

    up_runner, _ = build_upstream_runner(name, tmp_path / "up", spec=spec)
    assert up_runner.load(str(ckpt), map_location="cpu") == {"note": "from the TT runner"}
    obs = tt_env.get_observations()
    with torch.no_grad():
        out_up, out_mirror = up_runner.get_inference_policy()(obs), tt_runner.get_inference_policy()(obs)
    assert torch.equal(out_up, out_mirror), "the loaded upstream policy must equal the TT mirror bit for bit"
    alg = tt_runner.alg
    dev = alg.ctx.download(alg.tt_actor.forward(alg.tt_actor.obs_to_device(obs)), op="H-6")
    a_p, a = alg.tt_actor.distribution.dim_p, tt_env.num_actions
    assert compare(dev[:, :a], out_up, TOLERANCES["mlp_forward"]).passed  # the mean head
    # the std head: upstream computes the std of the loaded policy from the loaded weights
    with torch.no_grad():
        up_actor = up_runner.alg._raw_actor
        up_actor(obs, stochastic_output=True)
        up_std = up_actor.output_std
    dev_std = dev[:, a_p : a_p + a]
    dev_std = dev_std.clamp(*spec_range(up_actor)) if std_type == "scalar" else dev_std.clamp(*up_actor.distribution.log_std_range).exp()
    assert compare(dev_std, up_std, TOLERANCES["mlp_forward"]).passed

    # structure: exactly the keys / shapes / dtypes of an upstream checkpoint of the same configuration
    ref_runner, _ = build_upstream_runner(name, tmp_path / "ref", spec=spec)
    ref_runner.learn(num_learning_iterations=3)
    ref_ckpt = tmp_path / "model_up.pt"
    ref_runner.save(str(ref_ckpt), infos={"note": "from the TT runner"})
    a_ck, b_ck = torch.load(ckpt, weights_only=False), torch.load(ref_ckpt, weights_only=False)
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        assert shape_tree(a_ck[key]) == shape_tree(b_ck[key]), f"{key} differs structurally from an upstream checkpoint"
    assert set(a_ck) == set(b_ck)
    assert not any("std_param" in k for k in a_ck["actor_state_dict"]), "a heteroscedastic actor has no std parameter"
    assert all(torch.isfinite(s["exp_avg"]).all() and (s["exp_avg_sq"] >= 0).all() for s in a_ck["optimizer_state_dict"]["state"].values())


def spec_range(actor):
    return tuple(actor.distribution.std_range)


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_upstream_checkpoint_resumes_on_tt_with_identical_device_state(tt_mesh, std_type, tmp_path):
    name = "tracking12"
    spec = spec_for(name, std_type)
    up_runner, up_env = build_upstream_runner(name, tmp_path / "up", spec=spec)
    up_runner.learn(num_learning_iterations=3)
    ckpt = tmp_path / "model_up.pt"
    up_runner.save(str(ckpt))
    saved = torch.load(ckpt, weights_only=False)

    tt_runner, tt_env, _ = build_tt_runner(name, tmp_path / "tt", spec=spec, seed=7)  # a DIFFERENT initialisation: everything must come from the checkpoint
    tt_runner.load(str(ckpt), map_location="cpu")
    alg = tt_runner.alg
    ctx = alg.ctx
    for tag, model, sd in (("actor", alg.tt_actor, saved["actor_state_dict"]), ("critic", alg.tt_critic, saved["critic_state_dict"])):
        lin_names = sorted({k.rsplit(".", 1)[0] for k in sd if k.startswith("mlp.") and k.endswith("weight")}, key=lambda s: int(s.split(".")[1]))
        for layer, base in zip(model.mlp.layers, lin_names):
            w = layer.weight_from_host(ctx.download(layer.w, op="H-6", batch_sharded=False))
            b = layer.bias_from_host(ctx.download(layer.b, op="H-6", batch_sharded=False))
            assert torch.equal(w, sd[f"{base}.weight"]) and torch.equal(b, sd[f"{base}.bias"]), f"{tag} {base}"
    # Adam moments of every parameter, incl. both heads of the output layer, came across exactly (this goes through the layer's own two-head layout)
    assert alg.adam.t == int(next(iter(saved["optimizer_state_dict"]["state"].values()))["step"])
    state = saved["optimizer_state_dict"]["state"]
    for idx_in_opt, p in enumerate(alg.optimizer.param_groups[0]["params"]):
        for slot_p, dev_idx, kind, owner in alg._param_slots():
            if slot_p is p:
                for store, key in ((alg.adam.m, "exp_avg"), (alg.adam.v, "exp_avg_sq")):
                    host = ctx.download(store[dev_idx], op="H-6", batch_sharded=False)
                    got = owner.weight_from_host(host) if kind == "weight" else owner.bias_from_host(host)
                    assert torch.equal(got, state[idx_in_opt][key]), f"{key} of optimizer slot {idx_in_opt} ({kind})"
    obs = tt_env.get_observations()
    with torch.no_grad():
        ref_out = up_runner.get_inference_policy()(obs)
    dev = ctx.download(alg.tt_actor.forward(alg.tt_actor.obs_to_device(obs)), op="H-6")[:, : tt_env.num_actions]
    assert compare(dev, ref_out, TOLERANCES["mlp_forward"]).passed
    before = copy.deepcopy(alg.get_policy().state_dict())
    tt_runner.learn(num_learning_iterations=2)
    after = alg.get_policy().state_dict()
    last = max(int(k.split(".")[1]) for k in before if k.startswith("mlp.") and k.endswith("weight"))
    assert not torch.equal(before[f"mlp.{last}.weight"], after[f"mlp.{last}.weight"]), "training after load did not change the two-head output layer"
    assert all(torch.isfinite(v).all() for v in after.values() if v.is_floating_point())


def test_tt_trained_policy_exports_to_jit_and_onnx_with_matching_outputs(tt_mesh, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    spec = spec_for("tracking12", "scalar")
    tt_runner, tt_env, _ = build_tt_runner("tracking12", tmp_path / "tt", spec=spec)
    tt_runner.learn(num_learning_iterations=2)
    export_dir = str(tmp_path / "export")
    tt_runner.export_policy_to_onnx(export_dir)
    tt_runner.export_policy_to_jit(export_dir)
    obs = tt_env.get_observations()
    policy = tt_runner.alg.get_policy()
    x = torch.cat([obs[g] for g in policy.obs_groups], dim=-1)
    with torch.no_grad():
        want = policy(obs)  # the deterministic output: the mean head
        jit_out = torch.jit.load(f"{export_dir}/policy.pt")(x)
    assert want.shape[-1] == tt_env.num_actions
    assert torch.allclose(jit_out, want, rtol=1e-6, atol=1e-7)
    sess = ort.InferenceSession(f"{export_dir}/policy.onnx", providers=["CPUExecutionProvider"])
    name_in = sess.get_inputs()[0].name
    rows = 16
    onnx_out = np.concatenate([sess.run(None, {name_in: x[i : i + 1].numpy()})[0] for i in range(rows)])
    np.testing.assert_allclose(onnx_out, want[:rows].numpy(), rtol=1e-5, atol=1e-6)
