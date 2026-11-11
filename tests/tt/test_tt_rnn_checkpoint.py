# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.5 device tests: checkpoint compatibility with recurrent models in both directions (the RNN weights, their Adam moments in torch's layout) and the JIT / ONNX export of a TT-trained recurrent
policy, through the UNMODIFIED upstream OnPolicyRunner.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8nc -- python -m pytest -m tt tests/tt/test_tt_rnn_checkpoint.py -s -q -p no:cacheprovider
"""

import numpy as np
import torch

import pytest

from rsl_rl_tt.testing.parity import TOLERANCES, compare
from tests.tt.test_tt_checkpoint import build_upstream_runner, shape_tree
from tests.tt.test_tt_rnn_update import rnn_spec
from tests.tt.test_tt_rollout import build_tt_runner

pytestmark = pytest.mark.tt

CASES = {"gru": dict(rnn_type="gru", layers=1), "lstm-2-layers": dict(rnn_type="lstm", layers=2), "gru-critic-only": dict(rnn_type="gru", layers=1, actor=False)}


@pytest.mark.parametrize("case", list(CASES))
def test_tt_checkpoint_loads_into_upstream_with_identical_inference_and_structure(tt_mesh, case, tmp_path):
    spec = rnn_spec(**CASES[case])
    tt_runner, tt_env, _ = build_tt_runner("cartpole_po", tmp_path / "tt", spec=spec)
    tt_runner.learn(num_learning_iterations=3)
    ckpt = tmp_path / "model_tt.pt"
    tt_runner.save(str(ckpt))
    up_runner, _ = build_upstream_runner("cartpole_po", tmp_path / "up", spec=spec)
    up_runner.load(str(ckpt), map_location="cpu")
    obs = tt_env.get_observations()
    # the loaded upstream policy and the TT mirror are the same recurrent function: three steps from a fresh hidden state agree bit for bit
    pol_up, pol_tt = up_runner.get_inference_policy(), tt_runner.get_inference_policy()
    pol_up.reset(), pol_tt.reset()
    with torch.no_grad():
        for _ in range(3):
            assert torch.equal(pol_up(obs), pol_tt(obs))
    # structure: exactly the keys / shapes / dtypes of an upstream checkpoint of the same configuration
    ref_runner, _ = build_upstream_runner("cartpole_po", tmp_path / "ref", spec=spec)
    ref_runner.learn(num_learning_iterations=3)
    ref_runner.save(str(tmp_path / "model_up.pt"))
    a_ck, b_ck = torch.load(ckpt, weights_only=False), torch.load(tmp_path / "model_up.pt", weights_only=False)
    assert set(a_ck) == set(b_ck)
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        assert shape_tree(a_ck[key]) == shape_tree(b_ck[key]), f"{key} differs structurally from an upstream checkpoint"
    assert any(k.startswith("rnn.rnn.weight_hh_l0") for k in (a_ck["actor_state_dict"] if CASES[case].get("actor", True) else a_ck["critic_state_dict"]))
    assert all(torch.isfinite(s["exp_avg"]).all() and (s["exp_avg_sq"] >= 0).all() for s in a_ck["optimizer_state_dict"]["state"].values())


@pytest.mark.parametrize("case", list(CASES))
def test_upstream_checkpoint_resumes_on_tt_with_identical_device_state(tt_mesh, case, tmp_path):
    spec = rnn_spec(**CASES[case])
    up_runner, _ = build_upstream_runner("cartpole_po", tmp_path / "up", spec=spec)
    up_runner.learn(num_learning_iterations=3)
    ckpt = tmp_path / "model_up.pt"
    up_runner.save(str(ckpt))
    saved = torch.load(ckpt, weights_only=False)

    tt_runner, tt_env, _ = build_tt_runner("cartpole_po", tmp_path / "tt", spec=spec, seed=7)  # a DIFFERENT initialisation: everything must come from the checkpoint
    tt_runner.load(str(ckpt), map_location="cpu")
    alg = tt_runner.alg
    ctx = alg.ctx
    for tag, model, sd in (("actor", alg.tt_actor, saved["actor_state_dict"]), ("critic", alg.tt_critic, saved["critic_state_dict"])):
        for layer, base in zip(model.mlp.layers, (0, 2, 4)[: len(model.mlp.layers)]):
            w = layer.weight_from_host(ctx.download(layer.w, op="H-6", batch_sharded=False))
            assert torch.equal(w, sd[f"mlp.{base}.weight"]), f"{tag} mlp {base}"
        if model.recurrent:
            for p, idx, layer, kind in model.rnn.param_slots():
                key = {"w_ih": "weight_ih", "w_hh": "weight_hh", "b_ih": "bias_ih", "b_hh": "bias_hh"}[kind]
                got = model.rnn.weight_from_host(layer, kind, ctx.download(model.rnn.parameters()[idx], op="H-6", batch_sharded=False))
                assert torch.equal(got, sd[f"rnn.rnn.{key}_l{layer}"]), f"{tag} rnn {key} layer {layer}"
    # Adam moments of every parameter (MLP, std, RNN) came across exactly, through each owner's own layout
    state = saved["optimizer_state_dict"]["state"]
    assert alg.adam.t == int(next(iter(state.values()))["step"])
    checked = 0
    for slot_p, dev_idx, kind, owner in alg._param_slots():
        opt_idx = next(i for i, q in enumerate(alg.optimizer.param_groups[0]["params"]) if q is slot_p)
        for store, key in ((alg.adam.m, "exp_avg"), (alg.adam.v, "exp_avg_sq")):
            host = ctx.download(store[dev_idx], op="H-6", batch_sharded=False)
            if kind == "weight":
                got = owner.weight_from_host(host)
            elif kind == "bias":
                got = owner.bias_from_host(host)
            elif kind == "rnn":
                got = owner[0].weight_from_host(owner[1], owner[2], host)
            else:
                got = host[0, : owner.dim]
            assert torch.equal(got, state[opt_idx][key]), f"{key} of optimizer slot {opt_idx} ({kind})"
        checked += 1
    assert checked == len(alg.optimizer.param_groups[0]["params"])
    before = {k: v.clone() for k, v in alg._raw_actor.state_dict().items()}
    tt_runner.learn(num_learning_iterations=2)
    after = alg.get_policy().state_dict()
    assert all(torch.isfinite(v).all() for v in after.values() if v.is_floating_point())
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith(("mlp", "rnn")) or True)


def test_tt_trained_recurrent_policy_exports_to_jit_and_onnx_with_matching_outputs(tt_mesh, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    spec = rnn_spec("gru", 1)
    tt_runner, tt_env, _ = build_tt_runner("cartpole_po", tmp_path / "tt", spec=spec)
    tt_runner.learn(num_learning_iterations=2)
    export_dir = str(tmp_path / "export")
    tt_runner.export_policy_to_onnx(export_dir)
    tt_runner.export_policy_to_jit(export_dir)
    obs = tt_env.get_observations()[:1]  # the exported model keeps the hidden state of ONE environment
    policy = tt_runner.alg.get_policy()
    policy.reset()
    x = torch.cat([obs[g] for g in policy.obs_groups], dim=-1)
    with torch.no_grad():
        want = [policy(obs) for _ in range(3)]  # three recurrent steps of the mirror
    jit = torch.jit.load(f"{export_dir}/policy.pt")
    jit.reset()
    with torch.no_grad():
        got = [jit(x) for _ in range(3)]
    for g, w in zip(got, want):
        assert torch.allclose(g, w, rtol=1e-6, atol=1e-7)
    sess = ort.InferenceSession(f"{export_dir}/policy.onnx", providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    h = np.zeros((1, 1, spec.model["rnn_hidden_dim"]), dtype=np.float32)
    for w in want:
        out = sess.run(None, {names[0]: x.numpy(), names[1]: h})
        np.testing.assert_allclose(out[0], w.numpy(), rtol=1e-5, atol=1e-6)
        h = out[1]
