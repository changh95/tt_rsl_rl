# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.2 device tests: checkpoint compatibility with RND in both directions through the UNMODIFIED upstream OnPolicyRunner: `rnd_state_dict` (predictor, target, normalizers),
`rnd_optimizer_state_dict` and `rnd_update_counter`.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8rc -- python -m pytest -m tt tests/tt/test_tt_rnd_checkpoint.py -s -q -p no:cacheprovider
"""

import torch

import pytest

from rsl_rl_tt.testing.parity import TOLERANCES, compare
from tests.tt.test_tt_checkpoint import build_upstream_runner, shape_tree
from tests.tt.test_tt_rnd import rnd_spec
from tests.tt.test_tt_rollout import build_tt_runner
from tests.tt.test_tt_update import ref_rnd_params, tt_rnd_params

pytestmark = pytest.mark.tt

ITERATIONS = 3


@pytest.mark.parametrize("case", ["cartpole-actor", "tracking12-own"])
def test_tt_checkpoint_with_rnd_loads_into_upstream(tt_mesh, case, tmp_path):
    name, spec = rnd_spec(case)
    tt_runner, tt_env, _ = build_tt_runner(name, tmp_path / "tt", spec=spec)
    tt_runner.learn(num_learning_iterations=ITERATIONS)
    ckpt = tmp_path / "model_tt.pt"
    tt_runner.save(str(ckpt))
    saved = torch.load(ckpt, weights_only=False)
    alg = tt_runner.alg
    steps = ITERATIONS * spec.num_steps
    assert saved["rnd_update_counter"] == steps == alg.rnd.update_counter

    up_runner, _ = build_upstream_runner(name, tmp_path / "up", spec=spec)
    up_runner.load(str(ckpt), map_location="cpu")
    up = up_runner.alg.rnd
    assert up.update_counter == steps
    # the loaded upstream module equals the mirror bit for bit, and the mirror equals the device
    for k, v in alg.rnd.state_dict().items():
        assert torch.equal(v, up.state_dict()[k]), f"rnd_state_dict {k}"
    dev, mirror = tt_rnd_params(alg), ref_rnd_params(alg)
    for k in mirror:
        assert torch.equal(dev[k], mirror[k]), f"device vs mirror: {k}"
    # the target network is frozen: it is still the initial network
    ref_runner, _ = build_upstream_runner(name, tmp_path / "ref", spec=spec)
    for k, v in ref_runner.alg.rnd.state_dict().items():
        if k.startswith("target."):
            assert torch.equal(v, up.state_dict()[k]), f"the target network changed: {k}"
    # structure: exactly the keys / shapes / dtypes of an upstream checkpoint of the same configuration (after the same number of iterations)
    ref_runner.learn(num_learning_iterations=ITERATIONS)
    ref_ckpt = tmp_path / "model_up.pt"
    ref_runner.save(str(ref_ckpt))
    ref_saved = torch.load(ref_ckpt, weights_only=False)
    assert set(saved) == set(ref_saved)
    for key in ("rnd_state_dict", "rnd_optimizer_state_dict", "actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        assert shape_tree(saved[key]) == shape_tree(ref_saved[key]), f"{key} differs structurally from an upstream checkpoint"
    assert all(torch.isfinite(s["exp_avg"]).all() and (s["exp_avg_sq"] >= 0).all() for s in saved["rnd_optimizer_state_dict"]["state"].values())
    assert int(next(iter(saved["rnd_optimizer_state_dict"]["state"].values()))["step"]) == alg.rnd_tt.adam.t == ITERATIONS * alg.num_learning_epochs * alg.num_mini_batches
    # the loaded upstream RND computes the device's embedding distance
    obs = tt_env.get_observations()
    with torch.no_grad():
        x = up.state_normalizer(up.get_rnd_state(obs))
        want = torch.linalg.norm(up.target(x) - up.predictor(x), dim=1)
    r = alg.rnd_tt
    got = alg.ctx.download(r.embedding_distance(r.obs_to_device(obs)), op="H-9")[:, 0]
    rep = compare(got, want, TOLERANCES["rnd_intrinsic_reward"], f"{case} embedding distance of the loaded RND")
    assert rep.passed, str(rep)


@pytest.mark.parametrize("case", ["cartpole-actor", "tracking12-own"])
def test_upstream_checkpoint_with_rnd_resumes_on_tt_with_identical_device_state(tt_mesh, case, tmp_path):
    name, spec = rnd_spec(case)
    up_runner, _ = build_upstream_runner(name, tmp_path / "up", spec=spec)
    up_runner.learn(num_learning_iterations=ITERATIONS)
    ckpt = tmp_path / "model_up.pt"
    up_runner.save(str(ckpt))
    saved = torch.load(ckpt, weights_only=False)

    tt_runner, tt_env, _ = build_tt_runner(name, tmp_path / "tt", spec=spec, seed=7)  # a DIFFERENT initialisation: everything must come from the checkpoint
    tt_runner.load(str(ckpt), map_location="cpu")
    alg = tt_runner.alg
    ctx = alg.ctx
    r = alg.rnd_tt
    sd = saved["rnd_state_dict"]
    # networks: the predictor and the (frozen, random) target, bit for bit
    for tag, net in (("predictor", r.predictor), ("target", r.target)):
        lin_names = sorted({k.rsplit(".", 1)[0] for k in sd if k.startswith(f"{tag}.") and k.endswith("weight")}, key=lambda s: int(s.split(".")[1]))
        for layer, base in zip(net.layers, lin_names):
            w = layer.weight_from_host(ctx.download(layer.w, op="H-6", batch_sharded=False))
            b = layer.bias_from_host(ctx.download(layer.b, op="H-6", batch_sharded=False))
            assert torch.equal(w, sd[f"{base}.weight"]) and torch.equal(b, sd[f"{base}.bias"]), f"{base}"
    # normalizers and the counter
    stats = r.state_norm.host_stats()
    assert torch.equal(stats["mean"], sd["state_normalizer._mean"].reshape(-1)) and torch.equal(stats["var"], sd["state_normalizer._var"].reshape(-1)) and stats["count"] == int(sd["state_normalizer.count"])
    host = lambda t: float(ctx.download(t, op="H-6", batch_sharded=False)[0, 0])
    assert host(r.reward_norm.mean) == float(sd["reward_normalizer.emp_norm._mean"]) and host(r.reward_norm.var) == float(sd["reward_normalizer.emp_norm._var"])
    assert r.reward_norm.count == int(sd["reward_normalizer.emp_norm.count"])
    assert alg.rnd.update_counter == saved["rnd_update_counter"]
    # Adam moments of the predictor and its step counter
    state = saved["rnd_optimizer_state_dict"]["state"]
    assert r.adam.t == int(state[0]["step"])
    for slot_p, idx, kind, layer in r._param_slots():
        opt_idx = next(i for i, q in enumerate(alg.rnd.optimizer.param_groups[0]["params"]) if q is slot_p)
        for store, key in ((r.adam.m, "exp_avg"), (r.adam.v, "exp_avg_sq")):
            h = ctx.download(store[idx], op="H-6", batch_sharded=False)
            got = layer.weight_from_host(h) if kind == "weight" else layer.bias_from_host(h)
            assert torch.equal(got, state[opt_idx][key]), f"{key} of the predictor's optimizer slot {opt_idx} ({kind})"
    assert host(r.lr) == pytest.approx(saved["rnd_optimizer_state_dict"]["param_groups"][0]["lr"])
    before = {k: v.clone() for k, v in tt_rnd_params(alg).items()}
    tt_runner.learn(num_learning_iterations=2)
    after = tt_rnd_params(alg)
    assert any(not torch.equal(before[k], after[k]) for k in before), "training after load did not change the predictor"
    assert all(torch.isfinite(v).all() for v in after.values()) and alg.rnd.update_counter == saved["rnd_update_counter"] + 2 * spec.num_steps
