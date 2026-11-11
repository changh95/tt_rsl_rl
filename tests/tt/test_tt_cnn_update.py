# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.6 device tests, part 2 (single-chip semantics: whole-batch references): `CNNModel` (an image group through convolutions into the MLP head, plus a 1D group) on the device against upstream: the
model's forward pass (the head's first layer is given a row map for the encoder's latent layout), the layout round trip of the weights, and a whole `TTPPO.update()` against `PPO.update()` on identical
data - the losses and the actor / critic in function space (the convolution kernels included).

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8cu -- python -m pytest -m tt tests/tt/test_tt_cnn_update.py -s -q -p no:cacheprovider
"""

import json
import torch
from pathlib import Path

import pytest

from rsl_rl_tt.testing.parity import TOLERANCES, UPDATE_TOLERANCES, compare
from rsl_rl_tt.testing.reference import RefSpec, test_mesh_shape
from tests.tt.test_tt_rollout import build_tt_runner, small_spec
from tests.tt.test_tt_update import SMOOTH, VARIANTS, add_grad_noise, clone_upstream_alg, make_pair, resync_tt_from_ref, rollout_into_both

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
MEASURED: dict = {}
STEPS = 8


@pytest.fixture(scope="module", autouse=True)
def _dump():
    yield
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase8_6_cnn_update.json").write_text(json.dumps(MEASURED, indent=2, default=str))


def rec(key, value):
    MEASURED[key] = value
    print(f"[measure] {key}: {value}")
    return value


def cnn_spec(over=None, hidden=(32,)):
    base = small_spec("cartpole_img")
    n_chips = test_mesh_shape()[0] * test_mesh_shape()[1]
    return RefSpec(**{**base.__dict__, "num_envs": 128 * n_chips, "hidden": hidden, "algorithm": {**base.algorithm, **(over or {})}})


def test_the_weights_survive_the_device_layout_exactly(tt_mesh, tmp_path):
    """The head's first layer has the encoder's latent layout (position-major, channel padded) and the kernels their im2col layout: mirror -> device -> mirror is the identity, bit for bit."""
    runner, env, _ = build_tt_runner("cartpole_img", tmp_path, spec=cnn_spec())
    alg = runner.alg
    before = {k: v.clone() for k, v in alg._raw_actor.state_dict().items()}
    alg.tt_actor.sync_to_torch()
    for k, v in alg._raw_actor.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} changed in a round trip through the device layout"
    assert alg.tt_actor.cnn is not None and alg.tt_critic.cnn is not None and alg.tt_actor.obs_groups == ["policy", "image"]


@pytest.mark.parametrize("model", ["actor", "critic"])
def test_forward_matches_the_upstream_model(tt_mesh, model, tmp_path):
    runner, env, _ = build_tt_runner("cartpole_img", tmp_path, spec=cnn_spec())
    alg = runner.alg
    ctx = alg.ctx
    tt_model, ref = (alg.tt_actor, alg._raw_actor) if model == "actor" else (alg.tt_critic, alg._raw_critic)
    torch.manual_seed(2)
    for _ in range(3):  # a few steps so that the images vary
        obs, _, _, _ = env.step(torch.randn(env.num_envs, env.num_actions))
    out = ctx.download(tt_model.forward(tt_model.obs_to_device(obs)), op="H-6")
    want = ref(obs)
    rep = compare(out[:, : want.shape[1]], want.detach(), "mlp_forward", f"{model} forward")
    rec(f"cnn model forward {model}", {"rel_fro": rep.rel_fro})
    assert rep.passed, str(rep)


def state_of(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def outputs(actor_state, critic_state, ref_alg, obs):
    with torch.no_grad():
        return {"actor": torch.func.functional_call(ref_alg._raw_actor, actor_state, (obs,)), "critic": torch.func.functional_call(ref_alg._raw_critic, critic_state, (obs,))}


def run_pair(tmp_path, over, warm=2, noise=4e-3):
    runner, env, cfg, ref_alg = make_pair("cartpole_img", tmp_path, spec=cnn_spec(over))
    alg = runner.alg
    alg.train_mode()
    ref_alg.train_mode()
    obs = env.get_observations()
    for _ in range(warm):
        obs = rollout_into_both(alg, ref_alg, env, obs, STEPS)
        torch.manual_seed(1000)
        ref_alg.update()
        torch.manual_seed(1000)
        alg.update()
        resync_tt_from_ref(alg, ref_alg)
    obs = rollout_into_both(alg, ref_alg, env, obs, STEPS)
    resync_tt_from_ref(alg, ref_alg)
    noisy = clone_upstream_alg(ref_alg)
    add_grad_noise(noisy, noise)
    obs_eval = ref_alg.storage.observations.flatten(0, 1).clone()
    init = (state_of(ref_alg._raw_actor), state_of(ref_alg._raw_critic))
    torch.manual_seed(2000)
    ref_loss = ref_alg.update()
    torch.manual_seed(2000)
    noisy_loss = noisy.update()
    torch.manual_seed(2000)
    tt_loss = alg.update()
    alg.sync_to_torch()
    f = lambda a, c: outputs(a, c, ref_alg, obs_eval)
    f_ref, f_tt, f_noisy, f_0 = f(state_of(ref_alg._raw_actor), state_of(ref_alg._raw_critic)), f(state_of(alg._raw_actor), state_of(alg._raw_critic)), f(state_of(noisy._raw_actor), state_of(noisy._raw_critic)), f(*init)
    dev = lambda a: {tag: float((a[tag] - f_ref[tag]).double().norm() / (f_ref[tag] - f_0[tag]).double().norm().clamp(min=1e-30)) for tag in ("actor", "critic")}
    return {"ref_loss": ref_loss, "tt_loss": tt_loss, "tt": dev(f_tt), "noisy": dev(f_noisy), "alg": alg, "lr": (alg.learning_rate, ref_alg.learning_rate)}


CNN_VARIANTS = {"one-step-smooth": dict(num_learning_epochs=1, num_mini_batches=1, schedule="fixed", **SMOOTH), "full-clipped": VARIANTS["full-clipped"]}


@pytest.mark.parametrize("variant", list(CNN_VARIANTS))
def test_update_matches_upstream_from_identical_state(tt_mesh, variant, tmp_path):
    r = run_pair(tmp_path, CNN_VARIANTS[variant])
    tt, ref = r["tt_loss"], r["ref_loss"]
    for key, tol in (("value", "update_value_loss"), ("entropy", "update_entropy"), ("surrogate", "update_surrogate")):
        rep = compare(torch.tensor([tt[key]]), torch.tensor([ref[key]]), TOLERANCES[tol], f"{variant} loss.{key}")
        rec(f"{variant} loss.{key}", {"tt": tt[key], "ref": ref[key], "rel_fro": rep.rel_fro})
        assert rep.passed, str(rep)
    assert r["lr"][0] == pytest.approx(r["lr"][1], rel=1e-6)
    for tag in ("actor", "critic"):
        got, noise = r["tt"][tag], r["noisy"][tag]
        limit = UPDATE_TOLERANCES["full_update_clipped"] if variant == "full-clipped" else UPDATE_TOLERANCES["noise_factor"] * noise + UPDATE_TOLERANCES["noise_abs"]
        rec(f"{variant} function_deviation.{tag}", {"tt": got, "noisy_ref": noise})
        assert got <= limit, f"{variant} {tag}: {got:.3e} of the update differs (limit {limit:.3e}; noise baseline {noise:.3e})"
