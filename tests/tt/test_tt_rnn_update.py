# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.5 device tests, part 2 (single-chip semantics: whole-batch references): a whole `TTPPO.update()` with recurrent models (GRU / LSTM, 1 and 2 layers, as the actor, the critic or both) against
upstream `PPO.update()` with its recurrent mini-batch generator (trajectories split at the dones and padded, masks) on IDENTICAL data and hidden states: the losses and the actor / critic in function space (the
policy and the value of the whole rollout, computed by upstream's batch-mode forward of the parameters in question).

    scripts/tt_run.sh --chips 16 --timeout 1200 --tag p8nu -- python -m pytest -m tt tests/tt/test_tt_rnn_update.py -s -q -p no:cacheprovider
"""

import copy
import json
import torch
from pathlib import Path

import pytest

from rsl_rl_tt.testing.parity import TOLERANCES, UPDATE_TOLERANCES, compare
from rsl_rl_tt.testing.reference import RefSpec, test_mesh_shape, with_rnn
from tests.tt.test_tt_rollout import small_spec
from tests.tt.test_tt_update import SMOOTH, VARIANTS, add_grad_noise, clone_upstream_alg, make_pair, resync_tt_from_ref

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
MEASURED: dict = {}
STEPS = 8


@pytest.fixture(scope="module", autouse=True)
def _dump():
    yield
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase8_5_rnn_update.json").write_text(json.dumps(MEASURED, indent=2, default=str))


def rec(key, value):
    MEASURED[key] = value
    print(f"[measure] {key}: {value}")
    return value


def rnn_spec(rnn_type="gru", layers=1, actor=True, critic=True, hidden=32, over=None):
    base = small_spec("cartpole_po")
    n_chips = test_mesh_shape()[0] * test_mesh_shape()[1]
    algorithm = {**base.algorithm, **(over or {})}
    # recurrent mini-batches are slices of the environments of a chip: with 4 mini-batches every chip needs 128 environments (32 per mini-batch)
    return with_rnn(RefSpec(**{**base.__dict__, "algorithm": algorithm, "num_envs": 128 * n_chips}), rnn_type, hidden, layers, actor, critic)


def host_hidden(ctx, model):
    """The device hidden state of a recurrent model in torch's layout: [layers, N, H] (GRU) or a tuple of two (LSTM); None for an MLP."""
    if not model.recurrent:
        return None
    n = 2 if model.rnn.lstm else 1
    comps = [torch.stack([ctx.download(model.state[layer * n + k], op="H-6")[:, : model.rnn.hidden] for layer in range(model.rnn.num_layers)]) for k in range(n)]
    return tuple(comps) if model.rnn.lstm else comps[0]


def rollout_both(alg, ref_alg, env, obs):
    """One rollout of the TT algorithm mirrored into the upstream storage: the transitions AND the hidden states (read from the device before every step)."""
    ctx = alg.ctx
    A = env.num_actions
    dl = lambda t: ctx.download(t, op="H-6")
    ref_alg.storage.clear()
    with torch.inference_mode():
        for _ in range(STEPS):
            hidden = (host_hidden(ctx, alg.tt_actor), host_hidden(ctx, alg.tt_critic))
            actions = alg.act(obs)
            pend = alg._pending
            tr = ref_alg.transition
            tr.observations = obs
            tr.actions = actions.clone()
            tr.values = dl(pend["value"])[:, :1].clone()
            tr.actions_log_prob = dl(pend["logp"])[:, 0].clone()
            mean = dl(pend["mean"])[:, :A].clone()
            tr.distribution_params = (mean, alg._std_host.reshape(1, A).expand(env.num_envs, A).clone())
            tr.hidden_states = hidden
            obs, rewards, dones, extras = env.step(actions)
            tr.rewards = rewards.clone() + ref_alg.gamma * (tr.values * extras["time_outs"].unsqueeze(1).float()).squeeze(1)
            tr.dones = dones
            ref_alg.storage.add_transition(tr)
            tr.clear()
            alg.process_env_step(obs, rewards, dones, extras)
        alg.compute_returns(obs)
    # the device storage holds a chip's rows (its environments of step 0, 1, ...) one chip after the other: back to [steps, environments]
    n = ctx.n_chips
    unchip = lambda t: dl(t).reshape(n, STEPS, env.num_envs // n, 1).permute(1, 0, 2, 3).reshape(STEPS, env.num_envs, 1)
    ref_alg.storage.returns[:] = unchip(alg.storage.returns)
    ref_alg.storage.advantages = unchip(alg.storage.advantages)
    return obs


def policy_and_value(actor_state, critic_state, ref_alg, batch):
    """Upstream's batch-mode forward of the given parameters on the whole rollout: the actor's mean and the critic's value for every step of every environment."""
    out = {}
    for tag, model, state, hidden in (("actor", ref_alg._raw_actor, actor_state, batch.hidden_states[0]), ("critic", ref_alg._raw_critic, critic_state, batch.hidden_states[1])):
        with torch.no_grad():
            out[tag] = torch.func.functional_call(model, state, (batch.observations,), {"masks": batch.masks, "hidden_state": hidden})
    return out


def state_of(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def run_pair(tmp_path, spec, algorithm_over, warm=2, noise=4e-3):
    runner, env, cfg, ref_alg = make_pair("cartpole_po", tmp_path, spec=RefSpec(**{**spec.__dict__, "algorithm": {**spec.algorithm, **algorithm_over}}))
    alg = runner.alg
    assert alg.recurrent
    alg.train_mode()
    ref_alg.train_mode()
    obs = env.get_observations()
    for _ in range(warm):  # normalizers, Adam's moments, the learning rate and the carried hidden state are no longer at their initial values
        obs = rollout_both(alg, ref_alg, env, obs)
        torch.manual_seed(1000)
        ref_alg.update()
        torch.manual_seed(1000)
        alg.update()
        resync_tt_from_ref(alg, ref_alg)
    obs = rollout_both(alg, ref_alg, env, obs)
    resync_tt_from_ref(alg, ref_alg)
    noisy = clone_upstream_alg(ref_alg)
    add_grad_noise(noisy, noise)
    batch = next(ref_alg.storage.recurrent_mini_batch_generator(1, 1))  # the whole rollout as one batch (all environments, padded trajectories, masks, saved hidden states)
    init = (state_of(ref_alg._raw_actor), state_of(ref_alg._raw_critic))
    torch.manual_seed(2000)
    ref_loss = ref_alg.update()
    torch.manual_seed(2000)
    noisy_loss = noisy.update()
    torch.manual_seed(2000)
    tt_loss = alg.update()
    alg.sync_to_torch()
    f = lambda a, c: policy_and_value(a, c, ref_alg, batch)
    f_ref, f_tt, f_noisy, f_0 = f(state_of(ref_alg._raw_actor), state_of(ref_alg._raw_critic)), f(state_of(alg._raw_actor), state_of(alg._raw_critic)), f(state_of(noisy._raw_actor), state_of(noisy._raw_critic)), f(*init)
    dev = lambda a: {tag: float((a[tag] - f_ref[tag]).double().norm() / (f_ref[tag] - f_0[tag]).double().norm().clamp(min=1e-30)) for tag in ("actor", "critic")}
    return {"ref_loss": ref_loss, "tt_loss": tt_loss, "noisy_loss": noisy_loss, "tt": dev(f_tt), "noisy": dev(f_noisy), "alg": alg, "ref_alg": ref_alg,
            "lr": (alg.learning_rate, ref_alg.learning_rate)}


CASES = {
    "gru": dict(rnn_type="gru", layers=1),
    "lstm": dict(rnn_type="lstm", layers=1),
    "gru-2-layers": dict(rnn_type="gru", layers=2),
    "lstm-2-layers": dict(rnn_type="lstm", layers=2),
    "gru-actor-only": dict(rnn_type="gru", layers=1, critic=False),
    "lstm-critic-only": dict(rnn_type="lstm", layers=1, actor=False),
}
RNN_VARIANTS = {"one-step-smooth": dict(num_learning_epochs=1, num_mini_batches=1, schedule="fixed", **SMOOTH), "full-clipped": VARIANTS["full-clipped"]}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("variant", list(RNN_VARIANTS))
def test_recurrent_update_matches_upstream_from_identical_state(tt_mesh, case, variant, tmp_path):
    over = RNN_VARIANTS[variant]
    r = run_pair(tmp_path, rnn_spec(**CASES[case], over=over), {})
    tt, ref = r["tt_loss"], r["ref_loss"]
    for key, tol in (("value", "update_value_loss"), ("entropy", "update_entropy"), ("surrogate", "update_surrogate")):
        rep = compare(torch.tensor([tt[key]]), torch.tensor([ref[key]]), TOLERANCES[tol], f"{case}/{variant} loss.{key}")
        rec(f"{case}/{variant} loss.{key}", {"tt": tt[key], "ref": ref[key], "rel_fro": rep.rel_fro})
        assert rep.passed, str(rep)
    assert r["lr"][0] == pytest.approx(r["lr"][1], rel=1e-6)
    for tag in ("actor", "critic"):
        got, noise = r["tt"][tag], r["noisy"][tag]
        limit = UPDATE_TOLERANCES["full_update_clipped"] if variant == "full-clipped" else UPDATE_TOLERANCES["rnn_noise_factor"] * noise + UPDATE_TOLERANCES["noise_abs"]
        rec(f"{case}/{variant} function_deviation.{tag}", {"tt": got, "noisy_ref": noise})
        assert got <= limit, f"{case}/{variant} {tag}: {got:.3e} of the update differs (limit {limit:.3e}; noise baseline {noise:.3e})"
