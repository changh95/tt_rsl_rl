# SPDX-License-Identifier: BSD-3-Clause
"""Phase 5 device tests: the TRACED path (act step and mini-batch step captured once and replayed) must reproduce the eager path BIT FOR BIT
(same kernels, same inputs) and keep working across allocations, normalizer updates and checkpoint loads.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p5t -- python -m pytest -m tt tests/tt/test_tt_trace.py -s -q -p no:cacheprovider

One process = one mesh, opened WITH a trace region by the fixture below (`tt_mesh` of conftest.py opens none), so run this file on its own.

The persistent-state rule these tests guard (TTNN_ISSUES T-15): the intermediates of a captured trace are freed when the capture ends and are rewritten by
every replay, so anything that must keep its value across replays has to be allocated BEFORE the first capture; every runner of a test is therefore
constructed before any runner captures a trace (`TTPPO` refuses to be constructed afterwards).
"""

import torch

import pytest

from rsl_rl.runners import OnPolicyRunner
from rsl_rl_tt.testing.envs import make_env
from rsl_rl_tt.testing.reference import make_train_cfg
from tests.tt.test_tt_rollout import small_spec
from tests.tt.test_tt_update import tt_params

pytestmark = pytest.mark.tt

STEP_FIELDS = ("actor_obs", "critic_obs", "action", "mean", "value", "logp", "reward", "done")


@pytest.fixture(scope="module")
def traced_mesh():
    from rsl_rl_tt.device import open_tray_mesh

    tm = open_tray_mesh((1, 1), trace_region_size=128 * 1024 * 1024)
    yield tm
    tm.close()


def build(name, tmp_path, trace, seed=0, spec=None):
    spec = spec or small_spec(name)
    torch.manual_seed(seed)
    env = make_env(name, spec.num_envs, seed=seed)
    cfg = make_train_cfg(name, spec)
    cfg["algorithm"]["class_name"] = "rsl_rl_tt.ppo:TTPPO"
    cfg["algorithm"]["tt_cfg"] = {"mesh_shape": [1, 1], "trace": trace}
    return OnPolicyRunner(env, cfg, log_dir=str(tmp_path), device="cpu"), env


def release(*runners):
    for runner in runners:
        runner.alg.release_traces()


def host_state(alg):
    """Everything the device holds for training as host tensors: parameters, Adam moments, normalizer statistics, learning rate."""
    ctx = alg.ctx
    down = lambda t: ctx.download(t, op="H-6", batch_sharded=False)
    out = {f"param.{k}": v for k, v in tt_params(alg).items()}
    for k, (m, v) in enumerate(zip(alg.adam.m, alg.adam.v)):
        out[f"adam.m.{k}"], out[f"adam.v.{k}"] = down(m), down(v)
    out["adam.t"] = torch.tensor(float(alg.adam.t))
    for tag in ("tt_actor", "tt_critic"):
        stats = getattr(alg, tag).normalizer.host_stats()
        out[f"{tag}.mean"], out[f"{tag}.var"], out[f"{tag}.count"] = stats["mean"], stats["var"], torch.tensor(float(stats["count"]))
    out["lr"] = torch.tensor(alg.learning_rate)
    return out


def assert_state_equal(sa, sb, what):
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"{what}: {k} differs (max abs diff {float((sa[k].double() - sb[k].double()).abs().max()):.3e})"


@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
def test_traced_training_is_bit_identical_to_eager(traced_mesh, name, tmp_path):
    """Two runners from the same seed, one eager and one traced, run 4 iterations (rollouts, shuffles, updates, normalizer updates): every logged
    scalar and every piece of device state must be EQUAL, not merely close."""
    eager, _ = build(name, tmp_path / "eager", trace=False)
    traced, _ = build(name, tmp_path / "traced", trace=True)
    try:
        # the host draws the exploration noise and the shuffle permutations (allow-list H-4) from the GLOBAL torch RNG: both runs must see the same stream
        torch.manual_seed(4242)
        eager.learn(num_learning_iterations=4)
        torch.manual_seed(4242)
        traced.learn(num_learning_iterations=4)
        assert set(traced.alg._traces) >= {"act", "gae", "update.step.0", "update.step.3"}, "the traces were not captured"
        # logged scalars first: a mismatch then names the first iteration that differs
        ea, ta = (tmp_path / "eager" / "metrics.csv").read_text().splitlines(), (tmp_path / "traced" / "metrics.csv").read_text().splitlines()
        cols = ea[0].split(",")
        assert cols == ta[0].split(",")
        checked = 0
        for col in ("Loss/value", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate", "Policy/mean_std", "Train/mean_reward"):
            if col not in cols:  # e.g. no episode finished yet -> no episode statistics
                continue
            k = cols.index(col)
            va, vb = [r.split(",")[k] for r in ea[1:]], [r.split(",")[k] for r in ta[1:]]
            assert va == vb, f"{col} differs between eager and traced: eager {va} traced {vb}"
            checked += 1
        assert checked >= 4
        assert_state_equal(host_state(eager.alg), host_state(traced.alg), "eager vs traced")
    finally:
        release(traced)


def test_traced_rollout_is_not_overwritten_by_later_replays(traced_mesh, tmp_path):
    """T-15 regression: every stored step is read back right after it was stored and again at the end of the rollout; the traced values must equal
    the eager ones and must not change in between (a tensor allocated after the capture used to be overwritten by the next replay)."""
    eager, env_e = build("cartpole", tmp_path / "e", trace=False)
    traced, env_t = build("cartpole", tmp_path / "t", trace=True)
    try:
        snaps = {}
        for label, runner, env in (("eager", eager, env_e), ("traced", traced, env_t)):
            alg, ctx = runner.alg, runner.alg.ctx
            torch.manual_seed(99)
            alg.train_mode()
            obs = env.get_observations()
            now = []
            with torch.inference_mode():
                for _ in range(alg.storage.num_transitions_per_env):
                    actions = alg.act(obs)
                    obs, rewards, dones, extras = env.step(actions)
                    alg.process_env_step(obs, rewards, dones, extras)
                    now.append({k: ctx.download(alg.storage._steps[-1][k], op="H-6", batch_sharded=True) for k in STEP_FIELDS})
                end = [{k: ctx.download(s[k], op="H-6", batch_sharded=True) for k in STEP_FIELDS} for s in alg.storage._steps]
            snaps[label] = (now, end)
        assert "act" in traced.alg._traces
        for t, (a, b) in enumerate(zip(*snaps["traced"])):
            for k in STEP_FIELDS:
                assert torch.equal(a[k], b[k]), f"step {t} {k}: overwritten between insertion and the end of the rollout"
        for t, (a, b) in enumerate(zip(snaps["eager"][1], snaps["traced"][1])):
            for k in STEP_FIELDS:
                assert torch.equal(a[k], b[k]), f"step {t} {k}: traced differs from eager"
    finally:
        release(traced)


def test_a_second_algorithm_cannot_be_built_while_traces_are_alive(traced_mesh, tmp_path):
    """T-15 guard: constructing another TTPPO after a capture would put its state where the replays write -> refused, not silently corrupted."""
    first, _ = build("cartpole", tmp_path / "first", trace=True)
    try:
        first.learn(num_learning_iterations=1)
        assert first.alg._traces
        with pytest.raises(RuntimeError, match="captured trace is alive"):
            build("cartpole", tmp_path / "second", trace=True)
    finally:
        release(first)
    again, _ = build("cartpole", tmp_path / "third", trace=True)  # fine again once the traces are released
    release(again)


def run_iteration(alg, env, num_steps, seed, recorded=None):
    """One COMPLETE rollout + update of `alg`; returns (losses, the environment transitions it saw).

    With `recorded` (the transitions of another algorithm's iteration) the env is not stepped: the same observations / rewards / dones are fed in, so two
    algorithms can be compared on identical data. The algorithms must run one after the other, never interleaved: a tensor that is alive between two calls
    of one algorithm (e.g. the GAE returns between `compute_returns` and `update`) can sit where the OTHER algorithm's replays write (T-15)."""
    alg.train_mode()
    transitions = []
    with torch.inference_mode():
        if recorded is None:
            obs = env.get_observations()
            for t in range(num_steps):
                torch.manual_seed(seed + t)
                actions = alg.act(obs)
                nxt, rewards, dones, extras = env.step(actions)
                alg.process_env_step(nxt, rewards, dones, extras)
                # `extras` may also carry non-tensor entries (e.g. an episode-statistics "log" dict); the algorithm reads only the tensors ("time_outs")
                transitions.append((obs.clone(), actions.clone(), nxt.clone(), rewards.clone(), dones.clone(), {k: v.clone() for k, v in extras.items() if torch.is_tensor(v)}))
                obs = nxt
        else:
            for t, (obs, actions, nxt, rewards, dones, extras) in enumerate(recorded):
                torch.manual_seed(seed + t)
                assert torch.equal(alg.act(obs), actions), f"step {t}: sampled different actions than the reference algorithm"
                alg.process_env_step(nxt, rewards, dones, extras)
            transitions, nxt = recorded, recorded[-1][2]
        alg.compute_returns(nxt)
    torch.manual_seed(seed + 10_000)
    return alg.update(), transitions


def test_traced_path_survives_load(traced_mesh, tmp_path):
    """`load()` rewrites weights, normalizers, Adam moments and the learning rate IN PLACE, so already captured traces stay valid and use the new state:
    after loading a's checkpoint, b (with its own, different traces) must follow a bit for bit on identical data."""
    a, env_a = build("cartpole", tmp_path / "a", trace=True, seed=1)
    b, _ = build("cartpole", tmp_path / "b", trace=True, seed=2)  # both runners exist before either captures a trace (T-15)
    try:
        a.learn(num_learning_iterations=2)
        ckpt = tmp_path / "a.pt"
        a.save(str(ckpt))
        b.learn(num_learning_iterations=1)  # captures b's traces on b's OWN (different) state
        trace_ids = dict(b.alg._traces)
        assert set(trace_ids) >= {"act", "update.step.0"}
        before = host_state(b.alg)
        assert not torch.equal(before["param.actor.mlp.0.weight"], host_state(a.alg)["param.actor.mlp.0.weight"]), "b must start from a different state"
        b.load(str(ckpt), map_location="cpu")
        assert b.alg._traces == trace_ids, "load() must not touch the captured traces"
        assert_state_equal(host_state(a.alg), host_state(b.alg), "after load")
        # the replayed traces now read the loaded state: feed both algorithms the same data (a first, then b on a's recorded transitions) and compare
        steps = a.alg.storage.num_transitions_per_env
        for it in range(2):
            la, recorded = run_iteration(a.alg, env_a, steps, seed=500 + 100 * it)
            lb, _ = run_iteration(b.alg, None, steps, seed=500 + 100 * it, recorded=recorded)
            assert la == lb, f"iteration {it}: losses differ after load: {la} vs {lb}"
            assert_state_equal(host_state(a.alg), host_state(b.alg), f"iteration {it} after load")
        assert all(torch.isfinite(v).all() for v in host_state(b.alg).values())
    finally:
        release(a, b)
