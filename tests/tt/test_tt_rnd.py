# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.2 device tests, part 1: Random Network Distillation on the device against upstream `RandomNetworkDistillation` - the intrinsic reward step by step (every combination of state /
reward normalisation and weight schedule, and the three ways the RND state can relate to the actor / critic observations), its normalizers, the weight schedule, and the runner integration
(the unmodified `OnPolicyRunner` logs the intrinsic reward and the RND loss).

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8r -- python -m pytest -m tt tests/tt/test_tt_rnd.py tests/tt/test_tt_rnd_update.py -s -q -p no:cacheprovider

The chip-count independent tests (this file) also run on 1x2 / 2x4 (`TT_TEST_MESH`): the statistics of the normalizers are global like upstream's under torch.distributed.
"""

import json
import torch
from pathlib import Path

import pytest

from rsl_rl_tt.testing.parity import TOLERANCES, compare
from rsl_rl_tt.testing.reference import RefSpec
from tests.tt.test_tt_rollout import build_tt_runner, small_spec
from tests.tt.test_tt_update import make_pair

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
MEASURED: dict = {}
RND_REWARD = TOLERANCES["rnd_intrinsic_reward"]
RND_REWARD_STATS = TOLERANCES["rnd_reward_stats"]


@pytest.fixture(scope="module", autouse=True)
def _dump():
    yield
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase8_2_rnd.json").write_text(json.dumps(MEASURED, indent=2, default=str))


def rec(key, value):
    MEASURED[key] = value
    print(f"[measure] {key}: {value}")
    return value


def rnd_cfg(**over):
    cfg = {"predictor_hidden_dims": [48, 48], "target_hidden_dims": [48, 48], "num_outputs": 24, "activation": "elu", "state_normalization": True, "reward_normalization": True, "weight": 1.0, "learning_rate": 1e-3}
    cfg.update(over)
    return cfg


# env, observation sets. "rnd_state" = the actor's groups ("actor" source: nothing extra is stored), the critic's groups ("critic" source) or other groups ("own": a block of its own)
CASES = {
    "cartpole-actor": ("cartpole", None),
    "tracking12-own": ("tracking12", {"actor": ["policy"], "critic": ["policy", "privileged"], "rnd_state": ["privileged"]}),
    "tracking12-critic": ("tracking12", {"actor": ["policy"], "critic": ["policy", "privileged"], "rnd_state": ["policy", "privileged"]}),
    # the same groups in another order is a different state (and a block of its own in the storage); `policy` changes every step, `privileged` (domain-randomisation parameters) hardly does
    "tracking12-own-policy": ("tracking12", {"actor": ["policy"], "critic": ["policy", "privileged"], "rnd_state": ["privileged", "policy"]}),
}
SCHEDULES = {
    None: None,
    "constant": {"mode": "constant"},
    "step": {"mode": "step", "final_step": 3, "final_value": 0.3},
    "linear": {"mode": "linear", "initial_step": 1, "final_step": 5, "final_value": 0.1},
}


def rnd_spec(case, rnd_over=None, algorithm_over=None) -> tuple[str, RefSpec]:
    name, groups = CASES[case]
    base = small_spec(name)
    algorithm = {**base.algorithm, "rnd_cfg": rnd_cfg(**(rnd_over or {})), **(algorithm_over or {})}
    return name, RefSpec(**{**base.__dict__, "algorithm": algorithm, "obs_groups": groups})


def step_obs(env, n):
    """`n` observation batches of the test env under random actions (the data both RND implementations see)."""
    out = [env.get_observations()]
    for _ in range(n):
        obs, _, _, _ = env.step(torch.randn(env.num_envs, env.num_actions))
        out.append(obs)
    return out


@pytest.mark.parametrize(
    "case, state_norm, reward_norm, schedule",
    [
        ("cartpole-actor", False, False, None),
        ("cartpole-actor", True, True, None),
        ("tracking12-own", True, True, "linear"),
        ("tracking12-own", False, True, "step"),
        ("tracking12-critic", True, False, "constant"),
        ("tracking12-own-policy", True, True, "step"),
    ],
)
def test_intrinsic_reward_matches_upstream_step_by_step(tt_mesh, case, state_norm, reward_norm, schedule, tmp_path):
    """`TTRND.intrinsic_reward` vs `RandomNetworkDistillation.get_intrinsic_reward` on the same observations for 6 consecutive steps (the reward normalizer and the weight schedule are
    stateful: the accumulated state must agree too), after both state normalizers learned the same statistics."""
    name, spec = rnd_spec(case, dict(state_normalization=state_norm, reward_normalization=reward_norm, weight_schedule=SCHEDULES[schedule]))
    runner, env, cfg, ref_alg = make_pair(name, tmp_path, spec=spec)
    alg = runner.alg
    alg.train_mode()
    ref_alg.train_mode()
    tt, ref = alg.rnd_tt, ref_alg.rnd
    ctx = alg.ctx
    rows = env.num_envs // ctx.n_chips
    for k, v in alg.rnd.state_dict().items():
        assert torch.equal(v, ref.state_dict()[k]), f"initial RND state differs at {k}"
    assert alg.storage.rnd_source == {"cartpole-actor": "actor", "tracking12-own": "own", "tracking12-critic": "critic", "tracking12-own-policy": "own"}[case]
    torch.manual_seed(11)
    batches = step_obs(env, 8)
    for obs in batches[:3]:  # both state normalizers learn the same statistics first
        ref.update_normalization(obs)
        tt.update_normalization(tt.obs_to_device(obs), rows)
    if state_norm:
        s = tt.state_norm.host_stats()
        assert compare(s["mean"], ref.state_normalizer.mean, TOLERANCES["normalizer_update"]).passed
    worst = 0.0
    for t, obs in enumerate(batches[3:]):
        got = ctx.download(tt.step_reward(obs, rows), op="H-9")[:, 0]
        want = ref.get_intrinsic_reward(obs)
        rep = compare(got, want, RND_REWARD, f"{case} sn={state_norm} rn={reward_norm} {schedule} step {t}")
        worst = max(worst, rep.rel_fro)
        assert rep.passed, str(rep)
        assert tt.weight == pytest.approx(ref.weight, rel=1e-7), f"step {t}: weight {tt.weight} vs {ref.weight}"
        assert alg.rnd.update_counter == ref.update_counter == t + 1
        assert got.shape == want.shape and torch.isfinite(got).all()
    if reward_norm:  # the normalizer's own state: accumulator, running statistics
        rn, e = tt.reward_norm, ref.reward_normalizer
        host = lambda x: float(ctx.download(x, op="H-6", batch_sharded=False)[0, 0])
        assert rn.count == int(e.emp_norm.count)
        stats = {}
        for got, want, what in ((host(rn.mean), float(e.emp_norm._mean), "mean"), (host(rn.var), float(e.emp_norm._var), "var"), (host(rn.std), float(e.emp_norm._std), "std")):
            stats[what] = abs(got - want) / abs(want)
            assert stats[what] <= RND_REWARD_STATS.rel_fro, f"reward normalizer {what}: {got} vs {want}"  # the statistics inherit the noise of the rewards they are computed from
        avg = ctx.download(rn.avg, op="H-6")[:, 0]
        rep = compare(avg, e.disc_avg.avg, RND_REWARD, "discounted accumulator")
        assert rep.passed, str(rep)
        rec(f"reward normalizer {case} sn={state_norm} {schedule}", {"stats_rel_err": stats, "accumulator_rel_fro": rep.rel_fro})
    rec(f"intrinsic {case} sn={state_norm} rn={reward_norm} {schedule}", {"worst_rel_fro": worst})


def test_reward_normalizer_in_eval_mode_and_with_a_zero_std(tt_mesh, tmp_path):
    """Eval mode freezes the statistics and the accumulator (and matches upstream's eval-mode call); a std of exactly 0 returns the reward unchanged, like `if self.emp_norm._std > 0` upstream."""
    name, spec = rnd_spec("cartpole-actor")
    runner, env, cfg, ref_alg = make_pair(name, tmp_path, spec=spec)
    alg = runner.alg
    ctx = alg.ctx
    rows = env.num_envs // ctx.n_chips
    tt, ref = alg.rnd_tt, ref_alg.rnd
    rn = tt.reward_norm
    host = lambda x: float(ctx.download(x, op="H-6", batch_sharded=False)[0, 0])
    torch.manual_seed(2)
    batches = step_obs(env, 4)
    alg.train_mode()
    ref_alg.train_mode()
    for obs in batches[:2]:  # the statistics and the accumulator move in training mode, the same on both sides
        ctx.download(tt.step_reward(obs, rows), op="H-9")
        ref.get_intrinsic_reward(obs)
    alg.eval_mode()
    ref_alg.eval_mode()
    before = (host(rn.mean), host(rn.var), rn.count, ctx.download(rn.avg, op="H-6").clone())
    got = ctx.download(tt.step_reward(batches[2], rows), op="H-9")[:, 0]
    assert (host(rn.mean), host(rn.var), rn.count) == before[:3] and torch.equal(ctx.download(rn.avg, op="H-6"), before[3]), "eval mode must not update the normalizer"
    rep = compare(got, ref.get_intrinsic_reward(batches[2]), RND_REWARD, "eval-mode intrinsic reward")
    assert rep.passed, str(rep)
    alg.train_mode()
    rn.var = ctx.upload_scalar_into(rn.var, 0.0, op="H-8")
    rn.std = ctx.upload_scalar_into(rn.std, 0.0, op="H-8")
    rn.until = 0  # nothing learns: the variance (and so the std) stays 0
    out = ctx.download(tt.step_reward(batches[3], rows), op="H-9")
    x = tt.obs_to_device(batches[3])
    unscaled = ctx.download(tt.embedding_distance(x), op="H-9")
    assert torch.allclose(out, unscaled * tt.weight, rtol=1e-6), "a zero std must return the reward unchanged (times the weight)"


def test_the_runner_trains_with_rnd_and_logs_it(tt_mesh, tmp_path):
    """The unmodified `OnPolicyRunner` + `TTPPO` with `rnd_cfg`: `alg.intrinsic_rewards` is a host tensor of one value per environment after every step (the runner logs it), `Loss/rnd` and the
    `Rnd/*` statistics are in the logs, the counter and the weight follow the schedule, and the intrinsic reward is part of the stored rewards."""
    name, spec = rnd_spec("cartpole-actor", dict(weight_schedule=SCHEDULES["step"]))
    runner, env, cfg = build_tt_runner(name, tmp_path, spec=spec)
    alg = runner.alg
    assert type(alg).__name__ == "TTPPO" and alg.rnd is not None and alg.rnd_tt is not None
    runner.learn(num_learning_iterations=4, init_at_random_ep_len=True)  # random initial episode lengths: episodes finish within the 32 steps (the Rnd/* statistics are logged then)
    steps = 4 * spec.num_steps
    assert alg.rnd.update_counter == steps
    assert alg.intrinsic_rewards.shape == (env.num_envs,) and alg.intrinsic_rewards.dtype == torch.float32 and torch.isfinite(alg.intrinsic_rewards).all()
    assert alg.rnd.weight == pytest.approx(0.3)  # the step schedule switched to its final value after 3 steps
    rows = (tmp_path / "metrics.csv").read_text().splitlines()
    cols = rows[0].split(",")
    for col in ("Loss/rnd", "Rnd/mean_extrinsic_reward", "Rnd/mean_intrinsic_reward", "Rnd/weight"):
        assert col in cols, f"{col} missing from the log columns {cols}"
    last = rows[-1].split(",")
    rnd_loss = float(last[cols.index("Loss/rnd")])
    assert 0.0 < rnd_loss < 100.0 and float(last[cols.index("Rnd/weight")]) == pytest.approx(0.3)
    # the storage got the intrinsic reward: with a weight of 0 the rewards are the extrinsic ones, otherwise they differ
    host_ops = alg.ctx.host_ops.snapshot()
    assert host_ops["calls"]["H-9"] == steps and host_ops["bytes"]["H-9"] == steps * env.num_envs * 4


def test_rnd_host_traffic_is_the_allow_listed_transfers_only(tt_mesh, tmp_path):
    """One iteration with RND on a `own` state: per step one more observation upload for the stored RND state and one for the reward of the step after it, and the download of the intrinsic reward
    (H-9, logging only); the update adds the per-step Adam scalars of the predictor (H-8) and its loss (H-5). No other transfer appears."""
    name, spec = rnd_spec("tracking12-own")
    runner, env, cfg = build_tt_runner(name, tmp_path, spec=spec)
    alg = runner.alg
    alg.train_mode()
    obs = env.get_observations()
    ctx = alg.ctx
    steps = spec.num_steps
    ctx.host_ops.reset()
    with torch.inference_mode():
        for _ in range(steps):
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
        alg.compute_returns(obs)
    rollout = dict(ctx.host_ops.calls)
    assert rollout["H-9"] == steps and ctx.host_ops.bytes["H-9"] == steps * env.num_envs * 4
    # act: the actor's, the critic's and the stored RND observation (H-1, H-3 each) + the noise (H-4) + the actions (H-2); process_env_step: the packed scalars and the RND state of the new
    # observation (H-1; H-3 for the latter) + the step's two scalars, weight and update rate (H-8) + the download of the intrinsic reward (H-9)
    # compute_returns uploads the critic's observation of the last state: one more H-1 / H-3
    assert rollout == {"H-1": 5 * steps + 1, "H-2": steps, "H-3": 4 * steps + 1, "H-4": steps, "H-8": steps, "H-9": steps}, rollout
    ctx.host_ops.reset()
    alg.update()
    updates = alg.num_learning_epochs * alg.num_mini_batches
    n_norm = sum(m.normalizer is not None for m in (alg.tt_actor, alg.tt_critic)) + 1  # + the RND state normalizer
    calls = dict(ctx.host_ops.calls)
    assert calls == {"H-4": 1, "H-5": 3 + 1 + 1 + 1, "H-8": 2 * updates + 2 * updates + n_norm}, calls  # PPO's and the predictor's two Adam scalars per step, one rate per normalizer; the rnd loss is logged


@pytest.mark.parametrize("case", ["cartpole-actor", "tracking12-own-policy", "tracking12-critic"])
def test_the_rollout_stores_the_rnd_state_of_every_step(tt_mesh, case, tmp_path):
    """The RND state of EVERY step of the rollout (not only the last one) is in the storage, in the packed table and in the shuffled mini-batches' columns."""
    from rsl_rl_tt.tt_common import stage_obs

    name, spec = rnd_spec(case)
    runner, env, cfg = build_tt_runner(name, tmp_path, spec=spec)
    alg = runner.alg
    alg.train_mode()
    ctx = alg.ctx
    obs = env.get_observations()
    staged = []
    with torch.inference_mode():
        for _ in range(spec.num_steps):
            staged.append(stage_obs(obs, alg.rnd_tt.groups, alg.rnd_tt.state_dim_p))
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
        alg.compute_returns(obs)
        # the rows of a chip's storage are that chip's environments of step 0, 1, ...; the chips' blocks follow one another
        m = env.num_envs // ctx.n_chips
        expected = torch.cat([torch.cat([s[c * m : (c + 1) * m] for s in staged]) for c in range(ctx.n_chips)])
        stored = ctx.download(alg.storage.all_rnd_obs(), op="H-6")
        assert torch.equal(stored, expected), "the stored RND states differ from the observations the actor acted on"
        assert len({tuple(s[0].tolist()) for s in staged}) > 1, "the test needs observations that change from step to step"
        table = ctx.download(alg.storage.packed_table(), op="H-6")
        st = alg.storage
    cols = {"actor": (st.col_actor, st.col_actor + st.actor_dim_p), "critic": (st.col_critic, st.col_critic + st.critic_dim_p) if not st.share_obs else (st.col_actor, st.col_actor + st.actor_dim_p), "own": (st.col_rnd, st.col_rnd + st.rnd_dim_p)}[st.rnd_source]
    assert torch.equal(table[:, cols[0] : cols[1]], expected)
