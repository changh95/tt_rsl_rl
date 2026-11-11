# SPDX-License-Identifier: BSD-3-Clause
"""Phase 2 completion test: the UNMODIFIED upstream OnPolicyRunner builds `rsl_rl_tt.ppo:TTPPO` from the config, and TT inference drives full
rollouts on the CPU test environments, step by step identical (within the measured noise floors) to the upstream torch models.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p2roll -- python -m pytest -m tt tests/tt/test_tt_rollout.py -s -q -p no:cacheprovider

Teacher forcing: at every step the same observation goes through TTPPO.act and through deep copies of the upstream actor/critic with the SAME
global-generator state (the TT path draws its noise with `torch.randn` from the global generator, exactly what upstream's Normal.sample() does).
"""

import copy
import torch

import pytest

from rsl_rl.runners import OnPolicyRunner
from rsl_rl_tt.testing.envs import ENV_NAMES, make_env
from rsl_rl_tt.testing.parity import TOLERANCES, compare
from rsl_rl_tt.testing.reference import REF_SPECS, RefSpec, make_train_cfg, test_mesh_shape

pytestmark = pytest.mark.tt

STEPS = 8


def small_spec(name: str, **over) -> RefSpec:
    base = REF_SPECS[name]
    n_chips = test_mesh_shape()[0] * test_mesh_shape()[1]  # every chip needs a multiple of 32 environments
    kw = dict(num_envs=max(128, 32 * n_chips), num_steps=STEPS, iterations=1, hidden=base.hidden, algorithm={**base.algorithm, "num_mini_batches": 4, "num_learning_epochs": 1},
              model=base.model, critic_model=base.critic_model, obs_groups=base.obs_groups)  # (the model config of the recurrent / image specs is part of the spec)
    kw.update(over)
    return RefSpec(**kw)


def build_tt_runner(name: str, tmp_path, spec: RefSpec | None = None, tt_cfg: dict | None = None, seed: int = 0):
    spec = spec or small_spec(name)
    torch.manual_seed(seed)
    env = make_env(name, spec.num_envs, seed=seed)
    cfg = make_train_cfg(name, spec)
    cfg["algorithm"]["class_name"] = "rsl_rl_tt.ppo:TTPPO"
    cfg["algorithm"]["tt_cfg"] = tt_cfg if tt_cfg is not None else {"mesh_shape": test_mesh_shape()}
    return OnPolicyRunner(env, cfg, log_dir=str(tmp_path), device="cpu"), env, cfg


@pytest.mark.parametrize("name", ENV_NAMES)
def test_runner_builds_ttppo_and_rollout_matches_the_upstream_models_step_by_step(tt_mesh, name, tmp_path):
    runner, env, cfg = build_tt_runner(name, tmp_path)
    alg = runner.alg
    assert type(alg).__name__ == "TTPPO" and type(runner).__name__ == "OnPolicyRunner"
    ref_actor, ref_critic = copy.deepcopy(alg._raw_actor), copy.deepcopy(alg._raw_critic)  # identical weights, identical normalizer state
    ctx = alg.ctx
    ctx.host_ops.reset()
    alg.train_mode()
    obs = env.get_observations()
    n_obs_uploads = 1 if cfg["obs_groups"]["actor"] == cfg["obs_groups"]["critic"] else 2

    dl = lambda t: ctx.download(t, op="H-6")  # test-side downloads of device-resident step data (counted as H-6)
    with torch.inference_mode():
        for step in range(STEPS):
            rng = torch.get_rng_state()
            actions = alg.act(obs)
            pend = alg._pending
            tt = {"mean": dl(pend["mean"])[:, : env.num_actions], "std": alg._std_host.reshape(1, -1).expand(env.num_envs, -1), "value": dl(pend["value"])[:, :1], "logp": dl(pend["logp"])[:, 0], "actions": actions.clone()}
            torch.set_rng_state(rng)
            ref_actions = ref_actor(obs, stochastic_output=True)
            ref = {"mean": ref_actor.output_mean, "std": ref_actor.output_std, "value": ref_critic(obs), "logp": ref_actor.get_output_log_prob(ref_actions), "actions": ref_actions}
            assert actions.shape == (env.num_envs, env.num_actions) and actions.dtype == torch.float32 and actions.device.type == "cpu"
            assert compare(tt["mean"], ref["mean"], TOLERANCES["mlp_forward"], f"{name} step {step} mean").passed
            assert compare(tt["actions"], ref["actions"], TOLERANCES["mlp_forward"], f"{name} step {step} actions").passed
            assert compare(tt["std"], ref["std"], TOLERANCES["eltwise_fp32"], f"{name} step {step} std").passed
            assert compare(tt["value"], ref["value"], TOLERANCES["mlp_forward"], f"{name} step {step} value").passed
            rep = compare(tt["logp"], ref["logp"], TOLERANCES["row_sum_fp32"], f"{name} step {step} log_prob")
            # the log-prob of a sampled action barely depends on the mean (a - mean = std * noise), so it is far tighter than the mean itself
            assert rep.rel_fro < 5e-4, str(rep)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)

        alg.compute_returns(obs)

    st = alg.storage
    n = STEPS * env.num_envs
    for name_, tensor in (("returns", st.returns), ("advantages", st.advantages)):
        host = dl(tensor)
        assert host.shape == (n, 1) and torch.isfinite(host).all(), name_
    assert torch.isfinite(dl(st.packed_table())).all()
    # only allow-listed transfers happened, with exactly the expected counts: per step the obs uploads (actor [+ critic]), one noise upload,
    # one action download, ONE packed (reward, done, time_out) upload; compute_returns uploads the critic's obs once; no H-5 (std is cached)
    calls = dict(ctx.host_ops.calls)
    assert calls == {"H-1": STEPS * (n_obs_uploads + 1) + 1, "H-3": STEPS * n_obs_uploads + 1, "H-4": STEPS, "H-2": STEPS, "H-6": 3 * STEPS + 2 + 1}, calls
    # (Phase 5: the four unit rows of `packed_table` used to be uploaded here, on its first call, and are now built when the storage is constructed,
    # before any trace can be captured (TTNN_ISSUES T-15): the same four H-6 transfers, but outside this counting window.)


@pytest.mark.parametrize("name", ENV_NAMES)
def test_runner_learn_trains_end_to_end_with_the_device_update(tt_mesh, name, tmp_path):
    runner, env, _ = build_tt_runner(name, tmp_path)
    runner.learn(num_learning_iterations=3)  # rollout (TT inference) + TT update + logging, driven by the unmodified OnPolicyRunner
    alg = runner.alg
    assert alg.storage.step == 0  # cleared by the update
    assert 1e-5 <= alg.learning_rate <= 1e-2 and alg.learning_rate == alg.learning_rate
    std = alg.get_policy().output_std
    assert std.shape[-1] == env.num_actions and torch.isfinite(std).all()
    rows = [line.split(",") for line in (tmp_path / "metrics.csv").read_text().splitlines()]
    header = rows[0]
    for col in ("Loss/value", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate", "Policy/mean_std"):
        k = header.index(col)
        assert len(rows) == 4 and all(torch.isfinite(torch.tensor(float(r[k]))) for r in rows[1:]), col
    runner.save(str(tmp_path / "model.pt"))  # device -> mirror sync, then upstream's own save()
    sd = torch.load(tmp_path / "model.pt", weights_only=False)
    assert {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter"} <= set(sd)


def test_std_mirror_and_policy_export_path_work_without_an_update(tt_mesh, tmp_path):
    runner, env, _ = build_tt_runner("tracking12", tmp_path)
    alg = runner.alg
    assert torch.allclose(alg.get_policy().output_std, torch.ones(1, 12))  # init_std = 1.0
    sd = alg.save()
    assert {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"} <= set(sd)
    policy = runner.get_inference_policy()
    out = policy(env.get_observations())
    assert out.shape == (env.num_envs, 12)


@pytest.mark.parametrize(
    "spec_kwargs, match",
    [
        ({"num_envs": 100}, "num_envs of 100"),
        ({"num_envs": 32 * test_mesh_shape()[0] * test_mesh_shape()[1], "num_steps": 2, "algorithm": {"num_mini_batches": 8}}, "mini-batch"),  # 8 rows per chip and mini-batch
    ],
)
def test_alignment_rules_refuse_bad_configurations(tt_mesh, tmp_path, spec_kwargs, match):
    base = small_spec("cartpole")
    spec = RefSpec(**{**base.__dict__, **{k: ({**base.algorithm, **v} if k == "algorithm" else v) for k, v in spec_kwargs.items()}})
    with pytest.raises(ValueError, match=match):
        build_tt_runner("cartpole", tmp_path, spec=spec)


def test_unknown_tt_cfg_key_is_refused(tt_mesh, tmp_path):
    with pytest.raises(ValueError, match="unknown tt_cfg keys"):
        build_tt_runner("cartpole", tmp_path, tt_cfg={"mesh_shape": [1, 1], "bogus": 1})
