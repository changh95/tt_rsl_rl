# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests of the distillation support code (no device): the reference configuration, the teachers made by upstream PPO, and the upstream `Distillation` on the CPU test environments (the
oracle of the device tests): what its `load()` does with an RL checkpoint, that the runner refuses to learn without a teacher, the window semantics of its update."""

import dataclasses

import pytest
import torch

from rsl_rl.runners import DistillationRunner
from rsl_rl_tt.testing.envs import ENV_NAMES, make_env
from rsl_rl_tt.testing.reference import DISTILL_GROUPS, REF_SPECS, build_distillation_runner, make_distill_cfg, make_teacher_checkpoint


@pytest.fixture(scope="module")
def teachers(tmp_path_factory):
    d = tmp_path_factory.mktemp("teachers")
    return {name: make_teacher_checkpoint(name, str(d / f"{name}.pt"), seed=0, iterations=2, num_envs=64) for name in ENV_NAMES}


def test_the_teacher_checkpoint_is_an_rl_checkpoint_with_the_teachers_observation_groups(teachers):
    ck = torch.load(teachers["tracking12"], weights_only=False)
    assert "actor_state_dict" in ck and "student_state_dict" not in ck
    assert ck["actor_state_dict"]["mlp.0.weight"].shape[1] == 84  # the teacher of tracking12 sees `policy` (36) + `privileged` (48)
    assert make_distill_cfg("tracking12")["obs_groups"] == DISTILL_GROUPS["tracking12"] and make_distill_cfg("cartpole")["obs_groups"]["teacher"] == ["policy"]


@pytest.mark.parametrize("name", ENV_NAMES)
def test_the_upstream_runner_distills_on_every_test_env(name, teachers, tmp_path):
    spec = dataclasses.replace(REF_SPECS[name], num_envs=64, num_steps=8, iterations=2)
    runner, env, _ = build_distillation_runner(name, 0, str(tmp_path), teachers[name], spec=spec)
    alg = runner.alg
    assert alg.teacher_loaded and runner.current_learning_iteration == 0  # an RL checkpoint: only the teacher, not the iteration
    teacher_before = {k: v.clone() for k, v in alg._raw_teacher.state_dict().items()}
    runner.learn(num_learning_iterations=2)
    assert all(torch.equal(v, alg._raw_teacher.state_dict()[k]) for k, v in teacher_before.items()), "the teacher must not change"
    header = (tmp_path / "metrics.csv").read_text().splitlines()[0].split(",")
    assert "Loss/behavior" in header and "Loss/value" not in header


def test_the_runner_refuses_to_learn_without_a_teacher():
    spec = dataclasses.replace(REF_SPECS["cartpole"], num_envs=32, num_steps=8)
    env = make_env("cartpole", 32, seed=0)
    runner = DistillationRunner(env, make_distill_cfg("cartpole", spec), log_dir=None, device="cpu")
    with pytest.raises(ValueError, match="Teacher model parameters not loaded"):
        runner.learn(num_learning_iterations=1)


@pytest.mark.parametrize("epochs, window, applies", [(1, 8, 1), (2, 4, 4), (2, 6, 2), (3, 1, 24)])
def test_upstream_applies_one_optimizer_step_per_window_and_never_the_leftover_steps(epochs, window, applies, teachers, tmp_path):
    """The semantics the device implements: the step counter runs across epochs; one Adam step per `gradient_length` steps; the steps left over are never applied."""
    spec = dataclasses.replace(REF_SPECS["cartpole"], num_envs=32, num_steps=8)
    runner, env, _ = build_distillation_runner("cartpole", 0, None, teachers["cartpole"], algorithm=dict(num_learning_epochs=epochs, gradient_length=window), spec=spec)
    alg = runner.alg
    alg.train_mode()
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(8):
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
    alg.update()
    state = alg.optimizer.state
    steps = {int(s["step"]) for s in state.values()}
    assert steps == {applies}, f"{steps} optimizer steps, expected {applies}"
    assert all(p not in state for p in [alg._raw_student.distribution.std_param]), "the std parameter never receives a gradient: torch keeps no optimizer state for it"
