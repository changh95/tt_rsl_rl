# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests of the evaluation / rendering helpers of the G1 demo (`rsl_rl_tt/demos/g1_render.py`, used by `examples/g1_walk/play.py` and `scripts/render_g1_demo.py`):
checkpoint labels, the load path of a checkpoint written by the upstream runner, the measurement, a tiny headless render, the video and GIF writers, and that the example scripts
are importable (`--help`). Skipped when the model is not fetched (`scripts/fetch_g1_model.sh`)."""

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # before mujoco is imported anywhere: it picks its OpenGL backend at import

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

from rsl_rl_tt.demos.mujoco_g1 import MujocoG1VecEnv, g1_scene_path  # noqa: E402

try:
    g1_scene_path()
except FileNotFoundError:
    pytest.skip("the G1 model is not fetched: scripts/fetch_g1_model.sh", allow_module_level=True)

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl_tt.demos import g1_render  # noqa: E402
from rsl_rl_tt.demos.g1_cfg import g1_train_cfg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_labels_name_the_iteration_of_the_checkpoint():
    assert g1_render.label_of("random") == "iteration 1 (untrained)"
    assert g1_render.label_of("model_0.pt") == "iteration 1"
    assert g1_render.label_of("/x/y/model_100.pt") == "iteration 101"  # model_<it>.pt is written after it + 1 iterations
    assert g1_render.label_of("/x/y/model_1499.pt") == "iteration 1500"
    assert g1_render.label_of("/x/y/final.pt") == "final"


def test_a_checkpoint_of_the_upstream_runner_loads_into_the_same_policy(tmp_path):
    env = MujocoG1VecEnv(8, seed=1, num_threads=2)
    try:
        torch.manual_seed(5)
        runner = OnPolicyRunner(env, g1_train_cfg(csv_log=False), log_dir=None, device="cpu")
        obs = env.get_observations()
        with torch.inference_mode():  # move the observation normaliser away from its initial state
            runner.alg.actor.update_normalization(obs) if hasattr(runner.alg.actor, "update_normalization") else None
        path = str(tmp_path / "model_0.pt")
        runner.save(path)
        expected = runner.get_inference_policy(device="cpu")(obs)
    finally:
        env.close()
    with torch.inference_mode():
        loaded = g1_render.load_policy(path)
        probe = MujocoG1VecEnv(8, seed=1, num_threads=2)
        try:
            actual = loaded(probe.get_observations())
        finally:
            probe.close()
    assert torch.equal(actual, expected)


def test_measure_reports_the_falls_of_an_untrained_policy():
    m = g1_render.measure(g1_render.load_policy("random"), (0.4, 0.0, 0.0), steps=200, envs=8)
    assert set(m) >= {"mean_vx", "mean_yaw_rate", "episodes_finished", "fell_fraction", "robots"}
    assert m["episodes_finished"] >= 8 and m["fell_fraction"] == 1.0  # the untrained robot falls after ~80 steps, every one of them


def test_a_tiny_headless_render_and_the_writers(tmp_path):
    import mujoco

    try:
        frames = g1_render.render_panel(g1_render.load_policy("random"), [((0.4, 0.0, 0.0), 0.2)], skip=5, size=(64, 96), seed=0, label="iteration 1")
    except (mujoco.FatalError, ImportError, RuntimeError) as e:  # no EGL on this host: say so rather than pass silently
        pytest.skip(f"headless rendering unavailable: {e}")
    assert len(frames) == 2 and frames[0].shape == (64, 96, 3) and frames[0].dtype == np.uint8 and frames[0].std() > 1.0
    video, gif = tmp_path / "t.mp4", tmp_path / "t.gif"
    g1_render.write_video(frames * 2, str(video), fps=25)
    g1_render.write_gif(frames * 2, str(gif), real_time_fps=25, width=48, every=2)
    assert video.stat().st_size > 0 and gif.read_bytes()[:3] == b"GIF"


def test_iteration_numbers_and_episode_returns():
    assert [g1_render.iteration_of(c) for c in ("random", "model_0.pt", "/x/model_100.pt", "model_1499.pt")] == [1, 1, 101, 1500]
    trace = {"rewards": np.array([1.0, 2.0, -2.0, 0.5, 0.5]), "dones": np.array([False, False, True, False, False]), "dt": 0.02}
    # the return of the current episode: the step that ends an episode is included, the next step starts again from zero
    assert np.allclose(g1_render.episode_returns(trace), [1.0, 3.0, 1.0, 0.5, 1.0])


def test_the_reward_strip_and_the_learning_curve_have_the_requested_sizes(tmp_path):
    trace = {"rewards": np.linspace(0.0, 0.1, 50), "dones": np.zeros(50, dtype=bool), "dt": 0.02}
    trace["dones"][20] = True
    cum = g1_render.episode_returns(trace)
    strip = g1_render.draw_return_strip(trace, cum, 30, (32, 192), (0.0, float(cum.max())), g1_render.COLORS[0], "x")
    assert strip.shape == (32, 192, 3) and strip.dtype == np.uint8 and (strip != np.array(g1_render.BACKGROUND)).any()
    metrics = tmp_path / "metrics.csv"
    rows = ["iteration,Train/mean_reward,Metrics/forward_speed,Metrics/command_vx"] + [f"{i},{i * 0.1},{'' if i < 3 else 0.01 * i},0.5" for i in range(120)]
    metrics.write_text("\n".join(rows))
    img = g1_render.curve_strip(str(metrics), ["random", "model_100.pt"], (64, 384), ["a", "b"])
    assert img.shape == (64, 384, 3) and img.dtype == np.uint8 and img.std() > 1.0


def test_a_comparison_with_reward_and_curve_strips(tmp_path):
    import mujoco

    metrics = tmp_path / "metrics.csv"
    metrics.write_text("\n".join(["iteration,Train/mean_reward,Metrics/forward_speed,Metrics/command_vx"] + [f"{i},{i * 0.1},{0.01 * i},0.5" for i in range(120)]))
    video = tmp_path / "c.mp4"
    try:
        n = g1_render.render_comparison(["random", "random"], size=(128, 192), schedule=[((0.4, 0.0, 0.0), 0.2)], video=str(video), metrics=str(metrics), fps=25)
    except (mujoco.FatalError, ImportError, RuntimeError) as e:
        pytest.skip(f"headless rendering unavailable: {e}")
    assert n >= 2 and video.stat().st_size > 0


@pytest.mark.parametrize("script", ["examples/g1_walk/train.py", "examples/g1_walk/play.py", "scripts/render_g1_demo.py", "scripts/render_g1_gait.py"])
def test_the_scripts_start_and_describe_themselves(script):
    out = subprocess.run([sys.executable, str(REPO / script), "--help"], capture_output=True, text=True, timeout=120, cwd=REPO)
    assert out.returncode == 0 and "usage:" in out.stdout, out.stderr[-500:]
