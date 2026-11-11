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


def _write_run(tmp_path, iterations: int = 120):
    """A synthetic training run: metrics.csv with every column the strip reads, and a tracking-error json with three checkpoints."""
    import json

    terms = list(MujocoG1VecEnv.WEIGHTS)
    header = ["iteration", "Metrics/episode_length", "Loss/value"] + [f"Episode_Reward/{t}" for t in terms]
    rows = [",".join(header)]
    for i in range(iterations):
        length = 100 + 9 * i
        # iteration 3 has no finished episode: the statistics of the episodes are empty, the loss is not
        rows.append(",".join([str(i), "" if i == 3 else str(length), str(0.05 / (1 + i))] + ["" if i == 3 else str(MujocoG1VecEnv.WEIGHTS[t] * 0.01 * length * 0.02) for t in terms]))
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("\n".join(rows))
    tracking = tmp_path / "tracking_error.json"
    tracking.write_text(json.dumps({"points": [{"iteration": it, "lin_vel_error": e, "yaw_rate_error": e / 2} for it, e in ((1, 0.5), (101, 0.4), (1500, 0.1))]}))
    return str(metrics), str(tracking)


def test_the_reward_strip_and_the_training_plots_have_the_requested_sizes(tmp_path):
    trace = {"rewards": np.linspace(0.0, 0.1, 50), "dones": np.zeros(50, dtype=bool), "dt": 0.02}
    trace["dones"][20] = True
    cum = g1_render.episode_returns(trace)
    strip = g1_render.draw_return_strip(trace, cum, 30, (32, 192), (0.0, float(cum.max())), g1_render.COLORS[0], "x")
    assert strip.shape == (32, 192, 3) and strip.dtype == np.uint8 and (strip != np.array(g1_render.BACKGROUND)).any()
    metrics, tracking = _write_run(tmp_path)
    img = g1_render.curve_strip(metrics, ["random", "model_100.pt", "model_1499.pt"], (320, 1440), ["a", "b", "c"], tracking)
    assert img.shape == (320, 1440, 3) and img.dtype == np.uint8 and img.std() > 1.0


def test_every_reward_term_of_the_task_is_in_exactly_one_group_of_the_plot():
    grouped = [t for _, _, terms, _ in g1_render.REWARD_GROUPS for t in terms]
    assert sorted(grouped) == sorted(MujocoG1VecEnv.WEIGHTS)
    # the sign of a group is the sign of the weights of its terms: rewards above zero, penalties below
    for name, sign, terms, _ in g1_render.REWARD_GROUPS:
        assert all(np.sign(MujocoG1VecEnv.WEIGHTS[t]) == sign for t in terms), name


def test_the_reward_rates_are_the_weighted_sums_per_second_of_episode_time(tmp_path):
    metrics, _ = _write_run(tmp_path, iterations=6)
    xs, rate = g1_render.reward_rates(metrics, window=1)
    assert xs == [1, 2, 3, 5, 6]  # iteration 3 (x = 4) had no finished episode
    # the weighted sum of one episode is weight x 0.01 x length x dt: per second of episode time (length x dt) that is weight x 0.01
    for t, w in MujocoG1VecEnv.WEIGHTS.items():
        assert np.allclose(rate[t], w * 0.01), t
    _, smooth = g1_render.reward_rates(metrics, window=3)
    assert np.allclose(smooth["track_lin_vel"], MujocoG1VecEnv.WEIGHTS["track_lin_vel"] * 0.01)  # a constant stays constant under the trailing mean


def test_the_velocity_tracking_error_follows_the_command():
    """Zero action = the robots hold the keyframe pose: they stand still, so the error is ~0 for a zero command and ~|command| for a forward command."""
    zero = lambda obs: torch.zeros(obs.batch_size[0], 12)
    still = g1_render.tracking_error(zero, schedule=[((0.0, 0.0, 0.0), 0.3)], envs=4)
    forward = g1_render.tracking_error(zero, schedule=[((0.8, 0.0, 0.0), 0.3)], envs=4)
    assert still["lin_vel_error"] < 0.25 and still["yaw_rate_error"] < 0.25
    assert 0.55 < forward["lin_vel_error"] < 1.0 and forward["yaw_rate_error"] < 0.25
    assert forward["robots"] == 4 and forward["seconds"] == 0.3 and forward["fell_per_robot_minute"] >= 0.0


def test_the_tracking_curve_covers_the_untrained_network_and_the_saved_checkpoints(tmp_path):
    import json

    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl_tt.demos.g1_cfg import g1_train_cfg

    env = MujocoG1VecEnv(2, seed=0, num_threads=1)
    try:
        runner = OnPolicyRunner(env, g1_train_cfg(hidden=(32, 16), csv_log=False), log_dir=None, device="cpu")
        for it in (0, 100, 200):
            runner.save(str(tmp_path / f"model_{it}.pt"))
    finally:
        env.close()
    out = g1_render.tracking_curve(str(tmp_path), hidden=(32, 16), robots=2, log=lambda m: None)
    assert [p["iteration"] for p in out["points"]] == [1, 101, 201]  # model_0.pt would share the iteration 1 of the untrained network
    assert out["points"][0]["checkpoint"] == "random" and out["points"][1]["checkpoint"] == "model_100.pt"
    json.dumps(out)  # serialisable


def test_a_comparison_with_reward_and_training_strips(tmp_path):
    import mujoco

    metrics, tracking = _write_run(tmp_path)
    video = tmp_path / "c.mp4"
    try:
        n = g1_render.render_comparison(["random", "random"], size=(128, 192), schedule=[((0.4, 0.0, 0.0), 0.2)], video=str(video), metrics=str(metrics), tracking=tracking, fps=25)
    except (mujoco.FatalError, ImportError, RuntimeError) as e:
        pytest.skip(f"headless rendering unavailable: {e}")
    assert n >= 2 and video.stat().st_size > 0
    with pytest.raises(ValueError, match="go together"):
        g1_render.render_comparison(["random"], size=(128, 192), schedule=[((0.4, 0.0, 0.0), 0.1)], metrics=str(metrics))


@pytest.mark.parametrize("script", ["examples/g1_walk/train.py", "examples/g1_walk/play.py", "scripts/render_g1_demo.py", "scripts/render_g1_gait.py"])
def test_the_scripts_start_and_describe_themselves(script):
    out = subprocess.run([sys.executable, str(REPO / script), "--help"], capture_output=True, text=True, timeout=120, cwd=REPO)
    assert out.returncode == 0 and "usage:" in out.stdout, out.stderr[-500:]
