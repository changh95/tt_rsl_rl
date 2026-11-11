# SPDX-License-Identifier: BSD-3-Clause
"""Evaluation and rendering of a trained Unitree G1 policy (shared by `examples/g1_walk/play.py` and `scripts/render_g1_demo.py`).

The checkpoints are upstream-format `model_<it>.pt` files; they are loaded into the unmodified upstream `OnPolicyRunner` and the policy is run deterministically (the distribution
mean). `random` stands for an untrained policy. Headless rendering uses MuJoCo's EGL backend (`MUJOCO_GL=egl`, set by default here), the video is written with imageio-ffmpeg
(H.264) and the GIF with Pillow.

With `metrics` (the `metrics.csv` of the training run) the video also shows the REWARD and the LEARNING CURVE: under every panel the return of the current episode of that robot
(the sum of its rewards since the last reset, live; the same scale for all panels), and under the three panels the training curve (mean episode return and achieved vs
commanded speed against the iteration, log axis) with the iteration of every panel marked.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner
from rsl_rl_tt.demos.g1_cfg import g1_train_cfg
from rsl_rl_tt.demos.mujoco_g1 import MujocoG1VecEnv, quat_to_rotation

# (vx, vy, yaw rate), seconds: the commands the comparison video follows (inside the commands of training: vx 0.2-0.8, yaw rate +-0.4)
SCHEDULE = [((0.4, 0.0, 0.0), 5.0), ((0.8, 0.0, 0.0), 5.0), ((0.5, 0.0, 0.4), 5.0), ((0.5, 0.0, -0.4), 5.0)]


def load_policy(checkpoint: str, hidden: tuple[int, ...] = (512, 256, 128)):
    """The deterministic policy of a checkpoint (or of the untrained network for `random`)."""
    env = MujocoG1VecEnv(1, seed=0, num_threads=1)
    try:
        torch.manual_seed(0)
        runner = OnPolicyRunner(env, g1_train_cfg(hidden=hidden, csv_log=False), log_dir=None, device="cpu")
        if checkpoint != "random":
            runner.load(checkpoint, load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False})
        return runner.get_inference_policy(device="cpu")
    finally:
        env.close()


def label_of(checkpoint: str) -> str:
    """`model_100.pt` -> "iteration 101" (the checkpoint is written after that many iterations; `model_0.pt` is iteration 1)."""
    if checkpoint == "random":
        return "iteration 1 (untrained)"
    m = re.search(r"model_(\d+)", checkpoint)
    return f"iteration {int(m.group(1)) + 1}" if m else Path(checkpoint).stem


def iteration_of(checkpoint: str) -> int:
    """The number of the iteration a checkpoint stands for (1 = untrained), the one in `label_of`."""
    if checkpoint == "random":
        return 1
    m = re.search(r"model_(\d+)", checkpoint)
    return int(m.group(1)) + 1 if m else 1


def measure(policy, command: tuple[float, float, float], steps: int = 600, envs: int = 256, seed: int = 7) -> dict:
    """Mean body-frame forward speed and yaw rate of `envs` robots holding one command, and how many episodes ended in a fall (a time-out is not a fall)."""
    env = MujocoG1VecEnv(envs, seed=seed, num_threads=16)
    env.set_command(command)
    try:
        obs = env.get_observations()
        vx, wz, falls, episodes = [], [], 0, 0
        with torch.inference_mode():
            for t in range(steps):
                obs, _, dones, extras = env.step(policy(obs))
                s = env._state
                v_body = np.einsum("nji,nj->ni", quat_to_rotation(s[:, 4:8]), s[:, 1 + env.nq : 1 + env.nq + 3])
                if t >= 100:  # let the gait settle after the start / after a reset
                    ok = env._steps > 50
                    vx.append(float(v_body[ok, 0].mean()) if ok.any() else 0.0)
                    wz.append(float(s[ok, 1 + env.nq + 5].mean()) if ok.any() else 0.0)
                episodes += int(dones.sum())
                if "log" in extras:
                    falls += round(extras["log"]["Episode_Termination/fell"] * int(dones.sum()))
        return {"command_vx": command[0], "command_wz": command[2], "mean_vx": float(np.mean(vx)), "mean_yaw_rate": float(np.mean(wz)),
                "episodes_finished": episodes, "fell_fraction": falls / episodes if episodes else 0.0, "steps": steps, "robots": envs}
    finally:
        env.close()


def tracking_error(policy, schedule=SCHEDULE, envs: int = 128, seed: int = 7) -> dict:
    """Velocity tracking error of `envs` robots that all follow the command `schedule` of the video (a robot that fell starts again, like in training), deterministic policy.

    Per control step and robot: the length of (command - body-frame velocity) in the horizontal plane (m/s) and |command - yaw rate| (rad/s); the mean over all robots and steps. The step on which an
    episode ends is left out (the state it returns is the one after the reset). `fell_per_robot_minute`: falls per minute of robot time."""
    env = MujocoG1VecEnv(envs, seed=seed, num_threads=16)
    lin, yaw, falls, count = 0.0, 0.0, 0, 0
    try:
        with torch.inference_mode():
            for cmd, seconds in schedule:
                env.set_command(cmd)
                obs = env.get_observations()
                for _ in range(int(seconds / env.dt)):
                    obs, _, dones, extras = env.step(policy(obs))
                    s = env._state
                    v_body = np.einsum("nji,nj->ni", quat_to_rotation(s[:, 4:8]), s[:, 1 + env.nq : 1 + env.nq + 3])
                    ok = ~dones.numpy().astype(bool)
                    lin += float(np.linalg.norm(np.asarray(cmd[:2]) - v_body[ok, :2], axis=1).sum())
                    yaw += float(np.abs(cmd[2] - s[ok, 1 + env.nq + 5]).sum())
                    count += int(ok.sum())
                    if "log" in extras:
                        falls += round(extras["log"]["Episode_Termination/fell"] * int(dones.sum()))
        seconds_total = sum(sec for _, sec in schedule)
        return {"lin_vel_error": lin / count, "yaw_rate_error": yaw / count, "fell_per_robot_minute": falls / (envs * seconds_total / 60.0), "robots": envs, "seconds": seconds_total}
    finally:
        env.close()


def tracking_curve(directory: str, hidden: tuple[int, ...] = (512, 256, 128), robots: int = 128, seed: int = 7, log=print) -> dict:
    """`tracking_error` of the untrained network (`random`, iteration 1) and of every `model_<it>.pt` of `directory` except `model_0.pt` (one update only: it would share the iteration 1).
    Returns {"schedule", "policy", "points": [{"iteration", "checkpoint", "lin_vel_error", "yaw_rate_error", ...}]} (the json of `scripts/eval_g1_tracking.py`)."""
    import time

    found = sorted((p for p in Path(directory).glob("model_*.pt") if iteration_of(p.name) > 1), key=lambda p: iteration_of(p.name))
    points = []
    for ck in ["random"] + [str(p) for p in found]:
        t0 = time.perf_counter()
        r = tracking_error(load_policy(ck, hidden), envs=robots, seed=seed)
        points.append({"iteration": iteration_of(ck), "checkpoint": Path(ck).name if ck != "random" else "random", **r})
        log(f"iteration {points[-1]['iteration']:5d}: linear {r['lin_vel_error']:.3f} m/s, yaw rate {r['yaw_rate_error']:.3f} rad/s, {r['fell_per_robot_minute']:.2f} falls per robot-minute ({time.perf_counter() - t0:.0f} s)")
    return {"schedule": [{"vx": c[0], "vy": c[1], "yaw_rate": c[2], "seconds": sec} for c, sec in SCHEDULE], "policy": "deterministic", "points": points}


def render_panel(policy, schedule, skip: int, size: tuple[int, int], seed: int, label: str) -> list[np.ndarray]:
    """Frames of ONE robot following the command schedule (after a fall it starts again, like in training); `size` is (height, width); a frame every `skip` control steps."""
    return render_panel_traced(policy, schedule, skip, size, seed, label)[0]


def render_panel_traced(policy, schedule, skip: int, size: tuple[int, int], seed: int, label: str) -> tuple[list[np.ndarray], dict]:
    """`render_panel` and the per-control-step record of the run: {"rewards": [T] reward of every step, "dones": [T] bool (the episode ended: a fall or the time limit), "dt": step length}."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    env = MujocoG1VecEnv(1, seed=seed, num_threads=1)
    env.model.vis.headlight.ambient[:] = 0.45  # the scene's own light is fixed near the origin: a headlight that follows the camera keeps the robot lit while it walks away
    env.model.vis.headlight.diffuse[:] = 0.7
    h, w = size
    font = ImageFont.load_default(size=max(10, h // 20))
    renderer = mujoco.Renderer(env.model, h, w)
    data = mujoco.MjData(env.model)
    cam = mujoco.MjvCamera()
    cam.type, cam.trackbodyid, cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_TRACKING, 1, 3.2, 90.0, -8.0
    frames, obs, falls, t = [], env.get_observations(), 0, 0
    rewards, ended = [], []
    try:
        with torch.inference_mode():
            for cmd, seconds in schedule:
                env.set_command(cmd)
                obs = env.get_observations()
                for _ in range(int(seconds / env.dt)):
                    obs, reward, dones, _ = env.step(policy(obs))
                    rewards.append(float(reward[0]))
                    ended.append(bool(dones[0]))
                    falls += int(dones[0])
                    if t % skip == 0:
                        mujoco.mj_setState(env.model, data, env._state[0].copy(), env._spec)
                        mujoco.mj_forward(env.model, data)
                        renderer.update_scene(data, camera=cam)
                        img = Image.fromarray(renderer.render())
                        draw = ImageDraw.Draw(img)
                        draw.text((6, 4), f"{label}    t = {t * env.dt:4.1f} s    falls: {falls}", fill=(255, 255, 255), font=font)
                        draw.text((6, 8 + font.size), f"command: vx {cmd[0]:.1f} m/s, yaw rate {cmd[2]:+.1f} rad/s", fill=(255, 230, 120), font=font)
                        frames.append(np.asarray(img))
                    t += 1
    finally:
        renderer.close()
        env.close()
    return frames, {"rewards": np.array(rewards), "dones": np.array(ended), "dt": env.dt}


# ---- reward and learning curve ---------------------------------------------------------------------------------------------------------------------------
BACKGROUND = (22, 26, 34)
COLORS = [(235, 90, 90), (255, 180, 40), (90, 205, 130), (110, 170, 255), (200, 140, 255)]  # one per panel: also the colour of its curve and of its marker in the learning curve


def episode_returns(trace: dict) -> np.ndarray:
    """Return of the current episode after every control step: the cumulative reward since the last reset (the step that ended an episode is included, then it starts again from 0)."""
    out, acc = np.zeros(len(trace["rewards"])), 0.0
    for i, (r, d) in enumerate(zip(trace["rewards"], trace["dones"])):
        acc += r
        out[i] = acc
        if d:
            acc = 0.0
    return out


def draw_return_strip(trace: dict, cum: np.ndarray, step: int, size: tuple[int, int], span: tuple[float, float], color: tuple[int, int, int], label: str):
    """The live return of one panel up to control step `step`: a line over the time of the video, red ticks at the falls; `span` = (lowest, highest) value shared by all panels."""
    from PIL import Image, ImageDraw, ImageFont

    h, w = size
    img = Image.new("RGB", (w, h), BACKGROUND)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(10, h // 6))
    left, right, top, bottom = 8, w - 8, 18 + h // 8, h - 6
    total = len(cum)
    lo, hi = span
    ys = lambda v: bottom - (v - lo) / max(hi - lo, 1e-9) * (bottom - top)
    xs = lambda i: left + (i + 1) / total * (right - left)
    draw.line([(left, ys(0.0)), (right, ys(0.0))], fill=(70, 76, 90), width=1)
    draw.rectangle([left, top, right, bottom], outline=(50, 56, 70))
    start = 0
    for i in range(step + 1):
        if trace["dones"][i] or i == step:
            if i > start:
                draw.line([(xs(j), ys(cum[j])) for j in range(start, i + 1)], fill=color, width=2)
            if trace["dones"][i] and i < total - 1:  # an episode that ended before the end of the video: a fall (the 1000-step time limit coincides with the end of the 20 s video)
                draw.line([(xs(i), bottom), (xs(i), bottom - 6)], fill=(235, 90, 90), width=2)
            start = i + 1
    draw.text((left, 2), "return of this episode", fill=(200, 205, 215), font=font)
    value = f"{cum[step]:.1f}"
    draw.text((right - draw.textlength(value, font=font), 2), value, fill=color, font=font)
    return np.asarray(img)


# Reward terms of the demo task, grouped for the plot: (group, sign, terms, colour). The terms and their weights are the ones of `MujocoG1VecEnv.WEIGHTS` (the plot checks that every term is in a group).
REWARD_GROUPS = [
    ("velocity tracking", +1, ["track_lin_vel", "track_ang_vel"], (64, 166, 140)),
    ("gait shaping", +1, ["feet_phase", "contact_schedule", "feet_air_time"], (90, 140, 217)),
    ("alive bonus", +1, ["alive"], (150, 178, 204)),
    ("posture / balance penalties", -1, ["lin_vel_z", "ang_vel_xy", "orientation", "base_height", "hip_deviation"], (217, 166, 77)),
    ("effort / smoothness penalties", -1, ["action_rate", "torques", "dof_acc"], (166, 128, 204)),
    ("foot slip penalty", -1, ["feet_slip"], (217, 128, 102)),
    ("fall penalty (the episode ends)", -1, ["termination"], (191, 64, 77)),
]


def reward_rates(metrics: str, window: int = 9, dt: float = 0.02) -> tuple[list[int], dict[str, list[float]]]:
    """Reward of every term per second of episode time, per training iteration: the weighted per-episode sum of the finished episodes divided by their mean length x dt, smoothed over `window`
    iterations (trailing mean). Iterations without a finished episode are left out. `dt` = the control step of the task (5 substeps of 4 ms). Returns (x = iteration + 1, {term: rates})."""
    import csv

    terms = list(MujocoG1VecEnv.WEIGHTS)
    rows = [r for r in csv.DictReader(open(metrics)) if all(r.get(k, "") != "" for k in ["Metrics/episode_length"] + [f"Episode_Reward/{t}" for t in terms])]
    xs = [int(float(r["iteration"])) + 1 for r in rows]
    out = {}
    for t in terms:
        raw = [float(r[f"Episode_Reward/{t}"]) / (float(r["Metrics/episode_length"]) * dt) for r in rows]
        out[t] = [sum(raw[max(0, i - window + 1) : i + 1]) / len(raw[max(0, i - window + 1) : i + 1]) for i in range(len(raw))]
    return xs, out


def curve_strip(metrics: str, checkpoints: list[str], size: tuple[int, int], labels: list[str], tracking: str) -> np.ndarray:
    """The strip under the panels, against the training iteration (log axis), with the iteration of every panel marked:
    (1) velocity tracking error of the saved checkpoints (`tracking`: json of `scripts/eval_g1_tracking.py`), (2) the critic's value loss (the only estimation loss this model has; there is no separate encoder),
    (3) the reward terms of the actor-critic per second, stacked: rewards above zero, penalties below, the net reward as a line, with the terms and their weights listed on the right."""
    import csv
    import json
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = size
    bg, fg, dim, grid = [c / 255 for c in BACKGROUND], (0.82, 0.84, 0.88), (0.6, 0.63, 0.7), (0.3, 0.33, 0.4)
    weights = MujocoG1VecEnv.WEIGHTS
    grouped = [t for _, _, ts, _ in REWARD_GROUPS for t in ts]
    assert sorted(grouped) == sorted(weights), "every reward term of the task must be in exactly one group"
    stage_colors = [[c / 255 for c in COLORS[k % len(COLORS)]] for k in range(len(checkpoints))]
    stage_its = [iteration_of(ck) for ck in checkpoints]

    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=bg)
    top, bottom = 58, 34  # pixels above / below the axes
    cols = [(44, 270), (366, 270), (688, 430)]  # (left, width) in pixels of the three plots
    axes = [fig.add_axes([x / w, bottom / h, wd / w, (h - top - bottom) / h]) for x, wd in cols]

    def heading(i: int, title: str, *subtitle: str) -> None:
        fig.text(cols[i][0] / w, 1 - 11 / h, title, color=fg, fontsize=8.5, fontweight="bold", va="center")
        for k, line in enumerate(subtitle):
            fig.text(cols[i][0] / w, 1 - (24 + 9 * k) / h, line, color=dim, fontsize=6.3, va="center")

    def style(ax, ylabel: str) -> None:
        ax.set_facecolor(bg)
        ax.set_xscale("log")
        ax.set_xlim(0.8, 2000)
        ax.tick_params(colors=fg, labelsize=7, length=2)
        ax.set_xlabel("training iteration (log scale)", color=fg, fontsize=7, labelpad=1)
        ax.set_ylabel(ylabel, color=fg, fontsize=7, labelpad=2)
        ax.grid(alpha=0.35, color=grid, linewidth=0.5)
        for s in ax.spines.values():
            s.set_color(grid)

    def stages(ax, curve: dict | None = None, label: bool = True) -> None:
        """A line at the iteration of every panel (its colour) and, when `curve` = {iteration: value}, a dot on the curve."""
        for it, color in zip(stage_its, stage_colors):
            ax.axvline(it, color=color, linewidth=1.0, alpha=0.9, zorder=5)
            if curve is not None and it in curve:
                ax.plot([it], [curve[it]], "o", color=color, markersize=5, zorder=6)
            if label:
                ax.annotate(f"iter {it}", xy=(it, 1.0), xycoords=("data", "axes fraction"), color=color, fontsize=6.5, ha="right" if it > 1000 else "center", va="bottom", xytext=(0, 1), textcoords="offset points")

    # (1) velocity tracking error
    points = json.loads(Path(tracking).read_text())["points"]
    ax = axes[0]
    style(ax, "error")
    for key, color, name in (("lin_vel_error", (0.95, 0.95, 0.97), "linear velocity (m/s)"), ("yaw_rate_error", (0.55, 0.8, 0.95), "yaw rate (rad/s)")):
        xs, ys = [p["iteration"] for p in points], [p[key] for p in points]
        ax.plot(xs, ys, "-o", color=color, linewidth=1.3, markersize=2.6, label=name)
    ax.set_ylim(0, max(p["lin_vel_error"] for p in points) * 1.1)
    ax.legend(loc="center right", fontsize=6.5, facecolor=bg, edgecolor=grid, labelcolor=fg, framealpha=0.9, borderpad=0.4)
    stages(ax, {p["iteration"]: p["lin_vel_error"] for p in points})
    heading(0, "velocity tracking error", "length of (command - actual velocity) / |yaw rate error|:", "128 robots, the commands of the video, 20 s")

    # (2) the critic's value loss
    rows = [{k: (float(v) if v != "" else None) for k, v in r.items()} for r in csv.DictReader(open(metrics))]
    pts = [(r["iteration"] + 1, r["Loss/value"]) for r in rows if r.get("Loss/value") is not None]
    smooth = [(x, sum(v for _, v in pts[max(0, i - 8) : i + 1]) / len(pts[max(0, i - 8) : i + 1])) for i, (x, _) in enumerate(pts)]
    ax = axes[1]
    style(ax, "mean squared error")
    ax.plot(*zip(*pts), color=fg, linewidth=0.5, alpha=0.35)
    ax.plot(*zip(*smooth), color=fg, linewidth=1.4)
    ax.set_yscale("log")
    stages(ax, dict(smooth))
    heading(1, "value loss (the critic's estimate)", "PPO value loss of the critic, log scale (bold: mean of 9", "iterations). The model has no other estimation / encoder loss")

    # (3) the reward terms
    xs, rate = reward_rates(metrics)
    ax = axes[2]
    style(ax, "reward per second of episode time")
    for sign in (+1, -1):
        groups = [g for g in REWARD_GROUPS if g[1] == sign]
        ax.stackplot(xs, *[[sum(rate[t][i] for t in ts) for i in range(len(xs))] for _, _, ts, _ in groups], colors=[[c / 255 for c in col] for _, _, _, col in groups], alpha=0.9, linewidth=0)
    net = [sum(rate[t][i] for t in rate) for i in range(len(xs))]
    ax.plot(xs, net, color="white", linewidth=1.3, zorder=4)
    ax.axhline(0, color=(0.7, 0.72, 0.78), linewidth=0.6)
    stages(ax, dict(zip(xs, net)))
    heading(2, "reward the actor-critic maximises", "per second of episode time (mean of 9 iterations): rewards above zero,", "penalties below zero, the net reward is the white line")

    # the terms and their weights
    x0, y = 1136, 1 - 11 / h
    fig.text(x0 / w, y, "reward terms (weight, per second)", color=fg, fontsize=7.5, fontweight="bold", va="center")
    y -= 16 / h
    for name, sign, terms, col in REWARD_GROUPS:
        fig.patches.append(plt.Rectangle((x0 / w, y - 4 / h), 8 / w, 8 / h, transform=fig.transFigure, facecolor=[c / 255 for c in col], edgecolor="none"))
        fig.text((x0 + 13) / w, y, ("+ " if sign > 0 else "- ") + name, color=fg, fontsize=6.8, va="center")
        y -= 11 / h
        for line in textwrap.wrap(", ".join(f"{t} {weights[t]:g}" for t in terms), 62, break_long_words=False, break_on_hyphens=False):
            fig.text((x0 + 13) / w, y, line, color=dim, fontsize=6, va="center")
            y -= 9 / h
        y -= 3 / h
    fig.text(x0 / w, y - 2 / h, "no contact / collision term: a fall (pelvis below 0.45 m or tilted\nmore than 60 degrees) ends the episode and costs the fall penalty", color=dim, fontsize=6, va="top", linespacing=1.3)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def write_video(frames: list[np.ndarray], path: str, fps: float) -> None:
    """H.264 / yuv420p MP4 (sizes must be multiples of 16)."""
    import imageio.v2 as imageio

    writer = imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=16, ffmpeg_params=["-crf", "24", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"wrote {path}: {len(frames)} frames at {fps:.2f} fps = {len(frames) / fps:.1f} s, {Path(path).stat().st_size / 1e6:.1f} MB")


def write_gif(frames: list[np.ndarray], path: str, real_time_fps: float, width: int = 540, every: int = 4, colors: int = 40) -> None:
    """A small GIF preview of the video (every `every`-th frame, `width` pixels wide, `colors` colours): ~1-2 MB for 20 s."""
    from PIL import Image

    images = []
    for f in frames[::every]:
        img = Image.fromarray(f)
        images.append(img.resize((width, round(width * img.height / img.width)), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=colors))
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(round(1000 * every / real_time_fps)), loop=0, optimize=True)
    print(f"wrote {path}: {len(images)} frames, {Path(path).stat().st_size / 1e6:.1f} MB")


def render_comparison(checkpoints: list[str], *, hidden: tuple[int, ...] = (512, 256, 128), fps: int = 25, size: tuple[int, int] = (352, 480), seed: int = 7,
                      video: str | None = None, gif: str | None = None, schedule=SCHEDULE, metrics: str | None = None, tracking: str | None = None) -> int:
    """The checkpoints side by side, every robot following the same scripted commands; returns the number of frames.

    `metrics`: the `metrics.csv` of the training run and `tracking`: the json of `scripts/eval_g1_tracking.py`; add the live return of every robot under its panel and, under all panels, the velocity
    tracking error, the value loss and the reward terms against the training iteration (`curve_strip`)."""
    probe = MujocoG1VecEnv(1, seed=0, num_threads=1)
    skip = max(1, round(1.0 / (fps * probe.dt)))
    real_time_fps = 1.0 / (skip * probe.dt)
    probe.close()
    results = [render_panel_traced(load_policy(ck, hidden), schedule, skip, size, seed, label_of(ck)) for ck in checkpoints]
    panels = [r[0] for r in results]
    n = min(len(p) for p in panels)
    h, w = size
    if (metrics is None) != (tracking is None):
        raise ValueError("`metrics` (metrics.csv of the run) and `tracking` (json of scripts/eval_g1_tracking.py) go together")
    if metrics is None:
        frames = [np.concatenate([p[i] for p in panels], axis=1) for i in range(n)]
    else:
        from PIL import Image, ImageDraw

        live_h, curve_h = 16 * round(h * 0.23 / 16), 16 * round(h * 0.9 / 16)
        traces = [r[1] for r in results]
        cums = [episode_returns(tr) for tr in traces]
        span = (min(0.0, min(float(c.min()) for c in cums)), max(1.0, max(float(c.max()) for c in cums)))  # one scale for all panels
        labels = [label_of(ck) for ck in checkpoints]
        curves = curve_strip(metrics, checkpoints, (curve_h, w * len(panels)), labels, tracking)
        frames = []
        for i in range(n):
            top = []
            for k, p in enumerate(panels):
                panel = Image.fromarray(p[i])
                ImageDraw.Draw(panel).rectangle([0, 0, w, 5], fill=COLORS[k % len(COLORS)])  # the colour of this panel: also its curve below and its marker in the learning curve
                strip = draw_return_strip(traces[k], cums[k], min(i * skip, len(cums[k]) - 1), (live_h, w), span, COLORS[k % len(COLORS)], labels[k])
                top.append(np.concatenate([np.asarray(panel), strip], axis=0))
            frames.append(np.concatenate([np.concatenate(top, axis=1), curves], axis=0))
    if video:
        write_video(frames, video, real_time_fps)
    if gif:
        write_gif(frames, gif, real_time_fps, **({"width": 720, "every": 6, "colors": 40} if metrics else {}))
    return n
