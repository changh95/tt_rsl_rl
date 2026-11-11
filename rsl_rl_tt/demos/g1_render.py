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


def curve_strip(metrics: str, checkpoints: list[str], size: tuple[int, int], labels: list[str]) -> np.ndarray:
    """The training curve under the panels: mean episode return (left) and achieved vs commanded forward speed (right) against the iteration (log axis), with the iteration of every panel marked."""
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = size
    rows = [{k: (float(v) if v != "" else None) for k, v in r.items()} for r in csv.DictReader(open(metrics))]
    series = lambda key: [(r["iteration"] + 1, r[key]) for r in rows if r.get(key) is not None]

    def smooth(points: list, window: int = 9) -> list:
        """Trailing mean over `window` points: the speed of the finished episodes of one iteration is noisy, the trend is what the plot is for."""
        return [(x, sum(v for _, v in points[max(0, i - window + 1) : i + 1]) / len(points[max(0, i - window + 1) : i + 1])) for i, (x, _) in enumerate(points)]
    bg, fg, grid = [c / 255 for c in BACKGROUND], (0.82, 0.84, 0.88), (0.3, 0.33, 0.4)
    fig, axes = plt.subplots(1, 2, figsize=(w / 100, h / 100), dpi=100, facecolor=bg)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.2, wspace=0.12)
    panels = [("training curve: mean episode return of all robots", "Train/mean_reward"), ("forward speed: achieved (solid) vs commanded (dashed), m/s", "Metrics/forward_speed")]
    for ax, (title, key) in zip(axes, panels):
        ax.set_facecolor(bg)
        ax.plot(*zip(*(series(key) if key == "Train/mean_reward" else smooth(series(key)))), color=fg, linewidth=1.4)
        if key == "Metrics/forward_speed":
            ax.plot(*zip(*smooth(series("Metrics/command_vx"))), color=fg, linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xscale("log")
        ax.set_title(title + (" (mean of 9 iterations)" if key != "Train/mean_reward" else ""), color=fg, fontsize=8, loc="left", pad=11)
        ax.tick_params(colors=fg, labelsize=7, length=2)
        ax.set_xlabel("training iteration (log scale)", color=fg, fontsize=7, labelpad=1)
        ax.grid(alpha=0.35, color=grid, linewidth=0.5)
        for s in ax.spines.values():
            s.set_color(grid)
        xs_all = dict(series(key))
        for k, (ck, color, label) in enumerate(zip(checkpoints, COLORS, labels)):
            it = iteration_of(ck)
            ax.axvline(it, color=[c / 255 for c in color], linewidth=1.0, alpha=0.9)
            y = xs_all.get(it)
            if y is not None:
                ax.plot([it], [y], "o", color=[c / 255 for c in color], markersize=5)
            ax.annotate(f"iteration {it}", xy=(it, 1.0), xycoords=("data", "axes fraction"), color=[c / 255 for c in color], fontsize=7, ha="right" if it > 1000 else "center", va="bottom", xytext=(0, 1), textcoords="offset points")
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
                      video: str | None = None, gif: str | None = None, schedule=SCHEDULE, metrics: str | None = None) -> int:
    """The checkpoints side by side, every robot following the same scripted commands; returns the number of frames.

    `metrics`: the `metrics.csv` of the training run; adds the live return of every robot under its panel and the training curve under all panels (see the module docstring)."""
    probe = MujocoG1VecEnv(1, seed=0, num_threads=1)
    skip = max(1, round(1.0 / (fps * probe.dt)))
    real_time_fps = 1.0 / (skip * probe.dt)
    probe.close()
    results = [render_panel_traced(load_policy(ck, hidden), schedule, skip, size, seed, label_of(ck)) for ck in checkpoints]
    panels = [r[0] for r in results]
    n = min(len(p) for p in panels)
    h, w = size
    if metrics is None:
        frames = [np.concatenate([p[i] for p in panels], axis=1) for i in range(n)]
    else:
        from PIL import Image, ImageDraw

        live_h, curve_h = 16 * round(h * 0.23 / 16), 16 * round(h * 0.5 / 16)
        traces = [r[1] for r in results]
        cums = [episode_returns(tr) for tr in traces]
        span = (min(0.0, min(float(c.min()) for c in cums)), max(1.0, max(float(c.max()) for c in cums)))  # one scale for all panels
        labels = [label_of(ck) for ck in checkpoints]
        curves = curve_strip(metrics, checkpoints, (curve_h, w * len(panels)), labels)
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
        write_gif(frames, gif, real_time_fps, **({"width": 720, "every": 5, "colors": 48} if metrics else {}))
    return n
