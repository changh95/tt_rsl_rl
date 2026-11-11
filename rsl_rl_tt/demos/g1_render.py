# SPDX-License-Identifier: BSD-3-Clause
"""Evaluation and rendering of a trained Unitree G1 policy (shared by `examples/g1_walk/play.py` and `scripts/render_g1_demo.py`).

The checkpoints are upstream-format `model_<it>.pt` files; they are loaded into the unmodified upstream `OnPolicyRunner` and the policy is run deterministically (the distribution
mean). `random` stands for an untrained policy. Headless rendering uses MuJoCo's EGL backend (`MUJOCO_GL=egl`, set by default here), the video is written with imageio-ffmpeg
(H.264) and the GIF with Pillow.
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
    try:
        with torch.inference_mode():
            for cmd, seconds in schedule:
                env.set_command(cmd)
                obs = env.get_observations()
                for _ in range(int(seconds / env.dt)):
                    obs, _, dones, _ = env.step(policy(obs))
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
    return frames


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
                      video: str | None = None, gif: str | None = None, schedule=SCHEDULE) -> int:
    """The checkpoints side by side, every robot following the same scripted commands; returns the number of frames."""
    probe = MujocoG1VecEnv(1, seed=0, num_threads=1)
    skip = max(1, round(1.0 / (fps * probe.dt)))
    real_time_fps = 1.0 / (skip * probe.dt)
    probe.close()
    panels = [render_panel(load_policy(ck, hidden), schedule, skip, size, seed, label_of(ck)) for ck in checkpoints]
    n = min(len(p) for p in panels)
    frames = [np.concatenate([p[i] for p in panels], axis=1) for i in range(n)]
    if video:
        write_video(frames, video, real_time_fps)
    if gif:
        write_gif(frames, gif, real_time_fps)
    return n
