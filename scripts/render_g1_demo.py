# SPDX-License-Identifier: BSD-3-Clause
"""Render what a trained Unitree G1 policy does: a video (several checkpoints side by side, following a scripted sequence of velocity commands) and an evaluation of each
(episode length, falls, speed tracking). The checkpoints are upstream-format `model_<it>.pt` files; they are loaded into the unmodified upstream `OnPolicyRunner` and the policy
is run deterministically (distribution mean). `random` renders an untrained policy.

    MUJOCO_GL=egl python scripts/render_g1_demo.py --checkpoints random .run/g1/tt_8chips/model_300.pt .run/g1/tt_8chips/model_1499.pt --video docs/porting/demo_g1/g1.mp4

Headless rendering uses EGL (software Mesa); the video is encoded with imageio-ffmpeg (H.264), the optional GIF with Pillow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl_tt.demos.g1_cfg import g1_train_cfg  # noqa: E402
from rsl_rl_tt.demos.mujoco_g1 import MujocoG1VecEnv  # noqa: E402

# (vx, vy, yaw rate, seconds): the commands the video follows
SCHEDULE = [((0.4, 0.0, 0.0), 5.0), ((0.8, 0.0, 0.0), 5.0), ((0.5, 0.0, 0.4), 5.0), ((0.5, 0.0, -0.4), 5.0)]  # inside the commands of training (vx 0.2-0.8, yaw rate +-0.4)


def load_policy(checkpoint: str, hidden: tuple[int, ...]):
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
    if checkpoint == "random":
        return "untrained"
    m = re.search(r"model_(\d+)", checkpoint)
    return f"iteration {int(m.group(1)) + 1}" if m else Path(checkpoint).stem


def measure(policy, command, steps: int, envs: int, seed: int) -> dict:
    """Mean body-frame forward speed, yaw rate and the fraction of falls over `steps` steps for one held command."""
    from rsl_rl_tt.demos.mujoco_g1 import quat_to_rotation

    env = MujocoG1VecEnv(envs, seed=seed, num_threads=16)
    env.set_command(command)
    try:
        obs = env.get_observations()
        vx, wz, falls, episodes = [], [], 0, 0
        with torch.inference_mode():
            for t in range(steps):
                obs, _, dones, extras = env.step(policy(obs))
                s = env._state
                rot = quat_to_rotation(s[:, 4:8])
                v_body = np.einsum("nji,nj->ni", rot, s[:, 1 + env.nq : 1 + env.nq + 3])
                if t >= 100:  # let the gait settle after the start / after a reset
                    ok = env._steps > 50
                    vx.append(float(v_body[ok, 0].mean()) if ok.any() else 0.0)
                    wz.append(float(s[ok, 1 + env.nq + 5].mean()) if ok.any() else 0.0)
                episodes += int(dones.sum())
                if "log" in extras:
                    falls += round(extras["log"]["Episode_Termination/fell"] * int(dones.sum()))
        return {"command_vx": command[0], "command_wz": command[2], "mean_vx": float(np.mean(vx)), "mean_yaw_rate": float(np.mean(wz)),
                "episodes_finished": episodes, "fell_fraction": falls / episodes if episodes else 0.0, "steps": steps}
    finally:
        env.close()


def render_panel(policy, schedule, skip: int, size: tuple[int, int], seed: int, label: str) -> list[np.ndarray]:
    """Frames of ONE robot following the command schedule (after a fall it starts again, like in training)."""
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


def write_gif(frames: list[np.ndarray], path: str, real_time_fps: float, width: int = 540, every: int = 4, colors: int = 40) -> None:
    """A small GIF preview of the video (every `every`-th frame, `width` pixels wide, `colors` colours): ~1 MB for 20 s."""
    from PIL import Image

    images = []
    for f in frames[::every]:
        img = Image.fromarray(f)
        images.append(img.resize((width, round(width * img.height / img.width)), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=colors))
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(round(1000 * every / real_time_fps)), loop=0, optimize=True)
    print(f"wrote {path}: {len(images)} frames, {Path(path).stat().st_size / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="`random` and/or model_<it>.pt files")
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", type=int, nargs=2, default=[352, 480], metavar=("H", "W"), help="per panel; multiples of 16 for the video")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-envs", type=int, default=256)
    ap.add_argument("--eval-steps", type=int, default=600)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--video", default=None, help="output MP4 (H.264, real-time frame rate)")
    ap.add_argument("--gif", default=None, help="optional small GIF")
    ap.add_argument("--eval-out", default=None, help="evaluation JSON")
    args = ap.parse_args()
    hidden = tuple(args.hidden)

    probe = MujocoG1VecEnv(1, seed=0, num_threads=1)
    skip = max(1, round(1.0 / (args.fps * probe.dt)))
    real_time_fps = 1.0 / (skip * probe.dt)
    probe.close()
    panels, report = [], []
    for ck in args.checkpoints:
        policy = load_policy(ck, hidden)
        if not args.no_eval:
            rep = {"label": label_of(ck), "checkpoint": ck, "by_command": [measure(policy, c, args.eval_steps, args.eval_envs, args.seed) for c in ((0.4, 0.0, 0.0), (0.8, 0.0, 0.0), (0.5, 0.0, 0.4))]}
            report.append(rep)
            print(json.dumps(rep), flush=True)
        if args.video or args.gif:
            panels.append(render_panel(policy, SCHEDULE, skip, tuple(args.size), args.seed, label_of(ck)))
    if report and args.eval_out:
        Path(args.eval_out).write_text(json.dumps(report, indent=2))
    if panels:
        n = min(len(p) for p in panels)
        if args.video:
            import imageio.v2 as imageio

            writer = imageio.get_writer(args.video, fps=real_time_fps, codec="libx264", macro_block_size=16, ffmpeg_params=["-crf", "24", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            for i in range(n):
                writer.append_data(np.concatenate([p[i] for p in panels], axis=1))
            writer.close()
            print(f"wrote {args.video}: {n} frames at {real_time_fps:.2f} fps = {n / real_time_fps:.1f} s, {Path(args.video).stat().st_size / 1e6:.1f} MB")
        if args.gif:
            write_gif([np.concatenate([p[i] for p in panels], axis=1) for i in range(n)], args.gif, real_time_fps)


if __name__ == "__main__":
    main()
