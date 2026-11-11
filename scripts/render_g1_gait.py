# SPDX-License-Identifier: BSD-3-Clause
"""A close-up, slow-motion video of the trained Unitree G1 walking: two views of ONE robot (side and oblique) and the foot contacts, to see the gait itself.

The policy walks at a held command for `--warmup` seconds (not recorded, so that the gait has settled), then every 20 ms control step of `--seconds` seconds is rendered and the video is
played at HALF speed (25 fps for 50 steps per second). The two squares show the feet: green while a foot touches the floor (foot height < 2 cm), grey while it is in the air.

    MUJOCO_GL=egl python scripts/render_g1_gait.py --checkpoint .run/g1/tt_8chips/model_1499.pt --command 0.8 --out docs/porting/demo_g1/gait.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.demos.g1_render import label_of, load_policy  # noqa: E402
from rsl_rl_tt.demos.mujoco_g1 import MujocoG1VecEnv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--command", type=float, default=0.8, help="forward speed command (m/s)")
    ap.add_argument("--warmup", type=float, default=4.0)
    ap.add_argument("--seconds", type=float, default=6.0, help="seconds of simulation recorded (the video lasts twice as long: half speed)")
    ap.add_argument("--size", type=int, nargs=2, default=[352, 480], metavar=("H", "W"), help="per view; multiples of 16")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    policy = load_policy(args.checkpoint, tuple(args.hidden))
    env = MujocoG1VecEnv(1, seed=args.seed, num_threads=1)
    env.set_command((args.command, 0.0, 0.0))
    env.model.vis.headlight.ambient[:] = 0.45
    env.model.vis.headlight.diffuse[:] = 0.7
    h, w = args.size
    font = ImageFont.load_default(size=max(10, h // 20))
    renderer = mujoco.Renderer(env.model, h, w)
    data = mujoco.MjData(env.model)
    cams = []
    for azimuth in (90.0, 140.0):  # the side view and an oblique front view
        cam = mujoco.MjvCamera()
        cam.type, cam.trackbodyid, cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_TRACKING, 1, 2.6, azimuth, -6.0
        cams.append(cam)

    obs = env.get_observations()
    writer = imageio.get_writer(args.out, fps=0.5 / env.dt, codec="libx264", macro_block_size=16, ffmpeg_params=["-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    frames = 0
    try:
        with torch.inference_mode():
            for t in range(int((args.warmup + args.seconds) / env.dt)):
                obs, _, dones, _ = env.step(policy(obs))
                if t < int(args.warmup / env.dt):
                    continue
                mujoco.mj_setState(env.model, data, env._state[0].copy(), env._spec)
                mujoco.mj_forward(env.model, data)
                contact = [float(env._sd[0, -1, a + 2]) < env.FOOT_CONTACT_Z for a in env._foot_adr]
                views = []
                for cam in cams:
                    renderer.update_scene(data, camera=cam)
                    img = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(img)
                    draw.text((6, 4), f"{label_of(args.checkpoint)}    command {args.command:.1f} m/s    slow motion x0.5", fill=(255, 255, 255), font=font)
                    draw.text((6, 8 + font.size), f"t = {(t - int(args.warmup / env.dt)) * env.dt:4.2f} s", fill=(255, 230, 120), font=font)
                    for k, (name, on) in enumerate(zip(("L", "R"), contact)):
                        x0 = 8 + k * (font.size * 2 + 10)
                        draw.rectangle([x0, h - font.size * 2 - 8, x0 + font.size * 2, h - 8], fill=(60, 200, 90) if on else (70, 70, 70), outline=(255, 255, 255))
                        draw.text((x0 + font.size * 0.6, h - font.size * 2 + 2), name, fill=(0, 0, 0) if on else (200, 200, 200), font=font)
                    views.append(np.asarray(img))
                writer.append_data(np.concatenate(views, axis=1))
                frames += 1
    finally:
        writer.close()
        renderer.close()
        env.close()
    print(f"wrote {args.out}: {frames} frames = {frames * env.dt:.1f} s of simulation, played at half speed = {2 * frames * env.dt:.1f} s, {Path(args.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
