# SPDX-License-Identifier: BSD-3-Clause
"""Render what a trained humanoid policy does: a GIF (several checkpoints side by side) and an evaluation of each (episode length, forward speed, falls).

The checkpoints are upstream-format `model_<it>.pt` files (trained on the TT tray or on the CPU); they are loaded into the unmodified upstream `OnPolicyRunner` and the
policy is run deterministically (distribution mean). `random` renders an untrained policy (the actor of iteration 0).

    MUJOCO_GL=egl python scripts/render_humanoid_demo.py --checkpoints random .run/humanoid/tt_1chip/model_100.pt .run/humanoid/tt_1chip/model_599.pt \
        --out docs/porting/demo_humanoid/walk.gif --seconds 10

Headless rendering uses EGL (software Mesa); the GIF is written with Pillow (no ffmpeg needed).
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
from rsl_rl_tt.demos.humanoid_cfg import humanoid_train_cfg  # noqa: E402
from rsl_rl_tt.demos.mujoco_humanoid import MujocoHumanoidVecEnv  # noqa: E402


def load_policy(checkpoint: str, env: MujocoHumanoidVecEnv, hidden: tuple[int, ...]):
    cfg = humanoid_train_cfg(hidden=hidden, csv_log=False)
    torch.manual_seed(0)
    runner = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")
    if checkpoint != "random":
        runner.load(checkpoint, load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False})
    return runner.get_inference_policy(device="cpu")


def label_of(checkpoint: str) -> str:
    if checkpoint == "random":
        return "untrained"
    m = re.search(r"model_(\d+)", checkpoint)
    return f"iteration {int(m.group(1)) + 1}" if m else Path(checkpoint).stem


def evaluate(policy, hidden, checkpoint: str, steps: int, envs: int, seed: int) -> dict:
    """Run `envs` environments for `steps` steps with the deterministic policy and summarise the finished episodes."""
    env = MujocoHumanoidVecEnv(envs, seed=seed, num_threads=8)
    try:
        obs = env.get_observations()
        lengths, speeds, fell, finished = [], [], 0, 0
        with torch.inference_mode():
            for _ in range(steps):
                obs, _, dones, extras = env.step(policy(obs))
                log = extras.get("log")
                if log is not None:
                    k = int(dones.sum())
                    lengths += [log["Metrics/episode_length"]] * k
                    speeds += [log["Metrics/forward_speed"]] * k
                    fell += round(log["Episode_Termination/fell"] * k)
                    finished += k
        return {
            "checkpoint": checkpoint, "episodes_finished": finished, "mean_episode_length_steps": float(np.mean(lengths)) if lengths else None,
            "mean_forward_speed_m_s": float(np.mean(speeds)) if speeds else None, "fell_fraction": fell / finished if finished else None,
            "survived_steps_of_max": f"{env.max_episode_length} max",
        }
    finally:
        env.close()


def render_panel(policy, seconds: float, skip: int, size: tuple[int, int], seed: int, label: str) -> list[np.ndarray]:
    """Frames of ONE humanoid (a single-environment run; after a fall it starts again, like in training)."""
    import mujoco
    from PIL import Image, ImageDraw

    env = MujocoHumanoidVecEnv(1, seed=seed, num_threads=1)
    floor = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    env.model.geom_size[floor, :2] = 400.0  # only the DRAWN size of the plane (a plane collides everywhere): the fast policy runs off the 20 m of the model's floor
    h, w = size
    renderer = mujoco.Renderer(env.model, h, w)
    data = mujoco.MjData(env.model)
    cam = mujoco.MjvCamera()
    cam.type, cam.trackbodyid, cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_TRACKING, 1, 4.5, 90.0, -8.0
    frames, obs = [], env.get_observations()
    steps = int(seconds / env.dt)
    falls = 0
    try:
        with torch.inference_mode():
            for t in range(steps):
                obs, _, dones, _ = env.step(policy(obs))
                falls += int(dones[0])
                if t % skip == 0:
                    mujoco.mj_setState(env.model, data, env._state[0].copy(), env._spec)
                    mujoco.mj_forward(env.model, data)
                    renderer.update_scene(data, camera=cam)
                    img = Image.fromarray(renderer.render())
                    ImageDraw.Draw(img).text((6, 4), f"{label}   t = {t * env.dt:4.1f} s   x = {env._state[0, 1]:5.1f} m   falls: {falls}", fill=(255, 255, 255))
                    frames.append(np.asarray(img))
    finally:
        renderer.close()
        env.close()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="`random` and/or model_<it>.pt files")
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--size", type=int, nargs=2, default=[240, 320], metavar=("H", "W"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-envs", type=int, default=256)
    ap.add_argument("--eval-steps", type=int, default=1500)
    ap.add_argument("--out", required=True, help="output GIF; the evaluation goes to <out>.json")
    args = ap.parse_args()

    from PIL import Image

    hidden = tuple(args.hidden)
    probe = MujocoHumanoidVecEnv(1, seed=0, num_threads=1)
    skip = max(1, round(1.0 / (args.fps * probe.dt)))
    duration_ms = int(round(1000 * skip * probe.dt))
    probe.close()
    panels, report = [], []
    for ck in args.checkpoints:
        env = MujocoHumanoidVecEnv(args.eval_envs, seed=args.seed, num_threads=8)
        policy = load_policy(ck, env, hidden)
        env.close()
        report.append({"label": label_of(ck), **evaluate(policy, hidden, ck, args.eval_steps, args.eval_envs, args.seed)})
        print(report[-1], flush=True)
        panels.append(render_panel(load_policy(ck, MujocoHumanoidVecEnv(1, seed=0, num_threads=1), hidden), args.seconds, skip, tuple(args.size), args.seed, label_of(ck)))
    n = min(len(p) for p in panels)
    frames = [Image.fromarray(np.concatenate([p[i] for p in panels], axis=1)).convert("P", palette=Image.ADAPTIVE, colors=64) for i in range(n)]
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=True)
    Path(f"{args.out}.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}: {n} frames, {Path(args.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
