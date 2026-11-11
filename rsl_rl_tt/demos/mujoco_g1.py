# SPDX-License-Identifier: BSD-3-Clause
"""Unitree G1 velocity-command walking in MuJoCo as an rsl_rl `VecEnv`, simulated on the CPU (demo; the policy is trained on the Tenstorrent tray).

Robot: the MJX variant of the Unitree G1 from MuJoCo Menagerie (`scene_mjx.xml`, pinned commit, BSD-3-Clause, (c) Unitree Robotics; `scripts/fetch_g1_model.sh`): 29 position
actuators (hip kp 75 / ankle kp 20, kd 2, torque limits), hand-designed collision geoms, solver tuned for speed (dt 4 ms, 5 iterations), `knees_bent` keyframe.

Task (written for this demo with the usual legged-robot recipe of legged_gym / Isaac Lab / unitree_rl_gym; it is NOT a copy of any of them): follow a velocity command
(vx, vy, yaw rate) in the body frame on flat ground.
    action       12 leg joints: position target = knees_bent pose + `action_scale` (0.25) x action; waist and arms are held at the keyframe pose by their PD actuators
    control      20 ms (5 substeps of 4 ms)
    observation  policy (47): gyro x 0.25, gravity in the body frame, command x (2, 2, 0.25), leg joint positions minus the keyframe, leg joint velocities x 0.05,
                 last action, sin / cos of a 0.8 s gait phase.   privileged (8, critic only): body-frame linear velocity x 2, pelvis height, foot heights, foot contacts
    reward       dt x [ 1.5 tracking(v_xy) + 0.75 tracking(yaw rate)          (exp(-error^2 / 0.25))
                        - 2 vz^2 - 0.05 |w_xy|^2 - 2 |gravity_xy|^2 - 5 (pelvis height - 0.72)^2
                        - 0.01 |action change|^2 - 2e-5 |leg torque|^2 - 1e-7 |leg joint acceleration|^2 - 0.5 (hip roll / yaw deviation)^2
                        + 1.0 feet air time above 0.25 s (on touch-down) - 0.1 sliding of the feet in contact
                        + 1.5 foot height following the gait phase (exp(-error^2 / 0.01): 0 m in stance, a 8 cm half-sine in the swing of the 0.8 s period)
                        + 0.5 contact schedule (feet in contact exactly in their stance phase: alternating steps) 
                        + 0.2 alive ]   and  - 2 on a fall (not multiplied by dt)
    termination  pelvis below 0.45 m or tilted more than 60 degrees;  time-out after 1000 steps (20 s)
    commands     vx U(0.2, 0.8) m/s, vy U(-0.2, 0.2) m/s, yaw rate U(-0.4, 0.4) rad/s, resampled every 5 s
No domain randomisation, no pushes, no observation noise: this is a simulation-only demo. Thousands of environments are stepped by MuJoCo's multithreaded `rollout`
(one `MjData` per thread; the state of an environment is its `[time, qpos, qvel]` row; the feet's heights come from two `framepos` sensors added with `MjSpec`).
Conventions follow Isaac Lab's `RslRlVecEnvWrapper`: `dones` is `torch.long`, `extras["time_outs"]` is a bool tensor that is true only for time-limit truncations, the returned
observations are those AFTER the reset of finished environments, `extras["log"]` holds named episode statistics, the private RNG does not disturb the global one.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

__all__ = ["MujocoG1VecEnv", "g1_scene_path"]

REPO = Path(__file__).resolve().parents[2]


def g1_scene_path() -> str:
    """`scene_mjx.xml` of the fetched Menagerie model (`$G1_MODEL_DIR` or .run/assets/mujoco_menagerie/unitree_g1; `scripts/fetch_g1_model.sh` puts it there)."""
    base = Path(os.environ.get("G1_MODEL_DIR", REPO / ".run" / "assets" / "mujoco_menagerie" / "unitree_g1"))
    path = base / "scene_mjx.xml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/fetch_g1_model.sh (or set G1_MODEL_DIR)")
    return str(path)


def build_model():
    """The G1 scene with two foot-position sensors appended (the heights of the feet are not part of the physics state)."""
    import mujoco

    spec = mujoco.MjSpec.from_file(g1_scene_path())
    for side in ("left", "right"):
        spec.add_sensor(name=f"{side}_foot_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{side}_foot")
    m = spec.compile()
    # Raise the PD gains of the legs and the waist to the range used for legged-robot RL on the G1. The Menagerie values (hip / knee 75, ankle 20) cap the ankle torque at
    # ~5 N*m (kp x the 0.26 rad range of the ankle roll), far below the ~30 N*m that a 33 kg robot needs to stand on one foot: with them a policy cannot learn to step. The
    # torque limits of the model (`forcerange`) are unchanged.
    kp = np.array([100, 100, 100, 150, 40, 40] * 2 + [200, 200, 200] + [75] * 4 + [20] * 3 + [75] * 4 + [20] * 3, dtype=np.float64)
    kd = np.array([2, 2, 2, 4, 2, 2] * 2 + [5, 5, 5] + [2] * 14, dtype=np.float64)
    m.actuator_gainprm[:, 0] = kp
    m.actuator_biasprm[:, 1] = -kp
    m.actuator_biasprm[:, 2] = -kd
    return m


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """[N, 4] unit quaternions (w, x, y, z) -> [N, 3, 3] rotation matrices body -> world."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3))
    r[:, 0, 0], r[:, 0, 1], r[:, 0, 2] = 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)
    r[:, 1, 0], r[:, 1, 1], r[:, 1, 2] = 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)
    r[:, 2, 0], r[:, 2, 1], r[:, 2, 2] = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
    return r


class MujocoG1VecEnv(VecEnv):
    FRAME_SKIP = 5
    NUM_LEG = 12
    ACTION_SCALE = 0.25
    ACTION_CLIP = 3.0
    GAIT_PERIOD, STANCE_FRACTION = 0.8, 0.55
    FOOT_CONTACT_Z = 0.02
    AIR_TIME_THRESHOLD = 0.25  # the swing of the 0.8 s gait cycle lasts 0.36 s: a threshold of 0.4 s could never be reached
    SWING_HEIGHT = 0.08
    FOOT_Z0 = -0.0035  # height of the foot site when the foot stands on the floor
    TARGET_HEIGHT = 0.72
    TERMINATE_HEIGHT, TERMINATE_GRAVITY_Z = 0.45, -0.5
    COMMAND_RESAMPLE_STEPS = 250
    CMD_RANGE = {"vx": (0.2, 0.8), "vy": (-0.2, 0.2), "wz": (-0.4, 0.4)}
    WEIGHTS = {
        "track_lin_vel": 1.5, "track_ang_vel": 0.75, "lin_vel_z": -2.0, "ang_vel_xy": -0.05, "orientation": -2.0, "base_height": -5.0, "action_rate": -0.01, "torques": -2.0e-5,
        "dof_acc": -1.0e-7, "hip_deviation": -0.5, "feet_air_time": 1.0, "feet_slip": -0.1, "feet_phase": 1.5, "contact_schedule": 0.5, "alive": 0.2, "termination": -2.0,
    }

    def __init__(self, num_envs: int, seed: int = 0, num_threads: int | None = None, max_episode_steps: int = 1000, device: str = "cpu") -> None:
        import mujoco
        from mujoco import rollout

        self._mj = mujoco
        self.model = m = build_model()
        self._spec = mujoco.mjtState.mjSTATE_FULLPHYSICS.value
        self._nstate = int(mujoco.mj_stateSize(m, self._spec))
        self.nq, self.nv, self.nu = int(m.nq), int(m.nv), int(m.nu)
        assert self._nstate == 1 + self.nq + self.nv and self.nu == 29
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
        self.default_q = np.asarray(m.key_qpos[key][7:], dtype=np.float64)  # 29 joint angles
        self.default_state = np.zeros(self._nstate)
        self.default_state[1 : 1 + self.nq] = m.key_qpos[key]
        self.ctrl_low, self.ctrl_high = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.kp, self.kd = m.actuator_gainprm[:, 0].copy(), -m.actuator_biasprm[:, 2].copy()
        self.force_low, self.force_high = m.actuator_forcerange[:, 0].copy(), m.actuator_forcerange[:, 1].copy()
        self._foot_adr = [int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"{s}_foot_pos")]) for s in ("left", "right")]
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        self._hip_idx = np.array([names.index(n) - 1 for n in ("left_hip_roll_joint", "left_hip_yaw_joint", "right_hip_roll_joint", "right_hip_yaw_joint")])
        self.dt = self.FRAME_SKIP * float(m.opt.timestep)
        self.num_envs, self.device = num_envs, device
        self.num_actions = self.NUM_LEG
        self.max_episode_length = int(max_episode_steps)
        self.num_threads = int(num_threads or min(48, os.cpu_count() or 8))
        self._datas = [mujoco.MjData(m) for _ in range(self.num_threads)]
        self._ro = rollout.Rollout(nthread=self.num_threads)
        self._rng = np.random.default_rng(seed)  # private: PPO's noise and shuffles use the global torch generator
        self.cfg = {"env_name": "MujocoG1", "num_envs": num_envs, "seed": seed, "max_episode_length": self.max_episode_length, "dt": self.dt, "weights": dict(self.WEIGHTS),
                    "action_scale": self.ACTION_SCALE, "num_threads": self.num_threads}
        n = num_envs
        self._state = np.zeros((n, self._nstate))
        self._out = np.zeros((n, self.FRAME_SKIP, self._nstate))
        self._sd = np.zeros((n, self.FRAME_SKIP, int(m.nsensordata)))
        self._steps_t = torch.zeros(n, dtype=torch.long)
        self._steps = self._steps_t.numpy()  # shares memory: `episode_length_buf` is this tensor
        self._cmd = np.zeros((n, 3))
        self.fixed_command: tuple[float, float, float] | None = None  # set to hold one command (evaluation / video)
        self._last_action = np.zeros((n, self.NUM_LEG))
        self._last_dq = np.zeros((n, self.NUM_LEG))
        self._air_time = np.zeros((n, 2))
        self._last_contact = np.zeros((n, 2), dtype=bool)
        self._last_foot_xy = np.zeros((n, 2, 2))
        self._sums = {k: np.zeros(n) for k in self.WEIGHTS}
        self._vx_sum = np.zeros(n)
        self.step_seconds = 0.0  # time spent inside `step` (the CPU simulation), for the demo's time breakdown
        self._reset(np.arange(n))
        self._obs, self._priv = self._compute_obs()

    # -- VecEnv ------------------------------------------------------------------------------------------------------------------
    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._steps_t

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:  # OnPolicyRunner.learn(init_at_random_ep_len=True) assigns a random tensor
        self._steps_t.copy_(value.to(torch.long))

    def _td(self) -> TensorDict:
        return TensorDict({"policy": torch.from_numpy(self._obs.copy()), "privileged": torch.from_numpy(self._priv.copy())}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        return self._td()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        import time

        t0 = time.perf_counter()
        n, dt = self.num_envs, self.dt
        action = np.clip(actions.detach().cpu().numpy().astype(np.float64), -self.ACTION_CLIP, self.ACTION_CLIP)
        target = np.tile(self.default_q, (n, 1))
        target[:, : self.NUM_LEG] += self.ACTION_SCALE * action
        ctrl = np.clip(target, self.ctrl_low, self.ctrl_high)
        self._ro.rollout(self.model, self._datas, self._state, np.repeat(ctrl[:, None, :], self.FRAME_SKIP, axis=1), state=self._out, sensordata=self._sd)
        self._state[:] = self._out[:, -1]
        self._steps += 1
        s = self._state
        q, dq = s[:, 1 + 7 : 1 + self.nq], s[:, 1 + self.nq + 6 :]
        quat, v_world, w_body = s[:, 4:8], s[:, 1 + self.nq : 1 + self.nq + 3], s[:, 1 + self.nq + 3 : 1 + self.nq + 6]
        rot = quat_to_rotation(quat)
        v_body = np.einsum("nji,nj->ni", rot, v_world)
        gravity = -np.stack([2 * (quat[:, 1] * quat[:, 3] - quat[:, 0] * quat[:, 2]), 2 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1]), 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)], axis=1)
        height = s[:, 3]
        foot = np.stack([self._sd[:, -1, a : a + 3] for a in self._foot_adr], axis=1)  # [n, 2, 3]
        contact = foot[:, :, 2] < self.FOOT_CONTACT_Z

        # --- reward terms -----------------------------------------------------------------------------------------------------------
        cmd = self._cmd
        lin_err = np.square(cmd[:, :2] - v_body[:, :2]).sum(-1)
        ang_err = np.square(cmd[:, 2] - w_body[:, 2])
        leg_q, leg_dq, leg_ctrl = q[:, : self.NUM_LEG], dq[:, : self.NUM_LEG], ctrl[:, : self.NUM_LEG]
        tau = np.clip(self.kp[: self.NUM_LEG] * (leg_ctrl - leg_q) - self.kd[: self.NUM_LEG] * leg_dq, self.force_low[: self.NUM_LEG], self.force_high[: self.NUM_LEG])
        first_contact = contact & ~self._last_contact
        self._air_time += dt
        air_reward = ((self._air_time - self.AIR_TIME_THRESHOLD) * first_contact).sum(-1) * (np.linalg.norm(cmd[:, :2], axis=1) > 0.1)
        self._air_time[contact] = 0.0
        foot_xy = foot[:, :, :2]
        foot_speed2 = np.square((foot_xy - self._last_foot_xy) / dt).sum(-1)
        phase = (self._steps * dt / self.GAIT_PERIOD) % 1.0
        leg_phase = np.stack([phase, (phase + 0.5) % 1.0], axis=1)
        stance = leg_phase < self.STANCE_FRACTION
        terms = {
            "track_lin_vel": np.exp(-lin_err / 0.25), "track_ang_vel": np.exp(-ang_err / 0.25),
            "lin_vel_z": np.square(v_body[:, 2]), "ang_vel_xy": np.square(w_body[:, :2]).sum(-1), "orientation": np.square(gravity[:, :2]).sum(-1),
            "base_height": np.square(height - self.TARGET_HEIGHT), "action_rate": np.square(action - self._last_action).sum(-1), "torques": np.square(tau).sum(-1),
            "dof_acc": np.square((leg_dq - self._last_dq) / dt).sum(-1), "hip_deviation": np.square(q[:, self._hip_idx] - self.default_q[self._hip_idx]).sum(-1),
            "feet_air_time": air_reward, "feet_slip": (contact * foot_speed2).sum(-1), "feet_phase": np.exp(-np.square(foot[:, :, 2] - (self.FOOT_Z0 + self._swing_profile(leg_phase))).sum(-1) / 0.01), "contact_schedule": (contact == stance).mean(-1), "alive": np.ones(n),
        }
        terminated = (height < self.TERMINATE_HEIGHT) | (gravity[:, 2] > self.TERMINATE_GRAVITY_Z)
        terms["termination"] = terminated.astype(np.float64)
        reward = np.zeros(n)
        for k, v in terms.items():
            weighted = self.WEIGHTS[k] * v * (1.0 if k == "termination" else dt)  # the fall penalty is a one-off cost, not a rate
            reward += weighted
            self._sums[k] += weighted
        self._vx_sum += v_body[:, 0]
        self._last_action, self._last_dq, self._last_contact, self._last_foot_xy = action, leg_dq.copy(), contact, foot_xy.copy()

        truncated = (self._steps >= self.max_episode_length) & ~terminated  # a fall at the very last step counts as a fall
        done = terminated | truncated
        extras: dict = {"time_outs": torch.from_numpy(truncated.copy())}
        ids = np.nonzero(done)[0]
        if ids.size:
            length = self._steps[ids].astype(np.float64)
            extras["log"] = {
                **{f"Episode_Reward/{k}": float(v[ids].mean()) for k, v in self._sums.items()},
                "Metrics/forward_speed": float((self._vx_sum[ids] / length).mean()),
                "Metrics/command_vx": float(self._cmd[ids, 0].mean()),
                "Metrics/episode_length": float(length.mean()),
                "Episode_Termination/fell": float(terminated[ids].mean()),
                "Episode_Termination/time_out": float(truncated[ids].mean()),
            }
        resample = np.nonzero(~done & (self._steps % self.COMMAND_RESAMPLE_STEPS == 0))[0]
        if resample.size:
            self._sample_commands(resample)
        if ids.size:
            self._reset(ids)
        self._obs, self._priv = self._compute_obs()
        self.step_seconds += time.perf_counter() - t0
        return self._td(), torch.from_numpy(reward.astype(np.float32)), torch.from_numpy(done.astype(np.int64)), extras

    def close(self) -> None:
        if getattr(self, "_ro", None) is not None:
            self._ro.close()
            self._ro = None

    # -- internals ---------------------------------------------------------------------------------------------------------------
    def _swing_profile(self, leg_phase: np.ndarray) -> np.ndarray:
        """Target foot height above the floor for the gait phase of each leg: 0 during the stance, a half-sine of height `SWING_HEIGHT` during the swing."""
        progress = np.clip((leg_phase - self.STANCE_FRACTION) / (1.0 - self.STANCE_FRACTION), 0.0, 1.0)
        return np.where(leg_phase >= self.STANCE_FRACTION, self.SWING_HEIGHT * np.sin(np.pi * progress), 0.0)

    def _sample_commands(self, ids: np.ndarray) -> None:
        if self.fixed_command is not None:
            self._cmd[ids] = np.asarray(self.fixed_command)
            return
        k = ids.size
        self._cmd[ids, 0] = self._rng.uniform(*self.CMD_RANGE["vx"], k)
        self._cmd[ids, 1] = self._rng.uniform(*self.CMD_RANGE["vy"], k)
        self._cmd[ids, 2] = self._rng.uniform(*self.CMD_RANGE["wz"], k)

    def set_command(self, command: tuple[float, float, float]) -> None:
        """Hold one command for every environment from now on (evaluation, video)."""
        self.fixed_command = tuple(command)
        self._cmd[:] = np.asarray(command)

    def _reset(self, ids: np.ndarray) -> None:
        k = ids.size
        self._state[ids] = self.default_state
        self._state[ids, 1 + 7 : 1 + self.nq] += self._rng.uniform(-0.05, 0.05, (k, self.nq - 7))  # joint angle noise
        self._state[ids, 1 + self.nq :] = self._rng.uniform(-0.05, 0.05, (k, self.nv))
        self._steps[ids] = 0
        self._sample_commands(ids)
        self._last_action[ids] = 0.0
        self._last_dq[ids] = 0.0
        self._air_time[ids] = 0.0
        self._last_contact[ids] = True
        self._last_foot_xy[ids] = self._foot_xy_at_default
        for v in self._sums.values():
            v[ids] = 0.0
        self._vx_sum[ids] = 0.0

    @property
    def _foot_xy_at_default(self) -> np.ndarray:
        if not hasattr(self, "_cached_foot_xy"):
            d = self._mj.MjData(self.model)
            self._mj.mj_setState(self.model, d, self.default_state, self._spec)
            self._mj.mj_forward(self.model, d)
            self._cached_foot_xy = np.stack([d.sensordata[a : a + 2] for a in self._foot_adr])
        return self._cached_foot_xy

    def _compute_obs(self) -> tuple[np.ndarray, np.ndarray]:
        s = self._state
        q = s[:, 1 + 7 : 1 + 7 + self.NUM_LEG]
        dq = s[:, 1 + self.nq + 6 : 1 + self.nq + 6 + self.NUM_LEG]
        quat, v_world, w_body = s[:, 4:8], s[:, 1 + self.nq : 1 + self.nq + 3], s[:, 1 + self.nq + 3 : 1 + self.nq + 6]
        rot = quat_to_rotation(quat)
        v_body = np.einsum("nji,nj->ni", rot, v_world)
        gravity = -np.stack([2 * (quat[:, 1] * quat[:, 3] - quat[:, 0] * quat[:, 2]), 2 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1]), 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)], axis=1)
        phase = (self._steps * self.dt / self.GAIT_PERIOD) % 1.0
        policy = np.concatenate(
            [0.25 * w_body, gravity, self._cmd * np.array([2.0, 2.0, 0.25]), q - self.default_q[: self.NUM_LEG], 0.05 * dq, self._last_action, np.sin(2 * np.pi * phase)[:, None], np.cos(2 * np.pi * phase)[:, None]], axis=1
        )
        # foot heights / contacts of the CURRENT state need a sensor read, which the rollout returned for the last substep of the previous call; after a reset the default stance applies
        foot_z = np.where(self._just_reset_mask()[:, None], -0.0035, self._foot_z_cache)
        privileged = np.concatenate([2.0 * v_body, s[:, 3:4], foot_z, (foot_z < self.FOOT_CONTACT_Z).astype(np.float64)], axis=1)
        return policy.astype(np.float32), privileged.astype(np.float32)

    def _just_reset_mask(self) -> np.ndarray:
        return self._steps == 0

    @property
    def _foot_z_cache(self) -> np.ndarray:
        return np.stack([self._sd[:, -1, a + 2] for a in self._foot_adr], axis=1)
