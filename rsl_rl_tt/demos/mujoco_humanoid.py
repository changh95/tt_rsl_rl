# SPDX-License-Identifier: BSD-3-Clause
"""A MuJoCo humanoid walking task as an rsl_rl `VecEnv`, simulated on the CPU (demo; the policy is trained on the Tenstorrent tray).

The model and the task are Gymnasium's `Humanoid-v5` (https://gymnasium.farama.org/environments/mujoco/humanoid/, MIT / Apache-2.0 assets, installed with
`pip install "gymnasium[mujoco]"`): 17 torque actuators (ctrl range +-0.4), frame skip 5 at a 3 ms time step (dt = 15 ms), reset noise 1e-2, and
    reward     1.25 * forward velocity of the root + 5.0 while healthy - 0.1 * sum(ctrl^2)      (scaled by `reward_scale`, see below)
    terminated the torso height leaves (1.0, 2.0) m ("fell over");  truncated after `max_episode_steps` (1000 = 15 s)
    observation  qpos[2:] (torso height, orientation, joint angles: 22) and qvel (23): 45 floats, group "policy" (the critic uses the same group)
    action       [-1, 1]^17 (clipped) -> ctrl = 0.4 * action
Differences from Gymnasium's environment, all deliberate: the forward velocity is that of the root body (not of the centre of mass), the 303 contact / inertia features
of the v5 observation are left out, and the reward is multiplied by `reward_scale` (default 0.1): rsl_rl has no reward normalisation and its clipped value loss
(`clip_param` 0.2) moves the critic by at most 0.2 per update, which is too slow for returns of several hundred.

Thousands of environments are stepped at once by MuJoCo's multithreaded `rollout` (no GIL, one `MjData` per THREAD instead of per environment: 0.6 MB each); the state of an
environment is just its `[time, qpos, qvel]` row. Conventions follow Isaac Lab's `RslRlVecEnvWrapper`: `dones` is `torch.long`, `extras["time_outs"]` is a bool tensor that is
true only for time-limit truncations, the returned observations are those AFTER the reset of finished environments, `extras["log"]` holds named episode statistics on steps in
which some environment finished, and the private RNG does not disturb the global one that PPO uses.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

__all__ = ["MujocoHumanoidVecEnv", "humanoid_xml_path"]


def humanoid_xml_path() -> str:
    import gymnasium

    return os.path.join(os.path.dirname(gymnasium.__file__), "envs", "mujoco", "assets", "humanoid.xml")


class MujocoHumanoidVecEnv(VecEnv):
    FRAME_SKIP = 5
    CTRL_LIMIT = 0.4
    FORWARD_WEIGHT, HEALTHY_REWARD, CTRL_COST_WEIGHT = 1.25, 5.0, 0.1
    HEALTHY_Z = (1.0, 2.0)
    RESET_NOISE = 1e-2

    def __init__(self, num_envs: int, seed: int = 0, num_threads: int | None = None, max_episode_steps: int = 1000, reward_scale: float = 0.1, device: str = "cpu") -> None:
        import mujoco
        from mujoco import rollout

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(humanoid_xml_path())
        self._spec = mujoco.mjtState.mjSTATE_FULLPHYSICS.value
        self._nstate = int(mujoco.mj_stateSize(self.model, self._spec))
        self.nq, self.nv, self.num_actions = int(self.model.nq), int(self.model.nv), int(self.model.nu)
        assert self._nstate == 1 + self.nq + self.nv, "the humanoid has no activation / plugin state"
        self.dt = self.FRAME_SKIP * float(self.model.opt.timestep)
        self.num_envs, self.device, self.reward_scale = num_envs, device, float(reward_scale)
        self.max_episode_length = int(max_episode_steps)
        self.num_threads = int(num_threads or min(48, os.cpu_count() or 8))
        self._datas = [mujoco.MjData(self.model) for _ in range(self.num_threads)]
        self._ro = rollout.Rollout(nthread=self.num_threads)
        self._rng = np.random.default_rng(seed)  # private: PPO's noise and shuffles use the global torch generator
        self.cfg = {"env_name": "MujocoHumanoid", "num_envs": num_envs, "seed": seed, "max_episode_length": self.max_episode_length, "reward_scale": reward_scale, "dt": self.dt, "num_threads": self.num_threads}
        self.obs_dim = (self.nq - 2) + self.nv
        self._state = np.zeros((num_envs, self._nstate))
        self._out = np.zeros((num_envs, self.FRAME_SKIP, self._nstate))
        self._steps_t = torch.zeros(num_envs, dtype=torch.long)
        self._steps = self._steps_t.numpy()  # shares memory: `episode_length_buf` is this tensor
        self._x0 = np.zeros(num_envs)  # root x at the start of the episode (forward speed statistics)
        self._sums = {k: np.zeros(num_envs) for k in ("forward", "healthy", "ctrl")}
        self.step_seconds = 0.0  # time spent inside `step` (the CPU simulation), for the demo's time breakdown
        self._reset(np.arange(num_envs))
        self._obs = self._compute_obs()

    # -- VecEnv ------------------------------------------------------------------------------------------------------------------
    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._steps_t

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:  # OnPolicyRunner.learn(init_at_random_ep_len=True) assigns a random tensor
        self._steps_t.copy_(value.to(torch.long))

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": torch.from_numpy(self._obs.copy())}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        import time

        t0 = time.perf_counter()
        n = self.num_envs
        ctrl = self.CTRL_LIMIT * np.clip(actions.detach().cpu().numpy().astype(np.float64), -1.0, 1.0)
        x_before = self._state[:, 1].copy()  # qpos[0]: root x
        self._ro.rollout(self.model, self._datas, self._state, np.repeat(ctrl[:, None, :], self.FRAME_SKIP, axis=1), state=self._out)
        self._state[:] = self._out[:, -1]
        self._steps += 1
        forward = self.FORWARD_WEIGHT * (self._state[:, 1] - x_before) / self.dt
        z = self._state[:, 3]
        healthy = (z > self.HEALTHY_Z[0]) & (z < self.HEALTHY_Z[1])
        ctrl_cost = self.CTRL_COST_WEIGHT * np.square(ctrl).sum(-1)
        terms = {"forward": forward, "healthy": self.HEALTHY_REWARD * healthy, "ctrl": -ctrl_cost}
        reward = self.reward_scale * (terms["forward"] + terms["healthy"] + terms["ctrl"])
        for k, v in terms.items():
            self._sums[k] += self.reward_scale * v
        terminated = ~healthy
        truncated = (self._steps >= self.max_episode_length) & ~terminated  # a fall at the very last step counts as a fall
        done = terminated | truncated
        extras: dict = {"time_outs": torch.from_numpy(truncated.copy())}
        ids = np.nonzero(done)[0]
        if ids.size:
            length = self._steps[ids].astype(np.float64)
            speed = (self._state[ids, 1] - self._x0[ids]) / (length * self.dt)
            extras["log"] = {
                **{f"Episode_Reward/{k}": float(v[ids].mean()) for k, v in self._sums.items()},
                "Metrics/forward_speed": float(speed.mean()),
                "Metrics/episode_length": float(length.mean()),
                "Episode_Termination/fell": float(terminated[ids].mean()),
                "Episode_Termination/time_out": float(truncated[ids].mean()),
            }
            self._reset(ids)
        self._obs = self._compute_obs()
        self.step_seconds += time.perf_counter() - t0
        return (
            TensorDict({"policy": torch.from_numpy(self._obs.copy())}, batch_size=[n]),
            torch.from_numpy(reward.astype(np.float32)),
            torch.from_numpy(done.astype(np.int64)),
            extras,
        )

    def close(self) -> None:
        if getattr(self, "_ro", None) is not None:
            self._ro.close()
            self._ro = None

    # -- internals ---------------------------------------------------------------------------------------------------------------
    def _reset(self, ids: np.ndarray) -> None:
        k, mj = ids.size, self._mj
        qpos0 = np.asarray(self.model.qpos0)
        self._state[ids, 0] = 0.0
        self._state[ids, 1 : 1 + self.nq] = qpos0 + self._rng.uniform(-self.RESET_NOISE, self.RESET_NOISE, (k, self.nq))
        self._state[ids, 1 + self.nq :] = self._rng.uniform(-self.RESET_NOISE, self.RESET_NOISE, (k, self.nv))
        self._steps[ids] = 0
        self._x0[ids] = self._state[ids, 1]
        for v in self._sums.values():
            v[ids] = 0.0

    def _compute_obs(self) -> np.ndarray:
        return self._state[:, 3:].astype(np.float32)  # qpos[2:] and qvel (everything after time, x and y)
