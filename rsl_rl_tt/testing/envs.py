# SPDX-License-Identifier: BSD-3-Clause
"""CPU test environments for the rsl_rl -> TT port (BRIEF Phase 1). Host only; never touches a device.

Three batched, seed-deterministic `rsl_rl.env.VecEnv` implementations with TensorDict observation groups:

  cartpole   batched CartPole (continuous force), early termination, time-out at 500 steps; obs group `policy`
  pendulum   batched Pendulum swing-up (gym dynamics), fixed 200-step episodes, no termination; obs group `policy`
  tracking12 12-actuator double-integrator command tracking, shaped like an Isaac Lab locomotion task: domain-randomisation
             parameters are exposed only in the critic-side group `privileged`, fixed-length time-outs AND early termination,
             `extras["log"]` episode statistics on reset steps

Conventions follow Isaac Lab's `RslRlVecEnvWrapper`: `dones` is `torch.long`, `extras["time_outs"]` is a bool tensor that is true only
for time-limit terminations (a terminated-and-timed-out env counts as terminated), the returned observations are those AFTER the reset
of finished envs, and `extras["log"]` is present only on steps in which some env finished. Each env owns a private
`torch.Generator`, so a run is a pure function of (seed, actions) and does not disturb the global RNG that PPO uses for action noise
and mini-batch shuffling.
"""

from __future__ import annotations

import math
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

__all__ = ["BatchedCartPole", "BatchedPendulum", "Tracking12", "make_env", "ENV_NAMES"]


class _BatchedEnv(VecEnv):
    """Shared episode bookkeeping. Subclasses implement the physics and the observation groups."""

    obs_group_dims: dict[str, int]
    time_out_steps: int
    step_dt: float = 0.02  # seconds per control step (Isaac Lab's `env.unwrapped.step_dt`: upstream scales the RND weight with it)

    @property
    def unwrapped(self):
        return self

    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        device: str = "cpu",
        dones_dtype: torch.dtype = torch.long,
        time_outs_dtype: torch.dtype = torch.bool,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.seed = seed
        self._dones_dtype = dones_dtype
        self._time_outs_dtype = time_outs_dtype
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)
        self.max_episode_length = self.time_out_steps
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cfg = {"env_name": type(self).__name__, "num_envs": num_envs, "seed": seed, "max_episode_length": self.time_out_steps}
        self._episode_sums: dict[str, torch.Tensor] = {}
        self._reset(torch.arange(num_envs, device=device))
        self._obs = self._compute_obs()

    # -- helpers -------------------------------------------------------------------------------------------------------------
    def _uniform(self, shape, low: float, high: float) -> torch.Tensor:
        return (torch.rand(shape, generator=self._gen) * (high - low) + low).to(self.device)

    def _normal(self, shape) -> torch.Tensor:
        return torch.randn(shape, generator=self._gen).to(self.device)

    # -- subclass hooks ------------------------------------------------------------------------------------------------------
    def _reset(self, ids: torch.Tensor) -> None:  # re-initialise the state of envs `ids`
        raise NotImplementedError

    def _physics(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the state; return (rewards [N], terminated [N] bool)."""
        raise NotImplementedError

    def _compute_obs(self) -> TensorDict:
        raise NotImplementedError

    def _episode_log(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {}

    # -- VecEnv API ----------------------------------------------------------------------------------------------------------
    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert actions.shape == (self.num_envs, self.num_actions), actions.shape
        rewards, terminated = self._physics(actions.to(self.device))
        self.episode_length_buf += 1
        time_out = (self.episode_length_buf >= self.time_out_steps) & ~terminated
        done = terminated | time_out
        extras: dict = {"time_outs": time_out.to(self._time_outs_dtype)}
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel() > 0:
            log = self._episode_log(ids)
            log["Episode_Termination/terminated"] = terminated[ids].float().mean()
            log["Episode_Termination/time_out"] = time_out[ids].float().mean()
            extras["log"] = {k if "/" in k else f"/{k}": v for k, v in log.items()}
            self._reset(ids)
            self.episode_length_buf[ids] = 0
        self._obs = self._compute_obs()
        return self._obs, rewards, done.to(self._dones_dtype), extras


# ---------------------------------------------------------------------------------------------------------------------------------
class BatchedCartPole(_BatchedEnv):
    """Classic cart-pole with a continuous force input in [-1, 1] (scaled by 10 N). Reward +1 per step; max return 500."""

    num_actions = 1
    obs_group_dims = {"policy": 4}
    time_out_steps = 500
    obs_groups_cfg = {"actor": ["policy"], "critic": ["policy"]}

    GRAVITY, MASS_CART, MASS_POLE, LENGTH, FORCE_MAG, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    THETA_LIMIT, X_LIMIT = 12 * 2 * math.pi / 360, 2.4

    def _reset(self, ids: torch.Tensor) -> None:
        if not hasattr(self, "_s"):
            self._s = torch.zeros(self.num_envs, 4, device=self.device)
        self._s[ids] = self._uniform((ids.numel(), 4), -0.05, 0.05)

    def _physics(self, actions: torch.Tensor):
        x, x_dot, th, th_dot = self._s.unbind(1)
        force = actions[:, 0].clamp(-1.0, 1.0) * self.FORCE_MAG
        total_mass = self.MASS_CART + self.MASS_POLE
        pm_len = self.MASS_POLE * self.LENGTH
        cos, sin = torch.cos(th), torch.sin(th)
        temp = (force + pm_len * th_dot**2 * sin) / total_mass
        th_acc = (self.GRAVITY * sin - cos * temp) / (self.LENGTH * (4.0 / 3.0 - self.MASS_POLE * cos**2 / total_mass))
        x_acc = temp - pm_len * th_acc * cos / total_mass
        x = x + self.TAU * x_dot
        x_dot = x_dot + self.TAU * x_acc
        th = th + self.TAU * th_dot
        th_dot = th_dot + self.TAU * th_acc
        self._s = torch.stack([x, x_dot, th, th_dot], dim=1)
        terminated = (x.abs() > self.X_LIMIT) | (th.abs() > self.THETA_LIMIT)
        return torch.ones(self.num_envs, device=self.device), terminated

    def _compute_obs(self) -> TensorDict:
        return TensorDict({"policy": self._s.clone()}, batch_size=[self.num_envs], device=self.device)


class BatchedCartPolePO(BatchedCartPole):
    """The cart-pole of `BatchedCartPole` WITHOUT the velocities: the policy sees only the cart position and the pole angle, so a feed-forward policy cannot balance the pole for long, while
    a recurrent one can estimate the velocities from the history (Phase 8.5, `rsl_rl_tt.testing.reference.with_rnn`). Everything else (dynamics, reward, termination) is the same."""

    obs_group_dims = {"policy": 2}

    def _compute_obs(self) -> TensorDict:
        return TensorDict({"policy": self._s[:, [0, 2]].clone()}, batch_size=[self.num_envs], device=self.device)


class BatchedCartPoleImage(BatchedCartPole):
    """The cart-pole with an IMAGE observation (Phase 8.6): the velocities as a 1D group `policy`, and the positions as a 2-channel 8 x 8 image `image` (channel 0: a vertical bar at the column of the cart
    position, channel 1: a vertical bar at the column of the pole angle, both Gaussian across the columns) - a policy has to read the positions off the image. Dynamics, reward, termination as `BatchedCartPole`."""

    obs_group_dims = {"policy": 2, "image": 128}  # 128 = 2 x 8 x 8 (flattened)
    obs_groups_cfg = {"actor": ["policy", "image"], "critic": ["policy", "image"]}
    IMAGE = 8

    def _compute_obs(self) -> TensorDict:
        n, w = self.num_envs, self.IMAGE
        cols = torch.arange(w, dtype=torch.float32).reshape(1, 1, w)
        centre_x = ((self._s[:, 0] / self.X_LIMIT + 1.0) * 0.5 * (w - 1)).reshape(n, 1, 1)
        centre_t = ((self._s[:, 2] / self.THETA_LIMIT + 1.0) * 0.5 * (w - 1)).reshape(n, 1, 1)
        bar = lambda c: torch.exp(-((cols - c) ** 2) / 2.0).expand(n, w, w)  # [n, H, W]: constant along the rows
        image = torch.stack([bar(centre_x), bar(centre_t)], dim=1)  # [n, 2, H, W]
        return TensorDict({"policy": self._s[:, [1, 3]].clone(), "image": image.to(self.device)}, batch_size=[n], device=self.device)


class BatchedPendulum(_BatchedEnv):
    """Gym Pendulum-v1 dynamics, torque in [-1, 1] scaled by 2. Reward = -(angle^2 + 0.1 w^2 + 0.001 u^2); 200 steps, no termination."""

    num_actions = 1
    obs_group_dims = {"policy": 3}
    time_out_steps = 200
    obs_groups_cfg = {"actor": ["policy"], "critic": ["policy"]}

    G, M, L, DT, MAX_SPEED, MAX_TORQUE = 10.0, 1.0, 1.0, 0.05, 8.0, 2.0
    step_dt = 0.05

    def _reset(self, ids: torch.Tensor) -> None:
        if not hasattr(self, "_s"):
            self._s = torch.zeros(self.num_envs, 2, device=self.device)
        self._s[ids, 0] = self._uniform((ids.numel(),), -math.pi, math.pi)
        self._s[ids, 1] = self._uniform((ids.numel(),), -1.0, 1.0)

    def _physics(self, actions: torch.Tensor):
        th, thdot = self._s.unbind(1)
        u = actions[:, 0].clamp(-1.0, 1.0) * self.MAX_TORQUE
        angle = ((th + math.pi) % (2 * math.pi)) - math.pi
        cost = angle**2 + 0.1 * thdot**2 + 0.001 * u**2
        thdot = (thdot + (3 * self.G / (2 * self.L) * torch.sin(th) + 3.0 / (self.M * self.L**2) * u) * self.DT).clamp(-self.MAX_SPEED, self.MAX_SPEED)
        th = th + thdot * self.DT
        self._s = torch.stack([th, thdot], dim=1)
        return -cost, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _compute_obs(self) -> TensorDict:
        th, thdot = self._s.unbind(1)
        return TensorDict({"policy": torch.stack([torch.cos(th), torch.sin(th), thdot], dim=1)}, batch_size=[self.num_envs], device=self.device)


class Tracking12(_BatchedEnv):
    """12 independent double-integrator actuators that must track a position command (a stand-in for a legged-robot joint task).

    qdd = (gain * tau_max * a - damping * qd + bias) / mass, semi-implicit Euler, dt = 0.02. The domain-randomised parameters
    (mass, damping, gain, bias; 4 x 12 values) are drawn per episode and exposed only in the `privileged` group, which the
    critic reads in addition to `policy`. Commands are redrawn every 100 steps. Episodes end by time-out after 300 steps or early
    when any |q - q_cmd| exceeds 3 ("lost tracking").
    Reward per step: exp(-mean(err^2)/0.25) - 0.02 mean((a - a_prev)^2) - 0.005 mean(a^2) - 2 [on termination].
    """

    num_actions = 12
    obs_group_dims = {"policy": 36, "privileged": 48}
    time_out_steps = 300
    obs_groups_cfg = {"actor": ["policy"], "critic": ["policy", "privileged"]}

    DT, TAU_MAX, CMD_PERIOD, FAIL_ERR, OBS_NOISE = 0.02, 4.0, 100, 3.0, 0.01
    N = 12

    def _reset(self, ids: torch.Tensor) -> None:
        n, k = self.num_envs, ids.numel()
        if not hasattr(self, "q"):
            for name in ("q", "qd", "q_cmd", "mass", "damping", "gain", "bias", "last_action"):
                setattr(self, name, torch.zeros(n, self.N, device=self.device))
            for name in ("tracking", "action_rate", "effort", "termination"):
                self._episode_sums[name] = torch.zeros(n, device=self.device)
        self.q[ids] = self._uniform((k, self.N), -0.5, 0.5)
        self.qd[ids] = 0.0
        self.q_cmd[ids] = self._uniform((k, self.N), -1.0, 1.0)
        self.mass[ids] = self._uniform((k, self.N), 0.7, 1.3)
        self.damping[ids] = self._uniform((k, self.N), 0.5, 1.5)
        self.gain[ids] = self._uniform((k, self.N), 0.8, 1.2)
        self.bias[ids] = self._uniform((k, self.N), -0.5, 0.5)
        self.last_action[ids] = 0.0
        for v in self._episode_sums.values():
            v[ids] = 0.0

    def _physics(self, actions: torch.Tensor):
        a = actions.clamp(-1.0, 1.0)
        tau = self.gain * self.TAU_MAX * a
        qdd = (tau - self.damping * self.qd + self.bias) / self.mass
        self.qd = (self.qd + qdd * self.DT).clamp(-10.0, 10.0)
        self.q = (self.q + self.qd * self.DT).clamp(-6.0, 6.0)
        # redraw the command at every CMD_PERIOD-th step of the episode
        redraw = ((self.episode_length_buf + 1) % self.CMD_PERIOD == 0).nonzero(as_tuple=False).flatten()
        if redraw.numel() > 0:
            self.q_cmd[redraw] = self._uniform((redraw.numel(), self.N), -1.0, 1.0)
        err = self.q - self.q_cmd
        terminated = err.abs().amax(1) > self.FAIL_ERR
        r_track = torch.exp(-(err**2).mean(1) / 0.25)  # positive and informative from the start (initial mean err^2 is ~0.4)
        r_rate = -0.02 * ((a - self.last_action) ** 2).mean(1)
        r_effort = -0.005 * (a**2).mean(1)
        r_term = -2.0 * terminated.float()  # without a termination penalty an agent that only collects penalties learns to fail early
        self.last_action = a
        self._episode_sums["tracking"] += r_track
        self._episode_sums["action_rate"] += r_rate
        self._episode_sums["effort"] += r_effort
        self._episode_sums["termination"] += r_term
        return r_track + r_rate + r_effort + r_term, terminated

    def _episode_log(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {f"Episode_Reward/{k}": v[ids].mean() for k, v in self._episode_sums.items()}

    def _compute_obs(self) -> TensorDict:
        policy = torch.cat([self.q - self.q_cmd, self.qd, self.last_action], dim=1)
        policy = policy + self.OBS_NOISE * self._normal(policy.shape)
        privileged = torch.cat([self.mass, self.damping, self.gain, self.bias], dim=1)
        return TensorDict({"policy": policy, "privileged": privileged}, batch_size=[self.num_envs], device=self.device)


ENV_CLASSES = {"cartpole": BatchedCartPole, "pendulum": BatchedPendulum, "tracking12": Tracking12}
ENV_NAMES = tuple(ENV_CLASSES)
RECURRENT_ENV_CLASSES = {"cartpole_po": BatchedCartPolePO}  # environments that need a memory (not part of ENV_NAMES: the golden curves and the v1 suites are defined on the three above)


IMAGE_ENV_CLASSES = {"cartpole_img": BatchedCartPoleImage}  # an environment with an image observation group (Phase 8.6)


def env_class(name: str):
    for registry in (ENV_CLASSES, RECURRENT_ENV_CLASSES, IMAGE_ENV_CLASSES):
        if name in registry:
            return registry[name]
    raise ValueError(f"unknown test env {name!r}; choose one of {ENV_NAMES + tuple(RECURRENT_ENV_CLASSES) + tuple(IMAGE_ENV_CLASSES)}")


def make_env(name: str, num_envs: int, seed: int = 0, device: str = "cpu", **kwargs) -> VecEnv:
    return env_class(name)(num_envs, seed=seed, device=device, **kwargs)
