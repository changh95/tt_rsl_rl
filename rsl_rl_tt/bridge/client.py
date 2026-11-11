# SPDX-License-Identifier: BSD-3-Clause
"""The TT-host side of the bridge (Phase 7, ADR-0003): `RemoteVecEnv`, an rsl_rl `VecEnv` whose environment lives behind a ZeroMQ endpoint.

The unmodified `OnPolicyRunner` uses it like any environment: `get_observations()`, `step(actions)`, `num_envs`, `max_episode_length`, `cfg`, and the settable
`episode_length_buf` (the runner randomizes it at the start of `learn(init_at_random_ep_len=True)`). Conventions are those of Isaac Lab's `RslRlVecEnvWrapper`:
`dones` is `torch.long`, `extras["time_outs"]` is a bool tensor (present only if the environment sends it), `extras["log"]` carries named floats, observations are a
`TensorDict` of float32 groups on the CPU (the runner moves them to the algorithm's device). Nothing is retried silently: a reply that does not arrive within
`timeout_s` raises `TimeoutError` (BRIEF 2-6: a hang is reported, not hidden).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl_tt.bridge import protocol


class RemoteEnvError(RuntimeError):
    """The environment server answered a request with an error (the server-side traceback is in the message)."""


@dataclass
class BridgeStats:
    """Cumulative traffic and round-trip times of the calls made through one client (`step` and the others separately)."""

    calls: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    sent_bytes: int = 0
    received_bytes: int = 0
    step_rtt_s: list[float] = field(default_factory=list)

    def add(self, op: str, dt: float, sent: int, received: int) -> None:
        self.calls[op] = self.calls.get(op, 0) + 1
        self.seconds[op] = self.seconds.get(op, 0.0) + dt
        self.sent_bytes += sent
        self.received_bytes += received
        if op == "step":
            self.step_rtt_s.append(dt)


class RemoteVecEnv(VecEnv):
    def __init__(self, endpoint: str, timeout_s: float = 300.0) -> None:
        import zmq

        self._zmq = zmq
        self.endpoint, self.timeout_s = endpoint, timeout_s
        self._sock = None
        self.stats = BridgeStats()
        self._connect()
        _, arrays, meta = self._call("info")
        if meta.get("protocol", protocol.PROTOCOL_VERSION) != protocol.PROTOCOL_VERSION:
            raise RemoteEnvError(f"the server speaks protocol {meta['protocol']}, this client {protocol.PROTOCOL_VERSION}")
        self.num_envs = int(meta["num_envs"])
        self.num_actions = int(meta["num_actions"])
        self.max_episode_length = torch.from_numpy(arrays["max_episode_length"].copy()) if "max_episode_length" in arrays else int(meta["max_episode_length"])
        self.device = "cpu"
        self.cfg = meta.get("env_cfg") or {}
        self.agent_cfg = meta.get("agent_cfg")  # the server's `agent_cfg.to_dict()` (None if it was not given): train with exactly this configuration
        self.obs_shapes = {g: tuple(s) for g, s in meta["obs_shapes"].items()}

    # -- transport ---------------------------------------------------------------------------------------------------------
    def _connect(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self._zmq.Context.instance().socket(self._zmq.REQ)
        self._sock.setsockopt(self._zmq.LINGER, 0)
        self._sock.connect(self.endpoint)

    def _call(self, op: str, arrays: dict[str, np.ndarray] | None = None, meta: dict | None = None) -> tuple[str, dict[str, np.ndarray], dict]:
        frames = protocol.pack(op, arrays, meta)
        t0 = time.perf_counter()
        self._sock.send_multipart(frames, copy=False)
        if not self._sock.poll(int(self.timeout_s * 1000)):
            self._connect()  # a REQ socket that did not get its reply cannot be used again
            raise TimeoutError(f"no reply to {op!r} from the environment server at {self.endpoint} within {self.timeout_s:.0f} s")
        reply = self._sock.recv_multipart(copy=False)
        dt = time.perf_counter() - t0
        self.stats.add(op, dt, protocol.nbytes(frames), protocol.nbytes(reply))
        rop, rarrays, rmeta = protocol.unpack(reply)
        if rop == "error":
            raise RemoteEnvError(f"{rmeta.get('error')}\n--- server traceback ---\n{rmeta.get('trace')}")
        return rop, rarrays, rmeta

    def _obs(self, arrays: dict[str, np.ndarray]) -> TensorDict:
        return TensorDict({k[4:]: torch.from_numpy(v.copy()) for k, v in arrays.items() if k.startswith("obs/")}, batch_size=[self.num_envs])

    # -- VecEnv ------------------------------------------------------------------------------------------------------------
    def get_observations(self) -> TensorDict:
        _, arrays, _ = self._call("get_observations")
        return self._obs(arrays)

    def reset(self) -> tuple[TensorDict, dict]:
        _, arrays, meta = self._call("reset")
        return self._obs(arrays), ({"log": meta["log"]} if meta.get("log") else {})

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        a = actions.detach().to(torch.float32).cpu().contiguous().numpy()
        _, arrays, meta = self._call("step", {"actions": a})
        rewards = torch.from_numpy(arrays["rewards"].copy())
        dones = torch.from_numpy(arrays["dones"].astype(np.int64))
        extras: dict = {}
        for key, value in arrays.items():
            if key.startswith("extras/"):
                name = key[7:]
                extras[name] = torch.from_numpy(value.astype(bool) if name == "time_outs" else value.copy())
        if meta.get("log"):
            extras["log"] = dict(meta["log"])
        return self._obs(arrays), rewards, dones, extras

    @property
    def episode_length_buf(self) -> torch.Tensor:
        _, arrays, _ = self._call("get_episode_length_buf")
        return torch.from_numpy(arrays["episode_length_buf"].copy())

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self._call("set_episode_length_buf", {"episode_length_buf": value.detach().cpu().to(torch.int64).contiguous().numpy()})

    def close(self) -> None:
        """Ask the server to stop (it closes the environment) and drop the connection."""
        try:
            self._call("close")
        finally:
            if self._sock is not None:
                self._sock.close(linger=0)
                self._sock = None
