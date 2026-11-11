# SPDX-License-Identifier: BSD-3-Clause
"""The environment side of the bridge (Phase 7, ADR-0003): serves any rsl_rl-style vectorized environment over ZeroMQ.

`serve(env, endpoint, agent_cfg=...)` works with the object an Isaac Lab training script hands to the rsl_rl runner (`RslRlVecEnvWrapper`) or with any other
object that has the `rsl_rl.env.VecEnv` interface (`get_observations`, `step`, `num_envs`, `num_actions`, `max_episode_length`, `episode_length_buf`, `cfg`).
It needs only numpy, pyzmq and (duck-typed) torch tensors; it does NOT import rsl_rl or ttnn, so it can run inside an Isaac Lab Python of a different version from
the TT host's. The agent configuration (`agent_cfg.to_dict()` of Isaac Lab) is sent to the client in the `info` reply, so the TT side trains with exactly the
configuration the environment was set up for.

Strict lock-step request / reply (one client); a request that raises on the server is answered with an `error` message and the loop keeps running.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable

import numpy as np

from rsl_rl_tt.bridge import protocol


def _np(x: Any, dtype: str = "float32") -> np.ndarray:
    """A torch tensor (any device / dtype), numpy array or Python scalar/list as a contiguous numpy array of `dtype`."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x).astype(dtype, copy=False))


def _jsonable(obj: Any) -> Any:
    """Best-effort JSON conversion of a configuration object (`to_dict()` of Isaac Lab configclasses, dicts, dataclasses): unknown leaves become their repr."""
    if hasattr(obj, "to_dict") and not isinstance(obj, dict):
        obj = obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "item") and getattr(obj, "ndim", 1) == 0:
        return obj.item()
    return repr(obj)


def _log_scalars(extras: dict) -> dict[str, float]:
    """`extras["log"]` / `extras["episode"]` of Isaac Lab (named scalars, tensors or floats) as plain floats (the rsl_rl logger averages them)."""
    out: dict[str, float] = {}
    for source in ("episode", "log"):
        for key, value in (extras.get(source) or {}).items():
            try:
                out[str(key)] = float(np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value, dtype=np.float64).mean())
            except (TypeError, ValueError):
                continue  # not a number (e.g. a string): not part of the training statistics
    return out


def _obs_arrays(obs: Any) -> dict[str, np.ndarray]:
    return {f"obs/{group}": _np(tensor) for group, tensor in obs.items()}


class EnvServer:
    """Request handlers around one environment object (separate from the socket loop so that tests can call them directly)."""

    def __init__(self, env: Any, agent_cfg: Any = None):
        self.env = env
        self.agent_cfg = agent_cfg
        self.steps = 0

    def info(self) -> list:
        env = self.env
        obs = env.get_observations()
        arrays = {}
        mel = env.max_episode_length
        meta = {
            "num_envs": int(env.num_envs),
            "num_actions": int(env.num_actions),
            "obs_shapes": {g: list(t.shape[1:]) for g, t in obs.items()},
            "agent_cfg": _jsonable(self.agent_cfg) if self.agent_cfg is not None else None,
            "env_cfg": _jsonable(getattr(env, "cfg", None)),
            "server_time": time.time(),
        }
        if hasattr(mel, "shape") and getattr(mel, "ndim", 0) > 0:
            arrays["max_episode_length"] = _np(mel, "int64")
        else:
            meta["max_episode_length"] = int(mel)
        return protocol.pack("info", arrays, meta)

    def handle(self, op: str, arrays: dict[str, np.ndarray], meta: dict) -> list:
        env = self.env
        if op == "info":
            return self.info()
        if op == "get_observations":
            return protocol.pack("obs", _obs_arrays(env.get_observations()))
        if op == "reset":
            result = env.reset()
            obs, extras = result if isinstance(result, tuple) else (result, {})
            return protocol.pack("obs", _obs_arrays(obs), {"log": _log_scalars(extras or {})})
        if op == "step":
            import torch

            actions = torch.from_numpy(np.array(arrays["actions"], dtype=np.float32)).to(env.device)
            obs, rewards, dones, extras = env.step(actions)
            self.steps += 1
            out = _obs_arrays(obs)
            out["rewards"] = _np(rewards)
            out["dones"] = _np(dones, "uint8")
            for key, value in extras.items():
                if key not in ("log", "episode") and hasattr(value, "detach"):  # tensor extras (time_outs, ...)
                    out[f"extras/{key}"] = _np(value, "uint8" if str(value.dtype) in ("torch.bool", "torch.uint8") else "float32")
            return protocol.pack("step", out, {"log": _log_scalars(extras)})
        if op == "get_episode_length_buf":
            return protocol.pack("episode_length_buf", {"episode_length_buf": _np(env.episode_length_buf, "int64")})
        if op == "set_episode_length_buf":
            import torch

            env.episode_length_buf = torch.from_numpy(np.array(arrays["episode_length_buf"], dtype=np.int64)).to(env.episode_length_buf.device)
            return protocol.pack("ok")
        if op == "close":
            if hasattr(env, "close"):
                env.close()
            return protocol.pack("ok")
        raise protocol.ProtocolError(f"unknown operation {op!r}")


def serve(env: Any, endpoint: str, agent_cfg: Any = None, stop: Callable[[], bool] | None = None, poll_ms: int = 200, log: Callable[[str], None] = print) -> EnvServer:
    """Bind `endpoint` (e.g. "tcp://*:5555", "ipc:///tmp/env.sock") and serve requests until a `close` request arrives or `stop()` returns True. Returns the handler."""
    import zmq

    server = EnvServer(env, agent_cfg)
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 1000)
    sock.bind(endpoint)
    log(f"[env server] serving {int(env.num_envs)} environments on {endpoint}")
    try:
        while not (stop and stop()):
            if not sock.poll(poll_ms):
                continue
            frames = sock.recv_multipart(copy=False)
            op = "?"
            try:
                op, arrays, meta = protocol.unpack(frames)
                reply = server.handle(op, arrays, meta)
            except Exception as exc:  # noqa: BLE001 - the client gets the traceback, the server stays up
                reply = protocol.pack("error", meta={"error": repr(exc), "trace": traceback.format_exc()})
                log(f"[env server] request {op!r} failed: {exc!r}")
            sock.send_multipart(reply, copy=False)
            if op == "close":
                break
    finally:
        sock.close()
    log(f"[env server] stopped after {server.steps} steps")
    return server
