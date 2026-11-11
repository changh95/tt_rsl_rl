# SPDX-License-Identifier: BSD-3-Clause
"""Wire format of the environment bridge (Phase 7, ADR-0003): ZeroMQ multipart messages, a JSON header plus raw array buffers.

    frame 0      JSON header  {"v": 1, "op": "<name>", "arrays": [{"n": name, "d": dtype, "s": [shape]}, ...], "meta": {...}}
    frame 1..k   the arrays, in the order of the header, C-contiguous little-endian raw bytes (no copy on send, no parsing on receive)

Only numpy and the standard library are imported here, so the server can run inside an Isaac Lab Python that is a different version from the one of the TT host
(no pickle, no torch / tensordict types on the wire). Operations (client -> server, one reply each, strict request / reply):

    info                     -> meta: num_envs, num_actions, max_episode_length (int; or the array "max_episode_length"), obs shapes, agent_cfg, env_cfg, protocol version
    get_observations         -> arrays "obs/<group>"
    reset                    -> arrays "obs/<group>", meta {"log": {...}}
    step      "actions"      -> arrays "obs/<group>", "rewards" float32 [N], "dones" uint8 [N], optional "extras/time_outs" uint8 [N] (and other tensor extras),
                                meta {"log": {name: float}}
    get_episode_length_buf   -> array "episode_length_buf" int64 [N]
    set_episode_length_buf   "episode_length_buf" int64 [N]
    close
    (any failure on the server side is answered with op "error", meta {"error", "trace"})
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1
DTYPES = {"float32": np.float32, "float64": np.float64, "int64": np.int64, "int32": np.int32, "uint8": np.uint8}


class ProtocolError(RuntimeError):
    """A message that does not follow the wire format."""


def pack(op: str, arrays: dict[str, np.ndarray] | None = None, meta: dict[str, Any] | None = None) -> list:
    """Frames of one message: the JSON header followed by the arrays' buffers (memoryviews, i.e. zero-copy for `socket.send_multipart(copy=False)`)."""
    arrays = arrays or {}
    desc, buffers = [], []
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype.name not in DTYPES:
            raise ProtocolError(f"array {name!r} has dtype {arr.dtype.name}; the bridge carries {sorted(DTYPES)}")
        if arr.dtype.byteorder == ">":
            arr = arr.astype(arr.dtype.newbyteorder("<"))
        desc.append({"n": name, "d": arr.dtype.name, "s": list(arr.shape)})
        buffers.append(memoryview(arr).cast("B") if arr.size else b"")
    header = json.dumps({"v": PROTOCOL_VERSION, "op": op, "arrays": desc, "meta": meta or {}}, separators=(",", ":")).encode()
    return [header, *buffers]


def unpack(frames: list) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    """(op, arrays, meta) of a received message. `frames` may be bytes or zmq Frames; the arrays are read-only views of the received buffers (copy before modifying)."""
    if not frames:
        raise ProtocolError("empty message")
    first = frames[0]
    header = json.loads(bytes(first.buffer if hasattr(first, "buffer") else first))
    if header.get("v") != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version {header.get('v')} != {PROTOCOL_VERSION}")
    descs = header["arrays"]
    if len(frames) != 1 + len(descs):
        raise ProtocolError(f"header announces {len(descs)} arrays but {len(frames) - 1} buffers arrived")
    arrays = {}
    for d, frame in zip(descs, frames[1:]):
        buf = frame.buffer if hasattr(frame, "buffer") else frame
        dtype = DTYPES[d["d"]]
        shape = tuple(d["s"])
        expect = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if len(buf) != expect:
            raise ProtocolError(f"array {d['n']!r}: {len(buf)} bytes received, {expect} expected for {d['d']}{list(shape)}")
        arrays[d["n"]] = np.frombuffer(buf, dtype=dtype).reshape(shape)
    return header["op"], arrays, header.get("meta", {})


def nbytes(frames: list) -> int:
    return sum(len(f.buffer) if hasattr(f, "buffer") else len(f) for f in frames)
