# SPDX-License-Identifier: BSD-3-Clause
"""Round trip and bandwidth of the environment bridge (Phase 7, ADR-0003). Host only (no device); the server runs in its own process on loopback.

One step moves the actions to the environment host (num_envs x 12 float32) and the observations of tracking12's shapes back (36 + 48 float32 per environment) with
rewards, dones and time-outs: 342 bytes per environment. Compared on the same payload:

  zmq-tcp / zmq-ipc   the bridge (`RemoteVecEnv` <-> `serve`): JSON header + raw buffers, zero-copy send
  raw-tcp             a plain TCP socket with the same payloads and no serialisation at all: the floor of what loopback can do
  zmq-pickle          ZeroMQ `send_pyobj` of the torch tensors: the naive alternative that raw buffers are meant to beat

    python scripts/bench_bridge.py --out docs/porting/measurements/bridge_bench.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import socket
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OBS = {"policy": 36, "privileged": 48}
ACTIONS = 12


class NullEnv:
    """Constant observations: only the transport is measured."""

    def __init__(self, n: int) -> None:
        import torch

        self.num_envs, self.num_actions, self.max_episode_length, self.device, self.cfg = n, ACTIONS, 1000, "cpu", {}
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self._obs = {g: torch.zeros(n, d) for g, d in OBS.items()}
        self._r, self._d, self._t = torch.zeros(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.bool)

    def get_observations(self):
        return self._obs

    def step(self, actions):
        return self._obs, self._r, self._d, {"time_outs": self._t}


def _zmq_server(endpoint: str, n: int) -> None:
    from rsl_rl_tt.bridge.server import serve

    serve(NullEnv(n), endpoint, log=lambda m: None)


def _pickle_server(endpoint: str, n: int) -> None:
    import torch
    import zmq

    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(endpoint)
    reply = {"obs": {g: torch.zeros(n, d) for g, d in OBS.items()}, "r": torch.zeros(n), "d": torch.zeros(n, dtype=torch.long), "t": torch.zeros(n, dtype=torch.bool)}
    while True:
        msg = sock.recv_pyobj()
        if msg == "close":
            sock.send_pyobj("ok")
            return
        sock.send_pyobj(reply)


def _raw_server(port: int, request_bytes: int, reply_bytes: int) -> None:
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    buf = bytearray(request_bytes)
    payload = b"\x00" * reply_bytes
    while True:
        view = memoryview(buf)
        got = 0
        while got < request_bytes:
            k = conn.recv_into(view[got:], request_bytes - got)
            if k == 0:
                return
            got += k
        if buf[:1] == b"\xff":
            return
        conn.sendall(payload)


def _stats(times: list[float], request_bytes: int, reply_bytes: int) -> dict:
    t = sorted(times)
    med = statistics.median(t)
    return {"median_ms": med * 1e3, "p95_ms": t[int(0.95 * (len(t) - 1))] * 1e3, "min_ms": t[0] * 1e3, "MB_per_s": (request_bytes + reply_bytes) / med / 1e6}


def bench_zmq(endpoint: str, n: int, steps: int) -> dict:
    import torch

    from rsl_rl_tt.bridge.client import RemoteVecEnv

    server = mp.get_context("spawn").Process(target=_zmq_server, args=(endpoint, n), daemon=True)
    server.start()
    try:
        env = RemoteVecEnv(endpoint, timeout_s=120)
        actions = torch.zeros(n, ACTIONS)
        for _ in range(10):
            env.step(actions)
        env.stats.step_rtt_s.clear()
        sent0, recv0 = env.stats.sent_bytes, env.stats.received_bytes
        for _ in range(steps):
            env.step(actions)
        req, rep = (env.stats.sent_bytes - sent0) // steps, (env.stats.received_bytes - recv0) // steps
        out = _stats(env.stats.step_rtt_s, req, rep) | {"request_bytes": req, "reply_bytes": rep}
        env.close()
        return out
    finally:
        server.join(timeout=10)
        if server.is_alive():
            server.terminate()


def bench_pickle(endpoint: str, n: int, steps: int) -> dict:
    import torch
    import zmq

    server = mp.get_context("spawn").Process(target=_pickle_server, args=(endpoint, n), daemon=True)
    server.start()
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.connect(endpoint)
    actions = torch.zeros(n, ACTIONS)
    times = []
    try:
        for i in range(steps + 10):
            t0 = time.perf_counter()
            sock.send_pyobj(actions)
            sock.recv_pyobj()
            if i >= 10:
                times.append(time.perf_counter() - t0)
        sock.send_pyobj("close")
        sock.recv_pyobj()
    finally:
        sock.close(linger=0)
        server.join(timeout=10)
        if server.is_alive():
            server.terminate()
    return _stats(times, n * ACTIONS * 4, n * (sum(OBS.values()) * 4 + 4 + 8 + 1))


def bench_raw(port: int, n: int, steps: int, request_bytes: int, reply_bytes: int) -> dict:
    server = mp.get_context("spawn").Process(target=_raw_server, args=(port, request_bytes, reply_bytes), daemon=True)
    server.start()
    time.sleep(0.5)
    cli = socket.create_connection(("127.0.0.1", port))
    cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    payload = b"\x00" * request_bytes
    buf = bytearray(reply_bytes)
    times = []
    for i in range(steps + 10):
        t0 = time.perf_counter()
        cli.sendall(payload)
        view, got = memoryview(buf), 0
        while got < reply_bytes:
            got += cli.recv_into(view[got:], reply_bytes - got)
        if i >= 10:
            times.append(time.perf_counter() - t0)
    cli.sendall(b"\xff" + b"\x00" * (request_bytes - 1))
    cli.close()
    server.join(timeout=10)
    if server.is_alive():
        server.terminate()
    return _stats(times, request_bytes, reply_bytes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[256, 4096, 16384, 65536])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results: dict = {}
    port = 15600
    for n in args.envs:
        row = {}
        zmq_tcp = bench_zmq(f"tcp://127.0.0.1:{port}", n, args.steps)
        port += 1
        row["zmq-tcp"] = zmq_tcp
        row["zmq-ipc"] = bench_zmq(f"ipc://{REPO}/.run/bridge_bench_{n}.sock", n, args.steps)
        row["raw-tcp"] = bench_raw(port, n, args.steps, zmq_tcp["request_bytes"], zmq_tcp["reply_bytes"])
        port += 1
        row["zmq-pickle"] = bench_pickle(f"tcp://127.0.0.1:{port}", n, args.steps)
        port += 1
        results[n] = row
        print(f"num_envs {n}: request {zmq_tcp['request_bytes'] / 1e6:.2f} MB, reply {zmq_tcp['reply_bytes'] / 1e6:.2f} MB per step")
        for name, r in row.items():
            print(f"  {name:11s} median {r['median_ms']:8.3f} ms  p95 {r['p95_ms']:8.3f} ms  {r['MB_per_s']:8.0f} MB/s")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
