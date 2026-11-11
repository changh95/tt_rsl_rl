# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests of the environment bridge (Phase 7, ADR-0003): `RemoteVecEnv` <-> `serve(env, endpoint)` over ZeroMQ (ipc loopback, the server in a thread).

The defining property: an environment behind the bridge is INDISTINGUISHABLE from the same environment in process: observations, rewards, dones, time-outs and the
episode logs are identical bit for bit, and training with the unmodified `OnPolicyRunner` (upstream PPO on the CPU here, TTPPO in `test_tt_bridge_device.py`) produces
the same metrics. The wire format, the error path and the timeout are tested separately.
"""

import csv
import threading

import numpy as np
import pytest
import torch

from rsl_rl.runners import OnPolicyRunner
from rsl_rl_tt.bridge import protocol
from rsl_rl_tt.bridge.client import RemoteEnvError, RemoteVecEnv
from rsl_rl_tt.bridge.server import serve
from rsl_rl_tt.testing.envs import make_env
from rsl_rl_tt.testing.reference import REF_SPECS, RefSpec, make_train_cfg


# -- wire format -------------------------------------------------------------------------------------------------------------------
def test_protocol_roundtrip_keeps_dtype_shape_and_bytes():
    arrays = {
        "a": np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32),
        "b": np.arange(7, dtype=np.int64),
        "c": np.array([[1, 0], [0, 1]], dtype=np.uint8),
        "d": np.zeros((0, 4), dtype=np.float32),  # empty arrays are legal (an environment with no privileged group)
        "e": np.float64([1.5]),
    }
    op, got, meta = protocol.unpack(protocol.pack("step", arrays, {"x": 1}))
    assert op == "step" and meta == {"x": 1} and list(got) == list(arrays)
    for k, v in arrays.items():
        assert got[k].dtype == v.dtype and got[k].shape == v.shape and np.array_equal(got[k], v), k


def test_protocol_rejects_malformed_messages():
    with pytest.raises(protocol.ProtocolError, match="dtype"):
        protocol.pack("x", {"a": np.zeros(3, dtype=np.complex64)})
    frames = protocol.pack("x", {"a": np.zeros((2, 2), dtype=np.float32)})
    with pytest.raises(protocol.ProtocolError, match="expected"):
        protocol.unpack([frames[0], b"\x00" * 7])  # wrong number of bytes
    with pytest.raises(protocol.ProtocolError, match="buffers"):
        protocol.unpack([frames[0]])  # the header announces an array that never arrived
    with pytest.raises(protocol.ProtocolError, match="version"):
        protocol.unpack([b'{"v": 99, "op": "x", "arrays": [], "meta": {}}'])


# -- helpers -----------------------------------------------------------------------------------------------------------------------
class Served:
    """An environment served in a thread; `.client` is the connected RemoteVecEnv."""

    def __init__(self, env, endpoint, agent_cfg=None):
        self.env, self.stop = env, threading.Event()
        self.thread = threading.Thread(target=serve, args=(env, endpoint), kwargs={"agent_cfg": agent_cfg, "stop": self.stop.is_set, "poll_ms": 20, "log": lambda m: None}, daemon=True)
        self.thread.start()
        self.client = RemoteVecEnv(endpoint, timeout_s=30)

    def close(self):
        try:
            self.client.close()
        finally:
            self.stop.set()
            self.thread.join(timeout=10)


@pytest.fixture
def endpoint(tmp_path):
    return f"ipc://{tmp_path}/env.sock"


def served(env, endpoint, agent_cfg=None):
    s = Served(env, endpoint, agent_cfg)
    return s


# -- indistinguishable from the local environment --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
def test_stepping_through_the_bridge_equals_stepping_in_process(name, endpoint):
    local, remote_env = make_env(name, 64, seed=3), make_env(name, 64, seed=3)
    s = served(remote_env, endpoint)
    try:
        env = s.client
        assert env.num_envs == local.num_envs and env.num_actions == local.num_actions and env.max_episode_length == local.max_episode_length
        o_l, o_r = local.get_observations(), env.get_observations()
        assert set(o_l.keys()) == set(o_r.keys()) and all(torch.equal(o_l[k], o_r[k]) for k in o_l.keys())
        g = torch.Generator().manual_seed(0)
        saw_log = saw_timeout = False
        for t in range(350):  # long enough for episodes to end by termination and by time-out (tracking12: 300 steps)
            actions = torch.randn(64, local.num_actions, generator=g)
            a = local.step(actions.clone())
            b = env.step(actions.clone())
            for k in a[0].keys():
                assert torch.equal(a[0][k], b[0][k]), f"step {t}: obs group {k} differs"
            assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) and b[2].dtype == torch.long and a[2].dtype == b[2].dtype, f"step {t}: rewards / dones differ"
            assert set(a[3]) == set(b[3]), f"step {t}: extras keys {set(a[3])} vs {set(b[3])}"
            assert torch.equal(a[3]["time_outs"], b[3]["time_outs"]) and b[3]["time_outs"].dtype == torch.bool
            saw_timeout |= bool(b[3]["time_outs"].any())
            if "log" in a[3]:
                saw_log = True
                assert set(a[3]["log"]) == set(b[3]["log"])
                for key in a[3]["log"]:
                    assert float(b[3]["log"][key]) == pytest.approx(float(torch.as_tensor(a[3]["log"][key]).double().mean()), rel=1e-6, abs=1e-9), key
        assert saw_log, "the test never produced an episode log"
        assert name != "tracking12" or saw_timeout, "the test never produced a time-out"
    finally:
        s.close()


def test_episode_length_buf_can_be_read_and_set_through_the_bridge(endpoint):
    local, remote_env = make_env("cartpole", 64, seed=1), make_env("cartpole", 64, seed=1)
    s = served(remote_env, endpoint)
    try:
        env = s.client
        assert torch.equal(env.episode_length_buf, local.episode_length_buf) and env.episode_length_buf.dtype == torch.long
        new = torch.randint(0, 400, (64,))
        env.episode_length_buf = new  # what OnPolicyRunner.learn(init_at_random_ep_len=True) does
        assert torch.equal(remote_env.episode_length_buf, new) and torch.equal(env.episode_length_buf, new)
    finally:
        s.close()


def test_info_carries_the_agent_configuration_of_the_server(endpoint):
    cfg = {"seed": 7, "algorithm": {"class_name": "PPO", "num_mini_batches": 4}, "obs_groups": {"actor": ["policy"]}, "clip_actions": None}
    s = served(make_env("cartpole", 64, seed=0), endpoint, agent_cfg=cfg)
    try:
        assert s.client.agent_cfg == cfg and s.client.obs_shapes == {"policy": (4,)}
        assert isinstance(s.client.cfg, dict) and s.client.cfg.get("num_envs") == 64
    finally:
        s.close()


def test_a_failing_request_is_reported_with_the_server_traceback_and_the_server_survives(endpoint):
    s = served(make_env("cartpole", 64, seed=0), endpoint)
    try:
        with pytest.raises(RemoteEnvError, match="server traceback"):
            s.client.step(torch.zeros(63, 1))  # wrong number of environments
        obs, rewards, dones, extras = s.client.step(torch.zeros(64, 1))  # the next request is served normally
        assert rewards.shape == (64,) and dones.dtype == torch.long
    finally:
        s.close()


def test_a_silent_server_raises_a_timeout_instead_of_hanging(tmp_path):
    import zmq

    ep = f"ipc://{tmp_path}/mute.sock"
    mute = zmq.Context.instance().socket(zmq.REP)
    mute.bind(ep)
    try:
        with pytest.raises(TimeoutError, match="no reply"):
            RemoteVecEnv(ep, timeout_s=0.3)
    finally:
        mute.close(linger=0)


# -- training through the bridge ---------------------------------------------------------------------------------------------------------
def _spec(name):
    base = REF_SPECS[name]
    return RefSpec(num_envs=128, num_steps=8, iterations=3, hidden=base.hidden, algorithm={**base.algorithm, "num_mini_batches": 4, "num_learning_epochs": 1})


def _train(env, name, log_dir, seed=5):
    torch.manual_seed(seed)
    runner = OnPolicyRunner(env, make_train_cfg(name, _spec(name)), log_dir=str(log_dir), device="cpu")
    runner.learn(num_learning_iterations=3, init_at_random_ep_len=True)
    return list(csv.DictReader(open(log_dir / "metrics.csv")))


@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
def test_training_through_the_bridge_reproduces_training_in_process(name, endpoint, tmp_path):
    """The unmodified OnPolicyRunner with upstream PPO: every logged scalar of every iteration is identical with the environment in process and behind the bridge."""
    in_process = _train(make_env(name, 128, seed=5), name, tmp_path / "local")
    s = served(make_env(name, 128, seed=5), endpoint)
    try:
        bridged = _train(s.client, name, tmp_path / "bridge")
    finally:
        s.close()
    assert len(in_process) == len(bridged) == 3
    assert list(in_process[0]) == list(bridged[0])
    for a, b in zip(in_process, bridged):
        for col in a:
            if col != "wall_s" and not col.startswith("Perf/"):  # wall-clock times and throughput differ by definition
                assert a[col] == b[col], f"iteration {a['iteration']}: {col} {a[col]} (in process) vs {b[col]} (bridge)"


# -- the configuration handed to the TT host ------------------------------------------------------------------------------------------
def _agent_cfg():
    import json

    return json.loads(json.dumps(make_train_cfg("tracking12", _spec("tracking12"), logger=None), default=repr))  # what travels in the `info` reply: JSON


def test_tt_agent_cfg_changes_only_what_defines_the_backend():
    from rsl_rl_tt.bridge.config import CSV_WRITER, TT_ALGORITHM, tt_agent_cfg

    base = _agent_cfg()
    snapshot = __import__("copy").deepcopy(base)
    cfg = tt_agent_cfg(base, mesh_shape=(2, 4), trace=True)
    assert base == snapshot, "the argument must not be modified"
    assert cfg["device"] == "cpu" and cfg["algorithm"]["class_name"] == TT_ALGORITHM and cfg["logger"] == {"class_name": CSV_WRITER}
    assert cfg["algorithm"]["tt_cfg"] == {"mesh_shape": [2, 4], "trace": True, "replica_check_interval": 10}
    # everything else is exactly what the environment host sent: same experiment
    for key in ("device", "logger"):
        base.pop(key, None), cfg.pop(key, None)
    base["algorithm"].pop("class_name"), cfg["algorithm"].pop("class_name"), cfg["algorithm"].pop("tt_cfg")
    assert cfg == base
    assert tt_agent_cfg(_agent_cfg(), mesh_shape=(1, 1))["algorithm"]["tt_cfg"] == {"mesh_shape": [1, 1], "trace": True}


def test_tt_agent_cfg_refuses_what_the_backend_cannot_run():
    from rsl_rl_tt.bridge.config import tt_agent_cfg

    old_format = {k: v for k, v in _agent_cfg().items() if k not in ("actor", "critic")} | {"policy": {"class_name": "ActorCritic"}}  # rsl_rl 3.x style
    with pytest.raises(ValueError, match="rsl_rl v5 format"):
        tt_agent_cfg(old_format)
    other_runner = _agent_cfg() | {"class_name": "DistillationRunner"}
    with pytest.raises(NotImplementedError, match="only OnPolicyRunner"):
        tt_agent_cfg(other_runner)
    other_algo = _agent_cfg()
    other_algo["algorithm"]["class_name"] = "Distillation"
    with pytest.raises(NotImplementedError, match="only PPO"):
        tt_agent_cfg(other_algo)


def test_a_saved_agent_configuration_loads_from_json_and_yaml(tmp_path):
    import json

    import yaml

    from rsl_rl_tt.bridge.config import load_agent_cfg

    cfg = _agent_cfg()
    (tmp_path / "agent.json").write_text(json.dumps(cfg))
    (tmp_path / "agent.yaml").write_text(yaml.safe_dump(cfg))
    assert load_agent_cfg(tmp_path / "agent.json") == cfg and load_agent_cfg(tmp_path / "agent.yaml") == cfg


# -- tools for the GPU reference comparison -----------------------------------------------------------------------------------------------
def test_tensorboard_run_converts_to_the_csv_layout_and_curves_compare(tmp_path):
    """A run with the DEFAULT (TensorBoard) logger of upstream rsl_rl, as an Isaac Lab GPU run writes it, converts to the CSV layout of this repository and the
    comparison tool applies the Phase 4 criterion to it."""
    import importlib.util
    import sys as _sys

    scripts = __import__("pathlib").Path(__file__).resolve().parents[2] / "scripts"
    _sys.path.insert(0, str(scripts))
    tb = importlib.import_module("tb_to_csv")
    cmp_ = importlib.import_module("compare_curves")
    cfg = make_train_cfg("cartpole", _spec("cartpole"), logger=None)  # no `logger` key: upstream's default, TensorBoard
    torch.manual_seed(2)
    OnPolicyRunner(make_env("cartpole", 128, seed=2), cfg, log_dir=str(tmp_path / "tb"), device="cpu").learn(num_learning_iterations=3, init_at_random_ep_len=True)
    assert importlib.util.find_spec("tensorboard") is not None
    assert tb.convert(tmp_path / "tb", tmp_path / "gpu_like.csv") == 3
    rows = cmp_.load(str(tmp_path / "gpu_like.csv"))
    assert [int(r["iteration"]) for r in rows] == [0, 1, 2] and "Train/mean_reward" in rows[0] and "Loss/value" in rows[0]
    report = cmp_.compare([str(tmp_path / "gpu_like.csv")], [str(tmp_path / "gpu_like.csv")])
    assert report["passes"] and report["tt_mean"] == report["reference_mean"]  # a run is within the criterion of itself
    worse = tmp_path / "worse.csv"
    worse.write_text("iteration,Train/mean_reward\n" + "\n".join(f"{i},{0.5 * r['Train/mean_reward']}" for i, r in enumerate(rows)) + "\n")
    assert not cmp_.compare([str(tmp_path / "gpu_like.csv")], [str(worse)])["passes"]  # half the reward is outside the 5 % margin


# -- committed evidence of the user-facing path ---------------------------------------------------------------------------------------
def test_committed_loopback_run_through_the_scripts_is_identical_to_the_in_process_run():
    """docs/porting/isaac_loopback: `scripts/isaac_env_server.py --local-env tracking12` + `scripts/train_isaac.py` (TTPPO on one chip, traced, configuration received from
    the server, 300 iterations) reproduces the Phase 5 in-process traced run of the same seed in EVERY logged scalar, and meets the Phase 4 criterion against the golden runs."""
    import json
    import pathlib

    docs = pathlib.Path(__file__).resolve().parents[2] / "docs" / "porting"
    read = lambda p: list(csv.DictReader(open(p)))
    local, bridged = read(docs / "tt_runs_trace" / "tracking12" / "seed0.csv"), read(docs / "isaac_loopback" / "tracking12_seed0_bridge.csv")
    assert len(local) == len(bridged) == 300 and list(local[0]) == list(bridged[0])
    for a, b in zip(local, bridged):
        for col in a:
            if col != "wall_s" and not col.startswith("Perf/"):
                assert a[col] == b[col], f"iteration {a['iteration']}: {col}"
    report = json.loads((docs / "isaac_loopback" / "compare_vs_golden.json").read_text())
    assert report["passes"] and report["all_finite"] and report["tt_mean"] >= report["threshold"]
