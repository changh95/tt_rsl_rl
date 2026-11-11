# SPDX-License-Identifier: BSD-3-Clause
"""Phase 7 device test: TTPPO trained through the environment bridge (`RemoteVecEnv`, configuration received from the server and turned into the TT configuration by
`tt_agent_cfg`) is identical, bit for bit, to TTPPO trained with the same environment in process: every logged scalar and every parameter.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p7 -- python -m pytest -m tt tests/tt/test_tt_bridge_device.py -q -p no:cacheprovider
"""

import copy
import csv
import threading

import pytest
import torch

from rsl_rl.runners import OnPolicyRunner
from rsl_rl_tt.bridge.client import RemoteVecEnv
from rsl_rl_tt.bridge.config import tt_agent_cfg
from rsl_rl_tt.bridge.server import serve
from rsl_rl_tt.testing.envs import make_env
from rsl_rl_tt.testing.reference import REF_SPECS, RefSpec, make_train_cfg, test_mesh_shape
from tests.tt.test_tt_update import tt_params

pytestmark = pytest.mark.tt


def _agent_cfg_as_sent(name, num_envs):
    import json

    base = REF_SPECS[name]
    spec = RefSpec(num_envs=num_envs, num_steps=8, iterations=3, hidden=base.hidden, algorithm={**base.algorithm, "num_mini_batches": 4, "num_learning_epochs": 1})
    return json.loads(json.dumps(make_train_cfg(name, spec, logger=None), default=repr))  # the JSON that travels in the server's `info` reply


def _train(env, cfg, log_dir):
    torch.manual_seed(11)
    runner = OnPolicyRunner(env, cfg, log_dir=str(log_dir), device="cpu")
    try:
        runner.learn(num_learning_iterations=3, init_at_random_ep_len=True)
        rows = list(csv.DictReader(open(log_dir / "metrics.csv")))
        return rows, tt_params(runner.alg)
    finally:
        runner.alg.release_traces()


@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
def test_ttppo_through_the_bridge_equals_ttppo_in_process(tt_mesh, name, tmp_path):
    rows_, cols_ = test_mesh_shape()
    num_envs = max(128, 32 * rows_ * cols_)
    sent = _agent_cfg_as_sent(name, num_envs)
    cfg = tt_agent_cfg(sent, mesh_shape=(rows_, cols_), trace=False)  # `tt_mesh` has no trace region; traced == eager is tested in test_tt_trace.py
    local_rows, local_params = _train(make_env(name, num_envs, seed=5), copy.deepcopy(cfg), tmp_path / "local")  # (the runner adds keys to the dict it is given)

    endpoint = f"ipc://{tmp_path}/env.sock"
    stop = threading.Event()
    server = threading.Thread(target=serve, args=(make_env(name, num_envs, seed=5), endpoint), kwargs={"agent_cfg": sent, "stop": stop.is_set, "poll_ms": 20, "log": lambda m: None}, daemon=True)
    server.start()
    env = RemoteVecEnv(endpoint, timeout_s=60)
    try:
        received = tt_agent_cfg(env.agent_cfg, mesh_shape=(rows_, cols_), trace=False)  # the configuration of the TT run comes FROM the server
        assert received == cfg
        bridged_rows, bridged_params = _train(env, copy.deepcopy(received), tmp_path / "bridge")
        assert env.stats.calls["step"] == 3 * 8  # three iterations of eight steps crossed the bridge
    finally:
        env.close()
        stop.set()
        server.join(timeout=10)

    assert len(local_rows) == len(bridged_rows) == 3
    for a, b in zip(local_rows, bridged_rows):
        for col in a:
            if col != "wall_s" and not col.startswith("Perf/"):
                assert a[col] == b[col], f"iteration {a['iteration']}: {col}: {a[col]} (in process) vs {b[col]} (bridge)"
    for k in local_params:
        assert torch.equal(local_params[k], bridged_params[k]), f"parameter {k} differs between in-process and bridged training"
