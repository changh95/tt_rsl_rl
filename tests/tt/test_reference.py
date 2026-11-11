# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests of the upstream reference harness (BRIEF Phase 1): the unmodified OnPolicyRunner trains on every test env,
deterministically, and the CSV writer captures the Logger's scalars."""

import csv
import json
import torch
from pathlib import Path

import pytest

from rsl_rl_tt.testing.csv_writer import CsvLogWriter
from rsl_rl_tt.testing.envs import ENV_NAMES
from rsl_rl_tt.testing.reference import REF_SPECS, RefSpec, make_train_cfg, train_reference

TINY = {name: RefSpec(num_envs=64, num_steps=8, iterations=3, hidden=(16, 16), algorithm=REF_SPECS[name].algorithm) for name in ENV_NAMES}


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_csv_writer_one_row_per_iteration_and_union_of_columns(tmp_path):
    w = CsvLogWriter(str(tmp_path))
    w.add_scalar("Loss/value", 1.5, 0)
    w.add_scalar("Loss/surrogate", -0.1, 0)
    w.add_scalar("Loss/value", 1.2, 1)
    w.add_scalar("Train/mean_reward", 7.0, 1)  # a tag that first appears later
    w.stop()
    rows = _rows(tmp_path / "metrics.csv")
    assert [r["iteration"] for r in rows] == ["0", "1"]
    assert rows[0]["Train/mean_reward"] == "" and rows[1]["Train/mean_reward"] == "7.0" and rows[1]["Loss/value"] == "1.2"
    assert list(rows[0].keys())[:2] == ["iteration", "wall_s"]
    w.store_config({"env": object(), "a": [1, 2]}, {"x": {"y": (1, 2)}})
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["train_cfg"] == {"x": {"y": [1, 2]}} and cfg["env_cfg"]["a"] == [1, 2]


@pytest.mark.parametrize("name", ENV_NAMES)
def test_upstream_runner_trains_on_every_env_and_logs_the_expected_scalars(name, tmp_path):
    info = train_reference(name, 0, str(tmp_path), iterations=3, spec=TINY[name], num_threads=2)
    rows = _rows(tmp_path / "metrics.csv")
    assert len(rows) == 3 and info["iterations"] == 3
    cols = set(rows[0].keys())
    assert {"iteration", "wall_s", "Loss/value", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate", "Policy/mean_std", "Perf/collection_time", "Perf/learning_time"} <= cols
    assert all(torch.isfinite(torch.tensor(float(r["Loss/value"]))) for r in rows)
    assert (tmp_path / "config.json").exists() and list(tmp_path.glob("model_*.pt"))  # config dumped, final model saved by the runner


@pytest.mark.parametrize("name", ["pendulum", "tracking12"])
def test_upstream_training_is_deterministic_given_the_seed(name, tmp_path):
    """Golden curves are only meaningful if the reference reproduces itself bit for bit (same torch build, same thread count)."""
    a, b = tmp_path / "a", tmp_path / "b"
    train_reference(name, 3, str(a), iterations=3, spec=TINY[name], num_threads=2)
    train_reference(name, 3, str(b), iterations=3, spec=TINY[name], num_threads=2)
    ra, rb = _rows(a / "metrics.csv"), _rows(b / "metrics.csv")
    keys = [k for k in ra[0] if k.startswith(("Loss/", "Policy/", "Train/"))]
    assert keys and all(x[k] == y[k] for x, y in zip(ra, rb) for k in keys)
    c = tmp_path / "c"
    train_reference(name, 4, str(c), iterations=3, spec=TINY[name], num_threads=2)
    assert [r["Loss/value"] for r in _rows(c / "metrics.csv")] != [r["Loss/value"] for r in ra]


@pytest.mark.parametrize("name", ENV_NAMES)
def test_reference_specs_satisfy_the_tile_alignment_rule(name):
    """BRIEF 3.3: per-chip batch and mini-batch must be multiples of 32 (no padding, no short last mini-batch); required from Phase 2 on."""
    spec = REF_SPECS[name]
    batch = spec.num_envs * spec.num_steps
    mini = spec.algorithm["num_mini_batches"]
    assert batch % mini == 0 and (batch // mini) % 32 == 0 and spec.num_envs % 32 == 0


def test_make_train_cfg_uses_obs_groups_and_the_csv_logger():
    cfg = make_train_cfg("tracking12")
    assert cfg["obs_groups"] == {"actor": ["policy"], "critic": ["policy", "privileged"]}
    assert cfg["logger"]["class_name"].endswith(":CsvLogWriter")
    assert cfg["actor"]["distribution_cfg"]["class_name"] == "GaussianDistribution" and "distribution_cfg" not in cfg["critic"]
    assert make_train_cfg("cartpole", logger=None).get("logger") is None
