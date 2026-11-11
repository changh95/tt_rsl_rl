# SPDX-License-Identifier: BSD-3-Clause
"""The upstream (CPU torch) reference run: configs per test env and a helper that trains with the UNMODIFIED `OnPolicyRunner`.

Used for the golden baselines (`scripts/train_reference.py`) and later as the oracle in parity tests. Nothing here depends on ttnn.
"""

from __future__ import annotations

import copy
import os
import time
import torch
from dataclasses import dataclass

from rsl_rl.runners import OnPolicyRunner
from rsl_rl_tt.testing.envs import ENV_CLASSES, make_env

CSV_LOGGER = {"class_name": "rsl_rl_tt.testing.csv_writer:CsvLogWriter"}


@dataclass(frozen=True)
class RefSpec:
    num_envs: int
    num_steps: int
    iterations: int
    hidden: tuple[int, ...]
    algorithm: dict
    init_std: float = 1.0
    obs_normalization: bool = True


_PPO = {
    "class_name": "PPO",
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "clip_param": 0.2,
    "gamma": 0.99,
    "lam": 0.95,
    "value_loss_coef": 1.0,
    "entropy_coef": 0.005,
    "learning_rate": 1.0e-3,
    "max_grad_norm": 1.0,
    "optimizer": "adam",
    "use_clipped_value_loss": True,
    "schedule": "adaptive",
    "desired_kl": 0.01,
    "normalize_advantage_per_mini_batch": False,
}

# num_envs * num_steps is divisible by num_mini_batches * 32 in every spec (tile alignment is a Phase-2 requirement).
REF_SPECS: dict[str, RefSpec] = {
    "cartpole": RefSpec(num_envs=1024, num_steps=24, iterations=100, hidden=(64, 64), algorithm=_PPO),
    "pendulum": RefSpec(num_envs=1024, num_steps=32, iterations=200, hidden=(64, 64), algorithm={**_PPO, "gamma": 0.95}),
    # entropy_coef 0: the env clips actions to [-1, 1], so a growing std costs nothing and an entropy bonus would only inflate it
    "tracking12": RefSpec(num_envs=2048, num_steps=24, iterations=300, hidden=(128, 128, 64), algorithm={**_PPO, "entropy_coef": 0.0}),
}


def make_train_cfg(env_name: str, spec: RefSpec | None = None, logger: dict | None = CSV_LOGGER) -> dict:
    spec = spec or REF_SPECS[env_name]
    dist = {"class_name": "GaussianDistribution", "init_std": spec.init_std, "std_type": "scalar"}
    cfg = {
        "num_steps_per_env": spec.num_steps,
        "save_interval": 10_000,
        "check_for_nan": True,
        "obs_groups": copy.deepcopy(ENV_CLASSES[env_name].obs_groups_cfg),
        "actor": {"class_name": "MLPModel", "hidden_dims": list(spec.hidden), "activation": "elu", "obs_normalization": spec.obs_normalization, "distribution_cfg": dist},
        "critic": {"class_name": "MLPModel", "hidden_dims": list(spec.hidden), "activation": "elu", "obs_normalization": spec.obs_normalization},
        "algorithm": copy.deepcopy(spec.algorithm),
    }
    if logger is not None:
        cfg["logger"] = dict(logger)
    return cfg


def build_reference_runner(env_name: str, seed: int, log_dir: str | None, spec: RefSpec | None = None, num_envs: int | None = None, tt: bool = False):
    """Seed torch and the env, build env + the UNMODIFIED upstream OnPolicyRunner (device cpu). Returns (runner, env, spec).

    tt=True swaps only the algorithm class (`rsl_rl_tt.ppo:TTPPO`, single chip); everything else, including the runner, is upstream's.
    That needs an open tray mesh, i.e. a process started through scripts/tt_run.sh."""
    spec = spec or REF_SPECS[env_name]
    torch.manual_seed(seed)
    env = make_env(env_name, num_envs or spec.num_envs, seed=seed)
    cfg = make_train_cfg(env_name, spec)
    if tt:
        cfg["algorithm"]["class_name"] = "rsl_rl_tt.ppo:TTPPO"
        cfg["algorithm"]["tt_cfg"] = {"mesh_shape": [1, 1]}
    runner = OnPolicyRunner(env, cfg, log_dir=log_dir, device="cpu")
    return runner, env, spec


def train_reference(env_name: str, seed: int, log_dir: str, iterations: int | None = None, spec: RefSpec | None = None, num_threads: int | None = None, tt: bool = False) -> dict:
    """Train with the unmodified upstream runner (algorithm = upstream PPO, or TTPPO with tt=True); the CSV metrics land in `log_dir/metrics.csv`."""
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    os.makedirs(log_dir, exist_ok=True)
    runner, _, spec = build_reference_runner(env_name, seed, log_dir, spec, tt=tt)
    n_it = iterations if iterations is not None else spec.iterations
    t0 = time.time()
    runner.learn(num_learning_iterations=n_it, init_at_random_ep_len=True)
    wall = time.time() - t0
    return {"env": env_name, "seed": seed, "backend": "tt" if tt else "upstream-cpu", "iterations": n_it, "wall_s": wall, "num_threads": torch.get_num_threads(), "num_envs": spec.num_envs, "num_steps": spec.num_steps}
