# SPDX-License-Identifier: BSD-3-Clause
"""The rsl_rl v5 runner configuration of the MuJoCo humanoid demo (the same dictionary drives upstream PPO on the CPU and `rsl_rl_tt.ppo:TTPPO`)."""

from __future__ import annotations


def humanoid_train_cfg(
    *, num_steps: int = 24, hidden: tuple[int, ...] = (512, 256, 128), num_mini_batches: int = 4, num_epochs: int = 5, init_std: float = 0.5, learning_rate: float = 1.0e-3,
    entropy_coef: float = 0.0, save_interval: int = 100, csv_log: bool = True,
) -> dict:
    """Standard PPO for a locomotion task: ELU MLPs with observation normalisation, adaptive learning rate (target KL 0.01), gamma 0.99, lambda 0.95.

    `entropy_coef` 0 because the environment clips the actions to [-1, 1] (a growing std would cost nothing and an entropy bonus would only inflate it); `init_std` 0.5:
    with a std of 1 half of the exploration noise is clipped away. `save_interval` is in iterations (checkpoints `model_<it>.pt`, upstream format)."""
    cfg = {
        "num_steps_per_env": num_steps,
        "save_interval": save_interval,
        "check_for_nan": True,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel", "hidden_dims": list(hidden), "activation": "elu", "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": init_std, "std_type": "scalar"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": list(hidden), "activation": "elu", "obs_normalization": True},
        "algorithm": {
            "class_name": "PPO", "num_learning_epochs": num_epochs, "num_mini_batches": num_mini_batches, "clip_param": 0.2, "gamma": 0.99, "lam": 0.95,
            "value_loss_coef": 1.0, "entropy_coef": entropy_coef, "learning_rate": learning_rate, "max_grad_norm": 1.0, "optimizer": "adam",
            "use_clipped_value_loss": True, "schedule": "adaptive", "desired_kl": 0.01, "normalize_advantage_per_mini_batch": False,
        },
    }
    if csv_log:
        cfg["logger"] = {"class_name": "rsl_rl_tt.testing.csv_writer:CsvLogWriter"}
    return cfg
