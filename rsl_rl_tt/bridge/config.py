# SPDX-License-Identifier: BSD-3-Clause
"""Turn the agent configuration of an environment host into the configuration of a TT training run (Phase 7).

Isaac Lab trains with `OnPolicyRunner(env, agent_cfg.to_dict(), log_dir, device)`; the server hands that very dictionary to the TT host (`RemoteVecEnv.agent_cfg`, or
the `params/agent.yaml` of a GPU reference run), and the only changes made here are the ones that define the TT backend: the algorithm class and its `tt_cfg`, the
device of the host-side torch work (the CPU: the TT host has no GPU), and optionally a CSV log writer next to whatever logger the configuration names. Everything else
(network sizes, learning rates, `obs_groups`, `clip_actions`, ...) stays exactly as the environment host set it, so the two runs are the same experiment.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

TT_ALGORITHM = "rsl_rl_tt.ppo:TTPPO"
CSV_WRITER = "rsl_rl_tt.testing.csv_writer:CsvLogWriter"


def load_agent_cfg(path: str | Path) -> dict:
    """A saved agent configuration: JSON, or YAML (Isaac Lab's `params/agent.yaml`)."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        loaded = yaml.unsafe_load(text) if "!!python" in text else yaml.safe_load(text)  # Isaac Lab's dump_yaml may tag tuples; a plain dict otherwise
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: the agent configuration must be a mapping")
    return loaded


def tt_agent_cfg(agent_cfg: dict[str, Any], *, mesh_shape: tuple[int, int] = (1, 1), trace: bool = True, csv_log: bool = True, extra_tt_cfg: dict | None = None) -> dict:
    """The runner configuration of the TT run for the agent configuration `agent_cfg` (a copy; the argument is not modified)."""
    cfg = copy.deepcopy(agent_cfg)
    missing = [k for k in ("algorithm", "actor", "critic", "obs_groups") if k not in cfg]
    if missing:
        raise ValueError(
            f"the agent configuration has no {missing}: the TT backend needs the rsl_rl v5 format (separate actor / critic models, `obs_groups`); "
            f"got keys {sorted(cfg)}. Use an Isaac Lab version that trains with rsl_rl 5.x (docs/porting/ISAAC_RUNBOOK.md)"
        )
    if cfg.get("class_name", "OnPolicyRunner") != "OnPolicyRunner":
        raise NotImplementedError(f"runner {cfg['class_name']!r}: only OnPolicyRunner (PPO) is in the v1 scope")
    algorithm = cfg["algorithm"]
    if algorithm.get("class_name", "PPO") not in ("PPO", TT_ALGORITHM):
        raise NotImplementedError(f"algorithm {algorithm['class_name']!r}: only PPO is in the v1 scope")
    algorithm["class_name"] = TT_ALGORITHM
    algorithm["tt_cfg"] = {"mesh_shape": list(mesh_shape), "trace": bool(trace), **({"replica_check_interval": 10} if mesh_shape[0] * mesh_shape[1] > 1 else {}), **(extra_tt_cfg or {})}
    cfg["device"] = "cpu"
    if csv_log:
        cfg["logger"] = {"class_name": CSV_WRITER}
    return cfg
