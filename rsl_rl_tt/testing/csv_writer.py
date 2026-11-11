# SPDX-License-Identifier: BSD-3-Clause
"""A `rsl_rl.utils.LogWriter` that stores every scalar the upstream `Logger` emits as one CSV row per iteration.

Use it through the runner config (no upstream patch):  cfg["logger"] = {"class_name": "rsl_rl_tt.testing.csv_writer:CsvLogWriter"}.
It also dumps `config.json` (env cfg + train cfg) next to the CSV. The CSV is rewritten whenever an iteration completes, so an
interrupted run still leaves a valid file. Columns are `iteration`, `wall_s` (seconds since the writer was created), then every tag
seen so far, sorted by first appearance.
"""

from __future__ import annotations

import csv
import json
import os
import time

from rsl_rl.utils import LogWriter


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return repr(obj)


class CsvLogWriter(LogWriter):
    def __init__(self, log_dir: str, filename: str = "metrics.csv", **kwargs) -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self._t0 = time.time()
        self._columns: list[str] = []
        self._rows: dict[int, dict[str, float]] = {}
        self._current: int | None = None

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        if self._current is not None and global_step != self._current:
            self._flush()
        self._current = global_step
        row = self._rows.setdefault(global_step, {"iteration": global_step, "wall_s": round(time.time() - self._t0, 3)})
        row[tag] = float(scalar_value)
        if tag not in self._columns:
            self._columns.append(tag)

    def store_config(self, env_cfg, train_cfg) -> None:
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump({"env_cfg": _jsonable(env_cfg), "train_cfg": _jsonable(train_cfg)}, f, indent=2)

    def _flush(self) -> None:
        cols = ["iteration", "wall_s", *self._columns]
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, restval="")
            w.writeheader()
            for it in sorted(self._rows):
                w.writerow(self._rows[it])

    def stop(self) -> None:
        self._flush()
