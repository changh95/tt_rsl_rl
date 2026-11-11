# SPDX-License-Identifier: BSD-3-Clause
"""Generate the ONE-chip reference of the Phase 6 N-chip equivalence test (`tests/tt/data/mc_step_reference.pt`): three mini-batch steps on the union mini-batches
(`rsl_rl_tt/testing/mc_equiv.py`). Device job on one chip; commit the file (a few hundred KB).

    scripts/tt_run.sh --chips 16 --timeout 300 --tag mcref -- python scripts/mc_step_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.device import open_tray_mesh  # noqa: E402
from rsl_rl_tt.testing import mc_equiv  # noqa: E402


def main() -> None:
    out = REPO / "tests" / "tt" / "data" / mc_equiv.REFERENCE
    out.parent.mkdir(parents=True, exist_ok=True)
    with open_tray_mesh((1, 1)):
        runner = mc_equiv.make_runner([1, 1])
        alg = runner.alg
        pool, obs = mc_equiv.make_pool(alg)
        res = mc_equiv.run_steps(alg, pool, obs)
        mc_equiv.save_reference(out, res)  # only what the comparison uses (parameters / Adam moments are not compared one by one)
        print(f"saved {out} ({out.stat().st_size / 1e3:.0f} KB); learning rates after each step: {res['lrs']}")
        moved = {t: float((res[f'{t}_after'] - res[f'{t}_before']).abs().max()) for t in ('actor', 'critic')}
        print("outputs moved by the 3 steps (max abs):", moved)
        runner.alg.release_traces()


if __name__ == "__main__":
    main()
