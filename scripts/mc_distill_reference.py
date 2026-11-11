# SPDX-License-Identifier: BSD-3-Clause
"""Generate the ONE-chip reference of the N-chip equivalence test of the distillation update (`tests/tt/data/mc_step_reference_distill.npz`; `rsl_rl_tt/testing/distill_equiv.py`).
Device job on one chip; the teacher is a short upstream PPO run on the CPU (made on first use under `.run/teachers`, git-ignored). Commit the .npz (a few hundred KB).

    scripts/tt_run.sh --chips 16 --timeout 600 --tag mcref -- python scripts/mc_distill_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.device import open_tray_mesh  # noqa: E402
from rsl_rl_tt.testing import distill_equiv  # noqa: E402
from rsl_rl_tt.testing.reference import make_teacher_checkpoint  # noqa: E402


def main() -> None:
    out = REPO / "tests" / "tt" / "data" / distill_equiv.REFERENCE
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = REPO / ".run" / "teachers" / "tracking12_mc.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if not ckpt.exists():
        make_teacher_checkpoint(distill_equiv.ENV, str(ckpt), seed=0, iterations=5, num_envs=128)
    with open_tray_mesh((1, 1)):
        runner = distill_equiv.make_runner([1, 1], str(ckpt))
        res = distill_equiv.run_update(runner.alg)
        distill_equiv.save_reference(out, res)
        print(f"saved {out} ({out.stat().st_size / 1e3:.0f} KB); behavior loss {res['loss']:.6f}; the student's outputs moved by {float((res['after'] - res['before']).abs().max()):.4f} (max abs)")
        runner.alg.release_traces()


if __name__ == "__main__":
    main()
