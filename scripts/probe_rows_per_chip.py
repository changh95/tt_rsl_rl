# SPDX-License-Identifier: BSD-3-Clause
"""Phase 6 probe: does the fp32 forward pass of a row depend on how many rows share its chip? (T-14 says no for one chip; the N-chip equivalence test assumed the
same across chip counts.) Runs the actor / critic forward on the SAME observations with the SAME parameters on the current mesh and compares with the 1-chip
reference outputs (`tests/tt/data/mc_step_reference.pt`, 1536 rows on one chip), then with a plain torch float64 forward as the yardstick for the matmul noise floor.

    scripts/tt_run.sh --chips 16,17 --timeout 300 --tag probe_rows -- env TT_TEST_MESH=1x2 python scripts/probe_rows_per_chip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.device import open_tray_mesh  # noqa: E402
from rsl_rl_tt.testing import mc_equiv  # noqa: E402
from rsl_rl_tt.testing.reference import test_mesh_shape  # noqa: E402


def main() -> None:
    rows, cols = test_mesh_shape()
    with open_tray_mesh((rows, cols), fabric_config="FABRIC_1D" if rows * cols > 1 else None):
        runner = mc_equiv.make_runner([rows, cols])
        alg = runner.alg
        pool, obs = mc_equiv.make_pool(alg)
        ctx = alg.ctx
        out_a = ctx.download(alg.tt_actor.forward(alg.tt_actor.obs_to_device(obs)), op="H-2")[:, : alg.tt_actor.output_dim]
        out_c = ctx.download(alg.tt_critic.forward(alg.tt_critic.obs_to_device(obs)), op="H-2")[:, :1]
        ref = torch.load(REPO / "tests/tt/data" / mc_equiv.REFERENCE, map_location="cpu")
        with torch.no_grad():
            want_a, want_c = alg._raw_actor(obs).double(), alg._raw_critic(obs).double()
        n = rows * cols
        print(f"mesh {rows}x{cols}: {obs.batch_size[0]} rows, {obs.batch_size[0] // n} per chip")
        for tag, got, ref1, want in (("actor", out_a, ref["actor_before"], want_a), ("critic", out_c, ref["critic_before"], want_c)):
            d = (got - ref1).abs()
            print(f"  {tag}: vs 1-chip TT: max abs diff {float(d.max()):.3e} ({int((d > 0).sum())} of {d.numel()} values differ), outputs up to {float(ref1.abs().max()):.2f}; "
                  f"TT vs float64 torch: this mesh {float((got.double() - want).abs().max()):.3e}, 1-chip {float((ref1.double() - want).abs().max()):.3e}")
        runner.alg.release_traces()


if __name__ == "__main__":
    main()
