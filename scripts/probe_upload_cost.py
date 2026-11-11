# SPDX-License-Identifier: BSD-3-Clause
"""Phase 5 probe (HOST ONLY, no device is opened): where do the ~74 ms per act step at 65536 envs go? Times the host-side pieces of one obs + noise upload:
staging, the RNG draw, `ttnn.from_torch` with TILE vs ROW_MAJOR layout (a host tensor is built; tilize happens on the host)."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl_tt.tt_common import pad_matrix, stage_obs  # noqa: E402


def med(fn, repeats: int = 7) -> float:
    fn()
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return statistics.median(t) * 1e3


def main() -> None:
    import ttnn
    from tensordict import TensorDict

    torch.set_num_threads(16)
    for n in (4096, 65536):
        obs = TensorDict({"policy": torch.randn(n, 36)}, batch_size=[n])
        buf = stage_obs(obs, ["policy"], 64)
        noise_real = torch.randn(n, 12)
        noise_pad = pad_matrix(noise_real, n, 32)
        print(f"N={n}")
        print(f"  stage_obs [{n}, 64]                       {med(lambda: stage_obs(obs, ['policy'], 64)):8.2f} ms")
        print(f"  torch.randn [{n}, 12] + pad_matrix -> 32   {med(lambda: pad_matrix(torch.randn(n, 12), n, 32)):8.2f} ms")
        for layout_name, layout in (("TILE", ttnn.TILE_LAYOUT), ("ROW_MAJOR", ttnn.ROW_MAJOR_LAYOUT)):
            t_obs = med(lambda: ttnn.from_torch(buf, dtype=ttnn.float32, layout=layout))
            t_noise = med(lambda: ttnn.from_torch(noise_pad, dtype=ttnn.float32, layout=layout))
            t_noise_real = med(lambda: ttnn.from_torch(noise_real, dtype=ttnn.float32, layout=layout))
            print(f"  from_torch {layout_name:9s}: obs [{n}, 64] {t_obs:8.2f} ms | noise [{n}, 32] {t_noise:8.2f} ms | noise [{n}, 12] {t_noise_real:8.2f} ms")


if __name__ == "__main__":
    main()
