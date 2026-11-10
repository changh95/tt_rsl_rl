# Parity tolerances

Source of every number: on-device measurements on Blackhole (tt-metal 0.81.1 wheel, KMD 2.10.0), 2026-11-10, against a **float64 torch
reference**, fp32 inputs. Repros (all device tests, run through `scripts/tt_run.sh`): `tests/tt/test_smoke_device.py`,
`test_smoke_tray.py`, `test_noise_floor.py`, `test_noise_floor_split.py`, `test_noise_floor_mesh.py`; raw numbers in
`.run/results/*.json` (not committed; regenerate with the commands in each file).

Rule (BRIEF 2-9): a failing test never loosens a tolerance. A tolerance changes only after a root-cause analysis recorded in
`DECISIONS.md`. Initial rule used below: **tolerance = ~3x the measured noise floor, rounded to one significant figure**, so a
real bug (wrong formula, wrong layout, wrong broadcast, order-1 errors) fails by orders of magnitude while hardware rounding passes.
Metrics: `rel_fro = ||out - ref||_F / ||ref||_F`; `max_rel`/`ULP` are element-wise.

| Component | Metric | Tolerance | Measured floor (evidence) |
|-----------|--------|-----------|---------------------------|
| fp32 host <-> device transfer (TILE) | bit exact | `torch.equal` | bit exact, 1 chip and 8 chips (sharded) |
| fp32 matmul / linear (forward and the two backward matmuls), HiFi4 + `fp32_dest_acc_en`, approx off | rel_fro vs fp64 | **4e-3** | 1.17e-3 (default config identical); K = 48..1024, M up to 16384, normal and heavy-tailed inputs; HiFi3 1.28e-3; per-chip spread on 8 chips 1.16-1.18e-3. torch fp32 itself: 2.4e-7. Multi-pass operand splits reach only 3.1e-4 (a 2.6e-4 floor on the result, independent of K), so they are not a way to tighten this |
| fp32 matmul, any other fidelity / no fp32 accumulation | **not allowed on the TT path** | - | LoFi 3.1e-2, HiFi2 7.2e-3, no fp32 acc 8e-3 |
| `exp, log, sqrt, rsqrt, reciprocal, tanh` (default, non-approximate), `elu`, `elu_bw` | max_rel vs fp64 (and max ULP) | **5e-7** (<= 6 ULP) | max_rel 7e-8..1.4e-7, <= 2 ULP over [-20,20] (exp), [1e-6,1e3] (log/sqrt/rsqrt), [0.01,100], [-10,10]; `elu_bw` at x == 0 returns 1.0 like torch. `exp(x)` is inf for x >= 88.8 and 0 below about -87.3 (flush), like fp32 |
| `fast_and_approximate_mode=True` on `exp`/`tanh` | **not allowed** | - | rel_fro 2.1e-2 / 2.4e-3 |
| `ttnn.softplus` | not used in v1 | - | max_rel 4.9e-4 (naive form); do not introduce without a new measurement |
| row sum / mean over N rows (4000..65536) | rel_fro | **1e-6** | 9e-8..1.3e-7 (element max_rel up to 1e-4 only where the true sum cancels to ~0: compare with rel_fro or an absolute floor, not per-element) |
| variance over rows | rel_fro | **2e-7**, **composed two-pass only**: `mean((x - mean)^2)` | 3.7e-8..5e-8 incl. offset-100 and 1e-3-scale data. `ttnn.var` itself: 2.4e-4 (N=4096), 1.3e-5 (N=65536) — **not used**; one-pass `E[x^2]-E[x]^2`: 9e-4 with a non-zero mean — **not used** |
| global gradient norm `sqrt(sum(square(g)))`, 5e4..3e6 values | rel | **1e-7** | 1.9e-8..4.2e-8 |
| `max` over rows | exact | `==` | 0 error |
| cross-chip fp32 sum, **gather (both mesh axes) + `ttnn.sum`**, 2x4 mesh, 8K..1M values/chip, chips with 4 decades of scale difference | rel_fro vs fp64 sum; chips bit-identical | **2e-7**; `torch.equal` across chips | 5.1e-8 (`ttnn.sum`), 2.7e-8 (fixed-order `ttnn.add` chain); bit-identical across the 8 chips |
| cross-chip fp32 `ttnn.all_reduce` (fallback only) | rel_fro vs fp64 sum | **2e-3** | 5.5e-4 (flat) / 5.7e-4 (per mesh axis); bit-identical across chips |
| composite quantities (MLP forward over L layers, log-prob, GAE, one full Adam step, one PPO update) | to be set | - | **Not measured yet.** Set in Phase 2/3 from the measured noise of the composed op (about 3x), with the numbers added here and the evidence recorded; never copied from a pass/fail run |

Notes
- The matmul floor means component parity with upstream torch is ~1e-3 per matmul layer, not 1e-6. Gradient formulas are therefore
  verified against fp64 torch autograd at these floors; training equivalence is statistical (multi-seed learning curves; BRIEF Phase 4).
- The first mini-batch of an update will see a policy ratio that differs from 1 by ~1e-3 because the rollout forward and the update forward
  run different matmul shapes. This is documented behaviour, not something to tune away.
- `ttnn` closes the mesh at `close_mesh_device` but keeps the `/dev` node open until process exit; tests share one session-scoped mesh.
