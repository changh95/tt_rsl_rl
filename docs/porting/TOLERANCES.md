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
| MLP forward (actor mean and critic value; 1-4 layers, hidden 33..256, obs 4..84, out 1..29, with/without normalization, ELU/tanh/ReLU, padded dims) | rel_fro vs torch fp32 with an absolute floor: `||err||_rms / max(||ref||_rms, 0.1)` (padded outputs must be exactly 0) | **1.5e-2** | Phase 2, `tests/tt/test_tt_modules.py`, rows 32..4096: plain rel_fro 2.5e-3..3.5e-3 for actor outputs (rms 0.14..0.57), 7.7e-3 for a random-init critic whose outputs are tiny (rms 0.057: cancellation inflates the relative error); absolute rms error 3e-4..1.5e-3, growing with the activation scale; with the 0.1 floor the worst case is 4.4e-3 (critic), about one third of the tolerance |
| `EmpiricalNormalization.forward` `(x-mean)/(std+eps)` | max_rel / ULP (eltwise row) | as the eltwise row: 5e-7, <= 6 ULP | 1.7e-7, 2 ULP (dim 4..84) |
| `EmpiricalNormalization.update` running mean / var after each of 6 consecutive updates of 32..4096 rows | rel_fro vs upstream | **1e-6** | mean 2.8e-7, var 1.4e-7 (counts match exactly) |
| Gaussian head: std incl. clamping, `mean + std*noise`, log_prob (sum over real dims), entropy; scalar and log std, 1/12/29 action dims | std/sample/entropy/log_prob rel_fro | std as eltwise (0 error); sample, entropy, log_prob **1e-6** | sample 0 (scalar) / 6e-8 (log), entropy 7.7e-8, log_prob 1e-7 |
| PPO loss values and gradients (clipped surrogate and value loss with torch's `max` tie rule and inclusive `clamp` gradients, entropy bonus, std chain rule scalar/log) given IDENTICAL network outputs | rel_fro with an absolute floor (`||err||_rms / max(||ref||_rms, 1e-6)`) | **2e-5** | Phase 3, `tests/tt/test_tt_backward.py::test_ppo_loss_formulas_*`: surrogate loss 6e-6, d_mean 2.8e-6, d_std 3.0e-6, value loss 1.2e-7, log_prob 7e-8, d_value 0 (6 network configs, 64/512 rows) |
| MLP layer gradients (dW, db per layer: matmul transposes, bias row sums, ELU/tanh derivatives, padding) for the same output gradient | rel_fro with floor 1e-4 | **1.5e-2** (ELU, tanh) | 3.4e-3..4.9e-3 over 5 configs (rows 64/512); padded rows/columns of every gradient are exactly 0 |
| the same with ReLU | rel_fro | **1.2e-1** | 3.7e-2: the ReLU gate `z > 0` flips for units within the forward matmul noise of 0; ELU/tanh are smooth |
| PPO gradients end to end (forward noise + gate flips), data kept >= 0.02 away from the clip boundaries | rel_fro with floor 1e-4 | **8e-2** | <= 2.8e-2 over 6 configs; WITHOUT the margin one configuration (29 action dims) reaches 1.0e-1: the clipped objectives have a discontinuous gradient at the boundaries, so samples whose log-ratio is within the forward noise (~1e-2 for 29 dims) take the other branch. Expected and unavoidable; see DECISIONS ADR-0001 note |
| Adam step (fp32; m, v, bias corrections, lr as device tensor), 8 steps with changing lr, 2 parameter sets | rel_fro vs `torch.optim.Adam` | **1e-7** | 1.9e-8 |
| `clip_grad_norm_` (total norm and clipped gradients), 5 scales x max_norm | rel_fro | **2e-7** | norm 6.8e-8; clipped gradients exact |
| KL-adaptive learning rate: every branch and both clamps over 54 consecutive updates | lr vs the python-float rule | rel 1e-6 | <= 2.3e-7 (fp32 vs float64 after up to 30 multiplications by 1.5) |
| GAE returns and advantages (device loop, incl. the unbiased-std global normalisation) | rel_fro vs the upstream loop on host tensors | **1e-6** | returns 0 (bit exact), advantages 4.7e-8 (24 steps x 128 envs, with/without normalisation, 1 step) |
| rollout packing, permutation gather and mini-batch slices vs `RolloutStorage.mini_batch_generator` (same RNG state) | bitwise | `torch.equal` | bit exact (`tosa_gather` of fp32 rows with UINT32 indices; separate and shared actor/critic obs) |
| `PPO.update` losses (means over the update): value loss | rel_fro with floor 1e-3 | **5e-3** | <= 1.0e-3 (value loss differs by the matmul noise of the critic output, accumulated over 20 mini-batches) |
| `PPO.update` entropy loss | rel_fro | **2e-5** | <= 1e-6 (depends on the std parameter only) |
| `PPO.update` surrogate loss (normalised advantages, O(1)) | absolute | **2e-3** | <= 2.3e-4 |
| KL(old \|\| new) mean, 1/12/29 action dims, scalar and log std | rel_fro vs `torch.distributions.kl_divergence` | row-sum row: 1e-6 | measured in `test_kl_matches_upstream_kl_divergence`; KL of a distribution with itself is < 1e-6 |
| one whole `update()` (Adam, per-model clipping, 4 mini-batches x 5 epochs) from IDENTICAL state, smooth objective or single step: deviation of the actor mean / critic value of the updated parameters, as a fraction of the update (see below) | `||f_TT - f_ref|| / ||f_ref - f_init||` on the rollout obs | **10x the noise baseline + 1e-3** | TT 4e-4..1.1e-3 vs baseline (upstream with 4e-3-rms-relative gradient noise) 8e-5..4e-4 (cartpole, pendulum, tracking12) |
| the same with the standard clipped objective (20 Adam steps) | same | **8e-2** | <= 2.5e-2. The clipped surrogate / value gradients are DISCONTINUOUS at the clip boundaries: a sample whose ratio is within the forward noise of 1 +- eps takes the other branch on the device. Evidence: with `clip_param = 1e3` and unclipped value loss the same update deviates only 4e-4..1.1e-3 |
| 20 consecutive updates on identical data (no resync), smooth objective | same, worst over the 20 iterations and both models | **3e-2** | <= 1.0e-2 (noise baseline <= 4.4e-3) |
| the same, standard clipped objective | same | **2.5e-1** | <= 6.9e-2 (cartpole; tracking12 6.6e-2 vs a noise baseline of 4.2e-2). O(1) errors (wrong sign, missing term) show up as deviations >= 1 |
| composite quantities beyond this (the learning curves: Phase 4) | statistical | per BRIEF Phase 4 | - |

Why function space, and why a noise baseline (Phase 3 finding). Adam's first steps move every component by about +-lr whatever its gradient, so components with near-zero gradient flip sign under any numerical noise and a parameter-by-parameter comparison of two correct implementations shows 5..40 % "error" of the update size. What matters is the effect on the policy and the value, which is what the deviation above measures, and its natural size is obtained by injecting gradient noise of the measured matmul size into the REFERENCE: `add_grad_noise` in `tests/tt/test_tt_update.py`. Second finding: with identical data and state the TT KL of the first mini-batch is exactly 0 (rollout and update forwards are bit-identical on the device) whereas torch's is ~1e-9; upstream's `0 < kl < desired_kl/2` rule therefore raises the learning rate one mini-batch earlier in upstream. The parity runs use `schedule = "fixed"`; the schedule itself is covered by the learning-rate and KL tests.

Notes
- The matmul floor means component parity with upstream torch is ~1e-3 per matmul layer, not 1e-6. Gradient formulas are therefore
  verified against fp64 torch autograd at these floors; training equivalence is statistical (multi-seed learning curves; BRIEF Phase 4).
- The first mini-batch of an update will see a policy ratio that differs from 1 by ~1e-3 because the rollout forward and the update forward
  run different matmul shapes. This is documented behaviour, not something to tune away.
- `ttnn` closes the mesh at `close_mesh_device` but keeps the `/dev` node open until process exit; tests share one session-scoped mesh.
