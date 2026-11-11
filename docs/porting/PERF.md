# PERF.md — Phase 5: one-chip performance (traced path)

All numbers: one Blackhole chip (tray 3, `/dev/tenstorrent/16`, 34 GB DRAM = 8 banks x 4.26 GB, compute grid 12 x 10), fp32 policy (HiFi4, TOLERANCES.md), ttnn 0.81.1,
host AMD EPYC 9354P (32 cores / 64 threads). Raw data: `docs/porting/measurements/bench_*.json`, `opprofile_*.json`, `probe_upload_device.json`; commands at the end.

## 1. Method

* **Algorithm cost only.** `scripts/bench_matrix.py` drives the unmodified `OnPolicyRunner`'s PPO (upstream CPU) or `TTPPO` with a NULL environment (pre-generated random
  observations, constant dones), so only the algorithm is measured: one iteration = 24 `act` + `process_env_step` + `compute_returns` + `update` (5 epochs x 4 mini-batches, upstream
  defaults). Shapes follow the `tracking12` task: actor obs 36, critic obs 84, 12 actions, ELU MLPs.
* The cost of the real CPU `tracking12` environment is listed separately (section 2, last column; `--env-cost`). Isaac Lab on a GPU host adds its own transfers (Phase 7).
* Median over 3 measured iterations after 2 warm-ups (TT: the warm-ups include JIT compilation, the eager first step and the trace captures; CPU: 1 warm-up, 3 iterations, 2 at 65536 envs).
  Upstream CPU uses 32 torch threads (the physical cores); at the smallest sizes fewer threads are faster, see section 5 (+-20 % run-to-run noise there).
* Device jobs run alone (no CPU benchmark in parallel). TT times of the traced path are wall times of the Python loop; the traced GAE and update replays are asynchronous, so their device time
  appears in the next blocking call (`returns_ms` is only the dispatch; look at the totals).

## 2. Headline: TT traced vs upstream CPU, per training iteration (24 steps x num_envs samples)

| num_envs | hidden | CPU ms | TT ms | speedup | TT act | TT process | TT update | TT env-steps/s | CPU env-steps/s | real env step cost (CPU) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | [512, 256, 128] | 3264 | 447 | **7.3x** | 79 | 36 | 332 | 220 k | 30 k | 38 ms |
| 4096 | [1024, 1024, 512] | 16153 | 913 | **17.7x** | 100 | 36 | 777 | 108 k | 6.1 k | 38 ms |
| 16384 | [512, 256, 128] | 19986 | 1469 | **13.6x** | 252 | 45 | 1172 | 268 k | 19.7 k | 121 ms |
| 16384 | [1024, 1024, 512] | 65174 | 3743 | **17.4x** | 339 | 24 | 3380 | 105 k | 6.0 k | 121 ms |
| 65536 | [512, 256, 128] | 73224 | 5522 | **13.3x** | 894 | 57 | 4567 | 285 k | 21.5 k | 391 ms |
| 65536 | [1024, 1024, 512] | 238472 | 12530 | **19.0x** | 1358 | 72 | 11096 | 126 k | 6.6 k | 391 ms |

(`process` = `process_env_step`, 24 calls; the last column is the real `tracking12` CPU environment for 24 steps, not included in either side.) With 16 mini-batches the largest
configuration takes 14.9 s (16.0x; measured one revision earlier, before the GAE trace): more, smaller mini-batches are slightly slower.

The remaining time is the `update`: 74-89 % of an iteration, compute / bandwidth bound (section 4).

## 3. What the trace buys

Eager TT (same code, `trace` off) vs traced, 4096 envs: [512, 256, 128]: 833 -> 447 ms (**1.87x**; update 715 -> 332, act 91 -> 79); [1024, 1024, 512]: 901 -> 913 ms (**1.0x**).
A trace removes the per-op host dispatch (about 550 ops per mini-batch step, ~170 per GAE, ~120 per act step; one replay of the act graph is 0.8 ms against ~10 ms eager). It pays where the device
is dispatch bound (small batches or small networks) and does nothing where the device is busy (large hidden sizes). The traced path is bit-identical to the eager one (`tests/tt/test_tt_trace.py`).

Earlier Phase 5 steps (4096 envs, [512, 256, 128], algorithm only): CPU 3.2 s; TT untraced 2.34 s; exact bf16-split `embedding` shuffle instead of `gather` (T-12) 1.05 s; persistent state + trace,
host-side TILE -> ROW_MAJOR upload change, composed ELU backward, traced GAE: 0.45 s.

## 4. Where the time goes

Iteration breakdown (`--profile`: every section is bracketed by a device synchronisation, so the sum is the iteration; ms per iteration):

| section | 4096 envs | share | 65536 envs | share |
|---|---:|---:|---:|---:|
| `update`: mini-batch step replays (20) | 315 | 66 % | 4310 | 75 % |
| `process_env_step` (H2D reward/done, 24x) | 36 | 8 % | 130 | 2 % |
| `act`: obs + noise upload (H2D, 24x) | 32 | 7 % | 384 | 7 % |
| `act`: compute (trace replay, 24x) | 20 | 4 % | 262 | 5 % |
| `act`: download actions (D2H, 24x) | 18 | 4 % | 187 | 3 % |
| `update`: shuffle gather | 12 | 3 % | 183 | 3 % |
| `update`: normalizer update | 6 | 1 % | 27 | 0.5 % |
| `compute_returns` (GAE trace) + `pack_table` | 6 | 1 % | 75 | 1 % |
| total | 474 | | 5767 | |

(Real `tracking12` env step: 38 ms / 391 ms on top, if it runs on the same host.) Host <-> device traffic per iteration: 4096 envs 90 MB up (observations, noise, reward/done, permutation) and
12.6 MB down (actions); 65536 envs 1.44 GB up and 0.20 GB down. Only the allow-listed transfers occur (H-1..H-8, counted in every JSON; nothing else crosses).

Op profile of one eager mini-batch step (`scripts/profile_ops.py`, every ttnn op bracketed by a device sync, warm program cache; 65536 envs = 393216 rows per mini-batch, [512, 256, 128]):
step 216 ms. `matmul` 14 calls 59 ms + `linear` 8 calls 32 ms (91 ms, 28 %; ~10 TFLOP/s fp32), `multiply` 149 calls 54 ms, `square` 40 calls 21 ms, `exp` 7 calls 21 ms, `elu` 6 calls 19 ms,
`minimum` 9 calls 19 ms, `add` 97 calls 14 ms, ... A full-activation element-wise pass over a 393216 x 512 fp32 tensor takes ~3 ms (read + write 1.6 GB: DRAM bandwidth bound). For
[1024, 1024, 512] at 16 mini-batches: step 166 ms, `matmul` 56 + `linear` 43 ms. The hardware profiler was not used (its environment variables belong to `scripts/tray_env.sh`).

Optimisations found by these measurements (all keep results bit-identical to the previous revision or within the existing tolerances):
* exact fp32 shuffle through 3 bf16 pieces + `ttnn.embedding` (T-12): 40x on the gather, 2.34 s -> 1.05 s per iteration at 4096 envs;
* persistent device state + one captured trace per mini-batch step, per act step and for the GAE (T-15, T-16);
* ROW_MAJOR host uploads, tilized on the device inside the trace: the host-side TILE conversion cost 25 ms per 65536 x 64 fp32 upload against 5 ms for the write itself
  (`probe_upload_device.json`): act upload 74 -> 16 ms per step at 65536 envs;
* `ttnn.elu_bw` replaced by `g * exp(min(z, 0))`: `elu_bw` took 15.5 ms per call (29 % of an update step, 5x its own forward); step 257 -> 216 ms, and the 65536 x [1024, 1024, 512]
  configuration now also fits with the upstream default of 4 mini-batches (it ran out of DRAM before).

## 5. Unfavourable regimes

| num_envs | hidden | CPU ms (32 threads) | CPU best ms (threads) | TT ms | TT vs CPU best |
|---:|---|---:|---:|---:|---:|
| 256 | [64, 64] | 141 | 113 (1) | 99 | 1.1x |
| 256 | [512, 256, 128] | 266 | 266 (32) | 125 | 2.1x |
| 1024 | [64, 64] | 210 | 211 (32) | 133 | 1.6x |
| 1024 | [512, 256, 128] | 742 | 696 (32) | 185 | 3.8x |

* The device has a FIXED cost per iteration of ~100 ms (256 envs, [64, 64]: 24 act round trips of ~0.8 ms, 24 `process_env_step` of ~1.2 ms, 20 replayed update steps of ~2.5 ms): at 256 envs with a
  [64, 64] network the TT iteration is at break-even with a single CPU thread (99 vs 113 ms), and a smaller problem would lose to the CPU. Training small tasks on the device is not worth it.
* Per-step latency matters if the environment is remote: each environment step needs one act round trip (upload obs + noise, replay, download actions) of ~3 ms at 4096 envs and ~35 ms at 65536 envs
  (section 4, divided by 24).
* Device memory (34 GB) bounds the batch: the rollout slots, the packed table and the shuffled table scale with `num_envs x steps x 256 columns x 4 B` (1.6 GB each at 65536 envs) and the
  mini-batch activations with `rows x hidden`; 65536 x [1024, 1024, 512] with 4 mini-batches is the largest configuration that was run and fits.
* The first iteration of a run costs more than the later ones (eager first steps, program cache, trace captures): 1.8 s against 0.45 s at 4096 envs with the kernels already built
  (a cold kernel cache in `.run/ttnn_cache` adds the JIT compilation; not measured separately).

## 6. Precision experiments (BRIEF: "only if learning parity holds")

* Math fidelity of matmuls / reductions, HiFi4 (default) vs HiFi2 vs LoFi, [512, 256, 128]: 4096 envs 523 / 517 / 516 ms, 65536 envs 6309 / 6373 / 6263 ms: **no speed-up** (1.00-1.01x). The update is not
  limited by the multiply passes, so lowering the fidelity only costs accuracy (matmul rel error 1.2e-3 -> 7.2e-3 -> 3.1e-2, TOLERANCES.md); the learning-parity runs were therefore not made.
  Option: `tt_cfg["math_fidelity"]`.
* bf16 activations / parameters were NOT tried: element-wise passes over the activations are DRAM-bandwidth bound (section 4), so bf16 storage could shorten them, but it changes every tolerance
  of TOLERANCES.md and needs dtype plumbing through all modules and a learning-parity study; it is the main remaining lever for the update (upper bound: about 2x on the bandwidth-bound part).
  Not part of v1 (fp32 policy, BRIEF 3.4).

## 7. Phase 5 criteria

* act, update and GAE are traced with preallocated persistent buffers (T-15: every tensor that survives a replay is allocated before the first capture, `TTPPO` refuses to be built while another
  algorithm's captured traces are alive on the mesh); traced == eager bit for bit on cartpole and tracking12 (4 iterations, all device state), after `load()`, and the rollout storage is not
  overwritten by replays (`tests/tt/test_tt_trace.py`, 5 tests).
* The Phase 2-4 suites pass again (157 device tests on the eager path, 360 host tests) and the Phase 4 end-to-end criterion is met on the TRACED path: `docs/porting/tt_runs_trace/`
  (3 envs x 3 seeds): cartpole 500.0 (golden 500.0, threshold 475.0), pendulum -152.2 (-152.1, -159.7), tracking12 191.6 (190.7, 181.2), no NaN/Inf; wall time per run 14 s / 36 s / 77 s
  (eager TT 68 / 143 / 322 s, upstream CPU 20 / 50 / 230 s).
* Iteration breakdown, op profile, the benchmark matrix with the unfavourable regimes and the precision experiments: sections 2-6.

## 8. Reproduce

```bash
scripts/run_phase5_final.sh                      # everything below, fail-fast, device jobs one after the other (~25 min)
scripts/run_bench_matrix.sh                      # TT traced matrix: num_envs {4096, 16384, 65536} x hidden {[512,256,128], [1024,1024,512]}
scripts/run_bench_cpu.sh                         # upstream CPU matrix + real env cost (host only)
scripts/run_bench_small.sh                       # small regimes (TT + CPU) and eager TT at 4096 envs
scripts/tt_run.sh --chips 16 --timeout 900 -- python scripts/profile_ops.py --envs 65536 --hidden 512 256 128      # op profile
scripts/tt_run.sh --chips 16 --timeout 900 -- python scripts/bench_matrix.py --backend tt --trace --profile --envs 4096 --hidden 512 256 128   # section profile
scripts/tt_run.sh --chips 16 --timeout 3600 -- python scripts/train_tt.py --trace --out docs/porting/tt_runs_trace  # end-to-end on the traced path
```

## 9. Not done / next levers

* `process_env_step` (about 8 % at 4096 envs) is eager (~10 ops per step); it could join the act trace of the next step.
* The critic's observations contain the actor's observations (36 of 84 columns): uploading the critic's observations once and slicing the actor's on the device saves ~35 % of the upload.
* Real overlap of the host work (noise draw, staging) with the device (the loop is synchronous at the action download, one sync per environment step by design, H-2).
* bf16 activations (section 6). Multi-chip data parallel: Phase 6.
