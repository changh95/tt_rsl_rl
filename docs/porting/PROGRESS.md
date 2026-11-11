# Progress log

Newest entry first. Each entry: date, phase, done / next / blocked / key numbers, related commits.
Read order for a new session: `BRIEF.md` -> `CLAUDE.md` -> this file.

## 2026-11-11 — Demo 2 follow-up (branch `tt/demo-g1-gait`, not merged): `examples/g1_walk/`, initial / middle / final video, eight-chip scaling profile
**Done**: (1) `examples/` (`README.md`, `g1_walk/{train.py, play.py, README.md}`): the G1 demo as a short example on top of the public rsl_rl API (`MujocoG1VecEnv` + `g1_train_cfg` + `tt_agent_cfg` + the upstream `OnPolicyRunner`); `train.py` smoke-tested on one chip
(3 iterations, 256 robots), `play.py` on its checkpoint. The evaluation / rendering code moved from `scripts/render_g1_demo.py` into the library module `rsl_rl_tt/demos/g1_render.py` (the scripts are thin wrappers); `tests/tt/test_g1_render.py` (8 host tests: labels, checkpoint
load path == upstream inference policy exactly, measurement, tiny EGL render, video / GIF writers, `--help` of the four scripts). (2) New comparison video / GIF `docs/porting/demo_g1/g1.{mp4,gif}`: untrained / iteration 101 / iteration 1500 (before: 201 instead of 101; iteration 101
keeps its balance but does not move (0.00-0.01 m/s, 0 falls), 201 already follows the commands, so 101 is the stage that still looks different); `eval.json` now has the four checkpoints (256 robots x 600 steps each). (3) `gait.mp4`: slow-motion two-view close-up of the final gait with foot contacts
(`scripts/render_g1_gait.py`). (4) Profile of one iteration, one chip vs eight (`train_g1_demo.py --profile`, `profile_1chip/`, `profile_8chips/`): CPU simulation 830 / 845 ms, PPO update replay 308 / 112 ms (2.75x with 8x the chips), acting 56 / 42, step processing 28 / 34, iteration 1290 / 1123 ms (1.15x);
upper bound with free chips 1.55x (Amdahl); explanation in `docs/porting/demo_g1/README.md` ("Why eight chips are not eight times faster"). **Next**: nothing pending; merge to `main` only if the owner asks (not a phase).

## 2026-11-12 — Demo 2 (not a phase; merged from branch `tt/demo-g1`): Unitree G1 walking on command, CPU simulation + training on ALL EIGHT chips
**Done**: `rsl_rl_tt/demos/mujoco_g1.py` (G1 MJX model of MuJoCo Menagerie, pinned commit, fetched by `scripts/fetch_g1_model.sh`: velocity-command task, 12 leg joints, privileged critic observations, phase-based gait rewards;
4096 robots stepped by MuJoCo's multithreaded `rollout`), `g1_cfg.py`, `scripts/train_g1_demo.py` / `render_g1_demo.py` / `plot_g1_curves.py`, 9 host tests (`tests/tt/test_mujoco_g1.py`: conventions, determinism, geometry, rollout == `mj_step` EXACTLY
with the warm start and within 2.6e-3 rad cold-started, fall / time-out / reset, commands, private RNG, short upstream-PPO run with the asymmetric critic). Result (`docs/porting/demo_g1/README.md`): 2x4 mesh, 4096 envs, 1500 iterations
(147 M samples, 27.6 min, replicas bit-identical at every 10th update): follows 0.4 / 0.8 m/s and a turn command (0.39 / 0.78 / 0.51 m/s + 0.38 rad/s achieved) with 0 falls in 12 s x 256 robots; video `g1.mp4` (20 s), curves.
Per iteration: upstream CPU PPO 3.93 s (algorithm 3.05), TT one chip 1.35 s (0.51), TT eight chips 1.10 s (0.27): the CPU simulation is 76 % of the eight-chip run.
**Two failed attempts, recorded**: (1) the model's PD gains (ankle kp 20, roll range 0.26 rad => ~5 N*m) made stepping physically impossible; (2) with a sparse contact reward the policy stood still for 450+ iterations; fixed by raising the leg / waist gains
and a dense foot-height-vs-phase reward. One seed; simulation only (no randomisation, no noise). Merged to `main` on 2026-11-12 at the owner's explicit request (not a phase, so no `phase-N-done` tag).

## 2026-11-12 — Demo (not a phase; merged from branch `tt/demo-humanoid`): MuJoCo humanoid walking, CPU simulation + TT training
**Done**: `rsl_rl_tt/demos/` (`MujocoHumanoidVecEnv`: Gymnasium Humanoid-v5 task as an rsl_rl `VecEnv` with Isaac Lab conventions, thousands of environments stepped by MuJoCo's multithreaded `rollout`, 165 k env-steps/s on 64 threads;
`humanoid_cfg.py`), `scripts/train_humanoid_demo.py` / `render_humanoid_demo.py` / `plot_humanoid_curves.py`, 8 host tests (`tests/tt/test_mujoco_humanoid.py`: conventions, determinism, reward formula, rollout == plain `mj_step` to 5e-8,
fall / time-out / auto-reset, private RNG, a short upstream-PPO training), dependency `gymnasium[mujoco]==1.4.0` (ENV.md). Result (`docs/porting/demo_humanoid/README.md`): TT, one chip, 4096 envs, 600 iterations (59 M samples, 13.3 min):
the humanoid stands up at iteration ~121 and then runs at 2.75 m/s (deterministic evaluation 2.9 m/s, 0 falls in 256 episodes); upstream PPO on the CPU learns the same way (same curve against the iteration) at 3.73 s per iteration against 1.33 s
(training part 2.78 s vs 0.43 s). The CPU simulation is 68 % of the TT run. One seed. Merged to `main` on 2026-11-12 at the owner's explicit request (not a phase, so no `phase-N-done` tag). Video: `docs/porting/demo_humanoid/walk.mp4` (15 s, real time, 1.3 MB, linked from both READMEs; `render_humanoid_demo.py --video`, dependency `imageio-ffmpeg==0.8.1`).

## 2026-11-12 — Phase 7 DONE on the TT host: environment bridge, scripts and runbook; Isaac Lab itself NOT run (no GPU host / Isaac Lab available)
**Done**
- `rsl_rl_tt/bridge/` (ADR-0003 plan as accepted: ZeroMQ + raw float32 buffers): `protocol.py` (JSON header + raw buffers, numpy only), `server.py` (`serve(env, endpoint, agent_cfg=...)` for any rsl_rl-style env, e.g. Isaac Lab's
  `RslRlVecEnvWrapper`; no rsl_rl / ttnn import, so it runs in an Isaac Lab Python of another version), `client.py` (`RemoteVecEnv`, an rsl_rl `VecEnv`: dones long, bool time_outs, episode logs, settable
  `episode_length_buf`, `TimeoutError` instead of a hang), `config.py` (`tt_agent_cfg`: the agent configuration of the environment host with only the TT-defining keys changed, v5 format required). Dependency: `pyzmq==26.4.0` (ENV.md).
- Scripts: `isaac_env_server.py` (`--local-env` loopback, VERIFIED; `--task` Isaac Lab mode, written against the public Isaac Lab v2.3.0 `train.py` / `RslRlVecEnvWrapper` and NOT run), `train_isaac.py` (TT host: connect, take the
  configuration from the server, train on 1 / 2 / 8 chips, CSV + upstream-format checkpoints), `bridge_smoke.py` (connection check against the real server), `tb_to_csv.py` + `compare_curves.py` (GPU reference vs TT: the Phase 4
  criterion and a figure), `bench_bridge.py`. Runbook for the GPU host: `docs/porting/ISAAC_RUNBOOK.md` (Isaac-Cartpole, a locomotion task, TT checkpoint replay in `play.py`).
- Evidence: an environment behind the bridge is indistinguishable from the same one in process (`tests/tt/test_bridge.py`, 14 host tests: obs / rewards / dones / time-outs / logs step by step over 350 steps, upstream PPO training:
  every logged scalar; `tests/tt/test_tt_bridge_device.py`: TTPPO every scalar and every parameter, cartpole and tracking12). User-facing path on loopback: `isaac_env_server.py --local-env tracking12` + `train_isaac.py` on one chip,
  traced, 300 iterations, configuration received from the server: identical to the Phase 5 in-process run in every logged scalar, golden criterion met (`docs/porting/isaac_loopback/`, committed and tested), 83 s vs 77 s.
- Measured on loopback (PERF.md 7c): 0.6-0.9 ms per step up to 4096 environments, 8.9 ms at 65536 (25.6 MB per step, 2.9 GB/s); pickling the tensors is 2.6-8x slower; raw TCP floor 0.04-5 ms. gRPC / Arrow Flight not measured (extra
  dependencies on the Isaac Lab Python; the transport costs 5 % of an iteration already).
**Not done (needs the owner / the GPU host)**: ISAAC_HOST is still unknown. Running the Isaac mode of the server, the Isaac-Cartpole and locomotion comparisons against GPU reference runs and the replay of a TT checkpoint
in Isaac Lab's `play.py` follow the runbook; its first step (`bridge_smoke.py`) shows within a minute whether the server matches your Isaac Lab. Isaac Lab v2.3.0 trains with the rsl_rl 3.x configuration format; the TT backend needs 5.x
(Isaac Lab `main`): `tt_agent_cfg` says so explicitly if it receives the old one.
**v1 status**: all seven phases are done on the TT host (tags `phase-0-done` ... `phase-7-done`).

## 2026-11-11 — Phase 6 DONE: tray multi-chip data parallel, 1x2 and 2x4 (tag `phase-6-done`)
**Done**
- `TTPPO` runs on a mesh (`tt_cfg["mesh_shape"]` = [1, 2] or [2, 4], fabric FABRIC_1D set automatically): the batch is sharded over the chips; parameters, Adam state and the learning rate are replicated; every chip computes the
  loss / gradient of ITS mini-batch, the gradients are averaged over the chips BEFORE the per-model clip (`TTContext.mean_over_chips` = `all_gather` along every mesh axis + fp32 `ttnn.sum`, ADR-0004 A-1; inside the traces),
  the KL of the adaptive schedule is the mean over the chips, observation-normalizer statistics are GLOBAL (upstream torch.distributed formula), advantage normalisation and shuffling are chip-LOCAL (an independent
  permutation of every chip's own rows), `replica_check_interval` verifies that all chips hold bit-identical parameters / Adam moments / normalizer statistics / learning rate. Single-chip path unchanged (Phase 2-5 suites pass).
- Tests (`tests/tt/test_tt_multichip.py`, 12 per mesh; `scripts/run_phase6_final.sh`): exactly the selected nodes open (all eight on 2x4); cross-chip sum/mean 1.1e-7 vs float64 and bit-identical on every chip; global normalizer
  statistics vs upstream on the union; per-chip shuffle bit-exact; per-chip advantage statistics; **N chips vs 1 chip**: three mini-batch steps on the union mini-batches (loss -> gradient exchange -> clip -> Adam -> mean-KL
  adaptive LR) deviate by 1.3e-3 (actor) / 6.2e-4 (critic) of the update size on BOTH meshes (limit 2e-2 fixed beforehand), learning-rate sequence identical; replicas bit-identical after training, traced and eager, and a deliberately
  corrupted replica is detected; traced == eager bit for bit on 2 and 8 chips (also after `load()`). The chip-count independent suites (forward, rollout, checkpoint, normalizer: 84 tests) and `test_tt_trace.py` (5) pass on
  1x2 and 2x4; single chip 157 + 5; host tests 399.
- Learning on the meshes (`docs/porting/tt_runs_mesh_1x2/`, `tt_runs_mesh_2x4/`, traced, 3 envs x 3 seeds, replicas checked every 10 updates): Phase 4 criterion met on both: cartpole 476.8 (1x2) / 499.9 (2x4) [golden 500.0, threshold
  475.0], pendulum -151.8 / -151.8 [-152.1], tracking12 191.1 / 193.4 [190.7]. The cartpole 1x2 mean is close to the threshold because seed 0 lost the policy late (value-loss blow-up at the learning-rate cap); this happens on ONE
  chip as well: over seeds 3-20, 2/18 (1 chip), 5/18 (1x2), 1/18 (2x4) collapsed after reaching 500, no trend with the chip count (Fisher p 0.40 / 1.0).
- Scaling (`docs/porting/PERF.md` 7b): 8 chips are 2.0-2.2x faster than one at 4096 envs and 5.2-5.3x at 65536 envs (1.47 M env-steps/s for [512,256,128]; 1.07 s / 2.38 s per iteration against 73 s / 238 s for upstream CPU = 69x / 100x).
  What limits small batches is the host loop (one act + process_env_step per environment step) and the gradient exchange.
- Findings: **T-14 corrected** (the bit-identical-rows property holds for the same chip count only: with different rows PER CHIP the matmul accumulation order changes, max 1.2e-4 vs 1.7e-3 noise floor; a first version of the
  N-vs-1 test asserted bit equality and was corrected by this measurement, not by loosening a tolerance); **T-17** (CCL inside a trace works). The 1-chip equivalence reference is a fixture (`tests/tt/data/mc_step_reference.npz`;
  `.npz` because `*.pt` is git-ignored on purpose).
**Next**: Phase 7 (Isaac Lab integration): ISAAC_HOST is unknown, so the env bridge (ZeroMQ + float32 raw buffers, ADR-0003) is built and validated on the same host with a loopback server wrapping the CPU test env first;
runbook and scripts for the GPU host (Isaac-Cartpole -> one locomotion task), TT checkpoint replay in Isaac Lab play.

## 2026-11-11 — Phase 5 DONE: 1-chip performance, traced path (tag `phase-5-done`)
**Done**
- act step, mini-batch update step and GAE are captured once as Metal traces and replayed, with persistent in-place device state. Traced == eager BIT FOR BIT (cartpole, tracking12, 4 iterations, all
  device state; after `load()`; `tests/tt/test_tt_trace.py`, 5 device tests). Phase 2-4 suites pass again (157 device tests, 373 host tests) and the Phase 4 criterion is met on the TRACED path
  (`docs/porting/tt_runs_trace/`, 3 envs x 3 seeds): cartpole 500.0 (golden 500.0), pendulum -152.2 (-152.1), tracking12 191.6 (190.7), no NaN/Inf; 14 / 36 / 77 s per run (eager 68 / 143 / 322 s, CPU 20 / 50 / 230 s).
- Benchmark matrix (null env, algorithm cost per iteration, TT traced vs upstream CPU 32 threads): num_envs 4096 / 16384 / 65536 x hidden [512,256,128] / [1024,1024,512]: 7.3x / 13.6x / 13.3x and
  17.7x / 17.4x / 19.0x faster (0.45 s vs 3.3 s at 4096 x [512,256,128]; 12.5 s vs 238 s at 65536 x [1024,1024,512]); 105-285 k env-steps/s. Unfavourable regimes: break-even with one CPU thread at 256 envs x [64,64].
  Full tables, iteration breakdown, op profile, precision experiments (HiFi2/LoFi: no speed-up, so no learning-parity runs; bf16 not tried), reproduce commands: `docs/porting/PERF.md`.
- Optimisations found by profiling: exact bf16-split `embedding` shuffle (T-12, 40x), ROW_MAJOR uploads tilized on the device (act upload 74 -> 16 ms/step at 65536 envs), `elu_bw` replaced by
  `g * exp(min(z,0))` (15.5 ms per call, 29 % of an update step), persistent state + traces.
- Findings recorded: **T-15** (a trace's intermediates are freed after capture but rewritten by every replay: tensors allocated afterwards that keep a value across replays are silently overwritten) with the
  persistent-state rule and a construction guard in `TTPPO`; **T-16** (capture preconditions). Incident of the first traced run (rule 2-6, capture left open -> timeout): record in `docs/porting/incidents/`; the chip was
  verified with the open/close smoke afterwards, no reset was ever needed.
- CI of this fork switched off at the owner's request (workflows run by hand only; `gh workflow` / repo settings stay out of scope, rule 2-10).
**Next**: Phase 6 (tray multi-chip data parallel: 1x2 -> 2x4, synchronous gradient/statistics exchange by all_gather + sum, ADR-0004 A-1; the normalizer's multi-chip statistics and the batch-sharded
tensors of the traced path are the open items; the persistent-state rule T-15 applies to every new buffer).

## 2026-11-11 — Phase 4 DONE: 1-chip end-to-end training and checkpoint compatibility (tag `phase-4-done`)
**Done**
- TT training of all three envs x 3 seeds (same config/seeds/iterations as the golden runs) with the unmodified `OnPolicyRunner` + `TTPPO` on one chip: `docs/porting/tt_runs/` (CSVs, `summary.json`,
  `<env>_vs_golden.png`). Phase 4 criterion met on every env, no NaN/Inf anywhere: cartpole 500.0 x3 (golden 500.0, threshold 475.0), pendulum -154.2 (golden -152.1, threshold -159.7), tracking12 193.7 (golden 190.7,
  threshold 181.2). Wall time per run 68 s / 143 s / 322 s (golden CPU 20 / 50 / 230 s), total 27 min.
- Checkpoint compatibility (7 device tests): TT save -> unmodified upstream `load()` gives bit-identical inference and an optimizer state with the exact structure of an upstream checkpoint; upstream save -> TT `load()`
  reproduces weights, normalizer statistics, Adam moments (exp_avg compared tensor by tensor), step counter and learning rate on the device, and TT continues training; a partial load (no optimizer) keeps the current
  moments like upstream; a TT-trained policy exports to TorchScript and ONNX through the runner's own export methods and onnxruntime reproduces it (1e-5), consistent with the device forward.
- `TTPPO.load()/save()` with device<->torch Adam-state conversion; `get_policy()` no longer downloads Adam (cheap per-iteration logging); `onnxruntime==1.23.2` added to the venv (ENV.md).
- Host tests: 360 pass (incl. the committed TT runs); device tests: 156 in two single-process invocations.
**Next**: Phase 5 (1-chip performance): trace for act / update / GAE with preallocated buffers, fewer syncs, the shuffle (T-12: the fp32 row gather is ~50 ns/element), iteration-time breakdown and op profile,
benchmark matrix (num_envs 4096/16384/65536 x hidden [512,256,128]/[1024,1024,512]), `PERF.md`.

## 2026-11-11 — Phase 3 DONE: 1-chip backward, Adam, PPO update, device storage/GAE (tag `phase-3-done`)
**Done**
- `rsl_rl_tt/losses.py` (clipped surrogate/value losses and their gradients with torch's `max`-tie and inclusive-`clamp` rules, entropy bonus through the std, masked padded dims, KL),
  `optim.py` (`TTAdam`, per-model `clip_grad_norm` = C-1, device-evaluated `TTAdaptiveLR`), `storage.py` (device rollout storage, GAE, packed table + ONE permutation gather,
  mini-batch slices), `modules.py` (`forward_train` / `backward` of the MLP, Gaussian backward rows), `ppo.py` (`TTPPO.act/process_env_step/compute_returns/update`, `load()` and
  device<->torch Adam-state sync for checkpoints). The unmodified `OnPolicyRunner.learn()` now trains end to end with TT inference and TT updates on all three envs.
- Parity (149 device tests in one process, 347 host tests): per-parameter gradients vs autograd, loss formulas given identical outputs (<= 6e-6), MLP VJP for ELU/tanh/ReLU (<= 4.9e-3, ReLU 3.7e-2),
  Adam vs `torch.optim.Adam` (1.9e-8), clip_grad_norm (6.8e-8), GAE (bit exact), shuffled mini-batches vs upstream's generator (bit exact), whole `update()` from identical state, 20-update drift.
- Allow-list: H-7 retired; the update path moves only the permutation (H-4), logged scalars + std (H-5) and Adam/normalizer scalars (H-8): asserted exactly in `test_update_host_traffic_*`.
**Key numbers** (TOLERANCES.md has the table): whole update from identical state deviates 4e-4..1.1e-3 of its size with a smooth objective (noise baseline 8e-5..4e-4), <= 2.5e-2 with the clipped
objective (discontinuous clip gates); 20 consecutive updates drift <= 1.0e-2 (smooth) / <= 6.9e-2 (clipped) of the cumulative update. Iteration time (untraced): cartpole 1024x24: TT 675 ms (update 587) vs CPU 177 ms
(update 145); tracking12 2048x24: TT 1094 ms vs CPU 734 ms.
**Findings**: Adam makes parameter-wise comparison meaningless (function-space metric + gradient-noise baseline instead); the device KL of the first mini-batch is exactly 0 (upstream ~1e-9), so adaptive-LR
parity is tested separately; fp32 row gather is ~50 ns/element (T-12) and its row-major variant returns wrong data; `ttnn.sum` without dim is rank-0 (T-13).
**Next**: Phase 4 (1-chip end-to-end learning + compatibility): train TT on the three envs x 3 seeds against the golden curves, checkpoint round trips (TT save -> upstream load -> same outputs; upstream save -> TT resume;
optimizer state; JIT/ONNX export), NaN/Inf checks, plots.

## 2026-11-11 — Phase 2 DONE: 1-chip forward on Blackhole (tag `phase-2-done`)
**Done**
- `rsl_rl_tt/tt_common.py` (tile padding, compute config, the ONLY host<->device transfer helpers with per-allow-list-id counters; unknown ids raise),
  `rsl_rl_tt/modules.py` (device-side `TTLinear`/`TTMLP`/`TTMLPModel`, `TTEmpiricalNormalization` with the upstream Welford recurrence and a device-tensor rate,
  `TTGaussianDistribution` with scalar/log std, clamping, masked log_prob/entropy; no torch import), `rsl_rl_tt/ppo.py` (`TTPPO(PPO)`, injected via
  `cfg["algorithm"]["class_name"] = "rsl_rl_tt.ppo:TTPPO"` into the UNMODIFIED `OnPolicyRunner`; upstream models stay as mirrors), `device.acquire_tray_mesh`.
- Padding: all device dims are multiples of 32, zero padded (exact: padded outputs are 0), action-dim padding masked in log_prob/entropy; num_envs and
  mini-batch sizes that are not multiples of 32 per chip are refused with a clear error (tested).
- Allow-list: H-8 added (per-iteration scalars as device tensors). Everything on the TT path is counted; H-7 (host storage/GAE) is the Phase 2 temporary.
- Tests: 77 device tests in one process (56 component + 21: random-shape sweep with padding, checkpoint load + resync, teacher-forced rollouts on all three envs
  against deep copies of the upstream models, exact host-op counts, alignment/config refusals) and 4 new host invariants (347 host tests pass).
- Measured composite tolerances (`TOLERANCES.md`): MLP forward rel_fro 2.5e-3..3.5e-3 (critic 7.7e-3; tolerance 1.5e-2 with a 0.1 absolute floor), normalizer
  forward 2 ULP, update 2.8e-7, Gaussian head 1e-7..7.7e-8.
- Full-size rollouts with TT inference end to end (`scripts/bench_rollout.py`, `measurements/phase2_rollout_timing.json`): act 2.7 / 2.7 / 3.9 ms per step
  (cartpole 1024 envs / pendulum 1024 / tracking12 2048) vs upstream CPU 0.8 / 1.0 / 2.8 ms; rollout 73 / 96 / 120 ms. Untraced, with 4 downloads per step (H-7):
  this is the Phase 5 baseline, not a result.
**Findings**: `fast_and_approximate_mode=False` is rejected for fp32 binary ops (TTNN_ISSUES T-10); accurate fp32 `mean`/`sum` need `fp32_dest_acc_en` in the
config (T-11); relative error of a small-output network needs an absolute floor (a random-init critic has output rms 0.06).
**Next**: Phase 3 (1-chip backward, optimizer, PPO update): hand-written VJPs, PPO losses, per-model grad-norm clipping (C-1), device Adam with device-tensor
scalars, adaptive LR, device RolloutStorage/GAE/mini-batches; gradient parity per parameter vs upstream autograd; Adam step; full `update()`; 20-iteration drift.

## 2026-11-11 — Phase 1 DONE (host only): CPU test envs, golden baselines, parity utilities
**Done**
- Owner approved ADR-0001/0002/0003/0004(A-1) and Phase 1 (2026-11-10); Phase 0 merged to `main`, tag `phase-0-done`.
- `rsl_rl_tt/testing/envs.py`: three seed-deterministic batched `VecEnv`s with TensorDict obs groups, Isaac Lab wrapper conventions
  (`dones` long, `time_outs` bool, obs after reset, `extras["log"]` on reset steps, private RNG so PPO's global RNG is untouched):
  `cartpole` (early termination, 500-step time-out), `pendulum` (gym swing-up, 200 steps), `tracking12` (12 double-integrator actuators,
  command tracking, critic-only privileged domain-randomisation group, 300-step time-out AND early termination). 4096 envs step in 0.3-2 ms.
- `rsl_rl_tt/testing/reference.py`, `csv_writer.py`: the UNMODIFIED upstream `OnPolicyRunner` is configured per env and a `LogWriter` subclass
  (injected through `cfg["logger"]`, no upstream patch) records every Logger scalar per iteration; `scripts/train_reference.py` produces the goldens.
- Golden baselines (`docs/porting/golden/`, 3 seeds each, CSV + `summary.json` + curve PNGs): cartpole 500.0 +- 0.0 in ~20 s/run, pendulum
  -152.1 +- 2.4 in ~50 s, tracking12 190.7 +- 1.4 in ~230 s (random policies: 25.5 / -1224 / 14.9). Training is bit-reproducible for a given
  seed/torch build/thread count (tested).
- `rsl_rl_tt/testing/parity.py`: max abs/rel error, rel_fro, PCC, fp32 ULP, exact mode, non-finite handling, worst-element report; the registry
  `TOLERANCES` mirrors `TOLERANCES.md` (a test checks the numbers appear in the table). `golden.py` encodes the Phase 4 acceptance rule.
- Host tests: 343 pass (envs 24, parity 10, reference 10, golden 16, plus the Phase 0 suite).
**Findings worth knowing**
- A first `tracking12` reward (tiny tracking kernel, only penalties otherwise) taught the agent to terminate early; fixed with a wider kernel and a
  termination penalty (as in Isaac Lab tasks). With actions clipped by the env, an entropy bonus inflates std for free, so `entropy_coef = 0` there.
- The upstream `Logger` stores a git diff of the whole repo (`git/rsl_rl.diff`), not of `rsl_rl/` only; the "rsl_rl equals the tag" check stays the
  invariants test.
- Reference specs already satisfy the Phase 2 tile rule (batch and mini-batch multiples of 32), tested.
**Next**: Phase 2 (1-chip forward): TT `MLPModel` forward, `EmpiricalNormalization`, `GaussianDistribution`, torch->TT weight copy, padding path, `act()` wired
to the CPU env rollouts (rollout storage may stay on the host in this phase: H-7). First device job of the project phase uses `tt_run.sh --chips 16`.

## 2026-11-10 (later) — Phase 0 DONE, waiting for the owner's approval before Phase 1
**Done**
- Owner decisions received: ADR-0005 delegated to the assistant ("do not disturb the other trays") -> option II implemented; ADR-0006 wheel;
  C-1 two grad norms; C-2 latest = v5.5.2 (newest release tag; upstream `main` is +4 untagged commits).
- ADR-0005 II built: `tt_run.sh` -> `unshare --user --map-root-user --mount` -> `scripts/ns_mask_dev.sh` (tmpfs + bind of the selected chips only,
  moved over `/dev/tenstorrent`); `device.py` check (e) refuses to continue unless the visible nodes equal the selected chips. 18 new host tests.
- ADR-0006 done: `ttnn==0.81.1` + sfpi 7.89.0 (sha256-verified) in `.venv`; private caches via `scripts/ttnn_env.sh`.
- First device runs (all through `tt_run.sh`, one job at a time, tray 3 only): 1 chip OK; 8 chips as a 2x4 mesh OK; `FABRIC_1D` + `all_reduce` OK;
  the fd guard saw exactly the selected nodes each time. No error, hang or reset.
- Noise floors measured (`TOLERANCES.md`, `docs/porting/measurements/`), ADR-0001 evidence completed, ADR-0004 amendment A-1, `TTNN_ISSUES.md` T-1..T-9.
- Hook live in this session (canary), `.claude/**` and `scripts/tray_env.sh` untouched since the owner lock.
**Key numbers**
- fp32 matmul (HiFi4 + fp32 acc): rel_fro 1.17e-3 vs fp64 (torch fp32 2.4e-7); no split/chunking gets below 3.1e-4. Everything else <= ~1e-7 (eltwise <= 2 ULP).
- `ttnn.all_reduce` fp32: 5.5e-4; `all_gather` (2 axes) + `ttnn.sum`: 5.1e-8, bit-identical on all 8 chips, 0.3-2.9 ms. `ttnn.var`: 2.4e-4 -> use two-pass (3.7e-8).
- Mesh: 2x4, 8 chips, open 1 s (warm). Host tests: 283 pass.
**Next (after approval)**: Phase 1 (host-only reference + parity harness, golden curves from upstream with the CPU test env), then Phase 2.
Open for the owner: ADR-0001/0002/0004 (+A-1) approval, ISAAC_HOST (ADR-0003), Isaac Lab's rsl_rl version (C-2 check).

## 2026-11-10 — Phase 0 (in progress; all host-side work done, STOPPED before any device open)
**Done**
- Repo `changh95/tt_rsl_rl` (public, non-fork) seeded from a full-history clone of upstream; `main` = tag `v5.5.2` (pristine),
  upstream tags pushed, `upstream` push URL disabled, work branch `tt/phase-0`.
- Section-0 values resolved: tray 3 = `/dev/tenstorrent/16..23` = BDFs `0000:c1:00.0..c8:00.0`; PYTHON_ENV = `<repo>/.venv`
  (Python 3.10.12, torch 2.11.0+cpu, numpy 1.26.4); `RSL_RL_VERSION = v5.5.2` (provisional).
- Safety net, complete and tested on the host (265 tests passing, no device involved):
  - `scripts/tray_env.sh` (single source of device env; BDF tokens; `--chips` subset; refuses chips outside 16..23 and empty
    selections), `scripts/ttnn_env.sh` (software env only), `scripts/tt_run.sh` (flock, `timeout --foreground`, PID log),
    `scripts/kill_my_jobs.sh` (recorded PIDs only, with their descendants);
  - `rsl_rl_tt/device.py`: the single guarded entry point `open_tray_mesh()` (env == tray_env.sh evaluated in a clean sub-shell,
    tray lock held, sysfs node->BDF check, mesh shape == #chips, then ttnn import + open, then `/proc/self/fd` check with
    close-and-raise, close via context manager + atexit, fabric disabled after close);
  - `.claude/hooks/tray_guard.{sh,py}` (command-position Bash guard, fails closed) + `.claude/settings.json` (hook registration and
    `permissions.deny` on `.claude/**` and `scripts/tray_env.sh`; written last, so those files are now locked for the assistant);
  - tests: `test_tray_guard.py` (218), `test_device_guard.py` (31), `test_repo_invariants.py` (`rsl_rl/` == tag; only device.py
    may open/enumerate devices incl. `ttnn.get_num_devices`), `test_claude_settings.py`.
- Verified by running with harmless commands (no ttnn): env seen by a job, tray-busy exit 75, timeout -> SIGINT -> clean Python
  shutdown, `kill_my_jobs.sh` SIGINT-first, stale-PID pruning. Found and fixed a real bug: group-wide SIGINT from `timeout` killed
  `tee`/`grep` in job-control pipelines (now `--foreground`).
- Research docs: `UPSTREAM_MAP`, `UPSTREAM_SEMANTICS` (52 rows), `TTRL_NOTES`, `TTNN_OP_SURVEY` (19 on-device probes Q1-Q19),
  `ENV`, `DECISIONS` (ADR-0001 proposed A, ADR-0002/0003/0004, **ADR-0005/0006 pending**, host allow-list, C-1/C-2).

**Key findings that need the owner**
1. **ADR-0005 (touches BRIEF 2-1/2-2):** UMD enumeration briefly open()s every `/dev/tenstorrent/*` node before applying
   `TT_VISIBLE_DEVICES`; the fd check cannot see that. Options: accept / mask other nodes in a private user+mount namespace
   (feasible here without sudo) / admin-level isolation.
2. **ADR-0006:** the tt-metal tree has no `_ttnn.so` (bindings OFF); `import ttnn` is impossible there. Candidate: PyPI wheel
   `ttnn==0.81.1` (same version; inspected, not installed).
3. **C-1:** upstream clips actor and critic separately (`ppo.py:309-310`), BRIEF 3.7 says one joint norm.
4. **C-2:** `v5.5.2` vs the Isaac Lab pin.
5. fp32 matmul = TF32-class operands + fp32 accumulation (~1e-3 rel. error expected) -> first on-device measurement; sets tolerances.

**Next (after the owner's decisions, `/hooks` review and a session restart)**
1. Implement the chosen isolation (ADR-0005) in `tt_run.sh`; install the chosen ttnn (ADR-0006) into `.venv`.
2. Smoke (a): `tt_run.sh --chips 16` -> 1x1 mesh via `open_tray_mesh` -> small matmul vs torch -> close; confirm the fd guard.
3. Smoke (b): all 8 chips -> mesh (2x4 expected) -> `FABRIC_1D` -> `ttnn.all_reduce` -> exactly 8 nodes held, all allowed.
4. Noise-floor probes (survey Q10 etc.) -> `TOLERANCES.md`; finalize ADR-0001; then STOP and report for Phase 1 approval.

**Blocked on the owner:** ADR-0005, ADR-0006, C-1, C-2 (details in `DECISIONS.md`).
