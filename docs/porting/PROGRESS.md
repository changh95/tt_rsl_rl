# Progress log

Newest entry first. Each entry: date, phase, done / next / blocked / key numbers, related commits.
Read order for a new session: `BRIEF.md` -> `CLAUDE.md` -> this file.

## 2026-11-12 — Phase 8.6 still BLOCKED: chip 16 opened again, then a second timeout (exit 124) on the `load()` scenario (branch `tt/phase-8-6-cnn`, not merged)
**Done**: the owner approved a reset of chip 16 ("do the reset yourself, continue"); I started `tt-smi -r 16` inside the one-chip namespace of `tt_run.sh --chips 16` (only node 16 exists there; the number is the /dev node number, checked in the tt_umd bindings). The session ended during that call and the log has no tt-smi output,
so it is unknown whether the reset ran; a smoke test then passed (`test_smoke_device.py`, 1 passed in 0.99 s) and no second reset was issued. The CNN trace file (`tests/tt/test_tt_cnn_trace.py`) completed in 6.7 s (bit identity passes, `survives_load` still fails its assertion).
**Blocked (BRIEF 2-6)**: `survives_load` run ALONE (`-k`) in a fresh process hit the 120 s timeout (exit 124); stopped, no follow-up job, no reset, chip state unknown. The incident note (second section) has the commands, the hypothesis (T-15 family, two traced `TTPPO` in one process) and the rules for the next debug jobs.
**Host-only finding**: the encoder's `sync_from_torch` writes in place (`upload_into`) like the MLP layers, so stale weights after `load()` is not the cause; it stays open. **Needs the owner**: a go for one smoke test on chip 16 (or a reset); then debug with the whole file and captured output, multi-chip, checkpoints, e2e `cartpole_img`, merge + tag `phase-8-6-done`; then 8.7 SAC.

## 2026-11-11 — Phase 8.6 IN PROGRESS, BLOCKED on the device (branch `tt/phase-8-6-cnn`, not merged): `CNNModel`; chip 16 does not open after a hung debug job
**Blocked**: a scratch debug job (`scripts/tt_run.sh --chips 16 --timeout 600`) hit its timeout (exit 124); afterwards chip 16 cannot be opened (`MMIO per-op timeout: 4B load took 55484 us`, `Setting power state failed on device 16`). BRIEF 2-6: no reset, no hardware tool, no job on another node, stopped and reported;
details, logs and the proposed next steps: `docs/porting/incidents/2026-11-11_cnn_debug_job_timeout_chip16_mmio.md`. **Needs the owner**: check / reset chip 16.
**Done before the incident** (committed on the branch): `rsl_rl_tt/cnn.py` (`TTCNN`: Conv2d as an im2col matmul over exact row gathers with constant indices, the adjoint gathers for the input gradient (an input pixel is read by at most one output pixel per kernel offset), stride, zero padding, elu / relu / tanh, `MaxPool2d(3, 2, 1)`
with torch's first-maximum gradient rule, global average pool, hand-written backward; `TTCNNStack`), `TTMLPModel` for a `CNNModel` (the 1D groups normalised + the encoders' latents -> the MLP head, whose first layer has a ROW MAP from torch's flatten order to the encoder's latent layout; staged observations with the images
flattened), `TTPPO` for CNN actor / critic (the encoders' parameters in Adam / checkpoints through a generic `encoder`; refuses `share_cnn_encoders`, RND / symmetry / distillation with images), `BatchedCartPoleImage` (`cartpole_img`: velocities as a 1D group + the positions as a 2 x 8 x 8 image), `small_spec` carries the model config.
Tests green on node 16 before the incident: 7 component tests (forward <= 2.4e-3 and every gradient <= 5.6e-3 vs torch autograd for 6 layer configurations; the gathers bit-exact), 5 model / update parity tests (forward 1.5e-3 / 2.4e-3, a whole update vs upstream: actor 1.1e-4 smooth / 4.7e-4 clipped, critic 3.7e-5 / 4.8e-3, losses within the v1 rows),
the eager-vs-traced bit-identity test. **Open**: `test_traced_cnn_path_survives_load` fails (after `load()` the second algorithm's act returns garbage: a T-15-type state problem of the new encoder in a trace, not yet understood); then multi-chip, checkpoints with the encoder, the end-to-end run, docs, merge. **Next**: after the owner has looked at the chip, the smoke test on node 16, then the debug inside pytest.

## 2026-11-11 — Phase 8.5 DONE (branch `tt/phase-8-5-rnn`, tag `phase-8-5-done`): `RNNModel` (GRU, LSTM) in PPO on the device
**Done**: `rsl_rl_tt/rnn.py` (`TTRNN`: GRU / LSTM stacks of any depth with every gate block tile-aligned (`pad_heads`), the cell step, a whole-sequence forward with resets after dones and the hand-written BPTT; the weight gradients as single matmuls over the stacked steps), `TTMLPModel` for an `RNNModel` (normaliser -> RNN -> MLP head; persistent hidden state stepped in place, `reset`, `sequence_forward_train` / `sequence_backward`),
`TTPPO` for recurrent actor and / or critic: the hidden states live on the device and are stepped inside the traced act step (the warm-up passes are made harmless by restoring the state), zeroed after done steps, the state at the START of the rollout is kept for the update, `compute_returns` steps the critic without advancing it, the update uses a FIXED gather into mini-batch order (upstream's recurrent generator does not shuffle:
mini-batch i = the environments [i E, (i + 1) E), whole trajectories) and one traced step per mini-batch that unrolls the sequence and the BPTT; the RNN parameters are part of Adam, the optimizer-state conversion, the replica check and the checkpoints (JIT / ONNX export of the mirror works). **The design claim that makes this fixed-shape**: upstream's split-at-dones-and-pad trajectories are exactly a scan over the T steps of every environment with the state
zeroed after each done and started from the state saved at the rollout start; proved on the upstream recurrent PPO itself, GRU / LSTM x 1 / 2 layers, outputs AND gradients (`tests/tt/test_rnn_host.py`, host). Also: `TraceMixin` users unchanged, `BatchedCartPolePO` (cart-pole without velocities) and `RefSpec.model` / `with_rnn` / `--rnn` in the scripts, `done` column of the packed table. Refused with a reason:
recurrent models with the heteroscedastic Gaussian or RND (upstream feeds padded trajectories to the RND networks); symmetry is refused by upstream itself.
**Verification** (`scripts/run_phase8_final.sh`, all green; v1, 8.1 - 8.4 suites unchanged): 1 chip 313 passed (286 + 27 new) + traces 5 + 3 + 3 + 3 + 3 + 4; 1x2 and 2x4: 138 passed (+175 single-chip-semantics tests skipped) + multi-chip 33 (28 + 5 new) + traces 5 + 3 + 3 + 3 + 3 + 4; host tests +4. Numbers: one step of GRU / LSTM (1, 2 layers) vs torch 9e-4 (`mlp_forward` row); a sequence of 6 steps with dones: outputs <= 1.8e-3, BPTT gradients of every parameter and of the
inputs <= 2.4e-3 (`mlp_backward` row 1.5e-2; padded columns exactly 0); a whole `update()` vs upstream's recurrent update from identical state and hidden states, 6 configurations (GRU, LSTM, 2 layers, recurrent actor only, critic only) x (one smooth step, a full clipped update): losses within their v1 rows, actor <= 4.6e-3 of the update (new rule `rnn_noise_factor` 20 x noise baseline; baseline 3.6e-4 -- root cause in `TOLERANCES.md`),
critic <= 1.5e-4, clipped <= 3.9e-3 (limit 8e-2); traced == eager BIT FOR BIT (logged scalars, all device state incl. the RNN parameters and the live hidden states; GRU, 2-layer LSTM, actor-only; 1 chip, 1x2, 2x4; also after `load()`); 2 chips vs upstream with one mini-batch (which is then exactly upstream's): actor 5e-4 / 5e-5, critic 5e-5 / 3e-5 of the update, replicas bit-identical;
checkpoints TT -> upstream and upstream -> TT (RNN weights and their Adam moments in torch's layout, exactly equal); a TT-trained GRU policy exports to JIT and ONNX with matching outputs. **End to end** (`docs/porting/golden_rnn_{gru,lstm}` = upstream CPU, `tt_runs_rnn_{gru,lstm}` = TT 1 chip traced, SIX seeds each, partially observable cart-pole, actor + critic GRU / LSTM of 64 units, 512 environments, 150 iterations, the Phase 4 rule):
GRU 412.5 vs 439.3 +- 93.9 (threshold 345.4), LSTM 327.9 vs 374.5 +- 152.3 (threshold 222.2): pass; the seeds of this task spread widely (upstream itself: GRU 233..500, LSTM 126..500, and the same seed gives different curves with 5 and 8 CPU threads), so this is a weak statement, like pendulum's. **Speed** (`PERF.md` 7f): 1.7x faster than the upstream CPU at 512 environments / GRU 64, 6.8x (GRU 256) and 6.2x (LSTM 256) at 4096 environments.
**Not done**: a recurrent student / teacher in the distillation, recurrent models in the G1 demo, heteroscedastic / RND with recurrent models. **Next**: 8.6 CNNModel.

## 2026-11-11 — Phase 8.4 DONE (branch `tt/phase-8-4-distillation`, tag `phase-8-4-done`): distillation (student / teacher behaviour cloning) on the device
**Done**: `rsl_rl_tt/distillation.py` (`TTDistillation(TraceMixin, Distillation)`: the unmodified `DistillationRunner`, student and teacher on the device, student actions with host noise + the teacher's deterministic action per step into preallocated slots, the whole update on the device: one traced step per time index that ADDS its gradients to persistent accumulators, one traced apply (gradient mean over the chips, optional clip, Adam, zero the accumulators),
upstream's window semantics kept (summed losses = summed gradients, the step counter across epochs, leftover steps accumulated and never applied), mse and huber losses, the student's normalizer after the epochs), `rsl_rl_tt/traced.py` (`TraceMixin`: `_capture` / `_replay` / `release_traces` factored out of `TTPPO`), checkpoints both ways (`student_state_dict`, `teacher_state_dict`, `optimizer_state_dict`; an RL checkpoint loads the teacher only), `--distill` in the scripts,
`reference.make_distill_cfg` / `make_teacher_checkpoint` / `build_distillation_runner` / `train_distillation`, `distill_equiv.py` + `scripts/mc_distill_reference.py`. No new allow-list entry. Refused with a reason: recurrent models, a heteroscedastic student / teacher, mixed precision, other optimizers.
**Verification** (`scripts/run_phase8_final.sh`, all green; v1, 8.1, 8.2 and 8.3 suites unchanged): 1 chip 286 passed (267 + 19 new) + traces 5 + 3 + 3 + 3 + 3; 1x2 and 2x4: 131 passed (+155 single-chip-semantics tests skipped) + multi-chip 28 (24 + 4 new) + traces 5 + 3 + 3 + 3 + 3; host tests +9 (`test_distillation_host.py`: the upstream oracle, the teachers, the window semantics). Numbers: student / teacher actions of a rollout vs upstream step by step 2.8e-4 / 3.9e-3 (`mlp_forward` row);
the accumulated gradient of a window equals the SUM of the per-step autograd gradients (`mlp_backward` row); a whole `update()` vs upstream from identical state, 2 envs x 5 window / epoch / loss variants (one window of 8 steps, four windows, two windows with 4 leftover steps, every step with huber, huber outside its quadratic zone at a large learning rate): behaviour loss <= 6.8e-3 (new row `update_behavior_loss` 2e-2),
student in function space <= 1.8e-4 / 2.0e-3 for 1-2 Adam steps (noise-baseline limit) and <= 1.6e-2 for 3+ steps (new row `distill_many_steps` 3e-2 with its root cause: the noise-floor sign flips of the first Adam step stay on a consistent regression gradient; measured window by window they appear in the first step and do not grow), normalizer 1e-6; traced == eager BIT FOR BIT (every logged scalar and all device state incl. the leftover accumulators; 1 chip, 1x2, 2x4, also after `load()`);
N chips vs 1 chip on the same stored rollout and replicas bit-identical; checkpoints TT -> upstream and upstream -> TT with every tensor exactly equal; the teacher is unchanged by the distillation; the runner refuses to learn without a teacher. **End to end** (`docs/porting/golden_distill` = upstream CPU, `tt_runs_distill` = TT 1 chip traced, 3 seeds, teachers = upstream PPO runs on the CPU: cartpole / pendulum / tracking12 (teacher sees `privileged` too), the Phase 4 rule):
the student's return cartpole 500.0 vs 500.0, pendulum -235.1 vs -233.0 +- 3.1 (threshold -244.7), tracking12 154.2 vs 154.3 +- 0.34 (threshold 146.6): all pass, and the behaviour-loss curves agree (cartpole 13.77 / 0.0350 / 0.0138 / 0.00174 vs 13.86 / 0.0357 / 0.0139 / 0.00175 at iterations 0 / 15 / 30 / 59; pendulum and tracking12 likewise). Wall time per run (3 seeds, 60 / 100 / 100 iterations): TT 3.1 / 6.5 / 10.3 s vs upstream CPU 5.4 / 11.7 / 24.8 s (the CPU environment is most of the TT time).
**Not done**: recurrent models (8.5), heteroscedastic student / teacher. **Next**: 8.5 RNNModel (GRU, LSTM).

## 2026-11-11 — Phase 8.3 DONE (branch `tt/phase-8-3-symmetry`, tag `phase-8-3-done`): symmetry augmentation and mirror loss on the device
**Done**: `rsl_rl_tt/symmetry.py` (`MirrorMaps`: the user's `data_augmentation_func` is evaluated ONCE on probe tensors (new allow-list entry **H-10**) and its affine mirror maps are derived and verified against it; `TTSymmetry`: the maps on the device, exact for signed permutations via a bf16 split, the mirrored copies of the rollout as extra columns of the packed table,
the mirror loss and its gradient inside the traced mini-batch step, all four modes: augmentation, mirror loss, both, logging only; 2 and 4 copies), `Loss/symmetry` through the unmodified runner, `--symmetry` in the scripts, the test symmetries of the three CPU envs (`rsl_rl_tt/testing/symmetry.py`), `mc_step_reference.py --symmetry`. Refused with a reason: non-affine functions, a mirrored
group that depends on another group, a first block that is not the original, the combination with the heteroscedastic Gaussian.
**Verification** (`scripts/run_phase8_final.sh`, all green; v1, 8.1 and 8.2 suites unchanged): 1 chip 267 passed (233 + 34 new) + traces 5 + 3 + 3 + 3; 1x2 and 2x4: 113 passed (+154 single-chip-semantics tests skipped) + multi-chip 24 (20 + 4 new) + traces 5 + 3 + 3 + 3; host tests +10. Numbers: a signed permutation is applied BIT FOR BIT (the plain fp32 matmul: 4e-4), a general matrix 1.2e-3;
the mirrored copies of a real rollout equal the user function's output exactly (cartpole K = 2, tracking12 K = 4, actor / critic observations and actions); a whole `update()` vs upstream from identical state, 3 configurations x 4 modes x (one smooth step, a full clipped update): actor <= 1.9e-3 (one smooth step; noise baseline <= 4.5e-4, limit 10x + 1e-3) and <= 2.5e-2 (clipped, limit 8e-2), critic <= 3.7e-3, the four losses within their v1 rows,
`Loss/symmetry` <= 1.56e-2 (new row `update_symmetry_loss` 5e-2: the loss is a squared difference of two actor outputs, root cause in `TOLERANCES.md`); logging only changes NOTHING (the parameters after 6 iterations are bit-identical to a run without the extension); the mirror loss makes the policy more symmetric; traced == eager bit for bit (1 chip, 1x2, 2x4, also after `load()`);
N chips vs 1 chip on the union mini-batch with 4 copies + the mirror loss within the 2e-2 limit, replicas bit-identical; checkpoints keep the structure of an upstream one (no state of its own). **End to end** (`docs/porting/golden_sym` = upstream CPU, `tt_runs_sym` = TT 1 chip traced, 3 seeds, the true symmetries of the envs, augmentation + mirror loss 0.2): cartpole 500.0 vs 500.0,
pendulum -151.1 vs -152.4 +- 1.3 (threshold -160.0), tracking12 184.4 vs 183.7 +- 1.5 (threshold 174.5): all pass, and the symmetry-loss curves agree (iteration 0: 0.0077 vs 0.0074, 0.0140 vs 0.0139, 0.0031 vs 0.0031). **Cost** (`PERF.md` 7e): +38 % per iteration with 2 copies (the update's mini-batches have twice the rows; the rollout is untouched).
**Not done**: non-affine symmetry functions; heteroscedastic + symmetry. **Next**: 8.4 distillation.

## 2026-11-11 — Phase 8.2 DONE (branch `tt/phase-8-2-rnd`, tag `phase-8-2-done`): Random Network Distillation on the device
**Why / order**: the owner said to work through the post-v1 list in order (`POST_V1.md`, ADR-0007); 8.2 is the next item. **Done**: `rsl_rl_tt/rnd.py` (`TTRND`: frozen target + trained predictor MLPs, state normalizer, own Adam / constant learning rate, the predictor's MSE step inside the traced mini-batch step, gradients averaged over the chips;
`TTRewardNormalization`: `EmpiricalDiscountedVariationNormalization` with a per-environment accumulator and global statistics), the intrinsic reward of every environment step as its own trace (`rnd.reward`) added to the reward before the time-out bootstrap, the rollout storage's RND-state block (only when
the RND state is neither the actor's nor the critic's observation), `Loss/rnd` / `Rnd/*` logging through the unmodified runner (the per-step intrinsic reward goes to the host for the runner's statistics: new allow-list entry **H-9**), checkpoints both ways (`rnd_state_dict`, `rnd_optimizer_state_dict`, `rnd_update_counter`),
`--rnd` in `scripts/train_reference.py` / `train_tt.py`, `mc_step_reference.py --rnd`, `RefSpec.obs_groups`, `step_dt` of the test envs.
**Verification** (`scripts/run_phase8_final.sh`, all green, the v1 and 8.1 suites unchanged): 1 chip 233 passed (203 + 30 new) + traces 5 + 3 + 3; 1x2 and 2x4: 105 passed (+128 single-chip-semantics tests skipped) + multi-chip 20 (16 + 4 new) + traces 5 + 3 + 3; host tests +5. Numbers: intrinsic reward vs upstream `get_intrinsic_reward`, 6 consecutive steps, 6 configurations (state / reward
normalisation, constant / step / linear weight schedule, RND state = the actor's / the critic's / its own groups): <= 2.6e-3 (limit 8e-3); reward-normalizer statistics <= 5.8e-3 (1.8e-2); predictor gradients vs autograd <= 5.8e-3 (limit 1.5e-2, `mlp_backward`); predictor loss <= 5.4e-3 and BIASED -0.5 % (limit 1.6e-2; root cause in `TOLERANCES.md`); five Adam steps of the predictor in function space 2.1e-3 / 3.5e-3
(noise baseline 2.5e-3 / 5.8e-3); a whole `update()` vs upstream from identical state: actor / critic <= 7.4e-3 (clipped limit 8e-2), predictor <= 1.2e-3 (noise-baseline limit), `Loss/rnd` <= 5.4e-3, RND state normalizer 1e-6; traced == eager BIT FOR BIT (every logged scalar and all device state incl. the predictor, its Adam moments, both normalizers and the accumulator; 1 chip, 1x2, 2x4, also after
`load()`); N chips vs 1 chip on the union mini-batch (actor / critic / predictor): 1x2 1.3e-3 / 6.2e-4 / 0, 2x4 1.3e-3 / 6.2e-4 / 2.0e-3 of the update (limit 2e-2), replicas bit-identical; checkpoints TT -> upstream and upstream -> TT with every RND tensor, the Adam moments and the counter exactly equal. **Three new tolerance rows with their measured bases** (`rnd_intrinsic_reward`, `rnd_reward_stats`,
`rnd_predictor_loss`; not loosenings), ADR-0007 amendment A-2. **End to end** (`docs/porting/golden_rnd` = upstream CPU, `tt_runs_rnd` = TT 1 chip traced, 3 seeds, RND weight 5 / 2 / 5 x `step_dt`, the Phase 4 rule): cartpole 500.06 vs 500.06 +- 0.014, pendulum -150.5 vs -151.5 +- 0.35 (threshold -159.1), tracking12 192.6 vs 193.8 +- 2.2 (threshold 184.1): all pass, and the RND-specific curves agree too
(`Rnd/mean_intrinsic_reward` per episode: iteration 0 8.71 vs 8.67 on cartpole, 1.449 vs 1.449 on pendulum, 1.457 vs 1.457 on tracking12; end of training 0.0595 vs 0.0604, 0.0150 vs 0.0151, 0.234 vs 0.233; `Loss/rnd` the same). **Cost** (`PERF.md` 7d): +31 % per iteration with RND (tracking12, traced; the eager first version was +67 %).
**Two traps found by the new tests, both fixed**: a tensor alive across the replay of another trace is overwritten (T-15: the reward is now the first thing `process_env_step` does, and the staged RND observation is a persistent buffer); the eager storage keeps references, so a persistent staging buffer made every step point at the last observation. The second one passed the upstream parity tests, because the `privileged` group of the test env hardly
changes between steps; found by the eager-vs-traced test, covered by `test_the_rollout_stores_the_rnd_state_of_every_step`. **Not measured / not done**: the RND observation is uploaded twice per step (once for the reward, once by the next `act`); RND with recurrent / CNN models (not available yet). **Next**: 8.3 symmetry (augmentation / mirror loss).

## 2026-11-11 — G1 video: the strip under the panels now shows tracking error, value loss and reward terms (branch `tt/demo-g1-metrics`, merged to `main` at the owner's request)
**Why**: the owner asked for (1) the velocity tracking error, (2) the loss of the estimation layer / encoder, (3) the reward the actor-critic uses (penalties included) instead of the mean-return / speed curves. **Done** (`rsl_rl_tt/demos/g1_render.py`: `tracking_error`, `tracking_curve`, `reward_rates`, `REWARD_GROUPS`, new `curve_strip`;
`scripts/eval_g1_tracking.py`, `--tracking` of `scripts/render_g1_demo.py`; `examples/g1_walk/play.py` measures the tracking error of the checkpoints next to the metrics once): `g1.mp4` is now 1440x752, `g1.gif` 3.2 MB. (1) Tracking error = length of (command - body-frame velocity) and
|yaw-rate error|, 128 robots, the commands of the video, deterministic policy, measured on the 16 saved checkpoints (`tracking_error.json`): 0.48 m/s untrained (40 falls per robot-minute), 0.54 at iteration 101 (stands still), 0.13 at 201, 0.075 at 1500 (yaw rate 0.24 -> 0.056 rad/s); only 16 points, so it is not dense
where the learning happens. (2) **The model has no estimator / encoder and so no estimation loss**: the only one is the critic's value loss (`Loss/value`, 0.34 at its peak in iteration 2, 4e-4 at the end), which is what the plot shows, labelled as such. (3) The 16 reward terms in 7 groups, per second of episode time, stacked (rewards above,
penalties below, net as a line): net +4.15 per second at iteration 1500 (= 83 per episode): tracking +2.15, gait +1.95, alive +0.2, penalties -0.07 / -0.06 / -0.01, no falls. **There is no contact / collision penalty in this task** (flat ground, a fall ends the episode with -2), said on the plot and in the README. A retrain with extra logging for a dense tracking-error curve was
considered and not done (the TT code changed since the run, so it would not reproduce it; every number in the README would have to be remade). Host tests: 15 in `test_g1_render.py` (6 new / rewritten). **Next**: 8.2 RND.

## 2026-11-11 — Phase 8.1 DONE (branch `tt/phase-8-1-heteroscedastic`, merged to `main` at the owner's request): `HeteroscedasticGaussianDistribution` on the device
**Why / order**: the owner put the Isaac Lab / GPU-host verification aside and said to start on something else; the "v1 이후" list of BRIEF §1 is ordered by effort in `POST_V1.md` and ADR-0007; 8.1 is the smallest item. **Done**: `TTLinear` / `TTMLP` two-head output layer (mean | std, each tile-aligned; the Unflatten of upstream's
`[2, A]` output), `TTHeteroscedasticGaussian` + per-row `HeteroState` (scalar and log std, std clamp gate), per-sample std in the rollout storage (one more column block, only for this distribution), per-row log-prob / entropy / KL, the std head trained through the ordinary MLP backward
(`d_out = [d_mean | d_raw]`), the logged std, Adam-moment conversion through the layer's own layout (checkpoints both ways), `RefSpec.distribution`, `--distribution` / `--std-type` / `--golden` in `train_reference.py` / `train_tt.py`, `mc_equiv` + `mc_step_reference.py --hetero`. No new host computation (allow-list unchanged).
**Verification** (`scripts/run_phase8_final.sh`, all green; v1 suites unchanged): 1 chip 203 passed (157 v1 + 46 new) + traces 5 + 3 new; 1x2 and 2x4: 89 (84 + 5) + multi-chip 16 (12 + 4) + traces 5 + 3; host 450 incl. 11 new. Numbers: PPO loss formulas given identical network outputs 7e-6 (limit 2e-5), KL <= 2e-7;
end-to-end gradients vs autograd median 4e-3, worst 2.9e-2 (a loss on 29 action dims; limit 8e-2); one `update()` vs upstream in function space (mean AND std head): one-step / smooth <= 1.5e-3 (limit 10x the noise baseline + 1e-3), clipped <= 3.2e-2 (limit 8e-2); traced == eager bit for bit (1 chip, 1x2, 2x4, also after `load()`);
N chips vs 1 chip on the union mini-batch 1.2e-3 / 3.9e-4 of the update (actor / critic; limit 2e-2) on both meshes; checkpoints TT -> upstream and upstream -> TT with identical Adam moments, JIT / ONNX export. **One new tolerance row with its root cause** (not a loosening): `update_entropy_hetero` 1.5e-3 (measured <= 3.7e-4): a state-dependent
std makes the entropy a function of the matmul output (`TOLERANCES.md`, ADR-0007). **End to end** (`docs/porting/golden_hetero` = upstream CPU, `tt_runs_hetero` = TT 1 chip traced, 3 seeds, scalar std, the Phase 4 rule): cartpole 270 vs 286 +- 139, pendulum -843 vs -594 +- 306 (threshold -900), tracking12 114.6 vs 113.8 +- 13.4: all pass; the reference seeds
spread widely on cartpole and pendulum, so with 3 seeds this is a weak statement there. **Not measured**: speed of the heteroscedastic path (the per-row state adds elementwise work to the mini-batch step; v1 numbers are unaffected). **Next**: 8.2 RND (second MLP pair + own Adam + intrinsic reward before GAE).

## 2026-11-11 — Demo clean-up (branch `tt/demo-g1-curves`, merged to `main` at the owner's request): G1 video with reward and learning curve; humanoid demo removed
**Done** (both at the owner's request): (1) `g1.mp4` / `g1.gif` now show, under every panel, the live return of that robot's current episode (one scale for all panels, red ticks = falls) and, under all panels, the training curve of the run (mean episode return, and achieved vs commanded
speed, against the iteration on a log axis) with the iteration of every panel marked in its colour (`rsl_rl_tt/demos/g1_render.py`: `render_panel_traced`, `draw_return_strip`, `curve_strip`; `--metrics` of `scripts/render_g1_demo.py`, `examples/g1_walk/play.py` finds the `metrics.csv` next to the checkpoints; 3 new host tests).
At the end of the 20 s the returns are 84.1 (iteration 1500; curve 83), 54.4 (101; curve 45) and ~5 (untrained; curve 0.4). (2) The MuJoCo humanoid demo is removed from the repository (code, scripts, tests, README section, `docs/porting/demo_humanoid/`); the G1 demo does not use any of it; it stays in the git history.
Host tests: 434 (431 after removing the 8 humanoid tests, +3).

## 2026-11-11 — Demo 2 follow-up (branch `tt/demo-g1-gait`, merged to `main` as `5823028` at the owner's explicit request; not a phase, so no tag): `examples/g1_walk/`, initial / middle / final video, eight-chip scaling profile
**Commit `0aa87d7`.** **Done**: (1) `examples/` (`README.md`, `g1_walk/{train.py, play.py, README.md}`): the G1 demo as a short example on top of the public rsl_rl API (`MujocoG1VecEnv` + `g1_train_cfg` + `tt_agent_cfg` + the upstream `OnPolicyRunner`); `train.py` smoke-tested on one chip
(3 iterations, 256 robots), `play.py` on its checkpoint. The evaluation / rendering code moved from `scripts/render_g1_demo.py` into the library module `rsl_rl_tt/demos/g1_render.py` (the scripts are thin wrappers); `tests/tt/test_g1_render.py` (8 host tests: labels, checkpoint
load path == upstream inference policy exactly, measurement, tiny EGL render, video / GIF writers, `--help` of the four scripts). (2) New comparison video / GIF `docs/porting/demo_g1/g1.{mp4,gif}`: untrained / iteration 101 / iteration 1500 (before: 201 instead of 101; iteration 101
keeps its balance but does not move (0.00-0.01 m/s, 0 falls), 201 already follows the commands, so 101 is the stage that still looks different); `eval.json` now has the four checkpoints (256 robots x 600 steps each). (3) `gait.mp4`: slow-motion two-view close-up of the final gait with foot contacts
(`scripts/render_g1_gait.py`). (4) Profile of one iteration, one chip vs eight (`train_g1_demo.py --profile`, `profile_1chip/`, `profile_8chips/`): CPU simulation 830 / 845 ms, PPO update replay 308 / 112 ms (2.75x with 8x the chips), acting 56 / 42, step processing 28 / 34, iteration 1290 / 1123 ms (1.15x);
upper bound with free chips 1.55x (Amdahl); explanation in `docs/porting/demo_g1/README.md` ("Why eight chips are not eight times faster"). **Next**: nothing pending.

## 2026-11-12 — Demo 2 (not a phase; merged from branch `tt/demo-g1`): Unitree G1 walking on command, CPU simulation + training on ALL EIGHT chips
**Done**: `rsl_rl_tt/demos/mujoco_g1.py` (G1 MJX model of MuJoCo Menagerie, pinned commit, fetched by `scripts/fetch_g1_model.sh`: velocity-command task, 12 leg joints, privileged critic observations, phase-based gait rewards;
4096 robots stepped by MuJoCo's multithreaded `rollout`), `g1_cfg.py`, `scripts/train_g1_demo.py` / `render_g1_demo.py` / `plot_g1_curves.py`, 9 host tests (`tests/tt/test_mujoco_g1.py`: conventions, determinism, geometry, rollout == `mj_step` EXACTLY
with the warm start and within 2.6e-3 rad cold-started, fall / time-out / reset, commands, private RNG, short upstream-PPO run with the asymmetric critic). Result (`docs/porting/demo_g1/README.md`): 2x4 mesh, 4096 envs, 1500 iterations
(147 M samples, 27.6 min, replicas bit-identical at every 10th update): follows 0.4 / 0.8 m/s and a turn command (0.39 / 0.78 / 0.51 m/s + 0.38 rad/s achieved) with 0 falls in 12 s x 256 robots; video `g1.mp4` (20 s), curves.
Per iteration: upstream CPU PPO 3.93 s (algorithm 3.05), TT one chip 1.35 s (0.51), TT eight chips 1.10 s (0.27): the CPU simulation is 76 % of the eight-chip run.
**Two failed attempts, recorded**: (1) the model's PD gains (ankle kp 20, roll range 0.26 rad => ~5 N*m) made stepping physically impossible; (2) with a sparse contact reward the policy stood still for 450+ iterations; fixed by raising the leg / waist gains
and a dense foot-height-vs-phase reward. One seed; simulation only (no randomisation, no noise). Merged to `main` on 2026-11-12 at the owner's explicit request (not a phase, so no `phase-N-done` tag).

## 2026-11-12 — Demo (not a phase; merged from branch `tt/demo-humanoid`): MuJoCo humanoid walking, CPU simulation + TT training  — **REMOVED from the repository on 2026-11-11 at the owner's request** (code, scripts, tests, README section and `docs/porting/demo_humanoid/`; it stays in the git history)
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
