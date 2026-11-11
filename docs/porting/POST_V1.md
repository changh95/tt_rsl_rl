# After v1: the remaining upstream features (BRIEF §1 "v1 이후"), order and design notes

The owner said (2026-11-11) to leave the Isaac Lab / GPU-host verification aside and start on something else; the features below are what BRIEF §1 lists as "after v1". The order is by effort and
by how much new machinery each needs on top of the traced fixed-shape design (BRIEF §3.3). Every item ends with the same acceptance set as v1: component parity against upstream autograd, an update
parity test, traced == eager bit for bit, 1x2 / 2x4 mesh equivalence in function space, an end-to-end run against the upstream CPU run, checkpoint compatibility (`load()` both ways).

| # | Feature (upstream `rsl_rl` v5.5.2) | What it needs beyond v1 | Effort | Status |
|---|---|---|---|---|
| 8.1 | `HeteroscedasticGaussianDistribution` (state-dependent std from the MLP) | per-row std in log-prob / entropy / KL, an output layer with two heads, the rollout storing the std, gradient through the std head | small-medium | **done** (branch `tt/phase-8-1-heteroscedastic`, results in `PROGRESS.md`) |
| 8.2 | RND (`rsl_rl/extensions/rnd.py`) | a second pair of MLPs (predictor + frozen target) with its own Adam, an intrinsic reward added before GAE, its own normalizers | medium | **done** (branch `tt/phase-8-2-rnd`, results in `PROGRESS.md`) |
| 8.3 | Symmetry augmentation / mirror loss | user callbacks (`data_augmentation_func`) that act on observations and actions: either a host callback (a new allow-list entry) or a fixed mirror matrix on the device | medium | **done** (branch `tt/phase-8-3-symmetry`, results in `PROGRESS.md`) |
| 8.4 | Distillation (`algorithms/distillation.py`, teacher / student, behaviour cloning) | a second algorithm class, a loss without a critic, the teacher's forward on the device | medium | **done** (branch `tt/phase-8-4-distillation`, results in `PROGRESS.md`) |
| 8.5 | `RNNModel` (GRU, LSTM) | storage of hidden states, trajectory splitting / padding with masks (shapes that vary per update: needs a fixed maximum length for the traces), GRU / LSTM forward and backward on the device | large | not started |
| 8.6 | `CNNModel` | conv2d forward / backward, image observation groups | large | not started |
| 8.7 | SAC | not in the pinned upstream tag (v5.5.2 has only PPO and Distillation) | unknown | not started |

## 8.1 Heteroscedastic Gaussian: design

Upstream (`rsl_rl/modules/distribution.py`): the actor MLP ends in `Linear(hidden, 2 A)` + `Unflatten(-1, (2, A))`; `mlp_output[..., 0, :]` is the mean, `mlp_output[..., 1, :]` the std ("scalar":
clamped to `std_range`) or the log std ("log": clamped to `log_std_range`, std = exp). The std head's weights start at zero and its bias at `init_std` (or its log). There is no std parameter.

* **Layout.** The last `TTLinear` gets two output heads: the device weight is `[in_p, 2 A_p]` with the mean head in columns `0..A_p` and the std head in `A_p..2 A_p` (each padded to a tile multiple on its own), so
  both halves are tile-aligned slices. The host <-> device conversion (`linear_to_host` / `linear_from_host`, the optimizer-state conversion of checkpoints) is done by the layer, which knows its heads.
* **Distribution state per row.** `TTGaussianDistribution` keeps its std / var / log-std rows as persistent tensors that are recomputed after every parameter change. The heteroscedastic distribution has no
  parameter: the same quantities (std, var, log std, 1/var, 1/std^3, the clamp gate) are computed from every forward pass, per row, and are passed around as an object with the same attribute names, so the
  loss code is the same elementwise code with `[rows, A_p]` instead of `[1, A_p]` operands. Padded action columns get std 1 (masked out of every sum).
* **Backward.** `d logp / d mean = diff / var` as before; `d logp / d std = diff^2 / std^3 - 1 / std` per row; the entropy bonus adds `- c_e / (rows std)` per element; the gradient to the raw output of the std
  head is that times the clamp gate (scalar: inside `[lo, hi]`, inclusive like torch; log: times std). The output gradient is the concatenation `[d_mean | d_raw]`, the rest is the existing MLP backward.
* **Rollout storage** additionally stores the std of every step (torch: `distribution_params = (mean, std)`); the packed table gets one more column block, only when the distribution is heteroscedastic
  (the v1 layout is unchanged). The KL for the adaptive learning rate uses the stored std per row.
* **Logging.** `get_policy().output_std` is the std of the last forward pass in upstream; here the mirror gets the per-column mean of the last mini-batch's std (a persistent `[1, A_p]` tensor written by every
  step, downloaded once per iteration, allow-list H-5).
* **Not in this item:** `BetaDistribution` (not on the owner's list).

## 8.2 RND: design

Upstream (`rsl_rl/extensions/rnd.py`, `PPO.process_env_step` / `update`): predictor and target MLPs on the RND state (`obs_groups["rnd_state"]`, default: the policy group); the intrinsic reward of an environment step is
`||target(s') - predictor(s')||_2` on the observation AFTER the step (`s'`), normalised by the running std of its discounted sum (`EmpiricalDiscountedVariationNormalization`, optional) and multiplied by a weight (scaled with the env's `step_dt`; constant / step /
linear schedule over the number of environment steps); it is added to the extrinsic reward BEFORE the time-out bootstrap. In `update()` the predictor minimises `mse(predictor(s), target(s))` on the observations of every PPO mini-batch with its OWN Adam (constant
learning rate, no clipping, one step per mini-batch); the state normalizer learns from the whole rollout after the epochs.

* **Module.** `TTRND` holds two `TTMLP`s (the target is only ever run forward), the optional state normalizer (`TTEmpiricalNormalization`), the optional reward normalizer (`TTRewardNormalization`: a per-environment accumulator `avg = gamma avg + r` that is never reset, scalar running
  mean / var / std with the Welford recurrence over the `num_envs` values of `avg`, global over the chips; `r / std`, the reward unchanged while the std is not positive), a `TTAdam` and the learning rate as a device scalar. The weight schedule is evaluated on the host from the
  mirror's `update_counter` (a Python number per step, like the learning rate of v1; the counter lives on the upstream module so that `save()` / `load()` handle it).
* **Rollout.** `act()` additionally stages the RND observation when it is not the actor's or the critic's (`obs_to_device`, ROW_MAJOR upload tilized on the device); `process_env_step()` uploads the RND state of the NEW observation, computes the intrinsic reward on the device, adds it to the reward and
  downloads it for the runner's logging (H-9). Traced as its own capture (`rnd.reward`): the warm-up runs on clones of the state, so that no step is applied twice; first thing in `process_env_step` (T-15). Evaluation mode runs the graph eagerly and never advances any statistics.
* **Storage.** `rnd_source` is `"actor"` / `"critic"` when the groups of `rnd_state` equal those of the actor / critic (then the stored observation IS the RND state, nothing extra) and `"own"` otherwise (a block of `pad_up(num_states)` columns in the packed table, before the scalars).
* **Update.** The mini-batch step (eager and traced) ends with the predictor: forward, `d loss / d prediction = 2 (prediction - target) / (rows x num_outputs)`, the ordinary MLP backward, the gradient mean over the chips, `TTAdam.step` with the predictor's own bias corrections
  (`prepare_step()` once per step). The mean loss is logged as `Loss/rnd` (H-5). After the epochs the state normalizer learns from the stored RND states.
* **Checkpoints.** `sync_to_torch` writes the predictor, the normalizers' buffers and the Adam moments into the upstream module / its optimizer; upstream's `save()` then produces `rnd_state_dict`, `rnd_optimizer_state_dict` and `rnd_update_counter`; `load()` goes the other way (the
  discounted-reward accumulator is not part of a checkpoint upstream either).
* **Multi-chip.** The predictor's gradients are averaged over the chips, the state and reward statistics are global, the accumulator is chip-local (it is per environment); `replica_max_diff` covers all replicated RND state.
* **Not in this item:** sharing the observation upload with the next `act()` (performance only), RND with recurrent / CNN models (not available yet).

## 8.3 Symmetry: design

Upstream (`rsl_rl/extensions/symmetry.py`, `PPO.update`): a USER function `data_augmentation_func(env, obs, actions)` returns the observations (a TensorDict of all groups) / actions followed by their mirrored copies (`[R * K, ...]`, original rows first). With
`use_data_augmentation` every mini-batch is replaced by that augmented batch (old log-probabilities, values, advantages and returns repeated K times; the surrogate and value losses average over all K R samples; entropy and KL use the first R rows); with
`use_mirror_loss` the loss gets `coeff * mse(actor(mirrored obs), mirror(actor(obs)).detach())` (the mean actions, the mirrored copies only); with neither flag the mirror loss is computed for logging only (`Loss/symmetry`).

* **Why the function is probed once.** It is arbitrary Python on torch tensors; it cannot run inside a trace, and a per-step host call would download every mini-batch. Every symmetry of a robot policy is a signed permutation of the observation / action entries, i.e. AFFINE, so `rsl_rl_tt/symmetry.py`
  evaluates the function ONCE at construction (allow-list H-10) on a zero row + unit vectors and derives `mirrored_k(x) = x @ W_k + b_k` per observation group and for the actions, then VERIFIES the maps against the function on random inputs in both ways upstream calls it
  (observations and actions together, each alone). Refused with a reason, never approximated: a function that is not affine, whose mirrored group depends on another group, whose first block is not the original, or that returns a different number of copies for the actions. Only the groups of the actor and critic are checked.
* **On the device.** A map whose entries are exactly representable in bfloat16 (every signed permutation) is applied EXACTLY: the fp32 input is split into three bf16 parts (like the shuffle), each is multiplied with the bf16 matrix (fp32 accumulation; products with 0 / +-1 are exact) and the parts are added; a plain fp32 matmul
  would round the inputs (4e-4 relative, measured). Other matrices use the fp32 HiFi4 matmul (1.2e-3 measured). The mirrored copies of the whole rollout (actor observations, with augmentation also critic observations and actions) are computed once per update and are extra COLUMNS of the packed table
  (`TTRolloutStorage(extra_cols=...)`), so the shuffle moves them with their rows and the traced mini-batch step only slices and concatenates; the mirrored action means of the mirror loss are computed in the step.
* **The step.** K copies: concatenate the slices, repeat the per-sample scalars, run the ordinary forward / loss / backward on K R rows (denominators K R), KL on the first R rows. Mirror loss with augmentation: the augmented forward's means are the loss's (rows >= R against the mirrored means of rows < R), its gradient (x coefficient) is added to
  the PPO gradient of those rows. Mirror loss without augmentation: one more forward on the mirrored observations only (the original rows' mean is the PPO forward's), its backward is added to the actor's gradients before the clip. Logging only: the same forward, no backward.
* **Not in this item**: non-affine symmetry functions; the combination with the heteroscedastic Gaussian (refused with a message: the per-row std would need the entropy / KL on the original rows only); recurrent models (upstream refuses them too).

## 8.4 Distillation: design

Upstream (`rsl_rl/algorithms/distillation.py`, `DistillationRunner`): a student MLP (observation set `student`) imitates a frozen teacher MLP (observation set `teacher`, loaded from an RL checkpoint's `actor_state_dict`) by behaviour cloning. `act`: the student acts stochastically,
the teacher's deterministic action is stored (`privileged_actions`). `update`: `num_learning_epochs` passes over the stored steps IN ORDER (the whole batch of one time index at a time: no shuffling, no mini-batches), per step `loss_fn(student(obs), teacher_action)` (mse or huber, mean over all
elements), the losses of `gradient_length` consecutive steps are SUMMED before one backward pass (the step counter runs across epochs; leftover steps are accumulated but never applied), optional `clip_grad_norm_` of the student, one Adam step; afterwards the student's normalizer learns from the rollout.
There is no critic, no GAE, no value or entropy term, no KL; the student's std parameter is only used for sampling and never trained.

* **Class.** `rsl_rl_tt/distillation.py:TTDistillation(TraceMixin, Distillation)`; `Distillation.construct_algorithm` (inherited) builds the torch student / teacher (the mirrors), `TraceMixin` (`rsl_rl_tt/traced.py`, factored out of `TTPPO`) gives `_capture` / `_replay` / `release_traces`.
* **Storage.** Per step the student's observations ([N, d_p]) and the teacher's actions ([N, A_p]) in preallocated device slots; nothing else (rewards and dones are not needed by a feed-forward student; they are not even uploaded: `process_env_step` only files the step's device tensors).
* **Update.** One captured step per time index (`distill.step.<t>`: student forward with the cache, loss, the ordinary MLP backward, the gradients ADDED to persistent accumulators, the loss to a persistent scalar) and one `distill.apply` (gradient mean over the chips, optional clip, Adam with its own bias-correction scalars, zero the accumulators);
  the host walks the fixed schedule epoch x step, replaying the step trace and, every `gradient_length` steps, the apply trace. Leftover steps stay in the accumulators and are zeroed at the start of the next update (upstream's `loss` is a local variable). mse: `d = 2 diff / (rows x A)`; huber (delta 1): `clamp(diff, -1, 1) / (rows x A)`.
* **Multi-chip.** The loss of a chip is the mean over its rows; on equal shards the mean of the chips' gradients is the gradient of the global mean (unlike PPO there is no chip-local advantage normalisation), so the update on N chips equals the update on one chip on the same rollout; the student's normalizer statistics are global.
* **Checkpoints.** `save()` writes `student_state_dict`, `teacher_state_dict`, `optimizer_state_dict` (the Adam state has entries for the MLP parameters only: torch creates none for the std parameter, which never receives a gradient); `load()` of an RL checkpoint (an `actor_state_dict`) loads the teacher only, like upstream, and the teacher's device copy follows.
* **Not in this item:** recurrent student / teacher (8.5), a heteroscedastic student / teacher, the RND and symmetry extensions (upstream refuses them too).
