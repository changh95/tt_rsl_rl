# After v1: the remaining upstream features (BRIEF §1 "v1 이후"), order and design notes

The owner said (2026-11-11) to leave the Isaac Lab / GPU-host verification aside and start on something else; the features below are what BRIEF §1 lists as "after v1". The order is by effort and
by how much new machinery each needs on top of the traced fixed-shape design (BRIEF §3.3). Every item ends with the same acceptance set as v1: component parity against upstream autograd, an update
parity test, traced == eager bit for bit, 1x2 / 2x4 mesh equivalence in function space, an end-to-end run against the upstream CPU run, checkpoint compatibility (`load()` both ways).

| # | Feature (upstream `rsl_rl` v5.5.2) | What it needs beyond v1 | Effort | Status |
|---|---|---|---|---|
| 8.1 | `HeteroscedasticGaussianDistribution` (state-dependent std from the MLP) | per-row std in log-prob / entropy / KL, an output layer with two heads, the rollout storing the std, gradient through the std head | small-medium | **done** (branch `tt/phase-8-1-heteroscedastic`, results in `PROGRESS.md`) |
| 8.2 | RND (`rsl_rl/extensions/rnd.py`) | a second pair of MLPs (predictor + frozen target) with its own Adam, an intrinsic reward added before GAE, its own normalizers | medium | not started |
| 8.3 | Symmetry augmentation / mirror loss | user callbacks (`data_augmentation_func`) that act on observations and actions: either a host callback (a new allow-list entry) or a fixed mirror matrix on the device | medium | not started |
| 8.4 | Distillation (`algorithms/distillation.py`, teacher / student, behaviour cloning) | a second algorithm class, a loss without a critic, the teacher's forward on the device | medium | not started |
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
