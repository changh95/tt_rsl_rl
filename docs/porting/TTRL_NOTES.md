# tt-rl reference notes

Read-only analysis of the reference implementation `tt-rl`, written for the rsl_rl -> Tenstorrent Blackhole port
(BRIEF.md section 4, Phase 0, item 6). Language: English.

## 0. Provenance, scope and how to read this document

| Item | Value |
|---|---|
| Repository | https://github.com/scaleresearch/tt-rl |
| How it was obtained | `git clone --depth 1` into a scratch directory outside this repo. Shallow clone: only one commit exists, so there is no history to consult. |
| Commit read | `6fa5ead55458dbe0690091bb1aa96d327ca37ae5` (2026-11-01, "Add a Dockerfile for running the host test suite on Linux x86_64") |
| Package version | `ttrl.__version__ = "0.1.0"` (`ttrl/__init__.py:6`), development status "Alpha" (`pyproject.toml:18`) |
| License | MIT. `LICENSE:1-3`: "MIT License / Copyright (c) 2026 Antoni Bertel". `pyproject.toml:13-15` repeats `license = "MIT"` and `authors = [{ name = "Antoni Bertel" }]`. Source files carry no per-file license header (they start with a module docstring). |
| What I did | Read files with `ls`/`grep`/`cat`/`Read` only. Nothing from tt-rl was executed, imported or installed. No Tenstorrent device, `ttnn`, `ttml` or `tt-smi` was touched. |

Conventions:

* `path:line` or `path:line-line` is relative to the tt-rl repository root, at the commit above. Where a claim rests on a
  docstring or comment rather than on executable code, I say so ("comment says").
* `local:` marks a file of THIS repository, read only to compare semantics (`local:rsl_rl/algorithms/ppo.py` etc.). `git describe --tags`
  in the local checkout reports `v5.5.2` (HEAD `11563319134f987f70fb034410496fe3cc7327a9`), and `git status --short` shows no change under `rsl_rl/`.
  The sibling document `UPSTREAM_SEMANTICS.md` (written separately) is the authority for upstream behaviour; its rows S01, S13, S26/S27, S30, S35
  agree with the corresponding rows of my section 3.7 table.
* "(unclear)" means I could not determine the answer from the source. Section 13 collects them.
* IMPORTANT VERSION GAP. tt-rl pins `ttnn>=0.80.0,<0.81` and says a ttnn minor bump "is a porting job"
  (`pyproject.toml:27-38,51`). Every hardware measurement and API behaviour quoted below was recorded by tt-rl's author on
  ttnn 0.80.x, Blackhole, "p300c" (e.g. `ttrl/engine/ops/select.py:7-8`). Our TT_METAL is v0.81.1 (BRIEF section 0). Nothing here
  has been verified on our stack; treat every quoted behaviour as a hypothesis to re-measure.

## 1. Key findings (read this first)

1. tt-rl's PPO is NOT traced. `PPO` "holds no captured trace" (`ttrl/algos/ppo/agent.py:233-236`); its update runs eagerly and
   host-side (`ttrl/algos/ppo/learn.py:17-19`). Trace capture/replay is used only by DQN and SAC (`ttrl/algos/dqn/agent.py:151-156`,
   `ttrl/algos/sac/agent.py:175-197`, `ttrl/algos/sac/targets.py:55-91`). So tt-rl gives us a working trace *harness* and a worked
   DQN/SAC traced update, but no example of a traced PPO act/update/GAE. BRIEF section 3.3 (trace everywhere) goes beyond tt-rl.
2. "Manual gradients" in tt-rl means a tape-based reverse-mode autograd whose nodes carry hand-written per-op VJPs built from
   ttnn forward ops (`ttrl/engine/autograd.py:1-5,56-`), not a hand-derived static backward for the whole MLP. The per-op VJPs are
   a useful, tested oracle; the dynamic tape itself is not what BRIEF section 3.2 (A) describes.
3. The optimizer is deliberately kept OUTSIDE the trace (`ttrl/engine/step/core.py:35-40`). Adam (`ttrl/engine/optim/adam.py`) runs
   on device in fp32 with in-place updates, but its learning rate and bias-correction scalars are host Python floats baked into each
   dispatched op (`adam.py:106-107,127,129`). BRIEF section 3.3 wants fwd+bwd+clip+Adam inside the trace with LR as a device
   tensor: tt-rl does not do that and its comment says why it did not (frozen scalars; measured no gain: `core.py:35-40`).
4. Multi-chip is NOT gradient all-reduce. It is N independent actor-learners on N independently opened single-chip devices, and
   every `sync_every` updates (default 50) their parameters are downloaded, summed in float32 on the HOST with numpy, divided by N
   and uploaded back; Adam moments stay local (`ttrl/engine/mesh/sync.py:36-37,60-70,138-157,228-249`;
   `ttrl/train/config/device.py:44`; `ttrl/train/trainer.py:203-208`). The device collective `ttnn.all_reduce` is never called
   (only mentioned in comments, `sync.py:5-9,15`). This confirms the brief's description and means tt-rl has NO CCL code for us to reuse.
5. Device opening and chip auto-selection are pervasive and unsafe for our tray rules: `acquire()` picks the first free chip and
   polls for up to 120 s (`ttrl/engine/device/acquire.py:56-140`); `ttrl.run`, when `TT_VISIBLE_DEVICES` is unset, first scopes the process to "the lowest chip nobody
   holds" (`ttrl/train/scoping.py:53-58,77-89`); liveness probes enumerate EVERY `/dev/tenstorrent/*` node and run `fuser` on each
   (`ttrl/engine/device/liveness.py:89-110,158-182`); `ttrl doctor` opens a free chip in a child process
   (`ttrl/cli/probes/hardware.py:57-66`, `ttrl/cli/probes/chip_report.py:33`). None of this may be used (section 9, 11).
6. The PPO math differs from upstream rsl_rl in several semantics-relevant ways (shared trunk, ReLU, state-dependent log-std head,
   0.5 factor and separate clip epsilon on the value loss, single global grad-norm clip, linear LR anneal instead of adaptive KL,
   observation normalization on the host in the env wrapper, host-side GAE and numpy `std` with ddof=0). Section 3.7 lists them.
   The PPO code is a reference for *how to do things on ttnn*, not a semantic oracle for rsl_rl.
7. Valuable hardware/runtime lessons are recorded in comments (trace wedge on exception, forbidden host reads and writes during
   capture, grads buffer re-binding, allocation in the eager gap, padding bug when reading trace outputs, `moreh_adam` rejecting
   fp32, host-only all-reduce rationale, `TT_VISIBLE_DEVICES` read at import time). Section 12 tabulates them with citations.

## 2. Repository structure and entry points

Top level (`git ls-files`): `ttrl/` (the library), `configs/` (Python run configs, 10 files), `examples/` (two standalone
projects), `tests/`, `tools/` (lint, plotting, release check), `.github/workflows/` (CI incl. a self-hosted hardware job),
`Dockerfile` (host test suite), `pyproject.toml`, `README.md`, `assets/` (GIFs).

`ttrl/` layout (one line per package, what matters to us in bold):

| Package | Role |
|---|---|
| `ttrl/engine/` | **The compute substrate.** `autograd.py` (tape), `ops/` (matmul, elementwise, unary, reduce, shape, select, broadcast, functional), `optim/` (Adam), `tensor/` (host<->device transfer, in-place copy, `StaticArena`), `step/` (`TrainStep`, `InputBinder`, `read_metrics`), `trace.py`, `device/` (acquire, liveness, scope, profile), `mesh/` (multi-chip), `conv/`, `nn/` (Module), `layers.py` (Linear, LayerNorm, ReLU, Tanh, Sequential), `recurrent.py`, `carry.py` |
| `ttrl/algos/` | `ppo/` (config, agent, learn, loss, policy, recurrent GRU path), `dqn/`, `sac/`, `grads.py` (grad norm/clip), `gaussian.py` (Normal log-prob/entropy), `scaling.py`, `checkpoint.py`, `registry.py` |
| `ttrl/nets/` | encoders (vector MLP tower, Nature CNN), heads, `ActorCritic`, `ActingNet` (per-width views), `obs.py` (observation upload buffers) |
| `ttrl/buffers/` | `rollout.py` (PPO rollout, GAE, minibatches), replay buffers for DQN/SAC |
| `ttrl/envs/` | gymnasium/ALE/MuJoCo adapters, sync/async vector envs, `vec_normalize.py` (obs/reward normalization on the host) |
| `ttrl/train/` | `Trainer`, `loop.py`, `replicas.py`, `run.py`, `scoping.py`, `device.py`, `config/`, `checkpoint/` (npz + JSON), `evaluate.py` |
| `ttrl/cli/` | argparse shell with two commands: `train` and `doctor` (`ttrl/cli/__init__.py:18-21`) |
| `ttrl/deploy/` | numpy/ONNX export of a trained policy (no device) |

Entry points:

* Console script `ttrl = "ttrl.cli:main"` (`pyproject.toml:116-119`) -> `ttrl train <config.py> [dotted.key=value ...]` and `ttrl doctor`
  (`ttrl/cli/train.py`, `ttrl/cli/doctor.py`).
* Python API: `ttrl.load_config(path, overrides=[...])` then `ttrl.run(config)` (`README.md:106-113`; `ttrl/train/run.py:22-71`).
  `load_config` executes the config file as a Python module (`ttrl/train/config/loader.py:41`).
* Single-chip helper `ttrl.device_for(DeviceConfig)` = scope the process then `acquire()` (`ttrl/train/device.py:36-67`).
* PPO call path: `run()` (`train/run.py:22`) -> `scope_for_run` (`:52`) -> `acquire` (`:68`) -> `Trainer` -> `open_replicas`
  (`train/replicas.py:61`) -> `builders.build_agent` (`train/builders.py:117`) -> `PPO` (`algos/ppo/agent.py:42`) -> `run_loop`
  (`train/loop.py:162`) -> per round `Replica.step` (`loop.py:131-137`: `agent.act`, `env.step`, `agent.observe`) -> when the rollout is
  full (`agent.py:193-197`) `learn.update` (`algos/ppo/learn.py:46`).

## 3. The PPO implementation

### 3.1 Autograd versus manual gradients

* `ttrl/engine/autograd.py` defines `Tensor` (a node wrapping a `ttnn.Tensor`, `autograd.py:56-74`), `backward`, and `no_grad`
  (`:30-44`). Its docstring: "ttnn ships forward ops and a set of fused `*_bw` kernels but no tape" (`:3-5`).
* Each op in `ttrl/engine/ops/*.py` builds a forward result with ttnn and, if any input requires grad, attaches a `_backward(grad)`
  closure (e.g. `ops/matmul.py:55-86`). `Tensor.backward()` runs an explicit-stack post-order DFS (`autograd.py:137-153`), seeds the
  scalar loss with an fp32 tensor of ones (`:167-170`), calls each node's closure in reverse topological order (`:171-182`) and
  clears the tape in a `finally` (`:183-193`). It refuses non-scalar roots (`:155-163`) and a second backward on the same graph
  (`:130-135`). Gradients fan-in with `ttnn.add` in an fp32 accumulator (`GRAD_ACCUM_DTYPE`, `:16`, `_accumulate` `:106-114`).
* VJP rules actually used by the MLP+PPO path (all in terms of ttnn forward ops):
  * matmul: `dA = matmul(grad, B, transpose_b=True)`, `dB = matmul(A, grad, transpose_a=True)`, both at HiFi4/fp32-acc, then summed
    over broadcast batch dims (`ops/matmul.py:72-85`; the reduce helper is `ops/broadcast.py:8-30`).
  * bias add `(B,N)+(1,N)`: grad reduced with `ttnn.sum(grad, dim, keepdim=True)` (`ops/broadcast.py:26`).
  * relu: the one fused backward kernel used, `ttnn.relu_bw(grad, x)[0]` (`ops/unary.py:34`); exp/tanh/sigmoid reuse the saved forward
    output (`ops/unary.py:70-85,103-118,119-135`); log: `grad / x` (`unary.py:87-100`).
  * min/max/clamp are composed from relu identities (`ops/select.py:39-62`): `min(a,b) = a - relu(a-b)`; `clamp(x,lo,hi) =
    hi - relu(hi - (relu(x-lo)+lo))`. The module comment says native `ttnn.clip/clamp/minimum/maximum` and their `_bw` accept fp32 and
    are exact on 0.80 and ~2.3x cheaper, but the composition is kept because tt-rl's torch-twin test harvests relu sites
    (`select.py:1-30`). Gradient is 0 at the bound for the composition, 1 for the native kernels (`select.py:26-29`).
  * log_softmax is composed (`ops/functional.py:160-170`) with a detached max shift (`ops/reduce.py:71-79`); entropy
    `-sum(softmax * log_softmax)` (`functional.py:173-178`).
* The hand-written pieces are therefore per-op VJPs; the graph is dynamic Python. A traced step still builds the tape once during
  capture and replays the recorded device commands afterwards (section 4).
* The tape once leaked ~486 MB per update when backward used a recursive helper; the explicit stack is the fix (`autograd.py:139-141`).

### 3.2 Network (the "MLP")

* Observation encoder for vectors: `Linear -> ReLU` per hidden width, default `hidden=(256, 256)`; the first layer folds
  `ObsSpec.scale` into its weight (`ttrl/nets/encoders.py:50-71,147-172`).
* `Linear` (`ttrl/engine/layers.py:22-60`): weight stored `(in_features, out_features)` (TRANSPOSE of torch, `:43-47`), bias `(1, out)`,
  init `uniform(-1/sqrt(in), 1/sqrt(in))` from the GLOBAL numpy RNG (`:42,48-54`), forward `matmul(x, W)` then broadcast `add(bias)`
  (`:57-60`). Parameters are fp32 device tensors in TILE layout (`Tensor.from_numpy` default dtype, `autograd.py:78-82`;
  `engine/tensor/transfer.py:11-20`). No ELU exists anywhere in the package (grep for `elu` finds only `softplus` helpers in
  `algos/gaussian.py`). The init matches torch's default `nn.Linear` init (comment at `layers.py:25`; both weight and bias are drawn uniform(-1/sqrt(in), 1/sqrt(in))), which is also
  upstream's effective default: `UPSTREAM_SEMANTICS.md` S30 states the orthogonal `MLP.init_weights` is never called.
* `ActorCritic.forward` returns `(policy_out, value)`; for a symmetric observation ONE shared trunk is placed in both the actor and
  critic slots and is run once (`ttrl/nets/models.py:42-71`; built that way in `ttrl/nets/build.py:109-117`).
* Heads: categorical logits (`CategoricalHead`, `nets/heads.py:44`), scalar value (`ValueHead`, `:55`), and for continuous actions a
  `GaussianHead` with TWO linear heads `mean` and `log_std` over the trunk features, `log_std` clamped to `[-20, 2]`
  (`nets/heads.py:29-30,65-93`). This is a state-dependent log-std, unlike rsl_rl's scalar/log `std_param`.

### 3.3 Loss formulas (as implemented)

All in `ttrl/algos/ppo/loss.py:42-83`; `reduce` is `mean_all` for the feedforward path (`learn.py:81-83`):

```
ratio        = exp(new_logp - old_logp)                                   (loss.py:59)
surrogate    = min(ratio * A, clamp(ratio, 1-clip_eps, 1+clip_eps) * A)   (loss.py:60-63)
policy_loss  = - reduce(surrogate)                                        (loss.py:64)
v_clipped    = v_old + clamp(v - v_old, -value_clip_eps, +value_clip_eps) (loss.py:69-70)
value_loss   = 0.5 * reduce( max((v - R)^2, (v_clipped - R)^2) )          (loss.py:71-76)
entropy      = reduce(entropy_per_row)                                    (loss.py:77)
loss         = policy_loss + value_coef * value_loss - entropy_coef * entropy   (loss.py:79-80)
```

* The 0.5 on the value term is documented as the SB3/CleanRL convention (`loss.py:73-74`); upstream rsl_rl has no 0.5 (section 3.7).
* `clip_eps` and `value_clip_eps` are separate fields, both default 0.1 (`algos/ppo/config.py:75-76`). Other defaults:
  `lr 2.5e-4, gamma 0.99, lam 0.95, n_steps 128, minibatch_size 32, epochs 4, value_coef 0.5, entropy_coef 0.01, max_grad_norm 0.5,
  normalize_adv True, anneal_lr False` (`config.py:69-81`).
* Categorical log-prob: `one_hot_gather(log_softmax(logits), actions)`; the one-hot is built on the HOST per call and uploaded
  (`algos/ppo/policy.py:64-76`; `engine/ops/functional.py:98-109,135-153`). A device-only one-hot exists (`one_hot_device`, `:112-123`,
  `ttnn.eqz(lanes - idx)`, lane constants cached because a fresh upload is a new address, `:28-33`) but PPO's update does not use it.
* Gaussian log-prob `sum(-0.5 z^2 - log_std) - 0.5*dim*log(2*pi)` and exact entropy `sum(log_std) + 0.5*dim*(log(2*pi)+1)`
  (`algos/gaussian.py:64-71,90-99`).
* A padding-aware `masked_mean` (device-side valid count, so a trace does not freeze it) exists for the recurrent path
  (`loss.py:27-39`).

### 3.4 GAE, advantage normalization, minibatches (all on the HOST, numpy)

* `RolloutBuffer` holds `[n_steps, n_envs, ...]` numpy arrays (`ttrl/buffers/rollout.py:35-106`).
* `compute_gae` (`rollout.py:195-226`): loop `t = T-1..0` over lanes in numpy fp32:
  `delta = r + gamma * V_next * (1 - done) - V`, `adv = delta + gamma*lam*(1-done)*adv_prev`, multiplied by a `valid` mask that cuts the
  chain at autoreset padding slots (`:217-222`); `returns = advantages + values` (`:225`). `dones` stores `terminated` ONLY; time-limit
  truncation is handled by gymnasium's NEXT_STEP autoreset convention, not by adding `gamma*V` to the reward (`rollout.py:60-64`).
* `last_values` (bootstrap) is one device forward + download per update (`algos/ppo/learn.py:34-43`).
* Advantage normalization: ONCE per rollout, over the valid transitions only, `(adv - mean) / (std + 1e-8)` with numpy `.std()`
  (population, ddof=0) (`rollout.py:268-270`). There is no per-minibatch option.
* Minibatches (`rollout.py:241-282`): select valid slots (`:251`), `n_full = total // minibatch_size` and drop the remainder (`:253,271-273`;
  "a trailing partial batch is dropped rather than yielded ragged", `:244-247`), a FRESH `rng.permutation(total)` per epoch (`:272`), gather
  `obs[idx]`, `actions[idx]`, ... with numpy fancy indexing on the host (`:275-282`). Every minibatch is then uploaded (section 5).

### 3.5 Gradient-norm clipping

`clip_grad_norm(params, max_norm)` (`ttrl/algos/grads.py:29-43`): ONE global L2 norm over every parameter of the actor-critic
(`learn.py:98` passes `agent.net.parameters()`), computed on the host by downloading each gradient with `to_host` and summing squares in
float64 (`grads.py:13-26`); if `total > max_norm` every gradient is replaced by `ttnn.mul(grad, max_norm/(total+1e-6))` (`:38-42`).
This REBINDS `param.grad` to a fresh tensor (`:42`), which is incompatible with a trace (the trace-safe spelling is in
`TrainStep._rebind_grads`, `engine/step/core.py:120-133`). A device-only squared-norm exists for traced steps: `grad_sq_device`
(`grads.py:46-60`, `ttnn.sum(ttnn.mul(g, g))` per parameter then `ttnn.add`); it only measures, it does not clip. Nothing in tt-rl
clips on device.

### 3.6 Adam (device, fp32)

`ttrl/engine/optim/adam.py`:

* Runs entirely on device with primitive ttnn ops; no fused kernel: `ttnn.moreh_adam`'s dtype check lists only BFLOAT16 in 0.77, 0.80
  and 0.81-rc1, so it rejects fp32 parameters (`adam.py:7-10`; comment).
* dtype: parameters must be `MASTER_DTYPE = ttnn.float32` or the constructor raises (`adam.py:45-56`; `engine/tensor/precision.py:28`).
  Moments are `ttnn.zeros_like(p.data)` hence fp32 (`adam.py:71-72`). There is no separate "master copy": the fp32 parameter IS the master
  and is the tensor the forward uses (compute is fp32 as well, section 6). bfloat8_b "must stay unnamed" for gradients/optimizer state
  (`precision.py:19-21`).
* State layout is non-canonical: it stores `S = m/(1-b1)` and `V = v/(1-b2)` so that bias correction folds into two host scalars
  (`adam.py:82-83`; `engine/optim/scalars.py:13-27`: `alpha_t = lr*(1-b1)/(1-b1^t)/c`, `eps_t = eps/c`, `c = sqrt((1-b2)/(1-b2^t))`).
  Algebraically equal to torch Adam with eps added after the bias-corrected sqrt (derivation checked by hand; I did not run it).
  `state_dict` converts back to canonical `m`, `v` (`engine/optim/checkpoint.py:15-26`) and `load_state_dict` converts forward again
  (`:102-104`); `t`, per-parameter `steps`, `lr`, `betas`, `eps`, `weight_decay` are saved (`:16-24`).
* One update per parameter = 8 ttnn dispatches, all `output_tensor=` in place (`adam.py:120-134`): `mul(g,g)->s`; `addalpha(s,V,b2)->V`;
  `addalpha(g,S,b1)->S`; `sqrt(V)->s`; `add(s,eps_t)->s`; `div(S,s)->s`; `mul(s,alpha_t)->s`; `sub(p,s)->p`. An optional L2
  `weight_decay` adds one `addalpha(g, p, wd)` (`:110-114`); that is L2-in-gradient, not decoupled AdamW. Scratch buffers are
  allocated once at construction because "a device allocation in the eager gap between two trace replays can be handed an address the
  trace already owns" (`:73-80`; comment).
* Per-parameter update counts `steps[i]` drive bias correction (`:62-68,106-107`); `p.grad is None` parameters are skipped (`:104-105`).
* `lr` is a Python attribute read on every step; PPO's linear anneal just assigns it (`learn.py:54-56`).

### 3.7 Semantic differences versus upstream rsl_rl (secondary, from the local checkout)

Upstream references are `local:rsl_rl/algorithms/ppo.py` unless stated. These must be confirmed against the pinned tag.

| Topic | tt-rl PPO | Upstream rsl_rl (local checkout) |
|---|---|---|
| Actor/critic | One shared trunk when obs is symmetric (`nets/build.py:109-117`) | Separate `actor` and `critic` models, optimizer over both (`ppo.py:92-93,100-102`) |
| Activation | ReLU (`encoders.py:166-170`); no ELU | `MLPModel(activation="elu")` (`local:rsl_rl/models/mlp_model.py:37`) |
| Action std | State-dependent `log_std` linear head, clamp [-20,2] (`heads.py:65-93`) | Learned `std_param`, "scalar" or "log" parametrization (`local:rsl_rl/modules/distribution.py:149,165-167`) |
| Value loss | `0.5 * mean(max(...))`, own `value_clip_eps` (`loss.py:69-76`) | `mean(max(...))` with no 0.5, clip by the same `clip_param` (`ppo.py:278-281`); weight `value_loss_coef=1.0` default (`:50`) |
| Surrogate | `-mean(min(r*A, clip(r)*A))` (`loss.py:60-64`) | `mean(max(-A*r, -A*clip(r)))` (`ppo.py:269-274`); mathematically identical |
| Adv. normalization | Host numpy, population std, valid transitions, once per rollout (`rollout.py:268-270`) | `torch.std` (unbiased) once after GAE, or per minibatch if configured (`ppo.py:190-191,214-216`) |
| Time-outs | NEXT_STEP autoreset convention; `dones` = terminated only (`rollout.py:60-64`) | `rewards += gamma * V * time_outs` (`ppo.py:154-158`) |
| Grad clipping | one global norm over actor+critic (`learn.py:98`) | `clip_grad_norm_` called separately on actor and critic parameters (`ppo.py:309-310`) |
| LR schedule | optional linear anneal (`learn.py:54-56`) | adaptive KL schedule using analytic KL of distribution params, optional all-GPU averaging (`ppo.py:241-266`) |
| Minibatch order | fresh permutation every epoch (`rollout.py:272`) | one `randperm` per update shared by all epochs (`local:rsl_rl/storage/rollout_storage.py:232,243-248`) |
| Obs normalization | host-side env wrapper, `NormalizeObservation` (`envs/vec_normalize.py:78`) | in-model normalizers updated after the epochs (`ppo.py:328-330`) |
| KL metric | host k3 estimator for logging only (`learn.py:101-111`) | analytic Gaussian KL drives LR (`ppo.py:243`) |
| Gaussian acting log-prob | computed on the HOST in numpy (`policy.py:166-177`) while the update recomputes it on device (`:179-185`) | one torch method, `get_output_log_prob`, for both (`ppo.py:131,234`) |

The last row matters for BRIEF section 3.4: tt-rl's old and new Gaussian log-probs come from different code paths (host numpy vs
device fp32 composition); only the categorical case uses the same device `log_softmax` for both (`policy.py:44-62,74`).

## 4. Trace capture and replay

### 4.1 What is traced

| Component | Traced? | Where |
|---|---|---|
| PPO act, update, GAE | No | `algos/ppo/agent.py:233-236`; `algos/ppo/learn.py:17-19` |
| DQN update (online+target forward, Huber loss, backward, grad-norm sum) | Yes, mode `traced_fwd_bwd` (default) | `algos/dqn/agent.py:151-163`, `algos/dqn/losses.py:77-140`, `algos/dqn/config.py:65` |
| SAC critic region and actor(+alpha) region (two traces) | Yes | `algos/sac/step.py:63-96`, `algos/sac/agent.py:175-197` |
| SAC Polyak target blend | Yes (its own trace) | `algos/sac/targets.py:45-91` |
| Optimizer step (Adam) | NEVER inside a trace | `engine/step/core.py:35-40,95` |
| Acting forward (DQN/SAC/PPO) | No (eager, under `no_grad`) | `algos/dqn/agent.py:106-108`, `algos/ppo/agent.py:121-129` |
| Recurrent acting step | Designed for a "single-step traced loop" (carry with two buffers) | `engine/carry.py:1-43` (I did not read the call sites; unclear whether production code traces it) |

`TrainStep` has exactly two modes: `"eager"` and `"traced_fwd_bwd"` (`engine/step/core.py:18`). Only the forward + loss + `backward()` are
captured; the optimizer is eager, run after each replay (`core.py:88-97`).

### 4.2 Capture mechanics

* `capture(device, record)` (`engine/trace.py:95-126`) is a CALLBACK API, not a context manager, so no caller can leave a capture open
  (`trace.py:1-9`: an exception between `begin_trace_capture` and `end_trace_capture` "wedges the chip" until SIGKILL). Sequence:
  `ttnn.begin_trace_capture(device)` -> `record()` -> `ttnn.end_trace_capture(device, trace_id)` (`:108-111`). On any `BaseException`
  (incl. KeyboardInterrupt) it calls `_abort_capture` = `end_trace_capture` then `release_trace`, each in its own try/except
  (`:39-53,112-115`), then raises `TraceCaptureError` (KeyboardInterrupt/SystemExit are re-raised unchanged, `:116-117`; there is deliberately
  no eager fallback, `:34-37`). A host read inside capture
  is detected by matching the ttnn error text `"Reads are not supported during trace capture."` (`:29,118-124`).
* `CapturedTrace.replay(blocking=None)` calls `ttnn.execute_trace(device, trace_id, blocking=...)` (`:69-79`); default
  `blocking=False` (`engine/config.py:12-13`). `close()` calls `ttnn.release_trace` (`trace.py:81-86`), idempotent.
* Preconditions stated in `capture`'s docstring (`trace.py:96-106`): `record` must dispatch only device work, and "nothing whose program is
  not already in the program cache" (capture records commands, a first-time compile is not recorded), so "warm the cache by running the
  same work eagerly once beforehand". "Nothing has executed when this returns", so the caller must replay once before reading outputs.
* `TrainStep._capture` (`core.py:99-118`): `optimizer.zero_grad()`; warm-up pass `step_fn(inputs)` eagerly (`:106`) and discard its grads;
  `zero_grad()`; `capture(device, lambda: step_fn(inputs))` (`:110-112`); remember `(param, param.grad)` pairs (`:113-114`); one BLOCKING
  replay so the first call returns real numbers (`:116-118`). Each later `run()` = non-blocking `replay`, `_rebind_grads` (a replay runs
  no Python, so `param.grad` is re-pointed at the buffer the trace writes: `core.py:120-133`), eager `optimizer.step()`, then
  `read_metrics` (the metric download is the synchronization point; if there are no metrics it calls `ttnn.synchronize_device`,
  `core.py:135-141`).
* Sanity gates in tests: traced equals eager bit-for-bit (`tests/hardware/test_determinism_fingerprint.py:1-30`, enumerating
  `ttrl.engine.step.MODES`), and the program cache must not grow after warm-up (`tests/engine/test_step.py:195-262`, using
  `dev.num_program_cache_entries()`; `tests/engine/test_program_cache.py:35,39` uses `dev.set_program_cache_misses_allowed(False)`).

### 4.3 Buffers: how they are preallocated

* Invariant stated everywhere: "address stability". Nothing a trace touches may be re-bound; contents change, addresses do not
  (`engine/tensor/arena.py:1-6`; `nets/obs.py:99-106`).
* Inputs: DQN allocates all step inputs once, as zero tensors in `step_inputs` (`algos/dqn/losses.py:44-74`: rewards, dones, discounts,
  one-hot actions, per-leaf obs/next_obs, optional masks and IS weights with ones). SAC uses `BatchBuffers` (`algos/offpolicy.py:66-103`,
  load = `restore_in_place`). Observation buffers are `ObsBuffer`/`GroupBuffer`/`ObsViews` (`nets/obs.py:98-213`). `StaticArena`
  (`engine/tensor/arena.py:46-86`) exists and is tested but I found no production caller (grep over `ttrl/`).
* Writing into a bound buffer: `InputBinder.load(name, array)` (`engine/step/inputs.py:28-40`): host tensor from numpy
  (`ttnn.Tensor(arr, dtype).to(layout)`) then `ttnn.copy_host_to_device_tensor(host, t.data)`. Device-to-device in-place copy idiom:
  `ttnn.add(src, 0.0, output_tensor=dst)` (`engine/tensor/arena.py:21-32`); used for target sync and checkpoint restore
  (`engine/nn/module.py:164-189`, `copy_tensor_in_place`).
* Parameters and optimizer state are fp32 device tensors created once; restores go through `restore_in_place` (host tensor ->
  `copy_host_to_device_tensor`, `engine/tensor/transfer.py:23-39`; comment at `:30-32` says the host tensor may die on return in 0.80).
* Gradients: allocated by the capture pass's `backward()`; `TrainStep` stores them and re-binds before each optimizer step (see 4.2).
* Loop-invariant constants are cached per device (e.g. `_LANE_CACHE`, `ops/functional.py:28-33,77-84`; matmul compute config
  `_MATMUL_ACC_CFG`, `ops/matmul.py:25-39`), all keyed by `id(device)` and purged BEFORE `close_device` because `id()` is reused
  (`device/acquire.py:28-52,106-123`; `ops/matmul.py:22-26,48-52`).
* Scratch for Adam is preallocated for the allocation-in-the-gap hazard (`adam.py:73-80`).

### 4.4 Per-iteration changing values

* Data (rewards, dones, discounts, actions one-hots, masks, IS weights, SAC reparameterization noise) is passed by writing into the
  bound input buffers before each replay (`algos/dqn/agent.py:186-199`; `algos/sac/agent.py:218-227`, noise from the host RNG,
  `agent.py:223-225`).
* The learning rate and Adam bias-correction scalars are NOT traced values: the optimizer runs eagerly each step and passes host
  floats as op scalars (`adam.py:107,127,129`), because "its bias-correction scalars are functions of the update count, which a captured
  trace would freeze at capture time and then reapply forever" (`core.py:35-37`; comment). Moving the optimizer inside the trace was
  "measured at +1.3% to +1.7% against a 1.0% noise floor" (worse), because those dispatches were never on the critical path
  (`core.py:38-40`).
* In-trace scalars that must follow state are kept as device tensors: SAC's temperature is `exp(log_alpha)` computed INSIDE the region
  from a device parameter (`algos/sac/step.py:51-54`). Constants such as `tau`, `gamma`, clip epsilons, `huber_delta` are passed as Python floats (e.g. `algos/sac/targets.py:45-52`), so
  they are baked into the capture (inference from the code).
* Target-network hard sync (DQN) runs outside the trace via in-place copies every `target_sync_interval` updates
  (`algos/dqn/agent.py:205-208`).

### 4.5 How shapes are fixed

* The network is built for one batch size (`NETWORK_BATCH_FIELD = "minibatch_size"` for PPO, `algos/ppo/config.py:104`; `batch_size` for
  DQN/SAC), and `check_network_batch` refuses more lanes than that (`algos/scaling.py:75-101`). For conv nets this is a hard hardware
  limit (module docstring `algos/ppo/config.py:1-7` and `configs/pong_ppo.py:65-70`: minibatch 256 does not fit L1, 128 does).
* Acting at a different lane count uses a per-width "view" of the net with its own buffers (`nets/acting.py:69-114,117-160`); for a pure
  MLP the module comment says a matmul does not depend on the batch (`nets/encoders.py:147-153`) but acting still keeps one buffer set per
  width.
* Ragged tails are dropped or forbidden: PPO drops the trailing partial minibatch (`buffers/rollout.py:244-247,253`); the recurrent path
  requires `seq_len` to divide `n_steps` and `minibatch_size` "because one fixed T is what the captured graph is built for"
  (`algos/ppo/config.py:190-205`, error text). There is no tile-multiple (32) validation.
* A host-visible padding trap: multi-element tensors are tile-padded on device; reading a captured trace's multi-element OUTPUT through a
  packed concat returned zeros after the first replay, while a direct read was correct (`engine/step/metrics.py:30-44`). Hence
  `read_metrics` packs only scalar metrics into one download and reads per-sample metrics individually (`metrics.py:22-107`).

### 4.6 Trace region size

* A single integer `trace_region_size`, default 32 MiB (`engine/config.py:18-20`, comment: "holds the traced forward and backward of the
  largest net in this repo"), also `DeviceConfig.trace_region_size` default `32 * 1024 * 1024` (`train/config/device.py:37`;
  `configs/pong_ppo.py:31`). It is passed at device open time only: `ttnn.open_device(..., trace_region_size=...)`
  (`device/acquire.py:92-96`), `ttnn.CreateDevices(..., trace_region_size=...)` (`mesh/devices.py:85-88`),
  `ttnn.open_mesh_device(..., trace_region_size=...)` (`mesh/open.py:139-142`).
* There is NO sizing logic: no estimate of the required bytes, no check that several traces fit, no handling of an overflow other than the
  generic wrapping of any capture exception into `TraceCaptureError` (`trace.py:112-125`). (unclear what ttnn does on overflow; not
  addressed in the code or tests I read.) DQN's per-step trace, SAC's two traces and the Polyak trace all share the one region (inference).
* `l1_small_size = 32768` is a separate open-time setting needed for conv2d only (`engine/config.py:14-17`).

## 5. Host <-> device data flow

### 5.1 Per environment step (PPO, feedforward)

`Replica.step` (`train/loop.py:131-137`): `agent.act(obs)` -> `env.step(actions)` (host) -> `agent.observe(transition)`.

Inside `PPO.act` (`algos/ppo/agent.py:111-144`, under `no_grad`):

1. Upload the observation: `ActingNet.__call__` -> `ObsViews.load` -> `GroupBuffer.load` -> `ObsBuffer.load` -> `InputBinder.load` (`nets/acting.py:161-164`,
   `nets/obs.py:146-152,206-209`). Vector obs = `arr.reshape(batch, size).astype(float32)` (`nets/obs.py:72`); images are transposed NCHW->NHWC
   (`:74-75`). The host tensor is built with `ttnn.Tensor(arr, dtype).to(layout)` and copied with `ttnn.copy_host_to_device_tensor`
   (`engine/step/inputs.py:39-40`) = TILIZE ON THE HOST for inputs below 1 MiB (`inputs.py:13,51`). At or above 1 MiB (large image
   batches) it uploads row-major into a reused staging buffer and tilizes ON DEVICE (`ttnn.to_layout`, `ttnn.copy`) because the host
   tilize "is mostly padding" (8x inflated, single-threaded) (`inputs.py:42-84`; comment). A batch narrower than the buffer leaves the tail
   stale on purpose (`nets/obs.py:181-204`).
2. Forward in eager mode at the lane-count view (dispatch of every op from Python).
3. Downloads (each blocking, `Tensor.numpy()` -> `to_host` = `ttnn.typecast` if not fp32, `tensor.cpu()`, host
   `to(ttnn.ROW_MAJOR_LAYOUT)` untilize, `to_numpy()`: `engine/tensor/transfer.py:42-51`):
   categorical: `log_softmax(logits).numpy()` (`algos/ppo/policy.py:51`) + `value.numpy()` (`agent.py:141`) = 2 downloads;
   Gaussian: `mean.numpy()`, `log_std.numpy()` (`policy.py:168-169`) + value = 3 downloads.
4. Host: sampling (inverse-CDF or `mean + exp(log_std)*N(0,1)`), old log-prob, clipping to action bounds (`policy.py:51-62,166-177`).
5. `observe` stores into the numpy `RolloutBuffer` (`agent.py:180-187`). Rewards/dones never go to the device during acting.

So per env step: one upload per observation leaf, 2-3 blocking downloads, host sampling. Sync points are the downloads. This is above
the BRIEF target of "one action download per step".

### 5.2 Per PPO iteration (rollout full -> `learn.update`)

`algos/ppo/learn.py:46-73`:

1. Optional LR anneal (host float) (`:54-56`).
2. `last_values`: one obs upload + forward + `value.numpy()` (`:34-43`).
3. GAE, advantage normalization, permutation, minibatch gather: all host numpy (section 3.4).
4. Per minibatch (`learn.py:76-112`):
   * observation upload into the fixed `ObsBuffer` (`agent._obs(batch.obs)`, `:82`);
   * FOUR fresh device allocations per minibatch from host arrays: `old_logp`, `advantages`, `returns`, `old_value`
     (`loss.py:54-57`, `Tensor.from_numpy` = host tilize + `.to(device)`, `engine/tensor/transfer.py:20`), plus the host-built one-hot of the
     actions (`policy.py:75`) or the continuous action tensor (`policy.py:183-184`). These are not fixed-address buffers (no trace, so
     it is allowed);
   * forward, loss, `loss.backward()` (`learn.py:96-97`);
   * `clip_grad_norm`: one blocking download PER PARAMETER (`grads.py:24`);
   * `optimizer.step()` (device, `:99`);
   * metric downloads: `loss.numpy()`, `policy_loss`, `value_loss`, `entropy`, and `new_logp.numpy()` for the k3 KL (`learn.py:104-111`).
5. Buffer reset, `update_count += 1` (`:68-69`).

Everything that BRIEF section 3.3 wants resident on the device (rollout storage, advantages, minibatch gather, clip, Adam in the
trace) is on the host or eager here. DQN/SAC's off-policy replay buffers are also host-side (`ttrl/buffers/`); README line 8-9 states
"Environments and replay buffers stay on the host."

### 5.3 Other transfers

* Checkpoint: parameters and (canonicalized) Adam moments are downloaded to numpy (`engine/nn/module.py:107-115`; `optim/checkpoint.py:15-26`);
  written as one `.npz` with a JSON `__meta__` member, atomically, no pickle (`train/checkpoint/file.py:68-74`; `checkpoint/atomic.py:21-37`).
  This format is NOT upstream-compatible (section 11).
* Multi-chip averaging: whole parameter set device->host->device every `sync_every` updates (`mesh/sync.py:30-34`).

## 6. dtype, compute-kernel config, math fidelity

* Everything is float32 end to end: parameters, activations, gradients, grad accumulators, Adam state, losses, log-probs, advantages
  (`engine/tensor/precision.py:1-21`; `autograd.py:15-16`). The only non-fp32 path is a guarded bf16->fp32 download (`transfer.py:42-51`).
  bfloat16/bfloat8_b appear only in the metrics packer's allowed list (`step/metrics.py:12`). No bf16 experiments or flags exist for PPO.
* Matmul: every `ttnn.matmul` (forward and both backward matmuls) is called with
  `compute_kernel_config = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)`,
  memoized per device (`engine/ops/matmul.py:25-45`). Convolutions use the identical config (`engine/conv/config.py:22-58`). `math_approx_mode` and
  `packer_l1_acc` are NOT passed, so they are whatever ttnn defaults to (unclear: not stated in tt-rl). (`DeviceProfile.compute_kernel_config`,
  `device/profile.py:26-46`, has a helper with `math_approx_mode=True` default but the matmul path does not use it.)
* Measured accuracy floor (comment, p300c Blackhole, ttnn 0.80, fp32 `(8,16)@(16,32)` with entries up to 12, max abs error vs float64):
  LoFi 0.31, HiFi2 0.099, HiFi3 0.0166, HiFi4 0.0159, i.e. "roughly 1.3e-3 relative at the best setting and no kernel option gets below it"
  (`ops/matmul.py:10-20`). HiFi4 is "this hardware's accuracy ceiling, not full fp32" (`:14`).
* Eltwise/unary ops are called with NO compute-kernel config and no approximation-mode argument (`ops/unary.py`, `ops/elementwise.py`);
  measured fp32 backward errors: `relu_bw` 0.0, `tanh_bw` 2.09e-7, `sigmoid_bw` 1.64e-7, `exp_bw` 9.54e-7, `sqrt_bw` 1.91e-6; `clip_bw`/`clamp_bw`
  exact (`engine/tensor/precision.py:8-13`; `ops/unary.py:4-8`). The ttnn 0.80 docstrings list BFLOAT16 only and are said to be wrong
  (`precision.py:12-13`; `ops/select.py:13-14`). `log_bw` computes `G/X` and showed 1.22e-4 error near zero, so `log` uses `ttnn.div` (`unary.py:8-10`).
* Parity-test tolerances tt-rl uses (starting points only; BRIEF rule 9 requires our own noise-floor evidence): per-parameter
  gradient Pearson r > 0.9999 and relative Frobenius error < 1e-2 against a float64 torch twin (`tests/_fakes/torch_twin.py:31-39,257-282`),
  loss terms within 2e-2 relative (`tests/hardware/test_algos_ppo_parity.py:154-156`). The twin takes the ReLU gate pattern from the device
  as data because "around 7 entries per 32768 differ" between fp32 and float64 forward and one flipped gate moves a whole rank-1 term
  (`torch_twin.py:3-18,61-153`).
* Tile layout: all parameters and activations are TILE_LAYOUT fp32 (`transfer.py:20`); row-major only for conv observation buffers
  (`transfer.py:11-18`). No explicit `memory_config` is passed for matmul/eltwise (grep found `DRAM_MEMORY_CONFIG` only in conv weight
  preparation, `engine/conv/weights.py:143`), i.e. default (interleaved DRAM) placement.

## 7. Multi-chip: what it really does

Verified precisely (this is NOT synchronous gradient all-reduce):

1. Topology: one process, N INDEPENDENT single-chip devices opened together with `ttnn.CreateDevices(local_ids, l1_small_size=...,
   trace_region_size=...)` and `enable_program_cache()` each (`engine/mesh/devices.py:46-116`; comment `:54-56`: sequential `open_device`
   per chip "deadlocks on real hardware once a parameter is downloaded; the batch CreateDevices/CloseDevices API ... does not").
   Called from `train/replicas.py:219-246` when `device.chips != 1`.
2. Replicas: one full `agent` + env + optimizer + (for DQN/SAC) traces per chip (`train/replicas.py:61-128`). Each replica steps its own
   environments and learns from its own data; the loop drives replicas sequentially from one Python thread (`train/loop.py:201-206,225-233`).
   Config budgets measured in frames are divided by N (`algos/scaling.py:34-72`; for PPO only `total_frames`,
   `algos/ppo/config.py:119`), `n_steps` is not divided, so N replicas collect N times the experience per update round (`README.md:142-145`).
3. Synchronization: `ParamSync.maybe_average()` after every round in which updates happened (`train/trainer.py:203-208`), averaging every
   `sync_every` calls (`mesh/sync.py:128-136`, default `DeviceConfig.sync_every = 50`, `train/config/device.py:44`). For PPO one call = one whole
   rollout update (all epochs and minibatches). `average` also runs before every evaluation and checkpoint (`train/periodic.py:39-40,66-67`).
4. Averaging mechanics (`mesh/sync.py:138-157,228-249`): for each parameter, download every replica's tensor to numpy fp32 (`:231`), sum on the
   host (`host_all_reduce_sum`, `:60-70`), divide by N and `restore_in_place` into each replica's buffer (`:244-249`). It is "not atomic"
   (`:152-158`). Adam moments are never synced (`:143-144`); `reset_optimizer_state` zeros them when the chip count changes on resume
   (`mesh/resume.py:10-33`). Gradients are never exchanged.
5. Why host: a MeshDevice and independently-opened devices on the same chips cannot coexist in one process
   (`MeshDeviceImpl::ScopedDevices adopts ALL devices active in the context`, `sync.py:15-34`); `ttnn.all_reduce` consumes one multi-device tensor
   on one MeshDevice; the "shards 1 and 2 both come back 3" bug was measured on an OLDER ttnn and "the figure cannot be claimed for 0.80";
   0.80's `all_reduce` keyword set changed (`sync.py:3-13`; comment). So the device collective was never run on 0.80 in this repo.
6. A mesh path exists as groundwork only: `open_mesh_for_chips` sets `FabricConfig.FABRIC_1D` (needs >= 2 chips; "hard error below two chips")
   BEFORE `ttnn.open_mesh_device(l1_small_size=..., trace_region_size=...)` with NO shape/device ids (ttnn auto-discovers), validates the device
   count after the open, and resets fabric to DISABLED after `close_mesh_device` (`mesh/open.py:103-188`). `replica_axis` reads the replica axis
   off the mesh shape because "a P300 pair auto-discovers as MeshShape([2, 1])" (`:75-100`). One MeshDevice per process is a hard limit; a second makes
   the first hang forever in `synchronize_device` (`:47-72`). Nothing in the training path uses the mesh.
7. Initial consistency: `ParamSync.broadcast(src=0)` exists so replicas start identical, because averaging N independent inits shrinks the weight
   scale by sqrt(N) (`sync.py:179-193`; test `tests/engine/test_mesh_broadcast.py:14-33`). I found NO call to `broadcast` in `ttrl/` (grep: only the
   definition); the test docstring refers to a `_build_multi_chip` that does not exist at this commit (`tests/engine/test_mesh_broadcast.py:15`).
   Replicas receive the SAME agent seed (`train/builders.py:140` passes `config.seed` to every replica; only the ENV seed is offset by
   `index * SEED_STRIDE`, `train/replicas.py:31,123`), so with a global `np.random.seed(seed)` before network construction
   (`algos/ppo/agent.py:75-76`) their initial weights are probably identical (inference), but their action/permutation RNG streams are
   identical too, contradicting the comment that they "stay distinct" (`sync.py:190-192`). (unclear / possible discrepancy.)
8. Consistency checks worth noting: `param_schema`/`replica_schema` compare parameter NAMES and shapes across replicas before any positional
   operation (`mesh/sync.py:73-99,209-218`).

Consequence for us: tt-rl contains no reusable code for synchronous gradient all-reduce, and its measured reasons against mixing mesh and
independent devices are directly relevant to a MeshDevice-native design (one mesh per process; fabric config before open; discovery-based shape).

## 8. RNG handling

| Purpose | Mechanism | Where |
|---|---|---|
| Weight init | GLOBAL numpy RNG `np.random.uniform`, seeded by `np.random.seed(seed)` in the agent constructor | `engine/layers.py:48-54`; `algos/ppo/agent.py:75-76` |
| Action noise / sampling | `np.random.default_rng(seed)` per agent; categorical = inverse-CDF with one `rng.random(batch)` per lane (per factor for multi-discrete); Gaussian = `mean + exp(log_std) * rng.standard_normal(mean.shape)` | `algos/ppo/agent.py:102,130-131`; `algos/ppo/policy.py:57-58,119-120,173` |
| Eval acting | separate stream `default_rng(seed).spawn(1)[0]` so eval cadence does not perturb training | `agent.py:103,130` |
| Minibatch permutation | the rollout buffer's own `default_rng(seed)`, `permutation(total)` once per epoch, gather on host | `buffers/rollout.py:106,272-282` |
| SAC noise | host `standard_normal` uploaded into a bound buffer before each traced step | `algos/sac/agent.py:223-225` |
| Checkpointing | `rng_state` = `_rng.bit_generator.state` and the buffer's `batch_rng` are saved/restored; the GLOBAL numpy RNG used for init is not | `algos/ppo/state.py:38,63`; `algos/ppo/partial.py:81,113` |

Notes: the agent `_rng` and the buffer `_rng` are seeded with the same `seed` (`agent.py:102`, `buffers/rollout.py:106` via `seed=seed`,
`agent.py:83-88`), so their streams coincide (inference; harmless for correctness but not independent). There is no device RNG anywhere; this
is exactly BRIEF section 3.5's "parity mode" (host noise uploaded into a fixed buffer), which SAC already does.

## 9. Device-open code and device selection (FLAGGED)

Our port must have exactly one guarded entry point and must never auto-pick a chip or touch a node outside the tray (BRIEF section 2). Every
site in tt-rl that opens a device or chooses one:

| What | Where | Why it is a problem for us |
|---|---|---|
| `acquire(chip_id=None)`: candidates = `enumerate_chips()` (every `/dev/tenstorrent/<n>`), first free chip wins, otherwise polls `time.sleep(5)` up to 120 s | `engine/device/acquire.py:56-140` (`:76,85-124,140`), `engine/device/liveness.py:23-28,89-110` | auto-pick and wait loop; enumerates all nodes |
| `ttnn.open_device(device_id=..., l1_small_size=..., trace_region_size=...)` | `acquire.py:92-96` | direct `ttnn.open_device` call (our rule 2 allows one such call, in `device.py`) |
| `status()`/`survey()`: `subprocess.run(["fuser", "/dev/tenstorrent/<n>"])` for each chip, plus flock files under `/tmp/ttrl-locks` | `liveness.py:158-182,217-233,29,262-299` | queries (and locks) chips regardless of tray membership; writes into `/tmp` |
| `scope_for_run`: if `TT_VISIBLE_DEVICES` is unset, scope to `_first_free_chip()` = lowest chip with `status(c).free`, else `chips[0]`; then `os.environ.update` | `train/scoping.py:34-89` (`:54,77-89`), `engine/device/scope.py:199-223` | auto-pick; mutates the environment in-process |
| `ttrl.run` / `device_for` always call `scope_for_run` then `acquire()` | `train/run.py:52,68-71`; `train/device.py:62-67` | entry points that open a chip |
| `resolve_chips("all" or int)` = `discovered[:n]` | `engine/mesh/topology.py:36-58` | auto-pick |
| `open_independent_devices_for_chips` -> `ttnn.CreateDevices` | `engine/mesh/devices.py:46-116` | multi-open path |
| `open_mesh_for_chips` -> `ttnn.set_fabric_config` + `ttnn.open_mesh_device` with NO device ids (discovery over everything visible) | `engine/mesh/open.py:103-188` (`:133-142`) | safe only if the visible set is exactly the allowed set |
| `ttrl doctor`: picks `free[0]` among all enumerated chips, spawns `python -m ttrl.cli.probes.chip_report <chip>` with a scoped env, which does `acquire(chip_id=0, wait=False)` and a capture/replay probe | `cli/doctor.py:21-52`; `cli/probes/hardware.py:41-103` (`:57-66,94-98`); `cli/probes/chip_report.py:29-41,44-72` | explicitly forbidden by BRIEF rule 2-4 |
| Hardware tests open `acquire(0)` / `device.acquire()` unscoped | e.g. `tests/hardware/test_algos_ppo_parity.py:27-30`; `tests/engine/test_program_cache.py:47` | would open `/dev/tenstorrent/0` or the first free chip; do not run or copy |
| tt-rl never executes `tt-smi` (only quoted in error text) | `engine/mesh/sync.py:26`, `engine/mesh/devices.py:69` | for the record: no reset call exists in the library |

Useful facts recorded in comments (all for ttnn 0.80, so re-verify):

* `ttnn.open_device(device_id=N)` does not isolate chip N: UMD maps hugepages and takes the driver mutex `CHIP_IN_USE_<n>_PCIe` for EVERY local chip,
  and discovery locks every visible node at `import ttnn` (`engine/device/scope.py:7-14`, `engine/mesh/topology.py:77-85`). `TT_VISIBLE_DEVICES` is read
  once at ttnn init (`scope.py:28`), so it must be set before `import ttnn`, and then `device_id` is a LOCAL index into the visible set
  (`scope.py:93-134`). This supports our single guarded entry point plus `scripts/tray_env.sh`.
* Splitting a paired board with `TT_VISIBLE_DEVICES` needs `TT_MESH_GRAPH_DESC_PATH` pointing at `p100_mesh_graph_descriptor.textproto`
  inside the ttnn install, else "Custom fabric mesh graph descriptor path must be specified" (`scope.py:29-40`; `cli/probes/traps.py:41-77`). Board-specific; our tray topology is unknown (BRIEF section 0), so do not copy.
* ttnn keeps its `/dev/tenstorrent/*` fds open for the life of the process even after `close_device` (six fds measured), so a child process sees
  its parent as a holder (`liveness.py:40-50`). Relevant for a `/proc/self/fd` guard that runs after open and again at exit.
* Cache purge before close and `try/finally` ordering that does not let a failing purge skip `close_device` (`acquire.py:106-123`) are good patterns.
* `device_already_open()` and `_holds_open()` read `/proc/self/fd` and compare link targets with `/dev/tenstorrent/` (`scope.py:137-148`;
  `liveness.py:147-155`): the same mechanism BRIEF rule 2(b) asks for.

## 10. TTNN APIs used by tt-rl (exact spellings, to be looked up and re-verified on v0.81.1)

All are used in tt-rl on ttnn 0.80. A "check" means: confirm signature, dtype support (fp32), trace-capture legality and behaviour on 0.81.1 in
tt-metal's own docs/tests before relying on it. (I did not verify any of them.)

Device / runtime
* `ttnn.open_device(device_id=, l1_small_size=, trace_region_size=)`, `ttnn.close_device(dev)` (`acquire.py:92,123`);
  `ttnn.CreateDevices(ids, l1_small_size=, trace_region_size=)` -> dict, `ttnn.CloseDevices(dict)` (`mesh/devices.py:85,116`);
  `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D | DISABLED)`, `ttnn.open_mesh_device(l1_small_size=, trace_region_size=)`,
  `mesh.get_device_ids()`, `mesh.shape`, `ttnn.close_mesh_device(mesh)` (`mesh/open.py:133-187,164`).
* `device.enable_program_cache()` (`acquire.py:100`), `device.num_program_cache_entries()` (`tests/engine/test_step.py:217`),
  `device.set_program_cache_misses_allowed(bool)` (`tests/engine/test_program_cache.py:35`), `ttnn.synchronize_device(device)` (`step/core.py:141`).
* Introspection: `device.arch()`, `ttnn.get_arch_name()`, `ttnn.Arch.BLACKHOLE`, `device.compute_with_storage_grid_size()`, `device.dram_grid_size()`,
  `device.get_num_devices()`, `ttnn.TILE_SIZE` (`device/profile.py:61-77`).
* `ttnn.init_device_compute_kernel_config(arch, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True[, math_approx_mode, packer_l1_acc,
  dst_full_sync_en])` (`ops/matmul.py:33-37`, `device/profile.py:39-46`).

Trace
* `ttnn.begin_trace_capture(device)` -> trace_id, `ttnn.end_trace_capture(device, trace_id)`, `ttnn.execute_trace(device, trace_id, blocking=bool)`,
  `ttnn.release_trace(device, trace_id)` (`trace.py:108,111,79,51,86`).

Tensors and transfer
* `ttnn.Tensor(np_float32_array, ttnn.float32)` host tensor; `.to(ttnn.TILE_LAYOUT | ttnn.ROW_MAJOR_LAYOUT)`; `.to(device)`; `.cpu()`; `.to_numpy()`
  (`engine/tensor/transfer.py:20,50-51`); `ttnn.copy_host_to_device_tensor(host, dst)` (`transfer.py:38`, `inputs.py:40,73`); `ttnn.typecast(t, ttnn.float32)`
  (`transfer.py:49`); `ttnn.is_tensor_storage_on_device(t)` (`arena.py:30`; the comment notes `ttnn.is_device_tensor` was unbound in 0.80);
  `ttnn.to_layout(t, layout)`, `ttnn.copy(src, dst)`, `ttnn.deallocate(t)`, `ttnn.reshape(t, shape)` (`inputs.py:74-84`);
  `ttnn.zeros_like(t)` (`arena.py:69`, `adam.py:71`), `ttnn.full_like(t, value)` (`autograd.py:167,255`); `t.get_layout()`, `t.dtype`, `t.shape`, `t.device()`,
  `t.storage_type()`.

Math
* `ttnn.matmul(a, b, compute_kernel_config=, transpose_a=, transpose_b=)` (`matmul.py:45,80,83`).
* `ttnn.add/sub/mul/div` (tensor-tensor with broadcast, tensor-scalar, and `output_tensor=` for in-place writes), `ttnn.rsub(t, scalar)`, `ttnn.neg`,
  `ttnn.subtract` (`ops/elementwise.py:19-134`, `functional.py:123`), `ttnn.addalpha(a, b, alpha, output_tensor=)` (`adam.py:111,121-122`).
* `ttnn.relu`, `ttnn.relu_bw(grad, x)[0]`, `ttnn.abs`, `ttnn.sign`, `ttnn.exp`, `ttnn.log`, `ttnn.tanh`, `ttnn.sigmoid`, `ttnn.sqrt` (also with `output_tensor=`)
  (`ops/unary.py:26-139`, `adam.py:126`).
* `ttnn.sum(t)` and `ttnn.sum(t, dim=, keepdim=)`, `ttnn.max(t, dim=, keepdim=)`, `ttnn.argmax(t, dim=, keepdim=)` (returns uint32 -> typecast to fp32)
  (`ops/reduce.py:14,48,79,98`).
* `ttnn.concat(list, dim=)`, `ttnn.slice(t, begins, ends, strides)`, `ttnn.repeat(t, reps)`, `ttnn.squeeze(t, 0)`, `ttnn.reshape` (`ops/shape.py:24,42,52`,
  `ops/broadcast.py:29,54`); `ttnn.eqz`, `ttnn.isfinite`, `ttnn.logical_not` (`functional.py:123,194`).

Looked at and rejected, with reasons recorded by tt-rl
* `ttnn.moreh_adam`: check_tensor accepts BFLOAT16 only (`adam.py:7-10`).
* `ttnn.clip/clamp/minimum/maximum` and `*_bw`: fp32-correct on 0.80 and ~2.3x cheaper, unused only for test-harness reasons (`ops/select.py:1-30`).
* `ttnn.gather`/`ttnn.scatter_add`: need UINT16/UINT32 indices and have "no bound backward" (`ops/functional.py:10-13`); never exercised.
* `ttnn.layer_norm`, `ttnn.moreh_layer_norm_backward`, `ttnn.moreh_logsoftmax_backward`: no bound backward on the tape (`engine/layers.py:70-74`,
  `ops/functional.py:8-10`).
* `ttnn.all_reduce`: signature changed between versions; never run on 0.80 here (`mesh/sync.py:3-13`).
* `ttnn.zeros` inside capture: "Writes are not supported during trace capture" (`ops/shape.py:122-125`).

## 11. Classification for our port

Assumed design constraints (BRIEF section 3): fixed shapes + trace everywhere; fp32 master weights and Adam state; synchronous gradient
all-reduce (MeshDevice-native, 1x1 -> 1x2 -> tray); upstream-rsl_rl-compatible cfg and checkpoint format; parity-testable against upstream torch
modules; one guarded device-open entry point.

### 11.1 TAKE AS-IS (copy or transliterate with minimal change, MIT notice required for every copied file)

These are small, self-contained, algorithmic or mechanical, and carry no semantic coupling to tt-rl's PPO. "As-is" still implies re-verification on
ttnn v0.81.1.

1. `ttrl/engine/trace.py` (126 lines): callback `capture()`, abort/release path on `BaseException`, `CapturedTrace`, host-read error detection. Exactly the
   safe-capture pattern our rules need (wedge on exception).
2. In-place copy idiom and fixed-address discipline: `copy_in_place` (`ttnn.add(src, 0.0, output_tensor=dst)`), `restore_in_place`
   (`engine/tensor/arena.py:21-43`, `engine/tensor/transfer.py:23-39`).
3. `InputBinder` and `_staging_shape` (`engine/step/inputs.py`) as the model for obs/reward/done uploads into bound buffers, including the >= 1 MiB
   device-tilize staging path. Re-measure thresholds on our boards.
4. Scalar-metric packing + per-sample caveat: `read_metrics` (`engine/step/metrics.py:22-107`), as the one-download-per-iteration logging path.
5. Adam algebra as a REFERENCE ORACLE: `adam_fused_scalars` (`engine/optim/scalars.py`) and the canonical m/v conversion in `AdamCheckpointMixin`
   (`engine/optim/checkpoint.py:15-26,102-104`), which is how a device state maps to torch's `exp_avg`/`exp_avg_sq`.
6. Per-op VJP formulas and compositions verified against torch in tt-rl: matmul backward with transpose flags, broadcast-gradient reduction
   (`ops/matmul.py:72-85`, `ops/broadcast.py:8-55`), log_softmax/entropy, `normal_logp`, `gaussian_entropy` (`algos/gaussian.py:64-99`), stable
   softplus, relu-identity min/max/clamp (`ops/select.py`).
7. Device-side `grad_sq_device` pattern for a trace-legal global gradient norm (`algos/grads.py:46-60`).
8. HiFi4 + `fp32_dest_acc_en` matmul config memoization with purge-before-close (`ops/matmul.py:10-52`), and the cache-purge discipline
   (`device/acquire.py:28-52`).
9. `param_schema`/`replica_schema` name-and-shape consistency checks (`mesh/sync.py:73-99`) as the basis of our periodic cross-chip consistency check.
10. Test methodology (not code to ship): float64 torch twins built from the device weights, `DeviceGates` relu-gate harvesting
    (`tests/_fakes/torch_twin.py:61-168`), DRAM-flatness leak test (`tests/engine/test_dram_flatness.py`), program-cache stability gates
    (`tests/engine/test_step.py:195-262`), traced == eager fingerprint (`tests/hardware/test_determinism_fingerprint.py`).

### 11.2 ADAPT (what must change)

| tt-rl piece | Required change for our design |
|---|---|
| Eager PPO update (`learn.py`, `loss.py`) | Rebuild as fixed-shape traced steps: (a) act, (b) minibatch update (fwd+bwd+clip+Adam), (c) GAE. tt-rl proves only the DQN/SAC "forward+backward in a trace, optimizer outside" shape. The formulas in `loss.py:59-80` are a good oracle for the loss, but see the semantic deltas in 3.7 (value-loss 0.5 and separate clip eps must go; use upstream's). |
| Tape autograd | BRIEF ADR-0001 (A) wants hand-written static fwd/bwd. Reuse the VJP rules; the tape can serve as an in-process oracle or capture-time builder (tt-rl DQN builds the tape during capture and replays), but avoid per-step Python tape construction in the hot path. |
| Adam outside the trace with host scalar LR/bias correction | Optimizer inside the trace. `lr`, `1/(1-b1^t)`, `1/(1-b2^t)` (or the folded `alpha_t`, `eps_t` of `adam_fused_scalars`) must become device tensors written in place before each replay or computed on device from an in-trace step counter. Candidate approaches: host-computed scalars uploaded into bound `(1,1)` buffers per replay (tiny upload), or device-side `exp(t*log(b))`. Neither is demonstrated in tt-rl; look up and test. All ops need `output_tensor=` / fixed addresses. KL-adaptive LR (upstream) also needs a device LR tensor. |
| `clip_grad_norm` (host download per parameter, rebinding) | Device-only: sum of squares (reuse `grad_sq_device`), sqrt, scale = min(1, max_norm/(norm+1e-6)) as a device tensor, in-place multiply of every grad buffer. Upstream clips actor and critic SEPARATELY (`ppo.py:309-310`), not one global norm; follow upstream (confirm in `UPSTREAM_SEMANTICS.md`). |
| Host GAE, advantage normalization, permutation and gather (`rollout.py`) | Move storage, GAE and advantage normalization on device (BRIEF 3.3). Use upstream's `torch.std` (unbiased), optional per-minibatch normalization, time-out bootstrapping via `gamma*V*time_outs` (`ppo.py:154-158`), done-mask GAE (`ppo.py:176-191`). Permutation: host `torch.randperm` once per update and index upload (upstream shares one permutation across epochs, `rollout_storage.py:232`); row gather on device. tt-rl never gathers on device; `ttnn.gather`/embedding semantics for fp32 rows must be looked up. |
| Obs upload per step and 2-3 downloads per step | Preallocated fixed-shape obs buffer (tt-rl's `ObsBuffer`/`InputBinder` pattern), act traced, one action download per step. Sampling noise: host generator into a bound buffer (SAC pattern, `sac/agent.py:223-225`) for parity mode. Log-prob of the sampled action must come from the same device path as the update (fixes tt-rl's host-vs-device Gaussian log-prob split, `policy.py:166-185`). |
| Network | Separate actor and critic MLPs, ELU (+ backward), scalar/log `std_param` (not a state-dependent head), EmpiricalNormalization on device. tt-rl has none of ELU, scalar std or on-device normalization (its `Linear` init already equals torch's default, so no change is needed there). Linear layout `(in, out)` (`layers.py:43-47`) vs torch `(out, in)`: transpose at the checkpoint boundary. |
| Parameter init RNG (global numpy) | Parity tests should load weights from the torch module; init RNG is irrelevant to parity. |
| Metrics | Keep `read_metrics`-style packing; do not read multi-element trace outputs through a concat. |
| Trace region | Add explicit sizing: measure the bytes of act+update+GAE traces and set `trace_region_size` in the single device entry point; tt-rl has no sizing or overflow handling (4.6). |
| Replica consistency | `ParamSync.broadcast`/`replica_schema` ideas -> periodic parameter-consistency check across chips (BRIEF 3.6), not a training mechanism. |
| Checkpoint | tt-rl's npz+JSON format (`train/checkpoint/file.py:68-103`, own `FORMAT` token, no pickle, atomic write `checkpoint/atomic.py`) is not upstream-compatible. We need upstream's `{actor_state_dict, critic_state_dict, optimizer_state_dict, ...}` via `torch.save` (`ppo.py:374-385`). Reuse only: atomic write-then-rename (`atomic.py:21-37`), canonical-moment conversion, "validate every key and shape before touching device buffers" in `Module.load_state_dict` (`engine/nn/module.py:117-162`). |
| Device open helpers | Keep only: `/proc/self/fd` scanning (`scope.py:137-148`), cache purge/close ordering, `DeviceProfile` probing (`device/profile.py:61-77`). Rewrite as one guarded entry: env check against `scripts/tray_env.sh`, fd subset check after open (allowed nodes 16-23), close on mismatch. |
| ttnn version | Re-verify every item in section 10 on v0.81.1; tt-rl is pinned to 0.80 and says minor bumps break APIs. |
| Parity tolerances | Use tt-rl's 0.9999 / 1e-2 / 2e-2 only as hypotheses; derive ours from measured noise floors (BRIEF rule 9). |

### 11.3 DO NOT USE

1. Periodic parameter averaging multi-chip (`mesh/sync.py` `ParamSync` as a training mechanism, `host_all_reduce_sum`, `open_independent_devices_for_chips`,
   `train/replicas.py` replica model, `device.chips`/`sync_every`): the semantics differ from rsl_rl multi-GPU (synchronous gradient averaging), the
   reduction is a host numpy round trip of all parameters, Adam moments diverge per chip, it never initializes replicas via `broadcast` in the code I read,
   and it is sequentially driven from one thread.
2. The CLI: `ttrl train`, `ttrl doctor` and `ttrl/cli/probes/*` (BRIEF rule 2-4; `doctor` opens a free chip).
3. Any chip auto-selection or liveness scan: `acquire(chip_id=None)`, `scope_for_run`/`_first_free_chip`, `enumerate_chips`/`physical_chips`/`survey`/`status`
   (glob of all `/dev/tenstorrent/*`, `fuser`), `resolve_chips("all"|n)`, `/tmp/ttrl-locks` lock directory, `ttrl.run`, `device_for`, the hardware tests'
   `acquire(0)` fixtures, and `open_mesh_for_chips` with ttnn auto-discovery unless the visible set is exactly our tray.
4. The `TT_MESH_GRAPH_DESC_PATH` / `p100_mesh_graph_descriptor.textproto` single-chip hack (`scope.py:29-40`): board-specific, our topology is unknown.
5. tt-rl's PPO semantics as an oracle (shared trunk, ReLU, state-dependent log-std head, 0.5-scaled value loss with separate clip eps, global grad-norm clip,
   linear LR anneal only, numpy ddof=0 advantage std, NEXT_STEP autoreset handling, per-epoch fresh permutation). Upstream rsl_rl is the oracle.
6. The eager host-heavy PPO update as a production path (per-parameter downloads in `clip_grad_norm`, per-minibatch fresh uploads, host GAE). Acceptable only
   as a debug/eager reference path.
7. `clip_grad_norm`'s `param.grad = ttnn.mul(...)` re-binding (`grads.py:42`) inside any traced region.
8. Everything outside v1 scope: DQN, SAC, replay buffers, gymnasium/ALE/MuJoCo env layer, conv stack, recurrent GRU path, deploy/ONNX export, TrainConfig/Trainer/
   loggers/trackers, the `uv.lock` pins. tt-rl's `numpy<2` / Python 3.12 pins (`pyproject.toml:40-53`) are irrelevant to our `PYTHON_ENV`.

## 12. Hazards and lessons recorded in tt-rl comments (all ttnn 0.80; re-verify)

| Lesson | Evidence |
|---|---|
| An exception between begin/end trace capture wedges the chip until SIGKILL; capture through a callback with a `BaseException` abort path | `engine/trace.py:1-9,39-53,112-117`; `cli/probes/traps.py:17-20,41-81` |
| No host READ and no host WRITE during capture (`ttnn.zeros` is a write); the error string is `"Reads are not supported during trace capture."` | `trace.py:29`; `ops/shape.py:122-125`; `ops/reduce.py:91-93` |
| Programs must be compiled before capture (warm-up eager pass) and capture executes nothing: a first replay is needed | `trace.py:96-106`; `step/core.py:106,116-118` |
| A replay runs no Python: Python-side `param.grad` must be re-pointed at the traced buffers | `step/core.py:120-133` |
| A device allocation between two replays can be handed an address the trace owns, so scratch is preallocated | `optim/adam.py:73-80`; `ops/functional.py:28-33`. Note: `read_metrics` itself allocates a concat tensor after replay (`step/metrics.py:107`) - how this is reconciled is unclear |
| Reading multi-element tile-padded trace outputs via a packed concat returned zeros after the first replay | `step/metrics.py:30-44` |
| Optimizer inside the trace freezes bias-correction scalars | `step/core.py:35-37` |
| `moreh_adam` rejects fp32; bf16 moments would flatten training (`b2=0.999`) | `optim/adam.py:7-10,45-56` |
| `*_bw`, clip, min/max accept fp32 on 0.80 despite BFLOAT16-only docstrings | `ops/unary.py:4-10`; `ops/select.py:7-14`; `tensor/precision.py:8-13` |
| HiFi4 is the fp32 accuracy ceiling (~1.3e-3 relative) | `ops/matmul.py:10-20` |
| Reverse-mode tape held by a recursive helper leaked ~486 MB/update | `autograd.py:139-141`; `tests/engine/test_dram_flatness.py` |
| Device-id-keyed caches must be purged before `close_device` (`id()` reuse gives silent wrong arithmetic) | `device/acquire.py:37-52`; `ops/matmul.py:22-26` |
| `import ttnn` locks every visible node; `TT_VISIBLE_DEVICES` is read once at init | `device/scope.py:7-14`; `mesh/topology.py:77-85` |
| ttnn keeps `/dev/tenstorrent` fds after `close_device` | `device/liveness.py:40-50` |
| Sequential `open_device` per chip deadlocks after a parameter download; use batch `CreateDevices` | `mesh/devices.py:54-56` |
| Mesh and standalone devices cannot coexist; second mesh in a process hangs `synchronize_device`; FABRIC_1D needs >= 2 chips and must be set before open | `mesh/sync.py:15-34`; `mesh/open.py:47-72,124-136` |
| `ttnn.Tensor.cpu()` has no timeout; a hung chip blocks forever | `mesh/sync.py:39-45` |
| Conv-only: on 0.77, ~6.4% of traced DQN updates diverged until an override; not reproducible on 0.80 | `engine/conv/config.py:88-128` |
| Tests may be stale: `tests/engine/test_program_cache.py:77` builds `ParamSync(mesh=mesh, ...)` and patches `_write_device_result_in_place`, neither of which exists in `mesh/sync.py` at this commit | `mesh/sync.py:103-116` |

## 13. License obligations and open questions

### 13.1 License (MIT)

* Text: `LICENSE:1-21`; copyright holder "Antoni Bertel", year 2026. The one condition: "The above copyright notice and this permission notice shall
  be included in all copies or substantial portions of the Software." (`LICENSE:12-13`). The software is provided without warranty (`LICENSE:15-21`).
* Our repository is BSD-3-Clause (upstream rsl_rl, `LICENSE` at repo root); MIT is permissive and compatible, so MIT-licensed code may be included if the
  MIT notice travels with it.
* Required practice for anything copied or closely transliterated (BRIEF section 3.0 item 4):
  1. File-level header in each affected file: origin (`https://github.com/scaleresearch/tt-rl`), commit `6fa5ead55458dbe0690091bb1aa96d327ca37ae5`,
     original path, a note of what changed, and "MIT License, Copyright (c) 2026 Antoni Bertel".
  2. The full MIT text, once, in a third-party notices file shipped with the package (for example under the existing `licenses/` directory or a
     `THIRD_PARTY_NOTICES`), listing each tt-rl file used.
  3. Record the decision in `DECISIONS.md` (source, commit, files, reason), so a later re-sync with tt-rl is traceable.
* Re-implementing an idea from the algorithm or from a documented ttnn behaviour (not copying text) does not need the notice, but credit the reference
  in the design docs where the design was derived from it.
* Do not copy `assets/` (GIFs) or `uv.lock`; they are not needed and not covered by any statement I read beyond the repo-wide MIT file.
* Never execute tt-rl code, including `tools/` and the console script (BRIEF rule 2-4).

### 13.2 Open questions / unclear points

1. Behaviour on ttnn v0.81.1 for every API in section 10 (tt-rl targets 0.80). In particular `ttnn.all_reduce`, `FabricConfig`, mesh open and trace APIs.
2. Trace region sizing and overflow behaviour; whether act + update + GAE traces fit in one region for our nets (tt-rl: 32 MiB for its largest net,
   no logic).
3. How the "allocation in the eager gap may alias trace memory" hazard (`adam.py:73-80`) is reconciled with `read_metrics` allocating after replay.
4. Whether a captured backward can write fp32 gradients in place with a device-resident LR/step-counter input (not demonstrated by tt-rl); whether
   `ttnn.pow`/`exp(t*log b)` or per-replay scalar uploads are the better way to get bias correction (not in tt-rl).
5. Whether `ttnn.gather` / embedding supports an fp32 row gather suitable for minibatch selection (never exercised).
6. CCL: no working collective exists in tt-rl for 0.80/0.81; the "1 and 2 both come back 3" bug (older ttnn) is unresolved for 0.80.
7. `math_approx_mode` and `packer_l1_acc` defaults for matmul and unary ops (tt-rl never sets them; BRIEF 3.4 wants approximate modes off on the loss path).
8. Whether the HiFi4 ~1.3e-3 floor and the measured `*_bw` errors hold on our tray's silicon and on v0.81.1.
9. Replica initialization and RNG streams in tt-rl multi-chip runs (section 7, item 7): `broadcast` is defined and tested but not called in the tree I read.
10. Whether tt-rl's recurrent/carry path (`engine/carry.py`, `algos/ppo/recurrent.py`) is traced in production (I only read their docstrings); relevant only to
    post-v1 RNNModel work.
11. The `local:` comparisons in section 3.7 are from the checked-out repo (tag v5.5.2 per `git describe`, provisional in BRIEF section 0); `UPSTREAM_SEMANTICS.md`
    is the authority and should be re-read if `RSL_RL_VERSION` changes.
