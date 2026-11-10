# Decisions (ADRs) and the host-computation allow-list

Status values: **Accepted** (owner-confirmed or fixed by the brief), **Proposed** (Phase 0 recommendation, waiting for owner
approval at the Phase 0 stop), **Pending** (needs a result that is not available yet).
Evidence sources: `UPSTREAM_MAP.md`, `UPSTREAM_SEMANTICS.md`, `TTRL_NOTES.md`, `TTNN_OP_SURVEY.md`, `ENV.md`.

## Clarifications of the brief (owner approval requested)

### C-1 — Gradient clipping is per model in upstream, not one joint norm  — *Proposed*
BRIEF 3.7 lists "max_grad_norm over actor+critic as a whole". Upstream v5.5.2 clips **separately**:
`clip_grad_norm_(actor.parameters(), max_grad_norm)` and `clip_grad_norm_(critic.parameters(), max_grad_norm)`
(`rsl_rl/algorithms/ppo.py:309-310`), after the multi-GPU gradient mean (`:305-306`). The actor group includes the std
parameter. Because the project goal is numerical equivalence with upstream (BRIEF §1.2), the port implements **two
independent norms (actor, critic)** — a joint norm would change the update whenever one norm exceeds the threshold.
Verified directly in the source by the maintainer of this file (2026-11-10).
*Action requested:* confirm "follow upstream (two norms)".

### C-2 — `RSL_RL_VERSION = v5.5.2` is provisional  — *Pending*
Chosen as the latest upstream tag. It must be compared with the version the user's Isaac Lab pins (`pip show rsl-rl-lib`
on the GPU host) before Phase 1 golden baselines are produced. Switching later only replaces `rsl_rl/` (BRIEF 3.1), but
golden curves and `UPSTREAM_SEMANTICS.md` rows must then be re-checked.

## ADR-0001 — Training backend (A: hand-written TTNN fwd/bwd, B: tt-train `ttml`)  — *Pending*
To be written from `TTNN_OP_SURVEY.md` (op coverage, fp32, trace compatibility, `ttml` build state) and `TTRL_NOTES.md`
(tt-rl's tape of per-op VJPs is a reusable oracle; tt-rl does not trace PPO and keeps Adam scalars on the host).
Default per BRIEF 3.2 stays **A**.

## ADR-0002 — Plugin injection point and runner subclass  — *Proposed*
**Context.** Upstream resolves the algorithm from `cfg["algorithm"]["class_name"]` via `resolve_callable`
(`utils.py:167-176`) and calls the static `construct_algorithm(obs, env, cfg, device)` (`on_policy_runner.py:39-40`).
The runner touches only `train_mode/eval_mode/broadcast_parameters/act/process_env_step/compute_returns/update/
learning_rate/get_policy/save/load` on the algorithm (`UPSTREAM_MAP.md` §2.5). It never touches `alg.actor/critic/storage/
optimizer` (`on_policy_runner.py:66`).
**Decision.**
1. `cfg["algorithm"]["class_name"] = "rsl_rl_tt.<module>:TTPPO"` (a qualified name; a bare name does not resolve for a
   plugin package). Runner `device="cpu"` (host staging). **No `TTOnPolicyRunner` subclass and no upstream patch** for the
   single-process design (one `MeshDevice` for 1..8 chips, data parallel inside the algorithm).
2. `TTPPO` takes **Route B** (inherit `PPO.construct_algorithm`) first: obs-group validation and torch default weight init
   are inherited, so initial weights are identical to upstream for a given seed. The TT `__init__` uploads the torch
   actor/critic weights and discards the upstream torch storage/optimizer. It must accept `rnd_cfg`, `symmetry_cfg`,
   `multi_gpu_cfg`, `grad_reduce_bucket_mb`, implement `compile(mode)` (`ppo.py:463`) and expose
   `state_dict()/load_state_dict()/param_groups[0]["lr"]/learning_rate` for checkpoints (`ppo.py:379,405-406`).
   Route A (own `construct_algorithm`) is the fallback if Route B's side effects get in the way.
3. TT settings (mesh shape, dtype policy, trace on/off, parity vs performance RNG mode) are consumed by `TTPPO` as extra
   algorithm kwargs, because unknown keys otherwise raise `TypeError` (`UPSTREAM_MAP.md` §3.4). Whether **Isaac Lab's** cfg
   classes can carry these extra keys is unverified (open question Q2); if not, the fallback is a side YAML or environment
   variables read by `TTPPO`, never an upstream patch.
4. `get_policy()` returns a cached upstream `MLPModel` mirror (weights downloaded only when stale), because the runner calls
   `get_policy().output_std` **every iteration** (`on_policy_runner.py:124`).
5. A runner subclass is only needed for multi-process mode (hard-wired NCCL/CUDA, `on_policy_runner.py:215-257`), which the
   mesh design does not use. Device close is handled by `rsl_rl_tt/device.py` (`finally` + `atexit`) plus an explicit
   `alg.close()`.
**Consequences.** Upstream stays byte-identical to the tag (tested). Open items: Q1 (which `runner.alg.*` attributes Isaac
Lab accesses), Q2 (extra cfg keys), Q3 (dtypes of `dones`/`time_outs`).

## ADR-0003 — Environment location and bridge  — *Pending (ISAAC_HOST unknown)*
The host has no NVIDIA GPU tooling (`ENV.md`), so Isaac Sim is assumed to run on a separate GPU host. Default plan
(BRIEF 3.8 / Phase 7): `RemoteVecEnv` (an rsl_rl `VecEnv`) on the TT host talking to a small server wrapping the Isaac Lab
env on the GPU host, ZeroMQ + raw float32 buffers; same-host would connect directly. Alternatives (gRPC, shared memory,
Arrow Flight) and the measured round-trip/bandwidth are compared in Phase 7. Nothing is built before Phase 7.

## ADR-0004 — Multi-chip statistics synchronisation  — *Proposed*
**Context.** What upstream multi-GPU actually synchronises (`UPSTREAM_MAP.md` §9):

| What | Upstream | Source |
|------|----------|--------|
| initial params + buffers | broadcast once at the start of `learn()` | `ppo.py:467-474` |
| gradients | bucketed all-reduce(SUM)/world_size **before** clipping | `ppo.py:305-306,476-486` |
| grad clipping | per model, on already-averaged grads | `ppo.py:309-310` |
| obs-normalization stats | **global** (count/mean/var all-reduced; keyed on `torch.distributed.is_initialized()`) | `normalization.py:62-71` |
| advantage normalization | **rank-local** (no collective) | `ppo.py:190-191,214-216` |
| KL for the adaptive LR | all-reduce mean | `ppo.py:246-249` |
| learning rate | rank 0 decides, broadcast through a float32 tensor, every rank adopts it | `ppo.py:251-262` |
| noise / randperm | per rank, independent | `rollout_storage.py:232` |

**Decision.** The mesh data-parallel design reproduces exactly this, with chip = rank:
- parameters + Adam state replicated; gradients all-reduced (mean) before the two clips; every chip takes the same step;
- normalizer statistics global (all-reduce of count/sum/centered second moment), as in upstream;
- **advantage normalization chip-local** (per chip's rollout, or per chip's mini-batch, per the option) — *not* global —
  so that "N chips" equals "upstream with world_size=N";
- KL averaged over chips, one global LR kept as a **device tensor** (so trace replay does not freeze it); the float32
  round-trip of the LR through a tensor is reproduced;
- initial parameter/buffer replication is part of construction (upstream does it at the start of `learn()`).
**Test design (Phase 6).** "N chips == 1 chip" is only meaningful for the same global sample set *and* the chip-local
advantage normalization: the reference is upstream torch code run over N explicit shards with explicit averaging (or
upstream gloo with `world_size=N`), not a single big batch. A periodic consistency check of parameters across chips is added.
**Consequences.** Differs deliberately from tt-rl's periodic parameter averaging (`TTRL_NOTES.md`), which is on the
DO-NOT-USE list.

## Host-computation allow-list (BRIEF 2-8)
Silent torch/host fallbacks on the TT path are forbidden. Only the items below may run on the host. Each is counted by a
runtime counter that is logged; any other host fallback must fail the tests.

| ID | Allowed host operation | Scope | Why |
|----|------------------------|-------|-----|
| H-1 | upload obs / reward / done / time_out per env step (pre-allocated staging buffer) | all phases | env lives on the host (BRIEF §1.1) |
| H-2 | download actions per env step (one sync point) | all phases | the env consumes them |
| H-3 | obs-group concatenation | all phases | data shaping at the boundary (BRIEF 3.3) |
| H-4 | parity mode: action noise from the host torch generator; `torch.randperm` for mini-batches, only indices uploaded | parity mode | same RNG streams as upstream (BRIEF 3.5) |
| H-5 | per-iteration scalars for logging (losses, KL, LR, std) | all phases | log lines |
| H-6 | weight download for `get_policy()` mirror, `save()`, export | on demand | upstream checkpoint/JIT/ONNX formats |
| H-7 | **temporary (Phase 2 only):** rollout storage on the host | Phase 2 | moves to the device in Phase 3 (BRIEF Phase 2) |
