# Decisions (ADRs) and the host-computation allow-list

Status values: **Accepted** (owner-confirmed or fixed by the brief), **Proposed** (Phase 0 recommendation, waiting for owner
approval at the Phase 0 stop), **Pending** (needs a result that is not available yet).
Evidence sources: `UPSTREAM_MAP.md`, `UPSTREAM_SEMANTICS.md`, `TTRL_NOTES.md`, `TTNN_OP_SURVEY.md`, `ENV.md`.

## Clarifications of the brief (owner approval requested)

### C-1 — Gradient clipping is per model in upstream, not one joint norm  — *Accepted (owner, 2026-11-10: "그래")*
BRIEF 3.7 lists "max_grad_norm over actor+critic as a whole". Upstream v5.5.2 clips **separately**:
`clip_grad_norm_(actor.parameters(), max_grad_norm)` and `clip_grad_norm_(critic.parameters(), max_grad_norm)`
(`rsl_rl/algorithms/ppo.py:309-310`), after the multi-GPU gradient mean (`:305-306`). The actor group includes the std
parameter. Because the project goal is numerical equivalence with upstream (BRIEF §1.2), the port implements **two
independent norms (actor, critic)** — a joint norm would change the update whenever one norm exceeds the threshold.
Verified directly in the source by the maintainer of this file (2026-11-10).
*Decision:* follow upstream (two norms: actor incl. std, critic).

### C-2 — `RSL_RL_VERSION = v5.5.2`  — *Accepted for now (owner, 2026-11-10: "최신 버전으로 해줘 일단")*
`v5.5.2` is the newest upstream **release tag** (checked with `git fetch upstream --tags` on 2026-11-10). Upstream `main` is 4 untagged
commits ahead (`b61a24b` SAC + FlashSAC, `feb2267` PPO configuration checks #260, `a2176ae` algorithm statistics for the logger #265,
`8945bdc` removal of unused `init_weights` #262; all committed 2026-11-10). They are not a release and `main` is still moving, so the
pin stays on the tag; the PPO-related ones (#260/#265) are cheap to adopt later. Bump plan: when upstream tags a newer release, or at a
phase boundary if the owner asks for `main`: replace `rsl_rl/` with that ref (BRIEF 3.1), update `version_pin.py`, re-check the
`UPSTREAM_SEMANTICS.md` rows and re-run the parity/golden baselines. Still to compare with the version Isaac Lab pins
(`pip show rsl-rl-lib` on the GPU host) before Phase 7.

## ADR-0001 — Training backend (A: hand-written TTNN fwd/bwd, B: tt-train `ttml`)  — *Proposed: A*
**Evidence (source-level; nothing is measured on hardware yet — `TTNN_OP_SURVEY.md`).**
- **B (`ttml`) is not viable for v1.** Not built here (`BUILD_TT_TRAIN=OFF`, no `_ttml`); no ELU/tanh/softplus/sqrt/log autograd
  ops, which an ELU-MLP + Gaussian policy needs; bf16-centric (fp32 only for optimizer master weights). Its AdamW(+fp32 master),
  `clip_grad_norm` and DDP `synchronize_gradients` (all-reduce, then multiply by 1/N) are useful design references only.
- **A needs only ops that exist in v0.81.1:** fp32 matmul/linear (HiFi4 + `fp32_dest_acc_en`, approx off), `elu` + `elu_bw`, `exp/log/
  sqrt/rsqrt/square/reciprocal`, `where/clamp/minimum/maximum/lt/gt`, fp32 reductions (sum/mean/max, whole tensor), `ttnn.copy` /
  `fill` in place, row gather via `ttnn.gather`/`tosa_gather` (fp32 table, UINT32 index; `ttnn.embedding` is bf16-only so it is
  NOT used), `ttnn.all_reduce` (fp32 TILE, Sum) then ×1/N, and mesh trace capture/replay.
- **Global grad norm** has no fp32 fused op (`moreh_clip_grad_norm` is bf16-only): compose it from `square` + `sum` + `sqrt` +
  `clamp` + `multiply` in fp32 — one norm for the actor group (incl. the std parameter) and one for the critic (DECISIONS C-1).
- **Adam** runs on device in fp32 (master weights + moments), per parameter, in place; `lr`, `1-β^t` bias corrections and the PPO
  adaptive LR are **device tensors** updated by `copy_host_to_device_tensor` outside the trace, because a trace freezes Python
  scalars (survey item 8; tt-rl keeps these scalars on the host and does not trace PPO at all, `TTRL_NOTES.md`).
- **Noise** is sampled outside the trace (host generator in parity mode, BRIEF 3.5), because replayed traces repeat `ttnn.rand*`
  noise.
- **Backward:** hand-written per-op VJPs (tt-rl's tape is a reusable oracle for matmul transpose flags, broadcast reductions);
  correctness is judged by gradient parity against upstream torch autograd (BRIEF 3.2), per parameter.
**Main numerical risk.** fp32 matmul multiplies TF32-class operands (accumulation is fp32), so expect ~1e-3 relative error vs torch.
This is measured first (noise-floor probe, survey Q10) and sets `TOLERANCES.md`. If parity needs more, the contingency is a split-fp32
matmul (hi/lo operand decomposition, 2-3 matmuls) — decided only with the measurement in hand (BRIEF 2-9: tolerances are never
loosened just because a test fails).
**Decision.** Option A. ADR stays *Proposed* until the Phase 0 noise-floor measurements exist.

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

## ADR-0005 — Isolation of the other trays' device nodes  — *Accepted: option II, delegated to the assistant by the owner on 2026-11-10 ("다른 트레이의 작업을 방해하지 않는 한으로 너가 해줘")*
**Finding (`TTNN_OP_SURVEY.md` 11.3, source-level; not yet observed on hardware).** `TT_VISIBLE_DEVICES` (BDF tokens) makes UMD
*initialise and use* only the listed chips. But UMD's enumeration `open()`s every `/dev/tenstorrent/N` on the host (O_RDWR, one
`GET_DEVICE_INFO` ioctl, then `close()`), 1 to 3 times per enumeration call and from several callers, before it filters. Two
consequences: (1) the `/proc/self/fd` check required by BRIEF 2-2(b) passes even then, because those probe fds are already closed —
it proves what is *held*, not that nothing else was *ever opened*; (2) a source comment says tt-kmd messages the chip's firmware
on every `open()` (kernel side, unverifiable here). So "no other tray's chip is ever opened" cannot be guaranteed by software
inside the process; this was not visible when the brief was written.
**Facts established on this host (read-only):** nodes 16..23 map to BDFs `0000:c1:00.0 .. 0000:c8:00.0` (consecutive buses, one tray);
all `/dev/tenstorrent/*` are `crw-rw-rw- root:root`, so permissions protect nothing; unprivileged user+mount namespaces work
(`unshare -Urm`, Ubuntu 22.04, `max_user_namespaces=2318138`, no `sudo`); `bwrap` is not installed.
**Dry run with `/dev/null` only (no Tenstorrent node touched):** a bind-mounted char device in the namespace opens O_RDONLY but its
O_WRONLY/O_RDWR open fails with EACCES; opening `/dev/null` directly inside the namespace works for O_RDWR.
**Options**
- **I. Accept** the transient probes: treat "never open other chips" as "never initialise, use, reset or touch them"; rely on
  `TT_VISIBLE_DEVICES`, the sysfs BDF check and the fd check. Simplest; leaves the possible tt-kmd power-state message on all other
  nodes at every run, on a host where other people's jobs are running.
- **II. Mask in a private mount namespace (recommended, needs your approval):** `tt_run.sh` runs the job under `unshare -Urm`, and
  before exec'ing it bind-mounts `/dev/null` over every `/dev/tenstorrent/*` entry that is not 16..23 (via the `mount(2)` syscall, no
  `sudo`, invisible to every other process on the host). Per the survey, a node that cannot be opened read-write is silently dropped
  from UMD's map, so enumeration then never reaches another tray's driver. The 8 allowed nodes are left at their original paths.
  *Unverified:* that the real allowed nodes still open O_RDWR from inside the namespace, and that UMD/hugepage/sysfs access works
  there. Verification plan: the first device job opens only `--chips 16` and checks the fd list; if anything fails, fall back to I or III.
- **III. OS-level isolation by an administrator** (ACL/udev permissions or a container exposing only 8 nodes). Strongest, needs root.
**Decision (2026-11-10): option II.** The owner delegated the choice with one condition: never disturb the jobs other agents run on
the other trays. Option I would send an open() (and, per the tt-kmd comment, a firmware message) to every other tray's chip on every run;
II touches nothing but this job's own namespace. Implemented as an allow-list, which is stronger than the proposal above (UMD lists
`/dev/tenstorrent` by file name only — `PCIDevice::get_all_device_ids`, `pci_device.cpp:1168` — so placeholder files work):
`scripts/tt_run.sh` -> `unshare --user --map-root-user --mount` -> `scripts/ns_mask_dev.sh` mounts a private tmpfs, bind-mounts only the
SELECTED chips (`--chips`) onto placeholders in it and moves it over `/dev/tenstorrent`. Other trays' nodes do not exist for the job, so
nothing can open them; the nodes of tray 3 that this run did not select are hidden too. The job is uid 0 of its own user namespace
(maps to our uid, no host privilege); PID/IPC/network/`/dev/shm`/`/tmp` are shared with the host, so tray-wide coordination (our flock,
the kernel-side kmd locks, UMD's `/dev/shm` objects) is unchanged and other people's processes see nothing. `tt_run.sh` has no way to
skip the mask and refuses to start the job when it cannot be built. `device.py` independently requires that `/dev/tenstorrent` shows
exactly the selected chips before it imports ttnn (check (e)), and still reads `/proc/self/fd` after opening (2-2(b)/(c) unchanged).
**Verified on this host (no device open):** 18 host tests with fake nodes (view, host untouched, char-device bind opens O_RDWR, refusal
paths, exec keeps the PID); on the real tray: a job started with `--chips 16` sees exactly `/dev/tenstorrent/16` (a character device),
holds no tenstorrent fd, sysfs BDF and view checks pass.
**Still to verify on hardware (first device job, chip 16 only):** the real node opens O_RDWR from inside the namespace and UMD/KMD work
there (hugepages/sysfs/ioctls); if not, stop and report (fall back to I/III only with the owner).

## ADR-0006 — How `import ttnn` is provided  — *Accepted: option a, PyPI wheel (owner, 2026-11-10: "그래")*
**Finding.** `/home/ttuser/experiments/tt-metal` was configured with `WITH_PYTHON_BINDINGS=OFF`: there is no `_ttnn.so`, so
`import ttnn` cannot work from it (`ttml` is not built either). Someone else changed that tree while the survey ran (egg-info
removed, CMake reconfigured with tests ON at 17:19), so it is also not a stable base. Pure-Python `ttnn` files are not enough.
**Options**
- **a. PyPI wheel `ttnn==0.81.1` into the dedicated venv (recommended).** Same version as the reference tree. Downloaded to a scratch
  directory and **inspected only** (not installed, not imported): `ttnn-0.81.1-cp310-cp310-manylinux_2_34_x86_64.whl`, 103 MB, bundles
  `_ttnn.cpython-310-*.so`, `libtt_metal.so`, `libtt-umd.so`, `_ttnncpp.so`, an OpenMPI, firmware/runtime sources; requirements are
  the same pure-Python set as the tree (`numpy<2, loguru, networkx, graphviz, pyyaml, click, pandas, seaborn, ml_dtypes`). Open item:
  the riscv `sfpi` toolchain is fetched by `ttnn/download_sfpi.py` (`sfpi-version` file) — whether first use needs network access is
  unverified. Also unverified: compatibility with the installed kernel driver (2.10.0) — the same question applies to any route.
- **b. Separate bindings build** of the same sources in a private clone/build dir (new build, long, heavy CPU load on a shared host;
  never in the shared tree).
- **c. Reuse another tree's `_ttnn.so`** — rejected: ABI must match the `libtt_metal.so` it links; other people's trees.
`scripts/ttnn_env.sh` currently only sets hygiene variables (private `TT_METAL_CACHE`/`TT_METAL_LOGS_PATH`, no bytecode in trees).

**Done (2026-11-10), all inside `.venv`:** `pip install ttnn==0.81.1` (dry-run first: only pure-Python deps were new — click, graphviz, loguru,
matplotlib, pandas, pyyaml, seaborn and their dependencies; torch 2.11.0+cpu and numpy 1.26.4 untouched). **sfpi:** the wheel does not ship the
riscv toolchain and the JIT looks in `<ttnn>/runtime/sfpi`, then `/opt/tenstorrent/sfpi`. The host's `/opt/tenstorrent/sfpi` is 7.67.0 and
the wheel needs **7.89.0** (`ttnn/tt_metal/sfpi-version`), so the official release `sfpi_7.89.0_x86_64_debian.txz` was downloaded from
github.com/tenstorrent/sfpi, its sha256 (`de16ebb1...b610b954`, from the wheel's own version file) verified, and extracted to
`.venv/lib/python3.10/site-packages/ttnn/runtime/sfpi` (436 MB, venv only; the tt-metal tree and `/opt` are untouched).
`ttnn.__init__` sets its runtime root to the installed package directory automatically (no env var needed). Whether the wheel works with
KMD 2.10.0 and this hardware is verified by the first device job.

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
