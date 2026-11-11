# Decisions (ADRs) and the host-computation allow-list

Status values: **Accepted** (owner-confirmed or fixed by the brief), **Proposed** (recommendation, waiting for owner
approval), **Pending** (needs a result that is not available yet). At the end of Phase 0 (2026-11-10) the owner approved all open ADRs
(0001, 0002, 0003 as the working plan, 0004 with amendment A-1) and Phase 1.
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

## ADR-0001 — Training backend (A: hand-written TTNN fwd/bwd, B: tt-train `ttml`)  — *Accepted: A (owner, 2026-11-10: "전부 승인", after the Phase 0 measurements below)*
**Evidence (source-level survey: `TTNN_OP_SURVEY.md`; hardware measurements: the "MEASURED" block below).**
- **B (`ttml`) is not viable for v1.** Not built here (`BUILD_TT_TRAIN=OFF`, no `_ttml`); no ELU/tanh/softplus/sqrt/log autograd
  ops, which an ELU-MLP + Gaussian policy needs; bf16-centric (fp32 only for optimizer master weights). Its AdamW(+fp32 master),
  `clip_grad_norm` and DDP `synchronize_gradients` (all-reduce, then multiply by 1/N) are useful design references only.
- **A needs only ops that exist in v0.81.1 (all present; see the measurements below):** fp32 matmul/linear (HiFi4 + `fp32_dest_acc_en`, approx off), `elu` + `elu_bw`, `exp/log/
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
**Main numerical risk — now MEASURED on chip 16 (2026-11-10; `TOLERANCES.md`, repros in `tests/tt/test_noise_floor*.py`).**
- fp32 matmul with HiFi4 + `fp32_dest_acc_en` (the default config gives the same numbers) has rel_fro **1.17e-3** vs a float64 reference
  (torch fp32: 2.4e-7), independent of shape (K = 48..1024, M up to 16384) and input distribution (normal / heavy-tailed). HiFi3 1.28e-3,
  HiFi2 7.2e-3, LoFi 3.1e-2, no fp32 accumulation 8e-3.
- The contingency "split-fp32 matmul" **does not recover fp32**: multi-pass operand splits (3-6 passes, several bit layouts, K-chunked
  with exact `ttnn.add` of the partials) bottom out at **3.1e-4**, because even with both operands exactly representable (4 mantissa
  bits) the matmul result carries a constant relative error of 2.6e-4, independent of K, of `packer_l1_acc` and of chunking
  (TF32-class precision on the output path). So: no cheap way to fp32-exact matmul; ~4x better for 6x the matmuls is not worth it as
  the default. The split stays available as a debugging aid only.
- Everything else that the PPO step needs is fp32-accurate: `exp/log/sqrt/rsqrt/reciprocal/tanh/elu/elu_bw` <= 2 ULP (rel 1.4e-7),
  `sum`/`mean` over rows ~1e-7, global norm via `sqrt(sum(square))` 2-4e-8, `max` exact, fp32 host<->device round trip bit exact.
  **Do not use:** `ttnn.var` (rel 2.4e-4; compose the two-pass variance instead: 3.7e-8), the one-pass `E[x^2]-E[x]^2` (9e-4 with a
  non-zero mean), `fast_and_approximate_mode=True` (exp 2e-2, tanh 2e-3), `ttnn.softplus` (rel 5e-4 at large negative x; not needed in v1).
- Consequence for tests: component parity against upstream torch is bounded by the matmul noise floor (~1.2e-3 per matmul layer), not by
  1e-6. Correctness of gradient formulas is therefore checked against an fp64 torch reference with the measured floors, and the
  end-to-end claim is statistical (multi-seed learning curves, BRIEF Phase 4), exactly as the brief plans. The first mini-batch of an
  update will see a policy ratio that differs from 1 by ~1e-3 (rollout and update forward use different matmul shapes), which PPO's
  clipping tolerates; this is a property to document, not to tune away.
**Decision.** Option A. Evidence now exists; recommended **Accepted** (owner approval requested at the Phase 0 stop).

**Implementation note (Phase 3, 2026-11-11).** The training step is hand-written as designed: `TTMLP.backward` (per-layer VJP with `matmul(transpose_a/b)`, `elu_bw`, bias row sums),
`PPOLosses` (clipped surrogate/value losses with torch's tie and clamp rules, entropy through the std only, masked padded action dims), `clip_grad_norm` per model (C-1),
`TTAdam` (functional, fp32 moments, lr and bias corrections as device tensors), `TTAdaptiveLR` (device-evaluated KL rule), `TTRolloutStorage` (per-step device tensors, GAE loop on the
device, ONE packed table gathered by the host-drawn permutation, contiguous mini-batch slices). `update()` downloads only the three logged losses, the learning rate and the std (H-5)
and uploads the permutation (H-4) and two Adam scalars per step (H-8); the H-7 allow-list entry (host storage) is retired. Per iteration cost is dominated by per-op dispatch (about 20
optimizer steps x ~170 ops) and the fp32 row gather (T-12): measured 590 ms update (1024 envs x 24 steps, cartpole) vs 145 ms for upstream on the CPU, the Phase 5 baseline.
Numerical findings: (1) a parameter-by-parameter comparison against upstream after Adam is meaningless (near-zero gradient components flip sign and take full-size steps), parity is judged in
function space against a noise baseline (TOLERANCES.md); (2) the clipped objectives are discontinuous at the clip boundaries, so a whole update deviates ~2 % (20 steps) from upstream unless the
objective is smooth (then 0.1 %); (3) with identical state the first mini-batch's KL is exactly 0 on the device, so the adaptive LR rule is compared separately.

**Implementation note (Phase 4, 2026-11-11).** Checkpoints are the upstream ones: `TTPPO.save()` syncs the device into the torch mirrors (weights, normalizer statistics, and the Adam moments written into
`optimizer.state` in torch's layout: `step`, `exp_avg`, `exp_avg_sq` per parameter in the optimizer's order, std parameter first) and returns upstream's own dict; `TTPPO.load()` first brings the torch optimizer
up to date, runs upstream's `PPO.load` and rebuilds the device copies (weights, normalizers, Adam moments and step counter, learning rate as a device tensor). `get_policy()` syncs only the models (not Adam), so the
runner's per-iteration logging stays cheap. A TT checkpoint therefore loads into an unmodified upstream runner (bit-identical inference), an upstream checkpoint resumes on TT (device state compared tensor by
tensor), and the optimizer state has exactly the structure of an upstream one (tested). Upstream exports ONNX with a fixed batch of 1; the exported TT-trained policy matches the torch mirror (1e-5) and the device forward.

## ADR-0002 — Plugin injection point and runner subclass  — *Accepted (owner, 2026-11-10)*
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

**Implementation note (Phase 2, 2026-11-11).** `rsl_rl_tt.ppo:TTPPO` subclasses upstream `PPO` and inherits `construct_algorithm` unchanged:
the runner resolves `cfg["algorithm"]["class_name"] = "rsl_rl_tt.ppo:TTPPO"`, upstream builds the torch actor/critic (bit-identical
initialisation for the same seed), and `TTPPO.__init__` receives the TT-only options through the `tt_cfg` keyword (`mesh_shape`,
optional `fabric_config`/`trace_region_size`/`l1_small_size`), which upstream never sees. No runner subclass and no upstream patch are needed.
The upstream models stay as **mirrors** (source of the initial weights, sink of `sync_to_torch()` for `get_policy()`, `save()`, export; a
`_device_ahead` flag decides when a device -> mirror sync is needed). `get_policy().output_std` is served from a cached host std copy
(H-5), because the runner reads it every iteration. Parity-mode noise: `torch.randn(rows, A)` from the global generator is exactly what
upstream's `Normal.sample()` consumes (tested bit for bit), so TT and upstream draw the same stream. Everything the TT path uploads or
downloads goes through `TTContext` (counted per allow-list id; unknown ids raise; a host test forbids `from_torch`/`to_torch` elsewhere and
forbids any torch import in the device-side model code).

## ADR-0003 — Environment location and bridge  — *Accepted as the working plan (owner, 2026-11-10); the concrete bridge is decided in Phase 7 once ISAAC_HOST is known*
The host has no NVIDIA GPU tooling (`ENV.md`), so Isaac Sim is assumed to run on a separate GPU host. Default plan
(BRIEF 3.8 / Phase 7): `RemoteVecEnv` (an rsl_rl `VecEnv`) on the TT host talking to a small server wrapping the Isaac Lab
env on the GPU host, ZeroMQ + raw float32 buffers; same-host would connect directly. Alternatives (gRPC, shared memory,
Arrow Flight) and the measured round-trip/bandwidth are compared in Phase 7. Nothing is built before Phase 7.

## ADR-0004 — Multi-chip statistics synchronisation  — *Accepted incl. amendment A-1: gradient sum = gather + sum, not `all_reduce` (owner, 2026-11-10)*
**Amendment A-1 (hardware measurement, 2026-11-10, 2x4 mesh on FABRIC_1D, `tests/tt/test_noise_floor_mesh.py`).** The stock fp32
`ttnn.all_reduce` is NOT fp32-accurate: rel_fro **5.5e-4** vs a float64 sum of the 8 shards (torch fp32 sum: 5.6e-8) at 8K..1M values per
chip, in every variant (flat, or per mesh axis 1 then 0: 5.7e-4). It is bit-identical across chips and takes 0.23-0.40 ms. The alternative
`all_gather(dim=0, cluster_axis=1)` then `all_gather(dim=0, cluster_axis=0)` (a pure copy; `all_gather` on a 2x4 mesh needs a
`cluster_axis` under `FABRIC_1D`) followed by `ttnn.sum(dim=0)` gives **5.1e-8**, bit-identical across chips, 0.3-2.9 ms for 8K..1M values
per chip; a fixed-order chain of `ttnn.add` gives 2.7e-8 (1.3-1.8 ms). Therefore every place this ADR (and ADR-0001) says "all-reduce then
x 1/N" means: **gather along both mesh axes, sum in fp32, multiply by 1/N** (gradients, normalizer count/sum/second moment, KL). Gradient
buffers of an RL MLP are small (1e5-1e6 values), so the 8x gather volume is irrelevant. `ttnn.all_reduce` stays a fallback only if a
tolerance derived from 5.5e-4 is acceptable for that quantity.
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

## ADR-0007 — Post-v1 features: order, and the heteroscedastic Gaussian  — *Working decision (assistant, 2026-11-11, after the owner said to leave the GPU host aside and start on something else); the owner can veto*

* **Order** (BRIEF §1 "v1 이후"; `POST_V1.md`): by effort and by how much new machinery each needs on top of the traced fixed-shape design: 8.1 heteroscedastic Gaussian, 8.2 RND, 8.3 symmetry, 8.4 distillation, 8.5 RNNModel, 8.6 CNNModel, 8.7 SAC (not in the pinned
  upstream tag). Every item has the v1 acceptance set (component parity, update parity, traced == eager bit for bit, multi-chip equivalence in function space, end-to-end run against upstream, checkpoint compatibility); no tolerance of an existing row is loosened.
* **8.1 design** (details in `POST_V1.md`): the actor's output layer keeps upstream's `[mean | std]` shape and becomes a two-head `TTLinear` (each head tile-aligned); the std head is trained through the ordinary MLP backward (`d_out = [d_mean | d_raw]`); the rollout storage
  stores the std of every sample (one more column block, only for this distribution); the KL uses it. No new host computation (allow-list unchanged).
* **A new tolerance row, `update_entropy_hetero` (rel_fro 1.5e-3), with its root cause** (`TOLERANCES.md`): with a state-dependent std the update's entropy is a function of the actor's matmul output, so it carries the matmul noise (measured <= 3.7e-4); with a state-independent std
  (v1) it depends on a parameter only and keeps its 2e-5 row. The formula itself is verified to 7e-6 on identical network outputs.

## Host-computation allow-list (BRIEF 2-8)
*H-7 (temporary host rollout storage, Phase 2) was retired on 2026-11-11: storage, GAE and mini-batches are on the device since Phase 3.*
Silent torch/host fallbacks on the TT path are forbidden. Only the items below may run on the host. Each is counted by a
runtime counter that is logged; any other host fallback must fail the tests.

| ID | Allowed host operation | Scope | Why |
|----|------------------------|-------|-----|
| H-1 | upload obs / reward / done / time_out per env step (pre-allocated staging buffer) | all phases | env lives on the host (BRIEF §1.1) |
| H-2 | download actions per env step (one sync point) | all phases | the env consumes them |
| H-3 | obs-group concatenation | all phases | data shaping at the boundary (BRIEF 3.3) |
| H-4 | parity mode: action noise from the host torch generator; `torch.randperm` for mini-batches, only indices uploaded | parity mode | same RNG streams as upstream (BRIEF 3.5) |
| H-5 | per-iteration scalars for logging (losses, KL, LR, std) | all phases | log lines |
| H-6 | weight / statistics transfers between the torch mirrors and the device: initial upload, download for `get_policy()`, `save()`, export, and one-time constants | on demand | upstream checkpoint/JIT/ONNX formats |
| H-8 | upload per-iteration scalars as device tensors (normalizer update rate, learning rate, Adam bias corrections, adaptive-LR value) | all phases | a trace freezes Python scalars, so every value that changes per iteration travels as a tiny device tensor (ADR-0001); added 2026-11-11 in Phase 2 |
