# Progress log

Newest entry first. Each entry: date, phase, done / next / blocked / key numbers, related commits.
Read order for a new session: `BRIEF.md` -> `CLAUDE.md` -> this file.

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
