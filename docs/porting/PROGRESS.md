# Progress log

Newest entry first. Each entry: date, phase, done / next / blocked / key numbers, related commits.
Read order for a new session: `BRIEF.md` -> `CLAUDE.md` -> this file.

## 2026-11-10 — Phase 0 (in progress)
**Done**
- GitHub repo `changh95/tt_rsl_rl` created (public, non-fork, empty) and seeded from a full-history clone of upstream;
  `main` = tag `v5.5.2` (pristine), upstream tags pushed, `upstream` push URL disabled, work branch `tt/phase-0`.
- Section-0 values resolved with the owner: tray 3 = `/dev/tenstorrent/16..23` (8 nodes present), TT_METAL =
  `~/experiments/tt-metal` v0.81.1 (read-only), PYTHON_ENV = `<repo>/.venv` (Python 3.10.12, torch 2.11.0+cpu, numpy 1.26.4),
  `RSL_RL_VERSION = v5.5.2` (provisional).
- `BRIEF.md`, `CLAUDE.md` (rules copied verbatim), README with phase table.
- Safety net: `scripts/tt_run.sh`, `scripts/kill_my_jobs.sh`, `.claude/hooks/tray_guard.{sh,py}` (command-position
  matching, fails closed). Host tests: `tests/tt/test_tray_guard.py` (218 cases incl. false-positive protection) and
  `tests/tt/test_repo_invariants.py` (`rsl_rl/` identical to the tag; only `device.py` may open devices) — 221 passed.
- Research docs: `UPSTREAM_MAP.md`, `UPSTREAM_SEMANTICS.md` (52 semantic rows), `TTRL_NOTES.md` (tt-rl commit `6fa5ead`),
  `DECISIONS.md` (ADR-0002/0003/0004 proposed, host allow-list, clarifications C-1/C-2).
- Commit `9c92dc4` pushed to `origin/tt/phase-0`.

**Key findings so far**
- BRIEF 3.7 says one grad norm over actor+critic; upstream clips actor and critic **separately** (`ppo.py:309-310`) — see
  DECISIONS C-1 (owner confirmation requested).
- Pure-config plugin injection is feasible: `cfg["algorithm"]["class_name"]="rsl_rl_tt.<mod>:TTPPO"`, runner device `cpu`;
  no runner subclass and no upstream patch needed for the single-process mesh design.
- Upstream multi-GPU semantics are mixed (grads mean, normalizer stats global, advantage normalization rank-local, KL mean,
  LR from rank 0) — ADR-0004 reproduces them.
- tt-rl does not trace PPO, keeps Adam scalars on the host, and uses periodic parameter averaging across chips
  (DO-NOT-USE); its per-op VJP tape is a reusable gradient oracle.

**Next (remaining Phase 0)**
1. `TTNN_OP_SURVEY.md` (source-level op/CCL/trace/mesh survey; in progress) -> determines the `TT_VISIBLE_DEVICES` ID
   space and how `import ttnn` works from the source tree without writing into it.
2. `scripts/tray_env.sh` (single source of device env vars) and `rsl_rl_tt/device.py` (single guarded entry point:
   env check, `/proc/self/fd` node check, close in `finally`+`atexit`) + their tests.
3. `.claude/settings.json` (hook registration + `permissions.deny`) — written LAST, after the guard tests pass, because
   the deny rules lock `.claude/**`. Then ask the owner for `/hooks` review and a session restart.
4. After restart: smoke tests via `tt_run.sh` (1x1 matmul vs torch; tray-wide mesh + all-reduce with exactly 8 nodes),
   noise-floor measurements -> `TOLERANCES.md`, ADR-0001, then STOP and report for approval of Phase 1.

**Blocked / waiting on the owner**
- Confirm C-1 (two grad norms, as upstream) and C-2 (provisional `v5.5.2` vs the Isaac Lab pin).
