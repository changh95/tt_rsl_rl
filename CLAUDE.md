# CLAUDE.md — rsl_rl → Tenstorrent Blackhole port (`changh95/tt_rsl_rl`)

Read order at the start of EVERY session: `docs/porting/BRIEF.md` → this file → `docs/porting/PROGRESS.md`. Then continue from "Next" in PROGRESS.md.

This repo is a clone of `leggedrobotics/rsl_rl` (BSD-3) pinned at tag **v5.5.2** (provisional, see BRIEF §0). Upstream code in `rsl_rl/` is NOT modified (oracle + plugin target). The TTNN implementation lives in `rsl_rl_tt/`. `AGENTS.md`/`CONTRIBUTING.md` are upstream's rules for upstream contributions; for `rsl_rl_tt/`, `tests/tt/`, `scripts/`, `docs/porting/` this file and BRIEF.md take precedence.

## Goal (summary of BRIEF §1)
Run the whole training side of Isaac Lab's rsl_rl (rollout policy inference + PPO update) with TTNN on Blackhole; the environment stays where it is (CPU test env or Isaac Lab on a GPU host).
v1 is done when: (1) all training state lives on device, (2) numerically equivalent to upstream (per-component parity + multi-seed curves), (3) rsl_rl public API/config format preserved, (4) checkpoints are bidirectionally compatible with upstream (`load()`, JIT/ONNX export, Isaac Lab play), (5) scales from 1 chip to the 8-chip tray as synchronous-gradient-all-reduce data parallel.
v1 scope: PPO, MLPModel, GaussianDistribution (scalar/log std), EmpiricalNormalization, RolloutStorage/GAE, OnPolicyRunner integration. Everything else needs owner approval.
Confirmed design (do not re-litigate; BRIEF §3.0): 1-chip bring-up then mesh; plugin structure; fixed shapes + trace as the default design; tt-rl (MIT) is a reference implementation (cite source/license when copying code, never run its CLI).

## Resolved values (BRIEF §0, 2026-11-10)
TRAY_ID=3, allowed nodes `/dev/tenstorrent/16`..`23` (8 nodes, BDFs `0000:c1:00.0`..`c8:00.0`) — nothing else, ever. `TT_VISIBLE_DEVICES` holds the BDFs (an integer there is a BDF-sorted index, NOT a /dev number). ttnn = PyPI wheel `ttnn==0.81.1` in `PYTHON_ENV` + sfpi 7.89.0 next to it (ADR-0006, accepted); the TT_METAL reference tree `/home/ttuser/experiments/tt-metal` (v0.81.1) is read-only and not used at runtime. PYTHON_ENV=`/home/ttuser/experiments/rsl_rl/.venv`, WORKDIR=`/home/ttuser/experiments/rsl_rl`, RSL_RL_VERSION=v5.5.2 (newest release tag; C-2). Decided by the owner 2026-11-10: ADR-0005 option II (every device job runs in a private user+mount namespace in which `/dev/tenstorrent` lists only the selected chips, so other trays' nodes do not exist for it; `scripts/tt_run.sh` -> `scripts/ns_mask_dev.sh`), ADR-0006, C-1 (grad clip per model), C-2. Proposed and waiting for approval: ADR-0001 (backend A, now with measurements), ADR-0002/0004 (incl. amendment A-1: cross-chip sums are all_gather + sum, not `ttnn.all_reduce`). Unknown: ISAAC_HOST. TRAY_MESH_SHAPE = 2x4 (verified, `FABRIC_1D` collectives work). Device env vars are set ONLY in `scripts/tray_env.sh`; software-side ttnn env in `scripts/ttnn_env.sh`.
Device work is allowed through `scripts/tt_run.sh` only. The hook must be live in the session (canary: a blocked harmless command), one device job at a time, one open mesh per process, no resets on any error (stop and report).

## ABSOLUTE RULES — tray isolation and shared-server hygiene (BRIEF §2, copied verbatim)

이 섹션은 Phase 0에서 `CLAUDE.md`에 그대로 복사한다. 내 승인 없이 수정하지 않는다. 이 규칙에 닿을 수 있는 상황이면 무조건 멈추고 물어본다.

1. 쓸 수 있는 칩은 `TRAY_ID` 트레이의 8개뿐이다. 다른 칩은 열기·리셋·플래시·조회 모두 금지다.
2. 디바이스를 여는 코드는 `rsl_rl_tt/device.py`의 단일 진입점 하나뿐이다. 이 함수는 (a) 디바이스 관련 환경변수가 `scripts/tray_env.sh` 값과 일치하는지 확인하고, (b) 연 직후 `/proc/self/fd`를 읽어 이 프로세스가 연 `/dev/tenstorrent/*` 노드가 `ALLOWED_DEV_NODES`의 부분집합인지 검증하고, (c) 어긋나면 즉시 닫고 예외로 중단한다. 다른 곳에서 `ttnn.open_device`/`ttnn.open_mesh_device`를 직접 호출하지 않는다. 이를 검사하는 테스트를 둔다.
3. 디바이스를 쓰는 실행은 전부 `scripts/tt_run.sh`(tray_env 로드 → flock → timeout)를 거친다. 칩은 프로세스 단위로 독점되므로 디바이스 잡은 한 번에 하나만 돌린다. 서브에이전트도 디바이스 잡을 병렬로 돌리지 않는다.
4. 실행 금지 (예외 없음, 필요하면 나에게 요청):
   - `tt-smi` 전부. `-r`(인자 없으면 호스트의 모든 칩 리셋), `-glx_reset*`(Galaxy 전체 리셋)은 물론, `-ls`/`-s`/GUI 같은 조회도 호스트의 모든 칩을 건드리므로 금지.
   - `tt-flash`, `tt-topology`, `ipmitool`, 펌웨어·드라이버·hugepages 변경, `sudo`, 재부팅.
   - 하드웨어 health check/diag 스크립트 (리셋을 포함할 수 있음).
   - tt-rl의 `ttrl doctor`/`ttrl train`처럼 빈 칩을 자동으로 고르거나 모든 칩을 여는 도구.
5. 같은 호스트에 다른 사람 잡이 돈다. 이름 기반 `pkill`/`killall` 금지. 내가 띄운 PID만 `.run/pids`에 기록해서 그것만 종료한다. `/dev/shm`, `/tmp`의 남의 파일, 공용 캐시를 지우지 않는다.
6. 행(hang)이나 디바이스 에러(타임아웃, busy, Ethernet/fabric 에러)가 나면 리셋을 시도하지 않는다. 로그를 모으고 멈춘 뒤 보고한다.
7. tt-metal 소스·빌드와 ttnn·torch·rsl_rl 버전은 내 승인 없이 바꾸지 않는다. 패키지 설치는 `PYTHON_ENV` 안에서만 하고 새 의존성은 기록한다. TTNN 버그나 미지원 op를 만나면 최소 재현 스크립트를 만들고 `docs/porting/TTNN_ISSUES.md`에 기록한 뒤 우회책을 문서화한다.
8. TT 경로의 조용한 호스트/torch 폴백 금지. 호스트 연산은 `DECISIONS.md`에 명시적으로 허용된 것만 쓰고, 런타임 카운터로 집계해 로그에 남긴다. 허용되지 않은 폴백이 일어나면 테스트가 실패해야 한다.
9. 테스트가 실패했다고 허용오차를 완화하지 않는다. 근본 원인을 분석하고 근거와 함께 `DECISIONS.md`에 기록한 경우에만 조정한다.
10. git: remote는 `origin` = `changh95/tt_rsl_rl`, `upstream` = leggedrobotics/rsl_rl 두 개뿐이다. push는 `origin`에만 하고 `upstream`에는 절대 push하지 않는다(PR도 금지). 작업은 기능 브랜치에서 하고, 테스트가 통과한 마일스톤 단위로 커밋해 `origin`에 push한다. `main` 병합은 Phase 완료 시점에만. force-push, 히스토리 재작성, 브랜치 삭제, 저장소 설정 변경(visibility, 권한 등) 금지. 커밋에 비밀정보·대용량 산출물(체크포인트, 로그 원본) 금지 — `.gitignore`로 막는다.

Safety net (auxiliary only — the rules above stay binding even where the machinery does not catch something): `.claude/hooks/tray_guard.sh` (Bash PreToolUse hook), `permissions.deny` on `.claude/**` and `scripts/tray_env.sh`, `rsl_rl_tt/device.py` guard, `scripts/tt_run.sh` wrapper, `upstream` push URL set to an invalid value. Do not edit `.claude/**` or `scripts/tray_env.sh` (also not via Bash); ask the owner.

## Working agreements (BRIEF §5)
- Language: code, comments, commit messages, docs in English; reports to the owner in Korean, format: 3-line summary → numbers → decisions needed.
- Phase 0 end, any ADR change, and anything touching the rules above: STOP and wait for approval.
- Progress: update `docs/porting/PROGRESS.md` (date, phase, done / next / blocked / key numbers, commit hash) and the README phase table at every unit of work; commit + push to `origin` on a feature branch. Phase done → merge to `main`, tag `phase-N-done`, record evidence.
- Never invent TTNN APIs: use docstrings/`help()`, tt-metal tests/models examples, and minimal repro scripts.
- Parallelize reading/research with subagents if useful, but run device jobs strictly one at a time.
- Git identity for commits in this clone is local-only: `changh95 <changh95@users.noreply.github.com>`. Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Commands
```bash
# host tests (no device; `-m "not tt"` is the default via pytest.ini)
source .venv/bin/activate && pytest tests/tt -q

# device jobs — ONLY through the wrapper (tray_env -> flock -> timeout, PID logged in .run/pids)
# --chips selects a subset of the tray (16 = 1x1 bring-up, 16,17 = 1x2, default = all 8); anything outside 16..23 is refused
scripts/tt_run.sh --chips 16 --timeout 600 -- python -m pytest -m tt tests/tt/<file>.py -q
scripts/tt_run.sh --timeout 600 -- python scripts/<script>.py
# device code opens the mesh ONLY via:  from rsl_rl_tt.device import open_tray_mesh   (the single guarded entry point)

# Phase 2/3 device tests share one mesh fixture, so run the files that use it in ONE process (about 30 s warm):
scripts/tt_run.sh --chips 16 --timeout 900 --tag p3 -- python -m pytest -m tt tests/tt/test_tt_modules.py tests/tt/test_tt_rollout.py tests/tt/test_tt_backward.py tests/tt/test_tt_update.py -q -p no:cacheprovider

# Phase 5: the traced path opens its OWN mesh with a trace region -> run this file on its own (traced == eager bit for bit)
scripts/tt_run.sh --chips 16 --timeout 600 --tag p5t -- python -m pytest -m tt tests/tt/test_tt_trace.py -q -p no:cacheprovider
# Phase 5: benchmarks / profiles / traced end-to-end training, fail-fast (~25 min); results in docs/porting/PERF.md
scripts/run_phase5_final.sh
# Phase 6: multi-chip. The mesh shape comes from TT_TEST_MESH and must match --chips (1x2: --chips 16,17; 2x4: all eight, i.e. no --chips)
scripts/tt_run.sh --chips 16,17 --timeout 900 --tag p6 -- env TT_TEST_MESH=1x2 python -m pytest -m tt tests/tt/test_tt_multichip.py -q -p no:cacheprovider
scripts/tt_run.sh --timeout 900 --tag p6 -- env TT_TEST_MESH=2x4 python -m pytest -m tt tests/tt/test_tt_multichip.py -q -p no:cacheprovider
scripts/run_phase6_final.sh      # all device suites on 1 chip, 1x2 and 2x4, fail-fast; scripts/run_bench_scaling.sh = chips x num_envs benchmark
# Phase 7: environment bridge (host tests; loopback; the device test needs one chip)
python -m pytest tests/tt/test_bridge.py -q          # 14 host tests (needs pyzmq in the venv)
scripts/tt_run.sh --chips 16 --timeout 900 --tag p7 -- python -m pytest -m tt tests/tt/test_tt_bridge_device.py -q -p no:cacheprovider
# GPU-host procedure and scripts: docs/porting/ISAAC_RUNBOOK.md (isaac_env_server.py, train_isaac.py, bridge_smoke.py, tb_to_csv.py, compare_curves.py)
# Demo (not a phase; on main since 2026-11-12): MuJoCo humanoid walking, CPU simulation + TT training (docs/porting/demo_humanoid/README.md)
scripts/tt_run.sh --chips 16 --timeout 7200 --tag humanoid -- python scripts/train_humanoid_demo.py --backend tt --envs 4096 --iterations 600 --out docs/porting/demo_humanoid/tt_1chip
MUJOCO_GL=egl python scripts/render_humanoid_demo.py --checkpoints random .run/humanoid/tt_1chip/model_150.pt .run/humanoid/tt_1chip/model_599.pt --out docs/porting/demo_humanoid/walk.gif
MUJOCO_GL=egl python scripts/render_humanoid_demo.py --checkpoints random .run/humanoid/tt_1chip/model_150.pt .run/humanoid/tt_1chip/model_599.pt --seconds 15 --fps 33 --size 352 480 --no-eval --video docs/porting/demo_humanoid/walk.mp4
# Demo 2 (on main since 2026-11-12): Unitree G1 walking on command, MuJoCo on the CPU, PPO on all eight chips (docs/porting/demo_g1/README.md)
scripts/fetch_g1_model.sh && scripts/tt_run.sh --timeout 14400 --tag g1 -- python scripts/train_g1_demo.py --mesh 2x4 --envs 4096 --iterations 1500 --out docs/porting/demo_g1/tt_8chips
MUJOCO_GL=egl python scripts/render_g1_demo.py --checkpoints random .run/g1/tt_8chips/model_100.pt .run/g1/tt_8chips/model_1499.pt --video docs/porting/demo_g1/g1.mp4
# Examples (examples/README.md): the G1 demo as a short example; play.py needs no device
scripts/tt_run.sh --timeout 14400 --tag g1 -- python examples/g1_walk/train.py --mesh 2x4 --envs 4096 --iterations 1500      # -> .run/examples/g1_walk
MUJOCO_GL=egl python examples/g1_walk/play.py --checkpoint random .run/examples/g1_walk/model_100.pt .run/examples/g1_walk/model_1499.pt --video g1.mp4 --gif g1.gif
# MULTI-CHIP RULE (ADR-0004): chip-local loss / advantage normalisation / shuffle, gradients and KL averaged over the chips (gather + fp32 sum, never ttnn.all_reduce), global normalizer statistics.
# TRACED-PATH RULE (TTNN_ISSUES T-15): everything that keeps a value across trace replays is allocated BEFORE the first capture; one TTPPO per process.

# stop only jobs started by this repo (reads .run/pids)
scripts/kill_my_jobs.sh

# upstream integrity: rsl_rl/ must equal the pinned tag
git diff v5.5.2 -- rsl_rl/        # must be empty (also tested)
```
(Concrete commands are finalized as the scripts land; see PROGRESS.md.)

## Where things are
- `docs/porting/BRIEF.md` work order · `PROGRESS.md` log + next steps · `DECISIONS.md` ADRs · `TOLERANCES.md` · `PERF.md` · `TTNN_ISSUES.md`
- `docs/porting/UPSTREAM_MAP.md`, `UPSTREAM_SEMANTICS.md`, `TTRL_NOTES.md`, `TTNN_OP_SURVEY.md`, `ENV.md`
- `examples/` runnable examples (`g1_walk/`) · `rsl_rl/` upstream (untouched) · `rsl_rl_tt/` TT implementation (`device.py` = the only place that opens devices; `testing/` = host-only CPU envs, reference runs, golden access, parity utilities) · `tests/tt/` · `scripts/` · `.claude/`
- `docs/porting/golden/` golden baselines of the upstream reference (regenerate: `python scripts/train_reference.py`)
