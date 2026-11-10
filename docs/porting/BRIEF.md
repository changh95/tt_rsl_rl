<!--
Saved verbatim from the original work order (Korean) on 2026-11-10.
ONLY section 0 was changed: the placeholders were replaced by the values confirmed with the owner on 2026-11-10
(see the "Resolved on 2026-11-10" notes). Everything else is the original text.
New sessions: read BRIEF.md -> CLAUDE.md -> docs/porting/PROGRESS.md in this order.
-->

# rsl_rl → Tenstorrent Blackhole 포팅 작업 지시서 (Claude Code용)

이 문서 전체가 너에게 주는 작업 지시서다. 몇 주 걸리는 장기 작업이라 세션이 끊기거나 컨텍스트가 압축될 수 있다. 작업은 전부 GitHub 저장소 `changh95/tt_rsl_rl`(upstream rsl_rl을 clone한 저장소)에서 하고, 진행 상황도 그 저장소에 기록한다. Phase 0에서 이 문서를 그 저장소의 `docs/porting/BRIEF.md`로 저장하고, 새 세션을 시작할 때마다 `BRIEF.md` → `CLAUDE.md` → `docs/porting/PROGRESS.md` 순서로 읽고 이어서 진행해라.

## 0. 내가 채워 둔 값

```text
TRAY_ID            = 3
ALLOWED_DEV_NODES  = /dev/tenstorrent/16 ~ 23   (8개 노드, 2026-11-10 존재 확인)
ALLOWED_PCI_BDFS   = (미제공)
TT_VISIBLE_DEVICES = (미정 — 선택한 TT_METAL의 tt-umd 소스에서 ID 체계를 확인한 뒤 확정. Galaxy에서는 UMD logical ID와 /dev 번호가 다를 수 있음)
TRAY_MESH_SHAPE    = 모름   (Phase 0에서 허용된 칩만으로 확인)
TRAY_LAUNCH_NOTES  = (미제공)
TT_METAL           = /home/ttuser/experiments/tt-metal   (v0.81.1, 소스 빌드 존재, python_env 없음)
PYTHON_ENV         = /home/ttuser/experiments/rsl_rl/.venv   (전용 venv를 새로 만든다. tt-metal 트리는 수정하지 않는다)
RSL_RL_VERSION     = v5.5.2   (임시: 내 Isaac Lab이 쓰는 버전과 일치하는지 미확인. upstream 태그: v5.5.0 / v5.5.1 / v5.5.2)
ISAACLAB_VERSION   = 모름
ISAAC_HOST         = 모름   (Phase 7 전에 확정)
GITHUB_REPO        = changh95/tt_rsl_rl   (고정. public, non-fork, 2026-11-10 생성)
UPSTREAM_REPO      = https://github.com/leggedrobotics/rsl_rl   (고정)
WORKDIR            = /home/ttuser/experiments/rsl_rl
```

Resolved on 2026-11-10 (owner answers): new public non-fork repo; tray 3 = `/dev/tenstorrent/16`~`23` and no other tray may ever be used; "follow all recommendations" for `RSL_RL_VERSION` (v5.5.2, provisional), `WORKDIR`, `TT_METAL` (`~/experiments/tt-metal` v0.81.1 + dedicated venv); `TRAY_MESH_SHAPE`/`TRAY_LAUNCH_NOTES`/`ISAAC_HOST` unknown.

값이 비었거나 서로 맞지 않으면(노드 수 ≠ 8, ID 체계 불확실 등) 디바이스를 열기 전에 멈추고 물어봐라. 추측으로 채우지 마라. `TRAY_MESH_SHAPE`가 "모름"이면 Phase 0에서 허용된 칩만으로 확인하고, 그걸로 안 되면 질문해라.

## 1. 목표와 범위

Isaac Lab이 쓰는 rsl_rl(`RSL_RL_VERSION`)의 학습 측 전체 — 롤아웃 중 정책 추론과 PPO 업데이트 — 를 TTNN으로 Blackhole에서 실행한다. 환경(시뮬레이터)은 원래 자리(CPU 테스트 환경, 또는 GPU 호스트의 Isaac Lab)에 둔다.

v1 완료 조건:
1. 학습 상태(파라미터, Adam 상태, 관측 정규화 통계, 롤아웃 버퍼, advantage/return)가 전부 디바이스에 상주한다. 호스트↔디바이스 트래픽은 스텝당 obs·reward·done 업로드와 action 다운로드, 이터레이션당 로그용 스칼라 정도로 제한한다.
2. upstream과 수치적으로 동등하다: 부품별 parity 테스트 + 여러 seed의 학습 곡선.
3. rsl_rl의 공개 API와 설정 형식을 유지해서, Isaac Lab train/play 스크립트가 설정 변경(또는 최소 패치)만으로 동작한다.
4. 체크포인트가 upstream 형식과 양방향 호환이다. upstream `load()`, JIT/ONNX export, Isaac Lab play가 그대로 동작한다.
5. 1칩에서 트레이 8칩 data-parallel로 확장된다. 의미론은 동기식 gradient all-reduce(= rsl_rl multi-GPU와 동일)다.

v1 범위: PPO, MLPModel, GaussianDistribution(scalar/log std), EmpiricalNormalization, RolloutStorage/GAE, OnPolicyRunner 연동.
v1 이후(내 승인 후): RNNModel(GRU → LSTM), HeteroscedasticGaussianDistribution, RND, Symmetry, Distillation, CNNModel, SAC.

## 2. 절대 규칙 — 트레이 격리와 공용 서버 위생

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

## 3. 설계 방향

### 3.0 확정 사항 (내가 결정함 — 다시 논의하지 않는다)
1. 1칩 bring-up → 멀티칩 확장. 코드는 처음부터 mesh-native로 쓰고, 1x1에서 parity를 맞춘 뒤 1x2 → 트레이 전체로 올린다. (3.6)
2. 플러그인 구조: `changh95/tt_rsl_rl`은 upstream을 clone한 저장소지만, upstream 코드(`rsl_rl/`)는 `RSL_RL_VERSION` 태그 그대로 두고 TT 구현은 같은 저장소의 별도 패키지 `rsl_rl_tt/`에 둔다. (3.1)
3. trace가 기본 설계: act·update·GAE를 고정 shape + trace replay로. trace 없는 경로는 디버깅용으로만. (3.3)
4. tt-rl(MIT)을 참고 구현으로 적극 활용한다. 코드를 가져오면 출처·라이선스를 표기한다. 단, 섹션 2-4대로 CLI 실행은 금지.

Phase 0에서 이 결정을 막는 문제를 발견하면 혼자 바꾸지 말고 근거와 함께 보고해라. 아래 나머지 항목은 Phase 0에서 검증하고 ADR로 확정한다.

### 3.1 통합: 플러그인 (확정)
rsl_rl 5.x는 커스텀 알고리즘·모델 클래스를 라이브러리 수정 없이 설정으로 주입할 수 있다. 구조:
- 저장소 구조: `rsl_rl/`(upstream, `RSL_RL_VERSION` 태그 그대로 — 정답 oracle이자 플러그인이 꽂히는 대상), `rsl_rl_tt/`(TT 구현: `TTPPO`, 모델·분포·정규화·스토리지, `device.py`), `tests/tt/`, `scripts/`, `docs/porting/`, `.claude/`. 두 패키지 모두 `PYTHON_ENV`에 editable로 설치한다.
- `rsl_rl/` 아래 파일은 수정하지 않는다. `git diff <RSL_RL_VERSION> -- rsl_rl/`가 비어 있는지(또는 승인된 패치 목록과 일치하는지) 확인하는 테스트를 둔다. 이렇게 해야 upstream 새 버전으로 올릴 때 `rsl_rl/`만 갈아끼우면 된다.
- upstream `OnPolicyRunner`를 그대로 쓴다. runner의 torch device는 `cpu`(호스트 스테이징)로 두고, mesh shape·dtype 정책·trace 사용 여부 같은 TT 설정은 알고리즘 cfg로 받는 구조를 우선 검토한다. Isaac Lab cfg 클래스가 추가 키를 받지 못하면 별도 YAML이나 환경변수로 받는 방법을 검토한다.
- `get_policy()`와 export는 디바이스 가중치를 내려받은 torch 미러 모델을 돌려줘서 upstream JIT/ONNX export를 재사용한다.

주입만으로 안 되는 지점이 있으면 얇은 `TTOnPolicyRunner` 서브클래스까지는 허용한다. upstream 소스 패치가 필요해 보이면 패치 내용과 사유를 정리해서 내 승인을 받는다. 실제 주입 지점(설정에서 클래스를 지정하는 방식, 알고리즘 생성 경로, save/load가 무엇을 직접 만지는지)은 고정 버전 소스에서 확인한다.

### 3.2 학습 백엔드 (ADR-0001)
- A (기본): 손으로 쓴 TTNN forward/backward. MLP(Linear + ELU), Gaussian log_prob/entropy, PPO 손실, 전역 grad-norm clipping, 디바이스 Adam. 그래프가 작고 고정이라 수치·fusion·trace를 완전히 통제할 수 있다.
- B: tt-train(`ttml`) autograd/optimizer. 필요한 op 전부, fp32 optimizer 상태, 멀티디바이스, trace 호환이 설치 버전에서 확인될 때만 고른다.

어느 쪽이든 정답 판정은 upstream torch 모듈의 autograd 대비 gradient parity로 한다.

### 3.3 성능 설계: 고정 shape + trace (확정)
작은 MLP에서는 Python에서 op를 하나씩 디스패치하는 오버헤드가 연산 시간보다 크다. 그래서:
- hot path 텐서는 전부 사전 할당하고 shape를 고정한다. (1) 롤아웃 act 스텝, (2) minibatch update 스텝(fwd + bwd + clip + Adam), (3) GAE/return 계산을 각각 trace capture → replay 한다. trace 영역 크기처럼 mesh를 열 때 정해야 하는 설정은 `device.py` 진입점에서 처리한다.
- 칩당 batch와 minibatch가 tile(32)의 배수가 되도록 num_envs·num_steps·num_mini_batches 조합을 검증하고, 안 맞으면 명확한 에러로 거부한다. 조용히 패딩해서 의미를 바꾸지 않는다. 짧은 마지막 minibatch가 생기지 않게 한다.
- 이터레이션마다 바뀌는 값(adaptive LR 등)은 trace 안에 상수로 굳지 않도록 디바이스 텐서 입력으로 둔다.
- obs는 사전 할당 버퍼로 host→device 복사하고 tilize는 디바이스에서 한다. 스텝당 동기화 지점은 action 다운로드 1회를 목표로 한다. obs group 결합(concat)은 경계에서 하는 데이터 정리라 호스트에서 해도 된다(폴백 아님).

### 3.4 수치 정책
- master weight와 Adam moment는 fp32. 처음엔 matmul도 fp32(또는 HiFi4 + fp32 dest accumulation)로 parity부터 맞춘다. bf16 등은 Phase 5에서 학습 곡선 영향을 측정한 뒤에만 채택한다.
- 롤아웃의 old log_prob과 업데이트의 new log_prob은 같은 커널 경로·dtype·compute config로 계산한다. 다르면 첫 minibatch에서도 ratio가 1이 아니게 되어 clipping이 편향된다. 이걸 확인하는 테스트를 둔다.
- 차원 패딩: 패딩된 action 차원은 log_prob·entropy 합에서 마스킹하고, std 파라미터의 패딩 칸은 고정값에 gradient 0. batch 평균의 분모는 실제 샘플 수.
- batch 축 reduction(평균·분산·손실 평균)은 fp32로 누적한다. 손실 경로의 exp/log/ELU는 근사 모드를 끄고 정확도를 확인한다.

### 3.5 난수와 셔플
- parity 모드: action 노이즈는 호스트 torch generator로 만들어 업로드한다(레퍼런스와 같은 스트림). minibatch 순열도 호스트 `torch.randperm`으로 만들어 인덱스만 올리고, 행 gather는 디바이스에서 한다.
- 성능 모드: 디바이스 RNG 허용. 분포 검증 테스트로 확인한다.

### 3.6 멀티칩: 1칩 bring-up → mesh 확장 (확정)
코드는 처음부터 MeshDevice 기준으로 쓰고, 1x1 → 1x2 → 트레이 전체(`TRAY_MESH_SHAPE`) 순서로 올린다. env batch와 스토리지는 batch 축으로 칩에 샤딩하고, 파라미터와 optimizer 상태는 replicate하고, gradient는 CCL all-reduce(mean) 후 모든 칩이 같은 Adam 스텝을 밟는다. rsl_rl multi-GPU의 동기식 gradient 평균과 같은 의미론이다(주기적 파라미터 평균이 아님). 칩 간 파라미터 일관성을 주기적으로 검사한다. 관측 정규화 통계, advantage 정규화, KL 기반 LR을 칩별로 할지 전역으로 할지는 upstream multi-GPU 동작을 확인해서 ADR-0004로 정한다.

### 3.7 반드시 보존할 upstream 의미론
고정 버전 소스에서 확인해 `docs/porting/UPSTREAM_SEMANTICS.md`에 항목별로 소스 위치와 대응 테스트를 적는다: time-out bootstrapping, done 마스크를 쓰는 GAE, advantage 정규화(전역/minibatch별 옵션), clipped surrogate, clipped value loss, entropy 계수, actor+critic 전체에 대한 max_grad_norm, KL 기반 adaptive LR과 schedule, Adam 하이퍼파라미터, 정규화 통계 업데이트 시점, std 파라미터화, obs group/obs_set 결합 방식.

### 3.8 환경 위치
Galaxy 호스트에는 NVIDIA GPU가 없을 가능성이 높으니 Isaac Sim은 별도 GPU 호스트에서 돈다고 가정한다. 학습 루프(runner + 알고리즘)는 TT 호스트에 두고 환경만 원격으로 붙인다: TT 호스트의 `RemoteVecEnv`(rsl_rl VecEnv 구현) ↔ GPU 호스트의 서버(Isaac Lab env 래핑). 같은 호스트라면 직접 연결한다. (Phase 7)

## 4. 단계별 계획 — 완료 기준을 채워야 다음 Phase로

### Phase 0 — 환경 파악, 안전장치, 조사 (끝나면 ⛔ 멈추고 보고)
1. 저장소 준비:
   - `UPSTREAM_REPO`를 clone하고 `RSL_RL_VERSION` 태그를 기준으로 작업 브랜치를 만든다(upstream 히스토리 보존). remote를 규칙 2-10대로 설정한다.
   - `changh95/tt_rsl_rl`이 GitHub에 이미 있는지 확인한다. 없으면 만들지 말고 멈춰서 나에게 생성(또는 `gh repo create` 실행 허락)을 요청한다. 비어 있지 않은 저장소면 덮어쓰지 말고 내용을 보고한다.
   - 첫 push 전에 `git remote -v`로 push 대상이 `changh95/tt_rsl_rl`뿐인지 확인한다.
   - 최상위 README에 이 저장소가 rsl_rl(BSD-3) 기반이라는 것, 원 저장소 링크, 기준 태그, 진행 상황 문서 위치를 적는다. upstream LICENSE는 유지한다.
   - 이 문서를 `docs/porting/BRIEF.md`로 저장한다. 섹션 2 전문 + 목표 요약 + 실행/테스트 명령 + 문서 위치로 `CLAUDE.md`를 작성한다.
2. 안전장치:
   - `scripts/tray_env.sh`: 섹션 0의 값으로 디바이스 관련 환경변수를 설정하는 유일한 곳.
   - `scripts/tt_run.sh`: tray_env 로드 → 트레이별 lock 파일로 `flock` → `timeout`(먼저 SIGINT, 유예 후 kill) → 명령 실행. 띄운 PID는 `.run/pids`에 기록.
   - `rsl_rl_tt/device.py`: 섹션 2-2의 단일 진입점. close는 `finally`와 atexit 양쪽에서 보장.
   - `.claude/hooks/tray_guard.sh` + `.claude/settings.json`: Bash PreToolUse 훅이 섹션 2-4, 2-5의 금지 명령을 exit 2로 차단한다. 소스 읽기(grep/cat)가 오탐으로 막히지 않도록 명령어 위치 기준으로 매칭하고, 차단/허용 케이스를 직접 테스트한다. `permissions.deny`로 `.claude/**`와 `scripts/tray_env.sh` 편집을 막는다(Bash로 우회 수정하는 것도 금지). 훅은 `upstream` remote로의 push와 force-push(`--force`, `-f`, `+refspec`)도 차단한다. 다 만들면 나에게 `/hooks` 확인과 세션 재시작을 요청한다. 훅은 보조 장치일 뿐이고 규칙은 그대로 지킨다.
3. `docs/porting/ENV.md`: ttnn/tt-metal 버전(커밋), tt-umd, Python, torch, rsl_rl 고정 버전, 호스트의 NVIDIA GPU 유무(`nvidia-smi` 존재 여부만).
4. 스모크 테스트 (모두 `tt_run.sh` 경유):
   - (a) 1x1 mesh로 첫 허용 칩 열기 → 작은 matmul을 torch와 비교 → 닫기. fd 가드가 실제로 동작하는지 확인.
   - (b) 트레이 전체 mesh 열기 → 작은 all-reduce → 열린 노드가 정확히 8개이고 전부 허용 목록인지 확인. 트레이 단위 실행에 추가 설정이 필요한데 `TRAY_LAUNCH_NOTES`로 해결되지 않으면 멈추고 질문.
5. upstream 정독: 고정 버전 rsl_rl의 runner, PPO, models, modules, storage, env 인터페이스, multi-GPU 경로, save/load, export를 읽고 `docs/porting/UPSTREAM_MAP.md`(컴포넌트별 대응표와 주입 지점)와 `UPSTREAM_SEMANTICS.md`(3.7 체크리스트)를 작성한다.
6. 조사:
   - 필요한 TTNN op의 존재·dtype·정확도: matmul/linear(fp32 포함), ELU와 그 미분, exp/log/sqrt, clamp/minimum/maximum/where, 축 reduction, concat/slice, 행 gather(embedding 계열), 난수, CCL(all-reduce 또는 reduce-scatter + all-gather), trace capture/replay, compute kernel config(math fidelity, fp32 dest acc, approx mode).
   - fp32 matmul과 주요 eltwise의 실측 오차 바닥(noise floor). 이후 허용오차의 근거가 된다.
   - tt-train(`ttml`) 사용 가능 여부와 빌드 상태.
   - 참고 구현 tt-rl(github.com/scaleresearch/tt-rl, MIT) 정독: PPO의 수동 gradient, 디바이스 Adam, trace capture/replay, 호스트↔디바이스 데이터 흐름을 정리해 `docs/porting/TTRL_NOTES.md`에 쓰고, 그대로 가져올 것/고쳐 쓸 것/쓰지 않을 것(예: 주기적 파라미터 평균 방식의 멀티칩)을 구분한다. CLI는 실행하지 않는다.
   - tt-metal tech_reports(Programming Mesh of Devices, CCL Performance Tuning Tips), tests/와 models/의 trace·CCL 사용 예.
7. `docs/porting/DECISIONS.md`에 ADR 작성: 0001 학습 백엔드(A/B), 0002 플러그인 주입 지점과 서브클래스 필요 여부(구조 자체는 확정), 0003 환경 위치와 브릿지, 0004 멀티칩 통계 동기화.
8. ⛔ 멈추고 보고: 발견한 것, 리스크, 계획 수정안. 내 승인 후 Phase 1.

### Phase 1 — 레퍼런스와 parity 하네스 (호스트만)
- 고정 버전 upstream rsl_rl(CPU torch)을 정답으로 쓴다. 재구현하지 말고 upstream 모듈을 import해서 비교한다.
- CPU 테스트 환경 (rsl_rl VecEnv 구현, TensorDict obs group 포함):
  1. 배치 CartPole
  2. Pendulum swing-up
  3. Isaac Lab 보행 태스크와 비슷한 차원의 합성 과제 — 12차원 액추에이터의 이중적분기형 명령 추종. 도메인 랜덤화 파라미터는 critic 전용 privileged obs로, 고정 길이 time-out과 조기 종료를 모두 포함.

  전부 seed로 결정적이어야 하고, 4096 env 이상에서도 CPU로 충분히 빨라야 한다.
- upstream으로 환경별 seed 3개 학습 → 골든 곡선(CSV)과 소요 시간 저장.
- parity 유틸: max abs/rel error, PCC, 실패 시 최악 원소 리포트. 허용오차 표는 `docs/porting/TOLERANCES.md` (Phase 0 noise floor 근거).
- 완료 기준: 골든 베이스라인이 있고, 각 환경이 upstream으로 합리적인 시간 안에 학습된다.

### Phase 2 — 1칩(1x1 mesh) forward
- TT MLPModel forward(actor/critic), EmpiricalNormalization(정규화 + 통계 업데이트), GaussianDistribution(mean/std, 주입 노이즈 샘플링, log_prob, entropy).
- torch → TT 가중치 복사 유틸, 패딩 경로를 포함한 랜덤 shape 테스트.
- `act()`를 CPU 테스트 환경 롤아웃에 연결한다. 이 Phase에서만 스토리지를 호스트에 둬도 된다(DECISIONS에 임시 허용으로 기록).
- 완료 기준: 부품별 forward parity 통과, TT 추론으로 롤아웃이 끝까지 돈다.

### Phase 3 — 1칩 backward, optimizer, PPO update
- ADR-0001대로 backward, PPO 손실 전부, 전역 grad-norm clipping, Adam, adaptive LR을 구현한다.
- RolloutStorage, GAE, minibatch 생성을 디바이스로 옮긴다.
- 테스트: 파라미터별 gradient parity, Adam 1스텝 parity, `update()` 1회 전체 parity(각 손실, KL, LR), 같은 데이터를 넣었을 때 20 이터레이션 동안의 파라미터 궤적 drift.
- 완료 기준: 전부 통과, update 경로에서 허용되지 않은 호스트 폴백 0.

### Phase 4 — 1칩 end-to-end 학습과 호환성
- 테스트 환경 × seed 3개를 TT로 학습해 골든과 비교한다. 초기 기준: TT 최종 평균 보상 ≥ 레퍼런스 평균 − max(레퍼런스 seed 간 표준편차, |레퍼런스 평균|의 5%), NaN/Inf 없음, 곡선 그래프 첨부.
- 체크포인트 왕복: TT 저장 → upstream `load()` → 추론 출력 일치. upstream 저장 → TT에서 이어 학습. optimizer state까지 upstream과 같은 키 구조. TT 체크포인트로 upstream ONNX export → onnxruntime 출력 일치.
- 완료 기준: 위 전부 통과.

### Phase 5 — 1칩 성능
- trace(act, update, GAE), 사전 할당 버퍼, 동기화 최소화. trace 경로에서 Phase 2–4 테스트를 다시 통과해야 한다.
- 이터레이션 시간 분해(env step / H2D / act / D2H / update)와 op 프로파일. 비교 대상: upstream CPU, 그리고 내가 주는 upstream GPU 수치.
- 벤치 매트릭스: num_envs {4096, 16384, 65536} × hidden {[512, 256, 128], [1024, 1024, 512]}.
- 정밀도 실험(bf16 matmul 입력 등)은 학습 parity가 유지될 때만 채택한다.
- 결과는 `docs/porting/PERF.md`에 수치로 남긴다. TT가 불리한 구간도 그대로 적는다.

### Phase 6 — 트레이 멀티칩
- 1x2 → `TRAY_MESH_SHAPE`. 3.6의 data-parallel 구현.
- 테스트: N칩 vs 1칩 동치(같은 전역 샘플 집합이 되도록 인덱스를 구성해 비교), K 이터레이션 후 칩 간 파라미터 일치, 열린 노드 검증(정확히 8개).
- 스케일링 벤치(칩 1/2/8 × num_envs)를 PERF.md에 추가한다.

### Phase 7 — Isaac Lab 연동
- `ISAAC_HOST`가 같은 호스트면 설정 변경으로 연결하고, 필요한 Isaac Lab 쪽 최소 패치 목록을 만든다.
- 별도 GPU 호스트면 env 브릿지를 만든다. 기본은 ZeroMQ + float32 raw 버퍼(대안은 ADR-0003에서 비교). reset, time-out, extras(에피소드 로그)를 전달하고, 왕복 지연과 대역폭을 측정한다. 서버는 Isaac Lab의 Python(ttnn과 버전이 다를 수 있음)에서 돌기 때문에 의존성을 최소로 한다.
- Isaac Lab 태스크의 agent cfg는 GPU 호스트에서 dict/YAML로 내보내 TT 호스트에서 그대로 로드한다(설정 일치 보장).
- 같은 호스트 loopback(서버가 CPU 테스트 환경을 래핑)으로 먼저 검증한 뒤, 내가 GPU 호스트에서 돌릴 runbook과 스크립트를 준다: Isaac-Cartpole → 보행 태스크 하나(예: Anymal-D 또는 Go2 flat). GPU 레퍼런스 대비 곡선 비교, TT 체크포인트를 GPU 호스트의 Isaac Lab play로 재생하는 절차 포함.

## 5. 작업 방식과 보고
- 진행 상황은 `changh95/tt_rsl_rl`의 `docs/porting/PROGRESS.md`에 기록하고, 작업 단위마다 갱신해 커밋·push한다: 날짜, Phase, 완료 / 다음 / 막힌 것 / 핵심 수치, 관련 커밋 해시. README에는 Phase별 상태 표(대기/진행/완료)를 두고 PROGRESS.md와 함께 갱신한다.
- Phase 완료 시: 기능 브랜치를 `main`에 병합하고, `phase-N-done` 태그를 달고, PROGRESS.md에 완료 기준 충족 근거(테스트 결과, 수치, 그래프 경로)를 남긴다.
- 결정은 `DECISIONS.md`(ADR), 허용오차는 `TOLERANCES.md`, 성능은 `PERF.md`, TTNN 이슈는 `TTNN_ISSUES.md`.
- 코드·주석·커밋 메시지·문서는 영어, 나에게 하는 보고는 한국어. 보고 형식: 3줄 요약 → 수치 → 내가 결정할 것.
- 디바이스 테스트는 pytest marker `tt`로 구분하고 `tt_run.sh`로만 돌린다. 호스트 테스트는 디바이스 없이 통과해야 한다.
- 코드 읽기·조사는 서브에이전트로 병렬화해도 되지만 디바이스 잡은 직렬로.
- API를 추측으로 지어내지 않는다. 설치된 ttnn의 docstring/`help()`, tt-metal의 tests·models 코드에서 실제 사용 예를 찾고, 최소 재현 스크립트로 확인한다.
- Phase가 끝날 때마다 짧게 보고한다. Phase 0 종료, ADR 변경, 섹션 2에 닿는 상황에서는 반드시 멈추고 승인을 기다린다.

## 6. 참고 자료
- rsl_rl 문서(Overview / Configuration / API)와 소스 — 고정 버전 기준
- Isaac Lab `isaaclab_rl`의 rsl_rl 래퍼, train/play 스크립트 — `ISAACLAB_VERSION` 기준
- tt-metal: tech_reports(Programming Mesh of Devices, CCL Performance Tuning Tips), TTNN API 문서, tests/와 models/의 trace·CCL 예제
- tt-train(`ttml`) — ADR-0001 B안 검토 시
- tt-rl(scaleresearch/tt-rl, MIT) — 참고 구현 (코드 재사용 시 출처 표기, CLI 실행 금지)
