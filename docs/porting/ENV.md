# Environment (Phase 0)

Recorded 2026-11-10. Values marked *(pending)* are filled in once the corresponding Phase 0 step has run.

## Host
| Item | Value |
|------|-------|
| OS / kernel | Ubuntu 22.04.5 LTS, Linux 6.8.0-107-generic |
| CPU / RAM | 64 logical CPUs, 566 GiB RAM |
| NVIDIA GPU tooling | **none** (`nvidia-smi` not found) — Isaac Sim must run on a separate GPU host (ADR-0003) |
| Tenstorrent kernel driver | `tenstorrent` module 2.10.0 (`/sys/module/tenstorrent/version`) |
| Shared host | yes: other people's jobs run here. This project only ever uses tray 3 = `/dev/tenstorrent/16`..`23` |

## Software stack
| Item | Value |
|------|-------|
| tt-metal tree (`TT_METAL`) | `/home/ttuser/experiments/tt-metal`, tag `v0.81.1`, commit `d80d9482a418cf04ddcbeecf6adb67010d3091e6` (2026-10-22). Built (`build/lib`), no `python_env`. **Read-only for this project** (BRIEF 2-7). |
| tt-umd | submodule `tt_metal/third_party/umd` @ `889651ea5525379cc6bf1807e56404661b48a76d` |
| ttnn import path | *(pending: Phase 0 step 4, see TTNN_OP_SURVEY.md item 13)* |
| `PYTHON_ENV` | `/home/ttuser/experiments/rsl_rl/.venv` (dedicated; created 2026-11-10 with system `python3 -m venv`) |
| Python | 3.10.12 |
| torch | 2.11.0+cpu (CPU build; the tt-metal tree pins `torch==2.11.0` for x86_64; no CUDA on this host) |
| numpy | 1.26.4 (tt-metal requires `numpy>=1.24.4,<2`) |
| tensordict | 0.15.5 |
| onnx / onnxscript | 1.23.2 / 0.7.2 |
| tensorboard | 2.21.0 |
| pytest | installed in the venv |
| rsl_rl (`RSL_RL_VERSION`) | `v5.5.2` (commit `1156331`, 2026-10-28) — **provisional**, not yet checked against the user's Isaac Lab pin. `rsl-rl-lib 5.5.2` and `rsl_rl_tt` are both installed editable from this repo (`pip install -e .`; `include = ["rsl_rl*"]` in upstream `pyproject.toml` picks up both packages without touching upstream files). |
| Isaac Lab (`ISAACLAB_VERSION`) | unknown |

## Install notes (new dependencies are recorded here, BRIEF 2-7)
```bash
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install "numpy>=1.24.4,<2" pytest
.venv/bin/python -m pip install -e . "numpy>=1.24.4,<2"      # rsl-rl-lib 5.5.2 + rsl_rl_tt (editable)
```
Transitive: GitPython, onnx, onnxscript, tensordict, tensorboard (all from upstream `pyproject.toml`).

## Tray / device facts
| Item | Value |
|------|-------|
| `TRAY_ID` | 3 |
| Allowed nodes | `/dev/tenstorrent/16` .. `/dev/tenstorrent/23` — 8 nodes, all present (char devices, mode 666 root:root, so file permissions do **not** protect other trays; the guards in `device.py`/hooks do) |
| `TT_VISIBLE_DEVICES` ID space | *(pending: determined from the tt-umd/tt-metal source, see TTNN_OP_SURVEY.md item 11)* |
| Tray mesh shape | *(pending: determined in Phase 0 step 4 with the allowed chips only)* |
