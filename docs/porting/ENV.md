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
| ttnn import path | **PyPI wheel `ttnn==0.81.1`** installed in `.venv` (ADR-0006, owner-approved 2026-11-10; cp310 manylinux_2_34, 103 MB; same version as the reference tree). The tt-metal tree has no `_ttnn.so` (`WITH_PYTHON_BINDINGS=OFF`) and is not used at runtime. `import ttnn` works inside the masked namespace (tested, opens no device). The wheel's runtime root is its own package directory. |
| sfpi (riscv JIT toolchain) | **7.89.0** extracted from the official release (`sfpi_7.89.0_x86_64_debian.txz`, sha256 `de16ebb14dec...b610b954` verified against the wheel's `sfpi-version`) to `.venv/lib/python3.10/site-packages/ttnn/runtime/sfpi` (436 MB). The host's `/opt/tenstorrent/sfpi` is 7.67.0 (does not match; the JIT prefers `<runtime>/sfpi`). |
| ttnn private paths | `scripts/ttnn_env.sh`: `TT_METAL_CACHE=.run/tt_metal_cache`, `TT_METAL_LOGS_PATH=.run/tt_logs`, `TTNN_CONFIG_OVERRIDES` -> `.run/ttnn_cache`, `.run/ttnn_tmp`, `.run/ttnn_reports` (ttnn's defaults `~/.cache/ttnn` and `/tmp/ttnn` are shared with other users and are left alone). |
| Other trees on this host | `~/experiments/{pi05,qwen3-32b,tt-models}/tt-metal` and `~/experiments/tt-metal-newton` belong to other work; the reference tree was modified by someone else on 2026-11-10 17:08-17:19 (egg-info removed, CMake reconfigured with tests ON). Treat all of them as read-only and unstable. |
| `PYTHON_ENV` | `/home/ttuser/experiments/rsl_rl/.venv` (dedicated; created 2026-11-10 with system `python3 -m venv`) |
| Python | 3.10.12 |
| torch | 2.11.0+cpu (CPU build; the tt-metal tree pins `torch==2.11.0` for x86_64; no CUDA on this host) |
| numpy | 1.26.4 (tt-metal requires `numpy>=1.24.4,<2`) |
| tensordict | 0.15.5 |
| onnx / onnxscript | 1.23.2 / 0.7.2 |
| tensorboard | 2.21.0 |
| pytest | installed in the venv |
| rsl_rl (`RSL_RL_VERSION`) | `v5.5.2` (commit `1156331`, 2026-10-28) = newest upstream release tag (2026-11-10); `main` is +4 untagged commits (C-2). Not yet checked against the user's Isaac Lab pin. `rsl-rl-lib 5.5.2` and `rsl_rl_tt` are both installed editable from this repo (`pip install -e .`; `include = ["rsl_rl*"]` in upstream `pyproject.toml` picks up both packages without touching upstream files). |
| Isaac Lab (`ISAACLAB_VERSION`) | unknown |

## Install notes (new dependencies are recorded here, BRIEF 2-7)
```bash
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install "numpy>=1.24.4,<2" pytest
.venv/bin/python -m pip install -e . "numpy>=1.24.4,<2"      # rsl-rl-lib 5.5.2 + rsl_rl_tt (editable)
.venv/bin/python -m pip install ttnn==0.81.1                  # 2026-11-10, ADR-0006; torch/numpy untouched (checked with --dry-run first)
# sfpi 7.89.0 toolchain for the JIT (the wheel does not ship it): official release, sha256-verified, venv-local
curl -fsSL -o sfpi.txz https://github.com/tenstorrent/sfpi/releases/download/7.89.0/sfpi_7.89.0_x86_64_debian.txz
echo "de16ebb14decfa4930a6f1b75c14655330cd62653bb38d90bd108163b610b954  sfpi.txz" | sha256sum -c -
tar -xf sfpi.txz -C .venv/lib/python3.10/site-packages/ttnn/runtime/
.venv/bin/python -m pip install onnxruntime==1.23.2          # 2026-11-11, Phase 4: ONNX export round-trip tests (adds coloredlogs, flatbuffers, humanfriendly; torch/numpy untouched)
.venv/bin/python -m pip install pyzmq==26.4.0                # 2026-11-11, Phase 7: env bridge (libzmq 4.3.5 is bundled in the wheel; nothing else changed)
.venv/bin/python -m pip install "gymnasium[mujoco]==1.4.0"      # 2026-11-12, MuJoCo demos (first the humanoid demo, removed on 2026-11-11 at the owner's request; now the G1 demo): mujoco 3.17.0 + the Humanoid model of gymnasium (adds Farama-Notifications, ImageIO, PyOpenGL, etils, glfw, websockets...; torch/numpy/ttnn untouched, checked with --dry-run first); headless rendering: MUJOCO_GL=egl (Mesa)
.venv/bin/python -m pip install imageio-ffmpeg==0.8.1             # 2026-11-12, demo video: a static ffmpeg (libx264) inside the wheel; nothing else changed (dry-run first); no system ffmpeg on this host
# G1 demo (2026-11-12): no pip package; `scripts/fetch_g1_model.sh` sparse-clones the unitree_g1 directory of google-deepmind/mujoco_menagerie at commit 370c43965323c0944b433b202aac5058b5f907a9 into .run/assets (git-ignored, 38 MB, BSD-3-Clause, (c) Unitree Robotics); the files are data, nothing in them is executed
```
Transitive: GitPython, onnx, onnxscript, tensordict, tensorboard (all from upstream `pyproject.toml`); from the ttnn wheel: click, graphviz, loguru,
matplotlib, pandas, pyyaml, seaborn (+ contourpy, cycler, fonttools, kiwisolver, pyparsing, python-dateutil, pytz, six, tzdata).

## Tray / device facts
| Item | Value |
|------|-------|
| `TRAY_ID` | 3 |
| Allowed nodes | `/dev/tenstorrent/16` .. `/dev/tenstorrent/23` — 8 nodes, all present (char devices, mode 666 root:root, so file permissions do **not** protect other trays; the guards in `device.py`/hooks do) |
| Node -> PCI BDF (sysfs symlink read, no open) | 16=`0000:c1:00.0` 17=`0000:c2:00.0` 18=`0000:c3:00.0` 19=`0000:c4:00.0` 20=`0000:c5:00.0` 21=`0000:c6:00.0` 22=`0000:c7:00.0` 23=`0000:c8:00.0` (consecutive PCI buses = one tray). Re-verified by `device.py` on every open. |
| `TT_VISIBLE_DEVICES` ID space | An integer token is a **0-based index into the nodes sorted by PCI BDF**, NOT the `/dev/tenstorrent/N` number (UMD `pci_device.cpp:265-325`; tt-metal's own tests say "logical IDs (BDF-sorted indices)"). `TT_METAL_VISIBLE_DEVICES` is a dead variable. So `scripts/tray_env.sh` uses **full BDF tokens** (typos fail closed). |
| What `TT_VISIBLE_DEVICES` does NOT guarantee | UMD enumeration briefly open()s/closes every `/dev/tenstorrent/*` node before filtering. Resolved by ADR-0005 option II: every job runs in a private user+mount namespace where `/dev/tenstorrent` lists only the selected chips (`scripts/ns_mask_dev.sh`), so no other tray's node exists for it. |
| Tray mesh shape | not confirmed on hardware. Source reading: metal's auto-discovery over 8 visible chips infers **2x4** (a `1x8` request is placed in ring order); fabric bring-up on a tray subset is the main unknown (survey Q3). To be confirmed in the smoke test. |
