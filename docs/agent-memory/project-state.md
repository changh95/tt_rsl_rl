# Project state (as of 2026-11-19)

Port of rsl_rl (Isaac Lab's PPO stack, upstream v5.5.2 kept byte-identical in `rsl_rl/`) to Tenstorrent Blackhole through TTNN as the plugin package `rsl_rl_tt/`.
Read order for a new session: `docs/porting/BRIEF.md` → `CLAUDE.md` → `docs/porting/PROGRESS.md`.

- **Done:** Phases 0-7 (tags `phase-0-done` … `phase-7-done`): traced act / update / GAE (7-19x faster than upstream CPU per iteration), 1x2 and 2x4 data parallel (5.3x on 8 chips at 65,536 envs, replicas bit-identical), ZeroMQ env bridge + `scripts/train_isaac.py` + `ISAAC_RUNBOOK.md` (Isaac Lab itself never run: no GPU host then).
- **Post-v1 done:** 8.1 heteroscedastic Gaussian, 8.2 RND, 8.3 symmetry, 8.4 distillation, 8.5 RNN (GRU/LSTM with BPTT on the device; upstream's padded trajectory mini-batches equal a per-env scan with resets after dones, proved on the host). Each merged and tagged `phase-8-N-done`.
- **Demo:** Unitree G1 walking on command (`examples/g1_walk/`, `docs/porting/demo_g1/`), model fetched by `scripts/fetch_g1_model.sh` (not committed), trained on 8 chips (2x4); CPU simulation is ~76 % of an iteration, so more chips barely help. Lessons: the model's ankle PD gains had to be raised; a sparse contact reward leaves the policy standing (needs a dense foot-height-vs-phase reward). The humanoid demo was removed at the owner's request.
- **8.6 CNN — BLOCKED** (branch `tt/phase-8-6-cnn`, not merged): `TTCNN` + CNN PPO implemented, component and parity tests green; `test_traced_cnn_path_survives_load` fails (a T-15-type trace state problem after `load()`, not understood). Two debug runs hit timeouts and left the chip unopenable; the second reset's outcome is unknown. Details: `docs/porting/incidents/2026-11-11_cnn_debug_job_timeout_chip16_mmio.md`. On a new host: first check whether it reproduces.
- **Next (BRIEF 4-b):** Phase 0-b host move → finish 8.6 → Phase 7 on the RTX 5090 host (same-host Isaac Lab) → G1 demo on 1 chip and multi-chip → 8.7 SAC.
- Isaac Lab note: v2.3.0 uses the rsl_rl 3.x config format; the TT backend needs 5.x (Isaac Lab main).
- 2026-11-12: the Galaxy working tree was deleted; restored from GitHub plus session records (nothing else lost from git).
