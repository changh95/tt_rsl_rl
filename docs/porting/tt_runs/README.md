# TT end-to-end training runs (one Blackhole chip) vs the upstream golden curves

Produced by `scripts/train_tt.py` (device job, one process, runs one after the other on the open 1x1 mesh of tray 3, node 16):
the UNMODIFIED upstream `OnPolicyRunner` with `cfg["algorithm"]["class_name"] = "rsl_rl_tt.ppo:TTPPO"`, fp32, the same configuration, seeds and
iteration counts as the golden runs in `../golden/`. Regenerate: `scripts/tt_run.sh --chips 16 --timeout 3600 --tag train -- python scripts/train_tt.py`.

| env | golden (upstream CPU, 3 seeds) | TT 1 chip (3 seeds) | Phase 4 threshold | passes | wall time per run |
|-----|-------------------------------|---------------------|-------------------|--------|-------------------|
| cartpole | 500.0 +- 0.0 | 500.0, 500.0, 500.0 (mean 500.0) | 475.0 | yes | 68 s (golden CPU: 20 s) |
| pendulum | -152.1 +- 2.4 | -148.1, -160.1, -154.3 (mean -154.2) | -159.7 | yes | 143 s (golden CPU: 50 s) |
| tracking12 | 190.7 +- 1.4 | 194.9, 191.8, 194.4 (mean 193.7) | 181.2 | yes | 322 s (golden CPU: 230 s) |

Criterion (BRIEF Phase 4): TT final mean reward (mean of the last 10 iterations, averaged over seeds) >= reference mean - max(reference seed std, 5 % of |mean|),
no NaN/Inf in any logged scalar (`tests/tt/test_tt_runs.py` checks the committed CSVs). `<env>_vs_golden.png` overlays the curves.
The TT runs are NOT faster than the CPU yet: they are untraced and per-op dispatched (Phase 5).
