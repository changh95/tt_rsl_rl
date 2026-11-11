# Golden baselines (upstream rsl_rl, CPU torch)

Reference curves produced by the **unmodified** upstream `rsl_rl` (tag `v5.5.2`, `OnPolicyRunner` + `PPO` + `MLPModel`, device `cpu`)
on the three CPU test environments of `rsl_rl_tt/testing/envs.py`. They are the target for the Phase 4 end-to-end comparison
(BRIEF Phase 4: TT final mean reward >= reference mean - max(seed std, 5 % of |mean|)).

| File | Content |
|------|---------|
| `<env>/seed<k>.csv` | one row per iteration: `iteration`, `wall_s`, and every scalar the upstream `Logger` emits (`Train/mean_reward`, `Train/mean_episode_length`, `Loss/*`, `Policy/mean_std`, `Perf/*`, `Episode_*`) |
| `summary.json` | per env/seed: iterations, wall time, first and final reward (mean of the last 10 iterations); mean/std over seeds; machine and library versions |
| `<env>_curves.png` | `Train/mean_reward` of the 3 seeds |

Regenerate (host only, no device; about 5 minutes with 3 runs in parallel, 8 torch threads each):

    python scripts/train_reference.py

The run is a pure function of (seed, torch build, thread count): `tests/tt/test_reference.py` checks that two runs agree bit for bit.
Full run directories (config.json, model checkpoints, rsl_rl git state) go to `.run/golden_runs/` and are not committed; the
checkpoints are what the Phase 4 resume tests use, so regenerate them with the command above when needed.

Environments (see the module docstring for the physics): `cartpole` (continuous force, early termination, 500-step time-out, max return
500), `pendulum` (gym swing-up, 200 steps, no termination, random policy -1224, a solved policy about -150),
`tracking12` (12 double-integrator actuators tracking a position command, critic-only privileged randomisation parameters, 300-step
time-out and early termination, `extras["log"]` episode statistics).

Results (final reward = mean of `Train/mean_reward` over the last 10 iterations; 3 seeds; torch 2.11.0+cpu, 8 threads per run):

| env | envs x steps | iterations | final reward (mean +- std over seeds) | wall time per run | random policy |
|-----|--------------|------------|----------------------------------------|-------------------|---------------|
| cartpole | 1024 x 24 | 100 | 500.0 +- 0.0 (max 500) | ~20 s | 25.5 |
| pendulum | 1024 x 32 | 200 | -152.1 +- 2.4 | ~50 s | -1224 |
| tracking12 | 2048 x 24 | 300 | 190.7 +- 1.4 | ~230 s | 14.9 |

Phase 4 acceptance thresholds derived from these (`rsl_rl_tt.testing.golden.phase4_threshold`): cartpole 500 - max(0, 25) = 475,
pendulum -152.1 - max(2.4, 7.6) = -159.7, tracking12 190.7 - max(1.4, 9.5) = 181.2.
