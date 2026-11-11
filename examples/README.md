# Examples

Small, readable end-to-end examples of the Tenstorrent backend (`rsl_rl_tt`). Each one is a short script on top of the public rsl_rl API (`OnPolicyRunner` + a training config); the
TT backend is switched on by one function (`rsl_rl_tt.bridge.config.tt_agent_cfg`). Device jobs go only through `scripts/tt_run.sh` (see `CLAUDE.md`).

| Example | What it shows |
|---|---|
| [`g1_walk/`](g1_walk/README.md) | A Unitree G1 learns to walk on command: MuJoCo simulates 4096 robots on the CPU, PPO trains on all eight chips of the tray (28 min); `play.py` replays checkpoints and records the video |

The longer-running, documented demos behind them live in `docs/porting/` (`demo_g1/`, `demo_humanoid/`); the examples use the same library code in `rsl_rl_tt/demos/`.
