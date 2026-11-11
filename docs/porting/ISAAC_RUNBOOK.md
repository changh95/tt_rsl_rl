# Isaac Lab on a GPU host, training on the Tenstorrent tray: runbook (Phase 7)

```
 GPU host (Isaac Lab python)                          TT host (this repository, tray 3)
 ┌──────────────────────────────┐   ZeroMQ, tcp      ┌────────────────────────────────────────────────┐
 │ scripts/isaac_env_server.py  │ <----------------> │ scripts/train_isaac.py                         │
 │   RslRlVecEnvWrapper(env)    │  actions  ->       │   unmodified OnPolicyRunner                    │
 │   serve(env, endpoint)       │  <- obs, rewards,  │   rsl_rl_tt.ppo:TTPPO on 1 / 2 / 8 chips       │
 │   (+ agent_cfg.to_dict())    │     dones, extras  │   RemoteVecEnv (rsl_rl VecEnv, CPU tensors)    │
 └──────────────────────────────┘                    └────────────────────────────────────────────────┘
```

The environment stays on the GPU host; policy inference, rollout storage, GAE and the PPO update run on the tray. One environment step crosses the network once (actions
down, observations up); nothing else does. The agent configuration is taken from the GPU host (the server sends `agent_cfg.to_dict()`), so both sides run the same experiment.

## What is verified here, and what is not

| Verified on the TT host (tests and recorded runs) | NOT verified here (no Isaac Lab, no GPU on this host) |
|---|---|
| wire format, error path, timeout, `episode_length_buf`, `info` (`tests/tt/test_bridge.py`, 14 host tests) | `scripts/isaac_env_server.py --task ...`: written against the public Isaac Lab v2.3.0 `train.py` and `RslRlVecEnvWrapper` (BSD-3-Clause), never run |
| the environment behind the bridge is indistinguishable from the same environment in process: step by step (obs, rewards, dones, time-outs, episode logs), and in training (upstream PPO: every logged scalar; TTPPO on a chip: every scalar and every parameter) | whether your Isaac Lab version trains with the rsl_rl 5.x configuration format (`actor` / `critic` / `algorithm` / `obs_groups`) that the TT backend needs |
| the user-facing scripts end to end on loopback: tracking12, 300 iterations, TTPPO on one chip, configuration received from the server: identical to the in-process run in every logged scalar, golden criterion met (`docs/porting/isaac_loopback/`) | Isaac-Cartpole and the locomotion task: curves against the GPU reference, `play.py` replay of a TT checkpoint |
| round trip and bandwidth on loopback (`docs/porting/PERF.md`, "Environment bridge") | latency and bandwidth of YOUR network |

If anything in the Isaac mode of the server does not match your Isaac Lab version, keep these lines in your own launcher (everything else in the script is Isaac Lab boilerplate):

```python
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)           # as in Isaac Lab's train.py
from rsl_rl_tt.bridge.server import serve
serve(env, "tcp://*:5555", agent_cfg=agent_cfg.to_dict())                    # blocks until the TT host closes the connection
```

## 0. Versions

* The TT backend is pinned to **rsl_rl v5.5.2** (`rsl_rl/` in this repository is byte-identical to that tag). The configuration the GPU host sends must be in the rsl_rl 5.x format. Isaac Lab
  v2.3.0's `train.py` still uses the older `policy:` format (rsl_rl 3.x); the Isaac Lab `main` branch uses `actor` / `critic` (`RslRlMLPModelCfg`). If the received configuration has no `actor` /
  `critic` / `obs_groups`, `train_isaac.py` stops with a message saying so. Install `rsl-rl-lib==5.5.2` in the Isaac Lab Python for the GPU reference run and for `play.py`, so that checkpoints and
  configurations mean the same on both sides.
* The two Pythons may differ (the bridge carries JSON + raw float32 buffers, no pickle, no torch types). The server needs `numpy`, `pyzmq`, `torch` (all present in an Isaac Lab Python except possibly
  `pyzmq`: `pip install pyzmq`).

## 1. GPU host: get the code, start a reference run

```bash
git clone https://github.com/changh95/tt_rsl_rl && cd tt_rsl_rl          # only rsl_rl_tt/bridge/{protocol,server}.py and scripts/isaac_env_server.py are used there
# GPU reference run (3 seeds), exactly as you would train anyway; the configuration is saved in <run>/params/agent.yaml
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --headless --num_envs 4096 --seed 0 --max_iterations 150
python scripts/tb_to_csv.py logs/rsl_rl/<experiment>/<run> --out gpu_cartpole_seed0.csv      # TensorBoard -> the CSV layout used here (repeat for every seed)
```

(The location of Isaac Lab's training script differs between versions; use the one of your installation.)

## 2. GPU host: serve the environment

```bash
./isaaclab.sh -p /path/to/tt_rsl_rl/scripts/isaac_env_server.py --task Isaac-Cartpole-v0 --num_envs 4096 --headless --seed 0 --endpoint tcp://*:5555 --dump-agent-cfg agent_served.yaml
```

The server resets the environment (the wrapper does), then answers requests until the TT host closes the connection. One server process serves one training run; start it again for the next seed.
Open TCP port 5555 from the TT host. Use `--endpoint tcp://127.0.0.1:5555` plus an SSH tunnel if only SSH is allowed.

## 3. TT host: check the connection (no training, no device)

```bash
source .venv/bin/activate
python scripts/bridge_smoke.py --endpoint tcp://GPU_HOST:5555 --steps 50
```

It prints the number of environments / actions, the observation groups, whether the agent configuration was received (and warns if it is not in the 5.x format), checks shapes, dtypes
(`float32` observations, `long` dones) and finiteness over 50 random steps, and reports the median round trip and the bytes per step. Fix what it complains about before training.
(`bridge_smoke.py` leaves the server running; `train_isaac.py` stops it when training ends unless `--keep-server`.)

## 4. TT host: train

```bash
# one chip (chip 16)
scripts/tt_run.sh --chips 16 --timeout 7200 --tag isaac_cartpole -- python scripts/train_isaac.py --endpoint tcp://GPU_HOST:5555 --iterations 150 --log-dir logs/isaac_cartpole_tt_seed0
# the whole tray: num_envs must be a multiple of 8 x 32 = 256 and (num_envs x num_steps_per_env / num_mini_batches) a multiple of 256
scripts/tt_run.sh --timeout 7200 --tag isaac_cartpole8 -- python scripts/train_isaac.py --endpoint tcp://GPU_HOST:5555 --mesh 2x4 --iterations 150 --log-dir logs/isaac_cartpole_tt8_seed0
```

`--agent-cfg params/agent.yaml` replaces the configuration received from the server with a saved one (e.g. the GPU reference run's). Only the algorithm class, its `tt_cfg`, the host-side device
(`cpu`) and the log writer are changed (`rsl_rl_tt/bridge/config.py`; the result is saved in `<log-dir>/agent_cfg_used.json`). `<log-dir>/metrics.csv` has every scalar the upstream logger writes;
`model_<it>.pt` are upstream-format checkpoints. `--seed N` sets the training seed (use the same seed for the server: `--seed` there sets the environment's).

A job that hits its timeout or a device / fabric error is NOT retried or reset (BRIEF 2-6): collect `.run/logs/<tag>*` and report. A silent GPU host shows up as
`TimeoutError: no reply to 'step' ... within N s` (`--timeout`, default 600 s), not as a hang.

## 5. Compare with the GPU reference

```bash
python scripts/compare_curves.py --ref gpu_cartpole_seed0.csv gpu_cartpole_seed1.csv gpu_cartpole_seed2.csv --tt logs/isaac_cartpole_tt_seed0/metrics.csv logs/isaac_cartpole_tt_seed1/metrics.csv \
       --title "Isaac-Cartpole-v0: TT vs GPU" --out docs/porting/isaac_cartpole
```

It applies the Phase 4 criterion (mean over the TT seeds of the final reward >= reference mean - max(reference std, 5 % of |mean|), all values finite), writes `<out>.json` and `<out>.png`, and exits
non-zero if the criterion fails. Expect close curves, not identical ones: the GPU and the TT run use different random number streams, and the multi-chip run uses chip-local advantage
normalisation and shuffling (ADR-0004), like upstream multi-GPU.

## 6. A locomotion task

Same procedure with a flat-terrain walking task, e.g. `Isaac-Velocity-Flat-Anymal-D-v0` or `Isaac-Velocity-Flat-Unitree-Go2-v0` (names as in Isaac Lab 2.x: check with
`./isaaclab.sh -p scripts/environments/list_envs.py`), `--num_envs 4096`, more iterations (the GPU reference run's `max_iterations`). Larger observations cost more per step:

    bytes per step = num_envs x (4 x (sum of observation widths) + 4 x num_actions + 4 + 1 + 1)       e.g. 4096 envs, observations of total width 238, 12 actions: ~4 MB

Measured on loopback (`docs/porting/PERF.md`): 4096 envs of the tracking12 shapes (342 bytes per environment) take 0.9 ms per step, 65536 envs 8.9 ms; on a real network add the latency and divide the
bytes by the bandwidth. An iteration has `num_steps_per_env` steps (24 by default): for the 4096-environment case the bridge costs ~20 ms per iteration on loopback, ~35-60 ms over 10 GbE.
A 1 GbE link would dominate (125 MB/s: 8 ms per MB per step, i.e. ~32 ms per step and ~0.8 s per iteration for the 4 MB example).

## 7. Replay a TT checkpoint in Isaac Lab

```bash
scp tt_host:/path/to/tt_rsl_rl/logs/isaac_cartpole_tt_seed0/model_149.pt .
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Cartpole-v0 --num_envs 32 --checkpoint model_149.pt
```

The file is an upstream-format checkpoint (TT-trained weights, normalizer statistics and optimizer state; Phase 4 tests load it into the unmodified `OnPolicyRunner` and export it to TorchScript /
ONNX). `play.py` builds the runner from the same agent configuration and calls `runner.load(...)`: use the same rsl_rl version on both sides.

## 8. What to send back

For each run: the `compare_curves.py` output (`.json`, `.png`), `<log-dir>/agent_cfg_used.json`, the `bridge_smoke.py` output, the versions (`pip list | grep -i -E "rsl|isaac|torch|zmq"` on both hosts),
and any error text. Failures of the Isaac mode of the server are expected to be small API differences; the traceback of the server is forwarded to the TT host in the error message
(`RemoteEnvError`), so the TT host's log is enough to see them.
