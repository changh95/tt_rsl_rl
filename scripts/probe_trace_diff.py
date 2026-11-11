# SPDX-License-Identifier: BSD-3-Clause
"""Phase 5 diagnosis probe: where does the traced path first differ from the eager path?

Three runners are built from the same seed in ONE process (one mesh, opened with a trace region): all eager, traced ACT only (the update runs
eagerly) and traced UPDATE only (the rollout runs eagerly). Each runs one iteration with the same global RNG stream and the same environment seed;
the stored rollout (per step) and the parameters / losses after the update are compared against the eager runner.

    scripts/tt_run.sh --chips 16 --timeout 300 --tag probe_trace -- python scripts/probe_trace_diff.py [--iters 2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl_tt.device import open_tray_mesh  # noqa: E402
from rsl_rl_tt.testing.envs import make_env  # noqa: E402
from rsl_rl_tt.testing.reference import make_train_cfg  # noqa: E402
from tests.tt.test_tt_rollout import small_spec  # noqa: E402
from tests.tt.test_tt_update import tt_params  # noqa: E402

FIELDS = ("actor_obs", "critic_obs", "action", "mean", "value", "logp", "reward", "done")


def build(name: str, trace: bool, seed: int = 0):
    spec = small_spec(name)
    torch.manual_seed(seed)
    env = make_env(name, spec.num_envs, seed=seed)
    cfg = make_train_cfg(name, spec)
    cfg["algorithm"]["class_name"] = "rsl_rl_tt.ppo:TTPPO"
    cfg["algorithm"]["tt_cfg"] = {"mesh_shape": [1, 1], "trace": trace}
    return OnPolicyRunner(env, cfg, log_dir=None, device="cpu"), env, spec


def run_iteration(alg, env, spec, trace_act: bool, trace_update: bool, rng_seed: int) -> dict:
    """One rollout + update; returns the stored steps (host), the losses and the parameters."""
    ctx = alg.ctx
    torch.manual_seed(rng_seed)
    alg.train_mode()
    obs = env.get_observations()
    now: list = []
    alg.trace_enabled = trace_act
    with torch.inference_mode():
        for _ in range(spec.num_steps):
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
            now.append({k: ctx.download(alg.storage._steps[-1][k], op="H-6", batch_sharded=True) for k in FIELDS})  # right after the step was stored
        steps = [{k: ctx.download(s[k], op="H-6", batch_sharded=True) for k in FIELDS} for s in alg.storage._steps]
        alg.compute_returns(obs)
    alg.trace_enabled = trace_update
    losses = alg.update()
    return {"now": now, "steps": steps, "losses": losses, "params": tt_params(alg), "lr": alg.learning_rate}


def compare(ref: dict, other: dict, label: str) -> bool:
    ok = True
    # (1) the values right after each step was stored = the act computation itself; (2) the same tensors at the end of the rollout: were they overwritten?
    bad_now = [(i, k) for i, (a, b) in enumerate(zip(ref["now"], other["now"])) for k in FIELDS if not torch.equal(a[k], b[k])]
    print(f"  [{label}] values right after each step vs eager: {'IDENTICAL' if not bad_now else f'{len(bad_now)} differences, first {bad_now[0]}'}")
    changed = [(i, k) for i, (a, b) in enumerate(zip(other["now"], other["steps"])) for k in FIELDS if not torch.equal(a[k], b[k])]
    print(f"  [{label}] tensors that changed between insertion and the end of the rollout: {len(changed)} {changed[:4]}")
    first = None
    for i, (a, b) in enumerate(zip(ref["steps"], other["steps"])):
        for k in FIELDS:
            x, y = a[k], b[k]
            if x.shape != y.shape or not torch.equal(x, y):
                d = float((x.double() - y.double()).abs().max()) if x.shape == y.shape else float("nan")
                first = first or (i, k, d)
                print(f"  [{label}] step {i:2d} {k:10s} differs, max abs diff {d:.3e}")
                ok = False
    print(f"  [{label}] rollout: {'IDENTICAL' if first is None else f'first difference at step {first[0]} field {first[1]} ({first[2]:.3e})'}")
    for k in ref["losses"]:
        eq = ref["losses"][k] == other["losses"][k]
        print(f"  [{label}] loss {k:10s} eager {ref['losses'][k]:.9g} other {other['losses'][k]:.9g} {'equal' if eq else 'DIFFERENT'}")
        ok &= eq
    worst = max(float((ref["params"][k] - other["params"][k]).abs().max()) for k in ref["params"])
    same = all(torch.equal(ref["params"][k], other["params"][k]) for k in ref["params"])
    print(f"  [{label}] parameters: {'IDENTICAL' if same else f'DIFFERENT (max abs diff {worst:.3e})'}; lr {ref['lr']} vs {other['lr']}")
    return ok and same


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cartpole")
    ap.add_argument("--iters", type=int, default=2)
    args = ap.parse_args()
    tm = open_tray_mesh((1, 1), trace_region_size=128 * 1024 * 1024)
    runners = {}
    try:
        for label in ("eager", "act", "update"):
            runners[label] = build(args.env, trace=label != "eager")
        modes = {"eager": (False, False), "act": (True, False), "update": (False, True)}
        results = {label: [] for label in runners}
        # run the runners iteration by iteration with the same RNG seeds
        for it in range(args.iters):
            for label, (alg_runner, env, spec) in runners.items():
                results[label].append(run_iteration(alg_runner.alg, env, spec, *modes[label], rng_seed=1000 + it))
        for it in range(args.iters):
            print(f"iteration {it}:")
            for label in ("act", "update"):
                compare(results["eager"][it], results[label][it], f"eager vs traced-{label}")
    finally:
        for runner, _, _ in runners.values():
            runner.alg.release_traces()
        tm.close()


if __name__ == "__main__":
    main()
