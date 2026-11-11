# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.4 device tests: the traced distillation (one trace per time index that adds its gradients to persistent accumulators, one trace for the apply, one for the act step) is
BIT-identical to the eager path, and `load()` keeps the captured traces valid.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8dt -- python -m pytest -m tt tests/tt/test_tt_distillation_trace.py -s -q -p no:cacheprovider

Like `test_tt_trace.py` this file opens its own mesh with a trace region (run it on its own); also runs on 1x2 / 2x4 with `TT_TEST_MESH`.
"""

import dataclasses
import torch

import pytest

from rsl_rl_tt.testing.reference import REF_SPECS, build_distillation_runner
from tests.tt.test_tt_distillation import STEPS, teachers, tt_student_params  # noqa: F401  (teachers is a fixture)
from tests.tt.test_tt_trace import N_CHIPS, traced_mesh  # noqa: F401  (traced_mesh is a fixture)

pytestmark = pytest.mark.tt

ALGORITHM = dict(num_learning_epochs=2, gradient_length=6, loss_type="mse", max_grad_norm=1.0)  # 16 steps: two windows of 6 and four leftover steps


def build(name, tmp_path, teachers, trace, seed=0, algorithm=None):
    spec = dataclasses.replace(REF_SPECS[name], num_envs=max(128, 32 * N_CHIPS), num_steps=STEPS, iterations=1)
    return build_distillation_runner(name, seed, str(tmp_path), teachers[name], tt=True, tt_cfg={"trace": trace}, algorithm=algorithm or ALGORITHM, spec=spec)


def host_state(alg):
    ctx = alg.ctx
    down = lambda t: ctx.download(t, op="H-6", batch_sharded=False)
    out = {f"param.{k}": v for k, v in tt_student_params(alg).items()}
    for k, (m, v) in enumerate(zip(alg.adam.m, alg.adam.v)):
        out[f"adam.m.{k}"], out[f"adam.v.{k}"] = down(m), down(v)
    out["adam.t"] = torch.tensor(float(alg.adam.t))
    stats = alg.tt_student.normalizer.host_stats()
    out["student.mean"], out["student.var"], out["student.count"] = stats["mean"], stats["var"], torch.tensor(float(stats["count"]))
    for k, g in enumerate(alg._acc_grads):
        out[f"acc.{k}"] = down(g)  # the leftover steps stay in the accumulators until the next update zeroes them
    return out


def assert_equal(sa, sb, what):
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"{what}: {k} differs (max abs diff {float((sa[k].double() - sb[k].double()).abs().max()):.3e})"


@pytest.mark.parametrize("name", ["cartpole", "tracking12"])
def test_traced_distillation_is_bit_identical_to_eager(traced_mesh, name, teachers, tmp_path):  # noqa: F811
    eager, _, _ = build(name, tmp_path / "eager", teachers, trace=False)
    traced, _, _ = build(name, tmp_path / "traced", teachers, trace=True)
    try:
        torch.manual_seed(4242)
        eager.learn(num_learning_iterations=4)
        torch.manual_seed(4242)
        traced.learn(num_learning_iterations=4)
        assert set(traced.alg._traces) >= {"act", "distill.apply", "distill.step.0", f"distill.step.{STEPS - 1}"}, "the traces were not captured"
        ea, ta = (tmp_path / "eager" / "metrics.csv").read_text().splitlines(), (tmp_path / "traced" / "metrics.csv").read_text().splitlines()
        cols = ea[0].split(",")
        assert cols == ta[0].split(",") and "Loss/behavior" in cols
        differ, checked = {}, 0
        for col in ("Loss/behavior", "Policy/mean_std", "Train/mean_reward"):
            if col not in cols:
                continue
            k = cols.index(col)
            va, vb = [r.split(",")[k] for r in ea[1:]], [r.split(",")[k] for r in ta[1:]]
            if va != vb:
                differ[col] = (va, vb)
            checked += 1
        assert not differ, "logged scalars differ between eager and traced: " + "; ".join(f"{c}: eager {a} traced {b}" for c, (a, b) in differ.items())
        assert checked >= 2
        assert_equal(host_state(eager.alg), host_state(traced.alg), "eager vs traced")
    finally:
        traced.alg.release_traces()


def test_traced_path_survives_load(traced_mesh, teachers, tmp_path):  # noqa: F811
    """`load()` rewrites the student, the teacher, the Adam moments IN PLACE: b follows a bit for bit after loading a's distillation checkpoint."""
    a, env_a, _ = build("cartpole", tmp_path / "a", teachers, trace=True, seed=1)
    b, env_b, _ = build("cartpole", tmp_path / "b", teachers, trace=True, seed=2)
    try:
        a.learn(num_learning_iterations=2)
        ckpt = tmp_path / "a.pt"
        a.save(str(ckpt))
        b.learn(num_learning_iterations=1)
        trace_ids = dict(b.alg._traces)
        assert not torch.equal(tt_student_params(a.alg)["mlp.0.weight"], tt_student_params(b.alg)["mlp.0.weight"])
        b.load(str(ckpt), map_location="cpu")
        assert b.alg._traces == trace_ids
        sa, sb = host_state(a.alg), host_state(b.alg)
        for k in sa:
            if not k.startswith("acc."):  # the accumulators are update-local state (zeroed at the start of every update), not part of a checkpoint
                assert torch.equal(sa[k], sb[k]), f"after load: {k} differs"
        # the same data into both, one iteration each: identical afterwards
        torch.manual_seed(9)
        obs = env_a.get_observations()
        recorded = []
        with torch.inference_mode():
            a.alg.train_mode()
            for t in range(STEPS):
                torch.manual_seed(100 + t)
                actions = a.alg.act(obs)
                nxt, rewards, dones, extras = env_a.step(actions)
                a.alg.process_env_step(nxt, rewards, dones, extras)
                recorded.append((obs.clone(), nxt.clone(), rewards.clone(), dones.clone(), {k: v.clone() for k, v in extras.items() if torch.is_tensor(v)}, actions.clone()))
                obs = nxt
        la = a.alg.update()
        b.alg.train_mode()
        with torch.inference_mode():
            for t, (o, nxt, rewards, dones, extras, actions) in enumerate(recorded):
                torch.manual_seed(100 + t)
                assert torch.equal(b.alg.act(o), actions)
                b.alg.process_env_step(nxt, rewards, dones, extras)
        lb = b.alg.update()
        assert la == lb, f"losses differ after load: {la} vs {lb}"
        sa, sb = host_state(a.alg), host_state(b.alg)
        assert_equal(sa, sb, "after one more update")
    finally:
        a.alg.release_traces()
        b.alg.release_traces()
