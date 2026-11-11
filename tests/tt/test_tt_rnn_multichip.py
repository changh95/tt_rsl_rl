# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.5 device tests on a multi-chip mesh: recurrent models under the data-parallel semantics of ADR-0004. With ONE mini-batch the chips' mini-batches (their own environments, whole trajectories) are
together exactly upstream's recurrent mini-batch (all environments), so an update on N chips must equal upstream's update on the same data (function space, like the single-chip tests: the gradients are the
mean over the chips of the chips' means = the mean over all samples); replicas stay bit-identical; traced == eager.

    scripts/tt_run.sh --chips 16,17 --timeout 900 --tag p8nm -- env TT_TEST_MESH=1x2 python -m pytest -m tt tests/tt/test_tt_rnn_multichip.py -q -p no:cacheprovider
    scripts/tt_run.sh --timeout 900 --tag p8nm -- env TT_TEST_MESH=2x4 python -m pytest -m tt tests/tt/test_tt_rnn_multichip.py -q -p no:cacheprovider
"""

import pytest
import torch

from rsl_rl_tt.testing.parity import TOLERANCES, UPDATE_TOLERANCES, compare
from rsl_rl_tt.testing.reference import test_mesh_shape
from tests.tt.test_tt_rnn_update import rnn_spec, run_pair
from tests.tt.test_tt_update import SMOOTH

_ROWS, _COLS = test_mesh_shape()
pytestmark = [pytest.mark.tt, pytest.mark.skipif(_ROWS * _COLS == 1, reason="needs a multi-chip mesh (TT_TEST_MESH=1x2 or 2x4)")]
N = _ROWS * _COLS


@pytest.mark.parametrize("case", [dict(rnn_type="gru", layers=1), dict(rnn_type="lstm", layers=2)], ids=["gru", "lstm-2-layers"])
def test_update_on_several_chips_matches_upstream_with_one_mini_batch(tt_mesh, case, tmp_path):
    over = dict(num_learning_epochs=1, num_mini_batches=1, schedule="fixed", **SMOOTH)
    r = run_pair(tmp_path, rnn_spec(**case, over=over), {})
    # (the logged losses of a chip are chip 0's local means, not the means over all environments: not compared here)
    assert r["alg"].ctx.n_chips == N and r["lr"][0] == pytest.approx(r["lr"][1], rel=1e-6)
    for tag in ("actor", "critic"):
        got, noise = r["tt"][tag], r["noisy"][tag]
        limit = UPDATE_TOLERANCES["rnn_noise_factor"] * noise + UPDATE_TOLERANCES["noise_abs"]
        print(f"\n{N} chips vs upstream, {tag}: {got:.3e} of the update differs (noise baseline {noise:.3e}, limit {limit:.3e})")
        assert got <= limit, f"{tag}: {got:.3e} of the update differs (limit {limit:.3e}; noise baseline {noise:.3e})"
    assert r["alg"].replica_max_diff() == 0.0


def build_runner(trace=False, interval=0, seed=0, rnn_type="gru"):
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl_tt.testing.envs import make_env
    from rsl_rl_tt.testing.reference import make_train_cfg

    spec = rnn_spec(rnn_type, 1)
    torch.manual_seed(seed)
    env = make_env("cartpole_po", spec.num_envs, seed=seed)
    cfg = make_train_cfg("cartpole_po", spec)
    cfg["algorithm"]["class_name"] = "rsl_rl_tt.ppo:TTPPO"
    cfg["algorithm"]["tt_cfg"] = {"mesh_shape": [_ROWS, _COLS], "trace": trace, "replica_check_interval": interval}
    return OnPolicyRunner(env, cfg, log_dir=None, device="cpu"), env


@pytest.mark.parametrize("trace", [False, True])
def test_replicas_stay_bit_identical_over_training(tt_mesh, trace):
    """The parameters (incl. the RNN's), Adam state and normalizers of every chip are identical after every update (the hidden states are per environment, hence per chip)."""
    runner, _ = build_runner(trace=trace, interval=1)
    try:
        runner.learn(num_learning_iterations=3, init_at_random_ep_len=True)  # raises RuntimeError by itself if the replicas diverge
        assert runner.alg.replica_max_diff() == 0.0
    finally:
        runner.alg.release_traces()


def test_traced_training_is_bit_identical_to_eager_on_several_chips(tt_mesh):
    eager, _ = build_runner(trace=False, rnn_type="lstm")
    traced, _ = build_runner(trace=True, rnn_type="lstm")
    from tests.tt.test_tt_rnn_trace import full_state
    from tests.tt.test_tt_trace import assert_state_equal

    try:
        torch.manual_seed(4242)
        eager.learn(num_learning_iterations=3, init_at_random_ep_len=True)
        torch.manual_seed(4242)
        traced.learn(num_learning_iterations=3, init_at_random_ep_len=True)
        assert {"act", "gae", "update.step.0"} <= set(traced.alg._traces)
        assert_state_equal(full_state(eager.alg), full_state(traced.alg), "eager vs traced on several chips (recurrent)")
        assert traced.alg.replica_max_diff() == 0.0
    finally:
        traced.alg.release_traces()
