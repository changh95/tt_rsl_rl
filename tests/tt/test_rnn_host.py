# SPDX-License-Identifier: BSD-3-Clause
"""Host-only test of the central design claim of the recurrent port (docs/porting/POST_V1.md, 8.5): upstream's split-at-dones-and-pad mini-batches are EXACTLY a scan over the T steps of every environment
with the hidden state zeroed after each done step and started from the state saved at the start of the rollout. Checked on the upstream recurrent PPO itself (real rollouts with dones, GRU and LSTM, 1 and
2 layers): the actor's / critic's batch-mode outputs and the gradients of a loss agree with the scan, mini-batch by mini-batch."""

import dataclasses
import torch

import pytest

from rsl_rl.runners import OnPolicyRunner
from rsl_rl_tt.testing.envs import make_env
from rsl_rl_tt.testing.reference import REF_SPECS, make_train_cfg, with_rnn

T = 24


@pytest.mark.parametrize("rnn_type, layers", [("gru", 1), ("lstm", 1), ("gru", 2), ("lstm", 2)])
def test_the_scan_with_resets_equals_upstream_trajectory_batches(rnn_type, layers):
    spec = with_rnn(dataclasses.replace(REF_SPECS["cartpole_po"], num_envs=64, num_steps=T, iterations=1), rnn_type, 16, layers)
    torch.manual_seed(0)
    env = make_env("cartpole_po", 64, seed=0)
    env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    cfg = make_train_cfg("cartpole_po", spec, logger=None)
    cfg["algorithm"]["num_mini_batches"] = 2
    runner = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")
    alg = runner.alg
    alg.train_mode()
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(T):
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
        alg.compute_returns(obs)
    st = alg.storage
    assert int(st.dones.sum()) > 5, "the test needs trajectories that end inside the rollout"
    actor = alg._raw_actor
    nmb = 2
    for k, batch in enumerate(st.recurrent_mini_batch_generator(nmb, 1)):
        lo, hi = k * 64 // nmb, (k + 1) * 64 // nmb
        # upstream: trajectories split at the dones, padded, masked
        up = actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=False)  # [T, E, A]
        # the scan: one sequence per environment, the state of step 0 saved at the start of the rollout, zeroed after every done step
        h0 = tuple(h[0, :, lo:hi] for h in st.saved_hidden_state_a) if len(st.saved_hidden_state_a) > 1 else st.saved_hidden_state_a[0][0, :, lo:hi]
        state = tuple(h.clone() for h in h0) if isinstance(h0, tuple) else h0.clone()
        outs = []
        for t in range(T):
            x = actor.obs_normalizer(torch.cat([st.observations[g][t, lo:hi] for g in actor.obs_groups], dim=-1))
            out, state = actor.rnn.rnn(x.unsqueeze(0), state)
            outs.append(actor.mlp(out[0]))
            keep = 1.0 - st.dones[t, lo:hi].float()
            state = tuple(s * keep.reshape(1, -1, 1) for s in state) if isinstance(state, tuple) else state * keep.reshape(1, -1, 1)
        mine = torch.stack(outs)
        assert torch.allclose(up.detach(), mine.detach(), atol=1e-5, rtol=1e-5), f"mini-batch {k}: the scan differs from upstream's trajectory batch by {float((up - mine).abs().max()):.2e}"
        # the gradients of a (nonlinear) loss agree too: BPTT through the trajectories vs through the scan
        proj = torch.randn(1, 1, up.shape[-1], generator=torch.Generator().manual_seed(k))
        actor.zero_grad()
        (up * proj).pow(2).sum().backward()
        g_up = {n: p.grad.clone() for n, p in actor.named_parameters() if p.grad is not None}
        actor.zero_grad()
        (mine * proj).pow(2).sum().backward()
        for n, p in actor.named_parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad, g_up[n], atol=1e-5, rtol=1e-4), f"mini-batch {k}: gradient of {n} differs"
