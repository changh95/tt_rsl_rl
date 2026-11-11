# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests for the heteroscedastic Gaussian (Phase 8.1): the multi-head padding layout and the fixture of the device tests."""

import pytest
import torch

from rsl_rl_tt.tt_common import linear_from_host, linear_to_host, pad_heads, pad_matrix, pad_up, unpad_heads


def test_one_head_is_plain_padding():
    m = torch.randn(7, 50)
    assert torch.equal(pad_heads(m, 32, 64, 1), pad_matrix(m, 32, 64))
    assert torch.equal(unpad_heads(pad_heads(m, 32, 64, 1), 7, 50, 1), m)


@pytest.mark.parametrize("d", [1, 5, 12, 32, 33, 64])
def test_two_heads_are_tile_aligned_and_round_trip(d):
    m = torch.randn(9, 2 * d)
    d_p = pad_up(d)
    out = pad_heads(m, 32, 2 * d_p, 2)
    assert out.shape == (32, 2 * d_p)
    assert torch.equal(out[:9, :d], m[:, :d]) and torch.equal(out[:9, d_p : d_p + d], m[:, d:])  # mean head | std head, each starting on a tile boundary
    assert torch.count_nonzero(out[:, d:d_p]) == 0 and torch.count_nonzero(out[:, d_p + d :]) == 0 and torch.count_nonzero(out[9:]) == 0
    assert torch.equal(unpad_heads(out, 9, 2 * d, 2), m)


@pytest.mark.parametrize("d", [1, 12, 33])
def test_a_two_head_linear_converts_to_the_device_layout_and_back(d):
    lin = torch.nn.Linear(17, 2 * d)
    w, b = linear_to_host(lin.weight, lin.bias, pad_up(17), 2 * pad_up(d), heads=2)
    assert w.shape == (pad_up(17), 2 * pad_up(d)) and b.shape == (1, 2 * pad_up(d))
    # the device computes x @ w + b: the first d_p columns are the mean head, the next d_p the std head
    x = torch.randn(5, 17)
    y = pad_matrix(x, 5, pad_up(17)) @ w + b
    expect = lin(x)
    assert torch.allclose(y[:, :d], expect[:, :d], atol=1e-6) and torch.allclose(y[:, pad_up(d) : pad_up(d) + d], expect[:, d:], atol=1e-6)
    w2, b2 = linear_from_host(w, b, 2 * d, 17, heads=2)
    assert torch.equal(w2, lin.weight.detach()) and torch.equal(b2, lin.bias.detach())


def test_the_randomised_std_head_makes_the_std_state_dependent_and_clamps():
    """The fixture of the device tests: with the randomised std head the std differs from row to row, and for the narrow range part of it is clamped."""
    pytest.importorskip("tensordict")
    from tests.tt.test_tt_hetero_update import make_hetero
    from tests.tt.test_tt_modules import make_obs

    actor, _ = make_hetero(5)
    obs = make_obs(512, {"policy": 36}, seed=8, scale=1.5, offset=0.4)
    actor(obs, stochastic_output=True)
    std = actor.output_std
    assert std.std(dim=0).min() > 1e-3
    assert ((std <= 0.55 + 1e-6) | (std >= 1.1 - 1e-6)).float().mean() > 0.05
