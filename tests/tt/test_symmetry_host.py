# SPDX-License-Identifier: BSD-3-Clause
"""Host-only tests of the symmetry support: the derivation of the affine mirror maps from a user function (`rsl_rl_tt/symmetry.py`), the test symmetry functions, and the upstream
`Symmetry` extension on the CPU test environments (the oracle of the device tests)."""

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.extensions import Symmetry
from rsl_rl_tt.symmetry import MirrorMaps, bf16_exact
from rsl_rl_tt.testing import symmetry as sym_funcs

DIMS = {"policy": 36, "privileged": 48}
ALL = {"policy", "privileged"}


def maps(func, dims=DIMS, used=ALL, actions=12):
    return MirrorMaps(func, None, dims, actions, used)


def test_the_sign_mirror_is_a_diagonal_map_and_the_dihedral_one_has_four_blocks():
    m = maps(sym_funcs.tracking12_sign_mirror)
    assert m.num_aug == 2
    w, b = m.obs["policy"][0]
    assert torch.equal(w, -torch.eye(36)) and float(b.abs().max()) == 0.0
    w, b = m.obs["privileged"][0]
    assert torch.equal(torch.diagonal(w), torch.cat([torch.ones(36), -torch.ones(12)])) and torch.count_nonzero(w) == 48
    d = maps(sym_funcs.tracking12_dihedral_mirror)
    assert d.num_aug == 4 and len(d.obs["policy"]) == 3 and len(d.actions) == 3
    # the swap: a permutation matrix (every column holds exactly one +1), the composition one with signs
    w, _ = d.obs["policy"][1]
    assert bf16_exact(w) and torch.equal(w.sum(0), torch.ones(36)) and torch.equal(w.sum(1), torch.ones(36)) and not torch.equal(w, torch.eye(36))
    w, _ = d.actions[2]
    assert torch.equal(w.abs().sum(0), torch.ones(12)) and float(w.sum()) == -12.0


def test_the_derived_maps_reproduce_the_function_on_random_inputs():
    g = torch.Generator().manual_seed(3)
    for func in (sym_funcs.tracking12_dihedral_mirror, sym_funcs.tracking12_affine_mirror):
        m = maps(func)
        obs = TensorDict({k: torch.randn(40, d, generator=g) for k, d in DIMS.items()}, batch_size=[40])
        act = torch.randn(40, 12, generator=g)
        aug_obs, aug_act = func(None, obs, act)
        for k in range(1, m.num_aug):
            for name in DIMS:
                w, b = m.obs[name][k - 1]
                assert torch.allclose(aug_obs[name][40 * k : 40 * (k + 1)], obs[name] @ w + b, atol=1e-6)
            w, b = m.actions[k - 1]
            assert torch.allclose(aug_act[40 * k : 40 * (k + 1)], act @ w + b, atol=1e-6)
    affine = maps(sym_funcs.tracking12_affine_mirror)
    assert float(affine.obs["policy"][0][1].abs().min()) == pytest.approx(0.25) and not bf16_exact(torch.tensor([[0.3]]))


@pytest.mark.parametrize("func, why", [
    (sym_funcs.tracking12_nonaffine_mirror, "not affine"),
    (sym_funcs.tracking12_mixing_mirror, "depends on another group"),
    (sym_funcs.tracking12_identity_first_block_wrong, "must be the original"),
])
def test_functions_that_are_not_supported_are_refused_with_a_reason(func, why):
    with pytest.raises(NotImplementedError, match=why):
        maps(func)


def test_only_the_groups_of_the_actor_and_critic_are_checked():
    """A group the algorithm never reads may be mirrored in any way (here: not at all affinely); the used groups are still derived."""
    def func(env=None, obs=None, actions=None):
        o, a = sym_funcs.tracking12_sign_mirror(env, obs, actions)
        if o is not None:
            o["privileged"][o.batch_size[0] // 2 :] = o["privileged"][o.batch_size[0] // 2 :] ** 2  # the unused group is mirrored non-affinely
        return o, a

    m = maps(func, used={"policy"})
    assert set(m.obs) == {"policy"} and torch.equal(m.obs["policy"][0][0], -torch.eye(36))
    with pytest.raises(NotImplementedError, match="not affine"):
        maps(func, used=ALL)


def test_compose_gives_the_block_diagonal_map_of_a_group_list():
    m = maps(sym_funcs.tracking12_sign_mirror)
    w, b = m.compose(["policy", "privileged"], 1)
    assert w.shape == (84, 84) and torch.equal(w[:36, :36], -torch.eye(36)) and torch.count_nonzero(w[:36, 36:]) == 0 and b.shape == (84,)
    w2, _ = m.compose(["privileged", "policy"], 1)  # another order = another concatenation
    assert torch.equal(torch.diagonal(w2)[:12], torch.ones(12)) and torch.equal(torch.diagonal(w2)[36:], -torch.ones(48))


@pytest.mark.parametrize("name, func", [("cartpole", sym_funcs.cartpole_mirror), ("pendulum", sym_funcs.pendulum_mirror)])
def test_the_test_functions_are_true_symmetries_of_the_dynamics(name, func):
    """Step an env from a state and from its mirrored state with the mirrored action: the observation, reward and done of the second are the mirror of the first."""
    from rsl_rl_tt.testing.envs import make_env

    env_a, env_b = make_env(name, 64, seed=1), make_env(name, 64, seed=1)
    obs = env_a.get_observations()
    mirrored, _ = func(None, obs, None)
    n = 64
    env_b._s = env_a._s.clone() * (-1.0 if name == "cartpole" else torch.tensor([-1.0, -1.0]))  # state of the mirrored system (pendulum state [theta, thdot])
    act = torch.randn(64, env_b.num_actions).clamp(-1, 1)
    obs_a, r_a, d_a, _ = env_a.step(act)
    obs_b, r_b, d_b, _ = env_b.step(-act)
    want, _ = func(None, obs_a, None)
    assert torch.allclose(obs_b["policy"], want["policy"][n:], atol=1e-5)
    assert torch.allclose(r_a, r_b, atol=1e-5) and torch.equal(d_a, d_b)


def test_the_upstream_extension_accepts_the_test_functions_and_returns_the_documented_layout():
    sym = Symmetry(env=None, data_augmentation_func="rsl_rl_tt.testing.symmetry:tracking12_dihedral_mirror", use_data_augmentation=True, use_mirror_loss=True, mirror_loss_coeff=0.5)
    obs = TensorDict({"policy": torch.randn(8, 36), "privileged": torch.randn(8, 48)}, batch_size=[8])
    o, a = sym.data_augmentation_func(env=None, obs=obs, actions=torch.randn(8, 12))
    assert o.batch_size[0] == 32 and a.shape == (32, 12) and torch.equal(o["policy"][:8], obs["policy"])
