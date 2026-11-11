# SPDX-License-Identifier: BSD-3-Clause
"""Symmetry functions (`symmetry_cfg["data_augmentation_func"]`) of the CPU test environments, in the format upstream expects: `func(env, obs, actions)` where either argument may be None;
the result is `(obs_aug, actions_aug)` with the original rows first, then every mirrored copy (`[R * K, ...]`); a missing argument comes back as None. Host only; used by the tests, the
reference runs and the end-to-end runs of Phase 8.3 (`rsl_rl_tt.testing.symmetry:<name>` is how a config names them).

The functions are TRUE symmetries of the test environments, so that training with them still learns:

  cartpole     x -> -x on every observation, force -> -force                        (K = 2)
  pendulum     [cos, sin, thdot] -> [cos, -sin, -thdot], torque -> -torque           (K = 2)
  tracking12   sign flip: q, qd, q_cmd, the action and the bias of every joint change sign, mass / damping / gain do not (K = 2); the dihedral version adds the swap of the
               joints j <-> j + 6 and their composition (K = 4: original, flip, swap, flip o swap)

plus functions that the TT port must REFUSE (not affine, mixing two observation groups) and an affine one with a constant offset, for the tests of the derivation of the mirror maps.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

N = 12  # actuators of tracking12


def _assemble(obs, actions, obs_fns, act_fns):
    """The generic wrapper: `obs_fns[k]` maps a dict of group tensors to the mirrored dict of copy k + 1, `act_fns[k]` the actions."""
    new_obs = new_act = None
    if obs is not None:
        groups = {k: v for k, v in obs.items()}
        blocks = [groups] + [fn(groups) for fn in obs_fns]
        new_obs = TensorDict({k: torch.cat([b[k] for b in blocks], dim=0) for k in groups}, batch_size=[obs.batch_size[0] * len(blocks)])
    if actions is not None:
        new_act = torch.cat([actions] + [fn(actions) for fn in act_fns], dim=0)
    return new_obs, new_act


# ---- cartpole / pendulum -------------------------------------------------------------------------------------------------------------------------
def cartpole_mirror(env=None, obs=None, actions=None):
    return _assemble(obs, actions, [lambda g: {k: -v for k, v in g.items()}], [lambda a: -a])


def pendulum_mirror(env=None, obs=None, actions=None):
    flip = torch.tensor([1.0, -1.0, -1.0])
    return _assemble(obs, actions, [lambda g: {k: v * flip for k, v in g.items()}], [lambda a: -a])


# ---- tracking12 ---------------------------------------------------------------------------------------------------------------------------------
def _swap(x):
    """The joints j <-> j + 6 in every block of 12 entries of the last dimension."""
    d = x.shape[-1]
    idx = torch.arange(d).reshape(-1, N)
    idx = torch.cat([idx[:, 6:], idx[:, :6]], dim=1).reshape(-1)
    return x[..., idx]


def _flip_policy(x):
    return -x  # q - q_cmd, qd, last action: all odd


def _flip_privileged(x):
    sign = torch.cat([torch.ones(3 * N), -torch.ones(N)])  # mass, damping, gain even; bias odd
    return x * sign


def tracking12_sign_mirror(env=None, obs=None, actions=None):
    return _assemble(obs, actions, [lambda g: {"policy": _flip_policy(g["policy"]), "privileged": _flip_privileged(g["privileged"])}], [lambda a: -a])


def tracking12_dihedral_mirror(env=None, obs=None, actions=None):
    flip = lambda g: {"policy": _flip_policy(g["policy"]), "privileged": _flip_privileged(g["privileged"])}
    swap = lambda g: {"policy": _swap(g["policy"]), "privileged": _swap(g["privileged"])}
    both = lambda g: flip(swap(g))
    return _assemble(obs, actions, [flip, swap, both], [lambda a: -a, _swap, lambda a: -_swap(a)])


# ---- for the derivation tests ----------------------------------------------------------------------------------------------------------------
def tracking12_affine_mirror(env=None, obs=None, actions=None):
    """Affine, not linear: a constant offset on the mirrored observations and actions (not a symmetry of the env; a test of the bias path)."""
    return _assemble(obs, actions, [lambda g: {k: -v + 0.25 for k, v in g.items()}], [lambda a: -a + 0.5])


def tracking12_nonaffine_mirror(env=None, obs=None, actions=None):
    return _assemble(obs, actions, [lambda g: {k: v**2 for k, v in g.items()}], [lambda a: -a])


def tracking12_mixing_mirror(env=None, obs=None, actions=None):
    """The mirrored `policy` depends on `privileged` too (their first 36 / 48 entries are added): a map between groups."""
    return _assemble(obs, actions, [lambda g: {"policy": -g["policy"] + g["privileged"][..., :36], "privileged": g["privileged"]}], [lambda a: -a])


def tracking12_identity_first_block_wrong(env=None, obs=None, actions=None):
    """The first block is not the original observations."""
    if obs is None:
        return None, (None if actions is None else torch.cat([actions, -actions], dim=0))
    new = TensorDict({k: torch.cat([-v, v], dim=0) for k, v in obs.items()}, batch_size=[obs.batch_size[0] * 2])
    return new, (None if actions is None else torch.cat([actions, -actions], dim=0))
