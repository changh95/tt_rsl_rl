# SPDX-License-Identifier: BSD-3-Clause
"""Symmetry augmentation and mirror loss (upstream `rsl_rl.extensions.Symmetry`, post-v1 item 8.3, docs/porting/POST_V1.md) on the device.

Upstream calls a USER function `data_augmentation_func(env, obs, actions)` on every mini-batch (observations: a TensorDict of all groups; it returns the batch followed by its mirrored
copies: `[R * K, ...]`, the original rows first). A Python callback inside a captured trace is not possible, so the function is evaluated ON THE HOST, ONCE, at construction (allow-list
H-10), on probe inputs, and what it computes is derived from that:

    mirrored_k(x) = x @ W_k + b_k        per observation group and for the actions, for every copy k = 1 .. K - 1

i.e. the function must be AFFINE (every symmetry of a legged robot is a signed permutation of the observation / action entries: a left-right swap with sign flips). The probe is the zero
row plus the unit vectors; the derived maps are VERIFIED on random inputs against the function itself (augmenting observations and actions together, as `augment_batch` does, and separately,
as `compute_loss` does); a function that is not affine, that mixes observation groups, or whose first block is not the original is refused with a clear error (never approximated).

On the device the mirrored copies of the observations and the actions of the whole rollout are computed once per update (`mirror_blocks`, extra columns of the packed table, shuffled together
with the rows they belong to) and the mirrored action means of the mirror loss are computed in the step (`mirror_actions`). A matrix whose entries are exactly representable in bfloat16
(0, +-1, ...: every signed permutation) is applied EXACTLY: the fp32 input is split into three bfloat16 parts (exactly, like the shuffle's gather), each part is multiplied with the bf16
matrix (products of one entry and 0 / +-1 are exact, accumulated in fp32) and the parts are added back; a plain fp32 matmul would round the inputs (rel_fro 4e-4, measured). Any other matrix uses
the fp32 HiFi4 matmul at the matmul noise floor.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from rsl_rl_tt.tt_common import TTContext, pad_matrix, pad_up, pad_vector

PROBE_ATOL = 1e-5  # the derived maps must reproduce the user function on random inputs to this (absolute, inputs ~ N(0, 1))


class MirrorMaps:
    """The affine maps of a user symmetry function, derived by probing it (host, once; allow-list H-10).

    `obs[group][k] = (W [d, d], b [d])` and `actions[k] = (W [A, A], b [A])` for the copies k = 1 .. K - 1 (copy 0 is the original: verified)."""

    def __init__(self, func: Any, env: Any, obs_dims: dict[str, int], num_actions: int, used: set[str]) -> None:
        """`obs_dims`: ALL observation groups (the user function gets the whole TensorDict, like upstream); `used`: the groups of the actor / critic: only their maps are derived and checked."""
        self.num_aug = 0
        self.used = used
        self.obs: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.actions: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._derive(func, env, obs_dims, num_actions)

    # -- derivation -------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _probe_obs(obs_dims: dict[str, int]) -> TensorDict:
        total = sum(obs_dims.values())
        probe = torch.zeros(total + 1, total)
        probe[1:] = torch.eye(total)  # row 0: the zero vector, row 1 + i: the unit vector of entry i (over all groups, concatenated)
        out, col = {}, 0
        for name, d in obs_dims.items():
            out[name] = probe[:, col : col + d].clone()
            col += d
        return TensorDict(out, batch_size=[total + 1])

    def _derive(self, func: Any, env: Any, obs_dims: dict[str, int], num_actions: int) -> None:
        total = sum(obs_dims.values())
        probe = self._probe_obs(obs_dims)
        batch = total + 1
        with torch.no_grad():
            aug_obs, _ = func(env=env, obs=probe, actions=None)
        if aug_obs.batch_size[0] % batch != 0 or aug_obs.batch_size[0] // batch < 2:
            raise NotImplementedError(f"the symmetry function returned {aug_obs.batch_size[0]} observation rows for {batch} probe rows: expected the original rows followed by at least one mirrored copy")
        self.num_aug = aug_obs.batch_size[0] // batch
        offsets, col = {}, 0
        for name, d in obs_dims.items():
            offsets[name] = (col, col + d)
            col += d
        for name, d in obs_dims.items():
            if name not in self.used:
                continue
            self.obs[name] = []
            for k in range(self.num_aug):
                y = aug_obs[name][k * batch : (k + 1) * batch].detach().to(torch.float32)
                if y.shape[1] != d:
                    raise NotImplementedError(f"the symmetry function changes the size of observation group '{name}' ({d} -> {y.shape[1]})")
                b = y[0].clone()
                w = y[1:] - b  # [total, d]: row i = the response to the unit vector i
                lo, hi = offsets[name]
                foreign = torch.cat([w[:lo], w[hi:]])
                if foreign.numel() and float(foreign.abs().max()) > PROBE_ATOL:
                    raise NotImplementedError(f"the mirrored observation group '{name}' depends on another group: only per-group mirror maps are supported")
                if k == 0:
                    if float((w[lo:hi] - torch.eye(d)).abs().max()) > PROBE_ATOL or float(b.abs().max()) > PROBE_ATOL:
                        raise NotImplementedError(f"the first block of the symmetry function's output must be the original observations (group '{name}' is not)")
                else:
                    self.obs[name].append((w[lo:hi].clone(), b))
        # the actions
        probe_a = torch.zeros(num_actions + 1, num_actions)
        probe_a[1:] = torch.eye(num_actions)
        with torch.no_grad():
            _, aug_act = func(env=env, obs=None, actions=probe_a)
        if aug_act is None or aug_act.shape[0] != self.num_aug * (num_actions + 1):
            raise NotImplementedError("the symmetry function must mirror the actions into the same number of copies as the observations")
        for k in range(self.num_aug):
            y = aug_act[k * (num_actions + 1) : (k + 1) * (num_actions + 1)].detach().to(torch.float32)
            b = y[0].clone()
            w = y[1:] - b
            if k == 0:
                if float((w - torch.eye(num_actions)).abs().max()) > PROBE_ATOL or float(b.abs().max()) > PROBE_ATOL:
                    raise NotImplementedError("the first block of the symmetry function's output must be the original actions")
            else:
                self.actions.append((w.clone(), b))
        self._verify(func, env, obs_dims, num_actions)

    def _verify(self, func: Any, env: Any, obs_dims: dict[str, int], num_actions: int) -> None:
        """The derived maps against the function itself on random inputs, in the two ways upstream calls it: observations and actions together, and each alone."""
        g = torch.Generator().manual_seed(0)
        n = 64
        obs = TensorDict({name: torch.randn(n, d, generator=g) for name, d in obs_dims.items()}, batch_size=[n])
        actions = torch.randn(n, num_actions, generator=g)
        with torch.no_grad():
            both_obs, both_act = func(env=env, obs=obs.clone(), actions=actions.clone())
            only_obs, _ = func(env=env, obs=obs.clone(), actions=None)
            _, only_act = func(env=env, obs=None, actions=actions.clone())
        for k in range(1, self.num_aug):
            for name in self.obs:
                w, b = self.obs[name][k - 1]
                want = obs[name] @ w + b
                for tag, got in (("together with the actions", both_obs), ("alone", only_obs)):
                    err = float((got[name][k * n : (k + 1) * n] - want).abs().max())
                    if err > PROBE_ATOL:
                        raise NotImplementedError(f"the symmetry function is not affine in observation group '{name}' (copy {k}, called {tag}: the derived map is off by {err:.2e}): only affine mirror maps are supported")
            w, b = self.actions[k - 1]
            want = actions @ w + b
            for tag, got in (("together with the observations", both_act), ("alone", only_act)):
                err = float((got[k * n : (k + 1) * n] - want).abs().max())
                if err > PROBE_ATOL:
                    raise NotImplementedError(f"the symmetry function is not affine in the actions (copy {k}, called {tag}: the derived map is off by {err:.2e}): only affine mirror maps are supported")

    # -- composition for a set of groups ------------------------------------------------------------------------------------------
    def compose(self, groups: list[str], k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The block-diagonal map of copy `k` (1 .. K - 1) for the concatenation of `groups`: (W [d, d], b [d])."""
        ws, bs = zip(*(self.obs[g][k - 1] for g in groups))
        return torch.block_diag(*ws), torch.cat(bs)


def bf16_exact(w: torch.Tensor) -> bool:
    """True when every entry is exactly representable in bfloat16 (0, +-1, +-0.5, ...: all signed permutations)."""
    return bool(torch.equal(w.to(torch.bfloat16).to(torch.float32), w))


class _DeviceMap:
    """One affine map on the device: `x @ W + b` for [rows, dim_p] batch-sharded tensors (W, b are replicated, padded to tiles with zeros)."""

    def __init__(self, ctx: TTContext, w: torch.Tensor, b: torch.Tensor) -> None:
        self.ctx = ctx
        ttnn = ctx.ttnn
        dim_p = pad_up(w.shape[0])
        self.exact = bf16_exact(w)
        self.w = ctx.upload(pad_matrix(w, dim_p, dim_p), op="H-6", batch_sharded=False)
        self.w_bf16 = ttnn.typecast(self.w, ttnn.bfloat16) if self.exact else None
        self.b = ctx.upload(pad_vector(b, dim_p), op="H-6", batch_sharded=False) if float(b.abs().max()) > 0 else None

    def __call__(self, x: Any) -> Any:
        ctx = self.ctx
        ttnn = ctx.ttnn
        if self.exact:
            h1 = ttnn.typecast(x, ttnn.bfloat16)
            r1 = ttnn.subtract(x, ttnn.typecast(h1, ttnn.float32))
            h2 = ttnn.typecast(r1, ttnn.bfloat16)
            h3 = ttnn.typecast(ttnn.subtract(r1, ttnn.typecast(h2, ttnn.float32)), ttnn.bfloat16)
            y = None
            for part in (h1, h2, h3):
                p = ttnn.matmul(part, self.w_bf16, dtype=ttnn.float32, compute_kernel_config=ctx.compute_cfg)
                y = p if y is None else ttnn.add(y, p)
        else:
            y = ttnn.matmul(x, self.w, compute_kernel_config=ctx.compute_cfg)
        return y if self.b is None else ttnn.add(y, self.b)


class TTSymmetry:
    """Symmetry augmentation / mirror loss on the device. `mirror` = the upstream `Symmetry` object of `PPO.symmetry`."""

    def __init__(self, ctx: TTContext, mirror: Any, obs_dims: dict[str, int], actor_groups: list[str], critic_groups: list[str], num_actions: int, share_obs: bool) -> None:
        self.ctx, self.mirror = ctx, mirror
        self.use_da = bool(mirror.use_data_augmentation)
        self.use_loss = bool(mirror.use_mirror_loss)
        self.coeff = float(mirror.mirror_loss_coeff)
        self.maps = MirrorMaps(mirror.data_augmentation_func, mirror.env, obs_dims, num_actions, set(actor_groups) | set(critic_groups))
        ctx.host_ops.count("H-10", 4 * (sum(obs_dims.values()) + 1) * sum(obs_dims.values()) + 4 * (num_actions + 1) * num_actions)  # the probe tensors the user's function was evaluated on
        self.num_aug = self.maps.num_aug
        self.num_actions = num_actions
        self.share_obs = share_obs
        self.actor = {k: _DeviceMap(ctx, *self.maps.compose(actor_groups, k)) for k in range(1, self.num_aug)}
        self.critic = {k: self.actor[k] if share_obs else _DeviceMap(ctx, *self.maps.compose(critic_groups, k)) for k in range(1, self.num_aug)} if self.use_da else {}
        self.action = {k: _DeviceMap(ctx, *self.maps.actions[k - 1]) for k in range(1, self.num_aug)}

    # -- the extra columns of the packed table -------------------------------------------------------------------------------------
    def extra_columns(self, actor_dim_p: int, critic_dim_p: int, act_dim_p: int) -> list[tuple[str, int]]:
        """(name, width) of the columns the packed table gets per mirrored copy: the observations the actor sees; with data augmentation also the critic's and the actions."""
        cols = []
        for k in range(1, self.num_aug):
            cols.append((f"sym_actor_{k}", actor_dim_p))
            if self.use_da:
                if not self.share_obs:
                    cols.append((f"sym_critic_{k}", critic_dim_p))
                cols.append((f"sym_action_{k}", act_dim_p))
        return cols

    def mirror_blocks(self, storage: Any) -> list:
        """The mirrored copies of the whole rollout, in the order of `extra_columns` (computed once per update, on the device)."""
        actor_obs = storage.all_actor_obs()
        critic_obs = None if self.share_obs or not self.use_da else storage.all_critic_obs()
        actions = storage._cat("action") if self.use_da else None
        out = []
        for k in range(1, self.num_aug):
            out.append(self.actor[k](actor_obs))
            if self.use_da:
                if not self.share_obs:
                    out.append(self.critic[k](critic_obs))
                out.append(self.action[k](actions))
        return out

    def mirror_actions(self, mean: Any) -> Any:
        """The mirrored copies of `mean` ([R, A_p]) stacked along the rows: [(K - 1) R, A_p] (the targets of the mirror loss)."""
        parts = [self.action[k](mean) for k in range(1, self.num_aug)]
        return parts[0] if len(parts) == 1 else self.ctx.ttnn.concat(parts, dim=0)

    def loss_and_grad(self, mean_mirrored: Any, target: Any) -> tuple[Any, Any]:
        """MSE between the actor's mean on the mirrored observations and the mirrored mean on the original ones (a constant): (loss [1, 1], d loss / d mean_mirrored, 2 diff / n,
        without the coefficient). `n` = rows x real action dims; the padded columns are zero on both sides."""
        ttnn = self.ctx.ttnn
        diff = ttnn.subtract(mean_mirrored, target)
        n = float(diff.shape[0] * self.num_actions)  # per chip, like every other mean of the update (chip-local loss, gradients averaged: ADR-0004)
        loss = ttnn.multiply(ttnn.sum(ttnn.square(diff), dim=(0, 1), keepdim=True, compute_kernel_config=self.ctx.compute_cfg), 1.0 / n)
        return loss, ttnn.multiply(diff, 2.0 / n)
