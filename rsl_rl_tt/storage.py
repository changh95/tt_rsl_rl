# SPDX-License-Identifier: BSD-3-Clause
"""Rollout storage, GAE and mini-batch construction on the device (BRIEF Phase 3; replaces the temporary host storage H-7 of Phase 2).

Layout: every field of a step is a device tensor of `num_envs` rows (batch-sharded over the mesh). A step is added as references (nothing
is copied); at the end of the rollout the steps are concatenated along the rows in the order of upstream's `flatten(0, 1)` (row = step *
num_envs + env), so the host-drawn permutation of `update()` indexes exactly the same samples as upstream's `torch.randperm`.

GAE follows `rsl_rl.algorithms.ppo.PPO.compute_returns` line by line, including `advantages = returns - values` after the loop and the
unbiased-std global advantage normalisation with eps 1e-8. The packed table that `update()` shuffles holds, per row,
    [actor obs | critic obs (if different) | action | old mean | old std (state-dependent std only) | RND state (only when it is neither the actor's nor the critic's
     observation) | scalars(32: value, return, advantage, old log-prob)]
so one row gather moves a whole sample; mini-batches are contiguous row slices of the shuffled table (upstream also reuses the SAME permutation
in every epoch).
"""

from __future__ import annotations

from typing import Any

from rsl_rl_tt.tt_common import TTContext, unit_row

SCALAR_COLS = {"value": 0, "returns": 1, "advantages": 2, "old_logp": 3}
SCALAR_BLOCK = 32


class TTRolloutStorage:
    """Device rollout storage.

    `persistent=False` (eager path): a step is stored by reference (nothing is copied).
    `persistent=True` (traced path): every step has PREALLOCATED device buffers (`preallocate()`, before the first trace is captured) and
    `add_transition` copies into them in place. A trace's intermediate tensors are freed when its capture ends, so any tensor that is allocated
    AFTER a capture and must keep its value across replays can land on those addresses and is overwritten by the next replay (found in Phase 5:
    docs/porting/TTNN_ISSUES.md T-15). The persistent-state rule of the traced path: everything that survives a replay is allocated first."""

    def __init__(
        self, ctx: TTContext, num_envs: int, num_transitions_per_env: int, actor_dim_p: int, critic_dim_p: int, act_dim_p: int, share_obs: bool, persistent: bool = False,
        store_std: bool = False, rnd_dim_p: int = 0, rnd_source: str | None = None, extra_cols: list | None = None,
    ) -> None:
        """`store_std`: also keep the per-sample std of the action distribution (a state-dependent std: upstream's `distribution_params = (mean, std)`); the packed
        table then has one more column block. The layout is unchanged otherwise.

        `rnd_source`: where the observations of the RND state come from (None = no RND). "actor" / "critic": the RND state IS that observation (same groups in the same
        order: nothing extra is stored); "own": a separate block of `rnd_dim_p` columns per step (the state of RND is a different set of observation groups).

        `extra_cols`: [(name, width)] more column blocks of the packed table, filled by the caller at the end of the rollout (`packed_table(extra=...)`): the mirrored copies of the
        symmetry extension, which are computed once per update and must be shuffled together with the rows they belong to."""
        self.ctx = ctx
        self.store_std = store_std
        assert rnd_source in (None, "actor", "critic", "own"), rnd_source
        self.rnd_source = rnd_source
        self.rnd_own = rnd_source == "own"
        self.rnd_dim_p = rnd_dim_p if self.rnd_own else 0
        self.num_envs, self.num_transitions_per_env = num_envs, num_transitions_per_env
        self.actor_dim_p, self.critic_dim_p, self.act_dim_p, self.share_obs = actor_dim_p, critic_dim_p, act_dim_p, share_obs
        self.persistent = persistent
        self.step = 0
        self._steps: list[dict] = []
        self._slots: list[dict] | None = None
        self._out: dict = {}  # persistent returns / advantages of a traced GAE
        self.returns = self.advantages = None
        # column layout of the packed table
        self.col_actor = 0
        self.col_critic = actor_dim_p
        self.col_action = actor_dim_p + (0 if share_obs else critic_dim_p)
        self.col_mean = self.col_action + act_dim_p
        self.col_std = self.col_mean + act_dim_p
        self.col_rnd = self.col_std + (act_dim_p if store_std else 0)
        self.extra_cols: dict[str, tuple[int, int]] = {}
        col = self.col_rnd + self.rnd_dim_p
        for name, width in extra_cols or []:
            self.extra_cols[name] = (col, col + width)
            col += width
        self.col_scalars = col
        self.width = self.col_scalars + SCALAR_BLOCK
        # constant selector rows of `packed_table` (persistent device state: created here, before any trace is captured)
        self._unit_cols = {name: ctx.upload(unit_row(col, SCALAR_BLOCK), op="H-6", batch_sharded=False) for name, col in SCALAR_COLS.items()}

    def preallocate(self) -> None:
        """Allocate the per-step buffers of the whole rollout (zeros). Must run before the first trace capture (see the class docstring)."""
        zeros = lambda cols: self.ctx.zeros(self.num_envs, cols, batch_sharded=True)
        slots = []
        for _ in range(self.num_transitions_per_env):
            slot = dict(actor_obs=zeros(self.actor_dim_p), action=zeros(self.act_dim_p), mean=zeros(self.act_dim_p), value=zeros(1), logp=zeros(1), reward=zeros(1), done=zeros(1))
            if self.store_std:
                slot["std"] = zeros(self.act_dim_p)
            if self.rnd_own:
                slot["rnd_obs"] = zeros(self.rnd_dim_p)
            slot["critic_obs"] = slot["actor_obs"] if self.share_obs else zeros(self.critic_dim_p)
            slots.append(slot)
        self._slots = slots

    @property
    def batch_size(self) -> int:
        """Samples of one rollout over ALL chips."""
        return self.num_envs * self.num_transitions_per_env

    @property
    def local_batch_size(self) -> int:
        """Samples of one rollout on ONE chip (the environments are sharded over the chips): the denominator of every chip-local statistic."""
        return self.batch_size // self.ctx.n_chips

    # -- rollout ----------------------------------------------------------------------------------------------------------
    def add_transition(self, *, actor_obs: Any, critic_obs: Any, action: Any, mean: Any, value: Any, logp: Any, reward: Any, done: Any, std: Any = None, rnd_obs: Any = None) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        fields = dict(actor_obs=actor_obs, critic_obs=critic_obs, action=action, mean=mean, value=value, logp=logp, reward=reward, done=done)
        if self.store_std:
            assert std is not None, "a storage with store_std needs the std of every transition"
            fields["std"] = std
        if self.rnd_own:
            assert rnd_obs is not None, "a storage with its own RND state block needs the RND observations of every transition"
            fields["rnd_obs"] = rnd_obs
        if self.persistent:
            assert self._slots is not None, "preallocate() must run before the first transition of a persistent storage"
            slot = self._slots[self.step]
            for key, tensor in fields.items():
                if not (key == "critic_obs" and self.share_obs):  # shared observations: the critic slot IS the actor slot
                    self.ctx.put(slot[key], tensor)
            fields = slot
        self._steps.append(fields)
        self.step += 1

    def clear(self) -> None:
        self.step = 0
        self._steps = []
        self.returns = self.advantages = None

    def _cat(self, key: str) -> Any:
        parts = [s[key] for s in self._steps]
        return parts[0] if len(parts) == 1 else self.ctx.ttnn.concat(parts, dim=0)

    # -- GAE --------------------------------------------------------------------------------------------------------------
    def compute_returns(self, last_values: Any, gamma: float, lam: float, normalize_advantages: bool) -> None:
        """Returns and advantages on the device (`last_values` [num_envs, 1] = critic(obs after the last step))."""
        ttnn = self.ctx.ttnn
        n = self.num_transitions_per_env
        assert self.step == n, f"compute_returns needs a full rollout ({self.step} of {n} steps stored)"
        advantage = None
        returns: list = [None] * n
        for step in reversed(range(n)):
            s = self._steps[step]
            next_values = last_values if step == n - 1 else self._steps[step + 1]["value"]
            not_terminal = ttnn.subtract(1.0, s["done"])  # 1 - done
            # delta = r + not_terminal * gamma * V(next) - V(step)
            delta = ttnn.subtract(ttnn.add(s["reward"], ttnn.multiply(ttnn.multiply(not_terminal, gamma), next_values)), s["value"])
            carry = ttnn.multiply(ttnn.multiply(not_terminal, gamma * lam), advantage) if advantage is not None else None
            advantage = delta if carry is None else ttnn.add(delta, carry)
            returns[step] = ttnn.add(advantage, s["value"])
        self.returns = returns[0] if n == 1 else ttnn.concat(returns, dim=0)
        self.advantages = ttnn.subtract(self.returns, self._cat("value"))
        if normalize_advantages and self.local_batch_size > 1:  # chip-LOCAL statistics, like a rank of upstream multi-GPU (ADR-0004)
            self.advantages = self.normalize(self.advantages, self.local_batch_size)
        if self.persistent:  # a (traced) GAE writes its results in place into persistent buffers (adopted on the first pass)
            for key in ("returns", "advantages"):
                self._out[key] = self.ctx.put(self._out.get(key), getattr(self, key))
                setattr(self, key, self._out[key])

    def bind_outputs(self) -> None:
        """After a replay of the traced GAE: point `returns` / `advantages` at the persistent buffers the trace wrote (`clear()` drops the references)."""
        self.returns, self.advantages = self._out["returns"], self._out["advantages"]

    def normalize(self, x: Any, rows: int) -> Any:
        """(x - mean) / (std + 1e-8) with torch's unbiased std over `rows` samples (two-pass variance)."""
        ttnn = self.ctx.ttnn
        cfg = self.ctx.compute_cfg
        mean = ttnn.mean(x, dim=0, keepdim=True, compute_kernel_config=cfg)
        centered = ttnn.subtract(x, mean)
        var = ttnn.multiply(ttnn.mean(ttnn.square(centered), dim=0, keepdim=True, compute_kernel_config=cfg), rows / (rows - 1.0))
        return ttnn.divide(centered, ttnn.add(ttnn.sqrt(var), 1e-8))

    # -- update inputs ----------------------------------------------------------------------------------------------------
    def all_actor_obs(self) -> Any:
        return self._cat("actor_obs")

    def all_critic_obs(self) -> Any:
        return self.all_actor_obs() if self.share_obs else self._cat("critic_obs")

    def all_rnd_obs(self) -> Any:
        """The RND state of every stored step, [batch, rnd dim_p] (the actor's / critic's observations when the RND state is the same set of groups)."""
        assert self.rnd_source is not None, "this storage was built without RND"
        return {"actor": self.all_actor_obs, "critic": self.all_critic_obs, "own": lambda: self._cat("rnd_obs")}[self.rnd_source]()

    def _unit(self, name: str) -> Any:
        """[1, 32] row with a 1 in the column of scalar `name` (selects a scalar column when blocks are assembled)."""
        return self._unit_cols[name]

    def packed_table(self, extra: list | None = None) -> Any:
        """[batch, width] fp32 table of the whole rollout in upstream's flatten order (advantages/returns must be computed); `extra`: the tensors of the `extra_cols` blocks, in order."""
        ttnn = self.ctx.ttnn
        assert self.returns is not None, "compute_returns() must run before update()"
        scalars = None
        for name, column in (("value", self._cat("value")), ("returns", self.returns), ("advantages", self.advantages), ("old_logp", self._cat("logp"))):
            block = ttnn.multiply(column, self._unit(name))  # [rows, 1] x [1, 32] -> [rows, 32], exact (x * 1.0 / x * 0.0)
            scalars = block if scalars is None else ttnn.add(scalars, block)
        parts = [self.all_actor_obs()]
        if not self.share_obs:
            parts.append(self._cat("critic_obs"))
        parts += [self._cat("action"), self._cat("mean")] + ([self._cat("std")] if self.store_std else []) + ([self._cat("rnd_obs")] if self.rnd_own else []) + list(extra or []) + [scalars]
        return ttnn.concat(parts, dim=1)

    def shuffle(self, table: Any, permutation: Any) -> Any:
        """Exact fp32 row gather with the (host-drawn) permutation, a UINT32 ROW_MAJOR [1, batch] device tensor (`ctx.upload_index`).

        `ttnn.tosa_gather`/`gather` move fp32 at ~50 ns per element (TTNN_ISSUES T-12) and `ttnn.embedding` accepts bf16 tables only, so every fp32
        value is split EXACTLY into three bf16 values (h1 = bf16(x), r1 = x - h1, h2 = bf16(r1), r2 = r1 - h2, h3 = bf16(r2); the subtractions are
        exact in fp32), the 3 x width bf16 columns are gathered with `embedding` (a page copy, ~1 ms per 100k rows) and added back: bit-exact for
        every finite value with |x| >= 1e-30 (below that the tiny remainders hit the fp32 flush-to-zero threshold: absolute error <= 2e-38) and 40-50x faster than the fp32 gather (Phase 5 probe, TTNN_ISSUES T-12)."""
        ttnn = self.ctx.ttnn
        n, w = self.local_batch_size, self.width  # per chip: every chip gathers from its own shard with its own (local) indices
        h1 = ttnn.typecast(table, ttnn.bfloat16)
        r1 = ttnn.subtract(table, ttnn.typecast(h1, ttnn.float32))
        h2 = ttnn.typecast(r1, ttnn.bfloat16)
        r2 = ttnn.subtract(r1, ttnn.typecast(h2, ttnn.float32))
        h3 = ttnn.typecast(r2, ttnn.bfloat16)
        pieces = ttnn.to_layout(ttnn.concat([h1, h2, h3], dim=1), ttnn.ROW_MAJOR_LAYOUT)
        gathered = ttnn.to_layout(ttnn.reshape(ttnn.embedding(permutation, pieces, layout=ttnn.ROW_MAJOR_LAYOUT), (n, 3 * w)), ttnn.TILE_LAYOUT)
        parts = [ttnn.typecast(gathered[:, k * w : (k + 1) * w], ttnn.float32) for k in range(3)]
        return ttnn.add(ttnn.add(parts[0], parts[1]), parts[2])

    def split(self, rows: Any) -> dict:
        """Column views of a (mini-)batch of table rows."""
        s = self.col_scalars
        view = lambda a, b: rows[:, a:b]
        return {
            "actor_obs": view(self.col_actor, self.col_actor + self.actor_dim_p),
            "critic_obs": view(self.col_actor, self.col_actor + self.actor_dim_p) if self.share_obs else view(self.col_critic, self.col_critic + self.critic_dim_p),
            "actions": view(self.col_action, self.col_action + self.act_dim_p),
            "old_mean": view(self.col_mean, self.col_mean + self.act_dim_p),
            **({"old_std": view(self.col_std, self.col_std + self.act_dim_p)} if self.store_std else {}),
            **{name: view(a, b) for name, (a, b) in self.extra_cols.items()},
            **({"rnd_obs": {"actor": view(self.col_actor, self.col_actor + self.actor_dim_p), "critic": view(self.col_actor, self.col_actor + self.actor_dim_p) if self.share_obs else view(self.col_critic, self.col_critic + self.critic_dim_p),
                            "own": view(self.col_rnd, self.col_rnd + self.rnd_dim_p) if self.rnd_own else None}[self.rnd_source]} if self.rnd_source else {}),
            "old_values": view(s + SCALAR_COLS["value"], s + SCALAR_COLS["value"] + 1),
            "returns": view(s + SCALAR_COLS["returns"], s + SCALAR_COLS["returns"] + 1),
            "advantages": view(s + SCALAR_COLS["advantages"], s + SCALAR_COLS["advantages"] + 1),
            "old_logp": view(s + SCALAR_COLS["old_logp"], s + SCALAR_COLS["old_logp"] + 1),
        }
