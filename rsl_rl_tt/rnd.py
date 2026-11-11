# SPDX-License-Identifier: BSD-3-Clause
"""Random Network Distillation (upstream `rsl_rl.extensions.RandomNetworkDistillation`) on the device (post-v1 item 8.2, docs/porting/POST_V1.md).

Two MLPs of the same shape on the RND state (the concatenated observation groups `obs_groups["rnd_state"]`): a frozen random TARGET and a trained PREDICTOR. The intrinsic reward of a
step is the length of (target - predictor) embedding on the observation AFTER the step, optionally normalised by the running std of its discounted sum
(`EmpiricalDiscountedVariationNormalization`) and scaled by a (scheduled) weight; it is added to the extrinsic reward before the time-out bootstrap and GAE. The predictor learns the
MSE to the target on every PPO mini-batch with its OWN Adam (constant learning rate, no gradient clipping, one step per mini-batch).

Built from the upstream torch module (the "mirror", `PPO.rnd`) like every other TT module: the mirror is the source of the initial weights and the sink of `sync_to_torch()`
(checkpoints). Everything that changes lives on the device: predictor weights and Adam moments, the state normalizer statistics, the discounted-reward accumulator and its statistics.
All of it is allocated when the algorithm is constructed, before any trace can be captured (TTNN_ISSUES T-15).
"""

from __future__ import annotations

from typing import Any

import torch

from rsl_rl_tt.modules import TTEmpiricalNormalization, TTMLP
from rsl_rl_tt.optim import TTAdam
from rsl_rl_tt.tt_common import TTContext, pad_up, pad_vector, stage_obs


class TTRewardNormalization:
    """`EmpiricalDiscountedVariationNormalization` of a [num_envs] reward on the device.

    Per call (upstream `forward`, in training mode): `avg = gamma * avg + reward` (a per-environment accumulator that is NEVER reset on episode ends, like upstream's), the running
    mean / variance (Welford recurrence of `EmpiricalNormalization.update`, here over the `num_envs` values of `avg`) learn from `avg`, and the reward is divided by the running std
    (`rew / std`, no epsilon; the reward is returned unchanged while the std is not positive). Several chips: the statistics are global (count = rows x chips), the accumulator is local.

    The update rate `total / count` is a device scalar (computed on the host, `next_rate`), so the same graph can be replayed as a trace; a rate of 0 leaves the statistics untouched
    (upstream: `until` reached)."""

    def __init__(self, ctx: TTContext, mirror: Any, num_envs: int) -> None:
        self.ctx, self.mirror, self.num_envs = ctx, mirror, num_envs
        self.emp = mirror.emp_norm
        self.gamma = float(mirror.disc_avg.gamma)
        self.until = self.emp.until
        self.count = 0
        self.training = True
        self.avg = ctx.zeros(num_envs, 1, batch_sharded=True)  # upstream: None until the first call = `gamma * 0 + reward`, so zeros are the same thing
        self.mean = self.var = self.std = None
        self.sync_from_torch()

    def buffers(self) -> dict:
        return {"avg": self.avg, "mean": self.mean, "var": self.var, "std": self.std}

    def sync_from_torch(self) -> None:
        ctx, e = self.ctx, self.emp
        self.count = int(e.count)
        self.mean = ctx.upload_scalar_into(self.mean, float(e._mean.reshape(-1)[0]), op="H-6")
        self.var = ctx.upload_scalar_into(self.var, float(e._var.reshape(-1)[0]), op="H-6")
        self.std = ctx.upload_scalar_into(self.std, float(e._std.reshape(-1)[0]), op="H-6")

    def sync_to_torch(self) -> None:
        ctx, e = self.ctx, self.emp
        host = lambda t: float(ctx.download(t, op="H-6", batch_sharded=False)[0, 0])
        e._mean.fill_(host(self.mean))
        e._var.fill_(host(self.var))
        e._std.fill_(host(self.std))
        e.count.fill_(self.count)

    def next_rate(self, rows: int) -> float:
        """Host bookkeeping of the sample count (like upstream's long): the update rate `total / count` of this call, 0 when nothing is learned (eval mode, `until` reached)."""
        if not self.training or (self.until is not None and self.count >= self.until):
            return 0.0
        total = rows * self.ctx.n_chips
        self.count += total
        return total / self.count

    def apply(self, reward: Any, rate: Any, bufs: dict, update: bool = True) -> Any:
        """Normalise `reward` ([rows, 1], batch-sharded) with the state in `bufs` (the persistent buffers, or clones of them for a warm-up); `update`: advance the accumulator and the statistics
        (in place; the same statements as `TTEmpiricalNormalization.update`, with `rate` a [1, 1] device tensor)."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        if update:
            avg = ctx.put(bufs["avg"], ttnn.add(ttnn.multiply(bufs["avg"], self.gamma), reward))
            cfg = ctx.compute_cfg
            mean_x = ttnn.mean(avg, dim=0, keepdim=True, compute_kernel_config=cfg)
            var_x = ttnn.mean(ttnn.square(ttnn.subtract(avg, mean_x)), dim=0, keepdim=True, compute_kernel_config=cfg)
            if ctx.n_chips > 1:
                global_mean = ctx.mean_over_chips(mean_x)
                var_x = ctx.mean_over_chips(ttnn.add(var_x, ttnn.square(ttnn.subtract(mean_x, global_mean))))
                mean_x = global_mean
            delta_mean = ttnn.subtract(mean_x, bufs["mean"])
            new_mean = ttnn.add(bufs["mean"], ttnn.multiply(delta_mean, rate))
            inner = ttnn.add(ttnn.subtract(var_x, bufs["var"]), ttnn.multiply(delta_mean, ttnn.subtract(mean_x, new_mean)))
            var = ctx.put(bufs["var"], ttnn.add(bufs["var"], ttnn.multiply(inner, rate)))
            ctx.put(bufs["mean"], new_mean)
            ctx.put(bufs["std"], ttnn.sqrt(var))
        std = bufs["std"]
        positive = ttnn.gt(std, 0.0)  # [1, 1]: 1 where the running std is positive
        safe = ttnn.add(std, ttnn.subtract(1.0, positive))  # std, or 1 where it is not positive (no division by zero in the unselected branch)
        return ttnn.add(ttnn.multiply(ttnn.divide(reward, safe), positive), ttnn.multiply(reward, ttnn.subtract(1.0, positive)))


class TTRND:
    """The RND extension on the device. `mirror` = the upstream `RandomNetworkDistillation` instance of `PPO.rnd`."""

    def __init__(self, ctx: TTContext, mirror: Any, num_envs: int) -> None:
        self.ctx, self.mirror = ctx, mirror
        self.groups = list(mirror.obs_groups["rnd_state"])
        self.state_dim = int(mirror.num_states)
        self.state_dim_p = pad_up(self.state_dim)
        self.out_dim = int(mirror.num_outputs)
        self.predictor = TTMLP(ctx, mirror.predictor)
        self.target = TTMLP(ctx, mirror.target)
        for net in (self.predictor, self.target):
            if net.in_features != self.state_dim or net.out_features != self.out_dim:
                raise NotImplementedError("the RND networks must map the RND state to the embedding (plain MLPs)")
        self.state_norm = TTEmpiricalNormalization(ctx, mirror.state_normalizer, self.state_dim) if mirror.state_normalization else None
        self.reward_norm = TTRewardNormalization(ctx, mirror.reward_normalizer, num_envs) if mirror.reward_normalization else None
        self.adam = TTAdam(ctx, self.predictor.parameters())
        self.lr = ctx.upload_scalar(float(mirror.optimizer.param_groups[0]["lr"]), op="H-8")  # persistent (a trace reads it); the predictor's learning rate never adapts
        self.weight = float(mirror.initial_weight)
        mirror.weight = self.weight
        self.training = True
        # the reward path of one environment step runs on persistent buffers (so it can be captured as a trace): the RND observation (ROW_MAJOR, written in place), one [1, 32] row of the step's scalars
        # (column 0: the weight, column 1: the update rate of the reward normalizer; H-8) and the intrinsic reward (read by the reward add and downloaded for logging, H-9)
        self.num_envs = num_envs
        self.obs_in = ctx.upload_rm_into(None, torch.zeros(num_envs, self.state_dim_p), op="H-6", batch_sharded=True)
        self.scalars = ctx.upload(pad_vector(torch.tensor([self.weight, 0.0]), 32), op="H-8", batch_sharded=False)
        self.out = ctx.zeros(num_envs, 1, batch_sharded=True)

    # -- host staging ------------------------------------------------------------------------------------------------------
    def obs_to_device(self, obs: Any) -> Any:
        """H-3 concat + H-1 upload of the RND state of `obs`, as a TILE tensor [rows, state_dim_p] (a ROW_MAJOR upload tilized on the device: cheap on the host)."""
        ctx = self.ctx
        ctx.check_batch(obs[self.groups[0]].shape[0], "number of environments / samples")
        buf = stage_obs(obs, self.groups, self.state_dim_p)
        ctx.host_ops.count("H-3", buf.numel() * 4)
        return ctx.ttnn.to_layout(ctx.upload_rm_into(None, buf, op="H-1", batch_sharded=True), ctx.ttnn.TILE_LAYOUT)

    def allocate_act_buffers(self) -> None:
        """Persistent buffers for the RND observation of the step the actor acts on, when the storage keeps it (the RND state is neither the actor's nor the critic's observation): a tensor that
        is allocated after a trace was captured and lives across the replay of another trace would be overwritten by it (TTNN_ISSUES T-15), so this one is written in place."""
        ctx = self.ctx
        self._act_rm = ctx.upload_rm_into(None, torch.zeros(self.num_envs, self.state_dim_p), op="H-6", batch_sharded=True)
        self._act_tile = ctx.zeros(self.num_envs, self.state_dim_p, batch_sharded=True)

    def stage_act_obs(self, obs: Any, persistent: bool) -> Any:
        """The RND state of the observation the actor acts on (H-3, H-1) for the rollout storage. `persistent` (the traced path: the storage copies it into its preallocated slot): written in place into
        the persistent tile buffer. Otherwise (eager: the storage keeps a REFERENCE to every step's tensor) a fresh tensor, one per step."""
        ctx = self.ctx
        ctx.check_batch(obs[self.groups[0]].shape[0], "number of environments / samples")
        buf = stage_obs(obs, self.groups, self.state_dim_p)
        ctx.host_ops.count("H-3", buf.numel() * 4)
        ttnn = ctx.ttnn
        if not persistent:
            return ttnn.to_layout(ctx.upload_rm_into(None, buf, op="H-1", batch_sharded=True), ttnn.TILE_LAYOUT)
        ctx.upload_rm_into(self._act_rm, buf, op="H-1", batch_sharded=True)
        return ctx.put(self._act_tile, ttnn.to_layout(self._act_rm, ttnn.TILE_LAYOUT))

    # -- intrinsic reward ----------------------------------------------------------------------------------------------------
    def embedding_distance(self, state: Any) -> Any:
        """||target(s) - predictor(s)||_2 of the (normalised) state, [rows, 1] (padded embedding columns are zero in both networks)."""
        ttnn = self.ctx.ttnn
        x = self.state_norm.forward(state) if self.state_norm is not None else state
        diff = ttnn.subtract(self.target.forward(x), self.predictor.forward(x))
        return ttnn.sqrt(ttnn.sum(ttnn.square(diff), dim=1, keepdim=True, compute_kernel_config=self.ctx.compute_cfg))

    def step_weight(self) -> float:
        """The weight of this step: advances the step counter (upstream: once per `get_intrinsic_reward`, i.e. per environment step) and evaluates the weight schedule."""
        m = self.mirror
        m.update_counter += 1
        m.weight = m.weight_scheduler(step=m.update_counter, **m.weight_scheduler_params) if m.weight_scheduler is not None else m.initial_weight
        self.weight = float(m.weight)
        return self.weight

    def stage(self, obs: Any, rows: int) -> None:
        """Host side of one environment step's reward: the RND observation of the NEW observations (H-3, H-1) and the step's scalars (H-8) into their persistent buffers; advances the counters
        (`rows` = environments per chip)."""
        ctx = self.ctx
        ctx.check_batch(obs[self.groups[0]].shape[0], "number of environments")
        buf = stage_obs(obs, self.groups, self.state_dim_p)
        ctx.host_ops.count("H-3", buf.numel() * 4)
        ctx.upload_rm_into(self.obs_in, buf, op="H-1", batch_sharded=True)
        rate = self.reward_norm.next_rate(rows) if self.reward_norm is not None else 0.0
        ctx.upload_into(self.scalars, pad_vector(torch.tensor([self.step_weight(), rate]), 32), op="H-8", batch_sharded=False)

    def traceable(self) -> bool:
        """True when the reward graph may be captured: its structure is fixed at capture time and it advances the reward normalizer only in training mode (evaluation runs it eagerly)."""
        return self.reward_norm is None or self.reward_norm.training

    def _buffers(self) -> dict:
        return {"in": self.obs_in, "scalars": self.scalars, "out": self.out, **(self.reward_norm.buffers() if self.reward_norm is not None else {})}

    def graph(self, bufs: dict | None = None) -> None:
        """The device ops of one step's reward (eager, or captured as a trace by `TTPPO`): tilize the observation, embedding distance, reward normalisation (training: accumulator and statistics
        advance in place), weight; the result is written in place into `out`. `bufs`: the persistent buffers, or clones of them (the warm-up before a capture must not change the real state)."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        bufs = bufs or self._buffers()
        reward = self.embedding_distance(ttnn.to_layout(bufs["in"], ttnn.TILE_LAYOUT))
        if self.reward_norm is not None:
            reward = self.reward_norm.apply(reward, bufs["scalars"][:, 1:2], bufs, update=self.reward_norm.training)
        ctx.put(bufs["out"], ttnn.multiply(reward, bufs["scalars"][:, 0:1]))

    def warm_up_buffers(self) -> dict:
        """Clones of every buffer the graph WRITES (a warm-up pass of the graph on them compiles its programs without touching the real state, before a capture)."""
        ttnn = self.ctx.ttnn
        bufs = self._buffers()
        return {k: (v if k in ("in", "scalars") else ttnn.multiply(v, 1.0)) for k, v in bufs.items()}

    def step_reward(self, obs: Any, rows: int) -> Any:
        """`get_intrinsic_reward` for the observations after an environment step, eagerly: stage, run the graph, return the persistent `out` [num_envs, 1] (the next call overwrites it)."""
        self.stage(obs, rows)
        self.graph()
        return self.out

    # -- predictor loss ------------------------------------------------------------------------------------------------------
    def loss_and_grads(self, state: Any, rows: int) -> tuple[Any, list]:
        """MSE between the predictor's and the target's embedding of a mini-batch (`state`: [rows, state_dim_p]) -> (loss [1, 1], gradients of the predictor parameters).

        `mse_loss` averages over all `rows x out_dim` elements, so d loss / d prediction = 2 (prediction - target) / (rows x out_dim); the normaliser statistics are not touched."""
        ttnn = self.ctx.ttnn
        x = self.state_norm.forward(state) if self.state_norm is not None else state
        target = self.target.forward(x)
        prediction, cache = self.predictor.forward_train(x)
        diff = ttnn.subtract(prediction, target)
        n = float(rows * self.out_dim)
        loss = ttnn.multiply(ttnn.sum(ttnn.square(diff), dim=(0, 1), keepdim=True, compute_kernel_config=self.ctx.compute_cfg), 1.0 / n)
        grads = [t for pair in self.predictor.backward(cache, ttnn.multiply(diff, 2.0 / n)) for t in pair]
        return loss, grads

    def apply(self, grads: list) -> None:
        """One Adam step of the predictor (`adam.prepare_step()` must have run); the parameters are written in place."""
        self.predictor.set_parameters(self.adam.step(self.predictor.parameters(), grads, self.lr))

    def update_normalization(self, states: Any, rows: int) -> None:
        """The state normalizer learns from the whole rollout after the epochs (upstream `update_normalization`)."""
        if self.state_norm is not None:
            self.state_norm.update(states, rows)

    def set_training(self, training: bool) -> None:
        self.training = training
        if self.state_norm is not None:
            self.state_norm.training = training
        if self.reward_norm is not None:
            self.reward_norm.training = training

    # -- mirror synchronisation (H-6) ----------------------------------------------------------------------------------------
    def tensors(self) -> list:
        """Everything that is replicated and must be identical on every chip (the replica check of a mesh run)."""
        out = self.predictor.parameters() + self.target.parameters() + self.adam.m + self.adam.v
        if self.state_norm is not None:
            out += [self.state_norm._mean, self.state_norm._var]
        if self.reward_norm is not None:
            out += [self.reward_norm.mean, self.reward_norm.var, self.reward_norm.std]
        return out

    def _param_slots(self) -> list[tuple]:
        """For every parameter of the torch optimizer, in its order: (torch param, index in `predictor.parameters()`, kind, layer)."""
        slots = {}
        lin = [m for m in self.mirror.predictor if type(m).__name__ == "Linear"]
        for k, mod in enumerate(lin):
            slots[id(mod.weight)] = (mod.weight, 2 * k, "weight", self.predictor.layers[k])
            slots[id(mod.bias)] = (mod.bias, 2 * k + 1, "bias", self.predictor.layers[k])
        return [slots[id(p)] for p in self.mirror.optimizer.param_groups[0]["params"] if id(p) in slots]

    def sync_from_torch(self) -> None:
        """The torch mirror (after `load()`) -> the device: weights of both networks, normalizer statistics, Adam moments / step, learning rate (`update_counter` lives on the mirror)."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        self.predictor.sync_from_torch()
        self.target.sync_from_torch()
        if self.state_norm is not None:
            self.state_norm.sync_from_torch()
        if self.reward_norm is not None:
            self.reward_norm.sync_from_torch()
        self.lr = ctx.upload_scalar_into(self.lr, float(self.mirror.optimizer.param_groups[0]["lr"]), op="H-8")
        state = self.mirror.optimizer.state
        m, v = [ttnn.zeros_like(t) for t in self.adam.m], [ttnn.zeros_like(t) for t in self.adam.v]
        step = 0
        for p, idx, kind, layer in self._param_slots():
            if p not in state or "exp_avg" not in state[p]:
                continue  # a fresh torch optimizer: Adam restarts from zero moments
            step = max(step, int(state[p]["step"]))
            convert = layer.weight_to_host if kind == "weight" else layer.bias_to_host
            m[idx] = ctx.upload(convert(state[p]["exp_avg"]), op="H-6", batch_sharded=False)
            v[idx] = ctx.upload(convert(state[p]["exp_avg_sq"]), op="H-6", batch_sharded=False)
        self.adam.load_state(m, v, step)
        self.weight = float(self.mirror.weight) if hasattr(self.mirror, "weight") else float(self.mirror.initial_weight)

    def sync_to_torch(self) -> None:
        """The device -> the torch mirror: predictor weights, normalizer statistics, Adam moments (a checkpoint needs all of it)."""
        import torch

        ctx = self.ctx
        self.predictor.sync_to_torch()
        if self.state_norm is not None:
            self.state_norm.sync_to_torch()
        if self.reward_norm is not None:
            self.reward_norm.sync_to_torch()
        for p, idx, kind, layer in self._param_slots():
            host = []
            for tensor in (self.adam.m[idx], self.adam.v[idx]):
                t = ctx.download(tensor, op="H-6", batch_sharded=False)
                host.append(layer.weight_from_host(t) if kind == "weight" else layer.bias_from_host(t))
            self.mirror.optimizer.state[p] = {"step": torch.tensor(float(self.adam.t)), "exp_avg": host[0].clone(), "exp_avg_sq": host[1].clone()}
