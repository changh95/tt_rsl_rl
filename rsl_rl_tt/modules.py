# SPDX-License-Identifier: BSD-3-Clause
"""TTNN counterparts of upstream `MLPModel`, `EmpiricalNormalization` and `GaussianDistribution` (BRIEF Phase 2, ADR-0001 option A).

Each TT module is built from the matching UPSTREAM torch module (its "mirror"): the mirror is the source of the initial weights (so the
initialisation is bit-identical to upstream for the same torch seed) and the sink of `sync_to_torch()` (used by `get_policy()`, `save()`
and the export paths, allow-list H-6). The device copy is the working copy. This file imports no torch and contains no torch compute: tensors only cross
through `TTContext.upload/download` and the padding helpers of `tt_common` (a test enforces that `ttnn.from_torch`/`to_torch` appear
nowhere else).

Numerics follow the measured noise floors of docs/porting/TOLERANCES.md: fp32 everywhere, HiFi4 + fp32 accumulation for matmul, the
accurate (non-approximate) mode for elementwise ops, `mean` of squared deviations (two-pass) instead of `ttnn.var`.
"""

from __future__ import annotations

import math
from typing import Any

from rsl_rl_tt.tt_common import TTContext, linear_from_host, linear_to_host, mask_row, pad_up, pad_vector, stage_obs

HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
ACTIVATIONS = ("elu", "relu", "tanh", "identity")  # v1 scope: what was measured in Phase 0 (BRIEF 1: everything else needs approval)


def _acc(ttnn: Any) -> dict:
    """Extra kwargs for fp32 binary ops: none. `fast_and_approximate_mode=False` is rejected for FLOAT32 outputs ("only supported for a
    BFLOAT16 output", TT_FATAL, 0.81.1: the flag only controls bf16 rounding) and the default path is the full-fp32 one; the measured
    errors (tests/tt/test_tt_modules.py, TOLERANCES.md) are what is relied on, not the flag. Kept as a single switch point."""
    return {}


class TTEmpiricalNormalization:
    """`(x - mean) / (std + eps)` with Welford-style running statistics, per feature, over the whole batch (upstream semantics)."""

    def __init__(self, ctx: TTContext, mirror: Any, dim: int) -> None:
        self.ctx, self.mirror, self.dim = ctx, mirror, dim
        self.dim_p = pad_up(dim)
        self.eps = float(mirror.eps)
        self.until = mirror.until
        self.count = 0  # host bookkeeping of the sample count (a Python int, exactly like upstream's long); never a tensor op
        self.training = True
        self.sync_from_torch()

    # -- state ------------------------------------------------------------------------------------------------------------
    def sync_from_torch(self) -> None:
        ctx = self.ctx
        m = self.mirror
        mean = pad_vector(m._mean.reshape(-1), self.dim_p, 0.0)
        var = pad_vector(m._var.reshape(-1), self.dim_p, 1.0)
        self.count = int(m.count)
        self._mean = ctx.upload(mean, op="H-6", batch_sharded=False)
        self._var = ctx.upload(var, op="H-6", batch_sharded=False)
        self._refresh_derived()

    def sync_to_torch(self) -> None:
        ctx, m = self.ctx, self.mirror
        mean = ctx.download(self._mean, op="H-6", batch_sharded=False)[0, : self.dim]
        var = ctx.download(self._var, op="H-6", batch_sharded=False)[0, : self.dim]
        m._mean.copy_(mean.reshape(1, -1))
        m._var.copy_(var.reshape(1, -1))
        m._std.copy_(var.sqrt().reshape(1, -1))
        m.count.fill_(self.count)

    def _refresh_derived(self) -> None:
        ttnn = self.ctx.ttnn
        self._std = ttnn.sqrt(self._var)
        self._std_eps = ttnn.add(self._std, self.eps, **_acc(ttnn))  # (std + eps)

    # -- compute ----------------------------------------------------------------------------------------------------------
    def forward(self, x: Any) -> Any:
        ttnn = self.ctx.ttnn
        return ttnn.divide(ttnn.subtract(x, self._mean, **_acc(ttnn)), self._std_eps, **_acc(ttnn))

    def update(self, x: Any, rows: int) -> None:
        """Learn the statistics of the batch `x` ([rows, dim_p], batch-sharded); same recurrence as upstream `EmpiricalNormalization.update`."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        if not self.training or (self.until is not None and self.count >= self.until):
            return
        if ctx.n_chips != 1:
            raise NotImplementedError("global normalizer statistics over several chips arrive in Phase 6 (ADR-0004: gather + sum)")
        self.count += rows
        rate = ctx.upload_scalar(rows / self.count, op="H-8")  # a device tensor, so a trace does not freeze it
        cfg = ctx.compute_cfg
        mean_x = ttnn.mean(x, dim=0, keepdim=True, compute_kernel_config=cfg)
        centered = ttnn.subtract(x, mean_x, **_acc(ttnn))
        var_x = ttnn.mean(ttnn.square(centered), dim=0, keepdim=True, compute_kernel_config=cfg)  # two-pass variance (ttnn.var is 2e-4 off)
        delta_mean = ttnn.subtract(mean_x, self._mean, **_acc(ttnn))
        new_mean = ttnn.add(self._mean, ttnn.multiply(delta_mean, rate), **_acc(ttnn))
        # var += rate * (var_x - var + delta_mean * (mean_x - new_mean))
        inner = ttnn.add(ttnn.subtract(var_x, self._var, **_acc(ttnn)), ttnn.multiply(delta_mean, ttnn.subtract(mean_x, new_mean, **_acc(ttnn))), **_acc(ttnn))
        self._var = ttnn.add(self._var, ttnn.multiply(inner, rate), **_acc(ttnn))
        self._mean = new_mean
        self._refresh_derived()

    # -- host views (for tests / logging) ---------------------------------------------------------------------------------
    def host_stats(self) -> dict[str, Any]:
        ctx = self.ctx
        return {
            "mean": ctx.download(self._mean, op="H-6", batch_sharded=False)[0, : self.dim],
            "var": ctx.download(self._var, op="H-6", batch_sharded=False)[0, : self.dim],
            "std": ctx.download(self._std, op="H-6", batch_sharded=False)[0, : self.dim],
            "count": self.count,
        }


class TTLinear:
    """y = x @ W + b on padded tensors; W is stored as [in_p, out_p]."""

    def __init__(self, ctx: TTContext, mirror: Any) -> None:
        self.ctx, self.mirror = ctx, mirror
        self.in_features, self.out_features = mirror.in_features, mirror.out_features
        self.in_p, self.out_p = pad_up(self.in_features), pad_up(self.out_features)
        self.sync_from_torch()

    def sync_from_torch(self) -> None:
        w, b = linear_to_host(self.mirror.weight, self.mirror.bias, self.in_p, self.out_p)
        self.w = self.ctx.upload(w, op="H-6", batch_sharded=False)
        self.b = self.ctx.upload(b, op="H-6", batch_sharded=False)

    def sync_to_torch(self) -> None:
        w = self.ctx.download(self.w, op="H-6", batch_sharded=False)
        b = self.ctx.download(self.b, op="H-6", batch_sharded=False)
        w_t, b_t = linear_from_host(w, b, self.out_features, self.in_features)
        self.mirror.weight.data.copy_(w_t)
        self.mirror.bias.data.copy_(b_t)

    def forward(self, x: Any) -> Any:
        return self.ctx.ttnn.linear(x, self.w, bias=self.b, compute_kernel_config=self.ctx.compute_cfg)


class TTMLP:
    """Linear + activation stack of an upstream `rsl_rl.modules.MLP` (last layer linear)."""

    def __init__(self, ctx: TTContext, mirror: Any) -> None:
        self.ctx = ctx
        self.layers: list[TTLinear] = []
        self.acts: list[str | None] = []  # activation after each linear (None for the last one)
        mods = list(mirror)
        for i, mod in enumerate(mods):
            if type(mod).__name__ == "Linear":
                self.layers.append(TTLinear(ctx, mod))
                nxt = mods[i + 1] if i + 1 < len(mods) else None
                self.acts.append(_activation_name(nxt))
            elif i == 0 or not _is_activation(mod):
                raise NotImplementedError(f"MLP layer {type(mod).__name__} is outside the v1 scope (Linear + {ACTIVATIONS})")
        self.in_features = self.layers[0].in_features
        self.out_features = self.layers[-1].out_features

    def forward(self, x: Any) -> Any:
        ttnn = self.ctx.ttnn
        for layer, act in zip(self.layers, self.acts):
            x = layer.forward(x)
            if act == "elu":
                x = ttnn.elu(x, alpha=1.0)
            elif act == "relu":
                x = ttnn.relu(x)
            elif act == "tanh":
                x = ttnn.tanh(x)
        return x

    def forward_train(self, x: Any) -> tuple[Any, list]:
        """Forward pass that keeps what the backward pass needs: per layer (input, pre-activation, activation output)."""
        ttnn = self.ctx.ttnn
        cache = []
        for layer, act in zip(self.layers, self.acts):
            z = layer.forward(x)
            if act == "elu":
                a = ttnn.elu(z, alpha=1.0)
            elif act == "relu":
                a = ttnn.relu(z)
            elif act == "tanh":
                a = ttnn.tanh(z)
            else:
                a = z
            cache.append((x, z, a))
            x = a
        return x, cache

    def backward(self, cache: list, grad_out: Any) -> list[tuple[Any, Any]]:
        """Hand-written VJP of the Linear + activation stack: returns [(dW [in_p, out_p], db [1, out_p])] per layer.

        Padding stays inert: padded inputs are zero (so padded rows of dW are zero), grad_out has zero padded columns (masked by the
        caller), and padded weight rows are zero (so no gradient leaks into padded hidden units)."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        cfg = ctx.compute_cfg
        grads: list = [None] * len(self.layers)
        g = grad_out
        for i in reversed(range(len(self.layers))):
            x_in, z, a = cache[i]
            act = self.acts[i]
            if act == "elu":
                g = ttnn.elu_bw(g, z, alpha=1.0)[0]
            elif act == "tanh":
                g = ttnn.multiply(g, ttnn.subtract(1.0, ttnn.square(a)))  # d tanh = 1 - tanh^2
            elif act == "relu":
                g = ttnn.multiply(g, ttnn.gt(z, 0.0))  # torch: zero gradient at z == 0
            dw = ttnn.matmul(x_in, g, transpose_a=True, compute_kernel_config=cfg)
            db = ttnn.sum(g, dim=0, keepdim=True, compute_kernel_config=cfg)
            grads[i] = (dw, db)
            if i > 0:
                g = ttnn.matmul(g, self.layers[i].w, transpose_b=True, compute_kernel_config=cfg)
        return grads

    def parameters(self) -> list:
        """Device parameter tensors in torch order: weight0, bias0, weight1, bias1, ..."""
        return [t for layer in self.layers for t in (layer.w, layer.b)]

    def set_parameters(self, values: list) -> None:
        for k, layer in enumerate(self.layers):
            layer.w, layer.b = values[2 * k], values[2 * k + 1]

    def sync_from_torch(self) -> None:
        for layer in self.layers:
            layer.sync_from_torch()

    def sync_to_torch(self) -> None:
        for layer in self.layers:
            layer.sync_to_torch()


def _activation_name(mod: Any) -> str | None:
    if mod is None:
        return None
    name = type(mod).__name__
    table = {"ELU": "elu", "ReLU": "relu", "Tanh": "tanh", "Identity": "identity"}
    if name not in table:
        if name == "Linear":
            return None
        raise NotImplementedError(f"activation {name} is outside the v1 scope {ACTIVATIONS}")
    if name == "ELU" and abs(getattr(mod, "alpha", 1.0) - 1.0) > 0:
        raise NotImplementedError("ELU with alpha != 1")
    return table[name]


def _is_activation(mod: Any) -> bool:
    return type(mod).__name__ in {"ELU", "ReLU", "Tanh", "Identity"}


class TTGaussianDistribution:
    """State-independent diagonal Gaussian (std parameter [1, A_p]); padded action columns are masked out of log_prob / entropy."""

    def __init__(self, ctx: TTContext, mirror: Any) -> None:
        if type(mirror).__name__ != "GaussianDistribution":
            raise NotImplementedError(f"{type(mirror).__name__} is outside the v1 scope (GaussianDistribution, scalar/log std)")
        self.ctx, self.mirror = ctx, mirror
        self.dim = mirror.output_dim
        self.dim_p = pad_up(self.dim)
        self.std_type = mirror.std_type
        self.std_range = tuple(mirror.std_range)
        self.log_std_range = tuple(mirror.log_std_range)
        self.learn_std = (mirror.std_param if self.std_type == "scalar" else mirror.log_std_param).requires_grad
        self.mask = ctx.upload(mask_row(self.dim, self.dim_p), op="H-6", batch_sharded=False)
        self.sync_from_torch()

    # -- state ------------------------------------------------------------------------------------------------------------
    def _param(self) -> Any:
        return self.mirror.std_param if self.std_type == "scalar" else self.mirror.log_std_param

    def sync_from_torch(self) -> None:
        fill = 1.0 if self.std_type == "scalar" else 0.0  # padded columns: std = 1 (log std = 0), never trained, masked out of the sums
        self.param = self.ctx.upload(pad_vector(self._param().detach(), self.dim_p, fill), op="H-6", batch_sharded=False)
        self.refresh()

    def sync_to_torch(self) -> None:
        host = self.ctx.download(self.param, op="H-6", batch_sharded=False)[0, : self.dim]
        self._param().data.copy_(host)

    def refresh(self) -> None:
        """Recompute std, var and the log-prob constants from the parameter (call after every parameter change)."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        if self.std_type == "scalar":
            self.std = ttnn.clamp(self.param, min=self.std_range[0], max=self.std_range[1])
            self.log_std = ttnn.log(self.std)
        else:
            self.log_std = ttnn.clamp(self.param, min=self.log_std_range[0], max=self.log_std_range[1])
            self.std = ttnn.exp(self.log_std)
        self.var = ttnn.square(self.std)
        self.two_var = ttnn.multiply(self.var, 2.0)
        self.log_prob_const = ttnn.add(self.log_std, HALF_LOG_2PI, **_acc(ttnn))  # log(std) + 0.5 log(2 pi), per action column
        # entropy per sample = sum_real (0.5 + 0.5 log(2 pi) + log std): identical for every row
        ent = ttnn.add(self.log_std, 0.5 + HALF_LOG_2PI, **_acc(ttnn))
        self.entropy_dev = ttnn.sum(ttnn.multiply(ent, self.mask), dim=-1, keepdim=True, compute_kernel_config=ctx.compute_cfg)
        # rows used by the backward pass (all masked, so padded action columns never receive a gradient)
        self.inv_std = ttnn.multiply(ttnn.reciprocal(self.std), self.mask)
        self.inv_var = ttnn.multiply(ttnn.reciprocal(self.var), self.mask)
        self.inv_std3 = ttnn.multiply(self.inv_var, self.inv_std)
        # d(param) = d(std) * dstd/dparam * [param inside the clamp range] (torch clamp backward is inclusive)
        if self.std_type == "scalar":
            lo, hi = self.std_range
            inside = ttnn.multiply(ttnn.ge(self.param, lo), ttnn.le(self.param, hi))
            self.param_gate = ttnn.multiply(inside, self.mask)
        else:
            lo, hi = self.log_std_range
            inside = ttnn.multiply(ttnn.ge(self.param, lo), ttnn.le(self.param, hi))
            self.param_gate = ttnn.multiply(ttnn.multiply(inside, self.std), self.mask)  # dstd/dlog_std = std

    # -- compute ----------------------------------------------------------------------------------------------------------
    def sample(self, mean: Any, noise: Any) -> Any:
        """mean + std * noise (the noise comes from the host generator in parity mode, allow-list H-4)."""
        ttnn = self.ctx.ttnn
        return ttnn.add(ttnn.multiply(noise, self.std), mean, **_acc(ttnn))

    def log_prob(self, actions: Any, mean: Any) -> Any:
        """Sum over the real action dims of  -(a - mean)^2 / (2 var) - log std - 0.5 log(2 pi); returns [rows, 1]."""
        ttnn = self.ctx.ttnn
        diff = ttnn.subtract(actions, mean, **_acc(ttnn))
        term = ttnn.subtract(ttnn.neg(ttnn.divide(ttnn.square(diff), self.two_var, **_acc(ttnn))), self.log_prob_const, **_acc(ttnn))
        return ttnn.sum(ttnn.multiply(term, self.mask), dim=-1, keepdim=True, compute_kernel_config=self.ctx.compute_cfg)

    def log_prob_from_diff(self, diff: Any) -> Any:
        """`log_prob` given diff = actions - mean (the backward pass needs diff as well)."""
        ttnn = self.ctx.ttnn
        term = ttnn.subtract(ttnn.neg(ttnn.divide(ttnn.square(diff), self.two_var, **_acc(ttnn))), self.log_prob_const, **_acc(ttnn))
        return ttnn.sum(ttnn.multiply(term, self.mask), dim=-1, keepdim=True, compute_kernel_config=self.ctx.compute_cfg)

    def host_std(self) -> Any:
        return self.ctx.download(self.std, op="H-5", batch_sharded=False)[0, : self.dim]


class TTMLPModel:
    """Upstream `MLPModel` on device: obs groups -> (optional) empirical normalization -> MLP [-> Gaussian head]."""

    def __init__(self, ctx: TTContext, mirror: Any) -> None:
        self.ctx, self.mirror = ctx, mirror
        self.obs_groups = list(mirror.obs_groups)
        self.obs_dim = mirror.obs_dim
        self.obs_dim_p = pad_up(self.obs_dim)
        self.normalizer = TTEmpiricalNormalization(ctx, mirror.obs_normalizer, self.obs_dim) if mirror.obs_normalization else None
        self.mlp = TTMLP(ctx, mirror.mlp)
        if self.mlp.in_features != self.obs_dim:
            raise NotImplementedError("only plain MLP models (mlp input == concatenated obs) are in the v1 scope")
        self.distribution = TTGaussianDistribution(ctx, mirror.distribution) if mirror.distribution is not None else None
        self.output_dim = self.distribution.dim if self.distribution is not None else self.mlp.out_features
        self.output_dim_p = pad_up(self.output_dim)

    # -- host staging (H-3 concat + H-1 upload) ---------------------------------------------------------------------------
    def obs_to_device(self, obs: Any) -> Any:
        ctx = self.ctx
        ctx.check_batch(obs[self.obs_groups[0]].shape[0], "number of environments / samples")
        buf = stage_obs(obs, self.obs_groups, self.obs_dim_p)
        ctx.host_ops.count("H-3", buf.numel() * 4)
        return ctx.upload(buf, op="H-1", batch_sharded=True)

    # -- compute ----------------------------------------------------------------------------------------------------------
    def latent(self, obs_dev: Any) -> Any:
        return self.normalizer.forward(obs_dev) if self.normalizer is not None else obs_dev

    def forward(self, obs_dev: Any) -> Any:
        """MLP output on device ([rows, out_p]): the distribution mean for an actor, the value (column 0) for a critic."""
        return self.mlp.forward(self.latent(obs_dev))

    def forward_train(self, obs_dev: Any) -> tuple[Any, list]:
        return self.mlp.forward_train(self.latent(obs_dev))

    def update_normalization(self, obs_dev: Any, rows: int) -> None:
        if self.normalizer is not None:
            self.normalizer.update(obs_dev, rows)

    def set_training(self, training: bool) -> None:
        if self.normalizer is not None:
            self.normalizer.training = training

    # -- mirror synchronisation (H-6) -------------------------------------------------------------------------------------
    def sync_from_torch(self) -> None:
        if self.normalizer is not None:
            self.normalizer.sync_from_torch()
        self.mlp.sync_from_torch()
        if self.distribution is not None:
            self.distribution.sync_from_torch()

    def sync_to_torch(self) -> None:
        if self.normalizer is not None:
            self.normalizer.sync_to_torch()
        self.mlp.sync_to_torch()
        if self.distribution is not None:
            self.distribution.sync_to_torch()
