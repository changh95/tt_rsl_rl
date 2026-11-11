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

from rsl_rl_tt.tt_common import TTContext, linear_from_host, linear_to_host, mask_row, pad_heads, pad_up, pad_vector, stage_obs, unpad_heads

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
        self._mean = ctx.upload_into(getattr(self, "_mean", None), mean, op="H-6")
        self._var = ctx.upload_into(getattr(self, "_var", None), var, op="H-6")
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
        ctx = self.ctx
        ttnn = ctx.ttnn
        self._std = ctx.put(getattr(self, "_std", None), ttnn.sqrt(self._var))
        self._std_eps = ctx.put(getattr(self, "_std_eps", None), ttnn.add(self._std, self.eps, **_acc(ttnn)))  # (std + eps)

    # -- compute ----------------------------------------------------------------------------------------------------------
    def forward(self, x: Any) -> Any:
        ttnn = self.ctx.ttnn
        return ttnn.divide(ttnn.subtract(x, self._mean, **_acc(ttnn)), self._std_eps, **_acc(ttnn))

    def update(self, x: Any, rows: int) -> None:
        """Learn the statistics of the batch `x` ([rows, dim_p], batch-sharded); same recurrence as upstream `EmpiricalNormalization.update`.

        `rows` is the number of rows on ONE chip. On several chips the statistics are GLOBAL, like upstream under torch.distributed (normalization.py: the global
        mean first, then the local variances combined around it): the count is `rows * n_chips`, the chips' means are averaged and the variance is the average of
        `var_c + (mean_c - mean)^2` (all chips hold the same number of rows); the averages are the fp32 gather + sum of ADR-0004 A-1, so every chip ends with the
        same statistics bit for bit."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        if not self.training or (self.until is not None and self.count >= self.until):
            return
        total = rows * ctx.n_chips
        self.count += total
        rate = ctx.upload_scalar(total / self.count, op="H-8")  # a device tensor, so a trace does not freeze it
        cfg = ctx.compute_cfg
        mean_x = ttnn.mean(x, dim=0, keepdim=True, compute_kernel_config=cfg)
        centered = ttnn.subtract(x, mean_x, **_acc(ttnn))
        var_x = ttnn.mean(ttnn.square(centered), dim=0, keepdim=True, compute_kernel_config=cfg)  # two-pass variance (ttnn.var is 2e-4 off)
        if ctx.n_chips > 1:
            global_mean = ctx.mean_over_chips(mean_x)
            var_x = ctx.mean_over_chips(ttnn.add(var_x, ttnn.square(ttnn.subtract(mean_x, global_mean))))
            mean_x = global_mean
        delta_mean = ttnn.subtract(mean_x, self._mean, **_acc(ttnn))
        new_mean = ttnn.add(self._mean, ttnn.multiply(delta_mean, rate), **_acc(ttnn))
        # var += rate * (var_x - var + delta_mean * (mean_x - new_mean))
        inner = ttnn.add(ttnn.subtract(var_x, self._var, **_acc(ttnn)), ttnn.multiply(delta_mean, ttnn.subtract(mean_x, new_mean, **_acc(ttnn))), **_acc(ttnn))
        self._var = ctx.put(self._var, ttnn.add(self._var, ttnn.multiply(inner, rate), **_acc(ttnn)))  # in place: the act trace reads these buffers
        self._mean = ctx.put(self._mean, new_mean)
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
    """y = x @ W + b on padded tensors; W is stored as [in_p, out_p].

    `heads` > 1 (the output layer of a heteroscedastic actor: mean | std): the out_features are `heads` blocks of equal size and every block is padded on its own
    (out_p = heads * pad_up(out_features / heads)), so the blocks are tile-aligned column slices of the output."""

    def __init__(self, ctx: TTContext, mirror: Any, heads: int = 1) -> None:
        self.ctx, self.mirror, self.heads = ctx, mirror, heads
        self.in_features, self.out_features = mirror.in_features, mirror.out_features
        assert self.out_features % heads == 0, "the output blocks of a multi-head layer have the same size"
        self.in_p, self.out_p = pad_up(self.in_features), heads * pad_up(self.out_features // heads)
        self.sync_from_torch()

    # torch layout <-> padded device layout of this layer (also used for the Adam moments of a checkpoint)
    def weight_to_host(self, w: Any) -> Any:
        """torch weight layout [out, in] -> padded [in_p, out_p]."""
        return pad_heads(w.t(), self.in_p, self.out_p, self.heads)

    def bias_to_host(self, b: Any) -> Any:
        return pad_heads(b.reshape(1, -1), 1, self.out_p, self.heads)

    def weight_from_host(self, w: Any) -> Any:
        return unpad_heads(w, self.in_features, self.out_features, self.heads).t().contiguous()

    def bias_from_host(self, b: Any) -> Any:
        return unpad_heads(b, 1, self.out_features, self.heads)[0].contiguous()

    def sync_from_torch(self) -> None:
        w, b = linear_to_host(self.mirror.weight, self.mirror.bias, self.in_p, self.out_p, self.heads)
        self.w = self.ctx.upload_into(getattr(self, "w", None), w, op="H-6")
        self.b = self.ctx.upload_into(getattr(self, "b", None), b, op="H-6")

    def sync_to_torch(self) -> None:
        w = self.ctx.download(self.w, op="H-6", batch_sharded=False)
        b = self.ctx.download(self.b, op="H-6", batch_sharded=False)
        w_t, b_t = linear_from_host(w, b, self.out_features, self.in_features, self.heads)
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
        # a trailing `Unflatten(-1, (heads, D))` (upstream's MLP for a distribution whose input_dim is a list, e.g. the heteroscedastic Gaussian: [2, A]) splits the
        # last Linear's output into `heads` blocks of D; on the device that is the layout of `TTLinear(heads=...)`
        self.heads = 1
        if len(mods) >= 2 and type(mods[-1]).__name__ == "Unflatten":
            size = tuple(mods[-1].unflattened_size)
            if type(mods[-2]).__name__ != "Linear" or mods[-1].dim not in (-1, 1) or len(size) != 2:
                raise NotImplementedError(f"MLP output reshape {mods[-1]} is outside the supported scope (Linear + Unflatten(-1, (heads, D)))")
            self.heads = size[0]
            mods = mods[:-1]
        for i, mod in enumerate(mods):
            if type(mod).__name__ == "Linear":
                last = i == len(mods) - 1
                self.layers.append(TTLinear(ctx, mod, heads=self.heads if last else 1))
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
                # d elu(z)/dz = 1 for z > 0 and exp(z) otherwise = exp(min(z, 0)) (torch: also 1 at z == 0). `ttnn.elu_bw` takes 15.5 ms per call at 393216 x 512
                # (Phase 5 op profile: 29 % of a whole update step, 5x its own forward); this two-op form takes ~1 ms.
                g = ttnn.multiply(g, ttnn.exp(ttnn.minimum(z, 0.0)))
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
        """Write new parameter values into the persistent weight/bias tensors in place."""
        ctx = self.ctx
        for k, layer in enumerate(self.layers):
            layer.w, layer.b = ctx.put(layer.w, values[2 * k]), ctx.put(layer.b, values[2 * k + 1])

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


class _GaussianOps:
    """Sampling and log-probability of a diagonal Gaussian from the tensors `std`, `two_var`, `log_prob_const` and `mask` of the object (each [1, A_p] for the
    state-independent distribution, [rows, A_p] for the per-row state of the heteroscedastic one: the same elementwise code broadcasts either way)."""

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


class TTGaussianDistribution(_GaussianOps):
    """State-independent diagonal Gaussian (std parameter [1, A_p]); padded action columns are masked out of log_prob / entropy."""

    per_row = False  # std is the same for every row (see TTHeteroscedasticGaussian)

    def __init__(self, ctx: TTContext, mirror: Any) -> None:
        if type(mirror).__name__ != "GaussianDistribution":
            raise NotImplementedError(f"{type(mirror).__name__} is not a state-independent GaussianDistribution (scalar/log std)")
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
        self.param = self.ctx.upload_into(getattr(self, "param", None), pad_vector(self._param().detach(), self.dim_p, fill), op="H-6")
        self.refresh()

    def sync_to_torch(self) -> None:
        host = self.ctx.download(self.param, op="H-6", batch_sharded=False)[0, : self.dim]
        self._param().data.copy_(host)

    def refresh(self) -> None:
        """Recompute std, var and the log-prob / backward rows from the parameter (call after every parameter change). The derived tensors are
        PERSISTENT: they are written in place so that a captured trace keeps reading the current values."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        put = lambda name, value: setattr(self, name, ctx.put(getattr(self, name, None), value))
        if self.std_type == "scalar":
            put("std", ttnn.clamp(self.param, min=self.std_range[0], max=self.std_range[1]))
            put("log_std", ttnn.log(self.std))
        else:
            put("log_std", ttnn.clamp(self.param, min=self.log_std_range[0], max=self.log_std_range[1]))
            put("std", ttnn.exp(self.log_std))
        put("var", ttnn.square(self.std))
        put("two_var", ttnn.multiply(self.var, 2.0))
        put("log_prob_const", ttnn.add(self.log_std, HALF_LOG_2PI, **_acc(ttnn)))  # log(std) + 0.5 log(2 pi), per action column
        # entropy per sample = sum_real (0.5 + 0.5 log(2 pi) + log std): identical for every row
        ent = ttnn.add(self.log_std, 0.5 + HALF_LOG_2PI, **_acc(ttnn))
        put("entropy_dev", ttnn.sum(ttnn.multiply(ent, self.mask), dim=-1, keepdim=True, compute_kernel_config=ctx.compute_cfg))
        # rows used by the backward pass (all masked, so padded action columns never receive a gradient)
        put("inv_std", ttnn.multiply(ttnn.reciprocal(self.std), self.mask))
        put("inv_var", ttnn.multiply(ttnn.reciprocal(self.var), self.mask))
        put("inv_std3", ttnn.multiply(self.inv_var, self.inv_std))
        # d(param) = d(std) * dstd/dparam * [param inside the clamp range] (torch clamp backward is inclusive)
        if self.std_type == "scalar":
            lo, hi = self.std_range
            inside = ttnn.multiply(ttnn.ge(self.param, lo), ttnn.le(self.param, hi))
            put("param_gate", ttnn.multiply(inside, self.mask))
        else:
            lo, hi = self.log_std_range
            inside = ttnn.multiply(ttnn.ge(self.param, lo), ttnn.le(self.param, hi))
            put("param_gate", ttnn.multiply(ttnn.multiply(inside, self.std), self.mask))  # dstd/dlog_std = std

    def host_std(self) -> Any:
        return self.ctx.download(self.std, op="H-5", batch_sharded=False)[0, : self.dim]


class TTHeteroscedasticGaussian:
    """State-dependent diagonal Gaussian (upstream `HeteroscedasticGaussianDistribution`): the actor's output layer has two heads, the mean in output columns
    0..A_p and the std ("scalar") or log std ("log") in A_p..2 A_p, clamped to the std range. There is no parameter of its own: the per-row quantities of every
    forward pass are computed by `from_output` and handed around as a `HeteroState`. Padded action columns have std 1 and are masked out of every sum."""

    per_row = True
    learn_std = False  # no std parameter: the std head is part of the MLP

    def __init__(self, ctx: TTContext, mirror: Any) -> None:
        if type(mirror).__name__ != "HeteroscedasticGaussianDistribution":
            raise NotImplementedError(f"{type(mirror).__name__} is not a HeteroscedasticGaussianDistribution")
        self.ctx, self.mirror = ctx, mirror
        self.dim = mirror.output_dim
        self.dim_p = pad_up(self.dim)
        self.std_type = mirror.std_type
        self.std_range = tuple(mirror.std_range)
        self.log_std_range = tuple(mirror.log_std_range)
        self.mask = ctx.upload(mask_row(self.dim, self.dim_p), op="H-6", batch_sharded=False)
        self.pad_one = ctx.upload(1.0 - mask_row(self.dim, self.dim_p), op="H-6", batch_sharded=False)  # 1 in the padded columns: their std

    def sync_from_torch(self) -> None:
        pass  # no parameters of its own (they are the MLP's last layer)

    def sync_to_torch(self) -> None:
        pass

    def refresh(self) -> None:
        pass

    def from_output(self, out: Any) -> tuple[Any, "HeteroState"]:
        """Split the actor's output [rows, 2 A_p] into the mean [rows, A_p] and the per-row distribution state."""
        return out[:, : self.dim_p], HeteroState(self, out[:, self.dim_p : 2 * self.dim_p])


class HeteroState(_GaussianOps):
    """The per-row quantities of a heteroscedastic Gaussian for one batch of rows, computed on first use (the act step needs only std / two_var /
    log_prob_const, the update step the rest). All tensors are [rows, A_p]; `gate` is what the std head's raw output gradient is multiplied with
    (torch's clamp passes the gradient where lo <= raw <= hi, inclusive; for the "log" type the chain rule through exp adds the factor std)."""

    per_row = True
    learn_std = False

    def __init__(self, dist: TTHeteroscedasticGaussian, raw: Any) -> None:
        self.dist, self.ctx, self.raw, self.mask = dist, dist.ctx, raw, dist.mask
        self._cache: dict = {}

    def _get(self, key: str, make) -> Any:
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    @property
    def std(self) -> Any:
        def make() -> Any:
            ttnn, d = self.ctx.ttnn, self.dist
            if d.std_type == "scalar":
                s = ttnn.clamp(self.raw, min=d.std_range[0], max=d.std_range[1])
                return ttnn.add(ttnn.multiply(s, d.mask), d.pad_one)
            return ttnn.exp(self.log_std)

        return self._get("std", make)

    @property
    def log_std(self) -> Any:
        def make() -> Any:
            ttnn, d = self.ctx.ttnn, self.dist
            if d.std_type == "scalar":
                return ttnn.log(self.std)
            return ttnn.multiply(ttnn.clamp(self.raw, min=d.log_std_range[0], max=d.log_std_range[1]), d.mask)

        return self._get("log_std", make)

    @property
    def var(self) -> Any:
        return self._get("var", lambda: self.ctx.ttnn.square(self.std))

    @property
    def two_var(self) -> Any:
        return self._get("two_var", lambda: self.ctx.ttnn.multiply(self.var, 2.0))

    @property
    def log_prob_const(self) -> Any:
        return self._get("log_prob_const", lambda: self.ctx.ttnn.add(self.log_std, HALF_LOG_2PI, **_acc(self.ctx.ttnn)))

    @property
    def inv_std(self) -> Any:
        return self._get("inv_std", lambda: self.ctx.ttnn.multiply(self.ctx.ttnn.reciprocal(self.std), self.mask))

    @property
    def inv_var(self) -> Any:
        return self._get("inv_var", lambda: self.ctx.ttnn.multiply(self.ctx.ttnn.reciprocal(self.var), self.mask))

    @property
    def inv_std3(self) -> Any:
        return self._get("inv_std3", lambda: self.ctx.ttnn.multiply(self.inv_var, self.inv_std))

    @property
    def gate(self) -> Any:
        def make() -> Any:
            ttnn, d = self.ctx.ttnn, self.dist
            lo, hi = d.std_range if d.std_type == "scalar" else d.log_std_range
            inside = ttnn.multiply(ttnn.ge(self.raw, lo), ttnn.le(self.raw, hi))
            if d.std_type == "log":
                inside = ttnn.multiply(inside, self.std)  # dstd / dlog_std = std
            return ttnn.multiply(inside, d.mask)

        return self._get("gate", make)

    @property
    def entropy_rows(self) -> Any:
        """Entropy of every row (summed over the real action dims): [rows, 1]."""

        def make() -> Any:
            ttnn = self.ctx.ttnn
            ent = ttnn.add(self.log_std, 0.5 + HALF_LOG_2PI, **_acc(ttnn))
            return ttnn.sum(ttnn.multiply(ent, self.mask), dim=-1, keepdim=True, compute_kernel_config=self.ctx.compute_cfg)

        return self._get("entropy_rows", make)

    @property
    def entropy_dev(self) -> Any:
        """Mean entropy over the rows: [1, 1] (upstream: `entropy_batch.mean()`)."""
        return self._get("entropy_dev", lambda: self.ctx.ttnn.mean(self.entropy_rows, dim=0, keepdim=True, compute_kernel_config=self.ctx.compute_cfg))

    def host_std(self) -> Any:
        return self.ctx.download(self.std, op="H-5")[:, : self.dist.dim]


def _make_distribution(ctx: TTContext, mirror: Any) -> Any:
    if mirror is None:
        return None
    name = type(mirror).__name__
    if name == "GaussianDistribution":
        return TTGaussianDistribution(ctx, mirror)
    if name == "HeteroscedasticGaussianDistribution":
        return TTHeteroscedasticGaussian(ctx, mirror)
    raise NotImplementedError(f"{name} is not supported by the TT port (GaussianDistribution, HeteroscedasticGaussianDistribution)")


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
        self.distribution = _make_distribution(ctx, mirror.distribution)
        self.output_dim = self.distribution.dim if self.distribution is not None else self.mlp.out_features
        self.output_dim_p = pad_up(self.output_dim)

    # -- host staging (H-3 concat + H-1 upload) ---------------------------------------------------------------------------
    def obs_to_device(self, obs: Any, into: Any = None) -> Any:
        """H-3 concat + H-1 upload; with `into` the data is written in place into that persistent device tensor (a traced step reads it)."""
        ctx = self.ctx
        ctx.check_batch(obs[self.obs_groups[0]].shape[0], "number of environments / samples")
        buf = stage_obs(obs, self.obs_groups, self.obs_dim_p)
        ctx.host_ops.count("H-3", buf.numel() * 4)
        return ctx.upload_into(into, buf, op="H-1", batch_sharded=True)

    def obs_to_device_rm(self, obs: Any, into: Any = None) -> Any:
        """Like `obs_to_device`, but into a ROW_MAJOR persistent tensor (cheap upload); the traced act step tilizes it on the device."""
        ctx = self.ctx
        ctx.check_batch(obs[self.obs_groups[0]].shape[0], "number of environments / samples")
        buf = stage_obs(obs, self.obs_groups, self.obs_dim_p)
        ctx.host_ops.count("H-3", buf.numel() * 4)
        return ctx.upload_rm_into(into, buf, op="H-1", batch_sharded=True)

    # -- compute ----------------------------------------------------------------------------------------------------------
    def latent(self, obs_dev: Any) -> Any:
        return self.normalizer.forward(obs_dev) if self.normalizer is not None else obs_dev

    def forward(self, obs_dev: Any) -> Any:
        """MLP output on device ([rows, out_p]): the distribution mean for an actor, the value (column 0) for a critic."""
        return self.mlp.forward(self.latent(obs_dev))

    def forward_train(self, obs_dev: Any) -> tuple[Any, list]:
        return self.mlp.forward_train(self.latent(obs_dev))

    def policy_forward(self, obs_dev: Any) -> tuple[Any, Any]:
        """Actor forward pass: (mean [rows, A_p], distribution state). The state is the (state-independent) distribution itself or, for a heteroscedastic
        distribution, the per-row `HeteroState`; both have the attributes the losses use (std, var, log_std, inv_var, ..., log_prob, sample)."""
        out = self.forward(obs_dev)
        if self.distribution.per_row:
            return self.distribution.from_output(out)
        return out, self.distribution

    def policy_forward_train(self, obs_dev: Any) -> tuple[Any, Any, list]:
        out, cache = self.forward_train(obs_dev)
        if self.distribution.per_row:
            mean, state = self.distribution.from_output(out)
            return mean, state, cache
        return out, self.distribution, cache

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
