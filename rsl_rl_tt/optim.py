# SPDX-License-Identifier: BSD-3-Clause
"""Device-side optimizer pieces (BRIEF Phase 3): Adam, per-model global grad-norm clipping (DECISIONS C-1), and the KL-adaptive learning rate.

Every per-iteration scalar (learning rate, the two Adam bias corrections) is a device tensor, uploaded through `TTContext.upload_scalar`
(allow-list H-8) or computed on the device, so nothing is frozen into a trace later (ADR-0001). The formulas are those of
`torch.optim.Adam` (single-tensor path, fp32, no weight decay / amsgrad) and of `torch.nn.utils.clip_grad_norm_`.
"""

from __future__ import annotations

from typing import Any

from rsl_rl_tt.tt_common import TTContext


class TTAdam:
    """Adam with fp32 moments on the device. `step()` is functional: it returns the new parameter tensors for the caller to store."""

    def __init__(self, ctx: TTContext, params: list, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        self.ctx = ctx
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # host step counter (bookkeeping only)
        ttnn = ctx.ttnn
        self.m = [ttnn.zeros_like(p) for p in params]
        self.v = [ttnn.zeros_like(p) for p in params]
        # the two bias-correction scalars are PERSISTENT device tensors written by `prepare_step()` before every (replayed) step
        self.inv_bc1 = ctx.upload_scalar(1.0, op="H-8")
        self.inv_sqrt_bc2 = ctx.upload_scalar(1.0, op="H-8")

    def load_state(self, m: list, v: list, t: int) -> None:
        """Replace the moments (device tensors, same order and shapes as the parameters) and the step counter (checkpoint resume)."""
        assert len(m) == len(self.m) and len(v) == len(self.v)
        for k in range(len(m)):
            self.m[k] = self.ctx.put(self.m[k], m[k])
            self.v[k] = self.ctx.put(self.v[k], v[k])
        self.t = int(t)

    def prepare_step(self) -> None:
        """Advance the host step counter and write the step's two bias-correction scalars into their persistent device tensors (H-8)."""
        self.t += 1
        self.ctx.upload_scalar_into(self.inv_bc1, 1.0 / (1.0 - self.beta1**self.t), op="H-8")
        self.ctx.upload_scalar_into(self.inv_sqrt_bc2, 1.0 / (1.0 - self.beta2**self.t) ** 0.5, op="H-8")

    def step(self, params: list, grads: list, lr: Any) -> list:
        """One Adam step for all parameters (`prepare_step()` must have run). `lr` is a [1, 1] device tensor. Returns the NEW parameter values;
        the moments are updated in place."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        neg_step_size = ttnn.neg(ttnn.multiply(lr, self.inv_bc1))  # -lr / bias_correction1
        new_params = []
        for k, (p, g) in enumerate(zip(params, grads)):
            m = ttnn.add(self.m[k], ttnn.multiply(ttnn.subtract(g, self.m[k]), 1.0 - self.beta1))  # exp_avg.lerp_(grad, 1 - beta1)
            v = ttnn.add(ttnn.multiply(self.v[k], self.beta2), ttnn.multiply(ttnn.square(g), 1.0 - self.beta2))  # exp_avg_sq.mul_(beta2).addcmul_(g, g, 1 - beta2)
            denom = ttnn.add(ttnn.multiply(ttnn.sqrt(v), self.inv_sqrt_bc2), self.eps)  # sqrt(v) / sqrt(bc2) + eps
            new_params.append(ttnn.add(p, ttnn.multiply(ttnn.divide(m, denom), neg_step_size)))  # p.addcdiv_(m, denom, value=-step_size)
            self.m[k] = ctx.put(self.m[k], m)
            self.v[k] = ctx.put(self.v[k], v)
        return new_params


def clip_grad_norm(ctx: TTContext, grads: list, max_norm: float) -> tuple[list, Any]:
    """`torch.nn.utils.clip_grad_norm_` on a list of device gradients: returns (clipped grads, total norm [1, 1] device tensor).

    total = sqrt(sum_k ||g_k||^2); coef = min(max_norm / (total + 1e-6), 1); g_k *= coef. (One call per model: actor and critic are clipped
    separately, exactly like upstream; DECISIONS C-1.)"""
    ttnn = ctx.ttnn
    cfg = ctx.compute_cfg
    total_sq = None
    for g in grads:
        sq = ttnn.sum(ttnn.square(g), dim=(0, 1), keepdim=True, compute_kernel_config=cfg)  # [1, 1]
        total_sq = sq if total_sq is None else ttnn.add(total_sq, sq)
    total = ttnn.sqrt(total_sq)
    coef = ttnn.minimum(ttnn.multiply(ttnn.reciprocal(ttnn.add(total, 1e-6)), max_norm), 1.0)
    return [ttnn.multiply(g, coef) for g in grads], total


class TTAdaptiveLR:
    """The KL-driven learning-rate rule of upstream PPO.update, evaluated on the device after every mini-batch forward:

        kl > 2 desired_kl           -> lr = max(1e-5, lr / 1.5)
        0 < kl < desired_kl / 2     -> lr = min(1e-2, lr * 1.5)

    `schedule == "fixed"` (or desired_kl None) keeps the learning rate constant. The value lives in a [1, 1] device tensor."""

    LR_MIN, LR_MAX, FACTOR = 1e-5, 1e-2, 1.5

    def __init__(self, ctx: TTContext, learning_rate: float, desired_kl: float | None, schedule: str) -> None:
        self.ctx = ctx
        self.desired_kl = desired_kl
        self.adaptive = desired_kl is not None and schedule == "adaptive"
        self.lr = ctx.upload_scalar(learning_rate, op="H-8")  # persistent: updated in place by `update()` / `set()`

    def set(self, learning_rate: float) -> None:
        self.lr = self.ctx.upload_scalar_into(self.lr, learning_rate, op="H-8")

    def update(self, kl_mean: Any) -> Any:
        if not self.adaptive:
            return self.lr
        ttnn = self.ctx.ttnn
        lr = self.lr
        lr_down = ttnn.maximum(ttnn.multiply(lr, 1.0 / self.FACTOR), self.LR_MIN)
        lr_up = ttnn.minimum(ttnn.multiply(lr, self.FACTOR), self.LR_MAX)
        too_big = ttnn.gt(kl_mean, 2.0 * self.desired_kl)
        small = ttnn.multiply(ttnn.lt(kl_mean, self.desired_kl / 2.0), ttnn.gt(kl_mean, 0.0))
        self.lr = self.ctx.put(self.lr, ttnn.where(too_big, lr_down, ttnn.where(small, lr_up, lr)))
        return self.lr
