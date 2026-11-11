# SPDX-License-Identifier: BSD-3-Clause
"""PPO losses and their hand-written gradients on the device (BRIEF Phase 3, ADR-0001 option A).

Everything here mirrors `rsl_rl.algorithms.ppo.PPO.update` of the pinned tag, including the places where torch's autograd has special rules:
  * `torch.max(a, b)` (the clipped surrogate and the clipped value loss) sends the gradient to the larger operand and splits it 50/50 on an
    exact tie;
  * `clamp` passes the gradient where min <= x <= max (inclusive);
  * the entropy bonus enters through the std only (every row has the same entropy);
  * all denominators are the REAL number of samples (a mini-batch is a multiple of 32 rows by construction; nothing is padded in rows).
Padded action columns are masked in every sum, so they never contribute a loss or a gradient. All tensors are device tensors; nothing is
downloaded here (the loss values stay on the device until the end of `update()`, allow-list H-5).
"""

from __future__ import annotations

from typing import Any

from rsl_rl_tt.modules import TTGaussianDistribution
from rsl_rl_tt.tt_common import TTContext


class PPOLosses:
    def __init__(self, ctx: TTContext, *, clip_param: float, value_loss_coef: float, entropy_coef: float, use_clipped_value_loss: bool) -> None:
        self.ctx = ctx
        self.clip, self.c_v, self.c_e, self.clipped_value = clip_param, value_loss_coef, entropy_coef, use_clipped_value_loss

    def _mean_rows(self, x: Any) -> Any:
        return self.ctx.ttnn.mean(x, dim=0, keepdim=True, compute_kernel_config=self.ctx.compute_cfg)

    # -- actor ------------------------------------------------------------------------------------------------------------
    def actor(self, dist: TTGaussianDistribution, mean: Any, actions: Any, old_logp: Any, adv: Any, rows: int) -> dict:
        """Surrogate loss and the gradients w.r.t. the distribution mean (-> MLP backward) and the std parameter.

        Returns {"logp", "surrogate_loss" [1,1], "d_mean" [rows, A_p], "d_param" [1, A_p] or None}."""
        ttnn = self.ctx.ttnn
        cfg = self.ctx.compute_cfg
        diff = ttnn.subtract(actions, mean)
        logp = dist.log_prob_from_diff(diff)
        ratio = ttnn.exp(ttnn.subtract(logp, old_logp))
        neg_adv = ttnn.neg(adv)
        s1 = ttnn.multiply(neg_adv, ratio)
        lo, hi = 1.0 - self.clip, 1.0 + self.clip
        s2 = ttnn.multiply(neg_adv, ttnn.clamp(ratio, min=lo, max=hi))
        surrogate_loss = self._mean_rows(ttnn.maximum(s1, s2))
        # d max(s1, s2) / d ratio = -adv * gate, gate = 1 where the unclipped branch wins or the ratio is inside [lo, hi] (tie -> both branches), else 0
        inside = ttnn.multiply(ttnn.ge(ratio, lo), ttnn.le(ratio, hi))
        gate = ttnn.maximum(inside, ttnn.gt(s1, s2))
        d_logp = ttnn.multiply(ttnn.multiply(ttnn.multiply(neg_adv, gate), ratio), 1.0 / rows)  # [rows, 1]: dL/dlogp per sample
        d_mean = ttnn.multiply(ttnn.multiply(diff, dist.inv_var), d_logp)  # dlogp/dmean = diff / var (masked row)
        d_param = None
        if dist.learn_std:
            s1_cols = ttnn.sum(ttnn.multiply(ttnn.square(diff), d_logp), dim=0, keepdim=True, compute_kernel_config=cfg)  # sum_i dL/dlogp_i diff^2
            s0 = ttnn.sum(d_logp, dim=0, keepdim=True, compute_kernel_config=cfg)  # sum_i dL/dlogp_i  -> [1, 1]
            # dlogp/dstd = diff^2/std^3 - 1/std ; entropy bonus: -c_e * d(mean entropy)/dstd = -c_e / std
            d_std = ttnn.subtract(ttnn.multiply(s1_cols, dist.inv_std3), ttnn.multiply(ttnn.add(s0, self.c_e), dist.inv_std))
            d_param = ttnn.multiply(d_std, dist.param_gate)
        return {"logp": logp, "surrogate_loss": surrogate_loss, "d_mean": d_mean, "d_param": d_param}

    # -- critic -----------------------------------------------------------------------------------------------------------
    def critic(self, value: Any, old_value: Any, returns: Any, rows: int) -> dict:
        """Value loss and its gradient w.r.t. the critic output column: {"value_loss" [1,1], "d_value" [rows, 1] (already x c_v / rows)}."""
        ttnn = self.ctx.ttnn
        err = ttnn.subtract(value, returns)
        t1 = ttnn.square(err)
        if self.clipped_value:
            delta = ttnn.subtract(value, old_value)
            v_clipped = ttnn.add(old_value, ttnn.clamp(delta, min=-self.clip, max=self.clip))
            err_c = ttnn.subtract(v_clipped, returns)
            t2 = ttnn.square(err_c)
            value_loss = self._mean_rows(ttnn.maximum(t1, t2))
            # torch.max backward: gradient to the larger branch, 50/50 on a tie; the clipped branch only passes it inside the clamp range
            gt1, gt2 = ttnn.gt(t1, t2), ttnn.gt(t2, t1)
            tie = ttnn.eq(t1, t2)
            w1 = ttnn.add(gt1, ttnn.multiply(tie, 0.5))
            w2 = ttnn.add(gt2, ttnn.multiply(tie, 0.5))
            inside = ttnn.multiply(ttnn.ge(delta, -self.clip), ttnn.le(delta, self.clip))
            d = ttnn.add(ttnn.multiply(w1, ttnn.multiply(err, 2.0)), ttnn.multiply(ttnn.multiply(w2, inside), ttnn.multiply(err_c, 2.0)))
        else:
            value_loss = self._mean_rows(t1)
            d = ttnn.multiply(err, 2.0)
        return {"value_loss": value_loss, "d_value": ttnn.multiply(d, self.c_v / rows)}

    # -- KL (adaptive learning rate; no gradient) ---------------------------------------------------------------------------
    def kl_mean(self, dist: TTGaussianDistribution, old_mean: Any, old_var: Any, old_log_std: Any, mean: Any) -> Any:
        """mean over rows of KL(N(old_mean, old_std) || N(mean, std)) summed over the real action dims: a [1, 1] device scalar."""
        ttnn = self.ctx.ttnn
        cfg = self.ctx.compute_cfg
        var_ratio = ttnn.multiply(old_var, dist.inv_var)  # rows of the (masked) 1 / new var; padded columns are 0 -> handled by the mask below
        t1 = ttnn.multiply(ttnn.square(ttnn.subtract(old_mean, mean)), dist.inv_var)
        log_var_ratio = ttnn.multiply(ttnn.subtract(old_log_std, dist.log_std), 2.0)
        # 0.5 * (var_ratio + t1 - 1 - log(var_ratio)), padded columns masked out
        term = ttnn.multiply(ttnn.multiply(ttnn.subtract(ttnn.add(ttnn.add(var_ratio, t1), -1.0), log_var_ratio), 0.5), dist.mask)
        return self._mean_rows(ttnn.sum(term, dim=-1, keepdim=True, compute_kernel_config=cfg))
