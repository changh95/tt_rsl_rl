# SPDX-License-Identifier: BSD-3-Clause
"""`TTDistillation`: upstream `Distillation` (a student MLP imitates a frozen teacher MLP by behaviour cloning) with the rollout inference AND the whole update on Tenstorrent hardware
(post-v1 item 8.4, docs/porting/POST_V1.md).

Injection (no upstream patch, the unmodified `DistillationRunner` / `OnPolicyRunner`):

    cfg["algorithm"]["class_name"] = "rsl_rl_tt.distillation:TTDistillation"
    cfg["algorithm"]["tt_cfg"] = {"mesh_shape": [1, 1], "trace": True}       # optional; consumed here, upstream never sees it

`Distillation.construct_algorithm` (inherited) builds the upstream torch student / teacher exactly as upstream does (identical initial weights for the same torch seed); those are the MIRRORS
(device -> mirror sync, allow-list H-6). The teacher's weights come from an RL checkpoint through the ordinary `load()` (upstream's rules: an `actor_state_dict` loads the teacher only) and are
copied to the device after it. What upstream does per iteration, and how it is done here:

* `act`: the student acts stochastically (host noise from the global generator: the stream `Normal.sample()` uses, H-4), the teacher's deterministic action is computed in the same step and
  stays on the device; the student's observations and the teacher's actions of every step are kept in preallocated device slots (nothing else is stored: the loss needs only these).
* `update`: `num_learning_epochs` passes over the steps IN ORDER (no shuffling, no mini-batches: one batch of all environments per step); per step the student's mean action is compared with the
  teacher's (mse or huber loss, mean over all elements); the LOSSES of `gradient_length` consecutive steps are summed before one backward pass (so the gradient is the SUM of the per-step
  gradients, which are all taken at the same parameters), then optional clipping and one Adam step. The step counter runs across epochs; steps left over at the end of the update (when
  `num_learning_epochs x num_steps_per_env` is not divisible by `gradient_length`) are accumulated but never applied, exactly like upstream. On the device: one traced step per time index that
  adds its gradients to persistent accumulators, one traced "apply" (gradient mean over the chips, clip, Adam, zero the accumulators). The student's normalizer learns from the rollout after
  the epochs (the teacher's never does).
* Not supported (refused at construction): recurrent models (item 8.5), a heteroscedastic student / teacher, mixed precision, optimizers other than Adam.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.algorithms import Distillation
from rsl_rl_tt.device import acquire_tray_mesh
from rsl_rl_tt.modules import TTMLPModel
from rsl_rl_tt.optim import TTAdam, clip_grad_norm
from rsl_rl_tt.traced import CAPTURED_TRACES as _CAPTURED_TRACES, TraceMixin
from rsl_rl_tt.tt_common import SectionTimer, TTContext, pad_matrix


class TTDistillationStorage:
    """Device rollout storage of the distillation: per step the student's observations ([N, d_p], batch-sharded) and the teacher's actions ([N, A_p]).

    `persistent=True` (traced): preallocated slots written in place (`preallocate()` before the first capture, T-15). Otherwise a step is stored by reference."""

    def __init__(self, ctx: TTContext, num_envs: int, steps: int, obs_dim_p: int, act_dim_p: int, persistent: bool) -> None:
        self.ctx, self.num_envs, self.steps, self.obs_dim_p, self.act_dim_p, self.persistent = ctx, num_envs, steps, obs_dim_p, act_dim_p, persistent
        self.step = 0
        self._steps: list[dict] = []
        self._slots: list[dict] | None = None

    def preallocate(self) -> None:
        zeros = lambda cols: self.ctx.zeros(self.num_envs, cols, batch_sharded=True)
        self._slots = [dict(obs=zeros(self.obs_dim_p), teacher=zeros(self.act_dim_p)) for _ in range(self.steps)]

    def add(self, obs, teacher) -> None:
        if self.step >= self.steps:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        if self.persistent:
            assert self._slots is not None, "preallocate() must run before the first transition of a persistent storage"
            slot = self._slots[self.step]
            self.ctx.put(slot["obs"], obs)
            self.ctx.put(slot["teacher"], teacher)
            self._steps.append(slot)
        else:
            self._steps.append(dict(obs=obs, teacher=teacher))
        self.step += 1

    def slot(self, t: int) -> dict:
        return self._steps[t] if not self.persistent else self._slots[t]  # type: ignore[index]

    def all_obs(self):
        parts = [s["obs"] for s in self._steps]
        return parts[0] if len(parts) == 1 else self.ctx.ttnn.concat(parts, dim=0)

    def clear(self) -> None:
        self.step = 0
        self._steps = []


class TTDistillation(TraceMixin, Distillation):
    """Distillation on the tray mesh (a single chip until Phase 6)."""

    def __init__(self, student, teacher, storage, *, tt_cfg: dict | None = None, **kwargs) -> None:
        if kwargs.get("multi_gpu_cfg") is not None:
            raise NotImplementedError("rsl_rl's torch.distributed multi-GPU mode is replaced by the mesh data-parallel design (Phase 6)")
        if getattr(student, "is_recurrent", False) or getattr(teacher, "is_recurrent", False):
            raise NotImplementedError("recurrent models are not supported by the TT distillation yet (post-v1 item 8.5)")
        if str(kwargs.get("optimizer", "adam")).lower() != "adam":
            raise NotImplementedError("the TT port implements Adam only (BRIEF 1: other optimizers need approval)")
        if kwargs.get("use_mixed_precision"):
            raise NotImplementedError("mixed precision is outside the scope (fp32 policy, BRIEF 3.4)")
        super().__init__(student, teacher, storage, **kwargs)
        tt_cfg = dict(tt_cfg or {})
        mesh_shape = tuple(tt_cfg.pop("mesh_shape", (1, 1)))
        self.trace_enabled = bool(tt_cfg.pop("trace", False))  # capture / replay the act step and the update's steps (results are bit-identical to eager)
        self.math_fidelity = str(tt_cfg.pop("math_fidelity", "HiFi4"))
        self.replica_check_interval = int(tt_cfg.pop("replica_check_interval", 0))
        self._mesh_kwargs = {k: tt_cfg.pop(k) for k in ("fabric_config", "trace_region_size", "l1_small_size") if k in tt_cfg}
        if tt_cfg:
            raise ValueError(f"unknown tt_cfg keys: {sorted(tt_cfg)}")
        steps = storage.num_transitions_per_env
        if self.trace_enabled:
            # one trace per time index (forward, loss, gradient accumulation) plus the apply and the act step (x2 on several chips: the gradient exchange of the apply)
            self._mesh_kwargs.setdefault("trace_region_size", max(128, 12 * (steps + 2)) * (2 if mesh_shape[0] * mesh_shape[1] > 1 else 1) * 1024 * 1024)
        if mesh_shape[0] * mesh_shape[1] > 1:
            self._mesh_kwargs.setdefault("fabric_config", "FABRIC_1D")
        self.tray = acquire_tray_mesh(mesh_shape, **self._mesh_kwargs)
        if _CAPTURED_TRACES.get(id(self.tray), 0):
            raise RuntimeError("a captured trace is alive on this mesh: construct every algorithm BEFORE the first trace is captured, or call release_traces() on the one that owns them (TTNN_ISSUES T-15)")
        self.ctx = TTContext.create(self.tray.mesh_device, math_fidelity=self.math_fidelity, mesh_shape=self.tray.mesh_shape)
        if self.trace_enabled:
            self.tray.add_close_hook(self.release_traces)
        self.ctx.check_batch(storage.num_envs, "num_envs")
        self.tt_student = TTMLPModel(self.ctx, self._raw_student)
        self.tt_teacher = TTMLPModel(self.ctx, self._raw_teacher)
        for name, model in (("student", self.tt_student), ("teacher", self.tt_teacher)):
            if model.distribution is None:
                raise ValueError(f"the {name} needs a distribution_cfg (GaussianDistribution)")
            if model.distribution.per_row:
                raise NotImplementedError(f"the {name} uses a heteroscedastic distribution: not supported by the TT distillation")
        if self.tt_student.cnn is not None or self.tt_teacher.cnn is not None:
            raise NotImplementedError("image observations (CNNModel) are not supported by the TT distillation yet")
        self.share_obs = self.tt_student.obs_groups == self.tt_teacher.obs_groups
        self.storage = TTDistillationStorage(self.ctx, storage.num_envs, steps, self.tt_student.obs_dim_p, self.tt_teacher.output_dim_p, persistent=self.trace_enabled)
        self.loss_type = "huber" if self.loss_fn is torch.nn.functional.huber_loss else "mse"
        self._pending: dict | None = None
        self._traces: dict = {}
        self._buf: dict = {}
        # PERSISTENT-STATE RULE (T-15): everything a trace keeps across replays is allocated before the first capture: Adam moments, the gradient and loss accumulators, the rollout slots
        params = self.tt_student.mlp.parameters()
        self.adam = TTAdam(self.ctx, params)
        self.lr = self.ctx.upload_scalar(self.learning_rate, op="H-8")
        ttnn = self.ctx.ttnn
        self._acc_grads = [ttnn.zeros_like(p) for p in params]
        self._acc_loss = self.ctx.upload_scalar(0.0, op="H-8")
        if self.trace_enabled:
            self.storage.preallocate()
        self.timer = SectionTimer(self.ctx, enabled=False)
        self._weights_stale = False
        self._optimizer_stale = False
        self._refresh_std_mirror()

    # -- the student's std (a state-independent parameter that distillation never trains) -----------------------------------------
    def _refresh_std_mirror(self) -> None:
        """The runner reads `get_policy().output_std` every iteration: the mirror's distribution needs a current std."""
        std = self.tt_student.distribution.host_std()
        self._raw_student.distribution._distribution = Normal(torch.zeros(1, std.numel()), std.contiguous())

    # -- mirror synchronisation -----------------------------------------------------------------------------------------------------
    def sync_to_torch(self, optimizer: bool = False) -> None:
        if self._weights_stale:
            self.tt_student.sync_to_torch()
            self._weights_stale = False
        if optimizer and self._optimizer_stale:
            self._optimizer_state_to_torch()
            self._optimizer_stale = False

    def sync_from_torch(self) -> None:
        """The torch models / optimizer (after `load()`) -> the device: student, teacher, Adam moments (H-6)."""
        self.tt_student.sync_from_torch()
        self.tt_teacher.sync_from_torch()
        self._optimizer_state_from_torch()
        self._refresh_std_mirror()
        self._weights_stale = self._optimizer_stale = False

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        self.sync_to_torch(optimizer=True)
        out = super().load(loaded_dict, load_cfg, strict)
        self.sync_from_torch()
        return out

    def _param_slots(self) -> list[tuple]:
        """For every parameter of the torch optimizer THAT HAS A GRADIENT (the MLP's: the std parameter never receives one, so torch keeps no state for it), in its order: (torch param, device index, kind, layer)."""
        lin = [m for m in self._raw_student.mlp if type(m).__name__ == "Linear"]
        slots = {}
        for k, mod in enumerate(lin):
            slots[id(mod.weight)] = (mod.weight, 2 * k, "weight", self.tt_student.mlp.layers[k])
            slots[id(mod.bias)] = (mod.bias, 2 * k + 1, "bias", self.tt_student.mlp.layers[k])
        return [slots[id(p)] for p in self.optimizer.param_groups[0]["params"] if id(p) in slots]

    def _optimizer_state_from_torch(self) -> None:
        ctx = self.ctx
        state = self.optimizer.state
        m, v = [ctx.ttnn.zeros_like(t) for t in self.adam.m], [ctx.ttnn.zeros_like(t) for t in self.adam.v]
        step = 0
        for p, idx, kind, layer in self._param_slots():
            if p not in state or "exp_avg" not in state[p]:
                continue
            step = max(step, int(state[p]["step"]))
            convert = layer.weight_to_host if kind == "weight" else layer.bias_to_host
            m[idx] = ctx.upload(convert(state[p]["exp_avg"]), op="H-6", batch_sharded=False)
            v[idx] = ctx.upload(convert(state[p]["exp_avg_sq"]), op="H-6", batch_sharded=False)
        self.adam.load_state(m, v, step)

    def _optimizer_state_to_torch(self) -> None:
        ctx = self.ctx
        for p, idx, kind, layer in self._param_slots():
            host = []
            for tensor in (self.adam.m[idx], self.adam.v[idx]):
                t = ctx.download(tensor, op="H-6", batch_sharded=False)
                host.append(layer.weight_from_host(t) if kind == "weight" else layer.bias_from_host(t))
            self.optimizer.state[p] = {"step": torch.tensor(float(self.adam.t)), "exp_avg": host[0].clone(), "exp_avg_sq": host[1].clone()}

    def get_policy(self):
        self.sync_to_torch()
        return super().get_policy()

    def save(self) -> dict:
        self.sync_to_torch(optimizer=True)
        return super().save()

    def train_mode(self) -> None:
        super().train_mode()
        self.tt_student.set_training(True)
        self.tt_teacher.set_training(False)  # the teacher's normalizer never learns

    def eval_mode(self) -> None:
        super().eval_mode()
        self.tt_student.set_training(False)
        self.tt_teacher.set_training(False)

    # -- rollout --------------------------------------------------------------------------------------------------------------------------
    def _act_graph(self, s_in, t_in, noise, persist: bool) -> dict:
        """The device ops of one act step: the student's forward pass + sample, the teacher's forward pass. `persist` (traced): ROW_MAJOR persistent inputs tilized here, results written in place."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        if persist:
            s_in = ttnn.to_layout(s_in, ttnn.TILE_LAYOUT)
            t_in = s_in if self.share_obs else ttnn.to_layout(t_in, ttnn.TILE_LAYOUT)
            noise = ttnn.to_layout(noise, ttnn.TILE_LAYOUT)
        dist = self.tt_student.distribution
        mean = self.tt_student.forward(s_in)
        action = dist.sample(mean, noise)
        teacher = self.tt_teacher.forward(t_in)
        out = dict(action=action, teacher=teacher)
        if persist:
            out["obs"] = s_in
            out = {k: ctx.put(self._buf.get("act." + k), v) for k, v in out.items()}
            for k, v in out.items():
                self._buf["act." + k] = v
        return out

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Student actions (the one download per step, H-2); the student's observations and the teacher's actions go to the device storage."""
        ctx = self.ctx
        student, teacher = self.tt_student, self.tt_teacher
        rows = obs.batch_size[0]
        dist = student.distribution
        traced = self.trace_enabled
        noise_host = pad_matrix(torch.randn(rows, student.output_dim), rows, dist.dim_p)  # parity mode: the stream upstream's Normal.sample() draws (H-4)
        with self.timer.section("act.upload_obs"):
            if traced:
                s_in = student.obs_to_device_rm(obs, into=self._buf.get("act.s_in"))
                self._buf["act.s_in"] = s_in
                t_in = s_in if self.share_obs else teacher.obs_to_device_rm(obs, into=self._buf.get("act.t_in"))
                if not self.share_obs:
                    self._buf["act.t_in"] = t_in
                noise = ctx.upload_rm_into(self._buf.get("act.noise"), noise_host, op="H-4", batch_sharded=True)
                self._buf["act.noise"] = noise
            else:
                s_in = student.obs_to_device(obs)
                t_in = s_in if self.share_obs else teacher.obs_to_device(obs)
                noise = ctx.upload(noise_host, op="H-4", batch_sharded=True)
        with self.timer.section("act.compute"):
            if traced and "act" in self._traces:
                self._replay("act")
                out = {k: self._buf["act." + k] for k in ("action", "teacher", "obs")}
            else:
                out = self._act_graph(s_in, t_in, noise, persist=traced)
                if traced:
                    out = self._act_graph(s_in, t_in, noise, persist=True)  # warm-up 2: the in-place copy programs must be cached before a capture
                    self._capture("act", lambda: self._act_graph(s_in, t_in, noise, persist=True))
        self._pending = dict(obs=out["obs"] if traced else s_in, teacher=out["teacher"])
        with self.timer.section("act.download_actions"):
            return ctx.download(out["action"], op="H-2")[:, : student.output_dim].contiguous()

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Upstream stores rewards and dones too, but a feed-forward student uses neither: the step's data (observations, teacher actions) is stored here."""
        pending, self._pending = self._pending, None
        assert pending is not None, "process_env_step() without a preceding act()"
        self.storage.add(pending["obs"], pending["teacher"])

    def compute_returns(self, obs: TensorDict) -> None:
        """Nothing to compute: distillation has no return targets."""

    # -- update ---------------------------------------------------------------------------------------------------------------------------
    def _step_graph(self, t: int) -> None:
        """One step of the update: the student's mean action on the stored observations of time index `t`, the loss against the teacher's action, its gradients ADDED to the accumulators."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        slot = self.storage.slot(t)
        student = self.tt_student
        out, cache = student.forward_train(slot["obs"])
        diff = ttnn.subtract(out, slot["teacher"])  # the padded columns are 0 on both sides
        n = float(diff.shape[0] * student.output_dim)  # all elements of the mean: rows (per chip) x real action dims
        cfg = ctx.compute_cfg
        if self.loss_type == "mse":
            loss = ttnn.multiply(ttnn.sum(ttnn.square(diff), dim=(0, 1), keepdim=True, compute_kernel_config=cfg), 1.0 / n)
            d_out = ttnn.multiply(diff, 2.0 / n)
        else:  # huber, delta 1: 0.5 d^2 inside, |d| - 0.5 outside; the gradient is d clamped to [-1, 1]
            absd = ttnn.abs(diff)
            inside = ttnn.le(absd, 1.0)
            per = ttnn.where(inside, ttnn.multiply(ttnn.square(diff), 0.5), ttnn.subtract(absd, 0.5))
            # the padded columns (diff = 0) would contribute 0 (inside branch): nothing to mask
            loss = ttnn.multiply(ttnn.sum(per, dim=(0, 1), keepdim=True, compute_kernel_config=cfg), 1.0 / n)
            d_out = ttnn.multiply(ttnn.clamp(diff, min=-1.0, max=1.0), 1.0 / n)
        grads = [g for pair in student.mlp.backward(cache, d_out) for g in pair]
        for k, g in enumerate(grads):
            self._acc_grads[k] = ctx.put(self._acc_grads[k], ttnn.add(self._acc_grads[k], g))
        self._acc_loss = ctx.put(self._acc_loss, ttnn.add(self._acc_loss, loss))

    def _apply_graph(self) -> None:
        """One optimizer step on the accumulated gradients (their SUM over the window): gradient mean over the chips, optional clip, Adam, zero the accumulators."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        grads = list(self._acc_grads)
        if ctx.n_chips > 1:  # upstream `reduce_parameters`: the gradients of the chips are averaged BEFORE the clip
            grads = [ctx.mean_over_chips(g) for g in grads]
        if self.max_grad_norm:
            grads, _ = clip_grad_norm(ctx, grads, self.max_grad_norm)
        self.tt_student.mlp.set_parameters(self.adam.step(self.tt_student.mlp.parameters(), grads, self.lr))
        for k, g in enumerate(self._acc_grads):
            self._acc_grads[k] = ctx.put(g, ttnn.multiply(g, 0.0))

    def update(self) -> dict[str, float]:
        """Upstream `Distillation.update` on the device (see the module docstring); with `trace` the steps and the apply are replayed."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        st = self.storage
        steps = st.steps
        assert st.step == steps, f"update needs a full rollout ({st.step} of {steps} steps stored)"
        self.num_updates += 1
        for k, g in enumerate(self._acc_grads):  # whatever was left over from the last update is discarded (upstream: `loss` is a local variable)
            self._acc_grads[k] = ctx.put(g, ttnn.multiply(g, 0.0))
        self._acc_loss = ctx.put(self._acc_loss, ttnn.multiply(self._acc_loss, 0.0))
        cnt = 0
        for _epoch in range(self.num_learning_epochs):
            for t in range(steps):
                key = f"distill.step.{t}"
                if self.trace_enabled and key in self._traces:
                    with self.timer.section("update.step.replay"):
                        self._replay(key)
                else:
                    self._step_graph(t)  # eager (always when tracing is off; the first occurrence of each step when it is on)
                    if self.trace_enabled:
                        self._capture(key, lambda t=t: self._step_graph(t))
                cnt += 1
                if cnt % self.gradient_length == 0:
                    self.adam.prepare_step()
                    if self.trace_enabled and "distill.apply" in self._traces:
                        with self.timer.section("update.apply.replay"):
                            self._replay("distill.apply")
                    else:
                        self._apply_graph()
                        if self.trace_enabled:
                            self._capture("distill.apply", self._apply_graph)
        with self.timer.section("update.normalizer"):
            self.tt_student.update_normalization(st.all_obs(), steps * st.num_envs // ctx.n_chips)  # all the rows of the rollout on ONE chip; built HERE: a tensor allocated before a replay and read after it would be overwritten (T-15)
        loss = float(ctx.download(self._acc_loss, op="H-5", batch_sharded=False)[0, 0]) / cnt
        self._weights_stale = self._optimizer_stale = True
        st.clear()
        if ctx.n_chips > 1 and self.replica_check_interval and self.num_updates % self.replica_check_interval == 0:
            diff = self.replica_max_diff()
            if diff != 0.0:
                raise RuntimeError(f"the chips' replicas of the student differ (max abs diff {diff:.3e}) after update {self.num_updates}")
        return {"behavior": loss}

    def replica_max_diff(self) -> float:
        """Largest absolute difference between any chip's copy and chip 0's: student parameters, Adam moments, normalizer statistics (0.0 = bit-identical)."""
        ctx = self.ctx
        if ctx.n_chips == 1:
            return 0.0
        tensors = self.tt_student.mlp.parameters() + self.adam.m + self.adam.v
        if self.tt_student.normalizer is not None:
            tensors += [self.tt_student.normalizer._mean, self.tt_student.normalizer._var]
        worst = 0.0
        for t in tensors:
            copies = ctx.per_chip(t)
            worst = max(worst, float((copies - copies[:1]).abs().max()))
        return worst
