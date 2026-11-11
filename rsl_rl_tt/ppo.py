# SPDX-License-Identifier: BSD-3-Clause
"""`TTPPO`: the upstream PPO with rollout inference AND the whole update running on Tenstorrent hardware (BRIEF Phases 2-3; ADR-0001/0002).

Injection (no upstream patch, the unmodified `OnPolicyRunner`):

    cfg["algorithm"]["class_name"] = "rsl_rl_tt.ppo:TTPPO"          # the qualified name; the runner device stays "cpu"
    cfg["algorithm"]["tt_cfg"] = {"mesh_shape": [1, 1]}              # optional; consumed here, upstream never sees it

`PPO.construct_algorithm` (inherited) builds the upstream torch actor/critic exactly as upstream does, so the initial weights are
bit-identical for the same torch seed; those models are the MIRRORS (device -> mirror sync, allow-list H-6, happens before `get_policy()`
/ `save()` / export whenever the device is ahead). Training state lives on the device: parameters, the normalizers' statistics, Adam
moments, the learning rate (a device tensor), the rollout storage, GAE and the shuffled mini-batches.

What still crosses the host boundary per step / iteration (all counted by `ctx.host_ops`, see docs/porting/DECISIONS.md):
H-1 upload obs / reward / done / time_out, H-2 download the actions, H-3 obs-group concat, H-4 host noise and `randperm` (parity mode),
H-5 per-iteration scalars for logging, H-8 tiny per-step scalar tensors (Adam bias corrections).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.algorithms import PPO
from rsl_rl_tt.device import acquire_tray_mesh
from rsl_rl_tt.losses import PPOLosses
from rsl_rl_tt.modules import TTMLPModel
from rsl_rl_tt.optim import TTAdam, TTAdaptiveLR, clip_grad_norm
from rsl_rl_tt.storage import TTRolloutStorage
from rsl_rl_tt.tt_common import SectionTimer, TTContext, linear_from_host, pack_step_scalars, pad_matrix, pad_vector, unit_row


_CAPTURED_TRACES: dict[int, int] = {}  # id(TrayMesh) -> number of live captured traces on that mesh (guard against the T-15 overwrite hazard)


class TTPPO(PPO):
    """PPO on the tray mesh (a single chip until Phase 6)."""

    def __init__(self, actor, critic, storage, *, tt_cfg: dict | None = None, **kwargs) -> None:
        if kwargs.get("rnd_cfg") or kwargs.get("symmetry_cfg"):
            raise NotImplementedError("RND and symmetry extensions are outside the v1 scope of the TT port")
        if kwargs.get("multi_gpu_cfg") is not None:
            raise NotImplementedError("rsl_rl's torch.distributed multi-GPU mode is replaced by the mesh data-parallel design (Phase 6)")
        if getattr(actor, "is_recurrent", False) or getattr(critic, "is_recurrent", False):
            raise NotImplementedError("recurrent models are outside the v1 scope")
        if str(kwargs.get("optimizer", "adam")).lower() != "adam":
            raise NotImplementedError("v1 implements Adam only (BRIEF 1: other optimizers need approval)")
        if kwargs.get("use_mixed_precision"):
            raise NotImplementedError("mixed precision is outside the v1 scope (fp32 policy, BRIEF 3.4)")
        super().__init__(actor, critic, storage, **kwargs)
        tt_cfg = dict(tt_cfg or {})
        mesh_shape = tuple(tt_cfg.pop("mesh_shape", (1, 1)))
        self.trace_enabled = bool(tt_cfg.pop("trace", False))  # capture/replay the act step and the mini-batch step (Phase 5); results are bit-identical to eager
        self._mesh_kwargs = {k: tt_cfg.pop(k) for k in ("fabric_config", "trace_region_size", "l1_small_size") if k in tt_cfg}
        if tt_cfg:
            raise ValueError(f"unknown tt_cfg keys: {sorted(tt_cfg)}")
        if self.trace_enabled:
            self._mesh_kwargs.setdefault("trace_region_size", 128 * 1024 * 1024)
        self.tray = acquire_tray_mesh(mesh_shape, **self._mesh_kwargs)
        if _CAPTURED_TRACES.get(id(self.tray), 0):
            # T-15: the intermediates of a captured trace are freed after the capture, and every replay rewrites them. Device state that this algorithm
            # allocates now could sit on those addresses and be silently overwritten by the other algorithm's replays.
            raise RuntimeError(
                "a captured trace is alive on this mesh: construct every TTPPO BEFORE the first trace is captured, or call release_traces() on the "
                "algorithm that owns them (TTNN_ISSUES T-15)"
            )
        self.ctx = TTContext.create(self.tray.mesh_device)
        if self.trace_enabled:
            self.tray.add_close_hook(self.release_traces)  # captured traces must be released before the device is closed
        self._validate_batch_alignment(storage)
        self.tt_actor = TTMLPModel(self.ctx, self._raw_actor)
        self.tt_critic = TTMLPModel(self.ctx, self._raw_critic)
        if self.tt_actor.distribution is None:
            raise ValueError("the actor needs a distribution_cfg (GaussianDistribution)")
        share = self.tt_actor.obs_groups == self.tt_critic.obs_groups
        self.storage = TTRolloutStorage(  # replaces the host RolloutStorage that upstream's construct_algorithm built
            self.ctx, storage.num_envs, storage.num_transitions_per_env, self.tt_actor.obs_dim_p, self.tt_critic.obs_dim_p, self.tt_actor.distribution.dim_p, share,
            persistent=self.trace_enabled,
        )
        self.losses = PPOLosses(
            self.ctx, clip_param=self.clip_param, value_loss_coef=self.value_loss_coef, entropy_coef=self.entropy_coef, use_clipped_value_loss=self.use_clipped_value_loss
        )
        self.lr_sched = TTAdaptiveLR(self.ctx, self.learning_rate, self.desired_kl, self.schedule)
        self._col0 = self.ctx.upload(unit_row(0, 32), op="H-6", batch_sharded=False)  # places the critic's single output in column 0
        self._pending: dict | None = None
        self._traces: dict = {}  # name -> trace id (act, update.step.<i>)
        self._buf: dict = {}  # persistent device buffers that traces read and write (inputs, outputs, accumulators)
        # PERSISTENT-STATE RULE of the traced path (TTNN_ISSUES T-15): a trace's intermediates are freed when its capture ends, so anything that keeps a
        # value across replays must be allocated BEFORE the first capture. Here: Adam moments, loss accumulators, the rollout storage slots (and the
        # constants of storage / losses / distribution created above). Buffers that are rewritten before every use (update.table, act inputs) are exempt.
        self.adam: TTAdam | None = TTAdam(self.ctx, self._params())
        for name in ("value", "surrogate", "entropy"):
            self._acc("update.acc." + name)
        if self.trace_enabled:
            self.storage.preallocate()
        self.timer = SectionTimer(self.ctx, enabled=False)  # `alg.timer.enabled = True` profiles the sections below (adds device syncs)
        self._weights_stale = False  # True while the device holds newer parameters than the torch mirrors
        self._optimizer_stale = False  # the same for the Adam moments
        self._refresh_std_mirror()

    # -- configuration checks -----------------------------------------------------------------------------------------------
    def _validate_batch_alignment(self, storage) -> None:
        """BRIEF 3.3: per-chip batch and mini-batch are multiples of 32; never pad silently, never allow a short last mini-batch."""
        ctx = self.ctx
        envs, steps = storage.num_envs, storage.num_transitions_per_env
        ctx.check_batch(envs, "num_envs")
        batch = envs * steps
        if batch % self.num_mini_batches != 0:
            raise ValueError(f"num_envs*num_steps_per_env = {batch} is not divisible by num_mini_batches = {self.num_mini_batches}")
        ctx.check_batch(batch // self.num_mini_batches, "mini-batch (num_envs*num_steps_per_env/num_mini_batches)")

    # -- policy mirror ------------------------------------------------------------------------------------------------------
    def _refresh_std_mirror(self) -> None:
        """The runner reads `get_policy().output_std` every iteration (allow-list H-5): keep a cached host copy in the torch mirror."""
        std = self.tt_actor.distribution.host_std()
        self._std_host = std.contiguous()
        self._raw_actor.distribution._distribution = Normal(torch.zeros(1, std.numel()), std)

    def sync_to_torch(self, optimizer: bool = False) -> None:
        """Device parameters and statistics -> the upstream torch models (H-6); with `optimizer=True` also the Adam moments -> the torch optimizer.

        `get_policy()` (export, inference, the runner's per-iteration logging) needs only the models; `save()` needs the optimizer state as well."""
        if self._weights_stale:
            self.tt_actor.sync_to_torch()
            self.tt_critic.sync_to_torch()
            self._weights_stale = False
        if optimizer and self._optimizer_stale:
            self._optimizer_state_to_torch()
            self._optimizer_stale = False

    def sync_from_torch(self) -> None:
        """The upstream torch models / optimizer (e.g. after `load()`) -> the device copies, incl. Adam moments and the learning rate (H-6, H-8)."""
        self.tt_actor.sync_from_torch()
        self.tt_critic.sync_from_torch()
        self.tt_actor.distribution.refresh()
        self.lr_sched.set(self.learning_rate)
        self._optimizer_state_from_torch()
        self._refresh_std_mirror()
        self._weights_stale = self._optimizer_stale = False

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Upstream's `PPO.load` on the torch mirrors/optimizer, then the device copies are rebuilt from them (also the Adam moments and the LR).

        Before loading, the torch optimizer is brought up to date with the device, so that `load_cfg` without "optimizer" keeps the CURRENT moments exactly like upstream."""
        self.sync_to_torch(optimizer=True)
        ok = super().load(loaded_dict, load_cfg, strict)
        self.sync_from_torch()
        return ok

    # -- optimizer state <-> torch.optim.Adam layout (checkpoint compatibility) -------------------------------------------------
    def _param_slots(self) -> list[tuple]:
        """For every parameter of the torch optimizer, in its order: (torch param, device index in `_params()`, kind, layer-or-dist)."""
        a, c = self.tt_actor, self.tt_critic
        n_a = len(a.mlp.parameters())
        learn = a.distribution.learn_std
        slots = {}
        for tag, tt_model, mirror, base in (("actor", a, self._raw_actor, 0), ("critic", c, self._raw_critic, n_a + (1 if learn else 0))):
            lin_idx = [i for i, m in enumerate(mirror.mlp) if type(m).__name__ == "Linear"]
            for k, mod_i in enumerate(lin_idx):
                lin = mirror.mlp[mod_i]
                slots[id(lin.weight)] = (lin.weight, base + 2 * k, "weight", tt_model.mlp.layers[k])
                slots[id(lin.bias)] = (lin.bias, base + 2 * k + 1, "bias", tt_model.mlp.layers[k])
        if learn:
            d = self._raw_actor.distribution
            std_param = d.std_param if d.std_type == "scalar" else d.log_std_param
            slots[id(std_param)] = (std_param, n_a, "std", a.distribution)
        return [slots[id(p)] for p in self.optimizer.param_groups[0]["params"] if id(p) in slots]

    def _optimizer_state_from_torch(self) -> None:
        """Torch optimizer state -> the device Adam moments, written IN PLACE (a traced step keeps reading the same tensors)."""
        ctx = self.ctx
        state = self.optimizer.state
        slots = self._param_slots()
        if self.adam is None:
            self.adam = TTAdam(ctx, self._params())
        m, v = [ctx.ttnn.zeros_like(t) for t in self.adam.m], [ctx.ttnn.zeros_like(t) for t in self.adam.v]
        step = 0
        for p, idx, kind, owner in slots:
            if p not in state or "exp_avg" not in state[p]:
                continue  # a fresh torch optimizer: Adam restarts from zero moments (m, v stay zero, step 0)
            step = max(step, int(state[p]["step"]))
            for store, key in ((m, "exp_avg"), (v, "exp_avg_sq")):
                t = state[p][key]
                if kind == "weight":
                    store[idx] = ctx.upload(pad_matrix(t.t(), owner.in_p, owner.out_p), op="H-6", batch_sharded=False)
                elif kind == "bias":
                    store[idx] = ctx.upload(pad_vector(t, owner.out_p), op="H-6", batch_sharded=False)
                else:
                    store[idx] = ctx.upload(pad_vector(t, owner.dim_p), op="H-6", batch_sharded=False)
        self.adam.load_state(m, v, step)

    def _optimizer_state_to_torch(self) -> None:
        if self.adam is None:
            return
        ctx = self.ctx
        for p, idx, kind, owner in self._param_slots():
            host = []
            for tensor in (self.adam.m[idx], self.adam.v[idx]):
                t = ctx.download(tensor, op="H-6", batch_sharded=False)
                if kind == "weight":
                    host.append(linear_from_host(t, t[:1], owner.out_features, owner.in_features)[0])
                elif kind == "bias":
                    host.append(t[0, : owner.out_features].contiguous())
                else:
                    host.append(t[0, : owner.dim].contiguous())
            self.optimizer.state[p] = {"step": torch.tensor(float(self.adam.t)), "exp_avg": host[0].clone(), "exp_avg_sq": host[1].clone()}

    def get_policy(self):
        self.sync_to_torch()
        return super().get_policy()

    def save(self) -> dict:
        self.sync_to_torch(optimizer=True)
        return super().save()

    def train_mode(self) -> None:
        super().train_mode()
        self.tt_actor.set_training(True)
        self.tt_critic.set_training(True)

    def eval_mode(self) -> None:
        super().eval_mode()
        self.tt_actor.set_training(False)
        self.tt_critic.set_training(False)

    # -- rollout ------------------------------------------------------------------------------------------------------------
    def _capture(self, name: str, fn) -> None:
        """Record `fn()` (device ops only, persistent tensors, every program already in the program cache) as a trace; the ops are recorded, not executed.

        Whatever happens inside, the capture is ALWAYS ended (an exception that left it open made every later read fail with "Reads are not supported
        during trace capture" and hung the process until the job timeout: docs/porting/incidents/2026-11-11_trace_capture_left_open.md)."""
        ttnn = self.ctx.ttnn
        trace_id = ttnn.begin_trace_capture(self.ctx.mesh, cq_id=0)
        try:
            fn()
        except BaseException:
            try:
                ttnn.end_trace_capture(self.ctx.mesh, trace_id, cq_id=0)
                ttnn.release_trace(self.ctx.mesh, trace_id)
            except BaseException as exc:
                print(f"rsl_rl_tt.ppo: could not end the failed trace capture cleanly: {exc!r}", file=__import__("sys").stderr)
            raise
        ttnn.end_trace_capture(self.ctx.mesh, trace_id, cq_id=0)
        self._traces[name] = trace_id
        _CAPTURED_TRACES[id(self.tray)] = _CAPTURED_TRACES.get(id(self.tray), 0) + 1

    def _replay(self, name: str) -> None:
        self.ctx.ttnn.execute_trace(self.ctx.mesh, self._traces[name], cq_id=0, blocking=False)

    def release_traces(self) -> None:
        for trace_id in self._traces.values():
            self.ctx.ttnn.release_trace(self.ctx.mesh, trace_id)
        _CAPTURED_TRACES[id(self.tray)] = _CAPTURED_TRACES.get(id(self.tray), 0) - len(self._traces)
        self._traces.clear()

    def _act_graph(self, actor_in, critic_in, noise, persist: bool) -> dict:
        """The device ops of one act step (forward passes, sample, log-prob, value). With persist=True the results are written in place into
        persistent buffers (what a trace needs); otherwise they are fresh tensors."""
        ctx = self.ctx
        actor, critic = self.tt_actor, self.tt_critic
        dist = actor.distribution
        mean = actor.forward(actor_in)
        action = dist.sample(mean, noise)
        logp = dist.log_prob(action, mean)
        value = critic.forward(critic_in)[:, 0:1]
        out = dict(action=action, mean=mean, logp=logp, value=value)
        if persist:
            out = {k: ctx.put(self._buf.get("act." + k), v) for k, v in out.items()}
            for k, v in out.items():
                self._buf["act." + k] = v
        return out

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions (forward passes on the device); everything but the actions stays on the device until process_env_step()."""
        ctx = self.ctx
        actor, critic = self.tt_actor, self.tt_critic
        rows = obs.batch_size[0]
        dist = actor.distribution
        share = self.storage.share_obs
        traced = self.trace_enabled
        assert not traced or self.storage.persistent, "a traced act step needs the persistent (preallocated) rollout storage"
        # parity mode (BRIEF 3.5): the noise is drawn on the host from the global generator, the stream upstream's Normal.sample() uses
        # (randn of shape [rows, A]); padded action columns get zero noise.
        noise_host = pad_matrix(torch.randn(rows, actor.output_dim), rows, dist.dim_p)
        with self.timer.section("act.upload_obs"):
            if traced:  # persistent input buffers, written in place (copy_host_to_device_tensor)
                actor_in = actor.obs_to_device(obs, into=self._buf.get("act.actor_in"))
                self._buf["act.actor_in"] = actor_in
                if share:
                    critic_in = actor_in
                else:
                    critic_in = critic.obs_to_device(obs, into=self._buf.get("act.critic_in"))
                    self._buf["act.critic_in"] = critic_in
                noise = ctx.upload_into(self._buf.get("act.noise"), noise_host, op="H-4", batch_sharded=True)
                self._buf["act.noise"] = noise
            else:
                actor_in = actor.obs_to_device(obs)
                critic_in = actor_in if share else critic.obs_to_device(obs)
                noise = ctx.upload(noise_host, op="H-4", batch_sharded=True)
        with self.timer.section("act.compute"):
            if traced and "act" in self._traces:
                self._replay("act")
                out = {k: self._buf["act." + k] for k in ("action", "mean", "logp", "value")}
            else:
                out = self._act_graph(actor_in, critic_in, noise, persist=traced)
                if traced:
                    # warm-up: the first persist=True pass ADOPTS the outputs as persistent buffers; only the second one runs the in-place `ttnn.copy`
                    # programs, and a program that is not in the program cache cannot be loaded during capture ("Warm up before capturing a trace")
                    out = self._act_graph(actor_in, critic_in, noise, persist=True)
                    self._capture("act", lambda: self._act_graph(actor_in, critic_in, noise, persist=True))
        # Traced: these are the trace's persistent buffers, overwritten by the next replay; `process_env_step` -> `add_transition` copies them into this
        # step's preallocated storage slot before that happens (no tensor is allocated here: see the persistent-state rule in __init__).
        self._pending = dict(actor_obs=actor_in, critic_obs=critic_in, action=out["action"], mean=out["mean"], value=out["value"], logp=out["logp"])
        with self.timer.section("act.download_actions"):
            return ctx.download(out["action"], op="H-2")[:, : actor.output_dim].contiguous()  # the one per-step sync point (H-2)

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Record one environment step on the device: upload (reward, done, time_out) in ONE tensor, bootstrap time-outs with V(s_t)."""
        ttnn = self.ctx.ttnn
        pending, self._pending = self._pending, None
        assert pending is not None, "process_env_step() without a preceding act()"
        with self.timer.section("process_env_step"):
            packed = self.ctx.upload(pack_step_scalars(rewards, dones, extras.get("time_outs")), op="H-1", batch_sharded=True)
            reward, done, time_out = packed[:, 0:1], packed[:, 1:2], packed[:, 2:3]
            # upstream: rewards += gamma * values * time_outs  (the pre-step value V(s_t), see UPSTREAM_SEMANTICS)
            reward = ttnn.add(reward, ttnn.multiply(ttnn.multiply(pending["value"], time_out), self.gamma))
            self.storage.add_transition(reward=reward, done=done, **pending)

    def compute_returns(self, obs: TensorDict) -> None:
        """GAE on the device (storage.compute_returns); the last value comes from the device critic."""
        critic = self.tt_critic
        with self.timer.section("compute_returns"):
            last_values = critic.forward(critic.obs_to_device(obs))[:, 0:1]
            self.storage.compute_returns(last_values, self.gamma, self.lam, normalize_advantages=not self.normalize_advantage_per_mini_batch)

    # -- update -------------------------------------------------------------------------------------------------------------
    def _params(self) -> list:
        a, c = self.tt_actor, self.tt_critic
        return a.mlp.parameters() + ([a.distribution.param] if a.distribution.learn_std else []) + c.mlp.parameters()

    def _set_params(self, new: list) -> None:
        a, c = self.tt_actor, self.tt_critic
        na = len(a.mlp.parameters())
        a.mlp.set_parameters(new[:na])
        k = na
        if a.distribution.learn_std:
            a.distribution.param = self.ctx.put(a.distribution.param, new[k])
            a.distribution.refresh()
            k += 1
        c.mlp.set_parameters(new[k:])

    def _acc(self, name: str):
        """Persistent [1, 1] loss accumulators (in-place adds, so a traced mini-batch step keeps writing the same tensors)."""
        if name not in self._buf:
            self._buf[name] = self.ctx.upload_scalar(0.0, op="H-8")
        return self._buf[name]

    def _minibatch_step(self, i: int, rows: int) -> None:
        """One mini-batch optimisation step on the persistent state: forward, KL / learning rate, losses, backward, per-model clipping, Adam.

        Reads rows i*rows..(i+1)*rows of the persistent shuffled table, writes parameters, Adam moments, the learning rate and the loss accumulators IN
        PLACE: the same function is executed eagerly and recorded as a trace. `adam.prepare_step()` (host side) must have run."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        st, actor, critic = self.storage, self.tt_actor, self.tt_critic
        dist = actor.distribution
        with self.timer.section("update.minibatch.slice"):
            batch = st.split(self._buf["update.table"][i * rows : (i + 1) * rows, :])
            adv = st.normalize(batch["advantages"], rows) if self.normalize_advantage_per_mini_batch and rows > 1 else batch["advantages"]
        with self.timer.section("update.minibatch.forward"):
            mean, cache_a = actor.forward_train(batch["actor_obs"])
            value_out, cache_c = critic.forward_train(batch["critic_obs"])
            entropy = dist.entropy_dev
        with self.timer.section("update.minibatch.kl_lr"):
            lr = self.lr_sched.update(self.losses.kl_mean(dist, batch["old_mean"], self._buf["update.old_var"], self._buf["update.old_log_std"], mean))
        with self.timer.section("update.minibatch.loss"):
            la = self.losses.actor(dist, mean, batch["actions"], batch["old_logp"], adv, rows)
            lc = self.losses.critic(value_out[:, 0:1], batch["old_values"], batch["returns"], rows)
        for name, value in (("value", lc["value_loss"]), ("surrogate", la["surrogate_loss"]), ("entropy", entropy)):
            acc = self._acc("update.acc." + name)  # accumulate BEFORE the optimizer step: `entropy` is a persistent tensor that the step rewrites in place
            ctx.put(acc, ttnn.add(acc, value))
        with self.timer.section("update.minibatch.backward"):
            grads_a = [t for pair in actor.mlp.backward(cache_a, la["d_mean"]) for t in pair] + ([la["d_param"]] if dist.learn_std else [])
            grads_c = [t for pair in critic.mlp.backward(cache_c, ttnn.multiply(lc["d_value"], self._col0)) for t in pair]
        with self.timer.section("update.minibatch.clip"):
            # upstream clips the actor group (incl. std) and the critic group SEPARATELY (DECISIONS C-1)
            grads_a, _ = clip_grad_norm(ctx, grads_a, self.max_grad_norm)
            grads_c, _ = clip_grad_norm(ctx, grads_c, self.max_grad_norm)
        with self.timer.section("update.minibatch.adam"):
            self._set_params(self.adam.step(self._params(), grads_a + grads_c, lr))

    def update(self) -> dict[str, float]:
        """PPO optimisation epochs over the stored rollout (upstream `PPO.update`), entirely on the device; with `trace` the mini-batch step is replayed."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        st, actor, critic = self.storage, self.tt_actor, self.tt_critic
        dist = actor.distribution
        n = st.batch_size
        mb = self.num_mini_batches
        if not 1 <= mb <= n:
            raise ValueError("num_mini_batches must be between 1 and the rollout batch size.")
        rows = n // mb
        if self.adam is None:
            self.adam = TTAdam(ctx, self._params())
        # the rollout's std (parameters do not change before the first optimizer step), snapshotted into persistent buffers
        for key, value in (("update.old_var", dist.var), ("update.old_log_std", dist.log_std)):
            self._buf[key] = ctx.put(self._buf.get(key), value)
        with self.timer.section("update.pack_table"):
            packed = st.packed_table()
        # one permutation per update, reused by every epoch (upstream mini_batch_generator); drawn on the host (parity mode, H-4)
        with self.timer.section("update.shuffle_gather"):
            shuffled = st.shuffle(packed, ctx.upload_index(torch.randperm(n), op="H-4"))
            self._buf["update.table"] = ctx.put(self._buf.get("update.table"), shuffled)
        for name in ("value", "surrogate", "entropy"):
            acc = self._acc("update.acc." + name)
            ctx.put(acc, ttnn.multiply(acc, 0.0))

        for _epoch in range(self.num_learning_epochs):
            for i in range(mb):
                self.adam.prepare_step()
                key = f"update.step.{i}"
                if self.trace_enabled and key in self._traces:
                    with self.timer.section("update.minibatch.replay"):
                        self._replay(key)
                else:
                    self._minibatch_step(i, rows)  # eager (always when tracing is off; the first occurrence of each step when it is on)
                    if self.trace_enabled:
                        self._capture(key, lambda i=i: self._minibatch_step(i, rows))

        # the normalizers learn from the whole rollout AFTER the epochs (upstream order)
        with self.timer.section("update.normalizers"):
            # built HERE, not before the epochs: a tensor allocated before a replay and read after it can be overwritten by that replay (T-15)
            actor.update_normalization(st.all_actor_obs(), n)
            critic.update_normalization(st.all_critic_obs(), n)

        num_updates = self.num_learning_epochs * mb
        host = lambda t: float(ctx.download(t, op="H-5", batch_sharded=False)[0, 0])
        loss_dict = {k: host(self._acc("update.acc." + k)) / num_updates for k in ("value", "surrogate", "entropy")}
        self.learning_rate = host(self.lr_sched.lr)
        for group in self.optimizer.param_groups:  # keep the (unused) torch optimizer's lr in line for logging / checkpoints
            group["lr"] = self.learning_rate
        self._weights_stale = self._optimizer_stale = True
        self._refresh_std_mirror()
        st.clear()
        return loss_dict
