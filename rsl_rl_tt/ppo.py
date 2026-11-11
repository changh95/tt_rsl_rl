# SPDX-License-Identifier: BSD-3-Clause
"""`TTPPO`: the upstream PPO with its policy/value inference running on Tenstorrent hardware (BRIEF Phase 2; ADR-0002).

Injection (no upstream patch, the unmodified `OnPolicyRunner`):

    cfg["algorithm"]["class_name"] = "rsl_rl_tt.ppo:TTPPO"          # the qualified name; the runner device stays "cpu"
    cfg["algorithm"]["tt_cfg"] = {"mesh_shape": [1, 1]}              # optional; consumed here, upstream never sees it

`PPO.construct_algorithm` (inherited) builds the upstream torch actor/critic exactly as upstream does, so the initial weights are
bit-identical for the same torch seed; those models are the MIRRORS. `TTPPO` mirrors them onto the open tray mesh and runs `act()` and the
last-value computation there. The mirrors stay the source of truth for `get_policy()`, `save()` and export (device -> mirror sync: H-6).

Phase 2 scope: `update()` is not implemented yet (Phase 3). Temporary host pieces, all on the allow-list of docs/porting/DECISIONS.md and
counted by `ctx.host_ops`: rollout storage and GAE on the host (H-7), obs concat (H-3), noise from the host generator (H-4).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.algorithms import PPO
from rsl_rl_tt.device import acquire_tray_mesh
from rsl_rl_tt.modules import TTMLPModel
from rsl_rl_tt.tt_common import TTContext, pad_matrix


class TTPPO(PPO):
    """PPO whose actor/critic forward passes run on the tray mesh (single chip in Phase 2)."""

    def __init__(self, actor, critic, storage, *, tt_cfg: dict | None = None, **kwargs) -> None:
        if kwargs.get("rnd_cfg") or kwargs.get("symmetry_cfg"):
            raise NotImplementedError("RND and symmetry extensions are outside the v1 scope of the TT port")
        if kwargs.get("multi_gpu_cfg") is not None:
            raise NotImplementedError("rsl_rl's torch.distributed multi-GPU mode is replaced by the mesh data-parallel design (Phase 6)")
        if getattr(actor, "is_recurrent", False) or getattr(critic, "is_recurrent", False):
            raise NotImplementedError("recurrent models are outside the v1 scope")
        super().__init__(actor, critic, storage, **kwargs)
        tt_cfg = dict(tt_cfg or {})
        mesh_shape = tuple(tt_cfg.pop("mesh_shape", (1, 1)))
        self._mesh_kwargs = {k: tt_cfg.pop(k) for k in ("fabric_config", "trace_region_size", "l1_small_size") if k in tt_cfg}
        if tt_cfg:
            raise ValueError(f"unknown tt_cfg keys: {sorted(tt_cfg)}")
        self.tray = acquire_tray_mesh(mesh_shape, **self._mesh_kwargs)
        self.ctx = TTContext.create(self.tray.mesh_device)
        self._validate_batch_alignment(storage)
        self.tt_actor = TTMLPModel(self.ctx, self._raw_actor)
        self.tt_critic = TTMLPModel(self.ctx, self._raw_critic)
        if self.tt_actor.distribution is None:
            raise ValueError("the actor needs a distribution_cfg (GaussianDistribution)")
        self._device_ahead = False  # True while the device holds newer parameters than the torch mirrors (set by update(), Phase 3)
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
        dist = self._raw_actor.distribution
        dist._distribution = Normal(torch.zeros(1, std.numel()), std)

    def sync_to_torch(self) -> None:
        """Device parameters and statistics -> the upstream torch models (H-6). Called before save/export when the device is ahead."""
        self.tt_actor.sync_to_torch()
        self.tt_critic.sync_to_torch()
        self._device_ahead = False

    def get_policy(self):
        if self._device_ahead:
            self.sync_to_torch()
        return super().get_policy()

    def save(self) -> dict:
        if self._device_ahead:
            self.sync_to_torch()
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
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store the transition (upstream `PPO.act`), with the forward passes on the device."""
        ctx = self.ctx
        actor, critic = self.tt_actor, self.tt_critic
        rows = obs.batch_size[0]
        dist = actor.distribution

        actor_in = actor.obs_to_device(obs)
        critic_in = actor_in if critic.obs_groups == actor.obs_groups and critic.obs_dim_p == actor.obs_dim_p else critic.obs_to_device(obs)
        mean = actor.forward(actor_in)
        # parity mode (BRIEF 3.5): the noise is drawn on the host from the global generator, the stream upstream's Normal.sample() uses
        # (randn of shape [rows, A]); padded action columns get zero noise.
        noise_host = pad_matrix(torch.randn(rows, actor.output_dim), rows, dist.dim_p)
        noise = ctx.upload(noise_host, op="H-4", batch_sharded=True)
        action = dist.sample(mean, noise)
        log_prob = dist.log_prob(action, mean)
        value = critic.forward(critic_in)[:, 0:1]

        a = actor.output_dim
        self.transition.hidden_states = (None, None)
        actions = ctx.download(action, op="H-2")[:, :a].contiguous()  # the one per-step sync point that stays (H-2)
        # --- temporary host storage (H-7, Phase 2 only) ---
        self.transition.values = ctx.download(value, op="H-7")[:, :1].contiguous()
        self.transition.actions_log_prob = ctx.download(log_prob, op="H-7")[:, 0].contiguous()
        mean_host = ctx.download(mean, op="H-7")[:, :a].contiguous()
        std_host = self._std_host.reshape(1, a).expand(rows, a).contiguous()  # cached host copy of the std (refreshed after every parameter change)
        self.transition.distribution_params = (mean_host, std_host)
        self.transition.actions = actions
        self.transition.observations = obs
        return actions

    def compute_returns(self, obs: TensorDict) -> None:
        """Last-step value on the device, then the upstream GAE on the host (H-7, Phase 2 only; moves to the device in Phase 3)."""
        ctx = self.ctx
        critic = self.tt_critic
        last_values = ctx.download(critic.forward(critic.obs_to_device(obs))[:, 0:1], op="H-7")[:, :1].contiguous()
        st = self.storage
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):  # identical to rsl_rl.algorithms.ppo.PPO.compute_returns
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch and st.advantages.numel() > 1:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        raise NotImplementedError("TTPPO.update() (backward, Adam, PPO losses on the device) is the Phase 3 deliverable")

