# UPSTREAM_MAP — component map and injection points of rsl_rl v5.5.2

Pinned source: tag `v5.5.2` (HEAD `1156331 Bump version to 5.5.2`, `pyproject.toml:7`). Declared deps: `torch>=2.6.0`,
`tensordict>=0.7.0` (`pyproject.toml:28-29`). At the time of writing `git status` shows `rsl_rl/` unmodified
(only untracked `CLAUDE.md` and `docs/porting/`).

Conventions used in this file and in `UPSTREAM_SEMANTICS.md`:

- Every `path:line` is relative to the repo root and points into the unmodified v5.5.2 tree.
- `[TORCH]` marks a behavior that is defined by torch/tensordict, not by the pinned source. It is stated from
  knowledge of those libraries and MUST be pinned by a test before a TT implementation relies on it. No torch
  was imported to write this document.
- `[UNVERIFIED]` marks anything that cannot be established from the pinned source (mainly Isaac Lab side behavior).
- Names such as "N" (num_envs), "T" (num_steps_per_env), "A" (num_actions), "D" (obs dim) are used for shapes.

---------------------------------------------------------------------------------------------------------------------

## 1. Package layout

| Path | LOC | Role | In v1 scope |
|---|---|---|---|
| `rsl_rl/runners/on_policy_runner.py` | 257 | `OnPolicyRunner`: construction, `learn()`, save/load, export, multi-GPU setup | yes |
| `rsl_rl/runners/distillation_runner.py` | 25 | `DistillationRunner(OnPolicyRunner)`, only checks `alg.teacher_loaded` (`:19-25`) | no |
| `rsl_rl/algorithms/ppo.py` | 486 | `PPO` | yes |
| `rsl_rl/algorithms/distillation.py` | 314 | `Distillation` (student/teacher) | no |
| `rsl_rl/models/mlp_model.py` | 255 | `MLPModel` plus `_TorchMLPModel`, `_OnnxMLPModel` export wrappers | yes |
| `rsl_rl/models/rnn_model.py`, `cnn_model.py` | 248, 252 | `RNNModel`, `CNNModel` (subclasses of `MLPModel`) | no |
| `rsl_rl/modules/mlp.py` | 95 | `MLP(nn.Sequential)` | yes |
| `rsl_rl/modules/distribution.py` | 464 | `Distribution` base, `GaussianDistribution` (yes), `HeteroscedasticGaussianDistribution`, `BetaDistribution` (no) | partly |
| `rsl_rl/modules/normalization.py` | 143 | `EmpiricalNormalization` (yes), `EmpiricalDiscountedVariationNormalization` (RND only) | partly |
| `rsl_rl/modules/rnn.py`, `cnn.py` | 86, 193 | `RNN`, `HiddenState`, `CNN` | no |
| `rsl_rl/storage/rollout_storage.py` | 359 | `RolloutStorage` with `Transition`, `Batch` | yes (feedforward path) |
| `rsl_rl/env/vec_env.py` | 102 | abstract `VecEnv` | yes (interface) |
| `rsl_rl/extensions/rnd.py`, `symmetry.py` | 242, 137 | RND, Symmetry (PPO extensions) | no |
| `rsl_rl/utils/utils.py` | 463 | `resolve_callable/class/obs_groups/optimizer/nn_activation`, `check_nan`, `compile_model`, `reduce_gradients_in_buckets`, trajectory pad/unpad | yes |
| `rsl_rl/utils/logger.py` | 339 | `Logger` (called by the runner) | consumed |
| `rsl_rl/utils/log_writer.py`, `wandb_log_writer.py`, `neptune_log_writer.py` | 47, 85, 95 | `LogWriter` ABC and backends | no |

`rsl_rl/algorithms/__init__.py` exports `PPO`, `Distillation`; `rsl_rl/models/__init__.py` exports `CNNModel`, `MLPModel`,
`RNNModel`; `rsl_rl/modules/__init__.py` exports `CNN`, `MLP`, `RNN`, `BetaDistribution`, `Distribution`,
`EmpiricalDiscountedVariationNormalization`, `EmpiricalNormalization`, `GaussianDistribution`,
`HeteroscedasticGaussianDistribution`, `HiddenState`. These package-level exports matter for simple-name resolution
(section 3.1).

---------------------------------------------------------------------------------------------------------------------

## 2. Runner (`OnPolicyRunner`)

### 2.1 Construction path (`on_policy_runner.py:26-54`)

Signature: `OnPolicyRunner(env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu")` (`:26`).

1. Store `env`, `cfg = train_cfg` (the SAME dict object, it is mutated later), `device` (`:28-30`).
2. `_configure_multi_gpu()` (`:33`, body `:215-257`): reads `WORLD_SIZE` (default 1). If `WORLD_SIZE <= 1` it sets
   `self.cfg["multi_gpu"] = None` and returns (`:218-226`). Otherwise it sets
   `cfg["multi_gpu"] = {"global_rank", "local_rank", "world_size"}` (`:233-237`), requires
   `device == f"cuda:{local_rank}"` (`:240`), calls `torch.distributed.init_process_group(backend="nccl", ...)` (`:255`)
   and `torch.cuda.set_device` (`:257`).
3. `obs = self.env.get_observations()` (`:36`). NOT moved to `device`; used only for keys, shapes, dtypes.
4. `alg_class = resolve_callable(self.cfg["algorithm"]["class_name"])` (`:39`), then
   `self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)` (`:40`). `construct_algorithm` is
   a `@staticmethod` (`ppo.py:428-429`). The annotation `alg: PPO` (`:23`) is not enforced anywhere (there is no
   `isinstance` in the runners; grep of `rsl_rl/runners/*.py` finds only attribute access).
5. `Logger(log_dir, cfg, env_cfg=env.cfg, num_envs, is_distributed, gpu_world_size, gpu_global_rank, device)` (`:43-52`).
6. `self.current_learning_iteration = 0` (`:54`).

### 2.2 Config keys that the runner / PPO.construct_algorithm / Logger actually read

| Key | Required | Where read | Notes |
|---|---|---|---|
| `algorithm` (dict) | yes | `on_policy_runner.py:39`, `ppo.py:432` | `class_name` required; remaining keys become constructor kwargs |
| `algorithm.rnd_cfg` | no | `on_policy_runner.py:97,125`, `logger.py:59,221,257` via `.get("rnd_cfg")`, `ppo.py:438` | truthiness decides whether runner reads `alg.intrinsic_rewards` and `alg.rnd.weight` |
| `actor`, `critic` (dict) | yes | `ppo.py:433-434` | read by `PPO.construct_algorithm`, NOT by the runner |
| `obs_groups` (dict) | yes | `ppo.py:440` | `KeyError` if missing; mutated in place by `resolve_obs_groups` (`utils.py:253-278`) |
| `num_steps_per_env` | yes | `on_policy_runner.py:84`, `ppo.py:457`, `logger.py:176` | |
| `save_interval` | yes | `on_policy_runner.py:129` | |
| `check_for_nan` | no (default True) | `on_policy_runner.py:90` | |
| `torch_compile_mode` | no | `ppo.py:463` | passed to `alg.compile(mode)` |
| `logger` | no (default `"tensorboard"`) | `logger.py:79-108` | str, or dict with `class_name` resolved by `resolve_class` |
| `run_name`, `wandb_project`, `neptune_project` | no | `logger.py:240,91,100` | |
| `multi_gpu` | written by runner | `on_policy_runner.py:225,233`, read at `ppo.py:460` | any user value is overwritten |
| anything else | ignored | n/a | there is no schema validation at the runner level |

### 2.3 `learn()` loop (`on_policy_runner.py:56-142`)

1. Optional `init_at_random_ep_len`: `env.episode_length_buf = torch.randint_like(buf, high=int(env.max_episode_length))` (`:59-62`).
2. `obs = env.get_observations().to(self.device)` (`:65`); `alg.train_mode()` (`:66`).
3. If distributed: `alg.broadcast_parameters()` (`:69-71`). Then `logger.init_logging_writer()` (`:74`).
4. Per iteration, inside `with torch.inference_mode():` (`:83`, block covers `:84-106`), repeat `num_steps_per_env` times:
   - `actions = alg.act(obs)` (`:86`)
   - `obs, rewards, dones, extras = env.step(actions.to(env.device))` (`:88`)
   - if `cfg.get("check_for_nan", True)`: `check_nan(obs, rewards, dones)` (`:90-91`; runs on env-device tensors, before `.to`)
   - `obs, rewards, dones = (x.to(self.device) ...)` (`:93`); `extras` is NOT moved
   - `alg.process_env_step(obs, rewards, dones, extras)` (`:95`)
   - `logger.process_env_step(rewards, dones, extras, intrinsic_rewards)` (`:99`) uses the SAME `rewards` tensor after PPO saw it
5. Still inside inference mode: `alg.compute_returns(obs)` with the last `obs` returned by `env.step` (`:106`).
6. Outside inference mode: `loss_dict = alg.update()` (`:109`); `current_learning_iteration = it` (`:113`).
7. `logger.log(..., loss_dict, learning_rate=alg.learning_rate, action_std=alg.get_policy().output_std, rnd_weight=...)` (`:116-126`).
   NOTE: `alg.get_policy()` is called every iteration just to read `output_std`.
8. If `logger.writer is not None and it % save_interval == 0`: `self.save(".../model_{it}.pt")` (`:129-130`); a final save after
   the loop (`:133-134`). `logger.writer` is `None` when `log_dir is None` or on non-zero ranks (`logger.py:78,110`).
9. `except BaseException`: `logger.abort_logging_writer()` and re-raise (`:136-139`); else `logger.stop_logging_writer()` (`:142`).
   There is no hook for the algorithm in the exception path.

### 2.4 Save / load / inference / export (runner side)

- `save(path, infos=None)` (`:144-151`): `d = alg.save(); d["iter"] = current_learning_iteration; d["infos"] = infos;
  torch.save(d, path)`; then `logger.save_model(path, it)`.
- `load(path, load_cfg=None, strict=True, map_location=None)` (`:153-169`): `torch.load(path, weights_only=False,
  map_location=map_location)`; `load_iteration = alg.load(loaded_dict, load_cfg, strict)`; if truthy,
  `current_learning_iteration = loaded_dict["iter"]`; returns `loaded_dict["infos"]`. `learn()` then restarts at
  `start_it = current_learning_iteration` (`:77`), i.e. the saved iteration is executed again (off by one vs. "next").
- `get_inference_policy(device=None)` (`:171-174`): `alg.eval_mode()`, then `return alg.get_policy().to(device)`.
  `nn.Module.to` is in-place, so this moves the live policy object.
- `export_policy_to_jit(path, filename)` (`:176-187`): `alg.get_policy().as_jit()`, `.to("cpu")`, `torch.jit.script(...)`, `.save()`.
- `export_policy_to_onnx(path, filename, verbose)` (`:189-209`): `alg.get_policy().as_onnx()`, `.to("cpu")`, `.eval()`,
  `torch.onnx.export(model, model.get_dummy_inputs(), path, export_params=True, opset_version=18, verbose=verbose,
  input_names=model.input_names, output_names=model.output_names)`. The exporter implementation depends on the torch
  version [TORCH, UNVERIFIED which default].
- `add_git_repo_to_log(path)` (`:211-213`): logger only.

### 2.5 Complete surface the runner uses on `alg` and on the policy

Every `self.alg.<x>` access in `rsl_rl/runners/*.py` (grep, complete):

| Access | Line | Contract |
|---|---|---|
| `alg_class.construct_algorithm(obs, env, cfg, device)` | `:40` | static method returning the algorithm |
| `alg.train_mode()` / `alg.eval_mode()` | `:66` / `:173` | |
| `alg.broadcast_parameters()` | `:71` | only if `WORLD_SIZE > 1` |
| `alg.act(obs)` | `:86` | returns a torch tensor `[N, A]`; runner calls `.to(env.device)` on it |
| `alg.process_env_step(obs, rewards, dones, extras)` | `:95` | |
| `alg.intrinsic_rewards` | `:97` | only if `cfg["algorithm"].get("rnd_cfg")` is truthy |
| `alg.compute_returns(obs)` | `:106` | called under `torch.inference_mode()` |
| `alg.update()` | `:109` | returns `dict[str, float]` (values used in `value:.4f` formatting, `logger.py:206-208,252-253`) |
| `alg.learning_rate` | `:123` | python float (passed to `writer.add_scalar`) |
| `alg.get_policy()` | `:124,174,178,191` | returns the policy model |
| `alg.rnd.weight` | `:125` | only with `rnd_cfg` |
| `alg.save()` / `alg.load(d, load_cfg, strict)` | `:146` / `:166` | `load` returns truthy to restore the iteration counter |
| `distillation_runner.py:22` `alg.teacher_loaded` | | DistillationRunner only |

On the policy returned by `get_policy()`: `.output_std` (tensor, `.mean().item()` taken at `logger.py:211,264`), `.to(device)`,
`.as_jit()`, `.as_onnx()`, and being callable as `policy(obs_tensordict)` for inference (`tests/runners/test_on_policy_runner.py:248-254`).

The runner NEVER touches `alg.actor`, `alg.critic`, `alg.storage`, `alg.optimizer` directly. (The in-repo tests do,
see section 13.3.)

---------------------------------------------------------------------------------------------------------------------

## 3. Config to class resolution

### 3.1 `resolve_callable` (`utils.py:99-176`)

| Input | Behavior | Lines |
|---|---|---|
| already callable (class, function) | returned as is (`callable()` check first) | `:122-123` |
| non-str, non-callable | `TypeError` | `:126-127` |
| `"pkg.mod:Attr.Nested"` | `importlib.import_module(mod)`, then `getattr` chain; errors propagate (`ImportError`, `AttributeError`) | `:130-138` |
| `"pkg.mod.Class"` | tries module prefixes from longest to shortest; `AttributeError` if module found but attr not; `ImportError` if no module found | `:141-164` |
| `"Name"` (no `.` or `:`) | `pkgutil.iter_modules(rsl_rl.__path__, "rsl_rl.")` imports each top-level `rsl_rl.*` subpackage and returns the first with `hasattr(module, name)`; else `ValueError` | `:167-176` |

Consequences for a plugin in a separate package `rsl_rl_tt`: a simple name like `"TTPPO"` is NOT resolvable (search is
limited to subpackages of `rsl_rl`). Use `"rsl_rl_tt.algorithms:TTPPO"`, `"rsl_rl_tt.algorithms.TTPPO"`, or pass the class
object itself in the dict (tests do this for the logger: `tests/runners/test_on_policy_runner.py:325`;
`docs/guide/overview.rst:18-19` states custom classes can be passed via config).

### 3.2 `resolve_class(cfg)` (`utils.py:179-189`)

`class_cfg = copy.deepcopy(cfg); return resolve_callable(class_cfg.pop("class_name")), class_cfg`. So the input dict is
not mutated, `class_name` is mandatory (`KeyError` otherwise), and ALL remaining keys are forwarded verbatim as
constructor kwargs. Class objects survive `deepcopy` (classes are atomic).

### 3.3 Where each `class_name` is resolved and how the class is called

| cfg path | Resolved at | Call | Lines |
|---|---|---|---|
| `cfg["algorithm"]["class_name"]` | runner, then again inside `construct_algorithm` | `alg_class.construct_algorithm(obs, env, cfg, device)`; inside: `alg_class(actor, critic, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"])` | `on_policy_runner.py:39-40`, `ppo.py:432,460` |
| `cfg["actor"]["class_name"]` | `PPO.construct_algorithm` | `actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)` | `ppo.py:433,449` |
| `cfg["critic"]["class_name"]` | `PPO.construct_algorithm` | `critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)` | `ppo.py:434,453` |
| `<model cfg>["distribution_cfg"]["class_name"]` | `MLPModel.__init__` | `dist_class(output_dim, **dist_cfg)`; MLP output dim becomes `distribution.input_dim` | `mlp_model.py:67-73` |
| `cfg["logger"]["class_name"]` | `Logger.init_logging_writer` | `writer_class(log_dir=log_dir, **writer_cfg)` | `logger.py:107-108` |
| `alg_cfg["symmetry_cfg"]["data_augmentation_func"]` | `Symmetry.__init__` | `resolve_callable` | `symmetry.py:60` (out of scope) |
| optimizer | NOT class-resolved: `resolve_optimizer(name_string)` accepts only `adam`, `adamw`, `sgd`, `rmsprop` | `torch.optim.*(params, lr=learning_rate)` | `utils.py:71-96`, `ppo.py:100-102` |
| storage | NOT resolved: `RolloutStorage(...)` is hard-coded | | `ppo.py:457` |

### 3.4 Behavior of extra / unknown keys

| Level | Behavior | Evidence |
|---|---|---|
| top-level runner cfg | unknown keys silently ignored; only the keys of section 2.2 are read | grep of `cfg[` / `cfg.get(` over `rsl_rl/` |
| `cfg["algorithm"]` | everything except `class_name` (and `share_cnn_encoders`, popped at `ppo.py:451`) is passed as `**alg_cfg` to the algorithm constructor. `PPO.__init__` has NO `**kwargs` (`ppo.py:40-68`), so an unknown key raises `TypeError`. A key duplicating `device` or `multi_gpu_cfg` raises `TypeError` (multiple values). `Distillation.__init__` accepts `**kwargs` (`distillation.py:68`), PPO does not | `ppo.py:460` |
| `rnd_cfg`, `symmetry_cfg` | `resolve_rnd_config` / `resolve_symmetry_config` ALWAYS leave the key present, set to `None` when unused, so `alg_class.__init__` always receives `rnd_cfg=` and `symmetry_cfg=` kwargs when constructed through `PPO.construct_algorithm` | `rnd.py:226-241`, `symmetry.py:133-136`, `ppo.py:443,446` |
| `cfg["actor"]`, `cfg["critic"]` | everything except `class_name` becomes model kwargs. `MLPModel.__init__` has no `**kwargs` (`mlp_model.py:30-40`), so unknown keys raise `TypeError` | `ppo.py:449,453` |
| `distribution_cfg` | everything except `class_name` becomes distribution kwargs. `GaussianDistribution` accepts `init_std, std_range, std_type, learn_std` only (`distribution.py:144-151`) | `mlp_model.py:68-69` |
| `cfg["logger"]` | same pattern, kwargs forwarded to the writer | `logger.py:107-108` |

So TT-specific settings can travel inside the dict cfg only if the TT class consumes them (explicit kwargs or `**kwargs`).
Whether Isaac Lab's cfg classes can emit extra keys is [UNVERIFIED] (outside this source).

### 3.5 Observation groups / observation sets

- `cfg["obs_groups"]`: `dict[str, list[str]]`, mapping an observation SET (e.g. `"actor"`, `"critic"`, `"rnd_state"`) to a list of
  observation GROUPS (keys of the env's obs TensorDict) (`vec_env.py:70-83`, `docs/guide/configuration.rst` "Observation Configuration").
- `PPO.construct_algorithm` (`ppo.py:437-440`): `default_sets = ["actor", "critic"]` (+ `"rnd_state"` if rnd); then
  `cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)`.
- `resolve_obs_groups` (`utils.py:192-287`):
  - empty dict gives a warning only (`:234-238`);
  - for every configured set: empty list is `ValueError` (`:243-244`); a group absent from `obs` is `ValueError` (`:246-251`);
  - for each default set missing from the dict: if a group with the set's name exists use `[set_name]`; elif `"policy"` exists use
    `["policy"]`; else `ValueError` (`:254-278`), with warnings for the fallbacks;
  - prints the resolved sets (`:281-285`); mutates and returns the same dict.
- Concatenation is done by each model, not by the algorithm: `MLPModel.get_latent` does
  `torch.cat([obs[g] for g in self.obs_groups], dim=-1)` in list order (`mlp_model.py:115`), then applies the normalizer
  (`:117`). `obs_dim` is the sum of last-dim sizes; every group must be 2-D `[N, d]` (`mlp_model.py:179-189`).
- The rollout storage stores EVERY group of the env's obs TensorDict, not only the groups used by actor/critic
  (`rollout_storage.py:142-148`).

---------------------------------------------------------------------------------------------------------------------

## 4. PPO (`rsl_rl/algorithms/ppo.py`)

### 4.1 Constructor (`:40-122`)

`PPO(actor, critic, storage, num_learning_epochs=5, num_mini_batches=4, clip_param=0.2, gamma=0.99, lam=0.95,
value_loss_coef=1.0, entropy_coef=0.01, learning_rate=0.001, max_grad_norm=1.0, optimizer="adam",
use_clipped_value_loss=True, schedule="adaptive", desired_kl=0.01, normalize_advantage_per_mini_batch=False,
use_mixed_precision=False, device="cpu", rnd_cfg=None, symmetry_cfg=None, multi_gpu_cfg=None, grad_reduce_bucket_mb=25)`
(`:40-68`).

What it does: `self.actor = actor.to(device)`, `self.critic = critic.to(device)` (`:92-93`); keeps `_raw_actor`/`_raw_critic`
for state_dict ops, export and `get_policy` (`:96-97`); builds ONE optimizer over
`list(dict.fromkeys(chain(actor.parameters(), critic.parameters())))` with only `lr=learning_rate` (`:100-102`);
`self.storage = storage`, `self.transition = RolloutStorage.Transition()` (`:105-106`); stores hyperparameters as attributes
(`:109-122`), including `self.learning_rate` (python float) and `self.schedule`, `self.desired_kl`.

### 4.2 `construct_algorithm(obs, env, cfg, device)` (`:428-465`, static)

1. `resolve_class` for `algorithm`, `actor`, `critic` (`:432-434`).
2. Resolve obs groups (`:437-440`), RND config (`:443`), symmetry config (`:446`).
3. `actor = actor_class(obs, obs_groups, "actor", env.num_actions, **actor_cfg).to(device)` (`:449`); optional shared CNN (`:451-452`);
   `critic = critic_class(obs, obs_groups, "critic", 1, **critic_cfg).to(device)` (`:453`). Both models are therefore built
   (and their weights initialized from the torch default RNG) before the algorithm constructor runs.
4. `storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)` (`:457`).
5. `alg = alg_class(actor, critic, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"])` (`:460`).
6. `alg.compile(cfg.get("torch_compile_mode"))` (`:463`): unconditional call; `compile(mode=None)` is a no-op via `compile_model`
   (`:417-426`, `utils.py:335-336`).

### 4.3 State held by `PPO`

`actor`, `critic` (possibly `torch.compile`d wrappers) and `_raw_actor`, `_raw_critic`; `optimizer`; `storage`; `transition`
(per-step scratch); `learning_rate` (float); `rnd`, `symmetry` (None in v1); `intrinsic_rewards` (only with RND); the hyperparameter
attributes of 4.1; multi-GPU: `is_multi_gpu`, `gpu_global_rank`, `gpu_world_size`, `grad_reduce_bucket_mb` (`:71-81`).

`Transition` fields set per step (`rollout_storage.py:24-65`): `observations` (a reference to the obs TensorDict passed to `act`),
`actions [N,A]`, `values [N,1]`, `actions_log_prob [N]`, `distribution_params (mean [N,A], std [N,A])` [TORCH for the std shape:
`Normal` broadcasts `scale` to the shape of `loc`; consistent with `tests/algorithms/test_ppo.py:164`], `hidden_states (None, None)`
for MLP, `rewards [N]`, `dones [N]`.

### 4.4 `act(obs)` (`:124-135`)

Order: record hidden states `(actor.get_hidden_state(), critic.get_hidden_state())` (`:127`); `actions = actor(obs,
stochastic_output=True).detach()` (`:129`); `values = critic(obs).detach()` (`:130`); `actions_log_prob =
actor.get_output_log_prob(actions).detach()` (`:131`); `distribution_params = tuple(p.detach() for p in
actor.output_distribution_params)` (`:132`); `transition.observations = obs` (`:134`); return `transition.actions`.
The actor forward both samples AND stores the distribution inside the model (`mlp_model.py:103-106`); `get_output_log_prob`,
`output_distribution_params`, `output_entropy` read that stored state.

### 4.5 `process_env_step(obs, rewards, dones, extras)` (`:137-164`)

`transition.rewards = rewards.clone()` (the comment at `:142` says the clone is needed because rewards are bootstrapped on
time-outs below; consequence: the env's tensor stays unmodified, and the runner hands that same `rewards` tensor to the logger afterwards,
`on_policy_runner.py:99`); `transition.dones = dones` (`:144`); RND branch (`:147-151`, out of scope); time-out bootstrap
(`:154-158`); `storage.add_transition(transition)`; `transition.clear()`; `actor.reset(dones)`, `critic.reset(dones)` (`:161-164`).
The `obs` argument is the NEXT observation and is only used by RND.

### 4.6 `compute_returns(obs)` (`:166-191`)

`last_values = critic(obs).detach()` (`:171`) (after saving the critic's hidden state and restoring it, `:170-173`); reverse loop
over `T` writing `st.returns[step]` in place (`:175-186`); `st.advantages = st.returns - st.values` (`:188`); global advantage
normalization unless `normalize_advantage_per_mini_batch` (`:190-191`). Formulas are in `UPSTREAM_SEMANTICS.md` S02-S06.

### 4.7 `update()` (`:193-358`)

1. Choose generator: recurrent if `actor.is_recurrent or critic.is_recurrent` else `storage.mini_batch_generator(num_mini_batches,
   num_learning_epochs)` (`:204-207`).
2. For each mini-batch (`:210`): optional per-minibatch advantage normalization (`:214-216`); symmetry (`:219-220`); under optional
   bf16 autocast (`:223-225`): actor forward with `stochastic_output=True` to refresh the stored distribution (`:228-233`),
   `actions_log_prob` (`:234`), `values = critic(batch.observations, ...)` (`:235`), `distribution_params`, `entropy`
   (`:237-238`); KL and adaptive LR (`:241-266`); surrogate (`:269-274`); value loss (`:277-283`); total loss (`:285`).
3. `optimizer.zero_grad(); loss.backward()` (`:297-298`); `reduce_parameters()` if multi-GPU (`:305-306`);
   `clip_grad_norm_(actor.parameters(), max_grad_norm)` then `clip_grad_norm_(critic.parameters(), max_grad_norm)` (`:309-310`);
   `optimizer.step()` (`:311`); accumulate `.item()` of value loss, surrogate loss, mean entropy (`:317-319`).
4. After all mini-batches: `obs = storage.observations.flatten(0, 1)`; `actor.update_normalization(obs)`,
   `critic.update_normalization(obs)` (`:327-332`).
5. Divide accumulated losses by `num_learning_epochs * num_mini_batches` (`:335-338`); return
   `{"value", "surrogate", "entropy"}` (+ `"rnd"`, `"symmetry"` when enabled) (`:345-353`).
6. `storage.clear()` (`:356`), i.e. only resets the write index.

### 4.8 Other methods

- `train_mode()` / `eval_mode()` (`:360-372`): `actor.train()/eval()`, `critic.train()/eval()` (+ rnd).
- `get_policy()` returns `self._raw_actor` (`:413-415`): the live, uncompiled actor.
- `save()` / `load()`: section 10.
- `broadcast_parameters()` / `reduce_parameters()`: section 9.

---------------------------------------------------------------------------------------------------------------------

## 5. Models

### 5.1 `MLPModel(nn.Module)` (`models/mlp_model.py:19-193`)

Constructor: `(obs, obs_groups, obs_set, output_dim, hidden_dims=(256,256,256), activation="elu", obs_normalization=False,
distribution_cfg=None)` (`:30-40`). Registration order (matters for `parameters()` / `state_dict()` order):

1. `obs_groups, obs_dim = _get_obs_dim(...)` (plain attributes, `:57`).
2. `obs_normalizer`: `EmpiricalNormalization(obs_dim)` if `obs_normalization`, else `torch.nn.Identity()` (`:59-64`).
3. `distribution`: `dist_class(output_dim, **dist_cfg)` or `None` (`:66-73`); MLP output dim = `distribution.input_dim` or `output_dim`.
4. `mlp = MLP(latent_dim, mlp_output_dim, hidden_dims, activation)` (`:76`).
5. `distribution.init_mlp_weights(mlp)` (`:79-80`). For `GaussianDistribution` this is the base-class no-op (`distribution.py:120-129`).
   `MLP.init_weights` (orthogonal init, `mlp.py:80-89`) is never called by any model (grep): weights keep torch's `nn.Linear` default init.

Class attribute `is_recurrent = False` (`:27`).

`forward(obs, masks=None, hidden_state=None, stochastic_output=False)` (`:82-108`): unpad if masks given and model not recurrent
(`:97`); `latent = get_latent(obs, masks, hidden_state)` (`:99`); `mlp_output = mlp(latent)` (`:101`); with a distribution: if
`stochastic_output`, `distribution.update(mlp_output)` then `return distribution.sample()` (`:103-106`), else
`distribution.deterministic_output(mlp_output)` (`:107`); without a distribution return `mlp_output` (`:108`).
IMPORTANT: the deterministic path does NOT update the stored distribution.

Other API used by the algorithm/runner:

| Member | Lines | Behavior |
|---|---|---|
| `get_latent` | `:110-118` | concat groups, normalize |
| `reset(dones=None, hidden_state=None)`, `get_hidden_state()`, `detach_hidden_state(dones)` | `:120-130` | no-ops / `None` for MLP |
| `output_mean`, `output_std`, `output_entropy`, `output_distribution_params` | `:132-150` | read the stored `distribution` state |
| `get_output_log_prob(outputs)` | `:152-154` | `distribution.log_prob` |
| `get_kl_divergence(old_params, new_params)` | `:156-160` | `distribution.kl_divergence` |
| `update_normalization(obs)` | `:170-177` | if `obs_normalization and obs_groups`: concat groups, `obs_normalizer.update(...)` |
| `as_jit()` / `as_onnx()` | `:162-168` | return `_TorchMLPModel(self)` / `_OnnxMLPModel(self)` |

`state_dict` layout of an actor with `obs_normalization=True` and a Gaussian distribution (from registration order and
`EmpiricalNormalization` buffers `normalization.py:31-34`): `obs_normalizer._mean [1,D]`, `obs_normalizer._var [1,D]`,
`obs_normalizer._std [1,D]`, `obs_normalizer.count` (scalar int64), `distribution.std_param [A]` (or `distribution.log_std_param [A]`),
`mlp.0.weight`, `mlp.0.bias`, `mlp.2.weight`, ... (MLP is `nn.Sequential` with layers at even indices and activations at odd indices,
`mlp.py:53-60,77-78`). The critic has `distribution = None` and, without normalization, `obs_normalizer = Identity()` (no entries).
`parameters()` order therefore starts with the std parameter, then MLP weights. This order defines the Adam state indices
in `optimizer_state_dict` (section 10).

### 5.2 `RNNModel`, `CNNModel` (not in v1)

- `RNNModel(MLPModel)` (`rnn_model.py:18-113`): `is_recurrent = True`; adds `rnn = RNN(obs_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)`
  registered AFTER the MLP parts (`:73-74`); `get_latent` runs the RNN (`:76-84`); hidden state is stored inside the model (`modules/rnn.py`).
  Export wrappers `_TorchGRUModel`, `_TorchLSTMModel`, `_OnnxRNNModel` (`:116-248`).
- `CNNModel(MLPModel)` (`cnn_model.py:19-163`): 2-D obs groups through `CNN` encoders, 1-D groups through the MLP path; accepts shared
  `cnns` (`:39`, wired by `share_cnn_encoders`, `ppo.py:451-452`). Export wrappers `_TorchCNNModel`, `_OnnxCNNModel` (`:165-252`).

### 5.3 Export helpers of `MLPModel`

- `_TorchMLPModel` (`mlp_model.py:196-218`): deep copies `obs_normalizer` and `mlp`; `deterministic_output` is
  `distribution.as_deterministic_output_module()` or `Identity`. `forward(x)` takes a PRE-CONCATENATED tensor:
  `deterministic_output(mlp(obs_normalizer(x)))`. Has `@torch.jit.export reset()` no-op.
  Scriptability relies on `EmpiricalNormalization.update/inverse` being `@torch.jit.unused` (`normalization.py:50,80`).
- `_OnnxMLPModel` (`:221-255`): same body; `input_size = model.obs_dim`; `get_dummy_inputs() = (torch.zeros(1, input_size),)`;
  `input_names = ["obs"]`, `output_names = ["actions"]`; `is_recurrent = False`.

What the helpers require from a model: attributes `obs_normalizer` (module), `mlp` (module), `distribution` (None or object with
`as_deterministic_output_module()`), `obs_dim` (int). Any model returned by `get_policy()` that exposes these (or overrides
`as_jit`/`as_onnx`) reuses the upstream export path unchanged.

---------------------------------------------------------------------------------------------------------------------

## 6. Modules

### 6.1 `MLP(nn.Sequential)` (`modules/mlp.py:16-95`)

`Linear(in, h0), act, Linear(h0, h1), act, ..., Linear(h_last, out)`; no activation after the last Linear unless `last_activation`
is given (`:55-73`); `-1` in `hidden_dims` means "use input_dim" (`:51`); if `output_dim` is a list/tuple the last layer is
`Linear(h_last, prod(output_dim))` followed by `nn.Unflatten(-1, output_dim)` (`:63-70`). One activation module instance is reused
at every activation slot (`:48,56,60`; stateless). Default activation of `MLPModel` is `"elu"` (`mlp_model.py:37`);
`resolve_nn_activation` (`utils.py:34-68`) maps `elu`, `selu`, `relu`, `crelu`(=CELU), `lrelu`, `tanh`, `sigmoid`, `softplus`, `gelu`,
`swish`(=SiLU), `mish`, `identity`.

### 6.2 `Distribution` interface (`distribution.py:15-129`)

A plugin distribution must provide: `update(mlp_output)`, `sample()`, `deterministic_output(mlp_output)`,
`as_deterministic_output_module()`, `input_dim`, `mean`, `std`, `entropy` (summed over last dim), `params` (tuple of tensors), `log_prob(outputs)`
(summed over last dim), `kl_divergence(old_params, new_params)` (summed over last dim), `init_mlp_weights(mlp)`. The distribution is
STATEFUL: `update` stores the current `torch.distributions` object that all the read accessors use.

### 6.3 `GaussianDistribution` (`:132-238`)

State-independent std. Constructor `(output_dim, init_std=1.0, std_range=(1e-6, 1e6), std_type="scalar", learn_std=True)` (`:144-151`).
- `std_type="scalar"`: `std_param = nn.Parameter(init_std * ones(output_dim), requires_grad=learn_std)` (`:165-166`).
- `std_type="log"`: `log_std_param = nn.Parameter(log(init_std * ones(output_dim)), requires_grad=learn_std)` (`:167-168`).
- Other `std_type`: `ValueError` (`:169-170`).
- `std_range[0] = max(std_range[0], 1e-6)`; `log_std_range = [log(lo), log(hi)]` via numpy (`:173-175`).
- `Normal.set_default_validate_args(False)` is called globally in the constructor (`:181`).
- `update(mlp_output)` (`:183-191`): scalar: `std = std_param.clamp(lo, hi)`; log: `log_std = log_std_param.clamp(log_lo, log_hi); std = exp(log_std)`;
  then `self._distribution = Normal(mean=mlp_output, std)`. No softplus anywhere in the Gaussian (softplus is only in `BetaDistribution`, `:367-368,377-379`).
- `sample()` = `Normal.sample()` (`:193-195`): no reparametrization.
- `deterministic_output` = the MLP output (mean) (`:197-199`); export module `_IdentityDeterministicOutput` (`:438-442`).
- `entropy` = `Normal.entropy().sum(-1)` (`:220-223`); `log_prob(x)` = `Normal.log_prob(x).sum(-1)` (`:230-232`);
  `params = (mean, std)` of the current distribution (`:225-228`);
  `kl_divergence(old, new)` = `kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(-1)` (`:234-238`).
- Closed forms of the torch functions are listed as [TORCH] in `UPSTREAM_SEMANTICS.md` S10, S11, S13.

`HeteroscedasticGaussianDistribution` (`:241-323`): MLP outputs `[..., 2, A]` (mean, std or log_std); same clamping; `init_mlp_weights`
zeroes the std-head weights and sets bias to `init_std` (or `log(init_std + 1e-7)`) (`:315-323`). `BetaDistribution` (`:326-435`):
`alpha = softplus(raw) + 1`, samples rescaled to `action_range`. Both are outside v1.

### 6.4 `EmpiricalNormalization` (`modules/normalization.py:15-83`)

Constructor `(shape, eps=1e-2, until=None)` (`:18`). Buffers: `_mean` zeros `[1,D]`, `_var` ones `[1,D]`, `_std` ones `[1,D]`, `count` int64 0
(`:31-34`). `mean` / `std` properties return `squeeze(0).clone()` (`:36-44`). `forward(x) = (x - _mean) / (_std + eps)` (`:46-48`).
`update(x)` (`:50-78`, `@torch.jit.unused`): returns immediately if `not self.training` (`:53-54`) or if `until is not None and count >= until`
(`:55-56`); otherwise batch moments `mean_x`, `var_x` (population variance, `unbiased=False`), optional distributed all-reduce
combination (`:62-71`), then the parallel-moment merge (`:73-78`). `MLPModel` constructs it with defaults only (`mlp_model.py:62`), so
`eps = 1e-2`, `until = None`, and there is no clipping argument in this version. Details in `UPSTREAM_SEMANTICS.md` S21-S24.

`EmpiricalDiscountedVariationNormalization` (`:86-119`) is used only by RND reward normalization.

---------------------------------------------------------------------------------------------------------------------

## 7. `RolloutStorage` (`storage/rollout_storage.py`)

Constructor `(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device="cpu")` (`:125-133`); PPO builds it with
`("rl", N, T, obs, [A], device)` (`ppo.py:457`).

| Buffer | Shape | dtype | Lines |
|---|---|---|---|
| `observations` (TensorDict, batch `[T, N]`, ALL env groups) | `[T, N, *group_shape]` | dtype of the env obs tensor | `:142-149` |
| `rewards` | `[T, N, 1]` | default float (float32) | `:150` |
| `actions` | `[T, N, A]` | float32 | `:151` |
| `dones` | `[T, N, 1]` | `uint8` (`.byte()`) | `:152` |
| `values`, `actions_log_prob`, `returns`, `advantages` (rl) | `[T, N, 1]` | float32 | `:160-164` |
| `distribution_params` (rl) | tuple of `[T, N, A]`, allocated lazily on first `add_transition` | float32 | `:162,193-197` |
| `saved_hidden_state_a/c` | lazily allocated for recurrent models | | `:167-168,333-359` |

`add_transition(transition)` (`:173-205`): `OverflowError` if `step >= T` (`:176-177`); `observations[step].copy_(obs)`, `actions[step].copy_`,
`rewards[step].copy_(rewards.view(-1, 1))`, `dones[step].copy_(dones.view(-1, 1))` (`:180-183`; the `copy_` converts dtype, so float/bool dones
become uint8); `values[step].copy_`, `actions_log_prob[step].copy_(.view(-1, 1))`, `distribution_params[i][step].copy_` (`:191-199`);
`step += 1`. `clear()` only sets `step = 0` (`:207-209`); buffers are not zeroed.

`mini_batch_generator(num_mini_batches, num_epochs=8)` (`:225-259`), used by `PPO.update` with `num_epochs = num_learning_epochs`:
- `batch_size = N * T` (`:229`); `ValueError` unless `1 <= num_mini_batches <= batch_size` (`:230-231`).
- `indices = torch.randperm(batch_size, requires_grad=False, device=self.device)` (`:232`): ONE permutation per call, computed lazily at the
  first `next()` of the generator (it is a generator function), and reused for every epoch.
- Flatten with `flatten(0, 1)` (`:235-241`): flat index `i = t * N + n` (time-major).
- `for epoch in range(num_epochs): for i in range(num_mini_batches)`: `start = i * batch_size // num_mini_batches`,
  `stop = (i + 1) * batch_size // num_mini_batches`, `batch_idx = indices[start:stop]` (`:243-248`). Sizes differ by at most one and no
  sample is dropped (remainders are retained; tests `tests/storage/test_rollout_storage.py:371-387`); the LAST batch is the largest, never shorter.
- Yields `Batch(observations, actions, values, advantages, returns, old_actions_log_prob, old_distribution_params)` gathered with
  `x[batch_idx]` (`:251-259`). Per-batch shapes: obs TensorDict `[B]`, `actions [B,A]`, `values/advantages/returns/old_actions_log_prob [B,1]`,
  `old_distribution_params ([B,A], [B,A])`; `masks = None`, `hidden_states = (None, None)` (`:83,122`).

`recurrent_mini_batch_generator` (`:262-331`) partitions ENVIRONMENTS/trajectories instead of samples (no randperm); not in v1.
`generator()` (`:212-222`) is distillation only.

---------------------------------------------------------------------------------------------------------------------

## 8. `VecEnv` interface (`env/vec_env.py`)

Abstract attributes (`:21-42`): `num_envs: int`, `num_actions: int`, `max_episode_length: int or Tensor`, `episode_length_buf: Tensor`,
`device`, `cfg: dict or object`. Methods:

- `get_observations() -> TensorDict` (`:44-55`). The returned TensorDict and its tensors must not change until the transition of the next
  step has been copied into rollout storage.
- `step(actions [N, A]) -> (obs: TensorDict, rewards: Tensor (N,), dones: Tensor (N,), extras: dict)` (`:57-102`).
  - `obs` is a TensorDict of observation groups (batch size `[N]`; test env: `tests/conftest.py:12-18`, `tests/runners/test_on_policy_runner.py:41-45`).
  - `rewards` shape `(num_envs,)` (`:65`); PPO does `rewards.clone()` then `+=` a `[N]` tensor (`ppo.py:143,155-158`), so a `[N,1]` reward would fail.
  - `dones` shape `(num_envs,)` (`:66`). Dtype is not specified; the test env returns float (`test_on_policy_runner.py:49`); storage casts to uint8.
  - `extras["time_outs"]` (tensor `[N]`): marks terminations caused by time limits (`:94-96`); read at `ppo.py:154-158`.
  - `extras["log"]` (or `extras["episode"]`): logging only (`:98-100`, `logger.py:130-133`).
- Which `env` members the library actually touches: `get_observations`, `step`, `num_envs`, `num_actions`, `device` (`on_policy_runner.py:88`),
  `cfg` (`on_policy_runner.py:45`, stored by `LogWriter.store_config`), `episode_length_buf` and `max_episode_length` only for
  `init_at_random_ep_len` (`:59-62`), `env.unwrapped.step_dt` only with RND (`rnd.py:239`).

---------------------------------------------------------------------------------------------------------------------

## 9. Multi-GPU path

Process model: one process per GPU, started externally with `WORLD_SIZE`, `RANK`, `LOCAL_RANK` env vars; NCCL and CUDA are hard-coded
(`on_policy_runner.py:215-257`). Each rank has its own `env` and its own rollout storage with `num_envs` environments per rank (implied by the
total-timesteps formula `T * N * world_size`, `logger.py:176`). Only rank 0 logs (`logger.py:67`).

| What | Synced? | How | Source |
|---|---|---|---|
| Initial parameters AND buffers of actor/critic (+ RND predictor/target) | yes, once, at the start of `learn()` | `for t in model.state_dict().values(): broadcast(t, src=0)`; includes normalizer `_mean/_var/_std/count` and the std parameter. NOT done at construction, NOT after `load()` | `ppo.py:467-474`, `on_policy_runner.py:69-71` |
| Optimizer state | no | each rank keeps its own Adam moments (identical in practice because gradients are identical) | n/a |
| Gradients | yes, every mini-batch, mean | after `backward()` and BEFORE clipping: bucketed `all_reduce(SUM)` then `/ world_size`, over `chain(actor.parameters(), critic.parameters()[, rnd.parameters()])`, only params with `grad is not None`; bucket size `grad_reduce_bucket_mb` (default 25 MB); a gradient larger than a bucket is reduced in contiguous slices | `ppo.py:305-306,476-486`, `utils.py:345-392` |
| Gradient clipping | local, but on already-averaged grads, so identical on every rank | `clip_grad_norm_` per model | `ppo.py:309-310` |
| Observation-normalization statistics | yes, GLOBAL | inside `EmpiricalNormalization.update`, keyed on `torch.distributed.is_initialized()` (NOT on `multi_gpu_cfg`): `all_reduce(count)`, `all_reduce(mean * n)`, then `all_reduce(n * (var + (local_mean - global_mean)^2))`, giving statistics of the union of all ranks' batches | `normalization.py:62-71` |
| Advantage normalization | NO, rank-local | global-over-local-rollout (default) or per-local-mini-batch; no collective | `ppo.py:190-191,214-216` |
| KL used by the adaptive schedule | yes | `kl_mean` per rank, `all_reduce(SUM)`, `/= world_size` | `ppo.py:246-249` |
| Learning rate | yes | rank 0 applies the rule; then `lr_tensor = torch.tensor(self.learning_rate, device=...)` (float32 by default), `broadcast(src=0)`, and EVERY rank (including rank 0) sets `self.learning_rate = lr_tensor.item()` | `ppo.py:251-262` |
| Loss / entropy numbers | no | local means, only rank 0 logs | `ppo.py:317-319` |
| RNG / permutation | no | each rank draws its own noise and its own `randperm`; upstream never seeds (`grep seed` finds nothing under `rsl_rl/`) | `rollout_storage.py:232` |
| Env stepping | independent per rank | | n/a |

The mean gradient equals the mean of the per-rank mini-batch-mean losses' gradients (ranks use equal mini-batch sizes). Tests:
`tests/algorithms/test_distributed.py` (gloo, 2 ranks).

---------------------------------------------------------------------------------------------------------------------

## 10. Checkpoints (`save` / `load`)

### 10.1 Checkpoint dict

Written by `OnPolicyRunner.save` -> `torch.save` (pickle):

| Key | Content | Source |
|---|---|---|
| `actor_state_dict` | `_raw_actor.state_dict()` (params AND buffers, section 5.1 for the key layout) | `ppo.py:377` |
| `critic_state_dict` | `_raw_critic.state_dict()` | `ppo.py:378` |
| `optimizer_state_dict` | `optimizer.state_dict()`: torch format (`state` keyed by integer param index, plus `param_groups` with `lr` and `params` indices). The indices follow the order of `dict.fromkeys(chain(actor.parameters(), critic.parameters()))` | `ppo.py:379,100-102` |
| `rnd_state_dict`, `rnd_optimizer_state_dict`, `rnd_update_counter` | only if RND | `ppo.py:381-384` |
| `iter` | `current_learning_iteration` | `on_policy_runner.py:147` |
| `infos` | the `infos` argument of `save` (default `None`) | `on_policy_runner.py:148` |

NOT saved: RNG state, storage, normalizer anything beyond the buffers inside the state_dicts, the schedule/`desired_kl` settings, the
`transition` scratch.

### 10.2 What `load()` touches directly

`OnPolicyRunner.load` (`:153-169`): `torch.load(path, weights_only=False, map_location=map_location)` (so a GPU-saved checkpoint needs
`map_location="cpu"` on a host without CUDA), then `alg.load(...)`, then `current_learning_iteration = loaded_dict["iter"]` (if
`alg.load` returned truthy), returns `loaded_dict["infos"]`.

`PPO.load(loaded_dict, load_cfg, strict)` (`ppo.py:387-411`): `load_cfg=None` means `{actor, critic, optimizer, iteration, rnd: True}`
(`:390-397`).
- `actor`: `_raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)` (`:400-401`)
- `critic`: `_raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)` (`:402-403`)
- `optimizer`: `optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])` and `self.learning_rate = optimizer.param_groups[0]["lr"]` (`:404-406`)
- `rnd`: only with RND (`:407-410`)
- returns `load_cfg.get("iteration", False)` (`:411`)

So the algorithm's checkpoint contract is: actor/critic expose `state_dict()/load_state_dict()`; the optimizer exposes
`state_dict()/load_state_dict()` and `param_groups[0]["lr"]`; `learning_rate` is a settable attribute.

---------------------------------------------------------------------------------------------------------------------

## 11. `get_policy()` and export

- `PPO.get_policy()` returns `_raw_actor` (`ppo.py:413-415`), the live actor, not a copy and not the compiled wrapper.
- Inference: `policy = runner.get_inference_policy(device)`; `policy(obs)` runs `MLPModel.forward` with `stochastic_output=False`, so it
  returns the Gaussian MEAN (`mlp_model.py:107`, `distribution.py:197-199`), after normalization in eval mode (no stat updates, `normalization.py:53-54`).
  No action clipping or squashing exists anywhere.
- JIT: `torch.jit.script(_TorchMLPModel)` after `.to("cpu")`; input is the concatenated, un-normalized observation tensor `[B, obs_dim]`
  (normalizer is inside the exported module). ONNX: same graph, input name `obs`, output `actions`, opset 18.
- Both exporters operate on deep copies of `obs_normalizer` and `mlp` (`mlp_model.py:202-203,229-230`). A policy object that is a torch
  `MLPModel` mirror with downloaded weights reuses the exporters unchanged; a policy that holds device tensors would need its own
  `as_jit`/`as_onnx`.

---------------------------------------------------------------------------------------------------------------------

## 12. Extensions and distillation (out of v1, for completeness)

- RND (`extensions/rnd.py`) and Symmetry (`extensions/symmetry.py`) are constructed inside `PPO.__init__` from `rnd_cfg` / `symmetry_cfg`
  (`ppo.py:84-89`). `resolve_rnd_config` reads `env.unwrapped.step_dt` (`rnd.py:239`). The runner and logger read `alg.rnd` only if
  `cfg["algorithm"]["rnd_cfg"]` is truthy.
- `Distillation` (`algorithms/distillation.py`) has its own `construct_algorithm` (`:263-300`) and interface (`act`, `process_env_step`,
  `compute_returns` no-op, `update`, `save`, `load`, `get_policy` returns the student). It confirms that the runner works with any class
  exposing the section 2.5 surface, because `DistillationRunner` only subclasses to add a teacher check.

---------------------------------------------------------------------------------------------------------------------

## 13. INJECTION POINTS

### 13.1 Mechanism per replaceable component

| Component to replace | Mechanism (cfg key -> code) | Pure cfg? | What stays hard-coded / remaining requirement |
|---|---|---|---|
| Whole algorithm (`TTPPO`) | `cfg["algorithm"]["class_name"]` -> `resolve_callable` (`on_policy_runner.py:39`) -> `class.construct_algorithm(obs, env, cfg, device)` (`:40`) | YES | Must be a class with a static `construct_algorithm` and the section 2.5 surface. Subclassing `PPO` is optional (no `isinstance` check). Use a qualified name or the class object, NOT a bare name (section 3.1) |
| Algorithm construction (what actor/critic/storage/optimizer get built) | `construct_algorithm` is owned by the injected class, so a TT class can override it completely | YES | If instead the TT class inherits `PPO.construct_algorithm`, then: (a) torch actor/critic are built from cfg first, (b) an upstream torch `RolloutStorage` is built on `device`, (c) the TT constructor receives `rnd_cfg=None, symmetry_cfg=None, multi_gpu_cfg=<None or dict>` plus every other key of `cfg["algorithm"]` as kwargs, (d) `alg.compile(cfg.get("torch_compile_mode"))` is called (`ppo.py:463`), so a `compile(mode)` method is required |
| Actor / critic model classes | `cfg["actor"]["class_name"]`, `cfg["critic"]["class_name"]` -> `resolve_class` in `PPO.construct_algorithm` (`ppo.py:433-434`) | YES, when the algorithm uses `PPO.construct_algorithm` or an override that reads the same keys | Called as `cls(obs, obs_groups, "actor"/"critic", num_actions or 1, **cfg_rest)` and then `.to(device)` (`ppo.py:449,453`); unknown keys raise `TypeError` unless the class takes `**kwargs` |
| Distribution | `<model>["distribution_cfg"]["class_name"]` -> `resolve_class` in `MLPModel.__init__` (`mlp_model.py:67-69`) | YES, but only for `MLPModel`-derived models | A TT model that does not derive from `MLPModel` must implement its own resolution. The interface is in section 6.2 |
| Obs normalization | only the bool `obs_normalization`; the class is hard-coded `EmpiricalNormalization` (`mlp_model.py:62`) | NO (not class-injectable) | The TT model class implements its own normalizer; for checkpoint compat it must expose buffers `_mean`, `_var`, `_std`, `count` in the model state_dict |
| Rollout storage | hard-coded in `PPO.construct_algorithm` (`ppo.py:457`) | NO via cfg | Overriding `construct_algorithm` in the TT class (allowed, section above) or replacing `alg.storage` in the TT constructor; the generator is only consumed inside `PPO.update`, which a TT algorithm replaces anyway |
| Optimizer | `optimizer` is a name string resolved by `resolve_optimizer` to one of 4 torch optimizers (`utils.py:71-96`, `ppo.py:100`) | NO | A TT algorithm implements its own optimizer; it should still accept the `optimizer` kwarg (`"adam"`) so the cfg keeps its format, and expose `state_dict()/load_state_dict()/param_groups` (`ppo.py:379,405-406`) |
| Logger | `cfg["logger"]["class_name"]` (`logger.py:107-108`) | YES | not needed for TT |
| Runner class | not a cfg key; chosen by the caller (`OnPolicyRunner(...)`) | n/a | [UNVERIFIED] whether Isaac Lab's scripts allow another runner class |
| Device for the runner | constructor arg `device`; `"cpu"` is accepted and makes every runner `.to(device)` a no-op for CPU tensors | YES | `act` must return a torch tensor; TT staging/uploading is internal to the algorithm |

### 13.2 What stays hard-coded in the runner (cannot be changed by cfg)

1. The algorithm method names and call order of section 2.5; the runner never touches `alg.actor/critic/storage/optimizer`
   (so the "runner assumes `alg.actor.train()`" worry does not apply in 5.5.2: it calls `alg.train_mode()`, `on_policy_runner.py:66`).
2. `torch.inference_mode()` around `act`, `process_env_step`, `compute_returns` (`:83-106`) and NOT around `update` (`:109`).
   Tensors created inside that block are inference tensors; in-place updates to them outside the block raise in torch [TORCH].
3. Obs/rewards/dones are torch tensors moved with `.to(self.device)` (`:93`); actions are torch and moved with `.to(env.device)` (`:88`);
   `check_nan` runs on torch tensors (`:91`, `utils.py:290-316`).
4. `alg.get_policy().output_std` is evaluated EVERY iteration (`:124`), and `alg.learning_rate` must be a python float every iteration (`:123`).
5. `torch.save` / `torch.load(weights_only=False)` of a plain dict with the extra keys `iter` and `infos` (`:146-149,165-169`).
6. `torch.jit.script` for JIT export and `torch.onnx.export(..., opset_version=18)` for ONNX (`:186,200-209`).
7. Multi-process mode is NCCL + CUDA only (`:240,255,257`).
8. The runner mutates the caller's cfg (`cfg["multi_gpu"]`, `cfg["obs_groups"]` in place).
9. Logger formatting uses `loss_dict` values as floats and `cfg["algorithm"].get("rnd_cfg")` (`logger.py:59,206-208,221,252-253,257`).

### 13.3 Every place upstream reaches into algorithm internals (what a TT algorithm class must expose)

Runner and logger (complete list): section 2.5.

In-repo callers that touch attributes (evidence of the de-facto public surface):
- `runner.alg.actor`, `runner.alg.critic`, with `.named_parameters()` and `.state_dict()` (`tests/runners/test_on_policy_runner.py:147-148,163,165,195,198,203,232,240,278`);
- `runner.logger.writer` (`:329,337`), `runner.env`, `runner.current_learning_iteration` (`:153,172,213,219`);
- PPO tests use `ppo.storage` (`.advantages`, `.returns`, `.values`, `.actions_log_prob`, `.rewards`, `.observations`), `ppo.transition`, `ppo.optimizer.param_groups`,
  `ppo.learning_rate`, `ppo.gamma`, `ppo.actor.get_kl_divergence`, `ppo.critic.forward` (`tests/algorithms/test_ppo.py`).
Isaac Lab and other downstream callers are outside this source: [UNVERIFIED]. Before declaring the TT class compatible, grep Isaac Lab's
rsl_rl integration for `runner.alg.*` accesses (open question Q1).

What `PPO` itself reaches into on the models (a TT model class used with an upstream-style algorithm, or a TT algorithm that imitates PPO):
`is_recurrent`, `forward(obs, masks=, hidden_state=, stochastic_output=)`, `get_hidden_state()`, `reset(dones=None, hidden_state=None)`,
`get_output_log_prob`, `output_distribution_params`, `output_entropy`, `get_kl_divergence`, `update_normalization`, `parameters()`,
`state_dict/load_state_dict`, `train/eval`, `to`, `as_jit`, `as_onnx`, `output_std` (`ppo.py:87,127,129-132,163-164,173,204,228-238,243,309-310,329-330,401-403`; `on_policy_runner.py:124,174,178,191`).

What `PPO` reaches into on the storage: `add_transition`, `clear`, `num_transitions_per_env`, `observations`, `values`, `rewards`, `dones`, `returns`,
`advantages` (assigned), `mini_batch_generator`, `recurrent_mini_batch_generator` (`ppo.py:161,168-191,205-207,328,356`).

What `MLPModel` reaches into on the distribution: `update`, `sample`, `deterministic_output`, `input_dim`, `init_mlp_weights`, `mean`, `std`, `entropy`, `params`,
`log_prob`, `kl_divergence`, `as_deterministic_output_module` (`mlp_model.py:67-80,103-107,137-160,205`).

### 13.4 Where a thin `TTOnPolicyRunner` or an upstream patch would be needed

| Situation | Needed? | Concrete reason | Minimal change |
|---|---|---|---|
| Single process, one `MeshDevice` for 1 to 8 chips (data parallel inside the algorithm), runner device `"cpu"` | NO subclass, NO patch | everything above is reachable through `class_name` plus the `construct_algorithm` static method | none |
| Multi-process (one process per chip group) | subclass | `_configure_multi_gpu` hard-codes NCCL and `cuda:<local_rank>` (`on_policy_runner.py:215-257`) | override `_configure_multi_gpu` to set `self.cfg["multi_gpu"]` and an alternative process group; also `alg.broadcast_parameters` then runs (`:69-71`) and `EmpiricalNormalization.update` would use `torch.distributed` if initialized (`normalization.py:62`) |
| Deterministic device close / trace release at the end of `learn()` | optional | the runner has no finalizer hook; `finally`-like handling exists only for the logger (`:136-142`) | prefer `atexit` plus an explicit `alg.close()` in the TT class; a subclass wrapping `learn()` in `try/finally` is the fallback |
| `get_policy()` being called every iteration (`:124`) | NO | only a performance trap if `get_policy()` rebuilds a mirror from device weights | cache the mirror, make `output_std` a cheap read from the std parameter; no runner change |
| Fused act/step/transfer loops, per-iteration trace begin/end hooks | optional | `learn()` is monolithic (`:56-142`) | a subclass overriding `learn()` would have to copy it; prefer doing the work inside `act`/`compute_returns`/`update` |
| Per-iteration fps/`collection_size` accounting for multi-chip in one process | cosmetic | `Logger.log` multiplies by `gpu_world_size` which is 1 (`logger.py:176`) | ignore it; the `Logger` class itself is hard-coded (`on_policy_runner.py:43`), `cfg["logger"]` only selects the writer backend |
| Upstream source patch | NONE required for v1 | no hard-coded dependency forces one | optional convenience patches (injecting storage/optimizer by class name) are not required and should not be proposed unless a later phase needs them |

### 13.5 Recommended wiring (derived from the above; a design suggestion, not an upstream fact)

`cfg["algorithm"]["class_name"] = "rsl_rl_tt.<module>:TTPPO"`, runner `device="cpu"`. Two viable construction routes:

- Route A (own `construct_algorithm`): full control, no torch storage/optimizer built, but must re-implement obs-group resolution by calling
  `rsl_rl.utils.resolve_obs_groups` and replicate the `cfg["multi_gpu"]` and `torch_compile_mode` handling.
- Route B (inherit `PPO.construct_algorithm`): gets obs-group validation and torch-default weight init for free (identical init stream to
  upstream for a given seed, which makes initial-weight parity trivial); the TT `__init__` then uploads the torch actor/critic weights and may
  discard the upstream storage. Requires the TT `__init__` to accept `rnd_cfg`, `symmetry_cfg`, `multi_gpu_cfg`, `grad_reduce_bucket_mb` and any
  extra TT kwargs, and to implement `compile(mode)`.

`get_policy()` should return an upstream `MLPModel` mirror (so `as_jit/as_onnx` and `output_std` work unchanged).

---------------------------------------------------------------------------------------------------------------------

## 14. Open questions (cannot be resolved from the pinned source)

- Q1. Which `runner.alg.*` / `runner.alg.actor.*` / `get_policy()` attributes does Isaac Lab's rsl_rl wrapper, train/play scripts, and its
  export utilities access? (Not in this repo.)
- Q2. Can Isaac Lab's agent cfg classes carry extra keys (for example a `tt_cfg` dict inside `algorithm`), or must TT settings come from a
  separate YAML / environment variables? `PPO.__init__` rejects unknown keys, so the TT class must consume them.
- Q3. dtype of `dones` and `extras["time_outs"]` as produced by Isaac Lab (the library accepts float or bool because of `copy_` and promotion).
- Q4. Installed torch/tensordict versions at test time (the `.venv` did not exist when this was written); all `[TORCH]` statements need a
  confirming test against the installed version.
- Q5. Whether `torch.onnx.export` in the installed torch uses the legacy or the dynamo exporter (affects ONNX export only, no TT impact).
