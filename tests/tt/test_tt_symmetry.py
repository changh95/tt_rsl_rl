# SPDX-License-Identifier: BSD-3-Clause
"""Phase 8.3 device tests, part 1 (chip-count independent): the mirror maps on the device (exact for signed permutations, the matmul floor otherwise), the algorithm built from a symmetry
configuration (extra columns of the packed table, the refusals), and checkpoints with the extension on.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p8s -- python -m pytest -m tt tests/tt/test_tt_symmetry.py tests/tt/test_tt_symmetry_update.py -s -q -p no:cacheprovider
"""

import json
import torch
from pathlib import Path

import pytest

from rsl_rl_tt.symmetry import MirrorMaps, _DeviceMap
from rsl_rl_tt.testing.parity import TOLERANCES, compare
from rsl_rl_tt.testing.reference import RefSpec, symmetry_cfg
from tests.tt.test_tt_checkpoint import build_upstream_runner, shape_tree
from tests.tt.test_tt_rollout import build_tt_runner, small_spec

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
MEASURED: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _dump():
    yield
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase8_3_symmetry.json").write_text(json.dumps(MEASURED, indent=2, default=str))


def rec(key, value):
    MEASURED[key] = value
    print(f"[measure] {key}: {value}")
    return value


# (env, symmetry function, groups) of the configurations of the tests
CASES = {
    "cartpole-K2": ("cartpole", "cartpole_mirror"),
    "tracking12-K2": ("tracking12", "tracking12_sign_mirror"),
    "tracking12-K4": ("tracking12", "tracking12_dihedral_mirror"),
}


def sym_spec(case, **kwargs) -> tuple[str, RefSpec]:
    name, func = CASES[case]
    base = small_spec(name)
    return name, RefSpec(**{**base.__dict__, "algorithm": {**base.algorithm, "symmetry_cfg": symmetry_cfg(name, func=func, **kwargs)}})


def wide_inputs(rows, d, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, d, generator=g) * torch.pow(10.0, torch.randint(-3, 3, (rows, d), generator=g).float())  # values over six decades: any rounding of the input shows


def test_a_signed_permutation_is_applied_exactly_and_a_general_matrix_at_the_matmul_floor(tt_ctx):
    ctx = tt_ctx
    rows, d = 256 * ctx.n_chips, 64
    g = torch.Generator().manual_seed(1)
    perm = torch.randperm(d, generator=g)
    sign = torch.where(torch.rand(d, generator=g) < 0.5, -1.0, 1.0)
    w = torch.zeros(d, d)
    w[perm, torch.arange(d)] = sign
    b = torch.where(torch.rand(d, generator=g) < 0.5, 0.25, 0.0)
    x = wide_inputs(rows, d, 2)
    xd = ctx.upload(x, op="H-1", batch_sharded=True)
    out = ctx.download(_DeviceMap(ctx, w, torch.zeros(d))(xd), op="H-9")
    assert torch.equal(out, x @ w), "a signed permutation must be applied bit for bit"
    out_b = ctx.download(_DeviceMap(ctx, w, b)(xd), op="H-9")
    assert torch.equal(out_b, x @ w + b), "the bias of an affine map is added exactly"
    general = torch.randn(d, d, generator=g) * 0.3
    m = _DeviceMap(ctx, general, torch.zeros(d))
    assert not m.exact
    rep = compare(ctx.download(m(xd), op="H-9"), x @ general, "matmul_fp32", "general matrix")
    rec("mirror map general matrix", {"rel_fro": rep.rel_fro})
    assert rep.passed, str(rep)


@pytest.mark.parametrize("case", ["cartpole-K2", "tracking12-K4"])
def test_the_device_maps_equal_the_user_function_on_the_observations_and_actions_of_a_rollout(tt_mesh, case, tmp_path):
    """The mirrored copies of a real rollout, computed on the device (extra columns of the packed table), equal what the user function returns for the same observations and actions."""
    from rsl_rl_tt.tt_common import stage_obs

    name, spec = sym_spec(case)
    runner, env, cfg = build_tt_runner(name, tmp_path, spec=spec)
    alg = runner.alg
    sym = alg.sym_tt
    K = sym.num_aug
    assert K == {"cartpole-K2": 2, "tracking12-K4": 4}[case] and alg.storage.extra_cols and sym.use_da and sym.use_loss
    alg.train_mode()
    ctx = alg.ctx
    obs = env.get_observations()
    observed, acts = [], []
    with torch.inference_mode():
        for _ in range(spec.num_steps):
            observed.append(obs.clone())
            actions = alg.act(obs)
            acts.append(actions.clone())
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
        alg.compute_returns(obs)
    st = alg.storage
    table = ctx.download(st.packed_table(extra=sym.mirror_blocks(st)), op="H-6")
    m = env.num_envs // ctx.n_chips
    order = lambda per_step: torch.cat([torch.cat([t[c * m : (c + 1) * m] for t in per_step]) for c in range(ctx.n_chips)])  # chip c: its envs of step 0, 1, ...
    allobs = torch.cat([o for o in observed])  # [T * N] rows in step-major order
    all_act = torch.cat(acts)
    func = sym.mirror.data_augmentation_func
    # the function applied to the whole rollout: the original rows first, then each copy; (the rows of a chip are reordered like the storage's)
    aug_obs, aug_act = func(env=env, obs=allobs, actions=all_act)
    n = allobs.batch_size[0]
    for k in range(1, K):
        want_actor = stage_obs(aug_obs[k * n : (k + 1) * n], alg.tt_actor.obs_groups, alg.tt_actor.obs_dim_p)
        a, b = st.extra_cols[f"sym_actor_{k}"]
        got = table[:, a:b]
        want = order([want_actor[t * env.num_envs : (t + 1) * env.num_envs] for t in range(spec.num_steps)])
        assert torch.equal(got, want), f"copy {k}: the mirrored actor observations differ from the user function's"
        a, b = st.extra_cols[f"sym_action_{k}"]
        want_a = torch.zeros(n, st.act_dim_p)
        want_a[:, : env.num_actions] = aug_act[k * n : (k + 1) * n]
        assert torch.equal(table[:, a:b], order([want_a[t * env.num_envs : (t + 1) * env.num_envs] for t in range(spec.num_steps)])), f"copy {k}: the mirrored actions differ"
        if not st.share_obs:
            a, b = st.extra_cols[f"sym_critic_{k}"]
            want_c = stage_obs(aug_obs[k * n : (k + 1) * n], alg.tt_critic.obs_groups, alg.tt_critic.obs_dim_p)
            assert torch.equal(table[:, a:b], order([want_c[t * env.num_envs : (t + 1) * env.num_envs] for t in range(spec.num_steps)])), f"copy {k}: the mirrored critic observations differ"


def test_the_algorithm_refuses_what_it_cannot_do_with_a_reason(tt_mesh, tmp_path):
    name, spec = sym_spec("tracking12-K2", mirror_loss=False)
    bad = RefSpec(**{**spec.__dict__, "algorithm": {**spec.algorithm, "symmetry_cfg": symmetry_cfg(name, func="tracking12_nonaffine_mirror")}})
    with pytest.raises(NotImplementedError, match="not affine"):
        build_tt_runner(name, tmp_path / "a", spec=bad)
    hetero = RefSpec(**{**spec.__dict__, "distribution": "HeteroscedasticGaussianDistribution"})
    with pytest.raises(NotImplementedError, match="Heteroscedastic"):
        build_tt_runner(name, tmp_path / "b", spec=hetero)


@pytest.mark.parametrize("case, kwargs", [("cartpole-K2", {}), ("tracking12-K4", {"data_augmentation": False}), ("tracking12-K2", {"data_augmentation": False, "mirror_loss": False})])
def test_training_runs_and_logs_the_symmetry_loss(tt_mesh, case, kwargs, tmp_path):
    """The unmodified runner trains with the extension in each of its modes (augmentation + mirror loss, the mirror loss alone, logging only); `Loss/symmetry` is logged."""
    name, spec = sym_spec(case, **kwargs)
    runner, env, cfg = build_tt_runner(name, tmp_path, spec=spec)
    runner.learn(num_learning_iterations=3)
    rows = (tmp_path / "metrics.csv").read_text().splitlines()
    cols = rows[0].split(",")
    assert "Loss/symmetry" in cols
    losses = [float(r.split(",")[cols.index("Loss/symmetry")]) for r in rows[1:]]
    assert all(0.0 < v < 1e3 for v in losses), losses
    sym = runner.alg.sym_tt
    assert sym.use_da == kwargs.get("data_augmentation", True) and sym.use_loss == kwargs.get("mirror_loss", True)


def test_the_checkpoint_of_a_run_with_symmetry_has_the_structure_of_an_upstream_one(tt_mesh, tmp_path):
    """The extension keeps no state of its own: the checkpoint of a TT run is loadable by an upstream run of the same configuration and structurally identical to its checkpoint."""
    name, spec = sym_spec("tracking12-K2")
    tt_runner, tt_env, _ = build_tt_runner(name, tmp_path / "tt", spec=spec)
    tt_runner.learn(num_learning_iterations=2)
    ckpt = tmp_path / "tt.pt"
    tt_runner.save(str(ckpt))
    up_runner, _ = build_upstream_runner(name, tmp_path / "up", spec=spec)
    up_runner.load(str(ckpt), map_location="cpu")
    obs = tt_env.get_observations()
    with torch.no_grad():
        assert torch.equal(up_runner.get_inference_policy()(obs), tt_runner.get_inference_policy()(obs))
    up_runner.learn(num_learning_iterations=2)
    ref_ckpt = tmp_path / "up.pt"
    up_runner.save(str(ref_ckpt))
    a, b = torch.load(ckpt, weights_only=False), torch.load(ref_ckpt, weights_only=False)
    assert set(a) == set(b)
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        assert shape_tree(a[key]) == shape_tree(b[key])
    tt_runner.load(str(ckpt), map_location="cpu")  # and a TT run resumes from it with the extension on
    tt_runner.learn(num_learning_iterations=1)
