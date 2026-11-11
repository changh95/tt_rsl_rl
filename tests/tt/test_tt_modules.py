# SPDX-License-Identifier: BSD-3-Clause
"""Phase 2 device tests, one component at a time (BRIEF Phase 2: per-component forward parity, random shapes incl. padding):
TTLinear / TTMLP / TTMLPModel forward, TTEmpiricalNormalization (normalize + statistics update), TTGaussianDistribution
(sample / log_prob / entropy), mirror synchronisation, host-op accounting.

    scripts/tt_run.sh --chips 16 --timeout 900 --tag p2mod -- python -m pytest -m tt tests/tt/test_tt_modules.py -s -q -p no:cacheprovider

The measured errors are collected in `.run/results/phase2_modules.json`; docs/porting/TOLERANCES.md derives the composite tolerances from them.
"""

import copy
import json
import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, GaussianDistribution
from rsl_rl_tt.testing.parity import TOLERANCES, compare
from rsl_rl_tt.tt_common import pad_matrix, pad_up

pytestmark = pytest.mark.tt

RESULTS = Path(__file__).resolve().parents[2] / ".run" / "results"
MEASURED: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _dump_results():
    yield
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase2_modules.json").write_text(json.dumps(MEASURED, indent=2, default=str))


def _record(key, rep):
    MEASURED[key] = {"rel_fro": rep.rel_fro, "err_rms": rep.err_rms, "ref_rms": rep.ref_rms, "max_abs": rep.max_abs, "max_rel": rep.max_rel, "max_ulp": rep.max_ulp, "pcc": rep.pcc}
    print(f"[measure] {key}: rel_fro={rep.rel_fro:.3e} err_rms={rep.err_rms:.3e} ref_rms={rep.ref_rms:.3e} max_rel={rep.max_rel:.3e} max_abs={rep.max_abs:.3e} max_ulp={rep.max_ulp:.3g}")
    return rep


def make_obs(rows, groups: dict[str, int], seed=0, scale=1.0, offset=0.0):
    g = torch.Generator().manual_seed(seed)
    return TensorDict({k: torch.randn(rows, d, generator=g) * scale + offset for k, d in groups.items()}, batch_size=[rows])


def make_model(groups, used, out_dim, hidden, activation="elu", norm=False, dist=True, seed=0, std_type="scalar", init_std=1.0, std_range=(1e-6, 1e6)):
    torch.manual_seed(seed)
    obs = make_obs(64, groups, seed=99)
    cfg = {"class_name": "GaussianDistribution", "init_std": init_std, "std_type": std_type, "std_range": std_range} if dist else None
    model = MLPModel(obs, {"actor": used}, "actor", out_dim, hidden_dims=hidden, activation=activation, obs_normalization=norm, distribution_cfg=cfg)
    return model


def feed_normalizer(model, groups, batches=3, rows=256, seed=5):
    """Give the torch model's normalizer non-trivial statistics (so the test exercises real mean/std, not the identity)."""
    for i in range(batches):
        obs = make_obs(rows, groups, seed=seed + i, scale=2.5, offset=1.5)
        model.update_normalization(obs)


# ---- TTLinear ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_in, n_out", [(4, 64), (36, 50), (48, 100), (100, 12), (32, 32), (1, 33)])
@pytest.mark.parametrize("rows", [32, 96, 1024])
def test_linear_parity_with_padding(tt_ctx, n_in, n_out, rows):
    from rsl_rl_tt.modules import TTLinear

    torch.manual_seed(1)
    lin = torch.nn.Linear(n_in, n_out)
    x = torch.randn(rows, n_in)
    tt = TTLinear(tt_ctx, lin)
    out = tt_ctx.download(tt.forward(tt_ctx.upload(pad_matrix(x, rows, pad_up(n_in)), op="H-1", batch_sharded=True)), op="H-2")
    assert out.shape == (rows, pad_up(n_out))
    assert torch.count_nonzero(out[:, n_out:]) == 0, "padded output columns must stay exactly zero"
    ref = (x.double() @ lin.weight.detach().double().t() + lin.bias.detach().double())
    rep = _record(f"linear[{n_in}->{n_out}] rows={rows}", compare(out[:, :n_out], ref, TOLERANCES["matmul_fp32"]))
    assert rep.passed, str(rep)


# ---- MLP model forward ---------------------------------------------------------------------------------------------------------
MODEL_CASES = [
    # groups, used groups, out_dim, hidden, activation, normalization
    ({"policy": 4}, ["policy"], 1, [64, 64], "elu", True),
    ({"policy": 36, "privileged": 48}, ["policy", "privileged"], 1, [128, 128, 64], "elu", True),
    ({"policy": 36, "privileged": 48}, ["policy"], 12, [128, 128, 64], "elu", True),
    ({"policy": 48}, ["policy"], 29, [50, 70], "elu", False),
    ({"policy": 17}, ["policy"], 5, [33, 33, 33], "tanh", True),
    ({"policy": 9}, ["policy"], 3, [96], "relu", False),
    ({"a": 10, "b": 7, "c": 3}, ["c", "a"], 7, [256, 256, 128], "elu", True),
]


@pytest.mark.parametrize("groups, used, out_dim, hidden, act, norm", MODEL_CASES, ids=[f"case{i}" for i in range(len(MODEL_CASES))])
@pytest.mark.parametrize("rows", [32, 160, 4096])
def test_model_forward_parity(tt_ctx, groups, used, out_dim, hidden, act, norm, rows):
    from rsl_rl_tt.modules import TTMLPModel

    model = make_model(groups, used, out_dim, hidden, act, norm, dist=True)
    if norm:
        feed_normalizer(model, groups)
    tt = TTMLPModel(tt_ctx, model)
    rows *= tt_ctx.n_chips  # several chips: `rows` rows PER CHIP (the rows are independent, so the result must not depend on how many chips share them)
    obs = make_obs(rows, groups, seed=7, scale=2.0, offset=0.5)
    with torch.no_grad():
        ref = model(obs)  # deterministic output = distribution mean
    out = tt_ctx.download(tt.forward(tt.obs_to_device(obs)), op="H-2")
    assert out.shape == (rows, pad_up(out_dim)) and torch.count_nonzero(out[:, out_dim:]) == 0
    rep = _record(f"model[{hidden},{act},norm={norm},out={out_dim}] rows={rows}", compare(out[:, :out_dim], ref, TOLERANCES["mlp_forward"]))
    assert rep.passed, str(rep)


def test_model_without_distribution_is_a_critic(tt_ctx):
    from rsl_rl_tt.modules import TTMLPModel

    model = make_model({"policy": 36, "privileged": 48}, ["policy", "privileged"], 1, [128, 64], dist=False, norm=True)
    feed_normalizer(model, {"policy": 36, "privileged": 48})
    tt = TTMLPModel(tt_ctx, model)
    rows = max(128, 32 * tt_ctx.n_chips)
    obs = make_obs(rows, {"policy": 36, "privileged": 48}, seed=3)
    with torch.no_grad():
        ref = model(obs)
    out = tt_ctx.download(tt.forward(tt.obs_to_device(obs))[:, 0:1], op="H-2")
    assert tt.distribution is None and out.shape == (rows, 1)
    rep = _record(f"critic[128,64,norm] rows={rows}", compare(out, ref, TOLERANCES["mlp_forward"]))
    assert rep.passed, str(rep)


# ---- EmpiricalNormalization -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [4, 36, 84])
def test_normalizer_forward_matches_upstream(tt_ctx, dim):
    from rsl_rl_tt.modules import TTEmpiricalNormalization

    norm = EmpiricalNormalization(dim)
    for i in range(4):
        norm.update(torch.randn(512, dim, generator=torch.Generator().manual_seed(i)) * (1 + i) + i)
    tt = TTEmpiricalNormalization(tt_ctx, norm, dim)
    x = torch.randn(256, dim, generator=torch.Generator().manual_seed(11)) * 3 + 1
    out = tt_ctx.download(tt.forward(tt_ctx.upload(pad_matrix(x, 256, pad_up(dim)), op="H-1", batch_sharded=True)), op="H-2")
    rep = _record(f"normalizer.forward dim={dim}", compare(out[:, :dim], norm(x), TOLERANCES["eltwise_fp32"]))
    assert rep.passed, str(rep)
    assert torch.count_nonzero(out[:, dim:]) == 0  # padded columns stay zero (x = 0, mean = 0)


@pytest.mark.parametrize("dim", [4, 36, 84])
def test_normalizer_update_matches_upstream_over_many_batches(tt_ctx, dim):
    from rsl_rl_tt.modules import TTEmpiricalNormalization

    ref, mirror = EmpiricalNormalization(dim), EmpiricalNormalization(dim)
    tt = TTEmpiricalNormalization(tt_ctx, mirror, dim)
    n = tt_ctx.n_chips  # several chips: every chip holds `local` rows, the statistics are those of the UNION (upstream torch.distributed semantics, ADR-0004)
    for i, local in enumerate([512, 96, 4096, 32, 1024, 2048]):
        rows = local * n
        x = torch.randn(rows, dim, generator=torch.Generator().manual_seed(100 + i)) * (0.5 + i) + 3.0 * (i % 2)
        ref.update(x)
        tt.update(tt_ctx.upload(pad_matrix(x, rows, pad_up(dim)), op="H-1", batch_sharded=True), local)
        s = tt.host_stats()
        assert s["count"] == int(ref.count)
        r_mean = _record(f"normalizer.update mean dim={dim} step={i}", compare(s["mean"], ref.mean, TOLERANCES["normalizer_update"]))
        r_var = _record(f"normalizer.update var dim={dim} step={i}", compare(s["var"], ref._var.reshape(-1), TOLERANCES["normalizer_update"]))
        assert r_mean.passed and r_var.passed, str(r_mean) + str(r_var)


def test_normalizer_respects_until_and_eval_mode(tt_ctx):
    from rsl_rl_tt.modules import TTEmpiricalNormalization

    n = tt_ctx.n_chips  # the count is global: every update adds 64 rows per chip
    norm = EmpiricalNormalization(8, until=100 * n)
    tt = TTEmpiricalNormalization(tt_ctx, norm, 8)
    x = pad_matrix(torch.randn(64 * n, 8), 64 * n, 32)
    dev = tt_ctx.upload(x, op="H-1", batch_sharded=True)
    tt.update(dev, 64)
    assert tt.count == 64 * n
    tt.update(dev, 64)  # 64 n < until = 100 n -> still learns, count 128 n
    assert tt.count == 128 * n
    tt.update(dev, 64)  # count >= until -> frozen
    assert tt.count == 128 * n
    tt.until = None
    tt.training = False
    tt.update(dev, 64)  # eval mode never learns
    assert tt.count == 128 * n


# ---- GaussianDistribution -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [1, 12, 29])
@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_gaussian_log_prob_entropy_sample(tt_ctx, dim, std_type):
    from rsl_rl_tt.modules import TTGaussianDistribution

    torch.manual_seed(2)
    ref = GaussianDistribution(dim, init_std=0.7, std_type=std_type, std_range=(0.05, 4.0))
    with torch.no_grad():
        (ref.std_param if std_type == "scalar" else ref.log_std_param).add_(torch.randn(dim) * (0.3 if std_type == "scalar" else 0.4))
    tt = TTGaussianDistribution(tt_ctx, ref)
    rows = 256
    mean = torch.randn(rows, dim)
    noise = torch.randn(rows, dim, generator=torch.Generator().manual_seed(4))
    mean_dev = tt_ctx.upload(pad_matrix(mean, rows, pad_up(dim)), op="H-1", batch_sharded=True)
    noise_dev = tt_ctx.upload(pad_matrix(noise, rows, pad_up(dim)), op="H-4", batch_sharded=True)
    ref.update(mean)
    # std (incl. clamping) and entropy
    r_std = _record(f"gauss.std dim={dim} {std_type}", compare(tt.host_std(), ref.std[0], TOLERANCES["eltwise_fp32"]))
    ent = tt_ctx.download(tt.entropy_dev, op="H-5", batch_sharded=False)[0, 0]
    r_ent = _record(f"gauss.entropy dim={dim} {std_type}", compare(ent.reshape(1), ref.entropy[0].reshape(1), TOLERANCES["row_sum_fp32"]))
    # sampling with injected noise == upstream's mean + std * noise
    act_dev = tt.sample(mean_dev, noise_dev)
    act = tt_ctx.download(act_dev, op="H-2")
    assert torch.count_nonzero(act[:, dim:]) == 0
    expect = mean + ref.std * noise
    r_act = _record(f"gauss.sample dim={dim} {std_type}", compare(act[:, :dim], expect, TOLERANCES["row_sum_fp32"]))  # mean + std*noise cancels: judge by norm, not per element
    # log_prob of those actions
    lp = tt_ctx.download(tt.log_prob(act_dev, mean_dev), op="H-2")[:, 0]
    r_lp = _record(f"gauss.log_prob dim={dim} {std_type}", compare(lp, ref.log_prob(expect), TOLERANCES["row_sum_fp32"]))
    for r in (r_std, r_ent, r_act, r_lp):
        assert r.passed, str(r)


def test_upstream_sample_equals_mean_plus_std_times_randn():
    """The parity-mode noise contract (BRIEF 3.5): upstream's Normal.sample() consumes exactly randn([rows, A]) of the global generator."""
    dist = GaussianDistribution(12, init_std=0.8)
    mean = torch.randn(128, 12)
    dist.update(mean)
    torch.manual_seed(1234)
    upstream = dist.sample()
    torch.manual_seed(1234)
    ours = mean + dist.std * torch.randn(128, 12)
    assert torch.equal(upstream, ours)


# ---- synchronisation and accounting ---------------------------------------------------------------------------------------------
def test_sync_round_trip_is_bit_exact_and_counts_h6(tt_ctx):
    from rsl_rl_tt.modules import TTMLPModel

    groups = {"policy": 36, "privileged": 48}
    model = make_model(groups, ["policy"], 12, [128, 64], norm=True)
    feed_normalizer(model, groups)
    before = copy.deepcopy(model.state_dict())
    tt = TTMLPModel(tt_ctx, model)
    for p in model.parameters():
        p.data.zero_()  # destroy the mirror: only the device copy knows the weights now
    model.obs_normalizer._mean.zero_()
    tt_ctx.host_ops.reset()
    tt.sync_to_torch()
    after = model.state_dict()
    for k, v in before.items():
        if k == "distribution.std_param" or k.endswith(("weight", "bias", "_mean", "_var", "_std", "count")):
            assert torch.equal(after[k], v), k
    assert tt_ctx.host_ops.calls["H-6"] > 0 and set(tt_ctx.host_ops.calls) == {"H-6"}


def test_host_op_accounting_for_one_model_upload(tt_ctx):
    from rsl_rl_tt.modules import TTMLPModel

    groups = {"policy": 36, "privileged": 48}
    tt = TTMLPModel(tt_ctx, make_model(groups, ["policy", "privileged"], 1, [64], dist=False))
    tt_ctx.host_ops.reset()
    tt.obs_to_device(make_obs(32 * tt_ctx.n_chips, groups))
    assert dict(tt_ctx.host_ops.calls) == {"H-3": 1, "H-1": 1}
    with pytest.raises(RuntimeError, match="not on the allow-list"):
        tt_ctx.host_ops.count("H-99")
    with pytest.raises(ValueError, match="not a multiple of"):
        tt.obs_to_device(make_obs(40, groups))


# ---- random shape sweep (padding paths) and checkpoint loading ----------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_random_shape_sweep(tt_ctx, seed):
    import random

    from rsl_rl_tt.modules import TTMLPModel

    g = random.Random(1000 + seed)
    groups = {f"g{i}": g.randint(1, 70) for i in range(g.randint(1, 3))}
    used = g.sample(list(groups), k=g.randint(1, len(groups)))
    hidden = [g.randint(1, 140) for _ in range(g.randint(1, 3))]
    out_dim, act, norm, rows = g.randint(1, 40), g.choice(["elu", "tanh", "relu"]), g.choice([True, False]), g.choice([32, 64, 288])
    rows *= tt_ctx.n_chips  # rows per chip (see test_model_forward_parity)
    model = make_model(groups, used, out_dim, hidden, act, norm, seed=seed)
    if norm:
        feed_normalizer(model, groups)
    tt = TTMLPModel(tt_ctx, model)
    obs = make_obs(rows, groups, seed=seed, scale=1.5, offset=0.3)
    with torch.no_grad():
        ref = model(obs)
    out = tt_ctx.download(tt.forward(tt.obs_to_device(obs)), op="H-2")
    assert torch.count_nonzero(out[:, out_dim:]) == 0, "padding leaked into the output"
    rep = _record(f"sweep{seed} groups={groups} used={used} hidden={hidden} act={act} norm={norm} out={out_dim} rows={rows}", compare(out[:, :out_dim], ref, TOLERANCES["mlp_forward"]))
    assert rep.passed, str(rep)


def test_loading_a_checkpoint_into_the_mirror_then_syncing_updates_the_device_copy(tt_ctx):
    from rsl_rl_tt.modules import TTMLPModel

    groups = {"policy": 36}
    a = make_model(groups, ["policy"], 12, [64, 64], norm=True, seed=0)
    b = make_model(groups, ["policy"], 12, [64, 64], norm=True, seed=1)
    feed_normalizer(b, groups, seed=50)
    with torch.no_grad():
        b.distribution.std_param.mul_(0.5)
    tt = TTMLPModel(tt_ctx, a)
    obs = make_obs(max(64, 32 * tt_ctx.n_chips), groups, seed=3)
    a.load_state_dict(b.state_dict())  # e.g. OnPolicyRunner.load()
    stale = tt_ctx.download(tt.forward(tt.obs_to_device(obs)), op="H-2")[:, :12]
    tt.sync_from_torch()
    fresh = tt_ctx.download(tt.forward(tt.obs_to_device(obs)), op="H-2")[:, :12]
    with torch.no_grad():
        ref = b(obs)
    assert not compare(stale, ref, TOLERANCES["mlp_forward"]).passed  # without the sync the device still holds the old weights
    assert compare(fresh, ref, TOLERANCES["mlp_forward"]).passed
    assert compare(tt.distribution.host_std(), b.distribution.std_param.detach(), TOLERANCES["eltwise_fp32"]).passed
    assert tt.normalizer.count == int(b.obs_normalizer.count)
