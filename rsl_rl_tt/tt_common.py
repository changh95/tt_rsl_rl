# SPDX-License-Identifier: BSD-3-Clause
"""Shared plumbing of the TT training path: tile padding, the compute-kernel config, and the ONLY host<->device transfer helpers.

Every upload/download of the TT path goes through `TTContext.upload` / `TTContext.download` (a test forbids `ttnn.from_torch` /
`ttnn.to_torch` anywhere else in `rsl_rl_tt/`), and every transfer names the allow-list entry of docs/porting/DECISIONS.md (H-1 ... H-8) it
belongs to. `HostOpCounter` counts them (calls and bytes); an unknown id raises, so a host fallback that is not on the list cannot happen
silently (BRIEF 2-8). This module is the only place of the TT path that may touch torch tensors for staging.

Layout conventions (BRIEF 3.3/3.4): all device tensors are 2D, float32, TILE layout, with BOTH dimensions padded to multiples of 32; weights
are stored as [in_p, out_p] (the transpose of torch's [out, in]); a batch is sharded over the mesh along dim 0 (replicated state such as
weights and Adam moments is replicated). Padding is exact zeros, so padded matmul inputs/outputs stay zero through bias-free-padding + ELU
(ELU(0) = 0); the places where padding would change a result (log-prob / entropy sums over action dims) use explicit masks.
"""

from __future__ import annotations

import torch
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

TILE = 32

# docs/porting/DECISIONS.md "Host-computation allow-list"; keep in sync (tests/tt/test_tt_invariants.py checks the ids against the document). H-7 (temporary host rollout
# storage, Phase 2) was retired in Phase 3 when storage, GAE and mini-batches moved to the device.
HOST_OPS = {
    "H-1": "upload obs / reward / done / time_out per env step",
    "H-2": "download actions per env step (the one per-step sync point)",
    "H-3": "obs-group concatenation (boundary data shaping)",
    "H-4": "parity mode: action noise / randperm from the host torch generator",
    "H-5": "per-iteration scalars for logging",
    "H-6": "weight / statistics download for get_policy(), save(), export",
    "H-8": "upload per-iteration scalars as device tensors (normalizer update rate, learning rate, Adam bias corrections)",
    "H-9": "download of the per-step intrinsic reward of RND (the runner logs it every step; logging only, never fed back)",
    "H-10": "construction-time probing of the user's symmetry function on the host (derives the affine mirror maps once; never per step)",
}


def pad_up(n: int, m: int = TILE) -> int:
    return -(-n // m) * m


class HostOpCounter:
    """Counts host<->device transfers per allow-list id. An id that is not on the list raises."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.bytes: Counter[str] = Counter()

    def count(self, op_id: str, nbytes: int = 0) -> None:
        if op_id not in HOST_OPS:
            raise RuntimeError(f"host operation {op_id!r} is not on the allow-list (DECISIONS.md H-1..H-6, H-8..H-10); add it there first or remove the fallback")
        self.calls[op_id] += 1
        self.bytes[op_id] += nbytes

    def reset(self) -> None:
        self.calls.clear()
        self.bytes.clear()

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {"calls": dict(self.calls), "bytes": dict(self.bytes)}

    def __repr__(self) -> str:
        return "HostOpCounter(" + ", ".join(f"{k}: {self.calls[k]} calls/{self.bytes[k]} B" for k in sorted(self.calls)) + ")"


@dataclass
class TTContext:
    """The open mesh plus everything the TT modules share. Create it with `TTContext.create(mesh)` (inside `open_tray_mesh`)."""

    ttnn: Any
    mesh: Any
    n_chips: int
    compute_cfg: Any
    host_ops: HostOpCounter = field(default_factory=HostOpCounter)
    mesh_shape: tuple[int, int] = (1, 1)  # (rows, cols) of the mesh: the cross-chip sums gather along each axis of size > 1

    @classmethod
    def create(cls, mesh: Any, math_fidelity: str = "HiFi4", mesh_shape: tuple[int, int] | None = None) -> TTContext:
        """`math_fidelity` ("HiFi4" = the fp32 policy of BRIEF 3.4 / TOLERANCES.md; "HiFi3", "HiFi2", "LoFi" only for the Phase 5 precision experiments).
        `mesh_shape` defaults to a single row of all chips."""
        import ttnn

        fidelity = getattr(ttnn.MathFidelity, math_fidelity)
        cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        n = mesh.get_num_devices()
        return cls(ttnn=ttnn, mesh=mesh, n_chips=n, compute_cfg=cfg, mesh_shape=tuple(mesh_shape) if mesh_shape else (1, n))

    # -- cross-chip arithmetic (ADR-0004 amendment A-1) --------------------------------------------------------------------------
    def sum_over_chips(self, t: Any, scale: float | None = None) -> Any:
        """Sum over all chips of a tensor whose logical shape is the same on every chip but whose VALUES differ (e.g. a locally computed gradient), times `scale`
        (1/N for a mean). `ttnn.all_reduce` is not fp32-accurate (rel 5.5e-4, T-4), so every chip gathers all shards along each mesh axis (a pure copy) and sums
        them in fp32 with `ttnn.sum` (5.1e-8, bit-identical on every chip): the replicas of parameters and optimizer state cannot drift apart."""
        ttnn = self.ttnn
        if self.n_chips == 1:
            return t if scale is None else ttnn.multiply(t, scale)
        shape = tuple(t.shape)
        x = ttnn.reshape(t, (1, *shape))  # [1, R, C]: all_gather along dim 0 stacks the chips' shards
        for axis, size in enumerate(self.mesh_shape):
            if size > 1:
                x = ttnn.all_gather(x, dim=0, cluster_axis=axis)
        s = ttnn.reshape(ttnn.sum(x, dim=0, keepdim=True, compute_kernel_config=self.compute_cfg), shape)
        return s if scale is None else ttnn.multiply(s, scale)

    def mean_over_chips(self, t: Any) -> Any:
        """Mean over the chips of a per-chip value (identity on one chip: no op is added to the single-chip path)."""
        return t if self.n_chips == 1 else self.sum_over_chips(t, 1.0 / self.n_chips)

    # -- checks ------------------------------------------------------------------------------------------------------------
    def check_batch(self, rows: int, what: str = "batch") -> int:
        """Rows of a batch-sharded tensor: every chip must hold a multiple of 32 rows (no padding, no short tile; BRIEF 3.3)."""
        if rows % (self.n_chips * TILE) != 0:
            raise ValueError(f"{what} of {rows} rows is not a multiple of {self.n_chips * TILE} (= {self.n_chips} chip(s) x tile {TILE}); choose num_envs / num_steps / num_mini_batches accordingly")
        return rows // self.n_chips

    # -- transfers (the only ones) -----------------------------------------------------------------------------------------
    def upload(self, host: torch.Tensor, *, op: str, batch_sharded: bool) -> Any:
        """2D float32 host tensor (both dims multiples of 32; or a replicated [1, cols] row; or a [rows, 1] column) -> device TILE tensor, batch-sharded over dim 0 or replicated."""
        ttnn = self.ttnn
        assert host.dim() == 2 and host.dtype == torch.float32, (host.shape, host.dtype)
        row_vector = host.shape[0] == 1 and not batch_sharded  # replicated [1, cols] parameter rows (bias, mean, std) keep their logical shape
        col_vector = host.shape[1] == 1 and host.shape[0] % TILE == 0  # [rows, 1] per-sample scalars (value, log-prob, advantage, ...) keep theirs
        assert (host.shape[0] % TILE == 0 or row_vector) and (host.shape[1] % TILE == 0 or col_vector), f"upload shape {tuple(host.shape)} is not tile aligned"
        self.host_ops.count(op, host.numel() * 4)
        mapper = ttnn.ShardTensorToMesh(self.mesh, dim=0) if batch_sharded else ttnn.ReplicateTensorToMesh(self.mesh)
        return ttnn.from_torch(host.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=mapper)

    def download(self, dev: Any, *, op: str, batch_sharded: bool = True) -> torch.Tensor:
        """Device tensor -> host float32: batch-sharded tensors are concatenated over dim 0, replicated ones return chip 0's copy."""
        ttnn = self.ttnn
        host = ttnn.to_torch(dev, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        self.host_ops.count(op, host.numel() * 4)
        if not batch_sharded and self.n_chips > 1:
            host = host[: host.shape[0] // self.n_chips]
        return host.to(torch.float32)

    # -- persistent buffers (trace-compatible state) ----------------------------------------------------------------------------
    def put(self, current: Any, new: Any) -> Any:
        """Device -> device: write `new` into the persistent tensor `current` IN PLACE (so its address, which traces capture, never changes).
        `current is None` adopts `new` as the persistent tensor."""
        if current is None:
            return new
        self.ttnn.copy(new, current)
        return current

    def upload_into(self, current: Any, host: torch.Tensor, *, op: str, batch_sharded: bool = False) -> Any:
        """Host -> device into the persistent tensor `current` in place (`copy_host_to_device_tensor`); a fresh upload when there is none yet."""
        if current is None:
            return self.upload(host, op=op, batch_sharded=batch_sharded)
        ttnn = self.ttnn
        self.host_ops.count(op, host.numel() * 4)
        mapper = ttnn.ShardTensorToMesh(self.mesh, dim=0) if batch_sharded else ttnn.ReplicateTensorToMesh(self.mesh)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper), current)
        return current

    def upload_rm_into(self, current: Any, host: torch.Tensor, *, op: str, batch_sharded: bool = False) -> Any:
        """Like `upload_into`, but the device tensor is ROW_MAJOR: no tilize on the host (a TILE host tensor costs ~5x more per byte, Phase 5 probe
        `scripts/probe_upload_device.py`); the consumer tilizes on the device (`ttnn.to_layout`, 0.2 ms for 65536 x 64 fp32), e.g. inside a trace."""
        ttnn = self.ttnn
        self.host_ops.count(op, host.numel() * 4)
        mapper = ttnn.ShardTensorToMesh(self.mesh, dim=0) if batch_sharded else ttnn.ReplicateTensorToMesh(self.mesh)
        if current is None:
            return ttnn.from_torch(host.contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, mesh_mapper=mapper)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host.contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper), current)
        return current

    def upload_scalar_into(self, current: Any, value: float, *, op: str) -> Any:
        """A replicated [1, 1] scalar written into the persistent tensor `current` (per-iteration scalars of a traced step)."""
        if current is None:
            return self.upload_scalar(value, op=op)
        ttnn = self.ttnn
        self.host_ops.count(op, 4)
        host = ttnn.from_torch(torch.full((1, 1), float(value)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        ttnn.copy_host_to_device_tensor(host, current)
        return current

    def upload_scalar(self, value: float, *, op: str) -> Any:
        """A replicated [1, 1] float32 device tensor (per-iteration scalars travel as device tensors so a trace cannot freeze them)."""
        ttnn = self.ttnn
        self.host_ops.count(op, 4)
        return ttnn.from_torch(torch.full((1, 1), float(value)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def upload_index(self, host_idx: torch.Tensor, *, op: str) -> Any:
        """A UINT32 ROW_MAJOR [1, N] index row for `ttnn.embedding` (the row gather of the shuffle); the permutation itself is drawn on the host (allow-list H-4).

        On several chips `host_idx` holds the chips' LOCAL permutations one after the other (chip c's slice = `host_idx[c * n_local:(c + 1) * n_local]`, indices
        into that chip's own rows) and is sharded along the index axis: every chip shuffles its own shard of the batch (upstream: an independent randperm per rank)."""
        ttnn = self.ttnn
        assert host_idx.dim() == 1 and host_idx.numel() % (self.n_chips * TILE) == 0, host_idx.shape
        self.host_ops.count(op, host_idx.numel() * 4)
        mapper = ttnn.ShardTensorToMesh(self.mesh, dim=1) if self.n_chips > 1 else ttnn.ReplicateTensorToMesh(self.mesh)
        return ttnn.from_torch(host_idx.to(torch.int64).reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, mesh_mapper=mapper)

    def per_chip(self, dev: Any, *, op: str = "H-6") -> torch.Tensor:
        """Download EVERY chip's copy of a (replicated or sharded) tensor as [n_chips, ...] float32 (tests and the replica consistency check, H-6)."""
        host = self.ttnn.to_torch(dev, mesh_composer=self.ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        self.host_ops.count(op, host.numel() * 4)
        return host.to(torch.float32).reshape(self.n_chips, host.shape[0] // self.n_chips, *host.shape[1:])

    def zeros(self, rows: int, cols: int, *, batch_sharded: bool, op: str = "H-6") -> Any:
        return self.upload(torch.zeros(rows, cols), op=op, batch_sharded=batch_sharded)


# -- host-side padding helpers (staging; the device never sees unpadded data) ----------------------------------------------------
def pad_matrix(m: torch.Tensor, rows: int, cols: int, fill: float = 0.0) -> torch.Tensor:
    """Zero- (or `fill`-) pad a 2D tensor to [rows, cols]."""
    out = torch.full((rows, cols), fill, dtype=torch.float32)
    out[: m.shape[0], : m.shape[1]] = m.detach().to(torch.float32)
    return out


def pad_vector(v: torch.Tensor, cols: int, fill: float = 0.0) -> torch.Tensor:
    """1D parameter vector -> a [1, cols] row (logical shape 1 x cols; TILE layout pads the row dimension internally)."""
    out = torch.full((1, cols), fill, dtype=torch.float32)
    out[0, : v.numel()] = v.detach().to(torch.float32).flatten()
    return out


def pad_heads(m: torch.Tensor, rows_p: int, cols_p: int, heads: int = 1) -> torch.Tensor:
    """[R, heads * D] -> [rows_p, cols_p] where every one of the `heads` blocks of D columns is zero-padded on its own to cols_p / heads columns
    (heads == 1 is `pad_matrix`). The two-head output layer of a heteroscedastic actor (mean | std) uses this layout: both halves start on a tile boundary."""
    d, d_p = m.shape[1] // heads, cols_p // heads
    out = torch.zeros(rows_p, cols_p)
    for h in range(heads):
        out[: m.shape[0], h * d_p : h * d_p + d] = m[:, h * d : (h + 1) * d].detach().to(torch.float32)
    return out


def unpad_heads(m: torch.Tensor, rows: int, cols: int, heads: int = 1) -> torch.Tensor:
    """Inverse of `pad_heads`: [rows_p, cols_p] -> [rows, cols] (cols = heads * D)."""
    d, d_p = cols // heads, m.shape[1] // heads
    return torch.cat([m[:rows, h * d_p : h * d_p + d] for h in range(heads)], dim=1)


def linear_to_host(weight: torch.Tensor, bias: torch.Tensor | None, in_p: int, out_p: int, heads: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """torch Linear (weight [out, in], bias [out]) -> padded device layout: W^T as [in_p, out_p] and the bias as a [1, out_p] row (`heads` output blocks, see `pad_heads`)."""
    w = pad_heads(weight.detach().t(), in_p, out_p, heads)
    b = pad_heads((bias if bias is not None else torch.zeros(weight.shape[0])).detach().reshape(1, -1), 1, out_p, heads)
    return w, b


def linear_from_host(w: torch.Tensor, b: torch.Tensor, out_features: int, in_features: int, heads: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `linear_to_host`: padded [in_p, out_p] / [1, out_p] -> torch layout [out, in] / [out]."""
    return unpad_heads(w, in_features, out_features, heads).t().contiguous(), unpad_heads(b, 1, out_features, heads)[0].contiguous()


def mask_row(dim: int, dim_p: int) -> torch.Tensor:
    """[1, dim_p] row with 1 for the real columns and 0 for the padding."""
    m = torch.zeros(1, dim_p)
    m[0, :dim] = 1.0
    return m


def stage_obs(obs: Any, groups: list[str], dim_p: int) -> torch.Tensor:
    """H-3: concatenate the selected observation groups into one zero-padded [rows, dim_p] float32 staging tensor."""
    rows = obs[groups[0]].shape[0]
    buf = torch.zeros(rows, dim_p)
    col = 0
    for group in groups:
        part = obs[group]
        buf[:, col : col + part.shape[1]] = part
        col += part.shape[1]
    return buf


def unit_row(col: int, width: int = TILE) -> torch.Tensor:
    """[1, width] row with a 1 in column `col` (selects / places a scalar column when per-sample scalars are assembled into a block)."""
    row = torch.zeros(1, width)
    row[0, col] = 1.0
    return row


def pack_step_scalars(reward: torch.Tensor, done: torch.Tensor, time_out: torch.Tensor | None) -> torch.Tensor:
    """H-1 staging of one env step: [num_envs, 32] with column 0 = reward, 1 = done, 2 = time_out (a single upload per step)."""
    n = reward.shape[0]
    buf = torch.zeros(n, TILE)
    buf[:, 0] = reward.to(torch.float32).reshape(n)
    buf[:, 1] = done.to(torch.float32).reshape(n)
    if time_out is not None:
        buf[:, 2] = time_out.to(torch.float32).reshape(n)
    return buf


class SectionTimer:
    """Optional wall-clock profile of named sections (host time with a device synchronisation at each boundary). Off by default: `section()` is a no-op."""

    def __init__(self, ctx: "TTContext | None" = None, enabled: bool = False) -> None:
        self.ctx, self.enabled = ctx, enabled
        self.paused = False  # True while a trace is being captured: a device synchronisation is a TT_FATAL inside a capture
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def section(self, name: str):
        timer = self

        class _Section:
            def __enter__(self_inner):
                self_inner.on = timer.enabled and not timer.paused
                if self_inner.on:
                    timer.ctx.ttnn.synchronize_device(timer.ctx.mesh)
                    import time

                    self_inner.t0 = time.perf_counter()
                return self_inner

            def __exit__(self_inner, *exc):
                if self_inner.on:
                    import time

                    timer.ctx.ttnn.synchronize_device(timer.ctx.mesh)
                    dt = time.perf_counter() - self_inner.t0
                    timer.totals[name] = timer.totals.get(name, 0.0) + dt
                    timer.counts[name] = timer.counts.get(name, 0) + 1

        return _Section()

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()

    def report(self) -> dict[str, dict[str, float]]:
        return {k: {"total_ms": v * 1e3, "calls": self.counts[k], "ms_per_call": v * 1e3 / self.counts[k]} for k, v in sorted(self.totals.items(), key=lambda kv: -kv[1])}
