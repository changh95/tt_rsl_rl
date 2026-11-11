# SPDX-License-Identifier: BSD-3-Clause
"""CNN encoders (upstream `rsl_rl.modules.CNN` inside `CNNModel`) on the device with a hand-written forward and backward (post-v1 item 8.6, docs/porting/POST_V1.md).

Supported (everything else is refused at construction with a reason): `nn.Conv2d` with dilation 1 and zero padding ("none" / "zeros"), no normalisation layers, the activations of the MLP (elu, relu, tanh),
`MaxPool2d(3, 2, 1)` after a layer, `AdaptiveAvgPool2d(1)` as the global pool, flatten.

Layout. An activation of B images of H x W pixels with C channels is a 2D tile tensor [B H W, C_p] (row = (image, y, x), channels padded to a tile multiple with zeros). A convolution is an "im2col" matmul:
the K x K shifted copies of the input (`patches`, one gather per kernel offset) are concatenated along the columns, [B Ho Wo, K^2 C_in_p], and multiplied with the weight matrix [K^2 C_in_p, C_out_p] (row
k C_in_p + c holds `weight[:, c, ky, kx]` with k = ky K + kx). A gather of rows with a CONSTANT index tensor is exact (the fp32 rows are split into three bf16 pieces, gathered with `ttnn.embedding` and added back: the
shuffle of `TTRolloutStorage`); the padding is an extra row (0 for a convolution, -3e38 for a max pool). The adjoint of the gather of offset k is again a gather (an input pixel is read by at most ONE output
pixel per offset: y s + ky - p = y' has at most one solution), so the gradient with respect to the input needs no scatter either. Max pool: the gradient goes to the FIRST maximum of a window in scan order
(torch's rule); the matmuls are fp32 HiFi4 like everywhere else.

The latent of an encoder is its last activation reshaped to [B, Ho Wo C_p] (position-major, channels padded): the head's first layer is given a row map that places torch's (c, y, x)-ordered flatten rows there.
"""

from __future__ import annotations

from typing import Any

import math
import torch

from rsl_rl_tt.tt_common import TTContext, pad_up

PAD_MAX = -3.0e38  # the padding value of a max pool: finite (the bf16 split stays exact), below every activation


def _out_dim(size: int, k: int, s: int, p: int) -> int:
    return (size + 2 * p - (k - 1) - 1) // s + 1


class _Geometry:
    """Constant gather indices of one window geometry (kernel k, stride s, padding p) over B images of H x W pixels: forward `idx[k]` ([B Ho Wo], the input row read by every output row for kernel offset
    k, B H W = the padding row) and the adjoint `adj[k]` ([B H W], the output row that reads every input row, B Ho Wo = the zero row)."""

    def __init__(self, ctx: TTContext, batch: int, h: int, w: int, k: int, s: int, ph: int, pw: int) -> None:
        self.ctx = ctx
        self.ho, self.wo = _out_dim(h, k, s, ph), _out_dim(w, k, s, pw)
        n_in, n_out = batch * h * w, batch * self.ho * self.wo
        self.n_in, self.n_out, self.k = n_in, n_out, k
        b = torch.arange(batch).reshape(-1, 1, 1)
        yo = torch.arange(self.ho).reshape(1, -1, 1)
        xo = torch.arange(self.wo).reshape(1, 1, -1)
        yi = torch.arange(h).reshape(1, -1, 1)
        xi = torch.arange(w).reshape(1, 1, -1)
        fwd, adj = [], []
        for ky in range(k):
            for kx in range(k):
                iy, ix = yo * s + ky - ph, xo * s + kx - pw
                inside = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
                idx = torch.where(inside, b * (h * w) + iy * w + ix, torch.full((batch, self.ho, self.wo), n_in))
                fwd.append(idx.reshape(-1))
                ny, nx = yi - ky + ph, xi - kx + pw  # the output pixel (y, x) that reads input pixel (yi, xi) at this offset
                ok = (ny % s == 0) & (nx % s == 0) & (ny >= 0) & (nx >= 0) & (ny // s < self.ho) & (nx // s < self.wo)
                row = torch.where(ok, b * (self.ho * self.wo) + (ny // s) * self.wo + (nx // s), torch.full((batch, h, w), n_out))
                adj.append(row.reshape(-1))
        n = ctx.n_chips
        up = lambda t: ctx.upload_index(torch.cat([t] * n), op="H-6")  # every chip gathers within its own images: the same local indices on every chip
        self.idx = [up(t) for t in fwd]
        self.adj = [up(t) for t in adj]


class TTCNN:
    """One upstream `CNN` (a `torch.nn.Sequential` of Conv2d / activation / MaxPool2d / pools / Flatten) on the device. `mirror` = the torch module; `batch` = images per chip of the batches it will see (a
    separate geometry is built per batch size in `batches`)."""

    def __init__(self, ctx: TTContext, mirror: Any, input_hw: tuple[int, int], in_channels: int, batches: list[int]) -> None:
        self.ctx, self.mirror = ctx, mirror
        self.in_channels, self.in_hw = in_channels, tuple(input_hw)
        self.ops: list[dict] = []  # conv / act / maxpool / avgpool, in order
        h, w, c = input_hw[0], input_hw[1], in_channels
        self.convs: list[dict] = []
        mods = list(mirror)
        for i, m in enumerate(mods):
            name = type(m).__name__
            if name == "Conv2d":
                if m.dilation != (1, 1) or m.groups != 1 or m.kernel_size[0] != m.kernel_size[1] or m.stride[0] != m.stride[1]:
                    raise NotImplementedError("Conv2d: only square kernels / strides, dilation 1, groups 1")
                if m.padding_mode != "zeros":
                    raise NotImplementedError(f"Conv2d padding_mode {m.padding_mode!r}: only zero padding")
                op = {"op": "conv", "mod": m, "k": m.kernel_size[0], "s": m.stride[0], "p": tuple(m.padding), "cin": m.in_channels, "cout": m.out_channels, "hw_in": (h, w), "index": len(self.convs)}
                h, w, c = _out_dim(h, op["k"], op["s"], op["p"][0]), _out_dim(w, op["k"], op["s"], op["p"][1]), m.out_channels
                op["hw_out"] = (h, w)
                self.convs.append(op)
                self.ops.append(op)
            elif name in ("ELU", "ReLU", "Tanh", "Identity"):
                if name == "ELU" and abs(m.alpha - 1.0) > 0:
                    raise NotImplementedError("ELU with alpha != 1")
                self.ops.append({"op": "act", "act": {"ELU": "elu", "ReLU": "relu", "Tanh": "tanh", "Identity": None}[name]})
            elif name == "MaxPool2d":
                if (m.kernel_size, m.stride, m.padding) != (3, 2, 1) or m.dilation != 1 or m.ceil_mode:
                    raise NotImplementedError("MaxPool2d: only the (3, 2, 1) pooling of the CNN module")
                op = {"op": "maxpool", "k": 3, "s": 2, "p": (1, 1), "hw_in": (h, w)}
                h, w = _out_dim(h, 3, 2, 1), _out_dim(w, 3, 2, 1)
                op["hw_out"] = (h, w)
                self.ops.append(op)
            elif name == "AdaptiveAvgPool2d":
                self.ops.append({"op": "avgpool", "hw_in": (h, w)})
                h, w = 1, 1
            elif name == "Flatten":
                pass  # the latent layout below IS the flatten
            elif name in ("BatchNorm2d", "LayerNorm", "AdaptiveMaxPool2d"):
                raise NotImplementedError(f"{name} in a CNN encoder is not supported by the TT port (normalisation layers, global max pooling)")
            else:
                raise NotImplementedError(f"CNN layer {name} is outside the supported set")
        self.out_hw, self.out_channels = (h, w), c
        self.out_c_p = pad_up(c)
        self.latent_p = h * w * self.out_c_p  # columns of the latent [B, h w c_p]
        self.in_c_p = pad_up(in_channels)
        self.geom: dict[tuple, _Geometry] = {}
        self.batches = list(batches)
        for b in self.batches:
            for op in self.ops:
                if op["op"] in ("conv", "maxpool"):
                    key = (b, op["hw_in"], op["k"], op["s"], op["p"])
                    if key not in self.geom:
                        self.geom[key] = _Geometry(ctx, b, *op["hw_in"], op["k"], op["s"], *op["p"])
        self._pad_rows: dict[tuple, Any] = {}
        self.w: list[Any] = [None] * len(self.convs)
        self.b: list[Any] = [None] * len(self.convs)
        self.sync_from_torch()

    # -- parameters ---------------------------------------------------------------------------------------------------------------------------
    def _cin_p(self, op: dict) -> int:
        return pad_up(op["cin"])

    def weight_to_host(self, layer: int, kind: str, t: Any) -> Any:
        """torch layout -> padded device layout (`kind` "w": [cout, cin, k, k] -> [k^2 cin_p, cout_p]; "b": [cout] -> [1, cout_p]); also used for the Adam moments of a checkpoint."""
        op = self.convs[layer]
        cin_p, cout_p = self._cin_p(op), pad_up(op["cout"])
        if kind == "b":
            out = torch.zeros(1, cout_p)
            out[0, : op["cout"]] = t.detach().to(torch.float32)
            return out
        k = op["k"]
        out = torch.zeros(k * k * cin_p, cout_p)
        for ky in range(k):
            for kx in range(k):
                base = (ky * k + kx) * cin_p
                out[base : base + op["cin"], : op["cout"]] = t.detach().to(torch.float32)[:, :, ky, kx].t()
        return out

    def weight_from_host(self, layer: int, kind: str, t: Any) -> Any:
        op = self.convs[layer]
        cin_p = self._cin_p(op)
        if kind == "b":
            return t[0, : op["cout"]].contiguous()
        k = op["k"]
        out = torch.zeros(op["cout"], op["cin"], k, k)
        for ky in range(k):
            for kx in range(k):
                base = (ky * k + kx) * cin_p
                out[:, :, ky, kx] = t[base : base + op["cin"], : op["cout"]].t()
        return out

    def sync_from_torch(self) -> None:
        ctx = self.ctx
        for i, op in enumerate(self.convs):
            self.w[i] = ctx.upload_into(self.w[i], self.weight_to_host(i, "w", op["mod"].weight), op="H-6")
            self.b[i] = ctx.upload_into(self.b[i], self.weight_to_host(i, "b", op["mod"].bias), op="H-6")

    def sync_to_torch(self) -> None:
        ctx = self.ctx
        for i, op in enumerate(self.convs):
            op["mod"].weight.data.copy_(self.weight_from_host(i, "w", ctx.download(self.w[i], op="H-6", batch_sharded=False)))
            op["mod"].bias.data.copy_(self.weight_from_host(i, "b", ctx.download(self.b[i], op="H-6", batch_sharded=False)))

    def parameters(self) -> list:
        return [t for i in range(len(self.convs)) for t in (self.w[i], self.b[i])]

    def set_parameters(self, values: list) -> None:
        ctx = self.ctx
        for i in range(len(self.convs)):
            self.w[i], self.b[i] = ctx.put(self.w[i], values[2 * i]), ctx.put(self.b[i], values[2 * i + 1])

    def param_slots(self) -> list[tuple]:
        """(torch parameter, index in `parameters()`, layer, kind) in `parameters()` order (the torch order of a CNN: weight, bias per conv)."""
        out = []
        for i, op in enumerate(self.convs):
            out += [(op["mod"].weight, 2 * i, i, "w"), (op["mod"].bias, 2 * i + 1, i, "b")]
        return out

    # -- latent <-> torch flatten ------------------------------------------------------------------------------------------------------------
    def latent_rows(self) -> list[int]:
        """For every feature of torch's flatten of the output ([C, H, W] order) the column of the device latent ([H W C_p] order)."""
        h, w = self.out_hw
        c = self.out_channels
        rows = []
        for ch in range(c):
            for y in range(h):
                for x in range(w):
                    rows.append((y * w + x) * self.out_c_p + ch)
        return rows

    # -- gathers -----------------------------------------------------------------------------------------------------------------------------------
    def _split(self, x: Any, pad_value: float) -> Any:
        """[R, C_p] fp32 -> the three bf16 pieces of every row (exact) as a ROW_MAJOR [R + 1, 3 C_p] table whose last row is the padding row."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        c = x.shape[1]
        h1 = ttnn.typecast(x, ttnn.bfloat16)
        r1 = ttnn.subtract(x, ttnn.typecast(h1, ttnn.float32))
        h2 = ttnn.typecast(r1, ttnn.bfloat16)
        h3 = ttnn.typecast(ttnn.subtract(r1, ttnn.typecast(h2, ttnn.float32)), ttnn.bfloat16)
        table = ttnn.to_layout(ttnn.concat([h1, h2, h3], dim=1), ttnn.ROW_MAJOR_LAYOUT)
        key = (c, pad_value)
        if key not in self._pad_rows:
            pad = ctx.upload(torch.full((1, c), pad_value), op="H-6", batch_sharded=False)
            p1 = ttnn.typecast(pad, ttnn.bfloat16)
            q1 = ttnn.subtract(pad, ttnn.typecast(p1, ttnn.float32))
            p2 = ttnn.typecast(q1, ttnn.bfloat16)
            p3 = ttnn.typecast(ttnn.subtract(q1, ttnn.typecast(p2, ttnn.float32)), ttnn.bfloat16)
            self._pad_rows[key] = ttnn.to_layout(ttnn.concat([p1, p2, p3], dim=1), ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.concat([table, self._pad_rows[key]], dim=0)

    def _gather(self, table: Any, idx: Any, n_out: int, c: int) -> Any:
        ttnn = self.ctx.ttnn
        g = ttnn.to_layout(ttnn.reshape(ttnn.embedding(idx, table, layout=ttnn.ROW_MAJOR_LAYOUT), (n_out, 3 * c)), ttnn.TILE_LAYOUT)
        parts = [ttnn.typecast(g[:, k * c : (k + 1) * c], ttnn.float32) for k in range(3)]
        return ttnn.add(ttnn.add(parts[0], parts[1]), parts[2])

    def _patches(self, geom: _Geometry, x: Any, pad_value: float) -> list:
        c = x.shape[1]
        table = self._split(x, pad_value)
        return [self._gather(table, geom.idx[k], geom.n_out, c) for k in range(geom.k * geom.k)]

    def _unpatch(self, geom: _Geometry, parts: list, c: int) -> Any:
        """The adjoint of `_patches`: sum over the kernel offsets of the gradient pieces gathered back to the input rows."""
        ttnn = self.ctx.ttnn
        total = None
        for k, d in enumerate(parts):
            table = self._split(d, 0.0)
            g = self._gather(table, geom.adj[k], geom.n_in, c)
            total = g if total is None else ttnn.add(total, g)
        return total

    # -- forward ------------------------------------------------------------------------------------------------------------------------------------
    def to_rows(self, image: Any, batch: int) -> Any:
        """Flat observations [B, C H W] (torch's (c, y, x) order, columns padded) -> [B H W, C_in_p]."""
        ttnn = self.ctx.ttnn
        h, w = self.in_hw
        c = self.in_channels
        x = ttnn.reshape(image[:, : c * h * w], (batch, c, h * w))
        x = ttnn.permute(x, (0, 2, 1))  # [B, H W, C]
        x = ttnn.reshape(x, (batch * h * w, c))
        if self.in_c_p != c:
            x = ttnn.pad(x, ((0, 0), (0, self.in_c_p - c)), 0.0)
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def forward(self, image: Any, batch: int, train: bool) -> tuple[Any, Any]:
        """The latent [B, H W C_p] of `batch` images ([B, flat image columns]); with `train` also the cache for `backward`."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        cfg = ctx.compute_cfg
        x = self.to_rows(image, batch)
        cache: list = []
        for op in self.ops:
            if op["op"] == "conv":
                geom = self.geom[(batch, op["hw_in"], op["k"], op["s"], op["p"])]
                patches = ttnn.concat(self._patches(geom, x, 0.0), dim=1) if op["k"] > 1 else self._patches(geom, x, 0.0)[0]
                z = ttnn.linear(patches, self.w[op["index"]], bias=self.b[op["index"]], compute_kernel_config=cfg)
                cache.append({"op": op, "geom": geom, "patches": patches})
                x = z
            elif op["op"] == "act":
                prev = x
                if op["act"] == "elu":
                    x = ttnn.elu(prev, alpha=1.0)
                elif op["act"] == "relu":
                    x = ttnn.relu(prev)
                elif op["act"] == "tanh":
                    x = ttnn.tanh(prev)
                cache.append({"op": op, "z": prev, "a": x})
            elif op["op"] == "maxpool":
                geom = self.geom[(batch, op["hw_in"], op["k"], op["s"], op["p"])]
                parts = self._patches(geom, x, PAD_MAX)
                out = parts[0]
                for p in parts[1:]:
                    out = ttnn.maximum(out, p)
                cache.append({"op": op, "geom": geom, "parts": parts, "out": out})
                x = out
            else:  # global average pool over the spatial positions of every image
                hw = op["hw_in"][0] * op["hw_in"][1]
                x = ttnn.mean(ttnn.reshape(x, (batch, hw, x.shape[1])), dim=1, keepdim=False, compute_kernel_config=cfg)
                cache.append({"op": op, "hw": hw})
        h, w = self.out_hw
        latent = ttnn.reshape(x, (batch, h * w * self.out_c_p))
        return latent, (cache if train else None)

    # -- backward -----------------------------------------------------------------------------------------------------------------------------------
    def backward(self, cache: list, d_latent: Any, batch: int) -> list:
        """Gradients of the parameters in `parameters()` order for d loss / d latent ([B, H W C_p])."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        cfg = ctx.compute_cfg
        h, w = self.out_hw
        g = ttnn.reshape(d_latent, (batch * h * w, self.out_c_p))
        grads: list = [None] * (2 * len(self.convs))
        first_conv = self.convs[0]
        for entry in reversed(cache):
            op = entry["op"]
            if op["op"] == "act":
                z, a = entry["z"], entry["a"]
                if op["act"] == "elu":
                    g = ttnn.multiply(g, ttnn.exp(ttnn.minimum(z, 0.0)))
                elif op["act"] == "tanh":
                    g = ttnn.multiply(g, ttnn.subtract(1.0, ttnn.square(a)))
                elif op["act"] == "relu":
                    g = ttnn.multiply(g, ttnn.gt(z, 0.0))
            elif op["op"] == "maxpool":
                geom, parts, out = entry["geom"], entry["parts"], entry["out"]
                taken, dparts = None, []
                for p in parts:  # torch: the gradient goes to the FIRST maximum of the window in scan order
                    is_max = ttnn.eq(p, out)
                    sel = is_max if taken is None else ttnn.multiply(is_max, ttnn.subtract(1.0, taken))
                    taken = sel if taken is None else ttnn.add(taken, sel)
                    dparts.append(ttnn.multiply(g, sel))
                g = self._unpatch(geom, dparts, g.shape[1])
            elif op["op"] == "avgpool":
                hw = entry["hw"]
                g = ttnn.reshape(ttnn.multiply(ttnn.repeat(ttnn.reshape(g, (batch, 1, g.shape[1])), (1, hw, 1)), 1.0 / hw), (batch * hw, g.shape[1]))
            else:  # conv
                i, geom, patches = op["index"], entry["geom"], entry["patches"]
                grads[2 * i] = ttnn.matmul(patches, g, transpose_a=True, compute_kernel_config=cfg)
                grads[2 * i + 1] = ttnn.sum(g, dim=0, keepdim=True, compute_kernel_config=cfg)
                if i > 0 or op is not first_conv:  # the input of the first layer is the image: no gradient needed
                    dp = ttnn.matmul(g, self.w[i], transpose_b=True, compute_kernel_config=cfg)
                    cin_p = self._cin_p(op)
                    parts = [dp[:, k * cin_p : (k + 1) * cin_p] for k in range(op["k"] * op["k"])]
                    g = self._unpatch(geom, parts, cin_p)
        return grads


class TTCNNStack:
    """The encoders of a `CNNModel` (one `TTCNN` per 2D observation group) behind the interface of `TTRNN` for the parameters: `parameters()` / `set_parameters()` / `param_slots()` /
    `weight_to_host(layer, kind, t)` / `weight_from_host(...)`, where a layer is the pair (group index, convolution index)."""

    def __init__(self, ctx: TTContext, cnns: list[TTCNN]) -> None:
        self.ctx, self.stack = ctx, cnns
        self.offsets = []
        n = 0
        for c in cnns:
            self.offsets.append(n)
            n += len(c.parameters())

    def parameters(self) -> list:
        return [t for c in self.stack for t in c.parameters()]

    def set_parameters(self, values: list) -> None:
        for c, off in zip(self.stack, self.offsets):
            c.set_parameters(values[off : off + len(c.parameters())])

    def param_slots(self) -> list[tuple]:
        out = []
        for g, (c, off) in enumerate(zip(self.stack, self.offsets)):
            out += [(p, off + idx, (g, layer), kind) for p, idx, layer, kind in c.param_slots()]
        return out

    def weight_to_host(self, layer: tuple, kind: str, t: Any) -> Any:
        return self.stack[layer[0]].weight_to_host(layer[1], kind, t)

    def weight_from_host(self, layer: tuple, kind: str, t: Any) -> Any:
        return self.stack[layer[0]].weight_from_host(layer[1], kind, t)

    def sync_from_torch(self) -> None:
        for c in self.stack:
            c.sync_from_torch()

    def sync_to_torch(self) -> None:
        for c in self.stack:
            c.sync_to_torch()
