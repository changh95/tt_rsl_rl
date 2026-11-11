# SPDX-License-Identifier: BSD-3-Clause
"""GRU / LSTM (upstream `rsl_rl.modules.RNN` over `torch.nn.GRU` / `torch.nn.LSTM`) on the device with a hand-written forward and back-propagation through time (post-v1 item 8.5, docs/porting/POST_V1.md).

Layout. Every gate block is padded to a tile multiple on its own (`pad_heads`): the weights of a layer are `w_ih^T` [in_p, G H_p], `w_hh^T` [H_p, G H_p] and the biases [1, G H_p] with
G = 3 (GRU: r, z, n) or 4 (LSTM: i, f, g, o) in torch's gate order, so every gate is a tile-aligned column slice. The hidden state of layer l is [N, H_p] (LSTM: also the cell state), batch-sharded. Padded
columns stay exactly zero in the state (zero initial state, zero padded weights and biases: r = z = 0.5, n = 0 keep a zero hidden at zero; the LSTM's g = 0, c = 0) and receive zero gradients
(the padded rows of the next layer's / the head's weights are zero), so nothing needs masking.

Equations (torch's): GRU  r = s(W_ir x + b_ir + W_hr h + b_hr), z = s(...), n = tanh(W_in x + b_in + r * (W_hn h + b_hn)), h' = (1 - z) n + z h.
LSTM i, f, o = s(...), g = tanh(...), c' = f c + i g, h' = o tanh(c').  A layer's input is the previous layer's h'.

The update processes a mini-batch as one sequence per environment (all T steps of an env slice), resetting the state to zero after every step on which the environment was done: that is exactly upstream's
split-at-dones-and-pad formulation (each trajectory starts from the saved state, i.e. zero after a done and the carried state at the start of the rollout; the padded steps are masked), whose valid samples are
all T x E steps of the env slice. `forward_sequence` records what the backward pass needs, `backward_sequence` is the hand-written BPTT.
"""

from __future__ import annotations

from typing import Any

import torch

from rsl_rl_tt.tt_common import TTContext, pad_heads, pad_up, unpad_heads


class TTRNN:
    """A stack of GRU / LSTM layers (`mirror` = the upstream `rsl_rl.modules.RNN` wrapper; its `.rnn` is the `torch.nn.GRU` / `torch.nn.LSTM`)."""

    def __init__(self, ctx: TTContext, mirror: Any) -> None:
        self.ctx, self.mirror = ctx, mirror
        torch_rnn = mirror.rnn
        name = type(torch_rnn).__name__
        if name not in ("GRU", "LSTM"):
            raise NotImplementedError(f"recurrent module {name} is outside the scope (GRU, LSTM)")
        if torch_rnn.bidirectional or torch_rnn.dropout or not torch_rnn.bias or torch_rnn.batch_first or getattr(torch_rnn, "proj_size", 0):
            raise NotImplementedError("only the plain unidirectional GRU / LSTM with biases, no dropout and no projection is supported")
        self.lstm = name == "LSTM"
        self.gates = 4 if self.lstm else 3
        self.num_layers = torch_rnn.num_layers
        self.hidden = torch_rnn.hidden_size
        self.hidden_p = pad_up(self.hidden)
        self.input_size = torch_rnn.input_size
        self.in_p = [pad_up(self.input_size)] + [self.hidden_p] * (self.num_layers - 1)
        self.w_ih: list[Any] = [None] * self.num_layers
        self.w_hh: list[Any] = [None] * self.num_layers
        self.b_ih: list[Any] = [None] * self.num_layers
        self.b_hh: list[Any] = [None] * self.num_layers
        self.sync_from_torch()

    # -- parameters ------------------------------------------------------------------------------------------------------------
    def _host(self, layer: int) -> tuple:
        r = self.mirror.rnn
        g, h = self.gates, self.hidden
        in_f = self.input_size if layer == 0 else h
        w_ih = pad_heads(getattr(r, f"weight_ih_l{layer}").detach().t(), self.in_p[layer], g * self.hidden_p, g)
        w_hh = pad_heads(getattr(r, f"weight_hh_l{layer}").detach().t(), self.hidden_p, g * self.hidden_p, g)
        b_ih = pad_heads(getattr(r, f"bias_ih_l{layer}").detach().reshape(1, -1), 1, g * self.hidden_p, g)
        b_hh = pad_heads(getattr(r, f"bias_hh_l{layer}").detach().reshape(1, -1), 1, g * self.hidden_p, g)
        assert in_f > 0
        return w_ih, w_hh, b_ih, b_hh

    def weight_to_host(self, layer: int, kind: str, t: Any) -> Any:
        """torch layout -> padded device layout of one parameter (`kind` in w_ih, w_hh, b_ih, b_hh): also used for the Adam moments of a checkpoint."""
        g = self.gates
        if kind == "w_ih":
            return pad_heads(t.t(), self.in_p[layer], g * self.hidden_p, g)
        if kind == "w_hh":
            return pad_heads(t.t(), self.hidden_p, g * self.hidden_p, g)
        return pad_heads(t.reshape(1, -1), 1, g * self.hidden_p, g)

    def weight_from_host(self, layer: int, kind: str, t: Any) -> Any:
        g, h = self.gates, self.hidden
        in_f = self.input_size if layer == 0 else h
        if kind == "w_ih":
            return unpad_heads(t, in_f, g * h, g).t().contiguous()
        if kind == "w_hh":
            return unpad_heads(t, h, g * h, g).t().contiguous()
        return unpad_heads(t, 1, g * h, g)[0].contiguous()

    def sync_from_torch(self) -> None:
        ctx = self.ctx
        for layer in range(self.num_layers):
            for name, host in zip(("w_ih", "w_hh", "b_ih", "b_hh"), self._host(layer)):
                getattr(self, name)[layer] = ctx.upload_into(getattr(self, name)[layer], host, op="H-6")

    def sync_to_torch(self) -> None:
        ctx, r = self.ctx, self.mirror.rnn
        names = {"w_ih": "weight_ih", "w_hh": "weight_hh", "b_ih": "bias_ih", "b_hh": "bias_hh"}
        for layer in range(self.num_layers):
            for kind, torch_name in names.items():
                t = ctx.download(getattr(self, kind)[layer], op="H-6", batch_sharded=False)
                getattr(r, f"{torch_name}_l{layer}").data.copy_(self.weight_from_host(layer, kind, t))

    def parameters(self) -> list:
        """Device parameters in torch's registration order: per layer w_ih, w_hh, b_ih, b_hh."""
        return [t for layer in range(self.num_layers) for t in (self.w_ih[layer], self.w_hh[layer], self.b_ih[layer], self.b_hh[layer])]

    def set_parameters(self, values: list) -> None:
        ctx = self.ctx
        for layer in range(self.num_layers):
            base = 4 * layer
            self.w_ih[layer], self.w_hh[layer] = ctx.put(self.w_ih[layer], values[base]), ctx.put(self.w_hh[layer], values[base + 1])
            self.b_ih[layer], self.b_hh[layer] = ctx.put(self.b_ih[layer], values[base + 2]), ctx.put(self.b_hh[layer], values[base + 3])

    def param_slots(self) -> list[tuple]:
        """(torch parameter, index in `parameters()`, layer, kind) for every parameter, in `parameters()` order."""
        r = self.mirror.rnn
        out = []
        for layer in range(self.num_layers):
            for k, (kind, torch_name) in enumerate((("w_ih", "weight_ih"), ("w_hh", "weight_hh"), ("b_ih", "bias_ih"), ("b_hh", "bias_hh"))):
                out.append((getattr(r, f"{torch_name}_l{layer}"), 4 * layer + k, layer, kind))
        return out

    # -- state -----------------------------------------------------------------------------------------------------------------
    def zero_state(self, rows: int) -> list:
        """The zero state: one [rows, H_p] tensor per layer (GRU) or two (h, c) per layer (LSTM), batch-sharded, flat list [h_0, (c_0), h_1, (c_1), ...]."""
        n = 2 if self.lstm else 1
        return [self.ctx.zeros(rows, self.hidden_p, batch_sharded=True) for _ in range(self.num_layers * n)]

    # -- one layer, one step -------------------------------------------------------------------------------------------------
    def _gate(self, a: Any, k: int) -> Any:
        hp = self.hidden_p
        return a[:, k * hp : (k + 1) * hp]

    def _cell(self, layer: int, x: Any, h: Any, c: Any, cache: bool) -> tuple:
        """One step of one layer: (h', c' or None, cache dict)."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        cfg = ctx.compute_cfg
        gi = ttnn.linear(x, self.w_ih[layer], bias=self.b_ih[layer], compute_kernel_config=cfg)
        gh = ttnn.linear(h, self.w_hh[layer], bias=self.b_hh[layer], compute_kernel_config=cfg)
        if not self.lstm:
            r = ttnn.sigmoid(ttnn.add(self._gate(gi, 0), self._gate(gh, 0)))
            z = ttnn.sigmoid(ttnn.add(self._gate(gi, 1), self._gate(gh, 1)))
            ghn = self._gate(gh, 2)
            n = ttnn.tanh(ttnn.add(self._gate(gi, 2), ttnn.multiply(r, ghn)))
            h_new = ttnn.add(n, ttnn.multiply(z, ttnn.subtract(h, n)))  # (1 - z) n + z h
            return h_new, None, ({"x": x, "h": h, "r": r, "z": z, "n": n, "ghn": ghn} if cache else None)
        gates = ttnn.add(gi, gh)
        i, f, o = (ttnn.sigmoid(self._gate(gates, k)) for k in (0, 1, 3))
        g = ttnn.tanh(self._gate(gates, 2))
        c_new = ttnn.add(ttnn.multiply(f, c), ttnn.multiply(i, g))
        tc = ttnn.tanh(c_new)
        h_new = ttnn.multiply(o, tc)
        return h_new, c_new, ({"x": x, "h": h, "c": c, "i": i, "f": f, "g": g, "o": o, "tc": tc} if cache else None)

    def step(self, x: Any, state: list) -> tuple[Any, list]:
        """One time step through all layers (inference / rollout): returns (top layer's h', the new state, the same layout as `state`)."""
        n = 2 if self.lstm else 1
        new, inp = [], x
        for layer in range(self.num_layers):
            h = state[layer * n]
            c = state[layer * n + 1] if self.lstm else None
            h_new, c_new, _ = self._cell(layer, inp, h, c, cache=False)
            new += [h_new, c_new] if self.lstm else [h_new]
            inp = h_new
        return inp, new

    # -- a whole sequence (the update) ---------------------------------------------------------------------------------------
    def forward_sequence(self, xs: list, state0: list, keeps: list) -> tuple[list, list]:
        """The outputs of the top layer for the T steps `xs` (each [E, in_p]) from the initial state `state0`; after step t the state is multiplied by `keeps[t]` ([E, 1]: 1 - done, so the state is
        zero after a done) before it is fed to step t + 1. Returns (outputs [T] of [E, H_p], cache for `backward_sequence`)."""
        ttnn = self.ctx.ttnn
        n = 2 if self.lstm else 1
        state, outs, caches = list(state0), [], []
        for t, x in enumerate(xs):
            new, inp, step_cache = [], x, []
            for layer in range(self.num_layers):
                h = state[layer * n]
                c = state[layer * n + 1] if self.lstm else None
                h_new, c_new, cache = self._cell(layer, inp, h, c, cache=True)
                step_cache.append(cache)
                new += [h_new, c_new] if self.lstm else [h_new]
                inp = h_new
            outs.append(inp)
            caches.append(step_cache)
            state = [ttnn.multiply(s, keeps[t]) for s in new] if keeps[t] is not None else new
        return outs, caches

    def backward_sequence(self, caches: list, d_outs: list, keeps: list) -> tuple[list, list]:
        """BPTT: `d_outs[t]` = d loss / d (top layer's output at step t) ([E, H_p]); returns (parameter gradients in `parameters()` order, d loss / d inputs [T] of [E, in_p])."""
        ctx = self.ctx
        ttnn = ctx.ttnn
        cfg = ctx.compute_cfg
        T, L, n = len(caches), self.num_layers, 2 if self.lstm else 1
        hp = self.hidden_p
        dgi_all = [[None] * T for _ in range(L)]
        dgh_all = [[None] * T for _ in range(L)]
        d_inputs: list = [None] * T
        dh_next = [None] * L  # d loss / d h_in of step t + 1 (already multiplied by the keep mask of step t)
        dc_next = [None] * L
        for t in reversed(range(T)):
            d_above = d_outs[t]
            for layer in reversed(range(L)):
                cache = caches[t][layer]
                dh = d_above if dh_next[layer] is None else ttnn.add(d_above, dh_next[layer])
                if not self.lstm:
                    r, z, nn_, h, ghn = cache["r"], cache["z"], cache["n"], cache["h"], cache["ghn"]
                    dz = ttnn.multiply(dh, ttnn.subtract(h, nn_))
                    dn_pre = ttnn.multiply(ttnn.multiply(dh, ttnn.subtract(1.0, z)), ttnn.subtract(1.0, ttnn.square(nn_)))
                    dr_pre = ttnn.multiply(ttnn.multiply(dn_pre, ghn), ttnn.multiply(r, ttnn.subtract(1.0, r)))
                    dz_pre = ttnn.multiply(dz, ttnn.multiply(z, ttnn.subtract(1.0, z)))
                    dghn = ttnn.multiply(dn_pre, r)
                    dgi = ttnn.concat([dr_pre, dz_pre, dn_pre], dim=1)
                    dgh = ttnn.concat([dr_pre, dz_pre, dghn], dim=1)
                    dh_prev = ttnn.add(ttnn.multiply(dh, z), ttnn.matmul(dgh, self.w_hh[layer], transpose_b=True, compute_kernel_config=cfg))
                    dc_prev = None
                else:
                    i, f, g, o, tc, c = cache["i"], cache["f"], cache["g"], cache["o"], cache["tc"], cache["c"]
                    dc = ttnn.multiply(ttnn.multiply(dh, o), ttnn.subtract(1.0, ttnn.square(tc)))
                    if dc_next[layer] is not None:
                        dc = ttnn.add(dc, dc_next[layer])
                    do_pre = ttnn.multiply(ttnn.multiply(dh, tc), ttnn.multiply(o, ttnn.subtract(1.0, o)))
                    di_pre = ttnn.multiply(ttnn.multiply(dc, g), ttnn.multiply(i, ttnn.subtract(1.0, i)))
                    df_pre = ttnn.multiply(ttnn.multiply(dc, c), ttnn.multiply(f, ttnn.subtract(1.0, f)))
                    dg_pre = ttnn.multiply(ttnn.multiply(dc, i), ttnn.subtract(1.0, ttnn.square(g)))
                    dgi = dgh = ttnn.concat([di_pre, df_pre, dg_pre, do_pre], dim=1)
                    dh_prev = ttnn.matmul(dgh, self.w_hh[layer], transpose_b=True, compute_kernel_config=cfg)
                    dc_prev = ttnn.multiply(dc, f)
                dgi_all[layer][t], dgh_all[layer][t] = dgi, dgh
                d_above = ttnn.matmul(dgi, self.w_ih[layer], transpose_b=True, compute_kernel_config=cfg)  # d loss / d (this layer's input) = the layer below's d output
                if t > 0:  # the state that step t consumed is step t - 1's output times its keep mask
                    keep = keeps[t - 1]
                    dh_next[layer] = dh_prev if keep is None else ttnn.multiply(dh_prev, keep)
                    if self.lstm:
                        dc_next[layer] = dc_prev if keep is None else ttnn.multiply(dc_prev, keep)
            d_inputs[t] = d_above
        grads = []
        for layer in range(L):
            cat = lambda key: ttnn.concat([caches[t][layer][key] for t in range(T)], dim=0)
            x_all, h_all = cat("x"), cat("h")
            gi_all = ttnn.concat(dgi_all[layer], dim=0)
            gh_all = gi_all if self.lstm else ttnn.concat(dgh_all[layer], dim=0)
            grads += [
                ttnn.matmul(x_all, gi_all, transpose_a=True, compute_kernel_config=cfg),
                ttnn.matmul(h_all, gh_all, transpose_a=True, compute_kernel_config=cfg),
                ttnn.sum(gi_all, dim=0, keepdim=True, compute_kernel_config=cfg),
                ttnn.sum(gh_all, dim=0, keepdim=True, compute_kernel_config=cfg),
            ]
        return grads, d_inputs
