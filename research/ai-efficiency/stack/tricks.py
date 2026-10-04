"""
Tricks are parsed from a spec string such as

    keys=qa-wf-gs@3 values=tok@2/64 weights=rtn@4/128 sink=1 recent=64 window=0

Slot        syntax                 meaning
keys        <coder>@<bits>         key-cache coder from kv-transform-coding/coders.py, or kivi
values      tok@<bits>/<group>     per-token asymmetric uniform quantization, groups of channels
weights     rtn@<bits>/<group>     group-wise round-to-nearest on all decoder Linear layers
            rot@<bits>/<group>     same after a random orthogonal rotation of the input dimension
                                   (QuIP/QuaRot-style incoherence; at deployment the matching rotation
                                   is applied to activations or fused into the previous layer)
head        rtn@<bits>/<group>     same for the output projection (tied with the input embedding in Qwen)
sink        <n>                    first n tokens keep full-precision K and V
recent      <r>                    K/V of the r most recent tokens stay full precision
window      <w>                    evict everything except sink + last w tokens (StreamingLLM); 0 = off

Every trick reports its storage cost so the harness can account bytes exactly.
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "kv-transform-coding"))
from coders import turbo_mse, transform_code, weighted_basis_code  # noqa: E402

FP_BITS = 16  # baseline storage precision for anything not quantized


def kivi(b, group=128):
    """Per-channel asymmetric uniform, min/max per group of `group` tokens (KIVI-style keys)."""
    def fit(Ktr, Qtr):
        def enc(K):
            G = K.view(-1, group, K.shape[1])
            lo, hi = G.amin(1, keepdim=True), G.amax(1, keepdim=True)
            s = (hi - lo).clamp_min(1e-8) / (2 ** b - 1)
            return (torch.round((G - lo) / s) * s + lo).view_as(K)
        return enc
    return fit


KEY_CODERS = {
    "kivi": (kivi, lambda b, d: b + 2 * FP_BITS / 128),
    "turbo-c": (lambda b: turbo_mse(b, center=True), lambda b, d: b + FP_BITS / d),
    "klt-wf": (lambda b: transform_code(b, False, True, extra_bits=16), lambda b, d: b + FP_BITS / d),
    "qa-wf": (lambda b: transform_code(b, True, True, extra_bits=16), lambda b, d: b + FP_BITS / d),
    "qa-wf-gs": (lambda b: transform_code(b, True, True, gain_shape=True), lambda b, d: b + FP_BITS / d),
    "kbasis-qw": (lambda b: weighted_basis_code(b, "k", extra_bits=16), lambda b, d: b + FP_BITS / d),
    "qbasis-qw": (lambda b: weighted_basis_code(b, "q", extra_bits=16), lambda b, d: b + FP_BITS / d),
}


def quant_groups(x, b, group):
    """Asymmetric uniform quantization over contiguous groups of the last dim."""
    shp = x.shape
    G = x.reshape(*shp[:-1], shp[-1] // group, group)
    lo, hi = G.amin(-1, keepdim=True), G.amax(-1, keepdim=True)
    s = (hi - lo).clamp_min(1e-8) / (2 ** b - 1)
    return (torch.round((G - lo) / s) * s + lo).reshape(shp)


_ROT = {}


def rotation(n):
    """Fixed random orthogonal n x n matrix (one per input size, seeded)."""
    if n not in _ROT:
        g = torch.Generator().manual_seed(n)
        q, r = torch.linalg.qr(torch.randn(n, n, generator=g))
        _ROT[n] = q * torch.sign(torch.diag(r))
    return _ROT[n]


def quant_weight(W, b, g, kind):
    if kind == "rot":
        R = rotation(W.shape[1])
        return quant_groups(W @ R, b, g) @ R.T
    return quant_groups(W, b, g)


class Stack:
    def __init__(self, spec=""):
        self.spec = spec.strip()
        self.keys = self.values = self.weights = self.head = None
        self.sink = self.recent = self.window = 0
        for tok in self.spec.split():
            slot, arg = tok.split("=", 1)
            if slot == "keys":
                name, b = arg.split("@")
                assert name in KEY_CODERS, f"unknown key coder {name}"
                self.keys = (name, int(b))
            elif slot == "values":
                kind, rest = arg.split("@")
                b, g = rest.split("/")
                assert kind == "tok"
                self.values = (int(b), int(g))
            elif slot in ("weights", "head"):
                kind, rest = arg.split("@")
                b, g = rest.split("/")
                assert kind in ("rtn", "rot")
                setattr(self, slot, (int(b), int(g)))
                setattr(self, slot + "_kind", kind)
            elif slot in ("sink", "recent", "window"):
                setattr(self, slot, int(arg))
            else:
                raise ValueError(f"unknown slot {slot}")

    def tokens(self):
        return self.spec.split()

    def without(self, tok):
        return Stack(" ".join(t for t in self.tokens() if t != tok))

    # ---- storage accounting (bits per stored element) ----
    def key_bits(self, d):
        return KEY_CODERS[self.keys[0]][1](self.keys[1], d) if self.keys else FP_BITS

    def value_bits(self):
        return self.values[0] + 2 * FP_BITS / self.values[1] if self.values else FP_BITS

    def weight_bits(self):
        return self.weights[0] + 2 * FP_BITS / self.weights[1] if self.weights else FP_BITS

    def head_bits(self):
        return self.head[0] + 2 * FP_BITS / self.head[1] if self.head else FP_BITS

    # ---- application ----
    def apply_weights(self, model):
        with torch.no_grad():
            if self.weights:
                b, g = self.weights
                for layer in model.model.layers:
                    for m in layer.modules():
                        if isinstance(m, torch.nn.Linear):
                            m.weight.copy_(quant_weight(m.weight, b, g, self.weights_kind))
            if self.head:  # if tied, this also quantizes the input embedding (same stored matrix)
                model.lm_head.weight.copy_(quant_weight(model.lm_head.weight, *self.head, self.head_kind))

    def make_key_coder(self, Ktr, Qtr):
        return KEY_CODERS[self.keys[0]][0](self.keys[1])(Ktr, Qtr)

    def quant_values(self, V):
        return quant_groups(V, *self.values) if self.values else V
