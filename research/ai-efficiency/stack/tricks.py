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
            gptq@<bits>/<group>    GPTQ: column-by-column rounding with Hessian-based error feedback,
                                   Hessians from the calibration set (one fp pass, not sequential)
            rgptq@<bits>/<group>   GPTQ in the rotated basis
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


def gptq(W, H, b, g, block=128, damp=0.01):
    """GPTQ (Frantar et al. 2022) with asymmetric group-wise grids; returns the dequantized weight.
    W: (out, in), H: (in, in) = sum of x x^T over calibration inputs. g must divide block."""
    W = W.clone().float()
    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.diag(H).mean() * torch.eye(len(H))
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    n = W.shape[1]
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1, Err = W[:, i1:i2].clone(), torch.zeros(W.shape[0], i2 - i1)
        for i in range(i2 - i1):
            if i % g == 0:  # grid for the next group, from the error-updated weights
                grp = W1[:, i:i + g]
                lo, hi = grp.amin(1), grp.amax(1)
                sc = (hi - lo).clamp_min(1e-8) / (2 ** b - 1)
            w, d = W1[:, i], U[i1 + i, i1 + i]
            q = torch.round((w - lo) / sc).clamp(0, 2 ** b - 1) * sc + lo
            Q[:, i1 + i] = q
            e = (w - q) / d
            W1[:, i:] -= e[:, None] * U[i1 + i, i1 + i:i2][None, :]
            Err[:, i] = e
        W[:, i2:] -= Err @ U[i1:i2, i2:]
    return Q


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
                assert kind in ("rtn", "rot", "gptq", "rgptq")
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
    def apply_weights(self, model, calib=None):
        """calib: (n, seq) token ids; needed for gptq/rgptq (Hessians from one fp forward pass)."""
        with torch.no_grad():
            if self.weights:
                b, g = self.weights
                kind = self.weights_kind
                linears = [m for layer in model.model.layers for m in layer.modules()
                           if isinstance(m, torch.nn.Linear)]
                if kind in ("gptq", "rgptq"):
                    H = {m: torch.zeros(m.in_features, m.in_features) for m in linears}
                    def accumulate(mod, inp, out):  # must return None, or it replaces the layer output
                        x = inp[0].reshape(-1, mod.in_features).float()
                        H[mod].add_(x.T @ x)
                    hooks = [m.register_forward_hook(accumulate) for m in linears]
                    for x in calib:
                        model(x[None])
                    for h in hooks:
                        h.remove()
                    for m in linears:
                        if kind == "rgptq":
                            R = rotation(m.in_features)
                            m.weight.copy_(gptq(m.weight @ R, R.T @ H[m] @ R, b, g) @ R.T)
                        else:
                            m.weight.copy_(gptq(m.weight, H[m], b, g))
                        del H[m]
                else:
                    for m in linears:
                        m.weight.copy_(quant_weight(m.weight, b, g, kind))
            if self.head:  # if tied, this also quantizes the input embedding (same stored matrix)
                assert self.head_kind in ("rtn", "rot"), "head supports rtn/rot"
                model.lm_head.weight.copy_(quant_weight(model.lm_head.weight, *self.head, self.head_kind))

    def make_key_coder(self, Ktr, Qtr):
        return KEY_CODERS[self.keys[0]][0](self.keys[1])(Ktr, Qtr)

    def quant_values(self, V):
        return quant_groups(V, *self.values) if self.values else V
