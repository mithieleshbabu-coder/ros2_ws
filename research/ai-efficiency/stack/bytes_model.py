"""Analytic memory traffic per decode step: weights read once + KV cache of every sequence in the batch."""
import torch
from tricks import Stack, FP_BITS

SERVING_POINTS = {"ctx4k_b1": (4096, 1), "ctx32k_b8": (32768, 8)}


def model_sizes(model):
    cfg = model.config
    lin = sum(m.weight.numel() for layer in model.model.layers for m in layer.modules()
              if isinstance(m, torch.nn.Linear))
    return {
        "L": cfg.num_hidden_layers,
        "Hkv": cfg.num_key_value_heads,
        "D": getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
        "lin": lin,
        "head": model.lm_head.weight.numel(),  # read in full every decode step for the logits
        # norms and biases are read every step; the input embedding only one row per token (~0)
        "small": sum(p.numel() for n, p in model.named_parameters() if "norm" in n or "bias" in n),
    }


def step_bytes(st, sz, C, B):
    w = sz["lin"] * st.weight_bits() / 8 + sz["head"] * st.head_bits() / 8 + sz["small"] * FP_BITS / 8
    n = min(C, st.sink + st.window) if st.window else C
    n_fp = min(n, st.sink + st.recent)
    per_tok_q = sz["L"] * sz["Hkv"] * sz["D"] * (st.key_bits(sz["D"]) + st.value_bits()) / 8
    per_tok_fp = sz["L"] * sz["Hkv"] * sz["D"] * 2 * FP_BITS / 8
    return w + B * (n_fp * per_tok_fp + (n - n_fp) * per_tok_q)


def bytes_report(st, sz):
    out = {}
    for name, (C, B) in SERVING_POINTS.items():
        b, b0 = step_bytes(st, sz, C, B), step_bytes(Stack(), sz, C, B)
        out[name] = {"MB_per_step": b / 2 ** 20, "ceiling_speedup": b0 / b}
    return out
