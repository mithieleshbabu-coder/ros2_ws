"""
Key-cache quantization inside a real LLM (Qwen2.5-0.5B, CPU).

1. Capture post-RoPE keys and queries (what the KV cache actually stores / attends with)
   for every layer and KV head on calibration text (WikiText-2 train).
2. Fit each key coder per (layer, KV head). With GQA, the queries that read a KV head are
   the 7 query heads in its group; their pooled second moment is C_q.
3. Run the model with keys replaced by their quantized reconstruction (values stay full
   precision) and report perplexity plus the mean relative attention-output error.

turbo-c is turbo-mse on mean-centred keys: a fairer TurboQuant baseline, since LLM keys
share a large offset (k_proj bias) that is irrelevant to softmax attention.

kbasis-qw / qbasis-qw: key-PCA basis (AATC-style) or query-covariance eigenbasis (OSCAR-style)
with query-weighted water-filling -- the closest prior work, at the same allocation rule.

Eval sets: WikiText-2 test (in-domain) and Python stdlib source code (domain shift).

Bit budgets are equal per key: b*64 + 16 bits for every method.
turbo-mse, turbo-c and qa-wf-gs spend the 16 bits on the key norm, kivi on its group ranges; klt-wf and qa-wf get them as
extra water-filling bits.
"""
import argparse, glob, json, math, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from datasets import load_dataset
from coders import turbo_mse, transform_code, weighted_basis_code

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
p.add_argument("--seq", type=int, default=512)
p.add_argument("--n-calib", type=int, default=16)
p.add_argument("--n-eval", type=int, default=16)
p.add_argument("--bits", default="2,3,4")
p.add_argument("--methods", default="kivi,turbo-mse,turbo-c,klt-wf,qa-wf,qa-wf-gs")
p.add_argument("--out", default="results/kv_llm_results.json")
p.add_argument("--keep-sink", action="store_true", help="keep token 0 (attention sink) keys in full precision, for all methods")
args = p.parse_args()
torch.manual_seed(0)
torch.set_num_threads(4)

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
cfg = model.config
L, HKV = cfg.num_hidden_layers, cfg.num_key_value_heads
GROUP = cfg.num_attention_heads // HKV


def chunks(text, n):
    ids = tok(text, return_tensors="pt").input_ids[0]
    k = min(n, len(ids) // args.seq)
    return ids[: k * args.seq].view(k, args.seq)


wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
calib = chunks("\n".join(wiki["train"]["text"][:20000]), args.n_calib)
evals = {
    "wiki": chunks("\n".join(wiki["test"]["text"]), args.n_eval),
    "code": chunks("\n".join(open(f).read() for f in sorted(glob.glob("/usr/lib/python3.11/*.py"))[:60]),
                   args.n_eval),
}


# ---------- attention hook: capture or quantize keys ----------
STATE = {"mode": "fp", "store": {}, "coders": None, "err": []}


def kvq_attention(module, query, key, value, attention_mask, **kw):
    li = module.layer_idx
    if STATE["mode"] == "capture":
        B, H, T, D = key.shape
        ks = STATE["store"].setdefault(li, {"k": [[] for _ in range(HKV)], "q": [[] for _ in range(HKV)]})
        for h in range(HKV):
            ks["k"][h].append(key[:, h].reshape(-1, D))
            ks["q"][h].append(query[:, h * GROUP:(h + 1) * GROUP].reshape(-1, D))
    elif STATE["mode"] == "quant":
        B, H, T, D = key.shape
        kq = torch.stack([STATE["coders"][li][h](key[:, h].reshape(-1, D)).view(B, T, D)
                          for h in range(HKV)], 1)
        if args.keep_sink:
            kq[:, :, 0] = key[:, :, 0]
        ref, _ = sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
        out, w = sdpa_attention_forward(module, query, kq, value, attention_mask, **kw)
        STATE["err"].append(float((out - ref).norm() / ref.norm()))  # Frobenius ratio per layer call
        return out, w
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)


AttentionInterface.register("kvq", kvq_attention)
model.set_attn_implementation("kvq")


@torch.no_grad()
def run(ids, bs=4):
    nll, n = 0.0, 0
    for i in range(0, len(ids), bs):
        x = ids[i:i + bs]
        loss = model(x, labels=x).loss
        nll += float(loss) * x.numel(); n += x.numel()
    return math.exp(nll / n)


# ---------- capture calibration keys/queries ----------
t0 = time.time()
STATE["mode"] = "capture"
run(calib)
STATE["mode"] = "fp"
data = {}
for li, d in STATE["store"].items():
    for h in range(HKV):
        K, Q = torch.cat(d["k"][h]), torch.cat(d["q"][h])
        data[li, h] = (K[torch.randperm(len(K))[:8192]], Q[torch.randperm(len(Q))[:8192]])
STATE["store"] = {}
print(f"captured calibration K/Q in {time.time()-t0:.0f}s", flush=True)


# ---------- coders ----------
def kivi(b, group=128):
    """KIVI-style baseline: per-channel asymmetric uniform quantization, min/max computed on the
    fly for each group of `group` tokens (fp16 min+max per channel per group = 32/group bits
    per value, i.e. 0.25 at group=128 -- the same overhead as a 16-bit norm per 64-d key).
    Rows arrive as (batch*seq) in order and seq is a multiple of `group`."""
    def fit(Ktr, Qtr):
        def enc(K):
            G = K.view(-1, group, K.shape[1])
            lo, hi = G.amin(1, keepdim=True), G.amax(1, keepdim=True)
            s = (hi - lo).clamp_min(1e-8) / (2 ** b - 1)
            return (torch.round((G - lo) / s) * s + lo).view_as(K)
        return enc
    return fit


MAKE = {
    "kivi": kivi,
    "turbo-mse": turbo_mse,
    "turbo-c": lambda b: turbo_mse(b, center=True),
    "klt-wf": lambda b: transform_code(b, False, True, extra_bits=16),
    "qa-wf": lambda b: transform_code(b, True, True, extra_bits=16),
    "kbasis-qw": lambda b: weighted_basis_code(b, "k", extra_bits=16),  # AATC-style
    "qbasis-qw": lambda b: weighted_basis_code(b, "q", extra_bits=16),  # OSCAR rotation + allocation
    "qa-wf-gs": lambda b: transform_code(b, True, True, gain_shape=True),
}

results = []
for name, ids in evals.items():
    t0 = time.time()
    ppl = run(ids)
    results.append(dict(eval=name, method="fp32", bits=32, ppl=ppl, attn_err=0.0))
    print(f"{name:5s} fp32               ppl={ppl:8.3f} ({time.time()-t0:.0f}s)", flush=True)

for b in map(int, args.bits.split(",")):
    for m in args.methods.split(","):
        STATE["coders"] = {li: [MAKE[m](b)(*data[li, h]) for h in range(HKV)] for li in range(L)}
        STATE["mode"] = "quant"
        for name, ids in evals.items():
            STATE["err"] = []
            t0 = time.time()
            ppl = run(ids)
            ae = float(np.mean(STATE["err"]))
            results.append(dict(eval=name, method=m, bits=b, ppl=ppl, attn_err=ae))
            print(f"{name:5s} {m:10s} b={b}  ppl={ppl:8.3f} attn-err={ae:.4f} ({time.time()-t0:.0f}s)", flush=True)
        STATE["mode"] = "fp"
        json.dump(results, open(args.out, "w"), indent=1)
