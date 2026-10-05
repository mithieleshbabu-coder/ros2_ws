"""
Run one stack of tricks (or a leave-one-out ablation) against the fixed benchmark and
append every result to results/ledger.jsonl.

    python run.py --spec "keys=qa-wf-gs@3 values=tok@3/64 sink=1"
    python run.py --spec "keys=qa-wf-gs@3 values=tok@3/64 sink=1" --ablate

Benchmark (fixed — do not change it to flatter a result; add a new one instead):
  quality  perplexity on WikiText-2 test and Python stdlib source, 512-token sequences,
           per-sequence NLL stored so any two runs can be compared paired, with a 95% CI
  bytes    analytic memory traffic per decode step (weights + KV cache read once per token),
           at two serving points; the ratio to the all-fp16 baseline is the speedup ceiling for
           bandwidth-bound decode. CPU wall time is not reported: fake-quantized tensors say
           nothing about real kernels.
Calibration (for data-aware key coders) runs on WikiText-2 train after weight quantization.
"""
import argparse, glob, json, math, os, subprocess, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from datasets import load_dataset
from tricks import Stack
from bytes_model import model_sizes, bytes_report

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "results", "ledger.jsonl")

p = argparse.ArgumentParser()
p.add_argument("--spec", default="")
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
p.add_argument("--seq", type=int, default=512)
p.add_argument("--n-calib", type=int, default=16)
p.add_argument("--calib", default="wiki", choices=["wiki", "mix"],
               help="wiki: WikiText-2 train; mix: half WikiText-2 train, half Python stdlib files "
                    "disjoint from the code eval set")
p.add_argument("--n-eval", type=int, default=32)
p.add_argument("--ablate", action="store_true", help="also run the spec with each trick removed")
p.add_argument("--note", default="")
args = p.parse_args()
torch.set_num_threads(4)

tok = AutoTokenizer.from_pretrained(args.model)


def chunks(text, n):
    ids = tok(text, return_tensors="pt").input_ids[0]
    k = min(n, len(ids) // args.seq)
    return ids[: k * args.seq].view(k, args.seq)


wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
STDLIB = sorted(glob.glob("/usr/lib/python3.11/*.py"))  # [:60] is the code eval set, [60:] calibration only
if args.calib == "wiki":
    CALIB = chunks("\n".join(wiki["train"]["text"][:20000]), args.n_calib)
else:
    h = args.n_calib // 2
    CALIB = torch.cat([chunks("\n".join(wiki["train"]["text"][:20000]), args.n_calib - h),
                       chunks("\n".join(open(f).read() for f in STDLIB[60:160]), h)])
EVALS = {
    "wiki": chunks("\n".join(wiki["test"]["text"]), args.n_eval),
    "code": chunks("\n".join(open(f).read() for f in STDLIB[:60]),
                   args.n_eval),
}

# ---------- attention with all KV tricks applied ----------
S = {"stack": Stack(), "mode": "eval", "store": {}, "coders": None, "hkv": None, "group": None}


def stack_attention(module, query, key, value, attention_mask, scaling=None, **kw):
    st, li = S["stack"], module.layer_idx
    B, H, T, D = query.shape
    Hkv = key.shape[1]
    G = H // Hkv
    if S["mode"] == "capture":
        d = S["store"].setdefault(li, {"k": [[] for _ in range(Hkv)], "q": [[] for _ in range(Hkv)]})
        for h in range(Hkv):
            d["k"][h].append(key[:, h].reshape(-1, D))
            d["q"][h].append(query[:, h * G:(h + 1) * G].reshape(-1, D))
    Kq, Vq = key, value
    if S["mode"] == "eval":
        if st.keys:
            Kq = torch.stack([S["coders"][li][h](key[:, h].reshape(-1, D)).view(B, T, D) for h in range(Hkv)], 1)
        Vq = st.quant_values(value)
        if st.sink:
            Kq, Vq = Kq.clone(), Vq.clone()
            Kq[:, :, :st.sink], Vq[:, :, :st.sink] = key[:, :, :st.sink], value[:, :, :st.sink]
    rep = lambda x: x.repeat_interleave(G, dim=1)
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    allowed = j <= i
    if st.window and S["mode"] == "eval":
        allowed &= (j < st.sink) | (i - j < st.window)
    scores = query @ rep(Kq).transpose(-1, -2) * scaling
    recent = None
    if st.recent and S["mode"] == "eval":
        recent = (i - j < st.recent)
        scores = torch.where(recent, query @ rep(key).transpose(-1, -2) * scaling, scores)
    A = torch.softmax(scores.masked_fill(~allowed, float("-inf")), -1)
    if recent is None:
        out = A @ rep(Vq)
    else:
        out = (A * recent) @ rep(value) + (A * ~recent) @ rep(Vq)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register("stack", stack_attention)


@torch.no_grad()
def seq_nll(model, ids):
    return [float(model(x[None], labels=x[None]).loss) for x in ids]


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE).decode().strip()
    except Exception:
        return "unknown"


def run(spec):
    t0 = time.time()
    torch.manual_seed(0)  # calibration subsampling and any randomized coder are reproducible
    st = Stack(spec)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    model.set_attn_implementation("stack")
    S["stack"], S["mode"] = Stack(), "eval"  # weight calibration always sees the uncompressed KV path
    st.apply_weights(model, CALIB)
    S["stack"] = st
    if st.keys:
        S["mode"], S["store"] = "capture", {}
        seq_nll(model, CALIB)
        S["coders"] = {}
        for li, d in S["store"].items():
            S["coders"][li] = []
            for h in range(len(d["k"])):
                K, Q = torch.cat(d["k"][h]), torch.cat(d["q"][h])
                K, Q = K[torch.randperm(len(K))[:8192]], Q[torch.randperm(len(Q))[:8192]]
                S["coders"][li].append(st.make_key_coder(K, Q))
        S["store"] = {}
    S["mode"] = "eval"
    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_sha(), "model": args.model,
           "spec": spec, "calib": args.calib, "n_calib": args.n_calib, "n_eval": args.n_eval, "seq": args.seq, "note": args.note}
    for name, ids in EVALS.items():
        nll = np.array(seq_nll(model, ids))
        se = nll.std(ddof=1) / math.sqrt(len(nll))
        res[name] = {"ppl": float(math.exp(nll.mean())),
                     "ppl_ci95": [float(math.exp(nll.mean() - 1.96 * se)), float(math.exp(nll.mean() + 1.96 * se))],
                     "nll": [round(float(x), 5) for x in nll]}
    res["bytes"] = bytes_report(st, model_sizes(model))
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(res) + "\n")
    return res


def paired(a, b, name):
    """Mean per-sequence NLL difference b - a with 95% CI (positive = b is worse)."""
    d = np.array(b[name]["nll"]) - np.array(a[name]["nll"])
    se = d.std(ddof=1) / math.sqrt(len(d))
    return d.mean(), 1.96 * se


def show(r, label):
    bt = r["bytes"]
    print(f"{label:28s} wiki {r['wiki']['ppl']:7.3f}  code {r['code']['ppl']:6.3f}  "
          f"ceiling x{bt['ctx4k_b1']['ceiling_speedup']:.2f} (4k,b1)  x{bt['ctx32k_b8']['ceiling_speedup']:.2f} (32k,b8)",
          flush=True)


full = run(args.spec)
show(full, args.spec or "(fp baseline)")
if args.ablate:
    base = run("")
    show(base, "(fp baseline)")
    print("\nleave-one-out: what each trick costs in quality (paired NLL diff vs full stack, 95% CI)")
    for t in Stack(args.spec).tokens():
        r = run(Stack(args.spec).without(t).spec)
        dw, cw = paired(r, full, "wiki")
        dc, cc = paired(r, full, "code")
        show(r, f"without {t}")
        print(f"{'':28s} cost of {t}: wiki {dw:+.4f}±{cw:.4f}  code {dc:+.4f}±{cc:.4f} nats/token")
