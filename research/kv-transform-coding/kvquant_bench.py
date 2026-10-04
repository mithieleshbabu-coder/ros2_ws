"""
Key quantization for attention / inner-product search, at a fixed bit budget.

Question: TurboQuant rotates keys *randomly* and spends the same bits on every
coordinate. Classical transform coding says: rotate to the data's principal axes
(KLT) and allocate bits by variance (reverse water-filling). And because attention
only uses q.k, the error that matters is (k - k_hat)^T C_q (k - k_hat), with
C_q = E[q q^T] -- so do the transform coding in the query-weighted space.

Methods (only keys are quantized; queries stay full precision, as in a KV cache)
  turbo-mse   random rotation of k/||k||, Lloyd-Max scalar quantizer, b bits/coord, ||k|| stored
  turbo-prod  (b-1)-bit turbo-mse + 1-bit QJL sketch of the residual (unbiased q.k estimate)
  klt-unif    #1 ablation: PCA rotation of centred keys, b bits on every coordinate
  klt-wf      #1: PCA rotation + greedy bit allocation by variance (reverse water-filling)
  qa-unif     #2 ablation: query-weighted space, PCA rotation, uniform bits
  qa-wf       #1+#2: query-weighted space (k' = C_q^{1/2} k), PCA rotation, water-filling
  qa-wf-gs    qa-wf on the direction k/||k|| plus the stored norm (gain-shape, like turbo; +16 bits)

All methods spend exactly b*D bits per key (turbo also stores a norm, i.e. +16 bits).
KLT/QA store their rotation, mean and bit table once per head (amortized, not counted).

Scenarios
  glove-shift calibration on the 20k most frequent words, evaluation on rare words
             (rank 70k-120k): tests whether the data-aware methods break under shift.
  synth-iso  synth-head without the extra per-channel scaling: checks the gains are not
             just an artefact of how anisotropic the synthetic head was made.
  glove-ip   keys and queries are both raw GloVe vectors (D=300); same distribution,
             so #2 should give little -> a control.
  synth-head a synthetic attention head over real hidden states: h = GloVe vector,
             k = h W_k, q = h W_q, v = h W_v (d_head=128), each W = Gaussian x a
             log-normal per-channel scale, so q and k are anisotropic *differently*.
             Real LLM keys/queries are needed to confirm anything (next step).

Metrics
  ip-err   E[(q.k - q.k_hat)^2] / Var(q.k)                     (lower is better)
  R@10     recall of the true top-10 keys by q.k among 50k     (higher is better)
  attn-err ||o - o_hat|| / ||o||, softmax attention over 2048-key contexts (lower is better)
"""
import json, math
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(4)
BITS = [1, 2, 3, 4]


# ---------- Lloyd-Max codebooks for N(0,1) ----------
def lloyd_max(b, n=2_000_000, iters=200):
    x = np.sort(np.random.randn(n))
    c = np.quantile(x, (np.arange(2 ** b) + 0.5) / 2 ** b)
    for _ in range(iters):
        edges = (c[1:] + c[:-1]) / 2
        idx = np.searchsorted(edges, x)
        c = np.bincount(idx, x, 2 ** b) / np.bincount(idx, minlength=2 ** b)
    edges = (c[1:] + c[:-1]) / 2
    mse = np.mean((x - c[np.searchsorted(edges, x)]) ** 2)
    return torch.tensor(c, dtype=torch.float32), torch.tensor(edges, dtype=torch.float32), mse


CB = {b: lloyd_max(b) for b in range(1, 9)}
UNIT_MSE = [1.0] + [CB[b][2] for b in range(1, 9)]  # distortion of a unit-variance coord at b bits


def quant(z, b):
    """Lloyd-Max quantize a tensor of ~N(0,1) values at b bits (b=0 -> 0)."""
    if b == 0:
        return torch.zeros_like(z)
    c, e, _ = CB[b]
    return c[torch.bucketize(z, e)]


# ---------- methods: fit(K_train, Q_train) -> encode(K) -> scorer(Q) ----------
def rand_orth(D, seed):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(D, D, generator=g))
    return q * torch.sign(torch.diag(r))


def turbo_mse(b):
    def fit(Ktr, Qtr):
        D = Ktr.shape[1]
        R = rand_orth(D, 1)
        def enc(K):
            n = K.norm(dim=1, keepdim=True)
            z = (K / n) @ R * math.sqrt(D)  # coords ~ N(0,1)
            return (quant(z, b) / math.sqrt(D)) @ R.T * n
        return enc
    return fit


def turbo_prod(b):
    def fit(Ktr, Qtr):
        D = Ktr.shape[1]
        mse_enc = turbo_mse(b - 1)(Ktr, Qtr) if b > 1 else (lambda K: torch.zeros_like(K))
        S = torch.randn(D, D, generator=torch.Generator().manual_seed(2))
        def enc(K):
            Kh = mse_enc(K)
            r = K - Kh
            return Kh, torch.sign(r @ S.T), r.norm(dim=1)
        def score(Q, code):
            Kh, sg, rn = code
            qjl = math.sqrt(math.pi / 2) / D * (Q @ S.T) @ sg.T * rn[None]
            return Q @ Kh.T + qjl
        enc.score = score
        return enc
    return fit


def allocate(lam, total, cap=8):
    """Greedy marginal-return bit allocation (optimal for convex per-coord distortion)."""
    bits = np.zeros(len(lam), dtype=int)
    for _ in range(total):
        gain = np.array([lam[i] * (UNIT_MSE[bits[i]] - UNIT_MSE[bits[i] + 1]) if bits[i] < cap else -1
                         for i in range(len(lam))])
        bits[gain.argmax()] += 1
    return bits


def transform_code(b, query_aware, waterfill, gain_shape=False):
    def fit(Ktr, Qtr):
        if gain_shape:  # code the direction k/||k|| with the transform coder, store ||k|| separately
            inner = transform_code(b, query_aware, waterfill)(Ktr / Ktr.norm(dim=1, keepdim=True), Qtr)
            def enc(K):
                n = K.norm(dim=1, keepdim=True)
                return inner(K / n) * n
            enc.bits = inner.bits
            return enc
        D = Ktr.shape[1]
        if query_aware:
            C = Qtr.T @ Qtr / len(Qtr) + 1e-6 * torch.eye(D)
            ev, U = torch.linalg.eigh(C)
            W = U @ torch.diag(ev.sqrt()) @ U.T       # C_q^{1/2}
            Winv = U @ torch.diag(1 / ev.sqrt()) @ U.T
        else:
            W = Winv = torch.eye(D)
        Kp = Ktr @ W
        mu = Kp.mean(0)
        lam, V = torch.linalg.eigh(torch.cov((Kp - mu).T))
        lam = lam.clamp_min(1e-12)
        bits = allocate(lam.numpy(), b * D) if waterfill else np.full(D, b)
        sd = lam.sqrt()
        def enc(K):
            z = ((K @ W - mu) @ V) / sd
            zq = torch.stack([quant(z[:, i], int(bits[i])) for i in range(D)], 1)
            return ((zq * sd) @ V.T + mu) @ Winv
        enc.bits = bits
        return enc
    return fit


METHODS = {
    "turbo-mse": turbo_mse,
    "turbo-prod": turbo_prod,
    "klt-unif": lambda b: transform_code(b, False, False),
    "klt-wf": lambda b: transform_code(b, False, True),
    "qa-unif": lambda b: transform_code(b, True, False),
    "qa-wf": lambda b: transform_code(b, True, True),
    "qa-wf-gs": lambda b: transform_code(b, True, True, gain_shape=True),
}
import os
if os.environ.get("METHODS"):
    METHODS = {k: METHODS[k] for k in os.environ["METHODS"].split(",")}


# ---------- scenarios ----------
H = torch.tensor(np.load("glove120k.npy"))
perm = torch.randperm(len(H))
tr, db, qs, qtr = perm[:20000], perm[20000:70000], perm[70000:71000], perm[71000:91000]


def scenario(name):
    if name == "glove-ip":
        return H[tr], H[qtr], H[db], H[qs], H[db] @ torch.randn(300, 128) / 300 ** .5
    if name == "glove-shift":  # calibrate on the 20k most frequent words, test on rare words
        rare = torch.arange(70000, 120000)
        return H[:20000], H[20000:40000], H[rare], H[rare[torch.randperm(len(rare))[:1000]]], H[rare] @ torch.randn(300, 128) / 300 ** .5
    g = torch.Generator().manual_seed(7)
    spread = 0.0 if name == "synth-iso" else 1.0  # synth-iso: no extra per-channel scaling
    def proj():
        return torch.randn(300, 128, generator=g) / 300 ** .5 * torch.exp(spread * torch.randn(128, generator=g))
    Wk, Wq, Wv = proj(), proj(), proj()
    Ktr, Qtr = H[tr] @ Wk, H[qtr] @ Wq
    return Ktr, Qtr, H[db] @ Wk, H[qs] @ Wq, H[db] @ Wv


def evaluate(scen):
    Ktr, Qtr, K, Q, Vv = scenario(scen)
    S = Q @ K.T
    top = S.topk(10, dim=1).indices
    # attention: each query attends over its own random 2048-key context
    g = torch.Generator().manual_seed(3)
    ctx = torch.stack([torch.randperm(len(K), generator=g)[:2048] for _ in range(len(Q))])
    temp = 2.5 / S.gather(1, ctx).std(1).mean()  # logit std ~2.5: peaked but not one-hot
    def attn(Sm):
        a = torch.softmax(Sm.gather(1, ctx) * temp, 1)
        return torch.einsum("qc,qcd->qd", a, Vv[ctx])
    o = attn(S)
    out = []
    for b in BITS:
        for m, make in METHODS.items():
            enc = make(b)(Ktr, Qtr)
            code = enc(K)
            Sh = enc.score(Q, code) if hasattr(enc, "score") else Q @ code.T
            ip = float(((S - Sh) ** 2).mean() / S.var())
            found = Sh.topk(10, dim=1).indices
            rec = float(np.mean([len(set(top[i].tolist()) & set(found[i].tolist())) / 10 for i in range(len(Q))]))
            ae = float(((o - attn(Sh)).norm(dim=1) / o.norm(dim=1)).mean())
            extra = f" bits used: max={enc.bits.max()} zero-bit coords={int((enc.bits == 0).sum())}" if hasattr(enc, "bits") and "wf" in m else ""
            print(f"{scen:10s} b={b} {m:10s} ip-err={ip:.4f} R@10={rec:.3f} attn-err={ae:.4f}{extra}", flush=True)
            out.append(dict(scenario=scen, bits=b, method=m, ip_err=ip, recall10=rec, attn_err=ae))
    return out


import sys
scens = sys.argv[1:] or ["glove-ip", "synth-head", "glove-shift", "synth-iso"]
results = [r for sc in scens for r in evaluate(sc)]
json.dump(results, open("kv_results_" + "_".join(scens) + ".json", "w"), indent=1)
