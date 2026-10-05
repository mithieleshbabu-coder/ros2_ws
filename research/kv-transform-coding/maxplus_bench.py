"""
Does a learned max-plus (tropical / morphological) projection compress embeddings
better than standard baselines at equal storage?

Data : GloVe 300-d word vectors (first 120k words), L2-normalised, cosine similarity.
Split: 20k train (for learned methods / PCA fit), 50k database, 1k held-out queries.
Task : nearest-neighbour search done entirely in the compressed space.
Metric:
  R10@10  - fraction of the true top-10 neighbours found in the compressed top-10
  R10@100 - same, but within the compressed top-100 (search-then-rerank setting)

Every method stores d float32 numbers per vector (int8 baseline: 4d int8 = same bytes).

Methods
  random     : Gaussian JL projection, y = W x
  pca        : top-d principal directions
  pca-int8   : 4d PCA dims, each scalar-quantised to int8 (same bytes as d floats)
  lin-learn  : y = W x, W trained with the same loss as the tropical models (fair control)
  maxplus    : y_i = max_j (W_ij + x_j)                         (the TCFP Phase-2 operator)
  morph      : d/2 max-plus + d/2 min-plus outputs (dilation + erosion, Ritter/Sussner style)
  maxplus-sx : y_i = max_j (W_ij + A_ij * x_j) - extra multiplicative weight per entry.
               Not multiplier-free any more; included to see whether *any* max-based
               layer helps, separately from the "no multiplications" claim.

Training (learned methods): listwise similarity distillation. For a batch of vectors,
target = softmax(cos_sim / tau) over the batch, prediction = softmax(-||z_a - z_b||^2 * s).
Hard max has gradient only at the argmax, so max-based layers are trained with
logsumexp(.. / t) and t annealed towards 0; evaluation always uses the hard max.
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--data", default="glove120k.npy")
p.add_argument("--dims", default="8,16,32,64")
p.add_argument("--seeds", default="0,1,2")
p.add_argument("--steps", default=2000, type=int)
p.add_argument("--batch", default=1024, type=int)
p.add_argument("--out", default="results.json")
args = p.parse_args()
torch.set_num_threads(4)

X = np.load(args.data)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
rng = np.random.default_rng(1234)
perm = rng.permutation(len(X))
Xtr = torch.tensor(X[perm[:20000]])
Xdb = torch.tensor(X[perm[20000:70000]])
Xq = torch.tensor(X[perm[70000:71000]])

# exact ground truth
gt = (Xq @ Xdb.T).topk(10, dim=1).indices


def recall(zq, zdb, k):
    d2 = torch.cdist(zq, zdb)  # L2 search in compressed space
    found = d2.topk(k, dim=1, largest=False).indices
    hits = [len(set(gt[i].tolist()) & set(found[i].tolist())) for i in range(len(gt))]
    return float(np.mean(hits)) / 10


def evaluate(f):
    with torch.no_grad():
        zq, zdb = f(Xq), f(Xdb)
    return recall(zq, zdb, 10), recall(zq, zdb, 100)


# ---------------- non-learned baselines ----------------
def random_proj(d, seed):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(300, d, generator=g) / d ** 0.5
    return lambda x: x @ W


mu = Xtr.mean(0)
_, _, Vt = torch.linalg.svd(Xtr - mu, full_matrices=False)


def pca(d):
    V = Vt[:d].T
    return lambda x: (x - mu) @ V


def pca_int8(d):
    V = Vt[: min(4 * d, 300)].T
    ztr = (Xtr - mu) @ V
    lo, hi = ztr.min(0).values, ztr.max(0).values
    scale = (hi - lo) / 255

    def f(x):
        z = (x - mu) @ V
        q = torch.clamp(torch.round((z - lo) / scale), 0, 255)
        return q * scale  # dequantised for distance computation
    return f


# ---------------- learned models ----------------
class Linear(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.W = nn.Parameter(Vt[:d].T.clone())  # PCA init: fair, strong start
        s.b = nn.Parameter(-(mu @ s.W).detach())

    def forward(s, x, t=None):
        return x @ s.W + s.b


def smax(a, dim, t):
    return torch.amax(a, dim) if t is None else t * torch.logsumexp(a / t, dim)


class MaxPlus(nn.Module):
    """y_i = max_j (W_ij + x_j) [optionally with min-plus half, or per-entry scale A_ij]."""
    def __init__(s, d, morph=False, scaled=False, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        s.morph, s.scaled = morph, scaled
        s.W = nn.Parameter(0.3 * torch.randn(d, 300, generator=g))
        if scaled:
            s.A = nn.Parameter(torch.randn(d, 300, generator=g))

    def forward(s, x, t=None):
        if s.scaled:
            a = s.W[None] + s.A[None] * x[:, None, :]
        else:
            a = s.W[None] + x[:, None, :]
        if not s.morph:
            return smax(a, 2, t)
        h = a.shape[1] // 2
        return torch.cat([smax(a[:, :h], 2, t), -smax(-a[:, h:], 2, t)], 1)


def train(model, seed, tau=0.05):
    torch.manual_seed(seed)
    logs = nn.Parameter(torch.tensor(3.0))  # learned similarity scale
    opt = torch.optim.Adam(list(model.parameters()) + [logs], lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    eye = torch.eye(args.batch, dtype=torch.bool)
    for step in range(args.steps):
        frac = step / args.steps
        t = 0.1 * (0.01 / 0.1) ** frac  # smooth-max temperature 0.1 -> 0.001
        if isinstance(model, Linear):
            t = None
        idx = torch.randint(0, len(Xtr), (args.batch,))
        xb = Xtr[idx]
        target = F.softmax((xb @ xb.T / tau).masked_fill(eye, -1e9), 1)
        z = model(xb, t)
        pred = F.log_softmax((-torch.cdist(z, z) ** 2 * logs.exp()).masked_fill(eye, -1e9), 1)
        loss = F.kl_div(pred, target, reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    return lambda x: model(x, None)  # hard max at eval


results = []
for d in map(int, args.dims.split(",")):
    for seed in map(int, args.seeds.split(",")):
        runs = {"random": lambda: random_proj(d, seed)}
        if seed == 0:  # deterministic baselines: run once
            runs |= {"pca": lambda: pca(d), "pca-int8": lambda: pca_int8(d)}
        runs |= {
            "lin-learn": lambda: train(Linear(d), seed),
            "maxplus": lambda: train(MaxPlus(d, seed=seed), seed),
            "morph": lambda: train(MaxPlus(d, morph=True, seed=seed), seed),
            "maxplus-sx": lambda: train(MaxPlus(d, scaled=True, seed=seed), seed),
        }
        for name, make in runs.items():
            t0 = time.time()
            r10, r100 = evaluate(make())
            results.append(dict(method=name, d=d, seed=seed, r10=r10, r100=r100))
            print(f"d={d:3d} seed={seed} {name:11s} R10@10={r10:.3f} R10@100={r100:.3f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(results, open(args.out, "w"), indent=1)
