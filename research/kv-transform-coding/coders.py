"""Key coders shared by the benchmarks. Each method is make(b) -> fit(K_train, Q_train) -> enc(K) -> K_hat."""
import math
import numpy as np
import torch


# ---------- Lloyd-Max codebooks for N(0,1) ----------
def lloyd_max(b, n=2_000_000, iters=200):
    x = np.sort(np.random.RandomState(0).randn(n))
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


def turbo_mse(b, center=False):
    """center: subtract the calibration key mean first. Free for attention: q.mu is the same
    for every key, so it shifts all scores of a query equally and softmax ignores it."""
    def fit(Ktr, Qtr):
        D = Ktr.shape[1]
        R = rand_orth(D, 1)
        mu = Ktr.mean(0) if center else torch.zeros(D)
        def enc(K):
            Kc = K - mu
            n = Kc.norm(dim=1, keepdim=True)
            z = (Kc / n) @ R * math.sqrt(D)  # coords ~ N(0,1)
            return (quant(z, b) / math.sqrt(D)) @ R.T * n + mu
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


def transform_code(b, query_aware, waterfill, gain_shape=False, extra_bits=0):
    """extra_bits: added to the b*D budget (e.g. 16, to match coders that store a norm)."""
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
        bits = allocate(lam.numpy(), b * D + extra_bits) if waterfill else np.full(D, b)
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


def weighted_basis_code(b, basis, extra_bits=0):
    """Orthogonal basis + query-weighted water-filling, with the basis chosen as in prior work:
      basis="k": PCA of the keys (AATC-style, arXiv 2608.14191)
      basis="q": eigenbasis of C_q = E[q q^T] (OSCAR-style rotation, arXiv 2605.17757)
    Coordinate c (direction v_c) has key variance s_c = v_c^T C_k v_c and query weight
    w_c = v_c^T C_q v_c; bits are allocated on w_c * s_c (diagonal approximation of q.k error).
    Compare transform_code(query_aware=True), whose basis is the eigenbasis of
    C_q^{1/2} C_k C_q^{1/2}: there the weighted error is exactly diagonal."""
    def fit(Ktr, Qtr):
        D = Ktr.shape[1]
        mu = Ktr.mean(0)
        Ck = torch.cov((Ktr - mu).T)
        Cq = Qtr.T @ Qtr / len(Qtr)
        V = torch.linalg.eigh(Ck if basis == "k" else Cq)[1]
        s = torch.einsum("dc,de,ec->c", V, Ck, V).clamp_min(1e-12)
        w = torch.einsum("dc,de,ec->c", V, Cq, V)
        bits = allocate((w * s).numpy(), b * D + extra_bits)
        sd = s.sqrt()
        def enc(K):
            z = ((K - mu) @ V) / sd
            zq = torch.stack([quant(z[:, i], int(bits[i])) for i in range(D)], 1)
            return (zq * sd) @ V.T + mu
        enc.bits = bits
        return enc
    return fit
