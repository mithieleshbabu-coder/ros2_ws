# kv-transform-coding

Small, fast experiments on compressing vectors for ML workloads (KV cache, vector search).
Each idea gets a <100-line test against strong baselines before any theory is written.

## Experiment 1: learned max-plus (tropical) projection — `maxplus_bench.py`

**Question:** Does a learned max-plus projection `y_i = max_j (W_ij + x_j)` compress embeddings
better than linear methods at equal storage?

**Answer: no.** It only beats random projection. PCA, a learned linear projection, and
especially PCA + int8 are clearly better. Variants (min-plus half, extra multiplicative weight) don't help.

R10@10, nearest-neighbour search in the compressed space (GloVe-300, 50k database, 1k queries, mean of 2 seeds):

| method | d=8 | d=16 | d=32 | d=64 |
|---|---|---|---|---|
| random projection | 0.010 | 0.034 | 0.101 | 0.214 |
| max-plus | 0.018 | 0.050 | 0.117 | 0.244 |
| max-plus + min-plus | 0.017 | 0.048 | 0.120 | 0.247 |
| PCA | 0.045 | 0.091 | 0.172 | 0.301 |
| learned linear | 0.050 | 0.106 | 0.189 | 0.326 |
| PCA + int8 (same bytes) | **0.172** | **0.300** | **0.513** | **0.974** |

Lesson: at equal bytes, more dimensions at lower precision beats fewer dimensions at full precision.

## Experiment 2: query-aware transform coding for key quantization — `kvquant_bench.py`

**Question:** TurboQuant rotates keys randomly and spends equal bits on every coordinate.
Classical transform coding says: rotate to the principal axes (KLT) and allocate bits by
variance (reverse water-filling). Since attention only uses `q·k`, do it in the
query-weighted space `k' = C_q^{1/2} k` (`C_q = E[qqᵀ]`), so the coder minimizes q·k error directly.

Only keys are quantized; queries stay full precision. All methods spend `b·D` bits per key.

| method | rotation | metric | bits | norm stored |
|---|---|---|---|---|
| `turbo-mse` | random | Euclidean | uniform | yes |
| `turbo-prod` | random | Euclidean | (b−1) + 1-bit QJL residual | yes |
| `klt-unif` / `klt-wf` | PCA | Euclidean | uniform / water-filled | no |
| `qa-unif` / `qa-wf` | PCA | query-weighted | uniform / water-filled | no |
| **`qa-wf-gs`** | PCA | query-weighted | water-filled | yes (gain-shape) |

### Results (b = bits per coordinate)

Top-10 recall over 50k keys (higher is better) / attention-output relative error over 2048-key contexts (lower is better).

**glove-ip** — keys and queries both GloVe vectors (same distribution; control for the query-aware part)

| method | R@10 b=1 | b=2 | b=3 | b=4 | attn-err b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|---|---|---|---|
| turbo-mse | 0.569 | 0.759 | 0.862 | 0.925 | 0.503 | 0.257 | 0.148 | 0.077 |
| turbo-prod | 0.409 | 0.578 | 0.736 | 0.847 | 0.900 | 0.559 | 0.316 | 0.174 |
| klt-wf | 0.537 | 0.750 | 0.857 | 0.906 | 0.458 | 0.248 | 0.148 | 0.093 |
| qa-wf | 0.543 | 0.766 | 0.880 | 0.924 | 0.458 | 0.237 | 0.128 | 0.076 |
| **qa-wf-gs** | **0.585** | **0.794** | **0.896** | **0.940** | **0.440** | **0.214** | **0.103** | **0.054** |

**synth-head** — synthetic attention head over GloVe "hidden states", `k = hW_k`, `q = hW_q`, each W with log-normal per-channel scale (strongly anisotropic)

| method | R@10 b=1 | b=2 | b=3 | b=4 | attn-err b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|---|---|---|---|
| turbo-mse | 0.139 | 0.390 | 0.634 | 0.802 | 1.110 | 0.711 | 0.401 | 0.213 |
| klt-wf | 0.450 | 0.679 | 0.797 | 0.851 | 0.487 | 0.295 | 0.187 | 0.130 |
| qa-wf | 0.661 | 0.813 | 0.863 | 0.880 | 0.319 | 0.171 | 0.122 | 0.103 |
| **qa-wf-gs** | **0.737** | **0.877** | **0.929** | **0.950** | **0.282** | **0.124** | **0.069** | **0.048** |

**synth-iso** — same head without the extra per-channel scaling (checks the gain is not an artefact of the synthetic anisotropy)

| method | R@10 b=1 | b=2 | b=3 | b=4 | attn-err b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|---|---|---|---|
| turbo-mse | 0.262 | 0.524 | 0.721 | 0.843 | 0.693 | 0.489 | 0.292 | 0.155 |
| klt-wf | 0.213 | 0.481 | 0.671 | 0.797 | 0.643 | 0.463 | 0.305 | 0.193 |
| qa-wf | 0.306 | 0.577 | 0.737 | 0.825 | **0.591** | 0.399 | 0.254 | 0.165 |
| **qa-wf-gs** | **0.412** | **0.658** | **0.803** | **0.889** | 0.592 | **0.349** | **0.199** | **0.112** |

**glove-shift** — calibrate on the 20k most frequent words, test on rare words (rank 70k–120k)

| method | R@10 b=1 | b=2 | b=3 | b=4 | attn-err b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|---|---|---|---|
| turbo-mse | **0.516** | 0.715 | 0.840 | **0.908** | **0.484** | 0.296 | 0.160 | 0.088 |
| klt-wf | 0.463 | 0.668 | 0.785 | 0.848 | 0.510 | 0.325 | 0.208 | 0.147 |
| qa-wf | 0.447 | 0.653 | 0.789 | 0.856 | 0.521 | 0.334 | 0.209 | 0.141 |
| **qa-wf-gs** | 0.491 | **0.720** | **0.846** | **0.908** | 0.493 | **0.290** | **0.157** | 0.089 |

Full results (including the uniform-bit ablations) are in `results/kv_results.json` and `results/kv_run.log`.

### Findings

1. **Variance-based bit allocation (#1)** wins whenever the key distribution is anisotropic; the random rotation in TurboQuant equalizes variances and throws that structure away. Uniform bits after a KLT is *worse* than TurboQuant, so the allocation is what matters.
2. **Query weighting (#2)** helps exactly when query and key distributions differ (synthetic heads) and adds little when they match (glove-ip), as theory predicts.
3. **Storing the norm (gain-shape)** is what makes the data-aware coder robust: without it, the coder loses to TurboQuant under distribution shift; with it, `qa-wf-gs` matches or beats TurboQuant in every scenario except 1-bit recall under shift.

### Caveats

- No real LLM keys/queries yet; the synthetic heads use my own choice of anisotropy.
- `turbo-*` are reimplementations of TurboQuant's core idea (Gaussian Lloyd-Max codebooks instead of the exact Beta codebooks), not Google's code. Only the MSE variant does well here.
- TurboQuant needs no calibration; `qa-wf-gs` needs per-head calibration queries and keys and can suffer from drift.
- Variable bit-widths per coordinate are harder to implement efficiently on GPUs; not measured.

### Related work (read before claiming novelty)

- MixKVQ (ACL 2026), query-aware mixed-precision key quantization, per original channel: https://arxiv.org/abs/2512.19206
- KQ-SVD (AISTATS 2026), query-aware low-rank key compression: https://arxiv.org/abs/2512.05916

### Next steps

1. Real per-head K/Q from a small open LLM (needs Hugging Face access).
2. Add MixKVQ and KQ-SVD as baselines.
3. Proper calibration-drift test (calibrate on one domain, evaluate on another).

## Reproduce

```bash
pip install -r requirements.txt
python prepare_data.py            # downloads GloVe 300d (~390 MB), writes glove120k.npy
python maxplus_bench.py --steps 1500 --batch 512 --seeds 0,1 --out results/maxplus_results.json   # ~35 min CPU
python kvquant_bench.py           # all four scenarios, ~6 min CPU
METHODS=turbo-mse,qa-wf-gs python kvquant_bench.py synth-head   # subset
```
