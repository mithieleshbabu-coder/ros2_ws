# AI efficiency: problem breakdown and literature map (Oct 2026)

Purpose: know what is already proven, so we only spend experiments on what isn't.
Numbers marked *reported* come from the cited paper/vendor and were not reproduced here.

---

## 1. Where the time and money actually go

**Two phases with different physics.**

| phase | what happens | bound by | user-visible metric |
|---|---|---|---|
| prefill | whole prompt processed in parallel | compute (FLOPs) | time to first token (TTFT) |
| decode | one token at a time; every step re-reads all weights + the whole KV cache | **memory bandwidth** | time between tokens (TBT), tokens/s |

- Decode leaves tensor cores mostly idle; production systems run at ~30–50% model-FLOPs utilization ([Reddit/mlscaling](https://www.reddit.com/r/mlscaling/comments/1uftsr3/the_real_llm_inference_bottleneck_isnt_compute/), [Google Cloud](https://cloud.google.com/blog/topics/developers-practitioners/five-techniques-to-reach-the-efficient-frontier-of-llm-inference)).
- **The gap is widening.** Peak FLOPs grew ~3.0×/2 yr vs DRAM bandwidth ~1.6×/2 yr; A100→B200: ~7× BF16 compute vs ~4× HBM bandwidth. Break-even arithmetic intensity rises ~23× per decade ([arXiv 2608.28048](https://arxiv.org/pdf/2608.28048), [AI and Memory Wall](https://www.alphaxiv.org/overview/2403.14123)). HBM is supply-constrained through 2026.
- **Every byte not moved per decoded token is speed.** Decode cost per token ≈ (weight bytes + batch × context × KV bytes/token) / bandwidth.

**Workload shifts that make it worse (2025–26):**
- *Long context*: KV cache grows linearly with context and can exceed the weights. In vision-language models, visual tokens are 80–90% of memory ([arXiv 2604.05546](https://arxiv.org/html/2604.05546v1)).
- *Reasoning models*: thinking tokens are billed output tokens and are often wasted. Accuracy is an inverted U in chain-of-thought length; models "overthink" easy problems (900+ tokens for "2+3") ([arXiv 2604.10739](https://arxiv.org/html/2604.10739v1), [Redis](https://redis.io/blog/token-budget-aware-llm-reasoning/)).
- *Agents*: 9.2× more LLM calls per request than plain chain-of-thought; context grows 33–44× within one task; prefix-cache hit rate ranges from 99% to under 1% depending on how the agent edits its context ([arXiv 2608.15127](https://arxiv.org/pdf/2608.15127), [SemiAnalysis](https://inferencex.semianalysis.com/blog/brief-overview-of-agentic-workloads)).

**Price trend.** GPT-4-class tokens went from ~$20/M (late 2022) to ~$0.40/M (2026), reported as roughly 1000× in three years. Drivers: hardware (2–3× per generation), serving software (GPU utilization 30–40% → 70–80% via continuous batching and paged attention), smaller/distilled models, and competition ([arXiv 2603.28576](https://arxiv.org/html/2603.28576v1), [GPUnex](https://gpunex.com/blog/ai-inference-economics-2026)).

---

## 2. The stack, layer by layer: what is already proven

Maturity: **standard** = shipped in mainstream engines (vLLM/SGLang/TensorRT-LLM/llama.cpp); **proven** = strong published results, partial adoption; **research** = recent, unsettled.

### L0. What gets computed at all (biggest lever, cheapest to apply)

| technique | reported result | maturity |
|---|---|---|
| Routing / cascades (cheap model first, escalate) | RouteLLM: 85% cost cut on MT-Bench, 45% MMLU, 35% GSM8K at 95% of GPT-4 quality ([LMSYS](https://lmsys.org/blog/2024-07-01-routellm/)) | proven |
| Distillation into small task models | 7B models at ~1–2% of frontier cost per token for most production tasks ([tianpan](https://tianpan.co/blog/2026-04-10-knowledge-distillation-economics-production-ai)) | standard |
| Reasoning-length control | Chain of Draft: 7.6% of tokens on some tasks, ~80% saving on GSM8K for −4 pts; TALE: −67% tokens, <3% drop; CoT pruning >50% shorter at same accuracy ([Redis](https://redis.io/blog/token-budget-aware-llm-reasoning/), [OpenReview](https://openreview.net/forum?id=8xSU8Oscvg)) | proven |
| Prefix caching / KV reuse across requests | hit rate 0% → 90% ≈ 10× GPU bill cut on agent workloads; stateful serving 2.1–4.2× on multi-turn ([SemiAnalysis](https://inferencex.semianalysis.com/blog/brief-overview-of-agentic-workloads), [ContextPilot](https://proceedings.mlsys.org/paper_files/paper/2026/hash/b0131b6ee02a00b03fc3320176fec8f5-Abstract-Conference.html), [LMCache](https://github.com/lmcache/lmcache)) | standard |

### L1. Architecture (needs (re)training; decided by model builders)

| technique | reported result | maturity |
|---|---|---|
| GQA (shared KV heads) | standard in nearly all 2024–26 models | standard |
| MLA (latent KV, DeepSeek) | TransMLA converts GQA→MLA: 93% KV reduction, 10.6× speedup at 8K on LLaMA-2-7B, ~6B tokens of fine-tuning ([arXiv 2502.07864](https://arxiv.org/html/2502.07864v5)) | proven |
| Hybrid linear attention (Gated DeltaNet / Mamba-2 + few full-attention layers, ~3:1) | Qwen3-Next/3.5, Kimi Linear; HyLo upcycling: −90% KV, 2M-token context ([arXiv 2604.24715](https://arxiv.org/html/2604.24715v1), [Raschka](https://sebastianraschka.com/llms-from-scratch/ch04/08_deltanet/)) | standard (new models) |
| Trainable sparse attention (NSA, DeepSeek DSA) | matches full attention, large speedups at 64K ([arXiv 2502.11089](https://arxiv.org/abs/2502.11089)) | proven |
| Cross-layer KV sharing (YOCO, FusedKV) | −50% KV with equal or better perplexity, 332M–4B ([FusedKV, ICLR 2026](https://arxiv.org/html/2512.03870v1)) | proven |
| MoE | standard for frontier open models; offloading on consumer GPUs: expert loading can be >80% of latency, MoBiLE 1.6–1.7× ([arXiv 2510.12357](https://arxiv.org/pdf/2510.12357), [arXiv 2512.16473](https://arxiv.org/pdf/2512.16473)) | standard / research |
| Gated attention / softpick (removes attention sinks and massive activations) | fewer outliers, better quantization ([arXiv 2601.22966](https://arxiv.org/html/2601.22966v1), [Softpick](https://arxiv.org/html/2504.20966v3)) | proven (in Qwen3.5) |
| Byte-level patches (BLT) | better scaling at fixed inference cost; BLT-D: up to −92% memory bandwidth ([arXiv 2605.08044](https://arxiv.org/pdf/2605.08044)) | research |

### L2. Numerics (post-training, applies to existing models)

| target | technique | reported result | maturity |
|---|---|---|---|
| weights 4-bit | GPTQ, AWQ, NVFP4/MXFP4 | near-lossless; NVIDIA reports 4-bit at near-FP8 quality ([NVIDIA](https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/)) | standard |
| weights 2–3-bit | QuIP#, AQLM, QTIP, D2Quant | usable but lossy; D2Quant 2-bit Qwen3-8B: 57.2 vs 54.1 prior SOTA avg zero-shot ([arXiv 2602.02546](https://arxiv.org/html/2602.02546v1)) | research |
| weights + activations 4-bit | QuaRot, SpinQuant, FlatQuant | FlatQuant W4A4 <1% drop on LLaMA-3-70B; 2.3× prefill / 1.7× decode ([arXiv 2410.09426](https://arxiv.org/html/2410.09426v2)) | proven |
| KV cache | KIVI, KVQuant, TurboQuant, OSCAR, KVTC, AATC | TurboQuant quality-neutral at 3.5 bits; OSCAR ~2.28 bits, −3.8 pts vs BF16 where TurboQuant collapses on Qwen3-4B reasoning; KVTC: PCA + DP bit allocation + entropy coding, evaluated at 8–64× compression ([TurboQuant](https://openreview.net/forum?id=tO3ASKZlok), [OSCAR](https://arxiv.org/abs/2605.17757), [KVTC](https://proceedings.iclr.cc/paper_files/paper/2026/file/3fb6f10bd2784f6cfb6a6ed6280df40c-Paper-Conference.pdf), [AATC](https://arxiv.org/pdf/2608.14191)) | proven; our Experiment 3 reproduces the direction |
| training precision | FP8 (DeepSeek-V3), NVFP4 | NVFP4 12B on 10T tokens tracks FP8 (<1.5% loss gap); 1.59× vs 1.33× for FP8 ([arXiv 2509.25149](https://arxiv.org/html/2509.25149v2), [NVIDIA](https://developer.nvidia.com/blog/using-nvfp4-low-precision-model-training-for-higher-throughput-without-losing-accuracy/)) | proven |
| pruning | Wanda/SparseGPT 50%, 2:4 | 2:4 → ~1.6× on linear layers but only ~1.24× end-to-end ([Wanda](https://arxiv.org/pdf/2306.11695), [D2Prune](https://arxiv.org/pdf/2601.09176)) | proven, modest gains |

### L3. Which tokens and how much state are kept

| technique | reported result | maturity |
|---|---|---|
| KV eviction (StreamingLLM, H2O, SnapKV, PyramidKV) → learned (TRIM-KV, LookaheadKV) | LookaheadKV beats SnapKV/PyramidKV across budgets 64–2048 with 14.5× less eviction overhead ([Samsung](https://research.samsung.com/blog/LookaheadKV-Fast-and-Accurate-KV-Cache-Eviction-by-Glimpsing-into-the-Future-without-Generation), [TRIM-KV](https://openreview.net/forum?id=qCaq3jGb0S)) | proven |
| Eviction + quantization combined | complementary; ThinKV and CAKE+KIVI hold quality at ~6% of cache ([ThinKV](https://arxiv.org/pdf/2510.01290), [CAKE](https://arxiv.org/pdf/2503.12491)) | research |
| Visual token pruning (VLMs) | LearnPruner: ~95% quality at 5.5% of visual tokens, 3.2×; TopV 2.1× ([arXiv 2604.23950](https://www.alphaxiv.org/abs/2604.23950.md)) | proven |
| Tiered KV storage (GPU→CPU→SSD→remote) | LMCache, Mooncake | standard |

### L4. Decoding algorithm

| technique | reported result | maturity |
|---|---|---|
| Speculative decoding, EAGLE-3 | 1.57–1.9× on 70B / Kimi K2.5; P-EAGLE (parallel drafting) +1.69× over EAGLE-3 at low concurrency, gains shrink at high batch ([vLLM](https://vllm.ai/blog/2026-03-13-p-eagle), [AMD](https://rocm.blogs.amd.com/artificial-intelligence/eagle3-speculative-decoding/README.html), [Jarvislabs](https://jarvislabs.ai/blog/speculative-decoding-vllm-faster-llm-inference)) | standard |
| Suffix decoding (no draft model) | 1.33× on 70B, wins on some agentic/code workloads | standard |
| Self-speculation / early exit (LayerSkip, DSSD) | up to 2.33× ([ACL Findings 2026](https://aclanthology.org/2026.findings-acl.802.pdf), [LayerSkip](https://arxiv.org/abs/2404.16710)) | proven |
| Diffusion / multi-token LMs | 5–10× throughput, still behind on broad reasoning ([Nemotron-Labs-Diffusion](https://research.nvidia.com/publication/2026-05_nemotron-labs-diffusion-tri-mode-language-model-unifying-autoregressive.md), [D2F](https://iclr.cc/virtual/2026/poster/10007008)) | research |

### L5. Kernels

| technique | reported result | maturity |
|---|---|---|
| FlashAttention 2/3 | standard | standard |
| Megakernels (whole forward pass in one persistent kernel) | MPK: up to 1.7× lower latency; Hazy Research: ~2.5× vs vLLM on H100 for small models; kernel-launch overhead ~14.6% of decode time ([MPK, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/cheng), [mirage](https://github.com/mirage-project/mirage)) | proven |

### L6. Serving systems

| technique | reported result | maturity |
|---|---|---|
| Continuous batching + paged attention | core of the 30–40% → 70–80% utilization jump | standard |
| Prefill/decode disaggregation (DistServe, Mooncake) | goodput gains; standard at large providers ([DistServe](https://arxiv.org/abs/2401.09670), [Hao AI Lab retro](https://haoailab.com/blogs/distserve-retro/)) | standard |

### L7. Training

| technique | reported result | maturity |
|---|---|---|
| Muon optimizer | ~2× compute efficiency vs AdamW; distributed versions near-Adam overhead ([DMuon](https://arxiv.org/html/2606.27153v1)) | proven |
| FP8 / NVFP4 training | see L2 | proven |

### Adjacent: vector search

| technique | reported result | maturity |
|---|---|---|
| RaBitQ / Extended RaBitQ | theoretical error bound, no codebook training; IVF-RaBitQ on cuVS 3× QPS vs graph methods at 95% recall ([LanceDB](https://www.lancedb.com/blog/feature-rabitq-quantization), [arXiv 2602.23999](https://arxiv.org/html/2602.23999v1)) | standard |

### Adjacent: edge robotics (VLA models)

- LiteVLA-Edge: 4-bit GGUF + llama.cpp on Jetson Orin, ~150 ms / 6.6 Hz end to end ([arXiv 2603.03380](https://arxiv.org/html/2603.03380)); Jetson-PI, GigaBrain-0-Small: same goal ([arXiv 2607.12659](https://www.alphaxiv.org/abs/2607.12659.md), [arXiv 2510.19430](https://arxiv.org/pdf/2510.19430)). Mostly applies standard LLM tricks; robot-specific compression is thin.

---

## 3. What we do not need to prove again

Treat as given and build on top: decode is bandwidth-bound; 4-bit weights are near-lossless; KV cache quantizes well to 3–4 bits with outlier handling; rotation plus bit allocation by variance beats uniform; query-aware weighting helps; attention sinks and massive activations cause most quantization failures; eviction and quantization are complementary; speculative decoding gives 1.5–2× at low batch and less at high batch; prefix caching dominates agent cost; routing and distillation are the biggest single cost levers.

---

## 4. Assumptions worth challenging

Each is quietly assumed by much of the work above. *Evidence* marks what we have seen ourselves.

1. **Tricks compose independently.** Papers test one trick, or two at most (quantization + eviction). How compression interacts with speculative acceptance is explicitly called unstudied ([SpecKV](https://arxiv.org/pdf/2605.02888)). Real deployments stack 5–10 tricks, and whether their errors add, cancel or amplify is unknown.
2. **Calibrate once, offline.** Data-aware coders (OSCAR, AATC, KVTC, ours) fit on one dataset. *Evidence:* our coders held up from WikiText to code on 0.5B, but needed extra robustness (stored norm) on 1.5B.
3. **Outliers are a fixed, known set of tokens (the sink).** *Evidence:* on 1.5B, keeping only token 0 in full precision did not remove the error floor; a per-key norm did. There are more outlier tokens than the first one.
4. **Perplexity predicts downstream quality.** Known to break for long-context and reasoning tasks; errors accumulate over long generations (OSCAR's KL curves).
5. **Equal treatment of layers, heads and tokens.** Per-layer budgets help (KVTuner, PyramidKV), but joint allocation across tricks (bits vs tokens vs layers) is rarely optimized.
6. **Memory traffic is the only cost.** At high batch, decode becomes compute-bound again; tricks that add compute (rotations, dequantization, draft models) can lose. Gains reported at batch 1 often shrink at batch 64 (P-EAGLE over EAGLE-3: +55–69% throughput at concurrency 1, +5–25% at concurrency 64).
7. **A trick's best setting is fixed.** The best speculation length, bit width and eviction budget likely depend on each other and on the workload.

---

## 5. Opportunities, ranked for our constraints (CPU-only, small models, solo)

| # | opportunity | why it might be open | can we test on CPU? |
|---|---|---|---|
| A | **Composition science**: measure how stacked tricks interact (key quant × value quant × weight quant × eviction × sink handling), find the best combined allocation of a byte budget | assumption 1; papers rarely go beyond pairs | yes: quality is measured directly, bytes are computed analytically |
| B | **Compressed self-drafting**: the same model with 2-bit KV + 4-bit weights as its own speculative draft; acceptance rate vs compression level | the SpecKV gap; free draft without training | yes for acceptance rate; speedup needs a GPU |
| C | **Outlier-token anatomy**: which tokens and layers break calibrated coders beyond the sink, and the cheapest fix | assumption 3; our own evidence | yes |
| D | **Joint byte allocation across layers × tricks** (one global optimizer instead of per-trick settings) | assumption 5 | yes |
| E | Edge robotics / VLA compression | thin literature, venture interest | partly (needs robot data) |

**The system in `stack/` is built for A–D:** every trick is a plug-in, every run is measured with the same benchmark and logged, and leave-one-out ablations show which trick actually earns its bytes.
