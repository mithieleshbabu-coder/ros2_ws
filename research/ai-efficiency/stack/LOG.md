# Experiment log

Newest first. Each entry: what was run, the result, what surprised us, what it changes.
Raw numbers live in `results/ledger.jsonl`.

## 007 — mixed-domain calibration — `results/sweep_007_calibmix.log`

Same stack; calibration = 8 WikiText-2 train + 8 Python stdlib sequences (files disjoint from the
code eval set) instead of 16 WikiText-2. Wiki-only → mixed:

| weights | wiki | code |
|---|---|---|
| rot@4/128 (control, no weight calibration) | 18.72 → 18.72 | 5.43 → 5.40 |
| gptq@4/128 | 18.36 → 18.83 | 5.93 → 5.31 |
| **rgptq@4/128** | 17.30 → **17.33** | 5.29 → **5.00** |
| gptq@3/128 | 34.05 → 40.77 | 29.02 → 10.37 |
| rgptq@3/128 | 21.99 → 24.58 | 9.93 → 6.24 |

- Calibration domain decides where GPTQ puts its error: plain GPTQ trades Wikipedia for code
  almost one-for-one. With rotation the trade nearly disappears at 4 bits: mixed calibration fixes
  code (5.29 → 5.00) at no Wikipedia cost.
- The key coder's calibration change has no measurable effect (control row).
- **Default from here: `--calib mix`.** Any calibrated trick needs the evaluation domains in its
  calibration set, or a second-domain test to catch the damage.
- Next: predict-then-verify search over weights × head × keys × values (`search.py`, 008).

## 006 — GPTQ weights — `results/sweep_006_gptq.log`

Head rtn@8/128 and KV stack fixed. Calibration: 16 × 512 WikiText-2 train tokens.

| weights | wiki | code | 4k b1 | 32k b8 |
|---|---|---|---|---|
| rot@4/128 | 18.72 | 5.43 | ×3.04 | ×4.16 |
| gptq@4/128 | 18.36 | **5.93** | ×3.04 | ×4.16 |
| **rgptq@4/128** | **17.30** | **5.29** | ×3.04 | ×4.16 |
| rot@3/128 | 43.89 | 11.80 | ×3.50 | ×4.35 |
| gptq@3/128 | 34.05 | **29.02** | ×3.50 | ×4.35 |
| rgptq@3/128 | 21.99 | 9.93 | ×3.50 | ×4.35 |

- **Rotation + GPTQ stack**: 4-bit rgptq matches 5-bit RTN-quality (17.30 vs 17.24 wiki) for ×3.04
  instead of ×2.69.
- **GPTQ alone overfits the calibration domain.** Calibrated on Wikipedia, it improves Wikipedia
  but makes code *worse* than plain rotation at 4 bits (5.93 vs 5.43), and at 3 bits code
  collapses (29.0) while Wikipedia doesn't. Rotation removes most of this. Direct evidence for
  assumption 2 ("calibrate once"): error feedback fitted on one domain moves error onto others.
  The calibration set is small (8k tokens), so part of this is the known need for more and more
  diverse calibration data; the next test mixes domains.
- 3-bit weights are still not usable on a 0.5B model with any method here.
- Process failure: the first launch crashed on a hook bug and a watcher waited 2 h on its own
  process name. New rule: smoke-test every new trick (8-bit ≈ fp) before a sweep.

### Best frontier so far (Qwen2.5-0.5B, all with KV stack keys3/values3/sink/recent32)

| config | wiki | code | ceiling 4k b1 | ceiling 32k b8 |
|---|---|---|---|---|
| fp16 | 16.14 | 4.77 | ×1.00 | ×1.00 |
| w rtn8 + head8 | 16.39 | 4.81 | ×2.00 | ×3.53 |
| w rot5 + head8 | 16.88 | 5.00 | ×2.69 | ×3.98 |
| w rgptq4 + head8 | 17.30 | 5.29 | ×3.04 | ×4.16 |

## 005 — rotated (incoherent) weight quantization — `results/sweep_005_rot.log`

Head rtn@8/128 and KV stack fixed.

| weights | wiki | code | 4k b1 | 32k b8 |
|---|---|---|---|---|
| rtn@5/128 | 17.24 | 5.07 | ×2.69 | ×3.98 |
| rot@8/128 | 16.38 | 4.82 | ×2.00 | ×3.53 |
| rot@5/128 | **16.88** | **5.00** | ×2.69 | ×3.98 |
| rot@4/128 | 18.72 | 5.43 | ×3.04 | ×4.16 |
| rot@3/128 | 43.89 | 11.80 | ×3.50 | ×4.35 |

- Rotation helps at 5 bits (17.24 → 16.88 for the same bytes). The plain-RTN 4/128 run had an fp16
  head, so it isn't directly comparable to rot@4; GPTQ (006) gives the direct comparison.
- 3 bits still collapses even with rotation; that's what GPTQ's error feedback is for.
- Added `gptq` / `rgptq` weight kinds (unit test: half of RTN's output error at 3/4/8 bits).

## 004 — output-head quantization — `results/sweep_003_head.log`

Same KV stack. Best configurations so far (wiki / code ppl, bandwidth ceiling 4k b1 / 32k b8):

| weights | head | wiki | code | 4k b1 | 32k b8 |
|---|---|---|---|---|---|
| rtn@8/128 | fp16 | 16.40 | 4.81 | ×1.59 | ×3.18 |
| rtn@8/128 | rtn@8/128 | 16.39 | 4.81 | **×2.00** | ×3.53 |
| rtn@8/128 | rtn@4/128 | 17.66 | 5.03 | ×2.30 | ×3.75 |
| rtn@5/128 | rtn@8/128 | 17.24 | 5.07 | **×2.69** | ×3.98 |
| rtn@5/128 | rtn@4/128 | 18.53 | 5.30 | ×3.26 | ×4.25 |
| rtn@4/128 | fp16 | 19.65 | 5.76 | ×2.19 | ×3.68 |

- An 8-bit head is free and takes the 8-bit-weight stack from ×1.59 to ×2.00.
- `w5 + head8` is better on both axes than `w4 + fp16 head`: the cheapest bytes were in the part
  nobody quantized, not in pushing the decoder layers lower. A small "stacking" result:
  the right allocation beats more aggressive compression.
- 4-bit head costs ~0.075 nats; the head is more sensitive than decoder weights at equal bits.
- Next: rotated (incoherent) weight quantization, the standard fix for RTN outliers (005, running).

## 003 — weight-precision sweep, KV stack fixed — `results/sweep_002_weights.log`

KV stack fixed at `keys=qa-wf-gs@3 values=tok@3/64 sink=1 recent=32` (KV only: wiki 16.367).

| weights | wiki | code | ceiling 4k b1 | ceiling 32k b8 |
|---|---|---|---|---|
| fp16 | 16.37 | 4.81 | ×1.04 | ×2.52 |
| rtn@8/128 | 16.40 | 4.81 | ×1.59 | ×3.18 |
| rtn@6/128 | 16.58 | 4.87 | ×1.84 | ×3.41 |
| rtn@5/128 | 17.24 | 5.07 | ×2.00 | ×3.54 |
| rtn@4/32 | 18.05 | 5.29 | ×2.05 | ×3.57 |
| rtn@4/128 | 19.65 | 5.76 | ×2.19 | ×3.68 |
| rtn@3/32 | 30.01 | 8.26 | ×2.25 | ×3.71 |

- 8-bit RTN is free; quality falls off fast below 5 bits; 3-bit RTN collapses.
- **Surprise:** even 8-bit weights only reach ×1.59 at 4k b1. The output projection (vocab 151,936 ×
  896 ≈ 136M params, ~27% of the model, tied with the input embedding) stays fp16 and is read in full
  every decode step. For small models with big vocabularies the output head is a hidden bottleneck,
  and quantizing the decoder layers harder buys little: 4 → 3 bits gains only ×2.19 → ×2.25.
- **Changes:** added a `head` slot (output-projection quantization) and fixed the byte accounting so
  the head and the input-embedding lookup are counted separately (equivalent for tied models, so old
  entries stay comparable). Test 004 is running.

## 002 — first stack, leave-one-out ablation (Qwen2.5-0.5B) — `results/ablation_001.log`

Stack `weights=rtn@4/128 keys=qa-wf-gs@3 values=tok@3/64 sink=1 recent=32`:
wiki 19.65 / code 5.76 (fp: 16.14 / 4.77); bandwidth ceiling ×2.19 (4k, b1), ×3.68 (32k, b8).

| trick | quality cost (paired Δ NLL, nats/token, wiki / code) | what it buys |
|---|---|---|
| weights rtn@4/128 | **+0.183 ±0.014 / +0.179 ±0.013** | ×1.04 → ×2.19 at 4k b1 |
| keys qa-wf-gs@3 | +0.011 ±0.004 / +0.010 ±0.003 | ×1.73 → ×3.68 at 32k b8 (with values) |
| values tok@3/64 | +0.003 ±0.003 / +0.002 ±0.002 | (same) |
| sink=1 | **−0.184 / −0.145** (removing it hurts) | ~0 bytes |
| recent=32 | −0.067 / −0.075 | ~0 bytes |

- **The weights are the whole problem.** Plain 4-bit RTN costs ~17× more quality than 3-bit keys
  and values together. KV compression at ~3.3 bits is nearly free here (+0.014 nats) once sink and
  recent are kept, and it doubles the ceiling at 32k × 8.
- **The costs add up.** Total full-stack loss is ln(19.647/16.135) = 0.197 nats; the sum of the
  three compression costs is 0.183 + 0.011 + 0.003 = 0.197. In this regime the tricks don't
  interact, at least to first order. Worth testing where this breaks (lower bits, longer context).
- **Free wins are the largest effects.** Sink and recent cost no bytes and recover 0.25 nats
  together, more than any compression trick costs.
- **Changes:** the next lever is weight quantization (RTN is the weakest method there is). Next:
  sweep weight precision with the KV stack fixed, then replace RTN with a stronger quantizer.

## 001 — system validation (Qwen2.5-0.5B)

- fp baseline through `stack_attention`: wiki 16.135, code 4.767, identical to the stock model.
- `keys=qa-wf@2 sink=1` (32 calibration sequences): wiki 17.879, code 5.661, against 17.948 / 5.676
  in `kv-transform-coding`. The difference was unseeded calibration sampling; runs are now seeded.
- **Surprise:** keys-only compression gives a ceiling of just ×1.02 at 4k context, batch 1, because
  the ~1 GB of weights dominates traffic on a small GQA model (2 KV heads). The KV cache only matters
  at long context × batch (×1.49 at 32k, batch 8). Single-trick KV papers report KV memory
  reduction, not end-to-end bytes, which hides this.
- **Changes:** tricks have to be judged on end-to-end bytes per decode step at a stated serving
  point. That's the reason for the stack.
