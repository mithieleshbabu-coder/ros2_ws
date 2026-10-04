# Experiment log

Newest first. Each entry: what was run, the result, what surprised us, what it changes.
Raw numbers live in `results/ledger.jsonl`.

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
