# Experiment log

Newest first. Each entry: what was run, the result, what surprised us, what it changes.
Raw numbers live in `results/ledger.jsonl`.

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
