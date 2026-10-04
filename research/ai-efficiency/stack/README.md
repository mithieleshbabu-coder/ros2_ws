# stack — many small tricks, measured together

The goal is to find out which combination of efficiency tricks gives the most quality per byte
moved, and to learn how tricks interact, instead of testing each one alone. See
`../LANDSCAPE.md` for what is already proven and which assumptions this is built to test.

## How it works

```
spec string ──► Stack (tricks.py) ──► model with tricks applied ──► fixed benchmark ──► ledger
"weights=rtn@4/128 keys=qa-wf-gs@3 values=tok@3/64 sink=1 recent=32"
```

- **Tricks** are slots in a spec string (`tricks.py` lists the syntax). Adding a trick means
  adding a slot, its storage cost, and where it acts: on weights at load time, or inside
  `stack_attention` in `run.py`.
- **The benchmark is fixed.** Perplexity on WikiText-2 test (in-domain) and Python source
  (domain shift), 32 × 512 tokens each, per-sequence NLL stored. Memory traffic per decode
  step is computed exactly from the storage cost of each trick at two serving points
  (4k context, batch 1; 32k context, batch 8). `ceiling_speedup` is the fp16 / stack ratio,
  i.e. the decode speedup if the kernels were perfectly bandwidth-bound. Never change the
  benchmark to make a result look better; add a new benchmark next to it.
- **Every run is appended to `results/ledger.jsonl`** with git commit, spec, notes,
  perplexities with 95% CI, per-sequence NLLs and bytes. Nothing is overwritten.
- **Ablations** (`--ablate`) rerun the stack with each trick removed and report what each
  trick costs in quality as a *paired* NLL difference with a 95% CI, which is far more
  sensitive than comparing two perplexities.

## Usage

```bash
python run.py --spec ""                                    # fp baseline
python run.py --spec "keys=qa-wf@2 sink=1"                 # one stack
python run.py --spec "weights=rtn@4/128 keys=qa-wf-gs@3 values=tok@3/64 sink=1" --ablate
python run.py --model Qwen/Qwen2.5-1.5B --spec "..."       # other model
```

## Rules of the system

1. Validate before trusting: the fp baseline through `stack_attention` must reproduce the stock
   model (it does: 16.135 / 4.767 on Qwen2.5-0.5B), and every new trick gets a sanity run that
   converges to fp at high precision.
2. Compare paired, with confidence intervals; a difference inside the CI is not a result.
3. Log failures as carefully as wins. `LOG.md` is the human-readable record of what was tried,
   what surprised us, and why.
4. Literature search before building any new trick (`../LANDSCAPE.md`).
5. Quality is measured; speed is modeled. Real speedups need GPU kernels, so the ledger reports a
   bandwidth ceiling, never a claimed speedup.

## Current tricks

| slot | options | storage |
|---|---|---|
| keys | `kivi`, `turbo-c`, `klt-wf`, `qa-wf`, `qa-wf-gs`, `kbasis-qw`, `qbasis-qw` @ bits | b + 16/d_head bits (kivi: b + 0.25) |
| values | `tok@b/g`: per-token asymmetric uniform over groups of g channels | b + 32/g bits |
| weights | `rtn@b/g`: group-wise round-to-nearest, all decoder Linear layers | b + 32/g bits |
| sink | first n tokens in fp16 | 16 bits |
| recent | last r tokens in fp16 (KIVI residual / OSCAR recent window) | 16 bits |
| window | StreamingLLM eviction: keep sink + last w tokens | evicted tokens cost 0 |

## Known limits

- Quality is measured at 512-token context; `window` savings are modeled at 4k/32k, where
  quality is untested.
- Weight quantization is plain RTN; GPTQ/AWQ-class methods would do better at the same bits.
- Two small models (Qwen2.5-0.5B/1.5B); perplexity only.
