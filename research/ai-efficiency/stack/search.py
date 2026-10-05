"""
Predict-then-verify search over stack configurations.

Finding 002 showed per-trick quality costs add up (to first order). So instead of running every
combination, measure each option of each slot *alone* on top of a reference, predict every
combination as reference + sum of costs, compute its bytes exactly, keep the Pareto frontier
(best predicted quality for its bytes), and verify the frontier with real runs. The verification
also tests additivity at scale: predicted vs measured NLL is reported for every verified config.

    python search.py measure              # run any missing single-option measurements
    python search.py frontier             # predicted Pareto frontier for both serving points
    python search.py verify --top 6       # run the frontier configs and compare to prediction
"""
import argparse, itertools, json, math, os, subprocess, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from tricks import Stack
from bytes_model import model_sizes, step_bytes, SERVING_POINTS

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "results", "ledger.jsonl")

BASE = "sink=1 recent=32"  # free tricks, always on (finding 002)
SLOTS = {  # option strings; None = leave the slot at fp16
    "weights": [None, "rot@8/128", "rot@5/128", "rgptq@4/128", "rgptq@3/128"],
    "head": [None, "rtn@8/128", "rtn@6/128", "rtn@4/128"],
    "keys": [None, "qa-wf-gs@4", "qa-wf-gs@3", "qa-wf-gs@2"],
    "values": [None, "tok@4/64", "tok@3/64", "tok@2/64"],
}
ORDER = ["weights", "head", "keys", "values"]


def spec_of(choice):
    toks = [f"{s}={choice[s]}" for s in ORDER if choice.get(s)]
    return " ".join(toks + [BASE])


def ledger(model, calib, n_eval):
    rows = {}
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            r = json.loads(line)
            if (r["model"] == model and r.get("calib", "wiki") == calib and r["n_eval"] == n_eval):
                rows[r["spec"]] = r  # latest wins
    return rows


def run(spec, a):
    cmd = [sys.executable, "-u", os.path.join(HERE, "run.py"), "--spec", spec, "--model", a.model,
           "--calib", a.calib, "--n-eval", str(a.n_eval), "--note", "search"]
    subprocess.run(cmd, check=True, cwd=HERE, stderr=subprocess.DEVNULL)


def mean_nll(r):
    return {d: float(np.mean(r[d]["nll"])) for d in ("wiki", "code")}


def predict_all(rows):
    """Reference + sum of single-option costs, for every combination."""
    ref = rows[spec_of({})]
    ref_nll = mean_nll(ref)
    cost = {}
    for s in ORDER:
        for o in SLOTS[s]:
            if o is None:
                cost[s, o] = {"wiki": 0.0, "code": 0.0}
                continue
            r = rows.get(spec_of({s: o}))
            if r is None:
                return None
            m = mean_nll(r)
            cost[s, o] = {d: m[d] - ref_nll[d] for d in m}
    preds = []
    for combo in itertools.product(*[SLOTS[s] for s in ORDER]):
        choice = dict(zip(ORDER, combo))
        p = {d: ref_nll[d] + sum(cost[s, choice[s]][d] for s in ORDER) for d in ref_nll}
        preds.append((spec_of(choice), p))
    return preds


def frontier(preds, sz, point):
    C, B = SERVING_POINTS[point]
    base_bytes = step_bytes(Stack(), sz, C, B)
    pts = []
    for spec, p in preds:
        speed = base_bytes / step_bytes(Stack(spec), sz, C, B)
        pts.append((speed, (p["wiki"] + p["code"]) / 2, spec, p))
    pts.sort(key=lambda x: -x[0])  # fastest first; keep a point only if it beats every faster one
    out, best = [], math.inf
    for sp, q, spec, p in pts:
        if q < best - 1e-9:
            out.append((sp, spec, p))
            best = q
    return out[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["measure", "frontier", "verify"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--calib", default="mix")
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()

    needed = [spec_of({})] + [spec_of({s: o}) for s in ORDER for o in SLOTS[s] if o]
    if a.cmd == "measure":
        have = ledger(a.model, a.calib, a.n_eval)
        for spec in needed:
            if spec not in have:
                print("measuring", spec, flush=True)
                run(spec, a)
        return

    rows = ledger(a.model, a.calib, a.n_eval)
    preds = predict_all(rows)
    if preds is None:
        sys.exit("missing single-option measurements; run `search.py measure` first")
    sz = model_sizes(AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32))
    to_verify = []
    for point in SERVING_POINTS:
        fr = frontier(preds, sz, point)
        print(f"\npredicted frontier at {point} (mean ppl wiki/code from predicted NLL):")
        for sp, spec, p in fr:
            print(f"  x{sp:5.2f}  wiki {math.exp(p['wiki']):7.3f}  code {math.exp(p['code']):6.3f}  {spec}")
        step = max(1, len(fr) // a.top)
        to_verify += [spec for _, spec, _ in fr[::step]]
    if a.cmd == "verify":
        print("\nverifying (predicted vs measured mean NLL, nats/token):")
        pred = dict(preds)
        for spec in dict.fromkeys(to_verify):
            if spec not in rows:
                run(spec, a)
                rows = ledger(a.model, a.calib, a.n_eval)
            m = mean_nll(rows[spec])
            p = pred[spec]
            print(f"  wiki pred {p['wiki']:.4f} meas {m['wiki']:.4f} ({m['wiki'] - p['wiki']:+.4f})   "
                  f"code pred {p['code']:.4f} meas {m['code']:.4f} ({m['code'] - p['code']:+.4f})   {spec}",
                  flush=True)


if __name__ == "__main__":
    main()
