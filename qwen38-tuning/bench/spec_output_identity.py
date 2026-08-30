"""Does speculation change what the model says?

THE REASON THIS EXISTS. This project's metric is verified accepted coding tasks
per hour, not tok/s. A decoder that is 23 % faster and changes one token in a
diff has not made the worker faster; it has made it faster at producing
something else. Every speculation number in docs/reports/ is a throughput
number, and throughput is only worth quoting if the output is the same.

Speculative decoding in llama.cpp is verification-based: the target model checks
each drafted token and rejects what it would not have produced, so the output is
expected to be identical to no speculation. The DFlash2 PR claims the same, "up
to 4.6x the speed of autoregressive decoding, with the same output".

Expected is not measured. This measures it.

WHY GREEDY. temperature 0, top_k 1, fixed seed. Under sampling, two runs of the
SAME arm differ, so a difference between arms would prove nothing. Greedy makes
any difference attributable to the decoder -- and if a mismatch appears, that is
a real finding about the arm, not about the sampler.

WHAT A MISMATCH WOULD MEAN. Not automatically a bug in DFlash2: an accepted
draft token followed by a different numerical path can diverge on floating point
alone. It would mean the throughput comparison needs a quality column beside it
before anyone acts on it, which is the point.
"""
import argparse
import difflib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dflash2_arena as A

PROMPTS = [
    "// Write a Python function that merges two sorted lists.\n",
    "// Explain what this does, then rewrite it more simply:\n"
    "def f(x):\n    return [i for i in range(x) if all(i%j for j in range(2,i))]\n",
    "// Fix the off-by-one:\nfor (int i = 0; i <= n; ++i) sum += a[i];\n",
]


def generate(prompt, n_predict):
    r = A.post("/completion",
               {"prompt": prompt, "n_predict": n_predict, "cache_prompt": False,
                "temperature": 0.0, "top_k": 1, "seed": 42},
               timeout=900)
    return r["content"]


def collect(ctx, label, extra, n_predict):
    p, fh, log, _ = A.start(ctx, extra, "identity-" + label.replace("+", "-"))
    if p is None:
        raise RuntimeError("server failed to start for arm " + label)
    try:
        return [generate(pr, n_predict) for pr in PROMPTS]
    finally:
        A.kill()
        fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--n-predict", type=int, default=192)
    ap.add_argument("--out", default=str(A.ROOT / "results"
                                         / "spec-output-identity.jsonl"))
    a = ap.parse_args()

    arms = dict(A.ARMS)
    reference = collect(a.ctx, "none", arms["none"], a.n_predict)
    print("reference captured: %d prompts, %d chars"
          % (len(reference), sum(len(t) for t in reference)), flush=True)

    rows = []
    for label in ("ngram-mod", "dflash2", "dflash2+ngram"):
        got = collect(a.ctx, label, arms[label], a.n_predict)
        per_prompt = []
        for i, (ref, cand) in enumerate(zip(reference, got)):
            same = ref == cand
            entry = {"prompt": i, "identical": same,
                     "ref_chars": len(ref), "cand_chars": len(cand)}
            if not same:
                # Keep the actual divergence, not just a boolean. A claim that
                # two outputs differ is not usable without showing where.
                diff = list(difflib.unified_diff(
                    ref.splitlines(), cand.splitlines(),
                    lineterm="", n=1))[:40]
                entry["first_diff"] = diff
                for j, (x, y) in enumerate(zip(ref, cand)):
                    if x != y:
                        entry["diverges_at_char"] = j
                        break
            per_prompt.append(entry)
        row = {"arm": label, "ctx": a.ctx, "n_predict": a.n_predict,
               "all_identical": all(e["identical"] for e in per_prompt),
               "prompts": per_prompt}
        rows.append(row)
        print("  %-15s identical: %s"
              % (label, "yes" if row["all_identical"]
                 else [e["prompt"] for e in per_prompt if not e["identical"]]),
              flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("\nwrote %d rows to %s" % (len(rows), out))

    bad = [r["arm"] for r in rows if not r["all_identical"]]
    if bad:
        print("\nOUTPUT DIFFERS for: %s" % ", ".join(bad))
        print("Throughput for those arms cannot be quoted without a quality "
              "column beside it.")
        return 1
    print("\nEvery arm reproduced the no-speculation output byte for byte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
