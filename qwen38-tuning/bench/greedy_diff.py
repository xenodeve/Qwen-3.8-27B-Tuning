"""Capture and compare the actual greedy output of several arms, not just its hash.

`model_arena` records a SHA-256 of a greedy continuation as an invariance check.
It has been quietly useful all project: Q4_K_XL, UD-Q2_K_XL and UD-IQ2_XXS all
returned `227749403A7404D4` on the same mechanical rewrite -- rename one
attribute, one correct answer -- which is a strong hint that quantization down to
8.39 GiB had not damaged this behaviour.

`AD-IQ1_M` returned `9A18CB695347E7E3`. That is the first divergence the probe
has recorded, and a hash cannot say whether the difference is a harmless
formatting choice or a wrong answer. This prints the text.

Also checks the rewrite mechanically -- the task is "rename `order` to `usage`
everywhere", so a correct answer contains no surviving `self.order` and the same
number of `usage` references as the original had `order` references. That turns
"different" into "different and wrong" or "different but right".

Usage:
    python greedy_diff.py --arms iq2xxs-nomtp,iq1m-nomtp
"""
import argparse
import difflib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import model_arena as A

ROOT = Path(r"C:\AI\qwen38-tuning")


def answer_only(text):
    """Strip reasoning and prose, leaving the code the model actually proposes.

    The first version of grade() counted `self.order` across the WHOLE reply and
    scored Ornith-9B as having failed the rename. It had not: its <think> block
    enumerates every `self.order` in the original before renaming them, so the
    grader was reading the model's working and calling it the answer. Exactly
    the class of bug this project keeps finding -- a plausible wrong number
    rather than a crash.
    """
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", body, flags=re.S)
    if blocks:
        return max(blocks, key=len)
    return body


def grade(text):
    """Mechanical check of the rename task. Not a judge -- a counter."""
    code = answer_only(text)
    stale = len(re.findall(r"self\.order\b", code))
    renamed = len(re.findall(r"self\.usage\b", code))
    # The prompt's class uses self.order 6 times.
    return {"stale_order_refs": stale, "usage_refs": renamed,
            "rename_complete": stale == 0 and renamed >= 5,
            "graded_chars": len(code)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="iq2xxs-nomtp,iq1m-nomtp")
    # 400 was too small and produced a wrong verdict: Ornith-9B spent most of it
    # on a <think> block and was cut off at "class LRUCache:\n    def", which the
    # grader read as a failed rename. That is the THIRD time in one session that
    # an undersized probe budget has looked like lost capability -- 1024 made
    # Q2_K_XL's tool calling look broken at 40 %, and 3072 truncated 18 of
    # IQ1_M's 60 corpus attempts into NameErrors. The artifact that reasons
    # longest always looks like the weakest one when the budget is set for the
    # shortest. Budget for the most verbose arm, never the control.
    ap.add_argument("--n-predict", type=int, default=1600)
    ap.add_argument("--out", default="greedy-diff.jsonl")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",")]
    for a in arms:
        if a not in A.ARMS:
            sys.exit("unknown arm %r" % a)

    out = ROOT / "results" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    texts = {}

    for tag in arms:
        repo, quant, extra = A.ARMS[tag]
        p, fh, log, free_before = A.start(tag, repo, quant, extra, 0)
        if p is None:
            print("  %-16s FAILED TO START" % tag, flush=True)
            continue
        try:
            body = json.dumps({"prompt": A.CODE_PROMPT,
                               "n_predict": args.n_predict,
                               "temperature": 0.0, "top_k": 1, "seed": 42,
                               "cache_prompt": False}).encode()
            req = urllib.request.Request(A.BASE_URL + "/completion", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read().decode())
            text = d.get("content") or ""
            g = grade(text)
            texts[tag] = text
            row = dict(arm=tag, chars=len(text), **g)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(row, text=text)) + "\n")
            print("  %-16s %d chars  stale_order=%d  usage=%d  rename_complete=%s"
                  % (tag, len(text), g["stale_order_refs"], g["usage_refs"],
                     g["rename_complete"]), flush=True)
        except Exception as e:
            print("  %-16s PROBE FAILED: %s" % (tag, e), flush=True)
        p.kill()
        fh.close()

    A.kill_server()

    if len(texts) >= 2:
        base = arms[0]
        for tag in arms[1:]:
            if tag not in texts or base not in texts:
                continue
            print("\n===== %s vs %s =====" % (base, tag))
            a = texts[base].splitlines()
            b = texts[tag].splitlines()
            same = a == b
            print("  identical: %s" % same)
            if not same:
                shown = 0
                for line in difflib.unified_diff(a, b, lineterm="",
                                                 fromfile=base, tofile=tag):
                    print("  " + line[:160])
                    shown += 1
                    if shown > 60:
                        print("  ... diff truncated")
                        break

    print("\nFull text of every arm is in %s" % out)


if __name__ == "__main__":
    main()
