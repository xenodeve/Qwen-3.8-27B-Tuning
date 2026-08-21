"""Does this artifact actually finish a thought and emit an answer?

The cheapest possible gate, and it earned its place on first use. Qwen3.8-27B
UD-IQ1_S (Dynamic V3) is the fastest 27B this project has measured -- 50.7 tok/s
through the raw completion path, fully resident at 128K with 1.4 GB spare -- and
it produced NOTHING usable through the chat path across twelve attempts. Its
reasoning block ran 17,000 to 37,000 characters, repeating

    "I think this is correct. Let me finalize."
    "Actually, I realize I should reconsider."
    "Let me write the final version."

until it stopped with `finish_reason: stop` and `content: ""`. It could not exit
its own reasoning loop, so it never wrote the answer. That is catastrophic
repetition -- a documented low-bit failure -- and no amount of token budget
fixes it, because the budget was not the binding constraint.

Three probes per arm, roughly two minutes, run BEFORE the 30-task corpus. What
it separates:

    finish=length, content empty   -> budget-bound; raise it and retry
    finish=stop,   content empty   -> the model never left <think>; reject
    finish=stop,   content present -> proceed to the corpus

The distinction matters because both look identical in a pass/fail column, and
this project has already published four wrong verdicts by confusing them.

Usage:
    python answer_screen.py --arm v3-iq2xxs
    python answer_screen.py --arm v3-iq1m --trials 3 --max-tokens 8192
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import check_output_contract

ROOT = Path(r"C:\AI\qwen38-tuning")
ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"

DEV = ("You are a precise Python engineer. Reply with one fenced ```python "
       "block containing only the requested code. No explanation, no usage "
       "examples, no tests.")

PROBES = [
    "Implement a function merge_intervals(intervals) that merges overlapping "
    "intervals in a list of [start,end] pairs and returns them sorted by start.",
    "Implement a function is_balanced(s) returning True if the brackets in s "
    "are balanced, handling (), [] and {}.",
    "Implement a function rotated_search(nums, target) that finds target in a "
    "rotated sorted array and returns its index, or -1.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="label recorded with the result")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--out", default="answer-screen.jsonl")
    args = ap.parse_args()

    out = ROOT / "results" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    answered = truncated = empty_after_stop = contract_ok = 0
    rows = []

    for i in range(min(args.trials, len(PROBES))):
        body = json.dumps({
            "messages": [{"role": "developer", "content": DEV},
                         {"role": "user", "content": PROBES[i]}],
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort},
            "cache_prompt": False}).encode()
        req = urllib.request.Request(ENDPOINT, data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1200) as f:
                d = json.loads(f.read().decode())
        except Exception as e:
            print("  probe %d REQUEST FAILED: %s" % (i + 1, e), flush=True)
            rows.append(dict(arm=args.arm, probe=i + 1, error=str(e)))
            continue
        wall = time.time() - t0
        ch = d["choices"][0]
        msg = ch["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        finish = ch.get("finish_reason")
        contract = check_output_contract(content)

        if finish == "length":
            truncated += 1
        elif not content.strip():
            empty_after_stop += 1
        else:
            answered += 1
        if contract["ok"]:
            contract_ok += 1

        row = dict(arm=args.arm, probe=i + 1, finish_reason=finish,
                   reasoning_chars=len(reasoning), content_chars=len(content),
                   completion_tokens=(d.get("usage") or {}).get("completion_tokens"),
                   contract_ok=contract["ok"],
                   contract_violations=contract["violations"],
                   wall_s=round(wall, 1))
        rows.append(row)
        print("  probe %d  finish=%-8s reasoning=%-7d content=%-6d contract=%-5s %5.0fs"
              % (i + 1, finish, len(reasoning), len(content), contract["ok"], wall),
              flush=True)

    n = len([r for r in rows if "error" not in r])
    verdict = dict(arm=args.arm, kind="SUMMARY", probes=n,
                   answered=answered, truncated=truncated,
                   empty_after_stop=empty_after_stop,
                   contract_ok=contract_ok,
                   max_tokens=args.max_tokens,
                   reasoning_effort=args.reasoning_effort)
    # The three outcomes need different responses, so name which one applies.
    if n == 0:
        verdict["gate"] = "NO DATA"
    elif empty_after_stop == n:
        verdict["gate"] = "REJECT - never leaves the reasoning block"
    elif truncated == n:
        verdict["gate"] = "RAISE BUDGET - every probe hit the token limit"
    elif answered == 0:
        verdict["gate"] = "REJECT - no probe produced content"
    elif answered < n:
        verdict["gate"] = "MIXED - inspect before running the corpus"
    else:
        verdict["gate"] = "PASS - proceed to the corpus"

    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(verdict) + "\n")

    print("\n=== ANSWER SCREEN %s ===" % args.arm)
    for k, v in verdict.items():
        print("  %-20s %s" % (k, v))


if __name__ == "__main__":
    main()
