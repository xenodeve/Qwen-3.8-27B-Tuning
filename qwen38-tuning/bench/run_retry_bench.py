"""Measure the retry economics the research had to assume.

`run_bench.py` runs each task N times independently, which is the right shape
for estimating a pass RATE. It is the wrong shape for the question the new
research actually decides on:

    "A worker may have substantially lower Pass@1 than Q4 and still be the
     superior production worker if Xeno preserves final accepted quality,
     average attempts remain low, and Verified Merged Tasks/Hour increases."

That calculation needs p2 -- the probability that a failed task passes on an
evidence-assisted retry -- and the research states plainly that it assumed it:
`p2 = min(p1 + 0.10, 0.95)`, with H = 60 s of fixed overhead. Both are
measurable here, and for a heavily quantized worker p2 is exactly the number
that decides the lane: cheap detectable errors are an economic cost, silent ones
are a quality regression.

So: attempt once; on failure, retry ONCE with the actual traceback pasted in --
the evidence a real Xeno loop would hand back -- and record what happened.
Arithmetic lives in harness.retry_economics, which is tested.

Usage:
    python run_retry_bench.py --label q4-tuned
    python run_retry_bench.py --label q2kxl --escalation-s 90 --overhead-s 60
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tasks import TASKS
from run_bench import post, extract_code, verify
from harness import (retry_economics, check_output_contract,
                     compose_developer, CONTRACT)

ROOT = Path(r"C:\AI\qwen38-tuning")

# The contract lives in harness.CONTRACT so that compose_developer() and
# check_output_contract() cannot drift apart. `--skill FILE` prepends real skill
# text in front of it -- see the 2026-08-21 note in harness.compose_developer.
DEVELOPER = CONTRACT


def ask(endpoint, messages, args):
    payload = {
        "messages": messages,
        "temperature": args.temperature, "top_p": args.top_p,
        "top_k": args.top_k, "min_p": args.min_p,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort},
        "cache_prompt": False,
    }
    t0 = time.time()
    r = post(endpoint, payload)
    wall = time.time() - t0
    choice = r["choices"][0]
    usage = r.get("usage", {})
    # finish_reason distinguishes "the model got it wrong" from "the model ran
    # out of budget mid-reasoning". Both score as a failed attempt, and only one
    # of them is about the model. A quantization that reasons longer fails the
    # second way more often, which would otherwise read as lower capability.
    return choice["message"], usage, wall, choice.get("finish_reason")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--passes", type=int, default=1,
                    help="how many times to run the whole corpus")
    # 90 s is the research's stated Q4 attempt cost; 60 s its H. Both are
    # parameters here rather than constants precisely because they were assumed.
    ap.add_argument("--escalation-s", type=float, default=90.0)
    ap.add_argument("--overhead-s", type=float, default=60.0)
    ap.add_argument("--out", default="retry-bench.jsonl")
    # Inject real skill text ahead of the contract. Repeatable, order kept:
    #   --skill C:/Users/.../skills/karpathy-guidelines/SKILL.md
    #   --skill C:/Users/.../skills/tdd/SKILL.md
    # The corpus has always sent a 35-token developer message while the real
    # worker runs with these in its prompt, so every quality number so far
    # describes a configuration nobody ships.
    ap.add_argument("--skill", action="append", default=[],
                    help="path to a SKILL.md to prepend, repeatable")
    args = ap.parse_args()

    # Read the skills verbatim. A paraphrase would measure the paraphrase.
    skills = [Path(f).read_text(encoding="utf-8") for f in args.skill]
    developer = compose_developer(skills)
    if skills:
        print("developer message: %d skills, %d chars (contract kept, last)"
              % (len(skills), len(developer)), flush=True)

    out = ROOT / "results" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    truncations = 0
    contract_violations = 0
    attempts_seen = 0

    for p in range(1, args.passes + 1):
        for task in TASKS:
            messages = [{"role": "developer", "content": developer},
                        {"role": "user", "content": task["prompt"]}]
            attempts = 0
            request_failed = False
            per_attempt = []
            wall_total = 0.0
            accepted = False
            first_error = None
            tokens = 0

            for attempt in (1, 2):
                attempts = attempt
                try:
                    msg, usage, wall, finish = ask(args.endpoint, messages, args)
                except Exception as e:
                    wall_total += 0.0
                    first_error = first_error or ("REQUEST: %s" % e)
                    request_failed = True
                    break
                wall_total += wall
                tokens += usage.get("completion_tokens") or 0
                # Format-constraint adherence, scored on the RAW reply before
                # extract_code() repairs it. Separate from pass/fail on purpose:
                # redefining a passing task now would make every number
                # collected before 2026-08-20 incomparable.
                contract = check_output_contract(msg.get("content", ""))
                if not contract["ok"]:
                    contract_violations += 1
                per_attempt.append({"attempt": attempt,
                                    "contract_ok": contract["ok"],
                                    "contract_violations": contract["violations"],
                                    "tokens": usage.get("completion_tokens"),
                                    "finish_reason": finish,
                                    "reasoning_chars": len(msg.get("reasoning_content") or ""),
                                    "wall_s": round(wall, 1)})
                attempts_seen += 1
                if finish == "length":
                    truncations += 1
                code = extract_code(msg.get("content", ""))
                ok, err = verify(code, task["test"], task["id"], attempt)
                if ok:
                    accepted = True
                    break
                first_error = first_error or (err[0] if err else "")
                if attempt == 2:
                    break
                # The evidence a real loop hands back: the model's own output
                # plus the exact failure. Nothing else -- no hints, no rewrite
                # of the task, or the retry measures the prompt, not the model.
                messages = messages + [
                    {"role": "assistant", "content": msg.get("content", "")},
                    {"role": "user", "content":
                     "Your code failed verification with:\n\n%s\n\nFix it. Reply "
                     "with one fenced ```python block containing the corrected "
                     "code only." % (err[0] if err else "unknown failure")},
                ]

            rec = dict(label=args.label, corpus_pass=p, task=task["id"],
                       difficulty=task["difficulty"], attempts=attempts,
                       accepted=accepted, wall_s=round(wall_total, 1),
                       completion_tokens=tokens, per_attempt=per_attempt,
                       censored=(not accepted) and bool(
                           per_attempt and
                           per_attempt[-1].get("finish_reason") == "length"),
                       error=None if accepted else first_error)
            # A task whose FINAL attempt ended at the token limit is censored:
            # the model was mid-answer when the budget ran out, so its outcome
            # is unknown rather than negative. Scoring it as a failure is the
            # bias that made a 1-bit artifact look like 20/30 when a larger
            # budget showed 27/30 -- the same mistake at a quieter volume.
            last_truncated = bool(per_attempt and
                                  per_attempt[-1].get("finish_reason") == "length")
            records.append({"attempts": attempts, "accepted": accepted,
                            "wall_s": wall_total,
                            "censored": (not accepted) and last_truncated,
                            "request_failed": request_failed})
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print("  %-16s pass%d attempts=%d %s %.1fs %s"
                  % (task["id"], p, attempts, "ACCEPT" if accepted else "ESCALATE",
                     wall_total, "" if accepted else (first_error or "")[:80]),
                  flush=True)

    e = retry_economics(records, args.escalation_s, args.overhead_s)
    e.update(label=args.label, kind="SUMMARY",
             attempts_truncated_by_budget=truncations, max_tokens=args.max_tokens,
             attempts_seen=attempts_seen,
             output_contract_violations=contract_violations,
             output_contract_pct=(round(100.0 * (attempts_seen - contract_violations)
                                        / attempts_seen, 1) if attempts_seen else None))
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(e) + "\n")

    print("\n=== RETRY ECONOMICS %s ===" % args.label)
    for k, v in e.items():
        print("  %-26s %s" % (k, v))
    print("\n  p1/p2 measured, not assumed. merged_tasks_per_hour charges "
          "%.0fs escalation and %.0fs fixed overhead."
          % (args.escalation_s, args.overhead_s))


if __name__ == "__main__":
    main()
