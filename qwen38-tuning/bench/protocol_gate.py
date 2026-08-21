"""Protocol gate — can this candidate still operate the tool interface?

Why this is a GATE and not another score: the coding corpus extracts a fenced
Python block, so a model that degraded into "here is what I would do" prose
scores exactly the same as one that emitted a clean tool call. For a heavily
quantized worker that is the *expected* failure mode, and the new research makes
100 % schema compliance a precondition before a candidate may become default --
ahead of any tok/s number.

Three things are checked, in the order they break:

  1. does it emit a call at all, with the right name and parseable arguments
  2. are the required fields -- including a NESTED array of objects -- present
  3. does the tool_call_id round-trip: feed the result back and see whether the
     model continues instead of re-issuing the same call

Scoring lives in harness.check_tool_call, which is tested. This file only drives
the server, which is deliberately not unit-tested (see the seam note in
tests/test_harness.py).

Usage:
    python protocol_gate.py --label q2kxl --trials 10
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import check_tool_call, line_repetition_pct

ROOT = Path(r"C:\AI\qwen38-tuning")
ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "apply_patch",
        "description": "Apply an edit to a file in the repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative file path"},
                "reason": {"type": "string", "description": "One-line rationale"},
                "edits": {
                    "type": "array",
                    "description": "Edits to apply, in file order",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line": {"type": "integer"},
                            "replacement": {"type": "string"},
                        },
                        "required": ["line", "replacement"],
                    },
                },
            },
            "required": ["path", "reason", "edits"],
        },
    },
}]

SPEC = {"name": "apply_patch", "required": ["path", "reason", "edits"], "max_calls": 1}

TASK = (
    "The file src/cache.py line 14 reads `self.order.remove(k)` and line 22 reads "
    "`victim = self.order.pop()`. Rename the attribute `order` to `usage` on both "
    "lines. Use the apply_patch tool. Do not reply with prose or code fences."
)


def post(payload, timeout=900):
    req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def nested_edits_ok(args):
    """The nested array is where a damaged model degrades first: it emits the
    flat fields correctly and flattens or stringifies the list of objects."""
    if not isinstance(args, dict):
        return False, "arguments are not an object"
    edits = args.get("edits")
    if not isinstance(edits, list):
        return False, "edits is %s, not an array" % type(edits).__name__
    if not edits:
        return False, "edits array is empty"
    for i, e in enumerate(edits):
        if not isinstance(e, dict):
            return False, "edits[%d] is %s, not an object" % (i, type(e).__name__)
        for f in ("line", "replacement"):
            if f not in e:
                return False, "edits[%d] missing %r" % (i, f)
    return True, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default="protocol-gate.jsonl")
    args = ap.parse_args()

    out = ROOT / "results" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    calls_ok = nested_ok = roundtrip_ok = 0
    field_omissions = truncated = 0
    reasoning_lengths = []
    t_start = time.time()

    for trial in range(1, args.trials + 1):
        messages = [
            {"role": "developer", "content":
             "You are a coding agent. Use the provided tools to make changes."},
            {"role": "user", "content": TASK},
        ]
        payload = {"messages": messages, "tools": TOOLS, "tool_choice": "auto",
                   "temperature": args.temperature, "max_tokens": args.max_tokens,
                   "cache_prompt": False}
        try:
            r = post(payload)
        except Exception as e:
            print("  trial %-3d REQUEST FAILED %s" % (trial, e), flush=True)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(label=args.label, trial=trial,
                                        error="REQUEST: %s" % e)) + "\n")
            continue

        choice = r["choices"][0]
        msg = choice["message"]
        # finish_reason separates the two failures that look identical in a
        # pass/fail column: "length" means the budget ran out mid-reasoning,
        # anything else means the model chose not to call. Without it, a probe
        # parameter gets reported as a capability difference.
        finish = choice.get("finish_reason")
        reasoning_chars = len(msg.get("reasoning_content") or "")
        verdict = check_tool_call(msg.get("tool_calls") or [], SPEC)
        field_omissions += sum(1 for e in verdict["errors"] if "missing" in e)
        reasoning_lengths.append(reasoning_chars)
        if finish == "length":
            truncated += 1
        nested, nested_err = (False, "no call")
        if verdict["ok"]:
            calls_ok += 1
            nested, nested_err = nested_edits_ok(verdict["args"])
            if nested:
                nested_ok += 1

        # Round two: feed the tool result back and see whether the model
        # continues. A model that re-issues the identical call has lost the
        # tool_call_id correlation, which in an agent loop is an infinite loop.
        rt = None
        rt_reason = None
        if verdict["ok"]:
            call = msg["tool_calls"][0]
            follow = messages + [
                {"role": "assistant", "content": msg.get("content") or "",
                 "tool_calls": msg["tool_calls"]},
                {"role": "tool", "tool_call_id": call.get("id", "call_0"),
                 "content": json.dumps({"status": "applied", "lines_changed": 2})},
            ]
            try:
                r2 = post({"messages": follow, "tools": TOOLS, "tool_choice": "auto",
                           "temperature": args.temperature,
                           # same budget as round one: round two also reasons,
                           # and a smaller cap here would truncate the
                           # continuation and be scored as a lost round-trip
                           "max_tokens": args.max_tokens,
                           "cache_prompt": False})
                c2 = r2["choices"][0]
                m2 = c2["message"]
                repeated = bool(m2.get("tool_calls"))
                rt = not repeated and bool((m2.get("content") or "").strip())
                # Round two has the same two failure modes as round one and they
                # are not interchangeable: re-issuing the call means the
                # tool_call_id correlation was lost (an infinite loop in a real
                # agent), while running out of budget means only that the model
                # was still reasoning. Record which.
                rt_finish = c2.get("finish_reason")
                rt_reason = ("repeated_call" if repeated
                             else "truncated" if rt_finish == "length"
                             else "empty" if not rt else None)
                if rt:
                    roundtrip_ok += 1
            except Exception as e:
                rt = False
                rt_reason = "request_failed"
                nested_err = nested_err or ("round two failed: %s" % e)

        row = dict(label=args.label, trial=trial, call_ok=verdict["ok"],
                   errors=verdict["errors"], nested_ok=nested,
                   nested_error=nested_err, roundtrip_ok=rt,
                   finish_reason=finish, roundtrip_failure=rt_reason,
                   reasoning_chars=reasoning_chars,
                   completion_tokens=(r.get("usage") or {}).get("completion_tokens"))
        # Keep the text whenever anything failed. "no tool call emitted" without
        # the reply is undiagnosable: refusing, explaining, and emitting a fenced
        # patch instead are three different defects with three different fixes.
        if not verdict["ok"] or not nested or rt is False:
            row["reply_excerpt"] = ((msg.get("content") or "")[:600] or None)
            reasoning = msg.get("reasoning_content") or ""
            row["reasoning_excerpt"] = (reasoning[:400] or None)
            # Instrument fault, 2026-08-21: this probe recorded a reasoning
            # LENGTH of 16,341 characters and kept 400 of the text. Three
            # documents then asserted the model was "looping" -- an inference
            # nobody could check, because the trace was gone. It was not
            # looping: a full capture scored 0.00 % line repetition and ended on
            # `stop`, and the code it wrote passed.
            #
            # So keep the whole trace, and answer the question with a number
            # rather than leaving it to be inferred from the length.
            if reasoning:
                row["reasoning_repetition_pct"] = line_repetition_pct(reasoning)
                d = ROOT / "logs" / "reasoning"
                d.mkdir(parents=True, exist_ok=True)
                f = d / ("%s-trial%s.txt" % (args.label, trial))
                f.write_text(reasoning, encoding="utf-8")
                row["reasoning_path"] = str(f)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print("  trial %-3d call=%-5s nested=%-5s roundtrip=%-5s %s"
              % (trial, verdict["ok"], nested, rt,
                 ("; ".join(verdict["errors"]) or nested_err or "")), flush=True)

    n = args.trials
    summary = dict(label=args.label, kind="SUMMARY", trials=n,
                   call_compliance_pct=round(100.0 * calls_ok / n, 1),
                   nested_schema_pct=round(100.0 * nested_ok / n, 1),
                   roundtrip_pct=round(100.0 * roundtrip_ok / n, 1),
                   required_field_omissions=field_omissions,
                   truncated_by_budget=truncated,
                   max_tokens=args.max_tokens,
                   median_reasoning_chars=(sorted(reasoning_lengths)[len(reasoning_lengths)//2]
                                           if reasoning_lengths else None),
                   wall_s=round(time.time() - t_start, 1))
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    print("\n=== PROTOCOL GATE %s ===" % args.label)
    for k, v in summary.items():
        print("  %-26s %s" % (k, v))
    print("\n  gate: 100% call compliance required before this arm may be default")


if __name__ == "__main__":
    main()
