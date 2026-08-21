"""Stability gate — does the server survive a long agent loop on this model?

The new research makes this a HARD gate, ahead of any quality number:

    "A model that achieves 75 tok/s but occasionally hangs a server slot after a
     tool loop is a worse Xeno worker than 13 tok/s Q4."

`prefix_cache_gate.py` already answers "is the prefix reused, and what breaks
it?" over four turns. This answers a different question that four turns cannot:
does anything degrade, corrupt, or hang across a hundred of them.

Method: one append-only conversation, agent-shaped (system + tool schemas + repo
context, then alternating user/assistant/tool_result), grown one turn at a time
through /completion so the timings block stays visible. Every `--perturb-every`
turns the system block is edited -- the way injecting a skill would -- which must
invalidate the prefix; the turn AFTER that must re-establish reuse. A model that
never recovers reuse is as broken as one that hangs, and much harder to notice.

What is recorded per turn: evaluated prompt tokens, cached tokens, decode rate,
wall time, and whether the reply was non-empty. What is reported: turns
survived, hangs, empty replies, p50/p95 wall, steady-state reuse, and recovery
behaviour around each forced invalidation.

Context is bounded on BOTH sides. A previous corpus in this project asserted only
a lower bound, so a 112K-token input passed its own size test and then failed
every request with HTTP 400 -- 0/18 in four seconds, which reads as "the model
cannot do deep context at all".

Usage:
    python stability_gate.py --label q4-tuned --turns 100
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import median

ROOT = Path(r"C:\AI\qwen38-tuning")
ENDPOINT = "http://127.0.0.1:8080/completion"

TOOLS = "\n".join(
    '<tool name="%s">{"type":"object","properties":{"path":{"type":"string"},'
    '"content":{"type":"string"},"opts":{"type":"object"}}}</tool>' % n
    for n in ["read_file", "write_file", "run_tests", "grep", "list_dir",
              "apply_patch", "git_status", "git_diff"]
)

REPO = "\n".join(
    "# src/module_%02d.py\n"
    "class Handler%02d:\n"
    "    def __init__(self, config):\n"
    "        self.config = config\n"
    "        self.cache = {}\n"
    "    def process(self, item):\n"
    "        key = item.get('id')\n"
    "        if key in self.cache:\n"
    "            return self.cache[key]\n"
    "        return self.transform(item)\n" % (i, i)
    for i in range(24)
)


def system_block(revision):
    """Revision 0 is the steady-state prompt. Any other revision prepends a
    block, which is exactly what injecting a skill does to a real agent."""
    base = ("You are a coding agent operating inside a repository. Inspect "
            "before editing. Prefer minimal diffs. Always run the tests after "
            "a change.\n")
    if revision == 0:
        return base
    return "<skill id=\"%d\">Follow the project conventions.</skill>\n%s" % (revision, base)


def post(prompt, n_predict, timeout):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                       "temperature": 0.7, "cache_prompt": True}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--turns", type=int, default=100)
    ap.add_argument("--perturb-every", type=int, default=10)
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default="stability-gate.jsonl")
    args = ap.parse_args()

    out = ROOT / "results" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    revision = 0
    tail = ""
    walls, reuse_steady = [], []
    hangs = empties = 0
    recovered = failed_recovery = 0
    turns_done = 0
    t_start = time.time()

    for turn in range(1, args.turns + 1):
        perturbed = args.perturb_every and turn % args.perturb_every == 0
        if perturbed:
            revision += 1

        head = ("<system>\n%s</system>\n<tools>\n%s\n</tools>\n<repo>\n%s\n</repo>\n"
                % (system_block(revision), TOOLS, REPO))
        tail += ("<user>Turn %d: inspect module_%02d and state one concrete risk "
                 "in its caching, in one sentence.</user>\n<assistant>"
                 % (turn, turn % 24))
        prompt = head + tail

        # Both bounds. Too small means the probe is not agent-shaped; too large
        # means every request 400s and the run reports a model failure instead.
        approx_tokens = len(prompt) // 3
        if approx_tokens > args.ctx - args.n_predict - 256:
            print("  stopping at turn %d: prompt ~%d tokens approaches the %d "
                  "window -- this is the probe's limit, not the model's"
                  % (turn, approx_tokens, args.ctx), flush=True)
            break

        try:
            d, wall = post(prompt, args.n_predict, args.timeout)
        except Exception as e:
            hangs += 1
            print("  turn %-4d REQUEST FAILED after %ss: %s"
                  % (turn, args.timeout, e), flush=True)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(label=args.label, turn=turn,
                                        error="REQUEST: %s" % e)) + "\n")
            if hangs >= 3:
                print("  three consecutive-class failures -- stopping", flush=True)
                break
            continue

        t = d["timings"]
        content = d.get("content") or ""
        if not content.strip():
            empties += 1
        tail += content + "</assistant>\n<tool_result>ok</tool_result>\n"
        turns_done = turn

        prompt_n = int(t.get("prompt_n", 0))
        cache_n = int(t.get("cache_n", 0))
        total_in = prompt_n + cache_n
        reuse = round(100.0 * cache_n / total_in, 1) if total_in else 0.0
        walls.append(wall)

        # A perturbed turn SHOULD re-evaluate almost everything; the turn after
        # it should be back to appending. Anything else means the cache never
        # recovers, which four-turn probes cannot see.
        if perturbed:
            pass
        elif turn > 1 and (turn - 1) % args.perturb_every == 0:
            if reuse >= 50.0:
                recovered += 1
            else:
                failed_recovery += 1
        else:
            reuse_steady.append(reuse)

        # prompt_ms and predicted_ms separately, not just wall: a perturbed
        # turn's wall time is prefill PLUS the decode of n_predict tokens, and
        # dividing wall by prompt_n to get a "prefill rate" understates it by
        # the whole decode. Projecting that wrong rate to a 16K window is how a
        # plausible number gets published.
        row = dict(label=args.label, turn=turn, perturbed=perturbed,
                   prompt_n=prompt_n, cache_n=cache_n, reuse_pct=reuse,
                   prompt_ms=round(t.get("prompt_ms", 0), 1),
                   predicted_ms=round(t.get("predicted_ms", 0), 1),
                   pp_tok_s=round(t.get("prompt_per_second", 0), 1),
                   tok_s=round(t.get("predicted_per_second", 0), 2),
                   wall_s=round(wall, 2), empty=not content.strip())
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if turn % 5 == 0 or perturbed:
            print("  turn %-4d %seval %-6d cached %-6d reuse %5.1f%%  %5.2fs  "
                  "%s tok/s" % (turn, "PERTURB " if perturbed else "        ",
                                prompt_n, cache_n, reuse, wall, row["tok_s"]),
                  flush=True)

    ws = sorted(walls)
    summary = dict(
        label=args.label, kind="SUMMARY", turns_requested=args.turns,
        turns_survived=turns_done, hangs=hangs, empty_replies=empties,
        p50_wall_s=round(median(ws), 2) if ws else None,
        p95_wall_s=round(ws[max(0, int(len(ws) * 0.95) - 1)], 2) if ws else None,
        steady_reuse_median_pct=round(median(reuse_steady), 1) if reuse_steady else None,
        invalidation_recovered=recovered, invalidation_not_recovered=failed_recovery,
        total_wall_s=round(time.time() - t_start, 1))
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    print("\n=== STABILITY GATE %s ===" % args.label)
    for k, v in summary.items():
        print("  %-30s %s" % (k, v))


if __name__ == "__main__":
    main()
