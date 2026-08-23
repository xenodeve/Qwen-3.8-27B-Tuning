"""Does prefix reuse survive the depth we actually serve?

Scoreboard item #15 -- the last untested idea from the RTX 3090 pool, and the
one `08-rtx3090-transfer.md` calls "the single largest untested idea left".
Their `PREFIX_CACHE=1` took turn 2 of a 24K chat from ~23 s to 1.15 s and a
100K prefix from 169 s cold to 4.7 s.

WHAT IS ALREADY ANSWERED, AND WHAT IS NOT. `prefix_cache_gate.py` ran on
2026-08-22 and showed append-only turns reusing cache and any edit ahead of the
suffix collapsing it to cache_n=1. That run was at 3,878 tokens. The window we
serve is 98,304 and cold prefill there is 74.3 tok/s against 1,129 at 16,384,
so the shallow result licenses nothing at depth -- this project has assumed
transfer across depth once already and `CORRECTIONS.md` 22 records how that went.

WHY DEPTH COULD BREAK IT, FROM SOURCE. Qwen3.8 loads as arch `qwen35`, which is
on the `llm_arch_supports_rs_rollback` whitelist (`src/llama-arch.cpp:1044`), so
`common_context_can_seq_rm` classifies the context as SEQ_RM_TYPE_RS -- partial
removal bounded by `n_rs_seq` (`common/common.cpp`). And `n_rs_seq` is
`draft.n_max`, or **0 when no model-based drafter is loaded**
(`common/common.h:386`). Every worker-*.ps1 runs `--spec-type ngram` alone, so
n_rs_seq is 0 there and the recurrent half can only be removed whole. What
covers the difference is `--ctx-checkpoints`, default 32
(`common/common.h:613`), which the server enables precisely for RS and FULL
contexts (`tools/server/server-context.cpp:3372-3376`). Whether 32 checkpoints
span a 40K conversation is not stated anywhere and is what this measures.

DEPTH IS READ BACK, NOT ASSUMED. The prompt is built to a CHARACTER budget and
the real token count is taken from the server as `prompt_n + cache_n`.
`dflash2_arena.filler()` sizes by `n_tokens * 3`, and measured chars/token here
is 7.0-7.4 -- so a "ctx 40,000" prompt built that way is about 17,000 tokens.
Rows here carry the depth the server saw.

Usage:
    python prefix_cache_depth.py --chars 28000          # ~4K tokens, replicates the 08-22 run
    python prefix_cache_depth.py --chars 300000         # ~40K tokens, the served regime

Needs a running llama-server and the raw /completion endpoint: only that one
returns `cache_n`, and `harness.cache_reuse_pct` raises rather than reporting a
plausible 0 % if it is missing.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from dflash2_arena import CORPUS_DIR, CORPUS_FILES, corpus_hash
from harness import cache_reuse_pct

ENDPOINT = "http://127.0.0.1:8080/completion"
OUT = Path(r"C:\AI\qwen38-tuning\results\prefix-cache-depth.jsonl")

SYSTEM = ("You are a coding agent operating inside a repository. Inspect before "
          "editing. Prefer minimal diffs. Always run the tests after a change.\n")

TOOLS = "\n".join(
    f'<tool name="{n}">{{"type":"object","properties":{{"path":{{"type":"string"}},'
    f'"content":{{"type":"string"}},"opts":{{"type":"object"}}}}}}</tool>'
    for n in ["read_file", "write_file", "run_tests", "grep", "list_dir",
              "apply_patch", "git_status", "git_diff"]
)

# Append-only, the way an agent loop grows: each entry is what the client sends
# on top of everything it already sent.
TURNS = [
    "<user>Find the caching bug in the handler module.</user>\n<assistant>",
    "</assistant>\n<tool_result>read: cache never evicts, unbounded growth</tool_result>\n<assistant>",
    "</assistant>\n<tool_result>tests: FAILED test_memory_bound</tool_result>\n<assistant>",
    "</assistant>\n<user>Now apply the same fix to the sibling module.</user>\n<assistant>",
]


def repo_block(n_chars, regime):
    """n_chars of frozen real source. Raises if the corpus cannot cover it.

    Same rule as `dflash2_arena.filler`: truncating to fit would measure a
    shallower conversation than the row claims, which is the fault
    `CORRECTIONS.md` 20 records.
    """
    name = CORPUS_FILES[regime]
    text = (CORPUS_DIR / name).read_text(encoding="utf-8", errors="replace")
    if len(text) < n_chars:
        raise ValueError(
            f"{name} holds {len(text)} chars but the run asked for {n_chars}. "
            f"Truncating would measure a shallower conversation than the row "
            f"reports. Use --regime real-code-deep, or lower --chars.")
    return text[:n_chars]


def erase_slot(id_slot=0):
    """Drop the slot's cache so the next request is genuinely cold.

    This is what lets two depths share one boot. They must: free VRAM at boot
    moves 9,326-10,732 MiB and `--fit` follows it, so a shallow round and a deep
    round measured in different boots are not comparable -- `CLAUDE.md` forbids
    exactly that comparison. Without an erase they would not be comparable
    either, for the opposite reason: the shallow prompt is a strict prefix of
    the deep one, so the deep cold turn would start already warm.

    Needs `--slot-save-path` on the server or the endpoint 404s
    (`server-context.cpp:4549`), so a missing flag must raise here rather than
    silently leaving the cache in place and reporting the warm number as cold.
    """
    req = urllib.request.Request(
        f"http://127.0.0.1:8080/slots/{id_slot}?action=erase",
        data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def gen(prompt, n_predict=8, cache_prompt=True):
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict,
        "temperature": 0.0, "top_k": 1, "seed": 42,
        "cache_prompt": cache_prompt,
    }).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read().decode())
    return d["timings"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, required=True,
                    help="character budget for the repo block (NOT tokens)")
    ap.add_argument("--regime", default="real-code-deep", choices=sorted(CORPUS_FILES))
    ap.add_argument("--tag", default="", help="free label written into every row")
    ap.add_argument("--erase-first", action="store_true",
                    help="drop the slot cache so turn-1 is cold (needs --slot-save-path)")
    args = ap.parse_args()

    if args.erase_first:
        erase_slot()
        print("  slot cache erased -- turn-1 is cold", flush=True)

    chash = corpus_hash(args.regime)
    base = (f"<system>\n{SYSTEM}</system>\n<tools>\n{TOOLS}\n</tools>\n"
            f"<repo>\n{repo_block(args.chars, args.regime)}\n</repo>\n")

    rows = []

    def record(label, timings, wall, note=""):
        depth = timings["prompt_n"] + timings["cache_n"]
        row = dict(
            label=label, tag=args.tag, regime=args.regime, corpus_sha=chash,
            chars=args.chars, depth_tokens=depth,
            prompt_n=timings["prompt_n"], cache_n=timings["cache_n"],
            reuse_pct=round(cache_reuse_pct(timings), 1),
            prompt_ms=round(timings["prompt_ms"], 1),
            wall_s=round(wall, 2), note=note,
        )
        rows.append(row)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {label:<24} depth={depth:>7} eval={row['prompt_n']:>7} "
              f"reuse={row['reuse_pct']:>5.1f}%  prompt_ms={row['prompt_ms']:>9.1f} "
              f"wall={row['wall_s']:>7.2f}s  {note}", flush=True)
        return row

    print(f"=== append-only turns, {args.chars} chars of {args.regime} "
          f"(sha {chash}) ===", flush=True)
    convo = base
    for i, turn in enumerate(TURNS, 1):
        convo += turn
        t, w = gen(convo)
        record(f"turn-{i}", t, w, "cold" if i == 1 else "append-only")
        convo += " Understood. Proceeding with the next step."

    print("\n=== head edit -- the case an agent loop hits on a re-serialize ===",
          flush=True)
    t, w = gen(convo.replace("Prefer minimal diffs.", "Prefer small, surgical diffs."))
    record("edit-head", t, w, "one sentence changed near the front")

    print("\n=== VERDICT ===")
    cold = rows[0]
    warm = [r for r in rows if r["note"] == "append-only"]
    if warm:
        med = sorted(r["prompt_ms"] for r in warm)[len(warm) // 2]
        print(f"  depth measured          : {cold['depth_tokens']:,} tokens "
              f"({args.chars:,} chars -> {args.chars / max(cold['depth_tokens'], 1):.2f} chars/token)")
        print(f"  cold turn-1 prompt_ms   : {cold['prompt_ms']:,.1f}")
        print(f"  median append prompt_ms : {med:,.1f}")
        print(f"  saved per warm turn     : {cold['prompt_ms'] - med:,.1f} ms "
              f"({100.0 * (cold['prompt_ms'] - med) / cold['prompt_ms']:.1f} %)")
        reuse = ", ".join("%.1f%%" % r["reuse_pct"] for r in warm)
        print(f"  append-only reuse       : {reuse}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
