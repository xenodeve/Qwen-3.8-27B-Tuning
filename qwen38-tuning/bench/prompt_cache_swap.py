"""What does `-cram` buy when an agent switches between two conversations?

`--cache-ram` defaults to **8192 MiB** (`common/common.h:615`) and no profile
here has ever set it. Nothing in this project had noticed it was on until
2026-08-23, when a `POST /slots/0?action=erase` failed to produce a cold turn:
the deep prefix probe came back at 17.8 % reuse because erase clears the slot
and not the RAM cache.

WHAT IT ACTUALLY STORES, from source. `prompt_save` calls
`llama_state_seq_get_data_ext` (`server-context.cpp:261-274`) -- **the whole
sequence state**, attention KV and recurrent state together, not a token list.
The server saves an idle slot into it and clears the slot for the next task
(`:1374-1379`). So this is genuine save/restore across requests, which is the
closest thing in llama.cpp to the RTX 3090 stack's `PREFIX_CACHE=1`.

WHY THE IN-SLOT PROBE CANNOT SEE IT. `prefix_cache_depth.py` grows ONE
conversation, so every turn hits the live slot and the RAM cache is never
consulted. The case that needs it is the one an agent actually produces:
work on task A, switch to task B, come back to A. With one slot, A's state must
have gone somewhere or A's second turn pays a full cold prefill.

THE ARMS. Same conversation pair, same order, two boots:

    -cram 8192   the default nobody set
    -cram 0      disabled

A→B→A→B→A. If the cache works, the SECOND visit to A reuses; if it does not,
every switch costs a cold prefill. Read `cache_n` and `prompt_ms`, not wall
time -- `harness.cache_reuse_pct` raises rather than reporting a plausible 0 %
when the timings block cannot answer.

Two distinct conversations, not one prompt edited: an edit would collapse reuse
for the reason `prefix-cache-depth.jsonl` already records, and would measure
that instead.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from dflash2_arena import CORPUS_DIR, CORPUS_FILES, corpus_hash
from harness import cache_reuse_pct
from prefix_cache_depth import SYSTEM, TOOLS, ENDPOINT

OUT = Path(r"C:\AI\qwen38-tuning\results\prompt-cache-swap.jsonl")


def conversation(text, task):
    return (f"<system>\n{SYSTEM}</system>\n<tools>\n{TOOLS}\n</tools>\n"
            f"<repo>\n{text}\n</repo>\n<user>{task}</user>\n<assistant>")


def gen(prompt, n_predict=8):
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict,
        "temperature": 0.0, "top_k": 1, "seed": 42, "cache_prompt": True,
    }).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read().decode())
    return d["timings"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=150000,
                    help="character budget PER conversation (not tokens)")
    ap.add_argument("--tag", required=True,
                    help="the -cram value this boot was started with")
    args = ap.parse_args()

    regime = "real-code-deep"
    full = (CORPUS_DIR / CORPUS_FILES[regime]).read_text(
        encoding="utf-8", errors="replace")
    need = 2 * args.chars
    if len(full) < need:
        raise ValueError(
            f"{CORPUS_FILES[regime]} holds {len(full)} chars but two "
            f"conversations of {args.chars} need {need}. Overlapping them would "
            f"make B a prefix of A and measure in-slot reuse instead.")

    # Disjoint halves: B must not share a prefix with A, or the second visit
    # would be explained by ordinary common-prefix reuse.
    a = conversation(full[:args.chars], "Find the caching bug in this code.")
    b = conversation(full[args.chars:need], "List every unhandled edge case here.")

    chash = corpus_hash(regime)
    rows = []

    def record(label, timings, wall):
        depth = timings["prompt_n"] + timings["cache_n"]
        row = dict(label=label, cram=args.tag, corpus_sha=chash,
                   chars=args.chars, depth_tokens=depth,
                   prompt_n=timings["prompt_n"], cache_n=timings["cache_n"],
                   reuse_pct=round(cache_reuse_pct(timings), 1),
                   prompt_ms=round(timings["prompt_ms"], 1),
                   wall_s=round(wall, 2))
        rows.append(row)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {label:<12} depth={depth:>7} eval={row['prompt_n']:>7} "
              f"reuse={row['reuse_pct']:>5.1f}%  prompt_ms={row['prompt_ms']:>9.1f} "
              f"wall={row['wall_s']:>7.2f}s", flush=True)
        return row

    print(f"=== A/B/A/B/A at {args.chars} chars each, -cram {args.tag} "
          f"(corpus {chash}) ===", flush=True)
    for i, (name, prompt) in enumerate(
            [("A-cold", a), ("B-cold", b), ("A-return", a),
             ("B-return", b), ("A-again", a)]):
        t, w = gen(prompt)
        record(name, t, w)

    print("\n=== VERDICT ===")
    by = {r["label"]: r for r in rows}
    a_cold, a_ret = by["A-cold"], by["A-return"]
    saved = a_cold["prompt_ms"] - a_ret["prompt_ms"]
    print(f"  A cold        : {a_cold['prompt_ms']:,.1f} ms, {a_cold['reuse_pct']:.1f} % reuse")
    print(f"  A after B     : {a_ret['prompt_ms']:,.1f} ms, {a_ret['reuse_pct']:.1f} % reuse")
    print(f"  saved on return: {saved:,.1f} ms "
          f"({100.0 * saved / a_cold['prompt_ms']:.1f} %)")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
