"""Does Q8_0 KV change the model's output at depth?

The greedy probe used in the sweeps sends a 4-token prompt, so it barely touches
the KV cache — precisely the thing Q8 quantizes. It is a valid check for flags
that do not change arithmetic (thread counts, batch sizes, fit margins) and a
weak one here.

This test instead fills the context with ~50K tokens, then generates greedily
(temperature 0, top_k 1, fixed seed) so the whole continuation is decided by
attention over a deeply-populated cache. If F16 and Q8_0 produce identical text
under those conditions, Q8 KV is output-neutral for this workload. If they
diverge, the divergence point tells us how far in the cache precision starts to
matter.
"""
import difflib, hashlib, json, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from depth_sweep import filler, kill, start, post
from harness import completion_timeout_s

CTX = 65536
OUT = Path(r"C:\AI\qwen38-tuning\results\kv-equivalence.json")


def sample(extra, tag):
    p, fh, log, _ = start(CTX, extra, tag)
    if p is None:
        raise RuntimeError(f"{tag}: server failed to start")
    prompt = filler(int(CTX * 0.8)) + "\n\n# Summarise what every Handler class above has in common:\n"
    try:
        r = post("/completion", {"prompt": prompt, "n_predict": 200,
                                 "temperature": 0.0, "top_k": 1, "seed": 42,
                                 "cache_prompt": False},
                 timeout=completion_timeout_s(CTX))
    finally:
        # Teardown must survive a raise. See depth_sweep.run(); on 2026-08-21 a
        # timeout skipped the kill and the resident server broke the NEXT step.
        p.kill(); fh.close()
    return r["content"], r["timings"]


if __name__ == "__main__":
    print(f"greedy generation at ctx={CTX}, prompt ~{int(CTX*0.8)} tokens\n", flush=True)

    f16_text, f16_t = sample([], "equiv-f16")
    print(f"  F16 : prompt_n={f16_t['prompt_n']} tg={f16_t['predicted_per_second']:.2f} tok/s", flush=True)

    q8_text, q8_t = sample(["-ctk", "q8_0", "-ctv", "q8_0"], "equiv-q8")
    print(f"  Q8  : prompt_n={q8_t['prompt_n']} tg={q8_t['predicted_per_second']:.2f} tok/s", flush=True)

    same = f16_text == q8_text
    # Where the two continuations first differ, in tokens-ish, is the useful
    # number: identical is best, but "diverged after 180 of 200" is very
    # different from "diverged at token 3".
    common = 0
    for a, b in zip(f16_text, q8_text):
        if a != b:
            break
        common += 1

    result = dict(
        ctx=CTX, prompt_n=f16_t["prompt_n"],
        identical=same,
        f16_hash=hashlib.sha256(f16_text.encode()).hexdigest()[:16].upper(),
        q8_hash=hashlib.sha256(q8_text.encode()).hexdigest()[:16].upper(),
        common_prefix_chars=common,
        f16_len=len(f16_text), q8_len=len(q8_text),
        f16_tok_s=round(f16_t["predicted_per_second"], 2),
        q8_tok_s=round(q8_t["predicted_per_second"], 2),
    )
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n  identical: {same}")
    if not same:
        print(f"  common prefix: {common} of {min(len(f16_text), len(q8_text))} chars")
        print("\n  first divergence:")
        for line in list(difflib.unified_diff(
                f16_text.splitlines(), q8_text.splitlines(),
                "F16", "Q8", lineterm="", n=1))[:12]:
            print("   ", line)
    print(f"\n-> {OUT}")
