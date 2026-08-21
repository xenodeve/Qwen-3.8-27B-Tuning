"""Measure the cold start a real harness feels, through the real harness.

    python scripts/bench-cold-start.py --harness qwen --runs 3
    python scripts/bench-cold-start.py --harness claude-xeno --runs 3 --label warmed

WHY IT READS THE SERVER LOG AND NOT JUST THE CLOCK. Wall time around a CLI
includes its own bootstrap, which for these harnesses is seconds and varies. The
number that matters is the one llama-server prints, and it is authoritative:

    prompt eval time = 21630.42 ms / 16796 tokens (1.29 ms per token, 776.50 ...)

So each run records both, and the two disagreeing is itself information: a large
gap is harness overhead, not prefill.

A run that produces NO prompt-eval line is recorded as a failure, never as a
zero. Thirteen instrument faults in this project produced a plausible number
instead of an error; this is the cheapest possible defence against the",
fourteenth.
"""
import argparse, json, re, subprocess, time
from pathlib import Path

ROOT = Path(r"C:\AI\qwen38-tuning")
LOG = ROOT / "logs" / "worker.err"
OUT = ROOT / "results" / "cold-start.jsonl"

HARNESS = {
    "qwen": [r"C:\Users\xenod\AppData\Local\qwen-code\bin\qwen.cmd", "-p"],
    "claude-xeno": [r"C:\Users\xenod\.claude\claude-xeno.bat", "-p"],
}

EVAL = re.compile(r"prompt eval time = *([0-9.]+) ms / *([0-9]+) tokens")


def parse_evals(text):
    """Every (ms, tokens) pair in a slice of server log."""
    return [(float(m.group(1)), int(m.group(2))) for m in EVAL.finditer(text)]


# Self-check. The parser is the whole instrument; a silent regex drift here
# would report every run as a failure, or worse, as a different number.
_SAMPLE = ("209.48.198.830 I slot print_timing: id  0 | task 18754 | prompt eval "
           "time =   21630.42 ms / 16796 tokens (    1.29 ms per token,   776.50 "
           "tokens per second)")
assert parse_evals(_SAMPLE) == [(21630.42, 16796)], parse_evals(_SAMPLE)


def run_once(harness, prompt, timeout_s):
    start_bytes = LOG.stat().st_size if LOG.exists() else 0
    cmd = HARNESS[harness] + [prompt]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout_s)
        rc, err = p.returncode, (p.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        rc, err = None, f"timeout after {timeout_s}s"
    wall = round(time.time() - t0, 2)

    with LOG.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(start_bytes)
        evals = parse_evals(f.read())

    row = {"harness": harness, "wall_s": wall, "returncode": rc,
           "n_evals": len(evals)}
    if evals:
        # INSTRUMENT FAULT, FIXED 2026-08-21: this took evals[0] as "the"
        # prefill. A harness makes several calls per turn -- Qwen Code sends a
        # 603-token one before the real 54,499-token request -- so the first is
        # the smallest and reporting it hid the whole cold start inside
        # "harness overhead". Same shape as instrument fault 9, which averaged
        # acceptance from the first of five generations.
        biggest_ms, biggest_tok = max(evals, key=lambda e: e[1])
        total_ms = sum(ms for ms, _ in evals)
        row.update(
            prefill_tokens=biggest_tok,
            prefill_ms=biggest_ms,
            prefill_tok_s=round(biggest_tok / (biggest_ms / 1000), 1),
            all_prefill_ms=round(total_ms, 1),
            all_prefill_tokens=sum(t for _, t in evals),
            overhead_s=round(wall - total_ms / 1000, 2),
            evals=[[round(ms, 1), t] for ms, t in evals])
    else:
        row.update(note="no prompt eval line -- the request never reached the "
                        "model, or the server log moved", stderr_tail=err)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", choices=sorted(HARNESS), required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--prompt", default="reply with exactly the word: ok")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, a.runs + 1):
        row = run_once(a.harness, a.prompt, a.timeout)
        row.update(run=i, label=a.label)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + chr(10))
        if "prefill_ms" in row:
            print("  run {}: rc={} | biggest prefill {} tok in {:.1f} s = {} tok/s"
                  " | {} calls, {} tok total | wall {} s, overhead {} s".format(
                      i, row["returncode"], row["prefill_tokens"],
                      row["prefill_ms"] / 1000, row["prefill_tok_s"],
                      row["n_evals"], row["all_prefill_tokens"],
                      row["wall_s"], row["overhead_s"]), flush=True)
        else:
            print("  run {}: NO PREFILL RECORDED - {}".format(i, row["note"]),
                  flush=True)


if __name__ == "__main__":
    main()
