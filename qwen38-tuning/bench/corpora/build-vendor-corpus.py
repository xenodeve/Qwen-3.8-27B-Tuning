"""Build `real-code-vendor.txt` -- run ONCE, then never again without a reason.

Same discipline as `build-deep-corpus.py`: the output is FROZEN EVIDENCE,
`filler()` reads the committed file, and this script is not wired to the
measurement path. Re-running it after `llama.cpp` moves produces a different
file, which is why the output is committed and hashed.

WHY A THIRD CORPUS EXISTS.

On 2026-08-24 every arena row at ctx 147,456 came back void: the model answered
a 64,210-token `real-code-deep` prompt in **9 tokens** and stopped on EOS, while
a 43,162-token prompt of the same corpus at the same ctx ran the full 512-token
budget (issue #44). Two explanations fit and they lead to different places:

  LENGTH   any prompt of that size collapses, and the arena cannot measure the
           window we serve at all.

  CONTENT  this corpus, at that slice, ends somewhere that makes EOS the
           greedy continuation -- and a different corpus of the same length
           would run.

`real-code-deep` cannot separate them, because it is one text and the long
slice is the short slice plus more of the same 45 files. A corpus from an
UNRELATED codebase, cut to the same length, can: run both at 147,456 and the
one that differs is the variable.

WHAT GOES IN.

  llama.cpp/gguf-py/gguf/*.py     the GGUF reader/writer library

Real Python from a real project, written by people who have never seen this
repo -- which is the point. It is also thematically adjacent to what the worker
reads all day, so the workload stays honest rather than becoming a synthetic
stress test.

ONE LIBRARY, NOT THREE. Adding `scripts/` and `examples/` gives 1,141,245 chars
at **1.6 %** 24-word repetition; `gguf-py/gguf` alone gives 597,630 at **0.7 %**,
against `real-code-deep`'s 0.4 %. Both pass the 5 % refusal, but `ngram-mod` is
one of the arms measured on this text and it keys on exactly that window, so the
more repetitive set would have flattered it. 597,630 chars still covers ctx
147,456 with room to spare -- more than `real-code-deep`'s 406,146 does.

NOT this repo's own source, and NOT `real-code-deep` rearranged. Tiling or
reordering an existing corpus is the fault behind CORRECTIONS.md 20: it changes
the bytes without changing what the text is, and manufactures exactly the
repeated windows `ngram-mod` keys on.
"""

import hashlib
import pathlib
import sys

VENDOR = pathlib.Path(r"C:\AI\llama.cpp")
OUT = pathlib.Path(__file__).parent / "real-code-vendor.txt"

SOURCES = [
    (VENDOR / "gguf-py" / "gguf", "*.py"),
]


def collect():
    """Every matching *.py, in sorted order.

    Sorted, because the concatenation order decides the bytes and an unstable
    order would make the file unreproducible for no gain.
    """
    files = []
    for d, pat in SOURCES:
        files.extend(sorted(p for p in d.glob(pat) if p.is_file()))
    return files


def main():
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
    import harness

    files = collect()
    if not files:
        raise SystemExit("no source found under %s -- is llama.cpp checked out?"
                         % VENDOR)

    parts = []
    for p in files:
        rel = p.relative_to(VENDOR).as_posix()
        parts.append("# ==== %s ====\n%s" % (rel, p.read_text(encoding="utf-8",
                                                              errors="replace")))
    text = "\n\n".join(parts)

    line_pct = harness.line_repetition_pct(text)
    win24 = harness.window_repetition_pct(text, 24)
    win8 = harness.window_repetition_pct(text, 8)

    if win24 > 5.0:
        raise SystemExit(
            "refusing to write: %.1f %% of 24-word windows repeat. A corpus "
            "that repeats at the width ngram-mod keys on manufactures the hits "
            "the measurement is trying to observe." % win24
        )

    OUT.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]
    print("wrote %s" % OUT)
    print("  %d files, %d chars, fills ctx ~%d at 3 chars/token"
          % (len(files), len(text), len(text) // 3))
    print("  line_repetition   %.1f %%  (boilerplate; not what ngram-mod sees)"
          % line_pct)
    print("  window_repetition %.1f %% at n=24, %.1f %% at n=8" % (win24, win8))
    print("  sha256[:16]       %s" % digest)


if __name__ == "__main__":
    main()
