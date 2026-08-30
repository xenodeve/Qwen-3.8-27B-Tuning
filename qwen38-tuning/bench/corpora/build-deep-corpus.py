"""Build `real-code-deep.txt` -- run ONCE, then never again without a reason.

The output is FROZEN EVIDENCE, not a generated artifact. `filler()` reads the
committed file; it does not call this script. That separation is the whole fix
from CORRECTIONS.md 20, where the benchmark built its prompt from source files
that were being edited between runs and produced 78.9 against 105.4 tok/s on
byte-identical arguments.

So this script exists to make the provenance reproducible and auditable, NOT to
be part of the measurement path. Re-running it after the tree moves produces a
different file, which is exactly why the output is committed and hashed and the
script is not wired to anything.

WHAT GOES IN, and why the PowerShell profiles do not:

  qwen38-tuning/bench/*.py         the harness and its drivers
  qwen38-tuning/bench/tests/*.py   the tests, which are real code too
  scripts/*.py                     the documentation tools

  qwen38-tuning/scripts/*.ps1      EXCLUDED. 56 worker profiles that are near
                                   copies of one another: 16.5 % repetition at
                                   a 24-word window against 0.4 % for the set
                                   above. Including them takes the whole corpus
                                   to 4.5 % and buys depth this project would
                                   then have to distrust.

Judged with `harness.window_repetition_pct`, which measures what ngram-mod
actually keys on, rather than `line_repetition_pct`, which counts a shared
`import sys` as repetition and would have rejected honest multi-file code.
"""

import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).parent / "real-code-deep.txt"

SOURCES = [
    ROOT / "qwen38-tuning" / "bench",
    ROOT / "qwen38-tuning" / "bench" / "tests",
    ROOT / "scripts",
]


def collect():
    """Every *.py under the source directories, in sorted order.

    Sorted, because the concatenation order decides the bytes and an unstable
    order would make the file unreproducible for no gain.
    """
    files = []
    for d in SOURCES:
        files.extend(sorted(p for p in d.glob("*.py") if p.is_file()))
    return files


def main():
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
    import harness

    files = collect()
    parts = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
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
