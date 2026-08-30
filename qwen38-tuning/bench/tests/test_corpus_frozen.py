"""The benchmark prompt must not change when the benchmark's source changes.

INSTRUMENT FAULT (2026-08-22, issue #18). `dflash2_arena.filler(n, "real-code")`
built its prompt by reading this directory's own source files -- harness.py,
depth_sweep.py, model_arena.py, opencode_corpus.py, kv_sweep.py -- and slicing
the first `n * 3` characters.

Between the run behind report 29 and the `--arms ngram-nmin` sweep, 3,045 bytes
were appended to `harness.py` (24,306 -> 27,351) to add a parser. The prompt
budget is 24,576 characters, so the workload went from *harness.py plus the
first 270 characters of depth_sweep.py* to *the first 24,576 characters of
harness.py alone*.

The same arm, with byte-identical arguments, then measured **78.9 tok/s** in one
run and **105.4** in the other -- a 33 % difference on a project whose stated
noise floor is 13.6 %. Nothing was throttling: 49 C, no power cap, no thermal
slowdown, zero throttle counters. The workload had changed underneath the
measurement.

Paired verdicts *within* each run survive, because one run sees one prompt.
Absolute rates across runs do not, and neither would any future run.

The fix is a frozen corpus file plus a hash recorded on every row, so a change
becomes visible in the data instead of invisible in the tree.
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dflash2_arena as arena

CORPUS = Path(arena.__file__).parent / "corpora" / "real-code.txt"


def test_the_corpus_is_a_committed_file():
    assert CORPUS.is_file(), (
        "the real-code prompt must come from a frozen file, not from live source"
    )


def test_the_prompt_is_exactly_the_frozen_file_plus_the_question():
    """Behaviour, not source text.

    Three tests in this suite have now failed because they searched the source
    for a string and matched their own explanatory prose. A test that reads
    source instead of running it is a test of the comments.

    This asserts the prompt is a pure function of the frozen file: if filler()
    ever reads a live file again, the equality breaks.
    """
    n = 4096
    body = CORPUS.read_text(encoding="utf-8", errors="replace")[:n * 3]
    assert arena.filler(n, "real-code").startswith(body)


def test_the_prompt_is_stable_across_calls():
    a = arena.filler(8192, "real-code")
    b = arena.filler(8192, "real-code")
    assert a == b


def test_the_corpus_hash_is_reported():
    """Every row carries it, so a silent change becomes a visible one."""
    h = arena.corpus_hash("real-code")
    assert isinstance(h, str) and len(h) == 16
    expect = hashlib.sha256(CORPUS.read_bytes()).hexdigest()[:16]
    assert h == expect


def test_a_bigger_budget_extends_rather_than_reshuffles():
    """Growing n must append, not slide the window over different content.

    The fault was that the window's CONTENT moved. Comparing the corpus slices
    rather than the whole prompts, because each prompt ends with a fixed
    question that is not part of the corpus.
    """
    body = CORPUS.read_text(encoding="utf-8", errors="replace")
    assert body[:8192 * 3].startswith(body[:2048 * 3])
    assert arena.filler(8192, "real-code").startswith(body[:2048 * 3])


def test_the_synthetic_regime_is_unaffected():
    """It was always generated, never read from disk, and stays that way."""
    assert arena.filler(4096, "synthetic") == arena.filler(4096, "synthetic")
    assert arena.corpus_hash("synthetic") is None
