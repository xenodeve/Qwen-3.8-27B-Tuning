"""A corpus too small for the requested window must FAIL, not quietly shrink.

THE FAULT. `dflash2_arena.filler(n_tokens, "real-code")` returns
`text[:n_tokens * 3]`. `real-code.txt` is 91,868 characters, so every request
above ~30,600 tokens silently returns the whole file and a prompt far shorter
than the caller asked for. At ctx 65,536 the caller believes it measured a
65,536-token window and actually measured a 30,600-token one. The run completes,
the rate is plausible, and nothing anywhere says the window was not filled.

That is this project's defining failure mode -- "an instrument that returns a
believable number instead of a failure is worse than one that crashes" -- and
it was live in the driver that produced every decoder verdict of 2026-08-22.
Those verdicts are unaffected because they were all taken at ctx 16,384, which
the corpus covers with room to spare. Any future run at depth would not have
been.

THE SECOND CORPUS. `real-code-deep.txt` exists so the depth question can be
asked at all. It is NOT a replacement: `real-code.txt` is frozen evidence whose
hash is stamped into rows already measured, and it is never modified. Deep runs
carry a different hash and are compared only with each other.

WHY IT IS ALLOWED TO BE BIGGER WITHOUT BEING WORSE. Judged on
`window_repetition_pct` at n=24 -- what ngram-mod actually keys on -- the deep
corpus is *cleaner* than the incumbent (0.4 % against 0.6 %), even though it
scores nearly double on `line_repetition_pct` because 43 files of real Python
share `import sys` and `try:`. Picking on the line metric would have rejected
honest code for its boilerplate. See `test_window_repetition.py`.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena
import harness

CORPORA = arena.CORPUS_DIR
CHARS_PER_TOKEN = 3          # the ratio filler() itself assumes
SERVED_CTX = 98304           # worker-iq2s-quality.ps1


def corpus_chars(name):
    return len((CORPORA / name).read_text(encoding="utf-8", errors="replace"))


# ---- the silent-truncation guard -------------------------------------------

def test_filler_raises_when_the_corpus_cannot_fill_the_window():
    too_many = corpus_chars("real-code.txt") // CHARS_PER_TOKEN + 10_000
    with pytest.raises(ValueError) as e:
        arena.filler(too_many, "real-code")
    assert "real-code.txt" in str(e.value)


def test_the_error_names_both_sizes_so_the_reader_can_act():
    too_many = corpus_chars("real-code.txt") // CHARS_PER_TOKEN + 10_000
    with pytest.raises(ValueError) as e:
        arena.filler(too_many, "real-code")
    msg = str(e.value)
    assert str(corpus_chars("real-code.txt")) in msg
    assert str(too_many * CHARS_PER_TOKEN) in msg


def test_a_window_the_corpus_can_serve_still_works():
    text = arena.filler(16384, "real-code")
    assert len(text) >= 16384 * CHARS_PER_TOKEN


def test_the_synthetic_regime_is_generated_and_never_raises():
    """It builds blocks on demand, so no corpus can be too small for it."""
    assert len(arena.filler(65536, "synthetic")) >= 65536 * CHARS_PER_TOKEN


# ---- the deep corpus -------------------------------------------------------

def test_the_deep_corpus_exists():
    assert (CORPORA / "real-code-deep.txt").is_file()


def test_the_deep_corpus_fills_the_window_we_actually_serve():
    assert corpus_chars("real-code-deep.txt") >= SERVED_CTX * CHARS_PER_TOKEN


def test_the_deep_regime_serves_a_depth_the_shallow_one_refuses():
    deep = arena.filler(SERVED_CTX, "real-code-deep")
    assert len(deep) >= SERVED_CTX * CHARS_PER_TOKEN
    with pytest.raises(ValueError):
        arena.filler(SERVED_CTX, "real-code")


def test_the_deep_corpus_is_no_more_ngram_friendly_than_the_incumbent():
    """The property that makes it usable, pinned so an edit cannot erode it.

    Judged on the window ngram-mod keys on, not on lines.
    """
    deep = (CORPORA / "real-code-deep.txt").read_text(encoding="utf-8", errors="replace")
    shallow = (CORPORA / "real-code.txt").read_text(encoding="utf-8", errors="replace")
    assert harness.window_repetition_pct(deep, 24) <= harness.window_repetition_pct(shallow, 24) + 1.0


def test_the_deep_corpus_is_nowhere_near_the_known_bad_filler():
    deep = (CORPORA / "real-code-deep.txt").read_text(encoding="utf-8", errors="replace")
    assert harness.window_repetition_pct(deep, 8) < 10.0


# ---- provenance ------------------------------------------------------------

def test_each_regime_reports_its_own_corpus_hash():
    a = arena.corpus_hash("real-code")
    b = arena.corpus_hash("real-code-deep")
    assert a and b and a != b


def test_a_generated_regime_reports_no_corpus_hash():
    assert arena.corpus_hash("synthetic") is None


def test_the_incumbent_corpus_is_untouched_by_the_addition():
    """Its hash is stamped into rows already measured. It is evidence, not scratch."""
    assert arena.corpus_hash("real-code") == "5672a9bcce74c0d0"
