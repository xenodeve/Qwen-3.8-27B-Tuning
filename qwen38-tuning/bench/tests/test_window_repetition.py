"""`window_repetition_pct` — the repetition metric that matches what ngram-mod sees.

WHY A SECOND METRIC. `line_repetition_pct` exists because `depth_sweep.filler()`
repeated one class definition 962 times and every n-gram figure in this project
was measured on it. It answers "how much of this text is duplicated lines", and
for that filler it screams: 84.5 %.

It is the wrong instrument for choosing a corpus. `ngram-mod` keys a hash on a
window of `n_match` tokens (`common/ngram-mod.cpp:15-25`) — nothing about it
looks at lines. So:

  - A file full of `import sys` and `try:` scores HIGH on lines and LOW on
    windows, because the 24 tokens surrounding each occurrence differ. Real
    multi-file code is exactly this, and 43 files of this repo's own source
    score 18.9 % on lines while being honest text.
  - A repeated BLOCK scores high on both, and that is the case that flatters an
    n-gram drafter into a fake verdict.

Choosing a deep corpus on the line metric would have rejected honest code for
its boilerplate, or — worse in the other direction — accepted tiled text whose
lines happen to vary while its windows repeat.

THE MEASURE IS A PROXY AND SAYS SO. It splits on whitespace, not with the
model's tokenizer, so a "window" here is n whitespace-separated words rather
than n llama tokens. It is used to compare candidate corpora against each other
and against a known-bad filler, never to predict an absolute hit rate.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness


def test_text_with_no_repeated_window_scores_zero():
    text = " ".join(str(i) for i in range(200))
    assert harness.window_repetition_pct(text, n=8) == 0.0


def test_text_repeated_once_scores_about_half():
    """Every window of the second copy has been seen; the first copy's have not."""
    block = " ".join("tok%d" % i for i in range(200))
    pct = harness.window_repetition_pct(block + " " + block, n=8)
    assert 45.0 <= pct <= 55.0, pct


def test_boilerplate_repetition_does_not_score_as_window_repetition():
    """The case the line metric gets wrong, and the reason this function exists."""
    body = []
    for i in range(40):
        body.append("import sys")
        body.append("def handler_%d ( payload ) :" % i)
        body.append("    return transform_%d ( payload , mode = %d )" % (i, i))
    text = "\n".join(body)
    lines = harness.line_repetition_pct(text)
    windows = harness.window_repetition_pct(text, n=8)
    assert lines > 30.0, "the line metric should flag the repeated import: %s" % lines
    assert windows < 10.0, "the window metric should not: %s" % windows


def test_a_tiled_corpus_is_caught_even_when_every_line_differs():
    """Tiling with a changing index is what filler() did. Lines differ, windows do not."""
    block = " ".join("alpha beta gamma delta epsilon zeta eta theta".split() * 6)
    text = "\n".join("%s idx%d" % (block, i) for i in range(30))
    assert harness.window_repetition_pct(text, n=8) > 80.0


def test_n_is_the_window_width_and_a_wider_window_never_scores_higher():
    """Every 24-token repeat is also an 8-token repeat, never the reverse."""
    text = ("the quick brown fox jumps over the lazy dog " * 4
            + "a completely different tail of words appears here now ok " * 4)
    narrow = harness.window_repetition_pct(text, n=4)
    wide = harness.window_repetition_pct(text, n=16)
    assert wide <= narrow


def test_text_shorter_than_the_window_is_zero_not_a_crash():
    assert harness.window_repetition_pct("one two three", n=24) == 0.0


def test_empty_text_is_zero_not_a_division_error():
    assert harness.window_repetition_pct("", n=24) == 0.0


@pytest.mark.parametrize("n", [0, -1])
def test_a_nonpositive_window_raises_rather_than_returning_a_number(n):
    """This harness raises rather than guessing -- a zero-width window is a bug."""
    with pytest.raises(ValueError):
        harness.window_repetition_pct("some text here", n=n)
