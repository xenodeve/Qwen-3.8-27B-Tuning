"""A row built from a 4-token generation is not a measurement.

INSTRUMENT FAULT (2026-08-22, issue #18). The `draft-n` sweep was re-run at
ctx 65,536 and reported a clean, tight, RESOLVED verdict:

    n-3-default  [19.1, 19.1]  +4.3 %
    n-4-base     [18.2, 18.4]  (baseline)
    n-7-clamp    [ 7.8,  8.1]  -56.5 %  RESOLVED

Every arm also showed `acc 0.0`, `ngram-mod decline 100 %` and
`draft-dflash mean len 1.0`, which read as "speculation stops working at depth"
-- a large and alarming finding.

It was not. The server log says:

    eval time = 112.32 ms /     4 tokens
    eval time =  61.45 ms /     2 tokens

**The generations produced two to four tokens, not the 512 that were asked
for.** The frozen corpus is ~28,000 tokens, and at ctx 65,536 the arena asks
for a prompt of 32,768, so the whole corpus was consumed and the model answered
in a few tokens. Four speculation calls across the run, one per completion, and
an acceptance rate computed over nothing.

`rate()` already refused a zero rate. It did not refuse a rate measured over
four tokens, so the arena produced three arms, six rows, a tight range and a
RESOLVED verdict from noise. That is the failure this project exists to refuse:
an instrument returning a believable number instead of a failure.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import generation_is_measurable


def t(predicted_n, rate=42.0):
    return {"predicted_n": predicted_n, "predicted_per_second": rate}


def test_a_full_length_generation_is_measurable():
    assert generation_is_measurable([t(512), t(512), t(511)], n_predict=512) is True


def test_the_four_token_generation_that_caused_this_is_refused():
    assert generation_is_measurable([t(4), t(2), t(4)], n_predict=512) is False


def test_a_generation_that_stopped_early_but_did_real_work_is_kept():
    """A model that answers in 300 of 512 tokens has still been measured.

    The guard is against a generation that never started, not against one that
    finished. Anything at or above a quarter of the budget counts.
    """
    assert generation_is_measurable([t(300), t(280), t(310)], n_predict=512) is True


def test_one_short_sample_among_good_ones_still_fails_the_row():
    """Medians hide it, which is exactly how it got through the first time.

    A row is a paired datapoint; if any of its samples is not a measurement,
    the row's median is not one either.
    """
    assert generation_is_measurable([t(512), t(3), t(512)], n_predict=512) is False


def test_no_timings_is_not_measurable():
    assert generation_is_measurable([], n_predict=512) is False


def test_a_zero_rate_is_still_refused():
    """The old guard's job, kept: a rate of zero is missing data."""
    assert generation_is_measurable([t(512, 0.0)], n_predict=512) is False


def test_the_threshold_is_a_fraction_of_what_was_ASKED_for():
    """Not an absolute count -- a 64-token budget is legitimately short."""
    assert generation_is_measurable([t(60), t(64)], n_predict=64) is True
    assert generation_is_measurable([t(4), t(4)], n_predict=64) is False
