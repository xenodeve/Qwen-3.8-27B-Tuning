"""Read the per-implementation speculation statistics out of a server log.

WHY THIS EXISTS. With `--spec-type draft-dflash,ngram-mod` the summary line

    draft acceptance = 0.46013 (352 accepted / 765 generated), mean len = 3.26

pools both speculators, and pooling them hides the only thing worth knowing.
The real log, from the run behind report 29:

    ngram-mod    : #calls(b,g,a) = 4, 542,  31, #gen drafts =  31, mean acc len = 18.00
    draft-dflash : #calls(b,g,a) = 4, 511, 511, #gen drafts = 511, mean acc len =  2.91

`ngram-mod` was asked 542 times and produced a draft **31** times -- it declines
94.3 % of the time -- and `draft-dflash` was called exactly the 511 times ngram
declined. When ngram does fire it is worth **six times more per draft**. None of
that is visible in the pooled line, and the pooled line is what every earlier
measurement in this project read.

The declines are `common/speculative.cpp:1993`: if the n-gram table misses
before `n_min` successors, the whole draft is discarded, not truncated. So
`--spec-ngram-mod-n-min` is a fire-rate knob, and this parser is how its effect
is read.

`llama.cpp` prints these only at LOG_TRC, so the server must run at `-lv 4` or
higher. Our arena already runs `-lv 5`, which is why the data existed before the
question was asked.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import parse_spec_impl_stats

# Copied verbatim from logs/dflash2-dflash2-ngram-real-code-c16384-r3.log,
# the cumulative block printed after the last task of that run.
REAL = """
0.42.130.633 I spec common_specu: statistics        ngram-mod: #calls(b,g,a) =    4    542     31, #gen drafts =     31, #acc drafts =    28, #gen tokens =    920, #acc tokens =   527, #mean acc len = 18.00, #acc rate/pos = (0.903, 0.806), dur(b,g,a) = 0.688, 0.851, 0.021 ms
0.42.130.640 I spec common_specu: statistics     draft-dflash: #calls(b,g,a) =    4    511    511, #gen drafts =    511, #acc drafts =   390, #gen tokens =   2041, #acc tokens =   974, #mean acc len = 2.91, #acc rate/pos = (0.763, 0.526, 0.362, 0.254), dur(b,g,a) = 0.004, 3421.333, 0.148 ms
"""


def test_both_implementations_are_found():
    s = parse_spec_impl_stats(REAL)
    assert set(s) == {"ngram-mod", "draft-dflash"}


def test_the_decline_rate_is_the_headline():
    """Calls-for-generation minus drafts produced. The number the pooled line hides."""
    s = parse_spec_impl_stats(REAL)
    assert s["ngram-mod"]["n_call_draft"] == 542
    assert s["ngram-mod"]["n_gen_drafts"] == 31
    assert s["ngram-mod"]["decline_pct"] == pytest.approx(94.3, abs=0.1)
    assert s["draft-dflash"]["decline_pct"] == pytest.approx(0.0, abs=0.001)


def test_mean_accepted_length_is_per_implementation():
    s = parse_spec_impl_stats(REAL)
    assert s["ngram-mod"]["mean_acc_len"] == pytest.approx(18.00)
    assert s["draft-dflash"]["mean_acc_len"] == pytest.approx(2.91)


def test_draft_time_is_carried_because_the_cheap_one_is_not_the_fast_one():
    """ngram drafts in 0.851 ms cumulative; dflash takes 3421.333 ms for the same run."""
    s = parse_spec_impl_stats(REAL)
    assert s["ngram-mod"]["t_draft_ms"] == pytest.approx(0.851)
    assert s["draft-dflash"]["t_draft_ms"] == pytest.approx(3421.333)


def test_token_counts_survive():
    s = parse_spec_impl_stats(REAL)
    assert s["ngram-mod"]["n_gen_tokens"] == 920
    assert s["ngram-mod"]["n_acc_tokens"] == 527
    assert s["draft-dflash"]["n_gen_tokens"] == 2041


def test_the_last_block_wins_because_the_counters_are_cumulative():
    """The server prints after every completion and the numbers only grow.

    Taking the first block would report the first task, not the run.
    """
    earlier = REAL.replace("542     31", "320     11").replace("#gen drafts =     31",
                                                              "#gen drafts =     11")
    s = parse_spec_impl_stats(earlier + REAL)
    assert s["ngram-mod"]["n_gen_drafts"] == 31


def test_a_log_without_the_lines_returns_empty_not_zeros():
    """Absent is not the same fact as zero, and a caller must be able to tell.

    A log written at -lv 3 has no statistics lines at all; reporting 0 % decline
    for that would be a measurement of the verbosity setting.
    """
    assert parse_spec_impl_stats("nothing to see here") == {}


def test_an_implementation_that_was_never_asked_is_reported_as_unknown():
    """Dividing by n_call_draft = 0 must not produce 0 % or a crash."""
    none_asked = REAL.replace("=    4    542     31", "=    0      0      0").replace(
        "#gen drafts =     31", "#gen drafts =      0")
    s = parse_spec_impl_stats(none_asked)
    assert s["ngram-mod"]["decline_pct"] is None
