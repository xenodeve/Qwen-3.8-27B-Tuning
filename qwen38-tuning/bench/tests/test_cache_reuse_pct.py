"""`cache_reuse_pct` must fail loudly when the timings block cannot answer it.

The incident this guards: `results/prefix-cache.jsonl` was written by a driver
that read `timings.get("cache_n")` with a default, so a response whose timings
block lacked the field would have recorded `cache_n=None` and been rendered as
a 0 % reuse row -- indistinguishable from a real cache miss. That is the fault
class this project's north star names: an instrument that returns a believable
number instead of a failure.

`cache_n` is also the field most likely to disappear, because it is the newest
of the three and only the raw /completion endpoint emits it -- /v1/chat/completions
returns a timings block without it. Pointing this driver at the wrong endpoint
must be a crash, not a page of 0 % rows.
"""
import pytest

from harness import cache_reuse_pct


def test_full_reuse_is_100_pct():
    assert cache_reuse_pct({"cache_n": 3981, "prompt_n": 0}) == pytest.approx(100.0)


def test_cold_prefill_is_0_pct():
    assert cache_reuse_pct({"cache_n": 0, "prompt_n": 3878}) == pytest.approx(0.0)


def test_partial_reuse_is_the_cached_share_of_the_whole_prompt():
    # the append-only control row from the 2026-08-22 run
    assert cache_reuse_pct({"cache_n": 3981, "prompt_n": 28}) == pytest.approx(
        100.0 * 3981 / 4009)


def test_missing_cache_n_raises_rather_than_reporting_a_miss():
    # /v1/chat/completions returns prompt_n without cache_n. Reporting 0 % here
    # would read as "the cache did not work" for a request that was never asked.
    with pytest.raises(KeyError):
        cache_reuse_pct({"prompt_n": 3878})


def test_missing_prompt_n_raises():
    with pytest.raises(KeyError):
        cache_reuse_pct({"cache_n": 3981})


def test_none_valued_field_raises_rather_than_coercing_to_zero():
    with pytest.raises(TypeError):
        cache_reuse_pct({"cache_n": None, "prompt_n": 28})


def test_empty_prompt_raises_rather_than_dividing_by_zero():
    # both zero means nothing was submitted; a percentage is undefined, and
    # returning 0.0 would claim the cache missed on a prompt that never existed.
    with pytest.raises(ValueError):
        cache_reuse_pct({"cache_n": 0, "prompt_n": 0})


def test_negative_counter_raises():
    with pytest.raises(ValueError):
        cache_reuse_pct({"cache_n": -1, "prompt_n": 10})
