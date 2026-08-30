"""`repo_block` must raise when the corpus cannot cover the requested depth.

Same incident as `dflash2_arena.filler`'s guard, one file over: that function
used to return `text[:n]` unconditionally, so a run asking for more than the
corpus held silently measured a shallower window and reported a plausible rate
for a depth it never reached (`CORRECTIONS.md` 20). A prefix-cache probe has
the same exposure and a worse consequence -- the whole question is "does reuse
survive DEPTH", so a row that quietly reports a shallower conversation answers
a different question than the one asked.

DISCLOSURE: `repo_block` was written before this file, not red-first. The rule
in `CLAUDE.md` is that every primitive in `bench/` is written red-first, and
that did not happen here. Rather than restage it, each assertion below was
confirmed to fail against a deliberately broken `repo_block` (guard removed,
and budget silently clamped) before being kept -- see the mutation note in the
session record. A green test nobody has seen fail is not evidence.
"""
import pytest

from prefix_cache_depth import repo_block
from dflash2_arena import CORPUS_DIR, CORPUS_FILES


def _corpus_len(regime):
    return len((CORPUS_DIR / CORPUS_FILES[regime]).read_text(
        encoding="utf-8", errors="replace"))


def test_returns_exactly_the_requested_number_of_chars():
    assert len(repo_block(5000, "real-code-deep")) == 5000


def test_a_budget_the_corpus_cannot_cover_raises():
    too_big = _corpus_len("real-code-deep") + 1
    with pytest.raises(ValueError):
        repo_block(too_big, "real-code-deep")


def test_the_error_names_the_file_and_both_numbers():
    too_big = _corpus_len("real-code-deep") + 1
    with pytest.raises(ValueError) as e:
        repo_block(too_big, "real-code-deep")
    msg = str(e.value)
    assert CORPUS_FILES["real-code-deep"] in msg
    assert str(too_big) in msg
    assert str(_corpus_len("real-code-deep")) in msg


def test_the_shallow_corpus_raises_where_the_deep_one_does_not():
    # the exact confusion the two regimes exist to prevent: a depth real-code
    # cannot reach must not silently succeed by truncation.
    n = _corpus_len("real-code") + 1
    assert n < _corpus_len("real-code-deep")
    repo_block(n, "real-code-deep")            # deep covers it
    with pytest.raises(ValueError):
        repo_block(n, "real-code")             # shallow must refuse


def test_exact_fit_is_allowed():
    n = _corpus_len("real-code")
    assert len(repo_block(n, "real-code")) == n


def test_unknown_regime_raises_rather_than_falling_back_to_a_default():
    with pytest.raises(KeyError):
        repo_block(1000, "synthetic")
