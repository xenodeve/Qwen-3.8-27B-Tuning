"""The n-gram family set must carry `-ts`, and must vary only the n-gram config.

TWO INCIDENTS THIS GUARDS, both found 2026-08-27.

1. `-ts` IS NOT OPTIONAL AT DEPTH. `-sm tensor` splits EVENLY when given no
   ratio (`llama-model.cpp:707`), the 12 GB card drives the display, and that
   configuration decoded at 0.38 tok/s (CORRECTIONS 33). `dual-decoder` carries
   no `-ts`, so its 147,456 rows ran the even split -- report 36 section 4 names
   those numbers and says they are "recorded here only so nobody quotes them as
   current", while the register quotes them as the decoder verdict. A set that
   forgets `-ts` measures a machine we do not serve.

2. THE WINNER CHANGES WITH THE WINDOW, so the arms must be distinguishable in
   the argv, not merely in their names. `n-match` 24 wins at 16,384 and 16 wins
   at 65,536 (`02-decoders.md`), and we ship 12, which loses at both. If two
   arms assembled to the same command line the sweep would come back flat and
   the flat result would be written up as "the family does not matter".

`ngram-cache` is DISQUALIFIED and must not appear: its greedy hash differs from
a same-depth baseline, so it changes the answer and is not draft-and-verify.

WHAT THIS FILE CANNOT DO is prove the server honoured any of it. Only the boot
log can, and the residency guard added the same day
(`test_a_spilled_arm_gets_no_verdict.py`) is what refuses a rate whose placement
does not match the baseline's.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena

EXPECTED_ARMS = ["ngram-mod-base", "nm-16", "nm-24", "map-k", "map-k4v"]


def last_flag(args, name):
    val = None
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            val = args[i + 1]
    return val


def arm_set():
    return arena.ARM_SETS["dual-ngram-family"]


def assembled(extra):
    return arena.server_argv(147456, list(extra))


def test_the_arms_are_the_ones_pinned_here():
    assert [name for name, _, _ in arm_set()] == EXPECTED_ARMS


@pytest.mark.parametrize("name", EXPECTED_ARMS)
def test_every_arm_carries_the_computed_split(name):
    """Not `-sm tensor` alone: without a ratio that is the even split."""
    argv = assembled(dict((n, a) for n, a, _ in arm_set())[name])
    assert last_flag(argv, "-sm") == "tensor", name
    assert last_flag(argv, "-ts") == "7819,15490", name


@pytest.mark.parametrize("name", EXPECTED_ARMS)
def test_every_arm_runs_both_cards(name):
    env = dict((n, e) for n, _, e in arm_set())[name]
    assert env["CUDA_VISIBLE_DEVICES"] == arena.BOTH_CARDS, name


def test_the_baseline_is_what_the_profile_serves():
    """A baseline that is not the served config answers a different question."""
    argv = assembled(dict((n, a) for n, a, _ in arm_set())["ngram-mod-base"])
    assert last_flag(argv, "--spec-type") == "ngram-mod"
    assert last_flag(argv, "--spec-ngram-mod-n-match") == "12"


def test_the_three_n_match_arms_reach_three_distinct_values():
    got = {}
    for name, args, _ in arm_set():
        if last_flag(assembled(args), "--spec-type") == "ngram-mod":
            got[name] = last_flag(assembled(args), "--spec-ngram-mod-n-match")
    assert sorted(got.values()) == ["12", "16", "24"], got


def test_the_variant_arms_are_distinct_spec_types():
    types = [last_flag(assembled(a), "--spec-type") for _, a, _ in arm_set()]
    assert types.count("ngram-map-k") == 1
    assert types.count("ngram-map-k4v") == 1


def test_ngram_cache_is_absent_because_it_changes_the_answer():
    for name, args, _ in arm_set():
        assert last_flag(assembled(args), "--spec-type") != "ngram-cache", name


def test_no_arm_loads_a_drafter_file():
    """No external drafter loads under -sm tensor, and one that tried would
    swamp the effect being measured."""
    for name, args, _ in arm_set():
        assert "-md" not in assembled(args), name


def test_no_two_arms_assemble_to_the_same_command_line():
    """A duplicate arm reports a spread and no effect present to find."""
    seen = [tuple(assembled(a)) for _, a, _ in arm_set()]
    assert len(set(seen)) == len(seen), "two arms are the same run"


def test_only_the_ngram_configuration_varies():
    """Split, micro-batch and KV are held constant across every arm."""
    for name, args, _ in arm_set():
        argv = assembled(args)
        assert last_flag(argv, "-ub") == "1024", name
        assert last_flag(argv, "-ctk") == "q4_0", name
        assert last_flag(argv, "-ctv") == "q4_0", name
