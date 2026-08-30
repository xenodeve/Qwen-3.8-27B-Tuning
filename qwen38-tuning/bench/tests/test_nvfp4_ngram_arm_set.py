r"""The n-gram family retuned for NVFP4, because the winner was chosen on another file.

WHY THIS EXISTS. `--spec-ngram-mod-n-match 12` is what every profile serves, and
it was chosen on `UD-Q4_K_XL`: at 147,456 it beat 16 and 24, and both map-k
variants declined 100 % of their drafts. On NVFP4 the same setting collapses --
acceptance 55.4 -> 22.1, and paired with MTP it reports `ngram-mod decline
97.2 %`. It is barely firing.

That is not a decoder fault. `ngram-mod` drafts from repetition in the text the
model is producing, so a different artifact writing differently is a different
problem for it. This project's own rule is that a verdict at one depth does not
transfer; a verdict on one ARTIFACT does not either, and nothing had tested that.

THE PAIRING IS HELD FIXED AT `draft-mtp`, because that is what would be served:
NVFP4 + draft-mtp + ngram-mod measured +41.2 % [+39.9, +43.0] over the served
configuration, and DFlash2 beside it added +0.2 % with the sign flipping while
costing 650 MiB of headroom and a patched binary. So the open question is not
which drafter -- it is which n-gram rides along with MTP.

COSTS ONLY BOOTS. None of these settings moves an allocation, which is why the
same sweep on UD-Q4_K_XL was worth running and is worth running again here.

`ngram-cache` is excluded, as everywhere else: its greedy hash differs from a
same-depth baseline, so it changes the answer rather than drafting for it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena

EXPECTED = ["mtp+nm12-base", "mtp+nm16", "mtp+nm24", "mtp+map-k", "mtp+map-k4v"]


def last_flag(args, name):
    val = None
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            val = args[i + 1]
    return val


def arm_set():
    return arena.ARM_SETS["nvfp4-ngram-retune"]


def assembled(extra):
    return arena.server_argv(147456, list(extra))


def by_name(n):
    return dict((k, a) for k, a, _ in arm_set())[n]


def test_the_arms_are_the_ones_pinned_here():
    assert [n for n, _, _ in arm_set()] == EXPECTED


@pytest.mark.parametrize("name", EXPECTED)
def test_every_arm_runs_the_nvfp4_target_and_keeps_mtp(name):
    """The drafter is settled; only the n-gram varies."""
    argv = assembled(by_name(name))
    assert last_flag(argv, "-m").endswith("NVFP4-MTP-VERY-LOW.gguf"), name
    assert last_flag(argv, "--spec-type").startswith("draft-mtp,"), name
    assert "-md" not in argv, name


def test_the_baseline_is_the_setting_every_profile_serves():
    argv = assembled(by_name("mtp+nm12-base"))
    assert last_flag(argv, "--spec-type") == "draft-mtp,ngram-mod"
    assert last_flag(argv, "--spec-ngram-mod-n-match") == "12"


def test_the_three_n_match_arms_reach_three_distinct_values():
    got = {}
    for name in ("mtp+nm12-base", "mtp+nm16", "mtp+nm24"):
        got[name] = last_flag(assembled(by_name(name)), "--spec-ngram-mod-n-match")
    assert sorted(got.values()) == ["12", "16", "24"], got


def test_the_variant_arms_swap_the_n_gram_not_the_drafter():
    for name, want in (("mtp+map-k", "draft-mtp,ngram-map-k"),
                       ("mtp+map-k4v", "draft-mtp,ngram-map-k4v")):
        assert last_flag(assembled(by_name(name)), "--spec-type") == want, name


def test_ngram_cache_is_absent_because_it_changes_the_answer():
    for name, args, _ in arm_set():
        assert "ngram-cache" not in last_flag(assembled(args), "--spec-type"), name


@pytest.mark.parametrize("name", EXPECTED)
def test_every_arm_is_the_tensor_split_on_both_cards(name):
    argv = assembled(by_name(name))
    assert last_flag(argv, "-sm") == "tensor", name
    assert last_flag(argv, "-ts") == "7819,15490", name
    env = dict((n, e) for n, _, e in arm_set())[name]
    assert env["CUDA_VISIBLE_DEVICES"] == arena.BOTH_CARDS, name


def test_nothing_but_the_n_gram_varies():
    for name, args, _ in arm_set():
        argv = assembled(args)
        assert last_flag(argv, "-ub") == "1024", name
        assert last_flag(argv, "--spec-draft-n-max") == "3", name


def test_no_two_arms_assemble_to_the_same_command_line():
    seen = [tuple(assembled(a)) for _, a, _ in arm_set()]
    assert len(set(seen)) == len(seen)
