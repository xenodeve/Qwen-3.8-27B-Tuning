"""The 2x2 arm set must actually vary both flags, independently.

THE INCIDENT THIS GUARDS. `--spec-ngram-mod-n-min` was swept as a fire-rate
knob and measured nothing -- 16/8/4/2 came back 79.7/79.7/79.7/79.8 -- because
the arms had been designed against a misreading of the flag. Twelve boots, a
plausible spread, and no effect present to find. A factorial fails the same way
for a cheaper reason: `_pair()` substitutes NGRAM (n_match=12, the incumbent)
whenever `extra_ngram` is omitted, so an arm that forgets to pass `_ngram(...)`
silently reverts to the value it was meant to be measured against while its
name goes on claiming otherwise.

That is this project's defining failure mode -- an instrument that returns a
believable number instead of a failure -- and here it is one missing argument
away. So the arm set is pinned to what its names promise, and to being a true
cross product rather than three arms and a duplicate.

Nothing here reads source text. It reads the argument lists the arena will hand
to llama-server, which is the thing that decides what gets measured.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena


# arm name -> (--spec-draft-n-max, --spec-ngram-mod-n-match)
EXPECTED = {
    "combo-base-n4-m12": ("4", "12"),
    "combo-n7-m12":      ("7", "12"),
    "combo-n4-m24":      ("4", "24"),
    "combo-n7-m24-both": ("7", "24"),
}


def flag(args, name):
    """The value following `name` in an argv list, or None if absent."""
    for i, a in enumerate(args):
        if a == name:
            return args[i + 1] if i + 1 < len(args) else None
    return None


def arm_set():
    return arena.ARM_SETS["draft-n-x-nmatch"]


def test_arm_names_are_exactly_the_ones_pinned_here():
    assert [name for name, _ in arm_set()] == list(EXPECTED)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_each_arm_carries_the_values_its_name_claims(name):
    args = dict(arm_set())[name]
    n_draft, n_match = EXPECTED[name]
    assert flag(args, "--spec-draft-n-max") == n_draft
    assert flag(args, "--spec-ngram-mod-n-match") == n_match


def test_it_is_a_true_cross_product_not_three_arms_and_a_duplicate():
    seen = {(flag(a, "--spec-draft-n-max"), flag(a, "--spec-ngram-mod-n-match"))
            for _, a in arm_set()}
    assert seen == {("4", "12"), ("7", "12"), ("4", "24"), ("7", "24")}


def test_every_arm_runs_both_speculators_so_the_flags_can_bite():
    """n_match is inert without ngram-mod, n-max inert without the drafter.

    An arm missing either would still boot and still produce a rate.
    """
    for name, args in arm_set():
        assert flag(args, "--spec-type") == "draft-dflash,ngram-mod", name
        assert flag(args, "-md") is not None, name


def test_nothing_else_varies_across_the_arms():
    """Two knobs move. Everything else must be identical or the 2x2 is not one."""
    stripped = []
    for _, args in arm_set():
        a = list(args)
        for name in ("--spec-draft-n-max", "--spec-ngram-mod-n-match"):
            i = a.index(name)
            del a[i:i + 2]
        stripped.append(a)
    assert all(s == stripped[0] for s in stripped)
