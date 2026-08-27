"""The deployment question: WHICH PAIRING do we serve, not "is DFlash2 worth it".

WHAT THIS CORRECTS. The `dual-dflash-tensor` set compared `ngram-mod` against
`draft-dflash,ngram-mod` and `draft-dflash` alone. That answers a MECHANISM
question -- how much of the gain is the drafter and how much is the pairing --
and it answered it: +19.4 % alone, +113.1 % paired. It does not answer what to
serve, because the obvious rival was never in it.

`draft-mtp` needs no second file: the head is inside UD-Q4_K_XL. It LOADS at
ctx 147,456 on the SERVED, UNPATCHED binary -- 66+0, CUDA0 with 1,571 MiB free
and CUDA1 with 861 -- which is a depth `draft-dflash` cannot reach at all, since
the ladder put its ceiling at 65,536. If MTP is anywhere close on rate it wins
the trade outright, because it costs neither the patch nor three quarters of the
context window.

WHY IT HAS NO RATE YET. All three paired rounds at 147,456 were voided by the
output guard: `copied_window_fraction = [0.519, 0.0, 0.23]`, identical across
rounds and therefore deterministic. Three unpaired readings before the guard ran
gave 44.5 / 54.3 / 92.7 tok/s, which are exactly the numbers CORRECTIONS 32 says
not to trust -- a speculative rate rises with how predictable the text is, and
copying the prompt is maximally predictable. NOTE THE MIDDLE ROUND IS 0.0: one
round did not copy, so this is not a closed door.

THIS SET IS A SEPARATE SET, not an extra arm on `dual-dflash-tensor`, because
that one has already written rows and adding an arm would change what those rows
were a comparison within.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena

EXPECTED = ["ngram-mod-base", "mtp+ngram", "dflash+ngram"]


def last_flag(args, name):
    val = None
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            val = args[i + 1]
    return val


def arm_set():
    return arena.ARM_SETS["dual-pairings"]


def assembled(extra):
    return arena.server_argv(65536, list(extra))


def by_name(name):
    return dict((n, a) for n, a, _ in arm_set())[name]


def test_the_arms_are_the_ones_pinned_here():
    assert [n for n, _, _ in arm_set()] == EXPECTED


def test_the_incumbent_is_the_baseline_and_carries_no_drafter():
    argv = assembled(by_name("ngram-mod-base"))
    assert last_flag(argv, "--spec-type") == "ngram-mod"
    assert "-md" not in argv


def test_mtp_never_loads_a_sidecar_file():
    """The head is inside UD-Q4_K_XL. `-md` would add a 1.4 GB file for nothing,
    and would silently make this a different experiment."""
    argv = assembled(by_name("mtp+ngram"))
    assert last_flag(argv, "--spec-type") == "draft-mtp,ngram-mod"
    assert "-md" not in argv, "draft-mtp must not be given a sidecar"


def test_dflash_does_load_its_file():
    argv = assembled(by_name("dflash+ngram"))
    assert last_flag(argv, "--spec-type") == "draft-dflash,ngram-mod"
    assert last_flag(argv, "-md") == arena.DRAFTER


def test_both_rivals_carry_the_same_ngram_settings_as_the_baseline():
    """Otherwise the comparison is decoder AND n-gram tuning at once."""
    base = last_flag(assembled(by_name("ngram-mod-base")), "--spec-ngram-mod-n-match")
    for name in ("mtp+ngram", "dflash+ngram"):
        assert last_flag(assembled(by_name(name)), "--spec-ngram-mod-n-match") == base, name


@pytest.mark.parametrize("name", EXPECTED)
def test_every_arm_is_the_tensor_split_with_a_ratio_on_both_cards(name):
    argv = assembled(by_name(name))
    assert last_flag(argv, "-sm") == "tensor", name
    assert last_flag(argv, "-ts") == "7819,15490", name
    env = dict((n, e) for n, _, e in arm_set())[name]
    assert env["CUDA_VISIBLE_DEVICES"] == arena.BOTH_CARDS, name


def test_only_the_decoder_varies():
    for name, args, _ in arm_set():
        argv = assembled(args)
        assert last_flag(argv, "-ub") == "1024", name
        assert last_flag(argv, "-ctk") == "q4_0", name
        assert last_flag(argv, "-ctv") == "q4_0", name


def test_no_two_arms_assemble_to_the_same_command_line():
    seen = [tuple(assembled(a)) for _, a, _ in arm_set()]
    assert len(set(seen)) == len(seen)
