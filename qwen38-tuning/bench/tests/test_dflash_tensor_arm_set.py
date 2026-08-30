"""The patched-binary DFlash2 set: both arms on the SAME split, nothing else moves.

WHY IT EXISTS. 2026-08-27 a local patch made `draft-dflash` load under
`-sm tensor` for the first time. The rate then has to be measured against
`ngram-mod` -- what the profile serves -- and the comparison is only meaningful
if the two arms differ in the decoder and in nothing else.

THE MISTAKE THIS REPLACES. The first rate for the patched build came from an
ad-hoc script whose prompt was ONE code block repeated to fill the window --
close to 100 % duplicate lines. It reported 167.51 tok/s. `ngram-mod` turns
repetition into throughput and this project has retracted a figure for exactly
that reason before (CORRECTIONS 2 and 32). The arena has the real-code corpus,
`generation_is_original` and `copied_window_fraction`; the script had none of
them. A measurement instrument that exists and is not used is worse than one
that does not exist, because its absence is invisible in the number.

THE BINARY IS NOT THE SERVED ONE and no test can check that from here. The patch
mirrors the target's output projection, so this set may answer "how much does
DFlash2 buy on the tensor split" and may NOT be compared against any row taken
on the served build. `QWEN38_LLAMA_EXE` must point at the patched binary or the
arms cannot load at all -- which is itself the check.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena

EXPECTED = ["ngram-mod-base", "dflash+ngram", "dflash"]


def last_flag(args, name):
    val = None
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            val = args[i + 1]
    return val


def arm_set():
    return arena.ARM_SETS["dual-dflash-tensor"]


def assembled(extra):
    return arena.server_argv(65536, list(extra))


def test_the_arms_are_the_ones_pinned_here():
    assert [n for n, _, _ in arm_set()] == EXPECTED


@pytest.mark.parametrize("name", EXPECTED)
def test_every_arm_is_the_tensor_split_with_a_ratio(name):
    """`-sm tensor` without `-ts` is the even split -- the 0.38 tok/s config."""
    argv = assembled(dict((n, a) for n, a, _ in arm_set())[name])
    assert last_flag(argv, "-sm") == "tensor", name
    assert last_flag(argv, "-ts") == "7819,15490", name


@pytest.mark.parametrize("name", EXPECTED)
def test_every_arm_runs_both_cards(name):
    env = dict((n, e) for n, _, e in arm_set())[name]
    assert env["CUDA_VISIBLE_DEVICES"] == arena.BOTH_CARDS, name


def test_the_baseline_is_what_the_profile_serves():
    argv = assembled(dict((n, a) for n, a, _ in arm_set())["ngram-mod-base"])
    assert last_flag(argv, "--spec-type") == "ngram-mod"
    assert last_flag(argv, "--spec-ngram-mod-n-match") == "12"
    assert "-md" not in argv, "the baseline must not load a drafter"


def test_the_drafter_arms_load_the_drafter_file():
    for name in ("dflash+ngram", "dflash"):
        argv = assembled(dict((n, a) for n, a, _ in arm_set())[name])
        assert "-md" in argv, name
        assert last_flag(argv, "-md") == arena.DRAFTER, name


def test_the_pair_and_the_solo_drafter_are_distinguishable():
    """The pairing was +48.5 % over ngram-mod at 16,384 and the solo drafter
    +34.7 %; collapsing them would report one arm twice."""
    solo = last_flag(assembled(dict((n, a) for n, a, _ in arm_set())["dflash"]),
                     "--spec-type")
    pair = last_flag(assembled(dict((n, a) for n, a, _ in arm_set())["dflash+ngram"]),
                     "--spec-type")
    assert solo == "draft-dflash"
    assert pair == "draft-dflash,ngram-mod"


def test_only_the_decoder_varies():
    for name, args, _ in arm_set():
        argv = assembled(args)
        assert last_flag(argv, "-ub") == "1024", name
        assert last_flag(argv, "-ctk") == "q4_0", name
        assert last_flag(argv, "-ctv") == "q4_0", name


def test_no_two_arms_assemble_to_the_same_command_line():
    seen = [tuple(assembled(a)) for _, a, _ in arm_set()]
    assert len(set(seen)) == len(seen)
