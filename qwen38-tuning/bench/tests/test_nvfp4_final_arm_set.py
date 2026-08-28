r"""The proposal against the incumbent, in ONE run, because chaining is not measuring.

WHAT THIS REPLACES. Two verdicts exist and they were taken in different runs:

    nvfp4-mtp+ngram (n-match 12) against q4-ngram-base   +41.2 %  RESOLVED
    mtp+nm24 against mtp+nm12-base                       +27.1 %  RESOLVED

Multiplying those to claim about +79 % would be a cross-run comparison, and this
project's own standard forbids it: raw decode is not compared across boots
because the spread is measured and the cause is unknown. Two arms in one set,
rotated, is the only figure that can be quoted for a decision.

WHAT IS BEING PROPOSED. `esatapedico` NVFP4-MTP VERY-LOW with its baked-in MTP
head and `ngram-mod` at `n-match 24`, against `UD-Q4_K_XL` with `ngram-mod` at
`n-match 12` -- which is what every profile serves today.

24 IS THE VALUE THAT LOST ON THE OTHER ARTIFACT. At this exact depth on
UD-Q4_K_XL, 12 beat both 16 and 24, and `map-k` declined 100 % of its drafts. On
NVFP4 `map-k` recovers to +15.4 % RESOLVED and 24 wins at spread 0.4 %. The
n-gram tuning does not survive an artifact change, which nothing had tested
until the developer asked.

NO PATCH IN THIS SET. The MTP head is inside the file, so there is no `-md` and
no mirrored output projection; both arms run the SERVED binary. That is a
property worth as much as the rate: DFlash2 beside this measured +0.2 % with the
sign flipping while costing 650 MiB and a binary nobody outside this project has
reviewed.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena

EXPECTED = ["q4-ngram-base", "nvfp4-mtp+nm24"]


def last_flag(args, name):
    val = None
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            val = args[i + 1]
    return val


def arm_set():
    return arena.ARM_SETS["nvfp4-final"]


def assembled(extra):
    return arena.server_argv(147456, list(extra))


def by_name(n):
    return dict((k, a) for k, a, _ in arm_set())[n]


def test_the_arms_are_the_ones_pinned_here():
    assert [n for n, _, _ in arm_set()] == EXPECTED


def test_the_incumbent_is_exactly_what_the_profiles_serve():
    argv = assembled(by_name("q4-ngram-base"))
    assert argv.count("-m") == 1, "the incumbent must not restate the model"
    assert last_flag(argv, "-m") == arena.TARGET
    assert last_flag(argv, "--spec-type") == "ngram-mod"
    assert last_flag(argv, "--spec-ngram-mod-n-match") == "12"


def test_the_proposal_is_nvfp4_with_the_retuned_n_gram():
    argv = assembled(by_name("nvfp4-mtp+nm24"))
    assert last_flag(argv, "-m").endswith("NVFP4-MTP-VERY-LOW.gguf")
    assert last_flag(argv, "--spec-type") == "draft-mtp,ngram-mod"
    assert last_flag(argv, "--spec-ngram-mod-n-match") == "24", (
        "24 is the value that LOST on UD-Q4_K_XL and won here; serving 12 would "
        "measure the tuning of the other artifact")


def test_neither_arm_needs_the_patched_binary():
    """No sidecar drafter means no mirrored output projection and no patch."""
    for name, args, _ in arm_set():
        assert "-md" not in assembled(args), name


@pytest.mark.parametrize("name", EXPECTED)
def test_every_arm_is_the_tensor_split_on_both_cards(name):
    argv = assembled(by_name(name))
    assert last_flag(argv, "-sm") == "tensor", name
    assert last_flag(argv, "-ts") == "7819,15490", name
    env = dict((n, e) for n, _, e in arm_set())[name]
    assert env["CUDA_VISIBLE_DEVICES"] == arena.BOTH_CARDS, name


def test_the_micro_batch_and_kv_are_held_constant():
    for name, args, _ in arm_set():
        argv = assembled(args)
        assert last_flag(argv, "-ub") == "1024", name
        assert last_flag(argv, "-ctk") == "q4_0", name


def test_no_two_arms_assemble_to_the_same_command_line():
    seen = [tuple(assembled(a)) for _, a, _ in arm_set()]
    assert len(set(seen)) == len(seen)
