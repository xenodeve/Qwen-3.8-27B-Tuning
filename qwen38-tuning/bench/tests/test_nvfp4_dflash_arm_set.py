r"""NVFP4 with DFlash2 beside it, against NVFP4 with the head it ships with.

WHERE THIS STARTS. At ctx 147,456, three paired rounds on real vendor code:

    q4-ngram-base     24.4 / 25.6 / 25.7   baseline (what we serve)
    nvfp4-ngram       17.8 / 22.7 / 18.3   -22.4 %  RESOLVED
    nvfp4-mtp+ngram   35.0 / 36.0 / 35.9   +41.2 %  RESOLVED

So the MTP head inside the NVFP4 file is worth more than the whole artifact
change, and `ngram-mod` alone on NVFP4 is a LOSS -- its acceptance falls from
55.4 to 22.1 because that artifact writes text the n-gram cannot predict.

THE QUESTION THIS SET ASKS. DFlash2 drafts from the model rather than from
repetition, which is exactly the weakness `nvfp4-ngram` exposed. On UD-Q4_K_XL
at 65,536 the `draft-dflash,ngram-mod` pairing measured +123.8 % against +38.9 %
for `draft-mtp,ngram-mod`. If that ordering survives onto NVFP4 at the served
depth, it beats the current champion.

WHY IT NEEDS THE PATCHED BINARY, and this is not optional. DFlash2's candidate
selector runs a TopK over the TARGET's LM head. Under `-sm tensor` those logits
are axis 0 -- scattered across both cards -- and llama.cpp aborts at
ggml-backend-meta.cpp:543. vLLM hits the same component from the other side and
refuses a QUANTIZED target LM head with "DFlash2 requires an unquantized target
LM head for candidate TopK". Two runtimes, one fragile point. Ours is mirrored
by a local patch that costs 1,080 MiB measured.

MEMORY IS THE RISK. `nvfp4-ngram` finished with 3,797 MiB free; the drafter's
buffer and the mirror together are about 1,618, which should leave roughly
2,180. Should is not does, and a rung that loads can still die on the first real
request -- the arena pushes one through every arm.

THE DRAFTER IS Q2_K_S-MIX, 535 MiB, not the 1,090 MiB Q4_K_M. Measured
2026-08-27: its buffer is 538.42 MiB against 786.35, it reaches 163,840 where
Q4_K_M does not, and its author's table puts throughput within a few percent of
the larger file at every n_max.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena

EXPECTED = ["nvfp4-mtp+ngram", "nvfp4-dflash+ngram"]


def last_flag(args, name):
    val = None
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            val = args[i + 1]
    return val


def arm_set():
    return arena.ARM_SETS["nvfp4-dflash"]


def assembled(extra):
    return arena.server_argv(147456, list(extra))


def by_name(n):
    return dict((k, a) for k, a, _ in arm_set())[n]


def test_the_arms_are_the_ones_pinned_here():
    assert [n for n, _, _ in arm_set()] == EXPECTED


@pytest.mark.parametrize("name", EXPECTED)
def test_both_arms_run_the_same_nvfp4_target(name):
    """Otherwise this compares artifacts, not decoders."""
    argv = assembled(by_name(name))
    assert last_flag(argv, "-m").endswith("NVFP4-MTP-VERY-LOW.gguf"), name


def test_the_reference_arm_is_the_current_champion():
    argv = assembled(by_name("nvfp4-mtp+ngram"))
    assert last_flag(argv, "--spec-type") == "draft-mtp,ngram-mod"
    assert "-md" not in argv, "the MTP head is inside the file"


def test_the_dflash_arm_loads_the_small_drafter_and_pairs_it_with_ngram():
    argv = assembled(by_name("nvfp4-dflash+ngram"))
    assert last_flag(argv, "--spec-type") == "draft-dflash,ngram-mod"
    assert last_flag(argv, "-md").endswith("Q2_K_S-MIX.gguf"), (
        "the 535 MiB drafter, not the 1,090 MiB one: its buffer is 538.42 MiB "
        "against 786.35 and memory is the risk here")


def test_the_two_arms_differ_only_in_the_drafter():
    a, b = assembled(by_name(EXPECTED[0])), assembled(by_name(EXPECTED[1]))
    assert last_flag(a, "-ub") == last_flag(b, "-ub")
    assert last_flag(a, "-ts") == last_flag(b, "-ts")
    assert last_flag(a, "--spec-ngram-mod-n-match") == last_flag(b, "--spec-ngram-mod-n-match")


@pytest.mark.parametrize("name", EXPECTED)
def test_every_arm_is_the_tensor_split_on_both_cards(name):
    argv = assembled(by_name(name))
    assert last_flag(argv, "-sm") == "tensor", name
    assert last_flag(argv, "-ts") == "7819,15490", name
    env = dict((n, e) for n, _, e in arm_set())[name]
    assert env["CUDA_VISIBLE_DEVICES"] == arena.BOTH_CARDS, name


def test_no_two_arms_assemble_to_the_same_command_line():
    seen = [tuple(assembled(a)) for _, a, _ in arm_set()]
    assert len(set(seen)) == len(seen)
