r"""NVFP4 against the artifact we serve, with the model overridden per arm.

WHY THE MODEL IS IN THE ARM. `dflash2_arena.server_argv` hardcodes `-m TARGET`
from one module constant, so a set that varies the ARTIFACT cannot exist unless
the arm appends its own `-m`. llama.cpp takes the LAST occurrence -- the same
plain-setter behaviour `test_ubatch_arm_set.py` already depends on for `-ub`.
Pinned here because if it ever took the first instead, every arm would silently
run the same file and the sweep would report a decoder difference where none was
tested.

WHAT IS BEING ASKED. NVFP4 VERY-LOW loads at 229,376 and answers a 65,643-token
request; `UD-Q4_K_XL` reaches about 250,000. Both hold 147,456, so that is where
they can be compared without depth being a second variable.

AND THE MTP QUESTION IS THE POINT. One unpaired run of NVFP4 + `draft-mtp` came
back with `draft acceptance = 0.21053 (12 accepted / 57 generated), mean len
1.63`, against 0.488-0.554 and mean 16-18 for the `ngram-mod` we serve. If that
holds, MTP is COSTING throughput on this artifact rather than buying it, so
`nvfp4-ngram` must be in the set beside `nvfp4-mtp+ngram` -- otherwise the sweep
cannot tell "NVFP4 is slower" from "MTP is slower".

THIS SET COMPARES TWO DIFFERENT MODEL FILES. Nothing about the result transfers
to a row taken on either artifact alone, and the quality of neither has ever been
measured here.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena

EXPECTED = ["q4-ngram-base", "nvfp4-ngram", "nvfp4-mtp+ngram"]


def last_flag(args, name):
    val = None
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            val = args[i + 1]
    return val


def first_flag(args, name):
    for i, a in enumerate(args):
        if a == name:
            return args[i + 1] if i + 1 < len(args) else None
    return None


def arm_set():
    return arena.ARM_SETS["nvfp4-vs-q4"]


def assembled(extra):
    return arena.server_argv(147456, list(extra))


def by_name(n):
    return dict((k, a) for k, a, _ in arm_set())[n]


def test_the_arms_are_the_ones_pinned_here():
    assert [n for n, _, _ in arm_set()] == EXPECTED


def test_the_baseline_runs_the_artifact_we_serve_and_does_not_override_it():
    argv = assembled(by_name("q4-ngram-base"))
    assert argv.count("-m") == 1, "the baseline must not restate the model"
    assert last_flag(argv, "-m") == arena.TARGET
    assert last_flag(argv, "--spec-type") == "ngram-mod"


@pytest.mark.parametrize("name", ["nvfp4-ngram", "nvfp4-mtp+ngram"])
def test_the_nvfp4_arms_override_the_model_and_the_override_survives(name):
    """The hardcoded -m comes first; the arm's own -m must be the one that wins."""
    argv = assembled(by_name(name))
    assert argv.count("-m") == 2, name
    assert first_flag(argv, "-m") == arena.TARGET, name
    assert last_flag(argv, "-m").endswith("NVFP4-MTP-VERY-LOW.gguf"), name


def test_the_two_nvfp4_arms_differ_only_in_the_decoder():
    a = assembled(by_name("nvfp4-ngram"))
    b = assembled(by_name("nvfp4-mtp+ngram"))
    assert last_flag(a, "-m") == last_flag(b, "-m")
    assert last_flag(a, "--spec-type") == "ngram-mod"
    assert last_flag(b, "--spec-type") == "draft-mtp,ngram-mod"


def test_no_arm_loads_a_sidecar_drafter():
    """The MTP head is inside the NVFP4 file; -md would add a file for nothing."""
    for name, args, _ in arm_set():
        assert "-md" not in assembled(args), name


@pytest.mark.parametrize("name", EXPECTED)
def test_every_arm_is_the_tensor_split_with_a_ratio_on_both_cards(name):
    argv = assembled(by_name(name))
    assert last_flag(argv, "-sm") == "tensor", name
    assert last_flag(argv, "-ts") == "7819,15490", name
    env = dict((n, e) for n, _, e in arm_set())[name]
    assert env["CUDA_VISIBLE_DEVICES"] == arena.BOTH_CARDS, name


def test_the_ngram_settings_are_identical_across_all_three():
    vals = {last_flag(assembled(a), "--spec-ngram-mod-n-match") for _, a, _ in arm_set()}
    assert vals == {"12"}, vals


def test_no_two_arms_assemble_to_the_same_command_line():
    seen = [tuple(assembled(a)) for _, a, _ in arm_set()]
    assert len(set(seen)) == len(seen)
