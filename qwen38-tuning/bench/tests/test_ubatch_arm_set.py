"""The `-ub` arm set must actually reach llama-server as different values.

THE INCIDENT THIS GUARDS, and it is a live hazard rather than a historical one.
`dflash2_arena.start()` hardcodes `-ub 256` into the argv it builds, and appends
the arm's `extra` AFTER it. So an arm that sets `-ub 128` produces an argv
containing `-ub 256 ... -ub 128`, and whether the run measures 128 or 256
depends entirely on llama.cpp taking the LAST occurrence.

If it took the first, every arm in this set would run at 256, the sweep would
come back flat, and the flat result would be written up as "`-ub` has no
effect" -- which is exactly how `--spec-ngram-mod-n-min` produced twelve boots,
a plausible spread and no effect present to find (`05-runtime-flags.md`).

So this file pins two separate things:

  1. the arm set varies `-ub` and nothing else, and
  2. the value that survives to the END of the assembled argv is the one the
     arm's name claims -- because that is the one llama.cpp's parser keeps
     (`common/arg.cpp:1666`, a plain setter that overwrites `params.n_ubatch`).

Reading the last occurrence rather than the first is the whole point. A helper
that returns the first match would report 256 for every arm and pass a test that
looks identical to this one.

WHAT THIS FILE CANNOT DO is prove the server honoured it. Only the boot log can,
via `llama_context: n_ubatch = N`, and the sweep asserts that before trusting
any rate.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena

EXPECTED = {"ub-256-base": "256", "ub-128": "128", "ub-64": "64"}


def last_flag(args, name):
    """The value following the LAST occurrence of `name`. See the docstring."""
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
    return arena.ARM_SETS["ubatch"]


def assembled(extra):
    """The argv `start()` will hand to llama-server, without launching it."""
    return arena.server_argv(98304, list(extra))


def test_arm_names_are_exactly_the_ones_pinned_here():
    assert [name for name, _ in arm_set()] == list(EXPECTED)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_the_surviving_ubatch_is_the_one_the_arm_name_claims(name):
    argv = assembled(dict(arm_set())[name])
    assert last_flag(argv, "-ub") == EXPECTED[name]


def test_the_hardcoded_256_really_is_still_in_front_of_the_override():
    """If this stops being true the guard above has become decorative.

    It is pinned so that a future edit removing the hardcoded -ub also has to
    come here and decide what the test is now for.
    """
    argv = assembled(["-ub", "64"])
    assert first_flag(argv, "-ub") == "256"
    assert last_flag(argv, "-ub") == "64"


def test_the_arms_are_three_distinct_ubatch_values():
    seen = {last_flag(assembled(a), "-ub") for _, a in arm_set()}
    assert seen == {"256", "128", "64"}


def test_every_arm_runs_the_served_decoder_and_no_sidecar():
    """The set measures throughput cost, not VRAM, and the arms must say so.

    The VRAM question was closed by `ubatch_preflight.py`: a 4x cut in ubatch
    returns 66 MiB, against the 45-376 MiB band the DFlash2 arms are unreliable
    in. Adding the sidecar here would spend hours on timeouts to re-answer that,
    and would put a 146x-spread arm underneath a measurement whose expected
    effect is small.

    So the baseline is `ngram-mod` -- what all four worker profiles serve, and
    stable within 4 % over six boots. `-md` must be absent: an arm that quietly
    loaded the drafter would swamp the effect being measured.
    """
    for name, args in arm_set():
        argv = assembled(args)
        assert last_flag(argv, "--spec-type") == "ngram-mod", name
        assert "-md" not in argv, name


def test_nothing_but_ubatch_varies_across_the_arms():
    stripped = []
    for _, args in arm_set():
        a = list(args)
        while "-ub" in a:
            i = a.index("-ub")
            del a[i:i + 2]
        stripped.append(a)
    assert all(s == stripped[0] for s in stripped)
