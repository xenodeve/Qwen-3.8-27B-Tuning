"""The pinned-allocation arms must differ in exactly the two flags they name.

WHAT THE SET IS FOR. `CLAUDE.md` forbids comparing raw decode across boots
because free VRAM at boot moves 9,326-10,732 MiB and `--fit` follows it. The
counter-move is to give `-ngl` an explicit number and turn `--fit` off, since
`common/fit.cpp` only adjusts arguments the user did not set. If that lowers the
boot-to-boot spread, a standing constraint on every measurement in this project
becomes negotiable -- which is why the arms have to be exactly right.

THE HAZARD IS THE SAME ONE AS `-ub`, TWICE OVER. `server_argv()` hardcodes both
`-ngl auto` and `--fit on`, and appends the arm's `extra` after them, so the
pinned arm's argv carries each flag twice. Both are plain setters
(`common/arg.cpp:2746` for `-ngl`, `:2825` for `--fit`), so the last occurrence
wins -- but a helper reading the first would report `auto` and `on` for both
arms, the sweep would come back flat, and flat would be written up as "pinning
does not help".

The baseline arm deliberately passes NO override: it must read `auto`/`on` from
the hardcoded prefix, because that is the configuration every measurement in
this project has actually used.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena

# arm name -> (last -ngl, last --fit)
EXPECTED = {
    "fit-auto-base": ("auto", "on"),
    "pinned":        ("65",   "off"),
}


def last_flag(args, name):
    val = None
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            val = args[i + 1]
    return val


def arm_set():
    return arena.ARM_SETS["pinned-alloc"]


def assembled(extra):
    return arena.server_argv(98304, list(extra))


def test_arm_names_are_exactly_the_ones_pinned_here():
    assert [name for name, _ in arm_set()] == list(EXPECTED)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_the_surviving_flags_are_the_ones_the_arm_name_claims(name):
    argv = assembled(dict(arm_set())[name])
    ngl, fit = EXPECTED[name]
    assert last_flag(argv, "-ngl") == ngl
    assert last_flag(argv, "--fit") == fit


def test_the_baseline_carries_no_override_at_all():
    """It must measure the configuration this project has always used.

    An override that happened to restate the defaults would still be wrong: it
    would silently survive a future change to `server_argv`'s prefix while the
    baseline claimed to track it.
    """
    base = dict(arm_set())["fit-auto-base"]
    assert "-ngl" not in base
    assert "--fit" not in base


def test_the_two_arms_differ_in_nothing_else():
    stripped = []
    for _, args in arm_set():
        a = list(args)
        for flag in ("-ngl", "--fit"):
            while flag in a:
                i = a.index(flag)
                del a[i:i + 2]
        stripped.append(a)
    assert all(s == stripped[0] for s in stripped)


def test_both_arms_run_the_served_decoder_and_no_sidecar():
    """The question is boot-to-boot spread, not which decoder wins.

    `ngram-mod` is what all four worker profiles serve and lands within 4 % over
    six boots, so it is the arm whose spread is worth measuring. A drafter would
    add a 146x-spread arm underneath a variance measurement.
    """
    for name, args in arm_set():
        argv = assembled(args)
        assert last_flag(argv, "--spec-type") == "ngram-mod", name
        assert "-md" not in argv, name
