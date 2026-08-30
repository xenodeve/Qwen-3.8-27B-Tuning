r"""`-DflashN`: serve DFlash2 at a chosen draft depth, and 4 is now measurable.

WHY THIS IS A PARAMETER AND NOT AN EDIT

`-Dflash` has served `--spec-draft-n-max 2` since it was written, and the
comment beside it says why: the recurrent state is 149.62 MiB x (1 + n_max), so
4 -> 2 returns 299 MiB, and at 131,072 the run finishes with 634/530 MiB free.
Every one of those MiB was needed.

4 is now the measured best on the split we serve. Three paired rounds at ctx
65,536 on the patched mirror (results/tensor-draft-depth-65536.jsonl, issue
#56): 4 is 55.72 tok/s against 7's 52.64, +109.2 % over `ngram-mod`, and 7 is
-6.5 % against 4 in every round while costing 308 MiB more. So the interesting
range is 2 to 4, not 4 to 7, and it is a knob rather than a new default:

  2  what has been served, and the only value measured AT 131,072
  4  the measured best at 65,536, and what every DFlash2 figure in the
     register was taken at -- but 299 MiB dearer, and NOT measured at depth

THE BUDGET ALREADY ASSUMES 4. `$DFLASH_DRAFTER_MIB = 1936` is the measured
resident cost at n_max 4 -- 748.12 of it is the recurrent state -- so serving at
2 leaves 299 MiB of unbudgeted slack and serving at 4 spends exactly what the
launcher already reserves. Nothing in the fitting arithmetic has to change.

THE HAZARD

The depth ceiling is not a preference. `-Dflash` caps the window at 131,072
because 147,456 LOADS, answers /health, and dies on the first real request. A
larger n_max eats the same headroom that ceiling exists to protect, and the
value measured at 131,072 is 2 -- not 4. A switch that let a user raise the
draft depth while quietly keeping the 131,072 ceiling would be shipping an
untested combination as though it were tested, so the profile must say so where
it is read.
"""
import os
import re
import subprocess

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q4-dual.ps1")
SERVE = os.path.join(ROOT, "serve.ps1")
HUB = os.path.join(ROOT, "serve-hub.bat")
LAUNCHERS = os.path.join(ROOT, "launchers")
BATS = ["serve-dual-dflash-n4.bat", "serve-dual-dflash-n4-lan.bat"]


def _whatif(script, *args):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", script, "-WhatIf"] + list(args),
        capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr


def preview(*args):
    out = _whatif(PROFILE, *args)
    m = re.search(r"WhatIf: would run\s*\n\s*(.+)", out)
    assert m, out
    return m.group(1), out


def depth(line):
    m = re.search(r"--spec-draft-n-max (\d+)", line)
    return m.group(1) if m else None


# ------------------------------------------------------------- the default

def test_the_default_is_still_two():
    """2 is the only value measured at the served 131,072. Changing the
    default would ship an untested depth to anyone running the old launcher."""
    line, _ = preview("-Dflash")
    assert depth(line) == "2", line


# --------------------------------------------------------------- the switch

def test_it_serves_the_requested_depth():
    line, _ = preview("-Dflash", "-DflashN", "4")
    assert depth(line) == "4", line


def test_it_still_carries_the_drafter_and_the_pairing():
    line, _ = preview("-Dflash", "-DflashN", "4")
    assert "draft-dflash,ngram-mod" in line, line
    assert "-md" in line and "-ngld 99" in line, line


def test_it_is_refused_without_dflash():
    """A draft depth with no drafter is a flag nobody reads and a launch
    nobody can attribute."""
    out = _whatif(PROFILE, "-DflashN", "4")
    assert "FATAL" in out, out


def test_a_depth_above_the_clamp_is_refused():
    """speculative.cpp:989 clamps at block_size - 1 = 7 for this drafter.
    Passing 8 does not error in llama.cpp -- it is silently clamped, and the
    row would record a value the run did not use."""
    out = _whatif(PROFILE, "-Dflash", "-DflashN", "8")
    assert "FATAL" in out, out


def test_a_negative_depth_is_refused():
    out = _whatif(PROFILE, "-Dflash", "-DflashN", "-1")
    assert "FATAL" in out, out


def test_zero_means_NOT_GIVEN_and_falls_back_to_the_default():
    """PowerShell cannot leave an [int] unset, so 0 is the sentinel. It is
    documented here because `-DflashN 0` reads like "no drafting" and is not:
    it serves the default 2, which is the one behaviour a reader could be
    surprised by."""
    line, _ = preview("-Dflash", "-DflashN", "0")
    assert depth(line) == "2", line


@pytest.mark.parametrize("n", ["2", "4", "7"])
def test_every_value_in_range_is_accepted(n):
    line, _ = preview("-Dflash", "-DflashN", n)
    assert depth(line) == n, line


# ------------------------------------------------- it says what it costs

def test_the_preview_names_the_depth_and_its_price():
    """The recurrent state is 149.62 x (1 + n_max). Raising it from the
    default spends headroom that the 131,072 ceiling exists to protect, and
    the developer reads this banner before deciding."""
    _, out = preview("-Dflash", "-DflashN", "4")
    assert "149.62" in out or "299" in out, out
    assert "4" in out


def test_it_warns_that_four_is_unmeasured_at_the_served_depth():
    _, out = preview("-Dflash", "-DflashN", "4")
    low = out.lower()
    assert "65,536" in out or "65536" in out, out
    assert "not measured" in low or "unmeasured" in low or "never" in low, out


def test_the_window_ceiling_still_applies():
    """131,072 is not a preference: 147,456 loads, answers /health, and dies
    on the first real request."""
    line, _ = preview("-Dflash", "-DflashN", "4",
                      "-Ctx", "200704")
    m = re.search(r"-c (\d+)", line)
    assert m and int(m.group(1)) <= 131072, line


def test_it_is_still_refused_with_maxctx():
    out = _whatif(PROFILE, "-Dflash", "-DflashN", "4",
                  "-MaxCtx")
    assert "FATAL" in out, out


# ----------------------------------------------------------------- plumbing

def test_it_reaches_the_profile_through_serve():
    out = _whatif(SERVE, "-Dual", "-Dflash", "-DflashN", "4")
    assert re.search(r"DflashN\s+4", out), out


# ---------------------------------------------------------------- launchers

@pytest.mark.parametrize("name", BATS)
def test_the_launcher_exists(name):
    assert os.path.exists(os.path.join(LAUNCHERS, name))


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_asks_for_the_depth(name):
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read()
    live = "\n".join(l for l in body.splitlines()
                     if not l.strip().upper().startswith("REM"))
    assert "-Dflash" in live and "-DflashN 4" in live, live


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_names_the_measurement_it_rests_on(name):
    """Every launcher here carries its own evidence. This one rests on three
    paired rounds at a depth it does not serve, and that has to be visible."""
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read()
    assert "65,536" in body or "65536" in body, body
    assert "55.7" in body or "109" in body, body


def test_lan_and_loopback_differ_only_in_lan():
    a = open(os.path.join(LAUNCHERS, BATS[0]), encoding="ascii").read()
    b = open(os.path.join(LAUNCHERS, BATS[1]), encoding="ascii").read()

    def call(t):
        return [l for l in t.splitlines() if "serve.ps1" in l][0]

    assert call(b).replace(" -Lan", "") == call(a)


def test_the_hub_offers_it():
    body = open(HUB, encoding="ascii").read()
    for n in BATS:
        assert n in body, n


def test_every_hub_key_is_wired_to_a_file():
    body = open(HUB, encoding="ascii").read()
    m = re.search(r"choice /c (\S+) /n /m \"  Choose", body)
    assert m, "the menu prompt changed shape"
    keys = m.group(1)
    assert keys.endswith("Q"), keys
    n = len(keys) - 1
    for i in range(1, n + 1):
        assert 'if "%%SEL%%"=="%d" (' % i in body, ("key %d has no branch" % i, keys)
    assert 'if "%%SEL%%"=="%d" goto :done' % (n + 1) in body
