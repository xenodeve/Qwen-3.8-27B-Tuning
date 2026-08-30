r"""`-Nvfp4 -Dflash`: the NVFP4 artifact drafted by DFlash2 instead of its own head.

WHY THIS COMBINATION WAS REFUSED, AND WHY IT IS NOT ANY MORE

The profile rejected it outright:

    FATAL: -Nvfp4 already carries a drafter; -Dflash and -Mtp are others.

That was correct when the only evidence was `results/nvfp4-dflash-147456.jsonl`
-- **+0.2 % and the sign flips** -- so the switch would have shipped a
configuration measured to be worthless. **That run has since been shown to have
given DFlash2 none of what it wants** (ctx 147,456 against its best of 65,536,
`n_max` 3 against 4, `n-match` 12 against 24, the last being a window this
project records collapsing on NVFP4). Re-measured on 2026-08-30:

    65,536    +67.9 % [+65.8, +71.5] over ngram-mod, RESOLVED
    147,456   44.48 / 44.56 / 44.23, spread 0.7 %, against MTP's pooled 42.77
              over six rounds and two boot series spanning 9.3 %

So the combination is now measured, twice, and gets a switch.

WHAT THE SWITCH MUST NOT PRETEND

**It is not faster.** +4.0 % at the served depth is under the 13.6 % floor and
across boots. What it is, is **steadier** -- 0.7 % against 9.3 % -- and it
**costs about 950 MiB more headroom**, 1,450 against 2,400. A launcher that
advertised speed would be selling a number this project does not have.

THE SETTINGS ARE NOT NEGOTIABLE, because each was measured separately

  n-match 24   NVFP4's own window. 12 collapses here: acceptance 55.4 -> 22.1.
               This is what -Nvfp4 already serves, and DFlash2 must not change it.
  n_max 4      DFlash2's measured best, 2026-08-30. -DflashN still overrides.
  the mirror   draft-dflash under -sm tensor aborts on an unpatched binary at
               ggml-backend-meta.cpp:543.

AND THE CEILING IS DIFFERENT FROM `-Dflash`'s

`-Dflash` alone caps the window at 131,072 because that ceiling was measured on
`UD-Q4_K_XL`, where 147,456 loads and dies on the first real request. **NVFP4 is
5 GB smaller and 147,456 was measured working here**, finishing with 1,450 MiB.
Applying the Q4 cap to the NVFP4 path would silently serve a shallower window
than the one the evidence covers -- so the cap must not fire when `-Nvfp4` is on,
and nothing above 147,456 has been measured with this pairing either.
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
BATS = ["serve-dual-nvfp4-dflash.bat", "serve-dual-nvfp4-dflash-lan.bat"]
MIRROR = r"C:\AI\llama.cpp-mirror\build-mirror\bin\llama-server.exe"


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


def val(line, flag):
    m = re.search(re.escape(flag) + r" (\S+)", line)
    return m.group(1) if m else None


# ------------------------------------------------------------ it is allowed now

def test_the_combination_is_no_longer_refused():
    _, out = preview("-Nvfp4", "-Vision", "-Dflash")
    assert "already carries a drafter" not in out, out


def test_mtp_is_still_refused_with_nvfp4():
    """Only DFlash2 was measured. -Mtp on NVFP4 would be a SECOND copy of a head
    already in the file."""
    out = _whatif(PROFILE, "-Nvfp4", "-Vision", "-Mtp")
    assert "FATAL" in out, out


def test_dflash_and_mtp_together_are_still_refused():
    out = _whatif(PROFILE, "-Nvfp4", "-Vision", "-Dflash", "-Mtp")
    assert "FATAL" in out, out


# ------------------------------------------------- it serves the measured thing

def test_it_serves_the_dflash_pairing_not_mtp():
    line, _ = preview("-Nvfp4", "-Vision", "-Dflash")
    assert "draft-dflash,ngram-mod" in line, line
    assert "draft-mtp" not in line, line


def test_it_loads_the_nvfp4_artifact():
    line, _ = preview("-Nvfp4", "-Vision", "-Dflash")
    assert "NVFP4" in val(line, "-m"), line


def test_it_carries_the_drafter_file():
    line, _ = preview("-Nvfp4", "-Vision", "-Dflash")
    assert "-md" in line and "DFlash2" in val(line, "-md"), line
    assert val(line, "-ngld") == "99", line


def test_the_draft_depth_defaults_to_the_measured_four():
    line, _ = preview("-Nvfp4", "-Vision", "-Dflash")
    assert val(line, "--spec-draft-n-max") == "4", line


def test_dflashn_still_overrides_it():
    line, _ = preview("-Nvfp4", "-Vision", "-Dflash", "-DflashN", "2")
    assert val(line, "--spec-draft-n-max") == "2", line


def test_the_ngram_window_stays_nvfp4s_own():
    """24, not 12. 12 collapses on this artifact -- acceptance 55.4 -> 22.1 --
    and every NVFP4 figure was measured with 24."""
    line, _ = preview("-Nvfp4", "-Vision", "-Dflash")
    assert val(line, "--spec-ngram-mod-n-match") == "24", line


def test_it_runs_the_patched_mirror():
    line, _ = preview("-Nvfp4", "-Vision", "-Dflash")
    assert "llama.cpp-mirror" in line.split()[0], line


def test_it_keeps_the_tensor_split_with_a_ratio():
    line, _ = preview("-Nvfp4", "-Vision", "-Dflash")
    assert val(line, "-sm") == "tensor", line
    assert re.search(r"-ts \d+,\d+", line), line


# ----------------------------------- the Q4 ceiling must not leak onto this path

def test_the_window_is_not_capped_at_the_q4_ceiling():
    """131,072 is UD-Q4_K_XL's ceiling. NVFP4 is 5 GB smaller and 147,456 was
    measured working, finishing with 1,450 MiB. Applying the Q4 cap here would
    serve a shallower window than the evidence covers, silently."""
    line, _ = preview("-Nvfp4", "-Vision", "-Dflash")
    assert int(val(line, "-c")) > 131072, line


def test_it_does_not_exceed_the_depth_that_was_measured():
    """Nothing above 147,456 has been measured with this pairing."""
    line, _ = preview("-Nvfp4", "-Vision", "-Dflash", "-Deep")
    assert int(val(line, "-c")) <= 147456, line


# ---------------------------------------------- it says what it is, and is not

def test_the_preview_does_not_claim_a_speedup():
    _, out = preview("-Nvfp4", "-Vision", "-Dflash")
    low = out.lower()
    assert "0.7" in out or "stead" in low or "consist" in low, out
    assert "950" in out or "headroom" in low, out


# ---------------------------------------------------------------- launchers

@pytest.mark.parametrize("name", BATS)
def test_the_launcher_exists(name):
    assert os.path.exists(os.path.join(LAUNCHERS, name))


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_asks_for_both_switches(name):
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read()
    live = "\n".join(l for l in body.splitlines()
                     if not l.strip().upper().startswith("REM"))
    assert "-Nvfp4" in live and "-Dflash" in live, live


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_states_the_trade_rather_than_a_speedup(name):
    """Every launcher here carries its own evidence. This one's evidence is
    that it is NOT faster -- 4 % under the floor -- and costs 950 MiB."""
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read()
    assert "950" in body, body
    assert "0.7" in body and "9.3" in body, body


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
