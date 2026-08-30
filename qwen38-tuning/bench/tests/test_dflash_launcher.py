r"""A fifth and sixth icon: DFlash2 on two cards, and it is NOT the deep one.

WHY IT IS A SEPARATE PAIR (issue #52, 2026-08-27). On the tensor split with a
patched llama.cpp, `draft-dflash,ngram-mod` measured **+123.8 %** [+121.9,
+125.1] over the served `ngram-mod` at ctx 65,536 -- more than double the
decode. It cannot be a flag on the existing dual launchers because it changes
three things at once that the developer has to choose deliberately:

  1. A DIFFERENT BINARY. It needs C:\AI\llama.cpp-mirror, built from a LOCAL
     PATCH that mirrors the target's output projection. Unpatched, the arm
     aborts at ggml-backend-meta.cpp:543 -- TOP_K cannot take axis-0 logits.
     The patch is reviewed by nobody outside this project.
  2. A SHALLOWER WINDOW. The measured ceiling is 131,072 against the ~250,000
     the `-MaxCtx` launchers choose. 147,456 LOADS AND THEN DIES on the first
     real request, which is why this pair must NOT carry -MaxCtx: "the deepest
     that fits" is the wrong question when the depth above the answer passes a
     health check.
  3. ALMOST ALL THE HEADROOM. At 131,072 it finishes with 634 / 530 MiB against
     the served configuration's ~2,210. The measured line in this project is
     336 MiB died and 488 survived.

THE ASSERTIONS HERE READ RESOLVED VALUES, NOT FILE TEXT. `worker-q4-dual.ps1`
gained `-WhatIf`, which resolves the split, the window and the whole argv and
then exits without launching. Checking that the file CONTAINS "131072" would
pass on a comment; checking that the argv carries `-c 131072` is the property.
Eight assertions in this suite have measured the shape of a file instead
(`docs/agents/traps.md` 16).
"""
import os
import re
import subprocess

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
LAUNCHERS = os.path.join(ROOT, "launchers")
DFLASH = os.path.join(LAUNCHERS, "serve-dual-dflash.bat")
DFLASH_LAN = os.path.join(LAUNCHERS, "serve-dual-dflash-lan.bat")
BOTH = [DFLASH, DFLASH_LAN]
SERVE = os.path.join(ROOT, "serve.ps1")


def read(path):
    with open(path, "rb") as fh:
        return fh.read().decode("ascii")


# ------------------------------------------------------------ the .bat files

@pytest.mark.parametrize("path", BOTH)
def test_the_launcher_exists_at_the_root(path):
    assert os.path.exists(path), path + " is missing"


@pytest.mark.parametrize("path", BOTH)
def test_it_anchors_on_its_own_folder(path):
    """%CD% is not the .bat's folder when it is double-clicked from elsewhere.

    Asserted as a path that RESOLVES rather than as the literal
    `%~dp0serve.ps1`: the launchers moved into `launchers/` on 2026-08-29 and
    now climb one level, and the old string check called that a regression
    while the file it points at was right there.
    """
    line = next(l for l in read(path).splitlines()
                if "serve.ps1" in l and "%~dp0" in l
                and not l.strip().lower().startswith("rem"))
    rel = line[line.index("%~dp0") + len("%~dp0"):].split('"')[0]
    target = os.path.normpath(
        os.path.join(os.path.dirname(path), rel.replace("\\", os.sep)))
    assert os.path.exists(target), "%s points at %s" % (path, target)


@pytest.mark.parametrize("path", BOTH)
def test_it_keeps_the_window_open_on_failure(path):
    assert "pause" in read(path).lower()


@pytest.mark.parametrize("path", BOTH)
def test_it_survives_a_restricted_execution_policy(path):
    assert "ExecutionPolicy Bypass" in read(path)


@pytest.mark.parametrize("path", BOTH)
def test_it_asks_for_both_cards_and_the_drafter(path):
    t = read(path)
    assert "-Dual" in t, path
    assert "-Dflash" in t, path


@pytest.mark.parametrize("path", BOTH)
def test_it_does_NOT_ask_for_the_deepest_window(path):
    """-MaxCtx would choose 147,456 here, which loads and then dies."""
    assert "-MaxCtx" not in read(path), (
        path + " carries -MaxCtx; the DFlash ceiling is 131,072 and the rung "
        "above it passes a health check before dying on the first request")


def test_only_the_lan_one_exposes():
    assert "-Lan" not in read(DFLASH)
    assert "-Lan" in read(DFLASH_LAN)


@pytest.mark.parametrize("path", BOTH)
def test_it_is_readable_by_cmd(path):
    raw = open(path, "rb").read()
    raw.decode("ascii")
    assert not raw.startswith(b"\xef\xbb\xbf"), "a BOM makes cmd choke"
    assert b"\r\n" in raw


# --------------------------------------------- what the switch actually does

def _whatif(*args):
    """serve.ps1 resolves everything and exits without touching the GPU."""
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", SERVE, "-WhatIf"] + list(args),
        capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr


def test_the_switch_reaches_the_two_card_profile():
    out = _whatif("-Dual", "-Dflash")
    assert "worker-q4-dual.ps1" in out, out
    assert re.search(r"-Dflash\s+True", out), out


def test_dflash_without_dual_is_refused():
    """The patch and the ceiling are both two-card facts."""
    out = _whatif("-Dflash")
    assert "FATAL" in out, out


def _profile_whatif(*args):
    """The PROFILE's own dry run: resolves the split, the window and the argv."""
    profile = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q4-dual.ps1")
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", profile, "-WhatIf"] + list(args),
        capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr


def test_the_window_is_capped_at_the_measured_ceiling():
    out = _profile_whatif("-Dflash")
    assert re.search(r"-c\s+131072", out), out


def test_it_uses_the_patched_binary_and_says_so():
    out = _profile_whatif("-Dflash")
    assert "llama.cpp-mirror" in out, out


def test_it_serves_the_pairing_and_not_the_drafter_alone():
    """draft-dflash alone was +19.4 %; paired with ngram-mod it is +123.8 %."""
    out = _profile_whatif("-Dflash")
    assert "draft-dflash,ngram-mod" in out, out


def test_the_micro_batch_is_halved_because_the_memory_is_needed():
    out = _profile_whatif("-Dflash")
    assert re.search(r"-ub\s+512", out), out


def test_the_default_dual_profile_is_untouched_by_any_of_this():
    """The switch must be opt-in: no -Dflash means the served configuration."""
    out = _profile_whatif()
    assert "llama.cpp-mirror" not in out, out
    assert "draft-dflash" not in out, out
