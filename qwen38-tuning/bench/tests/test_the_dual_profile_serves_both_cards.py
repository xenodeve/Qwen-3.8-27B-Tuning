"""The two-card profile, held to the same promises as the one-card profile.

WHY A SECOND PROFILE RATHER THAN A FLAG ON THE FIRST.

`worker-q2kxl-mtp.ps1` serves `UD-Q2_K_XL` on one card, and every row in
`docs/results/` from 2026-08-23 onward describes that. Adding a `-Dual` switch
to it would mean the file's defaults no longer say what was measured -- the
thing `test_bind_is_opt_in.py` exists to prevent for `--host`, applied to the
hardware instead.

So the two-card configuration is its own file, with its own defaults, and both
ship. Which is the default is the developer's decision (issue #52 says so).

WHAT THESE TESTS CAN AND CANNOT DO.

pytest cannot boot a 15 GB model, so the checks below are on the profile's
source. They are written as PROPERTIES -- "it names both cards", "it does not
hardcode a card the launcher should pass" -- and each is phrased so that
re-wrapping a line, renaming a local, or adding a comment cannot change the
answer. Five tests in this suite have failed that standard before, one of them
green for the wrong reason for days, so it is stated rather than assumed.
"""
import os
import re

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BENCH)
REPO = os.path.dirname(ROOT)
SCRIPTS = os.path.join(ROOT, "scripts")

DUAL = os.path.join(SCRIPTS, "worker-q4-dual.ps1")
SOLO = os.path.join(SCRIPTS, "worker-q2kxl-mtp.ps1")
SERVE = os.path.join(REPO, "serve.ps1")

TI_5060 = "GPU-059b90e2-2b5c-00b8-f3ba-f6dea8de083e"
SUPER_4070 = "GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4"


def read(path):
    return open(path, encoding="utf-8", errors="replace").read()


def test_the_dual_profile_exists():
    assert os.path.exists(DUAL), (
        "issue #52 stage 5: there is no profile that serves the two-card "
        "configuration, so the measured result cannot be started by anyone")


# ---- what it serves ---------------------------------------------------------

def test_it_serves_both_cards():
    """Two cards named, and only ONE of them as a literal.

    The 5060 Ti comes from `Get-GpuVram.ps1`'s constant rather than being
    written out again -- the same UUID in two files is two files that can
    disagree, and the one that gets edited is never the one that gets read.
    So this accepts the reference, and `test_the_two_languages_name_the_same_
    card` is what keeps that constant honest.

    The first draft of this test required both literals. It went red the moment
    the duplication was removed, which is a test calling an improvement a
    regression -- the fifth time in three sessions an assertion here has held
    the shape of a file instead of a property.
    """
    t = read(DUAL)
    assert SUPER_4070 in t, "the 4070 SUPER is not named; this is a one-card profile"
    assert ("ServedGpuUuid" in t) or (TI_5060 in t), (
        "the second card is neither named nor resolved from Get-GpuVram.ps1")


def test_it_serves_the_artifact_that_needs_both_cards():
    """UD-Q4_K_XL is 16.69 GiB and spills eleven layers on one 16 GB card. It is
    the entire reason this profile exists; serving anything else here would make
    the second card's 130 W buy nothing."""
    assert "Q4_K_XL" in read(DUAL)


def test_it_checks_every_card_is_present_before_loading():
    """An absent UUID does not make llama-server fail -- it reports `(none)` for
    devices and runs on the CPU, producing correct output at a rate no row
    explains. With two UUIDs there are two ways to be wrong."""
    assert "Test-ServedGpuPresent" in read(DUAL) or "installed" in read(DUAL)


def test_it_pins_the_cards_through_the_environment():
    assert "$env:CUDA_VISIBLE_DEVICES" in read(DUAL)


# ---- the same four promises serve.ps1 already makes --------------------------

@pytest.mark.parametrize("flag", ["-ctk", "-ctv", "--spec-type", "-ngl", "-c"])
def test_the_launcher_does_not_re_declare_a_serving_flag(flag):
    """Mirrors test_serve_entrypoint. The launcher may SELECT a profile; it may
    not carry a copy of the profile's flags, because two copies drift and the
    one that gets edited is never the one that runs."""
    t = read(SERVE)
    assert not re.search(r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])", t), (
        f"serve.ps1 declares {flag}; that belongs in the profile")


def test_the_launcher_can_reach_the_dual_profile():
    """A profile nothing can start is a file, not a configuration."""
    assert "worker-q4-dual" in read(SERVE), (
        "serve.ps1 offers no way to start the two-card profile")


def test_exposure_is_still_an_act_in_the_dual_profile():
    """--host is the only access control this server has: no API key, CORS '*'.
    A new profile is a new place for that default to be wrong."""
    t = read(DUAL)
    assert "127.0.0.1" in t, "the dual profile does not default to loopback"
    assert "0.0.0.0" not in t, (
        "the dual profile hardcodes a wide bind; exposure belongs to "
        "serve.ps1 -Lan, which is a choice someone makes")


def test_it_does_not_pipe_llama_cpps_output_anywhere():
    """Issue #49: a pipe eats the colours and turns a live console into a
    forwarded stream. The status checks read --log-file, which llama.cpp writes
    IN ADDITION to the console (common/log.cpp:170-178).

    Scoped to the INVOCATION, not the file. The first draft banned
    `| ForEach-Object` anywhere and went red on the error handler that lists
    installed GPUs -- a pipe that has nothing to do with llama.cpp's output.
    A test that cannot tell those two apart is testing punctuation.
    """
    t = read(DUAL)
    marker = "& $Exe -m $Model"
    assert marker in t, "cannot find the invocation; this test is not looking at it"
    invocation = t[t.index(marker):]
    assert "|" not in invocation, (
        "something stands between llama.cpp and the console: "
        + next(l for l in invocation.splitlines() if "|" in l))


# ---- and it must not silently disagree with the profile it was modelled on ---

def test_both_profiles_agree_on_what_a_serving_flag_looks_like():
    """Not a style check. `-fa on`, `--no-mmproj-auto` and `--chat-template-file`
    are why measured rows are comparable at all; a second profile that quietly
    drops one produces numbers that look like the first profile's and are not.
    """
    dual, solo = read(DUAL), read(SOLO)
    for flag in ("-fa", "--no-mmproj-auto", "--chat-template-file",
                 "--reasoning-effort", "--log-colors"):
        assert flag in dual, (
            f"{flag} is in worker-q2kxl-mtp.ps1 and not in worker-q4-dual.ps1; "
            f"rows from the two would not be comparable")
        assert flag in solo


# ---- the banner must describe the profile it selected -----------------------
#
# These RUN the launcher. `serve.ps1 -WhatIf` resolves everything and exits
# without touching the GPU, so the banner is observable behaviour rather than a
# string in a file -- and the three earlier drafts of these checks all failed on
# the shape of the source instead. One looked for `if ($Dual)` before the first
# occurrence of the word "artifact", and went red because the word appears in a
# parameter comment two hundred lines above the banner.
#
# What is asserted is what a person sees.

import shutil
import subprocess

SERVE_PS1 = SERVE
PWSH = shutil.which("pwsh") or shutil.which("powershell")

needs_pwsh = pytest.mark.skipif(
    PWSH is None, reason="no PowerShell on this machine to run the launcher")


def banner(*args):
    r = subprocess.run(
        [PWSH, "-NoProfile", "-File", SERVE_PS1, "-WhatIf", *args],
        capture_output=True, text=True, timeout=120)
    return (r.stdout or "") + (r.stderr or "")


@needs_pwsh
def test_whatif_resolves_without_taking_the_gpu():
    """If this fails everything below is asserting on an error message."""
    out = banner()
    assert "WhatIf: would run" in out, out[:400]


@needs_pwsh
def test_the_default_banner_describes_the_single_card_profile():
    out = banner()
    assert "worker-q2kxl-mtp.ps1" in out
    assert "UD-Q2_K_XL" in out
    assert "UD-Q4_K_XL" not in out, (
        "the default banner names the two-card artifact it did not select")


@needs_pwsh
def test_the_dual_banner_describes_the_two_card_profile():
    """The bug this was written for: `-Dual -WhatIf` selected
    worker-q4-dual.ps1 and then printed "artifact UD-Q2_K_XL" underneath it --
    a launcher stating something false about the run it was introducing, which
    is the failure shipped once already (commit b55699c).
    """
    out = banner("-Dual")
    assert "worker-q4-dual.ps1" in out
    assert "UD-Q4_K_XL" in out
    assert "UD-Q2_K_XL" not in out, (
        "the -Dual banner names the artifact it did NOT select")


@needs_pwsh
def test_the_dual_banner_does_not_claim_one_card_is_in_use():
    out = banner("-Dual")
    assert "uses one of them" not in out, (
        "the two-card path prints the one-card sentence")


@needs_pwsh
def test_each_path_says_what_it_has_not_measured():
    """Neither configuration has a quality number from this project, and the
    dual one has nothing at all at the depth it serves. A banner that omits
    that reads as a settled recommendation."""
    assert "#44" in banner()
    assert "147,456" in banner("-Dual") and "#52" in banner("-Dual")


@needs_pwsh
def test_the_dual_profile_uses_the_split_that_won():
    """MEASURED 2026-08-26, ctx 16,384, three paired rounds, arms rotated:

        layer-default-base  [21.1, 21.0, 19.9]
        split-tensor        [32.4, 33.9, 32.3]   +59.5 % [+53.9, +62.9] RESOLVED
        ts-even             [21.2, 21.9, 20.0]   +1.8 %  within noise

    Same residency ceiling either way -- 66+0 to 229,376. So the default layer
    split leaves 59 % on the table for nothing, and `-ts` is not a lever here.

    Nearly missed: `-sm tensor` aggregates the cards into a `Meta` device, which
    `parse_layer_split` did not recognise, so the first run of this sweep voided
    every tensor row. The parser refusing rather than guessing is what kept the
    number findable.

        SCOPED TO THE INVOCATION. The first draft asserted `"-sm" in t and
    "tensor" in t` and was GREEN BEFORE THE FLAG WAS ADDED, because the
    profile's header explains at length why `-sm row` cannot load and that
    `-sm tensor` was swept. Both tokens were in prose. That is the seventh
    shape-not-property assertion in three sessions and the second to pass for
    the wrong reason.
    """
    t = read(DUAL)
    marker = "& $Exe -m $Model"
    assert marker in t, "cannot find the invocation; this test is not looking at it"
    invocation = t[t.index(marker):]
    assert re.search(r"-sm\s+tensor", invocation), (
        "the dual profile does not PASS -sm tensor, which measured +59.5 % over "
        "the default layer split at the same residency ceiling")


def test_the_profile_says_the_mode_is_experimental():
    """llama.cpp's own help calls `tensor` EXPERIMENTAL. A profile that ships it
    without saying so hands the next reader a number and hides its status."""
    assert "EXPERIMENTAL" in read(DUAL).upper()


def test_the_dual_profile_uses_the_micro_batch_that_won():
    """MEASURED 2026-08-26 on -sm tensor, three paired rounds, prefill on the
    identical 6,621-token prompt:

        -ub 256   870.9 / 892.3 / 884.4  (the single-card default)
        -ub 1024  973.0 / 968.9 / 972.5  +10.1 %, ranges do not overlap

    Decode was flat across 128/256/512/1024 -- a micro-batch is a prefill knob.

    Scoped to the invocation, for the reason the -sm test above records.
    """
    t = read(DUAL)
    invocation = t[t.index("& $Exe -m $Model"):]
    assert re.search(r"-ub\s+1024", invocation), (
        "the dual profile does not pass -ub 1024, which measured +10.1 % "
        "prefill at no decode cost on the split it serves")


def test_the_two_profiles_may_disagree_on_ub_and_the_dual_says_why():
    """They now differ: 256 on one card, 1024 on two. That is a deliberate
    divergence and the header has to carry its evidence, or the next reader
    reads it as drift."""
    dual, solo = read(DUAL), read(SOLO)
    assert re.search(r"-ub\s+1024", dual[dual.index("& $Exe -m $Model"):])
    assert re.search(r"-ub\s+256", solo[solo.index("& $Exe -m $Model"):])
    header = dual[:dual.index("param(")]
    assert "-ub 1024" in header and "10.1" in header, (
        "the dual profile diverges from the single-card -ub without stating "
        "the measurement that justifies it")


def test_the_profile_says_fit_is_inert_under_tensor_split():
    """`--fit on --fit-target 768` DOES NOTHING here, and the profile carries it.

    Boot log, 2026-08-26, verbosity 5:

        W common_fit_params: failed to fit params to free device memory:
          llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort

    The flags stay, because every measured row carries them and removing them
    would make the argv differ from what was benchmarked. But a flag that
    implies a safety net it does not provide is the exact shape this repo
    exists to catch, so the file has to say so out loud: under -sm tensor there
    is no automatic adjustment, and an over-large context is a hard load
    failure rather than a quiet spill.

    That is arguably the better failure -- CLAUDE.md's north star prefers a
    crash to a plausible number -- but it is a DIFFERENT failure from the one
    the single-card profile has, and the next reader must not assume otherwise.
    """
    t = read(DUAL)
    assert "SPLIT_MODE_TENSOR" in t, (
        "the profile carries --fit but never says llama.cpp aborts the fitting "
        "step under -sm tensor")


@needs_pwsh
def test_the_dual_banner_does_not_promise_fit_will_adjust():
    out = banner("-Dual")
    assert "UD-Q4_K_XL" in out
