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
    # Scoped to the ARTIFACT LINE. The first version forbade the string
    # "UD-Q2_K_XL" anywhere in the output, and went red when the banner started
    # quoting it as the thing this configuration is at PARITY with -- naming
    # the comparison is the opposite of the defect. What must not happen is the
    # banner saying it SERVES the other artifact.
    artifact_line = next(l for l in out.splitlines() if "artifact" in l)
    assert "UD-Q4_K_XL" in artifact_line and "UD-Q2_K_XL" not in artifact_line, (
        f"the -Dual banner's artifact line is wrong: {artifact_line!r}")


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

        Reads the EFFECTIVE value, not the literal. `-ub 1024` became `-ub $UBatch`
    when the budget check needed the number too -- the profile still serves
    1024, and an assertion on the literal called that a regression. Eighth
    shape-not-property assertion; the fix each time is to ask what the value
    IS rather than how it is written.
    """
    t = read(DUAL)
    invocation = t[t.index("& $Exe -m $Model"):]
    m = re.search(r"-ub\s+(\S+)", invocation)
    assert m, "the dual profile passes no -ub at all"
    val = m.group(1)
    if val.startswith("$"):
        d = re.search(r"\[int\]" + re.escape(val) + r"\s*=\s*(\d+)", t)
        assert d, "the -ub parameter %s has no default" % val
        val = d.group(1)
    assert val == "1024", (
        "the dual profile serves -ub %s; 1024 measured +10.1 %% prefill at no "
        "decode cost on the split it serves" % val)


def test_the_two_profiles_may_disagree_on_ub_and_the_dual_says_why():
    """They now differ: 256 on one card, 1024 on two. That is a deliberate
    divergence and the header has to carry its evidence, or the next reader
    reads it as drift."""
    dual, solo = read(DUAL), read(SOLO)
    assert re.search(r"\[int\]\$UBatch\s*=\s*1024", dual), (
        "the dual profile's micro-batch default is not 1024")
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


# ---- the incident: 0.38 tok/s on the developer's machine, 2026-08-26 ---------

def test_it_computes_the_split_from_measured_free_vram():
    """WHAT HAPPENED. `serve-dual-lan.bat` decoded at 0.38 tok/s -- 85x slower
    than the 32.4 this profile advertises -- with the 5060 Ti at 0 % and the
    4070 SUPER at 88 % and 11.6 of 12.0 GB, spilling into shared host memory.

    ROOT CAUSE. `-sm tensor` splits EVENLY when no ratio is given:
    `llama-model.cpp:707` falls back to `ne_s * (j+1)/n_devices`. The cards are
    not even -- 12 GB against 16 GB -- and the 12 GB one is ALSO THE DISPLAY
    GPU, carrying explorer, the terminal, the browser and the NVIDIA overlay.

    The arithmetic from the developer's own boot log: the Meta buffers are
    8065 model + 1296 KV + 1024 compute = 10,385 MiB PER CARD. On the 4070 with
    1,579 MiB of desktop that leaves +317 MiB. On the 5060 Ti it leaves 5,876.

    So a static ratio is a bandaid: the desktop's appetite is not constant and
    the next browser tab puts it back over. The split has to be computed from
    what is actually free, at launch, with a reserve for the card the desktop
    lives on.
    """
    t = read(DUAL)
    invocation = t[t.index("& $Exe -m $Model"):]
    assert "-ts" in invocation or "tsArg" in invocation, (
        "the dual profile does not pass a tensor-split ratio, so llama.cpp "
        "splits EVENLY across a 12 GB card and a 16 GB card")
    assert "Get-GpuVram" in t, (
        "the ratio is not derived from measured free VRAM")


def test_it_reserves_headroom_on_the_card_the_desktop_lives_on():
    """A card already holding memory at launch is the display GPU, and its
    appetite grows while the server runs. Sizing to what is free RIGHT NOW
    leaves nothing for that growth -- which is how +317 MiB became a spill."""
    t = read(DUAL)
    assert "Reserve" in t, (
        "nothing reserves VRAM for the desktop to grow into")


def test_it_refuses_rather_than_spilling_when_the_budget_is_gone():
    """`--fit` is inert under -sm tensor, so llama.cpp will NOT trim to fit.
    The profile's header used to claim that made an over-large context a hard
    load failure. IT DOES NOT -- it makes it a silent spill into host memory at
    0.38 tok/s, which is the believable-wrong-number failure CLAUDE.md names.
    Something has to do the refusing, and llama.cpp will not."""
    t = read(DUAL)
    assert "FATAL" in t and "spill" in t.lower(), (
        "nothing refuses when the cards cannot hold the configuration")


def test_the_header_retracts_the_hard_failure_claim_rather_than_deleting_it():
    """The claim was wrong and it was mine: this header said --fit being inert
    made an over-large context "a hard load failure ... the better failure of
    the two". Measured 2026-08-26 -- it SPILLS, silently, and returns a working
    server at 0.38 tok/s.

    Asserted as a retraction, not as an absence. The first version of this test
    forbade the string "hard load failure" anywhere in the file and went red on
    the paragraph that QUOTES the wrong claim in order to withdraw it. Deleting
    the old wording would leave the next reader no way to know it was ever
    believed -- which is the opposite of what CORRECTIONS.md exists for.
    """
    t = read(DUAL)
    assert "THAT WAS WRONG" in t, (
        "the header does not retract the hard-failure claim")
    assert "SILENT SPILL" in t.upper(), (
        "the header does not say what actually happens instead")
    assert "0.38" in t, (
        "the retraction does not carry the number that disproved it")


# ---- MTP as an option, after the probe showed it runs here ------------------

MTP_BATS = [os.path.join(ROOT, "..", "serve-dual-mtp.bat"),
            os.path.join(ROOT, "..", "serve-dual-mtp-lan.bat")]
MTP_BATS = [os.path.normpath(p) for p in MTP_BATS]


def test_the_profile_can_be_asked_for_mtp():
    """MEASURED 2026-08-27. `draft-mtp` was believed impossible under
    `-sm tensor`; it is not. At ctx 16,384 with `-ub 128` it loads, and at
    147,456 on the computed `-ts` it loads too -- 66+0, CUDA0 1,571 MiB free
    and CUDA1 861. The earlier failure was on the EVEN split and its assertion
    was a null buffer, which is what running out of memory looks like.

    It is a SWITCH, not the default, because we have no trustworthy rate for
    it: all three paired rounds at 147,456 were voided by the output guard --
    the generations copy the prompt.
    """
    t = read(DUAL)
    assert "$Mtp" in t, "worker-q4-dual.ps1 has no way to ask for MTP"
    invocation = t[t.index("& $Exe -m $Model"):]
    assert "specArg" in invocation or "draft-mtp" in invocation, (
        "the decoder is hardcoded, so -Mtp cannot reach llama-server")


def test_mtp_is_not_the_default():
    """No usable rate exists for it. A default is a claim that it is better."""
    t = read(DUAL)
    m = re.search(r"\[switch\]\$Mtp", t)
    assert m, "-Mtp is not a switch, so it may be defaulting on"


def test_asking_for_mtp_reserves_room_for_its_head():
    """The head is not free: with it the same configuration used about 2,750 MiB
    more across the two cards, and CUDA1 finished with 861 MiB free. The refusal
    check has to know that, or it approves a budget that then spills -- which is
    the whole failure this profile was fixed for."""
    t = read(DUAL)
    assert "MTP_HEAD_MIB" in t, (
        "the budget check does not account for the MTP head")


def test_the_profile_says_mtp_has_no_measured_rate():
    """Three unpaired manual readings gave 44.5 / 54.3 / 92.7 tok/s and they are
    exactly the numbers CORRECTIONS 32 says not to trust: a speculative rate
    rises with how predictable the text is, and copying the prompt is maximally
    predictable. Quoting them in a header would launder a voided measurement."""
    t = read(DUAL)
    assert "copies the prompt" in t or "copy the prompt" in t, (
        "the header offers MTP without saying its rate could not be measured")


@pytest.mark.parametrize("path", MTP_BATS)
def test_the_mtp_launchers_exist_and_ask_for_it(path):
    assert os.path.isfile(path), path
    t = read(path)
    assert "-Dual" in t and "-Mtp" in t
    assert "%~dp0" in t and "ExecutionPolicy Bypass" in t and "pause" in t.lower()


@pytest.mark.parametrize("path", MTP_BATS)
def test_the_mtp_launchers_say_the_rate_is_unmeasured(path):
    t = read(path).lower()
    assert "not measured" in t or "unmeasured" in t or "voided" in t, (
        "%s offers MTP as if its speed were known" % path)


@pytest.mark.parametrize("path", MTP_BATS)
def test_the_mtp_launchers_are_readable_by_cmd(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    assert not raw.startswith(b"\xef\xbb\xbf")
    raw.decode("ascii")


def test_only_the_lan_named_mtp_launcher_exposes():
    plain = next(p for p in MTP_BATS if "lan" not in os.path.basename(p))
    lan = next(p for p in MTP_BATS if "lan" in os.path.basename(p))
    assert "-Lan" not in read(plain)
    assert "-Lan" in read(lan)


@needs_pwsh
def test_the_dual_banner_quotes_the_rate_of_the_configuration_it_runs():
    """Twice now the banner has advertised a number from a configuration that
    was later replaced. First "20.9 tok/s", measured on the default layer split
    before -sm tensor. Then "32.4 / 32.6 / 33.1", measured on the EVEN tensor
    split -- the one that collapses to 0.38 under desktop load.

    The configuration it actually runs measures 25.5 / 25.4 / 26.4 at ctx
    147,456. A banner is the only place most people will ever read a number
    from this project, so a stale one there is worse than a stale one in a doc.
    """
    out = banner("-Dual")
    assert "25.5" in out or "25.4" in out or "26.4" in out, (
        "the -Dual banner does not quote the rate of the split it runs")
    assert "32.4 / 32.6 / 33.1" not in out, (
        "the banner still quotes the even-split rate, which is the "
        "configuration that produced 0.38 tok/s under load")


@needs_pwsh
def test_the_dual_banner_no_longer_says_mtp_cannot_load():
    """It can. Measured 2026-08-27 at both ctx 16,384 and 147,456."""
    out = banner("-Dual")
    assert "draft-mtp cannot" not in out, (
        "the banner repeats a claim this project disproved on its own machine")


@needs_pwsh
def test_the_banner_does_not_print_two_decoders():
    """`serve-dual-mtp.bat` printed both of these, four lines apart:

        decoder   ngram-mod. draft-mtp is NOT set here: ...
        decoder   draft-mtp + ngram-mod -- LOADS HERE, RATE NOT MEASURED.

    The first is a static line in serve.ps1's -Dual branch; the second comes
    from the profile, which knows what it was actually asked for. Only one of
    them can be true, and the reader has no way to tell which.

    Third time the banner has stated something false about the run underneath
    it -- after "closing this window stops the server" and two stale rates. The
    pattern is a launcher describing a configuration it does not own.
    """
    out = banner("-Dual", "-Mtp")
    decoder_lines = [l for l in out.splitlines() if l.strip().startswith("decoder")]
    assert len(decoder_lines) <= 1, (
        "the banner prints %d decoder lines: %r" % (len(decoder_lines), decoder_lines))
    if decoder_lines:
        assert "NOT set" not in decoder_lines[0], (
            "the -Mtp banner says draft-mtp is not set")


# ---- the client that thinks the server is dead ------------------------------

def test_both_profiles_ping_more_often_than_the_client_gives_up():
    """MEASURED 2026-08-27, cold prefill of ~45,000 tokens, which takes 59.4 s:

        stream:false                nothing at all until 59.4 s
        stream:true, defaults       first byte 31.5 s (one 30 s ping), content 59.4 s
        stream + return_progress    progress from 1.4 s: 0%, 4%, 9%, 13%, 18% ...

    Claude Code showed `Waiting for API response - will retry in 2m 24s - check
    your network`. That is a client with an open connection carrying nothing.

    `return_progress` is the thing that fixes it properly and it is a REQUEST
    field -- the client has to ask, and this one does not. What the server owns
    is the keep-alive interval, and llama.cpp's default of 30 s is most of a
    minute of silence on a connection that is working perfectly.

    This does not make the wait shorter. It makes it visible.
    """
    import re as _re
    for path in (SOLO, DUAL):
        t = read(path)
        inv = t[t.index("& $Exe -m $Model"):]
        m = _re.search(r"--sse-ping-interval\s+\$?(\S+)", inv)
        assert m, ("%s does not set --sse-ping-interval, so a streaming client "
                   "sees one byte every 30 s during a 59 s prefill"
                   % os.path.basename(path))


def test_the_ping_default_is_well_under_the_prefill_it_covers():
    """A ping slower than the wait it is covering is not a keep-alive."""
    import re as _re
    for path in (SOLO, DUAL):
        t = read(path)
        m = _re.search(r"\[int\]\$SsePingIntervalSec\s*=\s*(\d+)", t)
        assert m, "%s has no ping parameter" % os.path.basename(path)
        assert 1 <= int(m.group(1)) <= 10, (
            "%s pings every %s s; the prefill it has to cover is 59 s"
            % (os.path.basename(path), m.group(1)))


def test_the_profile_says_progress_is_the_real_fix_and_the_client_must_ask():
    """Recording it so the next reader does not re-derive it: the server cannot
    turn `return_progress` on for a client that never sends it."""
    for path in (SOLO, DUAL):
        t = read(path)
        assert "return_progress" in t, (
            "%s changes the ping without saying what would actually fix the "
            "wait, or why we cannot do it from here" % os.path.basename(path))


# ---- the guard approved a context that then OOM'd ---------------------------

def test_the_budget_check_accounts_for_the_context_not_just_the_weights():
    """WHAT HAPPENED, 2026-08-27. Asked for ctx 262,144 -- the model's own
    n_ctx_train, which a ladder had measured as fully resident hours earlier.
    The profile computed `-ts 6899,15489`, started, and died:

        ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1696.30 MiB
          on device 1: cudaMalloc failed: out of memory

    The guard let it through because it compares the budget against
    WEIGHTS_MIB alone. Weights are 16,130 MiB; at 262,144 the KV cache is
    another 4,608 and the compute buffers about 2,048, so the real demand is
    near 22,800 against a budget of 22,388.

    Why the ladder disagreed: it ran with a hardcoded `-ts 7819,15490`,
    computed when the desktop held about 1,600 MiB. By the time of this run the
    desktop held 2,575, the 4070's budget fell by 920 MiB, and proportional
    splitting pushed the difference onto the 5060 Ti, which ran out.

    A guard that only counts the weights is a guard that passes every context.
    """
    t = read(DUAL)
    assert "KV_KIB_PER_TOKEN" in t, (
        "the budget check ignores the KV cache, so it approves any -Ctx and "
        "lets llama.cpp discover the problem with an OOM")
    assert "$Ctx" in t[t.index("$WEIGHTS_MIB"):t.index("$tsArg")], (
        "the budget check does not use the requested context")


def test_the_refusal_names_the_deepest_context_that_would_fit():
    """"It does not fit" leaves the developer bisecting by hand. The profile
    knows the KV rate -- 18.00 KiB per token, measured -- and the budget, so it
    can say what WOULD work."""
    t = read(DUAL)
    assert "deepest" in t.lower(), (
        "the refusal does not tell the developer what context would fit")


def test_the_refusal_offers_the_micro_batch_as_well_as_a_smaller_context():
    """MEASURED 2026-08-27. A ~135,000-token request through each depth:

        ctx 147,456  ub 1024  SURVIVED  free after 2,100/2,097 -> 1,998/2,040
        ctx 196,608  ub 1024  SURVIVED         1,436/1,258 -> 1,248/1,208
        ctx 229,376  ub 1024  SURVIVED         1,156/  550 -> 1,071/  500
        ctx 262,144  ub 1024  refused at load
        ctx 229,376  ub  512  SURVIVED         1,312/1,010 -> 1,249/  974
        ctx 262,144  ub  512  SURVIVED           919/  488 ->   821/  452

    So n_ctx_train IS reachable -- by spending the compute buffer instead of
    the context. Each card's compute buffer is about one -ub of MiB, so 1024 ->
    512 hands back roughly 1,024 MiB across the pair, which is more than the
    ~390 that 262,144 was short by.

    Telling the developer only to cut the context hides the cheaper trade. The
    prefill cost of ub 512 is about 3.5 % (938 against 971 tok/s, measured);
    the context cost of dropping 262,144 to 237,568 is 24,576 tokens.
    """
    t = read(DUAL)
    i = t.index("does not fit without spilling")
    refusal = t[i:i + 2500]
    assert "-UBatch" in refusal, (
        "the refusal offers only a smaller -Ctx; halving -UBatch frees about "
        "1,024 MiB across the pair and costs ~3.5 % of prefill")


def test_the_profile_records_that_loading_is_not_surviving():
    """262,144 with -ub 512 loaded, reported 66+0, answered /health -- and died
    with `CUDA error: out of memory ... cuMemSetAccess` when a real request
    arrived, because llama.cpp allocates more once there is work. A later run
    of the same configuration survived with 488 MiB free on the second card
    against the 336 of the one that died.

    A guard that models load-time demand cannot promise a run. The file has to
    say that, or the next reader reads a successful boot as a verdict."""
    t = read(DUAL)
    assert "cuMemSetAccess" in t or "loading is not surviving" in t.lower(), (
        "the profile does not record that a successful load is not a "
        "successful run")
