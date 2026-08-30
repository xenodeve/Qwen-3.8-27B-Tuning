r"""A seventh and eighth icon: NVFP4 with its baked-in MTP head.

WHY IT IS ITS OWN PAIR (issue #50, 2026-08-29). Measured +63.1 % [+58.3, +65.6]
RESOLVED over the served configuration at ctx 147,456 -- 39.4 / 42.6 / 42.6
against 24.9 / 25.7 / 25.7, three paired rounds rotated, baseline spread 3.3 %
(`results/nvfp4-final-147456.jsonl`). It is the fastest thing measured here.

Unlike the DFlash2 pair it costs NO patch, NO sidecar drafter and NO unreviewed
binary -- the MTP head is inside the model file and it runs on
`llama.cpp-blackwell`, the binary already served. It also finishes with MORE
headroom than the incumbent.

WHAT IT DOES CHANGE IS THE MODEL FILE, and that is why it cannot be a default:
quality has never been measured on this project's own artifacts, and `ngram-mod`
acceptance falls 55.4 -> 22.1 on this artifact, which is direct evidence it
writes differently rather than merely faster. This pair exists so the developer
can try it on real work; nothing else changes.

TWO THINGS THE PROFILE MUST GET RIGHT, AND BOTH ARE MEASURED:

  1. THE N-GRAM IS RETUNED. Every other profile serves `n-match 12`, which won
     on `UD-Q4_K_XL`. On NVFP4 that value is worth 32.4-36.5 tok/s and `24` is
     worth 42.9-43.1 -- +27.1 % RESOLVED -- and `24` is the value that LOST on
     the other artifact at this exact depth. A verdict does not transfer across
     artifacts. Serving 12 here would silently give away a third of the gain.
  2. THE CEILING IS 229,376, not `n_ctx_train`. 262,144 does not come up;
     229,376 survived a 65,643-token request with 846/526 MiB free. As with the
     DFlash pair, `-MaxCtx` must not be used: "the deepest that fits" is the
     wrong question when the rung above the answer can pass a health check.

ASSERTIONS READ RESOLVED VALUES, NOT FILE TEXT -- `worker-q4-dual.ps1` has
`-WhatIf`. Checking the file CONTAINS "24" would pass on a comment.
"""
import os
import re
import subprocess

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
LAUNCHERS = os.path.join(ROOT, "launchers")
NVFP4 = os.path.join(LAUNCHERS, "serve-dual-nvfp4.bat")
NVFP4_LAN = os.path.join(LAUNCHERS, "serve-dual-nvfp4-lan.bat")
BOTH = [NVFP4, NVFP4_LAN]
SERVE = os.path.join(ROOT, "serve.ps1")
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q4-dual.ps1")


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
def test_it_asks_for_both_cards_and_the_artifact(path):
    t = read(path)
    assert "-Dual" in t, path
    assert "-Nvfp4" in t, path


@pytest.mark.parametrize("path", BOTH)
def test_it_does_NOT_ask_for_the_deepest_window(path):
    """200,704 is the measured ceiling; the rungs above it die on a real
    half-window request (CORRECTIONS 35)."""
    assert "-MaxCtx" not in read(path), (
        path + " carries -MaxCtx; the NVFP4 ceiling is 200,704")


def test_only_the_lan_one_exposes():
    assert "-Lan" not in read(NVFP4)
    assert "-Lan" in read(NVFP4_LAN)


@pytest.mark.parametrize("path", BOTH)
def test_it_is_readable_by_cmd(path):
    raw = open(path, "rb").read()
    raw.decode("ascii")
    assert not raw.startswith(b"\xef\xbb\xbf"), "a BOM makes cmd choke"
    assert b"\r\n" in raw


# --------------------------------------------- what the switch actually does

def _whatif(script, *args):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", script, "-WhatIf"] + list(args),
        capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr


def test_the_switch_reaches_the_two_card_profile():
    out = _whatif(SERVE, "-Dual", "-Nvfp4")
    assert "worker-q4-dual.ps1" in out, out
    assert re.search(r"-Nvfp4\s+True", out), out


def test_nvfp4_without_dual_is_refused():
    """The split and the budget are both two-card facts."""
    out = _whatif(SERVE, "-Nvfp4")
    assert "FATAL" in out, out


def test_it_serves_the_nvfp4_file_and_not_the_q4_one():
    out = _whatif(PROFILE, "-Nvfp4")
    assert "NVFP4-MTP-VERY-LOW.gguf" in out, out
    assert "UD-Q4_K_XL.gguf" not in out, out


def test_it_serves_the_baked_in_mtp_head_beside_the_ngram():
    out = _whatif(PROFILE, "-Nvfp4")
    assert re.search(r"--spec-type\s+draft-mtp,ngram-mod", out), out


def test_the_ngram_is_RETUNED_for_this_artifact():
    """n-match 12 won on UD-Q4_K_XL and gives away +27.1 % here."""
    out = _whatif(PROFILE, "-Nvfp4")
    assert re.search(r"--spec-ngram-mod-n-match\s+24", out), out
    assert not re.search(r"--spec-ngram-mod-n-match\s+12", out), out


def test_the_incumbent_still_serves_n_match_12():
    """The retune must not leak onto the artifact it LOST on."""
    out = _whatif(PROFILE)
    assert re.search(r"--spec-ngram-mod-n-match\s+12", out), out


def test_it_uses_the_SERVED_binary_and_not_the_patched_one():
    out = _whatif(PROFILE, "-Nvfp4")
    assert "llama.cpp-blackwell" in out, out
    assert "llama.cpp-mirror" not in out, out


def test_the_window_is_capped_at_the_measured_ceiling():
    out = _whatif(PROFILE, "-Nvfp4", "-Ctx", "262144")
    assert re.search(r"-c\s+200704", out), out


def test_it_refuses_a_SECOND_mtp_head():
    """`draft-mtp` is already inside this file; asking for it again is asking
    for a second copy of what is there."""
    out = _whatif(PROFILE, "-Nvfp4", "-Mtp")
    assert "FATAL" in out, out


def test_dflash_is_ALLOWED_here_since_it_was_measured():
    """CHANGED 2026-08-30, and the change is the point.

    This assertion used to require FATAL for `-Dflash` too, and that was right
    while the only evidence was +0.2 % with the sign flipping. That run gave
    DFlash2 ctx 147,456 against its measured best of 65,536, `n_max` 3 against
    4, and `n-match` 12 -- a window this project records collapsing on this
    very artifact, acceptance 55.4 -> 22.1.

    Re-measured: +67.9 % [+65.8, +71.5] RESOLVED over `ngram-mod` at 65,536,
    and 44.48 / 44.56 / 44.23 at 147,456 against the head's pooled 42.77.
    `results/nvfp4-dflash-65536.jsonl`, `results/nvfp4-dflash-147456-n4.jsonl`,
    issue #50. The refusal was encoding a handicapped measurement.

    It is allowed, not preferred: `serve-dual-nvfp4.bat` still serves the head,
    and the launcher for this one says in its own text that it is NOT faster.
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Dflash")
    assert "FATAL" not in out, out
    assert "draft-dflash,ngram-mod" in out, out


def test_both_drafters_at_once_are_still_refused():
    out = _whatif(PROFILE, "-Nvfp4", "-Dflash", "-Mtp")
    assert "FATAL" in out, out


def test_maxctx_with_nvfp4_is_refused():
    out = _whatif(PROFILE, "-Nvfp4", "-MaxCtx")
    assert "FATAL" in out, out


def test_the_alias_names_the_artifact_that_is_actually_serving():
    """--alias is what a client sees as the model name.

    Caught by reading the dry run rather than the test list: the profile
    hardcoded `Qwen3.8-27B-Q4_K_XL` and kept it while serving the NVFP4 file.
    That is the same fault as CORRECTIONS 34 one layer out -- a provenance
    string that names the module default instead of what ran -- except this one
    is visible to every caller of /v1/models, not only to a reader of a JSONL.
    """
    out = _whatif(PROFILE, "-Nvfp4")
    assert re.search(r"--alias\s+\S*NVFP4", out), out
    assert not re.search(r"--alias\s+\S*Q4_K_XL", out), out


def test_the_incumbent_alias_is_unchanged():
    out = _whatif(PROFILE)
    assert re.search(r"--alias\s+Qwen3\.8-27B-Q4_K_XL", out), out


# ------------------------------------- the launcher must not describe the wrong run

def test_the_banner_does_not_announce_the_artifact_it_is_not_serving():
    """Trap 17: the launcher describing what it does not own.

    This project has caught that failure five times by RUNNING a launcher
    rather than reading it, and `-Nvfp4` reintroduced it: the header said
    "Qwen3.8-27B UD-Q4_K_XL across BOTH cards -- 16.69 GiB" over a run that
    loads a 13.84 GiB NVFP4 file, and the rate line quoted the incumbent's
    25.5 / 25.4 / 26.4 four rows above the +63.1 % that replaces it.
    """
    out = _whatif(SERVE, "-Dual", "-Nvfp4")
    assert "UD-Q4_K_XL" not in out, out
    assert "16.69 GiB" not in out, out
    assert "25.5 / 25.4 / 26.4" not in out, out


def test_the_banner_still_describes_the_incumbent_correctly():
    """The branch must not leak the other way."""
    out = _whatif(SERVE, "-Dual")
    assert "UD-Q4_K_XL" in out, out
    assert "NVFP4" not in out, out


# ---------------------------------------------- the deep pair, at the MEASURED ceiling

DEEP = os.path.join(LAUNCHERS, "serve-dual-nvfp4-deep.bat")
DEEP_LAN = os.path.join(LAUNCHERS, "serve-dual-nvfp4-deep-lan.bat")
BOTH_DEEP = [DEEP, DEEP_LAN]


@pytest.mark.parametrize("path", BOTH_DEEP)
def test_the_deep_launcher_exists_at_the_root(path):
    """147,456 is the bench depth. Real work on this machine runs near 250,000
    from the -MaxCtx launchers, and the NVFP4 pair could not go past its
    default, so the fastest configuration was also the shallowest."""
    assert os.path.exists(path), path + " is missing"


@pytest.mark.parametrize("path", BOTH_DEEP)
def test_the_deep_launcher_is_readable_by_cmd(path):
    raw = open(path, "rb").read()
    raw.decode("ascii")
    assert not raw.startswith(b"\xef\xbb\xbf"), "a BOM makes cmd choke"
    assert b"\r\n" in raw


@pytest.mark.parametrize("path", BOTH_DEEP)
def test_the_deep_launcher_asks_for_the_artifact_and_the_depth(path):
    t = read(path)
    assert "-Dual" in t, path
    assert "-Nvfp4" in t, path
    assert "-Deep" in t, path


@pytest.mark.parametrize("path", BOTH_DEEP)
def test_the_deep_launcher_still_refuses_the_budget_question(path):
    """-Deep is a MEASURED constant, not "the deepest that fits". 262,144 does
    not come up at all, so asking the budget is the wrong question here too."""
    assert "-MaxCtx" not in read(path), path


def test_only_the_lan_deep_one_exposes():
    assert "-Lan" not in read(DEEP)
    assert "-Lan" in read(DEEP_LAN)


def test_deep_serves_the_measured_ceiling():
    """200,704, NOT the 229,376 first recorded -- see CORRECTIONS 35.

    229,376 was called the ceiling because it survived a 65,643-token request,
    a QUARTER of its own window. Asked for the arena's standard half-window
    slice it loads with 206 MiB free on the second card and dies on the request
    with `cudaMalloc failed: out of memory`. 206 is below the 336 this project
    has already recorded as dying.
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Deep")
    assert re.search(r"-c\s+200704", out), out
    assert not re.search(r"-c\s+229376", out), out


def test_deep_keeps_everything_else_the_same():
    """Only the window moves. A depth switch that also changed the decoder or
    the n-gram would make the two pairs incomparable."""
    out = _whatif(PROFILE, "-Nvfp4", "-Deep")
    assert "NVFP4-MTP-VERY-LOW.gguf" in out, out
    assert re.search(r"--spec-type\s+draft-mtp,ngram-mod", out), out
    assert re.search(r"--spec-ngram-mod-n-match\s+24", out), out
    assert re.search(r"--alias\s+\S*NVFP4", out), out


def test_deep_without_nvfp4_is_refused():
    """On UD-Q4_K_XL the deep question is a budget one and -MaxCtx answers it.
    229,376 is a ceiling measured on ONE artifact and does not transfer."""
    out = _whatif(PROFILE, "-Deep")
    assert "FATAL" in out, out


def test_deep_and_maxctx_together_are_refused():
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-MaxCtx")
    assert "FATAL" in out, out


def test_the_switch_reaches_the_profile_through_serve():
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Deep")
    assert re.search(r"-Deep\s+True", out), out


def test_the_deep_banner_says_the_headroom_is_the_cost():
    """1,133 and 654 MiB free after a 91,428-token request. This project has
    measured 336 dying and 488 surviving, so the deep rung sits above that line
    but not far above it, and the launcher must not be quiet about that."""
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Deep")
    assert "200,704" in out or "200704" in out, out
    assert "654" in out, out
