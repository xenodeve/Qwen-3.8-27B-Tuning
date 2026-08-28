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
NVFP4 = os.path.join(ROOT, "serve-dual-nvfp4.bat")
NVFP4_LAN = os.path.join(ROOT, "serve-dual-nvfp4-lan.bat")
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
    assert "%~dp0serve.ps1" in read(path)


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
    """229,376 is the measured ceiling; 262,144 does not come up."""
    assert "-MaxCtx" not in read(path), (
        path + " carries -MaxCtx; the NVFP4 ceiling is 229,376")


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
    assert re.search(r"-c\s+229376", out), out


def test_it_refuses_the_other_two_drafters():
    """draft-mtp is already inside this file; the others are different models."""
    for other in ("-Dflash", "-Mtp"):
        out = _whatif(PROFILE, "-Nvfp4", other)
        assert "FATAL" in out, (other, out)


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
