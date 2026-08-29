r"""A profile that adopts what Unsloth Studio does, as ONE testable bundle.

Called `-Beta` from 2026-08-29 -- it was `-Lean` while the bundle was only
about memory, and the name stopped fitting once it also carried decoder
settings the developer wanted tried.

Studio runs the same model file on the same two cards and differs from us on
eleven flags (`docs/researchs/unsloth-studio-config-2026-08-29.md`). Adopting
them one at a time is eleven sweeps; adopting them all silently is a profile
nobody can reason about. `-Beta` is the middle: **one switch, one bundle, one
paired measurement**, and if the bundle wins it gets bisected.

WHAT IS IN THE BUNDLE, AND WHY EACH ONE

  --cache-ram 0        MEASURED HERE: a real session held 20.4 GB working set
  --ctx-checkpoints 0  and 34.4 GB private, with 32 checkpoints reaching 350 MiB
                       each. Studio disables both. NOT FREE -- our log shows
                       checkpoints RESTORED at 47,940-50,091, so this trades
                       host RAM for re-prefill.
  --load-mode none     VENDOR: Studio's auto "picks None when it can prove the
                       model fits without paging, since a mapped read is slower".
  --kv-unified         Studio sets it; may be inert at -np 1.
  --threads 2          Studio uses 2 against our 18. Everything is GPU-resident,
                       and the draft sampler falls back to the CPU under this
                       split, so the right number is not obvious either way.
  --metrics            free Prometheus endpoint. No throughput claim.

WHAT IS DELIBERATELY *NOT* IN IT

  the n-gram parameters and the --spec-type order -- they have their own arm
  sets and belong in a sweep, not in a profile;
  --parallel 4 -- Studio shares one window across four slots; we serve one
  conversation and want the whole window for it;
  -c ~41,000 -- their `auto` picks shallow "while prioritizing GPU speed"
  (VENDOR). Depth is the point of this machine;
  -ub 512 -- their default. Ours is 1024 and MEASURED at +10.1 % prefill;
  --spec-draft-n-max 2 -- their documented default. Ours is 3 and our own
  acceptance per position (0.690, 0.448, 0.284) supports it;
  the sampler -- a QUALITY lever, and quality is unmeasured on every artifact
  here.

SEPARATE FROM THE BUNDLE, AND NOT OPTIONAL: `--fit on --fit-target 768` is
MEASURED INERT under `-sm tensor` -- `llama_params_fit is not implemented for
SPLIT_MODE_TENSOR` on every boot. It is removed from the profile outright, for
every arm, because a flag that does nothing is a flag a reader believes did
something. Studio passes `--fit off` for the same reason.
"""
import os
import re
import subprocess

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q4-dual.ps1")
SERVE = os.path.join(ROOT, "serve.ps1")


def _whatif(script, *args):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", script, "-WhatIf"] + list(args),
        capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr


# ---------------------------------------------- the inert flag, removed for good

@pytest.mark.parametrize("args", [(), ("-Nvfp4",), ("-Nvfp4", "-Beta")])
def test_fit_is_turned_OFF_not_merely_unstated(args):
    """`--fit` DEFAULTS TO ON, so removing the flag changes nothing.

    This started as "delete the inert flag", and booting it showed the fitting
    warning still there -- `--fit [on|off] ... default: 'on'`. Silence needs
    `--fit off`, which is what Unsloth Studio passes. Deleting the flag was a
    no-op dressed as a cleanup, and only running it said so.

    `--fit-target` goes with it: it is a margin for a fitting step that never
    runs under this split.
    """
    out = _whatif(PROFILE, *args)
    assert re.search(r"--fit\s+off", out), out
    assert "--fit-target" not in out, out


@pytest.mark.parametrize("args", [(), ("-Nvfp4",), ("-Nvfp4", "-Beta")])
def test_the_split_is_still_tensor(args):
    """Removing --fit must not have touched what makes it inert."""
    out = _whatif(PROFILE, *args)
    assert re.search(r"-sm\s+tensor", out), out


# ------------------------------------------------------------------ the bundle

BUNDLE = ["--cache-ram", "0", "--ctx-checkpoints", "0",
          "--load-mode", "none", "--kv-unified", "--metrics",
          "-t", "2"]


def test_beta_applies_the_whole_bundle():
    out = _whatif(PROFILE, "-Nvfp4", "-Beta")
    for flag, value in (("--cache-ram", "0"), ("--ctx-checkpoints", "0"),
                        ("--load-mode", "none")):
        assert re.search(re.escape(flag) + r"\s+" + value, out), (flag, out)
    assert "--kv-unified" in out, out
    assert "--metrics" in out, out
    assert re.search(r"-t\s+2\b", out), out


def test_without_beta_none_of_it_appears():
    """Opt-in. The bundle is unmeasured and must not leak into the default."""
    out = _whatif(PROFILE, "-Nvfp4")
    for flag in ("--cache-ram", "--ctx-checkpoints", "--load-mode",
                 "--kv-unified", "--metrics"):
        assert flag not in out, (flag, out)
    assert re.search(r"-t\s+18\b", out), out


def test_beta_keeps_everything_we_measured():
    """The bundle borrows what Studio does DIFFERENTLY. It must not quietly
    revert a value this project measured and won on."""
    out = _whatif(PROFILE, "-Nvfp4", "-Beta")
    assert re.search(r"-ub\s+1024", out), "-ub 1024 is +10.1 % prefill, MEASURED"
    assert re.search(r"--spec-type\s+draft-mtp,ngram-mod", out), out
    assert re.search(r"-np\s+1", out), "one conversation gets the whole window"


def test_beta_reaches_the_profile_through_serve():
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Beta")
    assert re.search(r"-Beta\s+True", out), out


def test_the_banner_says_the_bundle_is_unmeasured():
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Beta")
    assert "UNMEASURED" in out.upper(), out


# ---------------------------------------------------- the launcher, at the developer's request

BETA = os.path.join(ROOT, "launchers", "serve-dual-nvfp4-beta.bat")
BETA_LAN = os.path.join(ROOT, "launchers", "serve-dual-nvfp4-beta-lan.bat")
BOTH_BETA = [BETA, BETA_LAN]


def read_bat(path):
    with open(path, "rb") as fh:
        return fh.read().decode("ascii")


@pytest.mark.parametrize("path", BOTH_BETA)
def test_the_beta_launcher_exists(path):
    """Shipped for an UNMEASURED bundle, which is normally forbidden here --
    the exception is that the developer asked for it in order to be the
    measurement. The file must say that, which the next test checks."""
    assert os.path.exists(path), path


@pytest.mark.parametrize("path", BOTH_BETA)
def test_it_admits_the_bundle_is_unmeasured(path):
    """A launcher that presents a hypothesis as a result is trap 17 again."""
    assert "UNMEASURED" in read_bat(path).upper(), path


@pytest.mark.parametrize("path", BOTH_BETA)
def test_it_asks_for_the_bundle_and_the_same_everything_else(path):
    """AT 200,704, because that is the depth actually served.

    This shipped at 147,456 first, on the reasoning that it made a clean A/B
    against the shallow pair. The developer runs 200k. An A/B against a
    configuration nobody uses answers a question nobody asked, so the pair moved
    and its partner is now serve-dual-nvfp4-deep.bat.
    """
    t = read_bat(path)
    for flag in ("-Dual", "-Nvfp4", "-Vision", "-Beta", "-Deep"):
        assert flag in t, (path, flag)


@pytest.mark.parametrize("path", BOTH_BETA)
def test_it_is_readable_by_cmd(path):
    raw = open(path, "rb").read()
    raw.decode("ascii")
    assert not raw.startswith(b"\xef\xbb\xbf"), "a BOM makes cmd choke"
    assert b"\r\n" in raw


def test_only_the_lan_beta_one_exposes():
    assert "-Lan" not in read_bat(BETA)
    assert "-Lan" in read_bat(BETA_LAN)


def test_it_names_the_number_to_compare_against():
    """The point of the icon is an A/B the developer runs. A launcher that does
    not say what to compare it with leaves them to remember.

    Asserted as "two working-set figures at this depth", not as two literals:
    the first version hardcoded 15.28 and 2.03, which were measured at ctx
    65,536, and moving the pair to 200,704 made them the wrong numbers for the
    file they were in.
    """
    t = read_bat(BETA)
    assert t.count("GB working set") >= 2, (
        "the .bat must state BOTH sides of the memory pair it claims")
    assert "200,704" in t, "it must say which depth those figures came from"


# ------------------------------------------------- thinking, the Unsloth mechanism

def test_beta_uses_the_models_own_template_with_kwargs():
    """Studio does not pass a template file at all.

    It uses the one inside the GGUF and steers it with
    `--chat-template-kwargs {"enable_thinking": true, "preserve_thinking": true}`.
    We pass `--chat-template-file qwen38-late-system.jinja` plus
    `--reasoning-effort medium`, and NEITHER the template's reason for existing
    nor the choice of `medium` is written down anywhere in this repository.

    `-Beta` borrows their mechanism whole, which is the only way to find out
    whether ours is doing anything. Note what `preserve_thinking` maps to on our
    side: `--reasoning-preserve`, a flag we do not set and which our own boot log
    suggests -- "chat template supports preserving reasoning, consider enabling
    it via --reasoning-preserve".

    THEIR MECHANISM, ADAPTED TO OUR BINARY, BECAUSE BOOTING IT SAID SO. Copying
    `--chat-template-kwargs {"enable_thinking": true, "preserve_thinking": true}`
    verbatim starts and thinks, but the log answers back twice:

        W Setting 'enable_thinking' via --chat-template-kwargs is deprecated.
          Use --reasoning on / --reasoning off instead.
        I chat template supports preserving reasoning, consider enabling it via
          --reasoning-preserve

    So one kwarg is deprecated and the other DOES NOTHING -- the server still
    asks for `--reasoning-preserve` after being handed `preserve_thinking`.
    Copying a command line from a different build is copying its bugs.
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert "--chat-template-file" not in out, out
    assert "--reasoning-effort" not in out, out
    assert "--chat-template-kwargs" not in out, "deprecated on this build"
    assert re.search(r"--reasoning\s+on", out), out
    assert "--reasoning-preserve" in out, out


def test_without_beta_the_template_file_is_still_used():
    """The default keeps what it has always had. This is an experiment, not a
    migration."""
    out = _whatif(PROFILE, "-Nvfp4", "-Deep")
    assert "qwen38-late-system.jinja" in out, out
    assert re.search(r"--reasoning-effort\s+medium", out), out
    assert "--chat-template-kwargs" not in out, out


def test_no_json_blob_has_to_cross_two_shells():
    """The first version passed a JSON object as one argv entry, which has to
    survive PowerShell and then cmd intact. Using the flags llama.cpp actually
    wants removes the problem rather than solving it."""
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert "{" not in out.split("llama-server.exe")[-1], out


# ------------------------------------- the two decoder values added on request

def test_beta_takes_unsloths_draft_depth():
    """2, not our 3.

    3 is llama.cpp's own default (`--spec-draft-n-max N (default: 3)`) and we
    were getting it by not setting anything. Studio sets 2 deliberately -- its
    UI documents 2 for MTP on GPU and 3 for CPU/Mac, so 2 is THEIR choice for a
    GPU run, not a standard.

    Our real-use counters argue for 3: acceptance per position
    (0.690, 0.448, 0.284), so position three still lands 28 % of the time.
    Putting 2 in the bundle is how that argument gets tested rather than
    repeated.
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert re.search(r"--spec-draft-n-max\s+2\b", out), out


def test_beta_takes_unsloths_ngram_bounds():
    """48 and 64 -- which are llama.cpp's DEFAULTS, not Studio's tuning.

    `--spec-ngram-mod-n-min` defaults to 48 and `n-max` to 64. Studio simply
    does not set them. WE are the ones deviating, to 16 and 32, and this
    project's own register shows those two carried through from an older sweep
    where they were "held constant" rather than chosen.

    `n-match` stays 24 on both sides -- also the default, and separately
    measured here at +27.1 % over 12.
    """
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert re.search(r"--spec-ngram-mod-n-min\s+48\b", out), out
    assert re.search(r"--spec-ngram-mod-n-max\s+64\b", out), out
    assert re.search(r"--spec-ngram-mod-n-match\s+24\b", out), out


def test_the_default_keeps_our_values():
    out = _whatif(PROFILE, "-Nvfp4", "-Deep")
    assert re.search(r"--spec-draft-n-max\s+3\b", out), out
    assert re.search(r"--spec-ngram-mod-n-min\s+16\b", out), out
    assert re.search(r"--spec-ngram-mod-n-max\s+32\b", out), out
