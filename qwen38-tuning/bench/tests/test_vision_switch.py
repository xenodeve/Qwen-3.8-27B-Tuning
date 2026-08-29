r"""Images: the model advertises them and this profile turned them off.

WHAT THE DEVELOPER HIT, 2026-08-29. Claude Code sent five images through
`serve-dual-nvfp4-deep-lan.bat` and got five HTTP 500s:

    image input is not supported - hint: if this is unexpected, you may need
    to provide the mmproj

The model is not the problem. Its own chat template, read out of the GGUF at
load, begins `{%- set image_count = namespace(value...` -- it handles images.
The vision tower is a SEPARATE FILE, and this project switched it off on
purpose: `docs/reports/16-OPTIMIZATION-SURFACE.md` records the whole
`--mmproj*` family as *"Not applicable -- text only"*. That was true of a
benchmark harness and is not true of a coding agent that pastes screenshots.

WHAT MAKES IT MORE THAN ADDING A FLAG. The tower is a SECOND MODEL, and on this
machine `-sm tensor` has never loaded one: `draft-dflash` aborts in
`ggml-backend-meta.cpp` for exactly that reason, which is why DFlash2 needs a
patched binary. Whether the projector hits the same wall is a question about
this split, not about llama.cpp, and only running it answers it.

It also costs 888 MiB with GPU offload on, which is the default -- against the
692 MiB the deep rung finishes a large request with. The budget check has to see
it, or the profile will start something that cannot serve.
"""
import os
import re
import subprocess

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
LAUNCHERS = os.path.join(ROOT, "launchers")
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q4-dual.ps1")
SERVE = os.path.join(ROOT, "serve.ps1")


def _whatif(script, *args):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", script, "-WhatIf"] + list(args),
        capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr


def test_the_default_is_still_text_only():
    """Vision is opt-in. It costs 888 MiB and nothing measured needs it."""
    out = _whatif(PROFILE)
    assert "--no-mmproj-auto" in out, out
    assert "mmproj-BF16.gguf" not in out, out


def test_vision_passes_the_projector():
    out = _whatif(PROFILE, "-Nvfp4", "-Vision")
    assert re.search(r"-mm\s+\S*mmproj-BF16\.gguf", out), out


def test_vision_drops_the_flag_that_suppresses_it():
    """--no-mmproj-auto and -mm together is a contradiction to read later."""
    out = _whatif(PROFILE, "-Nvfp4", "-Vision")
    assert "--no-mmproj-auto" not in out, out


def test_vision_changes_nothing_else():
    out = _whatif(PROFILE, "-Nvfp4", "-Vision")
    assert "NVFP4-MTP-VERY-LOW.gguf" in out, out
    assert re.search(r"--spec-type\s+draft-mtp,ngram-mod", out), out
    assert re.search(r"--spec-ngram-mod-n-match\s+24", out), out


def _weights_term(text):
    """The weights figure from the profile's demand line, which it prints on
    EVERY launch and not only when it refuses.

    That was not true until 2026-08-29. This test passed while the developer's
    server held both cards -- the profile refused, and the refusal carried the
    arithmetic -- and went red the moment the cards were free and it started
    normally. A test whose subject is only visible on the failure path is a test
    of the machine's mood. The profile now prints the demand either way, which
    is the better behaviour anyway: the run that clears by 200 MiB and the one
    that clears by 6,000 used to look identical.
    """
    m = re.search(r"([\d,]+) weights", text)
    return int(m.group(1).replace(",", "")) if m else None


def test_the_projector_is_in_the_budget():
    """888 MiB the guard cannot see is 888 MiB it will hand to a spill."""
    plain = _weights_term(_whatif(PROFILE, "-Nvfp4"))
    vision = _weights_term(_whatif(PROFILE, "-Nvfp4", "-Vision"))
    assert plain is not None and vision is not None, (plain, vision)
    assert vision - plain == 888, (plain, vision)


def test_the_switch_reaches_the_profile_through_serve():
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Vision")
    assert re.search(r"-Vision\s+True", out), out


def test_the_banner_says_images_are_on_and_what_they_cost():
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Vision")
    assert "888" in out, out


def test_the_banner_is_silent_about_vision_when_it_is_off():
    out = _whatif(SERVE, "-Dual", "-Nvfp4")
    assert "mmproj" not in out.lower(), out


# ------------------------------------------------ the launchers, once it was proven

NVFP4 = os.path.join(LAUNCHERS, "serve-dual-nvfp4.bat")
NVFP4_LAN = os.path.join(LAUNCHERS, "serve-dual-nvfp4-lan.bat")
DEEP = os.path.join(LAUNCHERS, "serve-dual-nvfp4-deep.bat")
DEEP_LAN = os.path.join(LAUNCHERS, "serve-dual-nvfp4-deep-lan.bat")
SERVED_PAIR = [NVFP4, NVFP4_LAN]
DEEP_PAIR = [DEEP, DEEP_LAN]


def read(path):
    with open(path, "rb") as fh:
        return fh.read().decode("ascii")


@pytest.mark.parametrize("path", SERVED_PAIR)
def test_the_served_pair_carries_vision(path):
    """Folded into the launchers the developer actually uses, 2026-08-29,
    rather than kept as a separate icon.

    It costs headroom and NOT window: both carry 147,456 either way, and a large
    request finishes with 1,205 / 2,450 MiB free against about 2,395 without the
    tower. A separate icon for a capability that costs no context is two icons
    where one will do, and the one without it would be the one people picked by
    accident and then wondered why images 500.
    """
    assert "-Vision" in read(path), path


@pytest.mark.parametrize("path", DEEP_PAIR)
def test_the_deep_pair_stays_text_only(path):
    """200,704 loaded and answered a SMALL picture with 614 MiB free -- between
    the 488 this project measured surviving and the 336 dying -- and vision
    beside a large text prompt is untested at any depth. Turning it on here
    would ship an untested combination as a default."""
    assert "-Vision" not in read(path), path


def test_the_separate_vision_icons_are_gone():
    """Folding it in makes them duplicates. Two launchers differing only in a
    flag one of them always sets is a fork waiting to drift."""
    for name in ("serve-dual-nvfp4-vision.bat",
                 "serve-dual-nvfp4-vision-lan.bat"):
        assert not os.path.exists(os.path.join(ROOT, name)), name


@pytest.mark.parametrize("path", SERVED_PAIR + DEEP_PAIR)
def test_none_of_them_hold_a_projector_path(path):
    """The launcher holds no configuration: the mmproj path lives in the
    profile and only there."""
    assert "mmproj" not in read(path).lower(), path
