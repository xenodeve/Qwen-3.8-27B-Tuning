r"""`-TheirMirror`: Unsloth's source, our patch, our build.

WHAT IT IS FOR

The developer asked for NVFP4 + DFlash2 + ngram on Unsloth's `0.3.0`. Their
SHIPPED binary cannot do it -- it aborts at `ggml-backend-meta.cpp:543` the
moment DFlash2 is asked for under `-sm tensor` (issue #52, `5f87e12`). So their
source was copied out of `%USERPROFILE%\.unsloth`, the mirror patch applied, and
the tree built here: `C:\AI\llama.cpp-unsloth-mirror`.

That makes FOUR binaries on this machine, and the point of the switch is that
choosing between them is never implicit:

    llama.cpp-blackwell         10499   unpatched   the served default
    llama.cpp-mirror            10499   PATCHED     -Dflash
    ~/.unsloth/.../bin          10679   unpatched   -TheirBuild
    llama.cpp-unsloth-mirror    10679   PATCHED     -TheirMirror   <- this

WHY IT IS A SWITCH OF ITS OWN AND NOT `-TheirBuild`

`-TheirBuild` runs the binary Unsloth SHIPPED. Pointing it at a tree we patched
and compiled ourselves would make one flag mean two different artifacts, and
every row and log that named it would be ambiguous forever. `-TheirBuild` is
also refused with `-Dflash` for a reason that has not changed: their shipped
binary really cannot do it.

THE HAZARDS

1. TWO SWITCHES CHOOSING THE EXE. `-Dflash` already sets `$Exe`, and so do
   `-TheirBuild` and this. Any pair of them is a launch nobody can attribute,
   so the combinations are refused rather than ordered.

2. THE CPU-RUN FAULT. A llama-server that cannot find `cudart64_13.dll` reports
   no CUDA devices and serves happily from the CPU. CUDA 13 keeps it in
   `%CUDA_PATH%\bin\x64`, not `bin`. The three runtime DLLs were copied beside
   this binary so it is self-contained like our mirror, and the profile still
   refuses if it cannot see them -- a believable slow number from the wrong
   hardware is this repository's north-star failure.

3. THE BUILD NUMBER LIES, and it will be read. The banner says
   `0.3.0-dev (build 215, commit 9f55aee)`. `0.3.0-dev` and the "Compiled by
   the Unsloth team" line are THEIRS; `build 215` and the commit are OUR
   repository's git, counted by their build system because the copied tree has
   no `.git`. A log from this binary must never be read as 10499's.

WHAT IS NOT ESTABLISHED, and the launcher says so in its own text: **it has
never been verified to load DFlash2 under `-sm tensor`.** The build finished
while the cards were busy. The failure mode if it cannot is a loud abort, not a
wrong number, which is why shipping the icon ahead of the probe is acceptable
here -- but it is stated, not assumed.
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
BATS = ["serve-dual-nvfp4-dflash-theirmirror.bat",
        "serve-dual-nvfp4-dflash-theirmirror-lan.bat"]
THEIR_MIRROR = r"C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe"


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


# ------------------------------------------------------------- the binary

def test_it_runs_the_patched_unsloth_tree():
    line, _ = preview("-Nvfp4", "-Vision", "-Dflash", "-TheirMirror")
    assert line.split()[0].strip('"').lower() == THEIR_MIRROR.lower(), line


def test_without_it_dflash_still_runs_our_mirror():
    line, _ = preview("-Nvfp4", "-Vision", "-Dflash")
    assert "llama.cpp-mirror" in line.split()[0], line
    assert "unsloth-mirror" not in line.split()[0], line


def test_the_binary_is_self_contained():
    """It must not depend on a PATH the launcher happens to set. Three CUDA
    runtime DLLs sit beside it, as they do beside our mirror."""
    d = os.path.dirname(THEIR_MIRROR)
    if not os.path.isdir(d):
        pytest.skip("the patched Unsloth tree is not built on this machine")
    for dll in ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"):
        assert os.path.isfile(os.path.join(d, dll)), dll


def test_the_profile_refuses_if_the_runtime_is_not_there():
    """The failure is not a crash. It is a server that starts, answers, and is
    slow -- a believable number from the wrong hardware."""
    src = open(PROFILE, encoding="utf-8", errors="replace").read()
    i = src.index("$Exe = $THEIR_MIRROR_EXE")
    region = src[i:i + 2500]
    assert "cudart" in region, "nothing checks the CUDA runtime is reachable"
    assert "FATAL" in region, "a missing runtime must refuse, not warn"


# ------------------------------------- hazard 1: one switch chooses the exe

def test_it_needs_dflash():
    """Its whole reason to exist is the one thing their shipped binary cannot
    do. Without -Dflash it would be a second way to say -TheirBuild."""
    out = _whatif(PROFILE, "-Nvfp4", "-Vision", "-TheirMirror")
    assert "FATAL" in out, out


def test_it_is_refused_with_theirbuild():
    out = _whatif(PROFILE, "-Nvfp4", "-Vision", "-Dflash", "-TheirMirror",
                  "-TheirBuild")
    assert "FATAL" in out, out


def test_theirbuild_is_still_refused_with_dflash():
    """Unchanged, and for the unchanged reason: their SHIPPED binary aborts."""
    out = _whatif(SERVE, "-Dual", "-Dflash", "-TheirBuild")
    assert "FATAL" in out, out


# ------------------------------------------- it serves the measured settings

def test_it_keeps_everything_the_nvfp4_dflash_icon_serves():
    """The two icons must differ in the BINARY and in nothing else, or a delta
    between them has two causes.

    `-ts` is normalised out. It is computed from free VRAM at launch, so two
    previews taken seconds apart legitimately disagree -- comparing it would
    test the desktop, not the switch. What matters is that it is PRESENT in
    both, which the assertion below still requires.
    """
    a, _ = preview("-Nvfp4", "-Vision", "-Dflash")
    b, _ = preview("-Nvfp4", "-Vision", "-Dflash", "-TheirMirror")

    def rest(line):
        tail = line.split(" ", 1)[1]
        assert re.search(r"-ts -?\d+,-?\d+", tail), tail
        return re.sub(r"-ts -?\d+,-?\d+", "-ts <computed>", tail)

    assert rest(a) == rest(b), (
        "the two icons differ in more than the binary, so a delta between them "
        "would have two causes", a, b)


# --------------------------------------- hazard 3: the banner must be explained

def test_the_preview_warns_that_the_build_number_is_ours():
    _, out = preview("-Nvfp4", "-Vision", "-Dflash", "-TheirMirror")
    low = out.lower()
    assert "0.3.0" in out or "10679" in out, out
    assert "build number" in low or "not 10499" in low or "our" in low, out


def test_the_preview_says_it_is_unverified():
    _, out = preview("-Nvfp4", "-Vision", "-Dflash", "-TheirMirror")
    low = out.lower()
    assert "unverified" in low or "never been" in low or "not been" in low, out


# ----------------------------------------------------------------- plumbing

def test_it_reaches_the_profile_through_serve():
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Vision", "-Dflash", "-TheirMirror")
    assert re.search(r"TheirMirror\s+True", out), out


# ---------------------------------------------------------------- launchers

@pytest.mark.parametrize("name", BATS)
def test_the_launcher_exists(name):
    assert os.path.exists(os.path.join(LAUNCHERS, name))


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_asks_for_the_switches(name):
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read()
    live = "\n".join(l for l in body.splitlines()
                     if not l.strip().upper().startswith("REM"))
    for flag in ("-Nvfp4", "-Dflash", "-TheirMirror"):
        assert flag in live, (name, flag, live)


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_states_that_it_is_unverified(name):
    """It was built while the cards were busy and has never loaded DFlash2
    under -sm tensor. The person clicking it is told."""
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read().upper()
    assert "UNVERIFIED" in body or "NEVER BEEN" in body, name


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_explains_the_build_number(name):
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read()
    assert "10499" in body and "0.3.0" in body, name


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
