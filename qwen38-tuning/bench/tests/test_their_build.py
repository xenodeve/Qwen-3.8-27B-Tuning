r"""`-TheirBuild`: run on Unsloth Studio's llama-server, not ours.

THE CONFOUND THIS EXISTS TO REMOVE

Every comparison in this project against Studio assumed one binary. It is two:

    ours     version 0.1.2-dev   build 10499   commit 1deefcca3
    Studio   version 0.3.0-dev   build 10679   commit b84725557

The build NUMBERS differ by 180. **That is not a count of commits between
them**, and saying so would be an inference dressed as a count: our own HEAD is
`1deefcca3`, a local commit *"Add p_min in DFlash2"*, and neither `b84725557`
nor the `50f068f` reported upstream for build 10679 is in our clone. **Both
sides are forks.** What separates them is a source/build-lineage delta of
unknown length.

So `-Clone` -- their whole command line on our binary -- cannot separate "their
flags are better" from "their build is different". This switch supplies the
other cell:

                        our flags        their flags
    our build           icons 1/2/7/8    icon 9   (-Clone)
    their build         -TheirBuild      icon 10  (-Clone -TheirBuild)

WHAT IS AND IS NOT DIFFERENT ABOUT THE TWO BUILDS, checked rather than assumed:

  flags       their `--help` is a strict SUPERSET: ten they have and we lack
              (--kv-unified-per-slot, --spec-synth-len, --spec-synth-rates,
              --tensor-read-lazy, -ncffn, -mmdev, --rpc, three --video-*), and
              NONE we have that they lack. Every flag on their command line
              exists in our build.
  defaults    identical on every flag either side sets.
  SASS        theirs sm_86/89/90/100/120a, ours sm_89/120a. BOTH carry native
              code for BOTH cards here, so neither is JIT-ing from PTX -- the
              2.6x size difference is other people's GPUs, not ours.
  features    `ARCHS = 860,890,900,1000,1200 | USE_GRAPHS = 1 |
              BLACKWELL_NATIVE_FP4 = 1` against our
              `ARCHS = 890,1200 | USE_GRAPHS = 1 | BLACKWELL_NATIVE_FP4 = 1`.
              The compile-time feature set that matters is the same.

So what is left is a SOURCE/BUILD-LINEAGE DELTA, which no amount of
reading `--help` will settle.

THE FAULT THIS SWITCH MUST NOT HAVE

Launched with a bare PATH, Studio's binary prints

    device_info:
      - CPU     : 13th Gen Intel(R) Core(TM) i5-13500

and **no CUDA devices at all**, then serves happily from the CPU. It is
Studio that prepends the loader path; CUDA 13 keeps `cudart64_13.dll` and
`cublas64_13.dll` in `%CUDA_PATH%\bin\x64`, not `bin`, and nothing beside their
binary supplies them.

A profile that shipped this without the path would produce a believable slow
number from a CPU run and call it a build comparison -- which is the exact
failure this repository's north star is about. The switch therefore refuses to
launch unless it can SEE the runtime it is about to need.
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
BATS = ["serve-dual-nvfp4-clone-theirbuild.bat",
        "serve-dual-nvfp4-clone-theirbuild-lan.bat"]
STUDIO_EXE = os.path.join(os.path.expanduser("~"), ".unsloth", "llama.cpp",
                          "build", "bin", "Release", "llama-server.exe")


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

def test_it_runs_studios_binary():
    line, _ = preview("-Nvfp4", "-Vision", "-Clone", "-TheirBuild")
    assert line.split()[0].strip('"').lower() == STUDIO_EXE.lower(), line


def test_the_default_is_still_ours():
    line, _ = preview("-Nvfp4", "-Vision", "-Clone")
    assert "llama.cpp-blackwell" in line, line
    assert ".unsloth" not in line.split()[0], line


def test_the_swap_happens_before_the_build_guard():
    """`-Dflash` carries the comment for this: the guard reads ggml-cuda.dll
    BESIDE $Exe, so swapping the binary afterwards checks one file and runs
    another. That fault put fifteen rows on a build with no Blackwell
    kernels."""
    src = open(PROFILE, encoding="utf-8", errors="replace").read()
    swap = src.index("$Exe = $STUDIO_EXE")
    guard = src.index("$dll = Join-Path (Split-Path $Exe -Parent)")
    assert swap < guard, "the binary is swapped after the architecture guard"


def test_their_build_passes_our_architecture_guard():
    """Not an opinion -- their ggml-cuda.dll really does carry sm_89 and
    sm_120a, so the guard should let it through rather than needing
    -IKnowTheBuildIsWrong."""
    _, out = preview("-Nvfp4", "-Vision", "-Clone", "-TheirBuild")
    assert "has no sm_" not in out, out


# ------------------------------------------- the CPU-run fault, refused loudly

def test_it_prepends_the_cuda_runtime_path():
    src = open(PROFILE, encoding="utf-8", errors="replace").read()
    assert "bin\\x64" in src, (
        "CUDA 13 keeps cudart64_13.dll in %CUDA_PATH%\\bin\\x64, and their "
        "binary finds no GPU without it")
    assert "CUDA_PATH" in src


def test_it_refuses_rather_than_running_on_the_cpu():
    """The failure mode is not a crash. It is a server that starts, answers,
    and is slow -- a believable number from the wrong hardware."""
    src = open(PROFILE, encoding="utf-8", errors="replace").read()
    i = src.index("$Exe = $STUDIO_EXE")
    region = src[i:i + 3000]
    assert "cudart" in region, "nothing checks that the CUDA runtime is reachable"
    assert "FATAL" in region, "a missing runtime must refuse, not warn"


def test_the_preview_names_the_build_it_would_run():
    """A dry run that did not say which binary it had chosen would be the
    quietest possible version of this bug."""
    _, out = preview("-Nvfp4", "-Vision", "-Clone", "-TheirBuild")
    assert "10679" in out or "unsloth" in out.lower(), out


# ----------------------------------------------------------------- plumbing

def test_it_reaches_the_profile_through_serve():
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Vision", "-Clone", "-TheirBuild")
    assert re.search(r"TheirBuild\s+True", out), out


def test_it_is_refused_with_dflash():
    """-Dflash chooses a third binary. Two switches choosing $Exe is a launch
    nobody can attribute."""
    out = _whatif(SERVE, "-Dual", "-Dflash", "-TheirBuild")
    assert "FATAL" in out, out


def test_it_works_without_clone_too():
    """OUR flags on THEIR build is the fourth cell of the 2x2 and the one that
    isolates the build by itself."""
    line, _ = preview("-Nvfp4", "-Deep", "-Vision", "-Beta", "-TheirBuild")
    assert ".unsloth" in line.lower(), line
    assert "-c 200704" in line, line


# ---------------------------------------------------------------- launchers

@pytest.mark.parametrize("name", BATS)
def test_the_launcher_exists(name):
    assert os.path.exists(os.path.join(LAUNCHERS, name))


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_asks_for_both_switches(name):
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read()
    live = "\n".join(l for l in body.splitlines()
                     if not l.strip().upper().startswith("REM"))
    assert "-Clone" in live and "-TheirBuild" in live, live


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_names_both_builds(name):
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read()
    assert "10679" in body and "10499" in body, \
        "a build comparison must name both builds where somebody will read it"


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
