"""A result row must say which binary produced it, and `--version` cannot.

THE INCIDENT THIS GUARDS, and it happened on 2026-08-24 in this repo.

`C:\\AI\\llama.cpp-dflash2` and `C:\\AI\\llama.cpp-blackwell` are built from the
same tree at the same commit. Both report, byte for byte:

    version: 0.1.2-dev (build 10499, commit 1deefcca3)
    built with MSVC 19.44.35228.0 for Windows AMD64

They differ in one thing: `CMAKE_CUDA_ARCHITECTURES`. The first has SASS for
`sm_89` only, so on this `sm_120` card the driver JIT-compiles Ada PTX and the
same corpus at the same ctx measured **22.67 tok/s against 96.92** -- with
byte-identical buffer sizes, `65+0`, no OOM, and nothing in the log saying the
kernels were JIT'd.

So the two failure modes this file exists for:

  1. `dflash2_arena.EXE` is a module constant. Re-pointing it by editing the
     file makes the Ada figure unreproducible; leaving it means the new build
     can never be measured. It needs an override that is *recorded*, not one
     that is silent.
  2. A row measured on either binary lands in the same JSONL looking identical.
     A later reader comparing 22.67 against 96.92 would reach for the boot-VRAM
     story, the corpus, the arm -- every explanation except the only true one,
     because the file does not carry the distinguishing fact.

The fix is that provenance comes from the CODE OBJECTS, not from the version
string. `cuda_archs()` reads them out of `ggml-cuda.dll` next to the exe. That
is the only thing on this machine that tells the two builds apart.

WHAT THIS FILE CANNOT DO is prove the kernels that ran came from the SASS rather
than from JIT'd PTX. Nothing available here can: a matching arch in the binary
means JIT was not *required*, which is as close as this instrument gets.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena
import provenance

ADA = r"C:\AI\llama.cpp-dflash2\llama-server.exe"
BLACKWELL = r"C:\AI\llama.cpp-blackwell\llama-server.exe"


# --------------------------------------------------------------- the override

def test_resolve_exe_falls_through_to_the_default(monkeypatch):
    monkeypatch.delenv(provenance.ENV_VAR, raising=False)
    assert provenance.resolve_exe(ADA) == ADA


def test_resolve_exe_honours_the_environment(monkeypatch):
    monkeypatch.setenv(provenance.ENV_VAR, BLACKWELL)
    assert provenance.resolve_exe(ADA) == BLACKWELL


def test_an_empty_variable_does_not_produce_an_empty_argv0(monkeypatch):
    """Exporting the variable blank is a plausible operator slip. It must read
    as "not set" rather than as "launch the empty string", which would fail
    somewhere far from the cause."""
    monkeypatch.setenv(provenance.ENV_VAR, "")
    assert provenance.resolve_exe(ADA) == ADA


def test_server_argv_launches_the_module_exe():
    """EXE is resolved once at import, which is when the environment is read.
    What this pins is that `server_argv` carries THAT value rather than a second
    literal written out again inside the function -- the two drifting apart is
    how an override gets honoured in the log banner and ignored in the launch."""
    assert arena.server_argv(16384, [])[0] == arena.EXE


# ------------------------------------------------------------- the provenance

@pytest.mark.skipif(not os.path.isfile(ADA), reason="Ada build not installed")
def test_ada_build_reports_only_sm_89():
    assert arena.cuda_archs(ADA) == ("sm_89",)


@pytest.mark.skipif(not os.path.isfile(BLACKWELL), reason="Blackwell build not installed")
def test_blackwell_build_reports_both_architectures():
    """120 is written 120a by llama.cpp's own cmake, and the arch-specific
    variant is what the BLACKWELL_MMA_AVAILABLE paths are compiled against.
    Asserting on `sm_120` alone would pass on a binary that has neither."""
    assert arena.cuda_archs(BLACKWELL) == ("sm_120a", "sm_89")


@pytest.mark.skipif(not (os.path.isfile(ADA) and os.path.isfile(BLACKWELL)),
                    reason="both builds needed to prove they are distinguishable")
def test_the_two_builds_are_distinguishable_though_their_versions_are_not():
    """The whole reason this function exists. If these ever compare equal, the
    JSONL cannot separate a JIT figure from a native one and every rate in it
    becomes unattributable."""
    assert arena.cuda_archs(ADA) != arena.cuda_archs(BLACKWELL)


def test_cuda_archs_reports_absence_rather_than_guessing():
    """A missing dll or missing cuobjdump must not look like 'no Blackwell
    support'. The caller has to be able to tell 'checked, none there' from
    'could not check' -- conflating them is how an unverified run gets filed as
    a verified one."""
    assert arena.cuda_archs(r"C:\AI\nonexistent-build\llama-server.exe") is None
