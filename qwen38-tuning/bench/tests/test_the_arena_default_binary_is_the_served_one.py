r"""The arena's DEFAULT binary must be the one the serving profiles launch.

THE OTHER HALF OF test_the_binary_covers_every_visible_gpu.py. That file guards
the OBSERVATION -- it reads llama.cpp's own `CUDA : ARCHS =` line at boot and
raises when a visible card is missing from it. It works: it has stopped four
launches since 2026-08-27, including one on 2026-08-29.

Its own docstring names the cause and does not fix it:

    "The default was never updated after the second card arrived."

`dflash2_arena.EXE` defaulted to `C:\AI\llama.cpp-dflash2`, built
`CMAKE_CUDA_ARCHITECTURES=89` -- 141 sm_89 cubins, no sm_120a, no PTX -- on a
machine whose second card is compute capability 12.0. Every sweep since has had
to remember `QWEN38_LLAMA_EXE`, and forgetting it is not a rare mistake: it
produced fifteen published rows on the wrong machine, and it is what the guard
caught again while launching the split-mode sweep.

A guard that fires often is a guard doing its job around a default that is
wrong. The default is now `llama.cpp-blackwell`, the binary the serving
profiles use, so the arena measures what we serve unless told otherwise.

`QWEN38_LLAMA_EXE` still overrides it -- that is how the patched mirror build
gets measured -- and the boot-time guard still runs, because a default cannot
know what a future card will be.
"""
import os
import re
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
sys.path.insert(0, BENCH)

import dflash2_arena as A  # noqa: E402


def test_the_default_is_the_blackwell_build():
    assert A.DEFAULT_EXE.endswith("llama-server.exe"), A.DEFAULT_EXE
    assert "llama.cpp-blackwell" in A.DEFAULT_EXE, A.DEFAULT_EXE


def test_the_default_is_not_an_ada_only_build():
    """Named explicitly so a future edit back to one is a red test, not a
    silently plausible sweep."""
    for ada in ("llama.cpp-dflash2", "llama.cpp-cuda"):
        assert ada not in A.DEFAULT_EXE, A.DEFAULT_EXE


def test_the_default_is_what_the_serving_profile_launches():
    """The invariant that matters: the arena measures the served binary.

    Read out of the profile rather than hardcoded here, so the two cannot drift
    apart without this going red.
    """
    profile = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q4-dual.ps1")
    with open(profile, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    m = re.search(r'\[string\]\$Exe\s*=\s*"([^"]+)"', text)
    assert m, "could not find the profile's default $Exe"
    assert m.group(1) == A.DEFAULT_EXE, (m.group(1), A.DEFAULT_EXE)


def test_the_env_var_still_overrides():
    """The mirror build is measured through it; removing the override would
    make the DFlash2 work unmeasurable."""
    from provenance import resolve_exe, ENV_VAR
    old = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = r"C:\somewhere\else\llama-server.exe"
    try:
        assert resolve_exe(A.DEFAULT_EXE) == r"C:\somewhere\else\llama-server.exe"
    finally:
        if old is None:
            del os.environ[ENV_VAR]
        else:
            os.environ[ENV_VAR] = old
