r"""Icon 2 serves the upstream build, and the switch that puts it there.

`-Upstream` was added 2026-09-02 at the developer's request. **It is not a speed
change and the tests say so**: measured against the served 10499 at ctx 147,456
on the real-code corpus, upstream 10729 is **-0.16 %**, inside the noise
(`results/builds-nvfp4-147456.jsonl`, four arms, all within +/-1.1 %). What it
buys is that the newer source can be SERVED rather than only benchmarked.

**Why a switch and not an edited default.** `C:\AI\llama.cpp-blackwell` is read
by icons 1, 2, 3, 4, 7, 8, 9 and C. Overwriting it would move all of them at
once with no marker in any recorded row -- the failure `build-dflash2.ps1`'s
header exists to warn about, in its own words: *"Overwriting that directory would
switch the runtime under all of them at once, silently."*

**And why it refuses rather than falls back.** Three other switches already
choose the executable. A run whose binary two of them disagree about, or one that
quietly serves the old binary because the new one is missing, is CORRECTIONS 41 --
a build comparison whose arms shared a binary while every row named the pin.
"""
import os
import re
import subprocess
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
SERVE = os.path.join(ROOT, "serve.ps1")
LAUNCHERS = os.path.join(ROOT, "launchers")
STAGED = r"C:\AI\llama.cpp-upstream\llama-server.exe"


def _serve_src():
    return open(SERVE, encoding="utf-8", errors="replace").read()


def test_the_switch_exists():
    assert re.search(r"\[switch\]\$Upstream\b", _serve_src())


def test_both_icon_2_launchers_ask_for_it():
    """The LAN twin is spelled out separately in this repo and has been forgotten
    before; if only one carries the switch, the two icons serve different builds."""
    for name in ("serve-dual-nvfp4-deep.bat", "serve-dual-nvfp4-deep-lan.bat"):
        src = open(os.path.join(LAUNCHERS, name), encoding="utf-8", errors="replace").read()
        assert "-Upstream" in src, name
        assert "-Dual -Nvfp4 -Deep -Vision" in src, name


def test_no_other_launcher_was_moved_onto_the_new_binary():
    """The request was icon 2. Every other icon keeps the binary it was measured
    on, and a stray `-Upstream` elsewhere would be scope growth nobody asked for."""
    for name in sorted(os.listdir(LAUNCHERS)):
        if not name.endswith(".bat") or name.startswith("serve-dual-nvfp4-deep"):
            continue
        src = open(os.path.join(LAUNCHERS, name), encoding="utf-8", errors="replace").read()
        assert "-Upstream" not in src, name


def test_it_refuses_to_share_the_binary_choice_with_another_switch():
    src = _serve_src()
    block = src[src.index("if ($Upstream) {"):]
    block = block[:block.index("\n}")]
    for other in ("$Dflash", "$TheirBuild", "$TheirMirror"):
        assert other in block, other
    assert "exit 1" in block


def test_it_refuses_rather_than_falling_back_when_the_binary_is_absent():
    src = _serve_src()
    block = src[src.index("if ($Upstream) {"):]
    block = block[:block.index("\n}")]
    assert "Test-Path" in block and "exit 1" in block


@pytest.mark.skipif(not os.path.exists(STAGED), reason="upstream build not staged")
def test_the_staged_binary_is_the_build_the_measurement_used():
    """`results/builds-nvfp4-147456.jsonl` measured commit 458681e1d. A different
    binary at this path would make the ledger row describe something else."""
    out = subprocess.run([STAGED, "--version"], capture_output=True, text=True,
                         cwd=os.path.dirname(STAGED), timeout=60)
    text = (out.stdout or "") + (out.stderr or "")
    assert "458681e1d" in text, text[:200]
    assert "10729" in text, text[:200]


@pytest.mark.skipif(not os.path.exists(STAGED), reason="upstream build not staged")
def test_the_staged_directory_carries_the_cuda_runtime_beside_the_exe():
    """Not produced by the build. Without them llama-server does not start, and a
    previous script in this repo reported success while the binary could not load."""
    here = os.listdir(os.path.dirname(STAGED))
    for stem in ("cudart64_", "cublas64_", "cublasLt64_", "ggml-cuda", "llama-server-impl"):
        assert any(f.startswith(stem) for f in here), (stem, here)
