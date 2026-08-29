r"""An arm may choose the BINARY, and the row must record the one it chose.

WHY

`icon B` against `icon 7` is one reading per side, taken in different boots,
against a measured 48.9 % same-arm drift at depth (CORRECTIONS 23). +26 % does
not survive that as evidence. What settles it is the thing this bench was built
for: both binaries alternating inside ONE session, several rounds, order
rotated.

The arena could not do it. `EXE` is resolved once at import from
`QWEN38_LLAMA_EXE`, so every arm in a run used the same binary and an arm's
`env` could not change it — the one comparison that needed pairing was the one
comparison the harness could not pair.

THE HAZARD THIS FILE EXISTS FOR

Letting an arm choose the binary immediately creates CORRECTIONS 34 a second
time. That entry is about the `target` column recording the MODULE DEFAULT for
every row, so every NVFP4 arm was written down as having run the Q4 control's
file. `exe` and `cuda_archs` are the same shape of column and would fail the
same way: a build A/B whose rows all name build 10499 is worse than no A/B,
because it looks complete.

So the first assertion here is not "can an arm pick a binary" — it is **"does
the row name the binary that arm actually used"**.

AND THE OTHER ONE. Studio's binary finds NO CUDA device with a bare PATH and
serves from the CPU without saying so. A sweep that did that would produce a
full set of believable numbers from the wrong hardware. The arm set carries the
loader path, and the harness must refuse rather than warn.
"""
import os
import re
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as arena  # noqa: E402

STUDIO_EXE = os.path.join(os.path.expanduser("~"), ".unsloth", "llama.cpp",
                          "build", "bin", "Release", "llama-server.exe")


# ------------------------------------------------- an arm can choose the binary

def test_server_argv_uses_the_arms_binary():
    argv = arena.server_argv(16384, [], env={"QWEN38_LLAMA_EXE": STUDIO_EXE})
    assert argv[0] == STUDIO_EXE, argv[0]


def test_without_an_env_it_is_still_the_module_default():
    """Every existing arm passes no env and must be untouched."""
    assert arena.server_argv(16384, [])[0] == arena.EXE


def test_an_unrelated_env_does_not_change_the_binary():
    argv = arena.server_argv(16384, [], env={"GGML_CUDA_GRAPH_OPT": "1"})
    assert argv[0] == arena.EXE, argv[0]


# ------------------------------- the row names the binary THAT ARM ran, not EXE

def test_the_row_records_the_arms_binary():
    """CORRECTIONS 34 IN ITS SECOND FORM.

    That entry is about `target` recording the module default for every row, so
    every NVFP4 arm was written down as having run the Q4 control's file while
    the guarding test stayed green. `exe` is the same shape of column. A build
    A/B whose rows all say 10499 is worse than no A/B: it looks complete.
    """
    r = arena.new_row(16384, "theirs", 1, "synthetic", [],
                      {"QWEN38_LLAMA_EXE": STUDIO_EXE}, 1000)
    assert r["exe"] == STUDIO_EXE, r["exe"]


def test_the_row_records_the_arms_architectures_too():
    """`cuda_archs` is read from the dll BESIDE the exe. Reading it beside the
    module default while running another binary is the same lie one column
    over, and it is the column that says whether the run was JIT-ing."""
    r = arena.new_row(16384, "theirs", 1, "synthetic", [],
                      {"QWEN38_LLAMA_EXE": STUDIO_EXE}, 1000)
    ours = arena.new_row(16384, "ours", 1, "synthetic", [], None, 1000)
    if r["cuda_archs"] is None or ours["cuda_archs"] is None:
        pytest.skip("cuobjdump unavailable; None means not checked, not none present")
    assert r["cuda_archs"] != ours["cuda_archs"], (
        "both builds reported the same architectures, which is only true if "
        "the arm's dll was not the one read")


def test_a_row_with_no_env_still_names_the_default():
    r = arena.new_row(16384, "ours", 1, "synthetic", [], None, 1000)
    assert r["exe"] == arena.EXE


# ------------------------------------------------------------- the arm set

def test_the_arm_set_exists():
    assert "build-ab" in arena.ARM_SETS


def test_both_arms_carry_the_same_argv():
    """The whole point. If the two arms differ in any flag, the delta has two
    causes and cannot be attributed to the build -- the shape of CORRECTIONS
    26 and 28."""
    arms = arena.ARM_SETS["build-ab"]
    assert len(arms) == 2, arms
    extras = [arena.arm_parts(a)[1] for a in arms]
    assert extras[0] == extras[1], extras


def test_the_two_arms_name_different_binaries():
    envs = [arena.arm_parts(a)[2] or {} for a in arena.ARM_SETS["build-ab"]]
    exes = {e.get("QWEN38_LLAMA_EXE", arena.EXE) for e in envs}
    assert len(exes) == 2, exes


def test_the_arm_labels_say_which_build():
    labels = [arena.arm_parts(a)[0] for a in arena.ARM_SETS["build-ab"]]
    joined = " ".join(labels)
    assert "10499" in joined and "10679" in joined, (
        "a label that does not name the build makes the JSONL unreadable "
        "without cross-referencing the exe column", labels)


def test_their_arm_carries_the_cuda_loader_path():
    """Studio's binary finds NO CUDA device with a bare PATH and serves from
    the CPU without saying so. A sweep that did that would produce a full set
    of believable numbers from the wrong hardware."""
    for a in arena.ARM_SETS["build-ab"]:
        label, _, env = arena.arm_parts(a)
        if (env or {}).get("QWEN38_LLAMA_EXE") == STUDIO_EXE:
            path = (env or {}).get("PATH", "")
            assert "x64" in path, (
                "CUDA 13 keeps cudart64_13.dll in %CUDA_PATH%\\bin\\x64", label, path)
            return
    pytest.fail("no arm in build-ab runs Studio's binary")


def test_our_arm_does_not_inherit_their_loader_path():
    """Both arms must differ in the binary and in nothing that could itself
    move a number. Prepending a CUDA directory to one side only is a second
    variable unless the other side is left alone deliberately -- our binary
    ships its own cublas beside it and needs no path."""
    for a in arena.ARM_SETS["build-ab"]:
        label, _, env = arena.arm_parts(a)
        if (env or {}).get("QWEN38_LLAMA_EXE", arena.EXE) == arena.EXE:
            assert "PATH" not in (env or {}), (label, env)
            return
    pytest.fail("no arm in build-ab runs our binary")


# ----------------------------------------------------------- it refuses loudly

def test_a_missing_binary_is_refused_not_defaulted():
    """An arm naming a binary that is not there must stop the run. Falling back
    to the module default would run both arms on one build and report a flat
    sweep as a real comparison."""
    with pytest.raises(Exception):
        arena.server_argv(16384, [], env={
            "QWEN38_LLAMA_EXE": r"C:\nope\llama-server.exe"}, verify=True)


SRC = open(os.path.join(BENCH, "dflash2_arena.py"), encoding="utf-8").read()


def test_the_source_still_records_the_exe_per_row():
    """Guarding the guard: CORRECTIONS 34's own test asserted a literal string
    and stayed green through the fault. This one asserts the column is NOT the
    bare module constant any more."""
    assert "exe=EXE, cuda_archs=cuda_archs(EXE)" not in SRC, (
        "the row still reads the module default; an arm that overrides the "
        "binary will be recorded as having run the wrong one")
