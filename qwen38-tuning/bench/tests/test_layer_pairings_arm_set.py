r"""DFlash2 against MTP on BOTH builds, under the one split where both load.

THE QUESTION, AND WHY IT HAS A SHAPE INSTEAD OF BEING ONE RUN

Which drafter is faster on Unsloth's build 10679 -- DFlash2 or MTP? It cannot
be asked in the configuration this project serves. Under `-sm tensor` DFlash2
aborts on build 10679 at `ggml-backend-meta.cpp:543`,
`GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)` (issue #52, commit
5f87e12). Probed at ctx 65,536 on UD-Q4_K_XL, all four combinations load under
`-sm layer` and only there, so the split mode is FORCED rather than chosen.

WHY BOTH BUILDS RUN IN THE SAME ROUNDS

`+26 % from the newer build` is currently supported by two single readings from
two different boots, against a measured 48.9 % same-arm drift at this depth
(CORRECTIONS 23). Quoting the existing tables for our side would repeat exactly
that error while looking like a control. Three more arms in the same rotation
cost one pass and make the build a paired variable instead of a confound.

THE FOUR HAZARDS THIS FILE EXISTS FOR

1. A MIXED SPLIT MODE. If any arm carried `-sm tensor` the set would vary the
   split AND the build at once, and no delta in it could be attributed. This is
   the shape of CORRECTIONS 26 and 28.

2. THE ARMS DRIFTING APART. The decoder argv must be byte-identical across the
   two builds. If it is not, "their build is faster" and "their arm had a
   different flag" are the same number.

3. CORRECTIONS 34, THIRD FORM. That entry is about a column recording the
   MODULE DEFAULT rather than what the arm ran, so every row named the wrong
   file while the guarding test stayed green. Here the target is UD-Q4_K_XL and
   the module default is UD-IQ2_XXS -- a set that inherited the default would
   run a different model than the one the issue describes and say nothing.

4. `QWEN38_LLAMA_EXE` MOVING OUR SIDE UNDER US. `arm_exe` falls back to the
   module `EXE`, which `resolve_exe` reads from that variable at import. A
   developer who exported it -- which is how the patched mirror gets measured --
   would turn this A/B into B/B or A/C, silently, and every row would still
   carry a plausible number. Our arms therefore PIN their binary rather than
   inheriting it.

AND THE LOADER PATH. Studio's binary finds NO CUDA device with a bare PATH and
serves from the CPU without saying so, because CUDA 13 keeps cudart64_13.dll in
%CUDA_PATH%\bin\x64. A sweep that did that returns a full set of believable
numbers from the wrong hardware.
"""
import os
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as arena  # noqa: E402

SET = "layer-pairings"
STUDIO_EXE = os.path.join(os.path.expanduser("~"), ".unsloth", "llama.cpp",
                          "build", "bin", "Release", "llama-server.exe")
CTX = 65536


def arms():
    return arena.ARM_SETS[SET]


def parts():
    return [arena.arm_parts(a) for a in arms()]


def exe_of(env):
    return (env or {}).get(arena.ENV_VAR) or arena.EXE


def decoder_of(extra):
    """The `--spec-type` value, which is what names the arm's decoder."""
    for i, tok in enumerate(extra):
        if tok == "--spec-type" and i + 1 < len(extra):
            return extra[i + 1]
    return None


# --------------------------------------------------------------- the set

def test_the_arm_set_exists():
    assert SET in arena.ARM_SETS


def test_it_is_three_decoders_on_two_builds():
    assert len(arms()) == 6, [p[0] for p in parts()]


def test_each_build_carries_the_same_three_decoders():
    by_exe = {}
    for _, extra, env in parts():
        by_exe.setdefault(exe_of(env), []).append(decoder_of(extra))
    assert len(by_exe) == 2, list(by_exe)
    a, b = by_exe.values()
    assert sorted(a) == sorted(b), by_exe


def test_the_incumbent_is_present_on_both_builds():
    """Without `ngram-mod` the two drafter figures float: there is nothing for
    them to be a percentage OF, and the run answers a comparison nobody asked
    for instead of the one in the issue."""
    plain = [decoder_of(e) for _, e, _ in parts()].count("ngram-mod")
    assert plain == 2, [decoder_of(e) for _, e, _ in parts()]


def test_both_drafters_are_present_on_both_builds():
    kinds = [decoder_of(e) for _, e, _ in parts()]
    assert kinds.count("draft-dflash,ngram-mod") == 2, kinds
    assert kinds.count("draft-mtp,ngram-mod") == 2, kinds


# ------------------------------------------- hazard 1: one split mode only

def test_every_arm_uses_the_layer_split():
    """`-sm layer` is forced, not chosen: DFlash2 cannot load under `-sm
    tensor` on build 10679 at all."""
    for label, extra, _ in parts():
        assert "-sm" in extra, label
        assert extra[extra.index("-sm") + 1] == "layer", (label, extra)


def test_no_arm_smuggles_in_the_tensor_split():
    for label, extra, _ in parts():
        assert "tensor" not in extra, (
            "one arm on the tensor split varies the split AND the build, and "
            "no delta in the set could be attributed", label)


def test_no_arm_passes_a_tensor_ratio():
    """`-ts` is meaningless under `-sm layer` -- llama.cpp divides by free VRAM
    there -- and passing it to some arms and not others would be a second
    variable wearing a no-op's clothes."""
    for label, extra, _ in parts():
        assert "-ts" not in extra, (label, extra)


# --------------------------------- hazard 2: the two builds run the same argv

def test_the_same_decoder_carries_identical_argv_on_both_builds():
    by_decoder = {}
    for label, extra, env in parts():
        by_decoder.setdefault(decoder_of(extra), []).append((label, extra))
    for decoder, got in by_decoder.items():
        assert len(got) == 2, (decoder, got)
        (la, ea), (lb, eb) = got
        assert ea == eb, (
            "the two builds run different flags for one decoder, so a delta "
            "between them has two causes", decoder, la, lb, ea, eb)


def test_the_micro_batch_is_held_constant():
    for label, extra, _ in parts():
        assert "-ub" in extra, label
        assert extra[extra.index("-ub") + 1] == "1024", (label, extra)


# ------------------------------------- hazard 3: the target is in the arm

def test_every_arm_names_its_target_explicitly():
    for label, extra, _ in parts():
        assert "-m" in extra, (
            "the arm inherits the module default, which is a different "
            "artifact than the one this experiment is about", label)


def test_the_target_is_the_q4_k_xl_file_with_the_mtp_head():
    """`draft-mtp` has no sidecar -- the head lives in the main file -- so on
    an artifact without one the MTP arm does not merely lose, it cannot run."""
    for label, extra, _ in parts():
        assert "UD-Q4_K_XL" in extra[extra.index("-m") + 1], (label, extra)


def test_the_row_records_that_target_and_not_the_module_default():
    """CORRECTIONS 34 in its third form. `arm_target` reads the LAST `-m` of
    the resolved argv, which is the same answer llama.cpp gives itself."""
    for label, extra, _ in parts():
        got = arena.arm_target(CTX, extra)
        assert "UD-Q4_K_XL" in got, (label, got)
    assert "UD-Q4_K_XL" not in arena.TARGET, (
        "the module default already IS this file, so this test proves "
        "nothing and the hazard it guards is unguarded")


# ---------------------------- hazard 4: neither side inherits its binary

def test_our_arms_pin_the_served_binary_rather_than_inheriting_it():
    """`resolve_exe` reads QWEN38_LLAMA_EXE at import. Exporting it is how the
    patched mirror gets measured, and it would turn this A/B into B/B without
    changing a single label."""
    ours = [(l, env) for l, e, env in parts()
            if exe_of(env) != STUDIO_EXE]
    assert len(ours) == 3, ours
    for label, env in ours:
        assert arena.ENV_VAR in (env or {}), (
            "this arm inherits the module default and moves with the "
            "environment", label)


def test_our_arms_name_the_served_build_not_the_mirror():
    for label, _, env in parts():
        exe = exe_of(env)
        if exe != STUDIO_EXE:
            assert "llama.cpp-blackwell" in exe, (label, exe)
            assert "mirror" not in exe, (
                "the mirror is the PATCHED build; using it here would make "
                "the patch a third variable", label, exe)


def test_their_arms_run_studios_binary():
    theirs = [l for l, _, env in parts() if exe_of(env) == STUDIO_EXE]
    assert len(theirs) == 3, theirs


def test_the_labels_name_the_build():
    for label, _, env in parts():
        want = "10679" if exe_of(env) == STUDIO_EXE else "10499"
        assert want in label, (
            "a label that does not name its build makes the JSONL unreadable "
            "without cross-referencing the exe column", label, want)


# ------------------------------------------------- the CPU-run fault

def test_their_arms_carry_the_cuda_loader_path():
    for label, _, env in parts():
        if exe_of(env) == STUDIO_EXE:
            path = (env or {}).get("PATH", "")
            assert "x64" in path, (
                r"CUDA 13 keeps cudart64_13.dll in %CUDA_PATH%\bin\x64, and "
                r"their binary finds no GPU without it", label, path)


def test_our_arms_do_not_inherit_that_path():
    """Our binary ships its own cublas beside it. Prepending a CUDA directory
    to one side only is fine; prepending it to both for symmetry would be a
    change nobody measured."""
    for label, _, env in parts():
        if exe_of(env) != STUDIO_EXE:
            assert "PATH" not in (env or {}), (label, env)


# ----------------------------------------------- the drafters themselves

def test_the_dflash_arms_carry_their_sidecar():
    for label, extra, _ in parts():
        if decoder_of(extra) == "draft-dflash,ngram-mod":
            assert "-md" in extra, (
                "draft-dflash is a second model file; without -md the arm "
                "silently runs as plain ngram-mod", label)
            assert extra[extra.index("-ngld") + 1] == "99", (label, extra)


def test_the_mtp_arms_carry_no_sidecar():
    """The nextn head is inside UD-Q4_K_XL. A sidecar would add about 1.4 GB
    for nothing and quietly make this a different experiment."""
    for label, extra, _ in parts():
        if decoder_of(extra) == "draft-mtp,ngram-mod":
            assert "-md" not in extra, (label, extra)


def test_each_drafter_sets_its_draft_depth_explicitly():
    """`--spec-draft-n-max` defaults to 3 (common.h:325) and is the largest
    lever found on this axis. Leaving either arm at the default would make the
    depth a property of the code rather than of the experiment."""
    want = {"draft-dflash,ngram-mod": "4", "draft-mtp,ngram-mod": "3"}
    for label, extra, _ in parts():
        d = decoder_of(extra)
        if d in want:
            assert "--spec-draft-n-max" in extra, (label, extra)
            got = extra[extra.index("--spec-draft-n-max") + 1]
            assert got == want[d], (label, d, got)


def test_the_ngram_window_is_the_served_one_everywhere():
    """Every arm pairs with `ngram-mod`, and it must be the tuned window the
    worker profiles serve rather than llama.cpp's defaults -- otherwise the
    incumbent in this table is not the incumbent anybody runs."""
    for label, extra, _ in parts():
        for flag, value in zip(arena.NGRAM[::2], arena.NGRAM[1::2]):
            assert flag in extra, (label, flag)
            assert extra[extra.index(flag) + 1] == value, (label, flag)


# --------------------------------------------------------- both cards

def test_every_arm_sees_both_cards():
    """The module pins the served card. This experiment is a two-card one, and
    an arm that lost the pin lift would run 66 layers on one 16 GB card and
    report it beside five arms that did not."""
    for label, _, env in parts():
        assert (env or {}).get("CUDA_VISIBLE_DEVICES") == arena.BOTH_CARDS, \
            (label, (env or {}).get("CUDA_VISIBLE_DEVICES"))


# ------------------------------------------------ it refuses rather than lies

def test_a_missing_studio_binary_stops_the_run():
    for label, extra, env in parts():
        if exe_of(env) == STUDIO_EXE and not os.path.isfile(STUDIO_EXE):
            with pytest.raises(Exception):
                arena.server_argv(CTX, extra, env=env, verify=True)
            return
    pytest.skip("Studio's binary is present; the refusal path needs it absent")
