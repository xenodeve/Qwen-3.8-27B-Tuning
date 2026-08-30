r"""DFlash2 on NVFP4, at the depth and settings where DFlash2 is at its best.

THE QUESTION, AND WHY IT IS BEING ASKED AGAIN

`results/nvfp4-dflash-147456.jsonl` put `draft-dflash,ngram-mod` at **+0.2 %
with the sign flipping** against `draft-mtp,ngram-mod` on NVFP4, and that number
is why the ledger says DFlash2 has no case on this artifact. It is also the
number standing between the developer and a smaller head-less NVFP4 file
(`esatapedico/...-BUDGET`, 134 MiB smaller; `...-STARVED`, 257 MiB), because
stripping the MTP head only makes sense if something else drafts better.

**That run gave DFlash2 none of what it is now known to want.** It was ctx
147,456 -- above DFlash2's 131,072 ceiling on `UD-Q4_K_XL` and more than twice
its measured best of 65,536 -- at `--spec-draft-n-max 3`, and with the n-gram at
`n-match 12`, a window the register records as COLLAPSING on this artifact
(acceptance 55.4 -> 22.1) while 24 wins.

So this set gives it every one of those, at once:

  ctx        65,536   where DFlash2 peaked on the other artifact
  n_max      4        measured best 2026-08-30: 55.72 against 52.64 at 7
  n-match    24       NVFP4's own tuned window; +63.1 % was measured with it

**Three variables move together, deliberately.** The question is binary -- does
DFlash2 have a case on NVFP4 AT ALL -- and a run that handicapped it could not
answer that. If it wins here, attributing the win to depth, draft size or the
n-gram window is a SEPARATE experiment, and the write-up has to say so.

WHY MTP IS NOT AN ARM

The developer excluded it: it has been measured repeatedly. The consequence is
stated where the result will be read -- **this set cannot compare DFlash2 to
MTP**, because MTP's NVFP4 figures are from other boots and this depth has a
measured 48.9 % same-arm drift (CORRECTIONS 23). Quoting across that gap is how
`+26 %` happened. What this set CAN answer is DFlash2 against the incumbent, in
one rotation.

WHY THE INCUMBENT IS STILL HERE

Without it the DFlash2 figure has nothing in its own rounds to be a percentage
of, and the only available comparison would be the cross-boot one this file just
said not to make.
"""
import os
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as arena  # noqa: E402

SET = "nvfp4-dflash-65536"
MIRROR = arena.MIRROR_EXE
CTX = 65536


def parts():
    return [arena.arm_parts(a) for a in arena.ARM_SETS[SET]]


def decoder_of(extra):
    for i, tok in enumerate(extra):
        if tok == "--spec-type" and i + 1 < len(extra):
            return extra[i + 1]
    return None


def value_of(extra, flag):
    return extra[extra.index(flag) + 1] if flag in extra else None


# --------------------------------------------------------------------- the set

def test_the_arm_set_exists():
    assert SET in arena.ARM_SETS


def test_it_is_the_incumbent_and_dflash():
    got = sorted(decoder_of(e) for _, e, _ in parts())
    assert got == ["draft-dflash,ngram-mod", "ngram-mod"], got


def test_mtp_is_not_in_it():
    """Excluded on purpose. The docstring records what that costs."""
    for label, extra, _ in parts():
        assert "draft-mtp" not in (decoder_of(extra) or ""), label


# ------------------------------------------- DFlash2 gets its best-known setting

def test_the_draft_depth_is_the_measured_best():
    """4, not the 3 the +0.2 % run used. Measured 2026-08-30 on the other
    artifact: 55.72 against 52.64 at the clamp of 7."""
    for label, extra, _ in parts():
        if decoder_of(extra) == "draft-dflash,ngram-mod":
            assert value_of(extra, "--spec-draft-n-max") == "4", (label, extra)
            return
    pytest.fail("no dflash arm")


def test_the_ngram_window_is_the_one_TUNED_FOR_THIS_ARTIFACT():
    """n-match 12 is the UD-Q4_K_XL window and the register records it
    collapsing here -- acceptance 55.4 -> 22.1. 24 is what +63.1 % was measured
    with. Using 12 would handicap both arms on the artifact under test."""
    for label, extra, _ in parts():
        assert value_of(extra, "--spec-ngram-mod-n-match") == "24", (label, extra)


def test_both_arms_carry_the_same_ngram_window():
    windows = {tuple(e[e.index("--spec-ngram-mod-n-match"):][:6]) for _, e, _ in parts()}
    assert len(windows) == 1, windows


# ----------------------------------------------------- held at the served shape

def test_every_arm_loads_the_served_nvfp4_file():
    for label, extra, _ in parts():
        assert value_of(extra, "-m") == arena.NVFP4_VERY_LOW, (label, extra)


def test_every_arm_uses_the_tensor_split_with_its_ratio():
    for label, extra, _ in parts():
        assert value_of(extra, "-sm") == "tensor", (label, extra)
        assert value_of(extra, "-ts") == "7819,15490", (
            "-sm tensor without the ratio is the 0.38 tok/s configuration "
            "(CORRECTIONS 33)", label)


def test_every_arm_holds_the_micro_batch():
    for label, extra, _ in parts():
        assert value_of(extra, "-ub") == "1024", (label, extra)


def test_every_arm_sees_both_cards():
    for label, _, env in parts():
        assert (env or {}).get("CUDA_VISIBLE_DEVICES") == arena.BOTH_CARDS, label


# --------------------------------------------------------------- the binary

def test_every_arm_pins_the_mirror():
    """`draft-dflash` under `-sm tensor` aborts on an unpatched binary, and an
    arm that inherits the module default runs whatever QWEN38_LLAMA_EXE holds.
    Both arms pin it so the pair differs only in the decoder."""
    for label, _, env in parts():
        assert (env or {}).get(arena.ENV_VAR) == MIRROR, (label, env)


def test_the_row_records_the_mirror():
    label, extra, env = parts()[0]
    r = arena.new_row(CTX, label, 1, "synthetic", extra, env, 1000)
    assert r["exe"] == MIRROR, r["exe"]


# ------------------------------------------------------------------ the drafter

def test_the_dflash_arm_uses_the_SMALL_drafter():
    """The 535 MiB Q2_K_S-MIX file, which is what the +0.2 % run used and what
    reaches depth. Swapping the drafter as well would add a fourth variable to
    a set that already moves three."""
    for label, extra, _ in parts():
        if decoder_of(extra) == "draft-dflash,ngram-mod":
            assert value_of(extra, "-md") == arena.DFLASH_SMALL, (label, extra)
            assert value_of(extra, "-ngld") == "99", (label, extra)


def test_the_incumbent_carries_no_drafter():
    for label, extra, _ in parts():
        if decoder_of(extra) == "ngram-mod":
            assert "-md" not in extra, (label, extra)
            assert "--spec-draft-n-max" not in extra, (label, extra)
