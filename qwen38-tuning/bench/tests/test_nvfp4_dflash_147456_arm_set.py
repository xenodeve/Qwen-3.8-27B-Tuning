r"""DFlash2 on NVFP4 at 147,456 — the depth MTP's figures already live at.

WHY A SINGLE ARM, WHICH THIS BENCH NORMALLY REFUSES

`nvfp4-dflash-65536` settled that DFlash2 works on NVFP4: **+67.9 %**
[+65.8, +71.5] RESOLVED over the incumbent, acceptance 50.0 against the old
run's 22.1. What it could not do is compare DFlash2 to MTP, because MTP's NVFP4
figures are at 147,456 and that is a different depth and a different boot.

The obvious fix is a three-arm rotation at one depth. The developer declined it,
on the grounds that MTP has been measured repeatedly at this exact setting, and
**checking that claim is what makes this run admissible**:

    nvfp4-final-147456.jsonl        nvfp4-mtp+nm24   39.43 / 42.61 / 42.55
    nvfp4-ngram-retune-147456.jsonl mtp+nm24         43.10 / 42.99 / 42.93

Six rounds, **two independent boot series**, same artifact, same depth, same
`n-match 24`, same `--spec-draft-n-max 3`. Five of the six fall in 42.5-43.1 and
the whole range spans **9.3 %**.

That is the number that decides it. CORRECTIONS 23 measures up to 48.9 %
same-arm drift at depth, and quoting across a gap like that is how `+26 %`
happened — but this comparator does not have that spread. It has 9.3 % across
two boots. **A cross-boot comparison against a six-round, two-series, 9.3 %
comparator is a different act from one against a single reading**, and this file
exists so the next reader is told which one this is.

WHAT THIS ARM MUST MATCH, OR THE COMPARISON IS NOT ONE

Everything MTP's rows held: `NVFP4 VERY-LOW`, ctx 147,456, `-sm tensor -ts
7819,15490 -ub 1024`, `n-match 24`, both cards, `real-code-vendor`, the patched
mirror. The ONE thing that differs is the decoder — plus `--spec-draft-n-max`,
which is 3 for MTP and 4 for DFlash2 because those are each drafter's own
measured best and matching them would match a number that does not mean the
same thing twice (`nextn_predict_layers = 1` against `block_size = 8`).

THE RISK, STATED BEFORE THE RUN

At 65,536 this arm finished with 2,828 MiB free. 147,456 adds about 1,440 MiB of
KV at 18.00 KiB/token, and `n_max` 2 -> 4 adds another 299, leaving roughly
1,100. The earlier 147,456 DFlash2 run used `n_max 2` and did load. **If this
arm fails to load, that is the result** and the fallback is `n_max 2`.
"""
import os
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as arena  # noqa: E402

SET = "nvfp4-dflash-147456"
CTX = 147456


def parts():
    return [arena.arm_parts(a) for a in arena.ARM_SETS[SET]]


def value_of(extra, flag):
    return extra[extra.index(flag) + 1] if flag in extra else None


def test_the_arm_set_exists():
    assert SET in arena.ARM_SETS


def test_it_is_one_arm():
    """Deliberate, and the docstring carries the justification: the comparator
    is six rounds across two boot series spanning 9.3 %."""
    assert len(parts()) == 1, [p[0] for p in parts()]


def test_it_is_the_dflash_pairing():
    _, extra, _ = parts()[0]
    assert value_of(extra, "--spec-type") == "draft-dflash,ngram-mod", extra


# ---------------------------------- everything MTP's rows held, held identically

def test_it_loads_the_same_nvfp4_file():
    _, extra, _ = parts()[0]
    assert value_of(extra, "-m") == arena.NVFP4_VERY_LOW, extra


def test_it_carries_the_ngram_window_mtp_was_measured_with():
    """`n-match 24`. Using 12 would make this incomparable with the rows it
    exists to be compared against -- and 12 is the window this project records
    collapsing on NVFP4."""
    _, extra, _ = parts()[0]
    assert value_of(extra, "--spec-ngram-mod-n-match") == "24", extra


def test_it_uses_the_tensor_split_with_its_ratio():
    _, extra, _ = parts()[0]
    assert value_of(extra, "-sm") == "tensor", extra
    assert value_of(extra, "-ts") == "7819,15490", extra


def test_it_holds_the_micro_batch():
    _, extra, _ = parts()[0]
    assert value_of(extra, "-ub") == "1024", extra


def test_it_sees_both_cards():
    _, _, env = parts()[0]
    assert (env or {}).get("CUDA_VISIBLE_DEVICES") == arena.BOTH_CARDS


def test_it_pins_the_mirror():
    """`draft-dflash` under `-sm tensor` aborts on an unpatched binary, and
    after CORRECTIONS 41 an arm that inherits the module default is the exact
    failure to avoid."""
    _, _, env = parts()[0]
    assert (env or {}).get(arena.ENV_VAR) == arena.MIRROR_EXE, env


def test_the_row_records_the_mirror_and_the_nvfp4_target():
    """CORRECTIONS 34 was found in these very files: every `nvfp4-*` row at
    147,456 records `UD-Q4_K_XL` in its `target` column while its argv names
    NVFP4. The rates survive; the column did not. This asserts the new rows do
    not repeat it."""
    label, extra, env = parts()[0]
    r = arena.new_row(CTX, label, 1, "synthetic", extra, env, 1000)
    assert r["exe"] == arena.MIRROR_EXE, r["exe"]
    assert "NVFP4" in r["target"], r["target"]


# ------------------------------------------------ the drafter and its depth

def test_it_uses_the_small_drafter():
    _, extra, _ = parts()[0]
    assert value_of(extra, "-md") == arena.DFLASH_SMALL, extra
    assert value_of(extra, "-ngld") == "99", extra


def test_the_draft_depth_is_dflash2s_own_best_not_mtps():
    """4, measured best for DFlash2 on 2026-08-30. MTP's rows are at 3, its own
    default, because `nextn_predict_layers = 1` and `block_size = 8` do not mean
    the same thing -- matching the two would match a number, not a setting."""
    _, extra, _ = parts()[0]
    assert value_of(extra, "--spec-draft-n-max") == "4", extra


def test_the_label_names_the_settings_that_make_it_comparable():
    label, _, _ = parts()[0]
    assert "nm24" in label.replace(" ", ""), label
    assert "n4" in label.replace(" ", ""), label
