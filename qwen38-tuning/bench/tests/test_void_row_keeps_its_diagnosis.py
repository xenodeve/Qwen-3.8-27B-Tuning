r"""A voided row must still say WHY it was voided, and the layer count must not
be a constant that names one artifact.

INSTRUMENT FAULT, 2026-08-24, issue #44. Eighteen rows at ctx 147,456 on
`UD-Q2_K_XL` came back `measurable: False` with this note on every one:

    ValueError: no assignment pass has 65 layers; passes seen: [66, 66, 66]

Two separate defects, stacked, and the second hid the first.

DEFECT ONE -- `TARGET_LAYERS = 65`, a module constant carrying the comment
"Qwen3.8-27B: 64 blocks plus the MTP head". `UD-IQ2_XXS`, the artifact the arena
defaulted to, has **no** MTP head: 64 blocks plus output is 65, and the comment
described a model it was not measuring. `UD-Q2_K_XL` does have one -- `blk.64`
-- so its passes are 66 and no pass ever matches. `parse_layer_split` behaved
exactly as designed and raised rather than falling back to another pass; the
caller handed it a number that named the wrong model.

DEFECT TWO -- the `except` in `run_arm` did this:

    if not measurable:
        row["note"] = "generations too short to measure: predicted_n=..."
    ...
    row["split"] = parse_layer_split(text, expect_layers=TARGET_LAYERS)   # raises
    except Exception as exc:
        row["note"] = "%s: %s" % (type(exc).__name__, exc)                # CLOBBERS

The rows were already unmeasurable for a real and completely different reason --
every generation produced **9 tokens against a 512-token budget** -- and the note
that said so was overwritten by a complaint about layer counting. The evidence
that identified the actual problem survived only because `predicted_n` happens to
be its own column.

This is the invariant the 2026-08-24 ship log records as bought with a session:
**a harness that deletes its own evidence cannot be debugged.** It was fixed in
`real_task_bench` that day for transcripts, and the same shape was live here.

WHAT IS NOT CLAIMED: that 65 should have been 66. Replacing one artifact's
constant with another artifact's constant is the same defect wearing a new
number. The target is identified from the log as the model with the most layers,
which is true of every configuration this project runs -- the DFlash2 drafter is
6 against a target's 65 or 66. If a drafter ever has more layers than its target
this inverts, and that is why the rule is stated here rather than assumed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dflash2_arena as arena
from harness import target_layer_count


def _pass(n, device="CUDA0", first_cpu_at=None):
    out = []
    for i in range(n):
        d = "CPU" if first_cpu_at is not None and i >= first_cpu_at else device
        out.append(f"load_tensors: layer {i:>3} assigned to device {d}, is_swa = 0")
    return "\n".join(out)


# The two shapes this project actually produces.
SIDECAR_LOG = "\n".join([_pass(6), _pass(65), _pass(65), _pass(6)])   # DFlash2 + IQ2_XXS
BAKED_IN_LOG = "\n".join([_pass(66), _pass(66), _pass(66)])           # draft-mtp + Q2_K_XL


def test_it_finds_the_target_under_a_sidecar_drafter():
    """The drafter's 6-layer pass is last; the target's 65 is what we mean."""
    assert target_layer_count(SIDECAR_LOG) == 65


def test_it_finds_the_target_when_the_head_is_baked_in():
    """UD-Q2_K_XL: 64 blocks, the MTP block at blk.64, and output."""
    assert target_layer_count(BAKED_IN_LOG) == 66


def test_it_raises_rather_than_guessing_when_there_are_no_lines():
    with pytest.raises(ValueError):
        target_layer_count("no assignment lines here")


def test_the_derived_count_makes_the_split_readable_on_both_shapes():
    """The whole point: the pair must work where the constant did not."""
    from harness import parse_layer_split
    for log, expect in ((SIDECAR_LOG, 65), (BAKED_IN_LOG, 66)):
        n = target_layer_count(log)
        assert n == expect
        gpu, cpu = parse_layer_split(log, expect_layers=n)
        assert gpu + cpu == expect


def test_a_spill_in_the_target_is_still_visible():
    """A count that silently picked the drafter could never show this."""
    from harness import parse_layer_split
    spilled = "\n".join([_pass(6), _pass(66, first_cpu_at=64), _pass(6)])
    n = target_layer_count(spilled)
    assert n == 66
    assert parse_layer_split(spilled, expect_layers=n) == (64, 2)


def test_the_arena_no_longer_carries_an_artifact_specific_layer_constant():
    """65 was right for the artifact it was written against and wrong for the
    one we serve. A constant cannot be right for both."""
    assert not hasattr(arena, "TARGET_LAYERS"), (
        "TARGET_LAYERS is back; the count must come from the log, not from a "
        "number that names one artifact")


def test_a_later_failure_cannot_erase_an_earlier_diagnosis():
    """`run_arm` records why a row is unmeasurable, then does more work that can
    raise. The later error must not overwrite the earlier explanation."""
    row = {"note": "generations too short to measure: predicted_n=[9, 9, 9] "
                   "against n_predict=512"}
    arena.record_fault(row, ValueError("no assignment pass has 65 layers"))
    assert "predicted_n=[9, 9, 9]" in row["note"], (
        "the original diagnosis was clobbered: %r" % row["note"])
    assert "no assignment pass" in row["note"], "the later fault must also appear"


def test_a_fault_with_no_earlier_note_reads_normally():
    row = {}
    arena.record_fault(row, ValueError("boom"))
    assert row["note"] == "ValueError: boom"
