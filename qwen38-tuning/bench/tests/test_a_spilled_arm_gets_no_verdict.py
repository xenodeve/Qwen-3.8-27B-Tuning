"""An arm that ran at a different layer split is not comparable -- no verdict.

THE GAP (found 2026-08-27, task #43). `run_arm` records `row["split"]` from
`parse_layer_split` and prints it in the live line, and then `report()` --
the output that becomes a row in `docs/results/` -- NEVER LOOKS AT IT.

So an arm that silently spilled layers to the CPU is paired against a fully
resident baseline and the difference is attributed to whatever the arm varied.
It does not crash and it does not look wrong: `55+11` against `66+0` produces a
believable number, which is the failure this project exists to refuse.

This is the same defect as the 0.38 tok/s incident one altitude up. There the
spill was silent because `--fit` is inert under `-sm tensor`; here the spill is
VISIBLE in the row and the report throws the observation away.

OBSERVATION, NOT PREDICTION. A launch-time budget guard has to model the
allocator, and this project has twice found its model wrong -- the guard that
counted weights only, then the one that ignored what happens after load. The
split is read from llama.cpp's own load report, so it catches a spill from any
cause, including one no arithmetic anticipated.

WHY MOST OF THIS FILE TESTS A FUNCTION AND NOT THE PRINTED LINE. The first
draft of these tests asserted `"%" not in line.split("]")[1]` and would have
failed on the word `spread 3.3 %` no matter what the verdict said -- an
assertion measuring the shape of a line rather than the property
(`docs/agents/traps.md` 16). `residency_note` returns the value, so the value
is what is asserted; one behavioural test covers the printing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dflash2_arena as arena


# --------------------------------------------------------------- the function

def test_identical_residency_is_comparable():
    assert arena.residency_note(["66+0"], ["66+0"]) is None


def test_a_spilled_arm_is_not_comparable():
    note = arena.residency_note(["66+0"], ["55+11"])
    assert note is not None, "a spilled arm was called comparable"


def test_the_note_names_both_splits():
    """A refusal the reader cannot act on is only half an instrument."""
    note = arena.residency_note(["66+0"], ["55+11"])
    assert "55+11" in note and "66+0" in note, (
        "the reader cannot see WHICH residency differed: %r" % note)


def test_residency_that_moved_between_rounds_is_not_comparable():
    """Three rounds and one of them spilled -- the mean hides it."""
    assert arena.residency_note(["66+0"], ["66+0", "60+6"]) is not None


def test_a_baseline_that_moved_between_rounds_is_not_comparable():
    """Every delta in the block is computed against it, so it voids the block."""
    assert arena.residency_note(["66+0", "55+11"], ["66+0"]) is not None


def test_rows_with_no_split_recorded_stay_comparable_to_each_other():
    """Absence is not a spill.

    Rows predating this field, and fault rows, carry no split. Voiding those
    would retro-void sweeps that were fine. Two unknowns are treated as equal;
    a known against an unknown is not, because that IS a real difference in
    what was verified.
    """
    assert arena.residency_note([None], [None]) is None
    assert arena.residency_note(["66+0"], [None]) is not None


# ------------------------------------------------------------- the report

def _rows(base_split="66+0", arm_split="55+11"):
    out = []
    for rnd, v in enumerate([26.2, 25.6, 26.7], 1):
        out.append(dict(ctx=147456, regime="real-code", arm="ngram-mod-base",
                        round=rnd, tg_med=v, split=base_split))
    for rnd, v in enumerate([9.1, 9.4, 9.2], 1):
        out.append(dict(ctx=147456, regime="real-code", arm="dflash2",
                        round=rnd, tg_med=v, split=arm_split))
    return out


def _line_for(out, arm):
    return next(l for l in out.splitlines() if arm in l and "[" in l)


def test_the_report_refuses_the_delta_and_says_why(capsys):
    """The signature of a delta is its bracketed range, `[-65.5, -63.3]`.

    Checked on `[+` / `[-` rather than on `%`, because the line also carries
    `spread 3.3 %` and a naive `%` check passes or fails for the wrong reason.
    """
    arena.report(_rows())
    line = _line_for(capsys.readouterr().out, "dflash2")
    assert "[+" not in line and "[-" not in line, (
        "the arm ran at a different residency and still got a delta: %s"
        % line.strip())
    assert "55+11" in line and "66+0" in line, (
        "the refusal does not name the residencies: %s" % line.strip())


def test_a_matching_arm_still_gets_its_delta(capsys):
    """The guard must not swallow every comparison it was added to protect."""
    arena.report(_rows(arm_split="66+0"))
    line = _line_for(capsys.readouterr().out, "dflash2")
    assert "[-" in line or "[+" in line, (
        "same residency, so this arm is comparable and must keep its delta: %s"
        % line.strip())
