r"""The worker's transcript must survive the run, and a dead log must not read as a verdict.

TWO INCIDENTS, BOTH FROM 2026-08-24, BOTH IN ONE RUN.

`xeno-skills:306` against the `dflash2+ngram` arm came back:

    FAIL     ctx_high_water=None  wall=617.7s  files=0

and neither half of that line could be acted on.

INCIDENT 1 -- the evidence was deleted by the harness that produced it.

`run_one` writes the worker's stdout to `<scratch>/clones/<repo>-<n>.stdout.txt`.
Its comment says "beside the clone, never inside it, so per-task cleanup cannot
take it", and that is true of `_cleanup`. But `main()` ends with

    shutil.rmtree(scratch, ignore_errors=True)

on the whole timestamped root, which is the clone's *parent*. So the transcript
goes with it -- on every run, PASS or FAIL. Ten minutes of worker output, the
only record of what it did, destroyed at the moment it became interesting.

`edit_canary.py` already solved this: `transcript_path()` puts them under the
BASE scratch directory, and `D:\bench-scratch\transcripts\canary-*.txt` from
2026-08-23 are still there. The real-task harness just never adopted it.

INCIDENT 2 -- the harness read a log the server was not writing.

`--log` defaults to `logs/real-task-server.log`. The run above served from
`logs/dflash2-serve-dflash2-ngram.log`, and `--log` was not passed. So:

    offset = os.path.getsize(a.log)      # 92,903,796 -- a stale file from 08-22

and every subsequent read started past the end of a file that never grew.
`ctx_high_water` came back None, and the row was classified FAIL.

**FAIL is a verdict about the worker.** The worker was never measured. The true
peak was 62,570 tokens against n_ctx 98,304 with `truncated = 0`, sitting in the
log the whole time.

WHAT THIS FILE DELIBERATELY DOES *NOT* CHANGE.

`harness.classify_outcome` says, and is right:

    An unknown high-water stays FAIL: missing data is not evidence of a missing
    window, and excusing a failure on absent evidence is how a benchmark stops
    reporting failures at all.

That rule stays. A task whose log simply had no matching lines is still a FAIL.
What is different here is stronger and checkable: **the log did not grow by a
single byte while a worker ran for ten minutes.** That is not absent evidence
about the window, it is a definite fault in the instrument, and the two must not
share an outcome.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edit_canary
import real_task_bench as rtb


# ------------------------------------------------------- transcript placement

def test_transcript_lives_outside_the_deleted_scratch_root(tmp_path):
    base = tmp_path / "bench-scratch"
    run_root = base / "20260824-011713"
    p = rtb.transcript_path(base, run_root, "xeno-skills", 306)
    assert run_root not in p.parents, (
        f"{p} is inside the run root, which main() deletes wholesale")
    assert base in p.parents


def test_transcript_names_the_run_so_two_runs_do_not_collide(tmp_path):
    base = tmp_path / "bench-scratch"
    a = rtb.transcript_path(base, base / "20260824-011713", "xeno-skills", 306)
    b = rtb.transcript_path(base, base / "20260824-0999", "xeno-skills", 306)
    assert a != b, ("same task on two runs writes the same file; the second "
                    "would silently overwrite the first's evidence")


def test_transcript_survives_deleting_the_run_root(tmp_path):
    import shutil
    base = tmp_path / "bench-scratch"
    run_root = base / "20260824-011713"
    (run_root / "clones").mkdir(parents=True)
    p = rtb.transcript_path(base, run_root, "xeno-skills", 306)
    # edit_canary's writer, not a second copy of it: real_task_bench streams the
    # subprocess straight into the file and has no writer of its own, so one
    # here would exist only to be tested.
    edit_canary.save_transcript(p, "worker said things")
    shutil.rmtree(run_root)
    assert p.exists(), "the run root took the transcript with it"
    assert p.read_text(encoding="utf-8") == "worker said things"


# --------------------------------------------------------- the dead-log check

def test_a_log_that_did_not_grow_is_an_instrument_fault(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("old content from a previous session\n", encoding="utf-8")
    fault = rtb.log_fault(str(log), since_offset=log.stat().st_size,
                          new_offset=log.stat().st_size, worker_ran_s=617.7)
    assert fault, "617 s of work and zero new bytes is not a normal run"
    assert "server.log" in fault, "the message must name the file to look at"


def test_a_log_that_grew_is_not_a_fault(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("x" * 100, encoding="utf-8")
    assert rtb.log_fault(str(log), since_offset=0, new_offset=100,
                         worker_ran_s=617.7) is None


def test_a_task_too_short_to_expect_output_is_not_a_fault(tmp_path):
    """A worker that died in two seconds may legitimately have logged nothing.
    Flagging that would turn a real, fast failure into an instrument excuse --
    exactly what classify_outcome's docstring warns against."""
    log = tmp_path / "server.log"
    log.write_text("x", encoding="utf-8")
    assert rtb.log_fault(str(log), since_offset=1, new_offset=1,
                         worker_ran_s=2.0) is None


def test_a_missing_log_is_a_fault_naming_the_path(tmp_path):
    missing = tmp_path / "nope" / "server.log"
    fault = rtb.log_fault(str(missing), since_offset=0, new_offset=0,
                          worker_ran_s=617.7)
    assert fault and "nope" in fault


def test_the_fault_makes_the_row_void_not_fail():
    """VOID means 'not measured'. FAIL means 'measured, and the worker lost'.
    Reporting the second when the first is true is how 0-for-5 got published
    against a window the operator had mis-set (classify_outcome docstring)."""
    row = {"outcome": "FAIL", "changed_files": 0}
    rtb.apply_log_fault(row, "server.log did not grow")
    assert row["outcome"] == "VOID"
    assert "did not grow" in row["note"]


def test_applying_no_fault_leaves_the_row_alone():
    row = {"outcome": "FAIL", "changed_files": 0, "note": "the worker changed nothing"}
    rtb.apply_log_fault(row, None)
    assert row["outcome"] == "FAIL"
    assert row["note"] == "the worker changed nothing"


# -------------------------------------------------------------- the wiring
#
# The three tests below read the source. They exist because everything above
# passes on a version of this file where the new functions are defined and never
# called -- which is the shape of the defect being fixed: `run_one` already had a
# comment claiming the transcript was safe from cleanup, and it was not.

SRC = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "real_task_bench.py"), encoding="utf-8").read()


def test_run_one_writes_the_transcript_through_transcript_path():
    assert "out_path = transcript_path(" in SRC, \
        "the transcript path is still built inline; the helper is dead code"
    assert "clone.parent / (" not in SRC, \
        "the old under-the-run-root path is still present"


# The CALL, not the `def` line -- both contain the same text, and matching the
# definition would make the ordering test below compare a function against
# itself and pass for the wrong reason.
CALL = "\n    apply_log_fault(row, fault)"


def test_the_log_fault_is_computed_and_applied():
    assert "fault = log_fault(" in SRC, "log_fault is defined but never called"
    assert CALL in SRC, \
        "the fault is computed and then dropped, which is worse than not " \
        "computing it -- the row looks checked"


def test_the_fault_is_applied_after_the_outcome_is_set():
    """apply_log_fault overrides `outcome`. Called before classify_outcome, the
    override would be silently replaced and the row would ship as FAIL again."""
    i_classify = SRC.index('row["outcome"] = classify_outcome(')
    i_apply = SRC.index(CALL)
    assert i_apply > i_classify, "the VOID override runs before it is overwritten"


def test_a_pass_is_never_voided_by_the_log_check():
    """If the worker edited files and the repo's own verify went green, the task
    demonstrably happened. A log-reading problem cannot un-happen it; it only
    costs us the high-water number."""
    row = {"outcome": "PASS", "changed_files": 3}
    rtb.apply_log_fault(row, "server.log did not grow")
    assert row["outcome"] == "PASS"
    assert "did not grow" in row.get("note", "")
