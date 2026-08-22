"""The edit canary must not be able to report a success it did not observe.

WHY IT EXISTS. Five real GitHub issues ran 1,427-2,400 s each and changed no
files; three exited rc=0. Before blaming the model, the tool path has to be
cleared: can this worker EDIT AN EXISTING TRACKED FILE in a fresh clone at all?

Past evidence says the broad version of that question is already answered no:
`results/opencode-corpus.jsonl` shows `wrote_target: True` on 7 of 11 rows with
real filenames. But those tasks CREATED A NEW FILE in an empty scratch
directory. A real task must `read -> edit -> save` a file that already exists in
a git checkout, which is a different tool and a different trust situation. The
canary tests that narrower path and nothing else.

WHAT THIS TEST GUARDS. A canary is a detector, and a detector that reports a
clean result is indistinguishable from one that stopped working. So:

  - `EDITED` may be returned ONLY when a real diff was observed. Not when the
    worker says it edited, not when the tool call appears in the transcript.
    The worker claiming success is exactly the evidence that cannot be trusted
    here, because the five failing tasks also exited 0.
  - The transcript must be written OUTSIDE the clone. The last run's transcript
    died with the scratch root, which is why the five failures have no
    mechanism attached; a canary that repeats that failure teaches nothing.
  - A file that git does not track cannot be the target, or the canary silently
    degrades into the create-a-new-file case that is already known to work.

The four outcomes are deliberately distinguishable, because each sends the
investigation somewhere different: an edit that happened, a tool refused, a tool
never called, and an edit claimed with no diff to show for it.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edit_canary


# ---- the rule that makes the detector worth running ------------------------

def test_edited_requires_a_real_diff_not_a_claim():
    """The worker exiting 0 and saying it edited is not evidence. The diff is."""
    assert edit_canary.classify(rc=0, diff_bytes=0,
                                transcript="I have edited the file successfully.") != "EDITED"


def test_edited_is_returned_when_a_diff_is_present():
    assert edit_canary.classify(rc=0, diff_bytes=118, transcript="edit ok") == "EDITED"


def test_a_diff_counts_even_when_the_worker_exited_nonzero():
    """The artifact outranks the exit code in both directions."""
    assert edit_canary.classify(rc=1, diff_bytes=42, transcript="") == "EDITED"


# ---- the three no-diff outcomes must stay distinguishable ------------------

def test_a_permission_refusal_is_its_own_outcome():
    t = "tool edit: permission denied for write in this directory"
    assert edit_canary.classify(rc=0, diff_bytes=0, transcript=t) == "TOOL_DENIED"


def test_never_calling_the_edit_tool_is_its_own_outcome():
    t = "I read the file and concluded no change was necessary."
    assert edit_canary.classify(rc=0, diff_bytes=0, transcript=t) == "NO_EDIT_ATTEMPTED"


def test_an_edit_that_left_no_diff_is_its_own_outcome():
    t = "Called edit on README.md, replacing one line."
    assert edit_canary.classify(rc=0, diff_bytes=0, transcript=t) == "EDIT_NO_DIFF"


def test_the_four_outcomes_are_distinct():
    assert len(set(edit_canary.OUTCOMES)) == 4


# ---- the transcript must outlive the clone ---------------------------------

def test_transcript_path_is_outside_the_clone(tmp_path):
    """It died with the scratch root last time. That is why nothing is known."""
    scratch = tmp_path / "scratch"
    clone = scratch / "clones" / "repo-1"
    p = edit_canary.transcript_path(clone, scratch)
    assert clone not in p.parents and p != clone
    assert str(scratch) in str(p)


def test_transcript_is_written_even_when_the_canary_fails(tmp_path):
    """A failed canary is the case whose evidence matters most."""
    scratch = tmp_path / "scratch"
    clone = scratch / "clones" / "repo-1"
    p = edit_canary.transcript_path(clone, scratch)
    p.parent.mkdir(parents=True, exist_ok=True)
    edit_canary.save_transcript(p, "the worker said no")
    assert p.read_text(encoding="utf-8") == "the worker said no"


# ---- the target must be a file git already tracks --------------------------

def test_an_untracked_target_is_refused(tmp_path):
    with pytest.raises(ValueError) as e:
        edit_canary.assert_tracked(tmp_path, "not-a-real-file.txt", tracked=[])
    assert "not-a-real-file.txt" in str(e.value)


def test_a_tracked_target_is_accepted(tmp_path):
    edit_canary.assert_tracked(tmp_path, "README.md", tracked=["README.md", "LICENSE"])
