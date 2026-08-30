"""A task that ran out of context did not fail. It was not given room.

INSTRUMENT FAULT (2026-08-22, issue #18). The first real-task benchmark ran
five open issues against a 32,768-token window and reported:

    5 tasks: 0 PASS, 5 FAIL, 0 VOID
    context high-water: min 32767  median 32767  max 41377

Every baseline was green, so nothing was excluded, and the headline reads as a
verdict on the worker: *nought for five*. It is not one. 32,767 is `n_ctx - 1`,
and the server log carries `exceeds the available context size (32768 tokens)`
six times and `truncated = 1` four times.

The tasks filled the window. Reporting that as FAIL blames the model for a
number the operator chose, and it is the exact shape of mistake this benchmark
exists to avoid: a believable verdict produced by a broken setup.

WINDOW_BOUND is its own outcome. It is not a worker failure and must never be
counted as one -- but unlike VOID it is also a **result**: it says the window is
too small for this class of task, which is the whole of Q2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import classify_outcome


def test_a_saturated_window_is_not_a_failure():
    o = classify_outcome(verify_exit=1, changed_files=0,
                         ctx_high_water=32767, n_ctx=32768)
    assert o == "WINDOW_BOUND"


def test_a_high_water_above_the_window_is_also_window_bound():
    """The API rejects the request, so the peak recorded can exceed n_ctx."""
    assert classify_outcome(1, 0, 41377, 32768) == "WINDOW_BOUND"


def test_a_task_with_room_to_spare_that_changed_nothing_is_a_real_failure():
    """Half the window used and no diff: that is the worker, and it counts."""
    assert classify_outcome(1, 0, 16000, 32768) == "FAIL"


def test_a_green_verify_with_a_diff_is_a_pass():
    assert classify_outcome(0, 3, 16000, 32768) == "PASS"


def test_a_green_verify_with_no_diff_is_a_failure_not_a_pass():
    """Changing nothing passes the tests that were already passing."""
    assert classify_outcome(0, 0, 16000, 32768) == "FAIL"


def test_a_pass_that_also_saturated_the_window_is_still_a_pass():
    """It got there. Running close to the edge is not a disqualification."""
    assert classify_outcome(0, 2, 32767, 32768) == "PASS"


def test_the_margin_is_proportional_not_a_fixed_token_count():
    """A 4,096 window is saturated far below 32,767."""
    assert classify_outcome(1, 0, 4090, 4096) == "WINDOW_BOUND"
    assert classify_outcome(1, 0, 2000, 4096) == "FAIL"


def test_an_unknown_high_water_cannot_prove_saturation():
    """No reading is not evidence of room, and not evidence of the lack of it.

    It stays FAIL rather than being excused, because excusing on missing data
    is how a benchmark stops reporting failures at all.
    """
    assert classify_outcome(1, 0, None, 32768) == "FAIL"
