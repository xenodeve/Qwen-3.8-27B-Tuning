r"""One window, one server, and closing the window ends it.

WHY THIS REPLACES SEVERAL TESTS (issue #49). The launcher grew a mode for every
way a server could outlive the terminal that started it: `-Detach` to make it
happen on purpose, a branch to report on a server this window did not start, a
hunt through old log files for the residency of a process we never watched. Each
was reasonable on its own and together they were machinery for a situation the
developer does not want to exist.

Making close mean close removes the situation rather than handling it. A server
cannot outlive its window, so there is nothing to detach from, nothing to adopt
afterwards, and no stale log to go looking through.

WHAT STAYS AND WHY.

**The port guard.** Two windows can still race, and two orchestrators cannot
share port 8080 -- an armed queue once killed a running corpus and the summary
still printed a plausible number. What changes is the story it tells: a server on
the port now means ANOTHER WINDOW IS OPEN, not that something was left behind.

**The job object.** It is what makes the invariant true rather than hoped for.
Measured 2026-08-25: without it, killing the parent left llama-server running and
answering, because Windows does not propagate a parent's death down the tree.

WHAT THIS FILE CANNOT CHECK. That an interactive window close kills it -- a
headless session has no window handle to deliver the close to. The mechanism is
verified on the strictly harder case, a hard kill that runs no cleanup at all.
"""
import os
import re

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
SERVE = os.path.join(ROOT, "serve.ps1")


def read():
    with open(SERVE, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def test_there_is_no_way_to_outlive_the_window():
    """-Detach was the one supported way to break the invariant. A launcher that
    offers both stories has to explain which one is running, every time."""
    s = read()
    assert "$Detach" not in s, (
        "-Detach is back; then 'closing this window stops the server' is true "
        "only sometimes, and the developer has to know which time it is")


def test_the_server_is_never_started_as_a_separate_process():
    """Start-Process for the profile is exactly the thing that produced an
    orphan. The only Start-Process left should be the elevation prompt."""
    s = read()
    for m in re.finditer(r"Start-Process[^\r\n]*", s):
        line = m.group(0)
        if "$profileScript" in line or "-File" in line:
            raise AssertionError(
                "the profile is launched as a separate process: %r" % line)


def test_the_port_guard_survives_because_two_windows_can_still_race():
    s = read()
    assert re.search(r"Already serving", s), "the port guard is gone"


def test_a_busy_port_now_means_another_window_not_a_leftover():
    """The message has to match the model. Under the old design a server on the
    port might have been abandoned; under this one someone has it open."""
    s = read()
    m = re.search(r"Already serving", s)
    tail = s[m.start():m.start() + 1000]
    assert re.search(r"another window|window that started it|owns it", tail, re.I), (
        "the busy-port message still describes an orphan rather than a peer")


def test_the_status_does_not_hunt_for_a_foreign_log():
    """Reading an old serve-*.log to guess the residency of a process this
    window never watched was only meaningful when servers could be inherited."""
    s = read()
    assert not re.search(r"Get-ChildItem[^\r\n]*serve-\*", s), (
        "the launcher still searches old logs for a server it did not start")


def test_the_job_object_is_what_makes_the_invariant_true():
    s = read()
    assert "AssignProcessToJobObject" in s
    assert re.search(r"0x2000|KILL_ON_JOB_CLOSE", s)
