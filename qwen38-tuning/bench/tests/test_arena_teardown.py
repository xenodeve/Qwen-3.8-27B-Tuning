"""The VRAM settle wait must actually run between arms.

THE REGRESSION (2026-08-22, issue #18). The guard was split in two: `run_arm`
killed the server in its `finally`, and `start()` then called `kill()` again to
decide whether to wait. The second call found nothing to kill, returned False,
and the wait was skipped for every arm of the run.

The guard was present, imported, called -- and inert. Nothing failed. The run
proceeded and produced plausible tok/s numbers with no settle wait at all,
which is instrument fault 7 restored after being fixed.

A test that only checked "wait_for_vram_release exists" would have passed
throughout. These check that a teardown followed by a startup actually waits.

NOTHING HERE TOUCHES THE MACHINE, and that is a rule this file learned the
hard way. A test in an earlier version called arena.kill() for real to check
it reports "nothing running" as False. A measurement was in flight, so it
killed the server being benchmarked and the run died mid-round with
WinError 10054. Its docstring claimed the call was safe because it "cannot
stop a server that is not there" -- an assumption about machine state that a
unit test does not get to make. Every test below goes through FakeGpu.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dflash2_arena as arena


class FakeGpu:
    """A GPU that holds memory until it is killed, then releases it."""

    def __init__(self):
        self.resident = True
        self.waits = []
        self.kills = 0

    def vram(self):
        return [0, 2400 if self.resident else 9900]

    def kill(self):
        self.kills += 1
        if not self.resident:
            return False            # taskkill exit 128: nothing to kill
        self.resident = False
        return True

    def wait(self, floor_mib=None, **kw):
        self.waits.append(floor_mib)
        return [9900, 9900]


@pytest.fixture
def gpu(monkeypatch):
    g = FakeGpu()
    monkeypatch.setattr(arena, "vram", g.vram)
    monkeypatch.setattr(arena, "kill", g.kill)
    monkeypatch.setattr(arena, "wait_for_vram_release", g.wait)
    return g


def test_stopping_a_running_server_waits(gpu):
    arena.stop_server()
    assert gpu.kills == 1
    assert len(gpu.waits) == 1, "killed a resident model and did not wait"


def test_the_floor_demands_a_real_release(gpu):
    """Read while resident, plus the minimum rise -- not the post-kill reading.

    A floor taken after the release has already happened is a floor every
    reading clears, which is the same as passing None.
    """
    arena.stop_server()
    assert gpu.waits == [2400 + arena.VRAM_MIN_RISE_MIB]


def test_stopping_twice_does_not_wait_for_nothing(gpu):
    """The second call has nothing to release; waiting would time out."""
    arena.stop_server()
    arena.stop_server()
    assert gpu.kills == 2
    assert len(gpu.waits) == 1


def test_teardown_then_startup_still_waits_once(gpu):
    """The exact shape that broke: run_arm tears down, then start() sets up.

    Whichever of the two performs the kill must be the one that waits. Before
    the fix the teardown killed and the setup decided -- so neither waited.
    """
    arena.stop_server()          # run_arm's finally
    arena.stop_server()          # start()'s setup for the next arm
    assert len(gpu.waits) == 1, (
        "a full teardown/startup cycle waited %d times; the guard is inert"
        % len(gpu.waits)
    )
