"""One arena at a time. The port is the resource, and it is not shareable.

THE INCIDENT (2026-08-22, issue #18). A second arena was launched while the
first was still finishing its last round. For a while both drove port 8080, and
the older one's teardown killed the younger one's server. The younger run's log
ends mid-load with no error at all -- 65/65 layers offloaded, then nothing --
because the process did not fail, it was killed.

That run failed loudly only because the kill landed during a load. Had it
landed between generations, the arm would have finished with fewer samples and
a plausible tok/s, and nothing in the output would have said why.

CLAUDE.md names this exact failure: "Two orchestrators cannot share port 8080.
An armed queue once killed a running corpus and the summary still printed a
plausible number." `scripts/swap-model.sh` takes a lock for it. The arena did
not, so it inherited the incident instead of the fix.

These tests pin the refusal, not the mechanism -- a later version may take a
real file lock instead of probing the port, and should still pass.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dflash2_arena as arena


def test_it_refuses_to_start_when_the_port_is_taken(monkeypatch):
    monkeypatch.setattr(arena, "port_owner", lambda: 4242)
    with pytest.raises(RuntimeError, match="8080"):
        arena.require_exclusive_port()


def test_the_refusal_names_the_process(monkeypatch):
    """A message saying only 'port busy' sends the reader hunting.

    The whole point is that another orchestrator is running; say which.
    """
    monkeypatch.setattr(arena, "port_owner", lambda: 4242)
    with pytest.raises(RuntimeError, match="4242"):
        arena.require_exclusive_port()


def test_a_free_port_is_not_an_error(monkeypatch):
    monkeypatch.setattr(arena, "port_owner", lambda: None)
    arena.require_exclusive_port()          # must not raise


def test_the_check_runs_before_any_measurement(monkeypatch):
    """Refuse up front, not after the first arm has already been disturbed.

    Ordering is the whole value: a check that fires after run_arm has started
    reports a collision the run has already suffered.
    """
    import inspect
    src = inspect.getsource(arena.main)
    assert "require_exclusive_port()" in src, "main() does not check at all"
    assert src.index("require_exclusive_port()") < src.index("run_arm("), (
        "the exclusivity check runs after measurement has begun"
    )


def test_port_owner_reports_none_when_nothing_listens():
    """Executed for real: it only reads, and reading a free port is harmless.

    This must not be mocked -- the point is that the probe works on this
    machine. It cannot disturb a running measurement because it starts no
    process and stops none.
    """
    owner = arena.port_owner()
    assert owner is None or isinstance(owner, int)
