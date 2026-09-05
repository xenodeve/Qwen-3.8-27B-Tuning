r"""The EXL3 server must not stay dead after its TP children die (issue #75).

INCIDENT. 2026-09-04 21:00, boot exl3-serve-20260904-201512: a 130K request
stalled at its first forward pass, `## Synchronization timeout in kernel:
pg_all_reduce_cpu_kernel` x336, then `RuntimeError: CPU reduce process
timeout` in every TP child. From then on /health said {"ok": true, "busy":
false} while every completion answered 500 in 0.4 s, for an hour, until a
human killed the tree. The developer's standing instruction: restart it
yourself. So the server recognises the fork's dead-child signatures, leaves a
flag for the launcher, and exits; serve-exl3.cmd relaunches on the flag.
"""
import os
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING = os.path.dirname(BENCH)
ROOT = os.path.dirname(TUNING)
SERVING = os.path.join(TUNING, "serving", "exl3")
sys.path.insert(0, SERVING)
import watchdog  # noqa: E402


def read(path):
    with open(path, encoding = "utf-8") as fh:
        return fh.read()


# --- recognising the fault, and only the fault --------------------------------

@pytest.mark.parametrize("exc", [
    RuntimeError("CPU reduce process timeout"),                    # model_tp_backend via child
    RuntimeError("Synchronization timeout in kernel: pg_all_reduce_cpu_kernel"),
    TimeoutError("Timed out waiting for worker"),                  # model_tp.py DISPATCH_TIMEOUT
])
def test_the_forks_dead_child_signatures_are_fatal(exc):
    assert watchdog.is_fatal(exc)


@pytest.mark.parametrize("exc", [
    AssertionError("cache: not enough space"),   # the fork's context/cache 400 path
    ValueError("invalid JSON"),
    RuntimeError("CUDA out of memory"),          # a request problem, the process still works
    KeyError("messages"),
])
def test_ordinary_request_errors_are_not_fatal(exc):
    """A watchdog that restarts the server on every error is a different
    outage: one bad request would evict every cached prefix."""
    assert not watchdog.is_fatal(exc)


# --- dying leaves a flag and exits, once --------------------------------------

def test_die_writes_the_flag_with_the_reason_and_exits_with_the_restart_code(tmp_path, monkeypatch):
    flag = tmp_path / "exl3-restart.flag"
    monkeypatch.setattr(watchdog, "FLAG_PATH", str(flag))
    monkeypatch.setattr(watchdog, "_dying", False)
    calls = []
    watchdog.die("CPU reduce process timeout", exit_fn = calls.append, delay_s = 0)
    assert flag.read_text(encoding = "utf-8").strip() == "CPU reduce process timeout"
    assert calls == [watchdog.RESTART_CODE]
    assert watchdog.RESTART_CODE == 3


def test_die_is_idempotent_so_twenty_dead_children_do_not_race(tmp_path, monkeypatch):
    """The 20 child tracebacks on 2026-09-04 arrived as ten queued requests
    were woken one after another; each would call die()."""
    flag = tmp_path / "exl3-restart.flag"
    monkeypatch.setattr(watchdog, "FLAG_PATH", str(flag))
    monkeypatch.setattr(watchdog, "_dying", False)
    calls = []
    watchdog.die("first", exit_fn = calls.append, delay_s = 0)
    watchdog.die("second", exit_fn = calls.append, delay_s = 0)
    assert calls == [watchdog.RESTART_CODE]
    assert flag.read_text(encoding = "utf-8").strip() == "first"


def test_check_dies_only_on_a_fatal_exception(tmp_path, monkeypatch):
    flag = tmp_path / "exl3-restart.flag"
    monkeypatch.setattr(watchdog, "FLAG_PATH", str(flag))
    monkeypatch.setattr(watchdog, "_dying", False)
    calls = []
    watchdog.check(ValueError("no"), exit_fn = calls.append, delay_s = 0)
    assert calls == [] and not flag.exists()
    watchdog.check(RuntimeError("CPU reduce process timeout"), exit_fn = calls.append, delay_s = 0)
    assert calls == [watchdog.RESTART_CODE] and flag.exists()


# --- the hooks are in the server and the loop is in the launcher --------------

def test_both_generation_error_paths_in_the_server_call_the_watchdog():
    server = read(os.path.join(SERVING, "server.py"))
    assert "import live_timing, effort, anthropic_routes, watchdog" in server
    # the stream worker's except and the non-stream except both reach it
    assert server.count("watchdog.check(e)") >= 2


def test_the_recipe_relaunches_on_the_flag_and_stamps_a_new_log():
    """cmd's %ERRORLEVEL% after `python ... | pwsh Tee-Object` is pwsh's, not
    python's, so the exit code cannot carry the signal: the flag file does.
    The stamp must be computed inside the loop or every relaunch appends to
    the dead boot's log."""
    cmd = read(os.path.join(TUNING, "scripts", "serve-exl3.cmd"))
    assert "exl3-restart.flag" in cmd
    lines = cmd.splitlines()
    loop = next(i for i, l in enumerate(lines) if l.strip().lower().startswith(":again"))
    stamp = next(i for i, l in enumerate(lines) if "set STAMP=" in l)
    flag_set = next(i for i, l in enumerate(lines) if "set FLAG=" in l and "exl3-restart.flag" in l)
    check = next(i for i, l in enumerate(lines) if i > loop and 'if exist "%FLAG%"' in l)
    assert flag_set < loop < stamp < check
    assert any("goto again" in l.lower() for l in lines[check:])


# --- "down" is wider than a dead child (developer, 2026-09-05) -----------------
# "หมายถึงให้เขียน server ให้มีการ restart เองเมื่อ server down": any exit that
# was not asked for is relaunched; a process that is alive but no longer
# answers /health is made to exit so the same loop catches it.

def test_the_recipe_relaunches_on_any_exit_unless_the_stop_flag_was_left():
    """A crash with no Python exception (a CUDA driver abort, an access
    violation in the extension) leaves no restart flag and no traceback in
    our code; it must come back too. Stopping on purpose is a flag the
    stopper writes first (stop-exl3.cmd), so the loop can tell the two
    apart. Ctrl+C in the window ends the batch itself."""
    cmd = read(os.path.join(TUNING, "scripts", "serve-exl3.cmd"))
    lines = cmd.splitlines()
    assert any("set STOP=" in l and "exl3-stop.flag" in l for l in lines)
    loop = next(i for i, l in enumerate(lines) if l.strip().lower().startswith(":again"))
    stop_check = next(i for i, l in enumerate(lines) if i > loop and 'if exist "%STOP%"' in l)
    goto = [i for i, l in enumerate(lines) if "goto again" in l.lower()]
    assert goto and stop_check < goto[-1]
    # the relaunch is not inside the restart-flag branch any more: it happens after it
    flag_check = next(i for i, l in enumerate(lines) if i > loop and 'if exist "%FLAG%"' in l)
    assert flag_check < goto[-1]


def test_the_recipe_gives_up_after_three_fast_deaths_so_a_broken_config_does_not_loop_forever():
    """A model directory that does not exist, a cache that does not fit: the
    server exits within seconds, and a loop with no guard would relaunch it
    every 5 s until someone notices. Three exits within 120 s of their
    start stop the loop; one long-lived run resets the count. A pass also ends
    when :8000 already answers -- on 2026-09-05 a loop relaunched into a held port
    ten times, four minutes of model load each, and the guard never saw a fast death."""
    cmd = read(os.path.join(TUNING, "scripts", "serve-exl3.cmd"))
    assert "set /a FAST+=1" in cmd and "set FAST=0" in cmd
    assert "LSS 420" in cmd and "GEQ 3" in cmd   # 420: a model load alone is 2-4 min (2026-09-05 relaunch storm)
    assert "ToUnixTimeSeconds" in cmd
    assert "something already answers on :8000" in cmd


def test_stop_script_leaves_the_stop_flag_and_kills_by_command_line():
    """Stop-Process by name matched the wrong python and returned 255 on the
    server's pids on 2026-09-05; the tree is found by its command line and
    killed with taskkill /T. The flag goes down FIRST so the loop sees it."""
    stop = read(os.path.join(TUNING, "scripts", "stop-exl3.cmd"))
    lines = [l for l in stop.splitlines() if not l.strip().lower().startswith("rem")]
    set_line = next(i for i, l in enumerate(lines) if "set STOP=" in l and "exl3-stop.flag" in l)
    flag_line = next(i for i, l in enumerate(lines) if '> "%STOP%"' in l)
    kill_line = next(i for i, l in enumerate(lines) if "taskkill" in l.lower())
    assert set_line < flag_line < kill_line
    assert "/T" in lines[kill_line] and "server.py" in stop


def test_self_probe_exits_when_health_stops_answering_but_tolerates_one_miss(tmp_path, monkeypatch):
    """Alive-but-deaf: the process is up, the port accepts, /health never
    returns. Nothing raises inside our code, so only a probe sees it. One
    miss is a busy event loop; two in a row is down."""
    calls, answers = [], iter([True, False, True, False, False])
    sleeps = []
    monkeypatch.setattr(watchdog, "_dying", False)
    monkeypatch.setattr(watchdog, "FLAG_PATH", str(tmp_path / "exl3-restart.flag"))
    watchdog.run_self_probe(probe = lambda: next(answers), misses_allowed = 1,
                            sleep_fn = sleeps.append, exit_fn = calls.append, delay_s = 0)
    assert calls == [watchdog.RESTART_CODE]
    assert len(sleeps) == 5          # one sleep per probe, and it stopped at the second miss


def test_self_probe_asks_the_servers_own_health_and_treats_a_timeout_as_a_miss():
    assert watchdog.probe_health(port = 1, timeout_s = 0.2) is False   # nothing listens on :1


def test_the_server_starts_the_self_probe_once_the_app_is_up():
    server = read(os.path.join(SERVING, "server.py"))
    assert "watchdog.start_self_probe(" in server
    assert "on_startup" in server


def test_die_says_so_when_the_flag_cannot_be_written(tmp_path, monkeypatch, capsys):
    """Review 2026-09-06: the flag write was `except Exception: pass`, so a
    relaunch would lose the one diagnostic the loop prints (the reason). The
    exit must still happen, and the failure must be on the console."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    monkeypatch.setattr(watchdog, "FLAG_PATH", str(blocker / "exl3-restart.flag"))
    monkeypatch.setattr(watchdog, "_dying", False)
    codes = []
    watchdog.die("CPU reduce process timeout", exit_fn = codes.append, delay_s = 0)
    assert codes == [watchdog.RESTART_CODE]
    out = capsys.readouterr().out
    assert "flag" in out and "CPU reduce process timeout" in out


def test_the_relaunch_loop_honours_a_stop_asked_for_between_passes():
    """Review 2026-09-06 (spec axis): the stop flag was read only after the
    python pipeline ended. A stop-exl3.cmd issued during the 5 s sleep or the
    :8000 pre-check found no python to kill, left the flag, and the loop
    relaunched anyway. The check must sit at :again, before the launch."""
    with open(os.path.join(TUNING, "scripts", "serve-exl3.cmd"), encoding = "utf-8") as fh:
        script = fh.read()
    head = script[script.index(":again"):script.index("server.py")]
    assert 'if exist "%STOP%"' in head, "no stop-flag check between :again and the launch"
