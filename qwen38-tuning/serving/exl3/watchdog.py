"""Exit the EXL3 server when its tensor-parallel children are dead, so the
launcher can bring it back (issue #75, 2026-09-05).

The fork runs one worker process per card plus a CPU all-reduce process. When
the reduce path stalls past its deadline (`SYNC_TIMEOUT 2 x 45 s` in
exllamav3_ext/parallel/timeout.cuh) every child raises and stays dead, but the
aiohttp parent keeps serving: /health answers ok, every completion answers 500
in under a second, and nothing recovers. Seen 2026-09-04 21:00 for an hour.

"Down" is wider than a dead child (developer, 2026-09-05): the launcher loop
relaunches on ANY exit that was not asked for (stop-exl3.cmd leaves
exl3-stop.flag first), and `start_self_probe` makes an alive-but-deaf process
exit too -- /health polled from a thread, two consecutive misses -> die().

`check(exc)` is called from both generation error paths in server.py. On a
fatal signature it writes FLAG_PATH with the reason and exits with RESTART_CODE
after a short delay (so the in-flight error reply is flushed). serve-exl3.cmd
relaunches when the flag exists. Anything that is not a dead-child signature is
left alone: one bad request must not evict every cached prefix.
"""
import os
import threading
import time
import urllib.request

FLAG_PATH = os.environ.get("EXL3_RESTART_FLAG", r"C:\AI\qwen38-tuning\logs\exl3-restart.flag")
RESTART_CODE = 3
FATAL = (
    "CPU reduce process timeout",        # all_reduce_cpu_*.cpp -> every TP child
    "Synchronization timeout",           # timeout.cuh, printed by the kernel
    "Timed out waiting for worker",      # model_tp.py DISPATCH_TIMEOUT: a child never answered
)
_dying = False
_lock = threading.Lock()


def is_fatal(exc):
    text = str(exc)
    return any(sig in text for sig in FATAL)


def die(reason, exit_fn = os._exit, delay_s = 1.0):
    """Leave the flag and exit, once. Later callers (the queued requests that
    wake one after another on the same dead children) return without effect."""
    global _dying
    with _lock:
        if _dying:
            return
        _dying = True
    try:
        os.makedirs(os.path.dirname(os.path.abspath(FLAG_PATH)), exist_ok = True)
        with open(FLAG_PATH, "w", encoding = "utf-8") as fh:
            fh.write(str(reason).strip() + "\n")
    except Exception as e:
        print(f" ## watchdog: could not write the restart flag {FLAG_PATH}: {e!r}", flush = True)
    print(f" ## watchdog: TP children dead ({reason}); exiting {RESTART_CODE} for relaunch", flush = True)
    if delay_s:
        threading.Timer(delay_s, exit_fn, [RESTART_CODE]).start()
    else:
        exit_fn(RESTART_CODE)


def check(exc, **kw):
    if is_fatal(exc):
        die(str(exc), **kw)


# --- alive but deaf: the process is up and /health never answers -------------

PROBE_INTERVAL_S = 30.0
PROBE_TIMEOUT_S = 20.0


def probe_health(port, timeout_s = PROBE_TIMEOUT_S):
    """True when this process's own /health answers ok within timeout_s.
    /health does not take the generation lock, so a long prefill does not
    count as a miss; only a wedged event loop or a dead listener does."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout = timeout_s) as r:
            return r.status == 200
    except Exception:
        return False


def run_self_probe(probe, misses_allowed = 1, interval_s = PROBE_INTERVAL_S,
                   sleep_fn = time.sleep, **die_kw):
    """Loop until the probe misses misses_allowed + 1 times in a row, then
    die(). One miss is tolerated: an event loop busy with a 200 MB request
    body can hold /health past the timeout once."""
    misses = 0
    while True:
        sleep_fn(interval_s)
        if probe():
            misses = 0
            continue
        misses += 1
        if misses > misses_allowed:
            die(f"self-probe: /health unanswered {misses} times", **die_kw)
            return


def start_self_probe(port, **kw):
    t = threading.Thread(target = run_self_probe, args = (lambda: probe_health(port),),
                         kwargs = kw, daemon = True, name = "exl3-self-probe")
    t.start()
    return t
