r"""A server that is already up must report as much as one that just booted.

WHAT WENT WRONG (issue #49). Running `.\serve.ps1` against a healthy server
printed two lines --

    Already serving on port 8080 -- alias 'qwen38', Q2_K - Medium, build ...
    Restarting a healthy server is not an improvement. Nothing to do.

-- and returned to the prompt. No context window, no layer split, no VRAM, no
URLs, and no word that the thing keeps running in the background. The developer
asked, reasonably, whether it reports the model's state at all.

It was reporting less on the path taken MOST OFTEN. A fresh boot happens once;
"is it up, and how is it doing" is asked all day, and that was the branch with
nothing in it.

WHAT THE STATUS MUST READ RATHER THAN ASSUME.

The already-serving branch did not launch the process, so it knows nothing about
how it was started -- not the bind, not the flags. Everything it prints has to
come from the live server or the operating system:

  * the bind, from the listening socket, NOT from whether -Lan was passed on
    this invocation. A server started earlier with -Lan is reachable now; a
    -Lan passed now to an already-running loopback server changes nothing. The
    socket is the only honest source.
  * context, alias, build, from /props.
  * residency, from the newest boot log, and absent if no log survives -- the
    launcher may be looking at a server it did not start.

WHAT IS NOT ASSERTED. That the numbers are right. These are shape checks on a
PowerShell script; the values are verified by running it against a live server.
"""
import os
import re

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
SERVE = os.path.join(ROOT, "serve.ps1")


def read():
    with open(SERVE, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def test_there_is_one_status_report_not_two():
    """Two copies drift, and the one on the rarely-taken path is the one that
    goes stale unnoticed."""
    s = read()
    assert re.search(r"function\s+Show-ServerStatus", s), (
        "the status block is not a single function both paths call")
    assert len(re.findall(r"Show-ServerStatus", s)) >= 3, (
        "the status function is defined but not called from both paths")


def test_the_already_serving_path_reports_status_before_exiting():
    s = read()
    m = re.search(r"Already serving", s)
    assert m, "the already-serving branch is gone"
    tail = s[m.start():m.start() + 900]
    assert "Show-ServerStatus" in tail, (
        "the already-serving branch still exits without reporting anything")


def test_the_bind_is_read_from_the_socket_not_from_the_switch():
    """This branch did not start the process. Reporting the bind from -Lan
    would describe this invocation rather than the running server."""
    s = read()
    m = re.search(r"function\s+Show-ServerStatus", s)
    body = s[m.start():]
    assert "Get-NetTCPConnection" in body, (
        "the status does not read the listening socket")


def test_the_status_reports_the_window_and_the_memory():
    s = read()
    m = re.search(r"function\s+Show-ServerStatus", s)
    body = s[m.start():]
    assert "n_ctx" in body, "no context window in the status"
    assert "nvidia-smi" in body, "no VRAM in the status"


def test_each_path_says_who_owns_the_process():
    """Rewritten when the launcher moved to the foreground. It used to promise
    the server keeps running when the window closes; in the foreground design
    that is FALSE, and a leftover reassurance is worse than none -- a believable
    sentence about the wrong world.

    The two paths now own the process differently and each says which:
      * foreground -- Ctrl+C stops the server, so does closing the window
      * already-serving -- this window did not start it, so Ctrl+C will not
        reach it, and here is how to stop it
    """
    s = read()

    fg = [ln for ln in s.splitlines()
          if "Write-Host" in ln and re.search(r"Ctrl\+C stops the server", ln)]
    assert fg, "the foreground path never says Ctrl+C stops the server"

    m = re.search(r"Already serving", s)
    tail = s[m.start():m.start() + 900]
    # Updated when -Detach went. A server on the port can no longer be a
    # leftover -- it belongs to a window that is still open -- so the useful
    # instruction is "close that window", not "Stop-Process".
    assert re.search(r"Close that window", tail), (
        "the already-serving path does not say how to stop a peer's server")
    assert re.search(r"reaches nothing|will not reach it", tail), (
        "the already-serving path does not say this terminal does not own it")
