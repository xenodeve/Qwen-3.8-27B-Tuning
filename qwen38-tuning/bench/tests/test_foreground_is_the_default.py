r"""The server runs IN the terminal, not beside it.

WHAT WAS WRONG (issue #49). The launcher started the profile with
`Start-Process -WindowStyle Hidden`, waited for health, printed a status block,
and then optionally tailed the log file. So "live terminal" was a **tail of a
file written by a different process** -- two processes, and the window was
watching rather than running. The developer asked for one script and one process,
which is what a foreground server actually is.

WHAT CHANGES, and it is not only tidiness:

  * `Ctrl+C` now stops the SERVER, because the server is this process's child and
    the terminal owns it. Under the old design Ctrl+C stopped a tail and left the
    model running, which needed a printed warning to avoid being mistaken. The
    new behaviour is the one people already expect, so the warning goes.
  * Closing the window stops the server. Same reason. Said plainly rather than
    left to be discovered.
  * The residency check cannot happen "after boot", because the call never
    returns while the server runs. It happens INLINE: the pipeline watches for
    `offloaded N/M layers to GPU` as it streams past and prints the verdict
    there. `--fit` spills instead of refusing, so this is the one line that
    separates the configuration we asked for from a plausible impostor.

WHAT IS KEPT. The port guard, the firewall handling, the bind-is-opt-in rule and
the delegation rule are unchanged -- this is about who owns the process, not
about what gets served or to whom.

`-Detach` keeps the old behaviour for the case where the terminal is needed for
something else. It is not the default, because the default should be the thing
that does what it looks like it does.
"""
import os
import re

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
SERVE = os.path.join(ROOT, "serve.ps1")


def read():
    with open(SERVE, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def test_the_profile_is_invoked_in_this_process():
    """The call operator, not Start-Process. One script, one process -- that is
    the whole request."""
    s = read()
    assert re.search(r"&\s*\$profileScript", s), (
        "the profile is not invoked in-process")


def test_the_foreground_path_does_not_detach():
    """Start-Process may survive only for -Detach, which is opt-in."""
    s = read()
    for m in re.finditer(r"Start-Process[^\r\n]*\$profileScript[^\r\n]*", s):
        window = s[max(0, m.start() - 400):m.start()]
        assert "$Detach" in window, (
            "a detached launch is not guarded by -Detach: %r" % m.group(0))


def test_detach_exists_but_is_not_the_default():
    s = read()
    assert re.search(r"\[switch\]\s*\$Detach", s), "no -Detach escape hatch"
    assert not re.search(r"\[switch\]\s*\$Detach\s*=\s*\$true", s)


def test_the_profile_is_splatted_by_name_not_by_position():
    """Found by running it, not by reading it.

    `& $profileScript @arr` where $arr is an ARRAY splats POSITIONALLY. The
    profile's first parameter is $Ctx, so the run died with

        Cannot convert value "-Verbosity" to type "System.Int32"

    -- the flag name itself arriving as the context size. Named splatting needs
    a HASHTABLE. The failure is loud, which is the only reason it was cheap; the
    same mistake with a parameter that happens to accept a string would have
    started a server configured by accident.
    """
    s = read()
    m = re.search(r"&\s*\$profileScript\s+@(\w+)", s)
    assert m, "the profile is not invoked with splatting"
    name = m.group(1)
    assert re.search(r"\$" + name + r"\s*=\s*@\{", s), (
        "$%s is not a hashtable, so @%s splats positionally" % (name, name))


def test_residency_is_checked_inline_as_the_stream_passes():
    """It cannot be checked after boot -- the call does not return while the
    server runs. --fit spills rather than refusing, so this line is the only
    thing separating the configuration we asked for from an impostor."""
    s = read()
    assert re.search(r"offloaded \(?\\d\+\)?/", s) or "offloaded" in s, (
        "nothing watches for the layer-assignment line")
    m = re.search(r"ForEach-Object", s)
    assert m, "the output is not piped through a per-line handler"


def test_the_status_is_printed_when_the_server_reports_listening():
    """`srv llama_server: listening on http://...` is llama.cpp saying it is
    ready. Printing the status before that would describe a server that is not
    yet answering."""
    s = read()
    assert "listening on" in s, (
        "the launcher does not wait for llama.cpp's own readiness line")


def test_it_says_ctrl_c_stops_the_server_now():
    """The opposite of what it used to say, and saying the old thing would be
    worse than saying nothing."""
    s = read()
    printed = [ln for ln in s.splitlines()
               if "Ctrl+C" in ln and "Write-Host" in ln]
    assert printed, "Ctrl+C is never explained to the developer"
    assert any(re.search(r"stops the server|shuts.*down|closing this window",
                         ln, re.I) for ln in printed), (
        "Ctrl+C is printed without saying it now stops the server: %r" % printed)


def test_no_stale_promise_that_the_server_survives_the_terminal():
    """The old text said the server keeps running when the window closes. In
    the foreground design that is false, and a leftover reassurance is worse
    than none -- it is a believable sentence about the wrong world."""
    s = read()
    for ln in s.splitlines():
        if "Write-Host" not in ln:
            continue
        assert not re.search(r"keeps running in the background", ln, re.I), (
            "stale claim left over from the detached design: %r" % ln.strip())
