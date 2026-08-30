r"""Stop forwarding llama.cpp's output and it cannot be spoiled on the way.

WHY (issue #49). Colour kept disappearing, and every fix uncovered another layer
doing it. `--log-colors` defaults to `auto`, which means "colour only when
writing to a terminal" -- and reading the output line by line meant it was not
writing to one. Turning the flag on fixed that and the colour still vanished,
because PowerShell 7 strips the codes at print time whenever the output is not a
console. Each fix was correct and the next layer was still there.

The pattern is the tell: **anything that carries the output can drop something
from it.** So the launcher stops carrying it. llama.cpp writes straight to the
console, exactly as it does when run by hand, and there is no forwarder left to
get this wrong.

HOW THE CHECKS SURVIVE WITHOUT THE PIPE.

`--log-file` writes to BOTH -- `common/log.cpp:170-178` prints the entry to
stdout/stderr and then again to the file. So the same lines are readable from
disk while the console gets them untouched. A small watcher reads the file for
the two markers -- `offloaded N/M layers to GPU` and llama.cpp's own
`listening on http` -- prints the residency verdict and the status, and exits.

WHY A WATCHER PROCESS IS NOT THE THING WE JUST REMOVED. The server used to be
the detached one, which is what let it outlive its window. Here the SERVER is in
the foreground and the watcher is the short-lived sibling; it shares the console
so its output lands in the same window, it joins the same job object so it dies
with everything else, and it exits on its own once it has reported.

WHAT IS NOT ASSERTED. That the colours render. That needs a console and an eye.
What can be checked here is that nothing in this repository stands between
llama.cpp and the terminal.
"""
import os
import re

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
SERVE = os.path.join(ROOT, "serve.ps1")
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q2kxl-mtp.ps1")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def test_the_profile_output_is_not_piped_anywhere():
    """The one property that makes the colour question go away for good."""
    s = read(SERVE)
    m = re.search(r"&\s*\$profileScript[^\r\n]*", s)
    assert m, "the profile is not invoked"
    line = m.group(0)
    assert "|" not in line, (
        "the profile's output is still piped, so something can still drop part "
        "of it: %r" % line)
    assert "2>&1" not in line, (
        "merging stderr routes it through PowerShell instead of the console")


def test_nothing_reformats_the_output_line_by_line():
    s = read(SERVE)
    assert "ForEach-Object {" not in s or "$line" not in s, (
        "a per-line handler is back between llama.cpp and the terminal")


def test_the_profile_can_write_a_log_file():
    """--log-file writes to BOTH the console and the file, which is what makes
    the checks possible without touching the stream."""
    p = read(PROFILE)
    assert "--log-file" in p, "the profile cannot write the log the watcher reads"
    assert re.search(r"\$logFileArg\s*=\s*if", p), (
        "the empty case is not an empty array; an inline $(if ...) would pass "
        "an empty string as a real argument, which only breaks the DEFAULT path")
    assert re.search(r"\$LogFile\s*=\s*''", p), (
        "the profile's log default is not off; a profile run by hand would "
        "start writing files nobody asked for")


def test_the_launcher_asks_for_the_log_rather_than_declaring_the_flag():
    s = read(SERVE)
    code = [ln for ln in s.splitlines()
            if not ln.strip().startswith('#') and '--log-file' in ln]
    assert not code, "serve.ps1 declares the flag: %r" % code
    assert "LogFile" in s, "serve.ps1 never asks for a log file"


def test_the_watcher_reads_the_file_and_not_the_stream():
    s = read(SERVE)
    assert re.search(r"offloaded", s), "nothing checks residency any more"
    assert re.search(r"\[IO\.File\]::Open|Get-Content|Select-String", s), (
        "the watcher does not read the log file")


def test_the_log_path_exists_before_it_is_handed_to_the_profile():
    """Found by running it. $profileArgs was built with LogFile = $log a hundred
    lines BEFORE $log was assigned, so the profile received an empty string, no
    log was written, and the watcher polled a path nothing would ever create.

    It then waited ten minutes and said NOTHING -- the status block simply never
    appeared, with no error anywhere. A silent nothing is the worst shape a
    failure can take here."""
    s = read(SERVE)
    use = s.index("LogFile = $log")
    assign = s.index("$log   = Join-Path")
    assert assign < use, (
        "LogFile is read at char %d and assigned at char %d -- the profile gets "
        "an empty path" % (use, assign))


def test_the_watcher_complains_when_the_log_never_appears():
    """The reason the ordering bug cost a whole run to find. A watcher that
    times out in silence is indistinguishable from one that had nothing to
    report."""
    s = read(SERVE)
    assert re.search(r"never appeared|no log was written|gave up", s), (
        "the watcher can time out without saying why")


def test_the_watcher_shares_the_console():
    """Hidden or redirected, its output would go somewhere the developer is not
    looking -- which is how the status block would quietly stop appearing."""
    s = read(SERVE)
    m = re.search(r"Start-Process[^\r\n]*pwsh[^\r\n]*", s)
    watchers = [ln for ln in s.splitlines()
                if "Start-Process" in ln and "NoNewWindow" in ln]
    assert watchers, (
        "the watcher does not share this console, so its report lands "
        "somewhere the developer cannot see")
