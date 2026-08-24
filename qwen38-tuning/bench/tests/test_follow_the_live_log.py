r"""Watch what llama.cpp is printing, without the watching being mistaken for the
server.

WHY (issue #49). The launcher detaches the server and returns to the prompt, so
everything llama.cpp writes while it runs -- slot lifecycle, prompt and eval
timings, speculation counters, warnings -- goes to a file nobody is looking at.
The developer asked to see it live.

THE FAILURE THIS MUST NOT CREATE.

Detaching already read as "it exited" once, which is what prompted the previous
fix. A follow mode makes that worse in the other direction: the terminal is now
full of server output, so **Ctrl+C looks like it stops the server**. It does not
-- it stops the tail. If the launcher does not say so plainly, the first instinct
on seeing scrolling logs is to press Ctrl+C and assume the model is down, and the
next command will be a redundant restart of a healthy server.

WHAT IT MUST READ RATHER THAN ASSUME.

On the already-serving path the launcher did not start the process. There may be
no log for it at all -- the server may predate this script, or have been started
by hand. Following a file that does not exist must say so, not hang on an empty
path or silently return.

WHAT IS NOT ASSERTED. That the tail works. Following a growing file cannot be
exercised from pytest without a live server; these are shape checks, and the
behaviour is verified by running it.
"""
import os
import re

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
SERVE = os.path.join(ROOT, "serve.ps1")


def read():
    with open(SERVE, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def test_the_switch_exists():
    assert re.search(r"\[switch\]\s*\$Follow", read()), (
        "serve.ps1 offers no way to watch the live log")


def test_it_tails_rather_than_dumping_once():
    """-Wait is what makes it live. Without it this is `cat` with extra steps."""
    s = read()
    assert re.search(r"Get-Content[^\r\n]*-Wait", s), (
        "nothing follows the log as it grows")


def test_it_says_ctrl_c_stops_the_watching_and_not_the_server():
    """The whole risk of this feature. A terminal full of server output invites
    Ctrl+C, and the previous confusion was already that a detached server reads
    as a stopped one."""
    s = read()
    assert re.search(r"Ctrl\+C", s), "Ctrl+C is never mentioned"
    # It must be SAID to the developer, not merely commented for a reader of
    # the source. The first version of this test matched the first Ctrl+C
    # anywhere in the file, which was a comment whose sentence wrapped -- so it
    # went red over a line break while the behaviour was right.
    printed = [ln for ln in s.splitlines()
               if "Ctrl+C" in ln and "Write-Host" in ln]
    assert printed, "Ctrl+C is only in a comment; the developer never sees it"
    assert any(re.search(r"server keeps|not the server|only.*(watching|tail)",
                         ln, re.I) for ln in printed), (
        "Ctrl+C is printed without saying it leaves the server running: %r"
        % printed)


def test_following_is_available_on_the_already_serving_path_too():
    """That is the path where you most want to look at a server you did not
    just start."""
    s = read()
    m = re.search(r"Already serving", s)
    tail = s[m.start():m.start() + 1200]
    assert "Follow" in tail, (
        "the already-serving branch cannot follow the log")


def test_a_missing_log_is_reported_not_silently_skipped():
    """The server may predate this script or have been started by hand. A
    follow with no file must say why nothing is happening."""
    s = read()
    m = re.search(r"\$Follow", s)
    body = s[m.start():]
    assert re.search(r"no log|No log|cannot follow|not found", body), (
        "following a server with no log says nothing about why")


def test_the_status_advertises_it():
    """A switch nobody knows about is a switch nobody uses."""
    s = read()
    m = re.search(r"function\s+Show-ServerStatus", s)
    body = s[m.start():]
    assert "-Follow" in body, (
        "the status block does not mention how to watch the live log")
