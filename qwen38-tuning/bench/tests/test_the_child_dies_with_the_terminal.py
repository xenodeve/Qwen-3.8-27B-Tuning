r"""The server must die with the terminal, and Windows will not do it for us.

MEASURED 2026-08-25, and it refuted what the launcher was printing.

`serve.ps1` told the developer "Ctrl+C stops the server; so does closing this
window." The second half was **false**. The process chain is

    cmd.exe -> pwsh.exe -> llama-server.exe

and killing the top `cmd.exe` left both descendants running and the server still
answering `/props`. Windows does not propagate a parent's death down the tree;
there is no POSIX process group doing it for us. Whatever the launcher printed,
the behaviour was: the model keeps the GPU.

That is the exact failure this repository is organised against -- a believable
sentence over an unchecked condition -- and it was in the one place a developer
would rely on it without testing.

WHY A JOB OBJECT AND NOT try/finally.

A `finally` block runs on a normal exit and on Ctrl+C. It does NOT run when the
process is killed outright, which is what closing a console window can amount to.
A Win32 **job object** with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` is enforced by
the kernel: when the last handle to the job closes -- including because the
process holding it died however it died -- every process in the job is
terminated. It is the only mechanism that holds when the script itself gets no
chance to run code.

WHAT IS NOT ASSERTED HERE. That closing an interactive console window kills it.
That needs a real window and an interactive session; from a headless test session
`MainWindowHandle` is 0 and the close cannot be delivered. What IS verified, in
the session that shipped this, is the strictly harder case: hard-killing the
owning `pwsh` with `Stop-Process -Force`, which runs no cleanup code at all.
"""
import os
import re

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
SERVE = os.path.join(ROOT, "serve.ps1")


def read():
    with open(SERVE, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def test_a_job_object_owns_the_child():
    s = read()
    assert "AssignProcessToJobObject" in s, (
        "nothing puts the server into a job object, so it outlives the terminal")
    assert "CreateJobObject" in s


def test_the_job_kills_on_close():
    """0x2000 is JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. Without it the job is a
    bookkeeping container and the child survives exactly as before."""
    s = read()
    assert re.search(r"0x2000|KILL_ON_JOB_CLOSE", s), (
        "the job object does not carry the kill-on-close limit")


def test_it_finds_the_server_process_rather_than_assuming_one():
    """serve.ps1 does not launch llama-server -- the profile does. The job can
    only own a process the launcher has actually located."""
    s = read()
    assert re.search(r"llama-server", s), (
        "nothing looks for the process the profile started")


def test_the_claim_matches_the_mechanism():
    """The sentence and the guarantee must ship together. If the job object is
    ever removed, this fails next to it rather than leaving a true-sounding
    line behind."""
    s = read()
    printed = [ln for ln in s.splitlines()
               if "Write-Host" in ln and re.search(r"closing this window", ln, re.I)]
    assert printed, "the launcher no longer tells the developer what closes it"
    assert "AssignProcessToJobObject" in s, (
        "the launcher claims the window closes the server without the mechanism "
        "that makes it true")


def test_it_says_what_was_not_verified():
    """From a headless session the interactive window-close cannot be
    delivered. Claiming it was tested would be the same fault one level up."""
    s = read()
    assert re.search(r"hard-kill|Stop-Process -Force|kill-on-close", s), (
        "the file does not record which case was actually exercised")
