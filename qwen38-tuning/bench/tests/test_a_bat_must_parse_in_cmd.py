r"""Every .bat must be something cmd.exe can actually parse and walk.

THE INCIDENT, 2026-08-29. Four launchers shipped with

    echo PowerShell 7 ^^(pwsh^^) was not found, and this needs it.

inside an `if errorlevel 1 ( ... )` block. In cmd, `^^` is an escaped caret, so
the `(` that follows is UNESCAPED and opens a block cmd cannot close. cmd
parses the whole block before running any of it, so the file died on

    was was unexpected at this time.

before reaching a single command. All four were completely non-functional and
EVERY TEST PASSED -- because every test either read the file as text or invoked
serve.ps1 directly, and neither of those is cmd.

This is the .bat -> serve.ps1 layer, which is where the 0.38 tok/s incident
lived too: the profile was correct and the launcher was not. Assertions about
that layer that never run cmd are assertions about a file, not a launcher.

HOW THIS TESTS IT WITHOUT SERVING A MODEL. The launcher is copied to a temp
directory with two substitutions: the line that starts the server becomes an
echo, and `pause` becomes an echo so a failure path cannot block. Everything
else -- the parentheses, the carets, the `where pwsh` probe, the errorlevel
branches -- is the real file, run by the real cmd. Reaching the marker proves
cmd parsed the file AND walked it to the launch.
"""
import glob
import os
import subprocess
import tempfile

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
BATS = sorted(glob.glob(os.path.join(ROOT, "*.bat")))
MARKER = "LAUNCH_REACHED"


def test_there_are_launchers_to_check():
    """A glob that silently matches nothing is a green suite that tests air."""
    assert len(BATS) >= 8, BATS


def _neutered(path):
    """The real file with only the launch and the blocking pause replaced."""
    out = []
    for line in open(path, encoding="ascii").read().splitlines():
        low = line.strip().lower()
        if "serve.ps1" in line and not low.startswith("rem"):
            out.append("echo " + MARKER)
        elif low.startswith("call "):
            # A launcher that delegates to another launcher -- the hub. Its
            # target is built from a variable, so it names no .bat literally
            # and a rule keyed on the filename would miss it.
            out.append("echo " + MARKER)
        elif low.startswith("choice "):
            # choice reports its key POSITION through ERRORLEVEL; exit 1 is
            # the first key, which every prompt here orders as the safe one.
            out.append("cmd /c exit 1")
        elif low.startswith("set /p"):
            # An interactive prompt would wait for a human who is not here.
            out.append("set SEL=1")
        elif low == "pause":
            out.append("echo PAUSE_HIT")
        else:
            out.append(line)
    return "\r\n".join(out) + "\r\n"


@pytest.mark.parametrize("path", BATS, ids=[os.path.basename(b) for b in BATS])
def test_cmd_can_parse_it_and_reach_the_launch(path):
    with tempfile.TemporaryDirectory() as d:
        # Stub every sibling: a launcher that checks its target exists before
        # calling it should have that check PASS here, not be edited out.
        for sibling in BATS:
            open(os.path.join(d, os.path.basename(sibling)), "w").close()
        copy = os.path.join(d, os.path.basename(path))
        with open(copy, "w", encoding="ascii", newline="") as fh:
            fh.write(_neutered(path))
        r = subprocess.run(["cmd.exe", "/c", copy], capture_output=True,
                           text=True, timeout=60)
    both = r.stdout + r.stderr
    assert "unexpected at this time" not in both, (
        "%s does not PARSE in cmd: %s" % (os.path.basename(path), both.strip()))
    assert MARKER in both, (
        "%s parses but never reaches its launch line: %s"
        % (os.path.basename(path), both.strip()))
