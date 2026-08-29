r"""Stopping our servers must not stop somebody else's.

THE INCIDENT, 2026-08-29. Clearing VRAM, an agent ran
`Get-Process llama-server | Stop-Process -Force` several times. That matches by
NAME, and this machine runs two different llama-servers:

    C:\AI\llama.cpp-blackwell\llama-server.exe                  ours
    C:\Users\xenod\.unsloth\llama.cpp\build\bin\Release\...     Unsloth Studio's

Studio's was killed on every pass and restarted itself each time, so the symptom
read as "the process will not die" rather than "you are killing the developer's
session". The VRAM never came back because the wrong process was being stopped.

The fix is that the filter is the PATH. This test pins that, because the by-name
form is shorter to type and will be reached for again.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _invocation import live_lines

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(os.path.dirname(BENCH), "scripts", "stop-our-servers.ps1")


def read():
    with open(SCRIPT, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def test_the_script_exists():
    assert os.path.exists(SCRIPT), SCRIPT


def test_it_filters_on_the_executable_path():
    t = read()
    assert "ExecutablePath" in t, "it selects by something other than the path"
    assert "OurRoot" in t, t


def test_it_never_stops_by_name_alone():
    """`Stop-Process -Name` or a bare `Get-Process llama-server | Stop-Process`
    is the exact form that caused the incident."""
    joined = os.linesep.join(live_lines(read()))
    assert not re.search(r"Stop-Process\s+-Name", joined), joined
    assert not re.search(r"Get-Process\s+llama-server[^\n]*Stop-Process", joined), joined


def test_it_says_out_loud_what_it_is_leaving_alone():
    """Silence here would look identical to 'there was nothing else running'."""
    assert "LEAVING ALONE" in read()


def test_it_supports_whatif():
    """A script whose whole job is killing processes should be previewable."""
    assert "SupportsShouldProcess" in read()
    assert "ShouldProcess" in read()
