r"""Two icons, and which one you click is the choice.

WHY (issue #49). The developer wants to start the worker by double-clicking,
without a terminal or a remembered flag.

TWO THINGS BREAK A DOUBLE-CLICKED .bat, and both are silent.

**The window vanishes.** `serve.ps1` detaches the server and returns, so a .bat
that just calls it flashes and closes before anything can be read -- including
the failure it was reporting. The launchers pass `-Follow`, so the window stays
and shows what llama.cpp is printing, and they `pause` on a non-zero exit so an
error is readable instead of gone.

**The working directory is not the folder.** Double-clicking from a shortcut, or
from a shell opened elsewhere, does not put `%CD%` at the .bat's location. A
relative path to `serve.ps1` then resolves against the wrong folder and the file
"does not exist". `%~dp0` is the .bat's own directory and is the only reliable
anchor.

WHY TWO FILES AND NOT ONE WITH A PROMPT.

`serve.ps1 -Lan` binds every interface, and `--host` is the only access control
this server has -- no API key, CORS `*`. A single icon that exposes it means the
exposure happens because someone wanted the model running, not because they
chose to be reachable. Two icons keep the choice where it was: `serve.bat` is
loopback, `serve-lan.bat` says what it does in its name, and clicking it is the
same act as typing the flag.

Neither carries serving configuration, for the reason `serve.ps1` does not: the
flags live in the profile, and a copy in a third file is a third thing to drift.
"""
import os

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
PLAIN = os.path.join(ROOT, "serve.bat")
LAN = os.path.join(ROOT, "serve-lan.bat")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


@pytest.mark.parametrize("path", [PLAIN, LAN])
def test_the_launcher_exists_at_the_root(path):
    assert os.path.isfile(path), path


@pytest.mark.parametrize("path", [PLAIN, LAN])
def test_it_anchors_on_its_own_folder(path):
    """%CD% is not the .bat's folder when it is opened from a shortcut or from
    a shell that started elsewhere."""
    assert "%~dp0" in read(path), (
        "%s resolves serve.ps1 relative to the working directory" % path)


@pytest.mark.parametrize("path", [PLAIN, LAN])
def test_it_keeps_the_window_open(path):
    """Without -Follow the server detaches, the script returns, and the console
    closes before anything can be read."""
    t = read(path)
    assert "-Follow" in t, "%s flashes and closes" % path
    assert "pause" in t.lower(), (
        "%s discards its own error message when it fails" % path)


@pytest.mark.parametrize("path", [PLAIN, LAN])
def test_it_survives_a_restricted_execution_policy(path):
    assert "ExecutionPolicy Bypass" in read(path), (
        "%s fails on a default machine" % path)


@pytest.mark.parametrize("path", [PLAIN, LAN])
def test_it_carries_no_serving_configuration(path):
    """Third file, same rule: the configuration lives in the profile."""
    t = read(path)
    for flag in ("--spec-type", "-ctk", "-ctv", "--reasoning-effort", ".gguf",
                 "-ngl", "--host"):
        assert flag not in t, "%s declares %r" % (path, flag)


def test_the_plain_launcher_does_not_expose():
    """The whole point of two files. Clicking the one that does not say lan
    must not bind every interface."""
    assert "-Lan" not in read(PLAIN), (
        "serve.bat exposes the server; then the second file is pointless and "
        "the exposure happens to anyone who wanted the model running")


def test_the_lan_launcher_says_so_in_its_name_and_does_it():
    assert "-Lan" in read(LAN)
    assert os.path.basename(LAN) == "serve-lan.bat"


@pytest.mark.parametrize("path", [PLAIN, LAN])
def test_it_is_readable_by_cmd(path):
    """A UTF-8 BOM makes cmd.exe choke on the first line -- it reads the BOM as
    part of the command and reports a garbled name. Plain ASCII, no BOM."""
    with open(path, "rb") as fh:
        raw = fh.read()
    assert not raw.startswith(b"\xef\xbb\xbf"), "%s starts with a BOM" % path
    raw.decode("ascii")  # raises if anything non-ASCII slipped in
