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

FOUR ICONS, AND THE SECOND AXIS IS WHICH HARDWARE.

`serve-dual.bat` and `serve-dual-lan.bat` were added for issue #52: a second
GPU makes `UD-Q4_K_XL` servable, and it needs `-Dual`. The axes are
independent -- one card or two, loopback or exposed -- so they are four files
rather than a switch, for exactly the reason the LAN pair is two.

The dual pair is NOT a strictly better default. It draws roughly 130 W more,
it depends on both cards actually being installed, and the artifact it serves
has never had its quality measured here. Which icon is right is the
developer's call, which is why neither pair implies the other.

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
DUAL = os.path.join(ROOT, "serve-dual.bat")
DUAL_LAN = os.path.join(ROOT, "serve-dual-lan.bat")

ALL = [PLAIN, LAN, DUAL, DUAL_LAN]
SINGLE_CARD = [PLAIN, LAN]
TWO_CARD = [DUAL, DUAL_LAN]
EXPOSED = [LAN, DUAL_LAN]
LOOPBACK = [PLAIN, DUAL]


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


@pytest.mark.parametrize("path", ALL)
def test_the_launcher_exists_at_the_root(path):
    assert os.path.isfile(path), path


@pytest.mark.parametrize("path", ALL)
def test_it_anchors_on_its_own_folder(path):
    """%CD% is not the .bat's folder when it is opened from a shortcut or from
    a shell that started elsewhere."""
    assert "%~dp0" in read(path), (
        "%s resolves serve.ps1 relative to the working directory" % path)


@pytest.mark.parametrize("path", ALL)
def test_it_keeps_the_window_open(path):
    """The window stays because the SERVER runs in it -- serve.ps1 invokes the
    profile in-process and does not return while it lives.

    The first version passed -Follow, from the design where the server was
    detached and the window tailed its log. That switch no longer exists, and a
    .bat still passing it would fail at the first line with a parameter error --
    a launcher broken by a rename in the thing it launches."""
    t = read(path)
    assert "-Follow" not in t, (
        "%s passes -Follow, which serve.ps1 no longer accepts" % path)
    assert "-Detach" not in t, (
        "%s detaches; then the window closes and the point is lost" % path)
    assert "pause" in t.lower(), (
        "%s discards its own error message when it fails" % path)


@pytest.mark.parametrize("path", ALL)
def test_it_survives_a_restricted_execution_policy(path):
    assert "ExecutionPolicy Bypass" in read(path), (
        "%s fails on a default machine" % path)


@pytest.mark.parametrize("path", ALL)
def test_it_carries_no_serving_configuration(path):
    """Third file, same rule: the configuration lives in the profile."""
    t = read(path)
    for flag in ("--spec-type", "-ctk", "-ctv", "--reasoning-effort", ".gguf",
                 "-ngl", "--host"):
        assert flag not in t, "%s declares %r" % (path, flag)


@pytest.mark.parametrize("path", LOOPBACK)
def test_a_launcher_without_lan_in_its_name_does_not_expose(path):
    """The whole point of splitting the files. Clicking one that does not say
    lan must not bind every interface."""
    assert "-Lan" not in read(path), (
        "%s exposes the server; then the separate file is pointless and the "
        "exposure happens to anyone who wanted the model running" % path)


@pytest.mark.parametrize("path", EXPOSED)
def test_a_launcher_that_exposes_says_so_in_its_name(path):
    assert "-Lan" in read(path)
    assert "lan" in os.path.basename(path)


@pytest.mark.parametrize("path", TWO_CARD)
def test_the_dual_launchers_ask_for_both_cards(path):
    """Without -Dual the file is a duplicate of serve.bat with a misleading
    name -- the worst of the four outcomes, because the icon says one thing and
    the run does another."""
    assert "-Dual" in read(path), "%s does not pass -Dual" % path


@pytest.mark.parametrize("path", SINGLE_CARD)
def test_the_single_card_launchers_do_not_quietly_use_both(path):
    assert "-Dual" not in read(path), (
        "%s serves the two-card profile from an icon that does not say so" % path)


@pytest.mark.parametrize("path", TWO_CARD)
def test_a_dual_launcher_says_what_clicking_it_costs(path):
    """The dual pair is not a free upgrade: ~130 W more, both cards required,
    and the artifact's quality has never been measured here. An icon that
    presents it as strictly better is the launcher making the developer's
    decision for them."""
    t = read(path).lower()
    assert "quality" in t, "%s does not say what has not been measured" % path
    assert ("watt" in t or " w " in t or "power" in t), (
        "%s does not say it costs more power" % path)


def test_the_four_names_are_distinct_and_say_their_two_axes():
    names = [os.path.basename(p) for p in ALL]
    assert len(set(names)) == 4
    assert names == ["serve.bat", "serve-lan.bat",
                     "serve-dual.bat", "serve-dual-lan.bat"]


@pytest.mark.parametrize("path", ALL)
def test_it_is_readable_by_cmd(path):
    """A UTF-8 BOM makes cmd.exe choke on the first line -- it reads the BOM as
    part of the command and reports a garbled name. Plain ASCII, no BOM."""
    with open(path, "rb") as fh:
        raw = fh.read()
    assert not raw.startswith(b"\xef\xbb\xbf"), "%s starts with a BOM" % path
    raw.decode("ascii")  # raises if anything non-ASCII slipped in


# ---- the README is the first thing anyone reads, so it is a launcher too -----

README = os.path.join(ROOT, "README.md")


def test_the_readme_does_not_advertise_a_flag_serve_ps1_rejects():
    """It did. "Just start it" showed `.\serve.ps1 -Follow` and
    `-Lan -AllowFirewall -Follow` long after -Follow was removed with the
    detached design -- so the first command in the most-read file failed with a
    parameter error.

    The same paragraph also said "Ctrl+C stops the watching, not the server",
    which is now exactly backwards: the server runs IN the window.
    """
    readme = read(README)
    serve = read(os.path.join(ROOT, "serve.ps1"))
    for flag in ("-Follow", "-Detach"):
        if flag in readme:
            assert flag in serve, (
                "README.md tells the reader to pass %s and serve.ps1 does not "
                "accept it" % flag)


def test_the_readme_offers_the_two_card_launcher():
    """A profile nobody is told about is a file. #52 stage 5."""
    readme = read(README)
    assert "serve-dual.bat" in readme, (
        "README.md does not mention the two-card launcher")


def test_the_readme_says_what_the_two_card_option_costs():
    """Same rule as the .bat headers: presenting it as a free upgrade makes the
    developer's decision for them."""
    readme = read(README).lower()
    assert "quality" in readme
