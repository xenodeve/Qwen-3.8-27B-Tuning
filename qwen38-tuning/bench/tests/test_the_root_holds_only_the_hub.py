r"""One icon at the top, the rest in a folder.

WHY (developer, 2026-08-29): twelve launchers at the repo root is a wall of
near-identical filenames in the one place a person looks first. The hub already
asks the question they answer; the root should ask it too, by having nothing
else in it.

WHAT MOVING THEM BREAKS, AND WHY THIS FILE EXISTS. Every launcher resolves its
own folder with `%~dp0` and used it for two things: finding `serve.ps1` and
setting the working directory. `%~dp0` follows the FILE, so a launcher in
`launchers/` resolves to `launchers/` and would look for `serve.ps1` there. The
failure is not subtle -- nothing starts -- but it is exactly the class of
launcher fault this repo keeps finding by running rather than reading, so it is
pinned here.
"""
import os

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
LAUNCHERS = os.path.join(ROOT, "launchers")

EXPECTED = sorted([
    "serve.bat", "serve-lan.bat",
    "serve-dual.bat", "serve-dual-lan.bat",
    "serve-dual-mtp.bat", "serve-dual-mtp-lan.bat",
    "serve-dual-dflash.bat", "serve-dual-dflash-lan.bat",
    "serve-dual-nvfp4.bat", "serve-dual-nvfp4-lan.bat",
    "serve-dual-nvfp4-deep.bat", "serve-dual-nvfp4-deep-lan.bat",
    "serve-dual-nvfp4-beta.bat", "serve-dual-nvfp4-beta-lan.bat",
    "serve-dual-nvfp4-beta-nokvu.bat", "serve-dual-nvfp4-beta-nokvu-lan.bat",
])


def read(path):
    with open(path, "rb") as fh:
        return fh.read().decode("ascii")


def test_the_root_holds_exactly_one_bat():
    at_root = sorted(f for f in os.listdir(ROOT) if f.lower().endswith(".bat"))
    assert at_root == ["serve-hub.bat"], at_root


def test_the_launchers_folder_holds_the_rest():
    assert os.path.isdir(LAUNCHERS), LAUNCHERS
    found = sorted(f for f in os.listdir(LAUNCHERS) if f.lower().endswith(".bat"))
    assert found == EXPECTED, found


@pytest.mark.parametrize("name", EXPECTED)
def test_each_launcher_reaches_the_entry_point_from_where_it_now_lives(name):
    """`%~dp0` follows the file. A launcher one directory down must climb.

    Asserted as a RESOLVED PATH that exists, not as the string `..`: the
    property is that the file it points at is there.
    """
    text = read(os.path.join(LAUNCHERS, name))
    live = [l for l in text.splitlines() if not l.strip().lower().startswith("rem")]
    # A line can NAME serve.ps1 without pointing at it -- the failure
    # message echoes it. Only the anchored ones are paths.
    hits = [l for l in live if "serve.ps1" in l and "%~dp0" in l]
    assert hits, "%s no longer names serve.ps1" % name
    for line in hits:
        i = line.index("%~dp0")
        rel = line[i + len("%~dp0"):].split('"')[0]
        resolved = os.path.normpath(os.path.join(LAUNCHERS, rel.replace("\\", os.sep)))
        assert os.path.exists(resolved), (
            "%s points at %s, which does not exist" % (name, resolved))


@pytest.mark.parametrize("name", EXPECTED)
def test_each_launcher_still_sets_a_working_directory_that_exists(name):
    text = read(os.path.join(LAUNCHERS, name))
    live = [l for l in text.splitlines() if not l.strip().lower().startswith("rem")]
    cds = [l for l in live if l.strip().lower().startswith("cd /d")]
    for line in cds:
        i = line.index("%~dp0")
        rel = line[i + len("%~dp0"):].split('"')[0]
        resolved = os.path.normpath(os.path.join(LAUNCHERS, rel.replace("\\", os.sep)))
        assert os.path.isdir(resolved), (
            "%s cds to %s, which is not a directory" % (name, resolved))


def test_the_hub_reaches_them_where_they_now_are():
    hub = read(os.path.join(ROOT, "serve-hub.bat"))
    live = [l for l in hub.splitlines() if not l.strip().lower().startswith("rem")]
    joined = "\n".join(live)
    assert "launchers" in joined, (
        "the hub does not mention the folder its targets moved into")
