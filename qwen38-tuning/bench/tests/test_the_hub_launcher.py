r"""One icon that asks which server you want, instead of twelve that assume.

WHY (developer, 2026-08-29): twelve `.bat` files at the repo root is a menu
already, just one rendered as filenames and read in a folder listing. A person
who has not memorised them cannot tell `serve-dual-nvfp4-deep-lan.bat` from
`serve-dual-mtp.bat` without opening both.

THE ONE RULE THAT MAKES THIS SAFE: **the hub holds no serving flag.** It picks a
`.bat` and calls it. Every launcher already refuses to hold configuration -- the
flags live in `qwen38-tuning/scripts/worker-q4-dual.ps1` and only there -- and a
chooser that assembled its own `-Dual -Nvfp4 -Vision` would be a second source
of truth that drifts from the first. This project has already shipped a launcher
that described a run it did not perform (trap 17, six instances).

THE LAN ANSWER DEFAULTS TO NO. `--host` is the only access control this server
has: no API key, CORS `*`, and `middleware_validate_api_key` returns true
immediately when no key is set. Exposure should be an act, which is why the
`lan` files were separate in the first place, and a hub that made LAN one
keystroke away from the default would undo that.

TESTING AN INTERACTIVE FILE. `test_a_bat_must_parse_in_cmd` runs every launcher
through real cmd with the launch line neutered; that helper also neuters `set /p`
so the menu does not wait for a human that is not there. What is still proven is
what matters: cmd parses the file and reaches a launch.
"""
import os
import re
import subprocess
import tempfile

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
HUB = os.path.join(ROOT, "serve-hub.bat")
LAUNCHERS = os.path.join(ROOT, "launchers")

# Every launcher the hub is allowed to offer, and the flag file that owns it.
OFFERED = [
    "serve-dual-nvfp4.bat", "serve-dual-nvfp4-lan.bat",
    "serve-dual-nvfp4-deep.bat", "serve-dual-nvfp4-deep-lan.bat",
    "serve-dual.bat", "serve-dual-lan.bat",
    "serve-dual-mtp.bat", "serve-dual-mtp-lan.bat",
    "serve-dual-dflash.bat", "serve-dual-dflash-lan.bat",
    "serve.bat", "serve-lan.bat",
]


def read(path):
    with open(path, "rb") as fh:
        return fh.read().decode("ascii")


def test_the_hub_exists():
    assert os.path.exists(HUB), HUB


def test_it_is_readable_by_cmd():
    raw = open(HUB, "rb").read()
    raw.decode("ascii")
    assert not raw.startswith(b"\xef\xbb\xbf"), "a BOM makes cmd choke"
    assert b"\r\n" in raw


def test_it_anchors_on_its_own_folder():
    assert "%~dp0" in read(HUB)


def test_it_holds_no_serving_flag():
    """The whole point. A chooser that assembles its own command line is a
    second source of truth for the flags."""
    t = read(HUB)
    for flag in ("-Nvfp4", "-Vision", "-Deep", "-MaxCtx", "-Mtp", "-Dflash",
                 "-Dual", "-sm ", "-ngl", "--spec-type", "-c 147456"):
        assert flag not in t, "%s appears in the hub; it belongs to the profile" % flag


def test_it_does_not_call_serve_ps1_directly():
    """Going straight to serve.ps1 would mean naming flags. It must go through
    the launcher that already owns them.

    Comments are exempt: the property is that it does not CALL it, and a file
    that explains why it does not is better than one that stays silent.
    """
    live = [l for l in read(HUB).splitlines()
            if not l.strip().lower().startswith("rem")]
    assert "serve.ps1" not in os.linesep.join(live)


def test_every_launcher_it_offers_exists():
    t = read(HUB)
    offered = [n for n in OFFERED if n in t]
    assert offered, "the hub offers nothing"
    for name in offered:
        assert os.path.exists(os.path.join(LAUNCHERS, name)), name


def test_it_offers_every_launcher_at_the_root():
    """A hub that silently omits an icon is worse than no hub: the icon still
    exists and the person who used the hub does not know it."""
    t = read(HUB)
    on_disk = {f for f in os.listdir(LAUNCHERS) if f.endswith(".bat")}
    missing = sorted(f for f in on_disk if f not in t)
    assert not missing, "not offered by the hub: %s" % missing


def test_the_lan_choice_is_not_the_default():
    """`--host` is the only access control this server has. Exposure is an act.
    The prompt must default to the loopback answer."""
    t = read(HUB)
    m = re.search(r"choice /c (\S+).*Expose on the LAN", t)
    assert m, "the LAN question is not asked through choice"
    assert m.group(1).upper().startswith("N"), (
        "the safe answer must be FIRST in the key list, so the neutered copy "
        "and a mis-keyed press both land on loopback: got %r" % m.group(1))


def test_it_says_what_lan_costs():
    t = read(HUB).lower()
    assert "no api key" in t or "no auth" in t, (
        "the hub offers LAN without saying the server has no authentication")


def _neutered_hub():
    out = []
    for line in read(HUB).splitlines():
        low = line.strip().lower()
        if low.startswith("choice "):
            # choice reports the POSITION in its key list through ERRORLEVEL.
            # `cmd /c exit 1` sets that to 1, which is the first key of both
            # prompts: menu option 1, and N for the LAN question. The safe
            # answer is deliberately first so a test picks it by default.
            out.append("cmd /c exit 1")
        elif low.startswith("set /p"):
            out.append("set SEL=1")
        elif low.startswith("call "):
            # ANY call, not only one naming a .bat literally: the hub builds
            # its target from a variable, so `call "%~dp0%PICK%"` carries no
            # ".bat" of its own and slipped through the narrower rule -- the
            # neutered copy then called an empty stub and emitted nothing.
            out.append("echo LAUNCH_REACHED")
        elif low == "pause":
            out.append("echo PAUSE_HIT")
        else:
            out.append(line)
    return "\r\n".join(out) + "\r\n"


def test_cmd_parses_it_and_a_choice_reaches_a_launch():
    """The siblings are STUBBED, not the existence check neutered.

    The hub refuses to call a launcher that is not there, and running the copy
    alone in a temp directory made that guard fire -- correctly. Stubbing the
    files exercises the guard instead of removing it, which also means the
    names the hub uses have to be the names on disk.
    """
    with tempfile.TemporaryDirectory() as d:
        os.mkdir(os.path.join(d, "launchers"))
        for name in OFFERED:
            open(os.path.join(d, "launchers", name), "w").close()
        copy = os.path.join(d, "serve-hub.bat")
        with open(copy, "w", encoding="ascii", newline="") as fh:
            fh.write(_neutered_hub())
        r = subprocess.run(["cmd.exe", "/c", copy], capture_output=True,
                           text=True, timeout=60)
    both = r.stdout + r.stderr
    assert "unexpected at this time" not in both, both.strip()
    assert "LAUNCH_REACHED" in both, both.strip()
