r"""Icon B: icon 7's settings on Unsloth's binary — the fourth cell of the 2x2.

WHY THIS ONE AND NOT ANOTHER CLONE

`-Clone` fixes the window at 107,899 because that is what Studio's server
happened to compute at 00:11. It is a baseline, not a way to serve: this machine
exists to serve 200,704. Icon A therefore answers "is the gap flags or build"
and then has nothing to offer afterwards.

This icon is the useful half. It is **icon 7 exactly** — the deepest window, our
micro-batch, our draft depth, our thinking flags — with only the binary changed:

                    our flags        their flags
    our build       1 / 2 / 7 / 8    icon 9   (-Clone)
    their build     **icon B**       icon A   (-Clone -TheirBuild)

MEASURED, at matched depth, and it is why the developer asked for it:

    ~48,000 depth   icon 8 (ours) 33.05 | icon 9 (their flags, our build) 33.00
                    icon A (their flags, THEIR build) **41.58**
    ~76,000 depth   icon 8 28.59  |  icon A **35.23**

Their whole flag set on our build changed **nothing** — 33.00 against 33.05.
Their build READ **+26 %** there, from ONE boot per side, and **the two binaries
have still never been paired.** A run on 2026-08-30 appeared to refute it and
did not: every arm launched the module default while every row recorded the pin,
so the "two builds" were one binary and the null was an artefact (CORRECTIONS 40
and 41). The status is **CONTESTED**, settled in neither direction.

That is what this icon is for: back to back against icon 7 it is the pairing
nobody has run.

WHAT THIS ICON IS NOT

Not a recommendation yet. One reading per side, taken in different boots, and
`CORRECTIONS.md` 23 measured the same arm drifting 48.9 % across boots at depth.
It is an A/B to run against icon 7 back to back, not a result to quote.
"""
import os
import re
import subprocess

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q4-dual.ps1")
SERVE = os.path.join(ROOT, "serve.ps1")
HUB = os.path.join(ROOT, "serve-hub.bat")
LAUNCHERS = os.path.join(ROOT, "launchers")
BATS = ["serve-dual-nvfp4-beta-theirbuild.bat",
        "serve-dual-nvfp4-beta-theirbuild-lan.bat"]
STUDIO_EXE = os.path.join(os.path.expanduser("~"), ".unsloth", "llama.cpp",
                          "build", "bin", "Release", "llama-server.exe")

BETA = ("-Nvfp4", "-Deep", "-Vision", "-Beta")


def _whatif(script, *args):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", script, "-WhatIf"] + list(args),
        capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr


def argv(*args):
    out = _whatif(PROFILE, *args)
    m = re.search(r"WhatIf: would run\s*\n\s*(.+)", out)
    assert m, out
    return m.group(1).split()


# ------------------------------------------------ it IS icon 7, bar the binary

def test_the_binary_is_theirs():
    v = argv(*BETA, "-TheirBuild")
    assert v[0].strip('"').lower() == STUDIO_EXE.lower(), v[0]


def test_nothing_else_moved():
    """The whole value of this icon. If any other flag differs, running it
    against icon 7 measures a bundle again and the answer cannot be attributed
    to the build."""
    a, b = argv(*BETA), argv(*BETA, "-TheirBuild")

    def scrub(v):
        # argv[0] is the binary, which is the point. -ts is computed from free
        # VRAM at the moment of the call and may legitimately differ between
        # two dry runs seconds apart.
        out, skip = [], False
        for t in v[1:]:
            if skip:
                skip = False
                continue
            if t == "-ts":
                skip = True
                continue
            out.append(t)
        return out

    assert scrub(a) == scrub(b), (scrub(a), scrub(b))


@pytest.mark.parametrize("flag,want", [
    ("-c", "200704"),                 # the depth this machine serves
    ("-ub", "1024"),                  # ours, MEASURED at +10.1 % prefill
    ("--spec-draft-n-max", "3"),      # ours; 2 was measured slower here
    ("--reasoning-effort", "medium"),  # CORRECTIONS 36
    ("--cache-ram", "0"),
    ("--load-mode", "none"),
])
def test_it_keeps_icon_sevens_values(flag, want):
    v = argv(*BETA, "-TheirBuild")
    assert v[v.index(flag) + 1] == want, (flag, v)


def test_it_keeps_the_bundle_flags():
    v = argv(*BETA, "-TheirBuild")
    for f in ("--kv-unified", "--metrics", "--reasoning-preserve"):
        assert f in v, (f, v)


def test_checkpoints_are_still_on():
    """`--ctx-checkpoints 0` left the -Beta bundle on 2026-08-29 after it made
    every request re-read the prompt from token 0 (CORRECTIONS 39). It must not
    come back through the binary switch.

    This asserted the flag was ABSENT until 2026-09-02, when the profile started
    naming it at **4** on its own evidence -- of 240 restores in
    `logs/serve-20260902-034815.log`, 185 used the newest checkpoint, 52 the
    second, 3 the third and none went deeper, against a default of 32. Absence
    was only ever a proxy for "not zero", and that is what the guard now says.
    """
    v = argv(*BETA, "-TheirBuild")
    assert v[v.index("--ctx-checkpoints") + 1] == "4", v


# ------------------------------------------------------------------ plumbing

def test_it_reaches_the_profile_through_serve():
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Deep", "-Vision", "-Beta",
                  "-TheirBuild")
    assert re.search(r"TheirBuild\s+True", out), out
    assert re.search(r"Beta\s+True", out), out


def test_the_preview_says_which_build():
    out = _whatif(PROFILE, *BETA, "-TheirBuild")
    assert "10679" in out, out


# ---------------------------------------------------------------- launchers

@pytest.mark.parametrize("name", BATS)
def test_the_launcher_exists(name):
    assert os.path.exists(os.path.join(LAUNCHERS, name))


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_asks_for_beta_and_their_build(name):
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read()
    live = "\n".join(l for l in body.splitlines()
                     if not l.strip().upper().startswith("REM"))
    for want in ("-Beta", "-TheirBuild", "-Deep", "-Vision"):
        assert want in live, (want, live)
    assert "-Clone" not in live, "this is icon 7's settings, not Studio's"


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_states_the_measurement_and_its_limit(name):
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read()
    assert "41.58" in body and "33.00" in body, \
        "the numbers that justify this icon are not where a reader will see them"
    assert "48.9" in body, \
        "one reading per side across boots must carry the drift this project measured"


def test_lan_and_loopback_differ_only_in_lan():
    a = open(os.path.join(LAUNCHERS, BATS[0]), encoding="ascii").read()
    b = open(os.path.join(LAUNCHERS, BATS[1]), encoding="ascii").read()
    def call(t):
        return [l for l in t.splitlines() if "serve.ps1" in l][0]
    assert call(b).replace(" -Lan", "") == call(a)


# ---------------------------------------------------------------------- hub

def test_the_hub_offers_it():
    body = open(HUB, encoding="ascii").read()
    for n in BATS:
        assert n in body, n


def test_every_hub_key_is_wired_to_a_file():
    body = open(HUB, encoding="ascii").read()
    m = re.search(r"choice /c (\S+) /n /m \"  Choose", body)
    assert m, "the menu prompt changed shape"
    keys = m.group(1)
    assert keys.endswith("Q"), keys
    n = len(keys) - 1
    for i in range(1, n + 1):
        assert 'if "%%SEL%%"=="%d" (' % i in body, ("key %d has no branch" % i, keys)
    assert 'if "%%SEL%%"=="%d" goto :done' % (n + 1) in body
