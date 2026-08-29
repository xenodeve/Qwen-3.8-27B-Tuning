r"""`-Clone`: Unsloth Studio's command line, on our binary, as a baseline.

WHY A CLONE AND NOT MORE SWITCHES

By 2026-08-30 the difference between Studio's server and ours was down to a
handful of flags, and every one of them had a plausible story:

    ours                        Studio
    -c 200704                   -c 107899
    -ts 7598,15288  (33/67)     -ts 7648,13509  (36/64)
    -ub 1024                    -ub 512   (default)
    -ngl auto                   -ngl -1
    --kv-unified                (unset)
    --ctx-checkpoints 32        --ctx-checkpoints 0
    --spec-draft-n-max 3        --spec-draft-n-max 2
    (unset)                     --slot-save-path ...
    (unset)                     --jinja, --no-context-shift explicit

Testing them one at a time is eight boots before the first answer. A clone is
ONE boot that says whether the remaining gap is in this list at all -- and if
the clone matches Studio, bisecting is worth doing; if it does not, the cause is
somewhere none of these flags reach and eight sweeps would have found nothing.

**A baseline, not a candidate.** Nothing here is proposed for serving. `-c
107,899` is half the window this machine exists to serve.

WHAT IS DELIBERATELY *NOT* CLONED, AND WHY EACH ONE

A literal copy would reproduce their bugs and break the comparison, so five
things stay ours. Each is asserted below so it cannot drift into silence.

  --reasoning-effort medium   ADDED. It is not on their command line because
                              Studio sends it in every request body. We have no
                              client that does, so copying the OMISSION serves
                              at the template's xhigh -- which is exactly how
                              CORRECTIONS 36 happened.
  --reasoning on
  --reasoning-preserve        INSTEAD of --chat-template-kwargs {...}. Our
                              build answers that kwarg with "deprecated" and
                              then asks for --reasoning-preserve anyway. Cloning
                              it would clone a bug.
  --alias                     OURS. The alias is the model name a client asks
                              for; changing it would mean changing the client
                              too, and then the A/B has two variables.
  -lv 4                       OURS. `forcing full prompt re-processing` and
                              `cached n_tokens` only print above verbosity 3,
                              and those lines are the reason to run this at all.
  --host/--port               OURS. Studio picks a random port per launch.
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
BATS = ["serve-dual-nvfp4-clone.bat", "serve-dual-nvfp4-clone-lan.bat"]


def _whatif(script, *args):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", script, "-WhatIf"] + list(args),
        capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr


def argv():
    out = _whatif(PROFILE, "-Nvfp4", "-Vision", "-Clone")
    m = re.search(r"WhatIf: would run\s*\n\s*(.+)", out)
    assert m, out
    return m.group(1).split()


def val(v, flag):
    assert flag in v, (flag, v)
    return v[v.index(flag) + 1]


# ------------------------------------------------- the flags copied literally

@pytest.mark.parametrize("flag,want", [
    ("-c", "107899"),           # their window, half of what this machine serves
    ("-ts", "7648,13509"),      # 36/64 against our 33/67
    ("-ub", "512"),             # their default; ours is 1024 and MEASURED +10.1 %
    ("-b", "2048"),
    ("-ngl", "-1"),             # ours says auto
    ("-t", "2"),
    ("-np", "1"),
    ("-fa", "on"),
    ("-sm", "tensor"),
    ("-ctk", "q4_0"),
    ("-ctv", "q4_0"),
    ("--fit", "off"),
    ("--spec-type", "draft-mtp"),
    ("--spec-draft-n-max", "2"),
    ("--cache-ram", "0"),
    ("--ctx-checkpoints", "0"),
    ("--load-mode", "none"),
])
def test_the_clone_carries_their_value(flag, want):
    assert val(argv(), flag) == want


@pytest.mark.parametrize("flag", ["--metrics", "--jinja", "--no-context-shift",
                                  "--slot-save-path"])
def test_the_clone_carries_their_bare_flag(flag):
    assert flag in argv()


def test_the_clone_does_not_set_kv_unified():
    """Studio's boot log reads `kv_unified = 'false'`. Ours sets it inside
    -Beta, and it is the leading suspect for a decode gap that does not widen
    with depth."""
    assert "--kv-unified" not in argv()


def test_the_clone_still_loads_the_vision_tower():
    """Studio passes --mmproj. A clone without it is a different model."""
    v = argv()
    assert "-mm" in v or "--mmproj" in v, v
    assert "--no-mmproj-auto" not in v


# ------------------------------------------- the five deliberate deviations

def test_the_clone_sets_the_effort_although_studio_does_not():
    """THE DEVIATION THAT MATTERS MOST. Copying their omission means serving at
    the chat template's xhigh, because no client of ours sends an effort per
    request. That is CORRECTIONS 36, and a clone that reproduced it would
    measure thinking length rather than throughput."""
    assert val(argv(), "--reasoning-effort") == "medium"


def test_the_clone_uses_our_thinking_flags_not_their_deprecated_kwarg():
    v = argv()
    assert "--chat-template-kwargs" not in v, (
        "our build answers that with 'deprecated' and then asks for "
        "--reasoning-preserve anyway")
    assert "--reasoning-preserve" in v
    assert val(v, "--reasoning") == "on"


def test_the_clone_keeps_our_alias():
    assert val(argv(), "--alias") == "Qwen3.8-27B-NVFP4-MTP"


def test_the_clone_keeps_a_verbosity_that_shows_the_cache_lines():
    """`forcing full prompt re-processing` is the line this baseline exists to
    read, and it does not print at Studio's verbosity 3."""
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Vision", "-Clone")
    assert re.search(r"-Verbosity\s+4", out), out


# ---------------------------------------------------- it does not leak anywhere

def test_without_clone_nothing_changes():
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Vision", "-Beta")
    assert "107899" not in out, out
    assert re.search(r"-ub\s+1024", out), out
    assert re.search(r"--spec-draft-n-max\s+3", out), out


def test_clone_and_beta_together_are_refused():
    """-Beta is our bundle and -Clone is theirs. Both at once is a command line
    nobody can attribute a number to."""
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Beta", "-Clone")
    assert "FATAL" in out, out


def test_clone_needs_two_cards():
    out = _whatif(SERVE, "-Nvfp4", "-Clone")
    assert "FATAL" in out, out


# ------------------------------------------------------------- the launchers

@pytest.mark.parametrize("name", BATS)
def test_the_launcher_exists(name):
    assert os.path.exists(os.path.join(LAUNCHERS, name))


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_asks_for_clone_and_holds_no_flag(name):
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read()
    live = "\n".join(l for l in body.splitlines()
                     if not l.strip().upper().startswith("REM"))
    assert "-Clone" in live, live
    for leaked in ("--cache-ram", "-ts ", "--spec-type"):
        assert leaked not in live, (leaked, live)


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_says_it_is_a_baseline_not_a_recommendation(name):
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read().upper()
    assert "BASELINE" in body
    assert "107,899" in body or "107899" in body, \
        "the halved window must be stated where somebody will read it"


def test_lan_and_loopback_differ_only_in_lan():
    a = open(os.path.join(LAUNCHERS, BATS[0]), encoding="ascii").read()
    b = open(os.path.join(LAUNCHERS, BATS[1]), encoding="ascii").read()
    def call(t):
        return [l for l in t.splitlines() if "serve.ps1" in l][0]
    assert call(b).replace(" -Lan", "") == call(a)


# ---------------------------------------------------------------------- the hub

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
    assert 'if "%%SEL%%"=="%d" goto :done' % (n + 1) in body, \
        "the quit position no longer matches the key list"


def test_the_slot_directory_is_ignored_by_git():
    """--slot-save-path writes KV state, which is prompt content."""
    ig = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read()
    assert "llama-slots" in ig, ig
