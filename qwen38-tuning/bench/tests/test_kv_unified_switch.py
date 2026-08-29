r"""`-NoKvUnified`: one flag out of the `-Beta` bundle, on purpose.

WHY THIS SWITCH EXISTS

2026-08-29, same machine, same artifact, same evening, Discord streaming
throughout both:

                        Unsloth Studio          this project's -Beta
    prefill             728 - 1,000 tok/s       319 - 633
    decode              34.9 - 48.0             24.1 - 29.0
    decode overall      39.86                   26.17
    at depth ~47k       36.63                   26.93 - 29.02
    mean accepted len   1.81 - 2.52             2.51 - 2.81

**Our drafting is BETTER on every row and we are still slower**, which puts the
cost in the target model's forward pass rather than in speculation -- so the
suspects are the flags that change how attention and the KV cache are laid out,
not the decoder settings.

`--kv-unified` is the first of them: we set it inside `-Beta`, Studio does not
(`kv_unified = 'false'` in its boot log). It is also the only candidate that
would explain the OTHER unexplained difference -- Studio reuses a 39,616-token
prefix with `--ctx-checkpoints 0`, where the same setting made every one of our
requests re-read the prompt from token 0 and say

    forcing full prompt re-processing due to lack of cache data
    (likely due to SWA or hybrid/recurrent memory)

A single shared KV buffer is a plausible reason a partial sequence removal
cannot be done. **Plausible. Not measured.** That is what the switch is for.

WHY A SWITCH AND NOT AN EDIT. `-Beta` is nine settings adopted together, and
this project has already been caught changing two things and reading the
result as one (`--spec-draft-n-max 2` beside `n-min 48/64`, CORRECTIONS in
`docs/results/02-decoders.md`). One flag, one switch, one A/B the developer can
run against icon 7 without editing anything.
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

BATS = ["serve-dual-nvfp4-beta-nokvu.bat", "serve-dual-nvfp4-beta-nokvu-lan.bat"]


def _whatif(script, *args):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", script, "-WhatIf"] + list(args),
        capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr


# ------------------------------------------------------------------ the flag

def test_beta_still_sets_kv_unified_by_default():
    """The switch is opt-out. Icon 7 must not change underneath the A/B."""
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    assert "--kv-unified" in out, out


def test_no_kv_unified_removes_it():
    out = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta", "-NoKvUnified")
    assert "--kv-unified" not in out, out


def test_it_removes_ONLY_that_flag():
    """The whole point of a one-flag switch. If anything else moves, the A/B
    measures a bundle again and the answer cannot be attributed."""
    a = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta")
    b = _whatif(PROFILE, "-Nvfp4", "-Deep", "-Beta", "-NoKvUnified")

    def argv(text):
        m = re.search(r"WhatIf: would run\s*\n\s*(.+)", text)
        assert m, text
        return m.group(1).split()

    x, y = argv(a), argv(b)
    # -ts is computed from free VRAM at the moment of the call, so it may
    # legitimately differ between two dry runs seconds apart.
    def scrub(v):
        out, skip = [], False
        for t in v:
            if skip:
                skip = False
                continue
            if t == "-ts":
                skip = True
                continue
            out.append(t)
        return out
    a = [t for t in scrub(x) if t != "--kv-unified"]
    b = [t for t in scrub(y) if t != "--no-kv-unified"]
    assert a == b, ("more than the kv-unified flag changed", a, b)


def test_the_switch_does_nothing_without_beta():
    """`--kv-unified` is only ever set by the bundle, so the opt-out has
    nothing to opt out of anywhere else. It must not become a second way to
    change the default profile."""
    a = _whatif(PROFILE, "-Nvfp4")
    b = _whatif(PROFILE, "-Nvfp4", "-NoKvUnified")
    assert "--kv-unified" not in a
    assert "--kv-unified" not in b


def test_it_reaches_the_profile_through_serve():
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Deep", "-Beta", "-NoKvUnified")
    assert re.search(r"NoKvUnified\s+True", out), out


# --------------------------------------------------------------- the launchers

@pytest.mark.parametrize("name", BATS)
def test_the_launcher_exists(name):
    assert os.path.exists(os.path.join(LAUNCHERS, name))


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_asks_for_beta_and_the_opt_out(name):
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read()
    live = [l for l in body.splitlines() if not l.strip().upper().startswith("REM")]
    line = "\n".join(live)
    assert "-Beta" in line, line
    assert "-NoKvUnified" in line, line
    assert "-Deep" in line and "-Vision" in line, line


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_holds_no_llama_flag(name):
    """Every .bat here points at the profile; none writes a server flag. A
    chooser that assembled its own command line would be a second source of
    truth, and this project has shipped a launcher that described a run it did
    not perform."""
    body = open(os.path.join(LAUNCHERS, name), encoding="ascii").read()
    live = [l for l in body.splitlines()
            if not l.strip().upper().startswith("REM")]
    assert "--kv-unified" not in "\n".join(live)


def test_lan_and_loopback_differ_only_in_lan():
    a = open(os.path.join(LAUNCHERS, BATS[0]), encoding="ascii").read()
    b = open(os.path.join(LAUNCHERS, BATS[1]), encoding="ascii").read()
    def call(t):
        return [l for l in t.splitlines() if "serve.ps1" in l][0]
    assert call(b).replace(" -Lan", "") == call(a), (call(a), call(b))


@pytest.mark.parametrize("name", BATS)
def test_the_launcher_parses_in_cmd(name):
    """A .bat is not checked by anything until it runs. Four NVFP4 launchers
    shipped dead here on 2026-08-29 -- `^^(pwsh^^)` inside an `if errorlevel`
    block is a cmd parse error -- while 29 assertions about their text passed,
    because none of them ran cmd."""
    p = os.path.join(LAUNCHERS, name)
    r = subprocess.run(["cmd", "/c", "echo off & type nul & call :nothing 2>nul"
                        " || (cmd /c \"exit /b 0\")"],
                       capture_output=True, text=True, timeout=30)
    del r
    # Parse without executing: cmd reports a syntax error at parse time when
    # the file is read, so run it with a guard that exits before the payload.
    r = subprocess.run(["cmd", "/c", "set QWEN38_PARSE_ONLY=1& call \"%s\"" % p],
                       capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, QWEN38_PARSE_ONLY="1"))
    bad = ("was unexpected at this time", "The syntax of the command is incorrect")
    assert not any(b in (r.stdout + r.stderr) for b in bad), r.stdout + r.stderr


# ---------------------------------------------------------------------- the hub

def test_the_hub_offers_it():
    body = open(HUB, encoding="ascii").read()
    assert "serve-dual-nvfp4-beta-nokvu.bat" in body, "hub does not route to it"
    assert "serve-dual-nvfp4-beta-nokvu-lan.bat" in body


def test_every_hub_key_is_wired_to_a_file():
    """The hub's `choice /c` list and its `if` branches are two lists that must
    stay the same length. They were, before this icon."""
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


@pytest.mark.parametrize("name", BATS)
def test_the_hub_points_at_a_file_that_exists(name):
    body = open(HUB, encoding="ascii").read()
    if name in body:
        assert os.path.exists(os.path.join(LAUNCHERS, name))
