r"""`--chat-template-file` must reach every launch, whatever else is switched on.

WHAT THIS GUARDS, issue #58.

Qwen3.8's own chat template counts the contiguous leading run of `system` and
`developer` messages and then RAISES on any that come later:

    line 47   {%- if sysns.count == loop.index0 and (message.role == 'system' ...
    line 110  {{- raise_exception('System message must be at the beginning.') }}

Claude Code sends exactly that -- its `SessionStart` hook output arrives as a
`role: "system"` message of 25-33 KB appended after the user turn. Issue #4 fixed
it in 2026-08-21 with `templates/qwen38-late-system.jinja`, the model's own
template with that one line rendering an ordinary system turn instead.

THE FIX THEN GOT LOST IN TWO PLACES, and neither loss was visible:

    $thinkArg = if ($Beta) { ...no template file... } else { ...template file... }

`-Beta` borrows Unsloth Studio's thinking mechanism, Studio passes no template
file, and the two unrelated concerns were bundled into one `if/else`. The
`$Clone` branch rebuilds `$argv` from scratch and never had it either. Five hub
icons -- 7, 8, 9, A, B -- returned HTTP 500 to every Claude Code request, fifteen
in a row in `logs/serve-20260831-023636.log` before the client gave up.

WHY THE OMISSION IS NOT A BASELINE. Studio does not need the fix because Studio's
client never sends a late system message. Copying the omission therefore
reproduces a client incompatibility, not a command line. This is CORRECTIONS 36
a second time: `-Beta` dropped `--reasoning-effort` for the same reason -- to
match Studio -- and served at `xhigh` for an afternoon because Studio supplies
the effort per REQUEST and we have no client that does.

SO THE SWEEP BELOW IS THE POINT OF THIS FILE. Asserting the two branches that
broke would pass today and miss the third. Every combination a launcher can
produce is checked, and the only way to omit the flag is to ask for it.
"""
import io
import os
import re
import subprocess

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q4-dual.ps1")
SERVE = os.path.join(ROOT, "serve.ps1")
TEMPLATE = os.path.join(ROOT, "qwen38-tuning", "templates",
                        "qwen38-late-system.jinja")
RAISE = "raise_exception('System message must be at the beginning.')"

# Every switch combination a launcher in `launchers/` can produce. The names are
# the hub icons so a failure says which one a person would have clicked.
ICONS = [
    ("1  nvfp4",                    ["-Nvfp4", "-Vision"]),
    ("2  nvfp4 deep",               ["-Nvfp4", "-Vision", "-Deep"]),
    ("4  mtp",                      ["-Mtp", "-Vision"]),
    ("5  dflash",                   ["-Dflash", "-Vision"]),
    ("7  nvfp4 beta",               ["-Nvfp4", "-Vision", "-Beta"]),
    ("8  nvfp4 beta nokvu",         ["-Nvfp4", "-Vision", "-Beta", "-NoKvUnified"]),
    ("9  nvfp4 clone",              ["-Nvfp4", "-Vision", "-Clone"]),
    ("A  nvfp4 clone theirbuild",   ["-Nvfp4", "-Vision", "-Clone", "-TheirBuild"]),
    ("B  nvfp4 beta theirbuild",    ["-Nvfp4", "-Vision", "-Beta", "-TheirBuild"]),
    ("D  nvfp4 dflash",             ["-Nvfp4", "-Vision", "-Dflash"]),
    ("E  nvfp4 dflash theirmirror", ["-Nvfp4", "-Vision", "-Dflash", "-TheirMirror"]),
]


def _whatif(script, *args):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", script, "-WhatIf"] + list(args),
        capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr


def preview(*args):
    out = _whatif(PROFILE, *args)
    m = re.search(r"WhatIf: would run\s*\n\s*(.+)", out)
    assert m, out
    return m.group(1), out


# ------------------------------------------------- the sweep, one row per icon

@pytest.mark.parametrize("icon,args", ICONS, ids=[i for i, _ in ICONS])
def test_every_icon_carries_the_template(icon, args):
    """A launch without it answers HTTP 500 to Claude Code, which is the only
    client this project actually serves."""
    line, _ = preview(*args)
    assert "--chat-template-file" in line, (icon, line)
    assert "qwen38-late-system.jinja" in line, (icon, line)


# --------------------------------------------- omitting it has to be asked for

def test_stocktemplate_omits_it():
    """Studio's template behaviour stays reachable -- explicitly, by someone who
    means it, rather than as a side effect of choosing a thinking mechanism."""
    line, _ = preview("-Nvfp4", "-Vision", "-StockTemplate")
    assert "--chat-template-file" not in line, line


def test_stocktemplate_composes_with_beta():
    """`-Beta -StockTemplate` is what `-Beta` alone used to be: Studio's whole
    mechanism including the omission. Nothing about the experiment is lost."""
    line, _ = preview("-Nvfp4", "-Vision", "-Beta", "-StockTemplate")
    assert "--chat-template-file" not in line, line
    assert "--reasoning-preserve" in line, line
    assert re.search(r"--reasoning\s+on", line), line


def test_stocktemplate_composes_with_clone():
    line, _ = preview("-Nvfp4", "-Vision", "-Clone", "-StockTemplate")
    assert "--chat-template-file" not in line, line


def test_beta_still_borrows_the_thinking_mechanism():
    """The two concerns are now separate, so separating them must not have taken
    the mechanism with it."""
    line, _ = preview("-Nvfp4", "-Vision", "-Beta")
    assert "--reasoning-preserve" in line, line
    assert re.search(r"--reasoning\s+on", line), line
    assert re.search(r"--reasoning-effort\s+medium", line), line
    assert "--chat-template-kwargs" not in line, "deprecated on this build"


# --------------------------------------------------------- the launch guard

def test_the_profile_refuses_a_launch_that_lost_the_flag():
    """The structural half, and the only half that stops a third recurrence.

    Items 1 and 2 fix the two branches that broke. A branch written next month
    that rebuilds `argv` -- which is exactly what `$Clone` does -- would drop it
    again and nothing would say so until a client saw 500s. The guard reads the
    FINAL argv, so it does not care how the argv was built.
    """
    src = io.open(PROFILE, encoding="utf-8", errors="replace").read()
    assert "StockTemplate" in src, "the switch does not exist"
    # the guard must test the assembled argv, not a branch variable
    m = re.search(r"if\s*\(\s*-not\s*\$StockTemplate[^\n]*\)[\s\S]{0,400}?FATAL",
                  src)
    if not m:
        m = re.search(r"\$argv[^\n]{0,80}chat-template-file[\s\S]{0,600}?FATAL", src)
    assert m, "nothing refuses a launch whose final argv lost the template"


# ------------------------------------------------------------ the file itself

def test_the_template_exists_and_is_the_patched_one():
    assert os.path.isfile(TEMPLATE), TEMPLATE
    body = io.open(TEMPLATE, encoding="utf-8", errors="replace").read()
    assert RAISE not in body, "this is the stock template, not the patched one"
    assert "'<|im_start|>system\\n' + content + '<|im_end|>'" in body, (
        "the replacement line is not the one the README documents")


def test_it_differs_from_the_served_artifacts_own_template_by_one_line():
    """The template belongs to the MODEL, so it has to be re-derived when the
    artifact changes -- `templates/README.md` says so and nothing enforced it.

    Checked against a live server because `/props` is where the stock text
    lives. Skipped when nothing is serving; the offline assertions above still
    hold, and this one is the strong form when it can run.
    """
    import json
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/props", timeout=20) as r:
            stock = json.loads(r.read()).get("chat_template") or ""
    except Exception:                                            # noqa: BLE001
        pytest.skip("no server on 8080 to read /props from")
    if not stock:
        pytest.skip("/props carried no chat_template")
    ours = io.open(TEMPLATE, encoding="utf-8", errors="replace").read()
    a = stock.replace("\r\n", "\n").rstrip("\n").split("\n")
    b = ours.replace("\r\n", "\n").rstrip("\n").split("\n")
    assert len(a) == len(b), (
        "the served artifact's template has a different shape; re-derive it "
        "per templates/README.md", len(a), len(b))
    differing = [i for i, (x, y) in enumerate(zip(a, b), 1) if x != y]
    assert differing == [110] or len(differing) == 1, (
        "exactly one line may differ, and it is the raise", differing)
    assert RAISE in a[differing[0] - 1], a[differing[0] - 1]


# ---------------------------------------------------------------- plumbing

def test_serve_forwards_the_switch():
    out = _whatif(SERVE, "-Dual", "-Nvfp4", "-Vision", "-StockTemplate")
    assert re.search(r"StockTemplate\s+True", out), out

# ------------------------------------------- the other six profiles, issue #58

# `worker-q4-dual.ps1` lost the flag because it ASSEMBLES its argv from pieces
# and two of those pieces forgot. The other six build one flat command line with
# the flag as a literal, and every conditional in them is a preflight guard that
# `exit 1`s rather than a branch that edits argv -- checked one by one on
# 2026-08-31, so none of them can lose it the way the dual profile did.
#
# THEY STILL GET A TEST, because "cannot lose it by construction" is a property
# of today's construction. Deleting or commenting out one line is all it takes,
# and the failure is silent until a client sees HTTP 500. The dual profile got a
# runtime guard; these get this.
FLAT_PROFILES = [
    "worker-5060ti.ps1",
    "worker-iq2s-2slot.ps1",
    "worker-iq2s-fast.ps1",
    "worker-iq2s-quality.ps1",
    "worker-iq2xxs-deep.ps1",
    "worker-q2kxl-mtp.ps1",
]
SCRIPTS = os.path.join(ROOT, "qwen38-tuning", "scripts")


def _live_lines(path):
    """The script minus comments. A flag inside a `#` line is documentation and
    does not reach llama-server -- which is exactly how this would regress."""
    out = []
    for line in io.open(path, encoding="utf-8", errors="replace").read().splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith("<#") or s.startswith("."):
            continue
        out.append(line)
    return "\n".join(out)


@pytest.mark.parametrize("name", FLAT_PROFILES)
def test_every_other_profile_ships_the_template(name):
    path = os.path.join(SCRIPTS, name)
    assert os.path.isfile(path), path
    live = _live_lines(path)
    assert "--chat-template-file" in live, (
        name, "the flag is gone from the command line, or only in a comment")
    assert "qwen38-late-system.jinja" in live, name


@pytest.mark.parametrize("name", FLAT_PROFILES + ["worker-q4-dual.ps1"])
def test_the_template_path_each_profile_names_exists(name):
    """A renamed template file would break every launch and nothing would say so
    until a server was started. These paths are literals; nothing resolves them
    until PowerShell does."""
    body = _live_lines(os.path.join(SCRIPTS, name))
    # Two shapes: the flag followed by a literal, and `worker-q4-dual.ps1`,
    # which names the path once in $TEMPLATE_FILE and passes the variable. A
    # test that only knew the first shape would report the dual profile has no
    # template at all -- which is the opposite of true.
    m = (re.search(r'--chat-template-file[",\s]+"?([^"\n`]+\.jinja)', body)
         or re.search(r'\$TEMPLATE_FILE\s*=\s*"([^"\n]+\.jinja)"', body))
    assert m, (name, "no template path found on the command line")
    raw = m.group(1).strip()
    resolved = raw.replace("$PSScriptRoot", SCRIPTS)
    assert os.path.isfile(resolved), (name, raw, resolved)
