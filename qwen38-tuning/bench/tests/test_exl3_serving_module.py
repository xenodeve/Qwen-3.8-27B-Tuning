r"""Our EXL3 server customisations live in qwen38-tuning/serving/exl3, not as
in-place edits of the fork (2026-09-04, issues #71 / #73).

WHY. Three patches went into C:\AI\exllamav3-mia\tools\serve_openai.py in one
day (--extra, live timing, effort, then the Anthropic routes), 241 changed
lines against a tree we pull updates into. The developer asked for a separate
module so a fork update does not erase the custom parts. The shape:

  serving/exl3/server.py            the fork's file + hooks marked `# xeno:`
  serving/exl3/upstream/serve_openai.py   the pristine copy it was cut from
  serving/exl3/upstream/COMMIT      the fork commit of that copy
  serving/exl3/{live_timing,effort,anthropic_compat,anthropic_routes}.py  ours

These tests pin that shape: the fork's tools/ stays pristine, every hook in
server.py is marked, the recorded base matches the fork commit, and the
launcher runs our server, not the fork's.
"""
import os
import re
import subprocess

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING = os.path.dirname(BENCH)
ROOT = os.path.dirname(TUNING)
SERVING = os.path.join(TUNING, "serving", "exl3")
FORK = os.environ.get("EXL3_FORK_DIR", r"C:\AI\exllamav3-mia")
OURS = ("live_timing", "effort", "anthropic_compat", "anthropic_routes", "watchdog", "loop_guard")


def read(name):
    with open(os.path.join(SERVING, name), encoding = "utf-8") as fh:
        return fh.read()


def test_every_custom_module_exists_and_server_imports_them():
    for m in OURS:
        assert os.path.exists(os.path.join(SERVING, m + ".py")), m
    server = read("server.py")
    assert "import live_timing, effort, anthropic_routes" in server
    # the translator is reached only through the routes module
    assert "import anthropic_compat" not in server


def test_nothing_of_ours_is_defined_inline_in_server():
    server = read("server.py")
    for sym in ("class _LiveTiming", "class LiveTiming", "def resolve_effort",
                "async def anthropic_messages", "def _anthropic_error", "EFFORT_ALIAS"):
        assert sym not in server, sym


def test_every_line_that_differs_from_upstream_is_a_marked_hook_or_header():
    """A fork update is a three-way merge; the merge is only cheap if every
    line we changed is findable. Unmarked additions are what silently vanish.
    The header docstring (everything before the first import) is exempt."""
    import difflib
    base = read(os.path.join("upstream", "serve_openai.py")).splitlines()
    ours = read("server.py").splitlines()
    first_import = next(i for i, l in enumerate(ours) if l.startswith("import argparse"))
    header = set(ours[:first_import])
    added = [l[1:] for l in difflib.unified_diff(base, ours, lineterm = "", n = 0)
             if l.startswith("+") and not l.startswith("+++")]
    unmarked = [l for l in added if "xeno" not in l and l.strip() and l not in header
                and not l.startswith("import argparse")]
    assert unmarked == [], unmarked


@pytest.mark.skipif(not os.path.isdir(os.path.join(FORK, ".git")), reason = "fork clone not on this machine")
def test_the_recorded_upstream_copy_is_the_fork_commit_it_names():
    commit = read(os.path.join("upstream", "COMMIT")).strip()
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    shown = subprocess.run(["git", "-C", FORK, "show", f"{commit}:tools/serve_openai.py"],
                           capture_output = True, text = True, encoding = "utf-8").stdout
    assert shown.replace("\r\n", "\n") == read(os.path.join("upstream", "serve_openai.py")).replace("\r\n", "\n")


@pytest.mark.skipif(not os.path.isdir(os.path.join(FORK, ".git")), reason = "fork clone not on this machine")
def test_the_fork_tools_tree_carries_none_of_our_edits():
    out = subprocess.run(["git", "-C", FORK, "status", "--porcelain", "--", "tools"],
                         capture_output = True, text = True).stdout.strip()
    assert out == "", out


def test_the_model_name_comes_from_the_loaded_directory_not_a_literal():
    """2026-09-04: the served file became SC 4.00bpw H5 while /v1/models and every
    response still said `qwen3.8-27b-exl3-3.5bpw-wm`, and Claude Code showed that.
    The name must derive from the -m directory; the profile launcher reads it."""
    server = read("server.py")
    assert 'MODEL_NAME = os.path.basename' in server
    assert '"id": "qwen3.8-27b-exl3-3.5bpw-wm"' not in server
    assert 'body.get("model", "qwen3.8-27b-exl3-3.5bpw-wm")' not in server
    bat = open(os.path.expanduser("~/.claude/claude-xeno-exl3.bat"), encoding = "ascii").read()
    assert "/v1/models" in bat and "--model=%MODELID%" in bat


def test_the_launcher_runs_our_server_not_the_forks():
    with open(os.path.join(TUNING, "scripts", "serve-exl3.cmd"), "rb") as fh:
        cmd = fh.read().decode("ascii")
    code = [l for l in cmd.splitlines() if not l.strip().lower().startswith("rem")]
    assert any("python.exe C:\\AI\\qwen38-tuning\\serving\\exl3\\server.py" in l for l in code), code
    assert not any("tools\\serve_openai.py" in l for l in code)
