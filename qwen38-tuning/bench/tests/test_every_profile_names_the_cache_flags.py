r"""A profile that does not name `--cache-ram` and `--ctx-checkpoints` inherits
llama.cpp's defaults, and both defaults are wrong for the windows this machine
serves.

THIS IS THE GUARD, NOT THE VALUES. `test_prompt_cache_budget.py` asserts what
`worker-q4-dual.ps1` passes. This one asserts that **every profile `serve.ps1`
can dispatch to** names the two flags at all, so a profile added later cannot
silently inherit 8192 MiB and 32 the way `worker-q2kxl-mtp.ps1` did for months.

WHAT INHERITING THEM COST, measured on `logs/serve-20260902-034815.log`: a live
two-agent session lost **68.2 % of the wall clock in its last half hour** to
re-prefilling what the prompt cache had just evicted, because at ctx 200,704 one
conversation's state reaches 9,801 MiB against an 8192 MiB cap. Issue #70,
`docs/results/05-runtime-flags.md`, CORRECTIONS 46.

WHY A TEXT CHECK AND NOT A DRY RUN. `worker-q2kxl-mtp.ps1` has no `-WhatIf`, and
`worker-q4-dual.ps1`'s costs a GPU probe per call -- the developer stopped a
batch of ten mid-session for exactly that reason. The per-profile VALUES are
already covered by dry-run tests; this file only has to answer "is the flag
named", which the source answers without launching anything.

WHY THE LIST IS PARSED RATHER THAN WRITTEN DOWN. A hardcoded list is a list that
goes stale the day someone adds a third worker -- which is the failure this test
exists to prevent, one level up.
"""
import os
import re

import pytest

import _invocation

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
SERVE = os.path.join(ROOT, "serve.ps1")
SCRIPTS = os.path.join(ROOT, "qwen38-tuning", "scripts")


def _dispatched_workers():
    """Every `worker-*.ps1` serve.ps1 can choose, read out of serve.ps1 itself."""
    src = open(SERVE, encoding="utf-8", errors="replace").read()
    live = "\n".join(_invocation.live_lines(src))
    names = sorted(set(re.findall(r"'(worker-[\w.-]+\.ps1)'", live)))
    assert names, "no worker script literals found in serve.ps1 -- has the " \
                  "dispatch been restructured? Teach this test the new shape."
    return names


def test_serve_dispatches_to_the_profiles_this_file_guards():
    """A sanity check on the parse itself. If this shrinks to one, the loop
    below stops covering the profile that was broken."""
    assert len(_dispatched_workers()) >= 2, _dispatched_workers()


@pytest.mark.parametrize("name", _dispatched_workers())
@pytest.mark.parametrize("flag", ["--cache-ram", "--ctx-checkpoints"])
def test_the_profile_names_the_flag(name, flag):
    """Silence means llama.cpp's default, and neither default fits this machine.

    `--cache-ram` defaults to 8192 MiB (`common/common.h:632`) and
    `--ctx-checkpoints` to 32 (`common/arg.cpp:1695`). Naming a flag is not the
    same as naming it correctly -- that is the other file's job -- but a profile
    that never mentions it cannot have decided anything about it.
    """
    path = os.path.join(SCRIPTS, name)
    src = open(path, encoding="utf-8", errors="replace").read()
    live = "\n".join(_invocation.live_lines(src))
    assert flag in live, (
        "%s never names %s outside its comments, so it serves llama.cpp's "
        "default. See issue #70 for what that cost." % (name, flag))


LAUNCHERS = os.path.join(ROOT, "launchers")


def _hub_llama_profiles():
    """Every `serve-*.ps1` a hub launcher calls directly, bypassing serve.ps1.

    Flash-Next (2026-09-24) was launched this way and inherited both defaults:
    `logs/flash-next-128k-20260924-052307.log` holds one 105,795-token entry at
    5,918 MiB (31 checkpoints), so the auto-mode classifier's 30K conversation
    evicted it and the next main-loop turn re-prefilled 66,890 tokens (274 s).
    Only scripts that start `llama-server` count -- EXL3 has no such flags.
    """
    names = set()
    for bat in os.listdir(LAUNCHERS):
        if bat.endswith(".bat"):
            src = open(os.path.join(LAUNCHERS, bat), encoding="utf-8",
                       errors="replace").read()
            names.update(re.findall(r"(serve-[\w.-]+\.ps1)", src))
    out = []
    for name in sorted(names):
        path = os.path.join(SCRIPTS, name)
        if os.path.exists(path) and "llama-server" in open(
                path, encoding="utf-8", errors="replace").read():
            out.append(name)
    return out


def test_the_hub_launches_llama_profiles_this_file_guards():
    """Sanity check on the parse: gsq and flash-next are both hub entries."""
    assert {"serve-gsq.ps1", "serve-flash-next.ps1"} <= set(_hub_llama_profiles())


@pytest.mark.parametrize("name", _hub_llama_profiles())
@pytest.mark.parametrize("flag", ["--cache-ram", "--ctx-checkpoints"])
def test_the_hub_profile_names_the_flag(name, flag):
    test_the_profile_names_the_flag(name, flag)
