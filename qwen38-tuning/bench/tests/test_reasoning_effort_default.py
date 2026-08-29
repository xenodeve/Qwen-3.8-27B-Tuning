r"""`medium` is the served default now, and every launcher must say so out loud.

THE DECISION, AND THE EVIDENCE UNDER IT.

Until 2026-08-24 every server this project launched ran at `xhigh` with an
unlimited thinking budget, because the chat template supplies both and nothing
overrode them -- not one of the five `worker-*.ps1` profiles, not
`dflash2_arena.py`. Nobody had chosen that; it was simply never set.

Artificial Analysis prices this model's three levels on the agentic axis --
the axis this project's metric sits on -- at **xhigh 51, medium 50, low 44**.
One point separates the setting everything used from the one below it; six
separate that from the bottom. On the general Intelligence Index the shape is
reversed (52 / 44 / 43), which is why "medium or low" is not one question.

`docs/results/05-runtime-flags.md` had also predicted, on 2026-08-18, that an
external review reports *xHigh taking 15 minutes where medium takes 3 for 90 % of
the result* -- and the four real-task runs of 2026-08-24 came in at 537.7 /
855.8 / 947.2 / 1,019.3 s with zero files changed four times out of four.

So the developer set `medium` as the default. This file makes that real rather
than aspirational.

WHY A SOURCE SCAN AND NOT JUST A UNIT TEST

The unit tests below prove the resolver works and that `server_argv` carries the
flag. They say nothing about the five PowerShell profiles that actually serve the
worker, and those are the files that were silently at `xhigh` for the whole life
of the project. `test_every_script_routes_its_exe.py` exists for the same reason
and found five scripts pointing at the wrong binary; this is that guard applied
to the flag that was never set.

WHAT THIS BREAKS, DELIBERATELY

Every figure recorded before this was taken at `xhigh`. Nothing measured after it
is comparable to them without saying so, which is why the row records `effort`
and why all nine results pages carry a banner naming the level they were taken
at. A default that changes silently is worse than one that is wrong.

WHAT IT CANNOT DO is prove the server honoured the flag. Only the boot log can,
via the rendered template line -- and a run that wants to trust its own number
should read that back, the way `run-arm-pair` reads `n_max`.
"""
import os
import re

from _invocation import from_source
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as arena
import provenance

PROFILES = os.path.join(os.path.dirname(BENCH), "scripts")


# ------------------------------------------------------------- the resolver

def test_the_default_is_medium(monkeypatch):
    monkeypatch.delenv(provenance.EFFORT_ENV_VAR, raising=False)
    assert provenance.resolve_effort() == "medium"


def test_the_environment_overrides_it(monkeypatch):
    monkeypatch.setenv(provenance.EFFORT_ENV_VAR, "xhigh")
    assert provenance.resolve_effort() == "xhigh"


def test_an_empty_variable_falls_through_to_medium(monkeypatch):
    monkeypatch.setenv(provenance.EFFORT_ENV_VAR, "")
    assert provenance.resolve_effort() == "medium"


def test_the_env_var_is_distinct_from_the_others():
    """Three separate levers -- binary, model, effort. Sharing one variable
    would make changing any of them silently change the others."""
    names = {provenance.ENV_VAR, provenance.TARGET_ENV_VAR,
             provenance.EFFORT_ENV_VAR}
    assert len(names) == 3, names


def test_default_is_not_the_template_default():
    """The whole point. If this ever reads 'default' or 'xhigh' the change has
    been undone and every subsequent measurement silently reverts."""
    assert provenance.DEFAULT_EFFORT == "medium"
    assert provenance.DEFAULT_EFFORT not in ("default", "xhigh")


# --------------------------------------------------------------- the argv

def test_server_argv_carries_the_flag():
    argv = arena.server_argv(16384, [])
    assert "--reasoning-effort" in argv
    assert argv[argv.index("--reasoning-effort") + 1] == arena.EFFORT


def test_the_flag_appears_once():
    """Twice would leave the value to llama.cpp's last-wins parsing, which is
    the hazard test_ubatch_arm_set.py was written for."""
    assert arena.server_argv(16384, []).count("--reasoning-effort") == 1


def test_an_arm_can_still_override_it():
    """`extra` is appended, so an arm that sets its own level wins on last-wins
    parsing -- that is how an effort sweep would be written."""
    argv = arena.server_argv(16384, ["--reasoning-effort", "low"])
    assert argv[len(argv) - 1] == "low"


# ------------------------------------------------------------- the profiles

def worker_profiles():
    return sorted(f for f in os.listdir(PROFILES)
                  if f.startswith("worker-") and f.endswith(".ps1"))


def test_there_are_worker_profiles_to_check():
    assert worker_profiles(), "found no worker-*.ps1; the scan would pass vacuously"


@pytest.mark.parametrize("name", worker_profiles())
def test_every_worker_profile_sets_the_effort(name):
    raw = open(os.path.join(PROFILES, name), encoding="utf-8",
               errors="replace").read()
    # COMMENTS ARE NOT CODE. This scanned the whole file until 2026-08-29 and
    # matched a sentence explaining the flag -- "--reasoning-effort medium.
    # Neither the file's reason..." -- capturing `medium.` with the full stop
    # and calling the profile misconfigured while it was set correctly two
    # dozen lines below. Seventh source-shape assertion this session to read
    # prose as configuration.
    text = os.linesep.join(l for l in raw.splitlines()
                           if not l.strip().startswith("#"))
    m = re.search(r"--reasoning-effort['\\\"]?\s*,?\s*['\\\"]?([^\s',\\\"]+)", text)
    assert m, (f"{name} sets no --reasoning-effort, so it serves at the "
               f"template's xhigh default -- the condition this change exists "
               f"to end")
    assert m.group(1).strip("`\"' ") == "medium", \
        f"{name} serves at {m.group(1)!r}, not the agreed default"


# ------------------------------------------------------------- the recording

SRC = open(os.path.join(BENCH, "dflash2_arena.py"), encoding="utf-8").read()


def test_the_row_records_the_effort():
    """Same lesson as exe, cuda_archs, env and target: a row that does not name
    a condition cannot be compared against one taken under a different one."""
    assert "effort=EFFORT" in SRC, "the row does not record the effort level"
