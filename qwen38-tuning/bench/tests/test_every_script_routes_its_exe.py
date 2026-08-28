"""No bench script may hardcode a llama-server path the operator cannot redirect.

THE HAZARD THIS GUARDS, and it is live rather than historical.

On 2026-08-24 six scripts in `bench/` named a server binary as a module
constant. Five of them pointed at `C:\\AI\\llama.cpp-cuda\\llama-server.exe`,
which -- like `llama.cpp-dflash2` -- carries SASS for `sm_89` only. The card is
`sm_120`. Running any of those five here JIT-compiles Ada PTX, takes 2.20x the
prefill time, and reports nothing: the boot log, the buffer sizes, the layer
split and `--version` are all identical to a correct run
(`docs/results/09-hardware.md`).

So a sweep launched from `model_arena.py` or `depth_sweep.py` today would fill a
JSONL with plausible slow numbers and no field anywhere saying which binary
produced them. That is the exact shape `CLAUDE.md`'s north star names.

Editing the constant is not the fix -- it makes the Ada figures unreproducible
and it is per-script, so the next new script starts the problem again. Routing
every one through `provenance.resolve_exe(default)` keeps each script's default
byte-identical while giving the operator one lever (`QWEN38_LLAMA_EXE`) and the
recorder one function (`cuda_archs`) that can tell the builds apart.

WHAT THIS FILE CANNOT DO is check a script that builds its path some other way --
by joining a directory constant, or reading a config. It scans for the assignment
form actually in use. A future script that invents a new form is not covered, and
the assertion message says so rather than implying full coverage.
"""
import os
import re
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

# The assignment form in use: a module-level EXE bound to a raw string literal.
HARDCODED = re.compile(r"^EXE\s*=\s*r?[\"']", re.MULTILINE)
ROUTED = re.compile(r"^EXE\s*=\s*resolve_exe\(", re.MULTILINE)


def scripts_defining_exe():
    out = []
    for name in sorted(os.listdir(BENCH)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(BENCH, name)
        text = open(path, encoding="utf-8", errors="replace").read()
        if re.search(r"^EXE\s*=", text, re.MULTILINE):
            out.append((name, text))
    return out


def test_some_script_defines_an_exe_at_all():
    """If this fails the scan is looking for the wrong thing, and every other
    assertion below would pass vacuously."""
    assert scripts_defining_exe(), "found no EXE assignment anywhere in bench/"


@pytest.mark.parametrize("name,text", scripts_defining_exe(),
                         ids=[n for n, _ in scripts_defining_exe()])
def test_exe_is_routed_through_the_resolver(name, text):
    assert not HARDCODED.search(text), (
        f"{name} binds EXE to a literal path. Wrap it: "
        f"EXE = resolve_exe(r\"...\") from provenance, so the operator can "
        f"redirect it and the row can record which build ran. "
        f"(This scan only sees `EXE = <literal>`; another form would slip past.)"
    )
    assert ROUTED.search(text), f"{name} defines EXE but not via resolve_exe()"


@pytest.mark.parametrize("name,text", scripts_defining_exe(),
                         ids=[n for n, _ in scripts_defining_exe()])
def test_the_default_is_still_a_real_path(name, text):
    """Routing must not have silently emptied a default. Every script keeps the
    binary it always used until the environment says otherwise.

    THE DEFAULT MAY BE A NAMED CONSTANT. This matched only the inline form
    `EXE = resolve_exe(r"...")` until 2026-08-29, and went red when
    `dflash2_arena` lifted its default to `DEFAULT_EXE` so a test could assert
    on the VALUE rather than on the text -- the default was still a real path
    the whole time. Fourth time in one session that a source-shape assertion
    called a refactor a regression.
    """
    m = re.search(r"^EXE\s*=\s*resolve_exe\(\s*(?:r?[\"']([^\"']+)[\"']"
                  r"|([A-Za-z_][A-Za-z0-9_]*))\s*\)", text, re.MULTILINE)
    assert m, f"{name}: resolve_exe() call has neither a literal nor a name"
    default = m.group(1)
    if default is None:                      # a named constant -- follow it
        name_ = m.group(2)
        d = re.search(r"^%s\s*=\s*r?[\"']([^\"']+)[\"']" % re.escape(name_),
                      text, re.MULTILINE)
        assert d, f"{name}: {name_} is passed to resolve_exe but never assigned"
        default = d.group(1)
    assert default.lower().endswith("llama-server.exe"), (
        f"{name}: default is {default!r}, which is not a llama-server binary")
