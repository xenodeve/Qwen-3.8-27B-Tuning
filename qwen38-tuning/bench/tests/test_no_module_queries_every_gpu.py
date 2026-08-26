"""One chokepoint may ask the driver about GPUs. Everything else goes through it.

WHY A CHOKEPOINT RATHER THAN A PATTERN PER CALL SITE.

On 2026-08-26 seven modules in `bench/` and four scripts in `scripts/` each
built their own `nvidia-smi --query-gpu=...` command line. When a second card
appeared, all eleven became wrong at once, and fixing them one at a time leaves
the twelfth -- written next week -- to start the problem again. That is the same
argument `test_every_script_routes_its_exe.py` makes for `resolve_exe`, and it
held there.

WHY THE SCAN LOOKS FOR `--query-gpu` AND NOT `nvidia-smi`.

Five earlier tests in this suite failed on the SHAPE of a file rather than on
behaviour, and one of them went red because a COMMENT mentioned the string it
forbade. `nvidia-smi` appears in prose here -- `hardware_baseline.py` explains in
its docstring that llama.cpp's free-VRAM figure is not nvidia-smi's, and
`worker-5060ti.ps1` says the same in a comment. Both are correct and neither is
a call.

`--query-gpu` is not prose. It appears only in an actual invocation, which is
what makes it a property this scan can hold without going blind or crying wolf.

WHAT THIS FILE CANNOT DO: catch a call assembled at runtime from pieces, or one
that reads the driver some other way (NVML, a DLL, WMI). It scans for the flag
in use. A future module that invents a new form is not covered, and the message
below says so rather than implying coverage it does not have.
"""
import os
import re
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BENCH)
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, BENCH)

QUERY = re.compile(r"--query-(gpu|compute-apps)")

# The chokepoints. Everything else routes through one of these.
PY_CHOKEPOINT = "gpu_device.py"
PS_CHOKEPOINT = "Get-GpuVram.ps1"

# Deliberate exception, with its reason. `collect-env.ps1` writes the machine
# INVENTORY -- listing every card is the point of it, and narrowing it to one
# would delete the evidence that a second card is installed at all.
PS_INVENTORY_ALLOWED = {"collect-env.ps1"}


def _files(folder, suffix):
    return [f for f in sorted(os.listdir(folder)) if f.endswith(suffix)]


def _text(path):
    return open(path, encoding="utf-8", errors="replace").read()


# Parametrised on the NAME only. Passing the source text as a parameter puts
# the whole file into the test id, which makes a failure report unreadable and
# hid a real error behind 180 KB of output the first time this ran.
def python_modules():
    return _files(BENCH, ".py")


def powershell_scripts():
    return _files(SCRIPTS, ".ps1")


def test_the_scan_finds_the_flag_somewhere():
    """Without this, a typo in QUERY makes every test below pass vacuously --
    the exact way `test_it_guards_the_port_before_launching` stayed green for
    days while checking nothing."""
    hits = ([n for n in python_modules() if QUERY.search(_text(os.path.join(BENCH, n)))]
            + [n for n in powershell_scripts() if QUERY.search(_text(os.path.join(SCRIPTS, n)))])
    assert hits, "found no --query-gpu anywhere; the scan is looking for the wrong flag"


@pytest.mark.parametrize("name", python_modules())
def test_no_bench_module_queries_the_driver_directly(name):
    if name == PY_CHOKEPOINT:
        return
    text = _text(os.path.join(BENCH, name))
    assert not QUERY.search(text), (
        f"{name} builds its own nvidia-smi query. With more than one card "
        f"installed that reading does not say which card it came from. "
        f"Import {PY_CHOKEPOINT} instead. (This scan sees the --query-gpu flag "
        f"only; a call assembled at runtime would not appear here.)")


@pytest.mark.parametrize("name", powershell_scripts())
def test_no_script_queries_the_driver_directly(name):
    if name in (PS_CHOKEPOINT,) or name in PS_INVENTORY_ALLOWED:
        return
    text = _text(os.path.join(SCRIPTS, name))
    assert not QUERY.search(text), (
        f"{name} builds its own nvidia-smi query. In PowerShell this fails "
        f"SILENTLY with two cards -- `-split` returns four elements and [0]/[1] "
        f"become the other card's numbers. Dot-source {PS_CHOKEPOINT} instead.")


def test_the_python_chokepoint_exists():
    assert os.path.exists(os.path.join(BENCH, PY_CHOKEPOINT))


def test_the_powershell_chokepoint_exists():
    assert os.path.exists(os.path.join(SCRIPTS, PS_CHOKEPOINT))
