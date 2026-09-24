"""Flash-Next as a gsq_compare candidate (issue #92 quality/time screen, 2026-09-24).

The screen pairs Flash-Next with GSQ IQ3_S-MTP at the common 65,536 context the
27B rows used. The candidate must be the SERVED 128k profile (hub J) with only the
context changed -- an argv that differs from the served one is not the served
configuration (39-OPTIMISATION-GUIDE, "a command line is not a configuration").

Incidents this guards:
- the 27B runners assert `layers == [66, 0]`; Flash-Next has 48 layers + output
  (49/49) and an MTP head (50/50), so a 27B check would reject a healthy boot;
- the served profile sets LLAMA_MOE_CACHE_MAX_TOKENS=4 (expert cache on verify
  batches) -- a candidate without it measures a different binary behaviour.
"""
import os
import shutil
import subprocess
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)
import gsq_compare as gsq  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(BENCH))
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "serve-flash-next.ps1")
PWSH = shutil.which("pwsh") or shutil.which("powershell")

# Flags whose VALUE legitimately differs between the served profile and a bench
# boot: the window under test, the listener, and the console log copy.
VARIANT = {"-c", "--host", "--port", "--log-file", "--log-colors"}


def _pairs(argv):
    """argv (without the exe) -> list of (flag, value-or-None), order kept."""
    out, i = [], 0
    while i < len(argv):
        flag = argv[i]
        if i + 1 < len(argv) and not argv[i + 1].startswith("--") and not (
                argv[i + 1].startswith("-") and len(argv[i + 1]) <= 6 and argv[i + 1][1:].isalpha()):
            out.append((flag, argv[i + 1])); i += 2
        else:
            out.append((flag, None)); i += 1
    return out


def _profile_argv():
    result = subprocess.run([PWSH, "-NoProfile", "-File", PROFILE, "-Profile", "128k", "-WhatIf"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    text = result.stdout.split("WhatIf: would run", 1)[1]
    return text.split()


def test_flash_next_is_a_registered_artifact():
    assert "flash_next" in gsq.EXPECTED_DIGESTS
    meta = gsq.artifact_metadata("flash_next")
    assert meta["quant"] == "Q2_0"


def test_flash_next_argv_sets_the_requested_context_and_port():
    argv = gsq.flash_next_argv(65536, 18080)
    assert argv[argv.index("-c") + 1] == "65536"
    assert argv[argv.index("--port") + 1] == "18080"


def test_flash_next_env_carries_the_verify_batch_cache_cap():
    assert gsq.FLASH_NEXT_ENV == {"LLAMA_MOE_CACHE_MAX_TOKENS": "4"}


def test_flash_next_expected_layers_are_its_own_not_the_27b_count():
    assert gsq.FLASH_NEXT_LAYERS == ["49/49", "50/50"]


@pytest.mark.skipif(PWSH is None, reason="PowerShell is required for the serving profile")
def test_flash_next_argv_is_the_served_128k_profile_but_for_the_window():
    served = _profile_argv()
    ours = gsq.flash_next_argv(131072, 8080)
    assert os.path.normcase(served[0]) == os.path.normcase(ours[0]), (served[0], ours[0])
    strip = lambda argv: [(f, v) for f, v in _pairs(argv[1:]) if f not in VARIANT]
    norm = lambda pairs: [(f, os.path.normcase(os.path.normpath(v)) if v and ("\\" in v or "/" in v) else v)
                          for f, v in pairs]
    assert norm(strip(ours)) == norm(strip(served))
