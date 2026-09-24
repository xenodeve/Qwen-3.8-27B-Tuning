"""Both Flash-Next profiles (hub I = 262k, J = 128k) must carry what Claude Code
needs from a llama-server. Each flag below names the incident it guards.

- `--chat-template-file qwen38-late-system.jinja`: Claude Code appends its
  SessionStart hook output as a late system message; the stock template raises
  and every request is HTTP 500 (2026-09-24, `templates/README.md`).
- `--sse-ping-interval 5`: llama.cpp's 30 s default leaves Claude Code silent
  through a long prefill and it shows "Waiting for API response ... check your
  network" (`worker-q4-dual.ps1`, 05-OPERATING-GUIDE section 3). Flash-Next
  prefills at ~200 tok/s, so a 40K turn is minutes of silence.
- `--reasoning-effort medium`: the template default is xhigh; report 35 made
  medium the served default everywhere else. On Flash-Next one xhigh turn
  thought 9,300 tokens (8 min) and the auto-mode classifier thought at xhigh
  on every tool call (flash-next-128k-20260924-052307.log).
- `--cache-ram 8192 --ctx-checkpoints 8`: CORRECTIONS 46, issue #70.
"""
import os
import shutil
import subprocess

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "serve-flash-next.ps1")
PWSH = shutil.which("pwsh") or shutil.which("powershell")

REQUIRED = [
    "--chat-template-file",
    "qwen38-late-system.jinja",
    "--sse-ping-interval 5",
    "--reasoning-effort medium",
    "--cache-ram 8192",
    "--ctx-checkpoints 8",
    # ngram-mod 24/16/64 (q-ng64, spec-mtp.csv ng-*, 2026-09-24): over 12/16/32
    # on the 128k duo, copy +23 % (t0) / +13 % (t1), fresh text +26 % / +6 %.
    # 12/16/64 won copy but lost fresh text (acceptance 0.42-0.54).
    "--spec-ngram-mod-n-match 24",
    "--spec-ngram-mod-n-min 16",
    "--spec-ngram-mod-n-max 64",
]

SPEC_TYPE = {"262k": "--spec-type ngram-mod ", "128k": "--spec-type draft-mtp,ngram-mod "}


def _whatif(profile):
    result = subprocess.run(
        [PWSH, "-NoProfile", "-File", PROFILE, "-Profile", profile, "-WhatIf"],
        capture_output=True, text=True, timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    # WhatIf wraps the command line at the console width; rejoin it.
    return " ".join(output.split()).replace("-- ", "--")


@pytest.mark.skipif(PWSH is None, reason="PowerShell is required for the serving profile")
@pytest.mark.parametrize("profile", ["262k", "128k"])
def test_flash_next_profile_carries_the_claude_code_flags(profile):
    argv = _whatif(profile)
    missing = [flag for flag in REQUIRED if flag not in argv]
    assert not missing, (profile, missing, argv)


@pytest.mark.skipif(PWSH is None, reason="PowerShell is required for the serving profile")
@pytest.mark.parametrize("profile", ["262k", "128k"])
def test_flash_next_profile_speculates_without_mtp_only_at_262k(profile):
    """262k + MTP over-committed GPU0 (prefill /5, 2026-09-24), so 262k runs
    ngram-mod alone: it needs no draft model and no VRAM."""
    argv = _whatif(profile) + " "
    assert SPEC_TYPE[profile] in argv, (profile, argv)
    assert ("-md " in argv) == (profile == "128k"), (profile, argv)


@pytest.mark.skipif(PWSH is None, reason="PowerShell is required for the serving profile")
def test_the_128k_draft_lives_on_the_5060ti_not_the_display_card():
    """The 4070 SUPER drives the desktop and its free VRAM moves by ~1 GB: with
    the draft there (-ncmoe 32) CUDA0 had 1,990-2,124 MiB free before a draft
    that needs ~2,530, and it spilled silently (q-ncmoe-hr, 2026-09-24). Draft
    and output.weight on the 5060 Ti with -ncmoe 36: llama holds 7,669 MiB on
    CUDA0 instead of 10,609, and it booted with 2,852 MiB free on CUDA1 for the
    draft. -ncmoe 34 OOMed CUDA1 at boot (1,674 free)."""
    argv = _whatif("128k") + " "
    assert "-devd CUDA1 " in argv, argv
    assert "-ncmoe 36 " in argv, argv
    assert "output.weight=CUDA0" not in argv, argv
