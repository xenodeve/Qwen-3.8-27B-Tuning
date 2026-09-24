"""The GSQ launcher must use the 262K profile and capacity-aware tensor split."""
import os
import re
import shutil
import subprocess

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "serve-gsq.ps1")
PWSH = shutil.which("pwsh") or shutil.which("powershell")


@pytest.mark.skipif(PWSH is None, reason="PowerShell is required for the serving profile")
def test_gsq_whatif_uses_262k_and_gives_the_16gb_card_more_tensor_budget():
    assert os.path.exists(PROFILE), PROFILE
    result = subprocess.run(
        [PWSH, "-NoProfile", "-File", PROFILE, "-WhatIf"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "-c 262144" in output, output
    assert "-sm tensor" in output, output
    assert "-lv 4" in output, output
    assert "-lv 5" not in output, output
    match = re.search(r"-ts\s+(\d+),(\d+)", output)
    assert match, output
    smaller, larger = (int(value) for value in match.groups())
    assert smaller < larger, output


def test_gsq_cmd_forwards_the_262k_profile():
    cmd = os.path.join(ROOT, "qwen38-tuning", "scripts", "serve-gsq.cmd")
    text = open(cmd, encoding="ascii").read()
    assert "serve-gsq.ps1" in text
    assert "262144" in text


def test_gsq_profile_uses_the_same_normal_verbosity_as_other_served_profiles():
    text = open(PROFILE, encoding="ascii").read()
    assert "'-lv', '4'" in text
    assert "'-lv', '5'" not in text




def test_gsq_profile_uses_the_late_system_template_for_claude_code_history():
    text = open(PROFILE, encoding="ascii").read()
    assert "qwen38-late-system.jinja" in text
    assert "'--chat-template-file'" in text


def test_gsq_profile_carries_the_llama_claude_compatibility_flags():
    text = open(PROFILE, encoding="ascii").read()
    assert "'--fit', 'off'" in text
    assert "'--fit-target'" not in text
    assert "'--cache-ram', '24576'" in text
    assert "'--ctx-checkpoints', '8'" in text
    assert "'--sse-ping-interval'" in text
    assert "'--min-p', '0.0'" in text
    assert "'--repeat-penalty', '1.05'" in text
    assert "'--log-colors', 'on'" in text
