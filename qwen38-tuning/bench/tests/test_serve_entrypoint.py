r"""One entry point at the root, and it must delegate rather than duplicate.

WHY IT EXISTS (issue #48). `qwen38-tuning/scripts/` holds **58 `.ps1` files** --
six `worker-*.ps1` plus `production-*`, `serve-*`, `probe-*` and one-offs from
earlier sessions, several of which serve artifacts that are no longer the default
at windows that are no longer the answer. Nothing in the tree says which is
current, so choosing correctly means reading `docs/reports/35` and the ledger
first. `serve.ps1` is the answer to "just start the good one".

WHY THE DUPLICATION CHECK IS THE POINT OF THIS FILE.

`worker-q2kxl-mtp.ps1` **is** the configuration. A launcher that copies its flags
becomes a second source of truth and drifts the first time one of them changes --
and the drift is silent, because both files run and both look right. This repo
already carries that failure: `CORRECTIONS.md` 20 is the incident where the
benchmark built its prompt from source being edited between runs and the same arm
read **78.9 against 105.4 tok/s**.

So the assertions below are mostly ABSENCES. A model path, a `-c`, a
`--spec-type`, a `-ctk`/`-ctv` or a `--reasoning-effort` in `serve.ps1` means the
config now lives in two places, and the test fails whether or not the two agree
today.

WHAT IS NOT ASSERTED. That the launcher works -- a PowerShell script that boots a
15 GB model cannot be exercised from pytest, and pretending otherwise would be a
test that passes while the thing is broken. What can be checked from here is the
SHAPE: that it delegates, that it guards the port, that it reads the layer split
back rather than assuming it, and that it does not present an open question as
settled. The boot itself is verified by running it.
"""
import os
import re
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
SERVE = os.path.join(ROOT, "serve.ps1")
PROFILE = os.path.join(ROOT, "qwen38-tuning", "scripts", "worker-q2kxl-mtp.ps1")


def text():
    with open(SERVE, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def test_the_entry_point_exists_at_the_root():
    """Anywhere else and it is the fifty-ninth script nobody can find."""
    assert os.path.isfile(SERVE), SERVE


def test_the_profile_it_delegates_to_exists():
    """A launcher pointing at a renamed profile fails at the worst moment."""
    assert os.path.isfile(PROFILE), PROFILE
    assert "worker-q2kxl-mtp.ps1" in text()


@pytest.mark.parametrize("flag", ["-c ", "--spec-type", "-ctk", "-ctv",
                                  "--reasoning-effort", "-ngl", "--fit-target",
                                  "-lv "])
def test_it_does_not_re_declare_a_serving_flag(flag):
    """The configuration lives in the profile. Copying a flag here makes two
    sources of truth that drift silently, because both files still run.

    COMMENTS ARE NOT CODE. This read the whole file until 2026-08-29 and went
    red on a comment explaining why `--fit-target 768` had been REMOVED -- the
    ninth assertion in this repo to read prose as configuration, and the one it
    flagged was a sentence arguing for the very rule it enforces.
    `_invocation.live_lines` exists for this and is what to reach for next
    time.
    """
    from _invocation import live_lines
    live = os.linesep.join(live_lines(text()))
    assert flag not in live, (
        "serve.ps1 declares %r; it must resolve and invoke the profile instead"
        % flag)


def test_it_does_not_carry_a_model_path():
    assert ".gguf" not in text()
    assert "huggingface" not in text().lower()


def test_it_guards_the_port_before_launching():
    """Two orchestrators cannot share 8080. An armed queue once killed a running
    corpus and the summary still printed a plausible number."""
    t = text()
    assert "8080" in t
    # Invoke-RestMethod is what the guard actually uses. The first version of
    # this list omitted it and passed anyway, matching Get-NetTCPConnection in
    # the STATUS block -- green for the wrong reason until that block moved to
    # its own file, which is the only thing that ever revealed it.
    assert re.search(r"Invoke-RestMethod|Invoke-WebRequest|TcpClient|Get-NetTCPConnection", t), (
        "nothing in serve.ps1 checks whether the port is already answering")


def test_it_reads_the_layer_split_back_rather_than_assuming_it():
    """`--fit` SPILLS rather than refusing: a projection at ctx 163,840 read as
    success in every field except the layer count, which said 64/66."""
    assert "offloaded" in text(), "serve.ps1 does not verify residency after boot"


def test_it_asks_the_profile_for_the_verbosity_it_needs():
    """The tensor-assignment lines only exist at verbosity 5, and the served
    default is 3. The first version of this launcher warned that residency was
    UNVERIFIED on every boot -- it was looking for a line the served profile
    never writes. The knob belongs to the profile; the launcher asks for it by
    PARAMETER, so the flag itself still lives in one file."""
    assert "-Verbosity" in text(), (
        "serve.ps1 does not request the verbosity that makes residency readable")
    profile = open(PROFILE, encoding="utf-8", errors="replace").read()
    assert "$Verbosity" in profile, "the profile does not accept -Verbosity"
    assert "[int]$Verbosity = 3" in profile, (
        "the profile's DEFAULT verbosity moved; every served row was measured "
        "at 3 and changing the default silently changes what they mean")


def test_it_reads_the_stream_llama_cpp_actually_writes_to():
    """llama.cpp logs to STDERR. An early version searched stdout, found an
    empty file, and reported residency unverified while the evidence sat in the
    other stream.

    It used to assert -RedirectStandardError, which belonged to the detached
    design. In the foreground there is no second process to redirect: `2>&1`
    merges stderr into the pipeline this script reads line by line. Same
    property, different mechanism -- and asserting the old mechanism would have
    gone red for the right reason and the wrong cause."""
    t = text()
    # Rewritten twice, because it kept asserting a MECHANISM. First
    # -RedirectStandardError (detached design), then `2>&1` (piped design).
    # The property that survived both: the layer-assignment lines must reach the
    # residency check by some route. They now arrive through llama.cpp's own
    # --log-file, which it writes IN ADDITION to the console.
    assert "LogFile" in t, (
        "nothing routes llama.cpp's output anywhere the residency check can read")


def test_it_names_the_open_question_instead_of_presenting_the_config_as_settled():
    """draft-mtp is worth +15.6 % to REMOVE at 147,456 when the budget is forced,
    and +127 % to KEEP in the one natural round at 98,304. A launcher that says
    nothing is overstating what this project knows."""
    t = text()
    assert "draft-mtp" in t, "the open question is not mentioned"
    assert re.search(r"#4[47]\b", t), (
        "the open question is mentioned without the issue that tracks it")


def test_it_can_show_the_command_without_taking_the_gpu():
    assert "WhatIf" in text(), (
        "no way to read the resolved config without booting a 15 GB model")
