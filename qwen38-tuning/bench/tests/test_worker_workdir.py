"""The worker's working directory must be pinned explicitly, not by `cwd=`.

THE INCIDENT, 2026-08-23, reproduced deliberately and then found in the wild.

`edit_canary` cloned `openclink` into the scratch root, launched the worker with
`cwd=<clone>`, and told it to edit `README.md`. The transcript says it edited
`C:\\AI\\README.md` -- the LIVE repository. `git diff` inside the clone was empty,
so the run scored `diff_bytes=0`, which is indistinguishable from "the worker
decided to do nothing".

`git status` on the live tree confirmed it: `M README.md`, first line ending in
CANARY. Reverted.

THE CAUSE WAS ALREADY WRITTEN DOWN. `opencode_corpus.py:50-62`, dated
2026-08-21: OpenCode keeps a per-project server alive between invocations, `run`
attaches to whichever is already listening, and that server carries **the
project root it was first started with**. The docstring even records the same
symptom -- *"every answer landed in C:\\AI\\qwen38-tuning while the harness looked
in the task directory and recorded 'no file written' on work the model had done
correctly."*

`opencode_corpus.py` defends itself by killing the server once before the run.
**`real_task_bench.py` never did**, and it is the driver that produced five real
tasks with `diff_bytes: 0` and three clean exits. So that result is an
instrument fault, not a measurement of the model.

WHY THE TEST IS SHAPED LIKE THIS. `cwd=` is the thing that looked right and was
not, so a test that only checked `cwd` would have passed throughout the
incident. What decides where files land is what the driver puts on the command
line, so that is what is pinned here: an explicit `--dir` naming an absolute
path. A driver may additionally kill the server; it may not rely on `cwd` alone.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edit_canary


def test_argv_pins_the_directory_explicitly(tmp_path):
    argv = edit_canary.worker_argv("local/qwen38", "do the thing", tmp_path)
    assert "--dir" in argv, (
        "cwd= is not enough: OpenCode attaches to a server carrying the project "
        "root it was first started with, and edited the live repo through it")


def test_the_dir_value_is_the_workdir(tmp_path):
    argv = edit_canary.worker_argv("local/qwen38", "do the thing", tmp_path)
    assert argv[argv.index("--dir") + 1] == str(tmp_path)


def test_the_dir_value_is_absolute(tmp_path):
    """A relative path would be resolved against whatever root the server holds."""
    argv = edit_canary.worker_argv("local/qwen38", "x", tmp_path)
    assert os.path.isabs(argv[argv.index("--dir") + 1])


def test_the_prompt_survives_and_is_last(tmp_path):
    """A flag inserted in the wrong place turns the prompt into a flag value."""
    argv = edit_canary.worker_argv("local/qwen38", "edit README.md", tmp_path)
    assert argv[-1] == "edit README.md"


def test_the_model_is_still_passed(tmp_path):
    argv = edit_canary.worker_argv("local/qwen38", "x", tmp_path)
    assert argv[argv.index("-m") + 1] == "local/qwen38"


@pytest.mark.parametrize("bad", ["", None])
def test_an_empty_workdir_raises_rather_than_defaulting(bad):
    """Defaulting is how the live repo got edited. Refuse instead."""
    with pytest.raises(ValueError):
        edit_canary.worker_argv("local/qwen38", "x", bad)
