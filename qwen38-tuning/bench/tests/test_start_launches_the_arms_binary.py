r"""`start()` must LAUNCH the binary the arm asked for, not just record it.

THE INCIDENT THIS FILE EXISTS FOR -- 2026-08-30

`test_build_ab_arm_set.py` was written to stop exactly one failure: a build A/B
whose rows all name one build. It asserts that `server_argv` returns the arm's
binary and that `new_row` records it, and both were true. The sweep still ran
every arm on the module default, because the seam it actually goes through is
neither of those:

    args = server_argv(ctx, extra)                      # env NOT passed
    p = subprocess.Popen(args, ..., env=launch_env(env or {}))

`server_argv` with no `env` resolves `arm_exe(None)` to `EXE`. So the process
ran `llama.cpp-blackwell` while every row said `llama.cpp-mirror`.

It was found by reading the command line of a leftover process, not by any
test:

    C:\AI\llama.cpp-blackwell\llama-server.exe ... --alias Qwen3.8-27B-arena

`--alias Qwen3.8-27B-arena` is the arena's own; `llama.cpp-blackwell` is not
what that arm pinned.

**The tell was in the results and was explained away.** Every arm reported
byte-identical draft counters across the two "builds" -- acceptance 46.3,
decline 98.9 %, mean length 15.9, the same digits on both sides -- and that was
written up as normal for greedy decoding on one prompt. It is also what one
binary measured twice looks like, and that is what it was.

So the assertion here is not "does `server_argv` honour env" -- another file
already covers that, and covering it was not enough. It is **"does the process
that actually gets spawned use the arm's binary"**, checked by capturing the
argv handed to `subprocess.Popen`.

CORRECTIONS 34 is the same shape and was supposed to be the last time: a column
recording a module default while something else ran. This is that entry's third
appearance, one seam further down, and the general form is now explicit -- **a
test that asserts on the function that BUILDS a command has not tested the
function that RUNS it.**
"""
import os
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as arena  # noqa: E402

# A pin that cannot resolve. Used only by the refusal test.
MISSING = r"C:\nowhere-at-all\llama-server.exe"


class _DeadProc:
    """A process that has already exited, so `start()` returns immediately
    instead of polling /health for four minutes."""

    returncode = 1

    def poll(self):
        return 1

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 1

    def kill(self):
        pass


@pytest.fixture
def pinned(tmp_path):
    """A binary that EXISTS. `start()` verifies the pin before spawning, so a
    path that is not there is refused -- correctly -- and would test the
    refusal rather than the launch."""
    exe = tmp_path / "pinned-llama-server.exe"
    exe.write_bytes(b"MZ")
    return str(exe)


@pytest.fixture
def spawned(monkeypatch, tmp_path):
    """Capture the argv `start()` hands to Popen, without launching anything.

    Only the llama-server spawn is intercepted. `subprocess.run` is built on
    Popen and is used elsewhere in this module -- swallowing every call breaks
    those in a way that looks like this test failing.
    """
    seen = {}
    real_popen = arena.subprocess.Popen

    def fake_popen(args, **kwargs):
        argv = list(args)
        if argv and "llama-server" in str(argv[0]).lower():
            seen["args"] = argv
            seen["env"] = kwargs.get("env")
            return _DeadProc()
        return real_popen(args, **kwargs)

    monkeypatch.setattr(arena.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(arena, "stop_server", lambda *a, **k: None)
    monkeypatch.setattr(arena, "free_for_env", lambda *a, **k: 12345)
    monkeypatch.setattr(arena, "ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return seen


def test_start_launches_the_binary_the_arm_pinned(spawned, pinned):
    """THE REGRESSION. This is the assertion whose absence voided a sweep."""
    arena.start(16384, [], "t", env={arena.ENV_VAR: pinned})
    assert spawned["args"][0] == pinned, (
        "start() spawned %s while the arm pinned %s -- every row will record "
        "the pin and every number will come from the default"
        % (spawned["args"][0], pinned))


def test_start_still_uses_the_default_when_the_arm_pins_nothing(spawned):
    arena.start(16384, [], "t", env={})
    assert spawned["args"][0] == arena.EXE


def test_start_with_no_env_at_all_uses_the_default(spawned):
    arena.start(16384, [], "t")
    assert spawned["args"][0] == arena.EXE


def test_an_unrelated_arm_env_does_not_move_the_binary(spawned):
    arena.start(16384, [], "t", env={"GGML_CUDA_GRAPH_OPT": "1"})
    assert spawned["args"][0] == arena.EXE


def test_the_launched_argv_and_the_recorded_exe_agree(spawned, pinned):
    """The two must not be able to disagree. They did, and the row was the one
    anybody would read."""
    env = {arena.ENV_VAR: pinned}
    arena.start(16384, [], "t", env=env)
    row = arena.new_row(16384, "arm", 1, "synthetic", [], env, 1000)
    assert spawned["args"][0] == row["exe"], (spawned["args"][0], row["exe"])


def test_the_arm_env_still_reaches_the_child_process(spawned, pinned):
    """Fixing the argv must not drop the environment: an arm can carry the CUDA
    loader path, and Studio's binary finds no CUDA device without it."""
    arena.start(16384, [], "t", env={arena.ENV_VAR: pinned, "PATH": "X;Y"})
    assert spawned["env"]["PATH"] == "X;Y"


def test_the_arm_flags_are_still_appended(spawned, pinned):
    arena.start(16384, ["-sm", "layer"], "t", env={arena.ENV_VAR: pinned})
    assert spawned["args"][-2:] == ["-sm", "layer"], spawned["args"][-4:]


# ------------------------------------------------------------ guarding the guard

SRC = open(os.path.join(BENCH, "dflash2_arena.py"), encoding="utf-8").read()


def test_the_source_no_longer_builds_the_argv_without_the_env():
    """CORRECTIONS 34's own test asserted a literal and stayed green through
    the fault. This one names the exact expression that caused it."""
    assert "args = server_argv(ctx, extra)\n" not in SRC, (
        "start() builds the command line without the arm's env again; the "
        "process will run the module default while the row names the pin")


def test_a_pin_at_a_path_that_is_not_there_refuses_to_launch(spawned):
    """The pin must fail loudly. Falling back to the default is how the whole
    A/B ran on one binary in the first place, and it produced a full set of
    believable rows."""
    with pytest.raises(Exception):
        arena.start(16384, [], "t", env={arena.ENV_VAR: MISSING})
    assert "args" not in spawned, "it spawned something anyway"
