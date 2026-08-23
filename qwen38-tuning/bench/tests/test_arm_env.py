"""An arm may set environment variables, and the row must record which.

WHY THIS EXISTS.

Some of what llama.cpp's CUDA backend can be told is not a command-line flag.
`grep getenv ggml/src/ggml-cuda/` at commit 1deefcca3 finds twelve knobs, and at
least one of them is an optimisation that is OFF unless asked for:

    static bool enable_graph_optimization = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;      // ggml-cuda.cu:4330
    }();

It also requires exactly one CUDA device, which is this machine. This project
has never set it.

The arena could not express that. `ARMS` entries are `(label, argv_extra)` and
`start()` launches with the parent environment, so the only way to test an env
knob was to export it and run the whole sweep again -- a comparison ACROSS
BOOTS, which `CLAUDE.md` forbids outright because the boot-to-boot spread is
real and its cause is unknown. Pairing inside a round is the only honest way to
measure one, so the arm has to carry it.

THE SECOND HALF IS THE RECORDING, AND IT IS THE HALF THAT BITES.

An env-carrying arm produces a row that looks exactly like a row without it:
same argv, same flags, same buffer sizes. `docs/reports/CORRECTIONS.md` 28 is
this project retracting a published comparison for precisely that reason -- two
correctly-measured numbers made false by nothing recording what differed between
them. So a row from an env arm must carry the env, and a row from a plain arm
must carry an empty mapping rather than nothing at all: absent and empty must
not be the same value, or "this arm set no variables" becomes indistinguishable
from "this row predates the feature".

WHAT THIS FILE CANNOT DO is prove llama.cpp honoured the variable. Nothing in
argv or the boot banner echoes `GGML_CUDA_GRAPH_OPT`, so the only evidence would
be a behavioural difference -- which is the thing being measured, and cannot be
its own control. The arm set's comment has to say so.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena


# ------------------------------------------------------------- normalisation

def test_a_two_tuple_arm_still_works_and_means_no_env():
    label, extra, env = arena.arm_parts(("none", []))
    assert label == "none"
    assert extra == []
    assert env == {}


def test_a_three_tuple_arm_carries_its_env():
    label, extra, env = arena.arm_parts(
        ("graph-opt", ["--spec-type", "ngram-mod"], {"GGML_CUDA_GRAPH_OPT": "1"}))
    assert label == "graph-opt"
    assert extra == ["--spec-type", "ngram-mod"]
    assert env == {"GGML_CUDA_GRAPH_OPT": "1"}


def test_every_arm_in_every_set_normalises():
    """A set whose entries do not unpack would fail at run time, hours in, after
    the GPU has already been held. Fail here instead."""
    for name, arms in arena.ARM_SETS.items():
        for arm in arms:
            label, extra, env = arena.arm_parts(arm)
            assert isinstance(label, str) and label, f"{name}: bad label in {arm!r}"
            assert isinstance(extra, list), f"{name}/{label}: extra is not a list"
            assert isinstance(env, dict), f"{name}/{label}: env is not a dict"
            assert all(isinstance(v, str) for v in env.values()), \
                f"{name}/{label}: env values must be strings, os.environ takes no ints"


def test_arm_parts_rejects_a_shape_it_cannot_read():
    """Silently treating a malformed arm as (label, []) would run the control
    config under the arm's name and report it as the arm's result."""
    with pytest.raises((ValueError, TypeError)):
        arena.arm_parts(("just-a-label",))


# ------------------------------------------------------------ the env reaches

def test_launch_env_layers_the_arm_over_the_process(monkeypatch):
    monkeypatch.setenv("SOME_EXISTING_VAR", "keep-me")
    env = arena.launch_env({"GGML_CUDA_GRAPH_OPT": "1"})
    assert env["GGML_CUDA_GRAPH_OPT"] == "1"
    assert env["SOME_EXISTING_VAR"] == "keep-me", \
        "the arm's env must ADD to the process environment, not replace it -- " \
        "llama-server needs PATH and CUDA_PATH to start at all"


def test_launch_env_with_nothing_is_still_a_full_environment():
    env = arena.launch_env({})
    assert env == dict(os.environ)


def test_the_graph_opt_arm_set_exists_and_pairs_a_control():
    """A single-arm set cannot be paired within a round, and an unpaired arm at
    this depth is exactly what CORRECTIONS 23 says not to trust."""
    arms = arena.ARM_SETS["graph-opt"]
    envs = [arena.arm_parts(a)[2] for a in arms]
    assert any(e.get("GGML_CUDA_GRAPH_OPT") == "1" for e in envs), \
        "the set does not actually enable the thing it is named for"
    assert any(e == {} for e in envs), \
        "no control arm: nothing in the round to pair the treated arm against"


def test_the_graph_opt_set_varies_only_the_environment():
    """If the arms also differ in argv, a difference in rate has two causes and
    the sweep cannot attribute it -- the shape of CORRECTIONS 26 and 28."""
    extras = {tuple(arena.arm_parts(a)[1]) for a in arena.ARM_SETS["graph-opt"]}
    assert len(extras) == 1, f"arms differ in argv as well as env: {extras}"
