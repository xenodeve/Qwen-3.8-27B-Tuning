"""A row must say which cards produced it, including when it used the default.

WHY THE `env` COLUMN IS NOT ENOUGH.

`new_row` already records the arm's env dict, so a two-card arm that sets
CUDA_VISIBLE_DEVICES explicitly is self-describing. The arm that is NOT is the
control: `solo` takes the pin from `launch_env`'s default, so its env dict is
empty and the row says nothing about hardware at all.

That is the worse half. The control is the row every other row is measured
against, and "no field" reads exactly like "the usual card" right up until the
day the default changes -- which on this machine it did, on 2026-08-26, when a
second GPU was installed and index 0 stopped meaning what it used to.

So the column is resolved from the environment a launch would ACTUALLY get,
not from what the arm asked for.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import gpu_device  # noqa: E402
import dflash2_arena as A  # noqa: E402

OTHER = "GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4"  # the 4070 SUPER


def test_a_default_row_still_names_its_card():
    """The control arm sets no env. Its row must not be silent about hardware."""
    row = A.new_row(16384, "ngram-mod", 1, "real-code", [], {}, 15000)
    assert row["devices"] == gpu_device.SERVED_GPU_UUID


def test_a_two_card_row_names_both():
    both = OTHER + "," + gpu_device.SERVED_GPU_UUID
    row = A.new_row(16384, "both-layer", 1, "real-code", [],
                    {"CUDA_VISIBLE_DEVICES": both}, 27000)
    assert row["devices"] == both


def test_the_devices_column_matches_what_the_process_would_get():
    """The two must be derived from the same place. If `devices` were computed
    independently of `launch_env`, a row could name one card and the process
    could get another -- which is the original defect with a column bolted on.
    """
    for env in ({}, {"CUDA_VISIBLE_DEVICES": OTHER}):
        row = A.new_row(16384, "x", 1, "real-code", [], env, 0)
        assert row["devices"] == A.launch_env(env)["CUDA_VISIBLE_DEVICES"]


def test_the_dual_gpu_arm_set_exists_and_carries_a_single_card_control():
    """Issue #51 stage 2. An arm set with only two-card arms has nothing to
    compare against, and CORRECTIONS 23 already says an unpaired arm at depth is
    not to be trusted."""
    arms = A.ARM_SETS["dual-gpu"]
    envs = [A.arm_parts(a)[2] for a in arms]
    devs = [e.get("CUDA_VISIBLE_DEVICES", gpu_device.SERVED_GPU_UUID) for e in envs]
    assert any("," not in d for d in devs), \
        "no single-card control: nothing says what the second card changed"
    assert any("," in d for d in devs), \
        "no two-card arm: the set does not test the thing it is named for"


def test_every_dual_gpu_arm_holds_the_decoder_fixed():
    """The drafter is placed on a card too. A speculative arm split across two
    GPUs moves the decoder AND the placement at once, and the delta would have
    two causes -- which is what CORRECTIONS 26 and 28 both are."""
    specs = set()
    for arm in A.ARM_SETS["dual-gpu"]:
        extra = A.arm_parts(arm)[1]
        specs.add(tuple(extra[extra.index("--spec-type") + 1:][:1])
                  if "--spec-type" in extra else ())
    assert len(specs) == 1, f"the dual-gpu arms use different decoders: {specs}"
