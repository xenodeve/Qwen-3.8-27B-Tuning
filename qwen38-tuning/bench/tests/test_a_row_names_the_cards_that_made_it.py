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


def test_the_nospec_set_removes_the_confound_the_dual_gpu_set_has():
    """Why a second set exists.

    At ctx 16,384 the `dual-gpu` arms decoded DIFFERENT TEXT -- ngram-mod
    accepted 93.3 % on one card and 58.5 % on two. That is not a sampling
    artifact to be averaged away: `SAMPLER` is already greedy (temperature 0,
    top_k 1, seed 42), and the text still differs because splitting a model
    across devices changes the reduction order, so the logits differ. On a
    split model you CANNOT decode the same tokens as on one card.

    With speculation off, every token costs exactly one forward pass whatever
    the token is, so the rate stops depending on the text -- the same property
    that makes prefill comparable.
    """
    arms = A.ARM_SETS["dual-gpu-nospec"]
    for arm in arms:
        extra = A.arm_parts(arm)[1]
        assert "--spec-type" not in extra, (
            f"{A.arm_parts(arm)[0]} still speculates; its decode rate would "
            f"depend on what it generated, which is the confound this set exists "
            f"to remove")
    devs = [A.arm_parts(a)[2].get("CUDA_VISIBLE_DEVICES", "") for a in arms]
    assert any("," in d for d in devs) and any("," not in d for d in devs), \
        "the set needs both a one-card control and a two-card arm"


def test_free_vram_is_summed_over_the_cards_the_arm_actually_uses():
    """`start()` recorded `vram()[1]` -- free memory on the SERVED card -- and
    wrote it into every row including the two-card ones. At ctx 16,384 that
    produced `free_before: 15983` on an arm running across 28 GB: a believable
    number describing hardware the arm was only half using.

    It is not a throughput field, which is why it survived a whole sweep
    unnoticed. It is still the shape CLAUDE.md's north star names.
    """
    one = A.free_for_env({"CUDA_VISIBLE_DEVICES": gpu_device.SERVED_GPU_UUID})
    both = A.free_for_env({"CUDA_VISIBLE_DEVICES":
                           OTHER + "," + gpu_device.SERVED_GPU_UUID})
    assert both > one, (
        f"two cards report no more free VRAM than one ({both} vs {one}); "
        f"the sum is not being taken")


def test_free_vram_with_no_arm_env_describes_the_served_card():
    assert A.free_for_env({}) == gpu_device.free_vram()


def test_the_split_set_compares_modes_and_ratios_without_speculation():
    """Issue #52 stage 1.

    Every arm runs on BOTH cards and differs only in how the model is divided
    between them. Speculation stays off for the reason CORRECTIONS 32 gives:
    on a split model the generated text changes with the split, so a
    speculative rate measures the text as well as the arm.
    """
    arms = A.ARM_SETS["dual-split"]
    both = OTHER + "," + gpu_device.SERVED_GPU_UUID
    for arm in arms:
        label, extra, env = A.arm_parts(arm)
        assert env.get("CUDA_VISIBLE_DEVICES") == both, \
            f"{label} is not running on both cards; it cannot be a split arm"
        assert "--spec-type" not in extra, \
            f"{label} speculates; see CORRECTIONS 32"
    modes = {tuple(A.arm_parts(a)[1]) for a in arms}
    assert len(modes) == len(arms), "two arms in the split set are identical"


def test_the_split_set_names_a_baseline_report_will_actually_pick():
    """report() selects by the '-base' suffix and otherwise takes the first arm
    ALPHABETICALLY, which silently decides what every delta is measured from."""
    labels = [A.arm_parts(a)[0] for a in A.ARM_SETS["dual-split"]]
    assert sum(l.endswith("-base") for l in labels) == 1, \
        f"exactly one arm must carry '-base'; got {labels}"


def test_the_split_baseline_is_the_llama_cpp_default():
    """The question is what to change FROM. If the baseline were one of the
    overrides, a null result would read as 'the default is fine' while never
    having run it."""
    base = next(a for a in A.ARM_SETS["dual-split"]
                if A.arm_parts(a)[0].endswith("-base"))
    extra = A.arm_parts(base)[1]
    assert "-ts" not in extra and "-sm" not in extra, \
        f"the baseline overrides the split it is supposed to be a control for: {extra}"


def test_the_depth_set_holds_everything_but_the_split_fixed():
    """Issue #52 stage 4.

    The question at 147,456 is whether -sm tensor's +59.5 % from ctx 16,384
    survives depth -- CORRECTIONS 23 says a verdict at one depth does not
    transfer, and this project has watched draft-mtp change SIGN between 16K
    and 131,072. So the two arms differ in the split and in nothing else, and
    both carry the -ub 1024 that stage 2 won.
    """
    arms = A.ARM_SETS["dual-depth"]
    both = OTHER + "," + gpu_device.SERVED_GPU_UUID
    seen = []
    for arm in arms:
        label, extra, env = A.arm_parts(arm)
        assert env.get("CUDA_VISIBLE_DEVICES") == both, label
        assert "--spec-type" not in extra, f"{label} speculates; see CORRECTIONS 32"
        assert "-ub" in extra and extra[extra.index("-ub") + 1] == "1024", (
            f"{label} does not carry the micro-batch stage 2 won")
        seen.append([x for x in extra if x not in ("-ub", "1024")])
    assert seen[0] != seen[1], "the two depth arms are identical"


def test_the_depth_baseline_is_the_configuration_being_challenged():
    """The profile serves -sm tensor. If IT were the baseline, a null result
    would read as "the change is safe" while the thing under test is whether
    the change was right at all. The default split is what to measure from."""
    base = next(a for a in A.ARM_SETS["dual-depth"]
                if A.arm_parts(a)[0].endswith("-base"))
    assert "tensor" not in A.arm_parts(base)[1], (
        f"the depth baseline already applies the split under test: "
        f"{A.arm_parts(base)[1]}")
