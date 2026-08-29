r"""What does the tensor split's CPU draft-sampling actually cost?

THE DEVELOPER'S PUSHBACK, 2026-08-29: sampling on the CPU may be slower than on
the GPU. Correct, and my earlier answer was worse than the evidence.

WHAT I SAID: "layer HAS backend sampling and is still 31 % slower, so the CPU
sampler is not the bottleneck."

WHY THAT IS TOO STRONG: the layer-versus-tensor comparison changed TWO things at
once -- the split, and whether the draft sampler could be offloaded. A net of
-31 % says the split's penalty exceeds the offload's benefit. IT DOES NOT BOUND
THE OFFLOAD'S BENEFIT AT ZERO. It could be worth 20 % while the split costs 51 %.

WHAT IS ACTUALLY GOING ON, read from the binary's help and both logs:

  -bs, --backend-sampling               enable backend sampling   default DISABLED
  --spec-draft-backend-sampling         offload DRAFT sampling    default ENABLED

The MAIN sampler is on the CPU under both splits -- the default is off and this
project never passes `-bs`. What the tensor split loses is the DRAFT sampling
offload, and the logs say so exactly:

  tensor  `spec common_specu: ... backend_sampling=1`
          `W set_sampler: backend sampling not supported with SPLIT_MODE_TENSOR;
             using CPU`
  layer   `spec common_specu: ... backend_sampling=1`
          (no refusal -- it ran on the backend)

THE ISOLATION. `-sm layer` is the only split here where the offload works, so it
is the only place the offload can be varied ALONE. One flag, everything else
held:

    layer-bs-on    default                              offload on the backend
    layer-bs-off   --no-spec-draft-backend-sampling     offload on the CPU

The delta is the offload's worth, X. Two things follow from X, and neither is
obtainable any other way here:

  the tensor split's TRUE advantage over layer is about 31 % + X, because tensor
  is already paying the CPU price;

  X is a tax this configuration pays and CANNOT avoid, since the offload is
  refused under the split that wins by 31 %. Knowing its size decides whether
  that is worth caring about or is noise.

MEASURED ON layer, WHICH IS NOT WHAT WE SERVE. The number is about the offload,
not about a servable configuration -- say so in the result.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as A  # noqa: E402

SET = "draft-sampling-cost"


def arms():
    return A.ARM_SETS[SET]


def test_the_arm_set_exists():
    assert SET in A.ARM_SETS


def test_it_is_a_pair():
    assert len(arms()) == 2, [a[0] for a in arms()]


def test_both_arms_run_on_the_layer_split():
    """The only split where the offload works, so the only place it can be
    varied on its own."""
    for label, extra, _ in arms():
        assert extra[extra.index("-sm") + 1] == "layer", label
        assert "-ts" not in extra, (
            "%s: a ratio under layer is a second variable" % label)


def test_exactly_one_arm_disables_the_offload():
    off = [a for a in arms() if "--no-spec-draft-backend-sampling" in a[1]]
    assert len(off) == 1, [a[0] for a in arms()]


def test_the_baseline_leaves_the_default_alone():
    """The offload is ENABLED by default; the baseline must not name the flag
    at all, or the pair varies a spelling rather than a behaviour."""
    label, extra, _ = arms()[0]
    assert label.endswith("-on") or label.endswith("-base"), label
    assert not any("spec-draft-backend-sampling" in f for f in extra), extra


def test_only_that_flag_differs():
    def strip(extra):
        return [f for f in extra if "spec-draft-backend-sampling" not in f]
    a, b = strip(arms()[0][1]), strip(arms()[1][1])
    assert a == b, "the arms differ by more than the offload:\n  %s\n  %s" % (a, b)


def test_both_arms_speculate_or_the_flag_is_meaningless():
    """A draft-sampling flag on an arm with no drafter measures nothing."""
    for label, extra, _ in arms():
        i = extra.index("--spec-type")
        assert "draft-mtp" in extra[i + 1], (label, extra[i + 1])
