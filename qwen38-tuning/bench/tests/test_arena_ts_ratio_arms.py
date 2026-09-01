r"""The `ts-ratio` arm set: does tilting the tensor split toward the Blackwell
card buy anything on an NVFP4 artifact?

WHY THIS IS NOT A RE-RUN OF A SETTLED QUESTION. `docs/results/09-hardware.md`
records `-ts 1,1` at "+1.8 %, noise" and the page carries a red retraction on the
sentence that followed it. That measurement was taken **under `-sm layer`, on
`UD-Q4_K_XL`** -- an artifact on which both cards run the SAME kernel, so there
was nothing for a ratio to buy. The same page also closes native FP4 as
"unreachable for us", and states why: *"Native FP4 needs MXFP4 or NVFP4 weights.
That is an artifact swap, not a flag."*

**The artifact was swapped.** We serve `NVFP4-MTP-VERY-LOW`, so at `mmq.cu:131`

    const bool use_native_fp4 = blackwell_mma_available(cc) &&
        (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);

is now true on the 5060 Ti and false on the 4070 SUPER, which is Ada and where
`blackwell_mma_available()` is false by construction. The two cards run different
kernels over the same tensors, and under `-sm tensor` every layer is split across
both -- so no layer runs the fast path alone. That is a mechanism the earlier
ratio measurement could not have had.

THE DESIGN. Three points, not two, and the total budget is held constant so only
the proportion moves. A single arm beating the control would prove nothing; the
claim is that the line slopes, so the counter-direction has to be measured too.

Headroom is the limit and it is small: at runtime the 4070 SUPER holds ~11.2 GB
of 12.0 and the 5060 Ti ~14.5 GB of 16.0. `tilt-5060` is deliberately the
largest push those numbers allow. **If it OOMs, that is the answer to how far
this can go**, and the row voids rather than lying.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as arena  # noqa: E402


def _arms():
    return dict((a[0], (a[1], a[2])) for a in arena.ARM_SETS["ts-ratio"])


def _ts(argv):
    return argv[argv.index("-ts") + 1]


def _pair(argv):
    return tuple(int(x) for x in _ts(argv).split(","))


def test_the_arm_set_has_three_points_not_two():
    """Two points cannot show a slope, and a slope is the claim."""
    assert sorted(_arms()) == ["control", "tilt-4070", "tilt-5060"]


def test_the_control_is_the_ratio_we_serve_at_this_depth():
    assert _ts(_arms()["control"][0]) == "7819,15490"


def test_the_total_budget_is_identical_across_arms():
    """If the total moved too, the delta would have two causes: the proportion
    and the size."""
    totals = {name: sum(_pair(argv)) for name, (argv, _) in _arms().items()}
    assert len(set(totals.values())) == 1, totals


def test_the_three_ratios_are_monotone_and_straddle_the_control():
    share = {name: _pair(argv)[1] / sum(_pair(argv)) for name, (argv, _) in _arms().items()}
    assert share["tilt-4070"] < share["control"] < share["tilt-5060"], share


def test_no_arm_asks_the_5060_for_more_than_the_card_has():
    """16,311 MiB total on the 5060 Ti. A budget above it is not an experiment,
    it is a load failure with extra steps."""
    for name, (argv, _) in _arms().items():
        assert _pair(argv)[1] <= 16311, (name, _pair(argv))


def test_only_the_ratio_moves():
    """Same target, same decoder, same ubatch, same split mode -- everything the
    +63.1 % arm carried, with one number changed."""
    base = _arms()["control"][0]
    for name, (argv, _) in _arms().items():
        assert len(argv) == len(base), name
        differing = [i for i, (a, b) in enumerate(zip(argv, base)) if a != b]
        assert differing == [] or differing == [base.index("-ts") + 1], (name, differing)
        assert arena.NVFP4_VERY_LOW in argv, name
        assert "draft-mtp,ngram-mod" in " ".join(argv), name
