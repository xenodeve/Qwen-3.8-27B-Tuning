"""Guards the `nvfp4-dspark-layer` arm set (DSpark under the layer split).

Why layer split, 2026-09-04: on build 10499 the DSpark drafter cannot ride the
tensor split -- its Markov head has no split axis for the meta backend
(`ggml-backend-meta.cpp:537`), and pinned to one device it borrows the target's
`output.weight` from a Meta buffer that a single-device scheduler cannot run
(`ggml-backend.cpp:930`). Under `-sm layer` neither applies. The set therefore
answers the mechanism question -- does DSpark v2 accept on this NVFP4 target --
not the serving question; the served profile is `-sm tensor` (+31 % over layer).
"""
import dflash2_arena as A


def test_layer_pair_differs_only_in_the_drafter():
    arms = A.ARM_SETS["nvfp4-dspark-layer"]
    assert [A.arm_parts(a)[0] for a in arms] == ["nvfp4-mtp-layer", "nvfp4-dspark-layer"]
    mtp, dsp = A.arm_parts(arms[0]), A.arm_parts(arms[1])
    assert mtp[1][:len(A.DUAL_LAYER)] == A.DUAL_LAYER
    assert dsp[1][:len(A.DUAL_LAYER)] == A.DUAL_LAYER
    assert "-ts" not in mtp[1] and "-ts" not in dsp[1]
    assert mtp[1][mtp[1].index("--spec-type") + 1] == "draft-mtp,ngram-mod"
    assert dsp[1][dsp[1].index("--spec-type") + 1] == "draft-dspark,ngram-mod"
    assert dsp[1][dsp[1].index("-md") + 1] == A.DSPARK_Q8
    assert mtp[2] == dsp[2]
    drafter_flags = ("-md", "-ngld", "--spec-type", "--spec-draft-n-max", "-ctkd", "-ctvd")
    strip = lambda v: [x for i, x in enumerate(v)
                       if x not in drafter_flags and v[i - 1] not in drafter_flags]
    assert strip(mtp[1]) == strip(dsp[1])
