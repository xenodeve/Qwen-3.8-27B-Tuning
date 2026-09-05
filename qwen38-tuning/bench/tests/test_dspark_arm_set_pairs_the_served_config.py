"""Guards the `nvfp4-dspark` arm set (DSpark v2 drafter on the served profile).

Incident it prevents, report 16 §decoders: `draft-dspark` was attempted once
here, "drafter path resolved empty, never launched" -- and the row was never
re-run. The set pairs the served config against the same config with the
DSpark drafter swapped in for the MTP head, so the only difference between the
two arms is the drafter (CORRECTIONS 26 and 28: an arm set that also changes a
flag produces a delta with two causes).
"""
import os

import dflash2_arena as A


def test_dspark_set_is_served_plus_one_drafter_swap():
    arms = A.ARM_SETS["nvfp4-dspark"]
    assert [A.arm_parts(a)[0] for a in arms] == ["nvfp4-served", "nvfp4-dspark"]
    served = A.arm_parts(arms[0])
    dspark = A.arm_parts(arms[1])
    # same served argv object for the baseline
    assert served[1] == A.arm_parts(A.ARM_SETS["nvfp4-served"][0])[1]
    assert served[2] == dspark[2]
    argv = dspark[1]
    assert argv[argv.index("--spec-type") + 1] == "draft-dspark,ngram-mod"
    assert argv[argv.index("-md") + 1] == A.DSPARK_Q8
    assert argv[argv.index("-ngld") + 1] == "99"
    # the DSpark drafter has no sliding window (5 full-attention layers, 8 kv heads):
    # 20 KiB/token fp16 = 2.9 GB at 147,456, which the served split cannot host
    # (first attempt 2026-09-04 08:0x: cudaMalloc 2016 MiB on device 1 failed).
    # q4_0 brings it to ~0.8 GB; the DFlash2 drafter never needed this because
    # its layers carry a 2,048-token sliding window.
    assert argv[argv.index("-ctkd") + 1] == "q4_0"
    assert argv[argv.index("-ctvd") + 1] == "q4_0"
    # and it cannot ride the tensor split: the Markov head has no split axis
    # (ggml-backend-meta.cpp:537 GGML_ASSERT(ret.axis != SPLIT_AXIS_UNKNOWN),
    # 2026-09-04 08:1x), so the drafter lives whole on the 5060 Ti (CUDA1 in
    # the arena's device order).
    assert argv[argv.index("-devd") + 1] == "CUDA1"
    # everything except the drafter flags is byte-identical to the served arm
    drafter_flags = ("-md", "-ngld", "--spec-type", "--spec-draft-n-max", "-ctkd", "-ctvd", "-devd")
    strip = lambda v: [x for i, x in enumerate(v)
                       if x not in drafter_flags and v[i - 1] not in drafter_flags]
    assert strip(argv) == strip(served[1])


def test_dspark_drafter_file_exists_so_the_path_cannot_resolve_empty_again():
    assert os.path.isfile(A.DSPARK_Q8), A.DSPARK_Q8
