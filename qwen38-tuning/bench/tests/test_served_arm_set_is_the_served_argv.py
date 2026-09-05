"""Guards the one-arm set used to pair llama.cpp against EXL3 in one boot (issue #71).

Incident it prevents, 2026-09-03: every EXL3 figure in results 10 was compared
against llama.cpp rows from a different boot, because the only arm sets carrying
the served config also carry a second arm (q4-ngram-base, mtp-solo...) and a
pairing run that pays two boots per round was never started. A single-arm set
that is byte-for-byte the served config (NVFP4 VERY-LOW, draft-mtp + ngram-mod,
n-max 3, n-match 24, both cards, tensor split) makes the pairing affordable --
and this test keeps that set from drifting into a second variable.
"""
import dflash2_arena as A


def test_served_set_has_exactly_the_served_arm():
    arms = A.ARM_SETS["nvfp4-served"]
    assert len(arms) == 1
    label, argv, env = A.arm_parts(arms[0])
    assert label == "nvfp4-served"
    assert argv[:len(A.DUAL_TENSOR)] == A.DUAL_TENSOR
    assert argv[argv.index("-m") + 1] == A.NVFP4_VERY_LOW
    assert argv[argv.index("--spec-type") + 1] == "draft-mtp,ngram-mod"
    assert argv[argv.index("--spec-draft-n-max") + 1] == "3"
    assert argv[argv.index("--spec-ngram-mod-n-match") + 1] == "24"
    assert "-md" not in argv  # the MTP head is inside the file
    assert env["CUDA_VISIBLE_DEVICES"] == A.BOTH_CARDS


def test_served_set_is_byte_identical_to_the_final_arm_that_measured_39_4():
    served = A.arm_parts(A.ARM_SETS["nvfp4-served"][0])
    final = next(a for a in A.ARM_SETS["nvfp4-final"]
                 if A.arm_parts(a)[0] == "nvfp4-mtp+nm24")
    assert served[1] == A.arm_parts(final)[1]
    assert served[2] == A.arm_parts(final)[2]
