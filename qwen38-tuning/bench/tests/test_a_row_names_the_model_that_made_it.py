r"""A row must name the model the ARM ran, not the one the module defaults to.

THE INCIDENT, 2026-08-29.

`nvfp4-final-147456.jsonl` was written to settle whether NVFP4 with its baked-in
MTP head beats the artifact we serve. Both arms recorded:

    target: ...\Qwen3.8-27B-UD-Q4_K_XL.gguf
    target_mib: 17093.08

while `nvfp4-mtp+nm24` had actually loaded a different file entirely, named in
its own `args` as `-m ...\Qwen3.8-27B-NVFP4-MTP-VERY-LOW.gguf`. A reader of the
raw file who did not parse `args` would conclude the two arms differed only in
their decoder flags -- which is the opposite of what the run was for.

THIS IS THE FAILURE MODE THE FIELD WAS ADDED TO PREVENT. Its own comment says
"two files on this machine share the name UD-Q2_K_XL and differ by 808 MiB, so
the path alone is not an identity". The field was blind to the one way an arm
can change its model: overriding `-m`.

Nothing was published from the bad column -- the arm labels carried the
distinction and the report reads `args` -- so no number is retracted. The
instrument is fixed because a plausible wrong column is exactly what this
project treats as worse than a crash.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as A  # noqa: E402


def test_an_arm_without_m_reports_the_default_target():
    """The control arm names no model. Its row must still name the default."""
    row = A.new_row(16384, "ngram-mod", 1, "real-code", [], {}, 15000)
    assert row["target"] == A.TARGET
    assert row["target_mib"] == A.model_size_mib(A.TARGET)


def test_an_arm_that_overrides_m_reports_ITS_model():
    """`-m <path>` in the arm's own flags is the model that ran."""
    row = A.new_row(16384, "nvfp4-mtp", 1, "real-code",
                    ["-sm", "tensor", "-m", A.NVFP4_VERY_LOW,
                     "--spec-type", "draft-mtp"],
                    {}, 15000)
    assert row["target"] == A.NVFP4_VERY_LOW
    assert row["target"] != A.TARGET


def test_the_size_follows_the_overridden_model():
    """A path alone is not an identity -- the size must move with it."""
    row = A.new_row(16384, "nvfp4-mtp", 1, "real-code",
                    ["-m", A.NVFP4_VERY_LOW], {}, 15000)
    assert row["target_mib"] == A.model_size_mib(A.NVFP4_VERY_LOW)
    assert row["target_mib"] != A.model_size_mib(A.TARGET)


def test_the_drafter_is_not_mistaken_for_the_target():
    """`-md` names the DRAFTER. It must not overwrite the target column."""
    row = A.new_row(16384, "dflash", 1, "real-code",
                    ["--spec-type", "draft-dflash", "-md", A.DFLASH_SMALL,
                     "-ngld", "99"],
                    {}, 15000)
    assert row["target"] == A.TARGET
