"""A run whose binary has no kernels for a visible GPU is not a measurement.

THE INCIDENT, 2026-08-27, and it was self-inflicted within the hour. The n-gram
family sweep was launched with QWEN38_TARGET set and QWEN38_EXE not set, so it
took `dflash2_arena.EXE`'s default -- C:\AI\llama.cpp-dflash2 -- whose
CMakeCache records CMAKE_CUDA_ARCHITECTURES=89. cuobjdump on its ggml-cuda.dll
lists 141 sm_89 cubins, NO sm_120a, and NO PTX to fall back on. The served
profile uses C:\AI\llama.cpp-blackwell, which carries 141 of each.

Fifteen rows came back at ctx 147,456 with `66+0` residency, both cards holding
memory, and rates near enough the served figures to look right. Every one of
those logs reads `CUDA : ARCHS = 890` while an RTX 5060 Ti of compute capability
12.0 was visible and in use. HOW they ran is not established; that they are not
the served binary is, and that alone voids them.

The default was never updated after the second card arrived. Nothing checked it:
the SERVING profiles carry a cuobjdump build guard, the arena does not.

WHY THIS IS AN OBSERVATION AND NOT A PRELAUNCH CHECK. Reading cubins out of a
DLL needs cuobjdump at a hardcoded CUDA path, and predicts what the process will
load rather than reading what it did. llama.cpp prints the architectures its
loaded backend was compiled for on every boot. Comparing that line to the
compute capabilities the driver reports is a measurement of the run itself, and
it costs one regex.

BLAST RADIUS, audited the same hour: 750 logs carry an ARCHS line. 191 read
`890,1200`, 399 read the eight-architecture upstream default from the old card,
160 read `890` -- and of those 160, EXACTLY THE 15 FROM THIS SWEEP had a second
CUDA device. No historical dual-GPU row is affected.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import archs_missing_for_gpus


ARCHS_BOTH = "system_info: n_threads = 18 / 20 | CUDA : ARCHS = 890,1200 | USE_GRAPHS = 1"
ARCHS_ADA  = "system_info: n_threads = 18 / 20 | CUDA : ARCHS = 890 | USE_GRAPHS = 1"
ARCHS_OLD  = "CUDA : ARCHS = 500,610,700,750,800,860,890,900 | USE_GRAPHS = 1"


def test_both_architectures_present_is_clean():
    assert archs_missing_for_gpus(ARCHS_BOTH, ["8.9", "12.0"]) == []


def test_the_incident_is_caught():
    """sm_89-only binary, Blackwell card visible."""
    assert archs_missing_for_gpus(ARCHS_ADA, ["8.9", "12.0"]) == ["12.0"]


def test_a_single_ada_card_on_an_ada_binary_is_clean():
    """The 160 sm_89-only logs are almost all correct runs, not faults."""
    assert archs_missing_for_gpus(ARCHS_ADA, ["8.9"]) == []


def test_the_old_upstream_default_covers_ada_but_not_blackwell():
    assert archs_missing_for_gpus(ARCHS_OLD, ["8.9"]) == []
    assert archs_missing_for_gpus(ARCHS_OLD, ["8.9", "12.0"]) == ["12.0"]


def test_every_missing_capability_is_named_not_just_the_first():
    got = archs_missing_for_gpus("CUDA : ARCHS = 890", ["12.0", "10.0"])
    assert sorted(got) == ["10.0", "12.0"]


def test_a_log_with_no_archs_line_raises_rather_than_passing():
    """Silence is the failure this guard exists to stop.

    Returning [] for a log that never said which architectures it carried would
    report `clean` on exactly the evidence that is missing.
    """
    with pytest.raises(ValueError):
        archs_missing_for_gpus("no such line here", ["8.9"])


def test_no_visible_gpus_raises():
    with pytest.raises(ValueError):
        archs_missing_for_gpus(ARCHS_BOTH, [])
