"""parse_layer_split must name WHICH model it counted, once a drafter is present.

INSTRUMENT FAULT (2026-08-22, issue #17). Loading UD-IQ2_XXS with the DFlash2
drafter emits four assignment passes:

    pass 1:  6 layers   the drafter, during --fit's memory probe
    pass 2: 65 layers   the target
    pass 3: 65 layers   the target, reserve pass
    pass 4:  6 layers   the drafter, for real

parse_layer_split() takes the LAST pass by design -- correct for a single-model
log, and it returned (6, 0) here. Every DFlash2 row would have recorded "6+0,
fully resident", a healthy-looking split describing the wrong model, and no
spill of the target's 65 layers could ever have shown up in it.

This is the project's stated worst case: an instrument that returns a
believable number instead of failing. Caught before it produced a row, not
after.

The fix is `expect_layers`: name how many layers the model you mean has, and
the parser picks the last pass of that size. Asking for a size no pass has is
an error, never a silent fallback to some other pass.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import parse_layer_split


def _pass(n, device="CUDA0", first_cpu_at=None):
    out = []
    for i in range(n):
        d = "CPU" if first_cpu_at is not None and i >= first_cpu_at else device
        out.append(f"load_tensors: layer {i:>3} assigned to device {d}, is_swa = 0")
    return "\n".join(out)


# The real shape: drafter, target, target reserve, drafter again.
TWO_MODEL_LOG = "\n".join([_pass(6), _pass(65), _pass(65), _pass(6)])


def test_without_expect_layers_it_still_reads_the_last_pass():
    """Unchanged for every existing caller -- and wrong here, which is the point."""
    assert parse_layer_split(TWO_MODEL_LOG) == (6, 0)


def test_expect_layers_selects_the_target_not_the_drafter():
    assert parse_layer_split(TWO_MODEL_LOG, expect_layers=65) == (65, 0)


def test_expect_layers_can_still_select_the_drafter_deliberately():
    assert parse_layer_split(TWO_MODEL_LOG, expect_layers=6) == (6, 0)


def test_a_spilled_target_is_visible_through_the_drafter():
    """The failure the old behaviour could not report: target layers on the CPU."""
    log = "\n".join([_pass(6), _pass(65, first_cpu_at=60), _pass(6)])
    assert parse_layer_split(log) == (6, 0), "last pass is still the drafter"
    assert parse_layer_split(log, expect_layers=65) == (60, 5)


def test_asking_for_a_size_no_pass_has_is_an_error():
    with pytest.raises(ValueError, match="no assignment pass"):
        parse_layer_split(TWO_MODEL_LOG, expect_layers=40)


def test_it_takes_the_last_pass_of_that_size_not_the_first():
    """Pass 2 is the load, pass 3 the reserve. Later passes reflect final placement."""
    log = "\n".join([_pass(65, first_cpu_at=60), _pass(65)])
    assert parse_layer_split(log, expect_layers=65) == (65, 0)
