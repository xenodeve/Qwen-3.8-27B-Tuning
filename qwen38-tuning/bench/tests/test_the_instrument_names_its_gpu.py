"""A VRAM reading must name the card it came from, or refuse.

THE INCIDENT THIS GUARDS, 2026-08-26.

A second GPU was connected. `nvidia-smi --query-gpu=memory.used,memory.free
--format=csv,noheader,nounits` began returning TWO lines, and the two languages
in this repo failed differently:

  Python   `[int(x) for x in o.split(",")]` on '1450, 10548\\n54, 15997'
           raises ValueError. Loud. The arena cannot start. Safe.

  PowerShell  `(nvidia-smi ...) -split '\\s*,\\s*'` yields a FOUR-element array,
           and $vram[0]/$vram[1] silently become the OTHER card's numbers --
           measured that day as used=1481 free=10517, the RTX 4070 SUPER.
           Show-ServerStatus.ps1 would report residency for a card that is not
           running the model, and say nothing.

The second is `CLAUDE.md`'s north star exactly: an instrument that returns a
believable number instead of a failure.

These tests run against the real machine on purpose. A mock would have agreed
with the old code -- the old code was correct for the machine it was written on,
and what changed was the world, not the parse. Only asking the actual driver can
tell us which card answered.
"""
import shutil
import subprocess
import sys
import os

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import gpu_device  # noqa: E402

ABSENT = "GPU-00000000-0000-0000-0000-000000000000"

pytestmark = pytest.mark.skipif(
    shutil.which("nvidia-smi") is None,
    reason="no nvidia-smi on this machine; the property is about the driver",
)


def test_more_than_one_gpu_is_present_so_these_tests_are_not_vacuous():
    """If this machine has one GPU, every assertion below passes for the wrong
    reason -- the old code was already correct in that world. Failing here means
    the guard needs re-checking on a two-card machine, not that the code broke.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    n = len([ln for ln in out.splitlines() if ln.strip()])
    if n < 2:
        pytest.skip(f"only {n} GPU visible; this guard is about ambiguity")
    assert n >= 2


def test_vram_returns_two_integers_where_the_old_parse_raised():
    used, free = gpu_device.vram()
    assert isinstance(used, int) and isinstance(free, int)
    assert free > 0, "a card reporting no free VRAM is not a reading we can use"


def test_the_card_that_answered_is_the_card_we_asked_for():
    """The reading is worthless if it came from the other slot. Ask the driver
    to name what it just measured, rather than trusting the index."""
    assert gpu_device.name() == gpu_device.SERVED_GPU_NAME


def test_an_absent_uuid_refuses_instead_of_answering():
    """The failure mode being guarded is a *plausible* number, so the wrong
    behaviour here is returning anything at all."""
    with pytest.raises(gpu_device.GpuNotPresent):
        gpu_device.vram(ABSENT)


def test_the_served_uuid_is_actually_installed():
    """A UUID constant that has drifted from the hardware turns every reading
    into the refusal above -- correct, but useless. Catch the drift here, where
    the message can say so, rather than in the middle of a sweep."""
    assert gpu_device.is_present(gpu_device.SERVED_GPU_UUID), (
        "the served GPU UUID is not installed on this machine; "
        f"visible: {gpu_device.installed()}")


def test_pinning_hides_the_other_card_from_a_child_process():
    """The env `vram()` reports on and the env llama-server is launched with must
    agree. If they can disagree, a row can name one card and be produced by
    another -- which is the whole defect, moved one layer down."""
    env = dict(os.environ, **gpu_device.pin_env())
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True, text=True, env=env).stdout
    # nvidia-smi ignores CUDA_VISIBLE_DEVICES, so this asserts on what the
    # variable SAYS, not on what nvidia-smi does with it.
    assert env["CUDA_VISIBLE_DEVICES"] == gpu_device.SERVED_GPU_UUID
    assert gpu_device.SERVED_GPU_UUID in out, (
        "the pinned UUID is not among the installed cards")
