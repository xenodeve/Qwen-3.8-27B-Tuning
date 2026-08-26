"""Nothing may launch llama-server without saying which card it may use.

THE HAZARD, 2026-08-26.

`--main-gpu` defaults to 0 and llama.cpp's default split spreads layers across
every visible device. On this machine index 0 is the RETIRED RTX 4070 SUPER, and
the `ggml-cuda.dll` in use carries `sm_89` beside `sm_120a` -- so the wrong card
is not merely reachable, it is fully supported. A run started today without a
device would spread the model over two cards of different architectures and
report a throughput number the register has no comparable row for.

The build guard already in `worker-q2kxl-mtp.ps1` cannot catch this: it asks
whether Blackwell SASS is present in the file, and it is. Which card the kernels
land on is not a property of the file.

WHY THE PIN IS AN ENVIRONMENT VARIABLE AND NOT `-dev`.

`CUDA_VISIBLE_DEVICES` accepts a UUID; `-dev`, `-mg` and `-ts` accept positions
in an enumeration that the driver may reorder. Verified 2026-08-26 that
`CUDA_VISIBLE_DEVICES=GPU-059b90e2-...` leaves llama-server reporting exactly
one device, and that an absent UUID leaves it reporting `(none)` -- at which
point it runs on CPU and produces plausible output at a catastrophic rate with
nothing in the row to say why. That is why presence is checked BEFORE launch,
not inferred from the run.
"""
import os
import re
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BENCH)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, BENCH)

import gpu_device  # noqa: E402
import dflash2_arena as A  # noqa: E402

WORKER = os.path.join(ROOT, "scripts", "worker-q2kxl-mtp.ps1")
SERVE = os.path.join(REPO, "serve.ps1")


def _text(p):
    return open(p, encoding="utf-8", errors="replace").read()


# --- the arena: assert on the environment a launch would actually get --------
# Not on the source text. `launch_env` is the function `start()` calls, so what
# it returns IS what reaches the process.

def test_the_arena_pins_the_card_it_launches_with():
    env = A.launch_env({})
    assert env.get("CUDA_VISIBLE_DEVICES") == gpu_device.SERVED_GPU_UUID


def test_the_arena_pin_survives_an_arm_that_sets_other_variables():
    """Arms carry their own env (GGML_CUDA_GRAPH_OPT and friends). If an arm's
    dict could displace the pin, one arm in a sweep would run on different
    hardware than the rest and the paired comparison would be meaningless."""
    env = A.launch_env({"GGML_CUDA_GRAPH_OPT": "1"})
    assert env.get("CUDA_VISIBLE_DEVICES") == gpu_device.SERVED_GPU_UUID
    assert env.get("GGML_CUDA_GRAPH_OPT") == "1"


def test_an_arm_may_still_override_the_pin_deliberately():
    """Issue #51 measures BOTH cards on purpose. A pin that cannot be lifted
    turns that experiment into an edit of this file, which is how a measurement
    config drifts from the served one."""
    both = "GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4," + gpu_device.SERVED_GPU_UUID
    env = A.launch_env({"CUDA_VISIBLE_DEVICES": both})
    assert env.get("CUDA_VISIBLE_DEVICES") == both


# --- the worker and the launcher: source, because pytest cannot boot a 15 GB
# model. Asserted on the UUID CONSTANT, which cannot be produced by a comment
# and cannot be broken by wrapping a line.

HELPER = os.path.join(ROOT, "scripts", "Get-GpuVram.ps1")


def test_the_worker_names_the_served_card():
    """Asserted on the REFERENCE, not on the literal.

    The first version of this test required the UUID string to appear in the
    worker. It was green, and it was measuring the wrong thing: it would have
    stayed green if the worker named a card nobody uses, and it went red the
    moment the literal was de-duplicated -- calling an improvement a
    regression. That is the fifth time in two sessions an assertion in this
    suite has held the shape of a file instead of a property.
    """
    t = _text(WORKER)
    assert "$env:CUDA_VISIBLE_DEVICES" in t, (
        "worker-q2kxl-mtp.ps1 does not pin a GPU. With two cards installed "
        "llama.cpp will use both, and no row will say so.")
    assert "ServedGpuUuid" in t, (
        "the worker pins a card but does not take the UUID from Get-GpuVram.ps1 "
        "-- a second copy of that constant is a second thing to forget")


def test_the_two_languages_name_the_same_card():
    """Python reads the VRAM, PowerShell launches the server. If those two
    constants ever disagree, a row records one card's memory while another card
    runs the model, and every field in it looks fine."""
    assert gpu_device.SERVED_GPU_UUID in _text(HELPER), (
        f"Get-GpuVram.ps1 does not name {gpu_device.SERVED_GPU_UUID}; "
        f"the PowerShell and Python halves have drifted")


def test_the_worker_checks_the_card_is_there_before_launching():
    """An absent UUID leaves llama-server with zero devices and it runs on CPU.
    The check has to happen before the model loads, or the failure is 40 minutes
    of plausible output."""
    t = _text(WORKER)
    assert "Test-ServedGpuPresent" in t, (
        "the worker pins a card but never checks it is installed")


def test_the_launcher_holds_no_gpu_of_its_own():
    """Same rule as -BindAddress: `serve.ps1` carries no serving configuration,
    so a measured row and a served session cannot diverge by editing the
    launcher. It may PASS a device; it may not DEFINE one."""
    t = _text(SERVE)
    assert gpu_device.SERVED_GPU_UUID not in t, (
        "serve.ps1 hardcodes a GPU UUID. The default belongs in "
        "worker-q2kxl-mtp.ps1 with every other serving flag.")


def test_the_launcher_can_pass_a_device_through():
    assert re.search(r"\$Device", _text(SERVE)), (
        "serve.ps1 offers no way to select a card, so issue #51's two-card arms "
        "would need the profile edited instead of a flag passed")
