r"""The model name a client sees must say which artifact it is talking to.

WHY (developer, 2026-08-27): "ใช้ชื่อ qwen38 มันไม่รู้คืออะไร".

Both served profiles announced themselves as `qwen38`. There are now two of
them, serving different files at different quantisations across different
hardware, and a client picking a model from a list could not tell them apart --
nor could a saved transcript say afterwards which one answered.

WHAT THE NAME HAS TO CARRY. The family, the size, and THE QUANTISATION, because
the quantisation is the only thing that differs between the two profiles that a
reader would care about. `docs/results/01-artifacts.md` records two different
files both named `Qwen3.8-27B-UD-Q2_K_XL.gguf` in different snapshot
directories, so even the filename is not an identity -- but the quant tier is
what anybody is actually choosing between.

WHAT THIS IS NOT. It is not a rename of anything on disk, and it does not
change a single measured row: `--alias` is the name in the HTTP API and nothing
else reads it. The one thing it DOES break is a client configured with the old
string, which is why `edit_canary` -- the only client in this repo that names a
model -- is checked here too.
"""
import os
import re

from _invocation import flag, from_source, resolved
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BENCH)
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, BENCH)

SOLO = os.path.join(SCRIPTS, "worker-q2kxl-mtp.ps1")
DUAL = os.path.join(SCRIPTS, "worker-q4-dual.ps1")


def read(p):
    return open(p, encoding="utf-8", errors="replace").read()


def alias_of(path, *args):
    """The alias as it reaches llama-server.

    THROUGH THE DRY RUN, not the source slice. This read the source until
    2026-08-29, when the dual profile's alias became a value chosen at runtime
    (`$alias = if ($Nvfp4) ...`) so that it could follow the artifact -- and the
    slice returned the literal `$alias`. `_invocation` warns about exactly this:
    "what the fallback cannot see: a value computed at runtime". A test that
    reads the shape of a line stops working the moment the line stops being the
    answer.
    """
    return flag(resolved(path, *args), "--alias")


def test_neither_profile_still_answers_to_qwen38():
    for path in (SOLO, DUAL):
        assert alias_of(path) != "qwen38", (
            "%s still announces itself as `qwen38`, which says nothing about "
            "which of the two it is" % os.path.basename(path))


def test_each_profile_names_its_own_quantisation():
    assert alias_of(SOLO) == "Qwen3.8-27B-Q2_K_XL"
    assert alias_of(DUAL) == "Qwen3.8-27B-Q4_K_XL"


def test_the_dual_profile_renames_itself_when_it_swaps_artifacts():
    """`-Nvfp4` loads a different file; the name a client sees must follow it.

    A profile that announced Q4_K_XL while serving NVFP4 would be confidently
    wrong to every caller of /v1/models -- and this one did, until the dry run
    was read by eye on 2026-08-29.
    """
    assert alias_of(DUAL, "-Nvfp4") == "Qwen3.8-27B-NVFP4-MTP"
    assert alias_of(DUAL, "-Nvfp4") != alias_of(DUAL)


def test_the_two_names_differ():
    """The whole point. Two profiles sharing a name is what made a client
    unable to tell which one it reached."""
    assert alias_of(SOLO) != alias_of(DUAL)


@pytest.mark.parametrize("path", [SOLO, DUAL])
def test_the_name_matches_the_file_the_profile_loads(path):
    """A name that says Q4 while loading Q2 is worse than `qwen38` was -- it is
    confidently wrong rather than merely uninformative."""
    out = resolved(path)
    alias, model = flag(out, "--alias"), flag(out, "-m")
    if model.startswith("$"):
        # No dry run on this profile yet, so `resolved` gave the source slice
        # and `-m` is still the variable. Follow it to its default -- and note
        # that this branch cannot see a model chosen at runtime, which is
        # precisely why the dual profile needed one.
        model = next(l for l in read(path).splitlines()
                     if "$Model =" in l and ".gguf" in l)
    quant = alias.rsplit("-", 1)[-1]                   # Q2_K_XL / Q4_K_XL
    assert quant in model, (
        "%s announces %r but loads %s"
        % (os.path.basename(path), alias, model[-60:]))


def test_the_only_client_that_names_a_model_was_updated():
    """`edit_canary.py` is the real-task driver and it passes a model string to
    the worker. Renaming the server without it is a rename that breaks the one
    thing that calls the server by name."""
    import edit_canary
    src = read(os.path.join(BENCH, "edit_canary.py"))
    m = re.search(r'--model".*?default="([^"]+)"', src, re.S)
    assert m, "cannot find edit_canary's default model"
    assert m.group(1) != "local/qwen38", (
        "edit_canary still asks for `local/qwen38`, which no profile serves")
    assert "Q4_K_XL" in m.group(1) or "Q2_K_XL" in m.group(1)
