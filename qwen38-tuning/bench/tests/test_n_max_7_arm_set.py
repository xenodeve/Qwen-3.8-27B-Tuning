r"""The n-max-7 arms must differ from the measured arms by the NUMBER and nothing else.

WHY 7, AND WHY IT WAS NEVER TRIED.

`common/common.h:325` defaults `n_max` to **3**. `common/speculative.cpp:989`
computes the ceiling from the drafter's own metadata:

    const int32_t n_draft_max = is_dspark && sample_from_anchor
                                ? block_size : block_size - 1;
    if (params.n_max > n_draft_max) { LOG_WRN("... clamping to %d"); }

and this project's own boot log prints `block_size=8` for DFlash2, so the ceiling
is **7**. Every DFlash2 figure this repo holds -- including all of report 29 --
was measured at **4**, a value the ledger records as "chosen without knowing
either number", with two independent reviews calling it the largest unclaimed
lever on the list.

The cost is known and flat: the Gated DeltaNet recurrent state is
`149.62 x (1 + n_max)` MiB and does not scale with context, so 4 -> 7 is
**+448.84 MiB**. On `UD-Q2_K_XL` that takes the arm from 12,973 to 13,422 of the
15,172 llama.cpp sees.

WHAT THIS FILE GUARDS

One number changed, nothing else. A raised `n_max` is only attributable if the
arm is otherwise byte-identical to the arm already measured -- `CORRECTIONS.md`
26 and 28 are both this project publishing a delta with two causes.

It also guards the MTP arm against `-md`. That arm exists to test whether the
ceiling applies to the head baked into `UD-Q2_K_XL`; feeding a sidecar would
answer a different question under the same label.

WHAT IT CANNOT DO is know whether 7 survives. `draft-mtp` prints no `block_size`
line, so its ceiling is unread -- and the server clamps with a WARNING rather
than an error, which means a run can silently measure 3 while the row says 7.
Reading `n_max=` back out of each boot log is the only check, and it belongs to
the run, not to this file.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena

ARMS = arena.ARM_SETS["n-max-7"]
BY_LABEL = {arena.arm_parts(a)[0]: arena.arm_parts(a) for a in ARMS}

MEASURED_DFLASH = dict(arena.arm_parts(a) and (arena.arm_parts(a)[0], arena.arm_parts(a)[1])
                       for a in arena.ARM_SETS["decoders"])["dflash2+ngram"]
MEASURED_MTP = dict((arena.arm_parts(a)[0], arena.arm_parts(a)[1])
                    for a in arena.ARM_SETS["mtp"])["draft-mtp+ngram"]


def n_max_of(extra):
    return extra[extra.index("--spec-draft-n-max") + 1]


def test_the_set_has_both_arms():
    assert set(BY_LABEL) == {"dflash2+ngram n7", "draft-mtp+ngram n7"}, sorted(BY_LABEL)


def test_both_request_seven():
    for label, (_l, extra, _env) in BY_LABEL.items():
        assert n_max_of(extra) == "7", label


def test_the_dflash_arm_differs_from_the_measured_one_only_in_the_number():
    got = list(BY_LABEL["dflash2+ngram n7"][1])
    want = list(MEASURED_DFLASH)
    want[want.index("--spec-draft-n-max") + 1] = "7"
    assert got == want, "the n7 arm is not the measured arm with one number changed"


def test_the_mtp_arm_differs_from_the_measured_one_only_by_adding_the_number():
    got = [x for x in BY_LABEL["draft-mtp+ngram n7"][1]
           if x not in ("--spec-draft-n-max", "7")]
    assert got == list(MEASURED_MTP), \
        "the mtp n7 arm changed something besides adding --spec-draft-n-max 7"


def test_the_mtp_arm_still_passes_no_sidecar():
    extra = BY_LABEL["draft-mtp+ngram n7"][1]
    assert "-md" not in extra and "-ngld" not in extra


def test_neither_arm_varies_the_environment():
    for label, (_l, _e, env) in BY_LABEL.items():
        assert env == {}, label
