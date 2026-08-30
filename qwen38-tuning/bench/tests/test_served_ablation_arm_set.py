r"""The served decoder against its own ablations, so the verdict is not a guess.

WHY THIS SET EXISTS.

`worker-q2kxl-mtp.ps1` serves `--spec-type draft-mtp,ngram-mod` at ctx 147,456,
and that choice rests on **one unpaired session per arm** on a single real task
(`docs/reports/35`). Two separate things say that is not enough:

  * `docs/results/02-decoders.md` records `draft-mtp` at **+81 % at 16K and
    -71 % at 131,072**. We serve at **147,456** -- deeper than the depth where
    the sign flipped. That measurement used a sidecar head on an artifact with
    none, so it does not transfer directly, and that is exactly the point: the
    number that would transfer has never been taken.

  * An operator on the same RTX 5060 Ti 16 GB published the paired curve we lack
    (`docs/researchs/hf-discussion-5060ti-mtp/`): 2.08x at 2,500 tokens decaying
    to **1.72x at 25,400**, and his measurement stops there. Ours runs six times
    deeper.

Issue #44. Three arms, because dropping MTP alone leaves `ngram-mod` -- the
decoder every other worker profile serves -- and dropping both is the only
honest floor.

WHAT WOULD SILENTLY INVALIDATE IT.

**A second changed variable.** `CORRECTIONS.md` 26 and 28 are both deltas with
two causes: 26 compared depth while the drafter also changed, 28 divided two
tok/s figures taken at different draft acceptance. So the ngram window must be
byte-identical in the two arms that have one, and no arm may carry a flag beyond
its `--spec-type` and that window.

**Passing `-md`.** `UD-Q2_K_XL` carries `blk.64.nextn.*`; a sidecar would move
the head into a file, add its weights to `fit_params_target`
(`server-context.cpp:1074`, gated only on "was -md given"), and label the result
as the configuration that was not run.

WHAT THIS FILE CANNOT DO is make the run valid. Alternating the arms within a
round is the runner's job -- across boots the same arm with byte-identical
counters has spanned **48.9 %** at 65,536 (`CORRECTIONS.md` 23). This file only
guarantees the three argv it will alternate between differ in one thing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dflash2_arena as arena

ARMS = arena.ARM_SETS["served-ablation"]
BY_LABEL = {arena.arm_parts(a)[0]: arena.arm_parts(a) for a in ARMS}


def spec_types(extra):
    """The value following `--spec-type`, split on commas; () if absent."""
    if "--spec-type" not in extra:
        return ()
    return tuple(extra[extra.index("--spec-type") + 1].split(","))


def test_the_set_is_the_served_arm_and_its_two_ablations():
    assert set(BY_LABEL) == {"draft-mtp+ngram", "ngram-mod", "none"}, sorted(BY_LABEL)


def test_the_served_arm_is_byte_identical_to_what_the_profile_launches():
    """If it drifts from the profile, the run measures something we do not
    serve and the verdict cannot be applied to the thing it was taken for."""
    label, extra, env = BY_LABEL["draft-mtp+ngram"]
    assert extra == ["--spec-type", "draft-mtp,ngram-mod"] + arena.NGRAM, extra
    assert env == {}


def test_the_ablations_remove_exactly_one_thing_each():
    mtp = spec_types(BY_LABEL["draft-mtp+ngram"][1])
    ngram = spec_types(BY_LABEL["ngram-mod"][1])
    none = spec_types(BY_LABEL["none"][1])
    assert mtp == ("draft-mtp", "ngram-mod")
    assert ngram == ("ngram-mod",), "dropping MTP must leave ngram-mod intact"
    assert none == (), "the floor arm must not speculate at all"


def test_both_speculating_arms_carry_the_same_ngram_window():
    """A different window between them is a second variable, and the delta
    becomes unattributable -- the shape of CORRECTIONS 26 and 28."""
    served = BY_LABEL["draft-mtp+ngram"][1]
    ablated = BY_LABEL["ngram-mod"][1]
    for flag, value in (("--spec-ngram-mod-n-match", "12"),
                        ("--spec-ngram-mod-n-min", "16"),
                        ("--spec-ngram-mod-n-max", "32")):
        for label, extra in (("draft-mtp+ngram", served), ("ngram-mod", ablated)):
            assert flag in extra, f"{label} is missing {flag}"
            assert extra[extra.index(flag) + 1] == value, f"{label} {flag}"


def test_the_floor_arm_carries_no_flags_at_all():
    assert BY_LABEL["none"][1] == [], BY_LABEL["none"][1]


def test_no_arm_passes_a_sidecar_drafter():
    for label, (_l, extra, _env) in BY_LABEL.items():
        assert "-md" not in extra, f"{label} passes -md; the head must come from the target"
        assert "-ngld" not in extra, f"{label} passes -ngld, which only means anything with -md"


def test_no_arm_sets_an_environment_variable():
    for label, (_l, _e, env) in BY_LABEL.items():
        assert env == {}, f"{label} varies the environment as well: {env}"


def test_no_arm_overrides_a_flag_server_argv_already_hardcodes():
    """`extra` is appended, so an override reaches llama-server twice and only
    last-wins parsing decides. An arm set that got that backwards would report a
    flat sweep taken entirely at the hardcoded value."""
    fixed = arena.server_argv(147456, [])
    hardcoded = {a for a in fixed if a.startswith("-")}
    for label, (_l, extra, _env) in BY_LABEL.items():
        clash = sorted({a for a in extra if a.startswith("--spec") is False
                        and a.startswith("-")} & hardcoded)
        assert clash == [], f"{label} re-specifies {clash}"
