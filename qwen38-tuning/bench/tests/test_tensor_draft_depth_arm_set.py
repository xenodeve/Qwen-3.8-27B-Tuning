r"""`--spec-draft-n-max` 4 against 7, where DFlash2 is actually fastest.

THE QUESTION

`draft-dflash,ngram-mod` under `-sm tensor` on the patched mirror is the
fastest paired figure this project holds -- 65.1 / 64.3 / 63.8 tok/s at ctx
65,536, +123.8 % over the incumbent. Every one of those rounds ran at
`--spec-draft-n-max 4`, a value the ledger records as "chosen without knowing
either number": `common.h:325` defaults it to 3, and `speculative.cpp:989`
clamps it at `block_size - 1`, which for this drafter is 7.

7 was measured ONCE, on 2026-08-24, in a different configuration, and it took
25 % off DFlash2's wall clock. If that carries here it raises the highest
number this project has.

WHY IT IS NOT OBVIOUSLY SAFE, AND WHY THE ARM SET STILL CARRIES IT

The recurrent state is 149.62 x (1 + n_max), so 4 -> 7 costs about 449 MiB --
on the arm that already finishes with the least headroom in the register.
Probed 2026-08-30 at this exact configuration, both load and both answer a real
request: n_max 4 leaves [1043, 770] MiB and n_max 7 leaves [870, 462]. **462
MiB on CUDA1 is between the two numbers the profile measured** -- 336 free died
on the first request, 488 survived 135,233 tokens -- so this arm may still die
on the real corpus at depth. That is a result the arena records, not a reason
to leave the arm out.

THE HAZARDS THIS FILE EXISTS FOR

1. THE WRONG BINARY. `draft-dflash` under `-sm tensor` needs the mirror patch;
   on the served binary it aborts at ggml-backend-meta.cpp:543. Existing tensor
   dflash arm sets reach the mirror by requiring QWEN38_LLAMA_EXE to be
   EXPORTED, which makes an arm set an incomplete description of its own
   experiment -- and if the variable is unset the whole sweep dies, while if it
   points somewhere else the sweep runs and reports. These arms pin it.

2. `-sm tensor` WITHOUT A RATIO. That is the even split, and on this pair it is
   the 0.38 tok/s configuration (CORRECTIONS 33). The ratio travels with the
   split mode here, never separately.

3. THE TWO DFLASH ARMS DIFFERING IN ANYTHING ELSE. The whole question is one
   integer. If they differ anywhere else the delta has two causes.

4. MTP DRAGGED TO 7 FOR SYMMETRY. `qwen35.nextn_predict_layers = 1` -- the head
   predicts ONE token ahead, and 7 measured -56 % on it with acceptance falling
   from 0.48-0.61 to 0.38-0.44. Matching the drafters to each other would be
   matching a number that does not mean the same thing twice.
"""
import os
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as arena  # noqa: E402

SET = "tensor-draft-depth"
MIRROR = r"C:\AI\llama.cpp-mirror\build-mirror\bin\llama-server.exe"
CTX = 65536


def parts():
    return [arena.arm_parts(a) for a in arena.ARM_SETS[SET]]


def decoder_of(extra):
    for i, tok in enumerate(extra):
        if tok == "--spec-type" and i + 1 < len(extra):
            return extra[i + 1]
    return None


def depth_of(extra):
    if "--spec-draft-n-max" not in extra:
        return None
    return extra[extra.index("--spec-draft-n-max") + 1]


def dflash_arms():
    return [(l, e) for l, e, _ in parts()
            if decoder_of(e) == "draft-dflash,ngram-mod"]


# ------------------------------------------------------------------ the set

def test_the_arm_set_exists():
    assert SET in arena.ARM_SETS


def test_it_is_four_arms():
    assert len(parts()) == 4, [p[0] for p in parts()]


def test_it_carries_both_draft_depths():
    assert sorted(depth_of(e) for _, e in dflash_arms()) == ["4", "7"], \
        [(l, depth_of(e)) for l, e in dflash_arms()]


def test_it_carries_the_incumbent_as_an_anchor():
    """Without `ngram-mod` in the same rotation the two dflash figures can only
    be compared to a table from another boot series, which is the thing this
    bench exists to stop."""
    assert [decoder_of(e) for _, e, _ in parts()].count("ngram-mod") == 1, \
        [decoder_of(e) for _, e, _ in parts()]


def test_it_carries_mtp_so_the_ordering_can_be_rechecked():
    assert [decoder_of(e) for _, e, _ in parts()].count("draft-mtp,ngram-mod") == 1


# ------------------------------------------- hazard 1: the patched binary

def test_every_arm_pins_the_mirror_build():
    for label, _, env in parts():
        assert (env or {}).get(arena.ENV_VAR) == MIRROR, (
            "draft-dflash under -sm tensor needs the mirror patch; an arm that "
            "inherits the module default runs the served binary and aborts, "
            "or runs whatever QWEN38_LLAMA_EXE happens to hold and reports",
            label, (env or {}).get(arena.ENV_VAR))


def test_the_pinned_binary_is_the_mirror_and_exists():
    """Named, not inherited -- and present. A pin at a path that is not there
    is the same failure as no pin, one boot later."""
    assert "llama.cpp-mirror" in MIRROR and "blackwell" not in MIRROR, MIRROR
    if not os.path.isfile(MIRROR):
        pytest.skip("the mirror build is not on this machine")


def test_the_row_records_the_mirror_and_not_the_default():
    """CORRECTIONS 34's shape: a column recording the module default while
    another binary ran. Here it would silently claim the patched result came
    from the served build."""
    label, extra, env = parts()[0]
    r = arena.new_row(CTX, label, 1, "synthetic", extra, env, 1000)
    assert r["exe"] == MIRROR, r["exe"]


# ------------------------------------------- hazard 2: the split and its ratio

def test_every_arm_uses_the_tensor_split():
    for label, extra, _ in parts():
        assert "-sm" in extra, label
        assert extra[extra.index("-sm") + 1] == "tensor", (label, extra)


def test_every_arm_carries_the_ratio_with_the_split():
    """CORRECTIONS 33: `-sm tensor` with no `-ts` is the even split, which on
    this pair is the 0.38 tok/s configuration. The ratio is not optional."""
    for label, extra, _ in parts():
        assert "-ts" in extra, (
            "-sm tensor without a ratio is the 0.38 tok/s configuration", label)
        assert extra[extra.index("-ts") + 1] == "7819,15490", (label, extra)


def test_no_arm_uses_the_layer_split():
    for label, extra, _ in parts():
        assert "layer" not in extra, (label, extra)


# ------------------------------- hazard 3: the two dflash arms differ in ONE int

def test_the_two_dflash_arms_differ_only_in_the_draft_depth():
    (la, ea), (lb, eb) = dflash_arms()

    def without_depth(e):
        i = e.index("--spec-draft-n-max")
        return e[:i] + e[i + 2:]

    assert without_depth(ea) == without_depth(eb), (
        "the two draft depths differ in something else too, so the delta has "
        "two causes", la, lb, ea, eb)


def test_both_dflash_arms_carry_the_drafter_file():
    for label, extra in dflash_arms():
        assert "-md" in extra, (
            "without -md the arm silently degrades to plain ngram-mod", label)
        assert extra[extra.index("-ngld") + 1] == "99", (label, extra)


def test_the_labels_name_the_depth():
    for label, extra in dflash_arms():
        assert "n%s" % depth_of(extra) in label.replace(" ", ""), (
            "a label that does not name its draft depth makes the JSONL "
            "unreadable without parsing argv", label)


# ----------------------------------- hazard 4: MTP is NOT dragged to the ceiling

def test_mtp_stays_at_three():
    """`qwen35.nextn_predict_layers = 1`. 7 measured -56 % on MTP with
    acceptance falling from 0.48-0.61 to 0.38-0.44. 7 is DFlash2's ceiling
    because its block_size is 8; it is not a shared setting."""
    for label, extra, _ in parts():
        if decoder_of(extra) == "draft-mtp,ngram-mod":
            assert depth_of(extra) == "3", (label, depth_of(extra))
            return
    pytest.fail("no MTP arm in the set")


def test_the_mtp_arm_carries_no_sidecar():
    for label, extra, _ in parts():
        if decoder_of(extra) == "draft-mtp,ngram-mod":
            assert "-md" not in extra, (label, extra)


# --------------------------------------------------------- held constant

def test_every_arm_names_the_same_target():
    seen = set()
    for label, extra, _ in parts():
        assert "-m" in extra, label
        seen.add(extra[extra.index("-m") + 1])
    assert len(seen) == 1, seen
    assert "UD-Q4_K_XL" in seen.pop()


def test_every_arm_holds_the_micro_batch_at_the_served_value():
    for label, extra, _ in parts():
        assert extra[extra.index("-ub") + 1] == "1024", (label, extra)


def test_every_arm_sees_both_cards():
    for label, _, env in parts():
        assert (env or {}).get("CUDA_VISIBLE_DEVICES") == arena.BOTH_CARDS, label


def test_the_ngram_window_is_the_served_one_everywhere():
    for label, extra, _ in parts():
        for flag, value in zip(arena.NGRAM[::2], arena.NGRAM[1::2]):
            assert flag in extra, (label, flag)
            assert extra[extra.index(flag) + 1] == value, (label, flag)


# ------------------------------------------------ it refuses rather than lies

def test_a_missing_mirror_stops_the_run():
    label, extra, env = parts()[0]
    bad = dict(env)
    bad[arena.ENV_VAR] = r"C:\nope\llama-server.exe"
    with pytest.raises(Exception):
        arena.server_argv(CTX, extra, env=bad, verify=True)
