r"""`ts-ratio-fine`: is there a gain between 66.5 % and the ratio that broke?

The first sweep (`ts-ratio-147456.jsonl`) established the slope and found its
edge in the same run: tilting AWAY from the Blackwell card cost
**-18.2 % [-20.6, -16.5] RESOLVED**, and tilting further TOWARD it was
**voided in all three rounds** -- not for memory, the arm loaded `66+0` with
2,286 MiB free, but by the prompt-copy guard, `copied_frac` [0, 0, 0.539]
reproducing to the digit each round.

So the served 66.5 % is the best point measured, and the question left is
whether anything sits between it and the cliff.

**The 5060 Ti was emptied to 14 MiB of 16,311 before this run.** That is worth
saying because it does NOT address the failure: `tilt-5060` was rejected for
output, not for OOM, so more headroom should change nothing. The `push` arm is
carried forward unchanged precisely to test that -- if it now scores, the earlier
void was memory pressure after all and the first sweep's reading was wrong.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as arena  # noqa: E402


def _arms():
    return dict((a[0], (a[1], a[2])) for a in arena.ARM_SETS["ts-ratio-fine"])


def _pair(argv):
    return tuple(int(x) for x in argv[argv.index("-ts") + 1].split(","))


def test_the_three_points_are_control_then_two_steps_toward_the_blackwell_card():
    assert sorted(_arms()) == ["control", "mid", "push"]
    share = {n: _pair(a)[1] / sum(_pair(a)) for n, (a, _) in _arms().items()}
    assert share["control"] < share["mid"] < share["push"], share


def test_mid_actually_sits_between_and_is_not_a_rounding_of_either():
    share = {n: _pair(a)[1] / sum(_pair(a)) for n, (a, _) in _arms().items()}
    assert share["mid"] - share["control"] > 0.005, share
    assert share["push"] - share["mid"] > 0.005, share


def test_push_is_the_same_ratio_that_was_voided_so_the_void_can_be_reproduced():
    """Carried forward byte-for-byte. If it scores now, the first reading was
    wrong and the void was memory, not output."""
    assert _pair(_arms()["push"][0]) == (7309, 16000)


def test_the_control_is_still_the_served_ratio():
    assert _pair(_arms()["control"][0]) == (7819, 15490)


def test_the_total_budget_is_identical_across_arms():
    totals = {n: sum(_pair(a)) for n, (a, _) in _arms().items()}
    assert len(set(totals.values())) == 1, totals


def test_only_the_ratio_moves():
    base = _arms()["control"][0]
    for name, (argv, _) in _arms().items():
        assert len(argv) == len(base), name
        differing = [i for i, (a, b) in enumerate(zip(argv, base)) if a != b]
        assert differing in ([], [base.index("-ts") + 1]), (name, differing)
