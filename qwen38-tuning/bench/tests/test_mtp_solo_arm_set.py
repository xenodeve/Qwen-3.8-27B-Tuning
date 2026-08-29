r"""Is the n-gram beside MTP earning its place, or is it dead weight?

THE DEVELOPER'S QUESTION, 2026-08-29: "MTP feels faster." Checked, and
**`draft-mtp` ALONE has never been measured on NVFP4** -- every arm this project
has run on that artifact carries an n-gram beside it, and the one exception is
the n-gram alone.

TWO PIECES OF EVIDENCE POINT OPPOSITE WAYS, which is why this needs measuring
rather than arguing:

  FOR the n-gram: on `real-code-vendor` at ctx 147,456, `n-match 24` is
  +27.1 % RESOLVED over `n-match 12` with MTP held fixed. It is doing real work
  on that corpus.

  AGAINST it: on the developer's own agent traffic, `ngram-mod` generated
  5 drafts in 4,653 calls -- and Unsloth Studio's single runs on this same
  artifact put MTP alone at 54.95 tok/s against 52.28 and 49.72 for MTP+ngram.

WHAT THIS ARM SET CAN AND CANNOT SETTLE. It runs on `real-code-vendor`, which is
the corpus an n-gram is BEST at -- repeated vendor source. So it is a test the
n-gram should win. **If it loses even here, that is decisive. If it wins, the
agent-traffic question is still open** and needs a regime this project does not
have. Say so in the result rather than generalising.

`--spec-draft-n-max` is swept in the same boots because it costs nothing extra
and answers a second question: Studio uses 2, which its own UI documents as
llama.cpp's default for MTP on GPU, while we deviate to 3. Our real-use
acceptance per position is (0.690, 0.448, 0.284) -- position 3 lands 28 % of the
time -- so 3 should win, and a test we expect to win is the useful kind.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)

import dflash2_arena as A  # noqa: E402

SET = "nvfp4-mtp-solo"


def arms():
    return A.ARM_SETS[SET]


def test_the_arm_set_exists():
    assert SET in A.ARM_SETS


def test_the_baseline_is_what_we_serve_today():
    label, extra, _ = arms()[0]
    assert label.endswith("-base"), label
    i = extra.index("--spec-type")
    assert extra[i + 1] == "draft-mtp,ngram-mod", extra[i + 1]
    j = extra.index("--spec-ngram-mod-n-match")
    assert extra[j + 1] == "24", extra[j + 1]


def test_there_is_an_mtp_only_arm():
    """The thing that has never been run on this artifact."""
    solos = [a for a in arms()
             if a[1][a[1].index("--spec-type") + 1] == "draft-mtp"]
    assert solos, [a[0] for a in arms()]


def test_no_mtp_only_arm_carries_an_ngram_flag():
    """An n-gram parameter on an arm with no n-gram is a flag that does nothing
    and a reader who thinks it did."""
    for label, extra, _ in arms():
        if extra[extra.index("--spec-type") + 1] != "draft-mtp":
            continue
        for flag in extra:
            assert not flag.startswith("--spec-ngram"), (label, flag)


def test_the_n_max_deviation_is_swept_in_the_same_boots():
    """3 is ours, 2 is llama.cpp's documented default for MTP on GPU."""
    n_maxes = {a[1][a[1].index("--spec-draft-n-max") + 1] for a in arms()}
    assert {"2", "3"} <= n_maxes, n_maxes


def test_every_arm_runs_the_nvfp4_file_and_nothing_else_differs():
    for label, extra, _ in arms():
        assert A.NVFP4_VERY_LOW in extra, label
        assert "-md" not in extra, "%s: the head is IN the file" % label
        assert extra[extra.index("-sm") + 1] == "tensor", label
        assert "-ts" in extra, label


def test_every_arm_asks_for_both_cards():
    for label, extra, env in arms():
        assert env.get("CUDA_VISIBLE_DEVICES") == A.BOTH_CARDS, label
