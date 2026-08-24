r"""A 512-token verbatim copy of the prompt is not a decode measurement.

INSTRUMENT FAULT, 2026-08-24, issue #44. `real-code-vendor` was built so the
arena could reach ctx 147,456, and it does: 7 of 7 prompt lengths generate the
full 512-token budget where `real-code-deep` collapsed to 9. The first arena row
taken on it read

    draft-mtp+ngram  195.13 tok/s  acc 100.0
      ngram-mod  decline 24.1 %  mean len 32.85  n_gen 1912  n_acc 1911
      draft-mtp  decline  0.0 %  mean len  3.84  n_gen   57  n_acc   54

**1,911 of 1,912 drafted tokens accepted, in runs averaging 32.85.** That is not
speculation succeeding, it is the model reproducing the prompt: `ngram-mod`
drafts by matching text already in the context, and a continuation that copies
the source it was given is exactly what it predicts perfectly. The generated
text agrees --

    '# ==== gguf-py/gguf/constants.py ====\nfrom __future__ import'

-- the model is continuing the corpus, not answering the question appended to it.
Three of the seven sweep lengths did this and four produced a real answer, and
`generation_is_measurable` passed all seven, because it counts TOKENS.

**195.13 tok/s is a copy rate.** A table mixing copy rows with answer rows inside
one arm's six rounds has a median that measures neither, and it would have looked
like the best number this project ever recorded.

WHY THE CHECK IS ON THE OUTPUT AND NOT ON THE COUNTERS. Gating on `ngram-mod`'s
`mean_acc_len` would be circular: `ngram-mod` is one of the arms under test, and
a guard that voids rows where it does well cannot be used to find out whether it
does well. The overlap between the OUTPUT and the PROMPT is a property of the
workload, and every arm sees the same one.

WHY 12-WORD WINDOWS. `--spec-ngram-mod-n-match 12` is what every worker profile
serves, so 12 is the width at which copying actually pays the decoder. The
project already measures corpus repetition the same way
(`harness.window_repetition_pct`), and reusing the idiom keeps the two numbers
readable side by side.

THE THRESHOLD IS A FIRST GUESS AND SAYS SO. 0.5 was chosen to separate the two
populations this session actually observed, not derived. It is recorded on the
row either way, so a later run can move it on evidence rather than on taste.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import copied_window_fraction, generation_is_original

PROMPT = ("def vram_settled(before, after):\n"
          "    return abs(after - before) < 64\n"
          "# the allocator keeps handing memory back for a while after a load\n"
          "# so a reading taken too early reports a number that is about to move\n")


def test_a_verbatim_continuation_scores_near_one():
    out = ("# the allocator keeps handing memory back for a while after a load\n"
           "# so a reading taken too early reports a number that is about to move\n")
    assert copied_window_fraction(out, PROMPT, n=12) > 0.9


def test_an_answer_in_its_own_words_scores_low():
    out = ("This function returns True when two VRAM readings are within 64 MiB "
           "of one another. The point is to avoid trusting a measurement taken "
           "while the driver is still releasing buffers, which would record a "
           "figure that changes a second later.")
    assert copied_window_fraction(out, PROMPT, n=12) < 0.2


def test_short_output_cannot_form_a_window_and_is_not_called_copying():
    """Fewer than n words yields no window at all. Returning 1.0 there would
    void every short answer as a copy; returning 0.0 says 'no evidence'."""
    assert copied_window_fraction("too short", PROMPT, n=12) == 0.0


def test_the_row_gate_refuses_a_copy_and_accepts_an_answer():
    copy = ("# the allocator keeps handing memory back for a while after a load\n"
            "# so a reading taken too early reports a number that is about to move\n")
    answer = ("This function returns True when two VRAM readings are within 64 "
              "MiB of one another. The point is to avoid trusting a measurement "
              "taken while the driver is still releasing buffers, which would "
              "record a figure that changes a second later.")
    assert generation_is_original([copy], PROMPT) is False
    assert generation_is_original([answer], PROMPT) is True


def test_all_generations_must_qualify_not_the_median():
    """A row is one paired datapoint. `generation_is_measurable` already refuses
    to average a good sample with a 3-token one, for the same reason."""
    copy = ("# the allocator keeps handing memory back for a while after a load\n"
            "# so a reading taken too early reports a number that is about to move\n")
    answer = ("This function returns True when two VRAM readings are within 64 "
              "MiB of one another. The point is to avoid trusting a measurement "
              "taken while the driver is still releasing buffers, which would "
              "record a figure that changes a second later.")
    assert generation_is_original([answer, answer, copy], PROMPT) is False


def test_missing_content_is_not_silently_treated_as_original():
    """A server that returns no content must void the row, not pass it. The
    whole point is that a believable number with no evidence behind it is worse
    than a failure."""
    assert generation_is_original([None], PROMPT) is False
    assert generation_is_original([], PROMPT) is False
