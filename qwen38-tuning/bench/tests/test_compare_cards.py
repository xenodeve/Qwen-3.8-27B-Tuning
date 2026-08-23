"""The comparison must refuse to divide two numbers that are not comparable.

THE INCIDENT THIS GUARDS, and it is this project's own, from 2026-08-24.

`docs/reports/CORRECTIONS.md` 28 retracts a published hardware verdict --
"4x slower than the 4070 SUPER" -- that was built by putting two correctly
measured numbers in one table:

    96.92 tok/s   dflash2_arena, ngram-mod, draft acceptance 60.2
    22.67 tok/s   hardware_baseline.py,     draft acceptance 0.14870

Both were right. The ratio was not, because `ngram-mod` is a speculative decoder
and its rate tracks acceptance directly. Nothing in either file objected; a human
read two numbers and wrote a ratio, and it shipped to a results page, the ledger,
an issue and a commit message before anyone noticed.

So `compare_cards.py` exists to make that specific mistake impossible to make
quietly, and this file pins the refusals rather than the arithmetic. The
arithmetic is one division. The refusals are the product.

Three of them, each from a real row in this repo's own data:

  acceptance   the incident above
  completeness `dflash2+ngram` on the 4070 SUPER has median 5.66 over FOUR of
               six rows -- the other two timed out -- with a maximum of 93.29.
               A median over the survivors of an arm that failed twice is not
               that arm's rate, and reporting it as one is how "the drafter is
               catastrophically slow" got written down instead of "the drafter
               is unreliable in a VRAM band".
  corpus       two sweeps on different prompts measure the prompt.

WHAT THIS FILE CANNOT DO is notice a confounder nobody has thought of. It checks
the three that have already cost this project a retraction. A fourth would pass
silently, which is why `compare_cards.py` prints the inputs beside every ratio
rather than only the verdict.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compare_cards as cc


def write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def row(arm="ngram-mod", tg=90.0, acc=60.0, corpus="abc", note=None, rnd=1):
    r = {"arm": arm, "tg_med": tg, "acceptance": acc, "corpus": corpus,
         "round": rnd, "cuda_archs": ["sm_89"]}
    if note:
        r["note"] = note
        r["tg_med"] = None
    return r


def run(capsys, tmp_path, base_rows, cur_rows):
    b = write(tmp_path, "base.jsonl", base_rows)
    c = write(tmp_path, "cur.jsonl", cur_rows)
    cc.compare(b, c)
    return capsys.readouterr().out


# ---------------------------------------------------------------- the refusals

def test_a_ratio_is_withheld_when_acceptance_differs(capsys, tmp_path):
    """The exact shape of CORRECTIONS 28."""
    out = run(capsys, tmp_path,
              [row(acc=60.2, tg=96.92)],
              [row(acc=14.87, tg=25.63)])
    assert "RATIO WITHHELD" in out
    assert "acceptance" in out
    assert "1.14x" not in out and "3.78x" not in out


def test_a_small_acceptance_difference_is_tolerated(capsys, tmp_path):
    """60.2 against 61.4 is the real pair from the two decoder sweeps. Refusing
    that would make the tool useless -- acceptance is never bit-identical across
    boots."""
    out = run(capsys, tmp_path,
              [row(acc=60.2, tg=96.92)],
              [row(acc=61.4, tg=84.74)])
    assert "RATIO WITHHELD" not in out
    assert "1.14x" in out


def test_a_ratio_is_withheld_when_a_side_has_an_incomplete_round(capsys, tmp_path):
    """`dflash2+ngram` on the 4070 SUPER: 4 measurable of 6."""
    out = run(capsys, tmp_path,
              [row(tg=5.66), row(tg=93.29), row(note="TimeoutError: timed out")],
              [row(tg=87.72), row(tg=90.27), row(tg=81.64)])
    assert "RATIO WITHHELD" in out
    assert "survivors" in out


def test_a_ratio_is_withheld_when_the_corpus_differs(capsys, tmp_path):
    out = run(capsys, tmp_path,
              [row(corpus="1a3ae4b813dd8447")],
              [row(corpus="something-else")])
    assert "RATIO WITHHELD" in out
    assert "corpus differs" in out


def test_the_withheld_arm_still_prints_both_sides(capsys, tmp_path):
    """Withholding must not hide the data. The reader has to be able to see what
    was measured and decide for themselves -- a refusal that prints nothing is
    just a different way of losing the number."""
    out = run(capsys, tmp_path,
              [row(acc=60.2, tg=96.92)],
              [row(acc=14.87, tg=25.63)])
    assert "96.92" in out and "25.63" in out


# ------------------------------------------------------------------- direction

def test_a_slower_current_says_slower(capsys, tmp_path):
    out = run(capsys, tmp_path, [row(tg=100.0)], [row(tg=50.0)])
    assert "2.00x slower" in out


def test_a_faster_current_says_faster_and_not_a_fraction(capsys, tmp_path):
    """0.50x faster would be read as half speed by exactly the reader this tool
    is for."""
    out = run(capsys, tmp_path, [row(tg=50.0)], [row(tg=100.0)])
    assert "2.00x faster" in out
    assert "0.50x" not in out


# ---------------------------------------------------------------------- spread

def test_spread_is_peak_to_peak_over_the_median():
    s = {"median": 100.0, "lo": 95.0, "hi": 105.0}
    assert cc.spread_pct(s) == pytest.approx(10.0)


def test_spread_is_none_rather_than_zero_when_there_is_no_median():
    """A missing spread reported as 0.0 % would read as a perfectly stable arm,
    which is the opposite of what no data means."""
    assert cc.spread_pct({"median": None, "lo": None, "hi": None}) is None


def test_an_arm_present_on_only_one_side_is_reported_not_dropped(capsys, tmp_path):
    """A silently dropped arm leaves a table that looks complete -- the failure
    harness.py's docstring names as this project's shared one."""
    out = run(capsys, tmp_path, [row(arm="dflash2")], [row(arm="ngram-mod")])
    assert "dflash2" in out and "ngram-mod" in out
