"""The report must never pool two prompt regimes into one series.

THE INCIDENT (2026-08-22, issue #18). The arena ran both regimes and the report
printed one block per ctx, pooling them:

    ngram-mod  [119.7, 119.4, 119.3, 53.0, 52.5, 49.3]   (baseline)

Three synthetic rounds and three real-code rounds, averaged into a single
baseline. Every delta computed against it was meaningless, and the header --
`ctx=16384`, with no regime -- was the only clue.

`ngram-mod` scores 119 where the prompt is 66.2 % duplicate lines and 52 where
it is 4.7 %. Pooling those is averaging a decoder's best case with its worst
and reporting the result as its performance. It does not crash and it does not
look wrong: it produces a plausible number, which is the failure mode this
project exists to refuse.

The regime split was written and the patch that installed it was applied
without an assertion, so it silently did not match. The code stayed unchanged
and the run reported anyway.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dflash2_arena as arena


def _rows():
    """Two regimes, deliberately far apart -- the real spread was 119 vs 52."""
    out = []
    for rnd, v in enumerate([120.0, 119.0, 119.5], 1):
        out.append(dict(ctx=16384, regime="synthetic", arm="ngram-mod",
                        round=rnd, tg_med=v))
    for rnd, v in enumerate([53.0, 52.0, 49.0], 1):
        out.append(dict(ctx=16384, regime="real-code", arm="ngram-mod",
                        round=rnd, tg_med=v))
    for rnd, v in enumerate([108.0, 108.5, 108.0], 1):
        out.append(dict(ctx=16384, regime="synthetic", arm="dflash2",
                        round=rnd, tg_med=v))
    for rnd, v in enumerate([69.5, 69.0, 69.8], 1):
        out.append(dict(ctx=16384, regime="real-code", arm="dflash2",
                        round=rnd, tg_med=v))
    return out


def test_each_regime_gets_its_own_block(capsys):
    arena.report(_rows())
    out = capsys.readouterr().out
    assert "synthetic" in out, "the report does not name the regime at all"
    assert "real-code" in out
    assert out.count("(baseline)") == 2, (
        "expected one baseline per regime; got %d" % out.count("(baseline)")
    )


def test_a_series_never_mixes_regimes(capsys):
    """Six values on one line is the signature of the pooling bug."""
    arena.report(_rows())
    for line in capsys.readouterr().out.splitlines():
        if "[" in line and "]" in line:
            n = line[line.index("[") + 1:line.index("]")].count(",") + 1
            assert n <= 3, (
                "a series carries %d rounds; three were run per regime, so "
                "this line pools regimes: %s" % (n, line.strip())
            )


def test_the_verdict_differs_between_regimes(capsys):
    """The whole point: dflash2 loses on synthetic and wins on real-code.

    A report that pools them can only produce one verdict, and it would be
    wrong for both.
    """
    arena.report(_rows())
    out = capsys.readouterr().out
    blocks = out.split("regime=")
    syn = next(b for b in blocks if b.startswith("synthetic"))
    real = next(b for b in blocks if b.startswith("real-code"))
    syn_line = next(l for l in syn.splitlines() if "dflash2" in l)
    real_line = next(l for l in real.splitlines() if "dflash2" in l)
    assert "-" in syn_line.split("]")[1], "dflash2 should be negative on synthetic"
    assert "+" in real_line.split("]")[1], "dflash2 should be positive on real-code"
