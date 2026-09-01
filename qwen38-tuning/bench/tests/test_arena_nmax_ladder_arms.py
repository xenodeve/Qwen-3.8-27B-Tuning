r"""`ngram-nmax-ladder`: reproduce the +15.6 %, and find whether 64 is the peak.

`ngram-window-147456.jsonl` put `--spec-ngram-mod-n-max 64` at **+15.63 %** over
the served 32 -- 52.76 against 45.63, per-arm spreads 0.8 % and 1.2 %, every row
`66+0` with free_after within 26 MiB. The harness labelled it *"clears this run's
spread, not the applied floor"*, the floor being 13.6 % measured at ctx 16,384 on
Ada, which `CLAUDE.md` says must be re-derived at depth. So: unconfirmed, and
worth confirming before anything touches the served profile.

**64 is llama.cpp's own default** (`--help`: *"maximum number of ngram tokens ...
(default: 64)"*). Our 32 is a deviation BELOW it, and `worker-q4-dual.ps1:1252-1264`
records why: 16/32 were *"held constant rather than chosen"*, and a 48/64 attempt
was *"REVERTED WITHOUT A VERDICT"* because on agent traffic the n-gram recorded
`#gen drafts = 0` and the change was inert either way.

That caveat travels with this result: the arena's `real-code-vendor` corpus makes
the drafter fire (it declines 97-98 % of calls but it fires), and the served
workload may not. **This ladder measures the corpus, not the served traffic.**

The ladder reproduces 32-vs-64 on fresh boots and asks whether more is better.
"""
import os, sys
BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)
import dflash2_arena as arena  # noqa: E402


def _arms():
    return dict((a[0], a[1]) for a in arena.ARM_SETS["ngram-nmax-ladder"])


def _v(argv, flag):
    return argv[argv.index(flag) + 1]


def test_the_ladder_climbs_from_the_served_value_through_the_default_and_past_it():
    got = [(_v(a, "--spec-ngram-mod-n-max")) for _, a in sorted(_arms().items())]
    assert sorted(int(x) for x in got) == [32, 64, 96, 128]


def test_the_control_is_the_served_value_and_the_second_rung_is_the_llama_cpp_default():
    assert _v(_arms()["nmax-32-served"], "--spec-ngram-mod-n-max") == "32"
    assert _v(_arms()["nmax-64-default"], "--spec-ngram-mod-n-max") == "64"


def test_n_min_stays_at_ours_because_48_was_measured_and_lost():
    """studio-48-64 was -10.58 % in the same run. Carrying 48 up the ladder
    would fold a measured loss into every rung."""
    for name, argv in _arms().items():
        assert _v(argv, "--spec-ngram-mod-n-min") == "16", name


def test_n_match_is_held_at_24_across_every_rung():
    for name, argv in _arms().items():
        assert _v(argv, "--spec-ngram-mod-n-match") == "24", name


def test_only_n_max_moves():
    base = _arms()["nmax-32-served"]
    for name, argv in _arms().items():
        assert len(argv) == len(base), name
        differing = [i for i, (x, y) in enumerate(zip(argv, base)) if x != y]
        assert differing in ([], [base.index("--spec-ngram-mod-n-max") + 1]), (name, differing)
