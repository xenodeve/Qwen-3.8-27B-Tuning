r"""A second deep corpus, from a codebase nobody here wrote.

WHY IT EXISTS. On 2026-08-24 every arena row at ctx 147,456 came back void: the
model answered a 64,210-token `real-code-deep` prompt in **9 tokens** and stopped
on EOS, while a 43,162-token prompt of the same corpus at the same ctx ran the
full 512-token budget, and a 48-token prompt did too (issue #44). So the context
setting is not the cause. Two explanations remain:

  LENGTH   any prompt of that size collapses, and the arena cannot reach the
           window we serve at all.
  CONTENT  this text, at that slice, makes EOS the greedy continuation.

`real-code-deep` cannot tell them apart -- its long slice is its short slice
plus more of the same 45 files, all written inside this repo. A corpus from an
unrelated project, cut to the same length, can.

WHAT THIS FILE GUARANTEES, and what it deliberately does not.

It does NOT assert the two corpora produce similar rates. They will not, and
they are not allowed to be compared: `test_corpus_depth.py` already records that
`real-code` and `real-code-deep` are DIFFERENT REGIMES carrying different hashes,
and this is a third. A cross-corpus ratio is not a quantity.

It DOES assert the new corpus cannot flatter `ngram-mod`, which is one of the
arms measured on it. `ngram-mod` keys a hash on a window of n_match TOKENS
(`common/ngram-mod.cpp:15-25`), so 24-word window repetition is the number that
matters and line repetition is not -- 30 % of these lines repeat because real
Python shares `import os` and `return None`.

    real-code-deep         406,146 chars   0.4 % at n=24
    gguf-py/gguf alone      597,630        0.7 %
    plus scripts+examples 1,141,245        1.6 %   <- rejected for this reason

The wider set passed the builder's 5 % refusal and was still the wrong choice.
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dflash2_arena as arena
import harness

CORPORA = Path(arena.__file__).parent / "corpora"
VENDOR = CORPORA / "real-code-vendor.txt"
DEEP = CORPORA / "real-code-deep.txt"
REGIME = "real-code-vendor"


def test_the_regime_is_registered_where_filler_looks_it_up():
    assert arena.CORPUS_FILES.get(REGIME) == "real-code-vendor.txt"


def test_the_regime_is_offered_on_the_command_line():
    """Registered but not selectable is a corpus nobody can run.

    Asserted through the parser rather than by reading the source: the first
    version of this test searched the 200 characters after `--regime` and broke
    the moment the choices list was wrapped onto its own line, which is a test
    that fails on formatting instead of on behaviour."""
    import subprocess
    out = subprocess.run([sys.executable, str(Path(arena.__file__)), "--help"],
                         capture_output=True, text=True, timeout=120)
    assert REGIME in (out.stdout + out.stderr), (
        "--help does not offer %s" % REGIME)


def test_the_corpus_is_a_committed_file():
    assert VENDOR.is_file(), "the prompt must come from a frozen file"


def test_it_is_not_the_deep_corpus_under_another_name():
    """A copy would answer nothing: the question is whether DIFFERENT text of
    the same length behaves differently."""
    assert VENDOR.read_bytes() != DEEP.read_bytes()
    assert arena.corpus_hash(REGIME) != arena.corpus_hash("real-code-deep")


def test_every_row_can_name_which_text_it_ran_on():
    h = arena.corpus_hash(REGIME)
    assert h and len(h) >= 8, h


def test_it_covers_the_window_we_serve():
    """filler() asks for int(ctx * 0.5) tokens and returns n * 3 chars, so
    ctx 147,456 needs 221,184. Below that the depth guard raises instead of
    truncating, which is correct and would also mean this corpus is useless."""
    need = int(147456 * 0.5) * 3
    assert VENDOR.stat().st_size >= need, (VENDOR.stat().st_size, need)
    text = arena.filler(int(147456 * 0.5), REGIME)
    assert len(text) >= need


def test_it_cannot_flatter_the_ngram_arm():
    """ngram-mod is measured on this text and keys on a 24-token window. The
    builder's own refusal is 5 %; this is the tighter bar that rejected the
    three-directory set at 1.6 %."""
    text = VENDOR.read_text(encoding="utf-8", errors="replace")
    win24 = harness.window_repetition_pct(text, 24)
    assert win24 < 1.0, (
        "%.1f %% of 24-word windows repeat; ngram-mod would be measuring the "
        "corpus rather than itself" % win24)


def test_the_builder_is_not_on_the_measurement_path():
    """CORRECTIONS.md 20: the benchmark once built its prompt from source that
    was being edited between runs, and reported 78.9 against 105.4 tok/s on
    byte-identical arguments. filler() must read the committed file."""
    src = Path(arena.__file__).read_text(encoding="utf-8")
    assert "build-vendor-corpus" not in src
