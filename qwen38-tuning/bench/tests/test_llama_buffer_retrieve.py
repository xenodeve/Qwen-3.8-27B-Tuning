r"""Retrieval for `llama-buffer` (#54): chunking and scoring, no server.

WHY CHARACTER N-GRAMS AND NOT WORDS

The queries this has to serve are Thai -- `นี่คือแผนอะไร`, `แล้วแผนนี่หละ` -- and
**Thai is written without spaces.** A whitespace tokeniser sees one enormous
token, matches nothing, and returns whichever chunk happened to be first. It
does not fail; it returns the wrong extract, and the model then answers
confidently from it. That is the shape this repo's north star names, arriving in
a product feature.

Character n-grams have no such failure. They are worse than a real Thai word
segmenter and better than anything that assumes spaces, and they cost no model,
no VRAM and no GGUF conversion.

This is probably also why Unsloth Studio runs `ragMode: "hybrid"`: its vector
half is `bge-small-en-v1.5`, an ENGLISH model, so on a Thai query the keyword
half is what carries it.

WHAT THESE TESTS REFUSE TO LET THROUGH

The first test is not "does it retrieve" -- it is "does a Thai query beat a
whitespace tokeniser here at all", because a green retrieval suite that silently
degraded to first-chunk-wins would be worse than no suite.
"""
import os
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(BENCH))
TOOLS = os.path.join(ROOT, "qwen38-tuning", "tools", "llama-buffer")


def _load(name):
    """By file, not by sys.path -- `bench/tap.py` already collided once."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "llama_buffer_" + name, os.path.join(TOOLS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def R():
    return _load("retrieve")


# --------------------------------------------------------------- tokenising

def test_thai_is_not_one_token(R):
    """The whole reason this file exists."""
    toks = R.tokens("นี่คือแผนอะไร")
    assert len(toks) > 3, toks
    assert len(set(toks)) > 3, toks


def test_english_still_tokenises_sensibly(R):
    toks = set(R.tokens("cookie migration plan"))
    assert any("cookie" in t for t in toks), sorted(toks)[:10]


def test_a_thai_query_overlaps_thai_text_that_shares_words(R):
    a = set(R.tokens("แผนการย้าย session ไปเก็บใน cookie"))
    b = set(R.tokens("นี่คือแผนอะไร"))
    assert a & b, "no n-gram in common between two Thai strings sharing 'แผน'"


def test_case_and_punctuation_do_not_split_a_match(R):
    assert set(R.tokens("Cookie-Migration!")) & set(R.tokens("cookie migration"))


# ---------------------------------------------------------------- chunking

DOC = "\n".join([
    "# Auth middleware cookie migration",
    "",
    "## Goal",
    "ย้าย session ของ Supabase จาก localStorage ไปเก็บเป็น httpOnly cookies",
    "ผ่าน @supabase/ssr เพื่อให้ middleware อ่านได้",
    "",
    "## Rollout",
    "ทำทีละขั้น เริ่มจาก staging แล้วค่อยขึ้น production",
    "",
    "## Streamer profile",
    "หน้าโปรไฟล์สตรีมเมอร์ยังไม่อยู่ในแผนนี้ อยู่ใน sprint 4",
])


def test_chunking_returns_more_than_one_chunk(R):
    cs = R.chunk(DOC, target_chars=120)
    assert len(cs) > 1, cs


def test_every_character_survives_chunking(R):
    """A chunker that drops text is a retriever that cannot find it."""
    joined = "".join(c.text for c in R.chunk(DOC, target_chars=120))
    for line in DOC.splitlines():
        if line.strip():
            assert line.strip() in joined, line


def test_a_chunk_knows_where_it_came_from(R):
    """The marker injected into the prompt has to say WHICH part this was, or
    the model cannot ask for the rest."""
    cs = R.chunk(DOC, target_chars=120)
    assert all(c.start >= 0 and c.end > c.start for c in cs)
    assert cs[0].start == 0
    assert cs[-1].end == len(DOC)


def test_chunks_do_not_overlap_or_leave_gaps(R):
    cs = R.chunk(DOC, target_chars=120)
    for a, b in zip(cs, cs[1:]):
        assert a.end == b.start, (a.end, b.start)


# --------------------------------------------------------------- retrieving

def test_a_thai_query_finds_the_thai_section(R):
    cs = R.chunk(DOC, target_chars=120)
    top = R.top_k(cs, "cookie migration คืออะไร", k=1)
    assert "cookie" in top[0].text.lower(), top[0].text


def test_a_thai_only_query_finds_the_right_section(R):
    """No shared Latin at all -- this is the case a word tokeniser loses."""
    cs = R.chunk(DOC, target_chars=120)
    top = R.top_k(cs, "โปรไฟล์สตรีมเมอร์", k=1)
    assert "สตรีมเมอร์" in top[0].text, top[0].text


def test_it_beats_first_chunk_wins(R):
    """The degenerate failure this suite exists to catch: a scorer that ties
    everything and returns whatever came first."""
    cs = R.chunk(DOC, target_chars=120)
    top = R.top_k(cs, "โปรไฟล์สตรีมเมอร์", k=1)
    assert top[0] is not cs[0], "retrieval returned the first chunk regardless"


def test_k_is_respected_and_ordered(R):
    cs = R.chunk(DOC, target_chars=80)
    top = R.top_k(cs, "session cookie", k=3)
    assert len(top) == 3
    s = [R.score(c, "session cookie") for c in top]
    assert s == sorted(s, reverse=True), s


def test_asking_for_more_than_exists_returns_everything(R):
    cs = R.chunk(DOC, target_chars=120)
    assert len(R.top_k(cs, "อะไรก็ได้", k=999)) == len(cs)


def test_an_empty_query_does_not_crash_or_lie(R):
    cs = R.chunk(DOC, target_chars=120)
    top = R.top_k(cs, "", k=2)
    assert len(top) == 2


# ------------------------------------------------------------- the marker

def test_the_extract_says_it_is_an_extract(R):
    """REDUCTION MUST BE STATED. A model handed five chunks with nothing saying
    so answers as though it read the document -- which is the confusion this
    whole tool is meant to prevent, arriving by a different door."""
    cs = R.chunk(DOC, target_chars=120)
    out = R.render(R.top_k(cs, "cookie", k=1), total=len(cs), name="plan.md",
                   original_chars=len(DOC))
    low = out.lower()
    assert "extract" in low or "excerpt" in low, out
    assert "plan.md" in out
    assert str(len(cs)) in out, "the marker does not say how many parts exist"


def test_the_marker_says_how_to_get_the_rest(R):
    cs = R.chunk(DOC, target_chars=120)
    out = R.render(R.top_k(cs, "cookie", k=1), total=len(cs), name="plan.md",
                   original_chars=len(DOC))
    assert "ask" in out.lower() or "request" in out.lower(), out


def test_the_extract_still_contains_the_retrieved_text(R):
    cs = R.chunk(DOC, target_chars=120)
    top = R.top_k(cs, "cookie", k=1)
    out = R.render(top, total=len(cs), name="plan.md", original_chars=len(DOC))
    assert top[0].text.strip() in out


def test_rendering_nothing_is_refused(R):
    """An empty extract with a confident marker is the worst possible output."""
    with pytest.raises(ValueError):
        R.render([], total=3, name="plan.md", original_chars=100)
