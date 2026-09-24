"""Offline A/B: pythainlp normalize/remove_repeat_vowels on stored Thai rows.

TDD red-first for tools/thai-normalize-ab.py (no GPU, no server restart).
Each test names the incident or claim it guards.
"""
import importlib.util
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING = os.path.dirname(BENCH)
TOOL = os.path.join(TUNING, "tools", "thai-normalize-ab.py")


def _load_tool():
    spec = importlib.util.spec_from_file_location("thai_normalize_ab", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_missing_vowel_forms_survive_both_functions():
    """#81 transcript: ดาวนโหลด/วดีโอ/ลิงก are MISSING vowels, and no
    dedup/reorder can restore a missing vowel. The A/B must report them
    unchanged, not claim a fix."""
    m = _load_tool()
    for broken in ["ดาวนโหลด", "วดีโอ", "ลิงก", "ไฟล"]:
        assert m.run_normalize(broken) == broken, broken
        assert m.run_repeat(broken) == broken, broken


def test_repeated_vowels_collapse():
    """The research claim: ดีีี -> ดี, นานาาา -> นานา. Verifies the library
    does what the research says on this machine, this version."""
    m = _load_tool()
    assert m.run_normalize("ดีีี") == "ดี"
    assert m.run_repeat("ดีีี") == "ดี"
    assert m.run_normalize("นานาาา") == "นานา"


def test_correct_thai_and_code_spans_are_byte_identical():
    """normalize() must not touch correct Thai (เว็บไซต์) or ASCII code
    spans (backticks, paths, JSON, URLs) -- the scope rule from the research."""
    m = _load_tool()
    text = ('Endpoint `/api/info` คืนสถานะ 200 ไฟล์ `Rick-Astley-213s.mp4` '
            'อยู่ที่ downloads/ เว็บไซต์ขายของ http://127.0.0.1:5000 '
            '{"status": 200} ชื่อไฟล์ถูกต้อง')
    assert m.run_normalize(text) == text, repr(m.run_normalize(text))
    assert m.code_violations(text, m.run_normalize(text)) == []


def test_ascii_code_runs_are_derived_not_listed():
    """The code-untouched check derives its set (ASCII runs holding code
    punctuation) instead of listing sentinel filenames -- a new filename
    tomorrow is covered the day it lands."""
    m = _load_tool()
    runs = m.code_runs('รัน `python app.py --port 5000` แล้วเปิด C:/logs/serve-2026.log ดู')
    assert any("app.py" in r for r in runs)
    assert any("serve-2026.log" in r for r in runs)
    # pure Thai prose yields no code runs at all
    assert m.code_runs("สรุปผลเป็นภาษาไทย 8-10 ประโยค") == []


def test_ab_row_records_before_after_counts_and_latency():
    """Every A/B row carries raw/normalized broken counts, the changed flag,
    code violations and per-arm latency -- auditable end to end (#81 fault 2:
    rows that stored counts without text)."""
    m = _load_tool()
    row = m.ab_row("ดาวนโหลดวิดีโอ ดีีี `app.py`")
    assert row["text"] == "ดาวนโหลดวิดีโอ ดีีี `app.py`"
    assert row["raw_broken"] >= 1          # ดาวนโหลด still counted broken
    assert row["norm_broken"] >= 1         # ... and still broken after normalize
    assert row["changed_norm"] is True     # ... but the row did change (ดีีี -> ดี)
    assert row["code_violations"] == []
    assert row["ms_norm"] >= 0 and row["ms_rep"] >= 0
    assert row["pythainlp_version"] == m.PYTHAINLP_VERSION


def test_broken_patterns_are_reused_not_copied():
    """Single source: the A/B counts with thai-sampler-pair's patterns, so a
    pattern fix there reaches here without a second edit."""
    m = _load_tool()
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "thai_sampler_pair",
        os.path.join(TUNING, "tools", "thai-sampler-pair.py"))
    pair = iu.module_from_spec(spec)
    spec.loader.exec_module(pair)
    assert m.BROKEN_PATTERNS == pair.BROKEN_PATTERNS
