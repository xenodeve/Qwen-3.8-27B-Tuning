"""Regression tests for the Thai repair precision incident (#89)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


TUNING = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "thai_eval", TUNING / "tools" / "thai-eval-offline.py")
thai_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(thai_eval)


def test_correct_thai_forms_are_byte_identical():
    raw = "ไฟล์ เว็บไซต์ โปรเจกต์ หรือ ลิงก์ โฟลเดอร์"
    assert thai_eval.dict_fix(raw) == raw


def test_dictionary_does_not_rewrite_any_known_correct_thai_word():
    from pythainlp.corpus.common import thai_words

    changed = [(word, thai_eval.dict_fix(word)) for word in thai_words()
               if thai_eval.dict_fix(word) != word]
    assert changed == []


def test_markdown_code_spans_are_byte_identical():
    raw = "แก้ดาวนโหลด แต่คง `ดาวนโหลด` และ ```txt\nดาวนโหลด\n```"
    expected = "แก้ดาวน์โหลด แต่คง `ดาวนโหลด` และ ```txt\nดาวนโหลด\n```"
    assert thai_eval.dict_fix(raw) == expected


def test_urls_and_paths_are_byte_identical():
    raw = ("แก้ดาวนโหลด https://example.test/ดาวนโหลด "
           "C:\\tmp\\ดาวนโหลด.txt /tmp/ดาวนโหลด.txt")
    expected = ("แก้ดาวน์โหลด https://example.test/ดาวนโหลด "
                "C:\\tmp\\ดาวนโหลด.txt /tmp/ดาวนโหลด.txt")
    assert thai_eval.dict_fix(raw) == expected


def test_json_and_tool_payloads_are_byte_identical():
    payloads = [
        '{"message":"ดาวนโหลด"}',
        '"ดาวนโหลด"',
        '<tool_call>{"path":"ดาวนโหลด"}</tool_call>',
    ]
    for raw in payloads:
        assert thai_eval.dict_fix(raw) == raw


def test_expected_text_oracle_counts_fixes_and_false_positives():
    rows = [
        {"text": "ดาวนโหลด", "expected": "ดาวน์โหลด"},
        {"text": "ไฟล์", "expected": "ไฟล์"},
    ]
    result = thai_eval.evaluate_expected(rows)
    assert result == {
        "rows": 2,
        "rows_fixed": 1,
        "rows_worse": 0,
        "rows_unresolved": 0,
        "exact_after": 2,
    }


def test_expected_text_oracle_fails_loudly_without_gold_rows():
    import pytest

    with pytest.raises(ValueError, match="no rows"):
        thai_eval.evaluate_expected([])
    with pytest.raises(ValueError, match="expected"):
        thai_eval.evaluate_expected([{"text": "ดาวนโหลด"}])


def test_cli_rejects_rows_without_expected_text(tmp_path):
    inp = tmp_path / "rows.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text(json.dumps({"text": "ดาวนโหลด"}, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(TUNING / "tools" / "thai-eval-offline.py"),
         "--in", str(inp), "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode != 0
    assert "expected" in (proc.stdout + proc.stderr)


def test_cli_writes_auditable_expected_and_repaired_text(tmp_path):
    inp = tmp_path / "rows.jsonl"
    out = tmp_path / "out.jsonl"
    source = {"id": "s1-r1", "text": "ดาวนโหลด", "expected": "ดาวน์โหลด"}
    inp.write_text(json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(TUNING / "tools" / "thai-eval-offline.py"),
         "--in", str(inp), "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    repaired = next(row for row in rows if row.get("arm") == "dict" and "output" in row)
    assert repaired == {
        "arm": "dict", "id": "s1-r1", "text": "ดาวนโหลด",
        "expected": "ดาวน์โหลด", "output": "ดาวน์โหลด",
        "changed": True, "exact": True,
    }
