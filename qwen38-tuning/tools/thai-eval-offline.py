"""Offline eval: precision-gated Thai repair arms against gold text.

Input rows require `text` and independently supplied `expected` strings.
Arms: raw | normalize | dict | symspell(d<=1, allowlist) | dict+symspell.
Metrics per arm: exact rows, rows fixed, rows worsened (FP), unresolved rows,
ms_cpu_poststep_row (CPU-only regex/dict cost per block -- independent of the
serving ctx; NOT model latency, do not compare with tok/s or TTFT).
Empty, malformed, or non-gold inputs fail instead of returning a summary.
Output JSONL carries raw, expected, and repaired text for every arm.

Usage: python tools/thai-eval-offline.py --in <gold.jsonl> [--out ...]
"""
import argparse
import datetime
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Retained only to reproduce the invalid historical metric in correction 52.
# It is not the precision oracle; evaluate_expected() is.
BROKEN_PATTERNS = [
    r"ดาวนโหลด", r"วดีโอ", r"เว็บไซต(?!์)", r"ลิงค(?!์)",
    r"แลว(?![่-๋แ])", r"หรื(?!อ)", r"ไฟล(?!์)", r"โฟลเดอร(?!์)",
    r"ตอไป", r"เกียวกับ", r"เครื่่อง",
]
LATIN_IN_THAI = re.compile(r"[ก-๛][A-Za-z][ก-๛]")

from pythainlp.util import normalize as _normalize

# Issue #89 precision subset. Each malformed form occurs in zero entries of
# PyThaiNLP 5.3.7's 62,101-word thai_words() corpus. Ambiguous short forms from
# v2 deliberately return raw rather than guessing.
DICT = {b: f for b, f in [
    ("ดาวนโหลด", "ดาวน์โหลด"), ("วดีโอ", "วิดีโอ"),
    ("ตอไป", "ต่อไป"), ("เกียวกับ", "เกี่ยวกับ"), ("เครื่่อง", "เครื่อง"),
    ("เปิิด", "เปิด"), ("แล่ว", "แล้ว"),
    ("ร้่อง", "เรื่อง"), ("คณภาพ", "คุณภาพ"), ("ใหคณฟง", "ให้คุณฟัง"),
    ("โครงส้าง", "โครงสร้าง"), ("ไม่่", "ไม่"),
    ("เป็็น", "เป็น"), ("ไฟลล", "ไฟล์"),
    ("ตดัสืนใจ", "ตัดสินใจ"), ("ขางเคยีง", "ข้างเคียง"),
    ("ยััง", "ยัง"), ("ว่่า", "ว่า"), ("ท้้้ง", "ทั้ง"), ("อย่่่าง", "อย่าง"),
    ("ท่ี", "ที่"), ("ชื่o", "ชื่อ"), ("นั้n", "นั้น"), ("เพื่o", "เพื่อ"),
    ("เริ่่ม", "เริ่ม"), ("สำหรบ", "สำหรับ"), ("จำเปน", "จำเป็น"),
    ("ถกตอง", "ถูกต้อง"), ("สดท้้าย", "สุดท้าย"),
]}
def _dict_pattern(broken):
    fixed = DICT[broken]
    suffix = fixed[len(broken):] if fixed.startswith(broken) else ""
    guard = f"(?!{re.escape(suffix)})" if suffix else ""
    return re.escape(broken) + guard


DICT_RE = re.compile("|".join(
    _dict_pattern(broken) for broken in sorted(DICT, key=len, reverse=True)))
PROTECTED_RE = re.compile(
    r"```.*?```|`[^`\n]*`|https?://[^\s<>()]+|[A-Za-z]:\\[^\s`]+|"
    r"(?:\./|\.\./|/)[^\s`]+",
    re.DOTALL)

_sym = None


def get_sym():
    global _sym
    if _sym is None:
        from symspellpy import SymSpell
        from pythainlp.corpus import tnc
        _sym = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
        for w, f in tnc.word_freqs():
            _sym.create_dictionary_entry(w, f)
    return _sym


ALLOW = set(DICT.values())


def sym_fix(text):
    s = get_sym()
    out = []
    for tok in re.findall(r"[ก-๛]+|.", text or ""):
        if not re.fullmatch(r"[ก-๛]+", tok):
            out.append(tok)
            continue
        r = s.lookup(tok, 1, include_unknown=True)
        if r and r[0].distance <= 1 and r[0].term in ALLOW:
            out.append(r[0].term)
        else:
            out.append(tok)
    return "".join(out)


def dict_fix(text):
    text = text or ""
    stripped = text.strip()
    if stripped.startswith(("<tool_call>", "<function=")):
        return text
    try:
        json.loads(stripped)
        return text
    except (json.JSONDecodeError, TypeError):
        pass
    out = []
    start = 0
    for match in PROTECTED_RE.finditer(text):
        out.append(DICT_RE.sub(lambda m: DICT[m.group(0)], text[start:match.start()]))
        out.append(match.group(0))
        start = match.end()
    out.append(DICT_RE.sub(lambda m: DICT[m.group(0)], text[start:]))
    return "".join(out)


def broken_count(text):
    """Historical incident counter only; never use it as a precision oracle."""
    words = sum(len(re.findall(pattern, text or ""))
                for pattern in BROKEN_PATTERNS)
    latin = len(LATIN_IN_THAI.findall(text or ""))
    return words + latin


def evaluate_expected(rows, repair=dict_fix):
    if not rows:
        raise ValueError("no rows to evaluate")
    result = {"rows": len(rows), "rows_fixed": 0, "rows_worse": 0,
              "rows_unresolved": 0, "exact_after": 0}
    for index, row in enumerate(rows):
        if (not isinstance(row, dict)
                or not isinstance(row.get("text"), str)
                or not isinstance(row.get("expected"), str)):
            raise ValueError(f"row {index} requires string text and expected")
        raw, expected = row["text"], row["expected"]
        fixed = repair(raw)
        before_ok = raw == expected
        after_ok = fixed == expected
        result["exact_after"] += after_ok
        result["rows_fixed"] += not before_ok and after_ok
        result["rows_worse"] += before_ok and not after_ok
        result["rows_unresolved"] += not before_ok and not after_ok
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = a.out or os.path.join(ROOT, "bench", "results", f"thai-eval-{stamp}.jsonl")
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    evaluate_expected(rows)
    agg, output_rows = {}, []
    arms = [("raw", lambda t: t), ("normalize", _normalize),
            ("dict", dict_fix), ("symspell", sym_fix),
            ("dict_sym", lambda t: sym_fix(dict_fix(t)))]
    for name, fn in arms:
        t0 = time.perf_counter()
        summary = evaluate_expected(rows, fn)
        ms = (time.perf_counter() - t0) * 1000 / len(rows)
        agg[name] = {**summary, "ms_cpu_poststep_row": round(ms, 2)}
        for index, row in enumerate(rows):
            fixed = fn(row["text"])
            output_rows.append({
                "arm": name,
                "id": row.get("id", index),
                "text": row["text"],
                "expected": row["expected"],
                "output": fixed,
                "changed": fixed != row["text"],
                "exact": fixed == row["expected"],
            })
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in output_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.write(json.dumps({"kind": "summary", "arms": agg}, ensure_ascii=False) + "\n")
    print(json.dumps({"results": out_path, "rows": len(rows), "arms": agg},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
