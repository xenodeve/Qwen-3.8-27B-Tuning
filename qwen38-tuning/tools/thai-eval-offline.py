"""Offline eval: dict + SymSpell arms on session blocks that actually break.

Input: bench/results/thai-session-<stamp>.jsonl (43 blocks, 36 broken).
Arms: raw | normalize | dict | symspell(d<=1, allowlist) | dict+symspell.
Metrics per arm: broken count, rows fixed, rows worsened (FP),
ms_cpu_poststep_row (CPU-only regex/dict cost per block -- independent of the
serving ctx; NOT model latency, do not compare with tok/s or TTFT).
Stdout ASCII-only summary; rows JSONL with text (auditable).

Usage: python tools/thai-eval-offline.py --in bench/results/thai-session-<stamp>.jsonl [--out ...]
"""
import argparse
import datetime
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import importlib.util as _iu
_spec = _iu.spec_from_file_location(
    "thai_sampler_pair", os.path.join(ROOT, "tools", "thai-sampler-pair.py"))
_pair = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_pair)
BROKEN_PATTERNS = _pair.BROKEN_PATTERNS
LATIN_IN_THAI = _pair.LATIN_IN_THAI

from pythainlp.util import normalize as _normalize

# v1 (11 pairs): session 096ebcae missing-vowel class + transcript loop word.
# v2 (2026-09-16, +24): session 3258ccec (FP8) doubling class + 096ebcae
# leftovers. Every fixed form verified in thai_words() except ไม่มี/ไม่-มี
# (wordlist stores ไม่ glued: ไม่มีที่ติ etc. -- kept: ไม่+มี both in TNC
# top pairs and the fix is unambiguous in context).
DICT = {b: f for b, f in [
    ("ดาวนโหลด", "ดาวน์โหลด"), ("วดีโอ", "วิดีโอ"), ("เว็บไซต", "เว็บไซต์"),
    ("ลิงก", "ลิงก์"), ("ไฟล", "ไฟล์"), ("โฟลเดอร", "โฟลเดอร์"),
    ("ตอไป", "ต่อไป"), ("เกียวกับ", "เกี่ยวกับ"), ("เครื่่อง", "เครื่อง"),
    ("แลว", "แล้ว"), ("หรื", "หรือ"),
    ("เปิิด", "เปิด"), ("โปรเจกต", "โปรเจกต์"), ("แล่ว", "แล้ว"),
    ("ร้่อง", "เรื่อง"), ("คณภาพ", "คุณภาพ"), ("ใหคณฟง", "ให้คุณฟัง"),
    ("สแกน", "สแกน"), ("โครงส้าง", "โครงสร้าง"), ("ไม่่", "ไม่"),
    ("เป็็น", "เป็น"), ("ดดู", "ดู"), ("ไฟลล", "ไฟล์"),
    ("กอน", "ก่อน"), ("ตดัสืนใจ", "ตัดสินใจ"), ("จิง", "จริง"),
    ("งาย", "ง่าย"), ("ผาน", "ผ่าน"),
    ("จดุ", "จุด"),
    ("ไมม", "ไม่มี"), ("ขางเคยีง", "ข้างเคียง"), ("ชอง", "ช่อง"),
    ("รุป", "รูป"),
    ("ยััง", "ยัง"), ("ว่่า", "ว่า"), ("ท้้้ง", "ทั้ง"), ("อย่่่าง", "อย่าง"),
    ("ไซ่", "ใช่"), ("ตอง", "ต้อง"), ("ท่ี", "ที่"), ("ชื่o", "ชื่อ"),
    ("นั้n", "นั้น"), ("เพื่o", "เพื่อ"), ("เริ่่ม", "เริ่ม"),
    ("ตัง", "ตั้ง"), ("สำหรบ", "สำหรับ"), ("จำเปน", "จำเป็น"),
    ("ถกตอง", "ถูกต้อง"), ("สดท้้าย", "สุดท้าย"),
]}
DICT_RE = re.compile("|".join(sorted(DICT, key=len, reverse=True)))

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
    return DICT_RE.sub(lambda m: DICT[m.group(0)], text or "")


def broken_count(text):
    words = sum(len(re.findall(p, text or "")) for p, _ in BROKEN_PATTERNS)
    latin = len(LATIN_IN_THAI.findall(text or ""))
    return words + latin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = a.out or os.path.join(ROOT, "bench", "results", f"thai-eval-{stamp}.jsonl")
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    agg = {}
    for name, fn in [("raw", lambda t: t), ("normalize", _normalize),
                     ("dict", dict_fix), ("symspell", sym_fix),
                     ("dict_sym", lambda t: sym_fix(dict_fix(t)))]:
        t0 = time.perf_counter()
        b0 = b1 = fixed = worse = 0
        for r in rows:
            before = broken_count(r["text"])
            after = broken_count(fn(r["text"]))
            b0 += before
            b1 += after
            if after < before:
                fixed += 1
            if after > before:
                worse += 1
        ms = (time.perf_counter() - t0) * 1000 / max(len(rows), 1)
        agg[name] = {"broken_before": b0, "broken_after": b1,
                     "rows_fixed": fixed, "rows_worse": worse,
                     "ms_cpu_poststep_row": round(ms, 2)}
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"arm": name, **agg[name]}, ensure_ascii=False) + "\n")
    print(json.dumps({"results": out_path, "rows": len(rows), "arms": agg},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())

