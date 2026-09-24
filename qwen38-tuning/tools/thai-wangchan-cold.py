"""Cold test: WangchanBERTa fill-mask on Qwen-broken Thai sentences.

Mask placement: each BROKEN pattern hit replaced by ONE <mask> (word-level
repair hypothesis). Reports top-3 per mask with scores + latency. No dict,
no normalize -- pure model signal. Stdout ASCII-safe (scores + verdicts);
full texts to --out jsonl (auditable).

Usage: python tools/thai-wangchan-cold.py [--out bench/results/thai-wangchan-cold-<stamp>.jsonl]
"""
import argparse
import datetime
import json
import os
import re
import sys
import time

CASES = [
    ("s1-miss", "การดาวนโหลดวิดีโอจาก YouTube ควรทำอย่างไร"),
    ("s2-miss", "เปิดไฟลในโฟลเดอรแล้วส่งลิงกให้เพื่อน"),
    ("s3-mix", "ผมจะสแกนโครงส้างโปรเจกตก่อนแล่วบอกร้่องคณภาพ"),
    ("s4-doub", "การเขีย็นเป็็นยังไงบ้าง"),
    ("s5-doub", "ไม่่เป็็น git repo ครับ"),
    ("s6-loopfrag", "ผมจะดดู index.html และไฟลลสคริปต์"),
    ("s7-clean", "การดาวน์โหลดวิดีโอจาก YouTube ควรทำอย่างไร"),
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import importlib.util as _iu
_spec = _iu.spec_from_file_location(
    "thai_sampler_pair", os.path.join(ROOT, "tools", "thai-sampler-pair.py"))
_pair = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_pair)


def mask_case(text):
    # One mask per case (pipeline allows exactly one): first BROKEN hit, else
    # the first doubled-mark run (doubling class has no BROKEN pattern).
    for pat, _ in _pair.BROKEN_PATTERNS:
        m = re.search(pat, text)
        if m:
            return text[:m.start()] + " <mask> " + text[m.end():], 1
    m = re.search(r"[่-์]{2,}", text)
    if m:
        return text[:m.start()] + " <mask> " + text[m.end():], 1
    return text, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    from transformers import pipeline
    t0 = time.perf_counter()
    fill = pipeline(task="fill-mask",
                    model="airesearch/wangchanberta-base-att-spm-uncased",
                    tokenizer="airesearch/wangchanberta-base-att-spm-uncased")
    load_s = round(time.perf_counter() - t0, 1)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = a.out or os.path.join(ROOT, "bench", "results",
                                     f"thai-wangchan-cold-{stamp}.jsonl")
    summary = {"model": "airesearch/wangchanberta-base-att-spm-uncased",
               "load_s": load_s, "cases": []}
    import unicodedata as _ud
    open(out_path, "w", encoding="utf-8").close()
    for cid, text in CASES:
            masked, n = mask_case(text)
            if n == 0:
                row = {"id": cid, "input": text, "masked": masked,
                       "masks": 0, "ms": 0.0, "top": [],
                       "note": "no maskable hit; skipped"}
                with open(out_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary["cases"].append({"id": cid, "masks": 0, "ms": 0.0,
                                        "top1": None, "skipped": True})
                continue
            masked = _ud.normalize("NFC", masked)
            t1 = time.perf_counter()
            res = fill(masked)
            ms = round((time.perf_counter() - t1) * 1000, 1)
            if isinstance(res, dict):
                res = [res]
            elif res and isinstance(res[0], list):
                res = res[0]  # one list per mask; single-mask case
            top = [{"tok": r.get("token_str", ""), "score": round(float(r.get("score", 0)), 4)}
                   for r in res[:a.top]]
            row = {"id": cid, "input": text, "masked": masked,
                   "masks": n, "ms": ms, "top": top}
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary["cases"].append({"id": cid, "masks": n, "ms": ms,
                                    "top1": top[0] if top else None})
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    sys.exit(main())
