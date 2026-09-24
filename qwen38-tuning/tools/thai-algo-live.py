"""Algorithm-only live test: fresh Thai prompts via G, dict scored offline.

Sends NEW Thai prompts (never in the eval sessions) through :8000 at the
transcript regime (multiturn + tool seed + temp 1.0 passthrough, the only
shape that reproduced), then scores raw vs dict_fix offline. No server
change, body-only knobs.

Usage: python tools/thai-algo-live.py [--base http://127.0.0.1:8000] [--rounds N]
"""
import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import importlib.util as _iu
for _name, _file in [("thai_sampler_pair", "thai-sampler-pair.py"),
                     ("thai_eval", "thai-eval-offline.py")]:
    _spec = _iu.spec_from_file_location(_name, os.path.join(ROOT, "tools", _file))
    _mod = _iu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    globals()[_name] = _mod

RESULTS = os.path.join(ROOT, "bench", "results")
SLEEP = 2.0

# Fresh topics: same risk words, new framings (not in TOPIC_NOUNS).
FRESH = [
    "การดาวน์โหลดไฟล์วิดีโอและการจัดเก็บในโฟลเดอร์ให้เป็นระเบียบ",
    "เว็บไซต์ที่แจกลิงก์ดาวน์โหลดกับข้อควรระวังเกี่ยวกับลิขสิทธิ์",
    "ขั้นตอนต่อไปหลังดาวน์โหลดเสร็จเกี่ยวกับการสำรองไฟล์งาน",
    "การแชร์ลิงก์ไฟล์วิดีโอในโฟลเดอร์ทีมเกี่ยวกับมารยาทการใช้งาน",
]


def build_messages(topic):
    return [
        {"role": "system", "content": thai_sampler_pair.SYSTEM_EN},
        {"role": "user", "content": "เช็คให้หน่อยว่า dependencies ครบไหม"},
        {"role": "tool", "content": thai_sampler_pair.SEED_BLOCK},
        {"role": "user", "content": "สรุปผลเป็นภาษาไทย 8-10 ประโยค เกี่ยวกับ" + topic},
    ]


def call(base, messages):
    body = {"model": "qwen3.8-27b-exl3", "messages": messages,
            "max_tokens": 900, "temperature": 1.0, "top_p": 0.95,
            "top_k": 20, "reasoning_effort": "medium", "thai_sampler": 0}
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read().decode("utf-8"))
    choice = out["choices"][0]
    return (choice["message"].get("content") or "", choice.get("finish_reason"),
            out.get("usage", {}).get("completion_tokens"))


def broken_count(text):
    words = sum(len(re.findall(p, text or ""))
                for p, _ in thai_sampler_pair.BROKEN_PATTERNS)
    latin = len(thai_sampler_pair.LATIN_IN_THAI.findall(text or ""))
    return words + latin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = a.out or os.path.join(RESULTS, f"thai-algo-live-{stamp}.jsonl")
    agg = {"n": 0, "thai": 0, "raw": 0, "fixed": 0,
           "rows_fixed": 0, "rows_worse": 0}
    for r in range(a.rounds):
        for i, topic in enumerate(FRESH):
            t0 = time.time()
            text, finish, out_toks = call(a.base, build_messages(topic))
            fixed = thai_eval.dict_fix(text)
            b0, b1 = broken_count(text), broken_count(fixed)
            thai = len(thai_sampler_pair.THAI.findall(text))
            agg["n"] += 1
            agg["thai"] += thai
            agg["raw"] += b0
            agg["fixed"] += b1
            agg["rows_fixed"] += b1 < b0
            agg["rows_worse"] += b1 > b0
            row = {"stamp": stamp, "round": r, "prompt": i,
                   "finish": finish, "out_toks": out_toks,
                   "content_len": len(text), "thai_chars": thai,
                   "raw_broken": b0, "dict_broken": b1,
                   "dict_changed": fixed != text,
                   "seconds": round(time.time() - t0, 1), "text": text}
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            time.sleep(SLEEP)
    print(json.dumps({"results": out_path, "agg": agg}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
