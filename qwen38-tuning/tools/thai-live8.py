"""Live 8-arm experiment on profile G: Thai repair stack at real session depth.

Reads thai-session rows (or --prompts), rebuilds the transcript's multiturn
shape (English tool-output prefix + long Thai summary ask), serves each arm
through :8000, and scores genuine breaks per 1k Thai chars with the fixed
counter (single-sourced from thai-sampler-pair).

Arms (body-only, no server change):
  A baseline    -- default (cap 0.6 on Thai-heavy)
  B minp05      -- min_p 0.05
  C minp10      -- min_p 0.10
  D t10pass     -- temperature 1.0 + thai_sampler 0 (passthrough)
  E t10minp10   -- temperature 1.0 + min_p 0.1 + thai_sampler 0
  F floatban    -- default (float ban rides unconditional in bias_for)
  G thaisamp    -- thai_sampler 1 (tighten triple 0.3/0.9/10)
  H dictfix     -- default + offline dict repair scored as separate columns

H is scored offline on every arm's text (raw_broken vs dict_broken), so the
dict arm costs no extra GPU rows.

Usage: python tools/thai-live8.py [--base http://127.0.0.1:8000]
         [--session bench/results/thai-session-<stamp>.jsonl] [--rounds N]
         [--dictfix-only]   # offline H columns on an existing live8 file
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import importlib.util as _iu
_spec = _iu.spec_from_file_location(
    "thai_sampler_pair", os.path.join(ROOT, "tools", "thai-sampler-pair.py"))
_pair = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_pair)
BROKEN_PATTERNS = _pair.BROKEN_PATTERNS
LATIN_IN_THAI = _pair.LATIN_IN_THAI

RESULTS = os.path.join(ROOT, "bench", "results")
SEED_BLOCK = _pair.SEED_BLOCK
SYSTEM_EN = _pair.SYSTEM_EN
LONG_ASK = "สรุปผลเป็นภาษาไทย 8-10 ประโยค เกี่ยวกับ{}"
TOPICS = _pair.TOPIC_NOUNS
SLEEP = 2.0

ARMS = [
    ("A", {}),
    ("B", {"min_p": 0.05}),
    ("C", {"min_p": 0.1}),
    ("D", {"temperature": 1.0, "thai_sampler": 0}),
    ("E", {"temperature": 1.0, "min_p": 0.1, "thai_sampler": 0}),
    ("F", {}),
    ("G", {"thai_sampler": 1}),
]

# Verified transcript pairs (arm 3 dict seed): broken -> fixed.
DICT = {b: f for b, f in [
    ("ดาวนโหลด", "ดาวน์โหลด"), ("วดีโอ", "วิดีโอ"), ("เว็บไซต", "เว็บไซต์"),
    ("ลิงก", "ลิงก์"), ("ไฟล", "ไฟล์"), ("โฟลเดอร", "โฟลเดอร์"),
    ("ตอไป", "ต่อไป"), ("เกียวกับ", "เกี่ยวกับ"), ("เครื่่อง", "เครื่อง"),
    ("แลว", "แล้ว"), ("หรื", "หรือ"),
]}
DICT_RE = re.compile("|".join(sorted(DICT, key=len, reverse=True)))


def dict_fix(text):
    return DICT_RE.sub(lambda m: DICT[m.group(0)], text or "")


def broken_count(text):
    words = sum(len(re.findall(p, text or "")) for p, _ in BROKEN_PATTERNS)
    latin = len(LATIN_IN_THAI.findall(text or ""))
    return words + latin


def call(base, messages, max_tokens, extra):
    body = {"model": "qwen3.8-27b-exl3", "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.6,
            "top_p": 0.95, "top_k": 20, "reasoning_effort": "medium"}
    body.update(extra)
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read().decode("utf-8"))
    choice = out["choices"][0]
    return (choice["message"].get("content") or "", choice.get("finish_reason"),
            out.get("usage", {}).get("completion_tokens"))


def build_messages(topic):
    msgs = [{"role": "system", "content": SYSTEM_EN},
            {"role": "user", "content": "เช็คให้หน่อยว่า dependencies ครบไหม"},
            {"role": "tool", "content": SEED_BLOCK},
            {"role": "user", "content": LONG_ASK.format(topic)}]
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--topics", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = a.out or os.path.join(RESULTS, f"thai-live8-{stamp}.jsonl")
    rows = []
    for r in range(a.rounds):
        for i in range(a.topics):
            topic = TOPICS[i % len(TOPICS)]
            messages = build_messages(topic)
            for arm, extra in ARMS:
                t0 = time.time()
                text, finish, out_toks = call(a.base, messages, 900, extra)
                fixed = dict_fix(text)
                rows.append({
                    "stamp": stamp, "round": r, "prompt": i, "arm": arm,
                    "extra": extra, "finish": finish, "out_toks": out_toks,
                    "content_len": len(text),
                    "raw_broken": broken_count(text),
                    "dict_broken": broken_count(fixed),
                    "dict_changed": fixed != text,
                    "thai_chars": len(_pair.THAI.findall(text)),
                    "seconds": round(time.time() - t0, 1),
                    "text": text,
                })
                with open(out_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                time.sleep(SLEEP)
    agg = {}
    for arm, _ in ARMS:
        sel = [x for x in rows if x["arm"] == arm]
        b = sum(x["raw_broken"] for x in sel)
        t = sum(x["thai_chars"] for x in sel)
        d = sum(x["dict_broken"] for x in sel)
        agg[arm] = {"raw": b, "thai": t, "per1k": round(b * 1000 / max(t, 1), 2),
                    "dict": d, "dict_per1k": round(d * 1000 / max(t, 1), 2)}
    print(json.dumps({"results": out_path, "arms": agg}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
