"""Replay problem sessions at depth: full history as context, new Thai ask at end.

Uses ONLY the two problem sessions (096ebcae EXL3 + 3258ccec FP8) -- their
own blocks, concatenated oldest-first as conversation history, with a fresh
Thai summary ask naming the same risk words. Prompt reaches ~12k tokens of
real session Thai (not synthetic filler), the depth regime where breaks
reproduced. Body: temp 1.0 passthrough (transcript regime).

Usage: python tools/thai-replay-depth.py [--base ...] [--rounds N]
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
S1 = os.path.join(RESULTS, "thai-session-20260916-024231.jsonl")
S2 = os.path.join(RESULTS, "thai-session-20260916-042801.jsonl")
SLEEP = 2.0

ASK = ("สรุปบทเรียนเป็นภาษาไทย 8-10 ประโยค เกี่ยวกับการดาวน์โหลดวิดีโอ "
       "การสำรองไฟล์งาน เว็บไซต์ที่แจกลิงก์ โฟลเดอร์จัดเก็บ และขั้นตอนต่อไป")


def load_history():
    hist = []
    for path in (S1, S2):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("text") or ""
            if len(t) > 24500:
                t = t[:2000]  # block 8 loop: keep head, drop the 24KB tail
            hist.append(t)
    return hist


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
            out.get("usage", {}).get("completion_tokens"),
            out.get("usage", {}).get("prompt_tokens"))


def broken_count(text):
    words = sum(len(re.findall(p, text or ""))
                for p, _ in thai_sampler_pair.BROKEN_PATTERNS)
    latin = len(thai_sampler_pair.LATIN_IN_THAI.findall(text or ""))
    return words + latin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    hist = load_history()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = a.out or os.path.join(RESULTS, f"thai-replay-{stamp}.jsonl")
    agg = {"n": 0, "prompt_toks": [], "thai": 0, "raw": 0, "fixed": 0,
           "rows_fixed": 0, "rows_worse": 0}
    for r in range(a.rounds):
        messages = ([{"role": "user", "content": h} for h in hist]
                    + [{"role": "user", "content": ASK}])
        t0 = time.time()
        text, finish, out_toks, prompt_toks = call(a.base, messages)
        fixed = thai_eval.dict_fix(text)
        b0, b1 = broken_count(text), broken_count(fixed)
        thai = len(thai_sampler_pair.THAI.findall(text))
        agg["n"] += 1
        agg["prompt_toks"].append(prompt_toks)
        agg["thai"] += thai
        agg["raw"] += b0
        agg["fixed"] += b1
        agg["rows_fixed"] += b1 < b0
        agg["rows_worse"] += b1 > b0
        row = {"stamp": stamp, "round": r, "finish": finish,
               "prompt_toks": prompt_toks, "out_toks": out_toks,
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
