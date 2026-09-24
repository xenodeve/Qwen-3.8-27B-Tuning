"""Cold test: Typhoon-1B repair on Qwen-broken Thai sentences (local CPU).

Backend: llama-cli (llama.cpp-blackwell) on CPU threads, Q2_K GGUF.
Prompt: fix-Thai-only instruction, temp 0.0 greedy. Reports input, output,
latency + verdict columns (same 7 cases as the WangchanBERTa cold test).
Stdout ASCII-safe summary; full rows to --out jsonl (auditable).

Usage: python tools/thai-typhoon-cold.py [--bin ...] [--model ...] [--threads 8]
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_CLI = r"C:\AI\llama.cpp-blackwell\llama-cli.exe"

CASES = [
    ("s1-miss", "การดาวนโหลดวิดีโอจาก YouTube ควรทำอย่างไร", "การดาวน์โหลดวิดีโอจาก YouTube ควรทำอย่างไร"),
    ("s2-miss", "เปิดไฟลในโฟลเดอรแล้วส่งลิงกให้เพื่อน", "เปิดไฟล์ในโฟลเดอร์แล้วส่งลิงก์ให้เพื่อน"),
    ("s3-mix", "ผมจะสแกนโครงส้างโปรเจกตก่อนแล่วบอกร้่องคณภาพ",
     "ผมจะสแกนโครงสร้างโปรเจกต์ก่อนแล้วบอกเรื่องคุณภาพ"),
    ("s4-doub", "การเขีย็นเป็็นยังไงบ้าง", "การเขียนเป็นยังไงบ้าง"),
    ("s5-doub", "ไม่่เป็็น git repo ครับ", "ไม่เป็น git repo ครับ"),
    ("s6-loopfrag", "ผมจะดดู index.html และไฟลลสคริปต์", "ผมจะดู index.html และไฟล์สคริปต์"),
    ("s7-clean", "การดาวน์โหลดวิดีโอจาก YouTube ควรทำอย่างไร",
     "การดาวน์โหลดวิดีโอจาก YouTube ควรทำอย่างไร"),
]

PROMPT = ("Fix Thai spelling only. Output ONLY the corrected sentence, nothing else.\n"
          "Do not translate, do not explain, do not change code, paths, URLs or English words.\n"
          "Input: {s}\nOutput:")


def run(binpath, model, text, threads, timeout=300):
    """One llama-cli call per case, stdin DEVNULL so it can never enter
    interactive mode (the 2026-09-16 RAM spike: a second -p call waited on
    stdin and held memory). Sequential only -- never parallel."""
    import subprocess as _sp
    t0 = time.perf_counter()
    p = _sp.run(
        [binpath, "-m", model, "-p", PROMPT.format(s=text),
         "--temp", "0.0", "-n", "128", "-t", str(threads),
         "--no-display-prompt", "-c", "1024"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        stdin=_sp.DEVNULL, timeout=timeout)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    out = (p.stdout or "").strip().splitlines()
    out = [l for l in out if l.strip()]
    return (out[-1].strip() if out else "", ms, p.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=LLAMA_CLI)
    ap.add_argument("--model", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.model:
        print(json.dumps({"error": "pass --model <typhoon1b.gguf>"}))
        return 1
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = a.out or os.path.join(ROOT, "bench", "results",
                                     f"thai-typhoon-cold-{stamp}.jsonl")
    summary = {"model": os.path.basename(a.model), "threads": a.threads, "cases": []}
    with open(out_path, "w", encoding="utf-8") as fh:
        for cid, text, want in CASES:
            out, ms, rc = run(a.bin, a.model, text, a.threads)
            row = {"id": cid, "input": text, "want": want, "output": out,
                   "match": out == want, "ms": ms, "rc": rc}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary["cases"].append({"id": cid, "match": out == want,
                                    "ms": ms, "out": out})
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    sys.exit(main())
