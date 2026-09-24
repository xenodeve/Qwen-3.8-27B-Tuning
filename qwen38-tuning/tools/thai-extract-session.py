"""Extract Thai prose spans from a Claude Code session transcript.

Reads a session .jsonl, collects assistant text blocks, strips fenced/inline
code, and writes one JSONL row per block with Thai text: raw text, Thai-char
count, broken-pattern hits. Offline, no GPU, stdout ASCII-only summary.

Usage: python tools/thai-extract-session.py --in <session.jsonl> --out bench/results/thai-session-<stamp>.jsonl
"""
import argparse
import datetime
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import importlib.util as _iu
_spec = _iu.spec_from_file_location(
    "thai_sampler_pair", os.path.join(ROOT, "tools", "thai-sampler-pair.py"))
_pair = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_pair)

THAI = re.compile(r"[ก-๛]")
FENCED = re.compile(r"```.*?```", re.DOTALL)
INLINE = re.compile(r"`[^`]+`")


def strip_code(text):
    text = FENCED.sub(" ", text or "")
    return INLINE.sub(" ", text)


def broken_count(text):
    words = sum(len(re.findall(p, text or "")) for p, _ in _pair.BROKEN_PATTERNS)
    latin = len(_pair.LATIN_IN_THAI.findall(text or ""))
    return words + latin


def iter_assistant_texts(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("type") != "assistant":
                continue
            content = (m.get("message") or {}).get("content", [])
            if not isinstance(content, list):
                continue
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    yield b.get("text") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = a.out or os.path.join(ROOT, "bench", "results", f"thai-session-{stamp}.jsonl")
    rows, agg = [], {"blocks": 0, "thai_blocks": 0, "thai_chars": 0, "broken": 0}
    for i, text in enumerate(iter_assistant_texts(a.inp)):
        prose = strip_code(text)
        n_thai = len(THAI.findall(prose))
        if n_thai < 20:
            continue
        b = broken_count(prose)
        agg["blocks"] += 1
        agg["thai_blocks"] += 1
        agg["thai_chars"] += n_thai
        agg["broken"] += b
        rows.append({"src": "session", "block": i, "thai_chars": n_thai,
                     "broken": b, "content_len": len(text), "text": prose})
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"results": out_path, "agg": agg}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
