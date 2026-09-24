"""Offline A/B: pythainlp normalize/remove_repeat_vowels on stored Thai rows.

No GPU, no server restart. Reads qwen38-tuning/bench/results/thai-*.jsonl
(rows carry `text`), counts BROKEN_PATTERNS (single-sourced from
tools/thai-sampler-pair.py) before/after each function, and checks that
ASCII code spans survive byte-identical.

Usage:
  python tools/thai-normalize-ab.py [--results bench/results] [--out bench/results/thai-normalize-ab-<stamp>.jsonl]
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys
import time

import pythainlp
from pythainlp.util import normalize, remove_repeat_vowels

PYTHAINLP_VERSION = pythainlp.__version__

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import importlib.util as _iu
_spec = _iu.spec_from_file_location(
    "thai_sampler_pair", os.path.join(ROOT, "tools", "thai-sampler-pair.py"))
_pair = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_pair)
BROKEN_PATTERNS = _pair.BROKEN_PATTERNS
LATIN_IN_THAI = _pair.LATIN_IN_THAI

# Code spans, derived by shape rather than by filename list: backtick spans,
# double-dash flags, and ASCII runs holding a path/code separator (./:).
# A bare number range like 8-10 in Thai prose is not code and must not match.
CODE_RUN = re.compile(
    r"`[^`]+`|--[A-Za-z][A-Za-z0-9\-]*"
    r"|[A-Za-z0-9_]*[./:][A-Za-z0-9_./:\-]*")


def run_normalize(text):
    return normalize(text)


def run_repeat(text):
    return remove_repeat_vowels(text)


def code_runs(text):
    """ASCII spans that look like code (paths, filenames, URLs, JSON, flags)."""
    return [m.group(0) for m in CODE_RUN.finditer(text or "")]


def code_violations(raw, fixed):
    """Raw code spans missing from the fixed text -- a repair that ate code."""
    return [r for r in code_runs(raw) if r not in (fixed or "")]


def broken_count(text):
    words = sum(len(re.findall(p, text or "")) for p, _ in BROKEN_PATTERNS)
    latin = len(LATIN_IN_THAI.findall(text or ""))
    return words + latin


def ab_row(text):
    t0 = time.perf_counter()
    norm = run_normalize(text)
    ms_norm = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    rep = run_repeat(text)
    ms_rep = (time.perf_counter() - t0) * 1000.0
    return {
        "text": text,
        "raw_broken": broken_count(text),
        "norm_broken": broken_count(norm),
        "rep_broken": broken_count(rep),
        "changed_norm": norm != text,
        "changed_rep": rep != text,
        "code_violations": code_violations(text, norm),
        "code_violations_rep": code_violations(text, rep),
        "ms_norm": round(ms_norm, 2),
        "ms_rep": round(ms_rep, 2),
        "pythainlp_version": PYTHAINLP_VERSION,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "bench", "results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = a.out or os.path.join(a.results, f"thai-normalize-ab-{stamp}.jsonl")
    rows, agg = [], {"n": 0, "raw": 0, "norm": 0, "rep": 0,
                      "changed_norm": 0, "changed_rep": 0,
                      "code_hits": 0, "ms_norm": 0.0, "ms_rep": 0.0}
    for path in sorted(glob.glob(os.path.join(a.results, "thai-*.jsonl"))):
        if os.path.basename(path).startswith("thai-normalize-ab-"):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = r.get("text")
                if not text:
                    continue
                row = ab_row(text)
                row["src"] = os.path.basename(path)
                rows.append(row)
                agg["n"] += 1
                agg["raw"] += row["raw_broken"]
                agg["norm"] += row["norm_broken"]
                agg["rep"] += row["rep_broken"]
                agg["changed_norm"] += row["changed_norm"]
                agg["changed_rep"] += row["changed_rep"]
                agg["code_hits"] += bool(row["code_violations"])
                agg["ms_norm"] += row["ms_norm"]
                agg["ms_rep"] += row["ms_rep"]
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    agg["ms_norm"] = round(agg["ms_norm"] / max(agg["n"], 1), 2)
    agg["ms_rep"] = round(agg["ms_rep"] / max(agg["n"], 1), 2)
    print(json.dumps({"results": out_path,
                      "pythainlp": PYTHAINLP_VERSION, "agg": agg},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
