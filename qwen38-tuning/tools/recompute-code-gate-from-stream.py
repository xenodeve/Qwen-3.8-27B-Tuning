"""Recompute ONLY the two code-gate checks that come from the transcript, and rescore.

Why this exists rather than `--regate`. `--regate` re-runs the hidden tests inside the
cell's work directory, and `WORK_ROOT` is keyed by cell name alone -- so
`D:\\qbench-work\\A-code2-codegate-r1` is shared by every results folder that has a cell
with that name. On 2026-09-07 a regate of the 2026-09-05 baseline read a work tree that a
run started minutes earlier was still writing, and scored a published cell 3/5 against
another day's half-finished checkout.

`test_runs_seen` and `red_then_green` are derived from `stream.jsonl` alone, which is
immutable once the cell is done. Those are exactly the two the detector fix (CORRECTIONS
48) changed, so this recomputes them and nothing else, and rebuilds `score` from the
stored values of the other three checks.

    python qwen38-tuning\\tools\\recompute-code-gate-from-stream.py            # dry run
    python qwen38-tuning\\tools\\recompute-code-gate-from-stream.py --write
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("qb", os.path.join(HERE, "quality-bench.py"))
qb = importlib.util.module_from_spec(_spec)
sys.modules["qb"] = qb
_spec.loader.exec_module(qb)


def rescore(gate):
    """The same five checks code_gate scores, from stored values."""
    checks = [
        gate.get("scope_ok"),
        not gate.get("placeholders"),
        (gate.get("test_runs_seen") or 0) > 0,
        gate.get("hidden_failed") == 0 and (gate.get("hidden_passed") or 0) > 0,
        gate.get("red_then_green"),
    ]
    return f"{sum(1 for c in checks if c)}/5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(
        os.path.dirname(HERE), "results", "quality-2026-09-05"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    changed = 0
    for sd in sorted(glob.glob(os.path.join(a.results, "*-r*"))):
        sj, st = os.path.join(sd, "summary.json"), os.path.join(sd, "stream.jsonl")
        if not (os.path.exists(sj) and os.path.exists(st)):
            continue
        s = json.load(open(sj, encoding="utf-8"))
        if not str(s.get("task", "")).startswith("code"):
            continue
        g = s.get("gate")
        if not g:
            continue
        outs = qb.test_command_outputs(st)
        before = (g.get("test_runs_seen"), g.get("red_then_green"), g.get("score"))
        g["test_runs_seen"] = len(outs)
        g["red_then_green"] = qb.red_before_green(outs)
        g["score"] = rescore(g)
        after = (g["test_runs_seen"], g["red_then_green"], g["score"])
        if before != after:
            changed += 1
            print(f"{os.path.basename(sd):40} runs {before[0]}->{after[0]}  "
                  f"rtg {before[1]}->{after[1]}  score {before[2]}->{after[2]}")
        if a.write:
            json.dump(s, open(sj, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n{changed} cell(s) changed. {'WRITTEN' if a.write else 'dry run -- pass --write'}")


if __name__ == "__main__":
    main()
