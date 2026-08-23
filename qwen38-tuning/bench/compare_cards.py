# -*- coding: utf-8 -*-
"""Compare two decoder sweeps arm-for-arm, and refuse the comparisons that lie.

WHY THIS IS A SCRIPT AND NOT A TABLE SOMEONE TYPES.

`docs/reports/CORRECTIONS.md` 28 is this project retracting a published
hardware verdict that was built by putting two correctly-measured numbers in one
table. Both were right. The table was not, because `ngram-mod` is a speculative
decoder whose rate tracks draft acceptance, and the two rows had acceptance 60.2
and 14.87. Nothing in either file said so; a human read two numbers and wrote a
ratio.

So this refuses to print a ratio when the two sides' acceptance differs by more
than a threshold, and prints why instead. It also refuses when the corpus hash
differs, and reports rather than hides an arm whose rows did not all complete --
a median over the survivors of an arm that timed out twice is not that arm's
rate, and `dflash2+ngram` on the 4070 SUPER is exactly that case: median 5.66
over 4 of 6 rows, with a max of 93.29.

Usage:
    python compare_cards.py BASELINE.jsonl CURRENT.jsonl
"""
import json
import statistics
import sys
from pathlib import Path

# Acceptance is a percentage in these files. Two arms whose speculation differs
# by more than this are not measuring the same thing, whatever else matches.
ACCEPTANCE_TOLERANCE_PCT = 5.0


def load(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarise(rows, arm):
    """Everything needed to decide whether this arm may be compared at all."""
    rs = [r for r in rows if r.get("arm") == arm]
    rates = [r["tg_med"] for r in rs if r.get("tg_med")]
    return {
        "n_rows": len(rs),
        "n_rates": len(rates),
        "median": statistics.median(rates) if rates else None,
        "lo": min(rates) if rates else None,
        "hi": max(rates) if rates else None,
        "acceptance": next((r.get("acceptance") for r in rs
                            if r.get("acceptance") is not None), None),
        "corpus": next((r.get("corpus") for r in rs if r.get("corpus")), None),
        "archs": next((tuple(r["cuda_archs"]) for r in rs
                       if r.get("cuda_archs")), None),
        "notes": sorted({r["note"] for r in rs if r.get("note")}),
    }


def spread_pct(s):
    """Peak-to-peak as a percentage of the median: the noise floor, per arm."""
    if s["median"] in (None, 0) or s["lo"] is None:
        return None
    return 100.0 * (s["hi"] - s["lo"]) / s["median"]


def compare(base_path, cur_path):
    base, cur = load(base_path), load(cur_path)
    arms = [a for a in dict.fromkeys(
        [r.get("arm") for r in base] + [r.get("arm") for r in cur]) if a]

    print(f"baseline  {base_path}   {len(base)} rows")
    print(f"current   {cur_path}   {len(cur)} rows")
    print()

    for arm in arms:
        b, c = summarise(base, arm), summarise(cur, arm)
        print(f"== {arm}")
        for tag, s in (("baseline", b), ("current ", c)):
            if s["median"] is None:
                print(f"   {tag}  no measurable rows ({s['n_rows']} attempted)"
                      f"  {s['notes']}")
                continue
            sp = spread_pct(s)
            print(f"   {tag}  median {s['median']:7.2f}  "
                  f"range {s['lo']:.2f}-{s['hi']:.2f}  "
                  f"spread {sp:5.1f}%  "
                  f"acc {s['acceptance']}  "
                  f"{s['n_rates']}/{s['n_rows']} rows  "
                  f"archs {s['archs']}")
            if s["notes"]:
                print(f"             notes: {s['notes']}")

        refusals = []
        if b["median"] is None or c["median"] is None:
            refusals.append("one side has no measurable rows")
        if b["corpus"] and c["corpus"] and b["corpus"] != c["corpus"]:
            refusals.append(f"corpus differs: {b['corpus']} vs {c['corpus']}")
        if b["acceptance"] is not None and c["acceptance"] is not None:
            d = abs(b["acceptance"] - c["acceptance"])
            if d > ACCEPTANCE_TOLERANCE_PCT:
                refusals.append(
                    f"draft acceptance differs by {d:.1f} points "
                    f"({b['acceptance']} vs {c['acceptance']}) -- this arm's rate "
                    f"tracks acceptance, so a ratio would measure the prompt, "
                    f"not the hardware (CORRECTIONS 28)")
        incomplete = [t for t, s in (("baseline", b), ("current", c))
                      if s["n_rates"] < s["n_rows"]]
        if incomplete:
            refusals.append(
                f"incomplete rounds on {', '.join(incomplete)} -- the median is "
                f"over survivors, which is not the arm's rate")

        if refusals:
            print("   RATIO WITHHELD:")
            for r in refusals:
                print(f"     - {r}")
        else:
            ratio = b["median"] / c["median"]
            direction = "slower" if ratio > 1 else "faster"
            print(f"   ratio    current is {abs(ratio if ratio > 1 else 1/ratio):.2f}x "
                  f"{direction} than baseline")
        print()


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    compare(sys.argv[1], sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
