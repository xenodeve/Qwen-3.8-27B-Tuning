"""Build the Q3-vs-Q4 comparison from the raw result files.

Reads whatever exists and reports it; missing arms are shown as missing rather
than silently omitted, so a partial run cannot read as a complete one.

    python make_report.py > ..\\results\\summary.md
"""
import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(r"C:\AI\qwen38-tuning\results")


def load(name):
    p = RESULTS / name
    if not p.exists():
        return []
    # utf-8-sig, not utf-8: PowerShell 5.1's Add-Content -Encoding utf8 writes a
    # BOM on the first write, so line 1 of every results file begins with ﻿.
    # Plain utf-8 parsing raised JSONDecodeError there and the except silently
    # dropped the first row of each sweep -- the baseline row, in every table.
    rows, dropped = [], 0
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().lstrip("﻿")
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            dropped += 1
    if dropped:
        print(f"<!-- WARNING: {dropped} unparseable line(s) in {name} -->")
    return rows


def spec_table():
    out = ["## Speculation matrix (synthetic decode)", ""]
    for tag, quant in (("q4", "UD-Q4_K_XL"), ("q3", "UD-Q3_K_XL")):
        rows = load(f"spec-matrix-{tag}.jsonl")
        out.append(f"### {quant}")
        out.append("")
        if not rows:
            out += ["_not run_", ""]
            continue
        out.append("| spec | n_max | prompt | tok/s median | min–max | acceptance | VRAM free |")
        out.append("|---|---|---|---|---|---|---|")
        for r in rows:
            acc = f"{r['acceptance_pct']}%" if r.get("acceptance_pct") is not None else "—"
            out.append(
                f"| {r['spec_type']} | {r['n_max'] or '—'} | {r['prompt']} | "
                f"**{r['tg_median']}** | {r['tg_min']}–{r['tg_max']} | {acc} | "
                f"{r['vram_free_mib']} MiB |"
            )
        out.append("")
    return out


def quality_table():
    rows = load("quality-bench.jsonl")
    summaries = [r for r in rows if r.get("kind") == "SUMMARY"]
    details = [r for r in rows if r.get("kind") != "SUMMARY"]

    out = ["## Quality benchmark (verified by execution)", ""]
    if not summaries:
        return out + ["_not run_", ""]

    out.append("| config | pass rate | verified tasks/hr | median tok/s | wall | temp | effort |")
    out.append("|---|---|---|---|---|---|---|")
    for s in summaries:
        out.append(
            f"| {s['label']} | **{s['pass_rate']}%** ({s['passed']}/{s['total_attempts']}) | "
            f"**{s['verified_tasks_per_hour']}** | {s['median_tok_s']} | "
            f"{s['total_wall_s']}s | {s.get('temperature')} | {s.get('reasoning_effort')} |"
        )
    out.append("")

    # Per-task pass counts expose WHERE a quant loses, which a single pass rate hides.
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    diff = {}
    for r in details:
        cell = by[r["task"]][r["label"]]
        cell[1] += 1
        cell[0] += 1 if r["passed"] else 0
        diff[r["task"]] = r.get("difficulty", "?")

    labels = [s["label"] for s in summaries]
    out.append("### Per-task pass counts")
    out.append("")
    out.append("| task | difficulty | " + " | ".join(labels) + " |")
    out.append("|---|---|" + "---|" * len(labels))
    for task in sorted(by, key=lambda t: (diff.get(t, ""), t)):
        cells = []
        for lab in labels:
            p, n = by[task].get(lab, [0, 0])
            cells.append(f"{p}/{n}" if n else "—")
        out.append(f"| `{task}` | {diff.get(task,'?')} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def env_note():
    rows = load("env-snapshots.jsonl")
    if not rows:
        return []
    free = [r["vram_free_mib"] for r in rows if "vram_free_mib" in r]
    if not free:
        return []
    return [
        "## Environment caveat",
        "",
        f"Free VRAM before load across {len(free)} recorded launches ranged "
        f"**{min(free)}–{max(free)} MiB**. `--fit on` derives the layer split from "
        "whatever is free at boot, so runs from different boots are not directly "
        "comparable. Comparisons below are within-sweep.",
        "",
    ]


if __name__ == "__main__":
    lines = ["# Qwen3.8-27B — Q3 vs Q4 optimization results", ""]
    lines += env_note()
    lines += spec_table()
    lines += quality_table()
    print("\n".join(lines))
