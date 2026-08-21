"""Find claims in the docs that later measurements superseded.

Every entry below is a claim this project published and then contradicted with
its own data. The failure mode this guards against is specific: a corrected
report and an uncorrected one both exist, an agent reads the wrong one, and the
correction may as well not have happened.

Each rule is (id, regex, what is wrong with it, where the correction lives).
A hit is not automatically a defect — a report that *describes* the retraction
will match too. The output is a worklist to read, not a verdict.

    python C:\\AI\\scripts\\audit-stale-claims.py            # list hits
    python C:\\AI\\scripts\\audit-stale-claims.py --ids      # list rule ids only

Exit status is always 0: this reports, it does not gate.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = (".git", "node_modules", "__pycache__", ".cache", ".venv", "researchs")

RULES = [
    ("ngram-magnitude",
     r"\+(?:94\.69|135\.89|200\.22|213\.08|330\.40|114\.64|112\.55|108\.49|120\.54)\s*%",
     "an n-gram figure measured on a prompt that is 84.5 % duplicate lines",
     "report 24, instrument fault 8"),

    ("ngram-cache-safe",
     r"ngram-cache",
     "disqualified — its greedy hash differs from its own same-depth baseline",
     "report 20 section 1.1, report 23 section 1"),

    ("ot-ssm-promoted",
     r"[Pp]romote.{0,40}`?-ot`?|most direct route to `?AD-IQ1_M`?",
     "the -ot route to AD-IQ1_M is closed; the flag gives three different "
     "outcomes depending on artifact and depth",
     "report 24 sections 1 and 1b"),

    ("iq2xxs-ceiling",
     r"`?(?:V3 )?UD-IQ2_XXS`?[^\n]{0,40}131,072|131,072[^\n]{0,30}(?:ceiling|resident)",
     "v3-iq2xxs holds 65+0 at 147,456; 131,072 was only the deepest rung tried",
     "report 24, step E3"),

    ("tasks-per-hour-608",
     r"60\.8",
     "verified_tasks_per_hour at max_tokens 3072; the 8,192 figure is 48.5 "
     "verified / 26.5 merged at the same 90 % accept",
     "report 23, START-HERE correction"),

    ("contract-as-violation",
     r"contract[_ ]violation[_ ]pct|violation rate",
     "output_contract_pct is the PASS rate, not the violation rate",
     "report 04 section 7, report 23"),

    ("reasoning-budget-zero",
     r"--reasoning-budget\s+0",
     "does not end the reasoning block; -rea off is the flag that does",
     "scripts/serve-v3-iq2xxs-fmt.ps1 header"),

    ("decoder-eliminated",
     r"draft-mtp|draft-dflash|eagle3|dspark",
     "eliminated on 160-token generations; if speculation warms up over a "
     "longer run those verdicts are provisional",
     "plan 04 P0, step W"),

    ("test-count",
     r"\b(?:81|89|92|98)\s+tests\b",
     "the suite is 103 tests",
     "bench/README.md"),

    ("reasoning-loops",
     r"loop(?:s|ing)?\s+(?:inside|in)\s+(?:the\s+)?reasoning|reasons?\s+until\s+the\s+budget",
     "the model does not loop -- a full trace scores 0.00 % repetition and ends "
     "on `stop`; the failure is in the agent loop",
     "CORRECTIONS.md 12"),

    ("below-vram-cliff",
     r"8\.56 tok/s|prefill 875|11\.64 tok/s|825\.5",
     "measured with free VRAM under ~300 MiB, the only place a collapse has "
     "been seen -- repeat it before quoting it",
     "CORRECTIONS.md 13, weakened by 14"),

    ("qwen-code-16796",
     r"16,?796|32,?768[^\n]{0,40}[Qq]wen [Cc]ode|[Qq]wen [Cc]ode[^\n]{0,40}32,?768",
     "Qwen Code's request is 54,499 tokens; 16,796 was what remained to prefill "
     "after cache reuse, and 32,768 makes Qwen Code fail with a 400",
     "CORRECTIONS.md 15"),

    ("vram-free-voids-a-row",
     r"(?:void|invalid)[^\n]{0,40}(?:row|result|measurement)|"
     r"`?vram_free`?[^\n]{0,30}validity condition|"
     r"below the line[^\n]{0,30}void",
     "free VRAM at settle does not order the outcomes -- 233 MiB ran 4.3x "
     "faster than 291 MiB on the same arm; it is a risk indicator, not a "
     "threshold that voids a row",
     "CORRECTIONS.md 14"),

    ("acceptance-one-generation",
     r"acceptance\s*(?:of|:)?\s*\d|acceptance\s+(?:from|collapse|100\s*%|4\s*%)",
     "rows before 2026-08-21 06:12 measured acceptance on the FIRST of five "
     "timed generations while tg_med is the median of all five",
     "CORRECTIONS.md 11"),

    ("greedy-hash-cross-depth",
     r"greedy[_ ]hash",
     "takes one of two values and switches on things that are not the arm "
     "under test; valid within a depth only",
     "report 24 section 6"),
]


def main():
    if "--ids" in sys.argv:
        for rid, _, why, where in RULES:
            print(f"{rid:26} {why}  [{where}]")
        return 0

    hits = {}
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(names):
            if not name.endswith(".md"):
                continue
            path = os.path.join(base, name)
            rel = os.path.relpath(path, ROOT)
            with open(path, encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
            for rid, pat, _, _ in RULES:
                rx = re.compile(pat)
                for n, line in enumerate(lines, 1):
                    if rx.search(line):
                        hits.setdefault(rid, []).append((rel, n))

    total = sum(len(v) for v in hits.values())
    print(f"{total} hits across {len({f for v in hits.values() for f, _ in v})} files\n")
    for rid, pat, why, where in RULES:
        v = hits.get(rid, [])
        if not v:
            continue
        files = {}
        for f, n in v:
            files.setdefault(f, []).append(n)
        print(f"## {rid}  ({len(v)} hits in {len(files)} files)")
        print(f"   wrong because: {why}")
        print(f"   correction in: {where}")
        for f in sorted(files):
            ns = ", ".join(str(x) for x in files[f][:8])
            more = "" if len(files[f]) <= 8 else f" +{len(files[f]) - 8}"
            print(f"     {f}: {ns}{more}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
