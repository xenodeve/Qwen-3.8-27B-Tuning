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
# Extensions scanned. `.md` alone was the original scope and it left the four
# `worker-*.ps1` profiles -- the things that actually run -- outside the audit
# entirely, which is where the RTX 3090 scan found a false claim about
# `--fit-target` on 2026-08-22. A stale claim in a served profile is worse than
# one in a report: a report is read by someone deciding, a profile header is read
# by someone about to launch.
SCANNED_SUFFIXES = (".md", ".ps1", ".sh")

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
     r"\b(?:60|81|89|92|98|103|108|111|136|212|233|246|253|262|269|278|287|288|295|310|318)\s+tests?\b",
     "the suite is 329 tests -- but a DATED report quoting its own count is a "
     "historical record and correct as written; only operational docs must be current",
     "bench/README.md, CLAUDE.md"),

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

    ("dflash-screened",
     r"[Dd][Ff]lash[^\n]{0,60}(?:screened|not competitive)",
     "DFlash 2 does not load on build 10472 at all -- that screen could not "
     "have run; llama.cpp support needs PR #27342",
     "CORRECTIONS.md 18"),

    ("real-code-absolute-rate",
     r"(?:78\.9|105\.4|100\.5)[^\n]{0,40}tok/s|"
     r"real.code[^\n]{0,60}(?:78\.9|105\.4)",
     "an absolute real-code tok/s figure is not comparable across runs before "
     "the corpus was frozen -- the prompt was built from bench/ source that "
     "was edited between runs; only paired within-round deltas survive",
     "CORRECTIONS.md 20"),

    ("iq2s-never-loaded",
     r"IQ2_S[^\n]{0,80}(?:never (?:been )?loaded|untested rung)|"
     r"(?:never (?:been )?loaded|untested)[^\n]{0,60}IQ2_S",
     "UD-IQ2_S has 38+ measured rows across six result files, dozens of logs "
     "and four worker profiles; it was given up for IQ2_XXS on purpose, to "
     "free VRAM for a drafter",
     "CORRECTIONS.md 19"),

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

    ("nmatch-12-independent",
     r"n-match 12[^\n]{0,60}(?:same cap|chosen independently)|"
     r"(?:same cap|chosen independently)[^\n]{0,50}n-match|"
     r"tuned profile already uses[^\n]{0,60}n-match",
     "measured 2026-08-22 (sweep-ngram-nmatch.jsonl, 12 rows, paired): the "
     "llama.cpp default n_match=24 is +34.6 % RESOLVED over the 12 we ship and "
     "8 is -14.5 %. their LOOKUP_NMAX caps a longest-match search with recency "
     "tie-breaks; ours is the hash key width of a keyless table with no length "
     "dimension -- the two flags share a number and nothing else, so agreement "
     "between them validated nothing",
     "CORRECTIONS.md 21"),

    ("fit-follows-boot-vram",
     r"9,?326\s*[-–]\s*10,?7?3?2|`?--fit`? follows it|and `?--fit`? follows|"
     r"free VRAM at boot moves",
     "the range is right and the mechanism is not. 9,326-10,732 MiB is "
     "nvidia-smi's view of the CARD; llama.cpp has reported 11,069 MiB free to "
     "the process in all 552 logs this project has kept, and --fit reasons from "
     "that one. 148 of 150 boots on our artifact say 'no changes needed'; the "
     "2 that acted are n-7-clamp at 65,536. Pinning -ngl and --fit off changes "
     "nothing measurable, verified 2026-08-23. The no-cross-boot rule stands; "
     "its stated cause does not",
     "CORRECTIONS.md 27"),

    ("prompt-length-boundary",
     r"boundary is prompt length|between 43k and 64k|"
     r"collapses? (?:above|past|beyond) (?:a )?\d+k[- ]token|"
     r"prompt[- ]length (?:threshold|boundary)",
     "there is no boundary. Seven cold points go 43,162->512, 46,909->1, "
     "51,038->1, 54,310->512, 57,780->512, 60,831->512, 64,210->9 -- failure is "
     "not monotonic in length, so length is not the variable. filler cuts the "
     "corpus at exactly n*3 characters and WHERE THE CUT LANDS decides it. The "
     "same seven lengths on real-code-vendor complete 7 of 7 including 70,322 "
     "tokens. The claim was published in a commit message, which nothing scans",
     "CORRECTIONS.md 30"),

    ("fa-all-quants-decided",
     r"FA_ALL_QUANTS`? rebuild for Q8 KV\?\s*\|\s*\*\*not needed|"
     r"Is `?FA_ALL_QUANTS`? needed for Q8 KV\?\s*\|\s*\*\*No|"
     r"Q8 KV is faster on the stock binary, so it was not needed",
     "the answer is right and the row is wider than it. GGML_TYPE_Q8_0 is in "
     "the always-compiled list at fattn.cu:340-352 and falls through to "
     "return true with the flag ON or OFF, so a Q8 result cannot test this "
     "option at all. What OFF actually removes is q4_1/q5_0/q5_1 as KV types "
     "and, at fattn.cu:442-446, every asymmetric K!=V pair -- none of which "
     "was ever run. Both builds are OFF. -fa auto degrades through a WARN at "
     "llama-context.cpp:547 rather than failing, so -ctk q5_1 -ctv f16 boots "
     "with flash attention silently off. Whether ON is worth a rebuild is "
     "UNMEASURED; the correction is to the word 'decided'",
     "CORRECTIONS.md 29"),

    ("blackwell-4x-slower",
     r"[Ff]our times slower|4x slow|4× slow|~?4x slower|"
     r"22\.67 tok/s|22\.67 tokens",
     "withdrawn as a HARDWARE verdict. 22.67 came from hardware_baseline.py at "
     "draft acceptance 0.14870; 96.92 came from dflash2_arena at acceptance "
     "60.2 -- ngram-mod is speculative and its tok/s tracks acceptance, so the "
     "two were never comparable, and the 4070 SUPER never ran "
     "hardware_baseline.py at all. What IS measured: the native sm_120a "
     "rebuild takes prefill 146,155 -> 66,582 ms with acceptance byte-identical "
     "in both, and per prefill token this card is 1.517 ms against the 4070 "
     "SUPER's 0.798 -- 1.90x slower, matching 4,608 CUDA cores vs 7,168. "
     "Decode across the two cards is UNMEASURED",
     "CORRECTIONS.md 28"),

    ("decode-collapse-98304",
     r"2\.8\s*[-–]\s*5\.0 tok/s|decode collapses|13 of 16 measurements|"
     r"13/16 timeouts|the window we serve is the one that does not work|"
     r"neither residency nor speculation explains",
     "the collapse belongs to the DFlash2 arms, not the window. Every row of "
     "sweep-ngram-nmatch-98304 loaded the drafter, so depth and drafter were "
     "never separated. Measured 2026-08-23 over six paired rounds: ngram-mod "
     "alone -- what all four worker profiles serve -- returns 96.92 tok/s "
     "median at ctx 98,304 with 6/6 rounds finishing, against 5.66 median and "
     "2 timeouts for dflash2+ngram. Free VRAM 769-2,117 MiB without the "
     "drafter vs 45-376 with it, no overlap",
     "CORRECTIONS.md 26"),

    ("chars-per-token-7",
     r"7\.0\s*[-–]\s*7\.4|~?7 chars/token|chars per token is [^\n]{0,12}7|"
     r"assumed 3 chars/token; real is ~7|it is 7\.0",
     "chars/token is ~3.4, not 7. dflash2_arena.py:478 asks for "
     "filler(int(ctx * 0.5)) -- half the window by design -- so ctx 98,304 "
     "sends 147,456 chars, not 294,912. The published 6.83 dropped the 0.5. "
     "The token counts and the '~40 % of N' conclusion are unaffected; the "
     "explanation and the accusation against filler() are not",
     "CORRECTIONS.md 25"),

    ("real-task-zero-diff",
     r"changed no files|0 PASS, 5 FAIL|the worker changed nothing|"
     r"no mechanism is attached|diff_bytes.{0,12}0,",
     "the five real-task rows measured where the harness LOOKED, not what the "
     "worker did: OpenCode attaches to a server carrying the project root it "
     "first started with, so with cwd= alone the worker edited C:/AI itself "
     "while git diff in the clone stayed empty. reproduced and fixed "
     "2026-08-23 -- with --dir the same task returns EDITED, 251 diff bytes, "
     "in 32.8 s",
     "CORRECTIONS.md 24"),

    ("nmatch-24-at-depth",
     r"(?:24|n-match 24)[^\n]{0,60}widen[^\n]{0,20}lead|"
     r"widen its lead at depth|"
     r"fuller table[^\n]{0,60}colliding",
     "measured 2026-08-22 at ctx 65,536 (sweep-ngram-nmatch-65536.jsonl): the "
     "optimum MOVES FROM 24 TO 16 and 24 becomes a null. the binding "
     "constraint at depth is fire rate, not collision -- 24 fires 18 times "
     "against 16's 39 for the same accepted length",
     "CORRECTIONS.md 22"),

    ("noise-floor-at-depth",
     r"13\.6\s*%[^\n]{0,60}(?:noise|floor|drift)|"
     r"(?:below|under)\s*13\.6\s*%|"
     r"Effects below 13\.6",
     "the 13.6 % floor was derived at ctx 16,384. at 65,536 the SAME arm with "
     "byte-identical counters spans up to 48.9 % across boots, so 13.6 % there "
     "would resolve pure drift. valid at 16,384; re-derive before using at "
     "depth",
     "CORRECTIONS.md 23"),

    ("greedy-hash-cross-depth",
     r"greedy[_ ]hash",
     "takes one of two values and switches on things that are not the arm "
     "under test; valid within a depth only",
     "report 24 section 6"),
]


# A rule that cannot match is worse than a missing rule: the audit still runs,
# still prints, and still reports the tree clean of something it stopped looking
# for. On 2026-08-24 a patch script widening `test-count` wrote `\\b` where it
# meant `\b` -- legal regex for "a literal backslash, then b", which nothing in
# this tree contains. The file imported, the audit ran, the rule was silent, and
# reading the diff did not reveal it. Only executing the pattern does.
#
# So every rule is exercised at import, before anything is scanned. This costs
# 26 regex compiles and is the cheapest guard in the repo.
_DOUBLED_ESCAPE = re.compile(r"\\\\[bswdBSWD]")

for _rid, _pattern, _why, _where in RULES:
    try:
        re.compile(_pattern)
    except re.error as _exc:
        raise SystemExit(f"audit rule {_rid!r} does not compile: {_exc}")
    if _DOUBLED_ESCAPE.search(_pattern):
        raise SystemExit(
            f"audit rule {_rid!r} contains a doubled escape, so it matches a "
            f"literal backslash rather than a character class. It would report "
            f"the tree clean of something it is no longer looking for.")
del _rid, _pattern, _why, _where


def main():
    if "--ids" in sys.argv:
        for rid, _, why, where in RULES:
            print(f"{rid:26} {why}  [{where}]")
        return 0

    hits = {}
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(names):
            if not name.endswith(SCANNED_SUFFIXES):
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
