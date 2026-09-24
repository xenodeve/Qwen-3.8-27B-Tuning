# GSQ, NVFP4 and EXL3: time to verified completion

Status: nine validation runs completed; EXL3 is the recommended foundation for
the next optimization experiment. No production default changed.
Reasoning effort: **medium**, explicitly requested. Common allocated context:
**147456**. Output budget: **4096 tokens, including visible thinking**.
Tracking: [issue 92](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/92).

## Decision criterion

The user's criterion is time per task **together with quality**. Prefill and
decode speed explain elapsed time; they do not independently choose the winner.
Language defects are candidates for mitigation, not an automatic rejection.
The initial language-only stopping decision is withdrawn in
[CORRECTIONS 52](../reports/CORRECTIONS.md).

This is a bounded comparison of operating configurations, not an isolated
quantization experiment or proof of a globally optimal configuration. No model
has been promoted. The current tasks are single-answer implementations verified
by executing frozen assertions, **not complete autonomous repository-editing jobs**.

## What was compared

| Artifact | Best-found configuration entering validation |
|---|---|
| GSQ IQ3_S-MTP | Tensor split; native MTP n3 plus ngram match24/min16/max64; temperature0.6; conditional Han bias |
| NVFP4 VERY-LOW | Current `worker-q4-dual.ps1 -Nvfp4` recipe, resolved from its actual preview; temperature1.0; conditional Han bias |
| EXL3 SC4.0bpw-H5 | Profile G native TP, cq4, gs9,15.5, MTP ndt3; temperature0.6; incumbent guards; allocation changed to common147456 |

Exact arguments, hashes, runtime context, listener ownership, guard sources,
build and GPU snapshots are in each run's `manifest.json`. The hardware is
4070 SUPER12GB plus5060Ti16GB. One inference process tree ran at a time.
GSQ uses 12120016960 file bytes, NVFP4 14862277984, and the two EXL3 weight
shards total16701229637; these are file sizes, not GPU-residency measurements.

GSQ screening covered tensor MTP n2/n3/n4, a layer-split challenger, no
speculation, ngram-only, and MTP/ngram combinations. NVFP4 retained temperature1.0
after a bounded1.0-versus0.6 pilot; lower temperature was slower on its two
tuning tasks and did not improve the visible Thai sample. EXL3 retained its
previously measured incumbent recipe. This does not imply every EXL3 parameter,
reasoning effort or output budget has been re-optimized.

Tuning used merge_intervals/toposort. Validation used held-out lfu_cache/tree_codec,
three boots per artifact, rotating order GSQ/NVFP4/EXL3, NVFP4/EXL3/GSQ, then
EXL3/GSQ/NVFP4. Requested seeds17/53/91 are recorded, but this EXL3 path does
not establish that it applies the seed. Treat rounds as repeated workload
observations, not assured independent random draws. No significance claim is made.

## Original session and actual depth

All runs use the same captured resume of session
`096ebcae-a6ae-4b80-b608-3b60f60a3fac`, not a fresh short conversation. Capture
SHA256 is `1721824613d680f3c0c052b0ca72ac64bbcca3f12bab96c7994887f3123d73f9`.
The original session file remains unchanged. Each boot first answers the original
summary request with an empty cache; code questions then replace only the final
actual user ask in a copy of that history, preserving the trailing ambient note.
Candidate answers are not inserted into other candidates' prompts.

| Request | llama.cpp actual prompt | EXL3 actual prompt |
|---|---:|---:|
| Cold summary | 138942 | 140688 |
| Warm LFU | 139047 | 140762 |
| Warm tree codec | 139039 | 140754 |

Allocated context147456 is identical; templates yield different actual lengths.
The historical corrupted response reports153948 input tokens including cache,
while the user's UI observation was around143.2k. Current client capture has
changed system/tool definitions and disabled MCP. **This is not byte-identical
historical-request reproduction or exact incident-depth reproduction.** It cannot
disprove the corruption the user observed. No padding or silent history truncation
was used to manufacture matching counts.

The original session has101 usable deduplicated requests:93 with cached input,
86 with at least90% cached input, median cached fraction98.15%. These are
**requests, not whole tasks**. This supports separately emphasizing warm use;
it does not justify inventing a weighted end-to-end task score.

## Results

MEASURED HERE: `validation/summary.json` and validation rows in `observations.json`.
Each artifact has three cold summaries and six warm coding attempts. Cold speed
is the median uncached prefill rate; decode is the median over all six coding
attempts, including failed/budget-limited ones. Task time adds external fixture
verification to request/preparation time, excluding model startup.

| Artifact | Cold prefill tok/s | Coding decode tok/s | Accepted code attempts | Median verified task, successful only | All six attempts spent |
|---|---:|---:|---:|---:|---:|
| EXL3 SC4.0-H5 | 496.6 | 42.1 | 6/6 | 43.72 s | 249.43 s |
| GSQ IQ3_S-MTP | 578.6 | 34.6 | 5/6 | 79.33 s | 483.67 s |
| NVFP4 VERY-LOW | 700.7 | 29.4 | 3/6 | 72.39 s | 463.06 s |

The successful-only medians are conditional on different surviving tasks and must
not rank the failing arms in isolation. Across all planned coding work, total
spent seconds divided by accepted tasks is41.57/96.73/154.35 for EXL3/GSQ/NVFP4.
That is a **campaign throughput index**, not an inferred retry time or time to
finish the still-failed tasks. Only EXL3 completed all six; its mean verified
task time is41.57s. Failed-task completion time is unknown for the other arms.

| Round | EXL3 LFU / tree | GSQ LFU / tree | NVFP4 LFU / tree |
|---|---|---|---|
| 1 | 56.38 / 23.42 s, both accepted | 35.80 / 37.68 s, both accepted | 49.52 s failed / 72.39 s accepted |
| 2 | 31.06 / 56.96 s, both accepted | 79.33 / 117.26 s, both accepted | 164.41 s budget-limited / 93.82 s accepted |
| 3 | 58.10 / 23.50 s, both accepted | 81.27 s accepted / 132.33 s budget-limited | 29.77 s failed / 53.17 s accepted |

The cold summary response medians are318.26s EXL3,274.62s GSQ and253.44s NVFP4.
NVFP4 therefore has a real cold-prefill advantage in these observations. All
nine complete boot/run/cleanup episodes took4107.69s (68.46min), separately from
configuration screening and offline analysis. This includes startup/cleanup,
so it is not substituted for response/task latency. Whole-episode totals are
1377.35/1417.24/1313.10s for EXL3/GSQ/NVFP4: the shortest campaign can still have
the most unfinished work.

**Recommendation:** use EXL3 as the foundation for the next warm, deep-context
agent optimization experiment. It combines all accepted code attempts with the
lowest observed aggregate warm-task cost and more readable Thai samples. Keep
NVFP4 as a cold-prefill challenger, and GSQ as a smaller-file candidate worth
revisiting with bounded completion/repair. This is not rejection based on Chinese
leakage and does not establish a universal winner outside these tasks.

## Thinking and engineering quality

Complete external model thinking, final text and stream timestamps are retained.
The study's `thinking-and-results.md` renders them for inspection, and
`qualitative-review.md` records specific checked findings. Visible thinking is not
an oracle about either correctness or unobserved internal computation.

- During tuning, GSQ MTP n4 decoded merge_intervals at43.17tok/s versus n3's38.01,
  yet the verified task took14.37s versus10.88s. Output grew308 versus145 tokens.
  Faster decoding did not compensate for longer generation in that observation.
- NVFP4 r1 LFU deletes a frequency bucket even when it still contains another
  key. The frozen fixture catches `KeyError: 2`: this is a demonstrated functional
  bug, not a spelling judgment.
- NVFP4 r3 LFU finishes quickly but constructs `defaultdict(int)` and then treats
  its values as keyed buckets. The first insertion raises `TypeError`. Its brief
  thinking names the correct OrderedDict design but does not deliver it. Short
  thinking is not automatically efficient successful work.
- NVFP4 r2 LFU has passing complete drafts inside thinking but no final answer
  at4096tokens. Offline verifier-assisted recovery is demonstrated, not deployed;
  its raw task remains budget-limited. No manual-inspection time is disguised as
  an automated recovery latency.
- GSQ r2 tree repeatedly miscounts a worked BFS example, but its final code passes.
  GSQ r3 continues reconsidering null trimming until the budget is exhausted.
  Neither fluent self-checking nor the number of thoughts establishes correctness.
- EXL3 LFU traces revisit the min-frequency invariant too; r3 corrects an initially
  wrong update idea. Final code passes independently despite damaged formatting in
  portions of the visible draft. Blindly harvesting thinking code would be unsafe.
- Supplemental probes are reported separately from the frozen functional score:
  capacity1 plus1000gets tests retained LFU container entries; a depth1500 integer
  chain tests tree round-tripping without changing the process recursion limit.
  These are concrete engineering checks, not a general code-quality percentage.

Supplemental results: EXL3 has no retained-entry growth in all three LFU probes
and passes all three depth1500 tree probes. GSQ r1 retains1000 extra entries after
1000gets; r2/r3 do not. Its r1 recursive tree raises RecursionError, r2 passes,
r3 has no final implementation to probe. NVFP4 r1/r2 recursive trees raise
RecursionError; r3's iterative tree passes. Its r1 LFU has no retained-entry
growth in the single-key probe but still fails the multi-key functional fixture;
r2 has no final class and r3 fails insertion, so no meaningful storage-growth
score is assigned to those two. Probe success never overrides functional failure.

## Language mitigation and remaining defects

The llama.cpp arms receive conditional Han bias built from their own GGUF
vocabulary, not reused EXL3 token IDs. EXL3 keeps its production guard. This
tests mitigated operating points; the initial unguarded screen is not pooled
with these runs to claim a causal guard speedup or accuracy improvement.

Guarding Han does not repair missing Thai vowels/tone marks or Latin substitutions.
GSQ and NVFP4 still show extensive Thai corruption in the deep summary samples.
EXL3 summaries are more readable in this small cohort, but still contain occasional
spelling/wording defects. A zero-pattern count is not certified correct Thai.
Chinese-mention exemptions and partial-byte token composition remain guard gaps.
False-positive language blocking on an independent multilingual control set is
not established by these validation tasks.

An explicit Unicode-name scan of all27 responses, both final and visible-thinking
channels, found no named Han ideographs. NVFP4 r3's final summary contains Cyrillic
`ЭК` in `บЭКเอนด`. This is evidence that Han masking is not a general language
repair, **not proof that banning Han caused another language to appear**.

Several summaries repeat old conversational claims about byte corruption and
PyThaiNLP as if established. Since the task is a history summary, this is not an
independent causal diagnosis. Evidence-grounding requires a separate task.

## Instrument and scope limits

- The first warm tuning pilot replaced a converted ambient note rather than the
  actual last user question. Its warm rows are preserved and explicitly invalid;
  the corrected builder has a reproducing regression test. Only tuning-v2 enters
  selection. An earlier malformed bracket fixture is also excluded.
- Cold prefill uses actual evaluated tokens and engine timing. Warm cache reuse
  is recorded separately. EXL3 console warm rates can divide full history by a
  short new-prefill interval; use response token counts/timings, not that console
  figure. Streaming thinking spans include transport/buffering, not exact GPU time.
- Failed and budget-limited attempts retain spent time. A missing final at the
  common budget is not a proven inability to solve the algorithm. Successful-only
  medians are conditional; they must be read beside accepted/attempted counts.
- Two held-out task types times three rounds are insufficient to certify general
  coding quality or a daily driver. Multi-tool editing, broader tasks, multilingual
  false positives, higher budgets, long soak, vision and smaller GSQ variants
  remain unmeasured here. The source plan is not claimed complete.
- Cross-engine templates, samplers, native guards and serving kernels differ.
  Results select an operating point for this workload, not a root cause of Thai
  corruption or a general ranking of quantization methods.

## Next bounded optimization, not yet measured

1. EXL3: keep code/tool payloads unchanged; evaluate Thai prose detection/repair
   on corrupted and clean controls, charging repair latency to task completion.
2. All contenders: evaluate a verifier-controlled completion policy or a larger
   common output budget. The two budget-limited traces expose opportunities but
   neither higher budget nor thought harvesting is certified by this campaign.
3. Confirm the recommendation with broader full agent/tool-loop tasks at the same
   measured context regime before changing defaults. Preserve cold and warm mixes
   separately; do not transport a conclusion to another depth.

## Evidence

Study directory:
`qwen38-tuning/results/gsq-time-quality-2026-09-20/`.

- `frozen-validation-configs.json`: choices frozen before held-out runs.
- `validation/queue.jsonl`: order, exact commands, exits, complete run directories.
- `validation/summary.json`: task statuses and costs, including unsuccessful work.
- Each run: `manifest.json`, `responses.jsonl`, raw stream files, `server.log`,
  `coding-verification.json`, applicable supplemental probe outputs.
- `historical-cache-mix.json`: original-session cache analysis and deduplication.
- `qualitative-review.md`, `thinking-and-results.md`: trace review and full outputs.
- `prevalidation-tests.xml`:1741 passed,2 skipped before the frozen campaign.

Experimental instrument: `qwen38-tuning/bench/gsq_compare.py` and incident-based
`tests/test_gsq_compare.py`. Final aggregation: `.codex/summarize_gsq_validation.py`.

## Final verification

- Fresh complete benchmark suite after the campaign: **1741 passed, 2 skipped,
  9 warnings**,251.88s; `final-tests.xml` retains the result. Compilation of the
  runner, scorer, aggregators and supplemental probes passed.
- Strict aggregation found9 completed uniquely identified runs and27 responses;
  all runtime allocations and listener ownership checks passed.
- All9 retained runner snapshots are text-identical to the current source.
  Snapshot SHA256 is8b779f50fdca9574ff4e88f6b07732d3af76057044d7c6242c98a362d6e616b2;
  its bytes use CRLF because of Windows text writing. Current source uses LF,
  SHA25624fa094431b87ceb98bb9e6dfa641eb69f20b7402b695756096b5aad5a81cab3.
  Different raw hashes here are a verified newline difference, not changed code.
- Original session SHA256 remains
  cb20525e58b6e8d1de0f2c6e507f1635a96cbac996f41eeb87bbcb75c15d4537.
- Documentation check:192 Markdown files,651 relative links, one existing broken
  link in OPEN-WORK-LEDGER to `docs/reports/2026-09-16-spark13-problem-inventory.md`.
  It predates this work; documentation is not claimed globally clean.
- The stale-claim auditor exited0 but still reports historical findings; its
  exit status is not interpreted as a clean repository-wide documentation gate.
- All experiment inference servers stopped; no default launch recipe changed.
  Issue92 remains open for the explicitly unmeasured gates above.
- Skill-protocol announcement misses in this session were reported on the
  existing [skill-feedback issue331](https://github.com/xenodeve/xeno-skills/issues/331#issuecomment-5744477511),
  rather than reporting no misses or opening a duplicate issue.
