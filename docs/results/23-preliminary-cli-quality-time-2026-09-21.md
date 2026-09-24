# Preliminary Claude Code CLI quality and task time — 2026-09-21

**Measured:** four actual artifacts, existing selected configurations, context
allocation65,536, medium effort, real Claude Code CLI2.1.258. One coding task,
one session per model. All four pass the frozen functional checks. **No stable
winner, long-horizon conclusion, maximum-context claim or default promotion.**

Tracking: issue #92. [Frozen preliminary protocol](../plans/2026-09-21-preliminary-cli-screen.md).
The developer explicitly authorized this reduced screen and confirmed the game
closed/GPU availability; the full Long Horizon PRD remains incomplete.

## Result

Source: [machine-readable aggregate](../../qwen38-tuning/results/preliminary-cli-2026-09-21/summary.json),
recomputed from each private `progress.json`, `session/summary.json`,
`verification/result.json`, and raw `session/stdout.jsonl`.

| Artifact | Parent hidden tests | Generated + original visible tests | Seconds to verified coding task | Generated tokens | Tool calls / errors |
|---|---:|---:|---:|---:|---:|
| Swift Q6_K | 7/7 | 10/10 | 58.781 | 2,106 | 5 / 0 |
| TURBO MTP-Q6_K | 7/7 | 8/8 | 56.500 | 1,591 | 5 / 0 |
| Dirk UD-Q6_K | 7/7 | 12/12 | 136.860 | 5,332 | 9 / 0 |
| GSQ IQ3_S-MTP | 7/7 | 13/13 | 74.531 | 3,253 | 6 / 0 |

All changed only `inventory/store.py` and `tests/test_store.py`. All added tests
for both requested behaviors. More generated tests are not automatically higher
quality: the specific coverage differences below matter more than the counts.

The timing starts immediately before the actual client invocation and ends after
parent verification, including CLI startup, generation/thinking, file tools,
recording/analysis overhead and verifier-copy/test work between those endpoints.
There were no post-verifier repair prompts, nudges or retries. Thus this is
**first-pass time to verified coding work**, not repair-loop economics.
Model hashing/guard initialization/load before session collection is separate:
Swift33.657s, TURBO33.344s, Dirk32.672s, GSQ23.953s. Those setup figures do not
include the recorder's subsequent repeated hash/preflight or final cleanup, and
must not be presented as complete cold-start totals. Independent re-verification
and manual review after the measured attempts are also outside the task clock.

TURBO and Swift are close in this single sample; their ordering is not established.
Dirk spent substantially more time and generated more tokens in this observation.
That is not a transferable long-horizon or across-task verdict.

## What the task and CLI actually did

Task: fix inventory underflow without mutating on error, include SKU/current
quantity in the error, implement strictly-below-threshold stock selection sorted
by quantity then SKU, and add tests. The existing Thai code1 brief was changed
only to state that **the parent runs tests**, because this restricted session
exposes `Read` and `Edit` rather than shell/test execution. The exact brief is
frozen in private `frozen-protocol.json`.

The CLI used its built-in bare/restricted prompt and real filesystem tools,
not direct single-answer API scoring. It is nevertheless **not the unrestricted
daily Claude Code profile**: external MCP, Bash and most tools/skills were not
available. No model self-test/TDD capability is inferred from parent test passes.

The selected profiles from [result22](22-four-model-comparison-2026-09-20.md)
were retained: embedded template, tensor split9500,14500 for Swift/TURBO and
8500,15468 for Dirk/GSQ, Q4_0 KV, MTP3+ngram24/16/64, sampler1.0/.95/20/0,
medium effort and conditional Han masking. All23 backend requests matched the
frozen sampler/seed29/output cap8192, and each carried55,328 banned-token entries.
Prefix caching was allowed for the actual multi-turn CLI history. Seed control
was requested; stochastic or cross-model equivalence is not claimed.

Both client-reported context and server allocation were65,536 for every model;
all boot logs passed66+0 residency and PID/listener ownership checks. Actual
maximum input depths were only3,779 / 3,380 / 9,001 / 4,886 tokens respectively.
**This is a shallow task inside a65K allocation, not a65K retention test.**

## Quality and complete visible-thinking review

The primary agent read **4/4 complete emitted-thinking/text exports and4/4 final
workspace diffs**, then reran verification on saved final artifacts. All four
passed again. Each candidate's generated tests were also run against the original
broken source: visible failures were5 / 5 / 6 / 8 respectively. Thus the added
tests detect real broken behavior; they are not merely present or always green.

- **Swift:** concise, consistent thinking and final implementation; generated
  tests cover strict threshold and tie order. It adds an **unrequested integer-only
  threshold restriction** (`workspace.diff:31–32`), so fractional thresholds are
  rejected. The frozen hidden suite does not test that domain; no general input
  compatibility claim follows from7/7.
- **TURBO:** shortest visible thinking (854 characters), correct scoped code, no
  tool errors. Its own tests omit the exact-threshold exclusion case; the external
  hidden suite supplies that check. The Thai final says sorting is by Latin
  alphabet (`review-visible.txt:42`), whereas the implementation uses Python
  string ordering. This is an explanation imprecision, not a failing hidden test.
- **Dirk:**11,545 visible-thinking characters, lengthy explicit drafts and repeated
  checks, including useful boundary reasoning. It completed; no runaway loop was
  established. Its final Thai includes the Vietnamese phrase **`hợp lệ`**
  (`review-visible.txt:300`; raw `stdout.jsonl:6750`). This is a concrete language
  defect despite functional success and the Han guard. Its generated tests cover
  exact threshold, zero quantity, unknown SKU noncreation and bad quantities.
- **GSQ:**5,042 visible-thinking characters, correct implementation, broad generated
  tests including unaffected other SKUs. The opening thought is repeated once
  (`review-visible.txt:3,9`), but the session completes with zero tool errors.
  No failing Thai-final behavior was identified in this one task.

Swift's visible-thinking count is1,540. Counts are characters from complete
emitted blocks, not hidden reasoning or estimated thinking tokens. The model
code/final reasoning agree on the central requirements in all four sessions.
The language observations do not establish general Thai quality or unguarded
language behavior.

## Incumbent extension: default NVFP4 and EXL3 4.0bpw

The developer then requested the same frozen task for the two existing incumbent
profiles and explicitly confirmed that llama.cpp means
`Qwen3.8-27B-NVFP4-MTP-VERY-LOW.gguf`, not a new LOW artifact. Prompt, fixture,
client/context/effort/seed/output/turn limits, parent verifier and review method
are unchanged. [Frozen extension](../plans/2026-09-21-preliminary-cli-incumbents.md),
[machine-readable aggregate](../../qwen38-tuning/results/preliminary-cli-2026-09-21/incumbent-extension-summary.json).

| Artifact | Parent hidden | Visible | Seconds to verified task | Generated tokens | Thinking chars | Tools/errors |
|---|---:|---:|---:|---:|---:|---:|
| NVFP4 VERY-LOW default | 7/7 | 11/11 | 141.735 | 5,278 | 11,030 | 9/0 |
| EXL3 SC4.0bpw H5 | 7/7 | 10/10 | 90.218 | 3,433 | 5,156 | 6/0 |

Both changed only the permitted implementation/test files. Independent fresh
re-verification passed, and their generated tests failed on the original broken
implementation (NVFP4:6 failures; EXL3:5), proving the added tests exercise the
missing behavior. This does not prove model-performed TDD: parent verification
still follows the completed session.

Time decomposition, using each engine's native measured counters:

| Component | NVFP4 llama.cpp | EXL3 4.0bpw |
|---|---:|---:|
| Client-through-parent-verifier wall | 141.735s | 90.218s |
| Engine prefill total | 10.838s | 24.767s |
| Engine decode total | 105.947s | 63.212s |
| Other measured interval | 24.950s | 2.239s |
| Cold first-request prefill | 648.9 tok/s | 60.4 tok/s |
| Aggregate fresh-token prefill | 736.3 tok/s | 214.7 tok/s |
| Aggregate decode | 49.8 tok/s | 54.3 tok/s |
| Per-request decode median/range | 51.4 / 43.3–61.9 | 62.2 / 13.6–74.5 |
| Separate setup/hash/load | 29.391s | 63.359s |

The slow EXL3 first prefill (18.348s for1,109 tokens) dominates its aggregate
prefill figure and is part of this cold session. Later EXL3 prefills range
409.8–759.2 tok/s. These are engine-native counters: EXL3 reports cached pages
through its own `live_timing.py`; no cross-engine counter formula was invented.
The “other” residual includes CLI/tool/transport/recording work and cannot be
assigned wholly to the model.

Context allocation/client window is65,536 for both. Maximum actual input was
8,192 tokens for NVFP4 (12.5% of allocation) and4,995 for EXL3 (7.6%). Cached
fractions across requests were77.2% and63.4%. Again this is a shallow coding
session, not context-retention evidence.

**NVFP4 thinking/final:** it reasons carefully about not over-validating
`threshold`, unknown SKUs, retaining zero-quantity entries, the inherited bool/int
quirk and avoiding an unrequested deletion change. This is good scope reasoning,
but it drafts and mentally re-verifies many test cases;11,030 visible-thinking
characters and5,278 generated tokens lead to the slowest measured task time of
all six. Its final Thai is clear and agrees with the diff. Functional quality is
complete on the frozen oracle; no extra input restriction like Swift's was added.

**EXL3 thinking/final:** it identifies the same essential boundaries, rejects
unrequested threshold validation, writes concise correct code and useful tests,
and finishes51.5s faster than NVFP4. However, its final Thai has a **severe
written-language defect**: misplaced/duplicated Thai marks, missing spaces and
misspellings throughout, e.g. `สิ่่งท่ี`, `ท้ัง`, `ขึ้่น`, `เหลื่อ`, `ส่วค ืน`,
`น้ี`, `ตวัตรวจ`, `หลังกาน` (`review-visible.txt:143–169`). The code and English
thinking remain readable and correct. Native EXL3 CJK/Thai/loop guards report no
failure because this is malformed Thai composition, not Han leakage or a stopped
loop. Passing7/7 must not erase this quality fault.

The profiles are the actual current operating points, not inherited arena
snapshots. NVFP4 was resolved from `worker-q4-dual.ps1 -Nvfp4 -Ctx65536 -WhatIf`:
dynamic split, fit off, checkpoints8, cache RAM24576, MTP3+ngram24/16/64,
Q4_0 KV, repeat penalty1.05 and patched late-system template. Runtime identity
names the exact VERY-LOW GGUF. EXL3 uses the pinned 4.0bpw H5 directory at revision
`b4e3574...`, MTP, fp8 KV4, tensor-parallel9/15.5 and draft3. Both shard hashes
were recomputed and matched recorded pins. Engine-native language guards are part
of the compared systems and differ by design.

Successful private evidence:
`C:\Users\xenod\AppData\Local\Temp\qwen-preliminary-incumbents-zj48yo_k`.
Independent checks:
`C:\Users\xenod\AppData\Local\Temp\qwen-incumbents-independent-hsgzwwhv`.
All234 current evidence files hash correctly; both session inventories validate,
all backend requests match the frozen sampler profile, no orphan/unresolved tool
results exist, ports8000/18080 are closed and leases released. Primary review
coverage is2/2 complete thinking/finals and2/2 diffs. Pre-run full gate:
**1925 passed,2 skipped,9 warnings in274.38s**.

Across all six single observations, time order is TURBO56.500, Swift58.781,
GSQ74.531, EXL390.218, Dirk136.860, NVFP4141.735 seconds. This is descriptive,
not a stable leaderboard. The important new trade is that EXL3 is much faster
than default NVFP4 here while its Thai final is clearly worse; functional tests
alone would have hidden that difference.

## Evidence and failed instrument attempt

Private successful root:
`C:\Users\xenod\AppData\Local\Temp\qwen-preliminary-cli-tisjjqrm`.
Each candidate contains raw frontend/backend wire, client stdout/stderr, journal,
readable history, frozen specification, source snapshots, server log, telemetry,
initial/final workspace and independent visible/hidden verification XML/logs.
[Evidence hash index](../../qwen38-tuning/results/preliminary-cli-2026-09-21/private-evidence-index.json)
covers the root; independent readback checked all464 files with zero hash/size
mismatches, and all four session inventories validated without errors. Port18080
was closed and the owned legacy lease archived after completion. All23 backend requests were complete/usable; no orphan or unresolved
tool results were found. GPU telemetry had30 / 28 / 69 / 37 samples with no
reported driver errors. Optional psutil was absent: RAM/process sampler values
remain unknown; Windows startup process checks supplemented the user clearance.

Independent fresh verifier root:
`C:\Users\xenod\AppData\Local\Temp\qwen-preliminary-independent-32o8o3td`.

Earlier invalid root:
`C:\Users\xenod\AppData\Local\Temp\qwen-preliminary-cli-z41ul_pg`.
The first Swift attempt consumed183.562s including setup and was excluded from
model-quality inference because our adapter truncated every fragmented tool
call to `{`. The other three were explicitly not_run. This cost is retained as
infrastructure failure, not a model failure or a silently discarded slow sample.
The bench-only adapter was corrected red-first; a real fragmented-tool CLI
canary then passed before the fresh four-model screen. Production EXL3 was not
modified by that correction. A prior launcher preflight also failed before GPU
use because Windows resolved `bash` to WSL; pinning the actual Git Bash binary
fixed that lease check. No user process was terminated.

Before the corrected GPU screen, the primary agent ran
`python -m pytest qwen38-tuning/bench/tests -q -rs`:
**1923 passed,2 skipped,9 warnings in258.28s**. Skips were the idle8080 `/props`
check and the installed-Studio missing-binary case; warnings were existing
`aiohttp.NotAppKeyWarning`. Hidden verification uses separate fresh copies and
runs before model-written visible tests, with a regression proving visible tests
cannot repair the implementation being graded by the hidden oracle.

Documentation link checking still reports the same eight previously documented
broken links; none points to the new report/protocol. Tracker and skill-feedback
publication were not performed because `gh` is unavailable. No commit or push.

## Preliminary decision

All four are viable for this specific restricted CLI coding task. **Swift/TURBO
are the lower-time candidates in this sample; GSQ adds broader tests at a higher
observed task time; Dirk adds useful checks but has the largest thought/time cost
and one mixed-language final defect.** A broader task plus rotated repeats is
needed to choose a foundation. No default changed, no long-horizon gate was
closed, and no raw private evidence was uploaded.
