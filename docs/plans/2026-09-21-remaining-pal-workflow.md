# Remaining models on the identical PAL workflow — 2026-09-21

Status: completed bounded evaluation; [result25](../results/25-remaining-pal-workflow-2026-09-21.md). Tracking: issue #92.

GSQ passes the original8 and overlap4 tests plus workflow/scope/evidence checks.
Dirk/NVFP4/admitted EXL3 pass original8 but fail overlap4. First EXL3 code passed
both suites but the runner rejected immediate port-release cleanup; retained as
excluded, with one unchanged-task infrastructure replacement after bounded polling
was added red-first. Five saved submissions independently verified; original PAL
unchanged. No default promotion or full Long Horizon completion.

## Scope

The developer requests the remaining four previously screened configurations on the same real PAL task as [result24](../results/24-q4-real-project-workflow-2026-09-21.md): Dirk UD-Q6_K, GSQ IQ3_S-MTP, actual source-default llama.cpp NVFP4-MTP-VERY-LOW, and EXL3 SC4.0bpw H5. No new artifact downloads or default promotion.

Preserve byte-identical `PAL_PROMPT` from `qwen38-tuning/tools/run-q4-pal-workflow.py`, detached baseline `200fcb9262d25e4002e30bf00c4364f55a42354e`, independent no-remote copies, two allowed source/test files, real restricted Claude Code2.1.258, fixed Docker visible-test MCP, context65536, medium effort, requested seed29, output8192, timeout1800s and64-turn policy. Existing malformed-JSON behavior is out of scope and preserved.

Each model starts fresh with no previous solution or hidden-test feedback. One attempt per configuration; model failure/timeout consumes its attempt. An independently demonstrated instrument failure permits a labeled replacement; retain excluded evidence and cost.

## Selected operating points

- Dirk/GSQ: stock embedded template, MTP3+ngram24/16/64, split8500,15468, llama.cpp port18080.
- NVFP4: `production_nvfp4_argv(65536)`, actual worker VERY-LOW profile and source memory admission check, port18080.
- EXL3: `exl3_argv(65536)`, native TP/cq4/gs9,15.5/ndt3 at8000; native health normalization, same OpenAI→adapter route as the prior incumbent screen.

Verify exact weights, engine/client/source hashes, tokenizer/config inputs, runtime context and listener ownership. Both EXL3 weight shards are recorded. No game/process/port conflicts may be ignored; never kill unrelated work.

## Scoring and timing

Original8-case hidden verification and visible RED→GREEN workflow checks remain inside the prior timing boundary. The existing4-case overlap audit is applied independently afterward, with its own time and result. Full acceptance requires original checks, audit, scope and evidence integrity; a partial score is not a verified complete task. Report failed-attempt cost rather than time to success for failures.

Preserve raw client/backend streams, emitted thinking/final, every tool/test result, all costs including compaction, source snapshots/diffs and sealed inventories. Additional clone HEAD/config/remotes/excluded-path checks occur outside the historical timer. Source inventory alone excludes several paths and is not a complete clone-integrity guarantee.

## Verification and reporting

Run red-first focused integration tests and the full idle instrument gate before measurements. Reuse the established PAL verifier, Docker test tool, recorder, locks and owned lifecycle; no host execution of candidate tests. Review changes at process/permission boundaries.

Primary session independently rechecks evidence hashes, original and overlap tests on saved submissions, thinking/final exports and diffs, original PAL HEAD/status, clone remotes and cleanup. Record one new result with four new rows and an eight-configuration PAL comparison, keeping historical scores immutable. Single samples and engine/boot differences preclude a general model or quant ranking. No patches copied to the original projects, commits, pushes or tracker mutation.
