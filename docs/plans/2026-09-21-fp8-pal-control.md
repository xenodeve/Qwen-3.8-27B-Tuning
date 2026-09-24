# claude-9arm FP8 control on the same PAL task — 2026-09-21

Status: completed bounded evaluations; [result26](../results/26-fp8-pal-control-2026-09-22.md) and OpenAI-path repeat [result28](../results/28-fp8-openai-pal-repeat-2026-09-22.md). Tracking: issue #92.

FP8 Anthropic path: original8/8, visible10/10, overlap2/4,169.297s. OpenAI path repeat: original8/8, visible12/12, overlap2/4,288.750s, RED→GREEN confirmed,33 observed requests, effective request188.86tok/s and24,688 reported reasoning tokens. Neither is a fully accepted complete PAL task. The common observation apparatus and DSH route diagnosis are recorded in [result27](../results/27-comparable-gateway-local-metrics-2026-09-22.md). Remote native speed/weights/hardware remain unknown. Original PAL unchanged; no promotion or patch deployment.

## Execution checkpoint — 2026-09-22

Added remote recorder/runner and two focused test files, leaving local recorder and installed profiles unchanged. Primary security review found an unauthenticated loopback credential-use path; fixed red-first with independent random per-session local authentication before capture/upstream.13 focused tests pass. Four simplify reviews applied shared policy constants, durable-write deduplication and TLS-context reuse. Full gate:1967 passed,3 skipped,9 warnings in345.74s (third skip: current GPU budget insufficient for a source-profile test; no model launched by that skip).

A real restricted CLI canary at `C:/Users/xenod/AppData/Local/Temp/qwen-fp8-canary-444welq7` read a harmless README and returned `FP8_CANARY_OK`; complete recorded evidence, no errors, two usable inference requests. Requested ID `qwen3.8-27b-fp8`, provider response ID `vllm/Qwen/Qwen3.8-27B-FP8`; client context65536 and actual wire max_tokens8192. Client model metadata separately advertises maxOutputTokens32000; it is not the actual requested cap. Credential scan found zero occurrences in evidence. This is transport/identity evidence, not a PAL score or weight attestation.

The OpenAI-path user-controlled launch completed at `C:/Users/xenod/AppData/Local/Temp/qwen-pal-fp8-openai-aunn3bmo`; primary independent verification is `C:/Users/xenod/AppData/Local/Temp/qwen-fp8-openai-independent-20260922`. It exposes reasoning and delivery timings through33 complete sidecars. The final full gate is2046 passed,3 skipped,9 warnings in318.11s; no original/default changes. The common metric apparatus is now complete; future route comparisons must preserve protocol identity and metric boundaries.

The developer requested Qwen3.8-27B FP8 through claude-9arm to assess the base model on the identical PAL task in [result25](../results/25-remaining-pal-workflow-2026-09-21.md). The initial implementation delegation was permission-denied twice, including after explicit authorization. No bypass or live test occurred. A later explicit instruction to start permitted renewed implementation.

## Frozen scope

- Exact model ID `qwen3.8-27b-fp8`, configured HTTPS `gateway.9arm.co` route from the installed 9arm profile. Credentials remain in memory in the final-hop forwarder, never in model workspace or evidence.
- Same pinned Claude Code client, identical `PAL_PROMPT` (SHA256 `eaa6548aa70cf6a2a1cd7d952a2abdccb9eb4a3b94009d5804916ec9dcd87456`), detached PAL baseline `200fcb9262d25e4002e30bf00c4364f55a42354e`, no-remote independent copy and two allowed files.
- Same restricted Read/Edit/fixed Docker visible-test tool policy, requested client context65536, output8192, medium effort,64-turn/1800-second budget, original8 hidden verification and separate4-case overlap audit. No prior solution, oracle hint or hidden feedback.
- Controlled claude-9arm route rather than untouched daily launcher: omit its additional SessionStart Qwen router, Headroom and differing profile limits. Do not modify installed settings. Do not use PAL clink's stale qwen3.6 default.
- Remote weights/hardware/runtime context and quantization cannot be locally attested. Record provider/requested/reported identities separately; do not fake local artifacts, health, GPU counters or weight hashes.

## Instrument and admission

Reuse existing client, durable history, analysis, inventory/diff and PAL Docker verifier utilities. Add narrow remote recorder/runner with red-first offline tests; leave local admission unchanged. Native Anthropic inference traverses a loopback capture before final-hop credential injection. Fixed HTTPS destination/model/routes, verified TLS, no redirects or ambient proxy, safe error handling. Record requested versus supported/observed sampling controls; no seed or runtime-context equivalence claim without evidence.

Capture client and wire streams, emitted thinking/final, tools/test results, source snapshots, before/after clone integrity, original verifier timing and all errors/censored costs. Require usable wire evidence, not stdout-only success. One fresh task attempt; only independently established infrastructure failure permits a labeled replacement, preserving prior attempts. A tiny non-task protocol canary may establish transport compatibility.

## Verification and reporting

Primary runs the instrument gate, reviews the new credential boundary, independently verifies evidence and repeats original/overlap tests on the saved submission, reads full emitted thinking/final/diff, and checks original PAL HEAD/status and owned cleanup. Original projects and defaults remain unchanged. No commit/push or public upload.

Compare task quality/workflow with result25, but report remote end-to-end attempt latency including network/queue. Native prefill/decode and remote resource usage stay unknown unless measured evidence is supplied. Update a new result/aggregate plus indexes/ledger; keep historical rows immutable. This is one task/operating point, not a causal base-model-versus-quant verdict.
