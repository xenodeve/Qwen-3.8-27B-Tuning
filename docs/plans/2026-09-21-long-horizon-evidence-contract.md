# Long-horizon campaign: decision and evidence contract

Status: offline hardening in progress; not yet a real-client certification or GPU result. Date:2026-09-21. Extends issue#92 and the approved Q6/IQ3 plan.

## Scope and interpretation

Swift Q6_K, TURBO MTP-Q6_K, Dirk UD-Q6_K, GSQ IQ3_S-MTP. Each uses its independently certified maximum viable context for this campaign. Comparisons are complete system/configuration comparisons, not weight-only causal comparisons. No production default change.

“One run” means one predeclared campaign with repeats and all evidence retained, not a single stochastic sample. A correctly executed campaign can conclude that no winner is distinguishable. Do not force an ordering or promise statistical certainty.

## Before the first GPU task

Freeze a campaign manifest and hash it before model output is examined:
- exact artifact revision/path/hash/quant; embedded/override template hash;
- engine binary/version, client version, runner/recorder/fixture/verifier hashes and dirty-source snapshots;
- actual per-request sampler/effort/token budget/guard policy, speculation and GPU split; verify wire settings rather than trusting launch flags;
- context ladder and certification rules, actual prompt-depth targets and output reserve;
- task episode/hidden-verifier IDs, fixed staged prompts, error injection schedule;
- task ordering, three rotated round seeds, time/turn budgets, continuation/retry policy;
- primary success criterion and practical quality/time trade-off thresholds before looking at results. If no user-defined tolerance exists, report pass rates and time separately rather than inventing a weighted score.

Preflight must verify loopback endpoint ownership, no competing inference/game workload, sufficient disk reserve, write/fsync/read round trip, private ignored evidence location, and a fake-client end-to-end transcript/reader smoke test. User has NOT yet confirmed Minecraft is closed. Do not begin GPU measurements until that is resolved and observed resources are recorded.

## Evidence to retain per session

1. Campaign/session/episode/attempt/request IDs and parent links; UTC and monotonic timestamps; ordered sequence numbers.
2. Every staged user prompt and actual wire message history including system/tool schema and compaction events; source hashes and byte/token sizes.
3. Raw client stdout and stderr, raw redacted-header wire copies, parsed streaming events and complete final outputs. Preserve malformed/truncated bytes for diagnosis.
4. Visible thinking exactly as emitted, with missing/empty/nonempty states; never infer hidden reasoning or classify quality solely by its length.
5. Tool invocation ID/name/arguments, results, permission denials, test outputs, retries and nudges. Preserve partial tool-argument streams and bind results to calls.
6. Backend-reported usage, cached/uncached prompt tokens, generated tokens, finish reason, prompt/decode timings; client request and complete verified-task wall time. Mark unavailable fields unknown.
7. VRAM/RAM/GPU utilization and process snapshots before boot, after load and during task; driver/device IDs; peak usage; server log; crashes/OOMs and cleanup results. Record telemetry overhead and rate.
8. Workspace initial/final hashes and actual diff, visible/hidden test outcomes and verifier execution time; hidden tests remain inaccessible to the model.
9. Setup, cold prefill, warm reuse, thinking/answer observation, tool, verification, repair and failure costs separately; retain all failed attempts in aggregate denominators.
10. Finalized event count and digest, immutable raw-file inventory, parsing/correlation errors and explicit end state. A result event or exit0 alone is not proof of complete evidence or correct work.

## Admission to result tables

Separate axes:
- **Evidence:** complete / incomplete / corrupt / unavailable.
- **Execution:** finished / deadline / turn cap / context limit / permission denied / transport/server failure / interrupted.
- **Task:** verified / failed / not verified.

Keep all attempts in the register. Do not turn timeout, truncation, missing verifier evidence, missing thinking or recorder errors into model quality passes. Missing thinking affects thinking-review coverage, not automatically functional correctness. A censored run is not a completed task.

For records admitted to a comparison, check artifact/config/fixture identity, expected cases and round counts, HTTP framing, SSE terminal state, client result consistency, task verifier evidence, file integrity, actual context usage and process ownership. Unknown/inconsistent provenance blocks a promotion verdict.

Report time per verified task as total eligible attempted-task time divided by verified completions; if none succeed, report no finite completion rate. Show boot/setup and invalid-instrument time separately, never erase them. Report per-task/per-round values and uncertainty; two easy repeated tasks do not establish broad coding or long-horizon capability.

## Prevention implemented in current components

- `agent_session_client.py`: preflight1GiB minimum free reserve and storage round trip; refusal to overwrite output files; raw stdout before JSON interpretation; private stderr; capture failures stop the owned child and cannot become `completed`; atomic client-run summary with elapsed time and return code.
- `session_history.py`: unique directory, flushed/synced append journal, poison-on-uncertain-write, atomic manifest; event count/hash at finish; `inspect_session()` distinguishes unfinished/corrupt evidence from finalized records; strict IDs/timestamps/nonnegative finite event times.
- `llama-tap/relay.py`: unique session-prefixed captures, request half-close preserved, bounded accept/session shutdown, sidecar capture errors, request/response credential-header redaction including unterminated/oversized headers, receive timestamp before downstream backpressure.
- `llama-tap/read_capture.py`: explicit HTTP framing and keep-alive pairing, malformed/truncated metadata, separate stream termination and evidence usability. Reader cannot certify task correctness.

## Integrated collection API (implemented; offline integration tested)

`bench/recorded_session.py:run_recorded_session()` now wraps the actual restricted
client function, existing transparent relay, live telemetry, event journal,
workspace snapshots and a parent-owned verifier. `run_registered_attempt()` ties
it to `bench/campaign_register.py`, preserving preflight failures as explicit
attempt outcomes rather than missing rows. The caller must supply a pre-launched
owned server, frozen artifact hashes, GPU UUIDs, isolated workspace and verifier;
this API does not launch a GPU model or assert that its maximum context is known.

Per-session output (private unless explicitly scrubbed):

- `specification.json`, `server-observed.json`: frozen expectations versus actual
  health/props, model path, runtime context, template and backend configuration.
- `recorder-source/`: source copies of the instrument used; all evidence files
  get sizes and SHA256 in `evidence-inventory.json`.
- `events.jsonl`, `stdout.jsonl`, `stderr.log`, `wire/`, `wire-requests.json`:
  original observations plus parsed requests/responses. Token usage is normalized
  by backend semantics; cached input is not confused with total context.
- `history.md`, `analysis-private.json`: readable complete message content,
  emitted thinking, tool inputs/results, repeated-call observations, compaction
  boundaries and raw client result usage. Stream deltas are not double-counted
  alongside complete messages. Repeated calls are not automatically bad edits.
- `resource-live.jsonl`, `resource-samples.json`: per-UUID GPU memory/utilization/
  clock/temperature/power/driver, RAM and limited process information, observation
  errors and collection time. Samples are flushed before being marked ready.
- `workspace-initial/`, `workspace-final/`, `workspace.diff`, before/after hash
  inventories, and `verification-private.json`: durable task artifacts and actual
  verifier results, kept separately from the model-accessible work directory.
- `summary.json`: task outcome, context high-water, client/verifier time, journal
  write time (a subset of wall time, not an additive charge), telemetry collection
  time, counts and explicit decision blockers. Raw answer/tool bodies are kept in
  private analysis, not copied into aggregate metrics.
- `manifest.json`: final journal hash/count. `verify_evidence()` rechecks both
  the journal and every inventoried artifact before later analysis.

The real telemetry path waits for its first persisted sample and blocks client
launch on missing GPU evidence or a suspected competing game/Java workload.
A suspected process is not proof that it is Minecraft. Driver reads are UUID
pinned and bounded; no zero-filled successful measurements on query failure.

An offline end-to-end test drives HTTP traffic through the real relay, observes
runtime context, records client events, retains changed files, invokes a verifier,
finalizes inventory and detects subsequent artifact modification. Other tests
cover invalid artifacts, context mismatch, competing workloads, failed verifier
infrastructure, incomplete wire data and unexecuted campaign cells.

Still required before a real campaign: access-control/ignored-directory deployment,
real Claude Code canary, owned model lifecycle and staged long-horizon task orchestration.
Manual quality review, guard false-change A/B, request-to-stage/compaction linkage
and counterfactual recorder overhead remain explicitly unmeasured. Full wire
histories retain the data for that analysis, but a caller's declared `original`
history policy is not mislabeled as observed proof that compaction never occurred.
Per-session `verified` is a functional result, not a promotion verdict;
`decision_ready` remains false pending campaign-level completeness and quality review.

Cleanup review applied the redundant journal-read/inventory-hash reductions;
shared atomic-writer extraction and relocating policy helpers were deferred to
avoid changing distinct durability/public interfaces during this integration.

Final full offline gate after integration and history-export fixes:
`python -m pytest qwen38-tuning/bench/tests -q` → **1884 passed, 3 skipped,
9 warnings in300.72s**. An earlier full run exposed a test waiting for sample
collection before persistence; a durable `wait_ready()` barrier now prevents that
race and is also used by the real preflight before launching the client.

## Limits and recovery

A disk/power failure cannot be guaranteed never to lose data. Stop scoring and mark uncertainty rather than promising durability the platform does not provide. Preserve old captures; interrupted sessions are inspected, not automatically resumed or overwritten. Header redaction is not a claim to remove arbitrary secrets embedded in prompt bodies. Use clean fixtures and local-only raw storage.

No automatic retry after partial model output unless explicitly part of the frozen policy; otherwise it changes the workload and can duplicate work. Infrastructure recovery starts a separately identified attempt and retains the previous failure/cost.
