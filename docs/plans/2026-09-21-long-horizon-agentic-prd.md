# PRD: Decision-grade long-horizon evaluation of four local Qwen artifacts

- Date: 2026-09-21
- Status: implementation continuation; GPU campaign NOT started
- Execution owner: the next primary agent, including GPT-5.6 Luna
- Tracking reference: [issue #92](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/92)
- Evidence contract: [recording and decision contract](2026-09-21-long-horizon-evidence-contract.md)

This is the local PRD and execution handoff explicitly requested by the developer. It is not a claim that `/to-prd`, tracker publication, or new issue decomposition has run. Tracker writes were unavailable: `gh` was not on PATH at handoff. Repository documents are English; any eventual GitHub PRD/issue body requires a full Thai mirror.

## Current execution state (2026-09-21, after W0/W1)

- W0 fresh offline gate: `1885 passed, 2 skipped, 9 warnings in 265.25s` before W1 edits.
- W1 validator is now implemented at `qwen38-tuning/bench/agent_context_bench.py`; its behavioral tests pass `6 passed`. It validates all four candidate keys/quant labels, local artifact SHA256, profile/runtime-context equality, UUID identities, episode/round/seed rotation, exact expected cell coverage (36 cells), and no-write `validate`/`dry-run`; `run` deliberately exits as not implemented until W2–W8 gates exist.
- W2 lock primitive is now implemented at `qwen38-tuning/bench/campaign_lock.py`; tests pass `4 passed`. It uses OS-level file locking, a diagnostic owner sidecar, and never deletes stale lock files. It is not yet wired into a model lifecycle runner.
- Fresh full gate after W1: `1893 passed, 2 skipped, 9 warnings in 262.50s`. The W2 targeted tests are additional; rerun full gate after the next source slice.
- W2 lock primitive is now implemented at `qwen38-tuning/bench/campaign_lock.py`; tests pass `4 passed`. It uses OS-level file locking, a diagnostic owner sidecar, and never deletes stale lock files. `qwen38-tuning/bench/model_session.py` is now also implemented and its offline lifecycle tests pass `5 passed`: occupied-port refusal, health timeout cleanup, owned stop and non-piped server logs.
- W3 protocol adapter is now implemented at `qwen38-tuning/bench/anthropic_adapter.py`; offline fake-OpenAI tests pass `3 passed` for nonstream, streaming terminal event/usage and loopback-only upstream validation. It reuses the repository's `anthropic_compat` translation.
- A real installed Claude Code canary was attempted through the loopback adapter, but returned `evidence_failure`, returncode1 and no client events. The first run used a temporary directory, so stderr was intentionally removed with that private temp directory; this is an unresolved canary blocker, not a pass. A rerun with persistent diagnostics was attempted after the user reported GPU clearance but was denied by the command permission classifier; do not retry by bypassing permissions. Capture the stderr in the next user-authorized canary run.
- W1/W2/W3 do not yet constitute a live model campaign. There is still no frozen checked-in campaign specification, fixed test tool, context certification ladder or long-horizon scheduler. Fresh full gate after W1–W3: `1911 passed, 2 skipped, 9 warnings in 270.35s`. Do not claim a live result from this state.

### Canary startup investigation (2026-09-21 continuation)

A specific startup incompatibility was traced without rerunning the denied live
canary. The original canary constructed `SessionHistory` without an explicit ID,
which generated `uuid.uuid4().hex` (32 characters), then passed it unchanged to
Claude's `--session-id`. The installed Claude Code **2.1.258** executable's
embedded JavaScript validates that argument with:

```javascript
var bn=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
function qn(e){if(typeof e!=="string")return null;return bn.test(e)?e:null}
```

The startup path returns `Error: Invalid session ID. Must be a valid UUID.` when
that parser returns null. Binary inspected:
`C:\Program Files\nodejs\node_modules\@anthropic-ai\claude-code\bin\claude.exe`,
SHA256 `22f5f3a44093e14c75a4d1c8ce25c730b21dd634318fbe3268e9057d12b17c41`.
This proves the original default ID was incompatible; it does not recover the
lost stderr or prove there are no subsequent live-client faults.

- RED: `test_default_history_id_is_accepted_by_claude_session_id_contract`
  failed on the missing UUID separators before the fix.
- Fix: `session_history.py` now generates `str(uuid.uuid4())`; explicitly supplied
  recorder IDs and historical evidence are unchanged. `recorded_session.py`
  already supplied canonical UUIDs, so this fault specifically affected the
  default-ID canary path, not that wrapper's explicit-ID generation.
- GREEN: `python -m pytest qwen38-tuning/bench/tests/test_agent_session_client.py qwen38-tuning/bench/tests/test_session_history.py qwen38-tuning/bench/tests/test_recorded_session.py -q`
  returned **56 passed**.
- Pre-fix full baseline returned **1911 passed, 2 skipped, 9 warnings**. The skips
  were no server on8080 for `/props` and Studio's installed binary preventing
  the missing-binary test; neither establishes live readiness.
- Final full offline command `python -m pytest qwen38-tuning/bench/tests -q -rs`
  returned **1912 passed, 2 skipped, 9 warnings in 261.07s** after the fix.
- The developer explicitly authorized the persistent loopback canary through the
  permission question in this continuation. It then **passed the protocol-only
  gate**: client exit0, final `CANARY_OK`, captured `message_stop`, ten client
  events, and finalized journal integrity. Evidence directory:
  `C:\Users\xenod\AppData\Local\Temp\qwen-canary-persistent-3yoe9swy`.
  `session/stdout.jsonl`, `session/stderr.log`, `session/events.jsonl`,
  `session/manifest.json` and `upstream-requests.jsonl` were retained.
- Independent stdout inspection found two important limits: the effective tools
  were **only `Edit` and `Read`**, despite the five requested names, and the client
  reported `contextWindow:200000` for the unrecognized `local-canary` model.
  Neither is a certified campaign profile. Stderr contains an
  `unrecognized_model` warning. The fake upstream's token usage and the client's
  estimated cost are synthetic bookkeeping, **not measured model performance or
  actual API spending**. No file-edit/test/recovery behavior was exercised.
- No GPU campaign or default change occurred. Next is W3's fixed visible-test
  tool and actual edit/fail/repair/pass plus denial canary, then verification of
  the effective tool/context profile. The protocol pass does not discharge W3.
- Documentation checks: `check-doc-links.py` exited1 with the eight already
  documented broken links; `audit-stale-claims.py` exited0 but reported1317 hits
  across183 files, so the documentation audit is not clean. `gh` remains absent;
  no tracker or skill-feedback publication was performed.

## 0. Successor: read this before doing anything

1. **Continue the current implementation; do not rebuild the recorder.** The source is in the dirty working tree, largely untracked, not in HEAD. A clean checkout will lose it.
2. **Use Q6 for Swift/TURBO/Dirk and IQ3_S-MTP for GSQ.** The developer mentioned Q4, then explicitly chose `Q6 ทั้งสาม + GSQ IQ3` when asked. No Q4 downloads or substitutions in this campaign.
3. **Use each candidate's independently certified maximum viable context.** The latest directive overrides the earlier equal-window requirement for THIS campaign only. Do not silently rewrite the permanent rule for other comparisons.
4. **Do not launch GPU work yet.** Minecraft was running and the developer offered to close it when notified. No closure confirmation was received. Finish offline gates first; then notify them and verify resources after confirmation. Never stop their game yourself.
5. **Offline tests are not a real-client canary.** No live Claude Code session has yet validated this new evidence wrapper; no new maximum context or long-horizon result exists.
6. **Do not stop at another progress-only message or green unit suite.** Execute the slices below until the requested results exist, or surface a specific evidenced external blocker. Do not invent a result to satisfy the goal.
7. **Do not promote a default, commit, push, clean the tree, or close #92.** None is authorized by this PRD. Do not create a worktree unless separately instructed; use isolated fixture copies for model work.
8. Treat a delegated report as a hypothesis. The primary agent must inspect changes and run final verification itself. Never delegate acceptance of completeness or the final ranking.

### The first concrete next step

Run a fresh offline baseline, then implement **W2: owned model lifecycle and lock** or a verified equivalent already in the repository. W1's `validate`/`dry-run` path is implemented; W2 must not start GPU work until the frozen search specification and process ownership gates are satisfied.

## 1. Problem and intended decision

The completed comparison measured two short standalone code tasks, language/math probes and a real-source approximately45K-token prompt. That is insufficient to decide which artifact can reliably perform a long coding assignment with tool use, changing requirements, recovery and retained early constraints.

The decision is: **which locally viable artifact/configuration is the best foundation for productive long-horizon work, considering quality and total time to verified completion?** Prefill and decode explain time, but cannot decide alone. A model decoding quickly while thinking in circles is not productive.

The campaign must produce reusable, inspectable evidence in one planned campaign, including repeated trials. “One campaign” does not mean one stochastic sample and does not guarantee a distinguishable winner. A valid inconclusive result is preferable to a confident unsupported ranking.

### Product goals

- G1: real tool-using Claude Code sessions for all four pinned artifacts.
- G2: a measured, bounded highest viable context for each selected configuration.
- G3: complete prompt/response/visible-thinking/tool history, including unsuccessful costs.
- G4: a frozen task suite with objective verifiers, contextual retention checks and qualitative review.
- G5: a decision report separating model/task failure, infrastructure failure, incomplete evidence and uncertainty.
- G6: a rerunnable instrument that prevents missing/corrupt evidence from becoming a plausible successful number.

### Non-goals

- Global search over every quant, engine, sampler or template.
- Q4 testing in this campaign; replacing actual Dirk with Sharp-on-NVFP4.
- A tok/s leaderboard, synthetic filler-only capacity demonstration, or short-task-only TURBO approval.
- Reconstructing hidden reasoning, uploading private histories, or automatic production promotion.
- General interactive-session recording for every application on the machine. Scope is the isolated benchmark sessions.

## 2. State at handoff: facts, not promises

### Environment and source state

- Root: `C:\AI`; Windows; Git Bash tool shell.
- Branch: `docs/spark13-observability`.
- HEAD: `d769a678120c2738ac914ace620672ce9e02ed84`.
- Many unrelated modified/untracked files existed before this work. Never reset, stash wholesale, clean, or overwrite them.
- Hardware in the completed four-artifact campaign: RTX5060Ti16GB + RTX4070SUPER12GB; driver616.92. Re-observe before new runs.
- Primary model endpoint is loopback; server/tap/client ports must belong to this campaign, not merely respond to health.
- Installed client help exposed `--bare`, `--restricted`, `--strict-mcp-config`, `--permission-mode dontAsk`, stream JSON and partial messages. Recheck installed behavior, not remembered flags.
- Windows launcher `C:\Program Files\nodejs\claude.CMD` pointed to `C:\Program Files\nodejs\node_modules\@anthropic-ai\claude-code\bin\claude.exe`. Verify existence/version; prefer the actual executable to avoid `.cmd` JSON/newline quoting corruption.

### Verified test evidence from the prior implementation turn

Last complete offline command:

```text
python -m pytest qwen38-tuning/bench/tests -q
1884 passed, 3 skipped, 9 warnings in 300.72s
```

This is a historical handoff baseline, not a fresh test run by Luna. The skips do not count as tested live behavior. Warning output was aiohttp `NotAppKeyWarning`. A prior full run caught a telemetry test racing on collection-before-persistence; `wait_ready()` now waits for the persisted first snapshot.

Existing documentation checks were not clean: eight broken links (seven vendored Sharp archive links and one older ledger link) and many stale-claim scanner hits. Do not claim a clean documentation gate or erase unrelated findings. Check the new PRD links separately.

### Implemented building blocks

| File | Implemented behavior | What it does NOT prove |
|---|---|---|
| `qwen38-tuning/bench/gsq_compare.py` | pinned models, SHA checks, launch recipes, runtime context equality, full-residency and listener helpers; direct-answer screens | not a long-horizon agent orchestrator |
| `qwen38-tuning/bench/agent_session_client.py` | restricted local client argv/env; stdin prompt; raw stdout/stderr; stream journal; timeout and owned-child handling; disk preflight | real installed client compatibility, complete descendant containment, resumable multi-stage sessions |
| `qwen38-tuning/bench/session_history.py` | ordered UTC/monotonic journal, unknown-vs-empty thinking, request context schema, atomic final manifest, event count/hash, integrity inspection, write-cost observation | all artifact integrity, task correctness, transparent automatic resume |
| `qwen38-tuning/bench/session_analysis.py` | complete-message analysis without double-counting partial deltas; tool IDs/results/repeats; compaction events; private readable history | semantic duplicate-edit/abandonment judgment, complete stage/request correlation |
| `qwen38-tuning/bench/session_telemetry.py` | explicit GPU UUIDs, memory/utilization/clocks/temp/power/driver, RAM, limited safe process observations, durable live JSONL, first-sample barrier | every possible competing workload, counterfactual instrumentation overhead |
| `qwen38-tuning/bench/gpu_device.py` | driver-query chokepoint; UUID pinning; 2s subprocess timeout | maximum-context certification |
| `qwen38-tuning/bench/campaign_register.py` | frozen spec hash, planned/running/terminal cells, atomic progress, expected-vs-actual summary | cross-process locking or actual evidence admission |
| `qwen38-tuning/bench/recorded_session.py` | composes client/tap/resources/provenance/workspace snapshots/verifier; normalized wire usage; artifact inventory; task outcome; registered attempt helper | executable campaign CLI, model startup/ownership, maximum-context search, staged episodes; `decision_ready` is currently always false |
| `qwen38-tuning/tools/llama-tap/relay.py` | transparent forwarding, unique capture IDs, half-close handling, bounded shutdown, credential-header redaction, capture errors | arbitrary prompt-body secret removal or semantic request completion |
| `qwen38-tuning/tools/llama-tap/read_capture.py` | HTTP framing/chunks/keep-alive, explicit partial data, SSE termination, request-index and usability fields | task success or complete cross-client attribution |

Critical distinction: `verify_evidence()` checks hashes and inventory, not every protocol/config/task precondition. `decide_outcome()` is an attempt-level gate, not permission to claim a campaign winner. An observed `verified` task may still have `decision_ready=false`.

### Results that exist and must not be relabeled

- [Result22](../results/22-four-model-comparison-2026-09-20.md): actual four artifacts, common65536 allocation, three rotated short-task rounds, one approximately45K real-source prompt screen.
- [Result21](../results/21-three-candidate-screen-2026-09-20.md): Sharp template on NVFP4, not actual Dirk Q6.
- The old original Claude session capture is larger than the common65K window. The later deep-corpus fixture is an explicitly derived source excerpt, NOT an original-session replay even where legacy case names say otherwise.
- Full thinking coverage was once overstated by a reviewer and corrected. Keep a per-session review-coverage register; do not inherit “all read” as fact.
- Earlier projected Q4/Q6 maximum contexts were estimates, not measurements or certified ceilings.

## 3. Frozen candidate identities

Use the constants in `gsq_compare.py` as the code source of truth and verify local bytes. These are handoff pins, not an instruction to fetch current `main` from Hugging Face.

| Key | Repository | Quant | Revision |
|---|---|---|---|
| swift | `ukisai/Swift-Qwen3.8-27B-GGUF` | Q6_K | `eb0e3a7dc70643c2ab920f5133750d02c20849ff` |
| turbo | `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF` | MTP-Q6_K | `c02caef111a8acf987947f35e1e288aa5450e184` |
| dirk | `peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF` | UD-Q6_K | `52cb3e759635ab4605e08790b6c47df8adcf0744` |
| gsq | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` | IQ3_S-MTP | `d562806dbafae37109975e970aae91b43e73b440` |

SHA256:

```text
swift 7f4de8abd5446c08b0f975a1b38e43d02639f4ae9ecdd2ddfd5c5b0f612bda59
turbo ac011aabe685edbdf542e49351eb6c76c0e5531408f2507f2235ab10931e23a5
dirk  06601c59c3dd1924c209603b3fc8369531127aa54974e1265a381c850bc37e0d
gsq   58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f
```

Swift's upstream checksum text disagrees with Hub LFS; this discrepancy is already represented in artifact metadata. Preserve it. Do not accept the contradictory checksum silently.

Dirk `--template stock` in the old runner means the artifact's embedded Sharp template, not Stock Qwen. Record actual template bytes/hash. Same flag spelling does not imply equal templates.

Starting profiles from bounded measurements: medium effort, MTP+ngram, draft max3, ngram match24/min16/max64, KV Q4_0, one slot; tensor split `9500,14500` for Swift/TURBO and `8500,15468` for Dirk/GSQ. These are candidates for certification, not proven best long-horizon configurations. Retain each model's tested appropriate configuration; do not force the same split just for symmetry.

## 4. Binding requirements and acceptance

| ID | Requirement | Evidence required |
|---|---|---|
| R01 | Four actual pinned artifacts, no substitution | local hash, runtime model path, quant and template identity |
| R02 | Real Claude Code agent executes an edit and receives test feedback | captured tool call/result, actual changed bytes, parent verification; not prose claiming an edit |
| R03 | Highest viable context is demonstrated per artifact/config | allocation, actual rendered depth, full residency, long-request and retained-state verifiers, repeat boots, failing boundary |
| R04 | Complete session evidence including failed work | raw client/wire history, prompt/thinking/tools, timing, resources, terminal reasons and verified artifact inventory |
| R05 | Fair fixed task protocol | frozen fixture/oracle/stage scripts, common task schedule, predeclared repeats/budgets/retry policy |
| R06 | Data loss cannot look like model success | negative tests plus live canary admission checks; unknowns never become zeroes |
| R07 | Time and quality jointly determine recommendations | verified-task economics, failure costs, requirement-level quality and reviewed thinking; no raw tok/s winner |
| R08 | Private, isolated, owned execution | loopback-only route, protected ignored raw storage, scoped tools, owned process trees/ports and cross-process lock |
| R09 | Every planned cell is accounted for | immutable register with terminal success/failure/invalid/censored/not-run and reasons; retries are separate IDs |
| R10 | Handoff is reproducible without conversation memory | documented CLI, immutable spec, recovery commands, source snapshots, machine-readable summary and linked report |

A requirement is not complete because a function with a similar name exists. Each completion claim names the test/run/artifact that demonstrates it.

## 5. Experiment design: separate capacity from utility

### 5.1 Two tracks within one campaign

**Track C — capacity certification:** requests and retained-state checks near each candidate allocation. The volume may differ by candidate because the aim is a capacity bound. Results cannot become a same-work performance ranking.

**Track A — long-horizon utility:** identical task episodes, inputs and staged requirements across all candidates, each running at its independently certified window/configuration. Allow natural history growth; do not pad one candidate simply because it has more memory. Compaction/recovery differences are measured system behavior. This is the primary decision track.

Keep calibration, certification and validation results in separate directories/register classes. Do not choose settings using held-out validation outcomes and then report those same outcomes as fresh validation.

**Freeze in two explicit phases:** first freeze a search specification (candidate identities, allowed profile changes, context grid, certification tasks and search budget). After certification, write a separate immutable validation specification referencing the selected profiles and their certificates. Freeze that second specification before revealing held-out episode outcomes. Never edit a frozen spec in place to fill in the discovered maximum. Predeclared search cells pruned by the search algorithm get `not_run` with the pruning reason; they are not missing data. Amendments produce a new specification ID and explain which prior evidence remains comparable.

### 5.2 Context search and certification

1. Start at65536. Coarse ladder:98304,131072,196608,262144. If65536 fails, descend by16384; refine the last successful interval in16384 steps.
2. A capacity profile is a tuple: artifact, engine, template, sampler/effort, KV type, speculation, device split, batch/ubatch and allocation. A change produces a new profile ID and requires re-certification.
3. Distinguish: launched, health-ready, runtime identity matched, small request passed, long request passed, early-state retention passed, repeated-certification passed.
4. Require66+0 model residency for the main class. Record measured layer evidence. Do not silently spill to CPU, disable MTP, enable offload to RAM or alter KV precision to salvage a rung.
5. Use real source and meaningful tool history. Target actual input at85–90% of allocation, with explicit generation reserve. Use actual tokenization/usage, not characters or allocated context as depth. A failure to reach the depth target is NOT a capacity pass.
6. Preserve a unique early requirement/fact that is not repeated in the final prompt. The later edit and hidden assertion must depend on it. Merely returning a generic arithmetic answer after45K text does not certify retention.
7. Log compaction, eviction, shifts, truncation and rejected requests. A compacted session can be usable operationally, but does not demonstrate retaining an uncompressed full-depth history.
8. Repeat the largest passing profile on two fresh boots and distinct long requests/retention cases. Report the highest passing grid point and first failed/untested bound, not an exact mathematical maximum.
9. Cap the search at262144 and a declared two-hour search budget per artifact. Budget exhaustion yields a bounded lower limit with untested upper range, not “maximum certified.” A larger follow-up needs a separately recorded extension.
10. Before promoting the selected capacity into Track A, run a real-client canary at that allocation. Loading a direct API request is insufficient to prove client context configuration or tool-history behavior.

Never extrapolate a speed or stability verdict from65K to192K. Never reuse the historical13.6% noise floor outside the depth where it was measured.

### 5.3 Long-horizon task suite

Implement three deterministic local episodes, each with four stages in the **same agent session/workspace**:

- **E1: inventory evolution.** Diagnose an existing cross-file stock bug; implement a feature; reveal a later requirement affecting previous choices; produce final compatible code/tests/README. Preserve an early invariant through all stages.
- **E2: ledger/data pipeline.** Parsing/reporting bug plus feature, invalid-input recovery, a public regression test failure, then integration and documentation. Reuse patterns from `fixtures/code-task-2`, not just its two-function short form.
- **E3: mixed-language tool recovery.** Thai/English requirements, deliberate missing-file or controlled test-tool error, recover without abandoning earlier requirements, final multi-file artifact and concise explanation. Include a legitimate Chinese-text fixture only in a labeled guard-preservation check, not as accidental language leakage.

Existing `code-task-1`, `code-task-2`, `think-task-*` are smoke/oracle-building material, not sufficient by themselves to call a run long-horizon.

Before any model sees an episode:
- freeze its initial tree and all staged prompts;
- prove the initial broken tree fails the intended assertions;
- prove a gold/reference solution passes visible and hidden assertions;
- include negative controls: unchanged tree, partial fix, hardcoded output and requirement-dropping solution must fail;
- keep hidden tests/expected solutions outside the model-readable workspace;
- predeclare exactly what test feedback the model may receive; do not leak hidden assertions through retries.

Three validation rounds, intended seeds29/43/71. Rotate model order; identical task-stage order. There are36 sessions (4 models × 3 episodes × 3 rounds). Each session gets a cumulative60-minute/128-assistant-turn budget across all four stages. Count actual complete assistant turns, not SSE chunks. Worst-case session budget is36 hours, plus certification and setup; this is a bound, not an ETA. Run serially on the shared GPUs.

Verify whether the client/server really transmits and honors seed/effort settings. If a seed cannot be controlled, label the repetitions honestly; never claim seed control from a requested CLI setting alone.

One standardized continuation nudge may be allowed after an agent stops before an unsatisfied visible stage; freeze its text and trigger before validation. Count its time and occurrence. Never silently keep nudging one model until it passes.

### 5.4 Test execution and permissions

The current client deliberately exposes only `Read,Write,Edit,Glob,Grep`. That is suitable for an edit canary but not a complete edit/test/recover agent loop.

Recommended extension: provide a parent-controlled local test tool with a small fixed schema (e.g. a visible suite/stage ID) mapped to predeclared commands. Do not give the model arbitrary shell text, arbitrary paths to hidden tests, global permission bypass, or network/deploy tools. Verify the exact installed client/MCP/tool configuration before using it; do not invent unsupported flags.

The parent always runs final hidden verification independently. Log command identity, exit code, stdout/stderr, elapsed time and artifact hashes. A failure to start the verifier is infrastructure-invalid, not a model quality failure.

Freeze the actual client/tool/system-prompt profile as part of the tested system. `--bare`/restricted mode is an isolation starting point, not proof of equivalence to the developer's full daily profile. Record every omitted skill/tool/configuration and validate that the selected tool set supports the actual staged work. If the representative harness prompt does not fit a candidate, report incompatibility; do not silently replace it with a tiny custom system prompt. Check the installed client's effective model-window and compaction behavior against the server window, including configurations below any CLI minimum advertised for `--autocompact`.

A disposable workspace is NOT an OS security sandbox. Keep credentials out of the environment, scope access, impose time/process limits, and document any remaining OS isolation limitation. Do not run unreviewed external issue code or install arbitrary dependencies as a shortcut to this local fixture campaign.

## 6. Evidence schema and privacy

The [evidence contract](2026-09-21-long-horizon-evidence-contract.md) defines the existing file layout; reuse it rather than creating a competing schema.

### Mandatory additional manifest provenance

`campaign_id`, `attempt_id`, `profile_id`, task/episode/stage/round IDs; model file/revision/quant/hash; template hash; engine/client executable/version/hash; source snapshot hashes; GPU UUIDs/driver; actual launch argv; observed runtime props; KV/speculation/split; fixture/verifier/staged-prompt hashes; guard configuration/version; budgets/order/seed policy; operating-system/runtime versions; privacy classification; boot/process ownership evidence.

### Per-request records

- Unique session/request/attempt association and parent/continuation links.
- Actual wire prompt/messages/tools/system instructions, byte sizes and safe references.
- Requested window and observed runtime window, separately from actual input tokens.
- Backend usage semantics: Anthropic uncached input + cache-read + cache-creation versus OpenAI total prompt tokens. Preserve raw usage. Never double-count cache tokens or assume missing means zero.
- Maximum output budget, observed generated tokens, finish reason and stop/censor/error.
- Receive timestamps, request wall time, backend prefill/decode if reported, cached/uncached regime. No fabricated thinking-token count.
- Declared history policy separately from observed compaction. Link actual before/after requests where possible; otherwise correlation stays unknown.

### Per-session records

Full emitted thinking and answers; raw/normalized tool inputs/results and partial arguments; real edits and snapshots; public and hidden test outcomes; retries/nudges/permission denials; resource samples and peaks; server logs; cold setup and request/tool/verification/repair costs; instrumentation write/collection cost; readable history; file inventory and hashes; capture and parser errors; terminal state; quality-review coverage.

Record guard before/after output only when a guard actually transforms output; sampler masking does not produce an observed unmasked counterfactual. Guard false changes and latency need a separate controlled A/B, not an invented before-output.

Use UTC and monotonic clocks. Measure recorder write time as a subset of wall time; do not add it again. A no-recorder calibration may estimate counterfactual overhead, but never replace the instrumented task denominator with a guessed subtraction.

### Orthogonal statuses

- Evidence: complete / incomplete / corrupt / unavailable.
- Execution: finished / timeout / turn cap / context limit / permission denial / transport failure / server failure / interrupted.
- Task: verified / failed / not verified.
- Decision: admissible / blocked / inconclusive.

A censored run can still have incomplete evidence. Do not let the single `outcome` field hide either dimension. A final SSE marker alone is not a valid complete answer; a complete file alone is not a valid verified task.

Raw evidence must be local, git-ignored, access-restricted and outside the model workspace. No automatic upload to GitHub/Hugging Face. Header redaction does not sanitize arbitrary prompt-body secrets. Use clean fixtures and dummy local auth; inspect exports before publication. Logs, settings and hidden tests must not be readable through the agent's allowed directories.

## 7. Required corrective checks before real GPU execution

These are verification obligations, not claims that every listed issue is a proven bug:

- **No executable campaign caller:** implement the entry point, not another isolated helper.
- **Schema not yet strict enough:** validate all required provenance before expensive work; runtime checks currently cover only a subset.
- **Integrity versus admission:** recheck protocol terminals, expected cells, identity, coverage, verifier and semantic state separately from SHA integrity.
- **Status compression:** retain evidence incompleteness on censored runs; never count infrastructure failure as model failure.
- **Process containment:** prove bounded cleanup when stdout closes before process exit, the parent exits before descendants, stderr capture fails, or journal writes fail. No unbounded joins and no process-name-wide kills.
- **Relay shutdown:** prove an active request cannot lose its tail or look complete after shutdown. A reader must distinguish transport closure from semantic completion.
- **Known file provenance:** record verifier/client/template/helper sources as well as model hash. Do not trust filename-only identity.
- **Resource policy:** explicit dual-GPU UUIDs, durable first sample before client launch, data gaps tracked; Java detection is a suspicion, not proof of Minecraft. Wait for user clearance as well as measured preflight.
- **Stage and request correlation:** raw data is already retained, but current per-request stage/compaction attribution is unresolved. Implement deterministic association or preserve explicit ambiguity.
- **Snapshot timing:** preserve the agent's final workspace before a verifier can mutate it; keep verifier-generated cache/test artifacts distinct. Catch symlinks/out-of-root paths.
- **Repeat/loop semantics:** repeated exact tool calls are observations, not automatically duplicate edits. Do not equate normal self-checking with runaway repetition.
- **Client shape:** prove real `stream-json` and partial-tool-event semantics. Avoid assuming every assistant event with the same ID is either an exact duplicate or a different complete message without checking the installed client's stream.
- **Permissions/test tool:** ensure agents can actually run permitted visible tests and receive feedback; a file-only canary cannot certify this.
- **Restarts:** never overwrite old evidence or automatically rerun already-final cells. Interrupted recovery is an explicit separate attempt.

## 8. Ordered implementation slices for Luna

Work one slice at a time: read its listed interfaces → add behavioral RED → observe the intended failure → implement minimally → run covering tests → inspect diff → record gate/evidence. Do not write a whole new harness and its tests in one unobserved pass.

| Slice | Change inventory | Acceptance before moving on |
|---|---|---|
| W0: recover baseline | read ledger, corrections/traps, this PRD, evidence contract and result22; inspect git and exact files | known dirty state preserved; fresh tests; no GPU launch; verify no existing equivalent caller appeared |
| W1: manifest + entry point | create `bench/agent_context_bench.py` and `tests/test_agent_context_bench.py`; reuse `CampaignRegister`, `run_registered_attempt` | `--validate-only` and `--dry-run` work without GPU/client launch; required fields/IDs/hashes/budgets validated; expected cells enumerated deterministically |
| W2: model lifecycle + lock | `bench/model_session.py` and `tests/test_model_session.py` if no suitable equivalent exists; reuse `gsq_compare` identity/residency/listener functions | exclusive cross-process lock; occupied-port refusal; PID+creation-time ownership; abort/cleanup never touches unrelated process; all failure paths recorded |
| W3: real client + scoped test tool | extend `agent_session_client.py`; `bench/fixture_test_tool.py` and `tests/test_fixture_test_tool.py` for the local fixed-test tool/config; no unrestricted shell | offline fake tests plus real canary eventually demonstrate edit → visible failure → repair → visible pass; denied out-of-scope operation remains denied |
| W4: evidence admission | `recorded_session.py`, `session_history.py`, `session_analysis.py`, telemetry/relay/reader only where needed | full raw-to-summary reconstruction; all mandatory identities; capture/error/status separation; per-stage linkage; matching IDs; mutated/truncated/absent files rejected |
| W5: staged fixtures and oracles | new `fixtures/long-horizon/` episodes, hidden gold/negative controls kept outside seeded workspaces | all broken/gold/partial/hardcoded controls verified offline; no solution or hidden-test leakage; staged requirements test real retention |
| W6: certification runner | `agent_context_bench.py` context ladder/profile selection and tests | fake-server ladder decisions pass before GPU; no unsupported maximum claim; repeated live context and retention results stored for each candidate |
| W7: campaign scheduler | same entry point/register, restart/continue and progress report | 36 planned cells; round rotation; budgets cumulative per session; no duplicate finished cells; one model at a time; all costs/terminal reasons retained |
| W8: scorer + review/report | focused summary/report helper plus tests; use existing analysis/inventory | recomputed summary matches raw evidence; invalids/censoring visible; quality-review coverage exact; recommendation can be inconclusive |
| W9: live campaign + final verification | execute, inspect all acceptance gates; docs/results register and ledger | all four have admissible results or specific blockers; final report meets R01–R10; no default change |

Proposed new names are implementation targets, not existing capabilities. If an equivalent already exists when Luna resumes, extend it instead and record the mapping.

**Legacy coordination is a change site:** `qwen38-tuning/scripts/swap-model.sh` already uses `qwen38-tuning/.port8080.lock`, owned by its calling job. W2 must inspect that protocol and prevent overlap with it, not introduce a second unrelated lock and call the machine exclusive. Windows versus shell PID namespaces require checked ownership; ambiguous/stale-looking locks are not permission to delete another job's lock. Do not invoke the legacy global-stop launch paths as an owned-session implementation.

Recommended additional new test sites: `tests/test_agent_context_bench.py` for schema/grid/rotation/budgets; `tests/test_model_session.py` for lock/ownership/teardown; `tests/test_fixture_test_tool.py` for visible-suite allowlisting and denied paths. Existing component test files in section9 remain regression gates. Keep any new scoring helper under `bench/` with a dedicated behavioral suite; choose its final name once the existing report helpers have been surveyed.

### CLI contract to implement in W1

One entry point with explicit subcommands or equivalent flags:

```text
python qwen38-tuning/bench/agent_context_bench.py validate --spec <path>
python qwen38-tuning/bench/agent_context_bench.py dry-run --spec <path>
python qwen38-tuning/bench/agent_context_bench.py canary --spec <path> --artifact <key>
python qwen38-tuning/bench/agent_context_bench.py certify --spec <path> --artifact <key>
python qwen38-tuning/bench/agent_context_bench.py run --spec <path>
python qwen38-tuning/bench/agent_context_bench.py inspect --campaign <directory>
```

These commands **do not exist yet**. They specify the interface to build, not commands to try now. `validate`/`dry-run` must not start a model, real client, test subprocess from generated code, or network download. `run` refuses an unapproved/unfrozen or incompletely certified spec. `inspect` is read-only and cannot repair/overwrite evidence silently.

Resumption inspects the durable register first. Planned cells may proceed; running cells require recorded interruption/recovery; terminal cells are not rerun. A retry has a new attempt ID referring to its predecessor and remains in the denominator.

## 9. Offline and live test matrix

| Failure/control | Required observation |
|---|---|
| Wrong model hash/path/template/runtime window | blocked before agent work; reason/evidence retained |
| Wrong/missing GPU identity or stalled driver query | bounded timeout, unknown measurement, no fabricated free memory |
| Game/competing workload or missing first durable sample | client not launched; no benchmark verdict |
| Disk full/read-only/capture write/fsync failure | stopped/invalid evidence, no success manifest; owned cleanup |
| Process exit/EOF races, hanging descendants | bounded teardown; incomplete cleanup recorded; never hangs next cell |
| Chunked/keep-alive/interim HTTP, malformed/truncated SSE/UTF-8 | correct request pairing or explicit unusable/unknown; never silent omission |
| Missing result/duplicate conflicting IDs/orphan tool result | explicit anomaly; no double-counting or guessed linkage |
| Missing/empty/nonempty thinking | distinct states; full emitted text retained; no invented hidden content |
| Compaction or context rejection | before/after artifacts and boundary; no full-depth claim from shortened history |
| Hidden verifier cannot start/crashes | infrastructure-invalid, distinct from a legitimate test assertion failure |
| Initial/gold/partial/no-edit/hardcoded fixture controls | correct oracle behavior; proves tests are meaningful |
| Journal/inventory/file altered after finish | detected on independent readback; row cannot enter report |
| Planned cell omitted or retried | omission/retry visible in register; totals reconcile |
| Reporter recomputation | deterministic aggregates match raw attempts and manifest version |

Existing targeted command (already-existing test files):

```text
python -m pytest qwen38-tuning/bench/tests/test_recorded_session.py qwen38-tuning/bench/tests/test_agent_session_client.py qwen38-tuning/bench/tests/test_session_history.py qwen38-tuning/bench/tests/test_session_analysis.py qwen38-tuning/bench/tests/test_session_telemetry.py qwen38-tuning/bench/tests/test_campaign_register.py qwen38-tuning/bench/tests/test_llama_tap.py qwen38-tuning/bench/tests/test_capture_framing.py -q
```

Full offline gate, **without inference running**:

```text
python -m pytest qwen38-tuning/bench/tests -q -rs
python scripts/check-doc-links.py
python scripts/audit-stale-claims.py
```

Record exact exit codes/counts and distinguish pre-existing documentation failures. One test reads live `/props`; running the suite concurrently with an experimental template previously produced a false template-regression failure. Do not repeat that overlap.

After source changes: simplify; before accepting the instrument: code review and security review for recording/permissions/process boundaries. Final verification remains with the primary agent. Use real fault probes, not assertions about strings in source code.

## 10. Measurement and recommendation rules

- Primary steady-session metric: total attempted task time through independent verification, including failed attempts/retries/nudges, divided by verified completions. With zero verified completions, report no finite time-to-verified-task; do not show zero seconds.
- Report setup/hash/load/cold-start and infrastructure-invalid time separately; never erase it. Invalid measurements are excluded from model-performance inference but included in an explicit operational cost register.
- Preserve per-episode/per-round results; show median/range and total costs. Do not substitute a favorable seed or remove a long thinking outlier.
- Show prefill, decode, cached-prefix regime, generated tokens per verified task and observed thinking/answer timing as explanatory components, with source/availability labels.
- Quality: final test/requirement coverage, stage retention, regressions, tool recovery, premature completion, observed loops, language, and thinking/final consistency. A fluent thought is not a pass; a short thought is not automatically good.
- Review all session finals and visible-thinking evidence or state precise reviewed/total coverage. Each finding names request/event/file anchors. Delegated “all read” is insufficient without checked coverage.
- Three rounds are a minimum repeat design, not proof of significance. Do not import old noise thresholds at different depth or pretend independent seeds exist when unverified.
- Select a shortlist if evidence supports it; declare inconclusive when trade-offs or uncertainty prevent a defensible single winner. Do not invent utility weights or an acceptable quality-loss percentage.
- A repairable language fault creates a measured optimization candidate. Guard overhead, false modifications and residual quality must be assessed before claiming the repaired system is superior. Guard repair cannot turn an unfinished/incorrect task into a hidden pass.

## 11. Stop/park and autonomy boundaries

Continue independently on implementation, offline tests, isolated fixture work and serial runs already within the frozen scope. Do not ask again about Q6, the four artifacts, the primary metric, or the per-model-context choice.

Stop the affected cell and preserve evidence for: OOM, lost capture, wrong artifact/context, missing ownership, unsafe path, timeout/turn cap, corrupt evidence or failed instrument preflight. Continue independent candidates only if the failure is candidate-specific and the instrument remains trustworthy. A shared instrument fault stops the campaign until fixed and reverified.

Park for a genuine external decision: Minecraft not cleared, unavailable permission/toolchain that cannot be safely resolved, insufficient disk without deleting user data, unexpected remote costs/downloads, changing artifact/quant/client/engine, or extending the frozen budget/workload after outcomes are known. A permission denial is not permission to retry through a different agent/tool.

Do not work around a denied command by laundering it through another agent. Do not kill services by name or port. Do not silently replace Claude Code with OpenCode, synthetic answers or direct API probes.

## 12. Deliverables and definition of done

- [ ] Executable validated/dry-run/canary/certify/run/inspect path, with documented flags and tests.
- [ ] Four pinned candidates and complete source/config/fixture provenance.
- [ ] Real local-client edit/test/recover canary, scoped permissions and private evidence verified.
- [ ] Per-candidate highest passing tested grid point, actual depth and retained-state evidence, repeated boots and boundary/untested range.
- [ ] Three frozen long-horizon episodes × three rotated rounds × four candidates, or explicit evidenced blockers/not-run reasons for every missing cell.
- [ ] Every attempt retained, including failures, censoring, retries and infrastructure costs; no orphaned planned cells.
- [ ] Raw and readable histories, resource observations, real workspace artifacts/verifier logs, integrity and semantic-admission report.
- [ ] Complete quality review or exact coverage limitations; no overstated thinking coverage.
- [ ] Recomputed machine-readable metrics and English report with a defensible recommendation or an explicit inconclusive verdict.
- [ ] Results index and ledger updated; tracker updated only if reachable/authorized, with bilingual bodies. No issue closure without separately justified evidence.
- [ ] Full final tests run by the primary agent; no unrelated dirty changes destroyed; no default promotion.

Expected report: next available numbered file under `docs/results/`, linked from `docs/results/README.md`. Raw private evidence belongs in a protected ignored location; only scrubbed metadata/aggregates are candidates for version control. The existing result22 is historical and must not be overwritten to pretend this campaign already ran.

**Not done:** PRD written; many unit tests green; model loaded; short task passed; telemetry recorded; agent claimed an edit; or all cells marked terminal with invalid evidence. None alone satisfies this PRD.

## 13. Luna execution prompt

Copy this with the file path after switching models:

```text
Continue C:\AI using docs/plans/2026-09-21-long-horizon-agentic-prd.md as the execution contract.
The target is Q6 Swift/TURBO/Dirk versus GSQ IQ3_S-MTP, each at its independently
certified maximum viable context, with real Claude Code long-horizon tasks and
complete private session evidence. Do not reinterpret this as Q4 or equal-context.
Read the handoff state and W0, preserve all dirty/untracked work, run the offline
baseline, then implement W1 onward one test-first slice at a time. Reuse the
existing recording modules; there is not yet an executable campaign runner.
Do not start GPU work until offline gates pass, I confirm Minecraft is closed,
and you verify resource/process ownership. Do not claim live results from fake
integration tests. Keep going through the frozen campaign and report verified
results or a specific evidenced blocker, without changing defaults or committing.
```
