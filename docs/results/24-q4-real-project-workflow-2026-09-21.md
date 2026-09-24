# Q4 first, then a real PAL workflow: quality, time and failed provenance — 2026-09-21

**Result:** both Q4 candidates pass the small code1 fixture and execute a genuine
read/edit/test-first loop on a detached PAL repository copy. However, **neither Q4
submission satisfies the complete real-task contract**. The original hidden
suite missed overlapping explicit/user config paths. A separate post-run audit
exposes that defect in both. Raising to Q6 does not fully solve it: Swift Q6
handles explicit files better but still fails explicit directories; TURBO Q6
also violates the explicit prohibition on broadly catching registry errors.

This is one sample per model/configuration, not a stable leaderboard or a claim
that Q4 generally beats/loses to Q6. No submitted patch was applied to the real
project. No default changed, commit, push or tracker mutation occurred.

Tracking: issue #92. [Execution record](../plans/2026-09-21-q4-pal-workflow.md).
[Machine-readable results](../../qwen38-tuning/results/q4-pal-2026-09-21/summary.json).

## 1. Scope and evidence admission

The developer asked for Swift/TURBO Q4 first, increasing quant only if poor,
then a task from an actual development project without disturbing the originals.
A read-only survey of all seven listed repositories selected PAL's configuration
registry: bounded, real, offline-testable and not the active run-journal task.

- Source: `D:/Github/pal-mcp-server`.
- Baseline: `200fcb9262d25e4002e30bf00c4364f55a42354e`.
- Detached local clone with independent Git objects and no remotes; no untracked
  `.codex`, real `.env`, logs, original virtual environment or credentials copied.
- Allowed changes: `clink/registry.py`, `tests/test_user_config_dir_rename.py`.
- Real Claude Code2.1.258; effective tools `Read`, `Edit`, and a fixed visible-test
  MCP tool. No arbitrary shell, provider tools or network-enabled model test tool.
- PAL tests run inside unprivileged, network-disabled Docker with read-only
  submission mounts; no original repository, Docker socket or user profile mount.
  A new container/temp area is used per invocation. Hidden output never feeds back
  to the model. This is a real coding/test workflow, **not the entire daily T4
  skill/plugin/GitHub/PR workflow**; the client is bare/restricted.
- Context65536, medium effort, requested seed29, output cap8192 per backend
  request,30-minute/64-turn PAL budget; no continuation nudge or human repair.
- All four PAL arms receive the byte-identical task prompt; its hash is recorded
  in the aggregate. Q6 controls were invoked only after Q4 failed the contract
  audit, and received no extra hint, solution, or audit feedback.

The original task said explicit `CLI_CLIENTS_CONFIG_PATH` configurations remain
fatal while only normal user overrides may downgrade an unsupported client.
Malformed JSON was initially misunderstood during planning: the existing helper
returns `None` and skips it. This was established before model exposure; the
final task preserves that behavior rather than adding an unrelated policy change.

### Minecraft exclusion

The developer invalidated all model tests from the contaminated hour, then
explicitly confirmed Minecraft closed. All five affected private roots are
retained with `INVALID-MINECRAFT.json`; none contributes to the tables below or
to MTP selection. See the plan for the complete exclusion list. The clean gate
returned1939 passed/2 skipped/9 warnings; clean qualification begins in
`qwen-q4-clean-qualification`, after that clearance. Game/server process probes
were repeated before each clean launch; unrelated processes were never killed.

## 2. Artifacts and operating points

| Arm | Weights | Weight bytes | Runtime |
|---|---|---:|---|
| Swift Q4 | Q4_K_M | 18,024,380,576 | MTP3 + ngram24/16/64 |
| TURBO Q4 | MTP-Q4_K_M file | 18,498,573,856 | **ngram24/16/64; MTP disabled** |
| Swift Q6 control | Q6_K | 22,884,407,456 | MTP3 + ngram24/16/64 |
| TURBO Q6 control | MTP-Q6_K file | 24,033,703,456 | **ngram24/16/64; MTP disabled** |

Q4 files were downloaded at the revisions shared with the existing Q6 pins,
locally SHA256-verified, and exposed via short-path hardlinks because the pinned
Windows loader could not open the long TURBO cache pathname. No second model
copy or quant substitution was made. Full hashes are in the plan and aggregate.
Swift's repository checksum text disagrees with the downloaded Q4 object; it was
not used as the authoritative identity.

All launches use llama.cpp b10499, tensor split9500,14500, Q4_0 KV, embedded
family template, sampler1.0/.95/20/0, conditional Han masking and verified66+0
residency. Requested/observed client and server windows agree at65536.

**Why TURBO MTP is disabled:** the clean conservative qualification found faster
MTP-request walls but different greedy output hashes, and the Thai-repair probe
hit its1024 output cap. Per the approved gate, ngram-only was selected on the same
MTP-bearing file; no regular Q4 file was downloaded. This does not prove MTP is
universally wrong or slower. Different output and truncation also prevent treating
that probe's aggregate time difference as a matched-work speedup.

Both quant levels within a family use the same chosen speculative setting.
Cross-family comparisons are operating-point comparisons, not weight-only tests.
Q4 PAL followed code1 in the same server boot; Q6 PAL used a fresh boot. All PAL
first requests nevertheless report `cache_n=0`. With no rotated repetitions,
thermal/kernel/cache-state differences are not ruled out.

## 3. Stage 1: small code1 fixture

Same inventory bug/feature fixture as result23, with a slightly shortened closing
instruction. Parent verification follows CLI completion; no model test tool is
available in this small-task phase. Do not present these as a paired Q6 quant
comparison using the earlier result23 times.

| Q4 arm | Hidden | Visible | Time to verified code1 | Backend generated tokens | Maximum input |
|---|---:|---:|---:|---:|---:|
| TURBO, ngram-only | 7/7 | 8/8 | 67.969s | 1,722 | 3,231 |
| Swift, MTP+ngram | 7/7 | 10/10 | 58.781s | 2,302 | 3,785 |

Both edit the correct two files, finish without tool errors and retain the
required behavior. Saved final artifacts passed independent Docker verification
again. Both Thai finals are readable in this sample. Swift says it added eight
tests but actually adds seven; this is a reporting error, not a failing code test.
Neither candidate's small-task pass predicted full correctness on the real task.

## 4. Stage 2: real PAL task and Q6 controls

**Times below are attempt cost through the original parent verifier, not time to
an accepted complete task. After contract audit there are zero fully accepted
PAL submissions, so time per verified PAL task is undefined—not zero seconds.**

| Arm | Original hidden suite | Post-run overlap audit | Visible tests | Attempt time | Test-tool runs | Test-first evidence |
|---|---:|---:|---:|---:|---:|---|
| TURBO Q4 | 8/8 | **0/4** | 7/7 | **206.297s** | 3 | confirmed |
| Swift Q4 | 8/8 | **0/4** | 10/10 | **683.891s** | 11 | confirmed |
| TURBO Q6 | **7/8** | **0/4** | 5/5 | 177.218s | 3 | confirmed |
| Swift Q6 | 8/8 | **2/4** | 9/9 | 333.563s | 4 | confirmed |

Test-first is not inferred from a claim: the protected MCP journal records a
failed test run with baseline implementation bytes and changed test bytes, then
a passing run with changed implementation and the same test hash as the final
submission. All four satisfy this measured sequence. It proves the observed
artifact/test sequence, not an exhaustive claim about every earlier edit.

### The missed oracle case

The eight original hidden tests exercised explicit config files/directories
outside user directories, but omitted this combination:

> `CLI_CLIENTS_CONFIG_PATH` explicitly selects a config file or directory that is
> also physically within a normal current/legacy user directory.

The requirement says explicit selection remains fatal. Both Q4 implementations
infer provenance from the parent/path location instead, so they incorrectly skip
it. Primary review caught this before publishing a full-correctness verdict.
The original sealed8/8 results remain intact; the separate audit is explicitly
post-run, not secretly added to the frozen score.

`pal_registry_overlap_audit.py` covers current/legacy × explicit file/directory:
- unchanged baseline:4/4 passes (the behavior must be preserved);
- TURBO Q4 and Swift Q4:0/4;
- TURBO Q6:0/4;
- Swift Q6:2/4, passing explicit-file cases but failing explicit-directory cases.

An independently written source-kind-aware reference passes both8/8 original
and4/4 audit tests. The contract is feasible; the original oracle was incomplete.
No model received this reference or audit before producing its submission.

## 5. Thinking, tool behavior, context and throughput

| PAL metric | TURBO Q4 | Swift Q4 | TURBO Q6 | Swift Q6 |
|---|---:|---:|---:|---:|
| Client-visible thinking characters | 7,812 | **43,646** | 6,074 | 25,559 |
| Backend generated tokens, all requests | 4,391 | **21,171** | 3,444 | 10,580 |
| Backend requests | 11 | 21 | 8 | 16 |
| Maximum actual input | 18,893 | **43,511** | 11,952 | 35,610 |
| Read / Edit / test-tool calls | 4 / 3 / 3 | **11 / 5 / 11** | 2 / 2 / 3 | 10 / 6 / 4 |
| File/tool errors | 0 | 2 | 0 | 2 |
| Automatic compactions | 0 | **1** | 0 | 0 |
| Engine prefill total | 23.551s | 83.452s | 16.195s | 49.641s |
| Engine decode total | 133.437s | 472.553s | 125.548s | 224.304s |
| Fresh-token prefill aggregate | 770.2 tok/s | 696.9 tok/s | 686.0 tok/s | 644.9 tok/s |
| Engine decode, time-weighted | 32.8 tok/s | 44.8 tok/s | 27.4 tok/s | 47.1 tok/s |

Rates are from backend `timings`, not allocated context or naive total-prompt/
prefill division. The decode aggregation time-weights the engine's native rates.
All time/cost of repeated calls and compaction remains in the attempt denominator.
No old shallow/deep noise floor or unpaired speedup verdict is imported.

**TURBO Q4:** direct progression: read files, baseline green, add three failing
tests, implement, reread and repair its own boolean-precedence mistake, green.
It avoids broad exception swallowing but overlooks explicit-path overlap. The
final overstates complete requirement coverage. It is the more economical Q4
workflow in this observation, not a fully correct autonomous submission.

**Swift Q4:** repeatedly calls the already-green test suite while saying it intends
to read supporting files. Nine baseline green runs occur before the actual RED
and GREEN. It acknowledges the repeated mistaken calls, eventually recovers,
and also attempts a directory Read and a nonexistent `.env` Read. Long planning
fills context and triggers automatic compaction:44,203 estimated pre-tokens to
13,828 post-tokens,149.818s. Crucially, its emitted reasoning identifies the
explicit-path overlap problem and the need for source-kind tagging, then chooses
the simpler parent-path check because it expects the corner case not to be tested
(`review-visible.txt:29–33,164–201`). This is a reasoning-to-implementation gap,
not a JSON formatting failure.

**TURBO Q6:** finishes faster in this sample with fewer tests/tokens, but implements
exactly the broad `except RegistryLoadError` pattern the prompt prohibited.
The original hidden missing-prompt test fails: a supported client with a nonexistent
custom prompt is warned away instead of raising. Its final says no broad catch
was added, contradicting the actual diff. Higher quant did not rescue adherence.

**Swift Q6:** carries `(file, search_base)` through the iterator and narrows the
unsupported-name check. That fixes the overlapping explicit-file case. It still
compares path equality rather than a categorical source kind, so an explicitly
selected user directory is misclassified. It uses fewer repeated baseline runs,
no compaction and about half the Q4 attempt time here, but remains incomplete.
This one sample does not establish a general quantization effect.

Review coverage: primary read all six client-visible thinking/final exports and
all six workspace diffs (two code1 + four PAL). The internal Swift-Q4 compaction
summary's semantic content was **not** separately reviewed; its backend request,
output and cost are retained. For Swift Q4, the final result's output-token field
is14,912 while cumulative backend generation is21,171: using only the final field
would hide6,259 compaction tokens. The table uses all backend requests.

## 6. Isolation, verification and retained evidence

All paths below are under `C:/Users/xenod/AppData/Local/Temp/`:

| Arm | Clean root |
|---|---|
| TURBO Q4 | `qwen-q4-pal-ckv2820y` |
| Swift Q4 | `qwen-q4-pal-ki6kzney` |
| TURBO Q6 | `qwen-q4-pal-6ydj6b2m` |
| Swift Q6 | `qwen-q4-pal-son3ntwx` |

Independent original-verifier reruns: `qwen-pal-all-independent-wn8z7jzr`.
Independent code1 Docker reruns: `qwen-code1-q4-independent-59sbzc12`.
Q4 overlap audit: `qwen-pal-overlap-audit-kr1jsfe1`.
Q6 overlap audit: `qwen-pal-q6-overlap-audit-s1e09h3l`.
Feasible reference: `qwen-pal-provenance-gold-qeynoeff`.

Each session's journal/inventory independently validates; full frontend/backend
streams and visible test logs remain private. All captured backend requests are
usable. All PAL original HEAD/status observations match before/after exactly;
all candidate clones have no remote. Other listed original projects were only
read during task selection, never edited or executed. Benchmark ports are closed,
owned leases released and no named benchmark test container remains. No patches
were copied back into the original PAL repository.

The Docker image was built by the developer after permission gating, with image
identity recorded in every protocol. PAL test execution has no host Python path.
The small code1 phase initially used its existing parent host verifier; both final
submissions were subsequently reverified in Docker. Do not claim that every
historical small-fixture verifier was OS-sandboxed.

One adversarial sandbox-probe command was permission-denied and not bypassed.
The exact real CLI Docker tool canary, safe reference/negative tests and static
security review did run; these distinctions are retained rather than claiming
all possible isolation probes passed. Optional host process/RAM telemetry remains
limited where psutil is absent. No absolute no-side-effect or adversarial-oracle
secrecy guarantee is inferred from passing tests.

## Final instrument checks

The primary session ran `python -m pytest qwen38-tuning/bench/tests -q -rs`
after the clean evaluations: **1939 passed,2 skipped,9 warnings in281.14s**.
The skips are idle8080 `/props` and the installed-Studio missing-binary case;
warnings are existing aiohttp `NotAppKeyWarning`. Independently checked all
**8,486 indexed evidence files**, zero size/hash mismatches. The original
contract checks were rerun in fresh containers for all four PAL submissions,
and the expanded provenance-aware reference passes both oracle sets.

`check-doc-links.py` still reports the same eight previously documented broken
links; none is introduced by the new report/plan. `gh` remains unavailable, so
tracker updates and the required skill-feedback publication were not performed.
This limitation is not a claim that all skill rules held. No commits or pushes.

## 7. Decision

- **Q4 is sufficient for the small code1 fixture in these two samples.**
- **Neither Q4 is yet sufficient to accept this real PAL task without review.**
- **Blindly increasing quant is not a demonstrated cure.** Swift Q6 improves some
  observed behavior but is still incomplete; TURBO Q6 introduces a broader error.
- Keep quality dimensions, test-first adherence, tool repetition and failed costs
  separate. For audited PAL, no finite time-per-verified-task is available because
  all four submissions miss at least one predeclared contract requirement.
- Next useful step is a stronger frozen oracle/task set and rotated repeats—not
  Q5/Q8 downloads or default promotion based on one green visible test suite.

No Q5/regular-TURBO artifact was downloaded. No claim is made that Q4 quality is
unchanged from Q6, or that this single task represents all seven development projects.
