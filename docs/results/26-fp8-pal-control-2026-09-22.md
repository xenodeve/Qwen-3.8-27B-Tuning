# Qwen3.8-27B FP8 through the controlled claude-9arm route — 2026-09-22

**Result:** the remote FP8 submission passes original hidden8/8 and visible10/10, but only2/4 overlap cases. It is **not an accepted complete PAL task**. It also fails the frozen identical-test-hash workflow gate, despite visibly performing a test-first cycle and legitimately correcting a mistaken test assertion. Keep those statements separate.

The observed provider model ID is `vllm/Qwen/Qwen3.8-27B-FP8`, requested as `qwen3.8-27b-fp8`. This is provider identity evidence, not an independent FP8 weight/hash attestation or a controlled causal quantization comparison.

Tracking: issue #92. [Plan](../plans/2026-09-21-fp8-pal-control.md). [Previous eight configurations](25-remaining-pal-workflow-2026-09-21.md). [Aggregate](../../qwen38-tuning/results/fp8-pal-2026-09-22/summary.json).

## 1. What ran

One actual restricted Claude Code2.1.258 session used the selected claude-9arm model/gateway/auth route. The ordinary launcher was deliberately not used unchanged: its Headroom hop, SessionStart Qwen router and differing context/output/compaction overrides were omitted. Installed profiles/defaults were not edited. The exact adaptation was disclosed before execution.

Matched task conditions:
- Byte-identical `PAL_PROMPT`, SHA256 `eaa6548aa70cf6a2a1cd7d952a2abdccb9eb4a3b94009d5804916ec9dcd87456`.
- Independent detached PAL baseline `200fcb9262d25e4002e30bf00c4364f55a42354e`, no remotes/private original files, only registry/test-file edits.
- Same requested CLI file-tool arguments and effective tool schemas: Edit, Read, fixed Docker visible-test MCP. No shell/network tool available to the model.
- Requested client context65536, output8192, medium effort,64-turn/1800s policy. Wire confirms max_tokens8192, adaptive thinking request and output_config.effort=medium; provider runtime context remains unknown. CLI metadata advertises maxOutputTokens32000, which is not the actual wire cap.
- Request sampler temperature1.0/top_p0.95/top_k20. Seed/min_p and internal provider sampler/cache/template behavior cannot be matched or attested through this route; do not assume them equal to local runs.
- Original8 hidden cases and visible verifier retain their historical timing boundary. The same4-case overlap audit runs afterward, outside the model session. No prior output, hidden feedback or repair prompt was given.

No local inference server was launched or stopped. Latency includes network/provider queue plus local CLI/tool/verifier cost. It is not local GPU speed. Ordinary CLI system metadata such as workspace path/date differ naturally across sessions.

## 2. Quality and attempt cost

| Measure | FP8 control |
|---|---:|
| Original hidden | **8/8** |
| Post-run overlap audit | **2/4** |
| Final visible suite | **10/10** |
| Attempt through original verifier | **169.297s** |
| Owned client execution | 157.453s |
| Original parent verifier | 9.719s |
| Backend inference requests | 21 |
| Provider-reported generated tokens, all requests | 14,124 |
| Maximum provider-reported actual input | 37,536 tokens |
| Read / Edit / fixed-test calls | 11 / 8 / 4 |
| Unavailable Glob calls | 2 |
| Tool errors | 4 |
| Compactions | 0 |
| Accepted complete task | **No** |
| Time per accepted complete task | **Undefined** |

### Same functional comparison, not a stable ranking

| Configuration | Original hidden | Overlap audit | Attempt cost |
|---|---:|---:|---:|
| GSQ IQ3 | 8/8 | **4/4** | 271.125s |
| **9arm FP8 control** | 8/8 | **2/4** | **169.297s** |
| Swift Q6 | 8/8 | 2/4 | 333.563s |
| Dirk Q6 | 8/8 | 0/4 | 455.187s |
| NVFP4 VERY-LOW | 8/8 | 0/4 | 312.672s |
| EXL3 H5 admitted replacement | 8/8 | 0/4 | 197.000s |
| TURBO Q4 | 8/8 | 0/4 | 206.297s |
| Swift Q4 | 8/8 | 0/4 | 683.891s |
| TURBO Q6 | 7/8 | 0/4 | 177.218s |

Historical values are unchanged from result24/25. EXL3's favorable first cleanup-excluded12/12 submission remains documented in result25; it is not silently replaced by the admitted failing trajectory. These different remote/local operating points do not support a speedup or quantization causality claim.

## 3. Why FP8 failed the contract

### Source category still inferred from path equality

FP8 adds `(config_path, is_user_config)` tuples, but computes the flag as `base in user_dirs`. This solves the explicit-file overlap cases: the file path is not equal to the user directory. It still misclassifies an explicitly selected directory equal to current/legacy user config as a normal user source.

The separate audit fails exactly `legacy-directory` and `current-directory`; both file cases pass. This is the same functional coverage as Swift Q6, not GSQ's correct categorical source tagging.

### Broad error catch with message classification

The code catches `RegistryLoadError` and classifies it using the substring `"is not supported by clink"`. This is not a structural distinction between unsupported names and other failures. A separate post-run review test uses a supported `opencode` config with a missing prompt filename containing that phrase. The unchanged baseline raises the expected missing-prompt error; the FP8 submission suppresses it and skips the supported client.

That supplemental test is **not added to the shared12-case score**. It independently demonstrates why the task prohibited broad error catching. Evidence: `test_error_classifier_audit.py`, `classifier-baseline/result.json` (1 passed), `classifier-fp8/result.json` (1 failed), in the independent verification root.

### Workflow gate versus actual TDD behavior

The four visible test runs were:
1. Baseline:4 passed.
2. New tests, baseline implementation:3 failed/7 passed. Two failures expose the requested feature; the third is a wrong expected exception type.
3. After implementation:1 failed/9 passed, leaving only that test assertion wrong.
4. After correcting the assertion to accept the existing Pydantic ValidationError:10 passed.

The frozen scorer requires identical test-file hashes at the baseline-source RED and final GREEN. The test correction changes that hash, so `red_then_green=false`. Preserve this score for comparability. **Do not paraphrase it as "FP8 never did TDD" or "cheated by removing a failing requirement":** the trace shows a genuine RED before implementation and a reasonable correction to the test's exception expectation. The independent overlap/exception-classification defects remain real regardless of this conservative workflow gate.

## 4. Thinking and tool behavior

The service emitted21 empty thinking blocks and **zero nonempty reasoning characters**. Both raw wire and client events agree: no thinking/reasoning delta text arrived. The request did ask for adaptive thinking with medium effort. Therefore reasoning quality cannot be directly compared with the local models' emitted traces. Zero observed text is not proof that the model performed no internal reasoning.

Observable behavior:
- Reads source/models/helpers and writes six new visible tests.
- Calls unavailable Glob twice, then tries to Read two directories; all four attempts return tool errors, with no forbidden shell execution.
- Recovers and completes edits/test runs without human help.
- Removes an unused ValidationError import rather than changing production schema-error semantics to satisfy its mistaken assertion; then corrects that assertion in the test.
- Final says explicit sources are always marked false, but the directory-overlap code contradicts that claim. Tests establish behavior only for the covered nonoverlapping paths.
- Its warning helper asserts a warning exists but does not assert exactly one; the actual implementation emits one for the tested case. More test prose is not exhaustive coverage.

The final uses English; this frozen task does not test Thai output. Native remote prefill/decode rates, queue time, hardware and cache implementation are unknown. CLI costUSD has costBasis=unknown and is not treated as a billing measurement.

## 5. Evidence, isolation and checks

Private task root:
`C:/Users/xenod/AppData/Local/Temp/qwen-pal-fp8-y3ms8li2`.

Independent verification/review root:
`C:/Users/xenod/AppData/Local/Temp/qwen-fp8-independent-2dpzvsgu`.

Read-only protocol canary:
`C:/Users/xenod/AppData/Local/Temp/qwen-fp8-canary-444welq7`.

Primary independently:
- Verified the sealed session and rehashed2,083 indexed task files, zero mismatches.
- Scanned task evidence for the actual gateway credential, zero occurrences; no profile/token was copied to evidence or model workspace.
- Reran original hidden, visible and overlap suites against `session/workspace-final` in the pinned Docker sandbox.
- Read the complete client-visible export and final diff; no nonempty thinking or compaction summary was available to review.
- Preserved original PAL before/after HEAD/status, including the existing untracked `.codex/config.toml`; only the two allowed candidate files changed.

The runner was initially permission-blocked; a user-controlled launch later completed. Exit0 means the runner finished, not model success: `result.json` explicitly records failed acceptance. No hidden result was returned to the model and no quality retry was performed.

Before scoring, the full instrument suite returned1967 passed,3 skipped,9 warnings in345.74s. Final focused rerun:13 passed in5.80s. Documentation check:208 Markdown files,717 relative links, the same8 pre-existing broken links; not a clean doc gate. `gh` remains unavailable, so tracker and required skill-feedback publication were not performed. No claim that every skill rule held.

Skips: idle8080 props, installed-Studio missing-binary case, and current insufficient-GPU-budget DFlash switch test. The remote control itself needs no local model GPU. Focused credential/timer tests passed13/13. A confirmed local-forwarder authentication finding was fixed red-first and independently re-reviewed before live use. Four simplify reviews were applied without changing the local recorder's admission behavior.

## 6. Conclusion

On this task, FP8 is more functionally complete than most admitted local submissions, ties Swift Q6 on the shared12-case coverage, and still falls short of GSQ's accepted result. This does **not** show that FP8 is generally worse than IQ3: different fine-tunes/templates/engines/provider settings and single trajectories prevent that inference.

The useful finding is that the provenance/exception-boundary failure also occurs on the FP8 service. Increasing numeric precision alone is not demonstrated to solve it. No default promotion, original-project patch, commit or push.
