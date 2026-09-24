# Remaining models on the same PAL task — 2026-09-21

**Result:** GSQ IQ3_S-MTP is the only accepted submission in the eight-configuration primary comparison: original hidden8/8, overlap audit4/4, and observed test-first workflow. Dirk Q6 and NVFP4 VERY-LOW pass the original suite but fail all four overlap cases. EXL3's admitted replacement also fails those cases. Its first, cleanup-excluded attempt passed both suites; that favorable evidence is retained and reported, not erased.

This is one task, one admitted attempt per configuration—not a general leaderboard, a quantization-quality law, or a production-promotion decision. No patch was applied to an original project.

Tracking: issue #92. [Plan](../plans/2026-09-21-remaining-pal-workflow.md). [Earlier Q4/Q6 PAL results](24-q4-real-project-workflow-2026-09-21.md). [Aggregate](../../qwen38-tuning/results/remaining-pal-2026-09-21/summary.json).

## 1. Same task and admission

The user asked to test the remaining previously screened models on the identical task. Actual Claude Code2.1.258 edited independent detached copies of PAL commit `200fcb9262d25e4002e30bf00c4364f55a42354e`, with no remotes/private original files. The original project remains at `D:/Github/pal-mcp-server`.

Verified across all five new sessions:
- Byte-identical task prompt to result24, SHA256 `eaa6548aa70cf6a2a1cd7d952a2abdccb9eb4a3b94009d5804916ec9dcd87456`.
- Same initial source bytes, two allowed edit files, context65536, medium effort, requested seed29, output8192,1800s/64-turn policy.
- Same effective tools: Read, Edit, fixed `mcp__visible-tests__run_visible_tests`. No model shell/network tool.
- Same original8-case hidden verifier and visible RED→GREEN hash contract, followed by the same independent4-case overlap audit. No oracle feedback or previous submission was shown to a candidate.
- Same client executable hash and Docker image identity as result24.

All tests of candidate Python ran in the existing unprivileged, network-disabled Docker sandbox with read-only workspace mounts. This is a real read/edit/test workflow, not the full everyday T4 skill/plugin/GitHub workflow; bare/restricted client policy remains.

## 2. Profiles

| Configuration | Selected runtime |
|---|---|
| Dirk UD-Q6_K | llama.cpp, stock template, MTP3+ngram24/16/64, tensor split8500,15468 |
| GSQ IQ3_S-MTP | llama.cpp, stock template, MTP3+ngram24/16/64, tensor split8500,15468 |
| NVFP4-MTP-VERY-LOW | Actual source-default worker profile from `production_nvfp4_argv(65536)`, including source memory admission check; not a substituted LOW artifact |
| EXL3 SC4.0bpw H5 | Native TP/cq4/gs9,15.5/ndt3, same OpenAI endpoint→adapter route as the preliminary incumbent screen |

Artifact, engine, source, config/tokenizer and client identities are frozen in private launch/protocol records. Both EXL3 shards are independently hashed. llama.cpp observations show66+0 residency. EXL3 uses explicitly labeled native-health normalization, not a fictitious native `/props` endpoint.

These are selected operating-point comparisons across engines/templates/weights, not pure quant ablations. GPU/process preflight found no game process before the batch; no unrelated server/process/container was stopped. Optional ongoing host process/RAM telemetry remains limited by absent psutil; do not infer continuous game-process observation from an empty list.

## 3. Primary eight-configuration comparison

**Time is the attempt through the original verifier, excluding the separate overlap audit, setup and later independent readback.** Failed rows have no time-to-success. Even accepted GSQ's271.125s is not a retrospectively invented end-to-end audited-completion wall: its separate audit took1.484s, and later human/agent review and waiting are outside that boundary.

| Configuration | Original hidden | Overlap audit | Visible suite | Attempt cost | Audited acceptance |
|---|---:|---:|---:|---:|---|
| **GSQ IQ3_S-MTP — new** | **8/8** | **4/4** | 6/6 | **271.125s** | **Accepted under this task's checks** |
| Dirk UD-Q6_K — new | 8/8 | 0/4 | 12/12 | 455.187s | Failed contract |
| NVFP4 VERY-LOW — new | 8/8 | 0/4 | 10/10 | 312.672s | Failed contract |
| EXL3 H5 — new, replacement | 8/8 | 0/4 | 7/7 | 197.000s | Failed contract |
| TURBO Q4 — result24 | 8/8 | 0/4 | 7/7 | 206.297s | Failed contract |
| Swift Q4 — result24 | 8/8 | 0/4 | 10/10 | 683.891s | Failed contract |
| TURBO Q6 — result24 | 7/8 | 0/4 | 5/5 | 177.218s | Failed contract |
| Swift Q6 — result24 | 8/8 | 2/4 | 9/9 | 333.563s | Failed contract |

All four new admitted sessions satisfy the measured RED→GREEN sequence, and only the two allowed files change. Passing visible tests still does not establish the full source-provenance behavior.

The discriminating case is unchanged: an explicit `CLI_CLIENTS_CONFIG_PATH` selection of a file/directory physically inside a current/legacy user directory must remain fatal. A file location is not its source category.

### EXL3 excluded attempt and infrastructure replacement

The first EXL3 session completed normally, retained complete wire/history evidence, and its saved submission independently passed original8/8, overlap4/4, visible10/10 and RED→GREEN. Attempt cost273.937s; it attempted an unavailable Bash tool once, which was rejected rather than executed.

However, the runner's immediate cleanup probe still connected to port8000 after owned-process stop returned. It correctly stopped the campaign and retained `evidence_failure`; independent later probes found the ports closed. The exact OS/child-listener release timing was not measured. The instrument fault was a one-shot cleanup check, not a demonstrated failure of the submitted PAL code.

A regression reproduced premature rejection (1 failed,13 passed). The runner now polls for release with a15-second deadline after stopping only its owned process; it does not kill by port or process name. Related tests passed27/27. Under the predeclared infrastructure-retry rule, EXL3 received one fresh unchanged task/baseline—not the audit or its earlier solution. Replacement cleanup passed, but its new solution failed all four overlap cases.

**Do not hide either result:** EXL3 produced a correct audited solution in the excluded session and an incorrect one in the admitted replacement. This is evidence of differing trajectories, not a stable pass probability. Requested seed29 does not establish deterministic whole-agent behavior; workspace/system metadata also differ. Both costs remain recorded:273.937s excluded plus197.000s replacement (470.937s of original-boundary attempts), before setup/audits/readback.

## 4. Thinking and execution quality

### GSQ: turns the right distinction into code

The emitted reasoning explicitly rejects parent-path classification because an explicit environment selection can overlap a user directory. It carries categorical origins (`bundled`, `explicit`, `legacy-user`, `current-user`) from discovery into loading, then downgrades only unsupported nonempty names from normal user origins. It also notices and repairs the empty-name pre-check before finishing, and preserves the existing path-based legacy deprecation warning rather than accidentally changing it.

Observed flow: baseline4 passed → add two parametrized failing cases → RED2 failed/4 passed → implement → GREEN6 passed. No tool error or compaction. Its own new tests are narrower than Dirk's, but its abstraction handles the external overlap audit correctly.

Limits: verbose/repetitive planning remains, and the final incorrectly describes existing malformed JSON as fatal. The code preserves the helper's actual skip behavior and passes the unchanged malformed-JSON oracle. Good code here does not make every explanation correct.

Evidence: private `gsq/review-visible.txt:81–85,208–212,366–386,413–447` and sealed `workspace.diff`.

### Dirk: broad test coverage, wrong provenance abstraction

Creates a dedicated `UnsupportedClientError` subclass and catches only that subclass, correctly avoiding broad error swallowing. Adds eight useful visible cases covering user skipping, bundled/explicit failure, incomplete configs, missing prompts and schema failures. Nevertheless, `_is_user_config_source` checks physical path prefixes, so all overlap cases fail. More tests and more reasoning did not fix the missing distinction.

One redundant baseline call, acknowledged in its emitted thinking;4 test calls overall, no tool errors. Repeated generic transitions such as "I need to investigate this further" add no corresponding new investigation at several edit steps. Its final overstates complete source-provenance coverage.

### NVFP4: narrow error handling, but parent-path assumption

Keeps the unsupported branch precise and returns `None` only for a purported user source; other resolution exceptions remain untouched. The classification is `source_path.parent in (USER_CONFIG_DIR, LEGACY_USER_CONFIG_DIR)`, which fails the overlap contract. Adds six visible cases but does not model overlapping authority sources.

Runs baseline three times, attempts to Read a directory once (error), then completes RED→GREEN. Its final says explicit paths cannot be treated as user-written, contradicting the implementation under overlap. No compaction.

### EXL3: contrasting trajectories

The excluded first attempt identifies that path equality is inadequate and deliberately carries the original search-base object, comparing identity with `is`. This passes the overlap audit on the existing iterator, though categorical source tags would be easier to maintain. It attempts an unavailable Bash tool once and recovers; visible RED→GREEN takes two test calls.

The admitted replacement instead uses file/parent-path equality, like NVFP4, and fails all four audit cases. It has no tool errors and also completes RED→GREEN in two calls. Both emitted thinking traces have frequent joined English words/poorly formatted illustrative snippets, while the final actual code is syntactically valid. This English task does not retest the earlier Thai-output finding.

## 5. New admitted-session metrics

| Metric | Dirk Q6 | GSQ IQ3 | NVFP4 VERY-LOW | EXL3 H5 replacement |
|---|---:|---:|---:|---:|
| Client-visible thinking characters | 32,408 | 25,786 | 30,756 | 19,220 |
| All backend generated tokens | 12,054 | 9,957 | 11,069 | 8,000 |
| Maximum actual input tokens | 26,110 | 21,806 | 26,338 | 30,812 |
| Read / Edit / test calls | 5 / 6 / 4 | 4 / 5 / 3 | 7 / 6 / 5 | 6 / 5 / 2 |
| Tool errors | 0 | 0 | 1 | 0 |
| Compactions | 0 | 0 | 0 | 0 |
| Engine prefill total | 42.808s | 30.115s | 29.356s | 44.966s |
| Engine decode total | 299.580s | 178.101s | 227.855s | 138.789s |
| Native fresh-token prefill aggregate | 604.5 tok/s | 645.4 tok/s | 827.5 tok/s | 694.8 tok/s |
| Native decode, time-weighted | 40.2 tok/s | 55.8 tok/s | 48.5 tok/s | 57.6 tok/s |

Engine counters explain cost, not correctness. EXL3 native prefill/cache semantics are not assumed identical to llama.cpp. No speedup claim is made from unpaired boots. No row approached its allocated65536-token window; this is not a maximum-context or Long Horizon certification.

## 6. Evidence and checks

Private roots under `C:/Users/xenod/AppData/Local/Temp/`:
- Main batch: `qwen-pal-remaining-8wqifpvm` (Dirk, GSQ, NVFP4, excluded EXL3).
- EXL3 replacement: `qwen-pal-remaining-9_hu1gcv`.
- Main independent saved-snapshot verification: `qwen-pal-remaining-independent-0uf2mbom`.
- Replacement independent verification: `qwen-pal-remaining-independent-npx71ghm`.

The primary session independently verified five sealed session inventories and rehashed8,435 main-batch files plus2,285 replacement files, with zero mismatches at readback. The subsequently added exclusion marker is a separate supplemental artifact. All five original/audit/visible suites were rerun against `session/workspace-final`, not mutable runtime workspaces. Primary reviewed all five complete client-visible thinking/final/tool exports and all five diffs. There were no compaction summaries to omit.

All five clone snapshots preserve HEAD/config with no remotes and no unexpected changes, including paths omitted by the older source-only inventory. Original PAL HEAD/status before/after agree exactly, retaining only the pre-existing untracked `.codex/config.toml`. No candidate patch copied back; no production/default change, commit or push.

Pre-run extended instrument gate:1951 passed,2 skipped,9 warnings. Final full idle
gate: **1953 passed,2 skipped,9 warnings in292.08s**, exit0. Skips are the idle8080
live-props check and installed-Studio missing-binary case; warnings are existing
aiohttp `NotAppKeyWarning`. `check-doc-links.py`:206 Markdown files,707 links,
the same8 previously documented broken links. `audit-stale-claims.py` reports1318
hits across184 files; this is not a clean documentation gate. Final checks found
ports8000/8080/18080 closed, no legacy lease and only the pre-existing redis
container running. Four simplify reviewers and a focused static security review
found no required changes in the initial extension. The later bounded listener-wait
repair received red-first regression and related-suite verification. `gh` is
unavailable; tracker/skill-feedback publication remains unperformed. The required
skill re-routing/review sequence was not fully repeated after that final repair;
this is not a claim of perfect skill compliance.

## 7. Decision

GSQ's categorical provenance approach is the strongest admitted result on this task despite its lower nominal quant. That establishes feasibility for this configuration and task—not general IQ3 superiority, reliable autonomous delivery, or that increasing/decreasing quant caused the outcome.

Keep GSQ as a candidate for a broader frozen real-task set with repeated/interleaved attempts. EXL3's first/replacement disagreement particularly argues against drawing reliability conclusions from one trajectory. Do not promote defaults, deploy these patches, or start a new quant sweep from this single task.
