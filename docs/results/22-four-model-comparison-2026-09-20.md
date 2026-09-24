# Four actual artifacts: time, throughput and quality — 2026-09-20

**Effort: medium, explicit in both request fields. Context allocation: 65,536 for every candidate.**

Status: results available for all four requested repositories. This is a bounded coding/language/math comparison, **not completion of the long-horizon promotion gate**. No production default changed. Tracked by issue #92.

## Artifacts and selected operating points

| Repository | Tested artifact | Tensor split |
| --- | --- | --- |
| `ukisai/Swift-Qwen3.8-27B-GGUF` | Q6_K | `9500,14500` |
| `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF` | MTP-Q6_K | `9500,14500` |
| `peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF` | UD-Q6_K | `8500,15468` |
| `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` | IQ3_S-MTP | `8500,15468` |

All use their artifact's embedded template. For Dirk that is embedded Sharp, **not** Stock Qwen and not the separate Sharp-on-NVFP4 experiment in result 21. The runner's `--template stock` means embedded metadata, not identical templates across artifacts.

All selected runs use `draft-mtp,ngram-mod`, draft max 3, ngram match 24/min16/max64, Q4_0 KV, tensor split, temperature 1.0, top_p .95, top_k 20, min_p 0, conditional Han mask, one slot and llama.cpp `b10499-1deefcca3`. Device proportions differ intentionally to obtain a viable operating point for each artifact. Manifests contain complete argv, actual model path, SHA256, GPU identity/driver, runtime context and layer evidence. All final validation and deep runs passed `66+0` and runtime context equality.

These are **selected tested configurations**, not proof of globally best configurations. The bounded search tested speculation off, MTP, MTP+ngram and a GPU redistribution for the larger Q6 artifacts. It did not exhaust all samplers, effort levels, quants or runtime flags. Do not describe the search as a global optimization.

Dirk pin: revision `52cb3e759635ab4605e08790b6c47df8adcf0744`, file `Dirk-Qwen3.8-27B-UD-Q6_K.gguf`, 21,983,696,416 bytes, SHA256 `06601c59c3dd1924c209603b3fc8369531127aa54974e1265a381c850bc37e0d`. Existing candidate identities remain in the manifests and extension plan.

## Evidence and experiment separation

Raw root: `qwen38-tuning/results/four-model-2026-09-20/`.

- `frozen-split-c65536/`: all four artifacts, speculation off, seed17, six tasks each.
- `spec-screen-c65536/`: MTP and MTP+ngram; includes failed Swift/TURBO boots.
- `rebalanced-spec-c65536/`: Swift/TURBO MTP and combined speculation with more memory assigned to the 4070 SUPER; all six requests completed per run.
- `validation-c65536/`: **three fresh rotated rounds**, seeds29/43/71, six tasks per artifact per round: 72 responses. No tuning results included in this validation table.
- `deep-c65536/`: one cold and two shared-prefix warm requests per artifact, 12 responses.
- `validation-summary.json`: exact table values and source response paths.
- `baseline-thinking-review.json`: main-session reading of all 24 baseline thinking/final pairs.

Rounds used separate server boots, not a same-boot A/B. Small cross-boot speed differences must not be treated as established improvements. The GSQ–Swift task-time gap below is too small for a confident ordering from these data.

## Primary metric: verified coding work

Two unique coding tasks (`merge_intervals`, `toposort`), each repeated with three seeds. Acceptance requires a complete final answer **and** execution against frozen assertions. Thinking alone never earns a pass.

Time per verified task = sum of all coding `time_to_response_s` / accepted coding attempts. In this runner that time includes preparation, request/generation and executable verification. All six coding attempts are included. Model load, checksum, tuning and manual-review time are excluded from this steady-service metric; failed boots remain in raw logs/queue records and are not silently relabeled successful tasks. This is not end-to-end multi-file agent throughput.

| Artifact | Code passes | Seconds / verified code task | Generated tokens / verified code task | Median engine decode tok/s on code | Thai repair exact | Basic math exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Swift Q6_K | 6/6 | 11.95 | 507.8 | 62.51 | 1/3 | 6/6 |
| TURBO MTP-Q6_K | 6/6 | 16.18 | 631.7 | 50.02 | 0/3 | 6/6 |
| Dirk UD-Q6_K | 6/6 | 21.49 | 803.7 | 53.61 | 3/3 | 6/6 |
| GSQ IQ3_S-MTP | 6/6 | 11.39 | 557.2 | 70.18 | 2/3 | 6/6 |

These are distinct dimensions, not an arbitrary weighted score. Thai repair exact-match is a single narrow incident fixture, not general Thai quality. Six math passes are repetitions of two elementary questions, not evidence that Swift's broader math issue is absent.

Dirk's round2 topological sort took 71.49 s request wall despite passing. Its slower aggregate is affected by thinking/output length rather than just decoder rate. TURBO round4 Thai repair hit 8192 generated tokens and `finish_reason=length`; it is a failure, not a fast successful answer. The full failed output is retained.

## Meaningful-depth prefill and warm task costs

The original captured Claude Code conversation is too large for this common allocation. **It was not silently truncated.** Instead, `deep-corpus-request.json` is an explicitly synthetic conversation containing 154,249 characters of real repository source, cut at a complete file boundary. `deep-corpus-provenance.json` records source hash and extraction. Runner case names such as `original-session-replay` are inherited labels: in this directory they do **not** establish original-session provenance.

All four served approximately45K real prompt tokens, exceeding half of the common allocation. This establishes survival for this workload/depth, not maximum context certification or real agent-history equivalence.

| Artifact | Cold prompt tokens | Cold prefill tok/s | Warm coding passes | Warm seconds / verified code task |
| --- | ---: | ---: | ---: | ---: |
| Swift | 45,271 | 703.73 | 2/2 | 12.46 |
| TURBO | 45,271 | 703.80 | 2/2 | 10.31 |
| Dirk | 45,422 | 694.91 | 2/2 | 15.29 |
| GSQ | 45,271 | 734.85 | 2/2 | 8.57 |

One deep round only: these are observations, not a paired performance ranking. Coding rows reuse ~45K prefix tokens; never label their short incremental prompt time as cold45K prefill. The main session read all 12 deep thinking/final pairs: algorithms and final solutions agree, and assertions passed. Dirk explicitly rechecks duplicate edges; Swift adds an unnecessary adjacency copy; neither is a correctness failure. Cold combinatorics answers were manually verified as `120` for all four.

## Language and reasoning evidence

All requests here use conditional Han masking. Therefore this campaign measures **guarded operating points**, not unguarded Chinese leakage or guard false-change rates. A zero-Han observation cannot clear the language gate.

Baseline complete-thinking review identified:

- TURBO notices `บอกร่อง` is unnatural but keeps it under a literal interpretation of preserving meaning. The final remains wrong.
- Dirk initially drafts `บอกรื่อง` and then self-corrects to `บอกเรื่อง`. It also invents that the ordinary prose prompt contains typos; a correct final does not make that reasoning claim true.
- GSQ and Swift sometimes describe Thai marks inaccurately despite producing the right final string. GSQ's prose rewrite also weakens clear eight-sentence adherence.
- Coding thought is generally consistent with final algorithms. Extra reasoning sometimes checks real edge cases, so length alone is not a loop detector or quality score.

Correction: the read-only reviewer initially claimed coverage of all72 validation thinking/final pairs, then disclosed that the claim preceded evidence from its own delegated reading. Full validation-trace reading coverage is therefore **unverified**, not complete. The main session did independently check the following load-bearing language findings against raw responses. TURBO round4 repeats `บอกร่อง` **1150 times** in thinking before truncating with no final. GSQ round2 explicitly reasons that `บอก` + `เรื่อง` equals the misspelling `บอกรื่อง`, then emits it. Swift rounds2 and4 emit unnatural `บอกร่องคุณภาพ`; Dirk repairs all three strings correctly but round2 topological-sort thinking contains10964 characters. These are concrete reasons to weigh quality and thought cost together. Ordinary drafting/checking/code repetition is not automatically a runaway loop; broader prose sentence-count findings remain qualitative rather than exact automatic grades.

## Failures and unresolved gates

The first fresh TURBO off-speculation attempt at65K failed CUDA allocation after dynamic VRAM-derived split changed from an earlier surviving8445:15468 to8338:15468. Frozen8500:15468 survived. MTP with that split then exhausted GPU1 on Swift/TURBO;9500:14500 survived for both without lowering context. This is evidence for the tested redistribution, not proof that every future desktop load will fit. Failed manifests/logs are preserved.

**No final daily-driver winner or default promotion.** GSQ and Swift are the leading candidates for wider verified-task evaluation in this bounded corpus; Dirk trades slower sampled coding for better sampled Thai repair. TURBO's failure is an optimization/reliability lead, not grounds to erase it from the comparison.

Still open: long-horizon multi-file/tool tasks, continuation nudges, duplicate edits, tool misuse/recovery, explicit abandonment counters, unguarded/guard A/B with false-change cost, harder held-out coding and math, complete end-to-end repair/retry cost, and wider best-config validation. In particular **TURBO has not passed its mandatory long-horizon gate**. These missing measurements are unknown, not zero.

## Instrument verification

After Dirk support, full idle gate:1757 passed,2 skipped,9 warnings. Frozen tensor split subsequently passed focused tests:30 passed. Final full idle gate after that change: **1758 passed, 2 skipped, 9 warnings in252.31s**. Documentation checks are not clean: the link checker found8 broken links (seven vendored Sharp archive links and one ledger link), and the stale-claim audit reported1316 hits across182 files. These broad repository findings were not silently fixed in this measurement slice. No commits, pushes, issue closures or default changes were made.
