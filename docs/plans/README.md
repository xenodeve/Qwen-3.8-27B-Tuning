# plans — what we intend to run, and what we asked outsiders

**These are intent, not results.** For results see
[`../reports/`](../reports/).

| # | document | what it is | status |
|---|---|---|---|
| 2026-09-22d | [PAL #149 focused arms](2026-09-21-fp8-pal-control.md) | Focused GSQ IQ3 and EXL3 PAL #149 runs after developer narrowed Q4-class attention | Completed focused slice: [result30](../results/30-pal-journal-focus-2026-09-22.md); GSQ accepted, EXL3 failed strict TDD gate |
| 2026-09-22c | [PAL #149 run journal](2026-09-21-fp8-pal-control.md) | Second real-project task from an open PAL issue; standalone storage-layer workflow across frozen arms | Partial: [result29](../results/29-pal-journal-2026-09-22.md); Swift Q4 timed out at1800.656s, remaining arms not run |
| 2026-09-22b | [Comparable gateway/local metrics](2026-09-21-fp8-pal-control.md) | DSH route diagnosis plus shared OpenAI/Anthropic stream observations, coverage-aware reporter and local/remote timing boundaries | Completed: [result28](../results/28-fp8-openai-pal-repeat-2026-09-22.md); original8/8, overlap2/4, effective request188.86 tok/s; no accepted complete task |
| 2026-09-22 | [claude-9arm FP8 PAL control](2026-09-21-fp8-pal-control.md) | Same frozen task through selected9arm FP8 route with isolated credential-free capture and unknown remote provenance | Completed: [result26](../results/26-fp8-pal-control-2026-09-22.md);8/8 original,2/4 overlap; no accepted complete task |
| 2026-09-21e | [Remaining PAL configurations](2026-09-21-remaining-pal-workflow.md) | Identical PAL task for Dirk Q6, GSQ IQ3, NVFP4 VERY-LOW and EXL3 H5, with independent overlap audit | Completed bounded evaluation: [result25](../results/25-remaining-pal-workflow-2026-09-21.md); GSQ accepted, three fail contract; first EXL3 cleanup-excluded evidence retained |
| 2026-09-21d | [Q4 and isolated PAL workflow](2026-09-21-q4-pal-workflow.md) | Q4 first, fixed Docker test tool, real PAL task and conditional Q6 controls; Minecraft-invalid attempts excluded | Completed bounded evaluation: [result24](../results/24-q4-real-project-workflow-2026-09-21.md); no submission passes full audited PAL contract |
| 2026-09-21c | [Preliminary CLI incumbent extension](2026-09-21-preliminary-cli-incumbents.md) | Same frozen code1 CLI task for source-default NVFP4 VERY-LOW and EXL3 4.0bpw H5 | Completed: [result23](../results/23-preliminary-cli-quality-time-2026-09-21.md); separate from Long Horizon |
| 2026-09-21b | [Preliminary Claude CLI quality/time screen](2026-09-21-preliminary-cli-screen.md) | Reduced first-pass screen: four existing selected profiles at65536, real CLI edits, parent hidden verification | Completed: [result23](../results/23-preliminary-cli-quality-time-2026-09-21.md); separate from Long Horizon |
| 2026-09-21a | [Long-horizon agent PRD and Luna handoff](2026-09-21-long-horizon-agentic-prd.md) | Canonical continuation: Q6 Swift/TURBO/Dirk versus GSQ IQ3, per-model certified context, real agent tasks, complete evidence, W0–W9 gates | Implementation incomplete; offline components tested; real client/GPU campaign pending under [issue #92](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/92) |
| 2026-09-21 | [Long-horizon evidence contract](2026-09-21-long-horizon-evidence-contract.md) | Recording schema, admission rules, privacy, integrated collection APIs and remaining gaps | Supporting specification for the PRD above; not a live result |
| 2026-09-20b | [Sharp, Swift and TURBO extension](2026-09-20-three-candidate-extension.md) | Three additional interventions after the GSQ comparison: controlled Sharp 2x2, Swift Q6, TURBO MTP-Q6 | Executing under [issue #92](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/92); no default change |
| 2026-09-20 | [GSQ comparison](2026-09-20-gsq-comparison.md) | IQ3_S-MTP against NVFP4 VERY-LOW and EXL3 H5, including original-session Thai replay | Nine-run bounded comparison complete: [result20](../results/20-gsq-time-quality-2026-09-20.md); full agent/repair/budget gates remain, [issue #92](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/92) |
| 00 | [Optimization plan](00-OPTIMIZATION-PLAN.md) | the original flag-tuning plan for Q4 | executed, superseded |
| 01 | [V3 Q1/Q2 test plan](01-V3-Q1-Q2-TEST-PLAN.md) | staged plan after Unsloth republished the repo | executed; reviewed by three agents in [report 14](../reports/14-PANEL-REVIEW.md) |
| 02 | [Research brief](02-RESEARCH-BRIEF-OPTIMIZATION-SURFACE.md) | self-contained brief sent to an external researcher, with hardware, prior results, and acceptance criteria. §10 records the final dispatched plan verbatim | sent; two replies reviewed in [17](../reports/17-EXTERNAL-RESEARCH-REVIEW.md), [18](../reports/18-RESEARCH-ROUND2-REVIEW.md) |
| 03 | [16-layer programme](03-SIXTEEN-LAYER-PROGRAMME.md) | every tunable layer, ordered by expected value on tok/s, context, VRAM and quality | executing; results in [report 20](../reports/20-SIXTEEN-LAYER-RESULTS.md) |
| 04 | [Revised plan 2026-08-21](04-REVISED-PLAN-2026-08-21.md) | what to do after the 16-layer sweep — supersedes 03's ordering | superseded by 06 |
| 05 | [Research brief 2026-08-21](05-RESEARCH-BRIEF-2026-08-21.md) | **For an external researcher.** Five open problems with the evidence for each, what was already ruled out, and the six external claims this project measured wrong |
| 06 | [Real-task benchmark](06-REAL-TASK-BENCHMARK.md) | **The runbook.** 19 real open issues from four repos, run against a same-model FP8 ceiling. Answers what context a real task needs, whether the skills earn their 38,064 tokens, and whether `UD-IQ2_XXS` is enough for T4 Labs' work | Phase 1 run 2026-08-22, its rows RETRACTED (CORRECTIONS §24); Phases 0 and 2-6 not started. **Two premises moved 2026-08-23** — the served window is not slow (96.92 tok/s, CORRECTIONS §26) and rotating between the 19 issues is nearly free (`-cram`, 343×). **Do not load the drafter for it.** See the banner in the runbook |
| 07 | [Dual-GPU open questions](07-DUAL-GPU-OPEN-QUESTIONS.md) | **A brief written for an outside reader**, self-contained: the machine, the build, the exact served invocation, every lever already swept with its verdict, the structural constraints (`-sm row` cannot load, no file-loaded drafter survives `-sm tensor`, `--cache-reuse` is unusable on a hybrid), and the ranked plan. Section 7 lists what NOT to suggest, so an external review does not re-tread closed ground | Written 2026-08-27 for external analysis. Its round-1 table is a snapshot of a sweep still running |

---

## Why 02 is worth reading even though it went to an outsider

It is the only document that states the machine, the workload, the metric, the
prior results and the measurement rules **in one place for someone with no
access to this machine**. If you need to explain this project to anything
outside it, start from that.

Its §4.3 lists the four ways the *previous* research went wrong, and §10 lists
the acceptance criteria every claim has to pass. Both exist because a reply that
looks thorough and invents its numbers costs more time than no reply at all.
