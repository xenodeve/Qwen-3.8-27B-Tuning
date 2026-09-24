# GSQ comparison with the two local incumbents

Status: resumed under the user's time-per-verified-task criterion. The earlier
language-only stop decision is superseded. No default promotion authorized.
Tracking issue: https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/92.

## Accepted scope

Evaluate the user-provided GSQ-RCO plan, starting with IQ3_S-MTP, against the
actual NVFP4 VERY-LOW llama.cpp artifact and EXL3 SC4.0bpw-H5 Profile G.
Quality screening precedes expensive tuning. EXL3 is an end-to-end reference,
not an isolated quantization comparison. Smaller GSQ variants, vision, and
single-GPU work follow only after the primary candidate passes screening.

## Frozen identities

- Repository starting HEAD: d769a678120c2738ac914ace620672ce9e02ed84.
- Working tree already contains user changes; preserve them.
- GSQ upstream: ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF,
  revision d562806dbafae37109975e970aae91b43e73b440.
- Candidate: Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf, 12120016960 bytes,
  expected SHA256 58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f.
- NVFP4 profile source: qwen38-tuning/scripts/worker-q4-dual.ps1;
  esatapedico snapshot bcd7a7d3e251d4ec0fd15c72584b5eb9e0981383,
  Qwen3.8-27B-NVFP4-MTP-VERY-LOW.gguf. SHA256:
  74ea17ea05e0e0241af8d5b29cdea38b3f4509f66d9b96c1ab05f0e1f0e537d9.
- EXL3 profile source: qwen38-tuning/scripts/serve-exl3.cmd;
  models/turboderp-Qwen3.8-27B-EXL3-SC4.0bpw-H5,
  Profile G normally allocates 262144; this comparison overrides it to 147456.
  Both shard hashes are recorded in each validation manifest under
  qwen38-tuning/results/gsq-time-quality-2026-09-20/validation/.

## Sequence and acceptance

1. Baseline tests, artifact hashes, build/GPU/driver records, profile resolution.
2. Candidate compatibility without speculation and short quality screening.
3. Frozen original session 096ebcae-a6ae-4b80-b608-3b60f60a3fac replay:
   retain roles/tool results and audit the actual token count. The reported
   incident depth is approximately 143.2k; do not substitute an assistant-only
   extract, concatenated sessions, synthetic padding, or a fresh conversation.
   Missing original request components must be stated, not reconstructed silently.
4. Matched llama.cpp comparisons with speculation off, then native MTP tuning;
   separate production-profile comparison against EXL3.
5. Alternate artifact order across at least three rounds; report uncertainty
   at the measured depth. Never import a historical noise constant blindly.
6. Validate larger contexts with substantial measured requests and generation.
7. Full quality and stability gates; optional tracks only if primary passes.

For Thai quality retain raw text and every active filter/guard. Known-pattern
counts are diagnostic signals, not spelling accuracy. Keep clean controls and
manual assessments separate from rule-derived counters. A loop stopped by a
guard is a failed generation, not a repaired answer.

Record allocated context, complete prompt tokens, cached/new tokens when
available, output tokens, finish reason, first-token/wall time, engine-native
timings, exact request, server log, process exit, per-GPU memory, and artifacts.
Retain outputs even when a measurement is invalid. Do not rank truncated,
copied, or failed responses as successful speed results.

## Change inventory and ownership

- Existing profile/benchmark sources: inspect and reuse before adding code.
- New comparison records under qwen38-tuning/results/gsq-2026-09-20/.
- Experimental profile integration only after compatibility is established.
- New bench primitives require incident-based tests observed red first.
- Fixture/runner inventory: delegate read-only lookup via clink.
- GPU ownership, run design, integration and final verification: main agent.

Completion may be a documented rejection or compatibility failure. No requirement
to manufacture successful MTP or a context ceiling when a prerequisite fails.

## Initial preflight (historical)

Full existing bench test suite was launched before edits; failures were diagnosed
below and the pre-validation suite passed 1741 tests with 2 skips. GPU inference
servers were absent at preflight. Downloads alone did not authorize inference.

## Preflight findings

- First suite: 1708 passed, 18 failed, 2 skipped. Seventeen failures were missing
  CUDA_PATH for the external Unsloth binary preview. One was CRLF checkout of the
  frozen real-code.txt: LF bytes reproduce its expected 5672a9bcce74c0d0 hash.
- Set CUDA_PATH only for test processes; pinned corpus text to LF in .gitattributes.
  Full re-run: 1726 passed, 2 skipped; baseline-tests.xml retains the result.
- Four new comparison tests separately observed red then green (artifact identity,
  unknown artifact refusal, measured-vs-allocated depth, tool-history preservation).
- Original incident message msg_3d8ae2c199ef (2026-09-16T08:30:54.277Z) reports
  input 10844 + cache read 143104 = 153948 tokens. Thai corruption is visible.
- Current Claude Code resume capture preserves the source session hash and carries
  56 messages and 27 tools. EXL3 count_tokens reports 140726. This is original-session
  replay with changed client context, NOT exact incident-depth reproduction.
- Short screen's bracket_matching score is invalid: its seventh assertion passes
  characters [40,39,92,39,41] (escaped closing quote), yet expects balanced brackets.
  Both incumbents return False consistently with the prompt's escape semantics.
  Preserve those rows; replace this screen case with frozen toposort for all arms.

## Gate decision

SUPERSEDED by the user directive below; retained as history only.

GSQ short prose leaked Han and original-session replay produced extensive Thai
orthographic corruption plus Han. Retain the artifact as research-only and do
not expand to large speculative sweeps, smaller variants, vision or soak at this
failed quality gate. This is a bounded negative screen, not completion of the
entire supplied integration plan. Current production defaults remain unchanged.

## Revised acceptance: user directive after the initial screen

Primary objective: minimize elapsed time to a verified successful task while
retaining task quality. Prefill and decode rates explain elapsed time; neither
alone ranks the candidates. Count reasoning, tool work, validation, repairs and
retries in task time. Report failed/unfinished tasks and total campaign time,
not just latency averaged over survivors. Model load time is recorded separately.

Use each artifact's best locally validated configuration at ONE shared allocated
context, initially 147456 (the smallest currently exercised normal window).
If an arm cannot support the complete workload, reduce the window for ALL arms
or report the incompatibility; never silently truncate the history. This latest
directive explicitly overrides the prior fixed Profile G allocation requirement.
Preserve the Profile G recipe except for the common allocation when comparing EXL3.
Report actual rendered prompt lengths as well: different templates may differ.

NVFP4 and EXL3 start from measured incumbent recipes, not claims of global
optimality. GSQ has no validated best configuration yet: screen native MTP draft
depths and relevant sampler settings before freezing its candidate. Separate
tuning data from final evaluation and rotate artifact order across >=3 rounds.
Keep reasoning effort medium for the initial comparison; tuning effort later
requires measuring its time/quality trade rather than assuming shorter is better.

Retain complete thinking and final text, plus streaming arrival timestamps where
available. Judge visible repeated drafting, self-checks, abandoned approaches,
unsupported claims and tool planning against the task outcome. Do not infer
unobserved internal cognition or exact token/phase timings from character counts.

Chinese leakage and Thai corruption are repairable defects to evaluate, not an
automatic artifact rejection. First test targeted mitigation (sampler/copy rules,
conditional language guard); score both raw and repaired outputs and include
mitigation latency and false changes. Preserve code, tool payloads and names.
Do not let a text filter disguise a failed or truncated task as a success.

Report a time/quality tradeoff table, with code correctness, grounding, language,
instruction adherence and stability separate. Do not invent a single weighted
score without declared weights. Prefer faster verified completion at comparable
quality; show unresolved tradeoffs for selecting the base to optimize next.
