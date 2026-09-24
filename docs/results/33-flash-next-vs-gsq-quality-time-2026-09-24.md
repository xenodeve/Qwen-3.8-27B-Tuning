# 33 — Flash-Next vs GSQ IQ3_S-MTP: quality and time on the frozen tasks, 2026-09-24

**Question (developer):** does moving from Qwen3.8-27B to Qwen3.8-Flash-Next buy enough
quality to accept its lower speed?

**Answer on this evidence:** no quality gain was observed, and Flash-Next took **1.6×**
(code1) to **3.8×** (PAL) longer per task. Two samples per model; quality is **not
separable** at this n. Nothing at depth > 65,536 was measured, so Flash-Next's one
structural advantage (a 262K window) is untested here.

Machine aggregate: [`summary.json`](../../qwen38-tuning/results/flash-next-vs-gsq-2026-09-24/summary.json).
Private evidence: `%TEMP%\qwen-preliminary-cli-9nqpdf5a` (code1),
`%TEMP%\qwen-pal-remaining-tcossc4c` (PAL).

## Protocol

Same tasks, prompts, hidden suites, sandbox image, limits and verifiers as
[result 23](23-preliminary-cli-quality-time-2026-09-21.md) (code1) and
[result 25](25-remaining-pal-workflow-2026-09-21.md) (PAL): context **65,536**, effort
**medium**, seed 29, no retries, code1 1,200 s / 32 turns, PAL 1,800 s / 64 turns.
Changes, all recorded in each run's `frozen-protocol.json`:

- **Paired in one sitting, ABBA:** `gsq-a, flash_next-a, flash_next-b, gsq-b`. GSQ is
  re-run as the control rather than compared against 2026-09-21 numbers.
- **Client 2.1.281** (`~/.local/bin/claude.exe`) for both. The pinned 2.1.258 file the
  earlier runs hashed was already gone (its directory emptied 2026-09-23 08:31).
- **Flash-Next = the served 128k profile J** with only `-c 65536`/port/log changed:
  `gsq_compare.flash_next_argv`, drift-checked against `serve-flash-next.ps1 -WhatIf` by
  `bench/tests/test_flash_next_candidate.py`. `-sm layer -ts 31,17 -ncmoe 36`, expert
  cache 64, MTP n3 on CUDA1 + ngram-mod 24/16/64, `LLAMA_MOE_CACHE_MAX_TOKENS=4`,
  working-set trim after boot. Shards, MTP head and engine SHA-256 in `summary.json`.
- GSQ = result 22's selected point (`-sm tensor -ts 8500,15468`, MTP n3 + ngram 24/16/64).
- Runners: `tools/run-flash-next-cli.py`, `tools/run-pal-flash-next.py` (copies of the
  result-23/25 runners). No legacy supervisor lease; any `.port8080.lock` refuses.
- **Overlap audit** (the 4 cases result 25 used to separate models) ran after the
  campaign: `verify_pal_workspace` with `pal_registry_overlap_audit.py` as the suite, same
  sandbox image, on a copy of each final work tree.

## code1 (easy; all 27B rows passed in result 23)

| cell | hidden | task wall |
|---|---|---:|
| gsq-a | pass | 112 s |
| flash_next-a | pass | 182 s |
| flash_next-b | pass | 186 s |
| gsq-b | pass | 116 s |

Flash-Next **1.6×** slower (184 vs 114 s mean); both pairs within 4 s of each other.

## PAL (the task that separated 27B quants in result 25)

| cell | hidden | overlap audit | red → green | task wall | accepted |
|---|---|---|---|---:|---|
| gsq-a | 8/8 | **4/4** | yes | 345 s | **yes** |
| flash_next-a | 8/8 | 0/4 | yes | 755 s | no |
| flash_next-b | 8/8 | **4/4** | **no** (6 test runs, never red first) | 1,504 s | no |
| gsq-b | 8/8 | 0/4 | yes | 244 s | no |

- Each model failed one of two attempts, in different ways. GSQ 1/2 accepted,
  Flash-Next 0/2 — not a difference at n = 2.
- Flash-Next **3.8×** slower (1,130 vs 295 s mean).
- **GSQ-b failing the audit revises result 25's reading**: "GSQ alone passes all 12"
  was one sample and GSQ does not pass reliably. Result 25 already called itself
  single-task evidence; this is the first repeat.

## Thinking (all eight sessions)

Counted from `session/stdout.jsonl`. Every thinking block was non-empty; no 40-char
repetition above 1 %; no Han characters. code1: Flash-Next thought **2.4–4×** less
(1,963 / 2,103 chars vs 4,989 / 7,984) for the same pass. PAL: similar volume
(17–37K chars); flash_next-b took 70 turns / 37 tool calls.

What separated the audit results is visible in the thinking (keyword search, then the
quoted sentence read by hand — not a graded review):

| cell | the overlap case in its thinking | audit |
|---|---|---|
| flash_next-b | *"Edge: env_path pointing directly to a file inside the user dir?"* — pursued | 4/4 |
| flash_next-a | *"If the user sets the env var to ~/.openclink/cli_clients, that's both."* — noticed, not handled | 0/4 |
| gsq-a | quotes the "explicit … must remain fatal" rule 15× and implements it | 4/4 |
| gsq-b | quotes the rule, never considers the overlap | 0/4 |

Flash-Next saw the edge case in both attempts, GSQ in one — suggestive, not a result.

## Not measured

Depth above 65,536; Thai spelling (PAL prompts are English; code1 finals were Thai,
547–728 Thai chars, not checked character by character); more than two attempts per
model; the second PAL task (#149).
