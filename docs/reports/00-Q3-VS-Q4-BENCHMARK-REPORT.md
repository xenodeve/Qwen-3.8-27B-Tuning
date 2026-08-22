# Qwen3.8-27B Local Coding Worker — Q3 vs Q4 Benchmark Report

> **Status:** complete for 16K context · **Date:** 2026-08-18 UTC+7
> **Answers:** `docs/plans/00-OPTIMIZATION-PLAN.md` Phases A, B, B2, C (partial C2)
> **Supersedes on measured points:** `docs/researchs/Qwen3.8-27B_Optimization_Research_Docs/` 02, 03, 04
> **Raw artifacts:** `C:\AI\qwen38-tuning\` — `EXPERIMENTS.md` (E0–E6), `results\*.jsonl`, `results\summary.md`, `scripts\`, `bench\`, `logs\`

---

## 0. Verdict

**Run `UD-Q4_K_XL` with built-in MTP speculative decoding at `--spec-draft-n-max 2`.**

The plan framed this as a reliability lane versus a performance lane. That framing
does not survive measurement: **Q4 wins on both axes simultaneously.**

| | **UD-Q4_K_XL + MTP n=2** | UD-Q3_K_XL + MTP n=2 |
|---|---|---|
| **Verified tasks / hour** | **33.6** | 22.2 |
| **Pass rate** | **90.0 %** (27/30) | 86.7 % (26/30) |
| Median tok/s across the task suite | **10.56** | 8.73 |
| Synthetic decode | **10.67** | 8.88 |
| Code-rewrite decode | **12.10** | 10.30 |
| Wall clock, same 30 tasks | **2 889 s** | 4 213 s |
| Tokens generated for the same work | 29 363 | 34 543 (+18 %) |
| Reasoning emitted | 82 653 ch | 103 486 ch (+25 %) |
| Disk footprint | 16.69 GiB | 12.52 GiB |
| Top-1 agreement vs BF16 (vendor proxy) | ~96 % | ~92.4 % |

**Q4 is 51 % more productive per hour.** Q3's disadvantage compounds — it decodes
slower *and* spends more tokens reaching an answer that is slightly worse.

Per plan §9 the selection rule is verified successful coding tasks per hour, not
smallest file, highest proxy, or highest raw tok/s. On that rule the answer is Q4,
and it is not close.

---

## 1. Machine and runtime

```text
OS       Windows 11
CPU      Intel Core i5-13500
GPU      RTX 4070 SUPER, 12282 MiB
RAM      48 GB DDR5
Runtime  C:\AI\llama.cpp-cuda
Build    llama.cpp b10472 / commit 60eeeb608, CUDA 12.4, driver 610.88
Context  16384 for every measurement in this report
```

Free VRAM before load ranged **9 933 – 10 530 MiB** across 22 recorded launches.
`--fit on` derives the layer split from whatever is free at boot, so runs from
different boots are not comparable. Every comparison in this report is
within-sweep, with the environment snapshotted before each load.

---

## 2. Production configuration

```powershell
C:\AI\llama.cpp-cuda\llama-server.exe `
  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL `
  --alias qwen38-q4 `
  -c 16384 `
  -ngl auto --fit on -fa on -np 1 `
  --no-mmproj-auto `
  --spec-type draft-mtp --spec-draft-n-max 2 `
  --host 127.0.0.1 --port 8080
```

Scripts: `qwen38-tuning\scripts\production-q4.ps1`, `production-q3.ps1`.

`--jinja` is omitted deliberately — already enabled by default in b10472, so
passing it is a no-op.

### 2.1 Client-side settings the server will not supply correctly

| setting | value | reason |
|---|---|---|
| `min_p` | **0.0** | server default is 0.05; the vendor specifies 0.0 for **both** thinking and non-thinking |
| `temperature` / `top_p` | 1.0 / 0.95 thinking · 0.7 / 0.80 non-thinking | two distinct published profiles; a single server default cannot serve both |
| `presence_penalty` | 0.0 thinking · 1.5 non-thinking | same split |
| `chat_template_kwargs.reasoning_effort` | send explicitly | template defaults to `xhigh` and silently remaps `high` → `xhigh`; accepts only `low`, `medium`, `xhigh` |

Operational reasoning profile used throughout: **`medium`**.

---

## 3. Speculation matrix (plan Phase C, protocol §6)

Same procedure for both quants: server restarted per config, environment snapshotted
before load, N=3 generations of 160 tokens, greedy equivalence sample captured.

Two prompt types, because the prompt turned out to decide the answer:

- **bench** — an 11-token instruction with nothing in context to copy.
- **code** — a long prompt containing the exact class the model is asked to rewrite
  with one attribute renamed. This is what a coding agent actually does most of the
  time, and it is the case llama.cpp's docs cite for `ngram-simple`.

| quant | spec | n_max | bench tok/s | code tok/s | acceptance (bench / code) |
|---|---|---|---|---|---|
| Q4 | none | — | 8.24 | 8.22 | — |
| Q4 | ngram-simple | 4 | 8.29 | 8.37 | — / 30.8 % |
| **Q4** | **draft-mtp** | **2** | **10.67** | **12.10** | **78.1 / 98.0 %** |
| Q4 | draft-mtp | 3 | 9.91 | 12.03 | 70.3 / 88.8 % |
| Q3 | none | — | 9.01 | 9.25 | — |
| Q3 | ngram-simple | 4 | 9.16 | 9.08 | — / 30.8 % |
| **Q3** | **draft-mtp** | **2** | 8.88 | **10.30** | 77.5 / 96.4 % |
| Q3 | draft-mtp | 3 | 7.27 | 9.92 | 64.1 / 99.1 % |

### 3.1 Draft-depth sweep (plan Phase C2, Q4 lane)

| n_max | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| tok/s range | **10.58–10.78** | 9.49–11.13 | 9.37–11.08 | 8.59–9.38 | 7.18–10.45 |
| acceptance | **77.5 %** | 68.6 % | 65.2 % | 56.4 % | 52.4 % |
| VRAM free after load | 772 MiB | 942 | 1155 | 1089 | 1275 |

Free VRAM *rises* with `n_max`, meaning `--fit` reserves progressively more for the
draft path and evicts target layers from the GPU. High depth is penalised twice:
falling acceptance and falling residency.

n=3 has a higher ceiling but a lower floor and a much wider spread. For a
throughput metric, **n=2** is correct: highest floor, tightest range, best acceptance.

### 3.2 Output equivalence (protocol §5)

Greedy samples (`temperature 0, top_k 1, seed 42`) were **byte-identical across every
speculative configuration** within each quant — Q4 `6F8AAC2789…`, Q3 `0659173109…`.

The greedy divergence reported for quantized targets in llama.cpp issue #25618
**does not reproduce** on this build and model. Plan §6's
`repeatable output corruption` stop condition is not triggered. MTP is a pure
performance toggle here.

---

## 4. Quality benchmark (plan §13, protocol §13–14)

10 coding tasks × 3 attempts × 2 configs = 60 samples. Each reply's code is
extracted and **executed against assertions in a subprocess with a 20-second
timeout**. Pass/fail is machine-verified — no LLM judge, no partial credit, no
human scoring. Tasks were chosen to probe what quantization damages: eviction
order, tie-breaking, cycle detection, operator precedence, transpositions.

Corpus and harness: `qwen38-tuning\bench\tasks.py`, `run_bench.py`.

| task | difficulty | Q4 | Q3 | Q4 wall | Q3 wall |
|---|---|---|---|---|---|
| `lru_cache` | easy | 3/3 | 3/3 | 188 s | 180 s |
| `merge_intervals` | easy | 3/3 | 3/3 | 129 s | 169 s |
| `bracket_matching` | easy* | 0/3 | 0/3 | 337 s | **889 s** |
| `toposort` | medium | 3/3 | 3/3 | 207 s | 252 s |
| `expr_eval` | medium | 3/3 | 3/3 | 572 s | 603 s |
| `rotated_search` | medium | 3/3 | 3/3 | 159 s | 139 s |
| `text_wrap` | medium | 3/3 | 3/3 | 253 s | 294 s |
| `lfu_cache` | hard | **3/3** | **2/3** | 514 s | 793 s |
| `damerau` | hard | 3/3 | 3/3 | 202 s | 367 s |
| `tree_codec` | hard | 3/3 | 3/3 | 328 s | 526 s |

\* Labelled easy, but it requires ignoring brackets inside quoted string literals
with backslash escapes. **Neither model ever solved it** — a capability ceiling of
Qwen3.8-27B at this quant range, not variance, and not a discriminator between quants.

**The entire measured quality gap is one task: `lfu_cache`, 3/3 versus 2/3.**

`bracket_matching` is the clearest illustration of the compounding penalty: Q3 spent
**2.6× the wall clock and still failed**.

---

## 5. Findings

### 5.1 MTP is the dominant speed lever — not the quant choice

| change | speed effect (bench / code) | fidelity cost |
|---|---|---|
| Q4 → Q3, −4.17 GiB | +9 % / +13 % | −3.6 pts top-1 |
| enable MTP on Q4 | **+30 % / +47 %** | **none measured** |

Giving up 4.17 GiB and 3.6 points of fidelity buys about 10 %. Turning on a flag
that already ships inside the GGUF buys 30–47 % and costs nothing.

Research doc 02 §5 and the plan's phase reordering rest on "quant size is a major
speed knob". The reordering was still the right call — it just turned out to answer
*which quant*, not *where the speed is*.

### 5.2 MTP compensates for CPU offload, so it helps Q4 more — mechanism

Actual layer placement, read from the verbose (`-lv 5`) load report rather than
inferred from VRAM totals:

| | layers on GPU | layers on CPU | MTP gain (bench / code) |
|---|---|---|---|
| **Q4** | 32 | **33 (51 %)** | **+30 % / +47 %** |
| **Q3** | 43 | 22 (34 %) | **−1 % / +11 %** |

Q4 additionally allocates: `CUDA0 KV 512.00 MiB`, `CUDA0 RS 205.73 MiB`,
`CPU RS 243.14 MiB`, `CUDA0 compute 189.70 MiB`, and a separate `64.00 MiB` draft
KV — confirming the qwen3_5 hybrid recurrent architecture in practice.

Draft acceptance is nearly identical between the quants (77.5–78.1 % bench,
96.4–98.0 % code), so this is **not** a draft-quality difference.

Speculative decoding amortises the cost of **one forward pass** across several
tokens. With 51 % of weights CPU-resident a forward pass is very expensive and
batching the verification is a large win. At 34 % the pass is already cheaper, the
constant draft overhead is unchanged, and the net gain collapses — on the short
prompt it goes slightly negative.

> **MTP is a compensation mechanism for poor VRAM fit, not an independent
> accelerator. The worse a model fits, the more it repays.**

This directly **inverts** continuation §3 and research doc 04 §7, which predicted
`UD-Q3_K_XL + MTP` as the likely performance winner on the theory that Q4 was too
VRAM-saturated to host MTP.

### 5.3 Prompt type moves the MTP result more than any tuning knob

| prompt | Q4 + MTP n=2 | acceptance |
|---|---|---|
| short instruction, nothing in context | 10.67 tok/s | 78.1 % |
| **rewrite the class given in the prompt** | **12.10 tok/s** | **98.0 %** |

Benchmarks that measure speculation with a short instruction prompt **understate**
what an agent workload will see. Future measurements must use a context-bearing prompt.

### 5.4 `ngram-simple` is not competitive here

Tested on both quants and on the prompt type its own documentation names
(source-code rewriting): **30.8 % acceptance**, converting to 8.37 vs 8.22 tok/s —
inside run-to-run noise. On a short prompt it drafts nothing at all.

An earlier `ngram-mod` result was dismissed on an unfair test (short prompt); this
retest was fair and reached the same verdict. Recommend dropping the ngram arm
unless a long-context test revives it.

### 5.5 Reasoning effort — inconclusive, and the reason matters

Plan Phase B2 hypothesis was that `reasoning_effort=xhigh` is the largest single
cost. Measured on a fixed agentic task, 2 runs per level:

| effort | wall_s | reasoning chars | rounds | bad tool args | reached patch |
|---|---|---|---|---|---|
| low | 67.8 / 85.3 | 384 / 570 | 3 / 4 | 0 | yes / yes |
| medium | 85.6 / 50.1 | 621 / 212 | 4 / 3 | 0 | yes / yes |
| xhigh | 84.2 / 106.9 | 632 / 1008 | 4 / 4 | 0 | yes / yes |

Reasoning was 212–1008 characters, roughly 50–250 tokens — **not** the
multi-thousand-token blocks the hypothesis assumed. Within-level spread (`medium`
spanned 50.1–85.6 s, 71 %) swamps between-level difference, and the run used n=2
after the project had already adopted an N≥3 rule. The task was also too easy: all
six runs produced a correct patch with zero malformed tool arguments.

What the data does support is more useful than the original question:

```text
3 rounds -> 67.8 s, 50.1 s           ~17-23 s per round
4 rounds -> 85.3, 85.6, 84.2, 106.9  ~21-27 s per round
```

Cost per agent round is near-constant. **Wall clock per task is driven by the number
of tool rounds, not by reasoning verbosity.** For a tasks-per-hour metric the lever
is fewer agent steps, not shorter thinking.

`medium` adopted provisionally. Revisit with a harder task and N≥5 during the real
workload phase.

### 5.6 Protocol correctness — all gates passed (plan Phase B)

| gate | result |
|---|---|
| plain completion | PASS |
| developer-role behaviour | PASS — merged into system by the template |
| simple tool call | PASS — `finish_reason: tool_calls` |
| nested object arguments | PASS — 3-level nesting + array, valid JSON |
| tool result → continuation | PASS |
| repeated tool loop | PASS — 2 tool rounds then correct synthesis |
| reasoning separation | PASS — see below |
| `min_p = 0.0` | applied per request |

**Reasoning separation is a non-issue.** `/v1/chat/completions` returns a separate
`reasoning_content` field; `content` is clean. The `reasoning_format: none` value
visible in `/props` governs `default_generation_settings`, which applies to the raw
`/completion` endpoint, not the OpenAI-compatible chat endpoint. No
`--reasoning-format` or `--reasoning-preserve` flag is needed for the basic case.
Plan Phase B item "no unwanted think leakage" is closed.

**Tool-call round trip works.** The Qwen3.8 wire format is XML
(`<tool_call><function=NAME><parameter=ARG>…`), and llama.cpp converts it correctly
into OpenAI `tool_calls`, including nested objects, arrays, multi-round loops and
`tool_call_id` correlation. The highest-risk integration point named in the plan is
clear; remaining risk moves from "can it parse" to "does it stay correct over long
sessions", which only a real workload can answer.

One instruction-drop was observed: a tool call omitted an instructed but
non-`required` field (`notify: true`). n=1, not a conclusion — but a schema
validator would not catch it, and downstream it would present as silently wrong
agent behaviour.

---

## 6. Corrections to the planning and research documents

| claim | source | status |
|---|---|---|
| ~2.5 GB MTP VRAM overhead | video, research 04 §2.5 | **wrong** — `blk.64.*` tensors total **285.8 MB**; measured VRAM went *down* |
| 4–5 draft-step sweet spot | video, research 04 §10 | **not reproduced** — peak is n=2–3; n≥5 regresses |
| 3× speedup | video | **not reproduced** — 1.30–1.47× here |
| `Q3 + MTP` is the performance lane | continuation §3, research 04 §7 | **inverted** — Q4 + MTP is faster on both prompt types |
| quant size is the dominant speed knob | research 02 §5 | **weak** — worth ~10 %; MTP is worth 30–47 % |
| `ngram-simple` is a serious candidate | research 04 §7 | **not here** — no measurable gain on either quant |
| think-block may leak into `content` | earlier machine report | **wrong** — separate `reasoning_content` field |
| `reasoning_effort=xhigh` is the largest single cost | earlier machine report | **overstated** — 50–250 reasoning tokens; effect below the noise floor |
| llama.cpp issue #25618 greedy divergence | research 04 §6 | **does not reproduce** on this build/model |
| free VRAM is a fixed 11 069 MiB | earlier machine report [C2] | **wrong** — ranges 9 933–10 530 MiB, a real `--fit` confounder |
| §15 benchmark figures · §6 size proxy · §4 State-4 fidelity reasoning | plan / research 02 | **confirmed accurate** |

---

## 7. Method notes and known measurement defects

Recorded so later readers can weight the numbers correctly.

- **N≥3 with reported spread** for all timing claims; conclusions rest on
  non-overlapping ranges, never point estimates. Baseline noise on this machine is
  ~18 % run to run, so any effect smaller than that is unmeasurable in one run.
- **Median bug, since fixed.** An early sweep script wrote `tg_median` that was
  actually the maximum: `[int](3/2)` is 2 in PowerShell, not 1, because `[int]`
  rounds half to even. Affected tables were relabelled min–max, and the conclusions
  drawn from them rest on range separation, so they stand.
- **Silent first-row drop, since fixed.** PowerShell 5.1 `Add-Content -Encoding utf8`
  writes a BOM on first write; the Python report generator parsed with plain
  `utf-8` and its `except JSONDecodeError` silently discarded line 1 of every
  results file — the baseline row of every table. Now reads `utf-8-sig` and warns
  on unparseable lines.
- **stderr is not failure.** `llama-server`, `llama-server --version` and
  `nvidia-smi` all write normal output to stderr. Under Windows PowerShell 5.1 with
  `$ErrorActionPreference = 'Stop'` the first such line raises a terminating
  `NativeCommandError`, which killed the launcher before the port was ever bound and
  presented as "llama.cpp failed to start". All automation now drops to `'Continue'`
  around native calls.
- **CPU contention observed.** Session MCP servers held CPU during the quality run;
  two `expr_eval` samples dipped to 8.5–8.9 tok/s. Throughput otherwise matched the
  isolated matrix (10–11 tok/s), so contamination appears minor but non-zero.
- **`tok_s` in the quality harness is content-only.** `usage.completion_tokens`
  excludes reasoning tokens, so that column understates true decode rate. Wall clock
  and pass rate are the decisive columns.
- **`/props` cannot confirm MTP state.** It reports `speculative.types = none` even
  when MTP is active. Confirm from the load log line
  `common_speculative_init_result: creating MTP draft context against the target model`.

---

## 8. Not tested — open risks

1. **Only 16K context was measured.** Every number here is at `-c 16384`. Scaling the
   *measured* `CUDA0 KV = 512 MiB` at 16K linearly gives **~8 GiB (F16) at 256K**, or
   ~4 GiB at Q8_0. That is enough to change the layer split completely and could
   reverse the ranking, since Q3 leaves 4.17 GiB more room.
   **This is the single most likely thing to invalidate the verdict.**

   *Correction:* an earlier draft of this section quoted ~16 GiB, carried over from
   the proxy estimate in research doc 07 §4. That proxy assumes every layer holds a
   growing KV cache. It does not: `qwen3_5` is hybrid, with roughly 16 of 64 layers
   doing full attention and the rest Gated DeltaNet, whose recurrent state is
   constant-size — visible on this machine as separate `RS buffer` allocations
   (205.73 MiB CUDA0 + 243.14 MiB CPU) that will *not* grow with context. The
   measured 512 MiB anchor from this exact model supersedes the proxy.
2. **Prompt-cache and multi-turn reuse untested** (plan Phase G). Agent economics
   depend on prefix stability, not total context length.
3. **KV placement and precision untested** (plan Phases H, I) — `--no-kv-offload`,
   `q8_0`. Also untuned: `--fit-target`, `--fit-ctx`, and `--cache-ram` (default
   8192 MiB against ~11 GB free host RAM — plan Phase J flags this as a paging risk).
   All three flags were verified present in b10472.
4. **VRAM headroom is thin and variable.** `--fit on` leaves 450–1 275 MiB free
   depending on config. A desktop app claiming VRAM mid-session can force driver
   eviction. Consider an explicit `--fit-target` margin in production (plan Phase E).
5. **Temperature 1.0 only.** The planned 0.6 comparison (plan Phase B3) was not run;
   the head-to-head was prioritised.
6. **No OpenCode / OpenClink / real-repo run** (plan Phases L, M, N). These are
   synthetic single-file coding tasks, not multi-file agent work with a live tool loop.
7. **AtomicChat challenger not evaluated** (plan Phase D). Its trigger condition —
   "Q3 too weak on verified tasks **and** Q4 too residency-bound" — is not met, since
   Q4 won outright. Adding it now would not answer a live question.

---

## 9. Recommended next steps, in order

1. **Context-depth sweep, 32K → 64K → 128K → 192K → 256K, on both quants**
   (plan Phase F). Measure cold prefill, generation at depth, and incremental cached
   turn separately. This is the only open question that could change the verdict.
2. **Prompt-cache and multi-turn behaviour** (plan Phase G), since agent cost is
   prefix-driven rather than length-driven.
3. **KV precision F16 vs Q8_0 at depth** (plan Phase I) — directly buys context headroom.
4. **`--fit-target` margins** (512 / 1024 / 1536 MiB) for stability under desktop load.
5. **Temperature 0.6 vs 1.0** on the winning config (plan Phase B3).
6. **OpenCode integration**, then OpenClink, then the real Xeno workload.

Do **not** re-tune draft depth, re-test ngram, or revisit the quant question at 16K.
Those are settled by the data above.

---

## 10. Reference — cold prefill projection

At the measured prompt-processing rate of **518.8 tok/s** (prompt_n = 4601),
straight-line and ignoring depth degradation:

```text
 16K prefill  ~   32 s
 64K prefill  ~  126 s   (2.1 min)
128K prefill  ~  253 s   (4.2 min)
256K prefill  ~  505 s   (8.4 min)
```

Prompt caching makes incremental turns cheap, but every cache miss, branch, restart
or compaction pays this in full, and the real 256K figure will be worse because
prompt processing degrades with depth. These are a sanity bound only — protocol §24
requires this be measured with `llama-bench -d`, not extrapolated.

This supports the revised framing already adopted in research doc 07: **256K is a
configured maximum, not a normal working set.**
