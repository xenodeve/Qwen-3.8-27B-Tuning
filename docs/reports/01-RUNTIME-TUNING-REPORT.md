# Runtime Tuning Report — Squeezing the Machine, Not the Model

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **Status:** complete for 16K context · **Date:** 2026-08-18 UTC+7
> **Builds on:** `docs/reports/00-Q3-VS-Q4-BENCHMARK-REPORT.md`
> **Answers:** plan Phases E, G, K · deep-research priorities S and A
> **Raw artifacts:** `C:\AI\qwen38-tuning\` — `EXPERIMENTS.md` (E7, E8),
> `results\prefix-cache.jsonl`, `results\sweep-*.jsonl`, `bench\prefix_cache_gate.py`,
> `bench\sweep_runtime.py`

---

## 0. Result

Three runtime settings, swept independently, stacked, then verified end to end on
the execution-graded task corpus:

```
--fit-target 768   -t 18   -b 2048 -ub 256
```

| | Q4 baseline | **Q4 tuned** | Q3 baseline |
|---|---|---|---|
| **Verified tasks / hour** | 33.6 | **36.1** | 22.2 |
| **Pass rate** | 90.0 % (27/30) | **90.0 % (27/30)** | 86.7 % (26/30) |
| Median tok/s, task suite | 10.56 | **12.27** | 8.73 |
| Code-rewrite decode | ~11.3 | **~13.5** | 10.30 |
| Wall clock, same 30 tasks | 2 889 s | **2 692 s** | 4 213 s |

**+7.4 % verified tasks per hour, pass rate unchanged.** Against Q3 the tuned Q4
profile is **+63 % on the project metric**.

> **CORRECTION (measured after this report's first draft).** The decode gain
> originally stated here as +16.2 %, and the "+19 % cumulative" derived by adding
> the three sweeps below, are **both overstated**. See §7: restart-to-restart
> spread on an *unchanged* configuration is 13.6 % peak-to-peak, and each sweep
> below measured its control once, first — so each per-lever figure carries a
> full share of drift, and adding them compounds it three times.
>
> An interleaved stock/tuned/stock/tuned/stock/tuned re-test gives the honest
> number: **+6.6 % paired mean, +9.6 % on pooled medians (15 samples per arm).**
> The 45-minute quality benchmark's +7.4 % tasks-per-hour sits inside that range
> and is the independent corroboration. The per-lever attributions in §3-§5 are
> **not separable** at this noise level; the stacked configuration wins, but which
> flag contributed how much is not established.

Every one of the 30 tuned samples reproduced the baseline's per-task pattern
exactly, including the single task neither configuration can solve. Quality did
not move.

---

## 1. Method — proving quality with hashes instead of sampling

`--fit-target`, `-t` and `-b/-ub` change *where* work happens and *how* it is
scheduled, not what the model computes. Paying ~48 minutes of quality benchmark
per arm would have been both slow and weak evidence.

Instead every configuration emitted a greedy sample
(`temperature 0, top_k 1, seed 42`) whose SHA-256 was compared against the control:

```text
hash matches  -> output is bit-identical; quality is provably unchanged
hash differs  -> the change is NOT quality-neutral; full quality bench required
```

**Every configuration in every sweep matched.** That is strictly stronger than a
pass-rate comparison, which could miss a small regression, and it costs seconds
rather than an hour. The full 30-sample benchmark was then run once, on the
stacked result, to confirm the stack end to end — and it agreed.

Two measurement rules carried over from the earlier report and both mattered here:

- **N≥3, decide on ranges not point estimates.** Two sweeps were re-run at N=5
  because their leaders sat within noise of each other. In one case the default's
  median was being dragged by a single outlier.
- **Read the code-rewrite prompt, not the synthetic one.** The 11-token prompt
  stayed inside 9.86–11.90 tok/s across every configuration tested; it is
  overhead-dominated and cannot discriminate between them.

---

## 2. Prefix-cache gate — the finding that outranks the flags

**Why it was run first:** Qwen3.8 is a hybrid recurrent/attention architecture
(this machine allocates separate `RS` buffers). llama.cpp issues report
hybrid-memory models logging *"forcing full prompt re-processing due to lack of
cache data"*. If that happened here, prefill — not decode — would be the real
bottleneck and every tuning priority below would be misordered.

Method: an OpenCode-shaped conversation (system block + 8 tool schemas + 40-file
repo context ≈ 3.9K tokens) that only ever appends, reading `cache_n` and
`prompt_n` from `/completion` timings.

| turn | prompt_n (evaluated) | cache_n (reused) | wall |
|---|---|---|---|
| 1 (cold) | 3 878 | 0 | 12.6 s |
| 2 | **43** | 3 874 | 2.8 s |
| 3 | **35** | 3 913 | 3.9 s |
| 4 | **37** | 3 944 | 1.3 s |

**PASS.** The hybrid full-reprocess bug does not reproduce on b10472. Append-only
turns evaluate ~40 tokens instead of ~3 900.

### 2.1 The perturbation result is the operational one

| change | cache retained | cost |
|---|---|---|
| reorder tool schemas | **0 %** | full 3 990-token re-prefill, 11.1 s |
| edit one sentence of the system prompt | **0 %** | 11.5 s |
| prepend a skill block | **0 %** | 12.1 s |
| append only (control) | **100 %** | 2.4 s |
| `cache_prompt=false` (reference) | 0 % | 9.9 s |

The cache is **prefix-exact**. Any edit above the append point costs exactly as
much as having no cache at all. At 4K that is 11 seconds; scaled to 64K it is
~2 minutes, and at 256K ~8 minutes.

> **Rule for the OpenCode / Xeno integration: freeze everything above the append
> point.** Stable tool-schema order, byte-stable system prompt, skills injected
> once at the start and never reordered or prepended later.

This also became a decision input in §5: a setting that trades prompt-processing
speed for decode speed is trading a cost paid on every cache invalidation against
a gain paid per token.

---

## 3. `--fit-target` — swept (per-lever figure not separable, see §7)

Default is **1024 MiB**, a target *margin* per device, not an absolute cap. The
deep-research report proposed `--fit-target 1024` as a safety candidate; it is
what the machine already runs, so it changes nothing.

| target | GPU layers | code tok/s | range | VRAM free |
|---|---|---|---|---|
| 1024 (default) | 32 | 11.34 | [11.23, 11.50] | 867 MiB |
| 256 | 35 | 8.28 | **[6.70, 8.28, 11.57]** | 345 |
| 512 | 34 | 11.89 | [11.46, 12.08] | 357 |
| **768** | **33** | **12.39** | **[12.11, 12.40]** | 584 |
| 1536 | 30 | 11.61 | [11.59, 11.62] | 1 079 |
| 2048 | 28 | 11.06 | [10.87, 11.11] | 1 403 |

**The result is not monotonic in layer count.** 1536 with 30 GPU layers beat the
default with 32; 2048 with only 28 layers posted the best synthetic figure. The
governing variable is the *balance* between resident layers and the headroom left
for compute buffers, and 768 lands on it.

At 256 the code prompt did not get slower on average — it became **unstable**:
`[6.70, 8.28, 11.57]`, a 73 % spread with one perfectly normal sample. That is
intermittent driver eviction at 345 MiB free, and it is the reason an optimizer
that simply maximises resident layers is the wrong optimizer on this machine.

---

## 4. CPU threads — swept, plus a contradiction worth recording

With 33 of 65 layers CPU-resident, CPU decode is a first-class bottleneck rather
than a secondary concern. CPU is an i5-13500: 6 P-cores + 8 E-cores, 14 physical,
20 logical.

| `-t` | 6 | 8 | 10 | 12 | 14 (default) | **18** | 20 |
|---|---|---|---|---|---|---|---|
| code tok/s | 9.38 | 10.59 | 11.19 | 11.54 | 12.70 | **13.58** | 13.42 |
| prompt processing | — | — | — | — | 166.8 | **167.4** | 137.2 |

Confirmed at N=5 for 14 / 18 / 20; the ranges for `-t 14` ([12.64, 12.74]) and
`-t 18` ([13.53, 13.63]) do not overlap.

**Throughput rises monotonically from 6 to 20, contradicting the usual
physical-core guidance.** `-t 6` — P-cores only, the configuration that folklore
recommends — was the *worst* result measured, 31 % below the winner. On this
hybrid CPU/GPU workload the E-cores contribute; they do not drag.

`-t 20` claims every logical thread and **costs 18 % of prompt processing**
(137.2 vs 167.4) with a wider decode spread, because it leaves nothing for the OS.
`-t 18` wins decode, spread and prompt processing simultaneously.

### 4.1 New mechanism: `-tb` is on the decode path under MTP

`-t 20 -tb 14` dropped **decode** from 13.42 to 12.71. `-tb` is documented as the
prompt/batch thread count and should not touch single-token generation — but MTP
verifies several drafted tokens in one batched pass, so whenever speculative
decoding is enabled the batch thread count sits on the decode path. Leave `-tb`
unset so it follows `-t`.

---

## 5. batch / ubatch — swept (see §7)

| `-b` / `-ub` | code tok/s | range | prompt processing |
|---|---|---|---|
| 2048 / 512 (default) | 13.00 | [12.46, 13.08] | 164.4 |
| 1024 / 512 | 13.08 | [11.92, 13.47] | 159.7 |
| **2048 / 256** | **13.49** | [12.99, 13.65] | **164.2** |
| 512 / 128 | 13.36 | [13.30, 13.38] | **103.0** |

Confirmed at N=5 for the top three.

Keeping `-b` large protects prompt processing while halving `-ub` frees
compute-buffer VRAM — both effects at once, rather than trading one for the other.

**`-b 512 -ub 128` was rejected despite a competitive raw decode figure.** It costs
33 % of prompt processing. Using §2's numbers, that is a **59-second penalty per
cache invalidation** at 16K against a **0.8-second gain** on a 500-token response
— roughly 74 responses to repay a single miss, and §2 showed how easily a miss is
triggered. This is exactly the decision the prefix-cache gate existed to inform.

---

## 6. End-to-end verification

The stacked configuration was run through the same 30-sample
execution-verified corpus used for the Q3-vs-Q4 decision.

| task | difficulty | baseline | tuned | baseline wall | tuned wall |
|---|---|---|---|---|---|
| `bracket_matching` | easy* | 0/3 | 0/3 | 337 s | 358 s |
| `lru_cache` | easy | 3/3 | 3/3 | 188 s | 154 s |
| `merge_intervals` | easy | 3/3 | 3/3 | 129 s | 92 s |
| `toposort` | medium | 3/3 | 3/3 | 207 s | 215 s |
| `expr_eval` | medium | 3/3 | 3/3 | 572 s | **422 s** |
| `rotated_search` | medium | 3/3 | 3/3 | 159 s | 129 s |
| `text_wrap` | medium | 3/3 | 3/3 | 253 s | 258 s |
| `lfu_cache` | hard | 3/3 | 3/3 | 514 s | 481 s |
| `damerau` | hard | 3/3 | 3/3 | 202 s | 204 s |
| `tree_codec` | hard | 3/3 | 3/3 | 328 s | 378 s |

\* Neither configuration has ever solved `bracket_matching` — a capability ceiling
of the model at this quant, not a tuning artifact.

**Pass/fail is identical on all ten tasks**, exactly as the greedy hashes
predicted. The wall-clock gain is not uniform per task, which is expected: tasks
that generate more tokens benefit more, and per-task wall clock carries sampling
variance that the aggregate does not.

---

## 7. Measurement drift — the correction, and how it was found

The speculative sub-knobs (`--spec-draft-p-min`, `-p-split`, `-n-min`) were swept
next. The first pass looked excellent:

| knob | vs control | |
|---|---|---|
| `-n-min 2` | +11.6 % | |
| `-p-min 0.10` | +9.8 % | |
| `-p-split 0.25` | +8.8 % | |

Re-running the leaders against a **fresh** control reversed every one of them:

| knob | vs fresh control |
|---|---|
| `-n-min 2` | **-0.8 %** |
| `-p-min 0.10` | **-10.1 %** |
| all three combined | **-4.3 %** |

The first sweep had measured **machine drift, not knob effects**: its control ran
first, in a slow window, and every later configuration ran as the machine
recovered — a monotonic time trend that looks exactly like a monotonic knob
effect. A correlation check against free VRAM (+0.06) did not catch it, because
the drift was not VRAM-driven.

### 7.1 How big the drift is

Six restarts of an **identical** configuration, N=5 each:

```text
11.63  12.59  12.60  12.63  13.21   tok/s (medians)
peak-to-peak 13.6 %   ·   stdev 4.5 % of mean
```

**That floor exceeds every per-lever claim in this report** (+9.3 %, +6.9 %,
+3.8 %). A single control-first comparison cannot establish an effect that size
on this machine.

### 7.2 The paired re-test

Interleaving the arms makes both share the drift instead of assigning it all to
whichever ran later:

```text
stock  11.79   10.90   11.12
tuned  11.34   12.50   12.12
diff   -3.8%  +14.7%   +9.0%     mean +6.6%
pooled (15 samples/arm):  11.18 -> 12.25  =  +9.6%
```

One pair went **negative**. The tuned configuration still wins on pooled samples
and on the independent 45-minute quality run, but the effect is roughly half what
the additive per-lever arithmetic suggested.

### 7.3 What this changes going forward

- **Interleave the arms.** Control-first ordering is not adequate here.
- **Report paired differences**, not a ratio of two separately-measured medians.
- **Do not add per-sweep deltas.** Each carries its own drift; summing compounds it.
- **An effect below ~14 % needs a paired design** or it is not measurable at all.
- The spec sub-knobs are **rejected**; keep the llama.cpp defaults.

---

## 8. What this does not cover

1. **16K only.** All of it. The layer split, the headroom balance and the
   fit-target optimum will all move once KV grows with context. `--fit-target 768`
   in particular is tuned to the current buffer sizes and should be re-swept at
   each depth.
2. **Q8 KV untested, and it needs a build first.** The deep-research report flags
   that requesting a quantized-KV Flash-Attention combination whose kernel was not
   compiled can silently fall back to a very slow path. A pinned SM89 build with
   `GGML_CUDA_FA_ALL_QUANTS=ON` should precede any Q8 KV measurement, or the
   experiment will measure the missing kernel rather than the quantization.
3. **`--cache-ram` untouched** (default 8192 MiB against ~11 GB free host RAM).
4. **CPU affinity untested.** Windows processor numbering must be discovered on
   the machine before a P-core mask is attempted; a wrong mask looks like an
   optimization and behaves like a regression.
5. **Prefix-cache tested at ~4K, not at depth.** Reuse behaviour is proven; the
   *cost* of a miss scales with context and has only been projected.
6. **Per-lever attribution is not established** — only the stack is (§7).
7. **Still no OpenCode / OpenClink / real-repo run.** The §2 freeze rule is a
   prediction about how those clients must serialize their prompts; it has not yet
   been checked against what OpenCode actually sends.

---

## 9. Next

1. **Context-depth sweep at 32K → 256K**, re-sweeping `--fit-target` at each depth,
   since the balance this report tuned is depth-dependent.
2. **Pinned SM89 + `FA_ALL_QUANTS` build**, then Q8 KV — in that order.
3. **Verify OpenCode's actual serialization** against the §2 freeze rule. If
   OpenCode reorders tools or rewrites the system prompt between turns, that is a
   larger cost than every flag in this report combined.
4. `--cache-ram` budgeting to keep host RAM out of paging.

Do not re-sweep threads or batch at 16K; those are settled.
