# Qwen3.8-27B Local Coding Worker on an RTX 4070 SUPER 12GB — Consolidated Findings

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **Date:** 2026-08-19 (UTC+7)
> **Audience:** an external reviewer with no access to this machine or to the
> other documents in `C:\AI\docs\reports\`. Everything needed to check the
> reasoning is restated here; nothing is left as a cross-reference to a file you
> cannot open.
> **Status of every number below:** measured on the machine described in §1,
> unless explicitly labelled *projected*, *vendor*, or *third-party*.

---

## 0. What changed, in one paragraph

This project spent its first phase tuning llama.cpp flags on `Qwen3.8-27B
UD-Q4_K_XL` and reached **12.3–13.7 tok/s** with roughly half the model resident
on the CPU. A deep-research document then argued that the remaining gains were
not in flags but in the **memory regime** — that an artifact small enough to
become fully GPU-resident would beat Q4 even at lower fidelity. That was tested.
It is correct, and by a wider margin than the research projected: **`UD-IQ2_XXS`
is fully resident (65/65 layers), decodes at 42.4 tok/s (+220 %), and accepted
the same 27 of 30 verification-gated coding tasks as Q4.** The new default is
IQ2_XXS with speculative decoding **off**. The largest unresolved risk is that
the task corpus used to establish "no quality loss" may not probe the failure
mode this class of quantization is known for — stated fully in §9.

---

## 1. System under test

| component | value |
|---|---|
| GPU | NVIDIA RTX 4070 SUPER, **12,282 MiB** total VRAM |
| CPU | Intel i5-13500 (20 logical threads; `-t 18` leaves two for the OS) |
| RAM | **47.69 GB** visible |
| OS | Windows 11 Pro 26200 |
| Runtime | llama.cpp **build 10472**, commit `60eeeb608`, CUDA, Clang 20.1.8 |
| Model family | Qwen3.8-27B, architecture `qwen3_5` |

**Model shape that matters for reading the tables.** 27B dense, **64 transformer
layers plus `blk.64`**, the built-in multi-token-prediction (MTP / nextn) head —
so a full layer assignment is **65** lines, and "65 + 0" below means everything
including the MTP head is on the GPU. Hybrid attention: roughly three quarters
Gated-DeltaNet linear layers, one quarter full attention, which is why the KV
cache grows more slowly with context than a fully-quadratic model would. Native
262,144-token context. Vision tower disabled throughout via `--no-mmproj-auto`.

**Free VRAM at boot is not constant** — it ranged **9,326 – 10,732 MiB** across
recorded launches. `--fit on` derives the GPU/CPU layer split from that number,
so the split is a property of the boot, not of the file. Every measurement below
parses the actual split out of the server log rather than assuming one.

### Artifacts tested (exact byte counts from the Hugging Face API)

| artifact | bytes | GiB | tested |
|---|---:|---:|---|
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17,923,394,624 | **16.69** | yes — control |
| `Qwen3.8-27B-UD-Q3_K_XL.gguf` | 13,441,059,904 | 12.52 | earlier phase — rejected |
| `Qwen3.8-27B-UD-Q2_K_XL.gguf` | 10,676,423,744 | **9.94** | yes |
| `Qwen3.8-27B-UD-IQ2_M.gguf` | 10,319,907,904 | 9.61 | no |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | 9,010,048,064 | **8.39** | yes — **new default** |

All from `unsloth/Qwen3.8-27B-GGUF`.

---

## 2. The metric

**Verified successful coding tasks per hour**, not tok/s. This comes from the
project's own planning documents and it has overturned decisions before: a
configuration with the best raw decode (`-b 512 -ub 128`) was rejected because it
cost 33 % of prompt processing for 2 % of decode, and prompt processing is what
every cache miss pays for.

Verification is **execution-based**, never an LLM judge. The model's code is
written to a file with a test appended and run in a subprocess with a 20-second
timeout; a non-zero exit is a failure and an infinite loop is a failure rather
than a hang.

Two derived figures are used below and defined here:

```
verified tasks/hour  = 3600 × accepted_tasks / worker_wall_seconds

merged tasks/hour    = 3600 × total_tasks
                     ────────────────────────────────────────────────────────
                       worker_wall + 90s × escalations + 60s × total_tasks
```

The 90 s (cost of escalating an unsolved task to Q4) and 60 s (fixed
review/CI/orchestration overhead per change) are the deep-research document's own
assumed constants, kept so its economic model can be evaluated on our data.

---

## 3. Measurement methodology — why these numbers should be believed

This section exists because several results in this project were published and
then withdrawn. The design below is what the withdrawals taught.

### 3.1 The noise floor

**Restart-to-restart spread on an *unchanged* configuration is 13.6 %
peak-to-peak** (stdev 4.5 %). Per-restart medians on an identical config:

```
11.63   12.59   12.60   12.63   13.21   tok/s
```

**Any claimed effect below ~14 % cannot be established by a single control-first
comparison on this machine.** That floor is larger than every individual runtime
flag this project ever measured.

It is not hypothetical. A speculative sub-knob sweep once produced `-n-min 2`
**+11.6 %**, `-p-min 0.10` **+9.8 %**, `-p-split 0.25` **+8.8 %**. Re-run against
a *fresh* control, they became **−0.8 %**, **−10.1 %**, **−4.3 %**. The first
sweep had measured machine drift: its control ran once, at one end of a monotonic
time trend, and every later configuration ran as the machine recovered. A
correlation check against free VRAM (+0.06) did **not** catch it, because the
drift was not VRAM-driven.

Separately, three control-first sweeps were once summed into "+19 % cumulative".
The paired re-measurement gives **+6.6 % mean / +9.6 % pooled**.

### 3.2 The design used for every comparison in this report

Two different quantizations **cannot share a boot** — the weights differ — so
flag-sweep interleaving is unavailable. The replacement:

```
round 1:   A   B   C
round 2:   C   B   A      ← order reversed, so no arm always runs in the same
round 3:   A   B   C        position within a round
```

Arms are paired **by round**, and the summary refuses to call an effect real
unless it (a) exceeds the 13.6 % floor **and** (b) keeps its sign in every round.
A mean of +40 % / −10 % is reported as unresolved, not as +15 %.

Per boot the harness records: free VRAM before launch, VRAM used and free after
load, the GPU/CPU layer split parsed from the log, three decode samples, prompt
processing rate, MTP draft/accept counts, and a greedy-decode hash.

### 3.3 The probe prompt matters more than the sample count

- An **11-token prompt cannot measure prompt processing**: it returned 13.7 tok/s
  where the real figure at 4,601 tokens was 518.8. Fixed per-request overhead
  dominates.
- An 11-token prompt **cannot measure speculation** either: it stayed inside
  9.86–11.90 tok/s across *every* configuration tested.
- MTP acceptance was **78.1 %** on a short instruction and **98.0 %** on a
  code-rewrite prompt.

Every decode figure below therefore comes from a code-rewrite prompt (~300
tokens: "rewrite this class, renaming one attribute"), never a toy prompt.

### 3.4 Instrument bugs found in this project

Every one produced a **plausible wrong number rather than a crash**. That is the
failure mode the harness is designed against; all of these are now covered by
regression tests (60 tests, all passing).

| bug | symptom | consequence |
|---|---|---|
| `[int](3/2)` is **2** in PowerShell (banker's rounding) | a field named `tg_median` held the **maximum** | every sweep table mislabelled |
| PowerShell 5.1 writes a **BOM** on first `Add-Content -Encoding utf8` | `json.loads` raised on line 1, swallowed by `except: pass` | **baseline row silently deleted from every table** |
| device token is `CUDA0,` with a trailing comma | `== "CPU"` matched nothing | split reported `32+0` instead of `32+33` |
| a `0.0 tok/s` sample from a generation that produced no tokens | folded into the sample list | median survives one, dies on two |
| deep corpus emitted `Handler0017` **twice** | two contradictory answers in context | the task measured nothing |
| corpus size assertion checked only a **lower** bound | a 112K-token corpus passed its own test | every request HTTP 400 → **0/18 in four seconds**, which reads as "the model cannot do deep context" |
| **new this session:** economics summary computed from a run where every request returned **HTTP 503** | escalation is a constant, so 30 tasks that never ran still produced **"24.0 merged tasks/hour"** | see §8 |

---

## 4. Result 1 — the residency cliff

Paired-boot arena, two independent runs of three rounds each, counterbalanced
order, three decode samples per boot.

| arm | layer split | free VRAM after load | decode tok/s (per round) | prompt processing |
|---|---|---:|---|---:|
| `Q4_K_XL` + MTP n=2 (control) | **33 + 32** | 743 – 931 MiB | 12.58 · 13.60 · 13.74 | 147 – 168 |
| `Q2_K_XL` + MTP n=2 | 55 + 10 | 793 MiB | 19.87 · 19.92 · 19.95 | 310 – 330 |
| `Q2_K_XL`, no speculation | 61 + 4 | 451 – 569 MiB | 21.26 · 21.84 · 21.98 | 394 – 510 |
| **`IQ2_XXS`, no speculation** | **65 + 0** | **1,178 – 1,190 MiB** | **42.44 · 42.47 · 42.47** | **809 – 818** |

```
arena 1 (3 rounds):
  Q2_K_XL nospec   per-round [+60.94, +61.78, +63.64]     mean  +62.12%   RESOLVED
  Q2_K_XL mtp2     per-round [+50.42, +47.56, +52.41]     mean  +50.13%   RESOLVED

arena 2 (3 rounds):
  Q2_K_XL nospec   per-round [+72.10, +60.59, +59.97]     mean  +64.22%   RESOLVED
  IQ2_XXS nospec   per-round [+237.36, +212.28, +209.10]  mean +219.58%   RESOLVED
```

### The shape is the finding

```
33 GPU / 32 CPU     12.6 – 13.7 tok/s
61 GPU /  4 CPU     21.3 – 22.0 tok/s      moving 28 layers to GPU:  +64 %
65 GPU /  0 CPU     42.4 – 42.5 tok/s      moving the last 4:        +95 %
```

**Almost all of the prize sits at the very end of the curve.** Four CPU-resident
layers cost about half the throughput. Prompt processing follows the same shape
and more sharply — **156 → 394-510 → 818 tok/s, a 5.2× span**.

This reframes an earlier finding of this project, that "more GPU layers is not
monotonically better", which was measured while sweeping 32–35 layers. That sweep
was nowhere near the cliff. **"Nearly resident" and "resident" are different
regimes**, and only the second one pays.

### Practical corollary

`Q2_K_XL` at 9.94 GiB is the worst of the three tested. It fails to reach full
residency, so it loses half the available throughput, **and** its free VRAM
(451–569 MiB) sits at or below the 512 MiB reserve this project adopted after a
`--fit-target 256` run produced intermittent driver eviction at 345 MiB free —
recognisable as a 73 % spread with one perfectly normal sample (`[6.70, 8.28,
11.57]`), i.e. *instability*, not a lower mean. Paying 1.55 GiB more than
IQ2_XXS for "better quality" bought no measurable quality and cost half the
speed.

---

## 5. Result 2 — speculative decoding inverts once the target is resident

Qwen3.8 ships a built-in MTP head, and on Q4 it is worth keeping. On a resident
artifact it is a **net loss of ~7 %**, and the layer split explains why:

```
Q2_K_XL, no speculation     61 + 4      21.3 – 22.0 tok/s
Q2_K_XL, MTP n=2            55 + 10     19.9 tok/s
                            ↑ the draft head's VRAM pushes SIX target
                              layers back onto the CPU
```

Speculation trades VRAM for arithmetic. When the target forward pass is expensive
— Q4, with half the model on the CPU — that trade pays. When the target is nearly
resident and already cheap, the residency it costs is worth more than the tokens
it saves. This project had previously recorded the same mechanism running the
other way ("MTP compensates for CPU offload"); this is that finding inverted, and
the deep-research document predicted it in words: *"speculative benefit can
collapse when the target forward pass becomes cheap."*

**Copying the Q4 flag set onto a low-bit artifact costs 7 % and looks like a
weakness of the artifact.**

Related, and separately established earlier: MTP does **not** change output —
greedy decoding is byte-identical across every speculative configuration tested;
the claimed ~2.5 GB MTP overhead is not real (`blk.64.*` totals 285.8 MB and VRAM
went *down*); and the draft-depth sweet spot is **n = 2–3**, not 4–5, with n ≥ 5
regressing.

---

## 6. Result 3 — the task corpus and the retry economics

Ten execution-verified Python tasks (LRU cache, interval merge, bracket matching,
topological sort, expression evaluator, rotated-array search, LFU cache,
Damerau-Levenshtein, tree codec, text wrap), run three times each = **30 tasks
per arm**, identical prompts, identical sampling, identical `max_tokens 3072`.

Protocol: attempt once; on failure retry **once** with the actual traceback
pasted back — the evidence a real agent loop would return — then escalate.

| | Q4_K_XL | Q2_K_XL | **IQ2_XXS** |
|---|---:|---:|---:|
| first-attempt success `p1` | **83.3 %** | **83.3 %** | 73.3 % |
| retry success `p2` | 40.0 % | 20.0 % | **62.5 %** |
| **locally accepted** | **90.0 %** (27/30) | 86.7 % (26/30) | **90.0 %** (27/30) |
| attempts per accepted task | **1.30** | 1.35 | 1.41 |
| escalations per 100 tasks | **10.0** | 13.3 | **10.0** |
| attempts truncated by the 3072 budget | 3 | 2 | **7** |
| worker wall for the 30 tasks | 4,008.7 s | 1,972.5 s | **1,599.0 s** |
| **merged tasks / hour** | 17.8 | 26.1 (+47 %) | **29.4 (+65 %)** |
| **verified tasks / hour** | 24.2 | 47.5 (+96 %) | **60.8 (+151 %)** |

**Final acceptance is a tie between Q4 and IQ2_XXS at 27/30**, reached in 40 % of
the wall clock. IQ2_XXS gets there differently: worse first attempt (22/30 vs
25/30), much better retry (5 of 8 vs 2 of 5).

Failures concentrate on the **same tasks for all three arms** — `bracket_matching`
above all, which no configuration or artifact in this project has ever solved.
That is a capability ceiling of the model family at this size, not a tuning or
quantization artifact.

### The research's central economic assumption does not hold

The deep-research document builds its entire "22.6 → 40.2 merged tasks/h" table
on an assumed relationship, which it labels honestly as an assumption:

```
p2 = min(p1 + 0.10, 0.95)
```

At our measured `p1 = 0.833` that predicts `p2 = 0.933`. **Measured `p2` is
0.40 on Q4, 0.625 on IQ2_XXS, and 0.20 on Q2_K_XL.** An evidence-assisted retry
succeeds roughly one time in three at best, not nine times in ten.

And failures are not cheap. Tasks needing a retry ran **350–420 s** against
**16–150 s** for tasks that passed first time; in one 10-task run the three
retried tasks consumed **71 % of the entire corpus wall clock**. A worker that
fails more does not merely fail more — it spends far longer doing it.

**This cuts against the argument that a weaker-but-faster worker wins on
economics.** It did not matter here, because the faster artifact was not the
weaker one. But the reasoning that would have justified adopting a genuinely
weaker worker was resting on a constant that is off by roughly 3× on this
machine, in the direction that makes weak workers *more* expensive.

---

## 7. Result 4 — agent-loop gates

Three gates the deep-research document names as preconditions ahead of any tok/s
number. Control values were established on Q4 first, so candidates are judged
against the production model rather than against a threshold from a document.

### 7.1 Tool calling

Probe: one `apply_patch` call whose arguments contain a **nested array of
objects**; then the tool result is fed back and the model must continue rather
than re-issue the call. `max_tokens 4096`, temperature 0.7, n = 15 per arm.

| | Q4_K_XL | Q2_K_XL | **IQ2_XXS** |
|---|---:|---:|---:|
| schema-correct call | 80.0 % (12/15) | 86.7 % (13/15) | **93.3 % (14/15)** |
| non-calls that were **budget truncations** | 3 of 3 | 2 of 2 | 1 of 1 |
| required-field omissions | **0** | **0** | **0** |
| `tool_call_id` round-trip | 60.0 % | 60.0 % | 66.7 % |
| **median reasoning per call** | **59 chars** | **2,811 chars** | **1,023 chars** |
| wall for 15 trials | 1,367.1 s | 1,121.8 s | **431.9 s** |

**Every single non-call, on every arm, was `finish_reason: "length"`** — the model
still reasoning when the budget ran out. Not one was a model declining to use the
tool. When these models emit a call, the schema is correct, including the nested
array; zero required-field omissions across 45 trials.

The round-trip rate is essentially identical across arms, which identifies it as
a property of the probe rather than of any quantization.

**The number that separates the arms is reasoning length: 59 characters on Q4
against 2,811 on Q2_K_XL — a factor of 48 on the same task.** That is the real
cost of these artifacts, and it is a budgeting problem, not a lost capability.

**This has a direct integration consequence.** An earlier run of this same probe
at `max_tokens 1024` scored Q2_K_XL at **40 % tool compliance** and would have
been reported as a collapse. It was the probe's budget. **A client tuned to Q4's
token appetite will truncate a low-bit model mid-reasoning and read it as a
refusal to call tools.**

### 7.2 Long-loop stability

One append-only agent conversation, 100 sequential turns, with the system block
edited every tenth turn — exactly what injecting a skill does to a real agent.

| | Q4_K_XL | Q2_K_XL | IQ2_XXS |
|---|---:|---:|---:|
| turns survived | 100/100 | 100/100 | 100/100 |
| hangs / stuck slots | **0** | **0** | **0** |
| forced invalidations recovered on the next turn | **9/9** | **9/9** | **9/9** |
| steady-state prefix reuse | 99.1 % | 99.0 % | **99.2 %** |
| p50 turn | 3.38 s | — | **1.34 s** |
| empty replies | 19/100 | **55/100** | **1/100** |

Steady-state reuse means roughly **46 tokens evaluated per turn** against a cache
that grew to 9,066 tokens. No latency drift with depth (first ten turns median
4.82 s, last ten 3.77 s on Q4). The prefix cache recovers on the very turn after
every invalidation, on every arm — this is the hybrid-memory failure mode the
research warns about for Qwen MoE/hybrid models, and it does not occur here.

**The empty-reply counts are unexplained and are *not* a quantization effect.**
19 / 55 / 1 is not monotonic in bit-width — the most aggressively quantized
artifact shows the behaviour least. An earlier draft proposed that low-bit models
are more sensitive to the probe's hand-built (non-chat-template) framing;
IQ2_XXS disproves it. Recorded as unknown rather than dressed in a story that
fits two points and breaks on the third.

### 7.3 The cost of a broken prefix — the highest-leverage number in this project

The ten forced invalidations above, at increasing cache depth, on Q4:

| tokens re-evaluated | wall |
|---:|---:|
| 2,925 | 10.64 s |
| 3,478 | 14.95 s |
| 4,008 | 15.20 s |
| 4,650 | 21.75 s |
| 5,450 | 23.64 s |
| 6,166 | 25.93 s |
| 6,850 | 26.35 s |
| 7,581 | 29.50 s |
| 8,290 | 32.30 s |
| 9,066 | 35.41 s |

Least-squares slope over all ten, **r² = 0.968**: prefill **265.5 tok/s**, fixed
offset 1.55 s. Projected cost of **one** prefix break:

| context | cost |
|---|---:|
| 16K | **63.3 s** |
| 32K | 125.0 s |
| 64K | **248.4 s** |

Set against the entire runtime-tuning campaign, worth **+6.6 %** of decode (about
0.8 tok/s, ~4 s on a 500-token reply):

> **One skill injection at 64K undoes more than five minutes' worth of every flag
> this project tuned.**

The prefix cache here is **exact**: editing one sentence *above* the append point
discards the whole cache, not the changed part. Measured earlier at 4K, a turn
goes from 2.4 s to 11.5 s when tool schemas are reordered or the system prompt is
edited.

*Method note:* this cost was first computed by dividing one perturbed turn's wall
time by its token count. That is wrong — wall time is prefill **plus** the decode
of the turn's 48 tokens — and the slope across ten points is the correct
estimator, because the decode component is constant and lands in the intercept.
In this instance the two agree closely (274.9 vs 265.5 tok/s, 3.5 %), so the
correction changed the number by almost nothing and the method by everything: a
single-point ratio has no residual to inspect and would keep being wrong at a
depth where decode is not small.

---

## 8. Result 5 — the depth ladder

`IQ2_XXS`, Q8_0 KV, speculation off. Cold prefill is n = 1 by design (a 256K
prefill costs eleven minutes); decode is n = 3 over the warm cache, which is also
how an agent behaves — one cold turn, then appends.

| ctx | layer split | **decode tok/s** | prompt proc | cold prefill | KV | free VRAM |
|---|---|---:|---:|---:|---:|---:|
| 16K | **65 + 0** | **42.4** | 818 | — | ~512 MiB *(F16)* | 1,178 MiB |
| 64K | 61 + 4 | **15.81** | 727 | 64.0 s | 2,040 MiB | 447 MiB |
| 128K | 47 + 18 | **5.15** | 474 | 196.2 s | 3,264 MiB | 503 MiB |
| 256K | 31 + 34 | **1.71** | 284 | 658.1 s | 4,352 MiB | 412 MiB |

Against the Q4 ladder:

| ctx | Q4 | IQ2_XXS | ratio |
|---|---:|---:|---:|
| 16K | 12.6 – 13.7 | 42.4 | **3.2×** |
| 64K | 5.10 | 15.81 | **3.1×** |
| 128K | 2.5 | 5.15 | **2.1×** |
| 128K cold prefill | ~720 s | **196.2 s** | **3.7×** |
| 256K | **stopped** — host paging | **1.71**, no paging | — |

### What the ladder is actually measuring

The decode column tracks the **layer split**, not the context length:

```
65 + 0     42.4 tok/s      KV ~0.5 GB
61 + 4     15.8 tok/s      KV  2.0 GB
47 + 18     5.2 tok/s      KV  3.3 GB
31 + 34     1.7 tok/s      KV  4.4 GB
```

This is the residency cliff of §4 seen from the other side. There, shrinking the
artifact bought GPU layers; here, growing the context spends them, because KV is
allocated from the same pool the weights live in. At **256K the split is 31 + 34
— worse than Q4's 33 + 32 at 16K.** Every advantage this artifact has comes from
residency, and at 256K there is none left.

The correct reading is therefore **not** "IQ2_XXS degrades with depth" (Q4 does
too, and faster in relative terms at 128K) but:

> **Depth spends the same VRAM the quantization was chosen to free.**

### 256K: reachable now, and what it costs

The Q4 attempt at 256K was **stopped, not measured**: host RAM free fell to
0.63 GB of 47.69 with a 10.11 GB pagefile and 296 pages/sec. Anything measured
under that pressure would have described Windows paging.

IQ2_XXS holds 256K with **15.4 GB of host RAM still free** — its CPU-resident
half is a far smaller thing to hold — and the run completed normally. But:

```
cold prefill      658 s   (11 minutes)
500-token reply   293 s   ( 5 minutes at 1.71 tok/s)
```

One deep question costs about a quarter of an hour, and each subsequent turn that
preserves the prefix costs five minutes per reply. **A budget for one deep
question, not for an agent loop.** Recorded as a capability, not a
recommendation.

---

## 9. What is **not** established — read this before acting on §4–§8

**1. Retrieval quality at depth has not been verified on IQ2_XXS.** The 30/30 at
64K and 10/10 at a 114,406-token prompt this project holds were measured on
**Q4**. Section 8 is throughput and residency only.

This gap matters more for this artifact than for a milder one, because the
documented failure mode of conventional 2-bit builds in this family is
*selective*. From PrismML's model card for a competing product, describing
conventional `IQ2_XXS` builds of Qwen3.6-27B (**vendor claim, not our
measurement**):

> *"IQ2_XXS falls to 57.5 on AIME26 and 56.4 on LiveCodeBench while still scoring
> 88.93 on MMLU-Redux — which is why casual testing misses the collapse."*

**2. The corpus may not probe that failure mode.** Ten single-function
implementations run three times is structurally closer to the benchmarks that
*hide* a selective collapse than to the ones that expose it. **No degradation was
detected; that is not the same as none existing**, and it is the single largest
caveat on the recommendation.

**3. No accuracy difference was *detected*; equivalence was not shown.** 27/30
vs 27/30 vs 26/30 at n = 30 cannot separate arms differing by a few percent.

**4. The token budget confounds every quality number here.** IQ2_XXS was
truncated on 7 of 30 corpus attempts at `max_tokens 3072` against Q4's 3, so its
`p1 = 73.3 %` is a **lower bound**. A generous-budget re-run would raise `p1` and
raise wall time; neither has been measured.

**5. No stability run at depth.** The 100-turn gate ran at 16K. At 128K the free
VRAM margin is 412–503 MiB, below the 512 MiB reserve adopted after the observed
driver-eviction incident. No instability was seen, but it was not stress-tested.

**6. F16 KV was not measured on IQ2_XXS at depth.** Q8_0 was inherited from the
Q4 result (identical task quality, 15–28 % faster warm turns from 64K up). The
argument is stronger for a resident model, not weaker — but it is inherited.

**7. Everything is synthetic.** Single-file coding tasks, a synthetic retrieval
corpus, a synthetic agent loop. No real repository, no OpenCode, no end-to-end
run.

---

## 10. Numbers this session published and then withdrew

Included because a reviewer should know which claims were unstable and why, and
because the pattern is consistent: **the instrument returns a plausible number
instead of a failure.**

| withdrawn claim | what it actually was |
|---|---|
| "Q2_K_XL tool compliance **40 %** vs Q4's 80 %" | the probe's `max_tokens 1024` truncating a model whose median reasoning is 2,811 chars. With `finish_reason` recorded: **every** non-call was a truncation, none a refusal. Corrected to **86.7 %** at 4096 |
| "Q4 `p1` = **70 %**, Q2 = 83.3 %" | a **10-task** sample (7/10). At 30 tasks Q4 is **83.3 %**, identical to Q2. Publishing it would have claimed the quantized model was *more accurate than its source* |
| "**24.0** merged tasks/hour" | a run in which **every request returned HTTP 503** because a model swap passed a `/health` check while the server was still loading. Escalation is charged as a constant, so 30 tasks that never ran still produced a plausible rate. The economics function now raises on zero worker time, and the swap helper now proves a real generation before returning |
| "Bonsai runs on our stock binary" | our `Q2_0` is llama.cpp's **group-64** type; Prism's `Q2_0.gguf` is **g128** and requires their llama.cpp fork. Same name, different format — a name match was mistaken for a format match |
| prefix cost derived from one perturbed turn | wall time is prefill **plus** decode; the slope across 10 points is the estimator. 274.9 → **265.5 tok/s** — the method was wrong even though the number barely moved |

One further trap, caught before it produced a number: **`-hf <repo>:Q2_0`
resolves by substring and matched `PQ2_0.gguf`**, a different file of **exactly
the same byte count** (7,165,121,600). Nothing in the size, the log, or the cache
layout would have reported it; two gigabytes had already transferred. Any `-hf`
fetch from a repo with more than one artifact must be verified against the
repository's OID list before it is measured.

---

## 11. Where the external research was wrong

The deep-research document was the most useful input this project has received —
it correctly redirected the work away from flag tuning and correctly predicted
the speculation inversion. These are the parts that did not survive checking, so
they are not repeated downstream:

| claim | reality |
|---|---|
| Ornith is at `deepreinforce-ai/…` | it is `ornith-ai/…`; a copy-pasted `-hf` would fail |
| Bonsai's ternary path should be *"isolated from the production llama.cpp binary"* — implying a fork is optional | the fork is **required** for the headline `Q2_0` (g128) artifact. A separate `Q2_g64` pack exists for mainline packing and is untested |
| `p2 = min(p1 + 0.10, 0.95)` | measured **0.20 – 0.625**; roughly 3× optimistic, in the direction that flatters weak workers |
| "required tool/schema compliance **100 %**" as an absolute gate | Q4 itself scores **80 %** on this probe at temperature 0.7. The gate must be control-relative or it rejects the production model |
| AtomicChat `AD-IQ2_XS` at 9.9 GB as a distinct arm to acquire | Unsloth's `UD-Q2_K_XL` is 9.94 GiB, same size class, same repo, same chat template and MTP head |
| speed/pass-rate columns ("18–35 tok/s", "~72–85 % first-pass") | explicitly **projections** in the source; the measured values are in §4 and §6 |

---

## 12. Current recommended configuration

### Everyday, 16K

```powershell
C:\AI\llama.cpp-cuda\llama-server.exe `
  -hf unsloth/Qwen3.8-27B-GGUF:UD-IQ2_XXS `
  --alias qwen38-iq2xxs -c 16384 `
  -ngl auto --fit on --fit-target 768 -fa on -np 1 `
  -t 18 -b 2048 -ub 256 `
  --no-mmproj-auto `
  --host 127.0.0.1 --port 8080
```

**No `--spec-type`** — see §5. **Do not add `-ctk/-ctv q8_0` at 16K**: measured
there as 86.7 % vs 90.0 % pass *and slower*, because at ~512 MiB of KV there is
nothing to reclaim and only the cost remains.

### Deep, 64K / 128K / 256K

Same, plus `-c <ctx> -ctk q8_0 -ctv q8_0`. Q8_0 KV from 64K up was measured on Q4
at identical task quality and 15–28 % faster warm turns.

### Escalation lane, and anything retrieval-critical

```powershell
C:\AI\llama.cpp-cuda\llama-server.exe `
  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL `
  --alias qwen38-q4 -c 16384 `
  -ngl auto --fit on --fit-target 768 -fa on -np 1 `
  -t 18 -b 2048 -ub 256 `
  --no-mmproj-auto `
  --spec-type draft-mtp --spec-draft-n-max 2 `
  --host 127.0.0.1 --port 8080
```

Each flag above won its own sweep on Q4: `--fit-target 768` over the 1024 default,
`-t 18` over the 14 physical cores (throughput rose monotonically 6 → 20,
contradicting the usual physical-core guidance; `-t 20` costs 18 % of prompt
processing), and `-b 2048 -ub 256` over the 512 default microbatch with prompt
processing unchanged. The stacked effect is **+6.6 % paired / +9.6 % pooled** —
*not* the "+19 %" that summing three control-first sweeps produced.

### Two rules that outrank every flag

1. **Freeze the prompt prefix.** Anything that reorders tool schemas, rewrites
   the system prompt, or injects a block above the append point discards the
   entire cache: **63 s at 16K, 248 s at 64K** (§7.3).
2. **Budget output tokens for the artifact, not for Q4.** These models reason
   10–48× longer per call (§7.1).

---

## 13. What to do next, in order of value

1. **Deep-context retrieval quality on IQ2_XXS at 64K and 128K.** The corpus and
   the harness already exist; two runs close the single largest open risk (§9.1).
   Until then, Q4 remains the profile for needle-in-100K-tokens work.
2. **Re-run the corpus at a generous `max_tokens`** so truncation stops
   confounding `p1` (§9.4), and report tokens-per-accepted-task alongside.
3. **Stability gate at 128K** on IQ2_XXS — thin VRAM margin, untested (§9.5).
4. **`--no-kv-offload` at 128K/256K.** At those depths KV is precisely what
   pushes layers off the GPU; trading PCIe latency for weight residency now aims
   at the binding constraint. Untested, possibly large.
5. **`UD-IQ2_M` (9.61 GiB)** — the only untested artifact between the two arms.
   Given the cliff, the sole question is whether it reaches 65/65.
6. **Integration.** OpenClink → OpenCode → llama-server on a real repository.
   Check two things first: whether OpenCode's serialization preserves a
   byte-stable prefix, and what `max_tokens` it sends.

Explicitly **not** worth further work: re-sweeping `-t`, `-b`/`-ub`,
`--fit-target`, draft depth, ngram, or speculative sub-knobs at 16K. Those are
settled and the residual effects sit below the 13.6 % noise floor.

---

## 14. Reproduction

| artefact | path |
|---|---|
| harness primitives + 60 regression tests | `C:\AI\qwen38-tuning\bench\harness.py`, `bench\tests\test_harness.py` |
| paired-boot cross-model arena | `bench\model_arena.py` |
| corpus with evidence-assisted retry | `bench\run_retry_bench.py` |
| tool-call protocol gate | `bench\protocol_gate.py` |
| 100-turn stability + prefix invalidation | `bench\stability_gate.py` |
| depth ladder | `bench\depth_sweep.py` |
| verified model swap | `scripts\swap-model.sh` |
| launch profiles | `scripts\production-iq2xxs.ps1`, `production-iq2xxs-deep.ps1`, `production-q4-tuned.ps1`, `production-q4-deep.ps1` |
| raw results | `C:\AI\qwen38-tuning\results\*.jsonl` |
| per-boot server logs | `C:\AI\qwen38-tuning\logs\*.log` |

Every summarising primitive — median, JSONL loading, layer-split parsing, paired
differences, retry economics, marginal-rate fitting, tool-call scoring — is
written test-first and **raises rather than guesses** on empty, corrupt, or
unaccounted input. That is a direct response to the table in §3.4: in this
project, the characteristic failure is not a crash but a believable wrong number.
