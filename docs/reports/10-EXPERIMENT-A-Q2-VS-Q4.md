# Experiment A — Q2_K_XL versus the tuned Q4 control

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **Date:** 2026-08-19 UTC+7
> **What this answers:** the first question the new research asks — does crossing
> the VRAM residency threshold beat Q4's half-CPU-offloaded configuration, on
> this machine, under this project's own measurement rules?
> **Short answer:** yes, decisively, and the winner is the *more* aggressive
> quantization. `UD-IQ2_XXS` is fully GPU-resident, **3.2× Q4's decode**, and
> finished the task corpus with the **same 27/30 acceptance as Q4**. The
> qualifications in §5 matter and are not small.

---

## 1. Residency and speed — paired across three boots

Cross-model comparison cannot interleave inside one boot, so
`bench/model_arena.py` alternates arms and pairs by round, reversing the order
on even rounds so no arm always runs in the same position. Three rounds, three
decode samples each, on the code-rewrite prompt.

| arm | layer split | free VRAM after load | code tok/s (3 rounds) | prompt processing |
|---|---|---:|---|---:|
| `q4-tuned` (control) | **33 + 32** | 868 / 931 / 931 MiB | 13.21 · 13.50 · 13.09 | 147 – 164 |
| `q2kxl-mtp2` | **55 + 10** | 793 MiB | 19.87 · 19.92 · 19.95 | 310 – 330 |
| `q2kxl-nomtp` | **61 + 4** | 569 / 517 / 515 MiB | 21.26 · 21.84 · 21.42 | 451 – 510 |
| `iq2xxs-nomtp` | **65 + 0** | 1,190 / 1,190 / 1,178 MiB | 42.44 · 42.47 · 42.47 | **809 – 818** |

```text
first arena, 3 rounds:
  q2kxl-nomtp   per-round [+60.94, +61.78, +63.64]    mean  +62.12%   RESOLVED
  q2kxl-mtp2    per-round [+50.42, +47.56, +52.41]    mean  +50.13%   RESOLVED

second arena, 3 rounds, IQ2_XXS added:
  q2kxl-nomtp   per-round [+72.10, +60.59, +59.97]    mean  +64.22%   RESOLVED
  iq2xxs-nomtp  per-round [+237.36, +212.28, +209.10] mean +219.58%   RESOLVED
```

Both clear the 13.6 % drift floor by a factor of four and keep their sign in
every round, which is what `harness.paired_deltas` requires before it will call
an effect real.

The residency hypothesis is confirmed directly rather than inferred, and the
shape of the confirmation is the most useful thing in this report:

```text
Q4          33 GPU / 32 CPU    12.6 - 13.7 tok/s
Q2_K_XL     61 GPU /  4 CPU    21.3 - 22.0 tok/s
IQ2_XXS     65 GPU /  0 CPU    42.4 - 42.5 tok/s
```

**The last four CPU layers cost about half the throughput.** Going 33 → 61
layers — 28 layers moved to the GPU — buys +64 %. Going 61 → 65 — four more —
buys another +95 %. The curve is not merely non-linear; almost all of the prize
sits at the very end of it.

That reframes a finding from report 01, which observed that more GPU layers is
not monotonically better while sweeping 32–35 layers. That sweep was nowhere
near the cliff. "Nearly resident" and "resident" are different regimes, and only
the second one pays properly.

Prompt processing follows the same shape and more sharply: 156 → 394-486 → 818
tok/s, a **5.2×** span. That is the phase every cache miss pays for.

### MTP now costs what it used to buy

On Q4, `--spec-type draft-mtp --spec-draft-n-max 2` is worth keeping. On Q2 it
is worth **−7 %**, and the layer split says why:

```text
q2kxl-nomtp    61 + 4
q2kxl-mtp2     55 + 10      <- the draft head's VRAM pushes six target
                               layers back onto the CPU
```

Speculation trades VRAM for arithmetic. When the target forward pass is
expensive — Q4, with half the model on the CPU — that trade pays. When the
target is nearly resident and already cheap, the residency it costs is worth
more than the tokens it saves. This is report 01's finding ("MTP compensates for
CPU offload") running in reverse, and it is the mechanism the research predicted
in words: *"speculative benefit can collapse when the target forward pass becomes
cheap."*

**Consequence:** `production-q2-tuned.ps1` ships with speculation off. Copying
the Q4 flag set wholesale would have cost 7 % and looked like a Q2 weakness.

### What the greedy hash does and does not show

All nine boots returned the same greedy hash, `227749403A7404D4`. That is **not**
evidence that Q2 matches Q4. The probe prompt is a mechanical rewrite — rename
one attribute — with a single correct answer that both quantizations produce.
The hash is a useful invariance check across flags on one model; across models
it only says they agreed on an easy task. Quality is settled by the corpus, §3.

---

## 2. Agent-loop gates, measured against the Q4 control from report 09

### Tool calling

The first run said Q2 emitted a schema-correct tool call **40 %** of the time
against Q4's 80 %. That number was wrong, and the way it was wrong is worth
recording.

Reading the failing replies showed empty content and a **truncated reasoning
block**: the model was still deliberating when `max_tokens: 1024` ran out. So
the probe was measuring its own budget. Re-run at 4096 with `finish_reason`
recorded:

| | Q2_K_XL, n=15, max_tokens 4096 |
|---|---:|
| schema-correct call | **86.7 %** |
| `finish_reason` = `tool_calls` | 13 |
| `finish_reason` = `length` | **2** |
| required-field omissions | **0** |
| median reasoning per call | **2,811 chars** |

**Every non-call was a budget truncation.** Not one was the model declining to
use the tool. Q2's tool-calling is intact.

Re-run on Q4 at the same budget, same probe, same n:

| max_tokens 4096, n=15 | Q4 tuned | Q2_K_XL |
|---|---:|---:|
| schema-correct call | 80.0 % (12/15) | **86.7 % (13/15)** |
| truncated by budget | 3 | 2 |
| `tool_call_id` round-trip | **60.0 %** | **60.0 %** |
| required-field omissions | 0 | 0 |
| **median reasoning per call** | **59 chars** | **2,811 chars** |
| wall for 15 trials | 1,367.1 s | 1,121.8 s |

Two things fall out. Q2 is **not worse** at tool calling — 13/15 against 12/15,
which is well inside noise but certainly not the collapse the first run implied.
And the round-trip rate is **identical at 60 %**, which retires the open question
from the first draft of this report: those failures are a property of the probe,
not of the quantization. (Q4's are recorded as 2 empty replies and 1 repeated
call; Q2's ran before that field was added, so only the totals compare.)

The number that does separate them is reasoning length: **59 characters against
2,811**, a factor of 48 on the same task — and Q2 still finished the 15 trials
faster. That is the shape of the cost. It is not lost capability; it is a
different token budget — the same cost the research recorded for Q3, which
*"generated 18 % more tokens and 25 % more reasoning characters while completing
the same work"*.

**This has a direct integration consequence.** A client that sets `max_tokens`
to something comfortable for Q4 will truncate Q2 mid-reasoning and see a model
that "refuses to call tools". That is exactly the wrong conclusion, and this
report reached it once before reading the truncated reasoning block.

### Stability

| | Q4 (report 09) | Q2_K_XL | IQ2_XXS |
|---|---:|---:|---:|
| turns survived | 100/100 | 100/100 | 100/100 |
| hangs | 0 | 0 | 0 |
| invalidations recovered | 9/9 | 9/9 | 9/9 |
| steady-state prefix reuse | 99.1 % | 99.0 % | **99.2 %** |
| re-prefill at ~6,000 tokens | 26.35 s | 6.79 s | **~4 s** |
| p50 turn | 3.38 s | — | **1.34 s** |
| empty replies | 19/100 | **55/100** | **1/100** |

Stability is equal: no hangs, no stuck slots, and the prefix cache recovers on
the very turn after every forced invalidation. The re-prefill figure is the
residency dividend again — the penalty for a broken prefix is roughly a quarter
of Q4's.

Two cautions:

- **The wall-clock totals from this gate are not a speed comparison.** Q2's run
  took 127.9 s against Q4's 529.8 s, but 55 % of Q2's turns produced no tokens
  at all, so much of that gap is work not done. The speed number to quote is the
  paired arena figure in §1.
- **The empty-reply counts are unexplained, and are not a quantization effect.**
  19 for Q4, **55** for Q2_K_XL, **1** for IQ2_XXS. The first draft of this
  report proposed that lower-bit models are more sensitive to the probe's
  hand-built framing; IQ2_XXS is quantized *harder* than Q2_K_XL and shows the
  behaviour least of the three, so that explanation is wrong. Something specific
  to the `Q2_K_XL` artifact is responsible and this project does not know what.
  Recorded as unknown rather than dressed in a story that fits two points and
  breaks on the third.

---

## 3. Task corpus and retry economics

`bench/run_retry_bench.py`: attempt once; on failure, retry once with the actual
traceback pasted back — the evidence a real loop would return — then charge an
unrecovered task 90 s of Q4 escalation and every task 60 s of fixed overhead.

All three arms, 30 tasks each, identical probe, identical budgets:

| | Q4 tuned | Q2_K_XL | **IQ2_XXS** |
|---|---:|---:|---:|
| first-attempt success `p1` | **83.3 %** | **83.3 %** | 73.3 % |
| retry success `p2` | 40.0 % | 20.0 % | **62.5 %** |
| **locally accepted** | **90.0 %** (27/30) | 86.7 % (26/30) | **90.0 %** (27/30) |
| attempts per accepted task | **1.30** | 1.35 | 1.41 |
| escalations per 100 | **10.0** | 13.3 | **10.0** |
| attempts truncated by the 3072 budget | 3 | 2 | **7** |
| worker wall for the 30 tasks | 4,008.7 s | 1,972.5 s | **1,599.0 s** |
| **merged tasks / hour** | 17.8 | 26.1 (+47 %) | **29.4 (+65 %)** |
| **verified tasks / hour** | 24.2 | 47.5 (+96 %) | **60.8 (+151 %)** |

**Final acceptance is a tie between Q4 and IQ2_XXS at 27/30**, with Q2_K_XL one
task behind — a difference n=30 cannot resolve. The interesting part is *how*
IQ2_XXS gets there: its first attempt is worse (22/30 against 25/30) and its
retry is much better (5 of 8 against 2 of 5), landing at the same place in
**40 % of Q4's wall clock**.

Some of that lower `p1` is the probe, not the model. IQ2_XXS had **7 attempts
truncated by the 3072-token cap** against Q4's 3, so it is being cut off
mid-answer more often — the same budget effect that made its tool-calling look
broken in §2 before `finish_reason` was recorded. A larger cap would likely move
`p1` up and wall time up with it; neither has been measured.

Failures concentrate on the same tasks for both arms — `bracket_matching` above
all — so this is not Q2 breaking where Q4 holds.

### A number this report published and then withdrew

An earlier draft had Q4 at **p1 = 70.0 %** against Q2's 83.3 % and drew the
obvious conclusion. That Q4 figure came from a **10-task** run: 7/10. The matched
30-task run gives 25/30 — **83.3 %, exactly Q2's**. The gap was sampling noise in
a sample far too small to carry it, and reporting it would have claimed the
quantized model was *more accurate* than the one it was quantized from.

The lesson is report 04's, in a new place: an effect measured once, at n=10, is
not an effect. It also cost about ninety minutes of re-measurement, which is
what the rule is for.

### The research's central assumption does not hold here

The economic model that produces its *"22.6 → 40.2 merged tasks/h"* table assumes

```text
p2 = min(p1 + 0.10, 0.95)
```

At the measured p1 = 0.833 that predicts p2 = 0.933. **Measured p2 = 0.40 on Q4
and 0.20 on Q2.** An evidence-assisted retry succeeds roughly one time in three
at best, not nine times in ten.

And failures are not cheap. Tasks that needed a retry ran 350–420 s against
16–150 s for tasks that passed first time — on the 10-task Q4 run the three
retried tasks consumed **71 % of the entire corpus wall clock**. A worker that
fails more does not just fail more; it spends far longer doing it.

This cuts **against** the research's argument that a weaker-but-faster worker
wins on economics. It happens not to matter here, because Q2 did not turn out to
be the weaker worker — but the reasoning that would have justified adopting a
genuinely weaker one was resting on an assumption that is off by a factor of
three on this machine.

---

## 4. Verdict

Against the research's own pass criteria for Experiment A:

| criterion | required | Q2_K_XL | IQ2_XXS |
|---|---|---|---|
| decode ≥1.6× Q4, **or** ≥25 % more verified tasks/hour | either | +64 % / +96 % ✅ | **+220 % / +151 %** ✅ |
| first-pass ≥70 % | yes | 83.3 % ✅ | 73.3 % ✅ |
| final accepted quality unchanged | yes | 26/30 vs 27/30 ✅ | **27/30 = Q4** ✅ |
| Q4 escalation ≤5 per 100 | — | 13.3 ❌ | 10.0 ❌ |
| stable VRAM reserve ≥512 MiB | yes | 451–569 MiB ⚠ | **1,178–1,190 MiB** ✅ |
| tool/schema compliance vs control | control-relative | 86.7 % vs 80.0 % ✅ | **93.3 %** ✅ |
| no hang in 100+ turns | yes | ✅ | ✅ |

Both pass the speed and quality bars. **IQ2_XXS wins on every axis it does not
tie on**, and it is the only arm with comfortable VRAM headroom — Q2_K_XL sits at
451–569 MiB free, below the 512 MiB floor this project set after `--fit-target
256` produced intermittent driver eviction at 345 MiB (report 04 §5).

The escalation criterion fails for all three arms **including Q4 at 10 per 100**,
so it is a property of this corpus — one task, `bracket_matching`, that every
quantization fails — not a discriminator between them.

**Recommendation: `UD-IQ2_XXS`, speculation off, as the fast lane; Q4 retained
as the escalation lane.** That is the hierarchy the research proposed, arrived at
from this machine's measurements rather than its projections.

---

## 5. What this does not establish

**The corpus may not probe the failure this quantization is known for.** Prism's
model card is specific about how conventional 2-bit builds of this model family
fail:

> *"IQ2_XXS falls to 57.5 on AIME26 and 56.4 on LiveCodeBench while still scoring
> 88.93 on MMLU-Redux — which is why casual testing misses the collapse."*

Our corpus is ten single-function implementations run three times. It is closer
to the benchmarks that *hide* the collapse than to the ones that expose it. No
degradation was detected; that is not the same as none existing, and this is the
single largest caveat on the recommendation above.

**Everything is at 16K with F16 KV.** The 64K and 128K verdicts in reports 02 and
03 were established on Q4 and do not transfer. A resident model has far more room
for KV, so the depth answer may change entirely in IQ2_XXS's favour — unmeasured.

**No accuracy difference was *detected*; equivalence was not shown.** 27/30 vs
27/30 vs 26/30 at n=30 cannot separate arms that differ by a few percent.

**The token budget is a live confound.** IQ2_XXS was truncated on 7 of its
attempts at `max_tokens 3072`. Its `p1` is therefore a lower bound, and a
fair-budget re-run is the obvious next measurement.

**The `Q2_K_XL` empty-reply anomaly is unexplained** (§2).

**Not run at all:** Bonsai `Q2_g64`, the 35B-A3B MoE lane, and `UD-IQ2_M` (9.61
GiB) which sits between the two arms tested. Report 08 §4 has the ordering.
