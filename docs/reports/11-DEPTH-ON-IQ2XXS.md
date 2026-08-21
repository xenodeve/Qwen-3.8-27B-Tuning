# Depth on IQ2_XXS — the Full Ladder, and What 256K Actually Costs

> **Date:** 2026-08-19 UTC+7
> **Why this exists:** the depth ladder in reports 02 and 03 was established on
> **Q4**, before Experiment A (report 10) replaced the 16K default with
> `UD-IQ2_XXS`. A resident model has VRAM that Q4 never had, so the depth verdict
> could not be assumed to transfer. It was re-measured.
> **Short answer:** ~3× faster at every depth, 128K goes from barely usable to
> practical, and **256K runs at all for the first time** — at a price that makes
> it a budget for one deep question rather than for an agent loop.

---

## 1. The ladder

`bench/depth_sweep.py q8 iq2xxs nospec`. Q8_0 KV throughout, because report 03
measured it as the winner from 64K up on Q4 at identical task quality.
Speculation off, per report 10 §1. Cold prefill is n=1 by design — a 256K prefill
costs eleven minutes and N=3 would cost most of an hour for one cell; decode is
n=3 over the warm cache, which is also how an agent actually behaves.

| ctx | layer split | **decode tok/s** | prompt processing | cold prefill | KV | free VRAM |
|---|---|---:|---:|---:|---:|---:|
| 16K | **65 + 0** | **42.4** | 818 | — | ~512 MiB *(F16)* | 1,178 MiB |
| 64K | 61 + 4 | **15.81** | 727 | 64.0 s | 2,040 MiB | 447 MiB |
| 128K | 47 + 18 | **5.15** | 474 | 196.2 s | 3,264 MiB | 503 MiB |
| 256K | 31 + 34 | **1.71** | 284 | 658.1 s | 4,352 MiB | 412 MiB |

Against the Q4 ladder (reports 02 §5, 03):

| ctx | Q4 | IQ2_XXS | ratio |
|---|---:|---:|---:|
| 16K | 12.6 – 13.7 | 42.4 | **3.2×** |
| 64K | 5.10 | 15.81 | **3.1×** |
| 128K | 2.5 | 5.15 | **2.1×** |
| 128K cold prefill | ~720 s | **196.2 s** | **3.7×** |
| 256K | **stopped** — host paging | **1.71**, no paging | — |

---

## 2. What the ladder is actually measuring

The decode column tracks the layer split, not the context length:

```text
65 + 0    42.4 tok/s     KV ~0.5 GB
61 + 4    15.8 tok/s     KV  2.0 GB
47 + 18    5.2 tok/s     KV  3.3 GB
31 + 34    1.7 tok/s     KV  4.4 GB
```

This is the residency cliff from report 10 §1 seen from the other side. There,
shrinking the artifact bought GPU layers. Here, growing the context spends them —
KV is allocated from the same pool the weights live in, so every doubling of the
window pushes another block of layers onto the CPU.

At **256K the split is 31 + 34**, which is *worse than Q4's 33 + 32 at 16K*.
Every advantage this artifact has comes from residency, and at 256K there is none
left to have.

The right reading is therefore **not** "IQ2_XXS degrades with depth" — Q4
degrades with depth too, and faster in relative terms at 128K. It is:

> **Depth spends the same VRAM the quantization was chosen to free.**

Which also predicts where the remaining headroom is. The 16K profile leaves
1,178 MiB free; the deep profiles leave 412–503 MiB, below the 512 MiB reserve
this project adopted after `--fit-target 256` produced intermittent driver
eviction at 345 MiB free (report 04 §5). The deep rows are running closer to that
edge than the everyday profile is, and no instability was observed — but the
margin is thin and was not stress-tested at depth.

---

## 3. 256K: reachable, and what it costs

The Q4 attempt at 256K was **stopped**, not measured: host RAM free fell to
0.63 GB of 47.69 with a 10.11 GB pagefile and 296 pages/sec (E11). Anything
measured under that pressure would have described Windows paging.

IQ2_XXS holds 256K with **15.4 GB of host RAM still free**. The reason is not
subtle: its CPU-resident half is a much smaller thing to hold. The stop condition
that ended the Q4 experiment does not fire here, and the run completed normally.

Reachable is not the same as usable:

```text
cold prefill        658 s   (11 minutes)
500-token reply     293 s   ( 5 minutes at 1.71 tok/s)
```

So one deep question costs roughly a quarter of an hour, and every subsequent
turn that keeps the prefix costs five minutes per reply. For an agent loop —
inspect, edit, run tests, read output, repair — that is not a workable budget.
For "load this entire codebase and answer one architectural question", it is.

**Recorded as a capability, not a recommendation.** Report 05's profile table
labels the 256K row accordingly.

---

## 4. What this does not establish

**Retrieval quality at depth has not been re-verified on this artifact.** This is
the largest gap in the report. The `30/30` at 64K and `10/10` at a 114,406-token
prompt in report 03 were measured on **Q4**. Everything above is throughput and
residency.

That gap matters more here than it would for a milder quantization, because the
warning attached to this class of artifact is specifically about *selective*
failure. Prism's model card, on conventional 2-bit builds of this model family:

> *"IQ2_XXS falls to 57.5 on AIME26 and 56.4 on LiveCodeBench while still scoring
> 88.93 on MMLU-Redux — which is why casual testing misses the collapse."*

Deep retrieval is exactly the kind of task where a quantization can look fine on
aggregate and fail on the specific span that matters. The instrument already
exists — `bench/run_deep_bench.py --v2` with the corpus scaled to the window —
and it has not been run on IQ2_XXS. Until it has, **`production-q4-deep.ps1`
remains the profile for work that depends on finding one fact inside 100K
tokens.**

**Other limits.**

- Cold prefill is **n=1** at every depth. The decode figures are n=3 within one
  boot; no arm here is paired across boots, so a difference of a few percent
  between adjacent rows would not be resolvable. The differences that carry this
  report are 2–3×, far above that.
- **No stability run at depth.** The 100-turn gate in report 10 §2 was at 16K.
  Whether prefix reuse and invalidation recovery hold at 128K on this artifact is
  untested, and the free-VRAM margin there is thinner.
- **F16 KV was not measured on IQ2_XXS at depth.** Q8_0 was chosen from the Q4
  result. It is very likely still correct — the argument (halving the cache
  returns residency, and residency is what this whole report is about) is
  stronger for a resident model, not weaker — but it is inherited, not measured.

---

## 5. Consequences for the operating guide

Report 05's profile table now reads:

| working context | profile | note |
|---|---|---|
| 16K | `production-iq2xxs.ps1` | the everyday default |
| 64K / 128K | `production-iq2xxs-deep.ps1 -Ctx …` | Q8_0 KV, speculation off |
| 64K / 128K, retrieval-critical | `production-q4-deep.ps1` | until §4 is closed |
| 256K | `production-iq2xxs-deep.ps1 -Ctx 262144` | one deep question, not a loop |
| 16K escalation lane | `production-q4-tuned.ps1` | with MTP n=2 |

`production-iq2xxs-deep.ps1` carries all of the above in its header comment,
including the retrieval-quality warning, so a reader who launches the script
without opening this report still meets the caveat.
