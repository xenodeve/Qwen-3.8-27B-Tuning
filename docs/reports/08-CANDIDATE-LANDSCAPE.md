# Candidate Landscape — What Is Actually Downloadable and Runnable Here

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **Date:** 2026-08-19 UTC+7
> **Why this exists:** a new deep-research document
> (`docs/researchs/Deep Research/Local Worker Model, Quantization, and Runtime
> Selection for RTX 4070 SUPER 12GB.md`) redirects the project away from further
> flag tuning and toward changing the model's memory regime. Its candidate table
> is the most useful thing this project has received in a while — and it is a
> **hypothesis about the outside world**. This report records which parts of it
> survived being checked against Hugging Face and against our own binary, before
> a single gigabyte was downloaded on the strength of it.

---

## 0. The redirection, in the research's own words

> *"Do not spend another tuning cycle on Q3, Q4 ngram, speculative sub-knobs, or
> tiny 16K batch/thread changes. Those questions are already answered by your
> machine measurements, and improvements below the ~14% restart-noise envelope
> are not where the next order-of-magnitude gain lives."*

That agrees with our own report 06, which had already reduced the remaining
runtime knobs to items expected to sit under the 13.6 % floor. **Accepted.** The
experimental program below replaces the flag queue.

The metric does not change: **verified successful coding tasks per hour**, not
tok/s. The research restates it as *Verified Merged Tasks/Hour* and adds the
economic reason a weaker worker can still win — a defect caught by tests is a
cost, not a quality regression.

---

## 1. Verified facts about the candidates

Checked 2026-08-19 against the Hugging Face API. Sizes are exact byte counts
from the repo listing, converted to GiB — not the round numbers the research
quotes, several of which came from vendor prose.

| artifact | repo | bytes | **GiB** |
|---|---|---:|---:|
| Qwen3.8-27B `UD-Q4_K_XL` *(control, on disk)* | `unsloth/Qwen3.8-27B-GGUF` | 17,923,394,624 | **16.69** |
| Qwen3.8-27B `UD-Q3_K_XL` *(rejected, on disk)* | same | 13,441,059,904 | **12.52** |
| Qwen3.8-27B `UD-Q2_K_XL` | same | 10,676,423,744 | **9.94** |
| Qwen3.8-27B `UD-IQ2_M` | same | 10,319,907,904 | **9.61** |
| Qwen3.8-27B `UD-IQ2_XXS` | same | 9,010,048,064 | **8.39** |
| Ternary Bonsai 27B `Q2_0` *(g128 — needs the PrismML fork)* | `prism-ml/Ternary-Bonsai-27B-gguf` | 7,165,121,600 | **6.67** |
| Ternary Bonsai 27B `Q2_g64` *(the mainline-packing build)* | same | 7,585,330,240 | **7.06** |
| Ternary Bonsai 27B `PQ2_0` *(identical size to `Q2_0`; undocumented)* | same | 7,165,121,600 | **6.67** |
| Bonsai `dspark-Q4_1` drafter | same | 1,946,393,568 | **1.81** |
| Qwen3.6-35B-A3B `UD-Q4_K_XL` | `unsloth/Qwen3.6-35B-A3B-GGUF` | 22,360,456,160 | **20.83** |
| Qwen3.6-35B-A3B `UD-IQ4_XS` | same | 17,730,509,792 | **16.51** |

### Five places the research, or this report, was wrong or imprecise

1. **Ornith lives at `ornith-ai/`, not `deepreinforce-ai/`.** Official GGUF
   builds exist for both sizes (`Ornith-1.0-9B-GGUF`, `Ornith-1.0-35B-GGUF`),
   plus Unsloth and bartowski repackages. A copy-pasted `-hf
   deepreinforce-ai/...` would have failed.

2. **Bonsai's headline artifact DOES need the fork — a claim this report got
   wrong once and is correcting.** Our binary lists both halves:

   ```text
   llama-quantize --help :  41 or Q2_0  : 2.25 bpw quantization (group 64)
   llama-server --help   :  --spec-type ... draft-dspark ...
   ```

   On the strength of that, this report first concluded that Experiment B runs
   on the stock binary. It does not. Prism's own Quickstart opens with *"Clone
   the PrismML fork of llama.cpp (includes the Q2_0_g128 hybrid-attention
   kernels)"*, and their `Ternary-Bonsai-27B-Q2_0.gguf` is **g128**, whereas
   mainline's `Q2_0` is **group 64**. Same name, different format. The name
   match was mistaken for a format match.

   What survives the correction: `--spec-type draft-dspark` genuinely is
   mainline, and Prism publishes a second pack, **`Q2_g64.gguf` (7.06 GiB)**,
   described as *"matching the 64-value-group Q2_0 packing in llama.cpp"*. That
   -- not `Q2_0` -- is the artifact to try on our binary, and whether mainline's
   kernels actually serve it is now an open question rather than an assumption.

3. **`-hf repo:Q2_0` silently resolves to the wrong file, and the sizes cannot
   catch it.** The Bonsai repo holds `Q2_0.gguf` and `PQ2_0.gguf` at **exactly
   the same byte count** (7,165,121,600). The quant tag matches by substring, so
   `:Q2_0` began downloading `PQ2_0` -- confirmed by comparing the cached blob
   name against the repo OIDs:

   ```text
   blob being written  e4781999...  = Ternary-Bonsai-27B-PQ2_0.gguf
   the intended file   868c1171...  = Ternary-Bonsai-27B-Q2_0.gguf
   ```

   Nothing in the size, the log, or the cache layout would have reported it. Two
   gigabytes had already transferred. **Any `-hf ...:<tag>` fetch from a repo
   with more than one artifact must be verified against the repo OID list**
   (`/api/models/<repo>/tree/main`) before it is measured, or fetched by exact
   filename with `hf download`.

4. **The "AtomicChat AD-IQ2_XS at 9.9 GB" arm is not a separate download.**
   Unsloth's `UD-Q2_K_XL` is 9.94 GiB — the same size class, from the repo we
   already use, with the same chat template and MTP head. The quantizer battle
   can be run without introducing a second vendor.

5. **Bonsai's mainline-compatible pack is 7.06 GiB (`Q2_g64`), not the 6.67 GiB
   `Q2_0`.** With the 1.81 GiB drafter the pair is **8.87 GiB** -- still the
   first candidate in this project where target *and* drafter could both be
   resident, but only if mainline serves the g64 pack at all.

### The one thing that decides Experiment A

Free VRAM at boot on this machine has ranged **9,326 – 10,530 MiB** across
recorded launches (report 04 §2, plus 9,326 measured after the most recent
reboot). Against that range:

```text
UD-Q2_K_XL   9.94 GiB  = 10 179 MiB   -> straddles the range: may or may not fit
UD-IQ2_XXS   8.39 GiB  =  8 592 MiB   -> under the floor: should be fully resident
Bonsai g64   7.06 GiB  =  7 230 MiB   -> comfortably resident, drafter too
```

So the two Q2 arms are not redundant: **they sit on opposite sides of the
residency threshold that the whole hypothesis is about**, and which side
`UD-Q2_K_XL` lands on is decided by the boot, not by the file. This is why the
harness parses the layer split out of the log every boot instead of trusting a
number from a previous run.

---

## 2. Experiment design

Cross-model comparison breaks the design used for every earlier result here.
Flag sweeps could interleave arms **inside one boot**; two quantizations cannot
share a boot, because the weights differ. Every arm change costs a restart, and
report 04 measured a **13.6 % peak-to-peak spread across restarts of an
unchanged config**. Control-first ordering would measure that drift and report
it as quantization.

The replacement is `bench/model_arena.py`:

```text
round 1:  A  B  C
round 2:  C  B  A      order reversed, so no arm always runs in the same
round 3:  A  B  C      position within a round
```

Each boot records free VRAM *before* launch, VRAM used/free after load, the GPU
+CPU layer split parsed from the log, three decode samples on the code-rewrite
prompt, prompt-processing rate, MTP draft/accept counts, and a greedy hash.
Per-round medians go to `harness.paired_deltas`.

`paired_deltas` was written test-first (8 new tests, `bench/tests/test_harness.py`,
38 passing) because summarising arithmetic is precisely what failed twice in this
project — a field named `median` that held the maximum, and per-sweep deltas that
were summed across independent controls into a "+19 %" that was really +6.6 %.
It refuses to call an effect real unless it **clears the 13.6 % floor and keeps
its sign in every round**; a mean of +40 %/−10 % is reported as unresolved.

### Staging

Stage 1 is a decode + residency probe, ~5 min per boot. It answers *"does the
smaller artifact actually cross the residency threshold, and by how much does
decode move?"* — cheaply enough that a dead lane dies before the expensive part.

Stage 2 is the 30-attempt corpus (`run_bench.py`), which is what produces pass
rate and verified tasks/hour. It costs roughly 45 min per arm at Q4 speed, so it
runs only on arms that survived stage 1.

Why an 11-token prompt is not used anywhere: it stayed inside 9.86–11.90 tok/s
across *every* configuration tested in this project and would resolve nothing
(report 04 §3).

---

## 3. What is a projection and must not be quoted as a result

The research is explicit that its speed and pass-rate columns are engineering
projections. Repeating them here so nothing leaks into a later summary as
measurement:

| claim | status |
|---|---|
| Q2 "18–35 tok/s", first-pass "~72–85 %" | **projection** |
| Bonsai "~35–65 tok/s at 16K", first-pass "~75–88 %" | **projection** |
| Bonsai vendor scores (coding 85.96 vs 88.74 FP16) | vendor benchmark, not ours |
| 35B-A3B "76.6 tok/s on RTX 4070 12GB" | **third-party community report**, different corpus |
| Qwen3.8-35B-A3B capability | projection about a model that has no public weights |
| Economic table (22.6 → 40.2 merged tasks/h) | model output from assumed \(p_1\), \(p_2\), \(H\) |

Ours to date, measured: **Q4 tuned = 90.0 % pass (27/30), 12.27 tok/s median,
36.1 verified tasks/hour** (report 01).

### Pass criteria adopted for Experiment A

From the research, unchanged, because they are stated in the right units:

- **≥ 21.5 code tok/s** (≈1.6× the present Q4 code lane), **or** ≥ 25 % more
  verified tasks/hour end-to-end;
- first-pass **≥ 70 %** with final accepted quality unchanged;
- stable VRAM reserve **≥ 512 MiB** — the `--fit-target 256` run already produced
  intermittent driver eviction at 345 MiB free and a 73 % spread (report 04 §5).

---

## 4. Order of work, and why

1. **Experiment A — Q2 quantizer battle.** Same family, same template, same tool
   protocol, same MTP head; only the weight representation and residency change.
   Cheapest answer per unit of integration risk.
2. **Experiment B — Ternary Bonsai.** Promoted in confidence, not in order: the
   fork risk the research assigned to it turned out not to exist here, and it is
   the only candidate whose target *and* drafter both fit. Its own vendor card
   warns that long-horizon agentic coding is not a strong target of the release,
   which is exactly the capability this project needs, so it is gated on protocol
   and stability before quality.
3. **Experiment C — MoE race.** `-ncmoe`/`--n-cpu-moe` exists in our build, so
   the physics is reachable. Gated behind a hostile multi-turn cache/state test
   before any quality number is collected, per the research's warning about
   llama.cpp hybrid/MoE cache invalidation.

**Status at the time of writing:** `UD-Q2_K_XL` and Bonsai `Q2_0` downloading;
no candidate measurement has been taken yet. Nothing in this report is a result.
