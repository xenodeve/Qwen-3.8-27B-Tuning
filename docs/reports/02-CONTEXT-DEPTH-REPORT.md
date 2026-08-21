# Context-Depth Report — Where This Machine Actually Operates

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **Status:** complete for 16K–128K; 256K stopped under the paging condition
> **Date:** 2026-08-18 UTC+7
> **Builds on:** `00-Q3-VS-Q4-BENCHMARK-REPORT.md`, `01-RUNTIME-TUNING-REPORT.md`
> **Answers:** plan Phase F, and the "single most likely thing to invalidate the verdict" flagged in report 00 §8
> **Raw artifacts:** `C:\AI\qwen38-tuning\` — `EXPERIMENTS.md` (E11), `results\depth-sweep.jsonl`,
> `results\kv-equivalence.json`, `bench\depth_sweep.py`, `bench\kv_equivalence.py`

---

## 0. Result

Three questions were open. All three are now answered, and two of the answers
are negative — which is why they were worth measuring.

| question | answer |
|---|---|
| Does Q3 overtake Q4 at depth? | **No.** Q3 keeps more GPU layers and prefills faster at every depth, and still never wins decode. |
| Is 256K usable? | **No.** It loads, then drives the host to 0.63 GB free RAM and 10 GB of pagefile. |
| Does Q8_0 KV help? | **At 64K yes — +17 % decode at identical task quality. At 16K no.** |

**Operating recommendation: 16K–32K with F16 KV for everyday work; 64K with Q8_0 KV
when the task genuinely needs the depth.** Beyond 64K this machine is not an
interactive coding agent regardless of quant or KV precision.

---

## 1. Q4 collapses with depth

~80 % of each window filled with realistic source text. Cold prefill measured
once per depth; decode measured 5× reusing the prefill via `cache_prompt`, which
is also how an agent behaves — one cold turn, then appends.

| ctx | GPU / CPU layers | KV | cold prefill | decode | vs 16K |
|---|---|---|---|---|---|
| **16K** | 33 / 32 | 512 MiB | 40 s | **9.77** | — |
| **32K** | 31 / 34 | 1 024 MiB | 80 s | 7.44 | −24 % |
| **64K** | 27 / 38 | 2 304 MiB | 205 s | 4.37 | −55 % |
| **128K** | 20 / 45 | 5 632 MiB | 481 s | **2.10** | **−78 %** |
| 256K | — | — | — | **stopped** | §4 |

The mechanism is direct: KV growth evicts GPU layers, 33 → 20 across the range.

At 128K a 500-token reply takes **4 minutes** and a cold prefill **8 minutes**.
That is not a tool anyone drives interactively.

The measured KV curve (512 → 1 024 → 2 304 → 5 632 MiB) also confirms the
correction already applied to report 00: 256K lands near **11 GiB**, not the
~16 GiB the research-doc proxy predicted, because `qwen3_5` is hybrid and only
about a quarter of its layers hold a growing cache.

---

## 2. Q3 never overtakes Q4 — the crossover does not exist

This was the largest open risk in the project. Report 00 §8 named it as
*"the single most likely thing to invalidate the verdict."*

| ctx | | GPU layers | KV | prompt processing | decode |
|---|---|---|---|---|---|
| 64K | Q4 | 27 | 2 304 MiB | 227.3 | **4.37** |
| 64K | Q3 | **34** | 2 048 MiB | **299.9** | 3.68 |
| 128K | Q4 | 20 | 5 632 MiB | 193.4 | **2.10** |
| 128K | Q3 | **26** | 5 120 MiB | **244.5** | 2.09 |

Q3 does exactly what the theory predicted — **7 more resident layers at 64K, 6
more at 128K, and consistently faster prefill** — and still loses or ties on
decode. Q4 is 19 % faster at 64K; at 128K they are indistinguishable.

**The verdict survives at every depth measured.**

The mechanism matches the 16K result, where Q3 baseline beat Q4 baseline but
Q3+MTP lost to Q4+MTP: `UD-Q3_K_XL` is expensive enough per token to give back
its residency advantage. **Layer count is not the only variable — per-layer cost
is too, and the smaller quant is not the cheaper one here.**

---

## 3. Q8_0 KV: the one deep lever that pays

| ctx | | GPU layers | KV | prompt processing | decode |
|---|---|---|---|---|---|
| 64K | F16 | 27 | 2 304 MiB | 227.3 | 4.37 |
| 64K | **Q8_0** | **29** | **1 224 MiB** | **256.1** | **5.10** (+16.7 %) |
| 128K | F16 | 20 | 5 632 MiB | 193.4 | 2.10 |
| 128K | **Q8_0** | **23** | **2 720 MiB** | **212.9** | **2.48** (+18.1 %) |

KV halves as expected, buying 2–3 GPU layers and ~17 % decode at both depths —
comfortably above the 13.6 % drift floor established in report 01 §7.

### 3.1 The custom build is not required

The deep-research report warned that requesting a quantized-KV Flash-Attention
kernel that was not compiled falls back to a catastrophically slow path, and
recommended building a pinned SM89 binary with `GGML_CUDA_FA_ALL_QUANTS=ON`
**before** testing Q8.

Measured on the stock b10472 binary, Q8 KV is **faster** than F16 at both depths.
**No rebuild needed.** That was worth testing rather than assuming — the
recommendation would have cost a source build for nothing.

### 3.2 The cheap equivalence check missed a real divergence

Report 01 §1 verifies quality with a greedy hash. For flags that do not change
arithmetic that is *stronger* than a pass-rate comparison. **Q8 KV changes the
arithmetic**, and the probe sends a 4-token prompt — so it barely touches the
very cache Q8 quantizes. It reported "hash identical", and that was not evidence.

Re-tested with ~46.5K tokens of context so the continuation is decided by
attention over a deeply-populated cache, identical greedy settings:

```text
prompt_n = 46 557
F16  hash 1A4F7C9924198E8A
Q8   hash 05C38B387571F755
common prefix: 1 character of 778
```

Completely different output. Divergence alone is not damage, so it was then
measured on the task corpus:

| config | pass rate | verified tasks/hr | median tok/s |
|---|---|---|---|
| tuned, **F16 KV** | **90.0 %** (27/30) | **36.1** | **12.27** |
| tuned, **Q8_0 KV** | 86.7 % (26/30) | 29.9 | 10.63 |

At 16K, Q8 KV is **worse on both axes** — it loses one `lfu_cache` attempt and
runs slower, because at 512 MiB of KV there is nothing meaningful to reclaim and
only the cost remains.

**`lfu_cache` is the corpus's precision canary**: it is the task Q3 loses (2/3),
the task Q8 KV loses (2/3), and the only task where any configuration has ever
differed from Q4/F16's clean sweep.

### 3.3 Measured at depth: Q8 KV costs nothing at 64K

The 16K corpus judges Q8 in the regime where Q8 has no benefit, so a deep-context
corpus was built for the question it could not answer
(`bench/deep_tasks.py`, `bench/run_deep_bench.py`).

Six execution-verified tasks whose correct answers depend on constants planted at
increasing depth in a shared ~44K-token repository prefix — arbitrary values
(`MAX_RETRIES = 7`, `TIMEOUT_MS = 8700`, `CHECKSUM_FIELD = "drain_token"`) that no
prior can supply. A model answering from priors, or one whose attention over a
quantized cache has degraded, writes code that fails the assertions. The shared
prefix means `cache_prompt` pays the deep prefill once and every later task reuses
it — which is also how an agent behaves.

| | F16 KV | **Q8_0 KV** |
|---|---|---|
| pass rate at 64K | 100 % (18/18) | **100 % (18/18)** |
| verified tasks / hour | 51.8 | **57.4** (+10.7 %) |
| warm turn, median | 51.2 s | **48.4 s** |
| cold prefill | 349.1 s | **321.0 s** |

Every task passed 3/3 on both arms, including `deep_combine_310` (three values
from the deepest planted class) and `deep_default_contrast` (a comparison across
the whole file).

**Q8_0 KV at 64K is free throughput.** The raw greedy divergence in §3.2 is real
but not task-relevant at this depth.

**Honest limit:** both arms hit 100 %, so this **bounds** the damage rather than
measuring a small one. With 18 samples at ceiling it rules out a large regression
and cannot resolve a 2–3 % loss. A harder corpus would be needed for that.

**Two defects the corpus's own tests caught before it was trusted**, both of which
would have produced a confident verdict from a broken instrument:

- `Handler0017` was emitted **twice** — once as a routine block at index 17 and
  once as the planted block — so "the class for shard 17" had two contradictory
  answers in context.
- The first size test checked only a lower bound, so a **112K-token** corpus
  passed and then failed every request with HTTP 400 against a 64K window
  (0/18, in four seconds). Both bounds are now asserted.

---

## 4. 256K is not viable

The server loaded at `n_ctx=262144` and began prefilling. It was stopped under
the protocol's paging condition, not on a throughput number:

```text
host RAM free   0.63 GB of 47.69
pagefile used   10.11 GB
llama-server    working set 26.64 GB
pages/sec       296
```

Any throughput measured under that pressure describes Windows paging rather than
the model. Recording it as a number would have been worse than recording it as a
stop.

This settles the framing already adopted in research doc 07: **256K is a
configured ceiling, not a working set.** On this machine it is not even a usable
ceiling with F16 KV.

---

## 5. What to run

| working context | configuration | why |
|---|---|---|
| **16K–32K** | tuned Q4, **F16 KV** — `production-q4-tuned.ps1` | 9.8–7.4 tok/s; Q8 measured worse (86.7 % vs 90.0 %) and slower |
| **64K** | tuned Q4, **Q8_0 KV** — `production-q4-deep.ps1` | 5.10 vs 4.37 tok/s at **identical** 18/18 task quality |
| 128K | possible but impractical | 2.1–2.5 tok/s, 8-minute cold prefill |
| 256K | **do not** | pages the host |

Combined with report 01 §2's prefix-cache rule, the practical guidance for the
agent integration is: **keep the working context small and the prefix frozen.**
Those two together matter far more than any flag — a preserved cache turns a
40-second prefill into 2.4 seconds, while the entire runtime-flag stack is worth
about 7–10 %.

---

## 6. Still not covered

1. **The deep corpus has a ceiling** — both arms scored 18/18, so a small Q8
   regression would be invisible (§3.3). Only 64K was tested this way; 128K is
   unmeasured for quality.
2. **`--cache-ram`** (default 8192 MiB) untouched — relevant given the 256K paging
   result.
3. **CPU KV placement** (`--no-kv-offload`) untested at depth, where it could trade
   PCIe latency for weight residency.
4. **Depth measurements are n=1 per configuration** for prefill. Decode is n=5.
   Given the 13.6 % drift floor, the 64K Q3-vs-Q4 gap (19 %) is meaningful but the
   128K tie (0.5 %) would need a paired design to call anything but a tie.
5. **No OpenCode / real-repo run at depth.**

---

## 7. IQ2_XXS at depth — moved

Everything in this report was measured on **Q4**. The ladder was re-run on
`UD-IQ2_XXS` after Experiment A replaced the 16K default, and it lives in its own
report rather than as an appendix here:

**→ [11-DEPTH-ON-IQ2XXS.md](11-DEPTH-ON-IQ2XXS.md)**

The one-line result: ~3× faster at every depth, 128K cold prefill down from
~720 s to 196 s, and 256K runs without host paging where Q4 had to be stopped —
at 1.71 tok/s. Retrieval quality at depth remains verified on Q4 only.
