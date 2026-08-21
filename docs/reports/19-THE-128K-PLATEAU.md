# The 128K Plateau — Weight Size Decides Residency, Not Speed

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **Date:** 2026-08-20 UTC+7 · Qwen3.8-27B only
>
> **The finding in one line:** once an artifact is fully GPU-resident at 128K,
> making it *smaller* buys no further speed. Three artifacts spanning 5.77 to
> 6.77 GiB — a 17 % size range and a 16 % spread at 16K — are
> **indistinguishable at 128K**.
>
> This changes the recommendation. The project has been treating "smallest that
> fits" as the optimum. At depth the optimum is **largest that still fits.**

---

> ### ⚠ The central claim of this report was refuted the same night
>
> §3.3 said *"No flag in report 16 can raise 27 tok/s while the window stays at
> 128K, because the cost is the cache and the cache is fixed by the window."*
>
> **At 01:24 on 2026-08-21, `--spec-type ngram-mod` returned 81.46 tok/s at
> 131,072** on this same artifact, same cache size, same `65+0` split — **+213 %**
> over the control measured beside it. See
> [report 22 §0](22-SESSION-RECORD-2026-08-20.md).
>
> **What survives:** the plateau is real *across artifacts*. Every fully-resident
> arm ties, because the cache is 2,304 MiB for all of them, so a bigger or smaller
> model does not change decode at depth. That part is measured over ten boots and
> still holds — and so does its consequence, *prefer the largest artifact that
> stays resident.*
>
> **What does not:** the step from "changing the artifact does not help" to
> "nothing helps". All ten boots had speculation **off**. n-gram does not shrink
> the cache; it reduces how many times the cache is read.

> **Correction, 2026-08-21.** `output_contract_pct` is the **pass** rate —
> `100 * (attempts_seen - contract_violations) / attempts_seen` — not the violation
> rate. Text written on 2026-08-20 read it backwards. The figures are unchanged;
> their direction is. Higher is better.

## 1. The measurement

`bench/kv_sweep.py --ctx 131072 --arms q4_0`, cold prefill paid once per boot,
warm decode measured over the reused prefix. Eight boots across two sessions.

| artifact | GiB | 16K tok/s | **128K tok/s** | split | free MiB | KV MiB |
|---|---:|---:|---|:--:|---:|---:|
| V3 `UD-IQ1_S` | 5.77 | **50.55** | 27.29 · 27.45 · **27.35 · 26.95** | 65+0 | 842 · 803 · 1834 · 1815 | 2304 |
| V3 `UD-IQ1_M` | 6.27 | 43.75 | 26.37 · 27.45 | 65+0 | 552 · 630 | 2304 |
| V3 `UD-IQ2_XXS` | 6.77 | 44.84 | 26.72 · 26.16 | 65+0 | 446 · 493 | 2304 |

```text
spread across all eight boots:  26.16 - 27.45 tok/s  =  4.9 %
restart-drift noise floor    :  13.6 %
```

**The spread is a third of the noise floor.** These are not three results; they
are one result measured eight times.

### 1.1 The replication is stronger than it looks

The four `IQ1_S` boots ran under **free-VRAM headroom of 842, 803, 1834 and
1815 MiB** — a 1 GB difference in how much room was left over — and returned
27.29, 27.45, 27.35, 26.95. **Doubling the idle headroom changed nothing.**

That is the cleanest possible statement of the finding: past the point of full
residency, spare VRAM is inert.

---

## 2. Why — the cache is the same size for all of them

`KV self size` is **2,304.0 MiB for all three artifacts**, because the cache is
sized by *context length × layer count × head dimensions*, not by how many bits
the weights carry. All three are the same 65-layer Qwen3.8-27B; only their
weight quantization differs.

So at 128K each arm is doing:

- the same attention work over the same 2,304 MiB of cache, every token
- weight matmuls that differ by 17 % in size

and the first term dominates. Decode at depth is bound by streaming the cache,
not by the weights.

At 16K the cache is ~288 MiB and the balance is reversed — which is exactly why
the 16K ordering (50.55 / 44.84 / 43.75) exists and why it does not survive to
depth.

### 2.1 The cost of depth itself

| artifact | 16K | 128K | loss |
|---|---:|---:|---:|
| V3 `IQ1_S` | 50.55 | ~27.3 | **−46 %** |
| V3 `IQ2_XXS` | 44.84 | ~26.4 | **−41 %** |
| V3 `IQ1_M` | 43.75 | ~26.9 | **−38 %** |

Roughly **40 % of throughput is the price of the 128K window**, paid by every
arm, and it is paid *even when nothing leaves the GPU*. This is a different cost
from the residency cliff and had not been separated before: previous depth
numbers (7.84 tok/s at 128K on pre-V3 `IQ2_XXS`) were measured at 58+7, so they
bundled "the window costs attention work" with "the window evicted your layers".

**Separated:**

| effect | size |
|---|---|
| the 128K window itself, still resident | **−40 %** |
| plus losing 7 layers to the CPU | 27.3 → 7.84, a further **−71 %** |

---

## 3. What this changes

### 3.1 At 128K, choose the largest arm that stays resident

Since speed is flat across the resident band, the only axis left is quality, and
quality rises with bits. From the residency ladder measured the same day
(`ctx-ceiling-q38.jsonl`, `q4_0` KV, ~9.8 GB free at boot):

| artifact | GiB | deepest fully-resident | corpus (unconstrained) |
|---|---:|---:|---|
| V3 `UD-IQ1_S` | 5.77 | **196,608** | 0 accepted, no fenced block 12/12 |
| V3 `UD-IQ1_M` | 6.27 | 163,840 | 10/21 decided, **41.5 % contract PASS** |
| **V3 `UD-IQ2_XXS`** | 6.77 | **131,072** | **19/27 decided**, **58.3 % contract PASS** |
| `AD-IQ1_M` (AtomicChat) | 7.91 | ✗ — `65+1`, one layer short | **27/31 — the best corpus of any 1-bit** |
| V3 `UD-Q2_K_XL` | 9.15 | ✗ — 54+12 | — |
| `UD-Q2_K_XL` pre-V3 | 9.94 | ✗ — 50+16 | — |

**`UD-IQ2_XXS` is the largest artifact that holds 128K, and it costs nothing in
speed to prefer it over the two smaller ones.** The earlier framing — that
`IQ1_S` was "the fastest artifact this project has ever measured" — is true at
16K and irrelevant at 128K.

### 3.2 Extra headroom is only worth what it buys in *depth*

`IQ1_S` reaches 196,608 and `IQ2_XXS` stops at 131,072. That is the whole value
of the smaller weights: **more window, at the same tok/s**. If the workload needs
160K–192K, `IQ1_S` is the only Qwen3.8-27B arm that reaches it — but it must
first be made to emit an answer at all.

### 3.3 The remaining lever at 128K is format, not throughput

Every arm resident at 128K is already at the plateau. No flag in report 16 can
raise 27 tok/s while the window stays at 128K, because the cost is the cache and
the cache is fixed by the window. **The only way to more accepted tasks per hour
at this depth is to stop wasting attempts** — and only 41.5 % to 58.3 % of attempts
currently produce no fenced code block at all.

That is why `--grammar-file` plus `--reasoning-budget 0` is the next experiment
and not a tuning sweep.

---

## 4. What is still open

- **`--ctx-checkpoints 8`** — `AD-IQ1_M` misses residency at 128K by a single
  layer (`65+1`, 338 MiB free). It is also the only 1-bit artifact with a good
  corpus (27/31). If the default 32 checkpoints are worth more than ~125 MiB,
  that flag alone promotes the best-quality small artifact into the resident
  band. Queued.
- **Desktop VRAM is a live variable.** 33 desktop processes held **2,202 MiB**
  during these runs. `AD-IQ1_M` *was* resident at 128K in an earlier session with
  10,730–10,962 MiB free at boot, and is not with ~9,796. The residency ladder
  in §3.1 is therefore a ladder *at this desktop state*, and any arm within
  ~1 GB of its ceiling should be treated as conditional.
- **Whether the plateau holds above 128K.** All eight boots are at 131,072. If
  decode is cache-bound, 196K should be proportionally slower again — and that
  number decides whether `IQ1_S`'s extra depth is usable or merely reachable.
