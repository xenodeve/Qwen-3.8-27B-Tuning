# Deep-Context Quality — Does Q8_0 KV Damage Retrieval?

> **Status:** complete — 64K and 128K, both KV types, execution-verified
> **Date:** 2026-08-18 UTC+7
> **Builds on:** [02-CONTEXT-DEPTH-REPORT.md](02-CONTEXT-DEPTH-REPORT.md)
> **Harnesses:** `bench\deep_tasks.py`, `bench\deep_tasks_v2.py`,
> `bench\run_deep_bench.py`, `bench\kv_equivalence.py`
> **Raw data:** `results\deep-quality.jsonl`, `results\kv-equivalence.json`

---

## 0. The question

Q8_0 KV buys **+16.7 % at 64K and +18.1 % at 128K** (report 02 §3). But its
quality was first measured only at 16K, where KV is 512 MiB, Q8 has nothing to
reclaim, and it scored **worse**: 86.7 % vs 90.0 %.

A verdict from that measurement describes the wrong regime. This report measures
quality **at the depth where Q8 actually helps**.

---

## 1. Why the cheap check was not enough

The runtime sweeps validate quality with a greedy-hash probe: identical hash means
bit-identical output, which is stronger than a pass-rate comparison. That holds for
flags that do not change arithmetic — thread counts, batch sizes, fit margins.

**Q8_0 KV changes the arithmetic**, and the probe sends a 4-token prompt
(`def fibonacci(n):`). It barely touches the cache Q8 quantizes. It reported
"identical", and that was not evidence.

Re-run at real depth, same greedy settings (`temperature 0, top_k 1, seed 42`):

```text
prompt_n = 46 557
F16  hash 1A4F7C9924198E8A
Q8   hash 05C38B387571F755
common prefix: 1 character of 778
```

Completely different continuations. **Rule this establishes: an equivalence probe
must exercise the thing being changed.**

Divergence is not damage, though — a long-context summary has many valid
continuations. So it had to be measured on tasks.

---

## 2. Corpus v1 — retrieval at depth

Six execution-verified tasks over a shared **~44K-token repository prefix**. Each
answer depends on an arbitrary constant planted at a known depth — `MAX_RETRIES = 7`,
`TIMEOUT_MS = 8700`, `CHECKSUM_FIELD = "drain_token"` — values no prior can supply.
A model answering from memory, or one whose attention over a quantized cache has
degraded, writes code that fails the assertions.

The shared prefix means `cache_prompt` pays the deep prefill once and every later
task reuses it, which is also how an agent behaves.

| | F16 KV | **Q8_0 KV** |
|---|---|---|
| pass rate at 64K | 100 % (18/18) | **100 % (18/18)** |
| verified tasks / hour | 51.8 | **57.4** (+10.7 %) |
| warm turn, median | 51.2 s | **48.4 s** |
| cold prefill | 349.1 s | **321.0 s** |

Identical quality, ~11 % more throughput. **But both arms ceilinged**, which bounds
the damage rather than measuring it.

---

## 3. Corpus v2 — built specifically to break the ceiling

Ten tasks, four kinds of difficulty, each aimed at a different failure mode:

- **Confusable neighbours** — every planted shard has decoys whose IDs differ by a
  digit or a transposition (`0203` / `0230` / `2003`), with near-miss constants and
  an upper-cased field name, placed **immediately before** the real block so a
  forward scan meets the wrong one first.
- **Multi-hop** — `DEPENDS_ON` chains needing two retrievals: 2941 → 1508 → 417 → 203.
- **Aggregation** — sum `MAX_RETRIES` across all four authoritative shards; missing
  any one fails.
- **Depth** — the last planted shard sits at **95 %** of the prefix.

### Result — same boot, both arms

The first Q8 arm ran before a machine restart. Comparing a post-restart F16 arm
against it would be the cross-boot comparison this project's own methodology
forbids, so **both arms were re-run in one boot** (`v2c-64k-f16`, `v2c-64k-q8`).

| | F16 KV | **Q8_0 KV** |
|---|---|---|
| pass rate at 64K | **100 % (30/30)** | **100 % (30/30)** |
| warm turn, median | 55.7 s | **47.2 s** (−15.3 %) |
| total wall | 1 859.9 s | **1 706.9 s** |
| cold prefill | 80.2 s\* | 246.6 s |

\* Not a fair cold comparison: the F16 arm inherited a warm prefix from an
aborted earlier run on the same server. **The warm median is the number to read** —
it is measured over 29 turns per arm, and the shared-prefix design means warm
turns are what an agent actually pays.

Every task passed 3/3 on **both** arms, including the 95 %-depth retrieval, both
two-hop chains, all three aggregations, and the distractor-rejection task.

An earlier attempt at this arm (`v2b-64k-f16`) was stopped at 3/30 and then
**retracted** — see §7.

---

## 4. What the evidence supports

Three independent measurements now agree:

| measurement | F16 | Q8_0 |
|---|---|---|
| v1 corpus, 6 tasks × 3 | 18/18 | 18/18, +10.7 % tasks/hr |
| v2 corpus, pre-restart | — | 30/30 |
| v2 corpus, **same boot, both arms** | 30/30 | 30/30, warm turns −15.3 % |

Across **78 completed Q8 samples at 64K with zero failures**, Q8_0 KV shows
**no measurable retrieval degradation at 64K** while buying ~15–17 % and halving KV.

Combined with the 16K measurement, the recommendation is depth-conditional and
evidenced on both sides:

```text
16K-32K   F16 KV     Q8 measurably worse (86.7% vs 90.0%) and no faster
64K       Q8_0 KV    identical quality, ~15% faster warm turns
128K      Q8_0 KV    identical quality, ~28% faster warm turns
```

### What it does not support

- **Not a claim that Q8 is exactly as good.** Both corpora ceilinged on this model,
  so a 2–3 % regression stays below resolution. Two independent designs failing to
  find a difference is evidence of *absence of a large effect*, not proof of zero.
- **128K used 10 samples per arm, not 30** (§4.1), so its resolution is coarser
  than the 64K result.
- **256K remains unmeasured for quality** — it pages the host, so there is nothing
  to measure.

### A diminishing return worth naming

Past this point the exercise becomes designing a task **the model itself** fails,
so that Q8 can be observed failing it more often. That measures corpus difficulty,
not KV precision. The useful remaining axis was **depth**, and it has now been
measured — see §4.1.

### 4.1 128K — the last unmeasured axis

Running the 64K corpus on a 128K-configured server would only have measured a 44K
retrieval against a different layer split. To exercise the window the corpus was
scaled to **1 550 blocks → a 114 406-token prompt**, with the planted shards still
placed by percentage so the deepest sits at ~95 % of a 114K context.

| | F16 KV | **Q8_0 KV** |
|---|---|---|
| pass rate at 128K | **100 % (10/10)** | **100 % (10/10)** |
| warm turn, median | 144.5 s | **104.0 s** (−28 %) |
| total wall | 1 993.3 s | **1 555.1 s** (−22 %) |
| cold prefill | 715.5 s | 701.1 s |

`--attempts 1` (10 samples per arm) because decode at 128K is ~2.3 tok/s; 30
samples per arm would have taken hours. Stated so the sample size is not mistaken
for the 30 used at 64K.

**Q8_0's advantage grows with depth** — ~15 % faster warm turns at 64K, ~28 % at
128K — which follows from KV being a larger share of the memory budget the deeper
you go. Quality is identical at both depths.

Host pressure at 128K stayed well clear of the stop condition: RAM free 4.22 GB,
pagefile 1.27 GB, against the 0.63 GB / 10.11 GB that stopped the 256K run.

---

## 5. Two instrument bugs the corpus tests caught before use

Both would have produced a confident verdict from a broken instrument. Both were
caught by tests written before the corpus was trusted
(`bench\tests\test_harness.py`, 28 tests).

- **Duplicate planted class.** `Handler0017` was emitted twice — once as a routine
  block at index 17, once as the planted block — so "the class for shard 17" had
  two contradictory answers in context and the task measured nothing.
- **Corpus overflowed the window.** The size test asserted only a lower bound, so a
  **112K-token** corpus passed and then failed every request with HTTP 400 against
  a 64K window: **0/18 in four seconds**. Read naively that says "the model cannot
  do deep context at all." Both bounds are asserted now — and the same test then
  caught v2 landing at 19.5K tokens because its blocks are shorter.

---

## 6. Verdict

**Use Q8_0 KV at 64K and above.** Identical task quality at both depths measured,
~15 % faster warm turns at 64K and ~28 % at 128K, KV halved. At 16K keep F16 — Q8
is measurably worse there (86.7 % vs 90.0 %) because there is no KV to reclaim and
only the cost remains.

Scripts: `production-q4-tuned.ps1` (16K, F16) · `production-q4-deep.ps1` (64K, Q8_0).

---

## 7. A wrong call, recorded

The first re-run attempt (`v2b-64k-f16`) was stopped at 3/30 after its third sample
took **1 057 s**. I read that as a decode collapse, and confirmed it with a probe
that returned **11.21 tok/s** — far above the 4.37 tok/s expected — which I took as
further evidence something was wrong.

**Both readings were mistakes.** The probe sent a 4-token prompt with
`cache_prompt=false`, measuring decode over an *empty* cache, while 4.37 tok/s was
measured with the window 80 % full. They measure different things — the exact error
[04-MEASUREMENT-METHODOLOGY.md](04-MEASUREMENT-METHODOLOGY.md) §3 warns about, made
by the person who wrote the warning.

The real explanation was ordinary variance: that attempt hit `max_tokens 1536` while
its predecessor answered in 212 tokens. At depth that is roughly 1 000 s of
generation. The re-run completed 30/30 without incident.

The retraction is recorded in-band in `results\deep-quality.jsonl` on the
`v2b-64k-f16` marker row (`retracted: true`), so the partial cannot be mistaken for
a result later.
