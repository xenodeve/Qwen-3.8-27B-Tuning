# 04 — Context depth: what is resident where, and how fast

> 🔴 **Every number on this page was measured at `reasoning_effort: xhigh` with
> an unlimited thinking budget — the cold-start row below asks *about* the flag but was itself taken at that default, so it is covered rather than excepted.** That is the model's chat-template
> default — the client sends no effort field, and **no `worker-*.ps1` profile and
> nothing in `bench/` has ever set the flag** (established 2026-08-24 from a boot
> log: [`05-runtime-flags.md`](05-runtime-flags.md)).
> Artificial Analysis prices this model's `medium` **one point** below `xhigh` on
> the agentic axis and `low` **six** below that
> ([`researchs/artificial-analysis`](../researchs/artificial-analysis/README.md)),
> so **effort is a live confound here, not a settled background condition.**

**The model is not the limit.** The loader reports `n_ctx_train = 262144` with no
scaling engaged at 163,840 — no YaRN, no rope extension. Every depth this
project has tried is inside the native window. **12 GB of VRAM is the only
ceiling.**

## The ladder, `q4_0` KV

| artifact | 131,072 | 147,456 | 163,840 | 196,608 | 229,376 |
|---|---|---|---|---|---|
| V3 `UD-IQ1_S` | `65+0` | — | — | **`65+0`** | `57+8` |
| V3 `UD-IQ1_M` | `65+0` | — | **`65+0`** | `60+5` | — |
| V3 `UD-IQ2_XXS` | `65+0` | **`65+0`** | `62+3` → `65+0` with a lowered `--fit-target` | — | — |
| `AD-IQ1_M` | `65+1` | — | — | — | — |
| V3 `UD-Q2_K_XL` | `54+12` | — | — | — | — |
| pre-V3 `UD-IQ2_XXS` | `58+7` | — | — | — | — |

> **Report 21 walks this ladder in steps of 32,768 and records the deepest rung
> that loaded.** That is not a ceiling. `v3-iq2xxs` was recorded at 131,072
> because 147,456 was never tried — and it holds `65+0` there. **Read every
> figure as "at least this deep".**

## Throughput at depth — `v3-iq2xxs`, `q4_0` KV, `--fixed-text`

| depth | split | baseline | `ngram-mod` | KV size |
|---|---|---|---|---|
| 16,384 | `65+0` | 40.1–41.8 | 85–88 (`map-k`: 93.7–98.9) | 288 MiB |
| 131,072 | `65+0` | 23.0–26.5 | **73.4–81.5** | 2,304 MiB |
| **147,456** | `65+0` | 24.9–25.6 | **108.3–109.2** | 2,592 MiB |
| 163,840 | `62+3` | 18.8–19.4 | 36.2–38.7 | 2,880 MiB |

**The 147,456 row is the fastest number this project has measured at any depth**,
and part of it is an artefact: the timed prompt gets *more repetitive* as it gets
longer, so the n-gram hit rate rises with depth. The `65+0` residency at that
depth is real and independent of the prompt; the 109 tok/s is not a clean read.
See `CORRECTIONS.md` §2.

**The drop from 147,456 to 163,840 is the residency cliff**, not the KV: 288 MiB
more cache costs three layers, and three layers cost ~22 % of decode.

## Throughput at depth — other artifacts

| artifact | depth | split | baseline | `ngram-mod` | acceptance |
|---|---|---|---|---|---|
| V3 `UD-IQ1_M` | 163,840 | `65+0` | 24.4–24.6 | **45.8–47.3** | 100 % |
| V3 `UD-IQ1_M` | 196,608 | `60+5` | 8.8–8.9 | — | — |
| V3 `UD-IQ1_M` | 196,608 | `65+0` via `-ot ssm` | 18.3–19.8 | **21.9–28.1** | 100 % |
| V3 `UD-IQ1_S` | 196,608 | `65+0` | 22.3–23.2 | 24.5–26.4 | **37.5 %** |
| `AD-IQ1_M` | 131,072 | `65+1` | **6.08** | — | — |

**Two rows worth staring at.**

`AD-IQ1_M` at `65+1` decodes at **6.08 tok/s** — one CPU layer against a resident
26.50. The cliff is far steeper at depth than the 33+32 → 61+4 → 65+0 ladder at
16 K suggested.

`v3-iq1s` at 196,608 gets only **+12 %** from n-gram where every other resident
arm gets +90 % or more, and its acceptance is 37.5 %. It is also the artifact
that scores 0 of 12 on the corpus.

## Prefill, which speculation cannot touch

| depth | prefill |
|---|---|
| 16,384 | ~10 s |
| 131,072 | ~114 s |
| 147,456 | ~122 s |
| 163,840 | ~150 s |
| 196,608 | ~190 s |

Roughly linear at 750–860 tok/s while resident. It collapses to **8.56 tok/s**
with `-ot ffn`, and to **240 tok/s** at `65+1`.

## What a REAL task consumes — tested 2026-08-22, and it overturns "what depth is worth"

**The first measurement of this in the project.** `bench/real_task_bench.py`,
real open GitHub issues, throwaway clones, scored by each repo's own verify
command. Raw `results/real-task-bench.jsonl`,
[report 31 §6](../reports/31-SESSION-RECORD-2026-08-22.md).

Context high-water read from **`n_tokens` on the `slot release` line**, maximum
over the task — not `prompt eval`, which reports only what survived cache reuse
and mis-sized three worker profiles (`CORRECTIONS.md` §15, §17).

| window served | high-water range | saturated? |
|---:|---|---|
| 32,768 | 32,767 – 41,377 | **yes, all five** |
| 65,536 | 54,324 – 72,056 | **four of five** |
| 98,304 | 56,861 – 88,668 | no |

**Every time the window grew, the tasks used it.** At 32,768 the server log
carried `exceeds the available context size (32768 tokens)` six times and
`truncated = 1` four times. The numbers at the two smaller windows are ceilings
the operator set, not requirements of the work.

### What this overturns

The section above asks what depth is worth and answers from the OpenCode
corpus's longest conversation, **13,741 tokens**. That is a *lean* harness
measurement and it is an order of magnitude below what a real agent task on a
real repository consumes.

`docs/plans/06` was written on the hypothesis that tasks peak near 40,000 and
that serving 98,304 reserves 1.5–2 GB for nothing. **The 40,000 figure came from
the run that saturated at 32,768.** It measured the ceiling, not the task.

> **`worker-iq2s-quality.ps1` at 98,304 is the minimum sensible window, not
> headroom.** Any plan to shrink the window to buy a higher quantisation rung
> has to be rewritten.

### ~~And the drafter still fits there~~ — it fits and does not work

At ctx 98,304 with `--spec-draft-n-max 4` and the DFlash2 drafter loaded, the
target is **`65+0` resident with 254 MiB free**. KV 1,728 MiB, recurrent state
748.12 MiB, drafter KV 45.00 MiB. This was predicted not to fit and it does.

> 🔴 **Fitting is not the test, and 254 MiB is the middle of the band that
> fails.** Measured 2026-08-23 over six paired rounds: arms with the sidecar at
> this depth finish with **45–376 MiB free**, time out **3 times in 12**, and
> spread **146×** on identical flags — 0.64 to 93.29 tok/s. Arms without it sit
> at 769–2,117 MiB and land within 4 % every time. **Residency reads healthy in
> both cases**, which is why this row was written as good news.
> [`CORRECTIONS.md` §26](../reports/CORRECTIONS.md).

🔴 **The true requirement above 98,304 is unknown.** Three windows, three
saturations; 98,304 was the first that held, and one task reached 88,668 of it.

🔴 **RETRACTED — [`CORRECTIONS.md` §24](../reports/CORRECTIONS.md).** The zero
diffs measured **where the harness looked, not what the worker did.** OpenCode
attaches to a server carrying the project root it first started with, so with
`cwd=` alone the worker edited **`C:\AI` itself** while `git diff` in the clone
stayed empty. Reproduced 2026-08-23: `cwd=` alone → `EDIT_NO_DIFF`, 0 diff
bytes, live tree modified; **with `--dir` → `EDITED`, 251 diff bytes, 32.8 s.**
Fixed in both drivers, pinned by `bench/tests/test_worker_workdir.py`.

**What survives:** the wall-clock times and the context high-water figures —
those came from the process and the server, not from the diff.

**~~A second cause is independent of it and still open: decode at this window
is 2.8–5.0 tok/s~~ — RETRACTED [`CORRECTIONS.md` §26](../reports/CORRECTIONS.md).**
That range belongs to the DFlash2 arms, not to the window: with `ngram-mod`
alone the same depth returns **96.92 tok/s over 6 of 6 rounds**. **So the
directory fault is the only established explanation for the zero diffs**, and
the next real-task run has one variable rather than two.

## ~~The window we serve is the one that does not work~~ — the DRAFTER does not work there

> 🔴 **RETRACTED 2026-08-23 — [`CORRECTIONS.md` §26](../reports/CORRECTIONS.md).**
> The section below is kept because its measurements are real; its **conclusion
> is not**. Every one of the sixteen rows loaded the DFlash2 sidecar, so depth
> and drafter were never separated. Re-measured over six paired rounds at the
> same ctx and corpus with the arms alternated:
>
> | arm | ok | timed out | median tok/s | free MiB after load |
> |---|---:|---:|---:|---|
> | `none` | 6/6 | 0 | 33.69 | 800–1,935 |
> | **`ngram-mod` — the decoder all four profiles run** | **6/6** | **0** | **96.92** | 769–2,117 |
> | `dflash2` | 5/6 | 1 | 49.31 | **45–376** |
> | `dflash2+ngram` | 4/6 | 2 | 5.66 | **153–240** |
>
> **`ngram-mod` at ctx 98,304 is faster than the 75.2 median recorded at
> 16,384.** The window is not the problem. `results/decoders-98304.jsonl`.
>
> **Read the artifact with the rate.** These rows are **`UD-IQ2_XXS` at 98,304**,
> a pairing **no profile serves** — `worker-iq2xxs-deep` runs that artifact at
> 131,072 and `worker-iq2s-quality` runs 98,304 on the 1.1 GB larger
> `UD-IQ2_S`. The decoder verdict transfers, since all four run `ngram-mod`; the
> absolute rate does not.

`results/sweep-ngram-nmatch-98304.jsonl`, 16 rows, deep corpus, four rounds —
**all sixteen with `--spec-type draft-dflash,ngram-mod`.**

| n_ctx | usable rows | decode tok/s | cold prefill tok/s |
|---:|---|---:|---:|
| 16,384 | 42 / 42 | median **75.2** | **1,129** (89 boots) |
| 65,536 | 12 / 12 | median **52.1** | **924** (18 boots) |
| **98,304, DFlash2 loaded** | **3 / 16** | median **4.2** (2.8-5.0) | **74.3** (62-87) |
| **98,304, `ngram-mod` alone** | **6 / 6** | median **96.92** | — |

**Thirteen of sixteen measurements timed out** against a 26.8-minute budget, and
the three that finished decoded at 2.8, 5.0 and 4.2 tok/s. Every arm was `65+0`
with acceptance still 59-77 %. The line that used to stand here — *"so neither
residency nor speculation explains it"* — was wrong: **speculation explains it,
and the sweep could not see that because it never varied.**

A 43,162-token prefill takes about **9.7 minutes with the drafter loaded** — and
that figure, like the decode one, belongs to that configuration rather than to
the window.

> **What the telemetry actually says, corrected 2026-08-23.** `split: 65+0`
> reads as healthy on every row and is not the tell. The tell was in
> `results/gpu-trace-98304.jsonl` (1,094 samples): **32 MiB free at minimum,
> 246 median, 100 % GPU utilisation, and 76 W on a ~220 W card**.
>
> This page used to call that *"the memory-bound signature"* and reason that LLM
> decode is memory-bound by nature, so it might not be pathology. **That reading
> is wrong, and the same trace refutes it:** `utilization_memory` has a median
> of **4 %**, and 2,615 of 2,699 samples sit at ≥ 90 % GPU with the memory
> controller idle. A memory-bound decode shows high *memory* utilisation. This
> is a card spinning at full clock and one-third power — waiting, not working.
>
> Sampled live on 2026-08-23 during a slow `dflash2+ngram` round:
> `free 196 MiB · util_gpu 100 % · util_memory 3 % · 2820 MHz · 70.18 W · 57 °C`
> — matching the old trace in every column, and now with the arm identified.
>
> **The mechanism, as far as it goes.** With a model-based drafter `n_rs_seq`
> is 4, so the server writes `created speculative checkpoint … size =
> 149.626 MiB` — one full recurrent-state plane — every few generated tokens.
> With `ngram-mod` alone `n_rs_seq` is 0 and no such checkpoint exists. In slow
> rounds the gap between checkpoints reaches **30.41 s** against a median 2.35 s
> in fast ones: a stall, not uniform slowness.
>
> **Still unexplained:** why some drafter rounds escape — 93.29 tok/s at 240 MiB
> free, while another managed 1.46 at 153 MiB. There is no clean threshold, only
> a band where the outcome is unreliable. The sysmem-fallback experiment is
> still worth running, but as a question about **that band**, not about depth.

⚠️ **A label caveat that applies to every depth row on this page.** A run
labelled "ctx N" fed a prompt of roughly **40 % of N** - 6,621 tokens at
"16,384", 28,122 at "65,536", 43,162 at "98,304". `--ctx` still sets the
allocation, so every residency and VRAM finding is unaffected; the **depth
labels** are what shift. Directions hold, because context did grow 4.2x between
the first two.

**The reason is `dflash2_arena.py:478`** - `filler(int(ctx * 0.5), regime)`,
which asks for half the window by design so the generation has room. It is
**not** that chars/token was mis-estimated: measured against the server's own
counts it is **~3.4**, against the 3 the harness assumes. The "7.0-7.4" this
page carried until 2026-08-23 dropped that 0.5 and is retracted in
[`CORRECTIONS.md` §25](../reports/CORRECTIONS.md).

## What depth is worth

A worker carries a fixed prefix — measured at **39,762–40,648 tokens** for a
Claude Code instance, four calls. So the working room is the window minus that:

```text
  131,072  ->  ~91,000 usable
  147,456  ->  ~107,000 usable
  163,840  ->  ~124,000 usable
```

**16,384 more tokens of window is 18 % more working room, not 12 %** — which is
the argument for chasing the depth even when the tok/s falls.

*Raw: `results/ctx-ceiling-q38.jsonl`, `results/kv-ngram-fixed.jsonl`,
`results/kv-deep-147k.jsonl`, `results/kv-deep-160k.jsonl`,
`results/kv-deep-192k.jsonl`, `results/kv-vram-160k.jsonl`. Reports 19, 21, 24.*

## Not tested

- **229,376 and 262,144 on any artifact that has a usable corpus.** Only
  `UD-IQ1_S` reaches that far and it produces nothing.
- **`UD-IQ2_S` at any depth.** The rung between the deep candidate and the
  quality default.
- **Anything past 147,456 with the desktop's VRAM freed.**

## UD-IQ2_S at 131,072 — tested 2026-08-21

| question | answer | evidence |
|---|---|---|
| Does `UD-IQ2_S` hold `65+0` at 131,072? | **Yes, with `--fit-target 192`.** 23.21 and 23.92 tok/s over two reversed rounds | report 25 |
| What does the default reserve cost? | `60+5`, 8.16-10.79 tok/s — 2.5-3x slower | report 25 |
| Do smaller compute buffers buy layers? | **No.** `-ub 128`, with `-b` at 1024 or 2048, still loads `60+5`; four rows agree to 0.7 % | report 25, phase 2 |
| What does the extra depth cost against 98,304? | ~11 % of decode (26.61 -> 23.2-23.9), inside the 13.6 % drift floor | report 25 |
| Against profile A at the same depth? | About half: 45 tok/s on `UD-IQ2_XXS` vs 23.2-23.9 here | report 25 |
| Does free VRAM at settle predict the collapse? | **No.** 233 MiB ran 4.3x faster than 291 MiB | `CORRECTIONS.md` 14 |
| Does the display move to the iGPU help? | **Not tested.** Named by every reviewer as the largest lever | — |

Raw: `qwen38-tuning/results/iq2s-131072-residency.jsonl`.

## Cold start — tested 2026-08-21, through Qwen Code itself

| question | answer | evidence |
|---|---|---|
| How big is Qwen Code's request? | **54,499 tokens.** An earlier entry said 16,796; that was what remained to prefill after cache reuse | `CORRECTIONS.md` 15 |
| What is the cold start? | Prefill. 53.4 s on the first run of a fresh server, 41.4 s on every run after | report 25 |
| Does the prefix repeat between runs? | **Yes, exactly.** Two runs captured through a proxy are identical for all 207,243 characters | report 25 |
| Does the cache hit across invocations? | **Barely.** About 12,700 of 54,499 tokens are reused; the rest is re-prefilled every time | report 25 |
| Does `--cache-ram -1` help? | **No, it is a regression.** Reuse drops to zero, measured twice | report 25 |
| Does `--cache-reuse 256` help? | **No.** Full 54,499 prefilled every run | report 25 |
| Does `-np 2` fix the slot clobbering? | **Untestable here.** At 131,072 it collapsed to 113.9 tok/s at 296 MiB free | report 25 |
| Does a larger `-ub` speed prefill? | **No.** 1,134-1,168 tok/s across `-ub` 256/512/1024, a 2.9 % span | report 25 |
| Does `reasoning_effort` affect cold start? | **No.** The template swaps one instruction sentence | the model's chat template |
| What is left? | The display on the Intel UHD 770. It returns the 1.4-2.0 GB that `-np 2` needs. **Untested** | — |

Raw: `qwen38-tuning/results/iq2s-prefill-microbatch.jsonl`,
`qwen38-tuning/results/cold-start.jsonl`.

## Cold start — ELIMINATED 2026-08-21

| question | answer | evidence |
|---|---|---|
| Is the cold start gone? | **Yes, two ways.** `-np 2 -sps 0.95` gives a full cache hit with every Qwen Code feature ON; turning `memory.enableManagedAutoMemory` off also works but costs the feature | report 26, `CORRECTIONS.md` 16 |
| Why did `-np 2` fail the first time? | `--slot-prompt-similarity` defaults to 0.10, so both prompts landed on the same slot | `CORRECTIONS.md` 16 |
| What caused it? | `memory.enableManagedAutoMemory` — a second subagent whose 195,929-char prompt evicted the main prefix from the slot | report 26 |
| Is the server's cache at fault? | **No.** One request replayed three times: 53.9 s, 0.4 s, 0.4 s | report 26 |
| Where must the warm-up run? | In the working directory. Qwen Code's prompt embeds it | report 26 |
| What is the cost? | Qwen Code no longer updates its own memories. Three settings were turned off together and are not isolated | report 26 |


## Qwen Code prompt size — tested 2026-08-21

| question | answer | evidence |
|---|---|---|
| How big is the prompt, and what is in it? | 54,711 tokens; `--safe-mode` leaves **14,399**, so the customization layer is 40,312 — 74 % | report 26 |
| Does the working directory matter? | **No.** 54,095 / 54,483 / 54,073 across three directories, a 410-token spread | report 26 |
| Is it the skill catalogue? | **No.** Disabling every skill level made the prompt *larger*, 57,526 | report 26 |
| Is it managed memory? | **Partly.** 6,103 tokens, and it collapses three calls into one | report 26 |
| What holds the other ~34,000? | **Not isolated.** Tool schemas are the candidate | — |
| How should prompt size be measured? | From `slot release ... n_tokens`. `prompt eval` reports what was left after cache reuse and understates the prompt | report 26 |


## Where Qwen Code's 54,711 tokens go — tested 2026-08-21

| question | answer | evidence |
|---|---|---|
| Is it the tool schemas? | **No.** 5,416 tokens, identical in baseline and safe mode, 8 tools either way | report 26 |
| Is it the system prompt? | **No.** 8,766 vs 5,372, a 3,394 difference | report 26 |
| What is it then? | **The skill catalogue: 38,064 tokens, 70 % of the prompt**, injected as a *user* message block | report 26 |
| How many skills? | **352 advertised, 344 user-scope**, ~110 tokens each. Safe mode advertises 9 | report 26 |
| How many calls per invocation? | 4 baseline against 1 in safe mode, so the catalogue is paid three times: **153,621 tokens against 14,064** | report 26 |
| Do the two modes share a cache prefix? | **70 tokens.** They cannot warm each other | report 26 |
| How was it measured? | Requests captured through a proxy, rendered by the server's `/apply-template`, counted by its `/tokenize` | report 26 |


## Removing the catalogue without losing the skills — tested 2026-08-21

| question | answer | evidence |
|---|---|---|
| Does `disable-model-invocation: true` remove a skill from the catalogue? | **Yes, exactly.** 18 files flagged, 18 fewer advertised | report 26 |
| What does one skill cost? | **~87 tokens.** 38,064 -> 36,496 for 18 skills | report 26 |
| What would flagging all 344 user-scope skills save? | roughly **30,000 tokens**, keeping every file, MCP server, memory feature and extension | report 26 |
| Is a flagged skill still invocable? | **Not tested.** That is the half that matters | — |
| How many skills are installed? | **257** in `~/.qwen/skills` | report 26 |
| Why is the same catalogue free on the gateway? | **Unexplained.** Prefix cache, prefill throughput, or a different call path — none tested | report 26 |


## The two open questions, closed — tested 2026-08-21

| question | answer | evidence |
|---|---|---|
| Is a flagged skill still invocable by name? | **No.** The registry reports it *not found*; advertised, the same call is only blocked by permissions | report 26 |
| Does the gateway receive a smaller payload? | **No.** 54,478 and 57,700 tokens against the local 54,485 and 56,277, five calls either side | report 26 |
| Does the gateway report a prefix cache? | **No.** `cached_tokens` is absent from every response | report 26 |
| Then why is it fast? | **Prefill throughput.** 54,478 tokens at 4.97 s to first byte is ~11,000 tok/s against our 900; the second big call reuses the prefix and drops to 1.41 s | report 26 |
| End to end for the same `hi` | **19.4 s on the gateway against ~171 s of local prefill** | report 26 |

