# 04 — Context depth: what is resident where, and how fast

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
| Is the cold start gone? | **Yes.** 4 tokens, 0.1 s prefill, 4.4 s wall on the developer's first turn after `warm-cache.ps1` | report 26 |
| What caused it? | `memory.enableManagedAutoMemory` — a second subagent whose 195,929-char prompt evicted the main prefix from the slot | report 26 |
| Is the server's cache at fault? | **No.** One request replayed three times: 53.9 s, 0.4 s, 0.4 s | report 26 |
| Where must the warm-up run? | In the working directory. Qwen Code's prompt embeds it | report 26 |
| What is the cost? | Qwen Code no longer updates its own memories. Three settings were turned off together and are not isolated | report 26 |

