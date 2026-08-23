# 27 — Prefill cannot be tuned on this machine

> 🔴 **Scoped to ctx ≤ 32,768 — and the title does not hold at depth.**
> Cold prefill is **1,129 tok/s at 16,384, 924 at 65,536 and 74.3 at
> 98,304** — a 15× collapse at the window actually served, with every layer
> resident (`65/65`).
>
> ⚠️ **But read the 74.3 with its configuration attached.** It was measured
> with the DFlash2 sidecar loaded, as was every row at that depth. The
> companion claim that *decode collapses with it to 2.8–5.0 tok/s* is
> **retracted — [CORRECTIONS §26](CORRECTIONS.md), 2026-08-23**: with
> `ngram-mod` alone, which is what all four worker profiles serve, decode at
> ctx 98,304 is **96.92 tok/s median over 6 of 6 rounds**. The prefill figure
> above has not been re-measured without the drafter and should be treated as
> a drafter-loaded number until it is.
>
> Every number below stands at the depth it was taken.
> See [32 §3.1](32-BENCHMARK-STATUS-BRIEF.md).

**Measured 2026-08-21.** Raw: `qwen38-tuning/results/prefill-kv-type.jsonl`.
Two rounds, arm order reversed, `UD-IQ2_S` at 32,768 with the default reserve —
the regime where phase-2 rows repeated to 0.7 %.

## The question

A harness's cold start is prefill, and prefill on this card runs at roughly
900 tok/s. The gateway that serves the same payload does it at roughly 11,000
(report 26). Can the gap be closed with settings?

## The KV cache type, the only untried axis

| arm | prefill r1 / r2 | decode r1 / r2 | KV | free MiB r1 / r2 |
|---|---|---|---|---|
| `q4_0` | 714.0 / **882.5** | 36.67 / 40.62 | 576 | 1,696 / 1,401 |
| `q8_0` | **984.0** / 871.1 | 39.00 / 39.33 | 1,088 | 1,056 / 1,294 |
| `f16` | 363.4 / 177.3 | 21.10 / 7.50 | 2,048 | 427 / 242 |
| `iq4_nl` | **abandoned** at 737 s, both rounds | — | — | — |

**`q4_0` and `q8_0` are indistinguishable.** Round 1 put `q8_0` 38 % ahead; round
2 put `q4_0` ahead. The spread *within* `q4_0` — 714 to 882.5, 24 % — is wider
than the gap *between* the arms, which is the 13.6 % drift floor behaving exactly
as documented. **A single round would have shipped a 38 % improvement that does
not exist.**

**`f16` cannot be measured cleanly here and should not be read as a slow kernel.**
Its 2,048 MiB of KV left 427 and 242 MiB free, so both rows sat in the collapse
regime. The result says f16 does not fit, not that it is slow.

**`iq4_nl` is unusable on this build.** The prefill never finished inside a
timeout sized from the floor rate, twice. Recorded as `ABANDONED`, not as a
number.

## What that closes

Every setting-level lever on prefill has now been measured and none of them move
it:

| lever | result | where |
|---|---|---|
| `-b` / `-ub` | 1,134–1,168 tok/s across a 4x change, a 2.9 % span | report 25 |
| KV cache type | no reliable difference between `q4_0` and `q8_0` | here |
| flash attention | already on |  |
| layer residency | already `65+0`; spilling costs 2.5–3x decode | report 25 |
| `--fit-target` | moves the split, not the rate | report 25 |
| CPU-MoE flags | not applicable, the model is dense |  |

**So prefill throughput is not a tuning problem on this hardware.** The two
levers that remain are not settings: send fewer tokens — which is what removing
the skill catalogue does, 10.9x on the work per invocation (report 26) — or use
a different class of card, which is what the gateway is.

## The smaller artifact does not prefill faster either

**Measured 2026-08-21.** Raw: `qwen38-tuning/results/artifact-prefill.jsonl`.
Same depth, same flags, two rounds with the order reversed. The most
reproducible pair of rows this project has recorded — 0.6 % and 1.1 % on prefill,
0.4 % and 0.3 % on decode.

| artifact | prefill r1 / r2 | decode r1 / r2 | free MiB |
|---|---|---|---|
| `UD-IQ2_S` 8.37 GB | 1,120.9 / 1,114.3 | 49.19 / 49.13 | 1,230 |
| `UD-IQ2_XXS` 7.27 GB | 1,133.3 / 1,132.6 | **77.36 / 77.57** | **2,286** |

**Prefill is identical.** A 1.6 % gap across 1.1 GB of weights, well inside the
noise floor. Bits per weight buys nothing on the axis that is the cold start.

**Decode is 57 % faster**, and the spare VRAM is real: **1,056 MiB**, which is
the gigabyte the question was about. That headroom is also the distance from the
collapse regime — the runs that fell to 92 and 242 MiB free this session had
none to give.

So the trade is exactly the one the residency work already implied, now with
numbers on both sides: **the gigabyte buys decode speed and safety margin, and it
does not buy a shorter cold start.** Whatever it costs in quality, it is not
paying for prefill.

**And the quality side has never been measured here.** The five points of vendor
top-1 between the two are quoted from a curve taken at 4,096 context; this
project's only real-work figure is profile A's 6 of 10 accepted tasks on
`UD-IQ2_XXS`, with no matching run on `UD-IQ2_S`. Choosing between them on
quality is a decision without evidence.

## What speculation costs in VRAM

**Measured 2026-08-21** at ctx 32,768 with the default reserve, reading settled
VRAM rather than estimating from file size.

| arm | VRAM used | free | cost |
|---|---|---|---|
| no speculation | 10,763 MiB | 1,235 | — |
| `--spec-type ngram-mod` (shipping) | 10,763 MiB | 1,235 | **0 MiB** |
| `--spec-type draft-mtp` + `-md` | 11,306 MiB | 692 | **564 MiB** |

**`ngram-mod` is free**, byte for byte identical to no speculation — it carries no
weights and never did. That is now measured rather than assumed.

**MTP costs 564 MiB**, and residency survives it: `offloaded 66/66 layers to GPU`
in the same run, so nothing was pushed to the CPU to make room and the figure is
the true additional cost, not a number hiding a spill.

**`draft-mtp` cannot run on the shipping artifact by itself.**

```text
  W llama_init_from_model: context type MTP requested but model doesn't contain MTP layers
  E common_speculative_init_result: failed to create MTP context
  E srv    load_model: failed to create MTP context
```

`UD-IQ2_S` has no MTP layers. The weights live in a separate 1.3 GB file,
`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, which must be passed with `-md`. On the GPU it
resolves to 564 MiB, not 1.3 GB.

**Whether it is worth 564 MiB is a different question, and this project already
answered it badly.** `draft-mtp` measured **+81 % at 16K and −71 % at 131,072**
on the same artifact. At the shipping 98,304 profile the machine settles with
about 400 MiB free — 564 MiB does not fit there at all without dropping layers,
and the depth where it does fit is the depth where it is a loss.
