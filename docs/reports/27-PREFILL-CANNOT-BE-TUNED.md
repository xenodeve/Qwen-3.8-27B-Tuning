# 27 — Prefill cannot be tuned on this machine

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
