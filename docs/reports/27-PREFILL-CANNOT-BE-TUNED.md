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
