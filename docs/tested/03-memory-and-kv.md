# 03 — Memory: KV, tensor placement, reserves, batch

The binding constraint on this machine is 12 GB, and everything on this page is
an attempt to spend it better.

## KV cache type

| type | verdict | evidence |
|---|---|---|
| **`q4_0` (both K and V)** | **the settled choice** | buys residency at every depth; no measurable quality cost |
| `f16` | correct, too large | the default; loses layers at depth |
| `q8_0` | works, larger than `q4_0` | no reason to prefer it here |
| `q5_1` | works | between the two, no advantage |
| **`-ctk q8_0 -ctv q4_0` (mixed)** | **unusable** | **no fast kernel in b10472** — prefill **29× slower**, cache **44 % larger**, one arm hung 65 minutes at 128 K |

The mixed-KV result came from external research predicting ~25 % VRAM saved. It
is the clearest case of the folder rule: the mechanism was plausible, the number
was invented, and the build has no kernel for it.

*Raw: `results/kv-sweep.jsonl`, `results/kv-kernel-screen.jsonl`,
`results/kv-128k-q38.jsonl`. Reports 11, 18, 22.*

## `-ot` — moving tensors to the CPU by name

| slice | frees | effect |
|---|---:|---|
| `ffn_` on 1 block (`blk.64`) | ~644 MiB | **prefill 240.6 → 8.56 tok/s.** 28× slower. A 93 K prompt would take three hours |
| `ffn_` on 10 blocks | 1,234–1,407 MiB | −11 % at depth, **−61 %** at 16 K |
| `ssm_` on 4 blocks | ~168 MiB | restores `65+0`; **+16.38 %** with speculation off |
| `ssm_` on 10 blocks | ~168 MiB | restores `65+0`; **+10.90 %**, under the drift floor |

**With speculation on, the ssm slice behaves three different ways** — 4 %
acceptance, no drafts at all, or 100 % — depending on artifact and depth. See
[`02-decoders.md`](02-decoders.md). It is not a lever that can be recommended
without naming the exact configuration.

**`-ot` also changes the greedy hash** — CPU and GPU floats differ. Any arm using
it is compared on speed only.

*Raw: `results/kv-ot-iq1m.jsonl`, `results/kv-deep-160k.jsonl`,
`results/kv-deep-192k.jsonl`, `results/kv-depth-levers.jsonl`.*

## `--fit-target` — the reserve nobody had questioned

The harness has passed **768 MiB** since the first sweep. At 163,840 on
`v3-iq2xxs` the artifact sits at `62+3` and needs roughly 576 MiB to reach
`65+0`, so the reserve is larger than the shortfall.

**First measurement, 2026-08-21 (V1, round 1 of 2):**

```text
  --fit-target 768 (default)   62+3    32.87 tok/s
  --fit-target 384             65+0    21.62
  --fit-target 192             65+0    31.12
  -ub 128                      63+2    29.70
  --fit-target 192 + -ub 128   65+0    28.33
```

**Lowering the reserve does buy residency** — `65+0` at a depth that otherwise
loses three layers, with no weights on the CPU and no `-ot`. That is the thing
V1 was built to find.

**It has not yet bought speed**, which was the point of finding it. Round 1 shows
every resident arm at or below the non-resident baseline. **Round 2 and the
acceptance column are not in yet**; do not read this table as settled.

*Raw: `results/kv-vram-160k.jsonl`. Report 24 §V1 when complete.*

## `--ctx-checkpoints`

External research: frees ~900 MiB. Measured: **10–16 MiB**, and no change to any
residency ladder. Tested at 8 and at the default.

*Raw: `results/ctx-ceiling-q38.jsonl` rows tagged `ckpt8`. Reports 18, 21.*

## Batch and micro-batch

| flag | tried | result |
|---|---|---|
| `-b 2048 -ub 256` | the standing default | — |
| `-ub 128` | 2026-08-21 | `63+2` at 163,840 — frees some, not enough alone |
| `-b 1024 -ub 128` | queued (V2) | — |

Smaller buffers trade prefill for VRAM. Unlike `-ot`, that is a trade whose price
is visible in the `pp` column.

## `-np` (parallel slots)

**Harmful, not inert.** `-np N` divides the context between slots: asking for
16,384 with two slots gives each 8,192, and an 11,663-token probe returns HTTP
400. At the 131,072 target two slots would give each 65,536 — abandoning the
goal. It also killed the queue step it was in.

*Report 20 §4.*

## The desktop's VRAM

Roughly **1,650–2,200 MiB** is held by the Windows desktop and whatever is open.
It moves boot to boot, which is the origin of the 13.6 % drift floor.

**Never tested.** Freeing it is the largest untouched lever on this machine and
it needs no code — it needs the display on the iGPU, or a session with nothing
else running. Open since 2026-08-20.
