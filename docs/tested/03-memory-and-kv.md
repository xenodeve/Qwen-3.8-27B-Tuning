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

## The recurrent state, and why `--spec-draft-n-max` is a VRAM knob

**Measured 2026-08-22 from `logs/dflash2-*.log` and `logs/ceil-*.log`.** This is
the largest single thing this project had never looked at.

Qwen3.8 is a hybrid: 48 Gated DeltaNet layers whose **recurrent state is a
separate allocation from the KV cache**, reported on its own line as
`llama_memory_recurrent: CUDA0 RS buffer size`.

### It does not scale with context

| ctx | KV buffer (q4_0) | **RS buffer** |
|---:|---:|---:|
| 32,768 | 576.00 MiB | **149.62 MiB** |
| 65,536 | 1152.00 MiB | **149.62 MiB** |
| 98,304 | 1728.00 MiB | **149.62 MiB** |
| 131,072 | 2304.00 MiB | **149.62 MiB** |

Flat to two decimals across a 4x range of depth. Deepening context does not
touch it.

### It scales with the DRAFT COUNT instead

At ctx 16,384 with `--spec-type draft-dflash` and `--spec-draft-n-max 4`, the RS
buffer is **748.12 MiB** — and `748.12 / 149.62 = 5.0000`.

Source confirms the mechanism. `common/common.h:390`:

```cpp
return needs_rs_seq ? draft.n_max : 0u;
```

`need_n_rs_seq()` returns `draft.n_max` for `DRAFT_MTP`, `DRAFT_EAGLE3`,
`DRAFT_DFLASH` and `DRAFT_DSPARK`; `common/common.cpp:1699` assigns it to
`cparams.n_rs_seq`. The allocation is one base copy plus one per draft position,
so the state has somewhere to roll back to when a draft is rejected.

> **RS buffer = 149.62 MiB x (1 + `--spec-draft-n-max`)** for this model.

| arm | `n-max` | RS buffer |
|---|---:|---:|
| `ngram-mod` alone (no model drafter) | — | **149.62 MiB** |
| `draft-dflash`, default | 3 | 598.5 MiB |
| `draft-dflash`, **what report 29 measured** | 4 | **748.12 MiB** |
| `draft-dflash`, at the `block_size - 1` clamp | 7 | **1,197 MiB** |

`ngram-mod` pays none of it — it is not a model drafter, so `need_n_rs_seq()`
returns 0. That is a second, previously unrecorded reason the n-gram family is
cheap here, on top of holding no weights.

### What this means for the drafter's measured cost

Report 29 recorded the DFlash2 drafter at **1,936 MiB resident** (free VRAM
2,376 without it against 440 with it). That number is now decomposed:

| | MiB |
|---|---:|
| drafter weights (`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`) | ~1,086 |
| drafter KV buffer (`f16`) | 45.00 |
| drafter compute buffer | 269.29 |
| **extra recurrent state on the TARGET** (748.12 − 149.62) | **598.50** |
| total | **~1,998** |

Against 1,936 measured. **Roughly a third of "the drafter's cost" is not the
drafter** — it is the target model's own recurrent state, replicated so
speculation can roll back.

### Two corrections this forces

**`-ctkd` / `-ctvd` are not the lever they were called.** On 2026-08-22 they were
recorded here as a VRAM lever on the grounds that the drafter cost 1,936 MiB and
its KV ran at `f16` while the target ran `q4_0`. The drafter's KV buffer is
**45.00 MiB**. Moving it to `q4_0` saves roughly **34 MiB**, not hundreds. The
flag is still untested and still free to try; the estimate was wrong.

**`--spec-draft-n-max` is not free.** An external scan called raising it from 4
toward the clamp of 7 "the biggest single unclaimed win" on throughput. It is
also **+449 MiB** of recurrent state, on a card whose margin at depth is ~600
MiB. Raise it and measure residency in the same round, or the throughput number
will be measured on an arm that spilled.

🔴 **Not yet measured:** whether the drafter's compute buffer (269.29 MiB, six
times its own KV) scales with `-ub`, with `n-max`, or with neither.

---

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
