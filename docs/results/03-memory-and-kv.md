# 03 — Memory: KV, tensor placement, reserves, batch

> 🔴 **Every number on this page was measured at `reasoning_effort: xhigh` with
> an unlimited thinking budget.** That is the model's chat-template
> default — the client sends no effort field, and **no `worker-*.ps1` profile and
> nothing in `bench/` has ever set the flag** (established 2026-08-24 from a boot
> log: [`05-runtime-flags.md`](05-runtime-flags.md)).
> Artificial Analysis prices this model's `medium` **one point** below `xhigh` on
> the agentic axis and `low` **six** below that
> ([`researchs/artificial-analysis`](../researchs/artificial-analysis/README.md)),
> so **effort is a live confound here, not a settled background condition.**
>
> **The served default became `medium` on 2026-08-24** — all five
> `worker-*.ps1` profiles and `dflash2_arena.server_argv` now set it, and the
> arena records `effort` on every row. **So this banner describes what is
> already on the page, not what will be added to it.** Anything measured after
> that date states its own level, and a figure from before it cannot be
> compared with one from after without saying which is which.

## 🔴 Three buffers that look fixed scale with context — measured 2026-08-24

**A projection built from buffers measured at one depth was wrong at another, and
a boot caught it.** `UD-Q2_K_XL` with `draft-mtp,ngram-mod` was projected to leave
1,790 MiB at ctx 163,840. It does not load there: `--fit` reports *"cannot meet
free memory target of 1522 MiB, need to reduce device memory by 154 MiB"* and
spills to **64/66**, and two CPU layers at depth cost what
[`04-context-depth.md`](04-context-depth.md) measures — `AD-IQ1_M` at `65+1`
decodes **6.08 tok/s** against 26.50 resident.

The projection treated as fixed three buffers that are not:

| buffer | 98,304 | 131,072 | 147,456 | 163,840 | rate |
|---|---:|---:|---:|---:|---|
| target KV | 1,728.00 | 2,304.00 | 2,592.00 | 2,880.00 | **18.00 KiB/token** |
| **target compute** | **472.27** | **616.27** | **688.27** | **777.57** | **~0.0047 MiB/token** |
| **MTP draft KV** | **384.00** | **512.00** | **576.00** | **640.00** | **4.00 KiB/token exactly** |
| **MTP compute** | **82.01** | **98.01** | **106.01** | **114.01** | **~0.0005 MiB/token** |

**Only the first row is the one everybody knows.** The other three add roughly
**290 MiB per 32,768 tokens** — enough to turn a 1,790 MiB projection into a
154 MiB shortfall over two rungs.

`-ub` does not change between these boots. The compute buffer grows with the
window anyway.

**The measured ceiling for that configuration is 147,456**, 66/66 resident:

```
model 8,965.31 | KV 2,592.00 | RS 598.50 | compute 688.27
MTP KV 576.00  | MTP compute 106.01 | CPU_Mapped 397.85
13,526 MiB of the 15,172 llama.cpp sees, leaving 1,646
--fit: "will leave 1727 >= 1450 MiB, no changes needed"
```

131,072 is the safer rung at **2,078 MiB** free.

> **Rule this leaves behind: a VRAM projection is not a residency verdict.** One
> boot settles it in under a minute, and `--fit` will silently spill layers
> rather than refuse — which reads as success in every field except the layer
> count.

*Raw: `logs/probe-q2kxl-mtp-131072.log`, `-147456.log`, `-163840.log`,
`scripts/worker-q2kxl-mtp.ps1`.*

---

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

**And it is a residency lever at depth.** `--spec-draft-n-max` multiplies the
RS buffer by `1 + n_max`, so at **ctx 65,536** the `n=7` arm loads **`63+2`** —
two layers on the CPU — and the recurrent state itself splits, **49.88 MiB**
landing on the host. `n=3` and `n=4` stay `65+0` there. The +23.4 % that `n=7`
buys at 16,384 does not survive that. Raw: `results/sweep-draft-n-65536.jsonl`.

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
| **`-ub` 256 / 128 / 64 at ctx 98,304** | **2026-08-23, priced** | compute buffer **472.27 → 428.27 → 406.27 MiB**; decode **−3.7 % (inconsistent) / −14.0 % RESOLVED**. See below |
| `-b 1024 -ub 128` | queued (V2) | — |

Smaller buffers trade prefill for VRAM. Unlike `-ot`, that is a trade whose price
is visible in the `pp` column.

**Now the price is measured, and it is bad.** A 4× cut in `-ub` returns
**66 MiB** of compute buffer and costs **14.0 % of decode [−14.8, −13.7],
RESOLVED** over three paired rounds against `ngram-mod` — the decoder every
worker profile serves. `results/ubatch-98304.jsonl`,
[`05-runtime-flags.md`](05-runtime-flags.md) for the full table and the
disqualified `ub-128` round.

**66 MiB is also not enough for the thing it was wanted for.** The arms that run
out of VRAM are the ones loading DFlash2, which finish with 45–376 MiB free and
are unreliable there ([`CORRECTIONS.md` §26](../reports/CORRECTIONS.md)); this
moves them to 111–442, the same band.

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

## 🔴 `--cache-reuse` re-indexes the attention KV and leaves the recurrent state behind — read from source 2026-08-27, NOT measured

**Register: source read on the exact revision that runs.** `llama-server.exe
--version` reports build 10499 commit `1deefcca3`; `C:\AI\llama.cpp` is at
`1deefcca3`. No GPU round was taken — every claim below is a line of code, and
the falsifiable prediction is at the end.

Qwen3.8-27B is a hybrid. Our own boot log
([`logs/bat-maxctx2.log`](../../qwen38-tuning/logs/bat-maxctx2.log)) says
`arch = qwen35`, `n_swa = 0` and **`n_rs_seq = 0`**. With `n_swa = 0` the model
builds **`llama_memory_hybrid`** (`llama-model.cpp:2296,2325`) — an attention KV
cache and a **separate recurrent (Gated DeltaNet) state**.

**The startup gate asks the wrong half.** The server disables `--cache-reuse`
when `llama_memory_can_shift()` is false (`server-context.cpp:1176-1185`), and
for a hybrid that call returns **`mem_attn->get_can_shift()`** —
`llama-memory-hybrid.cpp:133-136`, whose comment reads *"Shifting is trivially
supported for recurrent"*. That is true of a **position** and false of a
**state**. So the flag is accepted and no warning is printed.

**What the reuse loop then does.** `server-context.cpp:3180-3181`, per matched
chunk:

```cpp
slot.mem.seq_rm (slot.id, head_p, head_c);
slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);
```

`llama_memory_hybrid::seq_rm` tries the recurrent side first and refuses to
mutate the attention cache if it fails (`llama-memory-hybrid.cpp:113-119`) —
correct, as far as it goes. The problem is that **it does not fail.** For a
mid-sequence range (`head_p < head_c <= tail`),
`llama_memory_recurrent::seq_rm` (`llama-memory-recurrent.cpp:150-233`) takes
neither special branch — the bounded-rollback branch needs `p1 > cell.pos`, the
tail invalidation needs `cell.pos < p1`, and the cell loop matches on
`cells[i].pos`, which for a recurrent cell is the **tail** — and falls through to
**`return true` having touched nothing**.

**The result:** the attention KV is re-indexed to the new prompt, the DeltaNet
state still encodes the old prefix, and nothing reports it. Prefill is skipped
for tokens whose recurrent history was never rebuilt.

**The other branch is a crash, not a wrong answer.** If a removal does reach the
tail (`p1 > cell.pos`), the rollback branch computes `rollback` and requires
`rollback <= n_rs_seq`. Ours is 0, so it returns false, the hybrid returns false,
and `common_context_seq_rm` calls **`GGML_ABORT`** (`common/common.cpp`, the
`"failed to remove sequence"` line). The server dies rather than lying.

**And rollback cannot be switched on from the command line.** `n_rs_seq` has no
argument in `arg.cpp`; it is set from
`common_params_speculative::need_n_rs_seq()` (`common/common.h:386-392`), which
returns `draft.n_max` **only** for `draft-mtp`, `draft-eagle3`, `draft-dflash`
and `draft-dspark` — and `0` for every `ngram-*` type. We serve `ngram-mod`,
which is why the log reads 0. `qwen35` **is** in
`llm_arch_supports_rs_rollback` (`llama-arch.cpp`), so the capability exists and
is simply never provisioned for us. It would not save the mid-sequence case
anyway: that path never consults the snapshot index.

**Consequence for the served profile: do not set `--cache-reuse`.** It is the
shape this project exists to catch — an instrument that returns a believable
answer instead of a failure.

### The prediction, so this can be refuted cheaply

With `--cache-reuse 256`, greedy (`temperature 0.0, top_k 1, seed 42`), send a
prompt, then the same prompt with an edit in the middle, and compare the second
generation against a **cold** run of the edited prompt. **If the two differ, the
claim above is confirmed**; if they match token for token, it is wrong and this
entry must be retracted. A tail-edit variant should abort the server instead.

*Found while diffing all 322 `llama-server` flags against the 20 the dual
profile sets. Task #42. Related: [`04-context-depth.md`](04-context-depth.md)
for what a broken prefix costs.*

## 🟡 The `q8_0` KV verdict at 147,456 was measured on the even split — flagged 2026-08-27, NOT re-run

**The register says "`q8_0` cannot load at 147,456 — `cudaMalloc failed: out of
memory` on the 12 GB card". That run did not carry `-ts`.**

`logs/ceil-q4-q8_0-147456-dual-tensor-q8.log` records no argv, so the argument
list cannot be read from it directly. The split is still recoverable from the
load line:

| run | `-ts` banner | `Meta() model buffer size` |
|---|---|---|
| `bat-dual-fixed.log` | `-ts 7819,15490` | 10,805.96 MiB |
| `bat-maxctx2.log` | `-ts 6779,15489` | 11,141.82 MiB |
| **`ceil-q4-q8_0-147456-dual-tensor-q8.log`** | **absent** | **8,065.29 MiB** |

**8,065.29 x 2 = 16,130.58 MiB, the model exactly.** That is the even split —
`-sm tensor` divides evenly when given no ratio (`llama-model.cpp:707`), which is
the same configuration that produced 0.38 tok/s and was retracted in
[CORRECTIONS 33](../reports/CORRECTIONS.md).

**The arithmetic says it would probably fit with the computed split.** `q8_0` is
36 KiB/token, so 147,456 costs 5,184 MiB of KV; with 16,130 of weights, ~2,048 of
compute and the 768 MiB runtime reserve that is **24,130 MiB against 26,072-27,072
available** after the desktop. Under the even split the 12 GB card was instead
asked for 8,065 MiB of weights plus its half of the KV — and it OOM'd allocating
1,024.30 MiB on device 0, which is that card.

**This is not a retraction: nothing has been re-measured.** The verdict is
**confounded**, not disproved, and the row stays until a run with `-ts` either
loads or repeats the OOM.

**It is a quality lever, not a speed one.** `q8_0` measured **-0.3 %** against
`q4_0` at 16,384 — inside any floor. It matters only because KV precision is one
of the few knobs that could move quality, which this project has never measured
on its own artifacts.

### And `f16`/`bf16` KV have never been compared on two cards at all

This build compiles a flash-attention kernel for exactly four types — **`f16`,
`bf16`, `q4_0`, `q8_0`** (issue #43) — so our `q4_0` is on a supported FA path,
not a dequant fallback. But the `dual-kv` arm set holds only `q4_0` and `q8_0`.
The old *"`f16` not measurable here"* verdict was a **12 GB single-card**
constraint: at 16,384 `f16` KV is 1,152 MiB, which two cards now hold without
effort. At 147,456 it is 10,368 MiB and still does not fit.

**Hypothesis, unmeasured:** quantised KV trades memory bandwidth for dequant
work, and at a shallow depth the KV is small enough that the bandwidth saving is
worth little while the dequant is still paid every step — so `f16` may be the
faster kernel where it fits. That becomes practically relevant if the DFlash2
ladder (task #44) settles on a shallower served window.

*Task #46. Related: [`README.md`](README.md) KV row, and the `dual-kv` arm set in
`bench/dflash2_arena.py`, which carries no `-ts`.*
