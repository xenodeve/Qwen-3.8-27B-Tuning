# 09 — The machine itself, and what changed when the card did

**Every other file in this folder assumes one GPU. On 2026-08-23 that stopped
being true.** This page records which card produced which numbers, so a reader
can tell a stale figure from a current one without checking a date.

> 🔴 **Everything in files 01–08 was measured on the RTX 4070 SUPER 12 GB unless
> the row says otherwise.** `CLAUDE.md` forbids comparing raw decode across
> boots. Across hardware it is not a comparison at all — it is two different
> machines.

---

## The two cards

| | RTX 4070 SUPER | RTX 5060 Ti 16 GB |
|---|---|---|
| in service | until 2026-08-23 | from 2026-08-23 |
| VRAM, as the driver reports it | 12,281 MiB | **16,310 MiB** |
| VRAM, **as llama.cpp reports to the process** | **11,069 MiB**, in all 552 logs | **15,172 MiB** |
| compute capability | 8.9 (Ada) | **12.0 (Blackwell)** |
| memory bandwidth | ~504 GB/s | ~448 GB/s *(spec, not measured here)* |
| PCIe | gen4 x16 | gen5 x8 |

**The two VRAM rows are different measurements and only the second one matters
for `--fit`** — [`CORRECTIONS.md` §27](../reports/CORRECTIONS.md). The new card
confirmed it harder than the old one ever did: during one boot with a game
running, `nvidia-smi` reported **7,682 MiB** free while llama.cpp reported
**15,172**. A 7.5 GB gap on the same card at the same instant.

---

## What the new card allocates — measured 2026-08-23, ctx 98,304

One boot, `--spec-type ngram-mod`, corpus `real-code-deep`. **Byte-identical to
the old card on every buffer:**

| | 4070 SUPER | 5060 Ti |
|---|---:|---:|
| model, CUDA0 | 6,521.13 MiB | 6,521.13 MiB |
| model, CPU_Mapped | 397.85 MiB | 397.85 MiB |
| KV (16 attention layers, q4_0) | 1,728.00 MiB | 1,728.00 MiB |
| RS (`n_rs_seq = 0`) | 149.62 MiB | 149.62 MiB |
| compute (`-ub 256`) | 472.27 MiB | 472.27 MiB |
| split | `65+0` | `65+0` |
| `--fit` verdict | `no changes needed`, leaves **2,047** | `no changes needed`, leaves **6,150** |

**KV is 18.00 KiB per token** on this model at q4_0 with 16 attention layers.
That rate is flat and is what every projection below uses.

### What 16 GB actually buys

Projected from the measured rate, on this card:

| ctx | `ngram-mod` only | with DFlash2 (`n_rs_seq = 4`) |
|---:|---:|---:|
| 98,304 | 6,301 MiB free | 4,309 MiB free |
| 131,072 | 5,725 MiB free | — |
| 196,608 | 4,573 MiB free | 2,581 MiB free |
| **262,144** | **3,421 MiB free** | **1,429 MiB free** |

**262,144 is `n_ctx_train` for this model.** For the first time in this project
the ceiling is the model rather than the card. And the DFlash2 column matters:
on the old card the sidecar arms finished with **45–376 MiB** free and were
unreliable there ([`CORRECTIONS.md` §26](../reports/CORRECTIONS.md)); here the
same arms would have 1,429 MiB even at full native context — four to thirty
times that band.

**Whether DFlash2 now wins is unmeasured.** It lost on Ada because it competed
with the layers for a 12 GB budget. That constraint is gone; the verdict is not
transferable in either direction.

---

## The wrong-architecture build — found 2026-08-23, fixed 2026-08-24

`cuobjdump --list-elf` on the shipped `ggml-cuda.dll` in **both**
`llama.cpp-cuda` and `llama.cpp-dflash2`:

```
ELF (SASS)   ggml-cuda.*.sm_89.cubin     <- Ada only
PTX          ggml-cuda.*.sm_89.ptx
```

The card is `sm_120`. The driver JIT-compiles the Ada PTX, producing kernels
tuned for neither architecture — with byte-identical allocation, `65+0`, no OOM,
and **nothing in the log saying the kernels were JIT'd**. That is the exact shape
`CLAUDE.md`'s north star names: an instrument returning a believable number
instead of a failure.

**The fix was one flag.** `llama.cpp-blackwell` is the same tree, same commit,
built with `-DCMAKE_CUDA_ARCHITECTURES="89;120"` — see *The rebuild* below.

### What the rebuild actually bought — ctx 98,304, `bench/hardware_baseline.py`

Same script, same corpus slice, same flags, same card. **Draft acceptance came
out byte-identical at `0.14870 (40 accepted / 269 generated)` in both runs**, so
speculation is held fixed and the difference is the kernels:

| | JIT (`sm_89`) | native (`sm_120a`) | |
|---|---:|---:|---|
| prefill, 43,898 tokens | 146,155 ms | **66,582 ms** | **2.20× faster** |
| decode | 22.67 tok/s | **25.63 tok/s** | +13.1 % |
| boot wall | — | 8.3 s | |

**Prefill is where JIT hurt; decode barely moved.** That is coherent rather than
odd — prefill is compute-bound, so badly-tuned kernels cost dearly, while decode
at batch 1 is bandwidth-bound and less sensitive to kernel quality. **The +13.1 %
on decode is a single unpaired reading and is not claimed as significant.**

*Raw: `logs/dflash2-hwbase-98304-sm89-jit.log` and `logs/dflash2-hwbase-98304.log`.*

### ⚠️ Do not compare either of those decode figures to 96.92

**This page previously did, and it was wrong** —
[`CORRECTIONS.md` §28](../reports/CORRECTIONS.md). `96.92 tok/s` comes from
`results/decoders-98304.jsonl`, six `dflash2_arena` rounds whose every row records
**`acceptance 60.2`**. Both `hardware_baseline` runs record **14.87 %**.
`ngram-mod` is a speculative decoder and its tok/s tracks acceptance directly, so
those numbers differ for a reason that has nothing to do with which GPU ran them.
**The arena and `hardware_baseline` build their prompts differently and their
decode rates are not interchangeable.**

### The prefill comparison across cards, which *is* legitimate

`35,301 ms` is the cold turn-1 of **44,255 tokens** at ctx 98,304 on corpus
`real-code-deep` with `--spec-type ngram-mod`, on the 4070 SUPER
([`08-rtx3090-transfer.md`](08-rtx3090-transfer.md) §6). Ours is **43,898 tokens**
— 0.8 % apart — same corpus, same depth, same decoder, and prefill does not
involve speculation at all. Per token:

| | ms per prefill token |
|---|---:|
| RTX 4070 SUPER, native | **0.798** |
| RTX 5060 Ti, Ada PTX JIT | 3.330 |
| RTX 5060 Ti, native `sm_120a` | **1.517** |

**Correctly built, this card is still 1.90× slower at prefill than the 12 GB card
it replaced.** That is consistent with the silicon — 4,608 CUDA cores against
7,168, and 448 GB/s against 504 — and it means **the 5060 Ti bought VRAM, not
speed.** Plan around capacity, not throughput.

### Decode across the two cards — measured, and it is not the story

`results/decoders-98304-blackwell.jsonl`, three rounds, four arms, arms rotated
within each round, same corpus and ctx as `decoders-98304.jsonl` on the old card.
Produced by `bench/compare_cards.py`, which **withholds a ratio** when draft
acceptance differs, when the corpus hash differs, or when either side's median is
taken over the survivors of an arm that timed out:

| arm | 4070 SUPER | 5060 Ti | spread, old → new | |
|---|---:|---:|---|---|
| `none` | 33.69 | 26.42 | 3.7 % → **1.9 %** | **1.28× slower** |
| `ngram-mod` | 96.92 | 86.45 | 2.8 % → 5.7 % | **1.12× slower** |
| `dflash2` | 49.31 *(5/6 rows)* | 41.42 *(3/3)* | **107.4 % → 8.3 %** | ratio withheld |
| `dflash2+ngram` | **5.66** *(4/6 rows)* | **87.72** *(3/3)* | **1623.4 % → 9.8 %** | ratio withheld |

**Per arm the new card is 1.1–1.3× slower, which is the silicon and was
expected.** Acceptance matched closely enough to compare on the two clean arms
(60.2 vs 61.4 on `ngram-mod`); the two drafter arms are withheld because the Ada
medians are over survivors, not because the new numbers are doubtful.

**What actually changed is that the unusable arms became usable.** On Ada,
`dflash2+ngram` spanned **1.46 to 93.29 tok/s** with two timeouts in six rounds —
a median of 5.66 that describes a failure mode, not a rate. Here it is
**81.64–90.27 with none**, and it is the **fastest arm on the card**, ahead of
the `ngram-mod` every worker profile currently serves.

**The mechanism is free VRAM, and it was predicted before it was measured.**
`CORRECTIONS.md` §26 pinned the drafter's collapse to a **45–376 MiB** band. The
same arms finish here with **2,842–3,183 MiB**. Nothing about the drafter changed;
it stopped being squeezed.

> **So the 16 GB bought reliability and headroom, not throughput.** Every
> individual number got slightly worse and the configuration space got much
> larger. For a project whose metric is *verified accepted coding tasks per
> hour*, an arm that finishes 6 times out of 6 at 87.72 beats one that finishes
> 4 times out of 6 with a median of 5.66, and the tok/s column never said so.

**Not yet claimed:** that `dflash2+ngram` should replace `ngram-mod` in the
served profiles. Three rounds is thin, this is one depth, and `CLAUDE.md` warns
that a verdict at one depth does not transfer. No profile has been changed.

### The noise floor on this card, as far as three rounds can say

| arm | peak-to-peak over 3 rounds |
|---|---:|
| `none` | **1.9 %** |
| `ngram-mod` | 5.7 % |
| `dflash2` | 8.3 % |
| `dflash2+ngram` | **9.8 %** |

The retired **13.6 %** figure was Ada, 12 GB, **ctx 16,384**. At ctx 98,304 on
this card nothing exceeded **9.8 %**, and the arms without a sidecar sit far
below it. **Do not read 9.8 % as the new floor** — three rounds cannot establish
one, and `CORRECTIONS.md` §23 records the same arm spanning 48.9 % at 65,536 on
the old card with byte-identical counters. What it does support is that **an
effect smaller than ~10 % at this depth on a drafter arm is not yet an effect.**

**The guard.** `scripts/worker-5060ti.ps1` reads the code objects out of
`ggml-cuda.dll` before launching and **refuses to start** on a binary without
Blackwell SASS, naming what it found — re-demonstrated against the Ada build on
2026-08-24, exit 1, not a warning. Its check is `Select-String 'sm_120'`, a
substring match, which accepts `sm_120a`; **that it works is luck, not design** —
the string `120a` was not known when the guard was written.

---

## What Blackwell actually changes in llama.cpp — read this before hoping

Rebuilding removes the JIT penalty. It is tempting to also expect Blackwell
*features*, and this section exists so nobody spends a sweep looking for them.
Everything below was read out of the source tree at `1deefcca3` — the same commit
both binaries are built from — not inferred from the card's spec sheet.

**Every Blackwell-gated path in this build is FP4, and none can fire on our
artifact.** `ggml/src/ggml-cuda/mmq-config-blackwell.cuh` contains `CASE` rows for
`GGML_TYPE_MXFP4` and `GGML_TYPE_NVFP4` and nothing else. Its last line is:

```c
return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
```

**For every other tensor type, the Blackwell table *is* the Ampere table Ada
already used.** `mmq.cu:131` says the same thing from the other direction:

```c
const bool use_native_fp4 = blackwell_mma_available(cc) &&
    (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
```

Ada runs those two types too — `mmq-config-ampere.cuh` has 16 rows each — but
through `SRAM_LAYOUT_Q8_1` / `SRAM_LAYOUT_NVFP4` with `MMQ_ITER_K`. Blackwell uses
`SRAM_LAYOUT_FP4` with `MMQ_ITER_K_FP4`, the FP4 tensor-core path. **That is the
entire Ada→Blackwell delta in this build.**

**Flash attention picks the same kernel on both.** This model is
`n_embd_head_k = n_embd_head_v = 256`, `n_head 24`, `n_head_kv 4`, so the
`case 576:` Blackwell branch at `fattn.cu:202` belongs to GLM 4.7 Flash and
Deepseek, not to us. On the path we do take, `turing_mma_available(cc)` is already
true at `890`, and `cc >= GGML_CUDA_CC_ADA_LOVELACE` compares the **raw** cc,
which is `1200` on this card *even in the Ada build*. Neither flips.

| Blackwell-gated thing | fires on `UD-IQ2_XXS`? | why |
|---|---|---|
| `mmq-config-blackwell.cuh` tuning | **no** | MXFP4/NVFP4 rows only; everything else falls through to Ampere |
| `use_native_fp4` (`mmq.cu:131`) | **no** | requires `src0->type` to be MXFP4 or NVFP4 |
| `BLACKWELL_MMA_AVAILABLE` in `mma.cuh`, `quantize.cu`, `common.cuh:870` | **no** | NVFP4 subblock scales |
| `fattn.cu:202` | **no** | inside `case 576:`; our head dim is 256 |
| `GGML_CUDA_USE_PDL` | **not compiled** | see below |

### The one lever, and it is a model change — surveyed 2026-08-24

Native FP4 needs **MXFP4 or NVFP4 weights**. That is an artifact swap, not a
flag — and it is genuinely something the 4070 SUPER could not have used, since
`blackwell_mma_available()` is false on Ada by construction.

**The artifacts exist.** Sizes are from the Hub file listing; **none of this is
measured here and none of it has been downloaded.**

| GGUF | on disk | fits 14.82 GiB? |
|---|---:|---|
| [`esatapedico/…-NVFP4-BUDGET`](https://hf.co/esatapedico/Qwen3.8-27B-NVFP4-BUDGET-GGUF) `STARVED` | **13.59 GiB** | yes, ~1.23 GiB left |
| the same repo's `BUDGET` | **13.71 GiB** | yes, ~1.11 GiB left |
| [`esatapedico/…-NVFP4-MTP`](https://hf.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF) `VERY-LOW` | 13.84 GiB | yes, ~0.98 GiB left |
| the same repo's `COMPACT-LOW` | 14.12 GiB | marginal |
| [`quark75/Qwen3.8-27B-MXFP4-GGUF`](https://hf.co/quark75/Qwen3.8-27B-MXFP4-GGUF) | **15.71 GiB** | **no** |

**The floor is hard and it is not the head tensors.** All seven compact NVFP4
tiers share a **byte-identical 448-tensor NVFP4 backbone of 13.69 GB** — every
layer's attention and MLP. Trimming the LM head, embeddings and MTP block is all
that separates `STARVED` from `VERY-HIGH`, so **no NVFP4 build of this model gets
below ~13.6 GiB.**

**What that leaves for context.** Taking the smallest, `STARVED`: ~1,260 MiB free,
minus the **472 MiB** compute buffer this project measures at `-ub 256`, leaves
~788 MiB of KV at **18.00 KiB/token** — **roughly 44,000 tokens.**

**So the trade is explicit:** FP4 tensor cores and ~4.4 bpw weights at ~44K, against
`UD-IQ2_XXS` at 2.4 bpw with 98,304 measured and 262,144 projected. **For a
project whose stated goal is a usable 128K or more, that is probably the wrong
side of the trade** — but it is now a numbered choice rather than an open question.

**Two more things about the MTP repo, both double-edged.** Its MTP draft head is
**baked into the GGUF** (`blk.64.nextn.*`, enabled with `--spec-type draft-mtp`),
so no sidecar and no second allocation. But `CLAUDE.md` records `draft-mtp` at
**+81 % at 16K and −71 % at 131,072** on our own artifact, so at whatever depth
NVFP4 can reach, the drafter's sign is not predictable from either measurement.

**Nothing above has been run.** It is a survey of what is purchasable with a
download, priced in VRAM.

### 🔴 Decided 2026-08-24: NVFP4 is not being pursued

**Developer decision, on the numbers above.** The NVFP4 backbone floor is
~13.6 GiB against 15,172 MiB free, which leaves roughly **44,000 tokens** of
context once the 472 MiB compute buffer is taken. This project's goal is a usable
**128K or more**. Trading 98,304 measured — and 262,144 projected — for ~44K to
gain FP4 tensor cores is the wrong side of the trade, and no NVFP4 build of this
model gets below that floor because all seven compact tiers share the same
448-tensor backbone.

**Nothing was downloaded.** The 13.59 GiB file was never fetched; the decision
rests on the Hub's own file sizes and this card's measured 18.00 KiB/token KV
rate. **Reopen only if** the KV rate changes, a smaller NVFP4 backbone is
published, or the context target drops.

**So the native FP4 path in this build is unreachable for us, and that closes the
question of Blackwell-specific optimisation for this artifact entirely** — every
other Blackwell-gated path falls through to the Ampere table.

### The 16 GB upgrade does not unlock Q4 residency

[`11-DEPTH-ON-IQ2XXS.md`](../reports/11-DEPTH-ON-IQ2XXS.md) §3 already recorded
that the Q4 attempt at 256K was **stopped rather than measured**, on host-RAM
pressure, back on the 12 GB card. So Q4-does-not-fit is not new. **What is new is
that four more gigabytes do not fix it:**

| | |
|---|---:|
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` on disk | **16.69 GiB** |
| free VRAM as llama.cpp reports it here | **15,172 MiB** (14.82 GiB) |

The weights alone exceed the card before a single KV byte is allocated, so the
CPU-offload-and-page situation that ended the earlier experiment still applies.
The largest variant that leaves room for a deep window is **`UD-Q3_K_XL` at
12.52 GiB**, which is untried on this card. Any plan phrased as "16 GB unlocks
Q4 + 256K + full residency" is measured false and should be re-aimed at Q3 or
kept on IQ2.

### Parked deliberately: `GGML_CUDA_USE_PDL`

Programmatic Dependent Launch is gated `__CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER`
(900), so **Ada at 890 can never use it and Blackwell at 1200 can** — the only
non-FP4 capability difference found. But `grep` finds **no cmake option and no
header defining `GGML_CUDA_USE_PDL`** at this commit; it is dead code unless the
macro is forced onto the CUDA command line, and its own comment says it requires
dropping `__restrict__` throughout (llama.cpp PR #24030), which changes codegen
broadly. **It was kept out of the native build on purpose** — including it would
have destroyed the single-variable property proved below. It is a separate
experiment and has not been run.

---

## The rebuild, and how it was kept to one variable

Both delivery directories are built from `1deefcca3`. Configuring
`build-blackwell` and diffing `CMakeCache.txt` against `build-dflash2` gives
**345 entries on both sides and exactly one differing value**:

| | `build-dflash2` | `build-blackwell` |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `89` | `89;120` |
| everything else (344 entries) | — | identical |

**The first configure attempt did not have that property.** It defaulted
`LLAMA_BUILD_EXAMPLES=ON`, `LLAMA_BUILD_TESTS=ON` and `LLAMA_CURL` to values the
Ada build never used. Two of those only cost build time; `LLAMA_CURL` changes what
`llama-server` — the binary we benchmark — can do. The diff caught it before
anything was compiled. **Configure, diff, *then* build.**

**cmake rewrites the architecture without being asked:**

```
-- Replacing 120 in CMAKE_CUDA_ARCHITECTURES with 120a
-- Using CMAKE_CUDA_ARCHITECTURES=89;120a
```

`120a` is the arch-specific variant that enables the sm_120a-only instructions the
Blackwell paths compile against. **So the binary reports `sm_120a`, not `sm_120`**
— `cuobjdump --list-elf` on the new `ggml-cuda.dll` gives **141 `sm_120a` cubins
alongside 141 `sm_89`**, where the Ada one has only the 141 `sm_89`.

### ⚠️ The two builds are indistinguishable by every ordinary means

```
version: 0.1.2-dev (build 10499, commit 1deefcca3)
built with MSVC 19.44.35228.0 for Windows AMD64
```

**That is both of them, byte for byte.** Same commit, same compiler, same buffer
sizes, same `65+0` — and a 4× difference in decode. Nothing in `--version`,
nothing in the boot log, and nothing in a result row separated them.

So `dflash2_arena` now records `exe` and `cuda_archs` on **every** row, read out
of the code objects rather than the version string, and
`bench/tests/test_exe_provenance.py` pins that the two builds compare unequal.
`QWEN38_LLAMA_EXE` selects the binary, so the Ada figures stay reproducible
instead of being edited away.

---

## The healthy-load signature, which this project never had

`04-context-depth.md` recorded that `gpu-trace-98304.jsonl` showed **100 % GPU
utilisation at 4 % memory utilisation and 76 W** during the failing DFlash2 arms,
and noted there was **no control trace** to say whether that was abnormal.

There is one now. This card under a real 44K prefill and a 512-token decode:

```
GPU utilisation      99 %
memory utilisation   44 %
power             174.5 W   (TDP ~180 W)
```

**44 % memory utilisation is what work looks like on this model.** The old
signature — 100 % / **4 %** / 76 W — was a card spinning, not a card working,
which [`CORRECTIONS.md` §26](../reports/CORRECTIONS.md) argued from mechanism and
this now supports from a control.

---

## What transfers across the card change, and what does not

**Transfers — these are mechanisms, and a mechanism does not care which GPU runs
it:**

- `-ctk q4_0 -ctv q4_0` — no other KV type in this build has a fast kernel
- **`-cram` must never be 0** — 343× on task switching, and it caches sequence
  state in *host* RAM
- `--ctx-checkpoints` default 32 carries prefix reuse when `n_rs_seq = 0`
- an edit ahead of the suffix **zeroes** reuse rather than degrading it
- chars/token ≈ 3.4 — a property of the tokenizer and corpus
- `--fit` acts almost never, and reads a number `nvidia-smi` does not show

**Does not transfer — every rate, and every arm verdict:**

- 96.92 / 49.31 / 5.66 / 33.69 tok/s and the decoder ranking they produced
- the 45–376 MiB unreliability band
- the 13.6 % noise floor, and the 48.9 % spread at 65,536
- `-ub 64` costing 14.0 % of decode
- **`ngram-mod` as the right decoder** — it is the starting point here, not a
  verdict

*Raw: `logs/dflash2-hwbase-98304.log`, `bench/hardware_baseline.py`. Issue #40.*
