# 34 — Blackwell bought headroom, not speed — 2026-08-24

**The session that rebuilt llama.cpp for the card that is actually installed,
and then found the more interesting result was not the one it went looking for.**

Everything here is `results/decoders-98304-blackwell.jsonl`,
`logs/dflash2-hwbase-98304*.log`, and the llama.cpp tree at `1deefcca3`. The
register is [`results/09-hardware.md`](../results/09-hardware.md); this is the
argument.

---

## 1. The setup: a binary that could not see the card it was running on

The RTX 4070 SUPER 12 GB became an RTX 5060 Ti 16 GB on 2026-08-23. Every binary
this project had ever benchmarked was built `CMAKE_CUDA_ARCHITECTURES=89` — Ada
only. The card is `sm_120`. The driver JIT-compiles the Ada PTX, and the first
read on the new card came back **four times slower with nothing anywhere saying
why**: byte-identical buffers, `65+0`, no OOM, and a `--version` string identical
to a correct build.

So the session's first job was a rebuild, and its first discipline problem was
making sure the rebuild changed **one thing**.

### The `CMakeCache` diff, which earned its keep immediately

Configure `build-blackwell` from the same tree, diff `CMakeCache.txt` against
`build-dflash2` **before compiling anything**. The first attempt came back with
**five** differing entries, not one:

```
CMAKE_CUDA_ARCHITECTURES   89  ->  89;120     <- intended
LLAMA_BUILD_EXAMPLES       OFF ->  ON
LLAMA_BUILD_TESTS          OFF ->  ON
LLAMA_CURL                 OFF ->  (default)
CMAKE_MAKE_PROGRAM         winget ninja -> VS ninja
```

Two of those cost only build time. **`LLAMA_CURL` changes what `llama-server`
— the binary we benchmark — can do.** Reconfigured with every flag matched, the
diff reads **345 entries on both sides, one differing value.** That is the
sentence the rest of this report rests on, and it exists because the check ran
between configure and build rather than after.

**cmake then rewrote the architecture without being asked:**

```
-- Replacing 120 in CMAKE_CUDA_ARCHITECTURES with 120a
```

`120a` is the arch-specific variant carrying the sm_120a-only instructions the
Blackwell code paths compile against. The handoff had written `"89;120"`. The
result was right for a reason nobody in this project had written down.

---

## 2. What the rebuild bought, with speculation held fixed

Same script, same corpus slice, same flags. **Draft acceptance came out
byte-identical — `0.14870 (40 accepted / 269 generated)` — in both runs**, so the
speculative decoder is not a free variable and the difference is the kernels:

| | JIT (`sm_89`) | native (`sm_120a`) | |
|---|---:|---:|---|
| prefill, 43,898 tokens | 146,155 ms | **66,582 ms** | **2.20× faster** |
| decode | 22.67 tok/s | 25.63 tok/s | +13.1 %, one unpaired read |

**Prefill is where the JIT hurt; decode barely moved.** Coherent rather than
odd: prefill is compute-bound, so badly-tuned kernels cost dearly, while decode
at batch 1 is bandwidth-bound and less sensitive to kernel quality.

---

## 3. Then the number that had been published for a day turned out to be a
category error

`09-hardware.md`, the ledger, issue #40 and its commit all said **"four times
slower than the 4070 SUPER"**, from this table:

| | 5060 Ti (JIT) | 4070 SUPER |
|---|---:|---:|
| prefill | 146,155 ms | 35,301 ms |
| decode | **22.67 tok/s** | **96.92 tok/s** |

Tracing where each came from:

```
96.92   results/decoders-98304.jsonl -- dflash2_arena, 6 rounds, median of 3
        every one of its six ngram-mod rows:   acceptance 60.2
22.67   logs/dflash2-hwbase-98304*.log -- hardware_baseline.py, 1 generation
                                draft acceptance  0.14870
```

**`ngram-mod` is a speculative decoder and its tok/s tracks acceptance
directly.** 60.2 % against 14.87 % is a four-fold difference in how much
speculation is doing, produced by the two tools building their prompts
differently. And `hardware_baseline.py` was written *after* the swap — **the
4070 SUPER never ran it.** There was no same-instrument figure, and the table
filled the gap with the nearest available number instead of saying so.

Retracted as [`CORRECTIONS.md` §28](CORRECTIONS.md).

**The prefill row was always fine**, because prefill involves no speculation at
all. `35,301 ms` is the cold turn-1 of 44,255 tokens on the same corpus at the
same depth ([`08-rtx3090-transfer.md`](../results/08-rtx3090-transfer.md) §6),
0.8 % away from our 43,898. Per token:

| | ms per prefill token |
|---|---:|
| RTX 4070 SUPER, native | **0.798** |
| RTX 5060 Ti, Ada PTX JIT | 3.330 |
| RTX 5060 Ti, native `sm_120a` | **1.517** |

**Correctly built, this card is still 1.90× slower at prefill than the 12 GB card
it replaced.**

### What made it possible, and what was done about it

Two builds on this machine, same commit, same compiler, print the same version
string byte for byte and differ 2.20× in prefill. **Nothing in `--version`, the
boot log, or a result row separated them.**

So every arena row now carries `exe` and `cuda_archs`, read out of the code
objects rather than the self-report (`bench/provenance.py`,
`tests/test_exe_provenance.py`), and `bench/compare_cards.py` **withholds a
ratio** when acceptance differs by more than five points, when the corpus hash
differs, or when a median is taken over the survivors of an arm that timed out.
The three refusals are not hypothetical — each is a row in this repo's own data,
and the first is the mistake above.

---

## 4. The result the session did not go looking for

The decoder sweep, re-run natively. Three rounds, four arms, arms rotated within
each round, same corpus and ctx as the old card's file:

| arm | 4070 SUPER | 5060 Ti | spread, old → new | |
|---|---:|---:|---|---|
| `none` | 33.69 | 26.42 | 3.7 % → **1.9 %** | **1.28× slower** |
| `ngram-mod` | 96.92 | 86.45 | 2.8 % → 5.7 % | **1.12× slower** |
| `dflash2` | 49.31 *(5/6 rows)* | 41.42 *(3/3)* | **107.4 % → 8.3 %** | withheld |
| `dflash2+ngram` | **5.66** *(4/6 rows)* | **87.72** *(3/3)* | **1623.4 % → 9.8 %** | withheld |

**Per arm, the new card is 1.1–1.3× slower. That is the boring half and it
matches the silicon** — 4,608 CUDA cores against 7,168, 448 GB/s against 504.

**The interesting half is the spread column.** On Ada, `dflash2+ngram` spanned
**1.46 to 93.29 tok/s** with two timeouts in six rounds. Its median of 5.66
describes a failure mode, not a rate. Here it is **81.64–90.27 with none**, and
it is the **fastest arm on the card** — ahead of the `ngram-mod` every worker
profile currently serves.

**The mechanism was predicted before it was measured.**
[`CORRECTIONS.md` §26](CORRECTIONS.md) pinned the drafter's collapse to a
**45–376 MiB** band of free VRAM. The same arms finish here with
**2,842–3,183 MiB**. Nothing about the drafter changed. It stopped being
squeezed.

> **This project's metric is verified accepted coding tasks per hour.** An arm
> that finishes 6 of 6 at 87.72 beats one that finishes 4 of 6 with a median of
> 5.66, and the tok/s column never said so. **The 16 GB bought reliability and
> configuration space, not throughput** — which is a better purchase than the one
> that was hoped for, and a different one.

**No profile has been changed.** Three rounds is thin, this is one depth, and
`CLAUDE.md` warns that a verdict at one depth does not transfer — `draft-mtp` is
+81 % at 16K and −71 % at 131,072 on this same artifact.

---

## 5. What Blackwell itself gives us: FP4, and we cannot reach it

Read out of the tree, not inferred from the spec sheet.

`mmq-config-blackwell.cuh` contains `CASE` rows for `GGML_TYPE_MXFP4` and
`GGML_TYPE_NVFP4` and nothing else. Its last line:

```c
return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
```

**For every other tensor type, the Blackwell table *is* the Ampere table Ada
already used.** `mmq.cu:131` agrees from the other side: `use_native_fp4`
requires `src0->type` to be one of those two. `fattn.cu:202`'s Blackwell branch
sits inside `case 576:`; this model is `n_embd_head_k = 256`. On the path we do
take, `turing_mma_available(cc)` is already true at `890`, and
`cc >= GGML_CUDA_CC_ADA_LOVELACE` compares the **raw** cc, which is `1200` here
*even in the Ada build*.

**So there is no flag to sweep for.** The only lever is FP4 weights — a model
artifact change, and genuinely something Ada could not have used, since
`blackwell_mma_available()` is false there by construction.

**And it does not fit.** All seven compact NVFP4 tiers of the published
Qwen3.8-27B conversions share a byte-identical **13.69 GB** NVFP4 backbone; the
smallest complete file is **13.59 GiB** against **15,172 MiB** free. After the
472 MiB compute buffer that leaves roughly **44,000 tokens** of KV at
18.00 KiB/token. Against `UD-IQ2_XXS` at 98,304 measured and 262,144 projected,
that is the wrong side of the trade for a project whose goal is a usable 128K —
**but it is now a numbered choice rather than an open question.** Sizes and links
are in [`09-hardware.md`](../results/09-hardware.md); **none of it has been
downloaded or run.**

`GGML_CUDA_USE_PDL` is the one non-FP4 capability difference — Hopper-gated at
900, so Ada at 890 could never use it and Blackwell at 1200 can. It has **no
cmake option and no header defining it** at this commit, and its own comment says
it requires dropping `__restrict__` throughout. **Kept out of the build on
purpose:** including it would have destroyed the one-variable property §1 rests
on.

---

## 6. Two guards this session added because it tripped over them

**Five bench scripts still pointed at an Ada-only build.** `ctx_ceiling.py`,
`depth_sweep.py`, `kv_kernel_screen.py`, `model_arena.py` and `sweep_runtime.py`
each hardcoded `C:\AI\llama.cpp-cuda\llama-server.exe`. Launching any of them
today would have JIT'd, taken 2.2× the prefill time, and said nothing. All six
now route through `provenance.resolve_exe(default)` with their defaults
byte-identical, and a test scans `bench/` and fails on any `EXE` bound to a
literal.

**An audit rule that could not match.** Widening `test-count` wrote `\\b` where
it meant `\b` — legal regex for "a literal backslash, then b". The file
imported, the audit ran, and the rule reported the tree clean of a claim it had
stopped looking for. `audit-stale-claims.py` now exercises every rule at import
and refuses to run on a doubled escape; mutation-proved, not assumed.

**Both are the same shape as everything else here**: not a wrong answer, but a
right-looking answer from an instrument that had quietly stopped measuring.

---

## 6b. The one new knob the session could reach, and it is a null

`grep getenv ggml/src/ggml-cuda/` finds **twelve** runtime knobs that are not
flags. The arena could not test any of them — arms carried argv only, so an env
var meant re-running the sweep with it exported, which is a comparison across
boots. Arms now carry an env mapping and every row records it, which made one
measurable:

`GGML_CUDA_GRAPH_OPT` is off unless set to `1` (`ggml-cuda.cu:4330`), requires
CUDA graphs and **exactly one device** — both true here — and decode at batch 1
is precisely the many-small-kernels case graph optimisation exists for.

| | r1 | r2 | r3 | |
|---|---:|---:|---:|---|
| off | 79.4 | 82.3 | 84.6 | spread 6.6 % |
| on | 84.0 | 76.6 | 89.3 | spread **16.6 %** |
| paired | **+5.8 %** | **−6.9 %** | **+5.6 %** | mean +1.4 % |

`harness.paired_deltas` returns **`within noise / inconsistent`** — it resolves an
effect only when the sign is consistent *and* the magnitude clears the floor, and
this is neither. It did not reduce variance either; the treated arm is wider.

**And the null is ambiguous on purpose.** Nothing echoes the variable back, so
"no effect" and "not applied" are both consistent with this data. The register
says so rather than picking the flattering one.

---

## 7. What is open

- **Whether `dflash2+ngram` should replace `ngram-mod` in the served profiles.**
  Three rounds, one depth. Needs 6 rounds and at least one more depth.
- **The context ladder to 262,144**, now reachable for the first time —
  `n_ctx_train` is the ceiling rather than the card.
- **`GGML_CUDA_USE_PDL`** — needs a second build, and must not ride along with
  anything else.
- **Eleven more environment knobs**, listed in
  [`05-runtime-flags.md`](../results/05-runtime-flags.md) and untried.
- **A real noise floor at depth on this card.** Three rounds gave 1.9 / 5.7 /
  8.3 / 9.8 % peak-to-peak. That is not a floor; it is four samples of one.

*Issues [#40](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/40),
[#41](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/41).*
