# The 16-Layer Optimization Programme — Qwen3.8-27B, Ordered by Expected Value

> **Date:** 2026-08-20 UTC+7
> **Target:** Qwen3.8-27B, fully GPU-resident at ≥128K, highest tok/s, then
> quality from the fastest arm down until one is good enough.
> **Surface:** [`../reports/16-OPTIMIZATION-SURFACE.md`](../reports/16-OPTIMIZATION-SURFACE.md)
> · **Governing result:** [`../reports/19-THE-128K-PLATEAU.md`](../reports/19-THE-128K-PLATEAU.md)
>
> Every flag below was **verified to parse against build 10472 `60eeeb608`
> before being queued**, so no GPU time is spent on a typo. Ordering is by
> expected effect on the four axes the developer named — **tok/s, context
> window, VRAM, quality** — not by novelty.

---

## 0. The two facts that set the ordering

**1. At 128K, throughput is a plateau.** Ten boots across three artifacts and two
sessions: 24.98–27.46 tok/s, all at `65+0`, all with a 2,304 MiB cache. The
spread is inside the 13.6 % noise floor. *No lever can raise tok/s while the
window stays 128K, because the cost is the cache and the cache is fixed by the
window.* (Report 19.)

**2. The current failure is format, not reasoning.** 41.5 % (V3 `IQ1_M`) to
58.3 % (V3 `IQ2_XXS`) of corpus attempts emit **no fenced code block at all**,
having looped inside the reasoning block until the token cap.

Together these say: at the target depth the remaining wins are **VRAM → deeper
residency**, and **quality → fewer wasted attempts**. Pure throughput tuning at
128K is spending time on a number that cannot move.

---

## 1. Tier 1 — can change the answer outright

| # | layer | lever | axis | why it is here | status |
|---|---|---|---|---|---|
| 1 | 12 + 13 | `--grammar-file` + `--reasoning-budget 0` | **quality** | 41.5–58.3 % of attempts produce nothing. A grammar makes the fence a property of the sampler, not a hope about the prompt | **queued** `afk-q38-quality.sh` |
| 2 | 6 | `--ctx-checkpoints 8` | **VRAM → context** | `AD-IQ1_M` — the only 1-bit artifact with a good corpus (27/31) — misses `65+0` at 131,072 by **one layer** (338 MiB free, needs ~125). Default is 32 checkpoints of speculative VRAM an append-only agent never rewinds into | **queued** `afk-q38-ckpt.sh` |
| 3 | 9 | `--spec-draft-device none` / `-otd .*=CPU` with the 1.28 GiB standalone MTP drafter | **tok/s** | This project published "MTP does not pay on a resident target" (−8.8 %). That was measured with the head **on the GPU**, where its VRAM moved the split 61+4 → 55+10. Three arms put the same drafter in three places — the comparison the verdict never had | **queued** `afk-q38-depth-levers.sh` |
| 4 | 7 | `--context-shift` + `--keep 2048` | **context** | Changes the problem from "hold 256K of cache" to "move a 128K window", which sidesteps the weights-vs-cache competition entirely. Default is **disabled** | **queued** |
| 5 | 6 | `-ctk q8_0 -ctv q4_0` asymmetric | **VRAM** | K carries positional precision, V is more compressible. Buys cache at possibly lower quality cost than compressing both | **queued** |

---

## 2. Tier 2 — cheap, measurable, one boot each

| # | layer | lever | axis | prediction being tested | status |
|---|---|---|---|---|---|
| 6 | 9 | `ngram-simple` · `ngram-mod` · `ngram-map-k` · `ngram-map-k4v` · `ngram-cache` | tok/s | **Free** — no drafter file, no download. `map-k4v` keeps four candidate continuations, which is the shape of a file being edited | **queued** 16K screen |
| 7 | 8 | `-fa off` vs the build default `auto` | validity, then tok/s | Every cross-artifact comparison this project has published assumes FA resolved identically on every arm. **Nobody has read the loader's decision.** This is a validity check before it is an optimization | **queued** |
| 8 | 4 | `-ot "blk\.(5[0-9]\|6[0-4])\.ffn_.*=CPU"` and the `ssm_.*` twin | VRAM, tok/s | `--fit` treats layers as indivisible and identical. They are not. Run at **163,840** where `v3-iq2xxs` is 62+3 and three layers are already on the CPU — moving only the tail FFN should beat moving whole layers | **queued** depth levers |
| 9 | 11 | `--cpu-mask 0x0FFF` (P-cores 0–11, verified by `PercentProcessorPerformance`) | tok/s | Cannot matter at `65+0` where the CPU does no decode work. Run at **163,840** (62+3) where it can | **queued** depth levers |
| 10 | 6 | `--cache-reuse 256` | wall clock | Attacks the largest single cost ever measured here: **one broken prefix costs 63 s at 16K and 248 s at 64K** | **queued** |
| 11 | 14 | `--slot-save-path` | wall clock | Save/restore a slot's KV across restarts. An 11-minute 256K cold prefill becomes an NVMe read | **to arm** |
| 12 | 13 | `--prefill-assistant` · `-rea off` · `--reasoning-budget 2048` | quality, tok/s | Cheaper cousins of the grammar. `-rea off` is untested and this project's metric is tasks per hour, not thoughts per task | **to arm** |
| 13 | 12 | DRY vs `--repeat-penalty 1.05 --repeat-last-n 4096` vs `--top-n-sigma` vs `--mirostat 2` | quality | All **off by default**. `--repeat-last-n 64` cannot see a 4,000–8,000-token loop — that is arithmetic, not opinion | **to arm** |

---

## 3. Tier 3 — predicted inert, queued anyway so the prediction is falsifiable

The developer asked explicitly for these. A confirmed "inert, and here is the
number" closes a question permanently; a dropped one leaves it open forever.

| layer | lever | prediction | status |
|---|---|---|---|
| 4 | `-sm tensor` | inert — one GPU | **queued** 16K screen |
| 10 | `-np 2` | inert at one stream, and each slot costs KV | **queued** |
| 5 | `--load-mode none` | zero on decode, real on load time | **queued** |
| 5 | `--no-host` | unknown — the help is one line | **queued** |
| 8 | `--no-repack` | zero at `65+0`, non-zero at 33+32 | **queued** |
| 8 | `--no-op-offload` | small | **queued** |
| 11 | `--prio 2` · `--poll 0` | small, matters only under contention | **queued** |
| 12 | `--backend-sampling` | small speed gain; **must check greedy equivalence** | **queued** |
| 6 | `--swa-full` · `--no-kv-unified` | unknown — Qwen3.8 is hybrid Gated-DeltaNet, so whether SWA is even meaningful must be checked, not assumed | **queued** |

---

## 4. Tier 4 — needs an artifact, a build, or the developer

| layer | item | blocker |
|---|---|---|
| 3 | `--lora` / `--control-vector` | no adapter or vector exists for this model. `--lora-init-without-apply` would give **within-boot A/B**, which is worth more than its raw effect against a 13.6 % drift floor |
| 15 | llama.cpp newer than `60eeeb608`; `ik_llama.cpp`; DFlash 2 (**PR #27342**, unmerged) | a source build, which unpins the baseline every number in reports 00–19 was measured on. Do it as a *second* binary with its own re-measured control, never in place |
| 16 | HAGS, Windows power plan, `CUDA_MODULE_LOADING` | host settings; and **desktop VRAM is a live variable** — 33 processes held 2,202 MiB during these runs, which is what pushed `AD-IQ1_M` off residency |
| 2 | requantize with a code-focused imatrix | high host RAM; better to pick a publisher who already did it |

---

## 5. Already answered, and one answered for free today

| layer | question | answer |
|---|---|---|
| 1 | which model | 20 artifacts, 8 families — report 15 |
| 2 | which quantization | report 15; **and see below** |
| 6 | which KV type | only `f16`/`bf16`/`q8_0`/`q4_0` have a fast kernel; the rest collapse 7× in prefill |
| 6 | `--no-kv-offload` | reaches `65+0` and is **still slower** — PCIe per token |
| 8–11 | `-t`, `-tb`, `-b`, `-ub`, `--fit-target` | swept and settled, below the floor. **Do not re-sweep** |

### 5.1 Free result: which V3 artifacts kept the MTP head

Report 06 §0 item 7 asked whether `UD-Q2_K_XL` really kept its head, since it
loads 66 layers where smaller V3 arms load 65. Answered by grepping tensor names
out of loader logs already on disk — **no GPU time at all**:

```text
v3-q2kxl        blk.64.attn_q  blk.64.attn_k  blk.64.attn_v  blk.64.attn_output …   HEAD PRESENT
v3-iq2xxs       (no blk.64 tensors)                                                 HEAD REMOVED
v3-iq1s         (no blk.64 tensors)                                                 HEAD REMOVED
pre-V3 iq2xxs   blk.64.attn_q  blk.64.attn_k  …                                     HEAD PRESENT
```

`n_layer = 64` in every case; `blk.64` is the MTP head carried as a 65th block.

**Unsloth's documentation says the head was removed from "Q2_K_XL and smaller".
That is imprecise: `Q2_K_XL` kept it; removal starts at `IQ2_XXS`.** This also
confirms report 14's residual note — the pre-V3 file pays VRAM for a `blk.64`
tensor whether or not speculation is driven — and it is why the standalone 1.28
GiB drafter is required for any MTP experiment on `IQ2_XXS` or smaller.

---

## 6. Execution chain, as armed

```text
afk-qwen38-resident.sh    Stage A ceiling ✓   Stage B tg@128K (running)   Stage C levers
        ↓
afk-q38-ckpt.sh           does --ctx-checkpoints 8 buy residency?          ~15 min
        ↓
afk-q38-layers.sh         16K screen: kernel / ngram / placement+CPU       ~50 min
        ↓
afk-q38-depth-levers.sh   MTP placement · KV VRAM · prefix+window ·        ~90 min
                          -ot and P-core mask at 163,840
        ↓
afk-q38-quality.sh        grammar corpus, iq2xxs → iq1m → iq1s,
                          STOP at the first ≥80 % accepted and ≤10 % contract violations
```

**Why the 16K screen comes before the depth sweep:** a 16K boot costs about a
minute against three at 128K, and 19 levers × 2 rounds is 38 boots. Depth-specific
levers are excluded from it deliberately — at 16K the cache is ~288 MiB and every
one of them is about the cache.

**Why quality comes last:** a corpus run is 30–90 minutes. Spending one before
the lever picture is complete risks measuring an arm we would have configured
differently.
