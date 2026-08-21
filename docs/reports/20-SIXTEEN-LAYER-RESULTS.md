# The 16-Layer Programme — Measured Results

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **Date:** 2026-08-20 UTC+7 · **Target:** Qwen3.8-27B, fully GPU-resident at
> ≥128K, highest tok/s
> **Plan:** [`../plans/03-SIXTEEN-LAYER-PROGRAMME.md`](../plans/03-SIXTEEN-LAYER-PROGRAMME.md)
> **Surface:** [report 16](16-OPTIMIZATION-SURFACE.md) · **Governing result:** [report 19](19-THE-128K-PLATEAU.md)
>
> Twenty-one levers measured in one session, each paired within its own sweep
> against a control that ran in the same rounds. Every flag was verified to
> parse against build 10472 `60eeeb608` before any GPU time was spent on it.
>
> **Reading rule:** a result is `RESOLVED` only when the effect clears the
> **13.6 %** restart-drift floor *and* keeps its sign across both rounds.
> Anything else is noise, however large its mean looks.

---

> **Correction, 2026-08-21.** `output_contract_pct` is the **pass** rate —
> `100 * (attempts_seen - contract_violations) / attempts_seen` — not the violation
> rate. Text written on 2026-08-20 read it backwards. The figures are unchanged;
> their direction is. Higher is better.

## 0. The headline

**Tuned n-gram speculative decoding more than doubles decode at 16K for zero
VRAM and byte-identical output.** The win came from the layer this project had
searched least — and had assumed, for two days, that it already understood,
having tuned exactly one of its eleven options.

| | count | levers |
|---|---:|---|
| **Real gains** | 6 | five `ngram-*` decoders (tuned; the defaults are erratic — §1.3) and `draft-simple` |
| **Depth-dependent** | 3 | `draft-mtp` — **+81 % at 16K, −71 % at 128K** (§1.5) |
| **Real losses** | 2 | mixed KV, `draft-simple` on CPU |
| **Harmful in a way the number hides** | 2 | `-np 2` divides the context, `-fa off` will not load |
| **Confirmed inert** | 10 | see §5 |
| **Measured with the wrong instrument** | 2 | `--cache-reuse`, `--context-shift` |
| **Cannot run** | 3 | `draft-dflash` (tensor mismatch), `draft-eagle3`, `draft-dspark` (no checkpoint) |

**Two findings are about the instrument rather than the machine**, and both cost
a published claim:

1. The timed generations ran at `temperature 0.7`, so a content-dependent lever's
   measurement followed the text it happened to write. `ngram-cache` returned
   **+80.79 %** and **−30.56 %** in two sweeps three hours apart, both marked
   `RESOLVED` (§1.3). Fixed with `--fixed-text`.
2. Reading a lever's verdict at one depth and assuming it holds at another. MTP
   inverts completely between 16K and 128K (§1.5).

---

## 1. Layer 9 — the decoders. The result of the day, and the correction

> **Corrected 2026-08-20 23:30.** The first version of this section named
> `ngram-map-k` the winner at +94.69 % and reported MTP as losing everywhere.
> Re-running both, three hours later, refuted the first and inverted the second.
> The original numbers are kept below with what replication did to them, because
> a result that does not replicate is itself a finding about the instrument.

`v3-iq2xxs`, 16,384 context, q4_0 KV, two rounds, order reversed in round 2.

| decoder | tok/s | Δ vs control | per-round | verdict |
|---|---:|---:|---|---|
| control (`none`) | **41.16** | — | — | baseline |
| **`ngram-map-k`** | **86.73** | **+94.69 %** | +110.71, +78.67 | **RESOLVED** |
| ~~`ngram-cache`~~ | 74.22 / 74.54 | +80.79 % | +80.32, +81.27 | **DISQUALIFIED** -- changes the greedy hash, see 1.1 |
| `ngram-mod` | 66.75 / 65.85 | **+61.16 %** | +62.17, +60.14 | **RESOLVED** |
| `ngram-simple` | 42.39 / 82.09 | +51.31 % | +2.99, +99.64 | **RESOLVED** |
| `ngram-map-k4v` | 64.19 / 55.83 | +45.86 % | +55.95, +35.77 | **RESOLVED** |

**All five clear the floor. `ngram-map-k` more than doubles decode.**
**`ngram-cache` is nonetheless disqualified: it does not return the same
answer.** Section 1.1 has the correction.

### 1.1 It is free in every sense that matters here

```text
greedy_hash   control        04E5CAB1D14525C0
              ngram-map-k    04E5CAB1D14525C0
              ngram-cache    3EFE93950A8A980E   <-- DIFFERENT
              ngram-mod      04E5CAB1D14525C0
              ngram-simple   04E5CAB1D14525C0
              ngram-map-k4v  04E5CAB1D14525C0
```

> **Correction, 2026-08-21.** The `ngram-cache` line above originally read
> `04E5CAB1D14525C0` and this section claimed byte-identical output for all
> **five** arms. That was a transcription error, not a measurement: the row in
> `results/kv-decoders.jsonl` said `3EFE93950A8A980E` at the time it was
> written, and the independent `--fixed-text` sweep on 2026-08-21 returned the
> same `3EFE9395...` again, in four separate boots. The hash block was typed by
> hand instead of read from the data, and it certified as safe a decoder that
> is not. **`ngram-cache` changes the answer and must not ship** -- report 23
> section 1. The other four arms are unaffected and now verified twice.

**Four of five produce byte-identical output.** Same answer, sooner. Plus:

- **no VRAM cost** — free VRAM unchanged (2,759–2,794 MiB against the control's
  2,780), layer split `65+0` on every arm
- **no drafter file** — nothing to download, unlike MTP's 1.28 GiB or DFlash 2's
  1.06 GiB
- **prefill untouched** — 1,189–1,195 tok/s against the control's 1,189

### 1.2 Why it wins here and MTP does not

An n-gram decoder replays token sequences that already appeared in the context.
Code is the best case: identifiers, `self.`, indentation, and the block just
written all repeat. MTP predicts from a *learned head*, which must be held in
memory.

On a 12 GB card that difference decides everything. MTP's head competes with the
layers; the n-gram table lives in host memory and competes with nothing. **The
mechanism that has dominated every result in this project for two days simply
does not apply to it.**

### 1.3 The ranking did not replicate. The instrument was at fault

The same sweep, same artifact, same depth, three hours later:

| decoder | sweep 1 (20:08) | sweep 2 (23:08) | min round | verdict |
|---|---|---|---:|---|
| `ngram-mod` | +60.14, +62.17 | +63.69, +88.15 | +60.14 | all four rounds well clear |
| `ngram-map-k4v` | +35.77, +55.95 | +18.50, +60.02 | +18.50 | all four positive |
| `ngram-simple` | +2.99, +99.64 | +112.37, +53.03 | **+2.99** | one round below the floor |
| `ngram-map-k` | +110.71, +78.67 | **+3.68**, +135.78 | **+3.68** | one round below the floor |
| `ngram-cache` | +80.32, +81.27 | **−35.85, −25.27** | −35.85 | **SIGN REVERSED** |

`ngram-cache` went from **+80.79 % to −30.56 %**, and *both* sweeps were marked
`RESOLVED` — because the sign was consistent **within** each sweep.

**Cause, found in the probe:** the timed generations run at `temperature 0.7`
(`depth_sweep.py:189, 210`). Every round therefore writes *different text*, and
n-gram speculation replays sequences already present in the context — so its hit
rate follows the text, not the hardware.

The `greedy_hash` values that matched across all five arms came from a
**separate** request at `temperature 0.0, seed 42` (line 219). Reading those two
as one thing was the mistake; the hashes prove the decoders are lossless, not
that the timing was stable.

**This is a hole in the resolution criterion, not just in one lever.** The 13.6 %
floor was measured from boot-to-boot VRAM drift. It cannot see variance whose
source is the content being generated. Any two-round result on a
content-dependent lever carries the same risk.

**Fix applied:** `depth_sweep.FIXED_TEXT` / `kv_sweep --fixed-text` pins
temperature 0 and a fixed seed for the timed generations, and `fixed_text` is
recorded on every row so a number cannot be read without knowing which mode
produced it. Default is unchanged, so no existing result is invalidated. A
four-round `--fixed-text` re-run is queued.

### 1.4 The n-gram defaults are what made it erratic

Shortening the lookup length turned the least stable lever into the most stable:

| arm | knobs | Δ | per-round spread |
|---|---|---:|---|
| `ngram-map-k4v` **tuned** | `size-n 6`, `size-m 24` (defaults 12, 48) | **+114.50 %** | +113.26, +115.74 — **2.5 points** |
| `ngram-mod` **tuned** | `n-match 12`, `n-min 16`, `n-max 32` (defaults 24, 48, 64) | **+109.70 %** | +107.55, +111.86 — **4.3 points** |
| `ngram-simple` tuned | `size-n 6`, `size-m 24` | +52.63 % | −0.32, +105.58 |

A shorter lookup n-gram matches far more often, so the hit rate stops depending
on whether a long exact repeat happened to occur. **The defaults are too long for
this workload**, which explains both the erratic default arms and the tight
tuned ones in one mechanism.

`scripts/production-iq2xxs-ngram.ps1` now serves `ngram-mod` with
`--spec-ngram-mod-n-match 12`, not `ngram-map-k`, and records the correction.

### 1.5 MTP inverts with depth — it wins at 16K and loses at 128K

`v3-iq2xxs`, with the standalone 1.28 GiB drafter (V3 removed the built-in head
from `IQ2_XXS` and smaller — see §6.1). **Same flag, same artifact, opposite
verdict:**

| arm | 16,384 | 131,072 |
|---|---:|---:|
| `mtp-gpu` | **+81.48 %** (64.13 tok/s) | **−71.36 %** |
| `mtp-otd-cpu` (`-otd .*=CPU`) | +18.41 % (46.07) | −59.27 % |
| `mtp-cpu` (`--spec-draft-device none`) | −5.08 % (38.07) | −58.03 % |

**It is prefill, and it always was.** At 16K prefill is 10.4 s and the drafter
pushes it to 10.9 — half a second, against decode 64 % faster. At 131,072
prefill is 120 s and the drafter pushes it to **206 s**: 86 seconds lost before
the first character, which decode cannot earn back.

The split confirms it. `mtp-gpu` reaches **`66+0` at both depths** — nothing is
displaced, and at 128K the cache even *shrinks* to 1,872 MiB. VRAM is not the
mechanism here.

**This corrects report 15 §2.1**, which said MTP fails on a resident target
because the head's VRAM displaces layers. That explanation fits the original
`Q2_K_XL` measurement — that arm sat at `61+4` and the drafter pushed it to
`55+10`, so displacement was real *there*. It does not fit a target that is
already `65+0`, where the head costs prefill instead.

**So MTP pays when both hold:** prefill is short enough that the added cost is
smaller than the decode gain, **and** there is VRAM slack the drafter can take
without evicting layers. At 16K on `IQ2_XXS` both hold. At 128K neither does.

Even where it wins, it is beaten: tuned n-gram returns **+109 % to +114 %** at
the same depth for **zero VRAM**, against MTP's +81 % for 1,195 MiB.

#### The 131,072 detail

| arm | split | KV MiB | prefill | tok/s | Δ |
|---|:--:|---:|---:|---:|---:|
| control | 65+0 | 2,304 | 120–127 s | 22.63 / 21.45 | — |
| `mtp-gpu` | **66+0** | 1,872 | **206 s** | 6.81 / 5.83 | **−71.36 %** |
| `mtp-cpu` (`--spec-draft-device none`) | **0+66** | 2,304 | **541–595 s** | 9.32 / 9.17 | **−58.03 %** |
| `mtp-otd-cpu` (`-otd .*=CPU`) | 66+0 | 2,016 | 153–161 s | 8.91 / 9.03 | **−59.27 %** |

| arm | split | KV MiB | prefill | tok/s | Δ |
|---|:--:|---:|---:|---:|---:|
| control | 65+0 | 2,304 | 120–127 s | 22.63 / 21.45 | — |
| `mtp-gpu` | **66+0** | 1,872 | **206 s** | 6.81 / 5.83 | **−71.36 %** |
| `mtp-cpu` (`--spec-draft-device none`) | **0+66** | 2,304 | **541–595 s** | 9.32 / 9.17 | **−58.03 %** |
| `mtp-otd-cpu` (`-otd .*=CPU`) | 66+0 | 2,016 | 153–161 s | 8.91 / 9.03 | **−59.27 %** |

Two flag notes that cost real time to learn:

- **`--spec-draft-device none` does not mean "drafter on CPU".** It produced
  `0+66` — the *entire target* moved to the CPU, and prefill went to 541–595 s.
  The flag disables offload globally, not for the draft model alone. **`-otd
  ".*=CPU"` is the flag that does what was intended** — it keeps the target at
  `66+0` and moves only the drafter's tensors, for 467 MiB instead of 1,195.
- The external research predicted **+70 % to +85 %** from exactly this move
  (report 18 §3). Measured at depth: **−59 %**.

### 1.6 `draft-simple` works, and is priced out of the target

A same-family drafter (`empero-ai/Qwen3.8-2B-Distill`, Q4_K_M, 1.22 GiB) loads
with a matching vocabulary and delivers **+37.91 %** at 16K, per-round +35.51 and
+40.31 — **tighter than any default n-gram arm**, because it predicts from its
own weights and does not care what text is being written.

It costs **1,452 MiB** (free VRAM 3,126 → 1,674). At 16K that is affordable. At
131,072 the resident arm has ~800 MiB spare, so the drafter cannot be loaded
without pushing the target off `65+0` — the same wall MTP hits.

`--spec-draft-device none` on it reproduces the global-offload behaviour: `0+26`,
prefill 10 s → 70 s, **−61.83 %**.

### 1.7 `draft-dflash` — the stock loader rejects DFlash 2, and says why

```text
print_info: general.name = Qwen3.8-27B-DFlash2
llama_model_load: error loading model: done_getting_tensors:
    wrong number of tensors; expected 81, got 58
```

The file reads and its metadata parses; the **tensor layout does not match** what
`draft-dflash` expects. DFlash 2 is a different architecture, not a new
checkpoint of the same one, so
[PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) has to change the
loader — adding a name to a list would not be enough.

**Cost of this answer: one boot, four seconds.** The alternative was a source
build of an unmerged PR on an assumption.

`draft-eagle3` and `draft-dspark` remain unrunnable — no checkpoint exists for
Qwen3.8 in either ecosystem.

---

## 2. Layer 6 — KV cache

### 2.1 Mixed K/V types have no kernel. This is now measured, not suspected

`v3-iq2xxs`, 16,384, two rounds:

| arm | pp tok/s | prefill | tok/s | KV MiB | Δ |
|---|---:|---:|---:|---:|---:|
| `-ctk q4_0 -ctv q4_0` | **1,119** | 10.4 s | 39.49 | 288 | — |
| `-ctk q8_0 -ctv q4_0` | **39** | **299 s** | 9.25 | **416** | **−76.69 %** |
| same + `--ctx-checkpoints 8` | 40 | 288 s | 9.29 | 416 | −76.69 % |

**Prompt processing collapses 29×.** Worse than `q5_1` (144–170 tok/s), which
this project had already ruled out. And the cache is **44 % larger**, because
`q8_0` K is bigger than `q4_0` K.

Discovered the expensive way first: the same arm at **131,072** spent **65
minutes at 2–20 % GPU** on a prefill its symmetric neighbour finished in 105.6 s,
then died on a socket timeout. It was moved to 16K afterwards, where a fallback
kernel costs a minute instead of an hour.

**This was the external research's headline KV recommendation** — keep K at
`q8_0` for positional precision, drop V to `q4_0`, "~25 % VRAM saved". The
mechanism is plausible and the flag parses. It is unusable here for two
independent reasons, and the second is arithmetic: the research's baseline was
`q8_0/q8_0`. This project runs `q4_0/q4_0`, which is smaller than both.

**Our own kernel screen shares the blame.** It set `-ctk` and `-ctv` together on
every row and concluded "use f16, bf16, q8_0 or q4_0" without recording that the
pair must match. Both sides missed the same assumption.

### 2.2 `--ctx-checkpoints 8` frees nothing

Four artifacts, ceiling probe (split only, ~1 min per boot), against the
default of 32:

| artifact | ctx | with the flag | without |
|---|---:|---|---|
| `AD-IQ1_M` | 131,072 | 65+1, **322 MiB** free | 65+1, 338 MiB free |
| V3 `IQ1_S` | 196,608 | 65+0, still stops there | same |
| V3 `IQ1_M` | 163,840 | 65+0, still stops there | same |
| V3 `IQ2_XXS` | 131,072 | 65+0, still stops there | same |

And by throughput at 131,072 on V3 `IQ1_S`: **1,794 MiB free with the flag
against 1,784 without — a 10 MiB difference**, and 28.41 vs 28.67 tok/s (0.9 %).

The research claimed **~900 MiB recovered**. It is off by roughly **50×**.

Likely mechanism: checkpoints are allocated on use, not reserved at boot. An
append-only agent never rewinds, so lowering a ceiling that is never reached
frees nothing.

**Scope:** this says the flag does not free VRAM at boot. It does not say the
flag is irrelevant to a client that rewinds context mid-session.

### 2.3 `--swa-full` and `--no-kv-unified`

Both inconsistent in sign at 131,072 (+9.58 % and +9.67 %, ranges +0.44..+18.72
and +0.24..+19.10). The wide ranges are the control drifting — it returned 20.94
tok/s in round 2 while every other arm returned 24.86–25.04.

`--swa-full` reported `KV self size = 2,304.0 MiB`, **identical to the control**.
Qwen3.8 is hybrid Gated-DeltaNet; the flag has nothing to act on.

---

## 3. Layer 4 — tensor placement. The lever that reaches the mechanism

### 3.1 At 163,840, `-ot` does what `--fit` cannot

`v3-iq2xxs` at 163,840 sits at `62+3` under `--fit` — three layers on the CPU.

| arm | split | tok/s | free MiB | Δ |
|---|:--:|---:|---:|---:|
| control (`--fit` alone) | **62+3** | 14.93 / 16.70 | 382 / 473 | — |
| `-ot blk.(50-64).ffn_.*=CPU` | **65+0** | 14.01 / 13.95 | **1,310 / 1,407** | −11.31 % |
| `-ot blk.(50-64).ssm_.*=CPU` | **65+0** | 16.83 / 15.87 | 381 / 353 | +3.88 % |

**Both restore full residency.** `--fit` treats a layer as indivisible: when
VRAM runs short it evicts the whole thing. `-ot` moves only the named tensors,
so the layer stays.

`ot-ssm-tail` reaches `65+0` **at no measurable cost** (+3.88 %, sign
inconsistent = unchanged). `ot-ffn-tail` frees far more VRAM — 1,310–1,407 MiB —
and pays 11 % for it.

At 16,384, where the model is already `65+0`, the same flags can only cost:
`ot-ffn-tail` −61 % (15.51 vs 40.20 tok/s, freeing 1,234 MiB) and `ot-ssm-tail`
−19 % (32.59, freeing 168 MiB). That is the expected shape and it calibrates the
price per megabyte.

**Caveat that matters:** both `-ot` arms produced `greedy_hash 3EFE93950A8A980E`
against the control's `04E5CAB1D14525C0`. **Moving tensors to the CPU changes
the output** — CPU and GPU floating-point do not agree bit-for-bit. Unlike the
n-gram decoders, this is not a free speedup; it is a different computation.

### 3.2 `-sm tensor` is inert, as predicted

40.56 tok/s against the control's 40.20 at 16K, identical greedy hash. The
loader could not even report a layer split (`gpu=None/None`). One GPU, nothing
to split. **Prediction confirmed.**

---

## 4. Layers 8, 10 — two flags whose numbers hide what they do

### 4.1 `-fa off` will not load at all

```text
llama_init_from_model: quantized V cache requires flash_attn to be enabled
srv llama_server: exiting due to model loading error
```

Both rounds. **Flash attention is not a choice on this machine — it is a
precondition of using a quantized KV cache**, which this project does on every
run.

This closes a validity concern from report 16 §9: every cross-artifact comparison
here assumed `-fa auto` resolved the same way on every arm, and nobody had read
the loader's decision. It *must* resolve the same way, because the alternative
does not start.

### 4.2 `-np 2` is harmful, not inert

```text
slots = 2   n_ctx_slot = 8192
error: request (11663 tokens) exceeds the available context size
HTTP Error 400: Bad Request
```

**`-np N` divides the context between slots.** Asking for 16,384 with two slots
gives each slot 8,192, and the probe's 11,663-token prompt no longer fits.

At the 131,072 target, two slots would give each **65,536** — abandoning the goal
outright. Report 16 predicted "inert at one stream, and each slot costs KV". The
truth is stronger: it does not merely waste, it removes the depth.

This also killed the step it was in, so `pcore-mask`, `prio-high`, `poll-0` and
`backend-samp` never ran.

> **Resolved 2026-08-21 (report 23 section 4).** All four ran, two rounds at
> 16,384 with `--fixed-text`: `pcore-mask` +0.46 %, `prio-high` -2.02 %,
> `poll-0` +0.69 %, `backend-samp` +2.27 %. **All four inert.** Thread affinity,
> process priority, the polling strategy and GPU-side sampling do nothing on
> this workload. The placement group of the sixteen-layer surface is closed.

---

## 5. Confirmed inert — the predictions that held

All at 16,384 on `v3-iq2xxs`, two rounds, against the same control.

| lever | Δ | per-round | report 16 predicted |
|---|---:|---|---|
| `--no-op-offload` | +4.83 % | +1.74, +7.91 | small |
| `--load-mode none` | +2.71 % | +2.28, +3.15 | zero on decode |
| `--no-host` | +2.31 % | +1.00, +3.63 | unknown |
| `--no-repack` | −1.73 % | −5.13, +1.66 | zero at `65+0` |
| `-sm tensor` | ~0 | — | inert, one GPU |
| `--ctx-checkpoints 8` | +1.47 % | −16.64, +19.58 | medium at depth — **wrong** |
| `--no-kv-unified` | +9.67 % | +0.24, +19.10 | small at one slot |
| `--swa-full` | +9.58 % | +0.44, +18.72 | unknown — arch may not use SWA |

Every one is below the floor or inconsistent in sign. `--no-op-offload` has the
highest mean with a consistent sign, and 4.83 % is still a third of the floor.

**Writing the predictions down first is what makes this table useful.** Seven of
eight held; `--ctx-checkpoints` did not, and that is the one worth remembering.

---

## 6. Two answers that cost no GPU time at all

### 6.1 Which V3 artifacts kept the MTP head

Report 06 §0 item 7 had been open for two days. Answered by grepping tensor
names out of loader logs already on disk:

```text
v3-q2kxl        blk.64.attn_q  blk.64.attn_k  blk.64.attn_v  blk.64.attn_output …   PRESENT
v3-iq2xxs       (no blk.64 tensors)                                                 REMOVED
v3-iq1s         (no blk.64 tensors)                                                 REMOVED
pre-V3 iq2xxs   blk.64.attn_q  blk.64.attn_k  …                                     PRESENT
```

Confirmed independently by the parameter counts the loader prints: **26.90 B**
for the V3 arms without the head, **27.32 B** for those with it — a difference of
**0.42 B parameters**, one whole block.

**Unsloth's documentation says the head was removed from "Q2_K_XL and smaller".
That is imprecise: `Q2_K_XL` kept it; removal starts at `IQ2_XXS`.** This is why
any MTP experiment on `IQ2_XXS` or smaller needs the standalone drafter.

### 6.2 The quantization names do not describe the files

Same source — the tensor-type histogram the loader prints:

| file | GiB | params | **real bpw** | dominant types |
|---|---:|---:|---:|---|
| V3 `UD-IQ1_S` | 5.77 | 26.90 B | **1.84** | iq1_s 264 · iq1_m 25 · q8_0 96 |
| V3 `UD-IQ1_M` | 6.27 | 26.90 B | 2.00 | — |
| V3 `UD-IQ2_XXS` | 6.77 | 26.90 B | 2.16 | iq2_xxs 143 · iq1_s 100 · **q8_0 0** |
| `AD-IQ1_M` | 7.91 | 27.32 B | **2.49** | iq1_m 80 · iq2_s 160 · **q8_0 128** |
| pre-V3 `UD-IQ2_XXS` | 8.39 | 27.32 B | 2.64 | iq2_s 208 · iq1_m 96 |

**The file named `IQ1_M` is heavier than the file named `IQ2_XXS`** — 2.49 bits
per weight against 2.16. Only 80 of `AD-IQ1_M`'s tensors are actually `iq1_m`;
**128 of them are `q8_0`**, full 8-bit. V3 `IQ2_XXS` has no `q8_0` tensors at all.

Within one publisher and generation the ordering is exactly right (V3: 1.84 →
2.00 → 2.16). Across publishers it inverts.

**This retires a rule this project had been using:** "pick the smallest artifact
that still fits" assumed the name tracked the size. It does not, across vendors.
It also explains why `AD-IQ1_M` produces the best corpus of any 1-bit-named
artifact here — it is not a 1-bit model.

---

## 7. Measured with the wrong instrument — recorded so the rows are not misread

| lever | Δ at 131,072 | what the number actually means |
|---|---:|---|
| `--cache-reuse 256` | +9.62 % | **The probe never breaks a prefix.** It sends the same prompt with a short suffix — a case ordinary prompt caching already handles. This flag exists for KV shifting after a *mid-stream edit*, which was never exercised |
| `--context-shift --keep 2048` | −1.29 % | **The probe never fills the window.** Context shift only acts when the context is exhausted, and it never was |

Both read as "costs nothing to leave on". Neither says whether it helps.

Measuring them properly needs `stability_gate.py`, which forces a prefix
invalidation every tenth turn — the situation `--cache-reuse` was built for, and
where this project has measured the cost it should attack: **one broken prefix
costs 63 s at 16K and 248 s at 64K**.

---

## 8. What this changes

**Adopt now, at 16K:** `--spec-type ngram-map-k`. Doubles decode, identical
output, no VRAM, no download. Written up as
`scripts/production-iq2xxs-ngram.ps1`. *Not yet verified at depth or on the
pre-V3 artifact the production profile actually serves.*

**Stop considering:** mixed KV, `--ctx-checkpoints`, MTP in any placement,
`-np > 1`, `-fa off`, `-sm tensor`, `--no-repack`, `--no-op-offload`,
`--load-mode`, `--no-host`, `--swa-full`, `--no-kv-unified`.

**Promote, with the route to `AD-IQ1_M` struck out:** `-ot` on `ssm_*` tensors
restored `65+0` at 163,840 at no measurable throughput cost, which `--fit` could
not do. That result stands. Watch the greedy hash: it changes the output.

> **Correction, 2026-08-21 (report 23 section 2).** This paragraph originally
> called `-ot` *"the most direct route to `AD-IQ1_M` reaching 128K"*. That route
> is now either dead or untested, depending on the slice:
>
> - **`-ot` on `ffn_*` is dead there.** It frees the layer -- 644 MiB moved to
>   CPU, exactly as intended -- and prefill collapses from 240.6 to **8.56
>   tok/s**, twenty-eight times slower. A three-hour prefill per round.
> - **`-ot` on `ssm_*` was never reached** on that artifact; the queue died
>   first. It moves only ~168 MiB, so it may behave nothing like the FFN slice.
>   **Untested, not refuted.**
>
> And the premise was worse than reported anyway: `AD-IQ1_M` at `65+1` decodes
> at **6.08 tok/s** with a 386.9 s prefill. Freeing one layer was never going to
> make it a 128K artifact on its own.

**Unchanged:** the decision at 128K is still quality, not speed (report 19), and
the blocking failure is still format — only 41.5 % to 58.3 % of corpus attempts emit
no fenced code block. Doubling decode makes those attempts fail twice as fast.

---

## 9. Still open

- **Does the n-gram win survive to 131,072?** Prefill there is 110–127 s and
  speculation cannot touch it, so the wall-clock gain per task must be smaller
  than the tok/s figure suggests. Queued.
- **Does a GBNF grammar fix the format failure?** The largest open question in
  the project. Queued.
- **Can a graded `-ot` buy `AD-IQ1_M` its one missing layer at 131,072?** Queued
  with four slice sizes.
- **`draft-simple` and `draft-dflash`** — never run. Both drafters are now on
  disk (2B distill, 1.22 GiB; DFlash 2, 1.06 GiB). Queued.
- **`draft-eagle3` and `draft-dspark`** — no checkpoint exists for Qwen3.8.
- **Deep retrieval quality on anything but Q4** — still zero measurements, still
  the top item in report 06 §0, now three days running.
