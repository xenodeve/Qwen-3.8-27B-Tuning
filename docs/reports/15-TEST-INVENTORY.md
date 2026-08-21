# Test Inventory — Every Model, Every Quantization, Every Probe

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **Date:** 2026-08-20 UTC+7 · generated from `results/*.jsonl`, not from memory
>
> **What this is:** reports 00–14 each answer a question. This one answers
> *"what has actually been run?"* — one row per artifact, one column per probe,
> so a gap is visible as a blank cell rather than by reading eleven documents.
>
> **Two artifact generations are mixed below and are NOT comparable.** Unsloth
> republished `unsloth/Qwen3.8-27B-GGUF` in place on 2026-08-19T16:39:23Z with
> identical filenames. Rows marked **V3** are the new commit `27af057e`; every
> other Unsloth row is the old `f1bfb127`. Byte counts are pinned in
> `bench/depth_sweep.py`.

---

## 1. Every artifact measured, by decode speed

16K context, three greedy code generations per boot, median of medians. `Δ` is
**paired within its own arena file** against the control named in §1.1 — never
across files, because free VRAM at boot moves 9,326–10,732 MiB and `--fit`
follows it (report 04 §0).

| artifact | family | quant | GiB | tok/s | Δ vs its control | split | free MiB | pp tok/s |
|---|---|---|---:|---:|---:|:--:|---:|---:|
| `Bonsai-27B-Q1_0` | Bonsai 27B | Q1_0 ternary | 3.54 | **69.33** | +80.1 % | 65+0 | 5,716 | 987 |
| `Ornith-35B-A3B` | Ornith MoE | UD-IQ2_XXS | 10.71 | 68.37 | +79.5 % | 65+0 | **227** | 766 |
| `Qwen3.6-35B-A3B` | Qwen MoE | UD-IQ2_XXS | 10.02 | 68.09 | +78.5 % | 65+0 | **339** | 811 |
| `gpt-oss-20b` | gpt-oss | MXFP4 | 10.83 | 64.81 | +68.4 % | 25+0 | **363** | 808 |
| `Ornith-1.0-9B` | Ornith 9B | Q6_K | 6.85 | 61.17 | control | 65+0 | 3,932 | 2,129 |
| **V3** `UD-IQ1_S` | Qwen3.8-27B | IQ1_S | 5.77 | 50.55 | +29.3 % | 65+0 | 3,549 | 769 |
| `Ternary-Bonsai-27B` | Bonsai 27B | Q2_g64 | 7.06 | 49.84 | +17.7 % | 65+0 | 2,628 | 996 |
| `AD-IQ1_M` (AtomicChat) | Qwen3.8-27B | IQ1_M | 7.91 | 45.53 | control | 65+0 | 2,117 | 692 |
| `Ornith-1.0-9B` | Ornith 9B | Q8_0 | 8.87 | 44.94 | +16.7 % | 33+0 | 1,814 | 2,010 |
| **V3** `UD-IQ2_XXS` | Qwen3.8-27B | IQ2_XXS | 6.77 | 44.84 | +14.9 % | 65+0 | 2,417 | 745 |
| **V3** `UD-IQ1_M` | Qwen3.8-27B | IQ1_M | 6.27 | 43.75 | +21.3 % | 65+0 | 3,022 | 686 |
| `UD-IQ2_XXS` | Qwen3.8-27B | IQ2_XXS | 8.39 | 41.32 | control | 65/66+0 | 1,072 | 790 |
| **V3** `UD-IQ2_S` | Qwen3.8-27B | IQ2_S | 7.80 | 40.68 | +4.2 % | 65+0 | 1,463 | 740 |
| `AD-IQ2_XXS` (AtomicChat) | Qwen3.8-27B | IQ2_XXS | 8.36 | 39.87 | +4.0 % | 65+0 | 858 | 750 |
| `Ornith-35B` CPU-expert | Ornith MoE | IQ2_XXS `-ncmoe` | 10.71 | 37.17 | −2.6 % | 65+0 | 6,912 | 268 |
| `Qwen3.6-35B` CPU-expert | Qwen MoE | IQ2_XXS `-ncmoe` | 10.02 | 35.77 | −6.2 % | 65+0 | 7,106 | 255 |
| **V3** `UD-Q2_K_XL` | Qwen3.8-27B | Q2_K_XL | 9.15 | 30.99 | −8.5 % | 66+0 | 462 | 742 |
| `UD-Q2_K_XL` | Qwen3.8-27B | Q2_K_XL | 9.94 | 21.84 | −48.6 % | **61+4** | 451 | 487 |
| `UD-Q2_K_XL` + MTP n=2 | Qwen3.8-27B | Q2_K_XL | 9.94 | 19.92 | −53.1 % | **55+10** | 803 | 330 |
| `UD-Q4_K_XL` tuned | Qwen3.8-27B | Q4_K_XL | 16.69 | 13.12 | −68.0 % | **33+32** | 236 | 168 |

**The whole table is one mechanism.** Every row at `65+0` is fast and every row
with CPU layers is slow, and the gap between them is larger than every flag,
every KV type and every speculation setting combined. `Q4_K_XL` at `33+32` is
not slow because it is Q4; it is slow because half of it is in system RAM.

### 1.1 Controls, and why the Δ column cannot be read down the page

| arena file | control | what it was measuring |
|---|---|---|
| `arena-quant` | `iq2xxs-nomtp` | Q4 vs Q2 vs IQ2, MTP on/off |
| `arena-quantizer` | `iq2xxs-nomtp` | Unsloth vs AtomicChat, same nominal quant |
| `arena-iq1` | `iq1m-nomtp` | the first genuine 1-bit 27B |
| `arena-ornith` / `-bonsai` / `-moe` / `-remaining` | `iq2xxs-nomtp` | other model families |
| `arena-v3` | `iq2xxs-nomtp` | Dynamic V3 against the pre-V3 default |

`iq2xxs-nomtp` appears in 23 boots across six files at **38.2–42.5 tok/s**. That
spread *is* the restart-drift floor, measured. Any effect below 13.6 % is noise.

---

## 2. The decoder axis — speculative decoding, measured separately

The runtime exposes **eleven** speculative decoders. `llama-server --help`,
build 10472 commit `60eeeb608`:

```text
--spec-type none, draft-simple, draft-eagle3, draft-mtp, draft-dflash,
            draft-dspark, ngram-simple, ngram-map-k, ngram-map-k4v,
            ngram-mod, ngram-cache
```

**Four have been run. Seven have not.**

| decoder | status | what is known |
|---|---|---|
| `none` | • measured everywhere | the control in every table above |
| `draft-mtp` | • measured, Q4 / Q3 / Q2_K_XL | the only one that ever paid — and only on a CPU-offloaded target |
| `ngram-simple` | • measured, Q4 / Q3 | +1.6 % / +0.8 %, acceptance 31 % — below the noise floor |
| `ngram-mod` | ~ boot only | a preboot env snapshot exists; no paired result |
| `draft-dspark` | ✗ **never ran** | attempted 2026-08-20 00:06, failed to launch — see §2.3 |
| `draft-dflash` | ✗ never attempted | the decoder the research pushes hardest. **DFlash 2 is a separate drafter with an exact Qwen3.8-27B checkpoint — see §2.6** |
| `draft-eagle3` | ✗ never attempted | |
| `draft-simple` | ✗ never attempted | a plain smaller-model drafter |
| `ngram-map-k` / `-map-k4v` / `-cache` | ✗ never attempted | free — no drafter file needed |

### 2.1 MTP — the effect inverts across the residency cliff

`spec-matrix-q4` / `-q3`, at 16K, against `none` on the same artifact:

| target | decoder | n_max | tok/s | Δ | draft acceptance |
|---|---|---:|---:|---:|---:|
| `UD-Q4_K_XL` (33+32) | none | — | 8.24 | control | — |
| `UD-Q4_K_XL` | `ngram-simple` | 4 | 8.37 | +1.6 % | 31 % |
| `UD-Q4_K_XL` | **`draft-mtp`** | 2 | **12.10** | **+46.8 %** | **98 %** |
| `UD-Q4_K_XL` | `draft-mtp` | 3 | 12.03 | +46.0 % | 89 % |
| `UD-Q3_K_XL` | none | — | 9.25 | control | — |
| `UD-Q3_K_XL` | `ngram-simple` | 4 | 9.08 | +0.8 % | 31 % |
| `UD-Q3_K_XL` | `draft-mtp` | 2 | 10.30 | +14.3 % | 96 % |
| `UD-Q3_K_XL` | `draft-mtp` | 3 | 9.92 | +10.1 % | 99 % |
| `UD-Q2_K_XL` (61+4) | none | — | 21.84 | control | — |
| `UD-Q2_K_XL` | `draft-mtp` | 2 | 19.92 | **−8.8 %** | — |

**+46.8 % on Q4 and −8.8 % on Q2_K_XL, from the same flag.** The mechanism is
the residency cliff again, one level up: the draft head costs VRAM, and on
`Q2_K_XL` that VRAM moves the split from **61+4 to 55+10**. Six target layers
leave the GPU to make room for a drafter whose job is to make decoding faster.

So the rule is not "MTP is worth 1.3–1.5×". It is: **MTP pays only when the
target is already CPU-offloaded and the head costs nothing that was not already
lost.** Every artifact this project now recommends sits at `65+0`, which is the
regime where it does not pay.

Acceptance rate is high everywhere — 78–99 % — and is *not* the thing that
decides the outcome. The research's assumption that a good acceptance rate
implies a speedup does not survive contact with a 12 GB card.

### 2.2 The speculative sub-knobs are noise, and the second sweep proves it

Both sweeps ran on Q4 tuned, `draft-mtp`, paired within the sweep:

| knob | sweep 1 | sweep 2 (re-test) |
|---|---:|---:|
| `--spec-draft-n-min 2` | **+11.6 %** | **−0.8 %** |
| `--spec-draft-p-min 0.10` | **+9.8 %** | **−10.1 %** |
| `--spec-draft-p-split 0.25` | +8.8 % | (in combination) −4.3 % |
| `--spec-draft-p-min 0.05` | +5.8 % | not repeated |
| `--spec-draft-n-min 1` | +4.7 % | not repeated |
| `--spec-draft-p-split 0.00` | −1.6 % | not repeated |

**The two largest effects in sweep 1 reversed sign in sweep 2.** Every value sits
inside the 13.6 % restart-drift floor, and acceptance barely moved (83.4–88.4 %)
across all twelve configurations. These knobs are not tuned and should not be.

One thing they *did* establish: `greedy_hash` was `A2F070D5480ADEE4` for all
twelve. **Speculation does not change the output**, only the time to produce it —
which is what makes it safe to leave off.

### 2.3 DSpark was attempted and never actually ran

`arena-bonsai-spec` records one boot of `bonsai-g64` and **zero** of
`bonsai-dspark`. The log says why:

```text
common_speculative_init_result: loading draft model ''
llama_model_load_from_file_impl: exactly one out metadata, path_model, and file must be defined
srv load_model: failed to load draft model, ''
srv llama_server: exiting due to model loading error
```

The drafter path was **empty** — `-hfd repo:tag` had resolved to an empty string,
the same class of failure as the target-model tag matching
(`SESSION-STATE.md` §5 rule 1), one flag later. Round 2 then lost its baseline as well, so the file holds one unpaired
sample and no result.

`bench/model_arena.py:94` now passes an exact path and resolves correctly today:

```text
Ternary-Bonsai-27B-dspark-Q4_1.gguf   1,946,393,568 B   present since 2026-08-19 22:08
```

**The arm has not been re-run since the fix.** "Bonsai + DSpark" is untested, not
disproven.

### 2.4 Two drafters are on disk and neither has produced a number

| file | GiB | fetched | used |
|---|---:|---|---|
| `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | 1.28 | 2026-08-20 01:38 | never |
| `Ternary-Bonsai-27B-dspark-Q4_1.gguf` | 1.81 | 2026-08-19 22:08 | never (§2.3) |

The standalone MTP drafter matters because V3 removed the built-in head from
`Q2_K_XL` and smaller. The −8.8 % result above was measured where the head
displaced six layers — and **V3 `IQ1_S` holds 128K at 65+0 with 1,436 MiB
spare**, which is the one condition under which that result would not carry
over, because 1.28 GiB of drafter would not move the split at all.

### 2.5 What the research claims here, and what is actually verified

> *"Modern decoders (MTP, DFlash, DSpark, EAGLE-3/DFly, and n-grams) are now
> widely supported, promising 2–5× speedups when tuned."*
> — `deep-research-report (2).md` §Executive Summary

| claim | status on this machine |
|---|---|
| the decoders are supported by the runtime | **true** — all but DFly are `--spec-type` values in build 10472 |
| 2–5× speedup | **not observed.** Best measured is **1.47×**, on the slowest artifact. The 3.2× on this machine came from **residency**, not speculation |
| "when tuned" | the tuning knobs reversed sign between two sweeps (§2.2) |
| DFlash / DSpark / EAGLE-3 are better | **unmeasured** — no arm has ever launched one |
| "no exact Qwen3.8 DFlash drafter exists" | **superseded** — true for DFlash v1, false for DFlash 2 (§2.6) |

The gap is worth closing precisely because the research is emphatic about it and
this project has one measured counterexample and no coverage.


### 2.6 DFlash 2 is a different decoder, and an exact Qwen3.8-27B drafter exists

**DFlash 2 is not a version bump of DFlash — it is a different drafter**, from
Inco AI rather than the original Z-Lab checkpoint line. It predicts a whole block
in one pass, keeps the top candidates at *every* position, and traces one path
through them with a lightweight selector; two-tap dynamic convolutions keep the
draft from decaying toward the end of the block. It claims **lossless** decoding:
greedy output identical to the target.

This matters more than the other six unrun decoders, because of a specific fact
this project had recorded the opposite of:

> *"Exact public drafts: Qwen3.6-27B, Qwen3.6-35A3B, Gemma26-A4B, gpt-oss20;
> **no exact Qwen3.8 in current Z-Lab list**"*
> — `Candidate Inference Configurations…md` line 267

That was true of DFlash **v1**. It is **not** true of DFlash 2. Checked against
`/api/models/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main` on 2026-08-20:

| file | bytes | GiB |
|---|---:|---:|
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1,143,006,752 | **1.06** |
| `Qwen3.8-27B-DFlash2-Q8_0.gguf` | 2,056,414,752 | 1.92 |
| `Qwen3.8-27B-DFlash2-BF16.gguf` | 3,860,293,152 | 3.59 |

`base_model: Qwen/Qwen3.8-27B`. Vendor-reported acceptance length 5.13–5.39
across the three draft quants (GSM8K, 8 prompts, target `Q4_K_M`) — note the
Q4_K_M drafter scores **highest**, so the cheap one is not the compromise.

### 2.7 Our pinned binary cannot run it — verified, not assumed

```text
$ llama-server --spec-type draft-dflash2 -m /nonexistent.gguf --port 18080
error while handling argument "--spec-type": unknown speculative type: draft-dflash2

$ llama-server --spec-type draft-dflash  -m /nonexistent.gguf --port 18080
llama_model_load: error loading model: failed to load model from /nonexistent.gguf   <- arg accepted
```

Build 10472, commit `60eeeb608`. `draft-dflash` and `draft-dspark` parse;
`draft-dflash2` does not exist as a type.

**But the flag name is a red herring.** The vendor's own serve command uses
`--spec-type draft-dflash` — the DFlash 2 support in
[PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) *extends the
existing loader* rather than adding a new type, and that PR is **unmerged**.
So the open question is narrow and cheap:

**Does the stock `draft-dflash` loader read a DFlash 2 GGUF, or reject its
architecture?** One boot answers it. If it rejects, the cost becomes a source
build of an unmerged PR — a different order of commitment, and one that would
unpin the binary every number in this project was measured on.

### 2.8 Why this is the highest-value decoder test on the list

The 1.06 GiB Q4_K_M drafter is the only drafter this project has ever had that
is *sized to fit beside a resident target*:

| target | free VRAM at its measured point | room after a 1.06 GiB drafter |
|---|---:|---:|
| **V3** `UD-IQ1_S` at **128K**, 65+0 | 1,436 MiB | ~350 MiB — tight, at the reserve |
| **V3** `UD-IQ1_S` at 16K, 65+0 | 3,549 MiB | ~2,460 MiB |
| **V3** `UD-IQ1_M` at 16K, 65+0 | 3,022 MiB | ~1,930 MiB |
| `Bonsai-27B-Q1_0` at 16K, 65+0 | 5,716 MiB | ~4,630 MiB |

Every previous speculation result on this machine is contaminated by the same
confound: the drafter's VRAM pushed target layers to the CPU, so the measurement
was never *speculation vs no speculation*, it was *speculation vs residency*.
§2.1 is that confound, twice. **These four rows are the first configurations
where the confound does not apply** — the drafter fits in slack that is already
sitting idle.

And the lossless claim is directly checkable with an instrument this project
already has: `bench/greedy_diff.py` compares actual greedy text, not just its
hash. The research already warns that greedy divergence has been reported for
MTP and DSpark against *quantized* targets while BF16 matched — so "lossless"
is a claim to verify locally on the exact quant, not to inherit.


---

## 3. Which probes each artifact actually received

`•` = run · blank = **never run**

| artifact | 16K arena | corpus (30 tasks) | protocol gate | stability 100-turn | KV × depth | depth ladder | deep quality |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `UD-Q4_K_XL` | • | • | • | • | • | • | **•** |
| `UD-Q3_K_XL` | • | | | | | • | |
| `UD-Q2_K_XL` | • | • | • | • | | | |
| `UD-IQ2_XXS` | • | • | • | • | • | • | |
| `AD-IQ2_XXS` | • | | | | | | |
| `AD-IQ1_M` | • | • | • | | • | | |
| **V3** `UD-Q2_K_XL` | • | | | | | | |
| **V3** `UD-IQ2_S` | • | | | | | | |
| **V3** `UD-IQ2_XXS` | • | *queued* | | | | *queued* | |
| **V3** `UD-IQ1_M` | • | *running* | | | | *queued* | |
| **V3** `UD-IQ1_S` | • | • (rejected) | | | | *queued* | |
| `Ornith-1.0-9B` Q6 | • | • | • | • | • | | |
| `Ornith-1.0-9B` Q8 | • | | | | | | |
| `Ternary-Bonsai` Q2_g64 | • | • | • | • | • | | |
| `Bonsai-27B` Q1_0 | • | | | | | | |
| `Qwen3.6-35B-A3B` | • | | | | | | |
| `Ornith-35B-A3B` | • | | | | | | |
| `gpt-oss-20b` | • | | | | | | |

This matrix covers *artifacts*. The **decoder** axis is orthogonal and is in
§2 — seven of eleven decoders have never been launched at all.

**Read the blanks.** The four fastest artifacts on this machine — Bonsai-1bit,
both MoE arms, gpt-oss — have **a speed number and nothing else**. Not one has
been asked to finish a coding task. And **deep-context retrieval quality has one
`•` in the entire column, on Q4**, which has been the top open item for two days.

---

## 4. Corpus — the decision metric

Ten single-file coding tasks × 3 passes, one evidence-assisted retry, executed
and asserted. `accepted` counts a task that ran and passed its tests.

| arm | accepted | censored | wall s | accepted/hr | note |
|---|---:|---:|---:|---:|---|
| `iq2xxs` | 27/31 | 0 | 1,599 | **60.8** | the standing default |
| `ornith9b` @8192 | 23/31 | 0 | 1,653 | 50.1 | 9B, different family |
| `iq2xxs` @8192 | 27/61 | 0 | 2,005 | 48.5 | budget re-run |
| `q2kxl` | 26/32 | 0 | 1,972 | 47.5 | |
| `iq1m` @8192 | 27/31 | 0 | 2,756 | 35.3 | AtomicChat 1-bit — **works** |
| `iq1m` @3072 | 20/31 | 0 | 2,251 | 32.0 | same file, smaller budget |
| `q4-matched` | 27/31 | 0 | 4,009 | 24.2 | same accepted, 2.5× the wall |
| `bonsai-g64` @8192 | 27/31 | 0 | 4,572 | 21.3 | |
| `ornith9b` @3072 | 20/31 | 0 | 1,001 | 71.9 | budget-truncated |
| `bonsai-g64` @3072 | 15/31 | 0 | 2,876 | 18.8 | budget-truncated |
| **V3** `IQ1_S` | 0/6 | 2 | 1,430 | **0.0** | rejected — no fenced block 12/12 |
| **V3** `IQ1_M` | 9/47* | 7 | 3,699 | 8.8 | *in flight; see §4.1 |

**Four arms tie at 27/31 accepted and differ 2.9× in wall clock.** Capability and
throughput are separate axes; `merged_tasks_per_hour` multiplies them and must
not be read as a capability ranking on its own.

**The budget column is a treatment, not a detail.** `bonsai-g64` moved 15→27 and
`iq1m` moved 20→27 on the same weights, purely by raising `max_tokens` from 3072
to 8192. Anything measured at 3072 is a lower bound.

### 4.1 V3 1-bit fails the same way twice

Clean subset of the `v3-iq1m` re-run (the aborted first attempt is separable —
only the post-panel harness writes `per_attempt`):

```text
accepted_of_decided                9/15
attempts emitting no fenced block  17/33   (52 %)
attempts hitting the 8192 cap      15/33   (45 %)
reasoning chars              median 19,280   max 33,871
```

The `NameError: name 'evaluate' is not defined` lines in the log are the
**harness downstream of an empty answer**, not a coding failure. Both V3 1-bit
artifacts run away inside their reasoning block and never close it.

---

## 5. Protocol gate — nested tool call and `tool_call_id` round-trip

| arm | n | tool call | nested schema | round-trip | truncated | reasoning chars (median) |
|---|---:|---:|---:|---:|---:|---:|
| `q4` temp 0 | 10 | 100 % | 100 % | **100 %** | 0 | — |
| `ornith9b` | 15 | 100 % | 100 % | 93 % | 0 | 280 |
| `bonsai-g64` | 15 | 93 % | 93 % | 86 % | 0 | 322 |
| `q4` @4096 | 15 | 80 % | 80 % | 60 % | 3 | **59** |
| `iq2xxs` @4096 | 15 | 93 % | 93 % | 66 % | 1 | 1,023 |
| `q2kxl` @4096 | 15 | 86 % | 86 % | 60 % | 2 | **2,811** |
| `iq1m` | 15 | 100 % | 100 % | 46 % | 0 | 170 |
| `q4` tuned temp 0.7 | 10 | 80 % | 80 % | 70 % | 0 | — |
| `q2kxl` @1024 | 10 | 40 % | 40 % | 40 % | — | — |

The `q2kxl` 40 % row is the **withdrawn** number: `max_tokens 1024` cut the
reply mid-call. The same weights score 86 % at 4096. It is kept here as the
worked example of the failure mode, not as a result.

**Median reasoning spans 59 → 2,811 characters across quantizations of the same
model.** A client tuned to Q4's appetite truncates the others and reads it as a
refusal to call tools.

---

## 6. Stability — 100 turns, prefix deliberately invalidated every tenth

| arm | prefix reuse (median) | empty replies | tok/s |
|---|---:|---:|---:|
| `ornith9b` | 99.3 % | **0**/100 | 59.8 |
| `bonsai-g64` | 99.3 % | **0**/100 | 47.4 |
| `iq2xxs` | 99.2 % | 1/100 | 40.4 |
| `q4-tuned` | 99.1 % | 19/100 | 11.1 |
| `q2kxl` | 99.0 % | **55**/100 | 20.5 |

Prefix reuse is ~99 % everywhere and recovers after every forced invalidation —
that mechanism is settled. **The empty-reply column is not.** 19 on Q4, 55 on
`Q2_K_XL`, 1 on `IQ2_XXS` is not monotonic in quantization, so the obvious
explanation is wrong and no other has been tested.

---

## 7. Context depth — where everything currently dies

`UD-IQ2_XXS` unless noted. `tg` is decode tok/s at that depth.

| ctx | KV type | split | KV MiB | cold prefill | tg tok/s |
|---:|---|:--:|---:|---:|---:|
| 65,536 | q8_0 | **61+4** | 2,040 | 64 s | **15.81** |
| 131,072 | q4_0 | 58+7 | 2,016 | 139 s | 7.84 |
| 131,072 | q4_0 `--no-kv-offload` | 65+0 | 2,304 | 153 s | 5.26 |
| 131,072 | q8_0 | 47+18 | 3,264 | 190 s | 5.01 |
| 131,072 | q8_0 `--no-kv-offload` | 65+0 | 4,352 | 175 s | 3.41 |
| 262,144 | q4_0 | 46+19 | 3,168 | 472 s | **2.29** |
| 262,144 | q8_0 | 31+34 | 4,352 | 658 s | 1.71 |

Other artifacts at 128K:

| artifact | KV | split | KV MiB | prefill | tg tok/s |
|---|---|:--:|---:|---:|---:|
| `Ornith-9B` Q6 | q8_0 | **65+0** | 2,176 | 34 s | **46.6** |
| `Ornith-9B` Q6 | q4_0 | **65+0** | **1,152** | 34 s | 45.9 |
| `Bonsai` Q2_g64 | q4_0 | **65+0** | 2,304 | 93 s | **28.1** |
| `Bonsai` Q2_g64 | q8_0 | 51+14 | 3,536 | 148 s | 4.89 |
| `AD-IQ1_M` | q4_0 | **65+0** | 2,304 | 125 s | **24.0** |

**KV type buys residency, not speed.** On Bonsai, q4_0 vs q8_0 is +474 % — but
only because it moves the split from 51+14 to 65+0. On Ornith-9B, where both
types already fit at 65+0, q4_0 is **1.6 % slower** than q8_0. Same knob,
opposite verdict, decided entirely by which side of the cliff it lands on.

`--no-kv-offload` was expected to buy weight residency at 128K and does reach
65+0 — and is still **slower** than leaving 7 layers on the CPU (5.26 vs 7.84),
because it moves the whole cache across PCIe on every token.

### 7.1 KV kernel screen — only four types have a fast path

| KV type | pp tok/s | tg tok/s |
|---|---:|---:|
| f16 / bf16 / q8_0 / q4_0 | **1,174–1,183** | 39.1–39.9 |
| q5_1 / q5_0 / q4_1 / iq4_nl | **144–170** | 20.8–22.7 |

A 7× prefill collapse. `q5_1` looks like a sensible midpoint on paper and is
unusable in practice. **Use only f16, bf16, q8_0 or q4_0.**

**And only with K and V at the SAME type.** The screen above set `-ctk` and
`-ctv` together and never tested a mixed pair, which left an assumption nobody
had checked. Measured 2026-08-20:

| arm | ctx | prefill | tg |
|---|---:|---|---:|
| `-ctk q4_0 -ctv q4_0` | 131,072 | **105.6 s** | 28.67 |
| `-ctk q4_0 -ctv q4_0 --ctx-checkpoints 8` | 131,072 | 108.0 s | 28.41 |
| **`-ctk q8_0 -ctv q4_0`** | 131,072 | **timed out after 65 min** | — |

The mixed arm sat at 2–20 % GPU for over an hour on a prefill its symmetric
neighbour finished in 105 seconds, then died on a socket timeout. That is the
`q5_1` signature — a fallback kernel, not a hang.

This matters because **asymmetric KV is exactly what the external research
recommended** (report 18 §4.7: keep K at q8_0 for positional precision, drop V
to q4_0, "~25 % VRAM saved"). The mechanism is plausible and the flag parses.
It is still unusable here, because this build has no kernel for the pair.

---

## 8. Deep-context retrieval quality — the one column with almost nothing in it

**Every row below is `UD-Q4_K_XL`.** No other artifact has been probed at depth.

| run | passed | pass rate | verified/hr |
|---|---:|---:|---:|
| 64K v2c, q8_0 KV | 30/30 | 100 % | 63.3 |
| 64K v2c, f16 KV | 30/30 | 100 % | 58.1 |
| 64K v2, q8_0 KV | 30/30 | 100 % | 36.9 |
| 64K v1, q8_0 KV | 18/18 | 100 % | 57.4 |
| 64K v1, f16 KV | 18/18 | 100 % | 51.8 |
| 128K (114,406-token prompt), q8_0 KV | 10/10 | 100 % | 23.1 |
| 128K, f16 KV | 10/10 | 100 % | 18.1 |

Q4 retrieves perfectly at both depths across every probe class the corpus has —
25 %, 50 %, 95 % position, multi-hop, aggregate and distractor. **That result
belongs to Q4 and has never been shown to transfer.** The documented failure mode
of aggressive low-bit builds in this family is *selective*: aggregate scores hold
while long-span and reasoning-heavy tasks collapse.

---

## 9. A cheap signal, and the limit of it

`greedy_hash` is the hash of a fixed greedy completion at temperature 0.
Within the Qwen3.8-27B family the canonical value is `227749403A7404D4`:

| hash | artifacts |
|---|---|
| `227749403A7404D4` | `Q4_K_XL`, `Q2_K_XL` (±MTP), `UD-IQ2_XXS`, `AD-IQ2_XXS`, **V3** `IQ2_S`, **V3** `IQ1_M`, **V3** `Q2_K_XL` |
| `9A18CB69…` | `AD-IQ1_M` |
| `A28AE4A2…` | **V3** `UD-IQ2_XXS` |
| `FA18CC3C…` | **V3** `UD-IQ1_S` — the rejected arm |

Q4 down to V3 `Q2_K_XL` produce **byte-identical** greedy output. That is a real
and useful invariance: it shows the boot served the artifact it claimed to, and
it costs nothing.

**It does not predict usability.** V3 `IQ1_M` matches the canonical hash, passed
the 4-minute answer screen 3/3, and is failing the corpus on 52 % of attempts
with no answer emitted. Both cheap gates agree with each other and both are
wrong about the same arm. Treat them as floors, not gates.

---

## 10. What this inventory says to do next

1. **Deep-context quality on a non-Q4 artifact.** One `•` in that column, on the
   slowest artifact on the machine, gating every deep recommendation made here.
2. **A corpus on the four fastest artifacts.** Bonsai-1bit, both MoE arms and
   gpt-oss have speed numbers and no evidence they can finish a task.
3. **Re-measure the MoE arms above 512 MiB free.** They were taken at 227–363 MiB.
4. **The context ceiling above 128K** — the armed sweep, and the only run that
   addresses the stated goal.
5. **The `Q2_K_XL` 55/100 empty-reply anomaly**, still unexplained.
6. **The seven unrun decoders (§2).** Three of them — `ngram-map-k`,
   `ngram-map-k4v`, `ngram-cache` — need no drafter file and no download, so
   they cost one boot each. `draft-dspark` has its drafter on disk and a fixed
   launcher. `draft-dflash` and `draft-eagle3` need artifacts that have not been
   looked for. This is the axis the research is loudest about and the one with
   the least coverage.
