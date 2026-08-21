# Vendor quantization tables — Unsloth and AtomicChat, transcribed

**Source: the vendors' own published charts**, transcribed here so nobody has to
reopen an image. Three sources, filed under `researchs/` because the folder rule
applies: **nothing here is evidence until it is measured on this machine.**

| image | what it is |
|---|---|
| `unsloth pre-v3.jpg` | the **Dynamic V3.0 preview** release page |
| `unsloth v3.jpg` | the **Dynamic V3.0 final** release page |
| `atomic chat.jpg` | AtomicChat's AD layout against every other GGUF |

> **A naming correction this project owes itself.** Both Unsloth releases are
> Dynamic V3. The first says *"Qwen3.8 GGUFs use Unsloth Dynamic V3.0
> **(preview)**"*; the second says it is *"an update of our first shared **early
> preview** version of Dynamic v3.0"*. **There is no "pre-V3".** This repo has
> been calling the preview build that, which reads as a different generation of
> the technology rather than an earlier build of the same one.

---

## 1. The model itself

Both vendors agree, and it matches what our loader reports:

| | |
|---|---|
| architecture | **27 B dense**, 64 layers |
| context | **256K** native (`n_ctx_train = 262144`, confirmed locally) |
| modes | **hybrid thinking** — vision + reasoning |
| Unsloth extras | Developer Role Support for agentic tools like Codex; improved nested-object parsing for tool calling |

## 2. Recommended sampling — and the two presets are different

From the V3-preview page. **This is the most immediately actionable thing on
this page**, because a hybrid model has two presets and using the wrong one is
silent.

| parameter | **Thinking mode** | Non-thinking |
|---|---:|---:|
| `temperature` | **1.0** | 0.7 |
| `top_p` | **0.95** | 0.80 |
| `top_k` | **20** | 20 |
| `min_p` | **0.0** | 0.0 |
| `presence_penalty` | **0.0** | 1.5 |

**Where we stand against it:**

- `run_retry_bench.py` defaults are `temperature 1.0, top_p 0.95, top_k 20,
  min_p 0.0` — **the thinking preset, correct.**
- `llama-server`'s own defaults are `temperature 1.0, top_p 0.95, top_k 20,
  **min_p 0.05**`. OpenCode sends no sampling parameters, so it inherits those —
  **`min_p` 0.05 where the vendor says 0.0**, and that is the one real
  difference between the two measurements this project wants to compare.

## 3. Unsloth hardware guidance — and it moved between builds

Units are total memory (RAM + VRAM, or unified).

| build | 1-bit | 2-bit | 3-bit | 4-bit | 6-bit | 8-bit | BF16 |
|---|---|---|---|---|---|---|---|
| V3 **preview** | — | 11–13 GB | 13–16 GB | 17–19 GB | 24 GB | 31 GB | 56 GB |
| V3 **final** | **7–8 GB** | **9–11 GB** | 12–14 GB | 16–19 GB | 23–26 GB | 31 GB | 56 GB |

The final build adds a 1-bit row and shifts 2-bit down by ~2 GB. That is
consistent with what we measured: the same filename is **7.27 GB** in the final
build and **9.01 GB** in the preview.

## 4. Unsloth V3 final — top-1 % accuracy against BF16

![Unsloth Dynamic v3.0 for Qwen3.8 — top-1 % accuracy against BF16 by quant size, four providers](unsloth%20v3.jpg)

*Source image, kept so the transcription below can be checked against it rather
than trusted. Providers: **Un (New)** = Unsloth Dynamic v3.0, and three others
the chart labels only as `At`, `By`, `Ba`.*

Read off that chart. The x-axis is labelled **"Quant size (GB) with removal
of MTP"**, which independently confirms our own finding that V3 `IQ2_XXS` has no
`blk.64` MTP head.

| artifact | top-1 % vs BF16 | our measured `65+0` ceiling |
|---|---:|---|
| `UD-IQ1_S` | ~73 | **196,608** |
| `UD-IQ1_M` | ~76.5 | **163,840** |
| `UD-IQ2_XXS` | ~79 | **147,456** |
| `UD-IQ2_S` | **~84** | **98,304** |
| `UD-Q2_K_XL` | ~87 | below 131,072 (`54+12` there) |
| `UD-IQ3_XXS` | ~89 | not held |
| `UD-IQ3_S` | ~91 | not held |
| `UD-Q3_K_XL` | ~93 | not held |
| `UD-Q4_K_XL` | ~95 | not held |

**The steepest part of the whole curve is `IQ2_XXS` → `IQ2_S`: five points for
1.1 GB.** It is also exactly where our own bits-per-weight ladder is steepest,
which is why that artifact is the one worth testing.

Vendor claim: *">10 % top-1 % better accuracy at the same size compared to every
other provider."* Unverified here.

## 5. Unsloth V3 preview — the same curve, different numbers

![Unsloth Dynamic V3.0 preview release page for Qwen3.8-27B — hardware table, recommended sampling, and top-1 token agreement with BF16](unsloth%20pre-v3.jpg)

The preview page plots **"top-1 token agreement with BF16 (%)"** and reads
higher across the board — `UD-IQ2_XXS` ≈ 82.5 there against ≈ 79 in the final
chart. **The two are not the same metric on the same axis**, so the gap is not a
regression. Do not compare a number from one chart against a number from the
other.

| artifact | top-1 agreement % |
|---|---:|
| `UD-IQ2_XXS` | ~82.5 |
| `UD-IQ2_M` | ~85 |
| `UD-Q2_K_XL` | ~86 |
| `UD-IQ3_XXS` | ~90 |
| `UD-Q3_K_XL` | ~92.5 |
| `UD-Q4_K_XL` | ~96 |
| `UD-Q5_K_XL` | ~97 |
| `Q6_K` | ~97.7 |
| `UD-Q6_K_XL` | ~98.2 |
| `Q8_0` | ~98.5 |
| `UD-Q8_K_XL` | ~98.7 |

## 6. Unsloth's published model benchmarks

Qwen3.8-27B against its stated comparators. **Vendor-reported, not measured
here**, and the local artifacts are 2-bit quantizations of these weights, so
these are a ceiling rather than a forecast.

| benchmark | Qwen3.8-27B | Qwen3.6-27B | Qwen3.7-Plus | Muse Glimmer-30B | Opus4.6 Max |
|---|---:|---:|---:|---:|---:|
| Agentic terminal coding (Terminal Bench 2.1) | 73.0 | 63.4 | 64.0 | 51.7 | **78.2** |
| Agentic coding (SWE-bench Pro) | **61.7** | 53.5 | 57.6 | 51.2 | 53.4 |
| Repo-level code generation (NL2Repo) | 42.3 | 36.2 | 41.1 | — | **47.6** |
| Agentic coding (DeepSWE 1.1) | **42.2** | 13.3 | 14.2 | — | — |
| Software engineering (QwenSWEBench) | **79.0** | 49.3 | 59.2 | — | 63.8 |
| Long-horizon office work (CoWorkBench) | **70.7** | 61.0 | 65.1 | — | 68.2 |
| Professional job tasks (JobBench) | **33.4** | 21.8 | 27.6 | — | — |
| Frontier agentic (Agents' Last Exam) | pass@1 **20.4** / score **42.9** | 10.6 / 27.3 | 13.2 / 33.6 | — | — |
| Instruction following (IFBench) | **79.5** | 69.1 | 79.1 | 77.0 | 62.5 |
| Scientific reasoning (GPQA Diamond) | 89.2 | 87.8 | 90.3 | 83.5 | **91.3** |

## 7. AtomicChat — mean KL divergence against BF16

![AtomicChat AD layout against every other Qwen3.8-27B GGUF — mean KL divergence vs BF16 by file size](atomic%20chat.jpg)

Their methodology, from the chart: `eval_neutral` held-out, **4096 ctx**,
reference BF16, 4× RTX 5090, CUDA 13.0. **Lower is better.**

Their own recommendation table:

| memory | quant | size | top-1 | mean KL div |
|---|---|---:|---:|---:|
| **12 GB** | **`AD-IQ2_XS`** | 9.9 GB | **83.5 %** | 0.1617 |
| 16 GB | `AD-IQ3_S` | 13.8 GB | 92.4 % | 0.0325 |
| 24 GB | `AD-Q5_K` | 20.2 GB | 97.3 % | 0.0042 |
| 32 GB | `AD-Q6_K` | 25.0 GB | 98.7 % | 0.0011 |
| 48 GB | `Q8_0` | 28.9 GB | 98.9 % | 0.0006 |

Points read off the small end of the curve, where our card lives:

| artifact | ~size | ~mean KL |
|---|---:|---:|
| `AD-IQ1_M` | 8.5 GB | **0.36** — worst on the whole chart |
| `AD-IQ2_XXS` | 9.0 GB | 0.27 |
| `UD-IQ2_XXS` (Unsloth) | 9.0 GB | **0.23** |
| `AD-IQ2_XS` | 9.9 GB | 0.16 |
| `UD-IQ2_M` | 10.5 GB | 0.13 |
| `AD-IQ2_S` | 11.2 GB | 0.10 |

**Two things worth taking from this.**

**At ~9 GB, Unsloth beats AtomicChat** — `UD-IQ2_XXS` at 0.23 against
`AD-IQ2_XXS` at 0.27. AtomicChat's advantage on their own chart begins further
up the size axis. Note the 9 GB `UD-IQ2_XXS` on their chart is the **preview**
build; the final build is 7.27 GB and is not plotted.

**The AtomicChat file we hold is the one they rate worst.** `AD-IQ1_M` sits at
the far left at KL 0.36; their 12 GB recommendation is `AD-IQ2_XS` at 9.9 GB,
which we have never downloaded. That is worth remembering next to our own
measurement that `AD-IQ1_M` has the best corpus of any 1-bit-named file — the
two statements are about different things and both can be true.

---

## What this changes for us, and what it does not

**Changes:** the naming (V3-preview / V3-final, never "pre-V3"); the `min_p`
discrepancy is now a known difference rather than an unknown; `UD-IQ2_S` has
vendor evidence behind it, not only our inference.

**Does not change:** every accuracy number above is the vendor's, measured
against BF16 on their hardware at 4096 context. **None of it says anything about
whether the model finishes an agent round trip on a 12 GB card at 128K**, which
is the question this project exists to answer. Keep the ranking, ignore the
absolute values, and measure.
