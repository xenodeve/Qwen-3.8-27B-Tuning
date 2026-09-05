# ExLlama3 techniques llama.cpp does not have — a survey of the Mia-AiLab fork (2026-09-04)

> **External material. Nothing here is evidence until measured on this machine.**
> Sources: the fork's source tree at `C:\AI\exllamav3-mia` (@ 63b32f0, base
> 1.4.2) read against upstream 1.4.6 at `C:\AI\exllamav3-src`; the Mia-AiLab
> and RadixArk model cards on Hugging Face; upstream `doc/exl3.md`. Every
> vendor number is tagged **VENDOR** and names its hardware. What this repo has
> already measured is tagged **MEASURED HERE** with its file. Two research
> agents did the sweep; every `file:line` below was re-checked by hand.
>
> Written after issue #71 settled the speed question (EXL3 recipe = ~81 % of
> llama.cpp decode at 147K in one boot, `results/10-other-engines.md`). The
> question here is different: *does the platform have a technique the primary
> engine lacks, worth carrying over or worth one more EXL3 arm?*

## 1. The catalogue

| technique | where | vs upstream EXL3 | vs llama.cpp | state here |
|---|---|---|---|---|
| **DFlash2 block-diffusion drafter** (`-dm`, 1.93B, EXL3 5.0bpw 1.4 GB) | `architecture/dflash2.py`, `generator.py:272` | upstream-too | llama.cpp `draft-dflash` with the Q2_K_S drafter: **+0.2 % at 147,456** against `draft-mtp,ngram-mod` (`results/nvfp4-dflash-147456.jsonl`) | **UNMEASURED on EXL3**; EXL3 drafter downloaded 2026-09-04 |
| **DSpark drafter** (DFlash backbone + Markov/confidence heads, SpecForge-trained) | `architecture/dspark.py:1-25`, `EXL3_DSPARK_CONF` (`dspark.py:196`) | **fork-only** | llama.cpp mainline has `--spec-type draft-dspark`; here it was *"attempted, drafter path resolved empty, never launched, fixed and not re-run"* (report 16 §decoders) | **UNMEASURED on both engines**; RadixArk v2 checkpoint 2026-08-14, Q8_0 GGUF 1.45 GB downloaded 2026-09-04 |
| **Lossless rejection-sampling verify at T>0** (`EXL3_SPEC_RS=1`) | `generator.py:222-227, 1043-1065` | upstream-too | llama.cpp's speculative verify at T>0 accepts by sampled-token equality, not Leviathan p/q — a different acceptance rule; whether it is lossless is not established here | UNMEASURED; every arena arm is greedy, so it has never been exercised |
| **NVFP4 / FP8 KV cache with fused quantize-into-cache kernels** | `cache/nvfp4.py`, `cache/fp8.py` (absent upstream) | **fork-only** | llama.cpp has `q4_0`/`q8_0` KV, no FP4 format | **MEASURED HERE**: both bypass BC-attn graph capture (`bc_attn.py:430`) and decode at half of `-cq 4`; VENDOR claims "lossless in the noise" (cos-sim 0.99995 vs fp16) |
| **Integer k/v-bit cache 2–8 bits, per-side** (`-cq k[,v]`) | `cache/quant.py:13-41` | upstream-too | llama.cpp KV types are per-tensor-type, no asymmetric k/v | **MEASURED HERE**: `-cq 4` = 18 KiB/token and graph-captured — the lever that doubled decode; `-cq 8,4` voided (changed the output) |
| **Whole-step CUDA-graph decode ("BC-attn")** | `attention_fn/bc_attn.py:1-31`, `EXL3_BC_ATTN_TRACE` | upstream-too | llama.cpp has CUDA graphs too | MEASURED HERE indirectly: the arms where it captures are 2–2.7× the arms where it declines |
| **Hashed page reuse across jobs** (prompt cache by page hash, LRU-evictable) | `generator/pagetable.py:263-296, 575-646`, `job.py:314` | upstream-too | llama.cpp has the byte-identical-prefix cache + `--cache-ram`; same shape, page granularity instead of per-slot | MEASURED HERE: rounds 2–3 prefill in 0.2–2 s |
| **CPU page-cache tier** (`cpu_cache_size`) | `generator.py:229-235` | upstream-too | llama.cpp `--cache-ram` is the equivalent (results 46) | not usable here: *"not currently supported in tensor-parallel mode"* (`generator.py:233`) |
| **Native tensor-parallel without NCCL**, `-tp_*` per-module parallelism limits, `EXL3_TP_SPIN_RECV` | `model/model_tp_fn.py:13, 84`, `model_init.py:61-66` | upstream-too | llama.cpp `-sm tensor` is the equivalent and wins layer split by 31 % here | MEASURED HERE: TP +30 % over layer split; `-tp_linear_attn 1` and spin-recv inside the drift |
| **Recurrent-state checkpoint stashes for the 48 GDN layers** | `cache/recurrent.py:1-25` | upstream-too | llama.cpp `--ctx-checkpoints` is the equivalent (commit df66172) | not measured on EXL3 |
| **Int8 GEMV / MGEMM fusion thresholds** (`EXL3_INT8_GEMV`, `EXL3_MGEMM_K_THRESHOLD`) | `model/config.py:26-32` | upstream-too | no llama.cpp analogue (different weight format) | MEASURED HERE: `EXL3_INT8_GEMV=0` inside the drift |
| **Quantised-cache prefill staging** (`EXL3_QC_STAGING`, `EXL3_QC_PREFILL_NS`) | `attention_fn/triton_paged.py:868-882, 1836` | upstream-too | n/a | MEASURED HERE: auto-pick already optimal; staging 0 costs 10 % prefill |
| **Workload-matched calibration at quant time** (622K-token self-generated coding/math trace) | model card | quant-time, Mia-AiLab's choice | GGUF imatrix is the analogue; Unsloth's Dynamic V3 uses its own corpus | VENDOR: "measurably better acceptance and task behaviour than generic calibration of the same bpw"; no number given |
| **Trellis (QTIP-derived) quantisation, module-adaptive bpw, 6-bit head** | upstream `doc/exl3.md`; card: "3.5 bpw (module-adaptive), `mul1`, `-hq`" | upstream | no GGUF analogue below Q2 | VENDOR: Llama-3.1-70B "coherent at 1.6 bpw"; no Qwen3.8 perplexity/KL vs Q4_K or NVFP4 published anywhere found |
| **YaRN 1M config shipped** (`config.yarn-1m.json`) | model card | — | llama.cpp `--rope-scaling yarn` exists | VENDOR: needles past 262K fail; treat 262K as the limit |
| **MoE CPU expert offload** (`EXL3_MOE_CPU_*`) | `model/config.py:39`, `moe_cpu_host.py` | upstream-too (fork lacks `block_sparse_mlp_cpu.py`) | llama.cpp `-ot` / `--cpu-moe` | n/a — Qwen3.8-27B is dense |
| **OpenAI-compatible server with Qwen tool-call parsing and inline `<think>`** | `tools/serve_openai.py:11-16, 137-214` | **fork-only** (no `tools/` upstream) | llama-server does the same and also speaks the Anthropic path Claude Code uses | not tried; no Anthropic-API path, so Claude Code cannot talk to it without a proxy |
| **Fused sampler** (`EXL3_FUSED_SAMPLER`) | `sampler/custom.py:19` | upstream-too | — | irrelevant under greedy |

## 2. What is genuinely new to this project, in order of expected value

### 2.1 DSpark v2 — on **llama.cpp**, not only on EXL3

This is the one item in the survey that lands on the primary engine. RadixArk's
`Qwen3.8-27B-DSpark` (2026-08-14, SpecForge-trained, 1.86B, five full-attention
layers, target taps at layers 5/19/33/47/61, serving gamma 7) publishes, for
the **NVFP4 target**:

| workload | DSpark v1 accept length | **v2** |
|---|---:|---:|
| HumanEval | 3.04 | **3.85** |
| LiveCodeBench | 2.59 | **3.35** |
| RULER-8K | 4.96 | **6.30** |
| request-weighted, 64,675 prompts | 2.72 | **3.43** (+26 %) |

and throughput at concurrency 1 on one H200 against the FP8 target: HumanEval
autoregressive 95.8 tok/s, EAGLE (= the MTP head, 3 steps, 4 draft tokens)
165.5 (1.73×), **DSpark v2 254.8 (2.66×)** — i.e. **~1.5× over MTP** at batch 1
(**VENDOR**, H200, T = 1.0, thinking on, 128 prompts). MT-Bench is the weakest
case at 2.25× vs 1.64×.

**What this repo knows already:** the served binary lists `draft-dspark`
(`llama-server --help`, report 08 §2 — "genuinely mainline"); the one attempt
never launched because the drafter path resolved empty and was never re-run
(report 16, the ❌ row); every DFlash/DSpark verdict on llama.cpp so far was
about **DFlash2**, which gave +0.2 % at 147,456 against `draft-mtp,ngram-mod`
(`results/nvfp4-dflash-147456.jsonl`). DSpark v2 is a different drafter with a
Markov head, a confidence head and a published acceptance ~1.3× v1's; and the
guide's caution stands — *"on quantized targets … can differ from a non-quantized
target"* (guide §4) — our target is NVFP4, the same family the acceptance table
was measured on.

**Outcome, 2026-09-04 (results 02, last section):** measured under `-sm layer` at 65,536 — the only shape the drafter loads in on build 10499 — DSpark v2 is **−17 % against the MTP head**, acceptance 37 % vs 58 %; at 147,456 it OOMs. Closed. The run's control arm exposed something else: `-sm layer` +55 % over the served tensor split at 65,536 (ledger).

**Cost to measure (as planned before the run):** the Q8_0 GGUF (1.45 GB, `magnitudedev/Qwen3.8-27B-DSpark-GGUF`)
is on disk as of 2026-09-04; one arena arm set `draft-dspark,ngram-mod` beside
`nvfp4-served`, paired and rotated at 147,456, ~30 min. **Depth is the risk**:
`draft-mtp` itself flips from +81 % at 16K to −71 % at 131K on Q4 (CLAUDE.md);
the draft KV for a 1.86B five-layer drafter at 147K is small but not zero.

### 2.2 DFlash2 as EXL3's drafter (`-dm`)

The EXL3-quantised drafter (`Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw`, 1.4 GB,
1.93B params, block 8) is on disk. **VENDOR (DGX Spark / GB10, 273 GB/s):**
MTP alone ~2.2 accepted tokens/step, ~30 tok/s; DFlash2 greedy code-prose
2.7–2.8/step, 40–43 tok/s; the EXL3 quant of the drafter itself gave +33 % over
the bf16 drafter. Read against our numbers: the recipe at 144K accepts ~1.6
tokens/step (acc ≈ 320 over 512 with rej ≈ 260; `-ndt 3`), so the drafter
would have to lift acceptance well past that and pay for its own weight reads
across the PCIe TP split. On llama.cpp the same drafter idea was worth +0.2 %
at this depth. **One paired arm settles it**; VRAM allows it (4070 at 5.7 GB
+ 1.4 GB drafter, 5060 Ti with the KV).

### 2.3 Rejection sampling at T>0 (`EXL3_SPEC_RS=1`)

Only matters once anything is measured at the served sampler. Every arm in this
repo is greedy by rule (arena `SAMPLER`), so this cannot be compared without
first deciding a T>0 protocol — which is a decision, not a number.

### 2.4 Everything else in the table is either measured here already, equivalent to a llama.cpp lever, or not applicable to a dense model.

## 3. Quality — one VENDOR chart now exists, and it says more about NVFP4 than about EXL3

**turboderp's own KL / perplexity sweep for Qwen3.8-27B** (mirrored by Mia-AiLab at
`Mia-AiLab/Qwen3.8-27B-EXL3`, 2026-09-03; the developer brought the three PNGs
in on 2026-09-04). Protocol, from the chart headers: reference HF BF16;
**self-generated in-domain trace**, 19,016 input + 45,930 output tokens; mean
KL(p_FP ‖ p_quant) per token; noise floor 0.00045 (reference against itself under
bf16-rounding perturbation); x axis = quantised weight size excluding embeddings.
"SC" = *self-calibrated* — calibrated on the model's own generated trace, the
same idea Mia's 3.5bpw card describes. **VENDOR, one trace, the quantiser's own
distribution, not measured here.**

| artifact | weights (GiB, chart axis) | mean KL | perplexity |
|---|---:|---:|---:|
| FP8 (Qwen) | 25.1 | 0.0022 | 1.3355 |
| UD-Q6_K_XL | 23.3 | 0.0009 | 1.3338 |
| **NVFP4 (Unsloth)** | **18.0** | **0.0092** | **1.3458** |
| UD-Q5_K_XL | 18.0 | 0.0028 | 1.3370 |
| EXL3 6.00 / SC 6.00 H6 | 18.0 | 0.0009 / 0.0006 | 1.3321 / 1.3334 |
| UD-Q4_K_XL | 16.0 | 0.0051 | 1.3401 |
| EXL3 5.00 / SC 5.00 H6 | 15.1 | 0.0023 / 0.0016 | 1.3355 / 1.3348 |
| IQ4_XS | 14.0 | 0.0101 | 1.3487 |
| **EXL3 4.00 / SC 4.00 H5** | **12.2** | **0.0082 / 0.0062** | **1.3440 / 1.3424** |
| UD-Q3_K_XL | 13.5 | 0.0209 | 1.3645 |
| EXL3 3.00 / SC 3.00 H4 | 9.4 | 0.0332 / 0.0257 | 1.3806 / 1.3658 |
| UD-Q2_K_XL | 9.5 | 0.0587 | 1.4198 |

**What it says, if it holds on our data:**

1. **NVFP4 (Unsloth) is the worst point in the 4-bit class on this chart** —
   KL 0.0092 at 18 GiB against UD-Q4_K_XL 0.0051 at 16 GiB and EXL3 4.00 bpw
   0.0082 at 12.2 GiB (SC 0.0062). The KL-vs-size line for GGUF-UD passes
   *below* NVFP4 by ~2×. This is the first external quality number on the
   family this repo serves as primary; **it is not our file** — we serve
   `esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF` `VERY-LOW`, a different quantiser
   and a mixed-precision variant, so the point transfers as a hypothesis only.
2. **EXL3 4.00 bpw at 12.2 GiB ≥ NVFP4 at 18 GiB** on both KL and perplexity;
   SC 4.00 is 1.5× better again. Mia's 3.5bpw (ours, 14.2 GB with embeddings
   ≈ 10.7 GiB on this axis) is **not on the chart**; a plain EXL3 3.5 would
   interpolate around KL 0.015–0.02, an SC-style 3.5 around 0.010–0.013 —
   i.e. roughly NVFP4's quality at 60 % of its weight bytes. Interpolation,
   not a measurement.
3. The per-token histogram (third PNG) shows why "mean KL" flatters everyone:
   every quant has a long tail at 10⁻¹–10⁰ per token; the 4-bit EXL3 peak sits
   at ~10⁻² and the 2-bit families at ~10⁻¹.

**What this repo should do with it.** Nothing on speed changes. For quality —
the standing critical path (`nvfp4-is-the-primary-artifact`) — this is a
concrete external reason to run the repo's own quality gate on **three
artifacts in one boot: served NVFP4-MTP-VERY-LOW, UD-Q4_K_XL, and the EXL3
3.5bpw** (through the OpenAI server at :8000), rather than on NVFP4 alone.
Ledger row added.

## 4. Things the survey could not verify

- Any Windows support statement from the fork — it builds and runs here, which
  is the only evidence.
- Whether GDN (linear-attention) layers are split or replicated under native
  TP; `-tp_linear_attn 1` changed nothing measurable, which is consistent with
  either.
- Any third-party EXL3-vs-llama.cpp head-to-head on the same model and GPU;
  the one in this repo (results 10) may be the only one.
- Blackwell-specific kernels in the fork beyond the build's `sm_120` arch flag.
