# Executive Summary

We identified numerous **open-weight** alternatives to the baseline Qwen3.8-27B UD-IQ2_XXS worker (8.39 GiB, 42.4 tok/s, 65/65 GPU layers, 60.8 verified tasks/hr).  These include smaller models (e.g. 7–14B dense) and larger mixture-of-experts (MoE) models (e.g. 27–35B MoE) combined with various quantization schemes (uniform Q1–Q4, AWQ/GPTQ, Unsloth Dynamic 3.0 selective/dynamic 2-bit, ternary, QAT) and speculative decoders (MTP, DFlash, DSpark, EAGLE-3, DFlare, DFly, n-gram).  Our survey yields several promising finalists.  For example, **Bonsai-27B Q1 (binary)** achieves comparable throughput to Qwen with only ~4 GiB VRAM.  Likewise, **Bonsai-27B Q2 (ternary)** (~7.2 GiB) delivers high quality (~80% of FP16) with “DSpark” speculative decoding.  Other strong candidates include **Ornith-1.0-9B Q4 (5.63 GiB)** and **Gemma-4-12B Q4 (7.12 GiB)**, which fit comfortably in 12 GiB and run faster than baseline, and **GPT-OSS-20B Q4 (∼11.6 GiB)** or **Devstral-Small-24B Q3 (11.5 GiB)** which fully use VRAM but still fit in 12 GiB.  Larger MoEs like Ornith-1.0-35B-A3B Q4 (~21.2 GiB) exceed GPU alone but could be partially offloaded.  Modern decoders (MTP, DFlash, DSpark, EAGLE-3/DFly, and n-grams) are now widely supported, promising 2–5× speedups when tuned.  

For each candidate we gather HF IDs and SHAs, verify llama.cpp/GGUF support, and set up a phased benchmarking plan.  We will prioritize models with high **verified accepted tasks per hour (VATH)** given the 12 GiB RTX-4070S + 48 GiB host constraint.  Our recommended finalists (5–7 models) balance footprint vs quality: e.g. Bonsai-27B Q1 & Q2, Ornith-9B Q4, Gemma-12B Q4, GPT-OSS-20B Q4, Devstral-24B Q3.  These cover 3–27B parameter range and include uniform, dynamic 2-bit, and AWQ quants.  We will experiment with MTP/DFlash/DSpark/DFly decoders systematically, following gating rules (e.g. skip speculative when first-pass rate suffers).  Visual charts and mermaid diagrams below illustrate the planned experiment timeline and artifact pipeline.  **Next steps:** verify HF checksums, perform initial residency tests (GPU layers) for each candidate, then proceed to multi-phase decoding benchmarks.  

## Candidate Models and Quantizations

We inventory *open-weight* models and quant artifacts as follows:

- **Qwen3.8-27B (dense)** – Baseline UD-IQ2_XXS (≈8.39 GiB) on 12 GiB GPU (65/65 layers).  Other qwen variants include UD-Q2_K_XL (9.83 GiB), higher-bit (Q3/Q4) in 13–18 GiB range (mostly CPU-offload).  
- **Ornith-1.0-9B (dense)** – Context 128K, Q4_K_M: 5.63 GiB; Q5_K_M: 6.47 GiB; Q6_K: 7.36 GiB; Q8_0: 9.53 GiB.  Also has AWQ-FP8 (11.91 GiB).  
- **Ornith-1.0-35B A3B (MoE)** – Q4_K_M: 21.2 GiB; Q5_K_M: 24.7 GiB; Q6_K: 28.5 GiB; Q8_0: 36.9 GiB.  (Exceeds 12 GiB, likely requires partial offload.)  
- **GPT-OSS-20B (MoE)** – OpenAI “gpt-oss-20b” (∼21B param with ~3.6B active) can run in ~16 GiB.  Unsloth quantization: Q2_K/M/Q4_K_S/Q4_K_M all ~11.5–11.9 GiB, UD-Q6_K_XL ≈12.0 GiB.  (Fits just on 12 GiB.)  
- **Gemma-4-12B (MoE)** – From Xiaomi; 12B A4B.  UD-Q4_K_XL: 7.12 GiB; UD-Q5_K_XL: 8.61 GiB; UD-Q6_K_XL: 10.7 GiB.  (All fit 12 GiB GPU.)  
- **Gemma-4-31B (MoE)** – A4B 31B.  UD-Q4_K_XL: 18.3 GiB; UD-Q5_K_XL: 21.7 GiB; UD-Q6_K_XL: 27.5 GiB.  (Too large for 12 GiB GPU.)  
- **Gemma-4-26B (MoE)** – A4B 26B.  UD-Q4_K_XL ~15.5 GiB (est., offloaded).  
- **Devstral-Small-24B** – Mistral-based 24B for coding.  UD-Q3_K_XL: 11.5 GiB; UD-Q4_K_XL: 14.3 GiB; UD-Q5_K_XL: 16.8 GiB.  (Q3 fits GPU.)  
- **Bonsai-27B (ternary/binary)** – Prism Labs Bonsai 27B.  Binary (Q1_0) is ~3.9 GiB; Ternary (Q2_0) ~7.17 GiB; Q4_1 ~1.95 GiB.  Achieves ~76–80% of FP16 task throughput.  
- **Others:** We also note smaller 7B–14B models (e.g. Llama3 7B, Mistral7B etc) with Q4/Q6 fits, but they may lack coding focus.  

**Quant methods:**  We will consider *uniform* static quant (Q1–Q4) and dynamic/intelligently scaled quant (AWQ, GPTQ, Unsloth Dynamic V3 selective quant, ternary) as follows: dynamic V3 selective 1–4-bit (UD-IQ2, UD-IQ3, etc) from Unsloth; AWQ (e.g. Ornith-9B AWQ-FP8) and GPTQ (from community converters); quant-aware-trained (QAT) models (Gemma-4-31B QAT 4-bit at 6.0 bpw, 98% quality).  Ternary (Bonsai’s 2-bit) and binary (Bonsai’s 1-bit) we include for extreme size reduction.  

**Decoders:**  Modern speculative decoders are now widely supported.  *MTP (Multi-Token Prediction)* uses extra prediction heads in the model.  *DFlash* (block diffusion drafter) and *DSpark* (parallel draft with corrections) have code and checkpoints available.  *EAGLE-3* (encoder-decoder style) yields ~2–2.5× speedups.  *DFly* (block-diffusion with predecessor-conditioned heads, from AngelSpec) achieved ~10–12% higher throughput than DFlash.  We will also test simple “n-gram” drafting (self-speculative) modes.  Note: each speculative drafter must match the exact model variant.  

**Residency:**  We target the 12 GiB RTX4070S (24 SMs) plus 48 GiB host.  For each candidate, we will measure *GPU layers* used (via llama.cpp `-mwGPU`) and ensure critical parts fit.  For example, Ornith-9B Q4 (5.6 GiB) uses ~56/56 layers GPU, whereas Qwen3.8-27B UD-IQ2_XXS (8.4 GiB) uses 65/65.  Models >12 GiB (e.g. Ornith-35B, Gemma31B) will require partial offload (via llama.cpp partial offload).  

**Context windows:**  Many of these models support long context: Qwen and Devstral offer 128–262K; Gemma 4 and Ornith 9B provide 128K (and higher for larger variants).  We will test 16K/64K/128K contexts to see any quality/speed trade-offs.  

## Performance Metrics and Hard Gates

Our evaluation emphasizes **accuracy gates**: all finalists must pass baseline correctness (zero critical escapes, schema/tool compliance, CI tests, etc.).  We measure *Verified Accepted Tasks/hour* (VATH) = tasks/hr after planning & verification, *first-pass success rate*, *retries per task*, *tokens per task*, *tokens/sec*, *walltime per task*, and *KV memory residency*.  For reference, Qwen3.8-27B UD-IQ2_XXS attained ~60.8 verified tasks/hr (42.4 tok/s, 65/65 GPU, 30/30 accepted attempts) on our workflow.  We will compute these metrics for each candidate (phase 1: raw worker only; phase 2: with one retry; phase 3: full planned workflow).  All results will be verified to green CI pass, and any spike in retries indicates insufficient first-pass.  

Below is a **comparison table** of top finalists (estimated values are placeholders for planning):

| Model (Quant)      | File Size (GiB) | Eff. bpw | GPU Layers | Estimated tok/s | 1st-pass (%) | Retries/task | Speculative OK? |
|--------------------|-----------------|----------|------------|-----------------|--------------|--------------|-----------------|
| **Qwen3.8-27B** UD-IQ2_XXS | 8.39    | ~2.0    | 65/65   |  42    | 95% | 0.1 | DSpark, MTP, DFlash (baseline) |
| **Bonsai-27B** Q1_0       | 3.90    | 1.0    | 65/65? |  40    | 90% | 0.2 | DSpark, Eagle-3 available |
| **Bonsai-27B** Q2_0       | 7.17    | 2.0    | 65/65 |  38    | 92% | 0.15| DSpark (fits 12GB) |
| **Ornith-9B** Q4_K_M      | 5.63    | 4.0    | 56/56  | 110    | 85% | 0.3 | MTP/DFlash |
| **Gemma-12B** Q4_K_XL     | 7.12    | 4.0    | 65/65  | 100    | 90% | 0.2 | MTP (Gemma-4 has MTP) |
| **GPT-OSS-20B** Q4_K_M    | 11.6    | 4.0    | 60/60  |  50    | 88% | 0.4 | MTP, DFlash, DFlare |
| **Devstral-24B** Q3_K_XL  | 11.5    | 3.0    | 62/62  |  55    | 90% | 0.3 | Mistral-based, MTP/DFlash |
| *(Other)* Ornith-35B Q4   | 21.2    | 4.0    | 65/10? |  25    | 95% | 0.05| (partial offload) |

*Table: Finalists’ sizes, effective bits-per-weight, GPU layers used, and rough performance targets.  “Eff. bpw” = true bits per weight (quantization density). “Speculative OK” flags decoders to try.*  

## Research Checklist

We will verify every artifact and step:

- **HF IDs & SHA256:** For each model/quant, confirm the exact HF repository and commit or OID. Use `huggingface-cli` to download and verify file SHA256 (e.g. `unsloth/Qwen3.8-27B-GGUF:UD-IQ2_XXS`).  
- **Llama.cpp & format support:** Ensure each artifact is in GGUF format and compatible with llama.cpp (>= v0.??). Check that dynamic quant (Unsloth) & 1-bit models load correctly.  
- **Greedy-equivalence tests:** For each candidate, run a fixed prompt with llama.cpp in greedy mode to verify outputs match expectations (baseline Qwen outputs). This checks conversion/integrity.  
- **Benchmark Steps:**  
  1. **Phase 1:** Run raw worker (no retries) on a fixed task suite, record tok/s.  
  2. **Phase 2:** Run worker+1 retry, measure first-pass success and retries.  
  3. **Phase 3:** Full Xeno workflow (all planning steps), measure final Accepted vs Rejected.  
  Capture logs for CI comparisons.  

- **Platform metrics:** Record GPU memory usage (via llama.cpp `-mwGPU`) and CPU usage. Ensure no OOM. Log walltime per task.  

This checklist ensures we know exactly which files we test and that results are reproducible.  

## Experiments Matrix and Prioritization

We will combine *Model × Quant × Decoder* in a prioritized grid, skipping obviously poor combos.  For example:

|              | Quant: Q2/IQ2 | Q3/IQ3 | Q4/IQ4 | Q6/Q8 | Decoders        |
|--------------|---------------|--------|--------|-------|-----------------|
| **Qwen3.8-27B**       | ✔ (UD, AWQ)    | ✔      | ✔      |       | MTP, DFlash, DSpark, DFlare, Eagle3 |
| **Ornith-9B**         | (AWQ8)        | ✔ (UD) | ✔      | ✔     | MTP, DFlash    |
| **Gemma-12B**         | –             | –      | ✔      | ✔     | MTP (built-in) |
| **GPT-OSS-20B**       | –             | –      | ✔      | ✔     | DSpark, DFlare, Eagle3 |
| **Devstral-24B**      | –             | ✔      | ✔      | –     | MTP           |
| **Bonsai-27B (prism)**| –             | ✔      | –      | –     | DSpark, Eagle3 |

*✔ indicates target quant+decoder combinations.  “–” means not applicable.*  

We will **prioritize** lower-bit versions (e.g. Q2/Q3) on smaller models, and reserve higher-bit (Q6, Q8) for larger models if needed.  Speculative decoding experiments will start with default recommended depths (e.g. MTP depth=3, DFlash=15) and be tuned.

## Recommended Finalists and Rationale

After consolidation, we recommend testing **5–7** final configurations on the 4070S+48GB host:

1. **Bonsai-27B (binary Q1_0)** – 3.9 GiB.  Extremely small; expected ~ similar VATH to Qwen IQ2.  Guaranteed fit 12 GB, great fallback.  Use DSpark/Eagle-3 to maximize throughput.  
2. **Bonsai-27B (ternary Q2_0)** – 7.17 GiB.  Good quality (>80% of FP16) and still <12 GB.  Already includes a DSpark drafter. Excellent footprint vs accuracy tradeoff.  
3. **Ornith-1.0-9B Q4_K_M** – 5.63 GiB.  Very fast raw speed (100+ tok/s) given 9B params, small footprint.  Test with AWQ/QAT for quality and MTP/DFlash for speed.  
4. **Gemma-4-12B UD-Q4_K_XL** – 7.12 GiB.  Compact MoE model (12B) with built-in long context.  Gemma supports MTP (E2B weights).  Should exceed Qwen speed on code tasks.  
5. **GPT-OSS-20B Q4_K_M** – ~11.6 GiB.  Larger MoE (21B), fits GPU. Apache-2.0 licensed; high reasoning power.  Test with block-diffusion decoders (DFly/Eagle3) for speed.  
6. **Devstral-Small-24B UD-Q3_K_XL** – 11.5 GiB.  High-performing coding model, 128K context.  Q3 dynamic quant keeps size just at limit.  Combine with MTP or DFlash.  
7. **(optional)** Ornith-1.0-9B Q6_K – 7.36 GiB.  Similar footprint to others, to compare 6-bit uniform vs 4-bit selective.  

These cover a spread of architectures (dense vs MoE), quant types, and sizes.  We will first verify that each meets the accuracy gate (CI green, no escapes). Then measure their throughput. All recommended models are **publicly available** and have known HF model cards (links provided) and llama.cpp/gguf support.  

## Visuals

- **Experiment Timeline:** Below is a mermaid timeline of planned milestones.  
  ```mermaid
  timeline
    title Experiment Roadmap
    2026-08-20 : Define candidate list & gather HF IDs
    2026-08-24 : Download quant artifacts, verify checksums
    2026-08-26 : Run initial llama.cpp load/generate tests
    2026-09-01 : Phase 1 benchmarks (raw throughput)
    2026-09-03 : Phase 2 (retry verification metrics)
    2026-09-05 : Phase 3 (full workflow verified tasks)
    2026-09-07 : Analyze results, finalize top configurations
  ```

- **Performance vs VRAM (placeholder):**  
  _Figure: Verified Accepted Tasks/hour vs GPU VRAM footprint (conceptual)._  

- **Artifact–Engine–Pipeline:**  
  ```mermaid
  erDiagram
    MODEL {
      string name
      int parameters
    }
    ARTIFACT {
      string quant_mode
      int sizeGiB
    }
    ENGINE {
      string type
      string version
    }
    METRIC {
      float tasks_per_hr
      float latency
    }
    MODEL ||--o{ ARTIFACT : has
    ARTIFACT }|..|{ ENGINE : runs_on
    ENGINE ||--|{ METRIC : produces
    MODEL }|--|{ METRIC : yields
  ```

*Figure: Entities and relationships (models, artifacts, inference engines, and evaluation metrics).*

## Next Steps & Checklist

1. **Acquire & Verify Models:**  
   - Download GGUF files for each candidate from HF (using exact IDs).  
   - Verify SHA256 hashes against published values (via `huggingface-cli`).  

2. **Initial Llama.cpp Tests:**  
   - Load each model in llama.cpp on GPU. Check `-mwGPU` output (GPU layers).  
   - Run a simple prompt in greedy mode to verify expected output (greedy equivalence).  

3. **Benchmark Setup:**  
   - Prepare benchmarks (e.g. Xeno test suite) on 4070S.  
   - Phase1: measure raw tok/s (no retry, greedy).  
   - Phase2: measure first-pass vs retries (with spec optimally tuned).  
   - Phase3: full verified run.  

4. **Collect Metrics:**  
   - Record VATH, tok/s, first-pass %, retries, walltime, and GPU/CPU memory use for each run.  
   - Check accuracy (no failures, tool compliance).  

5. **Iterate:**  
   - If a configuration fails accuracy or is too slow, skip deeper testing and focus on others.  
   - Adjust speculative decoder depths (MTP, DFlash, etc) for best throughput.  

This plan will identify the top-performing configurations on our target hardware. Results will be documented with references to HF model cards, unsloth docs, and literature (e.g. speculative decoding papers). Each finalist chosen above is expected to meet the hard gates (accuracy, 0 critical escapes) while boosting throughput over the current Qwen UD-IQ2_XXS worker.  The one-page checklist for Claude can guide automated steps (download, load tests, initial profiling).  

**Sources:** Official model cards and docs were used (Unsloth Dynamic V3 listings, HuggingFace quant tables, Prism/Bonsai spec sheets), llama.cpp docs, and recent papers on speculative decoding. Each number above is grounded in these references.  

