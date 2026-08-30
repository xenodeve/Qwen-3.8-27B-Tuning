# docs/optimizations.md, docs/gotchas.md, docs/long-context.md, docs/quality.md, docs/docker.md, README.md, single-user/README.md, batch/README.md — the prose/measurement layer of syv-ai/qwen38-27b-rtx3090 (patched vLLM 0.27.1, Qwen3.8-27B W4A16 on one RTX 3090 24 GB @ 250 W)
**64 techniques.** 1892 source lines across 8 files.
Files read: `docs/optimizations.md` · `docs/gotchas.md` · `docs/long-context.md` · `docs/quality.md` · `docs/docker.md` · `README.md` · `single-user/README.md` · `batch/README.md`
> **What the reader could not see:** Everything in the slice existed and was readable in full (264/331/172/61/99/407/382/176 lines = 1892). Gaps and inconsistencies inside the slice worth flagging to the reader: - README.md:396 advertises "18 things that each cost us hours" but docs/gotchas.md ships 37 entries — a stale count. - docs/optimizations.md:3 says "Nine things stock vLLM doesn't give you", and README.md:257 repeats it; the actual list is 9 numbered items plus two speculative modes plus the lookup drafter, so the headline number undercounts. - docs/long-context.md:137 names the patch inline as `spec-decode-attn-int8.patch` while the link target is `../patches/spec-decode-int8-kv.patch`. Two names for one artifact; I could not resolve which is real without reading `patches/`. - Gotcha 23 (gotchas.md:130-134, "per request slot") is explicitly retracted by gotcha 33 (gotchas.md:223-229, "that is wrong"), yet gotcha 23's wording is reproduced unretracted in single-user/README.md:214-216 and docs/optimizations.md:240-241. A reader taking the single-user README alone gets the retracted model. - docs/optimizations.md:234-236 and :240-242 state the "4 request slots instead of 8 and 56k instead of 64k" fact twice in near-identical sentences — duplication, not disagreement. - No source file, patch, or benchmark script was in this slice, so every mechanism below is as the authors describe it; I verified none of it against `patches/`, `bench/`, `prepare/`, `drafter/` or `kvarn/`. - Several numbers are self-inconsistent across files and I recorded both readings rather than picking one: C1 DFlash2 greedy is 131.2 (single-user/README.md:61) vs 133 (README.md:79) vs "best runs 138.5" (optimizations.md:151); the copy row is 381 (optimizations.md:225) vs 382 (README.md:80) vs 379 (single-user/README.md:167). `VLLM_DFLASH2_LOOKUP_NSTRONG` is 8 with `_AGREE`=2 in optimizations.md:211-213 but 6 with `_AGREE`=0 in the single-user knob table (single-user/README.md:361) — the shipped defaults appear to have moved and the prose was not updated. - The batch/README.md:35 "int8 activations, MLP (default)" older-config row reads 876 e2e while the current table (batch/README.md:14) reads 942 for what is described as the same config; batch/README.md:17-19 explains this ("the difference is everything that landed since"), but the two tables are one page apart and read as a contradiction.

---

## EXISTS, NEVER SET — 6

### Requantize both untied embedding matrices to int8 g128
**Where (theirs):** `docs/optimizations.md:11-13` · `docs/optimizations.md:39-44` · `README.md:268-269` · `README.md:331-332`

**What it does.** Qwen3.8-27B has untied embeddings, so a public W4A16 quant still carries two separate 2.5 GB bf16 matrices — `lm_head` and `embed_tokens` — that the quantizer skipped. `prepare/quant_lm_head.py` and `prepare/quant_embed.py` rewrite both to int8 group-128 in place. Reclaims 2.6 GB of VRAM, which becomes KV cache pages. Round-trip error ~0.6% with no quality regression they could find.

**Mechanism.** Offline, CPU-only, in-place rewrite of the two embedding tensors to int8 with group size 128 (`prepare/quant_lm_head.py`, `prepare/quant_embed.py`, invoked at README.md:331-332). Measured effect on the cumulative ladder: 370 → 516 e2e output tok/s and ~585 tok/s steady-state decode, moving from a 66.7k-token pool to enough pages that 37 requests are resident.

**Why they needed it.** "the public W4A16 quants leave two 2.5 GB bf16 matrices alone. 2.6 GB back." (docs/optimizations.md:12-13) — "two separate 2.5 GB bf16 matrices (lm_head and embed_tokens) that nobody bothered to quantize" (docs/optimizations.md:41-42).

**Their numbers.** ~0.6% round-trip error, no quality regression found. 2.6 GB VRAM recovered. Cumulative ladder at 64 concurrent, 128 in / 512 out, `vllm bench serve` random dataset: baseline W4A16 + fp8 KV = 370 e2e tok/s (at 48 conc, 256/256) with a 66.7k-token pool; + int8 lm_head/embed = 516 e2e, ~585 steady-state decode, 37 requests resident (README.md:266-269).

**llama.cpp — EXISTS, NEVER SET.** llama.cpp never ships an unquantized embedding/head in a quantized GGUF — the mixture at src/llama-quant.cpp:456-471 explicitly forces Q5_K for the IQ2 family, and Q6_K otherwise. So the specific 2.6 GB vLLM was reclaiming does not exist here. What IS unused is the pair of flags that would let us go lower than the mixture chose, and they are unusual in that they return before tensor_type_fallback (src/llama-quant.cpp:684,687) so an incompatible block size is NOT silently corrected for these two tensors. llama-quantize is not staged in C:\AI\llama.cpp-dflash2 — it has to be built first.

**Equivalent here:** llama-quantize --token-embedding-type / --output-tensor-type (offline; requires --allow-requantize from an already-quantized source)

**Evidence (llama.cpp):** `src/llama-model.cpp:1368-1370` · `src/llama-quant.cpp:452-471` · `src/llama-quant.cpp:683-688` · `tools/quantize/quantize.cpp:410-427` · `src/llama-quant.cpp:717`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** token_embd costs zero VRAM here — LLM_TENSOR_TOKEN_EMBD is LAYER_INPUT and dev_input is hard-wired to CPU, so the vLLM saving does not exist on this side. output.weight does go to GPU. The k-quant mixture forces the head to Q5_K for IQ1/IQ2/IQ3_XXS ftypes, so on a UD-IQ2_XXS file it is already ~5.5 bpw, not bf16. Upside is bounded by (head bytes at Q5_K − head bytes at a lower type); dump the tensor types before assuming any. Unknown MiB until measured.

### Requantize the MTP draft module (int8, then GPTQ-int4)
**Where (theirs):** `docs/optimizations.md:20-22` · `docs/optimizations.md:70-77` · `single-user/README.md:238-243` · `single-user/README.md:248` · `single-user/README.md:257`

**What it does.** Qwen's shipped MTP draft module is bf16 (850 MB) and every draft token also runs the full 248k-row lm_head (1.3 GB), so each extra draft cost ~3 ms — MTP-3 was already slower than MTP-2. `prepare/quant_mtp.py` requantizes the module to int8; the shipped fast variant goes further with GPTQ int4 calibrated on the model's own hidden states. A draft then costs ~0.5-1 ms and four of them pay off.

**Mechanism.** Offline requantization of the draft module's weights (`prepare/quant_mtp.py`), plus the GPTQ-int4 variant fetched by `prepare/fetch_fast_variant.py`. Combined with the truncated draft head (next technique), the per-step cost falls from ~32 ms to ~24 ms while carrying more drafts.

**Why they needed it.** "each extra draft cost ~3 ms and MTP-3 was already slower than MTP-2" (docs/optimizations.md:73). And on why the precision is safe: "The MTP module's precision never touches output quality (drafts are verified exactly); it only moves acceptance, and the calibrated int4 keeps it." (docs/quality.md:60-61)

**Their numbers.** 850 MB bf16 module + 1.3 GB int8 head → ~3 ms per draft as shipped; ~0.5-1 ms after. Step time ~32 ms → ~24 ms (single-user/README.md:264). GPTQ-int4 MTP module (shipped fast variant) reads ~114 / ~124 tok/s (default / greedy) at C1 vs 109 / 112 for int4 lm_head alone, 2.8 / 3.0 tokens per step, acceptance at position 0 74% / 77% (single-user/README.md:256-257). Counter-experiment: keeping `mtp.fc` in bf16 with the rest int8 gives 88 / 96 tok/s and *worse* acceptance (67% / 70%) than the fully-int8 module (single-user/README.md:253).

**llama.cpp — EXISTS, NEVER SET.** The economics transfer exactly and are if anything sharper here than on a 24 GB card. What is unverified is whether llama-quantize handles an arch=='dflash' GGUF's tensors sensibly, and whether a requantized sidecar still satisfies the dflash metadata reads (dflash.block_size, sample_from_anchor, selector_top_k) — those are metadata, so they should survive, but the quantiser's mixture rules were not written with this arch in mind.

**Equivalent here:** llama-quantize on the draft/sidecar GGUF; -ngld / -otd / -cmoed control its placement

**Evidence (llama.cpp):** `common/speculative.cpp:2468` · `src/llama-model.cpp:1745-1747` · `tools/server/server-context.cpp:1032-1087` · `common/arg.cpp:4117-4144` · `common/speculative.cpp:966-980`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Every MiB the dflash sidecar occupies is a MiB taken off the target, one for one: --fit never sizes the draft (it is loaded with llama_model_load_from_file directly and -ngld 'auto' resolves to all layers), and the server compensates by adding the measured draft footprint to fit_params_target so the TARGET shrinks. On 9.5 GB free with a 6.77 GB target, a 1-2 GB sidecar is the difference between a usable context and a stub. Highest-leverage untried item in this list after the n_max sweep.

### k=4 is the knee, but only on FlashAttention — FlashInfer dies at k=4
**Where (theirs):** `README.md:297-303` · `single-user/README.md:275-294` · `single-user/README.md:287-290` · `docs/optimizations.md:96-101`

**What it does.** Speculative depth was swept: k=4 is the fastest, k=5 and k=6 lose. But on the FlashInfer backend — the only one supporting fp8 KV on Ampere, and fp8 KV is what makes 150k context fit — vLLM 0.27.1 dies with an illegal memory access as soon as one request finishes while another is mid-generation at 4 drafts. So the repo ships two configs: `CTX=fast` (FlashAttention, bf16 KV, 64k, k=4) and `CTX=long` (FlashInfer, fp8 KV, 150k, k=3), giving up ~7%.

**Mechanism.** `DRAFT_TOKENS` selects k (default 4, 3 for `CTX=long`/`huge`). vLLM PR #50021 is vendored as `patches/vllm-pr50021-gdn-spec-bounds.patch` for bounds checks in the DeltaNet speculative-decode kernels — it fixes an illegal memory access hit with several concurrent MTP requests, but explicitly "does not cure" the FlashInfer k=4 crash (single-user/README.md:280). The same k=4 config runs clean at C2/C4 on FlashAttention, which localises the bug to the FlashInfer spec-decode path.

**Why they needed it.** "on vLLM 0.27.1's FlashInfer backend (needed for fp8 KV, i.e. for 150k context) four drafts crash the engine with an illegal memory access as soon as one request finishes while another is mid-generation — club-3090 reports the same 'n=4 eventually dies, n=3 stable' pattern" (README.md:299-302). Cross-referenced to vLLM issues #40756 and #36498 (single-user/README.md:282-283).

**Their numbers.** k=4: ~114 / ~124 tok/s at C1. k=5: 106 / 105, 3.0 / 2.9 tokens per step. k=6: 76 / 94, 2.3 / 2.7. k=3 (`CTX=long`): 84 / 89 with base requant, 95 / 100 with the fast variant — "gives up ~7%". `DRAFT_TOKENS=2` costs ~5% and is the most conservative setting. k=3 passed every concurrency soak run (C2/C4/C8 with staggered finishes, 100k-token prompt, 4×6k-token generations) (single-user/README.md:292-294).

**llama.cpp — EXISTS, NEVER SET.** The vLLM finding is 'sweep k, the knee is real, and it is backend-dependent'. That transfers wholesale; only the backend hazards differ. There is no FlashInfer-class crash here, but there are three quantisation stairs and one silent memory multiplier, all named in the map or in the code above. Anything found at one context depth must be re-run at another — this project has already recorded a spec verdict flipping sign between 16K and 131,072.

**Equivalent here:** --spec-draft-n-max N (env LLAMA_ARG_SPEC_DRAFT_N_MAX), default 3

**Evidence (llama.cpp):** `common/arg.cpp:4076-4085` · `common/common.h:325` · `common/speculative.cpp:988-996` · `common/speculative.cpp:1181-1186` · `ggml/src/ggml-cuda/fattn.cu:469` · `ggml/src/ggml-cuda/mmvq.cu:337` · `src/llama-memory-recurrent.cpp:99-101`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** This is the single cheapest untried lever we have. The default is 3; a stock 16-wide DFlash sidecar allows up to 15 (16 for anchor-sampling DSpark), and the DFlash speculator drafts exactly params.n_max tokens every step, so n_max is literally the verify-block length. The sweep is not monotone for three independent reasons this build makes explicit: (a) quantized-KV FA leaves VEC at 1+n_max >= 3; (b) the weight matmuls leave MMVQ for MMQ at 1+n_max > 8; (c) the target's recurrent state is allocated at (1+n_max) rows. Read the effective value off the startup line `- n_max=%d, n_min=%d, p_min=%.2f`, and the memory cost off `RS buffer size`.

### GPTQ-quantize the DFlash2 drafter to W4A16 with a calibration set hooked from inside vLLM
**Where (theirs):** `docs/optimizations.md:136-146` · `single-user/README.md:78-81`

**What it does.** The DFlash2 drafter is 1.92B parameters, 3.85 GB in bf16 — read once per step, that is +5 ms on a 3090 and no net gain, and it leaves only a 21k-token KV pool. `drafter/capture_dflash2.py` hooks the drafter's own linear layers *inside vLLM* on 400 real prompts (~290k rows per layer, plus the context-KV precompute's input distribution for the k/v rows) and `drafter/quant_dflash2.py` GPTQ-quantizes the 36 matrices to W4A16 compressed-tensors (Marlin): 1.19 GB.

**Mechanism.** Calibration is captured from the live serving path rather than from an offline forward — the hooks sit on the drafter's linears while vLLM runs 400 real prompts, and the k/v rows get the context-KV precompute's own input distribution, which differs from the ordinary activation distribution. 36 matrices, GPTQ, compressed-tensors/Marlin format. Shipped as `syvai/Qwen3.8-27B-DFlash2-W4A16`, fetched by `prepare/fetch_dflash2.py`.

**Why they needed it.** "read once per step, that is +5 ms on a 3090 and no gain (106 / 112 tok/s, measured), and it leaves a 21k-token KV pool." (docs/optimizations.md:137-138) — bf16 DFlash2 was measured to be *slower* than MTP, so the quantization is what makes the mode exist at all.

**Their numbers.** 3.85 GB → 1.19 GB (1.92B params, 36 matrices). bf16 drafter: +5 ms/step, 106 / 112 tok/s, 21k-token KV pool. Calibration: 400 real prompts, ~290k rows per layer. int4 costs ~5% acceptance at default sampling (3.2 vs 3.4 tokens per step) and nothing at greedy; keeping `fc` in bf16 did not recover it (docs/optimizations.md:144-146).

**llama.cpp — EXISTS, NEVER SET.** The quantiser surface is rich (37 named ftypes, per-tensor regex overrides, --imatrix with include/exclude) and is exactly the right tool for shrinking a sidecar. The missing piece is the calibration provenance, and there is no seam inside common/speculative.cpp where the draft context's activations could be tapped for an imatrix — the draft ctx is constructed and owned entirely inside common_speculative_init_result.

**Equivalent here:** llama-quantize on the sidecar (--tensor-type, --pure, --imatrix all available offline); no in-server calibration hook

**Evidence (llama.cpp):** `tools/quantize/quantize.cpp:314-361` · `tools/quantize/quantize.cpp:436-470` · `common/speculative.cpp:2432-2496` · `tools/server/server-context.cpp:1032-1087`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Same 1:1 VRAM economics as technique 8 — the sidecar is loaded outside --fit and the server shrinks the target to pay for it. The calibration half does NOT transfer: llama.cpp's imatrix is produced by a separate offline tool over a text corpus, there is no hook that captures activations from the live serving path. Expect a plain quantisation, not a calibrated one, and expect an acceptance cost that vLLM measured at ~5 % at temp > 0 and nothing at greedy.

### Pin the KV pool in bytes (`KV_MEM` / `--kv-cache-memory`) instead of by utilization
**Where (theirs):** `docs/optimizations.md:166-168` · `docs/gotchas.md:100-104` · `docs/gotchas.md:239-251` · `single-user/README.md:96-100` · `single-user/README.md:362`

**What it does.** The V2 runner's profiled activation peak swings ~1 GiB between starts of the *same* config, which makes any utilization-based setting non-deterministic. The single-user DFlash2 mode therefore pins the pool by bytes (`KV_MEM`, 5583457484 = 5.2 GiB) rather than by `GPU_UTIL`, and start-up becomes reproducible.

**Mechanism.** `--kv-cache-memory` with a fixed byte count, set from `KV_MEM`. Setting `KV_MEM=` empty falls back to `GPU_UTIL`, which profiles actual free memory instead — the documented escape hatch.

**Why they needed it.** "the runner's profiled activation peak also swings ~1 GiB between starts, [so] this mode pins the pool by bytes ... and start-up is then deterministic (69,758 tokens twice over)." (docs/optimizations.md:166-168). The known failure of pinning: "`KV_MEM` assumes the card is headless, and the failure lands long after startup looks fine" — with a desktop session on the same card (Xorg + compositor + browser ≈ 1.3 GiB) the server starts, captures graphs, reports a pool, and then dies on a real request when the spec-decode `part_o` buffer cannot get its ~1.5 GiB (docs/gotchas.md:239-248).

**Their numbers.** Activation peak varies ~1 GiB between starts of the same config. `KV_MEM` default 5583457484 B = 5.2 GiB → 69,758 tokens, deterministic across restarts. A desktop session costs ~1.3 GiB; `KV_MEM=4000000000` was enough for the reporter of PR #12. `part_o` needs ~1.5 GiB.

**llama.cpp — EXISTS, NEVER SET.** The mechanism is spelled out in fit.h's own contract and confirmed by the branch that prints the no-change line. Note the trap: `-c 0` explicitly is NOT the same as omitting -c — the handler also sets fit_params_min_ctx = UINT32_MAX, which turns fit's context reduction off by a different route and prints `user has requested full context size`. And -c is padded up to a multiple of 256 unconditionally, so pick a multiple of 256 or the pinned value is not the value you asked for.

**Equivalent here:** pass a numeric -c N — fit modifies n_ctx if and only if it is 0, and prints `context size set by user to %u -> no change`

**Evidence (llama.cpp):** `common/fit.h:15-18` · `common/fit.cpp:368-370` · `common/fit.cpp:361-363` · `common/arg.cpp:1641-1644` · `src/llama-context.cpp:288` · `common/fit.cpp:377-379`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Directly aimed at this project's stated worst measurement problem. CLAUDE.md records free VRAM at boot moving 9,326–10,732 MiB with --fit following it, and declares effects below 13.6 % to be noise. A numeric -c removes the context half of that variance outright, at the cost of moving the variance into layer placement instead (fit step 3 will spill layers rather than shrink ctx when memory is tight). Pinning -c AND -fitt together, or -c plus -ngl all, pins both — with -ngl all the layer-placement pass throws and only the context step is skipped, which is now a no-op anyway. This could plausibly take the noise floor well below 13.6 %.

### `--max-num-batched-tokens 2048` — bigger prefill chunks make things worse
**Where (theirs):** `docs/gotchas.md:39-41` · `batch/README.md:83-89`

**What it does.** `--max-num-batched-tokens 8192` inflates the profiled activation peak, which shrinks the cache pool, which caps concurrency. 2048 wins on this card. The consequence for capacity planning is that prefill is a fixed shared resource: chunked prefill feeds everything through the same 2,048-token per-step budget, so concurrency does nothing for prefill and queueing is linear.

**Mechanism.** The chunked-prefill budget is a second-order memory knob, not just a latency knob: it feeds the startup activation profile, which sets the pool, which sets concurrency.

**Why they needed it.** "chunked prefill feeds everything through the same 2,048-token per-step budget, so prompt processing is a fixed resource the whole server shares, and queueing is linear (four 16k prompts at once means the last one waits ~40 s)." (batch/README.md:84-87)

**Their numbers.** 2048 beats 8192 on this card. Prefill is flat in concurrency: 1k inputs 1,812 / 1,820 / 1,806 tok/s at conc 1 / 4 / 8-16; 16k inputs 1,595 / 1,601 / 1,599 (batch/README.md:63-69). Falloff with length is mild — ~45% from 1k to 100k on the int8 path, ~34% on W4A16 — "because just 16 of 64 layers pay quadratic attention; this is one of the places the hybrid architecture genuinely helps."

**llama.cpp — EXISTS, NEVER SET.** The vLLM finding — the prefill chunk budget is a second-order memory knob, not just a latency knob, because it feeds the activation profile that sets the pool — carries over exactly, and llama.cpp makes the coupling explicit at the reserve call site. The concurrency half of their point (prefill is a shared fixed resource, queueing is linear) is moot at -np 1.

**Equivalent here:** -ub / --ubatch-size (default 512) and -b / --batch-size (default 2048)

**Evidence (llama.cpp):** `src/llama-context.cpp:595` · `src/llama-context.cpp:245` · `src/llama-context.cpp:247` · `common/common.h:443-444` · `common/speculative.cpp:2418-2423`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** -ub is the single knob that sizes the worst-case compute buffer: the reserve builds the prompt-processing graph at n_tokens = min(n_ctx, n_ubatch), so lowering -ub shrinks the compute buffer and hands the difference to KV, at some prefill throughput cost. On a 12 GB card with a 6.77 GB model that is a real trade and we have never swept it. Two silent clamps to respect: -ub is capped at -b, and -b is capped at n_ctx under causal attention, neither with a warning. Also note DFlash forces both upward behind your back, so the sweep must be done with the speculator actually enabled.

## absent, has a seam — 5

### fp16 Gated-DeltaNet recurrent state (`--mamba-ssm-cache-dtype float16`) — the real concurrency bound
**Where (theirs):** `docs/optimizations.md:16-17` · `docs/optimizations.md:49-57` · `README.md:270` · `batch/README.md:32-33` · `docs/quality.md:32-33`

**What it does.** 48 of the 64 layers are Gated DeltaNet, each holding a fixed recurrent state per sequence that Qwen's config asks for in fp32 — ~150 MB per request, allocated up front, read and written every decode step. On this architecture that state, not the KV cache, is what caps concurrency: with `--max-num-seqs 64` only 37 requests were ever actually running. Halving the dtype halves both the footprint and the per-step traffic; all 64 then run.

**Mechanism.** Engine flag `--mamba-ssm-cache-dtype float16` changes the per-sequence recurrent-state allocation dtype for the 48 GDN layers. The diagnostic that found it was a plain log line: `Running: 37 reqs, Waiting: 27` under `--max-num-seqs 64` (docs/optimizations.md:54).

**Why they needed it.** "On this architecture that state — not the KV cache — is what bounds concurrency" (docs/optimizations.md:52-53). And on the dtype choice specifically: "fp16 keeps 10 mantissa bits; we did not use bf16's 7" (docs/optimizations.md:56-57) — i.e. bf16 was rejected on mantissa width, not on range.

**Their numbers.** ~150 MB of fp32 state per request across 48 GDN layers. 37 of 64 requests resident before, 64 after. Perplexity unchanged to three decimals: 8.045 → 8.044 (en 10.68 / da 10.85 / code 3.05 both rows, docs/quality.md:32-33). Batch mode 64-concurrent 128/512: 516 → 707 e2e tok/s, ~585 → ~830 steady-state decode (README.md:269-270, batch/README.md:32-33). On 256/256 at 64 concurrent: 393 → 491 e2e.

**llama.cpp — absent, has a seam.** The seam is three literal arguments (src/llama-model.cpp:2274-2275 pure recurrent, :2314-2315 hybrid_iswa, :2335-2336 plain hybrid) plus a new flag in common/arg.cpp; nothing else reads type_r/type_s. The real finding is not the patch but the law it exposes: on this build the recurrent state does not scale with context, it scales with (1 + n_rs_seq), and n_rs_seq is set from --spec-draft-n-max for exactly the four model-based drafters we now use. Unverified whether the GDN/ssm CUDA kernels accept a non-F32 state — that is the risk, and it is why I say small-patch rather than one-flag.

**Equivalent here:** none — recurrent_type_r / recurrent_type_s are GGML_TYPE_F32 literals at the call site

**Evidence (llama.cpp):** `src/llama-model.cpp:2335-2336` · `src/llama-memory-recurrent.cpp:99-101` · `src/llama-memory-recurrent.cpp:118-127` · `common/common.h:386-392` · `common/common.cpp:1697`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** At -np 1 with ngram-mod only, n_rs_seq=0 so rows = mem_size*(1+0) = 1 and the RS buffer is one state per recurrent layer — halving it saves tens of MiB, not worth a patch. With draft-dflash it changes character: cparams.n_rs_seq = draft.n_max, so rows = 1+n_max. At n_max=3 that is 4× the state; at n_max=15 it is 16×. Read the exact figure off the `RS buffer size` line at startup before and after raising n_max — that is the number this patch would halve.

### Fuse the lookup proposal with the drafter's rather than substituting for it (`_NSTRONG` / `_AGREE`)
**Where (theirs):** `docs/optimizations.md:210-215`

**What it does.** A long match is trusted on its own; a short match is taken only if the drafter independently proposed the same first `_AGREE` tokens. Two independent sources agreeing — the drafter looked at the hidden state, the lookup looked at the text — is the cheap confidence signal, and it is what stops a coincidental 6-token match from costing acceptance on prose.

**Mechanism.** Threshold rule: match length ≥ `VLLM_DFLASH2_LOOKUP_NSTRONG` → take it unconditionally; shorter → require agreement on the first `_AGREE` tokens with the drafter's own proposal. Replaces an all-or-nothing first version.

**Why they needed it.** "Two independent sources agreeing is the cheap confidence signal ... and it is what stops a coincidental 6-token match from costing acceptance on prose, which the all-or-nothing first version did." (docs/optimizations.md:213-215)

**Their numbers.** `_NSTRONG` = 8 and `_AGREE` = 2 per docs/optimizations.md:211-213; the single-user knob table records `_NSTRONG` 6 and `_AGREE` 0 (single-user/README.md:361). `_NMIN_TAIL` (4) is the equivalent minimum for positions the drafter never proposed.

**llama.cpp — absent, has a seam.** CANNOT #3 and #4 in the speculative area foreclose the ensemble and the ordering as configuration, but they do not foreclose it as a patch — and the map names the exact function. The cheap partial is real and should be swept first: llama.cpp's all-or-nothing n_min is a cruder version of vLLM's _NSTRONG, and its current value of 48 is very conservative for code.

**Equivalent here:** nearest expressible approximation is the ngram-mod n_min gate (a trust threshold on match length, all-or-nothing); no cross-impl agreement test exists

**Evidence (llama.cpp):** `common/speculative.cpp:2710-2756` · `common/speculative.cpp:2201-2209` · `common/speculative.cpp:2796-2801` · `common/common.h:351-356`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The threshold half is a one-flag experiment today: --spec-ngram-mod-n-min is the 'how long must a match be before I trust it' knob and it sits at 48. The agreement half requires code. The seam is common/speculative.cpp:2710-2756, where the loop currently breaks on the first non-empty result — running two impls and comparing their first k tokens fits there, but every downstream accounting path assumes exactly one impl produced the draft (impl_last[seq_id], and accept(..., is_other=true) for all the others), so it is not a small change.

### Let a match overlap the suffix it matched, so a repeating pattern is proposed from its own period
**Where (theirs):** `docs/optimizations.md:217-218`

**What it does.** The lookup permits the matched region to overlap the suffix being matched. A repeating pattern — a list marker, an indent, a code fence — is then proposed from its own period rather than being missed because the only occurrence is the one currently being generated.

**Mechanism.** Removes the usual non-overlap constraint in suffix search, so period-p repetitions self-predict.

**Why they needed it.** "a repeating pattern (a list marker, an indent, a fence) is proposed from its own period instead of missed" (docs/optimizations.md:217-218).

**Their numbers.** No isolated figure; folded into the lookup's aggregate numbers.

**llama.cpp — absent, has a seam.** This is the one place in this slice where vLLM found something llama.cpp genuinely lacks and where the seam is a single named line rather than an architecture. Note the tradeoff the constant exists for: mod.add is called once per position, so incremental update costs one hash per generated token — negligible next to a decode step.

**Equivalent here:** none — ngram-mod updates its table in chunks and lags the generated text by up to 32 tokens

**Evidence (llama.cpp):** `common/speculative.cpp:1979-1986` · `common/ngram-mod.cpp:15-35` · `common/speculative.cpp:1992-2004`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Real and specific. I read draft_one: new n-grams are only folded into the table when `sinfo.i_last + 32 < cur_len`, so a repetition whose only occurrence is within the last ~32 generated tokens is invisible to the drafter. For a coding agent emitting repeated indents, list markers, fences, or a run of near-identical lines, that is exactly the window where a period-p pattern would self-predict. Changing the chunking constant (or updating incrementally) is a few lines. Unknown tok/s until measured, but the failure mode is concrete and the fix is local.

### KVarN 4-bit key / 2-bit value KV cache, ported to 0.27.1
**Where (theirs):** `docs/long-context.md:7-35` · `docs/long-context.md:66-74` · `README.md:98-125` · `single-user/README.md:359`

**What it does.** With fp8 KV the pool holds ~200k tokens because 16 attention layers × 4 KV heads × 256 dims × 2 bytes (K+V) is 2 KB per token, so the model's full 262,144 context is out of reach. KVarN (Huawei CSL) applies a Hadamard rotation plus iterative variance normalization and stores 4-bit keys / 2-bit values per 128-token tile — ~840 B/token/layer here. It ships as a fork of vLLM 0.23; `kvarn/` is this repo's port of its dense backend onto 0.27.1.

**Mechanism.** `--kv-cache-dtype kvarn_k4v2_g128 --block-size 128`, installed by `bash kvarn/install.sh`, selected with `KV=kvarn` (batch) or `CTX=huge` (single-user). Because the KV format is an engine-level choice in vLLM it cannot be switched per request, which is why it is a mode rather than a default.

**Why they needed it.** "The way past that is a smaller cache, not a different engine" (docs/long-context.md:12). And on when to take it: "Past ~100k of context you are buying 1.7× the context for less than half the decode rate, which is worth it when the alternative is not fitting the request at all and a bad trade otherwise." (docs/long-context.md:68-71)

**Their numbers.** ~840 B/token/layer vs 2 KB/token for fp8. Batch mode (no speculation): pool ~205-225k tokens → 302-344k with 64 slots, **420k with 4 slots**. Perplexity 8.223 → 8.236 (+0.16%) on 33k tokens en/da/code. Needle-in-a-haystack greedy correct at 4k/16k/30k/100k/240k, both depths. Prefill unchanged within ±5% (1,812/1,595/997 → 1,741/1,569/1,050 tok/s at 1k/16k/100k). Single stream at 100k: TTFT 99 s / 27 ms per token → 94 s / 33 ms (1.22×). Four 60k-token requests: only 3 fit, 256 s total, ITL 33 ms → all 4 resident, 242 s, ITL 49 ms. 64 concurrent short (128/512): 876 → 692 tok/s, 38 resident ("2048-token blocks cost as much per short request as fp8's 800-token block").

**llama.cpp — absent, has a seam.** The seam is nameable (fattn.cu:338-356 for type admission, fattn.cu:264-325 for the compiled instances, CMakeLists.txt:115-125 for what gets built) which is why this is 'possible' rather than 'impossible', but the work is kernel work, not configuration. Also note llama.cpp already applies an automatic Hadamard rotation to quantised K/V when the head dim is a multiple of 64 — the same quality mitigation KVarN uses — and logs whether it engaged as attn_rot_k / attn_rot_v at load. Check those two lines: if the head dim is not a multiple of 64 the rotation is silently skipped and our q4_0 cache degrades more than the flag suggests.

**Equivalent here:** none below q4_0; the seam is ggml_cuda_fattn_kv_type_supported plus the FATTN_VEC_CASES instantiation table

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:338-356` · `ggml/src/ggml-cuda/fattn.cu:264-325` · `ggml/src/ggml-cuda/fattn.cu:442-446` · `ggml/CMakeLists.txt:208` · `src/llama-kv-cache.cpp:308-338`

**Effort:** new-backend · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Not reachable at any sane cost — it needs a new ggml quant type, a matching FA kernel instantiation for both VEC and MMA, and the Hadamard/variance-normalisation machinery. The adjacent cheap thing IS real: this binary was built with GGML_CUDA_FA_ALL_QUANTS=OFF, which is why K and V must be the same type and why q5_0/q5_1 have no kernel. Rebuilding with it ON unlocks asymmetric pairs (e.g. -ctk q8_0 -ctv q4_0) and the q5 family, at the price of compile time and binary size. That is a build-flag experiment we have never run.

### Wait for the GPU to actually be free before starting (`ExecStartPre` gate)
**Where (theirs):** `docs/gotchas.md:12-17`

**What it does.** vLLM profiles free memory once at startup. If the previous process is still releasing VRAM at that moment, the cache pool comes out ~40% smaller and stays that way — no warning, the server runs fine, throughput is just quietly bad. The systemd units in both mode dirs carry an `ExecStartPre` gate that waits for the GPU to be actually free.

**Mechanism.** A pre-start command that polls until the card reports free before letting vLLM's one-shot memory profile run.

**Why they needed it.** "Restart onto a dirty GPU and you silently lose 25%." (docs/gotchas.md:12) — the canonical shape of every fault in this repo: no error, a plausible number.

**Their numbers.** ~40% smaller cache pool → ~25% throughput lost, silently.

**llama.cpp — absent, has a seam.** llama.cpp's fit path loads the model with no_alloc=true and reads ggml_backend_dev_memory for free VRAM at that moment, then subtracts the -fitt margin. It has no gate and no retry, and all fit failures are swallowed into a WARN plus a status, so the process starts anyway with whatever partial mutations already landed. That is precisely 'an instrument that returns a believable number instead of a failure'.

**Equivalent here:** none in llama.cpp — the seam is our own launcher under qwen38-tuning/scripts/, because --fit measures free device memory at the instant it runs

**Evidence (llama.cpp):** `common/fit.cpp:56-57` · `common/fit.cpp:559-563` · `common/fit.cpp:806-809` · `common/common.cpp:1294-1302`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Highest-value hygiene item in the whole slice for this project specifically. CLAUDE.md already records free VRAM at boot moving 9,326–10,732 MiB — a 1.4 GB swing — and states that --fit follows it and that effects below 13.6 % are therefore noise. That entire noise floor is this failure. A poll-until-free gate in the launch script, combined with a pinned -c (technique 22), attacks it from both ends. Nothing about the model changes; the measurement instrument gets several times sharper.

## partial — 11

### Split-KV Triton attention for the multi-query verify step
**Where (theirs):** `docs/optimizations.md:84-89` · `docs/gotchas.md:61-70` · `single-user/README.md:364` · `docs/long-context.md:137-141`

**What it does.** FlashAttention-2 only splits the KV sequence across thread blocks when a request has exactly one query token. A speculative verify step has k+1 query tokens, so FA2's varlen path falls back to one thread block per (request, head): on this 24-head model that is 24 blocks on the 3090's 82 SMs, leaving 58 SMs idle. A ~180-line Triton kernel (`patches/spec-decode-attn.patch`, `VLLM_SPEC_DECODE_ATTN=1`) restores the KV split for multi-query decode.

**Mechanism.** Custom Triton kernel that partitions the KV sequence across thread blocks even when `max_seqlen_q > 1`, writing partial outputs into `part_o` buffers and combining. bf16 KV only in the original; `patches/spec-decode-int8-kv.patch` extends it to the quantized `int8_per_token_head` cache on the Triton backend. vLLM's own Triton unified attention has the same restriction (`max_seqlen_q > 1` → 2-D kernel, docs/gotchas.md:65-66).

**Why they needed it.** "a 24-head model runs attention on 24 of the 3090's 82 SMs" (docs/optimizations.md:87-88). At long context the gap is enormous: "the Triton backend ... otherwise cannot split KV for a multi-query verify at all (`use_3d` is off whenever `max_seqlen_q > 1`, and every DFlash2 step is a verify)" (docs/long-context.md:138-140).

**Their numbers.** FA2 vs the Triton kernel per attention layer: 57 µs → 23 µs at 1.5k context; 1.3 ms → 120 µs at 16k (docs/optimizations.md:88-89). At 128k context with 8 query tokens: 1.3 ms for this kernel vs 7.4 ms for vLLM's unified attention and 10.1 ms for FA2 (docs/long-context.md:140-141). Combined with the sampler patch: +4% at default sampling; on the ladder 90 / 98 → 93 / 99 tok/s (single-user/README.md:249-254).

**llama.cpp — partial.** llama.cpp does not have vLLM's specific defect (one thread block per request-head); its MMA kernel is a proper tensor-core kernel. But it has a different, larger, equally invisible multi-query tax on exactly the same step. There is no flag to force VEC and no flag to keep the cache quantized on the MMA path — CANNOT, foreclosed at fattn-mma-f16.cuh:1962-1963. The actionable consequences are testable today: (a) --spec-draft-n-max 1 keeps quantized-KV attention on VEC; (b) at long context, -ctk f16 -ctv f16 pays 4× the KV VRAM but no per-step dequant, so the crossover against q4_0 moves with depth.

**Equivalent here:** BEST_FATTN_KERNEL_MMA_F16 handles multi-query verify — but with quantized KV it dequantizes the whole cache to F16 every call

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:469` · `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963` · `ggml/src/ggml-cuda/fattn-common.cuh:1022-1084` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912` · `ggml/src/ggml-cuda/fattn.cu:534-568`

**Effort:** new-backend · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No knob, but the diagnosis transfers and is probably the most important thing in this whole slice for us. With -ctk q4_0 -ctv q4_0, VEC is chosen only when Q->ne[1] <= 2, i.e. at most 1 drafted token. Any real speculative step therefore runs MMA_F16, and MMA passes need_f16_K = need_f16_V = true unconditionally, so K and V for the FULL padded n_kv of every layer are expanded to F16 into scratch on every decode step. That cost is O(context) per step and it is paid only when speculating. It is a concrete candidate mechanism for this project's own recorded draft-mtp result of +81 % at 16K and −71 % at 131,072 on the same artifact.

### Decouple the verify block from the drafter's block (`DFLASH_TOKENS`)
**Where (theirs):** `docs/optimizations.md:186-193` · `single-user/README.md:141-150` · `README.md:68-73`

**What it does.** `dflash_config.block_size` is a property of the checkpoint — 8 = one anchor plus the 7 mask tokens DFlash2 was trained for — and vLLM made it the target's verify length as well, so a verbatim copy could never exceed 8 tokens per step. It sat exactly on that ceiling. The drafter now keeps its own 7-token block while the target verifies a longer one (`DFLASH_TOKENS`, 15 → 16 query tokens), with the positions past the drafter's block filled from the context.

**Mechanism.** Separate the two lengths: the drafter still emits the 7 tokens it was trained for (no extra mask tokens, no extra pass of its candidate head), and the tail positions are filled by the lookup, which costs the drafter nothing. "the context is a free source of drafts, the drafter is not" (docs/optimizations.md:192-193).

**Why they needed it.** "That ceiling was binding: reproducing a document's first 60 lines accepted 7.83 of 8 drafts per step, so the copy ran at the block size, not at what the context could support." (single-user/README.md:144-146)

**Their numbers.** Measured at 25k context, greedy, tokens per step / decode tok/s (docs/optimizations.md:223-230): reproduce first 60 lines verbatim 4.72/159 (no lookup) → 7.83/260 (default) → **14.97/381** (`DFLASH_TOKENS=15`); "shorten this, keep the commands" 2.70/90 → 3.19/107 → 3.50/113; quote-and-explain 3.01/101 → 3.21/107 → 3.35/110; reproduce every command 4.62/153 → 5.23/173 → 5.32/166; free-form summary/QA 2.15/72 → 2.08/69 → 2.13/71; C1 8 short prompts 3.22/126 → 3.33/131 → 3.42/133. Net: **+47%** where the model reproduces its context. Costs 4 request slots instead of 8 and 56k of context instead of 64k.

**llama.cpp — partial.** llama.cpp gives a fallback chain, not an ensemble, and the priority order is rebuilt from a bitmask so the order you type is discarded — every n-gram speculator outranks every model-based one. So today ngram-mod wins whenever it clears its n_min gate and draft-dflash only runs when it does not. That is architecturally the opposite of vLLM's arrangement (drafter head + lookup tail) and is why the fusion is a large-patch rather than a config.

**Equivalent here:** --spec-draft-n-max sets the dflash verify block (up to block_size-1); ngram-mod's n_max sets a separate one; they cannot be combined into one draft

**Evidence (llama.cpp):** `common/speculative.cpp:2725-2726` · `common/speculative.cpp:2753-2755` · `common/speculative.cpp:2542-2552` · `common/speculative.cpp:2728-2733` · `common/speculative.cpp:1181-1186`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The tunable half is free and unused: n_max defaults to 3 while the sidecar allows 15, and the dflash draft() emits exactly n_max tokens every step, so n_max IS the verify block. The fusion half — filling the tail of a long dflash block from the n-gram table — does not exist and would need a real patch. The seam is the fallback loop in common_speculative_draft: it sets dp.drafting=false and breaks as soon as one impl returns non-empty. Turning that break into a concatenate-until-budget is the whole change, plus the truncation at 2728-2732 and the per-position accounting.

### Schedule the long verify block only while a copy is actually running
**Where (theirs):** `docs/optimizations.md:195-208` · `single-user/README.md:152-156` · `single-user/README.md:204-209`

**What it does.** Each extra verify position costs about 1 ms of attention at 25k context, so the long block must not be paid for on ordinary prose. The speculator reports per step how many of its proposals the scheduler should put up for verification: the drafter's 7 normally, the whole block only when (a) the lookup has a match with enough tokens left to fill the tail and (b) the step that just finished emitted at least a full short block's worth of tokens — **twice in a row**.

**Mechanism.** A per-step count returned by `next_num_draft_tokens`, computed from device tensors the replayed Triton kernels wrote. The two-in-a-row condition is the discriminator: "A single saturated step happens inside ordinary prose and the block it buys is wasted; two in a row is a copy." (docs/optimizations.md:201-202). The flags are read from a **pinned copy that landed asynchronously, one step stale** — deliberately, because reading them synchronously is a device synchronise on every decode step.

**Why they needed it.** Each extra verify position ≈ 1 ms of attention at 25k. Reading the flags synchronously "is a device synchronise on every decode step and measured 5%, more than the long block is worth on most work" (docs/optimizations.md:203-204) — so a one-step-stale flag is chosen on purpose.

**Their numbers.** ~1 ms of attention per extra verify position at 25k context. Synchronous flag read costs 5%. Against the same server with the long block disabled the trigger is a gain on every task: **+55% reproducing a document, +10% rewriting one, +2-3% on prose** (docs/optimizations.md:207-208).

**llama.cpp — partial.** llama.cpp has the per-step early stop but not the workload discriminator (vLLM's 'two saturated steps in a row means a copy'). The discriminator would need per-step host state, which llama.cpp can carry easily since nothing here runs inside a captured graph — the seam is the impl's own accept()/draft() pair plus dp.n_max. But p_min covers most of the benefit for one flag, so try it before writing anything.

**Equivalent here:** --spec-draft-p-min (alias --draft-p-min), default 0.00 = disabled; plus the automatic per-step dp.n_max context-edge budget

**Evidence (llama.cpp):** `common/arg.cpp:4101-4107` · `common/common.h:329` · `common/speculative.cpp:1254-1256` · `common/speculative.cpp:1262-1271` · `common/arg.cpp:4098` · `tools/server/server-context.cpp:441-460`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** p_min is a live, unset flag that does the confidence half of this: stop extending the draft once the top candidate's probability falls below P. For DFlash2 specifically it compares the softmax of the selector lattice at the argmax — the map notes this comparison was added by this very commit, i.e. it is the newest and least-exercised knob in the area. It is the cheapest way to stop paying for a long block on prose. Warning: it is parsed with std::stof and never range-checked, so p_min > 1.0 kills every draft silently and yields a plausible slow server.

### Sticky hold on the long block (`VLLM_DFLASH2_LOOKUP_STICKY`), single-request only
**Where (theirs):** `single-user/README.md:158-187` · `docs/gotchas.md:192-204` · `single-user/README.md:361`

**What it does.** The lookup flag drops out for reasons unrelated to the copy ending — a line the lookup cannot match, or a flag copy that had not landed yet — and dropping the long block immediately costs the two steps needed to re-earn it. So the long block is held for `_STICKY` (3) more steps. Because the counter is one number for the whole batch, the hold is applied **only with one request in flight**.

**Mechanism.** A batch-wide step counter that keeps the long block on through steps where the flags say no. The reason it must be gated to batch 1: "the block is one chunk through the recurrent layers, so that changes its greedy text" (docs/gotchas.md:198-199) — with several requests in flight, which block length a copying request gets depends on when the *other* requests arrived, and the resulting text differs. `bench/labd_soak.py` caught exactly this: a verbatim copy coming out differently in two rounds of an identical four-way batch, and OK in three of three with the hold off.

**Why they needed it.** "That is not a tuning knob for the average; it is what makes the mode reproducible." (single-user/README.md:161-162) — and the rule it generalises to: "Any controller state that outlives one step has to be per-request, or batch > 1 stops being reproducible." (docs/gotchas.md:192-193) The proper fix (per-request draft counts) is blocked: `get_uniform_token_count` in `gpu/cudagraph_utils.py` will not dispatch a graph for a ragged batch, which runs piecewise and costs 8% — more than the hold is worth.

**Their numbers.** `_STICKY` = 3 steps. Six consecutive runs of the same 25k-document prompt, greedy, hold off vs hold on (single-user/README.md:167-170): reproduce first 60 lines 13.92 ×6 / 362 → **15.21 ×6 / 379 (+5%)**; "quote and explain" 2.86 ×6 / 93 → 2.87-3.44 / 105 mean (+12%); free-form summary 2.13 / 71.1 unchanged; free-form Q&A 2.02 / 68.3 → 68.2 unchanged. GSM8K with the hold: 96.0% vs 96.5% — "one question, which is what a 200-question sample resolves" (single-user/README.md:218-219). Ragged-batch alternative costs 8%.

**llama.cpp — partial.** vLLM's rule — any controller state outliving one step must be per-request or batch > 1 stops being reproducible — is correct and llama.cpp violates it with the shared table. It costs us nothing at -np 1, but the auto-reset is a live behaviour we have never instrumented, and a table reset mid-run would show up as a throughput cliff with no log line at default verbosity.

**Equivalent here:** ngram-mod carries per-seq state (i_last, n_draft_last, n_low) but its 4M-entry hash table is shared across all sequences

**Evidence (llama.cpp):** `common/speculative.cpp:1914` · `common/speculative.cpp:2044-2054` · `common/speculative.cpp:1952-1957` · `common/speculative.cpp:1899-1905`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** At -np 1 the reproducibility hazard is moot — one sequence, no cross-request contamination. What IS live and worth watching is llama.cpp's own version of controller state outliving a step: the table resets itself on two automatic triggers, occupancy > 0.25 at begin(), and five consecutive accept rounds with acceptance below 0.25 (the comment says 0.5, the code says 0.25). Five bad rounds wipes the whole table and zeroes i_last, and on a coding agent that alternates between reproducing a file and writing prose that could be firing repeatedly. Neither threshold has a flag. Watch for it at LOG_TRC.

### vLLM's built-in per-token-head KV modes, measured — and why `int4_per_token_head` is the zero-install 262k option
**Where (theirs):** `docs/long-context.md:82-112`

**What it does.** vLLM 0.27.1 ships `int8_per_token_head`, `fp8_per_token_head` and `int4_per_token_head` (dynamic per-token, per-head scales; the int4 one with a rotation and asymmetric zero-points), all only in the Triton attention backend. Measured against fp8 and KVarN in the batch config: `int8_per_token_head` buys nothing over fp8 (same byte per element) and costs the Triton backend's long-context speed; `int4_per_token_head` is a genuine zero-install alternative to KVarN for the 262k use case.

**Mechanism.** `--kv-cache-dtype int4_per_token_head --attention-backend TRITON_ATTN --max-model-len 262144` (batch/start_qwen.sh: `KV=int4pth`). `fp8_per_token_head` does not start on sm86 at all — Triton's fp8 KV needs SM89+.

**Why they needed it.** The cost is attributed precisely: int4pth is slower "because vLLM's Triton attention is that much slower than FlashInfer/FlashAttention on this card at long context (the same backend tax the single-user mode avoids by staying on FlashAttention)" (docs/long-context.md:106-108) — i.e. the KV format is not what costs the time, the backend it forces you onto is. "If the Triton backend catches up, it becomes the simpler choice."

**Their numbers.** Batch mode at 0.93 util (docs/long-context.md:92-99): pool 164k (fp8) / 178k (int8pth) / **355k (int4pth, 262k fits, 1.35×)** / 302-420k (KVarN). Perplexity 8.235 / 8.231 / 8.257 (+0.3%) / +0.16%. Needle greedy: 100k ok for all; int4pth and KVarN also pass 240k. Prefill 1k/16k: 1,773/1,601 → 1,739/1,187 → 1,710/1,194 → 1,741/1,569 tok/s. 100k single stream: TTFT 100 s / 26.8 ms → 231 s / 40.8 ms → 220 s / 41.4 ms → 94 s / 33 ms. 64 concurrent short: 839 / 850 / 835 / 692 tok/s. Summary: int4pth is 2.3× the prefill time and 1.5× the decode time at 100k. TurboQuant (`turboquant_4bit_nc`) gives a similar 413k pool and ~15% slower decode, but its chunked-prefill path allocates O(context) scratch outside the memory profile and OOMs at 32k+ prompts at 0.972 and at 128k even at 0.90 (docs/long-context.md:76-80).

**llama.cpp — partial.** llama.cpp's parser accepts nine types but only five run on the GPU; --help advertises iq4_nl, q4_1, q5_0 and q5_1 as if usable and they have no CUDA FA kernel here. That is a live instrument fault of exactly the kind this project catalogues: pass -ctk iq4_nl and you get a server, not an error. Worth an entry in our own register.

**Equivalent here:** -ctk / -ctv with a nine-value parser whitelist, of which only f16, bf16, q4_0, q8_0 (and f32, upconverted) have a CUDA FA kernel in this build

**Evidence (llama.cpp):** `common/arg.cpp:305-315` · `ggml/src/ggml-cuda/fattn.cu:340-357` · `ggml/src/ggml-cuda/fattn.cu:321-325` · `ggml/src/ggml-cuda/fattn.cu:442-446`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The unswept options are exactly two: -ctk q8_0 -ctv q8_0 (double the KV bytes of q4_0, better quality, same kernel class) and the FA_ALL_QUANTS rebuild for q5_0/q5_1 and asymmetric pairs. On 12 GB with a 6.77 GB model, q8_0 probably costs more context than it is worth, but it is the honest control for any quality claim about q4_0. The important carry-over is vLLM's attribution lesson: the KV format was not what cost the time, the backend it forced them onto was. Here the equivalent is the VEC/MMA flip — the same trap in a different shape.

### `SPEC=dflash2 CTX=long` — int8 KV on Triton roughly doubles the DFlash2 context, and is only worth it for reproduction
**Where (theirs):** `docs/long-context.md:114-172` · `single-user/README.md:101-105`

**What it does.** The block drafter was pinned to `CTX=fast` (bf16 KV on FlashAttention, 64k at `DFLASH_TOKENS=7`, 56k at 15) because bf16 KV is 64 KB per token and the pinned 5.2 GiB pool is exactly that much. `CTX=long` moves it to an `int8_per_token_head` cache on the Triton backend and roughly doubles the context — but costs about 2:1 on everything except verbatim reproduction.

**Mechanism.** Two patches make it work and neither changes anything at bf16: `hybrid-sw-block-promote.patch` (previous entry) and `spec-decode-int8-kv.patch`, which teaches the split-KV verify kernel to read the quantized cache and wires it into the Triton backend — which otherwise cannot split KV for a multi-query verify at all.

**Why they needed it.** The cost is diagnosed rather than accepted: "What costs the mode is the step, at 91.7 ms against bf16's 48.2 ms at 50k — the Triton backend, the drafter's own five layers, and the int8 kernel's padded-stride penalty (its head dim is 260 B, so odd KV heads start off a 16-byte boundary; ~13% end to end, and reading the cache as int32 instead would recover most of it)." (docs/long-context.md:167-171). Prefill is the larger cost and this kernel cannot help there — a prefill chunk is 2048 query tokens, far above the block sizes it is for.

**Their numbers.** Context: 69,758 → **138,696** tokens at `DFLASH_TOKENS=7` (136,429 with prefix caching); 57,669 → **114,224** at 15. Per attention layer at 128k, 8 query tokens: 1.3 ms (this kernel) vs 7.4 ms (vLLM unified attention) vs 10.1 ms (FA2). On 112,655-token prompts with `PREFIX_CACHE=1`, `dflash2 CTX=long` (k=15) vs `mtp CTX=long` (docs/long-context.md:149-158): reproduce verbatim 14.19 tok/step / **154.8 tok/s** vs 3.81 / 101.4; list every command 5.32 / 69.8 vs 3.52 / **93.6**; rewrite-but-keep 5.09 / 66.8 vs 3.79 / **100.6**; quote-and-explain 2.17 / 34.6 vs 2.75 / **72.9**; summarize 2.17 / 34.8 vs 2.68 / **71.4**; answer a question 2.01 / 32.1 vs 2.34 / **61.9**; all six 3.10 / 47.0 vs 2.95 / **78.6**; TTFT first/cached turn 316.8 s / 6.1 s vs **151.9 s / 2.4 s**. Verdict: +53% on reproduction, about 2:1 behind everywhere else, twice the TTFT. Step time 91.7 ms vs bf16's 48.2 ms at 50k. Padded-stride penalty ~13% end to end (head dim 260 B).

**llama.cpp — partial.** The cost decomposition does not transfer (there is no Triton backend tax, no padded-stride penalty from a 260-byte head dim), but the workload-sensitivity finding does, and it is the sort of thing a single blended benchmark hides. Also transferable: their observation that prefill is the larger cost at long context and the verify kernel cannot help there — the same is true here, since a prompt-processing ubatch runs MMA regardless of speculation.

**Equivalent here:** the same trade is already made by our profile: -ctk q4_0 -ctv q4_0 with draft-dflash. There is no backend switch to pay for

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:469` · `src/llama-context.cpp:595` · `tools/server/server-context.cpp:634-637` · `common/speculative.cpp:2413-2427`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** We are already living in the equivalent configuration, but without vLLM's per-task breakdown. Their table is the useful part: the block drafter wins hugely on verbatim reproduction (+53 %) and loses roughly 2:1 on prose, quoting and summarising. A coding agent does both. Our +34.7 % is a single blended figure over 'real code'; splitting it by task class — reproduce / edit-and-keep / explain / free-form — would tell us whether draft-dflash is worth its VRAM on our actual workload or only on one slice of it.

### `gpu-memory-utilization 0.93` for the MTP path — the spec-decode DeltaNet workspace outgrows the startup profile
**Where (theirs):** `docs/gotchas.md:22-27` · `single-user/README.md:368` · `docs/gotchas.md:46-48`

**What it does.** Even with expandable segments, single-user mode runs at 0.93 rather than batch mode's 0.972: the speculative-decode path's DeltaNet workspace grows beyond what vLLM's startup memory profiling measures, and the engine dies mid-request on long generations at 0.95+. It survives short benchmarks, "which is exactly how it fools you."

**Mechanism.** Hand back headroom the startup profile cannot see. Soak-tested at 0.93 with a 100k-token prompt plus 6k-token generations at 4 concurrent. Related: `prompt_logprobs` on long prompts OOMs at 0.972 (a 300-token prompt needs ~300 MB of fp32 logits and there is no headroom), so quality checks also run at 0.93.

**Why they needed it.** "the engine dies mid-request on long generations at 0.95+. It survives short benchmarks, which is exactly how it fools you." (docs/gotchas.md:25-26)

**Their numbers.** 0.93 vs 0.972. Dies at 0.95+ on long generations. Soak: 100k-token prompt + 6k-token generations at 4 concurrent. `prompt_logprobs`: a 300-token prompt needs ~300 MB of fp32 logits.

**llama.cpp — partial.** vLLM's rule — hand back headroom the startup profile cannot see, and note that it survives short benchmarks, which is exactly how it fools you — is a good rule here too because the same two-phase structure exists (fit measures the target, speculation is initialised afterwards). The difference is that llama.cpp mostly re-reserves rather than discovering at run time, so the risk is smaller but not obviously zero.

**Equivalent here:** -fitt / --fit-target is the explicit margin; the reserve is what would have to miss something for the failure to occur

**Evidence (llama.cpp):** `common/speculative.cpp:2418-2423` · `common/speculative.cpp:2468` · `tools/server/server-context.cpp:1032-1087` · `common/arg.cpp:2851-2874`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The warning transfers even though the mechanism is better here. Two things enlarge memory AFTER --fit has measured the target: DFlash/DSpark force n_batch and n_ubatch up to n_parallel*(n_max+1), and the draft model is loaded outside --fit entirely with -ngld defaulting to all-layers-on-GPU. The server compensates for the second by adding the measured draft footprint to fit_params_target, so the target shrinks; I could not establish from the map whether the batch widening is accounted for. Until it is, raising --spec-draft-n-max should be treated as also raising the compute buffer, and -fitt should not be cut to the bone in the same experiment.

### `MAX_SEQS` is what makes `CTX=huge` single-user, and slots are nearly free
**Where (theirs):** `README.md:159-171`

**What it does.** Fire 8 concurrent requests at `CTX=huge` and the server runs **two**, with the other six queued — because this mode sets `MAX_SEQS=2`. That is a deliberate default for long-document sessions, not an engine limit and not a property of the block verify. `MAX_SEQS=8` lifts it: peak 5 concurrent on the same 8-stream test, with the KV pool **unchanged at 268,169 tokens**.

**Mechanism.** A recurrent-state slot costs ~8 MiB, so slots are close to free; what scales with the pool is the verify block length, not the slot count. Raising `MAX_SEQS` grows the *captured decode graphs*, which is the actual reason the default stays low.

**Why they needed it.** Corrects the natural inference that a low slot count is buying context. It is buying graph memory. Cross-references gotcha 33's measurement (`MAX_SEQS` 1 vs 8 moves the fixed term ~8 MiB total).

**Their numbers.** `MAX_SEQS=2` default → 2 of 8 concurrent run. `MAX_SEQS=8` → peak 5 concurrent, pool unchanged at 268,169 tokens. A recurrent-state slot ≈ 8 MiB. `DFLASH_TOKENS=15` does not boot at 240k on 24 GB, so reproduction mode and 240k are mutually exclusive.

**llama.cpp — partial.** vLLM's correction (a recurrent-state slot is ~8 MiB so slots are close to free; what scales is the verify block) is true of the recurrent half here too — mem_size = max(1, n_seq_max) is only one factor of n_rows — but it is false of the attention half, which is divided per slot. So the combined answer is different and the vLLM conclusion must not be carried over.

**Equivalent here:** -np / --parallel; but slots are NOT free here — without -kvu, n_ctx_seq = n_ctx / n_seq_max

**Evidence (llama.cpp):** `src/llama-context.cpp:290-303` · `tools/server/server.cpp:151-156` · `src/llama-memory-recurrent.cpp:99-101` · `tools/server/server-context.cpp:1600-1602` · `tools/server/server-context.cpp:2365-2369`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** The economics invert, so do not port the conclusion. On llama.cpp a slot costs a full share of the context: with kv_unified false, n_ctx_seq is n_ctx divided by n_seq_max and padded to 256, and n_ctx is then rewritten down to n_ctx_seq * n_seq_max with only a warning. Keep -np 1. One thing to know: OMITTING -np is not the same as -np 1 — the server turns a negative n_parallel into n_parallel = 4 AND kv_unified = true, which changes slot geometry, enables the idle-slot VRAM clear, and enables try_clear_idle_slots. That is the largest single behavioural difference in the server area between our profile and the default.

### Tool-call parser must be `qwen3_coder`, not `hermes`
**Where (theirs):** `README.md:377-381` · `single-user/README.md:370` · `batch/README.md:151`

**What it does.** Both launchers set `--enable-auto-tool-choice --tool-call-parser qwen3_coder`. The parser has to read Qwen's XML call format, which is what this model's chat template emits — not the JSON that `hermes` reads. Picking `hermes` fails *silently*.

**Mechanism.** `TOOL_PARSER` env var, default `qwen3_coder`. `TOOLS=0` turns tool calling off entirely, after which `tool_choice: "auto"` returns HTTP 400.

**Why they needed it.** "`hermes` parses the JSON a Qwen model does *not* produce here, and fails silently." (single-user/README.md:370) — the tool calls simply never materialise rather than erroring.

**Their numbers.** None.

**llama.cpp — partial.** The specific parser-name trap is vLLM's, and I could not establish from the map whether llama.cpp exposes an equivalent tool-call parser selector — that is in map_gaps. But the failure vLLM describes (silently wrong call format, no error) has an unambiguous llama.cpp counterpart in template fallback, and llama.cpp gives two direct instruments for it that we are not using.

**Equivalent here:** --jinja (default TRUE for the server) plus --chat-template / --chat-template-file; the map does not expose a parser-selection flag

**Evidence (llama.cpp):** `common/chat.cpp:759-781` · `common/chat.cpp:776-781` · `common/arg.cpp:1394-1398` · `tools/server/server-context.cpp:4580-4629` · `tools/server/server-context.cpp:4876-4886`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Directly relevant to serving a coding agent and easy to get silently wrong. The template resolution order is: explicit override, else the GGUF tokenizer.chat_template, else the GGUF tool_use variant, else built-in CHATML — and a literal source of "chatml" is treated as empty and falls through to that same chain. So a model whose GGUF lost its template serves ChatML-shaped output and the tool calls simply never materialise. Two unused endpoints answer the question in seconds: GET /props reports chat_template and chat_template_caps, and POST /apply-template returns exactly the prompt the template produces for a given chat body. Run both before trusting tool calling.

### The V2 runner's API-surface gaps (`thinking_token_budget`) and cold-start Triton JIT
**Where (theirs):** `single-user/README.md:106-110` · `docs/gotchas.md:104-107`

**What it does.** Records what changes when `SPEC=dflash2` moves the server onto vLLM's V2 model runner: it rejects the `thinking_token_budget` request parameter with HTTP 400 (and elsewhere is noted as answering it with 400), while everything else the repo uses — logprobs, prompt_logprobs, n, stop, seeds, structured outputs, penalties, streaming, thinking — was checked and passed 12/12. The first request after a cold start JIT-compiles four Triton kernels.

**Mechanism.** A `bench/api_smoke.py`-style compatibility sweep run against the V2 runner. Triton JIT artifacts cache in `~/.triton`.

**Why they needed it.** A mode switch that silently changes the API contract is the kind of thing that breaks a client rather than a benchmark; the repo enumerates exactly one incompatibility rather than saying "mostly compatible".

**Their numbers.** 12/12 parameters pass; one rejected (`thinking_token_budget`, HTTP 400). Cold-start Triton JIT ~5 s once, four kernels, cached in `~/.triton`.

**llama.cpp — partial.** The mode-switch-changes-the-contract lesson transfers as 'speculation is process-lifetime here, full stop', which is a stronger constraint than vLLM's. The JIT lesson maps onto graph eviction rather than kernel compilation, but has the same practical consequence for a benchmark harness.

**Equivalent here:** no runner switch; but enabling speculation does change server-visible behaviour, and there is a real cold-start effect from CUDA graph eviction

**Evidence (llama.cpp):** `tools/server/server-schema.cpp:197-198` · `tools/server/server-schema.cpp:227` · `ggml/src/ggml-cuda/common.cuh:1435-1444` · `ggml/src/ggml-cuda/ggml-cuda.cu:4253-4262`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Two carry-overs. (1) The API-contract one: llama.cpp has NO per-request speculative parameters at all — every speculative field in the request schema is inside an #if 0 block with the comment 'to keep things simple, we disable speculative parameter adjustments for now'. Speculation is a process-lifetime setting, so a per-profile server restart is the only way to change it, which shapes how a benchmark queue must be built. (2) The cold-start one: a captured CUDA graph is destroyed after 10 s unused (swept every 5 s), and re-arming needs two consecutive identical calls, so the first request after an idle gap pays a full re-capture. Warm the server before timing anything, and do not compare a first request to a steady-state one.

### Container image parity and the WSL2 memory-gate fallback
**Where (theirs):** `docs/docker.md:7-53` · `docs/docker.md:55-90`

**What it does.** The container is the same stack frozen: Python 3.12 venv, vLLM 0.27.1 pinned (torch 2.13 / cu130 / Triton 3.7.1), every patch in `patches/` applied and `verify.sh --install` run at build time, KVarN preinstalled. Measured in the container on the 3090 it matches the venv install — no container tax. An independent WSL2 reproduction at `e81fa39` validated all six launch configurations.

**Mechanism.** Compose profiles map to launch scripts (`single` → `single-user/start_qwen.sh`, `batch` → `batch/start_qwen.sh`), one at a time on one GPU. The entrypoint runs `verify.sh --no-server` before every start and refuses to serve on a FAIL. The image carries an nvcc (CUDA "base" + `cuda-nvcc`, not the 8 GB "devel" image) because FlashInfer JIT-compiles its fp8-KV attention kernel on first use and Triton needs a C compiler for its launchers. The 250 W power limit is a host setting (`sudo nvidia-smi -pl 250`) the container cannot set.

**Why they needed it.** On WSL2 the batch default may fail vLLM's startup free-memory gate: on an otherwise clean card WSL reported 22.75/24.0 GiB free, less than the 23.33 GiB requested by `GPU_UTIL=0.972`. "Keep 0.972 as the tuned native-Linux default; 0.93 is a WSL fallback." (docs/docker.md:80)

**Their numbers.** Image 9.5 GB; prepare step ~20 GB + a few minutes; first start 2-3 min of torch.compile/CUDA-graph/FlashInfer JIT, later starts ~1 min; healthcheck start period 15 min. Container measurements: single-user 112.6 / 115.7 tok/s (e2e / decode, default sampling), batch 950 tok/s on the 128/512 × 64 row, same KV pools as the venv. WSL2 gate: 22.75/24.0 GiB free vs 23.33 GiB requested; `GPU_UTIL=0.93` retains a 201,832-token FP8 pool. WSL2 six-profile matrix (docs/docker.md:64-71): single-long 159,326 tokens / 95.39 tok/s greedy C1; single-fast 93,791 / 114.17; single-huge 320,000 cold, 327,272 warm / 79.84-81.66 C1 sampled; batch 201,832 / 1,041.99-1,038.25 C64; batch KV=int4pth 437,414 / 1,043.84-1,044.06; batch KV=kvarn 334,183 cold, 350,192 warm / 843.72-852.42. Environment: kernel 6.6.87.2-microsoft-standard-WSL2, Ubuntu 24.04, driver 591.86, Docker 29.2.0 / Compose 5.0.2.

**llama.cpp — partial.** vLLM's entry is about proving the container matches the venv. Ours is the same question in a different form — what does this platform actually do with the flags we pass — and the map records four Windows-specific answers that each differ from the flag's documented meaning. Every one of them is the shape of fault this repo exists to catch.

**Equivalent here:** no container story in the map, but a set of Windows-specific behaviours that are the same class of environment-parity hazard

**Evidence (llama.cpp):** `src/llama-mmap.cpp:552` · `src/llama-mmap.cpp:558-572` · `src/llama-mmap.cpp:580-584` · `src/llama-model-loader.cpp:1459-1462` · `src/llama-mmap.cpp:86-94` · `src/llama-mmap.cpp:387-391`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** One concrete unused flag falls out. On Windows the whole GGUF is mapped and PrefetchVirtualMemory'd in full at load, and unmap_fragment — the step that would release offloaded pages — is an empty function, so resident host memory stays at roughly the full 6.77 GB for the life of the process whether or not the weights are on the GPU. -lm none drops mmap, which also enables the async pinned-memory upload path (gated off by use_mmap || check_tensors). That is a host-RAM and load-time lever, not a VRAM one. Also: -lm dio is inert on Windows (it is -lm none plus a misleading log line), --numa is a no-op, and read_alignment() is always 1 so the async staging buffer is 1 MiB rather than the 64 MiB NVMe-tuned size.

## already have it — 19

### Two-line patch so qwen3_5 actually calls vLLM's dequant-on-gather embedding kernel
**Where (theirs):** `docs/optimizations.md:14` · `docs/optimizations.md:45-48`

**What it does.** vLLM already ships a kernel that dequantizes an int-quantized embedding table during the gather, but the `qwen3_5` model code never wires it up — neither in the main model nor in the MTP draft module. Without the patch the int8 embedding tables from the previous technique cannot be used at all. `patches/qwen3_5-embed-quant.patch` fixes both call sites, two lines each.

**Mechanism.** Model-definition patch that routes the embedding lookup through vLLM's existing quantized-embedding path rather than the plain bf16 gather, in both `Qwen3_5` and its MTP draft module (`patches/qwen3_5-embed-quant.patch`, docs/optimizations.md:48).

**Why they needed it.** "vLLM ships a dequant-on-gather kernel for int-quantized embedding tables but the qwen3_5 model code never wires it up — neither in the main model nor in the MTP draft module." (docs/optimizations.md:45-47). It is the enabling half of the 2.6 GB saving; the quantized weights are inert without it.

**Their numbers.** Two lines each, two call sites. No standalone throughput figure — it is a prerequisite for the +146 e2e tok/s step above.

**llama.cpp — already have it.** There is no llama.cpp configuration in which a quantized embedding table is inert: the quantiser writes token_embd as a quant type by default (src/llama-quant.cpp:452-471) and the loader places it like any other tensor with no per-arch opt-in (src/llama-model-loader.cpp:1053-1065). The vLLM defect is a model-definition wiring gap that has no counterpart in a graph built from arch tables.

**Equivalent here:** ggml_get_rows on a quantized token_embd — no wiring needed

**Evidence (llama.cpp):** `src/llama-quant.cpp:452-471` · `src/llama-model-loader.cpp:1053-1065` · `src/llama-arch.cpp:672`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. Nothing to enable.

### W4A8-INT8 Marlin: int8 tensor cores for the batched GEMMs
**Where (theirs):** `docs/optimizations.md:18-19` · `docs/optimizations.md:58-64` · `batch/README.md:147` · `README.md:271-272`

**What it does.** At 40-64 concurrent sequences the decode step is bound by fp16 tensor-core math (~63 TFLOPS sustained at 250 W), not by memory. vLLM's `VLLM_MARLIN_INPUT_DTYPE=int8` path keeps weights at int4 and quantizes activations to int8 per token so the MMA runs on int8 tensor cores at 4× the fp16 rate. Applied to the MLP GEMMs by default (74% of the FLOPs).

**Mechanism.** `VLLM_MARLIN_INPUT_DTYPE=int8` on the GPTQ-Marlin path: int4 weights, per-token int8 activation quantization, int8 MMA. Exposed in batch mode as `INT8_ACT=int8` with `INT8_LAYERS` selecting which layers by regex (batch/README.md:147-148).

**Why they needed it.** "At 40-64 concurrent sequences the decode step is bound by fp16 tensor-core math (~63 TFLOPS sustained at 250 W)" (docs/optimizations.md:58-60) — a compute bound, so a 4×-rate MMA is the lever rather than any memory trick.

**Their numbers.** ~63 TFLOPS fp16 sustained at 250 W is the wall being hit. int8 MMA runs at 4× the fp16 rate. MLP-only (default) at 64 concurrent 128/512: 707 → 942 e2e tok/s, ~830 → ~1,094 steady-state decode; described as covering "74% of the FLOPs" (README.md:271). Gentle variant `INT8_LAYERS=gate_up`: 787 e2e / ~930 decode. Prefill gains too: int8 path is +50% at 1k tokens (1,812 vs 1,210 tok/s) tapering to +25% at 100k (997 vs 795) as the 16 attention layers take a bigger share (batch/README.md:63-82).

**llama.cpp — already have it.** llama.cpp's quantized matmul path has always quantized activations to 8-bit and used integer MMA/dp4a; there is no fp16-activation GEMM to escape from. The dispatch chain MMF → MMVQ → MMQ → cuBLAS is fixed and unflagged.

**Equivalent here:** MMQ (quantized weights × q8_1 activations on int8 tensor cores) and MMVQ (dp4a) — chosen automatically

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:1853-1865` · `ggml/src/ggml-cuda/mmq.cu:312-314` · `ggml/src/ggml-cuda/mmvq.cu:289-337`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero to gain, and the premise does not hold: at -np 1 decode is memory-bound, not tensor-core bound. On Ada, ggml_cuda_should_use_mmq short-circuits to true for supported types, so MMQ is already the path for anything above 8 query tokens.

### `INT8_LAYERS=.` needs `GPU_UTIL=0.95` — the all-linear scratch does not fit 0.972
**Where (theirs):** `docs/gotchas.md:108-112` · `docs/quality.md:43-47` · `batch/README.md:19-21`

**What it does.** Quantizing the activations of every linear rather than just the MLP is worth ~11% throughput but adds enough per-layer transient scratch that batch mode's 0.972 utilization OOMs at run time — inside the GDN chunk kernel, once ~17 requests are resident. The failure reads as every request returning HTTP 500 while `/health` still answers OK.

**Mechanism.** Extra per-layer activation-quantization scratch buffers push the transient peak past what the pool left; the throw is `torch.OutOfMemoryError` inside `chunk_fwd_o` (docs/gotchas.md:111). Fix is to hand back headroom: `GPU_UTIL=0.95`.

**Why they needed it.** "the extra per-layer scratch no longer fits batch mode's 0.972: the engine dies with `torch.OutOfMemoryError` inside `chunk_fwd_o` once ~17 requests are resident, which reads as every request returning 500 while `/health` still answers." (docs/gotchas.md:110-112) — a liveness probe that stays green through total failure.

**Their numbers.** ~11% throughput: 1,042 vs 942 tok/s at 64 concurrent. Dies once ~17 requests are resident. Quality cost of the row: +3.7% perplexity.

**llama.cpp — already have it.** llama.cpp budgets the analogous scratch — including the FA MMA dequant buffer — at reserve time rather than discovering it at run time, because ggml_backend_cuda_buffer_type_get_alloc_size routes FLASH_ATTN_EXT through the FA sizing function and ggml-alloc sizes blocks from that. So the failure vLLM hit is structurally prevented, and the remaining lever is the margin itself.

**Equivalent here:** -fitt / --fit-target (default 1024 MiB per device) plus the pp/tg/pp reserve sequence that budgets transient scratch up front

**Evidence (llama.cpp):** `common/arg.cpp:2851-2874` · `common/common.h:473` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912` · `src/llama-context.cpp:662-671` · `common/speculative.cpp:2418-2423`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** -fitt is a live, unswept knob: on a 12 GB card the default forfeits 1 GiB, and lowering it (e.g. -fitt 384) converts most of that into KV directly. The counter-risk is exactly vLLM's: transient scratch that the reserve did not see. Note DFlash/DSpark raise n_batch and n_ubatch to n_parallel*(n_max+1) behind your back, which enlarges the compute buffer after --fit has already measured.

### Sort-free small-k top-k/top-p sampler with multi-block softmax
**Where (theirs):** `docs/optimizations.md:89-95` · `docs/optimizations.md:133-134`

**What it does.** vLLM's top-k/top-p masking sorts the entire 248k-row vocabulary for every row, and its softmax runs one thread block per row — 140 µs for a single 248k-wide row, called several times per decode step. When top-k ≤ 64 is known on the host, the mask becomes a single `torch.topk`, the softmax is made multi-block, and drafts are sampled from the same truncated support as the target.

**Mechanism.** `patches/sampler-small-topk-fast-softmax.patch`. Host-known small k lets the mask be one `torch.topk` instead of a full sort; the softmax is re-parallelised across blocks rather than one block per row. Sampling drafts from the same truncated support as the target is a correctness-and-acceptance point, not just speed. The DFlash2 backport routes the V2 sampler through the same path (docs/optimizations.md:133-134).

**Why they needed it.** "vLLM's top-k/top-p masking sorts the whole 248k vocab for every row and its softmax runs one thread block per row (140 µs for a single 248k-wide row, called several times per step)" (docs/optimizations.md:90-92).

**Their numbers.** 140 µs for one 248k-wide row's softmax under the stock path, several such calls per step. Together with the split-KV attention: +4% at default sampling (docs/optimizations.md:95).

**llama.cpp — already have it.** llama.cpp's host sampler is already the shape vLLM had to patch in: with the default chain, top-k runs first and leaves 40 candidates, so neither top-p's >1024 branch nor its 256-element heuristic ever fires, and min-p runs the already-sorted branch. The only full-vocab sort in the area is the device implementation, which is off by default.

**Equivalent here:** host top_k uses std::partial_sort for k<=128; top_p uses a 256-element scratch heuristic; min_p filters in the log domain with no sort

**Evidence (llama.cpp):** `src/llama-sampler.cpp:198-205` · `src/llama-sampler.cpp:1563-1592` · `src/llama-sampler.cpp:1759-1780` · `src/llama-sampler.cpp:1645-1698` · `common/sampling.cpp:421-425`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero to gain, and one thing to avoid: -bs / --backend-sampling would move top-p onto a FULL-vocab ggml_argsort every token, which is strictly worse than the host path for a default k=40 chain. -bs is also mutually exclusive with a grammar, which a coding agent may want.

### Probabilistic draft sampling (`draft_sample_method: probabilistic`)
**Where (theirs):** `single-user/README.md:249` · `single-user/README.md:260` · `single-user/README.md:268-271` · `single-user/README.md:365` · `README.md:282`

**What it does.** Samples the draft token from the MTP head's distribution instead of taking its argmax. This is what rejection sampling actually wants at temperature > 0; at greedy it changes nothing. The single largest step in the MTP ladder before the draft-vocabulary fix.

**Mechanism.** Engine/spec-config setting `draft_sample_method: probabilistic` (exposed as `DRAFT_SAMPLE`, default `probabilistic`). At T=0 identical to greedy drafting; at T>0 greedy drafting is ~15% slower (single-user/README.md:365).

**Why they needed it.** "Probabilistic drafting samples the draft from the MTP distribution instead of taking its argmax, which is what rejection sampling wants at temperature > 0; at greedy it changes nothing." (single-user/README.md:268-270)

**Their numbers.** 78 / 99 → 90 / 98 tok/s (default / greedy), 2.2 → 2.6 tokens per step at default sampling, position-0 acceptance 58% → 69% (single-user/README.md:248-249). Greedy drafting on the fast variant: 97 / 124 tok/s, 2.3 / 3.0 tokens per step — i.e. it costs ~15% at T>0 and nothing at T=0 (single-user/README.md:260).

**llama.cpp — already have it.** common_sampler_init always appends dist(seed) at the end of the chain, which is also what fills the .p values that --spec-draft-p-min later tests. So llama.cpp is permanently in vLLM's 'probabilistic' mode for every drafter. Note the second half of vLLM's point does NOT carry over: llama.cpp uses greedy prefix-match acceptance even at temperature 1.0 unless the drafter supplied per-position distributions, and DFlash2's selector is the only thing in the tree that does.

**Equivalent here:** the draft sampler is hardcoded {TOP_K=10} with llama_sampler_init_dist appended — it samples, it does not argmax

**Evidence (llama.cpp):** `common/speculative.cpp:226-236` · `common/sampling.cpp:400-406` · `common/speculative.cpp:209-224` · `common/sampling.cpp:692-720` · `tools/server/server-context.cpp:3825-3831`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. It is already probabilistic and cannot be made otherwise; the block that would have made the draft sampler configurable is commented out at common/speculative.cpp:209-224.

### DFlash2 block drafter backported to 0.27.1 (vLLM PR #52816)
**Where (theirs):** `docs/optimizations.md:115-135` · `single-user/README.md:56-82` · `single-user/README.md:73-82`

**What it does.** Replaces Qwen's single-layer MTP chain drafter with DFlash2 (Inco, Aug 2026): 5 Qwen3-style layers that predict a whole 7-token block in one non-autoregressive pass from the target's layer 5/19/33/47/61 hidden states, plus a selector that walks a coherent path through 16 candidates per slot. vLLM's support lives on main's V2 model runner as PR #52816; `patches/dflash2-backport.patch` carries it to 0.27.1.

**Mechanism.** The backport carries the PR plus the pieces of main it silently relies on: sentinel `-1` sample rows, sliding-window null-block guards, K draft slots, NaN guards. Plus a semantic fix and two extensions listed below as separate entries. Runs on the V2 model runner. Enabled with `SPEC=dflash2`.

**Why they needed it.** "The one lever left after all of the above is acceptance, and Qwen's MTP head is a single-layer chain drafter at its ceiling." (docs/optimizations.md:117-118) DFlash2 is "a different drafter for this exact target" and on the bf16 model reports 4.80 tokens per step vs 4.28 for MTP at the same block size (docs/optimizations.md:122-124).

**Their numbers.** Upstream bf16 claim: 4.80 vs 4.28 tokens per step. Measured here: 26.5 ms per step vs MTP's 24.8; 3.14-3.34 tokens per step vs MTP's 2.8-2.9; **117.8 tok/s default sampling and 125.7 greedy at C1** (MTP 111-115 / 115-124), best runs of the drafter 133.8 / 138.5 (docs/optimizations.md:147-151). Full cohort table (single-user/README.md:59-64): C1 121.8 / 131.2 decode, C2 195.5 / 214.6, C4 278.9 / 285.7, C8 389.9 / 405.5 — "+10% at C1 at default sampling, +15% greedy". Quality unchanged by construction: perplexity 8.094, GSM8K 96.0-96.5%. MTP mode re-measured after the backport to prove it was untouched: 110.7 / 113.4 tok/s, 73,777-token pool.

**llama.cpp — already have it.** The whole DFlash/DSpark implementation is present at common/speculative.cpp:910-1347, with metadata-driven block size, the selector lattice, and the only dists-producing path in the tree. The one thing to watch is that draft-dflash's process() is a real implementation (a full encoder+decoder pass over every target batch on the draft context), unlike every ngram-* impl whose process() is a `// TODO: implement` stub — so draft-dflash pays a per-batch prefill cost that ngram-mod does not.

**Equivalent here:** --spec-type draft-dflash (this build IS PR #27342 DFlash2 on master, commit 1deefcca3)

**Evidence (llama.cpp):** `common/speculative.cpp:910-1347` · `common/speculative.cpp:1076` · `common/speculative.cpp:2016-2019` · `common/speculative.cpp:1238-1258`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Already measured on our own hardware: +34.7 % over ngram-mod on real code, +48.5 % for the draft-dflash,ngram-mod pair. Nothing to port.

### Letting the DFlash2 draft share the target's quantized lm_head
**Where (theirs):** `docs/optimizations.md:132-134`

**What it does.** Upstream vLLM refuses to let a draft model share a *quantized* lm_head with the target. The backport allows it, so the drafter does not carry its own copy of a 248k-row projection.

**Mechanism.** Part of `patches/dflash2-backport.patch`: "The draft also shares the target's *quantized* lm_head (upstream refuses)" (docs/optimizations.md:132-133). The V2 sampler in that path is also routed through the repo's sort-free small-k top-k/top-p kernel.

**Why they needed it.** The lm_head is 1.3 GB at int8 / 248k rows and is read once per draft token; a duplicate is unaffordable on a 24 GB card where the whole point is the KV pool.

**Their numbers.** No isolated figure given; it is part of the package that gets DFlash2 to 26.5 ms/step.

**llama.cpp — already have it.** llama.cpp does not refuse the sharing the way upstream vLLM does; llama-context.cpp:145-161 keeps ctx_other precisely for LLM_ARCH_GEMMA4_ASSISTANT and for EAGLE3/DFLASH sidecars lacking tok_embd/output, and llama-context.cpp:392 then shares the target's memory. So the capability is present and conditional on how the sidecar GGUF was exported, not on a flag.

**Equivalent here:** a DFLASH/EAGLE3 sidecar shipped without tok_embd/output shares the target's memory; cparams.ctx_other = ctx_tgt is honoured for exactly those cases

**Evidence (llama.cpp):** `src/llama-context.cpp:142-161` · `src/llama-context.cpp:385-396` · `common/speculative.cpp:2460-2461`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Free VRAM if our sidecar happens to carry its own head, and nothing if it does not — but the check is a one-line gguf dump and the answer changes the sidecar budget materially on 9.5 GB. Worth doing before any sidecar requantisation work (technique 8).

### Make the V2 runner's CUDA-graph memory explicit (`VLLM_V2_CUDAGRAPH_MEM_MIB`)
**Where (theirs):** `docs/optimizations.md:163-166` · `docs/gotchas.md:97-100` · `docs/gotchas.md:173-180`

**What it does.** The V2 model runner does not count its CUDA graphs when sizing the KV pool — upstream it returns 0, so ~1.2 GiB of graphs lands on top of whatever `--gpu-memory-utilization` was asked for. Ask for 0.93 and you run at 0.98. The patch makes that reservation an explicit knob.

**Mechanism.** `VLLM_V2_CUDAGRAPH_MEM_MIB` in `patches/hybrid-kv-groups-v2-cudagraph.patch`. Note the interaction with `--kv-cache-memory`: once the pool is pinned in bytes, this variable no longer *sizes* the pool, it only reserves headroom — so under-reserving produces a server that starts, logs a healthy pool, and dies on the first prefill with 50 MiB left (docs/gotchas.md:174-177).

**Why they needed it.** "upstream it returns 0, so ~1.2 GiB of graphs lands on top of `--gpu-memory-utilization` — ask for 0.93, run at 0.98." (docs/optimizations.md:164-165)

**Their numbers.** ~1.2 GiB unaccounted at defaults. Graph memory grows with the verify block: measured 1.82 GiB at `DFLASH_TOKENS=15`, 2.12 at 18, 2.27 at 20. The capture-list length barely matters — 2.21 GiB at 20 with `CG` cut from 63 to 42. Budgeting rule given: a request costs `64 KiB * context + 102 MiB * (DFLASH_TOKENS + 2)`, the second term being the aligned recurrent-state pages (docs/gotchas.md:177-180).

**llama.cpp — already have it.** CUDA graph capture in ggml replays over buffers that already exist; it does not allocate a second pool that the memory sizing could miss. The compute buffer is sized by the reserve passes and reported by common_memory_breakdown_print with an explicit `unaccounted` column, which is the diagnostic vLLM had to add.

**Equivalent here:** the pp→tg→pp reserve sequence sizes the compute buffer; -fitt is the explicit margin knob

**Evidence (llama.cpp):** `src/llama-context.cpp:576-671` · `common/fit.cpp:816-951` · `common/arg.cpp:2851-2874` · `ggml/src/ggml-cuda/ggml-cuda.cu:4234-4289`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero as a patch. -fitt is the equivalent explicit reservation and it is already a flag (default 1024 MiB, i.e. a whole GiB forfeited on a 12 GB card).

### Lookup drafting: propose the continuation of the most recent earlier occurrence in the request's own history
**Where (theirs):** `docs/optimizations.md:28-32` · `docs/optimizations.md:170-182` · `single-user/README.md:133-150` · `single-user/README.md:361`

**What it does.** The DFlash2 drafter attends to a 2,048-token window, but a long-context assistant spends much of its output *reproducing* things — quoting a document, listing commands, rewriting a paragraph while keeping the code — and those tokens sit verbatim tens of thousands of tokens back in the prompt. `patches/dflash2-lookup-drafting.patch` scans the request's own token buffer (the one vLLM already keeps) for the most recent occurrence of the longest suffix of what has been generated so far, and proposes the tokens that followed it.

**Mechanism.** One Triton program per request, batch-size independent, over the token history buffer vLLM already maintains. It searches suffixes between `VLLM_DFLASH2_LOOKUP_NMIN` (6) and `_NMAX` (12) tokens, prefers the longest match and breaks ties by recency, and applies an `NMIN`-token reject test before any candidate is extended. Losslessness: greedy verification never reads the draft distribution, and every position the lookup filled gets a **point mass** on the proposed token — a legal proposal for vLLM's rejection sampler, so acceptance becomes p(x) and the residual is computed from the same buffer.

**Why they needed it.** "those tokens are sitting verbatim in the prompt, tens of thousands of tokens beyond what the drafter can see" (docs/long-context.md-style framing at docs/optimizations.md:172-175). And on the `_NMAX` cap specifically: "a higher cap makes it choose an older long match over a newer short one, which is the worse predictor" (single-user/README.md:361) — the constant encodes a recency-beats-length prior.

**Their numbers.** Drafter window 2,048 tokens. Knob defaults: `_NMIN` 6, `_NMAX` 12, `_NSTRONG` 6, `_AGREE` 0, `_NMIN_TAIL` 4, `_LONGMIN` 6, `_STICKY` 3, `_ADAPTIVE` 1, `_CHEAP_CTX` 0 (single-user/README.md:361) — note docs/optimizations.md:211-213 quotes `_NSTRONG` 8 and `_AGREE` 2 instead. On the ladder at C1: 130 tok/s, up to 259 where the model reproduces its context, 3.3-7.8 tokens per step (README.md:289). The lookup does not decay with context: 14.19 of a possible 16 tokens per step at 112k, against 15.0 at 25k and 50k (docs/long-context.md:166-167).

**llama.cpp — already have it.** This is the same technique, already shipped, but tuned for a different regime. Two llama.cpp-specific limits are worth knowing before sweeping: n_match must be 1..1024 and n_min/n_max 0..1024 or the flag throws and startup fails; and a warning fires below n_match 16. Also the table is a keyless 4M-entry open-addressed-by-overwrite hash with no collision check, so a collision returns a plausible wrong token that merely fails to be accepted — lowering n_min raises the rate at which that costs a verify step.

**Equivalent here:** --spec-type ngram-mod (also ngram-simple / ngram-map-k / ngram-map-k4v / ngram-cache)

**Evidence (llama.cpp):** `common/speculative.cpp:1992-2004` · `common/common.h:351-356` · `common/arg.cpp:4163-4192` · `common/ngram-mod.cpp:27-41` · `common/speculative.cpp:1924-1927`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Already running, and already measured in combination (+48.5 % for draft-dflash,ngram-mod against ngram-mod alone). The unexploited part is the parameters: --spec-ngram-mod-n-match 24, --spec-ngram-mod-n-min 48, --spec-ngram-mod-n-max 64 are all still at struct defaults. I read draft_one: it walks the hash forward token by token and if it hits EMPTY at position i < n_min it CLEARS the whole draft. So with n_min=48 a match that yields 47 tokens produces nothing at all. vLLM's equivalent knob is _NMIN=6. Lowering n_min is a one-flag experiment with a plausible large effect on a coding agent, where 10-30 token verbatim continuations are common and 48-token ones are not.

### Fix the split-KV scratch size at startup (`VLLM_SPEC_DECODE_ATTN_QMAX`)
**Where (theirs):** `docs/gotchas.md:113-120`

**What it does.** The split-KV verify attention sized its partial buffers from the longest query block it had ever been asked for. Once the block could grow past the drafter's — and once a small prefill chunk could land on the same kernel — that "longest so far" changed mid-run, the buffers were reallocated, and the *captured decode graph went on reading the freed ones*. `VLLM_SPEC_DECODE_ATTN_QMAX` (set by `single-user/start_qwen.sh` from `DFLASH_TOKENS`) fixes the size at startup instead.

**Mechanism.** Replace lazy grow-on-demand allocation of the `part_o` partial buffers with a startup-fixed allocation, because CUDA graph capture froze the old pointers. The stated general rule: "A Triton kernel's scratch buffers may not grow after CUDA graph capture." (docs/gotchas.md:113)

**Why they needed it.** "the captured decode graph went on reading the freed ones: `CUDA error: an illegal memory access was encountered`, a few hundred tokens into the first request." (docs/gotchas.md:117-119)

**Their numbers.** Failure appeared a few hundred tokens into the first request. `part_o` needs ~1.5 GiB at the shipped configuration (docs/gotchas.md:246).

**llama.cpp — already have it.** The comment on the third reserve pass says it exists 'to avoid ggml-alloc reallocations during inference', and DFlash/DSpark additionally force n_batch and n_ubatch up to n_parallel*(n_max+1) so the reserve sees the widest verify shape. The pointer-after-free failure vLLM hit therefore has no path here.

**Equivalent here:** ggml-alloc sizes FA scratch from get_alloc_size during the reserve, and the reserve deliberately runs pp, then tg, then pp again

**Evidence (llama.cpp):** `src/llama-context.cpp:662-671` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912` · `common/speculative.cpp:2413-2427`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. The stated rule ('a kernel's scratch buffers may not grow after graph capture') is already structurally enforced.

### Rejected: skipping the drafter's forward pass during a copy
**Where (theirs):** `docs/gotchas.md:181-191`

**What it does.** An idea that looks like free money and was measured to lose: on a step the lookup controller selected, a qualifying match is long enough to take the head of the block too, so all seven of the drafter's tokens are replaced before anything is verified — so skip its forward and save ~3 ms of a 39 ms step. Measured, 15.21 tokens per step becomes 13.79 for a 5% cheaper step: a net 6% down.

**Mechanism.** The insight that explains the loss: "The drafter is covering the positions *past the end of the match*, which is exactly where a copy lands when the text it is reproducing diverges." (docs/gotchas.md:186-188). A restricted variant (skip only when the match reaches the end of the block) recovers the acceptance but only two runs in three, because the flag it keys on is one step stale and a stricter condition is more sensitive to that staleness. Both variants were removed.

**Why they needed it.** Recorded explicitly as a negative result: "Both variants are gone; this entry is here so the idea does not look untried." (docs/gotchas.md:190-191)

**Their numbers.** ~3 ms saved out of a 39 ms step (5% cheaper). 15.21 → 13.79 tokens per step. Net −6%. Restricted variant recovers acceptance in only 2 of 3 runs.

**llama.cpp — already have it.** Note two mitigations llama.cpp already has that vLLM's skip did not: every non-winning impl still receives accept(seq_id, n_accepted, is_other=true) so its state stays current even when it did not draft, and draft-dflash's process() runs on every target batch regardless, so its encoder features are not stale. So the skip here is cheaper than vLLM's was. It still cannot cover the divergence position.

**Equivalent here:** the fallback loop breaks on the first non-empty draft, so the losing impls' draft() is never called

**Evidence (llama.cpp):** `common/speculative.cpp:2725-2726` · `common/speculative.cpp:2753-2755` · `common/speculative.cpp:2796-2801` · `common/speculative.cpp:1076`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** llama.cpp already does the thing vLLM measured as a 6 % net loss — when ngram-mod clears its n_min gate, draft-dflash's draft() does not run at all. vLLM's diagnosis says why this can hurt: the model drafter covers the positions past the end of the match, which is exactly where a copy diverges. Our own measurement says the pair still beats dflash alone (+48.5 % vs +34.7 %), so empirically the loss is not dominating here — but that is at one context depth on one corpus, and it is the strongest argument I have seen for trying the fusion patch under technique 24/26.

### Two step-time stairs at 16 and 21 query tokens — why the only sensible blocks are 16 and 21
**Where (theirs):** `docs/gotchas.md:159-172`

**What it does.** A verify block costs step time in stairs, not smoothly, and the two stairs have separately identified causes. The first is the target's W4A16 GEMMs: GPTQ-Marlin tiles the M dimension in 16 rows (`m_block_size = 16 * thread_m_blocks`, `thread_m_blocks = div_ceil(prob_m, 16)`), so a 17th query token buys a second M block in all 64 layers and tokens up to 32 are then free. The second is the verify attention: `SpecDecodeAttention._plan` puts `q_len * G` rows in a 128-row tile, so with `G = 24/4 = 6` one tile holds `128 // 6 = 21` query tokens and a 22nd re-reads the request's whole KV segment.

**Mechanism.** Two independent quantisations composed: Marlin's M-dimension tiling (16) and the attention kernel's query-row tile (128/G = 21). Conclusion: "there are exactly two sensible block lengths — 16 query tokens, the last one on the bottom stair, and 21, the most tokens obtainable for the price of the second."

**Why they needed it.** It is the reason `DFLASH_TOKENS=15` (16 query tokens) is the shipped reproduction setting rather than any larger number, and the reason 31 was never measured: "31 pays both stairs and was never worth measuring; two attempts to start it died on memory first." (docs/gotchas.md:171-172)

**Their numbers.** Measured on a copy at 25k context: 39.5 ms/step at 16 query tokens (`DFLASH_TOKENS=15`), 47.8 at 19, 47.2 at 21 — a jump between 16 and 19, then flat. Attention cost per layer: 250 / 583 / 1132 µs at 8 / 16 / 32 query tokens. `G = 24/4 = 6`, tile holds `128 // 6 = 21`.

**llama.cpp — already have it.** vLLM's insight generalises perfectly: verify-block cost is a staircase set by independent kernel tilings, and you should sit on the last step of a stair. llama.cpp's stairs are just at different treads, and unlike vLLM's they are visible in the source rather than needing to be inferred from timings. On Ada the MMQ-vs-cuBLAS heuristic is dead (turing_mma_available short-circuits), so MMQ_DP4A_MAX_BATCH_SIZE=64 is not a third stair here.

**Equivalent here:** the same class of quantisation exists and is documented in the source: FA VEC→MMA at Q->ne[1] > 2 with quantized KV, and MMVQ→MMQ at ne11 > 8

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:469` · `ggml/src/ggml-cuda/mmvq.cuh:3` · `ggml/src/ggml-cuda/mmvq.cu:337` · `ggml/src/ggml-cuda/mmf.cu:176` · `ggml/src/ggml-cuda/mmq.cu:312-314`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** This is the map for the --spec-draft-n-max sweep, and it says the stairs are in different places than vLLM's. With -ctk q4_0 -ctv q4_0 on Ada: n_max = 1 gives 2 query tokens and keeps attention on the VEC kernel (no full-cache dequant); n_max >= 2 gives >= 3 and forces MMA_F16 with the per-step dequant of the entire cache. Separately, MMVQ_MAX_BATCH_SIZE is 8, so 1+n_max <= 8 (n_max <= 7) keeps the weight matmuls on the vector kernel and n_max >= 8 moves them to MMQ. MMF bails above 16 columns. So the candidate blocks worth measuring are n_max ∈ {1, 2, 7, 8, 15}, not a linear sweep — and the first stair is at 2, far earlier than anything vLLM saw.

### Aligned recurrent-state pages scale with the verify block, not with the slot count
**Where (theirs):** `docs/gotchas.md:223-229` · `docs/gotchas.md:130-134`

**What it does.** Corrects an earlier belief (gotcha 23) that `--mamba-cache-mode align` reserves state pages *per request slot*. Measured by asking for an impossible `max_model_len` and fitting the two numbers vLLM prints: the fixed term is 0.88 GiB at `DFLASH_TOKENS=7` and 1.66 GiB at 15 — a ratio of 0.53, which is exactly 9/17, i.e. `(k+2)` — while moving `MAX_SEQS` from 1 to 8 changes it by about 8 MiB in total.

**Mechanism.** Two-point fit of `needed(context)` from vLLM's own "X GiB KV cache is needed ... estimated maximum model length is Z" message. Consequence: dropping to one slot for a genuinely single-user server buys no context at all, and `MAX_SEQS=4` at a long block is about CUDA graph memory, not state pages.

**Why they needed it.** "Gotcha 23 says 'per request slot'; that is wrong." (docs/gotchas.md:224) — a self-retraction with the measurement that overturned it. Also corroborated in README.md:164-168: a recurrent-state slot costs ~8 MiB, "so the slots are close to free; what scales with the pool is the verify block length, not the slot count."

**Their numbers.** Fixed term 0.88 GiB at k=7, 1.66 GiB at k=15; ratio 0.53 = 9/17 = (k+2). `MAX_SEQS` 1 vs 8 moves it ~8 MiB total. Per-request budgeting rule elsewhere: `64 KiB * context + 102 MiB * (DFLASH_TOKENS + 2)` (docs/gotchas.md:179-180).

**llama.cpp — already have it.** vLLM's correction (it scales with the verify block, not the slot count) holds here in a stronger form because llama.cpp makes both factors explicit in one expression. Two clamps to know: n_rs_seq is silently forced to 0 for an arch not in llm_arch_supports_rs_rollback, logged at DEBUG only, and with n_rs_seq = 0 any partial seq_rm on the recurrent half fails, which pushes common_context_can_seq_rm to FULL and turns context checkpoints on.

**Equivalent here:** n_rows = mem_size * (1 + n_rs_seq), where mem_size = max(1, n_seq_max) and n_rs_seq = draft.n_max for draft-mtp/eagle3/dflash/dspark

**Evidence (llama.cpp):** `src/llama-memory-recurrent.cpp:99-101` · `src/llama-memory-recurrent.cpp:118-127` · `common/common.h:386-392` · `src/llama-context.cpp:104-109` · `src/llama-arch.cpp:1044-1055`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The same law, and it is a budget line we have to add before raising n_max. At -np 1 with ngram-mod only, n_rs_seq = 0 and the state is 1 row. Adding draft-dflash makes it 1+n_max rows: 4 rows at the default n_max=3, 16 rows at n_max=15. Whatever the RS buffer costs today, going to n_max=15 multiplies it by four relative to the default. The exact MiB is printed at startup as `RS buffer size` with the R and S widths — read it, do not estimate it.

### Ask for an impossible `max_model_len` to read out the memory model in 90 seconds
**Where (theirs):** `docs/gotchas.md:230-238`

**What it does.** A diagnostic technique rather than an optimisation. vLLM prints "X GiB KV cache is needed ... available Y GiB ... estimated maximum model length is Z" and dies in ~90 s — before torch.compile finishes and long before graph capture. Two such points give slope and intercept for `needed(context)`, and the slope can be checked against arithmetic rather than trusted.

**Mechanism.** Two-point linear inversion of vLLM's own startup estimate. The slope comes out at exactly `16 × 4 × 256 × 2 × 2 = 65,536` B/token for bf16 (16 attention layers × 4 KV heads × 256 head dim × 2 for K+V × 2 bytes), so the fit is verifiable against arithmetic. Documented caveat: `estimate_max_model_len` is a binary search over `max_memory_usage_bytes`, which rounds up to whole blocks, so the estimate is quantised by block size — at an 864-token block the granularity is coarse and a two-point inversion at small lengths is unreliable.

**Why they needed it.** "the cheapest way to read the memory model" — 90 s instead of a full startup, and it gives an answer that can be arithmetic-checked instead of trusted.

**Their numbers.** ~90 s per probe. bf16 slope exactly 65,536 B/token = 16 × 4 × 256 × 2 × 2.

**llama.cpp — already have it.** vLLM had to invert an error message because it publishes no breakdown; llama.cpp publishes one, split three ways with an unaccounted column that is exactly the residual vLLM was fitting for. The 90-second-probe trick is therefore obsolete here rather than portable.

**Equivalent here:** common_memory_breakdown_print — the model / context / compute / unaccounted table the server logs at startup; plus -fitp for the estimate-only form

**Evidence (llama.cpp):** `common/fit.cpp:816-951` · `common/arg.cpp:2836-2850` · `common/fit.cpp:953-984`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Better than the vLLM technique and free: llama.cpp prints the decomposition directly instead of requiring a two-point inversion of a failure message. -fitp / --fit-print is the estimate-without-running form but it is registered only for LLAMA_EXAMPLE_FIT_PARAMS, so it is not on llama-server — it needs the separate fit-params binary if we want the pre-flight number.

### Prefix caching for a hybrid model (`PREFIX_CACHE=1`, `--mamba-cache-mode align`)
**Where (theirs):** `docs/optimizations.md:33-35` · `docs/optimizations.md:110-113` · `docs/optimizations.md:243-255` · `batch/README.md:91-107` · `single-user/README.md:116-131` · `README.md:87-89`

**What it does.** vLLM keeps prefix caching opt-in for mamba/GDN hybrids. `PREFIX_CACHE=1` turns it on in both modes: the attention KV of the shared prefix is reused *and* the recurrent (GDN) state resumes from the last cached block boundary, so a follow-up turn or a shared system prompt is not re-prefilled. The state resume is exact, not approximate — answers come back token-identical.

**Mechanism.** `--enable-prefix-caching --mamba-cache-mode align`. The `align` mode is what makes the recurrent half work: state is snapshotted at cached block boundaries so it can be resumed rather than recomputed. Costs one extra recurrent-state page per request.

**Why they needed it.** "If your API backend sends the same instructions with every request, this is the single biggest thing in this repo for you." (docs/optimizations.md:252-253). And on why it composes with the lookup: "Prefill stops dominating a chat, and then drafting from the context is what makes the decode fast." (docs/optimizations.md:247)

**Their numbers.** Single-user, 4-turn chat over a 24k-token document, greedy (single-user/README.md:121-124): default 22.9 / 23.1 / 22.8 / 22.9 s per turn; with cache 23.5 / **1.15** / **0.85** / **0.89** s. README.md:88-89 quotes 0.56 s vs 22.4 s TTFT on a 25k-token document. Pool cost: 86,727 → 72,475 tokens (MTP) / 68,605 (DFlash2), ~16%; acceptance unaffected (2.23/2.03/2.28 tokens per step cached vs 2.27/1.80/1.96 uncached). Batch mode, 64 requests sharing a 5,820-token system prompt at concurrency 32 (batch/README.md:99-106): wall 222.2 s → **16.9 s**, output 10.6 → **133.9 tok/s**, median latency 94.9 → **8.0 s**, p90 155.3 → **15.2 s**. Pool cost ~14% (223,821 → 193,298 tokens, concurrency 1.49× → 1.29× at 150k) and nothing on workloads with no shared prefix (870 tok/s on 128/512, unchanged). A control run that changes the prefix every turn pays the full 23 s again. `CTX=huge` at 100k: turn 2 costs 4.7 s against 169 s cold (README.md:153).

**llama.cpp — already have it.** vLLM had to opt in and needed an align mode to make the recurrent half resumable; llama.cpp saves the full sequence state by default and does not need one. The one thing that does not transfer at all is non-prefix reuse: --cache-reuse is force-disabled on any M-RoPE/I-M-RoPE model because llama_memory_can_shift is false, and the server zeroes it with a warning at startup. Note also that context checkpoints on a non-SWA hybrid store ONLY the recurrent state — the attention KV must be recomputed — so checkpoint restore is not a substitute for the prompt cache.

**Equivalent here:** --cache-prompt (default enabled) for in-slot prefix reuse; -cram (default 8192 MiB) for the cross-request host-RAM prompt cache, saved with LLAMA_STATE_SEQ_FLAGS_NONE, i.e. full state including attention KV

**Evidence (llama.cpp):** `common/common.h:611` · `common/common.h:615` · `tools/server/server-context.cpp:256-280` · `tools/server/server-context.cpp:2355-2363` · `tools/server/server-context.cpp:3053` · `src/llama-kv-cache.cpp:1176-1178` · `src/llama-memory-hybrid.cpp:190-196`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** On by default and it is the single largest thing already working for a coding agent — a repeated system prompt plus a growing conversation is exactly the shape it serves. Two costs to know: the RAM cache defaults to 8 GiB of HOST memory, and --cache-idle-slots is on, so every new task does a full llama_state_seq_get_data of the idle slot into RAM before the new prompt starts — and that time lands inside the NEXT request's prompt_ms window, because t_start is only set once the slot enters PROCESSING_PROMPT.

### The KVarN decode tax is much larger for a speculating server, and grows with context
**Where (theirs):** `docs/long-context.md:37-64` · `single-user/README.md:359`

**What it does.** Isolates a result that is easy to get wrong: KVarN's 1.22× batch-mode tax at 100k becomes **2.13×** in single-user (speculating) mode at 112k. About 1.98× of that is raw step time; the rest is acceptance — the quantized cache shifts the target's logits enough that the draft head agrees less often, costing ~7% of acceptance.

**Mechanism.** Controlled comparison where the only variable is `--kv-cache-dtype`: MTP-3, one request, 112,648-token prompts, `PREFIX_CACHE=1`, two general tasks (summarize, answer-a-question), streamed so prefill is excluded (`bench/labd_bench.py <tag> --ctx 100000 --corpus ~/bench/labd_corpus_long.txt --tasks qa,summary`).

**Why they needed it.** "Speculation stays exact — the sampled distribution is unchanged, and perplexity moves +0.16% — but acceptance feeds straight back into throughput, so 'quality-neutral' does not imply 'speed-neutral' for a speculating server." (docs/long-context.md:58-60). And: "The tax is a function of context length, not a constant." (docs/long-context.md:64)

**Their numbers.** Single-user MTP-3 at 112k, fp8 (`CTX=long`) vs KVarN (`CTX=huge`): pool 174,489 → 292,035 tokens; decode summarize 71.4 → 33.1 tok/s; decode answer-a-question 64.9 → 30.8; both 68.1 tok/s / 14.7 ms per token → 32.0 tok/s / 31.3 ms (2.13×); accepted tokens per step 2.56 → 2.38 (−7%); TTFT cold 152.2 → 146.7 s. Decomposition: ~1.98× is raw step time. For short-prompt chat the gap nearly closes: 84 / 89 tok/s fp8 vs 79 / 88 KVarN, about 1.06×.

**llama.cpp — already have it.** vLLM's careful separation of the two components — raw step time versus acceptance loss from a shifted logit distribution — transfers, and llama.cpp has both. The step-time component is the MMA dequant. The acceptance component exists too, since the quantised cache moves the target's logits and greedy prefix-match acceptance is sensitive to that. llama.cpp reports acceptance separately (`draft acceptance = accepted/generated, mean len`) so the two can actually be decomposed rather than inferred.

**Equivalent here:** the same phenomenon: quantized KV + speculation forces MMA_F16, which dequantizes the entire cache to F16 per layer per step, so the tax is O(n_kv) and only paid while speculating

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963` · `ggml/src/ggml-cuda/fattn-common.cuh:1022-1084` · `ggml/src/ggml-cuda/fattn.cu:464` · `ggml/src/ggml-cuda/fattn.cu:469` · `tools/server/server-context.cpp:634-637`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The most directly useful finding in this slice after technique 14. It predicts that our -ctk q4_0 -ctv q4_0 choice is nearly free at 1-token decode but expensive per step once any speculator drafts, and that the expense grows linearly with depth. That is testable with pairs we already run, and it is a candidate explanation for this project's own recorded draft-mtp result (+81 % at 16K, −71 % at 131,072 on one artifact). The experiment: at one long depth, measure ngram-mod and draft-dflash under -ctk q4_0 -ctv q4_0 versus -ctk f16 -ctv f16, paired within a round and order-alternated.

### `--language-model-only` drops the vision tower
**Where (theirs):** `docs/gotchas.md:43-45`

**What it does.** Drops the vision tower cleanly with no weights loaded. If images are not needed, that is 2.7 GB of VRAM.

**Mechanism.** vLLM launch flag; no weights loaded rather than loaded-and-unused.

**Why they needed it.** 2.7 GB on a 24 GB card is roughly the whole embedding saving again.

**Their numbers.** 2.7 GB.

**llama.cpp — already have it.** llama.cpp never loads a vision tower unless asked. The relevant carry-over is the inverse: do not add --mmproj to this profile, because several server features are silently disabled by its mere presence.

**Equivalent here:** multimodal is opt-in via --mmproj; a text-only GGUF carries no vision tower

**Evidence (llama.cpp):** `tools/server/server-context.cpp:1165-1174` · `tools/server/server-context.cpp:3157`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero — already the default state, and worth keeping: loading an mmproj would force-disable both context shift and --cache-reuse with warnings, and make pre_decode GGML_ABORT on a reuse path.

### Benchmark discipline: real prompts, second run, and a quality battery before believing any tok/s
**Where (theirs):** `docs/gotchas.md:7-11` · `docs/gotchas.md:33-38` · `docs/gotchas.md:42` · `batch/README.md:173-176` · `single-user/README.md:229-234`

**What it does.** Three standing rules. (1) Never trust a throughput number without `bench/quality_battery.py` (perplexity + GSM8K against the live server). (2) Random-token benchmarks are meaningless for speculative decoding — the same server does 35, 83 or 151 tok/s on `--dataset-name random` depending on what the noise turns into, because acceptance depends entirely on whether the drafter can guess it; use `--dataset-name custom` with real prompts. (3) Benchmark twice — the first run after any restart includes JIT warmup and reads 30-50% low.

**Mechanism.** `bench/prompts_real.jsonl` holds 8 mixed English/Danish/code chat tasks with 1,024-token answers; `bench/run_benchmarks.sh batch|single` reproduces the published tables; `bench/real_rep.sh <tag> 3 0` repeats the single-stream row; `bench/quality_battery.py <tag>` runs the perplexity/GSM8K rows.

**Why they needed it.** "A benchmark cannot tell you the output is garbage." (docs/gotchas.md:7) — "a throughput number from a server that emits garbage is worth nothing, and the int8 path taught us that the hard way." (batch/README.md:175-176). On the random-prompt rule the repo is careful about scope: "(This is our own measurement, not a description of anyone else's harness — ninfer-3090's published cohorts use short real prompts, not random tokens.)" (docs/gotchas.md:36-38)

**Their numbers.** Random-token spread on one server: 35 / 83 / 151 tok/s (256 in, 1,024 out). First run after restart reads 30-50% low. Run-to-run greedy spread on the same server 5-8% (README.md:231-233), ±3-5% on tokens/step for a single run (README.md:293-296).

**llama.cpp — already have it.** The random-token point is doubly true for us: ngram-mod's acceptance is entirely a function of whether the text repeats, and llama.cpp's all-or-nothing n_min gate makes that binary. A random-token benchmark of this profile measures the hash table, not the model. --metrics is also disabled by default, so the Prometheus counters — including the per-position acceptance histogram — are available and unused.

**Equivalent here:** already this repo's operating standard; llama.cpp supplies the numbers and three specific traps in them

**Evidence (llama.cpp):** `tools/server/server-context.cpp:3313-3323` · `tools/server/server-common.h:400-402` · `tools/server/server-context.cpp:4477` · `tools/server/server-context.cpp:2444-2446` · `tools/server/server-context.cpp:3053` · `common/common.h:655`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The rules are already ours. What is new is three named instrument faults in the numbers we scrape, each of the exact shape this project catalogues. (1) [TAG_PROMPT_LOGITS]: a fully cached prompt has n_past decremented by one and n_prompt_cached assigned AFTER the decrement, so a 1000-token cached prompt reports cache_n=999, prompt_n=1, and prompt_per_second becomes one token over the whole slot wall time — a meaningless number that looks plausible. (2) predicted_per_second divides by n_gen−1, so a 1-token generation reports 0 t/s. (3) /metrics rate gauges are computed from buckets reset on every scrape, so a second scrape moments later returns 0; /slots does not reset them. Also: prompt_ms starts when the slot enters PROCESSING_PROMPT, so the idle-slot RAM cache save lands inside the next request's prompt time.

### Greedy is not deterministic across drafter configs — compare tokens/step, not tok/s
**Where (theirs):** `docs/gotchas.md:71-75` · `single-user/README.md:66-71` · `single-user/README.md:31-37`

**What it does.** The target rounds differently when it verifies 5 tokens vs 1, so a different drafter changes the generated text at near-ties, and the 8-prompt acceptance numbers move ±3%. Greedy repeats *within* a server session are bit-identical, but the text flips at near-ties between sessions and acceptance moves with it. The stable signal to compare is tokens per step, which both `bench/run_benchmarks.sh single` and `bench/real_rep.sh` print.

**Mechanism.** Numerical: the verify block is one chunk through the recurrent layers, so changing what is in it changes the last bits of the logits (single-user/README.md:225-227). `drafter/README.md` has an offline chain simulator that removes the noise.

**Why they needed it.** It is the reason the repo's own before/after comparisons are run several times in the same session, and the reason two of nine long greedy prompts differing under `LOOKUP=1` is reported as a near-tie flip rather than a distribution change.

**Their numbers.** ±3% on 8-prompt acceptance; ±3-5% run-to-run on tokens/step; 5-8% run-to-run spread on decode tok/s. Four bit-identical greedy repeats in a row read 125.0-126.6 e2e with the same step count to the token (single-user/README.md:67-68). 7 of 9 long greedy prompts token-identical against `LOOKUP=0`.

**llama.cpp — already have it.** The numerical argument transfers: the verify batch is one ubatch through the model, so changing its width changes the last bits of the logits and flips near-ties. llama.cpp makes this worse in one specific way — the graph is rebuilt whenever n_tokens changes — and better in another: it hands you the acceptance metric without extra tooling.

**Equivalent here:** `draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f` printed on every completion when drafting ran

**Evidence (llama.cpp):** `tools/server/server-context.cpp:634-637` · `common/speculative.cpp:2829-2872` · `tools/server/server-context.cpp:3883-3903` · `tools/server/server-context.cpp:3899`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The stable comparison metric is already printed for free and we should be quoting it alongside tok/s in every speculative comparison. The per-implementation breakdown is richer still (per-impl n_gen_tokens, n_acc_tokens, n_acc_tokens_per_pos, plus timings) but is LOG_TRC only, so it needs -lv 4. With --metrics on, the per-position histogram is also exported and is sized to common_speculative_n_max — 64 buckets wide under ngram-mod.

## impossible here — 2

### A 40k-row draft head whose vocabulary is counted over the model's own outputs
**Where (theirs):** `docs/optimizations.md:20-22` · `docs/optimizations.md:70-83` · `docs/gotchas.md:56-60` · `single-user/README.md:255` · `single-user/README.md:262-268` · `README.md:285`

**What it does.** A draft head can only propose tokens in its id list, so a token outside the list is a guaranteed rejection that also ends the chain. This repo builds a 40k-token draft head whose id list is counted over 5.4M tokens of the *model's own generated output* rather than over web text, and patches the drafter to use it. Coverage 97.5% of what the model generates (96% on code) against 92% (83% on code) for the web-text list.

**Mechanism.** `drafter/gen_data.py` generates the corpus, the frequency step in `prepare/build_draft_vocab.py` counts it into `prepare/draft_vocab_ids.json`, `prepare/build_draft_vocab.py` slices a 40k-row draft head out of the 248k-row lm_head, and `patches/qwen3_5-mtp-draft-vocab.patch` makes the drafter project onto that head instead. Toggle `MTP_DRAFT_VOCAB=0` reverts to the full head (single-user/README.md:369).

**Why they needed it.** "a token outside the draft vocabulary can never be proposed, so it is a guaranteed rejection that also cuts the chain" (docs/optimizations.md:79-80). "The draft vocabulary is the single-user ceiling." (docs/gotchas.md:56) — and on the failure mode: the earlier web-text list "had been silently capping acceptance at every position" (single-user/README.md:268).

**Their numbers.** 5.4M tokens of the model's own output counted. Coverage 97.5% overall / 96% on code, vs 92% / 83% for web text. Worth 10% of single-stream throughput on its own (docs/optimizations.md:22, :83); "92% vs 97.5% coverage was the difference between 98 and 109 tok/s greedy" (docs/gotchas.md:58). On the ladder: 93 / 99 → 107 / 109 tok/s, 2.6 → 2.9 tokens per step, position-0 acceptance 69% → 74% (single-user/README.md:254-255). Coverage saturates around 40k rows; the model only ever emits ~54k distinct tokens (docs/gotchas.md:59-60). A 49k-row vocab is *worse*: 109 / 115 vs ~114 / ~124 (single-user/README.md:259). Keeping the full 248k head gives higher acceptance (2.85 / 3.0 tokens per step, 74%/76%) but lower throughput, 85 / 91 tok/s (single-user/README.md:252).

**llama.cpp — impossible here.** Draft tokens are returned as a flat llama_tokens and laid straight into the target's batch as target vocabulary ids — there is no id-remap layer anywhere between the draft sampler (built on the DRAFT model's vocab) and the verify batch. The only vocab machinery in the whole area is an equality guard, and it is called from the draft-simple constructor only; eagle3/mtp/dflash/dspark construct without it. A truncated draft head would need a remap table that has no seam to live in.

**Equivalent here:** none

**Evidence (llama.cpp):** `common/speculative.h:67` · `tools/server/server-context.cpp:488-493` · `common/speculative.cpp:68-131` · `common/speculative.cpp:238-245` · `common/speculative.cpp:1001-1008`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero without a large-patch that does not have an obvious home. The nearest real lever is quantizing the sidecar's own output head (see technique 8), which buys VRAM rather than acceptance.

### Capture decode CUDA graphs at both block lengths
**Where (theirs):** `docs/gotchas.md:152-158` · `single-user/README.md:208-212`

**What it does.** The V2 runner captures uniform-decode graphs at `decode_query_len = num_speculative_tokens + 1` and dispatch requires an *exact* match, so scheduling the drafter's 8-token block on a 16-token server matches no graph at all and the step runs piecewise. Since short steps are the common case, that was an 8% tax on almost every step. The lookup patch adds the drafter's block to the capture list.

**Mechanism.** `cudagraph_utils.py` already knows how to capture several decode lengths (it does it for dynamic speculative decoding); the patch registers the drafter's own block length alongside the long one.

**Why they needed it.** "`decode_query_len` only described the long one, so every short step — the common case — fell back to piecewise and paid 8%" (single-user/README.md:210-212).

**Their numbers.** 27.9 ms against 25.9 ms for the same 8-token step on a 7-slot server — 8%. Costs 1.8 GiB of captured graphs instead of 1.45 GiB.

**llama.cpp — impossible here.** vLLM could add the short block to the capture list; llama.cpp cannot, because the key is which split rather than which shape and the upstream ggml graph is rebuilt whenever n_tokens changes. That makes it a hard constraint rather than a tuning item — and it means step-size stability is itself a performance property of a speculator here, which is not obvious and is not documented anywhere in our repo.

**Equivalent here:** none — the CUDA graph map is keyed on cgraph->nodes[0] (which split), not on step size, and llm_graph_params::allow_reuse requires ubatch.n_tokens to be equal

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:2574-2576` · `src/llama-graph.h:785` · `ggml/src/ggml-cuda/ggml-cuda.cu:4253-4268` · `tools/server/server-context.cpp:617-619` · `common/speculative.cpp:1181-1186` · `common/speculative.cpp:1995-1998`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** No fix available, but a first-rate prediction and a free instrument. A captured graph re-arms only after TWO consecutive calls with unchanged node properties, and any change resets warmup_complete to false. DFlash drafts exactly n_max tokens every step, so a dflash-only server has a constant 1+n_max verify shape and can stay captured. ngram-mod is all-or-nothing at n_max=64 or nothing, so a ngram-mod server alternates between 65-token and 1-token steps and can never get past warmup. The draft-dflash,ngram-mod pair alternates across three shapes. `graphs reused` is printed on every completion from llama_perf_context(ctx_tgt).n_reused — compare that counter across our three speculative profiles and we will know immediately how much of the measured difference is speculation and how much is graph capture.

## not applicable — 21

### Marlin int8 negative-group-scale bug fix (sign folded into the int4 codes at load)
**Where (theirs):** `docs/optimizations.md:18-19` · `docs/optimizations.md:64-67` · `docs/gotchas.md:7-11`

**What it does.** The W4A8-INT8 Marlin kernel reads its int16-requantized group scales as *unsigned*. AutoRound symmetric exports have ~50% negative scales, so on this checkpoint the path benchmarked beautifully and served nonsense. `patches/marlin-int8-negative-scales.patch` folds the sign into the int4 weight codes at load time so the kernel's unsigned read is correct.

**Mechanism.** Load-time weight transform: negate the int4 codes for any group whose scale is negative and store |scale|, so the kernel's unsigned int16 scale read reconstructs the same product. Implemented in `patches/marlin-int8-negative-scales.patch` (docs/optimizations.md:67).

**Why they needed it.** "on this checkpoint it produced garbage while benchmarking beautifully. The kernel reads its int16-requantized group scales as *unsigned*, and AutoRound symmetric exports have ~50% negative scales." (docs/optimizations.md:63-66). The corresponding gotcha is the repo's stated first principle: "A benchmark cannot tell you the output is garbage. The int8-activation path served nonsense for an hour of beautiful throughput numbers before a perplexity check caught it." (docs/gotchas.md:7-9)

**Their numbers.** ~50% of groups have negative scales in an AutoRound symmetric export. One hour of plausible throughput numbers served garbage before a perplexity check caught it.

**llama.cpp — not applicable.** llama.cpp has no load-time weight transform at all: the only one is CPU repack, and it is layout-only with get_alloc_size nullptr. There is nowhere for this class of sign bug to live.

**Equivalent here:** none — no Marlin kernel, no group-scale reinterpretation at load

**Evidence (llama.cpp):** `ggml/src/ggml-cpu/repack.cpp:4828-4829` · `src/llama-model-loader.cpp:1177-1203`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Zero as code. The discipline it encodes — never publish a throughput number from a newly-enabled kernel path without a quality check — is already this project's stated north star.

### Per-layer selection of int8 activations (`INT8_LAYERS` regex)
**Where (theirs):** `docs/optimizations.md:67-69` · `batch/README.md:148` · `docs/quality.md:30-47` · `docs/gotchas.md:108-112`

**What it does.** `patches/marlin-int8-layer-select.patch` makes int8-activation quantization selectable per layer by regex on the layer name, so the accuracy/throughput trade can be dialled — and, necessarily, so it can be kept off the int8-weight `lm_head`, which would otherwise refuse to load. Three shipped points: `gate_up`, `mlp` (default), `.` (every linear).

**Mechanism.** Regex match against the layer name decides whether that Marlin GEMM takes int8 activations. `INT8_LAYERS=gate_up` | `mlp` (default) | `.` (everything), or a hand-picked list produced by `bench/act_calib.py` (batch/README.md:148). The selection env vars are registered with vLLM so they enter the torch.compile cache key (see the compile-cache technique).

**Why they needed it.** Two reasons stated: the accuracy ladder is monotone and the user should pick a rung, and "keeps it off the int8-weight lm_head, which would otherwise refuse to load" (docs/optimizations.md:68-69).

**Their numbers.** Perplexity/throughput ladder, batch mode, 64 concurrent 128/512 (docs/quality.md:30-41): fp16 state W4A16 = PPL 8.044, GSM8K 95.5%, 707 tok/s. +gate_up int8 = 8.12 (+0.9%), GSM8K 95.5%, 787. +whole MLP (default) = 8.22 (+2.2%), GSM8K 95.0%, 942. +all linears = 8.34 (+3.7%), 1,042 tok/s. Per-language PPL for the all-linear row: en 10.93 / da 11.29 / code 3.20 — "every int8-activation step costs a little perplexity, mostly on code" (docs/quality.md:38-39). Steady-state decode for the all-linear row ~1,222 tok/s (README.md:272).

**llama.cpp — not applicable.** The only per-tensor regex machinery on the load path (-ot / -cmoe / -ncmoe) selects a buffer type, and the accepted names are only the per-device default bufts — CPU and CUDA0 on this box. There is no seam that would let a regex change which matmul kernel or which activation type a layer uses; ggml_cuda_mul_mat decides from shapes and types alone.

**Equivalent here:** none — -ot is per-tensor placement, not per-tensor precision

**Evidence (llama.cpp):** `common/arg.cpp:253-284` · `common/arg.cpp:2714-2719` · `ggml/src/ggml-cuda/ggml-cuda.cu:1853-1865`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. Activation precision is a property of the kernel llama.cpp picked, not a knob.

### Query-row tiling in the split-KV kernel (the silent 10-token cap)
**Where (theirs):** `docs/gotchas.md:66-70` · `docs/gotchas.md:163-168`

**What it does.** The first version of the split-KV verify kernel handled at most `BLOCK_M / (heads per kv head)` = 10 query tokens and fell back silently past that. Once the verify block grew to 16 (reproduction mode) that fallback doubled the step time at 25k context. It now tiles the query rows instead of capping them.

**Mechanism.** `SpecDecodeAttention._plan` packs `q_len * G` rows into a 128-row tile; with this model's `G = 24/4 = 6` grouped-query ratio one tile holds `128 // 6 = 21` query tokens. Past the tile the kernel now emits another tile of query rows rather than silently dropping to the unsplit path.

**Why they needed it.** "the kernel used to handle at most `BLOCK_M / (heads per kv head)` = 10 query tokens and fall back silently past that, which doubled the step at 25k context the moment the verify block grew to 16." (docs/gotchas.md:67-70)

**Their numbers.** Old cap 10 query tokens. Tile capacity `128 // 6 = 21`. A 22nd query token re-reads the request's whole KV segment: 250 / 583 / 1132 µs per layer at 8 / 16 / 32 query tokens (docs/gotchas.md:168).

**llama.cpp — not applicable.** The vec kernel processes at most 2 query columns (fattn-vec.cuh:553-572), but that is enforced by the selector at fattn.cu:469 which sends anything wider to MMA_F16. The failure mode vLLM describes — a kernel quietly handling fewer query tokens than asked and falling back — has no analogue.

**Equivalent here:** fattn-vec cols_per_block is capped at 2, but the selector routes past it to MMA rather than degrading silently

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-vec.cuh:553-572` · `ggml/src/ggml-cuda/fattn.cu:461-483`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. There is no silent cap to remove; the equivalent constant is a routing condition, not a fallback.

### The MTP ceiling arithmetic — why ~114 tok/s and not 150
**Where (theirs):** `single-user/README.md:296-304` · `README.md:305-310`

**What it does.** An explicit budget argument for why a single-layer chain drafter tops out where it does: the verify pass alone reads ~13 GB of weights (~17 ms at what this card actually delivers on 16-92 MB reads), plus ~4 ms of drafts and sampling; Qwen's MTP head agrees with the target on ~75-77% of first drafts on real text once it can propose the right tokens; so ~3 accepted tokens per ~24 ms step is the ceiling.

**Mechanism.** Not a code change — the reasoning that closed the optimisation search. Two things were measured rather than assumed and did not help: fine-tuning the MTP head on the model's own outputs (KL halves, greedy top-1 on response tokens unchanged), and retuning Marlin's tile configuration for M ≤ 16 on sm86 (3-7% per GEMM in isolation, nothing measurable end to end). The stated reason the second one failed: "the remaining gap to peak bandwidth is the memory system's ramp on 16-92 MB reads, not the kernel" (README.md:309-310).

**Why they needed it.** "Why not 150? ... Random-token benchmarks that show 150+ are measuring how repetitive noise is." (single-user/README.md:296-302) — and on the structural limit: "a tree drafter would [raise it], but the DeltaNet layers can't verify a tree." (single-user/README.md:303-304)

**Their numbers.** ~13 GB of weights read per verify pass, ~17 ms at achieved bandwidth on 16-92 MB reads, plus ~4 ms of drafts and sampling → ~24 ms step. ~75-77% first-draft agreement on real text. ~3 accepted tokens per step. Marlin M≤16 retune: 3-7% per GEMM in isolation, zero end-to-end. GDN decode kernel already runs at ~85% of the 3090's memory bandwidth with every variant within 3% (docs/gotchas.md:49-52).

**llama.cpp — not applicable.** This is a method, not a capability, and llama.cpp already emits both numbers the method needs on the per-request timings line. The structural caveat also carries: llama.cpp cannot verify a tree either — the draft is a flat token list laid into consecutive positions, and p_split, the only parameter that would fork a draft, is read by one example binary and by nothing in common/ or tools/.

**Equivalent here:** the ingredients are all reported: `draft acceptance = accepted/generated, mean len = 1 + accepted/verif_steps` and `graphs reused`

**Evidence (llama.cpp):** `tools/server/server-context.cpp:634-637` · `tools/server/server-context.cpp:617-619` · `common/speculative.h:67` · `examples/speculative/speculative.cpp:67`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No code. The budget argument is worth writing down for our own artifact: verify-pass bytes (6.77 GB at 2-bit plus the sidecar) divided by achieved bandwidth, plus draft cost, gives the step-time floor, and mean accepted length is printed for free at the end of every request. Doing that arithmetic once would tell us when to stop tuning.

### Backport semantic fix: 0.27.1 caches temperature-applied draft logits, main caches raw ones
**Where (theirs):** `docs/optimizations.md:129-133`

**What it does.** vLLM 0.27.1 caches draft logits with temperature already applied; main caches raw logits; PR #52816's candidate selector cached raw scores. Landing the PR unmodified on 0.27.1 would have verified against the wrong q for any 0 < T ≠ 1 — silently wrong sampling, not a crash.

**Mechanism.** A single semantic correction inside `patches/dflash2-backport.patch` reconciling which of the two logit conventions the selector reads.

**Why they needed it.** "0.27.1 caches temperature-*applied* draft logits, main caches raw ones, and the PR's selector cached raw scores — on 0.27.1 that would have verified against the wrong q for 0 < T ≠ 1." (docs/optimizations.md:130-133). This is the class of bug the whole repo is organised around: correct-looking speed with wrong output.

**Their numbers.** Affects the interval 0 < T ≠ 1 — i.e. everything except greedy and T=1.

**llama.cpp — not applicable.** There is no upstream/backport skew here — one tree, one convention. But the shape of the bug (two producers of a probability disagreeing about whether temperature has been applied) is reproducible in any speculative implementation, and llama.cpp's residual branch has exactly two such producers. I cannot resolve it from the map; it is listed in map_gaps.

**Equivalent here:** none as a port; but the same p-vs-q convention question exists in the residual accept path

**Evidence (llama.cpp):** `common/sampling.cpp:722-793` · `common/speculative.cpp:1238-1258` · `tools/server/server-context.cpp:3828-3830` · `common/sampling.cpp:433-434`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No port to do. One audit worth an hour: in the residual/maximal-coupling branch, q comes from dparams.dists which DFlash2 fills from its selector lattice, while p comes from the request's real chain including temperature. If those two are on different conventions the result is silently wrong sampling for 0 < T != 1 — exactly the failure class this repo is organised around, and exactly what vLLM caught. This path is only reachable with draft-dflash at temp > 0, which we now run.

### Hybrid KV-group sizing fix — stop the drafter's 5 sliding-window layers padding the target's 64
**Where (theirs):** `docs/optimizations.md:154-163` · `docs/gotchas.md:97-102` · `single-user/README.md:96-101`

**What it does.** vLLM sizes a hybrid model's KV groups by the *smallest* bucket of same-type layers. With the DFlash2 drafter present that smallest bucket is its 5 sliding-window layers, so the target's 16 attention layers got padded to 20 and its 48 GDN layers to 50 — 25% more pool consumed per token of context, to pad the layers that were not the problem. Since sliding-window groups only ever hold window-many blocks, `patches/hybrid-kv-groups-v2-cudagraph.patch` pads *those* instead, at ~7 MB per request.

**Mechanism.** Reverse which bucket absorbs the padding: sliding-window groups are bounded by the window, so over-provisioning them is O(window) per request rather than O(context) per layer. Takes the pool from 105 to 78 KB per token.

**Why they needed it.** "vLLM sizes a hybrid model's KV groups by the *smallest* bucket of same-type layers ... 25% more pool for every token of context, to pad the layers that were not the problem." (docs/optimizations.md:157-161). Without it "this mode caps out at ~40k" (single-user/README.md:99).

**Their numbers.** 16 attention layers padded to 20, 48 GDN layers padded to 50. Pool cost 105 → 78 KB per token (MTP for reference: 75). Context 45,383 tokens at 40k max-model-len → **69,758 at 64k**. Start-up becomes deterministic — 69,758 tokens twice over (docs/optimizations.md:162-168).

**llama.cpp — not applicable.** llama.cpp allocates KV per layer, with each layer's buffer type taken from the device that layer's weights landed on, and the draft model always gets its OWN memory module sized from its own inherited n_ctx/n_parallel. There is no shared page-size abstraction to equalise and therefore nothing to pad. The SWA cache, where a size-coupling could exist, is sized independently as n_swa*(...)+n_ubatch padded to 256 — and a Qwen3.5-style model declares no SWA at all, so the plain hybrid path is taken.

**Equivalent here:** none — no page-size unification across layer groups exists

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:209-217` · `common/speculative.cpp:2432-2496` · `src/llama-kv-cache-iswa.cpp:70-79` · `src/llama-model.cpp:2305-2344`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. The bug class cannot occur.

### Run the adaptive-block mode with `--no-async-scheduling`
**Where (theirs):** `docs/gotchas.md:121-129` · `docs/optimizations.md:203-205`

**What it does.** vLLM only feeds draft token ids — and therefore the *count* the worker wants verified — back to the scheduler on the synchronous path (`EngineCore.post_step`). With async scheduling on, every decode step is padded to `num_speculative_tokens` and a worker asking for fewer is ignored, silently. Adaptive block length therefore requires `ASYNC_SCHED=0`.

**Mechanism.** `--no-async-scheduling`. Companion trap: "`--async-scheduling` is already the default in 0.27.1. The flag exists and passing it changes nothing; `--no-async-scheduling` is what turns it off. Two hours of 'the adaptive block isn't working' was this." (docs/gotchas.md:127-129)

**Why they needed it.** Without it the whole "schedule the long block only during a copy" mechanism is inert and fails silently — the scheduler pads every step to the maximum.

**Their numbers.** At batch 1 the synchronous path costs under 1%.

**llama.cpp — not applicable.** There is no scheduler layer between the speculator and the batch: common_speculative_draft is called inline, the result is truncated to dp.n_max in place, and the draft is added to the same batch. Nothing pads a step to a maximum draft count behind your back.

**Equivalent here:** none — drafting and verification are synchronous within one server decode step

**Evidence (llama.cpp):** `tools/server/server-context.cpp:2936-2946` · `common/speculative.cpp:2728-2733` · `tools/server/server-context.cpp:488-493`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero.

### The DFlash draft pass is a captured CUDA graph, so its host Python runs once
**Where (theirs):** `docs/gotchas.md:135-147`

**What it does.** `DFlashSpeculator._generate_draft` — everything the speculator does per step, including the lookup — is replayed from a captured graph. The Triton kernels inside it do run every step and do read live buffers, so the lookup itself works; but host-side Python in there executes at *capture* time only. A counter, a pinned copy of a flag, a decision computed there is frozen at whatever the warm-up produced, silently. Anything the host must see per step belongs in a method the model runner calls per step (`next_num_draft_tokens`), reading device tensors the replayed kernels wrote.

**Mechanism.** Architectural rule for this codebase: device work goes inside the captured pass, host decisions go in `next_num_draft_tokens`, and the channel between them is a device tensor the kernels write. A related trap: `torch.cuda.is_current_stream_capturing()` is not a usable guard here — "It reads True inside the captured draft pass — which is correct, and exactly why a guard written as `if not is_current_stream_capturing():` silently disables the code it guards for the entire run, not just during warm-up." (docs/gotchas.md:144-147)

**Why they needed it.** "Three separate 'the trigger doesn't fire' debugging rounds were this." (docs/gotchas.md:142-143)

**Their numbers.** Three debugging rounds lost to it.

**llama.cpp — not applicable.** A ggml graph is data; the speculator's C++ runs on the host every step regardless of whether the backend replayed a captured graph. There is no capture-time-only execution to be surprised by, and no is_current_stream_capturing guard to get inverted.

**Equivalent here:** none — ggml graphs contain no host code; the analogous trap is the two-call CUDA-graph warmup rule

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:4249-4274` · `common/speculative.cpp:2710-2756`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Zero as a bug. The transferable half is the warmup rule itself, which is already covered under technique 29.

### Sliding-window block promote — halving the KV element size can *cost* memory on a hybrid model with a draft model
**Where (theirs):** `docs/gotchas.md:205-222` · `docs/long-context.md:128-136`

**What it does.** `unify_kv_cache_spec_page_size` equalizes page sizes by scaling a layer's block size up by the integer ratio `max_page / own_page`, and pads the *page* instead when that ratio is not an integer. Sliding-window layers are born at the backend's smallest kernel block — 16 — precisely because the code picking it assumes unify will scale it up. When the ratio is not an integer that assumption fails silently and every block of that layer pays a whole primary page. `patches/hybrid-sw-block-promote.patch` rounds such a layer's block *up* instead (16 → 864).

**Mechanism.** The divisibility that makes bf16 work is a coincidence: the target's 4 KV heads × 256 and the DFlash2 drafter's 8 × 128 both come to 4096 B per token per layer. `int8_per_token_head` breaks it by adding one fp32 scale *per head* — 2080 vs 2112 B/token, and 2112 = 2⁶·3·11 shares no factor with the primary page. The drafter's 5 layers then took `cdiv(2047 + 4096, 16) + 1 = 385` blocks of 1.71 MiB at 1.88% utilisation — a constant 5.155 GiB, 75.6% of the per-request budget. The tell in a log is an "estimated maximum model length" that is a small multiple of 16.

**Why they needed it.** The upstream comment states the assumption in `_largest_kernel_block_within` (`model_executor/layers/attention/attention.py`): "the smallest block is fine — `unify` scales it up by an integer ratio". "When the ratio is not an integer that assumption fails silently" (docs/gotchas.md:212-214). Without the patch the int8 long-context mode is strictly worse than bf16.

**Their numbers.** int8 needed **6.82 GiB to serve 32,768 tokens** where bf16 serves 69,758 in 5.2 GiB — 2.4× worse from halving the dtype. Patch rounds the block 16 → 864, turning that into **138,696 tokens**. Constants: 2080 vs 2112 B/token; 385 blocks × 1.71 MiB at 1.88% utilisation = 5.155 GiB = 75.6% of the per-request budget.

**llama.cpp — not applicable.** There is no cross-layer page equalisation step in llama.cpp, and the SWA cache — the layer class where vLLM's padding landed — is not even instantiated for this model family: qwen35.cpp assigns no swa_type, so hparams.swa_type stays NONE and create_memory routes to plain llama_memory_hybrid rather than hybrid_iswa. The server additionally force-disables --swa-full with a warning on a model with no SWA layers.

**Equivalent here:** none — no page-size unification, and qwen35-style models declare no SWA at all

**Evidence (llama.cpp):** `src/llama-model.cpp:2305-2344` · `src/llama-kv-cache.cpp:209-217` · `tools/server/server-context.cpp:1188-1195`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. The bug cannot occur.

### Bug B: prefix-cache hit + captured (FULL) verify corrupts one prompt length in every 128; mitigated by PIECEWISE
**Where (theirs):** `docs/gotchas.md:252-331` · `README.md:143-202`

**What it does.** Under `CTX=huge` with a CAPTURED (FULL) verify step, a request that hits the prefix cache and whose prompt length lands on one particular residue mod 128 collapses. `SPEC=dflash2 DFLASH_TOKENS=7` gives 1.97 tok/step and degenerate repetition (4/3595 characters verbatim, one 40-char block ×79); `SPEC=mtp` stops dead and returns `""` or `"#"` with `finish_reason=stop`. Every other residue is 794/794 verbatim. Deterministic — repeats are bit-identical. Mitigation: `CTX=huge` forces `cudagraph_mode=PIECEWISE` for every speculator.

**Mechanism.** Two conditions had to be seen together, by two people. The **hit** is necessary: a fresh server, one request, no warm-up, never collapses at any length — which is also why `PREFIX_CACHE=0` always looked clean. The **residue** decides whether a hit corrupts, and it is a clean function of the draft count: **R = 117 + k**, equivalently the final 128-token tile has exactly `11 - k` free slots, equivalently `L + free = 12` in every configuration measured. Fitted on three values of k across two speculators — treated as a fit, not a derivation — but it implies the step reserves or touches a fixed 12 slots regardless of verify block length, "the most specific lead this bug has produced."

**Why they needed it.** The `DFLASH_TOKENS=5` row is what makes the method sound: same attention block as 7 (2176), different broken residue (122 vs 124), "which rules out the attention block size. An earlier version of this entry claimed R tracked the verify block on three points where the two co-varied; that was retracted as unevidenced, and then confirmed by running the configuration that separates them." (docs/gotchas.md:303-308). And on scope: "This repo previously scoped that workaround to `dflash2` on the theory that MTP's short verify step captures correctly; it does not, and `SPEC=mtp CTX=huge` shipped with the bug." (docs/gotchas.md:321-323)

**Their numbers.** Residue table (docs/gotchas.md:291-294): dflash2 k=7 → verify block L=8, attention block 2176 (=17×128), broken R=124, free 4; dflash2 k=5 → L=6, block **2176**, broken R=**122**, free 6; dflash2 k=3 → L=4, block 2048 (=16×128), R=120, free 8; mtp k=3 → L=4, 2048, R=120, free 8. Confirmed periodic: 24,956 / 25,084 / 25,212 / 25,340 at k=7; 25,082 / 25,210 / 25,338 at k=5; 25,080 / 25,208 / 25,336 at k=3. PIECEWISE mitigation costs nothing measurable: `SPEC=mtp` over 8k/16k/32k/50k reads 87.8/86.1/70.4/63.5 tok/s captured against 93.5/83.8/70.3/59.6 piecewise. On bare metal the FULL capture also corrupts output outright: special-token ids leaking, 1 of 1,176 characters matching instead of all (README.md:146-151). Trade on short prompts: FULL_AND_PIECEWISE 38 tok/s (1.97/step) on copy@25k vs PIECEWISE **132 tok/s (7.83/step)** — 3.5× — while costing 13-18% on short prompts (de 78→74, en 125→102, code 202→176); past 8k the two are within noise (111.8 vs 109.3 at 8k, 78.2 vs 86.1 at 16k, 68.9 vs 73.3 at 32k, 58.4 vs 56.0 at 50k).

**llama.cpp — not applicable.** vLLM's bug is in their runner's interaction between a cache hit and a captured verify step. llama.cpp has both ingredients (prompt-cache restore and CUDA graph capture) but no reported defect, and I have no evidence of one. What is worth taking is the pairing of the two kill switches with the 256-padding fact — that is a ready-made debugging plan for a class of fault we have not yet seen but would find very hard to diagnose.

**Equivalent here:** no known analogue, but the two kill switches that would attribute one exist: GGML_CUDA_DISABLE_GRAPHS and LLAMA_GRAPH_REUSE_DISABLE

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/common.cuh:1255-1259` · `src/llama-context.cpp:279-285` · `src/llama-kv-cache.cpp:1233-1246` · `tools/server/server-context.cpp:3260-3298`

**Effort:** one-flag · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No bug to fix, but two unused diagnostic env vars that are the cheapest possible A/B if we ever see corruption or an unexplained cliff: GGML_CUDA_DISABLE_GRAPHS (existence-checked, value not parsed) kills capture, LLAMA_GRAPH_REUSE_DISABLE=1 kills llama-level graph reuse and logs `graph reuse disabled`. Flipping each one separately attributes a regression to capture versus reuse. Note the geometry that would host such a bug does exist here: get_n_kv rounds used_max_p1 up to 256 explicitly so the graph stays constant across batches, so a residue-mod-256 dependence is precisely the shape to look for.

### How to measure Bug B: 1-token sweep, fresh server per length
**Where (theirs):** `docs/gotchas.md:324-331`

**What it does.** Two measurement traps recorded as method. Sweep prompt length in steps of **1 token** — at a coarse grid one broken sample below and one above reads as a cliff, which is how it was first misdiagnosed. And send each length to a **fresh server**, or request N inherits request N−1's blocks and you measure history instead of length.

**Mechanism.** `bench/bugb_sweep.py` prints the `mod 128` column. The specific contamination named: `bench/labd_bench.py` sends two warm-ups on `doc[:4000]`, which arms the trigger for everything after it. Also: holding the document byte-identical and padding the *instruction* by one token turns a broken length clean, proving it is the token count rather than the corpus.

**Why they needed it.** Both traps produced wrong conclusions before they were understood — a coarse grid produced a fictitious "cliff", and a warm server produced results that were a function of request order.

**Their numbers.** Grid step must be 1 token; period is 128.

**llama.cpp — not applicable.** This is method, and it is the same method this project already runs on. The one concrete addition is the period: 256 here rather than 128 there, from an explicitly documented padding whose stated purpose is to keep the graph constant across batches.

**Equivalent here:** n/a — measurement method

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:1233-1246` · `src/llama-context.cpp:288` · `tools/server/server-context.cpp:2355-2363` · `common/common.h:615`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Both rules apply verbatim and one of them is already repo policy. The grid-resolution rule has a specific target here: n_kv is padded to a multiple of 256 (FATTN_KQ_STRIDE), and n_ctx is padded to 256 as well, so any length-dependent effect in this build will have period 256, not 128 — a coarse sweep will read as a cliff. The fresh-server rule is already enforced by swap-model.sh's lock, but note llama.cpp's own contamination source: --cache-idle-slots saves the previous slot's state into the RAM prompt cache on every new task, so request N genuinely inherits request N−1 unless the server is restarted or -cram 0 is set.

### `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is mandatory (and its WSL2 exception)
**Where (theirs):** `docs/gotchas.md:18-21` · `docs/docker.md:91-99`

**What it does.** The DeltaNet prefill kernels allocate transient workspace; without expandable segments the allocator fragments and the engine OOMs at runtime once `gpu-memory-utilization` goes past ~0.975. Both start scripts default it on. The exception: on some driver/dxgkrnl combinations its CUDA VMM calls crash Marlin repack.

**Mechanism.** PyTorch caching-allocator mode. The WSL2 failure signature is `RuntimeError: CUDA driver error: device not ready` inside `gptq_marlin_repack` on Windows driver 610.74 (WSL 2.1.5 and 2.7.12 alike; the driver-591.86 reproduction did not hit it). The scripts respect a pre-set value, so `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` in `.env` is the workaround.

**Why they needed it.** Batch mode runs at 0.972 utilization; without expandable segments the fragmentation ceiling is ~0.975, i.e. the tuned configuration does not exist without it.

**Their numbers.** OOM threshold ~0.975 utilization. Failure driver: Windows 610.74; working driver: 591.86.

**llama.cpp — not applicable.** Compute buffers are sized once by the reserve passes and ggml-alloc packs them; there is no per-step transient allocation to fragment a pool. The nearest llama.cpp concern is the opposite one — the reserve may under-budget if something widens n_batch afterwards, which is covered under technique 7/47.

**Equivalent here:** none — ggml allocates fixed buffers sized at reserve, there is no caching allocator to fragment

**Evidence (llama.cpp):** `src/llama-context.cpp:576-671` · `ggml/src/ggml-alloc.c:684`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero.

### Register selection env vars with vLLM so they enter the torch.compile cache key
**Where (theirs):** `docs/gotchas.md:28-32` · `docs/gotchas.md:76-80`

**What it does.** The torch.compile cache does not know about your env vars. Switching `INT8_LAYERS` between runs replays a compiled graph that expects the other layer set and dies with `KeyError: 'input_global_scale'`. The repo's patch registers the selection env vars with vLLM so they become part of the cache key.

**Mechanism.** `patches/speed-knobs-envs.patch` adds the knobs to `envs.py`. The general rule: "A stale torch.compile cache bites anything that changes tensor shapes behind vLLM's back. The compiled graph bakes in e.g. the Marlin workspace size; a new env knob that changes it must be registered in `envs.py` ... or you get `assert_size_stride ... expected size 328==82` from a cached artifact." Escape hatch for your own knobs: `VLLM_DISABLE_COMPILE_CACHE=1`.

**Why they needed it.** Any knob that changes tensor shapes is invisible to the compile cache unless declared, and the resulting failures name the symptom (a stride assert, a missing key) rather than the cause.

**Their numbers.** Two concrete failure strings recorded: `KeyError: 'input_global_scale'` and `assert_size_stride ... expected size 328==82`.

**llama.cpp — not applicable.** There is no persisted compiled artifact to go stale against a changed knob. The nearest hazard is at a different layer: a stale env var in a shell profile can kill startup outright, because LLAMA_ARG_DRAFT_MAX and LLAMA_ARG_DRAFT_MIN are still bound to removed options that call arg_removed() and abort.

**Equivalent here:** none — no compile cache; ggml graphs are rebuilt from the current parameters every run

**Evidence (llama.cpp):** `common/arg.cpp:4291-4325` · `common/arg.cpp:4297`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero.

### Restart once on a warm torch.compile cache before trusting the KV pool
**Where (theirs):** `docs/gotchas.md:81-90` · `docs/docker.md:81-90`

**What it does.** vLLM sizes the pool from the peak memory of a profiling forward pass, and on a cold torch.compile cache that pass also runs inductor's autotuning — inflating the measured peak. Batch mode profiles a 1.96 GiB activation peak instead of 1.09 GiB and comes up with 196k KV tokens instead of 224k. Restart once after the cache is warm and the pool is back to the README numbers.

**Mechanism.** Cache lives in `~/.cache/vllm` (venv) or the `qwen-cache` volume (Docker). The log tell is `Maximum concurrency ... 1.31x` instead of 1.49x. The WSL2 notes deliberately pin it the other way round — compile once from a cold cache, record vLLM's `Replace gpu_memory_utilization config with --kv-cache-memory=...` recommendation, verify the resulting pool exceeds `MAX_LEN`, and pass that machine/profile-specific byte value via `EXTRA_ARGS` — if you prefer transient headroom to KV pages. "Do not copy a byte value from a different card or profile."

**Why they needed it.** Two legitimate but opposite policies for the same phenomenon: maximise KV pages (warm start) or maximise transient headroom and determinism (cold-start byte value). The repo documents both and names the trade.

**Their numbers.** Activation peak 1.96 GiB cold vs 1.09 GiB warm → 196k vs 224k KV tokens; `Maximum concurrency` 1.31× vs 1.49×.

**llama.cpp — not applicable.** vLLM documents two legitimate opposite policies for the same phenomenon. llama.cpp's version of the phenomenon is not a compile cache but the free-memory-at-that-instant measurement, so the mechanism differs while the decision is identical.

**Equivalent here:** none — but the underlying policy question (maximise pages versus maximise determinism) maps onto -c / -fitt exactly

**Evidence (llama.cpp):** `common/fit.cpp:559-563` · `common/fit.h:15-18` · `common/arg.cpp:2851-2874`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No compile cache to warm. The policy fork does transfer and is worth deciding explicitly, once, in the repo: either let --fit maximise context per boot and accept that the number moves, or pin -c and -fitt and accept a smaller but reproducible configuration. For a measurement project the second is almost certainly right, and it is the same argument as technique 22.

### vLLM picks the speculative method from the model *path* string
**Where (theirs):** `docs/gotchas.md:91-96`

**What it does.** `"dflash" in model_path` switches `method` to dflash — for the *target* too, since MTP uses the target path as its draft model. A checkout under a directory with "dflash" in its name turns `SPEC=mtp` into a crash in `EAGLEConfig` (`'Qwen3_5Config' object has no attribute 'vocab_size'`). "Name your directories accordingly."

**Mechanism.** Substring match on the model path inside vLLM's speculative-config resolution.

**Why they needed it.** A filesystem-layout choice silently changes engine semantics, and the error names a config class rather than the path.

**Their numbers.** Failure string: `'Qwen3_5Config' object has no attribute 'vocab_size'`.

**llama.cpp — not applicable.** The lesson generalises: a configuration surface that infers rather than states will eventually infer something wrong quietly. llama.cpp infers from better evidence than a path string, but its append semantics are the sharper trap for us because we run scripted profiles.

**Equivalent here:** auto-detection reads the GGUF arch and tensor names plus a sidecar precedence list, never the path

**Evidence (llama.cpp):** `common/arg.cpp:4153-4162` · `common/speculative.cpp:2286-2288` · `common/arg.cpp:565` · `tools/server/server-task.cpp:83`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No path hazard, but three real argument hazards of the same family, each of which produces a working-but-wrong server rather than an error. (a) --spec-type APPENDS and never replaces; passing it twice accumulates both lists, and the usual 'specified multiple times' warning is deliberately suppressed for this one flag — so a wrapper script plus a manual flag silently gives you a chain. (b) If any name in one invocation is `none`, that invocation returns exactly {NONE} and discards everything else in it — but a later --spec-type appends to it. (c) The GGUF sniff reads ONLY the first split, so a sharded draft needs an explicit --spec-type. Confirm what actually took effect from GET /props, which reports the accumulated deduplicated speculative.types list.

### Delete `__pycache__` after installing patched files — rsync preserves mtimes and Python trusts them
**Where (theirs):** `docs/gotchas.md:148-151`

**What it does.** Copying a source file into `site-packages` with `rsync -a` can leave the `.pyc` newer than the `.py`, in which case the interpreter keeps running the old bytecode and every measurement lands on the previous revision.

**Mechanism.** `rsync -a` preserves mtimes; CPython's bytecode-staleness check is mtime-based. Fix: delete `__pycache__` after installing patched files.

**Why they needed it.** Recorded because it makes an entire measurement round silently describe the previous code revision — the same failure class as every other entry here.

**Their numbers.** None given.

**llama.cpp — not applicable.** The mechanism differs (mtime-based bytecode invalidation versus two directories) but the outcome and the detection difficulty are the same, and this repo already has a recorded incident of a delegated task reporting success with its file in the wrong directory.

**Equivalent here:** none — a compiled binary has no bytecode staleness

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/CMakeLists.txt:134-137`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Zero as stated, but the failure class is live here in a different form: the source tree is C:\AI\llama.cpp and the binary under measurement is C:\AI\llama.cpp-dflash2\llama-server.exe. Editing one and benchmarking the other produces a measurement that silently describes the previous revision — the identical fault. Any change to common/speculative.cpp must be followed by a rebuild and a restage, and the run should record the binary's --version build number, not the tree's commit.

### A model dir with no `tokenizer.json` is an empty vocabulary, reported as a reasoning-parser error
**Where (theirs):** `docs/gotchas.md:252-273`

**What it does.** `AutoTokenizer.from_pretrained` on a dir that has `config.json` but no tokenizer files returns a `Qwen2Tokenizer` with `vocab_size == 1` that encodes *everything* to `[]` — `tok.encode("hello world")` is `[]`, not an exception. Nothing complains until `VllmConfig.__post_init__` asks the qwen3 reasoning parser for `<think>`, gets `[]` back, and raises an error naming neither the tokenizer nor the directory.

**Mechanism.** The fix is at the verification layer, not the code layer: `verify.sh` now encodes `<think>` against every dir passed to `--model` rather than only checking the dir exists, and `docker/prepare.sh` counts `tokenizer.json` as part of a complete download.

**Why they needed it.** The error message is actively misleading: it "prints the strings as empty because they are the *unset* config fields, not the ones the parser supplied" (docs/gotchas.md:265-266). It was filed as a `SPEC=dflash2` bug (issue #15) and reproduces with no speculative config at all; only the single-user modes failed because they serve `models/Qwen3.8-27B-W4A16-AutoRound-fast` while batch mode serves the base dir.

**Their numbers.** `vocab_size == 1`. Error text: `ReasoningConfig: failed to tokenize reasoning strings: reasoning_start_str='', reasoning_end_str=''.`

**llama.cpp — not applicable.** The specific failure cannot occur, but the family — a missing or wrong input reported through an unrelated downstream component — has llama.cpp members, and both of the ones above would mislead someone reading a startup log to confirm what was loaded.

**Equivalent here:** the vocab is embedded in the GGUF and cannot be missing

**Evidence (llama.cpp):** `src/llama-model-loader.cpp:723-785` · `src/llama-model-loader.cpp:770-774` · `src/llama-model-loader.cpp:255-262`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Zero as stated. Two llama.cpp analogues of 'a load-time defect reported as something else' are worth knowing: the printed `file type` is only a guess from the most-common tensor type and an unknown type_max silently reports ALL_F32, so the startup header can be wrong with no error; and --override-kv can INJECT a metadata key that is absent from the file, because try_override runs before the key-not-found check.

### `SPEC=dflash2 CTX=huge` — KVarN on the V2 runner with no kernel work
**Where (theirs):** `README.md:98-134`

**What it does.** Combines the DFlash2 block drafter with the KVarN 4/2-bit cache: 268,169 tokens of pool at 245,760 max-model-len on the same pinned budget. The KVarN Triton kernels run unmodified on the V2 runner; the seven fixes in `kvarn/kvarn-v2-runner.patch` are allocator and geometry logic — including an upstream vLLM bug in the mamba align resume path, and a NaN path in the DFlash2 candidate selector that KVarN noise exposes on verbatim-reproduction content.

**Mechanism.** `bash kvarn/install.sh` applies `kvarn-v2-runner.patch` as its second stage. On WSL2 also set `VLLM_WSL2_ENABLE_PIN_MEMORY=1` — the V2 runner needs pinned memory and vLLM leaves it off by default there; its UVA buffers work fine on the paravirt driver.

**Why they needed it.** It is the only path to 240k with a block drafter, and its correctness argument is unusually careful because the mode inherits a lossy cache: 95.2% GSM8K at n=600 (±0.9 points) "sits inside the band rather than below it — which is the useful comparison, since this mode inherits KVarN's lossy 4/2-bit cache and should be judged against the other lossy configurations rather than against bf16." (README.md:129-132)

**Their numbers.** Two machines (WSL2 contributor box / this repo's bare metal), both RTX 3090 at 250 W, `bench/labd_bench.py --ctx 20000` (README.md:116-125): copy 130 tok/s @7.8 tok/step vs 164 @7.83; code/edit/quote/summary/qa 89/65/44/38/36 vs 109/83/58/51/43; all six 53 @3.0 vs 67 @3.15; verbatim reproduction of a 25k document correct / 1,150 of 1,150 chars; KV capacity 268,169 tokens on both; GSM8K 97.0% (n=200) vs 95.2% (n=600) and 95.0% (n=200); turn 2 over a 100k cached prefix 4.7 s vs 169 s cold. Deep-needle retrieval passes at 200k; 100k-deep needle correct on both turns.

**llama.cpp — not applicable.** This entry is the intersection of technique 16 and technique 38; the first is already ours and the second is out of reach, so the combination has nothing to add. The one carry-over is their correctness framing — judge a lossy-cache mode against the other lossy configurations rather than against the unquantised baseline — which applies directly to any quality claim we make about -ctk q4_0 on a 2-bit model.

**Equivalent here:** none — no KVarN, no second runner

**Evidence (llama.cpp):** `common/speculative.cpp:910-1347` · `ggml/src/ggml-cuda/fattn.cu:338-356`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. The composite does not exist and its two ingredients are covered separately (draft-dflash: already have it; KVarN: new-backend).

### Quality accounting: IFBench, perplexity, GSM8K per configuration, with the honest baseline
**Where (theirs):** `docs/quality.md:9-25` · `docs/quality.md:49-61` · `README.md:244-253`

**What it does.** Three checks against the exact serving stack rather than against an offline model. IFBench (AllenAI, 299 prompts, official eval scripts) with thinking at `reasoning_effort: xhigh` and model-default sampling; perplexity on ~33k held-out tokens (English Wikipedia, Danish web text, Python source); GSM8K (200 test questions, greedy, thinking off). The comparison point is Qwen's own model card for the unquantized model.

**Mechanism.** Per-configuration tables rather than one headline. The framing rule stated for speculation: single-user mode is W4A16 (int8 activations buy nothing at batch size 1) and speculative decoding is exact by construction, so with the base requantization its quality *is* the W4A16 row — only the int8-activation knobs in batch mode trade accuracy for speed.

**Why they needed it.** "The whole stack is quantized, so the honest question is what it costs." (README.md:246). And on the drafter specifically: "The MTP module's precision never touches output quality (drafts are verified exactly); it only moves acceptance, and the calibrated int4 keeps it." (docs/quality.md:60-61)

**Their numbers.** IFBench: strict W4A16 stack **78.3** prompt-level / 79.9 instruction-level; loose 81.7 / 82.8; strict batch-mode default (int8 MLP activations, fp16 state) **78.3** / 80.5; loose 80.3 / 82.3. Qwen's model card reports **79.5** unquantized — "about one point on the headline metric", and the batch-mode int8 activations cost nothing measurable (the two runs trade places within sampling noise on the sub-metrics). Single-user variants (docs/quality.md:54-58): base requantization (int8 lm_head, new draft vocab) PPL en 10.68 / da 10.85 / code 3.05 / all 8.045, GSM8K 95.5%, 107 / 109 tok/s C1; int4 lm_head round-to-nearest (not shipped) 10.81 / 11.09 / 3.07 / 8.17 (+1.5%), 109 / 112; **fast variant** (int4 lm_head GPTQ + int4 MTP GPTQ) 10.77 / 10.91 / 3.06 / **8.095 (+0.6%)**, GSM8K **96.5%**, ~114 / ~124. Note the round-to-nearest vs GPTQ contrast: same bit width, +1.5% vs +0.6% perplexity.

**llama.cpp — not applicable.** vLLM's separation of exact-by-construction knobs from lossy ones is exactly the right accounting and it maps cleanly, but the set membership differs: their lossy knobs were activation quantisation, ours is the KV cache. Their MTP-precision point does transfer verbatim — the sidecar's precision never touches output quality, only acceptance — which de-risks technique 8 and 19 considerably.

**Equivalent here:** n/a as a llama.cpp capability; llama-perplexity exists in the tree but is not in the staged binary set

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:308-338` · `common/sampling.cpp:692-720` · `tools/server/server-context.cpp:3825-3831`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The framing is what matters and it applies to us with one extra step. Their rule: speculation is exact by construction, so a speculating configuration's quality IS the underlying quant's quality, and only the knobs that change the model's own arithmetic trade accuracy. That mostly holds here — but two of our settings are NOT exact. -ctk q4_0 -ctv q4_0 changes the target's logits, and the automatic Hadamard rotation that mitigates it is silently skipped if the head dim is not a multiple of 64. So a per-configuration quality table for us needs rows for the KV type, not for the speculator. Confirm the rotation engaged by reading the attn_rot_k / attn_rot_v lines at load.

### Mode selection: speculation below ~C8, plain batching above
**Where (theirs):** `README.md:9-21` · `batch/README.md:44-56` · `single-user/README.md:50-54` · `single-user/README.md:85-94`

**What it does.** Both modes share one install — the mode is just which launch script you run. The published crossover: speculation wins below ~8 concurrent users, plain batching above. The single-user tables and batch cohort tables use the same protocol so they can be compared directly.

**Mechanism.** Two reasons for the crossover are given rather than one. First, rejected drafts cost more when the verify batch is bigger, so even within MTP the four-draft config wins to C2 and the three-draft config is ahead from C4 up. Second, memory: every DFlash2 request reserves 1+k = 8 recurrent-state slots (≈0.7 GB, vs 5 slots / 0.44 GB for MTP k=4), so only 5-6 long generations are resident and the rest queue — visible as `Running: 5 reqs, Waiting: 3` and a 2.7 s TTFT at C8.

**Why they needed it.** "Four drafts win up to two concurrent users; from C4 up the three-draft config is ahead (rejected drafts cost more when the verify batch is bigger), so for a shared box `CTX=long` or `DRAFT_TOKENS=3` is the better single-user config." (single-user/README.md:50-52). And: "One GPU, 1-4 users: DFlash2; more: MTP or batch mode." (single-user/README.md:88)

**Their numbers.** Batch cohorts (batch/README.md:46-49): C1 45.4 e2e / 45.6 decode / 110 ms TTFT; C2 81.8 / 83.7 / 197 ms; C4 153.8 / 162.7 / 343 ms; C8 298.4 / 321.0 / 638 ms; C64 ~1,035-1,094 decode. Single-user MTP `CTX=fast` (single-user/README.md:23-26): C1 111.1 / 120.0 decode, 2.75 / 2.90 tok/step, 108.7 / 116.9 e2e, 164 ms TTFT; C2 191.8 / 199.4; C4 268.5 / 280.9; C8 407.3 / 414.3 with 1,005 ms TTFT. DFlash2 at C8: 389.9 / 405.5 decode but only 252.1 / 274.1 e2e and 2,688 ms TTFT — MTP reads 329 e2e at C8 against DFlash2's 252. Per-position draft acceptance at C1 (MTP): 74% / 50% / 34% / 24% at T 1.0; 77% / 55% / 40% / 30% greedy. `CTX=long` cohorts with base requant: C1 84.7 / 89.3, C2 168.2 / 177.6, C4 289.2 / 303.7, C8 409.0 / 450.5. DFlash2 long-context weakness: 2.3-2.6 tokens per step at 12k/36k against MTP's 2.6-3.0, drafter prefill adds ~15% to TTFT, end to end within 5-10% with MTP ahead.

**llama.cpp — not applicable.** llama.cpp's per-slot draft budget dp.n_max is computed per sequence per step and the drafters batch across sequences, so the crossover mechanism exists — but with one slot the multi-sequence machinery collapses and the question does not arise.

**Equivalent here:** -np 1 with speculation is the only regime this target runs in

**Evidence (llama.cpp):** `tools/server/server-context.cpp:441-460` · `common/speculative.cpp:2728-2733` · `tools/server/server-context.cpp:1224-1226`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No decision to make — one card, one coding agent, one slot. One sub-finding is worth carrying: rejected drafts cost more when the verify batch is bigger, which is why their four-draft config wins to C2 and the three-draft config leads from C4. At -np 1 that argues for the larger n_max, which supports the technique 14 sweep going upward rather than downward. The counter-pressure is entirely from the kernel stairs (technique 34) and the recurrent-state multiplier (technique 35), not from concurrency.

### Cross-engine comparison discipline (vs ninfer-3090)
**Where (theirs):** `README.md:211-242` · `single-user/README.md:229-234`

**What it does.** Compares against ninfer-3090, a standalone C++/CUDA engine publishing cohort benchmarks for the same model on the same card, and enumerates every way the comparison is not like-for-like — including one the repo got wrong itself and corrected.

**Mechanism.** Their protocol: 1,024-token answers from 29-34-token prompts, greedy, MTP3, int8 KV, prefix reuse off, 8,192-token window, thinking **on** at `reasoning_effort=medium` — so their 1,024 tokens include reasoning. This repo's: 8 realistic chat prompts (English, Danish, code), 1,024-token answers, model-default sampling, thinking off. The correction: "Theirs is the **decode** column of their table; their end-to-end column reads 70.19 / 89.43 / 97.89 / 161.28, and an earlier version of this table quoted *those* against our decode rate, which was not like-for-like." (README.md:235-238)

**Why they needed it.** Remaining asymmetries are listed in both directions: "their C1 is a single prompt in a single run with no error bars, thinking is on for them and off for us, and they publish no power limit or driver version — ours is an RTX 3090 pinned at 250 W." Attribution of the gap: "mostly vLLM's continuous batching plus the memory this repo's requantization frees up."

**Their numbers.** Decode rate (C × 1000 / mean TPOT), ninfer-3090 MTP3 vs this repo batch / single-user MTP / single-user DFlash2 (README.md:220-226): C1 71.00 vs 45.5 / 111.1 / **121.8**; C2 90.66 vs 86.3 / 191.8 / **195.5**; C4 100.28 vs 168.3 / 268.5 / **278.9**; C8 165.33 vs 324.9 / **407.3** / 389.9; C64 not supported vs **~1,035**. Greedy DFlash2 reads 131.2 / 214.6 / 285.7 / 405.5. Run-to-run spread 5-8%, "so treat one-decimal differences between the three right-hand columns as noise — C1 and C8 are where the modes genuinely separate." Peak VRAM comparable: 23.0 vs 22.1 GiB at C8.

**llama.cpp — not applicable.** Pure methodology, and the most portable item in the slice. Its value here is that llama.cpp's server publishes several superficially similar rates with different denominators, and the retraction they document is precisely the mistake those denominators invite.

**Equivalent here:** n/a as a capability; but llama.cpp's own timing definitions are the like-for-like hazard

**Evidence (llama.cpp):** `tools/server/server-common.h:363-366` · `tools/server/server-common.h:400-402` · `tools/server/server-context.cpp:1268-1273` · `tools/server/server-context.cpp:4037-4043`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The rule — enumerate every way the comparison is not like-for-like, including the ones you got wrong — applies to any number we publish against another engine, and llama.cpp hands us three specific definitional gotchas to declare when we do. prompt_n EXCLUDES cached tokens while prompt_ms is measured from slot entry, so cache lookup, checkpoint restore and RAM-cache load are inside the numerator and outside the denominator. predicted_per_second divides by n_gen−1. And /metrics counts a generation only on slot release, so an in-flight request contributes nothing. Quoting a decode column against someone else's end-to-end column is the exact error they had to retract.
