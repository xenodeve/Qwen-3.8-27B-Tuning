# kvarn Triton decode kernels + tile store: triton_kvarn_decode.py (1196 lines), kvarn_store.py (286), triton_kvarn_sinkhorn.py (206), kvarn_decode.py (119)
**38 techniques.** 1807 source lines across 4 files.
Files read: `kvarn/files/vllm/v1/attention/ops/triton_kvarn_decode.py` · `kvarn/files/vllm/v1/attention/ops/kvarn_store.py` · `kvarn/files/vllm/v1/attention/ops/triton_kvarn_sinkhorn.py` · `kvarn/files/vllm/v1/attention/ops/kvarn_decode.py`
> **What the reader could not see:** Five things this slice references but does not contain, so the numbers/offsets they own could not be verified here: (1) `KVarNConfig` — owns `k_packed_offset`/`k_s_col_offset`/`k_zp_offset`/`k_s_row_offset`/`v_*` byte offsets and `tile_bytes_aligned`; the decode kernels take all of them as constexprs and never compute them (triton_kvarn_decode.py:193-200). The only figure quoted here is the docstring's "26880 for kvarn_k4v2_g128 @ head_dim 256" (triton_kvarn_decode.py:20-21) — unverified in-slice. (2) `triton_kvarn_store.py`, the Triton port that must be "byte-identical" to kvarn_store.py (kvarn_store.py:5-6) — so the actual serving-path quantiser is NOT in this slice; kvarn_store.py is the reference. (3) `.../quantization/kvarn/sinkhorn.py` with `variance_normalize` and `variance_normalize_batched` (kvarn_store.py:24-26, triton_kvarn_sinkhorn.py:168-171) — the PyTorch reference the Triton Sinkhorn claims semantic equality with. (4) `KVarNAttentionImpl` / `KVarNMetadataBuilder` — own `_tail_K_pool`, `_block_to_slot_t`, `_mid_o_buf`, `_H_fp16`, `_block_lookup_size`, `_sm_count`, and the pool-slot allocation policy; the 128-token tile *eviction/flush* decision (when a pool block becomes an int4 block) lives there, not here. (5) `_warm_decode_kernels`, cited as the pre-graph-capture autotune warmup (triton_kvarn_decode.py:513-514). Also absent: any measured accuracy number for the quantiser — only the target "cosine ≥ 0.999 vs this reference" (kvarn_decode.py:9-10) and the Sinkhorn's "~5e-7 rel" tile-output tolerance appear. No end-to-end tok/s figure for Qwen3.8-27B on a 3090 appears anywhere in these four files; every number is a microbench or a ratio.

---

## EXISTS, NEVER SET — 3

### Asymmetric per-row RTN with zero-point = row minimum
**Where (theirs):** `kvarn/files/vllm/v1/attention/ops/kvarn_store.py:50-70`

**What it does.** Quantisation of the balanced tile is asymmetric round-to-nearest over the full row with no sub-grouping: scale is (max−min)/qmax, zero-point is literally the row minimum, and the quantised value is clamped into [0, 2^bits−1].

**Mechanism.** `qmax = (1 << bits) - 1`; `scale = ((hi - lo)/qmax).clamp_min(1e-10)`; `zp = lo`; `q = clamp(round((tile - zp)/scale), 0, qmax).to(int32)` (kvarn_store.py:64-69). Returns `[R,C]` int32, `[R,1]` scale, `[R,1]` zp.

**Why they needed it.** 'Per-row asymmetric RTN over the full row (no sub-grouping)' — the Sinkhorn pass is what makes full-row RTN viable; the usual defence against per-row dynamic range (sub-group scales at 32 or 64 elements) is replaced by the variance balancing, which is why the tile only needs one scale per row. `clamp_min(1e-10)` guards a constant row.

**Their numbers.** bits ∈ {2, 3, 4} per docstring (kvarn_store.py:55); the non-batched entry points assert bits == 4 ('Stage 3a only validates 4-bit', kvarn_store.py:127, 257).

**llama.cpp — EXISTS, NEVER SET.** llama.cpp already has asymmetric round-to-nearest quantisation with a min-based zero point — at block-of-32 granularity rather than full-row, which is strictly finer and is why it needs no Sinkhorn pre-balancing to be viable. The flag exists and parses today; only the CUDA FA kernel is missing from this build.

**Equivalent here:** -ctk q4_1 / -ctv q4_1 (and q5_1) — asymmetric scale+min block quants, accepted by the parser

**Evidence (llama.cpp):** `common/arg.cpp:305-315 (kv_cache_types includes q4_1, q5_0, q5_1)` · `ggml/src/ggml-cuda/fattn.cu:343-348 (Q4_1/Q5_0/Q5_1 return false under #ifndef GGML_CUDA_FA_ALL_QUANTS)` · `ggml/src/ggml-cuda/fattn-vec.cuh:580-585 (DECL_FATTN_VEC_CASE instances for Q4_1/Q5_0/Q5_1 exist in the header)` · `C:\AI\llama.cpp\build-dflash2\CMakeCache.txt:660 (GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF)` · `ggml/src/ggml-cuda/cpy.cu:310-322 (the q4_1 KV copy kernels are compiled)`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Reachable only by rebuilding with -DGGML_CUDA_FA_ALL_QUANTS=ON. Note the direction of the trade: q4_1 stores a scale *and* a min per 32-element block, so it is 5.0 bits/element against q4_0's 4.5 — about +11% KV VRAM, which on this card costs context. It buys asymmetry, which is what the technique is for. Whether that is worth 11% of the KV budget on a 2-bit-weight model is unmeasured. Do not rebuild for this alone; rebuild once and get q5_0/q5_1 and asymmetric -ctk/-ctv in the same pass if the accuracy question is worth answering.

### Sliding-window block-loop truncation
**Where (theirs):** `triton_kvarn_decode.py:419-425` · `triton_kvarn_decode.py:449-452`

**What it does.** For a sliding-window layer the block loop starts at the first block overlapping the window instead of block 0, and keys before the window boundary are additionally masked inside the chunk.

**Mechanism.** `win_start = max(seq_len - SLIDING_WINDOW, 0)`, `blk_lo = win_start // GROUP`, then `for k in range(blk_lo, n_blocks)` (triton_kvarn_decode.py:423-426). Inside: `cmask = cmask & ((k*GROUP + cols) >= win_start)` (triton_kvarn_decode.py:451-452). SLIDING_WINDOW is a constexpr so the whole thing compiles away when zero.

**Why they needed it.** 'massive saving: ~window/GROUP blocks instead of all'. This is also the stated reason split-K is disabled for these layers — they are already cheap enough that the single-stage grid suffices (triton_kvarn_decode.py:785-787).

**llama.cpp — EXISTS, NEVER SET.** llama.cpp achieves more than the loop truncation (it does not allocate the out-of-window cells at all) but the whole path is unreachable on a model with no SWA layers.

**Equivalent here:** llama_kv_cache_iswa / llama_memory_hybrid_iswa with size_swa = n_swa + n_ubatch, and --swa-full to defeat it

**Evidence (llama.cpp):** `src/llama-kv-cache-iswa.cpp:70-79 (size_swa = GGML_PAD(min(size_base, n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256))` · `tools/server/server-context.cpp:1188-1195 (the server force-disables --swa-full and sets n_swa = 0 on a model with no SWA layers)` · `src/llama-model.cpp:2305 (with swa_type == NONE the plain llama_memory_hybrid is selected, not the iswa variant)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None for our model. llama.cpp's version of this saving is structural — an SWA layer gets a physically smaller cache rather than a truncated loop — but the map records that qwen35 declares no SWA, so swa_type stays NONE, --swa-full is inert, and there is no window to truncate. If the target model turns out to declare SWA layers this flips; check the startup log for a non-zero n_swa before assuming either way.

### Materialize path retained as an A/B control
**Where (theirs):** `triton_kvarn_decode.py:728-734` · `triton_kvarn_decode.py:826-859`

**What it does.** The older path — build a packed varlen fp16 K/V buffer then call stock flash_attn_varlen_func — is kept alive behind KVARN_FUSED_DECODE=0 as both a fallback and a comparison baseline.

**Mechanism.** `_kvarn_build_packed_kv_kernel[(B * max_blocks_per_req, Hk)]` fills `impl._fa_K_buf`/`_fa_V_buf` (triton_kvarn_decode.py:830-848), then `flash_attn_varlen_func(q=..., cu_seqlens_q=md.fa_cu_seqlens_q, cu_seqlens_k=md.fa_cu_seqlens_k, max_seqlen_q=1, max_seqlen_k=md.fa_max_seqlen_k_fixed, causal=False)` (triton_kvarn_decode.py:850-859). Note `max_seqlen_k` is a FIXED value from metadata, not a per-batch max — a graph-capture requirement.

**Why they needed it.** 'kept for A/B and as a fallback' (triton_kvarn_decode.py:734). The value is that the traffic claim (0.25x vs 2.25x) can be measured on the same machine in the same session rather than argued.

**llama.cpp — EXISTS, NEVER SET.** The discipline — keep the slower reference path reachable so the claim can be measured rather than argued — is available here as env kill switches rather than as a second code path, and the switches are unset in our profile.

**Equivalent here:** the env kill switches and the -fa / -ctk pairings, which are the A/B levers this build ships

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:313-317 (LLAMA_ATTN_ROT_DISABLE=1)` · `src/llama-context.cpp:279-285 (LLAMA_GRAPH_REUSE_DISABLE=1)` · `ggml/src/ggml-cuda/common.cuh:1255-1259 (GGML_CUDA_DISABLE_GRAPHS)` · `ggml/src/ggml-cuda/ggml-cuda.cu:4318-4344 (GGML_CUDA_GRAPH_OPT=1, requires graphs on and exactly one CUDA device — both true here)` · `common/arg.cpp:1744-1758 (-fa on|off|auto)` · `common/arg.cpp:2426-2450 (-ctk / -ctv)`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Four one-variable controls we have and have never used, each of which isolates one of the mechanisms this slice raised: LLAMA_ATTN_ROT_DISABLE isolates the Hadamard (technique 35), LLAMA_GRAPH_REUSE_DISABLE and GGML_CUDA_DISABLE_GRAPHS jointly attribute any speculation regression to graph reuse versus graph capture (technique 34), and -ctk f16 versus q4_0 under a fixed draft depth prices the MMA dequant (technique 11). All are same-boot, same-round pairable, which is what our own methodology requires. Note the irony worth recording: llama.cpp's quantised-KV path above two query columns IS the materialise path — we have been running the control, not the treatment, on every drafted step.

## absent, has a seam — 5

### Sinkhorn-then-RTN two-level scale factorisation
**Where (theirs):** `kvarn/files/vllm/v1/attention/ops/kvarn_store.py:131-148` · `kvarn/files/vllm/v1/attention/ops/kvarn_store.py:261-278` · `kvarn/files/vllm/v1/attention/ops/kvarn_decode.py:83`

**What it does.** A tile is first variance-balanced by a Sinkhorn-style alternating row/column normalisation, which yields a per-channel scale and a per-token scale. Only THEN is round-to-nearest asymmetric quantisation applied per row of the balanced tile. The RTN scale and zero-point are folded into one of the two Sinkhorn vectors, so the cache stores exactly three fp16 vectors per tile per axis rather than four.

**Mechanism.** `kvarn_store_tile_k` calls `variance_normalize(tile, iterations=16)` on the `[D, group]` tile (kvarn_store.py:131-133), takes `s_row_sinkhorn` as the per-channel vector and `s_col_sinkhorn` as the per-token vector (kvarn_store.py:137-138), runs `_asymmetric_rtn_per_row` on the balanced tile, then absorbs: `s_col_K = s_chan * rtn_scale`, `zp_K = s_chan * rtn_zp`, `s_row_K = s_tok` (kvarn_store.py:144-146). V is the mirror image in `[group, D]` orientation: absorption goes into the per-token vector, `s_row_V = s_tok * rtn_scale`, `zp_V = s_tok * rtn_zp`, `s_col_V` untouched (kvarn_store.py:274-276).

**Why they needed it.** The absorption is what makes dequantisation a two-multiply expression with no extra memory traffic — the identity the whole decode kernel is built around is `out[r,c] = (q[r,c] * s_col_K[r] + zp_K[r]) * s_row_K[c]` (kvarn_decode.py:83). Note the zero-point is inside the outer product: `zp` is a per-channel constant scaled by the per-token factor, not a plain additive offset, which is why a stock int4 dequant routine cannot be substituted.

**Their numbers.** sinkhorn_iters default 16 (kvarn_store.py:110, 240); reference-vs-Triton accuracy target is 'cosine ≥ 0.999' (kvarn_decode.py:9-10).

**llama.cpp — absent, has a seam.** llama.cpp's KV cache is a single ggml_type tensor per layer; every scale a block quant carries is inline in the block struct. A two-level factorisation with separate per-channel and per-token fp16 vectors is not expressible as a ggml_type (type traits are blck_size/type_size/from_float, row-wise). The seam is nameable — a new block layout in ggml-common.h, a new f32->type kernel in ggml-cuda/cpy.cu, dequant in fattn-vec.cuh, and the whitelist at arg.cpp:305-315 — but every one of those is new code, and the FA kernel is the hard part.

**Equivalent here:** none; nearest is the automatic Hadamard rotation of quantised K/V (attn_rot_k / attn_rot_v), which attacks the same outlier problem by a different route

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:319-336 (attn_rot_k/attn_rot_v enabled whenever ggml_is_quantized(type_k) && n_embd_head_k % 64 == 0)` · `src/llama-kv-cache.cpp:231 (k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) — the cache is one plain ggml_type, no side scale vectors)` · `ggml/src/ggml-cuda/cpy.cu:273-283 (ggml_cpy_f32_q4_0_cuda / cpy_blck_f32_q4_0 — the entire KV write path, per-block absmax)` · `common/arg.cpp:305-315 (kv_cache_types whitelist: nine ggml_types, nothing factorised)`

**Effort:** new-backend · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Would be the only route below q4_0's 4.5 bits/element, which is the smallest GPU-usable KV type on this binary. On 12 GB with a 6.77 GB model, KV is what --fit trades context against, so a real 2-bit KV roughly doubles the context --fit will grant. But llama.cpp already ships the Hadamard rotation for the same outlier problem, so the marginal accuracy gain of Sinkhorn on top is unknown, and the cost is a new ggml quant type plus a new fattn kernel family — a fork, not a config.

### Log-domain Sinkhorn with best-so-far imbalance tracking
**Where (theirs):** `kvarn/files/vllm/v1/attention/ops/triton_kvarn_sinkhorn.py:88-127` · `kvarn/files/vllm/v1/attention/ops/triton_kvarn_sinkhorn.py:82`

**What it does.** The 16 alternating column/row normalisation passes accumulate scales in log space, and each iteration re-measures an imbalance metric. The scales written out are not the last iteration's but the lowest-imbalance ones seen across all iterations — the loop is not assumed to converge monotonically.

**Mechanism.** `log_s_col`/`log_s_row` start at zero (exp = 1) (triton_kvarn_sinkhorn.py:64-65). Each iteration adds `tl.log(col_std_clipped)` to `log_s_col`, rebuilds `cur = tile / s_col_lin / s_row_lin`, then does the same for rows (triton_kvarn_sinkhorn.py:90-109). Imbalance is `max(col_std)/min(col_std) + max(row_std)/min(row_std)` with the minima floored at 1e-8 (triton_kvarn_sinkhorn.py:118-122); `better = imb <= imb_best` selects `sc_best`/`sr_best` via `tl.where` (triton_kvarn_sinkhorn.py:124-127). The final balanced tile is recomputed from the BEST scales, not the current ones (triton_kvarn_sinkhorn.py:130).

**Why they needed it.** The header states the Triton kernel must match the PyTorch reference: 'same 16 alternating col/row std-normalization passes, same best-so-far tracking via the imbalance metric, same clamps' (triton_kvarn_sinkhorn.py:6-9). Keeping the best snapshot means an iteration count that overshoots cannot make the tile worse than not iterating at all — the initial (identity-scale) imbalance is itself seeded as `imb_best` before the loop (triton_kvarn_sinkhorn.py:82).

**Their numbers.** ITERATIONS default 16 (triton_kvarn_sinkhorn.py:141, 149).

**llama.cpp — absent, has a seam.** This is an internal detail of a pass that does not exist here. Same seam as technique 1 and only meaningful contingent on it.

**Equivalent here:** none

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/cpy.cu:273-283 (the only KV quantisation kernel; single-pass absmax, no iteration)` · `src/llama-kv-cache.cpp:20-57 (ggml_gen_hadamard — llama.cpp's balancing step is a fixed orthonormal matrix, not an iterated fit)`

**Effort:** new-backend · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** None unless the Sinkhorn factorisation above is built. The transferable idea in isolation is the discipline: an iterative refinement that keeps the best-measured snapshot rather than the last iterate, because convergence is not monotone. llama.cpp has no iterative quantisation anywhere in the KV path to apply it to.

### KVARN_RTN_QUANTILE — percentile clipping instead of min/max
**Where (theirs):** `kvarn/files/vllm/v1/attention/ops/kvarn_store.py:29-42` · `kvarn/files/vllm/v1/attention/ops/kvarn_store.py:187` · `kvarn/files/vllm/v1/attention/ops/kvarn_store.py:221`

**What it does.** An opt-in env var replaces the per-row min/max used to set the quant range with the [q, 1−q] percentiles, deliberately clipping outliers so the remaining bits resolve the bulk of the distribution more finely.

**Mechanism.** `_rtn_range` reads `KVARN_RTN_QUANTILE`; if set and > 0 it returns `torch.quantile(t, q, dim=dim)` and `torch.quantile(t, 1-q, dim=dim)` instead of `amin`/`amax` (kvarn_store.py:36-42). Values outside the range are then clipped by the existing `torch.clamp(..., 0, qmax)` at quantise time. Only the BATCHED paths call it (kvarn_store.py:187, 221); the single-tile `_asymmetric_rtn_per_row` still uses raw amin/amax (kvarn_store.py:65-66).

**Why they needed it.** Quoting: 'Critical for k2v2 on models like Qwen3-30B-A3B-Thinking where K outliers (max/std ≈ 6) waste 2-bit resolution.' At 2 bits there are four levels total, so a single 6-sigma outlier consumes most of the range.

**Their numbers.** suggested q = 0.005; the motivating outlier statistic is K max/std ≈ 6 on Qwen3-30B-A3B-Thinking under the k2v2 preset.

**llama.cpp — absent, has a seam.** There is no percentile option and no env knob anywhere in the quantisation path, but the modification point is a single small kernel, so this is genuinely absent-but-possible rather than impossible.

**Equivalent here:** none

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/cpy.cu:273-283 (ggml_cpy_f32_q4_0_cuda -> cpy_blck_f32_q4_0: the single place a KV value becomes q4_0 on this GPU)` · `src/llama-kv-cache.cpp:319-336 (the Hadamard rotation already spreads outliers before quantisation)`

**Effort:** small-patch · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** The seam is small and named: cpy_blck_f32_q4_0 in ggml/src/ggml-cuda/cpy.cu is where the per-block scale is chosen, and a clip factor there would be a few lines behind an env var. But the motivating case is 2-bit quantisation over a *full row*, where four levels and one 6-sigma outlier means the outlier eats the range. We quantise 32 elements per scale and pre-rotate with a Hadamard, so the dynamic range inside a block is already far smaller. Expected gain here: low, and it is a source patch we would have to carry across every rebase. Not worth it before the two free experiments below (LLAMA_ATTN_ROT_DISABLE A/B, and the MMA-dequant question in technique 11).

### Sparse block→pool-slot table as the quantised/unquantised discriminator
**Where (theirs):** `triton_kvarn_decode.py:85-88` · `triton_kvarn_decode.py:229-231` · `triton_kvarn_decode.py:150-152`

**What it does.** A single int32 array indexed by block id encodes the entire two-tier cache: a non-negative entry means the block's 128 tokens live as rotated fp16 in the tail pool, −1 means they live quantised in the int4 cache. Every kernel branches on this one load.

**Mechanism.** `pool_slot = tl.load(Block_to_slot_ptr + safe_bid, mask=in_range, other=-1)` appears identically in the materialize kernel (triton_kvarn_decode.py:231), the fused decode kernel (triton_kvarn_decode.py:433), stage1 (triton_kvarn_decode.py:583) and the verify kernel (triton_kvarn_decode.py:1123). The comment names the two pool-resident cases: 'sink at k==0; in-progress tail at k==n_full' (triton_kvarn_decode.py:150-151), i.e. the attention-sink block and the block currently being filled.

**Why they needed it.** It makes the two-tier layout invisible to the block table — vLLM's paged block ids are unchanged, and the tier is a side lookup. It also means the flush of a filled 128-token tile from pool to int4 is a single table entry write plus the quantise, with no block-table surgery.

**llama.cpp — absent, has a seam.** Nameable seam (llama-kv-cache.cpp:209-232 for the buffers, the ggml_flash_attn_ext signature and the fattn kernel family for the read), so not impossible; but the effort is a new backend kernel and it would have to be carried across every upstream rebase.

**Equivalent here:** none — one ggml_type per cache, chosen once at context construction

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:231-232 (one k tensor and one v tensor per layer, each a single ggml_type)` · `src/llama-context.cpp:3613-3633 (the cache type is validated once at init and is immutable thereafter)` · `ggml/src/ggml-cuda/fattn.cu:442-446 (the kernel selector reads K->type and V->type — one type per tensor, no per-block tier)`

**Effort:** new-backend · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** This is the shape that would let us go below q4_0 without wrecking recent-token accuracy: bulk history at 2 bits, the attention sink and the in-progress tail at f16. On a 12 GB card where --fit trades KV against context, that is the single biggest structural win in this slice. But the seam — a second cache tensor plus a tier lookup per layer in llama_kv_cache, a new argument to ggml_flash_attn_ext, and a branch inside every fattn kernel — is a fork of ggml's attention op, not a patch. Not a candidate for this project.

### Padded-row early return for uniform-batch graph replay
**Where (theirs):** `triton_kvarn_decode.py:396-399` · `triton_kvarn_decode.py:1083-1085`

**What it does.** Rows added purely to pad a captured batch to a uniform size carry seq_len <= 0 and their programs return before any work, with the guarantee that the corresponding output row is never read.

**Mechanism.** `if seq_len <= 0: return` after the seq_lens load (triton_kvarn_decode.py:398-399, 1084-1085). Note the output buffer is left untouched — correctness rests on the consumer never reading those rows, stated as 'the output row is never read' (triton_kvarn_decode.py:397).

**Why they needed it.** 'Padded rows (uniform-batch graph capture/replay pads the token count) carry seq_len <= 0: nothing to attend'. Interacts with the stage-2 all-empty guard, which handles the same situation on the split-K route where stage 2 does read the row.

**llama.cpp — absent, has a seam.** llama.cpp explicitly forecloses multi-shape graph coexistence (map CANNOT #2) and the fix the vLLM stack used — pad to a constant shape, return early on the padding — is precisely what is missing. The seam exists and the diagnostic to justify the work already ships.

**Equivalent here:** none — llm_graph_params::allow_reuse requires exact ubatch.n_tokens equality, so a varying step size rebuilds the graph instead of padding to a bucket

**Evidence (llama.cpp):** `src/llama-graph.h:781-816 (allow_reuse requires ubatch.n_tokens == other.ubatch.n_tokens, plus equal n_seqs, n_seqs_unq and n_outputs)` · `src/llama-context.cpp:1332-1372 (the reuse gate; n_reused counted at :1348)` · `ggml/src/ggml-cuda/ggml-cuda.cu:4253-4268 (CUDA graph capture needs two consecutive identical calls; any property change resets warmup_complete to eager)` · `tools/server/server-context.cpp:617-619 (the server prints 'graphs reused = %10d' from llama_perf_context(ctx_tgt).n_reused on every completion)`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The premise is missing here and that is the point. With ngram-mod the step size alternates between 65 tokens (a 64-token draft plus the sampled token) and 1 token when the all-or-nothing n_min=48 gate clears the draft, so ubatch.n_tokens keeps changing, the graph is rebuilt, the scheduler re-splits, and CUDA-graph warmup never completes — eager execution for the whole run. Padding the verify ubatch to a fixed bucket and masking the surplus is the fix and the seam is nameable (llama-graph.h:785 plus the ubatch construction), but it is a large patch against upstream. What is free today: `graphs reused` is already printed in the slot timings on every completion and we have almost certainly never read it. If that number is near zero under speculation, this diagnosis is confirmed and it is worth pricing the patch; if it is high, drop the idea.

## partial — 3

### K packed channel-major, V packed token-major — the axis choice that removes a transpose
**Where (theirs):** `kvarn/files/vllm/v1/attention/ops/triton_kvarn_decode.py:273-277` · `kvarn/files/vllm/v1/attention/ops/triton_kvarn_decode.py:290-294` · `kvarn/files/vllm/v1/attention/ops/kvarn_store.py:8-11`

**What it does.** K is stored as [D, group] (channels × tokens, packed along tokens) and V as [group, D] (tokens × channels, packed along channels). K therefore dequantises directly into the [D, BN] layout the score matmul wants, and V directly into the [BN, D] layout the p·V matmul wants — neither needs an in-kernel transpose.

**Mechanism.** K byte address is `K_PACKED_OFFSET + d_offs[:,None]*(GROUP//PACK_K) + g_byte_k[None,:]` giving `[D, GROUP]` after the shift-and-mask (triton_kvarn_decode.py:273-276); V is `V_PACKED_OFFSET + g_offs[:,None]*(D//PACK_V) + d_byte_v[None,:]` giving `[GROUP, D]` (triton_kvarn_decode.py:290-293). In the fused decode loop `K_dg` comes out already `[D, BN]` and is fed straight to `tl.dot(q, K_dg)` (triton_kvarn_decode.py:469, 480), whereas the fp16 pool path — which stores tokens-major — must call `tl.trans(Kc)` (triton_kvarn_decode.py:459).

**Why they needed it.** kvarn_store.py:8-11 names this 'the KIVI K-axis orientation' / 'the KIVI V-axis orientation'. The payoff is visible in the kernel: the quantised path, which is the common case, pays no transpose, and only the pool path (at most two blocks per request) pays one.

**llama.cpp — partial.** The V axis is already switchable and is chosen by FA, not by quantisation quality. The K axis is fixed by the cache tensor shape and is baked into every FA kernel, every seq_rm, and the state serialisation. Changing it is not a patch.

**Equivalent here:** v_trans = !cparams.flash_attn gives V two layouts; K has exactly one

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:231-232 (k and v are ggml_new_tensor_3d(type, n_embd_k_gqa, kv_size, n_stream) — ne0 is the channel axis, so a 32-element q4_0 block runs ACROSS CHANNELS within one token)` · `src/llama-kv-cache.cpp:81 and :208 (v_trans = !flash_attn)` · `src/llama-kv-cache.cpp:319-336 (attn_rot_k — the Hadamard is llama.cpp's answer to per-channel K outliers)`

**Effort:** new-backend · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No action. The finding worth carrying is the diagnosis: llama.cpp quantises K *per token* (blocks of 32 channels), which is precisely the axis KIVI argues is wrong for K, and it compensates with the Hadamard rotation rather than by transposing. That makes LLAMA_ATTN_ROT_DISABLE=1 a meaningful A/B: it is the direct measure of how much our -ctk q4_0 accuracy depends on that compensation.

### Dual-source fused decode — int4 tiles and fp16 tail pool read in one kernel
**Where (theirs):** `triton_kvarn_decode.py:300-306` · `triton_kvarn_decode.py:430-477` · `triton_kvarn_decode.py:728-734`

**What it does.** The default decode path never materialises an fp16 K/V buffer. A single kernel walks a request's block table and, per block, either dequantises the int4 tile in registers or loads already-rotated fp16 rows from the tail pool, feeding both into the same online-softmax accumulation.

**Mechanism.** Per block `k`, `pool_slot = Block_to_slot_ptr[block_id]` decides the branch (triton_kvarn_decode.py:430-433). `pool_slot >= 0` → fp16 loads from `Tail_K_pool_ptr`/`Tail_V_pool_ptr` (triton_kvarn_decode.py:454-459); `pool_slot < 0` → the shift/mask/dequant path (triton_kvarn_decode.py:460-477). Both produce `K_dg [D,BN]` and `Vc [BN,D]` and fall through to the shared `tl.dot` + online softmax (triton_kvarn_decode.py:480-492). Both addresses (`tile_base`, `pool_base`) are computed unconditionally using `tl.where`-clamped safe indices so neither branch can compute an out-of-range pointer (triton_kvarn_decode.py:432-437).

**Why they needed it.** Quoting the header: 'This is the path that can beat FP16/TurboQuant: per step it moves ~int4 (0.25x FP16) KV traffic for the bulk history instead of the materialize path's ~2.25x.' The driver repeats it: the materialize path is '≥2.25x FP16 KV traffic; kept for A/B and as a fallback' (triton_kvarn_decode.py:733-734).

**Their numbers.** fused ≈0.25x FP16 KV traffic for bulk history; materialize ≥2.25x FP16. Selected by KVARN_FUSED_DECODE, default '1' (triton_kvarn_decode.py:736).

**llama.cpp — partial.** llama.cpp has both halves of the idea — a fused quantised read and a materialise fallback — but the selector is a hardcoded query-column threshold, not a per-block tier lookup, and speculation puts us on the wrong side of it by construction. No two-tier cache exists.

**Equivalent here:** BEST_FATTN_KERNEL_VEC reads q4_0 K/V directly (need_f16 = false); every other kernel materialises the whole cache to F16

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-vec.cuh:541-543 (need_f16_K = type_K == F16, need_f16_V = type_V == F16 — with q4_0 both are FALSE, so VEC dequantises nothing)` · `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963 (launch_fattn(..., true, true, true) — MMA passes need_f16_K = need_f16_V = true unconditionally)` · `ggml/src/ggml-cuda/fattn.cu:546-556 (get_alloc_size: TILE and MMA_F16 force need_f16 on both; VEC only when the source is F32)` · `ggml/src/ggml-cuda/fattn.cu:469 (with quantised K/V on Ada, VEC is chosen only if Q->ne[1] <= 2)` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912 (the F16 scratch is charged to the compute buffer)`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** This is the highest-value verdict in the slice and it costs nothing to act on. We have a fused quantised-KV read path already — but it is alive only at one or two query tokens per step. Plain decode gets it. The moment speculation drafts two or more tokens (Q->ne[1] = 1+n_draft >= 3) the kernel flips to MMA_F16 and llama.cpp dequantises the ENTIRE K and V cache for every layer on every verify step. With ngram-mod (n_max 64, n_min 48) every drafted step is Q->ne[1] = 65, so we are permanently on the materialise path. There is no log line for the flip. The measurement that follows: pair -ctk q4_0 -ctv q4_0 against -ctk f16 -ctv f16 within one round under draft-dflash at a fixed depth, and see whether q4_0's smaller cache is being paid back in dequant. Effect size unknown until measured; note the 13.6% noise floor.

### Shared-dequant multi-query verify with bottom-right causal limits
**Where (theirs):** `triton_kvarn_decode.py:1028-1038` · `triton_kvarn_decode.py:1087-1092` · `triton_kvarn_decode.py:1168-1173`

**What it does.** For speculative-decode verification, one program handles a whole request's QLEN candidate tokens at once, so all of them share each block's dequant. Each row of the Q tile carries its own causal cutoff so the triangular mask is applied per row inside the score tile.

**Mechanism.** Q tile is `[M, D]` with `M = QLEN * Q_PER_KV_PAD` (triton_kvarn_decode.py:1087); `j = r // Q_PER_KV_PAD` is the token index and `lane = r % Q_PER_KV_PAD` the head lane (triton_kvarn_decode.py:1089-1090); the per-row limit is `limit = seq_len - QLEN + j + 1` (triton_kvarn_decode.py:1092). Masking: `smask = cmask[None,:] & (kvpos[None,:] < limit[:,None])`, with the sliding window applied relative to each row's own limit (triton_kvarn_decode.py:1169-1172). Grid is (B, Hk, SPLITS) (triton_kvarn_decode.py:952).

**Why they needed it.** 'the per-token VQ_INDIRECT path above re-walks the context once per token, i.e. QLEN redundant dequants' (triton_kvarn_decode.py:1030-1032). The driver states the payoff: in uniform mode 'the request's QLEN tokens SHARE each block's dequant, so KV bytes and dequant ALU match single-token decode' (triton_kvarn_decode.py:886-888).

**Their numbers.** reduces verify dequant work by a factor of QLEN vs the per-token path.

**llama.cpp — partial.** The sharing property is already present; only the mechanism differs. The genuinely new information is that llama.cpp's kernel selector makes the quantised-read path unreachable above two query tokens, and that --spec-draft-n-max is the flag that straddles the threshold.

**Equivalent here:** the MMA_F16 path already shares one K/V expansion across all verify columns — but as a whole-cache F16 materialisation, not an in-register share

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963 (need_f16_K = need_f16_V = true: K and V are expanded once per layer per step and reused by every query column)` · `ggml/src/ggml-cuda/fattn.cu:469 (quantised K/V take VEC only when Q->ne[1] <= 2)` · `ggml/src/ggml-cuda/fattn-vec.cuh:553-572 (cols_per_block is 1 or 2 — there is no wider vec instance)` · `common/arg.cpp:4076-4085 (--spec-draft-n-max, default 3, applies to draft-dflash)` · `common/speculative.cpp:988-996 (the DFlash block-size clamp: with a stock 16-wide sidecar the largest usable n_max is 15)`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** llama.cpp already shares the dequant across drafted columns, so the QLEN-redundancy the vLLM stack fixed does not exist here. What DOES exist is an unused lever with a hard threshold: `--spec-draft-n-max 1` keeps the verify batch at Q->ne[1] = 2 and therefore on the VEC kernel, which reads q4_0 directly and materialises nothing; n_max >= 2 puts every verify step on MMA_F16 with a full-cache F16 expansion per layer. That is a one-flag A/B nobody here has run. Caveat before spending a round on it: shrinking n_max to 1 also throws away most of draft-dflash's measured +34.7%, so the interesting question is not 'is n_max 1 faster' but 'how much of q4_0's VRAM saving is being handed back as dequant at n_max 3-15' — and the honest way to ask that is the -ctk q4_0 vs f16 pairing in technique 11, not this flag.

## already have it — 16

### Head-dim >256 Sinkhorn fallback to batched PyTorch
**Where (theirs):** `kvarn/files/vllm/v1/attention/ops/triton_kvarn_sinkhorn.py:161-174`

**What it does.** If either tile dimension exceeds 256, the Triton kernel is bypassed entirely and the identical algorithm runs through `variance_normalize_batched` in PyTorch, with the outputs reshaped to the same contract.

**Mechanism.** `if max(R, C) > 256:` imports `variance_normalize_batched` and returns `(bal.contiguous(), s_col_b.reshape(N, C).contiguous(), s_row_b.reshape(N, R).contiguous())` (triton_kvarn_sinkhorn.py:167-174).

**Why they needed it.** Quoting: 'At large head_dim that tile is huge (e.g. head_dim 512 -> [512, 128] = 256 KB) and the Triton compiler hangs/explodes (128/256 compile fine).' The cost is judged acceptable because 'Flush is infrequent + off the decode hot path'. This is the register-residency design of the kernel hitting its ceiling, and the fix is a route-around rather than a tiled rewrite.

**Their numbers.** threshold max(R,C) > 256; head_dim 512 -> [512,128] = 256 KB tile; 128 and 256 compile fine.

**llama.cpp — already have it.** llama.cpp already implements exactly this shape: when the fast kernel cannot serve a configuration, route around it to a slower general implementation rather than failing. It does it at a coarser granularity (whole op to CPU backend) and, with -fa auto, tells you.

**Equivalent here:** BEST_FATTN_KERNEL_NONE -> the FLASH_ATTN_EXT node is scheduled on the CPU backend; plus -fa auto's device-mismatch probe

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:392-437 (head-dim table; default: return BEST_FATTN_KERNEL_NONE)` · `src/llama-context.cpp:504-551 (the -fa auto probe: if a fused FLASH_ATTN node lands on a different device from its layer, FA is disabled and logged)` · `src/llama-context.cpp:532-548 ('Flash Attention enabled' / 'Flash Attention not supported, set to disabled')`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Nothing to add — but the failure mode is live and worth one check on our profile. With `-fa on` (not auto) no probe runs, an unsupported combination silently falls to the CPU backend, and you get a working server that is enormously slower with no error (map CANNOT #6, src/llama-context.cpp:554). The startup line at llama-context.cpp:312 prints what you *asked* for, not the resolved state. Read the probe's own line instead. Qwen3-family head dims 128 and 256 are both in the accepted table, so this should not bite, but confirm rather than assume.

### Generic bits-per-byte packing convention shared by packer, unpacker and kernel
**Where (theirs):** `kvarn/files/vllm/v1/attention/ops/kvarn_store.py:85-99` · `kvarn/files/vllm/v1/attention/ops/kvarn_decode.py:36-60` · `kvarn/files/vllm/v1/attention/ops/triton_kvarn_decode.py:253-261`

**What it does.** A single addressing rule — value at index c lives in byte c//PACK at bit shift (c%PACK)*bits, PACK = 8//bits — is implemented three times (PyTorch pack, PyTorch unpack, Triton in-kernel unpack) and stated as the contract in each.

**Mechanism.** Pack: reshape last dim to `[C//pack, pack]` and OR in `q[..., j] << (j*bits)` for j in 1..pack-1 (kvarn_store.py:95-99). Unpack: `out[..., j::pack] = (packed >> (j*bits)) & mask` (kvarn_decode.py:58-59). Kernel: `g_byte_k = g_offs // PACK_K`, `g_shift_k = (g_offs % PACK_K) * K_BITS` (triton_kvarn_decode.py:258-259) with `PACK_K: tl.constexpr = 8 // K_BITS` and `MASK_K = (1 << K_BITS) - 1` (triton_kvarn_decode.py:253-256). `_pack_4bit` is the bits=4 special case: low nibble = even index, high nibble = odd index (kvarn_store.py:73-82).

**Why they needed it.** The docstrings explicitly cross-reference each other — 'matching the decode kernel's unpack (byte=idx//PACK, shift=(idx%PACK)*bits)' (kvarn_store.py:90-91) and 'Mirrors the pack/decode-kernel convention exactly' (kvarn_decode.py:41-43). The convention being written down in all three places is what made the two bit-width bugs below findable.

**llama.cpp — already have it.** llama.cpp does not restate a packing rule in three places; it declares the block layout once and derives every stride from type traits. Same goal, stronger mechanism.

**Equivalent here:** ggml block structs in ggml/src/ggml-common.h consumed by every backend through ggml_blck_size / ggml_type_size, with test-backend-ops as the cross-backend equality gate

**Evidence (llama.cpp):** `ggml/include/ggml.h:390-433 (the 40 live ggml_types; every layout is declared once)` · `src/llama-context.cpp:3613-3633 (the init-time guard that ggml_blck_size(type_k) must divide n_embd_head_k, refusing to build the context otherwise)`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Nothing to change — but the discipline is the same one this project's north star demands, and llama.cpp enforces it structurally rather than by convention: the layout is a struct, the strides come from type traits, and a mismatch is a context-construction failure rather than a plausible wrong number.

### GQA head-grouping so each int4 tile is dequantised once, not Q_PER_KV times
**Where (theirs):** `triton_kvarn_decode.py:371-377` · `triton_kvarn_decode.py:386-391` · `triton_kvarn_decode.py:480`

**What it does.** The kernel grid is (batch, KV head) rather than (batch, query head). One program serves all Q_PER_KV query heads sharing a KV head, so the shift-mask-scale work for a tile happens once and the result is reused across the whole query group via a single matmul.

**Mechanism.** `hk = tl.program_id(1)`; `qh = tl.arange(0, Q_PER_KV_PAD)`; `hq0 = hk * Q_PER_KV` (triton_kvarn_decode.py:387-390). Q is loaded as a `[Q_PER_KV_PAD, D]` tile (triton_kvarn_decode.py:410-411), and scores are `tl.dot(q, K_dg)` producing `[Q_PER_KV, BN]` in one tensor-core op (triton_kvarn_decode.py:480). Grid launched as `[(B, Hk)]` (triton_kvarn_decode.py:793).

**Why they needed it.** Quoting: 'The redundant-dequant penalty of the per-Q-head version scales with Q_PER_KV (4× on Qwen3-4B) and was the dominant cost.' This also converts the decode from a matrix-vector into a small matrix-matrix, which is why tl.dot (tensor cores) is usable at all.

**Their numbers.** Q_PER_KV = 2 (Qwen3-0.6B), 4 (Qwen3-4B), 8 (Qwen3-30B-A3B, 32B) (triton_kvarn_decode.py:313-315); 4× redundant dequant avoided on Qwen3-4B.

**llama.cpp — already have it.** The MMA kernel's effective batch is gqa_ratio × query tokens by construction; the K/V tile is loaded once per group. Same optimisation, arrived at from the tensor layout rather than the grid shape.

**Equivalent here:** gqa_opt / ncols2 packing in the MMA kernel

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:371-390 (gqa_ratio = Q->ne[2]/K->ne[2]; gqa_opt_applies requires gqa_ratio >= 2, a mask, no ALiBi, and K->ne[1] % FATTN_KQ_STRIDE == 0)` · `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962 (launch_fattn<DV, ncols1, ncols2> — the GQA group is the ncols2 dimension)` · `src/llama-kv-cache.cpp:1233-1246 (n_kv padded to 256 = FATTN_KQ_STRIDE, which is what makes gqa_opt legal at all)`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Nothing to change; llama.cpp already amortises the K/V tile across the whole query group. Worth knowing that gqa_opt has a precondition that can silently fail — it needs a mask, max_bias == 0, and n_kv divisible by 256. The 256 padding is unconditional (llama-kv-cache.cpp:1238), so it holds for us.

### fp16 scale vectors loaded through a uint16 pointer cast
**Where (theirs):** `triton_kvarn_decode.py:439-446` · `triton_kvarn_decode.py:588-593` · `triton_kvarn_decode.py:263-271`

**What it does.** The cache tile is a uint8 buffer, but the scale/zero-point fields inside it are fp16 at even offsets. The fused kernels reinterpret the tile pointer as uint16 and load each fp16 value in a single transaction instead of loading two bytes and reassembling.

**Mechanism.** `ku16 = (KV_cache_ptr + tile_base).to(tl.pointer_type(tl.uint16))`, then `tl.load(ku16 + (K_S_COL_OFFSET // 2) + d_offs).to(tl.float16, bitcast=True).to(tl.float32)` (triton_kvarn_decode.py:443-446). The superseded byte-pair form is still visible in the materialize kernel: two uint8 loads combined as `(lo | (hi << 8)).to(tl.float16, bitcast=True)` (triton_kvarn_decode.py:263-265). The shared-dequant verify kernel goes one step further and casts straight to `tl.pointer_type(tl.float16)`, dropping the bitcast (triton_kvarn_decode.py:1129-1134).

**Why they needed it.** Quoting: 'fp16 fields live at even byte offsets in the uint8 tile, so load them as a single uint16 (half the L1 transactions of the lo/hi byte pair)'. The kernel was profiled as 'bottlenecked on L1/TEX transaction rate, not DRAM bandwidth' (triton_kvarn_decode.py:45-47), so halving transactions is the direct lever.

**Their numbers.** half the L1 transactions per fp16 scale.

**llama.cpp — already have it.** An optimisation over an untyped uint8 tile layout. ggml's tiles are typed structs.

**Equivalent here:** block quant structs store the scale as ggml_half and load it as a half; there is no byte-pair reassembly anywhere

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/cpy.cu:2 and :285-295 (dequantize.cuh helpers read block scales directly from the block struct)` · `ggml/include/ggml.h:390-433 (block types are C structs with typed fields, not byte buffers)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None — llama.cpp never had the problem. The scale is a typed struct field, so the compiler emits the single-transaction load without anyone asking.

### Split-K flash-decoding stage 1 + log-sum-exp stage 2
**Where (theirs):** `triton_kvarn_decode.py:499-505` · `triton_kvarn_decode.py:558-561` · `triton_kvarn_decode.py:636-643` · `triton_kvarn_decode.py:646-674`

**What it does.** A third grid dimension partitions each request's KV blocks into contiguous slices. Stage 1 computes a partial attention output plus its log-sum-exp per slice; stage 2 combines the slices with LSE weighting into the final row.

**Mechanism.** `blocks_per_split = ceil(n_blocks / NUM_KV_SPLITS)`, `blk_lo = split * blocks_per_split`, `blk_hi = min(blk_lo + blocks_per_split, n_blocks)` (triton_kvarn_decode.py:559-561). Stage 1 writes `O_s = acc / l_i` and `lse_s = m_i + log(l_i)` to fp32 mid-buffers (triton_kvarn_decode.py:636-643). Stage 2, one program per output row, loads all splits' lse, takes the global max `g`, computes `w = exp(lse - g_safe)`, and outputs `sum(w*O)/sum(w)` (triton_kvarn_decode.py:660-674) with `num_warps=2` (triton_kvarn_decode.py:824).

**Why they needed it.** 'the extra grid dim parallelizes the KV sequence so ragged burst seqlens are load-balanced across SMs... This is what makes the decode kernel competitive with TurboQuant's _tq_decode_stage1 at burst.' Crucially, the combine is exact: 'Split-K is log-sum-exp-combined, so the count never changes the OUTPUT, only occupancy' (triton_kvarn_decode.py:69-70).

**Their numbers.** Qwen3.5-27B head_dim 256 at 16K context: 0.59x -> 0.96x same-batch, 'and lets KVarN out-throughput FP16's max feasible batch' (triton_kvarn_decode.py:768-769).

**llama.cpp — already have it.** llama.cpp has both flavours of KV-sequence splitting with an exact reduction. The vLLM stack's claim that the split count cannot change the output holds here for the same reason: the combine is max-and-sum, not an approximation.

**Equivalent here:** parallel_blocks + flash_attn_combine_results, and the stream-k variant with flash_attn_stream_k_fixup_*

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-common.cuh:916-970 (flash_attn_combine_results — combines partials by running max and running sum, i.e. the LSE combine)` · `ggml/src/ggml-cuda/fattn-common.cuh:723-900 (flash_attn_stream_k_fixup_uniform / _general)` · `ggml/src/ggml-cuda/fattn-common.cuh:1178-1183 (blocks_num.y = parallel_blocks; dst_tmp / dst_tmp_meta allocated only when parallel_blocks > 1)` · `ggml/src/ggml-cuda/fattn-common.cuh:1264-1271 (the combine launch)`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Already ours, automatically, in both the VEC path (parallel_blocks, stream_k=false — fattn-vec.cuh:543) and the MMA path (stream-k, fattn-mma-f16.cuh:1962). No flag, nothing to enable, and nothing to gain by reimplementing.

### Context-adaptive split count with a deployment-constant guarantee
**Where (theirs):** `triton_kvarn_decode.py:60-77` · `triton_kvarn_decode.py:39-40`

**What it does.** The number of KV splits is chosen from max_blocks_per_req — 32 up to 256 blocks, 64 beyond — by a single function used by both the launch site and the partial-buffer sizing, so the two can never disagree.

**Mechanism.** `adaptive_num_kv_splits(max_blocks_per_req)` returns `int(env)` if KVARN_NUM_KV_SPLITS is set, else 32 if `max_blocks_per_req <= 256`, else `KVARN_MAX_KV_SPLITS = 64` (triton_kvarn_decode.py:72-77). Called at the decode launch (triton_kvarn_decode.py:804) and in the verify path (triton_kvarn_decode.py:942, 1001).

**Why they needed it.** Two reasons stated. First, safety: it 'Depends only on the deployment's max_model_len (via max_blocks_per_req = ceil(max_model_len/group)), so it is CONSTANT per deployment -> CUDA-graph safe'. Second, the default of 16 was measured as too low: '16 split under-parallelized the stage-1 (B, Hk, SPLITS) grid at low batch'. And it is a free knob because LSE-combining makes it output-invariant.

**Their numbers.** Single-token decode microbench, Qwen3-4B, ctx 4.6K: 37us at 16 splits vs ~27us at 32, a ~28% stage-1 win. Same bench at 16K ctx: 82us -> 49us. Thresholds: 32 splits up to 256 blocks, 64 above; KVARN_NUM_KV_SPLITS env default 16 (triton_kvarn_decode.py:39).

**llama.cpp — already have it.** llama.cpp derives the split from measured occupancy inputs rather than a context-length threshold table. Stronger than a constant per deployment, and it is not part of any captured-graph key here because the launch geometry follows the (already 256-padded) tensor shapes.

**Equivalent here:** the parallel_blocks search and the stream-k block count in launch_fattn — derived from nsm and the tile counts, per call

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-common.cuh:1003 (nsm = ggml_cuda_info().devices[id].nsm)` · `ggml/src/ggml-cuda/fattn-common.cuh:1115 and :1152-1178 (parallel_blocks starts at max_blocks_per_sm, clamped to ntiles_KV, then raised while blocks_per_wave efficiency improves)` · `ggml/src/ggml-cuda/fattn-common.cuh:1133-1145 (nblocks_stream_k with a rounding/efficiency-loss test)`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** No env var to tune — and that cuts both ways. llama.cpp recomputes the split from the actual tile counts and SM count on every call, so it adapts to context depth automatically and cannot be mistuned, but it also cannot be swept the way KVARN_NUM_KV_SPLITS was. There is no measurement to run and no knob to set.

### Split-K enabled only in the long-context, under-occupied regime
**Where (theirs):** `triton_kvarn_decode.py:758-789`

**What it does.** Split-K is off by default and auto-enabled only when three conditions hold: the layer is not sliding-window, context is at least 16 blocks, and the single-stage grid (B × Hk programs) does not already fill the SMs. A fourth defensive check requires the pre-sized mid buffers to fit the batch.

**Mechanism.** `split_k = (use_fused and (_sw <= 0) and (max_blocks_per_req >= 16) and (B * Hk <= sm_count) and _mid_fits)` (triton_kvarn_decode.py:788-789), with `sm_count` from `impl._sm_count` or `torch.cuda.get_device_properties(device).multi_processor_count` (triton_kvarn_decode.py:781-782) and `_mid_fits = impl._mid_o_buf is not None and N <= impl._mid_o_buf.shape[0]` (triton_kvarn_decode.py:776). KVARN_SPLIT_K env overrides (triton_kvarn_decode.py:777-779).

**Why they needed it.** Quoting: 'At BURST (high batch) the single-stage (B,Hk) grid already saturates the GPU, so split-K's mid-buffer round-trip + stage-2 + empty-split waste roughly HALVE throughput.' The sliding-window exclusion has its own reason: 'Sliding-window layers read only ~window/GROUP blocks (single-stage is plenty + the windowed loop is in the single-stage kernel), so never split.' The _mid_fits guard is 'defensive — real decode batches always fit, but a padded dummy run can be wider'.

**Their numbers.** split-K roughly HALVES throughput at short context / high occupancy; threshold is max_blocks_per_req >= 16 (≈2048 tokens at GROUP=128) and B*Hk <= SM count.

**llama.cpp — already have it.** Same guard, expressed as an occupancy search instead of a boolean predicate. The sliding-window exclusion has no analogue because llama.cpp shrinks the SWA cache itself rather than truncating a loop.

**Equivalent here:** the use_stream_k / efficiency heuristic and the parallel_blocks <= ntiles_KV clamp

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-common.cuh:1126 (use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75)` · `ggml/src/ggml-cuda/fattn-common.cuh:1153 (parallel_blocks = min(parallel_blocks, ntiles_KV) — a short cache cannot over-split)` · `ggml/src/ggml-cuda/fattn-common.cuh:1156-1178 (the blocks_per_wave test; splitting stops once a wave is full)`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** None to act on. Note that on our GPU (cc 890 = Ada) use_stream_k is unconditionally true for the stream-k-capable kernels, so the MMA path always takes the stream-k decomposition regardless of context. The occupancy guard the vLLM stack had to add by hand is the blocks_per_wave loop here, and the empty-split waste it feared is prevented by the ntiles_KV clamp rather than by a threshold.

### Online-softmax NaN guard for the all-masked chunk
**Where (theirs):** `triton_kvarn_decode.py:483-492` · `triton_kvarn_decode.py:1175-1186`

**What it does.** When no key has yet been seen, the running max and the new max are both −inf and exp(−inf − −inf) is NaN. The kernels detect that state and force both the probability tile and the rescale factor to exactly zero.

**Mechanism.** `m_dead = m_new == -float('inf')`; `p = tl.where(m_dead[:,None], 0.0, tl.exp(scores - m_new[:,None]))`; `alpha = tl.where(m_dead, 0.0, tl.exp(m_i - m_new))` (triton_kvarn_decode.py:486-489), replicated in stage1 (triton_kvarn_decode.py:629-631) and the verify kernel (triton_kvarn_decode.py:1181-1183).

**Why they needed it.** Three concrete triggers are named: 'fully masked chunk(s) first, e.g. sliding-window start / an empty split prefix' (triton_kvarn_decode.py:484-485), and in the verify kernel 'a Q row whose bottom-right causal limit ends before this split's first block never sees a live key... -> NaN partial (the nonempty test below cannot catch NaN) -> NaN output after stage2' (triton_kvarn_decode.py:1176-1180). The last clause is the important one: the downstream `l_i > 0` emptiness test is not NaN-safe, so the guard has to be at the source.

**llama.cpp — already have it.** The guard exists in llama.cpp as a bound rather than a post-hoc where(). Same correctness property, different mechanism.

**Equivalent here:** the k_VKQ_sup out-of-bounds supremum and the -INF mask handling inside the FA kernels

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-mma-f16.cuh:558 and :716-742 (k_VKQ_sup bounds every KQ accumulation; oob_check gates each element)` · `ggml/src/ggml-cuda/fattn-common.cuh:1153 (parallel_blocks <= ntiles_KV, so an empty split is never launched)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Nothing to do. llama.cpp prevents the all-masked state structurally — it never launches a split with no keys, and out-of-range columns are excluded by an explicit supremum rather than being masked to -inf and then rescued. If we ever see a NaN in attention output this is where to look, but there is no knob and no patch on the table.

### Stage-2 all-empty-row guard
**Where (theirs):** `triton_kvarn_decode.py:661-673`

**What it does.** If every split of an output row is empty (all lse = −inf), the LSE combine would compute (−inf) − (−inf) = NaN and then 0/0. The kernel substitutes 0 for the global max in that case and floors the denominator, producing a zero output row instead.

**Mechanism.** `g_safe = tl.where(g == -float('inf'), 0.0, g)`; `w = tl.exp(lse - g_safe)`; `out = tl.sum(w[:,None]*O, axis=0) / tl.where(denom > 0, denom, 1.0)` (triton_kvarn_decode.py:668-673).

**Why they needed it.** Quoting: 'e.g. a padded / zero-length row under CUDA-graph capture or a fully-masked sliding window... with it w == 0, denom == 0 -> out == 0. Non-empty rows: g is finite, g_safe == g and denom > 0, so the arithmetic is bit-identical to before.' The bit-identity claim is the point — the guard is proven free on the normal path.

**llama.cpp — already have it.** The precondition for the bug (padded rows and possibly-empty splits) does not arise in this tree.

**Equivalent here:** flash_attn_combine_results over parallel_blocks that are guaranteed non-empty

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-common.cuh:1152-1153 ('parallel_blocks must not be larger than what the tensor size allows'; parallel_blocks = min(parallel_blocks, ntiles_KV))` · `ggml/src/ggml-cuda/fattn-common.cuh:949-968 (the combine walks exactly parallel_blocks partials)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None. llama.cpp does not pad batches for graph capture, so there are no zero-length rows for a combine to choke on, and the split count is clamped so no split is ever empty.

### Scatter-store kernel replacing a Python loop to make the store graph-capturable
**Where (theirs):** `triton_kvarn_decode.py:80-89` · `triton_kvarn_decode.py:92-141`

**What it does.** Writing newly-computed rotated fp16 K/V into the tail pool is done by a Triton kernel indexed off slot_mapping, rather than a host-side loop, so the entire store path can live inside a captured CUDA graph.

**Mechanism.** Grid (N, Hk); per program `sm = Slot_mapping_ptr[i]`, early-return if `sm < 0` (padding), `block_id = sm // GROUP`, `pos = sm % GROUP`, `pool_slot = Block_to_slot_ptr[block_id]`, early-return if `pool_slot < 0` (triton_kvarn_decode.py:118-129). Destination is `pool_slot*stride_pool_b + pos*stride_pool_t + hk*stride_pool_h + d` (triton_kvarn_decode.py:136-139). The bounds test `(block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)` precedes the lookup (triton_kvarn_decode.py:124).

**Why they needed it.** 'Replaces the Python for-loop in do_kv_cache_update so the whole store path is capturable.' The lookup table's mutation discipline is stated as the invariant that makes this safe: 'The lookup table is mutated only by the metadata builder (outside any captured region)' (triton_kvarn_decode.py:87-88).

**llama.cpp — already have it.** llama.cpp's KV write has always been an op in the compute graph, not host code.

**Equivalent here:** the KV write is a ggml_cpy into the cache view, driven by the k_idxs / v_idxs input tensors — already inside the graph

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:1405-1420 (build_input_k_idxs / build_input_v_idxs create graph input tensors for the destination indices)` · `src/llama-graph.cpp:479-484 (set_input_k_rot / set_input_v_rot and the idx inputs are filled once per ubatch, outside the graph)` · `ggml/src/ggml-cuda/cpy.cu:273-283 (the quantising copy kernel that performs the store)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Nothing to change. llama.cpp never had a host-side store loop; the index vector is a graph input filled per ubatch and the store is a device op, which is the same separation the technique achieves (mutation of the plan happens outside any captured region).

### Fixed-geometry materialize grid with per-program token clamping
**Where (theirs):** `triton_kvarn_decode.py:144-153` · `triton_kvarn_decode.py:206-220`

**What it does.** The build-packed-KV kernel launches with a grid of exactly (B × MAX_BLOCKS_PER_REQ, Hk) regardless of actual sequence lengths, and each program computes its own token count, returning immediately if it falls past the end of its request.

**Mechanism.** `b = bk // MAX_BLOCKS_PER_REQ`, `k = bk % MAX_BLOCKS_PER_REQ` (triton_kvarn_decode.py:208-209); `rem = seq_len - k*GROUP`, `n_tok = min(max(rem,0), GROUP)`, `if n_tok <= 0: return` (triton_kvarn_decode.py:214-216). Destination offset comes from the prefix sum: `dst_base = Cu_seqlens_ptr[b] + k*GROUP` (triton_kvarn_decode.py:220).

**Why they needed it.** 'Reads vLLM's persistent block_table + seq_lens directly (so a captured CUDA graph sees fresh data each replay), and writes the packed varlen fp16 K/V... Fixed grid (B * MAX_BLOCKS_PER_REQ, Hk) so the launch dims are constant per captured batch size.' A data-dependent grid cannot be captured; a data-dependent early return can.

**llama.cpp — already have it.** Identical pattern, arrived at independently and documented in-source at llama-kv-cache.cpp:1238.

**Equivalent here:** n_kv padded to 256 plus the k_VKQ_sup per-program bound; the grid follows tensor shapes that are constant by construction

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:1233-1246 (n_kv padded to FATTN_KQ_STRIDE = 256 explicitly to keep the graph constant)` · `ggml/src/ggml-cuda/fattn-mma-f16.cuh:558, :716-742 (k_VKQ_sup clamps each program to the real token count)` · `ggml/src/ggml-cuda/fattn-common.cuh:1115-1183 (blocks_num derived from tile counts, not from ragged sequence lengths)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Nothing to change. llama.cpp already pads the shape and clamps inside the kernel for the same reason — a data-dependent grid is not capturable, a data-dependent bound is.

### Bit-width-aware V stride — two separate shipped bugs from a hardcoded 4-bit assumption
**Where (theirs):** `triton_kvarn_decode.py:247-256` · `triton_kvarn_decode.py:290-291` · `triton_kvarn_decode.py:615-620`

**What it does.** The V packed-row stride must be D/PACK_V bytes where PACK_V = 8/V_BITS. Two kernels had it hardcoded as D//2, correct only for 4-bit V, and both broke under the shipped 2-bit-V preset.

**Mechanism.** Correct form: `v_addrs = tile_base + V_PACKED_OFFSET + cols[:,None]*(D // PACK_V) + d_byte_v[None,:]` (triton_kvarn_decode.py:620, 290-291, 473-474, 1162-1163).

**Why they needed it.** Two different symptoms are recorded for the same root cause. In the materialize kernel: 'The old hardcoded 4-bit V layout (stride D//2, shift (d%2)*4, mask 0xF) read past the 2-bit-V packed region of the default k4v2 preset into the V scales -> garbage V on this path' (triton_kvarn_decode.py:248-252). In stage1: 'with the shipped k4v2 preset (V_BITS=2 -> PACK_V=4) it strode 2x too far -> read garbage V + indexed past the tile (OOB illegal-access at long ctx)' (triton_kvarn_decode.py:616-618). One produced plausible wrong output, the other crashed — the same bug, in two kernels, with two failure modes.

**Their numbers.** default preset is k4v2: K_BITS=4, V_BITS=2, so PACK_V=4 and the wrong stride was exactly 2x too far.

**llama.cpp — already have it.** llama.cpp derives every stride from type traits and validates the one shape assumption at init. The bug shape is structurally unavailable.

**Equivalent here:** every stride derives from ggml_blck_size / ggml_type_size, plus an init-time divisibility guard and test-backend-ops

**Evidence (llama.cpp):** `src/llama-context.cpp:3613-3622 (K) and :3624-3633 (V): the context refuses to build with 'K cache type %s with block size %u does not divide n_embd_head_k=%u'` · `ggml/src/ggml-cuda/cpy.cu:273-294 (the copy kernels are templated on QK4_0 rather than a literal)` · `ggml/include/ggml.h:390-433 (block layouts declared once and shared by every backend)`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** No change — but this is the technique whose *lesson* lands hardest here. The same wrong stride produced plausible garbage in one kernel and an out-of-bounds crash in another: one of those would have been published as a number. llama.cpp forecloses the class by refusing to construct the context when the block size does not divide the head dim, which is exactly the 'crash rather than return a believable number' posture this project's north star asks for. Cite it as prior art, not as work.

### Shared-verify shipped DEFAULT OFF with the failure documented but unexplained
**Where (theirs):** `triton_kvarn_decode.py:928-938`

**What it does.** The shared-dequant verify path is gated behind KVARN_SHARED_VERIFY defaulting to '0', despite having passed numerical validation, because enabling it in real serving corrupts the speculative drafter through a mechanism the authors could not isolate.

**Mechanism.** Guard conditions: `qlen >= 2 and seq_lens is not None and NQ % qlen == 0 and (_m & (_m-1)) == 0` (Q-tile rows must be a power of two, triton_kvarn_decode.py:927-929) AND `os.environ.get('KVARN_SHARED_VERIFY', '0') == '1'` (triton_kvarn_decode.py:938).

**Why they needed it.** Quoted in full because it is the most transferable note in the file: 'numerically validated in isolation (matches the per-token kernel within fp32 reduction noise on live inputs, incl. on the failing trajectory), but serving with it corrupts the MTP drafter's proposals (invalid [-1,...] spec tokens, embedding index asserts at temperature>0, degenerate greedy output) through a mechanism not yet isolated — suspicion is an interaction with async scheduling / drafter metadata rather than kernel math. Re-enable for debugging only.' A kernel that is provably correct in isolation and still wrong in the system.

**llama.cpp — already have it.** llama.cpp uses exactly this pattern in four places. Nothing to add; something to cite.

**Equivalent here:** the env kill-switch family (LLAMA_ATTN_ROT_DISABLE, LLAMA_GRAPH_REUSE_DISABLE, GGML_CUDA_DISABLE_GRAPHS) and the #if 0'd per-request speculative parameters

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:313-317 (LLAMA_ATTN_ROT_DISABLE, with a WARN when it fires)` · `src/llama-context.cpp:279-285 (LLAMA_GRAPH_REUSE_DISABLE, logs 'graph reuse disabled')` · `ggml/src/ggml-cuda/common.cuh:1255-1259 (GGML_CUDA_DISABLE_GRAPHS — existence checked, value not parsed)` · `tools/server/server-schema.cpp:197-198 (the whole per-request speculative parameter block inside #if 0 with 'to keep things simple, we disable speculative parameter adjustments for now')`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** No code change — but this is the most transferable *practice* in the slice for this repo, and llama.cpp already models it: ship the path, default it off, and say in-source why. It is the same posture as CORRECTIONS.md. The concrete carry-over is the observation that a kernel can be numerically exact in isolation and still wrong in the system; in our terms that is 'a delegated run reported complete while its file sat in the wrong directory' and the answer is the same — verification happens in the running system, not in the unit.

### Shared fp16 Hadamard for Q rotation and output un-rotation
**Where (theirs):** `triton_kvarn_decode.py:719-726` · `triton_kvarn_decode.py:861-865` · `kvarn/files/vllm/v1/attention/ops/kvarn_decode.py:6-8`

**What it does.** The query is rotated by the same fp16 Hadamard matrix the K/V store used, via one tensor-core GEMM into a persistent buffer; attention runs entirely in the rotated frame; the output is un-rotated by a second GEMM with the same matrix, exploiting that H is its own inverse.

**Mechanism.** `H16 = impl._H_fp16 if impl._H_fp16 is not None else hadamard.to(torch.float16)`; `torch.mm(query.reshape(N,D), H16, out=q_rot_fp16)` into `impl._q_rot_fp16_buf[:N]` (triton_kvarn_decode.py:723-726). Un-rotation: `out_unrot = torch.mm(output_rot.reshape(N,D), H16)` (triton_kvarn_decode.py:864). The reference file notes 'the inverse Hadamard (matmul with H, which is its own inverse)' (kvarn_decode.py:7-8).

**Why they needed it.** Quoting: 'Use the SAME fp16 Hadamard the K/V store used (_H_fp16) so QKᵀ stays invariant; the old fp32 path added two [N,D] copies + a slower fp32 GEMM per layer for no accuracy benefit (H is orthonormal, well-conditioned).' The invariance argument is the substantive one — if store and decode used numerically different H, q·kᵀ would not be preserved.

**Their numbers.** the fp32 path cost two extra [N,D] copies plus a slower GEMM per layer.

**llama.cpp — already have it.** Every element of the technique is present: one shared matrix for Q and K so QK^T is invariant, a rotation of V with a matching un-rotation of the output, and self-inverse H so the same matrix serves both directions. It is on by default for our exact profile (quantised KV) and neither the ledger nor the register records it.

**Equivalent here:** attn_rot_k / attn_rot_v — automatic whenever the cache type is quantised and the head dim is a multiple of 64; kill switch LLAMA_ATTN_ROT_DISABLE=1

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:20-21 ('orthonormal Walsh-Hadamard rotation matrix // note: res^2 == I' — the same self-inverse property the technique relies on)` · `src/llama-kv-cache.cpp:319-336 (attn_rot_k / attn_rot_v conditions: quantised type and head dim % 64 == 0)` · `src/llama-graph.cpp:3018-3021 (q_cur AND k_cur rotated by the SAME matrix — QK^T invariance)` · `src/llama-graph.cpp:3024-3026 (v_cur rotated) and :3068-3069 (the attention output un-rotated by the same matrix)` · `src/llama-kv-cache.cpp:313-317 (LLAMA_ATTN_ROT_DISABLE=1, WARN 'attention rotation force disabled')` · `src/llama-kv-cache.cpp:338-339 (startup logs attn_rot_k = %d / attn_rot_v = %d)` · `src/llama-kv-cache.cpp:1418-1436 (K uses the largest power-of-two rotation dividing n_embd_head_k; V is deliberately kept at 64, with the PR comment explaining why)`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** We are already running this and did not know it. Two things follow. First, confirm from our own startup log that attn_rot_k = 1 and attn_rot_v = 1 — it silently does not engage if the head dim is not a multiple of 64, and -ctk q4_0 then degrades more on this model than the same flag does elsewhere. Second, LLAMA_ATTN_ROT_DISABLE=1 is an exists-but-unused one-env-var A/B that measures both halves at once: how much of our q4_0 KV quality depends on the rotation, and what the two extra GEMM-shaped ops per layer cost in tok/s. Pair it within a round; effects under 13.6% are noise.

### Rotation applied outside the quantiser, as a cuBLAS GEMM
**Where (theirs):** `kvarn/files/vllm/v1/attention/ops/kvarn_store.py:8-12` · `kvarn/files/vllm/v1/attention/ops/kvarn_store.py:115-117`

**What it does.** The store functions take already-rotated tiles; the Hadamard is a separate external GEMM, not fused into the quantise kernel.

**Mechanism.** 'Inputs to the K path are tile-shaped [D, group] ... after Hadamard rotation'; 'Caller is responsible for the external (K @ H).T GEMM' (kvarn_store.py:117) and 'the external V @ H GEMM' (kvarn_store.py:247). Note the K path needs the transpose, V does not — matching the two storage orientations.

**Why they needed it.** 'The Hadamard rotation is applied externally via a cuBLAS GEMM, identically to TurboQuant's MSE path' (kvarn_store.py:11-12). Keeping it as a library GEMM means it runs at tensor-core peak instead of inside a hand-written kernel, and makes the store path directly comparable to the baseline it is measured against.

**llama.cpp — already have it.** The technique keeps the rotation as a library GEMM to get tensor-core peak. llama.cpp keeps it as a graph op for the same portability reason and then does strictly better on CUDA via the hint.

**Equivalent here:** llama_mul_mat_hadamard emits ggml_mul_mat with GGML_HINT_SRC0_IS_HADAMARD, which ggml-cuda routes to a dedicated FWHT kernel

**Evidence (llama.cpp):** `src/llama-impl.h:57-75 (llama_mul_mat_hadamard: reshape to 2d, ggml_mul_mat(rot, cur), ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD), reshape back)` · `ggml/src/ggml-cuda/ggml-cuda.cu:1816 (if hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(...) succeeds, take that path)` · `ggml/src/ggml-cuda/fwht.cu:61 (ggml_cuda_op_fwht — the fast Walsh-Hadamard transform kernel)` · `ggml/include/ggml.h:445 (GGML_HINT_SRC0_IS_HADAMARD)`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Nothing to change, and llama.cpp is ahead here: the rotation is expressed as an ordinary ggml_mul_mat (so it falls back to the tuned matmul path on any backend) but carries a hint that lets CUDA substitute an O(n log n) FWHT for the O(n²) GEMM. Whatever cost the rotation has for us, it is not a dense 128×128 GEMM per layer. Worth knowing before attributing any q4_0 overhead to it.

### Per-batch task plan reused across all attention layers
**Where (theirs):** `triton_kvarn_decode.py:23-25` · `triton_kvarn_decode.py:704-707`

**What it does.** The block table, sequence lengths, cu_seqlens, block→slot lookup and dequant/pool block partitioning are computed once per batch in the metadata builder and passed to every layer's forward, so no per-layer host→device allocation occurs.

**Mechanism.** 'The task plan (block table, seq lens, block->pool-slot lookup) is built once per batch in KVarNMetadataBuilder.build and reused across all attention layer forwards in a step' (triton_kvarn_decode.py:23-25). Restated in the driver: 'precomputed once per batch ... and passed in via md — no per-layer host→GPU allocations' (triton_kvarn_decode.py:704-707).

**Why they needed it.** A 27B model has dozens of attention layers per step; any per-layer host work is multiplied by that count and lands directly on decode latency.

**llama.cpp — already have it.** The per-layer host-work multiplication the technique avoids never occurs here; graph inputs are set once per ubatch by construction.

**Equivalent here:** the slot_info / ubatch, kq_mask, k_idxs / v_idxs and rotation matrices are graph inputs set once per ubatch and read by every layer

**Evidence (llama.cpp):** `src/llama-graph.cpp:479-484 and :2748-2749 (the rotation and index inputs are created once per graph and filled once per ubatch)` · `src/llama-kv-cache.cpp:1405-1436 (build_input_k_idxs / build_input_k_rot / build_input_v_rot — one input tensor each, not per layer)` · `src/llama-context.cpp:1332-1372 (graph reuse: when the ubatch shape matches, even the graph itself is not rebuilt)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Nothing to change. llama.cpp goes further than reusing the plan across layers — when the ubatch shape is unchanged it reuses the entire built graph. Which is exactly why technique 34's shape churn matters.

## impossible here — 1

### Autotune space that trades pipelining for occupancy, including maxnreg caps
**Where (theirs):** `triton_kvarn_decode.py:42-57` · `triton_kvarn_decode.py:322-325`

**What it does.** All three decode kernels share one autotune config list spanning BLOCK_N ∈ {16,32,64} × num_warps ∈ {2,4} × num_stages ∈ {1,2}, plus two entries that cap register allocation at 64 and 96 registers per thread to raise the number of resident blocks.

**Mechanism.** `_DECODE_AUTOTUNE_CONFIGS` is a comprehension over (bn, nw, ns) plus `triton.Config({'BLOCK_N': 32}, num_warps=4, num_stages=2, maxnreg=mr) for mr in (64, 96)` (triton_kvarn_decode.py:51-57). Applied via `@triton.autotune(configs=..., key=['D','GROUP','Q_PER_KV','K_BITS','V_BITS'])` on the single-stage kernel (triton_kvarn_decode.py:322-325), on stage1 (triton_kvarn_decode.py:515-518), and with QLEN added to the key on the verify kernel (triton_kvarn_decode.py:1041-1044).

**Why they needed it.** The reasoning is profiler-driven and quoted in full: 'ncu on the burst single-stage kernel showed it pinned at ~25% occupancy (register-limited to 3 blocks/SM) and bottlenecked on L1/TEX transaction rate, not DRAM bandwidth. So beyond BLOCK_N x num_warps we let the autotuner trade pipelining for occupancy: num_stages=1 (no pipeline buffers, fewer registers) and a couple of maxnreg caps.' It is called 'pure upside' because 'online-softmax / split-K make the output reduction-order invariant (fp noise only)'. Separately: 'num_warps=8 was empirically slower at Q_PER_KV=8 (tl.dot with small M under-utilises threads)' and 'num_stages=3 vs 2 showed no difference in the micro-bench' (triton_kvarn_decode.py:315-318).

**Their numbers.** ~25% occupancy, register-limited to 3 blocks/SM (ncu, burst single-stage kernel). Stage 1's previous hardcoded BLOCK_N=16/num_warps=4 was beaten by BLOCK_N=32/num_warps=2 by ~25-40% across 4.6K-32K context on Qwen3-4B (triton_kvarn_decode.py:508-512).

**llama.cpp — impossible here.** Adding a search would mean adding a JIT or a config-selection cache to ggml-cuda, which is a change to the project's architecture rather than a patch. I cannot name a seam that would not amount to that.

**Equivalent here:** none at runtime; kernel configs are compile-time template instantiations selected by ggml_cuda_get_best_fattn_kernel

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:358-533 (kernel choice is a decision tree over cc, head dims, K/V types, gqa_ratio and Q->ne[1] — no search, no cache)` · `ggml/src/ggml-cuda/fattn-vec.cuh:535-543 (nthreads fixed per compute capability)` · `ggml/src/ggml-cuda/fattn-common.cuh:1115 (max_blocks_per_sm is measured, but only to size the split — never to re-pick a config)`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No runtime autotuner exists and ggml deliberately does not JIT, so there is nothing to enable. The one occupancy-adjacent knob we have and have never set is GGML_CUDA_GRAPH_OPT=1 (ggml/src/ggml-cuda/ggml-cuda.cu:4318-4344), which interleaves the Q/K/V branches across extra streams; it requires CUDA graphs on and exactly one CUDA device, both satisfied here, and it only fires on single-row decode-shaped nodes. That is a one-env-var experiment, not an autotuner.

## not applicable — 10

### Sinkhorn stability clamps with asymmetric log bounds
**Where (theirs):** `kvarn/files/vllm/v1/attention/ops/triton_kvarn_sinkhorn.py:21-24` · `kvarn/files/vllm/v1/attention/ops/triton_kvarn_sinkhorn.py:93-95`

**What it does.** Two independent clamps guard the iteration: the measured per-axis std is clipped to [1e-3, 1e3] before its log is taken, and the accumulated log-scale itself is clipped to [-0.3, 10.0]. The log bound is deliberately asymmetric — scales may grow by e^10 but may only shrink by e^-0.3.

**Mechanism.** `_CLIP_STD_MIN=1e-3`, `_CLIP_STD_MAX=1e3`, `_LOG_S_MIN=-0.3`, `_LOG_S_MAX=10.0` are module constants passed as constexprs (triton_kvarn_sinkhorn.py:21-24, 188-191). Applied as `col_std_clipped = max(min(col_std, CLIP_STD_MAX), CLIP_STD_MIN)` then `log_s_col = max(min(log_s_col + log(...), LOG_S_MAX), LOG_S_MIN)` (triton_kvarn_sinkhorn.py:93-95), and identically for rows (triton_kvarn_sinkhorn.py:104-106).

**Why they needed it.** The file gives no prose reason for the specific values, but the mechanism is clear from the maths: dividing by a scale below e^-0.3 ≈ 0.74 would AMPLIFY a channel rather than balance it, and the std clip prevents log(0) on a dead channel. Recorded here because the asymmetry is unusual and is the kind of constant a reimplementation would silently symmetrise.

**Their numbers.** CLIP_STD ∈ [1e-3, 1e3]; log_s ∈ [-0.3, 10.0].

**llama.cpp — not applicable.** Clamps on a state variable that does not exist in this tree. No seam.

**Equivalent here:** none

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/cpy.cu:273-283 (per-block absmax; no accumulated log-scale to clamp)`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** None. There is no accumulated scale in llama.cpp's KV quantiser to bound. The only carry-over is the general warning that a reimplementation silently symmetrises an asymmetric constant — which is a code-review note, not a change we can make here.

### Sinkhorn variance computed as E[x²]−E[x]² with Bessel correction
**Where (theirs):** `kvarn/files/vllm/v1/attention/ops/triton_kvarn_sinkhorn.py:71-76` · `kvarn/files/vllm/v1/attention/ops/triton_kvarn_sinkhorn.py:112-117`

**What it does.** Per-axis standard deviation is computed in one pass as sum of squares minus square of mean, then rescaled by R/(R-1) (or C/(C-1)) and floored at zero before the sqrt.

**Mechanism.** `col_var = tl.sum(cur*cur, axis=0)/R - col_mean*col_mean`, then `col_std = tl.sqrt(tl.maximum(col_var * R/(R-1), 0.0))` (triton_kvarn_sinkhorn.py:71-73). Same pattern for rows with C (triton_kvarn_sinkhorn.py:74-76) and repeated at the imbalance-measurement point (triton_kvarn_sinkhorn.py:112-117).

**Why they needed it.** One-pass variance avoids a second reduction over a tile held entirely in registers. The `tl.maximum(..., 0.0)` is the necessary guard: the E[x²]−E[x]² form is numerically capable of producing a small negative variance in fp32, and sqrt of that is NaN which would propagate into every scale. The R/(R-1) factor exists to match the PyTorch reference's sample-std convention exactly, since the two implementations must agree.

**llama.cpp — not applicable.** Presupposes machinery absent from this tree.

**Equivalent here:** none

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/cpy.cu:273-283 (q4_0 KV write is absmax-based; no variance is ever computed)`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** None. llama.cpp's KV quantiser never computes a variance, so neither the one-pass form nor the negative-variance guard has anywhere to land.

### Whole-tile-in-registers Sinkhorn with num_warps=8 to avoid local-memory reservation
**Where (theirs):** `kvarn/files/vllm/v1/attention/ops/triton_kvarn_sinkhorn.py:192-203` · `kvarn/files/vllm/v1/attention/ops/triton_kvarn_sinkhorn.py:11-13`

**What it does.** One Triton program holds an entire [R, C] fp32 tile plus a working copy in registers for all 16 iterations. The launch is forced to 8 warps rather than the default 4, purely to halve per-thread register footprint and stop the compiler spilling to CUDA local memory.

**Mechanism.** Grid is `(N,)` — one program per tile (triton_kvarn_sinkhorn.py:180). `tile` is loaded once (triton_kvarn_sinkhorn.py:61) and `cur` is recomputed from it each half-iteration, so two [R,C] fp32 arrays are live. `num_warps=8, num_stages=2` at the launch site (triton_kvarn_sinkhorn.py:202-203).

**Why they needed it.** Quoting the comment: at 4 warps 'the per-thread footprint is several KB of registers -> the compiler spills to CUDA local memory, and the driver permanently reserves local_bytes x max_threads x num_SMs of device memory for the context'. This was diagnosed as 'issue #10's missing-KV-capacity component' — i.e. the spill was eating the VRAM budget the KV cache needed.

**Their numbers.** ~2 GiB reserved local memory on a 188-SM part for the [256, 128] tile at 4 warps; 8 warps gives ~70% less reserved local memory AND ~4x faster flush (the spills were also the bottleneck); balanced-tile output unchanged within fp32 reduction noise ~5e-7 relative; 16 warps saves a bit more memory but is 2x slower than 8. For R=C=128 the tile is 64 KB fp32 (triton_kvarn_sinkhorn.py:11-12).

**llama.cpp — not applicable.** llama.cpp's CUDA kernels are hand-tuned templates with no runtime launch-config surface. There is no flag, env var or config key that changes warp count.

**Equivalent here:** thread counts are compile-time: ggml_cuda_fattn_vec_get_nthreads_host(cc) and the nwarps passed to launch_fattn; occupancy is measured, not guessed

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-vec.cuh:535-543 (nthreads chosen per compute capability, then nwarps = nthreads/WARP_SIZE, passed to launch_fattn)` · `ggml/src/ggml-cuda/fattn-common.cuh:1003 (nsm read from ggml_cuda_info) and :1115 (max_blocks_per_sm drives the split decision)`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No knob to turn. The transferable half is the diagnosis itself — register spill makes the driver reserve local_bytes × max_threads × num_SMs of device memory, which on a 12 GB card is VRAM that vanishes without appearing in any allocation. The llama.cpp analogue of that hidden line item is real and we do have it: quantised KV forces an F16 scratch charged to the compute buffer (ggml/src/ggml-cuda/fattn.cu:536-568, ggml-cuda.cu:906-912).

### Power-of-two padding of the GQA ratio with lane masking
**Where (theirs):** `triton_kvarn_decode.py:377-380` · `triton_kvarn_decode.py:410-411` · `triton_kvarn_decode.py:741-743`

**What it does.** Because tl.arange and tl.dot require power-of-two dimensions, a non-power-of-two GQA ratio is rounded up and the surplus query-head lanes are masked to zero on load and masked out on store.

**Mechanism.** Host side: `_qpk_pad = 1 << (_qpk - 1).bit_length() if _qpk > 1 else 1` (triton_kvarn_decode.py:743). Kernel side: `qmask = qh < Q_PER_KV` (triton_kvarn_decode.py:389); Q load uses `mask=qmask[:,None], other=0.0` (triton_kvarn_decode.py:410-411); the store uses the same mask (triton_kvarn_decode.py:495-496).

**Why they needed it.** Quoting: 'Q_PER_KV is padded to a power of 2 (Q_PER_KV_PAD) because tl.arange / tl.dot require pow2 dims; padded query heads are masked off (e.g. Qwen3.5 GQA 24/4 = ratio 6 -> pad to 8).' Masking Q to zero rather than leaving garbage matters because the padded lanes still participate in the tl.dot and their scores would otherwise be arbitrary.

**Their numbers.** Qwen3.5 GQA 24 query / 4 KV heads = ratio 6, padded to 8 — i.e. 25% of the tensor-core work is wasted on that architecture.

**llama.cpp — not applicable.** A workaround for a Triton constraint (tl.arange/tl.dot need power-of-two dims) that ggml's template dispatch does not have.

**Equivalent here:** ncols2 is chosen as a power-of-two divisor of gqa_ratio rather than padding it up

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:371-390 (gqa_ratio computed, gqa_opt_applies gated on it)` · `ggml/src/ggml-cuda/fattn.cu:392-437 (the head-dim table imposes gqa_ratio%8==0 for DK 192 and gqa_ratio%32==0 for DK 320 — llama.cpp refuses the case instead of padding)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None. llama.cpp never wastes tensor-core lanes on padded query heads because it selects a divisor of the ratio instead of rounding it up; a ratio it cannot serve falls to a different kernel rather than to masked waste. The vLLM stack's 25% waste at ratio 6 has no counterpart here.

### Per-channel scales hoisted out of the 128-token inner loop
**Where (theirs):** `triton_kvarn_decode.py:439-446` · `triton_kvarn_decode.py:448`

**What it does.** The three per-channel [D] vectors (K column scale, K zero-point, V column scale) are loaded once per block, outside the loop over BLOCK_N-sized token chunks, because they are constant across the tile's 128 tokens. Only the per-token [BN] vectors are loaded inside.

**Mechanism.** `s_col_K`, `zp_K`, `s_col_V` are loaded at the block level (triton_kvarn_decode.py:444-446) before `for c0 in range(0, GROUP, BLOCK_N)` (triton_kvarn_decode.py:448); `s_row_K`, `s_row_V`, `zp_V` are loaded per chunk indexed by `cols` (triton_kvarn_decode.py:464, 471-472).

**Why they needed it.** Comment: 'Per-channel (per-d) K/V scales — constant across the 128 tokens; load once.' With GROUP=128 and BLOCK_N=16 this is 8 chunks per block, so hoisting removes 7/8 of the per-channel scale traffic.

**Their numbers.** GROUP = 128 tokens per tile; BLOCK_N ∈ {16, 32, 64}.

**llama.cpp — not applicable.** The technique only exists because the two-level factorisation puts scales in separate vectors. llama.cpp's block quants do not.

**Equivalent here:** none — q4_0 has one scale per 32-element block, loaded with the block

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/cpy.cu:273-283 (one fp16 d per QK4_0=32 values)` · `src/llama-kv-cache.cpp:231-232 (no side scale vectors on the cache tensor)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None. There are no per-channel or per-token scale vectors in llama.cpp's KV cache to hoist; the scale travels inside the block it belongs to.

### Autotune key = model architecture × quant config
**Where (theirs):** `triton_kvarn_decode.py:311-316` · `triton_kvarn_decode.py:324`

**What it does.** The autotune cache key is (D, GROUP, Q_PER_KV, K_BITS, V_BITS), so Triton re-tunes once per model shape and quantisation preset rather than globally, and the tuning is warmed before CUDA-graph capture so it never fires mid-capture.

**Mechanism.** `key=['D','GROUP','Q_PER_KV','K_BITS','V_BITS']` (triton_kvarn_decode.py:324, 517). Warmup is done in `_warm_decode_kernels` (referenced, triton_kvarn_decode.py:513-514).

**Why they needed it.** 'The autotune key ... makes Triton re-tune per model architecture × quant config' because different Q_PER_KV values 'typically favour different BLOCK_N; the autotuner picks once on first call and caches'. Warming matters because an autotune sweep inside a graph capture would be captured or would fail.

**llama.cpp — not applicable.** Follows from technique 19 — no autotuner, hence no key.

**Equivalent here:** template instantiation per (DKQ, DV, ncols1, ncols2, type_K, type_V) at build time

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-vec.cuh:578-585 (DECL_FATTN_VEC_CASE / EXTERN_DECL_FATTN_VEC_CASES — one instance per head dim × K type × V type)` · `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1967-1969 (DECL_FATTN_MMA_F16_CASE per DKQ/DV/ncols1/ncols2)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None. The specialisation the autotune key buys at runtime, llama.cpp buys at compile time; that is also why GGML_CUDA_FA_ALL_QUANTS costs binary size and compile time rather than first-call latency.

### Triton integer-specialisation opt-out for MAX_BLOCKS_PER_REQ
**Where (theirs):** `triton_kvarn_decode.py:156-163`

**What it does.** The materialize kernel's MAX_BLOCKS_PER_REQ argument is explicitly excluded from Triton's automatic integer specialisation, forcing warmup and serving to share one compiled variant.

**Mechanism.** `@triton.jit(do_not_specialize=['MAX_BLOCKS_PER_REQ'])` (triton_kvarn_decode.py:163).

**Why they needed it.** Quoting: 'Triton still specialises integer args on value 1 and on divisibility by 16 — so the warmup's tiny n_blocks and serving's per-batch ceil(max_seq_len / group) would still land in different compiled variants (a JIT stall mid-serving and, on the chunked-prefill route, a fresh compile every time the value crosses a %16 boundary).' This is the subtle case: the arg was already demoted from constexpr and STILL caused recompiles.

**Their numbers.** Triton specialises on value==1 and on %16==0; noted as working on Triton 3.7.1.

**llama.cpp — not applicable.** There is no JIT in ggml-cuda, so integer specialisation cannot happen. The equivalent cache-churn hazard here is CUDA graph capture, and llama.cpp guards it by shape padding.

**Equivalent here:** none (no JIT); the analogous discipline is the unconditional 256-padding of n_kv to keep the graph constant

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:1233-1246 (get_n_kv rounds used_max_p1 up to 256 explicitly 'so the graph remains constant across batches and can be reused')` · `ggml/src/ggml-cuda/ggml-cuda.cu:4249-4274 (CUDA graph warmup: two consecutive calls with unchanged node properties before capture; any change resets it)`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Nothing to set. The idea does transfer as a way of reading llama.cpp: the 256-padding at llama-kv-cache.cpp:1238 exists for exactly the reason this technique exists — keep a shape-derived key constant so the compiled/captured artifact survives. That framing is what makes technique 34 below actionable.

### Demoting constexprs that churn the JIT cache every 128 tokens
**Where (theirs):** `triton_kvarn_decode.py:179-186` · `triton_kvarn_decode.py:345-351` · `triton_kvarn_decode.py:892-896`

**What it does.** MAX_BLOCKS_PER_REQ was changed from tl.constexpr to a plain runtime int in the materialize kernel, and removed entirely from the three fused kernels which never used it. The verify driver's max_ctx_blocks likewise stopped being forwarded.

**Mechanism.** In `_kvarn_build_packed_kv_kernel` it is now an untyped runtime parameter used only for the `bk // MAX_BLOCKS_PER_REQ` / `bk % MAX_BLOCKS_PER_REQ` decomposition (triton_kvarn_decode.py:186, 208-209). In `_kvarn_fused_decode_kernel`, `_kvarn_fused_decode_stage1` and `_kvarn_fused_verify_stage1` it is deleted, with a standing instruction: 'Launch sites must NOT pass MAX_BLOCKS_PER_REQ to these three kernels' (triton_kvarn_decode.py:351).

**Why they needed it.** Quoting: 'As a constexpr it was part of the JIT cache key, and kvarn_verify_attention receives a per-step max_ctx_blocks from _fused_verify_path -> a fresh compile every 128 tokens of context.' For the chunked-prefill caller the value 'changes every chunk / every 128 tokens of context — as a constexpr each new value was a fresh Triton compile.' After the change, 'at most the two adaptive_num_kv_splits variants, 32 and 64, get compiled' (triton_kvarn_decode.py:895-896).

**Their numbers.** one fresh compile per 128 tokens of context before the fix; at most 2 compiled variants after.

**llama.cpp — not applicable.** Same reason as 21.

**Equivalent here:** none (no JIT cache); nearest hazard is CUDA graph warmup reset on any node-property change

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:4253-4268 (warmup_complete requires two identical consecutive calls; any property change resets it to eager execution)` · `ggml/src/ggml-cuda/common.cuh:1426-1444 (the graph map is keyed on nodes[0], and entries unused for 10 s are destroyed on a 5 s sweep)`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No change available. The same failure shape is present here in a different guise and is diagnosable today — see technique 34.

### Masked slot lookup to keep Triton's type inference happy
**Where (theirs):** `triton_kvarn_decode.py:226-231` · `triton_kvarn_decode.py:202-205`

**What it does.** The slot lookup is deliberately written as a masked tensor load with `other=-1` rather than a scalar branch, and the materialize kernel deliberately omits a runtime batch-bound guard.

**Mechanism.** `in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP); safe_bid = tl.where(in_range, block_id, 0); pool_slot = tl.load(..., mask=in_range, other=-1)` (triton_kvarn_decode.py:229-231).

**Why they needed it.** Two distinct compiler workarounds are documented. For the lookup: 'Avoids mixing a Python int with a tensor across the branch below, which Triton's type inference rejects in the general-batch compilation' (triton_kvarn_decode.py:227-228). For the missing guard: 'b is always < B by construction (grid dim 0 == B*MAX_BLOCKS_PER_REQ), so no runtime-B guard is needed — avoiding it keeps the kernel free of a non-constexpr early-return that Triton's type inference mishandles' (triton_kvarn_decode.py:203-205). Both are cases where the safe-looking code does not compile and correctness is instead argued from the launch geometry.

**llama.cpp — not applicable.** Language-specific.

**Equivalent here:** none

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-mma-f16.cuh:716-742 (bounds are handled with an oob_check template parameter, resolved at compile time)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None. A Triton compiler workaround with no CUDA C++ counterpart; ggml resolves the equivalent branch at template-instantiation time.

### Verify seq_lens taken from the builder's CPU view, not the device tensor
**Where (theirs):** `triton_kvarn_decode.py:1049-1054` · `triton_kvarn_decode.py:1082`

**What it does.** The per-token causal lengths the verify kernel reads are built CPU-side by the metadata builder rather than read from the device seq_lens tensor, and a request's full length is recovered as its LAST token's entry.

**Mechanism.** `seq_len = tl.load(Seq_lens_ptr + b*QLEN + (QLEN-1))` (triton_kvarn_decode.py:1082) — indexing the last of the request's QLEN entries.

**Why they needed it.** Quoting: 'Built CPU-side in the builder: under async spec decode the device seq_lens tensor can disagree with the builder's CPU view, and the CPU view is the one the (validated) per-token path uses.' Two sources of truth for sequence length exist under async scheduling, and the choice of which one the kernel trusts is load-bearing.

**llama.cpp — not applicable.** The technique resolves an ambiguity that this architecture does not create.

**Equivalent here:** none — there is one source of truth, the ubatch built on the host each decode

**Evidence (llama.cpp):** `src/llama-context.cpp:1739 (llama_decode splits the batch by cparams.n_ubatch on the host)` · `src/llama-graph.cpp:479-484 (every position/mask/index input is set from the host-side ubatch before the graph runs)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None. llama.cpp has no async scheduler producing a second, possibly-stale device view of sequence lengths, so there is no choice to get wrong.
