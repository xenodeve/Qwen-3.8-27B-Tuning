# kvarn/files/vllm/v1/attention/backends/kvarn_attn.py — the KVarN attention backend (backend class, metadata + metadata builder, attention impl, store/flush/decode paths)
**49 techniques.** 2643 source lines across 1 files.
Files read: `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py`
> **What the reader could not see:** Everything this file DEPENDS on is outside the slice and I did not open it, so numeric constants that live there are quoted only as this file quotes them: (1) `KVarNConfig` / `KVARN_PRESETS` in `vllm/model_executor/layers/quantization/kvarn/config.py` — owns `group`, `key_bits`, `value_bits`, `sinkhorn_iters` (never given a numeric default anywhere in this file), `tile_bytes_aligned`, all eight field offsets (`k_packed_offset`, `k_s_col_offset`, `k_zp_offset`, `k_s_row_offset`, `v_packed_offset`, `v_s_col_offset`, `v_s_row_offset`, `v_zp_offset`), `pool_slots()`, and `fa_scratch_rows()` (the KVARN_FA_SCRATCH_CAP default value). (2) `vllm/v1/attention/ops/triton_kvarn_sinkhorn.py` — the actual Sinkhorn iteration (this file only calls `kvarn_sinkhorn_triton(tiles, iterations=...)` and consumes `(balanced, s_col, s_row)`; the balancing math itself is not here). (3) `vllm/v1/attention/ops/kvarn_store.py` — the RTN quantiser `kvarn_store_tile_{k,v}_batch_from_sinkhorn`; the actual rounding/zero-point derivation is not in this file. (4) `vllm/v1/attention/ops/triton_kvarn_decode.py` / `kvarn_decode.py` — the in-kernel dequant, the split-K decode, `adaptive_num_kv_splits`, `_kvarn_scatter_store_kernel`, `_kvarn_build_packed_kv_kernel`. Also: the module docstring (lines 10-19) describes a 4D vLLM-shape reinterpretation `(num_blocks, 128, num_kv_heads, 140)` that `get_kv_cache_shape` (line 313) contradicts — it returns 3D `(num_blocks, num_kv_heads, tile_bytes_aligned)`. The docstring appears stale. The docstring also says `head_dim=128, k_bits=4, v_bits=4` for the 17920 B figure while a later comment (line 1796) calls `k4v2` "the default preset"; the file never states which preset is actually default.

---

## EXISTS, NEVER SET — 1

### Sliding window carried into the impl so the decode kernel can bound its block loop
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1178-1183` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1568`

**What it does.** `self.sliding_window` is stored on the impl and passed to every warmed/launched decode kernel as the `SLIDING_WINDOW` constexpr, so window layers only walk the last `sliding_window` keys' blocks.

**Mechanism.** Set at 1183 (`sliding_window or 0`), part of the group key at 1192, and part of the `common` constexpr dict at 1568.

**Why they needed it.** Lines 1179-1182: "Sliding-window layers (e.g. Gemma-4: 50/60 layers, window 1024) only attend to the last `sliding_window` keys. Stored so the decode kernel can bound its block loop to the window — without this it reads the FULL history every step (16x too much work + wrong output past the window)."

**Their numbers.** Gemma-4: 50 of 60 layers sliding, window 1024; without the bound, "16x too much work"

**llama.cpp — EXISTS, NEVER SET.** llama.cpp implements the same optimisation one level lower — instead of bounding a kernel loop, it allocates a physically smaller SWA cache — but it is inert for us: qwen35.cpp declares no SWA, so hparams.swa_type stays NONE, the model routes to the PLAIN llama_memory_hybrid rather than the iswa variant, and the server force-disables --swa-full with a warning. So the machinery exists, is never exercised by this model, and no flag can change that.

**Equivalent here:** llama_kv_cache_iswa / llama_memory_hybrid_iswa with size_swa = n_swa + n_ubatch, and --swa-full to opt out

**Evidence (llama.cpp):** `src/llama-kv-cache-iswa.cpp:70-79` · `src/llama-model.cpp:2305-2344` · `tools/server/server-context.cpp:1188-1195`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero for Qwen3.8. Listed as exists-but-unused specifically so nobody spends a round tuning --swa-full expecting a KV saving — it will print 'swa_full is not supported by this model, it will be disabled' and change nothing.

## absent, has a seam — 4

### KIVI-orientation asymmetric quantisation: K per-channel, V per-token
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:5-8` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1831-1833` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1916-1918`

**What it does.** For one tile (one cache block × one kv-head) the K tile is transposed to `[D, group]` so the absorb/quantisation axis is the CHANNEL, and the V tile is left as `[group, D]` so the absorb axis is the TOKEN. Both are then Sinkhorn-balanced and RTN-packed with the same code, but the resulting per-row scale arrays mean different things: K's `s_col` is per-channel (length D) and V's `s_col` is per-head-dim (length D) while V's `zp`/`s_row` are per-token (length group).

**Mechanism.** `_flush_tail` lines 1832-1833: `K_tiles = K_rot.permute(1, 2, 0).contiguous() # [Hk, D, group]` and `V_tiles = V_rot.permute(1, 0, 2).contiguous() # [Hk, group, D]`. The batched path does the same for N blocks at once, lines 1917-1918: `K_tiles = K_rot.permute(0,2,3,1).reshape(nB*Hk, D, G)` / `V_tiles = V_rot.permute(0,2,1,3).reshape(nB*Hk, G, D)`, with the inline comment "Tiles: K [N, D, G] (absorb=channel), V [N, G, D] (absorb=token)."

**Why they needed it.** Module docstring line 6-7: "K is quantized per-channel, V per-token — KIVI orientation." K's outliers are channel-persistent, V's are token-local; a single orientation for both loses accuracy on one of them.

**llama.cpp — absent, has a seam.** Two separate halves, and llama.cpp fails both. ORIENTATION: every ggml block-quant type quantises along the row, and a KV row is one token's head vector, so both K and V are quantised per-token in 32-channel blocks. Per-CHANNEL K would need K stored transposed. The seam exists in name only — v_trans at src/llama-kv-cache.cpp:206 is exactly a transposed-cache flag — but it is tied to `!cparams.flash_attn`, and a transposed cache cannot be quantised at all (src/llama-context.cpp:463-467, quantized V requires FA). There is no CUDA FA kernel that reads a transposed quantised K, so this is a write-your-own-kernel job. ASYMMETRY: q4_0 is symmetric (scale only); q4_1 and q5_1 carry a min as well, i.e. asymmetric RTN, and the -ctk/-ctv whitelist already accepts them (common/arg.cpp:305-315). They are unreachable here only because GGML_CUDA_FA_ALL_QUANTS is OFF in this build (ggml/src/ggml-cuda/fattn.cu:343-348 returns false for Q4_1/Q5_0/Q5_1). That half is a rebuild, not a rewrite.

**Equivalent here:** none. The nearest thing is v_trans (V stored transposed when FA is off), plus the asymmetric weight types q4_1/q5_1 that the -ctk/-ctv parser accepts but this build has no FA kernel for

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:206` · `ggml/src/ggml-cuda/fattn.cu:340-357` · `common/arg.cpp:305-315` · `src/llama-context.cpp:3613-3633`

**Effort:** new-backend · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** The orientation half: not worth attempting on this hardware budget. The asymmetry half is cheap and testable — rebuilding with -DGGML_CUDA_FA_ALL_QUANTS=ON unlocks -ctk q4_1 -ctv q4_1 (5.0 bits/elem vs q4_0's 4.5, so ~11% MORE KV VRAM for an asymmetric quantiser) and, separately, mixed K/V pairs like -ctk q8_0 -ctv q4_0. On 9.5 GB free with a 6.77 GB model, +11% KV is a real cost; the question is whether it buys enough quality to matter, and that is unknown until measured.

### Route cached-multiquery (MTP verify) by context depth: fused kernel vs materialize+FlashAttention
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2470-2492` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2398-2410`

**What it does.** A verify step with cached history has two implementations. The materialize route builds the whole batch's rotated fp16 K/V into shared scratch with one Triton kernel and runs a single `flash_attn_varlen`. The fused route makes each query token a virtual decode row reading int4 tiles + fp16 pool directly, with no materialization. The choice is made per step from the CONTEXT depth in blocks.

**Mechanism.** Line 2483-2488: take the fused path iff `KVARN_FUSED_VERIFY=1` AND `max_query_len <= KVARN_FUSED_VERIFY_MAXQ` (default 8) AND `ceil(max_seq_len / group) >= KVARN_FUSED_VERIFY_MIN_BLOCKS` (default 64) AND `B > 0`. Otherwise materialize; and fall to `_decode_path_slow` if FA is absent, `head_size > 256`, or `total_k > self._fa_K_buf.shape[0]`.

**Why they needed it.** Lines 2474-2481, measured: "materialize wins short context (88 vs 81 tok/s @2K — its one write+read is cheap there and the fused per-call overhead shows); fused wins long context (51 vs 45 @32K, growing with depth — the materialize round-trip is the O(context)/step issue #10 MTP slowdown). Crossover ~12K; default threshold 64 blocks (8K)." The underlying problem is at 2403-2406: "the issue #10 long-context MTP collapse: measured 88 -> 45 tok/s from 2K -> 32K with the materialize route vs near-flat without MTP."

**Their numbers.** Qwen3.6-27B AWQ, single stream, MTP verify: materialize 88 tok/s vs fused 81 tok/s at 2K context; materialize 45 vs fused 51 at 32K. Materialize route alone degrades 88 → 45 tok/s from 2K → 32K, "near-flat without MTP". Crossover ~12K; shipped threshold 64 blocks = 8K; KVARN_FUSED_VERIFY_MAXQ default 8.

**llama.cpp — absent, has a seam.** This is the most transferable finding in the slice. llama.cpp HAS both routes and they behave exactly like KVarN's: the VEC kernel reads the quantised cache in place (no conversion), the MMA_F16 kernel calls launch_fattn with need_f16_K = need_f16_V = true and dequantises the ENTIRE padded K and V of that layer into F16 scratch on every call — i.e. O(context) work per layer per step, which is precisely the cost profile KVarN measured as 88 -> 45 tok/s from 2K to 32K. But llama.cpp picks between them on Q->ne[1] alone: with quantised KV, VEC only when Q->ne[1] <= 2, otherwise MMA_F16 (ggml/src/ggml-cuda/fattn.cu:469, :482). Context depth is not an input. Making it one is a real patch at a named seam (ggml_cuda_get_best_fattn_kernel), but it cannot actually be done cheaply because the VEC kernel physically caps at 2 query columns (fattn-vec.cuh:553-572 — cols_per_block is 1 or the constant 2, there is no wider instance), so 'prefer VEC at depth' means writing a wider quantised-KV vector kernel. Two consequences that need no patch at all: (1) every speculative verify step with n_draft >= 2 sits on the dequant-the-whole-cache path; (2) because the chain priority is hardcoded with every n-gram speculator ahead of every model-based one (common/speculative.cpp:2542-2552), the measured `draft-dflash,ngram-mod` pair is really ngram-mod FIRST with dflash as fallback — and ngram-mod drafts 64 tokens, so those steps run MMA_F16 with a 64+1-wide query.

**Equivalent here:** ggml_cuda_get_best_fattn_kernel routes on QUERY-TOKEN COUNT (Q->ne[1] <= 2 for quantised KV) and never on context depth; the MMA_F16 branch is llama.cpp's 'materialize' route

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:466-470` · `ggml/src/ggml-cuda/fattn.cu:482` · `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963` · `ggml/src/ggml-cuda/fattn-common.cuh:1022-1084` · `ggml/src/ggml-cuda/fattn-vec.cuh:553-572` · `common/speculative.cpp:2542-2552`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Concrete and testable today with a flag we already have: --spec-draft-n-max 1 makes a dflash verify step 2 query tokens wide, which is the VEC threshold, eliminating the per-step full-cache F16 expansion — at the cost of drafting only one token. That is a real crossover to measure, paired within one round, exactly like the vLLM crossover at ~12K. It also predicts that the +48.5% pair's advantage should SHRINK with context depth relative to dflash alone, since ngram-mod's 64-wide drafts guarantee the expensive kernel. VRAM: the F16 scratch is 2 bytes/element against q4_0's 0.5625, transient, charged to the compute buffer, and already budgeted at reserve time (the reserve runs pp shapes, so MMA is always the reserved case) — so it is VRAM you are already paying whether or not you ever hit it.

### Whole MTP verify step as ONE captured CUDA graph via UNIFORM_BATCH + a persistent vq plan
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:401-416` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:934-990` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2369-2392`

**What it does.** Spec-decode verify steps (uniform query length 1+num_spec) declare `AttentionCGSupport.UNIFORM_BATCH`, so the whole step replays as one full graph. The per-token plan the verify kernel needs — which block-table row each virtual row belongs to and what its causal length is — is written CPU-side into PERSISTENT pinned/device buffers between replays, so pointers stay stable and only values change.

**Mechanism.** `_cudagraph_support` at 412-416 is `UNIFORM_BATCH` when `KVARN_FUSED_VERIFY != 0`, else `UNIFORM_SINGLE_TOKEN_DECODE`. The plan is filled at 959-970: for each decode request, `committed = seq_lens_cpu[b] - ql`, and for `j in range(ql)`: `vq_req_host[i] = b`, `vq_seqlen_host[i] = committed + j + 1` (the bottom-right causal staircase). Buffers `_vq_req_buf` / `_vq_seqlen_buf` are allocated once at `max(num_decode_tokens, 4096)` rows (line 946) and copied non-blocking.

**Why they needed it.** Lines 402-410: "the whole MTP step replays as ONE full graph like vanilla FA, instead of ~num_layers eager attention calls between piecewise segments per step (the dominant MTP overhead once the materialize round-trip was gone; the gap to vanilla was 0.65-0.85x and worse under TP). All Python state mutation (slot allocation, sink marking, tile-boundary flush, the vq verify plan) happens in KVarNMetadataBuilder.build() between captured graph replays; the forward is pure tensor ops."

**Their numbers.** gap to vanilla FA before this change: 0.65-0.85×, "and worse under TP"

**llama.cpp — absent, has a seam.** llama.cpp's capture is strictly two-consecutive-identical-calls (ggml-cuda.cu:4253-4262) and ANY node-property change resets warmup to eager (:4265-4268). Upstream, graph reuse requires equal n_tokens (src/llama-graph.h:785); a differing count forces a rebuild, a re-split, a new uid, different node properties, and a warmup reset. Speculative decoding alternates step widths by construction — 1+n_draft on verify, then a variable number of accepted tokens — so this workload is the exact case the map flags as 'never captures'. KVarN's fix is to make the width UNIFORM by declaring it so and padding. The seam in llama.cpp is the same allow_reuse predicate plus the server's batch assembly, but making width constant means always submitting 1+n_max tokens and masking the rejected ones, which touches the batch allocator and the output-limit plumbing — not a small change. Note also the graph map is keyed on cgraph->nodes[0], i.e. WHICH SPLIT, not which shape (ggml-cuda.cu:2574-2576), so two widths cannot coexist as two captured graphs even in principle.

**Equivalent here:** CUDA graph capture exists and is on, but llm_graph_params::allow_reuse requires ubatch.n_tokens to be identical, so a variable-width verify step never re-arms capture

**Evidence (llama.cpp):** `src/llama-graph.h:781-816` · `src/llama-context.cpp:1332-1372` · `ggml/src/ggml-cuda/ggml-cuda.cu:4249-4274` · `ggml/src/ggml-cuda/ggml-cuda.cu:2574-2576` · `tools/server/server-context.cpp:617-619`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Do not patch first — MEASURE first, because the instrument is already printed at INFO on every completion: `graphs reused = %10d` (tools/server/server-context.cpp:617-619, from llama_perf_context().n_reused). If that number is near zero under draft-dflash and healthy without speculation, the eager-execution penalty is real and quantified, and only then is the patch worth costing. Two free experiments alongside it: LLAMA_GRAPH_REUSE_DISABLE=1 and GGML_CUDA_DISABLE_GRAPHS, each of which attributes a regression to reuse vs capture.

### Prefill first chunk attends RAW fp16 K/V — quantisation error never enters the prefill output
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2205-2224` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2166-2175`

**What it does.** When every request's whole prompt is in the current batch (`seq_len == query_len` on every row), attention runs on the uncompressed K/V tensors that just arrived, in parallel with those same tensors being written to the pool by `do_kv_cache_update`. Only later reads see quantised data.

**Mechanism.** `forward` routes there when `num_decodes == 0 and not has_cached_multiquery` (lines 2166-2175). `_prefill_first_chunk` calls `flash_attn_varlen` with `cu_q == cu_k == query_start_loc`, or falls back to a per-request SDPA loop.

**Why they needed it.** Docstring: "every request's full prompt is in the current batch, so attention runs on raw K/V via flash_attn_varlen. The K/V have already been written to the cache by do_kv_cache_update." `has_cached_multiquery` exists precisely to keep chunked-prefill continuations and verify steps OUT of this path (lines 366-372): "Such steps MUST attend over the cached K/V, so they route to the context-aware path rather than _prefill_first_chunk (which assumes a fresh prompt, cached_len == 0)."

**llama.cpp — absent, has a seam.** I verified this in the source rather than inferring it. build_attn does cpy_k/cpy_v into the cache, then sets k = mctx_cur->get_k(ctx0, il) and v = mctx_cur->get_v(ctx0, il) and runs build_attn_mha on those — so the tokens of the CURRENT ubatch attend to their own already-quantised K and V, even during prompt processing where the uncompressed k_cur/v_cur tensors are sitting right there in the graph. KVarN's prefill path avoids this entirely. The seam is exactly src/llama-graph.cpp:2806-2810: one would attend the current chunk against raw k_cur/v_cur and the prior context against the cache, then merge. That merge is the expensive part — two FA calls plus an online-softmax combine — which is why I call it large-patch rather than small.

**Equivalent here:** none — build_attn writes k_cur/v_cur to the cache and then reads back the QUANTISED view for the same step's attention

**Evidence (llama.cpp):** `src/llama-graph.cpp:2795-2801` · `src/llama-graph.cpp:2806-2808` · `src/llama-graph.cpp:2810`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown quality gain, and it is a QUALITY lever, not a speed one — it would cost extra work at prefill. Flagging it because our stack stacks two lossy layers (UD-IQ2_XXS weights and q4_0 KV) on a coding agent with long prompts, and this is the one place in the pipeline where a lossless read is available for free and is being thrown away. If we ever chase output quality rather than tok/s, this is where to look.

## partial — 3

### Retired sinks: finished requests' sink blocks stay fp16-resident for prefix-cache re-adoption, evicted lazily oldest-first
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:467-475` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:817-843` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:754-765` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:880-902`

**What it does.** When a request finishes, its complete sink block is not flushed to int4 and not discarded — it is moved to an insertion-ordered `_retired_sinks` dict and its fp16 data stays in the pool. If a later request prefix-cache-hits that physical block, it is un-retired and re-adopted with byte-identical fp16 data. Only when slot allocation actually runs dry is the OLDEST retired sink evicted, at which point it IS flushed to int4 so later cache hits still find a valid tile.

**Mechanism.** Reclaim branch line 832-834: `if full and bid in sinks: self._retired_sinks[bid] = None; continue`. Un-retire at 760-765 for any retired block named this step. Eviction at 882-902 inside the allocation loop: `if not free_slots and self._retired_sinks:` → `old = next(iter(self._retired_sinks))` → `_batched_flush(evict_pairs)` over every impl → free `old_slot`, clear `sinks`/`is_sink_t`/`b2s_t`.

**Why they needed it.** Lines 469-474: "A prefix-cache hit re-adopts the block with its fp16 data byte-identical — preserving KVarN's fp16-sink accuracy on multi-turn traffic, where every follow-up turn reuses the previous turn's first block." And lines 819-822: "the old discard destroyed its fp16-only data outright, garbling every multi-turn cache hit (issue #10 loops)." Lazy eviction is chosen so "residency therefore never shrinks live capacity".

**llama.cpp — partial.** The RE-ADOPTION half is already there and already on: llama-server keeps finished/idle slots' full sequence state in a host-RAM prompt cache, scores candidates by (f_keep, f_sim), evicts oldest-first, and refuses to trash an entry with f_keep < 0.25 — the same lazy-eviction shape. What is absent is the PRECISION half: the saved state is whatever the cache type was, so a q4_0 prefix is restored as q4_0, not as a preserved fp16 copy. There is no way to keep one prefix at a different precision (see technique 9).

**Equivalent here:** -cram / --cache-ram (host-RAM cross-request prompt cache, default 8192 MiB) + --cache-idle-slots + -sps for the re-adoption decision

**Evidence (llama.cpp):** `common/common.h:615` · `tools/server/server-task.cpp:1706-1783` · `tools/server/server-task.cpp:1790-1856` · `tools/server/server-context.cpp:1503-1551`

**Effort:** config · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** With -np 1 and a coding agent that re-sends a growing conversation, the RAM cache is the mechanism that already saves re-prefill. It also has a cost the map flags: every new task does a full llama_state_seq_get_data of the idle slot into RAM before the new prompt starts, and that lands inside the NEXT request's prompt_ms window (tools/server/server-context.cpp:2355-2363, t_start set later at :3053). If prompt_per_second looks wrong, that is a candidate cause. -cram 0 turns it off as a controlled comparison.

### Capped FlashAttention materialize scratch, with a documented fallback consequence
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1451-1473` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2494-2498`

**What it does.** The shared fp16 K/V scratch used by the materialize route is capped rather than sized to the theoretical maximum. A batch whose total context exceeds the cap silently takes `_decode_path_slow` instead.

**Mechanism.** Sizing delegated to `KVarNConfig.fa_scratch_rows(max_num_seqs, max_model_len)` (line 1472), overridable by `KVARN_FA_SCRATCH_CAP`. Enforcement at 2496-2498: `total_k = int(cu_k[-1].item()); if total_k <= 0 or total_k > self._fa_K_buf.shape[0]: return self._decode_path_slow(...)`.

**Why they needed it.** Lines 1453-1457: "The theoretical bound max_num_seqs * max_model_len is pathological (e.g. 256×8192 = 2.1M tokens ≈ 8.6 GB) and would starve the actual KV cache. Cap it at FA_SCRATCH_CAP tokens (~1 GB of fp16 K+V)." The port note at 1458-1465 adds that the original code "forced 1.0 GiB at max_model_len 262144 regardless of the cap" and names the trade-off explicitly.

**Their numbers.** uncapped bound 256×8192 = 2.1M tokens ≈ 8.6 GB; cap ≈ 1 GB of fp16 K+V; the pre-fix bug forced 1.0 GiB at max_model_len 262144

**llama.cpp — partial.** llama.cpp does the budgeting half correctly — the scratch is routed through the buffer-type alloc-size hook so ggml-alloc reserves it and --fit sees it — but there is no cap and no graceful degradation: if it does not fit, you do not silently take a slower path, you fail to allocate. Adding a cap would mean adding a fallback attention path, which does not exist for this shape. The available levers are indirect: smaller -c (less n_kv to expand) or smaller -ub (smaller pp graph).

**Equivalent here:** ggml_cuda_flash_attn_ext_get_alloc_size charges the F16 dequant scratch to the compute buffer; it is budgeted but NOT capped and has no fallback

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:534-568` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912` · `common/arg.cpp:2851-2874` · `common/common.h:473`

**Effort:** large-patch · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** On 12 GB the directly useful number here is -fitt, not the scratch: the default forfeits 1024 MiB PER DEVICE (common/common.h:473), which on this card is ~10% of total VRAM handed back for nothing. -fitt 384 is a one-flag experiment that converts most of that into KV cells or context. It is the single cheapest VRAM lever in the whole map and our profile has never set it.

### Pre-Sinkhorn tile dump hook for outlier analysis
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1979-2005`

**What it does.** Setting `KVARN_DUMP_TILES=/path.pt` makes the legacy flush path save the first chunk's raw fp32 K/V tiles — before Sinkhorn — together with the layer index parsed out of `impl.layer_name`, block ids, Hk, group, head_dim, key_bits, value_bits and sinkhorn_iters.

**Mechanism.** One-shot guard `cls._tiles_dumped`; layer index via `re.search(r"layers\.(\d+)\b", name)` with -1 fallback; `torch.save` of the dict at 1994-2002.

**Why they needed it.** Inline: "dump first chunk's raw (pre-Sinkhorn) tiles for outlier analysis" — it is the instrument for deciding whether the rotation+Sinkhorn is actually flattening the distribution the quantiser sees.

**llama.cpp — partial.** I checked: the seam already exists as a library-level callback in both llama_context_params and common_params, and the graph names the relevant tensors, so dumping the raw pre-rotation K/V of a chosen layer is a callback plus a file write — not a kernel change. What is missing is any CLI or server exposure; the tree has no eval-callback tool staged and no --eval-callback flag in common/arg.cpp, so today it is reachable only from code. Calling this 'partial' rather than 'absent-but-possible' because the mechanism is present and functioning, just not surfaced.

**Equivalent here:** llama_context_params.cb_eval / common_params.cb_eval — a per-node scheduler callback that can observe any graph tensor by name, including k_cur/v_cur before rotation. Not wired to any llama-server flag

**Evidence (llama.cpp):** `include/llama.h:378-379` · `common/common.h:481-482` · `src/llama-graph.cpp:2777-2784`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** This is the instrument, not the optimisation, and it is the one item on this list I would build first if we ever want to argue about KV quantisation quality from data rather than from tok/s. It would let us answer directly whether the Hadamard rotation is actually flattening what q4_0 sees on this model at head_dim 128 — which is the premise the entire rotation family rests on and which we have never measured here.

## already have it — 15

### Sylvester Hadamard rotation, cached per (head_dim, device)
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:88-98` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1712-1713` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1387-1389`

**What it does.** Builds a D×D Hadamard matrix by Sylvester doubling (`H = [[H,H],[H,-H]]` until it reaches d), divides by sqrt(d) so it is orthonormal, and memoises it with `functools.cache` keyed on (d, device_str). A second fp16 copy `self._H_fp16` is materialised once per impl in `_ensure_pool` so the hot store path never pays a `.float()`/cast allocation. Every K and V vector is multiplied by this matrix before quantisation; every query is multiplied by the same matrix before it touches rotated cache data.

**Mechanism.** `_hadamard_cached(d, device_str)` at line 89 loops `torch.cat([torch.cat([H,H],1), torch.cat([H,-H],1)],0)` and returns `(H / math.sqrt(d))` as fp32. `_build_hadamard` (line 97) normalises the device to a string so the cache key is hashable. `KVarNAttentionImpl._hadamard` (line 1712) calls it with `self.head_size`. The fp16 mirror is built at line 1389: `self._H_fp16 = self._hadamard(device).to(torch.float16).contiguous()`.

**Why they needed it.** The module docstring calls the scheme "KV-cache compression by Hadamard rotation + iterative variance-normalization (Sinkhorn-like) + asymmetric RTN" (lines 5-7). The rotation is the outlier-spreading step that makes 4-bit and 2-bit RTN survivable; being orthonormal it is exactly invertible, so line 2545 can claim "same fp16 Hadamard as the store side, so QK^T is invariant".

**llama.cpp — already have it.** llama.cpp already builds an orthonormal Walsh-Hadamard matrix by exactly the same Sylvester doubling (src/llama-kv-cache.cpp:20-57, with the note `res^2 == I` so it is its own inverse), memoises it host-side keyed by size in attn_rot_hadamard, and turns it on AUTOMATICALLY whenever the cache type is quantised and the head dim is a multiple of 64. With -ctk q4_0 -ctv q4_0 on a head-dim-128 model this is already live in our profile — unasked-for and unlogged except for the two lines `attn_rot_k = %d` / `attn_rot_v = %d` at load. One difference worth knowing: llama.cpp uses a BLOCK-DIAGONAL rotation for V, hardcoded to 64x64 (src/llama-kv-cache.cpp:1440-1452, with the comment that the smaller matrix seems beneficial for V), while K gets the largest power-of-2 divisor of head_dim, i.e. the full 128x128 at head_dim 128. So llama.cpp is not doing full-D rotation on V and has a source comment saying that was deliberate.

**Equivalent here:** attn_rot_k / attn_rot_v + ggml_gen_hadamard + the precomputed attn_rot_hadamard map; kill switch env LLAMA_ATTN_ROT_DISABLE=1

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:312-336` · `src/llama-kv-cache.cpp:20-57` · `src/llama-kv-cache.cpp:343-357` · `src/llama-kv-cache.cpp:1418-1456`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Already being paid for. The actionable thing is the reverse experiment: LLAMA_ATTN_ROT_DISABLE=1 is a free A/B that tells you how much of q4_0's survivability at 2-bit weights comes from the rotation. Read `attn_rot_k = 1` in the startup log to confirm it engaged before trusting any q4_0 quality claim.

### Rotate-on-store: K and V are rotated before they ever enter the fp16 pool
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2062-2071` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1428-1435` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1816-1819`

**What it does.** `do_kv_cache_update` views the incoming key/value as `(N, Hk, D)` and immediately matmuls them by the cached fp16 Hadamard into pre-allocated scratch, then scatters the ROTATED values into the fp16 tail pool. Consequence: the pool holds rotated data, the int4 tiles hold rotated data, and the flush path does no rotation at all — `_flush_tail`'s docstring says "Data is already rotated (rotation happens at do_kv_cache_update), so no `@ H` step here."

**Mechanism.** Lines 2068-2071: `k_rot = self._k_rot_scratch[:N]; torch.matmul(k_view, self._H_fp16, out=k_rot)` — the `out=` form is chosen because "`torch.matmul` `out=` is capture-friendly (uses the caching allocator's pool)" (line 2066). Scratch is `[max_num_batched_tokens, Hk, head_dim]` fp16 allocated once in `_ensure_pool` (lines 1429-1435) "so the captured forward never allocates".

**Why they needed it.** Rotating once at store time means the flush (which runs per completed 128-token tile, per layer) and the decode kernels never pay a rotation, and the store path stays pure tensor ops that are legal inside a captured CUDA graph (line 2044: "No Python loop, no allocator, no dict mutation. Safe inside a captured CUDA graph.").

**llama.cpp — already have it.** Identical ordering. k_cur and v_cur are multiplied by the Hadamard first, and only then handed to mctx_cur->cpy_k / cpy_v, so the KV cache holds rotated data and nothing downstream re-rotates it. The rotation is a graph node fused into the same forward expansion as the cache write (the source comment at src/llama-graph.cpp:2787-2789 says the nodes are expanded together deliberately so the scheduler cannot reorder them and split the graph) — the ggml equivalent of KVarN's 'capture-friendly, no allocator' requirement.

**Equivalent here:** llama_mul_mat_hadamard applied to k_cur/v_cur in llm_graph_context::build_attn, before cpy_k/cpy_v write to the cache

**Evidence (llama.cpp):** `src/llama-graph.cpp:2777-2784` · `src/llama-graph.cpp:2795-2801` · `src/llama-graph.cpp:3019-3026`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None to gain; it is the current behaviour. Worth knowing only so nobody re-invents it as a patch.

### Flush against the COMMITTED length, never the full seq_len — speculative tokens are never frozen into int4
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:767-815` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:610-620`

**What it does.** The tile-boundary flush decision uses `committed_len = seq_len - query_len` (exactly `num_computed_tokens`), not `seq_len`. Under MTP a step appends `num_spec+1` tokens of which a variable number will be rejected; those still sit in the fp16 pool and would be quantised if `seq_len` drove the flush. Because quantisation is irreversible, using the committed boundary is the difference between correct history and progressive corruption.

**Mechanism.** `query_lens_cpu` is derived from `cam.query_start_loc_cpu` differences (lines 614-618, no extra sync). Line 806: `committed_len = max(sl - q_len, 0)`, then `k = min(committed_len // GROUP - 1, bt_cols - 1)` and a backward walk while `1 <= k`.

**Why they needed it.** Quoted at lines 776-784: "under speculative decoding (MTP / draft) a step appends num_spec+1 tokens at once and seq_len jumps by a VARIABLE accepted amount... Quantizing a block to int4 is PERMANENT, so flushing a block that still contains a speculative (rejectable) token freezes wrong KV → progressive corruption → repetition-collapse / garbage." Timing is also spelled out at 768-774: the token completing block k is written by `do_kv_cache_update`, which runs AFTER the builder, so at builder time the pool only holds pre-step tokens.

**llama.cpp — already have it.** The irreversibility that forces KVarN's committed-length rule does not exist here: a rejected draft token's cache cell is just seq_rm'd and rewritten, because quantisation is per-token and idempotent. The genuine analogue is the RECURRENT half of this hybrid model, which cannot be partially rolled back unless n_rs_seq > 0 — and need_n_rs_seq() returns non-zero only for draft-mtp / eagle3 / dflash / dspark. So switching from ngram-mod to draft-dflash silently turned this on: the target context now reserves draft.n_max recurrent snapshots and can roll the DeltaNet state back directly instead of going through the checkpoint path.

**Equivalent here:** per-token KV cells are overwritten on rejection via llama_memory_seq_rm; with draft-dflash the target additionally gets cparams.n_rs_seq = draft.n_max for bounded recurrent-state rollback

**Evidence (llama.cpp):** `common/common.h:386-392` · `common/common.cpp:1697` · `src/llama-arch.cpp:1044-1055` · `src/llama-memory-recurrent.cpp:180-190` · `tools/server/server-context.cpp:3381-3384`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Directly relevant to today's +34.7% draft-dflash result and possibly part of the cause. Under ngram-mod, n_rs_seq was 0, common_context_can_seq_rm returned FULL, and every speculative step that partially accepted had to restore a checkpoint (which on a non-SWA hybrid stores ONLY the recurrent state — src/llama-memory-hybrid.cpp:190-196 — so the attention KV was recomputed). Under draft-dflash the RS branch is available. Two caveats to check before crediting it: (a) n_rs_seq is clamped to 0 for an arch not in llm_arch_supports_rs_rollback and that clamp logs at DEBUG only (src/llama-context.cpp:104-109), so run at -lv 5 once to confirm QWEN35/QWEN35MOE matched; (b) --spec-draft-n-max sets how many snapshots are reserved, so it is also a VRAM knob on the recurrent half.

### Single H2D of the slot/block index lists per flush, sliced on device per chunk (WSL fix)
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1904-1912`

**What it does.** The whole block set's slot indices and block ids are moved to the device once with `torch.as_tensor`, then each chunk slices the device tensor rather than doing its own host→device transfer.

**Mechanism.** Lines 1906-1907 build `slots_dev` / `bids_dev`; lines 1911-1912 take `slots_dev[c0:c0+CHUNK_BLOCKS]`.

**Why they needed it.** Inline: "WSL fix (PR #16): one H2D for the whole block set, slice on device per chunk (a torch.as_tensor H2D per chunk is a sync, ~100x on WSL)."

**Their numbers.** ~100× penalty per chunked H2D on WSL

**llama.cpp — already have it.** The pattern is already the rule here: cell indices are computed on the host once per ubatch into a single graph-input tensor and transferred with the other inputs, never per-chunk and never per-token. There is no per-op H2D in the KV write path to eliminate.

**Equivalent here:** build_input_k_idxs / set_input_k_idxs — one host-built I64 index tensor per ubatch, uploaded with the rest of the graph inputs

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:1390-1416` · `src/llama-kv-cache.cpp:1459-1470`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none, but the underlying warning generalises to our environment: any added instrumentation that does a per-step host<->device transfer will cost far more than the arithmetic it measures.

### Store scatter as a Triton kernel with GPU-side block_id→slot indirection
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2073-2093`

**What it does.** After rotation, tokens are scattered into the fp16 pool by a Triton kernel launched `(N, Hk)` that reads `slot_mapping`, divides by GROUP to get the block id, and looks the pool slot up in the GPU `_block_to_slot_t` tensor. No Python-side dictionary is consulted, so the whole update is legal inside a captured CUDA graph.

**Mechanism.** `_kvarn_scatter_store_kernel[(N, Hk)](k_rot, v_rot, slot_mapping[:N], self._block_to_slot_t, self._tail_K_pool, self._tail_V_pool, ...strides..., GROUP=cfg.group, D=D, NUM_BLOCKS_LOOKUP=self._block_lookup_size, num_warps=2, num_stages=2)`.

**Why they needed it.** Line 2091-2093: "No CPU bookkeeping here — fill tracking + flush triggering live in KVarNMetadataBuilder.build() (outside the captured region). This method is now pure tensor ops, safe inside a captured CUDA graph." The allocator mutates `_block_to_slot_t` only in `build()`, between replays, so a captured graph always sees fresh values.

**llama.cpp — already have it.** Same property, arrived at the same way: the destination indices are a device tensor, the store is one indexed scatter op, and no host data structure is consulted inside the forward. That is exactly what makes it capturable. llama.cpp's capture is at the whole-graph level rather than per-op, and it arms only after two consecutive calls with identical node properties (ggml-cuda.cu:4253-4262) — see technique 24 for why that matters under speculation.

**Equivalent here:** cpy_k / cpy_v as ggml_set_rows driven by the k_idxs / v_idxs input tensors; CUDA graph capture is on by default (GGML_CUDA_USE_GRAPHS, cc >= Volta)

**Evidence (llama.cpp):** `src/llama-graph.cpp:2795-2801` · `src/llama-kv-cache.cpp:1390-1416` · `ggml/src/ggml-cuda/ggml-cuda.cu:4218-4231` · `ggml/src/ggml-cuda/common.cuh:1255-1259`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none to add; but knowing the store path is already capture-clean means any CUDA-graph problem we see is attributable to graph SHAPE churn, not to the KV write.

### Query-side rotation instead of un-rotating the cache
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2543-2559` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2324-2332`

**What it does.** Because K and V are stored rotated, the attention paths rotate the QUERY by the same fp16 Hadamard, run attention entirely in the rotated frame, and rotate the OUTPUT back — instead of un-rotating the (much larger) K/V.

**Mechanism.** In `_cached_multiquery_path`: `q_rot = torch.mm(q.reshape(-1, D), H16).view(n_tok, self.num_heads, D)`, FA on `(q_rot, K_packed, V_packed)`, then `torch.mm(out_rot.reshape(-1, D), H16)`. The fused decode driver takes the fp32 Hadamard directly (`hadamard=self._hadamard(q.device)` at line 2327) and does the same inside the kernel.

**Why they needed it.** Line 2544-2545: "same fp16 Hadamard as the store side, so QK^T is invariant." One [n_tok, D] matmul replaces an [context, Hk, D] one per step.

**llama.cpp — already have it.** Identical strategy and for the identical reason: rotate the [n_tok, D] query rather than the [n_kv, D] cache, so QK^T is invariant. llama.cpp gets the un-rotation for free because ggml_gen_hadamard is normalised so that H^2 == I (the comment at src/llama-kv-cache.cpp:22-23 says so), which is why line 2814 can apply the SAME matrix to the output instead of a transpose.

**Equivalent here:** q_cur is Hadamard-rotated by the same self_k_rot matrix; the attention OUTPUT is rotated back by self_v_rot

**Evidence (llama.cpp):** `src/llama-graph.cpp:2777-2784` · `src/llama-graph.cpp:2812-2815` · `src/llama-kv-cache.cpp:22-23`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none to add — it is already the behaviour under -ctk q4_0 -ctv q4_0.

### Non-causal (bidirectional) verify support for DFlash cross-attention drafting
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:244-250` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:955-984` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:170-176`

**What it does.** When `CommonAttentionMetadata.causal` is False, each virtual verify row's limit becomes the flat full context length (`committed + ql`) instead of the causal staircase, and the shared-dequant kernel is forcibly disabled for that step because it bakes the staircase in.

**Mechanism.** `non_causal = not causal_flag` at 958; `vq_seqlen_host[i] = full if non_causal else committed + j + 1` at 969; `vq_qlen = uniform if (uniform >= 2 and not non_causal) else 0` at 984, which routes to the per-token kernel. `_causal_bool` (line 170) reduces a per-request tensor with `all().item()` — one D2H sync, in `build()` only, "never in forward()".

**Why they needed it.** Lines 246-250: "Causality is purely a masking choice; the KV quantization is independent of it. The per-token verify path attends each query row to [0, vq_seqlen[row]) — a flat full-context length per row gives bidirectional attention (used by DFlash cross-attention drafting)."

**llama.cpp — already have it.** llama.cpp already supports exactly this and already applies it to the speculator we are running: the DFlash/DSpark constructor sets the draft context non-causal for its whole life. Because the draft has its own llama_context and its own memory module (common/speculative.cpp:2464-2482), the setting is scoped to the draft and cannot leak into the target. The masking-vs-quantisation independence KVarN argues for is likewise structural here — the kq_mask is a graph input, the cache type is a context parameter.

**Equivalent here:** llama_set_causal_attn(ctx_dft, false) forced for draft-dflash / draft-dspark at construction

**Evidence (llama.cpp):** `common/speculative.cpp:1036` · `common/speculative.cpp:910-1347` · `src/llama-context.cpp:229-230`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Already active under --spec-type draft-dflash. Worth recording because it explains why dflash needs no special flag from us and why its draft context behaves differently from every other type's.

### Shared-dequant verify kernel exists but is default-OFF for latent corruption
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1624-1664`

**What it does.** A faster verify kernel that shares one dequant across a request's uniform-length query tokens is implemented and warm-up-compiled — but only when `KVARN_SHARED_VERIFY=1`, which defaults to 0.

**Mechanism.** Warmup gate at 1641-1643: `_qlen >= 2` AND `(_qlen * qpk_pad)` is a power of two AND `os.environ.get("KVARN_SHARED_VERIFY", "0") == "1"`. The power-of-two check "mirror[s] the driver's guard in triton_kvarn_decode.kvarn_verify_attention — the shared-dequant kernel's Q tile is [QLEN * Q_PER_KV_PAD, D] and tl.arange needs a power-of-2 row count".

**Why they needed it.** Verbatim at 1627-1630: "that kernel is only used with KVARN_SHARED_VERIFY=1 (default OFF — latent corruption, plan §4 'MTP'), so gate the warmup on it too instead of paying its compile + autotune sweep (and its local-memory reservation) for nothing."

**llama.cpp — already have it.** The optimisation KVarN gates off is already llama.cpp's default and is not experimental: one to_fp16 pass over the layer's K and V per call, shared by all Q columns in that step. llama.cpp's problem is the opposite of KVarN's — the shared dequant is unconditional and expensive (technique 23), not risky and optional.

**Equivalent here:** the MMA_F16 path dequantises K and V once per FLASH_ATTN_EXT call and every query column of that ubatch reads the same F16 scratch

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963` · `ggml/src/ggml-cuda/fattn-common.cuh:1022-1084`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none as a change. Relevant only as the correct mental model: at n_draft >= 2 the dequant is paid ONCE per layer per step, not once per drafted token, so the cost scales with context depth and layer count, not with draft length.

### Decode-kernel warmup at profile time so the CUDA-graph memory estimate is not over-charged
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1410-1426` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1531-1684` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2118-2129`

**What it does.** Every decode-path Triton kernel — the single-stage fused decode (with its `@triton.autotune` sweep), split-K stage1/stage2, both `VQ_INDIRECT` specialisations, optionally the shared verify kernel, and the packed-KV build kernel — is compiled and autotuned on tiny throwaway tensors (B=8, 4 blocks) during profiling. `_ensure_pool` is also deliberately called BEFORE the `attn_metadata is None` early return in `forward` so pool + scratch materialise inside the measured profiling window.

**Mechanism.** `_warm_decode_kernels` (line 1531) builds synthetic `cache`, `pool_k/v`, `b2s`, `bt`, `sl`, `q`, launches each kernel with the deployment's real constexprs (`D`, `GROUP`, `Q_PER_KV`, `Q_PER_KV_PAD`, `SLIDING_WINDOW`, `K_BITS`, `V_BITS`, all eight field offsets), then `torch.cuda.synchronize(device)`. Keyed by `("decode", device, head_dim, group, key_bits, value_bits, num_heads, num_kv_heads, sliding_window)`.

**Why they needed it.** Lines 1410-1420: "their one-time JIT + autotune cost (including the autotuner's benchmark scratch) used to land in the FIRST real decode — which, since v0.21, is the CUDA-graph memory estimation warmup. The estimate then absorbed those one-time costs and over-charged 'graph memory' by GiBs, directly shrinking the derived KV-cache capacity." Same reasoning at 2120-2126 for calling `_ensure_pool` early: it charges the pools to `non_kv_cache_memory` "instead of the CUDA-graph estimate — no gpu_worker.py hunk needed".

**Their numbers.** over-charged graph memory "by GiBs", shrinking derived KV-cache capacity

**llama.cpp — already have it.** Same discipline, different mechanism. llama.cpp reserves the graph at worst-case shapes before serving and explicitly re-runs the pp reserve last 'to avoid ggml-alloc reallocations during inference', so the compute buffer is sized by a deliberate warm pass rather than by whatever the first real request happened to do. --fit then measures free memory against that. There is no separate CUDA-graph memory estimate to over-charge.

**Equivalent here:** the reserve sequence: FA probe -> pp reserve -> tg reserve -> pp reserve again, then --fit measures against the resulting breakdown

**Evidence (llama.cpp):** `src/llama-context.cpp:576-671` · `src/llama-context.cpp:662-671` · `common/fit.cpp:29-70` · `common/fit.cpp:559-563`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None to change, but one honest caveat for this project: --fit measures free VRAM AT THAT INSTANT (common/fit.cpp:559-563), which is exactly the 9,326-10,732 MiB boot variance the repo already documents. That is the same class of measurement contamination KVarN hit, and it is why raw decode must never be compared across boots.

### Decode scratch shared across all layers and keyed by (device, head_dim, num_kv_heads)
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1073-1086` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1475-1530` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1441-1450`

**What it does.** Ten per-step throwaway decode buffers (q fp32, q rotated fp32/fp16, out rotated fp32, output fp32, fused out, split-K mid_o/mid_lse, FA K/V) are class-level dicts shared by every attention layer, but keyed by the (device, D, Hk) combination rather than device alone. Row count is `max(max_num_batched_tokens, max_num_seqs * Hq, 1)`.

**Mechanism.** `bkey = (device, D, Hk)` at line 1481; instance attributes are rebound to the shared tensors at 1521-1530.

**Why they needed it.** Sharing: "avoids 28× memory waste on the per-layer attention" (line 1075). The composite key: "heterogeneous-head models (e.g. Gemma-4: 256-dim/16-kv sliding layers + 512-dim/4-kv global layers)... a buffer sized for one combo's D/Hk is the wrong width for another (caused a reshape(N,512)-on-256-wide-buffer crash)" (1476-1480). The row count: "the old code silently assumed max_num_batched_tokens >= max_num_seqs * Hq, which breaks when it is set low" (1447-1449).

**Their numbers.** 28× memory saved vs per-layer scratch; Gemma-4 needs 2 scratch sets

**llama.cpp — already have it.** Per-layer scratch duplication is not possible in ggml: intermediates are graph tensors allocated out of one arena whose lifetimes ggml-alloc computes, so the 28x waste KVarN had to avoid never arises. The heterogeneous-head hazard (one buffer width wrong for another layer's D) also cannot occur, because allocation is per-tensor-shape, not per-preallocated-slab.

**Equivalent here:** the ggml-alloc compute arena — one buffer reused by every layer's intermediates, sized by the reserve passes

**Evidence (llama.cpp):** `src/llama-context.cpp:576-671` · `ggml/src/ggml-alloc.c:684`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Per-batch CPU materialisation of metadata, done once instead of per layer
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:373-376` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:599-620`

**What it does.** `seq_lens_cpu`, `query_lens_cpu` and `slot_mapping_cpu` are computed once in `build()` and carried on the metadata object, so no layer's `forward()` issues its own device→host sync.

**Mechanism.** Lines 607-608 prefer the runner's cached `cam._seq_lens_cpu` and fall back to `cam.seq_lens.tolist()`; query lengths are differenced from `cam.query_start_loc_cpu` (already host-side, no extra sync).

**Why they needed it.** Line 374-376: "Precomputed once per batch in the metadata builder and reused across all 28+ layer forward calls. Saves 28× .tolist() syncs per decode token." The port note at 603-606 adds a correctness constraint: "Do NOT use seq_lens_cpu_upper_bound (optimistic under async spec decode: the commit/flush decisions below need exact seq_len - query_len)."

**Their numbers.** 28+ syncs per decode token avoided (Qwen3-0.6B named at line 601)

**llama.cpp — already have it.** Same property by construction. There is no per-layer host round-trip in the decode path — index tensors, masks and position inputs are computed host-side once and uploaded with the graph inputs. The 28x-sync problem is a Python-framework problem.

**Equivalent here:** the ubatch and slot_info are built once per decode; set_input_* fill the graph inputs once and all layers read the same tensors

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:1459-1470` · `src/llama-context.cpp:1332-1372`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Persistent cu_seqlens buffers updated in place for graph capture
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:916-932`

**What it does.** `cu_seqlens_q` (an arange) and `cu_seqlens_k` (prefix sum of seq_lens) live in fixed device buffers with pinned host staging, updated in place each step rather than reallocated.

**Mechanism.** Lazy allocation to `max(B+1, 257)` rows at 920-925 ("257" chosen as "default max_num_seqs headroom"), host fill loop at 926-928, `copy_(..., non_blocking=True)` at 931-932.

**Why they needed it.** Line 917-918: "A captured graph bakes in tensor addresses, so cu_seqlens MUST live in fixed buffers updated in place — not recreated each step."

**Their numbers.** buffers pre-sized to 257 rows

**llama.cpp — already have it.** The requirement KVarN states — a captured graph bakes in addresses, so per-step data must be written into fixed buffers — is exactly how ggml graph inputs work when the graph is reused: the uid fast path at ggml-cuda.cu:2578-2591 exists precisely because the scheduler did not re-split and the pointers are unchanged. llama.cpp's problem is not pointer stability, it is shape stability (technique 24).

**Equivalent here:** graph input tensors live in the context's persistent input buffer and are refilled in place by set_input_* each step

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:1459-1470` · `src/llama-context.cpp:1332-1372` · `ggml/src/ggml-cuda/ggml-cuda.cu:2578-2591`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Deployment-constant MAX_BLOCKS_PER_REQ to stop Triton recompiling every 128 tokens of context
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2426-2436` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2505-2514`

**What it does.** The build-packed-KV and verify kernels are always launched with `MAX_BLOCKS_PER_REQ = ceil(max_model_len / group)` — a constant for the deployment — instead of a per-step `min(cdiv(max_seq_len, group), block_table.shape[1])`.

**Mechanism.** `max_ctx_blocks = max((self._max_model_len + group - 1) // group, 1)` at 2436 and `max_blocks = (md.fa_max_blocks_per_req or ...)` at 2512.

**Why they needed it.** Line 2429-2431: "the driver forwards it as the MAX_BLOCKS_PER_REQ constexpr, so a per-step min(cdiv(max_seq_len, group), block_table.shape[1]) recompiled the verify kernels every 128 tokens of context." Correctness is preserved because "the kernels bound the block loop by seq_len / vq_seqlen at runtime", and the wasted programs "exit at their first instruction (n_tok <= 0 before any block-table load)".

**Their numbers.** recompile every 128 tokens of context before the fix

**llama.cpp — already have it.** Same principle, already applied, and the source says so: n_kv is padded so 'the graph remains constant across batches and can be reused'. n_ctx is separately padded to 256 and --fit rounds down to 256. There is no kernel recompilation here in any case (AOT build), but the graph-stability motive is identical and llama.cpp already pays the quantisation-to-a-constant price for it.

**Equivalent here:** get_n_kv() rounds n_kv up to a multiple of 256 explicitly so the graph stays constant across batches and can be reused

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:1233-1246` · `src/llama-context.cpp:288` · `common/fit.cpp:344`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none, except as confirmation that context-size churn is already absorbed by the 256 padding and is NOT a candidate explanation for step-to-step variance.

### head_dim > 256 falls off FlashAttention onto SDPA
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2214-2224` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2490-2492` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:340-342`

**What it does.** The backend declares support for head sizes 128, 256 and 512, but every FlashAttention route is guarded by `head_size <= 256`; head_dim-512 layers take a per-request SDPA loop in prefill and `_decode_path_slow` in the cached-multiquery route.

**Mechanism.** `if _HAS_FLASH_ATTN and self.head_size <= 256:` at 2216; `if (not _HAS_FLASH_ATTN or self.head_size > 256 or self._fa_K_buf is None): return self._decode_path_slow(...)` at 2490-2492.

**Why they needed it.** Line 2213-2215: "FlashAttention caps head_dim at 256; the head_dim-512 global layers of Gemma-4 must use the SDPA path (handles arbitrary head_dim). Prefill is a one-time cost (decode dominates at long context), so SDPA here is fine."

**Their numbers.** FA head_dim cap 256; Gemma-4 global layers are head_dim 512

**llama.cpp — already have it.** Same mechanism, wider coverage: llama.cpp accepts DK 40/64/72/80/96/112/128/256 plus the MLA-shaped 192/320/512/576 under gqa conditions, and falls off to the CPU backend otherwise. A Qwen3-family head dim of 128 is squarely supported and takes the ordinary path. The failure mode differs though: KVarN falls to a slower CORRECT GPU path, llama.cpp falls to the CPU, which under -fa on is silent (technique 22).

**Equivalent here:** the head-dimension support table in ggml_cuda_get_best_fattn_kernel; unsupported dims return BEST_FATTN_KERNEL_NONE and the op is scheduled on the CPU backend

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:392-437` · `ggml/src/ggml-cuda/fattn.cu:435-437` · `ggml/src/ggml-cuda/fattn.cu:586-588`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none for this model.

### Slow multi-query fallback builds an explicit position mask in fp32
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2334-2367`

**What it does.** `_decode_path_slow` reconstructs each request's full fp16 K/V through `_gather_request_kv`, promotes q/K/V to fp32, and builds an explicit `k_pos <= q_pos` boolean mask offset by `cached_len = seq_len - (q_end - q_start)` before calling SDPA with GQA enabled.

**Mechanism.** Lines 2358-2365: `q_pos = arange(q_end-q_start).unsqueeze(1) + cached_len`, `k_pos = arange(seq_len).unsqueeze(0)`, `mask = k_pos <= q_pos`, then `F.scaled_dot_product_attention(..., attn_mask=mask, enable_gqa=self.num_kv_heads < self.num_heads)`.

**Why they needed it.** Kept "for correctness in edge cases" (line 2312) — head_dim > 256, missing FA, or a batch too large for the capped materialize scratch. Its cost is recorded at 2460-2465: "the per-request Python gather (per-block .item() syncs + Python dequant + fp32 SDPA, per layer, per step) made MTP decode unusably slow (< 5 tok/s) and its transient fp32 materializations inflated the CUDA-graph memory estimate by GiBs, collapsing the derived KV-cache capacity."

**Their numbers.** < 5 tok/s MTP decode on this path; CUDA-graph memory estimate inflated by GiBs

**llama.cpp — already have it.** llama.cpp's fallback attention is exactly this shape: an explicit additive F32 mask built host-side per ubatch and applied around a plain matmul+softmax, used whenever FA is unavailable (including the kq_b / ALiBi-bias case which FA refuses outright). It is a correctness path, same as KVarN's, and it is likewise slow.

**Equivalent here:** the non-FA path: kq_mask is F32 when flash_attn is off (F16 when on), and soft_max applies it explicitly

**Evidence (llama.cpp):** `src/llama-graph.cpp:38` · `src/llama-graph.cpp:789` · `src/llama-graph.cpp:2540-2542`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none as a change — but it is the path you land on if the -fa auto probe turns FA off, and with a quantised V cache that is not a slow fallback, it is an init failure (src/llama-context.cpp:463-467). Good: the instrument fails loudly rather than returning a plausible number.

## impossible here — 2

### Sinkhorn-like iterative variance normalisation before RTN, with a fused K+V launch only when the tile is square
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:101-127` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1835-1837`

**What it does.** `_sinkhorn_pack_kv` takes the K tiles `[N, D, group]` and V tiles `[N, group, D]`, runs `cfg.sinkhorn_iters` rounds of row/column variance balancing in Triton, and hands the balanced tile plus the extracted row scale and column scale to the int4 packer. When `head_dim == group` (the 128/128 case) K and V tiles have identical `[R, C]` shape, so they are concatenated on dim 0 and balanced in ONE Triton launch; when they differ (head_dim 256 with group 128) they are balanced in two separate launches.

**Mechanism.** Line 111 `if K_tiles.shape[1:] == V_tiles.shape[1:]` selects the fused branch: `kvarn_sinkhorn_triton(torch.cat([K_tiles, V_tiles], dim=0), iterations=cfg.sinkhorn_iters)` returns `(bal, sc, sr)` which is then sliced `[:nk]` / `[nk:]` back into K and V before `kvarn_store_tile_k_batch_from_sinkhorn(..., bits=cfg.key_bits)` and `..._v_batch_from_sinkhorn(..., bits=cfg.value_bits)`.

**Why they needed it.** The split is a bug fix documented inline (lines 107-110): "kvarn_sinkhorn_triton takes R, C as per-launch constexpr — so K and V must be balanced in SEPARATE launches. (A single torch.cat here assumed square and broke at head_dim=256.)"

**llama.cpp — impossible here.** Sinkhorn balancing produces per-row AND per-column scale vectors that must be stored alongside the packed data and re-applied at read. ggml quant formats are fixed structs with one (or one+min) scale per 32 elements and no room for a second axis of scales, and the KV cache is a plain ggml tensor of such blocks — there is nowhere to put s_row/s_col. Adding them means a new ggml_type plus new dequant paths in every FA kernel. The one adjacent seam that IS real: attn_rot_hadamard is built on the HOST at src/llama-kv-cache.cpp:343-357 and uploaded as a plain matrix, so a FIXED diagonal (a calibration-derived per-channel scale) could be folded into that matrix in a small patch — the matrix would stop being orthonormal, so the un-rotation at src/llama-graph.cpp:2814 would need the inverse instead of the same matrix. That is a different technique from Sinkhorn (static, not per-tile, not iterative) and I will not label it as this one.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:343-357` · `ggml/include/ggml.h:390-433` · `src/llama-kv-cache.cpp:1418-1456`

**Effort:** new-backend · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** unknown, and not reachable without new kernels. The static-diagonal near-neighbour is a genuine small-patch idea but nothing in the map or the source says it would help a 128-dim Qwen head.

### Sink blocks: every request's first 128 tokens stay fp16 forever
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1212-1216` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:728-752` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:786-792`

**What it does.** `block_table[r][0]` — the first cache block of each request — is marked a sink and is NEVER quantised. It keeps its pool slot for the request's lifetime, and the walk-back flush loop explicitly refuses to descend to k=0. This is a straight memory-for-accuracy trade: one full fp16 128-token tile per request per layer is held permanently.

**Mechanism.** Sink marking runs in `build()` step (1), lines 739-752: for each row, `s0 = block_table_np[b, 0]`; if already in `sinks` it is re-added to `blocks_needed` (so it keeps its slot), else if it is being written this step it is added to `sinks` and `is_sink_t[s0] = True`. The flush walk (line 808) breaks on `bid in sinks`, and its `while 1 <= k` bound never reaches index 0.

**Why they needed it.** Attention-sink tokens carry disproportionate mass; quantising them to 4 bits is where the scheme would visibly break. The comment at 728-738 also records the failure mode of getting this wrong: re-marking a sink whose data was already flushed "would allocate an EMPTY pool slot that is never written... and attention would read garbage for the whole first block — the issue #10 repetition-loops on multi-turn chat."

**llama.cpp — impossible here.** Mixed precision by cache REGION has no expression in llama.cpp: cparams.type_k / type_v are set once for the whole context and every layer's cache tensor is created with that one type. Even if you carved out a second region, this binary's FA kernel refuses K->type != V->type (ggml/src/ggml-cuda/fattn.cu:442-446) let alone two types within K, so a single attention call cannot read two precisions. Note llama.cpp does have an `attention sinks` concept in the graph (the `sinks` argument to build_attn) but that is learned per-head sink logits, an entirely different thing from keeping the first N tokens' KV in fp16.

**Equivalent here:** none — type_k/type_v are single context-wide values

**Evidence (llama.cpp):** `common/common.cpp:1727-1728` · `src/llama-kv-cache.cpp:206-217` · `ggml/src/ggml-cuda/fattn.cu:442-446`

**Effort:** new-backend · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** unknown quality gain, and unreachable. Mentioned only because it is the single highest-leverage accuracy idea in this slice that llama.cpp structurally cannot express.

## not applicable — 23

### Per-tile page layout: one contiguous 17920-byte record per (block, kv-head)
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:10-12` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:265-313` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1724-1756`

**What it does.** A cache page is not per-token. For head_dim=128, group=128, 4-bit K and 4-bit V, one (block, kv-head) record is 17920 bytes laid out as: 8192 B packed K + 256 B K s_col + 256 B K zp + 256 B K s_row, then 8192 B packed V + 256 B V s_col + 256 B V s_row + 256 B V zp. Each 256-byte field is 128 fp16 values. The KV-cache tensor shape is therefore 3D: `(num_blocks, num_kv_heads, tile_bytes_aligned)`.

**Mechanism.** `get_kv_cache_shape` (line 265) asserts `block_size == cfg.group` (line 310) and returns `(num_blocks, num_kv_heads, cfg.tile_bytes_aligned)` (line 313). `_flat_block` (line 1715) takes `kv_cache[block_id, head]` as a contiguous uint8 row. `_write_packed` (line 1724) writes each field by byte offset, `.view(torch.float16)` reinterpreting the fp16 scale arrays in place: K gets `s_col_K` (D×2 B), `zp_K` (D×2 B), `s_row_K` (group×2 B); V gets `s_col_V` (D×2 B), `s_row_V` (group×2 B), `zp_V` (group×2 B).

**Why they needed it.** Docstring at line 275-277: "Unlike TurboQuant's per-token slot, KVarN's scales are tile-shared, so one block per head is a single 17920-byte record. The natural shape is therefore (num_blocks, num_kv_heads, tile_bytes_aligned) — no leading 2 (K and V share the record), and no per-position dim."

**Their numbers.** 17920 B per (block, head) at head_dim=128, group=128, k_bits=4, v_bits=4 — 8192 + 3×256 for K, 8192 + 3×256 for V (lines 10-12)

**llama.cpp — not applicable.** llama.cpp has no page/record concept. Each layer gets one k tensor and one v tensor whose buffer type follows that layer's weights (src/llama-kv-cache.cpp:209-217), and addressing is by cell index with n_kv padded to 256. Scales live inside the ggml block struct, not in a side field, so there is no record to lay out and no byte-offset table to maintain.

**Equivalent here:** the KV cache is a flat per-layer ggml tensor of block-quant rows, sized and placed per layer

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:206-217` · `src/llama-kv-cache.cpp:1233-1246`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Page size chosen so vLLM's byte accounting works unchanged
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:278-283`

**What it does.** The total bytes per block, `num_kv_heads * tile_bytes_aligned`, is deliberately made to equal `block_size * num_kv_heads * slot_size` as computed by vLLM's `TQFullAttentionSpec.page_size_bytes`, by defining `slot_size = tile_bytes / block_size` — i.e. a fictitious per-token size that is really 1/128th of a tile.

**Mechanism.** Documented as an invariant in `get_kv_cache_shape`'s docstring rather than enforced by code: "The total bytes per block (= num_kv_heads * tile_bytes_aligned) equals block_size * num_kv_heads * slot_size from TQFullAttentionSpec.page_size_bytes when slot_size = tile_bytes / block_size, so vLLM's memory accounting works unchanged."

**Why they needed it.** KVarN has no per-token storage at all, but vLLM's allocator and its KV-capacity derivation are per-token; making the arithmetic agree avoids patching the allocator.

**llama.cpp — not applicable.** This is a workaround for vLLM's per-token allocator arithmetic. llama.cpp sizes the cache directly from n_ctx x n_embd_k_gqa x sizeof(type) and lets --fit probe actual free memory via llama_get_memory_breakdown, so there is no fictitious slot_size to reconcile.

**Equivalent here:** n/a — --fit measures real free VRAM and real memory breakdowns instead of deriving capacity from a per-token size

**Evidence (llama.cpp):** `common/fit.cpp:29-70` · `common/fit.cpp:559-563`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### One vLLM kernel block == one KVarN tile, advertised as a single supported size
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:206-238` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:499-510`

**What it does.** `get_supported_kernel_block_sizes` returns a ONE-element list — the active preset's group — instead of every preset's group, and `get_preferred_block_size` returns the group rather than falling back to the framework minimum. The metadata builder then asserts that the kernel block size it was handed equals the tile group.

**Mechanism.** Both call `_active_kvarn_group()` (line 160), which resolves the active `kvarn_*` cache-dtype string and looks up `KVARN_PRESETS[cache_dtype]["group"]`. The builder's assertion is at lines 506-510: `assert _spec_bs is None or _spec_bs == self._group`.

**Why they needed it.** Stated at lines 209-214: "0.27.1's select_common_block_size (worker/utils.py) picks the LARGEST int size dividing the (hybrid-aligned) manager block, so advertising every preset's group would hand a g64 preset a 128-token kernel block and trip the block_size == group assert." And at 227-232: the generic fallback "returns the MINIMUM supported size (64) whenever the framework default (16) is unsupported — which breaks any g128 preset run without an explicit --block-size".

**Their numbers.** supported presets: kvarn_k4v4_g128, kvarn_k4v2_g128, kvarn_k4v4_g64, kvarn_k4v2_g64 (lines 194-199); supported head sizes 128, 256, 512 (line 342)

**llama.cpp — not applicable.** There is no block-size negotiation between llama.cpp and its kernels — no framework picks a block size to hand the backend. get_n_kv() rounds up to 256 unconditionally, which is what makes K->ne[1] % FATTN_KQ_STRIDE == 0 true and hence keeps the vector kernel eligible. Nothing to advertise, nothing to assert.

**Equivalent here:** the closest fixed granularity is FATTN_KQ_STRIDE = 256, the n_kv padding, and it is not negotiable

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-common.cuh:9` · `src/llama-kv-cache.cpp:1233-1246`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### fp16 tail pool: partial tiles are never quantised, they live in a fixed-size fp16 side buffer
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1218-1232` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1327-1350` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1036-1048`

**What it does.** Each layer owns a `[POOL_SIZE, group, num_kv_heads, head_dim]` fp16 K pool and an identical V pool. A block that is not yet 128 tokens full lives entirely in that pool as fp16; only when the tile is complete AND all its tokens are committed does it get Sinkhorn-quantised into the int4 record. A per-group allocator maps `block_id → slot`, with a GPU mirror so the store kernel can do the lookup without a host round-trip.

**Mechanism.** `_ensure_pool` line 1343: `torch.zeros(pool_size, cfg.group, self.num_kv_heads, cfg.head_dim, dtype=torch.float16, device=device)` plus `torch.zeros_like` for V. `pool_size = cfg.pool_slots(max_num_seqs, max_num_batched_tokens)` unless `KVARN_POOL_SLOTS` pins it (`max(env_slots, 64)`, line 1338). The free list is initialised descending: `cls._free_slots[gk] = list(range(pool_size - 1, -1, -1))` (line 1358). GPU mirrors `_block_to_slot_t_per_device` (int32, -1 = no slot) and `_is_sink_t_per_device` (bool) are keyed `(device, group_key)` and sized `max(num_blocks_hint, max_known_block_id+1, 1024)` (line 1367).

**Why they needed it.** Quantising an incomplete tile would mean re-quantising it on every appended token, and the per-tile scales would be computed from a fraction of the tokens. Pool sizing is justified at lines 1329-1335: "Size the pool to the structural peak for the *capped* concurrency: sink + in-progress tail per active request... making the pool both exhaustion-safe (the scheduler can never exceed it) and OOM-safe."

**Their numbers.** pool sized ~2 × max_num_seqs ("covers sink + in-progress tail for the largest captured batch", lines 1307-1309); KVARN_POOL_SLOTS floor 64

**llama.cpp — not applicable.** KVarN needs the pool because its scales are shared across a 128-token tile, so an incomplete tile has no valid scale. llama.cpp quantises each token's K and V independently at cpy_k/cpy_v time in 32-channel blocks; every write is a complete, self-describing block. The whole class of pool/slot/allocator problems (techniques 8-20) does not exist here as a consequence of that one design choice.

**Equivalent here:** n/a — quantisation granularity is one token's head vector, so there is no partial tile

**Evidence (llama.cpp):** `src/llama-graph.cpp:2795-2801` · `src/llama-kv-cache.cpp:206-217`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none — and this is worth stating plainly, because roughly a third of the KVarN backend's complexity is bookkeeping that llama.cpp's per-token granularity makes unnecessary.

### Backward walk-back flush that stops at the first slotless block
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:786-815`

**What it does.** Instead of tracking which blocks are dirty, the builder walks each request's block-table row BACKWARD from the committed boundary and queues every block that still holds a pool slot, stopping at the first block that does not. Slot-holding + below-the-committed-boundary is used as the exact definition of "full but unflushed".

**Mechanism.** Lines 808-815: `while 1 <= k:` → `if (bid < 0 or bid in flush_seen or bid in sinks or bid not in dict_map): break` else append and `k -= 1`. `dict_map` membership is the ground truth because "a flush frees the slot" (line 666-668).

**Why they needed it.** Lines 791-793: "Idempotent under prefix sharing: a co-owner finds the block already queued (or slotless) and stops — no per-request state to collide." It avoids any per-request dirty bookkeeping, which is the exact thing that collides when vLLM shares physical blocks between requests.

**llama.cpp — not applicable.** A consequence of technique 8's absence: with no deferred flush there is no dirty-block set to discover, and no shared-physical-block collision to make idempotent. llama.cpp writes each token's KV once, at the step that produces it.

**Equivalent here:** none needed

**Evidence (llama.cpp):** `src/llama-graph.cpp:2795-2801`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Ordering constraint: mark sinks → flush → allocate, to cap the live-slot peak at 2·B
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:720-726`

**What it does.** The builder deliberately flushes (freeing slots) before allocating the new tails, so the peak number of simultaneously-held pool slots is one sink + one in-progress tail per request.

**Mechanism.** Comment block at 720-726 plus the code order: (1) sink marking at 739, (2) flush detection at 794, (2b) reclaim at 827, (3) allocation at 879.

**Why they needed it.** Verbatim: "Doing the flush before allocation caps the live-slot peak at 2·B (one sink + one in-progress tail per request). Allocating first would transiently need 3·B when every request crosses a block boundary in lockstep (sink + pending-flush full block + new tail) → 'pool exhausted' at large batch."

**Their numbers.** 2·B vs 3·B slots

**llama.cpp — not applicable.** No pool, no slots, no peak to cap. The KV cache is preallocated at context creation to n_ctx cells and never grows.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:206-217`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Reclaim policy split by completeness: complete → flush, partial → discard, complete sink → retire
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:817-843` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:456-466`

**What it does.** Slot-holding blocks that were neither written this step nor already queued belong to finished or preempted requests. A COMPLETE non-sink block is flushed to int4 (because vLLM's prefix cache may hand it to a future request that must find a valid tile), a COMPLETE sink is retired fp16-resident, and a PARTIAL block is simply discarded.

**Mechanism.** `full = self._block_fill.get(bid, 0) >= GROUP` at line 831 drives the three-way branch. `_block_fill` is maintained during the blocks-needed scan at line 693: `self._block_fill[bid] = min(sl, (k+1)*GROUP) - k*GROUP`.

**Why they needed it.** Line 824-826: "vLLM's prefix cache may hand it to a future request, which must find a valid int4 tile (the old discard left stale tile bytes). A PARTIAL block is discarded: vLLM never prefix-caches partial blocks."

**llama.cpp — not applicable.** Same root cause as 8/12/13: nothing is ever half-quantised, so there is no completeness to branch on.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:206-217`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Fill tracking keyed by PHYSICAL block id, never by request
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:456-466`

**What it does.** `_block_fill` maps physical `block_id → tokens present in the pool for that block after the current step`, and is explicitly not keyed by request or by sink identity.

**Mechanism.** A plain `dict[int, int]` on the builder, written at line 693 and popped on flush (865), discard (874), and eviction (895).

**Why they needed it.** Quoted: "Keyed by PHYSICAL block (never by request or by the sink block id): vLLM's prefix caching shares physical blocks across live requests and recycles ids across finished ones, so any request-identity proxy collides under sharing (the issue #10 repetition-collapse / stale-tile class). A partial block has exactly one writer, so the value has a single source."

**llama.cpp — not applicable.** The hazard being defended against — a request-identity proxy colliding when physical blocks are shared — cannot arise, because llama.cpp's unit of ownership IS the physical cell and it stores its own seq membership. There is no derived per-request bookkeeping to get wrong. At -np 1 there is one sequence anyway.

**Equivalent here:** cells carry seq_id sets; sharing is handled by the cell, not by a side table

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:79-153`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Vectorised batched flush: one index_select gather + one index_copy scatter per (layer, block-chunk)
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1853-1938` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:849-860`

**What it does.** All (layer, block) tiles due for flush in one step are batched: pool rows are gathered with one `index_select` per chunk, reshaped into `[nB*Hk, D, G]` K tiles and `[nB*Hk, G, D]` V tiles, Sinkhorn-balanced and packed in one call, the eight fields concatenated into a `[nB*Hk, tile_bytes]` record, padded to `tile_bytes_aligned`, and written back with one indexed assignment.

**Mechanism.** `_batched_flush` line 1854. Chunking: `CHUNK_BLOCKS = max(1, 2048 // max(Hk, 1))` (line 1899) "so one Sinkhorn launch stays bounded (~2k [R,C] tiles)". Record assembly at 1924-1936 concatenates in config-offset order, with fp16 scale tensors byte-reinterpreted via `.contiguous().view(torch.uint8)`, then `F.pad(rec, (0, T - rec.shape[1]))`. Write-back is a single `kvc[bid_t] = rec.view(nB, Hk, T)` (line 1938).

**Why they needed it.** Lines 1859-1865: the per-(layer, block, head) Python loops "exploded into ~10^5 tiny GPU ops on a synchronized burst (prefill completion, lockstep decode boundary) and dominated build() at high concurrency (issue #15: ~44 ms/step at B=256)". Accuracy is explicitly unchanged: "Numerically identical: same Sinkhorn, same RTN/pack math, same byte layout; only the data movement is batched."

**Their numbers.** issue #15: ~44 ms/step at B=256 before the change; ~10^5 tiny GPU ops on a synchronised burst; CHUNK_BLOCKS bounds a launch to ~2048 tiles

**llama.cpp — not applicable.** No flush exists to batch. The general principle — never emit per-(layer, block, head) tiny GPU ops — is already the ggml idiom: the KV write is a single indexed scatter per layer driven by a host-built index tensor.

**Equivalent here:** n/a; the analogous 'one indexed op instead of a loop' pattern is already how cpy_k/cpy_v work (ggml_set_rows with an index tensor)

**Evidence (llama.cpp):** `src/llama-graph.cpp:2795-2801` · `src/llama-kv-cache.cpp:1459-1470`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Legacy flush path chunked at 256 pairs to bound transient fp32 memory
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1940-2022`

**What it does.** The pre-vectorisation flush (still reachable via `KVARN_FAST_FLUSH=0`, and used by the tile-dump debug hook) stacks pool tiles into fp32 and processes them in chunks of 256 (layer, block) pairs, with explicit `del` of intermediates between stages.

**Mechanism.** `CHUNK_PAIRS = 256` at line 1956; `K_stack = torch.stack(K_list, dim=0)` then `del K_list, V_list`, `del K_stack, V_stack`, `del K_tiles, V_tiles`, `del K_out, V_out`.

**Why they needed it.** Lines 1950-1952: "at peak (48 layers × ~73 lockstep reqs = ~3.5k pairs), the unchunked stack hits >2 GB of fp32 working memory and OOMs on a memory-tight burst."

**Their numbers.** 48 layers × ~73 lockstep requests ≈ 3.5k pairs → >2 GB fp32 working memory unchunked

**llama.cpp — not applicable.** No flush path to chunk. llama.cpp bounds its own transient working memory structurally: the graph is reserved at a fixed worst-case shape and ggml-alloc sizes one arena from that, so there is no unbounded stack-everything-then-process step that could OOM on a burst.

**Equivalent here:** the analogous transient-memory bound is the compute buffer sized by the reserve passes at n_tokens = min(n_ctx, n_ubatch)

**Evidence (llama.cpp):** `src/llama-context.cpp:576-671` · `src/llama-context.cpp:595`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none directly. The one live lever with the same shape is -ub: it is the single knob that sizes the worst-case compute buffer, and on 12 GB that competes with KV.

### Pool slot is NOT freed by the flushing layer — it is freed once, by the builder, after every layer has flushed
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1845-1851` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:861-869`

**What it does.** A slot index addresses the same row in every layer's pool, so `_flush_tail` deliberately leaves the block→slot mapping intact; the builder frees the slot once after cross-producting every flush block with every impl.

**Mechanism.** Note at 1845-1851 in `_flush_tail`; the actual free at 863-869 pops from `dict_map`, appends to `free_slots`, and sets `b2s_t[bid] = -1`.

**Why they needed it.** Verbatim: "Freeing here would let layer 0's flush drop the slot, after which layers 1..N find no slot (`.get()` → None) and silently skip writing their int4 — corrupting all-but-the-first layer's history."

**llama.cpp — not applicable.** There is no shared per-layer slot to free. Each layer owns its own k/v tensors and indexes them with the same cell index, which is owned by the cache, not by any layer.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:206-217`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Slow reference read path: full dequant + un-rotate, with a silent zero fallback for slotless tails
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1758-1811` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2244-2300`

**What it does.** `_read_block_dequantized` reconstructs one block per kv-head by viewing the packed bytes as `[D, group//pack_k]` for K and `[group, D//pack_v]` for V, calling the dequant helpers, transposing and multiplying by H to undo the rotation. `_gather_request_kv` stitches a request's context: pool blocks are un-rotated with `x.float() @ H.T`, flushed blocks go through the dequant, and a tail block with NO pool slot is filled with ZEROS.

**Mechanism.** Pack widths are derived, not assumed: `pack_k = 8 // cfg.key_bits` (line 1783), `pack_v = 8 // cfg.value_bits` (line 1800). Un-rotate for K is `K_rot_DG.T @ H` (line 1792), for V is `V_rot_GD @ H` (line 1808), and for pool data `(x.float() @ H.T)` (line 2262). Zero fallback at lines 2290-2294.

**Why they needed it.** The pack-width derivation is a recorded bug fix: "The old fixed D // 2 assumed 4-bit V and broke k4v2 (view size mismatch), which is why this slow gather path had never worked for the default preset" (lines 1796-1799). Same for K at 1780-1782: "group // 2 only happens to be right for 4-bit K." The zero fallback is an unannotated accuracy hazard — a tail block with no slot silently contributes zero K/V rather than raising.

**llama.cpp — not applicable.** There is no Python-level gather to fall back to and no slotless-tail failure mode. llama.cpp's fallback is coarser and, per the map, more dangerous in one specific way: with -fa on (not auto) an unsupported combination is not detected, ggml_backend_cuda_device_supports_op returns false and the attention node is silently scheduled on the CPU backend, producing a working-but-enormously-slower server with no error. That is llama.cpp's own version of the 'plausible wrong result' hazard KVarN's zero-fallback represents.

**Equivalent here:** the CPU backend accepting FLASH_ATTN_EXT when no CUDA kernel matches

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:435-437` · `ggml/src/ggml-cuda/fattn.cu:586-588` · `src/llama-context.cpp:554-557`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Leave -fa at auto so the probe runs and prints 'Flash Attention enabled' / 'not supported, set to disabled'. The startup line that echoes the REQUESTED flash_attn type (src/llama-context.cpp:312) is not the resolved one and must not be quoted as evidence.

### Zero the vq plan tail on padded uniform batches
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:971-977`

**What it does.** When a graph-captured uniform batch is padded, `num_decode_tokens` exceeds the sum of real query lengths; the leftover rows of the pinned plan buffers are explicitly zeroed before the H2D copy.

**Mechanism.** `if i < num_decode_tokens: self._vq_seqlen_host[i:num_decode_tokens] = 0; self._vq_req_host[i:num_decode_tokens] = 0`.

**Why they needed it.** Verbatim: "zero the tail so the H2D copy below never ships garbage from the torch.empty pinned buffers into the kernel's per-row plan."

**llama.cpp — not applicable.** llama.cpp's graph inputs are filled exactly to n_tokens each step; there is no pinned plan array carried across padded captures whose tail could ship garbage. This hazard is created by technique 24 and would only appear here if 24 were implemented.

**Equivalent here:** none — there is no padded per-row plan buffer

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:1459-1470`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none today; it is the bug you would have to defend against if you ever built uniform-width capture.

### Flush-kernel JIT warmup at pool-init time
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1391-1408`

**What it does.** The Sinkhorn + int4-store kernels are compiled once at pool-init on a 1-tile dummy of the exact flush shapes, keyed by `(device, head_dim, group, key_bits, value_bits)`.

**Mechanism.** Lines 1403-1407: `k_dummy = torch.zeros(1, cfg.head_dim, cfg.group, ...)`, `v_dummy = torch.zeros(1, cfg.group, cfg.head_dim, ...)`, then `_sinkhorn_pack_kv(k_dummy, v_dummy, cfg)`.

**Why they needed it.** Lines 1391-1400: those kernels "are exercised ONLY at a tile-boundary flush, which never happens during vLLM's profiling/dummy run (no request crosses a block boundary there). So they JIT-compile on the FIRST real flush DURING serving — a multi-hundred-ms stall that surfaces as a latency spike and a `jit_monitor` 'JIT compilation during inference' warning, and disproportionately hurts low-concurrency aggregate throughput (the one-time cost lands inside a small measured window)."

**Their numbers.** multi-hundred-ms first-flush stall without the warmup

**llama.cpp — not applicable.** This build compiles a fixed kernel set for compute_89 ahead of time (only the four vec instances f16/q4_0/q8_0/bf16 exist, per the map's CMake evidence). There is no JIT, so there is no first-use compile stall to hide and no 'JIT during inference' failure mode. The corresponding llama.cpp cost is the CUDA GRAPH warmup, which is two calls, not hundreds of milliseconds.

**Equivalent here:** none needed — ggml CUDA kernels are AOT-compiled into the binary

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/CMakeLists.txt:115-125` · `ggml/src/ggml-cuda/fattn.cu:284-290`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none, but it removes a whole class of first-measurement artefact from our benchmarking worries — a latency spike on the first flush-equivalent cannot happen here.

### Split-K partial buffers sized by the split-K REGIME, not by max_num_seqs
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1489-1516`

**What it does.** The fp32 split-K partial buffers `mid_o [rows, splits, D]` and `mid_lse [rows, splits]` are sized to `(sm_count // Hk) * Hq` rows rather than `max_num_seqs * Hq`, because the driver only chooses split-K when `B*Hk <= sm_count`.

**Mechanism.** `_splits = adaptive_num_kv_splits((max_model_len + group - 1) // group)` at 1500; `mid_rows = max((_sm // max(Hk,1)) * Hq, Hq, 1)` at 1512; the SM count is cached per impl at 1323-1325 (`torch.cuda.get_device_properties(device).multi_processor_count`) so "the decode / verify drivers" don't "re-query cudaGetDeviceProperties on every call".

**Why they needed it.** Two stacked regressions: lines 1491-1495 "sized the buffer ~85x too big at typical configs: 256 MiB instead of ~3 MiB for max_num_seqs=2/Hq=24/64 splits; issue #10 follow-up", and lines 1505-1509 "Sizing to max_num_seqs*Hq over-reserved this fp32 partial buffer ~10-20x at high concurrency, where it competes directly with the int4 KV cache." The driver keeps a defensive fallback to single-stage if N ever exceeds the rows.

**Their numbers.** ~85× over-allocation → 256 MiB instead of ~3 MiB at max_num_seqs=2, Hq=24, 64 splits; a further ~10-20× over-reservation at high concurrency from the max_num_seqs sizing

**llama.cpp — not applicable.** llama.cpp's FA scratch is allocated per call through ggml_cuda_flash_attn_ext_get_alloc_size and charged to the compute buffer, sized from the actual tensor shapes rather than from a concurrency parameter — so the over-allocation failure mode KVarN fixed (sizing by max_num_seqs when the kernel only fires at low batch) has no analogue. I could not verify the split-k/fixup buffer's exact sizing from the map; see map_gaps.

**Equivalent here:** launch_fattn sizes its own parallel-block / fixup scratch internally; no user knob

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-common.cuh:1022-1084` · `ggml/src/ggml-cuda/fattn.cu:534-568`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** unknown

### Builder owner registry so MTP's two builders don't both drive the draft layer
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:417-432` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:525-546` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:559-565`

**What it does.** A class-level `_owner: dict[layer_name, builder]` decides which builder may allocate slots for, flush, and re-tag each layer's impl. A layer is claimed if unowned, already owned by the claimant, or the claimant has no MORE layers than the incumbent — so the drafter's 1-layer builder wins the draft layer.

**Mechanism.** Claim loop at 535-544 resolves impls via `get_layers_from_vllm_config(vllm_config, Attention, self._layer_names)` (builder-side discovery, no `attention.py` patch), then applies the size rule. `_owned_impls()` (line 562) re-derives the authoritative set at build time "because a later-constructed builder (the drafter's) may take a layer over".

**Why they needed it.** Lines 424-427: "Without single ownership both builders re-tag impl._group_key and flush the draft impl from different slot maps -> draft KV silently wrong -> lower acceptance."

**llama.cpp — not applicable.** Ownership ambiguity between a target-side and a draft-side manager cannot arise: the draft context builds its own memory except for the specific sidecar cases (GEMMA4_ASSISTANT, or EAGLE3/DFLASH sidecars shipped without tok_embd/output) where ctx_other is honoured and the target's memory is shared deliberately. Worth flagging that draft-dflash CAN land in that shared-memory case depending on how the sidecar was packaged — the branch is at src/llama-context.cpp:145-161.

**Equivalent here:** the draft always gets its own llama_context and its own memory module

**Evidence (llama.cpp):** `common/speculative.cpp:2464-2482` · `common/speculative.cpp:2460-2461` · `src/llama-context.cpp:142-161`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** unknown, and worth one look at the startup log: whether our dflash sidecar shares the target's memory or allocates its own changes the VRAM arithmetic and whether -ctkd/-ctvd matter at all.

### Drafter passes >= 1 allocate slots but never flush
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:573-590` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:796-798` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:827-830`

**What it does.** `build_for_drafting` passes `_allow_flush=(draft_index == 0)`. On later draft passes the builder allocates pool slots for the blocks the drafter writes but skips flush detection and reclaim entirely.

**Mechanism.** The flush loop at 798 iterates `range(B) if _allow_flush else ()`; the reclaim comprehension at 828-830 is likewise empty when `_allow_flush` is False.

**Why they needed it.** Lines 580-586: on `draft_index >= 1`, "`committed` OPTIMISTICALLY counts earlier draft tokens (rejectable next step) as final: flushing on that could freeze a block holding a to-be-rejected draft token into int4 for the draft layer, and the rewrite after rejection would re-slot the block with an empty pool slot (its history then reads as zeros)."

**llama.cpp — not applicable.** No flush and no shared slot pool between drafter and target, so the optimistic-committed-length hazard this guards against does not exist. The draft's own KV is in its own cache and is rewritten freely.

**Equivalent here:** none

**Evidence (llama.cpp):** `common/speculative.cpp:2464-2482`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Per-group state reset on builder (re)construction, because vLLM builds the KV cache twice
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1119-1157` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:545-546`

**What it does.** Every builder construction drops the block→slot map, free list, pool size, sink set, max-known-block-id and the GPU mirrors for its group, and per impl clears `_tails`, `_sink_blocks`, `_kv_cache_ref`, and the mirror pointers. Pool tensors, shared scratch and the compiled-kernel registry are deliberately KEPT.

**Mechanism.** `_reset_group_state(group_key, impls)` at 1120, called from `__init__` at 545-546.

**Why they needed it.** Lines 1126-1135: "In 0.27.1 that happens on every initialize_kv_cache — first for the minimal PROFILING KV cache (used for the CUDA-graph memory estimate) and again for the REAL one — and _cleanup_profiling_kv_cache never clears KVarN's cached impl._kv_cache_ref (it only knows _k_scale_cache/_v_scale_cache). So at builder construction the block->slot map, sink set, GPU mirrors, per-impl tail trackers and the cached kv_cache reference all describe a KV cache (block-id space) that no longer exists."

**llama.cpp — not applicable.** llama.cpp does build a context twice in a sense — once for measurement with no_alloc, once for real — but the probe allocates nothing and shares no cached state with the live context, so there is no stale-reference class of bug. The real caching hazard here is a different one the map already flags: --fit's mutations to cparams->n_ctx survive even when its layer-placement pass throws (common/fit.cpp:803-810).

**Equivalent here:** --fit's probe loads the model with no_alloc=true and builds a throwaway context, then the real load happens once

**Evidence (llama.cpp):** `common/fit.cpp:29-70` · `common/common.cpp:1294-1302`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none as a technique. The related live trap: adding -ot / -cmoe / -ncmoe to our profile silently converts --fit from 'fit layers and ctx' into 'reduce ctx only', with a WARN and no failure.

### block_table read as a numpy array rather than a nested Python list
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:621-629` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1018`

**What it does.** `build()` converts the block table once with `cam.block_table_tensor.cpu().numpy()` and indexes it lazily; the metadata's `block_table_cpu` field is left None.

**Mechanism.** Line 626 `block_table_np = cam.block_table_tensor.cpu().numpy()`, with `bt_rows`/`bt_cols` guards used everywhere the table is walked.

**Why they needed it.** Verbatim: "the full B×max_blocks nested-list build was ~7 ms/step at B=256 and dominated build() once the flush was vectorized (issue #15). We only touch column 0 (sinks) + a few per-request entries, so numpy's O(1) indexing avoids materializing ~8k Python ints every step."

**Their numbers.** ~7 ms/step at B=256; ~8k Python ints per step avoided

**llama.cpp — not applicable.** A Python-object-materialisation cost with no counterpart in a C++ implementation.

**Equivalent here:** n/a — C++

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:1459-1470`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### bf16 → fp16 boundary cast, justified by the 4-bit cache
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2145-2154` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:2053-2056`

**What it does.** All bf16 activations entering KVarN (query, key, value in `forward`; key/value in `do_kv_cache_update`) are cast to fp16 at the entry point, and the output is cast back to `output.dtype` on the way out. fp16 inputs pass through byte-identical.

**Mechanism.** `if query.dtype != torch.float16: query = query.to(torch.float16); key = ...; value = ...` at 2151-2154; output cast at 2183/2185 (`.to(output.dtype)`).

**Why they needed it.** Verbatim: "KVarN's compute (rotation matmul, scratch buffers, Triton stores) is fp16 internally... the cast is lossless for KVarN (fp16 mantissa > bf16, and the cache is 4-bit). Without this, bf16 q mixing with fp16 KV buffers trips 'Expected out BFloat16, got Half'." Note this is a range trade, not only a precision one — the argument covers mantissa but not bf16's wider exponent.

**llama.cpp — not applicable.** No dtype-boundary friction to resolve: ggml carries types per tensor, the FA node's precision is pinned to GGML_PREC_F32 for accumulation, and bf16/bf16 is one of the four compiled vec instances. Our profile is q4_0/q4_0 so neither path applies.

**Equivalent here:** F32 K/V are cast to F16 inside the graph when FA is used; bf16 is a first-class cache type with its own kernel

**Evidence (llama.cpp):** `src/llama-graph.cpp:2549-2555` · `ggml/src/ggml-cuda/fattn.cu:321-325` · `src/llama-graph.cpp:2562`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### "auto" cache-dtype trap: resolve the active preset instead of raising
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:130-167` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:289-307` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1203-1205`

**What it does.** When `get_kv_cache_shape` is handed a `cache_dtype_str` that is not a KVarN preset, it resolves the real preset from `KVarNAttentionImpl._active_cache_dtype` (recorded when the deployment's impl was constructed) or from the current vLLM config, and only raises if neither exists.

**Mechanism.** `_active_kvarn_cache_dtype()` (line 130) uses `get_current_vllm_config_or_none` "because 0.27.1's get_current_vllm_config() raises outside a set_current_vllm_config context" and falls back to the class attribute. `type(self)._active_cache_dtype = kv_cache_dtype` is stamped at 1205.

**Why they needed it.** Lines 290-295: "gpu_model_runner._reshape_kv_cache_tensors passes cache_dtype_str='auto' whenever spec.kv_quant_mode == KVQuantMode.NONE; the primary fix is KVQuantMode.KVARN (kv_cache_interface.py), but if a caller still hands us a non-KVarN string, resolve the active preset... instead of raising in from_cache_dtype."

**llama.cpp — not applicable.** A vLLM config-plumbing repair. llama.cpp has one cache-type string parsed once from -ctk/-ctv against a nine-name whitelist, and an unknown value aborts startup rather than being resolved from ambient state. Note the whitelist is a parser list only — it accepts iq4_nl and q5_1, which this build has no CUDA FA kernel for, so --help advertises types that cannot run on the GPU.

**Equivalent here:** kv_cache_type_from_str throws on any unrecognised string at argument-parse time

**Evidence (llama.cpp):** `common/arg.cpp:317-324` · `common/arg.cpp:305-315`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Blocks-first physical layout declared so vLLM's generic page padding stays legal
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:315-332`

**What it does.** `get_kv_cache_stride_order` returns identity `(0,1,2)`, or `(1,0,2,3)` with a prepended num_layers dim, declaring that the physical order is blocks-first.

**Mechanism.** Return values at 330-332.

**Why they needed it.** Lines 320-329: "With a prepended num_layers dim the physical order is (num_blocks, num_layers, num_kv_heads, tile), i.e. still blocks-first, so AttentionBackend.indexes_kv_by_block_stride() is True and 0.27.1's generic page padding in unify_kv_cache_spec_page_size (kv_cache_utils.py) is legal for KVarN layers (the kernels read kv_cache.stride(0)/stride(1), so a padded block stride is fine). Not exercised on Qwen3.8 (attention page == padded mamba page after _align_hybrid_block_size); safety net that replaces KVarN's 0.23 kv_cache_utils.py hunk."

**llama.cpp — not applicable.** There is no framework asking the backend to declare a stride order, and no generic page padding to keep legal.

**Equivalent here:** none — layout is fixed by the ggml tensor and its strides

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:206-217`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Capabilities deliberately declined: DCP and per-head quant scales
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1062-1066` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:252-254` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:344-349`

**What it does.** `supports_dcp = False`, `supports_quant_query_input = False`, `supports_per_head_quant_scales()` returns False, and `supports_mm_prefix()` returns True with an explicit correctness caveat.

**Mechanism.** Class attributes and classmethods as cited.

**Why they needed it.** DCP: "AttentionImplBase defaults supports_dcp=True; KVarN's decode kernels return no LSE and know nothing about decode-context-parallel KV sharding" (1063-1066). MM prefix is an honest half-claim: "Multimodal models (e.g. Gemma-4) set use_mm_prefix; text generation never materializes mm tokens so KVarN decode is unaffected. (Image/audio prefix full-attention correctness is unverified — text-only validated.)"

**llama.cpp — not applicable.** Neither capability has a counterpart. Context-parallel decode does not exist in llama.cpp (pipeline parallelism requires more than one device and is inert on a single 4070 SUPER), and ggml block quantisation already carries a scale every 32 elements, so a per-head scale would be coarser, not finer.

**Equivalent here:** n/a — no decode-context-parallel, and quant scales are per-32-element block, finer than per-head

**Evidence (llama.cpp):** `ggml/include/ggml.h:390-433` · `src/llama-context.cpp:427-433`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

## unmatched — 1

### Per-KV-cache-group allocator scoping (block_ids are only unique within a group)
**Where (theirs):** `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1096-1105` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:1184-1192` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:445-455` · `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:698-707`

**What it does.** All allocator state — block→slot dict, sink set, free list, pool size, max known block id — is a dict keyed by a group key; the GPU mirrors are keyed by `(device, group_key)`. The impl bootstraps with a config proxy key `(head_size, num_kv_heads, sliding_window)`, and the builder RE-TAGS its owned impls in `build()` with the true key: the sorted tuple of its exact layer names.

**Mechanism.** Class dicts at 1099-1105. Impl proxy key at 1192. Builder true key at 455: `self._group_key = tuple(sorted(self._layer_names))`, applied at 706-707: `for i in group_impls: i._group_key = gk`. `_ensure_pool` is written to be idempotent under the re-tag (lines 1351-1361).

**Why they needed it.** Two layered reasons. At 1185-1189: "vLLM gives each KV-cache group an INDEPENDENT block_id space. Heterogeneous models put KVarN layers in >1 group (e.g. Gemma-4: sliding head256/16kv + global head512/4kv), so a single global allocator aliases the two groups' block_ids -> wrong slots -> garbage." At 449-452: "A config proxy (head,kv,sw) is NOT enough: vLLM splits same-config layers into multiple groups (Gemma-4's repeating pattern -> 5 sliding groups all head256/16kv/1024), each with its own block_id space."

**Their numbers.** Gemma-4: 5 sliding groups all head256/16kv/window-1024

> **No verdict returned for this technique.** It was read but never matched against llama.cpp.
