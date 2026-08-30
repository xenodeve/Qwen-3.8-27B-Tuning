# Attention and KV under speculation — patches/spec-decode-attn.patch, patches/spec-decode-int8-kv.patch, patches/hybrid-sw-block-promote.patch, patches/vllm-pr50021-gdn-spec-bounds.patch, patches/hybrid-kv-groups-v2-cudagraph.patch (syv-ref, patched vLLM 0.27.1, RTX 3090 24 GB, Qwen3.8-27B)
**31 techniques.** 1010 source lines across 5 files.
Files read: `patches/spec-decode-attn.patch` · `patches/spec-decode-int8-kv.patch` · `patches/hybrid-sw-block-promote.patch` · `patches/vllm-pr50021-gdn-spec-bounds.patch` · `patches/hybrid-kv-groups-v2-cudagraph.patch`
> **What the reader could not see:** 1) The FOCUS item "how CUDA graphs are captured for several decode lengths" is NOT in this slice. hybrid-kv-groups-v2-cudagraph.patch touches CUDA graphs only as a *memory accounting* problem (`profile_cudagraph_memory` returning 0 on the V2 runner) — it contains no capture-shape/decode-length logic, no cudagraph_capture_sizes list, no uniform_decode_query_len handling. Whatever selects the set of captured decode lengths lives elsewhere in the repo (candidates by filename: patches/dflash2-backport.patch, patches/dflash2-lookup-drafting.patch, patches/speed-knobs-envs.patch, single-user/start_qwen.sh) and was not in my file list. The only CUDA-graph interaction in *this* slice is the constraint that the spec-attn partial buffers must never be reallocated after capture (spec-decode-attn.patch:23-27, :276-280). 2) These are patch files, not the applied tree. Every citation is a line in the .patch; the true in-tree line numbers of the touched vLLM files (v1/attention/backends/flash_attn.py, v1/attention/ops/spec_decode_attn.py, v1/attention/backends/triton_attn.py, v1/core/kv_cache_utils.py, v1/worker/gpu/model_runner.py) are only known via the hunk headers (@@ -1038, @@ -1688/-1719, @@ -688/-693, @@ -1067, @@ -1137, @@ -751/-752 ...). 3) No test files, no benchmark scripts, and no raw result logs backing any of the quoted numbers are in this slice — every figure is prose in a patch header, unverifiable from here. 4) spec-decode-attn.patch's second hunk is `@@ -0,0 +1,201 @@` (a whole new file) but the body I can count is ~200 added lines; no truncation was visible. 5) `_promote_indivisible_block_sizes` computes `per_block = spec.page_size_bytes // spec.block_size` (hybrid-sw-block-promote.patch:94) and then only tests `per_block <= 0` — the variable is otherwise unused. Reads like a leftover from an earlier formulation; I mention it because a reader porting the function will wonder what it was for. 6) The GDN patch explicitly disclaims a fix it does NOT provide: "it does NOT fix the k=4 crash described in single-user/README.md" (vllm-pr50021-gdn-spec-bounds.patch:10). single-user/README.md is not in my slice, so that crash is undescribed here. 7) The PR's own `mamba_utils.py` prefix-caching hunks were deliberately dropped (vllm-pr50021-gdn-spec-bounds.patch:7-9) — so this vendoring is only safe with prefix caching OFF, and nothing in this slice shows where prefix caching is disabled.

---

## EXISTS, NEVER SET — 2

### Explicit CUDA-graph memory reservation for the V2 model runner
**Where (theirs):** `patches/hybrid-kv-groups-v2-cudagraph.patch:110-127` · `patches/hybrid-kv-groups-v2-cudagraph.patch:18-28`

**What it does.** The V2 runner's `profile_cudagraph_memory` returns 0 ("TBD"), so its ~1.2 GiB of captured graphs is allocated on top of gpu_memory_utilization rather than inside it. This patch lets an operator subtract a measured figure via VLLM_V2_CUDAGRAPH_MEM_MIB before the KV cache is sized.

**Mechanism.** Inside `profile_cudagraph_memory`: `reserve_mib = float(os.environ.get("VLLM_V2_CUDAGRAPH_MEM_MIB", "0") or 0)`; if `reserve_mib > 0 and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE`, it logs "Reserving %.2f GiB for CUDA graphs before sizing the KV cache" and returns `int(reserve_mib * 1024 * 1024)`; otherwise falls through to the upstream `return 0` (hybrid-kv-groups-v2-cudagraph.patch:114-127). The worker already subtracts whatever this returns from the KV budget, so no other call site changes. The operator is told where to get the number: "Read the real figure from the startup line '... and X GiB for CUDAGraph memory'" (:25).

**Why they needed it.** It is the named cause of a production OOM: "ask for 0.93 and the process actually peaks near 0.98, which is how the DeltaNet spec path OOMs mid-request (main README, gotcha 4)" (hybrid-kv-groups-v2-cudagraph.patch:21-23). They explicitly rejected the faithful fix: "Profiling them the way V1 does needs a temporary KV cache and graph pool; this instead makes the reservation explicit and measurable" (:22-24).

**Their numbers.** ~1.2 GiB of graphs on the V2 runner; gpu_memory_utilization 0.93 requested vs ~0.98 actual peak. Also: "the profiled activation peak itself varies by ~1 GiB between starts on this model" (hybrid-kv-groups-v2-cudagraph.patch:26-27).

**llama.cpp — EXISTS, NEVER SET.** The technique is: make the unaccounted allocation (CUDA graphs, driver overhead, activation-peak variance) an explicit, operator-set reservation subtracted before the KV cache is sized, rather than hoping the utilisation fraction covers it. llama.cpp already has that knob and we have never touched it. -fitt defaults to 1024 MiB PER DEVICE, so on a 12 GB card --fit is currently forfeiting a full GiB by default, and it measures free memory at that instant (fit.cpp:56-57, 559-563) so it inherits boot-to-boot variance exactly as vLLM's utilisation fraction does. The memory-breakdown table the server logs has an 'unaccounted' column that is the direct diagnostic for whether the margin is right.

**Equivalent here:** -fitt / --fit-target MiB — the per-device margin --fit leaves free before sizing anything

**Evidence (llama.cpp):** `common/arg.cpp:2851-2874` · `common/common.h:473` · `common/fit.cpp:559-563` · `common/fit.cpp:56-57` · `src/llama-context.cpp:686-697`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Concrete and bidirectional on 12 GB. Downward: -fitt 384 hands roughly 640 MiB back to the KV cache, which at q4_0 (0.5625 B/element) is a meaningful context increase — the exact token count depends on n_embd_head_k * n_head_kv * n_layer for this model, which I have not computed. Upward: if we ever see a late OOM or a mid-run allocation failure under draft-dflash (which now creates a second context and its own memory module, so the footprint is larger than the map's ngram-mod baseline), raising -fitt is the correct first response rather than reducing -c by hand. Either way it must be measured within one boot per this repo's pairing rule, since --fit tracks free VRAM.

### Pin the pool with --kv-cache-memory rather than tune gpu_memory_utilization
**Where (theirs):** `patches/hybrid-kv-groups-v2-cudagraph.patch:26-28`

**What it does.** An operating rule derived from the run-to-run variance: for a fixed target context length, size the KV pool by absolute bytes instead of by a fraction of free memory.

**Mechanism.** "Note the profiled activation peak itself varies by ~1 GiB between starts on this model, so for a fixed context length prefer pinning the pool with --kv-cache-memory (what single-user/start_qwen.sh does for SPEC=dflash2) over tuning gpu_memory_utilization." (hybrid-kv-groups-v2-cudagraph.patch:26-28). The concrete value used appears in the sibling patch: `--kv-cache-memory=5583457484` (hybrid-sw-block-promote.patch:31).

**Why they needed it.** A fraction-of-free-memory budget makes every measurement depend on boot conditions; pinning bytes makes two runs comparable and makes an OOM deterministic rather than intermittent.

**Their numbers.** Activation peak varies ~1 GiB between starts on this model; the pinned figure in use is --kv-cache-memory=5583457484 (~5.20 GiB).

**llama.cpp — EXISTS, NEVER SET.** This is the single most directly applicable item in the slice, because it addresses a problem this repo has already documented against itself. The vLLM rule is: for a fixed target context, size the pool in absolute bytes so two runs are comparable and an OOM is deterministic. llama.cpp's equivalent is stronger and free — fit.h:15-18 states the contract that n_ctx is modified if and only if it equals 0, and fit.cpp:368-370 prints 'context size set by user to %u -> no change'. So passing a numeric -c removes --fit's context step entirely and the geometry stops following free VRAM. Adding an explicit -ngl N additionally aborts the layer-placement pass (fit.cpp:377-379, which throws when n_gpu_layers != the default -1), pinning placement too. There is one trap the map records and it matters: -c 0 is NOT the same as omitting -c — arg.cpp:1641-1644 sets fit_params_min_ctx = UINT32_MAX, which turns the reduction off by a different route and prints 'user has requested full context size'.

**Equivalent here:** an explicit numeric -c N (which makes --fit leave context alone entirely), optionally with an explicit -ngl N to stop its placement pass

**Evidence (llama.cpp):** `common/fit.h:15-18` · `common/fit.cpp:368-370` · `common/fit.cpp:344` · `common/arg.cpp:1641-1644` · `common/fit.cpp:377-379` · `common/fit.cpp:56-57`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** This is a measurement-validity win rather than a tok/s win, and by this project's own standards that is worth more. CLAUDE.md records that free VRAM at boot moves 9,326-10,732 MiB and --fit follows it, which is why effects below 13.6 % are called noise. Pinning -c to a fixed number removes the largest term in that variance for every future comparison, at the cost of one flag and the risk that a low-VRAM boot fails to start instead of silently shrinking — which is the correct failure mode for an instrument. Recommend pairing it with -fitt so the margin is explicit too.

## partial — 3

### Partial buffers allocated once and frozen for CUDA-graph address stability
**Where (theirs):** `patches/spec-decode-attn.patch:23-27` · `patches/spec-decode-attn.patch:92-95` · `patches/spec-decode-attn.patch:276-290`

**What it does.** SpecDecodeAttention allocates part_o/part_m/part_l a single time at construction, sized for the largest query block the server will ever see (qmax), and never grows them.

**Mechanism.** The constructor takes `qmax` and immediately allocates `n = max_num_reqs * num_heads * qmax * nseg` elements (spec-decode-attn.patch:287-290); `run()` asserts rather than reallocating: `assert max_query_len <= self.qmax, "too many query tokens per request for this kernel"` and `assert num_reqs <= self.max_num_reqs` (:314-315). The docstring records the failure mode: "a CUDA graph captures their addresses, so growing them later would leave the captured decode graph pointing at freed memory" (:278-280).

**Why they needed it.** This is a bug they hit, not a precaution. Their words: "They must not grow afterwards: a captured CUDA graph holds their addresses, and a later reallocation - which is what happened as soon as a small prefill chunk asked for a longer block - leaves the captured decode graph reading freed memory." (spec-decode-attn.patch:24-27). The trigger was a *prefill chunk*, i.e. a path nobody was thinking about when sizing a decode buffer.

**llama.cpp — partial.** The F16 dequant scratch is budgeted into the compute buffer at reserve time and is therefore address-stable, so llama.cpp does not have the specific freed-memory bug vLLM hit. But the analogous fragility does exist here in a different form, and the map already documents it: a CUDA graph is only captured after two consecutive calls with unchanged node properties, and any property change resets warmup_complete to eager execution (ggml-cuda.cu:4253-4268). parallel_blocks is recomputed per call from ntiles_KV = ceil(K->ne[1]/nbatch_fa), i.e. from context depth, and dst_tmp is re-sized with it (fattn-common.cuh:1181-1183). Combined with llm_graph_params::allow_reuse requiring ubatch.n_tokens equality (src/llama-graph.h:785), a workload that accepts a VARIABLE number of drafted tokens per step — which is what speculation is — plausibly never gets past graph warmup.

**Equivalent here:** compute-buffer scratch is graph-stable (ggml-alloc + ggml_cuda_flash_attn_ext_get_alloc_size); dst_tmp/dst_tmp_meta are pool allocations whose SIZE tracks parallel_blocks, which tracks context depth

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:906-912` · `ggml/src/ggml-cuda/fattn-common.cuh:1152-1183` · `ggml/src/ggml-cuda/ggml-cuda.cu:4249-4274` · `ggml/src/ggml-cuda/ggml-cuda.cu:4265-4268` · `tools/server/server-context.cpp:617-619`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Unknown in tok/s, but cheap to check and we already print the number: the per-completion log line 'graphs reused = %10d' (server-context.cpp:617-619) comes from llama_perf_context(ctx_tgt).n_reused. If that stays near zero under draft-dflash and is high under a non-speculative run at the same depth, graph re-capture is a real cost of speculation here and belongs in the tested register. No code change needed to find out.

### int8 per-token-head KV read with scales folded in after the dot
**Where (theirs):** `patches/spec-decode-int8-kv.patch:12-19` · `patches/spec-decode-int8-kv.patch:130-135` · `patches/spec-decode-int8-kv.patch:168-178` · `patches/spec-decode-int8-kv.patch:187-190`

**What it does.** Adds a QUANT compile-time branch to the same partial kernel so it can read an int8 KV cache with one fp32 scale per (token, head), multiplying the scales in after each tl.dot instead of dequantising K and V first. The claim is that this is exact, not an approximation.

**Mechanism.** Under `if QUANT:` the loaded int8 K and V are cast `k = k.to(tl.bfloat16)` / `v = v.to(tl.bfloat16)`; the scales are loaded per KV position with their own strides, `k_sc = tl.load(ks_ptr + blk * stride_ksb + slot * stride_kss + kvh * stride_ksh, mask=k_ok, other=0.0)` and likewise v_sc (spec-decode-int8-kv.patch:170-175). Then `s = tl.dot(qs, tl.trans(k)).to(tl.float32) * k_sc[None, :]` (:176) and `acc = acc * alpha[:, None] + tl.dot((p * v_sc[None, :]).to(tl.bfloat16), v).to(tl.float32)` (:188). The V scale is applied to the *probabilities* before the second dot rather than to V, which keeps V in int8-derived bf16.

**Why they needed it.** The exactness argument is theirs and is the transferable part: "Both scales are per (token, head), i.e. constant along D, so folding them in after the dot is exact rather than an approximation" — written out as `s = (q . k_int8) * k_scale` instead of `q . (k_int8 * k_scale)`, and `acc = (p * v_scale) . v_int8` instead of `p . (v_int8 * v_scale)` (spec-decode-int8-kv.patch:130-133). And on the cast: "int8 -> bf16 is itself exact (bf16 has 8 mantissa bits, int8 needs 7), so the only new rounding is the one bf16 multiply the stock Triton kernel also does" (:134-135).

**Their numbers.** Against a dequantized reference: max error 1e-4. Against the unquantized reference: 1.02% mean relative — "which is int8 quantization of the cache, not the kernel" (spec-decode-int8-kv.patch:18-19).

**llama.cpp — partial.** This is the one technique in the slice that names a real hole in our live path. llama.cpp's VEC kernel already does exactly what the patch does: need_f16_K/need_f16_V are set true only when the type IS F16, so for q4_0 no conversion happens and the block scale is applied after the dot product — the same exactness argument (scale constant along D) holds for ggml block quants. The MMA kernel instead passes need_f16_K = need_f16_V = true unconditionally (fattn-mma-f16.cuh:1962-1963), and launch_fattn then runs to_fp16 over ggml_nelements(K) and ggml_nelements(V) — the entire padded cache view for that layer — into scratch, every layer, every call (fattn-common.cuh:1022-1084). With quantized KV on Ada, VEC is taken only when Q->ne[1] <= 2 (fattn.cu:469). Our profile now runs draft-dflash, so every verify step carries 1+n_draft >= 3 query tokens and lands on MMA. Porting the technique means writing quantized-K/V support into the MMA kernel — a new set of kernel instantiations and dot paths, not a patch.

**Equivalent here:** the VEC kernel already reads q4_0/q8_0 K and V directly (need_f16_K = type_K == GGML_TYPE_F16), folding the block scale in after the dot; the MMA kernel does not and expands the whole cache to F16

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-vec.cuh:540-541` · `ggml/src/ggml-cuda/fattn-vec.cuh:543` · `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963` · `ggml/src/ggml-cuda/fattn-common.cuh:1022-1084` · `ggml/src/ggml-cuda/fattn.cu:469` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912`

**Effort:** new-backend · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Real but unquantified, and it cuts both ways. Cost side: with -ctk q4_0 -ctv q4_0 on the MMA path we pay a full q4_0->F16 expansion of the whole KV cache per layer per verify step, plus compute-buffer scratch at 2 bytes/element against q4_0's 0.5625 — the map notes that because the reserve pass always runs the MMA shape, that scratch is budgeted whether or not decode ever uses it, so it is eating VRAM at 12 GB right now. The cheap experiment this suggests costs one flag and no code: compare -ctk q4_0 -ctv q4_0 against -ctk f16 -ctv f16 at a context short enough that both fit, paired within one boot per this repo's rule. f16 removes the conversion entirely (need_f16_K becomes a no-op) and removes the scratch; q4_0 buys context. Which wins at 16K on a 6.77 GB model is genuinely unknown and is exactly the kind of thing this project exists to measure.

### Promotion logging that names both the before and after block size
**Where (theirs):** `patches/hybrid-sw-block-promote.patch:123-135` · `patches/hybrid-sw-block-promote.patch:35-36`

**What it does.** Each promotion logs the layer, old block size, new block size, resulting page bytes and the old maximum, which is what makes the 'bf16 is untouched' claim checkable at runtime rather than by argument.

**Mechanism.** `logger.info("Raising block size of %s from %d to %d tokens so its page (%d B) covers the maximum (%d B) instead of being padded to it at block %d.", ...)` (hybrid-sw-block-promote.patch:125-134).

**Why they needed it.** The bf16 row of their results table is asserted as "unchanged, and zero promotions logged" (:35-36) — i.e. the log line is the evidence for the no-regression claim, not an afterthought.

**llama.cpp — partial.** The transferable idea is that a log line naming before and after is what makes a 'nothing else changed' claim checkable at runtime rather than by argument — and llama.cpp is inconsistent about this. The DFlash clamp we now depend on does warn with both values ('requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d'). The MTP chained-head clamp at speculative.cpp:1446 does not warn at all. --fit's context rounding logs, and the n_ctx/n_seq_max round-down warns. The gap is small and local: adding a LOG_WRN at speculative.cpp:1446 would be a few lines.

**Equivalent here:** llama.cpp logs some automatic clamps with old and new values (the DFlash/DSpark block-size clamp) and performs others silently (the MTP chained-head clamp)

**Evidence (llama.cpp):** `common/speculative.cpp:990-996` · `common/speculative.cpp:1446` · `common/fit.cpp:344` · `common/fit.cpp:361-366` · `src/llama-context.cpp:292-302`

**Effort:** small-patch · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No performance value. Real diagnostic value for this project specifically: our profile's effective draft length is decided by three separate clamps (block size, dp.n_max, and the per-call budget) and only some announce themselves, which is precisely the 'plausible number instead of a failure' shape the repo's north star names. The MTP clamp does not affect us today since we are on draft-dflash, so this is low priority.

## already have it — 15

### Split-KV (flash-decoding) Triton attention for the verify step
**Where (theirs):** `patches/spec-decode-attn.patch:136-146` · `patches/spec-decode-attn.patch:169` · `patches/spec-decode-attn.patch:188-192` · `patches/spec-decode-attn.patch:209-212` · `patches/spec-decode-attn.patch:319`

**What it does.** Replaces FlashAttention-2 for speculative verify batches with a hand-written Triton kernel that splits the KV sequence across NUM_SEGMENTS=16 thread blocks per (request, kv-head, query-tile), so a batch-1 decode with a handful of query tokens saturates the GPU instead of leaving most SMs idle. Each block runs an online-softmax partial over its slice of the KV cache for all its query rows at once, and a second tiny kernel combines the partials.

**Mechanism.** `_spec_attn_partial` is launched with `grid = (num_reqs * ntile, Hkv, self.nseg)` (spec-decode-attn.patch:319). Inside, `pid = tl.program_id(0)`, `req = pid // NTILE`, `qtile = pid % NTILE`, `kvh = tl.program_id(1)`, `seg = tl.program_id(2)` (:188-192). The segment's key range is derived from the request's own kv length: `tiles_total = (kv_len + TILE - 1) // TILE`, `tiles_per_seg = (tiles_total + NSEG - 1) // NSEG`, `t0 = seg * tiles_per_seg`, `t1 = min(t0 + tiles_per_seg, tiles_total)` (:209-212) — so the split is over KV *tiles*, not raw positions, and short requests simply give empty segments. G = query heads per kv head is folded into the row dimension so one program covers all G query heads of its kv head.

**Why they needed it.** Their own words: "Neither vLLM's FlashAttention-2 path nor its Triton unified attention split the KV sequence across SMs when a request has more than one query token: with MTP k=4 (5 queries) on a 24-head model that leaves 24 (FA) or ~8 (Triton) thread blocks on an 82-SM RTX 3090 and the attention layer takes ~57 us for a 1.5k-token context." (spec-decode-attn.patch:139-143). FA2 only splits KV when a request has exactly one query token, which is exactly the case speculative decoding destroys.

**Their numbers.** ~57 us per attention layer at 1.5k-token context, MTP k=4 (5 query tokens), 24-head model, RTX 3090 (82 SMs), FA2 baseline — 24 thread blocks for FA2, ~8 for vLLM's Triton unified attention (spec-decode-attn.patch:5-8, :139-143).

**llama.cpp — already have it.** I read this rather than inferring it, because the map stops at kernel selection and says nothing about kernel internals. launch_fattn takes a stream_k parameter; the MMA kernel passes true (fattn-mma-f16.cuh:1963) and the branch at fattn-common.cuh:1120-1151 then spreads the work over min(max_blocks_per_sm*nsm, ntiles_KV*ntiles_dst) blocks, with use_stream_k forced true for cc >= GGML_CUDA_CC_ADA_LOVELACE (:1126) — our card is cc 890. Partial results are merged by flash_attn_stream_k_fixup_uniform/_general (:723, :807). The VEC kernel passes stream_k=false but takes the other branch, which searches for the parallel_blocks value with the best wave efficiency (:1156-1177), allocates dst_tmp/dst_tmp_meta (:1181-1183) and combines with flash_attn_combine_results (:916, launched :1270). So llama.cpp splits KV across SMs on BOTH paths, including when a request has many query tokens. The vLLM patch exists because FA2 only split KV at q_len==1; that limitation is not present here, so there is no gap to fill.

**Equivalent here:** launch_fattn stream-k (MMA path) and parallel_blocks + flash_attn_combine_results (VEC path) — automatic, no flag

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-common.cuh:1115-1151` · `ggml/src/ggml-cuda/fattn-common.cuh:1126` · `ggml/src/ggml-cuda/fattn-common.cuh:1152-1183` · `ggml/src/ggml-cuda/fattn-common.cuh:916-970` · `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963` · `ggml/src/ggml-cuda/fattn-vec.cuh:543`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Nothing to port. The value is negative-result value: it removes the single largest-looking item in this slice from the work queue, and it means a verify batch of 1+n_draft tokens from draft-dflash is not leaving the 4070 SUPER's SMs idle the way it would on stock vLLM.

### Query-row tiling (BLOCK_M rows, NTILE tiles) to lift the q_len cap
**Where (theirs):** `patches/spec-decode-attn.patch:13-21` · `patches/spec-decode-attn.patch:148-152` · `patches/spec-decode-attn.patch:198-202` · `patches/spec-decode-attn.patch:292-308`

**What it does.** The q_len x G query rows of one request are split into tiles of BLOCK_M rows rather than being required to fit in a single program, removing the hard cap of BLOCK_M // G query tokens per request. This is what makes a verify block longer than the drafter's affordable — previously a 16-token block exceeded the cap, fell back to FA2, and doubled the step.

**Mechanism.** Rows are enumerated `r = tl.arange(0, BLOCK_M)` and decoded as `ri = qtile * QT + r // G` (query token index) and `rg = r % G` (query head within the kv head), with validity `row_ok = (r < QT * G) & (ri < q_len)` (spec-decode-attn.patch:199-202). `_plan(q_len, G, D)` returns `(block_m, qt, ntile, warps)` where `qt = max(1, block_m // G)` and `ntile = triton.cdiv(q_len, qt)` (:307-308). block_m is picked by row count: <=32 -> 32, <=64 -> 64, else BLOCK_M_BIG=128; then floored to at least G and rounded up to a power of two via `1 << (block_m - 1).bit_length()` (:299-306). Warps follow the tile: `8 if block_m >= 128 else 4`.

**Why they needed it.** "That cap is what made a verify block longer than the drafter's unaffordable: a 16-token block fell back to FA2 and doubled the step at 25k context." (spec-decode-attn.patch:15-17). Restated in the module docstring as "That cap is what made lookup-augmented drafting unaffordable past a block of 7" (:150-151). The counter-pressure is stated too: "Each tile re-reads the KV segment, so BLOCK_M is chosen to keep the tile count at 1 where the register budget allows" (:151-152) — BLOCK_M_BIG=128 "with 8 warps, which keeps a 16-token block in one tile" (:171).

**Their numbers.** Per attention layer, batch 1, 25k-token context: 250 / 583 / 1,132 us at 8 / 16 / 32 query tokens, against FA2's 2,085 / 2,053 / 1,997 us (spec-decode-attn.patch:17-19). Note FA2 is flat in q — the split-KV kernel wins 8.3x at q=8 and only 1.76x at q=32.

**llama.cpp — already have it.** launch_fattn computes ntiles_x = ((Q->ne[1] + ncols1 - 1)/ncols1) and folds the GQA ratio into a second tile dimension (ntiles_z_gqa, fattn-common.cuh:1088-1090) — the same decomposition the patch adds by hand. There is no q_len cap in llama.cpp's FA to lift: an arbitrary Q->ne[1] is tiled, and the only consequence of a larger one is that ggml_cuda_get_best_fattn_kernel picks MMA_F16 rather than VEC (fattn.cu:469). The vLLM cap (BLOCK_M//G query tokens) is an artifact of their single-program-per-request kernel, which llama.cpp never had.

**Equivalent here:** ntiles_x = ceil(Q->ne[1]/ncols1) in launch_fattn, with ncols1/ncols2 chosen per config by the MMA case tables

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-common.cuh:1087` · `ggml/src/ggml-cuda/fattn.cu:358-533` · `ggml/src/ggml-cuda/fattn.cu:461-483`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None to port. Relevant only as reassurance that a long draft (e.g. the 15-token DFlash block, or ngram-mod's 64) does not hit a kernel-side length wall in this tree — the caps that do bite are the DFlash block-size clamp (common/speculative.cpp:990-996) and dp.n_max truncation (server-context.cpp:441-460), both already in the map.

### Online-softmax partial with explicit -inf-safe rescaling
**Where (theirs):** `patches/spec-decode-attn.patch:214-237`

**What it does.** Each segment accumulates its own running max m_i, running denominator l_i and unnormalised output acc over its KV tiles, using the standard streaming-softmax rescale — but with the all-masked case (a row whose whole tile is masked out, or an empty segment) handled explicitly so no NaN escapes.

**Mechanism.** Per KV tile: `m_new = tl.maximum(m_i, tl.max(s, 1))`; then `m_safe = tl.where(m_new == -inf, 0.0, m_new)` so `tl.exp(s - m_safe)` never computes inf-inf; `alpha = tl.exp(tl.where(m_i == -inf, -inf, m_i - m_safe))` gives alpha=0 for a still-empty accumulator instead of exp(nan); `l_i = l_i * alpha + tl.sum(p, 1)`, `acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v).to(tl.float32)` (spec-decode-attn.patch:231-237). Accumulators are fp32 (`m_i`, `l_i` fp32, `acc` [BLOCK_M, D] fp32, :214-216) while the matmuls run in bf16.

**Why they needed it.** A split-KV grid guarantees empty and fully-masked segments exist (a request shorter than NSEG tiles, or a causal row early in the verify block), so the -inf arithmetic is not a corner case here but the common case.

**llama.cpp — already have it.** Any split-KV FA implementation must carry per-partial max and denominator and rescale on merge; llama.cpp does, storing float2 (max, rowsum) meta per block and combining it in flash_attn_combine_results (:949-970) or the fixup kernels. Since llama.cpp already had split-KV before this patch existed, it already had the numerics that go with it. The specific -inf handling is an implementation detail of a Triton kernel, not a portable idea.

**Equivalent here:** the running (max, sum) meta carried through flash_attn_combine_results and the stream-k fixup kernels

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-common.cuh:916-970` · `ggml/src/ggml-cuda/fattn-common.cuh:723-806` · `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1438-1453`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None. The one thing worth remembering is the class of failure it guards — an empty or fully-masked KV segment producing NaN — which in llama.cpp would surface as garbage output, not a crash.

### Flat partial-buffer indexing ((req*Hq + head)*QMAX + i)*NSEG + seg
**Where (theirs):** `patches/spec-decode-attn.patch:239-244` · `patches/spec-decode-attn.patch:256-259` · `patches/spec-decode-attn.patch:287-290`

**What it does.** Partial outputs, maxes and denominators live in three flat preallocated tensors addressed by a single arithmetic index, with `seg` as the fastest-varying dimension so the combine kernel reads all NSEG partials of one (request, head, token) contiguously.

**Mechanism.** Writer: `hrow = kvh * G + rg`, `pidx = ((req * Hq + hrow) * QMAX + ri) * NSEG + seg`, then `tl.store(part_o_ptr + pidx[:, None] * D + d[None, :], acc, mask=row_ok[:, None])` plus masked stores of m_i and l_i (spec-decode-attn.patch:240-244). Reader: `base = ((req * Hq + h) * QMAX + i) * NSEG`, `segs = tl.arange(0, NSEG)`, `tl.load(part_m_ptr + base + segs)` (:259-261). Buffers are sized `n = max_num_reqs * num_heads * qmax * nseg`, with part_o [n, head_dim] fp32 and part_m/part_l [n] fp32 (:287-290). Note the index uses QMAX, not the actual q_len — the stride is fixed for the life of the server so the addressing is graph-stable.

**Why they needed it.** The QMAX-strided (rather than q_len-strided) layout is what allows the same buffers to serve every query-block length without reallocation, which the CUDA-graph constraint below makes mandatory.

**llama.cpp — already have it.** llama.cpp's equivalent partial storage is dst_tmp (parallel_blocks*ggml_nelements(KQV)) and dst_tmp_meta (parallel_blocks*ggml_nrows(KQV)), addressed by the combine kernel's own flat arithmetic at fattn-common.cuh:940-941. The graph-stability half of the vLLM idea is solved structurally rather than by a QMAX-strided layout: the FA node's F16 scratch is appended to the dst tensor and sized by ggml_cuda_flash_attn_ext_get_alloc_size, so ggml-alloc gives it a fixed address inside the compute buffer for the life of the context.

**Equivalent here:** dst_tmp / dst_tmp_meta sized from parallel_blocks, plus the F16 scratch charged to the compute buffer via ggml_cuda_flash_attn_ext_get_alloc_size

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-common.cuh:1181-1183` · `ggml/src/ggml-cuda/fattn-common.cuh:1264-1271` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None to port.

### Combine kernel: reweight partials by exp(m_seg - m_max)
**Where (theirs):** `patches/spec-decode-attn.patch:247-270`

**What it does.** A second Triton kernel, one program per (request, query head, query token), merges the NSEG partial softmaxes into the final output in the caller's output tensor and dtype.

**Mechanism.** Grid is `(num_reqs, Hq, max_query_len)` (spec-decode-attn.patch:332). It early-outs on `if i < q_len` for ragged requests (:258). Then `m_max = tl.max(m, 0)`, `m_max = tl.where(m_max == -inf, 0.0, m_max)`, `w = tl.exp(m - m_max)` — the comment notes "segments with -inf give 0" (:264-265) so empty segments contribute nothing without a branch. `l_tot = tl.sum(l * w, 0)`; the [NSEG, D] partial block is loaded at once and reduced as `o = tl.sum(o * w[:, None], 0) / tl.maximum(l_tot, 1e-30)` (:266-269), the clamp preventing a 0/0 for an all-masked row. The store casts to `out_ptr.dtype.element_ty` (:270), so normalisation happens in fp32 and only the final write is bf16.

**Why they needed it.** This is the required second half of any split-KV scheme; the interesting detail is that both the -inf sentinel and the 1e-30 clamp exist specifically because the verify block's causal mask produces genuinely empty rows.

**llama.cpp — already have it.** Same finding as technique 1. flash_attn_combine_results loads 2*parallel_blocks meta values, takes the max, reweights each partial by exp(m_l - m_max) and divides by the combined denominator (:949-970) — the identical algorithm. The stream-k path uses two fixup variants instead, chosen by whether the block count is a multiple of the tile count (:1228-1259).

**Equivalent here:** flash_attn_combine_results (VEC/parallel_blocks path) and flash_attn_stream_k_fixup_uniform/_general (MMA path)

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-common.cuh:916-970` · `ggml/src/ggml-cuda/fattn-common.cuh:723-806` · `ggml/src/ggml-cuda/fattn-common.cuh:1228-1271`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None to port.

### TILE size chosen against sm86's 99 KB shared memory
**Where (theirs):** `patches/spec-decode-attn.patch:316-318`

**What it does.** The KV tile length is 64 or 32 depending on the query-tile size and head dim, so that the q tile, one K tile, one V tile and the score block co-resident in shared memory fit the RTX 3090's per-SM limit.

**Mechanism.** `tile = 64 if (block_m <= 32 or D <= 128) else 32` (spec-decode-attn.patch:318), guarded by the comment "shared memory on sm86 is 99 KB: q tile + one K and one V tile + scores must fit" (:316). Combined with `num_stages=1` on the launch (:330), which forbids Triton from software-pipelining extra K/V buffers into that same budget.

**Why they needed it.** The large-BLOCK_M path (128 rows, 8 warps) exists to keep NTILE at 1, but 128 rows x a 64-wide score block at D=256 does not fit; halving TILE is the cheaper concession than halving BLOCK_M, because BLOCK_M is what controls KV re-reads.

**llama.cpp — already have it.** llama.cpp carries per-architecture MMA configuration tables and raises the per-kernel dynamic shared-memory cap explicitly (fattn-mma-f16.cuh:1020-1023), then feeds nbytes_shared into cudaOccupancyMaxActiveBlocksPerMultiprocessor to derive max_blocks_per_sm (fattn-common.cuh:1112-1113) — i.e. the shared-memory budget is not just respected, it feeds back into how many blocks the split-KV scheme launches. It is compiled-in tuning with no runtime knob, exactly as in the vLLM patch (which also hardcodes its choice).

**Equivalent here:** per-config nbatch_fa and nbytes_shared, with cudaFuncSetAttribute raising the dynamic shared-memory limit; not exposed as any flag

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1018-1024` · `ggml/src/ggml-cuda/fattn-common.cuh:1111-1113` · `ggml/src/ggml-cuda/fattn.cu:358-533`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. There is no flag here to set; changing it means editing the config tables and rebuilding, which this project has no measurement apparatus to justify.

### One-shot global QMAX resolution from speculative_config, floored at BLOCK_M // G
**Where (theirs):** `patches/spec-decode-attn.patch:88-111` · `patches/spec-decode-attn.patch:172`

**What it does.** Computes the per-request query-token cap exactly once per process and memoises it in a module global, taking it from the speculative config (1 + num_speculative_tokens), overridable by VLLM_SPEC_DECODE_ATTN_QMAX, floored at BLOCK_M // G and ceilinged at QMAX_TOKENS=64.

**Mechanism.** `_spec_attn_qmax(group)` guards on `global _SPEC_ATTN_QMAX is None`; reads `get_current_vllm_config().speculative_config` inside a try/except and sets `n = 1 + spec.num_speculative_tokens` (0 on failure); then `n = int(os.environ.get("VLLM_SPEC_DECODE_ATTN_QMAX", n or 0)) or n`; finally `_SPEC_ATTN_QMAX = min(QMAX_TOKENS, max(n, BLOCK_M // group))` (spec-decode-attn.patch:96-111). QMAX_TOKENS=64 is documented as "query tokens per request the caller may ask for" (:172).

**Why they needed it.** The inline comment states both the semantic change and the freeze: "The kernel tiles the query rows itself, so the cap is on query tokens per request rather than on q_len * group (which used to stop a verify block longer than 10). It is fixed for the life of the server: the partial buffers are sized for it once, and a captured CUDA graph holds their addresses." (spec-decode-attn.patch:92-95). The `max(n, BLOCK_M // group)` floor keeps the pre-tiling capability as a lower bound so nothing that worked before regresses.

**llama.cpp — already have it.** llama.cpp derives the per-sequence output limit once, as per_seq = min(n_batch, 1+n_draft) with n_draft = common_speculative_n_max taking the max over every enabled speculator, and that value is baked into cparams for the life of the process. The equivalent of the vLLM floor/ceiling is the DFlash block-size clamp, which lowers both n_max and n_min to the trained block size with a warning (speculative.cpp:990-996). The map's CANNOT #1 (no per-request speculative parameters; the whole schema block is inside #if 0) is the same 'fixed for the life of the server' property the patch is enforcing by hand.

**Equivalent here:** common_speculative_n_max + common_speculative_get_output_limits, computed once at startup from the enabled type list

**Evidence (llama.cpp):** `common/speculative.cpp:2351-2385` · `common/speculative.cpp:2512-2521` · `tools/server/server-context.cpp:42-54` · `common/speculative.cpp:990-996`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None to port. Worth knowing for our profile: with draft-dflash,ngram-mod chained, common_speculative_n_max takes the MAX over both, so the output limits are sized by ngram-mod's n_max=64, not by DFlash's block size — per_seq = min(n_batch, 65).

### FLASH_ATTN backend dispatch gate for the spec-decode kernel
**Where (theirs):** `patches/spec-decode-attn.patch:38-67` · `patches/spec-decode-attn.patch:82-85`

**What it does.** Intercepts the call just before `flash_attn_varlen_func` and reroutes to the Triton kernel when the batch is a multi-query verify and none of the features the kernel lacks are in play; otherwise falls through to FA2 untouched.

**Mechanism.** A single `if` conjunction (spec-decode-attn.patch:41-55): `_spec_attn_enabled()` (env VLLM_SPEC_DECODE_ATTN == "1", :85); `1 < max_seqlen_q <= _spec_attn_qmax(num_heads // num_kv_heads)`; `not is_quantized_kv_cache(self.kv_cache_dtype)`; sliding_window either None or both bounds negative; `not self.logits_soft_cap`; `alibi_slopes is None`; `sinks is None`; `causal is True`; `mm_mask_mod is None`; `rswa_mask_mod_fn is None`. On a hit it calls `_spec_attn_run(...)` with `query[:num_actual_tokens]` / `output[:num_actual_tokens]` and returns immediately (:56-67).

**Why they needed it.** "FA2 does not split the KV sequence when max_seqlen_q > 1, leaving most SMs idle" (spec-decode-attn.patch:39-40). The lower bound `1 < max_seqlen_q` matters as much as the upper: pure single-token decode is left to FA2, which already splits KV there. The feature list is the kernel's documented restriction set ("no sliding window / softcap / alibi", :160-161) turned into a runtime guard rather than an assert.

**llama.cpp — already have it.** llama.cpp already dispatches on exactly the variables the vLLM gate tests, including the same q>1 discriminator, and its feature-restriction checks (K->type == V->type at :442-446, head-dim table at :392-437, KQ-bias exclusion at src/llama-graph.cpp:2540-2542) are the same shape of guard. The difference is direction: vLLM leaves q==1 to FA2 and reroutes q>1; llama.cpp routes q<=2 to VEC and q>=3 to MMA when KV is quantized. I confirmed there is no env override — grep for getenv across ggml/src/ggml-cuda/fattn*.cu[h] returns nothing — which corroborates the map's CANNOT #1.

**Equivalent here:** ggml_cuda_get_best_fattn_kernel — a per-node, per-call dispatch on cc, head dims, K/V types, gqa_ratio and Q->ne[1]

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:358-533` · `ggml/src/ggml-cuda/fattn.cu:464` · `ggml/src/ggml-cuda/fattn.cu:469` · `ggml/src/ggml-cuda/fattn.cu:442-446`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None to port. The one operational consequence, already in the map, is that the flip is silent: there is no log line when a step crosses from VEC to MMA, so a throughput change at a given draft length has no diagnostic trace.

### Padded head dim D+4 carries the scale inline, so the addressing code is unchanged
**Where (theirs):** `patches/spec-decode-int8-kv.patch:7-13` · `patches/spec-decode-int8-kv.patch:125-129` · `patches/spec-decode-int8-kv.patch:201-202`

**What it does.** The quantized cache view is [num_blocks, block_size, Hkv, D + 4] — head dim padded by 4 bytes so one fp32 scale per (token, head) sits inline right after that head's int8 data — which means the kernel's pointer arithmetic needs no modification at all: only the element type and the scale multiply change.

**Mechanism.** "The padded head dim is already carried by stride_kh, so the addressing is untouched" (spec-decode-int8-kv.patch:11-12, restated :128-129). The kernel still indexes `k_ptr + blk*stride_kb + slot*stride_ks + kvh*stride_kh + d[None,:]` with `d = tl.arange(0, D)` — D is the unpadded head dim, stride_kh is the padded one, so the load naturally skips the 4 scale bytes. The scales are additionally exposed as separate f32 views [num_blocks, block_size, Hkv] whose strides are passed as six new kernel args `stride_ksb, stride_kss, stride_ksh, stride_vsb, stride_vss, stride_vsh` (:154-155). `run()` docstring: "key/value_cache: [num_blocks, block_size, Hkv, D] bf16, or [.., D + pad] int8 with k/v_scale_cache [num_blocks, block_size, Hkv] float32" (:201-202).

**Why they needed it.** It made the quant support a ~30-line diff on an existing kernel rather than a second kernel, because the padded stride does the skipping for free.

**llama.cpp — already have it.** llama.cpp never separated scales from data in the first place — the scale is inline in every ggml quant block, which is why the cache tensor is a plain ggml_tensor of type q4_0 and every consumer's addressing is unchanged. The constraint this creates is the block-size divisibility check (a q4_0 block is 32, so n_embd_head_k must be a multiple of 32; head dim 128 passes) at src/llama-context.cpp:3613-3633, which llama.cpp enforces as a hard init failure rather than silently.

**Equivalent here:** ggml block-quant formats carry the scale inside the block (q4_0 = fp16 d + 16 bytes of nibbles per 32 elements); K/V cache tensors use them directly

**Evidence (llama.cpp):** `ggml/include/ggml.h:390-433` · `src/llama-kv-cache.cpp:206-217` · `src/llama-context.cpp:3613-3633`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero to port.

### Zero-stride placeholders keep one kernel signature for both quant and bf16
**Where (theirs):** `patches/spec-decode-int8-kv.patch:206` · `patches/spec-decode-int8-kv.patch:214` · `patches/spec-decode-int8-kv.patch:220-224`

**What it does.** Rather than branching the launch, run() always passes the scale pointers and six scale strides, substituting a (0,0,0) tuple when there is no quantized cache, and flips a constexpr QUANT flag so Triton compiles the dead branch away.

**Mechanism.** `quant = k_scale_cache is not None` (spec-decode-int8-kv.patch:206); the launch splats `*(k_scale_cache.stride() if quant else (0, 0, 0))` and the same for v (:220-221); `QUANT=quant` joins the constexpr list (:224). Because QUANT is `tl.constexpr` (:159), the non-quant specialization contains no scale loads at all — the bf16 path is byte-identical in generated code to before the patch.

**Why they needed it.** Keeps a single kernel and single autotuning surface for two cache formats without paying for the unused one, and means the QUANT=0 specialization needs no re-benchmarking against the pre-patch kernel.

**llama.cpp — already have it.** llama.cpp achieves the same 'one source, no runtime cost for the unused format' outcome through template specialisation rather than a Triton constexpr flag: exactly four VEC instances are compiled in this build (f16-f16, q4_0-q4_0, q8_0-q8_0, bf16-bf16, per fattn.cu:321-325 and the CMake case list), each with need_f16_K/need_f16_V resolved at compile time. Same technique, different language.

**Equivalent here:** C++ template parameters type_K/type_V with per-pair instantiation; the dead branch does not exist in the generated code

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:284-290` · `ggml/src/ggml-cuda/fattn.cu:321-325` · `ggml/src/ggml-cuda/fattn-vec.cuh:539-543` · `ggml/src/ggml-cuda/CMakeLists.txt:115-125`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero to port. The build-configuration consequence is already in the map: GGML_CUDA_FA_ALL_QUANTS=OFF is why only those four pairs exist and why -ctk and -ctv must match.

### GDN: bound the accepted-token state lookup in fused_recurrent_gated_delta_rule
**Where (theirs):** `patches/vllm-pr50021-gdn-spec-bounds.patch:91-123`

**What it does.** Masks the `ssm_state_indices` load to the request's own row so a zero, stale or oversized accepted-token count cannot read outside the tensor, and zero-fills the output on the invalid path instead of leaving it uninitialised.

**Mechanism.** `i_t` is `num_accepted_tokens - 1`. The patch adds `idx_in_row = (i_t >= 0) & (i_t < stride_indices_seq)` and converts the load to `tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t, mask=idx_in_row, other=0)` (vllm-pr50021-gdn-spec-bounds.patch:110-115), so an out-of-row index yields 0 and falls into the pre-existing `state_idx <= 0` path. That path is also changed from a bare `return` to zero-filling the output first: `zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)`, `for _ in range(0, T): tl.store(p_o, zero, mask=mask_v); p_o += HV * V` (:117-122).

**Why they needed it.** The mechanism of the crash is spelled out in the added comment: "A zero accepted count gives ``i_t == -1`` (a read before this request's row, and before the tensor for ``i_n == 0``); a stale or too-large count reads past the row. The ``state_idx <= 0`` guard below only rejects non-positive values, so an out-of-range read that returns a garbage positive int32 flows into ``h0 + state_idx * stride_init_state_token`` and is dereferenced, faulting the SM." (vllm-pr50021-gdn-spec-bounds.patch:99-109). The existing guard was necessary but not sufficient — it rejected the wrong half of the value range.

**llama.cpp — already have it.** I read this rather than trusting the map, because the map asserts this path is dead in our profile and that assertion is stale. The rollback index in llama.cpp is a scalar derived from the accepted position (rollback = cell.pos - (p0 - 1)) and is explicitly range-checked — 'if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq)' at llama-memory-recurrent.cpp:184 — with a plain 'return false' otherwise, which the caller turns into the checkpoint-restore path rather than a bad read. Both failure directions the vLLM patch fixes (a zero count giving index -1, and an oversized stale count reading past the row) are covered by that single bound. QWEN35/QWEN35MOE are in llm_arch_supports_rs_rollback, and n_rs_seq is non-zero for us because draft-dflash is in the type list (common.h:386-392) — so unlike the map's ngram-mod baseline, this code is LIVE in our current configuration.

**Equivalent here:** llama_memory_recurrent::seq_rm bounds the rollback to [1, n_rs_seq] and returns false otherwise; the arch must be in llm_arch_supports_rs_rollback

**Evidence (llama.cpp):** `src/llama-memory-recurrent.cpp:180-190` · `src/llama-memory-recurrent.cpp:184` · `src/llama-arch.cpp:1044-1055` · `src/llama-context.cpp:104-109` · `common/common.h:386-392`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** No change to make. The finding that matters for the parent: several 'EXISTS BUT UNUSED' rows in the capability map (n_rs_seq == 0, the RS branch never taken, checkpoints on via the FULL branch) were written for --spec-type ngram-mod and are wrong now that draft-dflash is in use. common_context_can_seq_rm short-circuits to RS when llama_n_rs_seq(ctx) > 0 (common/common.cpp:1581-1585), which changes whether context checkpoints are created at all (server-context.cpp:3381-3384). That is a behavioural difference between the measured ngram-mod runs and today's draft-dflash runs that nobody has accounted for.

### GDN: same bound in fused_sigmoid_gating_delta_rule_update
**Where (theirs):** `patches/vllm-pr50021-gdn-spec-bounds.patch:129-162`

**What it does.** Applies the identical row-mask and zero-fill to the fused sigmoid-gating variant of the Gated DeltaNet update kernel.

**Mechanism.** Identical construction: `idx_in_row = (i_t >= 0) & (i_t < stride_indices_seq)`, masked load with `other=0`, and the `state_idx <= 0` branch zero-filling T timesteps of output before returning (vllm-pr50021-gdn-spec-bounds.patch:149-161). One extra consequence is noted: the early return also "returns before the ``final_state_idx`` load later" (:147-148), i.e. the invalid path is protected against a second unbounded read further down the kernel.

**Why they needed it.** Two kernels serve the same speculative path depending on fusion; patching only one leaves the fault reachable.

**llama.cpp — already have it.** The vLLM patch must fix two kernels because the same logical operation is fused two different ways depending on configuration. llama.cpp expresses the rollback once, in the memory module, above the kernel layer — the ggml graph for the recurrent update reads a state index that has already been validated. There is no second site to patch, which is a structural advantage worth noting rather than a gap.

**Equivalent here:** the same single seq_rm bound — llama.cpp has one rollback path, not two fused kernel variants

**Evidence (llama.cpp):** `src/llama-memory-recurrent.cpp:180-190` · `src/llama-memory-hybrid.cpp:11-64`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero.

### causal_conv1d_update: validate num_accepted against seqlen and emit zeros
**Where (theirs):** `patches/vllm-pr50021-gdn-spec-bounds.patch:21-44`

**What it does.** Checks the accepted-token count against the sequence length before using it as a conv-state offset, and on failure writes zeros over the whole output for that sequence and returns cleanly.

**Mechanism.** Replaces `conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1` with a load into `num_accepted`, then `if (num_accepted < 1) | (num_accepted > seqlen):` -> loop `for idx_token in tl.range(seqlen)` storing `zero` at `o_ptr + o_offset + idx_token*stride_o_token + idx_feats*stride_o_dim` with `mask=idx_feats < dim`, plus `if launch_pdl: tl.extra.cuda.gdc_launch_dependents()` before returning; only then `conv_state_token_offset = num_accepted - 1` (vllm-pr50021-gdn-spec-bounds.patch:28-42).

**Why they needed it.** Both directions are wrong: 0 accepted tokens gives offset -1, and a stale count larger than the drafted block reads past the conv state. The `launch_pdl` call on the early-exit path matters — skipping it would leave programmatic-dependent-launch consumers waiting on a producer that never signalled.

**llama.cpp — already have it.** In llama.cpp the conv state (n_embd_r) and the recurrent state (n_embd_s) are two tensors in one llama_memory_recurrent sized n_rows = mem_size * (1 + n_rs_seq), indexed by the same validated rollback. There is no separate conv-state offset computed from an unchecked accepted-token count. There is one further guard the vLLM patch has no counterpart for: the trailing 1+n_rs_seq tokens of a sequence must stay within a single ubatch (llama-memory-recurrent.cpp:422-424), which prevents a rollback target from being split across batches in the first place.

**Equivalent here:** the conv state and the recurrent state live in the same memory module and are covered by the same bound; widths come from n_embd_r() and n_embd_s()

**Evidence (llama.cpp):** `src/llama-memory-recurrent.cpp:99-101` · `src/llama-memory-recurrent.cpp:180-190` · `src/llama-hparams.cpp:183-229` · `src/llama-memory-recurrent.cpp:422-424`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero code change. Confirms that the class of illegal-memory-access crash vLLM hit under concurrent speculation is guarded here — relevant because we are on the speculative recurrent path now and would otherwise have to wonder.

### selective_scan_update: bound the initial token index by the ROW stride, not the element stride
**Where (theirs):** `patches/vllm-pr50021-gdn-spec-bounds.patch:50-85`

**What it does.** Adds `valid_initial_token` to the selective-scan speculative path, bounding `max(num_accepted - 1, 0)` against `stride_state_indices_batch`, and zero-fills the output for the whole sequence when the bound fails.

**Mechanism.** `valid_initial_token = init_token_idx < stride_state_indices_batch` (vllm-pr50021-gdn-spec-bounds.patch:62). The pointer advance is split in two: `state_batch_indices_ptr += pid_b * stride_state_indices_batch` happens first, then `if IS_SPEC_DECODING and not valid_initial_token:` zero-fills `out_ptr` over `seq_len` steps with `offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)` masked by `offs_m < dim` and returns; only on the valid path does `state_batch_indices_ptr += init_token_idx * stride_state_indices_T` follow (:73-83). The existing `tl.maximum(num_accepted - 1, 0)` clamp is deliberately preserved (:61) — the new test is additive.

**Why they needed it.** The choice of bound is the transferable part, and the comment argues it explicitly: "The bound is the ROW stride (elements per batch row), matching fused_recurrent.py's i_t < stride_indices_seq: for a contiguous [batch, T] tensor stride(0) == T. stride(1) would be 1 there, which rejects every accepted count > 1 (and a padded row keeps any overshoot inside this request's allocated slack)." (vllm-pr50021-gdn-spec-bounds.patch:54-60). Using the intuitively-named stride would have silently disabled speculation entirely.

**llama.cpp — already have it.** llama.cpp bounds against a semantic count (n_rs_seq = draft.n_max, the number of reserved recurrent-state snapshots) rather than against a tensor stride, so there is no wrong-but-plausible stride to pick. What transfers is the reasoning the vLLM comment makes explicit: choosing the intuitively-named bound would have silently rejected every accepted count > 1, i.e. disabled speculation rather than crashed. llama.cpp has one analogous silent-disable to be aware of — llama-context.cpp:104-109 clamps n_rs_seq to 0 for an arch not in llm_arch_supports_rs_rollback and logs it at DEBUG level only, so on an unsupported arch a request for bounded rollback silently becomes no rollback. QWEN35 is supported, so this does not bite us.

**Equivalent here:** the bound is n_rs_seq, a count of reserved snapshots, so the row-vs-element-stride ambiguity does not arise

**Evidence (llama.cpp):** `src/llama-memory-recurrent.cpp:99-101` · `src/llama-memory-recurrent.cpp:184` · `common/common.h:386-392` · `src/llama-context.cpp:104-109`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Zero to implement. Carries one operational note: if we ever run this speculation setup against a non-QWEN35 arch, the loss of bounded rollback will be invisible at default verbosity and will show up only as slower steps via the checkpoint path.

### Zero-fill rather than bare-return on every invalid speculative state path
**Where (theirs):** `patches/vllm-pr50021-gdn-spec-bounds.patch:29-38` · `patches/vllm-pr50021-gdn-spec-bounds.patch:75-81` · `patches/vllm-pr50021-gdn-spec-bounds.patch:116-122` · `patches/vllm-pr50021-gdn-spec-bounds.patch:155-161`

**What it does.** A cross-cutting discipline in all four kernel fixes: whenever the bound check rejects, the kernel writes zeros over its slice of the output tensor before returning, rather than returning immediately.

**Mechanism.** Each site constructs a zero vector of the output's element type and loops the sequence dimension storing it under the feature mask — see the four citations. In fused_recurrent and fused_sigmoid_gating the comment names the intent: "Skip invalid state indices without exposing uninitialized output" (:116, :155). Notably this changes behaviour of the *pre-existing* `state_idx <= 0` path too, which previously returned bare.

**Why they needed it.** vLLM's output buffers are reused across steps, so a bare return leaves the previous step's activations in place — a wrong but entirely plausible result, which is strictly worse than a crash for anyone trying to measure. This is the same failure class the bounds fix is preventing, one level up.

**llama.cpp — already have it.** The discipline — never let a rejected path leave the previous step's values where the next stage will read them as valid — is exactly this repo's stated north star, and llama.cpp implements it structurally on the recurrent side: a failed partial rollback returns false, common_context_can_seq_rm has already classified the context as FULL or RS at startup, and the server responds by restoring a checkpoint or re-processing, not by proceeding with whatever was in the state. Worth noting the honesty of the vLLM authors here: they changed the behaviour of a PRE-EXISTING bare-return path, i.e. the wrong-but-plausible output had been reachable before their patch and nobody had noticed.

**Equivalent here:** seq_rm returning false propagates into a checkpoint restore or a full re-prefill rather than leaving a stale buffer in play

**Evidence (llama.cpp):** `src/llama-memory-recurrent.cpp:184-188` · `common/common.cpp:1559-1599` · `tools/server/server-context.cpp:3381-3384` · `tools/server/server-context.cpp:3869`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No change to make. Its value is as a review criterion: when reading any future llama.cpp speculation patch, check that a rejected draft or a failed rollback cannot leave a partially-written buffer downstream. This project has thirteen documented instrument faults of exactly that shape.

## impossible here — 1

### Causal mask derived from kv position, not from a mask tensor
**Where (theirs):** `patches/spec-decode-attn.patch:156-158` · `patches/spec-decode-attn.patch:203` · `patches/spec-decode-attn.patch:229-230`

**What it does.** The verify block's causal structure is computed arithmetically from the paged-cache bookkeeping: query token i of a request is known to sit at kv position `seqused_k - q_len + i`, so no attention mask is materialised or passed.

**Mechanism.** `q_pos = kv_len - q_len + ri` (spec-decode-attn.patch:203) where `kv_len = tl.load(seqused_ptr + req)` is the kv length *including* the new tokens and `q_len` comes from cu_seqlens_q. The mask is then `allowed = k_ok[None, :] & (pos[None, :] <= q_pos[:, None]) & row_ok[:, None]`, applied as `s = tl.where(allowed, s, -inf)` (:229-230), fusing three conditions in one: KV bounds, causality, and row padding within BLOCK_M.

**Why they needed it.** Stated as a layout contract in the docstring: "Query token i of a request sits at kv position seqused_k - q_len + i and attends causally" (:157-158). It keeps the kernel signature identical to what the FLASH_ATTN backend already has on hand, so the gate can call it with the metadata vLLM already built.

**llama.cpp — impossible here.** llama.cpp builds an explicit kq_mask tensor in the graph and hands it to the FA node; the mask is not merely causality — it also encodes which cache cells belong to which sequence, the 256-cell padding from get_n_kv (src/llama-kv-cache.cpp:1233-1246), SWA windows where present, and the ALiBi slope path. Deriving position arithmetically would require the KV cells of a sequence to be contiguous and ordered, which the shared cache does not guarantee, and would have to be done in every backend, not just CUDA. The seam would be llm_graph_context::build_attn_mha at src/llama-graph.cpp:2540-2565, but changing it is a rewrite of the memory abstraction, not a patch. Note llama.cpp already banks most of the practical benefit a different way: flash_attn_mask_to_KV_max (fattn-common.cuh:1098-1109) scans the mask once per call to compute a per-tile KV bound so fully-masked tiles are skipped.

**Equivalent here:** none — llama.cpp materialises kq_mask (F16 with FA, F32 without) and passes it to every attention op

**Evidence (llama.cpp):** `src/llama-graph.cpp:38` · `src/llama-graph.cpp:789` · `src/llama-graph.cpp:950` · `src/llama-graph.cpp:2540-2565` · `ggml/src/ggml-cuda/fattn-common.cuh:1098-1109`

**Effort:** new-backend · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Zero. The mask is cheap relative to the F16 dequant that dominates the quantized-KV MMA path (technique 12), so this is the wrong thing to attack here.

## not applicable — 10

### Per-(heads, head_size, device) cache of the attention object, sized from scheduler max_num_seqs
**Where (theirs):** `patches/spec-decode-attn.patch:114-133`

**What it does.** `_spec_attn_run` keeps a module-level dict of SpecDecodeAttention instances so the buffers are shared by every layer with the same shape, and derives max_num_reqs from the scheduler config with a defensive floor from the actual batch.

**Mechanism.** `key = (impl.num_heads, impl.head_size, q.device)` into the module dict `_SPEC_ATTN` (spec-decode-attn.patch:79, :117-118). On miss, `max_reqs = get_current_vllm_config().scheduler_config.max_num_seqs` inside try/except with fallback 256, then `max_reqs = max(max_reqs, cu_seqlens_q.shape[0] - 1)` (:123-126) so a batch larger than the config still constructs. num_reqs is passed as `cu_seqlens_q.shape[0] - 1` (:133).

**Why they needed it.** Keying on shape rather than layer means a 24-layer model allocates one set of partial buffers, not 24 — and since the buffers are sized `max_reqs * num_heads * qmax * nseg` fp32 rows of head_dim, per-layer allocation would be a large multiple of the budget the CUDA-graph freeze already forces them to reserve up front.

**llama.cpp — not applicable.** The problem this solves — a per-layer Python object each allocating its own buffers, so a 24-layer model pays 24x — does not exist in a graph-allocator design. ggml-alloc assigns overlapping lifetimes to the same block, and the reserve passes at context construction size the compute buffer for the worst-case shape once (src/llama-context.cpp:595, 662-671).

**Equivalent here:** none needed — scratch is per-node inside one compute buffer that ggml-alloc reuses across layers

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:906-912` · `ggml/src/ggml-alloc.c:684` · `src/llama-context.cpp:576-671`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero.

### The honest cost of the inline padding: 16-byte load misalignment
**Where (theirs):** `patches/spec-decode-int8-kv.patch:29-35`

**What it does.** Documents that the D+4 layout costs 61% of the kernel's time in isolation because odd KV heads start 4 bytes off a 16-byte boundary, breaking Triton's vectorized loads — and names the unimplemented fix.

**Mechanism.** "the padded head dim means stride_kh is 260 B, so odd KV heads start 4 bytes off a 16-byte boundary and the vectorized loads break" (spec-decode-int8-kv.patch:29-31). Proposed but not done: "Reading the data as int32 (4-byte aligned, which 260 does satisfy) and unpacking would recover most of the 61%" (:34-35).

**Why they needed it.** They also record the counterintuitive conclusion that follows: "a contiguous int8 cache is still slightly *slower* than bf16 despite half the bytes -- this kernel is tile-bound, not bandwidth-bound, at these shapes, so a quantized cache buys context, never speed" (:32-34). Note this directly contradicts the docstring they added in the same patch (see next entry) — worth flagging to any reader.

**Their numbers.** 128k context, q=8, per attention layer: 2,151 us with the padded int8 cache, against a contiguous int8 cache's 1,335 us and bf16's 1,250 us — 61% penalty in isolation, "about 13% of the step" end to end (spec-decode-int8-kv.patch:30-32).

**llama.cpp — not applicable.** The specific defect (260-byte stride breaking vectorized loads) is an artifact of their chosen layout and cannot occur with ggml block quants. What transfers is the conclusion they drew from it — 'a quantized cache buys context, never speed' — which in llama.cpp is if anything more strongly true on the path we are actually on, since MMA converts the entire cache to F16 before touching it and charges the scratch to the compute buffer. Their conclusion was measured on their kernel and does not constitute evidence about llama.cpp; it is a hypothesis to test here.

**Equivalent here:** no D+4 layout exists; the analogous cost here is the MMA F16 dequant, not load misalignment

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-common.cuh:1022-1084` · `ggml/src/ggml-cuda/fattn.cu:534-568` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Unknown, but it sets the right prior for the -ctk experiment named under technique 12: do not expect quantized KV to be faster here, expect it to be smaller. If a measurement here shows q4_0 FASTER than f16 at equal context, that is a surprising result worth double-checking the instrument for.

### Bandwidth accounting for the bf16 path at 128k
**Where (theirs):** `patches/spec-decode-int8-kv.patch:137-139`

**What it does.** Records the measured achieved bandwidth of the bf16 kernel at long context as the justification for halving the KV bytes.

**Mechanism.** "at a 128k context and 8 query tokens it reads 524 MB of bf16 KV in 1,198 us, which is 437 GB/s of the 3090's ~936" (spec-decode-int8-kv.patch:137-139) — i.e. ~47% of peak HBM.

**Why they needed it.** This is the argument for int8 ("Halving the bytes matters because this kernel is bandwidth-bound at long context", :137). It is in tension with the patch header's own conclusion that the kernel is "tile-bound, not bandwidth-bound, at these shapes" (:33-34) and that quantization "buys context, never speed". A reader should treat the 437/936 GB/s figure as the measured fact and the bound-ness label as contested within the same file.

**Their numbers.** 524 MB of bf16 KV read in 1,198 us = 437 GB/s, against the 3090's ~936 GB/s peak, at 128k context and 8 query tokens.

**llama.cpp — not applicable.** llama.cpp exposes no profiling counter from which achieved HBM bandwidth could be read; the timings block and the slot log lines give wall time and token counts only. The technique is a measurement discipline, not a capability: compute the bytes the attention layer must read (n_layer * 2 * n_kv_padded * n_embd_head * n_head_kv * bytes_per_element, with n_kv padded to 256 per src/llama-kv-cache.cpp:1238) and divide by measured time to see how far from peak you are. The seam for that is our own bench harness, not llama.cpp, so I will not call it absent-but-possible in llama.cpp.

**Equivalent here:** no achieved-bandwidth counter exists; llama-server reports only t/s, per-token ms and graphs-reused

**Evidence (llama.cpp):** `tools/server/server-context.cpp:600-619` · `tools/server/server-context.cpp:634-637` · `src/llama-kv-cache.cpp:1233-1246`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Unknown as a speedup; useful as a check. A bytes/second figure computed this way would tell us whether decode at depth on the 4070 SUPER is bandwidth-limited (in which case KV quantisation should help despite the dequant) or kernel-limited (in which case it cannot). That is a cheap arithmetic addition to the bench, and it would let us distinguish those two worlds without guessing.

### TRITON_ATTN backend gate for the quantized verify path
**Where (theirs):** `patches/spec-decode-int8-kv.patch:78-114` · `patches/spec-decode-int8-kv.patch:21-27`

**What it does.** Mirrors the FLASH_ATTN gate inside triton_attn.py, so the int8 KV path also bypasses vLLM's own unified attention for multi-query verify batches. It is a strictly longer condition list than the FA gate because TRITON_ATTN supports more features.

**Mechanism.** The condition (spec-decode-int8-kv.patch:86-100) adds to the FA gate: `self._kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD`, `k_scale_cache is not None`, `attn_metadata.causal`, `self.sliding_window is None or self.sliding_window == (-1, -1)`, `self.chunk_lookback == -1`, `mm_prefix_range_tensor is None`, `attn_metadata.rswa_prefix_lens is None`, `output_scale is None`. On a hit it calls `_spec_attn_run(...)` with the two scale caches appended and returns before `unified_attention` (:101-116). It imports `_spec_attn_enabled`, `_spec_attn_qmax`, `_spec_attn_run` directly from the flash_attn backend module (:66-70), so the two backends share one QMAX global and one buffer cache.

**Why they needed it.** "vLLM's own unified attention cannot cover this case: it disables its split-KV ('3D') path whenever max_seqlen_q > 1 (triton_unified_attention.py, `use_3d`), and every DFlash2 step is a multi-query verify" (spec-decode-int8-kv.patch:21-24, :79-82). The sliding-window test is load-bearing for correctness of the whole speculative setup: "The drafter's own sliding-window layers are excluded by the window test below" (:85) — the same drafter layers that the block-promote patch is about.

**Their numbers.** Per attention layer at 128k context, 8 query tokens: 1.3 ms for this kernel against 7.4 ms for vLLM's unified attention and 10.1 ms for FA2 (spec-decode-int8-kv.patch:24-26).

**llama.cpp — not applicable.** This is technique 11 duplicated for a second vLLM backend. llama.cpp has a single ggml_cuda_get_best_fattn_kernel entry point that all FLASH_ATTN_EXT nodes go through, so there is no second gate to write and no shared-global-across-backends problem to solve. Their measured 1.3 ms vs 7.4 ms vs 10.1 ms comparison is evidence about vLLM's kernels and says nothing about this tree.

**Equivalent here:** llama.cpp has one CUDA FA dispatch, not two competing backends to gate separately

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:330-336` · `ggml/src/ggml-cuda/fattn.cu:570-583` · `ggml/src/ggml-cuda/fattn.cu:586-588`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero.

### Sliding-window block-size promotion instead of page padding
**Where (theirs):** `patches/hybrid-sw-block-promote.patch:52-136` · `patches/hybrid-sw-block-promote.patch:96-103` · `patches/hybrid-sw-block-promote.patch:142-151`

**What it does.** When a layer's KV page size does not divide the maximum page size, this raises that layer's block size up to the smallest multiple of its own kernel granularity whose natural page covers the maximum, and lets the resulting page become the new maximum — so every other layer pads up by a percent or two instead of that layer paying a full primary page per 16 tokens.

**Mechanism.** `_promote_indivisible_block_sizes(kv_cache_spec, max_page_size)` selects candidates: AttentionSpec, `spec.page_size_bytes < max_page_size`, `max_page_size % spec.page_size_bytes != 0`, `spec.block_size > 0` (hybrid-sw-block-promote.patch:80-87). For each, `ratio = cdiv(max_page_size, spec.page_size_bytes)`, `new_block_size = spec.block_size * ratio`, `new_spec = replace(spec, block_size=new_block_size)`, bailing entirely if the promoted page still falls short (:97-101). `new_max = max(new_max, new_spec.page_size_bytes)` (:103). Hooked into `unify_kv_cache_spec_page_size` immediately after `max_page_size = max(page_sizes)` and before the per-layer scale/pad loop (:145-151), so it rewrites both the spec dict and the maximum the loop works against.

**Why they needed it.** The failure is a silent assumption breaking. Their words: "Sliding-window layers are born at the backend's smallest kernel block -- 16 -- because the code that picks it assumes unify will scale it (the comment at model_executor/layers/attention/attention.py reads 'the smallest block is fine -- `unify` scales it up by an integer ratio'). When the ratio is not an integer, that assumption fails silently: the block stays 16 and every block pays a whole primary page." (hybrid-sw-block-promote.patch:7-11). And the economic argument for the direction of the fix: "Those layers hold hundreds of blocks each, so paying 1.5% on them is far cheaper than 385 nearly-empty pages on the drafter." (:25-26).

**Their numbers.** 3090, SPEC=dflash2, DFLASH_TOKENS=7, --kv-cache-memory=5583457484: int8_per_token_head estimated max model length 3,536 -> 94,176; int8_per_token_head pool at max_model_len 131,072 went from "would not start" to 138,696 tokens; bf16 unchanged at block size 448 / 16.88 GiB needed / 46,592 estimated max, with zero promotions logged. 138,696 vs bf16's 69,758 = 1.99x the context in the same pool (hybrid-sw-block-promote.patch:31-39).

**llama.cpp — not applicable.** The bug being fixed requires three things llama.cpp does not have: a paged allocator with a unified page size across layers, a shared pool between target and drafter layers, and sliding-window layers in the drafter competing in that pool. In llama.cpp each layer's KV is a plain tensor on the device its weights landed on (llama-kv-cache.cpp:209-217), the draft context always gets its OWN memory module (speculative.cpp:2460-2461, with target memory shared only for a GEMMA4_ASSISTANT or a sidecar shipping without tok_embd/output), and qwen35 declares no SWA at all so llama_memory_hybrid is chosen rather than the iswa variant (llama-model.cpp:2305). The server additionally force-disables swa_full on a model with no SWA layers (server-context.cpp:1188-1191).

**Equivalent here:** none — llama.cpp has no paged KV, no cross-layer unified page size, and the draft model gets its own separate memory module

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:209-217` · `src/llama-model.cpp:2281-2344` · `src/llama-model.cpp:2305` · `common/speculative.cpp:2460-2482` · `tools/server/server-context.cpp:1188-1195`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. Their 1.99x context result is entirely an artifact of vLLM's page geometry and does not indicate an equivalent 2x is sitting unclaimed here.

### The arithmetic accident that hid the bug at bf16
**Where (theirs):** `patches/hybrid-sw-block-promote.patch:13-21` · `patches/hybrid-sw-block-promote.patch:64-69`

**What it does.** Documents precisely why the same configuration is fine in bf16 and catastrophic in int8: the divisibility that unify relies on held only by coincidence of head-dim arithmetic, and the per-head fp32 scale destroys it.

**Mechanism.** Target: 4 KV heads x 256 head dim; drafter: 8 KV heads x 128. Both give 4096 B per token per layer, ratio exactly 28 (hybrid-sw-block-promote.patch:13-15). With a per-token-head quantized cache adding one fp32 scale *per head*: target `2*4*(256+4) = 2080` B/token, drafter `2*8*(128+4) = 2112` B/token, and "2112 = 2**6 * 3 * 11 shares no useful factor with the primary page, so no block size ever divides it" (:16-18, :67-69).

**Why they needed it.** This is the most transferable content in the patch: it shows the class of bug (a config-dependent integer coincidence load-bearing for a memory-layout invariant) and the exact perturbation that breaks it. The consequence: "The drafter's 5 layers then took cdiv(2047 + 4096, 16) + 1 = 385 blocks of 1.71 MiB each -- 1.88% utilised, a constant 5.155 GiB, 75.6% of the whole per-request budget" (:18-21).

**Their numbers.** 385 blocks x 1.71 MiB = 5.155 GiB constant, 1.88% utilised, 75.6% of the per-request budget, a 53x blow-up holding 33 KB of useful KV (hybrid-sw-block-promote.patch:18-21, :61-62). Before the fix: "int8 needed 6.82 GiB to serve 32,768 tokens where bf16 serves 69,758 in 5.2 GiB" (:20-21) — i.e. quantization made the cache strictly worse than bf16 on both axes.

**llama.cpp — not applicable.** The specific coincidence (4x256 and 8x128 both giving 4096 B/token until a per-head fp32 scale breaks it) has no counterpart here. But the bug CLASS — a config-dependent integer coincidence silently load-bearing for a memory-layout invariant — is worth carrying, and it is the same class this repo already collects as instrument faults. llama.cpp's equivalent invariants are checked loudly: a head dim not divisible by the cache block size refuses to build the context with a named error, and n_kv is padded to 256 explicitly with the stated reason that the graph must stay constant across batches.

**Equivalent here:** llama.cpp's structurally similar invariants (block size divides head dim; n_kv padded to FATTN_KQ_STRIDE) hard-error or pad explicitly instead of silently wasting

**Evidence (llama.cpp):** `src/llama-context.cpp:3613-3622` · `src/llama-context.cpp:3624-3633` · `src/llama-kv-cache.cpp:1233-1246` · `ggml/src/ggml-cuda/fattn-common.cuh:9`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No VRAM or tok/s. Its worth here is as a review question to ask of any future patch that changes head dim, KV type or block geometry: what integer relation is this configuration relying on, and does it hold for the next quant?

### Promotion guard: every non-promoted spec must be able to reach the new maximum
**Where (theirs):** `patches/hybrid-sw-block-promote.patch:105-121` · `patches/hybrid-sw-block-promote.patch:27-29`

**What it does.** Before committing the promotion, it verifies that every layer it did NOT promote can actually pad up to the new (larger) maximum page size — otherwise it returns the original spec and maximum untouched, and logs why.

**Mechanism.** Loop over the whole spec (hybrid-sw-block-promote.patch:107-121): skip if already promoted or already at new_max; `MambaSpec` passes unconditionally ("Mamba pads by construction"); `new_max % spec.page_size_bytes == 0` passes (it can scale); `AttentionSpec and spec.indexes_kv_by_block_stride` passes ("attention only if its backend reads pages by block stride"). Anything else triggers `logger.info("Not promoting draft block sizes: layer %s can neither scale nor pad to the resulting page size.", name)` and `return kv_cache_spec, max_page_size`. Note the function is all-or-nothing: a single unpaddable layer abandons all promotions.

**Why they needed it.** "Guarded: it only fires when every other spec can actually accept padding (Mamba by construction, attention only if its backend indexes KV by block stride), so it never turns a configuration that starts today into a NotImplementedError." (hybrid-sw-block-promote.patch:27-29). Raising the maximum page size is a global act — it makes every other layer's problem harder — so the guard is what keeps an optimisation for one config from being a startup regression for another.

**llama.cpp — not applicable.** This is the safety half of technique 18 and inherits its verdict: with no paged allocator and no shared cross-layer page size, there is nothing that could be raised globally and therefore nothing to guard against. The general principle (an all-or-nothing check before a global change that makes every other component's problem harder) is sound but has no site here.

**Equivalent here:** none — there is no promotion to guard

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:209-217` · `src/llama-model.cpp:2281-2344`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero.

### KV-cache group sizing that pads the sliding-window bucket instead of the big buckets
**Where (theirs):** `patches/hybrid-kv-groups-v2-cudagraph.patch:41-83` · `patches/hybrid-kv-groups-v2-cudagraph.patch:89-96` · `patches/hybrid-kv-groups-v2-cudagraph.patch:5-16`

**What it does.** Overrides vLLM's rule that the KV group size is the size of the smallest bucket of same-type layers, when that smallest bucket is sliding-window-only. It instead picks the largest common divisor of the non-sliding buckets, so the target's full-attention and Mamba layers get zero padding and the small sliding-window drafter bucket absorbs the padding instead.

**Mechanism.** `_prefer_padding_sliding_window_buckets(layer_buckets, spec_buckets, group_size)`: finds bucket indices where every spec is a `SlidingWindowSpec` (hybrid-kv-groups-v2-cudagraph.patch:57-61); returns unchanged unless `min(len(layer_buckets[i]) for i in sw) == group_size`, i.e. the sliding bucket really is the binding one (:65-66); computes `g_all = gcd` over all non-sliding bucket sizes (:67-69); then scans `for g in range(g_all, group_size, -1)`, skipping non-divisors of g_all, and accepts the first g where every sliding bucket's padding `(g - len % g) % g` is `<= len(bucket)` (:70-75). Wired in as the `else` branch to vLLM's existing `group_size = max_num_layers` special case (:93-96).

**Why they needed it.** "With the DFlash2 drafter that bucket is its 5 sliding-window layers, so the target's 16 full-attention layers get padded to 20 and the 48 GDN layers to 50 ('Add 4 padding layers, may waste at most 25%'): 25% more memory for every token of context, to pad layers that are not the problem." (hybrid-kv-groups-v2-cudagraph.patch:6-9). The reason padding the window bucket is safe is the key insight: "Sliding-window groups only ever hold window-many blocks (the manager frees blocks behind the window), so padding THEM is nearly free." (:9-10).

**Their numbers.** Buckets 16/48/5 -> group size 8: no full-attention or GDN padding, 3 padding layers on the 5-layer window group, costing ~7 MB per request; 9 groups instead of 15. Measured on the 3090: 105 -> 78 KB of pool per token, i.e. 45,383 tokens at 40k context -> 69,758 at 64k. MTP mode has no sliding-window bucket, takes the early return, and keeps its 73,777-token pool bit-identical (hybrid-kv-groups-v2-cudagraph.patch:12-16, :51-54).

**llama.cpp — not applicable.** llama.cpp's hybrid memory splits layers by a predicate (filter_attn = !is_recr(il), filter_recr = is_recr(il)) into exactly two memory modules, each sized from the actual layer count. There are no groups, no group size, no padding layers, and therefore no rule to override. Their 105 -> 78 KB/token result is a property of vLLM's grouping scheme.

**Equivalent here:** none — llama.cpp does not group layers into equal-size KV buckets and never pads layer counts

**Evidence (llama.cpp):** `src/llama-model.cpp:2281-2303` · `src/llama-memory-hybrid.cpp:11-64` · `src/llama-kv-cache.cpp:209-217`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero.

### Selective vendoring of an unmerged upstream PR, with the dropped hunks and the residual bug named
**Where (theirs):** `patches/vllm-pr50021-gdn-spec-bounds.patch:1-15`

**What it does.** Vendors only the kernel hunks of an open upstream PR (vllm-project/vllm#50021), explicitly dropping the parts that do not apply, recording the precondition that makes the drop safe, and naming a crash this patch does NOT fix.

**Mechanism.** "Vendored here (kernel hunks only; the mamba_utils.py prefix-caching hunks of the PR do not apply to 0.27.1 and are not needed with prefix caching off) because we hit illegal-memory-access crashes with several concurrent MTP requests. Note: it does NOT fix the k=4 crash described in single-user/README.md." (vllm-pr50021-gdn-spec-bounds.patch:6-10). Provenance and licence are recorded: "Source: https://github.com/vllm-project/vllm/pull/50021 (Apache-2.0)" (:15). The diff retains upstream index lines (e.g. `index 8335c849666d..e7387484cca6`) so the base can be identified.

**Why they needed it.** The dropped hunks carry a live precondition — this vendoring is only correct with prefix caching off — and the named residual crash stops a future reader from assuming the GDN spec path is now sound. The reproduction condition ("several concurrent MTP requests") is what distinguishes this from the single-user k=4 crash.

**llama.cpp — not applicable.** This is a process technique and the most directly adoptable non-code item in the slice, because we are in the same position: C:\AI\llama.cpp-dflash2 is built from an unmerged upstream PR, and the map identifies exactly which code that PR introduced (the DFlash/DSpark speculator, the maximal-coupling verifier in sampling.cpp, and one node-budget line at llama-context.cpp:2317-2321). What the vLLM patch headers do and our build does not is state, in the artifact itself, what was taken, what was deliberately left out, what precondition makes leaving it out safe, and which known bug it does NOT fix. Judged against llama.cpp as a capability the answer is not-applicable; judged as a practice it is straightforwardly worth copying.

**Equivalent here:** not a llama.cpp capability — a source-management practice; our build IS a vendored unmerged PR (#27342, DFlash2, commit 1deefcca3, build 10499)

**Evidence (llama.cpp):** `common/speculative.cpp:910-1347` · `common/sampling.cpp:722-793` · `src/llama-context.cpp:2317-2321`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** No VRAM or tok/s. Prevents a specific future error: a later session assuming the dflash2 build is upstream llama.cpp, or assuming a fix present in master is present here. The natural home is docs/results/ alongside the build, recording PR number, base commit, build number and what is known not to work.

### Patches carry their own ordered apply protocol and version stamp
**Where (theirs):** `patches/spec-decode-attn.patch:29-30` · `patches/spec-decode-int8-kv.patch:37-40` · `patches/hybrid-sw-block-promote.patch:41-44` · `patches/hybrid-kv-groups-v2-cudagraph.patch:30-33`

**What it does.** Every patch header ends with the exact apply command, the patch it must follow, and a 'Written against vLLM 0.27.1. Reapply after upgrades.' stamp.

**Mechanism.** All four use `patch -p1 -d venv/lib/python3.12/site-packages/vllm < patches/<name>.patch`, applied against the installed package directory rather than a source tree. The ordering constraints are stated as dependencies: spec-decode-int8-kv "after spec-decode-attn.patch" (spec-decode-int8-kv.patch:37), hybrid-sw-block-promote "after hybrid-kv-groups-v2-cudagraph.patch" (hybrid-sw-block-promote.patch:41). The dependency is real, not cosmetic — spec-decode-int8-kv.patch:44-59 edits functions that spec-decode-attn.patch creates, at line offsets that only exist post-apply (`@@ -1757,7` against a file the first patch extended to :1719+).

**Why they needed it.** Patching site-packages is destroyed by any pip upgrade, and these hunks have no fuzz tolerance for each other's line offsets. A sibling `patches/_check_applied.py` exists (not read) that presumably verifies application state.

**llama.cpp — not applicable.** The ordering half of this technique is specific to patching site-packages with no fuzz tolerance between hunks; a git checkout has none of that fragility. The stamping half transfers and is already partly done — the capability map is dated to build 10499 / commit 1deefcca3. What is NOT stamped anywhere durable is the BUILD configuration, which changes what the binary can do: GGML_CUDA_FA_ALL_QUANTS=OFF is why only four K/V type pairs exist and why -ctk must equal -ctv, and CMAKE_CUDA_ARCHITECTURES=89 is why the TILE kernel is unreachable. Those two facts decide the answers to several techniques in this slice and live only in a CMakeCache.

**Equivalent here:** we build from a git tree rather than patching an installed package, so the apply-order hazard does not exist; the version stamp does apply

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:321-325` · `ggml/CMakeLists.txt:208`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** No performance value. Concrete risk it removes: a future rebuild that flips GGML_CUDA_FA_ALL_QUANTS to ON would silently change which -ctk/-ctv combinations run on the GPU, invalidating every KV-type measurement taken before it, with no record of when the change happened. Recording the build flags next to the results is cheap insurance.
