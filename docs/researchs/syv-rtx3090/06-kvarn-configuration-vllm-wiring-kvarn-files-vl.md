# KVarN configuration + vLLM wiring: kvarn/files/vllm/model_executor/layers/quantization/kvarn/{config.py, sinkhorn.py, __init__.py}, kvarn/kvarn-0.27.1.patch, kvarn/kvarn-v2-runner.patch, kvarn/README.md, kvarn/install.sh
**49 techniques.** 1320 source lines across 7 files.
Files read: `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/sinkhorn.py` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/__init__.py` · `kvarn/kvarn-0.27.1.patch` · `kvarn/kvarn-v2-runner.patch` · `kvarn/README.md` · `kvarn/install.sh`
> **What the reader could not see:** 1) The README (kvarn/README.md:18) says the patch is "the seven small hunks upstream vLLM needs" and enumerates seven concerns, but kvarn-0.27.1.patch actually contains SIX diff'd files (config/cache.py, model_executor/layers/attention/attention.py, platforms/cuda.py, platforms/interface.py, utils/torch_utils.py, v1/attention/backends/registry.py, v1/kv_cache_interface.py = 7 files / 9 hunks). The count reconciles only if you count "backend registry + priority" as two, or count files rather than hunks. Note the "seven fixes" in kvarn-v2-runner.patch:3 is a *different* seven and the numbered list there runs 1..7 but item 6 is the mamba_hybrid fix and 7 the speculator fix — the list is internally consistent, just easy to confuse with the 0.27.1 patch's seven. 2) The docstring at config.py:432-434 claims "Default 16 mirrors the paper" but the code on the very next line reads os.environ.get("KVARN_SINKHORN_ITERS", "8") — the comment is stale relative to the constant. The dataclass default (config.py:64) is also 8. sinkhorn.py:33 keeps _DEFAULT_ITERATIONS = 16, so the pure-PyTorch reference and the shipped config disagree by default. 3) config.py:425 docstring example says a preset string "like `kvarn_k4v4`" but no such key exists in KVARN_PRESETS — every real key carries a `_g<N>` suffix (config.py:22-25). Passing the docstring's example raises. 4) config.py:62 sets the dataclass default value_bits=4, while the README (kvarn/README.md:5) and the shipped/headline preset are 4-bit keys / 2-bit values. The dataclass default is only reachable if you construct KVarNConfig() directly rather than via from_cache_dtype. 5) No kvarn_mla_* presets anywhere — explicitly not ported (config.py absent, __init__.py:12-13, kvarn-0.27.1.patch:9, :92-96). The MLA guard in platforms/cuda.py therefore protects against a dtype string that this tree cannot produce, but which an MLA model would route by prefix. 6) I did not read the four files that consume this config — kvarn_attn.py, kvarn_decode.py, kvarn_store.py, triton_kvarn_decode.py, triton_kvarn_sinkhorn.py — they exist in the tree but are outside my slice, so claims here about kernel behaviour are quoted from comments in my files, not verified against the kernels. 7) No test files for config.py or sinkhorn.py exist anywhere in the kvarn/ directory, despite sinkhorn.py:25-26 stating it is "intentionally pure (no model imports, no vLLM context) so that `pytest` can exercise it as a unit." 8) docs/long-context.md is referenced from README.md:46 but is outside the slice, so the 262k/perplexity numbers are unverified here.

---

## EXISTS, NEVER SET — 3

### Weight-aware pool budget: share of (util*total − weights) instead of a fixed slice of the card
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:225-249` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:193-206`

**What it does.** The fp16 pool's memory budget is computed as a share of the memory that actually remains after model weights, not as a fraction of total GPU memory. This auto-scales concurrency: a small model on a big card gets a large pool, a model that nearly fills the card gets a small one and degrades toward cap=1 rather than OOMing. The old fraction-of-total path survives as a fallback for when weight size cannot be read.

**Mechanism.** `pool_budget_bytes(total_gpu_bytes, gpu_memory_utilization, weight_bytes)` at config.py:225. If both weight_bytes and gpu_memory_utilization are supplied it computes `usable = gpu_memory_utilization * total_gpu_bytes - weight_bytes` and returns `max(0, int(share * usable))` with share from `KVARN_POOL_MEM_FRAC` or `POOL_USABLE_SHARE_DEFAULT` (config.py:244-247). Otherwise it returns `int(total_gpu_bytes * frac)` with frac from the same env var or `POOL_MEM_FRAC_DEFAULT` (config.py:248-249) — note the single env var is reinterpreted depending on which path runs (config.py:203-204: 'interpreted as share-of-usable when weights are known, else fraction-of-total').

**Why they needed it.** A named regression, config.py:195-199: 'Sizing the pool as a fixed fraction of *total* GPU memory was the bug behind issue #15: on a 4B/24GB card the pool got 0.08·24≈1.9 GB and concurrency capped to ~30 while the KV cache sat at ~3% utilization — ~10 GB of usable memory wasted.' The stated principle at config.py:192-194: 'The pool and the paged KV cache draw from the SAME pot: the memory left after model weights.'

**Their numbers.** POOL_MEM_FRAC_DEFAULT = 0.08 (legacy fraction of total, fallback); POOL_USABLE_SHARE_DEFAULT = 0.5 (share of post-weight usable) — config.py:205-206. Issue #15 case: 4B model, 24 GB card, pool 1.9 GB, concurrency ~30, KV cache ~3% utilised, ~10 GB wasted.

**llama.cpp — EXISTS, NEVER SET.** The principle — budget from what remains after weights, not from a fraction of the card — is exactly what --fit already does: it loads with no_alloc, reads the memory breakdown, and targets free_memory minus the per-device margin. So llama.cpp does not have KVarN's issue-#15 bug. What it does have is the margin sitting at a 1 GiB default that this profile has never touched, which is the same class of waste KVarN was fixing (memory reserved for nothing).

**Equivalent here:** --fit (on) + -fitt / --fit-target MiB (never set — default 1024 MiB per device)

**Evidence (llama.cpp):** `common/common.h:473` · `common/arg.cpp:2851-2874` · `common/fit.cpp:559-563` · `common/fit.cpp:290-373`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** -fitt is at its default 1024 MiB, so ~1 GiB of a 12 GB card is deliberately forfeited. Lowering it (e.g. -fitt 384) hands ~640 MiB back to --fit, which spends it on n_ctx and layer placement. That is a real, unmeasured lever on this exact profile. Risk: --fit measures FREE VRAM at that instant, and this box's free-at-boot moves 9,326-10,732 MiB, so a tight margin will OOM on a bad boot.

### V2-runner fix 1 — drafter's sliding-window layers forced back to bf16 via a copied cache_config
**Where (theirs):** `kvarn/kvarn-v2-runner.patch:81-94` · `kvarn/kvarn-v2-runner.patch:5-7`

**What it does.** In the DFlash drafter model, if the engine's cache_dtype is a kvarn preset, the Attention layer is constructed with a shallow copy of cache_config whose `cache_dtype` is reset to `"auto"`, so the drafter's SW attention uses the normal bf16 path.

**Mechanism.** In qwen3_dflash.py, before `self.attn = Attention(...)`: `if cache_config is not None and str(getattr(cache_config, "cache_dtype", "auto")).startswith("kvarn"): cache_config = _kv_copy(cache_config); cache_config.cache_dtype = "auto"` using `from copy import copy as _kv_copy` (kvarn-v2-runner.patch:87-91). The copy is what keeps the mutation local to this layer.

**Why they needed it.** kvarn-v2-runner.patch:6-7: 'the SW backend path has no KVarN support, and the drafter doesn't need it.' The drafter's KV is short-lived and re-derived every step, so compressing it buys nothing.

**llama.cpp — EXISTS, NEVER SET.** common_base_params_to_speculative overwrites result.cache_type_k/v with the speculative struct's own f16 defaults unconditionally, so the drafter is already insulated from the target's quantised cache the way the vLLM patch had to be made to do by hand. The interesting direction here is the opposite one, and it is a flag we hold and have never set. Caveat before trying it: a quantised draft cache needs the draft's own head dim to be divisible by the block size (32) and requires FA on the draft context, and this is a quality knob on the drafter — it will show up as lower acceptance, not as an error.

**Equivalent here:** -ctkd / --spec-draft-type-k and -ctvd / --spec-draft-type-v (default f16 for both, never set here)

**Evidence (llama.cpp):** `common/speculative.cpp:2405-2406` · `common/common.h:340-341` · `common/arg.cpp:4022-4047` · `src/llama-context.cpp:142-161` · `src/llama-context.cpp:385-396`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Live now that draft-dflash is in use. llama.cpp already applies KVarN's fix by default — -ctk q4_0 does NOT propagate to the draft, so the draft cache is f16. The unexplored lever is the inverse: if the dflash sidecar gets its own memory module, -ctkd q4_0 -ctvd q4_0 would cut its KV ~2.8x. Whether that is any bytes at all depends on whether this sidecar ships tok_embd/output — if it does not, ctx_other is honoured and it shares the target's cache, and both flags do nothing. Verifiable from the startup log: a second 'KV self size' line means a separate draft cache.

### Complete tunable surface: six environment variables, all with documented defaults and consequences
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:132` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:243` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:308` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:435-436` · `kvarn/kvarn-0.27.1.patch:138`

**What it does.** Every runtime knob is an env var read at config-construction time, with the default inline and the reason for the default in the adjacent comment. There is no config file and no CLI surface beyond the dtype string and standard vLLM flags.

**Mechanism.** KVARN_POW2_SLOT (default "0", config.py:132) — restores power-of-two per-token slot padding, only consulted at head_dim>=256. KVARN_POOL_MEM_FRAC (config.py:243) — dual-meaning: share-of-usable when weights are known (default 0.5), fraction-of-total otherwise (default 0.08). KVARN_FA_SCRATCH_CAP (default 262144, config.py:308) — token cap of the shared FA materialize scratch. KVARN_SINKHORN_ITERS (default "8", config.py:435). KVARN_SINK_TOKENS (default "128", config.py:436). KVARN_QUANT_SLIDING (== "1", kvarn-0.27.1.patch:138) — experimental, quantize SWA layers. All five config.py reads happen in property/classmethod bodies or `from_cache_dtype`, so they are re-read per construction rather than captured at import.

**Why they needed it.** The pool cap warning tells the operator which of these to reach for and why: 'To raise it: increase --gpu-memory-utilization or set KVARN_POOL_MEM_FRAC higher (the pool, not KV capacity, is the limit here)' (kvarn-0.27.1.patch:192-194). The FA cap comment tells them when NOT to: 'lower it only if KV capacity is the constraint' (config.py:302).

**Their numbers.** 6 env vars. Defaults: KVARN_POW2_SLOT=0, KVARN_POOL_MEM_FRAC={0.5 usable-share | 0.08 total-frac}, KVARN_FA_SCRATCH_CAP=262144, KVARN_SINKHORN_ITERS=8, KVARN_SINK_TOKENS=128, KVARN_QUANT_SLIDING unset.

**llama.cpp — EXISTS, NEVER SET.** The technique is really 'the whole tunable surface is env vars with documented defaults', and llama.cpp's equivalent surface exists and is largely untouched by this profile. This is the exists-but-unused verdict with the most immediate cash value, because every one of these is a single environment variable on an existing binary with no rebuild and no patch.

**Equivalent here:** LLAMA_ATTN_ROT_DISABLE, GGML_CUDA_GRAPH_OPT, LLAMA_GRAPH_REUSE_DISABLE, GGML_CUDA_DISABLE_GRAPHS, GGML_OP_OFFLOAD_MIN_BATCH, LLAMA_SERVER_SLOTS_DEBUG (plus LLAMA_ARG_* for every flag)

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:313-315` · `ggml/src/ggml-cuda/ggml-cuda.cu:4318-4344` · `ggml/src/ggml-cuda/ggml-cuda.cu:4342` · `src/llama-context.cpp:278-286` · `ggml/src/ggml-cuda/common.cuh:1255-1259` · `tools/server/server-context.cpp:1283-1289`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Three of these are unset here and worth a round each. GGML_CUDA_GRAPH_OPT=1 turns on multi-stream QKV concurrency; it requires CUDA graphs plus exactly one CUDA device, which this box satisfies — the cheapest untried experiment in the whole map, though note it only fires on single-row (decode-shaped) nodes, so speculative verify steps of 3+ tokens may not benefit. LLAMA_ATTN_ROT_DISABLE=1 prices the Hadamard rotation on -ctk q4_0 (expect a quality loss, possibly a small speed gain). LLAMA_GRAPH_REUSE_DISABLE=1 and GGML_CUDA_DISABLE_GRAPHS are the attribution probes for technique 40. LLAMA_SERVER_SLOTS_DEBUG makes /slots return prompt and generated text. All are noise-floor-bound: effects below 13.6 % are not real here.

## absent, has a seam — 2

### sink_tokens: first 128 tokens per request never quantized, replacing layer-level boundary skipping
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:65-66` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:436` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:411-421`

**What it does.** A per-request token-level fp16 exemption: the first N tokens of every request stay in fp16 permanently. This is the shipped mechanism; the inherited layer-level scheme (keep the first-N and last-N transformer layers in fp16, mirroring TurboQuant) is present but defaulted off.

**Mechanism.** `sink_tokens: int = 128` at config.py:65 with the comment 'first N tokens per request stay fp16 (NEVER quantised)', overridable via `KVARN_SINK_TOKENS` (config.py:436). `boundary_skip_layers: int = 0` at config.py:66, 'layer-level skipping off by default; sink_tokens replaces it'. The layer helper `get_boundary_skip_layers(num_layers, n=2)` (config.py:411-421) still exists: it returns `[str(i)]` for `range(n)` plus `range(num_layers-n, num_layers)`, clamped by `n = min(n, num_layers // 2)`, formatted for vLLM's `kv_cache_dtype_skip_layers`, and explicitly 'Mirrors TurboQuant (`TurboQuantConfig.get_boundary_skip_layers`)'. Note the docstring at config.py:55-57 still describes the old default of 2 while the field is 0.

**Why they needed it.** The attention-sink literature says the earliest tokens absorb disproportionate attention mass and quantizing them is disproportionately damaging. Doing it per-token rather than per-layer costs one fp16 block per request per layer (already the pool's structural sink slot, config.py:186-187) instead of an entire layer's cache. The layer variant is kept as an inherited escape hatch: 'Default 2 mirrors TurboQuant's default' (config.py:56-57).

**Their numbers.** sink_tokens default 128 (= one g128 tile); boundary_skip_layers default 0, inherited default was 2.

**llama.cpp — absent, has a seam.** Per-token exemption is foreclosed: one ggml tensor per layer holds every cell in one ggml_type, so two precisions cannot coexist in a cache. Per-LAYER is a nameable seam and I checked it in source — the k/v tensors are created inside the per-layer loop at llama-kv-cache.cpp:229-231 using the single ctor arguments type_k/type_v, and kernel selection is already per FLASH_ATTN_EXT node (ggml_cuda_get_best_fattn_kernel runs fresh per node), so a per-layer type vector would not break dispatch. It would need a new llama_context_params field, plumbing through llama_memory_hybrid's attention half, and the graph builder. Large patch, uncertain payoff, and it is upstream-hostile.

**Equivalent here:** none; cparams.type_k / type_v are a single value for the whole context

**Evidence (llama.cpp):** `common/common.cpp:1727-1728` · `src/llama-kv-cache.cpp:229-231` · `ggml/src/ggml-cuda/fattn.cu:330-336`

**Effort:** large-patch · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** unknown. The per-token variant is impossible; the per-layer variant is the interesting one and would let the first and last attention layers run f16 while the rest stay q4_0, at a VRAM cost proportional to (2 * f16 - 2 * q4_0) layers. Cannot be estimated without this model's n_embd_k_gqa and attention-layer count.

### V2-runner fix 7 — three-part NaN fix in the DFlash2 speculator, traced to KVarN quantization noise
**Where (theirs):** `kvarn/kvarn-v2-runner.patch:256-294` · `kvarn/kvarn-v2-runner.patch:34-52`

**What it does.** On ultra-peaked distributions the candidate selector's scores can become NaN. Three independent guards: sanitize scores to finite values before sampling and caching, clamp the Triton path-walk index into range, and zero-initialize the selector token buffer instead of leaving it uninitialized.

**Mechanism.** (a) `scores = torch.nan_to_num(scores, nan=-1e30, posinf=1e30, neginf=-1e30)` inserted before `self._sample_path(...)` and `self._cache_draft_logits(...)` (kvarn-v2-runner.patch:292), noted as 'Pure GPU op, cudagraph-capture-safe' (:291). (b) In the Triton kernel, `index = tl.where(index >= top_k, 0, index)` before the candidate load (:266) — 'NaN scores make `scores == best` match nowhere, so index becomes BLOCK_K and the load below reads past the candidate row' (:262-264); the recovery is benign because 'a wrong draft is just rejected by the verify' (:265). (c) `self._selector_tokens = torch.zeros(...)` replacing `torch.empty(...)` (:277) — 'an unwritten slot must hold a valid token id, never uninitialized VRAM' (:275-276).

**Why they needed it.** The two failure modes are distinct and one was silent: 'NaN made the path-walk kernel index past the candidate row (garbage token id -> the illegal memory access in the replayed decode graph on labd copy tasks) and poisoned the cached draft logits the rejection sampler divides by (silently WRONG tokens: greedy copy degenerated into repetition)' (kvarn-v2-runner.patch:38-42). The causal link to this slice: 'KVarN quantization noise on verbatim-reproduction content tips them over, which is why CTX=fast never showed it' (:36-37).

**Their numbers.** Validation: '6/6 labd tasks pass at 240k, copy output verbatim (599/600 chars vs the source, identical to MTP), GSM8K 97.0% (n=200 on the shipped config, author reference 96.0-96.5), needle + prefix resume unchanged' (kvarn-v2-runner.patch:43-46).

**llama.cpp — absent, has a seam.** I read common/speculative.cpp:1219-1283 directly. The DFlash2 path asserts the lattice pointer is non-null but never checks the score values; std::exp((scores[k]-max)/temp) with a NaN score gives NaN probs, sum NaN, and std::discrete_distribution over NaN weights is undefined. The greedy branch's 1.0f/sum < p_min comparison is simply false for NaN, so it never early-stops. A three-line std::isfinite sanitise on `scores` before the softmax at speculative.cpp:1238 is a named, small, local seam — the same fix vLLM applied. Before patching upstream code this would need a reproduction: verbatim-copy-style content at temperature 0.8 with draft-dflash, watching for degenerate repetition or an acceptance rate that does not match the trace counters.

**Equivalent here:** none — the seam is the DFlash2 selector block in common/speculative.cpp

**Evidence (llama.cpp):** `common/speculative.cpp:1219-1258` · `common/speculative.cpp:1238-1258` · `common/sampling.cpp:722-793` · `tools/server/server-context.cpp:3825-3831` · `common/common.h:386-392` · `src/llama-arch.cpp:1044-1055`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The highest-value item in this slice for us, because we run llama.cpp's DFlash2 (this build IS PR #27342) with q4_0 KV — the same combination vLLM traced the fault to. I read the selector: there is no non-finite guard. Two divergences matter. (1) llama.cpp's OOB read cannot happen — `predecessor` comes from std::max_element or std::discrete_distribution and is bounded by selector_top_k, so vLLM's fix (b) has no analogue-need. (2) llama.cpp's silent-corruption path is live: with temperature > 0 the selector fills dp.dists, and dists are the ONLY trigger for the residual/maximal-coupling accept rule, which divides by q(draft[i]). Default temp is 0.80, draft-dflash is active, and QWEN35 supports RS rollback — all three conditions for that path hold on this profile today. A NaN in the lattice would make acceptance decisions on NaN comparisons and produce wrong accepted tokens with no error and no acceptance-rate signature. At temperature 0 no dists are filled and a garbage draft is merely rejected, which is harmless.

## partial — 6

### Preset registry keyed by cache-dtype string (k/v bits + tile size in the name)
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:21-26` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:423-444`

**What it does.** Four named presets map a `--kv-cache-dtype` string to a frozen {key_bits, value_bits, group} triple. `from_cache_dtype(cache_dtype, head_dim)` is the single constructor: it validates the string against the dict, raises a ValueError listing every valid key on a miss, and builds the dataclass. The trailing `g<N>` in the name encodes the variance-normalisation tile size, so the block-size contract is visible in the flag the operator types.

**Mechanism.** `KVARN_PRESETS: dict[str, dict]` at config.py:21 holds `kvarn_k4v2_g128`, `kvarn_k4v4_g128`, `kvarn_k4v2_g64`, `kvarn_k4v4_g64`. `KVarNConfig.from_cache_dtype` (config.py:423) does a dict membership test, then splices preset['key_bits'], preset['value_bits'], preset['group'] plus the caller-supplied head_dim and two env-derived fields into the dataclass. Bit width is never hardcoded downstream — the comment at config.py:15-19 states it 'is fully parameterized in the quantizer and kernels (key_bits / value_bits), and the tile size flows through cfg.group everywhere (storage layout, Triton GROUP constexpr, flush / slot math), so additional presets are a one-line addition here.'

**Why they needed it.** vLLM identifies a KV cache format by a single string literal, so every tunable that changes the on-disk layout has to be reachable from that string. Making the preset the only entry point means the byte layout can never disagree with the dtype the engine registered.

**Their numbers.** 4 presets; g128 described as 'the current design point'; g64 'trades a little compression (more per-tile scale overhead per token) for finer quantization granularity (each tile's scales adapt to fewer tokens)' (config.py:11-13)

**llama.cpp — partial.** llama.cpp already has the shape of this: one operator-typed string is validated against a closed registry and throws with the valid list on a miss (kv_cache_type_from_str). What it does not have is the registry entry carrying a (key_bits, value_bits, group) triple — the string names a single ggml_type applied to both K and V, and this build additionally forbids K != V (ggml/src/ggml-cuda/fattn.cu:442-446). So the pattern is present, the content it would carry is not.

**Equivalent here:** -ctk/-ctv TYPE, validated against the kv_cache_types whitelist

**Evidence (llama.cpp):** `common/arg.cpp:305-315` · `common/arg.cpp:317-324` · `common/arg.cpp:2427-2451`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** none directly — we already type the string that selects the KV format

### Asymmetric key/value bit budget, justified by where quantization error lands in softmax
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:18-20` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:22`

**What it does.** The shipped preset spends 4 bits on keys and only 2 on values, rather than splitting the budget evenly. This is a deliberate error-propagation argument, not a tuning result.

**Mechanism.** Encoded purely as the preset constants `{"key_bits": 4, "value_bits": 2}` at config.py:22, which then drive `k_packed_bytes` (config.py:70-72) and `v_packed_bytes` (config.py:75-77) independently, so K storage is 2x V storage per tile at the same head_dim and group.

**Why they needed it.** Verbatim from config.py:18-20: 'Keys carry more quantization sensitivity than values (key error propagates through the softmax exponentials, value error is averaged out by the softmax weights), so the shipped preset spends more bits on keys.'

**Their numbers.** k4v2: keys 4 bits, values 2 bits. At head_dim=128, group=128: K = 128*128*4/8 = 8192 B, V = 128*128*2/8 = 4096 B per (block, head).

**llama.cpp — partial.** -ctk and -ctv are genuinely separate flags and the parser accepts different values, so the asymmetric budget is expressible. It is the kernel table that forecloses it: with GGML_CUDA_FA_ALL_QUANTS=OFF only four symmetric vec instances are compiled (f16/f16, q4_0/q4_0, q8_0/q8_0, bf16/bf16) and K->type != V->type returns BEST_FATTN_KERNEL_NONE before the type check. Rebuilding with -DGGML_CUDA_FA_ALL_QUANTS=ON compiles the mixed table at fattn.cu:265-319 and makes -ctk q8_0 -ctv q4_0 reachable. That is a CMake flag plus compile time, not a patch.

**Equivalent here:** -ctk q8_0 -ctv q4_0 (flags exist; kernel does not in this build)

**Evidence (llama.cpp):** `common/arg.cpp:305-315` · `ggml/src/ggml-cuda/fattn.cu:442-446` · `ggml/src/ggml-cuda/fattn.cu:321-325` · `ggml/CMakeLists.txt:208` · `build-dflash2/CMakeCache.txt:660`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Direction of the argument transfers. q4_0 is 18 B / 32 elems = 0.5625 B/elem, q8_0 is 34 B / 32 = 1.0625 B/elem, so K-half cache grows 1.89x and total attention KV ~1.47x. On ~9.5 GB free with a 6.77 GB model that will cost context — --fit would pick a smaller n_ctx. Whether the quality gain pays for the lost context is unknown and would have to be measured paired within one round.

### Four-stage quantization pipeline with RTN scale absorbed into the Sinkhorn axis
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:33-41` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:79-97`

**What it does.** Per (block, head): Hadamard rotation along head_dim, then iterative log-domain variance normalisation over the tile, then asymmetric per-row round-to-nearest, then the per-row RTN scale and zero-point are folded into the matching Sinkhorn scale axis so dequantization is a two-factor product rather than three.

**Mechanism.** config.py:34-41 documents the four steps. The absorption is asymmetric by tensor: for K the RTN scale/zp go into the per-channel axis (`s_col_K' = rtn_scale ⊙ s_chan_sinkhorn`, `zp_K' = rtn_zp ⊙ s_chan_sinkhorn`, config.py:83-84) leaving `s_row_K = s_tok_sinkhorn` untouched; for V they go into the per-token-in-tile axis (`s_row_V' = rtn_scale ⊙ s_tok_sinkhorn`, `zp_V' = rtn_zp ⊙ s_tok_sinkhorn`, config.py:94-95) leaving `s_col_V = s_chan_sinkhorn` untouched. Reconstruction is stated once at config.py:41: `x = (q * absorbed_scale + absorbed_zp) * other_scale`.

**Why they needed it.** Storing rtn_scale, rtn_zp, s_col and s_row separately would mean four fp16 vectors per tensor per tile and a three-multiply dequant in the inner loop. Absorption collapses it to three vectors and two multiplies without losing exactness, because the RTN scale is per-row and one Sinkhorn axis is already per-row.

**Their numbers.** K scales = (2*head_dim + group) fp16 values; V scales = (head_dim + 2*group) fp16 values (config.py:87, :97)

**llama.cpp — partial.** Read the source: attn_rot_k = !disable && ggml_is_quantized(type_k) && n_embd_head_k % 64 == 0, with a Hadamard matrix precomputed per power-of-two size and the K-shift path un-rotating, RoPEing and re-rotating. So llama.cpp already ships stage 1 of this pipeline, on by default, for exactly the reason KVarN does it. Stage 3 (asymmetric per-row RTN) is the ggml block quant itself. Stages 2 (iterative variance normalisation) and 4 (scale absorption) have no home: a ggml_type carries exactly one scale (plus min) per block, so a scheme with a per-token vector and a per-channel vector would require a new ggml quant type plus matching dequant in every FA kernel family. That is a new backend, not a patch.

**Equivalent here:** automatic Hadamard rotation of quantised K/V (stage 1) + ggml block RTN scale (stage 3); LLAMA_ATTN_ROT_DISABLE=1 to turn stage 1 off

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:313-336` · `src/llama-kv-cache.cpp:337-338` · `src/llama-kv-cache.cpp:20-57` · `src/llama-kv-cache.cpp:1863-1878`

**Effort:** new-backend · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Stage 1 is already earning on this profile if n_embd_head_k % 64 == 0 — check the two startup lines 'attn_rot_k = %d' / 'attn_rot_v = %d'. LLAMA_ATTN_ROT_DISABLE=1 is a free A/B to price what the rotation is buying at -ctk q4_0. Stages 2 and 4 are worth nothing here without new kernels.

### max_supported_seqs: invert the pool formula and cap the scheduler rather than size the pool to max_num_seqs
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:251-276` · `kvarn/kvarn-0.27.1.patch:179-200`

**What it does.** Instead of allocating whatever pool `max_num_seqs` implies (which OOMs at large values and exhausts at small ones), KVarN picks a memory budget first and then lowers vLLM's `max_num_seqs` to the largest value that budget supports. The result is a pool that is simultaneously OOM-safe and exhaustion-safe with no per-model tuning.

**Mechanism.** `max_supported_seqs` (config.py:251) computes `max_slots = budget / (slot_bytes_per_layer * num_layers)`, then solves `2*S + prefill_blocks + 8 <= max_slots` for S, returning `max(1, (max_slots - prefill_blocks - 8) // 2)` (config.py:274-276) — the exact algebraic inverse of `pool_slots`. Passing `frac` explicitly forces the legacy fraction-of-total path (config.py:268-269). The wiring is in `CudaPlatformBase.check_and_update_config`: kvarn-0.27.1.patch:179-186 calls it with the real num_kv_heads / layer count / max_num_batched_tokens / gpu_memory_utilization / weight_bytes, and :187-200 overwrites `scheduler_config.max_num_seqs` with a warning naming the old value, the new value, and which budget kind was used.

**Why they needed it.** config.py:189-192: 'Rather than size the pool to an arbitrary `max_num_seqs` (which can OOM at large values or exhaust if under-sized), we pick a memory budget and cap the scheduler's concurrency to what that budget supports.' The patch adds the operator-facing half: the warning explicitly says 'the pool, not KV capacity, is the limit here' so nobody chases the wrong knob (kvarn-0.27.1.patch:194).

**Their numbers.** Always returns >= 1 (config.py:276) — the degenerate case is a single-sequence engine, never a crash.

**llama.cpp — partial.** The pattern (pick a budget, then solve it for the knob that would otherwise OOM) is present in llama.cpp but applied to n_ctx rather than n_seq_max: --fit interpolates n_ctx on measured bytes-per-token between -fitc and n_ctx_train. Extending it to lower n_parallel would go in common/fit.cpp between the ctx-reduction step and the layer-assignment step, since cparams.n_seq_max is set from params.n_parallel in the same path — a named seam, but with -np 1 there is nothing to gain.

**Equivalent here:** --fit solves the budget for n_ctx (fit.cpp:290-373), never for n_parallel

**Evidence (llama.cpp):** `common/fit.cpp:290-373` · `common/fit.cpp:344` · `common/common.cpp:1698` · `tools/server/server.cpp:151-155`

**Effort:** small-patch · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** none at -np 1 — there is no concurrency to cap. The n_ctx analogue is already automatic.

### Documented non-fix: PIECEWISE cudagraph mode required for DFlash2 verify under prefix caching
**Where (theirs):** `kvarn/kvarn-v2-runner.patch:54-65`

**What it does.** The patch explicitly declines to fix one interaction and pushes it onto the caller: with prefix caching on, the DFlash2 verify step must not run inside a captured CUDA graph. PIECEWISE is the recommended mode because it keeps the compiled graphs and leaves only the multi-query verify uncaptured.

**Mechanism.** Not code — a stated operating constraint at kvarn-v2-runner.patch:54-60. `single-user/start_qwen.sh` runs `SPEC=dflash2 CTX=huge` with `cudagraph_mode=PIECEWISE`, and 'anyone driving vllm serve directly needs the same' (:59-60). The diagnosis is isolated: 'It is the capture rather than the drafter -- eager is clean, and so is PIECEWISE' (:57-58).

**Why they needed it.** Without it 'Acceptance collapses to ~1 token per step, and on some trees the output degrades outright' (:55-56). The trade is quantified rather than hidden, and the reason for taking it stated: 'The capture mode is fixed at boot, so this mode takes the trade that matches long shared prefixes' (:64-65).

**Their numbers.** 3 runs each at 250 W. Long context (labd copy@20k): 1.97 -> 7.83 tok/step and 38 -> 132 tok/s. Short-prompt decode drops 13-18%: de/en/code 78/125/202 -> 74/102/176 tok/s.

**llama.cpp — partial.** vLLM's problem was that capturing the verify collapsed acceptance; llama.cpp's is the mirror image — a variable step size means it never captures at all, silently, with the cost showing up only as a lower 'graphs reused'. The capture-mode knob does not exist here, but the diagnosis does and it is one log line plus two env kill switches.

**Equivalent here:** CUDA graph capture warmup + llm_graph_params::allow_reuse; the 'graphs reused' line in print_timings; GGML_CUDA_DISABLE_GRAPHS / LLAMA_GRAPH_REUSE_DISABLE as attribution probes

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:4249-4274` · `src/llama-graph.h:785` · `src/llama-context.cpp:1332-1372` · `tools/server/server-context.cpp:617-619` · `common/speculative.cpp:1234-1274` · `common/speculative.cpp:1992-2004`

**Effort:** one-flag · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Directly measurable and never measured here. llama.cpp captures a CUDA graph only after two consecutive calls with identical node properties, and allow_reuse requires ubatch.n_tokens to match. draft-dflash with p_min=0 and n_min=0 emits a CONSTANT draft length (I read the loop — it breaks early only if p_min > 0), so draft-dflash ALONE gives a constant 1+n_draft verify shape and can capture. ngram-mod is all-or-nothing at n_max=64, so it gives either a 65-token or a 1-token step. The measured +48.5% pair draft-dflash,ngram-mod therefore alternates two shapes and should never get past warmup. The 'graphs reused = %d' line in the completion timings is the free readout: compare it between draft-dflash alone (+34.7%) and the pair (+48.5%). If the pair still wins with reuse at ~0, that is a real finding about where the time goes.

### Robustness fixes in the online-softmax kernels and CUDA-graph verify plan
**Where (theirs):** `kvarn/README.md:40-42`

**What it does.** Four small hardening changes listed as a group: NaN guards in the online-softmax kernels for fully-masked chunks and all-empty split-K rows, elimination of per-context recompiles of the packed-KV kernel, and zeroing of verify-plan padding for CUDA-graph replays.

**Mechanism.** Enumerated at README.md:40-42: 'Small robustness fixes: NaN guards in the online-softmax kernels for fully-masked chunks / all-empty split-K rows, no per-context recompiles of the packed-KV kernel, verify-plan padding zeroed for CUDA-graph replays.' Implementations are in the Triton kernel files outside this slice.

**Why they needed it.** Each is a class of bug that only surfaces at the edges: a fully-masked chunk produces 0/0 in the online-softmax rescale; a split-K row with no assigned work has no max to combine; per-context recompiles cost latency proportional to the number of distinct context lengths seen; and uninitialized padding is stable-garbage under graph replay rather than random, so it fails reproducibly-wrongly. Note the same 'uninitialized buffer under graph replay' pattern independently appears as the zeros-not-empty fix in kvarn-v2-runner.patch:274-280.

**Their numbers.** 4 fixes

**llama.cpp — partial.** Two of the three parts have clear answers. Per-context recompiles: llama.cpp caches captured graphs in a per-context map keyed by the split's first node, with a uid fast path and a 10 s eviction — the cost structure is different but the concern is handled. Uninitialised plan padding under graph replay: llama.cpp builds the speculative verify batch fresh each step into common_batch, so there is no persistent padded plan to zero. The third — whether ggml-cuda's FA kernels produce NaN on a fully-masked chunk or an empty split-K row — is a kernel-numerics question the map does not cover, and I will not guess at it.

**Equivalent here:** CUDA graph keying/warmup covers the recompile half; the verify batch is rebuilt per step so there is no persistent plan padding; the FA-kernel NaN question is unjudged

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:2574-2591` · `ggml/src/ggml-cuda/common.cuh:1426-1444` · `tools/server/server-context.cpp:488-496`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** unknown — I cannot judge the kernel numerics from the map, and it is listed in map_gaps

## already have it — 8

### Closed-form byte-offset properties as the single source of the tile layout
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:150-181` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:69-107`

**What it does.** Every field inside a tile has a derived-property byte offset, computed by chaining from the previous field's offset plus its size. The Triton kernels read these rather than carrying their own layout constants, so a preset change relocates every field consistently.

**Mechanism.** Eight chained properties: `k_packed_offset`=0 (config.py:152), `k_s_col_offset`=+k_packed_bytes (:157), `k_zp_offset`=+head_dim*2 (:161), `k_s_row_offset`=+head_dim*2 (:165), `v_packed_offset`=+group*2 (:169), `v_s_col_offset`=+v_packed_bytes (:173), `v_s_row_offset`=+head_dim*2 (:177), `v_zp_offset`=+group*2 (:181). Sizes come from `k_packed_bytes` = ceil(head_dim*group*key_bits/8) (:72), `v_packed_bytes` = ceil(group*head_dim*value_bits/8) (:77), `k_scale_bytes` = (2*head_dim+group)*2 (:87), `v_scale_bytes` = (head_dim+2*group)*2 (:97); `tile_bytes` sums all four (:100-107).

**Why they needed it.** The layout must be identical in the store kernel, the decode kernel and the page-size accounting in two separate vLLM files. Deriving it once from (head_dim, key_bits, value_bits, group) is what makes 'additional presets are a one-line addition' (config.py:18) true.

**llama.cpp — already have it.** The property KVarN buys with eight chained offset properties — one derivation of the layout that the kernels, the allocator and the accounting all read — llama.cpp gets structurally from the ggml type system: the tensor is created with a ggml_type and every size is ggml_row_size of that type. There is no second copy of the layout to drift.

**Equivalent here:** ggml type traits (blck_size / type_size) + ggml_row_size / ggml_nbytes

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:229-231` · `src/llama-context.cpp:3613-3633`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Pool sized by attention-layer count only, not total layers (hybrid-model correction)
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:318-333` · `kvarn/kvarn-0.27.1.patch:172-178`

**What it does.** On a hybrid model the Mamba / linear-attention layers have no KV cache and therefore no KVarN pool. `num_kvarn_layers` asks the model config for the full-attention layer count specifically, so the pool is not over-reserved by the ratio of total layers to attention layers.

**Mechanism.** `KVarNConfig.num_kvarn_layers(model_config, parallel_config)` (config.py:318) calls `model_config.get_num_layers_by_block_type(parallel_config, "attention")` inside a bare try/except, returns it if truthy and positive, and otherwise falls back to `model_config.get_num_layers(parallel_config)` (config.py:327-333). The platform hook calls it at kvarn-0.27.1.patch:178 and feeds the result to `max_supported_seqs` as `num_layers`.

**Why they needed it.** config.py:321-324: 'On a hybrid model (Qwen3.5/3.6, Jamba, ...) the Mamba/linear-attention layers have no KVarN pool, so sizing the pool by ALL layers over-reserves it ~Nx and starves the Mamba/KV caches (OOM or cap collapse). For a dense transformer this equals total layers, so the dense path is unchanged.'

**Their numbers.** over-reservation factor stated as '~Nx' where N is the total/attention layer ratio

**llama.cpp — already have it.** For QWEN35/QWEN35MOE the memory is built as llama_memory_hybrid with filter_attn = !is_recr(il), so the attention KV cache is allocated only over full-attention layers and the recurrent half is sized by n_seq_max and n_rs_seq rather than by context. llama.cpp cannot make KVarN's ~Nx over-reservation mistake because there is no per-layer pool sized from a layer count in the first place.

**Equivalent here:** llama_memory_hybrid with filter_attn / filter_recr

**Evidence (llama.cpp):** `src/llama-model.cpp:2281-2303` · `src/llama-model.cpp:2305-2344` · `src/llama-memory-recurrent.cpp:99-101` · `src/models/qwen35.cpp:21-26`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none — already correct for Qwen3.5-style hybrids

### Weight size read off checkpoint files on disk, deliberately without a CUDA context
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:335-409` · `kvarn/kvarn-0.27.1.patch:157-171`

**What it does.** To make the budget weight-aware before the model loads, KVarN estimates per-rank weight bytes by stat-ing the checkpoint files. It resolves a local directory directly or the already-cached HF snapshot for a repo id (never downloading), prefers the shard set named in the safetensors/bin index manifest, and divides by tensor_parallel_size. It returns None on any failure so the caller silently falls back to the legacy budget.

**Mechanism.** Three-tier resolution in `estimate_weight_bytes` (config.py:335). Tier 1 (config.py:365-388): glob `**/*.{safetensors,bin}.index.json`, load `weight_map`, take `sorted(set(weight_map.values()))`, and sum those shards — but only `if names and all(os.path.exists(p) for p in shards)`, because 'a partial set would under-estimate the weights and over-grow the pool budget' (config.py:380-383). Tier 2 (:390-396): a canonical `model.safetensors` or `consolidated.safetensors`. Tier 3 (:398-407): sum every `*.safetensors`, else every `*.bin`. All three divide by `max(tensor_parallel_size, 1)`. Repo-id resolution uses `snapshot_download(model, local_files_only=True)` inside a try/except returning None (config.py:359-362).

**Why they needed it.** Two independent reasons. Timing: config.py:337-339 — the read must be 'exact, and cheap, with no CUDA context, which the early `check_and_update_config` hook must avoid'; the patch restates why at kvarn-0.27.1.patch:157-159, 'so we do NOT initialize a CUDA context in the parent here — doing so would force `spawn` multiprocessing for tensor parallelism.' Correctness: preferring the manifest 'avoids double-counting a repo that ships both a single consolidated checkpoint and the sharded HF set (e.g. Mistral-7B-Instruct-v0.3 carries `consolidated.safetensors` alongside `model-0000n-of-0000m.safetensors`, which a plain glob sums to ~2x the real weight size)' (config.py:345-349).

**Their numbers.** plain-glob double count on Mistral-7B-Instruct-v0.3 = ~2x real weight size

**llama.cpp — already have it.** --fit loads the model with mparams.no_alloc=true and load_mode=NONE, builds a context and reads llama_get_memory_breakdown. That is strictly better information than stat-ing checkpoint files (it is the real per-device allocation, not a file-size proxy) and it is free of KVarN's double-count hazard. The constraint that motivated the on-disk read — must not touch CUDA in the parent — does not exist in a single-process server.

**Equivalent here:** --fit's no_alloc probe load

**Evidence (llama.cpp):** `common/fit.cpp:29-70` · `common/common.cpp:1294-1302`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Patch hunk A — CacheDType literal registry
**Where (theirs):** `kvarn/kvarn-0.27.1.patch:1-16`

**What it does.** Adds the four kvarn preset strings to vLLM's `CacheDType` Literal in config/cache.py so `--kv-cache-dtype kvarn_k4v2_g128` validates at argument-parse time.

**Mechanism.** Four literal entries inserted after the turboquant block and before `int4_per_token_head` (kvarn-0.27.1.patch:9-13). install.sh asserts this landed by importing the Literal and checking membership: `assert "kvarn_k4v2_g128" in get_args(CacheDType)` (kvarn/install.sh:22-23).

**Why they needed it.** vLLM type-checks the dtype string against this Literal; an unlisted value is rejected before any KVarN code runs. This is the first of the seven hunks the README enumerates (kvarn/README.md:15-18).

**Their numbers.** 4 literals added

**llama.cpp — already have it.** The registration point exists and behaves the same way (validate at parse time, throw naming the value). The difference is that adding an entry to llama.cpp's list means adding a ggml_type with kernels, not a Python dataclass, and the list already advertises types this build cannot run (iq4_nl, q4_1, q5_0, q5_1 have no CUDA FA kernel here).

**Equivalent here:** const std::vector<ggml_type> kv_cache_types + kv_cache_type_from_str

**Evidence (llama.cpp):** `common/arg.cpp:305-315` · `common/arg.cpp:317-324`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Patch hunk G — platform-level head_dim gate, fail-fast at config time
**Where (theirs):** `kvarn/kvarn-0.27.1.patch:102-111`

**What it does.** In `check_and_update_config`, a kvarn dtype on a model whose head_dim is not 128, 256 or 512 raises a ValueError naming the dtype, the actual head_dim and the remedy, before any allocation happens.

**Mechanism.** `head_size = model_config.get_head_size()`; `if head_size not in (128, 256, 512): raise ValueError(...)` at kvarn-0.27.1.patch:105-111. The message ends with 'use a different --kv-cache-dtype for this model.'

**Why they needed it.** Stated at :103-104: 'Fail fast with a clear message rather than crashing deep in a kernel with a shape error otherwise.' The constraint itself is 'the variance-normalization tile is head_dim x group' (:102-103) — the Triton tile geometry assumes a power-of-two head dim in that range. config.py:48 independently says head_dim is a 'power of 2; tested at 128'.

**Their numbers.** supported head_dim: 128, 256, 512; config.py:48 says 'tested at 128'.

**llama.cpp — already have it.** llama.cpp fails fast on the same class of geometry error and with a named message: 'K cache type %s with block size %u does not divide n_embd_head_k=%u' returns nullptr from llama_init_from_model, and an unsupported head dim falls to BEST_FATTN_KERNEL_NONE. The one condition that degrades rather than errors — head dim not a multiple of 64 disabling the Hadamard rotation — is at least printed at INFO on every boot.

**Equivalent here:** K/V block-size divisibility check at context init; FA head-dim table; attn_rot_k/attn_rot_v startup lines

**Evidence (llama.cpp):** `src/llama-context.cpp:3613-3622` · `src/llama-context.cpp:3624-3633` · `ggml/src/ggml-cuda/fattn.cu:392-437` · `src/llama-kv-cache.cpp:337-338`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Actionable as a check, not a change: confirm from the startup log that attn_rot_k = 1 and attn_rot_v = 1 on this model. If head_dim is not a multiple of 64 the rotation is skipped silently-but-logged and -ctk q4_0 degrades more than it does elsewhere.

### SWA skip-layer append gated on the model actually having sliding-window layers
**Where (theirs):** `kvarn/kvarn-0.27.1.patch:112-152`

**What it does.** KVarN's decode path has no sliding-window mask, so sliding-window layers are pushed into `kv_cache_dtype_skip_layers` and kept full-precision. But the append only happens when the model genuinely has SWA layers, because a non-empty skip list has an expensive side effect in 0.27.1.

**Mechanism.** Reads `hf_text_config.layer_types`; if present, `_model_has_swa = "sliding_attention" in _layer_types`, else `_model_has_swa = hf_text_config.sliding_window is not None` (kvarn-0.27.1.patch:129-136) — explicitly 'Mirrors arg_utils' (:126). Then `elif "sliding_window" not in skip_layers and _model_has_swa: skip_layers.append("sliding_window")` plus an info log (:145-152). The patch header calls this out as 'the only behavioural change' from the 0.23 original (:78-80).

**Why they needed it.** Verbatim at :119-125: 'In 0.27.1 a non-empty kv_cache_dtype_skip_layers triggers Platform._align_heterogeneous_kv_block_size, which sizes the primary page with the FullAttentionSpec uint8 formula (not the packed KVarN slot) and re-pads every mamba page to it — on a hybrid model without SWA layers (Qwen3.5/3.8) that would double the GDN pages for nothing.' A defensive skip-list entry became a 2x memory regression on exactly the target model.

**Their numbers.** doubles the Gated-DeltaNet pages on a no-SWA hybrid

**llama.cpp — already have it.** The failure this guard prevents (a defensive skip-list entry silently doubling the recurrent pages on a no-SWA hybrid) cannot occur: llama.cpp has no per-layer KV dtype list at all, and on a model whose swa_type is NONE it routes to plain llama_memory_hybrid and prints 'swa_full is not supported by this model, it will be disabled'.

**Equivalent here:** the server force-disables --swa-full on a model with no SWA layers, with a warning

**Evidence (llama.cpp):** `tools/server/server-context.cpp:1188-1195` · `src/llama-model.cpp:2305` · `src/models/qwen35.cpp:21-26`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none — qwen35 declares no SWA, so the whole SWA surface is inert here

### MLA models excluded from the dense pool-sizing block by two independent guards
**Where (theirs):** `kvarn/kvarn-0.27.1.patch:88-97`

**What it does.** The entire dense KVarN config block is skipped both for `kvarn_mla*` dtype strings and for any model where `model_config.use_mla` is true, even if the dtype is a dense preset.

**Mechanism.** The condition at kvarn-0.27.1.patch:88-97 requires `cache_dtype.startswith("kvarn_")` AND `not cache_dtype.startswith("kvarn_mla")` AND `not getattr(model_config, "use_mla", False)`.

**Why they needed it.** The second guard is the non-obvious one, explained at :93-96: 'MLA models route ANY kvarn_ dtype (incl. kvarn_k4v2_g128) to the MLA latent-quant path, which has its own pool — the dense fp16 tail-pool sizing/skip-layers below must NOT run for them.' The dtype string alone cannot tell you which pool machinery will be used.

**llama.cpp — already have it.** llama.cpp has the analogous guard in the analogous place — an MLA model hard-errors with 'model does not support different K (%s) and V (%s) cache types' rather than silently taking the dense path — and the MLA cache is structurally distinct (has_v = !is_mla in the layer loop, so no V tensor is allocated at all). Nothing to port.

**Equivalent here:** MLA / DEEPSEEK4 refuse different K and V cache types at context init

**Evidence (llama.cpp):** `src/llama-context.cpp:3597-3600` · `src/llama-kv-cache.cpp:229-231`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none — Qwen3.8 is not MLA

### Pool materialization inside profile_run instead of a gpu_worker patch
**Where (theirs):** `kvarn/README.md:31-33`

**What it does.** The fp16 tail pools are allocated during vLLM's `profile_run` by doing a forward pass with `attn_metadata=None`, so vLLM's own memory profiler observes and charges for them. This removes the need for a hunk in gpu_worker.py that upstream KVarN required.

**Mechanism.** Described at README.md:31-33: 'Pools are materialized during `profile_run` (forward with `attn_metadata=None`) so vLLM's memory profiler charges them correctly — no `gpu_worker.py` hunk needed.' The implementation lives in kvarn_attn.py, outside this slice; config.py's `pool_bytes` docstring corroborates the intent — 'Reserved up front in the worker so the lazy pool allocation never pushes past the KV-memory limit' (config.py:286-287).

**Why they needed it.** Reducing the patch surface against upstream vLLM (fewer hunks to re-cut on a version bump) while getting correct accounting for free: memory allocated during profiling is automatically deducted from the KV-cache budget, so the pool and the paged cache cannot collectively overcommit.

**Their numbers.** eliminates 1 upstream hunk

**llama.cpp — already have it.** The property KVarN gets by allocating inside profile_run — everything transient is observed by the budget before the KV cache is sized — llama.cpp gets from the reserve sequence (FA probe, pp reserve, tg reserve, pp reserve again) plus --fit's no_alloc probe. The one hole is the documented one: common_speculative_init_result calls llama_model_load_from_file directly and never common_fit_params, so the draft is outside --fit and the server only reserves headroom against it.

**Equivalent here:** the pp/tg/pp reserve passes at context construction + the server's draft-footprint reservation

**Evidence (llama.cpp):** `src/llama-context.cpp:576-671` · `common/fit.cpp:29-70` · `common/speculative.cpp:2468` · `tools/server/server-context.cpp:1032-1087` · `common/arg.cpp:4117-4144`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none as a port. One adjacent unexamined item: --fit does not size the draft model, so with draft-dflash the sidecar loads at -ngld auto = all layers on GPU and the server compensates by shrinking the TARGET. -ngld is a flag we hold and have never set.

## impossible here — 3

### FA materialize-scratch cap that also caps the max_model_len term (upstream bug fix)
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:291-316`

**What it does.** The flash_attn_varlen fallback path packs the whole batch context into a shared fp16 K/V scratch. Rather than sizing it at the pathological max_num_seqs*max_model_len bound, it is capped at KVARN_FA_SCRATCH_CAP tokens — and critically, the `max_model_len` floor term is capped too, which the original was not.

**Mechanism.** `fa_scratch_cap()` (config.py:305-309) reads `KVARN_FA_SCRATCH_CAP` or returns `FA_SCRATCH_CAP_DEFAULT` = 262144. `fa_scratch_rows(max_num_seqs, max_model_len)` (config.py:311-316) returns `max(min(max_num_seqs*max_model_len, cap), min(max_model_len, cap), 4096)`. The inner `min(max_model_len, cap)` is the fix; there is also an unconditional 4096 floor. The comment at config.py:292-293 notes this was 'moved here from the backend's `_ensure_pool` so the cap is a single tunable.'

**Why they needed it.** config.py:298-300: 'the KVarN original used `max(min(S*L, CAP), L, 4096)`, i.e. the bare `max_model_len` term defeated the cap for a long-context deployment; the helper caps that term too.' The consequence of lowering it is documented rather than hidden: config.py:300-302, 'Contexts longer than the cap fall back to `_decode_path_slow` on the materialize route until segmented materialize lands (plan Phase 2), so lower it only if KV capacity is the constraint.'

**Their numbers.** FA_SCRATCH_CAP_DEFAULT = 262144 tokens, stated as '~1 GiB at Hk=4, D=256' (config.py:297); hard floor 4096 rows.

**llama.cpp — impossible here.** The exact analogue of KVarN's shared FA materialize scratch exists in ggml-cuda: MMA_F16 and TILE pass need_f16_K = need_f16_V = true unconditionally and launch_fattn converts the whole padded KV view. There is no environment variable and no flag to bound it, and no fallback path to degrade onto — capping it would mean changing the kernels. So the cap is not portable; the knowledge of the Q->ne[1] <= 2 cliff is.

**Equivalent here:** none — ggml_cuda_flash_attn_ext_get_alloc_size sizes the dequant scratch with no cap and no flag

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:469` · `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963` · `ggml/src/ggml-cuda/fattn-common.cuh:1022-1084` · `ggml/src/ggml-cuda/fattn.cu:534-568` · `src/llama-context.cpp:595`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Explains a real cost on this profile: with quantised KV, any FA call with Q->ne[1] >= 3 takes MMA_F16, which dequantises that layer's entire padded K and V to F16 (2 B/elem against q4_0's 0.5625). Speculative verify batches are 1+n_draft tokens, so draft-dflash at 15 puts every verify step on MMA. The adjacent live lever is --spec-draft-n-max 1 (verify batch of 2 query tokens, VEC kernel, no dequant) — almost certainly a net loss for throughput, but it is a one-flag probe that would price the dequant. VRAM is not the surprise here: the reserve pass always runs the MMA shape, so the scratch is already budgeted in the compute buffer at boot.

### Tile-orientation naming convention (s_col/s_row mean different things for K and V)
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/sinkhorn.py:16-23` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/__init__.py:6-8`

**What it does.** The Sinkhorn module is written axis-agnostically — `s_col` is always the axis-1 scale and `s_row` always axis-0 — and the per-channel / per-token semantics come entirely from how the caller orients the tile. K is fed as [D, group], V as [group, D], which is what makes keys per-channel-quantized and values per-token-quantized from one implementation.

**Mechanism.** Stated at sinkhorn.py:18-23: 'K tile is `[D, group]` (rows = channels, cols = tokens) -> `s_row` is per-channel, `s_col` is per-token. V tile is `[group, D]` (rows = tokens, cols = channels) -> `s_row` is per-token, `s_col` is per-channel.' The package docstring names the lineage: 'Keys are quantized per-channel (KIVI K-axis); values are quantized per-token (KIVI V-axis)' (__init__.py:7-8). config.py's scale-byte docstrings then use the semantic names (`s_chan_sinkhorn`, `s_tok_sinkhorn`) rather than col/row, at config.py:83-85 and :93-95.

**Why they needed it.** One implementation serves both tensors, and the transposition is free because the store kernel is writing the packed layout anyway. It also explains why `k_scale_bytes` and `v_scale_bytes` have mirrored shapes ((2D+g) vs (D+2g)): the absorbed pair always lands on whichever axis the RTN was per-row over.

**llama.cpp — impossible here.** I read the tensor creation: ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) — one row per cell, quantised in blocks of 32 along the channel axis. So llama.cpp is per-token for BOTH K and V, which is KIVI's V-axis applied to both. Changing K to per-channel would mean the cache is no longer row-per-token, breaking the cell allocator, seq_rm, K-shift and every FA kernel. llama.cpp's answer to the same problem is different and already shipped: rotate along head_dim so outlier channels are spread inside each block (llama-kv-cache.cpp:313-336, requires head dim % 64 == 0). Note also that llama.cpp already knows the axis matters — quantised V is refused without FA precisely because v_trans flips V's storage axis.

**Equivalent here:** none — both K and V are quantised along the embedding axis within one token

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:229-231` · `src/llama-kv-cache.cpp:313-336` · `src/llama-context.cpp:3602-3611` · `src/llama-model.cpp:2124`

**Effort:** new-backend · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** High explanatory value, no lever. It says llama.cpp uses the KIVI-wrong axis for K (scale shared across 32 channels of one token, not across tokens of one channel), which is why -ctk q4_0 is the more damaging half and why the Hadamard rotation exists at all. It also says the right asymmetry to try here is more bits on K, i.e. technique 2's -ctk q8_0 -ctv q4_0.

### Patch hunk C — backend registry entry and CUDA priority-list placement
**Where (theirs):** `kvarn/kvarn-0.27.1.patch:260-272` · `kvarn/kvarn-0.27.1.patch:54-71`

**What it does.** Adds `AttentionBackendEnum.KVARN` pointing at `vllm.v1.attention.backends.kvarn_attn.KVarNAttentionBackend`, and appends it to both branches of the CUDA platform's backend priority list, last in each.

**Mechanism.** One enum member at kvarn-0.27.1.patch:269. In platforms/cuda.py `_get_backend_priorities`, `AttentionBackendEnum.KVARN` is appended after TURBOQUANT in both the first list (:62) and the second (:70) — the patch context shows two symmetric branches. install.sh verifies the class actually resolves by calling `AttentionBackendEnum.KVARN.get_class().get_name()` and printing it (kvarn/install.sh:24-25).

**Why they needed it.** The backend must be discoverable by name for the selector to instantiate it. Placing it last means it is only chosen when the cache dtype demands it, never as a default over FLASH_ATTN / TRITON_ATTN.

**Their numbers.** appended in 2 places in the priority function

**llama.cpp — impossible here.** There is no pluggable attention-backend registry to add an entry to. Kernel choice is recomputed per FLASH_ATTN_EXT node from cc, head dims, K/V types, gqa_ratio and Q->ne[1], with no flag, env var or priority list. A KVarN-shaped backend would be a new kernel family plus a new selection branch in that function.

**Equivalent here:** none — ggml_cuda_get_best_fattn_kernel is a hardcoded decision tree

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:358-533` · `ggml/src/ggml-cuda/fattn.cu:330-336`

**Effort:** new-backend · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

## not applicable — 27

### Tile boundary IS the page boundary — no per-token scale slot
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:43-45` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:51-52`

**What it does.** KVarN requires `group == vLLM block_size` so that one vLLM block equals exactly one KVarN tile per head. Because scales are shared across the whole tile, the cache record for a (block, head) pair is one packed blob rather than block_size independently-scaled token records.

**Mechanism.** The `group` field (config.py:63) is documented at config.py:51-52 as 'KVarN tile size in tokens. Must equal vLLM block_size so that one vLLM block = one KVarN tile per head.' config.py:43-45: 'Cache layout (per (block, head)) is a single packed record ... There is no per-token slot because the scales are tile-shared; the block boundary IS the tile.' Everything downstream then divides `tile_bytes_aligned` by `group` to synthesise a fictitious per-token slot size for vLLM's page accounting (kvarn-0.27.1.patch:41, :231).

**Why they needed it.** vLLM's allocator thinks in per-token slots inside a block; KVarN's compression only works over a full tile. Pinning group to block_size is what lets a tile-scoped scheme be expressed in a token-scoped allocator without changing the allocator.

**Their numbers.** g128 default; the README notes the hybrid alignment makes the attention block 2048 tokens and 'vLLM splits it into 128-token kernel tiles, KVarN's invariant `tile == kernel block` holds' (kvarn/README.md:37-39)

**llama.cpp — not applicable.** The KV cache is one ggml tensor per layer, ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) — cells, not pages, and no per-token slot record to reconcile with a tile. The only 'block boundary' in llama.cpp is the ggml type's block size (32 for q4_0/q8_0), which runs along the embedding axis inside one token, and is constrained to divide n_embd_head_k. There is nothing here for a tile-equals-page invariant to fix.

**Equivalent here:** none — llama.cpp has no paged KV allocator

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:229-231` · `src/llama-kv-cache.cpp:1233-1246` · `src/llama-context.cpp:3613-3622`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### lcm(8, group) tile alignment replacing power-of-two slot padding (opt-in via KVARN_POW2_SLOT)
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:109-148` · `kvarn/README.md:34-36`

**What it does.** `tile_bytes_aligned` rounds the raw tile up so the derived per-token slot (`tile_bytes_aligned // group`) is an exact integer. The default rounds to a multiple of lcm(8, group) — 8 for the fp16 field loads, `group` for slot integrality. The upstream KVarN behaviour of rounding the per-token slot up to a power of two is retained but demoted to opt-in behind `KVARN_POW2_SLOT=1`, and only applies at head_dim >= 256.

**Mechanism.** config.py:132 gates the pow2 branch on `self.head_dim >= 256 and os.environ.get("KVARN_POW2_SLOT", "0") == "1"`; that branch computes `slot = ceil(tile_bytes/group)`, `slot_pow2 = 1 << (slot-1).bit_length()`, `aligned = slot_pow2 * group` (:133-135). The default branch is `unit = math.lcm(8, self.group); aligned = ceil(tile_bytes/unit)*unit` (:141-142). A hard assert at :145-147 enforces `aligned % group == 0` with a message naming both values. Padding is trailing only, so all eight offset properties are unchanged between the two modes and, per :129-131, 'every kernel reads `kv_cache.stride(0/1)` rather than assuming a size, so the two layouts are byte-compatible.'

**Why they needed it.** Two competing constraints. Pow2 exists because heterogeneous-head_dim models break vLLM's page-size unification: 'the raw slot has a fixed per-token-group scale term that doesn't scale with D, so slot(512)/slot(256) is not an integer and vLLM's KV-cache page-size unification (which scales block_size by that ratio) fails' (config.py:116-119). But paying it unconditionally wastes cache on homogeneous models, so the default became the relaxed rounding.

**Their numbers.** k4v2_g128 at D=256: default 26880 B = 210 B/token/head; pow2 padded it to 32768 B = 256 B/token/head, 'i.e. +22% KV-cache footprint for nothing on a homogeneous model' (config.py:126-129). README restates it as 840 B/token/layer instead of 1024, against fp8's 2048 (kvarn/README.md:35-36). Cited breaking case: Gemma-4 with 256-wide sliding-window layers and 512-wide global layers (config.py:116-117).

**llama.cpp — not applicable.** llama.cpp's alignment problem is a different one and already solved: n_kv is rounded to 256 so the graph stays constant across batches and so K->ne[1] % FATTN_KQ_STRIDE == 0 holds, which is a precondition for both gqa_opt and the vector kernel. There is no per-token slot whose padding could be inflated to a power of two, and no heterogeneous-head_dim page unification to break. Nothing to port and nothing to tune.

**Equivalent here:** n_kv padded to 256 (FATTN_KQ_STRIDE); n_ctx padded to 256; --fit rounds n_ctx down to 256

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:1233-1246` · `src/llama-context.cpp:288` · `common/fit.cpp:344`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none — the analogous waste is at most 255 cells

### fp16 tail pool: two side-buffer blocks per request per layer (attention sink + in-progress tile)
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:183-192` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:208-223`

**What it does.** Because a tile cannot be quantized until all `group` of its tokens exist, KVarN maintains a fixed-size fp16 side buffer alongside the paged cache. Per active request per layer it holds two fp16 blocks: the permanent attention-sink block and the in-progress tail. `pool_slots` computes the structural peak slot count for one step.

**Mechanism.** `_slot_bytes_per_layer(num_kv_heads)` = `group * num_kv_heads * head_dim * 4` (config.py:208-211) — the 4 is '2 bytes/elem × 2 tensors' for K and V. `pool_slots(max_num_seqs, max_num_batched_tokens)` (config.py:213-223) returns `max(2*max_num_seqs + prefill_blocks + 8, 8)` where `prefill_blocks = ceil(max_num_batched_tokens/group)`. `pool_bytes` (config.py:278-289) multiplies slots by slot bytes by layer count and is 'Reserved up front in the worker so the lazy pool allocation never pushes past the KV-memory limit.'

**Why they needed it.** Stated at config.py:185-189: 'a tile cannot be quantized until its `group` tokens all exist' and 'The pool must be pre-allocated at a fixed size (CUDA graphs), so its size bounds how many requests can run concurrently.' The 2x term is sink + tail; the prefill term covers 'the full blocks a chunked prefill can touch before flushing.'

**Their numbers.** 2*max_num_seqs + ceil(max_num_batched_tokens/group) + 8, floored at 8. Floor deliberately small: 'at large head_dim·heads·layers (e.g. Gemma-4 512·16·60 => ~251 MB/slot/layer) a big floor like 64 reserves tens of GB and leaves no room for the KV cache' (config.py:219-222).

**llama.cpp — not applicable.** The whole tail-pool exists because a KVarN tile spans `group` tokens and cannot be quantised until all of them exist. llama.cpp's quantisation axis is intra-token: the K row for one cell is n_embd_k_gqa elements quantised in blocks of 32 along that row, which is why the block size must divide n_embd_head_k. Every token is fully quantisable the moment it is computed, so there is no in-progress tile, no side buffer, no pool to size, and no concurrency cap derived from one. None of techniques 7-9 have a problem to solve here.

**Equivalent here:** none needed

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:229-231` · `src/llama-context.cpp:3613-3622`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none — and this is a structural advantage, not a gap

### NVML total-memory query to keep the parent process CUDA-free
**Where (theirs):** `kvarn/kvarn-0.27.1.patch:157-160`

**What it does.** The pool-budget computation in `check_and_update_config` gets total device memory through the platform's NVML-backed `get_device_total_memory()` rather than any torch.cuda call.

**Mechanism.** `total_gpu_bytes = cls.get_device_total_memory()` at kvarn-0.27.1.patch:160, with the comment 'Query total GPU memory via NVML (cls.get_device_total_memory) so we do NOT initialize a CUDA context in the parent here'.

**Why they needed it.** Verbatim, kvarn-0.27.1.patch:158-159: 'doing so would force `spawn` multiprocessing for tensor parallelism.' Touching CUDA in the config hook silently changes vLLM's multiprocessing start method for the whole engine.

**llama.cpp — not applicable.** llama-server is one process; there is no fork/spawn start-method to protect. --fit reads free and total memory through ggml_backend_dev_memory() and the CUDA context it initialises is the one it is going to use anyway.

**Equivalent here:** ggml_backend_dev_memory()

**Evidence (llama.cpp):** `common/fit.cpp:56-57` · `common/fit.cpp:111-116`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Sinkhorn iteration count reduced 16 -> 8 on measured convergence
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:64` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:432-435` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/sinkhorn.py:33`

**What it does.** The alternating column/row normalisation loop runs 8 iterations by default instead of the paper's 16, with an env override for testing convergence at larger scale.

**Mechanism.** `sinkhorn_iters: int = 8` at config.py:64 with the inline justification '# converges by ~4 iters; 8 lossless vs 16 (validated Qwen3-4B + Qwen3.6-27B AIME)'. `from_cache_dtype` reads `int(os.environ.get("KVARN_SINKHORN_ITERS", "8"))` (config.py:435). The pure-PyTorch reference keeps `_DEFAULT_ITERATIONS = 16` (sinkhorn.py:33), so reference and shipped default differ unless the config value is passed explicitly.

**Why they needed it.** Halving the iteration count halves the store-path cost of every tile flush. The claim is evidence-backed and names both the models and the benchmark: 'validated Qwen3-4B + Qwen3.6-27B AIME'. The env override exists because that evidence may not extend: config.py:433-434 says it is 'useful for testing convergence at large model scale (e.g. 48-layer 30B-A3B-Thinking-2507 may benefit from more).'

**Their numbers.** converges by ~4 iterations; 8 is lossless against 16 on Qwen3-4B and Qwen3.6-27B AIME; paper default 16; env KVARN_SINKHORN_ITERS.

**llama.cpp — not applicable.** There is no iterative normalisation in llama.cpp's KV path to count iterations of. The one preprocessing step it does have — the Hadamard rotation — is a single fixed matrix multiply, not a loop.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:313-336`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Best-so-far scale selection across Sinkhorn iterations (the loop is not assumed monotone)
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/sinkhorn.py:79-99` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/sinkhorn.py:5-7`

**What it does.** Rather than returning the scales from the final iteration, the loop tracks the lowest-imbalance state seen at any point (including iteration 0, i.e. the identity scales) and returns those. So the output is never worse-balanced than the unnormalised input.

**Mechanism.** `variance_normalize` (sinkhorn.py:55) seeds `imb_best = _imbalance(cur)` with `log_s_col`/`log_s_row` still zero (sinkhorn.py:75-82), so the identity is a candidate. Each iteration does a column pass then a row pass, recomputes `imb = _imbalance(cur)` (:92), and on `imb <= imb_best` clones `log_s_col.exp()` / `log_s_row.exp()` into `sc_best`/`sr_best` (:93-96). Final output is `balanced = m / sc_best / sr_best` (:98) — recomputed from the winning scales, not carried from the loop variable. The module docstring calls this 'tracking the lowest-imbalance state seen across all iterations (the best-so-far selection)' (sinkhorn.py:6-7).

**Why they needed it.** Alternating std-normalisation with clipping is not guaranteed to decrease imbalance monotonically — the clip bounds at sinkhorn.py:34-37 can push a step in the wrong direction. Best-so-far makes extra iterations free of downside risk, which is also what makes the 16->8 reduction safe to reason about.

**Their numbers.** `<=` rather than `<` on the comparison (sinkhorn.py:93), so ties prefer the later (more-iterated) state.

**llama.cpp — not applicable.** Depends entirely on there being an iterative scale search. llama.cpp's KV quantisation is one-shot round-to-nearest into a ggml block; there is no candidate state to keep the best of. (ggml's k-quant weight quantisers do search scales, but that path is offline in llama-quantize and never touches the KV cache.)

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:313-336`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Imbalance metric: sum of column-std spread and row-std spread, minimum 2
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/sinkhorn.py:40-52`

**What it does.** A single scalar per tile measuring how far the tile is from having uniform per-row and per-column standard deviation. It is the objective the best-so-far selection minimises.

**Mechanism.** `_imbalance(tile)` (sinkhorn.py:40) computes `sc = tile.std(dim=-2)` (per-column std) and `sr = tile.std(dim=-1)` (per-row std), then returns `sc.amax(-1)/sc.amin(-1).clamp_min(1e-8) + sr.amax(-1)/sr.amin(-1).clamp_min(1e-8)` (:49-52). Using negative dims means the same function serves both the [R,C] single-tile and [N,R,C] batched cases, returning a scalar or an [N] vector respectively.

**Why they needed it.** Documented at sinkhorn.py:44-46: 'Lower is better; a perfectly balanced tile has `imbalance == 2` (each std max equals its std min).' The known floor of exactly 2 is what makes the metric interpretable — you can read how far from balanced a tile got without a reference. The `clamp_min(1e-8)` guards a degenerate all-constant row or column from producing inf.

**Their numbers.** perfectly balanced tile scores exactly 2.0; denominator clamp 1e-8.

**llama.cpp — not applicable.** An objective function for a normalisation loop that does not exist here. Nothing in llama.cpp's KV path computes per-row or per-column statistics over a group of tokens.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:313-336`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Log-domain scale accumulation with std clipping and asymmetric log-scale bounds
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/sinkhorn.py:33-37` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/sinkhorn.py:84-90`

**What it does.** Scales are accumulated additively in log space rather than multiplicatively, and both the per-step std and the accumulated log-scale are clipped to hard bounds. The log-scale bounds are deliberately asymmetric — barely below zero, wide above.

**Mechanism.** Four module constants: `_CLIP_STD_MIN = 1e-3`, `_CLIP_STD_MAX = 1e3`, `_LOG_S_MIN = -0.3`, `_LOG_S_MAX = 10.0` (sinkhorn.py:34-37). Each half-pass does `col_std = cur.std(dim=0, keepdim=True).clamp(_CLIP_STD_MIN, _CLIP_STD_MAX)` then `log_s_col = (log_s_col + col_std.log()).clip(_LOG_S_MIN, _LOG_S_MAX)` and re-derives `cur = m / log_s_col.exp() / log_s_row.exp()` from the *original* m each time rather than dividing cur in place (sinkhorn.py:84-90). Re-deriving from m stops rounding error compounding across iterations.

**Why they needed it.** Log-domain accumulation turns a chain of multiplications into a chain of additions, which is numerically stable over 8-16 iterations and is what the module title calls out ('Log-domain iterative variance-normalization'). The asymmetric bounds encode that scales are expected to shrink outliers (large positive log-scale, up to e^10) far more than they amplify quiet channels (e^-0.3 ~ 0.74 floor) — an amplifying scale would push values toward the quantizer's clipping range.

**Their numbers.** std clipped to [1e-3, 1e3]; log-scale clipped to [-0.3, 10.0], i.e. multiplicative scale in [0.741, 22026].

**llama.cpp — not applicable.** No accumulated scale exists to hold in log space. A ggml block scale is computed once from the block's max and stored as one fp16.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:313-336`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Branchless masked best-so-far for the batched tile variant
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/sinkhorn.py:102-139`

**What it does.** The batched version balances N tiles at once, and each tile independently keeps its own best iteration. The per-tile selection is done with an arithmetic blend rather than indexing, so it stays a single fused GPU op.

**Mechanism.** `variance_normalize_batched(tiles [N,R,C], iterations)` (sinkhorn.py:102) keeps `log_s_col [N,1,C]` and `log_s_row [N,R,1]`. After each iteration `better = imb <= imb_best` is an [N] bool (sinkhorn.py:131-132); if any element is set it builds `mask = better.view(N,1,1).to(log_s_col.dtype)` and blends `sc_best = mask*log_s_col.exp() + (1-mask)*sc_best`, same for `sr_best`, and `imb_best = torch.where(better, imb, imb_best)` (sinkhorn.py:134-137). Axis indices shift by one versus the single-tile version (std over dim 1 and dim 2 instead of 0 and 1) to skip the batch axis.

**Why they needed it.** A prefill flush quantizes many (block, head) tiles simultaneously; per-tile Python branching would serialise it. The mask blend keeps the whole selection in tensor ops. Both variants are documented to 'return the same triple `(balanced, s_col, s_row)`' (sinkhorn.py:12-13), so the batched path is a drop-in for the reference.

**llama.cpp — not applicable.** A GPU-efficiency detail of a loop that has no counterpart. There is no batched tile quantisation step in llama.cpp — tokens are quantised into their cell rows as they are produced.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:229-231`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Patch hunk B — cache dtype -> torch dtype map (all presets are uint8)
**Where (theirs):** `kvarn/kvarn-0.27.1.patch:244-258`

**What it does.** Registers all four presets in `STR_DTYPE_TO_TORCH_DTYPE` as `torch.uint8`, so vLLM allocates the KV cache as a raw byte buffer that KVarN interprets itself.

**Mechanism.** Four entries appended to the dict in utils/torch_utils.py alongside the turboquant and nvfp4 entries (kvarn-0.27.1.patch:252-256).

**Why they needed it.** The tile is a packed heterogeneous record (sub-byte quantized data plus fp16 scale vectors at fixed offsets). Declaring it uint8 is what lets the offset properties in config.py address it as bytes; any typed dtype would impose an element size that does not match the record.

**llama.cpp — not applicable.** This hunk exists because vLLM must be told to treat the cache as an opaque byte buffer. ggml has no such impedance: the tensor is created with the quant type and ggml_row_size/ggml_nbytes give the bytes. There is no second type system to lie to.

**Equivalent here:** none needed — a ggml_type carries its own element/block layout

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:229-231`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Patch hunk D — KVQuantMode.KVARN as a distinct mode (not reusing TURBOQUANT or NONE)
**Where (theirs):** `kvarn/kvarn-0.27.1.patch:273-311` · `kvarn/README.md:23-27`

**What it does.** Adds `KVARN = 7` to the KVQuantMode IntEnum, an `is_kvarn` predicate mirroring `is_turboquant`, and a prefix branch in `get_kv_quant_mode` so any `kvarn_`-prefixed dtype maps to it. Ordering matters: the kvarn check sits before the `fp8` prefix check.

**Mechanism.** Enum member at kvarn-0.27.1.patch:285 with the comment 'KVarN Hadamard+Sinkhorn tile quant, packed K+V per (block, head) tile'. `is_kvarn` property at :294-297. Dispatch at :306-308: `if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("kvarn_"): return KVQuantMode.KVARN`, placed after the turboquant branch and before the fp8 branch (:309).

**Why they needed it.** Two reasons, both stated. From the patch (:281-284): 'KVarN gets its own mode so the model runner does not pass cache_dtype_str="auto" to KVarNAttentionBackend.get_kv_cache_shape (it does so for KVQuantMode.NONE) and so mixed-precision bookkeeping does not conflate KVarN with TurboQuant.' From the README (:24-27), the failure is fatal: '0.27.1 calls `get_kv_cache_shape(..., cache_dtype_str="auto")` for specs whose `kv_quant_mode` is `NONE`; KVarN's shape depends on the preset ... Without that the engine dies at KV-cache init.'

**Their numbers.** KVARN = 7 (TURBOQUANT = 6, NVFP4 = 5)

**llama.cpp — not applicable.** The bug this hunk prevents (a caller passing cache_dtype_str='auto' into a shape function) requires a layer of indirection between the dtype and the layout that llama.cpp does not have. cparams.type_k IS the mode and IS the layout.

**Equivalent here:** none — no quant-mode enum in the KV path

**Evidence (llama.cpp):** `common/common.cpp:1727-1728` · `src/llama-kv-cache.cpp:229-231`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Patch hunk E — KV-cache spec branch reusing TQFullAttentionSpec with a synthesised slot size
**Where (theirs):** `kvarn/kvarn-0.27.1.patch:17-53`

**What it does.** In `Attention.get_kv_cache_spec`, a `kvarn_`-prefixed dtype returns a `TQFullAttentionSpec` (TurboQuant's spec class, reused verbatim) whose `tq_slot_size` is KVarN's `tile_bytes_aligned // group`. That is what makes vLLM allocate blocks at the compressed size instead of the fp16 size.

**Mechanism.** `elif self.kv_cache_dtype.startswith("kvarn_")` at kvarn-0.27.1.patch:27, mirroring the TQ branch directly above it. It builds the config via `KVarNConfig.from_cache_dtype(self.kv_cache_dtype, self.head_size)` (:33-35), computes `slot_bytes = kvarn_cfg.tile_bytes_aligned // kvarn_cfg.group` (:41), and returns TQFullAttentionSpec with `head_size_v=self.head_size`, `kv_quant_mode=quant_mode`, `tq_slot_size=slot_bytes` (:42-50). The patch flags the one difference from the 0.23 original at :25-26: this version 'passes kv_quant_mode=quant_mode, which the 0.23 KVarN hunk did not have'.

**Why they needed it.** Justified at :37-40: 'We reuse the TQ spec class because the on-disk layout primitive (a uint8 block of per-(token,head) "slots") is the same; only the slot semantics differ.' Reusing a foreign quantizer's spec class avoids duplicating vLLM's page arithmetic; the tile-vs-slot impedance is resolved by the single division at :41.

**llama.cpp — not applicable.** There is no KV-cache spec object and no per-token slot to synthesise. The allocator asks ggml how big the tensor is, and the startup log prints the resulting MiB per K and V type — that is the entire accounting path.

**Equivalent here:** none — sizes come from ggml_row_size(type, n_embd_k_gqa) * kv_size

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:229-231` · `src/llama-kv-cache.cpp:300-302`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Patch hunk F — hybrid-model page alignment branch in Platform._align_heterogeneous_kv_block_size
**Where (theirs):** `kvarn/kvarn-0.27.1.patch:205-243`

**What it does.** When computing the single-token attention page size used to align heterogeneous (attention + mamba) KV groups, a kvarn dtype takes a branch that builds a block_size=1 `TQFullAttentionSpec` with the real KVarN slot bytes and reads its `page_size_bytes`, instead of falling through to the FullAttentionSpec formula.

**Mechanism.** `elif cache_config.cache_dtype.startswith("kvarn_")` at kvarn-0.27.1.patch:215, sitting after the turboquant branch (which additionally does an `lcm(tq_page, skip_page)`). The KVarN branch computes the same `slot_bytes = tile_bytes_aligned // group` (:231) and instantiates `TQFullAttentionSpec(block_size=1, num_kv_heads=..., head_size=..., head_size_v=..., dtype=kv_cache_dtype, kv_quant_mode=kv_quant_mode, tq_slot_size=slot_bytes).page_size_bytes` (:232-240). The patch header notes it is 'dense KVarN only, no kvarn_mla_* and no skip-layer lcm term' (:213-214).

**Why they needed it.** Verbatim at :217-222: 'The standard FullAttentionSpec formula over-sizes the page (uses FP16 head_size*dtype) and trips unify_kv_cache_spec_page_size in hybrid models, because the mamba padding is then sized to the FP16 page while the real KVarN page is ~4x smaller and no longer divides it.' This is the hunk that makes KVarN usable on Qwen3.x hybrids at all.

**Their numbers.** real KVarN page ~4x smaller than the FP16 page the default formula computes

**llama.cpp — not applicable.** llama.cpp's hybrid memory is literally two memory modules side by side: mem_attn gets type_k/type_v, mem_recr gets its own F32 tensors sized by n_seq_max and n_rs_seq. There is no shared page size to unify and therefore no over-sizing to correct. The relevant asymmetry here is the opposite one and already known: raising -c costs nothing on the recurrent half and everything on the attention half.

**Equivalent here:** none — llama_memory_hybrid sizes the attention and recurrent halves independently

**Evidence (llama.cpp):** `src/llama-model.cpp:2281-2303` · `src/llama-model.cpp:2314-2315` · `src/llama-memory-recurrent.cpp:99-101` · `src/llama-memory-hybrid.cpp:36-56`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### KVARN_QUANT_SLIDING escape hatch (quantize SWA layers when window > group)
**Where (theirs):** `kvarn/kvarn-0.27.1.patch:138-144`

**What it does.** An experimental env flag that removes `sliding_window` from the skip list entirely, so sliding-window layers get compressed too.

**Mechanism.** `_quant_sliding = os.environ.get("KVARN_QUANT_SLIDING") == "1"` (kvarn-0.27.1.patch:138); when set, a `while "sliding_window" in skip_layers: skip_layers.remove(...)` loop strips every occurrence and logs at info level (:140-144). It is the `if` branch of the same if/elif whose `elif` does the append, so setting it and the auto-append are mutually exclusive by construction.

**Why they needed it.** Labelled 'Experimental: quantize sliding-window layers too (window>group)' (:141). The correctness condition is implicit in that parenthetical — a window wider than the tile means a full tile is always in-window, so the missing sliding-window mask does not bite.

**llama.cpp — not applicable.** Requires both a per-layer KV dtype mechanism (absent, see technique 14) and a model with sliding-window layers (this one has none).

**Equivalent here:** none

**Evidence (llama.cpp):** `src/models/qwen35.cpp:21-26` · `src/llama-model.cpp:2305`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### V2-runner fix 2 — sliding-window specs pad to the mamba page instead of a full uniform page
**Where (theirs):** `kvarn/kvarn-v2-runner.patch:132-145` · `kvarn/kvarn-v2-runner.patch:7-8`

**What it does.** When a hybrid model has no skip-layer shared page (`skip_page_size_padded is None`) and the cache dtype is kvarn, the sliding-window spec's page padding is set to the mamba page size rather than being left unset, which would otherwise round each tiny SW block up to the full uniform page.

**Mechanism.** In attention.py, before building the `SlidingWindowSpec`: `if shared_page is None and str(vllm_config.cache_config.cache_dtype).startswith("kvarn"): shared_page = vllm_config.cache_config.mamba_page_size_padded` (kvarn-v2-runner.patch:139-142).

**Why they needed it.** kvarn-v2-runner.patch:7-8: 'SW specs without a shared page pad to the mamba page under KVarN -- otherwise each 16-token SW block burns a full 1.83 MB uniform page (84% of the pool).' Restated in the hunk comment as '26x overhead' (:137-138).

**Their numbers.** each 16-token SW block would occupy a 1.83 MB uniform page = 84% of the pool; 26x overhead.

**llama.cpp — not applicable.** No page padding exists to mis-size: the recurrent half is a pair of F32 tensors sized by n_embd_r/n_embd_s and n_seq_max, and the attention half is sized by ggml_row_size. There is also no sliding-window group on this model.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-memory-hybrid.cpp:36-56` · `src/llama-memory-recurrent.cpp:99-101`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### V2-runner fix 3 — _largest_kernel_block_within scales MultipleOf backends and prefers 128-divisors
**Where (theirs):** `kvarn/kvarn-v2-runner.patch:95-131` · `kvarn/kvarn-v2-runner.patch:9-12`

**What it does.** Backends that declare `MultipleOf(16)` (FlashAttention-2) previously contributed only their base block size as a candidate. The patch scales them up to whatever the page budget allows, and among the possible sizes searches downward for one that is simultaneously a divisor of the primary block and a multiple of 128.

**Mechanism.** In attention.py, `mults = [s.base for s in sizes if isinstance(s, MultipleOf)]` (kvarn-v2-runner.patch:104); if there is a page budget, `budget_tokens = page_budget // per_token_bytes` and `scaled = budget_tokens // base * base` (:107-108). Then a two-pass search: `for step in (128, base): for d in range(budget_tokens, step-1, -1): if fallback % d == 0 and d % step == 0: found = d; break` — 128 first, base second (:113-123). The scaled candidate is appended rather than replacing the list (:124-125), and the old `candidates = [s.base ...]` fallback becomes `candidates = mults` (:126-128).

**Why they needed it.** The invariant is spelled out at kvarn-v2-runner.patch:11-12: '128 is the only block size that is simultaneously a divisor of 2176 (=17x128), a FlashAttention-2 multiple of 16, and a KVarN tile multiple.' The hunk comment adds the scheduler consequence: preferring a divisor of the primary block matters 'so the scheduler LCM and the prefix hash work out — otherwise the SW block breaks block-size uniformity' (:109-112).

**Their numbers.** primary block 2176 = 17 x 128; FA2 declares MultipleOf(16); KVarN tile 128.

**llama.cpp — not applicable.** llama.cpp has one KV block granularity, 256, applied globally, and the kernel precondition K->ne[1] % FATTN_KQ_STRIDE == 0 is satisfied by construction. There are no per-group block sizes to reconcile and no LCM to search.

**Equivalent here:** n_kv padded to FATTN_KQ_STRIDE = 256

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn-common.cuh:9` · `src/llama-kv-cache.cpp:1233-1246` · `ggml/src/ggml-cuda/fattn.cu:458`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### V2-runner fix 4a — sliding-window groups excluded from the scheduler block-size LCM
**Where (theirs):** `kvarn/kvarn-v2-runner.patch:146-162`

**What it does.** When computing `scheduler_block_size = math.lcm(*group_block_sizes)`, sliding-window groups are filtered out of the input list, so the drafter's block size cannot inflate the LCM for every other group.

**Mechanism.** In kv_cache_utils.py, imports `SlidingWindowSpec as _SWS` locally and builds `_non_sw = [bs for g, bs in zip(groups, group_block_sizes) if not isinstance(g.kv_cache_spec, _SWS)]`, replacing group_block_sizes only `if _non_sw` (kvarn-v2-runner.patch:155-161) — the guard means an all-SW configuration falls back to the old behaviour rather than LCM-ing an empty list.

**Why they needed it.** kvarn-v2-runner.patch:151-154: 'sliding-window groups (the DFlash drafter) take no part in prefix matching and must not skew the LCM/GCD; their block is a divisor of the primary block by construction.'

**llama.cpp — not applicable.** There are no KV cache groups with independent block sizes, and no scheduler-level LCM over them.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:1233-1246`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### V2-runner fix 4b — explicit --prefix-match-unit outranks the mamba-divergence fallback
**Where (theirs):** `kvarn/kvarn-v2-runner.patch:165-175` · `kvarn/kvarn-v2-runner.patch:13-15`

**What it does.** The heuristic that backs the prefix-hash unit off to the scheduler block size when a mamba group's block differs from cache_config.block_size is now skipped entirely if the operator passed an explicit `--prefix-match-unit`.

**Mechanism.** One-token change in kv_cache_utils.py: `if any(...)` becomes `if cache_config.prefix_match_unit is None and any(isinstance(g.kv_cache_spec, MambaSpec) and g.kv_cache_spec.block_size != cache_config.block_size for g in groups)` (kvarn-v2-runner.patch:172-175).

**Why they needed it.** kvarn-v2-runner.patch:14-15 and the inline comment at :170-171: the fallback 'silently ignored' an explicit flag. The operator sets the flag precisely because they know the geometry; the heuristic overrode them without saying so.

**llama.cpp — not applicable.** llama.cpp matches cached prompts by longest common token prefix, not by hashed blocks, so there is no prefix-match unit to be overridden. The chunk-level analogue, --cache-reuse, is unreachable on this model anyway: QWEN35 is IMROPE so llama_memory_can_shift() is false and the server zeroes n_cache_reuse at startup. The general lesson (an explicit operator flag must outrank a heuristic) does apply to llama.cpp in other places, but not to anything in this area.

**Equivalent here:** prompt reuse is exact-token-prefix based (get_common_prefix), no hash unit

**Evidence (llama.cpp):** `tools/server/server-context.cpp:3125-3126` · `tools/server/server-context.cpp:1182-1186` · `src/llama-kv-cache.cpp:1171-1179`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### V2-runner fix 5a — hash-block divisibility assert relaxed in both directions
**Where (theirs):** `kvarn/kvarn-v2-runner.patch:176-192`

**What it does.** The coordinator's assertion that every group's block_size is divisible by the hash block size is widened to accept either direction of divisibility.

**Mechanism.** In kv_cache_coordinator.py the predicate becomes `block_size % hash_block_size == 0 or hash_block_size % block_size == 0` (kvarn-v2-runner.patch:186-188).

**Why they needed it.** kvarn-v2-runner.patch:182-184: 'sliding-window groups (the DFlash drafter) take no part in prefix matching; for them it suffices that the hash unit is an integer multiple of their block (e.g. 2176 = 17 x 128).'

**Their numbers.** example given: hash unit 2176, SW block 128, ratio 17.

**llama.cpp — not applicable.** No block-hash coordinator and no such assertion exist in llama.cpp's prompt cache.

**Equivalent here:** none

**Evidence (llama.cpp):** `tools/server/server-context.cpp:3125-3126`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### V2-runner fix 5b — SlidingWindowManager returns a clean miss instead of asserting (read side), and skips caching (write side)
**Where (theirs):** `kvarn/kvarn-v2-runner.patch:193-229` · `kvarn/kvarn-v2-runner.patch:16-26`

**What it does.** Two halves of the same fix. On the read path, when the SW block and the hash unit do not line up in either direction, the manager returns an empty hit tuple with 0 tokens rather than tripping an assert. On the write path, `cache_blocks` returns early when the SW block does not divide the hash unit, instead of asserting inside `resolve_block_hashes`.

**Mechanism.** Read side (kvarn-v2-runner.patch:196-209): the `assert alignment_tokens % kv_cache_spec.block_size == 0` is replaced by `if (kv_cache_spec.block_size % block_pool.hash_block_size != 0 or alignment_tokens % kv_cache_spec.block_size != 0): return tuple([] for _ in kv_cache_group_ids), 0`. Write side (:217-229): a new `cache_blocks(self, request, num_tokens, retention_interval=None)` override that does `if self.block_size % self.block_pool.hash_block_size != 0: return` before delegating to `super().cache_blocks(...)`.

**Why they needed it.** The write half is the one that mattered and is called out as non-optional at kvarn-v2-runner.patch:19-21: 'Without the write-side half, cache_full_blocks -> resolve_block_hashes asserts as soon as the drafter's SW group is in the picture, which is what made --prefix-match-unit 128 look like a multi-GPU workaround; it is neither optional nor TP-specific.' The both-directions read guard is issue #18: 'The SW block is usually smaller than the hash unit (16 against 2176), but the page arithmetic promotes it larger at some draft lengths -- 2048 against a 128 unit at DFLASH_TOKENS=3 -- which slipped past a smaller-only check and then tripped the upstream assert on the first request, killing the engine' (:22-26). The justification for degrading rather than fixing: 'prefix reuse is meaningless for a rolling window anyway' (:18-19, :205-206).

**Their numbers.** typical case SW block 16 vs hash unit 2176; inverted case SW block 2048 vs hash unit 128 at DFLASH_TOKENS=3; issue #18.

**llama.cpp — not applicable.** There is no cache manager with this shape to fix. The transferable content is a design opinion — degrade to a miss rather than abort — and for a benchmark instrument the abort is usually the right choice, so I would not import it.

**Equivalent here:** none, though llama.cpp has the opposite-direction hazard in several places

**Evidence (llama.cpp):** `tools/server/server-context.cpp:496` · `common/speculative.cpp:2111` · `common/speculative.cpp:2124`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** none as a port. As a principle it cuts against this project's north star: llama.cpp already prefers a hard stop in a few hot paths (the verify-batch GGML_ASSERT, the ngram-cache GGML_ABORT on a malformed file), which is the safer failure for a measurement rig.

### V2-runner fix 6 — mamba state seed uses the mamba spec's block size, not cache_config.block_size
**Where (theirs):** `kvarn/kvarn-v2-runner.patch:234-253` · `kvarn/kvarn-v2-runner.patch:27-32`

**What it does.** In mamba_hybrid's `add_request` align path, the running mamba state column index is computed by dividing by the mamba group's own block size instead of the engine-wide `cache_config.block_size`, which diverge as soon as `--block-size` is set explicitly.

**Mechanism.** `mamba_bs = (self._mamba_spec.block_size if self._mamba_spec is not None else self.cache_config.block_size)` then `self._mamba_state_idx_gpu[req_index].fill_((new_req_data.num_computed_tokens - 1) // mamba_bs)` (kvarn-v2-runner.patch:247-252). The safety argument for the lazily-populated `_mamba_spec` is given inline at :244-246: 'populated by the first batch's preprocess_state; resumes can only happen after that, and fresh requests (num_computed==0) seed -1 either way.'

**Why they needed it.** kvarn-v2-runner.patch:28-32: 'With --block-size 128 (KVarN tile) a resume from a 100k cached prefix indexed column 781 of a ~61-column block table -> illegal memory access in the align pre-copy.' Flagged as not KVarN's fault: '(Upstream vLLM bug: any hybrid with an explicit --block-size != mamba block size, prefix caching, and mamba_cache_mode=align hits this on the second request.)'

**Their numbers.** block sizes diverge 128 vs 2176; index 781 into a ~61-column block table; triggers on the second request.

**llama.cpp — not applicable.** The class of bug (two block sizes diverging and an index computed against the wrong one) cannot occur: n_rows = mem_size * (1 + n_rs_seq) with mem_size = max(1, n_seq_max), and rollback is bounded by n_rs_seq rather than addressed through a block table.

**Equivalent here:** none — the recurrent state is indexed per sequence, not by a block table

**Evidence (llama.cpp):** `src/llama-memory-recurrent.cpp:99-101` · `src/llama-memory-recurrent.cpp:180-190`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### install.sh marker-counting verification (detects a silently partial patch)
**Where (theirs):** `kvarn/install.sh:30-60` · `kvarn/kvarn-v2-runner.patch:47-52`

**What it does.** Because `patch --forward` exits non-zero both for a legitimate rerun and for a hunk that no longer applies, install.sh checks the *result* rather than the exit code: it parses the patch for `port(kvarn-v2)` markers per target file, counts the markers actually present in the installed file, and exits 1 with a per-file `found/expected` report if any file is short.

**Mechanism.** install.sh:12-16 does `cp -r files/vllm/. $SP/` then applies both patches with `patch -p1 -N -r /dev/null -d "$SP" < ... || true`. The inline Python (install.sh:18-61) walks the v2 patch line by line: `+++ b/` lines set the current target and seed `want[current] = 0`; lines starting with `+` containing `port(kvarn-v2)` increment it (install.sh:38-43). Then for each target it does `target.read_text().count("port(kvarn-v2)")` and collects `f"{rel}: {found}/{expected} markers"` for any shortfall (:44-49), printing a remediation message and `sys.exit(1)` (:50-59). Also purges `__pycache__` under any kvarn path (install.sh:17) and self-tests the config by printing `c.tile_bytes` and the per-token slot for `kvarn_k4v2_g128` at head_dim 256 (install.sh:26-28).

**Why they needed it.** install.sh:30-35 states the reasoning: 'Check the result, not the exit code. `patch --forward` cannot tell "already applied" (a legitimate rerun) from "does not apply" (a vLLM tree this patch was not cut against) -- both exit non-zero -- which is why the calls above end in `|| true`. That makes a stale hunk fail SILENTLY: you get a partial port and no warning.' The concrete incident that motivated it is in the patch header (:47-52): the zeros-not-empty hunk 'silently dropped on newer trees' because 'lookup-v2 gave _selector_tokens an explicit (max_num_reqs, draft_block) shape, so the old torch.empty_like pre-image is gone.' The failure message names the stakes: 'a partially applied port runs, but silently drops correctness fixes' (install.sh:55-56).

**Their numbers.** prints '(N markers across M files)' on success; the self-test prints 'tile bytes ... -> per token per head ... B (fp8: 256 B)' for k4v2_g128 @ D=256.

**llama.cpp — not applicable.** This is build tooling for a vendored Python port, with no llama.cpp counterpart. I am recording it as idea-only rather than not-applicable because we do run a non-master llama.cpp tree (build 10499 / 1deefcca3 = PR #27342), so a 'verify the tree is the tree you think it is' check has a real target — but nothing in the capability map is the seam for it.

**Equivalent here:** none in llama.cpp; the repo's own gate is scripts/ + bench/tests

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** none for llama.cpp. As a practice it matches this repo's stated north star exactly — check the result, not the exit code, because a tool that half-succeeds returns a plausible number. Relevant if we ever carry local patches on top of the dflash2 tree.

### Package surface deliberately limited to config + presets (MLA prototypes excluded from __init__)
**Where (theirs):** `kvarn/files/vllm/model_executor/layers/quantization/kvarn/__init__.py:11-19`

**What it does.** The package re-exports exactly two names, `KVarNConfig` and `KVARN_PRESETS`, and carries an explicit instruction that the MLA prototype modules must never be added.

**Mechanism.** `__all__ = ["KVarNConfig", "KVARN_PRESETS"]` at __init__.py:18, with the comment at :11-13: 'also re-export KVARN_PRESETS (the plan refers to it from this package). The MLA prototypes (mla_probe.py / mla_quant.py) are deliberately NOT ported and must not be imported here.'

**Why they needed it.** An `__init__` import is what turns an unported prototype into an import-time crash for every user of the package. Writing the prohibition into the file is cheaper than rediscovering it on the next vLLM bump.

**Their numbers.** 2 exported names

**llama.cpp — not applicable.** A Python packaging discipline. llama.cpp has no import-time surface that could turn an unported prototype into a startup crash.

**Equivalent here:** none

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Port-marker comment convention (`# port(0.27.1)` / `port(kvarn-v2)`) as the diff-audit mechanism
**Where (theirs):** `kvarn/README.md:12-14` · `kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py:123` · `kvarn/kvarn-v2-runner.patch:87`

**What it does.** Every line that differs from upstream KVarN carries a marker comment naming which port introduced it. Upstream KVarN's own headers are left intact. The markers double as machine-checkable evidence that a patch landed.

**Mechanism.** Two namespaces: `# port(0.27.1)` for the vLLM-version port (config.py:123, :137, :143; kvarn-0.27.1.patch:9, :25, :62, :78, :119, :213, :252, :268, :281, :293, :306) and `# port(kvarn-v2)` for the V2-runner port (kvarn-v2-runner.patch:87, :101, :110, :136, :151, :170, :182, :199, :219, :241, :262, :275, :285). README.md:13-14 states the rule: 'copied from KVarN and adapted to the 0.27.1 backend API (every change is marked `# port(0.27.1)`; upstream KVarN headers kept).' install.sh:41-42 then greps for the second namespace as its verification signal.

**Why they needed it.** The files are vendored copies of a third-party project living inside site-packages; without markers there is no diff to take against upstream and no way for the next vLLM bump to find the local deltas. Making the marker the verification key (install.sh) means the convention cannot silently rot — dropping a marker breaks the installer.

**llama.cpp — not applicable.** Not a capability of llama.cpp at all — a convention for vendored source. Recorded as idea-only because we do carry a non-master tree and would carry local deltas if technique 39 is acted on.

**Equivalent here:** none

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Low but non-zero: we build from a PR branch rather than master, so if we ever add a local hunk (e.g. the technique-39 NaN guard) a marker plus a grep check is how a later session finds it. No value to llama.cpp itself.

### Builder ownership registry so an MTP draft layer is not flushed twice
**Where (theirs):** `kvarn/README.md:28-30`

**What it does.** The impl-to-builder wiring uses vLLM's `get_layers_from_vllm_config` plus a small owner registry, replacing upstream KVarN's `attention.py` `impl.layer_name` patch hunk, so a layer shared with the MTP draft path is flushed by exactly one builder.

**Mechanism.** Stated at README.md:28-30: 'The impl->builder wiring uses `get_layers_from_vllm_config` instead of KVarN's `attention.py` `impl.layer_name` hunk, and a small owner registry so the MTP draft layer isn't flushed by two builders.' The registry itself lives in kvarn_attn.py, outside this slice.

**Why they needed it.** A double flush would quantize an in-progress tile twice or flush a tile that the other builder still considers open — a correctness bug that only appears with speculative decoding enabled. Using the public lookup instead of an upstream hunk again shrinks the patch surface.

**Their numbers.** eliminates 1 upstream hunk

**llama.cpp — not applicable.** No deferred quantisation means no flush and no ownership question. The nearest llama.cpp analogue — an MTP draft sharing the target model's context — is handled by constructing a second llama_context with ctx_type=LLAMA_CONTEXT_TYPE_MTP, and its memory sharing is decided once at construction rather than per step.

**Equivalent here:** none — there is no tile flush to own

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:229-231` · `common/speculative.cpp:2483-2494`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** none

### Explicit non-ported surface (MLA path, TQSlidingWindowSpec, Gemma-4 config hunk)
**Where (theirs):** `kvarn/README.md:43-44` · `kvarn/README.md:6-8`

**What it does.** The port scopes itself to the dense (non-MLA) path for one model on one card and says so, listing exactly what was dropped.

**Mechanism.** README.md:43-44: 'Not ported: the MLA path, `TQSlidingWindowSpec` (no sliding-window layers here), the Gemma-4 config hunk.' Scope stated at :6-8: 'ported onto the vLLM 0.27.1 this repo runs, dense (non-MLA) path only, and tuned for the Qwen3.8-27B / RTX 3090 setup here.' The drops are load-bearing elsewhere: no TQSlidingWindowSpec is why the platform hook pushes SWA layers into skip_layers (kvarn-0.27.1.patch:112-118) and why the drafter is forced to bf16 (kvarn-v2-runner.patch:87-91); no Gemma-4 hunk is why pow2 slot padding is opt-in (config.py:132).

**Why they needed it.** Every non-ported piece is a live constraint on the ported code, and three separate defensive branches in this slice exist only because of them. Stating the boundary once makes those branches legible instead of mysterious.

**Their numbers.** 3 named omissions

**llama.cpp — not applicable.** A documentation discipline, not a capability. Worth noting only because it is the same discipline this repo already enforces: state what was not done so that later defensive branches are legible rather than mysterious.

**Equivalent here:** none

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** none for llama.cpp; the practice is already this repo's standard (docs/tested register, CORRECTIONS.md)

### Headline measured results on the 3090, including the regressions
**Where (theirs):** `kvarn/README.md:46-49` · `kvarn/kvarn-v2-runner.patch:66-74`

**What it does.** Both documents report measured outcomes with the losses stated as prominently as the wins.

**Mechanism.** README.md:46-49 (source: docs/long-context.md, outside this slice) and kvarn-v2-runner.patch:66-74 (measured 'on an RTX 3090 (WSL2, 250 W power limit), fast variant, through single-user/start_qwen.sh itself, bench/labd_bench.py --ctx 20000').

**Why they needed it.** The regressions are attributed to specific mechanisms rather than left as noise: short-request throughput is lower because '2048-token blocks make each request cost as much as fp8's 800-token block, and prefill flushes cost time' (README.md:48-49) — i.e. the hybrid page alignment (kvarn-0.27.1.patch:205-243) and the tail-pool flush (config.py:185-187) are the named causes.

**Their numbers.** README: 262k context fits (420k-token pool at 4 slots vs ~200k with fp8); needle-in-a-haystack correct at 4k-240k; perplexity +0.16%; decode ~20% slower than fp8 at 100k context; MTP works; short-request throughput lower. V2-runner bench (RTX 3090, WSL2, 250 W, --ctx 20000): copy 130 / code 89 / edit 65 / quote 44 / summary 38 / qa 36 tok/s; copy at 7.8 tokens per verify step; 53 tok/s over all six tasks; 245760 max-model-len with 268,169 tokens of KV; 100k-deep needle correct in both turns; a 100k prefix-cache hit takes the follow-up turn from 169 s of prefill to 4.7 s; with captured verify instead, copy drops to 1.97 tok/step and 39 tok/s.

**llama.cpp — not applicable.** A reporting practice. The one methodological point worth carrying is that their decode regression is attributed to named mechanisms (2048-token blocks, prefill flush cost) rather than left as noise — which is the same standard as 'a measurement names the file its number came from'.

**Equivalent here:** none

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** none as a capability. The numbers themselves do not transfer: different card (24 GB 3090 vs 12 GB 4070 SUPER), different serving stack, different quantisation. Quoting any of them as a llama.cpp expectation would be exactly the error this repo's CORRECTIONS.md exists to catch.
