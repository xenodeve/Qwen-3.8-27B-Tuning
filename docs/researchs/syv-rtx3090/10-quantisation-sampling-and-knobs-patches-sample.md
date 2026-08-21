# Quantisation, sampling and knobs — patches/sampler-small-topk-fast-softmax.patch, patches/marlin-int8-layer-select.patch, patches/marlin-int8-negative-scales.patch, patches/qwen3_5-mtp-draft-vocab.patch, patches/qwen3_5-embed-quant.patch, patches/speed-knobs-envs.patch, patches/_check_applied.py
**23 techniques.** 950 source lines across 15 files.
Files read: `patches/sampler-small-topk-fast-softmax.patch` · `patches/marlin-int8-layer-select.patch` · `patches/marlin-int8-negative-scales.patch` · `patches/qwen3_5-mtp-draft-vocab.patch` · `patches/qwen3_5-embed-quant.patch` · `patches/speed-knobs-envs.patch` · `patches/_check_applied.py` · `docs/optimizations.md (partial, grep context around int8/embed/draft-vocab/sampler)` · `docs/gotchas.md (partial, grep context around compile-cache/INT8_LAYERS)` · `docs/quality.md (partial, lines 25-60)` · `batch/README.md (partial, lines 10-45)` · `batch/start_qwen.sh (partial, lines 1-30)` · `single-user/README.md (partial, grep context around draft vocab / knob table)` · `single-user/start_qwen.sh (partial, lines 1-41)` · `prepare/README.md (partial, lines 6-36)`
> **What the reader could not see:** Five things I expected and did not find, stated as absences rather than filled in. (1) None of the seven patch files carries a measured tok/s number of its own — every figure I attach below to a sampler or quantisation technique comes from the patch *headers* (microsecond-level kernel timings, which the headers do state) or from the repo's prose docs, which are outside my slice. The headline "+4% at default sampling" for the whole sampler patch lives at docs/optimizations.md:95, not in the patch. (2) `VLLM_DRAFT_TEMP_SCALE` is NOT registered in envs.py by either speed-knobs-envs.patch or sampler-small-topk-fast-softmax.patch. Its sibling `VLLM_DRAFT_TOPK_TOPP` is registered (speed-knobs-envs.patch:23,:35) precisely so it takes part in the torch.compile cache key. So by the repo's own stated rule, changing `VLLM_DRAFT_TEMP_SCALE` between runs can replay a stale compiled graph. Possibly harmless (it only scales a tensor, no shape change), but it is an inconsistency the patches never address. (3) Both Marlin int8 selection regexes are registered in envs.py (marlin-int8-layer-select.patch:34-39) yet the code that consumes them reads raw `os.environ` (marlin-int8-layer-select.patch:53-54), never `envs.VLLM_MARLIN_INT8_*`. The registration is therefore purely for the cache key; the two readers could drift. (4) `_DRAFT_TOPK_TOPP` / `_DRAFT_TEMP_SCALE` are evaluated at *module import* via `__import__("os").environ.get(...)` (sampler-small-topk-fast-softmax.patch:259-260), so they cannot be changed after the proposer module loads. No note in the patch says so. (5) Only `forward_native` gains a `k_max` parameter. The FlashInfer / `forward_cuda` variants are never patched — the `__call__` shim (sampler-small-topk-fast-softmax.patch:121-129) silently drops `k_max` for them, so the sort-free path is unreachable on those backends. The patch header does not mention this restriction. Also absent from my slice by construction: `prepare/build_draft_vocab.py`, `prepare/quant_embed.py`, `prepare/quant_lm_head.py` and `drafter/gptq_lm_head.py` — the producers of the artifacts these patches consume (`mtp_draft_vocab_ids.pt`, the int8 embedding tables). I did not read them.

---

## EXISTS, NEVER SET — 1

### The draft vocabulary must be counted over the model's own outputs
**Where (theirs):** `patches/qwen3_5-mtp-draft-vocab.patch:5-8` · `patches/qwen3_5-mtp-draft-vocab.patch:29`

**What it does.** The patch only consumes `mtp_draft_vocab_ids.pt`; the *content* of that id list turns out to be the single largest lever in the whole single-user configuration. A list counted over the model's own generations covers far more of what it actually emits than a list counted over web text, and every uncovered token is a guaranteed rejection that also truncates the acceptance chain.

**Mechanism.** The patch's own contribution is the loading contract — `_ids_path = os.path.join(model_config.model, 'mtp_draft_vocab_ids.pt')`, existence-checked, and the head is built to `_ids.numel()` rows (:29-38), with an info log naming the size: `logger.info('MTP drafter uses a %d-token draft head', int(_ids.numel()))` (:39). The list itself is produced by prepare/build_draft_vocab.py, which slices lm_head rows (hence prepare/README.md:9-10 requiring quant_lm_head.py to run first).

**Why they needed it.** 'the id list matters more than anything else in this repo's single-user numbers: a token outside the draft vocabulary can never be proposed, so it is a guaranteed rejection that also cuts the chain.' (docs/optimizations.md:77-80). The failure mode is structural, not statistical — a missing id is not a low-probability draft, it is an impossible one.

**Their numbers.** Own-output list: 97.5% coverage of what the model generates, 96% on code. Earlier web-text list: 92%, 83% on code. Cost of the worse list: 108 vs 98 tok/s greedy, i.e. '10% of single-stream throughput on its own' (single-user/start_qwen.sh:10-13; docs/optimizations.md:21-22,:80-83). In the ladder it is the single biggest step: 93/99 -> 107/109 tok/s, 2.6 -> 2.9 tokens per step, 69% -> 74% pos-0 acceptance (single-user/README.md:254-255). Counted over 5.4M tokens of the model's own outputs (docs/optimizations.md:81).

**llama.cpp — EXISTS, NEVER SET.** This is the one technique in the slice whose substance maps onto a llama.cpp flag we have and have never set. The vLLM insight is structural rather than statistical — a token the drafter cannot propose is not a low-probability draft, it is an impossible one, and it truncates the chain — and llama.cpp's static lookup cache is the mechanism that lets a corpus of the model's own output decide what the speculator can propose. It is worth a register entry and a paired measurement, not a launch-flag change on faith: the priority inversion means it could easily cost more than it earns.

**Equivalent here:** -lcs / --lookup-cache-static FNAME with --spec-type ngram-cache, cache built by examples/lookup/lookup-create.cpp

**Evidence (llama.cpp):** `common/arg.cpp:1622-1627 (-lcs / --lookup-cache-static, set_examples includes LLAMA_EXAMPLE_SERVER — verified by reading arg.cpp)` · `common/speculative.cpp:2222-2236 (create_state_ngram_cache; n_draft = 8 literal with '// TODO get from config?'; save_static/save_dynamic hardcoded false)` · `common/speculative.cpp:2111,2124 (a cache file that fails to load GGML_ABORTs)` · `common/speculative.cpp:2542-2552 (hardcoded priority: all ngram-* before all draft-*)` · `examples/lookup/CMakeLists.txt (llama-lookup-create target exists in-tree)` · `C:\AI\llama.cpp-dflash2\ (staged set is llama-server.exe / llama-cli.exe only)`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Unknown, and gated by a real hazard. The flag exists and is accepted by llama-server, and llama-lookup-create builds a static n-gram cache from any corpus — priming it from this agent's own past outputs is exactly 'count the draft distribution over what the model actually emits'. But: ngram-cache's draft length is the literal 8 with no flag, save_static/save_dynamic are hardcoded false so -lcd never persists anything, a malformed cache file GGML_ABORTs rather than degrading, llama-lookup-create is not in the staged binary set (only llama-server/llama-cli ship in C:\AI\llama.cpp-dflash2) so it must be built, and — the hazard — the chain priority is hardcoded with EVERY n-gram speculator ranked above every model-based one, so adding ngram-cache to a draft-dflash chain means it is tried BEFORE the drafter that measured +34.7% today. The same hazard is already live in our measured draft-dflash,ngram-mod pair.

## absent, has a seam — 1

### `VLLM_DRAFT_TEMP_SCALE` — draft sharpening, measured and rejected
**Where (theirs):** `patches/sampler-small-topk-fast-softmax.patch:260` · `patches/sampler-small-topk-fast-softmax.patch:271-275` · `patches/sampler-small-topk-fast-softmax.patch:16-17`

**What it does.** An extra divisor applied to the draft logits after the ordinary temperature division, letting the draft distribution be made sharper (value < 1) than the target's. A sharper draft concentrates proposals on the modes the target is most likely to accept.

**Mechanism.** `_DRAFT_TEMP_SCALE = float(__import__('os').environ.get('VLLM_DRAFT_TEMP_SCALE', '1.0'))` at import (:260), applied as `if _DRAFT_TEMP_SCALE != 1.0: logits.div_(_DRAFT_TEMP_SCALE)` (:274-275) — the exact-1.0 guard means the default costs nothing, not even a kernel launch. It sits before the top-k/top-p truncation, so sharpening changes which tokens survive top-p as well as their weights.

**Why they needed it.** The comment gives the theory and the patch header gives the result: 'rejection sampling stays exact for any draft distribution; a sharper draft can raise acceptance when the target is truncated by top-k/top-p' (:271-273), and then 'VLLM_DRAFT_TEMP_SCALE (default 1.0) can sharpen the draft distribution; measured no gain here.' (:16-17). This is a negative result kept in the tree as a knob rather than deleted — the reasoning is sound, the measurement did not support it.

**Their numbers.** 'measured no gain here' (patches/sampler-small-topk-fast-softmax.patch:17). Condition not stated in the patch beyond 'here' — this stack, default sampling.

**llama.cpp — absent, has a seam.** The seam is identical to the previous technique and even shorter: dp.temperature is already threaded from the request into the dflash2 lattice softmax, so a scale factor is one more field on common_speculative_draft_params and one more division. It is genuinely possible and genuinely cheap; it is just not promising. Note also it would only apply at temp>0 — the greedy branch has no temperature at all.

**Equivalent here:** one extra divisor at common/speculative.cpp:1245, where dp.temperature is already applied

**Evidence (llama.cpp):** `common/speculative.cpp:1243-1252 (dist.probs[k] = exp((scores[k] - max_score) / dp.temperature))` · `common/speculative.h:71-72 (float temperature; uint32_t seed)` · `tools/server/server-context.cpp:2943 (temperature passed from the request)`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Low. The stack that invented it measured no gain, and the effect should be smaller here: llama.cpp's DFlash2 draft distribution is only selector_top_k = 16 wide, so sharpening has far less mass to redistribute than over a 248k row. Not worth spending a measurement slot on before the truncation change above.

## partial — 2

### Sort-free small-k top-k/top-p (`apply_top_k_top_p_small_k`)
**Where (theirs):** `patches/sampler-small-topk-fast-softmax.patch:162-191`

**What it does.** Replaces the full-vocab sort + cumsum used to build the top-k/top-p mask with a single `torch.topk` over at most 64 candidates per row. It reproduces the reference semantics exactly — top-k by value threshold (so ties at the k-th value are all kept), then top-p among the survivors, always keeping the single largest entry — but never sorts the 248k-entry row.

**Mechanism.** Bucketed candidate width: `kk = SMALL_K_MAX if k_max > 32 else (32 if k_max > 16 else 16)` (:176) — 16/32/64 rather than exactly k_max, so the topk width takes one of three values and the kernel/launch shape is stable across steps. `vals, idx = torch.topk(logits, kk, dim=-1)` gives descending values (:177). The per-row k-th value is fetched by gather with a clamp so a request asking for more than kk cannot index out of range: `kth = vals.gather(1, (k.to(torch.long).clamp(1, kk) - 1).unsqueeze(1))` (:178), then `keep = vals >= kth` (:179) — a value threshold, not a rank cut, which is what preserves tie behaviour. Top-p runs on the candidate set only: masked entries are set to -inf and softmaxed (`v = torch.where(keep, vals, -inf); pr = torch.softmax(v, -1)`, :182-183), the ascending cumulative mass is built with `pr.flip(-1).cumsum(-1).flip(-1)` (:185) — i.e. at position j, the mass of everything ranked at or below j — and kept where `asc > (1 - p)` (:186), with `keep_p[:, 0] = True` forcing the argmax to survive even when a single token exceeds p (:187). Finally the full row is rewritten in place: `logits.fill_(-inf)` then `logits.scatter_(1, idx, torch.where(keep, vals, -inf))` (:189-190).

**Why they needed it.** Its own docstring gives the correctness argument for why working on the candidate set is not an approximation: 'Everything outside the top-k_max candidates is masked by top-k anyway, so working on the candidate set is exact.' (patches/sampler-small-topk-fast-softmax.patch:173-174). The upstream code it replaces carries vLLM's own admission of the problem, quoted in the rejection-sampler hunk: 'NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask, which is slow for large vocab sizes. This may cause performance issues.' (:232-233).

**Their numbers.** '~10x cheaper for a 248k vocab' (patches/sampler-small-topk-fast-softmax.patch:174); the header states '~6x cheaper' for the same substitution measured over the 5 verify rows (:8). Condition: 248k vocab, small-batch spec-decode verify rows, top_k <= 64.

**llama.cpp — partial.** llama.cpp already reaches the sort-free result by a different route: top-k runs first in the default chain and uses std::partial_sort when npartial <= 128, so with k=40 it is an O(V) selection, never an O(V log V) sort, and top-p/min-p then operate on 40 survivors. So the exact optimisation is already-have-it. The half llama.cpp does NOT have is working directly on a candidate tensor: it still builds and touches the full 248k row per sampled position. -bs/--backend-sampling closes that on the main path using the request's real sampler chain, and unlike the draft-side flag it is off by default. Caveats before trying it: it self-disables under a grammar or reasoning budget, is skipped when n_probs>0 && !post_sampling_probs, and its interaction with the DFlash2 residual accept path (which reads common_sampler_get_candidates for p_draft) is untested here.

**Equivalent here:** llama_token_data_array_partial_sort_inplace (already sort-free); the remaining full-row cost is common_sampler::set_logits, mitigable with -bs / --backend-sampling

**Evidence (llama.cpp):** `src/llama-sampler.cpp:198-205 (std::partial_sort when npartial <= 128)` · `src/llama-sampler.cpp:321-338 (top_k truncates; k = min(k, size))` · `common/sampling.cpp:158-165 (set_logits fallback: cur.resize(n_vocab), loop over every token id)` · `common/sampling.cpp:739 (verifier samples the target at every drafted position)` · `common/arg.cpp:2295-2301 and common/common.h:295 (-bs, default false)` · `common/sampling.cpp:421-431 (backend sampling self-disables under grammar / reasoning budget)`

**Effort:** one-flag · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** The sort is already avoided. What is NOT avoided: set_logits materialises one llama_token_data per vocab entry every time a position is sampled — n_vocab = 248,320 for this target (qwen38-tuning/logs, `n_vocab = 248320`), ~3 MB written per position, single-threaded. With draft-dflash the verifier samples the target at up to 1+n_draft positions per step (5 today at --spec-draft-n-max 4), so this is the direct analogue of vLLM's '1-2 ms for the 5 verify rows'. -bs is the flag that would move it to the GPU and it has never been set here. Magnitude unmeasured.

### Drafting from the target's truncated support
**Where (theirs):** `patches/sampler-small-topk-fast-softmax.patch:276-290` · `patches/sampler-small-topk-fast-softmax.patch:13-16`

**What it does.** The MTP drafter used to sample its proposals from the temperature-scaled but otherwise untruncated distribution. Any draft token that fell outside the target's top-k/top-p support therefore had target probability exactly zero and was rejected with certainty. This applies the *same* top-k/top-p truncation to the draft logits before sampling, so every proposal is at least in the target's support.

**Mechanism.** Inside `compute_probs_and_sample_next_token`, after the temperature division, a guarded call: `if (_DRAFT_TOPK_TOPP and sampling_metadata.top_k is not None and sampling_metadata.top_k_max is not None and sampling_metadata.top_k_max <= 64): logits = apply_top_k_top_p(logits, sampling_metadata.top_k, sampling_metadata.top_p, k_max=sampling_metadata.top_k_max)` (:279-290). It reuses the very same `apply_top_k_top_p` the target uses, so the two supports match by construction rather than by a re-implementation that could drift. The `<= 64` guard means the truncation is only applied when it routes to the cheap sort-free path — the comment says 'Only when cheap (small top-k)' (:278).

**Why they needed it.** Stated twice. Header: 'The MTP drafter sampled from the temperature-scaled but untruncated distribution, so drafts outside the target's top-k/top-p support were always rejected. Drafts are now taken from the same truncated support (rejection sampling stays exact).' (:13-16). Inline: 'draft from the same top-k/top-p-truncated support as the target (rejection sampling stays exact for any draft distribution; matching the target's truncation raises acceptance)' (:276-278). The exactness argument is the load-bearing part — rejection sampling corrects for *any* proposal distribution q, so changing q is free of quality risk and only moves the acceptance rate.

**Their numbers.** Whole sampler patch together with the split-KV verify attention is '+4%' at default sampling (docs/optimizations.md:94-95). In the single-user ladder, '+ sampler patch + split-KV verify attention' moves 90/98 -> 93/99 tok/s at 2.6/2.6 tokens per step and 69%/70% pos-0 acceptance (single-user/README.md:249,:254) — condition: C1, realistic chat prompts, default sampling / greedy.

**llama.cpp — partial.** Our sidecar is DFlash2 (dflash.selector_top_k = 16, dflash.block_size = 8, read from the GGUF), so the stochastic branch is live: at temp>0 the drafter builds q = softmax(selector_scores / dp.temperature) over the 16 lattice candidates and fills dp.dists, which is the only thing in the tree that turns on the maximal-coupling accept path. The target's top_k/top_p/min_p are never applied to q. The verifier then looks up p_draft by scanning the target's POST-chain candidate list; a draft outside that truncated set gets p_draft = 0, fails the accept test, and breaks the loop. That is exactly the failure this patch removes. The seam is concrete and small: common_speculative_draft_params already carries temperature and seed, filled by the server from slot.task->params.sampling; add top_k/top_p the same way and apply them to dist.probs before the discrete_distribution draw. Note top_k would mostly be inert (16 < 40) — top_p/min_p is where it would act.

**Equivalent here:** --spec-draft-p-min (exists, default 0.00, never set in our launch); full top-k/top-p truncation of the draft distribution does not exist

**Evidence (llama.cpp):** `common/speculative.cpp:1236-1258 (dflash2 dist build: dist.probs = exp((score - max)/dp.temperature), normalised, then discrete_distribution)` · `common/speculative.cpp:1005 (sparams.top_k = is_dflash2 ? selector_top_k : 10 — the draft sampler is bypassed entirely on the dflash2 path)` · `common/speculative.h:50-74 (common_speculative_draft_params: temperature, seed, dists — the seam)` · `tools/server/server-context.cpp:2936-2945 (server fills .temperature = slot.task->params.sampling.temp)` · `common/sampling.cpp:753-760 and 782-784 (p_draft scanned from the post-chain candidates; miss => reject and break)` · `common/speculative.cpp:1254-1271 (--spec-draft-p-min IS implemented for dflash2, both greedy and stochastic)` · `common/arg.cpp:4101-4107 (--spec-draft-p-min, default 0.00)`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Only bites at temperature > 0. At temp <= 0 the DFlash2 branch takes the greedy argmax over the lattice, emits no dists, and the server uses the greedy accept rule — truncation cannot change an argmax, so a coding agent running greedy gains nothing. At temp 0.7 with top_k 40 / top_p 0.95, some of the 16 selector candidates will sit outside the target's top-p mass; each such draw is a certain rejection that also truncates the chain. Magnitude unknown — we have never measured acceptance at temp>0 with draft-dflash. The free half is --spec-draft-p-min, which is implemented for DFlash2 both greedy and stochastic and is currently 0.00.

## already have it — 5

### Threading `k_max` to the rejection sampler and the main sampler
**Where (theirs):** `patches/sampler-small-topk-fast-softmax.patch:230-235` · `patches/sampler-small-topk-fast-softmax.patch:243-248`

**What it does.** Two one-line call-site changes that let the verify path and the ordinary sampling path actually reach the sort-free implementation. Without them the new code exists but is never entered.

**Mechanism.** In `apply_sampling_constraints`, `return apply_top_k_top_p(logits, top_k, top_p)` becomes `... , k_max=sampling_metadata.top_k_max)` (:234-235) — this is the call that sits directly under vLLM's own 'sorting ... is slow for large vocab sizes' note. In `Sampler`, the `topk_topp_sampler(...)` call gains `k_max=sampling_metadata.top_k_max,` (:247), which is the call the `__call__` shim intercepts.

**Why they needed it.** The rejection-sampler site is the one that costs '~1-2 ms for the 5 verify rows' at 248k vocab (patches/sampler-small-topk-fast-softmax.patch:6-7) — the spec-decode verify step applies the constraints to every one of the k+1 positions at once, so the sort cost is multiplied by the speculation depth.

**Their numbers.** 5 verify rows corresponds to 4 draft tokens plus the bonus token — the single-user default is `DRAFT_TOKENS=4` (single-user/README.md:363).

**llama.cpp — already have it.** The substance of this hunk is 'make the verify path apply the same constraints as the ordinary sampling path'. llama.cpp does that by construction: both accept variants call common_sampler_sample(gsmpl, ctx, idxs[i], ...) at every drafted position, which runs the identical chain object with the identical parameters. There is no separate constraint-application function for verification that could drift, and therefore no k_max to thread.

**Equivalent here:** common_sampler_sample called per verify position inside common_sampler_sample_and_accept_n

**Evidence (llama.cpp):** `common/sampling.cpp:692-720 (greedy variant: common_sampler_sample per position)` · `common/sampling.cpp:736-786 (residual variant: same call, grammar_first=true)` · `common/sampling.cpp:608-687 (one common_sampler_sample used by both)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero

### Per-layer regex selection for the Marlin int8-activation path
**Where (theirs):** `patches/marlin-int8-layer-select.patch:45-63` · `patches/marlin-int8-layer-select.patch:23-24`

**What it does.** Turns vLLM's all-or-nothing `VLLM_MARLIN_INPUT_DTYPE=int8` into a per-layer decision. Two regexes matched against the layer's prefix name decide whether that layer runs W4A8 (int4 weights, int8 activations, int8 tensor cores) or stays on the 16-bit activation path. This is simultaneously a crash fix and a quality/throughput dial.

**Mechanism.** `get_marlin_input_dtype(prefix)` already received the layer prefix; the patch inserts an early-return before the dtype dispatch. It reads `VLLM_MARLIN_INT8_EXCLUDE_RE` (default `"lm_head|mtp"`) and `VLLM_MARLIN_INT8_INCLUDE_RE` (default empty) via `os.environ`, and: if the exclude regex matches the prefix -> `return` (None, i.e. keep 16-bit activations); else if an include regex is set and does *not* match -> `return` (patches/marlin-int8-layer-select.patch:52-59). Only then does the original `int8`/`fp8` dispatch run (:60-63). An empty include regex means 'no include filter', so exclude-only is the default mode. Imports are function-local aliased (`import os as _os, re as _re`, :52) to keep the hunk self-contained.

**Why they needed it.** Two reasons, both stated: 'Stock vLLM applies VLLM_MARLIN_INPUT_DTYPE=int8 to every Marlin layer, which (a) crashes on this checkpoint's int8-weight lm_head (W8A8 is unsupported) and (b) gives no way to trade throughput against quality per layer type.' (patches/marlin-int8-layer-select.patch:3-6). The default exclude value is explained inline: 'lm_head is int8-weight -> W8A8 unsupported by marlin; mtp stays 16-bit' (:51). Note the interaction with the rest of the stack: this repo requantised lm_head to int8 itself (prepare/quant_lm_head.py), so it *created* the W8A8 collision that this default avoids.

**Their numbers.** Measured throughput/quality ladder, 64 concurrent, 128-in/512-out, RTX 3090: int8 on gate/up only (`INT8_LAYERS=gate_up`) 787 tok/s e2e, +0.9% perplexity; whole MLP (the shipped default) 876->942 tok/s, +2.2% PPL, GSM8K 95.0%; all linears (`INT8_LAYERS=.`) 1,025->1,042 tok/s and ~1,150-1,222 steady-state decode, +3.7% PPL — and the all-linears row requires `GPU_UTIL=0.95` because the extra transient scratch OOMs inside the GDN chunk kernel at batch mode's 0.972 default (batch/README.md:17-20,:34-36; docs/quality.md:33-37,:45-48). Baseline for comparison: W4A16 fp16-state, 707 tok/s. `INT8_LAYERS` is wired to `VLLM_MARLIN_INT8_INCLUDE_RE` at batch/start_qwen.sh:99, default `mlp`.

**llama.cpp — already have it.** Two separate llama.cpp facts answer this. (1) There is no runtime activation-dtype switch because llama.cpp does not have a 16-bit-activation path for quantised weights to begin with: MMQ packs src1 into q8_1 int8 blocks and does dp4a / int8-MMA dot products, and MMVQ (the decode-shaped kernel, ne11 <= 8) does the same. So W4A8-int8 is unconditional, which is also why the vLLM crash this patch avoids — a W8A8 lm_head — cannot occur here. (2) The per-layer quality/size trade is made offline at quantisation time, by regex, with the same first-match-wins semantics; that surface exists and is documented in the map. There is no seam and no need for a runtime selector.

**Equivalent here:** MMQ/MMVQ q8_1 activation quantisation (unconditional, no flag) + offline --tensor-type / --token-embedding-type / --output-tensor-type

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/mmq.cuh:45 (int8_t qs[QK8_1_MMQ] — activations packed as int8)` · `ggml/src/ggml-cuda/mmq.cuh:545-581 (ggml_cuda_mmq_vec_dot_q4_0_q8_1_dp4a etc.)` · `ggml/src/ggml-cuda/mmq.cu:312-314 (turing_mma_available short-circuits: MMQ always chosen on Ada for supported types)` · `ggml/src/ggml-cuda/mmvq.cu:289-337 (MMVQ, ne11 <= 8, same q8_1 activations)` · `tools/quantize/quantize.cpp:314-361 (--tensor-type regex), src/llama-quant.cpp:693-718 (first match wins, suppresses the mixture)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Nothing to switch on: int8 activations against int4/int2 weights are already the default for every quantised tensor on this GPU. The per-layer quality dial exists but is offline and already exercised — UD-IQ2_XXS is a per-tensor mixed quant by construction. llama-quantize is not staged in C:\AI\llama.cpp-dflash2, so using it would mean building it first.

### Vocab-truncated draft head for the MTP drafter
**Where (theirs):** `patches/qwen3_5-mtp-draft-vocab.patch:21-39` · `patches/qwen3_5-mtp-draft-vocab.patch:57-73`

**What it does.** The MTP draft module stops scoring the full 248k-row lm_head on every draft token and instead scores a ~40k-row slice, then scatters those logits back into a full-width row of -inf. The target model is untouched, so rejection sampling — and therefore output distribution — is unchanged; only which tokens can be proposed changes.

**Mechanism.** At construction, if `<model_dir>/mtp_draft_vocab_ids.pt` exists and `MTP_DRAFT_VOCAB != '0'`, load the id tensor to CPU and build a second head sized to it: `ParallelLMHead(int(_ids.numel()), config.hidden_size, quant_config=vllm_config.quant_config, prefix=maybe_prefix(prefix, 'draft_lm_head'))` (:26-38) — note it takes the same quant_config, so the draft head is quantised like everything else. A second `LogitsProcessor(int(self.model.draft_vocab_ids.numel()))` is created only when the head exists (:49-53). In `compute_logits`, the truncated path runs the small processor, lazily migrates the id tensor to the logits' device once and caches it back (`if ids.device != sub.device: ids = ids.to(sub.device); self.model.draft_vocab_ids = ids`, :67-69), then `full = sub.new_full((sub.shape[0], self.config.vocab_size), float('-inf')); full.index_copy_(1, ids, sub); return full` (:70-72). Returning a full-width -inf row rather than a compact one means every downstream consumer — sampler, top-k, rejection sampler — needs no change at all.

**Why they needed it.** Header: 'the MTP drafter scores only those rows instead of the full 248k-row lm_head; logits of all other ids are -inf. Speculative decoding stays exact (rejection sampling uses the target model), only the acceptance rate can change.' (patches/qwen3_5-mtp-draft-vocab.patch:5-8). The economics from docs: 'the shipped MTP draft module is bf16 (850 MB) and every draft token also runs the full 248k-row lm_head (1.3 GB), so each extra draft cost ~3 ms and MTP-3 was already slower than MTP-2 ... A draft now costs ~0.5-1 ms and four of them pay off.' (docs/optimizations.md:71-77).

**Their numbers.** 40,960 rows instead of 248k (prepare/README.md:22). Per-draft cost ~3 ms -> ~0.5-1 ms, which is what makes 4 drafts profitable where 3 previously were not (docs/optimizations.md:73,:77). Acceptance cost of truncation, C1 realistic chat prompts: 'k=4, full 248k head instead of 40k' gives 85/91 tok/s at 2.85/3.0 tokens per step and 74%/76% pos-0 acceptance, versus the 40k head's 84/89 at 2.5/2.4 and 69%/71% — i.e. the full head accepts more but is slower per draft (single-user/README.md:250,:252). A 49k vocab variant: 109/115 tok/s vs the shipped list's ~114/~124 (single-user/README.md:257,:259).

**llama.cpp — already have it.** The DFlash2 draft path never materialises a distribution over the target vocab: llama_get_embeddings_nextn returns a row whose first selector_top_k floats ARE candidate token ids and whose remainder is the predecessor-conditioned score lattice, and the drafter reads ids straight out of it (dist.ids[k] = (llama_token) row[k]). Our sidecar carries dflash.selector_top_k = 16, so each draft position costs a 16-wide read, not a vocab-wide matmul. That is the same economics the patch buys, achieved in the artifact rather than in the runtime. For the other draft types (draft-simple / mtp / eagle3) llama.cpp does run the draft model's full head and there is no truncation seam and, crucially, no id-remap step — a sidecar with a shrunk output head would emit indices into its own head that llama.cpp would treat as target vocab ids, silently wrong. So the technique is not portable to those types without new code.

**Equivalent here:** the DFlash2 selector lattice — the draft emits only selector_top_k candidate ids per position, never a full-vocab head

**Evidence (llama.cpp):** `common/speculative.cpp:1237-1240 (const float * row = lattice + (beg+i)*n_embd_dec; scores = row + selector_top_k + predecessor*selector_top_k)` · `common/speculative.cpp:1244 (dist.ids[k] = (llama_token) row[k])` · `Qwen3.8-27B-DFlash2-Q4_K_M.gguf metadata: dflash.selector_top_k = 16, dflash.block_size = 8, dflash.selector_rank = 256` · `common/speculative.cpp:988-996 (n_draft_max = block_size - 1 = 7 for this sidecar)` · `common/speculative.cpp:68-131 and 238 (vocab-compat guard exists only for draft-simple — nothing checks a dflash sidecar's head width)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** The cost this removes does not exist in our profile. Adjacent and real, though: our sidecar's dflash.block_size = 8 permits n_draft_max = block_size - 1 = 7, and we launch with --spec-draft-n-max 4 (qwen38-tuning/bench/dflash2_arena.py:65). Since per-draft cost here is a lattice read rather than a 248k head evaluation, the vLLM argument that cheap drafts make deeper drafts pay applies directly — depths 5, 6, 7 are unmeasured headroom.

### Wiring `quant_config` into VocabParallelEmbedding (both models)
**Where (theirs):** `patches/qwen3_5-embed-quant.patch:15-21` · `patches/qwen3_5-embed-quant.patch:24-32`

**What it does.** Four added lines total that route the token embedding table through vLLM's existing quantized-embedding kernel. vLLM already ships a dequant-on-gather path (`CompressedTensorsEmbeddingWNA16Int`); the qwen3_5 model code simply never handed `quant_config` to `VocabParallelEmbedding`, so a quantised embedding table on disk could not be used.

**Mechanism.** Two identical two-line hunks. In the main model: `VocabParallelEmbedding(self.vocab_size, config.hidden_size, quant_config=self.quant_config, prefix=maybe_prefix(prefix, 'embed_tokens'))` (:16-21). In the MTP draft module the same, sourcing from `vllm_config.quant_config` instead of `self.quant_config` (:27-32). The `prefix` argument is as load-bearing as `quant_config` — it is what lets the quantisation config's per-module exclude lists and the Marlin layer-select regexes address this layer by name.

**Why they needed it.** The header identifies the omission precisely: 'The kernel exists upstream; the qwen3_5 model code just never passes quant_config to VocabParallelEmbedding.' (patches/qwen3_5-embed-quant.patch:2-3). And why the second hunk is not optional: 'Two hunks: the main model, and the MTP draft module (without the second one, single-user mode crashes on load with "no parameter named embed_tokens.weight_packed in Qwen3_5MultiTokenPredictor").' (:4-6) — the MTP module has its own embedding table, so a checkpoint quantised on disk mismatches the un-quantised parameter shape there.

**Their numbers.** Qwen3.8-27B has untied embeddings, so lm_head and embed_tokens are two separate 2.5 GB bf16 matrices that public W4A16 quants leave alone; requantising both to int8 group-128 in place returns 2.6 GB of VRAM at ~0.6% round-trip error with 'no quality regression we could find' (docs/optimizations.md:39-44; ~1.3 GB each per prepare/README.md:14-15).

**llama.cpp — already have it.** There is no analogue of the vLLM omission because there is no layer-construction API in llama.cpp that can silently skip a quantisation config. The loader reads whatever ggml_type each tensor carries and there is no path from any flag to a change of type at load. Choosing the embedding's type is an offline decision with a dedicated flag that outranks the k-quant mixture, and the mixed-quant artifact we already run has made that choice. Nothing to wire.

**Equivalent here:** the quantiser treats token_embd like any other tensor; --token-embedding-type controls it explicitly

**Evidence (llama.cpp):** `src/llama-quant.cpp:683-688 (--token-embedding-type / --output-tensor-type checked first, return immediately)` · `src/llama-quant.cpp:452-454 (tied embeddings quantised by the OUTPUT rule)` · `src/llama-model.cpp:1368-1370 (dev_input hard-wired to CPU: 'very little benefit to offloading the input layer')` · `src/llama-model-loader.cpp:1110-1114 (TENSOR_DUPLICATED — the tied copy is re-tagged as output and follows dev_output)` · `common/speculative.cpp:2405-2406 (cache_type_k/v overwritten with the speculative struct's own f16 defaults — -ctk/-ctv never reach the draft)` · `common/arg.cpp:4022-4047 (-ctkd / -ctvd exist, accepted by every binary)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** No VRAM to recover from this: token_embd is quantised in the UD-IQ2_XXS artifact already, and it is pinned to the CPU device regardless (dev_input), so it costs host RAM not VRAM. The one caveat is the tied-embedding duplicate, which is re-tagged as the output tensor and does land in VRAM. The honest VRAM lever in the same spirit that IS unused: -ctkd/-ctvd. Our DFlash2 launch passes only -md, --spec-draft-n-max 4, -ngld 99, so the draft context's KV cache is f16 while the target's is q4_0 — the main model's -ctk/-ctv do not propagate.

### `_check_applied.py` — heuristic patch-presence check as a fallback for reverse dry-run
**Where (theirs):** `patches/_check_applied.py:1-33`

**What it does.** A ~20-line verifier answering 'is this patch's content present in the installed vLLM tree?' by textual search rather than by patch semantics. It exists because the exact method — `patch --reverse --dry-run` — gives false negatives when a second patch has since edited the same region.

**Mechanism.** Collect the substantial added lines: lines starting with `+` but not `+++`, stripped, longer than 24 characters, and not beginning with `#`, `"` or `'` (patches/_check_applied.py:14-18) — the length floor and the comment/docstring exclusion drop the lines most likely to be reformatted or to collide by chance. Cap at the first 400 (`added = added[:400]`, :21). Then walk the target tree, concatenate every `.py` file into one string with OSError swallowed per file (:23-31), and count how many candidate lines appear in it. Verdict: `sys.exit(0 if hits >= max(3, int(0.8 * len(added))) else 1)` (:32) — an 80% threshold with an absolute floor of 3, so a tiny patch still needs 3 real hits and a large one tolerates 20% drift from later edits. A patch that yields no candidate lines exits 1 rather than trivially passing (:19-20).

**Why they needed it.** Stated in the docstring: 'verify.sh checks patches with a reverse dry-run, which is exact but cannot see a patch whose hunks were disturbed by a second patch touching the same file (the DFlash2 pair does that). This is the fallback: take the substantial lines a patch adds and look for them in the tree.' (patches/_check_applied.py:3-6). It is a deliberate trade of exactness for robustness against patch-on-patch stacking — the same problem envs.py has with two patches editing it.

**Their numbers.** Thresholds: added lines must exceed 24 characters; at most 400 are sampled; the pass bar is `max(3, 80% of sampled)` hits (patches/_check_applied.py:17, :21, :32).

**llama.cpp — already have it.** The heuristic exists because the target is a pip-installed package with stacked patches, where the exact check (reverse dry-run) gives false negatives. Our tree is a git checkout built with CMake: the commit is the identity, and the running binary reports it two ways — --version at startup and build_info over HTTP. That said, the underlying discipline does apply here and is worth naming: the binary under test is C:\AI\llama.cpp-dflash2\llama-server.exe, staged separately from the source at C:\AI\llama.cpp, so 'is the binary I am measuring built from the source I am reading' is a real question and build_info is the way to answer it inside a benchmark run rather than by inspection.

**Equivalent here:** llama-server --version (build 10499, commit 1deefcca3) and GET /props build_info; plus git in the source tree

**Evidence (llama.cpp):** `tools/server/server-context.cpp:4580-4629 (GET /props reports build_info, model_path, model_ftype)` · `tools/server/server-context.cpp:650 ({"speculative", can_speculate()})` · `tools/server/server-task.cpp:83,142 (speculative.types reported as the comma-joined effective list)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero — the exact answer is already available, so the heuristic would be a downgrade

## not applicable — 14

### Host-side `top_k_max` on SamplingMetadata
**Where (theirs):** `patches/sampler-small-topk-fast-softmax.patch:30-32` · `patches/sampler-small-topk-fast-softmax.patch:305`

**What it does.** Adds one integer field, `top_k_max`, to vLLM's `SamplingMetadata`: the maximum top_k over all requests in the batch, or None when no request in the batch sets top-k at all. It is the enabling fact for every sort-free sampling path in this patch — the sampler can decide, before touching the GPU, whether a cheap exact algorithm is legal for this whole batch.

**Mechanism.** `InputBatch` already keeps a pinned CPU mirror of top_k. The patch computes the max from that mirror, not from the device tensor: `top_k_max=None if self.no_top_k else int(self.top_k_cpu[:num_reqs].max())` (patches/sampler-small-topk-fast-softmax.patch:305). Because it reads `top_k_cpu`, there is no device-to-host copy and no stream synchronisation — the value is available as a Python int while building the metadata. The declaration is `top_k_max: int | None = None` with the comment that it is 'vocab_size when a request has no top-k', i.e. the sentinel for 'nothing to truncate' is the vocab size, so a `<= 64` test naturally fails for un-truncated requests.

**Why they needed it.** The patch header states the cost being avoided: 'apply_top_k_top_p sorts the whole 248k vocab for every row (~1-2 ms for the 5 verify rows). With top_k <= 64 for every request in the batch (known on the host, no GPU sync) a single torch.topk over the vocab is exact and ~6x cheaper. SamplingMetadata gains a host-side `top_k_max` for this.' (patches/sampler-small-topk-fast-softmax.patch:6-9). The parenthetical 'known on the host, no GPU sync' is the whole design constraint — a device-side max would cost a sync per step and eat the saving.

**Their numbers.** ~1-2 ms per step for the 5 verify rows of a 248k vocab is the cost this unlocks removing; the model's default sampling is temperature 1.0, top-p 0.95, top-k 20, so top_k_max is 20 in the common case, comfortably under the 64 threshold (patches/sampler-small-topk-fast-softmax.patch:4-8).

**llama.cpp — not applicable.** The whole point of top_k_max is to learn a batch-wide top-k bound without a device→host sync. llama.cpp never has that problem: sampling is per-slot and entirely host-side, and top_k lives as a plain int32 in common_params_sampling, read directly by the sampler builder. There is no batch metadata object and no device mirror to reconcile. With -np 1 there is not even a batch to take a max over.

**Equivalent here:** none needed — params.sampling.top_k is already a host int

**Evidence (llama.cpp):** `common/common.h:229 (int32_t top_k = 40)` · `common/sampling.cpp:346-406 (chain assembly reads params.top_k on the host)` · `tools/server/server-context.cpp:1224-1226 (one slot per -np, one sampler each)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero

### Three-value bucketing of the topk width (16/32/64)
**Where (theirs):** `patches/sampler-small-topk-fast-softmax.patch:162` · `patches/sampler-small-topk-fast-softmax.patch:176`

**What it does.** Rather than calling `torch.topk(logits, k_max)` with whatever the batch's max top_k happens to be, the code rounds k_max up to one of three fixed widths — 16, 32 or 64 — with `SMALL_K_MAX = 64` as the cap and the eligibility threshold. Extra candidates beyond the request's k are harmless because the value-threshold `keep` mask discards them.

**Mechanism.** `kk = SMALL_K_MAX if k_max > 32 else (32 if k_max > 16 else 16)` (:176). The subsequent gather clamps each request's k into `[1, kk]` (:178), so a per-request k smaller than kk simply produces a higher `kth` threshold and more of the kk candidates fail `vals >= kth`. `SMALL_K_MAX = 64` is defined as a module constant (:162) and is reused verbatim as the dispatch gate (:207) and as the drafter's gate (:283).

**Why they needed it.** Not spelled out in words in the patch, but the structure is explicit: three discrete widths mean `torch.topk` sees one of three shapes instead of a shape that moves with whatever the request asked for. 64 is the ceiling at which the header's claim holds — 'With top_k <= 64 for every request in the batch ... a single torch.topk over the vocab is exact and ~6x cheaper' (:7-8).

**llama.cpp — not applicable.** The bucketing exists so torch.topk sees one of three shapes instead of a shape that moves with the request. llama.cpp's host top-k is std::partial_sort — a CPU call with no launch geometry. On the device path (-bs) ggml_top_k is built into the sampler graph once at llama_set_sampler time with the request's fixed k, so the shape is already constant for the life of that chain. There is nothing to bucket.

**Equivalent here:** none — no launch shape to stabilise

**Evidence (llama.cpp):** `src/llama-sampler.cpp:193-215 (host partial sort)` · `src/llama-sampler.cpp:1477-1502 (device top_k = one ggml_top_k + ggml_get_rows, built once)` · `src/llama-context.cpp:1209-1258 (llama_set_sampler builds the backend chain once)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero

### Dispatch gate for the sort-free path
**Where (theirs):** `patches/sampler-small-topk-fast-softmax.patch:194-214`

**What it does.** `apply_top_k_top_p` grows a `k_max` keyword and takes the small-k path only when four conditions all hold, otherwise falls through unchanged to the stock Triton/CPU/PyTorch implementations. The gate is placed *above* the existing CPU branch, so the small-k path is preferred on GPU but never taken on CPU.

**Mechanism.** `if (k is not None and k_max is not None and k_max <= SMALL_K_MAX and not current_platform.is_cpu()): return apply_top_k_top_p_small_k(logits, k, p, k_max)` (:204-210). Note `p` may be None — the small-k function handles that with `if p is not None` (:180) — but `k` may not, because the whole method is built on a k-th value threshold. The pre-existing `if p is None and k is None: return logits` early-out is left in place above it (:201-202).

**Why they needed it.** The CPU exclusion and the `k is not None` requirement are the correctness boundary: without a top-k there is no bounded candidate set, and the exactness argument ('everything outside the top-k_max candidates is masked by top-k anyway', :173-174) collapses.

**llama.cpp — not applicable.** A gate is only meaningful when two implementations of the same filter coexist and one is conditionally exact. llama.cpp ships exactly one host top-k and one host top-p, each with an internal fast path chosen from the data (already-sorted flag, size thresholds), not from a caller-supplied bound. There is no second implementation to route to and no CPU-vs-GPU fork to guard.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-sampler.cpp:333-337 (top_k skips sorting entirely if cur_p is already flagged sorted)` · `src/llama-sampler.cpp:1563-1592 (top_p's own internal 256/1024 heuristics)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero

### `_accepts_k_max` signature-introspection shim on `TopKTopPSampler.__call__`
**Where (theirs):** `patches/sampler-small-topk-fast-softmax.patch:121-129`

**What it does.** Lets callers unconditionally pass `k_max=` to the sampler module even though only `forward_native` was taught to accept it. `TopKTopPSampler` binds `self.forward` to one of several backend implementations at construction time (FlashInfer, Triton, native); the shim inspects the bound one once and either forwards `k_max` or drops it.

**Mechanism.** A `__call__` override that lazily caches the answer on the instance: `if not hasattr(self, '_accepts_k_max'): import inspect; self._accepts_k_max = 'k_max' in inspect.signature(self.forward).parameters` then dispatches with or without the kwarg (:121-129). The comment states the intent: 'forward variants that don't know k_max just ignore it' (:122).

**Why they needed it.** Minimises the patch surface against upstream. Only one of vLLM's several sampler backends had to be modified; the rest keep working untouched, at the cost of one `inspect.signature` call per sampler instance. The consequence is that the fast path silently does not apply on non-native backends (see notable_absences).

**llama.cpp — not applicable.** This is a Python packaging device for a monkeypatched dependency with several interchangeable backends. llama.cpp's sampler is a C++ vtable (llama_sampler_i) compiled from source in this tree; adding a parameter means editing the struct and rebuilding, and there is no runtime signature to inspect nor a second backend implementation to keep compatible.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-sampler.cpp:505-520 (llama_sampler_i interface, including backend_init)` · `include/llama.h (llama_sampler is a compiled struct, not a duck-typed object)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero

### Two-launch multi-block row softmax in Triton (`row_softmax_fp32`)
**Where (theirs):** `patches/sampler-small-topk-fast-softmax.patch:38-105`

**What it does.** A new file `v1/sample/ops/row_softmax.py` implementing softmax over a 2-D tensor with very few rows and a very wide last dimension, by splitting each row across 64 thread blocks instead of the one block per row that `torch.softmax` uses. Two kernel launches: one computes per-chunk (max, sum) partials, one combines them and writes the normalised output.

**Mechanism.** Grid is `(B, NCHUNK)` for both launches with `_NCHUNK = 64` and `_BLOCK = 4096` (:50-51, :103-104, num_warps=4). `_partial_kernel` (:55-72) gives block `c` the slice `[c*per, min(c*per+per, V))` where `per = ceil(V/NCHUNK)`, and walks it in BLOCK-sized strips running the standard online-softmax rescale: `s = s * exp(m - m_safe) + sum(exp(x - m_safe))` with `m_safe = where(m_new == -inf, 0.0, m_new)` guarding the all-masked case (:66-70). It stores (m, s) per chunk into `pm`/`ps` of shape [B, 64]. `_final_kernel` (:76-93) has every block re-load all 64 partials, reduce them to the row max `M` and total `S = sum(ps * exp(pm - M_safe))` (:82-84), form `inv = 1.0 / max(S, 1e-38)` (:85) — a floor, not a select, so an all -inf row yields zeros rather than NaN — and then re-read its own chunk of x and write `exp(x - M_safe) * inv`. x is read twice; the -inf sentinels appear in three places (`other=float('-inf')` on both loads, and the `tl.where(m == -inf, -inf, m - m_safe)` guard) so that fully-masked chunks contribute nothing.

**Why they needed it.** The module docstring states the problem exactly: 'torch.softmax launches one thread block per row; for a single request's 248k-entry logits row that is one SM doing ~140 us of work per call, and the spec-decode sampler calls it several times per step. This splits every row over NCHUNK blocks (two launches, ~10 us).' (patches/sampler-small-topk-fast-softmax.patch:41-44). The pathology is single-user decoding specifically: with one request there is one row, so torch's parallelisation strategy leaves 81 of the 3090's 82 SMs idle.

**Their numbers.** 140 us for one 248k-wide row under torch.softmax, versus ~10 us for the two-launch Triton version — condition: single logits row of width 248k, called several times per spec-decode step (each draft plus the target verify) (patches/sampler-small-topk-fast-softmax.patch:10-12, :41-44). Repeated in docs/optimizations.md:91-92.

**llama.cpp — not applicable.** ggml's CUDA softmax has the same one-block-per-row structure the patch complains about — rowx is derived from blockIdx and the kernel walks the whole row in one block. So the pathology would exist if llama.cpp ever softmaxed a full vocab row on device. It does not. On the host, top-k truncates to 40 candidates before anything softmaxes, and llama_sampler_dist does its own fused softmax over that 40-entry cur_p. Attention softmax is fused inside FLASH_ATTN_EXT. Even under -bs the chain runs top_k first, so device top_p/temp/dist see a 40-wide row. The idea (split one wide row across SMs) is portable to ggml in principle — the seam is ggml/src/ggml-cuda/softmax.cu — but there is no caller here that would benefit.

**Equivalent here:** ggml_cuda soft_max_f32 (one block per row, same shape as torch) — but no wide-row softmax is ever executed in this profile

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/softmax.cu:57-72 (const int64_t i01 = blockIdx.x; rowx = ...; x += rowx*ncols — one block per row)` · `common/sampling.cpp:346-406 (default chain order: top_k before top_p/min_p/temp/dist)` · `src/llama-sampler.cpp:1186-1214 (dist does its own fused softmax over cur_p, no sort)` · `src/llama-graph.cpp:2540-2565 (attention softmax fused into FLASH_ATTN_EXT)`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** zero as things stand. There is no 248k-wide softmax anywhere in the live path to accelerate.

### `softmax_fp32` shape-based fast-path dispatch
**Where (theirs):** `patches/sampler-small-topk-fast-softmax.patch:108-112`

**What it does.** A drop-in replacement for `x.softmax(dim=-1, dtype=torch.float32)` that routes to the Triton multi-block kernel only in the regime where it wins, and otherwise calls torch. This is what makes the kernel safe to splice into three different call sites without auditing each one's shapes.

**Mechanism.** Four-way predicate: `x.is_cuda and x.ndim == 2 and x.shape[0] <= 16 and x.shape[1] >= 16384 and x.stride(1) == 1` (:110). Rows <= 16 encodes 'few enough rows that torch's one-block-per-row scheme starves the GPU'; width >= 16384 encodes 'wide enough that splitting pays for the second launch and the double read of x'; the stride check enforces the contiguous last dim that `row_softmax_fp32` asserts (:98).

**Why they needed it.** The patch header scopes the replacement as 'a two-launch multi-block Triton softmax for <= 16 rows' (:12) — beyond that row count torch's own strategy already fills the machine, and the extra launch plus the second pass over x would be a loss.

**Their numbers.** Thresholds: <= 16 rows, >= 16384 columns (patches/sampler-small-topk-fast-softmax.patch:110).

**llama.cpp — not applicable.** A dispatcher choosing between two softmax implementations by (rows <= 16, cols >= 16384, contiguous) presupposes a second implementation and a caller that sometimes hands it a wide row. ggml has neither in this path: one soft_max kernel, and no full-vocab softmax reaches it. Nothing to dispatch on.

**Equivalent here:** none

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/softmax.cu:52-72 (single soft_max_f32 template, selected by ncols/block_size only)` · `common/sampling.cpp:400-406 (dist appended last; it softmaxes the truncated candidate set)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero

### Splicing `softmax_fp32` into the three hot softmaxes
**Where (theirs):** `patches/sampler-small-topk-fast-softmax.patch:151-154` · `patches/sampler-small-topk-fast-softmax.patch:223-227` · `patches/sampler-small-topk-fast-softmax.patch:291-293`

**What it does.** The wide-row softmax is substituted at exactly three call sites, each hit once or more per spec-decode step: the native sampler's probability computation, the rejection sampler's target-probability computation, and the MTP drafter's own probability computation.

**Mechanism.** Each site does a function-local import (`from vllm.v1.sample.ops.row_softmax import softmax_fp32 # syv patch`) and replaces `X.softmax(dim=-1, dtype=torch.float32)` with `softmax_fp32(X)`. In `forward_native` it becomes `probs = softmax_fp32(logits)` (:154); in `rejection_sample` it becomes `target_probs = softmax_fp32(target_logits)` with the pre-existing `assert target_probs.is_contiguous()` immediately after (:226-227) — the Triton kernel allocates with `torch.empty(B, V)` so contiguity holds; in `compute_probs_and_sample_next_token` it becomes `probs = softmax_fp32(logits)` (:293). The `log_softmax` used for `processed_logprobs` is deliberately left on torch (:150).

**Why they needed it.** Header: '140 us for one 248k-wide row, called several times per step (each draft, the target verify)' (:10-12). Three sites are what 'several times per step' resolves to: one per draft token via the proposer, plus the verify path's target probs, plus the sampler.

**llama.cpp — not applicable.** The three vLLM sites are: the sampler's probs, the rejection sampler's target probs, and the MTP drafter's probs. In llama.cpp the first two are the same 40-candidate host softmax inside llama_sampler_dist / the residual verifier, and the third does not exist at all — the DFlash2 drafter never computes a full-vocab distribution, it reads a 16-wide selector lattice and softmaxes those 16 scores. There is no wide row at any of the three positions.

**Equivalent here:** none

**Evidence (llama.cpp):** `common/speculative.cpp:1239-1252 (dflash2 builds dist.probs over selector_top_k entries only)` · `common/sampling.cpp:753-760 (residual verifier reads p from the post-chain candidate list)` · `src/llama-sampler.cpp:1186-1214 (dist's fused softmax over cur_p)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero

### `VLLM_DRAFT_TOPK_TOPP` kill switch
**Where (theirs):** `patches/sampler-small-topk-fast-softmax.patch:259` · `patches/speed-knobs-envs.patch:23` · `patches/speed-knobs-envs.patch:35`

**What it does.** Environment variable, default on, that disables the draft-side truncation above and returns the drafter to sampling from the untruncated distribution.

**Mechanism.** Read once at module import in the proposer: `_DRAFT_TOPK_TOPP = __import__('os').environ.get('VLLM_DRAFT_TOPK_TOPP', '1') == '1'` (patches/sampler-small-topk-fast-softmax.patch:259) — the `__import__` idiom avoids adding an import line to the patched file's header. Separately declared and registered in `envs.py` as `VLLM_DRAFT_TOPK_TOPP: bool = True` / `lambda: os.environ.get('VLLM_DRAFT_TOPK_TOPP', '1') == '1'` (patches/speed-knobs-envs.patch:23,:35) so it joins the torch.compile cache key.

**Why they needed it.** 'VLLM_DRAFT_TOPK_TOPP=0 disables.' (patches/sampler-small-topk-fast-softmax.patch:16). The registration is for the cache-key reason given in speed-knobs-envs.patch:5-9. Note the DFlash2 drafter has its own separate equivalent, `VLLM_DFLASH2_DRAFT_TOPK_TOPP` (single-user/README.md:362) — different variable, different drafter.

**llama.cpp — not applicable.** This exists only to gate the draft-side truncation, and its second reason for existing (registration in envs.py so it joins the torch.compile cache key) has no counterpart here at all. If the truncation were added, llama.cpp's own idiom is a --spec-draft-* option with an auto-generated LLAMA_ARG_* env alias, next to --spec-draft-p-min; that is a one-flag change on top of the small patch, not a separate technique.

**Equivalent here:** would be a --spec-draft-* flag in the common/arg.cpp:4076-4116 block, not an env var

**Evidence (llama.cpp):** `common/arg.cpp:4076-4116 (the --spec-draft-n-max/-n-min/-p-min/-p-split/-backend-sampling block, each with set_env)` · `common/speculative.h:50-74 (params struct the flag would land in)`

**Effort:** one-flag · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** zero on its own — nothing to switch off until the previous technique is patched in

### Registering the selection regexes in envs.py for the torch.compile cache key
**Where (theirs):** `patches/marlin-int8-layer-select.patch:32-39` · `patches/marlin-int8-layer-select.patch:6-10`

**What it does.** Declares `VLLM_MARLIN_INT8_INCLUDE_RE` and `VLLM_MARLIN_INT8_EXCLUDE_RE` in vLLM's `envs.py` — both in the type-annotation block and in the lambda registry — purely so that vLLM folds them into the torch.compile cache key. The consuming code does not use the registered values.

**Mechanism.** Two lines in the dataclass-style annotation block (`VLLM_MARLIN_INT8_INCLUDE_RE: str = ""`, `VLLM_MARLIN_INT8_EXCLUDE_RE: str = "lm_head|mtp"`, :23-24) and two lambdas in the environment_variables registry (:34-39). vLLM hashes its registered env vars into the compiled-graph cache key; registering is the whole mechanism.

**Why they needed it.** Stated as a concrete failure: '...and registers them in envs.py so they take part in the torch.compile cache key (otherwise switching the selection replays a stale compiled graph and crashes with KeyError: input_global_scale).' (patches/marlin-int8-layer-select.patch:7-10). Restated as a general gotcha with the escape hatch: 'The torch.compile cache does not know about your env vars. Switching INT8_LAYERS between runs replays a compiled graph that expects the other layer set and dies with KeyError: input_global_scale. Our patch registers the selection env vars with vLLM so they become part of the cache key; if you invent your own, VLLM_DISABLE_COMPILE_CACHE=1.' (docs/gotchas.md:28-32).

**llama.cpp — not applicable.** The failure being prevented (a cached compiled graph replayed under a different configuration, dying on a shape assertion) has no analogue. llama.cpp's only graph cache is CUDA graph capture, which is keyed on the split's first node pointer and compares the full node properties of every node on every call; any property change resets warmup_complete and drops to eager execution rather than replaying a stale graph. There is no persistent cache across process launches and nothing an env var could invalidate.

**Equivalent here:** none — no ahead-of-time compiled-graph cache exists

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:4249-4274 (two-call warmup, capture only on unchanged properties)` · `ggml/src/ggml-cuda/ggml-cuda.cu:4265-4268 (any property change resets warmup_complete)` · `ggml/src/ggml-cuda/ggml-cuda.cu:2600-2617 (properties compared = full tensor struct plus every src pointer/ne/nb)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero

### Folding negative group scales into the int4 codes at load time
**Where (theirs):** `patches/marlin-int8-negative-scales.patch:23-53` · `patches/marlin-int8-negative-scales.patch:1-11`

**What it does.** Fixes a silent numerical corruption in vLLM's W4A8-INT8 Marlin path. The kernel requantises fp16 group scales to int16 and then reads them back as *unsigned* int16, so any negative scale becomes a huge positive number and that whole group decodes to garbage. Rather than patching CUDA, this rewrites the weights once at load: for every group whose scale is negative, negate the scale and negate all its int4 codes, which leaves w = s*q unchanged while making every stored scale non-negative.

**Mechanism.** Guarded to the exact configuration where the bug applies: `c.act_type == torch.int8 and c.group_size != -1 and not c.has_g_idx and not c.zero_points` (:30-35) — i.e. int8 activations, grouped, symmetric, no activation reordering. It first normalises the parameter layouts so the arithmetic below is well-defined: `permute_param_layout_(_wq, input_dim=0, output_dim=1, packed_dim=0)` and the same for scales without packed_dim (:37-38). Then `_neg = _ws.data < 0`, short-circuiting on `if bool(_neg.any())` (:39-40). Row expansion from group-space to packed-row-space: eight int4 codes live in each int32, so a group of `group_size` weights spans `group_size // 8` packed rows — `_rows_per_group = c.group_size // 8; _neg_rows = _neg.repeat_interleave(_rows_per_group, dim=0)[: _q.shape[0]]` (:42-43), with the trailing slice guarding a ragged last group. Nibbles are extracted in one vectorised shot with `_shifts = torch.arange(0, 32, 4)` and `_nib = (_q.unsqueeze(-1) >> _shifts) & 0xF` (:44-45). The negation in uint4b8 (bias-8) encoding is `_flip = torch.clamp(16 - _nib, max=15)` (:46), applied only on flipped rows via `torch.where(_neg_rows.unsqueeze(-1), _flip, _nib)` (:47). Reassembly is an explicit 8-iteration OR loop `_out |= _nib[..., _i] << (4 * _i)` (:49-50), then `_wq.data = _out` and `_ws.data = torch.where(_neg, -_ws.data, _ws.data)` (:51-52), with an explicit `del` of the three temporaries (:53) since these are full weight-sized intermediates.

**Why they needed it.** The header names the kernel line and the checkpoint family: 'The int8-activation Marlin kernel (VLLM_MARLIN_INPUT_DTYPE=int8) requantizes the fp16 group scales to int16 and the CUDA kernel then reads them as *unsigned* int16 (marlin_template.h, reinterpret_cast<uint16_t*>). AutoRound symmetric exports store roughly half of all group scales negative, so every such group turns into garbage and the model produces nonsense (while still "benchmarking" fine).' (patches/marlin-int8-negative-scales.patch:3-9). That parenthetical is the important part: the failure is invisible to a throughput benchmark. docs/optimizations.md:63 puts it as 'on this checkpoint it produced garbage while benchmarking beautifully.'

**Their numbers.** 'AutoRound symmetric exports store roughly half of all group scales negative' / '~50% negative scales' (patches/marlin-int8-negative-scales.patch:6-7; docs/optimizations.md:65). Precision cost of the fix: 'numerically identical except that the rare code -8 becomes -7 (one LSB) in flipped groups' (patch :10-11), because uint4b8 code 0 maps to q = -8 whose negation +8 is not representable in 4 bits and is clamped to +7 (:27-28).

**llama.cpp — not applicable.** The bug being fixed is created by a kernel that requantises fp16 group scales to int16 and then reads them back through a uint16_t cast. llama.cpp has no such stage: the loader only places tensors in buffers — resident weight bytes always equal file bytes, and the single load-time transformation (CPU repack) is layout-only with get_alloc_size == nullptr. Scales are read by the dequant kernels in their stored signed type. There is no place for a sign-fold to go and no corruption to fix. Worth noting for its own sake: this is precisely the class of fault this project cares about — a wrong answer that benchmarks beautifully — and llama.cpp is structurally immune to this instance of it.

**Equivalent here:** none — llama.cpp never re-quantises or re-encodes a weight at load

**Evidence (llama.cpp):** `src/llama-model-loader.cpp:1053-1065 and 1177-1221 (loader assigns buffer types only; no type change)` · `ggml/src/ggml-cpu/repack.cpp:4828-4829 (repack is layout-only, allocation stays ggml_nbytes)` · `ggml/src/ggml-cuda/dequantize.cuh and convert.cu (scales read as stored half/float)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero

### The asymmetric-negation clamp and why -8 is the only loss
**Where (theirs):** `patches/marlin-int8-negative-scales.patch:26-28` · `patches/marlin-int8-negative-scales.patch:46`

**What it does.** Documents and handles the one point where the sign-fold is not bit-exact. In the uint4b8 encoding a stored nibble v represents q = v - 8, so negating q means v -> 16 - v. For v = 0 (q = -8) that gives 16, which does not fit in four bits.

**Mechanism.** `_flip = torch.clamp(16 - _nib, max=15)` (:46) — the clamp silently maps the impossible 16 to 15, i.e. q = -8 becomes q = +7 in magnitude terms, losing one least-significant bit for that single code. The patch spells the transform out in a comment: 'uint4b8 code v -> 16 - v, clamped to 15; v == 0 i.e. q == -8 loses one LSB, extremely rare' (:27-28).

**Why they needed it.** Two's-complement asymmetry: the int4 range is [-8, +7], so exactly one representable value has no negation. Calling it out explicitly is what lets a reader accept 'numerically identical' with a bounded exception rather than taking it on faith.

**Their numbers.** One LSB on code -8 only, described as 'extremely rare' (patches/marlin-int8-negative-scales.patch:28).

**llama.cpp — not applicable.** This is a consequence of the previous technique, not an independent one: it documents the single non-bit-exact code in a transform llama.cpp never performs. There is no int4 sign-flip anywhere in this tree to bound the error of.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-model-loader.cpp:1053-1065 (no weight-value transform at load)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero

### `MTP_DRAFT_VOCAB=0` fallback and the weight-loading skip
**Where (theirs):** `patches/qwen3_5-mtp-draft-vocab.patch:30` · `patches/qwen3_5-mtp-draft-vocab.patch:75-83`

**What it does.** Lets the truncated head be switched off at runtime without changing the checkpoint, and makes that switch safe by teaching the weight loader to ignore the `draft_lm_head.*` tensors that are then unused.

**Mechanism.** The env check is part of the construction condition: `if os.path.exists(_ids_path) and os.environ.get('MTP_DRAFT_VOCAB', '1') != '0'` (:30) — so both a missing artifact and an explicit 0 fall back to the shared full lm_head. In `remap_weight_names`, a guard is inserted at the top of the loop: `if 'draft_lm_head' in name and self.model.draft_lm_head is None: continue` (:78-80). Without it, a checkpoint that ships the extra tensors would fail to load whenever the feature is disabled — vLLM errors on weights with no matching parameter.

**Why they needed it.** 'Set MTP_DRAFT_VOCAB=0 to fall back to the shared full lm_head.' (patches/qwen3_5-mtp-draft-vocab.patch:8). The knob table describes the trade it buys: 'set 0 to draft with the full lm_head (more acceptance, slower per draft)' (single-user/README.md:369) — the acceptance/latency trade is genuinely two-sided, which is why the knob exists rather than being hardcoded.

**Their numbers.** Full head vs 40k head at k=4: 85/91 tok/s, 2.85/3.0 tokens per step, 74%/76% acceptance vs the truncated head's numbers (single-user/README.md:250,:252).

**llama.cpp — not applicable.** There is no truncated head to fall back from (see the selector-lattice verdict), so the switch has nothing to switch. The second half — teaching the weight loader to ignore unused tensors — also has no counterpart: llama.cpp's loader assigns buffers to whatever the GGUF contains and does not error on an extra tensor the way vLLM errors on an unmatched parameter. Worth recording the harder constraint behind this: every speculative field in the server's request schema sits inside an #if 0, so speculation is a process-lifetime setting here and any such switch would be a restart, not a knob.

**Equivalent here:** none — and speculation cannot be changed per request at all

**Evidence (llama.cpp):** `tools/server/server-schema.cpp:197-198 and :227 (speculative request fields disabled: 'to keep things simple, we disable speculative parameter adjustments for now')` · `common/speculative.cpp:1237-1244 (the dflash2 draft has no separate head to disable)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero

### Registering the four single-user speed knobs so they enter the compile cache key
**Where (theirs):** `patches/speed-knobs-envs.patch:18-23` · `patches/speed-knobs-envs.patch:31-35` · `patches/speed-knobs-envs.patch:5-9`

**What it does.** Declares `VLLM_MARLIN_TUNE` (bool, default False), `VLLM_MARLIN_TUNE_DIR` (str, default ""), `VLLM_SPEC_DECODE_ATTN` (bool, default False) and `VLLM_DRAFT_TOPK_TOPP` (bool, default True) in vLLM's envs.py. None of the four is *consumed* here — the sole purpose of registration is that vLLM hashes registered env vars into the torch.compile cache key.

**Mechanism.** Two hunks in one file: four annotated fields in the settings block (:20-23) and four lambdas in the environment_variables registry (:32-35), each `os.environ.get(NAME, default) == '1'` for the bools and a raw get for the dir. The patch header notes it is 'Independent of marlin-int8-layer-select.patch (same file, disjoint hunks)' (:9) — both patch envs.py and both must apply cleanly in either order.

**Why they needed it.** A specific, named crash: 'Every registered vLLM env var is part of the torch.compile cache key; VLLM_MARLIN_TUNE in particular changes the Marlin workspace tensor shape that is baked into the compiled graph, so without this a cached graph from a run without the flag fails with "assert_size_stride ... expected size 328==82".' (patches/speed-knobs-envs.patch:5-9). This is the same class of failure as the `KeyError: 'input_global_scale'` from the layer-select patch — a stale compiled graph that assumes the other configuration.

**Their numbers.** The failing assertion is `expected size 328==82` — the Marlin workspace shape differing between tuned and untuned builds (patches/speed-knobs-envs.patch:9).

**llama.cpp — not applicable.** Same reason as the earlier envs.py registration: there is no persistent compiled-graph cache in llama.cpp for an env var to key. The nearest thing, CUDA graph capture, re-validates every node's properties on every call and re-arms only after two consecutive identical calls, so a configuration change produces eager execution and a re-capture, never a stale replay with a wrong workspace shape. llama.cpp does have env kill switches of its own (LLAMA_ATTN_ROT_DISABLE, LLAMA_GRAPH_REUSE_DISABLE, GGML_CUDA_DISABLE_GRAPHS) but they exist as diagnostics, not as cache-key participants.

**Equivalent here:** none

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:4253-4268 (warmup/capture/reset cycle)` · `src/llama-context.cpp:279-285 (LLAMA_GRAPH_REUSE_DISABLE)` · `src/llama-kv-cache.cpp:308-336 (LLAMA_ATTN_ROT_DISABLE)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero

### Uniform patch application contract (`patch -p1 -d site-packages/vllm`)
**Where (theirs):** `patches/sampler-small-topk-fast-softmax.patch:19-20` · `patches/marlin-int8-layer-select.patch:12-15` · `patches/marlin-int8-negative-scales.patch:12-15` · `patches/qwen3_5-mtp-draft-vocab.patch:10-13` · `patches/qwen3_5-embed-quant.patch:8-11` · `patches/speed-knobs-envs.patch:10`

**What it does.** Every patch in the slice carries the same header shape: a prose paragraph explaining the mechanism and the failure it fixes, then the literal command to apply it, then the upstream version it was written against. The paths inside the diffs are relative to the installed package root (`a/envs.py`, `a/v1/sample/...`), not to a vLLM source checkout.

**Mechanism.** All six apply with `patch -p1 -d venv/lib/python3.12/site-packages/vllm < patches/<name>.patch`, run from the repo root. Four of the six state 'Written against vLLM 0.27.1'; qwen3_5-embed-quant.patch adds the maintenance note 'Reapply after upgrades.' (patches/qwen3_5-embed-quant.patch:11). Two patches (marlin-int8-layer-select and speed-knobs-envs) modify the same file, envs.py, and the latter explicitly declares them 'Independent ... (same file, disjoint hunks)' (patches/speed-knobs-envs.patch:9).

**Why they needed it.** These are patches against a pip-installed dependency, not a fork — the target is a venv's site-packages, so nothing survives `pip install -U vllm`. Recording the version and the exact command in the patch file itself is what makes them re-appliable by someone who did not write them; `_check_applied.py` and verify.sh then answer whether they currently are applied.

**Their numbers.** Target: vLLM 0.27.1, python3.12 venv.

**llama.cpp — not applicable.** There is no dependency being patched in place. llama.cpp here is a source tree at a named commit, built into a staged binary directory; the reproducibility record is the commit plus the CMakeCache, both of which the map already cites (GGML_CUDA_FA_ALL_QUANTS=OFF, CMAKE_CUDA_ARCHITECTURES=89). Nothing survives-a-pip-upgrade to defend against.

**Equivalent here:** CMake build from a git checkout; provenance is the commit

**Evidence (llama.cpp):** `C:\AI\llama.cpp\build-dflash2\CMakeCache.txt:64,660 (build configuration of record)` · `C:\AI\llama.cpp @ 1deefcca3, build 10499 (commit as identity)`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** zero
