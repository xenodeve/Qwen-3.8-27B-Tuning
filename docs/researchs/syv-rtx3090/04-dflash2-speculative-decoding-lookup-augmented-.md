# DFlash2 speculative decoding + lookup-augmented block drafting (LABD): patches/dflash2-backport.patch (995 lines), patches/dflash2-lookup-drafting.patch (947 lines)
**60 techniques.** 1942 source lines across 2 files.
Files read: `patches/dflash2-backport.patch` · `patches/dflash2-lookup-drafting.patch`
> **What the reader could not see:** Both files were present and fully readable. Things referenced from inside this slice but NOT contained in it, so I could not verify them: (1) `patches/spec-decode-attn.patch`, required for a verify block longer than 10 query tokens — the lookup patch header says FlashAttention-2 "does not split the KV sequence for multi-query decode; falling back to it doubles the step at 25k context", but the actual attention backend change is in another file. (2) `patches/sampler-small-topk-fast-softmax.patch`, which supplies the `apply_top_k_top_p(..., k_max)` signature that the backport's `top_k_max` feeds; the sort-free small-k algorithm itself is not here. (3) `prepare/fetch_dflash2.py` and `drafter/quant_dflash2.py`, named in the backport header as the requantisation path. (4) `bench/labd_soak.py`, cited as the harness that caught the STICKY nondeterminism. (5) `single-user/start_qwen.sh`, which sets `--no-async-scheduling` and defines `DFLASH_TOKENS`. (6) The DFlash1 base speculator `v1/worker/gpu/spec_decode/dflash/speculator.py` is only visible through its diff hunks, so `_run_model`, `draft_max_seq_len`, `seeds`, `use_fp64_gumbel`, `input_buffers` and the prepare-inputs kernel's unpatched body were not readable. Also absent: any statement of the DFlash2 checkpoint's actual `selector_rank`, `conv_kernel_size`, `conv_group_size` or `input_embedding_scale` values (they are read from `dflash_config` at runtime); the only concrete block figure stated is `block_size` 8 = 1 anchor + 7 mask tokens. No end-to-end tok/s figure for DFlash2-without-lookup appears anywhere in the slice.

---

## EXISTS, NEVER SET — 6

### FlashInfer radix top-k with a latching fallback to torch.topk
**Where (theirs):** `patches/dflash2-backport.patch:394-416` · `patches/dflash2-backport.patch:419-432`

**What it does.** Resolves a vocabulary-wide top-k implementation once (`@cache`) and wraps every call so that a runtime failure permanently demotes the process to `torch.topk` rather than retrying. Three separate opt-outs: non-CUDA platform, `has_flashinfer()` false, and the env var `VLLM_DFLASH2_TORCH_TOPK=1`.

**Mechanism.** `_flashinfer_topk()` is `@cache`-decorated and returns `flashinfer.top_k` or None. `_topk(scores, k)` consults a module-level `_TOPK_BROKEN` latch first; on success it calls `impl(scores, k, sorted=True, deterministic=True)`; on any exception it sets `_TOPK_BROKEN = True`, emits `logger.warning_once`, and returns `torch.topk(scores, k, dim=-1)`. Non-CUDA tensors also route to torch.topk.

**Why they needed it.** The docstring states the stake: "This top-k spans the vocabulary and is the selector's largest single cost, where the radix kernel is about twice torch.topk." The fallback exists because "flashinfer.top_k falls back to torch.topk if its JIT build fails" — the failure modes named in the except comment are "JIT build failure, unsupported arch, ...". The latch avoids paying a failing JIT attempt every decode step.

**Their numbers.** FlashInfer's radix top-k is "about twice torch.topk" — i.e. torch.topk runs the selector's dominant cost "at roughly half the speed". Condition: a top-k spanning the full vocabulary, per (request, step) draft row.

**llama.cpp — EXISTS, NEVER SET.** The specific technique — resolve a fast top-k once, latch to a fallback on failure — has no analogue: llama.cpp has exactly one host top-k and one device top-k, no JIT to fail. But judging this against the map turned up the adjacent unused flag, which is the actually valuable answer. -bs is off by default (the opposite of the DRAFT path's default). Only a contiguous PREFIX of the chain can leave the host, and the samplers with no device implementation are typical, xtc, mirostat, top_n_sigma, dry, adaptive_p — all of which are DISABLED in their profile, and a disabled sampler becomes an `empty` stub that still implements backend_init. So the entire live chain (logit_bias, top_k, top_p, min_p, temp_ext, dist) is device-capable and the prefix should not break. CUB is present in this build, so the ne0<=1024 argsort cap does not bite on a 151k vocab. Caveats are real: grammar or reasoning-budget force it off with a warning (sampling.cpp:421-431), and n_probs>0 with pre-sampling probs silently skips it per request.

**Equivalent here:** -bs / --backend-sampling (GPU-side sampling for the MAIN path)

**Evidence (llama.cpp):** `common/arg.cpp:2295-2301` · `common/common.h:295` · `src/llama-sampler.cpp:746-765` · `src/llama-sampler.cpp:505-520` · `tools/server/server-context.cpp:1732-1744` · `common/sampling.cpp:421-431`

**Effort:** one-flag · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** unknown, but this is a one-flag experiment they have never run. -bs moves the whole default sampler chain onto the GPU and removes a 151k-entry logits readback per accepted token. Refuses to engage if the coding agent sends a grammar or json_schema, or asks for logprobs — and it self-disables with a warning rather than failing, so the log must be checked.

### `last_num_emitted`: what the previous step actually produced
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:168-176`

**What it does.** Records, per request, how many tokens the step that just ran emitted, by differencing the sampling slots it was given against the ones it threw away. This is the saturation signal the adaptive controller uses.

**Mechanism.** `self.last_num_emitted = (num_sampled - num_rejected) if num_sampled is not None else None`, computed in `prepare_inputs`. The comment defines both terms: "num_sampled counts the sampling slots it was given (bonus + drafts), num_rejected how many of those were thrown away, so the difference is the tokens it emitted."

**Why they needed it.** The controller needs evidence that a copy is *running*, not merely that a match exists; the fuse kernel's own comment names the arithmetic: "the step that just finished accepted every token it was given (prev_acc = 1 + accepted, and a short block can produce at most 1 + draft_block)".

**llama.cpp — EXISTS, NEVER SET.** vLLM had to derive saturation by differencing sampling slots against rejections. llama.cpp is handed the number directly: common_speculative_accept is called with accepted.size()-1 and the per-slot counters are updated on the same path. So the input to an adaptive controller is already computed, already per-slot, already host-side — it is simply never consumed by any decision. That is the definition of exists-but-unused. Caveat from the map worth carrying: on a checkpoint-restore round the counting returns early and the replay subtracts one to avoid double counting, so a naive consumer must use the same accessors rather than re-deriving.

**Equivalent here:** slot.stats.n_draft_tokens / n_draft_accepted / n_draft_verif_steps, and common_speculative_accept's accepted.size()-1

**Evidence (llama.cpp):** `tools/server/server-context.cpp:2966` · `tools/server/server-context.cpp:3877` · `tools/server/server-context.cpp:3883-3903` · `tools/server/server-context.cpp:634-637`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The signal already exists per slot per step and today feeds only the log line and Prometheus. Wiring it into dp.n_max is the small patch of technique 43. Value of reading it as-is: the `draft acceptance = X (a accepted / g generated), mean len = Y` line is printed on every completion and is the cheapest existing instrument for this question.

### `set_draft_tokens(num_draft=...)`: truncate the proposal handed to the scheduler
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:859-879`

**What it does.** Extends the draft-tokens handler so the number of tokens put up for verification can be fewer than the speculator produced, slicing the tensor and reporting the shorter count to the scheduler.

**Mechanism.** New optional `num_draft: int | None = None` parameter. `self.num_draft_tokens = draft_tokens.shape[1] if num_draft is None else num_draft`, and `if num_draft is not None: draft_tokens = draft_tokens[:, :num_draft]` before the structured-output path. Backwards compatible: `None` reproduces the old behaviour exactly.

**Why they needed it.** Docstring: "The DFlash2 lookup uses it to pay for a long verify block only while the request is reproducing its context." This is the plumbing that makes the controller's decision actually reach the scheduler.

**llama.cpp — EXISTS, NEVER SET.** I read the truncation: after each impl returns, if dp.n_max > 0 and the result is longer, it is resized with SPC_DBG 'truncating draft to %d tokens'. That is exactly the 'put up fewer tokens than the speculator produced' capability vLLM had to add a parameter for, and it is backwards-compatible in the same way (n_max <= 0 means no truncation). No patch needed to have it; a patch is needed only to have something interesting decide the number. Pairs with 43 and 48.

**Equivalent here:** the dp.n_max truncation in common_speculative_draft

**Evidence (llama.cpp):** `common/speculative.cpp:2728-2732` · `tools/server/server-context.cpp:2936-2946`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** The plumbing is complete. It is driven only by the context clamp today, never by a policy.

### NMAX=12 chosen against 32 so recency beats length
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:283-286`

**What it does.** Caps the suffix match length at 12 tokens rather than the 32 the `suffix_lookup` signature defaults to, deliberately limiting how long a match can score.

**Mechanism.** `self._lookup_nmax = int(os.environ.get("VLLM_DFLASH2_LOOKUP_NMAX", "12"))`, passed as `NMAX` to the kernel where it bounds the extension loop `for j in range(NMIN, NMAX)`. Because the score packs length in the high bits, a larger NMAX lets an older-but-longer match outrank a newer shorter one.

**Why they needed it.** Verbatim: "12, not 32: the kernel picks the longest match and breaks ties by recency, and a longer cap makes it prefer an older long match over a newer short one. On quote-and-explain work the newer one is the better predictor (3.21 vs 2.69 tokens per step), and copies match well past 12 either way." The last clause is the reason the cap costs nothing on the case it might have been expected to hurt.

**Their numbers.** 3.21 tokens per step at NMAX=12 vs 2.69 at NMAX=32, on quote-and-explain work.

**llama.cpp — EXISTS, NEVER SET.** The mechanism does not transfer — llama.cpp has no longest-match competition for a length cap to bias — but the empirical claim maps onto a knob they have never touched. ngram-mod's n_match IS the analogue of NMAX in the sense that matters: how much context has to agree before a successor is proposed. A longer window means fewer, more specific matches. The vLLM result says shorter wins on the workload class a coding agent spends most of its time in. Note this changes the hash key, so the table must warm again; measure across a full corpus, not a few requests.

**Equivalent here:** --spec-ngram-mod-n-match (default 24), and --spec-ngram-map-k4v-size-n / -size-m / -min-hits

**Evidence (llama.cpp):** `common/common.h:351-362` · `common/arg.cpp:4163-4192` · `common/arg.cpp:4194-4285` · `common/speculative.cpp:1924-1927`

**Effort:** one-flag · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** unknown, and a cheap one-flag A/B. ngram-mod's window is 24 tokens; the vLLM measurement says a shorter, more recent match predicts better on quote-and-explain work (3.21 vs 2.69 tokens/step at 12 vs 32). llama.cpp warns below 16 ('poor quality is possible') but accepts 1..1024, so 16 and 12 are both reachable. ngram-map-k4v's size_n=12 / size_m=48 / min_hits=1 is an entirely unused speculator with three tunables.

### NSTRONG=NMIN and AGREE=0: take any qualifying match in the head
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:287-292`

**What it does.** Ships with the head gate effectively collapsed — `nstrong == nmin == 6` and `agree_min == 0` means `take_head` reduces to `match_len >= 6`, so drafter agreement is never actually required in the shipped configuration, even though the mechanism for requiring it is present and tested.

**Mechanism.** Defaults `VLLM_DFLASH2_LOOKUP_NSTRONG=6` equal to `VLLM_DFLASH2_LOOKUP_NMIN=6`, and `VLLM_DFLASH2_LOOKUP_AGREE=0`. In `take_head = (match_len >= nstrong) | ((match_len >= nmin) & (agree >= agree_min))` the first disjunct then subsumes the second.

**Why they needed it.** Verbatim: "NSTRONG = NMIN and AGREE = 0 means 'take any match of NMIN or more' for the positions the drafter also proposed. Measured better at C1 (3.33 vs 3.27 tokens per step) than requiring drafter agreement for medium matches." The mechanism is kept because the header still argues for it in principle; the measurement is what set the defaults.

**Their numbers.** 3.33 vs 3.27 tokens per step at C1, taking any match of NMIN+ versus requiring drafter agreement for medium-length matches.

**llama.cpp — EXISTS, NEVER SET.** vLLM's threshold study concluded that requiring drafter agreement for medium matches cost acceptance (3.33 vs 3.27 tokens/step), so they ship with the head gate collapsed to a bare length test. llama.cpp's equivalent gate is n_min, and it is set an order of magnitude stricter than anything in that study. The counter-argument is real and must be measured rather than assumed: more short drafts means more verify batches that mostly reject, and each verify batch on this box runs MMA_F16 with a full-cache F16 expansion (technique 47), so a failed draft is not free. Pair within a round and alternate the order — the acceptance-rate number alone will move in the opposite direction to throughput here, which is precisely the shape of instrument fault this repo keeps cataloguing.

**Equivalent here:** --spec-ngram-mod-n-min (default 48, range 0..1024)

**Evidence (llama.cpp):** `common/speculative.cpp:1992-2004` · `common/common.h:351-356` · `common/arg.cpp:4163-4192` · `common/speculative.cpp:2728-2732`

**Effort:** one-flag · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** unknown, and this is the other one-flag experiment worth running. I read draft_one: it walks up to n_max successors and the moment the hash returns EMPTY it either truncates (if i >= n_min) or CLEARS THE WHOLE DRAFT. With the defaults that means 48 consecutive hash hits or nothing — a 47-token match produces zero tokens. --spec-ngram-mod-n-min 8 (or 4) would let short matches through, which is exactly the 'take any match of NMIN or more' policy vLLM measured as better.

### Lookup hit counter
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:312` · `patches/dflash2-lookup-drafting.patch:721`

**What it does.** Maintains a single scalar device counter of how many requests took a head-level lookup override, for offline instrumentation.

**Mechanism.** `self._lookup_hits = torch.zeros((), dtype=torch.int64, device=device)`; the fuse kernel does `tl.atomic_add(hits_ptr, take_head.to(tl.int64))`. A zero-dim tensor, so it is never read on the hot path.

**Why they needed it.** Not stated. Note it counts `take_head` (head overrides), not tail fills or long-block entries, so it is not the same population the controller decides on.

**llama.cpp — EXISTS, NEVER SET.** vLLM's counter is one atomic scalar counting head-level overrides, and its own note admits it does not count the population the controller actually decides on. llama.cpp's instrumentation is richer and already wired; the only thing missing is the log level. Credit goes only to impl_last — the impl that actually produced the draft — while every other impl gets accept(is_other=true) with its counters untouched, so the attribution is clean for a chained profile. One caution before trusting the per-position histogram: it is sized common_speculative_n_max, which is 64 for ngram-mod, while dp.n_max truncation near the context edge shortens drafts without shortening the array.

**Equivalent here:** common_speculative_print_stats (per-impl counters, LOG_TRC only) + the Prometheus per-position histogram

**Evidence (llama.cpp):** `common/speculative.cpp:139-161` · `common/speculative.cpp:2829-2872` · `common/speculative.cpp:2796-2801` · `tools/server/server-context.cpp:3899` · `common/arg.cpp:3533-3539`

**Effort:** one-flag · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Strictly better than what vLLM built, and switched off. llama.cpp keeps per-impl n_gen_drafts, n_acc_drafts, n_gen_tokens, n_acc_tokens, a per-POSITION acceptance array, and t_begin/t_draft/t_accept timings — with gen_perf hardcoded true, so the timing is always being collected and thrown away. It prints only at LOG_TRC. With their chained draft-dflash,ngram-mod profile they currently cannot say which impl served which fraction of steps; -lv 4 answers that for free. Separately --metrics exposes ..._num_accepted_tokens_per_pos_total, which shows where in a 64-token ngram-mod draft acceptance dies.

## absent, has a seam — 8

### NaN guards before the rejection sampler's two argmax reductions
**Where (theirs):** `patches/dflash2-backport.patch:303-312` · `patches/dflash2-backport.patch:313-322`

**What it does.** In `rejection_sampler_utils.py`, maps NaN entries of the per-block local maxima to -inf immediately before `tl.argmax`, in both the target-argmax path and the resampled-argmax path.

**Mechanism.** `local_max = tl.where(local_max != local_max, float("-inf"), local_max)` before `max_block_idx = tl.argmax(local_max, axis=0)`; and `resampled_local_max = tl.where(resampled_local_max != resampled_local_max, float("-inf"), resampled_local_max)` before `resampled_max_block_idx = tl.argmax(...)`. The `x != x` idiom is the NaN test.

**Why they needed it.** Header: "NaN guards before the rejection sampler's argmax", backported from vLLM main. `tl.argmax` over a block containing NaN selects arbitrarily, which silently returns a wrong token rather than failing.

**llama.cpp — absent, has a seam.** I grepped common/sampling.cpp: the only finiteness checks are on penalty parameters, not on logits. The greedy accept loop compares sampled ids and the temp<=0 path is an argmax mask over cur_p; neither maps NaN to -inf first. Whether ggml-cuda's top-k/argmax degrade gracefully on a NaN row I did not chase into the kernels, which is why this is idea-only rather than a confident defect claim. If you ever see a speculative run produce a token that no sampler could have chosen, this is the first place to instrument. Marked idea-only because the Triton specifics do not move and the underlying defect is unproven here.

**Equivalent here:** none; the seam is common_sampler_sample_and_accept_n and llama_sampler_init_temp's greedy rewrite

**Evidence (llama.cpp):** `common/sampling.cpp:196-204` · `common/sampling.cpp:692-720` · `src/llama-sampler.cpp:270-286` · `src/llama-sampler.cpp:193-215`

**Effort:** small-patch · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** unknown. Only matters if a 2-bit UD-IQ2_XXS target with q4_0 KV ever emits a non-finite logit. This project has not recorded one.

### Asymmetric head/tail acceptance thresholds in the fuse kernel
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:637-708`

**What it does.** Fuses the lookup proposal with the drafter's rather than substituting for it, using *different* thresholds for the two halves of the verify block. Head positions (which the drafter also proposed) require a strong match or drafter agreement; tail positions (which nobody proposed) require only a weak match plus consistency with the head.

**Mechanism.** `take_head = (match_len >= nstrong) | ((match_len >= nmin) & (agree >= agree_min))`. `tail = idx >= draft_block`; `take_tail = (match_len >= nmin_tail) & (take_head | (agree >= draft_block))`. `from_lookup = tl.where(tail, take_tail, take_head) & (idx < valid) & kmask`, and the proposal is written back with `tl.store(draft_tokens_ptr + req * draft_stride + idx, looked, mask=from_lookup)`. The `use` mask stored for the point-mass rewrite is `(from_lookup | tail) & kmask` — tail positions always get a point mass "since nothing else wrote a draft distribution for them".

**Why they needed it.** Verbatim: "The two halves of the block are not the same bet, so they do not share a threshold. Head ...: the drafter already proposed these, so taking the lookup's token *replaces* a considered guess and a coincidental match is a loss. ... Tail ...: nobody proposed these, so anything the match supplies is free and the threshold drops to `nmin_tail`." And from the patch header: "Two independent sources agreeing is a cheap confidence signal -- the drafter looked at the hidden state, the lookup looked at the text." Also: "All-or-nothing overriding on a 6-token match loses acceptance on prose."

**llama.cpp — absent, has a seam.** I read the loop. common_speculative_draft iterates impls in the hardcoded priority order, and for each sequence sets dp.drafting = false the moment an impl returns a non-empty result; when no sequence is still drafting it breaks out. Exactly one impl's flat token list survives per sequence per step and is recorded in impl_last. So their draft-dflash,ngram-mod pair is 'ngram-mod first, dflash only when ngram-mod returned nothing' — a coarse fallback, never a fusion, and the head/tail distinction has nowhere to live. The seam for a fusion patch is that loop plus common_speculative_draft_params (dp.result is a single llama_tokens the impls write into). A fused version would need dp.result to carry provenance per position and the accept side to keep dists consistent (technique 41). Large patch, but the seam is nameable and local to common/speculative.cpp.

**Equivalent here:** none — chaining is fallback, not fusion

**Evidence (llama.cpp):** `common/speculative.cpp:2710-2756` · `common/speculative.h:49-54` · `common/speculative.cpp:2542-2552`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown. This is the heart of LABD and it is genuinely absent. Before building it, note the cheaper adjacent experiment: --spec-ngram-mod-n-min defaults to 48 (technique 55), so today a 47-token match yields nothing at all.

### Leading-agreement computation clamped by `valid`
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:697-701`

**What it does.** Computes how many leading tokens the two proposals agree on, branchlessly, and clamps the result by how many tokens the match actually supplied — so a match that ran out of history cannot claim full agreement by default.

**Mechanism.** `disagree = (drafted != looked) & (idx < valid) & (idx < draft_block) & kmask`; `agree = tl.minimum(tl.min(tl.where(disagree, idx, draft_block)), valid)`. The `tl.where(disagree, idx, draft_block)` + `tl.min` idiom finds the first disagreeing index without a loop; `draft_block` is the identity element for the min.

**Why they needed it.** Verbatim: "Positions the match did not supply are not agreement: without this a match with one or two tokens left scores a full-block agreement and walks through the gate." Also documented at the load: "draft_tokens is int64 ...; the history the lookup reads is int32, so the proposal is widened before it is written back" (`looked = looked.to(tl.int64)`).

**llama.cpp — absent, has a seam.** An agreement measure between two proposals requires two proposals to exist simultaneously, which the fallback loop prevents (see 37). If that patch is ever written, this hunk is the part most worth copying verbatim: the failure it names — a match with one or two tokens left scoring full-block agreement and walking through the gate — is exactly the 'plausible number instead of a failure' shape this repo's north star warns about, and it is not obvious from first principles.

**Equivalent here:** none

**Evidence (llama.cpp):** `common/speculative.cpp:2710-2756`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown, and strictly downstream of technique 37 — worthless on its own.

### Point-mass draft-logit rewrite that preserves rejection-sampling exactness
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:724-761` · `patches/dflash2-lookup-drafting.patch:48-54`

**What it does.** Rewrites one (request, step) row of the sparse draft-logit cache to a point mass on whatever token is now in `draft_tokens`, so the overridden proposal has a legal proposal distribution q. This is what makes the lookup override lossless for sampling requests.

**Mechanism.** `_point_mass_draft_logits_kernel`, grid `(num_reqs * num_speculative_steps,)`. Early-returns on `req_state < 0` and on `use_ptr[req * num_steps + step] == 0`. Erases the previously cached candidates (`tl.store(logits_base + old, -inf, mask=mask)`), stores `0.0` at the proposed token's vocab offset, and — the invariant-preserving step — fills *every* one of the `top_k` slots of `cached_candidate_ids` with that same token id, so the next step's erase pass covers the row exactly. The token is read back out of `draft_tokens` rather than passed in.

**Why they needed it.** Verbatim from the docstring: "keeping the cache's invariant that every finite entry of a row is listed in `cached_candidate_ids` (so the next step erases it). The token is read back from draft_tokens, so q always describes the proposal that is actually verified." And the losslessness argument from the header: "every position the lookup filled gets a point mass on the proposed token, which is a legal proposal distribution for vLLM's rejection sampler: the acceptance probability becomes p(x) and the residual (p - q)+ is computed from the same buffer, so the output distribution is unchanged." Greedy requests "never read q at all".

**llama.cpp — absent, has a seam.** llama.cpp's selection between the greedy and residual accept rules is `spec_dists.size() == spec_draft.size()`. The only producer of dists is the DFlash2 selector. So llama.cpp's answer to 'what q does an overridden token have?' is currently 'none, fall back to greedy', which is a coarser version of the losslessness problem vLLM solved with a point mass. The seam is exactly the same shape: push a single-entry common_speculative_token_dist (id, prob 1.0) into dp.dists for any position an override filled. Small patch given the fusion of technique 37; meaningless without it.

**Equivalent here:** dp.dists left empty => the greedy accept rule is used even at temperature 1.0

**Evidence (llama.cpp):** `common/speculative.cpp:1239-1258` · `tools/server/server-context.cpp:3825-3831` · `common/sampling.cpp:722-793` · `common/sampling.cpp:692-720`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown; only bites at temp > 0. Note the standing behaviour it would fix is already live: with ngram-mod (or any non-DFlash2 impl) dp.dists stays empty, so llama.cpp uses greedy prefix-match acceptance even at temperature 1.0 — the output distribution is NOT the target's, today, in their profile.

### Candidate cache spans the whole verify block while the drafter fills only its head
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:211-226` · `patches/dflash2-lookup-drafting.patch:257-268`

**What it does.** Splits the `_cache_draft_logits_kernel`'s single `num_steps` parameter into two: `num_steps` (how many rows the drafter writes) and `cache_steps` (how many rows the cache holds), so the drafter's writes land at the right stride inside a cache sized for the longer verify block.

**Mechanism.** `cache_base = (req_state * cache_steps + step) * top_k` replacing `(req_state * num_steps + step) * top_k`; called with `num_steps=self.draft_block, cache_steps=self.num_speculative_steps`. `_cached_candidate_ids` is reallocated as `(max_num_reqs, num_speculative_steps, selector_top_k)` rather than mirroring `_selector_scores.shape` (which is now `[max_num_reqs, draft_block, selector_top_k]`).

**Why they needed it.** Comment: "The candidate cache spans the whole verify block, of which the drafter fills the first num_steps rows (dflash2/lookup.py fills the rest)." And on the allocation: "the point-mass rewrite keeps the same erase invariant for the positions the lookup fills beyond the drafter's own block."

**llama.cpp — absent, has a seam.** llama.cpp's dists vector is sized by what the impl pushed, not preallocated to a verify-block length, so there is no stride mismatch to fix — which also means there is no place for a longer verify block's tail rows to live. If the decoupling of technique 31 is built, dp.dists and dp.result would need to be sized by the verify block with the drafter filling a prefix, which is the same change vLLM made. The seam is common_speculative_draft_params in common/speculative.h.

**Equivalent here:** none

**Evidence (llama.cpp):** `common/speculative.h:36-43` · `common/speculative.cpp:1239-1258`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown; downstream of 31 and 37.

### Two qualifying steps in a row required to enter the long block
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:397-403`

**What it does.** The long block is only taken when the qualifying condition held on two consecutive steps, tracked with a single `_prev_want` boolean.

**Mechanism.** `if want and self._prev_want: long_block = True; ...` with `self._prev_want = want` at the end of the function.

**Why they needed it.** Verbatim: "Two qualifying steps in a row, not one: a single saturated step happens in the middle of ordinary prose (a quoted phrase, a repeated list marker) and the long block it buys is then wasted. Waiting for the second one costs the first step of a copy and removes the loss on quote-and-explain work."

**llama.cpp — absent, has a seam.** Pure control-policy content with no llama.cpp mechanism to attach to yet. The empirical claim — a single saturated step happens in ordinary prose and the long block it buys is wasted — is the kind of thing this project would have to re-measure on its own workload anyway, since it is workload-dependent rather than stack-dependent. Marked idea-only because the seam it needs (43) does not yet carry a controller.

**Equivalent here:** none; would live in the same dp.n_max computation as technique 43

**Evidence (llama.cpp):** `tools/server/server-context.cpp:2936-2946`

**Effort:** small-patch · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** unknown. Cheap to add if an adaptive controller is ever written; meaningless before then.

### STICKY hysteresis: hold the long block for 3 steps after the flag drops
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:331-352` · `patches/dflash2-lookup-drafting.patch:402-409`

**What it does.** Once a run has earned the long block, a counter keeps it for `VLLM_DFLASH2_LOOKUP_STICKY` (default 3) further steps even when the flags say no — because re-entry costs two steps and the flag drops out for reasons unrelated to the copy ending. Restricted to `num_reqs == 1` for determinism, not caution.

**Mechanism.** `self._sticky = self._lookup_sticky if num_reqs == 1 else 0` on a real entry; `elif self._sticky > 0 and num_reqs == 1: long_block, self._sticky = True, self._sticky - 1`; else both reset to False/0.

**Why they needed it.** The asymmetry is stated: "Entering the long block takes two qualifying steps in a row and leaving it took one, so a single step where the flag dropped out mid-copy cost three. It drops out for reasons that have nothing to do with the copy ending -- a line the lookup cannot match, or a flag copy that had not landed yet." A rejected alternative is recorded: "Gating the hold on 'the request is still emitting a full block a step' -- which should distinguish a late flag from a finished copy -- removes the whole effect (13.92 again): by the time the flag drops the step it describes was not saturated either, so the two are not independent evidence." And the batch restriction: "This counter is one number for the whole batch, and unlike the entry condition it keeps the long block on through steps where the flags say no -- so with several requests, which block length a copying request gets starts to depend on when the others arrived. Different block length, different rounding, different greedy text: bench/labd_soak.py caught a verbatim copy coming out differently in two rounds of an otherwise identical 4-way batch, and reproducibly did not with STICKY=0."

**Their numbers.** Without STICKY, one prompt measured 14.97 tokens per step on the first run and 13.76, 13.92, 13.92, 13.92, 13.92 on consecutive later runs of the same server. With STICKY=3 it is 15.21 every time. The rejected saturation-gated variant returned to 13.92.

**llama.cpp — absent, has a seam.** Same status as 45 — policy for a controller that does not exist yet. Worth recording one detail: vLLM had to restrict STICKY to num_reqs == 1 because a batch-wide hold made a copying request's block length depend on when other requests arrived, and they caught a verbatim copy coming out differently across otherwise identical runs. On a single-slot llama.cpp server that failure mode is unreachable, so the restriction would carry no cost. Also note their rejected variant (gating the hold on saturation) is a documented negative — do not re-derive it.

**Equivalent here:** none; same seam as 43

**Evidence (llama.cpp):** `tools/server/server-context.cpp:2936-2946` · `tools/server/server-context.cpp:1224-1226`

**Effort:** small-patch · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** unknown. The determinism hazard that forced vLLM to restrict this to batch 1 does not apply here: -np 1 means one slot, so a batch-wide counter and a per-request counter are the same thing.

### Lookup applied after the selector, inside the same captured graph
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:457-525`

**What it does.** `_apply_lookup` runs at the tail of `_generate_draft`, after the selector walk has written `draft_tokens[:, :draft_block]`, so the fuse kernel sees the drafter's actual proposal and can compare against it. All three kernels (suffix lookup, fuse, point-mass rewrite) are launched from inside the CUDA-graph-replayed region.

**Mechanism.** Order: `_sample_path` → `_cache_draft_logits` → `draft_tokens[:num_reqs, :draft_block].copy_(self._selector_tokens[:num_reqs])` → `_apply_lookup(num_reqs)`. Inside: `suffix_lookup(...)` with `k=self.num_speculative_steps` (the full verify block, not the drafter's), then `fuse_draft(...)` with `draft_block=self.draft_block`, then `_point_mass_draft_logits_kernel[(num_reqs * self.num_speculative_steps,)]`. All output buffers are preallocated in `__init__` and sliced (`self._lookup_tokens[:num_reqs]` etc.) so no allocation happens during capture.

**Why they needed it.** The head/tail fusion logic needs both proposals present; and being inside the captured graph is exactly why the long-block decision had to be lifted out to `next_num_draft_tokens` on the host.

**llama.cpp — absent, has a seam.** vLLM's ordering choice (lookup after the selector, so the fuse kernel can see the drafter's actual proposal) presupposes fusion, which llama.cpp does not have — that part is the large patch of technique 37. But the narrower ordering question is a genuine absent-but-possible with a tiny seam: the list at speculative.cpp:2542-2552, with a static_assert(COUNT == 11) immediately above it to keep in sync. Watch one interaction when testing: swapping the order changes which impl gets credited in impl_last and therefore what the per-impl LOG_TRC statistics attribute, so the before/after numbers are not comparing the same accounting unless read from the aggregate.

**Equivalent here:** the hardcoded speculator priority list — ngram-* always outranks every model-based type

**Evidence (llama.cpp):** `common/speculative.cpp:2538-2552` · `common/speculative.cpp:2725-2755` · `common/speculative.cpp:2343-2349`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** A small patch with a directly testable question behind it. I read the priority list: ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache, THEN draft-simple, eagle3, mtp, dflash, dspark. Command-line order is discarded (the list is rebuilt from a bitmask). So their measured draft-dflash,ngram-mod pair at +48.5% is running ngram-mod first with dflash as the fallback, and there is no flag that reverses it. Given dflash alone measured +34.7% over ngram-mod alone, 'dflash first, ngram-mod as fallback' is an obvious unmeasured configuration and reaching it means reordering ten lines.

## partial — 6

### Rank-based top-k/top-p truncation of the 16-candidate draft proposal, in registers
**Where (theirs):** `patches/dflash2-backport.patch:795-810` · `patches/dflash2-backport.patch:901-908` · `patches/dflash2-backport.patch:920-939`

**What it does.** Applies the request's own top-k and top-p to the drafter's proposal distribution over its `selector_top_k` candidates before sampling from it, and caches the truncated, renormalised distribution as q. Mass the drafter would have put outside the target's support is a guaranteed rejection, so removing it strictly improves acceptance without changing the output distribution.

**Mechanism.** Under `TRUNCATE`, softmax over the temperature-scaled candidate scores in registers: `mx = tl.max(t_scores)`, `pr = tl.exp(t_scores - mx)`, `pr = pr / tl.sum(pr)`. Then a pairwise comparison matrix `gt = t_scores[None, :] > t_scores[:, None]` yields both the rank (`rank = tl.sum(gt.to(tl.int32), axis=1)`) and the cumulative mass above each candidate (`mass_before = tl.sum(tl.where(gt, pr[None, :], 0.0), axis=1)`) without any sort. `keep = mask & (rank < req_top_k) & (mass_before < req_top_p)`; non-kept entries become `-inf` in `t_scores`, which is then both what is sampled from and what is cached as `realized`.

**Why they needed it.** Verbatim: "Apply the request's top-k/top-p to the proposal distribution over the candidates (the verify truncates the target the same way, so mass the draft puts outside that support is a guaranteed rejection). The truncated, renormalized q is what gets cached, so the verify stays lossless. Rank-based: 16 candidates, all in registers." The header calls it "the DFlash2 analogue of the MTP draft truncation; lossless".

**llama.cpp — partial.** llama.cpp's draft-side sampler is fixed to {TOP_K} with top_k = selector_top_k for DFlash2, and the block that would have made it configurable is commented out at speculative.cpp:209-224 — that comment block is the named seam. The request's top_k/top_p are not visible to the speculator at all. The plumbing to make them visible already exists and already carries two sampling fields (see technique 28), so this is genuinely a small patch: add top_k/top_p to common_speculative_draft_params, apply them to dist before sampling and before storing. It is lossless for the same reason vLLM argues — mass outside the target's support is a guaranteed rejection under the residual rule. But their coding agent is likely near-greedy, and at temp <= 0 the DFlash2 branch takes the argmax path and never builds a dist, so this buys nothing there. The p_min flag is the cheap thing to try first.

**Equivalent here:** --spec-draft-p-min (a confidence early-stop, not a support truncation); the draft sampler itself is hardcoded

**Evidence (llama.cpp):** `common/speculative.cpp:1249-1252` · `common/arg.cpp:4101-4107` · `common/common.h:329` · `common/speculative.cpp:209-224` · `common/speculative.cpp:1001-1008` · `tools/server/server-context.cpp:2936-2946`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown, and probably nil at temp 0. Two separate things here: (a) --spec-draft-p-min is a real unused one-flag knob that on the DFlash2 path breaks the block early when the selector is unconfident — I read the `if (dist.probs[predecessor] < params.p_min) break;` line; it defaults to 0.00 = disabled. (b) truncating q to the request's top-k/top-p is absent and only pays at temp > 0.

### Sampling-state buffers handed from the model runner to the speculator
**Where (theirs):** `patches/dflash2-backport.patch:182-193` · `patches/dflash2-backport.patch:910-912`

**What it does.** After constructing the speculator, the model runner duck-types for `set_sampling_states` and passes it the sampler's `sampling_states`, from which the speculator keeps only the GPU-side top_p and top_k tensors.

**Mechanism.** `if self.speculator is not None and hasattr(self.speculator, "set_sampling_states"): self.speculator.set_sampling_states(self.sampler.sampling_states)`. The speculator stores `self._req_top_p = sampling_states.top_p.gpu` and `self._req_top_k = sampling_states.top_k.gpu` — `[max_num_reqs]` buffers indexed by request-state index, the same indexing `sample_idx_mapping` produces.

**Why they needed it.** Comment: "DFlash2 truncates its proposal to the request's top-k/top-p." The `hasattr` guard keeps the hook inert for every other speculator, so the hunk does not need a type check.

**llama.cpp — partial.** I read the struct initialisation. Per slot per step the server fills drafting, n_max, n_past, id_last, prompt, result, dists, temperature and seed — two of those (temperature, seed) are exactly the 'hand the request's sampling state to the speculator' hook vLLM had to add via duck-typing. Adding top_k and top_p to that struct and to common_speculative_draft_params is mechanical. Marked partial rather than already-have-it because the two fields that technique 26 needs are the two that are missing.

**Equivalent here:** common_speculative_get_draft_params(spec, slot.id) — already carries .temperature and .seed

**Evidence (llama.cpp):** `tools/server/server-context.cpp:2936-2946` · `common/speculative.h:56-58`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** This is the seam that makes technique 26 cheap. Value of the seam itself: it already exists and already works.

### Decouple the drafter's own block (`draft_block`) from the verify block (`num_speculative_steps`)
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:92-126` · `patches/dflash2-lookup-drafting.patch:17-24`

**What it does.** Introduces `DFlashSpeculator.draft_block` — how many tokens the draft model itself proposes, taken from the checkpoint's `dflash_config.block_size - 1` — and keeps `num_speculative_steps` as the number of positions the target verifies. Every buffer and kernel argument is re-audited into one camp or the other. The extra verify positions are filled by the lookup at no cost to the drafter.

**Mechanism.** `trained_block = int((hf_config.dflash_config or {}).get("block_size", 0)) - 1`. `self.draft_block = self.num_speculative_steps` by default; only if `VLLM_DFLASH2_LOOKUP=1` and `0 < trained_block < num_speculative_steps` is it lowered to `trained_block`. Consequent changes: `num_query_per_req = 1 + self.draft_block`; `max_num_sampled_tokens = max_num_reqs * draft_block`; `sample_col = arange(draft_block).repeat(max_num_reqs)`; `num_sample = num_reqs * draft_block`; `draft_tokens[:num_reqs, :draft_block] = ...` (a slice assignment, not a whole-row one); the prepare-inputs kernel is passed `self.draft_block` where it used `num_speculative_steps`.

**Why they needed it.** Header, verbatim: "`dflash_config.block_size` is a property of the checkpoint (8 = 1 anchor + 7 mask tokens), and vLLM made it the target's verify length as well, so a verbatim copy could never exceed 8 tokens per step -- and it sat on that ceiling." The gating on `VLLM_DFLASH2_LOOKUP` is itself argued: "Only the lookup can fill the extra positions, so without it the drafter keeps being asked for the whole (longer) block, which is also the honest A/B control."

**Their numbers.** 7.83 of 8 tokens accepted per step while reproducing a document's first 60 lines — i.e. the 8-token verify block was the binding constraint, not acceptance. Checkpoint block_size = 8 = 1 anchor + 7 mask tokens; drafter's attention window 2,048 tokens.

**llama.cpp — partial.** llama.cpp already has the drafter-block concept (block_size from GGUF metadata, clamping n_max and n_min with a LOG_WRN) and a separate per-step budget (dp.n_max from context/remaining). What it does NOT have is a verify block longer than the draft: the server lays draft.size() tokens into the batch after the sampled token, so verify length is defined by draft length. Filling a longer tail from another source requires fusing two speculators, which the fallback loop forecloses — I read it: `break` fires as soon as no sequence is still drafting, so exactly one impl's flat token list is used per sequence per step. That fusion is the large patch, seam named in technique 37. The flag half needs no patch at all, which is why the status is partial rather than absent. Verify the effective value from the startup line at speculative.cpp:983-986, which prints block_size and the clamped n_max/n_min.

**Equivalent here:** --spec-draft-n-max (default 3) clamped by the DFlash block-size clamp; the verify batch is always exactly 1 + draft.size()

**Evidence (llama.cpp):** `common/common.h:325` · `common/speculative.cpp:988-996` · `common/speculative.cpp:2725-2755` · `tools/server/server-context.cpp:488-496`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** THE single most actionable finding in this slice, and it is one flag. --spec-draft-n-max defaults to 3. Their DFlash sidecar's dflash.block_size (16 if the key is absent) allows up to block_size-1. If they are running draft-dflash without setting -n-max, the drafter is proposing 3 tokens per step when it was trained to propose up to 15. Cost of raising it is unusually low on this box — see technique 47 for why. Magnitude: unknown until measured, paired within a round.

### Suffix-lookup Triton kernel: longest recent suffix match over the request's own token history
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:566-634` · `patches/dflash2-lookup-drafting.patch:764-808`

**What it does.** One Triton program per request scans that request's `all_token_ids` history for the most recent occurrence of the longest suffix (between NMIN and NMAX tokens) of what has been generated so far, and returns the tokens that followed that occurrence as a draft proposal. Reuses the int32 UVA buffer vLLM already maintains, so no extra state.

**Mechanism.** Grid `(num_reqs,)`, `BLOCK=1024`, `num_warps=4`. Reads `req_state` through `idx_mapping_ptr + req * idx_mapping_stride` and returns immediately if `< 0`. Bails if `total_len < NMIN + 2`. Sets `end_of_suffix = total_len - 1`, and sweeps candidate match-ends `e` over `[lo, hi)` where `hi = end_of_suffix` and `lo = tl.maximum(NMIN - 1, total_len - search_max)`. Per block of 1024 candidate ends it compares backwards token by token. On the winner it computes `valid = tl.minimum(k, end_of_suffix - end)`, loads `toks` from `base + end + 1 + idx` masked by `idx < valid`, and writes tokens, `match_len` and `valid`.

**Why they needed it.** Header: "When the model is quoting, listing, editing or otherwise reproducing something it was given -- which is most of what a long-context assistant does -- those tokens are already sitting verbatim in the prompt, tens of thousands of tokens back, where the drafter cannot see them." The drafter sees only a 2,048-token attention window.

**llama.cpp — partial.** llama.cpp has FOUR lookup speculators over the request's own history, so the core idea is not new here. But the semantics differ in ways that matter: ngram-mod hashes a FIXED window of n_match tokens (default 24) with no key stored and no collision check, so there is no longest-match dimension and no variable match length; it gets 'most recent wins' for free because add() overwrites the slot (I read ngram-mod.cpp:27-34). ngram-map keeps at most 4 m-grams per key with a min_hits gate. Neither implements 'longest suffix between NMIN and NMAX, tie-broken by recency'. Status is partial because the capability exists in a different shape, not because the shape is worse — but the vLLM shape's specific claim (short matches are usable, and recency beats length) is testable here with existing flags; see techniques 54 and 55.

**Equivalent here:** ngram-mod (4M-entry hash of the last n_match tokens), ngram-map-k / ngram-map-k4v (n-gram -> m-gram), ngram-cache (1..4-gram)

**Evidence (llama.cpp):** `common/ngram-mod.cpp:15-40` · `common/speculative.cpp:1887-2059` · `common/speculative.cpp:1833-1885` · `common/ngram-map.h:39-77`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** The capability is already theirs and already measured (ngram-mod is in their profile). The unused part is ngram-map-k4v with its own three tunables, which the register would show as never tried.

### Adaptive verify length: `next_num_draft_tokens` host controller
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:363-411` · `patches/dflash2-lookup-drafting.patch:892-908`

**What it does.** Decides, once per step on the host, how many of the proposed tokens the scheduler should actually put up for verification — the drafter's short block normally, the full long block only while a copy appears to be running. This is what keeps the long block from taxing ordinary prose.

**Mechanism.** Runs on the host, not in the kernel, "because the draft pass is replayed from a captured CUDA graph, so anything Python does in there runs at capture time only." Two inputs: the per-request `take_flags` the (replayed) fuse kernel wrote, and `last_num_emitted`. It computes `fused = (self._take_flags[:num_reqs] > 0) & (emitted >= 1 + self.draft_block)` — the second term is saturation evidence, since "a short block can produce at most 1 + draft_block". Returns `num_speculative_steps` or `draft_block`. The model runner picks it up by duck-typing (`hasattr(self.speculator, "next_num_draft_tokens")`) and passes it to `set_draft_tokens(..., num_draft=num_draft)`.

**Why they needed it.** Verbatim: "A long block costs step time on every request in the batch whether or not its tail is accepted, so it takes both, and unanimity across the batch." Header point 2: "Extra verify positions cost about 1 ms of attention each at 25k context."

**Their numbers.** ~1 ms of attention per extra verify position at 25k context. An extra verify position costs +6% per step at 1.5k of context against +27% at 25k.

**llama.cpp — partial.** I read both ends. get_n_draft_max() computes n_ctx - prompt.n_tokens() - 2, min'd with n_remaining()-1, and that number is written into dp.n_max per slot per step; common_speculative_draft then truncates any longer draft to it. That is precisely the host-side per-step verify-length control vLLM had to add, and it already exists — it simply has no acceptance-driven term. The inputs a controller would want are also already maintained host-side: slot.stats.n_draft_tokens, n_draft_accepted and n_draft_verif_steps. llama.cpp already has one crude adaptive mechanism of this family: ngram-mod resets its table after 5 consecutive rounds below 25% acceptance. So this is 'partial' rather than 'absent'.

**Equivalent here:** dp.n_max, set per slot per step at server-context.cpp:2936-2946 from get_n_draft_max()

**Evidence (llama.cpp):** `tools/server/server-context.cpp:441-460` · `tools/server/server-context.cpp:2936-2946` · `common/speculative.cpp:2728-2732` · `common/speculative.cpp:2044-2054` · `tools/server/server-context.cpp:3883-3903` · `src/llama-graph.h:785`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown, but the seam is already built and currently encodes only a context clamp. Adaptive SHORTENING is available today at that seam; adaptive LENGTHENING beyond block_size-1 needs the fusion of 37. Weigh against a llama.cpp-specific cost vLLM does not have: a varying 1+n_draft defeats llama-level graph reuse and CUDA-graph warmup (technique 50).

### Environment variable surface (14 switches, with defaults and stated reasons)
**Where (theirs):** `patches/dflash2-backport.patch:409` · `patches/dflash2-backport.patch:908` · `patches/dflash2-lookup-drafting.patch:276-300` · `patches/dflash2-lookup-drafting.patch:318-321` · `patches/dflash2-lookup-drafting.patch:351` · `patches/dflash2-lookup-drafting.patch:925-926`

**What it does.** Every behavioural choice in this stack is a named env var with a documented default, so each can be A/B'd independently against the same binary. The full set, with defaults: VLLM_DFLASH2_TORCH_TOPK=0 (force torch.topk over flashinfer); VLLM_DFLASH2_DRAFT_TOPK_TOPP=1 (truncate the draft proposal to the request's top-k/top-p); VLLM_DFLASH2_LOOKUP=0 (lookup drafting off by default); VLLM_DFLASH2_LOOKUP_NMIN=6; VLLM_DFLASH2_LOOKUP_NMAX=12; VLLM_DFLASH2_LOOKUP_NSTRONG=6; VLLM_DFLASH2_LOOKUP_AGREE=0; VLLM_DFLASH2_LOOKUP_NMIN_TAIL=4; VLLM_DFLASH2_LOOKUP_LONGMIN=6; VLLM_DFLASH2_LOOKUP_CHEAP_CTX=0 (disabled); VLLM_DFLASH2_LOOKUP_SEARCH=1<<30 (unbounded history scan); VLLM_DFLASH2_LOOKUP_ADAPTIVE=1; VLLM_DFLASH2_LOOKUP_STICKY=3; VLLM_DFLASH2_GRAPH_BOTH=1.

**Mechanism.** All read once in `DFlash2Speculator.__init__` (or module scope for the top-k one, or `cudagraph_utils` for GRAPH_BOTH). `_adaptive` additionally requires `self.draft_block < self.num_speculative_steps`, so it self-disables when there is no tail to schedule. On startup with lookup on, a single log line prints seven of them together: "DFlash2 lookup-augmented drafting on (k=%d nmin=%d nmax=%d nstrong=%d agree=%d nmin_tail=%d longmin=%d search=%d)".

**Why they needed it.** The comment block grouping them is itself the design statement: "Three separate questions, three thresholds: NMIN/NSTRONG/AGREE when to let the lookup replace a token the drafter proposed; NMIN_TAIL when to fill the positions the drafter never proposed; LONGMIN when the long verify block is worth its step time."

**llama.cpp — partial.** The A/B-everything-by-env-var discipline is partly already llama.cpp's: each --spec-* flag has an env var, so a profile file can flip one thing. What is missing is per-request switching, and the reason it is missing is a deliberate `#if 0`, not an architectural obstacle — which makes it absent-but-possible with the smallest possible seam. Two cautions before doing it: the p_min and n_min parsers have no range validation at all (a negative n_min casts to a huge size_t and silently discards every draft), so exposing them per request exposes that too; and GET /props already reports speculative.types, which is the honest way to confirm what a profile actually enabled.

**Equivalent here:** every speculative knob is a CLI flag with an LLAMA_ARG_* env var — but per-request speculative parameters are compiled out behind `#if 0`

**Evidence (llama.cpp):** `tools/server/server-schema.cpp:197` · `tools/server/server-schema.cpp:198` · `tools/server/server-schema.cpp:227` · `common/arg.cpp:4076-4192` · `tools/server/server-task.cpp:83`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The highest-leverage patch in the slice FOR THIS PROJECT specifically, and it is small. The request schema for speculative.n_max / n_min / p_min / type / ngram_size_n / size_m / min_hits is already written and sitting inside an `#if 0` block with the comment 'to keep things simple, we disable speculative parameter adjustments for now'. Deleting the guard would let them A/B speculative settings WITHIN one boot. Their own CLAUDE.md says free VRAM at boot moves 9,326-10,732 MiB, --fit follows it, and effects below 13.6% are noise — every speculative A/B they run today crosses a boot and eats that noise floor. This removes the boot from the comparison.

## already have it — 17

### DFlash reserves K draft KV slots per request, not K-1
**Where (theirs):** `patches/dflash2-backport.patch:76-85`

**What it does.** Patches `SpeculativeConfig`'s per-request draft slot reservation so the `dflash` method returns `num_speculative_tokens` outright, before the generic `parallel_drafting` branch that would have returned `num_speculative_tokens - 1`.

**Mechanism.** An early `if self.method == "dflash": return self.num_speculative_tokens` inserted ahead of the `if self.parallel_drafting:` branch which sets `slots_per_req = self.num_speculative_tokens - 1`.

**Why they needed it.** The comment states the layout: "DFlash uses one bonus query followed by K mask queries (vLLM main)." Parallel drafting's off-by-one assumption does not hold for DFlash's bonus+mask query block, so the generic formula under-reserves by one slot.

**llama.cpp — already have it.** llama.cpp makes exactly the same distinction vLLM had to patch in: the DFlash input layout is [id_last, <mask> * (block_size-1)], yielding at most block_size-1 draft tokens, while anchor-sampling DSpark yields a full block_size. I read the source comment at speculative.cpp:986-989 and it states the layout in the same terms. Separately, cparams.n_rs_seq = draft.n_max for the four model-based types, which is llama.cpp's per-sequence rollback reservation and is derived from the same number. Nothing to port.

**Equivalent here:** n_draft_max = (is_dspark && sample_from_anchor) ? block_size : block_size - 1

**Evidence (llama.cpp):** `common/speculative.cpp:988-996` · `common/common.h:386-392` · `common/common.cpp:1697`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None — llama.cpp already encodes the bonus-vs-mask distinction correctly. But see technique 31: the flag that feeds this clamp defaults to 3.

### `is_causal` takes precedence over the legacy per-layer causality default
**Where (theirs):** `patches/dflash2-backport.patch:91-102`

**What it does.** Rewrites `_dflash_layer_causal(config, layer_idx)` so an explicit top-level `config.is_causal` is consulted first, then `dflash_config['causal']`, and only then the legacy fallback of "causal iff this layer is a sliding-attention layer".

**Mechanism.** `is_causal = getattr(config, "is_causal", None); if is_causal is not None: return bool(is_causal)` inserted above the pre-existing `dflash_config.causal` override and the `layer_types[layer_idx] == _SLIDING_ATTENTION` fallback.

**Why they needed it.** From the patch header: "the DFlash2 config is top-level is_causal=false with all-sliding layer_types; without this hunk the drafter runs causal." The checkpoint's layer_types would otherwise silently flip every layer to causal — again a wrong-but-running drafter.

**llama.cpp — already have it.** The DFlash/DSpark constructor forces the draft context non-causal for its whole life, with no config precedence chain to get wrong. There is no layer_types fallback and no per-layer causality decision in llama.cpp's draft context, so the silent-wrong-drafter failure vLLM was guarding against cannot occur.

**Equivalent here:** llama_set_causal_attn(ctx_dft, false), unconditional for DFlash/DSpark

**Evidence (llama.cpp):** `common/speculative.cpp:1036`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None. The ambiguity vLLM had to resolve does not exist here.

### Class-attribute seams (`decoder_layer_cls`, `model_cls`) so DFlash2 subclasses DFlash1
**Where (theirs):** `patches/dflash2-backport.patch:129-144` · `patches/dflash2-backport.patch:154-171` · `patches/dflash2-backport.patch:647-648` · `patches/dflash2-backport.patch:679-680`

**What it does.** Turns two hard-coded constructor calls in the DFlash1 model into class attributes, so DFlash2 is a three-line subclass rather than a fork. `DFlashQwen3Model.decoder_layer_cls = DFlashQwen3DecoderLayer` and `DFlashQwen3ForCausalLM.model_cls = DFlashQwen3Model`; DFlash2 overrides both.

**Mechanism.** `self.layers = nn.ModuleList([self.decoder_layer_cls(...)])` replaces the literal `DFlashQwen3DecoderLayer(...)`, and `self.model = self.model_cls(...)` replaces `DFlashQwen3Model(...)`. `DFlash2Qwen3Model.decoder_layer_cls = DFlash2Qwen3DecoderLayer`, `DFlash2Qwen3ForCausalLM.model_cls = DFlash2Qwen3Model`.

**Why they needed it.** Not stated in prose, but the effect is that the DFlash1 weight mapper, the fused-KV precompute and the whole `load_weights` path are inherited unchanged by DFlash2 — only the layer body and the added selector differ.

**llama.cpp — already have it.** llama.cpp already shares one implementation across DFlash, DSpark and DFlash2: the same class is constructed with a type tag for dspark, and DFlash2 is a boolean set from metadata rather than a subclass. On the graph side src/models/dflash.cpp gates the conv/selector tensor creation on the presence of selector_hidden.weight, so DFlash1 and DFlash2 share one model file. The refactor vLLM needed is already the shape of this code.

**Equivalent here:** one class + a type tag (dflash/dspark) + an is_dflash2 flag

**Evidence (llama.cpp):** `common/speculative.cpp:2577-2581` · `common/speculative.cpp:975-979` · `src/models/dflash.cpp:125-145`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None — code organisation only.

### Registry entry mapping the DFlash2 architecture string
**Where (theirs):** `patches/dflash2-backport.patch:172-181`

**What it does.** Adds `"DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM")` to vLLM's speculative-model registry, next to the existing `DFlashDraftModel` and the DSpark entries.

**Mechanism.** One line in the registry dict in model_executor/models/registry.py.

**Why they needed it.** It is the same string `_is_dflash2_draft()` and the speculator factory key on, so the architecture name is the single dispatch token across config, registry and speculator.

**llama.cpp — already have it.** llama.cpp dispatches on GGUF arch plus tensor presence rather than an architecture string in a registry: arch=='dflash' + markov_w1.weight => draft-dspark, else draft-dflash; the DFlash2 refinement is the selector_top_k metadata key inside that. Sidecar precedence (mtp > dspark > dflash > eagle3) is already a documented ordering. The dispatch problem is solved by a different mechanism, equally completely.

**Equivalent here:** GGUF arch sniff at common/arg.cpp:564-571 + name->enum map at common/speculative.cpp:34-46

**Evidence (llama.cpp):** `common/arg.cpp:564-571` · `common/speculative.cpp:34-46` · `common/arg.cpp:544-562`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Speculator factory dispatch to DFlash2Speculator
**Where (theirs):** `patches/dflash2-backport.patch:194-208`

**What it does.** Inside the `speculative_config.method == "dflash"` branch of the spec_decode factory, checks the draft architectures for `DFlash2DraftModel` and returns `DFlash2Speculator` instead of `DFlashSpeculator`.

**Mechanism.** Lazy import of `vllm.v1.worker.gpu.spec_decode.dflash2.speculator.DFlash2Speculator` guarded by the architecture test, placed before the existing DFlash1 import.

**Why they needed it.** Both checkpoints use `method="dflash"`, so the method name alone cannot distinguish them; the architecture list is the discriminator.

**llama.cpp — already have it.** Same answer as 6 and 5 from the other side: because llama.cpp does not have two speculator classes, it does not need a factory that picks between them. The DFlash2 path is an in-method branch (`if (is_dflash2)`) that reads the selector lattice out of llama_get_embeddings_nextn. I read that branch directly.

**Equivalent here:** is_dflash2 branch inside the single DFlash implementation

**Evidence (llama.cpp):** `common/speculative.cpp:975-979` · `common/speculative.cpp:1219-1262`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### `seq_lens` clamped to `max_model_len`
**Where (theirs):** `patches/dflash2-backport.patch:291-299`

**What it does.** Clamps the draft attention's absolute sequence length so the bonus+mask block cannot push it past the model's maximum length.

**Mechanism.** `tl.store(out_seq_lens_ptr + req_idx, tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len))` replacing the unclamped store. The surrounding comment explains the quantity: "seq_lens is the absolute sequence length the draft attention reads up to (context + query), not just the count of accepted tokens this step."

**Why they needed it.** Listed in the header as part of the main-branch "prepare-inputs kernel hardening ... seq_lens clamp". Note the neighbouring `clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)` already existed; this makes the length agree with the clamped positions.

**llama.cpp — already have it.** I read get_n_draft_max: n_draft_max = n_ctx - prompt.n_tokens() - 2 (the -2 comment says it leaves room for a context shift), further min'd with n_remaining()-1. Any draft longer than that is truncated after the fact with SPC_DBG 'truncating draft to %d tokens'. So the length can never exceed the context. The waste the map flags is real and applies to their profile: ngram-mod builds a full 64-token lookup and can have it cut to a handful near the context edge, and n_draft_tokens is counted after truncation so the waste does not show up as a lower acceptance rate — it shows up as nothing at all.

**Equivalent here:** get_n_draft_max() + the post-hoc truncation in common_speculative_draft

**Evidence (llama.cpp):** `tools/server/server-context.cpp:441-460` · `common/speculative.cpp:2728-2732` · `src/llama-context.cpp:288`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None to gain; one known waste to be aware of.

### Host-side batch-max top-k (`top_k_max`) to enable the sort-free small-k path
**Where (theirs):** `patches/dflash2-backport.patch:323-337` · `patches/dflash2-backport.patch:341-351`

**What it does.** Adds `SamplingStates.top_k_max(idx_mapping_np)` returning the largest `top_k` in the current batch as a Python int, and threads it into both `apply_top_k_top_p` call sites (the states helper and the V2 sampler). It returns `None` — meaning "no small-k shortcut" — when the batch is empty or when any request's k equals the full vocab size.

**Mechanism.** `ks = self.top_k.np[idx_mapping_np]; if ks.size == 0 or np.any(ks == self.vocab_size): return None; return int(ks.max())`. Computed on the numpy mirror, so no device sync. Passed as the fourth positional argument to `apply_top_k_top_p`.

**Why they needed it.** Header: "the V2 sampler passes k_max so patches/sampler-small-topk-fast-softmax's sort-free small-k top-k/top-p path is used under the V2 runner too". Forcing DFlash2 onto the V2 runner would otherwise have lost an optimisation the V1 path already had.

**llama.cpp — already have it.** vLLM needed to thread a batch-max k down so a small-k shortcut stayed reachable under the V2 runner. llama.cpp decides this locally and unconditionally: top_k takes plain std::partial_sort whenever npartial <= 128, and the default k is 40, so the 128-bucket histogram path never fires in this profile. There is no runner split that could lose the optimisation, and no plumbing to add.

**Equivalent here:** llama_token_data_array_partial_sort_inplace's npartial <= 128 fast path

**Evidence (llama.cpp):** `src/llama-sampler.cpp:198-205` · `src/llama-sampler.cpp:321-338` · `common/common.h:229`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None to gain — the cheap path is already the one taken on every token.

### Grouped dynamic convolution with per-token coefficients and a block-boundary reset
**Where (theirs):** `patches/dflash2-backport.patch:435-455`

**What it does.** `_grouped_conv` computes a `taps`-tap causal FIR filter over the token axis of the hidden state, where the filter coefficients are a static learned `base` kernel plus a per-token `delta` predicted from the hidden state itself, shared across the `group_size` channels of each of `num_groups` groups. Critically, the filter is reset at every `block_size` boundary, so taps never cross from one request's query block into the next.

**Mechanism.** `blocks = hidden_states.unflatten(-1, (num_groups, group_size))`; `coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)` (the delta broadcasts across a group's channels, which is what makes it 'grouped'). Tap 0 is `coefficients[:, 0] * blocks`. Each further tap shifts the block tensor with `F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))` and multiplies by a positional gate `(position >= tap).view(-1, 1, 1)`, where `position` is the token index reduced modulo `block_size`. Output is `.flatten(-2)`.

**Why they needed it.** The modulo is what enforces the block boundary: a token at intra-block position < tap has no predecessor inside its own block, so that tap is zeroed. The lookup patch spells out the consequence of getting `block_size` wrong: "this convolution resets on the block boundary -- with the wrong size it runs across the boundary between two requests' blocks."

**llama.cpp — already have it.** The map did not cover the draft graph so I opened src/models/dflash.cpp. It reads dflash.conv_kernel_size and dflash.conv_group_size from the GGUF, validates n_embd % conv_group_size == 0 with the same error the vLLM patch raises, creates per-layer conv base tensors shaped {n_embd, kernel, 2} (the two-sided kernel) and a projection producing the per-token deltas, and implements the filter in build_dflash2_conv. This is the same mechanism, expressed in ggml. Their measured +34.7% for draft-dflash is evidence it works.

**Equivalent here:** build_dflash2_conv in src/models/dflash.cpp

**Evidence (llama.cpp):** `src/models/dflash.cpp:391-402` · `src/models/dflash.cpp:16-18` · `src/models/dflash.cpp:228-231` · `src/models/dflash.cpp:132-134`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None — already running whenever their DFlash2 sidecar is loaded.

### Two-sided convolution: `prepare` before the sublayer, `finish` after it
**Where (theirs):** `patches/dflash2-backport.patch:458-513` · `patches/dflash2-backport.patch:553-573`

**What it does.** `DFlashGroupedConv` holds a `base_kernel` of shape `[2, taps, hidden_size]` — one kernel for each side — and a single `kernel_projection` that emits both sides' deltas at once. `prepare()` convolves the input with side 0 and *returns the side-1 coefficients unused*; `finish()` applies them to the sublayer's output. The decoder layer wraps both attention and MLP this way.

**Mechanism.** `kernel_projection` is a `ReplicatedLinear(hidden_size, 2 * taps * num_groups, bias=False, quant_config=None)`. `prepare` reshapes its output to `(T, 2, taps, num_groups)` and returns `(self._convolve(h, coefficients[:, 0], 0), coefficients[:, 1])`. The layer's forward is: input_layernorm → `attention_conv.prepare` → `self_attn` → `attention_conv.finish` → post_attention_layernorm → `mlp_conv.prepare` → `mlp` → `mlp_conv.finish`. The `hidden_size % group_size` check raises at construction: "conv_group_size={group_size} must divide hidden_size={hidden_size}."

**Why they needed it.** One projection pass computes the coefficients for both the pre- and post-sublayer filters, so the second filter costs no extra linear layer; the coefficients are carried across the sublayer call in a local variable. `quant_config=None` and `params_dtype=model_config.dtype` keep these small projections unquantised even in a quantised drafter.

**llama.cpp — already have it.** The third dimension of dflash_attn_conv_base and dflash_ffn_conv_base is literally 2 — one kernel per side — and build_dflash2_conv takes a side argument (called with 0 before the sublayer at :574). Separate attn and ffn conv projections exist per layer, matching vLLM's attention_conv / mlp_conv pair. Already implemented, same structure.

**Equivalent here:** conv base tensors of shape {n_embd, kernel, 2} with a side index into build_dflash2_conv

**Evidence (llama.cpp):** `src/models/dflash.cpp:228-231` · `src/models/dflash.cpp:391-402` · `src/models/dflash.cpp:572-575`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Low-rank predecessor/successor codebook candidate selector (bigram edge scoring)
**Where (theirs):** `patches/dflash2-backport.patch:576-596` · `patches/dflash2-backport.patch:599-644`

**What it does.** Scores every (predecessor candidate → successor candidate) edge between consecutive draft steps, turning the drafter's per-step independent top-k lists into a scored lattice that a path walk can traverse. Each vocabulary token has two `rank`-dimensional embeddings — a predecessor code and a successor code — and the edge score is a hidden-state-modulated inner product added to the successor's unary logit.

**Mechanism.** `CandidateSelector` holds `predecessor_codebook` and `successor_codebook`, both `[vocab_size, rank]` frozen parameters, plus a `hidden_projection: ReplicatedLinear(hidden_size, rank, bias=False, quant_config=None)`. `_score_edges` gathers `successors = successor_table[candidate_ids]` and builds `predecessor_ids` by concatenating the anchor token (broadcast to `top_k`) with `candidate_ids[:, :-1]` — i.e. step s's predecessors are step s-1's candidates, and step 0's predecessor is the real last token. The score is `unary_logits[:, :, None] + torch.einsum("blpr,blcr->blpc", predecessors * hidden[:, :, None], successors)`, giving a `[batch, step, top_k_pred, top_k_succ]` tensor. The whole module is `@support_torch_compile`.

**Why they needed it.** This is the piece the backport header names as the reason V2 is forced: "the V2-runner DFlash2Speculator ... DFlash2DraftModel (grouped dynamic convolutions + candidate selector)". The rank-r factorisation is what makes a vocab×vocab bigram table affordable — only `2 * vocab_size * rank` parameters and an einsum over `top_k × top_k` per step.

**llama.cpp — already have it.** src/models/dflash.cpp creates selector_prev {rank, n_vocab}, selector_next {rank, n_vocab} and selector_hidden {n_embd, rank} — exactly the rank-r predecessor/successor codebook factorisation plus the hidden projection. The draft side then walks the lattice: `scores = row + selector_top_k + predecessor * selector_top_k` selects the predecessor's row of the [pred, succ] block, which is the same lattice layout. It even validates n_embd >= top_k*(top_k+1) so the lattice fits in the nextn embedding row.

**Equivalent here:** dflash_selector_prev / dflash_selector_next / dflash_selector_hidden + the lattice read in the DFlash2 draft branch

**Evidence (llama.cpp):** `src/models/dflash.cpp:139-145` · `src/models/dflash.cpp:127-137` · `common/speculative.cpp:1234-1258`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None to port; this is the mechanism their +34.7% is already buying.

### Input embedding scale and output multiplier / logit softcapping
**Where (theirs):** `patches/dflash2-backport.patch:662-676` · `patches/dflash2-backport.patch:682-687` · `patches/dflash2-backport.patch:716-720`

**What it does.** Three scalar knobs read from `dflash_config` reproduce the training-time numerics of the DFlash2 checkpoint: input embeddings are multiplied by `input_embedding_scale`, candidate scores by `output_multiplier`, and then optionally squashed by tanh softcapping.

**Mechanism.** `DFlash2Qwen3Model.embed_input_ids` returns `super().embed_input_ids(input_ids) * self.input_embedding_scale` where the scale is `float(draft_config.get("input_embedding_scale", 1.0))`. In `compute_candidates`: `values = values.float() * self.output_multiplier`, then if `final_logit_softcapping` is set, `values = torch.tanh(values / cap) * cap`. The cap is normalised at construction: `softcap = float(draft_config.get("final_logit_softcapping") or 0.0); self.final_logit_softcapping = softcap if softcap > 0 else None` — so both a missing key and an explicit 0 mean 'off'.

**Why they needed it.** Not argued in prose; these are checkpoint-numerics fidelity knobs, each defaulting to a no-op (1.0, 1.0, None) so a checkpoint without them behaves as before.

**llama.cpp — already have it.** Both keys are read with the same default-to-no-op convention (f_final_logit_softcapping initialised to 0.0f then optionally overwritten; the scale applied only `if (hparams.f_embedding_scale != 0.0f)`). These are standard llama.cpp hparams shared with other architectures, not something DFlash2 needed added. Checkpoint numerics fidelity is preserved.

**Equivalent here:** LLM_KV_EMBEDDING_SCALE and LLM_KV_FINAL_LOGIT_SOFTCAPPING read in dflash.cpp

**Evidence (llama.cpp):** `src/models/dflash.cpp:11-13` · `src/models/dflash.cpp:558-559`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Triton selector walk: greedy vs Gumbel path over the candidate lattice
**Where (theirs):** `patches/dflash2-backport.patch:747-835` · `patches/dflash2-backport.patch:914-939`

**What it does.** One Triton program per request walks the candidate lattice left to right, at each step loading the `top_k` edge scores out of the row selected by the *previously chosen* candidate index, picking one, and carrying that index forward. Greedy requests (temperature 0) take the argmax; sampling requests add Gumbel noise to the temperature-scaled scores.

**Mechanism.** `_selector_walk_kernel` is launched with grid `(num_reqs,)` and `num_warps=1`. Per step, `flat = row * num_steps + step` and `score_base = (flat * top_k + previous) * top_k` — the `previous` index selects which predecessor row of the `[pred, succ]` score block is read. Greedy: `best = tl.max(scores, axis=0)`, `index = tl.min(tl.where(scores == best, offsets, BLOCK_K), axis=0)` (min-index tie-break, deterministic). Sampling: `t_scores = scores / temperature`, add Gumbel noise, then the same max/min-index reduction over `sampled_scores`. `previous = index` at the loop bottom. `BLOCK_K = triton.next_power_of_2(selector_top_k)` with `mask = offsets < top_k`.

**Why they needed it.** The lattice from `_score_edges` is `[batch, step, pred, succ]`; a Viterbi-style full search would be far more work, so this is a greedy/sampled single path through it. The whole walk stays in registers for 16 candidates, which is why `num_warps=1`.

**llama.cpp — already have it.** I read the walk. It carries `predecessor` forward exactly as the Triton kernel does, indexes the lattice row by the previously chosen candidate, and branches on temperature: argmax via std::max_element for greedy, categorical sampling via std::discrete_distribution for temp>0, with a dedicated selector RNG seeded seed ^ 0x85ebca6b. Same algorithm, host-side. The min-index tie-break vLLM needed for determinism is std::max_element's first-max semantics here, which is the same guarantee.

**Equivalent here:** the is_dflash2 branch: argmax when temperature <= 0, std::discrete_distribution when temperature > 0

**Evidence (llama.cpp):** `common/speculative.cpp:1234-1262` · `common/speculative.cpp:1225-1232`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None to port. One unused knob attached to it — see technique 26 for --spec-draft-p-min.

### Cache temperature-APPLIED draft logits (0.27.1 convention, not main's)
**Where (theirs):** `patches/dflash2-backport.patch:19-22` · `patches/dflash2-backport.patch:823-831`

**What it does.** The selector walk stores `realized = t_scores` (post-division by temperature) for sampling requests, and `realized = scores` for greedy ones, rather than storing pre-temperature scores.

**Mechanism.** In each branch of the walk, a `realized` local is set, then `tl.store(realized_scores_ptr + candidate_base + offsets, realized, mask=mask & valid)`. The comment: "vLLM 0.27.1 caches temperature-APPLIED draft logits (its rejection sampler reads them as-is); upstream main divides on load instead."

**Why they needed it.** Header, verbatim: "0.27.1 caches temperature-APPLIED draft logits (main caches pre-temperature and divides on load), so the selector walk stores scores/temperature — without this the lossless verify would use softmax(scores) while the draft sampled from softmax(scores/T) for 0 < T != 1." This is a losslessness bug that only appears for temperatures other than 0 and 1 — exactly the range where it would look plausible.

**llama.cpp — already have it.** This was the one correctness question in the slice I thought llama.cpp might have wrong, so I read it. The DFlash2 branch builds dist.probs by softmaxing (scores / dp.temperature), normalises, samples the predecessor from THAT distribution, and pushes the same dist into dp.dists. So the cached q is the distribution actually sampled from — the exact invariant the vLLM patch had to restore. dp.temperature is threaded from slot.task->params.sampling.temp at server-context.cpp:2936-2946, so it is the request's own temperature. No bug here.

**Equivalent here:** dist.probs[k] = exp((scores[k] - max_score) / dp.temperature), normalised, then sampled from

**Evidence (llama.cpp):** `common/speculative.cpp:1239-1254` · `common/sampling.cpp:763-780`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None to fix. Confirms their DFlash2 verify is lossless at temperatures other than 0 and 1, which was worth checking.

### Clamp the convolution's block_size to the trained block, not the verify block
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:65-81`

**What it does.** When the verify block is longer than the drafter's own, the grouped convolution must still reset on the drafter's block boundary, so its `block_size` is the min of the requested block and the checkpoint's.

**Mechanism.** `block_size=min(1 + speculative_config.num_speculative_tokens, int(draft_config.get("block_size", 1 << 30)))` replacing the plain `1 + speculative_config.num_speculative_tokens`. The `1 << 30` default makes the clamp inert for a checkpoint that declares no block size.

**Why they needed it.** Verbatim: "The drafter emits the block its checkpoint was trained for even when the target verifies a longer one ... and this convolution resets on the block boundary -- with the wrong size it runs across the boundary between two requests' blocks." Cross-request contamination inside a batched convolution is the failure this prevents.

**llama.cpp — already have it.** vLLM needed this clamp because it decoupled the two lengths (technique 31) and the conv would otherwise run across a request boundary. llama.cpp never decouples them: block_size is the single number that sets the block layout at :1183 and caps n_max/n_min at :988-996, with a warning when it fires. The cross-request contamination this guards against is structurally unreachable. Note the corollary: if the fusion patch of technique 37 is ever written, THIS clamp is the thing that must not be relaxed along with it.

**Equivalent here:** block_size read from dflash.block_size and used for both the block layout and the n_max clamp

**Evidence (llama.cpp):** `common/speculative.cpp:966-980` · `common/speculative.cpp:988-996` · `common/speculative.cpp:1183`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None — and it cannot go wrong here, because llama.cpp clamps the request DOWN to the trained block instead of letting a longer block through.

### Matches are allowed to overlap the suffix they matched, so periodic patterns propose themselves
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:598-602` · `patches/dflash2-lookup-drafting.patch:44-46`

**What it does.** Does not require the candidate match to end before the suffix begins — only before the suffix's own end. A period-p repetition therefore matches at `e = t - p` and proposes its own continuation, which is how a repeated list marker, an indent level or a code fence gets drafted.

**Mechanism.** `hi = end_of_suffix` (not `end_of_suffix - NMAX`), with the comment: "Candidate match ends run over [lo, hi): any position before the suffix's own end. A candidate may overlap the suffix (a period-p repetition matches at e = t - p), which is how repeated list markers and indentation get proposed." The `valid = tl.minimum(k, end_of_suffix - end)` clamp is what keeps an overlapping match from reading past the end of real history: "A match close to the suffix yields fewer than k of them; the caller leaves the rest of the block to the drafter."

**Why they needed it.** Header point 4, verbatim: "A match may overlap the suffix it matched, so a repeating pattern (a list marker, an indent, a fence) is proposed from its own period instead of missed."

**llama.cpp — already have it.** ngram-mod records (window of n_match tokens -> successor) for every position as it goes and looks up the same way, so once a repeated list marker or indent has occurred with its 24-token context it proposes its own continuation on the next occurrence. There is no 'must end before the suffix begins' constraint to relax because there is no candidate-end sweep at all. The overlap property vLLM had to arrange deliberately falls out of the data structure here.

**Equivalent here:** ngram-mod's hash is updated as tokens are processed, so a period-p repetition inside the window is recorded and matched

**Evidence (llama.cpp):** `common/ngram-mod.cpp:27-34` · `common/speculative.cpp:1887-2059`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None new — the behaviour is already there, for free, in what they run.

### `set_req_states` hook: hand the request token history to the speculator
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:880-891` · `patches/dflash2-lookup-drafting.patch:413-415` · `patches/dflash2-lookup-drafting.patch:477-482`

**What it does.** Gives the speculator a reference to the model runner's `req_states`, from which the lookup reads `all_token_ids.gpu` (the int32 history) and `total_len.gpu`. No new buffer is allocated for the history.

**Mechanism.** Model runner: `if self.speculator is not None and hasattr(self.speculator, "set_req_states"): self.speculator.set_req_states(self.req_states)`, placed immediately after `req_states` is constructed and before `InputBuffers`. `_apply_lookup` no-ops when `self._req_states is None`, so the hook being absent degrades to plain DFlash2.

**Why they needed it.** Header: the scan uses "the int32 buffer vLLM already maintains, `req_states.all_token_ids`" — the whole cost argument depends on not duplicating the history.

**llama.cpp — already have it.** I read the draft-params initialisation: slot.spec_prompt is assigned slot.prompt.tokens.get_text_tokens() and a pointer to it goes into the draft params, so every impl sees the request's full token history without any extra buffer. ngram-mod's draft_one indexes back into it (prompt.at(cur_len - n + 1 + i)) to build its hash window. The no-extra-state property vLLM's cost argument depends on holds here too.

**Equivalent here:** dp.prompt = &slot.spec_prompt, set per slot per step

**Evidence (llama.cpp):** `tools/server/server-context.cpp:2936-2946` · `common/speculative.h:56-58` · `common/speculative.cpp:1985-2010`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None to build.

### Requires spec-decode-attn.patch above 10 verify query tokens
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:56-58`

**What it does.** A hard dependency: without a different attention backend, a verify block longer than 10 query tokens falls back to FlashAttention-2, which cannot split the KV sequence for multi-query decode.

**Mechanism.** Stated as a prerequisite in the header; the backend change itself is in another patch file.

**Why they needed it.** Verbatim: "patches/spec-decode-attn.patch for a verify block longer than 10 query tokens (FlashAttention-2 does not split the KV sequence for multi-query decode; falling back to it doubles the step at 25k context)." This is the constraint that makes the long block possible at all — without it, the long block would be net negative regardless of acceptance.

**Their numbers.** Falling back to FlashAttention-2 for a >10-query-token verify block doubles the step at 25k context.

**llama.cpp — already have it.** vLLM's constraint is that FlashAttention-2 cannot split the KV sequence for multi-query decode, so a >10-query-token verify block falls back and doubles the step at 25k. llama.cpp's MMA kernel is the multi-query-capable one and it is selected automatically for anything above the VEC threshold — no patch, no fallback. The VEC kernel does cap at 2 query columns (cols_per_block is 1 or the constant 2, no wider instance), which is exactly why the switch happens at 3 rather than at 10. Different threshold, and on the right side of it.

**Equivalent here:** BEST_FATTN_KERNEL_MMA_F16, which handles arbitrary query counts

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:461-483` · `ggml/src/ggml-cuda/fattn.cu:469` · `ggml/src/ggml-cuda/fattn-vec.cuh:553-572`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Good news for any long-block work here: llama.cpp has no cliff at 10 query tokens. It has one at 3, and it is already past it (technique 47). Relative to vLLM-on-FA2, a long verify block is structurally cheaper on this stack.

## impossible here — 1

### Capture CUDA-graph decode graphs for BOTH query lengths
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:911-943`

**What it does.** When lookup drafting is on, the decode step alternates between two query lengths (the drafter's block and the full verify block), so the graph capture list is extended to include the short one as well.

**Mechanism.** Guarded by `VLLM_DFLASH2_GRAPH_BOTH=1` (default on) AND `VLLM_DFLASH2_LOOKUP=1`. Reads `_block = int((hf_config.dflash_config or {}).get("block_size", 0))` and derives `_short = _block + (self.decode_query_len - self.vllm_config.num_speculative_tokens - 1)` — i.e. the short length carries over whatever offset the runner's `decode_query_len` has relative to `num_speculative_tokens`. Appended only if `0 < _short < self.decode_query_len`, with a log line naming both lengths.

**Why they needed it.** Verbatim: "A DFlash drafter whose checkpoint block is shorter than the verify block ... schedules either length, so both need a decode graph. Without this the short step -- the one taken on ordinary prose -- runs piecewise." The uncaptured length is the *common* one, so the miss would cost on the majority of steps.

**llama.cpp — impossible here.** Marked impossible rather than possible because the fix is not a llama.cpp-level patch: the graph key is the first node pointer of the split, and its stated purpose in the source comment is CPU/GPU splits, not decode shapes. Making several decode lengths coexist means changing the key and the reuse predicate in ggml core and in llm_graph_params — upstream surgery, not a local seam, and it would still have to survive the two-call warmup rule. The honest verdict is that llama.cpp cannot do what GRAPH_BOTH does, and that this cost is being paid today rather than only hypothetically after some future adaptive patch.

**Equivalent here:** none — the CUDA graph map is keyed on cgraph->nodes[0] (which split), not on shape

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:2574-2576` · `ggml/src/ggml-cuda/common.cuh:1426-1455` · `src/llama-graph.h:785` · `ggml/src/ggml-cuda/ggml-cuda.cu:4253-4268` · `tools/server/server-context.cpp:617-619`

**Effort:** large-patch · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Diagnostic, and it is live right now. Their profile alternates step sizes: ngram-mod is all-or-nothing (n_min 48, n_max 64), so a step is either 65 query tokens or 1. llm_graph_params::allow_reuse requires equal ubatch.n_tokens, so the ggml graph is rebuilt on every flip, which re-splits, which changes node properties, which resets CUDA-graph warmup_complete — and re-arming needs two consecutive identical calls. Read `graphs reused = %d` in the per-completion slot timings; if it is near zero, this is why.

## not applicable — 22

### Force the V2 model runner whenever the draft is DFlash2
**Where (theirs):** `patches/dflash2-backport.patch:43-54` · `patches/dflash2-backport.patch:60-69`

**What it does.** Adds a branch to vLLM's `use_v2_model_runner` decision in config/vllm.py that returns True as soon as the speculative draft model's architecture list contains `DFlash2DraftModel`. A companion helper `_is_dflash2_draft()` does the detection by reading `speculative_config.draft_model_config.architectures`, deliberately keying on the same string the speculator factory selects on.

**Mechanism.** `_is_dflash2_draft()` returns `"DFlash2DraftModel" in (draft_config.architectures or [])` after checking `spec.method == "dflash"`; the caller short-circuits the V1/V2 decision to True, alongside the existing `_dflash_needs_multi_kv_group()` and diffusion cases.

**Why they needed it.** Quoting the patch: "The DFlash2 candidate selector exists only in the V2 speculator. On V1 the same checkpoint drafts through DFlashProposer, which never calls it, so the draft degrades to DFlash1 silently." This is a failure that produces a working server with worse acceptance and no error — the exact class of silent degradation the guard exists to prevent.

**llama.cpp — not applicable.** There is no V1/V2 runner split in llama.cpp. The equivalent dispatch is a metadata read in the DFlash constructor — selector_top_k > 0 sets is_dflash2, which turns on the selector lattice, disables the draft backend sampler (`backend_sampling && !is_dflash2`), and switches top_k from 10 to selector_top_k. The *class* of failure vLLM guarded against does exist here though: the LOG_INF line at speculative.cpp:983-986 prints block_size, mask_token_id, n_extract and sample_from_anchor and does NOT print selector_top_k, so a sidecar missing that key runs as DFlash1 with no log line saying so. The only indirect tell is that dists get filled at temp>0 (which flips the accept rule to residual). Adding selector_top_k to that log line is a one-line patch and is the whole transferable content.

**Equivalent here:** GGUF metadata sniff: is_dflash2 = (dflash.selector_top_k > 0)

**Evidence (llama.cpp):** `common/speculative.cpp:975-979` · `common/speculative.cpp:983-986` · `common/speculative.cpp:1015`

**Effort:** small-patch · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No runtime value — llama.cpp has one code path. Diagnostic value only: today nothing in the startup log tells you the DFlash2 selector engaged.

### Dequantise the W4A16 qkv rows in-place for the context-KV precompute
**Where (theirs):** `patches/dflash2-backport.patch:108-126` · `patches/dflash2-backport.patch:145-151`

**What it does.** Replaces the fused-KV weight gather `a.qkv_proj.weight[a.q_size:]` with `_dense_kv_rows(a)`, a helper that returns rows `[q_size:]` of the qkv projection as a dense bf16 matrix even when qkv_proj is a compressed-tensors pack-quantized (W4A16/W8A16, symmetric, group) layer. This lets the drafter itself be quantised while its context-KV precompute still gets dense weights.

**Mechanism.** If a plain 2-D `.weight` exists it is sliced directly. Otherwise it reads `qkv.weight_packed` / `qkv.weight_scale`, infers `bits = 32 * packed.shape[1] // in_f` from the int32 packing density, calls `compressed_tensors.compressors.pack_quantized.base.unpack_from_int32(packed.data, bits, torch.Size([out_f, in_f]), packed_dim=1)`, then rescales per group: `group = in_f // scale.shape[1]`, reshape to `(out_f, in_f//group, group)`, multiply by `scale[..., None]` in fp32, flatten back, cast to the scale dtype (bf16 fallback), and slice `[q_size:]`. The result is concatenated across layers into `_fused_kv_weight` of shape `[num_layers * 2 * kv_size, hidden_size]`.

**Why they needed it.** The docstring pins the timing constraint: it "is called from load_weights, i.e. before the Marlin repack, so weight_packed/weight_scale are still in the plain checkpoint layout and can be dequantized here." It also flags a trap: "(weight_shape holds only the last-loaded shard of a fused qkv; use the tensors.)" — the recorded shape metadata is wrong for a fused projection, so shapes are recovered from the packed tensor and `qkv.input_size` instead.

**llama.cpp — not applicable.** This exists in vLLM because a torch-level precompute wanted dense bf16 rows out of a pack-quantised layer before Marlin repack. ggml has no such constraint: every matmul consumes the quantised tensor directly through its own kernel, and the DFlash graph builds its fused context-KV inside the ggml graph rather than by gathering raw weight rows in Python. Note also that llama.cpp cannot re-quantise or dequantise a weight at load time at all — the model-loading area's first CANNOT — so there is no seam here even if you wanted one.

**Evidence (llama.cpp):** `src/llama-model-loader.cpp:1053-1065` · `src/models/dflash.cpp:125-145`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### `sample_idx_mapping` sentinel of -1 instead of zero-init
**Where (theirs):** `patches/dflash2-backport.patch:211-220` · `patches/dflash2-backport.patch:223-228`

**What it does.** Changes the DFlash speculator's `sample_idx_mapping` buffer from `torch.zeros(...)` to `torch.full((max_num_sampled_tokens,), -1, ...)`, and changes the pre-capture reset from `.zero_()` to `.fill_(-1)`. Every DFlash2 kernel then tests `req_state >= 0` to decide whether a row is live.

**Mechanism.** `torch.full((max_num_sampled_tokens,), -1, dtype=torch.int32, device=device)`; the CUDA-graph capture preamble calls `self.sample_idx_mapping.fill_(-1)` alongside `sample_indices.zero_()` and `sample_pos.zero_()`.

**Why they needed it.** Header, verbatim: "sample_idx_mapping sentinel -1 (DFlash2's kernels test req_state >= 0; with 0.27.1's zero-init the CUDA-graph capture would scatter every padding row into request slot 0 and poison its proposal distribution)". The inline comment: "-1 = inert row (DFlash2's kernels test req_state >= 0); padded to -1 per step anyway". Zero is a legal request slot, so zero-init makes padding indistinguishable from request 0.

**llama.cpp — not applicable.** This is a CUDA-graph-capture padding hazard specific to a batched device-side speculator. llama.cpp's per-sequence state (impl_last, dparams, sinfos) is host-side C++ indexed directly by seq_id, and with -np 1 those vectors are length 1 — there are no padding rows that could alias into request slot 0. The hazard has no representation here.

**Evidence (llama.cpp):** `common/speculative.cpp:2201-2209` · `tools/server/server-context.cpp:1224-1226`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None on this box.

### Rejected-suffix context rows excluded from the DFlash prepare-inputs slot computation
**Where (theirs):** `patches/dflash2-backport.patch:232-248`

**What it does.** In the DFlash prepare-inputs Triton kernel, splits the notion of "context row" into `is_ctx` (the loop's iteration extent, unchanged) and a new `is_valid_ctx` (`j < num_valid_ctx`, where `num_valid_ctx = valid_ctx_end - ctx_start` and `valid_ctx_end = ctx_end - num_rejected`). Position loads, block-table loads and the query offset all move onto the valid extent.

**Mechanism.** Adds `num_valid_ctx = valid_ctx_end - ctx_start`; rewrites `is_query = (j >= num_valid_ctx) & (j < num_valid_ctx + num_query_per_req)` and `query_off = j - num_valid_ctx` (previously both used `num_ctx`); `ctx_pos` and `ctx_block_id` are loaded with `mask=is_valid_ctx` instead of `mask=is_ctx`.

**Why they needed it.** Header: DFlash "prepare-inputs kernel hardening (rejected-suffix context rows ... no longer write draft KV into physical block 0)". The tokens rejected by the previous verify are still inside the context span but are no longer real context, so their block-table entries are stale.

**llama.cpp — not applicable.** llama.cpp has no paged block table and no prepare-inputs kernel. After a partial acceptance it either seq_rm's the rejected suffix (PART) or restores a checkpoint and replays (FULL) — and on a Qwen3.5-style hybrid target the probe returns FULL, so the checkpoint path is the live one. Rejected tokens are removed from the cache rather than left in a context span that a kernel must learn to skip. Different memory model, problem does not arise.

**Evidence (llama.cpp):** `tools/server/server-context.cpp:3825-3840` · `tools/server/server-context.cpp:3869-3888`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Physical block 0 is a null block, never a writable KV slot (context rows)
**Where (theirs):** `patches/dflash2-backport.patch:262-273`

**What it does.** Guards the context slot-mapping computation so any row whose block-table entry resolves to block id 0 writes `PAD_SLOT_ID` instead of a real slot address.

**Mechanism.** `ctx_resident = is_valid_ctx & (ctx_block_id != 0)`, then `ctx_slot = tl.where(ctx_resident, ctx_block_id * block_size + (ctx_pos % block_size), PAD_SLOT_ID)`.

**Why they needed it.** Verbatim comment: "Block 0 is the null block. Old sliding-window context positions can map to it after eviction; rejected suffix rows are invalid context as well. Neither kind of row may write draft KV into physical block 0. (vLLM main)" — an unguarded write corrupts the shared null block, which every evicted position aliases.

**llama.cpp — not applicable.** There is no null block. llama.cpp's KV cache is a per-layer contiguous tensor addressed by cell index, with the buffer type derived from the layer's device; there is no shared sentinel block that an evicted position aliases to and could therefore be corrupted by a stray draft write. The whole class of bug is absent from the design.

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:209-217`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Same null-block guard for query rows
**Where (theirs):** `patches/dflash2-backport.patch:275-287`

**What it does.** Applies the identical block-0 test to the query half of the same kernel, so a query token whose block-table entry is a padding/evicted entry also maps to `PAD_SLOT_ID`.

**Mechanism.** `q_resident = is_query & (q_block_id != 0)`; `q_slot = tl.where(q_resident, q_block_id * block_size + (query_pos % block_size), PAD_SLOT_ID)`.

**Why they needed it.** Comment: "A null block is never a writable cache slot (sliding-window block tables contain evicted/global padding entries). (vLLM main)"

**llama.cpp — not applicable.** Same reason as 10 — no block table, no PAD_SLOT_ID, no null block for a query row to land in.

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:209-217`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Power-of-two fast path for the intra-block position
**Where (theirs):** `patches/dflash2-backport.patch:448-452`

**What it does.** Computes the intra-block token position with a bitmask when `block_size` is a power of two and with a modulo otherwise.

**Mechanism.** `if block_size & (block_size - 1) == 0: position = position & (block_size - 1) else: position = position % block_size`, applied to `torch.arange(hidden_states.shape[0])`.

**Why they needed it.** No reason stated in the patch. The default DFlash2 block is 8 (1 anchor + 7 mask tokens), which is a power of two, so the masked branch is the one normally taken; the modulo branch keeps correctness for non-power-of-two verify blocks, which the lookup patch makes reachable.

**llama.cpp — not applicable.** A per-token Python/torch scalar micro-optimisation. build_dflash2_conv works on ggml tensors laid out per block rather than computing an intra-block index token by token, so there is no modulo on a hot scalar path to replace with a bitmask. Nothing to port and nothing to gain.

**Evidence (llama.cpp):** `src/models/dflash.cpp:391-402`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Candidate head that works through a quantised (int8-Marlin) shared lm_head
**Where (theirs):** `patches/dflash2-backport.patch:689-708`

**What it does.** `compute_candidates` computes the drafter's top-k candidates through the *target's* lm_head, accepting any quant_method that exposes `apply()`, reshaping to 2-D for it, and masking out the vocabulary padding rows so padding can never be selected as a candidate.

**Mechanism.** Accepts `isinstance(qm, UnquantizedEmbeddingMethod) or hasattr(qm, "apply")`. Saves `shp = hidden_states.shape`, calls `qm.apply(self.lm_head, hidden_states.reshape(-1, shp[-1]), bias=None)`, reshapes back to `(*shp[:-1], -1)`. Then `num_pad = self.lm_head.shard_indices.num_org_vocab_padding; if num_pad > 0: logits[..., -num_pad:] = -float("inf")` before `_topk(logits, selector.top_k)`, and rebases the ids with `+ self.lm_head.shard_indices.org_vocab_start_index`.

**Why they needed it.** Header: "our stack: the draft shares the target's int8-Marlin lm_head, so compute_candidates accepts any quant_method with apply() (2-D input)". The inline comment repeats the shape constraint: "The draft shares the target lm_head, which may be a quantized (Marlin) ParallelLMHead here; its apply() wants a 2-D input." Note the error message left in place still says "DFlash2 requires an unquantized target LM head for candidate TopK", which is now stale relative to the widened check.

**llama.cpp — not applicable.** ggml matmuls consume quantised tensors natively, so there is no unquantized-head requirement to widen and no apply()-shape contract to satisfy. The structurally related fact in llama.cpp is that an EAGLE3/DFLASH sidecar shipping without tok_embd/output is the one case where cparams.ctx_other = ctx_tgt is honoured and the target's memory is shared — the same 'the draft borrows the target's head' arrangement, arrived at without any quantisation plumbing.

**Evidence (llama.cpp):** `src/llama-context.cpp:142-161` · `common/speculative.cpp:2460-2461`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Tensor-parallel candidate merge: all-gather then a second top-k
**Where (theirs):** `patches/dflash2-backport.patch:710-714`

**What it does.** Under TP > 1, each rank's local top-k values and (globally rebased) ids are all-gathered and a second top-k selects the global winners, with the ids gathered along by index.

**Mechanism.** `values = tensor_model_parallel_all_gather(values, dim=-1)`; `ids = tensor_model_parallel_all_gather(ids, dim=-1)`; `values, selected = _topk(values, selector.top_k)`; `ids = ids.gather(-1, selected)`. This runs after the local ids have already had `org_vocab_start_index` added, so the gathered ids are directly comparable.

**Why they needed it.** Not argued; it is the standard vocab-parallel reduction, but note it costs a second radix top-k over `world_size * top_k` values, and the ids are carried through `gather` rather than recomputed.

**llama.cpp — not applicable.** Single RTX 4070 SUPER. -sm tensor kills --fit outright before it measures anything, -sm row aborts the placement pass, and pipeline parallelism requires n_devices() > 1 so it stays false. There is no vocab-parallel split to reduce across.

**Evidence (llama.cpp):** `src/llama-context.cpp:427-433` · `common/fit.cpp:182-184`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None — one GPU.

### Gumbel noise keyed by absolute token position and candidate id, with fp32/fp64 variants
**Where (theirs):** `patches/dflash2-backport.patch:811-822`

**What it does.** Derives the sampling noise from a per-request seed and the predicted token's position so the draft's randomness is reproducible and matches what the verifier will key on, then draws one Gumbel per candidate id.

**Mechanism.** `position = tl.load(sample_pos_ptr + flat) - 1`; `gumbel_seed = tl.randint(seed, position)`. With `USE_FP64`: `uniform = tl_rand64(gumbel_seed, candidates, includes_zero=False)` and `noise = -tl.log(-tl.log(uniform))`. Without: `uniform = tl_rand32(...)` and `noise = -tl.log(-tldevice.log1p(-uniform))` — the log1p form is used in fp32 because `log(1-u)` loses precision for small u. The noise is indexed by the *candidate token id*, not by the slot, so the same token gets the same noise regardless of its rank. The DFlash1 hunk documents the off-by-one: "sample_pos is the predicted token's position Q; verification keys Gumbel by the predecessor (Q-1). sample_draft adds +1, so pass Q-2."

**Why they needed it.** Keying by (seed, position, token id) is what lets the target-side rejection sampler reproduce the draft's randomness; keying by slot index would break as soon as the candidate ordering differed.

**llama.cpp — not applicable.** vLLM keys the draft's noise by (seed, position, token id) so the target-side rejection sampler can REPRODUCE the draft's randomness. llama.cpp does not need to reproduce anything: the DFlash2 branch stores the realised proposal distribution q into dp.dists (ids + probs) and the residual verifier reads q directly, comparing u*q(x) <= p(x). Storing the distribution rather than replaying the noise is a different and simpler solution to the same problem, and it is already in place. The off-by-one hazard vLLM documents (position Q vs Q-1) cannot exist here.

**Evidence (llama.cpp):** `common/speculative.cpp:1239-1258` · `common/sampling.cpp:722-793` · `tools/server/server-context.cpp:3828-3830`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Placeholder-tensor trick for the disabled-truncation branch
**Where (theirs):** `patches/dflash2-backport.patch:920-932`

**What it does.** When truncation is off, the kernel still receives two tensor arguments in the `req_top_p`/`req_top_k` positions — arbitrary already-resident tensors — because Triton needs a pointer argument regardless, and the `TRUNCATE` constexpr compiles the loads away.

**Mechanism.** `self._req_top_p if truncate else self.temperature` and `self._req_top_k if truncate else self.sample_idx_mapping` are passed positionally, with `TRUNCATE=truncate` as a `tl.constexpr`. `truncate = self._truncate and self._req_top_p is not None`, so the buffers being unset (model runner never called `set_sampling_states`) also disables it.

**Why they needed it.** Not stated. Effect: no `None`-handling branch inside the kernel and no separately compiled signature, at the cost of two dead arguments.

**llama.cpp — not applicable.** An artefact of Triton needing a pointer argument for a compiled-away load. The equivalent llama.cpp code is a plain C++ `if` over host memory with no compiled kernel signature to keep stable. There is nothing to trick.

**Evidence (llama.cpp):** `common/speculative.cpp:1234-1262`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Dense `-inf` draft-logit buffer plus a sparse candidate-id cache with an erase-then-write invariant
**Where (theirs):** `patches/dflash2-backport.patch:888-903` · `patches/dflash2-backport.patch:837-867`

**What it does.** Allocates a full `[max_num_reqs, num_speculative_steps, vocab_size]` fp32 buffer filled with `-inf` once, and maintains it sparsely: each step erases exactly the `top_k` entries the previous step wrote (whose ids are remembered in `_cached_candidate_ids`) before writing the new ones. The buffer therefore never needs a full `fill_` and is always a valid dense distribution with `top_k` finite entries per row.

**Mechanism.** `_cache_draft_logits_kernel` runs one program per (request, step) flat index. It computes `cache_base = (req_state * num_steps + step) * top_k`, loads `old_token_ids` from `cached_candidate_ptr`, stores `-inf` at those vocab offsets, then stores the new `scores` at the new `token_ids` and overwrites `cached_candidate_ptr` with them. `mask = (req_state >= 0) & (offsets < top_k)` is what makes the -1 sentinel work.

**Why they needed it.** Comment on the allocation: "The selector samples a probabilistic path for non-greedy requests, so rejection sampling always needs the realized proposal distribution." The erase-then-write scheme is the reason a vocab-sized buffer is affordable per step — only `2 * top_k` scattered stores per (request, step) instead of a `vocab_size` memset.

**llama.cpp — not applicable.** llama.cpp's proposal distribution is a std::vector of common_speculative_token_dist, each holding selector_top_k ids and probs — sparse by construction, sized top_k, rebuilt per position per step. There is no vocab-sized device buffer whose staleness has to be maintained by an erase-then-write scheme, so the invariant this technique protects has nothing to protect.

**Evidence (llama.cpp):** `common/speculative.cpp:1239-1254` · `common/speculative.h:36-43`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Precomputed anchor-index gather for the selector's first predecessor
**Where (theirs):** `patches/dflash2-backport.patch:876-880` · `patches/dflash2-backport.patch:986`

**What it does.** Caches the strided indices of each request's bonus (anchor) query token inside the flattened input-id buffer, so the selector's step-0 predecessor token can be fetched with one gather.

**Mechanism.** `self._anchor_indices = torch.arange(max_num_reqs, dtype=torch.int64, device=device) * self.num_query_per_req`; at draft time `anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]`. Because `num_query_per_req = 1 + draft_block`, index `r * num_query_per_req` is request r's bonus token.

**Why they needed it.** `_score_edges` needs a real token id as step 0's predecessor (there is no step -1 candidate list); the anchor is that token.

**llama.cpp — not applicable.** The step-0 predecessor is simply dp.id_last, passed in the draft params as slot.sampled. There is no flattened multi-request input-id buffer to compute strided anchor offsets into. The walk starts from `int32_t predecessor = 0` indexing the lattice row that the anchor position already produced.

**Equivalent here:** dp.id_last

**Evidence (llama.cpp):** `tools/server/server-context.cpp:2936-2946` · `common/speculative.cpp:1234-1240`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Pack (length, position) into one int64 so "longest, then most recent" is a single max-reduction
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:561-563` · `patches/dflash2-lookup-drafting.patch:621-627`

**What it does.** Encodes each candidate match as `match_len * 2^32 + end_index` in an int64, so a plain `tl.max` over the block simultaneously implements the primary key (longest match) and the tie-break (largest end index = most recent), with no second pass and no branchy comparison.

**Mechanism.** `SCORE_STRIDE = tl.constexpr(1 << 32)`; `score = tl.where(alive, match.to(tl.int64) * SCORE_STRIDE + e.to(tl.int64), -1)`; `best = tl.maximum(best, tl.max(score, axis=0))`. Decoded afterwards as `match_len = (best // SCORE_STRIDE).to(tl.int32)` and `end = (best % SCORE_STRIDE).to(tl.int32)`. `best` is initialised to -1 so "no match anywhere" is detectable with `if best < 0: return`.

**Why they needed it.** Stated precondition, verbatim: "Positions are packed into one int64 score as len * SCORE_STRIDE + end_index, so that 'longest match, then most recent' is a single max-reduction. max_model_len < SCORE_STRIDE." The 2^32 stride is the correctness constraint the comment names.

**llama.cpp — not applicable.** There is no length dimension to reduce over: ngram-mod's window is fixed at n_match, so every match has the same length, and 'most recent' is already implicit in the overwrite policy of add(). ngram-map indexes a 262144-entry hash and keeps up to 4 values per key. Neither runs a scan whose reduction would benefit from packing. If someone wrote a true longest-match lookup impl for llama.cpp it would be a host-side C++ loop, where a plain (len, pos) comparison is clearer than a bit-packed int64 and costs nothing extra.

**Evidence (llama.cpp):** `common/ngram-mod.cpp:27-34` · `common/ngram-map.h:39-42`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** None.

### NMIN-token reject test before any candidate is extended
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:604-620` · `patches/dflash2-lookup-drafting.patch:12-13`

**What it does.** Splits the match loop in two: first an unconditional NMIN-iteration comparison that kills almost every candidate, then — only if any candidate in the 1024-wide block survived — an extension loop from NMIN up to NMAX. Blocks with no survivor skip the extension entirely.

**Mechanism.** Phase 1: `for j in range(NMIN): s_j = tl.load(base + end_of_suffix - j); t_j = tl.load(base + e - j, mask=alive & ((e - j) >= 0), other=-1); alive = alive & (t_j == s_j)`. The `other=-1` sentinel makes an out-of-range load fail the equality automatically. Phase 2 is guarded by `if tl.max(alive.to(tl.int32)) > 0:` and tracks `match = tl.where(alive, NMIN, 0)` incremented while `longer` holds; the suffix-side load there uses `other=-2`, a *different* sentinel from the history side's -1, so the two out-of-range conditions can never spuriously compare equal.

**Why they needed it.** Header claim: "An nmin-token reject test runs before any candidate is extended, so the scan costs a fraction of a millisecond whatever the batch size." Also in lookup.py's docstring: "Cost is one pass over the request's history ... with an NMIN-token reject test before any candidate is extended."

**llama.cpp — not applicable.** This exists to make a linear scan over the whole history affordable on a GPU. llama.cpp's lookups are O(1) hash probes (ngram-mod: one multiply-accumulate over n tokens then a modulo, then a single array read) or a hash-index lookup with a min_hits gate. There is no candidate sweep to prune, and process() for all four ngram impls is a stub returning true, so they cost nothing at prefill either.

**Evidence (llama.cpp):** `common/ngram-mod.cpp:15-40` · `common/ngram-map.cpp:401-404`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Padding rows default `take_flags` to 1 = "no objection"
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:681-687`

**What it does.** The fuse kernel writes 1 into `take_flags[req]` as its very first action, before the `req_state < 0` early return, so an inactive/padding row never vetoes a decision that is only taken when every active request agrees.

**Mechanism.** `tl.store(take_flags_ptr + req, 1)` unconditionally, then `if tl.load(idx_mapping_ptr + req * idx_mapping_stride) < 0: return`, then `tl.store(take_flags_ptr + req, 0)` for live rows before the real flag is computed at the end.

**Why they needed it.** Verbatim: "1 = 'no objection': padding rows must not veto a batch-wide decision that is taken only when every active request wants the long block." The write-1-then-overwrite-0 ordering is what makes the guarantee hold across the early return.

**llama.cpp — not applicable.** A device-kernel padding-row hazard. With -np 1 there is one slot and one sequence; the per-seq vectors are length 1 and there are no inactive rows that could veto a batch-wide decision. Even at higher -np, llama.cpp's equivalent decisions are host-side loops over live slots, not fixed-width kernels over padded batches.

**Evidence (llama.cpp):** `tools/server/server-context.cpp:1224-1226` · `common/speculative.cpp:2201-2209`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None on this box.

### `has_tail` flag: this request has something to put in the tail
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:709-721`

**What it does.** The fuse kernel exports a per-request boolean saying only that the lookup has a long-enough match with enough tokens left to fill the tail. It deliberately does NOT decide whether the long block is worth its step time; that is the host controller's job.

**Mechanism.** `has_tail = (match_len >= long_min) & (valid > draft_block) & take_tail`, stored as int32 into `take_flags_ptr + req`. Also `tl.atomic_add(hits_ptr, take_head.to(tl.int64))` maintains a global hit counter.

**Why they needed it.** The separation of concerns is argued twice. On why the decision cannot live in the kernel: "this kernel runs inside the draft model's captured CUDA graph, where host-side state is frozen at capture time." On why match length alone is not the criterion: "Match length alone is a poor predictor -- measured, it needs a threshold of 22 before it stops losing on prose, by which point it has given up most of the win on copy work" and "a 6-token match can carry fifteen correct tokens and a 12-token one none."

**Their numbers.** A match-length-only gate needs a threshold of 22 tokens before it stops losing on prose — at which point most of the win on copy work is gone. Illustrative counterexample given: a 6-token match carrying fifteen correct tokens vs a 12-token match carrying none.

**llama.cpp — not applicable.** The problem this solves — getting a decision OUT of a captured CUDA graph to the host — does not exist in llama.cpp, where the entire speculator including the draft-length decision is host C++ (common_speculative_draft is called from the server loop under queue_tasks.yield_to_queue). A llama.cpp implementation would just compute the condition in the same function that acts on it. Only the empirical claim transfers.

**Evidence (llama.cpp):** `common/speculative.cpp:2710-2756` · `tools/server/server-context.cpp:2936-2946`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** The negative result is the value: match length alone needed a threshold of 22 before it stopped losing on prose. Worth remembering if anyone here proposes gating a llama.cpp lookup on match length.

### One-step-stale flag read via a pinned async device-to-host copy
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:313-330` · `patches/dflash2-lookup-drafting.patch:388-396`

**What it does.** Reads the previous step's `take_flags` out of a pinned CPU buffer that was filled non-blockingly during the previous step, deliberately accepting one step of staleness in exchange for never synchronising the device.

**Mechanism.** `self._flags_cpu = torch.zeros(max_num_reqs, dtype=torch.int32, device="cpu").pin_memory()`. Each step: first consume last step's landed copy (`want = bool(self._flags_n and self._flags_cpu[:self._flags_n].all())`), then start the next one (`self._flags_cpu[:num_reqs].copy_(fused, non_blocking=True); self._flags_n = num_reqs`). Read-then-launch ordering is what makes the staleness exactly one step.

**Why they needed it.** Verbatim: "Reading it synchronously (`.item()`) is a device synchronise on every decode step and measured 5% -- more than the long block itself is worth on most work. One step of staleness costs a short step at the start of a copy run and a long one at its end." Note this reverses an earlier judgement recorded in the same repo's header for point 2, which says the flags "measured as not delivering the flags in time on this path" — the pinned copy is the version that shipped.

**Their numbers.** A synchronous `.item()` read of the flags measured 5% of step time on every decode step.

**llama.cpp — not applicable.** The 5% cost vLLM was avoiding is the cost of a device synchronise to read a flag a kernel wrote. llama.cpp's speculator state lives in host memory and is written by host C++ in the same thread that reads it, so a controller here reads current values with no sync and no staleness. The pinned-copy trick has nothing to buy.

**Evidence (llama.cpp):** `tools/server/server-context.cpp:2936-2946` · `common/speculative.cpp:2710-2756`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None — and this is a case where llama.cpp's architecture is simply cheaper.

### CHEAP_CTX: take the long block unconditionally below a context threshold (shipped off)
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:294-299` · `patches/dflash2-lookup-drafting.patch:381-386`

**What it does.** An early return in the controller that takes the full verify block whenever the request's context is shorter than `VLLM_DFLASH2_LOOKUP_CHEAP_CTX`, on the theory that an extra verify position is nearly free at short KV lengths. Default 0 = disabled.

**Mechanism.** `if self.draft_max_seq_len <= self._lookup_cheap_ctx: return self.num_speculative_tokens` placed before the flag logic, so it bypasses both the unanimity test and STICKY.

**Why they needed it.** This is the most interesting negative result in the slice. Verbatim: "Off by default: it measured +8% at C1 under one memory configuration and -13% under the one that ships (4 request slots, 56k), because what the extra positions cost depends on the paged-attention layout as much as on the KV length." The mechanism's own premise — cost is a function of KV length — is what the measurement falsified.

**Their numbers.** +8% at C1 under one memory configuration, -13% under the shipping configuration (4 request slots, 56k context). Supporting figure: an extra verify position costs +6% per step at 1.5k of context against +27% at 25k.

**llama.cpp — not applicable.** vLLM's negative result was that the cost of an extra verify position depends on the memory layout as much as on KV length: +8% at C1 under one configuration, -13% under the shipping one. llama.cpp has no CHEAP_CTX flag and no per-request context-based block choice, so the technique is not-applicable as written. But the lesson is directly instantiable here, and sharper: llama.cpp's cost cliff is not smooth in KV length at all, it is a discrete kernel switch at Q->ne[1] == 3, after which the dominant term is a full-cache F16 expansion charged to the compute buffer. Anyone reasoning about 'what does one more draft token cost on this box' from KV length alone will get it wrong in exactly the way the vLLM measurement warned about.

**Equivalent here:** no such flag; but the underlying question has a sharp llama.cpp-specific answer

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:469` · `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963` · `ggml/src/ggml-cuda/fattn-common.cuh:1022-1084` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** The most useful transfer in the slice, and it is a warning plus a free lunch. On Ada with -ctk q4_0 -ctv q4_0, the VEC kernel is chosen only when Q->ne[1] <= 2. Any speculative step with 2 or more draft tokens (1 sampled + 2 draft = 3 query tokens) runs MMA_F16, which dequantises the ENTIRE K and V cache for that layer to F16 on every call. Their profile is already always on that path: ngram-mod drafts 64, and draft-dflash at the default n_max=3 gives 4 query tokens. Consequence: the marginal cost of MORE draft tokens is small, because the dequant is O(n_kv) and independent of Q->ne[1]. That is a concrete reason to expect raising --spec-draft-n-max from 3 toward 15 to be cheap here — a hypothesis, to be measured paired within a round.

### `idx_mapping_stride`: reading a per-(request,step) mapping as per-request
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:483` · `patches/dflash2-lookup-drafting.patch:500` · `patches/dflash2-lookup-drafting.patch:585` · `patches/dflash2-lookup-drafting.patch:780-782`

**What it does.** Lets the lookup and fuse kernels, which are one-program-per-request, read the DFlash speculators' `sample_idx_mapping`, which is stored one-entry-per-(request, step), by striding over it.

**Mechanism.** Both kernels load `idx_mapping_ptr + req * idx_mapping_stride`, and both are passed `idx_mapping_stride=self.draft_block`. Documented in `suffix_lookup`'s docstring: "idx_mapping maps batch position -> request-state index, read with idx_mapping_stride (the DFlash speculators keep it as sample_idx_mapping, one entry per (request, step))." Note the stride is `draft_block`, not `num_speculative_steps`, because the mapping buffer is sized by the drafter's block.

**Why they needed it.** Avoids materialising a second per-request mapping buffer, and keeps the `-1` sentinel semantics shared with every other DFlash2 kernel.

**llama.cpp — not applicable.** An indexing accommodation between two kernels with different program granularities. llama.cpp indexes per-sequence state directly by seq_id in host vectors (impl_last[seq_id], dparams[seq_id]) and with -np 1 those are length 1. There is no flattened (request, step) buffer to stride over.

**Evidence (llama.cpp):** `common/speculative.cpp:2201-2209` · `tools/server/server-context.cpp:1224-1226`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### `VLLM_DFLASH2_LOOKUP_SEARCH`: bound how far back the scan reaches
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:300` · `patches/dflash2-lookup-drafting.patch:600`

**What it does.** Optionally limits the history scan to the most recent N tokens, trading match quality for scan cost. Defaults to `1 << 30`, i.e. the entire history.

**Mechanism.** `lo = tl.maximum(NMIN - 1, total_len - search_max)` sets the lower bound of the candidate-end sweep. `NMIN - 1` is the floor because a match ending earlier than that cannot have NMIN tokens behind it.

**Why they needed it.** No reason stated for the default; the header's cost argument ("a fraction of a millisecond whatever the batch size") rests on the NMIN reject test rather than on a bounded window, which is presumably why the default is unbounded.

**llama.cpp — not applicable.** There is no scan to bound: ngram-mod is an O(1) hash probe regardless of history length. The nearest thing llama.cpp has is a capacity bound rather than a window bound — a fixed 4*1024*1024-entry table (16 MiB, a literal with no flag, shared across sequences) that auto-resets when occupancy exceeds 0.25. So history is forgotten in bulk on a resize event rather than trimmed to a window, and neither behaviour is tunable.

**Evidence (llama.cpp):** `common/ngram-mod.cpp:15-40` · `common/speculative.cpp:1914` · `common/speculative.cpp:1952-1957`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None.

### Requires --no-async-scheduling to feed draft ids back to the scheduler
**Where (theirs):** `patches/dflash2-lookup-drafting.patch:35-37`

**What it does.** A deployment constraint, not a code change: the adaptive verify length depends on the scheduler seeing the draft token ids, which vLLM only does on the synchronous scheduling path.

**Mechanism.** Stated in the patch header; the flag is set by `single-user/start_qwen.sh` (not in this slice).

**Why they needed it.** Verbatim: "vLLM only feeds draft token ids back to the scheduler on the synchronous scheduling path, so this needs --no-async-scheduling (single-user/start_qwen.sh sets it; measured cost at batch 1 is under 1%)."

**Their numbers.** Cost of --no-async-scheduling measured at under 1% at batch 1.

**llama.cpp — not applicable.** llama.cpp's server has no async scheduling path to disable. Drafts are produced in-band: the slot's draft params are filled, then common_speculative_draft runs under queue_tasks.yield_to_queue, then the same iteration lays the draft into the batch. The scheduler and the speculator are the same loop, so the visibility this deployment constraint buys is unconditional here.

**Evidence (llama.cpp):** `tools/server/server-context.cpp:2955-2960` · `tools/server/server-context.cpp:2936-2946`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None — and llama.cpp does not pay the sub-1% cost either.
