# bench/*.py + bench/*.sh, single-user/start_qwen.sh, batch/start_qwen.sh, verify.sh, Dockerfile, docker-compose.yml, docker/* — measurement method and serving configuration
**75 techniques.** 3022 source lines across 26 files.
Files read: `bench/labd_accept.py` · `bench/demo_render.py` · `bench/labd_bench.py` · `bench/test_lookup_kernels.py` · `bench/conc_ladder.py` · `bench/labd_soak.py` · `bench/spec_attn_ctx_scan.py` · `bench/quality_battery.py` · `bench/act_calib.py` · `bench/demo_capture.py` · `bench/api_smoke.py` · `bench/test_spec_decode_attn.py` · `bench/bugb_sweep.py` · `bench/make_long_corpus.py` · `bench/tune_gdn.py` · `bench/run_benchmarks.sh` · `bench/real_rep.sh` · `bench/prompts_real.jsonl` · `single-user/start_qwen.sh` · `batch/start_qwen.sh` · `verify.sh` · `Dockerfile` · `docker-compose.yml` · `docker/entrypoint.sh` · `docker/prepare.sh` · `docker/requirements.txt`
> **What the reader could not see:** Every file named in the slice exists and was read in full. Absences worth flagging: (1) `bench/` contains no results directory and no committed result JSONs — every harness writes to `~/bench/results/` outside the repo, so none of the numbers quoted in comments can be re-derived from the tree. (2) `~/bench/labd_corpus.txt` (the frozen corpus every LABD harness depends on) is NOT in the repo; it is generated on first run from `~/qwen-serving/*.md`, which means "frozen" is per-machine, not per-repo — two boxes running labd_accept.py do not share a corpus. (3) Several harnesses hardcode `~/qwen-serving/api_key.txt` and `http://127.0.0.1:18020` with no override (labd_accept.py:91, labd_bench.py:28-29, labd_soak.py:46-47, demo_capture.py:27), while the newer ones (conc_ladder.py, api_smoke.py, bugb_sweep.py, quality_battery.py) resolve the key from the repo root and honour `PORT` — the older three would fail outside the author's home dir. (4) `bench/test_lookup_kernels.py:3` documents itself as `python test_lookup_v2.py`, a filename that does not exist. (5) The docs referenced constantly from serving comments (`docs/gotchas.md`, `docs/long-context.md`, `docs/docker.md`) are in `docs/` and outside this slice, so the gotcha numbers cited (4, 10, 14, 37) could not be verified here. (6) `single-user/start_qwen.sh` has no `--no-spec-baseline` flag, yet `bench/demo_capture.py:11` instructs the operator to invoke it that way to capture the baseline lane — the demo's stock-vLLM lane cannot be produced by the documented command.

---

## EXISTS, NEVER SET — 17

### Prefix-resync teacher forcing (acceptance without trajectory drift)
**Where (theirs):** `bench/labd_accept.py:1-30` · `bench/labd_accept.py:253-273`

**What it does.** Measures speculative-decode acceptance (tokens per forward step) for a configuration while holding the generated token sequence itself fixed, so two configurations are scored on predicting the *same* text. A target continuation T is captured once and frozen on disk; the replay walks it in chunks of C tokens, where request j sends `prompt = P + T[:j*C]` as raw token ids with `max_tokens=C`. Every chunk therefore begins from a byte-identical prompt on every server compared.

**Mechanism.** Loop at labd_accept.py:253 iterates `off in range(0, len(target), CHUNK)`; for each chunk it snapshots /metrics (labd_accept.py:255), calls `generate(prompt_ids + target[:off], len(want_ids))` (labd_accept.py:256), snapshots again, and diffs the four `vllm:spec_decode_*` counters. Forced tokens enter through the *prompt*, which vLLM writes into `req_states.all_token_ids` at offset 0 with `total_len = prefill_len` — the exact array `suffix_lookup` scans — so a forced prefix is indistinguishable from a produced one to both the drafter and the lookup (labd_accept.py:20-26).

**Why they needed it.** "labd_bench.py asks two servers the same question and compares tokens/step. That only works if both write the same text. On 'quote and explain' they do not -- greedy argmax ties break differently once the batch shape changes -- and the 10% swings it reported were trajectory divergence, not speed." (labd_accept.py:3-6)

**Their numbers.** The failure it replaces: 10% swings in labd_bench.py reported as speed differences were trajectory divergence. Defaults: --ctx 20000, --max-tokens 512, --chunk 128, --block 7.

**llama.cpp — EXISTS, NEVER SET.** Every primitive exists and none is used in the current profile. tokenize_input_prompts accepts an array of integers as the prompt (server-common.cpp:993-995 via json_is_array_and_contains_numbers), so a frozen target can be pushed in as ids; return_tokens returns the produced ids; the three spec counters are exposed on /metrics. A forced prefix is indistinguishable from a produced one to ngram-mod, which hashes the last n_match tokens of whatever is in the context (speculative.cpp:1887-2059). Two llama.cpp-specific caveats the vLLM harness does not have: [TAG_PROMPT_LOGITS] decrements n_past by 1 on a fully cached prompt (server-context.cpp:3313-3318) so each chunk always re-evaluates one token, and dp.n_max truncates a draft near the context edge (server-context.cpp:441-460).

**Equivalent here:** POST /completion with `prompt` as a raw int array + `return_tokens: true`, scored off llamacpp:spec_decode_* counters (--metrics)

**Evidence (llama.cpp):** `tools/server/server-common.cpp:993-995` · `tools/server/server-schema.cpp:34` · `tools/server/server-context.cpp:1774` · `tools/server/server-task.cpp:1551-1561` · `tools/server/server-context.cpp:3313-3318`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High. This project's own rule is that effects below 13.6 % are noise across boots; teacher forcing removes trajectory divergence entirely, so a paired ngram-mod vs draft-dflash comparison stops being a text-difference measurement. It would let the +34.7 % and +48.5 % figures be re-taken on identical token sequences. The harness is a new script (~200 lines), no llama.cpp change.

### Exact-token-id replay with a hard refusal to guess
**Where (theirs):** `bench/labd_accept.py:27-30` · `bench/labd_accept.py:156-163` · `bench/labd_accept.py:191-195`

**What it does.** The frozen target is captured and replayed as raw integer token ids, never as text, so nothing passes through a detokenize/retokenize round trip that could resegment the sequence. If the server does not return ids, the harness aborts rather than degrade.

**Mechanism.** Capture uses `logprobs:0` + `return_tokens_as_token_ids:true` (labd_accept.py:169), which renders each token as the string `"token_id:1234"`; `ids_from_logprobs` (labd_accept.py:156) strips the 9-char prefix and returns None on any token that does not match. `generate` raises `SystemExit("server did not honour return_tokens_as_token_ids -- cannot pin an exact token sequence, refusing to guess")` (labd_accept.py:193).

**Why they needed it.** "Exact ids, not text. The target is captured with `logprobs:0` + `return_tokens_as_token_ids:true`, so it is the model's own token ids, and it is replayed as ids. Nothing goes through a detokenize/retokenize round trip that could resegment it." (labd_accept.py:27-30)

**llama.cpp — EXISTS, NEVER SET.** llama-server has a first-class field for exactly this and it defaults to false, so nobody in this profile has ever set it. server-task.h:54 declares return_tokens = false; server-schema.cpp:34 registers it; server-context.cpp:1774 fills the array; it is emitted as `tokens` in both the non-OAI final result (server-task.cpp:346) and the per-chunk result (server-task.cpp:1056). Combined with prompt-as-ids input this closes the loop with no detokenise/retokenise round trip anywhere.

**Equivalent here:** `return_tokens: true` (server request field) → `tokens` array in the completion result

**Evidence (llama.cpp):** `tools/server/server-task.h:54` · `tools/server/server-schema.cpp:34` · `tools/server/server-context.cpp:1774` · `tools/server/server-task.cpp:346` · `tools/server/server-task.cpp:1056`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High, and cheaper here than in vLLM: llama.cpp hands back real ids, so there is no "token_id:1234" string-prefix hack to break. The refusal discipline still applies — if `tokens` comes back empty the harness must abort rather than fall back to re-tokenising text.

### Dirty-chunk detection: off-trajectory chunks are excluded and counted, and firstbad is located
**Where (theirs):** `bench/labd_accept.py:257-273` · `bench/labd_accept.py:286-291`

**What it does.** Compares each chunk's returned ids against the frozen target elementwise; a chunk that diverged is excluded from the headline but counted, and the exact token index of the first divergence is reported.

**Mechanism.** `clean = got == want_ids` (labd_accept.py:258); on the first dirty chunk `c['first_bad'] = off + next((i for i,(g,w) in enumerate(zip(got, want_ids)) if g != w), len(got))` (labd_accept.py:262). Counter deltas are accumulated only `if clean or flag('--keep-dirty')` (labd_accept.py:264). Output line prints `chunks=clean/total` and `firstbad=` per task.

**Why they needed it.** "the returned ids are compared against T[j*C : j*C+C] so a chunk that went off-trajectory is detected exactly rather than silently averaged in" (labd_accept.py:16-17). "Divergences show up as dirty chunks instead of as a fake speed difference." (labd_accept.py:71-72)

**llama.cpp — EXISTS, NEVER SET.** Nothing in llama.cpp does this; the ingredients (exact ids out, exact ids in) exist and are unused. Because llama.cpp never fills spec_dists outside DFlash2 (speculative.cpp:1238-1258), the residual/rejection accept path never runs with ngram-mod and the greedy prefix-match rule is used even at temperature 1.0 — so divergence is attributable to the sampler RNG, not to the drafter.

**Equivalent here:** compare the `tokens` array from return_tokens against the frozen target elementwise

**Evidence (llama.cpp):** `common/sampling.cpp:692-720` · `common/speculative.cpp:1238-1258` · `tools/server/server-context.cpp:3825-3831` · `tools/server/server-context.cpp:1774`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High and cheap once return_tokens is on. Note llama.cpp speculation is exact by construction — the greedy verifier accepts draft[i] only when it equals the target's own sample (sampling.cpp:692-720) — so a dirty chunk here means a tie broken differently or a genuine state bug, never "the speculator changed the text". That makes a dirty chunk a stronger signal here than in vLLM.

### slots/step as an interpolation that reveals long-block scheduling fraction
**Where (theirs):** `bench/labd_accept.py:274-283`

**What it does.** Derives what fraction of decode steps actually took the long verify block, from a metric that is not directly reported, by reading where mean draft-slots-per-step lands between the short block length and the configured maximum.

**Mechanism.** `slots = c['slots']/steps` from the `num_draft_tokens_total` delta; `long_frac = max(0.0, (slots - BLOCK) / max(NSPEC - BLOCK, 1))` (labd_accept.py:280). BLOCK is the drafter's own block (default 7), NSPEC the server's `num_speculative_tokens`.

**Why they needed it.** "slots/step sits at draft_block on short steps and at num_speculative_tokens on long ones, so where it lands between the two is the fraction of steps that took the long block. This is the number the 'never scheduled' claim should be checked against." (labd_accept.py:277-279)

**llama.cpp — EXISTS, NEVER SET.** The instrument exists, is compiled in unconditionally (gen_perf is a hardcoded `const bool = true`, speculative.cpp:156), and is gated only behind log verbosity — and the default verbosity is INFO (3), so TRC lines are invisible today. Credit goes only to the impl that produced the draft; every other impl gets accept(..., is_other=true) with its counters untouched, so the attribution is clean.

**Equivalent here:** common_speculative_print_stats — the per-implementation SPC_TRC statistics line, visible at LOG_TRC (-lv 4 / -v)

**Evidence (llama.cpp):** `common/speculative.cpp:2829-2872` · `common/speculative.cpp:139-161` · `common/speculative.cpp:156` · `common/speculative.cpp:2796-2801` · `common/log.h:24-32`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High and better than the vLLM technique. llama.cpp does not need an interpolation: with --spec-type draft-dflash,ngram-mod it records which impl produced each draft in impl_last[seq_id] and prints per-impl n_gen_drafts / n_acc_drafts / n_gen_tokens / n_acc_tokens plus t_draft_us. That directly answers which of the two speculators earned the +48.5 %, which the current INFO-level `draft acceptance` line cannot.

### Batch-mode rows measured with --ignore-eos, cohort rows without
**Where (theirs):** `bench/run_benchmarks.sh:47-52` · `bench/run_benchmarks.sh:76-81`

**What it does.** Forces the throughput rows to generate the full requested output length so aggregate tok/s is not a function of how early the model chose to stop, while leaving the realistic cohorts free to stop naturally.

**Mechanism.** `$B --dataset-name random --ignore-eos --random-input-len $1 --random-output-len $2 --num-prompts 256 --max-concurrency 64` for the "128 512" and "256 256" shapes; also on the 1x100k/256 and 4x60k/1024 long rows. The cohort loop at line 59 omits `--ignore-eos`.

**Why they needed it.** A throughput number over variable-length completions measures stopping behaviour as much as speed; the long-context rows especially would otherwise be dominated by early stops.

**Their numbers.** Batch rows: 128in/512out and 256in/256out, 256 prompts at 64 concurrent. Long rows: 1×100k prompt/256 out, and 4×60k/1024 at concurrency 4.

**llama.cpp — EXISTS, NEVER SET.** The flag exists at both levels and is off by default. Implementation detail worth knowing: --ignore-eos is not a stop-condition change, it is -INFINITY logit biases on every EOG token precomputed at model load (common.cpp:1345-1358), and it is silently downgraded to false if the vocab has no EOS.

**Equivalent here:** --ignore-eos (CLI) / `"ignore_eos": true` (per request), paired with n_predict

**Evidence (llama.cpp):** `common/arg.cpp:2235-2256` · `common/common.cpp:1340-1358` · `tools/server/server-schema.cpp:430-480`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate here, high if throughput rows are ever added. At -np 1 the effect is smaller than in a 64-way batch, but any fixed-length decode row still measures stopping behaviour unless ignore_eos is set. Cheap to adopt: it is a per-request field, so the same server serves both kinds of row.

### Prefill matrix as a separate opt-in sweep
**Where (theirs):** `bench/run_benchmarks.sh:65-75`

**What it does.** Measures prefill throughput in isolation by requesting exactly one output token, across a length × concurrency grid.

**Mechanism.** `pf()` runs `--random-output-len 1 --random-input-len $LEN` and computes `TotalInputTokens/BenchmarkDuration`. Grid: 1024 at conc 1/4/16 (16/32/64 prompts), 4096 at 1/4/16 (8/16/32), 16384 at 1/4/8 (4/8/16), 65536 at 1/2 (2/4), 102400 at 1 (2 prompts).

**Why they needed it.** Prompt count is scaled down as length rises so each cell takes comparable wall time; output-len 1 removes decode from the measurement entirely.

**Their numbers.** Grid spans 1k to 102,400 input tokens; concurrency 1-16 at short lengths, 1-2 at 64k-100k.

**llama.cpp — EXISTS, NEVER SET.** The tool ships in this tree (tools/llama-bench/llama-bench.cpp) with exactly the axes the vLLM prefill matrix sweeps, including a depth axis vLLM's does not have. Since llama-bench sets n_ctx = n_prompt + n_gen + n_depth itself (line 1281), a grid there is not comparable to a server run at a fixed -c — do not mix the two tables.

**Equivalent here:** tools/llama-bench with -p (n_prompt), -n (n_gen), -pg, and -d/--n-depth

**Evidence (llama.cpp):** `tools/llama-bench/llama-bench.cpp:568-599` · `tools/llama-bench/llama-bench.cpp:1281` · `tools/llama-bench/llama-bench.cpp:1568`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate. llama-bench already does the length × depth grid and reports pp/tg separately, and -d lets you measure at a KV depth rather than from empty — which is the number that actually matters at 12 GB. The limit: llama-bench builds its own llama_context and does not go through llama-server, so no speculation, no slots, no prompt cache. It answers prefill questions only.

### Shared-versus-distinct prompt switch to model prefix-cache-friendly clients
**Where (theirs):** `bench/conc_ladder.py:8-9` · `bench/conc_ladder.py:44-49`

**What it does.** Lets the same ladder run with per-stream unique prompts (the default, defeating prefix cache) or identical prompts (what a prefix-cache-friendly client looks like).

**Mechanism.** `make_prompt(i)` prepends `f"Document {i} (revision {i*7+3}). "` unless `--shared`; body is a fixed filler string repeated 260 times (~4k tokens) so length is identical in both modes and only the prefix differs.

**Why they needed it.** Prefix caching changes throughput by an order of magnitude on shared-prefix workloads (batch/start_qwen.sh:63-69), so a concurrency ladder that does not state which case it measured is uninterpretable.

**Their numbers.** ~4k-token prompts (FILLER × 260).

**llama.cpp — EXISTS, NEVER SET.** cache_prompt defaults to enabled (common.h:611) and is exposed per request (server-schema.cpp:31-32); with it off, n_past is forced to 0 and the whole prompt is re-decoded (server-context.cpp:3197-3200). --cache-ram defaults to 8192 MiB of host RAM for the cross-request cache, so even a fresh slot may be served from RAM unless -cram 0. Note that --cache-reuse (non-prefix chunk shifting) is a separate feature and is unreachable on an M-RoPE model.

**Equivalent here:** per-request `"cache_prompt": false` (or --no-cache-prompt) to defeat in-slot prefix reuse; -cram 0 to disable the RAM prompt cache

**Evidence (llama.cpp):** `common/common.h:611` · `tools/server/server-schema.cpp:31-32` · `tools/server/server-context.cpp:3197-3200` · `common/common.h:615` · `src/llama-kv-cache.cpp:1176-1178`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High for a coding agent, which is the workload here. Turn 2+ on the same file is almost entirely a prefix hit; a benchmark that never states which case it measured is uninterpretable. Both switches exist per-request or per-launch and neither is set today. cache_prompt is on by default, so every current number is implicitly the cache-friendly case.

### Quality battery: three-domain perplexity plus GSM8K, run after every kernel/quant change
**Where (theirs):** `bench/quality_battery.py:1-13` · `bench/quality_battery.py:39-81` · `bench/quality_battery.py:99-104`

**What it does.** Catches "benchmarks great, outputs garbage" in about a minute by measuring perplexity over three deliberately different distributions (English wikitext-2 test, Danish fineweb-2 test, vLLM's own Python source) and GSM8K exact-match.

**Mechanism.** Perplexity: 1,200-character windows sent to /completions with `max_tokens:1, prompt_logprobs:0, echo:False`; the returned `prompt_logprobs` list (first entry None) is summed and `exp(-Σlogp/n)` taken per language and pooled (quality_battery.py:59-81), 2-way threadpool. GSM8K: 200 test questions, thinking off, greedy, prompt suffixed with "Solve step by step, then give the final answer as 'Final answer: <number>'", answer extracted by regex with a last-number fallback, 32-way threadpool (quality_battery.py:87-103).

**Why they needed it.** "Catches 'benchmarks great, outputs garbage' in a minute — run it after every kernel/quant change." (quality_battery.py:2-3). run_benchmarks.sh:13 chains it: "a fast server that emits garbage is worth nothing."

**Their numbers.** 40 windows per domain × 1,200 chars; GSM8K n=200 default. Requires GPU_UTIL=0.93 because prompt_logprobs needs memory headroom (quality_battery.py:11-12, gotcha 10). Reference perplexity figures quoted elsewhere: 8.045 vs 8.046 fp32-vs-fp16 recurrent state; +2.2% PPL for full int8 activations, +0.9% for gate_up only, +0.6% for the int4-GPTQ MTP/lm_head, +0.2% for KVarN.

**llama.cpp — EXISTS, NEVER SET.** tools/perplexity exists in this tree and computes PPL directly rather than through the server, which sidesteps the fact that llama-server has no prompt_logprobs/echo (n_probs covers generated tokens only). The rotation and its two startup log lines (`attn_rot_k = %d` / `attn_rot_v = %d`) are the thing to check alongside the number, and LLAMA_ATTN_ROT_DISABLE=1 gives you the A/B.

**Equivalent here:** tools/perplexity (llama-perplexity) — offline PPL over a corpus; no server-side prompt_logprobs equivalent

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:308-338` · `src/llama-kv-cache.cpp:192-196` · `tools/server/server-schema.cpp:179-181`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Very high for this profile specifically. UD-IQ2_XXS is a 2-bit quant with -ctk q4_0 -ctv q4_0 on top, and the Hadamard rotation that mitigates quantised-KV damage is applied silently and only when n_embd_head_k % 64 == 0 — you get no error if it does not engage, just a worse model. A held-out PPL run over a code corpus is the only thing that would catch a quality collapse that no throughput number moves. The GSM8K half would need writing; the PPL half ships.

### Attention kernel correctness checked against an fp32 reference AND against FA2
**Where (theirs):** `bench/test_spec_decode_attn.py:34-49` · `bench/test_spec_decode_attn.py:64-77`

**What it does.** Validates the split-KV spec-decode attention kernel by comparing it to an explicit fp32 einsum reference, and prints FlashAttention-2's error against the same reference in the same line, so the tolerance can be judged relative to a kernel already trusted.

**Mechanism.** `ref()` gathers the paged KV through the block table, repeat_interleaves the 4 KV heads to 24 query heads, does `einsum("qhd,khd->hqk")*scale`, applies a causal mask built from `qpos = arange(L-q_len, L)` against `kpos = arange(L)` — i.e. the query block sits at the *end* of the sequence, as in a verify step — then softmax and `einsum("hqk,khd->qhd")`. Test prints `max|ours-ref|` and `max|FA-ref|` side by side, passing at err < 0.05.

**Why they needed it.** A verify step's query block is not at position 0, so an off-by-one in the causal mask is the natural bug; and an absolute error threshold is meaningless for bf16 unless you can see what a reference implementation's error is on the same shape.

**Their numbers.** 13 shape cases from kv=[432] q=5 up to kv=[25000] q=32 and kv=[1000] q=64, including ragged batches ([37,1000,4321], [433,431], [600,4000]) and block-size boundaries (432, 433/431 against BS=432). Timing grid: kv ∈ {1500, 4000, 25000, 60000} × q ∈ {8,16,32}. Model shapes: Hq=24, Hkv=4, D=256.

**llama.cpp — EXISTS, NEVER SET.** The harness ships and covers the op. Two llama.cpp facts make it worth running here: with q4_0 K/V and more than two query tokens the MMA_F16 path is taken, which dequantises the whole K and V cache to F16 into compute-buffer scratch on every call — a path with different numerics from the VEC path used at one query token. A speculative step and a plain decode therefore run different kernels on the same cache, and test-backend-ops is where that comparison belongs.

**Equivalent here:** tests/test-backend-ops — test_flash_attn_ext compares the CUDA op against the CPU reference at parameterised shapes, including quantised K/V types

**Evidence (llama.cpp):** `tests/test-backend-ops.cpp:7040-7138` · `tests/test-backend-ops.cpp:9863-9866` · `ggml/src/ggml-cuda/fattn.cu:469` · `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High if this repo ever patches the attention path, moderate otherwise. The existing cases already sweep hsk/hsv/nh/nr23/kv/nb and K/V type pairs, including asymmetric ones the CUDA build cannot run — so it doubles as a check on which pairs are actually supported in THIS binary. Adding cases at the exact shape a verify step produces (Q->ne[1] = 1+n_draft, head dim 128, q4_0/q4_0) is a config change, not new code.

### Offline context scan of the verify attention with four arms and no server
**Where (theirs):** `bench/spec_attn_ctx_scan.py:1-16` · `bench/spec_attn_ctx_scan.py:29-36` · `bench/spec_attn_ctx_scan.py:84-121`

**What it does.** Answers, before any KV-format work is started, three questions about the multi-query verify attention at real long-context lengths: what it costs per attention layer at 64k/128k/150k rather than the 25k it was tuned at; whether raising the hardcoded segment count is the whole win; and what vLLM's own Triton attention would do instead, both with its split-KV path gated off (as shipped) and forced open.

**Mechanism.** Allocates its own paged cache — no server needed, ~1.5 GB at the longest shape. Four arms per cell: `ours@{16,32,64}` (SpecDecodeAttention at three NUM_SEGMENTS values), `FA2` (flash_attn_varlen_func, fa_version=2), `triton2D` (stock `tua.unified_attention`), and `triton3D` (the same call with `seq_threshold_3D=1024, num_par_softmax_segments=32` and explicitly allocated `softmax_segm_output/max/expsum` buffers, spec_attn_ctx_scan.py:65-81). Both Triton arms are wrapped in try/except that records NaN and prints the exception type rather than aborting the scan.

**Why they needed it.** "its split-KV ('3D') path is gated off whenever max_seqlen_q > 1 (triton_unified_attention.py:1047), which every verify step is. Arm C measures that gate closed; arm D measures it forced open." (spec_attn_ctx_scan.py:10-13). "The split-KV kernel in patches/spec-decode-attn.patch was tuned at a 25k context with NUM_SEGMENTS hardcoded to 16." (spec_attn_ctx_scan.py:3-4)

**Their numbers.** KV lengths 25k/50k/64k/100k/128k/150k (or 25k/64k with --quick); q_len 8 and 16; segments 16/32/64. Shapes: Hq=24, Hkv=4, D=256, BS=480 — "the block size this model actually runs with" (spec_attn_ctx_scan.py:29). Timing: 5 warm-up iterations then 30 timed, CUDA events, reported in microseconds.

**llama.cpp — EXISTS, NEVER SET.** MODE_PERF exists (tests/test-backend-ops.cpp:478, selected at 1509) and test_flash_attn_ext is fully parameterised. Unlike the vLLM version there is no segment-count knob to sweep — llama.cpp chooses stream-k / parallel_blocks from occupancy at launch — so the arms reduce to kernel-by-shape and type-by-type, which is exactly what needs measuring here.

**Equivalent here:** tests/test-backend-ops MODE_PERF over test_flash_attn_ext at chosen kv/nb/type_KV

**Evidence (llama.cpp):** `tests/test-backend-ops.cpp:478` · `tests/test-backend-ops.cpp:1509` · `tests/test-backend-ops.cpp:7074` · `ggml/src/ggml-cuda/fattn-common.cuh:1022-1084` · `ggml/src/ggml-cuda/fattn.cu:469`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High and cheap. This answers, without a server and without touching the KV budget, the one question the map flags as the biggest hidden cost in the attention area: what quantised K/V actually costs on the MMA path at real depths, given that MMA dequantises the entire per-layer cache to F16 on every call at 2 bytes/element against q4_0's 0.5625. Sweeping kv at 16k/32k/64k and nb at 1, 2, 3 and 16 crosses the VEC→MMA boundary (Q->ne[1] <= 2 for quantised KV) and prices the flip directly.

### Per-token arrival-time capture with lanes recorded separately and replayed together
**Where (theirs):** `bench/demo_capture.py:1-19` · `bench/demo_capture.py:66-98` · `bench/demo_render.py:1-7`

**What it does.** Produces an honest side-by-side speed video on a machine with one GPU, by recording each configuration's token arrival times separately against a live server and replaying both at their true recorded speed.

**Mechanism.** demo_capture.py records `toks.append([round((now - first)*1000, 1), piece])` for every streamed content chunk — times relative to the first token, so the renderer never has to guess — plus ttft, decode_s and decode_tok_s per prompt. demo_render.py's `visible()` binary-searches the arrival list so that "at video time t a lane shows exactly the tokens whose recorded arrival was <= t" (demo_render.py:5-6). Every frame carries the stamp "recorded separately, replayed at true speed" (demo_render.py:169-171).

**Why they needed it.** "There is one GPU, so the two configurations cannot run at the same time. Each lane is captured separately against a live server and the video replays them together at their real recorded speed -- the timings are measured, the side-by-side is a replay." (demo_capture.py:3-5)

**Their numbers.** 3 prompts: chat, code, and a 25,000-token document reproduction (DOC_TOKENS=25000 at 2.9 chars/token, demo_capture.py:33-35).

**llama.cpp — EXISTS, NEVER SET.** timings_per_token defaults to false and is a per-request field, so it is a genuine never-set flag. The honest side-by-side discipline (record separately, replay at true speed, stamp the frame) transfers unchanged since there is still only one GPU.

**Equivalent here:** `"timings_per_token": true` puts the full timings block on every stream chunk; `return_tokens` gives ids per chunk

**Evidence (llama.cpp):** `tools/server/server-schema.cpp:20-21` · `tools/server/server-common.cpp:66-88` · `tools/server/server-context.cpp:1774`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Low for this repo's purposes — it is a presentation technique. If a demo is ever wanted, llama.cpp gives more than vLLM did: timings on every chunk rather than only at the end, so arrival times need not be reconstructed at all.

### Two different gpu-memory-utilization values, and the reason they differ
**Where (theirs):** `batch/start_qwen.sh:19-20` · `single-user/start_qwen.sh:50-52`

**What it does.** Sets 0.972 in batch mode and 0.93 in single-user mode, with the mechanism behind each named.

**Mechanism.** batch: `GPU_UTIL=${GPU_UTIL:-0.972}` for the fp8 path (0.93 for the two quantized-KV paths, batch/start_qwen.sh:44,48). single-user: `GPU_UTIL=${GPU_UTIL:-0.93}`.

**Why they needed it.** batch: "gpu-memory-utilization 0.972 is the sweet spot on a headless box (X/display holds ~220 MB; 0.98 fails the startup free-memory check)" (batch/start_qwen.sh:19-20). single-user: "0.93 here, NOT batch mode's 0.972: the DeltaNet workspace in the MTP decode path allocates beyond the startup memory profile (docs/gotchas.md, gotcha 4)." (single-user/start_qwen.sh:50-51)

**Their numbers.** 0.972 batch / 0.93 single-user; X/display holds ~220 MB; 0.98 fails the startup free-memory check.

**llama.cpp — EXISTS, NEVER SET.** The flag exists, is never set in this profile, and the map calls the 1 GiB default out explicitly as the direct lever. Two clamps to respect: the parser throws if you pass as many or more values than llama_max_devices(), and the value is in MiB and multiplied at parse time.

**Equivalent here:** -fitt / --fit-target MiB — the per-device margin --fit leaves free (default 1024 MiB)

**Evidence (llama.cpp):** `common/common.h:473` · `common/arg.cpp:2851-2874` · `common/fit.cpp:559-563` · `common/fit.cpp:56-57`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High and immediate. On a 12 GB card the default silently forfeits a full GiB before --fit even starts assigning layers, and lowering it is the single direct lever for more context or more offloaded layers. With a 6.77 GB model and ~9.5 GB free, moving -fitt from 1024 to, say, 384 hands roughly 640 MiB back to KV and compute. The risk is equally concrete and already documented in this repo: free VRAM at boot moves 9,326–10,732 MiB, and --fit measures free memory at that instant, so a tight margin will OOM on a bad boot. Sweep it, pick a value with headroom, then pin it.

### max-num-batched-tokens 2048 chosen against the KV pool, not against prefill speed
**Where (theirs):** `batch/start_qwen.sh:21-22` · `batch/start_qwen.sh:117` · `single-user/start_qwen.sh:300`

**What it does.** Caps the chunked-prefill batch at 2048 tokens on both launchers, deliberately below the value that would maximize prefill throughput.

**Mechanism.** `--max-num-batched-tokens 2048`.

**Why they needed it.** "max-num-batched-tokens 2048 beats 8192 here: bigger chunks inflate the profiled activation peak, which shrinks the KV/state page pool" (batch/start_qwen.sh:21-22). The mechanism is the memory profiler, not the kernels — the constant is set by a second-order effect.

**Their numbers.** 2048 vs 8192.

**llama.cpp — EXISTS, NEVER SET.** Both flags exist with defaults and both are silently clamped (n_batch capped at n_ctx, n_ubatch capped at n_batch, neither with a warning), so the effective values may already differ from what a script passes. The DFlash/DSpark path additionally forces n_batch and n_ubatch UP to n_parallel*(n_max+1) behind your back — the only place in the area that raises your batch — so a -ub sweep must be re-taken per --spec-type.

**Equivalent here:** -ub / --ubatch-size (default 512) and -b / --batch-size (default 2048)

**Evidence (llama.cpp):** `src/llama-context.cpp:595` · `src/llama-context.cpp:245-247` · `ggml/src/ggml-cuda/fattn.cu:534-568` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912` · `common/speculative.cpp:2418-2423`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High, and the same second-order mechanism applies. n_ubatch is the single knob that sizes the worst-case compute buffer: the reserve pass runs prompt processing at n_tokens = min(n_ctx, n_ubatch), and with quantised KV that reserve always takes the MMA path, which budgets an F16 dequant scratch for the ENTIRE K and V cache per layer at 2 bytes/element. Lowering -ub therefore shrinks the compute buffer, which leaves --fit more room for context or layers — a VRAM lever, not a speed lever. I have seen no evidence in the map that -ub has ever been swept for VRAM in this profile.

### Draft count reduced at long context because k=4 crashes on FlashInfer
**Where (theirs):** `single-user/start_qwen.sh:24-27` · `single-user/start_qwen.sh:73-87`

**What it does.** Uses 4 MTP drafts on the bf16/FlashAttention path and 3 on the fp8/FlashInfer and KVarN paths, because of a specific concurrency crash.

**Mechanism.** `if CTX=fast: DRAFT_TOKENS=4` else `DRAFT_TOKENS=3` (single-user/start_qwen.sh:75, 80, 85).

**Why they needed it.** "CTX=long: fp8 KV via FlashInfer, 150k context, 3 drafts (k=4 crashes on FlashInfer as soon as one request finishes while another is mid-generation, vLLM 0.27.1); the split-KV attention patch is bf16-KV only, so ~90/98 tok/s." (single-user/start_qwen.sh:25-27)

**Their numbers.** CTX=long: ~90 tok/s default sampling / ~98 greedy, versus CTX=fast's ~114/~124.

**llama.cpp — EXISTS, NEVER SET.** The default is stated at common.h:325 and the clamp at speculative.cpp:990-996 warns only when your request EXCEEDS the block size — it says nothing when you are far under it. This repo already knows the depth-dependence lesson (draft-mtp +81 % at 16K, −71 % at 131,072), so the sweep must be run at the depth the coding agent actually works at, and the verdict must not be transferred to another depth.

**Equivalent here:** --spec-draft-n-max (default 3) and --spec-draft-n-min (default 0)

**Evidence (llama.cpp):** `common/common.h:325` · `common/arg.cpp:4076-4085` · `common/speculative.cpp:966-996` · `common/speculative.cpp:1181`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Very high — likely the biggest single unclaimed win on this list. --spec-draft-n-max defaults to 3, and the DFlash block-size clamp only ever LOWERS it: with a stock 16-wide sidecar the largest usable value is 15 (16 for DSpark with anchor sampling). So unless the profile passes --spec-draft-n-max explicitly, today's draft-dflash run is proposing 3 tokens per step out of a block the sidecar was trained to fill 16 wide. The +34.7 % may be a fraction of what the artifact can do. Sweep n-max 3→15 and read mean len per step. The costs are real and priced in the next verdict.

### MTP module and lm_head requantized to int4 with GPTQ calibrated on hidden states
**Where (theirs):** `single-user/start_qwen.sh:14-16` · `single-user/start_qwen.sh:31-33` · `single-user/start_qwen.sh:44-47`

**What it does.** Shrinks the per-draft and per-verify weight traffic by requantizing the MTP draft module and the lm_head to int4, calibrated on the model's own hidden states, and ships it as a separate "fast" model directory that the launcher auto-selects when present.

**Mechanism.** `if [ -z "$MODEL" ] && [ -d "$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast" ]; then MODEL=...-fast; fi` (single-user/start_qwen.sh:44-46), falling back to the base dir.

**Why they needed it.** "the MTP module and lm_head requantized to int4 with GPTQ calibrated on the model's hidden states (drafter/): 850 -> 215 MB per draft, 1.27 -> 0.65 GB lm_head per verify, +0.6% perplexity, acceptance unchanged" (single-user/start_qwen.sh:14-16)

**Their numbers.** 850 → 215 MB per draft; 1.27 → 0.65 GB lm_head per verify; +0.6% perplexity; acceptance unchanged. Base dir (int8 lm_head/MTP) gives ~108/107 tok/s with the shipped draft vocab, vs the fast variant's ~114/124 — verify.sh:95 states "single-user mode is ~15% slower without it".

**llama.cpp — EXISTS, NEVER SET.** Judged on llama.cpp's own quantiser. Two precedence traps if this is ever attempted: --output-tensor-type and --token-embedding-type are checked FIRST and return immediately, skipping tensor_type_fallback, so a block-size-incompatible choice is not clamped for those two the way it is everywhere else; and --leave-output-tensor is an exact string compare on "output.weight" and does not protect a tied token_embd acting as the head.

**Equivalent here:** llama-quantize --output-tensor-type / --token-embedding-type / --tensor-type <regex>=<type>

**Evidence (llama.cpp):** `src/llama-quant.cpp:683-688` · `src/llama-quant.cpp:693-718` · `src/llama-quant.cpp:456-471` · `src/llama-model-loader.cpp:1110-1114` · `src/llama-model.cpp:1368-1370`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Uncertain but with one immediately actionable sub-finding. The per-tensor seam exists and is precise (regex, first match wins, and setting it suppresses the k-quant mixture for that tensor), but acting on it means re-quantising from an F16 source and llama-quantize is not staged in the dflash2 build directory. The actionable part costs nothing: for the IQ1/IQ2/IQ3_XXS ftypes the built-in mixture already forces the output head to Q5_K, so the head is probably NOT the cheap win it looks like on a UD-IQ2_XXS artifact; and if the model has tied embeddings, token_embd is resident twice — once in the CPU/mmap buffer and once in VRAM as the duplicated output tensor — which is a real VRAM line item no flag removes short of an -ot on the output tensor.

### KV pool pinned by absolute bytes rather than by gpu-memory-utilization
**Where (theirs):** `single-user/start_qwen.sh:145-157` · `single-user/start_qwen.sh:158-199`

**What it does.** Sets the KV pool with `--kv-cache-memory=<bytes>` instead of letting it be derived from a utilization fraction, because the profiled activation peak is not reproducible between starts.

**Mechanism.** `KV_MEM=${KV_MEM-5583457484}` (or 5261334938 for CTX=huge), then `[ -n "$KV_MEM" ] && EXTRA_ARGS="--kv-cache-memory=$KV_MEM ${EXTRA_ARGS}"` (single-user/start_qwen.sh:199). `KV_MEM=` (empty) falls back to GPU_UTIL. Paired with `VLLM_V2_CUDAGRAPH_MEM_MIB` (1400 or 1900) so the V2 runner counts its own graphs.

**Why they needed it.** "The V2 runner's profiled activation peak swings ~1 GiB between starts, so the pool is pinned by bytes rather than by gpu-memory-utilization: 5.2 GiB -> 69,758 tokens = 1.06x at 64k, leaving ~1.1 GiB for transients (the same margin MTP mode runs with). Soak-tested with a 60k prompt, 4x16k concurrent and 8x4k generations." (single-user/start_qwen.sh:147-151)

**Their numbers.** Activation peak swings ~1 GiB between starts. 5.2 GiB pinned → 69,758 tokens bf16 at 64k (1.06x); int8 KV → 136,429 tokens with prefix caching on (138,696 without); KVarN 5.26 GiB → 268,169 tokens at 245,760 max-model-len. CUDA graphs ~1.2-1.3 GiB at k=7 (1400 MiB budget), 1.8 GiB at k=15 (1900 MiB).

**llama.cpp — EXISTS, NEVER SET.** The contract is written into fit.h:15-18 and enforced at fit.cpp:368-370 and 377-379. Note the asymmetry the map calls out: at the library level -ngl auto (-1) and -ngl all (-2) are identical, and the ONLY thing that distinguishes them is whether --fit's placement pass is allowed to run. So `-ngl 99` or `-ngl all` pins placement while still letting the context reduction run — which is probably not what you want if you are also pinning -c.

**Equivalent here:** a numeric -c / --ctx-size plus an explicit -ngl N, which together take --fit out of the loop

**Evidence (llama.cpp):** `common/fit.h:15-18` · `common/fit.cpp:368-370` · `common/fit.cpp:377-379` · `common/arg.cpp:1641-1644` · `src/llama-model.cpp:1745-1748`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Highest value on this list for measurement integrity. This repo's stated reason it cannot compare raw decode across boots is that free VRAM at boot moves 9,326–10,732 MiB and --fit follows it. Pinning is the fix, and llama.cpp supports it exactly: --fit modifies n_ctx if and only if it equals 0 (a numeric -c is never adjusted, and it says so in the log: `context size set by user to %u -> no change`), and it refuses layer placement outright when n_gpu_layers is not the default -1. Pin -c and -ngl to values found by one --fit run on a bad boot, and cross-boot comparison becomes legitimate rather than noise. THE TRAP: passing `-c 0` explicitly is NOT the same as omitting -c — the handler also sets fit_params_min_ctx = UINT32_MAX, which turns OFF --fit's context reduction entirely.

### The pool cost of a longer verify block scales with the block, not with the slot count
**Where (theirs):** `single-user/start_qwen.sh:153-157` · `single-user/start_qwen.sh:178-188`

**What it does.** Records that cutting max-num-seqs to buy context does not work, and that the real cost of a long verify block is CUDA graph size plus one aligned recurrent-state page per speculative block.

**Mechanism.** MAX_SEQS is reduced to 4 at DFLASH_TOKENS>7 "for the graphs", not for the state pages; `CG=${CG:-$((MAX_SEQS * (DRAFT_TOKENS + 1)))}` sizes graph capture in multiples of k+1 tokens to cover MAX_SEQS requests (single-user/start_qwen.sh:197-198).

**Why they needed it.** "A longer verify block costs pool twice: bigger CUDA graphs, and one aligned recurrent state page per speculative block. That second term is what scales -- NOT the slot count: 1 slot and 8 slots differ by about 8 MiB in total, so cutting MAX_SEQS buys no context." (single-user/start_qwen.sh:154-157). And: "Decode graphs are captured for both block lengths (the drafter's and the full verify block), or the short step -- the common one -- runs piecewise and costs 8%. That is 1.8 GiB of graphs instead of 1.45." (single-user/start_qwen.sh:185-188)

**Their numbers.** 1 slot vs 8 slots: ~8 MiB total difference. Missing the short-block graph capture costs 8% on the common step; graphs 1.8 GiB vs 1.45 GiB. At DFLASH_TOKENS>7: 4 slots and 56k (57,669 tokens) instead of 8 slots and 64k.

**llama.cpp — EXISTS, NEVER SET.** The vLLM claim (cost scales with the block, not with the slot count) holds here for the same structural reason: n_seq_max is 1 and both cost terms are functions of n_max. The startup lines make it directly observable, so this is measurement rather than modelling. The clamp direction also matters: --spec-draft-n-min is clamped down alongside n-max by the DFlash block clamp, and n-min has NO input validation at all — a negative value casts to a huge size_t and would discard every draft with no error and no warning.

**Equivalent here:** the n_rs_seq / batch-widening interaction: cparams.n_rs_seq = draft.n_max for the model-based speculators, and DFlash forces n_batch/n_ubatch up to n_parallel*(n_max+1)

**Evidence (llama.cpp):** `common/common.h:386-392` · `common/common.cpp:1699` · `src/llama-memory-recurrent.cpp:99-101` · `common/speculative.cpp:2418-2423` · `common/arg.cpp:4086-4092`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High, as the cost side of the --spec-draft-n-max sweep. Raising n-max from 3 toward 15 for draft-dflash costs VRAM in two places, both documented and neither obvious. (1) n_rs_seq = draft.n_max, and the recurrent state is allocated as rows = mem_size * (1 + n_rs_seq) in F32 — so 3→15 is a 4x multiplier on the RS buffer, readable in the `RS buffer size` startup line. (2) DFlash/DSpark force n_batch and n_ubatch up to n_parallel*(n_max+1), which is the only place in the whole area that raises your batch behind your back — and n_ubatch is what sizes the worst-case compute buffer including the quantised-KV F16 dequant scratch. Measure both against the depth-ladder win before choosing a value.

## absent, has a seam — 19

### Six enumerated things the acceptance harness cannot control for
**Where (theirs):** `bench/labd_accept.py:44-72`

**What it does.** Publishes the harness's own biases with their direction and magnitude, rather than presenting the number as clean.

**Mechanism.** (1) Chunk boundaries reset the adaptive long-block state machine — `next_num_draft_tokens` returns `draft_block` while `last_num_emitted is None`, the long block needs two qualifying steps in a row (`want and self._prev_want`), and the flag copy is one step stale, so each chunk pays ~3 short steps and the harness under-reports long-block scheduling by about 3/steps-per-chunk; steps/chunk is printed so the bias is visible. (2) It does not force *within* a chunk, so divergence at token 5 of a 128-token chunk leaves 123 tokens off-trajectory; excluding dirty chunks is itself a selection because divergence happens where the model is unsure, which is where acceptance is low, so the clean-only number is biased high. (3) The target was captured under *some* configuration. (4) Wall clock is not comparable across --chunk values. (5) /metrics counters are process-global. (6) Ties still break differently.

**Why they needed it.** "Watch the dirty count, not just the tok/step." (labd_accept.py:57) and "tok/step is the measurement; tok/s is a sanity check." (labd_accept.py:72)

**Their numbers.** Under-reports long-block scheduling by ~3/steps-per-chunk.

**llama.cpp — absent, has a seam.** The practice transfers directly; only the enumerated items change. Two more llama.cpp-specific biases belong on the list: /metrics rate gauges are reset on every scrape (server-context.cpp:4477, 2444-2446) so a second scrape reads zero, and the per-position histogram is sized common_speculative_n_max — 64 buckets with ngram-mod in the type list even when draft-dflash produced the draft (server-context.cpp:3899).

**Equivalent here:** none — the seam is the harness docstring in qwen38-tuning/bench/

**Evidence (llama.cpp):** `common/speculative.cpp:1952-1957` · `common/speculative.cpp:2044-2054` · `common/speculative.cpp:1992-2004` · `tools/server/server-context.cpp:4477` · `tools/server/server-context.cpp:3899`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High for this repo specifically, because the llama.cpp bias list is different and knowable. Chunk boundaries perturb ngram-mod's two automatic resets (occupancy > 0.25 at begin, speculative.cpp:1952-1957; five consecutive rounds below 25 % acceptance, speculative.cpp:2044-2054), and its all-or-nothing n_min=48 gate means a chunk boundary can turn a 64-token draft into nothing at all (speculative.cpp:1992-2004). Publishing that list is exactly the CORRECTIONS.md discipline applied before the number is published rather than after.

### Frozen target files keyed by prompt hash, tagged with their capturer
**Where (theirs):** `bench/labd_accept.py:236-245` · `bench/labd_accept.py:58-62`

**What it does.** Pins the target continuation to a file whose name contains a hash of the exact prompt token ids, records which run captured it, and warns loudly when the current run is the one doing the capturing (i.e. scoring itself).

**Mechanism.** `sig = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()[:16]`, path `labd_target_{name}_{sig}_{MAXTOK}.json` (labd_accept.py:236-237). The JSON stores `captured_by: TAG`. On capture it prints "CAPTURED target ... (this run is scoring itself; freeze and reuse for the comparison)" (labd_accept.py:243-245). Target is truncated if prompt+target would exceed the server's reported max_model_len (labd_accept.py:248).

**Why they needed it.** "Forcing server B onto server A's text measures how well B predicts A's continuation, not B's own. Freeze the target once and reuse the same file for every configuration in a comparison; never recapture in the middle of one." (labd_accept.py:59-62)

**llama.cpp — absent, has a seam.** No llama.cpp involvement; the only llama.cpp-side input is the server's reported context limit, available from GET /props (default_generation_settings.n_ctx, the per-slot value taken from slots.back().n_ctx at server-context.cpp:3939-3941) so the target can be truncated the same way.

**Equivalent here:** none — the seam is qwen38-tuning/bench/ alongside the existing frozen artefacts

**Evidence (llama.cpp):** `tools/server/server-context.cpp:3939-3941` · `tools/server/server-context.cpp:4580-4629`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate. Pure harness hygiene, but it is the guard that stops "server B predicting server A's continuation" being reported as B's own acceptance. Given this project already has eleven published-then-retracted claims, the self-scoring warning line is worth the twenty lines it costs.

### Run the whole suite twice after a restart and keep the second
**Where (theirs):** `bench/run_benchmarks.sh:11-13`

**What it does.** Makes the double-run an explicit protocol step rather than leaving it to the operator, and chains a quality check after the speed check.

**Mechanism.** Header comment instructs: run twice after a restart, keep the second numbers, then run bench/quality_battery.py.

**Why they needed it.** "Run it twice after a restart and keep the second numbers: the first run after start includes JIT warmup and reads 30-50% low. Then run bench/quality_battery.py — a fast server that emits garbage is worth nothing." (run_benchmarks.sh:11-13)

**Their numbers.** First run after start reads 30-50% low.

**llama.cpp — absent, has a seam.** Purely a protocol change with no llama.cpp knob behind it, but the llama.cpp mechanisms that make it necessary are documented: graph eviction at common.cuh:1435-1444, and the resolve_fused_ops → pp reserve → tg reserve → pp reserve again sequence at llama-context.cpp:576-671.

**Equivalent here:** none — the seam is the bench runner in qwen38-tuning/bench/

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/common.cuh:1435-1444` · `src/llama-context.cpp:576-671` · `src/llama-context.cpp:504-551`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate to high. llama.cpp has its own version of the first-run penalty: the CUDA graph map is swept every 5 s and any graph unused for ≥ 10 s is destroyed, so the first decode after an idle gap pays a full re-capture. A restart additionally pays the FA probe and three reserve passes. Making "discard run 1" a protocol step rather than operator lore is cheap and directly serves the repo's no-verdict-before-evidence rule.

### Long corpus with a byte-identical frozen head and non-repeating filler
**Where (theirs):** `bench/make_long_corpus.py:1-18` · `bench/make_long_corpus.py:31-52` · `bench/labd_bench.py:15-19`

**What it does.** Extends the frozen corpus for long-context runs while (a) keeping every number previously taken at --ctx 20000 valid and (b) refusing to hand the suffix lookup free matches from repeated text.

**Mechanism.** `long = base + "".join(parts)` where parts are vLLM's own `v1/**/*.py` sources over 2000 chars, up to 900,000 characters of filler (make_long_corpus.py:26, 34-51), guarded by `assert long[:len(base)] == base, "the frozen head must stay byte-identical"` (make_long_corpus.py:52).

**Why they needed it.** "the repetition hands the suffix lookup free matches from an earlier copy of the same text, which flatters exactly the number the lookup is judged on" (make_long_corpus.py:6-8). "That is varied (no repeats) and it is the realistic shape for this mode anyway: a coding assistant with a large real codebase in context." (make_long_corpus.py:12-13)

**Their numbers.** Frozen corpus ~84k tokens; anything past ~--ctx 65000 silently measures a shorter prompt. Long corpus's first 244,038 characters are byte-identical to the frozen one. The long corpus runs ~2.9 chars/token, not the 3.6 labd_bench.py assumes, so --ctx N yields ~1.24N tokens; --ctx 100000 gave 112,655-token prompts.

**llama.cpp — absent, has a seam.** The mechanism is stated in llama.cpp source: common/ngram-mod.cpp:27-41 stores the successor with no key, and speculative.cpp:1992-2004 builds exactly n_max tokens or clears the result if fewer than n_min=48 hit — so a repeated corpus is the difference between 64 accepted-candidate tokens and nothing. The byte-identical-head assertion is the part that keeps previously taken short-context numbers valid.

**Equivalent here:** none — the seam is the corpus builder in qwen38-tuning/bench/

**Evidence (llama.cpp):** `common/ngram-mod.cpp:27-41` · `common/speculative.cpp:1914` · `common/speculative.cpp:1992-2004` · `common/common.h:351-356`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Very high, and more dangerous to get wrong here than in vLLM. ngram-mod is a 4M-entry open-addressed hash of the last n_match=24 tokens storing only the successor token, with no key and no collision check. A corpus built by repeating the same text until it reaches a target length hands that table guaranteed hits from the earlier copy — inflating exactly the acceptance number ngram-mod is being judged on. Any long-context ngram-mod or draft-dflash,ngram-mod measurement taken on a repeated corpus should be treated as suspect.

### Explicit warning when the requested context exceeds what the corpus can supply
**Where (theirs):** `bench/labd_bench.py:61-67`

**What it does.** Refuses to silently measure a shorter prompt than the operator asked for.

**Mechanism.** `_want = int(CTX * 3.6)`; if `len(_full) < _want`, print a stderr WARNING naming the corpus, its actual char count, the wanted count, and the fix (`--corpus ~/bench/labd_corpus_long.txt`) before slicing anyway.

**Why they needed it.** This is the repo's stated instrument-fault pattern: a harness that returns a plausible number for a prompt shorter than the one requested. make_long_corpus.py:6-7 names it: "labd_bench.py slices `corpus[:ctx * 3.6]` and would silently measure a shorter prompt than asked for".

**llama.cpp — absent, has a seam.** server-context.cpp:3118-3125 rejects rather than truncates, and the error result carries n_prompt_tokens and n_ctx, so an over-ask is visible. The remaining risk is one-directional and harness-local, which is exactly what the vLLM warning covers.

**Equivalent here:** none harness-side; server-side the analogous failure is loud already

**Evidence (llama.cpp):** `tools/server/server-context.cpp:3116-3127` · `tools/server/server-task.cpp:1505-1508`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate. llama-server does NOT silently truncate an over-long prompt — it returns ERROR_TYPE_EXCEED_CONTEXT_SIZE naming both counts — so the server end of this is safe. The unguarded end is the harness under-filling: asking for 100k and quietly measuring 40k because the corpus ran out. That is the same instrument-fault shape and it is entirely on our side.

### Realistic-prompt cohort dataset instead of random tokens
**Where (theirs):** `bench/prompts_real.jsonl:1-8` · `bench/run_benchmarks.sh:53-64` · `bench/real_rep.sh:17`

**What it does.** Supplies 8 hand-written prompts spanning code generation, technical exposition, Danish long-form prose, refactoring, word problems, JSON schema production, comparative writing, and summarization, used as the `custom` dataset for the cohort rows.

**Mechanism.** `--dataset-name custom --dataset-path $HERE/prompts_real.jsonl --custom-output-len 1024 --num-prompts 8 --max-concurrency $C`, swept over C ∈ {1,2,4,8} and T ∈ {default, 0}.

**Why they needed it.** Random-token datasets do not exercise the drafter or the lookup realistically; acceptance is content-dependent, and the same file is reused as the activation-calibration corpus (act_calib.py:96-97) so the quantization decision is made on the same distribution.

**Their numbers.** 8 prompts, 1,024 output tokens each, concurrency ladder 1/2/4/8, both default-sampling and greedy (T=0).

**llama.cpp — absent, has a seam.** llama-bench's -p flag generates synthetic token sequences and would be actively misleading for any speculation number, which is a concrete reason the cohort must live at the server level. The n_min=48 all-or-nothing gate is the mechanism that makes random input read as zero rather than as low.

**Equivalent here:** none — the seam is the prompt set in qwen38-tuning/bench/

**Evidence (llama.cpp):** `common/speculative.cpp:1992-2004` · `common/common.h:351-356` · `tools/llama-bench/llama-bench.cpp:568-575`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High. Both speculators here are content-dependent in a way random tokens destroy: ngram-mod hashes 24-token windows and needs 48 consecutive hits to emit anything, and DFlash is a learned denoiser. A random-token benchmark would report near-zero acceptance for both and tell you nothing. The measured +34.7 % was taken on real code, which is the right instinct — writing the cohort down as a fixed file is what makes it repeatable.

### Best-of-reps at each concurrency rung, keyed on per-stream rate
**Where (theirs):** `bench/conc_ladder.py:102-133`

**What it does.** Runs each concurrency level REPS times (default 2) and keeps the rep with the highest per-stream decode rate, discarding rungs where any stream failed to produce a usable measurement.

**Mechanism.** Inner loop over `rep in range(REPS)`; `good = [v for v in res.values() if v['decode_s'] and v['ntok'] > 1]`; if `len(good) != n` the rung prints `incomplete (k/n)` and is skipped entirely (conc_ladder.py:115-117); otherwise `if best is None or per > best[0]: best = (per, agg, tps, ttft, its)`.

**Why they needed it.** A rung with a partially-failed fan-out would otherwise average a smaller batch into the N-stream row; best-of-reps takes the least-contended observation rather than a mean polluted by background interference.

**Their numbers.** Defaults: --max-n 8, --out 256, --reps 2.

**llama.cpp — absent, has a seam.** No llama.cpp mechanism is involved. The rejection is on measurement-method grounds against CLAUDE.md's own pairing rule, not against any llama.cpp fact. At -np 1 the concurrency ladder itself is also moot: n_cmpl is hard-limited to [1, n_parallel] so n > 1 is a 400.

**Equivalent here:** none — the seam is the bench runner

**Evidence (llama.cpp):** `tools/server/server-schema.cpp:61-65` · `tools/server/server-context.cpp:2335-2348`

**Effort:** small-patch · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Negative here, and I would not adopt it. Best-of-N is a biased-high estimator, and this repo's stated method is the opposite: pair within a round and alternate the order, because free VRAM at boot moves and --fit follows it. Taking the best rep would systematically pick the boot with the most free VRAM. The one part worth keeping is the incomplete-rung rule: discard a rung where any stream failed rather than averaging a smaller batch into it.

### Named failure signatures for the soak, ordered by how much each means
**Where (theirs):** `bench/labd_soak.py:17-36`

**What it does.** Tells the reader what a cross-request state bug actually looks like, so a benign difference is not escalated and a real one is not dismissed.

**Mechanism.** "Judge it by what changed: a synonym or a line break is the documented behaviour, a duplicated span or a truncated URL is not." (labd_soak.py:27-28); "a cross-request state bug tends to show up as one request degenerating (repetition, truncation, a fragment repeated out of order) while the others are fine, and usually not the first in the batch" (labd_soak.py:29-31); "acceptance far below both means the batch-wide block decision is thrashing" (labd_soak.py:31-32). The two batch-sensitive mechanisms named up front: the block length is one decision for the whole batch, and the drafter's grouped convolution resets on the block boundary so a wrong block size would convolve across two requests' query blocks (labd_soak.py:5-10).

**Why they needed it.** "Everything about lookup-augmented drafting was tuned at batch 1, and two of its parts are batch-sensitive" (labd_soak.py:3-4). The confounder is also named: all four requests share one document, so the run also exercises several requests resuming from one cached prefix — "a likelier source of trouble than the lookup, and worth ruling in or out with PREFIX_CACHE=0 before blaming the drafter" (labd_soak.py:34-36).

**llama.cpp — absent, has a seam.** The practice transfers; the signatures must be re-derived from llama.cpp. Two of the four are already emitted as warnings, one is DBG-only and one is a startup line — so the list doubles as a note about which log level you must be at to see each.

**Equivalent here:** none — the seam is a signature list in docs/reports/ or the soak harness

**Evidence (llama.cpp):** `common/speculative.cpp:1952-1957` · `common/speculative.cpp:2044-2054` · `common/speculative.cpp:2728-2733` · `tools/server/server-context.cpp:1209-1218`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate. The llama.cpp signature list is different and writable today: `ngram_mod occupancy %.2f exceeds threshold` (the table filled past 25 % and was wiped), five consecutive rounds below 25 % acceptance triggering a reset, `truncating draft to %d tokens` at SPC_DBG near the context edge, and `speculative decoding will use checkpoints` meaning the target can only roll back fully. Each has a distinct meaning and a distinct remedy.

### Residue-class sweep (mod 128) that turns a false threshold into a real periodicity
**Where (theirs):** `bench/bugb_sweep.py:1-13` · `bench/bugb_sweep.py:45-74`

**What it does.** Sweeps prompt length one token at a time, tokenizes each prompt locally to get its exact token count, and prints `prompt_tok % 128` as a column — which converts a bug that looks like a length threshold into a visible one-in-128 residue class.

**Mechanism.** For each requested ctx it slices `DOC[:ctx*3.6]`, builds the exact chat content, computes `ptok = len(TOK.encode(content))` with a local AutoTokenizer loaded from the served model dir (bugb_sweep.py:31-32, 50), then prints `ptok % 128` alongside tok/step and the verbatim-match length.

**Why they needed it.** "Sweep in steps of 1 token near a broken length, not in steps of 100: at a coarse grid this reads as a threshold, which is how it was first (mis)diagnosed." (bugb_sweep.py:12-13). The corresponding serving comment: "what it does is corrupt one prompt length in every 128 (gotcha 37, bench/bugb_sweep.py), which a coarse sweep reads as a length threshold and a lucky sweep misses entirely" (single-user/start_qwen.sh:230-231).

**Their numbers.** Against SPEC=dflash2 CTX=huge PREFIX_CACHE=1 CUDAGRAPH_MODE=FULL_AND_PIECEWISE: every prompt length ≡124 (mod 128) collapses to 1.97 tok/step with degenerate repetition; every other residue is clean. Default PIECEWISE is flat across the same sweep.

**llama.cpp — absent, has a seam.** The specific vLLM bug (a prefix-cache hit at residue 117+k) has no llama.cpp counterpart I can point to, but the class of bug does, and the alignment constants that would generate one are documented in the map. Note the flip rule differs by cache type: with q4_0 K/V on Ada the VEC/MMA boundary is Q->ne[1] <= 2, not the 8192 KV rule.

**Equivalent here:** none — the seam is a fine-grained prompt-length sweep in qwen38-tuning/bench/

**Evidence (llama.cpp):** `src/llama-kv-cache.cpp:1233-1246` · `ggml/src/ggml-cuda/fattn.cu:458` · `ggml/src/ggml-cuda/fattn.cu:464` · `ggml/src/ggml-cuda/fattn.cu:469` · `src/llama-context.cpp:288`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High, because llama.cpp has its own 256-aligned lattice and at least one unlogged kernel flip. n_kv is padded to FATTN_KQ_STRIDE = 256 explicitly so the graph stays constant, and can_use_vector_kernel requires n_kv % 256 == 0; cparams.n_ctx is padded to 256 and --fit rounds it down to 256. On top of that, an unquantised cache with gqa_ratio > 4 silently switches from VEC to MMA_F16 once K->ne[1] reaches 8192, with no log line. A sweep in steps of 1 near a suspicious length is the only instrument that distinguishes a threshold from a residue class.

### Judging correctness on the output, never on an absolute tok/step threshold
**Where (theirs):** `bench/bugb_sweep.py:64-74`

**What it does.** Decides whether a run is broken by checking that the model's answer is a verbatim prefix of the source document, rather than by comparing acceptance to a fixed number.

**Mechanism.** Grows `n` while `ans[:n+1] in doc` to find the longest verbatim prefix; computes `rep` as the maximum count of any 40-char window appearing elsewhere in the answer; `flag = 'ok' if len(ans) > 40 and n >= len(ans) - 2 else 'BROKEN'`.

**Why they needed it.** "Judge on the OUTPUT, not on an absolute tok/step: the ceiling is the verify block, so a healthy k=3 run sits at 3.96 and an absolute threshold calls it broken. A working verbatim task reproduces the whole answer; a broken one returns a few characters, a degenerate repeat, or nothing at all." (bugb_sweep.py:69-73)

**Their numbers.** A healthy k=3 run sits at 3.96 tok/step — an absolute threshold tuned for k=7 would call it broken.

**llama.cpp — absent, has a seam.** The ceilings are explicit in source: common_speculative_n_max returns ngram_mod.n_max = 64 for ngram-mod, draft.n_max for model-based types, and the DFlash block clamp lowers both n_max and n_min to block_size-1. Nothing in llama.cpp normalises across them, so the comparison must be made on output correctness plus per-impl attribution.

**Equivalent here:** none — the seam is the pass/fail predicate in the bench harness

**Evidence (llama.cpp):** `common/speculative.cpp:2351-2385` · `common/speculative.cpp:988-996` · `common/common.h:325` · `common/common.h:351-356`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High, and llama.cpp makes the trap worse than vLLM does. The mean-len ceiling is per-implementation: ngram-mod can reach 65 (n_max 64 + 1), draft-dflash with a stock 16-wide sidecar is capped at 16, and --spec-draft-n-max defaults to 3 so a dflash run may ceiling at 4. An absolute threshold tuned on one --spec-type will call the other broken. Judge on whether the output is right, then read mean len as a speed number only.

### Per-layer activation-quantization error measurement to choose which layers get int8
**Where (theirs):** `bench/act_calib.py:1-11` · `bench/act_calib.py:56-84` · `bench/act_calib.py:104-110`

**What it does.** Measures, for every linear layer, the relative L2 error that per-token int8 activation quantization introduces into that layer's *output* — the quantity that actually matters — rather than into its input, and ranks layers so INT8_LAYERS can be set from data.

**Mechanism.** Registers forward pre-hooks on modules matching `(gate_up_proj|down_proj|in_proj_qkvz|out_proj|qkv_proj|o_proj)$`, excluding mtp and lm_head (act_calib.py:56, 82). Each hook subsamples up to 64 token rows via `torch.linspace`, computes the per-row scale `xs.abs().amax(-1)/127.0`, forms `q = round(xs/scale).clamp(-128,127)*scale`, then runs the module twice — `y = m(xs)` and `yq = m(q)` — and records `‖yq-y‖/‖y‖` as out_err and `‖q-xs‖/‖xs‖` as act_err (act_calib.py:63-73). Re-entrancy is prevented by an `m._calib_off` flag checked in a `guarded` wrapper (act_calib.py:68-78). Runs vLLM in-process with `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `enforce_eager=True`, and `VLLM_MARLIN_INPUT_DTYPE` popped from the environment so the plain W4A16 kernels are used.

**Why they needed it.** "Use it to pick INT8_LAYERS for batch/start_qwen.sh: layers with small error are safe to run with int8 activations, the rest cost perplexity. On Qwen3.8-27B the GDN in_proj (early layers) and down_proj (last layers) are the worst." (act_calib.py:7-9)

**Their numbers.** Reports worst 25 layers by out_err, plus median and max per layer kind. Calibration corpus: 8 wikitext windows + 8 Danish + 6 vLLM source files at 3,000 chars each, plus the 8 real prompts.

**llama.cpp — absent, has a seam.** There is no forward-hook seam in llama.cpp and no activation-quant path to calibrate — the only load-time transformation is CPU repack, which is layout-only. What does exist is the offline half: --tensor-type is a regex over tensor names, first match wins, and setting it suppresses the k-quant mixture for that tensor. Note that for IQ1/IQ2/IQ3_XXS ftypes the built-in mixture already forces the output head to Q5_K, so the head may not be the cheap win it looks like.

**Equivalent here:** tools/imatrix (per-tensor activation statistics) feeding llama-quantize --tensor-type / --token-embedding-type / --output-tensor-type

**Evidence (llama.cpp):** `src/llama-quant.cpp:693-718` · `src/llama-quant.cpp:683-688` · `src/llama-quant.cpp:456-471` · `ggml/src/ggml-cpu/repack.cpp:4828-4829`

**Effort:** large-patch · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Potentially the largest quality lever available at 2-bit, and the most expensive. llama.cpp does not quantise activations at load or at runtime — resident weight bytes always equal file bytes — so the vLLM measurement has no direct target. The transferable idea is the mirror image: measure which tensors are hurt most by being at IQ2_XXS and raise only those, which llama-quantize supports per-tensor by regex. Cost: you need the F16 source (~54 GB for a 27B) and llama-quantize is not staged in the dflash2 build dir.

### GPU lookup kernels tested against a plain-Python reference on adversarial sequences
**Where (theirs):** `bench/test_lookup_kernels.py:16-31` · `bench/test_lookup_kernels.py:53-73`

**What it does.** Verifies the suffix-lookup kernel (longest match, then most recent) against a naive O(n²) Python implementation over sequence shapes chosen to break it, across five (k, nmin, nmax) configurations.

**Mechanism.** `ref_lookup` scans every end position `e in range(nmin-1, t)`, extends the match backwards while tokens agree, and takes `(n, e) > best` — i.e. longest match, ties broken by most recent. Cases: random sequences over vocabularies of 6/20/3/60 so matches are common; a periodic `[5,6,7,8]*30` where "the match must be allowed to overlap the suffix"; a verbatim quote reappearing near the end (`body + body[40:80]`); and a strictly increasing sequence with no match at all. Comparison ignores tokens past `valid` as don't-care (test_lookup_kernels.py:67-68).

**Why they needed it.** The lookup is what fills the tail of the long verify block; a wrong match length silently degrades acceptance rather than erroring, and the periodic/overlapping case is exactly where a naive implementation stops early.

**Their numbers.** Configurations swept: (k,nmin,nmax) = (7,4,32), (15,4,32), (31,6,12), (15,8,64), (7,2,8).

**llama.cpp — absent, has a seam.** There is no GPU lookup kernel here to test, but the equivalent component exists and is untested against a reference. The adversarial-sequence idea maps directly: periodic text, a verbatim quote reappearing near the end, and strictly-increasing tokens with no match are all cases the 24-token hash window handles differently.

**Equivalent here:** none in llama.cpp's own tests; the seam is qwen38-tuning/bench/tests, where TDD is already mandatory

**Evidence (llama.cpp):** `common/ngram-mod.cpp:9-46` · `common/ngram-mod.cpp:27-41` · `common/speculative.cpp:1914` · `common/speculative.cpp:1952-1957`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate and concrete. ngram-mod is CPU-side and small (common/ngram-mod.cpp), and it has a specific, unmeasured weakness: it stores only the successor token with no key and no collision check, so a collision returns a plausible wrong token that speculation merely fails to accept. A reference implementation would let you measure the collision rate as a function of occupancy on a real code corpus — which bounds how much of the observed acceptance loss is hash noise rather than model behaviour.

### Extrapolating a per-layer microbenchmark to a whole step, with the constant terms named
**Where (theirs):** `bench/spec_attn_ctx_scan.py:125-131`

**What it does.** Converts the per-attention-layer microbenchmark into milliseconds per verify step, and states explicitly which parts of the step it is not measuring and why that is legitimate.

**Mechanism.** Second table multiplies the best segment count's time by 16 attention layers and divides by 1000, for both `ours@best` and FA2.

**Why they needed it.** "What a whole step costs: 16 attention layers, plus the GDN layers and the GEMMs, which do not grow with context. The verify attention is the only context-dependent term." (spec_attn_ctx_scan.py:125-126)

**Their numbers.** 16 attention layers in the model; GDN layers and GEMMs are context-independent.

**llama.cpp — absent, has a seam.** The map states the recurrent half does not grow with -c and points at the `RS buffer size` startup line as the number to read. The honest caveat to carry: the F16 dequant scratch for quantised KV is charged to the compute buffer per layer, so the extrapolation prices VRAM as well as time.

**Equivalent here:** none — the seam is the analysis on top of test-backend-ops MODE_PERF output

**Evidence (llama.cpp):** `src/llama-memory-recurrent.cpp:99-101` · `src/llama-memory-recurrent.cpp:118-127` · `src/llama-hparams.cpp:183-229` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate. The decomposition holds here for the same reason: on a hybrid Qwen3.5-style model the recurrent/DeltaNet state does not scale with context at all (rows = mem_size * (1 + n_rs_seq), width from n_embd_r/n_embd_s), so attention is the only context-dependent term and multiplying per-layer µs by the attention-layer count is legitimate. The number of attention layers must be read from the model, not assumed.

### API feature smoke suite: the request-level features a new model runner silently breaks
**Where (theirs):** `bench/api_smoke.py:1-6` · `bench/api_smoke.py:46-105`

**What it does.** Runs twelve independent feature checks against the live server — greedy determinism, seeded sampling determinism, logprobs/top_logprobs, n=2, stop strings, json_schema structured output, min_tokens with penalties, streaming, thinking mode with reasoning_content, completions echo+logprobs, a 20k-token prompt, and a negative test — printing PASS/FAIL per feature with each failure caught as an exception.

**Mechanism.** `check()` wraps each test in try/except and records `(name, ok, info)`. Determinism tests issue the same request twice and compare strings. `t_thinking` asserts both that the answer contains 391 and that `reasoning_content` or `reasoning` is non-empty. `t_thinking_budget_rejected` is an inverted test: it expects HTTP 400 and returns False if the request is accepted — pinning that the V2 runner rejects `thinking_token_budget` rather than silently ignoring it (api_smoke.py:93-98).

**Why they needed it.** "the request-level features a different model runner could break" (api_smoke.py:1-2). Everything in this repo is a patched runner, so per-request features are exactly what regresses without a throughput number moving.

**Their numbers.** 12 checks; 20k-token prompt test asserts prompt_tokens > 12000.

**llama.cpp — absent, has a seam.** Every feature the vLLM suite checks has a llama.cpp counterpart in the server schema, and none of them is covered by the 108 bench tests as far as the map shows. The inverted-test idea (assert a 400 rather than silent acceptance) maps onto llama.cpp's set_hard_limits fields, which throw, versus set_limits fields, which silently clamp — a distinction worth pinning per field.

**Equivalent here:** none shipped; the seam is qwen38-tuning/bench/tests against a live llama-server

**Evidence (llama.cpp):** `tools/server/server-schema.cpp:588-607` · `tools/server/server-schema.cpp:61-65` · `tools/server/server-context.cpp:1464` · `tools/server/server-context.cpp:4638-4642`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High, because this repo runs a patched build (build-dflash2, PR #27342 on top of master), which is exactly the situation the suite exists for: per-request features regress without any throughput number moving. Concrete llama.cpp checks: seeded determinism, n_probs/logprobs, stop strings, json_schema, streaming, a long prompt, and an inverted test that `n > 1` returns 400 at -np 1. Note two llama.cpp-specific traps for such a suite — an out-of-range id_slot WRAPS modulo the slot count instead of erroring, and POST /props returns success while changing nothing.

### verify.sh as an install-integrity gate with a three-way patch check
**Where (theirs):** `verify.sh:1-11` · `verify.sh:40-49` · `docker/entrypoint.sh:8-18`

**What it does.** Checks that the box is installed the way the published numbers assume — venv, exact vLLM version, every patch applied, model requantized, KVarN present — and blocks the server from starting if anything fails.

**Mechanism.** For each `patches/*.patch`, first try `patch -p1 -R --dry-run` inside vLLM's site-packages (an exact reverse test: it applies cleanly in reverse ⇒ it is applied). If that fails, fall back to `patches/_check_applied.py` for content matching, because "two patches touching the same file (the DFlash2 pair) can no longer be reversed individually once both are applied" (verify.sh:41-42). If forward `patch -N --dry-run` succeeds, report NOT applied with the exact command to fix it; otherwise report a version mismatch. Also greps `envs.py` for `VLLM_MARLIN_INT8_INCLUDE_RE` to confirm the env vars were registered. The container entrypoint runs `bash verify.sh --no-server || exit 1` before exec'ing either start script, with `VERIFY=0` as the escape hatch.

**Why they needed it.** "Check that this repo is installed the way the README numbers assume" (verify.sh:2). A missing patch would not error at startup — it would serve at a different speed and produce a plausible wrong benchmark.

**Their numbers.** Exit 0 on all PASS (WARNs allowed), 1 on any FAIL. Pinned version: vllm 0.27.1 — anything else is a WARN, "patches were written against 0.27.1" (verify.sh:27).

**llama.cpp — absent, has a seam.** The map documents each of these as a build-flag-dependent capability with a source line, and CANNOT #6 in the attention area is the killer: with `-fa on` and an unsupported combination there is no probe, no error, and the node is simply scheduled on CPU. Runtime side, GET /props reports build_info so the gate can also assert the running binary matches the one that was verified.

**Equivalent here:** none shipped; the seam is a preflight in qwen38-tuning/scripts/ asserting build identity before the server is allowed to start

**Evidence (llama.cpp):** `ggml/CMakeLists.txt:208` · `ggml/src/ggml-cuda/fattn.cu:442-446` · `src/llama-context.cpp:554` · `ggml/src/ggml-cuda/ggml-cuda.cu:5286-5289` · `tools/server/server-context.cpp:4580-4629`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Very high, and the llama.cpp version has teeth the vLLM one does not. Several capabilities in this profile are decided at BUILD time and are invisible at runtime: GGML_CUDA_FA_ALL_QUANTS=OFF means q5_0/q5_1 KV and any asymmetric -ctk/-ctv silently have no CUDA kernel (the FA op falls to CPU, a working-but-enormously-slower server with no error); LLAMA_LLGUIDANCE=OFF means a %llguidance grammar GGML_ABORTs; the CUDA arch list decides which kernels exist at all. A gate that asserts the build number, the commit, and those CMakeCache values before any measurement is the direct analogue of the patch check.

### Live-server verification reads the pool and backend the server actually chose out of its own log
**Where (theirs):** `verify.sh:143-159`

**What it does.** After confirming /health and a correct chat completion, extracts from qwen.log which attention backend was selected, how many KV tokens the pool actually holds, the reported maximum concurrency, whether Marlin kernels are in use, and whether KVarN capped max_num_seqs.

**Mechanism.** `grep -oE "Using [A-Z_]+ attention backend"`, `"GPU KV cache size: [0-9,]+ tokens"`, `"Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x"`, `"MarlinLinearKernel"`, and `"capping max_num_seqs [0-9]+ -> [0-9]+"`, each `| tail -1`. The functional check is a Danish question with `temperature:0, max_tokens:8, enable_thinking:false` grepped for "københavn|copenhagen".

**Why they needed it.** The pool size and backend are the two numbers every context claim depends on, and they are chosen at boot from free VRAM — reading them back from the log is the only way to know what the running server actually got rather than what was requested.

**llama.cpp — absent, has a seam.** Every one of those numbers is chosen at boot and none is asserted today. The FA case is the sharpest: CANNOT #7 in the attention area states outright that the startup log does not report the resolved FA state, and with -ctv q4_0 an FA-off outcome is a hard init failure rather than a slow server — so the scraper doubles as an early-failure detector.

**Equivalent here:** none shipped; the seam is a log scraper over llama-server's stderr plus GET /props and GET /slots

**Evidence (llama.cpp):** `src/llama-context.cpp:312` · `src/llama-context.cpp:532-548` · `tools/server/server-context.cpp:1220-1222` · `src/llama-kv-cache.cpp:337-338` · `src/llama-context.cpp:686-697`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Very high for this profile — arguably the single most useful item after pinning -c. With -ngl auto --fit on, what the server actually got is decided at launch from free VRAM at that instant, and the startup line at llama-context.cpp:312 prints the FA type you ASKED for, not the resolved one. The only truthful FA lines are the auto-probe's own `Flash Attention enabled` / `Flash Attention not supported, set to disabled`. Lines worth scraping every run: `srv init: n_slots, n_ctx_slot, kv_unified`, n_ctx/n_ctx_seq/n_batch/n_ubatch, `attn_rot_k`/`attn_rot_v`, `RS buffer size`, `graph nodes = %d (with bs=%d), %d (with bs=1)`, and `speculative decoding will use checkpoints`.

### Recurrent (GDN) state cache in fp16 instead of the config's fp32
**Where (theirs):** `batch/start_qwen.sh:6-9` · `batch/start_qwen.sh:115` · `single-user/start_qwen.sh:298`

**What it does.** Halves both the per-request recurrent state footprint and the state traffic per decode step, which is what makes the full 64-way batch fit and run.

**Mechanism.** `--mamba-ssm-cache-dtype float16` on both launchers.

**Why they needed it.** "the Gated DeltaNet recurrent state is fp32 by default (Qwen's config says so) and costs ~150 MB per resident request; fp16 halves that AND halves the state traffic per decode step. That is what lets all 64 requests actually run at once (fp32: only 37). Perplexity is unchanged (8.045 vs 8.046 on our en/da/code check)." (batch/start_qwen.sh:6-10)

**Their numbers.** ~150 MB per resident request at fp32 → half that at fp16; 64 concurrent requests fit vs 37 at fp32; perplexity 8.045 vs 8.046 on the en/da/code battery.

**llama.cpp — absent, has a seam.** CANNOT #1 in the KV/memory area is unambiguous: every construction site passes GGML_TYPE_F32 as a literal for both conv and recurrent state, -ctk/-ctv reach only the attention half, and no flag anywhere sets type_r or type_s. The seam is nonetheless small and named — three literals and one argument — which is why this is absent-but-possible rather than impossible. Whether the served model is a hybrid at all is a map gap.

**Equivalent here:** none — type_r/type_s are GGML_TYPE_F32 literals; the seam is the three call sites in src/llama-model.cpp plus one new arg in common/arg.cpp

**Evidence (llama.cpp):** `src/llama-model.cpp:2274-2275` · `src/llama-model.cpp:2335-2336` · `src/llama-memory-recurrent.cpp:99-101` · `src/llama-memory-recurrent.cpp:118-127` · `common/common.h:386-392`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Unknown in MiB until the `RS buffer size` startup line is read, but the mechanism that would make it matter here is specific and checkable. The recurrent state does NOT scale with context — rows = mem_size * (1 + n_rs_seq) — and n_rs_seq is 0 for ngram-mod but equals draft.n_max for draft-dflash. So switching from ngram-mod to draft-dflash multiplies the recurrent buffer by (1 + n_max), and raising --spec-draft-n-max from its default 3 toward 15 multiplies it again. Halving its element size would be a direct offset. Read the log line before deciding whether it is worth the patch.

### The prefill tax of the int8 long-context mode stated, with the workload it is and is not for
**Where (theirs):** `single-user/start_qwen.sh:95-97`

**What it does.** Records that the int8/Triton long-context DFlash2 mode more than doubles prefill time, and names the front-end shape that makes it worthwhile anyway.

**Mechanism.** Comment attached to the `ATTN_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"` branch.

**Why they needed it.** "Costs prefill: 251 s to load a 112k document against FLASH_ATTN's ~112 s. With PREFIX_CACHE=1 only the first turn pays it (5.9 s afterwards), which is why this is a mode for a RAG or coding front-end that loads a document once, not for general chat." (single-user/start_qwen.sh:95-97)

**Their numbers.** 112k-document prefill: 251 s (int8/Triton) vs ~112 s (FLASH_ATTN); 5.9 s on subsequent turns with PREFIX_CACHE=1.

**llama.cpp — absent, has a seam.** The mechanism is documented in the map with source lines and the conclusion follows: quantised KV costs extra compute-buffer VRAM during prompt processing, not just less cache VRAM. The measurement must pin -c and -ngl (see the pinning verdict) or the two arms will differ by more than the cache type.

**Equivalent here:** none measured; the seam is a paired -ctk f16 vs -ctk q4_0 prefill benchmark at fixed -c and -ngl

**Evidence (llama.cpp):** `src/llama-context.cpp:595` · `ggml/src/ggml-cuda/fattn.cu:469` · `ggml/src/ggml-cuda/fattn.cu:534-568` · `ggml/src/ggml-cuda/fattn-common.cuh:1022-1084` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High, because this profile has an unstated prefill tax of exactly the same kind. Prompt processing always runs at n_tokens = min(n_ctx, n_ubatch), which is far above the VEC kernel's Q->ne[1] <= 2 cutoff, so prefill on a quantised cache always takes MMA_F16 and always dequantises the whole K and V cache to F16 into compute-buffer scratch. -ctk q4_0 -ctv q4_0 therefore buys KV bytes and costs prefill time and compute-buffer VRAM, and nobody here has priced either. For a coding agent that loads a file once and then edits, the trade may be clearly right — but it should be a measured statement, not an assumption.

### Patches applied and verified at image build time
**Where (theirs):** `Dockerfile:26-30` · `docker/requirements.txt:1-9`

**What it does.** Applies every patch into vLLM's site-packages during the build and runs `verify.sh --install` as a build step, so an image that built at all is an image whose patch set is known good.

**Mechanism.** `SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1); for p in patches/*.patch; do patch -p1 -d "$SP" < "$p"; done; bash kvarn/install.sh; bash verify.sh --install` under `set -e`. Requirements are pinned to the exact reference venv resolution.

**Why they needed it.** "The stack every number in the READMEs was measured on. vllm==0.27.1 pins torch 2.13.0 (cu130), triton 3.7.1 and flashinfer-python 0.6.16.post3 itself; the rest is pinned to what the reference venv resolved to." (docker/requirements.txt:1-3)

**Their numbers.** vllm 0.27.1, torch 2.13.0/cu130, triton 3.7.1, flashinfer-python 0.6.16.post3, transformers 5.15.0, tokenizers 0.22.2, compressed-tensors 0.17.0, huggingface_hub 1.27.0.

**llama.cpp — absent, has a seam.** The value here is not the patch-reversal trick (there is no patch stack) but the principle: verify at build time, then assert the running binary is that build. The map already demonstrates that a missing build flag does not error — it silently changes which KV types have a CUDA kernel, and an unsupported FA node is quietly scheduled on the CPU backend.

**Equivalent here:** none shipped; the seam is the build script for build-dflash2 plus a post-build assertion over CMakeCache.txt and compile_commands.json

**Evidence (llama.cpp):** `ggml/CMakeLists.txt:208` · `ggml/src/ggml-cuda/fattn.cu:442-446` · `ggml/src/ggml-cuda/ggml-cuda.cu:5286-5289` · `tools/server/server-context.cpp:4580-4629`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High, and it folds into the install-integrity verdict. This repo builds a fork of llama.cpp at a specific commit with build options that decide capability, and there is currently nothing asserting that the binary being measured is the binary that was verified. The concrete assertions: llama-server --version reports build 10499 / commit 1deefcca3; GGML_CUDA_FA_ALL_QUANTS is OFF (or ON, if that is the intent); CMAKE_CUDA_ARCHITECTURES is 89; LLAMA_LLGUIDANCE is OFF. Runtime, GET /props reports build_info so the running process can be matched to the verified build.

## partial — 12

### Warm-up done before any counter is read, sized to swallow Triton JIT
**Where (theirs):** `bench/labd_accept.py:220-226` · `bench/labd_bench.py:80-91` · `bench/demo_capture.py:101-102` · `bench/run_benchmarks.sh:45`

**What it does.** Runs throwaway generations before the first measurement so that first-step Triton kernel compilation and CUDA-graph JIT do not land inside a measured decode window.

**Mechanism.** labd_accept.py issues two 64-token generations against a 4k-char slice of the corpus before the first `metrics()` call. labd_bench.py does the same via two chat completions. demo_capture.py runs one 32-token warm prompt. run_benchmarks.sh:45 runs a `vllm bench serve` random 256/256 job to /dev/null.

**Why they needed it.** "the first long-block step JIT-compiles Triton kernels, and those seconds would otherwise land inside the first task's decode window (worth 30% on it)." (labd_bench.py:80-81). labd_accept.py:222-223 adds "Done before any counter is read, so it never enters a measurement."

**Their numbers.** The first long-block step's JIT is worth 30% on the first task's decode window. Separately, run_benchmarks.sh:11-13: "the first run after start includes JIT warmup and reads 30-50% low."

**llama.cpp — partial.** The flag exists and is on, but common.cpp:1487-1521 warms with an empty run and then resets the samplers — it does not produce the ubatch shapes the benchmark will measure. Worse, under speculation the step size varies with acceptance, so warmup may never complete at all; see the verdict on VLLM_SPEC_DECODE_ATTN_QMAX.

**Equivalent here:** --warmup / --no-warmup (default enabled), but it is an EMPTY run, not a warm generation at the measured shape

**Evidence (llama.cpp):** `common/common.cpp:1487-1521` · `common/arg.cpp:1957-1961` · `ggml/src/ggml-cuda/ggml-cuda.cu:4253-4262` · `src/llama-graph.h:785`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High, for a different reason than in vLLM. There is no Triton JIT here — kernels are AOT-compiled for compute_89 — but CUDA graph capture needs TWO consecutive calls with unchanged node properties before it arms (ggml-cuda.cu:4253-4262), and llama-level graph reuse requires ubatch.n_tokens to be identical (llama-graph.h:785). The built-in empty warmup does not exercise the decode shape, so the first measured decode runs eagerly. A harness-side warm generation at the real shape is required before any counter is read.

### Decode rate derived from median TPOT × concurrency, not from wall clock
**Where (theirs):** `bench/run_benchmarks.sh:32-37` · `bench/run_benchmarks.sh:62`

**What it does.** Reports a per-stream decode figure that is robust to a few slow requests, by inverting the median time-per-output-token and multiplying by the concurrency.

**Mechanism.** `row()` extracts Mean TPOT, Median TPOT, Mean TTFT, Benchmark duration and Output token throughput from the vllm bench serve log with awk field indices, then `DEC = C*1000/medTPOT` (run_benchmarks.sh:35). The cohort rows use mean TPOT instead (`decode(C/meanTPOT)`, line 62) and label which one they used in the output string.

**Why they needed it.** The label itself carries the method — `decode(C/medTPOT)` vs `decode(C/meanTPOT)` — so a reader cannot confuse two differently-derived decode numbers.

**llama.cpp — partial.** llama.cpp gives per-token ms per request but /metrics exposes only mean rates, and those gauges reset on every scrape (server-context.cpp:4477, 2444-2446), so a median has to be assembled from the per-request timings blocks. Nothing prevents it; nothing provides it.

**Equivalent here:** per-request `predicted_per_token_ms` in the timings block; the median across requests must be computed harness-side

**Evidence (llama.cpp):** `tools/server/server-common.h:346-425` · `tools/server/server-context.cpp:4477` · `tools/server/server-context.cpp:2444-2446`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Low at -np 1, where concurrency is 1 and the median over a cohort is just a robustness choice rather than a different measurement. The useful half is the labelling discipline — carrying the derivation in the column name so two differently-derived decode numbers cannot be confused. That is worth adopting regardless.

### tok/pass: separating a scheduling failure from a kernel failure
**Where (theirs):** `bench/conc_ladder.py:11-15` · `bench/conc_ladder.py:119-124`

**What it does.** Reports mean tokens per forward pass alongside per-stream and aggregate throughput, so that a throughput plateau can be attributed to the scheduler batching poorly rather than to the kernels being slow.

**Mechanism.** Diffs `vllm:iteration_tokens_total_sum` and `vllm:iteration_tokens_total_count` around the window: `its = Δsum/Δcount` (conc_ladder.py:123-124). Expected value for a properly batched decode step at N streams and k drafts is N*(k+1).

**Why they needed it.** "A batched decode step at N streams and k drafts runs N*(k+1) query tokens, so if the aggregate rate stops rising with N while tok/pass stays flat, the streams are taking turns instead of batching -- a scheduling problem, not a kernel one. Prefill passes are in the same mean, so compare it across N rather than against N*(k+1) directly." (conc_ladder.py:12-15)

**llama.cpp — partial.** llama.cpp exposes no per-pass token histogram; the closest quantity is mean len, which already says how many query tokens a verify step carried. n_reused is incremented at llama-context.cpp:1348 and rendered at server-context.cpp:617-619, so the graph-reuse half of the question is answerable today at INFO.

**Equivalent here:** `graphs reused = %d` (llama_perf_context n_reused) + `mean len` from the draft-acceptance line; no iteration-tokens histogram

**Evidence (llama.cpp):** `tools/server/server-context.cpp:617-619` · `src/llama-context.cpp:1332-1372` · `tools/server/server-context.cpp:634-637`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate, and the diagnostic question is different here. At -np 1 there is nothing to batch across slots, so the vLLM version of the question is moot. The llama.cpp version worth asking is whether the step is running as a captured CUDA graph at all — `graphs reused` is printed on every completion and nobody appears to read it. A low reuse count under speculation is a scheduling-shaped failure with a kernel-shaped symptom.

### Concurrency soak with a same-composition determinism check as the hard failure
**Where (theirs):** `bench/labd_soak.py:17-32` · `bench/labd_soak.py:108-128`

**What it does.** Runs a fixed mix of copy-heavy and prose requests concurrently for R rounds and treats a text difference between two rounds of *identical batch composition* as an unambiguous bug, while treating a difference against the batch-1 reference as weak evidence requiring judgment.

**Mechanism.** Every round submits `[TASKS[i % len(TASKS)] for i in range(CONC)]` through a ThreadPoolExecutor, so composition is constant. The first round's copy output is stored; any later copy output differing from it sets `note += ' != ROUND 1'` and increments `bad` (a hard failure, exit 1). A difference from the separately-taken batch-1 reference increments only `soft` and prints `!= alone`.

**Why they needed it.** "`!= round 1` on a copy row. Every round runs the same four requests, so two rounds have the same batch composition and must produce the same text. A difference here has no innocent explanation." (labd_soak.py:20-22). Versus: "`!= alone` is weaker evidence. The batch-1 reference runs against a different batch composition, and the verify block is one chunk through the recurrent layers, so the last bits of the logits differ and a near-tie can flip." (labd_soak.py:24-27)

**Their numbers.** Defaults --conc 4, --rounds 3, --max-tokens 256, --ctx 20000. Acceptance sanity band: "it should sit between the batch-1 numbers for prose and for copying"; latency band: "no request should be slower than ~3x the batch-1 wall time for the same work."

**llama.cpp — partial.** Two llama.cpp facts change the shape of the test. First, the greedy verifier means the speculator cannot alter the output, so a text mismatch is unambiguous. Second, draft-dflash forces llama_set_causal_attn(ctx_dft, false) for the draft context's whole life (speculative.cpp:1036) and the draft context always sets n_rs_seq=0 with rollback via checkpoints — a plausible place for state to leak across requests, and worth one determinism run on the draft-dflash profile specifically.

**Equivalent here:** repeat the same request at temp 0 and compare text; there is no multi-slot composition to vary at -np 1

**Evidence (llama.cpp):** `common/sampling.cpp:692-720` · `common/speculative.cpp:1914` · `common/speculative.cpp:1036` · `common/speculative.cpp:2460` · `common/speculative.cpp:2044-2054`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate, and the target is different. At -np 1 there is no batch composition, but there IS cross-request state: the ngram-mod hash table is a single 16 MiB allocation shared across all sequences and persisting for the process lifetime, with two automatic resets that fire on history. A determinism check therefore probes speed stability, not text stability — llama.cpp speculation is exact (greedy accept-if-equal), so a text difference between two identical temp-0 requests would be a genuine bug with no innocent explanation, same as the vLLM round-1 rule.

### The empty-tokenizer trap: encode <think> in every directory that will be served
**Where (theirs):** `verify.sh:97-125` · `docker/prepare.sh:20-25`

**What it does.** Detects a model directory missing tokenizer.json before the server is started, by encoding a known string and asserting the result is non-empty.

**Mechanism.** For every dir that could be passed as `--model` (base and `-fast`), load `AutoTokenizer.from_pretrained(d)` and `encode("<think>", add_special_tokens=False)`; empty id list ⇒ FAIL with the exact remedy. docker/prepare.sh's `state()` function includes `tokenizer.json` and `tokenizer_config.json` in the required-file list for the same reason.

**Why they needed it.** "A served model dir with no tokenizer.json is not an error to transformers: it hands back a Qwen2Tokenizer with a 1-token vocabulary that encodes everything to []. The server then dies far downstream on 'ReasoningConfig: failed to tokenize reasoning strings', which names neither the dir nor the tokenizer. Encode <think> here instead." (verify.sh:98-101)

**llama.cpp — partial.** Judged against llama.cpp rather than transplanted: the tokenizer half is not-applicable, the template half is a real and unguarded silent fallback with an exact source line. Two templates are additionally patched in place at load (gpt-oss and Mistral guards), so what the server uses is not necessarily what the file contains.

**Equivalent here:** no equivalent trap (the GGUF carries the vocab), but the chat-template fallback is the llama.cpp version of the same shape

**Evidence (llama.cpp):** `common/chat.cpp:759-781` · `common/chat.cpp:776-781` · `common/chat.cpp:785-804` · `src/llama-model-loader.cpp:723-785` · `tools/server/server-context.cpp:4580-4629`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate. The specific trap does not exist — a GGUF with a broken vocab fails at load. The isomorphic llama.cpp trap does: common_chat_templates_init falls through to the built-in CHATML source when the GGUF has no tokenizer.chat_template, and a literal source of "chatml" is treated as empty and falls through the same chain. You get a servable-looking server producing subtly wrong prompts. The check is one assertion against GET /props chat_template. A second, related silent-wrong: the printed `file type` is a guess derived from the most-common tensor type and is OR'd with GUESSED, so the header line can misreport the quant.

### Four KV cache formats with the cost of each stated as a function of context
**Where (theirs):** `batch/start_qwen.sh:33-57` · `single-user/start_qwen.sh:24-29` · `single-user/start_qwen.sh:54-87`

**What it does.** Offers fp8 (default), KVarN 4-bit-key/2-bit-value, vLLM's built-in int4 per-token-head, and int8 per-token-head, each bound to a context length, a GPU_UTIL and an attention backend, with the decode penalty stated at a named context depth rather than as a constant.

**Mechanism.** batch mode: `KV=fp8` → `--kv-cache-dtype fp8`, 150k, util 0.972; `KV=kvarn` → `--kv-cache-dtype kvarn_k4v2_g128 --block-size 128`, 262k, util 0.93, plus `KVARN_POOL_MEM_FRAC` (default 0.25) for the fp16 staging pool holding tiles still being written; `KV=int4pth` → `--kv-cache-dtype int4_per_token_head --attention-backend TRITON_ATTN`, 262k. single-user: CTX=fast → FLASH_ATTN + bfloat16 KV, 64k, 4 drafts; CTX=long → fp8 (FlashInfer), 150k, 3 drafts; CTX=huge → kvarn_k4v2_g128 + block-size 128, 200k, 3 drafts, KVARN_POOL_MEM_FRAC 0.15.

**Why they needed it.** "The decode tax is a function of context, not a constant: ~6% on short prompts, but 2.13x at 112k (32.0 vs fp8's 68.1 tok/s), of which ~1.98x is step time and the rest is MTP acceptance falling from 2.56 to 2.38 tokens per step. Take it when the request would not otherwise fit, not for speed." (single-user/start_qwen.sh:57-61)

**Their numbers.** KVarN: 262k context (batch), ~2x token capacity, +0.2% perplexity, ~20% slower decode at long context. At 112k single-user: 32.0 tok/s vs fp8's 68.1 (2.13x), decomposed into ~1.98x step time and MTP acceptance 2.56 → 2.38 tok/step. int4pth: 262k, ~1.5x slower decode / 2.3x slower prefill at 100k than fp8. KVarN pool: ~20 KB/token effective; fp8 staging fraction 0.25 batch / 0.15 single-user — "0.25 keeps all 64 slots, smaller values cap max-num-seqs" (batch/start_qwen.sh:51-52).

**llama.cpp — partial.** The context-dependence is real here too, and for a llama.cpp-specific reason: on Ada with quantised K/V the VEC kernel is chosen only when Q->ne[1] <= 2, so a speculative verify step of three or more tokens runs MMA_F16 and dequantises the whole cache to F16 — a cost that grows linearly with n_kv. The quantised-KV tax therefore rises with both depth AND draft length, which no llama.cpp document states.

**Equivalent here:** -ctk/-ctv, but only f16, bf16, q4_0 and q8_0 have CUDA FA kernels in this build, and K must equal V

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:338-357` · `ggml/src/ggml-cuda/fattn.cu:442-446` · `ggml/src/ggml-cuda/fattn.cu:469` · `common/arg.cpp:305-315` · `ggml/CMakeLists.txt:208`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate. The menu is shorter than vLLM's: q4_0/q4_0 (in use, cheapest) and q8_0/q8_0 (the quality step, roughly double the KV bytes) are the only real choices, and q8_0 is probably unaffordable at 12 GB with a 6.77 GB model. q4_1/q5_0/q5_1/iq4_nl parse and appear in --help but have no kernel here — a live trap. Rebuilding with GGML_CUDA_FA_ALL_QUANTS=ON would unlock q5_0/q5_1 and asymmetric pairs at the cost of compile time and binary size; that is a config-level option nobody has priced. The transferable half is the framing: state the decode tax as a function of context, not as a constant.

### Split-KV Triton attention for the multi-query verify step
**Where (theirs):** `single-user/start_qwen.sh:17-19` · `single-user/start_qwen.sh:71-77` · `single-user/start_qwen.sh:88-99`

**What it does.** Replaces FlashAttention-2 on the verify step with a split-KV kernel that partitions the KV length across segments, because FA2 leaves most of the GPU idle when the query block is only a handful of tokens.

**Mechanism.** `export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-1}` from patches/spec-decode-attn.patch, enabled only on bf16-KV paths (CTX=fast) and on the DFlash2+int8 path via patches/spec-decode-attn-int8.patch; explicitly disabled on CTX=huge because "the KVarN backend brings its own dequant path" (single-user/start_qwen.sh:103-105).

**Why they needed it.** "split-KV attention for the 5-query verify step (FA2 leaves 58 of 82 SMs idle there)" (single-user/start_qwen.sh:17-18). For the int8 path: "vLLM's own Triton attention will not split KV for a multi-query verify, which is every step here, and costs 7.4 ms per layer at 128k against this kernel's 1.3" (single-user/start_qwen.sh:92-94).

**Their numbers.** FA2 leaves 58 of 82 SMs idle on the 5-query verify. At 128k, stock Triton attention costs 7.4 ms per layer versus this kernel's 1.3 ms.

**llama.cpp — partial.** llama.cpp does not have vLLM's specific defect (one thread block per request-head); its MMA kernel is a proper tensor-core kernel. But it has a different, larger, equally invisible multi-query tax on exactly the same step. There is no flag to force VEC and no flag to keep the cache quantized on the MMA path — CANNOT, foreclosed at fattn-mma-f16.cuh:1962-1963. The actionable consequences are testable today: (a) --spec-draft-n-max 1 keeps quantized-KV attention on VEC; (b) at long context, -ctk f16 -ctv f16 pays 4× the KV VRAM but no per-step dequant, so the crossover against q4_0 moves with depth.

**Equivalent here:** BEST_FATTN_KERNEL_MMA_F16 handles multi-query verify — but with quantized KV it dequantizes the whole cache to F16 every call

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/fattn.cu:469` · `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963` · `ggml/src/ggml-cuda/fattn-common.cuh:1022-1084` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912` · `ggml/src/ggml-cuda/fattn.cu:534-568`

**Effort:** new-backend · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** No knob, but the diagnosis transfers and is probably the most important thing in this whole slice for us. With -ctk q4_0 -ctv q4_0, VEC is chosen only when Q->ne[1] <= 2, i.e. at most 1 drafted token. Any real speculative step therefore runs MMA_F16, and MMA passes need_f16_K = need_f16_V = true unconditionally, so K and V for the FULL padded n_kv of every layer are expanded to F16 into scratch on every decode step. That cost is O(context) per step and it is paid only when speculating. It is a concrete candidate mechanism for this project's own recorded draft-mtp result of +81 % at 16K and −71 % at 131,072 on the same artifact.

### VLLM_SPEC_DECODE_ATTN_QMAX pinned to k+1 because a CUDA graph holds the buffer addresses
**Where (theirs):** `single-user/start_qwen.sh:134-137`

**What it does.** Sizes the split-KV kernel's partial-result buffers once, for the longest query block the server will ever see, and never grows them.

**Mechanism.** `export VLLM_SPEC_DECODE_ATTN_QMAX=${VLLM_SPEC_DECODE_ATTN_QMAX:-$((DRAFT_TOKENS + 1))}`.

**Why they needed it.** "The split-KV verify attention (patches/spec-decode-attn.patch) sizes its partial buffers once for the longest query block it will see -- a captured CUDA graph holds their addresses, so they must not be grown later." (single-user/start_qwen.sh:134-136)

**llama.cpp — partial.** CANNOT #2 in the attention area states this outright — the graph key is the first node pointer of the split, not the shape, so several decode lengths cannot coexist as captured graphs, and an alternating step size never captures. There is no padding mechanism in llama.cpp to make the verify batch a constant width, so this cannot be fixed by configuration; it can only be measured and, if large, argued about upstream.

**Equivalent here:** no buffer to pin, but the same underlying constraint exists: llm_graph_params::allow_reuse requires ubatch.n_tokens to be identical, and CUDA graph warmup resets on any node-property change

**Evidence (llama.cpp):** `src/llama-graph.h:785` · `src/llama-context.cpp:1332-1372` · `ggml/src/ggml-cuda/ggml-cuda.cu:4253-4268` · `ggml/src/ggml-cuda/ggml-cuda.cu:2574-2576` · `tools/server/server-context.cpp:617-619`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Potentially high as a diagnosis, with no fix available by flag. The vLLM insight — pin the step shape or lose the captured graph — applies here with more force, because llama.cpp's step size under speculation is 1 + accepted, which varies every step. The chain is explicit in the map: n_tokens changes → no graph reuse → sched reset and re-split → new uid → node properties differ → warmup_complete reset → eager execution. Re-arming needs two consecutive identical calls. So a variable-acceptance workload may never capture at all. The measurement is cheap and available today: compare `graphs reused` on the same prompt with --spec-type none versus ngram-mod versus draft-dflash. If reuse collapses under speculation, part of the speculation win is being paid back in eager execution.

### Lookup-augmented drafting with a verify block decoupled from the drafter's block
**Where (theirs):** `single-user/start_qwen.sh:117-132`

**What it does.** When the model is reproducing something already in its context, draft from the context rather than from the drafter; and allow the verify block to be longer than the drafter's trained block, filling the surplus positions from the request's own context at zero drafter cost.

**Mechanism.** `export VLLM_DFLASH2_LOOKUP=${LOOKUP:-1}` (patches/dflash2-lookup-drafting.patch). `DRAFT_TOKENS=${DFLASH_TOKENS:-7}` is the *verify* block. "The DFlash2 checkpoint always proposes the 7 tokens it was trained for, and any position past that is filled from the request's own context, costing the drafter nothing. The block length is adaptive -- the long block is only scheduled while the lookup is actually firing -- so ordinary steps still verify 8 tokens."

**Why they needed it.** "DFLASH_TOKENS=15 is 'reproduction mode': +50% where the model reproduces its context (388 vs 259 tok/s reproducing a document verbatim) and +9% on the short-prompt C1 set, against 3-20% on long-context work that mixes prose with quoting, 4 request slots instead of 8 and 56k of context instead of 64k. Worth setting for a coding assistant applying edits or a RAG front-end quoting sources; the default stays 7." (single-user/start_qwen.sh:127-131)

**Their numbers.** DFLASH_TOKENS=15: +50% on verbatim reproduction (388 vs 259 tok/s), +9% on the short-prompt C1 cohort, 3-20% on mixed long-context work. Cost: 4 request slots instead of 8, 56k context instead of 64k.

**llama.cpp — partial.** CANNOT #3 and #4 foreclose true fusion, so this is partial rather than already-have-it. But the tuning surface that is available maps closely onto what the vLLM technique buys: a lower n_min lets a shorter context match still produce a draft instead of falling through to DFlash, and n_match trades hash precision against hit rate (a warning fires below 16). Sweep at the depth the coding agent works at; the depth-transfer rule applies.

**Equivalent here:** --spec-type draft-dflash,ngram-mod (a fallback chain, not a fusion) with --spec-ngram-mod-n-match / -n-max / -n-min

**Evidence (llama.cpp):** `common/speculative.cpp:2542-2552` · `common/speculative.cpp:1992-2004` · `common/common.h:351-356` · `common/arg.cpp:4163-4192` · `common/speculative.cpp:1924-1927`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High, and it reframes the +48.5 % result. llama.cpp already has both halves — a learned drafter and a context lookup — and already runs them together, but as a fallback chain rather than one decoupled block. Two consequences nobody appears to have acted on. First, the priority order is hardcoded with every n-gram speculator ABOVE every model-based one, so ngram-mod wins every step it fires and draft-dflash only gets the steps ngram-mod declined; the order you type is discarded. Second, ngram-mod's three parameters (n_match=24, n_max=64, n_min=48) are tunable over 0..1024 and almost certainly never swept — and the n_min=48 gate is all-or-nothing, so it emits 64 tokens or nothing at all. Lowering n_min is the closest llama.cpp analogue to "let the lookup contribute a partial tail", and it is one flag.

### Prefix caching with recurrent-state resume, and a block-aligned prefix hash unit
**Where (theirs):** `single-user/start_qwen.sh:206-214` · `batch/start_qwen.sh:63-72`

**What it does.** Reuses the KV of a shared prompt prefix across requests AND resumes the recurrent (GDN) state from the last cached block boundary, rather than re-running the prompt through the recurrent layers; and forces the prefix hash unit to match KVarN's tile size.

**Mechanism.** `EXTRA_ARGS="--enable-prefix-caching --mamba-cache-mode align ${EXTRA_ARGS}"`; when CTX=huge additionally `--prefix-match-unit 128` because "KVarN runs --block-size 128; match the prefix hash unit to its tile so cache hits land on tile boundaries (a non-multiple of 128 corrupts the pool)" (single-user/start_qwen.sh:212-214).

**Why they needed it.** single-user: "Turn-2+ of a chat with a 24k document goes from ~23 s to ~1 s; costs one extra state page per request (~16% of the KV pool). Hybrid models keep this opt-in upstream." (single-user/start_qwen.sh:208-209). batch: "64 requests sharing a 5.8k-token system prompt (conc 32) take 222 s without it and 17 s with it. Costs ~14% of the KV pool (223,821 -> 193,298 tokens) and nothing on workloads with no shared prefix (870 vs 876 tok/s on the 128/512 row)." (batch/start_qwen.sh:66-69)

**Their numbers.** Single-user: turn 2+ of a 24k-document chat 23 s → ~1 s; ~16% of the KV pool. Batch: 64 requests over a 5.8k shared system prompt at conc 32: 222 s → 17 s; pool cost ~14% (223,821 → 193,298 tokens); no-shared-prefix cost 870 vs 876 tok/s.

**llama.cpp — partial.** Judged against llama.cpp's own memory code rather than transplanted. Two benchmark traps that belong with this verdict: [TAG_PROMPT_LOGITS] forces n_past-- on a fully cached prompt, so a 1000-token cached prompt reports cache_n=999 and prompt_n=1 with a meaningless prompt_per_second; and with --cache-idle-slots on and kv_unified false (which is what explicit -np 1 gives you), the idle slot's KV is copied to RAM but NOT cleared from VRAM, and that copy lands inside the next request's prompt_ms window.

**Equivalent here:** --cache-prompt (on by default, in-slot common-prefix reuse), -cram / --cache-ram (8192 MiB host cache, on by default), --cache-idle-slots, -ctxcp / -cms context checkpoints

**Evidence (llama.cpp):** `src/llama-memory-hybrid.cpp:190-202` · `tools/server/server-context.cpp:2283-2285` · `src/llama-kv-cache.cpp:1176-1178` · `tools/server/server-context.cpp:3313-3323` · `tools/server/server-context.cpp:2355-2363`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate to high for a coding agent, but the recurrent-resume half is weaker than vLLM's. In-slot prefix reuse and the RAM prompt cache are already on and already give the turn-2 win. The checkpoint mechanism does NOT do what the vLLM technique does on a hybrid model: the server always creates checkpoints with LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, and on a non-SWA hybrid that stores ONLY the recurrent state and skips the attention KV entirely — so a checkpoint restores DeltaNet state and nothing else, and the attention KV must be recomputed. Separately, --cache-reuse and context shift are unreachable on an M-RoPE / I-M-RoPE model, so the block-aligned-hash-unit idea has no target. The untuned knobs are -cram sizing, -ctxcp (default 32) and -cms (default 8192).

### PIECEWISE CUDA graph capture forced, because FULL corrupts one prompt length in 128
**Where (theirs):** `single-user/start_qwen.sh:215-252`

**What it does.** Forces `cudagraph_mode: PIECEWISE` whenever prefix caching is on at CTX=huge, on correctness grounds, after establishing that the corruption is not drafter-specific and that the performance cost is near zero at the depths this mode serves.

**Mechanism.** `[ "$CTX" = "huge" ] && CG_MODE=",\"cudagraph_mode\":\"${CUDAGRAPH_MODE:-PIECEWISE}\""`, folded into `--compilation-config`. PIECEWISE keeps the compiled graphs and leaves only the multi-query verify uncaptured.

**Why they needed it.** "Treat CUDAGRAPH_MODE=FULL_AND_PIECEWISE as unsafe: what it does is corrupt one prompt length in every 128 (gotcha 37, bench/bugb_sweep.py), which a coarse sweep reads as a length threshold and a lucky sweep misses entirely. This is NOT dflash2-only, which is what we used to claim here. ... The broken residue is R = 117 + k: 124 at DFLASH_TOKENS=7, 122 at 5, 120 at 3, and the same 120 under mtp k=3. Equivalently the last 128-token tile has 11-k free slots, i.e. verify block + free = 12 in every config measured. k=5 is what rules out the attention block size as the driver -- same 2176 block as k=7, different residue. mtp and dflash2 at the same draft count break identically, so the drafter is not implicated ... It also needs a prefix-cache HIT to fire at all, which is why PREFIX_CACHE=0 always looked clean." (single-user/start_qwen.sh:228-243). Closing note: "The old claim here that forcing it 'would cost decode for nothing' was wrong twice." (single-user/start_qwen.sh:248-249)

**Their numbers.** Broken residue R = 117 + k (124 at k=7, 122 at k=5, 120 at k=3, 120 under mtp k=3). DFlash2 long context, labd copy@20k: FULL 1.97 tok/step 38 tok/s vs PIECEWISE 7.83 tok/step 132 tok/s (3.5x). Short prompts de/en/code: FULL 78/125/202 vs PIECEWISE 74/102/176 tok/s (-13..18%). Depth ladder 8k/16k/32k/50k DFlash2: 112/78/69/58 FULL vs 109/86/73/56 PIECEWISE. Same ladder MTP: FULL 87.8/86.1/70.4/63.5 vs PIECEWISE 93.5/83.8/70.3/59.6. Under GPU passthrough on a VM the uncaptured verify is launch-bound and costs 2-3x instead (PR #13).

**llama.cpp — partial.** llama.cpp's capture is keyed on the split's first node pointer with a two-call warmup, which is a different structure from vLLM's mode enum — there is nothing to set to PIECEWISE. The switches exist and are documented as diagnostic rather than performance flags, which is exactly how they should be used here.

**Equivalent here:** GGML_CUDA_DISABLE_GRAPHS and LLAMA_GRAPH_REUSE_DISABLE — two all-or-nothing kill switches, no mode selector

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/common.cuh:1255-1259` · `src/llama-context.cpp:278-286` · `ggml/src/ggml-cuda/ggml-cuda.cu:4249-4274` · `ggml/src/ggml-cuda/ggml-cuda.cu:2574-2576`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate as a diagnostic, nil as a fix. There is no piecewise-vs-full choice here; capture is all-or-nothing per split. What transfers is the attribution method: flipping each kill switch independently and re-measuring is the cheapest way to decide whether a regression belongs to graph reuse (llama-level) or graph capture (CUDA-level), and both are one environment variable. Note GGML_CUDA_DISABLE_GRAPHS is checked for EXISTENCE, not value — setting it to 0 still disables graphs. The correctness lesson (a corruption that only fires on a cache hit at one length residue, invisible to a coarse sweep) pairs with the residue-sweep verdict.

### Idempotent model preparation driven by inspecting the safetensors weight map
**Where (theirs):** `docker/prepare.sh:15-38` · `docker/prepare.sh:41-63`

**What it does.** Decides which of the six preparation steps still need to run by looking at the actual tensor names in `model.safetensors.index.json`, rather than by touching marker files, and re-checks afterwards that nothing is still missing.

**Mechanism.** `state()` prints the steps still to do: `download` if any required file or shard is absent; `lm_head` if `lm_head.weight_packed` is not in the weight map; `embed` if no key ends with `embed_tokens.weight_packed`; `mtp` if `mtp.layers.0.mlp.down_proj.weight_packed` is absent; `draft` if `mtp.draft_lm_head.weight_packed` or `mtp_draft_vocab_ids.pt` is absent; `fast` and `dflash2` by directory existence. After running the steps, `LEFT=$(state | sed 's/\bdflash2\b//')` and a non-empty remainder is a hard failure — dflash2 is stripped because it is optional.

**Why they needed it.** The same predicate set is duplicated in verify.sh:78-88, so the preparation script and the verifier agree by construction on what "prepared" means. It also guards the tokenizer trap: "tokenizer.json belongs in this list: without it transformers builds an empty vocabulary rather than failing, and the dir stays servable-looking all the way to 'ReasoningConfig: failed to tokenize reasoning strings' at startup." (docker/prepare.sh:20-22)

**Their numbers.** ~19.5 GB base download (resumable), ~1 GB fast variant, ~1 GB W4A16 DFlash2 drafter. Runs CPU-only, no GPU needed.

**llama.cpp — partial.** Judged against llama.cpp's loader rather than transplanted. The weight-map-inspection idea has a weak analogue at best — the DEBUG-level line naming the first tensor moved off the GPU and a count of the rest is invisible at default verbosity, which is the closest thing to a "what actually got prepared" readout.

**Equivalent here:** --check-tensors (validate every tensor's rows at load); a GGUF is a single file so most of the multi-step prep has no counterpart

**Evidence (llama.cpp):** `common/arg.cpp:2882-2888` · `src/llama-model-loader.cpp:1556-1560` · `src/llama-model-loader.cpp:1459-1462` · `src/llama-model-loader.cpp:723-785` · `src/llama-model-loader.cpp:1341-1345`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Low to moderate. There are no six preparation steps to make idempotent — the GGUF either loads or it does not. The one transferable piece is integrity: --check-tensors runs ggml_validate_row_data per tensor and throws 'found tensors with invalid data', which is cheap one-time insurance on a 6.77 GB artifact downloaded once and measured against for months. Run it once, not in the serving path: it disables the async pinned-memory upload fast path outright. Related silent-wrong worth one assertion: the printed `file type` is a guess from the most-common tensor type and is OR'd with GUESSED, so it can misreport the quant without error.

## already have it — 14

### num_speculative_tokens discovered from Prometheus label cardinality
**Where (theirs):** `bench/labd_accept.py:227-229` · `bench/labd_accept.py:115-116` · `bench/labd_accept.py:139-144`

**What it does.** Reads the server's actual draft-slot count out of the server rather than trusting a command-line argument, by counting the distinct `position="N"` series on the per-position acceptance counter.

**Mechanism.** `PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"`, parsed with `POS_RE = re.compile(r'position="(\d+)"')`; then `NSPEC = max(len(metrics()[3]), BLOCK + 1)` (labd_accept.py:229).

**Why they needed it.** "num_speculative_tokens, straight from the server: the per-position counter has one series per draft slot, so its cardinality is what the server was started with." (labd_accept.py:227-228)

**llama.cpp — already have it.** Both routes exist. The per-position counter is sized common_speculative_n_max, i.e. the MAX over enabled types (64 for ngram-mod, draft.n_max for model-based ones), so its cardinality does NOT tell you the per-impl draft width in a chained profile — /props does. /props is not gated by --props and answers even while sleeping.

**Equivalent here:** GET /props → `speculative.types`, plus llamacpp:spec_decode_num_accepted_tokens_per_pos_total label cardinality

**Evidence (llama.cpp):** `tools/server/server-task.cpp:83` · `tools/server/server-context.cpp:650` · `tools/server/server-context.cpp:3899` · `common/speculative.cpp:2351-2385` · `common/arg.cpp:4153-4162`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate. /props is better than counting labels: it reports the comma-joined, accumulated, de-duplicated type list actually in effect, which matters because --spec-type APPENDS rather than replaces and a `none` in one invocation is silently re-appended to by a later one. This is the fastest way to confirm a profile really enabled what the script thinks it did.

### head/tail acceptance split at the drafter's block boundary
**Where (theirs):** `bench/labd_accept.py:268-270` · `bench/labd_accept.py:289-291`

**What it does.** Splits accepted tokens into those landing in positions the drafter proposed versus positions filled from the request's own context (the lookup tail), so the two sources of acceptance can be attributed separately.

**Mechanism.** For each position label in the per-position counter, the delta is added to `c['head']` when `pos < BLOCK` and to `c['tail']` otherwise (labd_accept.py:270); both are then divided by steps and printed as `head=` / `tail=`.

**Why they needed it.** The DFlash2 checkpoint proposes only the 7 tokens it was trained for; positions past that are filled from the request's context (single-user/start_qwen.sh:121-125), so a single tok/step number cannot tell you which mechanism earned the acceptance.

**llama.cpp — already have it.** llama.cpp never merges two speculators into one draft — exactly one impl's flat token list is used per sequence per step (speculative.cpp:2725-2726, 2753-2755) — so there is no head/tail within a single draft. The attribution question is therefore "which impl fired", which the per-impl counters answer exactly rather than by inference. Both instruments already exist.

**Equivalent here:** llamacpp:spec_decode_num_accepted_tokens_per_pos_total (per-position histogram) + the per-impl SPC_TRC counters

**Evidence (llama.cpp):** `tools/server/server-context.cpp:3883-3903` · `common/speculative.cpp:988-996` · `common/speculative.cpp:2725-2726` · `common/speculative.cpp:2741` · `common/speculative.cpp:2372-2374`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High for the draft-dflash,ngram-mod pair. DFlash is clamped to block_size-1 (15 with a stock 16-wide sidecar), while ngram-mod can draft up to 64 — so any acceptance at position ≥ 16 can only have come from ngram-mod. Combined with the per-impl line this attributes the +48.5 % between the two sources without any modelling.

### Metrics scraping that sums across engines and survives prometheus_client's _created lines
**Where (theirs):** `bench/labd_accept.py:127-145`

**What it does.** Parses /metrics into (drafts, draft_slots, accepted, {position: accepted}) summed over all engine label sets, without being fooled by the `_created` timestamp series prometheus_client emits alongside every counter.

**Mechanism.** Splits the metric name as `line.split('{',1)[0].split(' ',1)[0]` and the value as `line.rsplit(' ',1)[-1]`, then `+=` into a dict keyed by bare name — so multiple engine label sets accumulate. `_created` lines have a different bare name and fall through both branches (labd_accept.py:135-136 comment).

**Why they needed it.** Counters are per-engine-labelled; a naive `startswith(name + ' ')` misses them entirely, and `_created` lines silently double a naive prefix match. Note the older harnesses (labd_bench.py:45-50, labd_soak.py:72-79) still use the naive prefix form.

**llama.cpp — already have it.** The parsing hazards are properties of prometheus_client and of vLLM's multi-engine labelling, neither of which is present here. --metrics is off by default (common.h:655), so the endpoint itself is an exists-but-unused prerequisite for several other verdicts on this list.

**Equivalent here:** GET /metrics (--metrics), single process, hand-written text format

**Evidence (llama.cpp):** `tools/server/server-task.cpp:1520-1614` · `tools/server/server-context.cpp:3899` · `common/common.h:655` · `common/arg.cpp:3533-3539`

**Effort:** one-flag · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Low. llama-server is one process with no engine label set and writes its own Prometheus text, so the multi-engine summing and _created problems do not exist. The one label that does appear is position="N" on the per-position counter, so a naive startswith(name + ' ') prefix match still misses it — worth knowing when writing the scraper.

### Streamed decode-rate convention: first-token-to-last, prefill excluded, (n-1) tokens
**Where (theirs):** `bench/labd_bench.py:126-132` · `bench/conc_ladder.py:95-96` · `bench/conc_ladder.py:118` · `bench/demo_capture.py:87-92`

**What it does.** Defines one decode-rate convention used across every harness and the READMEs: time from the first streamed content chunk to the last, dividing by (completion_tokens - 1), with TTFT reported separately.

**Mechanism.** `t_first` is set on the first chunk carrying content (not on the first SSE event); `decode = (out - 1) / max(t_end - t_first, 1e-3)`, `ttft = t_first - t0`, `e2e = out / (t_end - t0)` (labd_bench.py:129-132). conc_ladder.py:118 computes per-stream as `sum((v['ntok']-1)/v['decode_s'])/n`. demo_capture.py:92: `rate = (n - 1) / decode_s`.

**Why they needed it.** "measured the way this repo reports decode — streamed, so the prefill is not averaged in" (labd_bench.py:4-5). conc_ladder.py:7-8: "Per-stream decode rate is measured first-token to last-token, so it excludes TTFT; the aggregate column does not, and is dominated by prefill once several long prompts arrive at once."

**llama.cpp — already have it.** n_gen_steps() = n_gen - 1 at server-common.h:400-402 is the same (n-1) convention; t_gen_us() is floored at 1 us. Adopting the vLLM wording changes nothing; the value is in documenting llama.cpp's two asymmetries so a reader cannot misread prompt_per_second on a cached prompt.

**Equivalent here:** the per-request `timings` block: predicted_per_second, predicted_ms, prompt_ms, cache_n

**Evidence (llama.cpp):** `tools/server/server-common.h:346-425` · `tools/server/server-common.h:400-402` · `tools/server/server-common.h:363-366` · `tools/server/server-context.cpp:3053`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** High as a consistency check, because llama.cpp's convention is already exactly this and the definitions differ from the obvious reading in two ways worth writing down: prompt_n EXCLUDES cached tokens while prompt_ms runs from when the slot entered PROCESSING_PROMPT (so RAM-cache load and checkpoint restore land inside the numerator but outside prompt_n), and predicted_per_second divides by n_gen-1 so a one-token completion reports 0 t/s.

### tok/step as the cross-session-stable comparison metric, e2e as the unstable one
**Where (theirs):** `bench/real_rep.sh:1-6` · `bench/real_rep.sh:12-13` · `bench/real_rep.sh:20-26`

**What it does.** Repeats one fixed cohort N times and reports tokens/step and ms/step per repeat, explicitly because those survive across sessions while end-to-end throughput does not.

**Mechanism.** `snap()` greps `^vllm:spec_decode_num_(drafts|accepted_tokens)_total`, filters out `created` lines, takes the last field. Around each repeat it diffs the pair and computes `steps = Δdrafts`, `tok/step = 1 + Δaccepted/steps`, `ms/step = 1000*duration/steps`, plus `decode = 1000/MeanTPOT`.

**Why they needed it.** "Prints tokens/step and ms/step per repeat, which is what to compare when two configurations look different (greedy e2e moves between sessions; tokens/step does not)." (real_rep.sh:3-4)

**Their numbers.** Cohort is fixed at 8 realistic prompts × 1,024 tokens, max-concurrency 1; default 3 repeats.

**llama.cpp — already have it.** The metric is already computed and already printed at INFO on every completion — nothing to add. server-context.cpp:2966 increments n_draft_tokens on the already-truncated draft, n_draft_accepted at 3877, n_draft_verif_steps per verify; the line at 634-637 renders both. The one caveat: dp.n_max truncation near the context edge shortens the draft before it is counted, so mean len falls at depth for a reason that is not the speculator's quality.

**Equivalent here:** `draft acceptance = %0.5f (%d accepted / %d generated), mean len = %5.2f` printed per completion, plus draft_n/draft_n_accepted in the timings JSON

**Evidence (llama.cpp):** `tools/server/server-context.cpp:634-637` · `tools/server/server-context.cpp:2966` · `tools/server/server-context.cpp:3877` · `tools/server/server-context.cpp:441-460` · `tools/server/server-common.h:346-425`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Highest-leverage item on this list for this repo. CLAUDE.md forbids comparing raw decode across boots because free VRAM moves 9,326–10,732 MiB and --fit follows it; mean len (= 1 + accepted/verif_steps) does not depend on how many layers landed on the GPU or how big the context ended up. Reporting mean len alongside t/s converts a class of comparison the repo currently calls noise into a legitimate one.

### No max_tokens on demo prompts, so neither lane is truncated at a flattering point
**Where (theirs):** `bench/demo_capture.py:37-40` · `bench/demo_capture.py:54-61`

**What it does.** Lets every demo answer run to its own stop token rather than to an operator-chosen cut-off.

**Mechanism.** `run()` takes `max_tokens=None` by default and only inserts the field into the payload `if max_tokens:` — used solely for the warm-up call.

**Why they needed it.** "No max_tokens -- each answer runs to its own stop token, so the video shows real completions rather than a truncation point chosen to flatter one lane." (demo_capture.py:38-40)

**llama.cpp — already have it.** Nothing to add on the llama.cpp side; the default already does the right thing. The failure mode is the reverse of vLLM's — not an operator-chosen truncation but an invisible context-capacity truncation.

**Equivalent here:** n_predict defaults to unbounded; the slot stops at its own stop token

**Evidence (llama.cpp):** `tools/server/server-context.cpp:1819-1828` · `common/common.h:561` · `tools/server/server-schema.cpp:588-607`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Low. Worth one caveat if adopted: --context-shift is DISABLED by default, so an unbounded generation that reaches n_ctx is cut with stop_type LIMIT and truncated=true, logged only at DEBUG. An unbounded demo lane can therefore end on a context limit rather than on a stop token, silently.

### A streamed chunk is not a token: char-proportional attribution of the reported total
**Where (theirs):** `bench/demo_render.py:79-95`

**What it does.** Converts SSE chunk counts into token counts for the live readout, because with a long verify block one SSE event carries a whole accepted run.

**Mechanism.** `prepare()` sums characters over all chunks, then assigns each chunk a cumulative token count of `cum_chars * n_out / total_chars` — monotone, sums to the server-reported total, and exact for the one-token-per-chunk (unspeculated) lane.

**Why they needed it.** "A chunk is not a token. With a 16-token verify block DFlash2 delivers a whole accepted run in one SSE event, so counting chunks would show 29 where 400 tokens arrived. The server only reports the total, so split it across chunks by character count." (demo_render.py:81-85)

**Their numbers.** Counting chunks would show 29 where 400 tokens arrived.

**llama.cpp — already have it.** The vLLM workaround exists because the server reported only a total. llama.cpp does not have that limitation: server-context.cpp:1774 attaches tokens to the partial result and server-task.cpp:1056 emits them per chunk.

**Equivalent here:** `return_tokens` yields the exact ids per chunk; timings_per_token yields exact counts per chunk

**Evidence (llama.cpp):** `tools/server/server-context.cpp:1774` · `tools/server/server-task.cpp:1056` · `tools/server/server-schema.cpp:20-21`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None — the approximation is unnecessary here. llama-server can report the true token ids and the true running counts on every chunk, so counting chunks is never required and the char-proportional split would only introduce error. The underlying warning (with speculation, one SSE event can carry a whole accepted run) is still true and worth remembering when reading a stream by eye.

### W4A8 Marlin: int8 activations for the MLP GEMMs, with a layer-selection regex
**Where (theirs):** `batch/start_qwen.sh:10-15` · `batch/start_qwen.sh:58-61` · `batch/start_qwen.sh:98-99`

**What it does.** Routes the MLP GEMMs through the int8 tensor-core Marlin path (weights stay int4) for a large aggregate throughput gain, with a gentler variant and an off switch, selected by a regex on layer name.

**Mechanism.** `export VLLM_MARLIN_INPUT_DTYPE=$INT8_ACT` and `export VLLM_MARLIN_INT8_INCLUDE_RE=$INT8_LAYERS`, where INT8_ACT defaults to `int8` and INT8_LAYERS to `mlp` (gate_up_proj + down_proj). Needs both Marlin patches from patches/. verify.sh:49 asserts the env var is registered in vLLM's envs.py.

**Why they needed it.** "int8 tensor-core (W4A8) Marlin path for the MLP GEMMs, weights stay int4. Roughly +35% aggregate on top of the state change for +2.2% perplexity." (batch/start_qwen.sh:10-13). Single-user mode deliberately does not use it: "Int8 activations are pointless at batch size 1 (memory-bound), so this mode stays W4A16." (single-user/start_qwen.sh:37-38)

**Their numbers.** mlp (default): ~+35% aggregate, +2.2% perplexity. gate_up only: ~+15%, +0.9% PPL. INT8_ACT= (empty): pure W4A16, quality-neutral.

**llama.cpp — already have it.** I checked this in source rather than inferring it: mmq.cu's type switch includes IQ2_XXS and mmq.cuh has the corresponding tile/VDR entries. Dispatch order in ggml_cuda_mul_mat is MMF → MMVQ → MMQ → cuBLAS, and MMVQ (also int8-activation) handles ne11 <= 8, so both the decode and the batched paths are already quantised-activation.

**Equivalent here:** the MMQ path — ggml quantises activations to Q8_1 on the fly for quantised-weight matmuls, and IQ2_XXS is supported

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/mmq.cu:258-326` · `ggml/src/ggml-cuda/mmq.cuh:86` · `ggml/src/ggml-cuda/mmq.cu:312-314` · `ggml/src/ggml-cuda/ggml-cuda.cu:1853-1865`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero to gain — this is already the default and there is nothing to turn on. ggml_cuda_should_use_mmq lists GGML_TYPE_IQ2_XXS among the supported types, and on Ada turing_mma_available short-circuits it to true unconditionally, so the weight matmuls already run as int-weight × int8-activation tensor-core GEMMs. There is no layer-selection regex and no quality knob, which also means no perplexity cost to weigh.

### expandable_segments required, with a WSL2 escape hatch
**Where (theirs):** `batch/start_qwen.sh:16-18` · `batch/start_qwen.sh:93-94` · `single-user/start_qwen.sh:279-282`

**What it does.** Turns on PyTorch's expandable-segment allocator to survive the DeltaNet prefill kernels' transient workspace, while leaving it overridable because the feature is unavailable under WSL2.

**Mechanism.** `export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}` — the `:-` makes it overridable from .env.

**Why they needed it.** "expandable_segments is required: the DeltaNet prefill kernels allocate transient workspace and fragment the allocator, OOMs at util >= 0.978 without it" (batch/start_qwen.sh:16-18). And: "expandable_segments needs CUDA VMM, which WSL2's paravirt driver rejects ('CUDA driver error: device not ready' during Marlin repack) — set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False in .env on WSL2." (single-user/start_qwen.sh:279-282)

**Their numbers.** Without it, OOM at gpu-memory-utilization ≥ 0.978.

**llama.cpp — already have it.** Checked in source: ggml-cuda.cu:688-689 picks ggml_cuda_pool_vmm when info.devices[device].vmm is set, otherwise a plain pool, and the choice is printed in the device banner. This is automatic rather than opt-in, so there is nothing here to adopt.

**Equivalent here:** ggml_cuda_pool_vmm — the CUDA VMM-backed pool, used automatically when the device reports VMM support

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:275-290` · `ggml/src/ggml-cuda/ggml-cuda.cu:536-552` · `ggml/src/ggml-cuda/ggml-cuda.cu:688-689` · `ggml/src/ggml-cuda/ggml-cuda.cu:337`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Low. ggml already uses a virtual-memory-management pool when the device advertises it (queried via CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED at load and reported in the device line), which is the same fragmentation remedy PyTorch's expandable_segments provides. There is no equivalent flag to set and no equivalent OOM-at-high-utilisation to fix. The WSL2 note is worth keeping only as a reminder that the pool selection is device-reported and the `VMM: yes/no` startup field tells you which one you got.

### draft_sample_method=probabilistic: drafts sampled rather than argmax'ed
**Where (theirs):** `single-user/start_qwen.sh:20-21` · `single-user/start_qwen.sh:202`

**What it does.** Samples the drafter's proposals from its distribution instead of taking the argmax, which raises acceptance whenever the target is sampling at temperature > 0.

**Mechanism.** `SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"`.

**Why they needed it.** "draft_sample_method=probabilistic: drafts are sampled, not argmax'ed, which lifts acceptance at temperature > 0" (single-user/start_qwen.sh:20-21). And the standing correctness claim: "Speculative decoding is exact: none of this changes what gets sampled." (single-user/start_qwen.sh:22)

**llama.cpp — already have it.** The draft sampler is one common_sampler per sequence built on the draft model's vocab with samplers = {TOP_K}, top_k = 10; sampling.cpp:405 appends dist unconditionally. CANNOT #6 records that the block which would have made this configurable is commented out, so there is no argmax variant to compare against without a fork.

**Equivalent here:** the fixed draft sampler chain — top_k(10) followed by an always-appended dist sampler

**Evidence (llama.cpp):** `common/sampling.cpp:400-406` · `common/speculative.cpp:226-236` · `common/speculative.cpp:1001-1008` · `common/speculative.cpp:209-224`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero to gain: llama.cpp already samples drafts rather than taking the argmax, and it is not configurable in either direction. common_sampler_init always appends llama_sampler_init_dist(seed) at the end of the chain, and that is also what fills in the .p values that --spec-draft-p-min then tests. So the acceptance lift this vLLM flag buys is already in the baseline here.

### Async scheduling turned OFF so the worker can choose the draft count
**Where (theirs):** `single-user/start_qwen.sh:138-144` · `single-user/start_qwen.sh:254-258`

**What it does.** Runs the scheduler synchronously when the adaptive long verify block is in use, because that is the only path on which vLLM feeds the worker's requested draft count back to the scheduler.

**Mechanism.** When `VLLM_DFLASH2_LOOKUP=1` and `DRAFT_TOKENS > 7`, set `ASYNC_SCHED=${ASYNC_SCHED:-0}`; then `ASYNC_ARGS=$([ "${ASYNC_SCHED:-1}" = 1 ] && echo --async-scheduling || echo --no-async-scheduling)`. The comment notes `--async-scheduling` is already the default in 0.27.1, so `--no-async-scheduling` is what actually changes behaviour.

**Why they needed it.** "Adaptive block length means the worker tells the scheduler how many draft tokens to put up for verification next step, and vLLM only feeds that back on the synchronous scheduling path (async scheduling pads every decode step to num_speculative_tokens). Measured cost of losing async scheduling at batch 1: under 1%." (single-user/start_qwen.sh:139-142)

**Their numbers.** Cost of losing async scheduling at batch 1: under 1%.

**llama.cpp — already have it.** Nothing to change. One llama.cpp-specific wrinkle worth carrying: the truncation happens AFTER the draft is built, so an ngram speculator can burn a full 64-token lookup and have it cut to a handful near the context edge, logged only at SPC_DBG — the cost is paid and invisible.

**Equivalent here:** the server is synchronous and already feeds the per-call budget dp.n_max to the speculator each step

**Evidence (llama.cpp):** `tools/server/server-context.cpp:441-460` · `tools/server/server-context.cpp:2936-2946` · `common/speculative.cpp:2728-2733`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. There is no async scheduling path to disable and no padding of every decode step to a fixed draft width. get_n_draft_max computes n_ctx - prompt.n_tokens() - 2, further min'd with n_remaining()-1, per generation step, and it is applied inside common_speculative_draft. The feedback loop this vLLM flag exists to restore is already the only behaviour here.

### Tool-call parser qwen3_coder pinned with an anti-correction warning
**Where (theirs):** `single-user/start_qwen.sh:260-276` · `batch/start_qwen.sh:74-90`

**What it does.** Sets `--enable-auto-tool-choice --tool-call-parser qwen3_coder` on both launchers, with a comment aimed squarely at a future maintainer who would 'fix' it to hermes.

**Mechanism.** `TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}`; `TOOL_ARGS=$([ "${TOOLS:-1}" = 1 ] && echo --enable-auto-tool-choice --tool-call-parser $TOOL_PARSER)`. Both flags are required or vLLM returns 400 on any request carrying `tools` with tool_choice "auto".

**Why they needed it.** "The parser has to match the format the chat template asks the model for, and Qwen3.8's asks for XML -- <tool_call><function=NAME><parameter=K>V</parameter> -- NOT the JSON body that hermes, the usual answer for a Qwen model, reads. Getting that wrong does not error: the call comes back as ordinary content and the client sees no tool_calls, which reads as the model being bad at tools rather than as a misconfigured server. The name is the call format, not the checkpoint -- nothing here is Qwen3-Coder. qwen3_coder, qwen3_xml and mimo are three names for one Qwen3EngineToolParser in 0.27.1, which is the tool-side adapter of the same parser engine that --reasoning-parser qwen3 already uses (vllm/parser/qwen3.py)." (single-user/start_qwen.sh:264-274)

**llama.cpp — already have it.** Checked in source: common/chat.h declares generic PEG formats (PEG_SIMPLE, PEG_NATIVE, and two model-specific ones) selected from the template rather than from a flag, so the vLLM name-the-parser problem is structurally absent. --jinja is already the server default.

**Equivalent here:** the tool-call format is derived from the chat template automatically; there is no --tool-call-parser to set wrongly

**Evidence (llama.cpp):** `common/chat.h:230-238` · `common/chat.cpp:759-781` · `common/arg.cpp:1394-1398` · `tools/server/server-context.cpp:1402-1428`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Low, and the specific failure mode does not exist — llama.cpp has no parser selector, so it cannot be set to the wrong one. The isomorphic risk is upstream of the parser: if the GGUF carries no tokenizer.chat_template, common_chat_templates_init falls through to the built-in CHATML source, and a literal source of "chatml" is treated as empty and falls through the same chain. Then the model is asked for a format it was not trained on and the tool call comes back as ordinary content — the same symptom, one layer earlier. One assertion on GET /props chat_template covers it.

### max-num-seqs 8 in single-user mode, justified by state slots rather than by latency
**Where (theirs):** `single-user/start_qwen.sh:35-38` · `single-user/start_qwen.sh:196` · `single-user/start_qwen.sh:201`

**What it does.** Caps concurrent sequences at 8 in low-latency mode (2 at CTX=huge with DFlash2, 4 at CTX=long or with a long verify block).

**Mechanism.** `MAX_SEQS=${MAX_SEQS:-8}` at single-user/start_qwen.sh:196 and :201; overridden to 2 (CTX=huge) or 4 (CTX=long / DRAFT_TOKENS>7) earlier in the DFlash2 branch.

**Why they needed it.** "max-num-seqs is 8 here: fewer state slots to reserve (each request holds k+1 recurrent-state slots), and past a handful of concurrent users you should be running batch mode anyway." (single-user/start_qwen.sh:35-37)

**Their numbers.** Each request holds k+1 recurrent-state slots.

**llama.cpp — already have it.** The flag exists and is already set correctly. The map calls the -np 1 versus no--np difference the single largest behavioural change in the server area, which makes it worth an assertion in the launch script rather than a silent reliance on the default.

**Equivalent here:** -np / --parallel (already at 1 in this profile)

**Evidence (llama.cpp):** `common/arg.cpp:1401` · `tools/server/server.cpp:151-155` · `src/llama-context.cpp:290-303` · `tools/server/server-context.cpp:1600-1602`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Nil to change — the profile is already at the minimum. The justification transfers and is worth writing down here because llama.cpp's cost is different and sharper: without kv_unified, n_ctx_seq = n_ctx / n_seq_max padded to 256 and n_ctx is then REWRITTEN to n_ctx_seq * n_seq_max, so raising -np divides the per-slot context directly. One trap in the other direction: OMITTING -np is not the same as -np 1 — the server turns the auto default into n_parallel = 4 AND kv_unified = true, which changes idle-slot behaviour, n_ctx_seq derivation and try_clear_idle_slots all at once.

### Keyless serving flagged as a WARN because both launchers bind 0.0.0.0
**Where (theirs):** `verify.sh:134-139` · `batch/start_qwen.sh:101-104` · `single-user/start_qwen.sh:285-287`

**What it does.** Treats a missing API key as a warning rather than an error, but says explicitly why it is worth a warning at all.

**Mechanism.** Both launchers do `if [ -z "$VLLM_API_KEY" ] && [ -f "$REPO/api_key.txt" ]; then export VLLM_API_KEY="$(cat ...)"; fi` and serve unauthenticated if neither is present. verify.sh warns with the remedy inline.

**Why they needed it.** "A key is optional: with neither api_key.txt nor VLLM_API_KEY the launchers export nothing and vLLM serves unauthenticated, which is a fine way to run this locally. Worth a WARN rather than silence only because both launchers bind 0.0.0.0." (verify.sh:135-137). Remedy given: `openssl rand -hex 24 > api_key.txt`.

**llama.cpp — already have it.** The capability exists and the default is the safe one, which is the opposite of the vLLM situation the WARN was written for. The residual exposure is the always-on /slots endpoint and the key-bypassed health/model routes, not the key itself.

**Equivalent here:** --host (default 127.0.0.1) and --api-key; /health, /v1/health, /models and /v1/models bypass the key check

**Evidence (llama.cpp):** `common/common.h:604` · `tools/server/server-http.cpp:196-217` · `common/common.h:653` · `tools/server/server-context.cpp:2430` · `tools/server/server-context.cpp:1283-1289`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Low, because the default is already safe: llama-server binds loopback unless told otherwise, so keyless local serving is not exposed. Worth exactly one assertion in the launch script — if --host is ever set to 0.0.0.0, require --api-key. Two llama.cpp specifics to record if that assertion is written: four endpoints bypass the API-key check entirely, and GET /slots is ENABLED by default (prompt and generated text are withheld unless LLAMA_SERVER_SLOTS_DEBUG is set, but per-slot metrics and request params are not).

## impossible here — 4

### fuse_draft acceptance-policy tests: strong match takes the block, weak match needs drafter agreement
**Where (theirs):** `bench/test_lookup_kernels.py:76-93` · `bench/test_lookup_kernels.py:95-115` · `bench/test_lookup_kernels.py:116-137`

**What it does.** Pins the policy that decides whether the context lookup or the drafter wins a position, as a truth table.

**Mechanism.** With nmin=4, nstrong=8, agree_min=2: match_len=12/agree=0 → take (strong match alone suffices); match_len=4/agree=0 → don't; match_len=4/agree=2 → take (weak match plus drafter agreement); match_len=2/agree=7 → don't (below nmin, agreement cannot rescue it). Decoupled-block tests then assert that when the drafter proposed only `draft_block` positions, the tail is *always* marked used — "it needs a point mass" (test_lookup_kernels.py:96-97) — and takes the lookup's continuation only when the two sources agree on the head. `take_flags` tests separate "this request has something to put in the tail" from "the long block is worth its step time", which is "the host's decision (next_num_draft_tokens)" (test_lookup_kernels.py:116-117): match_len=12/valid=k → True; match_len=12/valid=2 → False (nothing left to fill); match_len=2/valid=k → False (match too short).

**Why they needed it.** The tail positions have no drafter distribution behind them, so the rejection sampler needs a point mass there or the whole block is rejected; and the worker-side "can I fill it" decision must not be confused with the host-side "is it worth it" decision.

**Their numbers.** Policy constants under test: nmin=4, nstrong=8, agree_min=2, long_min=6.

**llama.cpp — impossible here.** CANNOT #3 and #4 in the speculative map foreclose it: dp.drafting is cleared and the loop breaks at the first non-empty result, and the priority list is rebuilt from a bitmask so command-line order is discarded. Implementing fusion means patching common/speculative.cpp:2710-2756 — a fork, not a configuration.

**Equivalent here:** none — llama.cpp cannot combine two speculators into one draft

**Evidence (llama.cpp):** `common/speculative.cpp:2725-2726` · `common/speculative.cpp:2753-2755` · `common/speculative.cpp:2542-2552` · `common/speculative.cpp:2343-2349`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** None without forking. common_speculative_draft breaks out of the impl loop as soon as one impl returns a non-empty draft, so multiple --spec-type values give a fallback chain, never an ensemble — there is no fusion policy to pin. Worse for the current profile: the priority order is hardcoded with every n-gram speculator ranked ABOVE every model-based one, so in draft-dflash,ngram-mod, ngram-mod wins every step it fires and DFlash only sees the leftovers. That is a fact worth knowing about the +48.5 % pair.

### GDN decode kernel autotune sweep over BV × num_warps
**Where (theirs):** `bench/tune_gdn.py:5-8` · `bench/tune_gdn.py:11-40`

**What it does.** Benchmarks the fused recurrent Gated DeltaRule decode kernel across block-V sizes and warp counts at two realistic decode batch sizes, reporting both microseconds and achieved GB/s.

**Mechanism.** Calls `fr.fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](**args, num_warps=W)` directly with `grid = (cdiv(V,BV), N*HV)`, timed by `triton.testing.do_bench(warmup=10, rep=50)`. Bandwidth is computed as `gb = 2*N*HV*V*K*2/1e9` — the state read plus write, in bf16 — which is the right denominator because the kernel is state-traffic bound.

**Why they needed it.** The recurrent state traffic per decode step is what `--mamba-ssm-cache-dtype float16` halves (batch/start_qwen.sh:6-9), so knowing the kernel's achieved bandwidth is what tells you whether the dtype change or the launch config is the binding constraint.

**Their numbers.** Qwen3.8-27B decode shapes: H=16 qk heads, HV=48 v heads, K=V=128, 80 state slots. Batches 24 and 48; configs (BV,warps) ∈ {(32,1),(32,2),(32,4),(64,2),(64,4),(64,8),(128,4),(128,8),(128,16)}.

**llama.cpp — impossible here.** Both the tuning surface and the thing it tunes are missing. The one adjacent, cheap experiment is GGML_CUDA_GRAPH_OPT=1, which interleaves Q/K/V branches across extra streams — but it only fires on single-row (decode-shaped) nodes with a hardcoded fan-out of 3, so a speculative verify step of 4+ query tokens would not qualify.

**Equivalent here:** none — ggml-cuda has no runtime kernel-config knobs and no autotuner

**Evidence (llama.cpp):** `src/llama-model.cpp:2335-2336` · `src/llama-model.cpp:2274-2275` · `ggml/src/ggml-cuda/ggml-cuda.cu:4318-4344` · `ggml/src/ggml-cuda/ggml-cuda.cu:4380-4395`

**Effort:** new-backend · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** None. Block sizes and warp counts are compile-time constants in the .cu sources; changing them means editing ggml and rebuilding, and there is no dispatch table to sweep. The lever the vLLM sweep exists to price — recurrent state traffic — is also absent: type_r and type_s are GGML_TYPE_F32 literals at every construction site with no flag anywhere in common/arg.cpp.

### Draft vocabulary counted over the model's own outputs, not over web text
**Where (theirs):** `single-user/start_qwen.sh:10-13` · `verify.sh:84-85` · `docker/prepare.sh:53-54`

**What it does.** Scores the MTP draft head over a 40k-token vocabulary whose id list was derived from the model's own generations rather than from a web-text corpus, because every out-of-vocabulary draft token is a forced rejection.

**Mechanism.** `prepare/build_draft_vocab.py --ids prepare/draft_vocab_ids.json` builds `mtp.draft_lm_head.weight_packed` plus `mtp_draft_vocab_ids.pt`; verify.sh:84 checks both exist and WARNs that "single-user mode drafts with the full lm_head" otherwise.

**Why they needed it.** "the id list matters: a vocabulary counted over the model's OWN outputs covers 97.5% of what it generates (96% on code); the earlier web-text list only 92% (83% on code), and every miss is a forced rejection" (single-user/start_qwen.sh:11-13)

**Their numbers.** Own-output vocab: 97.5% coverage (96% on code). Web-text vocab: 92% (83% on code). Effect: 108 vs 98 tok/s greedy.

**llama.cpp — impossible here.** CANNOT #6 and #15 in the speculative map foreclose both halves: the sampler is hardcoded at four call sites with the configurable version commented out at speculative.cpp:209-224, and common_speculative_are_compatible is called from exactly one place, the draft-simple constructor.

**Equivalent here:** none — the draft sampler is fixed to {TOP_K} with top_k=10 on the draft model's own vocab, and the block that would make it configurable is commented out

**Evidence (llama.cpp):** `common/speculative.cpp:230-233` · `common/speculative.cpp:1001-1008` · `common/speculative.cpp:209-224` · `common/speculative.cpp:238-245` · `common/speculative.cpp:1005`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Unknown and unreachable by configuration. There is no vocabulary-restriction mechanism on the draft side; the only variation llama.cpp allows is DFlash2 reading selector_top_k from the GGUF instead of the literal 10. Restricting the draft vocabulary would mean re-exporting the sidecar with a smaller head — an artifact change, not a serving change. Worth noting: the vocab-compatibility guard runs only for draft-simple, so a dflash sidecar with a mismatched vocab is never checked at all.

### Sliding-window block promotion so the drafter's layers stop padding the pool
**Where (theirs):** `single-user/start_qwen.sh:88-94` · `single-user/start_qwen.sh:145-149`

**What it does.** Stops the DFlash2 drafter's 5 sliding-window layers from being allocated at the target layers' block granularity, which nearly doubles the token capacity of the same pinned pool.

**Mechanism.** patches/hybrid-sw-block-promote.patch (int8/Triton path) and patches/hybrid-kv-groups-v2-cudagraph.patch (bf16 path).

**Why they needed it.** "the same 5.2 GiB pool holds 136,429 tokens instead of 69,758, because patches/hybrid-sw-block-promote.patch stops the drafter's 5 sliding-window layers from taking 385 nearly-empty blocks" (single-user/start_qwen.sh:89-91). And: "patches/hybrid-kv-groups-v2-cudagraph.patch stops the drafter's 5 sliding-window layers from padding the target's attention/GDN layers (78 instead of 105 KB of pool per token), which is what makes 64k reachable here." (single-user/start_qwen.sh:145-147)

**Their numbers.** 385 nearly-empty blocks reclaimed; 69,758 → 136,429 tokens in 5.2 GiB (int8). Pool per token 105 KB → 78 KB (bf16).

**llama.cpp — impossible here.** Two independent foreclosures. Each KV layer's buffer type is the buffer type of the device that layer's weights were assigned to, with no per-layer override short of -nkvo (all-or-nothing to CPU) — CANNOT #19 in the KV area. And the draft context's memory is separate by construction.

**Equivalent here:** none — the draft model gets its own independent memory module, and per-layer KV placement cannot be overridden

**Evidence (llama.cpp):** `common/speculative.cpp:2464-2482` · `src/llama-kv-cache.cpp:209-217` · `tools/server/server-context.cpp:1188-1195` · `src/llama-model.cpp:2305`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. The structural cause is absent: the draft context always gets its OWN memory module sized from the inherited n_ctx/n_parallel, so there is no shared pool whose block granularity the drafter's layers could pad. And a Qwen3.5-style model declares no SWA at all, so --swa-full is force-disabled with a warning and the entire llama_kv_cache_iswa path is unreachable. There is nothing to promote.

## not applicable — 9

### Documented rejection of the three obvious forcing alternatives
**Where (theirs):** `bench/labd_accept.py:31-43`

**What it does.** Records why structured outputs, prompt_logprobs/echo, and logit_bias/allowed_token_ids/bad_words all fail as ways to force a fixed sequence, so a future reader does not re-attempt them.

**Mechanism.** Guided decoding is rejected because the scheduler filters draft tokens against the grammar before scheduling them (`grammar.validate_tokens(spec_token_ids)` in `update_draft_token_ids{,_in_output}`, padding rejects with -1) — that mutates `num_draft_tokens` and `num_accepted_tokens`, the exact counters being measured, and mutates them *differently* for a 7-slot and a 15-slot server. prompt_logprobs/echo score during prefill, so there are no decode steps and no drafts to count. logit_bias/allowed_token_ids/bad_words are per-request constants and cannot force a different token at each position.

**Why they needed it.** "That mutates num_draft_tokens and num_accepted_tokens -- the exact counters being measured -- and mutates them differently for a 7-slot and a 15-slot server. Non-starter here." (labd_accept.py:36-38)

**llama.cpp — not applicable.** This is a recorded negative, not a capability. The specific mechanism cited (vLLM's scheduler filtering draft tokens against the grammar before scheduling) has no llama.cpp counterpart — llama.cpp never pre-filters a draft; it drafts blind and the grammar is applied at verify time. So the conclusion transfers but the reasoning must be re-derived from llama.cpp's own code before it is written down.

**Equivalent here:** none — the three rejected mechanisms have different shapes here

**Evidence (llama.cpp):** `common/sampling.cpp:739` · `common/sampling.cpp:692-720` · `common/sampling.cpp:331-344` · `tools/server/server-schema.cpp:179-181`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Low as a technique, moderate as a warning. The llama.cpp versions of the same three dead ends are: a grammar is applied per drafted position in the verifier (sampling.cpp:739 / 692-720) so it does perturb acceptance; there is no echo/prompt_logprobs at all (n_probs covers generated tokens only); and logit_bias is a per-request constant (sampling.cpp:331-344) so it cannot force a different token per position. Worth one line in docs/tested so nobody re-attempts them.

### Frozen benchmark corpus so numbers stay comparable as the source docs change
**Where (theirs):** `bench/labd_bench.py:8-9` · `bench/labd_bench.py:53-60` · `bench/labd_accept.py:198-206`

**What it does.** Builds the benchmark document once from the repo's own markdown, writes it to a fixed path, and never rebuilds it — so a doc edit cannot silently move a benchmark number.

**Mechanism.** If `~/bench/labd_corpus.txt` does not exist, concatenate all `~/qwen-serving/*.md` and `*/README.md` sorted, then repeat the whole set until the text reaches 200,000 characters, and write it (labd_bench.py:53-60). Thereafter the file is read as-is and sliced to `CTX * 3.6` characters.

**Why they needed it.** "The corpus is frozen on first run (~/bench/labd_corpus.txt) so numbers stay comparable as the docs it is built from change." (labd_bench.py:8-9)

**Their numbers.** 200,000 characters ≈ 84k tokens; slice ratio assumed 3.6 chars/token.

**llama.cpp — not applicable.** No llama.cpp seam and none needed. The value is real but the evidence for it lives in the bench harness, not in llama.cpp, so I am not claiming a llama.cpp fact here.

**Equivalent here:** none — corpus management is entirely harness-side

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Moderate. Nothing in llama.cpp cares, but the failure it prevents — a doc edit silently moving a benchmark number — is precisely the instrument-fault shape this repo catalogues. Whether qwen38-tuning/bench already freezes its corpus I cannot tell from the map.

### A negative result recorded in the harness that failed to reproduce it
**Where (theirs):** `bench/conc_ladder.py:16-18`

**What it does.** The tool's docstring records the bug report that motivated it and the fact that it did not reproduce, with the numbers and the exact configuration.

**Mechanism.** Docstring: "Written for issue #16 (a reported per-stream dip at exactly N=2, which did not reproduce here: 105 / 98 / 86 tok/s at N=1/2/3, SPEC=mtp k=3 PREFIX_CACHE=1)."

**Why they needed it.** Keeps a non-finding attached to the instrument that would find it, so the next report of the same symptom starts from the prior measurement rather than from zero.

**Their numbers.** 105 / 98 / 86 tok/s at N=1/2/3, SPEC=mtp k=3, PREFIX_CACHE=1 — no dip at N=2.

**llama.cpp — not applicable.** Nothing to judge on the llama.cpp side. This is a documentation habit the repo already encodes; the delta is placement, not existence.

**Equivalent here:** none — this repo already has docs/tested/README.md and CORRECTIONS.md for it

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Low incremental value — the practice already exists here in a stronger form (a register plus a corrections ledger plus an audit script). The one refinement worth borrowing is attaching the non-finding to the instrument that would find it, not only to the register, so the next person who sees the symptom starts from the prior measurement.

### The ratio is only drawn once both lanes have finished
**Where (theirs):** `bench/demo_render.py:368-376` · `bench/demo_render.py:174-181`

**What it does.** Prevents the headline speedup number from being inflated by comparing a finished lane's final average against a still-running lane's instantaneous dip; and stops a finished lane's elapsed clock.

**Mechanism.** The "Nx faster" pill is drawn only `if t_ms >= span` where `span = max(pa['tokens'][-1][0], pb['tokens'][-1][0])` (demo_render.py:357, 371). In `panel()`, `end_ms = prompt['tokens'][-1][0]; done = t_ms >= end_ms; t_ms = min(t_ms, end_ms)` so a finished lane's elapsed display freezes.

**Why they needed it.** "Only once BOTH lanes have finished: a live ratio would divide the finished lane's final average by the running lane's instantaneous dip and read high (11.4x where the honest answer is 8.9x)." (demo_render.py:368-370). And: "A lane that has stopped generating must stop its clock too, or it keeps counting while the other lane finishes and the elapsed time is a lie." (demo_render.py:175-177)

**Their numbers.** A live ratio read 11.4x where the honest answer is 8.9x.

**llama.cpp — not applicable.** No llama.cpp mechanism involved. Judged as a reporting rule, and one this repo's CORRECTIONS.md culture already implies.

**Equivalent here:** none — presentation discipline, no llama.cpp surface

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Low here, but the general shape is worth keeping: a ratio computed while one side is still accumulating reads high, and the vLLM record shows it reading 11.4x where the honest answer was 8.9x. This repo publishes ratios (+34.7 %, +48.5 %) and the analogous error would be computing them over unequal windows.

### Fixed global scale for every bar in the demo, and a guard against the caption drifting into false claims
**Where (theirs):** `bench/demo_render.py:146-156` · `bench/demo_render.py:333` · `bench/demo_render.py:218-227` · `bench/demo_render.py:260-263`

**What it does.** Draws every throughput bar on one shared 0..vmax scale so bars in different columns show the true ratio without arithmetic; and carries an in-code warning about two captions that have twice regressed into untrue claims.

**Mechanism.** `GLOBAL_MAX = max over all prompts and lanes of decode_tok_s * 1.08` (demo_render.py:333), passed to every `meter()` call. The caption guard: "Careful with this subtitle and with `kicker` in final_card(): both have twice drifted back into claims that are not true of this picture. It is NOT 'same weights' -- the middle lane runs the fast variant's int4-GPTQ lm_head and the left one does not, which is why their answers diverge -- and it is not 'speculative decoding only', because the middle lane is the whole serving stack and the right lane is a different engine altogether." (demo_render.py:219-227)

**Why they needed it.** "A horizontal bar on a fixed 0..vmax scale, so two bars drawn at the same scale in different columns still show the true ratio between them -- no arithmetic needed." (demo_render.py:147-149). The subtitle actually shipped is "one RTX 3090 @ 250 W · same prompts · not run concurrently" (demo_render.py:226-227).

**Their numbers.** Demo is measured at 250 W power limit; verify.sh:36 checks and prints the power limit because "README numbers are at 250 W".

**llama.cpp — not applicable.** No llama.cpp surface. The transferable part is where the guard lives, not what it says.

**Equivalent here:** none — presentation discipline

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Low as drawn, moderate as a habit. The caption-guard half — an in-code note naming the two specific untrue claims a caption has twice regressed into — is the same instinct as a rule in scripts/audit-stale-claims.py, and putting the guard next to the text rather than only in a ledger is a small improvement worth copying.

### --language-model-only to drop the vision tower
**Where (theirs):** `batch/start_qwen.sh:15` · `batch/start_qwen.sh:113` · `single-user/start_qwen.sh:296`

**What it does.** Skips loading the multimodal vision tower entirely on a text-only workload.

**Mechanism.** `--language-model-only` flag on both `vllm serve` invocations.

**Why they needed it.** "--language-model-only skips the vision tower entirely (~2.7 GB saved)" (batch/start_qwen.sh:15)

**Their numbers.** ~2.7 GB VRAM saved.

**llama.cpp — not applicable.** The multimodal interactions the map documents (cache_reuse and context shift force-disabled, the GGML_ABORT belt-and-braces) are all conditional on a loaded mmproj, which this profile does not have.

**Equivalent here:** none needed — no mmproj is loaded unless --mmproj is passed

**Evidence (llama.cpp):** `tools/server/server-context.cpp:1165-1174` · `tools/server/server-context.cpp:3157`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. A text GGUF carries no vision tower, and llama-server only enters the multimodal path when a projector file is explicitly supplied. There is nothing to switch off and no VRAM to recover.

### FlashInfer sampler disabled for a build reason, not a performance one
**Where (theirs):** `batch/start_qwen.sh:95-97` · `single-user/start_qwen.sh:283`

**What it does.** Turns off the FlashInfer sampling path while leaving its attention kernels in use.

**Mechanism.** `export VLLM_USE_FLASHINFER_SAMPLER=0`.

**Why they needed it.** "flashinfer's sampling.cu does not build with older system nvcc (12.0); the attention kernels JIT fine. Remove this if you have a recent CUDA toolkit." (batch/start_qwen.sh:95-96)

**llama.cpp — not applicable.** No component to disable. Judged as not-applicable rather than absent, since there is nothing missing that a llama.cpp user would want.

**Equivalent here:** none — there is no optional JIT-compiled sampler component

**Evidence (llama.cpp):** `ggml/CMakeLists.txt:208` · `common/sampling.cpp:219-224` · `ggml/src/ggml-cuda/common.cuh:110-112`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** Nil directly. The transferable shape — a capability that is present or absent because of how the binary was built, not because of a flag — is real here and is covered by the build-integrity-gate verdict. llama.cpp's instances are GGML_CUDA_FA_ALL_QUANTS (KV type coverage), LLAMA_LLGUIDANCE (a %llguidance grammar GGML_ABORTs rather than erroring), and the CUB availability that decides whether device argsort works above 1024 elements.

### Custom op opt-ins for rms_norm and silu_and_mul
**Where (theirs):** `batch/start_qwen.sh:118` · `single-user/start_qwen.sh:302`

**What it does.** Forces the hand-written CUDA implementations of RMSNorm and SiLU-and-multiply instead of letting torch.compile generate them, on both launchers.

**Mechanism.** `--compilation-config "{...,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}"`, alongside `max_cudagraph_capture_size` (64 in batch, `$CG` in single-user).

**Why they needed it.** Not stated in the launcher comments — this is one of the few flags in either script carrying no explanation.

**Their numbers.** max_cudagraph_capture_size: 64 in batch mode; 32 for MTP single-user; MAX_SEQS*(k+1) for DFlash2.

**llama.cpp — not applicable.** The vLLM flag exists to override a code generator llama.cpp does not have. The one thing worth noting is that this is also the only flag in either vLLM launcher carrying no explanation, so there is nothing to transfer even as a rationale.

**Equivalent here:** none needed — ggml always uses its own hand-written CUDA kernels; there is no compiler-generated alternative

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:4249-4274`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. There is no torch.compile equivalent and therefore no opt-in. The adjacent knob that does exist, max_cudagraph_capture_size, has no analogue either: llama.cpp captures whatever shape the step produces and re-arms on change.

### Docker base image chosen as CUDA base+nvcc, not devel, with a JIT cache volume
**Where (theirs):** `Dockerfile:5-9` · `Dockerfile:32-36` · `docker-compose.yml:49-54`

**What it does.** Builds on `nvidia/cuda:13.0.1-base-ubuntu24.04` plus a hand-picked set of nvcc/cudart/curand dev packages instead of the devel image, and persists every JIT cache in a named volume so the first-run compilation happens exactly once.

**Mechanism.** apt installs `cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcurand-dev-13-0 build-essential patch`. `ENV HOME=/cache` with `VOLUME ["/cache", "/app/models"]` so torch.compile (~/.cache/vllm), Triton (~/.triton), FlashInfer JIT (~/.cache/flashinfer) and the HF hub cache all land in the volume. The healthcheck sets `start_period: 900s` for "first start: torch.compile + CUDA graphs + FlashInfer JIT".

**Why they needed it.** "vLLM's wheels bring their own CUDA libraries, but FlashInfer JIT-compiles its fp8-KV attention kernel with nvcc on first use (batch mode, CTX=long) and Triton needs a C compiler for its launchers. The compiled kernels and the torch.compile cache live in the /cache volume, so that only happens once." (Dockerfile:5-9)

**Their numbers.** Healthcheck start_period 900 s; stop_grace_period 30 s; healthcheck interval 30 s / timeout 5 s / 3 retries.

**llama.cpp — not applicable.** The entire premise (first-run JIT that should happen once) does not exist for a ggml-CUDA build. The 900-second healthcheck start_period has no counterpart either.

**Equivalent here:** none — kernels are AOT-compiled for compute_89 at build time; there is no runtime JIT to cache

**Evidence (llama.cpp):** `ggml/src/ggml-cuda/ggml-cuda.cu:4249-4274` · `ggml/src/ggml-cuda/common.cuh:1435-1444`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** Zero. This build compiles only compute_89 ahead of time, so there is no nvcc at runtime, no Triton launcher compilation and no FlashInfer JIT. The only per-process warm-up cost is CUDA graph capture, which is in-memory and cannot be persisted to a volume — and which is evicted after 10 s of idle anyway.
