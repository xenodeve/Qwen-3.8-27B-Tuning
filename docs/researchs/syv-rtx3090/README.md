# syv-ai/qwen38-27b-rtx3090 — the whole repo, so nobody reads it again

**External material. Nothing here is evidence about this machine** — see
[`../README.md`](../README.md). The one exception is
[`01-llamacpp-capability-map.md`](01-llamacpp-capability-map.md), which was
read from our own `llama.cpp` tree and is therefore evidence about llama.cpp,
though not about what any of it is worth here. **No number in this folder was
measured on the 4070 SUPER.**

## What was scanned

`syv-ai/qwen38-27b-rtx3090` at `--depth 1`, 2026-08-22: a patched vLLM 0.27.1
stack serving Qwen3.8-27B on one RTX 3090 24 GB. **16,370 lines, 78 files.**

| | |
|---|---:|
| techniques extracted | **434** |
| llama.cpp capabilities mapped | **175** |
| agents | 24 completed, 1 failed |
| subagent tokens | 3,058,967 |
| wall clock | 26 min |

### The verdict spread

| status | count | means |
|---|---:|---|
| EXISTS, NEVER SET | 48 | **a flag we already have and have never set** |
| absent, has a seam | 49 | not there, but the seam is named |
| partial | 50 | something related exists, not the same thing |
| already have it | 117 | we do this, sometimes under another name |
| impossible here | 18 | the architecture forecloses it |
| not applicable | 151 | meaningless outside vLLM/Marlin/Triton |
| **no verdict** | 1 | read but never matched |

**The 152 not-applicable are the honest majority.** Most of this repo is
Triton kernels, Marlin tiling and vLLM scheduler internals, and it does not
travel. The value is concentrated in the first two rows.

---

## Hand-verified, by me, against `C:\AI\llama.cpp`

The 434 verdicts came from agents. **An agent's report is a hypothesis until
checked**, so the claims the rest of this folder leans on hardest were read
back from source by hand. These six are confirmed:

| claim | source | confirmed |
|---|---|---|
| `--spec-draft-n-max` defaults to **3** | `common/common.h:325` — `int32_t n_max = 3;` | yes |
| `--spec-draft-p-min` defaults to **0.0** | `common/common.h:329` — `float p_min = 0.0f;` | yes |
| main-path `backend_sampling` defaults to **false** | `common/common.h:295` | yes |
| **draft-path** `backend_sampling` defaults to **true** | `common/common.h:331` — a *different* field, and the grammar disable at `sampling.cpp:421` does not reach it | yes |
| `-bs` / `--backend-sampling` exists and is off | `common/arg.cpp:2296` — "enable backend sampling (experimental) (default: disabled)" | yes |
| `--fit-target` defaults to **1024 MiB** | `common/common.h:473` — `fit_params_target(llama_max_devices(), 1024*1024*1024)` | yes |

### And the check found an error in our own repo

`scripts/worker-iq2xxs-deep.ps1` stated:

> "`--fit-target 768` left at the default deliberately"

**768 is not the default. 1024 is.** We had already lowered it and the header
said the opposite, so anyone reading it would have concluded there was a full
GiB still to reclaim when most of it was already spent. Corrected 2026-08-22.

**That is the argument for this whole folder in one line:** the scan's value is
not that it found somebody else's clever trick, it is that mapping our own tool
exhaustively caught a false statement we had been carrying.

### What I did NOT verify

The other **428 verdicts**, and every `value_here` estimate in this folder. They
are agent output, reasoned from source, and unmeasured. Treat a row here the way
this project treats any external claim: as a thing to test, not a thing to cite.

---

## Flags we already have and have never set — 48

Ordered by effort. **None of these has been measured here**; the right-hand
column is an agent's estimate, not a result.

| technique | flag / knob | effort | estimated worth here |
|---|---|---|---|
| Exact-token-id replay with a hard refusal to guess | `return_tokens: true` (server request field) → `tokens` array in the completion result | one-flag | High, and cheaper here than in vLLM: llama.cpp hands back real ids, so there is no "token_id:1234" string-prefix hack to break. The refusal discipline still applies — if `tokens` comes back  |
| slots/step as an interpolation that reveals long-block scheduling fraction | common_speculative_print_stats — the per-implementation SPC_TRC statistics line, visible at LOG_TRC (-lv 4 / -v) | one-flag | High and better than the vLLM technique. llama.cpp does not need an interpolation: with --spec-type draft-dflash,ngram-mod it records which impl produced each draft in impl_last[seq_id] and  |
| Batch-mode rows measured with --ignore-eos, cohort rows without | --ignore-eos (CLI) / `"ignore_eos": true` (per request), paired with n_predict | one-flag | Moderate here, high if throughput rows are ever added. At -np 1 the effect is smaller than in a 64-way batch, but any fixed-length decode row still measures stopping behaviour unless ignore_ |
| Shared-versus-distinct prompt switch to model prefix-cache-friendly clients | per-request `"cache_prompt": false` (or --no-cache-prompt) to defeat in-slot prefix reuse; -cram 0 to disable the RAM prompt cache | one-flag | High for a coding agent, which is the workload here. Turn 2+ on the same file is almost entirely a prefix hit; a benchmark that never states which case it measured is uninterpretable. Both s |
| Per-token arrival-time capture with lanes recorded separately and replayed together | `"timings_per_token": true` puts the full timings block on every stream chunk; `return_tokens` gives ids per chunk | one-flag | Low for this repo's purposes — it is a presentation technique. If a demo is ever wanted, llama.cpp gives more than vLLM did: timings on every chunk rather than only at the end, so arrival ti |
| Two different gpu-memory-utilization values, and the reason they differ | -fitt / --fit-target MiB — the per-device margin --fit leaves free (default 1024 MiB) | one-flag | High and immediate. On a 12 GB card the default silently forfeits a full GiB before --fit even starts assigning layers, and lowering it is the single direct lever for more context or more of |
| max-num-batched-tokens 2048 chosen against the KV pool, not against prefill speed | -ub / --ubatch-size (default 512) and -b / --batch-size (default 2048) | one-flag | High, and the same second-order mechanism applies. n_ubatch is the single knob that sizes the worst-case compute buffer: the reserve pass runs prompt processing at n_tokens = min(n_ctx, n_ub |
| Draft count reduced at long context because k=4 crashes on FlashInfer | --spec-draft-n-max (default 3) and --spec-draft-n-min (default 0) | one-flag | Very high — likely the biggest single unclaimed win on this list. --spec-draft-n-max defaults to 3, and the DFlash block-size clamp only ever LOWERS it: with a stock 16-wide sidecar the larg |
| KV pool pinned by absolute bytes rather than by gpu-memory-utilization | a numeric -c / --ctx-size plus an explicit -ngl N, which together take --fit out of the loop | one-flag | Highest value on this list for measurement integrity. This repo's stated reason it cannot compare raw decode across boots is that free VRAM at boot moves 9,326–10,732 MiB and --fit follows i |
| k=4 is the knee, but only on FlashAttention — FlashInfer dies at k=4 | --spec-draft-n-max N (env LLAMA_ARG_SPEC_DRAFT_N_MAX), default 3 | one-flag | This is the single cheapest untried lever we have. The default is 3; a stock 16-wide DFlash sidecar allows up to 15 (16 for anchor-sampling DSpark), and the DFlash speculator drafts exactly  |
| Pin the KV pool in bytes (`KV_MEM` / `--kv-cache-memory`) instead of by utilization | pass a numeric -c N — fit modifies n_ctx if and only if it is 0, and prints `context size set by user to %u -> no change` | one-flag | Directly aimed at this project's stated worst measurement problem. CLAUDE.md records free VRAM at boot moving 9,326–10,732 MiB with --fit following it, and declares effects below 13.6 % to b |
| `--max-num-batched-tokens 2048` — bigger prefill chunks make things worse | -ub / --ubatch-size (default 512) and -b / --batch-size (default 2048) | one-flag | -ub is the single knob that sizes the worst-case compute buffer: the reserve builds the prompt-processing graph at n_tokens = min(n_ctx, n_ubatch), so lowering -ub shrinks the compute buffer |
| FlashInfer radix top-k with a latching fallback to torch.topk | -bs / --backend-sampling (GPU-side sampling for the MAIN path) | one-flag | unknown, but this is a one-flag experiment they have never run. -bs moves the whole default sampler chain onto the GPU and removes a 151k-entry logits readback per accepted token. Refuses to |
| NMAX=12 chosen against 32 so recency beats length | --spec-ngram-mod-n-match (default 24), and --spec-ngram-map-k4v-size-n / -size-m / -min-hits | one-flag | unknown, and a cheap one-flag A/B. ngram-mod's window is 24 tokens; the vLLM measurement says a shorter, more recent match predicts better on quote-and-explain work (3.21 vs 2.69 tokens/step |
| NSTRONG=NMIN and AGREE=0: take any qualifying match in the head | --spec-ngram-mod-n-min (default 48, range 0..1024) | one-flag | unknown, and this is the other one-flag experiment worth running. I read draft_one: it walks up to n_max successors and the moment the hash returns EMPTY it either truncates (if i >= n_min)  |
| Lookup hit counter | common_speculative_print_stats (per-impl counters, LOG_TRC only) + the Prometheus per-position histogram | one-flag | Strictly better than what vLLM built, and switched off. llama.cpp keeps per-impl n_gen_drafts, n_acc_drafts, n_gen_tokens, n_acc_tokens, a per-POSITION acceptance array, and t_begin/t_draft/ |
| Weight-aware pool budget: share of (util*total − weights) instead of a fixed slice of the card | --fit (on) + -fitt / --fit-target MiB (never set — default 1024 MiB per device) | one-flag | -fitt is at its default 1024 MiB, so ~1 GiB of a 12 GB card is deliberately forfeited. Lowering it (e.g. -fitt 384) hands ~640 MiB back to --fit, which spends it on n_ctx and layer placement |
| V2-runner fix 1 — drafter's sliding-window layers forced back to bf16 via a copied cache_config | -ctkd / --spec-draft-type-k and -ctvd / --spec-draft-type-v (default f16 for both, never set here) | one-flag | Live now that draft-dflash is in use. llama.cpp already applies KVarN's fix by default — -ctk q4_0 does NOT propagate to the draft, so the draft cache is f16. The unexplored lever is the inv |
| Complete tunable surface: six environment variables, all with documented defaults and consequences | LLAMA_ATTN_ROT_DISABLE, GGML_CUDA_GRAPH_OPT, LLAMA_GRAPH_REUSE_DISABLE, GGML_CUDA_DISABLE_GRAPHS, GGML_OP_OFFLOAD_MIN_BATCH, LLAMA | one-flag | Three of these are unset here and worth a round each. GGML_CUDA_GRAPH_OPT=1 turns on multi-stream QKV concurrency; it requires CUDA graphs plus exactly one CUDA device, which this box satisf |
| Applying the request's top-k/top-p to the DFlash2 selector's 16-candidate proposal | --spec-draft-p-min (alias --draft-p-min) — live on the DFlash2 selector path, default 0.00; and --spec-draft-n-min, default 0 | one-flag | The single most actionable item in this slice for us. We run --spec-type draft-dflash,ngram-mod and have never set --spec-draft-p-min; it defaults to 0.00, i.e. the confidence early-stop is  |
| Materialize path retained as an A/B control | the env kill switches and the -fa / -ctk pairings, which are the A/B levers this build ships | one-flag | Four one-variable controls we have and have never used, each of which isolates one of the mechanisms this slice raised: LLAMA_ATTN_ROT_DISABLE isolates the Hadamard (technique 35), LLAMA_GRA |
| Explicit CUDA-graph memory reservation for the V2 model runner | -fitt / --fit-target MiB — the per-device margin --fit leaves free before sizing anything | one-flag | Concrete and bidirectional on 12 GB. Downward: -fitt 384 hands roughly 640 MiB back to the KV cache, which at q4_0 (0.5625 B/element) is a meaningful context increase — the exact token count |
| Pin the pool with --kv-cache-memory rather than tune gpu_memory_utilization | an explicit numeric -c N (which makes --fit leave context alone entirely), optionally with an explicit -ngl N to stop its placemen | one-flag | This is a measurement-validity win rather than a tok/s win, and by this project's own standards that is worth more. CLAUDE.md records that free VRAM at boot moves 9,326-10,732 MiB and --fit  |
| Prefill matrix as a separate opt-in sweep | tools/llama-bench with -p (n_prompt), -n (n_gen), -pg, and -d/--n-depth | config | Moderate. llama-bench already does the length × depth grid and reports pp/tg separately, and -d lets you measure at a KV depth rather than from empty — which is the number that actually matt |
| Quality battery: three-domain perplexity plus GSM8K, run after every kernel/quant change | tools/perplexity (llama-perplexity) — offline PPL over a corpus; no server-side prompt_logprobs equivalent | config | Very high for this profile specifically. UD-IQ2_XXS is a 2-bit quant with -ctk q4_0 -ctv q4_0 on top, and the Hadamard rotation that mitigates quantised-KV damage is applied silently and onl |
| Attention kernel correctness checked against an fp32 reference AND against FA2 | tests/test-backend-ops — test_flash_attn_ext compares the CUDA op against the CPU reference at parameterised shapes, including qua | config | High if this repo ever patches the attention path, moderate otherwise. The existing cases already sweep hsk/hsv/nh/nr23/kv/nb and K/V type pairs, including asymmetric ones the CUDA build can |
| Offline context scan of the verify attention with four arms and no server | tests/test-backend-ops MODE_PERF over test_flash_attn_ext at chosen kv/nb/type_KV | config | High and cheap. This answers, without a server and without touching the KV budget, the one question the map flags as the biggest hidden cost in the attention area: what quantised K/V actuall |
| The pool cost of a longer verify block scales with the block, not with the slot count | the n_rs_seq / batch-widening interaction: cparams.n_rs_seq = draft.n_max for the model-based speculators, and DFlash forces n_bat | config | High, as the cost side of the --spec-draft-n-max sweep. Raising n-max from 3 toward 15 for draft-dflash costs VRAM in two places, both documented and neither obvious. (1) n_rs_seq = draft.n_ |
| Requantize both untied embedding matrices to int8 g128 | llama-quantize --token-embedding-type / --output-tensor-type (offline; requires --allow-requantize from an already-quantized sourc | config | token_embd costs zero VRAM here — LLM_TENSOR_TOKEN_EMBD is LAYER_INPUT and dev_input is hard-wired to CPU, so the vLLM saving does not exist on this side. output.weight does go to GPU. The k |
| Requantize the MTP draft module (int8, then GPTQ-int4) | llama-quantize on the draft/sidecar GGUF; -ngld / -otd / -cmoed control its placement | config | Every MiB the dflash sidecar occupies is a MiB taken off the target, one for one: --fit never sizes the draft (it is loaded with llama_model_load_from_file directly and -ngld 'auto' resolves |
| GPTQ-quantize the DFlash2 drafter to W4A16 with a calibration set hooked from inside vLLM | llama-quantize on the sidecar (--tensor-type, --pure, --imatrix all available offline); no in-server calibration hook | config | Same 1:1 VRAM economics as technique 8 — the sidecar is loaded outside --fit and the server shrinks the target to pay for it. The calibration half does NOT transfer: llama.cpp's imatrix is p |
| lm_head GPTQ calibrated on 300k captured final hidden states, with a held-out KL split | llama-imatrix --process-output (default OFF) plus -f <calibration corpus>, then llama-quantize --imatrix | config | Only reachable if we re-quantise the target ourselves, which needs a BF16/F16 source we do not have locally (6.76 GiB IQ2_XXS is what is on disk; --allow-requantize from it would compound lo |
| KL(bf16 head ‖ quantised head) on held-out hidden states as the acceptance metric for RTN vs GPTQ | llama-perplexity --kl-divergence-base FNAME (alias --save-all-logits) to dump reference logits, then --kl-divergence against the q | config | A ready-made, never-used instrument for exactly the question this repo keeps asking: does a quantisation change move the distribution or only the weights? It measures the target, not the dra |
| Self-distillation prompt mix with per-source quotas and a per-source thinking probability | llama-imatrix -f <calibration text> — the calibration corpus is a plain file argument | config | Zero at serving time; relevant only if we build a GGUF. The load-bearing idea does transfer though: whatever we pass to -f decides what every quantised tensor is tuned for, and a coding-agen |
| Asymmetric per-row RTN with zero-point = row minimum | -ctk q4_1 / -ctv q4_1 (and q5_1) — asymmetric scale+min block quants, accepted by the parser | config | Reachable only by rebuilding with -DGGML_CUDA_FA_ALL_QUANTS=ON. Note the direction of the trade: q4_1 stores a scale *and* a min per 32-element block, so it is 5.0 bits/element against q4_0' |
| The draft vocabulary must be counted over the model's own outputs | -lcs / --lookup-cache-static FNAME with --spec-type ngram-cache, cache built by examples/lookup/lookup-create.cpp | config | Unknown, and gated by a real hazard. The flag exists and is accepted by llama-server, and llama-lookup-create builds a static n-gram cache from any corpus — priming it from this agent's own  |
| Prefix-resync teacher forcing (acceptance without trajectory drift) | POST /completion with `prompt` as a raw int array + `return_tokens: true`, scored off llamacpp:spec_decode_* counters (--metrics) | small-patch | High. This project's own rule is that effects below 13.6 % are noise across boots; teacher forcing removes trajectory divergence entirely, so a paired ngram-mod vs draft-dflash comparison st |
| Dirty-chunk detection: off-trajectory chunks are excluded and counted, and firstbad is located | compare the `tokens` array from return_tokens against the frozen target elementwise | small-patch | High and cheap once return_tokens is on. Note llama.cpp speculation is exact by construction — the greedy verifier accepts draft[i] only when it equals the target's own sample (sampling.cpp: |
| `last_num_emitted`: what the previous step actually produced | slot.stats.n_draft_tokens / n_draft_accepted / n_draft_verif_steps, and common_speculative_accept's accepted.size()-1 | small-patch | The signal already exists per slot per step and today feeds only the log line and Prometheus. Wiring it into dp.n_max is the small patch of technique 43. Value of reading it as-is: the `draf |
| MTP module and lm_head requantized to int4 with GPTQ calibrated on hidden states | llama-quantize --output-tensor-type / --token-embedding-type / --tensor-type <regex>=<type> | large-patch | Uncertain but with one immediately actionable sub-finding. The per-tensor seam exists and is precise (regex, first match wins, and setting it suppresses the k-quant mixture for that tensor), |
| Vocab-truncated draft head (40,960-row lm_head slice) | the `d2t` GGUF tensor (LLM_TENSOR_D2T) — a draft-row -> target-token-id map that lets a DFlash/EAGLE3 sidecar ship its own reduced | large-patch | Our sidecar does not use it. Qwen3.8-27B-DFlash2-Q4_K_M.gguf is 1.92 B params / 1.05 GiB with no d2t line in the load log and no own output head — it borrows the target's 248,320-row IQ2_XXS |
| In-place int8 g128 requantisation of lm_head with a hard error gate | llama-quantize --output-tensor-type <ggml_type> | large-patch | The head is the one vocab-sized tensor that DOES live in VRAM here (dev_output, llama-model.cpp:1378) and is read on every decode step and every speculative verify step. Setting its type is  |
| embed_tokens int8 with bf16 (not fp16) scales, and a distinct backup name | llama-quantize --token-embedding-type <ggml_type> | large-patch | Near zero on VRAM, which is the honest answer and the useful one. token_embd is LLM_TENSOR_LAYER_INPUT and dev_input is hard-wired to CPU with the in-source comment that offloading the input |
| Round-to-nearest int8/int4 requantisation of the whole MTP draft module | run llama-quantize on the DFlash2 sidecar GGUF itself | large-patch | Our sidecar is already Q4_K_M: 1.92 B params, 1.05 GiB, 4.71 BPW (from the load log). Dropping to Q4_K_S or IQ4_XS would recover roughly 0.1-0.2 GiB. That matters more than it looks, because |
| W4A16 export of DFlash2 with a hand-written compressed-tensors config in vLLM's module-prefix namespace | llama-quantize --tensor-type <regex>=<type> (plus --pure) applied to the DFlash2 sidecar GGUF | large-patch | The nearest thing in this whole slice to an actionable VRAM lever, and still blocked. Our sidecar is Q4_K_M / 1.05 GiB; its vocab-sized tensors are selector_predecessor and selector_successo |
| `set_draft_tokens(num_draft=...)`: truncate the proposal handed to the scheduler | the dp.n_max truncation in common_speculative_draft | n/a | The plumbing is complete. It is driven only by the context clamp today, never by a policy. |
| Sliding window carried into the impl so the decode kernel can bound its block loop | llama_kv_cache_iswa / llama_memory_hybrid_iswa with size_swa = n_swa + n_ubatch, and --swa-full to opt out | n/a | zero for Qwen3.8. Listed as exists-but-unused specifically so nobody spends a round tuning --swa-full expecting a KV saving — it will print 'swa_full is not supported by this model, it will  |
| Sliding-window block-loop truncation | llama_kv_cache_iswa / llama_memory_hybrid_iswa with size_swa = n_swa + n_ubatch, and --swa-full to defeat it | n/a | None for our model. llama.cpp's version of this saving is structural — an SWA layer gets a physically smaller cache rather than a truncated loop — but the map records that qwen35 declares no |

---

## Absent, but the seam is named — 49

| technique | where it would go | effort |
|---|---|---|
| Wait for the GPU to actually be free before starting (`ExecStartPre` gate) | none in llama.cpp — the seam is our own launcher under qwen38-tuning/scripts/, because --fit measures free device memory at the instant it runs | config |
| Six enumerated things the acceptance harness cannot control for | none — the seam is the harness docstring in qwen38-tuning/bench/ | small-patch |
| Frozen target files keyed by prompt hash, tagged with their capturer | none — the seam is qwen38-tuning/bench/ alongside the existing frozen artefacts | small-patch |
| Run the whole suite twice after a restart and keep the second | none — the seam is the bench runner in qwen38-tuning/bench/ | small-patch |
| Long corpus with a byte-identical frozen head and non-repeating filler | none — the seam is the corpus builder in qwen38-tuning/bench/ | small-patch |
| Explicit warning when the requested context exceeds what the corpus can supply | none harness-side; server-side the analogous failure is loud already | small-patch |
| Realistic-prompt cohort dataset instead of random tokens | none — the seam is the prompt set in qwen38-tuning/bench/ | small-patch |
| Best-of-reps at each concurrency rung, keyed on per-stream rate | none — the seam is the bench runner | small-patch |
| Named failure signatures for the soak, ordered by how much each means | none — the seam is a signature list in docs/reports/ or the soak harness | small-patch |
| Residue-class sweep (mod 128) that turns a false threshold into a real periodicity | none — the seam is a fine-grained prompt-length sweep in qwen38-tuning/bench/ | small-patch |
| Judging correctness on the output, never on an absolute tok/step threshold | none — the seam is the pass/fail predicate in the bench harness | small-patch |
| GPU lookup kernels tested against a plain-Python reference on adversarial sequences | none in llama.cpp's own tests; the seam is qwen38-tuning/bench/tests, where TDD is already mandatory | small-patch |
| Extrapolating a per-layer microbenchmark to a whole step, with the constant terms named | none — the seam is the analysis on top of test-backend-ops MODE_PERF output | small-patch |
| API feature smoke suite: the request-level features a new model runner silently breaks | none shipped; the seam is qwen38-tuning/bench/tests against a live llama-server | small-patch |
| verify.sh as an install-integrity gate with a three-way patch check | none shipped; the seam is a preflight in qwen38-tuning/scripts/ asserting build identity before the server is allowed to start | small-patch |
| Live-server verification reads the pool and backend the server actually chose out of its own log | none shipped; the seam is a log scraper over llama-server's stderr plus GET /props and GET /slots | small-patch |
| Recurrent (GDN) state cache in fp16 instead of the config's fp32 | none — type_r/type_s are GGML_TYPE_F32 literals; the seam is the three call sites in src/llama-model.cpp plus one new arg in common/arg.cpp | small-patch |
| The prefill tax of the int8 long-context mode stated, with the workload it is and is not for | none measured; the seam is a paired -ctk f16 vs -ctk q4_0 prefill benchmark at fixed -c and -ngl | small-patch |
| Patches applied and verified at image build time | none shipped; the seam is the build script for build-dflash2 plus a post-build assertion over CMakeCache.txt and compile_commands.json | small-patch |
| fp16 Gated-DeltaNet recurrent state (`--mamba-ssm-cache-dtype float16`) — the real concurrency bound | none — recurrent_type_r / recurrent_type_s are GGML_TYPE_F32 literals at the call site | small-patch |
| Let a match overlap the suffix it matched, so a repeating pattern is proposed from its own period | none — ngram-mod updates its table in chunks and lags the generated text by up to 32 tokens | small-patch |
| NaN guards before the rejection sampler's two argmax reductions | none; the seam is common_sampler_sample_and_accept_n and llama_sampler_init_temp's greedy rewrite | small-patch |
| Point-mass draft-logit rewrite that preserves rejection-sampling exactness | dp.dists left empty => the greedy accept rule is used even at temperature 1.0 | small-patch |
| Two qualifying steps in a row required to enter the long block | none; would live in the same dp.n_max computation as technique 43 | small-patch |
| STICKY hysteresis: hold the long block for 3 steps after the flag drops | none; same seam as 43 | small-patch |
| Lookup applied after the selector, inside the same captured graph | the hardcoded speculator priority list — ngram-* always outranks every model-based type | small-patch |
| V2-runner fix 7 — three-part NaN fix in the DFlash2 speculator, traced to KVarN quantization noise | none — the seam is the DFlash2 selector block in common/speculative.cpp | small-patch |
| Capturing every token's final hidden state by hooking GPUModelRunner._model_forward | ggml_backend_sched_set_eval_callback — the same hook tools/imatrix and examples/eval-callback use | small-patch |
| KVARN_RTN_QUANTILE — percentile clipping instead of min/max | none | small-patch |
| `VLLM_DRAFT_TEMP_SCALE` — draft sharpening, measured and rejected | one extra divisor at common/speculative.cpp:1245, where dp.temperature is already applied | small-patch |
| Per-layer activation-quantization error measurement to choose which layers get int8 | tools/imatrix (per-tensor activation statistics) feeding llama-quantize --tensor-type / --token-embedding-type / --output-tensor-type | large-patch |
| Fuse the lookup proposal with the drafter's rather than substituting for it (`_NSTRONG` / `_AGREE`) | nearest expressible approximation is the ngram-mod n_min gate (a trust threshold on match length, all-or-nothing); no cross-impl agreement test exists | large-patch |
| Asymmetric head/tail acceptance thresholds in the fuse kernel | none — chaining is fallback, not fusion | large-patch |
| Leading-agreement computation clamped by `valid` | none | large-patch |
| Candidate cache spans the whole verify block while the drafter fills only its head | none | large-patch |
| Route cached-multiquery (MTP verify) by context depth: fused kernel vs materialize+FlashAttention | ggml_cuda_get_best_fattn_kernel routes on QUERY-TOKEN COUNT (Q->ne[1] <= 2 for quantised KV) and never on context depth; the MMA_F16 branch is llama.c | large-patch |
| Whole MTP verify step as ONE captured CUDA graph via UNIFORM_BATCH + a persistent vq plan | CUDA graph capture exists and is on, but llm_graph_params::allow_reuse requires ubatch.n_tokens to be identical, so a variable-width verify step never | large-patch |
| Prefill first chunk attends RAW fp16 K/V — quantisation error never enters the prefill output | none — build_attn writes k_cur/v_cur to the cache and then reads back the QUANTISED view for the same step's attention | large-patch |
| sink_tokens: first 128 tokens per request never quantized, replacing layer-level boundary skipping | none; cparams.type_k / type_v are a single value for the whole context | large-patch |
| Deriving the draft vocabulary from the model's own outputs rather than from external text | none shipped; the carrier is `d2t` (src/models/dflash.cpp:99-105) and the id list would be counted offline over our own server output | large-patch |
| MTP Hessians dumped from the trainer's own validation forward passes | none shipped; the seam is ggml_backend_sched_set_eval_callback (used by tools/imatrix) plus a speculation-aware imatrix mode | large-patch |
| GPTQ requantisation of the MTP module from the trainer's Hessians, reading weights out of the pre-quant backup | llama-quantize --imatrix <draft.imatrix> on the sidecar GGUF (imatrix-weighted, not GPTQ) | large-patch |
| DFlash2 Hessians captured from the drafter's own inputs during real speculation, in eager mode | ggml_backend_sched_set_eval_callback wired into a speculation-aware imatrix run | large-patch |
| Padded-row early return for uniform-batch graph replay | none — llm_graph_params::allow_reuse requires exact ubatch.n_tokens equality, so a varying step size rebuilds the graph instead of padding to a bucket | large-patch |
| KVarN 4-bit key / 2-bit value KV cache, ported to 0.27.1 | none below q4_0; the seam is ggml_cuda_fattn_kv_type_supported plus the FATTN_VEC_CASES instantiation table | new-backend |
| KIVI-orientation asymmetric quantisation: K per-channel, V per-token | none. The nearest thing is v_trans (V stored transposed when FA is off), plus the asymmetric weight types q4_1/q5_1 that the -ctk/-ctv parser accepts  | new-backend |
| Sinkhorn-then-RTN two-level scale factorisation | none; nearest is the automatic Hadamard rotation of quantised K/V (attn_rot_k / attn_rot_v), which attacks the same outlier problem by a different rou | new-backend |
| Log-domain Sinkhorn with best-so-far imbalance tracking | none | new-backend |
| Sparse block→pool-slot table as the quantised/unquantised discriminator | none — one ggml_type per cache, chosen once at context construction | new-backend |

---

## The files

| file | slice | techniques |
|---|---|---:|
| [01](01-llamacpp-capability-map.md) | **llama.cpp capability map — ours, and evidence** | 175 caps |
| [02](02-bench-py-bench-sh-single-user-start-qwen-sh-ba.md) | bench/*.py + bench/*.sh, single-user/start_qwen.sh, batch/start_qwen.sh, verify.sh, Dockerfile, docker-compose | 75 |
| [03](03-docs-optimizations-md-docs-gotchas-md-docs-lon.md) | docs/optimizations.md, docs/gotchas.md, docs/long-context.md, docs/quality.md, docs/docker.md, README.md, sing | 64 |
| [04](04-dflash2-speculative-decoding-lookup-augmented-.md) | DFlash2 speculative decoding + lookup-augmented block drafting (LABD): patches/dflash2-backport.patch (995 lin | 60 |
| [05](05-kvarn-files-vllm-v1-attention-backends-kvarn-a.md) | kvarn/files/vllm/v1/attention/backends/kvarn_attn.py — the KVarN attention backend (backend class, metadata +  | 49 |
| [06](06-kvarn-configuration-vllm-wiring-kvarn-files-vl.md) | KVarN configuration + vLLM wiring: kvarn/files/vllm/model_executor/layers/quantization/kvarn/{config.py, sinkh | 49 |
| [07](07-drafter-py-drafter-readme-md-prepare-py-prepar.md) | drafter/*.py + drafter/README.md + prepare/*.py + prepare/README.md — how the syv-ref stack builds and quantis | 45 |
| [08](08-kvarn-triton-decode-kernels-tile-store-triton-.md) | kvarn Triton decode kernels + tile store: triton_kvarn_decode.py (1196 lines), kvarn_store.py (286), triton_kv | 38 |
| [09](09-attention-and-kv-under-speculation-patches-spe.md) | Attention and KV under speculation — patches/spec-decode-attn.patch, patches/spec-decode-int8-kv.patch, patche | 31 |
| [10](10-quantisation-sampling-and-knobs-patches-sample.md) | Quantisation, sampling and knobs — patches/sampler-small-topk-fast-softmax.patch, patches/marlin-int8-layer-se | 23 |

---

## What this scan did NOT do

- **The completeness critic never ran.** It was the last agent and it died on
  `You've hit your session limit`. So **no independent pass checked whether
  every file in the tree is represented here.** The nine readers each list
  the files they opened, at the top of their own file, and that is the only
  coverage claim this folder can make.
- **Nothing was measured.** Every `value_here` is an estimate produced by
  reading source, and this project's own history is a list of estimates that
  reversed on contact.
- **Their numbers are theirs.** An RTX 3090 24 GB running vLLM with a W4A16
  checkpoint says nothing directly about a 12 GB card running a 2-bit GGUF.
