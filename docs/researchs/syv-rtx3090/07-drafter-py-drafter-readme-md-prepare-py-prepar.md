# drafter/*.py + drafter/README.md + prepare/*.py + prepare/README.md — how the syv-ref stack builds and quantises its two drafters (the Qwen3.8 MTP head and the DFlash2 block drafter), the GPTQ implementation, the Hessian capture, the draft-vocabulary derivation, and the calibration data
**45 techniques.** 2127 source lines across 19 files.
Files read: `drafter/README.md` · `drafter/gptq_utils.py` · `drafter/gptq_lm_head.py` · `drafter/collect_prompts.py` · `drafter/gen_data.py` · `drafter/capture.py` · `drafter/capture_dflash2.py` · `drafter/quant_dflash2.py` · `drafter/requant_mtp_gptq.py` · `drafter/train_mtp.py` · `drafter/export_mtp.py` · `prepare/README.md` · `prepare/build_draft_vocab.py` · `prepare/quant_lm_head.py` · `prepare/quant_embed.py` · `prepare/quant_mtp.py` · `prepare/fetch_fast_variant.py` · `prepare/fetch_dflash2.py` · `prepare/draft_vocab_ids.json`
> **What the reader could not see:** Everything named in the slice existed and was read in full; nothing was missing or unreadable. Line counts matched the brief closely (train_mtp.py 484, capture_dflash2.py 190, export_mtp.py 140, collect_prompts.py 131 incl. gen_data concatenation offset, quant_dflash2.py 119, capture.py 119, gptq_lm_head.py 95, gen_data.py 95, gptq_utils.py 89, requant_mtp_gptq.py 64, fetch_dflash2.py 20, drafter/README.md 124, build_draft_vocab.py 122, quant_lm_head.py 91, quant_mtp.py 87, quant_embed.py 77, fetch_fast_variant.py 44, prepare/README.md 36). Gaps a reproducer would hit, all of which point outside this slice: - Three vLLM patches are load-bearing and none are in the slice: patches/qwen3_5-mtp-draft-vocab.patch (without it the draft head is inert), patches/qwen3_5-embed-quant.patch (quant_embed.py:99), and whatever adds `method: "dflash"` speculative support plus the `_dense_kv_rows` backport and `VLLM_DFLASH2_DRAFT_TOPK_TOPP`. I could not verify how mtp_draft_vocab_ids.pt is consumed at runtime. - The DFlash2 selector top-k/top-p truncation and the DFlash2 forward itself are documented only in drafter/README.md:102-104; no code for them exists in drafter/ or prepare/. - Several headline numbers are asserted in README.md with no script in the slice that produces them: the +12% aggregate throughput for int8 lm_head, the perplexity deltas (+1.5% / +0.6%), GSM8K 96.5%, the 118.8 tok/s figure, and all DFlash2 tok/s and ms-per-step figures. The benchmark harness (bench/, single-user/start_qwen.sh, verify.sh, docs/optimizations.md) is outside the slice, so none of these are reproducible from what I read. - prepare/draft_vocab_ids.json is the shipped OWN-OUTPUT list (I verified 40,960 unique ids, range 0..248,076) but the corpus that produced it is not committed — drafter/data/ (prompts.jsonl, gen.jsonl, hidden.npy, tokens.npy, seqs.json) does not exist in the repo, so the draft vocabulary, the lm_head Hessian and both sets of drafter Hessians all require re-running ~6 h of GPU work from collect_prompts.py onward. - drafter/runs/ (e/mtp_hessians.pt, dflash2/hessians.pt) is likewise absent; requant_mtp_gptq.py and quant_dflash2.py cannot be run against the repo as checked out. - A latent bug worth flagging: capture.py:109 builds its `bad` list with `state["written"].get(seqs.index(s) if False else i + batch.index(s), 0)` — the `if False else` makes it always `i + batch.index(s)`, and `batch.index(s)` is an O(n) identity-less lookup that returns the first equal dict. It is only used for a progress printout (the authoritative check is the `missing` recount at line 114), but the expression is dead-code residue.

---

## EXISTS, NEVER SET — 9

### Vocab-truncated draft head (40,960-row lm_head slice)
**Where (theirs):** `prepare/build_draft_vocab.py:1-24` · `prepare/build_draft_vocab.py:91-121` · `prepare/build_draft_vocab.py:31` · `drafter/README.md:14-22`

**What it does.** The MTP drafter would otherwise run the full 248k-row lm_head once per draft token, and at 4-6 drafts per step that head read (1.3 GB at int8) dominates the entire draft cost. Instead a 40,960-row slice of lm_head is stored as a separate tensor `mtp.draft_lm_head.*` and the drafter scores only that. Tokens outside the shortlist simply can never be drafted; that position is rejected and the target's own sample is used, so the sampled distribution stays exact. `prepare/draft_vocab_ids.json` holds exactly 40,960 unique ids ranging 0..248,076.

**Mechanism.** Reads `model.safetensors.index.json` to find the shard holding `lm_head.weight_packed`, opens it with safe_open, and does `wp.index_select(0, ids_t)` / `ws.index_select(0, ids_t)` on the packed int32 codes and the per-group scales (build_draft_vocab.py:95-101). A new `mtp.draft_lm_head.weight_shape = [len(ids), K]` is written. The three tensors are appended to `model_extra_tensors.safetensors` (backed up first as `.bak-draft`), the weight_map is extended with three entries pointing at that file, and the id tensor is saved separately as `mtp_draft_vocab_ids.pt` for the runtime to map draft-head columns back to real token ids.

**Why they needed it.** "The MTP drafter has to run the 248k-row lm_head once per draft token, and at 4-6 drafts per step that head read (1.3 GB int8) dominates the draft cost. Speculative decoding stays exact no matter what the drafter proposes, so the drafter can use a head restricted to the N most frequent tokens" (build_draft_vocab.py:3-8).

**Their numbers.** 40,960 rows default (build_draft_vocab.py:31). Draft head size printed as (packed numel*4 + scale numel*2)/1e6 MB. Requires patches/qwen3_5-mtp-draft-vocab.patch to be usable.

**llama.cpp — EXISTS, NEVER SET.** This is not an idea llama.cpp lacks: the exact mechanism is implemented and tested for two speculator families. dflash.cpp:99-105 reads the `d2t` tensor meta, sets n_vocab_draft from its ne[0], logs 'DFlash using d2t mapping (draft_vocab_size = %lld)', and dflash.cpp:204 then sizes `output.weight` as {n_embd, n_vocab_draft}. eagle3.cpp:319-330 shows the scatter back to target ids. Note d2t reduces the output head only — dflash_selector_prev/next stay at full n_vocab (dflash.cpp:139-141).

**Equivalent here:** the `d2t` GGUF tensor (LLM_TENSOR_D2T) — a draft-row -> target-token-id map that lets a DFlash/EAGLE3 sidecar ship its own reduced-vocab output head

**Evidence (llama.cpp):** `src/models/dflash.cpp:99-105` · `src/models/dflash.cpp:203-204` · `src/llama-arch.cpp:648` · `src/llama-arch.cpp:919` · `src/models/eagle3.cpp:47-54` · `src/models/eagle3.cpp:319-330`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Our sidecar does not use it. Qwen3.8-27B-DFlash2-Q4_K_M.gguf is 1.92 B params / 1.05 GiB with no d2t line in the load log and no own output head — it borrows the target's 248,320-row IQ2_XXS head via ctx_other. That head is ~248320x5120 = 1.27 G params at ~2.06 bpw = ~0.33 GB read per draft step (one matmul over the whole block, not per token). A 40,960-row d2t head would cut that read to ~0.05 GB/step at the cost of ~0.13 GB extra VRAM for the draft's own Q4_K head. Direction is favourable; magnitude unmeasured. No llama.cpp code change — the work is authoring the GGUF.

### lm_head GPTQ calibrated on 300k captured final hidden states, with a held-out KL split
**Where (theirs):** `drafter/gptq_lm_head.py:119-131` · `drafter/gptq_lm_head.py:106-107` · `drafter/README.md:23-28`

**What it does.** Calibration rows for lm_head are the target model's own final hidden states captured by capture.py — the exact tensor lm_head consumes at serving time. A fixed seed samples NCAL+20000 rows without replacement, sorted; the first NCAL become the Hessian, the remaining 20,000 are held out for evaluation.

**Mechanism.** `rng = np.random.default_rng(0)`; `rows = np.sort(rng.choice(T, size=min(NCAL+20000, T), replace=False))`; `cal, held = rows[:NCAL], rows[NCAL:]`. The Hessian is accumulated in 32,768-row chunks pulled from the uint16 memmap and reinterpreted as bf16 via `.view(torch.bfloat16)` (line 127). Sorting the row indices makes the memmap gather sequential.

**Why they needed it.** Calibrating a head on generic text activations mis-weights the input covariance; the head is only ever fed post-final-RMSNorm states from this model on this traffic mix.

**Their numbers.** Default --calib-rows 400000; the shipped fast-variant command uses `--calib-rows 300000` (README.md:51). Held-out set fixed at 20,000 rows. Hessian K = 5120.

**llama.cpp — EXISTS, NEVER SET.** The technique is 'calibrate the head on activations the head actually consumes'. llama.cpp does exactly this by construction: llama-imatrix hooks the real model on a real corpus, so the rows it sees are the true inputs. The gap is the default: params.process_output is false (common/arg.cpp:3132-3136), so output.weight gets no importance data unless you ask. That is a flag we have never set.

**Equivalent here:** llama-imatrix --process-output (default OFF) plus -f <calibration corpus>, then llama-quantize --imatrix

**Evidence (llama.cpp):** `common/arg.cpp:3132-3136` · `tools/imatrix/imatrix.cpp:296-330` · `src/llama-quant.cpp:1199-1210` · `src/llama-quant.cpp:922-936` · `common/arg.cpp:1813`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Only reachable if we re-quantise the target ourselves, which needs a BF16/F16 source we do not have locally (6.76 GiB IQ2_XXS is what is on disk; --allow-requantize from it would compound loss). If we ever do build our own GGUF, --process-output is off by default, so the output head — the tensor read on every decode AND every verify step — is the one tensor a default imatrix run never calibrates. Worth knowing.

### KL(bf16 head ‖ quantised head) on held-out hidden states as the acceptance metric for RTN vs GPTQ
**Where (theirs):** `drafter/gptq_lm_head.py:133-156` · `drafter/README.md:23-28`

**What it does.** Instead of judging lm_head quantisation by weight error, the script computes the exact per-token KL divergence between the bf16 head's distribution and the dequantised head's distribution on held-out hidden states, and prints it for RTN and for GPTQ so the two are compared on the same rows in the same process.

**Mechanism.** `kl_eval(Wq)` loops the held-out rows in blocks of 1024, forms `lp = log_softmax((x @ W.t()).float())` and `lq = log_softmax((x @ Wq.bf16().t()).float())` over the full 248k vocab, and sums `(lp.exp() * (lp - lq))`, returning the mean per row. RTN is evaluated first and its tensors freed before the GPTQ pass to fit in VRAM (line 145). Frobenius round-trip error is also printed (line 155-156) but is explicitly the secondary number.

**Why they needed it.** Weight error does not predict distribution error, and the distribution is what determines both perplexity and drafter acceptance. This is how the +1.5%/+0.6% perplexity claim was grounded.

**Their numbers.** Round-to-nearest int4 on lm_head: KL 0.0068, +1.5% perplexity. GPTQ int4 with the 300k-row Hessian: KL 0.0029, +0.6% perplexity, GSM8K 96.5% unchanged. Combined with GPTQ on the MTP module: −1.8 ms per decode step, 108.6 → 118.8 tok/s greedy (README.md:23-28).

**llama.cpp — EXISTS, NEVER SET.** syv-ref's point is that Frobenius weight error does not predict distribution error, so it computes exact KL over the full vocab on held-out rows. llama.cpp ships the same comparison as a first-class flag pair: --kl-divergence-base writes the reference model's logits to a file, --kl-divergence computes the divergence of a second model against them (common/arg.cpp:2495-2503). Neither appears anywhere in this project's tested register.

**Equivalent here:** llama-perplexity --kl-divergence-base FNAME (alias --save-all-logits) to dump reference logits, then --kl-divergence against the quantised model

**Evidence (llama.cpp):** `common/arg.cpp:2495-2503` · `common/arg.cpp:2502-2503`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** A ready-made, never-used instrument for exactly the question this repo keeps asking: does a quantisation change move the distribution or only the weights? It measures the target, not the drafter, and costs a full logits dump (248,320 floats per token) plus two perplexity runs. No serving-time cost. Directly relevant if we ever compare UD-IQ2_XXS against another 2-bit build.

### In-place int8 g128 requantisation of lm_head with a hard error gate
**Where (theirs):** `prepare/quant_lm_head.py:25-57` · `prepare/quant_lm_head.py:50-52`

**What it does.** Rewrites the bf16 lm_head of the published W4A16 checkpoint as symmetric int8, group 128 along K, in compressed-tensors pack-quantized layout, in place, on the CPU. Aborts if the round-trip relative Frobenius error exceeds 1%.

**Mechanism.** Reshape to [out, in/128, 128], `scale = clamp(amax(|g|)/127, min=1e-10)`, `q = clamp(round(g/scale), -128, 127).int8()`, dequantise, `err = ||deq-w||/||w||`, `assert err < 0.01`. Packing is `compressed_tensors.compressors.pack_quantized.base.pack_to_int32(q, 8, packed_dim=1)`. Scales are stored **fp16** with the comment "linear layers use fp16 scales in this checkpoint" (line 55). The shard is backed up as `.bak` and the index/config rewritten, with `lm_head` removed from the quantization_config ignore list and a new `group_1` targeting `re:.*lm_head$` at 8 bits.

**Why they needed it.** "The published W4A16 quants of Qwen3.8-27B leave lm_head in bf16 — that's a 2.5 GB matrix (248k vocab) read every decode step. int8 halves the read and frees ~1.3 GB of VRAM for the KV/state pool" (quant_lm_head.py:4-7).

**Their numbers.** +12% aggregate throughput on an RTX 3090; round-trip error 0.64% Frobenius; ~1.3 GB freed (quant_lm_head.py:6-7).

**llama.cpp — EXISTS, NEVER SET.** llama.cpp cannot rewrite a tensor in place (nothing maps a CLI flag to a load-time type change), but the offline equivalent is a single flag that outranks everything else: --output-tensor-type returns first in llama_tensor_get_type (llama-quant.cpp:683-688) and even bypasses tensor_type_fallback. The 'hard error gate' half is absent — llama.cpp prints no per-tensor round-trip error (see #39).

**Equivalent here:** llama-quantize --output-tensor-type <ggml_type>

**Evidence (llama.cpp):** `tools/quantize/quantize.cpp:410-427` · `src/llama-quant.cpp:683-688` · `src/llama-quant.cpp:456-471` · `src/llama-model.cpp:1378`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The head is the one vocab-sized tensor that DOES live in VRAM here (dev_output, llama-model.cpp:1378) and is read on every decode step and every speculative verify step. Setting its type is a real VRAM + bandwidth lever. But it is not a one-flag change for us: it requires re-quantising from an F16 source we do not have on disk, and the built-in mixture already forces the head to Q5_K for IQ2-class ftypes (llama-quant.cpp:456-471), so the headroom may be small. Magnitude unknown — I could not read the actual output.weight type out of our GGUF, only the aggregate 'file type = IQ2_XXS'.

### embed_tokens int8 with bf16 (not fp16) scales, and a distinct backup name
**Where (theirs):** `prepare/quant_embed.py:145-153` · `prepare/quant_embed.py:94-99`

**What it does.** Same int8 g128 treatment for the untied 2.5 GB token embedding matrix, but the scale tensor is written in bfloat16 rather than fp16, and the shard backup is named `.bak_embed` rather than `.bak`.

**Mechanism.** `tensors[...weight_scale] = scale.squeeze(-1).to(torch.bfloat16)` with the comment "the embedding path creates scales in params_dtype (bf16), unlike the linears" (line 146). Config gets a `group_2` targeting `re:.*embed_tokens$`. The backup name carries a three-line justification: "quant_lm_head.py writes '.bak' for its own shard, and a checkpoint that puts embed_tokens and lm_head in one shard would have the second script overwrite the first one's pristine backup. drafter/train_mtp.py reads both" (lines 150-152).

**Why they needed it.** The dtype difference is a real load-time failure mode between two vLLM code paths; the backup-name collision would silently destroy the bf16 weights that train_mtp.py and gptq_lm_head.py both depend on.

**Their numbers.** Another ~1.3 GB freed; round-trip error 0.56% on an RTX 3090 (quant_embed.py:103). Also needs patches/qwen3_5-embed-quant.patch because "the qwen3_5 model code never passes quant_config to VocabParallelEmbedding" (lines 96-99).

**llama.cpp — EXISTS, NEVER SET.** The flag exists and we have never set it, so the status is honest; the value is the point. The scale-dtype and backup-name details are compressed-tensors plumbing with no GGUF counterpart — GGUF carries scales inside the block format.

**Equivalent here:** llama-quantize --token-embedding-type <ggml_type>

**Evidence (llama.cpp):** `tools/quantize/quantize.cpp:410-427` · `src/llama-quant.cpp:683-688` · `src/llama-model.cpp:1368-1370` · `src/llama-arch.cpp:672`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Near zero on VRAM, which is the honest answer and the useful one. token_embd is LLM_TENSOR_LAYER_INPUT and dev_input is hard-wired to CPU with the in-source comment that offloading the input layer has very little benefit (llama-model.cpp:1368-1370). Shrinking it frees host RAM, not the 12 GB we are fighting for. syv-ref's '~1.3 GB freed for the KV pool' does not transfer — that was a vLLM VocabParallelEmbedding on-device. Qwen3.8-27B also ships a separate output.weight (both token_embd.weight and output.weight appear in our load log), so the tied-embedding duplication that WOULD put a copy in VRAM does not apply either.

### Round-to-nearest int8/int4 requantisation of the whole MTP draft module
**Where (theirs):** `prepare/quant_mtp.py:28-40` · `prepare/quant_mtp.py:45-47` · `prepare/quant_mtp.py:56-71` · `prepare/quant_mtp.py:82-85`

**What it does.** Quantises the eight `mtp.*` linears (fc plus one full decoder layer, ~850 MB bf16) to int8 or int4 g128 in place, with `--keep-fc` to leave the 10240→5120 input projection in bf16.

**Mechanism.** Asserts all eight weights live in a single shard (`assert len(shards)==1`), then per-matrix does the standard amax/QMAX RTN with `QMAX = 2**(BITS-1)-1`, prints per-matrix round-trip relative error, packs with pack_to_int32 at the chosen bit width, rewrites the weight_map, and adds config group_3 targeting `re:^mtp\..*` (or `re:^mtp\.layers\..*` with --keep-fc).

**Why they needed it.** "In single-user mode that module runs once per draft token, so at 4 drafts/step it is read four times per step; int8 halves that traffic, int4 quarters it. The draft head only steers speculation (acceptance rate) — the sampled distribution stays exact either way — so this is a pure speed knob" (quant_mtp.py:5-9).

**Their numbers.** "Measured acceptance change: int8 none." (line 9). Round-to-nearest int4 costs "~2% acceptance on the MTP module" (README.md:24-25). --keep-fc leaves 105 MB in bf16.

**llama.cpp — EXISTS, NEVER SET.** llama-quantize is arch-agnostic — it walks tensors, so a dflash GGUF is quantisable like any other. The syv-ref finding that int8 costs zero acceptance and int4 costs ~2% is a useful prior for how far to push, but the correspondence is loose: their MTP module reads its own head, ours borrows the target's. Note llama-quantize is NOT staged in C:\AI\llama.cpp-dflash2 and would have to be built.

**Equivalent here:** run llama-quantize on the DFlash2 sidecar GGUF itself

**Evidence (llama.cpp):** `tools/quantize/quantize.cpp:90-115` · `src/llama-quant.cpp:1234-1237` · `src/llama-quant.cpp:374-420`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Our sidecar is already Q4_K_M: 1.92 B params, 1.05 GiB, 4.71 BPW (from the load log). Dropping to Q4_K_S or IQ4_XS would recover roughly 0.1-0.2 GiB. That matters more than it looks, because --fit does not size the draft (common/speculative.cpp:2468) — the server instead reserves the measured draft footprint and shrinks the TARGET (server-context.cpp:1032-1087), so every GB the draft gives back is a GB of target layers or KV. Blocked in practice by not having an F16 sidecar locally: requantising Q4_K_M -> IQ4_XS needs --allow-requantize (llama-quant.cpp:1234-1237) and compounds loss.

### Self-distillation prompt mix with per-source quotas and a per-source thinking probability
**Where (theirs):** `drafter/collect_prompts.py:35-131` · `drafter/collect_prompts.py:127-129`

**What it does.** Assembles ~6.8k prompts from six sources with fixed counts and a seeded RNG: UltraChat 200k (2,300, of which 25% of eligible ones keep 3 turns of history), Magicoder OSS-Instruct (1,100), syvai/da-instruction (1,100), syvai/reasoning-v1 (800), kobprof/skolegpt-instruct (1,000, system prompt preserved), GSM8K train (500). Each prompt is then tagged `think` true/false by a biased coin.

**Mechanism.** `R = random.Random(1234)` seeds everything. da-instruction is filtered toward long answers (`if len(ans) < 150 and R.random() < 0.85: continue`) and capped at 200 prompts per task type via a `seen_task` dict (collect_prompts.py:76-80). The thinking flag: `base = 0.7 if p["src"] in ("da-reasoning","gsm8k") else 0.4; p["think"] = R.random() < base`.

**Why they needed it.** This single prompt file is the calibration set for everything downstream — the draft vocabulary, the lm_head Hessian, the MTP Hessians and the DFlash2 Hessians all trace back to it, so its mix determines what all four are tuned for. The long-answer bias exists because the drafter is only exercised on generated tokens.

**Their numbers.** ~6.8k prompts, 45% thinking on (README.md:43-44). Quotas as listed; think probability 0.7 for reasoning/math, 0.4 otherwise.

**llama.cpp — EXISTS, NEVER SET.** llama.cpp makes the calibration corpus a bare file, with --chunks to bound it and --from-chunk to resume. The quota/thinking-probability structure is a property of the file, not of any tool. Marked exists-but-unused rather than not-applicable because 'which corpus' is a decision llama.cpp hands us and we have never made.

**Equivalent here:** llama-imatrix -f <calibration text> — the calibration corpus is a plain file argument

**Evidence (llama.cpp):** `common/arg.cpp:1085` · `common/arg.cpp:1738-1741` · `common/arg.cpp:3147-3151` · `src/llama-quant.cpp:922-936`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Zero at serving time; relevant only if we build a GGUF. The load-bearing idea does transfer though: whatever we pass to -f decides what every quantised tensor is tuned for, and a coding-agent workload is not wikitext. Since our target came from Unsloth we do not know their calibration mix — that is a real unknown behind every number in this project.

### W4A16 export of DFlash2 with a hand-written compressed-tensors config in vLLM's module-prefix namespace
**Where (theirs):** `drafter/quant_dflash2.py:93-119` · `drafter/quant_dflash2.py:100-116` · `drafter/quant_dflash2.py:9-12`

**What it does.** Writes the quantised drafter as a single `model.safetensors` plus a synthesised `quantization_config`: one pack-quantized group at `num_bits=BITS` targeting `["Linear"]`, optionally a second group for `fc` at a different width, and an ignore list naming the modules that must stay bf16 — the two grouped-conv kernel projections per layer, the candidate selector, and the hidden projection.

**Mechanism.** `ignore = ["re:.*kernel_projection$", "re:.*candidate_selector.*", "re:.*hidden_projection$"]` (+ `re:.*\.fc$` when fc-bits ≥ 16). The group dict is written literally with `group_size: 128, strategy: "group", symmetric: True, type: "int", actorder: None, observer: "memoryless_minmax", version: "0.17.0"`. Everything not in `LIN` is copied through untouched, so norms and convs survive in bf16. Files other than model.safetensors and config.json are copied from the source dir verbatim.

**Why they needed it.** The ignore patterns must be written "in vLLM's module-prefix namespace (model.layers.N..., model.fc, model.candidate_selector...)" (quant_dflash2.py:16-17) — the loader matches against runtime module paths, not checkpoint keys. The listed modules are left alone because "vLLM builds them with quant_config=None anyway" (line 10).

**Their numbers.** 1.61B params of layer matrices int4 GPTQ + fc (5120×25600) int4 by default; result ~1.0-1.19 GB instead of 3.85 GB bf16, "~3 ms less per decode step on a 3090, and the KV pool survives" (quant_dflash2.py:11-12, README.md:66-69).

**llama.cpp — EXISTS, NEVER SET.** The technique is 'quantise the drafter aggressively but name the tensors that must stay high-precision'. llama.cpp's equivalent is a regex-to-type map applied at quantise time: --tensor-type lowercases the pattern and regex_searches the tensor name (quantize.cpp:314-361), first match wins, and a match suppresses the built-in mixture for that tensor (llama-quant.cpp:712-715) — the same 'manual override' semantics as an ignore list, in GGUF's own namespace. tensor_type_fallback (374-420) then silently repairs block-size mismatches, which the vLLM path does not do.

**Equivalent here:** llama-quantize --tensor-type <regex>=<type> (plus --pure) applied to the DFlash2 sidecar GGUF

**Evidence (llama.cpp):** `tools/quantize/quantize.cpp:314-361` · `src/llama-quant.cpp:693-718` · `src/llama-quant.cpp:184-196` · `src/llama-quant.cpp:374-420`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The nearest thing in this whole slice to an actionable VRAM lever, and still blocked. Our sidecar is Q4_K_M / 1.05 GiB; its vocab-sized tensors are selector_predecessor and selector_successor at 256 x 248,320 each (~127 M params combined) and are prime --tensor-type targets, while conv kernel projections and selector_hidden are small enough to leave alone. Because --fit never sizes the draft (common/speculative.cpp:2468) and the server shrinks the TARGET by the measured draft footprint (server-context.cpp:1032-1087), every 100 MiB recovered here becomes target VRAM. Blocked on not having an F16 sidecar locally and on llama-quantize not being staged in C:\AI\llama.cpp-dflash2.

### Applying the request's top-k/top-p to the DFlash2 selector's 16-candidate proposal
**Where (theirs):** `drafter/README.md:102-104`

**What it does.** The DFlash2 candidate-selector walk proposes 16 candidates per position; the request's own top-k/top-p are applied to that proposal, with the cached distribution kept truncated so the verify step remains lossless. Described as "the DFlash2 analogue of the MTP draft truncation".

**Mechanism.** Documented only in the README for this slice (implementation lives outside drafter/prepare); toggled by `VLLM_DFLASH2_DRAFT_TOPK_TOPP`, on by default, `=0` disables.

**Why they needed it.** Same logic as the draft vocabulary: a candidate the sampler would never select is a guaranteed rejection, so filtering it out of the proposal costs nothing and may free a slot.

**Their numbers.** +2%, "inside the noise"; on by default anyway (README.md:102-104).

**llama.cpp — EXISTS, NEVER SET.** The vLLM technique filters the selector's 16-candidate proposal by the request's top-k/top-p. llama.cpp gives us half of that and only half: the p_min half is a real, wired, defaulted-off flag on exactly the DFlash2 selector; the top-k half is NOT configurable — sparams.top_k is set to selector_top_k read from the GGUF (16 for our sidecar) at speculative.cpp:1004, and the block that would have made the draft sampler configurable is commented out at 209-224. Also note, correcting the capability map's 'exists but unused' section (written for an ngram-mod-only profile): with draft-dflash active the selector DOES fill dp.dists (speculative.cpp:1258), so at temperature > 0 the maximal-coupling residual accept path in common/sampling.cpp:722-793 is live for us, not dead.

**Equivalent here:** --spec-draft-p-min (alias --draft-p-min) — live on the DFlash2 selector path, default 0.00; and --spec-draft-n-min, default 0

**Evidence (llama.cpp):** `common/speculative.cpp:1254-1271` · `common/speculative.cpp:1276-1281` · `common/arg.cpp:4101-4107` · `common/common.h:329` · `common/speculative.cpp:1004` · `common/speculative.cpp:209-224`

**Effort:** one-flag · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The single most actionable item in this slice for us. We run --spec-type draft-dflash,ngram-mod and have never set --spec-draft-p-min; it defaults to 0.00, i.e. the confidence early-stop is off. On the DFlash2 greedy path it computes softmax at the selector argmax as 1/sum(exp(s_k - s_max)) over the 16 candidates and breaks the block early below the threshold (speculative.cpp:1262-1271); on the stochastic path it breaks when the sampled candidate's probability is below it (1255-1257). With block_size = 8 on our sidecar, trimming low-confidence tail positions cuts verify-batch width, which also moves the FA kernel choice (Q->ne[1] <= 2 keeps the VEC kernel on quantised KV). Untested, no VRAM cost, one flag. Beware --spec-draft-n-min: it is all-or-nothing (result.clear() at 1276-1281) and has no input validation.

## absent, has a seam — 5

### Deriving the draft vocabulary from the model's own outputs rather than from external text
**Where (theirs):** `drafter/README.md:14-22` · `prepare/build_draft_vocab.py:21-23` · `drafter/gen_data.py:1-8`

**What it does.** The originally shipped id list was counted over external corpora (Danish web text/fineweb-2, English Wikipedia, Python source, plus 8.8M tokens of older model outputs). Replacing it with a list counted over 5.4M tokens generated by the target model itself (the drafter/gen_data.py corpus) raised coverage of what the model actually emits, and that single change was the largest speed win in the whole drafter directory.

**Mechanism.** gen_data.py generates 5.4M output tokens with the model's own default sampling params over the 6.8k-prompt mix; build_draft_vocab.py then counts token frequencies over that jsonl (its `texts_from` reads "prompt"/"response"/"messages"/"text" fields, build_draft_vocab.py:45-55) and takes the top N ids. Because the drafter is only ever asked to guess tokens the target will produce, the correct frequency distribution is the target's output distribution, not natural text.

**Why they needed it.** A token outside the slice "can never be drafted, so every such token is a guaranteed rejection *and* truncates the chain" (README.md:15-16) — coverage loss costs double: the rejection plus the whole remainder of the speculative chain.

**Their numbers.** Old external-corpus list: 92.1% coverage of generated tokens, 83% on code. Own-output list (5.4M tokens): 97.5% overall, 96% on code. Nothing else changed: 98.0 → 108.6 tok/s greedy, 90.0 → 107.4 tok/s at the model's default sampling (README.md:17-21).

**llama.cpp — absent, has a seam.** llama.cpp has no token-frequency counter and no tool that turns a corpus into a d2t list — nothing in tools/ or common/ counts output-token frequencies. The seam is entirely on the GGUF-authoring side (a script over our own bench corpora feeding a d2t tensor into the sidecar). The llama.cpp side of the seam is the d2t loader, which exists; the derivation side is absent.

**Equivalent here:** none shipped; the carrier is `d2t` (src/models/dflash.cpp:99-105) and the id list would be counted offline over our own server output

**Evidence (llama.cpp):** `src/models/dflash.cpp:99-105` · `src/llama-arch.cpp:648`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown. The syv-ref number (92.1% -> 97.5% coverage, 98.0 -> 108.6 tok/s) is not transferable — it was measured on a full-vocab MTP head being replaced by a slice, whereas our draft has no head of its own at all. The seam is real but the win here would be bandwidth, not acceptance.

### MTP Hessians dumped from the trainer's own validation forward passes
**Where (theirs):** `drafter/train_mtp.py:55` · `drafter/train_mtp.py:393-418` · `drafter/README.md:26-28`

**What it does.** `train_mtp.py --eval-only 1 --dump-hessians <file>` registers forward hooks on the eight MTP linears, runs the full validation set through the faithful reimplementation of the drafter, accumulates a fp32 Hessian of each layer's input, and saves `{module_name: H}`. Those Hessians are what requant_mtp_gptq.py consumes.

**Mechanism.** A dict maps checkpoint names to the live modules (`"mtp.fc": mtp.fc`, `"mtp.layers.0.self_attn.q_proj": mtp.layer.self_attn.q_proj`, …, train_mtp.py:397-404). `mk(name)` returns a forward hook that flattens `inp[0]` to [-1, K] and calls `accumulate_hessian`. Because eval runs the same depth-unrolled chain the drafter uses at serving time, the captured inputs include depth-2 states — the drafter's own hidden fed back — not just depth-1 inputs.

**Why they needed it.** The MTP module's input distribution is not observable from ordinary model activations: `fc` sees a concatenation of two RMS-normed tensors (target hidden + token embedding), and layers 2..d see the drafter's own output. Only the trainer's replica produces those rows without running vLLM.

**Their numbers.** The shipped command dumps them at `--max-seqs 400 --val-frac 0.4 --depths 2` (README.md:49-50), i.e. 160 validation sequences.

**llama.cpp — absent, has a seam.** The insight is that a drafter's input distribution is unobservable outside a running speculative decode. In llama.cpp the same is true and the same blocker exists in sharper form: llama-imatrix cannot even LOAD our sidecar standalone, because llama-context.cpp:154-161 throws 'dflash requires ctx_other to be set' for a sidecar with no tok_embd/output — the exact error our own run log emits during the fit probe. So capturing draft-side activations needs imatrix to run inside a server that has both contexts alive. The seam is named and real (the eval callback at ggml-backend.cpp:1979 is already how imatrix hooks tensors), but wiring it through common_speculative is a large patch.

**Equivalent here:** none shipped; the seam is ggml_backend_sched_set_eval_callback (used by tools/imatrix) plus a speculation-aware imatrix mode

**Evidence (llama.cpp):** `tools/imatrix/imatrix.cpp:296-330` · `ggml/src/ggml-backend.cpp:1730-1761` · `ggml/src/ggml-backend.cpp:1979-1980` · `src/llama-context.cpp:154-161`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown, and expensive. It only pays off if we also build our own sidecar quant, which we currently cannot (#17).

### GPTQ requantisation of the MTP module from the trainer's Hessians, reading weights out of the pre-quant backup
**Where (theirs):** `drafter/requant_mtp_gptq.py:152-176` · `drafter/requant_mtp_gptq.py:158-162` · `drafter/requant_mtp_gptq.py:177-182`

**What it does.** Produces a new model dir whose `mtp.*` linears are GPTQ int4 instead of RTN int4, keeping lm_head, the draft head and the ids from the source dir untouched.

**Mechanism.** bf16 MTP weights are re-read from `<orig>/model_extra_tensors.safetensors.bak-mtp` — the backup quant_mtp.py left behind — while the *current* extra-tensor file supplies everything else, with any key belonging to the eight linears skipped (`if any(k.startswith(m + ".") for m in LIN): continue`). Each weight is GPTQ'd on GPU against `HS[m]`, packed, and given fp16 scales. Finally it asserts `g["targets"] == ["re:^mtp\\..*"]` before flipping `num_bits` — a guard that the config it inherited is the one quant_mtp.py wrote.

**Why they needed it.** Round-to-nearest int4 costs ~2% acceptance on the MTP module; the calibrated version "keep[s] acceptance intact" (README.md:24-27). Reading from the .bak-mtp backup is what makes the quantisation re-runnable at different bit widths without re-downloading the checkpoint.

**Their numbers.** Together with the int4-GPTQ lm_head: −1.8 ms per decode step, 108.6 → 118.8 tok/s greedy (README.md:28).

**llama.cpp — absent, has a seam.** The importance-weighted half is buildable — llama-quantize consumes an imatrix keyed by tensor name (llama-quant.cpp:1199-1210) and would apply it to sidecar tensors without modification. What is missing is a producer for that imatrix (#18) and the error-propagation half (#6). The 'read from the pre-quant backup' discipline is not needed: llama-quantize never rewrites in place.

**Equivalent here:** llama-quantize --imatrix <draft.imatrix> on the sidecar GGUF (imatrix-weighted, not GPTQ)

**Evidence (llama.cpp):** `src/llama-quant.cpp:1199-1210` · `src/llama-quant.cpp:922-936` · `tools/imatrix/imatrix.cpp:296-330` · `src/llama-context.cpp:154-161`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown. Gated behind #18 (no way to capture draft-side activations today) and #6 (no GPTQ, only diagonal-importance weighting). The syv-ref gain (-1.8 ms/step) is bundled with their lm_head change and does not decompose.

### Capturing every token's final hidden state by hooking GPUModelRunner._model_forward
**Where (theirs):** `drafter/capture.py:54-92` · `drafter/capture.py:59-61` · `drafter/capture.py:16-18`

**What it does.** Replays the generated corpus through vLLM with `max_tokens=1` and records, for every token of every sequence, the exact tensor vLLM hands the MTP drafter as `hidden_states` (the output of the final RMSNorm). Result: a [T, 5120] bf16 array.

**Mechanism.** Two monkeypatches. `GPUModelRunner.execute_model` is wrapped only to stash `scheduler_output` into a dict, because the forward hook itself does not receive it. `GPUModelRunner._model_forward` is wrapped to take the returned hidden states, slice `[:n_tot]` where `n_tot = sum(num_scheduled_tokens[r] for r in input_batch.req_ids[:num_reqs])`, move to CPU as bf16 and `.view(torch.uint16)`, then walk the requests in `input_batch.req_ids` order writing each request's slice at its memmap offset. `VLLM_ENABLE_V1_MULTIPROCESSING=0` is set at import so the hooks live in the same process as the model.

**Why they needed it.** The drafter's input is not the model's logits or any exported tensor — it is an internal activation. Capturing it from the serving stack (rather than from a HF forward pass) guarantees the calibration rows are exactly what the drafter will see, including fp8 KV cache effects.

**Their numbers.** 1.7 h, 74 GB memmap for the 5.4M-token corpus (README.md:46-47). Engine at max_num_seqs=8, max_num_batched_tokens=8192, max_model_len=8192, prefix caching off, cudagraph capture size 16.

**llama.cpp — absent, has a seam.** llama.cpp already exposes a per-node eval callback and two consumers of it, so dumping the post-final-norm hidden state to a file is a small patch on top of examples/eval-callback, not new infrastructure. The monkeypatching and scheduler-output stashing syv-ref needs have no counterpart because the hook is a supported API here.

**Equivalent here:** ggml_backend_sched_set_eval_callback — the same hook tools/imatrix and examples/eval-callback use

**Evidence (llama.cpp):** `ggml/src/ggml-backend.cpp:1979-1980` · `ggml/src/ggml-backend.cpp:1730-1761` · `tools/imatrix/imatrix.cpp:296-330`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown as a speed lever — it is a diagnostic seam, not a serving change. Worth flagging for a different reason: setting the eval callback takes ggml-backend off its fast path (ggml-backend.cpp:1730 branches on `if (!sched->callback_eval)` and otherwise walks node ranges), so any capture run is not representative of serving throughput. That is llama.cpp's version of syv-ref's enforce_eager requirement.

### DFlash2 Hessians captured from the drafter's own inputs during real speculation, in eager mode
**Where (theirs):** `drafter/capture_dflash2.py:1-15` · `drafter/capture_dflash2.py:64-72` · `drafter/capture_dflash2.py:83-112` · `drafter/capture_dflash2.py:137`

**What it does.** Runs a full vLLM engine in-process with `speculative_config={"method":"dflash","model":DRAFT,"num_speculative_tokens":7}` on the bf16 DFlash2 drafter, hooks the drafter's linear modules, and accumulates a GPTQ Hessian per module from the activations the drafter genuinely receives while speculating on the self-distillation prompts at model-default sampling.

**Mechanism.** `enforce_eager=True` is mandatory — the docstring says "CUDA graphs/torch.compile would bypass module hooks". Hooks are `register_forward_pre_hook` on the *fused* vLLM modules: `self_attn.qkv_proj` (K=5120), `self_attn.o_proj` (4096), `mlp.gate_up_proj` (5120), `mlp.down_proj` (17408), `fc` (25600); the input width is read from `mod.input_size` or `mod.input_size_per_partition`. Sampling is `temperature=1.0, top_p=0.95, top_k=20` (the model default). `assert len(stats) == 4*5 + 2` guards that exactly 22 things were hooked (5 layers × 4 fused modules, plus fc and ctx_kv).

**Why they needed it.** The drafter's inputs depend on the target's hidden states at layers 5/19/33/47/61 and on its own recursive state; nothing outside a running speculative decode produces that distribution. The 22-module assert exists because a silently unhooked module falls back to RTN and would produce a plausible but worse checkpoint.

**Their numbers.** 400 prompts × max 384 tokens, k=7, ~20 min; max_model_len 4096, max_num_seqs 8, FLASH_ATTN backend, kv_cache_dtype bfloat16, prefix caching off (README.md:75-80, capture_dflash2.py:31, 64-72).

**llama.cpp — absent, has a seam.** The technique's core claim — that the drafter's input distribution only exists during real speculation — holds identically in llama.cpp, and sharper: DFlash forces llama_set_causal_attn(ctx_dft, false) and consumes target hidden states from specific extract layers ([6,20,34,48,62] per our GGUF metadata), none of which is reproducible outside the speculative path. The seam exists (eval callback) but nothing shipped hooks the draft context. The 22-module assert has a llama.cpp analogue in spirit: llama-quantize's 'did not find weights for' line (#38).

**Equivalent here:** ggml_backend_sched_set_eval_callback wired into a speculation-aware imatrix run

**Evidence (llama.cpp):** `ggml/src/ggml-backend.cpp:1979-1980` · `ggml/src/ggml-backend.cpp:1730-1761` · `tools/imatrix/imatrix.cpp:296-330` · `src/llama-context.cpp:154-161` · `common/speculative.cpp:910-1347`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown. Same gate as #18: our sidecar cannot be loaded standalone, so any capture must happen inside a live server. And the eager-mode requirement transfers exactly — setting the eval callback takes ggml-backend off its batched split path (ggml-backend.cpp:1730), so the captured run is not a throughput-representative run.

## partial — 4

### Minimal GPTQ implementation with one quantisation group per GPTQ block and no act-order
**Where (theirs):** `drafter/gptq_utils.py:20-73` · `drafter/gptq_utils.py:42` · `drafter/gptq_utils.py:23-25`

**What it does.** A ~50-line GPTQ (Frantar et al.) that takes W [N,K] and a fp32 Hessian H [K,K] of the layer input and returns int8 codes plus fp32 per-group scales, in the exact layout compressed-tensors pack-quantized expects. Column order is preserved, so no `g_idx` permutation tensor is produced.

**Mechanism.** Dead columns (zero Hessian diagonal) are pinned to H=1 and their weights zeroed (lines 31-33). Damping is `percdamp * mean(diag(H))` with percdamp=0.01 added to the diagonal (34-35). The inverse is taken as cholesky → cholesky_inverse → upper cholesky again, giving the standard GPTQ Cholesky factor of H^-1 (36-38). The outer loop walks K in blocks of `blocksize`; within a block each column i is quantised, `err = (w - q)/Hinv1[i,i]` is computed and subtracted from all later columns of the block weighted by `Hinv1[i, i:]` (65-68); after the block, the accumulated `Err1 @ Hinv[i1:i2, i2:]` is applied to the remaining columns (71). `assert blocksize == group` (line 42) forces exactly one quantisation group per GPTQ block.

**Why they needed it.** The assert carries its own reason: "one quantization group per GPTQ block keeps the scale logic simple". No act-order is chosen so that "Marlin needs no g_idx" (gptq_utils.py:24-25, quant_dflash2.py:7) — the serving kernel is the constraint on the quantiser, not accuracy.

**Their numbers.** percdamp default 0.01; group 128; symmetric int with codes in [-2^(b-1), 2^(b-1)-1]; scales stored fp16 at write time.

**llama.cpp — partial.** llama.cpp has importance-weighted quantisation but not GPTQ. The seam is src/llama-quant.cpp:1199-1210, where the only per-tensor state handed to the quantiser is `const float * imatrix` — a length-ne[0] vector, no [K,K] inverse-Hessian and no mechanism to propagate a column's rounding error into later columns. ggml_quantize_chunk's signature takes that vector and nothing else. GPTQ would need a new quant path there plus per-type kernels; act-order/g_idx has no GGUF representation either.

**Equivalent here:** imatrix-weighted k-quant / IQ-quant: llama-quantize --imatrix feeding a per-column importance vector into ggml_quantize_chunk

**Evidence (llama.cpp):** `src/llama-quant.cpp:1199-1210` · `src/llama-quant.cpp:727-760` · `ggml/src/ggml-quants.c:628-695` · `src/llama-quant.cpp:922-936`

**Effort:** large-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown, and not reachable cheaply. Our UD-IQ2_XXS was already built upstream with an imatrix. Adding true GPTQ error propagation would be a new quantiser path, and the payoff would land on a file we did not build.

### Streaming Hessian accumulation with running (2/n) rescale
**Where (theirs):** `drafter/gptq_utils.py:9-17`

**What it does.** Maintains H = (2/n_total) * X^T X incrementally over an unbounded stream of activation rows without ever materialising the full X, and without needing the row count in advance.

**Mechanism.** `H *= n_seen/(n_seen+n)` down-weights the existing accumulator by the new total, `n_seen += n`, then `X = sqrt(2/n_seen) * X.float()` and `H += X.t() @ X`. The 2/n factor is GPTQ's own scaling convention, kept so damping constants transfer. Used identically by three call sites: lm_head calibration, DFlash2 hooks, and train_mtp's --dump-hessians.

**Why they needed it.** The calibration sets are hundreds of thousands of rows wide (300k rows × 5120 for lm_head; 250k × 25600 for the DFlash2 fc), far too large to hold; and the loop must be interruptible/chunked without changing the result.

**llama.cpp — partial.** llama.cpp accumulates incrementally with counts (imatrix.cpp:327 `e.counts[ex]++`, 390 `acc` += per-column), normalises as values[i]/counts[i] at save (126-147), and can checkpoint every N iterations (--output-frequency, --save-frequency at common/arg.cpp:3109-3131). But what it accumulates is the DIAGONAL of X^T X (one float per input column), not the full [K,K] Hessian GPTQ needs. Partial because the streaming pattern is present and the object it streams is smaller.

**Equivalent here:** llama-imatrix's streaming accumulator: per-column sums of squared activations with per-expert counts, normalised at save time

**Evidence (llama.cpp):** `tools/imatrix/imatrix.cpp:296-330` · `tools/imatrix/imatrix.cpp:374-409` · `tools/imatrix/imatrix.cpp:126-147` · `common/arg.cpp:3109-3131`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The streaming machinery is already there and already bounded — 5,120 floats per tensor, not 5,120^2 — so the memory problem this technique solves does not exist here. No VRAM or tok/s effect at serving time.

### Generating the self-distillation corpus at the model's own default sampling params, resumably
**Where (theirs):** `drafter/gen_data.py:168-191` · `drafter/gen_data.py:157-166` · `drafter/gen_data.py:148-149` · `drafter/README.md:45`

**What it does.** Runs the served model offline over prompts.jsonl and stores raw token ids (prompt_ids + output_ids + finish reason) rather than text, at the model's own default sampling parameters, with thinking toggled per prompt via the chat template.

**Mechanism.** `base = llm.get_default_sampling_params()` is cloned per request and only `max_tokens`/`n` overridden — 2048 tokens if thinking, 1024 if not. Resumption reads the output file and skips ids already present. Prompts tokenising to >3000 ids are dropped (line 206). The engine runs `kv_cache_dtype="fp8"`, `mamba_ssm_cache_dtype="float16"`, `max_model_len=8192`, `max_num_seqs=64`, prefix caching **off**, and the documented launch sets `VLLM_MARLIN_INPUT_DTYPE=int8 VLLM_MARLIN_INT8_INCLUDE_RE=mlp`.

**Why they needed it.** The draft vocabulary and every Hessian must reflect the distribution the *server* actually produces, which means the server's default sampling settings, not greedy. Storing ids rather than text makes the corpus re-usable byte-exactly by capture.py, train_mtp.py and capture_dflash2.py without retokenisation drift. Prefix caching is off so no token is skipped.

**Their numbers.** 2.2 h wall clock, 5.4M output tokens (README.md:45). MAX_THINK 2048, MAX_NOTHINK 1024, chunk 512 prompts.

**llama.cpp — partial.** Two halves. The resumable-accumulation half already exists and is better factored than syv-ref's skip-what-is-in-the-file loop: --save-frequency writes periodic copies, --from-chunk resumes at a chunk index. The generate-at-default-sampling half is just running our own server and is not a llama.cpp capability question. Partial because only one half has a shipped surface.

**Equivalent here:** generation via llama-server/llama-cli; resumability via llama-imatrix --from-chunk / --save-frequency / --output-frequency

**Evidence (llama.cpp):** `common/arg.cpp:3125-3131` · `common/arg.cpp:3147-3151` · `common/arg.cpp:3109-3116` · `tools/imatrix/imatrix.cpp:63`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown; a build-time convenience with no serving effect.

### Per-matrix relative Frobenius error reported for every quantised weight, with a mean
**Where (theirs):** `drafter/quant_dflash2.py:77-91` · `drafter/requant_mtp_gptq.py:167-168` · `prepare/quant_mtp.py:63-65` · `drafter/README.md:105`

**What it does.** Every quantisation script computes `||dequant(q,s) - W|| / ||W||` per matrix immediately after quantising and prints it; quant_dflash2.py prints only layer 0 and non-layer matrices in full but reports the mean over all 36.

**Mechanism.** `rel = ((dequant(q, s) - W.float()).norm() / W.float().norm()).item()`, collected into a `stats` list of `(base, bits, rel)`. In the RTN scripts this is upgraded to a hard gate (`assert err < 0.01` in quant_lm_head.py:52 and quant_embed.py:143).

**Why they needed it.** It is the only per-matrix sanity signal available before a serving run; a matrix whose error is an order of magnitude off the rest means a shape, transpose or Hessian-key mismatch.

**Their numbers.** DFlash2 int4: "Relative weight error of the int4 matrices: 0.147 mean (Frobenius), like the MTP module" (README.md:105). lm_head int8 RTN: 0.64%. embed_tokens int8 RTN: 0.56%.

**llama.cpp — partial.** Partial rather than already-have-it: llama.cpp reports size and any fallback per tensor, and reports the fallback COUNT at the end (1302), but never computes ||dequant(q)-W||/||W||. The distributional check that syv-ref treats as secondary is the one llama.cpp does have, as an external tool (#11). The syv-ref hard gates (assert err < 0.01) have no counterpart.

**Equivalent here:** per-tensor type/size lines and a fallback count at llama-quantize; distributional error only via llama-perplexity --kl-divergence

**Evidence (llama.cpp):** `src/llama-quant.cpp:1195` · `src/llama-quant.cpp:1296-1303` · `src/llama-quant.cpp:381-418` · `common/arg.cpp:2495-2503`

**Effort:** small-patch · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** unknown, and low priority given we do not build our own GGUFs. If we ever did, the missing per-tensor error is a genuine blind spot — llama.cpp will tell you a tensor fell back to a different type (381-418) but not that a tensor quantised badly at its intended type. The seam for adding it is a few lines after llama_tensor_quantize_impl returns.

## already have it — 8

### Slicing the draft head out of the already-quantised lm_head, not out of bf16
**Where (theirs):** `prepare/build_draft_vocab.py:95-101` · `prepare/README.md:9-11` · `drafter/gptq_lm_head.py:184`

**What it does.** The draft head is never quantised on its own. It is a row-subset of whatever quantisation lm_head currently carries — packed codes and per-group scales are index_selected together — so the draft head and the verify head are bit-identical on the rows they share.

**Mechanism.** Order dependency is enforced by documentation rather than code: `quant_lm_head.py` must run before `build_draft_vocab.py` "because build_draft_vocab.py slices its rows" (prepare/README.md:9-11). Consequently, when lm_head's bit-width changes, the draft head is stale: gptq_lm_head.py's last line prints "(draft head still needs prepare/build_draft_vocab.py --ids if lm_head bits changed)".

**Why they needed it.** It makes the draft head free in build cost and guarantees the drafter's scores are drawn from the same numerical head the target uses, removing quantisation mismatch as a source of rejections.

**Their numbers.** In the fast-variant pipeline the sequence is gptq_lm_head.py → build_draft_vocab.py --ids → requant_mtp_gptq.py (README.md:51-53).

**llama.cpp — already have it.** The syv-ref technique index_selects the draft head out of the quantised lm_head so drafter and verifier share numerics. llama.cpp does not slice — it shares the tensor outright. llama-context.cpp:154-161 gates on the sidecar lacking tok_embd/output and then wires cparams.ctx_other = the target context. The failure mode syv-ref guards against (a stale draft head after lm_head's bit-width changes, gptq_lm_head.py:184) cannot occur here.

**Equivalent here:** ctx_other head sharing — a DFlash/EAGLE3 sidecar that ships no tok_embd/output uses the target's own tensors

**Evidence (llama.cpp):** `src/llama-context.cpp:154-161` · `src/models/dflash.cpp:97` · `src/models/dflash.cpp:203-204` · `common/speculative.cpp:2461`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Already in force and strictly stronger than the vLLM technique: zero extra bytes and bit-identical scoring by construction, not by a build-order convention. Proof it fires on our artifact: the run log carries the exact string from llama-context.cpp:157, 'dflash requires ctx_other to be set', during the fit probe — which only throws when model.tok_embd == nullptr || model.output == nullptr.

### Optional MSE scale-clipping search over 11 shrink factors
**Where (theirs):** `drafter/gptq_utils.py:21` · `drafter/gptq_utils.py:52-60` · `drafter/gptq_lm_head.py:108`

**What it does.** With `mse_clip=True` (exposed as `--mse-clip` on gptq_lm_head.py), the per-row group scale is chosen by grid search instead of by amax: 11 factors linearly spaced in [0.75, 1.0] scale the amax down, each is used to round-trip the block, and the factor minimising the squared error per row wins.

**Mechanism.** `for f in torch.linspace(0.75, 1.0, 11)`: `s_ = clamp(amax*f/qmax, 1e-8)`, `q_ = clamp(round(W1/s_),-qmax-1,qmax)*s_`, `e = ((q_-W1)**2).sum(1)`; a per-row `best`/`best_err` pair is updated with a boolean mask so different rows can pick different factors.

**Why they needed it.** Pure amax scaling wastes code range on a single outlier per group; trading a little clipping for finer resolution is the standard fix. It is off by default in the shipped pipeline — the shipped commands never pass --mse-clip.

**Their numbers.** 11 candidates over [0.75, 1.0]; per-row (not per-tensor) choice.

**llama.cpp — already have it.** syv-ref grid-searches 11 shrink factors in [0.75, 1.0] per row and keeps it off by default. llama.cpp does a wider search unconditionally: make_qx_quants computes an amax-based scale then scans 19 candidates (is = -9..9), recomputing sumlx/suml2 and keeping the best weighted fit (ggml-quants.c:672-691), with the imatrix as the weight. The IQ2/IQ3 paths at 3376-3401 and 3550-3580 do the same. This is strictly more search than the vLLM stack's optional path.

**Equivalent here:** make_qx_quants' 19-candidate scale search (is = -9..9) and the equivalent loops in the IQ paths

**Evidence (llama.cpp):** `ggml/src/ggml-quants.c:628-695` · `ggml/src/ggml-quants.c:672-691` · `ggml/src/ggml-quants.c:3550-3580` · `ggml/src/ggml-quants.c:3376-3401`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Already applied to whatever GGUF we load, on by default in every k-/IQ-quant path. Nothing to set.

### Row-chunked GPTQ over the 248k-row lm_head
**Where (theirs):** `drafter/gptq_lm_head.py:147-152`

**What it does.** Quantises lm_head 16,384 output rows at a time against the one shared Hessian, moving each chunk's results to CPU and emptying the CUDA cache between chunks, then concatenating.

**Mechanism.** `for r0 in range(0, V, 16384): q_, s_ = gptq_quantize(W[r0:r0+16384], H, ...)` with the comment "rows are independent under GPTQ; chunk to bound memory". Correct because the GPTQ update only ever propagates error along K (columns) — every output row is solved independently given H.

**Why they needed it.** The intermediate fp32 copies of a [248064, 5120] matrix will not fit on a 24 GB card alongside the bf16 original and the 5120² Hessian.

**llama.cpp — already have it.** Same correctness argument (output rows are independent), same implementation shape: llama_tensor_quantize_impl splits nrows into chunks and hands `first_row * n_per_row` offsets to worker threads (llama-quant.cpp:742-756). llama.cpp additionally quantises on CPU, so the VRAM constraint that forced syv-ref's 16,384-row chunks does not arise.

**Equivalent here:** llama_tensor_quantize_impl's row chunking across worker threads

**Evidence (llama.cpp):** `src/llama-quant.cpp:727-760` · `src/llama-quant.cpp:742-756`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Already how every llama-quantize run works; the 248k-row head is never materialised whole in fp32. Nothing to set.

### Scoring against the true next token, response tokens only — the criterion that killed the fine-tune
**Where (theirs):** `drafter/train_mtp.py:310-318` · `drafter/train_mtp.py:337-344` · `drafter/README.md:29-37`

**What it does.** Provides two different correctness criteria and makes the honest one the default for eval: (a) top-1 agreement between drafter argmax and target argmax, and (b) vLLM's actual greedy criterion — the drafted token id equals the true next token in the corpus. Positions are masked to response tokens unless `--include-prompt`.

**Mechanism.** `ok_true = (draft_ids[dl.argmax(-1)]) == x_true` maps the draft-head column back through the id list to a real token id before comparing (line 312). Validity mask: `valid = (t+dd) <= n-1` and, without --include-prompt, `& ((t+dd+1) >= n_prompt)` so the *predicted* token is a response token. An `MTP_DEBUG` env var additionally reports the in-vocab rate, the joint (agreement ∧ in-vocab) rates and how often the target's own argmax equals the true token (lines 313-317).

**Why they needed it.** This is the load-bearing negative result of the directory: the fine-tune "halves the KL and looks great on the naive metric — until you score only response tokens against the *actual* next token, where top-1 agreement is unchanged" (README.md:31-33). The naive metric measured agreement with the target's argmax, which the fine-tune trivially improves on prompt positions nobody drafts.

**Their numbers.** Fine-tuned vs original top-1 against the true token: 0.685 → 0.685, vLLM acceptance within noise. Training was KL over the draft vocab, depth-2 unrolled chains, 7M tokens, one epoch (README.md:30-34).

**llama.cpp — already have it.** llama.cpp never computes a 'top-1 agreement with the target's argmax' figure. Acceptance is counted only where a draft was actually laid into a verify batch and only against the token the target actually sampled (common/sampling.cpp:692-720 accepts while draft[i] == sampled id). Prompt positions are excluded by construction — the ngram-* speculators' process() is a stub and no draft is produced during prefill.

**Equivalent here:** the server's acceptance counters: n_draft_accepted / n_draft_tokens over real generation only

**Evidence (llama.cpp):** `tools/server/server-context.cpp:634-637` · `tools/server/server-context.cpp:2966` · `tools/server/server-context.cpp:3877-3903` · `common/sampling.cpp:692-720`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Already what we measure. Our +34.7% / +48.5% numbers come from this counter, not from an agreement proxy — so the failure mode that killed syv-ref's fine-tune (a metric that improves on positions nobody drafts) is structurally unavailable to us.

### Greedy chain simulation of vLLM's speculative loop from the per-position accept maps
**Where (theirs):** `drafter/train_mtp.py:354-367` · `drafter/train_mtp.py:385-386` · `drafter/README.md:111-115`

**What it does.** Converts per-position, per-depth correctness into the number the server actually cares about — accepted tokens per decode step — by walking each sequence exactly as the speculative loop does: at position p accept depths until the first miss, then jump forward by accepted+1.

**Mechanism.** `while p_ < n_ - 1 - DEPTHS:` accumulate `acc` over depths while `oks[d_][p_]` is true, break on the first false; `sim["steps"] += 1; sim["tokens"] += acc + 1; p_ += acc + 1`. Reported as `chain: {tok_per_step, pos0, steps}`. Starting position is `max(0, n_prompt - 2)` when prompt positions are excluded.

**Why they needed it.** A whole-sequence top-1 rate over-reports, because the loop lands disproportionately on hard positions: "Positions right after a rejection are systematically harder: vLLM's per-position acceptance is measured there, so it sits ~5 points below a whole-sequence top-1 rate. The chain simulation in train_mtp.py --eval-only accounts for that" (README.md:112-115).

**Their numbers.** `--eval-only --depths 4` prints 2.5 tok/step vs vLLM's measured 2.6 for the original head (README.md:36-37).

**llama.cpp — already have it.** syv-ref has to simulate the loop because their per-position maps come from an offline replica; they explicitly note the simulation exists to account for post-rejection positions being harder. llama.cpp measures the real loop: n_draft_verif_steps increments per verify and mean len is 1 + accepted/steps (server-context.cpp:634-637), plus a per-position accepted histogram sized to n_max (3899). There is nothing to port.

**Equivalent here:** `draft acceptance = ... mean len = 1 + accepted/verif_steps` and the Prometheus per-position histogram

**Evidence (llama.cpp):** `tools/server/server-context.cpp:634-637` · `tools/server/server-context.cpp:3883-3903` · `tools/server/server-context.cpp:3899` · `tools/server/server-task.cpp:1551-1561`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Measured directly rather than simulated, which is strictly better. One caveat the map flags and this technique's own reasoning predicts: dp.n_max truncation near the context edge (server-context.cpp:451, speculative.cpp:2728-2732) silently shortens drafts without lowering the acceptance ratio, because n_draft_tokens is counted after truncation — so mean len can fall with acceptance flat.

### Reusing one fused-module Hessian across its constituent sub-weights
**Where (theirs):** `drafter/quant_dflash2.py:36-47` · `drafter/quant_dflash2.py:19-20`

**What it does.** The checkpoint stores separate `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj` weights, but vLLM fuses them at load into `qkv_proj` and `gate_up_proj`, which is where the Hessians were captured. A dict maps each checkpoint weight name to the fused Hessian key it should use.

**Mechanism.** `LIN[f"layers.{i}.self_attn.{p}"] = f"layers.{i}.self_attn.qkv_proj"` for p in q/k/v; `LIN[f"layers.{i}.mlp.{p}"] = f"layers.{i}.mlp.gate_up_proj"` for gate/up; o_proj and down_proj map to themselves; `fc` maps to `fc` unless `--fc-bits 16`. Correct because fused modules share one input tensor, so the Hessian of the input is identical for every sub-weight.

**Why they needed it.** The docstring states it: "q/k/v share one input (vLLM fuses them into qkv_proj) and so do gate/up (gate_up_proj), so the Hessians are captured per fused module and reused for each sub-weight." It also cuts capture cost — 22 Hessians instead of 36.

**Their numbers.** 35 layer matrices (5 layers × 7) plus fc; ~40 s total on GPU (README.md:82-84).

**llama.cpp — already have it.** syv-ref needs a dict because the checkpoint stores q/k/v separately while vLLM fuses at load. llama.cpp fuses at GGUF-conversion time, so the tensor that is quantised and the tensor whose input was measured are the same object. llama-quant.cpp:76-91 (remap_imatrix) exists for the residual case where an imatrix was captured under different tensor names, and aborts loudly on a mapping error rather than silently reusing the wrong one.

**Equivalent here:** GGUF fuses qkv at conversion time (blk.N.attn_qkv.weight), so the imatrix is naturally per-fused-tensor; remap_imatrix handles name remapping

**Evidence (llama.cpp):** `src/llama-quant.cpp:76-91` · `tools/imatrix/imatrix.cpp:296-330`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Already how it works. Confirmed on our own artifact: the target loads blk.0.attn_qkv.weight as a single tensor, so there is one imatrix entry for the fused input by construction — no mapping table to maintain.

### Loud RTN fallback when a Hessian key is missing
**Where (theirs):** `drafter/quant_dflash2.py:61-63` · `drafter/quant_dflash2.py:30`

**What it does.** If the Hessian file lacks the key a weight maps to, the script does not abort — it round-to-nearest quantises that matrix and prints `!! no Hessian for <key>, RTN fallback`. `--rtn` forces this for everything, as a comparison baseline.

**Mechanism.** `if RTN or LIN[base] not in HS: if not RTN: print(...); q, s = rtn_quantize(W, bits, GROUP)`.

**Why they needed it.** The 22-module capture assert can be defeated by a vLLM version change in module naming; without the warning the result would be a checkpoint that loads, serves, and quietly accepts fewer tokens. It is the paired safety net to `assert len(stats) == 22` in the capture script.

**llama.cpp — already have it.** This is the one place llama.cpp is unambiguously ahead. syv-ref prints a warning and continues with RTN; llama.cpp does three escalating things: logs 'did not find weights for' per tensor (1203), THROWS on a size mismatch with an in-source comment saying people miss the warning and end up with most of the model unquantised by imatrix (1211-1218), and hard-aborts when a very-low-bit tensor has no imatrix (1222-1227). Exactly the 'crash rather than return a believable number' posture this repo argues for.

**Equivalent here:** 'did not find weights for %s', the imatrix-size-mismatch throw, and the hard abort for very-low-bit tensors without an imatrix

**Evidence (llama.cpp):** `src/llama-quant.cpp:1199-1210` · `src/llama-quant.cpp:1211-1218` · `src/llama-quant.cpp:1222-1227` · `src/llama-quant.cpp:1296-1303`

**Effort:** n/a · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** Already in force and stronger than the technique it mirrors. For IQ2-class targets like ours, a missing imatrix is fatal, not a silent downgrade — llama-quant.cpp:1222-1227 prints 'The result will be garbage, so bailing out' and throws. It also reports the fallback count at the end (1302).

### Measurement-noise discipline for acceptance numbers
**Where (theirs):** `drafter/README.md:116-120` · `drafter/README.md:87-99`

**What it does.** The README fixes the benchmark protocol used for every acceptance claim in the directory — 8 realistic prompts × 1,024 tokens against the fast-variant target — and states the noise floor for each sampling mode, plus the reason greedy is not reproducible across drafter configs.

**Mechanism.** "Greedy decoding with speculation is not bit-deterministic across drafter configs (verify batches of 5 vs 1 token round differently), so 8 prompts × 1k tokens has a ±3% spread on tokens/step. Repeat before believing a 2% difference. With DFlash2 the greedy spread is wider (3.1-3.6 tokens per step across launch configs), default sampling ±5% per run."

**Why they needed it.** Directly caused a near-miss: the ctx_kv Hessian blend "looked equal on a single default-sampling run, which is how it nearly shipped", and was only caught because greedy with four repeats (within 1.5 tok/s, identical step counts) was treated as the reproducible signal.

**Their numbers.** MTP greedy ±3% on tokens/step over 8×1k; DFlash2 greedy 3.1-3.6 tok/step across launch configs; DFlash2 default sampling ±5% per run. DFlash2 int4 vs bf16: 3.34-3.65 vs 3.54 greedy; 3.2 vs 3.4 at default sampling; per-step read 2.7 GB less, 31.4 → 28 ms with the base target, 26.5 ms with the fast variant. `fc` bf16 vs int4: 3.17 vs 3.17, no difference, for +0.26 GB.

**llama.cpp — already have it.** Marked already-have-it on the instrumentation, which is what llama.cpp can supply. syv-ref's specific noise figures (±3% MTP greedy, ±5% default sampling, 3.1-3.6 tok/step DFlash2 across launch configs) are their hardware and their harness and do not transfer as numbers — but the direction does, and it independently corroborates this repo's own rule that small deltas are noise. The 'greedy is not bit-deterministic across drafter configs because verify batches of 5 vs 1 token round differently' observation applies here too: llama.cpp's verify batch width is 1+n_draft and varies with acceptance.

**Equivalent here:** the per-completion `draft acceptance = X (N accepted / M generated), mean len = Y` line, the Prometheus spec_decode_* counters, and the per-implementation SPC_TRC statistics

**Evidence (llama.cpp):** `tools/server/server-context.cpp:634-637` · `tools/server/server-task.cpp:1551-1561` · `common/speculative.cpp:2829-2872` · `tools/server/server-context.cpp:4477` · `tools/server/server-context.cpp:2444-2446`

**Effort:** config · **Transferable:** yes

**Worth on a 4070 SUPER 12 GB:** The instruments exist and we already use them; the protocol is ours and stricter (13.6% noise floor, pair within a round, never compare raw decode across boots). Two llama.cpp-specific traps worth carrying into any repeat of today's +34.7% / +48.5% measurement: /metrics rate gauges are reset on every scrape (server-context.cpp:4477, 2444-2446) so a second scrape reads 0, and the per-implementation credit in a chained --spec-type goes only to spec->impl_last — so in draft-dflash,ngram-mod the two speculators' contributions are separable only at LOG_TRC.

## impossible here — 4

### Group scale computed from the error-compensated weights, after GPTQ has already updated them
**Where (theirs):** `drafter/gptq_utils.py:49-61`

**What it does.** Rather than deriving all group scales up front from the original weight matrix, the scale for group g is computed from `W1 = W[:, i1:i2]` as it exists at that moment — i.e. after every earlier column's quantisation error has been propagated into it by the GPTQ update.

**Mechanism.** Inside the block loop: `amax = W1.abs().amax(dim=1); scale = clamp(amax/qmax, min=1e-8)` on the freshly sliced (already error-compensated) block, stored into `scales[:, i1//group]`. A final pass re-derives the integer codes as `round(Q / scales.repeat_interleave(group,1)).clamp(...)` so the emitted codes are exactly consistent with the stored scales (line 72).

**Why they needed it.** The compensation can push weights outside the original group's dynamic range; deriving the scale from stale weights would clip precisely the columns GPTQ just decided to lean on.

**llama.cpp — impossible here.** This is a refinement inside a GPTQ loop. llama.cpp's block quantisers see the original weights only (ggml-quants.c:628 make_qx_quants takes `const float * x` and an optional weight vector); there is no compensated intermediate W to derive a scale from. Impossible without #6 first.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/llama-quant.cpp:1199-1210` · `ggml/src/ggml-quants.c:628-695`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** n/a — meaningless without GPTQ's error propagation, which llama.cpp does not have.

### A from-scratch reimplementation of the MTP drafter that matches vLLM to 1%
**Where (theirs):** `drafter/train_mtp.py:87-177` · `drafter/train_mtp.py:169-171` · `drafter/README.md:58-60`

**What it does.** Rebuilds the MTP module in plain PyTorch — `fc: Linear(2H, H)`, two pre-fc RMSNorms, one `Qwen3_5DecoderLayer` at the model's last full-attention layer index, a final norm, and rotary embeddings — loads the bf16 weights out of `model_extra_tensors.safetensors.bak-mtp`, and reproduces the drafter's arithmetic so training and evaluation numbers are trustworthy without running vLLM.

**Mechanism.** The forward is `x = fc(cat([pre_fc_norm_embedding(embed(x_{t+d})), pre_fc_norm_hidden(hid)], -1))` with positions `arange(d, d+L)` (lines 168-171). `mtp_layer_idx` is derived as the last index whose `layer_types` entry is `"full_attention"` (line 74). State-dict keys are remapped `mtp.layers.0.* → layer.*` with a strict assert that nothing is missing or unexpected (lines 188-199).

**Why they needed it.** "The trainer reproduces vLLM's drafter to 1% (checked by replaying captured drafter calls with their KV history) so its numbers are trustworthy" (README.md:58-59). Without that check, an eval-only harness measuring a drafter is exactly the "believable wrong number" failure mode.

**Their numbers.** Agreement with vLLM within 1%; chain simulation gives 2.5 tok/step vs vLLM's 2.6 for the original head at depths 4 (README.md:36-37).

**llama.cpp — impossible here.** The technique is 'build a replica and prove it matches the server before trusting its numbers'. There is nothing in llama.cpp to build a replica of a drafter for, because there is no training. The transferable residue is the divergence warning, which is why I marked it idea-only rather than not-applicable.

**Equivalent here:** none — llama.cpp has no drafter trainer or offline drafter replica to validate

**Evidence (llama.cpp):** `common/speculative.cpp:910-1347` · `examples/speculative/speculative.cpp:67`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** n/a as a capability. The underlying warning does apply here in a different form: llama.cpp already carries two divergent implementations of the speculative loop, and the map documents one consequence — --spec-draft-p-split is read only by examples/speculative/speculative.cpp:67 and by nothing the server runs. A number from llama-cli or the speculative example is not a number from llama-server.

### Depth-unrolled draft chains with log-sum-exp merged attention over the depth-1 KV history
**Where (theirs):** `drafter/train_mtp.py:98-139` · `drafter/train_mtp.py:113-134` · `drafter/train_mtp.py:80-84` · `drafter/train_mtp.py:141-153`

**What it does.** Trains/evaluates the drafter on chains of depth 1..4 the way vLLM actually runs them: depth d consumes the drafter's *own* depth-(d−1) hidden state, and attends to the depth-1 KV of all positions ≤ t plus one single key per intermediate depth of its own chain — all in one batched pass over every position t at once.

**Mechanism.** Depth 1 is plain `F.scaled_dot_product_attention(..., is_causal=True, enable_gqa=True)`. For depth ≥ 2 it runs a causal efficient-attention that also returns the per-row log-sum-exp (`torch.ops.aten._scaled_dot_product_efficient_attention` with the LSE output, wrapped as `_causal_attn_lse`), then merges in the chain terms manually: each chain key contributes a scalar score `s_ = (q * k).sum(-1) * scaling`, a running max `m` is taken over the LSE and all scores, weights `exp(lse1-m)` and `exp(s_-m)` are formed, and the outputs are combined as `num/den`. GQA is handled with `repeat_interleave(G, dim=1)` on the KV heads.

**Why they needed it.** A naive [L, depth*L] attention mask would be quadratic in a way the real drafter is not, and would train the model on keys it never sees at inference. The LSE merge lets the *whole sequence* of chain positions be trained in one pass while each query row still sees exactly the keys vLLM would give it.

**Their numbers.** depths configurable 1..4; the shipped fine-tune used `--depths 2 --depth-weights 1,0.5`.

**llama.cpp — impossible here.** This is a training-time attention construction. llama.cpp's only training surface is examples/training/finetune.cpp, a plain LM finetune with no drafter/MTP head support, and there is no path from it to an MTP or DFlash sidecar. The inference-side chain logic in common/speculative.cpp:1349-1785 is fixed and has no training counterpart.

**Equivalent here:** none

**Evidence (llama.cpp):** `examples/training/finetune.cpp` · `common/speculative.cpp:1349-1785`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** n/a

### Distillation loss restricted to the draft vocabulary, weighted by the target's in-vocab probability mass
**Where (theirs):** `drafter/train_mtp.py:281-322` · `drafter/train_mtp.py:294-302` · `drafter/train_mtp.py:54`

**What it does.** The training target is the *target model's* distribution over the draft vocab (computed on the fly from held hidden states through the bf16 lm_head), and each position's loss contribution is scaled by how much of the target's probability mass actually lies inside the draft vocabulary.

**Mechanism.** Per chunk of `--head-chunk` (default 1024) rows: `tl = (h_tgt @ lm_w.t()).float()`; `pf = softmax(tl)`; `cov = pf[:, draft_ids].sum(-1)` — the in-vocab mass — then `tl = tl[:, draft_ids]` and `p = softmax(tl)` is renormalised over the shortlist. The drafter's logits `dl = (z @ head_w.t())`; `ce = -(p * log_softmax(dl)).sum(-1)`; backward is `((ce * wrow).sum() * (weight/N)).backward()` with `wrow = cov` unless `--no-cov-weight`. KL is reported as `ce - entropy(p)`. Gradients are taken w.r.t. a detached `z_d` and later re-attached with `torch.autograd.backward(zs_v, grads)` (line 353) so the head chunking does not hold the whole graph.

**Why they needed it.** Positions whose true continuation is outside the draft vocab are unwinnable — the drafter will be rejected there regardless — so spending gradient on them is wasted. The post-hoc analysis found exactly this: "the KL gains were on prompt tokens and on positions whose true token is outside the draft vocab" (README.md:34-35).

**llama.cpp — impossible here.** A loss function; llama.cpp trains no drafters. The residual insight — positions whose true token is outside the draft shortlist are unwinnable — would matter only if we built a d2t sidecar (#1), and even then it is a training concern, not a llama.cpp one.

**Equivalent here:** none

**Evidence (llama.cpp):** `examples/training/finetune.cpp`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** n/a

## not applicable — 15

### Held-out coverage estimate by decimating the corpus at document granularity, plus an N-sweep
**Where (theirs):** `prepare/build_draft_vocab.py:66-70` · `prepare/build_draft_vocab.py:82-87`

**What it does.** While counting, every 10th document goes into a `held` counter instead of the `counts` counter used to pick ids. After picking the top-N, coverage is reported as the fraction of held-out token occurrences whose id is in the shortlist, and the same figure is printed for N = 16384, 32768, 49152, 65536 so the size can be chosen from the curve rather than guessed.

**Mechanism.** `(held if j % 10 == 0 else counts).update(ids)` (line 69) splits at the document (not token) level so the estimate is not contaminated by within-document repetition. The sweep at lines 84-87 rebuilds `set(counts.most_common(n_try)) | special` for each candidate N and recomputes coverage against the same held counter.

**Why they needed it.** Coverage is the only cheap proxy for acceptance loss and had to be measured on data not used to pick the ids; the sweep is what justified stopping at 40,960.

**Their numbers.** "Coverage barely improves past 40k rows (49k: 98.2%; the model only ever emits ~54k distinct tokens), and 49k measured no faster" (README.md:21-22). The shipped external-corpus list reported held-out coverage 95% (build_draft_vocab.py:23).

**llama.cpp — not applicable.** This is offline data methodology with no llama.cpp surface. Nothing in the tree counts tokens, splits held-out sets, or sweeps a shortlist size. It is a discipline for whoever builds the d2t list, not a capability llama.cpp can have or lack.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/models/dflash.cpp:99-105`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** unknown; it only becomes relevant if a d2t sidecar is ever built here.

### Forcing chat/thinking/tool control tokens into the draft vocabulary regardless of frequency
**Where (theirs):** `prepare/build_draft_vocab.py:73-81`

**What it does.** Before the top-N selection, a fixed set of control tokens is unioned into the shortlist and excluded from the frequency ranking so they do not consume top-N slots: the tokenizer's `all_special_ids` plus `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`.

**Mechanism.** `top = [t for t,_ in counts.most_common() if t not in special][: N - len(special)]` then `ids = sorted(set(top) | special)` (lines 80-81), so the final list is exactly N entries with the specials guaranteed present. When an explicit `--ids` file is given, `special` is reset to empty (line 75) except for the hard-coded names, so a shipped list is used byte-for-byte.

**Why they needed it.** These tokens appear at structurally critical, highly predictable positions (turn boundaries, think-block open/close, tool call framing); missing one there is a guaranteed rejection exactly where the drafter would otherwise be certain.

**llama.cpp — not applicable.** Same class as #3 — a rule for constructing an id list llama.cpp would then consume. There is no seam in llama.cpp where control tokens get privileged treatment in a draft; the draft head is whatever the GGUF ships.

**Equivalent here:** none (llama.cpp exposes special ids via llama_vocab_* but has no draft-shortlist builder)

**Evidence (llama.cpp):** `src/models/dflash.cpp:99-105`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** unknown; a d2t-build-time rule, not a runtime one.

### Building the fast variant as a hardlinked model dir with only the changed shard rewritten
**Where (theirs):** `drafter/gptq_lm_head.py:158-184` · `drafter/requant_mtp_gptq.py:142-151` · `drafter/export_mtp.py:24-35` · `drafter/export_mtp.py:51-53`

**What it does.** Every quantisation script emits a whole servable model directory but only physically writes the one or two safetensors files it changed; all other shards, the tokenizer and the draft-vocab ids are `os.link`ed from the source dir, and small json/jinja files are copied.

**Mechanism.** `os.link(S + f, D + f)` for `model-0000*.safetensors` except the shard being rewritten (gptq_lm_head.py:160-165). Because hardlinks alias inodes, any file about to be rewritten is `os.remove`d first — export_mtp.py:51-53 spells out the trap: "never truncate a hardlink". requant_mtp_gptq.py does the same before saving model_extra_tensors (lines 174-175).

**Why they needed it.** The base model is 19.5 GB; five experimental variants would otherwise be 100 GB. The remove-before-write rule exists because writing through a hardlink corrupts the source model dir silently.

**llama.cpp — not applicable.** The technique exists because HF checkpoints are multi-shard directories and the vLLM scripts rewrite in place. GGUF is a single artifact and llama-quantize's contract is read-one-file-write-another, so there is no hardlink aliasing hazard and no 'never truncate a hardlink' rule to encode. (llama-gguf-split and --keep-split exist but solve splitting, not variant dedup.)

**Equivalent here:** none needed — a GGUF is one file and llama-quantize always writes a new one

**Evidence (llama.cpp):** `src/llama-quant.cpp:1234-1237`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** n/a. The disk-space problem it solves (five 19.5 GB variant dirs) does not have a GGUF analogue at our sizes; our whole target is 6.76 GiB and the sidecar 1.05 GiB.

### Fixing the ignore list so the bf16 MTP module still loads once lm_head is quantised
**Where (theirs):** `prepare/quant_lm_head.py:71-85`

**What it does.** While rewriting config.json for the lm_head quantisation, the script also appends the eight `mtp.*` linear module names to `quantization_config["ignore"]` even though it is not touching them.

**Mechanism.** After `qc["ignore"] = [i for i in qc["ignore"] if i != "lm_head"]`, it loops over `mtp.fc`, `mtp.layers.0.mlp.{down,gate,up}_proj`, `mtp.layers.0.self_attn.{q,k,v,o}_proj` and appends any that are absent.

**Why they needed it.** Stated verbatim in the code: "The MTP draft head is stored in bf16 but missing from the ignore list, which breaks loading when speculative decoding is enabled (single-user mode)" (lines 72-73). The bug is latent in the published checkpoint and only fires once speculation is turned on.

**llama.cpp — not applicable.** The bug being fixed is a compressed-tensors config artifact: a bf16 submodule missing from an ignore list breaks loading once speculation is enabled. llama.cpp has no config-driven ignore list — quantisability is decided per tensor in code (tensor_allows_quantization, llama-quant.cpp:291-304) and a missing tensor fails loudly at create_tensor rather than silently. The nearest analogue is --tensor-type regex, covered under #37.

**Equivalent here:** none — GGUF has no per-module quantization ignore list; tensor_allows_quantization protects norms and 1-D tensors automatically

**Evidence (llama.cpp):** `src/llama-quant.cpp:291-304` · `src/models/dflash.cpp:97` · `src/models/dflash.cpp:203-204`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** n/a. Our sidecar is a separate GGUF, not a module inside the target checkpoint, so the latent-config bug class cannot occur.

### Aligning hooked activations to sequences via vLLM's '<counter>-<uuid>' request ids
**Where (theirs):** `drafter/capture.py:71-88` · `drafter/capture.py:79` · `drafter/README.md:108-110`

**What it does.** Recovers which rows of the fused, batched hidden-state tensor belong to which input sequence, in a scheduler that interleaves prefill chunks and decode steps from different requests in one forward pass.

**Mechanism.** Rows are consumed in `self.input_batch.req_ids` order, advancing `pos` by `num_scheduled_tokens[rid]` per request. The sequence index is recovered as `k = int(str(rid).split("-")[0])` — "LLM request ids are '<counter>-<uuid>'". A per-sequence `written` counter lets a sequence be filled across several forward passes (chunked prefill), and `take = min(n, s["n"] - w)` clamps the last chunk. At the end the script prints how many sequences never reached their full length.

**Why they needed it.** README.md:108-110 lists this under "Notes that cost time": the id format is a 0.27 implementation detail and the ordering must come from `input_batch.req_ids`, not from the order prompts were submitted.

**llama.cpp — not applicable.** This is recovery from an id-format accident in a specific vLLM version. llama.cpp's server assigns slot.id = i explicitly at server-context.cpp:1252 and the batch carries seq ids directly, so there is no string to parse and no ordering to reconstruct.

**Equivalent here:** none needed — the eval callback sees ubatch tensors and the server owns explicit slot/seq ids

**Evidence (llama.cpp):** `ggml/src/ggml-backend.cpp:1730-1761` · `tools/server/server-context.cpp:1252`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** n/a

### bf16 stored as uint16 in a .npy memmap, reinterpreted with Tensor.view on load
**Where (theirs):** `drafter/capture.py:47-48` · `drafter/gptq_lm_head.py:127` · `drafter/train_mtp.py:246-250` · `drafter/capture_dflash2.py:96-98`

**What it does.** All bulk activation storage uses `np.lib.format.open_memmap(..., dtype=np.uint16)` and every reader does `torch.from_numpy(np.array(chunk)).view(torch.bfloat16)`. Token ids go into a parallel int32 memmap and sequence boundaries into a small `seqs.json` with `{id, src, think, off, n, n_prompt}`.

**Mechanism.** numpy has no bf16 dtype, so the bit pattern is carried as uint16 and reinterpreted (not cast) on the torch side. `np.array(mm[slice])` forces a materialised copy before the view, since a memmap slice is not contiguous-owning.

**Why they needed it.** Halves the file size versus fp32 with zero conversion loss, and keeps the calibration rows numerically identical to what the GPU produced — a cast to fp32 and back would be lossless but a cast to fp16 would not be.

**Their numbers.** 74 GB for 5.4M tokens × 5120 dims; the DFlash2 wide-row dumps use the same scheme with a 250,000-row cap per module, ~56 GB of scratch (README.md:79).

**llama.cpp — not applicable.** The technique exists because numpy has no bf16 dtype. ggml does (BF16 is in the live type list at ggml.h:390-433), and llama.cpp mmaps tensor data in place rather than round-tripping through a numeric container, so there is nothing to reinterpret.

**Equivalent here:** GGML_TYPE_BF16 is a first-class ggml type; GGUF is mmapped directly

**Evidence (llama.cpp):** `ggml/include/ggml.h:390-433` · `src/llama-mmap.cpp:536-584`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** n/a

### Length-sorted greedy packing to a padded-token budget
**Where (theirs):** `drafter/train_mtp.py:266-276` · `drafter/train_mtp.py:39`

**What it does.** Groups variable-length sequences into micro-batches whose *padded* cost (longest sequence × batch size) stays under `--micro-tokens`, by sorting descending by length and closing a batch when adding the next sequence would exceed the budget.

**Mechanism.** `for s in sorted(seq_list, key=lambda s: -s["n"])`: `m = max(cur_max, s["n"])`; if `cur and m*(len(cur)+1) > budget`, flush. Descending order means `cur_max` is fixed by the first member, so the check is exact rather than a heuristic.

**Why they needed it.** The corpus mixes 1k-token non-thinking answers with 2k-token thinking traces plus prompts up to 3000; naive fixed-batch padding would waste most of the compute. Default budget 8192 padded tokens, 4096 in the shipped fine-tune command (README.md:56-57).

**Their numbers.** ~30 min/epoch at 4k tok/s with `--micro-tokens 4096` (README.md:57).

**llama.cpp — not applicable.** The waste this attacks is padded-batch waste in a fixed-shape trainer. llama.cpp's batch is a flat token list split by n_ubatch (llama-context.cpp:247) with continuous batching across slots (server-context.cpp:3022); sequences of different lengths never pad each other. No seam and no problem.

**Equivalent here:** none needed — llama.cpp batches ragged, with no padding

**Evidence (llama.cpp):** `src/llama-context.cpp:245-247` · `tools/server/server-context.cpp:3022`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** n/a, and at -np 1 there is a single sequence anyway.

### Two-tier Hessian accumulation: GPU fp32 for narrow inputs, memmap row dump for wide ones
**Where (theirs):** `drafter/capture_dflash2.py:32` · `drafter/capture_dflash2.py:93-111` · `drafter/capture_dflash2.py:166-183`

**What it does.** Modules with input width ≤ 5120 get a live fp32 [K,K] Hessian on the GPU updated in the hook; wider modules (down_proj at 17408, fc at 25600) instead have their raw input rows written to a uint16 memmap, capped at 250,000 rows, and the X^T X reduction is done in a second phase once the engine is gone.

**Mechanism.** At hook-install time: `if K <= SMALL_K: stats[mkey] = {"H": zeros(K,K,cuda,fp32), ...}` else `open_memmap(rows_<mkey>.npy, shape=(ROWCAP, K), dtype=uint16)`. The pre-hook branches on `"H" in st`. Phase 2 (`reduce_wide`) streams each memmap back in 8192-row blocks, views as bf16, and folds into a GPU Hessian via the same `accumulate_hessian`, then deletes the row files.

**Why they needed it.** Stated in the docstring: "a 17408^2 fp32 Hessian is 1.2 GB; five of them plus fc's 2.6 GB don't fit next to the model" (capture_dflash2.py:10-11). Roughly 56 GB of disk scratch is traded for VRAM headroom (README.md:79).

**Their numbers.** SMALL_K = 5120; ROWCAP = 250,000 rows per wide module; reduction block 8192 rows; fc Hessian is 25600² (README.md:83).

**llama.cpp — not applicable.** e.values is resized to src1->ne[0] * n_mat (imatrix.cpp:298-299, 374-375) — one float per input column per expert. The 1.2 GB / 2.6 GB fp32 Hessians that force the two-tier scheme simply do not exist in llama.cpp's design. The technique is a workaround for a problem llama.cpp does not have.

**Equivalent here:** none needed — llama.cpp's imatrix is O(K), not O(K^2)

**Evidence (llama.cpp):** `tools/imatrix/imatrix.cpp:296-303` · `tools/imatrix/imatrix.cpp:374-380`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** n/a. The 56 GB of scratch this technique trades for VRAM has no counterpart: llama.cpp's accumulator for the widest tensor here (ffn_down at 17,408) is 17,408 floats, not 17,408^2.

### Re-exec'ing the process to reclaim GPU memory an in-process vLLM engine will not release
**Where (theirs):** `drafter/capture_dflash2.py:160-163` · `drafter/capture_dflash2.py:186-189` · `drafter/README.md:121-122`

**What it does.** After capture, the small Hessians and a `rows_meta.json` are written to disk and the script replaces itself with a fresh interpreter running the same file with `--reduce-only`, which then does the wide-Hessian reduction on an empty GPU.

**Mechanism.** `os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), "--reduce-only"])`, with `__main__` dispatching on that flag to `reduce_wide()` instead of `main()`.

**Why they needed it.** "an in-process vLLM engine does not give its GPU memory back on `del llm`; the Hessian reduction re-execs the process" (README.md:121-122). The 25600² fp32 fc Hessian is 2.6 GB and cannot be built next to a still-resident 27B model plus KV pool.

**llama.cpp — not applicable.** llama-imatrix and llama-quantize are separate binaries and quantisation runs on CPU, so there is never a model resident on the GPU while a reduction needs VRAM. The failure this works around (an engine that does not free on `del`) has no llama.cpp counterpart.

**Equivalent here:** none needed — quantisation and imatrix capture are already separate processes

**Evidence (llama.cpp):** `tools/quantize/quantize.cpp:33-76` · `tools/imatrix/imatrix.cpp:63`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** n/a

### Locating the drafter module by breadth-first search over the runner's attribute graph
**Where (theirs):** `drafter/capture_dflash2.py:37-56` · `drafter/capture_dflash2.py:73-78`

**What it does.** Rather than hard-coding a path through vLLM internals, the script walks down from `llm.llm_engine.engine_core[.engine_core].model_executor.driver_worker.worker.model_runner` and BFSes over `__dict__` attributes to depth 4 looking for an `nn.Module` whose class name contains "DFlash" and which has a `.model` attribute, then prints where it found it.

**Mechanism.** A queue of `(obj, path, depth)` with an `id()`-based visited set; `nn.Module` instances that are not the target are not descended into (line 47-48), which prunes the entire parameter tree. Asserts the drafter was found before installing any hooks.

**Why they needed it.** The attribute path to the draft model differs between vLLM builds and engine-core wrappings (note the `getattr(core, "engine_core", core)` fallback at line 74); a hard-coded path silently AttributeErrors or, worse, finds the wrong module.

**llama.cpp — not applicable.** The draft model and context are explicit fields created at common/speculative.cpp:2464-2482 and its tensors are named members of llama_model. There is no attribute graph to search and no version-dependent path to guess.

**Equivalent here:** typed access — llama_model / llama_context are C structs with named fields

**Evidence (llama.cpp):** `src/models/dflash.cpp:97-151` · `common/speculative.cpp:2464-2482`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** n/a

### Capturing the fused context-KV precompute's input as a separate 'ctx_kv' Hessian by monkeypatching _project_context_kv
**Where (theirs):** `drafter/capture_dflash2.py:113-133` · `drafter/quant_dflash2.py:66-74` · `drafter/README.md:95-101`

**What it does.** DFlash2's k/v weights are applied to two different input populations: the ordinary query rows through `qkv_proj.forward`, and the context rows through a fused precompute that bypasses `forward` entirely. This captures the second population separately, so the k/v quantisation can be calibrated on the blend.

**Mechanism.** Capture: `dm._project_context_kv` is replaced by a wrapper that reproduces the precompute's own normalisation — `ops.rms_norm(normed, context_states, dm._hidden_norm_weight, dm._rms_norm_eps)` — accumulates that into a 5120² Hessian keyed `"ctx_kv"`, then calls the original. Quantise: in quant_dflash2.py, for any weight ending `.k_proj`/`.v_proj`, if `"ctx_kv" in HS` the two Hessians are combined by row count, `H = (nq*H + nc*Hc)/(nq+nc)`, while `q_proj` keeps the pure query Hessian — valid because "GPTQ is row-independent given H".

**Why they needed it.** On paper this is the correct calibration, and the code says so: "the k/v rows are also applied to the context rows by the fused precompute, so on paper this is the right calibration" (README.md:96-97).

**Their numbers.** **It measured 7% worse.** Greedy acceptance 3.34 → 3.12 tokens per step, 126 → 118 tok/s. "It looked equal on a single default-sampling run, which is how it nearly shipped; greedy is the reproducible signal here (four repeats land within 1.5 tok/s, step counts identical)." The shipped drafter is built from a Hessian file with the `ctx_kv` key removed; the code path is kept and warns in-line (README.md:95-101, quant_dflash2.py:70-71).

**llama.cpp — not applicable.** llama.cpp has no ctx_kv Hessian and no Hessian at all, so there is nothing to blend or to remove. I marked it idea-only rather than absent-and-impossible because the negative result is the payload: it is a documented instance of the exact 'believable number instead of a failure' shape this repo tracks, from an independent stack.

**Equivalent here:** none

**Evidence (llama.cpp):** `src/models/dflash.cpp:125-151` · `common/speculative.cpp:1036`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** n/a as a capability. The lesson is the transferable part and it is this project's own north star restated by a different team: a calibration change that was correct on paper measured 7% worse (3.34 -> 3.12 tok/step), looked equal on a single default-sampling run, and nearly shipped. Four greedy repeats within 1.5 tok/s is what caught it.

### Fast-variant assembly by hardlink with a fail-fast tokenizer check
**Where (theirs):** `prepare/fetch_fast_variant.py:106-131` · `prepare/fetch_fast_variant.py:115-120`

**What it does.** Builds `models/Qwen3.8-27B-W4A16-AutoRound-fast` by hardlinking shards 1-6 and copying tokenizer/template files from the base dir, then downloading only shard 7, `model_extra_tensors.safetensors`, `mtp_draft_vocab_ids.pt`, `config.json` and the index from `syvai/qwen3.8-27b-3090-fast-variant` (~1 GB).

**Mechanism.** `os.link` for `model-0000*.safetensors` except `model-00007-of-00007.safetensors`; `snapshot_download` with an `allow_patterns` list of exactly the five changed files; `os.remove(dst)` before each copy so a pre-existing hardlink is never written through. An assert on `tokenizer.json` guards the result.

**Why they needed it.** The assert carries its own explanation: "transformers treats a missing tokenizer as an empty vocabulary instead of an error, and vLLM only notices much later, as 'ReasoningConfig: failed to tokenize reasoning strings'" (lines 115-117) — a failure that surfaces thousands of lines away from its cause.

**Their numbers.** ~1 GB downloaded instead of 19.5 GB; the fast variant is "worth ~15% in single-user mode" (prepare/README.md:27-29).

**llama.cpp — not applicable.** The failure being guarded — transformers treating a missing tokenizer.json as an empty vocabulary rather than an error — cannot happen with GGUF, where vocab tensors and KV are part of the same file and a missing vocab fails at load. The hardlink assembly is covered by #13.

**Equivalent here:** none — GGUF carries the tokenizer inside the file

**Evidence (llama.cpp):** `src/llama-model-loader.cpp:723-785` · `src/llama-vocab.cpp:3021`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** n/a

### In-place scripts that back up exactly what they rewrite, and a verifier that names the missing step
**Where (theirs):** `prepare/README.md:32-36` · `prepare/quant_lm_head.py:59-62` · `prepare/quant_mtp.py:73-79` · `prepare/build_draft_vocab.py:115-116`

**What it does.** Each preparation script copies the file it is about to rewrite next to the original with a step-specific suffix — `.bak` (lm_head shard), `.bak_embed` (embed shard), `.bak-mtp` (extra tensors, index, config), `.bak-quant` (index, config), `.bak-draft` (extra tensors) — and `bash verify.sh --no-server` checks the model dir against every step and names the script to run for whatever is missing.

**Mechanism.** Plain `shutil.copy` before `save_file`, with distinct suffixes per script so two scripts touching one shard cannot clobber each other's pristine copy. Those backups are load-bearing downstream, not just insurance: train_mtp.py reads bf16 embed and lm_head out of them (train_mtp.py:208-214), requant_mtp_gptq.py reads bf16 MTP weights out of `.bak-mtp` (requant_mtp_gptq.py:153), gptq_lm_head.py reads bf16 lm_head out of `<shard>.bak` (gptq_lm_head.py:115-116), and export_mtp.py restores `config.json.bak-mtp` and `model.safetensors.index.json.bak-mtp` as its starting point (export_mtp.py:34-35).

**Why they needed it.** "Each in-place script backs up what it rewrites next to the original (`.bak*`), so a step can be undone without re-downloading 19.5 GB" (prepare/README.md:33-35). The whole rebuild pipeline is only re-runnable because the bf16 originals survive on disk.

**llama.cpp — not applicable.** Every backup suffix in the syv-ref pipeline exists because its scripts mutate a checkpoint directory in place and later steps read the pristine originals back out. llama.cpp has no in-place mutation path at all — there is no code from any CLI flag to a change of a tensor's ggml_type in an existing file — so the whole discipline is unnecessary rather than absent.

**Equivalent here:** none needed — llama-quantize reads one file and writes another

**Evidence (llama.cpp):** `tools/quantize/quantize.cpp:33-76` · `src/llama-quant.cpp:1234-1237`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** n/a

### Decode-time vs prefill-time hidden states are interchangeable for training
**Where (theirs):** `drafter/README.md:111-112`

**What it does.** Records a checked assumption behind capture.py: the hidden states captured by replaying whole sequences as prefill differ from the ones a decode step would produce, and the difference was measured rather than assumed away.

**Mechanism.** capture.py replays each sequence with `max_tokens=1, temperature=0.0` so essentially all rows come from prefill; the drafter at serving time consumes decode-produced states.

**Why they needed it.** If the two differed materially, the entire 74 GB capture — and with it the lm_head Hessian and the MTP training set — would be calibrated on the wrong distribution.

**Their numbers.** "Decode-time hidden states differ from prefill ones by ~0.9% (fp16 recurrent state); training on either gives the same drafter."

**llama.cpp — not applicable.** syv-ref measured the prefill/decode difference at ~0.9% rather than assuming it away, which is the right instinct. There is nothing in llama.cpp to apply it to, since llama.cpp captures no activations for training. I marked it idea-only because the analogous unmeasured assumption in llama.cpp is real and named.

**Equivalent here:** none — no capture, no training

**Evidence (llama.cpp):** `src/llama-context.cpp:34-43` · `src/llama-context.cpp:504-551` · `src/llama-context.cpp:595`

**Effort:** n/a · **Transferable:** idea-only

**Worth on a 4070 SUPER 12 GB:** n/a as a capability. The transferable half is the habit: llama.cpp has its own unchecked shape assumption of exactly this shape — the -fa auto probe decides Flash Attention on a graph with n_tokens_per_seq = 1 (llama-context.cpp:42) and never sees the prompt-processing shape, then clears auto_fa so it never re-runs (556). Whether that decision holds at n_ubatch is assumed, not measured.

### Deriving the dense KV shape from weight_packed instead of trusting weight_shape
**Where (theirs):** `drafter/README.md:122-124`

**What it does.** Notes that on a fused `qkv_proj`, the `weight_shape` parameter only records the shape of the last shard loaded into it, so any code needing the dense k/v row count must compute it from `weight_packed` and `input_size`.

**Mechanism.** "The fused `qkv_proj.weight_shape` parameter only holds the last-loaded shard's shape — derive the dense shape from `weight_packed`/`input_size` (the backport's `_dense_kv_rows` does)."

**Why they needed it.** Listed under "Notes that cost time". A silently wrong row count on a fused module yields a Hessian or a slice of the wrong width — the class of bug that produces a working but degraded checkpoint.

**llama.cpp — not applicable.** The bug is a compressed-tensors artifact: a fused module's weight_shape parameter records only the last shard loaded into it. GGUF has no shard-into-fused-parameter step — blk.N.attn_qkv.weight is one tensor with one ne[], read directly by the loader. The nearest llama.cpp hazard is the opposite one and it is already guarded loudly: an imatrix whose length disagrees with tensor->ne[0] throws rather than silently proceeding (llama-quant.cpp:1211-1218).

**Equivalent here:** none needed — every GGUF tensor carries its own ne[] and the fused attn_qkv shape is explicit

**Evidence (llama.cpp):** `src/llama-model-loader.cpp:723-775` · `src/llama-quant.cpp:1211-1218`

**Effort:** n/a · **Transferable:** no

**Worth on a 4070 SUPER 12 GB:** n/a
