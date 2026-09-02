# Speed sweep for Qwen3.8-27B NVFP4 — consolidated candidates (2026-09-01)

Hardware: RTX 5060 Ti 16GB (sm_120) + RTX 4070 SUPER 12GB (sm_89), Windows, asymmetric.
Baseline (measured): NVFP4 + native MTP/ngram ~39.4/42.6/42.6 tok/s; served ctx 147K (deep 200K).
Source sweep: 5 parallel sub-agent tasks (deleg_d05ffced). Engine-agnostic — we are NOT limited to llama.cpp.

## HIGH-LEVEL DECISION (from sweep, not assumption)

1. **vLLM/SGLang = effectively OUT on this box.** NVFP4/FP4 kernels are Blackwell sm_120-only → RTX 4070 SUPER (sm_89) has no NVFP4 fast path. No native Windows vLLM (needs WSL2/Docker). TP=2 across 16+12GB different-VRAM OOMs. DFlash + TP2 crashes on AWQ (TP=1 only). If ever revisited: Linux/WSL2 + single 5060 Ti only, drop TP.
2. **Two real engine options remain: llama.cpp (stay) and ExLlama3 fork (spike).** Only published head-to-head (Qwen3.6-27B, default mode, unknown hw) showed llama.cpp ~12% faster than ExLlamaV3 — but EXL3 3.5bpw fits a SINGLE 16GB card (14.2GB) and enables DFlash2. Must benchmark on our cards, not trust tables.
3. **Tensor-split constraint: the claim was TRUE and is now STALE — and there is a commit for it.** llama.cpp before **`ggml-org/llama.cpp#23792` (build b9455)** refused a quantized KV cache under `--split-mode tensor` outright, in `llama_init_from_model`: `"simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not implemented"`. Read out of Unsloth Studio's own source, `studio/backend/core/inference/llama_cpp.py:15577-15586`, which keeps the string only to fail fast on an older binary and records that *"Unsloth dropped its own pre-emptive gate once #23792 shipped"*. Our 10499 and Studio's 10715 are both far past b9455, which is why `-sm tensor` + `q4_0`/`q4_0` ran 9 boots here without complaint. **Removed as a blocker, with provenance rather than only a result.**
4. **Asymmetric KV (`-ctk q8_0 -ctv q4_0`) is DEPENDENT on the FA_ALL_QUANTS build.** Without it, `fattn.cu:442-445` `#ifndef GGML_CUDA_FA_ALL_QUANTS → if (K->type != V->type) return BEST_FATTN_KERNEL_NONE` kills the K≠V fast kernel (recorded: 29× slower prefill, 44% larger cache, hung 65 min at 128K). Opens only after Tier-1 item 1. Not a standalone task.
5. **`-ctk/-ctv nvfp4` does NOT exist** in this build — `arg.cpp kv_cache_types` accepts only F32/F16/BF16/Q8_0/Q4_0/Q4_1/IQ4_NL/Q5_0/Q5_1. Removed from queue.
6. ~~**`GGML_OP_OFFLOAD_MIN_BATCH`** retains candidate status for MoE offload~~ — **the knob is real, the premise is not.** It IS read by the CUDA backend (`ggml-cuda.cu:5501`, default 32; an earlier note here calling it cann-only was wrong). But `ggml-backend.cpp:959` consults it **only for weights already in a host buffer**, and there are none: **Qwen3.8-27B as we serve it is NOT an MoE.** The GGUF header gives 1,202 tensors and **zero** containing `exps`, no `expert_count`, `block_count 65` with `full_attention_interval 4` — 48 `ssm_*` blocks against 17 attention, dense FFN throughout. **CLOSED by measurement**, `results/cpumoe-147456.jsonl`: twelve rows at ctx 147,456, 45.5–45.8 tok/s across all four arms, every comparison inconsistent in sign, and `free_after` **2,521 MiB in all twelve** while `--cpu-moe` claims to move every MoE weight to the host.

---

## GROUP A — llama.cpp (current engine) — carry-forward candidates
[confidence: measured/plausible/unverified]

- ~~**MTP n-max/p-min high sweep**~~ — **MEASURED HERE 2026-09-01, all three claims fail.** Four arms, rotated, ctx 16,384, coding prompt: `n3` **67.86** · `n8` **56.80** · `n6 + p-min 0.6` **31.30** · `n16 + p-min 0.8` **does not boot** — the server exits during load with `0xC0000409`, reproducibly, all three rounds. `n8` drafts 19,143 against `n3`'s 10,467 and its acceptance falls to **37.1 %** from 65.9 %, so the extra drafting is close to pure waste. **`n16` and `p-min 0.8` were set together and are confounded** — which of the two kills the load is not established. [was: measured, discussion 25198]
- ~~**--backend-sampling**~~ — **MEASURED HERE: 63.11 default-on against 63.18 explicitly off**, identical draft counts and identical output hashes. The flag is `(default: enabled)`, so the served profile always had it on, and under `-sm tensor` the server logs the CPU fallback on every boot — llama.cpp issue #27467. **The fallback is complete and costs nothing**; turning it off only silences the warning. That the fallback happens at all was already on record in [results README](../results/README.md) line 112. [measured here]
- ~~**Asymmetric KV `-ctk q8_0 -ctv q4_0`** — strictly better precision~~ **— not on any binary we have.** See High-level item 4: `fattn.cu:442` drops every K≠V pair when `GGML_CUDA_FA_ALL_QUANTS` is off, and it is off in all four local builds. **This is a different gate from the SPLIT_MODE_TENSOR one in item 3** and was not lifted by #23792. Our own [results 03](../results/03-memory-and-kv.md) already measured the consequence: prefill **29× slower**, cache 44 % larger, one arm hung 65 minutes at 128K. Gated behind Tier-1 item 1. [refuted here]
- **Rebuild `-DCMAKE_CUDA_ARCHITECTURES='120a;89-real'` + `-DGGML_CUDA_FA_ALL_QUANTS=ON`** — native arch ~+19% gen per blog; ALL_QUANTS adds FA kernels for NVFP4/q5/q6 KV. [measured/plausible]
- **CUDA Graphs** (`GGML_CUDA_GRAPHS`, on by default bs=1) — up to 1.2x decode, small for 27B. [measured]
- **Draft on secondary GPU** `--spec-draft-device cuda:1` — run draft concurrently with target decode; local repo has DFlash2 support. [plausible]
- **`GGML_OP_OFFLOAD_MIN_BATCH`** (MoE offload threshold) — **CUDA-verified**, `ggml-cuda.cu:5501` in our tree and `:5534` in v0.3.0-dev, default **32**; it is read by the cann, metal, sycl and vulkan backends too. **An earlier note here calling it cann-only was mine and was wrong** — the grep behind it collapsed one line per matched token and kept only the first file. Candidate stands. [plausible, source-verified]
- ~~**`--n-cpu-moe`** — offload sparse MoE experts to CPU~~ — **REFUTED HERE.** `common/arg.cpp:2728` pushes `ffn_*_exps` buffer-type overrides, which match no tensor in this artifact. Measured inert at the served depth; VRAM does not move by one MiB under `--cpu-moe`. **And it could not have paid even with experts:** the 5060 Ti is on PCIe gen4 x4, about 7.9 GB/s, against ~14.9 GB of weights — one crossing per token is ~0.5 tok/s. Host DDR5 at 108 GB/s is not the constraint, the slot is. [refuted, `results/cpumoe-147456.jsonl`]
- **`--slot-save-path` / `--n-parallel 2-4`** — skip re-prefill of fixed prefixes. Only if workload has stable prefix. [plausible]
- **`GGML_CUDA_DEVICES`, `GGML_CUDA_ALLREDUCE`, `GGML_CUDA_GRAPH_OPT`** — code-verified envs (read in local source); all plausible, no public bench. Sweep cheap.
- **nvfp4 KV cache type `-ctk nvfp4`** — DOES NOT EXIST in this binary (accepted: F32/F16/BF16/Q8_0/Q4_0/Q4_1/IQ4_NL/Q5_0/Q5_1). Remove from plan. [rejected — source-verified]
- **Asymmetric KV `-ctk q8_0 -ctv q4_0`** — promising but gated behind the FA_ALL_QUANTS build (see High-level item 4). Only after Tier-1 item 1. [measured earlier: unusable without it]

## GROUP A (already measured by us / closed):
- -ub 1024 (+10.1% prefill), -fa on, -ctk/-ctv q4_0 (q8_0≈0 at shallow), -sm tensor (+31% at 147K).
- DONE by Opus 5 (ctx 16,384): p-min 0.7 (drop), spec-type order swap (no effect), n-max 3 wins, backend-sampling tie (default-on), build A/B +2.58% decode, PR #27140 (zero, our prefill ~990), CUDA_SCALE_LAUNCH_QUEUES (no effect), P2P (not run, no canary).

## GROUP A — what Unsloth Studio's own history and source settle (read 2026-09-01)

Studio is the nearest thing to an independent operator on this exact machine, so its
logs answer several questions for free. Read-only; `%USERPROFILE%\.unsloth` was not written to.

- **Studio has NEVER run DFlash2 under a tensor split.** Cross-tabulating every
  `Starting llama-server:` line by split mode and spec type: `draft-dflash` appears **12 times,
  every one of them with no `--split-mode`** (so layer). Under tensor it runs `draft-mtp` (36),
  `ngram-mod,draft-mtp` (7), `ngram-mod` (1) and bare (21) — the same decoders we serve.
  **Our "DFlash2 aborts under `-sm tensor` without the mirror patch" is not contradicted by
  Studio working; Studio simply never asks for that combination.**
- **And it would hide the abort if it did.** Studio carries two latches:
  `_is_tensor_split_assert` (`llama_cpp.py:15596`) matches the **#6415 split-axis warmup assert** —
  our abort — and `_TENSOR_QUANT_KV_UNSUPPORTED_MARKER` (`:15583`) matches the pre-#23792 KV error.
  Both **latch and downgrade to layer split rather than retry**, so from the UI the model "just works"
  and the tensor arm is silently abandoned.
- **The downgrades actually recorded on this machine were budget-driven, not abort-driven** — ten
  identical events: *"Tensor parallelism requested but the pooled VRAM budget cannot hold the weights,
  MTP reserve, and per-device compute buffers; falling back to layer split."*
- **Asymmetric KV: never launched.** All 177 explicit cache-type arguments in Studio's history are
  `--cache-type-k q4_0 --cache-type-v q4_0`. Its source knows the feature — it prices the dequant
  scratch off the lighter axis *"since ggml-org/llama.cpp#23792 an asymmetric `-ctk q4_0 -ctv f16`
  reaches the child"* — but never used it here. **So Studio cannot tell us what its
  `GGML_CUDA_FA_ALL_QUANTS` is set to; only a build of our own can (issue #43).**
- **Studio's build has moved to `10715` (`92cedc867`)**, from the `10679` our probe row records.
  The 2026-08-30 finding that their build still needs the mirror patch was measured on 10679 and
  **has not been re-checked on 10715.**
- **Two flags Studio sets that we measured as expensive:** `--cache-ram 0` and `--ctx-checkpoints 0`.
  On a hybrid model the second is SETTLED here as *do not turn off* — llama.cpp abandons the prompt
  and re-prefills, 51.6 s per request in `serve-20260829-125227.log`. Studio currently runs the
  Ornith-1.5-35B **mamba hybrid** with both set.

---

## GROUP A — client-side / context (the biggest proven lever, issue #55)
- **Tool schema pruning + lazy loading** — PROVEN: 35.2→45.6 tok/s, prefill 18.6s→554ms (~14x). Extend: retrieval-based schema selection, plain-English tool descriptions, strip credential/metadata. [high]
- **Prompt-caching discipline** — static shared content (system, tool schema, MCP builtin share) fixed at position 0; move per-user data out of prefix → same-slot KV reuse (TTFT 1.23-1.31x). Directly un-blocks the 17,881-token read split. [high]
- **Same-slot KV reuse / `cache_prompt`** + **persistent KV to disk** (CachyLLama fork ~7x). [high/medium]
- **Context trimming / obsolete-token drop** (`--ctx-shift`, SWA checkpoints) — trim stale middleware turns before prefill. [medium]
- **Prompt compression (LLMLingua-class)** for the 17,881-token read. [medium]
- **Greedy / temperature 0** on deterministic tasks — sampler overhead small but free. [medium]
- **KV q8_0** — halve KV memory, deeper 200K NVFP4. [high]

---

## GROUP B — ExLlama3 / EXL3 (alternative engine spike — context/parse verified)
Fork: https://github.com/MiaAI-Lab/exllamav3  (DBFlash2/DSpark/NVFP4-KV added; upstream x86-only → fork adds GB10/aarch64)

- **EXL3 3.5bpw = 14.2GB** → fits ONE 16GB card; drops tensor-split entirely. NVFP4 KV + built-in MTP head. [HIGH size-verified]
- **Multi-GPU asymmetric IS supported on x86** — `model.load(use_per_device=[14.2,6.0], tensor_p=True, tp_backend='native')` or `--gpu_split 14.2,6 --tensor_parallel -tpb native`; `native` needs no NCCL (matters on Windows). serve_openai.py's `-gs` is mislabeled int (single device) — split needs library API/TabbyAPI. [HIGH source-verified]
- **DFlash2 draft EXL3 5.0bpw (1.4GB)** — fork-only drafting; on GB10: +33% over bf16 draft, 47.5 tok/s HumanEval; ~+15% over MTP. Serve: `serve_openai.py -m <3.5bpw> -dm <DFlash2-5.0bpw> -cq nvfp4 -cs 262144`. [MEDIUM — GB10 only]
- **NVFP4 KV cache** (~4.5 bit/elem, ~18KB/token+MTP) — online dequant in Triton paged-attn; E2M1 cos-sim 0.99995 lossless-in-noise. [HIGH code+card]
- **Dynamic draft length + draft-skip EMA** `-dds -dskip` — cut draft compute at low acceptance. [HIGH implemented]
- **CAVEAT:** code prose (greedy) 40-43 tok/s on GB10 ≈ our current ~40; no consumer-GPU Qwen3.8 EXL3 benchmark published. The one head-to-head (Qwen3.6, default) favored llama.cpp ~12%. **Proof needed on our cards.**

---

## GROUP C — vLLM/SGLang (deprioritized for THIS box)
- NVFP4 sm_120-only; 4070 SUPER has no fast path. No native Windows. TP=2 across 16+12 OOMs. DFlash+TP2 crash.
- If switched to Linux/WSL2 + single 5060 Ti (drop 4070S): MTP single-GPU llama.cpp 1.75-2.3x, or vLLM DFlash (needs PR branch #40898). SGLang >200 tok/s on single 5090 NVFP4 (single-GPU, not ours).
- Reference: RTX 5090 llamachpu MTP 122 tok/s; 4x5060Ti Q8 TP4 MTP 52.2 tok/s.

---

## PROPOSED EXPERIMENT QUEUE (paired, at served depth 147K, correctness-gated)
Tier 1 (stay-on-llama.cpp, cheap, additive):
1. Rebuild native dual-arch `120a;89-real` + `GGML_CUDA_FA_ALL_QUANTS=ON` (A/B vs current build). **DO FIRST** — unlocks #2 via the K≠V gate, answers open issue #43 (flag was pre-"decided" on structure, not evidence). This is the corrected priority.
2. Asymmetric KV `-ctk q8_0 -ctv q4_0` at 147K (acceptance + VRAM) — **dependent on #1**; not standalone.
3. ~~MTP high n-max/p-min sweep on a CODING workload~~ — **DONE 2026-09-01, `n3` keeps it.** 67.86 / 56.80 / 31.30 for n3 / n8 / n6+p0.6, and n16+p0.8 does not boot. **Remaining question:** n16 and p-min 0.8 were confounded, so one arm at `n16` alone would say whether the crash is the draft depth or the threshold.
4. Client-side (biggest proven lever): issue #55 — REQUIRES an attended live Claude Code session via llama-tap to split the 17,881-token read; cannot run unattended. Hold.
5. ~~`--n-cpu-moe` + `GGML_OP_OFFLOAD_MIN_BATCH` sweep for MoE headroom~~ — **DONE 2026-09-01 and inert**, the artifact has no experts (above). The `-t` half survived as its own arm set and is measured separately.

Tier 2 (alternative engine spike — isolated, non-production):
6. ExLlama3 fork PoC on single 5060 Ti: load EXL3 3.5bpw + NVFP4 KV + MTP, measure tok/s at 147K on our prompt corpus. Compare vs llama.cpp paired baseline. Do NOT touch production llama.cpp session while GPU in use.
7. If ExLlama3 beats llama.cpp ≥ threshold: optional dual-GPU `use_per_device` split via library API.

Tier 3 (only if moving to Linux/WSL2 or dropping 4070S):
8. vLLM/SGLang DFlash/DFlash2 single-GPU on 5060 Ti.

## Stay-away (from sweep)
- TensorRT-LLM on Windows (Linux-only WO).
- DFlash+TP2 (crashes, TP=1 only).
- q4_0 KV at long ctx (collapses: -92% prefill@64K on DGX Spark) — we already use q4_0, but long-context quality/collapse needs its own check at 147K.

---

## Corrections from Opus 5 (2026-09-01) — recorded against the doc's earlier claims

These update earlier entries; the doc above already carries the corrected numbers inline, this is the changelog.

- **`--n-cpu-moe` is INERT here — the artifact has NO MoE experts.** GGUF has 1,202 tensors, zero `exps`, `block_count 65` with `full_attention_interval 4` = 48 ssm blocks vs 17 attention → **Qwen3.8-27B as served here is dense-hybrid, not A3B-MoE.** `--cpu-moe` measured 45.5–45.8 tok/s across 12 rows at 147,456, all signs inconsistent; `free_after` 2,521 MiB in all twelve. Closed by `results/cpumoe-147456.jsonl`.
- **`GGML_OP_OFFLOAD_MIN_BATCH` premise is also dead** for the same reason — `ggml-backend.cpp:959` consults it only for weights already in a host buffer, and there are none. (The earlier claim it speaks only via the 5060 Ti's PCIe gen4 x4 ≈ 7.9 GB/s slot ≈ 0.5 tok/s per full crossing also kills any expert-offload fantasy.)
- **MTP n-max sweep: `n3` keeps it.** n3 67.86 / n8 56.80 / n6+p-min0.6 31.30; **n16+p-min0.8 does not boot** (0xC0000409). n16+p0.8 were confounded — one arm at `n16` alone would isolate draft-depth vs threshold crash.
- **DFlash2 under `-sm tensor`:** Studio has never run it (12 `draft-dflash`, all layer). Two latches silently downgrade to layer split, so Studio "working" does not contradict our abort. Not re-checked on their newer 10715 build.

## External lead — abliterated NVFP4-MTP candidate (2026-09-01, via HF MCP)
- **`felippeburk/Huihui-Qwen3.8-27B-Abliterated-NVFP4-MTP-GGUF`** — GGUF + NVFP4 + MTP, llama.cpp-native, uncensored (abliterated). Structure matches our current stack (GGUF/NVFP4/MTP/-sm tensor) → drop-in artifact swap candidate if the team wants uncensored.
- Base chain: `sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4` (vLLM safetensors, 55.3K dl / 51 likes — popular) → GGUF wrapper.
- Variants: `renketong/...-abliterated-NVFP4-GGUF` (GGUF + vision/multimodal), `HangGlidersRule/Darkstar-...-NVFP4-Mixed-FP8` (ModelOpt W4A16), `Vtuber-plan` (transformers, reasoning/agent), `windowsxp811203` (NVFP4 safetensors).
- **Caveat:** uncensored-goal swap, NOT a speed lever. Must A/B quality (MTP acceptance / output contract / MRCR) before default, same as any artifact swap. Do not assume faster than current NVFP4-MTP. Check files/bpw tiers + 16GB fit (via hub_repo_details / hf_fs) before adopting.

## External lead — fixed Qwen chat template (2026-09-01, client-side / request-shape, via HF MCP)
- **`froggeric/Qwen-Fixed-Chat-Templates`** (v22.4, 1544 likes) — single `chat_template.jinja` drop-in that fixes official Qwen 3.5/3.6/3.8 template bugs: token waste, empty-think poisoning, tool-JSON crashes, agentic stalls. Engine-agnostic (llama.cpp / LM Studio / vLLM / MLX).
- **Relevance to us:** it maps onto our biggest lever — client-side / request shape (issue #55 family). Key wins: default reasoning `xhigh`→`medium` (0 injected tokens, KV parity), full `enable_thinking=false` non-reasoning fast path, canonical XML tool-arg + stringified-JSON fix (tool calling), `--reasoning-preserve` alias for llama.cpp.
- **Caveat:** it changes output/output contract (template-level token shaping, not a decode-speed flag) → must A/B quality, not just tok/s, before adopting. Verify against whatever template we currently serve.

## Real-session baseline (2026-09-01, Opus 5 measured — build 10729, first real-work data)
This is the FIRST number in this campaign that comes from real agent traffic, not a corpus or short boot. Treat it as the ground truth to compare all future A/B against. It differs from arena (46) and short boot (49.6) for a reason — deep context + two concurrent agents keep the GPU at 100% busy (busy 130 min of 126 wall).

```
build    10729 (458681e1d)      spec  ngram-mod + draft-mtp
requests 241
prefill  1,814,811 token   2,601.0s   697.7 tok/s
decode     187,456 token   5,217.2s    35.9 tok/s
busy                       7,818.2s   (130 min)  <- GPU never idle (2 agents)
wall                       7,561.0s   (126 min)
```

- **decode 35.9 tok/s** — real depth + contention, BELOW arena 46 / boot 49.6.
- **n-max 64 works on real traffic (closes the old doubt):** ngram-mod drafts 1,035, acc 22,063, mean acc len **22.32** vs draft-mtp drafts 62,748, acc 104,114, mean **2.66**. ngram fires 1.6% as often but yields 17.5% of all speculative tokens. This **refutes** worker-q4-dual.ps1:1252-1264's old "#gen drafts = 0 on agent traffic" (Opus 5 updating ledger/comments).
- **The dominant server-side cost is re-prefill, not decode or KV.** Prefill was 41% of server time (3,164s = 53 min of 130), 31 forced full re-prefills (not 23), all invalidated-to-null. Largest single: 167,237 tokens / 258s; 5 were over 150,000.
- **Root cause: GGUF has 48 SSM blocks of 65 → the checkpoint-invalidate path nulls everything instead of keeping the checkpoint nearest the fix point.** `--ctx-checkpoints` is already ON (default 32), never exceeded 9/32. It is NOT a flag problem — it is the invalidate LOGIC. No flag fixes it: `-ctxcp 32`, `-cram 8192`, `--cache-reuse` (= issue #42, GPU side untested), `--swa-full`, `--context-shift` all irrelevant.
- **Opened as issue #70** with the full numbers and 3 questions to close it. This is the single biggest remaining server-side lever (roughly matches issue #49 thread).

### Corrections from this measurement
- Do NOT go change `--ctx-checkpoints` (already on, cap never hit — 9 of 32 max observed).
- Do NOT try to "make checkpoints work" via flags — it must be fixed in the invalidation logic (issue #70).