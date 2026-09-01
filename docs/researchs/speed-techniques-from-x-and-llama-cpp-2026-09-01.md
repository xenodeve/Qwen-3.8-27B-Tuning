# Speed techniques from X and first-party llama.cpp sources

Date: 2026-09-01
Scope: compare public X post `https://x.com/analogalok/status/2088326480669667699` and first-party llama.cpp/Hugging Face material against this repository's measured evidence.

## Executive result

Do not copy the X command as a production profile. The post's strongest reusable ideas are already present in this repository: Q4 KV, native MTP, `n-max 4`, and GPU-resident inference. The highest-value remaining work is a small paired benchmark matrix around settings that are either untested here or have changed with the artifact/build:

1. Check/build the upstream quantized-KV prefill fix from PR #27140.
2. `--spec-draft-p-min 0.7` with native MTP on NVFP4.
3. `--spec-type ngram-mod,draft-mtp` versus `draft-mtp,ngram-mod` on the same NVFP4 request set.
4. `--spec-draft-n-max 2/3/4` at the served context.
5. Current upstream llama.cpp build versus the served build, paired on the same artifact and flags.
6. `--spec-draft-backend-sampling` separately from the other speculative flags.
7. `-t/-tb` and `--poll` only after the above, because they are CPU-side levers and may affect prefill/draft overhead rather than GPU decode.

No production default should change before the paired benchmark and quality/contract checks.

## External technique: X post

The post says it used `Qwen3.8-27B-UD-Q4_K_XL.gguf` on one RTX 4090 24 GB with llama.cpp.

Reported flags:

```text
-c 260000 -ngl 99 -ctv q4_0 -ctk q4_0
```

for the long-context profile, and:

```text
-c 130000 -ngl 99 -ctv q4_0 -ctk q4_0 \
--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.7
```

for the MTP profile.

Reported numbers are approximately 40.7 tok/s without MTP and 60.1 tok/s with MTP at 130K, plus 260K context with Q4 KV at about 40.7 tok/s. These are external claims. The post states a 28K prompt baseline for the context matrix; that is not equivalent to this repository's half-window real-request survival tests.

### What transfers

- Q4 K/V cache is a valid memory lever and is already used here.
- Native MTP is the correct first speculative-decoding path when the model artifact carries the head.
- `n-max 4` is a legitimate candidate, but its value is context- and artifact-dependent.
- Context capacity and decode speed should be treated as separate profiles.

### What does not transfer directly

- One RTX 4090 versus this machine's RTX 5060 Ti + RTX 4070 SUPER tensor split.
- `UD-Q4_K_XL` versus `NVFP4-MTP-VERY-LOW`.
- Their `p-min 0.7` result has not been measured in this repository.
- Their 260K result is not evidence that this machine survives a real 260K request.

## Repository evidence already available

Sources: `docs/results/README.md`, `docs/results/02-decoders.md`, `docs/results/09-hardware.md`, `qwen38-tuning/scripts/worker-q4-dual.ps1`, and `docs/researchs/unsloth-studio-config-2026-08-29.md`.

- Dual `UD-Q4_K_XL` with tensor split is fully resident through 229,376. A 262,144 request is not a safe served profile because real-request OOM has occurred.
- NVFP4-MTP-VERY-LOW has a baked-in MTP head and reaches a measured safe served ceiling of 200,704 after a real long request.
- At context 147,456, NVFP4 + `draft-mtp,ngram-mod` measured 39.4 / 42.6 / 42.6 tok/s against the served Q4 + ngram baseline at 24.9 / 25.7 / 25.7: +63.1% paired and resolved.
- Q4 KV is the current tested choice. Q8 KV was near-null at shallow context; the old deep-context Q8 claim was confounded by an even tensor split and is not a settled result.
- `-sm tensor` is already the measured faster split on this hardware, but is experimental in llama.cpp. On NVFP4 at 147,456 it beat layer split by 31.0% with both arms `66+0` resident.
- `-ub 1024` is already the measured prefill choice for the dual tensor split; decode was flat across 256/512/1024.
- `--spec-draft-p-min` was tested only in an older DFlash2 setup at 0.10 and 0.25 and was null within the floor. That does not answer 0.7 on native MTP/NVFP4.
- `n-match 24` is measured as better than 12 on NVFP4 at 147,456 (+27.1%), but the verdict is artifact- and context-specific.
- DFlash2 is not the next default path: it needs a sidecar and mirror patch, fails with images, and at NVFP4/147,456 is only +4.0% against MTP, below the project floor.
- The largest non-llama.cpp speed lever already measured is request shape: disabling unused tool schemas changed one observed run from 35.20 to 45.64 tok/s and reduced prefill from 18,618 ms to 554 ms. This is client-side, not a server flag.

## First-party llama.cpp findings

Source: `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md`

The current server documentation exposes these relevant levers:

- `-ub/--ubatch-size`: physical batch size; likely prefill-oriented. This repository has already measured 1024 as the dual-GPU prefill winner.
- `-t/--threads` and `-tb/--threads-batch`: CPU generation and batch/prompt threads. Worth a controlled sweep, but not assumed to improve GPU decode.
- `--poll`: CPU wait polling. Possible latency/CPU trade-off, not evidence of a decode win.
- `-fa/--flash-attn`: already enabled in the relevant profiles.
- `-ctk/-ctv`: already q4_0 in the relevant profiles.
- `--spec-draft-type-k/v`: draft KV precision. Relevant for an external draft model; native MTP still deserves an explicit check only if the server applies it to the native draft path.
- `-sm tensor`: already used and measured faster here; experimental status remains material.
- `--kv-unified`: a candidate for multi-sequence serving, but not expected to improve single-sequence `-np 1` decode without evidence.
- `--backend-sampling`: experimental and not a free speed switch; grammar can disable it. Do not enable for production without output-contract tests.
- `--load-mode`: affects loading/residency/pageout behavior, not an assumed decode improvement.

Source: `https://api.github.com/repos/ggml-org/llama.cpp/releases/latest`

The latest release response available during this research is `v0.3.0`, target commit `c1d0e7a004015f23bc0233470b747b596f29b264`. Its notes include changes to tensor-split state propagation, MTP support/fixes, common fit logic accounting for streams, and ggml changes. This makes a paired build A/B worthwhile, but it is not evidence that the newer build is faster on this hardware. The repository explicitly records that its previous build comparison was invalid because both arms launched the same binary.

## Ranked candidate experiments

### P0 — native MTP `p-min`

Arm A: current NVFP4 served flags.
Arm B: add `--spec-draft-p-min 0.7`.

Why: copied directly from the X post and not yet measured on this artifact/path.
Risk: may reduce useful drafts or alter output behavior; must record acceptance, accepted length, draft calls, and contract/repetition checks.
Expected value: unknown; candidate for a speed win, not a recommendation.

### P0 — MTP/ ngram ordering

Arm A: `draft-mtp,ngram-mod`.
Arm B: `ngram-mod,draft-mtp`.

Why: the local Studio observation uses the reverse order, and llama.cpp's speculative registry/cascade makes order potentially meaningful. Existing rates cannot answer this because earlier runs were not a clean paired order test.
Expected value: unknown; potentially meaningful if the first non-empty drafter wins.

### P0 — `n-max` at served depth

Arm values: 2, 3, 4; keep all other values fixed, including `n-match 24` and context 147,456.

Why: the X post uses 4; Studio used 2; the served profile currently uses 3. Local evidence shows `n-max` changes recurrent-state VRAM, and the best value moves with context.
Expected value: unknown; 4 may win, but memory headroom and long-request survival are gates.

### P1 — build A/B

Compare the served `llama.cpp-blackwell` build with a current upstream build, same artifact, same binary architecture support, same flags, same corpus, rotated order, and enough rounds to observe the local noise floor.

Why: upstream v0.3.0 contains relevant MTP/tensor-split/common-fit changes. Local history says the earlier build comparison was invalid because the harness recorded one binary while launching another.
Expected value: possibly a clean prefill/decode win; no claim until the 2x2 is actually paired.

### P1 — CPU-side generation/batch controls

Sweep `-t` and `-tb` separately: 2, 4, 8, 18; optionally `--poll 0/50/100`.

Why: Studio uses `--threads 2`, while the repository uses 18. This could reduce CPU contention or improve draft/sampling overhead, but may also hurt prompt processing.
Gate: compare prefill and decode separately and monitor GPU utilization; do not infer from wall-clock alone.

### P2 — `--kv-unified` / multi-slot profile

Only if the target is concurrent sessions (`-np > 1`).

Why: Studio sets it, but the local profile uses `-np 1`. It is not a priority for single-conversation decode and may change cache reuse semantics.

### P2 — cache/checkpoint policy

`--cache-ram 0` and `--ctx-checkpoints 0` are not raw decode optimizations. They trade host RAM for re-prefill. Keep the current cache behavior for long-context agent use unless the target workload is short, non-repeating conversations.

## Additional candidates from upstream research

### Q4/Q5 quantized-KV prefill regression fix — P0 build candidate

Upstream PR `https://github.com/ggml-org/llama.cpp/pull/27140` is directly relevant: it reports Qwen3.8-27B on 2x RTX 3090 where `q4_0` prefill was 74 tok/s before the fix and 1182 tok/s after, at the same ~9000-token prompt; q8_0 was already fast. The PR says the CUDA fix needs `GGML_CUDA_FA_ALL_QUANTS=ON` for the additional quantized KV types and reports numerically correct outputs.

This is not a speed flag. It is a source/build candidate and should move ahead of speculative micro-tuning because this repository serves `q4_0` K/V and has a dual-GPU prefill path. The local `C:\AI\llama.cpp` checkout currently reports HEAD `1deefcca395743049c3820ab8f9b15043f3e9446` (2026-08-21), so presence of the fix must be checked by commit/source, not assumed from the build number. Required A/B: same source tree except the fix (or a clean upstream build containing it), same `GGML_CUDA_FA_ALL_QUANTS` setting, same artifact/argv/corpus, and separate prefill/decode measurements.


Upstream PR `https://github.com/ggml-org/llama.cpp/pull/23287` reports about 7% higher speculative throughput on different hardware/model when draft sampling stays on the backend. Candidate flags are:

```text
--spec-draft-backend-sampling
```

The top-level `--backend-sampling` is experimental and can fall back to CPU for unsupported samplers/operations. This repository has not established that the native MTP path, tensor split, grammar, and current sampler chain all remain correct with it. Benchmark it separately with output-contract and identifier-preservation checks. Do not combine it with p-min/order/n-max in the first run.

### CUDA launch queues — P2 candidate

Upstream build documentation mentions `CUDA_SCALE_LAUNCH_QUEUES=4x` as a possible benefit for multi-GPU pipeline workloads. It is more likely to affect prompt processing/launch stalls than steady-state single-token decode. Test as an environment-only A/B with identical argv and verify GPU utilization and prefill separately.

### CUDA P2P — P2/high-risk candidate

`GGML_CUDA_P2P=1` may reduce host-mediated GPU transfers if the motherboard, driver, and topology support it. It is not assumed available on consumer cards and can cause crashes or incorrect output on unsupported systems. First run a P2P diagnostic and a correctness canary; only then run paired throughput. Never enable it in the served launcher based on a speed reading alone.

### Backend/build freshness

The upstream release currently identified is `v0.3.0`, target commit `c1d0e7a004015f23bc0233470b747b596f29b264`. Build A/B must include native `sm_89` and `sm_120a` cubins, Release mode, identical model/argv/corpus, and a check that the intended executable actually launched. This repository has already invalidated one build comparison because the harness recorded a different binary from the one it ran.

## Proposed benchmark protocol

Use the existing paired harness and a frozen real-code corpus.

- Artifact: `Qwen3.8-27B-NVFP4-MTP-VERY-LOW`.
- Context: 147,456 first; repeat promising arms at 200,704 only if the budget guard permits.
- Split: computed `-sm tensor`; verify `66+0` and no host spill.
- KV: `q4_0` K and V.
- Ubatch: 1024.
- MTP + ngram: keep `n-match 24` fixed.
- Arms: baseline, p-min 0.7, order reversed, n-max 2/3/4, then CPU thread candidates.
- Rotate arm order across at least three paired rounds.
- Record: prefill tok/s, decode tok/s, first-token latency, total latency, acceptance, accepted length, draft calls, free VRAM per device, residency, output contract, prompt-copy/repetition metrics.
- Reject any row with spill, timeout, prompt-copy guard failure, wrong binary, or a request that does not survive the intended context.

## Cross-source findings from the completed sweep

### Reddit

- [`r/LocalLLaMA` mixed laptop/eGPU benchmark](https://www.reddit.com/r/LocalLLaMA/comments/1w1v6c7/benchmarking_qwen3827b_at_q4q5q6_on_a_laptop_gpu/) used CUDA llama.cpp, layer split, Q8 K/V, ubatch 256, and a 5070 Ti Laptop 12GB + 5060 Ti 16GB. Q4_K_XL measured 22.30 tok/s decode; Q5 with MTP reportedly rose from about 19 to 32–38 tok/s at 55–70% acceptance. This is the closest hardware-shaped lead but uses Thunderbolt/eGPU and a different split/artifact; use only as supporting evidence for testing MTP and real-request OOM.
- [`r/LocalLLaMA` RTX PRO 4000 Blackwell](https://www.reddit.com/r/LocalLLaMA/comments/1w10qem/qwen3827b_on_a_24gb_rtx_pro_4000_blackwell_128k/) reports MTP3 on a 24GB Blackwell card, but uses a nonstandard NInfer fork and is not a llama.cpp reproduction.
- Reddit search through the direct JSON endpoint was blocked with HTTP 403 in this environment. The two posts above came from public search/index paths; Reddit coverage is therefore not exhaustive.

### Hugging Face

- [Unsloth discussion #72](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/72) reports 2x RTX 5060 Ti 16GB, tensor split, Q8 K/V, MTP n=2, about 800 prompt tok/s and 30 decode tok/s at 215,040 context. It is a useful same-GPU-family lead, but not the user's asymmetric 16GB+12GB pair.
- [Unsloth discussion #87](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/87) reports on an RTX 2080 Ti that at 32K context MTP n=2 (39.40 tok/s) beat n=1 (36.64) and n=3 (37.99) using q4_0 KV. It supports testing n=2/3/4 at each served depth, not choosing n=2 globally.
- [Qwen discussion #182](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/182) reports MRCR retrieval quality across BF16/FP8/TurboQuant KV, not llama.cpp throughput. FP8 was more stable than aggressive 3-bit at long context; TurboQuant values must not be equated with llama.cpp q4_0.
- The NVFP4-MTP collection has multiple tiers with an embedded MTP head. The local `VERY-LOW` artifact is already the relevant compact candidate; higher tiers are a quality experiment, not an assumed speed improvement.

### GitHub / llama.cpp

- [PR #27140](https://github.com/ggml-org/llama.cpp/pull/27140) is the strongest direct lead: Qwen3.8-27B on 2x RTX 3090 saw q4_0 prefill 74→1182 tok/s after a CUDA quantized-KV fix. It requires checking `GGML_CUDA_FA_ALL_QUANTS=ON` and the exact source/build. This is P0 before MTP micro-tuning.
- [PR #27489](https://github.com/ggml-org/llama.cpp/pull/27489) shares target/MTP compute buffers and saved about 1.02 GiB on an RTX 4090, with only a small 512-token throughput change (74.63→75.28 tok/s). The optimization is restricted to single-sequence, single-CUDA-device native MTP, so it does not directly cover our dual-GPU tensor split.
- [PR #25532](https://github.com/ggml-org/llama.cpp/pull/25532) reports about 8% from `--backend-sampling` on a different Qwen3.6/RTX 5090 setup, but [issue #27467](https://github.com/ggml-org/llama.cpp/issues/27467) reports CPU fallback under CUDA `SPLIT_MODE_TENSOR` in a cited build. Treat backend sampling as an isolated compatibility experiment, not a production flag.
- [PR #27173](https://github.com/ggml-org/llama.cpp/pull/27173) reports a +12.8% mixed gain from speculative chaining/output mirroring/scheduler changes on 2x RTX 5090. It is open and branch-specific; no direct Qwen3.8 + our asymmetric CUDA reproduction exists.
- [PR #24219](https://github.com/ggml-org/llama.cpp/pull/24219) proposes `TURBOPREFILL=1` and reports large prefill gains, but it is a PoC for layer split and does not establish a tensor-split Qwen3.8 path.
- [issues #27819](https://github.com/ggml-org/llama.cpp/issues/27819) and [PR #27858](https://github.com/ggml-org/llama.cpp/pull/27858) confirm that **upstream standard** DFlash2 + tensor split hits the output-weight/TOP_K split limitation. **Correction (local fork):** our own tree at `C:\AI\llama.cpp` has two local commits on top of upstream — `5ecbe1ac17` "support DFlash2" (common/speculative.cpp Rework + src/models/dflash.cpp, 20 files) and `1deefcca3` "Add p_min in DFlash2" — that implement DFlash2 in the llama.cpp speculative path, so **the upstream limitation does not apply to our fork**. DFlash2 candidate stays on our list for the tensor-split profile; verify with a real paired benchmark, not by assuming parity with the H200/SGLang numbers. The `dflash.selector_top_k` branch in common/speculative.cpp is the fork's DFlash2 path.
- [discussion #27164](https://github.com/ggml-org/llama.cpp/discussions/27164) shows that exact build/library matching matters for Qwen3.8 CUDA correctness; replacing only `llama-server` while leaving stale CUDA libraries can preserve broken kernels.

### X

The supplied [X post](https://x.com/analogalok/status/2088326480669667699) is reproducible as a command lead but not as local evidence: one RTX 4090, `UD-Q4_K_XL`, q4_0 KV, native MTP, `n-max 4`, and `p-min 0.7`. No additional X post was accepted as verified when the public search gateway was unavailable; X-derived values remain author claims unless independently reproduced.

## Revised action order

1. Verify/build the Q4 quantized-KV prefill fix from PR #27140; compare prefill and decode separately.
2. Run native MTP `p-min 0.7` on NVFP4 at ctx 147,456.
3. Pair decoder order: `draft-mtp,ngram-mod` versus `ngram-mod,draft-mtp`.
4. Pair `n-max 2/3/4` at ctx 147,456; repeat only winning candidates at 200,704.
5. Test `--spec-draft-backend-sampling` only after checking whether the current tensor-split build falls back to CPU.
6. A/B current upstream build with native `sm_89` + `sm_120a` and verify the launched binary/library set.
7. Test `CUDA_SCALE_LAUNCH_QUEUES=2x/4x` for prefill only.
8. Do not test P2P or experimental PoCs until a correctness canary and a rollback path exist.


- X post mirror/API data: `https://api.fxtwitter.com/status/2088326480669667699`
- X post: `https://x.com/analogalok/status/2088326480669667699`
- llama.cpp server documentation: `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md`
- llama.cpp latest release API: `https://api.github.com/repos/ggml-org/llama.cpp/releases/latest`
- NVFP4-MTP model card: `https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF`
- Local evidence: `C:/AI/docs/results/README.md`, `C:/AI/docs/results/02-decoders.md`, `C:/AI/docs/results/09-hardware.md`, `C:/AI/docs/researchs/unsloth-studio-config-2026-08-29.md`
