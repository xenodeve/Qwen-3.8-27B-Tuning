# External review of the dual-GPU brief — 2026-08-27

**What this is.** An outside agent was handed
[`plans/07-DUAL-GPU-OPEN-QUESTIONS.md`](../plans/07-DUAL-GPU-OPEN-QUESTIONS.md)
§4–§6 and came back having read llama.cpp source, upstream issues and community
reproductions. **This folder holds external material, unverified until measured
here** — so every claim below carries what *we* did with it.

**Four of its mechanism claims were checked against our own source tree at
`1deefcca3`, the exact commit our binaries report, and all four hold.** Nothing
below has been *measured* on this machine yet; a verified mechanism is still a
hypothesis about a rate.

---

## 1. DFlash2 under `-sm tensor` — a workaround aimed at our exact assertion

**The claim.** The tensor-parallel mapper assigns `output.weight →
SPLIT_AXIS_1` and `output.bias → SPLIT_AXIS_0`. A group of Meta-backend
operations cannot take an axis-0 input, which is the assertion we hit. Someone
running Qwen3.8 + DFlash2 + tensor split hit the same one and reported it fixed
by mapping the output projection to `MIRRORED`.

**Verified here, byte-exact:**

| what | where | what it says |
|---|---|---|
| the mapping | `src/llama-model.cpp:517-524` | `output.weight` → `GGML_BACKEND_SPLIT_AXIS_1`; `output.bias` → `GGML_BACKEND_SPLIT_AXIS_0` |
| the assertion | `ggml/src/ggml-backend-meta.cpp:541-544` | `handle_per_row`, whose entire body is `GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0); return src_ss[0];`, under the comment *"Some ops process data on a per-row bases"* |
| the target axis | `ggml/include/ggml-backend.h:369` | `GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10, // all values on all backends` |
| the arch to scope by | `src/llama-arch.h:154`, `llama-arch.cpp:140` | `LLM_ARCH_DFLASH`, name `"dflash"` — and our drafter logs `arch = dflash` |

So the chain is coherent: a per-row op receives `output.bias`, which is axis 0,
and aborts. **This is a mechanism, not a measurement.**

**What we did.** A git worktree at `C:\AI\llama.cpp-mirror`, detached at the
same `1deefcca3`, patched and saved to
`qwen38-tuning/patches/dflash-mirror-output-1deefcca3.patch` — 28 insertions, 0
deletions. **Deliberately narrower than the reported workaround**, which changed
the mapping for every architecture:

```cpp
if (std::regex_match(tensor_name, pattern_output_weight)) {
    if (ud->model->arch == LLM_ARCH_DFLASH) {
        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    }
    return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
}
```

Scoping it to the drafter's own architecture leaves the **target** model's split
byte-for-byte as it was, so every `UD-Q4_K_XL` rate this project holds stays
comparable against a run of the patched binary. That property is worth more here
than fidelity to the original report.

**`MIRRORED` is not free.** It duplicates the tensor on both devices, so the
drafter's output projection is paid twice. The probe records per-card free VRAM
for exactly that reason.

**🔴 An evidence gap of our own, found while checking this.** The raw log behind
our published `:543` claim **no longer exists** — no file under
`qwen38-tuning/logs/` contains the string `SPLIT_AXIS_0`, and benchmark output
is deleted here by standing instruction. What survives is a structured probe
matrix in `worker-q4-dual.ps1`'s header, written the day it was run:

```
ctx 16,384, -ub 128, lowest memory pressure this configuration reaches
  -sm tensor + ngram-mod  (control)          LOADED
  -sm tensor + draft-mtp  (baked-in head)    LOADED
  -sm tensor + draft-dflash (external -md)   FAILED, meta.cpp:543
  ... + -devd CUDA1                          FAILED, same
  ... + --no-spec-draft-backend-sampling     FAILED, same
  -sm layer  + draft-dflash                  LOADED
```

Good enough to act on, and **not** the raw artifact this project's own standard
asks for. Step 1 of the probe re-establishes the failure on the unpatched binary.

*Task #47.*

## 2. `-sm row` — the cause, and it is not our hardware

**The claim.** The CUDA implementation of `-sm row` was removed in an upstream
refactor; the message is what every CUDA GPU gets now, not a statement about the
4070 SUPER.

**Verified here, and this is the satisfying part:** `ggml/src/ggml-cuda/ggml-cuda.cu`
**does not contain the string `ggml_backend_split_buffer_type` at all.**
`src/llama-model.cpp:982-999` looks it up through
`ggml_backend_reg_get_proc_address` and throws
*"device %s does not support split buffers"* when the lookup returns null. The
entry point is simply not exported any more.

**So this is closed, and none of the explanations we were carrying were needed:**

| candidate | verdict |
|---|---|
| `sm_89` against `sm_120` | not the cause |
| `PXB` topology, no NVLink | not the cause |
| gen4 x4 on the 5060 Ti | not the cause |
| two different GPU models | not the cause |
| **the CUDA row path is gone** | **the cause** |

We never asserted a wrong cause in print — [`results/README.md`](../results/README.md)
records only the message. But the open implication that a matched pair might
behave differently is now dead, and **that experiment does not need running.**

*The upstream PR number and build tag were not verified here; the mechanism was,
and it is the mechanism that closes the question.*

## 3. Why `-sm tensor` still wins over a gen4 x4 link

**The claim.** llama.cpp's CUDA AllReduce is written for tensor parallelism on
machines *without* NVLink: pinned host memory over PCIe, with different
strategies by reduction size — a latency-oriented kernel for small reductions
(token generation) and copy-engine D2H/H2D for large ones (prefill).

**Partly verified:** `ggml/src/ggml-cuda/allreduce.cu` and `.cuh` exist. We did
**not** read the kernel selection logic, so the small/large split is
**unverified here**.

If it is right, our +59–65 % is not an anomaly needing explanation: at `-np 1`
the per-token reduction is small, and the layer split carries a serial
dependency through layer groups regardless. **We are not treating that as
settled** — it is a plausible account of a measured result, which is a different
thing from evidence for it.

### The finding that came with it, and it is a real lever

**`GGML_CUDA_ALLREDUCE` is an environment variable, and we have never set it.**
Verified at `ggml/src/ggml-cuda/ggml-cuda.cu:1207-1243`: it accepts `nccl`,
`internal`, `none`; an unrecognised value warns and falls through to `none`. And
the default is platform-dependent —

```c
#if defined(__linux__)
    ggml_backend_cuda_comm_init_nccl(ret);
#else
    ggml_backend_cuda_comm_init_internal(ret);
#endif
```

— so **on Windows we have been running `internal` all along without knowing it.**
Passing `none` makes the dispatch return false and lets the meta-backend's
butterfly run, which makes `internal` against `none` a clean A/B on how much of
tensor mode's win is the optimised collective.

**This is the surface [the brief](../plans/07-DUAL-GPU-OPEN-QUESTIONS.md) §6.5
said we had not swept.** The 322-flag diff covered CLI flags only; environment
variables were named as unexamined, and this is the first thing found there.

*Task #48.*

## 4. The slot swap is not the test we thought it was

**The claim, and it is simply correct.** This board offers one x16 and one x4.
Swapping the cards moves which one is starved:

```
before:  4070S x16   5060Ti x4
after:   5060Ti x16  4070S  x4
```

**The path between the two GPUs still has an x4 endpoint.** So a swap is *not*
an x4-versus-x16 inter-GPU bandwidth A/B, and a result from it must never be
written up as one. It would change the rate — more lanes for the faster card,
different display VRAM, a different computed `-ts` — but that is three effects at
once and none of them is the question.

A clean interconnect test needs x8/x8 or x16/x16, which this platform cannot do.
**Task #48 replaces this experiment**; [task #39](../OPEN-WORK-LEDGER.md) was
rewritten to say so.

## 5. `--cache-reuse` — our source read corroborated, and deprioritised anyway

The review reports maintainer and community discussion agreeing that the blocker
on a hybrid is the **recurrent state, not the KV cache**: a recurrent layer holds
one state updated per token rather than a per-token history, so the transformer
trick of excising the middle and shifting the remainder does not apply. It also
cites an upstream issue reporting `seq_rm()` returning true while the recurrent
state is not correctly restored on some rollback paths.

**That is the same shape our own source read found** — see
[`results/03-memory-and-kv.md`](../results/03-memory-and-kv.md). External
agreement does not upgrade a source read to a measurement, so **task #42's
falsification probe still runs**: it tests our build, not a report. But no
optimisation campaign will be built on this flag.

### What replaces it

**Recurrent checkpoints.** `--ctx-checkpoints` snapshots state during prefill.
Rather than reusing an arbitrary prefix, restore the checkpoint nearest *before*
an edit and re-prefill only the tail — which fits this workload exactly: a large
immutable vendor/system prefix followed by a small changing region.

**Do not quote our existing `--ctx-checkpoints` row at this.** Task #29 measured
it for **residency** at 128K. Nobody has measured it as a prefill-restore
mechanism and nobody has touched checkpoint *spacing*.

*Task #49.*

## 6. Build flags — no Blackwell flag we are missing

The review enumerated the current CMake CUDA surface (`GGML_CUDA_FORCE_MMQ`,
`FORCE_CUBLAS`, `NO_VMM`, `NO_PEER_COPY`, `FA_ALL_QUANTS`, `NCCL`,
`PEER_MAX_BATCH_SIZE` and others) and found **no flag specific to a mixed
Ada + Blackwell build** that we are missing. Our binary already carries native
`sm_89` and `sm_120a`, which removes the largest trap — PTX JIT at 2.20× the
prefill time with nothing in any log saying so.

`GGML_CUDA_FA_ALL_QUANTS` is called moot because the standard build already
compiles FlashAttention for `f16`, `bf16`, `q4_0` and `q8_0`, which covers every
type we use. **That agrees with what this project already recorded**: the flag is
`OFF` in both our builds and was closed on a Q8 result that could not test it
([CORRECTIONS §29](../reports/CORRECTIONS.md)).

**NCCL is deprioritised, not dismissed.** An external report on Windows 11 with
an `sm_89` + `sm_120` pair on CUDA 13.3 — the same shape as this machine — found
NCCL versus non-NCCL essentially flat on both prefill and generation.
**Unverified here.**

---

## What this changes about our own plan

**The DFlash mirror patch goes first**, ahead of the KV sweep, the AllReduce
A/B, and the machine change. It is the only item that could move the served
configuration from `tensor + ngram-mod` at ~26 tok/s to
`tensor + DFlash2 + ngram-mod`, and `DFlash2 + ngram-mod` is the fastest pairing
this project has ever measured — **42.26 / 43.65 tok/s** at 16,384 on the layer
split, where it is currently stranded.

**One thing did not change.** Everything above is a mechanism or a report.
Not one of it is a rate measured on this machine, and the patched binary has not
been built yet.

---

## 7. The drafter is a separate model by design, and smaller ones exist

**Asked 2026-08-27: is there a DFlash2 model with the head baked in, the way
`draft-mtp` uses a head inside `UD-Q4_K_XL`?** If there were, it would need no
`-md` sidecar, no 786.35 MiB buffer on device 1, and possibly no mirror patch.

**No.** [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
is the upstream drafter and its card says so outright:

> *"It is not a standalone language model: it runs inside a speculative decoding
> server and drafts tokens for the target model to verify."*

A 2B BF16 safetensors drafter, paired at run time. The `z-lab` GGUF we load is a
conversion of it. **The sidecar and its buffer are structural.**

### What the size ladder actually is

Read from the Hub, exact bytes:

| repo | file | on disk |
|---|---|---:|
| `z-lab` **(ours)** | `Q4_K_M` | **1,090 MiB** |
| `z-lab` | `Q8_0` / `BF16` | 1,961 / 3,681 MiB |
| `andrew-paul` | `Q3_K_M` *(imatrix)* | **874 MiB** |
| `andrew-paul` | `Q2_K` | 673 MiB |
| **`HermiHg`** | **`Q2_K_S-MIX`** *(imatrix, mixed 2–3 bit)* | **535 MiB** |
| `Anbeeld` | the full ladder Q2_K → bf16 | — |

### The external table that makes this worth testing

`HermiHg`'s card carries a measured comparison — `llama-server` with DFlash2 on
**one 24 GB NVIDIA card**, five replicates, one fixed prompt:

| `n_max` | metric | Q4_K_M | Q3_K_M | Q2_K | Q2_K_S-MIX |
|---|---|---:|---:|---:|---:|
| 4 | acceptance | 0.466 | 0.459 | 0.441 | 0.435 |
| 4 | tok/s | 99.2 | 98.3 | 96.1 | 95.6 |
| 5 | tok/s | 104.2 | 104.8 | 102.4 | 101.9 |

**Throughput moves by a few percent while the file halves.** And because DFlash2
is draft-and-verify, a weaker drafter costs **speed, not quality** — the target
still verifies every token.

**Not our machine, not our target, not our split.** One card rather than a
tensor split, and `ggml-org/Qwen3.8-27B-GGUF` rather than Unsloth's
`UD-Q4_K_XL`. Unverified here.

### Why it matters to us specifically

**The allocation that failed at 200,704 was 786.35 MiB on device 1 — the
drafter's own Meta buffer**, not the total budget. Scaling by file size predicts
**~630 MiB for Q3_K_M** and **~386 MiB for Q2_K_S-MIX**. If that holds, the
request that could not be satisfied becomes one that can.

`Q3_K_M` is also an **imatrix** quant and posts *higher* acceptance than our
`Q4_K_M` at `n_max` 3 (0.543 against 0.539) — smaller and possibly better.

### 🔴 A correction to something said in chat an hour earlier

`deresolution/Qwen3.8-27B-DFlash2-mxfp4` was floated here as interesting because
this build reports `BLACKWELL_NATIVE_FP4 = 1`. **That was wrong and it is
checked:** the repo is an **MLX** conversion for Apple Silicon, produced by
`mlx_vlm.convert`, and its README directs the reader to oMLX settings. It
mentions llama.cpp nowhere. Our `GGML_TYPE_MXFP4` (`ggml.h:429`) is a GGUF
tensor type added for gpt-oss — the same three letters, a different thing.

### And DFlash2 is upstream now

`HermiHg`'s card states DFlash 2 support was **merged into llama.cpp main on
2026-08-27**. Our build is `1deefcca3`, described by `git describe` as
`b10488-11-g1deefcca3`. **Unverified here**, and worth checking before anyone
plans another rebuild around PR #27342.
