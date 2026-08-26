# 36 — Dual-GPU on llama.cpp: everything measured, and the five things we cannot explain

**Written 2026-08-27 to be handed to an outside model.** It is deliberately
self-contained: hardware, build identity, exact command lines, exact error
strings, source line references, and every number with the run that produced it.

**Read the register split first.** Sections 1–5 are **measured on this machine**.
Section 6 is **five open questions about mechanism** — those are what we want
explained, and we have deliberately not guessed at answers there.

---

## 1. The machine, exactly

```
GPU 0 (CUDA0)  NVIDIA GeForce RTX 4070 SUPER   12,282 MiB  sm_89   PCI 0000:01:00.0
GPU 1 (CUDA1)  NVIDIA GeForce RTX 5060 Ti      16,311 MiB  sm_120  PCI 0000:06:00.0
driver 610.88 · Windows 11 Pro 26200 · CUDA 13.3
```

- **CUDA0 (the 12 GB card) drives the display.** `explorer.exe`, Windows
  Terminal, a Chromium browser, the NVIDIA overlay, `TextInputHost.exe` all hold
  memory on it — **about 1,600 MiB at rest**. CUDA1 holds 49 MiB.
- `nvidia-smi topo -m` reports **`PXB`** between them: several PCIe bridges, no
  NVLink, no peer link.
- **PCIe link state sampled under load**, 49 samples at 1 Hz, 34 with the GPU
  busy: **CUDA0 peaks at gen4 x16, CUDA1 at gen4 x4** against a `link.width.max`
  of 16 for both. The *generation* downtrains at idle and recovers; the *width*
  never does. **The 16 GB card is in an x4 slot.**

### The build

```
version: 0.1.2-dev (build 10499, commit 1deefcca3)   "Add p_min in DFlash2"
built with MSVC 19.44.35228.0 for Windows AMD64
CMAKE_CUDA_ARCHITECTURES = 89;120
system_info: CUDA : ARCHS = 890,1200 | USE_GRAPHS = 1 | BLACKWELL_NATIVE_FP4 = 1
```

`cuobjdump --list-elf ggml-cuda.dll` → **141 `sm_120a` cubins beside 141
`sm_89`**. One binary drives both cards natively; nothing is JIT-compiled.

`GGML_CUDA_FA_ALL_QUANTS` is **OFF**, so only `f16`, `bf16`, `q4_0`, `q8_0`
have flash-attention kernels.

### The model

`unsloth/Qwen3.8-27B-GGUF` → `Qwen3.8-27B-UD-Q4_K_XL.gguf`,
**17,923,394,624 bytes (16.69 GiB)**, `n_ctx_train = 262144`,
66 layers reported as `66/66` offloaded, of which **16 are attention layers**
carrying KV. Measured KV rate: **18.00 KiB per token at `-ctk q4_0 -ctv q4_0`**.

Draft model where used: `z-lab/Qwen3.8-27B-DFlash2-GGUF` →
`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, **1,143,006,752 bytes**, `LLM_ARCH_DFLASH`.

---

## 2. What the four split modes do here

`--split-mode {none,layer,row,tensor}`.

| mode | result on this pair |
|---|---|
| `none` | one card. `UD-Q4_K_XL` does not fit: layer split **`55+11`**, eleven layers on the CPU, **11.7 tok/s** |
| `layer` | works. Splits by llama.cpp's own free-VRAM view |
| `row` | **cannot load** — see §6.1 |
| `tensor` | works, **EXPERIMENTAL** per `--help`. Aggregates both cards into one virtual `Meta` device |

### `-sm row` fails at model load, every attempt, in about one second

```
E llama_model_load: error loading model: device CUDA0 does not support split buffers
E llama_model_load_from_file_impl: failed to load model
E common_fit_params: encountered an error while trying to fit params to free device memory: failed to load model
```

### `-sm tensor` builds an aggregate device

```
I llama_prepare_model_devices: creating a Meta device for tensor parallelism from 2 devices:
I llama_prepare_model_devices: using device Meta() (Meta()) (unknown id) - 26241 MiB free
D load_tensors: layer 0 assigned to device Meta(), is_swa = 0     [... all 66 ...]
I load_tensors: offloaded 66/66 layers to GPU
```

**Its buffer figures are per device, not totals.** At ctx 147,456, `-ub 1024`:

```
I load_tensors:      Meta() model buffer size  = 8065.29 MiB
I llama_kv_cache:    Meta() KV buffer size     = 1296.00 MiB
I sched_reserve:     Meta() compute buffer size= 1024.30 MiB
I load_tensors:  CPU_Mapped model buffer size  =  682.03 MiB
```

`8065.29 × 2 + 682.03 = 16,812 MiB`, which matches the 16.69 GiB artifact.

---

## 3. The incident: 0.38 tok/s, and why

The tuned two-card profile decoded at **0.38 tok/s** on the developer's machine
against the **32.4** it had been benchmarked at. Task Manager showed the
**5060 Ti at 0 % and 45 °C** while the **4070 SUPER ran at 88 %**, holding
**11.6 of 12.0 GB** with **0.7 GB in shared (host) memory**. Prefill collapsed
too: **16.4 tok/s on a 330-token prompt** against 973.

**Cause, from source.** `-sm tensor` splits **evenly** when given no ratio.
`llama-model.cpp:707`:

```c
int64_t high = tensor_split_scan.back() == 0.0f ?
    ne_s * (j+1)/ud->n_devices : ne_s * tensor_split_scan[j]/tensor_split_scan.back();
```

With `tensor_split == nullptr` the scan sums to `0.0f` and the slice is
`ne_s * (j+1)/n_devices` — **capacity is not consulted at all.**

Per-card demand was `8,065 + 1,296 + 1,024 = 10,385 MiB`:

| | total | desktop | demand | left |
|---|---:|---:|---:|---:|
| RTX 4070 SUPER | 12,282 | 1,579 | 10,385 | **+317 MiB** |
| RTX 5060 Ti | 16,311 | 49 | 10,385 | +5,876 MiB |

**`--fit` cannot rescue it**, on every boot:

```
W common_fit_params: failed to fit params to free device memory:
  llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort
```

**The fix we shipped** computes `-ts` at launch from `nvidia-smi` free VRAM
minus a reserve on whichever card already holds memory (2,500 MiB), and refuses
to start when the budget cannot hold the weights. On this machine it resolves to
**`-ts 7819,15490`**.

| `-ts` | decode | CUDA0 free after load |
|---|---|---|
| even (the default) | **0.38 tok/s** | +317 MiB |
| `2,3` | 31–33 tok/s | 1,511 MiB |
| `1,2` | 28–30 tok/s | 2,792 MiB |
| **computed `7819,15490`** | **25.8 / 42.7 / 78.3 tok/s** | **2,921 MiB** |

After the fix both cards sit at **95 %**, drawing **111 W and 119 W**.

---

## 4. Throughput, paired and rotated

Every figure below: three rounds, arms rotated between rounds, greedy sampler
(`temperature 0.0, top_k 1, seed 42`), `UD-Q4_K_XL`, `-ctk q4_0 -ctv q4_0`.

### ctx 147,456 (the depth we serve), corpus `real-code-vendor`

| arm | tok/s | own spread |
|---|---|---|
| `-sm tensor -ts 7819,15490 -ub 1024` + `ngram-mod` | **26.2 / 25.6 / 26.7** | 4.2 % |
| `-sm layer -ub 1024` + `ngram-mod` | 18.1 / 18.7 / 18.7 | 3.6 % |
| | **−29.2 %** [−31.0, −26.9] | |
| `-sm layer` + `draft-dflash,ngram-mod` | **FAILED TO LOAD** | — |
| `-sm tensor` + any external drafter | **FAILED TO LOAD** | — |

Earlier, same depth, `-sm tensor` **without** `-ts` (the even split, on a quiet
desktop where it happened to fit): 28.7 / 28.7 / 28.6 bare, and 32.4 / 32.6 /
33.1 with `ngram-mod`. **Those numbers are real but they describe a
configuration that collapses under desktop load** — they are recorded here only
so nobody quotes them as current.

### ctx 16,384, corpus `real-code` — and the ordering inverts

| arm | tok/s |
|---|---|
| **`-sm layer -ub 1024` + `draft-dflash,ngram-mod`** | **42.26 / 43.65** |
| `-sm tensor -ts 7819,15490 -ub 1024` + `ngram-mod` | 27.66 / 26.15 |
| `-sm layer -ub 1024` + `ngram-mod` | 24.44 / 23.00 |

**DFlash2 on the layer split is the fastest configuration measured anywhere in
this work — and it is unavailable at the depth we serve.**

### The knobs, on `-sm tensor`, ctx 16,384

- **`-ub`**: decode flat across 128/256/512/1024 (−1.1 %, −0.6 %). **Prefill is a
  staircase** on the identical 6,621-token prompt: 820 / 884 / 938 / **971**
  tok/s. `-ub 1024` is **+10.1 %**, ranges non-overlapping.
- **KV `q8_0`**: costs **nothing** at 16,384 (−0.3 %, 128 MiB) and **cannot load
  at 147,456** — `cudaMalloc failed: out of memory` allocating 1,024.30 MiB on
  device 0.
- **`-mg`**: not applicable. `--help` scopes it to `-sm none` or `-sm row`.

### Residency ladder, `-sm tensor -ts 7819,15490 -ub 1024`

| ctx | 196,608 | 229,376 | 245,760 | **262,144** |
|---|---|---|---|---|
| split | 66+0 | 66+0 | 66+0 | **66+0** |
| free | 3,408 | 2,347 | 1,951 | 1,515 MiB |

**Fully resident to `n_ctx_train`.** The ceiling is now the model, not the cards.

---

## 5. One methodological result, because it invalidates a whole class of number

**A speculative decode rate is partly a measurement of how repetitive the output
is**, and on a split model the output is not the same output.

Measured: at ctx 16,384 with `ngram-mod`, one card gave **165.1 / 164.6 / 163.9
tok/s** and two cards gave **35.9 / 35.8 / 35.6** — a stable, sign-consistent
−78.3 %. It is not a hardware result. The two arms **decoded different text**:
acceptance 93.3 % against 58.5 %, and counting distinct lines in what each
produced, **24 of 47 against 30 of 47**. The single-card arm fell into a tighter
repetition loop and `ngram-mod` converted that into throughput.

The sampler is already greedy. The text differs because **splitting a model
changes the order of the reductions and therefore the logits**. With speculation
**off**, the same comparison is **+1.5 %** [+1.1, +2.1], and prefill on the
identical prompt is **+57.4 %**.

---

## 6. The five open questions — this is what we want explained

We have not guessed at these. Each is stated with the evidence we have.

### 6.1 Why does `-sm row` report "device CUDA0 does not support split buffers"?

Both devices are CUDA devices on one host with a normal CUDA 13.3 install.
Is this about peer access (`cudaDeviceCanAccessPeer`) across a `PXB` topology
with no NVLink, about mismatched compute capabilities (`sm_89` vs `sm_120`),
about Windows WDDM, or about a build option? **Would it work on two identical
cards on the same board?**

### 6.2 Why can `-sm tensor` not host an external draft model?

**Both** `draft-mtp` and `draft-dflash` abort at the identical assertion, every
attempt:

```
C:\AI\llama.cpp\ggml\src\ggml-backend-meta.cpp:1522:
GGML_ASSERT(bufs.back() != nullptr) failed
```

Source around it (`ggml_backend_meta_buft_alloc_buffer`):

```c
std::vector<ggml_backend_buffer_t> bufs;
bufs.reserve(n_simple_bufts);
for (size_t i = 0; i < n_simple_bufts; i++) {
    bufs.push_back(ggml_backend_buft_alloc_buffer(
        ggml_backend_meta_buft_simple_buft(buft, i), size));
    GGML_ASSERT(bufs.back() != nullptr);
    ...
}
```

So one of the underlying per-device buffer allocations returned `nullptr`.
**Is this an out-of-memory condition being asserted instead of reported, or is
the Meta backend structurally unable to hold a second model's buffers?** The
distinction matters: the first would be fixable by making room, the second
would not. Note `ngram-mod`, which allocates no weights, works fine.

### 6.3 Why does DFlash2 load at ctx 16,384 and fail at 147,456?

Same binary, same drafter file, same `-sm layer`, only `-c` differs.

```
E llama_init_from_model: failed to initialize the context:
  dflash requires ctx_other to be set (this warning is normal during memory fitting)
W srv load_model: [spec] failed to measure draft model memory:
  failed to create llama_context from model
```

The throw is `llama-context.cpp:154-161`:

```c
if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
    if (model.tok_embd == nullptr || model.output == nullptr) {
        if (params.ctx_other == nullptr) {
            throw std::runtime_error(model.arch_name() +
                " requires ctx_other to be set (this warning is normal during memory fitting)");
        }
        cparams.ctx_other = params.ctx_other;
    }
}
```

and `common/speculative.cpp:2461` is what sets it: `cparams.ctx_other = ctx_tgt;`

**The message says the throw is normal during memory fitting.** So the probe
throwing is expected — the question is why the *outcome* differs with context
length. Our hypothesis, unverified: at 147,456 the fitting pass matters (there
is less headroom) and the failed draft-memory measurement becomes fatal, whereas
at 16,384 the load proceeds regardless. **We would like this confirmed or
replaced.** Is there a supported way to give the drafter its `ctx_other` at
fit time, or a flag that skips the draft-memory probe?

### 6.4 Why is `llama_params_fit` not implemented for `SPLIT_MODE_TENSOR`?

```
W common_fit_params: failed to fit params to free device memory:
  llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort
```

**Is this a known gap, is it planned, and is there an intended way to size a
tensor-parallel run?** As shipped, `--fit on --fit-target 768` is silently inert
under this mode, and an over-committed split does not fail — it **spills to host
memory and keeps serving**, which is how 0.38 tok/s went unnoticed until a human
looked at Task Manager.

### 6.5 Why is the tensor split's default even rather than capacity-weighted?

`-sm layer` splits by free VRAM. `-sm tensor` splits evenly
(`llama-model.cpp:707`). On two cards of different size — the common
consumer case — the even default overcommits the smaller one by construction.
**Is that deliberate (an assumption of homogeneous devices for tensor
parallelism), or simply unimplemented?**

And a related one: **llama.cpp's own free-VRAM figure does not track the
desktop.** It reported `11069 MiB free` on CUDA0 while `nvidia-smi` reported
10,419 and the desktop held 1,579. That constant has appeared in **all 552 boot
logs this project has kept**. Where does it come from, and is there a way to
make llama.cpp size against the driver's actual free memory instead?

---

## 7. Everything we did not test

Stated so this report is not read as exhaustive.

- **`-sm layer` with a capacity-computed `-ts`.** We tested `-ts 1,1` on layer at
  ctx 16,384 (+1.8 %, inside the floor) and wrongly generalised. Layer at
  147,456 gives CUDA0 `6,621 + 1,134 + 1,969 = 9,724 MiB`, leaving about
  **958 MiB** after the desktop — **more slack than tensor had, but by luck.**
- **Any depth between 16,384 and 147,456.** So we do not know where DFlash2 stops
  working, or where the layer/tensor ordering inverts.
- **DFlash2 at 147,456 with a smaller `-ub` or a shorter context.**
- **Throughput at 262,144** — only residency was measured there.
- **Quality, at any depth, on any artifact.** This project has never measured it.
- **Two identical cards**, which would separate "unequal capacity" from
  "unequal architecture" in almost everything above.

---

*Raw data: `qwen38-tuning/results/dual-*.jsonl`,
`qwen38-tuning/bench/ctx-ceiling-*.jsonl`, and the
`qwen38-tuning/logs/dflash2-*.log` each row names. Issues
[#50](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/50),
[#51](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/51),
[#52](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/52).
Retractions: [`CORRECTIONS.md`](CORRECTIONS.md) §31–§33.*
