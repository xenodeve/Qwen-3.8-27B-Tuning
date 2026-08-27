# 07 — The two-card configuration: what is settled, what is open, and what we are about to run

**Written 2026-08-27 to be read by someone outside this repository.** It is
self-contained: every number, flag and file path needed to reason about the
problem is quoted here rather than linked. Nothing below requires access to the
codebase.

**What we want from you.** A speed review. Decode and prefill on a single
machine, one user, one slot. Sections 1–4 are the current state so you can see
what has already been eliminated; section 5 is the ranked plan we are executing;
section 6 lists what we specifically want challenged. **Section 7 is the list of
things NOT to suggest** — each was tried here and the outcome is recorded.

> **How to read a number in this document.** Every rate is three paired rounds
> with the arms rotated between rounds, greedy sampling (`temperature 0.0,
> top_k 1, seed 42`). Where a claim was read from source rather than measured,
> it says so in the sentence. We separate those deliberately: this project has
> published thirteen instrument faults, and each one returned a plausible number
> rather than an error.

---

## 1. The machine

```
CUDA0   RTX 4070 SUPER 12 GB   sm_89    PCIe gen4 x16 under load   DRIVES THE DISPLAY
CUDA1   RTX 5060 Ti    16 GB   sm_120   PCIe gen4 x4  under load   holds 50 MiB idle
```

28 GB across two cards, **`PXB` topology, no NVLink**, one CPU with 20 threads.
An Intel UHD 770 iGPU exists and drives nothing.

Three facts that shape everything:

1. **CUDA0 is the older, smaller card, it enumerates first, and `--main-gpu`
   defaults to 0.** It is also the one drawing the desktop.
2. **The 5060 Ti's slot is x4, measured under load** across 49 samples, 34 of
   them busy. The PCIe *generation* downtrains at idle; the *width* never does.
   So the faster, larger card sits behind a quarter of the other card's
   bandwidth.
3. **The desktop's VRAM appetite is a live variable, 1,600–2,600 MiB.** It
   decided whether a 262,144-token context loaded or ran out of memory, twice,
   hours apart, on an otherwise identical configuration.

## 2. The build and the model

`llama.cpp` **build 10499, commit `1deefcca3`**, built with
`CMAKE_CUDA_ARCHITECTURES=89;120` — 141 `sm_120a` cubins beside 141 `sm_89`,
because one of the two cards is Ada. A binary without Blackwell SASS still runs
here, through PTX JIT, at **2.20x the prefill time with nothing in any log
saying so**, which is why the profile refuses to start against a binary whose
`ggml-cuda.dll` lacks the string.

Model: **Qwen3.8-27B, Unsloth `UD-Q4_K_XL`, 16.69 GiB on disk**, `arch = qwen35`,
65 layers, `n_ctx_train = 262,144`, `n_swa = 0`, `n_head_kv = 4`,
`n_embd_head_k = n_embd_head_v = 256`.

**It is a hybrid.** 48 of its layers are Gated DeltaNet with a recurrent state
that is separate from the attention KV cache. With `n_swa = 0` it builds
`llama_memory_hybrid`. This matters more than it sounds — see §4.3.

**It never fit on one 16 GB card at any depth.** Across two it is `66+0` —
fully resident — at every rung to 229,376 and spills one layer at 262,144. The
second card is worth **+79.9 %** [+77.3, +82.2] to it, which is the residency
cliff (`55+11` becoming `66+0`), not a parallelism gain.

## 3. What is served right now

```
llama-server.exe
  -m <UD-Q4_K_XL>
  --alias Qwen3.8-27B-Q4_K_XL
  -c <computed at launch, see below>
  -ngl auto --fit on --fit-target 768
  -fa on
  -np 1
  -sm tensor  -ts <computed at launch>
  -t 18  -b 2048  -ub 1024
  -ctk q4_0 -ctv q4_0
  --spec-type ngram-mod
  --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32
  --chat-template-file <late-system jinja>
  --reasoning-effort medium
  --sse-ping-interval 5
  --host 127.0.0.1 --port 8080
```

**Measured on this exact configuration, ctx 147,456, corpus of real vendor
code:** `26.2 / 25.6 / 26.7 tok/s` decode, own spread 4.2 %. Prefill about
971 tok/s on a 6,621-token prompt at ctx 16,384.

### `-ts` is computed at launch, and that is not cosmetic

`-sm tensor` **splits the model EVENLY when given no ratio**
(`llama-model.cpp:707`, `ne_s * (j+1)/ud->n_devices`). The cards are 12 GB
against 16, and the 12 GB one draws the display. The first version of this
profile shipped without `-ts` and **decoded at 0.38 tok/s** — the even split
left **+317 MiB** on the display card, the driver paged to host memory, and
every token crossed PCIe.

`--fit` does not catch it: **`llama_params_fit is not implemented for
SPLIT_MODE_TENSOR`**. The flag is accepted, logs `abort`, and the run continues.
We initially read that log line as a hard load failure. It is a **silent
spill**.

So the profile now measures free VRAM per card by UUID, subtracts a reserve
(**2,500 MiB** on whichever card already holds memory — the display card — and
512 MiB on an idle one), and passes the result as `-ts`. After: **25.8 / 42.7 /
78.3 tok/s at three depths, both cards at 95 % utilisation.**

### The context window is computed at launch too

`-MaxCtx` asks for the deepest window the *current* budget supports, capped at
`n_ctx_train`:

```
budget = sum(free VRAM per card, less the reserves) - RUNTIME_RESERVE_MIB(768)
demand = WEIGHTS(16,130 MiB) + KV(ctx x 18.00 KiB/token) + COMPUTE(2 x ubatch MiB)
```

It spends the micro-batch before the context: halving `-ub` frees about one
`-ub` of MiB across the pair for ~3.5 % of prefill, where the same MiB bought
with context costs tens of thousands of tokens.

`RUNTIME_RESERVE_MIB = 768` is a **measured line, not a model**: at 262,144 with
`-ub 512`, a run with 336 MiB free on the second card **died on its first real
request** with `cuMemSetAccess ... out of memory`, and one with 488 MiB survived
a 135,233-token request. **Loading is not surviving**, and every depth in this
project is now re-tested with a real request rather than a `/health` probe.

Two launches minutes apart chose **249,856** and **245,760** tokens. That
variance is the desktop moving, and it is the design working.

## 4. The structural findings — these are the constraints, not preferences

### 4.1 `-sm row` cannot load on this pair

`device CUDA0 does not support split buffers`, at model load. Not a tuning
question.

### 4.2 No externally-loaded drafter works under `-sm tensor`

`-sm tensor` builds a virtual **`Meta` device** aggregating both cards, and it
cannot host a second model.

- `draft-dflash` (a separate 1.06 GB file via `-md`) aborts at
  **`ggml-backend-meta.cpp:543`**, `GGML_ASSERT(src_ss[0].axis !=
  GGML_BACKEND_SPLIT_AXIS_0)` — a **graph-split axis** assertion, raised at
  negligible memory pressure. Structural, not OOM.
- `draft-mtp` (the head baked into the model) **does load** under `-sm tensor`.
  We published the opposite for a day; an outside review prompted the probe that
  disproved it. Its rate is still unknown: every paired round was **voided**
  because the generations copy the prompt rather than answer it.

So on the split we serve, **the only speculative option is one that needs no
weights** — the `ngram-*` family.

**This is the single largest unexploited gap.** `-sm layer` + `draft-dflash` +
`ngram-mod` at ctx 16,384 measures **42.26 / 43.65 tok/s**, against **27.66 /
26.15** for `-sm tensor` + `ngram-mod` at the same depth — the fastest
configuration measured anywhere in this work, and it will not load at the depth
we serve. **Every depth between 16,384 and 147,456 is untouched**, so where it
stops working is unknown.

### 4.3 `--cache-reuse` cannot be used, and llama.cpp does not detect it

**Read from source on commit `1deefcca3`, not measured.** We were about to
adopt it: a broken prefix costs 63 s at 16K and 248 s at 64K here, the largest
single cost this project has measured.

- The server disables `--cache-reuse` when `llama_memory_can_shift()` is false
  (`server-context.cpp:1176-1185`). For a hybrid that returns
  **`mem_attn->get_can_shift()` only** (`llama-memory-hybrid.cpp:133-136`),
  whose comment reads *"Shifting is trivially supported for recurrent"* — true
  of a **position**, false of a **state**. No warning is printed.
- The reuse loop then calls `seq_rm(id, head_p, head_c)` then `seq_add`
  (`server-context.cpp:3180-3181`). The hybrid tries the recurrent side first
  and will not mutate the attention cache if it fails — but for a mid-sequence
  range it **does not fail**: `llama_memory_recurrent::seq_rm`
  (`llama-memory-recurrent.cpp:150-233`) takes neither special branch and falls
  through to `return true` **having touched nothing**.
- Net: attention KV re-indexed to the new prompt, **DeltaNet state still
  encoding the old prefix**, no error. A removal that reaches the tail instead
  takes a bounded-rollback branch, requires `rollback <= n_rs_seq`, and ours is
  **0** — so it returns false and `GGML_ABORT`s. That path crashes rather than
  lying.
- **`n_rs_seq` has no command-line argument.** It is set by `need_n_rs_seq()`
  (`common/common.h:386-392`) only for `draft-mtp`, `draft-eagle3`,
  `draft-dflash`, `draft-dspark` — **zero for every `ngram-*` type**, which is
  why our boot log reads 0. `qwen35` *is* in `llm_arch_supports_rs_rollback`, so
  the capability exists and is simply never provisioned for us. It would not
  cover the mid-sequence path anyway.

**We have not run the falsification test yet** (greedy, one prompt, then the
same prompt with a mid edit, compared against a cold run of the edited prompt).
Until we do, the above is a source read.

### 4.4 The levers already swept on two cards

| lever | verdict |
|---|---|
| `-sm layer` vs **`-sm tensor`** | tensor, **+59.5 %** at 16,384 and **+65.4 %** at 147,456 |
| `-ts` ratio | **no lever within `layer`; decisive within `tensor`** (see §3) |
| `-ub` 128 / 256 / 512 / **1024** | 1024. Decode flat (−1.1 %, −0.6 %); **prefill 820 / 884 / 938 / 971 tok/s** |
| KV `q8_0` vs `q4_0` | −0.3 % at 16,384. See §6.4 — the depth half of this verdict is confounded |
| `-mg` | not applicable; `--help` scopes it to `-sm none` or `-sm row` |
| `-t` 8 / 12 / 18 / 24 | 18; differences under the floor |
| `-fa on` / `off` | **on**, required — `off` loses residency. Confirmed `flash_attn = enabled` in the boot log |
| `GGML_CUDA_GRAPH_OPT` | inert, and its body contains no `cudaGraph*` call |
| thread affinity, process priority, polling, GPU-side sampling | all inert, +0.46 % / −2.02 % / +0.69 % / +2.27 % |
| `--no-repack`, `--no-op-offload`, `--load-mode`, `--no-host`, `--swa-full`, `--no-kv-unified` | all inert |
| `-dt / --defrag-thold` | **dead flag** — accepted, prints a deprecation warning, does nothing (`common/arg.cpp:2522-2531`) |

## 5. The plan we are executing, in order

The ordering criterion is **ease x impact**, not impact alone.

### Running now — the n-gram family and `n-match`, at the served depth

`--spec-type` takes **eleven** values, five of which are weightless n-gram
variants: `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`,
`ngram-cache`. All of them load under `-sm tensor`. **Only `ngram-mod` has ever
been run on this machine.**

The family *was* swept — on the old single 12 GB card. `ngram-map-k` led at
16,384 (**+135.89 %** against `ngram-mod`'s +112.55 %) and lost at 131,072
(+120.54 % against **+200.22 %**). Those magnitudes are **upper bounds**: the
prompt was 84.5 % duplicate lines and every elimination was decided on 160-token
generations.

`n-match` rides in the same boots. We ship **12**, which loses at both depths
ever measured: **24** wins at 16,384 (+34.6 %) and **16** wins at 65,536, where
12 is the second-worst arm. It moves no allocation.

`ngram-cache` is **excluded**: its greedy hash `3EFE93950A8A980E` differs from a
same-depth baseline of `04E5CAB1D14525C0`, so it changes the answer and is not
draft-and-verify.

**Round 1 of 3, in flight as this is written** (ctx 147,456, real vendor code,
all arms `66+0` resident):

| arm | tok/s | draft acceptance | mean accepted length |
|---|---:|---:|---:|
| `ngram-mod` `n-match 12` (ours) | **25.62** | 55.4 | 18.11 |
| `n-match 16` | 20.09 | 31.6 | 10.86 |
| `n-match 24` | 23.29 | 65.9 | 22.45 |
| `ngram-map-k` (defaults) | 20.00 | — | — |

**One round is not a result**, and `ngram-map-k4v` had not reported when this
was written. Two shapes are worth your attention even so:

- **`n-match 24` accepts more drafts and longer ones, and is still slower than
  our 12.** That is the opposite of the 16,384 finding, where 24 won by 34.6 %.
- **`ngram-map-k` declined 100 % of its drafts** — acceptance is not merely low,
  it is empty, so the arm is paying draft cost for nothing at this depth. It won
  by a wide margin at 16,384 on the old card.

### Then, in order

1. **`-b` and `-ub` together at 2048.** The only raw prefill knob whose trend
   has not been run to its end. `-ub` above `-b` is **silently clamped**, so
   moving one alone produces an arm identical to its neighbour with nothing
   saying so. ~180 MiB per step.
2. **Falsify the `--cache-reuse` source read** (§4.3). If the generations match
   a cold run token for token, we are wrong and must retract.
3. **The DFlash2 depth ladder** — 32,768 / 49,152 / 65,536 / 98,304 in `-sm
   layer`, stopping at the first depth that fails and recording which of the two
   assertions it raises. Biggest prize, most expensive test.
4. **KV type with `-ts`**: `f16` / `bf16` / `q8_0` / `q4_0` at 16,384, and
   `q8_0` at 147,456 to settle §6.4. This build compiles an FA kernel for
   exactly those four types.
5. **The machine change** — display to the UHD 770, and possibly moving the
   5060 Ti to the x16 slot. Developer's hands. Deliberately *not* interleaved
   with the software campaign, because freeing 1.6–2.6 GB on the display card
   changes the computed `-ts` and therefore every rate: numbers before and after
   are not comparable.

## 6. What we would like challenged

1. **Is there a way to get a weight-bearing drafter onto `-sm tensor` at all?**
   `ggml-backend-meta.cpp:543` is a graph-split axis assertion, not memory
   pressure. Is this a known upstream limitation with a known workaround —
   pinning the drafter to one device with `-devd`, a different split axis, an
   older Meta backend revision? We deliberately did **not** rebuild at an older
   commit to test the pre-revert backend, because it would change the binary
   every measured row in this project stands on.
2. **Is `-sm tensor` even the right answer given a gen4 x4 link?** It moves
   activations between cards *inside every layer* rather than once per
   boundary, and it still beats `layer` by 59–65 %. That is surprising to us and
   we would like it stress-tested. Would the ordering invert if the 5060 Ti had
   x16?
3. **Is there a prefill lever we have not found?** Everything in §4.4 that
   touches prefill is either inert or already at its best measured value, and
   the one real remaining idea — reusing a broken prefix — appears to be
   structurally unavailable on a hybrid model (§4.3).
4. **`q8_0` KV at depth is a confounded verdict, not a disproved one.** The run
   that recorded `cudaMalloc failed: out of memory` on the 12 GB card reports
   `Meta() model buffer size = 8,065.29 MiB`, and 8,065.29 x 2 = 16,130.58 MiB
   is the model exactly — **it ran the even split**. The arithmetic says q8_0
   should fit with a computed `-ts`: 5,184 MiB of KV + 16,130 weights + ~2,048
   compute + 768 reserve = 24,130 MiB against 26,072–27,072 available. We have
   flagged it and not yet re-run it.
5. **Build-level options are an unexamined surface.** We compared all 322
   `llama-server` runtime flags against the 20 this profile sets, but we have
   **not** enumerated `-DGGML_*` cmake options or `GGML_CUDA_*` environment
   variables. Only `GGML_CUDA_GRAPH_OPT` was ever tested, and it was inert. If
   there is a known Blackwell or mixed-architecture build flag that matters
   here, that is the surface we have not swept.

## 7. Please do NOT suggest these — each was tried, with the outcome

| suggestion | what happened here |
|---|---|
| `-sm row` | cannot load: `device CUDA0 does not support split buffers` |
| `--main-gpu` / `-mg` | scopes to `-sm none` or `-sm row`; not applicable |
| `--defrag-thold` | deprecated no-op since `common/arg.cpp:2522-2531` |
| `--draft-max`, `--draft-min`, `--draft-n`, `--spec-ngram-size-n/-m`, `--spec-ngram-min-hits` | **removed** upstream; they call `arg_removed()` and abort startup |
| `--spec-draft-n-max` | governs a **file-loaded** drafter; none loads under `-sm tensor` |
| `--spec-draft-p-min <= 0.0625` | mathematically identical to 0.00 — `1/sum` is in `[1/16, 1]` by construction |
| `-ctkd` / `-ctvd` (drafter KV quant) | the drafter decodes 5 tokens per step, so quantised KV takes MMA_F16 with a full dequant rather than the VEC kernel |
| `--cache-ram` / prompt cache | already on by default at 8,192 MiB and measured at **343x** on task switching. Banked |
| `--lookup-cache-static/-dynamic` | the state behind `ngram-cache`, which is disqualified for changing the answer; also `n_draft` is hardcoded to 8 and a cache file that fails to load `GGML_ABORT`s |
| `--slot-prompt-similarity` | only decides whether a slot is reused at all; dies with `--cache-reuse` (§4.3) |
| larger `-np` / continuous batching | one user, one slot. `-np 0` throws; slots are built once and cannot be resized |
| CPU offload of the drafter (`--spec-draft-device none`, `-otd .*=CPU`) | **−59 %** and worse-than-GPU respectively, against an external prediction of +70–85 % |
| lowering reasoning effort for speed | a separate axis, already set to `medium`, and it changes answer quality rather than tok/s |

## 8. What we are not asking about

**Quality.** It is this project's largest open item — every argument for
`UD-Q4_K_XL` over the smaller `UD-Q2_K_XL` rests on an external
bits-per-weight ladder, and quality has never been measured on this project's
own artifacts. It is out of scope for a speed review, and we mention it only so
you do not assume the Q4 choice is settled on evidence we hold.
