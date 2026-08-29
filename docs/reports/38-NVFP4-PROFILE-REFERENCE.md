# The NVFP4 profile — what is running, where it came from, and what is unmeasured

**2026-08-29.** Written for a reader with **no prior context on this project**.
Everything here is either a measurement with its method stated, or is labelled
as unmeasured. Nothing is a recommendation from memory.

Companion documents, if you have the repo:
`docs/reports/37-DUAL-GPU-PROFILE-REFERENCE.md` (the older `UD-Q4_K_XL` profile
this replaces for speed), `docs/results/02-decoders.md` (every decoder number),
`docs/reports/CORRECTIONS.md` (35 claims this project published and later
contradicted with its own data — §34 and §35 are from this work).

---

## 1. The machine

| | |
|---|---|
| CUDA0 | **RTX 4070 SUPER 12 GB**, sm_89, PCIe **gen4 x16**, **drives the display** (1,600–3,000 MiB at rest) |
| CUDA1 | **RTX 5060 Ti 16 GB**, sm_120 (Blackwell), PCIe **gen4 x4** under load |
| link | `PXB`, **no NVLink** |
| driver | 616.56 |
| host | Windows 11, 47.7 GB RAM |

The width is the slot, not a downtrain: the *generation* downtrains at idle, the
**width never does**. Both cards are used together; neither can hold the model
alone.

## 2. The binary

```
llama.cpp  build 10499, commit 1deefcca3   (version string 0.1.2-dev)
built MSVC 19.44 for Windows AMD64
CMAKE_CUDA_ARCHITECTURES = "89;120"   -> 141 sm_120a cubins + 141 sm_89, NO PTX
path: C:\AI\llama.cpp-blackwell\llama-server.exe
```

**Stock upstream, no patches.** A separate patched tree exists in this repo
(`llama.cpp-mirror`, for DFlash2) and is **not** used by this profile.

Both architectures matter: one card is Ada. A binary built for `89` alone runs
on the Blackwell card through PTX JIT at ~2.2× the prefill time **with nothing
in any log to say so** — this project published 15 rows off such a build before
adding a guard that reads the cubins.

## 3. The model — the full provenance chain

```
Qwen/Qwen3.8-27B                      Alibaba, Apache-2.0
   |                                  27B dense, 64 layers, Gated DeltaNet +
   |                                  Gated Attention hybrid, native VLM,
   |                                  n_ctx_train = 262,144, MTP head
   v
unsloth/Qwen3.8-27B-NVFP4             Unsloth, Apache-2.0
   |                                  compressed-tensors, MIXED precision:
   |                                    group_0 F8  -> attention (q/k/v/o,
   |                                        linear_attn), lm_head, MLP 56-63
   |                                    group_1 NVFP4 -> MLP layers 0-55
   v
esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF        <- WHAT WE RUN
                                      F8 dequantized to BF16 in place (233
                                      tensors), NVFP4 left untouched, then
                                      convert_hf_to_gguf.py --outtype auto,
                                      then llama-quantize --tensor-type-file
```

**The file we serve**

| | |
|---|---|
| repo | `esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF` |
| revision | `bcd7a7d3e251d4ec0fd15c72584b5eb9e0981383` |
| file | **`Qwen3.8-27B-NVFP4-MTP-VERY-LOW.gguf`** |
| size | 14,862,277,984 bytes = **14,174 MiB / 13.84 GiB** |
| sha256 | `74ea17ea05e0e0241af8d5b29cdea38b3f4509f66d9b96c1ab05f0e1f0e537d9` |
| tensors | 1,202 — **448 NVFP4** + 744 F32 + 10 tier-dependent |
| `output.weight` | **Q3_K** |
| `token_embd.weight` | **Q2_K** |
| MTP head `blk.64.*` | **Q2_K** |
| vision projector | `mmproj-BF16.gguf`, 931,146,432 bytes, sha256 `83ee4f…c2d53` — **byte-identical to `unsloth/Qwen3.8-27B-GGUF`'s** |

**`VERY-LOW` is the SMALLEST of nine tiers.** The seven compact tiers share a
**byte-identical 448-tensor NVFP4 backbone** (verified per-tensor SHA-256 by the
publisher) and differ only in those 10 head tensors. So **the format is not the
variable between tiers — the heads are.**

| tier | size | lm_head | token_embd | MTP head | attention |
|---|---|---|---|---|---|
| **VERY-LOW ← ours** | 14.86 GB | `Q3_K` | `Q2_K` | `Q2_K` | NVFP4 |
| COMPACT-LOW | 15.16 GB | `Q4_K` | `Q3_K` | `Q2_K` | NVFP4 |
| LOW | 15.53 GB | `Q5_0` | `IQ4_XS` | `IQ4_XS` | NVFP4 |
| MEDIUM | 16.38 GB | `Q8_0` | `Q6_K` | `IQ4_XS` | NVFP4 |
| MID-HIGH | 16.90 GB | `Q8_0` | `Q8_0` | `Q8_0` | NVFP4 |
| HIGH | 17.57 GB | `BF16` | `Q6_K` | `IQ4_XS` | NVFP4 |
| VERY-HIGH | 19.69 GB | `BF16` | `BF16` | `BF16` | NVFP4 |
| HIGHEST | 23.19 GB | `Q8_0` | `BF16` | `BF16` | **`Q8_0`** |
| ORIG | 33.13 GB | `BF16` | `BF16` | `BF16` | **`BF16`** |

**The publisher's own honest caveat, quoted:** the compact tiers re-quantize
attention **F8 → BF16 → NVFP4**, a second quantization step, and "does lose some
attention precision vs the source". `HIGHEST` and `ORIG` avoid it.

**Why NVFP4 at all on this machine.** In this build,
`ggml/src/ggml-cuda/mmq-config-blackwell.cuh` covers exactly `GGML_TYPE_MXFP4`
and `GGML_TYPE_NVFP4` and nothing else — every other type falls through to the
Ampere table. **FP4 weights are the only way to reach the Blackwell fast path.**
It is not Blackwell-only: `mmq.cuh:129` defines a "Generic NVFP4" layout, so the
Ada card runs it too via DP4A. No `-ot` tensor placement is needed.

## 4. The exact command line

Resolved by the profile's own `-WhatIf`, not transcribed from source.

```
llama-server.exe
  -m <...>\Qwen3.8-27B-NVFP4-MTP-VERY-LOW.gguf
  --alias Qwen3.8-27B-NVFP4-MTP
  -c 147456                      # or 200704 with -Deep
  -ngl auto --fit on --fit-target 768
  -fa on -np 1
  -sm tensor -ts <computed>      # see below
  -t 18 -b 2048 -ub 1024
  -lv 3 --log-colors auto
  -ctk q4_0 -ctv q4_0
  --spec-type draft-mtp,ngram-mod
  --spec-draft-n-max 3
  --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32
  --no-mmproj-auto               # replaced by  -mm <...>\mmproj-BF16.gguf  with -Vision
  --chat-template-file <repo>\qwen38-tuning\templates\qwen38-late-system.jinja
  --reasoning-effort medium
  --sse-ping-interval 5
  --host 127.0.0.1 --port 8080   # 0.0.0.0 for the -lan launchers
```

**`-ts` is computed at every launch** from `nvidia-smi` free VRAM minus a reserve
(2,500 MiB on whichever card is holding memory — that one is drawing the
desktop; 512 MiB otherwise), proportional so both cards run out together. It is
**not a constant**. Leaving it unset is not neutral: `-sm tensor` splits
**evenly** without it (`llama-model.cpp:707`), which on a 12 GB display card and
a 16 GB card left +317 MiB and produced **0.38 tok/s** — an 85× silent spill to
host memory that still returned a working server. The profile now **refuses to
launch** when the budget cannot hold the weights, because llama.cpp will not:
`llama_params_fit is not implemented for SPLIT_MODE_TENSOR`, so `--fit` is inert
here and prints `abort` about the *fitting step*, not the load.

Four launchers exist (plus `-lan` variants that bind `0.0.0.0`):
`serve-dual-nvfp4.bat` at 147,456 and `serve-dual-nvfp4-deep.bat` at 200,704.
**None is a default.** The previously served configuration
(`unsloth/Qwen3.8-27B-GGUF`, `UD-Q4_K_XL`, `ngram-mod` alone at `n-match 12`)
is still what `serve-dual.bat` runs.

## 5. What is measured

**Method, because it decides whether the numbers mean anything.** Three paired
rounds, arms rotated every round, one boot per arm, on a frozen corpus of real
vendor source (llama.cpp's `gguf-py`, written by people who have never seen this
repo) sliced to `int(ctx * 0.5)` tokens, `temperature 0, top_k 1, seed 42`,
`n_predict 512`. Rates are the **median of three samples within a boot**; the
delta is computed **per round** and never across boots. This project has a
measured drift floor of 13.6 % at ctx 16,384 that does **not** transfer to
depth — at 65,536 the same arm with byte-identical counters has spanned 48.9 %
across boots.

**The headline, ctx 147,456** (`results/nvfp4-final-147456.jsonl`):

| arm | tok/s per round | spread |
|---|---|---|
| `UD-Q4_K_XL` + `ngram-mod` n-match 12 (the incumbent) | 24.90 / 25.73 / 25.73 | 3.3 % |
| **NVFP4 VERY-LOW + `draft-mtp,ngram-mod` n-match 24** | **39.43 / 42.61 / 42.55** | 8.1 % |

**+63.1 %, per-round pairings [+58.3, +65.6].**

**Mechanism, measured separately and NOT to be multiplied together:**

- NVFP4 + `ngram-mod` **without** MTP is **−22.4 %** — a *loss*. `ngram-mod`
  acceptance falls **55.4 → 22.1** on this artifact. **Neither half is the
  result; the pairing is.**
- On NVFP4, `n-match 24` is **+27.1 %** over `n-match 12`, and `map-k` recovers
  from declining **100 %** of its drafts on `UD-Q4_K_XL` to **+15.4 %**. Both
  verdicts are the *opposite* of what the same sweep found on the other
  artifact at the same depth.
- This artifact's MTP head **does not copy the prompt**: `copied_frac
  [0.0, 0.0, 0.0]` against `[0.519, 0.0, 0.23]` for Unsloth's head at the same
  depth. A question open for weeks; the copying belongs to the artifact.
- `draft-dflash` beside it is **+0.2 % and the sign flips** — no case here.

**Depth, one boot per rung with a half-window request:**

| ctx | prompt | outcome | free after (card0 / card1) |
|---|---|---|---|
| 229,376 | 114,688 | **loads, answers `/health`, then DIES** — `cudaMalloc failed: out of memory` on device 1, having loaded with **206 MiB** free there | — |
| **200,704** | 100,352 | survived 91,428 tokens | 1,133 / **654** MiB |
| 180,224 | 90,112 | survived | 1,379 / 1,174 MiB |
| 163,840 | 81,920 | survived | 1,458 / 1,601 MiB |

229,376 was published as the ceiling earlier the same day because it survived a
**65,643-token** request — a *quarter* of its own window. Retracted
(`CORRECTIONS.md` §35). This project's measured survival line is **336 MiB free
died, 488 survived**.

**Real use, 17 requests over 18 minutes through Claude Code** (server log, one
session, `-Deep`, LAN launcher):

| context reached | prefill t/s | decode t/s |
|---|---|---|
| 401 | 275 | 49.8 |
| 8,249 | 901 | 45.9 |
| 60,603 | 825 | 36.7 |
| 73,436 | — | 33.8 |
| 51,475 (second conversation) | 319 | 25.1 |

No crash, no truncation (`truncated = 0` on all 17). **These are single unpaired
readings and are not comparable with the paired figures above** — different
prompts, different generation lengths, `n_predict` 190–2,518 rather than 512.

**And the finding that matters most from that session:**

```
ngram-mod:  #calls = 4,653   #gen drafts =     5    #acc tokens =    19
draft-mtp:  #calls = 4,648   #gen drafts = 4,648    #acc tokens = 6,612
```

**`ngram-mod` fired 5 times in 4,653 opportunities on real agent traffic.** MTP
carried everything: 47.4 % of draft tokens accepted, mean accepted length 2.42,
per-position `(0.690, 0.448, 0.284)`. The **+27.1 % that `n-match 24` bought on
the benchmark corpus does not appear in this workload** — the corpus is source
code being continued, where a 24-token exact match into context is common, and
real agent output is not. It costs nothing (15 ms total across the session), but
it is not earning either.

## 6. What is NOT measured — and this is the honest half

- **Quality. Not on this artifact, not on any artifact this project serves, ever.**
  The proposal swaps the **model file**, not a flag, and `VERY-LOW` is the tier
  with a **Q3_K LM head and Q2_K embeddings**. `ngram-mod` acceptance halving
  55.4 → 22.1 is direct evidence it *writes differently* — whether differently
  is worse is unknown. **This is why nothing has become a default.**
- **Every tier except VERY-LOW.** `MID-HIGH` was downloaded and has **no rate at
  any depth**. `COMPACT-LOW` — `Q4_K`/`Q3_K` heads for **+300 MB** — has never
  been downloaded and looks like the obvious first thing to try against
  `VERY-LOW`.
- **No depth above 147,456 has a paired rate.**
- ~~**Vision has never loaded.**~~ **MEASURED 2026-08-29 and both predictions
  were wrong.** The tower loads under `-sm tensor` on the unpatched served
  binary and answers a real image correctly at 65,536, **147,456 and 200,704** —
  the `ggml-backend-meta` wall that blocks a sidecar drafter does not apply to
  `mmproj`, and 888 MiB does fit at the deep rung. **Vision beside a large text
  prompt is measured too:** every rung took a half-window request *and* an image
  on top of it, and all four answered correctly — 200,704 finishing with **464**
  MiB free, 180,224 with 534, 163,840 with 817, 147,456 with 1,068. Both
  launcher pairs now carry it. The thin margin at the cap is the remaining
  caveat, and the budget check refusing to start is what guards it.
- **`--spec-draft-n-max 3` has now been swept on this artifact — by Unsloth
  Studio, not by us, and it is a wash.** Two of its sessions differ in that flag
  and nothing else (`--spec-type draft-mtp` alone, ports 49297 and 51604):
  **45.58 tok/s at 3 against 44.90 at 2**, 17,211 and 9,286 generated tokens.
  Acceptance per drafted token is 46.0 % at 3 and 61.5 % at 2 — position 3 is
  the one that gets refused — and tokens per target forward pass are 2.38
  against 2.23, so the extra draft eval very nearly pays for itself and no more.
  **Neither session is paired**, prompts differ, and depth moves decode far
  harder than the flag does (49 tok/s at 10K, 35 at 68K on both). Depth-matched
  turns flip sign; the deepest pair differs by **0.09 tok/s**. Keep 3, which is
  llama.cpp's own default. The publisher runs `spec_n_max 6` with
  `spec_p_min 0.75`; we set no `p_min` at all, and that is still unswept.
- **`-DisplayReserveMiB 2500` and `RUNTIME_RESERVE_MIB 768`** each come from one
  incident and were never bisected.
- **`set_sampler: backend sampling not supported with SPLIT_MODE_TENSOR; using
  CPU`** appears in every boot. `draft-mtp` announces `backend_sampling=1` and
  it is then disabled. Cost unknown; it affects **every** two-card profile here,
  not just NVFP4.
- **Host RAM: 20.4 GB working set, 34.4 GB private**, with 4.3 GB of 47.7 free.
  The log shows `context checkpoints enabled, max = 32, min spacing = 8192` and
  individual checkpoints reaching **350 MiB**, restored several times in real
  use. Whether 32 is the right number here is unexamined.

## 7. Questions worth an outside opinion

1. **Is `VERY-LOW` the wrong tier for a coding agent?** `Q3_K` lm_head and
   `Q2_K` embeddings are the cheapest heads in the family, and the LM head is
   what shapes token choice. `COMPACT-LOW` costs 300 MB for `Q4_K`/`Q3_K`;
   `MID-HIGH` costs 2.7 GB for `Q8_0` everywhere but has no measured rate here
   and would eat the depth. **What would you measure, and against what?**
2. **How do you measure quality on this without a labelled set?** The constraint
   is one machine, one developer, hours not weeks.
3. **Is the `F8 → BF16 → NVFP4` attention round trip the thing to worry about,
   or the Q2_K/Q3_K heads?** They are separable: `HIGHEST` fixes the first at
   23.19 GB, the tier ladder fixes the second.
4. **`--spec-draft-n-max` and `p_min`:** the publisher uses 6 and 0.75 on a
   highly repetitive payload; our real traffic accepts 28 % at position 3. Is
   raising `n_max` on *agent* traffic likely to pay, or does verify cost
   dominate at 50k+ context?
5. **`ngram-mod` fires 5 times in 4,653 calls on real traffic.** Is a lower
   `n-match` worth trying, or is a prompt-lookup decoder simply the wrong tool
   for agent output and the slot better spent elsewhere?
6. ~~Can a vision projector load under `-sm tensor`, and is there room for it
   beside a real prompt?~~ **Both answered: yes.** 200,704 took a 91,428-token
   request and then an image, finishing with **464 MiB** free. The narrower
   question left is whether a margin that thin is one to ship at all, or whether
   a deep profile should reserve more than the 768 MiB `--fit-target` it asks
   for — a constant set from one incident and never bisected.
7. **`-sm layer` is −31.0 % on NVFP4** (measured, three paired rounds, both arms
   `66+0`) — so the tensor split's verdict *did* transfer across the artifact,
   unlike every other verdict tested this session. **Is there a reason to expect
   split-mode verdicts to be more portable than decoder verdicts**, or is that a
   coincidence of this pair of cards?
