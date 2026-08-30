# NVFP4 with DFlash2 — what exists on Hugging Face, 2026-08-30

**External material. Nothing here has been measured on this machine.** Sizes,
tensor counts and flag availability were read directly from the Hub and from our
own binaries; everything about speed or quality is somebody else's claim.

Prompted by the decision the same day that **NVFP4 is the primary artifact** and
`UD-Q4_K_XL` is secondary (`OPEN-WORK-LEDGER.md`).

---

## 1. Almost every NVFP4 + DFlash2 repo is unusable here, for one reason

Eight repositories match "NVFP4 DFlash2" on the Hub. **Seven are `safetensors`**
— for vLLM or transformers — and llama.cpp cannot load them:
`dfischermittwald`, `YourHighnessLA`, `maurienne-ai`, `TH-44`, `phaseonx11`,
`nbald`, `hamichok`.

**One publisher ships GGUF, and ships both halves of the pair.**

| repo | role | bytes | contents |
|---|---|---:|---|
| `costanzopadovano/Qwen3.8-27B-NVFP4-Q8-Hybrid-Analytical-GGUF` | target | 21,560,282,304 | 1,202 tensors — 168 NVFP4, 338 Q8_0, 696 F32; plus `mmproj-F16.gguf` 927,607,488 |
| `costanzopadovano/Qwen3.8-27B-DFlash2-NVFP4-GGUF` | drafter | 1,094,346,016 | 81 tensors — 49 NVFP4, 32 F32; `architecture: dflash`, `role: speculative_draft` |

Both are one-pass conversions of pinned upstream revisions, with SHA-256 for
every file in a `MANIFEST.json`. The target's base is
`unsloth/Qwen3.8-27B-NVFP4` at `16b6615af3548b88e2d8e382457bc705b00479cf`; the
drafter's is `z-lab/Qwen3.8-27B-DFlash2` at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

## 2. The target fits this machine, and it is a FIDELITY variant

**28,593 MiB across the two cards** (12,282 + 16,311), less roughly 1,800 for
the display, leaves about **26,800 MiB**. KV at `q4_0` is 18.00 KiB/token and the
compute buffer is `2 x -ub`.

| artifact | file | at ctx 131,072, `-ub 512` | spare |
|---|---:|---:|---:|
| NVFP4 `VERY-LOW` **(served today)** | 14,174 MiB | 17,502 | ~9,300 |
| NVFP4 `MID-HIGH` (on disk, **never measured**) | 16,129 | 19,457 | ~7,300 |
| **the hybrid** | **20,561** | 23,889 | **~2,100** |

It should also reach 147,456 with about 1,800 MiB spare. **Arithmetic, not a
measurement** — and this project has twice found a configuration that fits on
paper and dies on the first real request (`CORRECTIONS.md` §35).

**Why it is interesting rather than merely bigger.** The conversion keeps **338
analytically sensitive matrices at Q8_0** instead of compressing them to NVFP4,
leaving only 168 NVFP4. It buys fidelity with 6,387 MiB. **Quality is now the
critical path here and has never been measured on any artifact** — this is an
artifact positioned exactly on that axis.

## 3. Does it carry the MTP head? Consistent with yes, not proven

This matters because our whole NVFP4 result — **+63.1 % [+58.3, +65.6]
RESOLVED** — is the *pairing* `draft-mtp,ngram-mod`, and the artifact alone is
**−22.4 %**. A target with no head cannot reproduce it.

**The base has a head, in a separate file.** `unsloth/Qwen3.8-27B-NVFP4` ships
`model.safetensors` (22,568,192,096 B) **and** `model_mtp.safetensors`
(849,400,392 B). A GGUF converter chooses whether to fold the second one in.

**Our served file demonstrably folds it in.** Read from its own header on
2026-08-30:

```
tensors = 1202   kv = 46
general.architecture        = qwen35
qwen35.block_count          = 65
qwen35.nextn_predict_layers = 1
blk range 0..64
blk.64.nextn.eh_proj.weight, blk.64.nextn.enorm.weight,
blk.64.nextn.hnorm.weight,   blk.64.nextn.shared_head_norm.weight
```

64 transformer blocks plus the MTP head as **block 64**.

**The hybrid publishes the same total: 1,202 tensors.** A conversion that
dropped `model_mtp.safetensors` would lose all of `blk.64` — its attention, its
FFN and its four `nextn` tensors — and could not land on the same count. So the
count is consistent with the head being present, **and a count in a README is
not a header**. One `--verbose` boot, or the same parse run against the
downloaded file, settles it.

## 4. Their profile — two of its three novel flags already exist in our build

From the drafter's `MANIFEST.json` (`recommended_speculation`) and README, the
QVIR-1 R2 150K profile:

```text
--spec-type ngram-mod,draft-dflash
--spec-draft-n-max 4
--spec-draft-n-min 0
--spec-draft-p-min 0.55
--spec-draft-dflash-prefill-tail 16384
--spec-draft-type-k q8_0
--spec-draft-type-v q8_0
```

with a 153,600-token context, `q8_0` target KV, and **two RTX 5060 Ti 16 GB**.

Checked against `--help` on both of our binaries, 2026-08-30:

| flag | ours |
|---|---|
| `--spec-draft-p-min` | **present**, default `0.00`. We have never set it; measured null AT 0.0 only |
| `--spec-draft-type-k/v` (`-ctkd`/`-ctvd`) | **present**. The ledger already carries these as *"corrected downward, still untried"* |
| `--spec-draft-dflash-prefill-tail` | **absent** — their fork's own addition |

**Their `n-max 4` is the value we independently measured as best**, 2026-08-30:
55.72 tok/s against 52.64 at 7, `-sm tensor` on the patched mirror
(`results/tensor-draft-depth-65536.jsonl`).

**And their upstream base is `5ecbe1ac1`** — the same DFlash2 commit our mirror
is built from — so their configuration is far likelier to run on our binary than
an arbitrary fork's would.

**Their own limitations section is worth quoting, because it is the opposite of
a sales pitch:** *"the available sample does not establish a repeatable
kernel-only speedup"*, *"the R2 tool-heavy benchmark showed high sample
variance"*, and the drafter *"should therefore be treated as a VRAM-efficient
experimental artifact, not as a guaranteed acceleration"*. The NVFP4 drafter is
only **4.3 % smaller than the Q4_K_M draft** we already run — the `Q2_K_S-MIX`
drafter already on disk is 535 MiB, far smaller.

## 5. `club-3090` discussion 1076 — a different engine, and one claim that is not ours

<https://github.com/noonghunna/club-3090/discussions/1076>

**vLLM, TP=2, two RTX 3090 (sm_86).** Six DFlash2 tiers, drafter
`syvai/Qwen3.8-27B-DFlash2-W4A16` (1.2 GB, safetensors). Reported 128–231 tok/s
on code for `dual-ultrafast`, +131 % over their `dual-fast` baseline.

**None of it transfers.** Different inference engine, different cards, tensor
parallelism rather than a tensor split, and their own caveat that CUDA 13 makes
FlashInfer's radix top-k fall back to a slower sort so the figures are *"a floor
for the config"*. Their Ampere constraint — FlashAttention and fp8 KV mutually
exclusive on `sm_86` — does not apply to `sm_89` + `sm_120`.

**The one claim that reached us as a question:**

> `SPEC_N` must match the checkpoint's `dflash_config.block_size` — published
> exports use `block_size=8`, requiring `SPEC_N=7`.

**That is vLLM's requirement, not llama.cpp's.** `common/speculative.cpp:988` in
the tree we run:

```cpp
// DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
// block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) { ... clamping ... }
```

*"at most"*, and the clamp fires only on **exceeding** it. There is no lower
bound. Every draft is verified by the target, so a smaller `n_max` proposes
fewer tokens per step, never wrong ones — and this project has the direct check:
under `-sm layer`, no-speculation and DFlash2 output are **byte-identical**
(`results/02-decoders.md`).

**Our counters agree with the source rather than with the claim.** At 7,
acceptance is **51.9**; at 4 it is **61.8**. Drafting deeper is rejected more
often while the verify cost is paid anyway, which is the whole of the 6.5 %.

**What the discussion does corroborate:** nobody in this ecosystem has quality
numbers either. They shipped six tiers with *"verify-full passes 9/9 on every
dual slug, but the 8-pack, NIAH fill depth, and soak have not been run"*.

## 6. What would have to happen next

1. **Download the hybrid target and read its header** — 21.6 GB, no GPU needed.
   Settles §3 outright.
2. **`--spec-draft-p-min` and `-ctkd`/`-ctvd`** need no download at all. Both
   flags are in the binary we already serve and neither has ever been set here.
3. **Only then a rate**, paired against `NVFP4 VERY-LOW + draft-mtp,ngram-mod`
   at one depth in one rotation — and against `MID-HIGH`, which has been on disk
   for days with no rate at all.

**Nothing here changes a default.**
