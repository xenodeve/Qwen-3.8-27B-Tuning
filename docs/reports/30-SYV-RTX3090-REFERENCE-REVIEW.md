# 30 — `syv-ai/qwen38-27b-rtx3090`, checked against our own files

**2026-08-22.** Reviewed at `--depth 1`, 14.6 MB, in the session scratchpad. Not
vendored: it is a vLLM stack and nothing in it is loadable here.

The repo is real and unusually careful — 14 patches, a gotchas document that
explains *why* each number is what it is, and benchmark tables with the
sampling mode stated. This review records what was **verified against our own
files or its source**, what does not transfer, and the one number that changes
how we should read our own results.

---

## The finding that matters most, and it was not in the summary

Their **no-speculation baseline is 46 tok/s** on an RTX 3090 24 GB running a
heavily patched vLLM with a W4A16 checkpoint (`README.md:13`, "single-stream
(C1) decode rate, realistic prompts").

Ours, measured today at ctx 16,384 on real code, `none` arm:
**45.4, 45.4, 41.8 tok/s** (report 29).

**The baselines match.** A 24 GB card, a different serving engine, a different
quantization family and nine layers of custom optimization produce
approximately the decode rate a 12 GB 4070 SUPER produces on llama.cpp with
`UD-IQ2_XXS`.

Where they pull ahead is speculation, and only there:

| | theirs (3090, vLLM) | ours (4070S, llama.cpp) |
|---|---:|---:|
| no speculation | 46 | ~44 |
| best speculation | **131** (DFlash2, greedy) | **79** (`draft-dflash,ngram-mod`) |
| ratio over baseline | **2.8×** | **1.8×** |

**So the headroom we are missing is in the speculation stack, not in the card
and not in the quantization.** That is the useful reading, and it is cheaper to
act on than either of the other two.

*Caveat: their figure is at 64K and ours at 16,384, and "realistic prompts" is
their phrase, not a measured repetition percentage. The two numbers are close
enough to be worth noticing and not close enough to be a controlled comparison.*

---

## Verified against our own files

### The drafter sees 2,048 tokens, and our GGUF says so too

`docs/optimizations.md:102,172,259` states the block drafter reads a
2,048-token window. **Confirmed independently from the checkpoint we already
have:**

```text
dflash.attention.sliding_window         = 2048
dflash.attention.sliding_window_pattern = [True, True, True, True, True]
dflash.context_length                   = 262144
```

All five drafter layers slide. `context_length = 262144` is the *model's*
declared context, not what the drafter attends to.

**This reads differently than the summary suggested.** It is not that DFlash2
degrades as context grows — it is that DFlash2's view **does not grow at all**.
`ngram-mod` matches against the whole history. The two are complementary in
**range** as well as in mechanism, which is a structural reason for the
combination winning rather than a coincidence, and it predicts the pairing
should hold up at depth.

### Their "lookup drafting" is `ngram-mod`

`patches/dflash2-lookup-drafting.patch`, 947 lines: find an earlier occurrence
of the current suffix in the request's own context and propose that
continuation, gated by `VLLM_DFLASH2_LOOKUP`, `LOOKUP_NMIN=6`, longest match
with ties broken by recency, **match length capped at 12**.

llama.cpp's `ngram-mod` is the same algorithm, and our tuned profile already
uses **`--spec-ngram-mod-n-match 12`** — the same cap, chosen independently.

So the summary's "most important idea to import" is one **we ran and measured
today**: `--spec-type draft-dflash,ngram-mod`, **+48.5 % [+46.6, +50.1] over
`ngram-mod` alone on real code, RESOLVED** (report 29). They needed a 947-line
patch; llama.cpp takes a comma.

### The drafter quantization is already done for us

`drafter/README.md:67` — the DFlash2 drafter is 1.92B params, **3.85 GB in
bf16**, read once per decode step, so they requantize to W4A16 Marlin at
**1.19 GB**.

**We never had the bf16 problem.** `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` is
**1.06 GB on disk**, already quantized, and it is what report 29 measured.

The number to keep is the *resident* one: we measured **1,936 MiB**, a factor
of 1.79 over the file. Their 5 ms/step read cost is a property of their kernel
and does not transfer.

---

## What does not transfer, and why

**The stack.** vLLM 0.27.1 with W4A16 AutoRound safetensors, not GGUF. A 27B
W4A16 checkpoint is 16–19 GB before their optimizations; 12 GB does not hold
it. `docker compose --profile single up -d` is not a shortcut for us.

**The adaptive verify block — and this is where the summary should be
corrected.** It was presented as an idea to import. It is a tuning of kernels
we do not run:

- Their sweet spots are **16 and 21 query tokens**, and `docs/gotchas.md:158`
  gives the reason: GPTQ-Marlin tiles the M dimension in **16 rows**, so a 17th
  query token buys a second M block in all 64 layers; and their
  `SpecDecodeAttention._plan` puts `q_len × G` rows in a **128-row tile**, so
  with this model's `G = 24/4 = 6` one tile holds `128 // 6 = 21`.
- Both constants are properties of Marlin and of their own attention patch.
  Neither means anything to llama.cpp's CUDA kernels.
- And llama.cpp **cannot do it anyway**: `common/speculative.cpp:989` computes
  `n_draft_max = block_size - 1` = **7** and clamps a larger request with a
  warning. We verified that reading the source, before any of this.

**It also costs VRAM per slot, not per token.** `docs/gotchas.md:130` —
`DFLASH_TOKENS=31` with 8 slots wants **5.3 GiB** of recurrent-state pages
before a single token of context, and their single-user mode drops to 4 slots
to afford a long block. On a 12 GB card that is the whole argument.

**The 381 tok/s figure.** `README.md:14` states its conditions plainly: **25K
context, reproducing its own context** (quoting a document, applying an edit),
`SPEC=dflash2` + `DFLASH_TOKENS=15`, 15.0 tokens accepted per verify step. It
is a best case for a copy workload, and their own table gives 122–131 for
ordinary generation. Quoting 381 as DFlash2's speed is the same error as
quoting our 91.6 % acceptance from a 66.2 %-duplicate prompt.

## The largest directory, which the summary skipped entirely

`kvarn/` is **5,767 lines — the biggest thing in the repo**, and none of the
summary mentioned it.

It is a port of [KVarN](https://github.com/huawei-csl/KVarN) (Huawei CSL,
Apache-2.0) onto vLLM 0.27.1: a KV-cache compression scheme using Hadamard
rotation and iterative variance normalization to reach **4-bit keys and 2-bit
values per 128-token tile**, shipped as a native vLLM attention backend.

It is relevant because **KV cache is what competes with the drafter for VRAM**
at depth, and it is exactly the axis this project keeps running out of.

**And it is not reachable here.** Build 10499's own help:

```text
-ctk, --cache-type-k TYPE
      allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
```

The floor is **4 bits**. There is no 2-bit KV type in llama.cpp, so KVarN's
value compression has no equivalent short of writing an attention backend —
far outside this project. We already run `q4_0/q4_0`, which is the floor of
what exists.

### The lever that *was* in the same help output

```text
--spec-draft-type-k, -ctkd, --cache-type-k-draft TYPE   (default: f16)
--spec-draft-type-v, -ctvd, --cache-type-v-draft TYPE   (default: f16)
```

**The drafter has its own KV cache, with its own type, defaulting to `f16` —
and no profile or benchmark here has ever set it.** Every DFlash2 number in
report 29 was measured with the target at `q4_0` and the drafter at `f16`.

The drafter's KV is bounded by its 2,048-token sliding window, so this is not
gigabytes. But it is four bits per element against sixteen on a component
measured at **1,936 MiB resident**, it costs one flag to test, and it has never
been tried. 🔴 **Untested.**

---

---

## What is worth taking

| idea | status here |
|---|---|
| DFlash2 + context lookup | **already done and measured** — +48.5 % on real code, report 29 |
| Quantized drafter | **already had it** — our GGUF is 1.06 GB, never bf16 |
| Adaptive verify block > trained block size | **not possible in llama.cpp** (`speculative.cpp:989` clamps at 7), and its tuning constants are Marlin-specific |
| **Recurrent-state prefix cache** | **the one genuinely open idea.** Qwen3.8 is a hybrid: 48 Gated DeltaNet layers whose recurrent state is separate from KV. Their `PREFIX_CACHE=1` caches both, taking turn 2 of a 24K chat from ~23 s to **1.15 s**, and a 100K prefix from 169 s cold to **4.7 s**. Whether llama.cpp's `--cache-reuse` / `--slot-prompt-similarity` restores recurrent state or only KV is **unknown here and not answered by this repo** |
| KVarN 4-bit K / 2-bit V | **not reachable** — llama.cpp's KV types bottom out at 4 bits (`q4_0`, `q4_1`, `iq4_nl`); we already run the floor |
| **`-ctkd` / `-ctvd`, the drafter's own KV type** | 🔴 **never set.** Defaults to `f16` while the target runs `q4_0`. One flag, untested |
| Prefill expectations | **confirms ours is normal.** They report ~1,000 tok/s at 100K on a 3090 with a tuned stack; we measure ~900. Report 27's "prefill cannot be tuned here" stands, and now has an outside data point |

---

## What this review does not establish

Nothing in it was measured on this machine. Every number attributed to them is
read from their README, their `docs/`, or their patches; the two things checked
against our own files are the drafter's sliding window and the drafter's size,
both stated as such above.

The baseline comparison in §1 is the strongest claim here and it is still a
comparison of two numbers taken at different context depths under different
definitions of "realistic prompt". It is a reason to look at the speculation
layer first. It is not a measurement.
