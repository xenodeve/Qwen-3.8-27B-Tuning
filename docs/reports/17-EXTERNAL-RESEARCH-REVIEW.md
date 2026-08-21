# Review of the External Research Reply — Optimization Surface Brief

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **Date:** 2026-08-20 UTC+7
> **Subject:** the reply received to
> [`../plans/02-RESEARCH-BRIEF-OPTIMIZATION-SURFACE.md`](../plans/02-RESEARCH-BRIEF-OPTIMIZATION-SURFACE.md)
>
> **Why this exists.** §4.3 of the brief listed four claims from a *previous*
> research report that cost this project real time, and asked the next one not to
> repeat them. This review records which of that reply is usable, so the same
> errors are not re-derived in three weeks. Following report 14's practice: a
> critique that only lives in a chat log gets re-derived at cost later.

---

## 0. Verdict

**One section is genuinely new and useful. The rest is generic, and the framing
rests on an arithmetic error that makes several of its recommendations
inapplicable to this machine.**

| | |
|---|---|
| Top-3 ranked questions answered | **0 of 3** |
| ⚠️ items (predicted inert) resolved | **0 of 10** |
| "What layer have I not listed?" (question 10) | **not answered** |
| Citations resolvable to a source | **none** — markers like `[17†L222-L227]` have no URL |
| Genuinely new information | **§9 decoder taxonomy** — the distinction between the five `ngram-*` variants |

---

## 1. The foundational error

> *"We assume model weights fit GPU RAM with 8-bit quant"*
> *"model size (≈13 GB for 27B Qwen3.8 FP16)"*

Both are wrong by a large factor, and the second contradicts a number that was
in the brief:

| claim | reality |
|---|---|
| 27B fits in 12 GB at 8-bit | 27B at Q8_0 is **≈27 GB**. It does not fit. This project runs **1–2 bit** artifacts for exactly this reason |
| 27B FP16 ≈ 13 GB | 27B FP16 is **≈54 GB**. Measured on disk: `UD-Q4_K_XL` alone is **16.69 GiB** |

Everything downstream that assumes headroom — "use the smallest quant that fits
accuracy needs (e.g. Q5_0)", "if VRAM is tight, offload some model parts" — is
advice for a machine that has room to spare. This one does not. Q5_0 for a 27B is
~19 GB and is not a candidate; it is also one of the KV types this project
measured as having **no fast kernel**.

**The brief stated the constraint explicitly** ("a 27B model at 2 bits is ~8–9
GiB and the KV cache at 128K is ~2–4 GiB, out of one 12 GB pool") and gave the
measured artifact table. The reply did not use it.

---

## 2. Claims that contradict measurements already in the brief

| reply's claim | our measurement |
|---|---|
| `draft-mtp` — **"none extra [VRAM]"**, "≈1.5×" | **Wrong, and it is the single most important fact in §4.2.** The MTP head's VRAM moved the split from **61+4 to 55+10** and cost **−8.8 %** on a resident target. It is not free and it is not always positive |
| decoder speedups "~1.5–2.0×" across the board | Best ever measured here: **1.47×**, and negative on resident targets. §4.3 named "2–5× speedups" as a prior report's error and this reply reproduces the shape of it with invented numbers |
| KV quantization "**–10 % tokens/s**" | Measured **+474 %** when it moves the split and **−1.6 %** when the arm already fits at 65+0. Same knob, opposite verdicts, and neither is −10 % |
| test `--gpu-layers all` vs `0` vs half | This *is* the project's central finding (13.1 / 21.8 / 41.3 tok/s), stated in §4.1 of the brief. Proposing it as a new experiment means §4 was not read |
| P-core affinity "0.6 t/s → 4.5 t/s (~8×)" | A real community anecdote, but for a **70B model that is mostly CPU-resident**. Our production arms are **65+0 — entirely on GPU**. An 8× CPU-side gain cannot apply to a configuration where the CPU does almost no decode work. *Possibly* relevant to `Q4_K_XL` at 33+32; irrelevant to everything we would ship |

---

## 3. Factual errors about the runtime

Checked against `llama-server --help` on build 10472 `60eeeb608`:

| reply | actual |
|---|---|
| `-sps 1` = "slot 1", used with `--cache-prompt` to keep last prompt | `-sps` is **`--slot-prompt-similarity`**, a float, default **0.10**, "how much the prompt of a request must match the prompt of a slot". Not a slot index |
| "enable `--cache-prompt` to reuse context" | `--cache-prompt` is **enabled by default** in this build |
| `--cache-dir` to store GGUF for faster load | **No such flag** in this build |
| "`--flash-attn` default on CUDA 13+"; "ensure using CUDA 13.3+" | This build links **cudart64_12 / cublas64_12** — CUDA 12. `-fa` default is `auto` regardless |
| `--repeat_penalty`, `--mirostat 2 5.0` | flags are `--repeat-penalty` and `--mirostat N` with separate `--mirostat-lr` / `--mirostat-ent`. Underscore forms are not accepted |
| "Qwen3.8-30B-UD" as a model variant to compare | **No such artifact.** Not on the Unsloth repo tree, not anywhere this project has seen |
| "`--ctx-checkpoints` default 32 @8K; RAM 50→20 GB (example)" | The **32** and the **8192-token spacing** (`--checkpoint-min-step`) are correct. The RAM figures are invented and cite nothing |
| "`--tensor-split` to split memory if model exceeds GPU" | `-ts` splits across **multiple GPUs**. There is one GPU. It does not do what is described |

---

## 4. Questions that were asked and not answered

| # | question | what came back |
|---|---|---|
| 1 | **A working GBNF grammar** for "exactly one fenced Python block", its throughput cost, and its interaction with speculative decoding | "Create a simple grammar (e.g. force ``` fences)". No grammar, no cost, no interaction |
| 2 | **DRY vs repeat-penalty vs top-n-sigma vs Mirostat** against runaway reasoning, for code | A generic sampler table. Does not compare them for this failure, and does not address whether repetition penalties damage code (which is legitimately repetitive) |
| 3 | **DFlash 2 / llama.cpp PR #27342** — merged? prebuilt binary? will stock `draft-dflash` load a DFlash 2 GGUF? | **Not mentioned at all.** The reply describes `draft-dflash` generically and never addresses the version question the brief was specific about |
| 4 | `--override-tensor` **patterns** for attention-on-GPU / FFN-on-CPU | "offloading some FFN or MoE weights could allow larger context". No pattern syntax, no example, no data |
| 5 | `--context-shift` + `--keep` semantics | Invented a "manual sliding window by restarting each turn" instead. The flags were not discussed |
| 6 | `--cache-reuse` — what KV shifting can salvage from a mid-stream edit | Not addressed. `--cache-reuse` is named once in a table with no explanation |
| 7 | `--ctx-checkpoints` **VRAM cost** at 128K–256K | Gave uncited **system RAM** figures. The question was VRAM |
| 8 | Windows **P/E-core numbering** and how to discover it; separate masks for prefill vs decode | "Use Task Manager or PowerShell". No numbering method. The prefill-vs-decode split — which the runtime has separate flags for — is not mentioned |
| 9 | What changed in llama.cpp **since `60eeeb608`** | "many updates… a recent commit fixed MTP issues". No commit, no release, no date |
| 10 | **What layer have I not listed?** | Not answered |

---

## 5. What is worth keeping

### 5.1 The `ngram-*` taxonomy — the one genuinely new contribution

The brief asked what distinguishes the five n-gram decoders. This is the only
part of the reply that contains information not already on this machine:

| decoder | described as |
|---|---|
| `ngram-simple` | find the last matching n-gram in context, append the following tokens — trivial hash lookup |
| `ngram-map-k` | find n-grams repeating above a frequency threshold, append an m-gram |
| `ngram-map-k4v` | as map-k, but stores up to **4 candidate continuations** and picks the most frequent |
| `ngram-cache` | lookup based on **probabilities** of short n-grams |
| `ngram-mod` | **rolling-hash**, ~16 MB fixed table, constant-time fetch |

**Corroboration:** the 16 MB figure for `ngram-mod` matches an independent
statement in `Candidate Inference Configurations…md` line 272 ("`ngram-mod` docs
describe ~**16 MB** constant structure"). Two independent sources agreeing is
weak but real evidence.

**Why it changes what to run:** report 16 listed the three unrun n-gram decoders
as equally cheap. They are not equally *suited*. If `map-k4v` keeps four
candidate continuations, it is the variant built for a context where the same
prefix legitimately continues several ways — which is what a file being edited
looks like. And `ngram-simple`'s measured 31 % acceptance was at defaults with a
**cold** table, which the taxonomy explains.

**Status: plausible, uncited, and cheap to verify directly** — one boot each.

### 5.2 Two claims worth testing rather than believing

- **P-core affinity.** The 8× anecdote does not transfer to a resident arm, but
  the *direction* is worth one boot on `Q4_K_XL` at 33+32, the one arm where the
  CPU does real decode work. Report 16 predicted "unknown and asymmetric"; this
  does not resolve it, but it raises the prior.
- **Mirostat.** The description ("adaptive top-k to hit a target perplexity,
  avoiding both repetition and incoherence") is accurate and is the mechanism
  most directly aimed at the runaway-reasoning failure. Still no evidence for
  *code*, which is what was asked.

---

## 6. What this changes about how to ask

The brief already contained everything the reply got wrong: the VRAM arithmetic,
the MTP result, the KV result, the residency table, and an explicit instruction
not to re-propose settled sweeps. Restating constraints harder is therefore not
the fix.

Three changes for the next request:

1. **Ask one question, not sixteen.** The single most valuable unanswered item is
   a working GBNF grammar plus its cost. Asked alone, with the failing output
   pasted verbatim, it is answerable and checkable.
2. **Demand a resolvable URL per claim** and reject the answer without one.
   Every citation in this reply was an internal marker.
3. **Ask for the mechanism, not the number.** Invented speedup multipliers were
   the failure mode of the previous report too (§4.3). A described mechanism can
   be tested locally; a fabricated "~1.8×" cannot be distinguished from a real
   one until a boot is spent.

---

## 7. Net effect on the queue

| item | change |
|---|---|
| `ngram-map-k4v` | **promoted** — the taxonomy suggests it fits code editing better than `ngram-simple`; still one boot |
| `ngram-simple` re-test with a warm table / tuned `--spec-ngram-simple-*` | **added** — the 31 % acceptance may be a cold-table artifact |
| P-core affinity on `Q4_K_XL` (33+32) | **added, low priority** — the only arm where it could matter |
| everything else in report 16 §17 | **unchanged.** Nothing in the reply justified reordering it |
| DFlash 2 / PR #27342 | **still open** — must be settled locally, by trying the stock `draft-dflash` loader against the downloaded DFlash 2 GGUF |
