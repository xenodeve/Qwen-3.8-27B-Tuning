# 08 — what actually transferred from the RTX 3090 stack

**The register for one question: of the 434 techniques scanned out of
`syv-ai/qwen38-27b-rtx3090`, which ones were tried here, and what happened.**

Source of the pool: [`../researchs/syv-rtx3090/`](../researchs/syv-rtx3090/README.md)
— 434 techniques matched against a 175-capability map of our own llama.cpp, of
which **48 were "a flag we already have and have never set"**.

Flag semantics for the shortlist:
[`../researchs/llamacpp-flag-semantics-2026-08-22.md`](../researchs/llamacpp-flag-semantics-2026-08-22.md).

> **Read the status column literally.** *Measured* means a paired run on this
> card. *Read* means an argument from source with no run behind it. The two are
> not interchangeable and this project has confused them before.

---

## The scoreboard

| # | technique (their name) | our lever | status | outcome |
|---|---|---|---|---|
| 1 | "k=4 is the knee" / draft count reduced at long context | `--spec-draft-n-max` | **MEASURED** | **+23.4 % [+23.1, +23.5] RESOLVED at ctx 16,384.** Collapses at 65,536 |
| 2 | "NSTRONG=NMIN: take any qualifying match" | `--spec-ngram-mod-n-min` | **MEASURED** | **No effect.** 16/8/4/2 → 79.7/79.7/79.7/79.8 |
| 3 | "NMAX=12 chosen against 32 so recency beats length" | `--spec-ngram-mod-n-match` | **MEASURED** | **Refuted — shorter is worse.** The default `24` is **+34.6 % [+31.4, +40.8] RESOLVED** over the `12` we ship |
| 4 | Lookup hit counter | `-lv 4` per-impl statistics | **USED** | the finding under every other finding — see below |
| 5 | "A longer verify block costs KV pool per slot" | `n_rs_seq` | **USED** | gave the exact price: `149.625 MiB × (1 + n_max)` |
| 6 | Applying top-k/top-p to the selector proposal | `--spec-draft-p-min` | **READ** | sweepable, but **≤ 0.0625 is identical to 0.00** |
| 7 | Weight-aware pool budget | `-fitt` | **READ** | and it does not mean what our profiles say — below |
| 8 | V2-runner fix: drafter's KV forced back to bf16 | `-ctkd` / `-ctvd` | **READ — do not sweep** | saving ~32 MiB, paid for on the hot path |
| 9 | "Complete tunable surface: six environment variables" | `GGML_CUDA_GRAPH_OPT=1` | **READ — do not sweep** | cannot fire on our shape |
| 10 | FlashInfer radix top-k with a latching fallback | `-bs` | **READ — do not sweep** | offloads to sampler 2 of 10; dies on a grammar |
| 11 | DFlash2 + context lookup ("their most important idea") | `--spec-type draft-dflash,ngram-mod` | **MEASURED** | **+48.5 % over `ngram-mod` alone on real code** |
| 12 | Quantised DFlash2 drafter (3.85 GB bf16 → 1.19 GB W4A16) | — | **ALREADY HAD IT** | our GGUF is 1.06 GB and was never bf16 |
| 13 | Adaptive verify block beyond the trained block size | — | **IMPOSSIBLE** | `speculative.cpp:989` clamps at `block_size − 1` = 7 |
| 14 | KVarN 4-bit K / 2-bit V | — | **IMPOSSIBLE** | llama.cpp's KV types bottom out at 4 bits |
| 15 | Hybrid recurrent-state prefix cache | `--cache-reuse` / `-sps` | **OPEN** | whether llama.cpp restores DeltaNet state or only KV is unknown |

**Three measured wins — one of them by refuting the claim that produced it —
one measured null, five read-and-closed, two we already had, two impossible,
one open.**

**The refutation is one of the wins.** Their claim was that a *shorter* match
predicts better. Measured here the direction reverses, and acting on the claim
would have made the served profile slower still. A transfer that tells you to
undo something you already did is worth the same round as one that tells you to
add something.

---

## 1. `--spec-draft-n-max` — the transfer that paid

Two agents in the scan independently called it *"the biggest single unclaimed
win on this list"*. It was, and by more than they predicted.

| `n-max` | ctx 16,384, real-code | vs ours |
|---:|---|---|
| 3 — **the llama.cpp default** | 70.2, 70.5, 70.2 | **−11.5 %** |
| 4 — what we shipped | 79.3, 79.7, 79.5 | baseline |
| **7 — the clamp** | 97.7, 98.4, 98.2 | **+23.4 % RESOLVED** |

**The default sits 28 % below the best point** and the help text does not say so.
Full data and the VRAM price in
[`02-decoders.md`](02-decoders.md) and [report 31 §5](../reports/31-SESSION-RECORD-2026-08-22.md).

**It does not transfer up in depth.** At ctx 65,536, `n=7` loads `63+2` — two
layers on the CPU — and the recurrent state splits with 49.88 MiB landing on the
host. **There is no single value to ship: the best setting depends on the window,
and the window is set by the task.**

## 2. `--spec-ngram-mod-n-match` — their claim reversed, and it found our own error

The scan read their patch as *"NMAX=12 chosen against 32 so recency beats
length"*, and [report 30](../reports/30-SYV-RTX3090-REFERENCE-REVIEW.md) noted
approvingly that our profile already ran `12` — **"the same cap, chosen
independently"**. Convergence read as confirmation.

Measured, the direction reverses:

| `n-match` | ctx 16,384, real-code | vs ours | ngram drafts | ngram mean acc len |
|---:|---|---|---:|---:|
| **24 — the llama.cpp default** | 94.5, 96.3, 94.2 | **+34.6 % RESOLVED** | 29 | **23.45** |
| 16 | 69.2, 69.7, 69.5 | −1.5 %, within the floor | 25 | 19.20 |
| 12 — what we ship | 71.7, 73.3, 66.9 | baseline | 31 | 18.00 |
| 8 | 56.7, 62.7, 61.5 | **−14.5 % RESOLVED** | 43 | 8.95 |

**The trap the sweep was designed around is what produced the result.** A
shorter key is a strictly weaker requirement, so ngram fires more often — 43
drafts at `8` against 29 at `24` — and decodes slower anyway, because the
collapsed key returns the successor of whichever context last wrote the slot.
Mean accepted length falls **23.45 → 8.95** and the draft calls needed for the
same 512 tokens rise **475 → 649**. **Firing twice as often on a worse draft is
a loss**, which is exactly what `speculative.cpp:2545` sitting six lines above
`2551` predicts.

**Their number was not wrong, it was theirs.** Their cap is a bound on a
longest-match search with recency tie-breaks; ours is a hash key width into a
keyless table with no length dimension at all
(`common/ngram-mod.cpp:15-25, 37-41`). The two flags share a number and nothing
else — which the flag-semantics read had already said, and which the "chosen
independently" line in report 30 walked past.

Full data, the two limits on it, and the determinism note:
[`02-decoders.md`](02-decoders.md). Retraction:
[`CORRECTIONS.md` §21](../reports/CORRECTIONS.md).

## 3. The lookup counter — the instrument that made everything else visible

The scan called this *"strictly better than what vLLM built, and switched off"*.
It was switched off in the sense that nobody read it: `-lv 5` was already in our
arena, so the data existed before the question did.

It is why we know `ngram-mod` **declines 94–97 % of the calls it receives on real
code** while being worth **6× more per draft** when it does fire, and that
`draft-dflash` is called exactly the number of times ngram declines. The pooled
`draft acceptance` line cannot show any of that, and the pooled line is what every
earlier measurement in this project read.

**Nothing else on this page would have been findable without it.**

## 4. The two "do not sweep" verdicts that saved GPU rounds

Worth as much as a win, and each rests on a specific line:

- **`-ctkd`/`-ctvd`.** The drafter decodes `n_max + 1` = 5 tokens in **one**
  `llama_decode` (`speculative.cpp:1183, 1196`), so `Q->ne[1] == 5` and a
  quantised draft KV takes `MMA_F16` with a full dequantisation per layer per
  step — not the vector kernel. ~32 MiB saved, paid for on the hot path.
- **`GGML_CUDA_GRAPH_OPT`.** Its body (`ggml-cuda.cu:4339-4551`) contains **no
  `cudaGraph*` call at all**. "Our logs show CUDA graphs are on" is not evidence
  about this flag.
- **`-bs`.** Offloads the longest *prefix* of the sampler chain, which with our
  `--samplers` ends at position 2 of 10; `penalties` refuses outright
  (`llama-sampler.cpp:3018`), and it self-disables on a grammar — which the
  served profile needs.

## 5. `--spec-draft-p-min` — the arithmetic that redesigned the sweep

The scan called it *"the single most actionable item in this slice"*. It may be,
but not at the values anyone would try first.

The greedy check (`speculative.cpp:1264-1268`) compares `1/sum`, where
`sum = Σ_{k=0..15} exp(scores[k] − scores[argmax])` over the **16** selector
candidates. The argmax term is exactly 1 and every other term is ≤ 1, so:

> **`1/sum ∈ [0.0625, 1.0]`. Any `p_min ≤ 1/16` is mathematically identical to
> 0.00.**

A ladder starting at 0.05 would have repeated the `--spec-ngram-mod-n-min` error
exactly: four arms that cannot differ. The arm set in `dflash2_arena.ARM_SETS`
now starts at 0.10 for this reason.

It also saves **zero draft-side compute** — the whole block is decoded at
`speculative.cpp:1195` before any check — and `ngram-mod` outranks
`draft-dflash`, so every step ngram serves is a step where it does nothing.

## 6. `-fitt` — the transfer that found an error in our own profiles

`tools/server/server-context.cpp:1074` **adds the draft model's bytes** to
`fit_params_target` before `--fit` runs. With the 1,090 MiB sidecar, our
`--fit-target 768` reaches `fit.cpp` as roughly **1,900–2,100 MiB**.

Every worker-profile header describing 768 as "the margin the server leaves free"
is wrong in the configuration we now serve — and it explains why the drafter
fitted at ctx 98,304 when arithmetic from the raw buffer sizes said it should not.

---

## The thing this page most nearly got wrong

Two RESOLVED wins on the same baseline, the same corpus, the same binary. The
obvious next act is to ship both. **Measured, they cancel:**

| arm | ctx 16,384, real-code | vs the better single |
|---|---|---|
| `n-max 7`, `n-match 12` | 94.4, 95.1, 94.9 | — |
| `n-match 24`, `n-max 4` | 98.4, 97.7, 97.4 | — |
| **both** | 65.5, 63.4, 65.5 | **−33.8 % [−35.1, −32.7] RESOLVED** |

Both singles **replicated** in these rounds (+26.4 % and +30.5 % against their
earlier +23.4 % and +34.6 %), so this is an interaction and not a failure of
either half. Stacked they reach **52.4 % of what independence predicts**, and
the combination is the slowest arm in the set.

The mechanism is the cascade both levers act on. `n-match 24` makes `ngram-mod`
stricter; `n-max 7` makes `draft-dflash` longer and dearer per call. Alone each
is survivable. Together **ngram nearly stops firing — 12 drafts, 97.7 %
decline — and dflash pays a full 8-token draft on almost every step at a 34.9 %
hit rate**, generating 3,612 draft tokens to keep 1,262.
[`02-decoders.md`](02-decoders.md) carries the counters and the one unverified
step in the explanation.

> **A measured win plus a measured win is not a measured win.** Nothing on this
> scoreboard licenses adding two rows together — every row is a paired
> comparison against one baseline, and that is all it is.

---

## What did NOT transfer, and why

**The stack.** vLLM 0.27.1 with W4A16 safetensors. A 27B W4A16 checkpoint is
16–19 GB before their optimisations; 12 GB does not hold it.

**The adaptive verify block.** Presented as an idea to import; it is a tuning of
kernels we do not run. Its sweet spots of 16 and 21 query tokens are GPTQ-Marlin
tiling the M dimension in 16 rows and their own attention patch fitting
`128 // 6 = 21` rows per tile. llama.cpp clamps at 7 regardless.

**KVarN.** 5,767 lines, the largest directory in their repo, and unreachable:
llama.cpp's KV types bottom out at 4 bits and we already run that floor.

**Their numbers.** An RTX 3090 24 GB says nothing directly about a 12 GB card —
except for one comparison that does travel, and it is the most useful thing in
the whole review: **their no-speculation baseline is 46 tok/s and ours is ~44.**
The gap is entirely in the speculation stack, not the card and not the
quantisation. [Report 30](../reports/30-SYV-RTX3090-REFERENCE-REVIEW.md).

---

## Still open from this pool

- **Neither winner has been measured at the served depth.** Both verdicts are
  ctx 16,384; `worker-iq2s-quality.ps1` serves 98,304. `--spec-draft-n-max 7`
  is already known **not** to transfer — it spills to `63+2` at 65,536 — so the
  question is live for `n-match` too, and `n-match` costs no VRAM, which makes
  it the cheaper of the two to answer. **Answer `n-match 24` alone**, not the
  pair — see below.
- **#6 `--spec-draft-p-min`** at 0.10 / 0.25, arms defined and unrun.
- **#7 `-fitt`** — a step function with a dead zone whose step moves with boot
  VRAM. Read the fitted configuration from the log before measuring any rate.
- **#15 recurrent-state prefix reuse** — their `PREFIX_CACHE=1` took turn 2 of a
  24K chat from ~23 s to 1.15 s, and a 100K prefix from 169 s cold to 4.7 s.
  Whether llama.cpp's `--cache-reuse` restores DeltaNet state or only KV is
  **unknown here and not answered by their repo**. The single largest untested
  idea left in the pool.
- **The speculator order.** `speculative.cpp:2540-2552` hardcodes every `ngram-*`
  above every model-based type and discards command-line order, so our measured
  `draft-dflash,ngram-mod` **+48.5 %** ran *ngram first*. Since dflash alone beat
  ngram alone by **+34.7 %**, "dflash first" is unmeasured and reachable only by
  reordering ten lines.
