# 02 — Decoders (`--spec-type`)

Eleven values exist in build 10472. All eleven have been tried.

> **Every figure on this page carries the same caveat.** The timed prompt is
> **84.5 % duplicate lines** — one class repeated with a changing index, 962
> blocks at 147,456, adjacent blocks 99.5 % identical. An n-gram decoder drafts
> from context, so this is close to the best case that can be constructed for it.
> **The mechanism is real; the magnitudes are upper bounds.** Steps F1/F2
> re-measure at 73.17 % repetition. See `CORRECTIONS.md` §2.

> **And every elimination is provisional.** All were decided on **160-token**
> generations. An external review of this model reports speculation reaching rate
> only over a longer run. Step W tests 160 vs 512 vs 1024. See `CORRECTIONS.md`
> §8.

## The n-gram family — no drafter file, no VRAM, output identical

| arm | 16,384 | 131,072 | 147,456 | 163,840 | acceptance |
|---|---|---|---|---|---|
| `ngram-map-k` | **+135.89 %** | +120.54 % | — | — | 96.9–100 % |
| `ngram-mod` (short window) | +112.55 % | **+200.22 %** | **+330.40 %** | +100.48 % | 99–100 % |
| `ngram-map-k4v` (wide) | +114.64 % | +108.15 % | — | — | 83.2 % |
| `ngram-simple` | +51.31 % | — | — | — | — |
| ~~`ngram-cache`~~ | +108.49 % | — | — | — | **0 %** |

**The winner changes with depth.** `ngram-map-k` leads by 10 points at 16 K and
loses by 80 at 131,072. Use `ngram-map-k` at 16 K, `ngram-mod` at depth.

**`ngram-cache` is disqualified.** Greedy hash `3EFE93950A8A980E` against a
same-depth baseline of `04E5CAB1D14525C0` — it changes the answer, so it is not
draft-and-verify. Reported as safe in report 20 §1.1 for a day; that hash block
was typed by hand rather than read from the JSONL.

*Raw: `results/kv-decoders.jsonl`, `results/kv-ngram-fixed.jsonl`,
`results/kv-deep-*.jsonl`. Reports 20 §1, 23 §1, 24.*

## The drafter-model family — all need a file, all cost VRAM

| arm | result | why |
|---|---|---|
| `draft-mtp` | **+81 % @16K, −71 % @131,072** — re-measured 2026-08-21 and **confirmed**: 6.1–6.2 vs 45.9–48.1 tok/s at 131,072 with 467–773 MiB free, and a 1024-token run buys it only 4 % | the head is `blk.64`, 1.28 GiB on disk, **564 MiB on the GPU** (report 27). Not a VRAM artefact — report 28 |
| `draft-mtp` on CPU (`--spec-draft-device none`) | **−59 %** | external research predicted +70–85 % |
| `draft-mtp` with `-otd .*=CPU` | worse than GPU | |
| `draft-eagle3` | no usable head for this model | never produced a run |
| `draft-dflash` / DFlash 2 | cannot load on 10472. On **build 10499**: **+34.7 % [+31.1, +41.6] over `ngram-mod` on real code, RESOLVED** — and **−9.2 % on a repetitive prompt**. The verdict reverses with the prompt. Costs **1,936 MiB resident** (1.06 GB on disk) and `--fit` cannot measure it | [report 29](../reports/29-DFLASH2-AND-THE-PROMPT-THAT-FLATTERED-NGRAM.md), `results/dflash2-arena-warm.jsonl`. Issues #17, #18 |
| `draft-dflash,ngram-mod` **together** | **the best arm measured at 16,384: +48.5 % [+46.6, +50.1] over `ngram-mod` on real code, RESOLVED.** Inside the noise floor on a repetitive prompt | `--spec-type` takes a comma list — `common/arg.cpp:4155`. [report 29](../reports/29-DFLASH2-AND-THE-PROMPT-THAT-FLATTERED-NGRAM.md) |
| `draft-dspark` | tried with Ternary Bonsai | not competitive |
| `draft-simple` | needs a second full model | no room |

**The pattern, as it stood before 2026-08-22:** on a 12 GB card any drafter that
holds weights competes with the layers, and the residency cliff is steeper than
the speculation gain. The n-gram family wins here because it holds nothing.

**What changed.** That pattern was established on prompts built from repeated
blocks. `ngram-mod` drafts by matching text already in the context, so a
repetitive prompt is its best case — and every arm it beat was measured there.
On this repo's own source (4.7 % duplicate lines against the sweep prompt's
66.2 %) `ngram-mod` is worth only about **17 %** over no speculation at all,
where on the synthetic prompt it was worth **2.7×**. A drafter that holds weights
now has something to beat. [Report 29](../reports/29-DFLASH2-AND-THE-PROMPT-THAT-FLATTERED-NGRAM.md).

**Every `ngram-*` verdict in this file was set on a repetitive prompt** and is
owed a re-measurement, starting with report 20's "+200 % at 131,072".

*Raw: `results/mtp-sweep.jsonl`, `results/kv-decoders.jsonl`,
`results/spec-matrix-q*.jsonl`. Reports 20 §2, 22.*

## Interaction found 2026-08-21 — `-ot ssm` and speculation

One flag, three outcomes:

| where | acceptance | result |
|---|---|---|
| `v3-iq2xxs` @163,840, 10 blocks | **4 %** | slower than not offloading (32.4 vs 38.7) |
| `v3-iq2xxs` @163,840, 4 blocks | **no drafts at all** | level with baseline |
| `v3-iq1m` @196,608, 10 blocks | **100 %** | **+181.57 %** |

Reproduced in four boots at 4 %. Whether artifact or depth is responsible is
**unknown**, and nothing queued separates them.

*Raw: `results/kv-deep-160k.jsonl`, `results/kv-deep-192k.jsonl`. Report 24
§1, §1b.*

## The acceptance column may be worth more than the speed column

100 % on `v3-iq2xxs` and `v3-iq1m`; **37.5 %** on `v3-iq1s` — the artifact that
scores 0 of 12 on the corpus. A 30-second reading that may stand in for a
40-minute gate.

**Confounded with depth** — `v3-iq1s` was measured at 196,608 and the others
lower. Step V2 separates them. Not yet a usable signal.

## What speculation costs in VRAM — tested 2026-08-21

| question | answer | evidence |
|---|---|---|
| What does `ngram-mod` cost? | **0 MiB.** 10,763 MiB used with it and without it | report 27 |
| What does MTP cost? | **564 MiB**, with residency intact — `offloaded 66/66 layers to GPU` | report 27 |
| Can `draft-mtp` run on `UD-IQ2_S` alone? | **No.** *"model doesn't contain MTP layers"* — the weights are a separate 1.3 GB file passed with `-md` | report 27 |
| Is it worth the 564 MiB? | **Not at depth.** `draft-mtp` is +81 % at 16K and -71 % at 131,072, and the shipping 98,304 profile settles with ~400 MiB free | report 27, `CLAUDE.md` |

Raw: `qwen38-tuning/logs/mtp.err`.

## `--spec-draft-n-max` — tested 2026-08-22. The largest lever found, and window-dependent

ctx 16,384, real-code (frozen corpus), `draft-dflash,ngram-mod`, three rounds,
arms rotated, paired. Raw `results/sweep-draft-n.jsonl`, [report 31 §5](../reports/31-SESSION-RECORD-2026-08-22.md).

| `n-max` | rounds (tok/s) | vs ours | free after | acceptance |
|---:|---|---|---:|---:|
| 3 — **the default** | 70.2, 70.5, 70.2 | **−11.5 %** | 913 MiB | 53.1 % |
| 4 — what we ship | 79.3, 79.7, 79.5 | baseline | 667 MiB | 54.0 % |
| **7 — the clamp** | 97.7, 98.4, 98.2 | **+23.4 % [+23.1, +23.5] RESOLVED** | 443 MiB | 47.5 % |

**The default sits 28 % below the best point** and the help text does not say so.
The clamp is `block_size − 1` = 7 for this drafter (`common/speculative.cpp:989`);
a larger request is silently lowered.

**Acceptance falls as `n` rises while throughput climbs** — more tokens per
verify step outweighs a lower hit rate. `ngram-mod`'s mean accepted length rises
with it: 15.53 → 18.00 → 21.90.

**It is also a VRAM knob**, priced at **149.625 MiB per unit** — see
[`03-memory-and-kv.md`](03-memory-and-kv.md). **At ctx 65,536 `n=7` spills to
`63+2`** and the recurrent state splits, 49.88 MiB of it landing on the CPU.
`n=3` and `n=4` stay `65+0` there.

> **The best decoder setting depends on the window, and the window is set by the
> task.** There is no single value to ship.

## `--spec-ngram-mod-n-match` — tested 2026-08-22. We overrode the default, and it cost 26 %

ctx 16,384, real-code (frozen corpus `5672a9bcce74c0d0`), `draft-dflash,ngram-mod`,
`--spec-draft-n-max 4`, with `n-min 16` and `n-max 32` held constant. Three rounds,
arms rotated each round, paired. Raw `results/sweep-ngram-nmatch.jsonl`, 12 rows.

| `n-match` | rounds (tok/s) | vs ours | acceptance | ngram drafts | ngram mean acc len |
|---:|---|---|---:|---:|---:|
| **24 — the default** | 94.5, 96.3, 94.2 | **+34.6 % [+31.4, +40.8] RESOLVED** | 62.2 % | 29 | **23.45** |
| 16 | 69.2, 69.7, 69.5 | −1.5 % [−4.9, +3.9] within the floor | 53.5 % | 25 | 19.20 |
| 12 — what we ship | 71.7, 73.3, 66.9 | baseline | 54.0 % | 31 | 18.00 |
| 8 | 56.7, 62.7, 61.5 | **−14.5 % [−20.9, −8.0] RESOLVED** | 37.0 % | 43 | 8.95 |

**Every worker profile sets `12`; the llama.cpp default is 24 and it is 34.6 %
faster here.** The four `worker-*.ps1` scripts, `bench/kv_sweep.py` and
`dflash2_arena.NGRAM` all carry the overridden value.

**The trap held, and in both directions.** `n_match` is the hash key width
(`common/ngram-mod.cpp:15-25`), so a shorter key is a strictly weaker
requirement and ngram fires more often — 43 drafts at `8` against 29 at `24`.
It decodes *slower* anyway, because a collapsed key hands back the successor of
whichever context last wrote the slot: mean accepted length falls
**23.45 → 8.95**, ngram's own accepted-token yield falls from **651/921 to
342/1306**, and the draft calls needed to produce the same 512 tokens rise
**475 → 649**. Firing twice as often on a worse draft is a loss.

**`8` is RESOLVED by the repo's rule but not comfortably** — the mean clears the
floor and the sign is consistent, yet one round landed at −8.0 %, inside it. The
direction is solid; treat the magnitude as approximate.

⚠️ **Two limits on this row.** It was measured at `--spec-draft-n-max 4`, so the
two 2026-08-22 winners have **never run together**; and it was measured at
**ctx 16,384**, a quarter of the served window. The mechanism argues 24 should
widen its lead at depth — a fuller table means more distinct contexts colliding
on a short key — but that is a **hypothesis, not a measurement**, and this
project has been wrong about depth transfer before (`draft-mtp`, +81 % at 16K
and −71 % at 131,072).

> **Each arm's per-impl counters are byte-identical across all three rounds** —
> same calls, same drafts, same accepted length to two decimals; only the timing
> fields move. Decode is deterministic at temperature 0, so extra rounds
> re-measure the clock and nothing else. That is exactly what the pairing is
> for, but it means three rounds buy no second sample of drafter behaviour.

## The two winners crossed — tested 2026-08-22. **Do not stack them**

Same conditions as the two sweeps above, all four arms in the same rounds.
Raw `results/sweep-draft-n-x-nmatch.jsonl`, 12 rows, every arm `65+0`.

| arm | rounds (tok/s) | vs base | vs `n7-m12` | vs `n4-m24` |
|---|---|---|---|---|
| `n4 m12` — what we ship | 74.6, 75.2, 75.2 | baseline | — | — |
| `n7 m12` | 94.4, 95.1, 94.9 | **+26.4 % RESOLVED** | — | — |
| `n4 m24` | 98.4, 97.7, 97.4 | **+30.5 % RESOLVED** | — | — |
| **`n7 m24` — both** | 65.5, 63.4, 65.5 | −13.6 % | **−31.6 % [−33.3, −30.6] RESOLVED** | **−33.8 % [−35.1, −32.7] RESOLVED** |

**Both single effects replicated**, on a different day and different boot-VRAM
rolls: `n-max 7` came back +26.4 % against its earlier +23.4 %, `n-match 24`
+30.5 % against +34.6 %. So the combination arm is not a replication failure of
either half.

**Stacked, they land at 52.4 % of what independence predicts** — 64.8 tok/s
measured against 123.6 expected from 1.264 × 1.305. The combination is the
**slowest arm in the set**, below the incumbent it was supposed to beat twice
over.

### Why — the two levers push opposite sides of the same cascade

| arm | ngram drafts | ngram decline | **dflash accepted / generated** |
|---|---:|---:|---:|
| `n4 m12` | 31 | 94.3 % | 974 / 2,041 = **47.7 %** |
| `n7 m12` | 41 | 90.0 % | 775 / 2,564 = **30.2 %** |
| `n4 m24` | 29 | 93.9 % | 915 / 1,781 = **51.4 %** |
| **`n7 m24`** | **12** | **97.7 %** | 1,262 / **3,612** = **34.9 %** |

`n-match 24` makes `ngram-mod` **stricter** — it fires less often and much
better. `n-max 7` makes `draft-dflash` **longer and more expensive per call**,
8 tokens instead of 5.

Each is survivable alone. At `n4 m24` a stricter ngram is affordable because
dflash's short drafts are cheap to waste. At `n7 m12` expensive dflash drafts
are affordable because ngram still fires 41 times and covers the costly steps.
**Stacked, ngram nearly stops — 12 drafts, 97.7 % decline — and dflash pays the
full 8-token draft cost on almost every step at a 34.9 % hit rate**, generating
3,612 draft tokens to keep 1,262. That is the whole loss.

⚠️ **Why ngram's decline rises to 97.7 % is not established.** The trajectories
differ between arms, so the rates are not a clean comparison. A mechanism that
fits: `draft_one` only flushes new n-grams when `sinfo.i_last + 32 < cur_len`
and only up to `cur_len - n` (`speculative.cpp:1978-1979`), so a generation
accepting more tokens per step advances the context faster between table
updates and spends more of its time inside the blind window. **That is a
hypothesis.** It is testable against the occupancy trace at
`speculative.cpp:1950` and has not been tested.

### What to take from it

**Pick one.** `n-match 24` at `n-max 4` is the best point measured here and
also the cheaper one: it moves **no allocation at all**, while `n-max 7` costs
447 MiB of recurrent state (free after 1,272 → 825 MiB at the same `65+0`).

> **A measured win plus a measured win is not a measured win.** Both halves
> were RESOLVED against the same baseline on the same corpus, and their sum is
> worse than the thing they each beat.

**One bookkeeping note.** Against the baseline the combination reads −13.58 %
with a consistent sign across all three rounds — **0.02 points under the
13.6 % floor**, so `harness.paired_deltas` does not resolve it and this page
does not claim it. Nothing rests on that comparison: the verdict is carried by
the two against-the-singles deltas, which clear the floor by more than 17
points.

## `--spec-draft-p-min` — tested 2026-08-22. Null, and the counters say why

Same conditions, three paired rounds. Raw `results/sweep-p-min.jsonl`, 9 rows,
every arm `65+0`.

| `p-min` | rounds (tok/s) | vs base | **dflash decline** | dflash accepted / generated |
|---:|---|---|---:|---:|
| 0.00 — the default | 70.2, 76.0, 76.4 | baseline | 0.0 % | 974 / 2,041 = 47.7 % |
| 0.10 | 74.5, 76.5, 76.2 | +2.2 % [−0.3, +6.2] | **0.0 %** | 974 / 2,041 = 47.7 % |
| 0.25 | 75.2, 74.8, 75.6 | +1.5 % [−1.6, +7.1] | **2.2 %** | 1,008 / 2,026 = 49.8 % |

Both within the floor with the sign flipping. **Do not re-run it at these values.**

**The rate column is the weaker evidence here.** At `0.10` every
per-implementation counter is **byte-identical to the baseline** — same calls,
same drafts, same accepted tokens. The early-stop **never fired once**. At
`0.25` it fired on **2.2 %** of draft calls, nudged dflash's draft efficiency
from 47.7 % to 49.8 %, and bought no throughput.

**This sharpens the source bound rather than confirming it.** The read said
`1/sum ∈ [0.0625, 1.0]` by construction, so any `p_min ≤ 1/16` is identical to
`0.00`, and the arms were deliberately started at `0.10` to clear that. **They
did not clear it in practice**: the selector's confidence sits above `0.10`
essentially always on this workload, and above `0.25` on 97.8 % of calls. The
arithmetic bound was correct and still too generous — the empirical
distribution is much tighter than the algebraic one.

**Where that leaves the flag.** Its only possible gain is a narrower verify
batch — it saves *zero* draft-side compute, because the whole block is decoded
at `speculative.cpp:1195` before any check. Untested above `0.25`; the measured
trend gives no reason to expect a win, and dflash already accepts 2.91 of 5
drafted tokens, so a value aggressive enough to bite often would start
discarding tokens that would have been accepted.

> **Read the baseline's own spread before reading any delta on this page.**
> `pmin-0-base` measured 70.2 / 76.0 / 76.4 — **8.8 % across three boots of the
> same arm**, the first one on a boot with 613 MiB free against ~890 for the
> rest. That is the drift the 13.6 % floor exists to absorb, visible in one arm.

## Which speculator actually fires — tested 2026-08-22

From `common_speculative_print_stats` (LOG_TRC; our arena already ran `-lv 5`),
parsed by `harness.parse_spec_impl_stats`. Aggregated over 26 logs.

| regime | impl | calls | drafts | **decline** | mean acc len | cumulative draft ms |
|---|---|---:|---:|---:|---:|---:|
| real-code | `ngram-mod` | 4,488 | 129 | **97.1 %** | 13.65 | 6 |
| real-code | `draft-dflash` | 2,145 | 2,145 | 0.0 % | 2.85 | 12,863 |
| synthetic | `ngram-mod` | 734 | 184 | 74.9 % | 19.23 | 2 |
| synthetic | `draft-dflash` | 1,320 | 1,320 | 0.0 % | 4.65 | 8,094 |

**`ngram-mod` is not weak — it rarely fires.** On real code it declines 94–97 %
of calls, `draft-dflash` is called exactly the number of times it declines, and
when ngram *does* fire it is worth **six times more per draft**. `draft-dflash`
is also the expensive one by three orders of magnitude of draft time.

**The pooled `draft acceptance` line cannot show any of this.** With a chained
`--spec-type` it averages both speculators, and that average is what every
earlier measurement in this project read.

🔴 **The order is hardcoded and cannot be changed by a flag.**
`common/speculative.cpp:2540–2552` ranks every `ngram-*` above every model-based
type and rebuilds the list from a bitmask, discarding command-line order. So the
measured `draft-dflash,ngram-mod` **+48.5 %** ran *ngram-mod first, dflash as
fallback*. Since dflash alone beat ngram alone by **+34.7 %**, "dflash first" is
an obvious unmeasured configuration reachable only by reordering ten lines.

## The decoder verdicts re-measured — tested 2026-08-21

Two doubts stood against the eliminations. Both are now closed, and neither
rescued a decoder.

| question | answer | evidence |
|---|---|---|
| Was `draft-mtp`'s −71 % at 131,072 a VRAM collapse? | **No.** Re-run on `UD-IQ2_XXS` with **467–773 MiB free on every row** it still decodes 6.21 / 6.09 against `ngram-mod`'s 45.87 / 48.11 — 7.7x slower, reproducible to 2 % | report 28 |
| Does a long generation rescue it? (`CORRECTIONS.md` §8) | **No.** At `N_PREDICT = 1024`: `ngram-mod` 64.83 / 64.91, `draft-mtp` 54.18 / 54.08. The long run buys MTP **+4 %**, not the +47 % an external report described, and it finishes 17 % behind | report 28 |
| Is `ngram-mod` affected by generation length? | **No.** 64.83 / 64.91 at 1024 tokens against 65.06 / 60.33 at 160 | report 28 |
| Does DFlash 2 load on build 10472? | **No.** `wrong number of tensors; expected 81, got 58`, twice. llama.cpp support needs **PR #27342**; this build's `draft-dflash` is DFlash 1 | report 28, `CORRECTIONS.md` 18 |
| Does DFlash 2 load on build 10499? | **Yes.** Server reached its listening line and registered `draft-dflash` with `block_size=8`. The drafter really is DFlash **2**, not 1: `dflash.selector_top_k=16` in its GGUF, and `common/speculative.cpp:978` sets `is_dflash2 = selector_top_k > 0`. Tensor count is 81 — the number 10472 said it expected | `scripts/probe-dflash2-load.ps1` exit 0, issue #17 |
| Can `--fit` size a run that carries the DFlash2 drafter? | **No.** The fitter logs `[spec] failed to measure draft model memory: failed to create llama_context from model`, preceded by `dflash requires ctx_other to be set`. So `--fit` chooses layers **without accounting for the drafter's footprint**. On a 12 GB card whose margin at depth is ~600 MiB, that is a residency hazard, not a cosmetic warning | probe log, issue #17 |
| What is the largest usable `--spec-draft-n-max` for this drafter? | **7**, not 8. `common/speculative.cpp:989` computes `n_draft_max = block_size - 1` for dflash and clamps a larger request with a warning. Public posts quoting a block of 8 are describing the block size, not the draft cap | read from PR source |
| Is DFlash 2 worth revisiting? | **The build now exists, so the question is finally askable — and still unanswered.** Every public figure is from a bigger card: atomic.chat's 47.4→140.6 tok/s at 56 % acceptance is an RTX 6000; other results are 3090 24 GB, 5090, and a 2× 3090 tier table. The one 16 GB report reduced `n-max` to 5 and a 20 GB report to 3. The widely-quoted 381 tok/s is lookup-augmented drafting; the same post says ~133 for normal chat. This card is 12 GB and the drafter is 1.1 GB against IQ2_XXS's 1,056 MiB of returned headroom | inco.ai announcement, community posts — **none measured here** |
| Is `CORRECTIONS.md` §8 closed? | **For `draft-mtp` and `draft-dflash` only.** `draft-eagle3` never produced a run and `draft-dspark` was tried on a different model; both remain unmeasured under the long-generation rule | report 28 |

Raw: `qwen38-tuning/results/mtp-recheck.jsonl`,
`qwen38-tuning/results/step-w-long-generation.jsonl`.
