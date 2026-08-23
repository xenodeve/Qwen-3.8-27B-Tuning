# 02 — Decoders (`--spec-type`)

Eleven values exist in build 10472. All eleven have been tried.

> 🔴 **Every arm on this page was measured on the RTX 4070 SUPER 12 GB, and the
> decoder ranking has since changed on the card that is installed.** Re-measured
> at ctx 98,304 on the RTX 5060 Ti 16 GB, three rounds
> (`results/decoders-98304-blackwell.jsonl`): **`dflash2+ngram` went from a
> median of 5.66 tok/s with two timeouts in six rounds to 87.72 with none, and is
> now the fastest arm** — ahead of the `ngram-mod` every worker profile serves.
> Nothing about the drafter changed; it stopped being squeezed into the
> **45–376 MiB** band that [`CORRECTIONS.md` §26](../reports/CORRECTIONS.md)
> identified, and finishes with 2,842–3,183 MiB here.
> **Read [`09-hardware.md`](09-hardware.md) before quoting any elimination on
> this page as current.**

## `dflash2+ngram` on a REAL task — first measurement, 2026-08-24

Every other number on this page is tok/s on a generated prompt. This is one run
of `bench/real_task_bench.py` against a real open issue (`xeno-skills#306`) in a
throwaway clone, ctx 98,304, on the native `sm_120a` build.
**`results/real-task-dflash2ngram.jsonl`**, transcript preserved at
`D:\bench-scratch\transcripts\xeno-skills-306-20260824-014053.stdout.txt`.

**The server side is healthy.** Peak context **69,401 of 98,304** (70.6 %),
`truncated = 0` on every turn, 45 turns across the session, no timeout.
**The window was not the constraint and neither was the drafter.**

**Speculation works on real work, which the synthetic corpus could not show.**
Acceptance on the corpus is 0.614; here it ran **0.47–0.65**, same band.

**But tok/s is not one number — it tracks turn length, and agent turns are
short:**

| generated tokens | tok/s | acceptance | mean accepted len |
|---:|---:|---:|---:|
| 324 | **19.68** | 0.479 | 3.47 |
| 698 | 33.16 | 0.470 | 3.69 |
| 8,192 *(hit the cap)* | **62.85** | 0.654 | 7.18 |

**87.72 tok/s — this arm's corpus figure — is not what an agent loop gets.**
A short turn is dominated by per-request overhead, and a coding agent is mostly
short turns. Anywhere this project quotes a decoder rate as what the worker
delivers, that gap applies.

### The task itself FAILED, for a reason that is not about the decoder

`changed_files = 0` after 537.7 s. The preserved transcript says why, and it is
the shell:

```
$ ls -la
Get-ChildItem: A parameter cannot be found that matches parameter name 'la'.

$ ls -la . 2>/dev/null || dir /B
Out-File: Could not find a part of the path 'D:\dev\null'.
Get-ChildItem: Cannot find path 'D:\B' because it does not exist.
```

The worker emitted POSIX commands into PowerShell, spent its opening turns on
that, recovered into correct cmdlets, then explored the repository for nine
minutes and never reached an edit. **This is a worker/environment result, not a
decoder result.**

> ⚠️ **One task with no control proves nothing about the arm.** There is no
> `ngram-mod` run of the same task to compare against, so this says the arm
> *serves* real agent work correctly — it does not say it is better or worse
> than the incumbent at it. That comparison needs the same task on both arms.

## Four configurations on one real task — 2026-08-24

`xeno-skills#306`, ctx 98,304, native `sm_120a`, `q4_0` KV, one run each.
**Nothing completed the task. Zero files changed, four times out of four.**

| artifact | decoder | outcome | ctx high-water | wall | files |
|---|---|---|---:|---:|---:|
| `UD-IQ2_XXS` | `dflash2+ngram` | FAIL | 69,401 | 537.7 s | 0 |
| `UD-Q2_K_XL` | `dflash2+ngram` | **WINDOW_BOUND** | **98,303** | 1,019.3 s | 0 |
| `UD-Q2_K_XL` | **`draft-mtp`** | FAIL | 85,782 | 855.8 s | 0 |
| `UD-Q2_K_XL` | **`draft-mtp+ngram`** | FAIL | 82,696 | 947.2 s | 0 |

**On task success this says one thing only: the task is beyond this model class
at this window, whichever decoder runs it.** Four decoders and two artifacts is
not a decoder question any more.

### Decode rate, and why the short turns are the ones to read

An agent loop is mostly short turns, and rate depends heavily on generation
length ([above](#dflash2ngram-on-a-real-task--first-measurement-2026-08-24)). On
the turns under ~400 tokens, which is most of them:

| decoder on `UD-Q2_K_XL` | short-turn decode | acceptance | mean accepted len |
|---|---|---|---|
| **`draft-mtp+ngram`** | **54–67 tok/s** | 0.48–0.61 | 2.66–5.13 |
| `draft-mtp` | 44–57 tok/s | 0.53–0.66 | 2.58–2.99 |
| `dflash2+ngram` | 30–43 tok/s | 0.36–0.49 | 2.43–3.04 |

**`draft-mtp+ngram` is the fastest arm measured on this artifact**, peaking at
**67.25 tok/s**, against `dflash2+ngram`'s best of 42.83 on the same task. The
separation is a consistent cluster across seven early turns, not one point.

**Not a verdict.** One run per arm, turns are not paired, and different turns
carry different prompts. The arena's paired protocol is what would settle it.

### ⚠️ Both MTP arms ended in an 8,192-token runaway; `dflash2+ngram` did not

The final generation of **each** MTP arm hit the request cap:

```
draft-mtp         ... 1237 · 4252 · 424 · 8192 tok   @ 26.19 tok/s
draft-mtp+ngram   ... 2008 · 7170 ·  165 · 196 · 8192 tok   @ 26.60 tok/s
```

`dflash2+ngram` on the same artifact never exceeded 4,811 and terminated every
generation. **This is the "stops stopping" signature
([`researchs/superalesha-quant-ladder/`](../researchs/superalesha-quant-ladder/README.md))
appearing on the MTP arms of an artifact that does not show it otherwise** —
which, if it holds up, would be a cost of MTP rather than of the quantisation.
Two observations, one per arm. Not established.

### What MTP costs and returns, measured

`UD-Q2_K_XL` carries `blk.64` in the file, so `--spec-type draft-mtp` needs **no
`-md`** — a configuration this project had never run, because every earlier
`draft-mtp` figure fed a separate 1.3 GB head to an artifact that lacked one.

| | `dflash2+ngram` | `draft-mtp` |
|---|---:|---:|
| model, CUDA0 | 8,630.57 | **8,965.31** |
| target KV | 1,728.00 | 1,728.00 |
| MTP draft KV | — | 384.00 |
| RS | 748.12 *(n_max 4)* | 598.50 *(n_max 3)* |
| compute | 472.27 | 472.27 + 82.01 |
| separate drafter | 1,393.90 | **0** |
| **total on CUDA0** | **12,973** | **12,230** |
| **free of the 15,172 llama.cpp sees** | 2,199 | **2,942** |

**743 MiB returned, not the ~1,394 that removing the sidecar suggests.** The model
buffer itself grows **334.74 MiB** once the head is used, and `--fit` raises its
own target from 768 to 1,234 MiB for the 466 MiB MTP context.

*Raw: `results/real-task-q2kxl-draft-mtp*.jsonl`,
`logs/dflash2-serve-draft-mtp*.log`. The two `dflash2+ngram` rows predate the
provenance fix and do not name their model; the two MTP rows do.*

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

⚠️ **This is a ctx 16,384 verdict and it does NOT hold at 65,536.** Measured
since, on the deep corpus: **the optimum moves from 24 to 16**, and `24` turns
into a null. See the next section — and note that the reasoning which predicted
the opposite is printed there too, refuted.

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

## `--spec-ngram-mod-n-match` at ctx 65,536 — the optimum moves, and my prediction was backwards

ctx **65,536** on the **deep** frozen corpus (`1a3ae4b813dd8447`, 406,146
chars), `--spec-draft-n-max 4`, three paired rounds, arms rotated, **every arm
`65+0`**. Raw `results/sweep-ngram-nmatch-65536.jsonl`, 12 rows.

| `n-match` | rounds (tok/s) | vs ours | ngram drafts | ngram decline | ngram mean acc len |
|---:|---|---|---:|---:|---:|
| 24 — won at 16,384 | 44.8, 53.0, 42.9 | −9.7 % [−29.1, +16.8], **null** | 18 | 97.0 % | 19.78 |
| **16** | **91.9, 88.5, 83.4** | **+67.5 % [+45.3, +95.1] RESOLVED** | **39** | **91.3 %** | **21.59** |
| 12 — what we ship | 63.3, 45.3, 51.4 | baseline | 22 | 96.4 % | 11.68 |
| 8 | 52.8, 47.3, 35.4 | −14.5 % [−31.1, +4.3], **null** | 43 | 92.7 % | 9.12 |

**Side by side with the shallow run, the ranking inverts:**

| `n-match` | ctx 16,384 | ctx 65,536 |
|---:|---|---|
| 24 | **+34.6 % RESOLVED** | −9.7 %, null |
| 16 | −1.5 %, null | **+67.5 % RESOLVED** |
| 8 | −14.5 % RESOLVED | −14.5 %, null |

### The prediction this refutes — mine, from earlier the same day

This page said, in the section above: *"the mechanism argues 24 should widen its
lead at depth — a fuller table means more distinct contexts colliding on a short
key."* It was labelled a hypothesis. **It is wrong, and backwards.**

At depth the binding constraint is not collision, it is **fire rate**. A longer
key is a stricter requirement, and a deeper window does not rescue it — at
65,536 `n-match 24` fires **18** times against `16`'s **39**, at a decline of
97.0 % against 91.3 %, while the two produce almost the same accepted length
(19.78 against 21.59). The extra specificity of 24 costs more hits than the
collisions it avoids. `16` wins because it is the only value that fires often
**and** long; `12` fires rarely *and* short (11.68), and `8` fires often but
useless (9.12).

### 🔴 The 13.6 % floor is a ctx 16,384 number, and it does not hold here

Decode is deterministic at temperature 0 — every arm's counters are identical
across all three rounds — so **all of this spread is the clock**:

| `n-match` | within-arm spread @ 16,384 | within-arm spread @ 65,536 |
|---:|---:|---:|
| 24 | 2.2 % | **23.5 %** |
| 16 | 0.8 % | 10.3 % |
| 12 | 9.5 % | **39.5 %** |
| 8 | 10.6 % | **48.9 %** |

**The same arm, same counters, varies up to 48.9 % between boots at 65,536.**
The 13.6 % floor was derived at 16,384 and is far too permissive at depth.
Every depth verdict in this project needs it re-derived; three rounds is not
enough to do that.

**What survives that, and why the direction is still safe to act on.**
`nmatch-16`'s **worst** round (83.4) beats every other arm's **best** round
(63.3, 53.0, 52.8) — a 32 % separation that needs no pairing and no floor at
all. **The ranking is solid. The `+67.5 %` is not** — do not quote the
magnitude.

> **We ship `12`, and it is the second-worst arm at the depth we serve.** The
> value to test in production is **16**, not the 24 that won at 16,384. Still
> not shipped: `worker-iq2s-quality.ps1` serves 98,304, which is another window
> again, and this page has now been wrong once about assuming transfer.

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

## The decoders at the window we actually serve - tested 2026-08-23

**The first measurement of any decoder at ctx 98,304 without a drafter loaded.**
Every prior row at this depth ran `--spec-type draft-dflash,ngram-mod`, so
"depth" and "drafter" had never varied independently -
[`CORRECTIONS.md` 26](../reports/CORRECTIONS.md).

`results/decoders-98304.jsonl`, 24 rows, six paired rounds, arms rotated each
round, deep corpus sha `1a3ae4b813dd8447`.

| arm | ok | timed out | tg samples | median | free MiB after load |
|---|---:|---:|---|---:|---|
| `none` | 6/6 | 0 | 33.53 / 33.55 / 33.58 / 33.80 / 34.15 / 34.76 | **33.69** | 800-1,935 |
| **`ngram-mod`** | **6/6** | **0** | 96.14 / 96.40 / 96.80 / 97.04 / 98.85 / 98.88 | **96.92** | 769-2,117 |
| `dflash2` | 5/6 | 1 | 0.64 / 47.31 / 49.31 / 52.82 / 53.62 | 49.31 | **45-376** |
| `dflash2+ngram` | 4/6 | 2 | 1.46 / 4.53 / 6.78 / 93.29 | **5.66** | **153-240** |

**`ngram-mod` at 98,304 is faster than the 75.2 median recorded at 16,384**, and
`ngram-mod` is the decoder all four `worker-*.ps1` already run. Speculation is
worth **+188 %** over none at this depth.

> **Read the artifact before transferring this.** These rows are
> **`UD-IQ2_XXS` at ctx 98,304**, and **no worker profile serves that
> pairing** -- `worker-iq2xxs-deep` runs that artifact at 131,072, and
> `worker-iq2s-quality` runs 98,304 on `UD-IQ2_S`, which is 1.1 GB larger.
> What transfers directly is the decoder verdict. What does **not** transfer
> without measurement is the rate.

**The free-VRAM columns do not overlap, and that is the finding.** Arms without
the drafter sit at 769-2,117 MiB, finish 12 times out of 12 and spread 3-4 %.
Arms with it sit at 45-376 MiB every single time, time out 3 times in 12, and
spread **146x** on identical flags - 0.64 to 93.29 tok/s.

**The mechanism, as far as it is established.** With a model-based drafter
`n_rs_seq` is 4, so the server writes `created speculative checkpoint ... size =
149.626 MiB` - one full recurrent-state plane - every few generated tokens. With
`ngram-mod` alone `n_rs_seq` is 0 and no such checkpoint exists. In the slow
rounds the gap between checkpoints reaches **30.41 s** against a median 2.35 s
in the fast ones, which is a stall rather than uniform slowness. Sampled live
during a slow arm: `free 196 MiB, util_gpu 100 %, util_memory 3 %, 2820 MHz,
70.18 W` - matching `gpu-trace-98304.jsonl`'s medians in every column.

**Not established:** why some drafter rounds escape. 93.29 tok/s at 240 MiB free
against 1.46 at 153. There is no clean threshold, only a band in which the
outcome is unreliable.

> **This reverses the ctx 16,384 verdict completely.** There
> `draft-dflash,ngram-mod` is **+48.5 % RESOLVED**; here it is **-94 %** with a
> one-in-four chance of not finishing. Nothing was shipped on the strength of
> either - all four profiles still run `ngram-mod` alone, which this makes the
> right choice at the served window rather than a cautious one.

*Raw: `results/decoders-98304.jsonl`. Report 33, `CORRECTIONS.md` 26.*

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
