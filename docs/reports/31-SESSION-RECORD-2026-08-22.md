# 31 — session record, 2026-08-22

**Every measurement taken this session, with the condition it was taken under.**
Written before a context compaction, so nothing survives only in a chat log.

Reports 29 and 30 hold the two narrative findings. This file holds **the data**,
including the runs that turned out to be invalid — a discarded run is evidence
about the instrument and belongs on the record.

Instrument: `C:\AI\llama.cpp-dflash2`, **build 10499, commit `1deefcca3`** =
llama.cpp PR #27342 on master, compiled today by
`qwen38-tuning/scripts/build-dflash2.ps1`. Build 10472 untouched throughout.

Artifact: `Qwen3.8-27B-UD-IQ2_XXS.gguf` (Dynamic V3, 6.77 GB).
Drafter: `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, 1.06 GB on disk.
KV `q4_0`/`q4_0`, `-np 1`, `-b 2048 -ub 256`, `--fit on --fit-target 768`,
temperature 0.0, top_k 1, seed 42.

---

## 1. Buffer allocations — measured, and the formula they confirm

Read from `llama_kv_cache` / `llama_memory_recurrent` / `sched_reserve` lines.

### The recurrent state does not scale with context

| ctx | KV (`q4_0`) | **RS** |
|---:|---:|---:|
| 32,768 | 576.00 MiB | **149.62 MiB** |
| 65,536 | 1152.00 MiB | **149.62 MiB** |
| 98,304 | 1728.00 MiB | **149.62 MiB** |
| 131,072 | 2304.00 MiB | **149.62 MiB** |

Flat to two decimals across a 4× range of depth while KV quadruples.

### It scales with the draft count instead

At ctx 16,384, `draft-dflash`, every other buffer identical:

| `--spec-draft-n-max` | RS buffer | = 149.625 × |
|---:|---:|---:|
| 3 (the default) | 598.50 MiB | 4 |
| 4 | 748.12 MiB | 5 |
| 7 (the clamp) | 1197.00 MiB | 8 |

> **RS buffer = 149.625 MiB × (1 + `--spec-draft-n-max`)**

Mechanism: `common/common.h:390` returns `draft.n_max` from `need_n_rs_seq()`
for `DRAFT_MTP`, `DRAFT_EAGLE3`, `DRAFT_DFLASH`, `DRAFT_DSPARK`;
`common/common.cpp:1699` assigns it to `cparams.n_rs_seq`. One base copy plus
one per draft position, so a rejected draft has somewhere to roll back to.
**`ngram-mod` pays none of it** — it is not a model drafter.

### The drafter's real cost, decomposed

Report 29 measured the drafter at **1,936 MiB resident** (free VRAM 2,376
without it, 440 with it, ctx 16,384). That number is:

| | MiB |
|---|---:|
| drafter weights | 1,079.61 |
| drafter KV buffer (**`f16`** — `-ctkd`/`-ctvd` never set) | 45.00 |
| drafter compute buffer | 269.29 |
| **extra recurrent state on the TARGET** (748.12 − 149.62) | **598.50** |
| total | **~1,992** |

**Roughly a third of "the drafter's cost" is not the drafter.** It is the target
model's own state, replicated for speculative rollback.

### Whole-run footprints, ctx 16,384, real-code

`free_before − free_after`, median of three rounds:

| arm | footprint | vs n=4 |
|---|---:|---:|
| n-3 | 9,417 MiB | −154 |
| n-4 | 9,571 MiB | — |
| n-7 | 9,889 MiB | +318 |

n=3's −154 matches the RS prediction of −149.6. n=7's +318 is *less* than the
+448.9 the RS line implies, because `--fit` reclaims margin elsewhere.

---

## 2. Decoder comparison — report 29's data

Full write-up in
[report 29](29-DFLASH2-AND-THE-PROMPT-THAT-FLATTERED-NGRAM.md).
Raw: `results/dflash2-arena-warm.jsonl`, 24 rows. ctx 16,384, three rounds,
arms rotated, paired, `harness.paired_deltas` at the 13.6 % floor.

| arm | real-code (4.7 % dup) | synthetic (66.2 % dup) |
|---|---|---|
| `none` | 45.4, 45.4, 41.8 → −14.4 % **RESOLVED** | 43.5, 43.9, 43.4 → −63.5 % **RESOLVED** |
| `ngram-mod` | 53.0, 52.5, 49.3 (baseline) | 119.7, 119.4, 119.3 (baseline) |
| `draft-dflash` | 69.5, 69.1, 69.8 → **+34.7 % [+31.1, +41.6] RESOLVED** | 108.6, 108.4, 108.6 → −9.2 %, within floor |
| `draft-dflash,ngram-mod` | 78.9, 78.8, 72.2 → **+48.5 % [+46.6, +50.1] RESOLVED** | 137.5, 138.8, 115.0 → +9.2 %, within floor |

**The verdict reverses with the prompt.** `ngram-mod` is worth 2.7× on the
repetitive prompt and about **1.17×** on real source.

Acceptance, same runs:

| arm | synthetic | real code |
|---|---:|---:|
| `ngram-mod` | 60.2 % | 42.0 % |
| `draft-dflash` | **91.6 %** | 46.2 % |
| `draft-dflash,ngram-mod` | 63.6 % | 54.0 % |

DFlash2's 91.6 % is a property of the prompt, not of DFlash2.

---

## 3. Per-implementation statistics — the finding under the finding

`common_speculative_print_stats`, LOG_TRC only; our arena already ran `-lv 5`,
so this data existed before the question was asked. Parsed by
`harness.parse_spec_impl_stats`.

Aggregated over 26 logs:

| regime | impl | calls | drafts | **decline** | mean acc len | draft ms |
|---|---|---:|---:|---:|---:|---:|
| real-code | `ngram-mod` | 4,488 | 129 | **97.1 %** | 13.65 | 6 |
| real-code | `ngram-mod` | 1,626 | 93 | **94.3 %** | 18.00 | 2 |
| real-code | `draft-dflash` | 2,145 | 2,145 | 0.0 % | 2.85 | 12,863 |
| synthetic | `ngram-mod` | 734 | 184 | 74.9 % | 19.23 | 2 |
| synthetic | `draft-dflash` | 1,320 | 1,320 | 0.0 % | 4.65 | 8,094 |

**`ngram-mod` is not weak — it rarely fires.** On real code it declines 94–97 %
of the calls it receives, and `draft-dflash` is called exactly the number of
times ngram declines. When ngram does fire it is worth **6× more per draft**.
`draft-dflash` is also the expensive one: 12,863 ms of cumulative draft time
against ngram's 6 ms.

---

## 4. `--spec-ngram-mod-n-min` — swept, no effect, do not repeat

ctx 16,384, real-code, frozen corpus, three rounds.
Raw: `results/sweep-ngram-nmin.jsonl`.

| `n-min` | rounds (tok/s) | vs base |
|---:|---|---|
| 16 (ours) | 79.7, 79.6, 79.8 | baseline |
| 8 | 79.7, 79.5, 79.8 | −0.1 % |
| 4 | 79.7, 79.6, 79.8 | −0.0 % |
| 2 | 79.8, 79.8, 79.7 | +0.1 % |

Spread across twelve runs: **0.3 %**. At that repeatability a 1 % effect would
show.

**Why the hypothesis was wrong.** `ngram-mod` declines 93.7 % of calls, so
letting shorter drafts through looked like a large free win. It was a misreading
of `common/speculative.cpp:1993`: in `draft_one`, `i` counts **draft tokens
already produced**, not matched context. `n_min` is a minimum draft *length*,
and the declines happen at `i = 0` — the table misses on the very first
successor — where no value of `n_min` can help.

**What that leaves open:** the knob that governs the decline rate is
`--spec-ngram-mod-n-match` (default 24, ours 12). **Swept — §5b.** It turns out
not to close the decline either: the winning arm drafts *less* often and better.

---

## 5. `--spec-draft-n-max` — the largest win, and it is window-dependent

Raw: `results/sweep-draft-n.jsonl`, `results/sweep-draft-n-65536.jsonl`.

### ctx 16,384, real-code, three rounds, rotated, paired

| arm | rounds (tok/s) | vs n=4 | free after | acceptance |
|---|---|---|---:|---:|
| n-3 (the **default**) | 70.2, 70.5, 70.2 | **−11.5 % [−11.6, −11.4]** | 913 MiB | 53.1 % |
| n-4 (ours) | 79.3, 79.7, 79.5 | baseline | 667 MiB | 54.0 % |
| **n-7 (the clamp)** | 97.7, 98.4, 98.2 | **+23.4 % [+23.1, +23.5] RESOLVED** | 443 MiB | 47.5 % |

**The default of 3 sits 28 % below the best point**, and nothing in the flag's
help text says so. Acceptance *falls* as n rises while throughput climbs: more
tokens per verify step outweighs a lower hit rate. `ngram-mod`'s mean accepted
length also rises, 15.53 → 18.00 → 21.90.

The clamp is `block_size − 1` = 7 for this drafter
(`common/speculative.cpp:989`); a larger request is silently lowered.

### ctx 65,536 — the verdict reverses

| arm | rounds (tok/s) | split |
|---|---|---|
| n-3 | 19.1, 19.1 | 65+0 |
| n-4 | 18.2, 18.4 | 65+0 |
| n-7 | 7.8, 8.1 | **63+2 — spilled** |

**These rates are NOT valid** — see §7, fault 4. The **residency** facts are:
n=7 spills two layers at 65,536, n=3 and n=4 do not. Confirmed independently
when the real-task server was started at 65,536 with n=7: `63+2`, and the RS
buffer split with **49.88 MiB on the CPU**.

---

## 5b. `--spec-ngram-mod-n-match` — swept, and the default we overrode wins

Raw: `results/sweep-ngram-nmatch.jsonl`, 12 rows. ctx 16,384, real-code on the
frozen corpus `5672a9bcce74c0d0`, `draft-dflash,ngram-mod`, `--spec-draft-n-max 4`,
`n-min 16` and `n-max 32` fixed. Three rounds, arms rotated each round, paired.

| `n-match` | rounds (tok/s) | vs our 12 | acceptance | ngram drafts | mean acc len |
|---:|---|---|---:|---:|---:|
| **24 — the llama.cpp default** | 94.5, 96.3, 94.2 | **+34.6 % [+31.4, +40.8] RESOLVED** | 62.2 % | 29 | **23.45** |
| 16 | 69.2, 69.7, 69.5 | −1.5 % [−4.9, +3.9], within the floor | 53.5 % | 25 | 19.20 |
| 12 — every worker profile | 71.7, 73.3, 66.9 | baseline | 54.0 % | 31 | 18.00 |
| 8 | 56.7, 62.7, 61.5 | **−14.5 % [−20.9, −8.0] RESOLVED** | 37.0 % | 43 | 8.95 |

**The trap the arm set was designed around is the mechanism of the result.**
`n_match` is the hash key width (`common/ngram-mod.cpp:15-25`), so shortening it
is a strictly weaker match requirement and ngram fires more — 43 drafts at `8`
against 29 at `24`. It loses anyway: a collapsed key returns the successor of
whichever context last wrote the slot, so mean accepted length falls
**23.45 → 8.95**, accepted-token yield falls **651/921 → 342/1306**, and the
draft calls needed for the same 512 tokens rise **475 → 649**. Firing twice as
often on a worse draft is a loss, which `speculative.cpp:2545` sitting six lines
above `2551` already predicted.

**This retracts report 30**, which read our `12` agreeing with the 3090 stack's
`NMAX=12` as independent confirmation. The two flags share a number and nothing
else — theirs caps a longest-match search, ours is a key width with no length
dimension. `CORRECTIONS.md` §21.

**Determinism note, and it matters for how the rounds are read.** Every arm's
per-implementation counters are **byte-identical across all three rounds** —
same calls, same drafts, same accepted length to two decimals — and only the
timing fields move. Decode is deterministic at temperature 0, so extra rounds
re-measure the clock, which is what the pairing is for, but they buy **no second
sample of drafter behaviour**. A counter-level result here rests on one
trajectory per arm.

**`8` is RESOLVED by the rule and not comfortably.** Mean clears the floor, sign
is consistent, but one round landed at −8.0 %, inside it. Direction solid,
magnitude approximate.

**Two limits, both of which keep it out of the served profiles for now.** It was
measured at `--spec-draft-n-max 4`, so today's two winners have never run
together; and at ctx 16,384, a quarter of the served window, where this project
has already been burned once (`draft-mtp`, +81 % at 16K, −71 % at 131,072). The
mechanism argues 24 should widen its lead at depth — a fuller table means more
distinct contexts colliding on a short key — but **that is a hypothesis, and it
is written here as one.**

---

## 5c. The two winners crossed — they cancel, and that is the day's most useful result

Raw: `results/sweep-draft-n-x-nmatch.jsonl`, 12 rows. Same corpus, same depth,
same binary as §5 and §5b, all four arms inside the same three rounds, arms
rotated, every arm `65+0`.

| arm | rounds (tok/s) | vs base | vs `n7 m12` | vs `n4 m24` |
|---|---|---|---|---|
| `n4 m12` — what we ship | 74.6, 75.2, 75.2 | baseline | — | — |
| `n7 m12` | 94.4, 95.1, 94.9 | **+26.4 % RESOLVED** | — | — |
| `n4 m24` | 98.4, 97.7, 97.4 | **+30.5 % RESOLVED** | — | — |
| **`n7 m24` — both** | 65.5, 63.4, 65.5 | −13.6 % | **−31.6 % RESOLVED** | **−33.8 % RESOLVED** |

**Both singles replicated** — +26.4 % against §5's +23.4 %, +30.5 % against
§5b's +34.6 %, on different boots — so the combination arm is not a replication
failure of either half. **Stacked they reach 52.4 % of the independent
expectation** (64.8 measured against 123.6 from 1.264 × 1.305) and the pair is
the **slowest arm in the set**, below the incumbent both halves beat.

**The mechanism is the cascade they share.**

| arm | ngram drafts | ngram decline | dflash accepted / generated |
|---|---:|---:|---:|
| `n4 m12` | 31 | 94.3 % | 974 / 2,041 = 47.7 % |
| `n7 m12` | 41 | 90.0 % | 775 / 2,564 = 30.2 % |
| `n4 m24` | 29 | 93.9 % | 915 / 1,781 = 51.4 % |
| **`n7 m24`** | **12** | **97.7 %** | 1,262 / **3,612** = **34.9 %** |

`n-match 24` makes `ngram-mod` stricter — fires less, much better. `n-max 7`
makes `draft-dflash` longer and dearer, 8 tokens per call instead of 5. Each is
affordable alone: at `n4 m24` a strict ngram is fine because dflash's short
drafts are cheap to waste, and at `n7 m12` expensive dflash drafts are fine
because ngram still fires 41 times and covers the costly steps. **Stacked,
ngram nearly stops and dflash pays the full 8-token draft on almost every step
at a 34.9 % hit rate.**

⚠️ **One step of that explanation is not established.** Why ngram's decline
rises to 97.7 % is unattributed — the arms take different trajectories, so the
rates are not a clean comparison. A fitting mechanism: `draft_one` flushes new
n-grams only when `sinfo.i_last + 32 < cur_len` and only up to `cur_len - n`
(`speculative.cpp:1978-1979`), so a generation accepting more per step outruns
its own table. **Hypothesis, testable against the occupancy trace at
`speculative.cpp:1950`, untested.**

**What it changes.** **Pick one, and pick `n-match 24`** — the best point
measured, and it moves no allocation, where `n-max 7` costs 447 MiB of
recurrent state (free after 1,272 → 825 MiB at the same `65+0`). Still a
ctx 16,384 verdict; nothing shipped.

**Bookkeeping, stated so nobody upgrades it later.** Against the baseline the
pair reads **−13.58 %** with a consistent sign across all three rounds — **0.02
points under the 13.6 % floor**, so `harness.paired_deltas` does not resolve it
and neither does this report. Nothing rests on that comparison: the verdict
comes from the two against-the-singles deltas, which clear the floor by more
than 17 points.

> **A measured win plus a measured win is not a measured win.** Both halves
> were RESOLVED against the same baseline on the same corpus, and their sum is
> worse than the thing they each beat. The design that caught it was a 2×2
> rather than a single "both" arm — against one baseline, "both" could not have
> been told apart from either single failing to replicate.

---

## 6. Real-task benchmark — the first measurement of the project's own metric

`bench/real_task_bench.py`, throwaway clones from the GitHub remote, scored by
each repo's own verify command. Raw: `results/real-task-bench.jsonl`.

### Context high-water, across three windows

| window | high-water range | saturated? |
|---:|---|---|
| 32,768 | 32,767 – 41,377 | **yes, all** |
| 65,536 | 54,324 – 72,056 | **4 of 5** |
| 98,304 | 56,861 – 88,668 | no |

**Every time the window grew, the tasks used it.** The numbers seen at the two
smaller windows are ceilings the operator set, not requirements of the work.

**This refutes plan 06's own headline hypothesis** — that tasks peak near 40,000
and the 98,304 profile reserves 1.5–2 GB for nothing. The 40,000 figure came
from the run that saturated at 32,768. `worker-iq2s-quality.ps1` at 98,304 is
the **minimum sensible window**, and any plan to shrink it to buy a higher
quantisation rung must be rewritten.

### Outcomes at ctx 98,304 (`n-max 4`, drafter loaded, `65+0`, 254 MiB free)

| task | outcome | ctx high-water | wall | files changed | verify |
|---|---|---:|---:|---:|---:|
| `xeno-skills:306` | FAIL | 88,668 | 2,400 s (timeout) | 0 | 0 |
| `xeno-skills:314` | FAIL | 88,668 | 1,427 s | 0 | 0 |
| `openclink:144` | FAIL | 56,861 | 1,647 s | 0 | 0 |
| `openclink:145` | FAIL | 84,889 | 1,759 s | 0 | 0 |
| `openclink:149` | FAIL | 84,889 | 2,400 s (timeout) | 0 | 0 |

**Final: 5 tasks — 0 PASS, 5 FAIL, 0 WINDOW_BOUND, 0 VOID.** Context
high-water min 56,861, median 84,889, max 88,668. **Nothing saturated at this
window**, so none of these five can be excused by the context size.

**A green verify with no diff is a FAIL** — it passes the tests that were
already passing. Every baseline was green (`base=0`), so none of these is an
environment failure.

🔴 **Unexplained and the most important open question in this file.** With
room to spare, the worker ran 24–40 minutes per task and changed nothing. That
is now a genuine result about the worker and it has no mechanism attached.
The OpenCode transcript is written to `<clone>.stdout.txt` beside the clone and
is **deleted with the scratch root** — capture it before the next run.

### Earlier windows, for the record

At 32,768: five tasks, all `files=0`, high-water 32,767 (= `n_ctx − 1`) for
three of them. The server log carried `exceeds the available context size
(32768 tokens)` **six times** and `truncated = 1` **four times**. Those were
reported as FAIL by the first version of the harness — see §7, fault 5.

At 65,536: `xeno-skills:306` reached 54,324 without saturating and changed
nothing (a real FAIL); the other four saturated.

---

## 7. Instrument faults found today — seven

Each produced a *plausible* result rather than a crash. Two were caught by the
developer, not by any check in the repo.

| # | fault | how it read | fix |
|---|---|---|---|
| 1 | `parse_layer_split` returned the **drafter's** `6+0`, not the target's `65+0` — a drafter adds its own assignment passes and is assigned last | "fully resident" for the wrong model, in which a spill could never appear | `expect_layers`; `tests/test_layer_split_with_drafter.py` |
| 2 | 16-token warm turn left the n-gram table empty; the first timed sample of every ngram arm came in 35–40 % low (69.8 against 113.4 in the same boot) | a systematic bias hidden inside a median | full-length warm turn |
| 3 | the VRAM settle wait was split across kill and setup, so `run_arm`'s teardown killed the server and `start()`'s kill found nothing — **present, called, inert** | instrument fault 7 restored after being fixed | `stop_server()`; `tests/test_arena_teardown.py` |
| 4 | **generations of 2–4 tokens against a 512-token budget** were ranked as measurements: the frozen corpus is ~28,000 tokens and the arena asked for 32,768 | three arms, six rows, a tight range and a **RESOLVED −56.5 %** computed over noise | `generation_is_measurable`; `tests/test_short_generation_guard.py` |
| 5 | a task that **filled the context window** was scored as a worker failure | `0 PASS, 5 FAIL` — reads as a verdict on the model | `WINDOW_BOUND`; `--n-ctx` made required; `tests/test_window_bound.py` |
| 6 | `report()` **pooled both prompt regimes** into one baseline | `ngram-mod [119.7, 119.4, 119.3, 53.0, 52.5, 49.3]`, and `draft-dflash` read `+12.8 % within noise` instead of `−9.2 %` and `+34.7 % RESOLVED` | grouping by regime; `tests/test_arena_report_grouping.py` |
| 7 | **the benchmark built its prompt from its own source, which was being edited** | 78.9 vs 105.4 tok/s on byte-identical arguments — a 33 % gap on a 13.6 % noise floor | frozen `corpora/real-code.txt` + a `corpus` hash on every row; [CORRECTIONS §20](CORRECTIONS.md) |

Plus two harness bugs that failed loudly and cost only time: `bash` resolving to
WSL instead of Git bash, and `.git/objects` read-only files defeating
`shutil.rmtree` on Windows.

**Fault 6's cause is worth keeping:** the patch that split the regimes had been
written and applied *without an assertion*, so it silently did not match and the
old code ran. Every subsequent patch in this session asserts.

---

## 8. Retractions filed today

| # | claim | why it was wrong |
|---|---|---|
| [§19](CORRECTIONS.md) | "`UD-IQ2_S` has never been loaded once" | 38+ measured rows across six result files, dozens of logs, four worker profiles. A stale ledger row, copied into a plan without checking |
| [§20](CORRECTIONS.md) | absolute real-code tok/s figures | the prompt was built from `bench/` source edited between runs; only paired within-round deltas survive |
| [§21](CORRECTIONS.md) | "`n-match 12` — the same cap, chosen independently" (report 30) | not the same cap: theirs bounds a longest-match search, ours is a keyless hash key width. Swept the same day — the default `24` is **+34.6 % RESOLVED** over our `12` |

**§21 is the one worth reading twice.** The source read that refutes it was
written earlier the same day, in the same session, and says so plainly —
*"n_match changes key SPECIFICITY only"*. Report 30's sentence was written
anyway. **A source read does not correct a claim unless somebody goes back to
the claim**, and nothing in this repo does that automatically except
`audit-stale-claims.py`, which only knows the rules it has been given.

Also corrected: `worker-iq2xxs-deep.ps1` claimed `--fit-target 768` was "the
default". The default is **1024** (`common/common.h:473`). We had already spent
most of that GiB and the header said the opposite.

---

## 9. Source facts established by reading, not measuring

Each cited so a later session does not re-derive them.

| fact | where |
|---|---|
| `--spec-draft-n-max` default **3** | `common/common.h:325` |
| `--spec-draft-p-min` default **0.0** (confidence early-stop is **off**) | `common/common.h:329` |
| main-path `backend_sampling` default **false** | `common/common.h:295` |
| **draft-path** `backend_sampling` default **true** — a different field | `common/common.h:331` |
| `-bs` / `--backend-sampling` exists, off by default | `common/arg.cpp:2296` |
| `--fit-target` default **1024 MiB** | `common/common.h:473` |
| a grammar **disables** backend sampling | `common/sampling.cpp:421` |
| `--reasoning-budget` does the same | `common/sampling.cpp:427` |
| `--spec-type` is a **comma-separated list** | `common/arg.cpp:4155` |
| **speculator priority is hardcoded**: every `ngram-*` outranks every model-based type, and command-line order is discarded | `common/speculative.cpp:2540–2552` |
| the DFlash block clamp is `block_size − 1` = **7** | `common/speculative.cpp:989` |
| DFlash**2** is selected from the checkpoint, not a flag: `is_dflash2 = selector_top_k > 0` | `common/speculative.cpp:978` |
| our drafter is genuinely DFlash2: `dflash.selector_top_k = 16`, 81 tensors | the GGUF's own metadata |
| the drafter attends to **2,048 tokens**, all five layers sliding | `dflash.attention.sliding_window` in the GGUF |
| grammar allocates **nothing on device** | `src/llama-grammar.cpp` — no `ggml_backend`, `cuda`, `ggml_new_tensor` |
| KV cache types bottom out at **4 bits** (`q4_0`, `q4_1`, `iq4_nl`) | `common/arg.cpp:305–315` |

**Consequence of the priority list:** our measured `draft-dflash,ngram-mod`
+48.5 % ran **ngram-mod first with dflash as the fallback**, and no flag
reverses it. Since dflash alone beat ngram alone by +34.7 %, "dflash first" is
an obvious unmeasured configuration reachable only by reordering ten lines.

---

## 9b. Housekeeping that happened during this session

**`docs/tested/` was renamed to `docs/results/`** by the developer. All eight
files moved with their content intact (git records `R100` for seven of them),
and 25 references across 11 files were rewritten — `CLAUDE.md`, `DONE.md`, the
ledger, two `docs/agents/` files, three reports and two research files.

**I misdiagnosed the rename as damage.** `git status` showed eight deletions
under `docs/tested/`, and I connected them to the running OpenCode worker on the
theory that its server had persisted `C:\AI` as its project root — a hazard
`bench/opencode_corpus.py` genuinely documents. I stopped the benchmark and
killed every `node`/`opencode` process on that theory before the developer said
they had done the rename themselves.

Cost: the benchmark had already finished, so no measurement was lost, but the
kill interrupted its cleanup and left one clone behind. Removed by hand;
`D:ench-scratch` verified empty.

**The lesson is the one this file keeps repeating in a different costume:** eight
files vanishing from `git status` is an observation, and "the worker did it" was
a hypothesis I acted on before checking. The check available was one question to
the developer, or `git status --short` read as a rename rather than as deletions
— `git add -A` showed `R100` immediately afterwards.

---

## 9c. What transferred from the RTX 3090 scan

The scoreboard lives in [`../results/08-rtx3090-transfer.md`](../results/08-rtx3090-transfer.md)
and is the answer to "did any of that 434-technique scan actually get used".

Short version: **three measured wins** (`--spec-draft-n-max` at +23.4 % RESOLVED,
the `draft-dflash,ngram-mod` pair at +48.5 %, and `--spec-ngram-mod-n-match 24`
at +34.6 %), **one measured null** (`--spec-ngram-mod-n-min`), **five
read-and-closed** from source without spending a GPU round, **two we already
had**, and **two architecturally impossible**.

**One of the wins arrived by refuting the claim that produced it** (§5b): their
"shorter match beats longer" reverses here, and the round that showed it also
retracted a line in report 30 that had read the two projects' agreement as
confirmation.

Two of the read-and-closed verdicts are worth as much as a win, because each
saves a round that would have measured nothing:

- `-ctkd`/`-ctvd` — the drafter decodes 5 tokens in one `llama_decode`, so a
  quantised draft KV takes `MMA_F16` with a full dequant, not the vector kernel.
- `--spec-draft-p-min` — `1/sum ∈ [0.0625, 1.0]` by construction, so any value
  at or below 1/16 is identical to 0.00. A ladder starting at 0.05 would have
  repeated the `n_min` error exactly.

And one of them found an error in **our** profiles rather than a technique of
theirs: `--fit-target` has the draft model's bytes added to it before `--fit`
runs (`server-context.cpp:1074`), so our `768` reaches `fit.cpp` as roughly
1,900–2,100 MiB.

---

## 10. Still open

- **Why the worker changes nothing with room to spare** (§6). No mechanism.
- **Two flags from the 3090 scan still untested**: `--spec-draft-p-min` (arms
  defined in `dflash2_arena.ARM_SETS`, unrun) and `-fitt`. Of the original six,
  `--spec-ngram-mod-n-match` is now swept (§5b) and three were closed from
  source — `-bs`, `GGML_CUDA_GRAPH_OPT=1`, `-ctkd`/`-ctvd`. The semantics read
  is `researchs/llamacpp-flag-semantics-2026-08-22.md` (run `wf_fb00df02-337`).
- **The two draft-side winners have never run together**, and neither has been
  measured at the served depth. `--spec-draft-n-max 7` was measured at
  `n-match 12`; `n-match 24` at `n-max 4`. Both are ctx 16,384 verdicts and the
  profiles serve 65,536–98,304. `n-match` costs no VRAM, so it is the cheaper
  of the two to take up in depth.
- **The true context requirement above 98,304.** Three windows, three
  saturations; 98,304 was the first that held.
- **Grammar × drafter has never been run together** — the served profile needs
  both. Phase 6 of plan 06.
- **Depth for the draft count** is unmeasured, not measured-and-negative: the
  frozen corpus is too small to fill 65,536 honestly.
- **Every `ngram-*` verdict in the register** was set on a repetitive prompt and
  is owed a re-measurement, starting with report 20's "+200 % at 131,072".
