# 29 — DFlash2 wins on real code, loses on the prompt we had been using

**2026-08-22.** Build 10499, compiled from llama.cpp PR #27342 (commit
`1deefcca3`). Artifact `UD-IQ2_XXS` (Dynamic V3), `ctx 16,384`, `-ctk/-ctv q4_0`,
all arms `65+0` resident in every round.

Raw: `qwen38-tuning/results/dflash2-arena-warm.jsonl`, 24 rows.
Driver: `qwen38-tuning/bench/dflash2_arena.py`. Issues #17, #18.

---

## The result

Three rounds per regime, arms rotated each round, paired by round, verdict from
`harness.paired_deltas` — resolved only if the effect clears the **13.6 %**
drift floor **and** keeps its sign.

### real-code — this repo's own source, 4.7 % duplicate lines

| arm | rounds (tok/s) | vs `ngram-mod` | verdict |
|---|---|---|---|
| `none` | 45.4, 45.4, 41.8 | −14.4 % [−15.2, −13.6] | **RESOLVED** |
| `ngram-mod` | 53.0, 52.5, 49.3 | — | baseline |
| `draft-dflash` | 69.5, 69.1, 69.8 | **+34.7 % [+31.1, +41.6]** | **RESOLVED** |
| `draft-dflash,ngram-mod` | 78.9, 78.8, 72.2 | **+48.5 % [+46.6, +50.1]** | **RESOLVED** |

### synthetic — generated blocks, 66.2 % duplicate lines

| arm | rounds (tok/s) | vs `ngram-mod` | verdict |
|---|---|---|---|
| `none` | 43.5, 43.9, 43.4 | −63.5 % [−63.7, −63.2] | **RESOLVED** |
| `ngram-mod` | 119.7, 119.4, 119.3 | — | baseline |
| `draft-dflash` | 108.6, 108.4, 108.6 | −9.2 % [−9.3, −9.0] | within the floor |
| `draft-dflash,ngram-mod` | 137.5, 138.8, 115.0 | +9.2 % [−3.6, +16.3] | within the floor, sign flips |

**The verdict reverses between the two.** `draft-dflash` is −9.2 % on one prompt
and +34.7 % on the other, measured in the same session, on the same binary, at
the same window, with the same artifact. The only thing that changed was how
repetitive the text was.

---

## Why: the two drafters do different jobs

`ngram-mod` drafts by matching text it has already seen in the context. It costs
**0 MiB** and it is very strong exactly where the continuation is already on
screen. Where the model is writing something new, it has nothing to offer.

DFlash2 is a trained drafter. It does not need to have seen the text before.

The `none` row is the cleanest evidence:

| prompt | `none` vs `ngram-mod` | what `ngram-mod` is worth |
|---|---|---|
| synthetic, 66.2 % duplicate | −63.5 % | ~2.7× |
| real code, 4.7 % duplicate | −14.4 % | **~1.17×** |

**On real code `ngram-mod` buys about 17 %.** Its reputation in this project —
"+200 % at 131,072", "1.8× on real code" (report 20, `tested/02`) — was earned
on prompts built from repeated blocks.

Acceptance moves the same way, and it is the drafter's own view of the same
fact:

| arm | synthetic | real code |
|---|---:|---:|
| `ngram-mod` | 60.2 % | 42.0 % |
| `draft-dflash` | **91.6 %** | 46.2 % |
| `draft-dflash,ngram-mod` | 63.6 % | 54.0 % |

DFlash2's 91.6 % on the synthetic prompt is not a property of DFlash2. It is a
property of a prompt so repetitive that a trained drafter barely has to predict.

---

## What this corrects

Three things stated earlier in the same session, all from the synthetic prompt:

1. **"DFlash2 ties `ngram-mod`."** True only on the flattering prompt. On real
   code it wins by a third.
2. **"The combination is +23.2 %."** It is +48.5 % on real code and inside the
   noise floor on synthetic.
3. **"DFlash2's acceptance is 91.6 %."** On real code it is 46.2 %.

The synthetic prompt was **66.2 % duplicate lines** by
`harness.line_repetition_pct`; this repo's own source measures 0.6–4.8 %.
`depth_sweep.py` already warns about this in its own header — *"the sweep prompt
is 84.5 % duplicate lines, so treat the smaller number as the real one"* — and
the warning was read and then walked into.

---

## The instrument faults found on the way

Six, each caught before it reached a published row except the last, which
reached a chat message and is corrected above.

| # | fault | how it would have read |
|---|---|---|
| 1 | `parse_layer_split` returned the **drafter's** `6+0`, not the target's `65+0`: a drafter adds its own assignment passes and is assigned last | "fully resident" for the wrong model, in which a spill of the target's 65 layers could never appear |
| 2 | 16-token warm turn left the n-gram table empty, so the first timed generation of every ngram arm came in **35–40 % low** (69.8 against 113.4 in the same boot) | a systematic bias hidden inside a median |
| 3 | The VRAM settle wait was split across kill and setup, so `run_arm`'s teardown killed the server and `start()`'s kill found nothing to kill — **present, called, and inert** | instrument fault 7 restored after being fixed |
| 4 | A second arena was launched while the first was finishing; both drove port 8080 and the older teardown killed the younger server | the log ends mid-load with no error, because it did not fail |
| 5 | A unit test called the real `kill()` **while a measurement was in flight** and stopped the server it was benchmarking | `WinError 10054` mid-round |
| 6 | `report()` **pooled both regimes into one series** | `ngram-mod [119.7, 119.4, 119.3, 53.0, 52.5, 49.3]` — one baseline across two prompts, and every delta against it meaningless |

Fault 6 is the instructive one. The patch that was supposed to split the
regimes had been written but applied without an assertion, so it silently did
not match and the old code ran. It did not crash and the output did not look
wrong. Pooled, `draft-dflash` reads **+12.8 %, within noise**; split, it is
−9.2 % on one prompt and **+34.7 % RESOLVED** on the other.

---

## What this does NOT establish

- **Anything above 16,384.** A verdict at one depth does not transfer:
  `draft-mtp` is +81 % at 16K and −71 % at 131,072 on the same artifact. The
  drafter costs **1,936 MiB resident** (1.06 GB on disk — measured from free
  VRAM, 2,376 MiB without it against 440 MiB with it), and `--fit` **cannot
  measure it at all**, so residency at depth is an open question, not an
  extrapolation.
- **Anything about task success.** This is tok/s. This project's metric is
  verified accepted coding tasks per hour, and no number here measures it.
  `docs/plans/06-REAL-TASK-BENCHMARK.md` is the runbook for that.
- **That output is unchanged.** Speculative decoding in llama.cpp is
  verification-based and should be byte-identical to no speculation;
  `bench/spec_output_identity.py` was written to check it and **has not been
  run**.
- **Anything with a grammar loaded.** The production profile needs one — without
  it 41.5–58.3 % of corpus attempts emit no fenced block at all — and no run
  here had one. See `tested/05-runtime-flags.md`.

---

## What to do next

1. **Re-measure the `ngram-mod` verdicts that were set on repetitive prompts.**
   Report 20's "+200 % at 131,072" is the largest of them.
2. **Run Phase 6 of plan 06** — grammar × drafter, four boots. Cheap, and the
   configuration we intend to serve has never been run.
3. **Take DFlash2 up in depth**, watching residency rather than assuming it.
4. **Do not quote a decoder number without naming the prompt it was measured
   on.** That is the whole content of this report.
