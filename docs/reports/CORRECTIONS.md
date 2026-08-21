# Corrections register — every published claim this project later contradicted

**Read this before trusting any number in reports 00–24.** These reports were
written as the work happened, and several of them state things the machine
later disproved. Each report carries its own correction banner, but a banner
only helps a reader who opened *that* report. This is the list in one place.

`python C:\AI\scripts\audit-stale-claims.py` finds every line in the tree that
matches one of these, so a claim cannot quietly survive in a document nobody
reopened.

**A match is not automatically a defect.** A report that *describes* a
retraction matches too. The audit produces a worklist to read, not a verdict.

---

## 1. `output_contract_pct` is the PASS rate

`100 * (attempts_seen - contract_violations) / attempts_seen`. **Higher is
better.** It was read as a violation rate for a full day, so every sentence
written on 2026-08-20 that interprets it says the opposite of the truth. The
figures themselves never changed.

*Correction: report 04 §7, report 23. Affected: 15, 16, 19, 20, 22, START-HERE.*

## 2. Every n-gram percentage is an upper bound on a synthetic best case

`depth_sweep.filler()` repeats one class definition with a four-digit index —
962 blocks at 147,456, adjacent blocks **99.5 % identical**, **84.5 % of
non-blank lines exact duplicates**. An n-gram decoder drafts from what is
already in the context, so this is close to the most favourable text that could
be built for it. Acceptance pinned at 99–100 % across every depth is the tell,
and the figures *rising* with depth is the filler getting more repetitive, not
the model getting faster.

**Affected figures:** +94.69 %, +135.89 %, +200.22 %, +213.08 %, +330.40 %,
+120.54 %, +114.64 %, +112.55 %, +108.49 %.

**What survives:** n-gram costs no VRAM, needs no drafter file, and its output
is byte-identical to the unaccelerated run. The mechanism is not in question.
**What does not:** the magnitudes, pending steps F1/F2 at 73.17 % repetition.

*Correction: report 24 §"instrument fault 8". Measure with
`harness.filler_repetition_pct`.*

## 3. `ngram-cache` is disqualified, and report 20 said the opposite

Its greedy hash is `3EFE93950A8A980E` against a same-depth baseline of
`04E5CAB1D14525C0` — it changes the answer, so it is not doing draft-and-verify.
Report 20 §1.1 originally printed the baseline hash for it under the heading
*"Byte-identical output"*. **That block was typed by hand instead of read from
the JSONL**, and it certified as safe a decoder that is not.

*Correction: report 20 §1.1, report 23 §1.*

## 4. `-ot` on ssm tensors gives three different outcomes, and was promoted on one

Report 20 promoted it as *"the most direct route to `AD-IQ1_M` reaching 128K"*.
Measured since:

| where | split | acceptance | verdict |
|---|---|---|---|
| `v3-iq2xxs` @ 163,840, 10 blocks | `65+0` | **4 %** | slower than not offloading |
| `v3-iq2xxs` @ 163,840, 4 blocks | `65+0` | **no drafts at all** | level with baseline |
| `v3-iq1m` @ 196,608, 10 blocks | `65+0` | **100 %** | **+181.57 %** with n-gram |

Which half of the combination is responsible — artifact or depth — is unknown,
and no queued step separates them. The `AD-IQ1_M` route specifically is closed:
the ffn variant drops prefill from 240.6 to **8.56 tok/s**, and the `65+1`
baseline decodes at 6.08 tok/s regardless.

*Correction: report 24 §1 and §1b, report 23 §2.*

## 5. `v3-iq2xxs` holds `65+0` at 147,456, not 131,072

Report 21 walks a ladder in steps of 32,768 and records the deepest rung that
loaded. It measured 131,072 (`65+0`) and 163,840 (`62+3`) and **never tried
between them.** 147,456 is resident, and decodes faster than 131,072.

**Read every ceiling in report 21 as "at least this deep", never "no deeper".**

*Correction: report 24 §E3, banner on report 21.*

## 6. "60.8 tasks/hour" is `verified_tasks_per_hour` at `max_tokens 3072`

Written in several places as *"tasks/hour"* and once as *"the best number this
project has ever measured"*. The same artifact at the standard 8,192 budget
gives **48.5 verified / 26.5 merged**, at the same 90 % accept — so the budget
changed the wall clock, not the capability.

**Quote 48.5 when comparing against anything measured at 8,192.** The two are
not comparable, which is the whole point of standardising the budget.

*Correction: report 23, START-HERE banner.*

## 7. `--reasoning-budget 0` does not end the reasoning block

Documented as an immediate stop. Screened alone it ran to **24,709 characters**.
Paired with `--grammar-file` it returned `content_chars = 0` on 3 of 3 trials:
the model reasons freely, then emits end-of-turn at the point the grammar starts
to bind. **`-rea off` is the flag that ends the block.**

*Correction: `scripts/serve-v3-iq2xxs-fmt.ps1` header, report 22 §"the format
problem".*

## 8. Every decoder verdict is provisional until the warm-up check lands

`draft-mtp`, `draft-dflash`, `draft-eagle3` and `draft-dspark` were eliminated
on **160-token** timed generations. An external review of this model reports
speculation reaching rate only over a longer run — *"the MTP had gotten
extremely fast (91 tk/s vs 62 tk/s starting rate)"*. If that holds here, those
verdicts were measured before the thing being measured started working.

*Step W (`afk-q38-warmup.sh`) tests 160 vs 512 vs 1024. Until it reports, treat
every decoder elimination as unconfirmed.*

## 9. The greedy hash is comparable within a depth only

Across everything measured it takes one of exactly **two** values, and it
switches on things that are not the arm under test: at 131,072 every arm returns
`04E5CAB1…`, at 163,840 every arm returns `3EFE9395…`, and two different
artifacts return the same value. It is a divergence detector against a
same-depth, same-artifact baseline. It is not an equivalence proof and it means
nothing across depths.

*Correction: report 24 §6. The `ngram-cache` disqualification (§3 above) rests
on a same-depth comparison and is unaffected.*

## 10. The quality corpus measures a bare server; production runs an agent

`run_retry_bench.py` sends a 35-token developer message and grades one reply.
The real worker is a full Claude Code instance whose fixed prefix measured
**39,762–40,648 tokens** across four calls, with a tool loop, retries, and — per
`clink-subagents` §7 — `karpathy-guidelines` on every call and `tdd` whenever it
writes code.

**No quality number in this project describes the worker that ships.** A missing
code fence is a permanent failure in our harness and a self-correcting hiccup in
production.

*Plan 04 P5 and P6. Not yet corrected — this one needs a new harness, not a
banner.*

## 11. The `acceptance` column described one generation in five

`depth_sweep.run()` takes **five** timed generations and reports `tg_med` as the
median of all five — but computed `acceptance` from the **first one alone**. The
two columns were about different requests, and nothing said so.

Found 06:12 on 2026-08-21, while reading why an arm marked `acceptance: null`
decoded at **48.54 tok/s** against a 100 %-acceptance arm at 38.20.

**Every claim written that night from that column is weaker than it was stated:**

- *"`-ot ssm` collapses acceptance from 100 % to 4 %"* — one generation in five,
  though it did repeat at exactly 4.0 % across four boots.
- *"the four-block slice drafts nothing at all"* — the cold request drafted
  nothing. What the warm ones did was never recorded.
- *"acceptance may be a cheap coherence detector"* — built on the same column.
- *"speculation breaks whenever the arm reaches `65+0` at 163,840"* — the V1
  reading that prompted the check.

**Fixed** in `harness.draft_acceptance()`: weighted by drafts across every timed
generation, `None` when nothing drafted anywhere. `acceptance_cold` is still
written so rows from before the fix stay comparable. Five tests, suite at 108.

**Rows written before 2026-08-21 06:12 carry the old meaning.** Any conclusion
that rests on them needs re-measuring, not re-reading.

---

---

## What has NOT been contradicted

Stated so the list above is not read as "nothing here is reliable":

- **The residency cliff.** A GPU layer is worth about twice a CPU layer, and at
  depth far more: `65+1` on `AD-IQ1_M` at 131,072 decodes at 6.08 tok/s against
  26.50 resident. Measured many times, on several artifacts.
- **`q4_0` KV is the right choice.** It buys residency and no other KV type in
  this build has a fast kernel.
- **The 13.6 % drift floor**, as a floor. Whether it is now too conservative
  under `--fixed-text` is an open question, not a correction.
- **Bits per weight tracks quality** across five artifacts and two vendors —
  still a hypothesis, but nothing has contradicted it.
- **Depth is limited by VRAM, not the model.** `n_ctx_train = 262144`.
