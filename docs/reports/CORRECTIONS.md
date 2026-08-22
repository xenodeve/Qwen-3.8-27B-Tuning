# Corrections register — every published claim this project later contradicted

**Read this before trusting any number in reports 00–31.** These reports were
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

## 12. The model does not loop in its reasoning — it thinks, and finishes

Asserted in three documents on the strength of three numbers: reasoning up to
16,341 characters, tool round trips completing 10/16, and three corpus tasks
producing no output in 190–248 s. **None of them is the reasoning text**, and
the probe kept only 400 of its characters.

A full capture: **6,899 characters, 0.00 % line repetition, `finish_reason:
stop`**, and the code it then wrote passes. Long reasoning is this model's
documented normal mode. The same task takes 62.6 s direct and 247.6 s through
OpenCode with nothing to show, so **the failure is in the agent loop.**

*Full record: [post-mortem](2026-08-21-inferred-looping-from-three-numbers.md).
Fixed in `bench/protocol_gate.py` — the trace is kept and
`reasoning_repetition_pct` is recorded on every row.*

---

## 13. Results measured below ~300 MiB of free VRAM are the machine falling off a cliff

Across **all 235 sweep rows this project has recorded**, the six with free VRAM
under 300 MiB are the only ones whose prefill fell under 700 tok/s, and the two
under 250 MiB are the only ones under 120:

```text
  free 199 MiB   pp 106.3 tok/s   prefill 875.6 s    <- q4_0, IQ2_S @131,072
  free 226 MiB   pp 112.8         prefill 825.5 s    <- fit-192-ngram, round 1
  free 307 MiB   pp 841.1         prefill 110.7 s    <- SAME ARM, round 2
  free >= 300    n=229            pp median 837
```

**Same flags, same artifact, same depth, seven times apart on both axes.** The
81 MiB between rounds is the Windows desktop moving, not a setting. The
signature fits WDDM paging compute buffers out to system RAM rather than
failing the allocation.

**What this puts in doubt.** Any conclusion drawn from an arm whose row shows
`vram_free` under ~300 — including *"`-ot ffn` drops prefill from 240.6 to 8.56
tok/s"* (`CORRECTIONS.md` §4), which ran at 382 MiB: above the line, but not by
much, and the line is not sharp.

**What it explains.** `--fit-target`'s default of 768 MiB, which this project
spent a night calling an untested reserve, is the buffer that keeps the machine
off this cliff. Lowering it to 192 does free VRAM and does buy residency — and
it removes the margin that makes a result reproducible.

**Consequence for every future measurement:** `vram_free` is not a diagnostic
column, it is a **validity condition**. A row below the line is not a slow
result, it is a void one.

---

---

## 14. Free VRAM at settle is a risk indicator, not a validity condition

Section 13 above ends: *"A row below the line is not a slow result, it is a void
one."* Seven rows measured on 2026-08-21 falsify that in both directions.

```text
  UD-IQ2_S @131,072, --fit-target 192            free 233 MiB   pp 817.1 tok/s
  UD-IQ2_S @131,072, --fit-target 192 -ub 128    free 285 MiB   pp 721.8
  UD-IQ2_S @131,072, --fit-target 192 -ub 128    free 291 MiB   pp 188.1
  UD-IQ2_S @131,072, --fit-target 192            free 424 MiB   pp 836.1
```

**Two of the three rows under 300 MiB are among the fastest in the set, and the
slowest row under 300 MiB has more free VRAM than either of them.** 233 MiB ran
4.3x faster than 291 MiB, on the same artifact at the same depth in the same
pair of rounds.

So `free` at settle does not order the outcomes. The correlation reported in
section 13 was real in the 235-row corpus; the causal reading laid on top of it
-- a threshold that voids a row -- does not survive new data. Whatever drives
the collapse is not captured by a single reading taken once the server has
loaded. The desktop moves *during* a run, and nothing here samples that.

**What survives.** A row under 300 MiB is still the only place a collapse has
ever been seen, so it remains a reason to repeat a measurement rather than
publish it. That is a weaker claim than section 13 made, and it is the one the
data supports.

**What this puts in doubt.** Any decision made by discarding a row for sitting
under the line, and the reading of `--fit-target 768` as "the buffer that keeps
the machine off the cliff" -- `192` produced the two fastest 131,072 rows this
project has recorded.

Raw: `qwen38-tuning/results/iq2s-131072-residency.jsonl`. Report 25.

---

## 15. Qwen Code's request is 54,499 tokens, not 16,796 — and 32,768 breaks it

Report 25 read this server-log line as the size of Qwen Code's first turn:

```text
  prompt eval time = 21630.42 ms / 16796 tokens (776.50 tok/s)
```

**16,796 is what was left to prefill after cache reuse, not the size of the
request.** The request is **54,499 tokens**, and the correction arrived as a
failure rather than a slow number only because the window had already been
lowered on the strength of the wrong reading:

```text
  API Error: 400 request (54499 tokens) exceeds the available context size
  (32768 tokens), try increasing it
```

**What was published and is withdrawn.** That `-c 32768` is the profile for Qwen
Code, and the advice to set `contextWindowSize` in `~/.qwen/settings.json` to
match. Following it makes Qwen Code fail on every turn. The window sizes
measured since:

| harness | one request | fits 32,768 |
|---|---|---|
| lean OpenCode, longest of 10 real tasks | 13,741 | yes |
| Qwen Code | 54,499 | no |
| Claude Code with MCP loaded | 54,685 | no |
| OpenCode default profile, prefix alone | 99,073 | no |

**What survives.** The measurement the profile was built on is unaffected --
32,768 really does give 1,134-1,168 tok/s prefill and 45.3-50.3 tok/s decode
against 776-836 and 23.2-23.9 at 131,072. Only the claim about which harness
fits in it was wrong.

**The instrument shares the blame and has been fixed.**
`scripts/bench-cold-start.py` took the FIRST prompt-eval line of a run as "the"
prefill. A harness makes several calls per turn and Qwen Code's first is 603
tokens, so the real 54,499-token call was reported as harness overhead. It now
reports the largest call and the sum, and the same shape is already on file as
instrument fault 9.

---

## 16. "The cold start was a second subagent, not the server" — the second half was wrong

Report 26 and issue #8 concluded that the cold start belonged to the harness and
shipped a cure that switched off `memory.enableManagedAutoMemory`, costing Qwen
Code the ability to update its own memories.

**The developer refuted the framing in one sentence:** the same unmodified Qwen
Code runs fine against Qwen3.8-27B FP8 on a gateway. So the subagent evicts the
cache there too, and nobody notices, because a datacenter endpoint re-prefills
41,000 tokens in about a second. **Our prefill rate is what turns an eviction
into 41.4 s.** Both halves are necessary; only one was named.

That reframing made a server-side cure legitimate, and one had never been tried:

```text
  -c 98304  -np 1                    ~41,300 tok prefilled, 41.4 s   wall 58-71 s
  -c 110592 -np 2 -sps 0.95           0 tok, FULL CACHE HIT          wall 5.9-17.1 s
```

with **every Qwen Code memory feature left on**. `--slot-prompt-similarity`
defaults to 0.10, low enough that two prompts sharing a tool catalogue land on
the same slot — which is why an earlier `-np 2` measurement at the same depth,
with the default, changed nothing and was written off.

**A third instrument fault nearly buried it.** `bench-cold-start.py` reported
"no prompt eval line" as a failure. llama-server prints no timing line when there
is nothing to prefill, so a **total cache hit — the best outcome available —**
was recorded identically to a request that never arrived. Six such rows were read
as six failures. It now separates them by return code and wall clock.

**What survives from report 26.** Every negative result: the server's cache works
(53.9 s then 0.4 s on replay), `--cache-ram -1` and `--cache-reuse 256` are
regressions, `-ub` does nothing, the memory *files* are irrelevant, and the
prompt does not vary between runs. And the mechanism is still the subagent
evicting the slot — it is the *conclusion drawn from it* that was too narrow.

`scripts/worker-iq2s-2slot.ps1` is the profile. Report 26, issue #9.

---

## 17. The two-slot profile does not fit a real session, and the benchmark is why

`worker-iq2s-2slot.ps1` shipped in §16 with `-c 110592 -np 2`, giving 55,296 per
slot, sized against a Qwen Code request measured at **54,499 tokens**. The
developer's actual interactive session is **71,910**:

```text
  API Error: 400 request (71910 tokens) exceeds the available context size
  (55296 tokens), try increasing it
```

**Two slots cannot serve that conversation on this card.** Each would need
71,910, so 143,820 of context; the deepest this GPU holds fully resident is
131,072, and only with `--fit-target 192`, which settles at 233-424 MiB free.
The profile is correct only for a session small enough to fit half the window,
and a session grows.

**This is the third time a measured prompt size was smaller than reality**, and
each time the number came from `bench-cold-start.py` driving Qwen Code with
`-p "reply with exactly the word: ok"` from a directory with no project history:

| claimed | actual | how it surfaced |
|---|---|---|
| 16,796 | 54,499 | a 400 at `-c 32768` (§15) |
| 54,499 | 71,910 | a 400 at `-np 2`, 55,296 per slot (here) |

**A one-line synthetic prompt is not a session.** The harness's request size is
dominated by conversation history and project context, neither of which the
benchmark creates. Every window sized from it is sized from a floor.

**What survives.** The mechanism and the cure in §16 are unaffected: two slots at
`-sps 0.95` really do stop the eviction, measured at 0 tokens prefilled. It is
the *capacity* that was picked from the wrong number. For a 71,910-token session
the choices are a single slot and the cold start, or
`memory.enableManagedAutoMemory` off, which removes the eviction at any size.

---

## 18. §8 is answered for `draft-mtp`, and the DFlash 2 screen never happened

§8 held every decoder verdict open on the grounds that they were measured on
160-token generations. Re-run on 2026-08-21 at `N_PREDICT = 1024`, on
`UD-IQ2_XXS` so headroom could not be the confound:

```text
  ngram-mod    64.83 / 64.91 tok/s   (160-token figure: 65.06 / 60.33)
  draft-mtp    54.18 / 54.08         (160-token figure: 51.14 / 52.47)
```

**The long run gives MTP 4 %, not the +47 % the external report described**, and
it still finishes 17 % behind. §8's doubt is resolved against `draft-mtp` rather
than for it.

The cliff doubt is refuted too: at 131,072 MTP decodes 6.21 and 6.09 tok/s
against `ngram-mod`'s 45.87 and 48.11, with **467–773 MiB free on every row** —
comfortably above the line. The original −71 % was generous.

**What is withdrawn instead is the DFlash 2 row.**
`docs/results/02-decoders.md` records *"drafter 1.06 GiB, screened, not
competitive on 12 GB"*. The artifact does not load at all on build 10472:

```text
  E llama_model_load: done_getting_tensors: wrong number of tensors;
    expected 81, got 58
```

The vendor's announcement gives the reason — llama.cpp support for DFlash 2
arrives with **PR #27342**, which this build does not carry; the `draft-dflash`
flag here implements the first DFlash. **A screen that could not run is not a
screen.** The honest state is *cannot load, needs a newer llama.cpp*, and the
claimed 2.7–3.4× makes it worth revisiting when the build moves.

Report 28.

---

## 19. "`UD-IQ2_S` has never been loaded once" — it has, dozens of times

**Where it was written.** `docs/OPEN-WORK-LEDGER.md`, as a 🔴 UNTRACKED row:
*"8.37 GB, in the local cache since 2026-08-20 01:36, never loaded once."* From
there it was copied into `docs/plans/06-REAL-TASK-BENCHMARK.md` twice on
2026-08-22, including as the justification for which rung to benchmark first.

**What contradicts it.** The repository's own results, which were already
present when the row was written or shortly after:

| evidence | count |
|---|---|
| result files carrying `v3-iq2s` rows | 6 (`iq2s-131072-residency`, `iq2s-prefill-microbatch`, `kv-iq2s-128k`, `prefill-kv-type`, `ctx-ceiling-q38`, `arena-v3`) |
| measured rows, all `loaded: true` | 38+ |
| server logs naming it | dozens — `arena-r1..r3-v3-iq2s`, `ceil-v3-iq2s-*`, `depth-iq2s-*` |
| worker profiles serving it | 4 (`worker-iq2s-quality.ps1`, `-fast`, `-2slot`, `serve-v3-iq2s.ps1`) |

Sample rows: 26.61 tok/s at ctx 98,304 with 400 MiB free; 49.84 tok/s at 32,768
with 2,267 MiB free. `arena-v3.jsonl` records the artifact by full path,
`Qwen3.8-27B-UD-IQ2_S.gguf`.

**Two errors, and the second is the instructive one.**

The row was **stale** — plausibly true on 2026-08-20 and falsified by work done
after, with nobody returning to update it. That is the ordinary failure the
ledger exists to catch and did not.

The second error is mine and worse: **the claim was carried forward into a plan
without being checked.** `CLAUDE.md` says to read this file before quoting any
number, and the claim's register never improves by being repeated. A guess in a
ledger row became the stated reason a benchmark would start with one artifact.
It was caught by the developer, who remembered the actual history, not by any
check in the repo.

**What is actually open.** Not "has IQ2_S been tested" — it has, on throughput.
The open question is the **trade**: `UD-IQ2_S` (7.80 GB) was given up for
`UD-IQ2_XXS` (6.77 GB) **deliberately, to free VRAM for a drafter**, and the
drafter — DFlash2 — only became loadable on 2026-08-22 (§18, issue #17). Both
sides of that trade now exist and both fit. **Neither has a task-success
number**, which is what `docs/plans/06-REAL-TASK-BENCHMARK.md` §3.5 is for.

**Guarded by** `scripts/audit-stale-claims.py`, rule `iq2s-never-loaded`.

---

## 20. The real-code benchmark prompt was built from the benchmark's own source

**The claim.** Report 29's real-code figures — `ngram-mod` 53.0/52.5/49.3,
`draft-dflash` 69.5/69.1/69.8, the pair 78.9/78.8/72.2 tok/s — read as
properties of those decoders at ctx 16,384.

**What contradicts it.** Re-running the pair with **byte-identical arguments**
on 2026-08-22 produced **100.5 / 105.4 / 105.9 tok/s**. A 33 % gap on a project
whose stated noise floor is 13.6 %.

Not thermal: 49 °C, `SW Power Cap: Not Active`, `HW Thermal Slowdown: Not
Active`, every throttle counter 0 µs. Not the arguments: the two runs' `args`
fields are string-identical. Not the split: `65+0` in both.

**The cause.** `dflash2_arena.filler(n, "real-code")` built its prompt by
reading *this benchmark's own source* — `harness.py`, `depth_sweep.py`,
`model_arena.py`, `opencode_corpus.py`, `kv_sweep.py` — and slicing the first
`n * 3` characters.

Between the two runs **3,045 bytes were appended to `harness.py`**
(24,306 → 27,351) to add a stats parser. The prompt budget is 24,576
characters, so the workload moved from *`harness.py` plus the first 270
characters of `depth_sweep.py`* to *the first 24,576 characters of `harness.py`
alone*. Different text, different n-gram hit rate, different acceptance.

**I built a benchmark whose workload is generated from files I edit while
running it.** Every real-code number was silently tied to the state of `bench/`
at that instant.

**What survives.** The **paired, within-round** verdicts of both runs, because
one run sees one prompt: report 29's `draft-dflash` **+34.7 %** and the pair
**+48.5 %** over `ngram-mod` still stand. What does not survive is any absolute
rate quoted across runs, and any comparison between a pre- and post-edit run.

**The fix.** `bench/corpora/real-code.txt` is the corpus frozen as a committed
file, reconstructed from the tree at commit `674ea4b` — the state report 29 was
measured on, so its numbers stay interpretable. Every row now carries
`corpus`, a hash of that file. Verified: on the frozen corpus the pair measures
**79.7 tok/s** against report 29's **78.9**, and the decline rate returns to
**93.7 %** against **94.3 %**.

**Guarded by** `bench/tests/test_corpus_frozen.py` (6 tests), which pins that
the prompt is a pure function of the frozen file and that the hash is reported.

---

## 21. "`--spec-ngram-mod-n-match 12` — the same cap, chosen independently"

**The claim.** [Report 30](30-SYV-RTX3090-REFERENCE-REVIEW.md), 2026-08-22, on
the RTX 3090 stack's lookup patch: *"llama.cpp's `ngram-mod` is the same
algorithm, and our tuned profile already uses `--spec-ngram-mod-n-match 12` —
the same cap, chosen independently."* Written as reassurance: two projects
reaching the same number was read as evidence the number was right.

**What contradicts it.** The sweep the same report asked for.
`results/sweep-ngram-nmatch.jsonl`, 12 rows, ctx 16,384, frozen corpus, three
rounds, arms rotated and paired:

| `n-match` | rounds (tok/s) | vs our 12 |
|---:|---|---|
| **24 — the llama.cpp default** | 94.5, 96.3, 94.2 | **+34.6 % [+31.4, +40.8] RESOLVED** |
| 16 | 69.2, 69.7, 69.5 | −1.5 %, within the floor |
| 12 — what we ship | 71.7, 73.3, 66.9 | baseline |
| 8 | 56.7, 62.7, 61.5 | **−14.5 % [−20.9, −8.0] RESOLVED** |

**The cause — two flags that share a number and nothing else.** Their
`LOOKUP_NMAX` caps a *longest-match search* with ties broken by recency, so a
lower cap really does bias toward recent matches. Our `n_match` is the **hash
key width** of a keyless 4M-entry table (`common/ngram-mod.cpp:15-25`, `37-41`):
there is no length dimension to cap and recency is unconditional at every value,
because `add()` overwrites the slot. Lowering it does not buy recency. It buys
**key collapse** — more distinct contexts folding onto one slot, each stealing
the others' successor.

The counters say it directly. At `8` ngram fires *more*: 43 drafts against 29 at
`24`. Each draft is worth far less — mean accepted length **23.45 → 8.95**,
accepted-token yield **651/921 → 342/1306** — and the draft calls needed for the
same 512 tokens rise **475 → 649**.

**What this does not retract.** The `+48.5 %` for `draft-dflash,ngram-mod` over
`ngram-mod` alone, which was measured at `n-match 12` on both sides of the pair.
Nor report 30's reading of *their* patch, which is accurate about their code.
What is retracted is the inference that agreement between the two numbers
validated ours.

**Where it was already written down and not acted on.** The flag-semantics read
of the same day states it plainly — *"n_match changes key SPECIFICITY only… you
are buying only the 'shorter' half, and paying for it with key collapse"* — in
`researchs/llamacpp-flag-semantics-2026-08-22.md`, misreading (3). Report 30's
sentence was written anyway. **A source read does not correct a claim unless
somebody goes back to the claim.**

**Not yet a config change.** The verdict is ctx 16,384 and the served profiles
run 65,536–98,304; this project's own rule is that a verdict at one depth does
not transfer. The four `worker-*.ps1` scripts still carry `12` and stay that way
until measured at depth — recorded as open work, not as a pending edit.

**Guarded by** `scripts/audit-stale-claims.py`, rule `nmatch-12-independent`.

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
