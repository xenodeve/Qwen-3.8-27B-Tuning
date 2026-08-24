# Corrections register — every published claim this project later contradicted

**Read this before trusting any number in reports 00–32.** These reports were
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

## 22. "The mechanism argues `n-match 24` should widen its lead at depth"

**The claim.** Written 2026-08-22 in `results/02`, `results/08` and report 31
§5b, after `n-match 24` measured +34.6 % RESOLVED at ctx 16,384: *"the mechanism
argues 24 should widen its lead at depth — a fuller table means more distinct
contexts colliding on a short key."* Labelled a hypothesis in every copy, and
the recommendation that followed — *"pick `n-match 24`"* — was scoped to 16,384.

**What contradicts it.** The same sweep at **ctx 65,536** on the deep corpus,
`results/sweep-ngram-nmatch-65536.jsonl`, three paired rounds, every arm `65+0`:

| `n-match` | ctx 16,384 | ctx 65,536 |
|---:|---|---|
| 24 | **+34.6 % RESOLVED** | **−9.7 %, null** |
| 16 | −1.5 %, null | **+67.5 % RESOLVED** |
| 12 (ours) | baseline | baseline |
| 8 | −14.5 % RESOLVED | −14.5 %, null |

**The optimum moved from 24 to 16 and the prediction pointed the other way.**

**Why it was backwards.** The reasoning assumed the binding constraint at depth
is key collision. It is **fire rate**. A longer key is a stricter requirement
and a deeper window does not rescue it: at 65,536 `24` fires **18** times
against `16`'s **39** — 97.0 % decline against 91.3 % — for almost the same
accepted length (19.78 against 21.59). The specificity costs more hits than the
collisions it avoids.

**What this does not retract.** The 16,384 numbers, which were paired,
replicated in a later round set, and correctly scoped. Nor `CORRECTIONS` §21 —
our shipped `12` is beaten at *both* depths, by 24 at one and by 16 at the
other. It is the second-worst arm at the depth we actually serve.

**The lesson is one this repo already had written down.** `CLAUDE.md`: *"A
verdict at one depth does not transfer to another."* The hypothesis was an
argument for why *this* case would be the exception. **A mechanism story is not
a measurement, and being able to tell one in either direction is exactly why the
rule is stated without exceptions.**

**Guarded by** `scripts/audit-stale-claims.py`, rule `nmatch-24-at-depth`.

---

## 23. The 13.6 % noise floor is a ctx 16,384 number

**The claim.** Used project-wide as *the* drift floor —
`harness.NOISE_FLOOR_PCT`, quoted in `CLAUDE.md`, the bench README and every
report: *"Effects below 13.6 % are noise."* It was derived from 25 boots of one
control config **at ctx 16,384**.

**What contradicts it.** At ctx 65,536, decode is still deterministic at
temperature 0 — every arm's per-implementation counters are byte-identical
across all three rounds — so the entire spread below is the clock:

| `n-match` | within-arm spread @ 16,384 | @ 65,536 |
|---:|---:|---:|
| 24 | 2.2 % | **23.5 %** |
| 16 | 0.8 % | 10.3 % |
| 12 | 9.5 % | **39.5 %** |
| 8 | 10.6 % | **48.9 %** |

**The same arm, unchanged in every counter, varies by up to 48.9 % between
boots at 65,536.** A 13.6 % floor at that depth would resolve effects that are
entirely drift.

**What it does not invalidate.** Every verdict this project has resolved at
16,384, where the floor was measured and where the observed within-arm spread
is 0.8–10.6 %.

**What it does invalidate.** Any future use of the 13.6 % figure at depth, and
the magnitude of the one depth verdict taken so far — `n-match 16` at
**+67.5 %** is directionally safe (its worst round beats every other arm's best
by 32 %, which needs no floor) but **the number must not be quoted**.

**Not fixed in code, deliberately.** Three rounds cannot re-derive a floor.
`harness.paired_deltas` still defaults to 13.6 % and takes `floor_pct`
explicitly; inventing a depth-scaled constant from this sample would be the
same error one level up.

**Guarded by** `scripts/audit-stale-claims.py`, rule `noise-floor-at-depth`.

---

## 24. The five real tasks did not "change nothing" — the harness was watching the wrong tree

**The claim.** Report 31 §6, report 32 §2, `results/real-task-bench.jsonl`, and
every summary written from them: five real GitHub issues ran 1,427–2,400 s each,
**changed no files**, three exiting `rc=0`, with a green baseline and
`0 WINDOW_BOUND` — *"no mechanism is attached"*. It was treated as the project's
central open question and reported as such to an external reviewer.

**What contradicts it.** `bench/edit_canary.py`, run 2026-08-23 at ctx 16,384
against a fresh clone of `openclink` with the instruction to append one word to
`README.md`:

| | `cwd=` only (as `real_task_bench` ran it) | with `--dir` |
|---|---|---|
| outcome | `EDIT_NO_DIFF` | **`EDITED`** |
| rc | 1 | **0** |
| diff_bytes | **0** | **251** |
| wall clock | 130.3 s | **32.8 s** |
| the live `C:\AI` tree | **modified** | untouched |

**The first run edited `C:\AI\README.md` — the live repository.** Its transcript
says so in its own words (*"Target file: `C:\AI\README.md`"*), and `git status`
confirmed it: `M README.md`, first line ending in `CANARY`. **Reverted, verified
clean.** The second run, with the directory pinned on the argv, produced a real
diff **inside the clone** and left the live tree alone.

**The cause, and it was written down two days earlier.**
`bench/opencode_corpus.py:50-62`, dated 2026-08-21: OpenCode keeps a per-project
server alive between invocations, `run` attaches to whichever is already
listening, and **that server carries the project root it was first started
with.** The same docstring records the same symptom — *"every answer landed in
`C:\AI\qwen38-tuning` while the harness looked in the task directory and
recorded 'no file written' on work the model had done correctly."*

`opencode_corpus.py` defends itself by killing the server once before a run.
**`real_task_bench.py` never did.** It passed `cwd=<clone>`, which OpenCode does
not honour, and it is the driver that produced all five rows.

**What is retracted.** Every conclusion drawn from those five rows about the
model, the quantisation, the context window, or the workflow. `diff_bytes: 0`
measured where the harness looked, not what the worker did. **Three tasks
exiting `rc=0` was read as the worker deciding it was finished; it may equally
have been the worker finishing correctly in the wrong tree.**

**What survives.** The wall-clock times and the context high-water figures —
those came from the process and the server, not from the diff.

> **~~Two independent explanations for the same zero~~ — there is one.**
> This entry originally added that decode collapses to 2.8–5.0 tok/s at ctx
> 98,304 and *"stands on its own data"*. **It does not — see
> [§26](#26-decode-collapses-to-2850-toks-at-the-window-we-serve--the-window-is-fine-the-drafter-is-not).**
> Every row of that sweep loaded the DFlash2 sidecar; with `ngram-mod` alone the
> same depth returns 96.92 tok/s over 6 of 6 rounds. **The directory fault is
> the only established explanation for the five zero-diff rows**, so the next
> real-task run has one variable, not two.

**The fix.** `edit_canary.worker_argv()` puts `--dir <absolute path>` on the
command line and **raises on an empty directory rather than defaulting**, since
defaulting is exactly how the live tree was edited. `real_task_bench.py` now
calls it. Pinned by `bench/tests/test_worker_workdir.py`, which deliberately
does **not** assert on `cwd` — `cwd` is the thing that looked right and was not,
so a test built on it would have passed throughout the incident.

**Guarded by** `scripts/audit-stale-claims.py`, rule `real-task-zero-diff`.

---

## 25. "Chars per token is ~7.0–7.4, not 3" — it is ~3.4, and the harness was never wrong

**Where it was published.** [Report 32 §4](32-BENCHMARK-STATUS-BRIEF.md), under
*Corpus and instrument limits discovered*:

> **Chars per token is ~7.0–7.4, not 3.** `dflash2_arena.filler()` assumed 3, so
> every run labelled "ctx N" fed a prompt of about **40 % of N**

Repeated in [`04-context-depth.md`](../results/04-context-depth.md) §"What depth
is worth" and in the 2026-08-23 hand-off.

**What is actually true, measured 2026-08-23.** `dflash2_arena.py:478` reads:

```python
prompt = filler(int(ctx * 0.5), regime)
```

The arena asks for **half** the context by design, leaving room for the
generation. So at ctx 98,304 the request is `filler(49,152)`, which at the
assumed 3 chars/token produces **147,456 characters** — not 294,912. Against the
43,162 tokens the log reports, that is **3.42 chars/token**.

The published 6.83 comes from dividing by `98,304 × 3`: **the 0.5 was dropped.**
All three depths land in the same place once it is put back:

| labelled ctx | chars actually sent | tokens (from log) | chars/token |
|---:|---:|---:|---:|
| 16,384 | 24,576 | 6,621 | 3.71 |
| 65,536 | 98,304 | 28,122 | 3.50 |
| 98,304 | 147,456 | 43,162 | 3.42 |

**Measured directly, independent of the arena.** `bench/prefix_cache_depth.py`
sends a character budget and reads the token count back from the server:
**28,000 chars → 8,147 tokens (3.44)** and **150,000 chars → 44,255 tokens
(3.39)**, same corpus, same boot. `results/prefix-cache-depth.jsonl`.

**What survives, and what does not.**

- ✅ **The token counts are right.** 6,621 / 28,122 / 43,162 were read from
  server logs, not derived, and nothing here touches them.
- ✅ **"A run labelled ctx N fed about 40 % of N" is right** — 43,162 / 98,304 =
  43.9 %. Every depth-label caveat resting on it still stands.
- ❌ **The reason given for it is wrong.** It happens because the harness
  *deliberately* asks for half the window, not because a constant was
  mis-estimated by 2.3×.
- ❌ **`filler()`'s assumption of 3 was never the fault.** At a real 3.4 it is
  about 12 % low, which is a rounding choice, not an instrument fault. Report 32
  named it as one of the session's discovered instrument limits; it is not one.

**Why this one is worth recording even though no number moved.** The correction
that reached the ledger and the hand-off was *"the harness assumed 3 and reality
is 7"* — an accusation against a tool that was behaving correctly. Acting on it
would mean 'fixing' `filler()` to send 2.3× more text and silently doubling the
depth of every future row while the label stayed the same. **A wrong explanation
attached to a right number is not harmless; it is a wrong instruction to the
next person who reads it.**

**Guarded by** `scripts/audit-stale-claims.py`, rule `chars-per-token-7`.

---

## 26. "Decode collapses to 2.8–5.0 tok/s at the window we serve" — the window is fine; the drafter is not

**Where it was published.** [`04-context-depth.md`](../results/04-context-depth.md),
*"The window we serve is the one that does not work"*, and the
🔴 UNTRACKED row of the same name in
[`OPEN-WORK-LEDGER.md`](../OPEN-WORK-LEDGER.md):

> ctx 98,304: **13 of 16 measurements timed out** … and the three that finished
> decoded at 2.8 / 5.0 / 4.2 tok/s against a median **75.2** at 16,384 … every
> arm `65+0`, acceptance still 59–77 %, **so neither residency nor speculation
> explains it**

**What the sweep could not see.** Every one of those sixteen rows ran
`--spec-type draft-dflash,ngram-mod` with the DFlash2 sidecar loaded — readable
in each row's own `args` field. **No arm at that depth ever ran without it**, so
"depth" and "drafter" were never separated, and the clause *"neither residency
nor speculation explains it"* rests on a comparison that was never made.

**Measured 2026-08-23**, `results/decoders-98304.jsonl`, 24 rows, six paired
rounds, same ctx, same corpus (`real-code-deep`, sha `1a3ae4b813dd8447`), same
binary, arms alternated within each round:

| arm | ok | timed out | tg samples | median | free MiB after load |
|---|---:|---:|---|---:|---|
| `none` | 6/6 | 0 | 33.53 · 33.55 · 33.58 · 33.80 · 34.15 · 34.76 | **33.69** | 800–1,935 |
| **`ngram-mod`** | **6/6** | **0** | 96.14 · 96.40 · 96.80 · 97.04 · 98.85 · 98.88 | **96.92** | 769–2,117 |
| `dflash2` | 5/6 | 1 | 0.64 · 47.31 · 49.31 · 52.82 · 53.62 | 49.31 | **45–376** |
| `dflash2+ngram` | 4/6 | 2 | 1.46 · 4.53 · 6.78 · 93.29 | **5.66** | **153–240** |

**The two groups do not overlap on free VRAM, and that is the whole finding.**
Arms without the drafter sit at 769–2,117 MiB, finish 12 times out of 12, and
spread 3–4 %. Arms with it sit at 45–376 MiB every single time, time out 3 times
in 12, and spread **146×** — 0.64 to 93.29 tok/s on identical flags.

**What is retracted:**

- ❌ **"Decode collapses at the window we serve."** The profile we actually
  serve is `ngram-mod` alone, and it returns **96.92 tok/s median at ctx
  98,304** — *faster* than the 75.2 median recorded at 16,384. Depth is not the
  variable.
- ❌ **"Neither residency nor speculation explains it."** Speculation explains
  it. The sweep held it fixed and could not see it.
- ❌ **The 2.8–5.0 range as a property of the window.** It is the lower tail of
  a bimodal distribution belonging to the drafter arms.

**What survives:**

- ✅ **The numbers themselves.** 2.76 / 5.01 / 4.18 were real measurements of
  `draft-dflash,ngram-mod` at that depth, and this run reproduces that regime
  (1.46 / 4.53 / 6.78) alongside the fast one (93.29).
- ✅ **`65+0` on every row.** Confirmed again here — and confirmed *not* to be
  the explanation, since the fast and slow rounds of the *same arm* allocate
  byte-identically: model 6,521.13 MiB, KV 1,728.00, RS **748.12 on CUDA0**,
  compute 472.27, no OOM, and `--fit` logging *"will leave 849 >= 768 MiB, no
  changes needed"* in both.

**The mechanism, as far as it is established.** With a model-based drafter
`n_rs_seq` is 4, so the server writes a **`created speculative checkpoint …
size = 149.626 MiB`** — one full recurrent-state plane — every few generated
tokens. With `ngram-mod` alone `n_rs_seq` is 0 and no such checkpoint exists. In
the slow rounds the gap between checkpoints reaches **30.41 s** against a median
2.35 s in the fast ones, which is a stall rather than uniform slowness. Sampled
live during a slow arm: `free 196 MiB · util_gpu 100 % · util_memory 3 % ·
2820 MHz · 70.18 W · 57 °C` — matching `gpu-trace-98304.jsonl`'s medians
(246 MiB, 100 %, 4 %, 2820 MHz, 75.2 W, 57 °C) in every column.

**What is NOT established.** Why some drafter rounds escape — 93.29 tok/s at
240 MiB free while another round managed 1.46 at 153 MiB. There is no clean
threshold, only a band in which the outcome is unreliable. **The 100 % / 3 %
split is not the memory-bound signature `04-context-depth.md` called it** — a
memory-bound decode shows high *memory* utilisation, and this shows 3 %. It is a
card spinning, not a card working.

**Consequence for shipping, stated at the precision the measurement supports.**
These rows are **`UD-IQ2_XXS` at ctx 98,304**, and **no worker profile serves
that pairing** — `worker-iq2xxs-deep` runs that artifact at 131,072,
`worker-iq2s-quality` runs 98,304 on the 1.1 GB larger `UD-IQ2_S`.

- **Measured:** with this artifact at this depth, adding DFlash2 costs 94 % of
  decode and a one-in-four chance of not finishing — the exact opposite of its
  **+48.5 %** at ctx 16,384. The sharpest instance yet of the rule that a
  verdict at one depth does not transfer.
- **Inferred, and labelled as such:** the failures track free VRAM, and
  `UD-IQ2_S` is **larger** than the artifact measured, so a drafter beside it
  at 98,304 would have *less* headroom, not more. That argues the same way,
  and it has not been run.
- **Unchanged:** all four profiles run `ngram-mod` alone and none was
  modified. Nothing here licenses a claim about their absolute rates.

**Guarded by** `scripts/audit-stale-claims.py`, rule `decode-collapse-98304`.

---

## 27. "`--fit` follows the boot VRAM" — it does not. It has seen 11,069 MiB every time, 552 times

**Where it was published.** `CLAUDE.md`, the engineering north star, and repeated
across at least five documents:

> **Never compare raw decode across boots.** Free VRAM at boot moves
> 9,326–10,732 MiB and `--fit` follows it.

**Measured 2026-08-23**, by reading every server log this project has kept.

```
free-at-boot as llama.cpp reports it, across ALL 552 logs:
    552 x "RTX 4070 SUPER (12281 MiB, 11069 MiB free)"

fit decisions on our own artifact (dflash2-*.log, 150 boots with a fit pass):
    148 x "will leave N >= 768 MiB of free device memory, no changes needed"
      2 x fit actually acted -- both `n-7-clamp` at ctx 65,536, already in the ledger

layer split across those boots:
    301 x 65/65 (target)   224 x 6/6 (drafter)   8 x anything else
```

**The two numbers measure different things, and the wrong one was in the rule.**
9,326–10,732 MiB is `nvidia-smi` — free VRAM on the *card*, desktop included,
and it does move. **11,069 MiB is what CUDA reports to the process**, and it is
the number `--fit` reasons from. It has not varied once in 552 launches, so
`--fit` reaches the same decision every time and says so.

**What survives, and what does not.**

- ✅ **"Never compare raw decode across boots" still stands.** The spread is
  real and measured: 13.6 % peak-to-peak at ctx 16,384, up to 48.9 % at 65,536
  with byte-identical counters (§23). Nothing here touches that.
- ✅ **`nvidia-smi` free VRAM does move**, and 2026-08-23 caught it moving *mid
  run*: three `-ub 128` boots with byte-identical allocation read `free_after`
  of 759, 757 and **1,214 MiB**, and the third ran 6 % faster.
- ❌ **`--fit` is not the mechanism.** It cannot follow a number it never sees
  change. Whatever produces the boot-to-boot spread, this is not it.

**Why a wrong mechanism costs more than a wrong number here.** The stated cause
implies a fix — pin `-ngl`, turn `--fit` off, and the drift goes away. The RTX
3090 scan proposes exactly that and rates it *"highest value on this list for
measurement integrity"*. **It was tried on 2026-08-23 and changes nothing**:
`pinned_alloc_preflight.py` boots both forms at ctx 98,304 and they agree on
every observable — `65+0`, `n_ctx 98304`, model 6,521.13 MiB, KV 1,728.00,
compute 472.27, `free_after` 1,427. There was nothing to pin, because `--fit`
had already decided to leave everything alone.

**What this reopens.** The real source of the boot-to-boot spread is **unknown
and now unattributed**. The best current lead is contention from the desktop —
the ledger's *"1,650–2,200 MiB, the largest untouched lever on this machine"* —
supported by the `-ub 128` round above and by `gpu-trace-98304.jsonl`'s
signature of 100 % GPU utilisation at 4 % memory utilisation and 76 W. **That is
a hypothesis.** It should not be written into a rule the way this one was.

**Guarded by** `scripts/audit-stale-claims.py`, rule `fit-follows-boot-vram`.

---

## 28. "The 5060 Ti is 4× slower than the 4070 SUPER" — half that comparison was between two different instruments

**Where it was published**, on 2026-08-23, in `docs/results/09-hardware.md`, the
open-work ledger, issue #40 and the commit that shipped them:

> | | 5060 Ti (Ada PTX, JIT) | 4070 SUPER (native SASS) |
> | prefill, 43,898 tokens | **146,155 ms** | 35,301 ms |
> | decode | **22.67 tok/s** | 96.92 tok/s |
>
> **Four times slower with three times the headroom.**

**Measured 2026-08-24**, by reading where each number came from.

```
96.92 tok/s   results/decoders-98304.jsonl  -- dflash2_arena, 6 rounds, median of 3
              every one of its six ngram-mod rows records:   acceptance 60.2

22.67 / 25.63 logs/dflash2-hwbase-98304*.log -- hardware_baseline.py, 1 generation
              both runs record:            draft acceptance 0.14870 (40 / 269)
```

**`ngram-mod` is a speculative decoder, and its tok/s tracks draft acceptance
directly.** 60.2 % against 14.87 % is a four-fold difference in how much
speculation is doing, produced by the two tools building their prompts
differently — `hardware_baseline.py` takes the first 150,000 *characters* of
`real-code-deep`, the arena builds its prompt its own way. **Neither number is
wrong. Putting them in the same table was.**

`hardware_baseline.py` was written *after* the card was swapped, so **the 4070
SUPER never ran it.** There was no same-instrument figure to compare against, and
the table filled the gap with the nearest available number instead of saying so.

**What survives, and what does not.**

- ✅ **The prefill row was fine, and now has a control.** `35,301 ms` is the cold
  turn-1 of 44,255 tokens, same corpus, same ctx, same decoder
  (`08-rtx3090-transfer.md` §6), and **prefill does not involve speculation at
  all**. Per token: 4070 SUPER **0.798 ms**, 5060 Ti JIT 3.330, 5060 Ti native
  **1.517**.
- ✅ **The wrong-architecture finding stands and the rebuild confirmed it.** Same
  script, same corpus, same flags, acceptance byte-identical at 0.14870 in both:
  prefill **146,155 → 66,582 ms, 2.20× faster**. That comparison was always
  clean because both sides came from the same instrument.
- ❌ **"Four times slower" as a hardware verdict is withdrawn.** Correctly built,
  the gap that is actually measurable is **1.90× at prefill**, and it is a
  property of the silicon — 4,608 CUDA cores against 7,168, 448 GB/s against
  504. **The 5060 Ti bought VRAM, not speed.**
- ❌ **Decode across the two cards is unmeasured**, not slow and not fast. The
  arena sweep on this card is the only thing that can answer it, row-for-row.

**Why this one is worth its own entry.** Every other correction in this register
came from re-reading the project's own data. This one came from re-reading its
own *plumbing*: two numbers, both correctly measured, both correctly recorded,
made false by being placed side by side. **The failure was in the table, not in
the instrument** — which is why nothing anywhere flagged it, and why the result
row now carries `exe` and `cuda_archs` (`bench/tests/test_exe_provenance.py`) so
at least the *binary* can never again go unrecorded.

**Guarded by** `scripts/audit-stale-claims.py`, rule `blackwell-4x-slower`.

---

## 29. "`FA_ALL_QUANTS` is not needed" — true of Q8, and Q8 is the one KV type the flag does not gate

**Where it was published**, in three places, each stating the same reason:

> `reports/05-OPERATING-GUIDE.md:153` — | `FA_ALL_QUANTS` rebuild for Q8 KV? | **not needed** -- Q8 is faster on the stock binary | 02 SS3.1 |
> `reports/06-OPEN-QUESTIONS.md:211` — | Is `FA_ALL_QUANTS` needed for Q8 KV? | **No** -- Q8 is faster on the stock b10472 binary |
> `reports/16-OPTIMIZATION-SURFACE.md:228` — | build: `FA_ALL_QUANTS` | off | **decided** | Q8 KV is faster on the stock binary, so it was not needed |

**Read out of the tree 2026-08-24**, `ggml/src/ggml-cuda/fattn.cu:340-352`:

```c
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            return true;
```

**`GGML_TYPE_Q8_0` falls through to `return true` whether the flag is on or
off.** A Q8 measurement is structurally incapable of testing this option. The
answer to the question asked -- *is it needed for Q8?* -- is right. The row that
records it says **`decided`**, and what that forecloses is a different set
entirely: `q4_1`, `q5_0`, `q5_1`, and, at `fattn.cu:442-446`, **every asymmetric
K!=V pair**, none of which was ever run.

The caveat was on the page the decision came from.
`researchs/Deep Research/deep-research-optimization2.md:138` scopes the flag to
*"Only for asymmetric/non-stock KV experiments"* -- precisely the experiments the
`decided` row then closed.

**Verified state of both binaries**, `CMakeCache.txt`:

```
llama.cpp/build-blackwell   GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF
llama.cpp/build-dflash2     GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF
```

**And the failure is half-silent.** `-fa auto` is the default
(`llama-context.cpp:3534`); an unsupported KV type resolves through
`llama-context.cpp:547`, which emits `LLAMA_LOG_WARN(... "set to disabled")` and
**continues**. A quantized *V* cache then hard-fails at
`llama-context.cpp:3607-3610` -- but `-ctk q5_1 -ctv f16` boots, runs with flash
attention silently off, and returns a number.

**What is NOT claimed:** that turning the flag on would help. No KV type it
unlocks has ever been measured here, on either card. The correction is to the
word **`decided`**, not to the flag's value.

Found because an outside operator on the same RTX 5060 Ti 16 GB opened a config
with *"IMPORTANT: Compile llama.cpp with `GGML_CUDA_FA_ALL_QUANTS=ON`"* and a
`q5_0`/`q4_1` KV pair -- a line our binaries cannot express. Captured in
[`researchs/reddit-5060ti-quant-thread/`](../researchs/reddit-5060ti-quant-thread/README.md).

**Guarded by** `scripts/audit-stale-claims.py`, rule `fa-all-quants-decided`.

---

## 30. "The boundary is prompt length, between 43k and 64k" — there is no boundary, and the claim was published in a commit message

**Where it was published**, 2026-08-24, in the body of commit `6b717f7`:

> so the window is fine and the boundary is prompt length, between 43k and 64k.

Written from **two points**: a 43,162-token prompt generating the full 512-token
budget at ctx 147,456, and a 64,210-token prompt generating 9 and stopping on
EOS.

**Refuted the same hour by five more points.** Same boot, cold prefix cache,
varying only the prompt (`results/DIAG-length-real-code-deep.jsonl`):

```
43,162 -> 512   46,909 ->   1   51,038 ->   1   54,310 -> 512
57,780 -> 512   60,831 -> 512   64,210 ->   9
```

**Failure is not monotonic in length**, so length is not the variable. `filler`
cuts the corpus at exactly `n * 3` characters, so each length ends at a different
point in the source, and **where the cut lands** is what decides it.

**Confirmed by changing the other variable.** The same seven lengths on
`real-code-vendor` — 11 files of `llama.cpp`'s `gguf-py`, a codebase nobody here
wrote — complete **7 of 7**, including **70,322 tokens**, deeper than the length
that collapsed. Same model, same ctx, same greedy sampler
(`results/DIAG-length-real-code-vendor.jsonl`).

**Two points fit infinitely many curves and the mind supplies the straight one.**
Recorded as [`traps.md` 15](../agents/traps.md), together with 14 — the first
version of that sweep left `cache_prompt` on, so requests 2 through 7 processed
3,532 to 4,389 tokens instead of their own length and the variable under test was
the cache.

**The second half of this entry is the part worth keeping.** The claim went into
a **commit message**, which is a layer this project treats as durable: a
hypothesis written there reads as a result to everyone who comes after, and
nothing in the tooling scans commit bodies. The register, the reports and the
results pages all have a place to mark a claim unverified. A commit message does
not. **Do not put an unverified boundary in one.**

**Guarded by** `scripts/audit-stale-claims.py`, rule `prompt-length-boundary`.

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
