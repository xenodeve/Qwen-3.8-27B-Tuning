# Traps — the ways of working that failed here, and what catches them now

**`CORRECTIONS.md` records numbers this project got wrong. This records the
*methods* that produced them.** A wrong number gets retracted once; a wrong
method produces new wrong numbers forever, so it is the more expensive of the
two to leave undocumented.

Every trap below **actually happened in this repo** and cost real time. Each one
names its guard, and where nothing guards it, **it says so** — because a list
that implies everything is covered is worse than no list.

Read this with `CORRECTIONS.md`. That one tells you which figures to distrust;
this one tells you which of your own instincts to distrust.

---

## 1. A hazard documented in one driver does not protect another

**What happened.** `bench/opencode_corpus.py:50-62` documents, in detail, that
OpenCode keeps a per-project server alive and that `run` attaches to whichever
is listening — carrying **the project root it was first started with**. It even
records the symptom: *"every answer landed in `C:\AI\qwen38-tuning` while the
harness looked in the task directory and recorded 'no file written' on work the
model had done correctly."* It defends itself by killing the server first.

`bench/real_task_bench.py` never did. It passed `cwd=<clone>`, which OpenCode
does not honour. Five real GitHub issues then ran 24–40 minutes each, edited
**`C:\AI` — the live repository** — and were recorded as *"the worker changed
nothing"*. That result was written up, circulated, and reasoned from for a day.

**Why the obvious defence failed.** The knowledge existed, in the right
repository, in a file a few hundred lines away. Nobody re-read it, because
nothing made them.

**The guard.** `bench/tests/test_worker_workdir.py` pins the fix on the argv.
**And the general rule:** when you find a documented hazard in one driver,
**grep for every other caller of the same tool before moving on.** A docstring
is a note to whoever opens that file, not a property of the system.

## 2. Never write regex, paths or code through a shell heredoc

**What happened — four times in one session.**

| what was written | what reached the file |
|---|---|
| `C:\Program Files\Git` + `bash.exe` | `C:\Program Files\Gitinash.exe` |
| `\b` in a regex | a **literal backspace**, `0x08` |
| `\n` in a character class | a real newline, splitting the pattern |
| `\0` in a replacement | collapsed |

Each produced a file that looked plausible and behaved wrongly. The backspace
one sat inside a working audit rule and the script still ran.

**The guard — discipline only, nothing can test this.** Use the Write tool, or
build the string in Python with `chr(92)`. **Then assert.** A `str.replace()`
that matched nothing is how one of these survived: it reported success and
changed nothing.

## 3. `cwd=` is the thing that looks right and is not

**What happened.** See trap 1. The subprocess was launched with the correct
working directory and the tool ignored it.

**The lesson generalises past OpenCode:** when a mechanism silently fails, **do
not write the test against that mechanism.** A test asserting `cwd == clone`
would have passed throughout the entire incident. `test_worker_workdir.py`
deliberately asserts on the **argv** instead, because that is what decides where
files land.

**Ask of every guard: would this have been green while the bug was live?** If
yes, it is not a guard.

## 4. A silent truncation is worse than a crash

**What happened.** `dflash2_arena.filler()` returned `text[:n_tokens * 3]`. The
frozen corpus is 91,868 characters, so **every request above ~30,600 tokens
silently returned a shorter prompt than asked for.** A run at ctx 65,536 would
have believed it measured a 65,536-token window, actually measured ~30,600,
finished cleanly and written a plausible rate.

**And every run labelled "ctx N" fed a prompt of about 40 % of N** — 6,621
tokens at "16,384", 43,162 at "98,304". The allocations were right, since
`--ctx` sets those; the depth labels were not.

**But the reason is not the one this file gave until 2026-08-23**, and the
difference matters. `dflash2_arena.py:478` is `filler(int(ctx * 0.5), regime)`:
the arena asks for **half** the window on purpose, to leave room for the
generation. `filler`'s assumption of 3 characters per token is close — measured
against the server's own token counts it is **~3.4**, about 12 % low. The claim
that it was 7.0–7.4 came from dividing by the full `ctx × 3` and dropping the
0.5, and it is retracted in [`CORRECTIONS.md` §25](../reports/CORRECTIONS.md).
**A wrong explanation bolted onto a right number is its own trap:** acting on
that one would have meant "fixing" `filler` to send 2.3× more text, doubling
every future prompt while the label stayed put.

**The guard.** `bench/tests/test_corpus_depth.py`: `filler` now raises and names
both sizes. **The general rule is in `CLAUDE.md` and worth repeating here:** an
instrument that returns a believable number instead of a failure is worse than
one that crashes.

## 5. Your own parser may be discarding the data you are about to go looking for

**What happened.** `harness.parse_spec_impl_stats` assigns `out[name] = {...}`
inside its `finditer` loop, so every match overwrites the previous and only the
final cumulative block survives. **Its own docstring states the fact it was
throwing away** — *"the server reprints them after every completion, so the LAST
block is the run and the first block is the first task."*

An entire research ticket was spent establishing that per-request speculation
attribution was possible. It had been sitting in every log the project had ever
written.

**Nothing guards this.** Before concluding that data does not exist, **read what
your own parser does with the data that is already in front of it.**

## 6. A verdict at one depth does not transfer — and a mechanism story is not an exception

**What happened.** `--spec-ngram-mod-n-match 24` measured **+34.6 % RESOLVED** at
ctx 16,384. A mechanism was then written down explaining why it should widen its
lead at depth: a fuller table means more contexts colliding on a short key. It
was labelled a hypothesis. **It was backwards.** At 65,536 the optimum moves to
`16` and `24` becomes a null, because the binding constraint at depth is fire
rate, not collision.

**The trap is not the wrong guess.** It is that a plausible mechanism can be
told in either direction, so it feels like evidence and is not.
**`CLAUDE.md` states the depth rule without exceptions precisely so that
"but here is why this case is different" cannot be written.**

**The guard.** `scripts/audit-stale-claims.py`, rule `nmatch-24-at-depth`.

## 7. The 13.6 % noise floor is a ctx 16,384 number

**What happened.** Decode is deterministic at temperature 0 — every
per-implementation counter is byte-identical across rounds — so all spread is
the clock. Measured:

| arm | within-arm spread @ 16,384 | @ 65,536 |
|---|---:|---:|
| `n-match 12` | 9.5 % | **39.5 %** |
| `n-match 8` | 10.6 % | **48.9 %** |

**The same arm, unchanged in every counter, spans up to 48.9 % between boots at
depth.** A 13.6 % floor there resolves pure drift as an effect.

**The guard.** `scripts/audit-stale-claims.py`, rule `noise-floor-at-depth`.
`paired_deltas` still defaults to 13.6 % and takes `floor_pct` explicitly —
**deliberately not patched**, because three rounds cannot re-derive a floor and
inventing a depth-scaled constant would be the same error one level up.

## 8. Resolve decisions in dependency order, or the earlier answer is written against the wrong question

**What happened.** A ticket asking whether speculation could be attributed *per
period* was closed with "yes, per request, cleanly". The ticket **defining what a
period is** was answered afterwards — as a **time slice**, which a single request
routinely spans. The first answer was true and did not answer the requirement.

**Nothing guards this.** When closing a decision, **check what it depends on that
is still open**, and write the answer so it survives either resolution — or wait.

## 9. Verify what a subagent or an external reviewer returns, including the parts that are right

**What happened, twice.**

- A research agent reported that recurrent state is *"the larger of the two
  allocations"* so `-ctk` is ignored where it matters. True at ctx 16,384 (748
  vs ~288 MiB); **false at 65,536** (748 vs 1,152). RS is context-independent;
  KV is not. The correction was more useful than the claim.
- An external reviewer proposed a fix using `--approval-mode=yolo`. **Our runner
  is OpenCode, not Qwen Code**, and its CLI has no permission flag at all. The
  *underlying* hypothesis was excellent and led directly to the trap-1 discovery
  — but the specific fix would have been wasted work.

**The guard is discipline, and `CLAUDE.md` already states it.** A report is a
hypothesis until checked. **Check the load-bearing claims yourself, cheaply,
before building on them** — both of these took one command.

## 10. A stop hook, a linter, or any automated critic can be confidently wrong

**What happened.** A stop hook asserted five times that the measurement harness
and `results/*.jsonl` were "benchmark output owed deletion". Complying would have
destroyed **80 result files cited by 20 documents** and demoted every published
number in the repo to a guess.

But it was **right that a target existed** — checking the claim properly rather
than restating the rebuttal a second time is what turned up 157 committed files
of model-generated code in `bench/_deepwork/`.

**The rule.** Do not comply with an automated critic on an irreversible action,
and do not dismiss it either. **Go and look.** Then, if the reading is genuinely
ambiguous and the action cannot be undone, ask the developer — that is what
happened here, and the answer settled it.

## 11. Fix the class, not the instance

**What happened.** `bench/_work/` was gitignored. `bench/_deepwork/` — the same
kind of artifact, written by a sibling driver — was not, so **157 generated files
sat committed**. One line of `.gitignore` between them.

**The fix was the glob**, `bench/_*/`, not a second entry. A list has to be
remembered once per new driver, and that commit was what forgetting looks like.

**The guard.** `bench/tests/test_no_committed_worker_output.py` asserts the
invariant against `git ls-files` and **does not name any directory**, so a future
`_widework/` is caught without anyone updating a list.

## 12. Two explanations can both be true, and fixing one does not fix the other

**What happened.** Five real tasks produced nothing. Two independent causes were
found: the worker wrote to the wrong tree, **and** decode at the served ctx
98,304 measured **2.8–5.0 tok/s** against 75.2 at 16,384, with 13 of 16
measurements timing out.

Either one alone fully explains a zero, so both were written down and the rule
was: fix one at a time.

**Then the second one dissolved — and that is the sharper lesson.** Measured
2026-08-23, all sixteen of those rows had loaded the DFlash2 sidecar, so *depth*
and *drafter* never varied independently. Re-run with the arms alternated, the
profile actually served (`ngram-mod` alone) returns **96.92 tok/s at ctx
98,304**, faster than at 16,384, 6 rounds out of 6.
[`CORRECTIONS.md` §26](../reports/CORRECTIONS.md).

**So the trap has two halves, and the second is the one that bites.** Holding
two live explanations is right. But **an explanation drawn from a sweep in which
the suspected variable never changed is not an explanation** — it is the sweep's
own constant, wearing the label of a finding. The tell was there to read: the
sweep's title said `ngram-nmatch`, and every row's `args` field said
`draft-dflash,ngram-mod`. Nobody looked at `args` for four days.

**Ask of any second cause: what did this experiment hold fixed?** If the answer
includes the thing you are blaming, you have not measured it.

**Nothing guards this.** When a cause is found for a failure that cost hours,
**keep looking until the arithmetic actually adds up.** The next run must not
change both variables at once, or it will be unreadable.

---

## The shape common to almost all of them

## 13. The guard you need is often in the file you already imported

**What happened, 2026-08-23.** A one-off script was written to measure decode at
44K tokens, importing `dflash2_arena` on its first line for the corpus loader.
It returned 71.76 tok/s. The arena, measuring the same thing properly an hour
later, returned 96.4-98.9 over six boots.

The 26 % gap has a cause and the cause was written down. `dflash2_arena.py:483`
carries this comment, four lines above the code the script did not use:

> FULL LENGTH, not a token or two. A 16-token warm turn paid the prefill but
> left the n-gram table nearly empty, and the first TIMED generation of every
> ngram arm then came in 35-40 % low

The script had warmed the prefill and not the n-gram table, which is that
paragraph exactly. It also skipped `harness.generation_is_measurable`, so its
first attempt averaged 59- and 215-token generations against a 512-token budget
and reported a median anyway.

**This is trap 1 repeating inside one session.** There the hazard was documented
two days earlier in a file a few hundred lines away. Here it was documented in
the file the script imported, and the number it produced was plausible enough to
be reported to the developer before the arena contradicted it.

**The rule.** Before writing a new measurement script, read the docstrings and
comments of the module you are importing from -- they are where this project
stores its incident history, and a bespoke script starts with none of it.
**Prefer extending the existing harness over writing a parallel one**; if the
parallel one is genuinely warranted, list which of the host module's guards you
are choosing not to inherit, and say why in the file.

Eleven of these thirteen produced **a plausible number or a clean exit**, not
an error. That is the signature to watch for:

- `split: 65+0` while the card thrashes at 32 MiB free
- `diff_bytes: 0` while the worker edits another repository
- a full-length prompt reported for a window that was 40 % filled
- `rc=0` on a task that did nothing
- an audit rule containing a backspace, running fine
- a bespoke script reporting 71.76 tok/s where the harness reports 96.9

**When something reports success, ask what it would have reported had it
failed.** If the answer is "the same thing", you have not measured anything yet.
