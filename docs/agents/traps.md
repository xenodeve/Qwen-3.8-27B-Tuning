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

## 14. A probe that reuses the prompt cache is not a controlled experiment

**What happened, 2026-08-24.** Eighteen arena rows at ctx 147,456 were voided
because every generation produced 9 tokens against a 512-token budget. To find
where that starts, a one-off script booted once and swept the prompt from 49,152
to 73,728 requested tokens, ascending, **with `cache_prompt: True`** -- left on
because the arena sets it and it was copied without thought.

It came back **512 / 1 / 1 / 484 / 512 / 1 / 1**: not monotonic in length, and
briefly readable as a finding about depth.

**The tell was in a column already being printed.** `prompt_n` read 43,162 on
the first request and then **3,532 to 4,389** on every one after. Each later
prompt shared its prefix with the cached one, so the server processed only the
delta -- the requests were not the lengths they were labelled with. **The
variable under test was the cache.**

Re-run with `cache_prompt: False`, `prompt_n` tracks the requested length and
the numbers change.

**The rule.** A sweep whose inputs share a prefix must turn prefix reuse **off**,
or reboot between points. And when a probe copies settings from the harness,
list which ones are *measurement policy* rather than defaults -- `cache_prompt`
is there so the arena's timed generations skip a cold prefill they already paid
for, which is correct for the arena and wrong for anything varying the prompt.

---

## 15. Two points look like a line

**What happened, minutes later.** With the cache off, two clean points existed:
a 43,162-token prompt generated the full 512, a 64,210-token prompt generated 9.
The conclusion written down -- **in a commit message** -- was *"the boundary is
prompt length, between 43k and 64k"*.

**Seven points refuted it.**

```
43,162 -> 512   46,909 -> 1   51,038 -> 1   54,310 -> 512
57,780 -> 512   60,831 -> 512   64,210 -> 9
```

Failure is not monotonic in length, so length is not the variable. `filler` cuts
the corpus at exactly `n * 3` characters, so **each length ends at a different
point in the source** -- and what decides the outcome is where the cut lands.

**Confirmed by changing the text instead of the length.** The same seven lengths
on `real-code-vendor`, 11 files of `llama.cpp`'s `gguf-py`, complete **7 of 7**,
including 70,322 tokens -- deeper than the 64,210 that collapsed. Same model,
same ctx, same greedy sampler, same day.

**The rule.** Two points fit infinitely many curves, and the one the mind
supplies is a straight line through them. **A monotonic hypothesis needs a
monotonic test**: sample the interval before naming a threshold, and prefer
changing the *other* variable -- here the corpus -- over adding more points along
the one you already suspect.

**And do not put an unverified boundary in a commit message.** Commit messages
are the layer this project treats as durable; a hypothesis written there reads
as a result to everyone who comes after.

---

Fifteen of these nineteen produced **a plausible number or a clean exit**, not
an error. That is the signature to watch for:

- `split: 65+0` while the card thrashes at 32 MiB free
- `diff_bytes: 0` while the worker edits another repository
- a full-length prompt reported for a window that was 40 % filled
- `rc=0` on a task that did nothing
- an audit rule containing a backspace, running fine
- a bespoke script reporting 71.76 tok/s where the harness reports 96.9

**When something reports success, ask what it would have reported had it
failed.** If the answer is "the same thing", you have not measured anything yet.
---

## 16. The assertion that measures the shape of the file, not the behaviour

**What happened.** Over three sessions this suite grew **eight** assertions that
could be broken by re-wrapping a line, renaming a local, or adding a comment —
and **two of them passed for the wrong reason**, which is the half that matters.

The red ones, each caused by an improvement:

- required the literal `"-BindAddress"` *with the dash*, then went **blind**
  when the call moved to a hashtable where the key has none
- required `"0.0.0.0"` after a line offset; red when a function that *reads* the
  socket was added above it
- matched the first `Ctrl+C` in the file, which was a comment whose sentence
  wrapped
- forbade `--log-colors` anywhere; red on a comment explaining its absence
- required the served GPU's UUID as a literal; red when that literal was
  **de-duplicated** into `Get-GpuVram.ps1`
- banned `| ForEach-Object` anywhere; red on an error handler listing installed
  GPUs, a pipe with nothing to do with llama.cpp's output
- required `if ($Dual)` before the first occurrence of the word *"artifact"*;
  red because the word appears in a **parameter comment two hundred lines
  above** the banner
- matched `-ub 1024` as a literal in the argv; red when `-ub` became a
  parameter the budget check also needed

**The two that were green and should not have been:**

`test_it_guards_the_port_before_launching` omitted `Invoke-RestMethod` — the
thing the guard actually calls — and matched `Get-NetTCPConnection` in an
unrelated status block. **Green for days.** It surfaced only when that block was
moved to its own file for an unrelated reason.

`test_the_dual_profile_uses_the_split_that_won` asserted `"-sm" in t and
"tensor" in t` and was **green before the flag existed**, because the profile's
header explains at length why `-sm row` cannot load and that `-sm tensor` was
swept. Both tokens were in prose.

**The rule.** Name the property, then ask *what could break this assertion
without breaking the property*. If the answer is "wrapping a line", "renaming a
variable" or "adding a comment", the assertion is measuring the file.

**What actually works**, in order of preference:

1. **Run the thing.** `serve.ps1 -WhatIf` resolves everything and exits without
   touching the GPU, so the banner is observable behaviour. Every banner check
   here now does that, and it is how the launcher was caught printing the wrong
   artifact *and* a stale rate.
2. **Assert on the value, not the spelling.** Read `-ub`'s argument; if it is a
   variable, resolve its default. The profile still serves 1024 either way.
3. **Scope to the invocation.** `t[t.index("& $Exe -m $Model"):]` cannot be
   fooled by prose, which is what separates a flag that is *passed* from a flag
   that is *discussed*.
4. **One chokepoint, then forbid the pattern everywhere else.** `--query-gpu`
   appears only in real calls, never in prose — so "no module but `gpu_device`
   may contain it" is a property that cannot go blind or cry wolf.

**Guarded by** nothing automatic. The suite cannot tell a good assertion from a
bad one; only the question above can.

---

## 17. The launcher that describes configuration it does not own

**What happened, three times.** `serve.ps1` selects a profile and then prints a
description of it. Every time the profile changed, the description did not:

- it printed **"closing this window stops the server"** when it did not —
  killing the launcher left `llama-server` alive and answering (commit `b55699c`)
- `-Dual -WhatIf` selected `worker-q4-dual.ps1` and printed **"artifact
  UD-Q2_K_XL"** underneath the line that had just named the other file
- it advertised **"20.9 tok/s"**, then **"32.4 / 32.6 / 33.1"** — the first from
  before `-sm tensor`, the second from the *even* split that collapses to 0.38
  under desktop load
- with `-Mtp` it printed **two contradictory decoder lines four rows apart**: a
  static one saying `draft-mtp` is NOT set, and the profile's own saying it is

**Each was found by running it. None by reading it.**

**The rule.** The component that knows what it was asked for is the one that
should say so. The decoder line moved into the profile; the GPU line reads the
driver rather than naming a card from memory. A launcher may **select** and
**report what it read back** — it may not **describe**.

**Guarded by** `bench/tests/test_the_dual_profile_serves_both_cards.py`, which
runs `serve.ps1 -WhatIf` on both paths and reads the output a person sees.

---

## 18. A guard that models load time cannot promise a run

**What happened.** The two-card profile computes its split from free VRAM and
refuses when the budget cannot hold the model. Asked for `-Ctx 262144` with
`-ub 512`, it **approved**. The server loaded, reported `66+0`, answered
`/health` — and died the moment a real request arrived:

```
CUDA error: out of memory
  current device: 1, in function alloc at ggml-cuda.cu:648
  cuMemSetAccess(start_ptr, reserve_size, &access, 1)
```

llama.cpp allocates more once there is work to do. **A successful boot is not a
successful run**, and a guard built on load-time arithmetic will keep saying yes
to configurations that cannot serve.

**Worse, the same guard had already been wrong once in the other direction**: its
first version compared the budget against the *weights* alone, ignoring KV and
compute, which meant it approved **every** context.

**What settled it was re-testing every depth with a real request** — a 135,233
token prompt — and counting only the depths that *answered*:

```
ctx 147,456 ub 1024  SURVIVED   free after 2,100/2,097 -> 1,998/2,040
ctx 196,608 ub 1024  SURVIVED          1,436/1,258 -> 1,248/1,208
ctx 229,376 ub 1024  SURVIVED          1,156/  550 -> 1,071/  500
ctx 262,144 ub 1024  refused at load
ctx 229,376 ub  512  SURVIVED          1,312/1,010 -> 1,249/  974
ctx 262,144 ub  512  SURVIVED            919/  488 ->   821/  452
```

**The run that died had 336 MiB free on the second card; the one that survived
had 488.** The line sits between them — close enough that what the desktop is
doing decides which side you land on.

**The rule.** When a resource check gates a long-running process, the acceptance
test is the process doing its work, not the process starting. And say what the
guard cannot promise: this one refuses the impossible and **does not promise
comfort**, which is now written in the profile itself.

**Guarded by** `bench/tests/test_the_dual_profile_serves_both_cards.py`
(`test_the_profile_records_that_loading_is_not_surviving`) and by the profile's
own header carrying the ladder.

---

## 19. Retrying a failure that cannot change

**What happened.** An arm that could not load was booted again in every round.
`layer-dflash-ngram` failed in about a second with `dflash requires ctx_other to
be set`, and the sweep tried it twice more — each attempt costing a boot plus a
full VRAM-release wait, for an outcome fixed by the argv.

The developer named it: *"ทำไมเราต้องรอให้ run เสร็จด้วยในเมื่อ decoder ใช้ไม่ได้"*.

**The rule, and its limit.** A failure that is a **capability** does not change
between identical rounds, so try it once. A failure that is a **resource** can —
and this project has one of each: `draft-dflash` under `-sm tensor` fails at a
graph-split assertion at any memory pressure, while `draft-mtp` at 147,456
failed on the even split and loads on the computed one.

So the "dead" set is kept **per depth and per regime**, not globally. An arm that
cannot load at 262,144 may load at 16,384, and inheriting the verdict across
depths would skip a measurement that was available — which is exactly what
happened: `draft-dflash` runs fine on the layer split at 16,384 and is the
**fastest configuration measured anywhere in this work**.

**And the skipped rounds still get a row.** Omitting them makes an impossible
arm look *unpaired* — `report()` prints "NOT PAIRED (1 vs 3 rounds)" and the
reader concludes the sweep was interrupted. The row says it was not retried, and
carries the reason llama.cpp gave.

**Guarded by** `bench/tests/test_a_row_names_the_cards_that_made_it.py`
(`test_a_dead_arm_is_recorded_once_per_round_but_only_tried_once`).
