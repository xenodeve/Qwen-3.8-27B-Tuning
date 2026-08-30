# Plan 06 — benchmark the worker on real open issues, and measure the context a real task actually needs

**Status: intent, not results.** Nothing in this document has been run. Written
2026-08-22. Everything below that looks like a fact about a repo was read from
that repo on that date and is cited; everything that looks like a number about
the worker is a prediction and says so.

This is a **runbook**. An agent picking it up should be able to execute it
without asking a question. If you find a question it does not answer, that is a
defect in this file — fix the file, then continue.

---

## 0.0 STATUS, 2026-08-22 evening — what this plan got wrong

**Phase 0 has not run. Phase 1 has, partially, and it refuted this plan's
headline hypothesis.** Read this before the rest of the document; several
sections below are now historical.

### The context hypothesis was backwards

This plan says, in §0 and again in §6:

> "If real tasks peak at 40,000 tokens and we serve 98,304, roughly 1.5–2 GB is
> being reserved for nothing."

**Measured, three windows, each time saturating:**

| window served | high-water range | saturated? |
|---:|---|---|
| 32,768 | 32,767 – 41,377 | **yes, all five** |
| 65,536 | 54,324 – 72,056 | **four of five** |
| 98,304 | 56,861 – 88,668 | no |

The 40,000 figure came from the run that hit the wall at 32,768. **It measured
the ceiling, not the task.** `worker-iq2s-quality.ps1` at 98,304 is the
**minimum sensible window**, and §3.5's premise — that an over-provisioned
window is currency for a higher quantisation rung — has no currency to spend.

**And the drafter still fits there:** at 98,304 with `--spec-draft-n-max 4`,
`65+0` resident, 254 MiB free. I predicted it would not.

### A fourth outcome was needed

The rubric in §4 has three outcomes. It needs a fourth, and the reason is a run
that reported `0 PASS, 5 FAIL, 0 VOID` with every baseline green — which reads
as a verdict on the worker and was not one. The tasks had filled the window.

**`WINDOW_BOUND`** is now its own outcome (`harness.classify_outcome`,
`tests/test_window_bound.py`): not a worker failure, never totalled as one, but
still a *result* — it says this class of task does not fit that window.
`--n-ctx` is a required argument to `real_task_bench.py` for the same reason: a
harness that does not know the window cannot tell the two apart.

### The RTX 3090 pool has a scoreboard now

[`../results/08-rtx3090-transfer.md`](../results/08-rtx3090-transfer.md) records,
per technique, whether it was measured here, read and closed, already had, or
architecturally impossible. **Read it before adding a phase for anything from
that pool** — five entries are already closed from source, and two of those
closures each save a GPU round that would have measured nothing.

### Phases 4 and 6 are cheaper than written; three sweeps are now dead

Six flags were read from source before spending GPU time on them —
[`../researchs/llamacpp-flag-semantics-2026-08-22.md`](../researchs/llamacpp-flag-semantics-2026-08-22.md).
**Three are provably inert in our configuration** and should not be swept:
`-ctkd`/`-ctvd`, `GGML_CUDA_GRAPH_OPT`, `-bs`. A fourth caveat matters for
anyone reading §6: **`--spec-draft-p-min` ≤ 0.0625 is mathematically identical
to 0.00**, so a value ladder starting at 0.05 cannot differ from the baseline.

### One flag this plan relies on does not mean what it says

`--fit-target` is described throughout `scripts/` as the margin the server
leaves free. `tools/server/server-context.cpp:1074` **adds the draft model's
bytes to it** before `--fit` runs, so with the 1,090 MiB sidecar our
`--fit-target 768` reaches `fit.cpp` as roughly **1,900–2,100 MiB**. Any VRAM
arithmetic in this document that uses 768 as a margin is wrong.

### The finding this plan exists to explain, and cannot yet

At 98,304 — with room to spare, a green baseline and a green verify — **four of
five real issues ran 1,427–2,400 s and changed no files.** That is a genuine
worker result and it has **no mechanism attached**. The OpenCode transcript is
written beside the clone and deleted with the scratch root; §7 must capture it
before the next run.

---

## 0. The four questions

**Q1. Which serving configuration produces the most verified accepted coding
tasks per hour?**

This project's stated metric is exactly that, and **not one published number
measures it.** Every result so far is tok/s on a synthetic prompt. The most
recent one — `draft-dflash,ngram-mod` at +23.2 % over `ngram-mod` — was measured
on a prompt with **66.2 % duplicate lines**, which is `ngram-mod`'s best case and
nothing like real code (this repo's own source measures 0.6–4.8 %, by
`harness.line_repetition_pct`).

**Q2. What context window does a real task actually consume?**

Nobody has measured this either, and the cost of not knowing is already on the
record: **three worker profiles shipped at the wrong size in one day**
(`CORRECTIONS.md` §15 and §17), each sized from a number that did not mean what
it appeared to mean.

Q2 is worth more than Q1. Context window is paid for in VRAM whether or not it
is used, and on a 12 GB card every GB returned can be spent on
bits-per-weight, on a drafter, or on parallel slots. If real tasks peak at
40,000 tokens and we serve 98,304, roughly 1.5–2 GB is being reserved for
nothing.

**Q3. Does the xeno-skills workflow earn what it costs?**

Its cost is measured and large: report 26 found the 352-skill catalogue is
injected as a *user* message block worth **38,064 tokens, 70 % of the prompt,
paid three times per invocation** -- 153,621 tokens against 14,064 under
`--safe-mode`. **Nobody has ever measured its benefit.** A cost with no
measured benefit is not a workflow, it is a habit.

This is a clean A/B: same task, same model, skills on versus `--safe-mode`.
Whatever the answer, it is worth knowing -- if skills raise the PASS rate they
justify the tokens, and if they do not, 38,064 tokens per invocation is the
cheapest speed-up available anywhere in this project.

**Q4. Is `UD-IQ2_XXS` good enough for T4 Labs' real work?**

The question this whole repo exists for, and it has never been asked with
evidence. Nine artifacts have depth-throughput numbers; **not one has a
task-success number.**

**Q4 cannot be answered by a single PASS rate.** "8 of 19" means nothing on its
own: it is indistinguishable between *the quantization is too coarse* and *this
task set is hard for any model*. It needs a **ceiling**, and §6.0 is that.

**Order: Q4's ceiling, then Q2, then Q1, then Q3.** Do Q2 before Q1 -- a decoder
comparison at a window nobody uses compares two configurations of a fiction --
and do the ceiling before either, because it also tells you whether the task set
discriminates at all.

---

## 1. Safety — read this before running anything

The benchmark takes its tasks from six real projects. **It must never change
the state of their working directories.**

Measured 2026-08-22:

```text
MangaDock      branch perf/mit-layout-fit-and-merge   333 uncommitted files, 1 stash
T4 Fastwork    branch master                          440 uncommitted files, 4 stashes
xeno-skills    branch main                            clean
pal-mcp-server branch main                            clean
TipSpace       branch main                            clean
Clone Space    branch main                            clean
```

Two of them hold days of unfinished work that exists **nowhere else**. An agent
that runs `git checkout`, `git stash`, `git reset`, `git clean` or `git
restore` in `D:\Github\MangaDock` destroys it. And because this plan ends with
"delete the benchmark work", a benchmark that ran in-place would turn its own
cleanup step into the deletion of real work.

### The rules

1. **Never write to, or change git state in, `D:\Github\*`.** Reading is fine
   and this plan does it (that is where the verify commands in §4 came from).
   The prohibition is on mutation, not on inspection — an over-broad rule that
   also banned reading would just be ignored.
2. **Every task runs in a fresh clone from the GitHub remote**, never a copy of
   the local tree. The remote is also the state the issue was written against;
   a local tree with 333 dirty files is not.
3. **One scratch root per session**, recorded in the run manifest before any
   clone. Nothing outside it is ever deleted.
4. **No push, no PR, no issue comment, no label change** from a benchmark run.
   The tracker is an input. It is not a destination.
5. **NEVER close an issue a benchmark worked on.** Developer instruction,
   2026-08-22. The output of a benchmark run is a *measurement of the worker*,
   not a contribution to the project — nobody reviewed it, nobody is going to
   merge it, and the diff is deleted minutes later. Closing the issue would
   claim work that was thrown away, and the next person would find a closed
   issue with no commit behind it. **A benchmark PASS means the worker did the
   task, not that the task is done.**
6. **Never work in `D:\Github\*` — clone separately, every time.** Also a
   developer instruction, and it is the same rule as rule 2 stated from the
   other side: the reason is not tidiness, it is that MangaDock has 333
   uncommitted files and T4-Fastwork 440.
5. **A run that cannot clean up says so and stops.** It does not widen the
   deletion pattern to make cleanup succeed.
6. **`--dangerously-skip-permissions` is not used.** If a run needs it, the run
   is wrong.

### Scratch root

```text
D:\bench-scratch\<UTC-date>-<run-id>\
    manifest.json          run id, scratch root, task list, start time
    clones\<repo>\         one clone per repo, reused across that repo's tasks
    logs\                  llama-server logs, one per task
    results\               one JSONL row per task
```

`D:\bench-scratch` is chosen because C: had **68.5 GB free** on 2026-08-22 and
MangaDock alone is a 1,237 MB `.git`. Confirm free space before starting.

---

## 2. The pool, and why selection is stratified

211 issues are open across the six projects. **76 carry `ready-for-agent` and
are not `blocked`** — under the T4 convention (`t4-dev-workflow`, issue
lifecycle) those are the only issues an agent may work.

| repo | remote | open | ready-for-agent, unblocked |
|---|---|---:|---:|
| MangaDock | `Slow-Inc/MangaDock` | 82 | 69 |
| pal-mcp-server | `xenodeve/openclink` | 45 | 3 |
| xeno-skills | `xenodeve/xeno-skills` | 60 | 2 |
| T4 Fastwork | `Slow-Inc/T4-Fastwork` | 20 | 2 |
| Clone Space | `xenodeve/clone-space-mcp` | 4 | 0 |
| TipSpace | `Slow-Inc/TipUp` | 0 | 0 |

**91 % of the pool is one project.** Drawing at random would benchmark
MangaDock's frontend, not the worker. The selection in §3 is stratified by
repo, language, task shape and size on purpose.

Re-derive the pool with:

```powershell
$gh = "C:\Program Files\GitHub CLI\gh.exe"
& $gh issue list --repo Slow-Inc/MangaDock --state open --label ready-for-agent `
    --limit 200 --json number,title,labels
```

---

## 3. Task selection — 19 tasks

Chosen for **coverage and verifiability**. A task counts only if the repo can
prove it is done, so every row names its check. Sizes deliberately span a
hardcoded `0` to a concurrency race, so the benchmark has both a floor and a
ceiling.

### Tier A — cheap environment, fast verification (5 tasks)

| repo | issue | shape |
|---|---|---|
| xeno-skills | #306 | `feat(hooks)`: handoff path convention, and what makes a handoff VALID rather than merely present |
| xeno-skills | #314 | `feat(layer)`: T4-Compact supervisor skeleton — spawn, read the stream, watch the calls |
| openclink | #144 | `fix(clink)`: own the whole process tree, so cancellation means nothing is still writing |
| openclink | #145 | `fix(clink)`: drain output concurrently and boundedly; deadline on the post-kill drain |
| openclink | #149 | `feat(clink)`: durable run journal on the existing store |

~21 MB of clone for all five. Three unrelated problem domains (shell/hooks,
process lifetime, storage). **Start here** — it exercises the whole harness at
the lowest cost, and a harness bug found on a 3 MB repo is cheap.

### Tier B — bulk-mechanical, large diff, trivial judgment (1 task)

| repo | issue | shape |
|---|---|---|
| T4-Fastwork | #280 | `chore(format)`: make eslint and prettier agree, then clear the debt in one pass |

Included specifically because it is the shape `clink-subagents` claims to be
best at. Without it, Phase 3 has nothing to show.

### Tier C — MangaDock, small and self-contained (6 tasks)

| issue | shape |
|---|---|
| #668 | `[LOW]` Following-tab count badge hardcoded to 0 — the smallest real task in the pool |
| #656 | `[MED]` Review rating not integer/number-validated |
| #658 | `[MED]` Missing/non-numeric page query returns 500 instead of 400 |
| #657 | `[MED]` limit/offset NaN/negative unsanitized on list endpoints |
| #671 | `[LOW]` Reader comment fetch has no cancellation (stale race) |
| #663 | `[MED]` Follow button shows stale/wrong state across profiles |

### Tier D — MangaDock, backend and concurrency (4 tasks)

The ones a 2-bit model is most likely to fail. Present so the result has a
ceiling and not only a floor.

| issue | shape |
|---|---|
| #650 | `[HIGH]` Daily check-in double-claim race (`checkin.service.ts`) |
| #664 | `[MED]` Check-in insert-before-credit non-atomic (lost day) |
| #660 | `[MED]` PostgREST `.or()` filter injection via search input |
| #661 | `[MED]` Year-range filter breaks pagination and total count |

### Tier E — MangaDock, Python (3 tasks)

Different language and toolchain. All three are explicitly leftovers, so their
original context is already written down.

| issue | shape |
|---|---|
| #614 | `refactor(MIT)`: extract `load_dotenv()` out of import side-effect |
| #615 | `refactor(MIT)`: `BaseGPTTranslator` base-abstraction for retry/config |
| #623 | `bug(MIT)`: `custom_openai` crashes with TypeError on a None API response |

### Deliberately excluded, with reasons

- **Every `PRD:` / `epic`** (#155, #171, #178, #275, #434, #535, #685, #304 …).
  Unbounded by design; a worker that "fails" one has told us nothing.
- **#276, #626** — multi-day integration work.
- **Anything labelled `blocked`.**
- **`security(Backend)` #549** — RLS backstop, webhook integrity. A benchmark
  must not be the reason a security fix gets written by a 2-bit model and
  reviewed by nobody.

---

## 3.5 The decision this benchmark has to serve

The FP8 ceiling answers *how much is lost to quantization*. It does not answer
the question the developer actually has to decide: **within a fixed VRAM
budget, which rung do we serve?**

Three things compete for the same ~9.5 GB, and buying more of one means less of
the other two:

```text
  bits-per-weight   x   context window   x   drafter
```

That is why Q2 is not merely an economy. **A context window that turns out to
be over-provisioned is the currency that buys a higher rung.** If real tasks
peak at 40,000 tokens and we serve 98,304, the returned VRAM is not "savings" --

> 🔴 **Refuted — see §0.0.** Measured across three windows, real tasks
> saturated 32,768 (all five) and 65,536 (four of five) and reached
> 56,861–88,668 at 98,304. **98,304 is the minimum sensible window, not
> headroom**, so this section has no currency to spend.

it is the difference between IQ2_XXS and IQ2_S, or between no drafter and
DFlash2.

### The ladder, measured on disk 2026-08-22

| artifact | on disk | fits in ~9.5 GB free? |
|---|---:|---|
| `UD-Q4_K_XL` | 16.69 GB | **no** |
| `UD-Q3_K_XL` | 12.52 GB | **no** |
| `UD-Q2_K_XL` | 9.15 GB | weights only, essentially no room for KV or a drafter |
| `UD-IQ2_XXS` (pre-V3) | 8.39 GB | yes, thin |
| `AD-IQ2_XXS` (AtomicChat) | 8.36 GB | yes, thin |
| **`UD-IQ2_S`** | **7.80 GB** | yes — and it is **well measured already**: 38+ rows across six result files, 32,768 to 98,304, and the profile `worker-iq2s-quality.ps1` serves it. It was given up **on purpose** to free VRAM for a drafter |
| `AD-IQ1_M` | 7.91 GB | yes |
| `UD-IQ2_XXS` (Dynamic V3) | 6.77 GB | yes, with room to spare — the current default |
| `UD-IQ1_M` | 6.27 GB | yes |
| `UD-IQ1_S` | 5.77 GB | yes |

Drafters, same budget: `DFlash2-Q4_K_M` 1.06 GB on disk, `mtp-Qwen3.8-27B-Q4_0`
1.28 GB.

### File size is not resident cost — measure it

**Measured 2026-08-22:** the DFlash2 drafter is **1.06 GB on disk and 1,936 MiB
resident** (free VRAM 2,376 MiB without it, 440 MiB with it, same boot, same
window). A factor of 1.79. Every row above is an *input* to the arithmetic, not
the answer.

Worse, `--fit` **cannot measure the drafter at all** -- it logs `[spec] failed
to measure draft model memory` and then chooses layers without accounting for
it (issue #17). So a configuration that "fits" according to `--fit` may not.

**Therefore: each rung's resident cost is measured, not computed.** Boot it,
read free VRAM, record it. One boot per rung, before any task runs.

### The frontier, and what Phase 5 does

Once Q2 gives a required window, each rung has one honest question: *at that
window, does it fit, and what does it score?*

**Phase 5 — the budget frontier.** For each rung that fits the Q2 window:

1. Boot it at that window and record resident VRAM and the layer split
   (`expect_layers=65`; the default read returns the drafter's `6+0`).
2. If a drafter also fits, record that as a separate point.
3. Run the Phase 2 task subset.
4. Plot PASS rate against rung.

The output is not a winner. It is a **frontier**: the highest rung that still
fits the window real work needs, with its task score beside it. The developer
picks the point; the benchmark supplies the axes.

**The first rung to run is `UD-IQ2_S`, and the reason is not that it is
untested.** It is well measured on throughput — 26.61 tok/s at 98,304 with 400
MiB free, 49.84 at 32,768 with 2,267 MiB free — and `worker-iq2s-quality.ps1`
exists to serve it.

It runs first because of **the trade that was made deliberately**: IQ2_S was
given up for IQ2_XXS specifically to free VRAM for a drafter, and the drafter
only became loadable on 2026-08-22. So the question is not "does IQ2_S work" —
it does — but:

> Is `IQ2_XXS` + DFlash2 worth more than `IQ2_S` alone?

Both sides now exist, both fit, and **neither side has a task-success number**.
Choosing between artifacts on quality is a decision with no evidence at all —
report 27 says so in those words, and that is still true.

An earlier version of this section said IQ2_S "has never been loaded once",
copied from a ledger row without checking. It was wrong; see
[CORRECTIONS §19](../reports/CORRECTIONS.md).

### Do not assume the ladder is monotonic

More bits is not automatically better here, and this project has the receipt:
`draft-mtp` is +81 % at 16K and -71 % at 131,072 on the *same* artifact. A rung
that scores worse than the one below it is a finding, not a measurement error —
record it and check it, do not smooth it.

---

## 4. Verification — how a task is scored

Read from each repo's own configuration on 2026-08-22. **Where a repo has a T4
`verify` command, that command is the score.** Do not invent one.

| repo | verify command | source |
|---|---|---|
| xeno-skills | `bash tests/hooks/run-all.sh` | `.claude/t4.json` |
| openclink | `.venv/Scripts/python.exe -m pytest tests/ -q -m "not integration"` | `.claude/t4.json` |
| T4-Fastwork | `bun lint` and `bun test` | `package.json` scripts; **no `t4.json`** |
| MangaDock Backend | `npm test` (Jest), `npm run lint` | `Backend/package.json`, `CLAUDE.md` |
| MangaDock Frontend | `bun lint`, `bun test` | `Frontend/package.json`, `CLAUDE.md` |
| MangaDock MIT | `pytest` (has `pytest.ini`, `Makefile`) | `MIT/` |

### The rubric

Three outcomes only. There is no partial credit, because the metric is
*accepted* tasks.

| outcome | definition |
|---|---|
| **PASS** | the repo's verify command exits 0 **and** the diff addresses the issue's stated defect. **This does not close the issue** — see safety rule 5 |
| **FAIL** | verify is non-zero, or the diff does not address the defect, or no diff |
| **WINDOW_BOUND** | the task filled the context window before it could finish. **Not a worker failure and never totalled as one** — but unlike VOID it IS a result: it says this class of task does not fit that window. `harness.classify_outcome`, saturation at 98 % of `n_ctx` |
| **VOID** | the harness broke — server died, clone failed, env missing. **Not a worker failure and never counted as one** |

**A green verify with an off-target diff is a FAIL.** That distinction is the
entire reason this benchmark exists: a model that makes tests pass by editing
tests has not done the task. Record the diff so this is auditable, and record
**who judged it and how** — a human, or a stronger model, or a rule.

**Judging must not be done by the model under test.** If a stronger model
judges, name it in the row.

---

## 5. What every run records

One JSONL row per task attempt, in `results\`.

| field | source | why this exact source |
|---|---|---|
| `repo`, `issue`, `tier` | selection | |
| `config` | run parameters | which decoder/window arm |
| `model` | `local-iq2xxs` \| `gateway-fp8` | Q4. The ceiling comparison is meaningless if a row cannot say which side it is |
| `skills` | `on` \| `safe-mode` | Q3 |
| `mode` | `standard` \| `clink` | Phase 3 |
| `outcome` | rubric §4 | PASS / FAIL / VOID |
| `verify_cmd`, `verify_exit` | §4 table | the score, not a summary of it |
| `wall_clock_s` | harness | tasks per hour |
| **`ctx_high_water`** | **`n_tokens` on the `slot release` line** | **Q2.** NOT `prompt eval` — that reports only what survived cache reuse, and misreading it produced `CORRECTIONS.md` §15 and §17 |
| `turns` | harness | where a session's context actually goes |
| `tok_in`, `tok_out` | `/completion` timings | cost per task |
| `acceptance` | `harness.draft_acceptance(timings)` | weighted by drafts; `None` ≠ `0 %` |
| `split` | `harness.parse_layer_split(log, expect_layers=65)` | residency. **`expect_layers` is required** — without it a two-model log returns the drafter's `6+0` (issue #17) |
| `free_before`, `free_after` | `nvidia-smi` | boot drift, drafter footprint |
| `diff` | `git diff` in the clone | so a PASS is auditable |
| `notes` | harness | anything the row cannot express |

### Reading the context high-water mark

```text
slot release: id  0 | task 0 | stop processing: n_past = 41235, truncated = 0
```

Take **`n_tokens`** where the build emits it, else `n_past` at release. Take the
**maximum across the whole task**, not the last turn — a task that peaks at
turn 3 and ends at turn 9 needs the peak, and the window has to hold the peak.

Verify the parser against a known line before trusting a single number.
`bench/bench-cold-start.py` has three fixed instrument faults of exactly this
kind and a self-check that asserts its parser against a real log line; copy that
pattern rather than re-earning it.

---

## 6. Phases, in order

### Phase 0 — the ceiling, and whether the task set discriminates (Q4)

**Run all 19 tasks on the reference model first.** Without this the local
model's PASS rate is uninterpretable.

**Reference: Qwen3.8-27B FP8 via `gateway.9arm.co`.** Chosen because it is the
*same model at higher precision*. Any gap in PASS rate is then attributable to
**quantization alone** -- not to a different model, a different tokenizer, or a
different training mix. Substituting Claude or GPT here would answer a
different question ("is a frontier model better", which nobody doubts) and
would not isolate the variable this project exists to study.

- Mode: `standard`, skills ON -- the configuration the developer actually uses.
- One run per task. No repetition: this measures the tasks, not the server.
- Record `ctx_high_water` here too. The gateway is fast enough that Phase 0
  doubles as a cheap first estimate for Q2, though Phase 1 is authoritative
  because the local server's log is what the local window has to hold.

**How to read the result:**

| reference | local | reading |
|---|---|---|
| high | high | `UD-IQ2_XXS` is enough for this class of work. The headline answer to Q4 |
| high | low | quantization is the limit. The gap size is the cost of 2-bit, in tasks |
| low | low | **the task set does not discriminate.** Re-pick tasks; do not report a worker verdict |
| low | high | the harness favours local. Something is wrong -- investigate before reporting anything |

The third row is why Phase 0 comes first. Discovering it after 60 local runs
would waste the whole benchmark.

**The gateway is a network service.** Its failures are VOID, not FAIL, and a
run must be able to tell them apart -- a 502 is not a wrong answer.

### Phase 1 — how much context does a real task need? (Q2)

**One run per task. One config. A generous window. No repetition.**

This phase measures the **tasks**, not the decoder, so pairing and rotation do
not apply and would only cost hours.

- Config: `worker-iq2xxs-deep.ps1` (131,072, `ngram-mod`) — deliberately larger
  than anything expected, so nothing is truncated and the peak is real.
- Mode: `standard`.
- Output: distribution of `ctx_high_water` over 19 tasks.

**What to do with the answer:** take the 95th percentile, add the model's own
prompt overhead, round up to the next profile size. If p95 lands under 65,536,
say so loudly — that is roughly 1.5–2 GB of VRAM currently reserved for nothing,
and it is the largest lever this project has found since the desktop's
1,650–2,200 MiB.

**Do not average across modes.** `standard` and `clink` have different prompt
shapes; §6.3 exists because of that.

### Phase 2 — decoder comparison, at a window Phase 1 justifies (Q1)

- Arms: `ngram-mod` (incumbent) and `draft-dflash,ngram-mod` (the only arm that
  beat it on the synthetic prompt: +23.2 % [+16.3, +35.7], RESOLVED).
- **`none` is not run.** It measured 2.8× slower; on real tasks it would cost
  hours to re-confirm something already resolved.
- **`draft-dflash` alone is not run** unless Phase 1 shows tasks are
  context-poor. On the synthetic prompt it was −3.9 % [−9.6, +6.4] — inside the
  noise floor, and its own spread was only 0.8 % while the baseline's was 18 %,
  so the inconclusive verdict came from the baseline, not from it.
- Subset: 6–8 tasks spanning the size range, not all 19.
- Paired within a round, arms rotated per round, ≥3 rounds.
- Verdict from `harness.paired_deltas` — **resolved only if it clears 13.6 %
  AND keeps its sign.**

**Every task must be re-cloned between arms.** A second arm working in a clone
the first arm already edited is measuring a different task.

### Phase 3 — standard vs delegated workflow

`clink-subagents` changes the *shape* of the prompt, not only its length: the
orchestrator keeps the plan and hands each subagent a small self-contained
brief. If a task needs 60,000 tokens in `standard` and 20,000 per subagent
under delegation, that is a **larger lever on VRAM than any decoder setting**,
and a synthetic benchmark cannot see it.

- Invoke via `/using-clink`, then `/clink-subagents`.
- **`clink-masteragent` before any clink call** — `CLAUDE.md` requires it and
  it owns what may never be delegated.
- Record `ctx_high_water` **per agent**, not per task: the orchestrator's peak
  and the largest subagent's peak are different numbers and the window must
  hold the larger.
- The clink back-ends bill against flat subscriptions, so this phase is cheap
  in money and expensive in wall-clock. Budget accordingly.

### Phase 4 — do the skills earn their tokens? (Q3)

Same task, same model, same window; the only change is whether the skill layer
is loaded.

- Arm `skills-on`: the normal configuration.
- Arm `skills-off`: `--safe-mode`, which report 26 measured at **14,064 tokens
  against 153,621** for the same invocation.

Run on the **reference model as well as the local one**, if budget allows. On
the local model a skills win could be a context-pressure artifact -- 38,064
tokens is a large fraction of a small window, so skills could lose locally for
a reason that has nothing to do with whether the guidance is good. The
reference model separates *the skills are unhelpful* from *the skills do not
fit*.

**Subset: the same 6-8 tasks as Phase 2**, so the two phases share a spine and
their rows can be read together.

**What a result means.** If `skills-off` matches or beats `skills-on`, that is
not an argument to delete the skills -- it is an argument that the *catalogue
injection* is the wrong delivery mechanism for them, which is a different and
more actionable finding. Say which of the two the data supports, and do not
let a throughput result be read as a judgment on the content.

### Phase 5 — the budget frontier (the phase that produces the decision)

Specified in **§3.5**, which carries the ladder, the measured resident cost, and
why file size is not it. Summarised here so §6 is not missing a phase:

For every rung that fits the window Q2 justifies — boot it, record resident VRAM
and the layer split, add a drafter if it still fits, run the Phase 2 subset,
and plot PASS rate against rung.

**Start with `UD-IQ2_S` vs `UD-IQ2_XXS` + DFlash2** — the two sides of a trade
that was made on purpose before the drafter could be loaded, and has therefore
never been checked. Neither side has a task-success number.

The output is a frontier, not a winner. The developer picks the point.

### Phase 6 — grammar, and the combination nobody has run

**Cheap, and it gates the other phases' realism.** The configuration this
project intends to serve carries **both** a grammar and a drafter, and no
measurement has ever had both.

Why a grammar is not optional: without one, **41.5 % (`UD-IQ1_M`) to 58.3 %
(`UD-IQ2_XXS`) of corpus attempts emit no fenced code block at all**, having
looped inside the reasoning block until the token cap. On this benchmark's
rubric that is a FAIL every time, for a reason that has nothing to do with
whether the model knew the answer. **A task benchmark run without a grammar
measures formatting, not capability.**

What is already known, **from reading the source, not from a run**:

- `src/llama-grammar.cpp` allocates nothing on device — no `ggml_backend`, no
  `cuda`, no `ggml_new_tensor`. Its state is `std::vector` on the host. Expect
  ~0 MiB.
- `common/sampling.cpp:421` disables backend sampling when a grammar is
  present, and line 427 does the same for `--reasoning-budget`. Free for us:
  `common.h:295` defaults it off, no worker profile enables it, and it measured
  **+2.27 %, inert**.
- **`common.h:331` is a different field** — `backend_sampling = true` for the
  *draft* sampler, on by default — and the disable above does not reach it.

**Four boots, in one round, paired:**

| arm | grammar | drafter |
|---|---|---|
| 1 | off | off |
| 2 | **on** | off |
| 3 | off | **`draft-dflash,ngram-mod`** |
| 4 | **on** | **`draft-dflash,ngram-mod`** |

Record resident VRAM and tok/s for each. Arm 4 is the one that matters and the
one that has never existed.

**Run this before Phase 2**, and if a grammar costs materially more than the
predicted ~0, every window number from Phase 1 needs revisiting — the grammar
is not removable, so its cost is part of the budget, not an option in it.

---

## 7. Per-task procedure

Executed identically for every task. Deviating from it silently is how a
benchmark stops being one.

1. **Preflight.** Assert port 8080 is free
   (`bench/dflash2_arena.py:require_exclusive_port`). Two orchestrators on 8080
   killed a run on 2026-08-22 and the log ended mid-load with no error at all.
2. **Read the issue**: `gh issue view <n> --repo <remote> --comments`. Comments
   often carry the real constraint.
3. **Clone fresh** from the remote into `clones\<repo>-<issue>\`. Depth: shallow
   is fine unless the issue references a prior PR or commit, in which case take
   enough history to read it.
4. **Install** the repo's toolchain. Cache per repo where possible; MangaDock
   Backend/Frontend/MIT are three separate environments.
5. **Baseline the verify command** — run it *before* the worker touches
   anything. A repo whose tests are already red cannot score a task. If red,
   mark **VOID** and move on; do not fix the repo.
6. **Start the worker** on the phase's profile. Record `free_before`.
7. **Run the task** — `standard` or `clink` per phase.
8. **Verify** with §4's command. Capture exit code and output.
9. **Judge the diff** against the issue's stated defect (§4 rubric). Record the
   diff verbatim.
9b. **Copy the worker transcript out of the clone before deleting it.**
   `<clone>.stdout.txt` sits beside the clone and goes with the scratch
   root. Four tasks ran 24–40 minutes and changed nothing, and the only
   record of what the worker did was deleted with the tree.
10. **Record the row** to `results\`, including `ctx_high_water` from the log.
11. **Stop the server**, wait for VRAM to settle
    (`harness.vram_settled` with a `floor_mib`), record `free_after`.
12. **Delete the clone**, and verify the deletion by listing the path.

---

## 8. Cleanup protocol

A mechanism, not a promise.

```powershell
# The only deletion this benchmark performs. Guarded, not trusted.
$root = $manifest.scratch_root                     # e.g. D:\bench-scratch\2026-08-22-a
if (-not $root.StartsWith('D:\bench-scratch\')) { throw "refusing to delete $root" }
if (Test-Path 'D:\Github' -PathType Container -and $root -like 'D:\Github*') { throw "refusing" }
Remove-Item $root -Recurse -Force
if (Test-Path $root) { throw "cleanup FAILED, path still present: $root" }
```

1. Every clone lands under the recorded scratch root. Nothing else is a
   deletion target, ever.
2. After each task the clone is deleted and the deletion is **verified by
   listing the path**, not assumed from a zero exit code.
3. At session end the scratch root is deleted whole, and the run reports **what
   it deleted and what it could not**.
4. A clone that cannot be deleted — file lock, running `node`/`python` — is
   **reported by path**. The run does not retry with a wider pattern and does
   not kill unrelated processes.
5. `D:\Github\*` is checked against **before** any `Remove-Item`, not after.

---

## 9. Time budget and order

Wall-clock is the constraint; the GPU serves one arm at a time.

| order | phase | runs | note |
|---:|---|---:|---|
| 1 | harness shakedown on Tier A, reference model | 5 | a harness bug found on a 3 MB repo is cheap |
| 2 | **Phase 0** — ceiling, all 19 on the gateway | 14 more | no GPU time; the gateway is ~12× faster at prefill |
| 3 | **Phase 1** — context sizing, all 19 local | 19 | one run each, generous window |
| 4 | **Phase 2** — decoders, 6–8 tasks × 2 arms × 3 rounds | 36–48 | the long pole, and the only phase that needs pairing |
| 5 | **Phase 3** — standard vs clink, subset × 2 | 12–16 | clink back-ends are slow in wall-clock, cheap in money |
| 6 | **Phase 4** — skills on/off, subset × 2 (× 2 models) | 12–32 | run on the reference too if budget allows |
| 7 | **Phase 5** — the budget frontier, rungs that fit × subset | 18–40 | the phase that produces the decision. Needs Q2's window first |
| — | **Phase 6** — grammar × drafter, 4 boots | 4 | **do this early, before Phase 2.** Cheap, and it decides whether every other phase is measuring capability or formatting |

**Stop after any phase and the work still has value** — that ordering is
deliberate. Phase 0 alone answers Q4's headline. Phase 0 + 1 answers Q4 and Q2,
which are the two that change what gets served.

**Do not start Phase 2 before Phase 1 finishes.** Phase 2's window comes from
Phase 1's answer; running it first means running it twice.

**Notify on a phase boundary, not per task** (`CLAUDE.md`): a queue can hold the
GPU for hours, and one digest at the end of a phase is the unit of progress
worth interrupting for.

---

## 10. What would make this benchmark wrong

Written down in advance, so a later reader can check whether it happened.

- **Tests that were already red.** §7 step 5 exists for this. A repo baselined
  after the worker ran cannot distinguish the worker's damage from the repo's.
- **A prompt that flatters one arm.** The synthetic prompt was 66.2 % duplicate
  lines and handed `ngram-mod` its best case. Real issues avoid this by
  construction — that is the main reason for this plan.
- **Scoring by diff size or by "looks right".** Only the verify command and the
  stated defect count.
- **Judging with the model under test.**
- **Pooling `standard` and `clink` context numbers.**
- **Reusing a clone between arms.**
- **A VOID counted as a FAIL.** A harness failure is not a worker failure, and
  merging the two makes the worker look worse the buggier the harness is. A
  gateway 502 in Phase 0 is a VOID, not evidence about FP8.
- **Reporting a local PASS rate with no ceiling beside it.** "8 of 19" is not a
  finding. Q4 is a *gap*, and a gap needs two numbers.
- **Changing more than one factor between two rows being compared.** Model,
  skills, mode, decoder and window are five factors; each phase moves exactly
  one. A row that moved two answers nothing and cannot be salvaged afterwards.
- **Reading a Phase 4 result as a verdict on the skills' content.** It measures
  one delivery mechanism -- a 38,064-token catalogue injected as a user message
  -- not whether the guidance in those files is any good.

---

## 11. Open questions this plan does not settle

- **Depth transfer.** A verdict at one window does not carry to another:
  `draft-mtp` is +81 % at 16K and −71 % at 131,072 on the same artifact. Phase 2
  answers only at the window Phase 1 picks.
- **Quality of output**, as distinct from task success. Speculative decoding in
  llama.cpp is verification-based and should be byte-identical to no
  speculation; `bench/spec_output_identity.py` was written to check that and
  **has not been run**.
- **Whether 19 tasks is enough.** It is 25 % of the eligible pool and covers
  four repos and three languages. If PASS rates cluster at 0 % or 100 %, the
  set is not discriminating and needs re-picking, not re-running.
