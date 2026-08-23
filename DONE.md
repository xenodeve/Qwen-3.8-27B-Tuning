# Ship log

**What past sessions actually shipped, and how it was validated.** Newest on
top, one dated `##` heading per unit so an agent can jump rather than scan.

This file answers a question the ledger cannot: *has this already been tried, and
what came of it?* Archive to `DONE-archive-<period>.md` when it crosses a few
hundred lines.

---

## 2026-08-23 — eight techniques from the RTX 3090 pool, and the biggest one was already on

**The pool had one row left open.** `08-rtx3090-transfer.md` called
recurrent-state prefix reuse *"the single largest untested idea left"*. Answering
it required measuring at the window we serve, and that is what broke three
published claims. Narrative:
[report 33](docs/reports/33-WHAT-THE-3090-POOL-ACTUALLY-GAVE-US.md). PR #39,
issue #38.

**`-cram` is worth 343x and nobody knew it was on.** `--cache-ram` defaults to
**8192 MiB** and stores the whole sequence state — attention KV and recurrent
together — for idle slots. Returning to a 44K conversation after working on
another costs **118.2 ms at 100 % reuse**; with `-cram 0` it costs
**40,596 ms at 0 %**. Cold turns agree to 0.35 %, so the arms are comparable.
It surfaced only because a slot erase failed to produce a cold turn. Costs
898-928 MiB of host RAM per conversation; restore is a *move*, not a copy.

**Prefix reuse inside one conversation transfers too.** A warm turn costs the
same **~250 ms whether the conversation is 8,147 or 44,255 tokens** — 99.9 %
reuse, 99.3 % saved. It works despite `n_rs_seq = 0`, carried by
`--ctx-checkpoints`. But an edit ahead of the suffix does not degrade reuse, it
**zeroes** it, and at 44K that is 41.8 s.

**The window we serve was never the problem.** `04-context-depth.md` recorded
decode at ctx 98,304 as 2.8-5.0 tok/s and called it a property of the window.
All sixteen of those rows loaded DFlash2. Six paired rounds with the arms
alternated: **`ngram-mod` alone returns 96.92 tok/s median, 6 of 6 rounds
finishing** — faster than the 75.2 median at 16,384 — against 5.66 and two
timeouts for `dflash2+ngram`. Free VRAM splits cleanly and does not overlap:
769-2,117 MiB without the drafter, 45-376 with it.

**`--fit` was never following anything.** The north star said free VRAM at boot
moves 9,326-10,732 MiB and `--fit` follows it. **llama.cpp has reported 11,069
MiB free in all 552 logs this project has kept**, and 148 of 150 boots on our
artifact end in *"no changes needed"*. The moving range is `nvidia-smi`'s view
of the card; `--fit` reasons from the constant one. Pinning `-ngl 65 --fit off`
was tried and changes nothing observable. **The no-cross-boot rule stands and
its cause is now unattributed.**

**`-ub` refused.** A 4x cut returns 66 MiB and costs **14.0 % of decode,
RESOLVED** — and 66 MiB does not reach the band where it was wanted.

**Four closed by reading source, no GPU round spent.** The sharpest:
**fp16 recurrent state would return 360 MiB and corrupt silently** —
`gated_delta_net.cu` has zero type checks and casts the state to `float *`
unconditionally. The scan rates it `small-patch`; it is new-backend. Also found
available and never set: `"timings_per_token"` and `"return_tokens"`, both plain
request booleans that serve the recorder (#30-#36).

**Retracted: `CORRECTIONS.md` 25, 26 and 27**, taking the register to
twenty-seven. All three share one shape — **the conclusion was right and the
stated mechanism was wrong** — which is worse than a wrong number, because a
wrong mechanism tells the next reader what to fix. This session spent real time
pinning `-ngl` against a force that was not there.

**Nothing shipped.** All four `worker-*.ps1` run `ngram-mod` alone and are
correct as they stand for this window, which is now measured rather than hoped.
Suite 253 -> 287; `traps.md` gained a thirteenth entry, written about this
session's own bespoke script reporting 71.76 tok/s where the harness reports
96.9.

## 2026-08-23 — the benchmark was measuring the wrong tree, and the served window does not work

**Two independent faults, each of which alone explains a result this project
published.** Neither was visible in any column it records; both were found by
instruments built the same day.

**The worker was editing `C:\AI`, the live repository, not the clone.**
OpenCode keeps a per-project server alive and `run` attaches to whichever is
listening, carrying **the project root it was first started with** — so
`cwd=<clone>` was ignored. `git diff` in the clone was empty and the harness
recorded *"the worker changed nothing"* about work that may have been done
correctly in the wrong place. **Five real GitHub issues, 24–40 minutes each,
retracted** (`CORRECTIONS.md` §24). Reproduced deliberately: `cwd=` alone gives
`EDIT_NO_DIFF`, 0 diff bytes and a modified live tree; **with `--dir`, `EDITED`,
251 diff bytes, 32.8 s, live tree untouched.** Fixed in both drivers, pinned by
`bench/tests/test_worker_workdir.py`, which deliberately does **not** assert on
`cwd` — that is the thing that looked right and was not.

**The hazard had been documented two days earlier**, in `opencode_corpus.py`,
symptom and all. The driver that walked into it was a few hundred lines away and
nothing made anyone re-read it.

**~~Decode at the served ctx 98,304 is 2.8–5.0 tok/s~~** — measured correctly,
attributed wrongly, and **retracted the next day**
([`CORRECTIONS.md` §26](docs/reports/CORRECTIONS.md)). All sixteen of those rows
loaded the DFlash2 sidecar, so depth and drafter never varied independently.
With `ngram-mod` alone — what every worker profile serves — the same window
returns **96.92 tok/s over 6 of 6 rounds**. Cold prefill falls
**1,129 → 924 → 74.3** over the three depths with the drafter loaded, and has
not been re-measured without it. Found at all because `gpu_trace.py` happened to
be running and showed 100 % utilisation at **76 W** with **32 MiB free** — a
signature this entry read as memory-bound, which the same trace refutes at
`utilization_memory` **4 %**.

**Shipped:** `bench/gpu_trace.py`, `bench/edit_canary.py`,
`harness.window_repetition_pct`, `bench/corpora/real-code-deep.txt`, a
truncation guard on `filler()` that raises instead of silently shortening a
prompt, and `docs/agents/traps.md` — twelve ways of working that failed here,
each with its guard or an admission that none exists. Suite 246 → **253**.

**Also recorded:** `CORRECTIONS.md` §21–§24, report 32 as the standalone
hand-off, and a documentation sync across all 115 markdown files that repaired
**four control bytes living in the tree** — three BEL and one backspace, every
one a backslash swallowed by a shell heredoc, invisible to both existing
checkers.

## 2026-08-22 — DFlash2 measured, and the optimum moves with the window

**Built llama.cpp PR #27342 (build 10499) beside 10472** so neither replaces the
other, and measured DFlash2 against the incumbent on one binary.

**`draft-dflash` is +34.7 % over `ngram-mod` on real code and −9.2 % on the
prompt this project had been using** — the same session, same binary, same
window. The synthetic prompt was 66.2 % duplicate lines; the corpus is now
frozen and hashed into every row (`CORRECTIONS.md` §20).

**Six techniques from a 434-item scan of an external RTX 3090 stack were
measured**: three wins (`--spec-draft-n-max` +23.4 %, the `draft-dflash,ngram-mod`
pair +48.5 %, `--spec-ngram-mod-n-match 24` +34.6 %), two nulls, and one
refutation of the claim that produced it. Five more were closed by reading the
source without spending a GPU round.

**The two winners cancel when combined** — −31.6 % and −33.8 % against each
single arm, both RESOLVED, at 52.4 % of the independent expectation. And
**`n-match`'s optimum moves with the window**: 24 wins at 16,384, **16** wins at
65,536, and the value all four profiles ship (12) loses at both.

**Nothing shipped.** Every worker profile is unchanged, deliberately — a verdict
at one depth does not transfer, and the served window has no verdict at all.

## 2026-08-21 — PR #3 merges: the cold start turns out to belong to the harness

**Shipped:** merge commit `9e6e7ad`, 21 commits from 14 issues, +2,483 lines.
Issues #1, #2, #4–#14 closed with evidence; #15 left open on purpose.

**Serving.** `templates/qwen38-late-system.jinja` — the model's own chat template
with one line changed, so a trailing `system` message no longer 500s. Without it
Claude Code cannot talk to this server at all: its `SessionStart` hook output
arrives as a 25–33 KB trailing system block. 50 consecutive failures before, 0
after. Two more worker profiles joined the original pair, each with a header that
says which harness actually fits it.

**The finding.** The cold start is not the model, the quantization, CUDA, the KV
type, the micro-batch, the context window or the tool schemas — every one of those
was measured and is flat. It is a **352-skill catalogue injected as a user message
and paid three times per invocation**: 153,621 tokens against 14,064 under
`--safe-mode`, 171 s of prefill against 16 s, for the word `hi`. `qwen --safe-mode`
or `disable-model-invocation: true` both remove it; the second keeps every other
feature but also puts the skill out of the model's reach, which is measured.

**Control group.** The same catalogue costs nothing on a gateway because it
prefills at roughly 11,000 tok/s against our 900 — not because the harness sends
less. Captured through a recording proxy: 54,478 and 57,700 tokens against our
54,485 and 56,277, five calls either side.

**Decoders, re-measured.** `draft-mtp` loses with 467–773 MiB free and loses again
at `N_PREDICT = 1024`, so neither the VRAM cliff nor the 160-token rule explains
its −71 %. DFlash 2 does not load on build 10472 at all — llama.cpp support needs
PR #27342 — so the "screened, not competitive" register row described a screen
that could not have run. `ngram-mod` costs **0 MiB**, measured.

**Validated:** 136 tests (up from 111), 143 links 0 broken, every module parses.
`/code-review` and `/scrutinize` both ran before merge and found four things;
three are fixed on the branch, the fourth is the shape of the PR itself.

**Corrections §14–§18**, each with an audit rule. Four of the five are this
session contradicting itself within hours — the free-VRAM threshold, Qwen Code's
request size twice, and a profile sized from a benchmark prompt that was a floor.

**The gate that is now weaker, on purpose.** CI cannot run on this account, so
`required_status_checks` was removed on a developer-initiated waiver and
`.claude/t4.json` `verify` was widened from `pytest` alone to all three commands
the workflow runs. The web UI and other clones are unguarded until it is reverted:
**#15**, with a checklist.

---

## 2026-08-21 — the workspace comes under version control, and gets its gates

**Shipped:** `git init`, first commit of 432 files / 3.8 MB, public repo
`xenodeve/Qwen-3.8-27B-Tuning`. Then the T4 operating layer: `CLAUDE.md`
(bilingual), the hooks layer, the guards layer, CI, this log, and the ledger.

**Validated:** `python scripts/check-doc-links.py` — 83 files, 124 links, 0
broken. `pytest bench/tests/ -q` — 111 passing. `git config core.hooksPath`
returns `.githooks`.

**Deliberately not shipped:** a `build` job in `t4-verify.yml`. This repo builds
nothing, and a job that runs `true` is a green light that means nothing — the
exact failure `ci-cd-layer.md` warns about. Three real checks, not four with one
theatre.

**Excluded from the repo:** the vendored llama.cpp CUDA build (~1.7 GB of DLLs),
model artifacts, runtime logs, the port lock, the OpenCode worker scratch dir.
Scanned for secrets before making it public — none found; every `apiKey` in a
config this session wrote is `none` or empty, and auth comes from the
environment.

## 2026-08-21 — first measurement of the worker that actually ships

**Shipped:** `bench/opencode_corpus.py` — runs the coding corpus through
OpenCode against the local server, graded by executing the produced file against
the same hidden assertions `run_retry_bench.py` uses.

**Result:** `v3-iq2xxs` at 131,072, lean OpenCode profile — **6/10 accepted,
7/10 wrote the target file, 16.5 accepted tasks/hour**, decode 35–61 tok/s
(median 45) on real code.

**Why it matters:** every prior quality number described a single chat
completion graded for a fenced code block, which nothing in production does.
The failure mode changed with the harness — one task failed on `AssertionError`
(code that ran and was wrong) where the old harness failed on missing fences.

**Also shipped:** `bench/prefix_probe.py`, a recording endpoint that measures
what a harness sends without changing either side. It found OpenCode's default
profile sending **99,073 tokens** of prefix — one token over the window once its
32,000-token output reservation is added, which is why the first call failed
before any work started. The lean profile is **~5,377**, measured by our own
server's tokenizer. 94.5 % of the prefix was a skill catalogue and MCP tool
schemas the worker cannot use.

## 2026-08-21 — nine instrument faults found and fixed, suite 81 → 111

**Shipped, all red-first, each test named after the incident it guards:**

- `completion_timeout_s` — a flat `timeout=3600` spent one hour to the second on
  an arm whose prefill had collapsed to 8.56 tok/s. Now sized from the depth.
- `vram_settled` + `wait_for_vram_release` — a flat `sleep(5)` did not wait for
  the driver. **Measured: an 11,501 MiB teardown takes 9.87 s.** The `floor_mib`
  argument exists because "stopped moving" cannot tell *finished* from *not
  started*, a bug caught in the fix before it shipped.
- `try/finally` around the whole measurement — a raise used to leave a 12 GB
  server resident and take out the next queue step.
- `draft_acceptance` — acceptance was computed from the **first of five**
  generations while `tg_med` was the median of all five.
- `line_repetition_pct` — turns "is it looping" into a number.
- `--fixed-text`, `--n-predict`, `--filler` — the sweep's own text is now a
  declared variable rather than a hidden constant.

**Validated:** 111 tests pass; `kill()` exercised live on both branches;
sixteen server boots with sixteen clean teardowns during the night's batches.

## 2026-08-21 — the documentation becomes navigable, and auditable

**Shipped:** `docs/results/` (the register: has X been tried, what happened),
`docs/reports/CORRECTIONS.md` (twelve published claims later contradicted),
`scripts/check-doc-links.py`, `scripts/audit-stale-claims.py`.

**Why:** a corrected report and an uncorrected one both existed, and nothing
could find the copies. The audit found 257 lines matching a superseded claim
across 37 files on its first run — including two that a hand sweep had missed.

**Validated:** the audit immediately caught the "it loops" wording surviving in
two documents after the retraction was written.
