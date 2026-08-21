# Ship log

**What past sessions actually shipped, and how it was validated.** Newest on
top, one dated `##` heading per unit so an agent can jump rather than scan.

This file answers a question the ledger cannot: *has this already been tried, and
what came of it?* Archive to `DONE-archive-<period>.md` when it crosses a few
hundred lines.

---

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

**Shipped:** `docs/tested/` (the register: has X been tried, what happened),
`docs/reports/CORRECTIONS.md` (twelve published claims later contradicted),
`scripts/check-doc-links.py`, `scripts/audit-stale-claims.py`.

**Why:** a corrected report and an uncorrected one both existed, and nothing
could find the copies. The audit found 257 lines matching a superseded claim
across 37 files on its first run — including two that a hand sweep had missed.

**Validated:** the audit immediately caught the "it loops" wording surviving in
two documents after the retraction was written.
