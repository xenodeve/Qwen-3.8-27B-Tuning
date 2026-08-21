# Ship log

**What past sessions actually shipped, and how it was validated.** Newest on
top, one dated `##` heading per unit so an agent can jump rather than scan.

This file answers a question the ledger cannot: *has this already been tried, and
what came of it?* Archive to `DONE-archive-<period>.md` when it crosses a few
hundred lines.

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
