# Workflow — how work moves here

**PRD → issues → PR.** Never a PR without a referenced issue; the `pre-push`
guard and the `t4-gate` hook both block it.

This repo is a measurement project, so the pipeline has one shape the standard
does not assume: **most units of work are experiments, not features.** An
experiment ships a number, and the number is the deliverable.

---

## The loop for an experiment

```
1. State the question so it can come out either way  → verify: a wrong answer is
                                                        possible and named
2. Name the arms, the depth, the artifact, the rounds → verify: order reverses
                                                        between rounds
3. Run it                                             → verify: a row lands in
                                                        results/*.jsonl
4. Read the row, not the summary                      → verify: the number you
                                                        quote is in the file
5. Register it in docs/results/                        → verify: the row names
                                                        the raw file
6. If it contradicts something published, correct it  → verify: CORRECTIONS.md
                                                        entry AND an audit rule
```

**Step 6 is the one that gets skipped**, and skipping it is how two versions of a
finding end up in the tree. A retraction is not finished until
`scripts/audit-stale-claims.py` can find the surviving copies.

## The loop for a code change

**TDD is mandatory in `qwen38-tuning/bench/`.** Red first, and the test is named
after the incident it guards — read `bench/tests/test_harness.py` before adding
one. Every primitive there exists because the untested version returned a
believable wrong number.

```powershell
cd qwen38-tuning\bench ; python -m pytest tests\ -q
```

Anywhere else, `karpathy-guidelines`: the smallest thing that solves it, a diff
that traces line by line to the request, and success criteria you can check.

## Verification

The fast gate, and what CI runs:

| check | command | what it catches |
|---|---|---|
| `test` | `cd qwen38-tuning/bench ; python -m pytest tests/ -q` | the instrument returning a number instead of a failure |
| `lint` | `python scripts/check-doc-links.py` | a dead link in a documentation set that IS the deliverable |
| `typecheck` | `python -m compileall -q -x "_work" qwen38-tuning/bench scripts` | a heredoc patch that produced an unterminated string — this has happened twice |

**There is no `build`.** Nothing here builds, and a job that runs `true` is a
green light that means nothing.

## Issues and PRs

**Bilingual bodies** — English plus a full Thai mirror of the same depth. A
mirror is not a summary. Tracker only; `docs/` stays English.

**Close with evidence**: a commit, a passing test, or a measured number with the
file it came from. Never silently.

**Labels** — Type / Component / Severity plus the five triage roles. A `security`
issue must be `critical` or `Major`.

## Running unattended

Normal here — a queue can hold the GPU for hours. Rules learned the hard way:

- **One orchestrator per port.** `qwen38-tuning/scripts/swap-model.sh` takes a
  lock. Two queues sharing 8080 destroyed a 30-task corpus and the summary still
  printed a plausible number.
- **Check for duplicates before launching**: `ps -ef | grep afk-q`.
- **A step that fails must not take the next one with it.** It did, once: a
  timeout skipped the teardown, left a 12 GB server resident, and the following
  step died on a GPU that was still full.
- **One digest at the end**, enumerating every gate as ran / not-run / n-a. A
  list of what ran, with the skipped ones absent, reads as completeness.

## When a measurement contradicts a published claim

1. Write the correction **in the document that carries the claim**, in place.
2. Add an entry to `docs/reports/CORRECTIONS.md` saying what was wrong and where
   the correction lives.
3. Add a rule to `scripts/audit-stale-claims.py` so the surviving copies are
   findable.
4. If the claim reached something outward-facing, say so in the correction.

Steps 2 and 3 are what make it stick. Step 1 alone has failed here before.
