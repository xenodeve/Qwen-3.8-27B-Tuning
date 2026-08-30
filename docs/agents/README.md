# agents — the operating standard

**How to work in this repository.** `CLAUDE.md` is the entry point and points
here; these five files hold the detail it compresses.

| file | what it settles |
|---|---|
| [`traps.md`](traps.md) | **the ways of WORKING that failed here**, each with the guard that catches it — or a statement that nothing does. Item 3 of the session-start list |
| [`domain.md`](domain.md) | the glossary. Every term this project uses in a non-obvious sense, defined once |
| [`workflow.md`](workflow.md) | PRD → issues → PR, the gates, and what may never be delegated |
| [`issue-tracker.md`](issue-tracker.md) | the tracker, and why issue and PR bodies are bilingual |
| [`triage-labels.md`](triage-labels.md) | the label vocabulary and what each one commits you to |

---

## Read `traps.md` before you read the rest

`CORRECTIONS.md` records numbers this project got wrong. `traps.md` records the
**methods** that produced them, and it is the shorter read of the two.

**Thirteen of its fifteen traps produced a plausible number or a clean exit rather
than an error** — `split: 65+0` while the card thrashed at 32 MiB free,
`diff_bytes: 0` while the worker edited a different repository, `rc=0` on a task
that did nothing. That is the shape to watch for, and it is why this folder
exists at all.

---

## What belongs here, and what does not

**Here:** anything an agent needs in order to work correctly that is **not**
derivable from the code, the git history or the results. Conventions,
vocabulary, hazards, the tracker contract.

**Not here:** measurements (`../results/`), narrative and evidence
(`../reports/`), intent (`../plans/`), or external material (`../researchs/`).

These files are **English only**, by the rule in
[`issue-tracker.md`](issue-tracker.md): the bilingual requirement covers tracker
bodies — issues, PRDs, PR descriptions — never `docs/`.
