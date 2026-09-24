# Task-neutral thinking principles for an agent — what generalises, and what only looks like it does (2026-09-07)

> **Mixed provenance. Read the tags, not the sentences.**
> **MEASURED HERE** names the file in this repo the claim comes from.
> **EXTERNAL** is an outside claim, unverified on this machine.
> **UNMEASURED** is a candidate with no evidence on either side.
>
> Asked for: *"หลักการคิดที่ดีของ agent ที่สามารถใช้กับทุก task ร่วมกันได้มีอะไรบ้าง"* —
> which principles hold across every task, rather than per domain.
>
> **The caveat that governs the whole document.** Almost all the local evidence
> is a record of *this project's own agents*, which are frontier Claude
> sessions. The only body of evidence measured **on Qwen3.8-27B** is the
> 2026-09-05 quality bench. So most rows below are `MEASURED HERE (on us) /
> UNMEASURED (on Qwen)`, and that gap is stated per row rather than averaged
> away. A principle that holds for a frontier model in an agent loop is not
> thereby a principle that transfers to a 27B model reading a skill file.

---

## 1. The filter — what makes a principle task-neutral

A principle belongs in the shared layer if it constrains **how you decide and
how you check**, not **what a good artifact contains**. The test is a
substitution:

> Swap the domain — page → report → refactor → benchmark. If the sentence still
> means the same thing, it is task-neutral. If it needs rewriting, it is craft
> and belongs to the domain skill.

**MEASURED HERE** — this filter is not an aesthetic preference; it was forced by
an incident. `using-qwen38` carried *"include the deliverables a brief never
names — dark mode and Open Graph for a page"* in its rule 1. That sentence fails
the substitution test (a report has no Open Graph), and the router was the file
every task paid for. Issue #366 removed it. The replacement, rule 4, passes the
test by construction: *"a choice you would have made for any brief is a default,
not a decision"* reads identically for a colour, a section order and a module
boundary.

**The cost of getting this filter wrong is measured, and it is not neutral.**
Same brief, same model, each arm in its own isolated `CLAUDE_CONFIG_DIR`, three
reps (`docs/results/11-quality-bench-2026-09-05.md`, the arm table read directly,
not via the skill's restatement of it):

| arm | gate, each rep | OG | dark | **≤ 2 fonts** |
|---|---|---|---|---|
| the gate alone (checks, no knowledge) | **8/8 · 8/8 · 8/8** | 3/3 | 3/3 | 3/3 |
| the design knowledge suite alone (prose, no checks) | **4/8 · 4/8 · 4/8** | 0/3 | 0/3 | **0/3** |
| no skill at all | **6/8 · 6/8 · 6/8** | 0/3 | 0/3 | **3/3** |

Prose knowledge in context scored **below having nothing**. But the `≤ 2 fonts`
column does **not** support the reading first written here — *"the document that
states the rule is what broke it"* — and the per-rep numbers are why:

| arm | r1 | r2 | r3 |
|---|---|---|---|
| noskill | 2: Fraunces, Inter | **0:** | **0:** |
| designonly | 3: Fraunces, Space Grotesk, Space Mono | 3: Archivo, Fraunces, Space Mono | 3: Fraunces, Space Grotesk, Space Mono |
| gateonly | 2: Fraunces, Space Grotesk | 2: IBM Plex Sans Thai, Noto Serif Thai | 2: Kanit, Trirong |

**`noskill` scored 3/3 on the cap because two of its three runs loaded no
webfonts at all.** The check was green for the wrong reason — trap 16's exact
shape, and it was reported here before the raw column was read. Without the
skill the model largely did not do typography; with it, it did, and overshot by
exactly one family, a **mono** in two runs of three. Overshooting a numeric cap
with serif + sans + mono is a different failure from ignoring type entirely, and
the summary column collapses them.

**What survives, and it is the more interesting claim.** `noskill` scored
**6/8** against `designonly`'s **4/8** while producing a page with no
typography, no Open Graph and no dark mode, in 4.8 minutes and 2 tool calls.
Doing less scored higher than doing more. That is a **second, independent
instance of the #366 finding** — the gate scores the absence of eight defects,
so an arm that attempts less has fewer surfaces on which to be caught, and the
score cannot separate restraint from inactivity.

So the honest conclusion from this table is narrower than the one first drawn:
**prose knowledge without a check did not make the model do the wrong thing —
it made the model do more, and the gate charged it for the difference.** The
catalogue below stays short for the reason in §4.2, not because of this table.

---

## 2. The spine — one principle both bodies of evidence converge on

Two independent lines arrive at the same place from opposite directions.

**EXTERNAL.** Huang et al., *Large Language Models Cannot Self-Correct Reasoning
Yet* ([arXiv:2310.01798](https://arxiv.org/abs/2310.01798)), abstract, verified
by fetching the paper: *"LLMs struggle to self-correct their responses without
external feedback, and at times, their performance even degrades after
self-correction."* Re-reading your own work is not a check; it can be worse than
no check.

**MEASURED HERE.** `docs/agents/traps.md` records nineteen numbered method
failures in this repo. **Fifteen of them produced a plausible number or a clean
exit rather than an error** — `rc=0` on a task that did nothing, `diff_bytes: 0`
while the worker edited another repository, a full-length prompt reported for a
window that was 40 % filled. Self-review had nothing to catch on, because the
artifact looked exactly like success.

**MEASURED HERE, on Qwen.** A rule delivered as a sentence ("two font families
at most") transferred **4 times out of 9**. The same rule as a command with a
pass condition (`grep … | sort -u`, "the list has ≤ 2 entries") was run
**unprompted, six times, in the 8/8 run** (`qwen38-skill-style` rules 1–2).

So the spine is one sentence, and it has two halves that are both load-bearing:

> **Replace every judgement with an observation — and make sure the observation's
> failure looks different from its success.**

The second half is what most check-writing gets wrong, and this repo has four
separate incidents of exactly that:

- **trap 3** — a test asserting `cwd == clone` would have been green throughout
  the entire incident where the tool ignored `cwd`. *Ask of every guard: would
  this have been green while the bug was live?*
- **trap 16** — eight assertions that measured the shape of a file; **two were
  green for the wrong reason**, one for days. `"-sm" in t and "tensor" in t` was
  green **before the flag existed**, because both words appeared in a comment.
- **the design gate, 2026-09-06** — `gate-dark.js` printed a passing
  `invisible: 0` while its candidate regex examined nothing. The fix was not a
  better threshold; it was printing `candidates: n` so that a run which looked at
  nothing stops reading like a clean page.
- **#365, open** — check 3 greps for `prefers-color-scheme` and passes at one
  match, so a page that is dark and has no light mode scores `dark yes`. It has
  reported that since it was written.

The general form, which is the last line of `traps.md` and the most portable
sentence this project has produced:

> **When something reports success, ask what it would have reported had it
> failed. If the answer is "the same thing", you have not measured anything yet.**

---

## 3. The catalogue

Grouped by phase. Each row names its evidence and, separately, whether it has
been tested on Qwen.

### Before acting

| principle | evidence | on Qwen |
|---|---|---|
| **Enumerate what the brief leaves unsaid, as a table, before the first action.** An empty right-hand cell is a missing deliverable, not a choice. | **MEASURED HERE**: Open Graph 0/3 and dark mode 0/3 without the table, 3/3 and 3/3 with it (`11-quality-bench-2026-09-05.md`) | **yes** — this is the one principle with a direct Qwen A/B |
| **Check the brief for flaws with a command, not a feeling** — contradiction, missing fact, impossible ask, each checked. | **MEASURED HERE**: `qwen38-think`; on an impossible brief the skill produced a written pushback where the gate alone produced "Done. CODE GATE: 6/6" (2026-09-05 17:38) | **yes** |
| **Take the open choices from this brief's own subject.** A choice you would have made for any brief is a default, not a decision. | **MEASURED HERE (weakly)**: #366, driven by the developer's judgement on 26 pages, not by a score | **no** — the router carries it, nothing has measured it |
| **Resolve decisions in dependency order.** | **MEASURED HERE**: trap 8 — "can speculation be attributed per period" was closed *before* "what is a period", so the true answer did not answer the requirement | **no** |

### While acting

| principle | evidence | on Qwen |
|---|---|---|
| **Read what already exists before writing a parallel thing.** Prefer extending the existing harness; if you write a parallel one, list which of the host's guards you are choosing not to inherit. | **MEASURED HERE, three times**: traps 1, 5, 13. In trap 13 the hazard was documented **in the file the script imported**, four lines above the code it did not use — and the number it produced was reported to the developer before the harness contradicted it | **no** |
| **One variable at a time; ask what the experiment held fixed.** *If the answer includes the thing you are blaming, you have not measured it.* | **MEASURED HERE**: trap 12 — sixteen rows blamed depth; every one had loaded the DFlash2 sidecar, so depth and drafter never varied independently. The `args` column said so for four days | **no** |
| **A failure that is a capability does not change on retry; a failure that is a resource can.** Keep the dead set per depth and per regime, not globally. | **MEASURED HERE**: trap 19 — and inheriting the verdict globally would have skipped `draft-dflash` at 16,384, the fastest configuration measured anywhere in this work | **no** |
| **Never install, never search for tools, never spawn agents; say the tool is missing in one line and continue.** | **MEASURED HERE**: one run spent six turns installing a Thai spellchecker that does not exist; another two turns hunting for `gh` | **yes** |
| **A missing tool is a written line, not a stop.** Pushback is reported, never a question that halts the work. | **MEASURED HERE** (`qwen38-think`). **EXTERNAL** corroboration: the "hard failure mode" pattern — a deterministic exit path signalling inability rather than probabilistic guessing (blog-tier source, not a paper) | **yes** |

### Checking

| principle | evidence | on Qwen |
|---|---|---|
| **Would this guard have been green while the bug was live?** | **MEASURED HERE**: trap 3 | **no** |
| **Assert on behaviour, not on the shape of the file.** Run the thing; assert on the resolved value; scope to the invocation; one chokepoint then forbid the pattern elsewhere. | **MEASURED HERE**: trap 16 (eight brittle assertions, two false-green) and CORRECTIONS §34 | **no** |
| **A detector that reports nothing must be probed with a known defect** — feed it one, watch the count go 0 → 1, delete the probe. | **MEASURED HERE**: the `design-ship-gate` suite had 43 assertions and inverting `d < 60` to `d > 60` left **all 43 green**; the fixture-driven replacement fails 3 | **no** |
| **A verdict carries the configuration it was measured in.** | **MEASURED HERE**: CORRECTIONS §33 (`-ts` "is not a lever" — measured under one split mode) and §35 (a depth that loads is not a depth that serves; the probe's size is part of the claim) | **no** |
| **Copying a configuration copies the assumptions of whoever sends the rest of it.** | **MEASURED HERE**: CORRECTIONS §36, §39 — two Studio flags taken as one memory decision were two positions, and the one that mattered was never argued | **no** |
| **Agreement with a default is agreement with nobody.** | **MEASURED HERE**: CORRECTIONS §38 | **no** |

### Reporting

| principle | evidence | on Qwen |
|---|---|---|
| **Distinguish a verdict from a hypothesis, and name the file the number came from.** | **MEASURED HERE**: `CLAUDE.md`'s north star; thirteen documented instrument faults each produced a plausible wrong figure | partial |
| **Report what did not run, not only what did.** A true report with a hole in it reads as completeness. | **MEASURED HERE**: `t4-afk` — listing the gates that ran and omitting the rest is how nine PRs shipped with three gates at zero | **no** |
| **One report shape across every skill, and the order is fixed.** | **MEASURED HERE**: 2026-09-05 17:38 — two skills loaded, the model ended with the later one's report and the earlier skill's pushback had no line to land on | **yes** |
| **Do not put an unverified boundary in a commit message.** Commit messages are read as results by everyone who comes after. | **MEASURED HERE**: trap 15 / CORRECTIONS §30 — "the boundary is prompt length, between 43k and 64k", refuted by seven points | **no** |

### The one external principle with no local twin

**EXTERNAL** — Anthropic, [*Building Effective AI Agents*](https://www.anthropic.com/engineering/building-effective-agents)
and [*Effective context engineering for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents):
use the simplest architecture that works and add complexity only when
demonstrated necessary; prefer transparency — show the planning steps rather
than hiding them in implicit reasoning; find the smallest set of high-signal
tokens.

This is the same shape as the 4/8-below-6/8 result above, arrived at from
architecture rather than from token budget, and it is the argument for keeping
the shared layer small. **UNMEASURED here as stated** — this project has never
run a with/without on document length itself.

---

## 4. Four things that do **not** generalise, and one that is contested

1. **A verdict at one depth does not transfer to another.** `draft-mtp` is +81 %
   at 16K and −71 % at 131,072 on the same artifact (`CLAUDE.md`). And **a
   mechanism story is not an exception** — trap 6 records a plausible mechanism
   written to explain why `n-match 24` should widen its lead at depth; it was
   backwards. A mechanism can be told in either direction, so it feels like
   evidence and is not.
2. **More principles is not better.** §1's table is the evidence. Every line in
   the shared layer is paid on every task, and an unexecutable line can score
   below silence.
3. **The gate cannot validate the set.** `design-ship-gate` scores the *absence
   of eight defects*; it has never scored whether a page is good, and the arm
   that scored lowest on it (4/8) is the arm whose pages the developer prefers
   to look at (#366). So "the principles work" cannot be concluded from a gate
   score — the gate answers a narrower question than the one being asked.
4. **A principle proven on a frontier agent is not proven on a 27B model.** The
   "on Qwen" column above is mostly `no`. Nineteen traps were collected from
   sessions with far more capacity to notice; a rule that a frontier model
   applies from a sentence may need a command here.

**Contested, and therefore not used above.** A search snippet and a direct fetch
of *On the Limits of Innate Planning in Large Language Models*
([arXiv:2511.21591](https://arxiv.org/pdf/2511.21591)) **disagree on the
direction** of its scaling claim — the snippet says the
instruction-following-to-planning ratio approaches unity as models grow, the
fetched summary says instruction-following grows *faster* than planning. Both
readings agree only that small models show a gap between following an
instruction and planning, and that they adhere to a single dominant strategy.
**The scaling claim is not used in this document.** Related and also unverified
here: *Small Models Struggle to Learn from Strong Reasoners*
([arXiv:2502.12143](https://arxiv.org/html/2502.12143v1)) reports models ≤ 3B do
not consistently benefit from long chain-of-thought — a size an order of
magnitude below ours, so it bounds nothing about Qwen3.8-27B.

---

## 5. The distillation hypothesis — and the control this project has never run

**The developer, 2026-09-07:** *"การ Design ของ Qwen นั้นคล้าย Claude มาก … ให้ตั้ง
สมมุติฐานว่า Skill ที่ใช้กับ Claude ได้ก็สามารถใช้กับ Qwen ได้เช่นกัน"* and
*"ผมมีสมมุติฐานว่า Qwen ได้มาจาก Knowledge Distillation จาก Claude"*.

Adopted as the working hypothesis. What follows is what our data already says
about it, and the one thing that would settle it.

### The evidence that exists, and it points the developer's way

**MEASURED HERE.** The design skills on this machine name exactly one font
family — `Inter`, four times, in `design-setup`. They do **not** name Fraunces,
Space Grotesk, Space Mono, Newsreader, Archivo, IBM Plex, Anuphan or JetBrains
Mono. Every one of those came from the model, not from the skill text.

And the sharpest row is the one with nothing loaded: **`A-noskill-r1`, a fresh
`CLAUDE_CONFIG_DIR` with zero skills, produced `Fraunces, Inter`.** Fraunces
paired with a geometric sans is a signature of Claude's own design output, and
Qwen reached for it with no skill in context at all. The same pairing recurs in
`designonly` r1 and r3, `gateonly` r1 and `both` r1, and the developer's own
26-page folder runs `Anuphan, Inter, JetBrains Mono, Space Grotesk` — none of
which any skill names.

**The honest limit on that evidence:** these are also among the most-used
families on Google Fonts, so a shared web-design training corpus explains the
overlap without requiring distillation from Claude specifically. It is real
signal for a **shared prior**; it does not by itself identify the source.

### The structural finding: there is no control arm anywhere

**MEASURED HERE.** All 40 page rows of the 2026-09-05 bench carry the same
artifact string, `EXL3 SC 4.0bpw H5 @ 262,144`. `claude` appears throughout
`quality-bench.py` because **Claude Code is the harness**, pointed at our own
server by `ANTHROPIC_BASE_URL` (line 197) — not because any cell ran a frontier
model.

So the founding premise of the whole `qwen38` family — *"what its parameter
count cannot supply is the thinking-of that a frontier model gets from scale"*
(`qwen38-skill-style`) — **has never been measured against a model that has
that scale.** Every claim in this document's "on Qwen" column, and every rule in
the family justified as *"this model needs it written"*, is an assumption with
no control behind it. Some of the incidents used to justify those rules —
losing a report line when two skills collide, following a global `gh` rule into
a dead end — are things a frontier model in an unattended run does too, and
nothing here has checked.

**This is the strongest argument for the developer's hypothesis, and it comes
from our own data rather than from the model's lineage:** the null hypothesis
was never tested, so the default should be the simpler one — the skill works, or
it does not, and the model is not presumed to be the variable.

### What adopting it changes

- **Default to the Claude skill, unmodified.** Write a `qwen38-*` variant only
  where a Qwen-vs-control difference has been *measured*. Today there are zero
  such measurements.
- **Re-read the five router rules as unattended-run rules, not small-model
  rules.** Write the brief table · run the checks and paste the output · never
  install or search · take the open choice from the subject · stop at the
  report. Nothing in those five names a capacity limit; all five describe a run
  with nobody watching.
- **`qwen38-claude-code` keeps its justification either way.** A JSON schema
  says what a tool accepts and never what it is for; that gap is in the harness,
  not in the model, so covering it helps any model reading the same schemas.

### The experiment that settles it, and it is cheap

`quality-bench.py` selects its endpoint from a per-cell dict. **A control cell is
a new entry in that dict** — a frontier model and the real API base — not a new
code path, and **it does not need the GPU**. Run `noskill`, `designonly` and
`gateonly` on it, three reps, against the existing 2026-09-05 baseline.

| outcome | what it means |
|---|---|
| control also scores `designonly` < `noskill` | the effect is the **gate and the skills**, not the model; the family's premise is refuted and most `qwen38-*` rules become general |
| control scores `designonly` ≥ `noskill` | there is a real model difference, and for the first time it has a number |

**UNMEASURED. Not run.** It is the cheapest decisive experiment named anywhere
in this document and it is the one that has never been done.

---

## 6. Against the router as it stands

`using-qwen38` on `main` carries five rules. Mapping the catalogue onto them:

| catalogue item | in the router? |
|---|---|
| enumerate the unsaid, as a table | **yes** — rule 1 |
| run the checks and paste the output | **yes** — rule 2 |
| never install / search / spawn | **yes** — rule 3 |
| take open choices from the subject | **yes** — rule 4 (new, #366) |
| stop at the report | **yes** — rule 5 |
| check the brief for flaws with a command | **yes**, by delegation — every row starts with `qwen38-think` |
| **ask what a passing check would have printed had it failed** | **no** |
| **a report says what did not run, not only what did** | **no** |
| assert on behaviour not on file shape | n/a — authoring-side, lives in `qwen38-skill-style` |
| one variable at a time · dependency order · retry classification | **no**, and these are the least Qwen-relevant of the set |

**The two real gaps are the first two "no" rows, and they are the two halves of
the spine.** Rule 2 says *run the checks and paste the output*; nothing says the
output has to be able to look different. A model that pastes `invisible: 0` from
a detector that examined nothing has satisfied rule 2 completely — which is
precisely what `gate-dark.js` did before #359, and what check 3 still does in
#365.

**UNMEASURED** whether stating either in the router changes anything. Both are
one bench round to test, and both cost tokens on every task, so neither should
be added on the strength of this document alone. That is a decision, and it is
the developer's.

---

## 7. What would settle the open ones

Cheapest first. Each is one arm of `quality-bench.py` against the existing
2026-09-05 baseline, so all of them are comparable without a re-baseline.

| question | arm | cost |
|---|---|---|
| Does "say what a failing check would have printed" change anything? | router + one sentence in rule 2 | 1 arm × 3 reps |
| Does "report what did not run" change anything? | router + one line in the report block | 1 arm × 3 reps |
| Does the router's length itself cost score? | the current router vs a two-rule cut | 2 arms × 3 reps |
| Is rule 4 (open choices) doing anything at all? | router with and without rule 4, judged by the developer rather than the gate — **the gate cannot answer it** (§4.3) | 2 arms × 3 reps + a human read |

All four need the GPU and the EXL3 server, which is currently stopped.

---

## Sources

- [Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*](https://arxiv.org/abs/2310.01798) — abstract verified by fetch
- [Anthropic, *Building Effective AI Agents*](https://www.anthropic.com/engineering/building-effective-agents)
- [Anthropic, *Effective context engineering for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [*On the Limits of Innate Planning in Large Language Models*](https://arxiv.org/pdf/2511.21591) — **contested between two reads, not used**
- [*Small Models Struggle to Learn from Strong Reasoners*](https://arxiv.org/html/2502.12143v1) — ≤ 3B, bounds nothing here
- [ICML 2026 workshop, *Failure Modes in Agentic AI*](https://icml.cc/virtual/2026/workshop/54094) — error cascades, brittle tool use under interface shift, weak recovery; listing only, not read
- Local: `docs/agents/traps.md` · `docs/reports/CORRECTIONS.md` §§30, 33–39 ·
  `docs/results/11-quality-bench-2026-09-05.md` · `skills/qwen38/qwen38-skill-style/SKILL.md`
  (xeno-skills) · issues #359, #365, #366
