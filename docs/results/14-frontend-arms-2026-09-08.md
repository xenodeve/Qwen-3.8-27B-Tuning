# 14 — the design arms on the gateway, 2026-09-08

Artifact: **cell D, `vllm/Qwen/Qwen3.8-27B-FP8`**, ctx 131,072. **No GPU.** The page brief is
the developer's own, in Thai: *"สร้าง Landing Page แนะนำ Google โดยใช้ Editorial Minimalism ×
Modern Swiss × Liquid Glass × Visible Grid | ทำเป็น HTML ไฟล์เดียว"*. Nine cells: **five reps of `designonly`, two each of `gateonly` and `both`**, order rotated. Raw pages: `qwen38-tuning/results/quality-2026-09-08-d/D-<arm>-r<n>/page.html`.

**These are rankings, not rates.** The third round of `gateonly`/`both` was cut because a page
cell on this endpoint runs 4–36 minutes and would not have finished inside the window.

## What had to be fixed first

`brief_suffix` — the fix that made the gateway actually load skills (results 12 §4d) — was
wired into the **code** path only. Page arms build their brief from `ARMS[arm]["brief"]`,
which embeds slash tokens, so a frontend run on D would have repeated the zero-`Skill`-calls
fault the code path had already been fixed for. `page_brief()` now gives page arms the same
treatment; `skills=None` (the developer's whole `~/.claude`) is left alone. Four tests pin it.

Verified: `designonly` called `Skill` 7 and 4 times, `both` 8 times — it was 0 before.

## Result

| cell | gate | fonts | dark | **Thai chars** | 390 px overflow | output tok | wall |
|---|---|---:|---:|---:|---:|---:|---:|
| `designonly` r1 | **5/8** | 3 | 0 | **0** | 83 px | 42,256 | 9.1 min |
| `designonly` r2 | 7/8 | 2 | 0 | **0** | 0 | 21,104 | 3.8 min |
| `designonly` r3 | **NO PAGE** | — | — | — | — | 3,218 | 19.0 min |
| `designonly` r4 | 6/8 | 3 | 0 | **0** | 0 | 16,571 | 5.8 min |
| `designonly` r5 | **NO PAGE** | — | — | — | — | — | 22.7 min |
| `gateonly` r1 | **8/8** | 2 | 1 | 1671 | 0 | 26,106 | 25.6 min |
| `gateonly` r2 | **8/8** | 2 | 7 | 1307 | 0 | 127,014 | 22.2 min |
| `both` r1 | **8/8** | 2 | 1 | 1347 | 0 | 60,021 | 13.5 min |
| `both` r2 | **8/8** | 2 | 10 | 2037 | 0 | 49,483 | 35.5 min |

`designonly` is **0 Thai in 3 of 3 pages, 0 dark mode in 3 of 3, three fonts in 2 of 3** — and
in **2 of its 5 runs it produced no file at all**. `gateonly` and `both` are 8/8 in every run.

### And in two runs of five, `designonly` shipped nothing

`D-designonly-r3` ran **19 minutes**, called `Skill` seven times, wrote 3,218 output tokens,
exited **rc = 0** — and never called `Write` or `Edit`. Its answer ends mid-sentence, on the
words *"สร้างไฟล์เดียว:"* — "create a single file:" — and then stops. `r5` did the same
thing 22.7 minutes later, so this is **not** a one-off: the arm spends its budget describing
the design and never builds it. Neither `gateonly` nor `both` did this in any run.

`void_reason` does not mark it void, correctly: the model did produce output, so this is not
a dead run. The runner records `gate: null`, `pages: []` and prints **"NO page"**, which is
the honest outcome. It is called out here because a cell with a null gate is exactly the row
an average silently drops, and dropping it would turn a total failure into a missing sample.

### The finding that matters most is not the gate score

**`designonly` wrote the page in English in every run that produced one** — 0 Thai characters
in 3 of 3, against 1,300–2,000 in every other arm.

**Researched before drawing the obvious conclusion, and the obvious conclusion is wrong.**
Grepping the six skills of the design suite for any mention of language returns **nothing** —
the rule genuinely is not there, and `design-ship-gate` is the only skill that has it (its
brief-coverage row: *"the brief's language | the page copy is in it (2 / 9 answered a Thai
brief in English)"*). But that does **not** mean `using-design` produces English pages. The
developer's own folder of 26 pages, built with `using-design` and no gate, is **21 of 26 with
Thai in it** (0 in five).

So the behaviour is **unguarded, not absent**: the model gets it right most of the time
without being told, and got it wrong in every run here that produced a page. That is the more uncomfortable
finding of the two — an unguarded behaviour that usually works is exactly the kind that fails
without anything noticing.

It also shipped **no dark mode in any of the three**, three fonts in two of them, and 83 px
of horizontal overflow in the first.

**This is the same shape the coding track found on the same day** ([results 13](13-skill-vs-placebo-2026-09-08.md)):
a rule stated as prose does not transfer, and the same rule as a command with a pass
condition does. The two-font cap lives in `design-rules`, which `designonly` loads — and
`designonly` is the arm that broke it.

## What this does not settle — and it is the important part

**None of these numbers is about whether the page is good to look at.** The gate scores the
absence of eight defects; it has never scored beauty, and the arm the developer said looks
best (`using-design` alone, `docs/reports/CORRECTIONS.md` context and issue #366) is the arm
that scores worst here. **Both statements can be true at once**, and this run does nothing to
reconcile them: an English page with no dark mode can still be the prettiest object on the
screen.

So this document ranks the arms on **defects**, and the seven pages that exist are on disk precisely so
the ranking on **beauty** can be made by the person who has it.

## Recommendation, stated as a trade

For a Thai page brief on this artifact, **`both`** is the arm to use: it is the only one that
gets 8/8 **and** carries the design knowledge. `gateonly` matches it on defects and knows no
design; `designonly` knows design and, on this artifact, ships an English page — or no page at all,
two times in five.

The cost is real — `both` runs 13.5–35.5 min against `designonly`'s 3.8–9.1, and one
`gateonly` cell spent 127,014 output tokens. **UNMEASURED:** whether `both`'s pages are
better *to look at* than `designonly`'s. That needs the developer, not the gate.

## Open

- More reps of `gateonly` and `both`, and the same arms on **cell A** (4.0bpw), when the GPU
  is free.
- The beauty question itself. Nothing in this repo measures it, and #366 is the record of
  what happened the last time a gate score was read as if it did.
