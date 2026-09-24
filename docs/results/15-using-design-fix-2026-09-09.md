# 15 — three rules added to `using-design`, and what they closed (2026-09-09)

**This is the first shipped skill change of this loop.** Everything tried on 2026-09-07/08
was measured and rejected; this one was measured and kept.

Artifact: **cell D, `vllm/Qwen/Qwen3.8-27B-FP8`**, no GPU. Arm `designonly` (the design suite
alone, no gate), the developer's own Thai page brief. Thirteen cells over three configurations.

## Why these three, and why they were not a guess

[Results 14](14-frontend-arms-2026-09-08.md) measured the design suite used alone and found
two failures it could not explain away, then a grep found the cause of one of them:

- **2 of 5 runs produced no file at all** — 19 and 22.7 minutes, `rc = 0`, the answer ending
  mid-sentence on *"สร้างไฟล์เดียว:"*, `Write` never called.
- **0 Thai characters in 3 of 3 pages** for a Thai brief. Grepping all six skills of the
  suite for any mention of language returns **nothing**; the rule lived only in
  `design-ship-gate`.
- **`prefers-color-scheme` in 0 of 7 pages** across two days. Here the grep found the
  opposite of a gap: `design-rules` §4 covers dark mode *well* — surface elevation, borders,
  saturation — but all of it is **how to style a dark UI once you have chosen one**. Nothing
  says the page owes **both modes**. (The same confusion is open on the gate itself,
  `xeno-skills` #365.)

So: two genuinely absent rules, one present-but-misread. All three added to `using-design` as
a short block before the suite's own content, each with a one-line check and the run behind it.

## Result

| | page produced | Thai present | `prefers-color-scheme` | gate scores |
|---|---:|---:|---:|---|
| **before** — no rules | 3/5 | **0/3** | **0/3** | 5/8, 7/8, 6/8 |
| **after 1** — file-first + language | **4/4** | **4/4** | 0/4 | 4/8, 6/8, 5/8, 7/8 |
| **after 2** — + light *and* dark | **4/4** | 3/4 | **3/4** | **8/8, 8/8, 8/8**, 6/8 |

- **Thai, pooled over both "after" sets: 7/8 against 0/3 before — p = 0.024.**
- **`designonly` reached 8/8 three times. It had never scored above 7/8 in 7 earlier runs —
  0/7 against 3/4, p = 0.024.**
- Dark mode alone, 3/4 against 0/4 — **p = 0.071**. Right direction, not significant at n = 4.
- No-file failures went 2/5 → 0/8, but 8/8 against 3/5 is **p = 0.13**. Underpowered.

**Rule 1 was verified by mechanism, not only by outcome.** In `after 1` r3 the stream shows
`Write` as the **seventh** tool call, straight after the skill loads and before the long
elaboration — which is exactly what the rule asks for, and the opposite of the two runs that
planned in prose and shipped nothing.

## The honest limit — corrected 2026-09-09, and the correction matters

**First written here as "one run in four ignored both rules". That was wrong.** `after 2` r4
came back with 0 Thai, no `prefers-color-scheme` and 6/8 — and its transcript shows
**`Skill` called zero times**: two turns, one `Write`, finished in 2.7 minutes. It never read
the skill. Nothing was ignored.

Splitting all thirteen `designonly` cells by whether the skill actually loaded:

| | cells | Thai present | `prefers-color-scheme` |
|---|---:|---:|---:|
| skill loaded, **before** the rules | 5 | 0/3 pages | 0/3 |
| skill loaded, **after 1** | 4 | **4/4** | 0/4 |
| skill loaded, **after 2** | 3 | **3/3** | **3/3** |
| **skill not loaded** | **1** | 0 | 0 |

**Every run that read the rules followed them — 7 of 7 on language, 3 of 3 on dark mode.**
The single failure is the single run that never loaded the skill.

So the remaining defect is **not compliance, it is loading**: 12 of 13 runs called `Skill`,
one did not, and that one produced a number indistinguishable from a compliance failure. A
cell whose arm never loaded its skills is not a measurement of that arm at all — the same
class of fault as scoring a run that never reached the model, and it is guarded the same way
below.

n = 4 per configuration is still a ranking rather than a rate.

The gate score also **dropped** between *before* and *after 1* (mean 6.0 → 5.5) before rising
to 7.5. That is expected rather than alarming — the `after 1` pages contain Thai copy that
was not there before, so there is more page to find defects in — but it is stated because the
middle row is the one a summary would quietly drop.

## What is shipped, and how to undo it

The three rules are live in `~/.claude/skills/using-design/SKILL.md` (5,567 → 7,763 bytes).
The exact pre-edit file and the inserted block are kept beside the data:

```
qwen38-tuning/results/quality-2026-09-09-fix/using-design.SKILL.md.before
qwen38-tuning/results/quality-2026-09-09-fix/inserted-block.md
```

**Not yet in the `xeno-skills` repo.** That needs an issue and a PR, and `gh` is not installed
on this machine — so the change lives in the installed copy only, and the library still has
the old file.

## Open

- The same three rules on **cell A** (4.0bpw) when the GPU is free — this was measured on the
  FP8 gateway only.
- More reps: dark mode and the no-file fix are both directionally clear and underpowered.
- `xeno-skills` #365 — the gate passes a dark-only page, which rule 3 here works around
  rather than fixes.
