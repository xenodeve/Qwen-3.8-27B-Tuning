# 13 — `qwen38-code-gate` against a placebo, 2026-09-08

Artifact: **cell D, `vllm/Qwen/Qwen3.8-27B-FP8` on the 9arm gateway**, ctx 131,072. **No GPU
was used.** Two fixtures — `code2` (the ledger task, 15 reps per arm) and `code1` (the
inventory task, 10 reps per arm). **100 cells across the four arms, 0 voided**, plus 14 more
for the dismantling arm in §"Which half of the skill does the work". Arm order rotated every
round so a position effect cannot favour one arm. Every figure below was re-checked against
the raw `summary.json` files after the prose was written. Raw data:
`qwen38-tuning/results/quality-2026-09-08-d/`.

## The four arms

| arm | what the session carries |
|---|---|
| `noskill` | no skills at all |
| **`placebo`** | `qwen38-placebo` — a skill of the same shape and length that says nothing about tests, ordering, checks or reports |
| `codegate` | `qwen38-code-gate` |
| `family` | `using-qwen38` + `qwen38-think` + `qwen38-code-gate` + `karpathy-guidelines` |

**The placebo is the point of this run.** On D the brief must name the Skill tool (results
12 §4d), so every skill arm's brief ends *"ใช้ Skill tool โหลด … ก่อนเริ่มงาน แล้วทำตามที่เขียนไว้"* —
which is itself an instruction to follow written guidance. Without a content-free skill
carrying **exactly that same wording**, any effect could be the sentence rather than the
skill.

## Result

| arm | n | **tested before touching source** | gate 5/5 | hidden tests pass | mean output tok | mean wall | `Skill` calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| `noskill` | 15 | **0/15 (0 %)** | 3 | 14/15 | 11,934 | 1.4 min | 0.0 |
| `placebo` | 15 | **0/15 (0 %)** | 0 | 15/15 | 10,218 | 1.4 min | 1.0 |
| `codegate` | 15 | **15/15 (100 %)** | 15 | 15/15 | 28,439 | 4.0 min | 1.0 |
| `family` | 15 | **14/15 (93 %)** | 15 | 15/15 | 22,007 | 2.7 min | 4.1 |

Fisher one-sided:

- `codegate` vs `placebo` — **p = 6.4 × 10⁻⁹**
- `family` vs `placebo` — **p = 1.0 × 10⁻⁷**
- **`placebo` vs `noskill` — p = 1.00.** The control check: a skill that says nothing does
  nothing, though it was loaded (`Skill` 1.0 per run) and its brief carried the same
  "follow what is written" sentence.

That last row is what makes the first two believable. The effect is the skill's **content**,
not the act of loading a skill and not the instruction wording.

## It replicates on a second fixture

The same four arms on **`code1`** (a different fixture: a planted bug plus a small feature
in an inventory store), ten reps each, order rotated.

**Tested before touching source, both fixtures, 100 cells:**

| arm | `code1` | `code2` | **pooled** |
|---|---:|---:|---:|
| `noskill` | 0/10 | 0/15 | **0/25** |
| `placebo` | 0/10 | 0/15 | **0/25** |
| `codegate` | 10/10 | 15/15 | **25/25** |
| `family` | 10/10 | 14/15 | **24/25** |

`codegate` vs `placebo` **p = 7.9 × 10⁻¹⁵**; `family` vs `placebo` **p = 2.1 × 10⁻¹³**;
**`placebo` vs `noskill` p = 1.00**.

The separation is complete on both fixtures: every run that carried the skill tested first,
and no run without it ever did — 49 of 50 against 0 of 50.

## Which half of the skill does the work — the table, or the commands?

The full skill costs 2.4× the tokens of no skill, so it is worth knowing what earns that.
`qwen38-code-gate-min` is **section 0 alone** — the brief table, including its row
*"a test that fails before the change and passes after"* — with **checks 1–6 removed**.
2,138 bytes against 4,786.

On `code1`:

| arm | n | tested before touching source | gate 5/5 | mean output tok |
|---|---:|---:|---:|---:|
| `noskill` | 10 | 0/10 | 0 | 3,814 |
| `placebo` | 10 | 0/10 | 2 | 4,341 |
| **`codegatemin`** — table only | 10 | **1/10** | 6 | 6,732 |
| `codegate` — full | 10 | **10/10** | 10 | 11,754 |
| `family` | 10 | **10/10** | 9 | 10,847 |

**The table alone does almost nothing for the ordering.** `min` vs `placebo` **p = 0.50**
— not distinguishable from a skill that says nothing; full vs `min` **p = 6 × 10⁻⁵**.
It replicates on `code2` as well (**0/4** there): pooled over both fixtures the table-only
arm is **1/14** against the placebo's 0/25 (**p = 0.36**) and the full skill's 25/25
(**p = 1.7 × 10⁻⁹**). The
deliverable was named — the row *"a test that fails before the change and passes after"* is
in the table the model filled in — and in 9 runs of 10 it still wrote the source first.

**So enumeration is not what produces the behaviour; the executable checks are.** That is
`qwen38-skill-style` rule 2 ("a command and a pass condition per rule") doing the work, and
it bounds rule 1 ("enumerate; do not inspire"): naming a deliverable in a table does get it
*delivered* — `min` reaches **6/10** on the gate against the placebo's 2/10 and `noskill`'s
0/10, so the table is not inert — but it does not change the **order** the work is done in.
Only a check the model must run does that.

**Consequence for the skill: the 2.4× token cost is not trimmable by dropping the commands.**
They are the active ingredient.

### Which command? Two more ablations

`no6` is the full skill **minus check 6** ("RED before GREEN"). `t6` is **the table plus
check 6 only**, checks 1–5 removed. Ten reps each on `code1`, order alternated:

| arm | size | tested before touching source | gate 5/5 | mean output tok |
|---|---:|---:|---:|---:|
| `placebo` | — | 0/10 | 2 | 4,341 |
| table only | 2,138 B | 1/10 | 6 | 6,732 |
| **minus check 6** | 4,204 B | **7/10** | 9 | 10,299 |
| **table + check 6 only** | 2,850 B | **10/10** | 7 | 9,244 |
| full | 4,683 B | 10/10 | 10 | 11,754 |

Two things follow, and they point in different directions:

- **`t6` reproduces the full skill's ordering exactly** — 10/10 vs 10/10, p = 1.00 — at
  **61 % of the bytes and 79 % of the output tokens**. If ordering is all you want, the
  table plus one check buys it.
- **But check 6 is not the whole story.** Removing it drops ordering to 7/10, not to the
  table's 1/10 — so checks 1–5 push toward testing on their own, and **`codegate` vs `no6`
  is p = 0.11, not significant at n = 10.** "Check 6 is required" is *not* established here;
  what is established is that the table alone is not enough (1/10) and that either half of
  the command set gets most of the way.

**`t6` is not shipped on this evidence.** Its gate score is lower (7 of 10 cells at 5/5
against the full skill's 10), because checks 1–5 are what the other four gate items measure
— dropping them buys cheapness by giving up the scope, placeholder, secret and lint checks.
That is a trade for the developer to make, not a free win.

## What the skill does not buy

**Correctness.** Hidden tests pass 14/15, 15/15, 15/15, 15/15 on `code2` and 5/5 in every
arm on `code1` — every arm is at the ceiling, and the one failure is in `noskill`. Nothing here shows the skill making
the code more correct; it shows it changing **how the work is done**.

**It is not free.** `codegate` costs **2.4× the output tokens and 2.9× the wall clock** of
`noskill` (28,439 vs 11,934; 4.0 vs 1.4 min). `family` is cheaper than `codegate` alone
(22,007 tok, 2.7 min) at 93 % vs 100 % — four skills loaded, less spent, one miss.

## Against the EXL3 result of the day before

Results 12 measured the same skill on **EXL3 4.0bpw** at **6/15 (40 %)**, with the skill
verifiably loaded in all 15 runs (`Skill` called every time). Here it is 15/15.

**Do not read that as a quantisation finding.** The two are not the same arm: D's brief
names the Skill tool and adds *"แล้วทำตามที่เขียนไว้"*, A's carries only the slash token.
Separating artifact from wording needs an EXL3 run with D's suffix, which needs the GPU and
**has not been done**. What the placebo settles is only that, *within D*, the wording alone
does not produce the behaviour.

## What this changes

- **`qwen38-code-gate` is doing what it was written to do**, on this artifact, with a
  control. That is the first result in this programme with a placebo behind it.
- The two variants tried on 2026-09-07 — `-v2` (a step 0.5) and `-v3` (a reorder line) —
  remain **not shipped**; results 12 measured both as worse than the untouched file.
- Open, and cheap when the GPU is next free: **the same four arms on cell A**, to find out
  whether the 40 % / 100 % gap is the artifact or the brief's wording.

## Guards

`bench/tests/test_quality_bench_remote_cell.py` — 31 tests. Four pin `brief_suffix` (a
cell marked `skills_explicit` names the tool; a `noskill` arm is left alone), six pin
`tdd_order`, six pin the test-run detector, six pin the void rules, the rest the remote
cell. All were red before their fix.
