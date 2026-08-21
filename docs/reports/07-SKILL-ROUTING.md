# Skill Routing — Which Discipline Governed Which Decision

> **Date:** 2026-08-19 UTC+7
> **Why this exists:** the skills were invoked, but the evidence lived only in the
> conversation. A reader of the reports could not tell which discipline produced
> which decision, or where one was skipped. This is the durable record.

---

## 1. The route, and why

`ask-xeno` is the library entry point. Its table has four rows:

| you are… | enter at |
|---|---|
| working in a T4 repo | `using-t4` |
| handing work to another agent | `using-clink` |
| designing a web UI | `using-design` |
| **writing code, in any repo** | **`karpathy-guidelines`** |

**This project routes to `karpathy-guidelines`**, because `C:\AI` is not a T4 repo.
That was established by check, not assumption:

```text
absent  .git          (fatal: not a git repository)
absent  CLAUDE.md · AGENTS.md · CONTEXT.md
absent  docs/agents · docs/adr · docs/decisions
```

Every T4 mechanism therefore has nothing to operate on — no tracker for the
PRD → issues → PR gate, no issue body for the bilingual rule, no vault for
`t4-agent-memory`. `using-t4`'s own "When NOT to use" covers this case. The gap
that it names the case without giving a route is filed as
[xeno-skills#249](https://github.com/xenodeve/xeno-skills/issues/249).

What still applies from `using-t4` regardless of repo:

- **Session protocol step 1** — load `karpathy-guidelines` once, so every edit is
  surgical and goal-verified.
- **Session protocol step 4** — report each rule that did not hold as a
  `skill-feedback` issue. §4 below is that report.
- **The non-negotiables** — evidence before verdict, root cause before fix,
  skipping a rule requires a checkable proof.

---

## 2. Where each discipline shows up in the work

| discipline | rule | where it decided something | evidence |
|---|---|---|---|
| `karpathy-guidelines` §4 | define success criteria, loop until verified | every sweep states its hypothesis and its verify step before running | `EXPERIMENTS.md` E0–E13, each entry opens with a hypothesis and closes with a result that confirms or rejects it |
| `karpathy-guidelines` §2 | simplest thing that works | rejected `-b 512 -ub 128` despite the best raw decode — it cost 33 % of prompt processing for 2 % of decode | report 01 §5 |
| `karpathy-guidelines` §3 | clean up only your own mess | the stderr fix touched only the two scripts the agent had broken; the pre-existing Vulkan install was noted, not removed | report 05 §5 |
| `tdd` | red before green, one slice at a time | `bench/harness.py` — four functions, each red first | `ModuleNotFoundError: No module named 'harness'`, then `ImportError: cannot import name 'load_jsonl'` |
| `tdd` | test at seams, not internals | the four pure functions are tested; the server-driving code is deliberately not | `bench/tests/test_harness.py` header |
| evidence-before-verdict | name the command and output, or label it a hypothesis | "+11.6 % from `-n-min 2`" was reported, then **retracted** when a fresh control gave −0.8 % | report 04 §1 |
| root-cause-before-fix | reproduce, trace, falsify, then propose | the 13.6 % drift floor was measured (6 restarts, unchanged config) rather than assumed after the first surprising result | report 04 §0 |
| skipping needs proof | state a checkable fact, not a judgment | the non-T4 finding is a directory listing a reviewer can re-run, not "this doesn't feel like a T4 repo" | §1 above |

---

## 3. Decisions that a discipline changed

Not routing theatre — these are places where following the skill produced a
different answer than the obvious one.

**The 13.6 % drift floor exists because of "evidence before verdict."** Three
runtime flags had been reported as +9.3 %, +6.9 % and +3.8 %, summed to
"+19 % cumulative". The rule forced a re-test against a fresh control, which
reversed a whole sweep and produced the paired design that gives the honest
+6.6 – 9.6 %. Without it the project would have shipped a number twice too large.

**Q8 KV was nearly adopted on false evidence.** The greedy-hash check reported it
identical to F16. "Evidence before verdict" meant asking what the probe actually
exercised — a 4-token prompt, over the very cache Q8 quantizes. Re-run at 46 557
tokens the two share one character of 778. That triggered building the deep corpus,
which is what eventually justified Q8 properly.

**`tdd` caught two instrument bugs before they produced a verdict.** A duplicated
`Handler0017`, and a size assertion with only a lower bound that let a 112K-token
corpus pass and then fail every request with HTTP 400 — 0/18 in four seconds. Read
naively that says "the model cannot do deep context at all."

**A stop was retracted because of root-cause-before-fix.** A run was halted on a
"decode collapse" that turned out to be `max_tokens` variance; the confirming probe
was invalid for the same reason as the Q8 case. Recorded in report 03 §7 and
in-band in the results file.

---

## 4. Rules that did not hold — filed upstream

`using-t4` session protocol step 4. All three are on `xenodeve/xeno-skills`, one
issue per rule, searched `--state all` first, `--repo` passed explicitly.

| rule | outcome | where |
|---|---|---|
| `tdd` — red before green | **comment**, not a new issue: [#240](https://github.com/xenodeve/xeno-skills/issues/240) already tracks this rule. Red-first held for `harness.py` and was **skipped for both benchmark corpora**, which were written first and tested after | [#240 comment](https://github.com/xenodeve/xeno-skills/issues/240#issuecomment-5334641090) |
| `using-t4` — "When NOT to use" names the non-T4 case but gives no route, and session protocol steps 2–3 have no referent | new issue | [#249](https://github.com/xenodeve/xeno-skills/issues/249) |
| `tdd` — "No test is written at an unconfirmed seam" has no path for an unattended run; an autonomous goal directive forced a choice between stalling and improvising | new issue | [#250](https://github.com/xenodeve/xeno-skills/issues/250) |

The red-first skip is the one worth dwelling on: it was skipped on the **measuring
instrument**, which is worse than skipping it on ordinary code, because a broken
instrument returns a number rather than a failure.

---

## 5. What was deliberately not invoked

Naming these so a reader does not read absence as oversight.

- **`using-clink`** — no work was delegated to another agent. Every measurement ran
  locally against the machine under test, which is the only place the answer exists.
- **`using-design`** — one HTML artifact was produced and it went through
  `artifact-design`, the correct route for that. No product UI was built.
- **`t4-dev-workflow`, `t4-agent-memory`, `t4-engineering-records`, `t4-afk`,
  `t4-bro`** — all require T4 repo structure (§1). `t4-bro`'s register was followed
  in the chat regardless, since it costs nothing and the developer reads Thai.
- **`security-review`** — no trust boundary was touched. The server binds
  `127.0.0.1`, no credentials or user data are involved.
