# Post-mortem — three numbers were combined into a claim nobody could check

**Date:** 2026-08-21 · **Found by:** the developer, asking whether it was
actually looping · **Severity:** the claim reached a research brief written for
an external reader

---

## What was claimed

That Qwen3.8-27B at 2.16 bits per weight **loops inside its reasoning block
until the token budget runs out and emits nothing.**

It appeared in three places before anyone questioned it:

| document | wording |
|---|---|
| `docs/plans/04-REVISED-PLAN-2026-08-21.md` §0 | *"the model loops inside the reasoning block until the token budget runs out and never emits a fence"* |
| `docs/results/06-prompt-and-quality.md` | *"the model loops inside the reasoning block until the token budget runs out"* |
| `docs/plans/05-RESEARCH-BRIEF-2026-08-21.md` §2 | the whole section, headed *"the model reasons until the budget is gone and emits nothing"*, with five questions built on it |

The brief was written to be sent to an external researcher.

## What is actually true

`damerau` — one of the three corpus tasks that failed under OpenCode — sent
straight to the server with a 16,384-token budget:

```text
  reasoning        6,899 characters
  line repetition  0.00 %      not one line recurs
  finish_reason    stop        it ended on its own, nowhere near the cap
  content          643 characters, and the code PASSES the hidden tests
  wall             62.6 s
```

The trace is good work. It catches an ambiguity in the prompt unprompted —
*"Normally in Damerau-Levenshtein a transposition costs 2 … but here it says
costs 1"* — tests that reading against a worked example (`a[i-1]==b[j-2]?
'b'=='b' ✓`), concludes the task means OSA distance, and only then writes code.

**The model thinks for a long time and finishes.** That is this model's
documented normal mode: a public review of it measures xHigh at ~15 minutes of
thinking and medium at ~3 minutes for *"90 % of the result"*.

The same task through OpenCode took **247.6 s and produced nothing**, against
62.6 s direct. **The failure is in the agent loop, not the reasoning.**

## Root cause

Three signals were combined into a mechanism, and none of them contains the
text the claim was about:

1. `reasoning_chars` up to **16,341** — `results/protocol-budget.jsonl`
2. tool-call round trips completing only **10/16** — same file
3. three corpus tasks producing **no output** in 190–248 s —
   `results/opencode-corpus.jsonl`

Each is real. "Long reasoning" plus "does not finish" plus "no output" reads as
looping, and the reading was never checked, because **the text was not there to
check.**

`bench/protocol_gate.py:207` recorded the *length* of the reasoning and kept
**400 characters** of it:

```python
row["reasoning_excerpt"] = ((msg.get("reasoning_content") or "")[:400] or None)
```

The 400 characters it did keep were coherent, on-task reasoning. Nobody opened
them.

**The deeper fault is the shape of the inference.** Three measurements of
*symptoms* were used to assert a *mechanism*, in a project whose own first rule
is `No verdict before evidence` and which by that morning already carried
eleven entries in `CORRECTIONS.md`. The register was written and then not
applied to new writing.

## Fix — validated

**1. The probe keeps the whole trace and answers the question with a number.**
`bench/protocol_gate.py` now writes the full reasoning to
`logs/reasoning/<label>-trial<n>.txt` and records
`reasoning_repetition_pct` on the row. Looping stops being an inference from
length: a trace that recurs scores high, one that progresses scores 0.

**2. The measurement has a name that matches its use.**
`harness.filler_repetition_pct` was written to check a benchmark prompt; its
more valuable use is a reasoning trace. Canonical name is now
`harness.line_repetition_pct`, with the original kept as an alias rather than a
second copy.

**3. Three tests, named after this incident** —
`test_reasoning_that_never_repeats_scores_zero` carries the real trace's
structure, `test_a_trace_stuck_on_one_thought_scores_high` carries what looping
would actually look like, and `test_line_repetition_is_the_name_the_reasoning_check_uses`
pins the alias. Suite **111**, up from 108, all passing.

**4. All three documents retracted in place**, each carrying what the evidence
does and does not support.

## What was lost, and what was gained

The retraction cost about forty minutes and one measurement. Had the brief gone
out unchallenged, an external researcher would have spent their effort on
suppressing an aberration that does not exist, and their answer would have come
back sound and useless.

Two results came out of the same 63-second measurement that settled it:

- **n-gram acceptance on real code is 16.8 %** (188 of 1,121 drafts), against
  **99–100 %** on this project's synthetic benchmark prompt. That answers an
  open question filed as instrument fault 8 and closes two queued sweeps.
- **The real open problem is sharper than the false one.** Not *"why does it
  loop"* but *"how does a model that legitimately thinks for a minute survive a
  multi-turn agent loop that pays the thinking again every turn"* — which is a
  harness question, and a much better one to hand to a researcher.

## The transferable rule

**A claim about what a text says requires reading the text.** Length, duration
and outcome are properties *of* the text; none of them is the text. When an
instrument records a measurement *about* content, it must keep the content, or
every conclusion drawn from it is unfalsifiable by construction.

The tell was available and ignored: the claim had **no `file:line`, no quoted
excerpt, and no artifact** — while sitting in a repo whose records rule requires
exactly those.
