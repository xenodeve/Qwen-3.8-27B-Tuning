# Panel Review of the Test Plan — What Three Independent Agents Found

> **Date:** 2026-08-20 UTC+7
> **What this is:** before running
> [`../plans/01-V3-Q1-Q2-TEST-PLAN.md`](../plans/01-V3-Q1-Q2-TEST-PLAN.md)
> unattended for several hours, the design was put to three independent agents
> from different model families, each given the same self-contained brief and no
> shared context. This records what they said, which findings were acted on, and
> which are still open — because a critique that only lives in a chat log gets
> re-derived at cost later.
> **Panel:** `gpt-5.6-sol` (medium), `Gemini 3.1 Pro (High)`, `Grok 4.5 (high)`.

---

## 1. Where they converged — independent agreement, so signal

### The pre-V3 vs V3 comparison is not identifiable

Raised by **two agents independently**. Two treatments changed at once: the
weights were requantized *and* the built-in MTP head was removed from every
artifact at 2-bit and smaller. Any difference attributed to "Dynamic 3.0" could
belong to either, or to their interaction through the layer split.

Grok named the fix precisely: add **pre-V3 `IQ2_XXS` with speculation forced
off**, giving

```text
old + head   vs  old − head   ->  the head's effect
old − head   vs  new          ->  the generation effect
```

**Status: already satisfied, by accident.** `iq2xxs-nomtp` (pre-V3, no
`--spec-type`) and `iq2xxs-mtp2` (pre-V3, MTP n=2) were both already arms, and
the control used all along was the no-spec one.

**Residual, and reported as such:** the pre-V3 *file* still contains the
`blk.64` tensor and pays VRAM for it whether or not it is driven — which the
layer count makes visible (66 layers against V3's 65). So the measured
generation effect **bundles head removal**, and report 12 states it that way
rather than splitting a difference the arms cannot separate.

### Stage 4's arm-selection rule was inverted

Raised by **two agents independently**. The rule read *"the two arms with the
most free VRAM"*, which selects the **smallest and most damaged** artifacts —
exactly the ones least likely to survive the corpus that decides the question.

> Gemini: *"KV compression exists to fit BETTER weights, not to leave VRAM
> empty."*

**Status: fixed.** The rule is now *the largest arm that still holds full
residency once the 128K `q4_0` cache is allocated*, plus the Stage-3 utility
winner and the control. Quality **and** headroom together, never headroom alone.

---

## 2. Where they diverged — the most useful part

All three said the corpus is blind to something. **They named different things**,
and the disagreement is more informative than the agreement.

| agent | what the corpus cannot see | cheapest fix proposed |
|---|---|---|
| Codex | **cross-file interface drift** — signatures, schemas and invariants going out of step across dependent edits | one 2–3 file repository repair with an executable integration test |
| Grok | **format-constraint adherence** — and that all ten tasks are *re-derivable from the prompt*, so knowledge and API-recall damage is invisible | assert on the raw reply before extraction; no new tasks needed |
| Gemini | *(did not answer this; its brief was systems-focused)* | — |

**Grok's was actioned first** because it costs nothing and is the failure said to
appear *before* closed algorithmic coding degrades. `harness.check_output_contract`
now scores the raw reply — one fenced block, nothing outside it, no `__main__`
guard — as a **separate rate**, deliberately not folded into pass/fail, because
redefining a passing task mid-project would make every earlier number
incomparable.

It earned its place immediately: V3 `IQ1_S` emitted **no fenced block in twelve
of twelve attempts**, which the old harness had been reporting as
`NameError: name 'merge_intervals' is not defined` — a coding failure that never
happened.

**Codex's multi-file task is still not built.** It is the largest untaken
recommendation in this document.

---

## 3. What only one agent saw

### Gemini: the residency number cannot detect WDDM paging

> *"Windows WDDM permits VRAM overcommit. llama.cpp will still report 65/65 GPU,
> but performance falls off the residency cliff. This is almost certainly what
> caused the instability at 345 MiB headroom."*

Every conclusion in this project rests on the layer split the loader prints. The
mechanism Gemini described would make that number true and the claim false.

**Status: measured, and the hypothesis is closed — not confirmed.**
`bench/residency_check.py` reads the process's shared GPU memory during real
generation:

```text
pre-V3 IQ2_XXS    654 MiB free    shared 98 MiB / 9,417 dedicated = 1.04 %
V3 IQ1_S        3,290 MiB free    shared              …           = 1.41 %
```

**The ratio does not climb as headroom falls.** The arm with five times the
headroom has a slightly *higher* ratio, which is the signature of ordinary
pinned staging for host-to-device copies rather than eviction. The project's
residency conclusions hold.

Still open: the **~345 MiB regime** itself, where the unexplained
`[6.70, 8.28, 11.57]` spread was seen. The MoE arms sat at 227–335 MiB free and
are the right place to test it.

### Gemini: OS page-cache thrashing across sequential boots

The total payload across arms exceeds system RAM, so cycling through them evicts
earlier models from the standby list, and CPU-resident layers then block on SSD
reads.

**Status: judged narrower than stated, and not actioned.** For a *resident*
artifact the weights are copied to VRAM and the host page cache affects load time
only, not decode. The concern is real for arms with CPU-resident layers — `Q4` at
33+32, `Q2_K_XL` at 61+4 — and those are exactly the arms whose depth figures are
already the least load-bearing. Recorded rather than fixed.

### Codex: truncation is censoring, not failure

> *"No arm may be rejected while any evaluated response ended at the token limit.
> Treat every length-truncated result as censored."*

**Status: fixed, and it fired on first use.** `retry_economics` now separates
`decided` from `tasks`, counts `censored`, and raises a
`censoring_could_change_verdict` flag. On V3 `IQ1_S` it correctly split
`merge_intervals` (truncated — outcome unknown) from `lru_cache` (stopped —
genuinely failed). Before the change both would have counted as "1-bit is
broken", which is precisely the claim this project had already published once
and withdrawn.

### Grok: a generous budget introduces a *new* bias

> *"8192 stops false 'dumb' verdicts from truncation; without a time or token cap
> it invents false 'slow = weak quant' verdicts."*

The sharpest single observation of the review, and the table already contained
the evidence: **four arms tie at 27/30 accepted and differ only in wall clock**,
2,004 s to 4,572 s.

**Status: fixed.** `accepted_of_decided` is the capability axis,
`wall_per_accepted_s` the throughput axis, and `merged_tasks_per_hour` now
carries a printed warning that it is the two multiplied together and must not be
read as a capability ranking on its own.

### Codex: byte-count pinning is not enough

Recommended a manifest with path, SHA-256, byte count, quantization metadata and
MTP presence, emitted into every result, aborting on mismatch.

**Status: partially done.** V3 `IQ1_S` was verified by SHA-256 against the
repository OID — the first artifact in this project checked by hash rather than
size. `cached()` now raises on ambiguity instead of choosing. The full manifest
with per-result identifiers is **not built**.

---

## 4. Scorecard

| finding | source | status |
|---|---|---|
| Truncation is censoring, not failure | Codex #1 | **fixed**, caught a real case immediately |
| Capability and throughput must be separate numbers | Grok (c) | **fixed** |
| Stage 4 selection rule inverted | Gemini #2 + Codex #6 | **fixed** |
| WDDM silent eviction uncontrolled | Gemini #3 | **measured, hypothesis closed** |
| Format-constraint adherence invisible | Grok (a) | **fixed**, caught a real case immediately |
| Artifact identity needs hashes | Codex #8 | **partial** — hash checked, manifest not built |
| pre-V3 / V3 not identifiable | Codex #2 + Grok (b) | **satisfied**, residual reported honestly |
| Corpus blind to cross-file drift | Codex #7 | **not done** — largest open recommendation |
| OS page-cache thrashing | Gemini #1 | **judged narrower**, not actioned |
| Prefix-cache state uncontrolled per run | Codex #5 | **not done** |

---

## 5. The critique that was right and is still unanswered

Both Codex and Grok argued that the whole staged design is heavier than the
decision needs.

> Grok: *"The plan is a causal science protocol for 'what did Dynamic 3.0
> change?' The actual decision is an engineering pick: which artifact maximizes
> verified coding throughput on this card."*
>
> Codex: *"Replace the three-round all-arm speed screen and early protocol gate
> with one feasibility load plus one execution-verified pass over the ten unique
> tasks for every arm. Complete the remaining two passes only for finalists."*

**This was not adopted, and the reason should be on record.** The staged design
was kept because Stage 1 is cheap (35 minutes for six arms) and because the
project's whole history is of throughput numbers that turned out to mean nothing
without residency and stability beside them.

**The panel was partly vindicated anyway.** The stage that actually earned its
keep was the one *added after* the review — `answer_screen.py`, four minutes per
arm — which rejected V3 `IQ1_S` before it consumed a 90-minute corpus. That is
Codex's "cheap direct screen first" argument in a smaller form than either agent
proposed, and it is the change that saved the most time.
