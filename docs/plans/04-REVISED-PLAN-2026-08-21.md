# 04 — Revised plan, 2026-08-21

**Supersedes the ordering in `03-SIXTEEN-LAYER-PROGRAMME.md`.** That programme is
still the map of what is tunable; this changes what to do next and why.

Three things arrived after it was written: the deep-context measurements of the
last two hours, an external review of this exact model, and a reading of the
skills the real worker runs under. Each moved something.

---

## 0. The finding that reorders everything

**It was already in our own data, and nobody put the two columns side by side.**

| artifact | real bits/weight | accepted | contract pass |
|---|---:|---|---|
| V3 `UD-IQ1_S` | **1.84** | 0 of 12 | no fenced block at all, 12/12 |
| V3 `UD-IQ1_M` | ~2.0 | 10/21 | 41.5 % |
| V3 `UD-IQ2_XXS` | **2.16** | 19/27 | 58.3 % |
| `AD-IQ1_M` (AtomicChat) | **2.49** | 27/30 | — |
| pre-V3 `UD-IQ2_XXS` | **2.64** | 27/30 (90 %) | — |

**Five artifacts, two vendors, perfectly monotone.** Every step up in bits per
weight improves both the accepted count and the contract rate. Nothing else in
this project correlates that cleanly — not the flags, not the sampler, not the
KV type.

The project has spent two days trying to fix a **2.16 bpw** artifact with flags.
The table says the artifact is the variable.

### The mechanism this suggests, from an independent source

A commenter on the LocalLLaMA review of this exact model, running far larger
hardware:

> I mainly use **bf16** weight. Tried **q8** and it was **doubting itself at
> every step, like a paranoia** (compared to bf16).

If quantization damage shows up as *self-doubt* rather than as wrong answers,
then our measured failure would be the same effect four times further down the
bit ladder.

> **Corrected 08:30 — the premise was wrong.** This paragraph read *"the model
> loops inside the reasoning block until the token budget runs out"*. A full
> trace scores **0.00 % line repetition** and ends on `stop`. It thinks at
> length and finishes. What survives is the bpw correlation in the table above;
> the mechanism attached to it does not. See `CORRECTIONS.md` §12.

That reframes the fix. A grammar constrains the *output*; it cannot stop a model
from spending 8,192 tokens deciding. That is consistent with what we measured:
grammar moved contract pass 58.3 % → 84.3 % and moved accepted **down**, 19/27 →
16/27. The format got fixed. The thinking did not.

**This is a hypothesis, not a result.** One subjective comment, and our own table
is confounded — different vendors, different tensor mixes, different quantizers.
But it is testable, it explains everything we have seen, and the test is cheap.

### The rung we own and have never measured

> 🔴 **Retracted — [`CORRECTIONS.md` §19](../reports/CORRECTIONS.md).**
> `UD-IQ2_S` has **38+ measured rows** across six result files, dozens of
> logs and four worker profiles; it holds `65+0` at 131,072 with
> `--fit-target 192` at 23.21/23.92 tok/s. **Anyone quoting "never loaded"
> from here is quoting a plan, not a result.** What is genuinely untested is
> whether it beats `UD-IQ2_XXS` **plus a drafter** on accepted tasks/hour.

`Qwen3.8-27B-UD-IQ2_S.gguf` — **8.37 GB, already in the local cache since
2026-08-20 01:36, never loaded once.** It sits between `UD-IQ2_XXS` (6.77 GiB,
2.16 bpw) and pre-V3 `UD-IQ2_XXS` (8.39 GiB, 2.64 bpw): exactly the gap between
"fails on format" and "90 % accept".

If the bpw relationship is real, this artifact is the answer to the goal's sweet
spot, and it has been sitting on disk unopened.

---

## 1. What the machine measured overnight

*Rewritten 05:50. The first version of this section stated the first bullet as a
property of the flag. It is not one — D4 refuted it ninety minutes later.*

**`-ot ssm` breaks speculation on one combination, not in general.** On
`v3-iq2xxs` at 163,840 it restores `65+0` and takes n-gram acceptance from 100 %
to **4 %**, reproduced in four boots, leaving the resident arm slower than the
non-resident one (32.4 vs 38.7 tok/s). On `v3-iq1m` at 196,608 the same slice
holds **100 %** acceptance and is worth **+114.78 %** alone, **+181.57 %** with
n-gram on top. The four-block variant is a third behaviour again — it drafts
*nothing* (`draft_n` zero). Three outcomes from one flag; which half of the
combination is responsible is unknown and nothing queued separates them.

**A CPU layer at depth costs 22 %.** `v3-iq2xxs` at `62+3` gives 38.65 tok/s
where `v3-iq1m` at `65+0` gives 47.30 — same depth, same flags, acceptance 100 %
in both. Residency is worth roughly 576 MiB of anything we can find.

**The residency ceiling was wrong, in our favour.** `v3-iq2xxs` holds **`65+0` at
147,456**, which nobody had measured: report 21 recorded 131,072 resident and
163,840 at `62+3`, and never looked between them. **16K of context, free.**

**Depth is limited by VRAM alone.** `n_ctx_train = 262144`, no scaling engaged at
163,840. Nothing about the model stops us; only the card does.

**n-gram acceptance may be a coherence detector.** 100 % on `v3-iq2xxs` and
`v3-iq1m`, **37.5 %** on `v3-iq1s` — the artifact that scores 0 of 12. Currently
confounded with depth; step V2 separates them.

### And the instrument fault that qualifies all of it

**The timed prompt is 84.5 % duplicate lines.** `filler()` repeats one class with
a four-digit index — 962 blocks at 147,456, adjacent blocks 99.5 % identical. An
n-gram decoder drafts from context, so this is close to the best case that can be
constructed for it, and **every n-gram figure in this project was measured on
it**: +135.89 % at 16K, +200.22 % at 131,072, +330.40 % at 147,456. Acceptance
pinned at 99–100 % across every depth is the tell, and the figures rising with
depth is the filler getting more repetitive, not the model getting faster.

What survives: n-gram is free, costs no VRAM, needs no drafter file, and its
output is byte-identical. What does not: **the size of the numbers.** Steps F1
and F2 measure the same arms at 73.17 % repetition; the fall is the correction
owed.

---

## 2. What the external review changed

Hardware and absolute numbers do not transfer — the reviewer runs 3× 3090 plus a
P40 at `UD-Q8_K_XL`. The behavioural observations do.

**`reasoning_effort` — corrected 06:05. It HAS been swept here, and I said it
had not.** `results/reasoning-effort-sweep.jsonl` holds six rows from
`scripts/sweep-reasoning-effort.ps1`: `low`, `medium`, `xhigh`, two runs each, on
a tool-calling probe. **All six reached the patch**, wall 50–107 s, reasoning
384–1,008 characters rising with effort.

What is actually missing is narrower and still worth running: that sweep ran on
**Q4** with a tool probe and n=2, not on the 2-bit V3 artifacts where the looping
failure appears, and never through the corpus. The reviewer's finding — xHigh 15
minutes, **medium 3 minutes for "90 % of the result"**, low 3 seconds — is about
generation length on one prompt, which our probe was too short to see.

`--reasoning-budget` and `-rea off` are different levers again: `-rea off`
deletes the block, `low` shortens it. With 7 of 48 corpus attempts truncating on
budget, `low` still aims at the failure.

**The thinking can be steered by instruction.** Another commenter:

> The thinking problem is so easily solved with a system prompt … That's just the
> default of xhigh with no **"don't hedge, make conclusions, work forward, don't
> reconsider"** instructions.

Our developer message instructs the *output format* and has never said anything
about *how to think*. This costs nothing and targets the blocker directly.

**A concern about our own instrument.** The reviewer observed speculation warming
up over a long generation — *"the MTP had gotten extremely fast (91 tk/s vs 62
tk/s starting rate)"*. **Our timed generation is 160 tokens.** If speculative
decoders need a longer run to reach rate, every decoder number we hold is
understated, and `draft-mtp` was eliminated on that evidence.

---

## 3. What reading the worker's own skills changed

`qwen-agent` line 12: `claude-9arm` is `claude --model **qwen3.6-35b-a3b**` — a
**different, weaker model** than the one being tuned. Its A/B finding that
"scope beats guidelines" was measured there, so it does not settle whether
`karpathy-guidelines` helps Qwen3.8-27B. Treat that question as open.

`qwen-agent` line 79: `claude-9arm` is **a full Claude Code instance** with a
tool loop, file access, and the ability to invoke skills.

**Our corpus fires one chat completion and grades the reply.** Production runs an
agent that can read, edit, run tests and retry. A missing code fence is a
permanent failure in our harness and a self-correcting hiccup in production.
**Every quality number this project holds describes something nobody ships.**

---

## 4. The plan

Ordered by what each step can invalidate, not by cost. The goal's ordering —
performance first, then quality — is preserved: P0 is an instrument repair, P1 is
already running, and the quality work starts at P2.

### P0 — Does the 160-token probe understate speculation?

Add a long-generation arm (`n_predict` 512 and 1024) against the same baseline at
16,384 and 131,072. If the speculative advantage grows with length, **every
decoder verdict in report 20 is provisional** and `draft-mtp`, `draft-dflash` and
the eagle/dspark arms must be re-run before any of them stays eliminated.

First because it is cheap, and because a broken ruler makes everything measured
after it worthless.

### P0b — The prompt cache, and where an injected skill must go

**Also already measured, in `results/prefix-cache.jsonl`, and nobody carried it
forward.** Nine rows from `bench/prefix_cache_gate.py`:

```text
  turn-1                 prompt_n 3878   cache_n    0    12.6 s   cold
  turn-2 / 3 / 4         prompt_n 35-43  cache_n ~3900   1.3-3.9 s
  append-only-control    prompt_n   28   cache_n 3981    2.4 s    stays cached
  reorder-tool-schemas   prompt_n 3990   cache_n    1   11.1 s    FULL RE-PREFILL
  edit-system-prompt     prompt_n 3992   cache_n    1   11.5 s    FULL RE-PREFILL
  prepend-skill-block    prompt_n 4006   cache_n    1   12.1 s    FULL RE-PREFILL
  cache_prompt=false     prompt_n 3991   cache_n    0    9.9 s
```

Two things follow, and the second changes what P5 should do.

**Caching works and is worth a factor of five** on an append-only turn — 2.4 s
against 12.1 s on a 4 K prefix. Against the ~40 K prefix a real worker carries
that gap is far larger, and the corpus runs `cache_prompt: False` on every call.

**Injecting a skill at the FRONT destroys the cache.** The row is labelled
*"Xeno injects skills ahead of everything"* and it re-prefills the lot. So
`clink-subagents` §7 (hand over `karpathy-guidelines` every call) and the prompt
cache are in direct tension **as usually implemented** — and the fix is position,
not omission: the injected text has to sit inside the stable prefix, identical on
every call, ahead of anything that varies. P5 must measure both placements or it
will measure the cache instead of the skills.

### P1 — Finish the deep-context queue *(running)*

`D4`, `E1`, `E3`, `V1`, `V2`. `V1` is the one that matters: can `--fit-target` or
a smaller compute buffer buy `v3-iq2xxs` its `65+0` at 163,840 **without** the
acceptance collapse `-ot` caused? Worth 22 % of decode if it lands.

### P2 — The bpw ladder, on the rung we already own

`UD-IQ2_S`, 8.37 GB (= 7.80 GiB; one file, two units — not two artifacts),
never loaded. Three questions in one run:

> **Executed. This paragraph is intent, and the intent was carried out.**
> `v3-iq2s` now has 38+ measured rows across six result files and four
> `worker-iq2s-*.ps1` profiles. Anyone quoting "never loaded" from here is
> quoting a plan, not a result — [CORRECTIONS §19](../reports/CORRECTIONS.md).

1. **Where is its residency ceiling?** Between `UD-IQ2_XXS` (131,072) and pre-V3
   (`58+7` at 131,072). If it holds `65+0` at 131,072 it changes the answer.
2. **What is its contract rate?** The table in section 0 predicts something
   between 58.3 % and 90 %. A result outside that range falsifies the bpw story
   and is worth more than one inside it.
3. **What is its n-gram acceptance?** If the coherence-detector idea holds, this
   predicts the corpus before the corpus runs.

**This is the highest-value quality step, and it comes before any further flag
work**, because section 0 says the artifact is the variable and the flags are not.

### P3 — Steer the thinking, do not constrain the output

Three arms on one artifact, one change each, against the existing control:

- `reasoning_effort: low` — never tried here
- the developer message plus *"don't hedge, make conclusions, work forward,
  don't reconsider"* — never tried here
- both

If the bpw story is right, these treat the symptom and P2 treats the cause; both
are worth knowing, and these are free.

### P4 — Grammar alone, reasoning left on

Built and queued on 2026-08-21 03:29, aborted at 03:30 when the goal reordered.
`serve-v3-iq2xxs-gram.ps1` exists. The 26-point contract jump has never been
attributed because the only arm that showed it changed two things.

### P5 — Skill injection, now an open question again

`--skill` is wired into `run_retry_bench.py` (103 tests). Send the real
`karpathy-guidelines` and `tdd` text ahead of the contract sentence and measure.
Reopened by section 3: the A/B that suggested it would not help was run on a
different model.

Expect `tdd` to *hurt*: it instructs the model to write tests first, and the
contract forbids tests. That is worth measuring rather than assuming, because it
is what production actually sends.

### P6 — Measure the agent, not the server

The largest gap and the largest job. A harness that drives `claude-9arm` against
our server over the same corpus, so the number describes the worker that ships.
Not started; scoped here so it stops being invisible.

---

## 5. What this plan drops

- **`-ot` as a residency lever** for anything using speculation (section 1).
  Still valid with speculation off, which is not a configuration we ship.
- **`AD-IQ1_M` at 128K.** 6.08 tok/s at `65+1`; the `-ot` route is closed.
- **Further sampling sweeps.** Two passes found nothing above the 13.6 % floor.
  Section 0 says the search was on the wrong axis.
- **The greedy hash as a cross-depth signal.** It takes one of two values and
  switches on things that are not the arm under test (report 24 section 6).
