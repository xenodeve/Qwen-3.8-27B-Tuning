# 24 — Beyond 128K, at speed

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

**Status: in progress.** D1-D4 complete, E1 running; E3, V1, V2, W1-W4 queued.
running. Numbers below are final for the steps marked complete.

Report 21 answered *how deep does each artifact stay resident*. It never asked
*how fast is it down there* — `ctx_ceiling.py` checks whether a server loads and
records the layer split, and stops. So every context past 131,072 had residency
data and **no throughput data at all**.

This report fills that in, in the order the goal asks for: squeeze performance
first, find the fastest configuration, and only then trade speed for quality.

---

## 0. The headline, so far

**At 163,840 the fastest arm is not the fully-resident one.** `-ot` on the ssm
tensors does restore `65+0` exactly as report 20 promoted it to — and turning on
speculation afterwards makes the machine slower than leaving three layers on the
CPU.

```text
v3-iq2xxs @ 163,840, q4_0 KV, two rounds, --fixed-text

  arm                 split   tok/s            vs base    acceptance
  q4_0 (baseline)     62+3    19.36 / 18.83        --         --
  ot-ssm-4            65+0    22.72 / 21.73    +16.38 %       --
  ot-ssm-10           65+0    21.40 / 20.95    +10.90 %       --      under floor
  ngram-mod           62+3    37.89 / 38.65   +100.48 %    100.0 %    <-- fastest
  ot-ssm-10 + ngram   65+0    32.37 / 32.44    +69.74 %      4.0 %
```

Everything in the project up to now said residency wins. It still does -- without
speculation, `ot-ssm-4` at `65+0` beats the `62+3` baseline by 16 %.

**Narrowed 05:05 by step D4.** The first write-up of this section generalised
the inversion into a property of the flag. It is not one: the same offload on
`v3-iq1m` at 196,608 keeps acceptance at 100 % and is worth +114.78 % on its
own. What holds is the narrow statement — **on `v3-iq2xxs` at 163,840, the cost
`-ot ssm` imposes on speculation exceeds the benefit residency gives back.**
Section 1 has the full retraction and both data sets.

---

## 1. Why the drafter fails — and the retraction that followed

> **RETRACTED IN PART, 05:05 the same night, by step D4 on this machine.** The
> paragraph that used to close this section read: *"`-ot` on ssm tensors and
> speculative decoding are antagonistic."* **That is not true as a general
> claim.** D4 ran the same `ot-ssm-10` slice with the same n-gram configuration
> on `v3-iq1m` at 196,608 and measured **acceptance 100.0 %**, twice. The
> collapse is real where it was seen and does not generalise. What survives and
> what does not is set out below.

**What was measured, and still stands.** On `v3-iq2xxs` at 163,840, adding the
ssm offload to the n-gram arm took acceptance from 100 % to 4 %, in both rounds:

```text
  ngram-mod          62+3    acceptance 100.0 %    38.65 tok/s
  ot-ssm-10 + ngram  65+0    acceptance   4.0 %    32.44 tok/s
```

Same artifact, same depth, same speculative settings; the only difference is the
`-ot`. Two rounds, order reversed, both 4 %. **The effect is real there.**

100 % to 4 % is not a tax on accepted work, it is the near-total rejection of
drafted work. The drafter proposes, the target refuses, and the speculation
budget is spent producing tokens that are thrown away — which is why that arm
came out slower than doing no offload at all.

**What D4 refutes.** Take the same slice one artifact and one depth over:

```text
  v3-iq1m @ 196,608
  q4_0               60+5    --                     8.81 /  8.93 tok/s
  ot-ssm-10          65+0    --                    18.32 / 19.79   +114.78 %
  ot-ssm-10 + ngram  65+0    acceptance 100.0 %    21.93 / 28.06   +181.57 %
```

Here the ssm offload is worth **+114.78 %** on its own — it is rescuing the arm
from five CPU layers rather than three — and speculation on top of it works
perfectly, at full acceptance, for another 67 points.

**So the mechanism is narrower than the first reading of it.** `-ot ssm` does not
break speculation as such. Something about the D1 combination does, and the
candidates are the artifact (`iq2xxs` and `iq1m` carry different tensor types, so
"the ssm tensors" are not the same tensors), the depth, or an interaction. Step
E1 — `ot-ssm-4` against `ot-ssm-10`, both with n-gram, on `v3-iq2xxs` at 163,840
— is running now and tests whether the 4 % even reproduces.

**The lesson is the one this project keeps re-learning.** A verdict measured on
one artifact at one depth was written up as a property of a flag, three hours
after report 22 listed *"a verdict at one depth does not transfer to another"* as
trap number three. It took ninety minutes and one more arm to contradict it.

**Also missing from D4, and worth stating rather than glossing:** there is no
`ngram-mod` alone arm at 196,608 on `v3-iq1m`. In D1 that arm was the winner. Here
it was never run, so *"ot + ngram is best at 196,608"* is not established — only
that it beats the `60+5` baseline and the offload alone.

---

## 1b. E1 — the collapse reproduces, and it is not a scaling effect

`v3-iq2xxs` at 163,840, two more rounds, baseline is now the arm that would ship
(`ngram-mod` at `62+3`, 38.65 tok/s) rather than the bare control:

```text
  arm                 split   tok/s           vs ngram-mod   acceptance
  ngram-mod           62+3    36.24 / 38.17        --          100.0 %
  ot-ssm-4 + ngram    65+0    38.29 / 34.17     -2.41 %        none drafted
  ot-ssm-10 + ngram   65+0    33.37 / 32.07    -11.95 %          4.0 %
```

**The 4 % is real.** Third and fourth boots of `ot-ssm-10 + ngram`, both exactly
4.0 %. Whatever it is, it is deterministic on this artifact at this depth.

**It is not a dose-response.** The four-block slice does not land between 4 % and
100 %; it reports **no drafts at all** — `draft_n` is zero, so the harness records
`acceptance: null`. Ten blocks drafts and gets refused; four blocks does not
draft. That is two different failures, not one effect scaling with the number of
offloaded blocks, and it rules out the per-token-cost explanation this step was
written to test.

**Neither arm beats plain n-gram**, and both land under the drift floor with the
sign flipping — so the honest reading is that at this depth, on this artifact,
**residency and speculation are worth about the same, and the machine will give
you one or the other.** `ot-ssm-4` recovers by being resident roughly what it
loses by not speculating.

Set against D4, where the same slice on `v3-iq1m` at 196,608 held **100 %**
acceptance and was worth +181 %, the effect is specific to a combination and not
a property of the flag. Which half of the combination — artifact or depth —
remains unknown, and no queued step separates them.

---

## 2. Depth is limited by VRAM alone, not by the model

Checked in the loader output at both depths:

```text
print_info: n_ctx_train     = 262144
print_info: rope scaling    = linear
print_info: n_ctx_orig_yarn = 262144
llama_context: n_ctx_seq (163840) < n_ctx_train (262144)
```

**The artifact's native window is 262,144.** Both 131,072 and 163,840 sit inside
it, no YaRN or scaling extension is engaged, and the loader says so explicitly.

This matters for the goal, which asks to go past 128K if possible. It means the
ceiling is not a model property to be worked around — it is 12 GB of VRAM and
nothing else. Every MiB freed is context that the model already knows how to
use.

---

## 3. What depth costs, measured

```text
v3-iq2xxs, q4_0 KV, ngram-mod, --fixed-text
  131,072    81.46 / 73.41 tok/s     65+0    KV 2,304 MiB
  163,840    37.89 / 38.65 tok/s     62+3    KV 2,880 MiB
```

**32,768 more tokens of window costs a little over half the decode rate.**

Two things are mixed together in that drop and this report cannot yet separate
them: the extra 576 MiB of KV, and the three layers that got pushed to the CPU
to make room for it. Step E3 measures 147,456 — the midpoint — to find where the
curve bends. If the drop is smooth, it is the KV. If it falls off at the depth
where the split first leaves `65+0`, it is the layers, and then the question
becomes how to free 576 MiB by some means other than `-ot`.

---

## 4. A measurement caveat found on the way

**The greedy hash is not comparable across depths.** At 163,840 all five arms
returned `3EFE93950A8A980E`, including the plain `q4_0` baseline. At 131,072
every arm on the same artifact returned `04E5CAB1D14525C0`.

The greedy probe is a fixed short prompt (`def fibonacci(n):`, `n_predict 60`,
temperature 0, `cache_prompt: False`), so nothing about the *request* changed —
only the server's `-c`. Cause not yet established.

Two consequences, one procedural and one substantive:

- **Procedural.** Any report that compares a greedy hash between two depths is
  comparing nothing. Within a depth the comparison is still valid, which is what
  the 16K disqualification of `ngram-cache` rests on — it differed from its own
  same-depth baseline.
- **Substantive.** The window size may change what the model writes. That is a
  quality question and it cannot be settled by inference; it needs a corpus at
  the deep window, which is now on the list.

---

## 5. D2 — the residency payoff, priced

`v3-iq1m` already holds `65+0` at 163,840. Same depth, same KV, same flags as
D1's arms; the only difference is the artifact and therefore the split.

```text
                            split   tok/s           vs base
  v3-iq1m   q4_0            65+0    24.37 / 24.55      --
  v3-iq1m   ngram-mod       65+0    45.82 / 47.30   +90.34 %   acceptance 100 %
```

Put beside D1 at the same depth, this is the cleanest statement of what a CPU
layer costs at 163,840:

```text
  v3-iq2xxs  ngram-mod   62+3    37.89 / 38.65
  v3-iq1m    ngram-mod   65+0    45.82 / 47.30      +22 %
```

**Three CPU layers cost 22 % of decode**, with speculation on and acceptance at
100 % in both arms — so this is the residency effect alone, not a speculation
artefact. It also means the arm that would ship at this depth is leaving 22 % on
the table for want of about 576 MiB.

That is what makes step V1 the highest-value item queued. If `--fit-target` or a
smaller compute buffer buys `v3-iq2xxs` its `65+0` **without** the acceptance
collapse that `-ot` caused, it should land near 47 tok/s — the speed of `iq1m`
with the quality of `iq2xxs`.

The two artifacts are not interchangeable on quality, which is the whole reason
this matters:

```text
  v3-iq2xxs   19/30 accepted   58.3 % contract pass
  v3-iq1m     10/21 accepted   41.5 % contract pass
```

So at 163,840 there is currently a real trade — 38.65 tok/s at the better
quality, or 47.30 at the worse — and V1 is the attempt to remove it.

---

## 6. The greedy hash is bimodal, and that needs checking

Section 4 recorded that every arm at 163,840 returned `3EFE93950A8A980E` where
every arm at 131,072 returned `04E5CAB1D14525C0`. D2 adds that **`v3-iq1m`
returns `3EFE9395…` too** — a different artifact, same hash. And the same value
appeared once before, on `ngram-cache` at 16,384, where every other arm gave
`04E5CAB1…`.

So across everything measured, the greedy probe returns one of exactly **two**
values. Report 20 already established the harmless half of this: the probe has
one right answer, so many artifacts agreeing proves nothing. The new part is
that the *disagreement* is also binary, and it switches on things that are not
the arm under test.

This matters because the greedy hash is the project's cheap quality predictor —
thirty seconds standing in for a four-minute gate. A detector that flips between
two states for reasons not yet identified is not obviously detecting what it is
read as detecting.

**What still holds:** `ngram-cache` differed from its own same-depth,
same-artifact baseline in four boots, and that comparison is internally valid.
The disqualification stands.

**What does not:** any reading of the hash across depths, and any confidence
that a matching hash means the arms agree on harder prompts than
`def fibonacci(n):`.

**Deferred, deliberately.** The goal is performance first. This is logged as an
open item rather than chased now, because the fix is a better probe — several
prompts with distinguishable answers — and that is a quality-phase job.

---

## 7. Steps

| step | what it decides | state |
|---|---|---|
| D1 | `v3-iq2xxs` at 163,840 — does `-ot ssm` buy `65+0`, and what is fastest there | **complete**, sections 0–4 |
| D2 | `v3-iq1m` at 163,840 — already `65+0`; pure throughput | **complete**, section 5 |
| D3 | `v3-iq1s` at 196,608 — already `65+0`; the deepest resident config known. Its corpus is 0/12, so this measures the hardware ceiling, not a candidate | **complete** — 65+0, but n-gram gives only +12.14 %, acceptance 37.5 % |
| D4 | `v3-iq1m` at 196,608 — `60+5`; does `-ot ssm` rescue it as it did D1 | **complete**, section 1 — `-ot ssm` +114.78 %, with n-gram +181.57 %, acceptance 100 % |
| E1 | `ot-ssm-4` vs `ot-ssm-10`, both with n-gram — is the acceptance collapse continuous or a cliff | **running** — now the arm that decides whether the 4 % reproduces at all |
| E3 | 147,456 — where the speed/depth curve bends | queued |
| V1 | 163,840 — can `--fit-target` or a smaller compute buffer buy `65+0` **without** the acceptance collapse? The highest-value item queued, see section 5 | queued |
| V2 | 196,608 — if V1 found the MiB, the same lever one step deeper, where `v3-iq2xxs` does not currently go at all | queued |
