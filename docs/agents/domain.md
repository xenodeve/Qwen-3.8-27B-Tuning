# Domain glossary — what the words mean here

**Load-bearing.** A term in an issue title, a hypothesis, a test name or a PR
body uses the exact bold word below. Drifting to an alias is a defect, because
this repo's whole product is a record other agents read later.

A concept missing from this file is a signal: either language is being invented,
or something real is unmodelled. Add it rather than working around it.

---

## The apparatus

**artifact** — one GGUF file. Never "the model": this project serves nine
different quantizations of the same weights and the differences between them are
the main finding. *Aliases to avoid: model, quant, build.*

**bits per weight (bpw)** — the **real** figure from the loader's tensor-type
histogram, not the filename. `AD-IQ1_M` is 2.49 bpw and `UD-IQ2_XXS` is 2.16, so
the file named for 1-bit is the heavier one. **Never infer bpw from a name.**

**layer split** — written `65+0`, `62+3`, `58+7`: GPU layers plus CPU layers. The
single most predictive number in this project. `65+0` is **resident**; anything
else is not.

**resident** — every layer on the GPU. Not "fits", not "loads" — a server that
loads at `58+7` has loaded and is not resident, and the distinction is worth a
factor of four at depth.

**arm** — one configuration under test in a sweep. An arm is a set of flags, a
depth and an artifact; changing any of the three makes it a different arm.

**round** — one pass over every arm in a sweep. Sweeps run at least two rounds
with the order reversed, because boot-to-boot drift lands on whichever arm ran
later otherwise.

**the drift floor** — **13.6 %**. Measured peak-to-peak across 25 boots of one
control config. An effect smaller than this, or with an inconsistent sign across
rounds, is **unresolved** — not "small". *Alias to avoid: noise.*

## Speculation

**speculative decoding** — a drafter proposes tokens, the target model verifies
them. Verification is exact, so accepted output is byte-identical to what the
target would have produced alone.

**acceptance** — drafted tokens the target agreed with, as a percentage. **On
this project's synthetic benchmark prompt it reads 99–100 %; on real code it is
16.8 %**, because the benchmark prompt is 84.5 % duplicate lines. Always say
which.

**drafter** — the thing proposing tokens. An n-gram drafter holds nothing; a
model drafter holds weights and therefore competes with the layers, which is why
every model drafter loses on 12 GB.

## Measurement

**the corpus** — ten coding tasks graded by **executing** the produced code
against hidden assertions. The only quality signal here that is not a proxy.

**accepted** — the code ran and the assertions passed. Not "looks right", not
"the model replied".

**output contract** — one fenced `python` block and nothing else.
`output_contract_pct` is the **PASS** rate. It was read as a violation rate for a
full day; see `docs/reports/CORRECTIONS.md` §1.

**instrument fault** — a defect in the measuring apparatus that produced a
believable number instead of a failure. Thirteen are documented. This is the
project's characteristic failure mode and it has its own vocabulary for a
reason. *Alias to avoid: bug.*

**greedy hash** — a fingerprint of a fixed greedy completion, used to detect
whether an arm changed the answer. **Valid within one depth and one artifact
only** — it takes one of two values across everything measured and switches on
things that are not the arm under test.

**prefix** — everything a harness sends before the task. OpenCode's default
profile sends 99,073 tokens of it; the lean profile sends ~5,377. *Alias to
avoid: system prompt — the prefix is larger than that and includes tool schemas.*

## The record

**report** — narrative, dated, argues from evidence. Lives in `docs/reports/`.

**the register** — `docs/results/`. One row per thing tried: has X been tried,
what happened. A report says what a night meant; the register says whether
something was done. **A fact stated once inside a narrative is a fact nobody can
find**, which is why both exist.

**correction** — a published claim this project later contradicted with its own
data. Registered in `docs/reports/CORRECTIONS.md` **and** as a rule in
`scripts/audit-stale-claims.py`. A retraction with only one of those is
unfinished, because the copies in other files stay invisible.

**verified / hypothesis** — a number that names the file it came from, versus one
that does not. Every claim is one or the other, and saying which is not optional.
