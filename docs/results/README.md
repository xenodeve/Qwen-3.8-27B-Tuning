# tested — the register of what has actually been run

**This folder answers one question: has X been tried, and what happened?**

It exists because that question kept getting answered wrong. On 2026-08-21 the
plan stated *"`reasoning_effort` has never been swept here"* — while
`results/reasoning-effort-sweep.jsonl` had held six rows since 2026-08-18. In
the same hour, a design question about prompt caching was written up as open;
`results/prefix-cache.jsonl` had already answered it, including the specific
result that injecting a skill block at the front of the prompt forces a **full
re-prefill**.

Neither was hidden. Both were in reports. **Reports are narrative — they say
what a night meant — and a fact stated once inside a story is not findable.**

---

## How this folder differs from the other three

| folder | question it answers | shape |
|---|---|---|
| [`../reports/`](../reports/) | *what did we learn, and how?* | narrative, dated, argues from evidence |
| **`results/`** | *has X been tried? what happened?* | **a register. one row per thing tried** |
| [`../plans/`](../plans/) | *what do we intend to run?* | intent, not results |
| [`../researchs/`](../researchs/) | *what did someone else claim?* | unverified until measured here |

A row here is a pointer, not an argument. It names the thing, the verdict, the
raw file the number came from, and the report that explains it. If you want to
know *why*, follow the link; if you only need to know *whether*, stop here.

---

## The register

| file | covers |
|---|---|
| [`01-artifacts.md`](01-artifacts.md) | every model file loaded — size, real bits/weight, residency, quality |
| [`02-decoders.md`](02-decoders.md) | every `--spec-type` tried, and what each returned |
| [`03-memory-and-kv.md`](03-memory-and-kv.md) | KV types, `-ot`, `--fit-target`, checkpoints, batch |
| [`04-context-depth.md`](04-context-depth.md) | the depth ladder: what is resident where, and how fast |
| [`05-runtime-flags.md`](05-runtime-flags.md) | threads, placement, priority, polling, sampling |
| [`06-prompt-and-quality.md`](06-prompt-and-quality.md) | corpus arms, grammar, reasoning effort, prompt cache |
| [`07-telemetry-inventory.md`](07-telemetry-inventory.md) | **every value a run can yield**, which source it comes from, and what a restart would add |
| [`08-rtx3090-transfer.md`](08-rtx3090-transfer.md) | **what transferred from the RTX 3090 scan** — 434 techniques, which were tried here and what happened |

---

## Reading rules

**"Tested" is not "settled".** Several rows carry a caveat that makes the number
provisional — a probe too short, a prompt too repetitive, a sample of two. The
caveat column is not decoration; it is the difference between a result you can
act on and one you can only cite.

**Every number here is traceable.** If a row does not name a file under
`qwen38-tuning/results/`, it is not a measurement and says so.

**Before quoting anything, read
[`../reports/CORRECTIONS.md`](../reports/CORRECTIONS.md)** — twenty-seven claims this
project published and later contradicted. The rows here reflect the corrections;
older reports may not.

## Keeping it true

A sweep is not finished until its row lands here. That is the whole mechanism —
there is no hook, and the two incidents above are what it costs when the step is
skipped. `python C:\AI\scripts\audit-stale-claims.py` catches superseded *claims*
but cannot see a measurement nobody registered.

