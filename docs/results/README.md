# tested — the register of what has actually been run

> 🔴 **Every page in this folder now carries a banner naming the reasoning
> effort its numbers were taken at.** Established 2026-08-24: the model's chat
> template supplies `xhigh` with an unlimited thinking budget, the client sends
> no effort field, and **nothing in this repo has ever overridden either** — not
> one of the five `worker-*.ps1` profiles, and not `bench/dflash2_arena.py`,
> which has zero references. Three pages carry exceptions where a run set the
> flag deliberately; the rest are the default throughout
> ([`05-runtime-flags.md`](05-runtime-flags.md)).

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
| [`09-hardware.md`](09-hardware.md) | **which card produced which numbers.** The GPU changed on 2026-08-23 and a **second card was added 2026-08-26**; read this before quoting any rate from 01–08 |

**Answered 2026-08-26, two cards — all in [`09-hardware.md`](09-hardware.md):**

| question | verdict | raw |
|---|---|---|
| Does `--fit` work across two devices? | **yes** — *no changes needed*, splits by free VRAM 41:59 | boot log |
| Is the 5060 Ti's slot really x4? | **yes** — gen4 **x4** under load. The *generation* downtrains at idle; the width never does | 49 samples, 34 busy |
| Does `-sm row` work on this pair? | **no** — `device CUDA0 does not support split buffers`, fails at model load | `logs/dflash2-both-row-*.log` |
| Does the second card speed up `UD-Q2_K_XL`? | **prefill +57.4 %** [+56.0, +60.0]. **Decode +1.5 %** [+1.1, +2.1] with speculation off | `dual-gpu-16384.jsonl`, `dual-gpu-nospec-16384.jsonl` |
| Is the −78.3 % speculative decode figure a hardware result? | **no** — it measures how much the model repeated itself ([CORRECTIONS 32](../reports/CORRECTIONS.md)) | same |
| Does 28 GB make `UD-Q4_K_XL` resident? | **yes, to 229,376** — `66+0` at every rung including the served 147,456; one layer spills at 262,144 | `bench/ctx-ceiling-dual-q4*.jsonl` |
| What does the second card buy `UD-Q4_K_XL`? | **+79.9 %** [+77.3, +82.2] — it is the residency cliff: `55+11` becomes `66+0` | `dual-gpu-q4-nospec-16384.jsonl` |
| Noise floor, two-card machine, ctx 16,384 | **under 0.8 %** per arm across three boots. Not transferable to depth ([CORRECTIONS 23](../reports/CORRECTIONS.md)) | all of the above |
| Should the served profile move to Q4? | **UNDECIDED — the developer's call.** Costs about a third of raw decode; quality has never been measured here | — |


---

## Reading rules

**"Tested" is not "settled".** Several rows carry a caveat that makes the number
provisional — a probe too short, a prompt too repetitive, a sample of two. The
caveat column is not decoration; it is the difference between a result you can
act on and one you can only cite.

**Every number here is traceable.** If a row does not name a file under
`qwen38-tuning/results/`, it is not a measurement and says so.

**Before quoting anything, read
[`../reports/CORRECTIONS.md`](../reports/CORRECTIONS.md)** — twenty-eight claims this
project published and later contradicted. The rows here reflect the corrections;
older reports may not.

## Keeping it true

A sweep is not finished until its row lands here. That is the whole mechanism —
there is no hook, and the two incidents above are what it costs when the step is
skipped. `python C:\AI\scripts\audit-stale-claims.py` catches superseded *claims*
but cannot see a measurement nobody registered.

