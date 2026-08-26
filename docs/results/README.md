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

**Arm sets in `bench/dflash2_arena.py` added for the two-card work.** Named here
so nobody rebuilds one that exists — `--arms <name>`:

| set | what it compares | answered |
|---|---|---|
| `dual-gpu` | one card vs both, `ngram-mod`, layer split | yes — and its decode figure is [retracted](../reports/CORRECTIONS.md) §32 |
| `dual-gpu-nospec` | the same with speculation **off**, so the rate cannot follow the text | yes, **+1.5 %** |
| `dual-split` | `layer` vs `-sm tensor` vs `-ts 1,1` | yes, **tensor +59.5 %** |
| `dual-ubatch` | `-ub` 128 / 256 / 512 / 1024 on the tensor split | yes, **1024, +10.1 % prefill** |
| `dual-kv` | `q4_0` vs `q8_0` KV on the tensor split | yes, **q4_0 — q8_0 cannot load at depth** |
| `dual-depth` | the split, at the served 147,456 | yes, **tensor +65.4 %** |
| `dual-decoder` | `ngram-mod` vs none vs `draft-mtp` at depth | yes, **ngram-mod; MTP needed the computed `-ts`** |
| `dual-drafter` | tensor+`ngram` vs layer+`DFlash2` vs layer+`ngram` | yes, **tensor −29.2 % over layer; DFlash2 fails at depth** |
| `dual-mtp` | `ngram-mod` vs `draft-mtp,ngram-mod` vs none, on the served config | **partly — MTP voided, it copies the prompt** |

**Instrument added the same week:** `bench/gpu_device.py` and
`scripts/Get-GpuVram.ps1` are the only two places that ask the driver about a
GPU, pinned by UUID; a test forbids `--query-gpu` anywhere else. See
[CORRECTIONS §33](../reports/CORRECTIONS.md).

**Tuned 2026-08-26, issue #52 — the two-card configuration, `UD-Q4_K_XL`:**

| lever | verdict | raw |
|---|---|---|
| `-sm layer` vs **`-sm tensor`** | **tensor, +59.5 % at 16,384 and +65.4 % at 147,456** [+64.2, +67.3]. Also leaves 5,313 MiB free against 2,827. **EXPERIMENTAL in llama.cpp's own help** | `dual-split-16384.jsonl`, `dual-depth-147456.jsonl` |
| `-ts` ratio | **no lever.** `-ts 1,1` against the free-VRAM default of 41:59 is +1.8 % [+0.6, +4.1], inside the floor | `dual-split-16384.jsonl` |
| `-sm row` | **cannot load.** `device CUDA0 does not support split buffers` | `logs/dflash2-both-row-*.log` |
| `-ub` 128 / 256 / 512 / **1024** | **1024.** Decode flat; **prefill +10.1 %**, ranges do not overlap | `dual-ubatch-16384.jsonl` |
| KV `q4_0` vs `q8_0` | **q4_0 stays.** q8_0 is free at 16,384 (−0.3 %) and **cannot load at 147,456** — `cudaMalloc failed: out of memory` on the 12 GB card | `dual-kv-16384.jsonl` |
| `-mg` | **not applicable.** It selects a card for `-sm none` or `-sm row`; neither is in play | llama.cpp `--help` |
| `--fit` under `-sm tensor` | **inert.** `llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort`. `-ngl auto` still gives 66/66 | `logs/dual-profile-boot-verify.log` |
| Noise floor at **147,456** | **under 2 %** per arm across three boots | `dual-depth-147456.jsonl` |
| Tuned Q4 on two cards vs served Q2 on one | **parity.** 32.4/33.9/32.3 against 32.1/32.0/32.0, ranges overlap. Before the split was tuned the same comparison said −34 % | both |
| **Does the tuned dual profile work on the developer's machine?** | 🔴 **NO, until 2026-08-26 evening.** `serve-dual-lan.bat` decoded at **0.38 tok/s** -- `-sm tensor` splits EVENLY without `-ts`, and the 12 GB card is the DISPLAY GPU, leaving **+317 MiB** and spilling to host memory. Now computed at launch from measured free VRAM with a reserve on the display card: **25.8 / 42.7 / 78.3 tok/s, both cards at 95 %** ([CORRECTIONS 33](../reports/CORRECTIONS.md)) | `logs/serve-20260826-232107.log`, `logs/bat-dual-fixed.log` |
| Does **DFlash2** work on the dual split? | **No.** `draft-dflash` aborts at `ggml-backend-meta.cpp:1522` exactly as `draft-mtp` does. **No external drafter loads under `-sm tensor`** -- the Meta backend cannot host a second model. `ngram-mod` needs no weights and is the only speculative option | `logs/dflash-dual.log` |
| Can the tuned profile actually be started? | **yes** — `.\serve.ps1 -Dual`, booted end to end, 66/66 on the Meta device, `/health` ok, a real completion answered | `logs/dual-profile-boot-verify.log` |
| decoder on the tuned dual config, ctx 147,456 | **`ngram-mod`, and it is the only one that works.** `none` is -13.3 % [-13.8, -13.1] with each arm spreading 2.1 %; **`draft-mtp` CANNOT LOAD under `-sm tensor`** -- `GGML_ASSERT(bufs.back() != nullptr)` in `ggml-backend-meta.cpp:1522` | `dual-decoder-147456.jsonl` |
| the floor a verdict was compared against | **now printed.** `NOISE_FLOOR_PCT` is 13.6 -- Ada at ctx 16,384 -- and it called that tight -13.3 % "within noise". The constant is unchanged; the report states it and each arm's own spread, and names the third state | `harness.observed_spread_pct` |
| **How deep can the context actually go?** | **262,144 -- `n_ctx_train` -- with `-UBatch 512`.** Verified by pushing a **135,233-token request** through each rung, because loading is not surviving: 262,144 at `-ub 512` once loaded, answered `/health`, then died on the first real request with `CUDA error: out of memory ... cuMemSetAccess`. Survivors, free MiB display-card/other after the request: 147,456→1,998/2,040 · 196,608→1,248/1,208 · 229,376→1,071/500 · **262,144 @ ub512→821/452** | `logs/survive-c*.log` |
| Is 262,144 comfortable? | **No.** The run that died had 336 MiB free on the second card, the one that survived had 488 -- the line sits between them and the desktop decides which side you land on. 147,456 finishes with about 2,000 MiB on each card | same |
| Is `UD-Q4_K_XL` better than `UD-Q2_K_XL`? | **UNMEASURED HERE.** The only remaining argument for the switch, and it rests on an external ladder | — |


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

