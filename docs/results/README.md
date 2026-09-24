# tested — the register of what has actually been run

[34 — ThinkingCap Q4_K_M vs GSQ](34-thinkingcap-vs-gsq-quality-time-2026-09-24.md):
same protocol as 33. Same prefill/decode as GSQ, ~30 % fewer output tokens, 2–3× less
thinking on code1. code1: ThinkingCap 1/2 (one wrong self-written test), GSQ 2/2. PAL:
ThinkingCap 1/2 accepted, GSQ 0/2 (GSQ 1/4 across results 33+34). At least as good as GSQ
at the same speed on this n; depth > 65K untested.

[33 — Flash-Next vs GSQ quality/time](33-flash-next-vs-gsq-quality-time-2026-09-24.md):
code1 and PAL at 65,536, ABBA in one sitting, client 2.1.281. code1 all pass; Flash-Next
1.6× slower. PAL: GSQ 1/2 accepted (gsq-b fails the overlap audit — result 25's "GSQ
alone passes" was one sample), Flash-Next 0/2 (one audit fail, one no red→green); 3.8×
slower. No quality gain observed at n = 2; depth > 65K untested.

[32 — Flash-Next optimisation study](32-flash-next-optimization-2026-09-23.md):
`-sm tensor` disabled op offload (prefill 24 → 450–590 tok/s under `-sm layer`); VNNI Q2_0
decode ×1.8; GPU expert cache +28–36 %; 262K needs ub 512 (compute buffer ~ub × ctx per
card); mmap working-set trim frees ~15 GB RAM and stops decode collapses. Shipped hub I
(262K) / J (128K). `ngram-mod` 12/16/32 ~2× on copy-heavy output, −13 % on fresh text,
not shipped. MTP next. Eight instrument faults recorded.

[29 — PAL #149 run-journal task partial campaign](29-pal-journal-2026-09-22.md):
Second real-project task selected from PAL issue #149. Swift Q4 reached Claude Code
but timed out at1800.656s before producing candidate files; seven other arms were
not run. No quality score or model ranking; original project unchanged.

[30 — PAL #149 focused GSQ and EXL3](30-pal-journal-focus-2026-09-22.md):
GSQ9/9 hidden,40/40 visible,RED→GREEN confirmed,ACCEPTED; EXL39/9 hidden,
21/21 visible but RED→GREEN unconfirmed,FAILED_CONTRACT.742.688s and1024.859s.
This is one task only; the five-task Quick Discrimination Campaign remains incomplete.

[28 — FP8 OpenAI-path repetition](28-fp8-openai-pal-repeat-2026-09-22.md):
Same PAL task through `/v1/chat/completions` with shared observations. Original8/8,
visible12/12, overlap2/4, RED→GREEN confirmed;288.750s attempt.33 complete
requests:44,353 output tokens,24,688 reported reasoning tokens,188.86 effective
request tok/s. This is a new route repeat, not a replacement for result26.

[27 — Comparable gateway/local metrics](27-comparable-gateway-local-metrics-2026-09-22.md):
DSH/OpenAI versus benchmark Anthropic route diagnosis and shared stream observation
schema. DSH's reasoning_content path is verified; FP8 Anthropic result is not a
model-wide no-thinking result.33 observer +106 integration tests and full gate2046
passed; no fresh PAL repetition launched in this slice.

[26 — FP8 through controlled claude-9arm](26-fp8-pal-control-2026-09-22.md):
Same PAL task, provider reports `vllm/Qwen/Qwen3.8-27B-FP8`. Original8/8,
overlap2/4, visible10/10;169.297s original-verifier attempt, not accepted.
Actual test-first cycle observed but strict identical-test-hash gate fails after
correcting an assertion. No emitted nonempty thinking; remote engine/weights unknown.
Original PAL unchanged; independently rehashed2,083 evidence files with no mismatch.


[25 — Remaining models on the identical PAL workflow](25-remaining-pal-workflow-2026-09-21.md):
Dirk Q6, GSQ IQ3, actual NVFP4 VERY-LOW and EXL3 H5. GSQ alone passes all12
original+overlap cases among eight admitted configurations; other new rows pass8/8
original but fail0/4 overlap. EXL3's first cleanup-excluded submission passed12/12;
its admitted replacement did not. Both retained. Real isolated TDD, no original
project changes; single-task evidence, not a stable ranking.


[24 — Q4 and real PAL workflow](24-q4-real-project-workflow-2026-09-21.md):
clean post-Minecraft runs. Swift/TURBO Q4 both pass code1; both execute test-first
PAL work but fail explicit/user-path overlap audit. Fresh Q6 controls do not
fully fix it. Original PAL worktree unchanged; Docker test execution; all failed
and invalid costs retained. Original8-case passes are not full task acceptance.

[23 — Preliminary Claude CLI quality/time](23-preliminary-cli-quality-time-2026-09-21.md):
real restricted CLI, one code1 task per artifact at65536, now six operating points.
All pass7/7 hidden tests. Task seconds: TURBO56.500, Swift58.781, GSQ74.531,
EXL390.218, Dirk136.860, default NVFP4141.735. Full emitted-thinking/final review
found EXL3's functionally correct final has severe malformed Thai and Dirk has a
Vietnamese phrase; one shallow sample is not a stable winner or Long Horizon result.
First adapter-fault attempt remains retained as invalid.

[22 — Four actual artifacts](22-four-model-comparison-2026-09-20.md): Swift Q6_K,
TURBO MTP-Q6_K, Dirk UD-Q6_K and GSQ IQ3_S-MTP; common65536 allocation,
three rotated short-task validation rounds and a real-source ~45K prompt screen.
All pass6/6 coding attempts, but Thai repair and thinking costs differ. Selected
tested configurations, not globally best configs or a long-horizon promotion.

[21 — Sharp template preliminary matrix](21-three-candidate-screen-2026-09-20.md):
Stock/Sharp × medium/xhigh on NVFP4, one exploratory round; not actual Dirk weights.

[20 — Time per verified task and quality](20-gsq-time-quality-2026-09-20.md):
three rotated rounds at common147456 allocation, medium effort, original-session
history. EXL3 accepts6/6 held-out code attempts, GSQ5/6, NVFP4 3/6; NVFP4 has
the fastest cold prefill, EXL3 the lowest observed warm verified-work cost.
Full thinking and failures retained. Bounded single-answer comparison, not a
full agent benchmark; no default promotion.

[19 — GSQ initial screen](19-gsq-screen-2026-09-20.md): IQ3_S-MTP loaded and
completed original-session replay, with Thai/Han defects. Its language-only stop
is superseded: evaluation resumed on time per verified task plus quality at a
common context, including thinking and measured mitigation (CORRECTIONS 52).

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
| [`10-other-engines.md`](10-other-engines.md) | **has an engine other than llama.cpp been run on these weights.** ExLlama3 (Mia-AiLab fork) built and measured 2026-09-03: 51.5 tok/s at 14K on one card, 9–13 at 144K on two; fp8 KV +63 % over NVFP4 KV; single-card ceiling below 98K (issue #71). Since 2026-09-04 the daily driver at 262,144 (turboderp SC 4.0bpw H5, ~81 % of llama.cpp decode paired); the server carries its own watchdog, loop guard and Han ban (#75 #76 #77, last section) |
| [`11-quality-bench-2026-09-05.md`](11-quality-bench-2026-09-05.md) | **Three-artifact quality bench through Claude Code** (2026-09-05): one Thai brief, 4.0bpw H5 / 3.5bpw / NVFP4-LOW at each profile's largest window, with and without skills, eight executable checks run by the runner plus a reader's score and per-task server timings. One sample per cell. **3.5bpw writes broken Thai** (the model diagnosed it itself); 4.0bpw and NVFP4-LOW do not. Without skills all three omit OG and dark mode; with skills 4.0bpw scores 8/8. Afternoon: four isolated skill sets × 3 reps on 4.0bpw — **gate-only 8/8 ×3, design-suite-only 4/8 ×3, none 6/8 ×3, both 8/8 ×3** — the checklist moves the model, the knowledge alone lowers it. Evening: the `qwen38` family's pairs — `qwen38-think` turns "ask and stop" / "ship it as done" into a written pushback 4/4 on four planted briefs (no skill: 3/4 each); the code gate's only measurable effect is RED-before-GREEN (2/6 vs 0/3), the model's code being at the fixtures' ceiling either way. Generated from `qwen38-tuning/results/quality-2026-09-05/` by `tools/quality-report.py` |

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
| KV `q4_0` vs `q8_0` | **q4_0 stays** on speed — q8_0 is −0.3 % at 16,384, inside any floor. 🟡 **The "cannot load at 147,456" half is CONFOUNDED, flagged 2026-08-27:** that run's `Meta() model buffer size` is 8,065.29 MiB, exactly half the model, so it used the EVEN split — the configuration retracted in [CORRECTIONS 33](../reports/CORRECTIONS.md). The arithmetic says it would probably fit with the computed `-ts`. Not re-run — [results 03](03-memory-and-kv.md) | `dual-kv-16384.jsonl` |
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
| **Do the .bat files themselves work?** | 🔴 **NO, all four were dead on arrival — fixed 2026-08-29.** They carried `echo PowerShell 7 ^^(pwsh^^) was not found` inside an `if errorlevel 1 ( ... )` block. In cmd, `^^` is an escaped caret, so the `(` after it opens a block cmd cannot close, and cmd parses the whole block before running any of it: **`was was unexpected at this time`, before a single command ran.** Every test passed, because each one read the file as text or invoked `serve.ps1` directly — **neither of those is cmd.** Now `serve-dual-nvfp4.bat` boots to `n_ctx 147456` and answers a 70,322-token request at 40.14 tok/s. Guarded by `test_a_bat_must_parse_in_cmd.py`, which runs **every** `.bat` through real cmd with the launch line neutered | `bench/tests/test_a_bat_must_parse_in_cmd.py` |
| **How deep can NVFP4 actually be SERVED?** | **200,704, not the 229,376 first recorded** — [CORRECTIONS 35](../reports/CORRECTIONS.md). 229,376 was certified with a 65,643-token request, a **quarter** of its window; given a half-window one it loads with **206 MiB** free on device 1 and dies with `cudaMalloc failed: out of memory`. This project had already measured 336 dying and 488 surviving, and 206 is below both. `serve-dual-nvfp4-deep.bat` serves 200,704, verified end to end: 101,029-token request answered, 1,009 / 692 MiB free | `logs/serve-*.log`, [results 02](02-decoders.md) |
| **Does the split-mode verdict survive the artifact change?** | **YES — and it is the only verdict this session tested that did.** On NVFP4 at 147,456 with the served decoder, `-sm layer` is **−31.0 % [−32.9, −29.6] RESOLVED**: 31.3 / 31.4 / 30.1 against tensor's 44.5 / 45.2 / 44.9, baseline spread 1.6 %, **both arms `66+0`** so it is the split and not a spill. The `+65.4 %` it replaces was `UD-Q4_K_XL`, 2026-08-26, speculation OFF on both sides. **The sampler question is answered too:** `set_sampler: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU` is **absent from every layer log** — backend sampling is live there — **and layer is still 31 % slower**. Also acceptance 58.8 vs 45.4 and `ngram-mod` accepted length 17.54 vs 5.88, so the two splits are not decoding the same tokens ([CORRECTIONS 32](../reports/CORRECTIONS.md)) | `dflash2-arena.jsonl`, [results 02](02-decoders.md) |
| **How deep can the context go WITH vision?** | **200,704 — the profile's cap — and every rung below it.** Each was asked for a half-window request from the arena's corpus AND an image on top of that context, which is what pasting a screenshot into a long conversation does; the earlier probes were a small picture against an empty context and proved nothing about this. All four passed: 200,704 took 91,428 tokens then a picture, finishing **464**/1,187 MiB free · 180,224 → 534/1,703 · 163,840 → 817/2,057 · 147,456 → 1,068/2,413. ⚠️ **464 MiB is the thinnest margin of the four**, between the 336 seen dying and the 488 seen surviving on a *different* configuration. The budget check refusing to start is the safety, not the margin. The ladder stops at the cap: 229,376 dies without the tower already | `logs/vl-*.log`, [results 02](02-decoders.md) |
| **Does the vision projector load under `-sm tensor`?** | **YES, and the prediction that it would not was WRONG.** The reasoning — a projector is a second model and `-sm tensor` has never hosted one, `draft-dflash` aborts in `ggml-backend-meta.cpp` — does not carry to `mmproj`. `-mm mmproj-BF16.gguf` on the **unpatched served binary** loaded and answered a real 512×512 PNG correctly at **65,536, 147,456 AND 200,704**: *"Blue fills most; a yellow circle is in the middle."* Free after: 2,465/4,230 · 1,205/2,450 · **614**/1,294 MiB. ⚠️ **Those are after a TINY request** — vision beside a large text prompt is untested, and 614 MiB sits between the 488 that survived and the 336 that died. Without `-mm`, any image is HTTP 500 `image input is not supported`. **Now folded into `serve-dual-nvfp4{,-lan}.bat` rather than a separate icon** — it costs headroom, not window, and that pair was booted through cmd and shown a picture it had not seen before: `n_ctx 147456`, correct answer, text path still fine, free 1,206 / 2,450 MiB. The deep pair stays text-only | `logs/vision-*.log`, [results 02](02-decoders.md) |
| **Can the NVFP4 pair actually be started, end to end?** | **YES, booted 2026-08-29.** `worker-q4-dual.ps1 -Nvfp4` came up, announced itself as `Qwen3.8-27B-NVFP4-MTP` on `/v1/models`, and **survived a 70,322-token request, generating 200 real tokens** — prose, not a copy of the prompt. Decode read **32.56 tok/s**, prefill 717.2, finishing with 1,787/1,870 MiB free. ⚠️ **That rate is a boot check, not a measurement** — one unpaired reading at `n_predict` 200 against the sweep's 512, taken with the desktop holding more than it did during the sweep. **The number to quote is the paired +63.1 %**, not this. **Two faults found by running it:** the profile's budget guard correctly REFUSED a second boot while a leaked server still held both cards (it works), and the first probe fed a prompt with no instruction and got **1 token and empty content** — a fourth ad-hoc probe differing from the arena that already exists. The verification now uses `arena.filler(int(ctx*0.5), 'real-code-vendor')`, the sweep's own slice | `logs/verify-nvfp4-boot.log` |
| **Is there anything faster than what we serve at 147,456?** | **YES — NVFP4 with a baked-in MTP head, +63.1 % [+58.3, +65.6] RESOLVED.** `NVFP4-MTP-VERY-LOW` + `draft-mtp,ngram-mod` at `n-match 24` decodes 39.4 / 42.6 / 42.6 against 24.9 / 25.7 / 25.7 for the served `UD-Q4_K_XL` + `ngram-mod` n-match 12, three paired rounds rotated, baseline spread 3.3 %. **No patch, no sidecar drafter, served binary**, and it leaves ~400 MiB MORE headroom. **Quality UNMEASURED and it gates shipping — no default changed** — [results 02](02-decoders.md) | `nvfp4-final-147456.jsonl` |
| Is NVFP4 fast because of the artifact? | **No — the artifact ALONE is a loss.** NVFP4 + `ngram-mod` without MTP is **−22.4 %**: n-gram acceptance falls 55.4 → 22.1 because that file writes text the n-gram cannot predict. MTP fills exactly that gap. **The pairing is the result, neither half is** | `nvfp4-vs-q4-147456.jsonl` |
| Does the n-gram tuning transfer across artifacts? | **No.** On NVFP4, `n-match 24` is **+27.1 % over the 12 that won on `UD-Q4_K_XL`**, and `map-k` — which declined **100 %** of its drafts on the Q4 at this depth — recovers to **+15.4 %**. A verdict does not transfer across artifacts any more than across depths | `nvfp4-ngram-retune-147456.jsonl` |
| Does MTP's prompt-copying belong to MTP? | **No, to the ARTIFACT.** NVFP4's head reports `copied_frac [0.0, 0.0, 0.0]` and `predicted_n 512` in every round where Unsloth's head at the same depth reports `[0.519, 0.0, 0.23]`. Open for weeks; answered | `nvfp4-vs-q4-147456.jsonl` |
| DFlash2 on NVFP4 | **~~no case~~ — WITHDRAWN 2026-08-30. It works.** The `+0.2 %` arm was given none of what DFlash2 wants: ctx 147,456, `--spec-draft-n-max 3`, and `n-match 12` — the window the rows above record collapsing on this artifact (55.4 → 22.1) while 24 wins. **Re-measured at 65,536 / `n_max` 4 / `n-match` 24: +67.9 % [+65.8, +71.5] RESOLVED**, acceptance back to 50.0. At the served 147,456: 44.48 / 44.56 / 44.23 against MTP's pooled 42.77 over six rounds and two boot series spanning 9.3 % — **+4.0 %, under the floor and across boots so NOT resolved**, but DFlash2's worst round beats MTP's best. It still costs a sidecar drafter, the mirror patch, a binary we do not serve and ~950 MiB more headroom than MTP; **what it buys is consistency, not speed** | `nvfp4-dflash-65536.jsonl`, `nvfp4-dflash-147456-n4.jsonl` |
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

## 12 — the skill loop, and a change that was refuted

[`12-skill-loop-2026-09-07.md`](12-skill-loop-2026-09-07.md) — twenty paired `code2`
cells on 4.0bpw, testing one change to `qwen38-code-gate`. It scored **better on the
gate** (red-green 9/10 vs 7/10) and **worse on the property the gate stands for** (tested
before touching source, 1/10 vs 5/10, p = 0.070). Not shipped.

Two variants were tried and **both** made the model *less* likely to test first (40 % ->
10 % and 0 %); the untouched skill is the best of the three. Neither is shipped.

**The reusable finding is about the instrument.** `red_then_green` asks only whether some
test failed and a later one passed, which is what iteration looks like when the first
attempt is buggy — in all twenty runs the model wrote every source file and the README
before its first line of test. And at n = 5 the change read 5/5 against 3/5 and looked
like a win; n = 10 removed it.

## 13 — the code gate against a placebo

[`13-skill-vs-placebo-2026-09-08.md`](13-skill-vs-placebo-2026-09-08.md) — 60 cells on the
9arm FP8 gateway, no GPU, four arms including a **content-free placebo skill** carrying the
same "load this and follow it" wording as the real arms.

**Tested before touching source: `noskill` 0/15, `placebo` 0/15, `codegate` 15/15,
`family` 14/15.** Against the placebo, p = 6.4e-09 and 1.0e-07; **placebo vs noskill
p = 1.00**, which is the check that makes the rest believable — a skill that says nothing
does nothing, though it loaded.

**Replicated on a second fixture (`code1`)** — pooled over 100 cells: `noskill` 0/25,
`placebo` 0/25, `codegate` 25/25, `family` 24/25, p = 7.9e-15 against the placebo. The
separation is complete: 49 of 50 with the skill, 0 of 50 without.

**Which half does the work:** a variant carrying the brief table alone (checks 1-6 removed)
scores **1/10** on tested-first, not distinguishable from the placebo (p = 0.50), while the
full skill scores 10/10 (p = 6e-05). Enumeration gets the deliverable delivered; only an executable check changes
the **order** the work is done in. The token cost is therefore not trimmable by dropping the
commands - they are the active ingredient.

It does **not** buy correctness (hidden tests are at the ceiling in every arm) and it is not
free: 2.4x the output tokens and 2.9x the wall clock. The 40 % the same skill scored on EXL3
the day before ([results 12](12-skill-loop-2026-09-07.md)) is **not** shown to be a
quantisation effect — the two briefs differ, and separating them needs the GPU.

## 14 - the design arms, and a page written in the wrong language

[`14-frontend-arms-2026-09-08.md`](14-frontend-arms-2026-09-08.md) - nine page cells on the
FP8 gateway, no GPU. `gateonly` and `both` score **8/8 in every run**;
`designonly` scores 5/8 and 7/8.

**The headline is not the score.** `designonly` wrote the page in **English in 3 of 3** for a
Thai brief (0 Thai characters against 1,300-2,000 elsewhere), shipped **no dark mode in any
run**, and in **2 of its 5 runs produced no file at all** - 19 and 22.7 minutes, rc=0, the
answer ending mid-sentence on "create a single file:". It also and broke the two-font cap that lives in `design-rules` - the very skill
it loads. No skill in the design suite mentions language at all; only the gate does. But the
developer's own 26 `using-design` pages are **21 of 26 with Thai**, so the behaviour is
**unguarded rather than absent** - it usually works, and here it failed twice out of two. That is the same shape [results 13](13-skill-vs-placebo-2026-09-08.md) found in the
coding track on the same day: prose states a rule, a command enforces it.

**It says nothing about whether the pages are pretty.** The arm the developer prefers to look
at is the one scoring worst here, and both things can be true; the six pages are kept on disk
so that judgement can be made by someone who has it.

## 15 - the first shipped skill change

[`15-using-design-fix-2026-09-09.md`](15-using-design-fix-2026-09-09.md) - three rules added
to `using-design`, each aimed at a failure [results 14](14-frontend-arms-2026-09-08.md)
measured: write the file first, answer in the brief's language, ship light **and** dark.

| | page | Thai | `prefers-color-scheme` | gate |
|---|---:|---:|---:|---|
| before | 3/5 | 0/3 | 0/3 | 5, 7, 6 /8 |
| after | **8/8** | **7/8** | **3/4** | **8, 8, 8**, 6 /8 |

Thai p = 0.024; `designonly` reached **8/8 three times having never passed 7/8 in 7 earlier
runs** (p = 0.024). Dark alone p = 0.071 and the no-file fix p = 0.13 - right direction,
underpowered. **One run in four ignored both rules**, so a written rule raises the rate and
does not make the behaviour certain. Live in the installed skill; not yet in the library.

## 16 - the think skill, and a control that was not inert

[`16-think-vs-placebo-2026-09-09.md`](16-think-vs-placebo-2026-09-09.md) - `qwen38-think` on
an impossible ask (`think-task-4`), against a placebo and no skill, on the FP8 gateway.

| arm | pushed back | claimed done | mean wall |
|---|---:|---:|---:|
| `noskill` | **1/5** | **1/5** | **29.9 min** |
| `placebo` | 3/4 | 0/4 | 13.8 min |
| `think` | **5/5** | 0/5 | **3.8 min** |

`think` vs `noskill` **p = 0.024**; `think` vs `placebo` **p = 0.44 - not distinguishable**.
One no-skill run spent **30.5 min and 103k tokens reaching for `ctypes`** to defeat the
constraint, then declared the work done. **Here a skill is cheaper than no skill** - the
reverse of the code gate.

**The placebo was the wrong control**: written to be inert for the code-gate test, it discusses
scope and "belongs in a note", which is close to the behaviour under test. So `think` beats no
skill and is not yet separated from "a skill was loaded". A pushback-inert placebo is next.

## 17 — the Thai sampler, and a tightened triple that changes nothing

[`17-thai-sampler-2026-09-15.md`](17-thai-sampler-2026-09-15.md) — 48 paired Thai
prose requests on profile G (8 prompts × 2 arms × 3 rounds, one boot), default
sampler against `0.3 / top-k 10 / top-p 0.9` on Thai-heavy prompts.

| arm | broken script tokens | Thai chars |
|---|---:|---:|
| default | 0 | 5,425 |
| tightened | 0 | 5,496 |

**No effect; not shipped** — tightening is opt-in only (body `"thai_sampler": 1`).
Caveat: the base rate on short prose is ~0 in both arms, so this pair cannot
separate the arms on the incident regime (long agentic session). The run also
caught two instrument faults: a substring counter that flagged correct
`เว็บไซต์` (`เว็บไซต(?!์)` now), and rows that stored counts without text.

## 18 — Thai repair, algorithm-only

[`18-thai-repair-2026-09-16.md`](18-thai-repair-2026-09-16.md) — all 8 repair
candidates measured on two real YT Downloader sessions (EXL3 4.0bpw + FP8):
dict v2 (49 pairs) goes **40 broken → 0, FP 0, ~0.02 ms CPU/block**; spell
REJECT (FP + 2.9 s/sentence); SymSpell conditional but redundant under dict;
segmentation/Viterbi and Typhoon-CPU parked; every loop shape trips the
existing guard (`บริบ` period-4 at char 574 of 24,508, replayed).
