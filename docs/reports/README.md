# Qwen3.8-27B Local Coding Worker — Documentation Index

> ### → **[START-HERE.md](START-HERE.md)** ←
>
> **New to this project? Read that one document.** It covers what was tried,
> what it cost, what was learned and what is still open, from the beginning,
> in one place. Everything below is the detail behind it.

One document per topic. Read in this order; each builds on the one before.

| # | document | question it answers | status |
|---|---|---|---|
| 00 | [Q3 vs Q4 benchmark](00-Q3-VS-Q4-BENCHMARK-REPORT.md) | Which quant, at 16K? | complete |
| 01 | [Runtime tuning](01-RUNTIME-TUNING-REPORT.md) | Which server flags, and how much do they actually buy? | complete |
| 02 | [Context depth](02-CONTEXT-DEPTH-REPORT.md) | How deep can this machine go, and does the quant verdict survive? | complete to 128K, Q4 only |
| 03 | [Deep-context quality](03-DEEP-CONTEXT-QUALITY-REPORT.md) | Does Q8_0 KV damage retrieval at the depth where it helps? | complete |
| 04 | [Measurement methodology](04-MEASUREMENT-METHODOLOGY.md) | How to measure on this machine without fooling yourself | complete |
| 05 | [Operating guide](05-OPERATING-GUIDE.md) | What to actually run, and the rules that matter more than flags | complete |
| 06 | [Open questions](06-OPEN-QUESTIONS.md) | What is unmeasured, and what to do next | **rewritten after Experiment A** |
| 07 | [Skill routing](07-SKILL-ROUTING.md) | Which discipline governed which decision, and which rules did not hold | complete |
| 08 | [Candidate landscape](08-CANDIDATE-LANDSCAPE.md) | Which models in the new research actually exist, fit, and run on our binary | complete |
| 09 | [Agent-loop and system gates](09-AGENT-LOOP-AND-SYSTEM-GATES.md) | Tool-call, 100-turn stability and prefix-cost control values for Q4 | Q4 control complete |
| 10 | [Experiment A — Q2 vs Q4](10-EXPERIMENT-A-Q2-VS-Q4.md) | Does crossing the VRAM residency threshold beat Q4? | complete |
| 11 | [Depth on IQ2_XXS](11-DEPTH-ON-IQ2XXS.md) | 64K / 128K / 256K on the new default — and what 256K costs | throughput complete, quality open |
| 12 | [Dynamic V3 first results](12-DYNAMIC-V3-FIRST-RESULTS.md) | Unsloth republished the repo mid-session — what the V3 1-bit and 2-bit rungs actually do | stages 0–2 complete |
| 13 | [Cross-model results](13-CROSS-MODEL-RESULTS.md) | **Every other model and quant measured** — Ornith 9B/35B, Ternary Bonsai, 35B-A3B MoE, gpt-oss-20b, AtomicChat | complete |
| 14 | [Panel review](14-PANEL-REVIEW.md) | Three independent agents critiqued the test plan — what they found, what was fixed, what is still open | complete |
| 15 | [Test inventory](15-TEST-INVENTORY.md) | **Every model × quant × probe actually run** — one row per artifact, one column per probe, so a gap shows as a blank cell | complete |
| 16 | [Optimization surface](16-OPTIMIZATION-SURFACE.md) | **Complete catalogue of everything tunable** — 16 layers, all 248 runtime options accounted for including the ones judged inert, with predictions | complete |
| 17 | [External research review](17-EXTERNAL-RESEARCH-REVIEW.md) | What the reply to the research brief got right, wrong, and did not answer — recorded so it is not re-derived | complete |
| 18 | [Research round 2 review](18-RESEARCH-ROUND2-REVIEW.md) | The 10-workstream reply, verified flag-by-flag on this machine — six usable answers, every percentage discarded, the model table refuted | complete |
| 19 | [The 128K plateau](19-THE-128K-PLATEAU.md) | **At 128K every resident arm ties at ~27 tok/s** — weight size decides residency, not speed. Pick the largest that fits, not the smallest | complete |
| 20 | [16-layer results](20-SIXTEEN-LAYER-RESULTS.md) | **21 levers measured in one session** — n-gram decoding doubles throughput for free; eleven levers do nothing; four are harmful | complete |
| 21 | [Context ceiling](21-CONTEXT-CEILING.md) | How far past 128K each artifact stays fully resident — the residency half of the >128K goal | complete |
| 22 | [Session record 2026-08-20](22-SESSION-RECORD-2026-08-20.md) | **The arc of one nine-hour session** — 21 levers measured, four claims retracted, five instrument faults found, and n-gram at 128K going 26.5 → 81.5 tok/s | complete |
| 23 | [Session record 2026-08-21](23-SESSION-RECORD-2026-08-21.md) | **Latest.** n-gram re-measured on a fixed-text instrument and the winner **changes with depth** (`ngram-map-k` +135.89 % at 16K, `ngram-mod` +200.22 % at 128K); `ngram-cache` disqualified; `AD-IQ1_M` ruled out at 128K; four placement levers inert; two flat-constant harness faults fixed | complete |
| 24 | [Beyond 128K](24-BEYOND-128K.md) | **In progress.** Throughput past 131,072, which report 21 never measured. At 163,840 the fastest arm is NOT the fully-resident one: `-ot ssm` restores `65+0` and collapses speculative acceptance from 100 % to 4 % | paused 2026-08-21 |
| 25 | [IQ2_S at 131,072](25-IQ2S-AT-131072.md) | `--fit-target` against batch size at 131,072 — which one actually buys the layers | complete |
| 26 | [The cold start](26-COLD-START.md) | Where the first-request delay really comes from — **title half-retracted, see §16 and §17** | complete, partly retracted |
| 27 | [Prefill cannot be tuned](27-PREFILL-CANNOT-BE-TUNED.md) | Every prefill lever, and the IQ2_S/IQ2_XXS trade — **scoped to ctx ≤ 32,768; at 98,304 prefill collapses 15×** | superseded at depth |
| 28 | [Decoder recheck](28-DECODER-RECHECK.md) | The re-measurement that closed CORRECTIONS §8 | complete |
| 29 | [DFlash2 and the flattering prompt](29-DFLASH2-AND-THE-PROMPT-THAT-FLATTERED-NGRAM.md) | DFlash2 wins on real code by +34.7 % and loses on the prompt we had been using | complete |
| 30 | [RTX 3090 reference review](30-SYV-RTX3090-REFERENCE-REVIEW.md) | An external stack reviewed against our own files — **one claim retracted, §21** | complete |
| 31 | [Session record 2026-08-22](31-SESSION-RECORD-2026-08-22.md) | **Every measurement of that day with its conditions**, including the runs that turned out invalid | complete |
| 32 | [**Benchmark status brief**](32-BENCHMARK-STATUS-BRIEF.md) | **The standalone hand-off** — the whole state with every number traced to its file, written to be read cold | **current** |
| 33 | [**What the RTX 3090 pool actually gave us**](33-WHAT-THE-3090-POOL-ACTUALLY-GAVE-US.md) | **Eight techniques measured or closed in one session** — five wins, three retractions, no profile changed. The largest win was already switched on | **current** |
| 34 | [**Blackwell bought headroom, not speed**](34-BLACKWELL-BOUGHT-HEADROOM-NOT-SPEED.md) | **The rebuild for the card that is actually installed**, and the retraction it forced. Per arm the 5060 Ti is 1.1–1.3× slower than the 4070 SUPER — but `dflash2+ngram` went from a median of 5.66 with two timeouts to **87.72 with none**, because it stopped being squeezed. Every Blackwell-gated path in this build is FP4, and FP4 does not fit | **current** |
| 35 | [**Q2_K_XL, the MTP head that was already there, and the effort nobody had set**](35-Q2KXL-MTP-AND-THE-EFFORT-NOBODY-SET.md) | **Six configurations on one real task, zero files changed six times.** `UD-Q2_K_XL` carries `blk.64`, so `draft-mtp` runs with no sidecar and returns 743 MiB. `n_max 7` is +25 % wall clock on DFlash2 and −56 % on MTP, and the metadata says why. **Every server this project ever launched ran at `xhigh`;** `medium` is the default from now. A projection said ctx 163,840 would fit — a boot said 64/66 | **current** |
| ⚠ | [**Corrections register**](CORRECTIONS.md) | **Read before quoting any number.** Thirty published claims this project later contradicted with its own data, each with where the correction lives. **§25–§27 all share one shape: the conclusion was right and the stated mechanism was wrong**; **§28 does not — two correctly measured numbers were made false by being put in one table**; **§29 is a third shape — a correct answer filed against a wider question than it was asked**; **§30 is a fourth — a two-point hypothesis published in a commit message, a layer nothing scans** | current |
| ★ | [**Master report** (self-contained, for external review)](MASTER-REPORT-2026-08-19.md) | Everything above in one document that assumes no access to this machine | 2026-08-19 |
| — | [Session state](SESSION-STATE.md) | restart notes — the interrupted run it describes is now complete | superseded |

**Everything in reports 00–11 is the PRE-V3 generation.** Unsloth replaced every
file in `unsloth/Qwen3.8-27B-GGUF` in place on 2026-08-19T16:39:23Z, mid-session:
same filenames, new contents, new byte counts. Those reports remain internally
consistent and comparable to each other; they are not comparable to the current
repo. See 12.

> ⚠️ **The block below is the answer as of 2026-08-19 and is superseded.**
> Nothing runs that configuration now: all four worker profiles serve
> `UD-IQ2_XXS`/`UD-IQ2_S` **with** `--spec-type ngram-mod` and `q4_0` KV.
> For the current state read [32](32-BENCHMARK-STATUS-BRIEF.md). Kept as
> the historical answer.

**The one-line answer (revised 2026-08-19):** at 16K run **`UD-IQ2_XXS` with
speculation OFF** — fully GPU-resident, 3.2x Q4's decode, and the same 27/30 task
acceptance as Q4. Keep `UD-Q4_K_XL` with `--spec-type draft-mtp
--spec-draft-n-max 2` as the escalation lane, and — until 11 §4 is closed — for
any deep task whose success depends on **retrieval** rather than throughput.
At depth IQ2_XXS is 2-3x faster (11), but only Q4's deep *quality* is verified.
See 10 for the evidence and its limits.

**The residency cliff:** 33 GPU layers gives 12.6 tok/s, 61 gives 21.6, and 65
gives 42.4. The last four CPU layers cost about half the throughput. "Nearly
resident" is a different regime from "resident".

**The finding that outranks every flag:** the prefix cache is exact. Editing one
sentence above the append point costs a full re-prefill — 2.4 s becomes 11.5 s at
4K, and a measured 63 s at 16K / 248 s at 64K (10-point fit, r2 0.968).
See 05 for the rule and 09 for the curve.

**Depth guidance (revised):** 16K with F16 KV · 64K and 128K with `q8_0` KV ·
**256K now runs without host paging** on IQ2_XXS, at 1.71 tok/s behind an
11-minute cold prefill — a budget for one deep question, not an agent loop.
Retrieval quality at depth is verified on **Q4 only**; see 11 §4 before trusting
a low-bit artifact with a needle-in-100K-tokens task.

**Plans** live in `..\plans\` — `00-OPTIMIZATION-PLAN.md` (the original flag-tuning
programme, complete) and `01-V3-Q1-Q2-TEST-PLAN.md` (the current one: Qwen3.8-27B
Dynamic V3, 1-bit and 2-bit rungs).

**Newest first:** [31](31-SESSION-RECORD-2026-08-22.md) is the session record for
2026-08-22 — every measurement of that day with the condition it was taken under,
including the runs that turned out invalid. [30](30-SYV-RTX3090-REFERENCE-REVIEW.md)
reviews an external RTX 3090 stack against our own files;
[29](29-DFLASH2-AND-THE-PROMPT-THAT-FLATTERED-NGRAM.md) is the DFlash2 result and
the prompt that hid it.

[32](32-BENCHMARK-STATUS-BRIEF.md) is the **standalone hand-off** — the whole
state of the benchmark with every number traced to its file, written to be read
cold by someone with no context. Start there if you are not resuming a session.

Raw data and harnesses live in `C:\AI\qwen38-tuning\` —
`EXPERIMENTS.md` (E0–E13), `results\*.jsonl`, `bench\` (253 tests), `scripts\`, `logs\`.
