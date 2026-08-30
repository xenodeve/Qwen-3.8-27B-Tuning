# 32 — Benchmark status brief

**Written 2026-08-22 as a standalone hand-off for external research.** Everything
below names the file its number came from. Where a claim is reasoning rather than
measurement it says so. Read §6 before quoting any figure.

---

## 1. The apparatus

| | |
|---|---|
| GPU | one RTX 4070 SUPER, 12 GB (11,998 MiB usable) |
| Model | Qwen3.8-27B, `UD-IQ2_XXS` (Unsloth Dynamic V3), 65 layers |
| Drafter | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, 1.06 GB on disk |
| Server | llama.cpp **build 10499, commit `1deefcca3`** (PR #27342, DFlash2) |
| Source tree | `C:\AI\llama.cpp` — **`C:\AI\llama.cpp-dflash2` holds binaries only, no `common/`** |
| OS | Windows 11, WDDM |
| Served window | `worker-iq2s-quality.ps1` runs **ctx 98,304**, `-np 1`, `-lv 3` |

**Project metric is verified accepted coding tasks per hour, not tok/s.** Every
number in §3 is tok/s and is therefore a proxy, not the metric.

---

## 2. The question that drives everything

> 🔴 **SUPERSEDED 2026-08-23, after this brief was circulated. There IS a
> mechanism, and it is an instrument fault.** The worker was writing into
> `C:\AI` — the live repository — instead of the clone, so `git diff` in the
> clone was empty and the harness recorded "changed nothing" about work that may
> have been done correctly in the wrong tree. Reproduced deliberately and fixed:
> with the directory pinned on the argv the same task returns **`EDITED`, 251
> diff bytes, in 32.8 s**. Cause, evidence and what is retracted:
> [`CORRECTIONS.md` §24](CORRECTIONS.md). **Every conclusion below about the
> model, the quantisation or the workflow is withdrawn; the wall-clock and
> context high-water figures survive, because they came from the process and the
> server rather than from the diff.**
>
> **~~A second explanation is still live and independent of this one: decode
> collapses to 2.8–5.0 tok/s at ctx 98,304~~ — RETRACTED
> [CORRECTIONS §26](CORRECTIONS.md), 2026-08-23.** That collapse belongs to the
> DFlash2 arms, not to the window. The profile the worker actually runs
> (`ngram-mod` alone) returns **96.92 tok/s median at ctx 98,304, 6/6 rounds
> finishing** — faster than the 75.2 recorded at 16,384. **So the directory
> fault is now the only established explanation for the five zero-diff rows**,
> and the next real-task run has one variable, not two.


Five real GitHub issues were handed to a local coding agent backed by this
server. Raw: `qwen38-tuning/results/real-task-bench.jsonl`.

| issue | wall clock | how it ended | files changed |
|---|---:|---|---:|
| xeno-skills #306 | 2,400 s | **timed out** | 0 |
| xeno-skills #314 | 1,427 s | **exited rc=0** | 0 |
| openclink #144 | 1,646 s | **exited rc=0** | 0 |
| openclink #145 | 1,759 s | **exited rc=0** | 0 |
| openclink #149 | 2,400 s | **timed out** | 0 |

Baseline test suite green on all five (`verify_baseline_exit: 0`). Context
high-water **56,861–88,668** of 98,304, so `0 WINDOW_BOUND` — the window was
never the limit. `diff_bytes` is **0 on every row**.

**Three of five exited cleanly having changed nothing.** More time would not have
helped those; they stopped voluntarily. **No mechanism is attached.** The
worker's transcript was written to `<clone>.stdout.txt` and **deleted with the
scratch root**, so there is no record of what it did.

### Scale reference

- A real task in these repos: **openclink merged PRs add a median 259 lines
  across 5 files** (40 PRs sampled); xeno-skills median 78 lines, but that repo
  is mostly Markdown.
- The toy corpus this project has always measured on: **median 35 non-blank
  lines in 1 file** (`bench/_work/`, 43 files), accepted in **median 40.5 s**
  (`results/retry-bench.jsonl`, 445 rows with timings).
- **Derived, not measured:** at the worst decode rate ever recorded here
  (~44 tok/s), 2,400 s is ~105,000 tokens; a 259-line diff is ~3,000–4,000
  tokens. The model could emit the required change ~27 times over in the time it
  spent. **Raw generation capacity is not the limit for a task this size.**

---

## 3. Measured, with verdicts recomputed from raw data

Method throughout: paired within a round, arms rotated each round,
`harness.paired_deltas` — an effect is RESOLVED only if the mean clears the
**13.6 %** drift floor **and** the sign is consistent across rounds.

### 3.1 Prefill by depth — the largest single finding

Cold prefill (`task 0`, `prompt eval time` line), same metric at every depth,
parsed from `qwen38-tuning/logs/dflash2-*.log`:

| n_ctx | prompt tokens | boots | cold prefill tok/s (median) | range |
|---:|---:|---:|---:|---|
| 16,384 | 6,621 | 89 | **1,129** | 558–1,264 |
| 65,536 | 28,122 | 18 | **924** | 473–1,083 |
| **98,304** | **43,162** | 2 | **74.3** | 62–87 |

**A 15× collapse at the window actually served**, with every layer resident
(`offloaded 65/65 layers to GPU`). A 43,162-token prefill therefore takes about
**9.7 minutes**. Real tasks reached 56,861–88,668 tokens of context.

**~~Decode collapses with it~~ — RETRACTED
[CORRECTIONS §26](CORRECTIONS.md), 2026-08-23.**
`results/sweep-ngram-nmatch-98304.jsonl` did show **13 of 16 rows timing out**
against a 26.8-minute budget with the three survivors at **2.8, 5.0 and
4.2 tok/s**. Those numbers are real. **What was wrong is attributing them to the
window: all sixteen rows ran `--spec-type draft-dflash,ngram-mod`**, so depth
and drafter never varied independently. Six paired rounds, same ctx, same
corpus, arms alternated (`results/decoders-98304.jsonl`, 24 rows):

| arm | ok | timed out | median tok/s | free MiB after load |
|---|---:|---:|---:|---|
| `none` | 6/6 | 0 | 33.69 | 800–1,935 |
| **`ngram-mod` — the decoder every `worker-*.ps1` runs** | **6/6** | **0** | **96.92** | 769–2,117 |
| `dflash2` | 5/6 | 1 | 49.31 | **45–376** |
| `dflash2+ngram` | 4/6 | 2 | 5.66 | **153–240** |

**The window is fine and always was** — 96.92 is above the 75.2 median at
16,384. **Read the artifact with it:** these rows are `UD-IQ2_XXS` at ctx
98,304, which **no profile serves** — `worker-iq2xxs-deep` runs that artifact
at 131,072, `worker-iq2s-quality` runs 98,304 on the 1.1 GB larger
`UD-IQ2_S`. The decoder verdict transfers; the absolute rate does not. The clause *"neither residency nor speculation
explains it"* was exactly backwards: speculation explains it, and the sweep held
it fixed. The prefill table above carries the same caveat, since its 74.3 was
measured with the drafter loaded too.

**A real task needs a median 259 added lines. At 4 tok/s that is hours.** That
is **one of two independent reasons** the five real tasks produced nothing
*where the harness looked* — the other is the directory fault in
[`CORRECTIONS.md` §24](CORRECTIONS.md). **Fixing either does not fix the
other**, and the next real-task run must not change both at once.

⚠️ **`prompt eval time` is time-to-first-token from slot launch, not pure
prefill** — see §5.2. For a cold request the difference is negligible (5,821 ms
against ~65 ms of bookkeeping); for a cached one it is the whole number.

### 3.2 `--spec-ngram-mod-n-match` — the optimum moves with the window

ctx 16,384, `results/sweep-ngram-nmatch.jsonl`, 12 rows, corpus `5672a9bcce74c0d0`:

| arm | rounds (tok/s) | vs shipped `12` | verdict |
|---|---|---|---|
| `24` (llama.cpp default) | 94.5, 96.3, 94.2 | **+34.6 % [+31.4, +40.8]** | **RESOLVED** |
| `16` | 69.2, 69.7, 69.5 | −1.5 % [−4.9, +3.9] | within floor |
| `12` (shipped) | 71.7, 73.3, 66.9 | baseline | — |
| `8` | 56.7, 62.7, 61.5 | −14.5 % [−20.9, −8.0] | RESOLVED |

ctx 65,536 on the deep corpus, `results/sweep-ngram-nmatch-65536.jsonl`, 12 rows,
corpus `1a3ae4b813dd8447`, every arm `65+0`:

| arm | rounds (tok/s) | vs shipped `12` | verdict |
|---|---|---|---|
| `24` | 44.8, 53.0, 42.9 | −9.7 % [−29.1, +16.8] | within floor |
| **`16`** | **91.9, 88.5, 83.4** | **+67.5 % [+45.3, +95.1]** | **RESOLVED** |
| `12` (shipped) | 63.3, 45.3, 51.4 | baseline | — |
| `8` | 52.8, 47.3, 35.4 | −14.5 % [−31.1, +4.3] | within floor |

**The ranking inverts.** 24 wins at 16,384 and is a null at 65,536; 16 is a null
at 16,384 and wins at 65,536. **The value we ship (`12`) loses at both.**

Per-implementation counters at 65,536 explain it — the binding constraint at
depth is **fire rate**, not key collision:

| arm | ngram drafts | ngram decline | ngram mean acc len |
|---|---:|---:|---:|
| `24` | 18 | 97.0 % | 19.78 |
| `16` | 39 | 91.3 % | 21.59 |
| `12` | 22 | 96.4 % | 11.68 |
| `8` | 43 | 92.7 % | 9.12 |

### 3.3 The two winners cancel

`results/sweep-draft-n-x-nmatch.jsonl`, 12 rows, ctx 16,384, all arms `65+0`:

| arm | rounds | vs base | vs `n7 m12` | vs `n4 m24` |
|---|---|---|---|---|
| `n4 m12` (shipped) | 74.6, 75.2, 75.2 | baseline | — | — |
| `n7 m12` | 94.4, 95.1, 94.9 | +26.4 % RESOLVED | — | — |
| `n4 m24` | 98.4, 97.7, 97.4 | +30.5 % RESOLVED | — | — |
| **`n7 m24` both** | 65.5, 63.4, 65.5 | −13.6 % | **−31.6 % RESOLVED** | **−33.8 % RESOLVED** |

Both singles **replicated** on different boots. Stacked they reach **52.4 % of
the independent expectation** and are the slowest arm in the set. Mechanism:
`n-match 24` makes ngram stricter, `n-max 7` makes dflash dearer per call;
together ngram fires **12** times and dflash burns **3,612 draft tokens to keep
1,262** (34.9 %).

### 3.4 `--spec-draft-n-max` and `--spec-draft-p-min`

`results/sweep-draft-n.jsonl` at ctx 16,384: `n=3` (llama.cpp default) 70.2/70.5/70.2
(−11.5 %, inside the floor); `n=4` shipped, baseline; **`n=7` (the clamp)
97.7/98.4/98.2, +23.4 % [+23.1, +23.5] RESOLVED**. Costs **149.625 MiB per unit**
of recurrent state; spills to `63+2` at ctx 65,536.

`results/sweep-p-min.jsonl`, 9 rows: **null.** 0.10 → +2.2 % [−0.3, +6.2];
0.25 → +1.5 % [−1.6, +7.1]. **At `0.10` every per-implementation counter is
byte-identical to the baseline — the early-stop never fired once.** At `0.25` it
fired on 2.2 % of calls. The algebraic bound (`1/sum ≥ 1/16`) was correct and
still too generous.

### 3.5 GPU trace at ctx 98,304

`results/gpu-trace-98304.jsonl`, 1,094 samples over 91.9 minutes, 5 s cadence:

| | |
|---|---|
| free VRAM | min **32 MiB**, median 258, max 10,921 |
| power draw | min 39.4 W, median **75.9 W**, max **107.1 W** (card is ~220 W) |
| utilisation ≥ 95 % | **97 % of samples** |
| of those, power < 120 W | **100 %** |
| samples with < 400 MiB free | **99 %** |

**Caveat that matters:** `utilization.gpu` means "fraction of time at least one
kernel was resident", not SM occupancy. High utilisation at low power is the
memory-bound signature, and **LLM decode is memory-bound by nature** — so this
profile alone is not proof of pathology. **There is no control trace at ctx
16,384.** What is anomalous without a control is §3.1: prefill is compute-bound
and 74 tok/s is 15× below the shallow figure.

---

## 4. Corpus and instrument limits discovered

> 🔴 **The headline of this section was wrong and is retracted** —
> [`CORRECTIONS.md` §25](CORRECTIONS.md), 2026-08-23. The table below is
> correct; the sentence that used to introduce it was not. chars/token is
> **~3.4**, measured against the server's own token counts, and the reason a
> run fed ~40 % of its label is `dflash2_arena.py:478` —
> `filler(int(ctx * 0.5), regime)`, which asks for half the window **by
> design**. The "7.0–7.4" was `ctx × 3 ÷ tokens` with the 0.5 dropped.
> `filler()`'s assumption of 3 is about 12 % low, not 2.3× wrong.

**Every run labelled "ctx N" fed a prompt of about 40 % of N:**

| labelled | actual prompt |
|---:|---:|
| ctx 16,384 | 6,621 tokens |
| ctx 65,536 | 28,122 tokens |
| ctx 98,304 | 43,162 tokens |

`--ctx` still sets the **allocation**, so every VRAM/residency finding is
unaffected. What is affected is the **label** on depth findings: "n-match at
65,536" is really "n-match at 28k of context in a 65,536 window". The direction
of §3.2 holds because context did grow 4.2×.

**Corpora.** `bench/corpora/real-code.txt` 91,868 chars, 10.3 % duplicate lines.
`bench/corpora/real-code-deep.txt` 406,146 chars, 45 files, **0.4 % window
repetition at n=24** against the incumbent's 0.6 %. Chosen on
`harness.window_repetition_pct`, which measures what `ngram-mod` actually keys
on; the line-repetition metric would have rejected it for boilerplate.

**The 13.6 % noise floor is a ctx 16,384 number.** Decode is deterministic at
temperature 0 — per-implementation counters are byte-identical across rounds —
so all of this spread is the clock:

| arm | within-arm spread @ 16,384 | @ 65,536 |
|---|---:|---:|
| `n-match 24` | 2.2 % | 23.5 % |
| `n-match 16` | 0.8 % | 10.3 % |
| `n-match 12` | 9.5 % | **39.5 %** |
| `n-match 8` | 10.6 % | **48.9 %** |

**The same arm, unchanged in every counter, spans up to 48.9 % between boots at
65,536.** `n-match 16`'s +67.5 % is directionally safe only because its **worst**
round beats every other arm's **best** by 32 %, which needs no floor at all. **Do
not quote the magnitude.**

---

## 5. llama.cpp source facts established by reading

All citations are `C:\AI\llama.cpp`, build 10499.

### 5.1 Speculation counters
`common_speculative_print_stats` (`common/speculative.cpp:2829`) is called from
`server_slot::print_timings()` (`tools/server/server-context.cpp:590`) at all
three request-completion paths — **not** at exit. Counters are zeroed once
(`speculative.cpp:144-153`) and never reset, so **adjacent blocks difference to
give one request, per implementation.** Gate is `LOG_LEVEL_TRACE = 4`.

**Limit:** this is per-**request**, not per-time-slice. A request spanning many
periods (a 9.7-minute prefill) cannot be attributed *within itself* — the
counters carry no timestamps.

**The object is one per server** (`server-context.cpp:1231`, aliased at `:1256`),
so with `n_parallel > 1` counters pool across slots.

`/metrics` exposes `llamacpp:spec_decode_num_draft_tokens_total`,
`..._num_accepted_tokens_total`, `..._num_drafts_total` — monotonic, but **pooled
across implementations**.

### 5.2 The timings object is not what its names suggest
- **`prompt_ms` is time-to-first-token from slot launch**, not prefill:
  `t_start` at `server-context.cpp:3054`, `t_prompt_last` stamped again after the
  first token is sampled at `:3780`.
- Therefore **`prompt_ms + predicted_ms` is the entire slot-busy window**, and the
  server's `total time` line is literally that sum (`server-context.cpp:607-610`).
  **Any residual against wall-clock lies outside the slot.**
- **`prompt_per_second` is a cache-hit figure.** Same log, same boot:
  `task 0` cold = 6,621 tokens at **1,137.31 tok/s**; `task 186` with 6,617
  cached = 4 tokens at **38.49 tok/s**. A 30× swing with no hardware change.
  `n_prompt_cached = n_past; n_prompt_processed = 0` at `:3320`, and
  `[TAG_PROMPT_LOGITS]` at `:3313-3318` forces at least one token to be evaluated.
- **`predicted_per_second` cannot be re-derived** — it divides by `n_gen - 1`
  (`server-common.h:403, 419`).
- **No reasoning-token count exists in `timings`** — `server_slot_stats::to_json()`
  (`server-common.cpp:67-87`) carries `predicted_n` as one total. The
  reasoning/content split exists only in the **content**: `reasoning_content`
  assembled at `server-chat.cpp:394-403`, controlled by `--reasoning-format`
  (`common/arg.cpp:3621-3628`).
- **Time-to-first-token, Total and Chunks are client-side only.**

### 5.3 Ten argv-vs-effective transformations, most unlogged

| flag | what actually happens |
|---|---|
| `-np` | server default `-1` → auto-mode forces **4 slots + `kv_unified = true`** (`server.cpp:151-156`), silently discarding `--no-kv-unified`. **Our profiles all pass `-np 1`, verified — this does not reach us.** |
| `--fit-target` | draft-model bytes added first (`server-context.cpp:1074`), so `768` arrives at `fit.cpp` as ~1,900–2,100 MiB |
| `--spec-type` | **accumulates, does not replace** (`arg.cpp:4159-4160`); order is discarded and every `ngram-*` ranks above every model-based type (`speculative.cpp:2540-2552`) |
| `-ctk`/`-ctv` | **never apply to recurrent state** — `recurrent_type_r/s` hardcoded F32 (`llama-model.cpp:2316-2317`) |
| `-ngl auto` vs `all` | resolve identically; only difference is `all` aborts `--fit` |
| `-ub` | silently sizes the SWA cache (`llama-kv-cache-iswa.cpp:73`) |
| `--spec-draft-backend-sampling` | **no-op on DFlash2** (`speculative.cpp:1015`); draft `top_k` hardcoded at `:1005` |
| MTP draft depth | clamped silently (`speculative.cpp:1445-1446`) |
| `LLAMA_ARG_NO_*` | **presence alone** sets `"0"` (`arg.cpp:128-134`) |
| `-t 0` vs omitted, `-c 0` vs omitted | different code paths, different values |

**Values that survive every rewrite:** `llama_n_ctx()`, `llama_n_batch()`,
`llama_n_ubatch()`, `llama_n_ctx_seq()` read back off the live context, plus
`build_info` free on every response as `system_fingerprint`. `/props` publishes
the real post-fit `n_ctx` (`server-context.cpp:1202, 4592`).

### 5.4 KV versus recurrent state

At ctx 65,536: **KV buffer 1,152.00 MiB (q4_0)** against **RS buffer 748.12 MiB
(f32)**. RS is **context-independent** — it scales with `--spec-draft-n-max`, not
`-c`. So RS dominates at shallow context (~288 MiB KV at 16,384) and KV dominates
at depth, crossing over near 32K. **`-ctk q4_0` is worth more as the window grows
and less than assumed at 16,384**, where most KV verdicts here were taken.

### 5.5 Clock alignment
llama.cpp's log prefix is `M.SS.mmm.uuu` where **minutes are total, not mod 60**
(`common/log.cpp:97-106`), relative to `common_init()`, and uses **`system_clock`,
not `steady_clock`** — a clock step shifts every subsequent offset undetectably.
No absolute-time flag exists. Server log aligns to wall clock at **±21 ms** via
`GetProcessTimes`; combined with the GPU sampler, **±60 ms with a systematic
−50 ms bias** from `nvidia-smi` shutter lag.

---

## 6. Instrument faults found — read before trusting anything above

| fault | consequence |
|---|---|
| ~~`filler()` assumed 3 chars/token; real is ~7~~ **RETRACTED, [CORRECTIONS §25](CORRECTIONS.md)** | every "ctx N" run fed ~40 % of N — **that part holds**, but because `dflash2_arena.py:478` asks for `int(ctx * 0.5)` by design. chars/token is **~3.4**, so `filler()` was never the fault and this was never an instrument fault. |
| `filler()` silently truncated when the corpus was short | a run at 65,536 would have reported a plausible rate for a window it never filled. **Now raises.** |
| `harness.parse_spec_impl_stats` overwrites on each match | kept only the last cumulative block; **every per-request breakdown was already in the logs and discarded** |
| `dflash2_arena.py:439` uses `log.open("w")` | NTFS keeps the **original** creation time; **33 of 112 log files** have ctime 1,300+ s before last write. Anchoring on ctime misplaces a run by up to 21 minutes, silently |
| `bench/_deepwork/` was committed | 157 files of model-generated code in git; removed, and `bench/_*/` now gitignored |
| Earlier: benchmark built its prompt from its own source | 78.9 vs 105.4 tok/s on byte-identical arguments — see `CORRECTIONS.md` §20 |

Full register: `docs/reports/CORRECTIONS.md`, **24 entries**.

---

## 7. Running right now

- **`n-match` sweep at ctx 98,304**, `results/sweep-ngram-nmatch-98304.jsonl`,
  **8 of 16 rows**, 4 arms × 4 rounds (4 rounds because the floor at depth is
  much larger than 13.6 %).
- **`gpu_trace.py`** sampling every 5 s alongside it.
- Branch `build/dflash2-runtime`, **21 commits ahead of `main`, unpushed**.
- **No worker profile has been changed.** Nothing measured here has shipped.

---

## 8. Open questions worth research

1. **Why does prefill collapse 15× at ctx 98,304 with every layer resident?**
   The single highest-value question. Needs a control GPU trace at 16,384 to
   establish whether the power profile is anomalous or just what this workload
   looks like.
2. ~~Why did five real tasks change nothing?~~ **Answered** — the worker
   edited the live tree, not the clone (`CORRECTIONS.md` §24). The real
   question underneath it is now (1): a task cannot finish at 4 tok/s
   regardless of where it writes.
3. **Does `--cache-reuse` restore DeltaNet recurrent state, or only KV?** If only
   KV, a hybrid model re-computes recurrent state every turn, and with prefill at
   74 tok/s that alone could account for the 24–40 minutes. **Untested; the
   largest untested idea available.**
4. **What is the right `n-match` at the served 98,304?** 24 wins at 16,384, 16 at
   65,536, and the shipped 12 loses at both.
5. **Grammar × drafter has never been run together**, and the served profile needs
   both.
6. **Re-derive the noise floor at depth.** 13.6 % is a 16,384 number and within-arm
   spread reaches 48.9 % at 65,536.

---

## 9. Where the data lives

```
qwen38-tuning/results/*.jsonl      raw rows, one per measurement
qwen38-tuning/logs/dflash2-*.log   server logs, -lv 5, 112 files
qwen38-tuning/bench/harness.py     every summarising primitive
qwen38-tuning/bench/corpora/       the two frozen corpora
docs/reports/CORRECTIONS.md        24 retracted claims
docs/results/                      the register: what was tried, what happened
docs/OPEN-WORK-LEDGER.md           what is still open
```

Tracker: `xenodeve/Qwen-3.8-27B-Tuning`. Issue **#18** carries the measurement
thread; issue **#19** is the map for the run-recorder being designed.
