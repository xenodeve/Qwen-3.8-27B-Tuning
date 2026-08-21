# 05 — Runtime flags: threads, placement, sampling

Two full sweeps of this surface found **nothing above the 13.6 % drift floor**.
That is the headline of the page: the flags are not where the wins are.

## Placement and scheduling — all inert

Two rounds each at 16,384 on `v3-iq2xxs`, `--fixed-text`, order reversed:

| flag | effect | verdict |
|---|---|---|
| `pcore-mask` (thread affinity to P-cores) | +0.46 % | inert |
| `prio-high` (process priority) | −2.02 % | inert, sign flips |
| `poll-0` (polling strategy) | +0.69 % | inert |
| `backend-samp` (GPU-side sampling) | +2.27 % | inert |

Baseline decode was 38.6–38.65 and every arm landed between 36.4 and 39.5.

**A methodological note worth following up, not acting on.** Those pairs repeat
to within **0.05 percentage points across separate boots** — two orders of
magnitude tighter than the 13.6 % floor, which was derived from unpinned text.
`backend-samp` at +2.27 % with a range of 0.05 does not look like noise. But free
VRAM spanned only 2,872–3,016 MiB across these boots, a fifth of the 9,326–10,732
spread the floor came from. **A quiet night is not a smaller floor.**

*Raw: `results/kv-layers-16k.jsonl`. Reports 20 §4, 23 §4.*

## Threads and batch

| flag | tried | result |
|---|---|---|
| `-t` 8 / 12 / 18 / 24 | yes | 18 chosen; differences under the floor |
| `-b` / `-ub` | yes | see [`03-memory-and-kv.md`](03-memory-and-kv.md) — a VRAM lever, not a speed lever |
| `-fa on` / `off` | yes | `on` required; `off` loses residency |
| `-sm tensor` | yes | single GPU, no effect |
| `--no-repack`, `--no-op-offload`, `--load-mode`, `--no-host`, `--swa-full`, `--no-kv-unified` | yes | all confirmed inert, as report 16 predicted |

*Raw: `results/sweep-threads*.jsonl`, `results/sweep-batch*.jsonl`,
`results/kv-layers-16k.jsonl`, `results/kv-depth-levers.jsonl`.*

## Speculation flags — swept 2026-08-22, on the frozen corpus

Arena: `bench/dflash2_arena.py --regime real-code`, ctx 16,384,
`draft-dflash,ngram-mod`, three rounds, arms rotated, paired by round.
Raw: `results/sweep-*.jsonl`.

### `--spec-ngram-mod-n-min` — **no effect. Do not re-run it.**

| `n-min` | rounds (tok/s) | vs base |
|---:|---|---|
| 16 (ours) | 79.7, 79.6, 79.8 | baseline |
| 8 | 79.7, 79.5, 79.8 | −0.1 % |
| 4 | 79.7, 79.6, 79.8 | −0.0 % |
| 2 | 79.8, 79.8, 79.7 | +0.1 % |

Spread across all twelve runs: **0.3 %**. At that repeatability a 1 % effect
would be visible; there is none.

**The hypothesis and why it was wrong.** `ngram-mod` declines **93.7 %** of the
calls it receives on real code, and when it does fire it is worth **16.7
tokens** against `draft-dflash`'s 2.9 — so letting shorter drafts through looked
like a large free win.

It was a misreading of `common/speculative.cpp:1993`. In `draft_one`, `i` counts
**draft tokens already produced**, not matched context. `n_min` is therefore a
minimum draft *length*, and the declines happen at `i = 0` — the n-gram table
misses on the very first successor — where no value of `n_min` can help.

**What that leaves open.** The decline rate is real and large. The knob that
governs it is `--spec-ngram-mod-n-match` (default 24, ours 12): the width of the
context window the table is keyed on. That is a different flag and it has not
been swept.

## Grammar (GBNF) — what it costs, and the one thing nobody has measured

| question | answer | evidence |
|---|---|---|
| Does `--grammar-file` allocate VRAM? | **No — read from source, NOT measured.** `src/llama-grammar.cpp` contains no reference to `ggml_backend`, `cuda`, `ggml_new_tensor` or `device`. Its whole state is `std::vector` on the host: a pushdown stack and a rule table. It runs in the sampler chain after logits are copied back | source read on build 10499, `src/llama-grammar.cpp`, `src/llama-grammar.h` |
| Then what does it cost? | **CPU time per token**, which surfaces as tok/s and not as MiB. Unquantified here | — |
| Does it change anything else? | **Yes: it disables backend sampling.** `common/sampling.cpp:421` — `"backend sampling is not compatible with grammar, disabling"`. `--reasoning-budget` does the same at line 427, and `grammars/README.md` tells you to use both | `common/sampling.cpp:421,427` |
| Does that cost us anything? | **No.** `common.h:295` defaults `backend_sampling` to `false`, no worker profile enables it, and the flag was measured at **+2.27 % — inert** (table above) | `common.h:295`, `results/sweep-runtime*.jsonl` |
| 🔴 **Does a grammar work alongside a drafter?** | **Unmeasured, and the config we intend to serve needs both.** `common.h:331` is a *different* field — `backend_sampling = true` for the **draft** sampler, on by default — and the disable at `sampling.cpp:421` touches only the main one. So grammar + drafter runs in a state nothing has exercised | source read; **no run** |

**Why the last row matters.** The production profile has to carry a grammar —
without one, 41.5 % (`UD-IQ1_M`) to 58.3 % (`UD-IQ2_XXS`) of corpus attempts emit
no fenced code block at all — and it has to carry a drafter, because that is
where the speed is. Every measurement so far has one or the other.

**The check is cheap:** two boots, grammar on and off, read `nvidia-smi`. Then
the same pair with `--spec-type draft-dflash,ngram-mod` added. It has not been
run because the GPU was busy; that is a schedule, not a result.

## Sampling — two passes, nothing resolved

Arms tried across `answer-screen-sampling.jsonl` and `-sampling2.jsonl`:
`samp-base`, `samp-dry`, `samp-mirostat`, `samp-nsigma`, `samp-rep-default`,
`samp-rep4096`, `samp-grammar`, `samp-prefill`, `samp-backend`,
`samp-rbudget0`, `samp-rbudget2k`, `samp-rea-off`, `samp-dry-rb2k`, and the
second-pass repeats.

**Nothing moved a decision.** Two findings did come out of it, both negative:

- **`--reasoning-budget 0` does not end the reasoning block.** Screened alone it
  ran to **24,709 characters**. Paired with `--grammar-file` it returned
  `content_chars = 0` on 3 of 3 trials — the model reasons freely, then emits
  end-of-turn where the grammar starts to bind. **`-rea off` is the flag that
  ends the block.**
- **The screen itself is capped at 3 trials.** `answer_screen --trials 10`
  silently gives 3, because `for i in range(min(args.trials, len(PROBES)))` and
  `PROBES` has three entries. Every "n=10" screen in this project was n=3.

*Raw: `results/answer-screen-sampling*.jsonl`. Reports 20 §6, 22 §5.*

## `reasoning_effort` — swept, but not where it matters

**Corrected 2026-08-21.** This was written up as never tested; it was tested on
2026-08-18. `results/reasoning-effort-sweep.jsonl`, `low`/`medium`/`xhigh`, two
runs each, on a tool-calling probe:

| effort | wall | completion tokens | reasoning chars | reached the patch |
|---|---|---|---|---|
| `low` | 67.8 / 85.3 s | 453 / 608 | 384 / 570 | **both** |
| `medium` | 85.6 / 50.1 s | 610 / 352 | 621 / 212 | **both** |
| `xhigh` | 84.2 / 106.9 s | 588 / 741 | 632 / 1,008 | **both** |

Reasoning length rises with effort, as expected. **All six runs succeeded**, so
the probe cannot separate the levels on quality.

**What is genuinely untested:** this ran on **Q4** with n=2 and a tool probe. It
has never run on the 2-bit V3 artifacts, where the failure being chased is the
model looping inside the reasoning block until the budget runs out, and never
through the 30-task corpus. An external review of this model reports xHigh taking
**15 minutes** where medium takes 3 for *"90 % of the result"* — a difference our
probe was far too short to see.

*Raw: `results/reasoning-effort-sweep.jsonl` (note: UTF-8 BOM, read with
`utf-8-sig`).*

## Never tried

- **A system prompt that instructs how to think** rather than how to format —
  e.g. *"don't hedge, make conclusions, work forward, don't reconsider"*. Costs
  nothing, aims straight at the blocker.
- **`reasoning_effort: low` on a 2-bit artifact, through the corpus.**

## KV cache type against prefill — tested 2026-08-21

| question | answer | evidence |
|---|---|---|
| Is `q8_0` KV faster than `q4_0` for prefill? | **No reliable difference.** 714/882.5 against 984/871.1 over two reversed rounds; the within-arm spread is wider than the gap | report 27 |
| Is `f16` KV worth trying? | **Not measurable here.** 2,048 MiB of KV left 427 and 242 MiB free, both rows in the collapse regime | report 27 |
| Does `iq4_nl` KV work? | **No.** Prefill abandoned at the 737 s timeout, twice | report 27 |
| Can prefill be tuned at all? | **No.** Every setting-level lever is measured and none move it | report 27 |

Raw: `qwen38-tuning/results/prefill-kv-type.jsonl`.
