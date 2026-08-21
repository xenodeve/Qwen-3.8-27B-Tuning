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
