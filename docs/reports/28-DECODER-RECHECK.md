# 28 — The decoder verdicts, re-measured with headroom and a long generation

**Measured 2026-08-21.** Raw: `qwen38-tuning/results/mtp-recheck.jsonl` and
`qwen38-tuning/results/step-w-long-generation.jsonl`. All arms on
`UD-IQ2_XXS`, which returns 1,056 MiB against `UD-IQ2_S` — more than any drafter
here costs.

## Why this was re-run

Two open doubts, from the project's own corrections file:

- **The VRAM cliff.** `draft-mtp` was retired on **−71 % at 131,072**, measured
  on the artifact that settles closest to the edge, before the cliff was known
  (`CORRECTIONS.md` §13, §14). A row that collapsed for want of headroom looks
  exactly like a decoder that lost.
- **The 160-token rule.** `CORRECTIONS.md` §8: every decoder was eliminated on a
  160-token generation, and an external report has speculation only reaching rate
  over a longer run — *"the MTP had gotten extremely fast (91 tk/s vs 62 tk/s
  starting rate)"*.

## The cliff doubt: refuted, the verdict stands

160-token generations, two rounds, order reversed, free VRAM on every row:

| depth | arm | prefill r1 / r2 | decode r1 / r2 | free MiB |
|---|---|---|---|---|
| 131,072 | `ngram-mod` | 761.9 / 838.2 | **45.87 / 48.11** | 773 / 584 |
| 131,072 | `draft-mtp` | 474.3 / 508.6 | **6.21 / 6.09** | 677 / 467 |
| 65,536 | `ngram-mod` | 1,016.0 / 1,008.4 | **65.06 / 60.33** | 2,040 / 2,084 |
| 65,536 | `draft-mtp` | 973.3 / 905.5 | 51.14 / 52.47 | 734 / 672 |

**Every MTP row sat between 467 and 734 MiB free — above the cliff — and MTP
still lost.** At 131,072 it decodes **7.7× slower**, reproducibly to 2 %. The
−71 % was, if anything, generous.

## The 160-token doubt: also refuted

`N_PREDICT = 1024` at 65,536, two rounds:

| arm | decode r1 / r2 | vs its own 160-token figure |
|---|---|---|
| `ngram-mod` | **64.83 / 64.91** | 65.06 / 60.33 — unchanged |
| `draft-mtp` | 54.18 / 54.08 | 51.14 / 52.47 — **+4 %**, not +47 % |

Both arms repeat to 0.2 %. **The long run gives MTP four percent, not the
transformation the external report described**, and it still finishes 17 % behind
a decoder that costs no VRAM at all. `CORRECTIONS.md` §8 is answered for
`draft-mtp`: the verdict was not an artefact of the short generation.

## DFlash 2 does not load on this build

```text
  E llama_model_load: error loading model: done_getting_tensors:
    wrong number of tensors; expected 81, got 58
```

Twice, from `z-lab/Qwen3.8-27B-DFlash2-GGUF`, 1.1 GB. The vendor's own
announcement explains it: llama.cpp support for DFlash 2 arrives with **PR
#27342**, which build 10472 does not carry. The `draft-dflash` flag this build
exposes implements the **first** DFlash, which expects a different tensor set.

**So the register entry — *"screened, not competitive on 12 GB"* — describes a
screen that could not have run.** The honest state is *cannot load; needs a newer
llama.cpp*.

It is worth revisiting when the build moves: the vendor claims **2.7–3.4× the
throughput of autoregressive decoding on Qwen3.8-27B**, and unlike MTP the
drafter is 1.1 GB against `UD-IQ2_XXS`'s 1,056 MiB of returned headroom. Those
are vendor numbers on unstated hardware and nothing here has tested them.

## What stands

`ngram-mod` remains the decoder for this machine, and now for a measured reason
rather than an unconfirmed one: **it costs 0 MiB** (report 27) and it beats the
only drafter that will load, at both depths and at both generation lengths.
