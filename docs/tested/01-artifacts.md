# 01 — Artifacts

Every model file this project has loaded. `bpw` is the real bits per weight from
the loader's tensor histogram, **not** the filename.

## Qwen3.8-27B

| file | GiB | bpw | deepest `65+0` | decode | corpus | verdict |
|---|---:|---:|---|---|---|---|
| V3 `UD-IQ1_S` | 5.77 | **1.84** | **196,608** | 22.3–23.2 @192K | **0 of 12** — no fenced block, 12/12 | fastest, unusable |
| V3 `UD-IQ1_M` | — | ~2.0 | **163,840** | 24.4 base / 47.3 n-gram @160K | 10/21 · 41.5 % contract | depth yes, quality no |
| V3 `UD-IQ2_XXS` | 6.77 | **2.16** | **147,456** | 25.6 base / 109.2 n-gram @147K | 19/27 · 58.3 % contract | the deep candidate |
| V3 `UD-IQ2_S` | 7.80 | not read | **never loaded** | — | — | **the untested rung** |
| `AD-IQ1_M` (AtomicChat) | 7.91 | **2.49** | ≤131,072 (`65+1` there) | 6.08 @128K · 18.75 @16K | **27/30** | 16K only |
| pre-V3 `UD-IQ2_XXS` | 8.39 | **2.64** | `58+7` @131,072 | — | **27/30 (90 %)** · 48.5 verified/hr | the quality default, 16K |
| V3 `UD-Q2_K_XL` | 9.83 | — | `54+12` @131,072 | — | not measured | too big for depth |
| pre-V3 `UD-Q2_K_XL` | 10.68 | — | `50+16` @131,072 | — | 26/30 | superseded |
| `UD-Q3_K_XL` | 13.44 | — | — | — | — | control only |
| `UD-Q4_K_XL` | 17.92 | — | — | — | the deep-retrieval control | too big |

**The names do not describe the files.** `AD-IQ1_M` at 2.49 bpw is *heavier* than
`UD-IQ2_XXS` at 2.16 — only 80 of its tensors are 1-bit and 128 are full `q8_0`.
V3 `UD-IQ2_XXS` contains **zero** `q8_0` tensors, which is why it is a gigabyte
smaller than the pre-V3 file of the same name.

### The relationship that has not been contradicted

Sorted by bpw, both quality columns improve monotonically across five artifacts
and two vendors:

```text
  1.84   0 of 12          no fenced block at all
 ~2.0    10/21   41.5 %
  2.16   19/27   58.3 %
  2.49   27/30
  2.64   27/30   90 % accept
```

**Still a hypothesis.** Confounded by vendor, tensor mix and quantizer. But it is
the cleanest correlation in the project — cleaner than any flag — and `UD-IQ2_S`
at 7.80 GiB sits in the gap, already on disk, never opened. Registered in
`depth_sweep.QUANTS` as `v3-iq2s` on 2026-08-21.

*Raw: `results/ctx-ceiling-q38.jsonl`, `results/retry-bench.jsonl`,
`results/kv-deep-*.jsonl`. Explained in reports 12, 13, 21, 23, 24.*

## Other models

| model | why it was tried | outcome |
|---|---|---|
| `Ornith-1.0-9B` Q6_K / Q8_0 | a smaller dense model at full precision | 29.2 merged/hr, 72.0 verified/hr at 3,072 budget · 66.7 % accept |
| `Ternary-Bonsai-27B` Q2_g64 | ternary quantization, a different compression family | 17.9 merged/hr · below the Q2 arms |
| `Bonsai-27B` Q1_0 | the 1-bit end of the same family | screened only |

*Raw: `results/arena-ornith.jsonl`, `results/arena-bonsai.jsonl`,
`results/kv-sweep-bonsai.jsonl`.*

## Not tested

- **`UD-IQ2_S`** — on disk, never loaded. Highest-value gap.
- **Any Qwen3.8-27B above `Q4_K_XL`** — does not fit 12 GB with useful context.
- **`fp8` served locally** — only reachable through the 9arm gateway, where it
  answered two hard corpus tasks correctly first try, both with and without
  injected skills. Not a local option; useful as a reference point for what the
  weights can do when quantization is not the constraint.
  *Raw: none — run through `mcp__pal__clink`, graded by hand with
  `run_bench.verify`. Four calls, 2026-08-21.*
