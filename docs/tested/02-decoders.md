# 02 — Decoders (`--spec-type`)

Eleven values exist in build 10472. All eleven have been tried.

> **Every figure on this page carries the same caveat.** The timed prompt is
> **84.5 % duplicate lines** — one class repeated with a changing index, 962
> blocks at 147,456, adjacent blocks 99.5 % identical. An n-gram decoder drafts
> from context, so this is close to the best case that can be constructed for it.
> **The mechanism is real; the magnitudes are upper bounds.** Steps F1/F2
> re-measure at 73.17 % repetition. See `CORRECTIONS.md` §2.

> **And every elimination is provisional.** All were decided on **160-token**
> generations. An external review of this model reports speculation reaching rate
> only over a longer run. Step W tests 160 vs 512 vs 1024. See `CORRECTIONS.md`
> §8.

## The n-gram family — no drafter file, no VRAM, output identical

| arm | 16,384 | 131,072 | 147,456 | 163,840 | acceptance |
|---|---|---|---|---|---|
| `ngram-map-k` | **+135.89 %** | +120.54 % | — | — | 96.9–100 % |
| `ngram-mod` (short window) | +112.55 % | **+200.22 %** | **+330.40 %** | +100.48 % | 99–100 % |
| `ngram-map-k4v` (wide) | +114.64 % | +108.15 % | — | — | 83.2 % |
| `ngram-simple` | +51.31 % | — | — | — | — |
| ~~`ngram-cache`~~ | +108.49 % | — | — | — | **0 %** |

**The winner changes with depth.** `ngram-map-k` leads by 10 points at 16 K and
loses by 80 at 131,072. Use `ngram-map-k` at 16 K, `ngram-mod` at depth.

**`ngram-cache` is disqualified.** Greedy hash `3EFE93950A8A980E` against a
same-depth baseline of `04E5CAB1D14525C0` — it changes the answer, so it is not
draft-and-verify. Reported as safe in report 20 §1.1 for a day; that hash block
was typed by hand rather than read from the JSONL.

*Raw: `results/kv-decoders.jsonl`, `results/kv-ngram-fixed.jsonl`,
`results/kv-deep-*.jsonl`. Reports 20 §1, 23 §1, 24.*

## The drafter-model family — all need a file, all cost VRAM

| arm | result | why |
|---|---|---|
| `draft-mtp` | **+81 % @16K, −71 % @131,072** | the head is `blk.64`, 1.28 GiB. At `61+4` it displaces layers; at `65+0` the cost is prefill |
| `draft-mtp` on CPU (`--spec-draft-device none`) | **−59 %** | external research predicted +70–85 % |
| `draft-mtp` with `-otd .*=CPU` | worse than GPU | |
| `draft-eagle3` | no usable head for this model | never produced a run |
| `draft-dflash` / DFlash 2 | drafter 1.06 GiB | screened, not competitive on 12 GB |
| `draft-dspark` | tried with Ternary Bonsai | not competitive |
| `draft-simple` | needs a second full model | no room |

**The pattern:** on a 12 GB card any drafter that holds weights competes with the
layers, and the residency cliff is steeper than the speculation gain. The n-gram
family wins here because it holds nothing.

*Raw: `results/mtp-sweep.jsonl`, `results/kv-decoders.jsonl`,
`results/spec-matrix-q*.jsonl`. Reports 20 §2, 22.*

## Interaction found 2026-08-21 — `-ot ssm` and speculation

One flag, three outcomes:

| where | acceptance | result |
|---|---|---|
| `v3-iq2xxs` @163,840, 10 blocks | **4 %** | slower than not offloading (32.4 vs 38.7) |
| `v3-iq2xxs` @163,840, 4 blocks | **no drafts at all** | level with baseline |
| `v3-iq1m` @196,608, 10 blocks | **100 %** | **+181.57 %** |

Reproduced in four boots at 4 %. Whether artifact or depth is responsible is
**unknown**, and nothing queued separates them.

*Raw: `results/kv-deep-160k.jsonl`, `results/kv-deep-192k.jsonl`. Report 24
§1, §1b.*

## The acceptance column may be worth more than the speed column

100 % on `v3-iq2xxs` and `v3-iq1m`; **37.5 %** on `v3-iq1s` — the artifact that
scores 0 of 12 on the corpus. A 30-second reading that may stand in for a
40-minute gate.

**Confounded with depth** — `v3-iq1s` was measured at 196,608 and the others
lower. Step V2 separates them. Not yet a usable signal.
