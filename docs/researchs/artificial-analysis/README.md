# Artificial Analysis — the three effort levels of Qwen3.8-27B, priced

**External material. Captured 2026-08-24 from charts dated 23 Aug '26. Not
evidence here until measured here.**

Source: Artificial Analysis, Intelligence Index **v4.1.1** (9 evaluations:
GDPval-AA v2, τ³-Banking, Terminal-Bench v2.1, SciCode, Humanity's Last Exam,
GPQA Diamond, CritPt, AA-Omniscience, AA-LCR) and the **Agentic Index**, which is
the weighted average of the agentic benchmarks inside it (GDPval-AA v2,
τ³-Banking).

### The exact comparison, reproducible

[**Open the live Agentic Index view with this model
set**](https://artificialanalysis.ai/?models=muse-spark-1-2%2Cgemini-3-5-flash-lite%2Cinkling%2Ck-exaone-2-0-0803%2Cminimax-m3%2Cnemotron-3-5-lightning%2Ccommand-a-plus%2Cgpt-5-6-luna%2Cmuse-glimmer%2Cnvidia-nemotron-3-ultra-550b-a55b%2Cnvidia-nemotron-3-super-120b-a12b%2Csolar-open2-250b%2Ca-x-k2%2Cdeepseek-v4-pro%2Cqwen3-8-2-4t-a95b%2Cclaude-4-5-haiku-reasoning%2Cqwen3-8-27b%2Cmotif-3%2Cgemini-3-7-flash%2Cclaude-opus-5%2Cgpt-5-6-terra%2Cgrok-4-6%2Cclaude-fable-5%2Cglm-5-3%2Cgpt-5-6-sol%2Cmistral-medium-3-5%2Cgpt-5-5-pro%2Cgpt-oss-120b%2Ckimi-k3%2Cqwen3-8-27b-low%2Cqwen3-8-27b-medium&intelligence=agentic-index)

The three effort levels are separate entries in that selection — `qwen3-8-27b`
(which is `xhigh`), **`qwen3-8-27b-medium`** and **`qwen3-8-27b-low`** — which is
what makes the comparison in this file possible at all; most published charts
carry one row per model.

> ⚠️ **The link is a live view and the images are a capture.** They will drift
> apart: Artificial Analysis re-runs its suites, revises index versions (these
> images are **v4.1.1**), and adds models. **Quote the images, not the link**, and
> if the two ever disagree the images are what this project's reasoning was built
> on. Nothing here was fetched — the numbers below were read off the captures.

**Why it is filed.** It is the only source this project has that prices
`reasoning_effort` for *this model* on an *agentic* axis — the axis the worker
actually runs on — and it arrived the same night
[`results/05`](../../results/05-runtime-flags.md) established that **every server
this project has ever launched runs at `xhigh` with an unlimited thinking
budget**, because nothing overrides the template default.

---

## The three levels, both indices

![Intelligence Index](01-intelligence-index-2026-08-23.png)

![Agentic Index](02-agentic-index-2026-08-23.png)

| Qwen3.8-27B | Intelligence Index | **Agentic Index** |
|---|---:|---:|
| `xhigh` | **52** | **51** |
| `medium` | 44 | **50** |
| `low` | 43 | 44 |

### The two indices disagree about where the cost is, and that is the finding

```
Intelligence   xhigh -> medium   -8      medium -> low   -1
Agentic        xhigh -> medium   -1      medium -> low   -6
```

**On the agentic axis, dropping `xhigh` to `medium` costs one point. Dropping
`medium` to `low` costs six.** The reverse is true on the general axis.

**This project's metric is verified accepted coding tasks per hour**, which sits
on the agentic axis. So if the external review
[`results/05`](../../results/05-runtime-flags.md) cites is right that xhigh takes
**15 minutes where medium takes 3**, `medium` buys back most of the wall clock
for one point of agentic capability — and `low` is where the capability actually
goes.

**A question this project asked and answered wrongly for a moment:** *"medium or
low?"* was posed as if the two were the same direction. They are not.

### Where the model sits, for scale

On the Agentic Index `Qwen3.8-27B (xhigh)` at **51** places above
`GPT-5.6 Luna (max)` 47, `Gemini 3.7 Flash (high)` 45 and `Motif 3` 38, and below
`Kimi K3 (max)` 54 and `GPT-5.6 Sol (max)` 58. `Claude Opus 5 (max)` and
`GLM-5.3 (max)` lead at 59.

---

## ⚠️ What this cannot be used for here

- **These are the full-precision model through an API.** Our worker is
  `UD-IQ2_XXS` at **2.16 bpw** or `UD-Q2_K_XL` at ~2.9 — nothing on these charts
  measures a quantised local build, and
  [`results/01`](../../results/01-artifacts.md) shows task success falling
  monotonically with bits per weight across five artifacts. **The effort ranking
  may hold and the absolute numbers certainly do not.**
- **It is not a wall-clock measurement.** The charts price capability, not time.
  The "15 minutes against 3" figure comes from a different external review and is
  itself unverified here.
- **A one-point gap is inside most benchmark noise.** Artificial Analysis
  publishes no error bars on these bars, so `xhigh` 51 against `medium` 50 should
  be read as "no measurable difference", not as "xhigh is slightly better".
- **The agentic index is two benchmarks** (GDPval-AA v2, τ³-Banking), neither of
  which is a coding-agent loop against a real repository.

## What it changes about what to run next

Nothing until measured. But it makes **`--reasoning-effort medium`** the level to
try first on this machine, rather than `low` — and it makes a `low` run worth
doing second, as the arm most likely to show where capability breaks.

*Charts dated 23 Aug '26. Earlier captures of the same two indices (18 Aug,
22 Aug) exist in the operator's downloads and were not filed; nothing here
depends on the series.*
