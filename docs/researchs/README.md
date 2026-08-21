# researchs — external material. NOT our measurements

Everything here came from outside this machine: deep-research replies, vendor
documentation, model cards, screenshots.

> **Nothing in this folder is evidence until it has been measured here.**
> See [`../reports/17`](../reports/17-EXTERNAL-RESEARCH-REVIEW.md) and
> [`../reports/18`](../reports/18-RESEARCH-ROUND2-REVIEW.md) for the record of
> what survived contact and what did not.

## The chart this project keeps coming back to

![Unsloth Dynamic v3.0 for Qwen3.8-27B — top-1 % accuracy against BF16 by quant size](unsloth%20v3.jpg)

**Read the ranking, ignore the absolute values.** It is measured against BF16 on
the vendor's hardware, and it says nothing about whether a 2-bit file finishes an
agent round trip on a 12 GB card at 128K — which is the only question this repo
exists to answer.

What it does say, and what our own measurements agree with: **the steepest part
of the whole curve is `UD-IQ2_XXS` to `UD-IQ2_S`** — about five points of top-1
for 1.1 GB. Our own bits-per-weight ladder is steepest in the same place.

The x-axis label — *"Quant size (GB) with removal of MTP"* — independently
confirms something we found by grepping loader output: V3 `IQ2_XXS` has no
`blk.64` MTP head, where the preview build did.

Every figure on all three charts is transcribed in
[`vendor-quantization-tables.md`](vendor-quantization-tables.md), so nobody has
to reopen an image to look a number up.

---

| folder / file | what it is |
|---|---|
| [`vendor-quantization-tables.md`](vendor-quantization-tables.md) | Unsloth V3-preview, V3-final and AtomicChat charts transcribed from their images — sampling presets, hardware guidance, accuracy curves, and the naming correction (there is no "pre-V3") |
| [`Deep Research/`](Deep%20Research/) | seven replies from external research agents — model candidates, quantization strategy, decoder ecosystem, runtime selection |
| [`Qwen3.8-27B_Optimization_Research_Docs/`](Qwen3.8-27B_Optimization_Research_Docs/) | the original ten-document research pack that started the project, with its own README |
| `unsloth.jpg`, `unsloth v3.jpg` | Unsloth's Dynamic v3.0 announcement — top-1 % accuracy against quant size |
| `atomic chat.jpg` | AtomicChat's AD-layout chart — mean KL divergence against file size |

---

## The two vendor charts disagree, and both can be right

Unsloth measures **top-1 % token agreement over 32 tokens**; AtomicChat measures
**mean KL divergence at 4,096 context**. Both are proxies for "how close to the
unquantized model" over *short* spans.

Our corpus measures whether code runs and passes tests over **8,192 tokens**. The
V3 failure we actually see — looping inside the reasoning block for 19,000–34,000
characters and never emitting a fenced code block — **cannot be seen at 32
tokens**.

Also note the axes are not comparable: Unsloth plots "quant size **with removal
of MTP**", AtomicChat plots raw file size. The same artifact sits at different
x-positions on the two charts.
