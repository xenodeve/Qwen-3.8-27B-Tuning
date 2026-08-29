# researchs — external material. NOT our measurements

Everything here came from outside this machine: deep-research replies, vendor
documentation, model cards, screenshots.

> **Nothing in this folder is evidence until it has been measured here.**
> See [`../reports/17`](../reports/17-EXTERNAL-RESEARCH-REVIEW.md) and
> [`../reports/18`](../reports/18-RESEARCH-ROUND2-REVIEW.md) for the record of
> what survived contact and what did not.

## The one scan that replaced a repository

[`syv-rtx3090/`](syv-rtx3090/README.md) — `syv-ai/qwen38-27b-rtx3090`, a patched
vLLM 0.27.1 stack serving Qwen3.8-27B on one RTX 3090, read line by line so that
nobody has to open it again. **434 techniques**, each matched against a
**175-capability map of our own llama.cpp** at build 10499.

**Start at [`../results/08-rtx3090-transfer.md`](../results/08-rtx3090-transfer.md),
not at the scan.** Twenty-one of those techniques now carry a verdict here, and
the scan is a frozen capture that is deliberately never edited when a
measurement lands — so its **48 flags we already have and have never set** still
reads as if none of them had been tried. Three of its entries reverse on contact
with the register: one would corrupt output silently, one costs 14 % of decode
for 66 MiB, and one it rates highest-value changes nothing at all.

The scan's first concrete result was not a technique of theirs at all — it was a
false claim in one of our own worker profiles, caught by mapping our tool
exhaustively.

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
| [`hf-discussion-5060ti-mtp/`](hf-discussion-5060ti-mtp/README.md) | **The closest external match this project has** — same card, same model repo, same runtime, same decoder, and the participants name artifact, depth, KV type and effort. Carries the **only outside paired MTP-vs-no-MTP curve** on an RTX 5060 Ti: **2.08× at 2.5K falling to 1.72× at 25.4K**, decaying with depth while we serve at 147,456 (issue #44). Also the **third and fourth independent confirmations** that the template default is `xhigh`, and a `Vulkan-instead-of-CUDA` incident that is our own `sm_89`-on-`sm_120` fault one layer up. **Checked against our profile and nothing needed changing** |
| [`reddit-5060ti-quant-thread/`](reddit-5060ti-quant-thread/README.md) | **The first outside numbers on our exact card** — four commenters on an RTX 5060 Ti 16 GB. Its real value is not a number: one of them opens with *"IMPORTANT: compile with `GGML_CUDA_FA_ALL_QUANTS=ON`"*, and checking our own `CMakeCache` found it **OFF in both builds**, closed years ago on a Q8 result that [cannot test it](../reports/CORRECTIONS.md) (§29). Also: an independent operator reaching **`medium`** for the same reason we did, and a same-card NVFP4 decode figure that **strengthens** the decision not to pursue it |
| [`artificial-analysis/`](artificial-analysis/README.md) | **`reasoning_effort` priced on the agentic axis** — the axis the worker runs on. `xhigh` 51, `medium` **50**, `low` 44 on the Agentic Index, against 52/44/43 on the general one. **The two indices disagree about where the cost is**, and on ours `xhigh → medium` is one point while `medium → low` is six. Full-precision through an API, so the ranking may transfer and the numbers do not |
| [`superalesha-quant-ladder/`](superalesha-quant-ladder/README.md) | **12 formats × 720 tasks on 4× RTX 3090** — the only public ladder that tests the exact two GGUF files at the centre of our next decision. Its author then **audited his own methodology in public**: 90 of 150 tasks pass on every quant, so the ranking rests on 54 and gaps under 3 points are noise. **Only the cliff survives that** — `IQ2_XXS` 0.76 against `Q2_K_XL` 0.86, and the failure is *non-termination*, which this folder's own README had already described before his thread |
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

- [`unsloth-studio-config-2026-08-29.md`](unsloth-studio-config-2026-08-29.md) — **the `llama-server` command line Unsloth Studio builds for OUR artifact on THIS machine**, read out of its own logs and settings database rather than off a web page. It agrees with us on `q4_0` KV, MTP beside an n-gram, the tensor split, and — independently — on **`n-match 24`**. It differs on eleven other flags, including turning the prompt cache and context checkpoints **off**, which is where our 34 GB of host RAM goes. Nothing applied; seven items ranked by cost at the end.
