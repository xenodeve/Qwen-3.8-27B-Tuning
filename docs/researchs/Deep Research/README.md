# Deep Research — replies from external research agents

**Unverified.** These are inputs to be measured, not findings. Reviewed in
[`../../reports/17`](../../reports/17-EXTERNAL-RESEARCH-REVIEW.md) and
[`../../reports/18`](../../reports/18-RESEARCH-ROUND2-REVIEW.md).

| file | what it covers | what survived measurement |
|---|---|---|
| `Local Worker Model, Quantization, and Runtime Selection for RTX 4070 SUPER 12GB.md` | the first survey: which models and quants fit 12 GB | the candidate list was useful; the MoE size figure was wrong by 2× and its config lost 46–48 % |
| `Candidate Inference Configurations…md` | decoder ecosystem, drafter checkpoints, exact artifact identities | the `--spec-type` inventory is correct and led to the n-gram result. Its "no exact Qwen3.8 DFlash drafter" line was true for DFlash v1 and false for **DFlash 2** |
| `deep-research-report (2).md` | round-1 reply to our brief | generic; assumed a 27B fits at 8-bit on 12 GB, which is off by more than 2× |
| `Unsloth Dynamic V3.0 (Quantization Strategy).md` | vendor method notes | states Unsloth does not train on the imatrix calibration set and uses no QAT/QAD |
| `Ornith 1.5 9B & 35B-A3B vs Qwen 3.8 27B Q1 Dynamic V3.md` | cross-family comparison | Ornith-9B does hold 262,144 resident here — confirmed |
| `deep-research-optimization1.md`, `2.md` | earlier optimization surveys | superseded by our own measurements in reports 15, 16, 20 |

---

## Read these with the acceptance criteria in hand

[`../../plans/02-RESEARCH-BRIEF-OPTIMIZATION-SURFACE.md`](../../plans/02-RESEARCH-BRIEF-OPTIMIZATION-SURFACE.md) §10
lists the ten rules every claim was asked to meet — resolvable URL, mechanism
rather than an unsourced multiplier, exact artifact identity, build requirement,
VRAM arithmetic, a cheapest falsification test.

**Both replies ignored the citation rule and the no-unsourced-numbers rule.** The
mechanisms they describe are frequently sound and several led directly to real
results. The percentages attached to those mechanisms have been wrong every time
they were checked.
