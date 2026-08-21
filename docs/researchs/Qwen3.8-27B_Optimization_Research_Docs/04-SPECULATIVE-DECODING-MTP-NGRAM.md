# Speculative Decoding for Qwen3.8-27B — MTP vs ngram

## 1. llama.cpp Supported Methods

**PRIMARY**

https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md

Current llama-server supports methods including:

```text
draft-simple
draft-mtp
ngram-cache
ngram-simple
ngram-map-k
ngram-map-k4v
ngram-mod
```

## 2. Built-In MTP

`draft-mtp` uses **MTP heads from the main model**.

```text
--spec-type draft-mtp
--spec-draft-n-max N
```

Current master documentation reports default `N=3`.

The Qwen35 llama.cpp implementation includes a dedicated MTP graph:

https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp

It loads NextN/MTP data and supports a single MTP block.

This matches the local Qwen3.8 GGUF evidence of an appended NextN block.

## 3. Why MTP Can Help

Normal autoregressive decode:

```text
target pass
→ 1 token
→ target pass
→ 1 token
```

Speculative decode:

```text
draft predicts several tokens
→ target verifies a batch
→ accepted prefix emitted
```

It is most useful when the output is predictable enough that draft acceptance is high.

Coding/refactor/tool JSON often fits that profile better than high-entropy creative reasoning.

## 4. Why MTP Can Hurt

MTP adds:

```text
draft computation
draft context/KV/state
verification overhead
additional runtime memory
```

On a 12 GB card where Q4 already saturates VRAM, added MTP memory can push target weights to RAM.

Therefore:

```text
Q4 + MTP may be slower than expected
Q3 + MTP may be much more attractive
```

Measure actual layer split and VRAM delta.

## 5. Video Claims

VIDEO source:

https://youtu.be/NjfHqiNHTxk

Gemini extraction reported claims such as:

```text
built-in MTP
3× headline speedup
4–5 draft-step sweet spot
lower temperature → higher acceptance
~2.5 GB VRAM overhead in discussed setup
```

Treat all numeric values as hypotheses until measured on the target machine.

## 6. MTP Is Not Assumed Perfectly Lossless

Algorithmically, speculative decoding is intended to preserve target-model sampling.

However, a recent llama.cpp issue reports greedy divergence with some quantized targets under draft-model/MTP speculative paths:

https://github.com/ggml-org/llama.cpp/issues/25618

The same report found ngram speculation matched vanilla in its test matrix.

Therefore production test must compare:

```text
same quant
same prompt
same deterministic sampling
baseline
vs MTP
vs ngram
```

Check:

```text
token IDs if available
text
tool name
tool arguments
tool order
```

## 7. `ngram-simple`

Official llama.cpp docs describe `ngram-simple` as a low-overhead self-speculative method that searches token history for a matching n-gram and drafts following tokens.

The docs explicitly give **source-code rewriting** as an example use case.

Example:

```text
--spec-type ngram-simple
```

Why it matters on this machine:

```text
no draft neural model
minimal memory overhead
coding outputs often repeat code/schema/text already in context
```

This makes it a serious candidate, not just a fallback.

## 8. Other ngram Methods

### `ngram-map-k`

Uses keyed n-gram history and hit thresholds.

### `ngram-map-k4v`

Tracks multiple candidate continuations; experimental.

### `ngram-mod`

Lightweight hashed shared pool; docs cite iterative code/text and reasoning repetition as applications.

Do not overcomplicate the first benchmark. Start with:

```text
none
ngram-simple
draft-mtp
```

## 9. Do Not Combine MTP + ngram Initially

First establish independent performance.

Combined speculative methods can add verification/memory overhead and complicate attribution.

Only combine later if primary/current llama.cpp behavior and local evidence show a real benefit.

## 10. MTP Draft Depth

Video-derived hypothesis suggests 4–5 can be a throughput peak.

Benchmark:

```text
2
3
4
5
6
```

only after a simple `n=2` smoke test succeeds.

Metrics:

```text
accepted/drafted
acceptance rate
tok/s
wall time
VRAM
RAM
layer split
tool correctness
output equivalence
```

## 11. Sampling Interaction

Lower temperature can increase speculative acceptance, but changes output distribution.

Therefore:

```text
Experiment A:
same sampling, MTP off/on

Experiment B:
sampling sweep at fixed spec mode
```

Do not mix these effects.

## 12. Draft KV

llama.cpp exposes separate draft-cache types:

```text
--spec-draft-type-k
--spec-draft-type-v
```

Start conservative/default. Do not automatically quantize draft KV merely because main KV uses Q8.

## 13. First Matrix

| Quant | none | ngram-simple | MTP n=2 | MTP n=3 |
|---|---:|---:|---:|---:|
| UD-Q4_K_XL | ✓ | ✓ | ✓ | ✓ |
| UD-Q3_K_XL | ✓ | ✓ | ✓ | ✓ |

Only after this matrix should MTP depth be deeply tuned.

## Sources

- https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp
- https://github.com/ggml-org/llama.cpp/issues/25618
- https://youtu.be/NjfHqiNHTxk  (VIDEO-derived claims)
- MACHINE: `HANDOFF-BACK.md`
