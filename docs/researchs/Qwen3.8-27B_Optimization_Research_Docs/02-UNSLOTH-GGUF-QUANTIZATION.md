# Qwen3.8-27B GGUF / Quantization Documentation

## 1. Selected Artifact Source

**PRIMARY**

https://huggingface.co/unsloth/Qwen3.8-27B-GGUF

Current repository metadata:

```text
Author        unsloth
Architecture  qwen3_5
Base model    Qwen/Qwen3.8-27B
Format        GGUF
Tags          imatrix, conversational
License       Apache-2.0
```

Current Unsloth model documentation:

https://unsloth.ai/docs/models/qwen3.8

The supplied current page/screenshot reports Qwen3.8 GGUFs using **Unsloth Dynamic V3.0 (preview)** and calls out developer-role/tool-calling improvements.

## 2. Quant Decision History

The base model never changed. The default quant recommendation did.

```text
1. UD-Q3_K_XL
   ↓ AtomicChat fidelity-per-byte graph
2. AtomicChat AD-IQ3_S
   ↓ Unsloth agent/tool-specific support
3. UD-Q3_K_XL
   ↓ closer reading of Unsloth Qwen3.8 fidelity curve
4. UD-Q4_K_XL reliability-first hypothesis
```

Default recommendation changed three times after the first concrete selection.

## 3. Current Q3 vs Q4 Interpretation

From the supplied Unsloth Qwen3.8 graph, approximate Top-1 BF16 agreement:

```text
UD-Q3_K_XL ~92.4%
UD-Q4_K_XL ~96%
UD-Q5...   ~97%
Q8         ~98.5%
```

These rounded graph readings are useful as **quant-fidelity proxies**, not coding-success probabilities.

Do not calculate downstream benchmark score by multiplying by Top-1 agreement.

## 4. Measured Q4

**MACHINE**

```text
UD-Q4_K_XL disk size   16.69 GiB
label                   Q4_K - Small / mixed dynamic quant
```

The Qwen3.6-derived capacity proxy predicted roughly 17.6 GB, so the proxy was directionally sound.

## 5. Why Quant Is a Speed Knob on 12 GB VRAM

On the measured Q4 run:

```text
~10.2 GiB VRAM was free before model load
~505 MiB remained after load/fit
generation ~6.3–7.6 tok/s
```

Therefore a several-GB reduction in model footprint can change how many layers stay on GPU.

Quant affects:

```text
quality/fidelity
GPU residency
CPU-resident weight fraction
RAM headroom
long-context capacity
decode speed
```

This is why Q3-vs-Q4 moved ahead of fine KV/batch tuning.

## 6. Current Hypotheses

### Reliability

```text
UD-Q4_K_XL
```

Reasons:

- materially higher quant-fidelity proxy
- Unsloth Qwen3.8-specific developer/tool support
- likely lower long-horizon divergence risk

### Performance

```text
UD-Q3_K_XL
```

Reasons:

- smaller model
- more GPU-resident target weights
- more RAM left for KV/prompt cache
- may be especially attractive with MTP

## 7. New Candidate Interaction: Q3 + MTP

MTP requires extra runtime memory.

Q4 is already close to VRAM saturation, so MTP may force more Q4 weights to RAM.

Q3 can potentially:

```text
give up some quant fidelity
↓
recover several GB
↓
host more target weights + MTP state on GPU
↓
increase verified task throughput
```

Hence the critical matrix:

```text
Q4 baseline
Q4 + ngram
Q4 + MTP
Q3 baseline
Q3 + ngram
Q3 + MTP
```

## 8. AtomicChat

AtomicChat remains optional.

A prior user-provided chart made `AD-IQ3_S` attractive on fidelity-per-byte.

Later the argument became stronger in a 12 GB system because:

```text
1–2 GB smaller artifact
→ direct GPU residency gain
```

But AtomicChat and Unsloth charts use different metrics/evals. They are not numerically commensurable.

Only add one AtomicChat challenger if Q3/Q4 evidence leaves a real gap.

## 9. Selection Rule

Do not pick by:

```text
smallest file
highest Top-1 proxy
highest raw tok/s
```

Pick by:

> **Verified Successful Coding Tasks / Hour**

## Sources

- https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
- https://unsloth.ai/docs/models/qwen3.8
- https://huggingface.co/Qwen/Qwen3.8-27B
- VIDEO/USER screenshot: AtomicChat Qwen3.8 quant graph
- MACHINE: `HANDOFF-BACK.md`
