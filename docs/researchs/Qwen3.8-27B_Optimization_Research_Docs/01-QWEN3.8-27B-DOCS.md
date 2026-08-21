# Qwen3.8-27B — Model Documentation for Local Agent Optimization

Prepared: 2026-08-18

## 1. Exact Model Identity

**PRIMARY — Hugging Face metadata**

Repository:

https://huggingface.co/Qwen/Qwen3.8-27B

Current metadata checked in this research:

```text
Author         Qwen
Parameters     27,781.4M (~27.78B)
Architecture   qwen3_5
Task           image-text-to-text
Model class    AutoModelForMultimodalLM
License        Apache-2.0
```

This confirms Qwen3.8-27B is a dense ~27.8B multimodal model in the `qwen3_5` architecture family.

## 2. Local GGUF Evidence

**MACHINE**

The local Unsloth GGUF load reported:

```text
64 main blocks
plus an additional NextN / MTP block
256K model-context capability was confirmed in the research/handoff
```

The additional block contains MTP/NextN tensors.

## 3. Architecture Family — Important Proxy

The exact Qwen3.8 config internals were not fully available through indexed primary docs at research time, so the following is explicitly a **PROXY from Qwen3.5-27B**, which shares the `qwen3_5` architecture identifier.

Primary family reference:

https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json

Qwen3.5-27B text config reports:

```text
num_hidden_layers        64
full_attention_interval  4
num_attention_heads      24
num_key_value_heads      4
head_dim                 256
max_position_embeddings  262144
mtp_num_hidden_layers    1
```

Its stack is hybrid:

```text
linear/recurrent attention layers
+
periodic full-attention layers
```

A 3:1 pattern is used in this family (three linear/recurrent layers per full-attention layer).

**Do not silently claim every internal numeric field above is proven identical for Qwen3.8.** It is architecture-family evidence. The local Qwen3.8 GGUF does, however, independently confirm 64 main blocks + an MTP block.

## 4. Why the Hybrid Architecture Matters

A naïve 64-layer Transformer KV calculation overestimates long-context KV if only a subset of layers use full attention.

**INFERENCE / FAMILY-DERIVED**

If Qwen3.8 follows the same dense qwen3_5 text configuration:

```text
16 full-attention layers
4 KV heads
256 head dimension
K + V
```

then full-attention KV is approximately:

```text
32,768 scalar values per token
```

Approximate storage for this component:

| Context | F16 | Q8 | Q4 |
|---:|---:|---:|---:|
| 64K | ~4 GiB | ~2 GiB | ~1 GiB |
| 128K | ~8 GiB | ~4 GiB | ~2 GiB |
| 256K | ~16 GiB | ~8 GiB | ~4 GiB |

This does **not** include recurrent-state memory, graph buffers, checkpoints, model weights, application memory, or allocator overhead.

Treat as a planning estimate, not exact Qwen3.8 measured allocation.

## 5. MTP / NextN

**MACHINE**

Local GGUF showed NextN/MTP tensors after the main block stack.

**PRIMARY — llama.cpp source for qwen35 family**

https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp

The implementation:

- loads NextN/MTP metadata
- appends MTP decoder block(s) beyond the main stack
- provides `LLM_GRAPH_TYPE_DECODER_MTP`
- currently asserts support for a single MTP block

This strongly supports using Qwen3.8's built-in NextN head through llama.cpp `draft-mtp`, subject to local b10472 compatibility testing.

## 6. Multimodal Capability

Official metadata classifies Qwen3.8-27B as image-text-to-text.

For the coding-worker runtime the current plan deliberately uses:

```text
--no-mmproj-auto
```

therefore the local agent instance is text-only.

This is a deployment choice, not a limitation of the base model.

## 7. Reasoning Effort

**MACHINE**

Local chat-template inspection found:

```text
default reasoning_effort = xhigh
high → xhigh
accepted levels include low | medium | xhigh
```

This is performance-critical at 6–7 tok/s.

Reasoning effort should be treated as an orchestration/routing variable:

```text
low     easy leaf
medium  likely default candidate
xhigh   hard bounded leaf
```

This mapping is a hypothesis until verified on real tasks.

## 8. Sampling

From the supplied current Unsloth Qwen3.8 documentation, the thinking baseline was:

```text
temperature       1.0
top_p             0.95
top_k             20
min_p             0.0
presence_penalty  0.0
```

The local server had `min_p=0.05`, so that should be corrected for a vendor-aligned test.

**PROXY:** Qwen3.5 family documentation also motivates a lower-temperature precise-coding candidate. Therefore test:

```text
temp 1.0  official/general baseline
temp 0.6  precise-coding candidate
```

Changing temperature is not a free speed optimization: it changes output behavior.

## 9. Tool Calling

The model template observed locally supports capabilities including:

```text
tool calls
parallel tool calls
object/nested arguments
system role
reasoning effort
reasoning preservation
```

The internal model-side representation is XML-like. llama.cpp must parse that back into OpenAI-compatible `tool_calls`.

Capability flags are not enough; round-trip behavior must be tested.

## 10. Benchmarks

The earlier user-provided official Qwen3.8 benchmark screenshot contained these Qwen3.8-27B values:

```text
TerminalBench 2.1   73.0
SWE-bench Pro       61.7
NL2Repo             42.3
DeepSWE 1.1         42.2
QwenSWEBench        79.0
CoWork              70.7
```

Claude/Opus later reported that this transcription matched the published table.

Important methodology note from the supplied official text:

Several coding evaluations used a Claude Code harness with:

```text
temperature 1.0
top_p 0.95
256K context
```

Therefore these are model+harness results, not raw-model-only scores.

## 11. Exact Facts vs Family Proxies

### Exact / directly supported

```text
Qwen3.8-27B exists officially
~27.78B parameters
architecture identifier qwen3_5
multimodal model
local GGUF has 64 main blocks + NextN block
local GGUF can expose Qwen tool/reasoning template capabilities
```

### Architecture-family proxy

```text
3:1 linear/full attention pattern
24 Q heads
4 KV heads
256 head dim
exact recurrent-state dimensions
```

Use these for planning but verify exact Qwen3.8 config if/when the official config is exposed.

## Sources

- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/README.md
- https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp
- https://unsloth.ai/docs/models/qwen3.8
- MACHINE: `HANDOFF-BACK.md`
