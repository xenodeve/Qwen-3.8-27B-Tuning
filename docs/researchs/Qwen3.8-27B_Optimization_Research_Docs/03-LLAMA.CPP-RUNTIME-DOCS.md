# llama.cpp Runtime Documentation for Qwen3.8-27B

## 1. Current Runtime

**MACHINE**

```text
Path    C:\AI\llama.cpp-cuda
Build   b10472
Commit  60eeeb608
CUDA    12.4 package
Device  CUDA0: RTX 4070 SUPER
```

The earlier Winget install was a Vulkan build. CUDA was installed separately and is the chosen baseline.

## 2. Why llama.cpp Instead of Ollama

Ollama remains convenient, but this workload needs explicit control over:

```text
GPU layers
fit/headroom
KV placement
KV precision
speculative decoding
context depth
prompt cache
context checkpoints
batch/ubatch
```

llama.cpp exposes these directly.

## 3. Important Current Flags

Primary docs:

https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

Verify every copied flag against **local b10472 `--help`** because master can move ahead of the installed build.

Key families:

```text
-c, --ctx-size
-ngl, --n-gpu-layers
-fit, --fit
-fa, --flash-attn
-b, --batch-size
-ub, --ubatch-size
-np, --parallel
-dev, --device
-ctk, --cache-type-k
-ctv, --cache-type-v
-nkvo, --no-kv-offload
--jinja
--chat-template-kwargs
```

Opus verified these in b10472.

## 4. `--fit` and VRAM Headroom

Master llama.cpp docs currently expose:

```text
--fit on|off
--fit-target MiB
--fit-ctx N
```

Master default `fit-target` is documented as 1024 MiB.

The measured b10472 Q4 run ended with only ~505 MiB free, and free VRAM before launch varied substantially with desktop workload.

Action:

```text
inspect local b10472 --help for fit-target
```

If supported, benchmark explicit margins instead of relying on an unstable auto-fit boundary.

## 5. Flash Attention

Keep Flash Attention enabled/on or auto for baseline unless a correctness/regression test says otherwise.

It is also useful when quantized V cache is involved.

## 6. KV Placement

llama.cpp supports:

```text
--no-kv-offload
```

to keep KV on CPU instead of GPU.

Do not assume CPU KV is faster.

A/B:

```text
GPU/default KV
vs
CPU KV
```

The reason to test CPU KV is that freed VRAM may retain more target weights on GPU.

## 7. KV Types

Supported local b10472 types observed by Opus include:

```text
f32
f16
bf16
q8_0
q4_0
q4_1
iq4_nl
q5_0
q5_1
```

Initial main-KV quality/memory test:

```text
F16
vs
Q8_0
```

Do not jump directly to aggressive Q4 KV for production.

## 8. Prompt Caching

Master server docs:

```text
--cache-prompt / --no-cache-prompt
default enabled
```

Request-level `cache_prompt=true` means common-prefix KV can be reused so only the changed suffix is processed.

Important caveat from official docs:

different batching paths are not guaranteed bit-for-bit identical, so prompt caching can produce nondeterminism.

For agent economics, measure actual prefix reuse with OpenCode.

## 9. Context Checkpoints

Master docs currently expose:

```text
--ctx-checkpoints     default 32
--checkpoint-min-step default 8192
```

Qwen qwen3_5 is a hybrid recurrent/full-attention architecture family. Do **not** disable checkpoints merely to save memory until their role in recurrent-state reuse is understood.

## 10. Host Prompt Cache

Master docs currently expose:

```text
--cache-ram N
default 8192 MiB
```

On the measured Q4 baseline only ~11.35 GB host RAM remained free.

Therefore blindly allowing an 8 GB host prompt cache risks Windows paging once OpenCode/Claude/browser processes are included.

Candidate test:

```text
2048 MiB
4096 MiB
```

Measure cache benefits vs RAM pressure.

## 11. Parallelism

Target is one interactive coding worker.

Keep:

```text
--parallel 1
```

This avoids paying memory for serving concurrency that the workflow does not need.

## 12. Vision

The base model is multimodal, but the coding worker should start with:

```text
--no-mmproj-auto
```

This saves memory and isolates text-agent behavior.

## 13. Reasoning / Chat Template

The local Qwen template supports reasoning effort.

Use `--chat-template-kwargs` or request-level supported parameters only after verifying how b10472 propagates them.

Separate:

```text
reasoning parsing
from
reasoning preservation
```

For coding-agent baseline:

```text
parse reasoning correctly
do not preserve old reasoning by default
```

because retaining large thinking traces can inflate future context.

## 14. Metrics and Monitoring

Capture on every launch:

```text
nvidia-smi
free VRAM
free RAM
actual layer split
server load log
context size
KV type
KV placement
spec mode
reasoning profile
sampling profile
```

The PowerShell automation must tolerate normal native-tool output written to stderr; Opus already encountered this with llama.cpp/nvidia-smi.

## 15. Backend Choice

CUDA remains baseline for NVIDIA.

Vulkan can be retained as an A/B fallback only if a CUDA-specific regression is found.

## Sources

- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md
- https://github.com/ggml-org/llama.cpp
- MACHINE: `HANDOFF-BACK.md`
