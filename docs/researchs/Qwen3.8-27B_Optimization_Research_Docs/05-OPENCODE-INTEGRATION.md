# OpenCode Integration Documentation — Qwen3.8 via llama.cpp

## 1. Role

OpenCode is the **coding-agent harness**.

It is intentionally kept between OpenClink and llama.cpp:

```text
OpenClink
→ OpenCode
→ llama-server
→ Qwen
```

not:

```text
OpenClink
→ raw model API
```

because the local worker needs repository/file/shell/test loops.

## 2. Official Custom Provider Support

Primary docs:

https://opencode.ai/docs/providers

OpenCode explicitly supports OpenAI-compatible custom providers with:

```text
@ai-sdk/openai-compatible
options.baseURL
models
limit.context
limit.output
```

The current provider docs include a llama.cpp example pointing OpenCode to:

```text
http://127.0.0.1:8080/v1
```

## 3. Example Configuration

Verify against current schema before production:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "llama.cpp/qwen38-q4",
  "provider": {
    "llama.cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "qwen38-q4": {
          "name": "Qwen3.8-27B UD-Q4_K_XL",
          "limit": {
            "context": 262144,
            "output": 32768
          }
        }
      }
    }
  }
}
```

Add Q3 as a separate model ID when ready.

## 4. Integration Gates

Before calling OpenCode healthy:

```text
model listed
plain chat works
developer/system role behaves
tools round-trip
nested object args survive
tool result can be consumed
repeated tool loop works
reasoning is not incorrectly exposed as final content
```

## 5. Compaction During Benchmarking

OpenCode compaction can hide the true context boundary.

During controlled context experiments:

```text
disable automatic/preflight compaction where supported
log any recovery/manual compaction
```

Do not assume `auto=false` means no overflow recovery path exists; verify current version behavior.

## 6. Prefix Stability

llama-server prompt cache is only valuable if OpenCode preserves a common serialized prefix.

Measure whether OpenCode changes:

```text
system prompt
tool ordering/schema serialization
skill block ordering
history serialization
reasoning fields
```

between turns.

This is a critical agent-performance metric.

## 7. OpenCode Task Benchmark

Use a small controlled repo:

```text
inspect
→ edit one bounded target
→ run tests
→ observe
→ repair
→ rerun
→ return exact evidence
```

Metrics:

```text
task success
wrong edits
unnecessary edits
tool failures
test recovery
wall time
prompt reuse
```

## Sources

- https://opencode.ai/docs/providers
- https://opencode.ai/docs/config/
- https://opencode.ai/v2/docs/compaction
