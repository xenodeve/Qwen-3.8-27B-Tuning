# OpenClink + Xeno Integration Documentation

## 1. Target Lane

```text
Claude Code / Opus 5
→ Xeno policy/workflow
→ OpenClink delegation
→ OpenCode
→ llama-server
→ Qwen3.8
```

## 2. Existing OpenClink Support

**PROJECT**

Repository:

https://github.com/xenodeve/openclink

Current known file:

```text
conf/cli_clients/opencode.json
```

already defines an `opencode` client with:

```text
default
planner
codereviewer
```

roles.

Clink docs:

https://github.com/xenodeve/openclink/blob/main/docs/tools/clink.md

describe isolated CLI subagents, context separation, and per-call model selection.

## 3. Do Not Confuse Two Local-Model Paths

OpenClink `.env.example` also supports:

```text
CUSTOM_API_URL
CUSTOM_MODEL_NAME
```

for direct OpenAI-compatible model access.

That is a different path.

Desired coding lane:

```text
OpenClink
→ OpenCode harness
→ llama.cpp
```

Direct local API may still be useful for model-to-model utilities, but it is not the coding-worker lane being optimized here.

## 4. Ownership Rules

### Opus master owns

```text
task decomposition
architecture
acceptance criteria
integration
security/trust boundaries
final verification
escalation decisions
```

### Qwen local worker owns

```text
bounded code leaves
repo exploration
refactors
tests
debugging
documentation
bulk transformations
cheaply-verifiable work
```

Principle:

> **A delegated green is not a system green.**

## 5. Why Local Quant Quality Can Be Lower Than Master Quality

The system is designed to exploit:

```text
cheap local tokens
+
bounded delegation
+
strong verification
```

Therefore Q3 may be production-optimal even if Q4 has a higher fidelity proxy, if Q3's failures remain cheap and detectable and its throughput advantage is large.

## 6. Xeno Reliability Layers

Working model:

```text
skills             policy/workflow
OpenClink          delegation transport
software/hooks     mechanical enforcement
tests/traces       evidence
Opus/reviewer      judgment
```

## 7. Context Isolation

OpenClink CLI subagents are valuable because high-input delegated work can occur in fresh context and return a compact result.

This reduces master-context pollution.

## 8. Production Benchmark

The final benchmark must be run through the actual lane:

```text
Opus
→ OpenClink
→ OpenCode
→ Qwen
```

not only raw llama-server.

The harness can materially change real coding-agent performance.

## Sources

- https://github.com/xenodeve/openclink
- https://github.com/xenodeve/openclink/blob/main/conf/cli_clients/opencode.json
- https://github.com/xenodeve/openclink/blob/main/.env.example
- https://github.com/xenodeve/openclink/blob/main/docs/tools/clink.md
- PROJECT: xeno-skills planning documents referenced in prior handoffs
