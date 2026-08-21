# Qwen3.8-27B Local Worker — Research & Optimization Docs

Prepared: **2026-08-18 (UTC+7)**

This folder is a structured research handoff for optimizing **Qwen3.8-27B** as a local coding-agent worker on a Windows workstation with an RTX 4070 SUPER 12 GB and 48 GB RAM.

## Documents

1. [`00-OPTIMIZATION-PLAN.md`](00-OPTIMIZATION-PLAN.md) — single execution plan and phase order
2. [`01-QWEN3.8-27B-DOCS.md`](01-QWEN3.8-27B-DOCS.md) — model facts, architecture, context, reasoning, tool use, caveats
3. [`02-UNSLOTH-GGUF-QUANTIZATION.md`](02-UNSLOTH-GGUF-QUANTIZATION.md) — quant source, Q3/Q4 decision history, fidelity vs residency
4. [`03-LLAMA.CPP-RUNTIME-DOCS.md`](03-LLAMA.CPP-RUNTIME-DOCS.md) — CUDA runtime, flags, memory placement, prompt cache, checkpoints, fit
5. [`04-SPECULATIVE-DECODING-MTP-NGRAM.md`](04-SPECULATIVE-DECODING-MTP-NGRAM.md) — built-in MTP, ngram-simple, safety/equivalence tests
6. [`05-OPENCODE-INTEGRATION.md`](05-OPENCODE-INTEGRATION.md) — llama-server → OpenCode provider/harness integration
7. [`06-OPENCLINK-XENO-INTEGRATION.md`](06-OPENCLINK-XENO-INTEGRATION.md) — OpenClink lane, master/worker ownership, Xeno policy
8. [`07-CONTEXT-KV-CACHE-T4-COMPACT.md`](07-CONTEXT-KV-CACHE-T4-COMPACT.md) — KV/context strategy, prompt reuse, T4-Compact implications
9. [`08-BENCHMARK-PROTOCOL.md`](08-BENCHMARK-PROTOCOL.md) — experimental design, metrics, matrix, pass/fail gates
10. [`09-SOURCE-LEDGER.md`](09-SOURCE-LEDGER.md) — all primary, project, machine, video-derived, and proxy sources

## Evidence labels used throughout

- **PRIMARY** — official model/runtime/project documentation or source code
- **MACHINE** — measured directly on the operator's workstation
- **PROJECT** — user's OpenClink/Xeno repository documents
- **VIDEO** — technical claims extracted from the supplied video; hypotheses until independently verified
- **PROXY** — evidence from the same architecture family but not exact Qwen3.8-27B
- **INFERENCE** — engineering inference from multiple sources; must be validated locally

The package intentionally distinguishes facts from hypotheses. Do not silently promote a proxy, video claim, or engineering inference to an exact Qwen3.8 fact.
