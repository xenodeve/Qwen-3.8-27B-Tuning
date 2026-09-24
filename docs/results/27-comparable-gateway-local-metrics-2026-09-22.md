# Comparable gateway/local observations — 2026-09-22

**Status:** measurement apparatus updated and verified offline; no new PAL quality run was launched in this slice. A fresh OpenAI-path FP8 repetition remains pending user-controlled execution because the auto command classifier blocked the live command.

Tracking: issue #92. [Plan](../plans/2026-09-21-fp8-pal-control.md). Existing quality results remain immutable: [result25](25-remaining-pal-workflow-2026-09-21.md), [result26](26-fp8-pal-control-2026-09-22.md).

## DSH route diagnosis

A read-only inspection of the running DSH installation and a harmless authorized three-arm gateway probe established the reason the screenshot differs from result26:

- DSH `@deepseek-ai/dsh`0.1.1-rc.2 uses `https://gateway.9arm.co/v1/chat/completions` through the OpenAI-compatible `arm` provider, model ID `qwen3.8-27b-fp8` and declared context128000.
- The benchmark's first FP8 attempt used Anthropic `/v1/messages` and received empty thinking blocks. These are different API paths and response contracts.
- DSH's parser accepts the first nonempty `reasoning_content`, `reasoning` or `reasoning_text` response delta. The probe received **119 reasoning characters** on both OpenAI requests and **0** on Anthropic adaptive `/v1/messages`; provider model IDs were `qwen3.8-27b-fp8` versus `vllm/Qwen/Qwen3.8-27B-FP8`.
- DSH's footer formula is verified from its installed code: `TTFT = firstTokenTime - stepStartTime`; `tok/s = outputTokens / (completedTime - firstTokenTime)`. The screenshot's144tok/s therefore excludes TTFT and is not a native GPU decode counter.

This diagnoses protocol/response-field divergence. It does not prove why the gateway's Anthropic route suppresses or translates reasoning internally, and it does not prove identical backend execution beyond the provider-reported IDs.

## Shared observation apparatus

Added a versioned pure `StreamObservation` schema for OpenAI and Anthropic SSE. It records only safe numeric/enum data: clock metadata, body offsets, request-relative timings, first body/text/reasoning/tool/argument/semantic events, terminal/EOF/end, chunk gaps, counts, finish reason and fixed error codes. It never stores headers, prompts, response text, tool arguments or credentials.

Integrated optional observations into:
- remote FP8 gateway final-hop capture, with exact request hashes and explicit canonical semantic hashes;
- local OpenAI upstream `AdapterServer`, reading each HTTP body exactly once and draining the same iterator after semantic `[DONE]`;
- local PAL remaining runner, copying observation sidecars before session sealing;
- offline `session_comparison.py` and `report-session-comparison.py`.

The comparison reporter now separates:
- provider usage/cache/read-write/reasoning counts;
- first-body, first-semantic, first-text, first-reasoning, terminal, EOF and end distributions;
- weighted effective request throughput (`output tokens / complete observed request wall`), never native decode;
- separately labeled post-first-semantic throughput estimate for DSH-style UI comparison;
- tool-use/result latency and interval union;
- fixed-test duration, compaction, quality and context;
- native prefill/decode/GPU/queue fields as unavailable unless genuine server counters exist.

Old result24–26 sessions are backfilled only for fields they actually contain. Their fine-grained upstream timings remain unavailable; no TCP-offset reconstruction was invented. The old FP8 result still reports provider usage 509,038 input and14,124 output tokens, legacy first-byte/elapsed rows, client/tool/test timings, and unavailable native rates.

## Verification

- Shared observer focused tests:33 passed before final integration.
- Unified observation/adapter/remote/PAL/report tests: **106 passed** in the final focused run.
- Full bench gate after all changes: **2046 passed,3 skipped,9 warnings in318.11s**. Skips are the existing idle8080 props check, installed-Studio binary absence check, and insufficient-GPU-budget DFlash switch check.
- Security review found no new high/medium vulnerability after the previous loopback authentication fix; the gateway requires an independent per-session local nonce before it can inject the provider credential.
- DSH/OpenAI probe used only a harmless arithmetic prompt; no PAL solution, private history or hidden verifier was sent. No installed profile was modified.

A fresh OpenAI-path PAL repetition is deliberately not claimed here. The dry-run protocol is frozen as `pal-fp8-openai-observed-v2`, preserving the original task hash and labeling it a new measurement repeat rather than replacing result26. No defaults or original projects changed.
