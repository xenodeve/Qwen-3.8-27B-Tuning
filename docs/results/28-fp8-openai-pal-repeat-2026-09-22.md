# FP8 OpenAI-path repetition with shared observations — 2026-09-22

**Result:** the fresh FP8 repetition through the OpenAI-compatible `gateway.9arm.co/v1/chat/completions` path passes the original8/8 hidden suite and visible12/12, but fails2/4 overlap-audit cases. It is not a fully accepted PAL task. Unlike result26's Anthropic path, this run exposes reasoning and per-request delivery observations.

This is a new measurement repeat of result26, not a replacement or a quality retry. The task prompt/baseline stayed byte-identical; only the gateway protocol changed to test the route DSH actually uses.

## Quality and workflow

| Measure | FP8 OpenAI-path repeat |
|---|---:|
| Original hidden | **8/8** |
| Visible final suite | **12/12** |
| Overlap audit | **2/4** |
| Test-first evidence | **RED→GREEN confirmed**;12 visible runs |
| Scope violations | 0 |
| Attempt through original verifier | **288.750s** |
| Client execution | 278.156s |
| Parent verification | 8.657s |
| Client-visible assistant messages | 30 |
| Read / Edit / fixed-test calls | 23 / 9 / 12 |
| Tool errors | 4 |
| Automatic compactions | 2 |

The same two overlap cases fail: an explicit `CLI_CLIENTS_CONFIG_PATH` directory physically inside a normal user directory is still classified by path rather than authority. The original8 score and visible green suite do not imply complete correctness.

## Comparable token and delivery metrics

Captured on33 complete upstream OpenAI requests with the shared sidecar schema:

| Metric | Observed value |
|---|---:|
| Reported input tokens, all requests | **940,375** |
| Reported output tokens, all requests | **44,353** |
| Reported reasoning tokens | **24,688** |
| Reasoning characters observed in SSE | **107,656** |
| Visible text characters | 48,915 |
| Tool argument characters | 21,506 |
| Maximum reported input in one request | 37,536 tokens |
| Median first body | 0.575s |
| Median first exposed reasoning | 0.575s |
| Median first visible text | 4.467s (15 requests exposed text) |
| Median HTTP-body end | 2.148s |
| Median body-chunk gap | 0.014s |
| **Effective request throughput** | **188.86 output tok/s** |
| **Post-first-semantic estimate** | **206.82 tok/s** |

The two rates are explicitly different from native decode:

- **Effective request throughput** = reported output tokens ÷ complete upstream request wall time.
- **Post-first-semantic estimate** = reported output tokens ÷ time from first semantic event to HTTP EOF, matching DSH's broad denominator shape but not proving the same implementation boundary.
- Native prefill/decode, GPU, queue time and weight hash remain unavailable from the gateway.

The DSH screenshot's `144 tok/s` is consistent with a different route/UI calculation: DSH source computes output tokens divided by post-first-token decode duration, excluding TTFT. It cannot be substituted directly for the benchmark's effective request rate.

## Comparison with result26 and local arms

| Configuration | Route | Output tokens | Reasoning | Effective observed rate |
|---|---|---:|---:|---:|
| FP8 result26 | Anthropic `/v1/messages` | 14,124 | 0 exposed chars | unavailable (legacy capture) |
| **FP8 repeat** | **OpenAI `/v1/chat/completions`** | **44,353** | **107,656 chars / 24,688 reported tokens** | **188.86 tok/s** |
| GSQ IQ3 | local llama.cpp | 9,957 | local trace available | native local counters in result25 |
| Swift Q6 | local llama.cpp | 10,580 | local trace available | native local counters in result25 |

The FP8 result26 versus repeat difference is **protocol-path evidence**, not proof of nondeterminism or a model change. The OpenAI path returns reasoning fields that DSH parses; the Anthropic path did not. Do not average these two FP8 rows into one score.

## Evidence and isolation

Primary independent verification root:
`C:/Users/xenod/AppData/Local/Temp/qwen-fp8-openai-independent-20260922`.

- Original hidden and overlap tests rerun in the pinned Docker sandbox.
- 33 timing sidecars re-read and joined without ambiguity.
- 2,230 task files independently rehashed; zero mismatches.
- Credential scan found zero occurrences.
- Original PAL HEAD/status unchanged; only allowed two files changed in the clone.
- No original project patch, default change, commit or push.

Full instrument gate after the unified apparatus: **2046 passed,3 skipped,9 warnings in318.11s**. Existing documentation broken links and tracker limitations remain unchanged. The shared metric apparatus is complete; further comparisons should use this route/boundary schema rather than mixing legacy whole-client ratios with native local decoder rates.
