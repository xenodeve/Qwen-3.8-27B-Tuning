# Can the EXL3 server speak the Anthropic Messages API itself? — survey, 2026-09-04

**External material plus a read of the two local trees. Nothing here is a
measurement.** Asked because `claude-xeno-exl3` needs LiteLLM between Claude
Code and `serve_openai.py`, while `claude-xeno` talks to `llama-server`
directly.

## What exists today

| tree | server | Anthropic `/v1/messages` |
|---|---|---|
| ExLlamaV3 upstream 1.4.6 (`C:\AI\exllamav3-src`) | **none** — a library with `examples/` only | no |
| Mia-AiLab fork 1.4.2 (`C:\AI\exllamav3-mia`, one squashed commit `63b32f0`) | `tools/serve_openai.py`, aiohttp, three routes: `/v1/models`, `/health`, `/v1/chat/completions` | no; the word `anthropic` does not occur in the tree |
| TabbyAPI (the usual ExLlamaV3 server) | OpenAI + Kobold | **not merged.** PR [#442 "feature/anthropic messages api"](https://github.com/theroyallab/tabbyAPI/pull/442) by klement, opened 2026-08-01, +3,859 / −17 over 21 files, **0 comments, 0 reviews, untouched since the day it was opened** |
| `llama-server` | C++ | yes since PR [#17570](https://github.com/ggml-org/llama.cpp/pull/17570) ([HF blog](https://huggingface.co/blog/ggml-org/anthropic-messages-api-in-llamacpp)); the served build 10729 has it and `claude-xeno` uses it |

So: **ExLlamaV3 has no Anthropic endpoint anywhere merged.** The only code
that does it for this engine is an unreviewed PR.

## What PR #442 contains (the reusable part)

The endpoint is pure Python over TabbyAPI's OpenAI path — the same shape as
our fork's server, which is why it is worth reading:

| file | lines | what |
|---|---|---|
| `endpoints/Anthropic/utils/convert.py` | 499 | Anthropic request -> OpenAI chat request (system, content blocks, `tool_use`/`tool_result`, images, thinking blocks) and the reverse for the non-stream response |
| `endpoints/Anthropic/utils/stream.py` | 380 | OpenAI chunk stream -> Anthropic SSE (`message_start`, `content_block_start/delta/stop`, `message_delta`, `message_stop`; `thinking_delta`, `input_json_delta`) |
| `endpoints/Anthropic/types/messages.py` | 289 | pydantic request/response types |
| `endpoints/Anthropic/utils/messages.py` | 164 | template_vars, mid-conversation system messages |
| `endpoints/Anthropic/router.py` | 134 | `/v1/messages`, `/v1/messages/count_tokens`, model card fields |
| `endpoints/Anthropic/errors.py` | 105 | Anthropic error envelope |
| `tests/test_anthropic_messages.py` | 1,572 | the test suite — the most valuable file if we port |

It also reports prefix-cache hits as `cache_read_input_tokens`, which is what
Claude Code shows as cached tokens.

## Three ways to get there, with the cost of each

1. **Keep LiteLLM** (status quo). Works, and effort now flows end to end
   (`exllama3-platform-2026-09-03.md`, "Reasoning effort"). Cost: one Python
   process, and a translation layer we do not control — it silently dropped
   `output_config` without `thinking` beside it, and its adapter version
   decides what reaches the model. Pin `litellm==1.90.0`.
2. **Port PR #442's converter + stream into `serve_openai.py`** as a fourth
   route. ~900 lines of the PR are the converter, stream and types; our
   server already has `normalize_messages`, `build_tool_schemas`,
   `parse_tool_calls`, and a streaming loop that emits `reasoning_content`
   and `tool_calls` deltas — the port maps Anthropic blocks onto those, not
   onto the engine. Test-first with the **captured Claude Code request
   shape** (this session's echo: `thinking{adaptive}` + `output_config{effort}`,
   31 tools, `anthropic-beta` list, `?beta=true`) and PR #442's suite
   adapted. Expected size here: 400–600 lines + tests. Payoff: no proxy,
   byte-identical prefix under our control, `count_tokens` for Claude Code's
   context meter, cached-token reporting.
3. **Switch the engine server to TabbyAPI at PR #442's branch.** Gets the
   endpoint with its tests for free, but TabbyAPI wraps **upstream 1.4.6**,
   not the fork: the served recipe's MTP head, `-cq 4`, `-tp -tpb native` need
   re-checking there, and DFlash2/DSpark/NVFP4-KV (fork-only) are gone. The
   fork's live-timing lines and `--extra` would also be lost. A branch with
   zero review is not a base to serve from.

**Recommendation:** 2 if the proxy is to go; 1 until then. 3 only if TabbyAPI
merges #442 and the recipe is re-paired on upstream.

**Done the same day — option 2, issue #73.** `tools/anthropic_compat.py`
(pure translator, 23 tests in `bench/tests/test_exl3_anthropic_messages.py`)
and two routes in `serve_openai.py`: `/v1/messages` pipes the translated
request to the server's own `/v1/chat/completions` over loopback and
translates the JSON or SSE reply back; `/v1/messages/count_tokens` tokenises
through the same translation. Verified live 2026-09-04: non-stream, stream
with a `tool_use` block, `count_tokens`, the error envelope, and Claude Code
`-p` straight at :8000 with `--effort low` printing `reasoning effort = low`
in the serve log. `claude-xeno-exl3` no longer starts LiteLLM; the yaml stays
as a fallback. Known edge, pre-existing on the OpenAI route: a reply cut by
`max_tokens` before `</think>` comes back as text, not a thinking block, on
the non-stream path only.

**Moved out of the fork tree the same afternoon:** the routes and translator
live in `qwen38-tuning/serving/exl3/` (`anthropic_routes.py`,
`anthropic_compat.py`) and `server.py` there is the fork's file plus marked
hooks; the fork's `tools/serve_openai.py` is pristine again. Keep-alive added:
`message_start` immediately, then a `ping` every 5 s of prefill silence (ten
pings before content on a ~25K prompt), the same fix llama-server got with
`--sse-ping-interval 5`.

## Sources
[TabbyAPI PR #442](https://github.com/theroyallab/tabbyAPI/pull/442) ·
[TabbyAPI wiki, Usage](https://github.com/theroyallab/tabbyAPI/wiki/03.-Usage) ·
[llama.cpp PR #17570](https://github.com/ggml-org/llama.cpp/pull/17570) ·
[Anthropic Messages API in llama.cpp](https://huggingface.co/blog/ggml-org/anthropic-messages-api-in-llamacpp) ·
[LM Studio Anthropic compat](https://lmstudio.ai/docs/developer/anthropic-compat) ·
[Ollama Anthropic compat](https://docs.ollama.com/api/anthropic-compatibility) ·
[Claude Code with a local LLM, no proxy](https://www.shawnmayzes.com/ai-engineering/claude-code-local-llm-2026/) ·
local trees `C:\AI\exllamav3-src` (1.4.6) and `C:\AI\exllamav3-mia` (1.4.2), read 2026-09-04.
