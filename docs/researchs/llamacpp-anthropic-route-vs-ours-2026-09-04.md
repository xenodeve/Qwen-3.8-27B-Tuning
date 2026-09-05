# llama-server's Anthropic Messages route, read against ours — 2026-09-04

**Source read, not measurement.** `F:\llama-build\up` at 458681e1d (the tree
build 10729 came from): `tools/server/server-chat.cpp` 298–620
(`server_chat_convert_anthropic_to_oai`), `server-task.cpp` 731–960 and
1324–1460 (final and streaming formatters), `server-context.cpp` 3173–3182
(context overflow), `server.cpp` 263/280 (routes). Ours:
`qwen38-tuning/serving/exl3/anthropic_compat.py` + `anthropic_routes.py`.
Claude Code 2.1.258's binary was read for what the client actually consumes.

## Where the two agree

| | llama-server | ours |
|---|---|---|
| routes | `/v1/messages`, `/v1/messages/count_tokens` | same |
| system | string or text blocks concatenated → one `system` message | same (blocks joined with a blank line) |
| `tool_use` → `tool_calls` with `json.dumps(input)`; `tool_result` → `role: tool` after the assistant turn | yes | yes |
| tools → `function` with `input_schema` as `parameters` | yes (missing schema → `{}`) | yes (missing schema → tool skipped) |
| `stop_sequences` → `stop`; `temperature/top_p/top_k` pass through; `max_tokens` default | yes (4096) | yes (1024) |
| stop_reason | `end_turn` / `tool_use` / `max_tokens` | same |
| usage | `input_tokens = prompt − cached`, `cache_read_input_tokens` | same (from `timings.cached_tokens`) |
| stream shape | `message_start` → `content_block_start` (thinking, **no signature key**) → `thinking_delta`… → **`signature_delta ""`** → stop → text → tool_use with `input_json_delta` → `message_delta` → `message_stop` | same since 16:5x today (the signature_delta was the missing piece; before it Claude Code never rendered our thinking) |
| keep-alive during prefill | `--sse-ping-interval 5` | `ping` every 5 s (`anthropic_compat.pump`) |

## Where llama-server does more

1. **Billing-header normalisation** (`normalize_anthropic_billing_header`, PR #21793): rewrites the `cch=xxxxx;` stamp in Claude Code's system prompt to `fffff` because it changes per request and kills the prefix cache. **Read from the 2.1.258 binary: `cch=`, `cc_prev_req=` and `cc_prompt_id=` are added only for first-party OAuth (or Vertex) sessions.** A custom `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` profile gets `cc_version=…; cc_entrypoint=cli;` and nothing per-request — consistent with the 2026-09-02 serve logs where every turn hit the cache. So we do not need it today; the test to keep is "turn-2 `cached` ≠ 0 in the serve log" after any Claude Code update.
2. **Thinking blocks in history are kept** as `reasoning_content` on the assistant message. We drop them. For the Qwen3.8 template only the latest turn's reasoning is rendered anyway, and dropping keeps the prefix byte-stable — keep ours, but it is a documented difference.
3. **Images** (base64 / url, in user turns and inside `tool_result`) become `image_url` parts for the vision projector. Ours writes `[image omitted]` — the EXL3 file has no vision tower, so this is a capability gap of the artifact, not of the route.
4. **`message_start` carries the real `input_tokens` / `cache_read_input_tokens`** (llama-server emits it at the first decoded token, after prefill). Ours emits `message_start` before the upstream call (so pings can follow) with zeros, and the real numbers arrive in the final `message_delta`. Claude Code's context meter merges `message_delta.usage` last, so the end state is right; only the moment-of-first-token display differs.
5. **`stop_sequence`** is reported when a stop word fired; ours is always `null`.
6. **`metadata.user_id`** is forwarded (`__metadata_user_id`) for slot affinity with `-np > 1`; ours has one slot.

## Where ours does more

1. **Effort reaches the model.** llama-server maps only `thinking.type == "enabled"` + `budget_tokens` to `thinking_budget_tokens`; `output_config.effort` (what Claude Code 2.1.258 sends, with `thinking: {type: adaptive}`) is ignored, so **`claude-xeno` on llama-server runs the Qwen3.8 template default — xhigh — whatever `--effort` says**, the same defect we found and fixed on our side today. Fix on that side would be `chat_template_kwargs: {reasoning_effort: …}`, which llama-server passes through but Claude Code never sends.
2. **Live counters — retracted the same day.** We first streamed interim `message_delta.usage.output_tokens` every 32 tokens and `thinking_delta.estimated_tokens` as a per-block running total. Both are now gone and the stream matches llama-server: the interim usage events froze the live view (interim usage is opt-in, off), and Claude Code 2.1.258 consumes `estimated_tokens` as an *increment* per delta (`thinkingTokenEstimate += estimatedTokensDelta`, read from the binary), so a running total summed to N²/2 and the counter read **731.0k** after four minutes at ~25 tok/s (observed 2026-09-04 ~20:10). Without the field Claude Code estimates from the thinking text itself, which is what every llama-server user gets.
3. **`tool_choice: {type: tool, name}`** maps to the specific function; llama-server maps both `any` and `tool` to `required`.
4. **Error envelope** is Anthropic-shaped (`{"type":"error","error":{...}}`); llama-server returns its OpenAI-style error on this route.

## Where both fall short of Claude Code

- **Context overflow does not trigger Claude Code's auto-compaction.** Claude Code compacts only on an error matching `prompt is too long: N tokens > M`. llama-server says `request (N tokens) exceeds the available context size (M tokens)` (server-context.cpp:3182) — the exact text CORRECTIONS §15 records as "API Error: 400 request (54499 tokens) exceeds…" — and ours says `context/cache: …`. Neither matches. The fix is cheap on our side: pre-count and answer with the exact phrase.
- **Unknown block types are dropped silently** on both (`tool_reference`, `document`, `server_tool_use`, mid-conversation `role: system`).

## Closed the same evening (issue #74)

History thinking -> `reasoning_content`; mid-conversation `system` -> a marked user turn; unknown blocks reported (`unknown_blocks`, a warning line); `normalize_billing_header` (the `cch=` stamp); the pre-flight `prompt is too long: N tokens > M maximum` 400; a request JSONL log and `EXL3_CAPTURE_DIR`; `API_TIMEOUT_MS=3600000` in the profile; `tools/exl3-smoke.py`. Images and a real `input_tokens` in `message_start` stay open by design.

## What to take from it

1. Keep the stream shape byte-equal to llama-server's where Claude Code renders it (done for thinking; keep the test that pins `signature_delta`).
2. Add the `prompt is too long: N tokens > M` pre-check (ours only; issue to file).
3. Re-check the billing header after every Claude Code update: if `cch=` ever appears for a custom base URL, port `normalize_anthropic_billing_header` (five lines).
4. Note for the llama.cpp profile: effort is not honoured there; a `chat_template_kwargs` injection in our own route is the only way to get it, and headroom/`claude-xeno-direct` has no such hook today.
