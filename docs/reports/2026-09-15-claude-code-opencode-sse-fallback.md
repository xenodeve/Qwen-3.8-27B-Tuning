# Engineering record — Claude Code Spark1.3 SSE fallback

> **See also:** [problem inventory](2026-09-16-spark13-problem-inventory.md) - every defect hit during this investigation (P1a-P5 client-visible, P6-P14 in our own tooling), with evidence and status.

**Date:** 2026-09-15  
**Status:** Mitigation applied; end-to-end validation in a real interactive session is still pending  
**Affected path:** `claude-spark1.3` → `127.0.0.1:4003` (`spark13_wrap.py`) → `127.0.0.1:4002` (LiteLLM) → OpenCode Go Responses API  
**Claude Code:** `2.1.258`  
**Model:** `muse-spark-1.3-contributor`

## Summary

Claude Code displayed `Waiting for API response` after a single request in the existing Spark1.3 session. The request reached the upstream and returned HTTP 200, but the Anthropic SSE stream emitted by LiteLLM was not a valid single Claude Code message stream. Claude Code abandoned streaming and retried the same request with `stream=false`.

The local wrapper now normalizes the stream before forwarding it: it removes duplicate `message_start` events, closes open `content_block` blocks before terminal message events, and keeps the SSE heartbeat. The wire-level probe passes after the wrapper restart. A real interactive Claude Code session after this final mitigation is still required before calling the incident closed.

## Symptom

The user saw:

```text
Waiting for API response · will retry ... · check your network
```

The existing session transcript contains the earlier Claude Code warning:

```text
Streaming response ended before any complete data was received.
Retrying without streaming. If this keeps happening, check any proxy or gateway between Claude Code and your model provider.
```

Source: `C:/Users/xenod/.claude/projects/C--AI/95086579-cc96-425e-b899-c434fe77e5d7.jsonl`, line `310`, timestamp `2026-09-15T14:11:11.496Z`.

## Reliable reproduction

The user reproduced the symptom after continuing the session and sending one request. The wrapper log captured the complete fallback sequence at `22:47` local time:

```text
22:47:01 request POST /v1/messages?beta=true
22:47:01 messages bytes=1056054 stream=True
22:47:01 downstream ping
22:47:04 upstream headers status=200
22:47:07 upstream stream end
22:47:07 request POST /v1/messages?beta=true
22:47:07 messages bytes=276274 stream=False
22:47:07 request POST /v1/messages?beta=true
22:47:07 messages bytes=276256 stream=False
22:47:12 response status=200 bytes=444
```

Source: `C:/Users/xenod/.claude/logs/spark13-wrapper.log`.

This distinguishes the incident from an upstream connection timeout: the wrapper sent a heartbeat, received upstream headers with status `200`, and the client still abandoned the stream before accepting it as complete.

## Root cause

Two independent defects were confirmed, and they explain the two different
symptoms the developer reported.

### Defect A — malformed Anthropic SSE (fixed)

The LiteLLM Anthropic bridge emitted an invalid event sequence for Claude Code in the short/empty-visible-text case. A raw stream capture through `127.0.0.1:4003` before the mitigation contained:

```text
ping
message_start
message_start
content_block_start (thinking)
message_delta
message_stop
```

The stream therefore had two protocol defects:

1. `message_start` was emitted twice for one response.
2. The `thinking` `content_block_start` was not followed by `content_block_stop` before `message_delta`/`message_stop`.

The upstream HTTP status and connection were healthy. The failure was at the Anthropic SSE protocol boundary between LiteLLM and Claude Code.

A separate early hypothesis blamed HTTP/1.1 chunk framing. A direct socket capture of LiteLLM exposes chunk framing, but Python `urllib` decodes the transfer framing before `spark13_wrap.py` reads it. That hypothesis is not the confirmed cause; the wrapper retains a defensive decoder, but the confirmed failure is the duplicate/incomplete Anthropic event sequence above.

## Why this produced `Waiting for API response`

Claude Code received an HTTP 200 and a heartbeat, but its stream parser could not construct one complete Anthropic message from the event sequence. Claude Code treated the stream as ended before a complete response existed, displayed the network-style waiting state, and retried without streaming. The fallback response then completed, which explains why the request eventually returned instead of failing as a normal HTTP error.

### Defect B — the client's stall tracker ignores `ping` (open, mitigations applied)

Measured 2026-09-15 23:00–23:35 with the diagnostic counters in
`spark13_wrap.py` (`[DBG-sse]`). On several large requests (1.12 MB bodies) the
wrapper received upstream headers `200` and then **no body bytes at all** for
60–120 s:

```text
23:31:28 messages bytes=1125055 stream=True
23:31:28 downstream ping
23:31:28 upstream connect pending
23:31:35 upstream headers status=200
   (no upstream data; no [DBG-sse] summary; stream still open at 23:34)
```

while the small requests in the same window completed normally
(`[DBG-sse] rx=2980 ... visible=219 stop=end_turn`).

Claude Code's own rules, read from `bin/claude.exe` (v2.1.258):

```text
Streaming idle timeout: no chunks received for <ms>s, aborting stream
cli_streaming_idle_timeout
Stream completed with message_start but no content blocks completed -
  triggering non-streaming fallback
Stream idle timeout after thinking-only yield - retrying streaming
```

and, critically, the stream-event dispatcher drops pings before the stall
tracker sees them:

```js
function HJ(e){ return e.type==="ping" }
function gat(e,n,r){ ... if(HJ(e.event)) return; ... }
```

Consequences:

- An HTTP 200 plus periodic `event: ping` does **not** keep Claude Code's
  request alive. The stall tracker advances only on real message events
  (`message_start`, `content_block_*`, `message_delta`, `message_stop`).
- When the upstream proxy holds the response body open with no events, Claude
  Code shows `Waiting for API response · will retry in …` and then falls back to
  a non-streaming request.
- The fallback is why the wrapper log shows a `stream=False` request right after
  every stalled stream (e.g. `23:31:28` stream, `23:31:36` follow-up).

Mitigations applied for Defect B (`~/.claude-spark1.3.json` env):

```json
"CLAUDE_STREAM_IDLE_TIMEOUT_MS": "1800000",
"CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS": "300000"
```

Both are first-class Claude Code knobs (`bin/claude.exe` reads
`CLAUDE_STREAM_IDLE_TIMEOUT_MS`, clamped to a 30-minute maximum, and
`CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS`). Raising them stops Claude Code from
aborting a healthy-but-slow OpenCode Go response and is the correct client-side
fix; it needs a Claude Code restart to take effect.

Still open: why LiteLLM sometimes returns `200` and withholds the whole body for
1–2 minutes on ~1.1 MB requests. That is upstream/proxy behaviour, not the
wrapper.

**Superseded in part 2026-09-16:** the body is not withheld once — LiteLLM
flushes the Anthropic SSE stream in **bursts**, not per token. Measured with a
direct client on `:4002`: frames arrived at 2.44 s and 5.47 s, and a production
request delivered its whole 1497-byte body in one burst at 38.5 s while Claude
Code logged `Slow first byte: no stream chunk 30.0s after request sent`. That
batch flush is what starves the client's stall tracker. Instrumentation, the
per-chunk evidence and the remaining mechanism question are in
[2026-09-16-spark13-observability-and-batch-flush.md](2026-09-16-spark13-observability-and-batch-flush.md).

## Mitigation

Updated:

```text
C:/Users/xenod/.claude/spark13_wrap.py
```

The wrapper now:

- emits an Anthropic `event: ping` heartbeat while the upstream is silent;
- normalizes upstream SSE event boundaries instead of relaying malformed event sequences byte-for-byte;
- exposes only the first `message_start` for one downstream response;
- tracks open `content_block` indexes;
- emits `content_block_stop` for blocks still open before `message_delta` or `message_stop`;
- keeps the existing local `count_tokens` fallback and tool-choice rewrite.

The wrapper process was restarted after the code change. The LiteLLM process remained the already-restarted healthy process on `:4002`.

## Validation

### Before mitigation — red

The real Claude Code path reproduced the fallback at `22:47:01`–`22:47:12` as shown above.

### After mitigation — wire-level green

After restarting the wrapper at `22:54:20`, a real SSE request through `:4003` produced:

```text
ping
message_start
content_block_start
content_block_stop
content_block_start
content_block_delta
content_block_stop
message_delta
message_stop
```

The probe observed exactly one `message_start`, a terminal `message_stop`, and two closed content blocks. The wrapper log also showed a heartbeat, upstream status `200`, and normal stream completion:

```text
22:54:37 downstream ping
22:54:39 upstream headers status=200
22:54:55 downstream ping
22:54:55 upstream stream end
```

Process verification after the restart:

```text
LiteLLM :4002 — listening, process created 2026-09-15 22:46:00
Wrapper :4003 — listening, process created 2026-09-15 22:54:20
```

### Still required

- Continue the real Claude Code interactive session after the final wrapper restart.
- Confirm that no new `Streaming response ended...` warning or non-stream fallback occurs.
- Confirm tool-use and long-silent reasoning, not only the short wire probe.

Until those checks pass, this record is a mitigation record, not a closed post-mortem.

## Follow-ups

1. Add a replay regression test for the captured malformed sequence: duplicate `message_start` plus an unclosed thinking block.
2. Keep `spark13-wrapper.log` as the evidence source for request mode, upstream headers, heartbeats, and stream end.
3. Close the GitHub issue only after a real Claude Code session passes the end-to-end acceptance checks above.
