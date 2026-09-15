# spark1.3 observability: structured logs + immediate stall triggers

Date: 2026-09-16 (Asia/Bangkok)
Component: `claude-spark1.3` launcher chain
Status: implemented and verified; one upstream defect found and left open (see
"New finding")

Scope: build the debugging instrumentation requested after the repeated
`✻ Waiting for API response` reports, on both sides of the bridge, with a
trigger that fires the moment Claude Code reports a stall.

## Topology instrumented

```text
Claude Code (claude.exe 2.1.258)
  |  Anthropic /v1/messages  (+ x-spark13-req correlation header)
  v
:4003 spark13_wrap.py        SSE relay + protocol repair   -> src="wrap"
  |  OpenAI/Anthropic bridge
  v
:4002 litellm 1.90.0         spark13_callbacks.py          -> src="lite"
  |  /v1/responses
  v
https://opencode.ai/zen/go
```

Plus `src="client"` (Claude Code's own debug log) and `src="watch"` (the log
watcher). One request id (`req`) is shared by `wrap` and `lite` events because
the wrapper stamps `x-spark13-req` and the LiteLLM callback reads it back out of
`request_data["proxy_server_request"]["headers"]`.

## Files

| File | Role |
| --- | --- |
| `~/.claude/spark13_log.py` | shared JSONL timeline, incident/trigger writer, snapshot, CLI (`tail`, `state`, `incidents`, `snapshot`) |
| `~/.claude/spark13_wrap.py` | SSE relay: per-request lifecycle, per-chunk and per-event timing, stall thresholds, protocol-fix logging |
| `~/.claude/spark13_callbacks.py` | LiteLLM `CustomLogger`: per-chunk visibility from inside the proxy |
| `~/.claude/spark13_watch.py` | tails Claude Code's debug log; triggers on the client's own stall lines |
| `~/.claude/spark13_ensure.py` | start-or-verify each component (pid + port, never double-start) |
| `~/.claude/claude-spark1.3.bat` | brings the stack up, runs Claude Code with `--debug-file` |
| `~/.hermes/scripts/spark13-incident-feed.py` | cron feed that posts new incidents to Discord |

Logs (all under `~/.claude/logs/`):

```text
spark13-events.jsonl      one JSON object per event, all components
spark13-stack.log         human-readable mirror of the same events
spark13-incidents.jsonl   one record per triggered incident (with snapshot)
spark13-alerts.jsonl      short trigger notices (cron feed)
spark13-state.json        in-flight request registry
spark13-pids.json         component pid/port map
cc-debug/spark13.txt      Claude Code `--debug-file` output (verbose)
```

## Trigger thresholds

Derived from Claude Code's own constants in `bin/claude.exe` (v2.1.258):

| Threshold | Default | Meaning |
| --- | --- | --- |
| `SPARK13_GAP_WARN` | 5 s | silent upstream: logged |
| `SPARK13_GAP_BANNER` | 20 s | `Rbt=20000` - the client renders `Waiting for API response` |
| `SPARK13_GAP_IDLE` | 180 s | `CLAUDE_STREAM_IDLE_TIMEOUT_MS` reached - the client aborts the stream |

Client-side patterns that trigger immediately (from `spark13_watch.py`):
`Streaming idle warning`, `Streaming idle timeout`, `Streaming idle timeout
(byte-level)`, `Stream completed with message_start but no content blocks
completed`, `Stream idle timeout after thinking-only yield`, `Streaming
completed with N stall(s)`, `Retrying without streaming`,
`Waiting for API response`, `[Stall] stream_idle_partial ...`,
`Slow first byte: no stream chunk ...`. `[Stall]` ticks repeat every ~15 s, so
that trigger carries a 90 s cooldown.

Each trigger writes an incident record containing: the triggering line, the
in-flight request registry, the local socket table, the matching process list
with memory, and the last 25 timeline events.

## Verification performed

1. `spark13_watch.py --selftest` - 9/9 stall patterns produce a trigger
   (runs in an isolated `SPARK13_LOG_DIR`).
2. Watcher live tail - appended real stall lines to a file while the watcher
   followed it; `client.retry_without_streaming` and
   `client.banner_literal` incidents fired within ~0.2 s.
3. Induced upstream stall (`blackhole.py` on :4999, wrapper on :4005):
   incidents at 5.4 s (`over_warn_threshold`) and 20.2 s
   (`over_client_banner_threshold`), with the client receiving pings every 3 s
   in between.
4. Live production request captured end to end (`req w-65c2f91c`).
5. LiteLLM callback loaded from the yaml: `module.import` + per-chunk
   `lite.chunk` events for a real request.
6. Claude Code `--debug-file` produces `cc-debug/spark13.txt` (462 lines) and
   `CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000` is visibly applied
   (`idleDeadlineMs=1800000`).

## New finding: LiteLLM flushes the Anthropic SSE stream in batches

Captured 2026-09-16 00:0x with a direct client on `:4002` (wrapper bypassed):

```text
headers at 2.44s
  frame at 2.44s  message_start        372 bytes
  frame at 2.44s  message_start        374 bytes   <- duplicate, see below
  frame at 2.44s  content_block_start  137 bytes
  frame at 5.47s  content_block_stop    80 bytes
  frame at 5.47s  content_block_start  129 bytes
  frame at 5.47s  content_block_delta  154 bytes
  frame at 5.47s  content_block_stop    80 bytes
  frame at 5.47s  message_delta        172 bytes
  frame at 5.47s  message_stop          56 bytes
```

Frames arrive in bursts, not per token. For the large production request
(`req w-65c2f91c`: 232 KB body, `thinking: adaptive`,
`output_config.effort=xhigh`) the whole 1497-byte response body arrived in one
burst 38.5 s after the request, while Claude Code logged:

```text
[WARN] [Stall] stream_idle_partial lastChunkAgeMs=14999 bytesTotal=35 idleDeadlineMs=1800000
[WARN] Slow first byte: no stream chunk 30.0s after request sent (attempt 1)
[DEBUG] [API:timing] first byte after 38524ms
```

Consequence: the client's stall tracker has no message event to reset it for
20-38 s, so the `Waiting for API response` banner appears even though the relay
is healthy and pinging. `event: ping` does not help - the stream-event
dispatcher drops it before the tracker
(`function HJ(e){ return e.type==="ping" }`,
`function gat(e,n,r){ ... if(HJ(e.event)) return; ... }`).

Two candidate mechanisms remain to be separated:

- Zen Go itself may answer the `/v1/responses` call in one shot for
  `effort=xhigh` (no incremental deltas to forward).
- LiteLLM's Responses->Anthropic bridge may buffer before writing.

Next probe: stream the Zen Go `/v1/responses` endpoint directly and time frame
arrivals, then compare with `:4002`.

The duplicate `message_start` is emitted by LiteLLM (two frames in the same
burst); `spark13_wrap.py` keeps deduplicating it and closing unclosed content
blocks before terminal events.

## How to use it

```bash
python ~/.claude/spark13_ensure.py all        # bring the stack up
python ~/.claude/spark13_ensure.py status     # pids, listeners, probes
python ~/.claude/spark13_log.py tail 40       # newest timeline events
python ~/.claude/spark13_log.py incidents     # triggered incidents
python ~/.claude/spark13_log.py snapshot      # current state + sockets
python ~/.claude/spark13_watch.py --replay FILE
```

Discord delivery: cron job `spark13 stall trigger feed`
(`~/.hermes/scripts/spark13-incident-feed.py`, every 1 min) posts new incidents
to the originating thread and stays silent when nothing new happened.
