# System impact register

One line per system-affecting change, newest on top. **Pointers, not copies** —
the full record lives in the file named, and duplicating it here is how two
versions drift apart.

| date | change | record |
|---|---|---|
| 2026-09-16 | Claude Code Spark1.3 `Waiting for API response`: two causes found and fixed in the wrapper — `read(65536)` withheld every response under 64 KiB until EOF (client first byte 38524 ms -> 2502 ms), and nothing held the client's stall tracker during Zen Go's silent one-shot reasoning phase (empty `thinking_delta` heartbeat; 0 vs 3 stall ticks with a real client). Automatic remediation now fires on every trigger (heartbeat nudge into the live stream, re-send when the upstream is silent or returns 429/5xx before any output, restart a dead wrapper/LiteLLM and verify the port). The assistant's response was never terminated at the HTTP layer (`Connection: keep-alive`, no `Content-Length`, no `Transfer-Encoding`), which is why the client reported a stall on every turn even after the answer arrived; the wrapper now speaks HTTP/1.1 with chunked framing and a terminating `0`-chunk. Plus the observability stack that found the causes and CORRECTIONS 50/51 for the two wrong attributions along the way | [engineering record](2026-09-16-spark13-observability-and-batch-flush.md) |
| 2026-09-15 | Claude Code Spark1.3 SSE fallback: the wrapper now normalizes duplicate/incomplete Anthropic events; end-to-end interactive validation remains pending | [engineering record](2026-09-15-claude-code-opencode-sse-fallback.md) |
| 2026-08-21 | Retracted the claim that the model loops in its reasoning; the probe now keeps the full trace and scores its repetition. Suite 108 -> 111 | [post-mortem](2026-08-21-inferred-looping-from-three-numbers.md) |
