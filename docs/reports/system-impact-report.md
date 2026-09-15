# System impact register

One line per system-affecting change, newest on top. **Pointers, not copies** —
the full record lives in the file named, and duplicating it here is how two
versions drift apart.

| date | change | record |
|---|---|---|
| 2026-09-16 | Claude Code Spark1.3 observability: one structured timeline across client/wrapper/LiteLLM, incidents with snapshots, an immediate stall trigger on both sides, and the finding that LiteLLM flushes Anthropic SSE in bursts (the remaining cause of `Waiting for API response`) | [engineering record](2026-09-16-spark13-observability-and-batch-flush.md) |
| 2026-09-15 | Claude Code Spark1.3 SSE fallback: the wrapper now normalizes duplicate/incomplete Anthropic events; end-to-end interactive validation remains pending | [engineering record](2026-09-15-claude-code-opencode-sse-fallback.md) |
| 2026-08-21 | Retracted the claim that the model loops in its reasoning; the probe now keeps the full trace and scores its repetition. Suite 108 -> 111 | [post-mortem](2026-08-21-inferred-looping-from-three-numbers.md) |
