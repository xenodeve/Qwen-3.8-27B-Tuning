# Server-side proxies that could sit in front of `llama-server` — survey, 2026-09-03

**External material. Nothing here has been measured on this machine except the
rows tagged MEASURED HERE.** Every other claim is the vendor's README or a
paper. Searched the same day the first real session through a proxy cost one
full re-prefill (issue #55, comment of 2026-09-03).

## The one requirement, and where it came from

A proxy in front of `llama-server` may rewrite the request, **but it must
rewrite every turn of a session the same way**, because llama.cpp's prompt
cache matches the longest byte-identical prefix and nothing else.

MEASURED HERE (`logs/serve-20260902-160749.log` tasks 251187/251309,
`~/.headroom/logs/xeno-proxy.jsonl`): headroom 0.37.0 in `--no-optimize`
rewrote turn 1 (schema compaction + hoisting the mid-conversation system
message) and, once a signed thinking block existed, forwarded turn 2 as the
client's original bytes. Turn 2 started from `cached n_tokens = 0` and
prefilled 74,422 tokens in **89.8 s** to save 3,469 tokens on turn 1.

So the gate for every candidate below is the same and costs one boot:
**`cached n_tokens` on turn 2 must not be 0**, read from the serve log.

## Candidates

| candidate | form | Anthropic Messages | plain-`http://` upstream | prefix policy (vendor's words) | reduces | status here |
|---|---|---|---|---|---|---|
| **headroom** 0.37.0 — 68,610★, Apache-2.0, Python | proxy | yes | yes (`ANTHROPIC_TARGET_API_URL`) | `--mode cache` freezes prior turns; but for a non-Claude model id it hoists the system message and compacts schemas on turn 1 only, then locks to client bytes on signed thinking (`anthropic.py:3652`) | tool output, thinking, reads, schemas, system | **MEASURED HERE: −89.8 s per session start.** Needs a two-line local patch (skip `compact_tools` + `relocate_system_messages_to_top_level` for non-Claude ids) before any second trial |
| **tamp** — 91★, MIT, Node | proxy | auto-detects | `TAMP_UPSTREAM` is a URL; `http://` not documented | `TAMP_CACHE_SAFE`: *"compress newest only (prompt-cache safe)"* — earlier turns byte-identical; the two stages that rewrite older turns (`stale-inputs`, `stale-images`) are opt-in *because* they bust the cache | tool output: minify, columnar JSON, dedup, diff-replace (L1–4 lossless), LLMLingua sidecar from L5 | **MEASURED HERE the same day, against the 9arm gateway through a recording proxy** (`scratchpad/tamp/arm3`, three turns, lossless stages `cmd-strip,minify,toon,strip-lines,whitespace,dedup`): turn 2 went out with the new tool_result compressed by `toon` (2.5K → 562 chars, −78 %); **turn 3 carried that same block as the client re-sent it — the 5,327-char original**, because "cache-safe" means *newest block only, earlier blocks exactly as the client sent them*. For Anthropic that keeps their breakpoints intact; for llama-server it means the compressed form lives for one request, the history fills with originals, and the prefix diverges at that block on the next turn. **Prefix-safe only trivially, and no slower context growth at all.** Also: its startup probe took our headroom proxy on 8788 for an LLMLingua sidecar (`✓ LLMLingua-2 ready on :8788`) |
| **llmtrim** — 224★, MPL-2.0, Rust | HTTPS MITM via `HTTPS_PROXY` | yes | **no** — intercepts TLS hosts only; `LLMTRIM_EXTRA_HOSTS` adds hostnames, still TLS | *"nothing under a `cache_control` marker is rewritten"*; sorts tools/schemas to stabilise the prefix; `llmtrim recall r_…` restores a trimmed result | tool output, history, code skeletons, schemas; −31 % input on 112 A/B cases (vendor) | untried; would need TLS in front of llama-server or a reverse-proxy mode it does not advertise |
| **claude-code-cache-fix** — 426★, Node | proxy | yes | yes (`CACHE_FIX_PROXY_UPSTREAM`) | *"idempotent: if nothing needs fixing, the request passes through unmodified"* — sorts tools, relocates attachments to `messages[0]`, strips the `cc_version` fingerprint | **nothing** — a stabiliser, not a compressor | untried; on our capture (`claude-cli/2.1.258`, `sdk-cli`) the billing header is `cc_version=2.1.258.1e2; cc_entrypoint=sdk-cli;` with no per-request hash, and the live sessions hit the cache every turn, so the problem it fixes is not visible here today |
| **claude-litellm-llamacpp** — 0★, Python | LiteLLM callbacks | via LiteLLM | yes | truncate-middle and summarise **rewrite the history** | history | reference only: cache-hostile by design, and it leans on `--cache-reuse`, which this project ruled out for the hybrid model (ledger row on DeltaNet state; `llama-memory-recurrent.cpp:150-233`) |
| **LiteLLM prompt compression** | proxy callback | yes (`anthropic_messages`) | yes | not addressed in its docs | replaces low-relevance history with stubs + a `litellm_content_retrieve` tool | reference only until it says what stays byte-stable |
| **claude.cpp** | LiteLLM translation + server flags | via LiteLLM | yes | none | nothing on context | no |
| **TokenPilot** (EMNLP 2026 Findings, arXiv 2606.17016) | paper; code was `zjunlp/RSI`, 404 today | — | — | *"Ingestion-Aware Compaction … stabilises prompt prefixes"* + *"Lifecycle-Aware Eviction … a conservative batch-turn schedule"* | the design behind stories 38–40 | the theory to copy, not a tool to install |

Not proxies, but found on the way:

- **`CLAUDE_CODE_ATTRIBUTION_HEADER=0`** removes the billing header from the
  start of the system prompt; two write-ups blame it for llama.cpp cache misses
  and one says it works only from `settings.json`, not the shell. MEASURED
  HERE: not our failure today (header is version-stable, turns hit the cache),
  but it is the first thing to set if a future Claude Code version adds a
  per-request hash.
- **`ENABLE_TOOL_SEARCH=true`** is documented as *"set if your proxy forwards
  `tool_reference` blocks"* — whether llama-server accepts that shape is the
  free experiment #55 already lists.

## What the two trials say the proxy must be

Both off-the-shelf proxies failed the same way from opposite sides. headroom
rewrote turn 1 and then stopped (the signed-thinking lock); tamp rewrites only
the newest block and lets the client's originals refill the history. **Neither
re-derives the same bytes for the same block on every turn**, and that is the
whole requirement in front of llama-server: a compressor that is a pure
function of the block — `f(tool_result) -> bytes`, identical on turn 2, 3 and
30 — so the cached prompt and the next prompt agree, and the history holds the
compressed form instead of the original. That is the request-level reducer
#55 already lists as "ours regardless", now with the reason it cannot be bought.

## Order to try, and the cost of each

```text
GATE  every arm: turn-2 `cached n_tokens` != 0 in the serve log, tap behind the proxy
~~1   tamp~~  measured: newest-only compression, originals re-enter history next turn — does not serve either goal
1     headroom after the two-line patch                      (already installed and wired)
2     claude-code-cache-fix chained in front of 1             (only if a future CC version breaks the prefix)
—     llmtrim                                                 (needs TLS on the local path; park)
—     LiteLLM / claude-litellm-llamacpp                       (rewrite history; do not)
```

**Every candidate is judged on time per task and tokens per tool call on real
work (#55 goals), never on its own savings counter** — headroom's counter said
−5.1 % on the session that cost 90 s.

Sources: [headroom source in site-packages, read 2026-09-03] ·
[tamp](https://github.com/sliday/tamp) · [llmtrim](https://github.com/fkiene/llmtrim) ·
[claude-code-cache-fix](https://github.com/cnighswonger/claude-code-cache-fix) ·
[claude-litellm-llamacpp](https://github.com/TyrelCB/claude-litellm-llamacpp) ·
[claude.cpp](https://github.com/d4rks1d33/claude.cpp) ·
[LiteLLM prompt compression](https://docs.litellm.ai/docs/completion/prompt_compression) ·
[TokenPilot](https://arxiv.org/abs/2606.17016) ·
[the attribution-header write-up](https://www.mykolaaleksandrov.dev/posts/2026/06/claude-code-llamacpp-prompt-cache-fix/) ·
[claude-code-router #1217](https://github.com/musistudio/claude-code-router/issues/1217) ·
[Anthropic Messages API in llama.cpp](https://huggingface.co/blog/ggml-org/anthropic-messages-api-in-llamacpp)
