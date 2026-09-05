# audit — what Claude Code actually sends, read from real traffic

**Status: run once, 2026-09-03, against the boot in
`logs/serve-20260902-160749.log`.** The numbers it produced are in
[issue #55](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/55)
(comment of 2026-09-03) and the ledger row for `llama-buffer`.

Issue #55's GATE asked for the split of a session's first request into MCP
schemas against Claude Code's built-ins. `llama-tap` was the planned
instrument and has never captured a real request. These two scripts answer the
same question from what already exists on disk.

| | reads | gives |
|---|---|---|
| `preamble_audit.py` | the serve log + `~/.claude/projects/**/*.jsonl` | every cold prefill matched to the session or subagent that caused it; the prefill/decode split of the boot; per-session first-request size and tool-call mix; the token cost of every attachment Claude Code injected before the first reply (hooks, skill listing), tokenized by the served model |
| `list_mcp_tools.py` | `~/.claude.json` + the project's `.mcp.json` | every reachable MCP server's tools, rendered the way Claude Code hands them to the model and tokenized by `/tokenize`; output `mcp_tools-<date>.json` |

```powershell
python preamble_audit.py --log ..\..\..\logs\serve-20260902-160749.log `
    --boot "2026-09-02 16:07:55" --project D--Github-Agentic-Framework
python list_mcp_tools.py        # needs the `mcp` package and the server up
```

## What it cannot tell you

- **A transcript holds neither the system prompt nor the tool schemas.** The
  fixed preamble is inferred by subtraction: first request minus what the
  transcript does hold. The check on that inference is that the remainder
  matches subagent cold starts, which carry no hooks, skills or `CLAUDE.md`.
- **Servers that need Docker or OAuth cannot be listed from a script.** The
  MCP total is a floor and says so in the JSON (`"ok": false`).
- **It is not the wire.** The byte-level split is still `llama-tap`'s job.
- `--boot` is local time; the log's own stamps are minutes since start.

## Privacy

The transcripts are the developer's own conversations. The script prints
token counts, tool names and the first characters of a message head; it writes
nothing from them to the repository.
