# opencode — the worker profile

**The lean OpenCode configuration that drives the local server.** It exists
because OpenCode's default profile sends a **99,073-token prefix** — measured by
the gateway's own tokenizer for the prompt *"say READY"* — and that plus its
32,000-token output reservation is 131,073, one token over a 131,072 window. The
first call failed before any work started.

This profile sends **5,377**, measured by our own server's tokenizer. Same tool
loop, same ability to write files and run commands; verified by having it solve
a corpus task whose code then passed the hidden tests.

---

## Running a worker

Two things, in this order. They are separate because the server holds the GPU
for minutes and OpenCode does not.

**1. Start one profile.** They share port 8080 and the alias `qwen38`, so this
config works with either and swapping means restarting only the server:

```powershell
..\scripts\worker-iq2xxs-deep.ps1      # 131,072 ctx, ~79 % top-1
..\scripts\worker-iq2s-quality.ps1     #  98,304 ctx, ~84 % top-1
```

**2. Run OpenCode against it**, from a directory that is **not** under
`C:\Users\<you>`:

```powershell
$env:OPENCODE_DISABLE_CLAUDE_CODE     = "1"
$env:OPENCODE_DISABLE_EXTERNAL_SKILLS = "1"
$env:OPENCODE_DISABLE_DEFAULT_PLUGINS = "1"
$env:OPENCODE_CONFIG_DIR              = "C:\AI\qwen38-tuning\opencode"
opencode run -m local/qwen38 "<the task>"
```

If you change `-c` on the server, change `limit.context` in `opencode.json` to
match. OpenCode uses it to decide when to compact, and a value larger than the
server's window means it compacts too late.

---

## Why each of those settings, and what it cost to find

None of the environment variables is documented anywhere we could find. They
were found by searching strings in the binary for its own flag table.

| setting | frees | why |
|---|---|---|
| `OPENCODE_DISABLE_CLAUDE_CODE` | most of it | the broad switch: drops the Claude-Code-derived system prompt **and** the skill catalogue |
| `OPENCODE_DISABLE_EXTERNAL_SKILLS` | — | belt and braces with the above |
| `OPENCODE_DISABLE_DEFAULT_PLUGINS` | — | same |
| `OPENCODE_CONFIG_DIR` | ~10,000 tokens | **the one that finished the job.** Without it, OpenCode walks up from the working directory collecting every `.opencode` it finds, picks up `~\.opencode\skill` and injects 67 more skills |
| `mcp: { …: {enabled: false} }` | **~62,000 tokens** | 141 MCP tools and 265 KB of schema. Setting `"mcp": {}` is **not** enough — it adds nothing rather than disabling what is already there, so every server must be named with `enabled: false` |
| `tools: {webfetch, task, todowrite, skill: false}` | ~2,000 tokens | leaves `bash edit glob grep read write` — what a coding worker needs. `skill` is off because the injection layer handles that |

Measured at each step:

```text
as configured        141 tools   387 skills   99,073 tokens
MCP disabled          10 tools   387 skills  ~46,500
+ the skill switches   6 tools     0 skills   ~5,377
```

**Run from outside your home directory.** OpenCode resolves the project root by
walking up from the working directory, and it keeps a server alive between
invocations that carries the root it first started with. A second task launched
elsewhere writes its files into the first task's project — which cost this
project three corpus tasks graded as "no file written" on work the model had
done correctly.

## What this does not fix

**OpenCode sends no sampling parameters at all**, so the server's defaults
apply: `temperature 1.0, top_k 20, top_p 0.95, min_p 0.05`. Unsloth's published
preset for thinking mode is the same except **`min_p 0.0`**. That difference is
live and unmeasured.
