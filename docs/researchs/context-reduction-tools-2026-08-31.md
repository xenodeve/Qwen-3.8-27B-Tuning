# Seven tools that already do parts of `llama-buffer` — survey, 2026-08-31

**External material. Nothing here has been measured on this machine.** Every
claim below is the vendor's or the surveyor's; the only lines that are ours are
the three marked **CORRECTION** and the ladder at the end.

The source is a survey handed to the project on 2026-08-31 of what exists
already for the problem [issue #55](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/55)
proposes to build. **Checked the same day: none of these names appears anywhere
in this repository** — `super-mcp`, `context-compress`, `rag-rat`, `minirag`,
`claude-code-proxy` each return zero files across `.md`, `.py`, `.ps1`, `.json`.
**Serena is the exception and it is the interesting one: it is already
installed.**

| part of #55 | candidate | fit | install cost |
|---|---|---:|---|
| tool schema filtering | **Super-MCP Router** | ⭐⭐⭐⭐⭐ | one `npx`, then move MCP config |
| long tool output | **context-compress** | ⭐⭐⭐⭐⭐ | one `npm i -g` + `setup --auto` |
| code read / symbols | **Serena** | ⭐⭐⭐⭐½ | **already installed here** |
| provenance, staleness, git | **rag-rat** | ⭐⭐⭐⭐⭐ | plugin marketplace or `npm i -g` |
| documents and Thai | **minirag-mcp** | ⭐⭐⭐⭐½ | `uvx`, no VRAM |
| API / history compression | **claude-code-proxy** | ⭐⭐⭐ | a proxy process |
| whole-stack RAG reference | **Open WebUI** | — | a second client, not a fix |
| transparent arbitrary `Read` rewrite | **nothing** | ⭐⭐ | ours to write |

---

## 1. Super-MCP Router — tool schema filtering

The shape #55 asks for in stories 2–5:

```text
500 tools
   ↓  name + one-line description only
search for the relevant tool
   ↓
load the full schema only for the one about to be used
```

**Super-MCP already is that pattern.** It puts many MCP servers behind a single
MCP server and exposes only meta-tools:

```text
Claude Code
    │  sees a handful of meta-tools
    ▼
Super-MCP
    ├─ search_tools        BM25 over names and descriptions
    ├─ list_tools
    ├─ get_tool_details    full schema, on demand
    └─ use_tool
          ├─ GitHub MCP
          ├─ Supabase MCP
          ├─ Filesystem MCP
          └─ 100+ tools
```

```bash
npx -y super-mcp-router@latest
# or
npm install -g super-mcp-router
```

Then every existing MCP server moves into Super-MCP's config instead of being
connected by Claude Code directly.

### The native alternative, free to test first

Claude Code is reported to ship **native tool search in 2026** and to defer MCP
schemas by itself — but to **disable it by default when `ANTHROPIC_BASE_URL`
points at a non-first-party endpoint**, because many proxies do not support
`tool_reference` blocks. Our setup is exactly that:

```text
Claude Code  →  local llama.cpp
```

```text
ENABLE_TOOL_SEARCH=true
```

costs one environment variable. **llama.cpp does not advertise full
Anthropic `/v1/messages` spec compatibility and its docs mention `tools` and
`tool_choice` without claiming `tool_reference`**, so the likely outcome is a
loud failure — which is still cheaper than installing anything. Super-MCP is the
preferable route precisely because **llama.cpp never has to understand
`deferred_tool_reference`**: discovery happens entirely on the Claude Code side
of the wire.

### Caveat, and it is the one that decides the whole plan

**It can only reduce MCP tools.** It cannot hide Claude Code's built-ins:

```text
Read   Edit   Write   Bash   Glob   Grep   Task   WebFetch   …
```

> **CORRECTION — the number this is aimed at is not the one in #55's Problem
> Statement.** The **17,509** figure was measured with *"a toggle in the chat
> UI"* — Unsloth Studio. **Studio's tools are its own, not MCP servers, so no
> MCP router can act on that number at all.** Claude Code's comparable figure is
> **17,881**, measured separately
> ([guide Tier 0](../reports/39-OPTIMISATION-GUIDE.md)), and **the composition
> of neither has ever been measured.** If 17,881 is mostly built-in, this rung
> moves almost nothing.

---

## 2. `context-compress` — long tool output

Stories 18 and 19 want:

```text
12,000-line log
   ↓  full text kept outside the context
   ↓  only a summary / the relevant part is sent
   ↓  the whole thing stays fetchable and searchable
```

```bash
npm install -g context-compress
context-compress setup --auto
context-compress doctor
```

It installs three things at once — a **Claude Code `PreToolUse` hook**, an **MCP
server**, and a **CLI** — then intercepts long command output, indexes it in
**SQLite FTS5 / BM25**, and passes back only the compressed result:

```text
npm test
   │ 50 KB
   ▼
context-compress
   ├─ errors
   ├─ relevant output
   └─ full output, indexed → searchable later
```

**Vendor claim: up to 93 % reduction in aggressive mode.** That is a **VENDOR**
number on somebody else's workload. This project's own guide tags every line
`MEASURED HERE` / `VENDOR` / `UNMEASURED` for exactly this reason, and an
untagged number is the one to distrust.

---

## 3. Serena — symbols instead of whole files

Instead of:

```text
Read("service.ts")  →  4,000 lines  →  context
```

```text
find_symbol("UserService.update")        →  just that symbol
find_referencing_symbols(...)            →  just the references
```

It uses the Language Server Protocol for symbol-level retrieval and editing
across **30+ languages**, and the stated purpose is avoiding whole-file reads.

```text
/plugin install Serena
```

```bash
claude mcp add serena -- \
  uvx --from git+https://github.com/oraios/serena \
  serena start-mcp-server \
  --context ide-assistant \
  --project .
```

### It does not replace the `Read` seam

It does **not** do:

```text
Claude calls the built-in Read()
      ↓
hook intercepts
      ↓
offset/limit rewritten automatically
```

So the custom `PreToolUse(Read)` in #55 keeps its reason to exist. **Claude
Code's official hooks support matching `Read` and modifying tool input before
execution**, which is the mechanism that story 9 needs.

> **CORRECTION — Serena is already installed here, and has never been
> measured.** It is live in the agent's tool list right now — `find_symbol`,
> `find_referencing_symbols`, `get_symbols_overview` and the rest — and
> **nothing in this repository records it, benchmarks it, or states what it
> costs per turn.** An always-present tool with no number is the shape this
> project keeps writing corrections about. **So it is part of the baseline, not
> an arm: the first arm of any code-retrieval comparison must be Serena OFF**,
> or the baseline already contains the thing under test.

---

## 4. `rag-rat` — provenance, working tree, staleness

Stories 14–17 and most of the "future" list want:

```text
repo · branch · commit · working-tree hash · path · line range · content hash

working tree  >  branch  >  HEAD  >  history

edited file  →  its chunks invalidated
```

`rag-rat` is a local repo-intelligence MCP carrying:

- source provenance
- a **tree-sitter call graph**
- git history
- GitHub issue / PR rationale
- repo memories
- confidence and coverage scores
- a background file watcher
- a **dirty / untracked working-tree overlay**, separate from the committed index
- clean files indexed by `commit_sha`
- **stale-result validation and healing before returning a result**

It states explicitly that **uncommitted edits are reflected in queries**.

```text
claude plugin marketplace add cq27-dev/rag-rat
claude plugin install rag-rat@rag-rat
```

then, in the repo:

```text
Set up rag-rat in this repo.
```

or by CLI:

```bash
npm install -g @rag-rat/bin
rag-rat init
claude mcp add --scope project rag-rat -- rag-rat mcp
```

What it would cover from #55's future list:

```text
✓ provenance          ✓ symbol graph
✓ incremental index   ✓ callers / callees
✓ source freshness    ✓ git history
✓ working-tree priority   ✓ repo knowledge / memory
```

**Do not install Serena and `rag-rat` together at the start** — they overlap
heavily, and two reducers moved together measure their sum.

---

## 5. `minirag-mcp` — documents and Thai

For the half of the problem that is:

```text
117 KB plan  ≈ 27,000 tokens
        ↓
retrieve the ~2K that answer the question
```

Local-first hybrid retrieval:

```text
Hybrid Search
├─ vector similarity
└─ BM25 keyword
        ↓
Reciprocal Rank Fusion
```

Default embeddings cover **50+ languages** and run locally on **FastEmbed /
ONNX**, so **no VRAM is taken from Qwen** — which matters on a machine with a few
hundred MiB spare at the served depth. Formats: Markdown, TXT, PDF, DOCX, PPTX,
XLSX, HTML, CSV, EPUB, notebooks.

```bash
claude mcp add minirag \
  --scope user \
  --env BASE_DIR=C:\path\to\docs \
  -- uvx minirag-mcp
```

```bash
minirag-mcp sync --base-dir C:\path\to\docs
```

**The A/B this makes possible is the one that decides whether we write a
retriever at all:**

```text
A = #55's own       lexical char n-gram + whole words   (retrieve.py, 18 tests)
B = minirag         multilingual vector + BM25
```

---

## 6. `empero-org/claude-code-proxy` — history compression

```text
Claude Code
    ↓ Anthropic Messages
claude-code-proxy
    ↓ OpenAI-compatible
llama-server
```

```text
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
```

It supports current Anthropic messages and streaming tool use, has prompt
compression modes:

```text
none  |  compact  |  summarize
```

and **strips `thinking` / `redacted_thinking` from history** — which is story 20.

**But its stated limits rule it out as a core:**

```text
tool schemas     → untouched
user messages    → untouched
project context  → kept
```

so it addresses system-prompt boilerplate and some old history, and **not** the
17.5K of schemas, the pasted plan, or a smarter `Read`.

---

## 7. Open WebUI — a good reference, a different road

It connects to `llama-server` directly and brings Knowledge/RAG, hybrid search,
reranking and vector-DB integrations in the box:

```text
llama-server :8081
       ↓
Open WebUI
```

It is useful for answering *"how does off-the-shelf RAG compare to ours on
quality and latency?"* — but **it does not fix Claude Code traffic**, because it
is another client, not something Claude Code passes through:

```text
Open WebUI → llama.cpp          what it is
Claude Code → Open WebUI → llama.cpp    what would be needed
```

**Reference implementation, not a solution for #55.**

---

## What the experiment would look like, and the order

```text
                         Claude Code
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
          Super-MCP     context-compress    Read hook
              │               │                │
        tool schemas    Bash / log output      │
              │               │         Serena / rag-rat
      ┌───────┼───────┐       │
   GitHub  Supabase  RAG      │
                    │         │
                minirag       │
                              ▼
                       Anthropic API
                              │
                              ▼
                         llama-server
                        Qwen3.8-27B
```

**Do not turn them on together.** A verdict does not transfer here, and two
reducers moved at once measure their sum.

```text
GATE   split 17,881 into MCP vs built-in, with llama-tap
       no installation, one boot; decides whether rung 1 is worth anything
  |
  v
1      Super-MCP only               prompt tokens + correctness
2      context-compress only        tool-output workload
3      Serena OFF → ON → rag-rat    source-code tasks
4      minirag vs retrieve.py       Thai and long documents
5      what is left that we must still write
```

> **CORRECTION — the gate's instrument exists and has never been used.**
> `qwen38-tuning/tools/llama-tap/` was built and unit-tested under
> [issue #53](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/53), with
> a rule that it **forwards bytes and never re-serialises a request** — so a
> number taken through it is not a measurement of our own parser. Its README
> says *"Never yet run against Unsloth Studio. No capture from Studio exists
> yet."* Checked 2026-08-31: the only captures on disk are `0001` and `0002`,
> **84 bytes each** — health checks.

---

## What stays ours whatever the survey turns out to be

```text
off the shelf                      still ours
├─ tool discovery   Super-MCP      ├─ the exact policy
├─ tool output      context-compress  ├─ transparent Read interception
├─ code            Serena / rag-rat   ├─ request-level long-message reducer
├─ documents       minirag            ├─ quality gates
└─ provenance      rag-rat            ├─ token accounting
                                      └─ integration and benchmark
```

**The scope of #55 may shrink from "build a context buffer" to "glue, policy and
measurement" — but only if the survey survives contact.** And **every acceptance
criterion in #55 applies to a purchased component exactly as it does to a
written one.** A tool that reduces tokens and is never measured for quality is
the same defect the PRD exists to prevent, arriving by installation instead of
by code.
