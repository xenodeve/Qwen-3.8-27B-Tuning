# llama-buffer — a retrieval layer between the client and llama-server

**Status: the retrieval core is built and tested (issue #54). The proxy is not
written yet.** Nothing sits between any client and the server today.

---

## Why

Measured 2026-08-30, one boot, minutes apart, nothing changed but a toggle in
the chat UI (`../../logs/serve-20260830-010653.log`, tasks 2931 and 2994). The
message was `สวัสดี` both times:

| | tools ON | tools OFF |
|---|---|---|
| prompt | **17,843 tokens** | **334** |
| prefill | 18,618 ms | **554 ms** |
| whole answer | **21.5 s** | **1.5 s** |

**14× on the wall clock.** Every server flag this project has swept is worth
single digits; see [the guide's Tier 0](../../../docs/reports/39-OPTIMISATION-GUIDE.md).
Claude Code sends 17,881 tokens of tool schemas, skills and system prompt before
the first character of a greeting, and pastes whole files — a 117 KB plan is
~27,000 tokens.

## It is not `llama-tap`, and the difference is the whole point

| | [`../llama-tap/`](../llama-tap/README.md) | `llama-buffer` |
|---|---|---|
| contract | forwards bytes, **re-serialises nothing** | **rewrites the request** |
| for | auditing a client we do not control | reducing what our own client sends |

Two tools, two names, on purpose. A reader who confuses them will trust the
wrong one.

## What works today

`retrieve.py` — chunk a document, score its parts against a query, and render an
extract that says it is one. No model, no VRAM, no GGUF conversion.

On the developer's own 117 KB plan:

```
file  109,009 chars  ~27,252 tokens  ->  105 chunks

  "นี่คือแผนอะไร"                 -> parts 1,2,3,4,5       ~1,324 tokens   4.9 %
  "cookie migration goal"        -> parts 1,2,6,17,105    ~1,334 tokens   4.9 %
  "middleware อ่าน session ยังไง"  -> parts 17,20,21,22,24  ~1,546 tokens   5.7 %
```

The third row is the one that matters: a **Thai** question moved retrieval from
the title section to the middleware code, and the top hit is
`supabase.auth.getUser()`. Unsloth Studio's own retrieval sends 1,942 new tokens
for the same first question, so this is in the same range.

### Why character n-grams

The queries are Thai and **Thai is written without spaces.** A whitespace
tokeniser sees one enormous token, matches nothing, and returns whichever chunk
sorted first — it does not fail, it returns the **wrong** extract, and the model
answers confidently from it. `test_it_beats_first_chunk_wins` exists to catch
exactly that degradation.

Studio runs `ragMode: "hybrid"` for what is probably the same reason: its vector
half is `bge-small-en-v1.5`, an English model, so on a Thai query the keyword
half is what carries it.

### Why the marker is not optional

`render()` refuses to produce an empty extract, and always states what the text
is:

```
[extract of plan.md -- 5 of 105 parts, 5296 of 109009 characters.
 The rest of this file was NOT sent. If the answer is not in these parts, say
 so and ask for the specific section you need rather than guessing.]
```

A model handed five chunks with nothing saying so answers as though it read the
document — the exact confusion this tool exists to prevent, arriving by a
different door. **Stated truncation is a fact the model can act on. Silent
truncation is a believable wrong answer**, which is what this repository's north
star is about.

## What is not built

The proxy, and the three reducers behind it. Each is off until measured:

| | what it does | risk |
|---|---|---|
| `history` | drop reasoning blocks from old assistant turns | low |
| `long-message` | a message over N tokens becomes top-k chunks plus a marker | medium — the marker is what makes it honest |
| `tools` | pass only the tools that score for this turn, plus an always-keep list | **high — a missing tool is a wrong answer, not a slow one** |

`tools` holds 17,509 of those 17,843 tokens, so a buffer that will not touch it
misses the prize — and one that drops a tool the model then needs turns a slow
answer into a wrong one. It stays off by default until it is measured against a
task set.

## What has not been measured

**Quality.** Retrieval trades quality for speed and this project has never
measured the first half of that trade on any artifact. The same question
answered from the whole file and from an extract has to be compared before any
of this becomes a default.

## Files

| | |
|---|---|
| `retrieve.py` | chunking, scoring, and the extract marker |

Tests: `qwen38-tuning/bench/tests/test_llama_buffer_retrieve.py` (18).
