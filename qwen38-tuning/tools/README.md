# `qwen38-tuning/tools/` — instruments that are not the bench

The bench (`../bench/`) measures **our** server. This folder holds tools that
observe something else, or observe the bench from outside.

| tool | what it is for | status |
|---|---|---|
| [`llama-tap/`](llama-tap/README.md) | records what a client actually sends `llama-server`, on the wire, without altering it. Built because three published claims about Unsloth Studio's configuration were read off a command line and retracted — `--reasoning-effort` never appears in an argv and decided how our server behaved for an afternoon | built and unit-tested 2026-08-29 (#53); **never yet run against Studio**, so nothing in `docs/` cites it |

## The rule that applies to everything here

**An instrument may not change what it measures.** `llama-tap` forwards bytes
and observes a copy; it never re-serialises a request. If a tool here cannot
observe something without touching it, it does not go in this folder — it goes
in a plan, with the reason.

Tests for these live with the rest, in `../bench/tests/`, because the gate is
one command:

```powershell
cd qwen38-tuning\bench ; python -m pytest tests\ -q
```
