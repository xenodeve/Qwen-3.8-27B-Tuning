# `qwen38-tuning/tools/` — instruments that are not the bench

The bench (`../bench/`) measures **our** server. This folder holds tools that
observe something else, or observe the bench from outside.

| tool | what it is for | status |
|---|---|---|
| [`llama-tap/`](llama-tap/README.md) | records what a client actually sends `llama-server`, on the wire, without altering it. Built because three published claims about Unsloth Studio's configuration were read off a command line and retracted — `--reasoning-effort` never appears in an argv and decided how our server behaved for an afternoon | built and unit-tested 2026-08-29 (#53); **never yet run against Studio**, so nothing in `docs/` cites it |

| [`llama-buffer/`](llama-buffer/README.md) | reduces what a client sends BEFORE it reaches the server -- chunk a file, retrieve the parts that answer this turn, and state that the rest was cut. Built because the largest lever measured on this machine is not a flag: tools off took a `สวัสดี` turn from 21.5 s to 1.5 s | retrieval core built and tested 2026-08-30 (#54); **the proxy is not written**, so nothing sits in front of anything today |

**`llama-tap` and `llama-buffer` have opposite contracts and that is deliberate.**
One forwards bytes and re-serialises nothing, so it can audit a client we do not
control. The other rewrites the request, which is its whole purpose. Two names,
so a reader cannot trust the wrong one.

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
