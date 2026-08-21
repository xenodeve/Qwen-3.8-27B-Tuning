# 26 — The cold start was a second subagent, not the server

**Measured 2026-08-21 through Qwen Code itself.** Instrument:
`qwen38-tuning/scripts/bench-cold-start.py`. Raw:
`qwen38-tuning/results/cold-start.jsonl`.

## Result first

| | prefill | wall |
|---|---|---|
| before, first run on a fresh server | 54,025 tok, 53.4 s | 129 s |
| before, every run after | ~41,300 tok, 41.4 s | 56–68 s |
| **after, every run including the first** | **4 tok, 0.1 s** | **4.4–6.7 s** |

The one remaining 55-second prefill is paid by `scripts/warm-cache.ps1` while
nobody is waiting.

## What it was

`memory.enableManagedAutoMemory`. With it on, Qwen Code runs a **managed memory
extraction subagent** after every turn. Its system prompt is *different* from the
main agent's and nearly as large — 195,929 characters against 207,193 — so it
evicts the main prefix from llama-server's single slot, and the next invocation
re-prefills about 41,000 tokens.

Captured through a recording proxy, one invocation sends **five** large requests,
not one:

```text
  207,193 chars  "You are Qwen Code, a non-interactive CLI agent..."
  174,046
  207,717
  207,193
  195,929 chars  "You are now acting as the managed memory extraction subagent..."
```

With the setting off the harness sends **one** request per invocation, the slot
keeps the prefix, and the cache hits.

## What it was not

Every one of these was measured and is not the cause:

| suspect | result |
|---|---|
| the server's prompt cache | **works perfectly.** One captured request replayed three times: 53.9 s, then 0.4 s, then 0.4 s |
| a small interleaved call | harmless. `BIG → SMALL → BIG` returns 0.6 s |
| a volatile prefix | no. Two captured requests are identical for all 207,193 characters |
| `--cache-ram -1` | **regression.** Reuse drops to zero, measured twice |
| `--cache-reuse 256` | **regression.** Full 54,499 prefilled every run |
| `-np 2` at 110,592 | no change: still ~41,300 re-prefilled |
| `-np 2` at 131,072 | **VRAM collapse.** 113.9 tok/s at 296 MiB free, run timed out |
| a larger `-ub` | no. 1,134–1,168 tok/s across 256/512/1024 |
| the memory *files* | no. Moving `~/.qwen/memories/*` aside changed nothing |
| `reasoning_effort` | no. The template swaps one instruction sentence |

## The warm-up has to run in the working directory

The first attempt warmed from a background job's own directory and the developer's
first turn still paid 49.8 s. **Qwen Code's prompt embeds the working directory**,
so warming elsewhere warms a different prefix. `warm-cache.ps1` takes `-Work` and
defaults to the current location.

## The trade, stated

With `memory.enableManagedAutoMemory` off, **Qwen Code stops updating its own
memories.** That is a real feature, switched off for speed, and it is the
developer's call rather than this project's.
`memory.enableManagedAutoDream` and `memory.enableAutoSkill` were turned off in
the same measurement and **have not been isolated from each other** — the 41 s
could belong to any of the three, or to a combination.

## What this says about the earlier conclusions

Report 25 spent an afternoon on the server: reserve, micro-batch, slot count,
cache flags, and the context window itself. **None of it was the cold start.** The
one measurement that pointed the right way was the cheapest available — replaying
a captured request against the server with no harness in the path — and it should
have been the first, not the twelfth.
