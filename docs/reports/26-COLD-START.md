# 26 — The cold start was a second subagent, not the server

> 🔴 **The title's second half is retracted — [`CORRECTIONS.md` §16](CORRECTIONS.md).** The correction is in the body below, but it belongs
> here where the title is read.
>
> 🔴 **The two-slot capacity is also retracted — [`CORRECTIONS.md` §17](CORRECTIONS.md).** It was sized from a 54,499-token request; a real
> developer session measures **71,910** against 55,296 per slot. **The
> mechanism and the `-sps 0.95` cure survive; the capacity does not.**

**Measured 2026-08-21 through Qwen Code itself.** Instrument:
`qwen38-tuning/scripts/bench-cold-start.py`. Raw:
`qwen38-tuning/results/cold-start.jsonl`.

## Result

**Two cures, measured. Prefer the second — it costs nothing.**

| | prefill per invocation | wall | Qwen Code memory |
|---|---|---|---|
| before, `-np 1` | ~41,300 tok, 41.4 s | 58–71 s | on |
| turn off `memory.enableManagedAutoMemory` | 4 tok, 0.1 s | 4.4–6.7 s | **off — a feature lost** |
| **`-c 110592 -np 2 -sps 0.95`** | **0 tok, full cache hit** | **5.9–17.1 s** | **on** |

`scripts/warm-cache.ps1` pays the one cold prefill while nobody is waiting.
`scripts/worker-iq2s-2slot.ps1` is the profile.

**This section was rewritten after the developer refuted its first version.** It
had said the cold start was the harness and not the server, on the strength of
the setting cure. The refutation was one sentence: the same unmodified Qwen Code
is fine against a gateway FP8 endpoint, so the subagent evicts the cache there
too and nobody notices — a datacenter card re-prefills 41,000 tokens in about a
second. The cold start is the product of the eviction **and** our prefill rate.
`CORRECTIONS.md` §16.

## Why two slots, and why `-sps`

`--slot-prompt-similarity` defaults to **0.10**. Two prompts that share a tool
catalogue look alike to it and are routed to the **same** slot, so they evict
each other exactly as they did with one. An earlier `-np 2` measurement at this
same depth, with the default, changed nothing and was written off as a dead end.
Raising it to 0.95 forces them apart and the eviction stops.

The price is real: 110,592 of KV against 98,304, and 55,296 of window per
conversation rather than 98,304. That clears the 54,499-token request measured
for Qwen Code, but not by much.

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

## The prompt itself, measured — `--safe-mode` cuts 74 %

**Measured 2026-08-21, same directory, same `-p "hi"`, only the harness's own
customization layer differing.** Prompt size is read from the server's
`slot release ... n_tokens` line, which reports the whole slot whether the cache
hit or not:

| | prompt | calls | prefill at ~900 tok/s |
|---|---|---|---|
| baseline | 54,711 | 3 | ~60 s |
| `qwen --safe-mode` | **14,399** | 1 | **~16 s** |
| `skills.disabledLevels` all | 57,526 | 4 | — |
| skills off + managed memory off | 51,423 | 1 | — |

**The customization layer is 40,312 tokens — 74 % of the prompt**, and
`--safe-mode` also collapses three model calls per invocation into one.

**Two of the four rows are negative results worth keeping.** Disabling skills
made the prompt *larger*, not smaller, so the skill catalogue is not where the
tokens are. Turning managed memory off recovered only 6,103. Whatever holds the
remaining ~34,000 is something else `--safe-mode` disables — tool schemas are the
obvious candidate and are **not yet isolated**.

## The instrument was wrong twice more, and both are the same mistake

The first run of this waterfall reported a 3,147-token baseline, which would have
been a spectacular result and was nonsense: **it was measuring tokens prefilled,
not prompt size**, and the cache was warm from the previous arm. A cache hit
makes a large prompt look small.

That is the same shape as the two faults already on file — taking the first
prompt-eval line of three, and reading a full cache hit as a failure. All three
come from one habit: **treating a number llama-server prints about its own work
as though it described the request.** It does not. `n_tokens` on the release line
describes the request; `prompt eval` describes what was left to do.

## What the directory is not

A brief circulating for this problem attributed a 17,414-token gap to
project-specific context — `QWEN.md`, project memory, repo metadata — and
proposed diffing two directories. Measured across three:

```text
  C:/Users/xenod    54,095 tok
  C:/AI             54,483 tok
  C:/ocworker/run   54,073 tok
```

**A 410-token spread.** The working directory is not the variable. The 71,913
seen in a real session against 54,x in these runs is the interactive TUI against
`-p`, and that axis has not been measured.

## The 40,312 tokens, attributed exactly

**Measured 2026-08-21** by capturing every request of one `qwen -p "hi"`
invocation through a recording proxy, then rendering each through the server's
own `/apply-template` and counting with its own `/tokenize`. No byte estimates,
no subtraction between independent-looking knobs.

| run | call | total | without tools | tool schemas | n tools | system | messages |
|---|---|---|---|---|---|---|---|
| baseline | 1 | 633 | 360 | 273 | 1 | 284 | 2 |
| baseline | 2 | **54,485** | 49,069 | 5,416 | 8 | 8,766 | 2 |
| baseline | 3 | 42,226 | 41,964 | 262 | 1 | 1,576 | 5 |
| baseline | 4 | **56,277** | 50,861 | 5,416 | 8 | 8,766 | 5 |
| `--safe-mode` | 1 | **14,064** | 8,648 | 5,416 | 8 | 5,372 | 2 |

**Tool schemas are not the cause.** 5,416 tokens, *identical* in baseline and in
safe mode, from the same 8 tools — ToolSearch is doing its job. The system prompt
differs by only 3,394.

Block by block inside the messages:

| block | baseline | safe mode |
|---|---|---|
| **skill catalogue** | **38,064** | **1,037** |
| `tool_search` index | 1,897 | 1,897 |
| session/date reminders | 332 | 332 |
| the user's actual message | 1 | 1 |

**The skill catalogue is 70 % of the prompt, and it is injected as a *user*
message block** — which is why `skills.disabledLevels` did not remove it and made
the prompt larger instead.

```text
  <system-reminder>
  The following skills are available for use with the Skill tool. ...
  <available_skills><skill><name>agents-sdk</name><description>...
```

**352 skills advertised, 344 of them user-scope**, at roughly 110 tokens of name
and description each. Safe mode advertises 9 bundled ones.

## Inflation and amplification are two problems, and `--safe-mode` hides both

| | baseline | safe mode | ratio |
|---|---|---|---|
| largest single prompt | 56,277 | 14,064 | 4.0x |
| calls per invocation | 4 | 1 | 4x |
| **tokens the GPU must read** | **153,621** | **14,064** | **10.9x** |

At ~900 tok/s that is 171 s of prefill against 16 s, for the same one-word
prompt. The catalogue is paid **three times**, once per large call.

## First divergence

Baseline and safe-mode prompts share **70 tokens** before diverging. Whatever is
cached from one is worth almost nothing to the other, so the two modes cannot
warm each other — which is exactly why warming with `-p` did nothing for a real
session.

## What to do about it

Ranked by what the measurement supports:

1. **`qwen --safe-mode`** — available now, no configuration, 4x on the prompt and
   4x on the calls. Costs every customization.
2. **Prune what Qwen Code advertises.** 344 user-scope skills is the whole cost;
   `~/.qwen/skills` alone holds 69 and 5.2 MB. A skill that is never invoked by
   the model still charges its description on every call.
3. **`disable-model-invocation: true`** in a skill's frontmatter keeps it callable
   while removing it from the advertisement. **Not tested here.**

Nothing on this list touches the server, the quantization, or CUDA.

## `disable-model-invocation: true` works, and it is linear

**Measured 2026-08-21.** Eighteen skills in `~/.qwen/skills` had the line added
to their frontmatter, the catalogue was captured again through the proxy, and the
files were restored:

| | skills advertised | catalogue |
|---|---|---|
| baseline | 352 | 38,064 tok |
| 18 skills flagged | 334 | 36,496 tok |
| **per skill** | **−1** | **−87 tok** |

Exactly 18 fewer entries for 18 files. **The mechanism does what its name says
and the cost is linear**, so applying it to all 344 user-scope skills removes
roughly **30,000 tokens** — most of the 38,064 — while every skill file, every
MCP server, the memory features and the extensions all stay in place.

That is the difference between this and `--safe-mode`, which buys the same
reduction by turning everything off.

**Not tested: whether a flagged skill is still invocable.** The point is not to
delete skills from the prompt, it is to keep them reachable while they are idle,
and that half has no measurement behind it yet.

`~/.qwen/skills` holds **257** skills, not the 69 an earlier depth-limited count
reported.

## Still open: why the same catalogue costs nothing on a gateway

The same Qwen Code, the same 352 skills, against Qwen3.8-27B FP8 on a remote
gateway, answers quickly. **That is not explained by anything measured here.**
The catalogue is the root cause of the *work this machine performs*; it is not
the reason the two endpoints differ. Three candidates, none tested:

- the gateway holds a cross-request prefix cache over a 38,064-token block that
  never changes, which is the ideal case for one;
- its prefill throughput is an order of magnitude above 900 tok/s, so 153,621
  tokens cost seconds rather than minutes;
- the harness takes a different path against it and never sends four calls.

The experiment is a control group: same folder, same fresh session, same `hi`,
and capture the call count, the tokens per call, the cached-token count and the
time to first token from both. Until that is run, **no claim should be made about
why the gateway is faster.**

## The hidden skill is unreachable, not merely unadvertised

**Measured 2026-08-21.** `~/.qwen/skills/animation` was flagged with
`disable-model-invocation: true`, and the CLI was asked to invoke it **by exact
name** in both states:

| state | what the Skill tool returned |
|---|---|
| advertised | the call was made and **declined by the permission layer** in non-interactive mode |
| flagged | *"Skill 'animation' not found (it is not among the available skills)"* |

Two different failures. The second is the registry saying the skill does not
exist, not a permission denial.

**So the flag cannot carry a Dynamic Skill Injection design that hides a skill
and has a router ask the model to call it.** It removes the skill from the
model's reach entirely. Injecting the skill's *content* into the prompt when it
is needed remains possible; asking the model to invoke a hidden one does not.

## The gateway receives the same payload and is simply faster

**Control group, 2026-08-21.** Same folder, same fresh session, same `hi`, same
352-skill catalogue, captured through a recording proxy in front of
`gateway.9arm.co` — Qwen3.8-27B FP8 instead of the local IQ2_S:

| call | messages | tools | prompt tokens | TTFB | wall |
|---|---|---|---|---|---|
| 1 | 1 | 0 | 54 | 0.52 s | 0.52 s |
| 2 | 2 | 1 | not reported | 3.40 s | 3.40 s |
| 3 | 2 | 8 | **54,478** | 4.97 s | 8.94 s |
| 4 | 5 | 1 | not reported | 3.97 s | 3.97 s |
| 5 | 5 | 8 | **57,700** | 1.41 s | 3.16 s |

**Total wall time: 19.4 s**, against roughly 171 s of prefill locally.

**The harness is not doing anything different.** 54,478 and 57,700 against the
local 54,485 and 56,277 — the same calls, carrying the same catalogue. Nothing is
stripped or transformed on the way.

**Three hypotheses, resolved:**

- *A different call path* — **refuted.** Five calls either side, same shapes.
- *A reported cross-request prefix cache* — **not reported.**
  `prompt_tokens_details.cached_tokens` is absent from every response.
- *Prefill throughput* — **supported, and it is the whole difference.** 54,478
  tokens at 4.97 s to first byte implies roughly **11,000 tok/s** against our 900.
  Call 5 then takes 57,700 tokens to first byte in **1.41 s**, which is the
  signature of prefix reuse on top of that: the same prefix, a quarter of the
  latency.

**So the catalogue is the root cause of the work this machine performs, and the
gateway is not spared it — it absorbs the same 112,000+ input tokens an order of
magnitude faster.** Removing the catalogue is still the only lever available
here, because the other one is a different class of hardware.
