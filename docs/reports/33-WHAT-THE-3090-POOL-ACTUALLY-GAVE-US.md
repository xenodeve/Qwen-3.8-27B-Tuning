# 33 — What the RTX 3090 pool actually gave us, 2026-08-23

**Eight techniques measured or closed in one session. Five wins, three
retractions, and no profile changed.** The register of *what* is
[`08-rtx3090-transfer.md`](../results/08-rtx3090-transfer.md); this is the *why*,
and it argues one thing above all:

> **The largest win was already switched on, and three of the eight results
> removed a claim instead of adding a setting.**

Raw data: `results/prefix-cache-depth.jsonl`, `results/decoders-98304.jsonl`,
`results/ubatch-98304.jsonl`, `results/prompt-cache-swap.jsonl`. Every figure
below names the file it came from.

---

## 1. The session started at the last open item and fell through the floor

`08-rtx3090-transfer.md` had one row left marked OPEN — #15, recurrent-state
prefix reuse, which the register itself called *"the single largest untested
idea left in the pool"*. Their `PREFIX_CACHE=1` took turn 2 of a 24K chat from
~23 s to 1.15 s.

Answering it required measuring at the window we actually serve. That is what
broke everything else, because **the only prior measurements at ctx 98,304 all
had a drafter loaded**, and nobody had noticed.

## 2. Prefix reuse transfers, and the failure case is the expensive half

`results/prefix-cache-depth.jsonl`, one boot at ctx 98,304, `--spec-type
ngram-mod`.

| | 8,147 tokens | 44,255 tokens |
|---|---:|---:|
| turn 1, cold | 6,727 ms | 35,301 ms |
| turns 2–4, append-only | 218–257 ms | 228–265 ms |
| cache reuse | 99.5–99.7 % | **99.9 %** |
| one sentence edited near the front | 0.0 %, 6,627 ms | **0.0 %, 41,810 ms** |

**A warm turn costs the same ~250 ms whether the conversation is 8K or 44K.**
Prefill is a per-*conversation* cost, not a per-turn one — as long as the prompt
only grows.

It works despite `n_rs_seq = 0`. Qwen3.8 loads as arch `qwen35`, which is on the
`llm_arch_supports_rs_rollback` whitelist (`src/llama-arch.cpp:1044`), so the
recurrent half *can* be partially removed — but only by `n_rs_seq` tokens
(`src/llama-memory-recurrent.cpp:180-192`), and that is `draft.n_max`, **zero
without a model-based drafter** (`common/common.h:386`). Every `worker-*.ps1`
runs `ngram-mod` alone. What covers the gap is `--ctx-checkpoints`, default 32
(`common/common.h:613`), enabled by the server for exactly that case
(`tools/server/server-context.cpp:3372-3376`).

**The edit case does not degrade — it zeroes.** 0.0 % at both depths, and at 44K
that is 41.8 s. Anything injected ahead of the conversation must be byte-stable
across turns or the entire prefill is repaid every turn.

## 3. The window we serve was never the problem

`04-context-depth.md` recorded decode at ctx 98,304 as **2.8–5.0 tok/s with 13
of 16 rows timing out**, and called it a property of the window. It is a
property of the drafter. Every one of those sixteen rows ran
`--spec-type draft-dflash,ngram-mod` — readable in each row's own `args` field.

`results/decoders-98304.jsonl`, 24 rows, six paired rounds, arms alternated:

| arm | ok | timed out | median tok/s | free MiB after load |
|---|---:|---:|---:|---|
| `none` | 6/6 | 0 | 33.69 | 800–1,935 |
| **`ngram-mod`** | **6/6** | **0** | **96.92** | 769–2,117 |
| `dflash2` | 5/6 | 1 | 49.31 | **45–376** |
| `dflash2+ngram` | 4/6 | 2 | 5.66 | **153–240** |

**96.92 at 98,304 is faster than the 75.2 median recorded at 16,384.** The two
groups do not overlap on free VRAM, and that is the finding: arms without the
drafter finish 12 times out of 12 and spread 3–4 %; arms with it spread **146×**
on identical flags.

The mechanism, as far as it goes: with a drafter `n_rs_seq` is 4, so the server
writes `created speculative checkpoint … size = 149.626 MiB` — one full
recurrent-state plane — every few generated tokens. In slow rounds the gap
between checkpoints reaches **30.41 s** against a median 2.35 s in fast ones. A
stall, not uniform slowness. Retracted as
[`CORRECTIONS.md` §26](CORRECTIONS.md).

**Why it hid for four days.** The sweep was named `ngram-nmatch` and every row's
`args` said `draft-dflash,ngram-mod`. Nobody read `args`.

## 4. `-ub` — the mechanism was right and the price was wrong

The scan calls `-ub` *"the single knob that sizes the worst-case compute
buffer"*, and it is: the reserve pass builds the prompt-processing graph at
`n_tokens = min(n_ctx, n_ubatch)`.

`bench/ubatch_preflight.py`, read back from `llama_context: n_ubatch = N`:

| `-ub` | compute buffer | free after load |
|---:|---:|---:|
| 256 | 472.27 MiB | 825 MiB |
| 64 | 406.27 MiB | 891 MiB |

**A 4× cut returns 66 MiB** — not enough for the arms that need it, which sit at
45–376 MiB. And `results/ubatch-98304.jsonl` prices the other side:
**`-ub 64` costs 14.0 % of decode [−14.8, −13.7], RESOLVED.**

**The `ub-128` third round is worth more than the verdict.** All three of its
boots are byte-identical in the log — same `n_ubatch`, same 428.27 MiB compute
buffer, same `projected to use 8827 MiB vs 10919`, same `11069 MiB free` at
boot. Yet `free_after`, sampled while the server ran, reads **759 · 757 ·
1,214 MiB**, and the round with 457 MiB spare ran 6 % faster. **Something else
on the machine released that memory.** It disqualifies the round and points at
the next section.

## 5. `--fit` was never following anything

`CLAUDE.md`'s own north star said: *"Free VRAM at boot moves 9,326–10,732 MiB
and `--fit` follows it."* The scan proposes the counter-move — pin `-ngl`, turn
`--fit` off — and rates it *"highest value on this list for measurement
integrity"*.

`bench/pinned_alloc_preflight.py` boots both forms at ctx 98,304. **They agree on
every observable**: `65+0`, `n_ctx 98304`, model 6,521.13 MiB, KV 1,728.00,
compute 472.27, `free_after` 1,427.

Then reading every log this project has kept says why:

```
552 logs, all reporting "RTX 4070 SUPER (12281 MiB, 11069 MiB free)"
150 boots on our artifact with a fit pass -> 148 "no changes needed"
  2 where --fit acted, both n-7-clamp at ctx 65,536, already in the ledger
```

**9,326–10,732 MiB is `nvidia-smi`'s view of the card**, desktop included, and it
does move. **11,069 MiB is what CUDA reports to the process**, and that is the
number `--fit` reasons from. It has not varied once.

**The rule stands; its cause is withdrawn.** The spread is measured — 13.6 % at
16,384, up to 48.9 % at 65,536 with byte-identical counters. What produced it is
now **unattributed**. [`CORRECTIONS.md` §27](CORRECTIONS.md).

**This one cost real time.** A wrong mechanism implies a fix, the scan proposed
exactly that fix, and it was tried before the logs were read. Grepping 552 files
was available the whole time and would have closed the question in a minute.

## 6. The largest win was already on

An agent does not grow one conversation. It works on A, switches to B, and comes
back. With one slot, A's state has to have gone somewhere.

**It does, and nothing here knew.** `--cache-ram` defaults to **8192 MiB**
(`common/common.h:615`). `prompt_save` stores `llama_state_seq_get_data_ext` —
**the whole sequence state, attention KV and recurrent together**
(`server-context.cpp:261-274`). No profile, document or sweep in this project
had ever named it. It surfaced only because
`POST /slots/0?action=erase` failed to produce a cold turn: erase clears the
slot, not the RAM cache.

`results/prompt-cache-swap.jsonl`, two disjoint 44K conversations, A→B→A→B→A:

| | `-cram 8192` (default) | `-cram 0` |
|---|---:|---:|
| A cold | 40,513.5 ms · 0.0 % | 40,655.1 ms · 0.0 % |
| **A after B** | **118.2 ms · 100.0 %** | **40,596.0 ms · 0.0 %** |
| B after A | 121.2 ms · 100.0 % | 38,775.3 ms · 0.0 % |
| saved on return | **99.7 %** | 0.1 % |

**343×.** The cold turns agree to 142 ms (0.35 %), so the arms are comparable,
and every return is 100 % rather than partial. This is the closest thing
measured here to the 3090 stack's 169 s → 4.7 s.

**What bounds it**, from the log and the source: **898–928 MiB of host RAM per
conversation**; restore is a **move, not a copy**
(`prompt = std::move(it_best->prompt)`, `server-task.cpp:1858`), so an entry
leaves the cache when loaded; and `load()` refuses any entry whose common prefix
is under **25 %** of its length — *"don't trash large prompts"* — while requiring
a candidate to beat the incumbent on **both** `f_keep` and `f_sim`.

## 7. Four techniques closed by reading, no GPU round spent

- **fp16 recurrent state** — ☠️ **would not fail, it would lie.** The prize is
  real: `S (f32): 720.00 MiB`, so halving returns ~360 MiB, five times what
  `-ub` returns. But `ggml/src/ggml-cuda/gated_delta_net.cu` contains **zero**
  occurrences of `GGML_TYPE_F32`, casts the state `(const float *)`
  unconditionally, and strides by `sizeof(float)`. Flipping the literal yields a
  server that boots, saves the memory, reports `65+0`, decodes plausibly, and
  reads its state at twice the real span with nothing logged. The scan rates it
  `small-patch`; it is new-backend.
- **`"timings_per_token"` and `"return_tokens"`** — plain request booleans the
  server already accepts (`server-schema.cpp:20,34`). The first attaches the
  full `timings` object to **every stream chunk**, which is the finest clock the
  server offers and the natural spine for the recorder's phase-pure time unit
  (#30–#36). Never set here.
- **The speculator priority list is hardcoded** —
  `common/speculative.cpp:2540-2552` registers all five `ngram-*` types above
  all five model-based ones and discards command-line order. So the measured
  **+48.5 %** for `draft-dflash,ngram-mod` ran *ngram first*, and "dflash first"
  is unmeasured.
- **`-ctkd`/`-ctvd`, `GGML_CUDA_GRAPH_OPT`, `-bs`** were closed the same way on
  2026-08-22 and are unchanged.

---

## What this session changed about what may be assumed

**Nothing shipped.** All four `worker-*.ps1` run `ngram-mod` alone and none was
modified.

**And one precision this report owes.** Every rate above is
**`UD-IQ2_XXS` at ctx 98,304**, which **no profile serves** —
`worker-iq2xxs-deep` runs that artifact at 131,072 and `worker-iq2s-quality`
runs 98,304 on the larger `UD-IQ2_S`. The decoder verdict transfers, since all
four run `ngram-mod`. The absolute rate does not, and neither does the
drafter's failure band without re-measuring — though `UD-IQ2_S` being 1.1 GB
larger argues in the same direction. **An earlier draft of this report said
these rows were what the profiles serve. They are not.**

**Three claims came out.** [§25](CORRECTIONS.md) chars/token,
[§26](CORRECTIONS.md) the decode collapse, [§27](CORRECTIONS.md) `--fit`. All
three share one shape: **the conclusion was right and the mechanism was wrong.**

That shape is worse than a wrong number, because a wrong mechanism tells the
next reader what to fix. §25 would have had someone "repair" `filler()` and
double every prompt. §26 would have had someone chase a window that works. §27
did have someone — this session — pin `-ngl` against a force that was not there.

**The guard that catches this does not exist yet.** `audit-stale-claims.py`
matches published *figures*; none of these three was a wrong figure. The nearest
thing is [`traps.md` §12](../agents/traps.md), rewritten today to ask of any
second cause: **what did this experiment hold fixed?**

## What is left in the pool

- **`llama-bench` and `llama-perplexity` are configured but not built** — one
  `cmake --build` target each. `llama-bench -d/--n-depth` measures prefill and
  decode at a real KV depth **without a server**: no slot, no `--fit`, no prompt
  cache. That is the obvious instrument for the question §27 reopened.
- **Three or more conversations in `-cram` rotation**, and its RAM cost under a
  real task mix.
- **"dflash first"**, reachable only by reordering ten lines.
- **`n-match` at 98,304** — still a third window, still unmeasured.
