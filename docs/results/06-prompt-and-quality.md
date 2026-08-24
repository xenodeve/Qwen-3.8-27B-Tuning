# 06 — Prompt, output format, quality

> 🔴 **Every number on this page was measured at `reasoning_effort: xhigh` with
> an unlimited thinking budget, except the rows that state `-rea off` or `--reasoning-budget 0`.** That is the model's chat-template
> default — the client sends no effort field, and **no `worker-*.ps1` profile and
> nothing in `bench/` has ever set the flag** (established 2026-08-24 from a boot
> log: [`05-runtime-flags.md`](05-runtime-flags.md)).
> Artificial Analysis prices this model's `medium` **one point** below `xhigh` on
> the agentic axis and `low` **six** below that
> ([`researchs/artificial-analysis`](../researchs/artificial-analysis/README.md)),
> so **effort is a live confound here, not a settled background condition.**
>
> **The served default became `medium` on 2026-08-24** — all five
> `worker-*.ps1` profiles and `dflash2_arena.server_argv` now set it, and the
> arena records `effort` on every row. **So this banner describes what is
> already on the page, not what will be added to it.** Anything measured after
> that date states its own level, and a figure from before it cannot be
> compared with one from after without saying which is which.

> **Read this page knowing its limit.** `run_retry_bench.py` sends a **35-token**
> developer message and grades one reply. The worker that ships is a full Claude
> Code instance whose fixed prefix measured **39,762–40,648 tokens** across four
> calls, with a tool loop and retries. **No number on this page describes the
> worker that ships.** A missing code fence is a permanent failure here and a
> self-correcting hiccup there. `CORRECTIONS.md` §10.

## The corpus

Ten tasks (3 easy, 4 medium, 3 hard), three passes, graded by executing the
extracted code against hidden assertions. `output_contract_pct` is the **PASS**
rate: one fenced `python` block, nothing else.

| arm | accepted | contract pass | merged/hr | verified/hr |
|---|---|---|---|---|
| pre-V3 `UD-IQ2_XXS` @8,192 | 27/30 (90 %) | — | 26.5 | **48.5** |
| pre-V3 `UD-IQ2_XXS` @3,072 | 27/30 (90 %) | — | 29.4 | 60.8 |
| `AD-IQ1_M` @8,192 | 27/30 | — | 22.4 | 35.3 |
| V3 `UD-IQ2_XXS` | 19/27 | **58.3 %** | 18.3 | 20.2 |
| V3 `UD-IQ2_XXS` + `-rea off` | 15/30 | 58.0 % | 30.0 | 121.6 |
| V3 `UD-IQ2_XXS` + grammar + `-rea off` | 16/27 | **84.3 %** | 20.8 | — |
| V3 `UD-IQ1_M` | 10/21 | 41.5 % | 12.3 | — |
| V3 `UD-IQ1_S` | **0 of 12** | — | — | — |
| `Ornith-1.0-9B` @3,072 | — (66.7 %) | — | 29.2 | 72.0 |
| Ternary Bonsai 27B | — | — | 17.9 | — |

**The 60.8 and 48.5 rows are the same artifact at two token budgets.** Accept is
90 % either way, so the budget changed the wall clock, not the capability. Quote
48.5 against anything measured at 8,192. `CORRECTIONS.md` §6.

**`-rea off` at 121.6 verified/hr is the trap this table exists to expose.** It
is the fastest arm and it accepts 15 of 30. It answers wrong, quickly.

*Raw: `results/retry-bench.jsonl`. Reports 10, 12, 13, 15, 22.*

## The benchmark prompt at depth — three texts, and two ways a row can lie

**Measured 2026-08-24, issue #44.** The arena builds its prompt as
`filler(int(ctx * 0.5), regime)`, which returns `n * 3` characters of a frozen
corpus with one instruction line appended. At ctx 147,456 that is ~64,200 tokens
of real Python followed by *"# Explain what `vram_settled` guards against, then
write a test for it."*

**Eighteen rows at ctx 147,456 produced nothing.** Every generation ran **9
tokens against a 512-token budget** and stopped on EOS.
`generation_is_measurable` voided all eighteen, which is the guard working.

### It is not the window, and it is not the length

One boot at ctx 147,456, cold prefix cache, varying only the prompt:

| prompt tokens | generated | stop |
|---:|---:|---|
| 48 | 512 | limit |
| 43,162 | 512 | limit |
| 46,909 | **1** | eos |
| 51,038 | **1** | eos |
| 54,310 | 512 | limit |
| 57,780 | 512 | limit |
| 60,831 | 512 | limit |
| 64,210 | **9** | eos |

**Failure is not monotonic in length.** `filler` cuts at exactly `n * 3`
characters, so each length ends at a different point in the source — and what
decides the outcome is **where the cut lands**, not how much was taken. A cut
that leaves the model somewhere it reads as finished makes EOS the greedy
continuation of the appended instruction.

The context setting is exonerated twice over: a 48-token prompt and a
43,162-token prompt both run the full budget at the same ctx 147,456.

### Changing the text, not the length, confirms it

`real-code-vendor.txt` — 11 files of `llama.cpp`'s `gguf-py`, 597,630 chars,
`sha256[:16] d3a3e9920244ecb0`, built by
[`corpora/build-vendor-corpus.py`](../../qwen38-tuning/bench/corpora/build-vendor-corpus.py).
Real Python by people who have never seen this repository.

**The same seven lengths complete 7 of 7**, including **70,322 tokens** — deeper
than the 64,210 that collapsed. Same model, same ctx, same greedy sampler, same
hour.

### And then the new corpus lied in the other direction

The first arena row taken on it:

```
draft-mtp+ngram  195.13 tok/s  acc 100.0
  ngram-mod  decline 24.1 %  mean len 32.85  n_gen 1912  n_acc 1911
  draft-mtp  decline  0.0 %  mean len  3.84  n_gen   57  n_acc   54
```

**1,911 of 1,912 drafted tokens accepted, in runs averaging 32.85.** Not
speculation succeeding — `ngram-mod` drafts by matching text already in the
context, so a continuation that reproduces its own prompt is what it predicts
perfectly. The generated text says so plainly:

```
# ==== gguf-py/gguf/constants.py ====
from __future__ import
```

The model was **continuing the corpus, not answering the instruction**. Three of
the seven sweep lengths did this and four produced a real answer, and the length
guard passed all seven because it counts tokens. **195.13 tok/s is a copy rate**,
and it would have been the highest figure this project has ever recorded.

**Guarded since:** `harness.copied_window_fraction` measures 12-word windows of
the generation found verbatim in the prompt — 12 because
`--spec-ngram-mod-n-match 12` is what every worker profile serves, so that is the
width at which copying actually pays the decoder. Every generation must clear it,
not the median. The **0.5 limit is a first guess** separating the two populations
observed that day and was not derived; `copied_frac` is on every row so it can
move on evidence.

**The guard is on the output, never on the counters.** Voiding rows where
`ngram-mod`'s `mean_acc_len` is high would be circular: it is one of the arms
under test, and a guard that rejects rows where it does well cannot be used to
find out whether it does well.

### What this leaves

| text | hash | covers ctx | usable at 147,456 |
|---|---|---:|---|
| `real-code.txt` | frozen evidence | ~30,600 | no — depth guard raises |
| `real-code-deep.txt` | `1a3ae4b813dd8447` | ~135,000 by chars | **only with `--ignore-eos`** |
| `real-code-vendor.txt` | `d3a3e9920244ecb0` | ~199,200 | generates, but rows may be copies |

`--ignore-eos` forces the budget; the same 64,210-token prompt then runs 512
tokens and the text is an answer, opening `<think>` and reasoning about
`vram_settled`. It is **off by default and stamped on every row**, because past
the point the model would have stopped it decodes text it did not choose to
write — so a forced row's **draft acceptance is not comparable** with a natural
row's. Arm against arm within one forced run is unaffected: every arm decodes
under the same rule.

**Rows are compared within a corpus, never across.** Three hashes, three regimes.

*Raw: `results/DIAG-length-real-code-deep.jsonl`,
`results/DIAG-length-real-code-vendor.jsonl`. Traps 14 and 15.*

## The format failure, and what was tried against it

The measured blocker: attempts end without a fenced code block. 7 of 48 truncate
on budget.

> **Corrected 2026-08-21 08:30.** This line read *"the model loops inside the
> reasoning block"*. It does not — a full trace scores **0.00 % line
> repetition** and ends on `stop`. It reasons at length, which is this model's
> normal mode, and finishes. `CORRECTIONS.md` §12.

| treatment | contract pass | accepted | reading |
|---|---|---|---|
| nothing | 58.3 % | 19/27 | the control |
| `-rea off` alone | 58.0 % | 15/30 | **does not fix it** — the model moves where it reasons, into prose outside the fence and multiple blocks |
| `--grammar-file` + `-rea off` | **84.3 %** | 16/27 | format fixed, accepted fell |
| `--grammar-file` alone | **never run** | — | built 2026-08-21 03:29, aborted 03:30 when the goal reordered |
| `--reasoning-budget 0` | — | — | **does not end the block**; 24,709 characters alone, 0 content chars with the grammar |
| `max_tokens` 3,072 → 8,192 | — | 15/31 → **27/31** | on the same artifact. An undersized budget looks exactly like lost capability |

**The 26-point contract jump has never been attributed**, because the only arm
that showed it changed two things at once.

**And a grammar constrains output, not thinking.** It cannot stop a model
spending 8,192 tokens deciding — consistent with contract rising while accepted
fell.

*Raw: `results/retry-bench.jsonl`, `results/answer-screen*.jsonl`,
`grammars/python-fence.gbnf`. Reports 22, 23.*

## The prompt cache — measured, and it constrains skill injection

`results/prefix-cache.jsonl`, nine rows from `bench/prefix_cache_gate.py`:

```text
  turn-1                 prompt_n 3878   cache_n    0    12.6 s   cold
  turn-2 / 3 / 4         prompt_n 35-43  cache_n ~3900   1.3-3.9 s
  append-only-control    prompt_n   28   cache_n 3981    2.4 s
  reorder-tool-schemas   prompt_n 3990   cache_n    1   11.1 s   FULL RE-PREFILL
  edit-system-prompt     prompt_n 3992   cache_n    1   11.5 s   FULL RE-PREFILL
  prepend-skill-block    prompt_n 4006   cache_n    1   12.1 s   FULL RE-PREFILL
  cache_prompt=false     prompt_n 3991   cache_n    0    9.9 s
```

**Caching is worth a factor of five** on an append-only turn even at a 4 K
prefix. Against the ~40 K a real worker carries, far more.

**Injecting a skill at the front of the prompt re-prefills everything.** The row
is labelled *"Xeno injects skills ahead of everything"*. So `clink-subagents` §7
— hand the worker `karpathy-guidelines` on every call — and the prompt cache are
in tension **as usually implemented**. The fix is position, not omission: the
injected text must sit inside the stable prefix, byte-identical on every call,
ahead of anything that varies.

**The corpus runs `cache_prompt: False`**, so every quality number here paid a
cold prefill that production would not.

*Raw: `results/prefix-cache.jsonl`.*

**Re-measured at real depth, 2026-08-23, and a second mechanism found.**
The nine rows above are all at **3,878 tokens**, which is a tenth of what the
served profile carries. Both findings survive depth, and one of them gets much
more expensive there.

`results/prefix-cache-depth.jsonl`, `bench/prefix_cache_depth.py`, one boot at
ctx 98,304:

| | 8,147 tokens | 44,255 tokens |
|---|---:|---:|
| turn 1, cold | 6,727 ms | 35,301 ms |
| turns 2-4, append-only | 218-257 ms | 228-265 ms |
| cache reuse | 99.5-99.7 % | **99.9 %** |
| one sentence edited near the front | 0.0 %, 6,627 ms | **0.0 %, 41,810 ms** |

**A warm turn costs the same ~250 ms whether the conversation is 8K or 44K.**
Prefill is a per-conversation cost, not a per-turn one, as long as the prompt
only grows.

**The skill-injection tension above is confirmed and its price is now known.**
An edit ahead of the suffix does not degrade reuse, it **zeroes** it - 0.0 % at
both depths - and at 44K that is **41.8 s per turn**, six times the shallow
penalty. The fix stated above is unchanged and now load-bearing: the injected
text must sit inside the stable prefix, byte-identical on every call.

**The second mechanism is `-cram`, and it is what an agent switching tasks
actually needs.** `--cache-ram` defaults to **8192 MiB** and stores the whole
sequence state - attention KV and recurrent together - for idle slots. No
profile or document here had ever named it. `results/prompt-cache-swap.jsonl`,
two disjoint 44K conversations, A-B-A-B-A:

| | `-cram 8192` (default) | `-cram 0` |
|---|---:|---:|
| A cold | 40,513.5 ms, 0.0 % | 40,655.1 ms, 0.0 % |
| **A after B** | **118.2 ms, 100.0 %** | **40,596.0 ms, 0.0 %** |
| saved on return | **99.7 %** | 0.1 % |

**343x on one flag.** Costs 898-928 MiB of host RAM per conversation; restore is
a move, not a copy, so an entry leaves the cache when loaded; and `load()`
refuses any entry whose common prefix is under 25 % of its length.

**What this means for the corpus numbers below.** They still run
`cache_prompt: False`, so they still pay a cold prefill production would not -
and the gap is now measured at **40.5 s** rather than the 12.6 s the shallow
rows implied.

*Raw: `results/prefix-cache-depth.jsonl`, `results/prompt-cache-swap.jsonl`.
[Report 33](../reports/33-WHAT-THE-3090-POOL-ACTUALLY-GAVE-US.md),
[08 section 6](08-rtx3090-transfer.md).*

## Skill injection — measured once, on the wrong machine and the right one

`clink-subagents` §7 requires it: *"`karpathy-guidelines` on every call; `tdd`
whenever the worker writes or changes code."*

| evidence | model | result |
|---|---|---|
| `qwen-agent` A/B, 2 tasks × 3 runs | **`qwen3.6-35b-a3b`** | tightly-scoped prompt: **no difference**. Neutral prompt: guidelines arm surgical 3/3 vs 2/3 |
| this project, 2 hard corpus tasks × 2 arms | **`qwen3.8-27b-fp8`** | **all four PASS**, first try, no retry |

**Neither settles it.** The first ran on a different, weaker model. The second
had no failures to separate the arms — on fp8 the corpus is too easy, which is
a limit of the instrument, not a result.

**The path form of handoff does not work through `claude-9arm`.** The worker
attempted `Read` twice and `cat` twice; all four were permission-denied, and it
returned `READ_FAILED` as instructed. Without that sentinel the run would have
looked normal while measuring "no skill" in both arms. **Inline pasting is the
only mechanism that works there.**

*Raw: none — `mcp__pal__clink` calls graded by hand with `run_bench.verify`,
2026-08-21. `--skill` is wired into `run_retry_bench.py` for the local runs.*

## Deep-context retrieval quality

Verified on **`Q4_K_XL` only**. Nine artifacts have depth throughput numbers and
**none has a depth quality number**. Open since 2026-08-17.

## Never tried

- A corpus at 128 K or beyond, on any artifact.
- A corpus with n-gram enabled — the config that would ship.
- `reasoning_effort: low` through the corpus.
- A system prompt that instructs *how to think*.
- Anything that measures the agent loop rather than a single completion.
