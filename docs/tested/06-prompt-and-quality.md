# 06 — Prompt, output format, quality

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

## The format failure, and what was tried against it

The measured blocker: the model loops inside the reasoning block until the token
budget runs out and never emits a fence. 7 of 48 attempts truncate on budget.

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
