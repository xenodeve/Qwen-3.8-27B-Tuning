# 08 — what actually transferred from the RTX 3090 stack

> 🔴 **Every number on this page was measured at `reasoning_effort: xhigh` with
> an unlimited thinking budget.** That is the model's chat-template
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

**The register for one question: of the 434 techniques scanned out of
`syv-ai/qwen38-27b-rtx3090`, which ones were tried here, and what happened.**

Source of the pool: [`../researchs/syv-rtx3090/`](../researchs/syv-rtx3090/README.md)
— 434 techniques matched against a 175-capability map of our own llama.cpp, of
which **48 were "a flag we already have and have never set"**.

Flag semantics for the shortlist:
[`../researchs/llamacpp-flag-semantics-2026-08-22.md`](../researchs/llamacpp-flag-semantics-2026-08-22.md).

> **Read the status column literally.** *Measured* means a paired run on this
> card. *Read* means an argument from source with no run behind it. The two are
> not interchangeable and this project has confused them before.

---

## The scoreboard

| # | technique (their name) | our lever | status | outcome |
|---|---|---|---|---|
| 1 | "k=4 is the knee" / draft count reduced at long context | `--spec-draft-n-max` | **MEASURED** | **+23.4 % [+23.1, +23.5] RESOLVED at ctx 16,384.** Collapses at 65,536 |
| 2 | "NSTRONG=NMIN: take any qualifying match" | `--spec-ngram-mod-n-min` | **MEASURED** | **No effect.** 16/8/4/2 → 79.7/79.7/79.7/79.8 |
| 3 | "NMAX=12 chosen against 32 so recency beats length" | `--spec-ngram-mod-n-match` | **MEASURED, twice** | **Depth-dependent.** At 16,384 the default `24` is **+34.6 % RESOLVED**; at 65,536 the optimum is **`16`** and 24 is a null. Our `12` loses at both |
| 4 | Lookup hit counter | `-lv 4` per-impl statistics | **USED** | the finding under every other finding — see below |
| 5 | "A longer verify block costs KV pool per slot" | `n_rs_seq` | **USED** | gave the exact price: `149.625 MiB × (1 + n_max)` |
| 6 | Applying top-k/top-p to the selector proposal | `--spec-draft-p-min` | **MEASURED** | **Null.** At `0.10` the early-stop **never fires**; at `0.25` it fires on 2.2 % of calls and buys nothing |
| 7 | Weight-aware pool budget | `-fitt` | **READ** | and it does not mean what our profiles say — below |
| 8 | V2-runner fix: drafter's KV forced back to bf16 | `-ctkd` / `-ctvd` | **READ — do not sweep** | saving ~32 MiB, paid for on the hot path |
| 9 | "Complete tunable surface: six environment variables" | `GGML_CUDA_GRAPH_OPT=1` | **READ — do not sweep** | cannot fire on our shape |
| 10 | FlashInfer radix top-k with a latching fallback | `-bs` | **READ — do not sweep** | offloads to sampler 2 of 10; dies on a grammar |
| 11 | DFlash2 + context lookup ("their most important idea") | `--spec-type draft-dflash,ngram-mod` | **MEASURED** | **+48.5 % over `ngram-mod` alone on real code** |
| 12 | Quantised DFlash2 drafter (3.85 GB bf16 → 1.19 GB W4A16) | — | **ALREADY HAD IT** | our GGUF is 1.06 GB and was never bf16 |
| 13 | Adaptive verify block beyond the trained block size | — | **IMPOSSIBLE** | `speculative.cpp:989` clamps at `block_size − 1` = 7 |
| 14 | KVarN 4-bit K / 2-bit V | — | **IMPOSSIBLE** | llama.cpp's KV types bottom out at 4 bits |
| 15 | Hybrid recurrent-state prefix cache | `cache_prompt` + `--ctx-checkpoints` | **MEASURED** | **It transfers.** 99.9 % reuse and ~250 ms per warm turn at 44,255 tokens — but any edit ahead of the suffix costs a full re-prefill (41.8 s) |
| 16 | "max-num-batched-tokens chosen against the KV pool" | `-ub` / `--ubatch-size` | **MEASURED — and refused** | Mechanism correct, exchange rate bad. A 4x cut returns **66 MiB** and costs **−14.0 % decode RESOLVED**. `05-runtime-flags.md` |
| 17 | fp16 Gated-DeltaNet recurrent state | `recurrent_type_r/_s` literals | **READ — ☠️ DO NOT ATTEMPT** | Would return **360 MiB** and **corrupt silently**: the DeltaNet CUDA kernel has *no* type check and casts the state to `float *` unconditionally. Scan rates it `small-patch`; it is new-backend |
| 18 | Per-token arrival capture / exact-token replay | `"timings_per_token"`, `"return_tokens"` | **READ — AVAILABLE NOW** | Both are plain request booleans, no patch and no server flag. Directly serve the recorder (#30–#36). Never set here |
| 19 | Lookup applied after the selector rather than before | speculator priority list | **READ — needs a patch** | `speculative.cpp:2540-2552` hardcodes every `ngram-*` above every model-based type, so our measured `draft-dflash,ngram-mod` **ran ngram first**. "dflash first" is unmeasured |
| 20 | "Pin the KV pool in bytes instead of by utilization" — the scan's *highest value for measurement integrity* | explicit `-c N` + `-ngl N`, `--fit off` | **MEASURED — no effect, and it retired a rule** | Pinned and auto agree on every observable at ctx 98,304. `--fit` had nothing to pin: llama.cpp sees **11,069 MiB free in all 552 logs** and 148 of 150 boots say *no changes needed*. [`CORRECTIONS.md` §27](../reports/CORRECTIONS.md) |
| 21 | "Shared-versus-distinct prompt switch to model prefix-cache-friendly clients" | `-cram` / `--cache-ram` (default **8192 MiB**, never set here) | **MEASURED — the strongest single result on this page** | Returning to a 44K conversation after working on another: **118 ms at 100 % reuse with the default, 40,596 ms at 0 % with `-cram 0`.** A 343x difference on a flag nobody knew was on. §6 |

**Twenty-one rows, and the tally is the point of the page.** **Five measured
wins** — one by refuting the claim that produced it, and one on a flag that was
already switched on. Two measured nulls. **One measured and refused**, where the
mechanism held and the price did not. **One measured to no effect, which retired
a rule.** Two instruments we already had and had not read. Four read-and-closed
without a GPU round, one of which would have corrupted output silently had it
been tried. Three read and still live. One we already had, two the architecture
forecloses. **None open.**

**The largest win was not a setting to add.** `#21` — `-cram` — was **on by
default the whole time**, worth **343x** on the case an agent actually produces,
and had never been named in a profile, a document or a sweep. It was found only
because an erase that should have produced a cold turn did not.

**Two results retired a claim instead of adding a setting.** `#16` refused a
flag whose mechanism held, and `#20` refuted the stated cause of this project's
own no-cross-boot rule. Neither changes a profile. Both change what the next
reader is allowed to assume.

**The read-and-closed column is worth as much as the wins.** Four GPU rounds not
spent, and `#17` is the sharpest of them: following the scan's own `small-patch`
rating would have produced a server that boots, saves 360 MiB, reports a healthy
split and a plausible rate, and reads its recurrent state at twice the real
span with nothing logged.

**The refutation is one of the wins.** Their claim was that a *shorter* match
predicts better. Measured here the direction reverses, and acting on the claim
would have made the served profile slower still. A transfer that tells you to
undo something you already did is worth the same round as one that tells you to
add something.

---

## 1. `--spec-draft-n-max` — the transfer that paid

Two agents in the scan independently called it *"the biggest single unclaimed
win on this list"*. It was, and by more than they predicted.

| `n-max` | ctx 16,384, real-code | vs ours |
|---:|---|---|
| 3 — **the llama.cpp default** | 70.2, 70.5, 70.2 | **−11.5 %** |
| 4 — what we shipped | 79.3, 79.7, 79.5 | baseline |
| **7 — the clamp** | 97.7, 98.4, 98.2 | **+23.4 % RESOLVED** |

**The default sits 28 % below the best point** and the help text does not say so.
Full data and the VRAM price in
[`02-decoders.md`](02-decoders.md) and [report 31 §5](../reports/31-SESSION-RECORD-2026-08-22.md).

**It does not transfer up in depth.** At ctx 65,536, `n=7` loads `63+2` — two
layers on the CPU — and the recurrent state splits with 49.88 MiB landing on the
host. **There is no single value to ship: the best setting depends on the window,
and the window is set by the task.**

## 2. `--spec-ngram-mod-n-match` — their claim reversed, and it found our own error

The scan read their patch as *"NMAX=12 chosen against 32 so recency beats
length"*, and [report 30](../reports/30-SYV-RTX3090-REFERENCE-REVIEW.md) noted
approvingly that our profile already ran `12` — **"the same cap, chosen
independently"**. Convergence read as confirmation.

Measured, the direction reverses:

| `n-match` | ctx 16,384, real-code | vs ours | ngram drafts | ngram mean acc len |
|---:|---|---|---:|---:|
| **24 — the llama.cpp default** | 94.5, 96.3, 94.2 | **+34.6 % RESOLVED** | 29 | **23.45** |
| 16 | 69.2, 69.7, 69.5 | −1.5 %, within the floor | 25 | 19.20 |
| 12 — what we ship | 71.7, 73.3, 66.9 | baseline | 31 | 18.00 |
| 8 | 56.7, 62.7, 61.5 | **−14.5 % RESOLVED** | 43 | 8.95 |

**The trap the sweep was designed around is what produced the result.** A
shorter key is a strictly weaker requirement, so ngram fires more often — 43
drafts at `8` against 29 at `24` — and decodes slower anyway, because the
collapsed key returns the successor of whichever context last wrote the slot.
Mean accepted length falls **23.45 → 8.95** and the draft calls needed for the
same 512 tokens rise **475 → 649**. **Firing twice as often on a worse draft is
a loss**, which is exactly what `speculative.cpp:2545` sitting six lines above
`2551` predicts.

**Their number was not wrong, it was theirs.** Their cap is a bound on a
longest-match search with recency tie-breaks; ours is a hash key width into a
keyless table with no length dimension at all
(`common/ngram-mod.cpp:15-25, 37-41`). The two flags share a number and nothing
else — which the flag-semantics read had already said, and which the "chosen
independently" line in report 30 walked past.

Full data, the two limits on it, and the determinism note:
[`02-decoders.md`](02-decoders.md). Retraction:
[`CORRECTIONS.md` §21](../reports/CORRECTIONS.md).

## 3. The lookup counter — the instrument that made everything else visible

The scan called this *"strictly better than what vLLM built, and switched off"*.
It was switched off in the sense that nobody read it: `-lv 5` was already in our
arena, so the data existed before the question did.

It is why we know `ngram-mod` **declines 94–97 % of the calls it receives on real
code** while being worth **6× more per draft** when it does fire, and that
`draft-dflash` is called exactly the number of times ngram declines. The pooled
`draft acceptance` line cannot show any of that, and the pooled line is what every
earlier measurement in this project read.

**Nothing else on this page would have been findable without it.**

## 4. The two "do not sweep" verdicts that saved GPU rounds

Worth as much as a win, and each rests on a specific line:

- **`-ctkd`/`-ctvd`.** The drafter decodes `n_max + 1` = 5 tokens in **one**
  `llama_decode` (`speculative.cpp:1183, 1196`), so `Q->ne[1] == 5` and a
  quantised draft KV takes `MMA_F16` with a full dequantisation per layer per
  step — not the vector kernel. ~32 MiB saved, paid for on the hot path.
- **`GGML_CUDA_GRAPH_OPT`.** Its body (`ggml-cuda.cu:4339-4551`) contains **no
  `cudaGraph*` call at all**. "Our logs show CUDA graphs are on" is not evidence
  about this flag.
- **`-bs`.** Offloads the longest *prefix* of the sampler chain, which with our
  `--samplers` ends at position 2 of 10; `penalties` refuses outright
  (`llama-sampler.cpp:3018`), and it self-disables on a grammar — which the
  served profile needs.

## 5. `--spec-draft-p-min` — the arithmetic redesigned the sweep, and was still too generous

The scan called it *"the single most actionable item in this slice"*. It may be,
but not at the values anyone would try first.

The greedy check (`speculative.cpp:1264-1268`) compares `1/sum`, where
`sum = Σ_{k=0..15} exp(scores[k] − scores[argmax])` over the **16** selector
candidates. The argmax term is exactly 1 and every other term is ≤ 1, so:

> **`1/sum ∈ [0.0625, 1.0]`. Any `p_min ≤ 1/16` is mathematically identical to
> 0.00.**

A ladder starting at 0.05 would have repeated the `--spec-ngram-mod-n-min` error
exactly: four arms that cannot differ. The arm set in `dflash2_arena.ARM_SETS`
now starts at 0.10 for this reason.

It also saves **zero draft-side compute** — the whole block is decoded at
`speculative.cpp:1195` before any check — and `ngram-mod` outranks
`draft-dflash`, so every step ngram serves is a step where it does nothing.
*(That last clause turned out to be a small effect: `ngram-mod` serves only
about 6 % of steps on real code, so `p_min` is live on the other 94 %. It still
measured null.)*

**Measured, and the bound was still too generous.** 0.00 / 0.10 / 0.25, three
paired rounds: **+2.2 % and +1.5 %, both inside the floor with the sign
flipping**. The counters are the real result — at `0.10` **every counter is
byte-identical to the baseline**, so the early-stop *never fired*; at `0.25` it
fired on **2.2 %** of calls. The algebraic floor of 1/16 was correct and the
empirical distribution is far tighter than it: on this workload the selector's
confidence sits above 0.10 essentially always. **Designing the arms above the
algebraic bound was necessary and not sufficient.**
[`02-decoders.md`](02-decoders.md).

## 6. Prefix reuse — the last open item, and it transferred — measured 2026-08-23

`results/prefix-cache-depth.jsonl`, `bench/prefix_cache_depth.py`, one boot at
ctx 98,304, `--spec-type ngram-mod`, corpus `real-code-deep` sha `1a3ae4b813dd8447`.

Their `PREFIX_CACHE=1` took turn 2 of a 24K chat from ~23 s to 1.15 s. Ours:

| | 8,147 tokens | 44,255 tokens |
|---|---:|---:|
| turn 1, cold | 6,727 ms | 35,301 ms |
| turns 2–4, append-only | 218 / 223 / 257 ms | 228 / 254 / 265 ms |
| cache reuse on those turns | 99.5–99.7 % | **99.9 %** |
| saved per warm turn | **96.7 %** | **99.3 %** |
| one sentence edited near the front | 0.0 %, 6,627 ms | **0.0 %, 41,810 ms** |

**A warm turn costs the same ~250 ms whether the conversation is 8K or 44K
tokens.** That is the finding, and it is the one that reorders the cost model:
prefill is a per-conversation cost, not a per-turn one, as long as the prompt
only grows.

**It works despite `n_rs_seq = 0`, not because of a rollback.** Read from source
and confirmed in the boot log. Qwen3.8 loads as arch `qwen35`, which is on the
`llm_arch_supports_rs_rollback` whitelist (`src/llama-arch.cpp:1044`), so the
recurrent half *can* be partially removed — but only by `n_rs_seq` tokens
(`src/llama-memory-recurrent.cpp:180-192`), and `n_rs_seq` is `draft.n_max`,
**zero unless a model-based drafter is loaded** (`common/common.h:386`). Every
`worker-*.ps1` runs `--spec-type ngram-mod` alone, so
`common_context_can_seq_rm` classifies the context as `SEQ_RM_TYPE_FULL`. What
covers the gap is `--ctx-checkpoints`, **default 32** (`common/common.h:613`),
which the server enables for exactly the FULL and RS cases
(`tools/server/server-context.cpp:3372-3376`). The log confirms
`n_rs_seq = 0` on both probes.

**The reuse is not one mechanism but two, and the second was invisible to us.**
`--cache-ram` defaults to **8192 MiB** (`common/common.h:615`) — a server-level
prompt cache in host RAM into which *idle slots are saved and from which they
are restored*. It is why the deep probe's "cold" turn 1 came back at **17.8 %
reuse** after an explicit `POST /slots/0?action=erase`: erase clears the slot,
not the RAM cache, and 7,887 tokens of the shallow probe's prompt were still
recoverable. **No profile here has ever set `-cram`, and nothing in this
project had noticed it was on.** Its effect on a slot-thrashing multi-agent
workload is unmeasured.

**The cost of the failure case grows with depth and is the thing to design
against.** An edit ahead of the suffix does not degrade reuse, it zeroes it:
0.0 % at both depths, and at 44K that is **41.8 s** — six times the shallow
penalty. OpenCode may reserialize tool schemas between turns, and
`results/prefix-cache.jsonl` (2026-08-22, 3,878 tokens) already showed a tool
reorder, a one-sentence system-prompt edit, and a prepended skill block each
collapsing reuse to a single token. **Anything injected ahead of the
conversation must be byte-stable across turns or the entire prefill is repaid
every turn.**

**Limits of this measurement, stated.** One boot, one round per depth, no
repeats — so these are not paired figures and carry no verdict label. Decode was
not measured here (`n_predict` 8). The conversation is an agent-shaped synthetic,
not a captured OpenCode session, so it establishes that the *mechanism* works at
depth, not that OpenCode's actual serialization keeps it working.

### And the second mechanism is the one that matches their claim — `-cram`

The in-slot reuse above only covers **one growing conversation**. An agent that
switches between tasks does something the probe above cannot see: it leaves A,
works on B, and comes back. With one slot, A's state has to have gone somewhere.

**It does.** `--cache-ram` defaults to **8192 MiB** (`common/common.h:615`) and
**no profile, document or sweep in this project had ever mentioned it.** It was
found only because `POST /slots/0?action=erase` failed to produce a cold turn.

`prompt_save` stores `llama_state_seq_get_data_ext` — **the whole sequence
state, attention KV and recurrent together** (`server-context.cpp:261-274`), not
a token list. Idle slots are saved into it and cleared for the next task.

**Measured 2026-08-23**, `results/prompt-cache-swap.jsonl`,
`bench/prompt_cache_swap.py`. Two disjoint 44K-token conversations, A→B→A→B→A,
one boot per arm:

| | `-cram 8192` (the default) | `-cram 0` |
|---|---:|---:|
| A cold | 40,513.5 ms, 0.0 % | 40,655.1 ms, 0.0 % |
| **A after B** | **118.2 ms, 100.0 %** | **40,596.0 ms, 0.0 %** |
| B after A | 121.2 ms, 100.0 % | 38,775.3 ms, 0.0 % |
| A again | 114.6 ms, 100.0 % | 40,604.6 ms, 0.0 % |
| saved on return | **99.7 %** | 0.1 % |

**A 343× difference on one flag.** The two cold turns agree to within 142 ms
(0.35 %), so the arms are comparable; every return is **100 %** reuse, not
partial. This is the closest thing measured here to the 3090 stack's claim of a
100K prefix falling from 169 s to 4.7 s.

**What it costs and what bounds it**, from the log and the source:

- **898–928 MiB of host RAM per conversation** at this depth
  (`saving prompt with length 44261, total state size = 928.496 MiB`), which the
  cache accounts as ~1,200 MiB. Against 8,192 MiB that is roughly six.
- **Restore is a move, not a copy** — `prompt = std::move(it_best->prompt)`
  (`server-task.cpp:1858`), so an entry leaves the cache when it is loaded. With
  one slot and two tasks the cache holds exactly one, which is what the log
  shows.
- **A 25 % floor on what may be evicted.** `load()` skips any entry whose
  common prefix is under a quarter of its length — *"don't trash large prompts"*
  (`server-task.cpp:1810`) — and requires the candidate to beat the incumbent on
  **both** `f_keep` and `f_sim`. A short prompt therefore cannot displace a long
  cached one.

**Not measured:** three or more conversations in rotation, the RAM cost under a
real agent's task mix, and what `-cram` does when the host is under memory
pressure.

## 7. `-fitt` — the transfer that found an error in our own profiles

`tools/server/server-context.cpp:1074` **adds the draft model's bytes** to
`fit_params_target` before `--fit` runs. With the 1,090 MiB sidecar, our
`--fit-target 768` reaches `fit.cpp` as roughly **1,900–2,100 MiB**.

Every worker-profile header describing 768 as "the margin the server leaves free"
is wrong in the configuration we now serve — and it explains why the drafter
fitted at ctx 98,304 when arithmetic from the raw buffer sizes said it should not.

---

## The thing this page most nearly got wrong

Two RESOLVED wins on the same baseline, the same corpus, the same binary. The
obvious next act is to ship both. **Measured, they cancel:**

| arm | ctx 16,384, real-code | vs the better single |
|---|---|---|
| `n-max 7`, `n-match 12` | 94.4, 95.1, 94.9 | — |
| `n-match 24`, `n-max 4` | 98.4, 97.7, 97.4 | — |
| **both** | 65.5, 63.4, 65.5 | **−33.8 % [−35.1, −32.7] RESOLVED** |

Both singles **replicated** in these rounds (+26.4 % and +30.5 % against their
earlier +23.4 % and +34.6 %), so this is an interaction and not a failure of
either half. Stacked they reach **52.4 % of what independence predicts**, and
the combination is the slowest arm in the set.

The mechanism is the cascade both levers act on. `n-match 24` makes `ngram-mod`
stricter; `n-max 7` makes `draft-dflash` longer and dearer per call. Alone each
is survivable. Together **ngram nearly stops firing — 12 drafts, 97.7 %
decline — and dflash pays a full 8-token draft on almost every step at a 34.9 %
hit rate**, generating 3,612 draft tokens to keep 1,262.
[`02-decoders.md`](02-decoders.md) carries the counters and the one unverified
step in the explanation.

> **A measured win plus a measured win is not a measured win.** Nothing on this
> scoreboard licenses adding two rows together — every row is a paired
> comparison against one baseline, and that is all it is.

---

## What did NOT transfer, and why

**The stack.** vLLM 0.27.1 with W4A16 safetensors. A 27B W4A16 checkpoint is
16–19 GB before their optimisations; 12 GB does not hold it.

**The adaptive verify block.** Presented as an idea to import; it is a tuning of
kernels we do not run. Its sweet spots of 16 and 21 query tokens are GPTQ-Marlin
tiling the M dimension in 16 rows and their own attention patch fitting
`128 // 6 = 21` rows per tile. llama.cpp clamps at 7 regardless.

**KVarN.** 5,767 lines, the largest directory in their repo, and unreachable:
llama.cpp's KV types bottom out at 4 bits and we already run that floor.

**Their numbers.** An RTX 3090 24 GB says nothing directly about a 12 GB card —
except for one comparison that does travel, and it is the most useful thing in
the whole review: **their no-speculation baseline is 46 tok/s and ours is ~44.**
The gap is entirely in the speculation stack, not the card and not the
quantisation. [Report 30](../reports/30-SYV-RTX3090-REFERENCE-REVIEW.md).

---

## Still open from this pool

- ✅ **Unblocked and measured — and the answer inverted.** The prerequisite was
  a second frozen corpus, since `real-code.txt` is 2.1× short of 65,536;
  `real-code-deep.txt` (406,146 chars, 0.4 % window repetition at n=24) now
  exists. Measured there: **the optimum moves from 24 to 16**, `24` becomes a
  null, and **the value we ship (`12`) is the second-worst arm at that depth**.
  `CORRECTIONS.md` §22.
- 🔴 **98,304 is still a third window** and nothing licenses assuming 16 holds
  there. This page assumed transfer once and was wrong.
- 🔴 **The 13.6 % floor does not survive depth.** At 65,536 the same arm, with
  byte-identical counters, spans up to **48.9 %** across boots against
  0.8–10.6 % at 16,384. Every number on this page was resolved at 16,384 where
  the floor was measured; nothing here may be re-used at depth without
  re-deriving it. `CORRECTIONS.md` §23.
- **`--spec-draft-p-min` above 0.25** — untested, and the measured trend gives
  no reason to expect a win. At 0.25 the early-stop fires on 2.2 % of calls;
  a value aggressive enough to bite often would start discarding tokens that
  would have been accepted, since dflash already keeps only 2.91 of 5.
- **#7 `-fitt`** — a step function with a dead zone whose step moves with boot
  VRAM. Read the fitted configuration from the log before measuring any rate.
- ✅ **#15 recurrent-state prefix reuse — CLOSED, and it transferred.** 99.9 %
  reuse and ~250 ms per warm turn at 44,255 tokens; see §6. The question as it
  was posed — *"does `--cache-reuse` restore DeltaNet state or only KV"* — turned
  out to be aimed at the wrong flag: `--cache-reuse` is the **chunk-shifting**
  path, was never set here, and is gated on `llama_memory_can_shift`, which the
  hybrid answers by asking **only its attention half**
  (`src/llama-memory-hybrid.cpp:133-135`). What actually carries the reuse is
  plain `cache_prompt` plus `--ctx-checkpoints`. **Two things left open by it:**
  `--cache-reuse` itself is still unmeasured and would only bite on the
  edited-head case, and `-cram` (default 8192 MiB, never set by any profile) has
  no measurement at all.
- **The speculator order.** `speculative.cpp:2540-2552` hardcodes every `ngram-*`
  above every model-based type and discards command-line order, so our measured
  `draft-dflash,ngram-mod` **+48.5 %** ran *ngram first*. Since dflash alone beat
  ngram alone by **+34.7 %**, "dflash first" is unmeasured and reachable only by
  reordering ten lines.
