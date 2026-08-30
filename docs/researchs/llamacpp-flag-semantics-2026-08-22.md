# llama.cpp flag semantics, read from source — 2026-08-22

**Source reading, not measurement.** Six agents each traced one flag from
`common/arg.cpp` into the code that *consumes* the value, in
`C:\AI\llama.cpp` at commit `1deefcca3` — llama.cpp PR #27342 on master,
the tree that built build 10499.

This folder is normally external material. **This file is the exception:**
it describes our own tool, so its claims are checkable against a tree we
have. What it does **not** contain is a single measured number — every
prediction here is an argument from code and is owed a run before it is
quoted as a result.

## Why it was written

Two sweeps on 2026-08-22 were designed against a misreading of the source
and measured nothing:

- **`--spec-ngram-mod-n-min`** was swept as a fire-rate knob. In
  `common/speculative.cpp:1993`, `i` counts **draft tokens already
  produced**, not matched context, so it is a minimum draft *length* and
  the declines happen at `i = 0`. Four arms, twelve runs, 0.3 % spread.
- **`--fit-target 768`** was documented in a worker profile as "left at
  the default". The default is **1024** (`common/common.h:473`).

So each agent was asked for a **`misreading_risk`** field: *the
plausible-but-wrong reading, and the line that refutes it.* That field is
the reason this document exists, and it is reproduced in full below.

Each was also asked whether our current value is **already the only
sensible one** — because saving a GPU round is worth as much as finding a
win. **Three of six came back `already_at_best`.**

## Verdicts at a glance

| flag | default | ours | sweep? |
|---|---|---|---|
| `--spec-draft-p-min` | 0 | see below | yes, with caveats |
| `--spec-ngram-mod-n-match` | 24 (common | see below | yes, with caveats |
| `-ctkd` | f16 for both, hard-coded at common/com | see below | **NO — provably inert here** |
| `GGML_CUDA_GRAPH_OPT` | Unset = disabled | see below | **NO — provably inert here** |
| `-fitt` | 1024 MiB per device | see below | yes, with caveats |
| `-bs` | false (common | see below | **NO — provably inert here** |

---

## `--spec-draft-p-min (alias --draft-p-min, env LLAMA_ARG_SPEC_DRAFT_P_MIN)`

**Default.** 0.0f — common/common.h:329 (`float p_min = 0.0f;`), confirmed in the staged binary: `llama-server.exe --help` prints "minimum speculative decoding probability (greedy) (default: 0.00)". At 0.0 the greedy check is skipped entirely by the `if (params.p_min > 0.0f)` guard at common/speculative.cpp:1262, so the current setting is a true zero-cost no-op, not a threshold that always passes.

### What it actually does

On our exact config it is a confidence floor on the DFlash2 *selector lattice*, not on any token probability, and it touches nothing but the draft-dflash implementation.

Parsing: common/arg.cpp:4100-4106 writes it to params.speculative.draft.p_min (a float member of common_params_speculative_draft, common/common.h:329). Only the five draft-model speculators read it; every ngram speculator has its own struct and never sees it.

Which branch is live for us. common/speculative.cpp:978 sets `is_dflash2 = selector_top_k > 0` from the GGUF key `dflash.selector_top_k`. Our drafter (Qwen3.8-27B-DFlash2-Q4_K_M.gguf) has dflash.selector_top_k = 16 and dflash.block_size = 8, so is_dflash2 = true and control enters the DFlash2 lattice branch at speculative.cpp:1219. The classic "draft token probability" check at speculative.cpp:1328 sits in the `else` arm of that same `if (is_dflash2)` and is DEAD CODE for this artifact.

Inside the DFlash2 branch the code forks on the *request* temperature (speculative.cpp:1238), which the server fills from the request's sampling.temp (tools/server/server-context.cpp:2944):
- temperature > 0.0 (stochastic): speculative.cpp:1240-1256 builds a softmax over the 16 lattice scores divided by dp.temperature, samples an index with std::discrete_distribution, and speculative.cpp:1254 compares `dist.probs[predecessor] < params.p_min` — i.e. the temperature-scaled probability of the token it actually *sampled*.
- temperature == 0.0 (greedy) — OUR CASE: speculative.cpp:1259-1261 takes `predecessor = argmax(scores)`. speculative.cpp:1262 guards the whole check behind `if (params.p_min > 0.0f)`. speculative.cpp:1264-1267 computes `sum = Σ_{k=0..15} exp(scores[k] - scores[predecessor])`, and speculative.cpp:1268 compares `1.0f / sum < params.p_min`. That 1/sum is exactly softmax(scores)[argmax] over the 16 selector candidates, with no temperature scaling.

What the compared number is, precisely. speculative.cpp:1234-1236: `row = lattice + (beg+i)*n_embd_dec`; `scores = row + selector_top_k + predecessor*selector_top_k`. So row[0..15] are the 16 candidate token ids at block position i, and scores[0..15] are the transition scores from the previously chosen candidate index to those 16. p_min is compared against a softmax over SIXTEEN candidates — not over the 151k vocabulary. It is a "how sure is the selector which of its own 16 shortlist entries comes next", never "how likely is this token under the draft model".

What happens to the rest of the block when it trips. speculative.cpp:1269 is a bare `break` out of the per-position loop. Tokens already pushed at speculative.cpp:1272 stay in `result`; every remaining position of the block is discarded. Then speculative.cpp:1276-1281: `if (result.size() < params.n_min) { result.clear(); dp.dists->clear(); }` — a truncated draft shorter than n_min is thrown away whole. We do not pass --spec-draft-n-min, so n_min = 0 (common/common.h:328) and truncation only shortens; it can only produce an empty draft if it trips on the very first position (i = 1).

Critical cost note: the draft block was already computed. speculative.cpp:1195 does ONE `llama_decode(ctx_dft, batch)` over the whole block (n_max + 1 = 5 tokens, built at speculative.cpp:1182-1188) BEFORE any p_min check runs. So p_min saves zero draft-side compute. Its only possible saving is a narrower verification batch on the target model.

### The plausible-but-wrong reading

There are FOUR plausible-but-wrong readings here, and the first two would each produce a sweep that measures nothing.

1) WRONG: "it is the draft model's token probability floor — if the drafted token's probability drops below p_min, stop." REFUTED BY common/speculative.cpp:978 + 1219. That reading describes speculative.cpp:1328, which lives in the `else` arm of `if (is_dflash2)`. Our drafter's GGUF carries dflash.selector_top_k = 16, so speculative.cpp:978 sets is_dflash2 = true and the code `continue`s at speculative.cpp:1282 before ever reaching line 1328. The live comparison is speculative.cpp:1268 against a softmax over 16 selector lattice scores — a different quantity on a different distribution with a different support.

2) WRONG: "any small positive value will start trimming weak tokens, so 0.05 or 0.1 is a gentle first step." REFUTED BY the arithmetic of common/speculative.cpp:1264-1268. `sum = Σ_{k=0..15} exp(scores[k] - scores[predecessor])` where predecessor is the argmax (line 1259-1260), so every term is ≤ 1 and the k=argmax term is exactly 1. Therefore 1 ≤ sum ≤ 16, so 1/sum ∈ [0.0625, 1.0]. **Any p_min ≤ 1/selector_top_k = 0.0625 can NEVER trip.** A run at p_min = 0.05 is behaviourally identical to 0.00 except for a wasted 16-term exp loop per drafted token; any tok/s difference it shows is boot noise and the repo's 13.6 % rule applies. This is the exact shape of the --spec-ngram-mod-n-min mistake: a threshold swept in a range where the code proves it is inert.

3) WRONG: "we also run ngram-mod, so p_min shapes those drafts too." REFUTED BY common/speculative.cpp:1887-2060 — the ngram-mod implementation never references p_min; its only stop conditions are `mod.get(...) == EMPTY` (line 1994) and its own params.n_min (line 1995). Worse for attribution: common/speculative.cpp:2542-2551 registers NGRAM_MOD *before* DRAFT_DFLASH in the priority list, and the chaining loop at common/speculative.cpp:2724-2726 sets `dp.drafting = false` as soon as an implementation returns a non-empty result. So on every step where ngram-mod hits, DFlash is skipped entirely — common/speculative.cpp:1169-1174 makes its batch empty and it returns at line 1191 without a decode. p_min is live only on the subset of steps where ngram-mod missed.

4) WRONG: "the help says '(greedy)', so at temperature 0.0 this flag is inactive / it only matters when sampling." REFUTED BY common/speculative.cpp:1238 + 1259-1262. temperature 0.0 takes the `else` (greedy) arm, which is precisely where the 1/sum check at line 1268 lives. Temperature 0.0 is the case where the flag IS active. The word "greedy" in the help string describes which branch the check belongs to, not a condition under which it is disabled. Conversely, raising temperature above 0 does not disable it — it switches to a different check (line 1254) against the temperature-scaled probability of a *randomly sampled* index, with RNG in the loop. The two are not comparable measurements of the same knob.

Bonus trap: tools/server/README.md:886/1033/1098 show `"speculative.p_min"` in a /props payload, which suggests a per-request override. tools/server/server-schema.cpp:198-227 wraps that whole field registration in `#if 0`. In this build the CLI flag is the only way to set it, and a request body carrying speculative.p_min is silently ignored.

### Interactions

- **ngram-mod suppresses it.** NGRAM_MOD has higher speculator priority (speculative.cpp:2542-2551) and the chain stops at the first non-empty draft (speculative.cpp:2724-2726). Every step ngram-mod hits is a step where p_min does nothing. With --spec-ngram-mod-n-match 12 on a repetitive corpus the hit rate can be high, which dilutes any p_min effect toward zero.
- **--spec-draft-n-min clamps the damage into a cliff.** speculative.cpp:1276-1281: if p_min truncates the draft below n_min, the ENTIRE draft is discarded (result.clear() plus dists->clear()). We do not set it, so n_min = 0 (common.h:328) and truncation only shortens. If a sweep sets both, they confound multiplicatively.
- **selector_top_k silently sets the flag's dead zone.** It is read from GGUF metadata at speculative.cpp:975-978, not from any flag. Our value is 16, giving a hard no-op zone of p_min ≤ 0.0625. A different DFlash2 checkpoint changes that zone. There is no warning when p_min lands inside it.
- **Draft-model family flips the meaning.** With a DFlash **1** drafter (selector_top_k absent/0, is_dflash2 false) the same flag lands on speculative.cpp:1328, comparing the top vocabulary probability from common_sampler — no 1/16 floor, an entirely different scale. Build 10472's draft-dflash is DFlash 1 per scripts/probe-dflash2-load.ps1:10. Numbers for this flag do not transfer between the two builds.
- **--spec-draft-backend-sampling is irrelevant here.** The DFlash2 branch never calls common_sampler_sample; it reads the lattice from llama_get_embeddings_nextn (speculative.cpp:1221), and speculative.cpp:1187 passes `!is_dflash2` as the logits flag, so logits are not even produced. Only the dead else-arm at 1328 uses the sampler.
- **--spec-draft-n-max sets the block, p_min never does.** n_block_tokens = n_max + 1 always (speculative.cpp:1182-1184), so the draft-model decode at speculative.cpp:1195 costs the same regardless of p_min. Note also the clamp at speculative.cpp:991-997: block_size = 8 means n_max is silently clamped to 7; our 4 is under that.
- **Server-side per-request override does not exist in this build** (server-schema.cpp:198-227 is `#if 0`). CLI only.
- **Instrumentation gate:** the per-implementation `#gen drafts / #gen tokens / #mean acc len` line is SPC_TRC (speculative.cpp:2863) = LOG_TRC = verbosity level 4 (common/log.h:26,115). The existing probe script runs `-lv 3` and will NOT print it. The aggregate `draft acceptance = ... mean len = ...` line at server-context.cpp:634 is SLT_INF and does show at default verbosity, but it merges ngram-mod and dflash into one number and therefore cannot attribute a p_min effect.

### VRAM

No. Every use of p_min in the tree is a float comparison inside the draft loop (speculative.cpp:1254, 1268, and the dead 1328); nothing allocates, sizes a buffer, or feeds a context parameter from it. Buffer sizing for speculation comes from n_max via common_speculative_get_output_limits and common_speculative_n_max (speculative.cpp:2351-2361) — p_min is absent from both. Its only runtime effect is a shorter token vector handed to the target model, which cannot exceed what n_max already reserved. Second-order: because it does not move VRAM, it does not perturb --fit, so a p_min sweep is one of the few knobs here that does not itself shift the free-VRAM-at-boot confound.

### If it is measured

**Do not run a value ladder first. Run one liveness probe, because the code says most of the ladder is provably inert.**

Values, and why:
- **0.00 (baseline, keep).** The `if (params.p_min > 0.0f)` guard at speculative.cpp:1262 makes it a genuine zero-cost path.
- **0.0625 and below — DO NOT SWEEP.** 1/sum ≥ 1/selector_top_k = 1/16 by construction (speculative.cpp:1264-1268). 0.01/0.05 are mathematically identical to 0.00. Burning a round on them repeats the --spec-ngram-mod-n-min error exactly.
- **0.999 — run this ONE probe first, as a falsification run, not a candidate config.** If the flag is live at all, this should collapse the DFlash draft length. Read `#gen drafts` and `#gen tokens` for `draft-dflash` from the -lv 4 statistics line (speculative.cpp:2863) and compute mean generated length. Baseline is 4.00 (n_max = 4, no truncation possible at p_min = 0). Decision rule: if mean generated length at 0.999 is still ≈ 4.00, the selector's argmax softmax is essentially always ≥ 0.999, no smaller value can ever trip, and **the entire sweep is dead — stop and bank the GPU round.** If it drops materially, the knob is live and only then bisect **0.5, 0.8, 0.95** within a single round, paired and order-alternated.
- Expected outcome, stated in advance so a null result is not read as a failed measurement: **neutral to slightly negative.** The block decode at speculative.cpp:1195 happens before any check, so p_min saves no draft compute; its only saving is a narrower target verification batch. On one 27B IQ2_XXS sequence at -np 1, the target step is weights-bandwidth bound and a 3-token vs 5-token batch costs nearly the same wall time, while every truncated position is an accepted token thrown away. The code offers no mechanism by which this flag can produce a large win on this hardware.

Mandatory run conditions:
- **-lv 4**, not -lv 3. At -lv 3 the per-implementation statistics line is filtered out (common/log.h:26) and you cannot separate dflash from ngram-mod. Without that separation the run is unattributable.
- **temperature exactly 0.0 on every request.** A non-zero temp switches to speculative.cpp:1254, a different comparison against a randomly sampled index.
- **Leave --spec-draft-n-min unset (0).** Setting it turns truncation into whole-draft discard at speculative.cpp:1276.
- Same drafter GGUF throughout (selector_top_k = 16 sets the dead zone).
- Pair within a round and alternate order; the repo's ≥13.6 % noise floor applies.

Results that would be UNINTERPRETABLE — do not report these as findings:
1. Any comparison of p_min ∈ (0, 0.0625] against 0.00. The code proves identical behaviour; any delta is boot noise.
2. Any run where `#gen drafts` for draft-dflash is a small fraction of total draft steps (ngram-mod winning the chain at speculative.cpp:2724). p_min then applies to a minority of steps and a null tok/s result says nothing about the flag. Report the dflash-vs-ngram-mod draft-count split alongside every tok/s number or the number is not evidence.
3. Any run reporting only the aggregate `draft acceptance = ... mean len = ...` line (server-context.cpp:634). That merges both speculators.
4. Raw decode compared across boots (--fit follows free VRAM, 9,326–10,732 MiB).
5. Any comparison against a build/artifact where the drafter is DFlash 1 (selector_top_k = 0) — different code path at speculative.cpp:1328, different distribution, different scale.
6. Any run at a depth other than the one being reported. This project already has a spec knob (draft-mtp) that is +81 % at 16K and −71 % at 131,072 on the same artifact; a p_min verdict must name its context depth.

### Citations

- `C:\AI\llama.cpp\common\arg.cpp:4100-4106 — flag definition, writes params.speculative.draft.p_min`
- `C:\AI\llama.cpp\common\common.h:329 — `float p_min = 0.0f;` in common_params_speculative_draft`
- `C:\AI\llama.cpp\common\speculative.cpp:978 — `is_dflash2 = selector_top_k > 0;` (from GGUF dflash.selector_top_k = 16 on our drafter)`
- `C:\AI\llama.cpp\common\speculative.cpp:1195 — the single llama_decode of the whole draft block, BEFORE any p_min check`
- `C:\AI\llama.cpp\common\speculative.cpp:1219 — `if (is_dflash2)`: the branch our artifact takes`
- `C:\AI\llama.cpp\common\speculative.cpp:1236 — `scores = row + selector_top_k + predecessor*selector_top_k` (16 lattice scores)`
- `C:\AI\llama.cpp\common\speculative.cpp:1238 — `if (dp.temperature > 0.0f)`: the branch fork; false for us`
- `C:\AI\llama.cpp\common\speculative.cpp:1254 — stochastic check `dist.probs[predecessor] < params.p_min` (DEAD at temp 0)`
- `C:\AI\llama.cpp\common\speculative.cpp:1262 — `if (params.p_min > 0.0f)` guard: p_min=0 skips the check entirely`
- `C:\AI\llama.cpp\common\speculative.cpp:1264-1270 — `sum = Σ exp(scores[k]-scores[argmax])`; `if (1.0f/sum < params.p_min) break;` THE LIVE LINE`
- `C:\AI\llama.cpp\common\speculative.cpp:1272 — `result.push_back((llama_token) row[predecessor]);``
- `C:\AI\llama.cpp\common\speculative.cpp:1276-1281 — truncated draft below n_min is cleared entirely`
- `C:\AI\llama.cpp\common\speculative.cpp:1328 — the vocabulary-softmax p_min check; DEAD CODE for a DFlash2 artifact (else-arm of is_dflash2)`
- `C:\AI\llama.cpp\common\speculative.cpp:1887-2060 — common_speculative_impl_ngram_mod: p_min appears NOWHERE in it`
- `C:\AI\llama.cpp\common\speculative.cpp:1994-1996 — ngram-mod's only stopping rule: `token == EMPTY` then its own n_min`
- `C:\AI\llama.cpp\common\speculative.cpp:2542-2551 — speculator priority list: NGRAM_MOD is registered BEFORE DRAFT_DFLASH`
- `C:\AI\llama.cpp\common\speculative.cpp:2711-2726 — the chaining loop: first impl with a non-empty result sets `dp.drafting = false`, suppressing all later impls`
- `C:\AI\llama.cpp\common\speculative.cpp:1169-1174 — dflash draft() skips non-drafting seqs; batch.n_tokens==0 returns at 1191 without any decode`
- `C:\AI\llama.cpp\tools\server\server-context.cpp:2944 — `.temperature = slot.task->params.sampling.temp` (the request's temp drives the branch fork)`
- `C:\AI\llama.cpp\tools\server\server-schema.cpp:198-227 — the per-request `speculative.p_min` field is inside `#if 0`; NOT settable per request in this build`
- `C:\AI\llama.cpp\tools\server\server-context.cpp:634-641 — the SLT_INF acceptance line and common_speculative_print_stats call`
- `C:\AI\llama.cpp\common\log.h:24-26,114-115 — LOG_TRC needs verbosity >= 4; the per-impl statistics line is invisible at -lv 3`

---

## `--spec-ngram-mod-n-match`

**Default.** 24 (common.h:352 `int32_t n_match = 24;`, echoed by the staged binary's --help: "ngram-mod lookup length (default: 24)"). Also 24 in the --spec-default preset, arg.cpp:4632. Validated range 1..1024, arg.cpp:4186-4188. We run 12.

### What it actually does

It is the HASH KEY WIDTH of the ngram-mod table, and nothing else. It is passed once into the container constructor -- `mod(params.ngram_mod.n_match, 4*1024*1024)` (speculative.cpp:1914) -- and lands in `common_ngram_mod::n` (ngram-mod.h:33), where it is used in exactly one place: the loop bound of `idx()` (ngram-mod.cpp:15-25), which folds n tokens through an LCG (`res = res*6364136223846793005 + tokens[i]`) and then takes `res % entries.size()`. Since entries.size() is 4*1024*1024 = 2^22, that modulo is "keep the low 22 bits".

The table is a bare direct-mapped array of successors with NO stored key and NO verification. `add()` (ngram-mod.cpp:27-35) computes the slot from tokens[0..n-1] and writes `tokens[n]` -- the single token that followed that window -- unconditionally, last-writer-wins. `get()` (ngram-mod.cpp:37-41) returns `entries[i]` blind. A collision therefore returns a plausible but wrong successor with no signal.

YES -- n_match governs how often ngram-mod drafts at all, and at i = 0 it is the SOLE governor. In `draft_one` the key for the first lookup is built at speculative.cpp:1986-1990: result[0..n-2] = the last n-1 prompt tokens, result[n-1] = `dparams.id_last` (the token just sampled). So the i=0 key is exactly "the last n_match tokens of context". `mod.get(result.data() + 0)` (1993) succeeds only if that exact n_match-token window was seen before. Shorter key = strictly weaker requirement: every 24-token window repeat is also a 12-token window repeat, never the reverse. So the i=0 hit rate is monotonically non-decreasing as n_match falls. That is the fire-rate knob, and the measured 93.7 % decline-at-i=0 is a statement about n_match and about nothing else.

What n_match buys in fire rate it pays for in key collapse: with a shorter window more genuinely different contexts share the same key, and last-writer-wins hands back the successor of whichever occurrence was most recent. That wrong token is rejected by the target, and -- because n_min=16 forces every draft that fires to be at least 16 tokens -- a bad first token means a ~0 acceptance fraction, which feeds the reset loop at speculative.cpp:2042-2058 (five drafts with f_acc < 0.25 and no good one in between => `mod.reset()`, the whole shared table wiped).

### The plausible-but-wrong reading

Six wrong readings are available here. In rough order of how much GPU time each would waste:

(1) "More ngram-mod fires = faster." This is the trap that makes the whole sweep uninterpretable if unguarded. ngram-mod is registered ABOVE draft-dflash (speculative.cpp:2545 vs 2551, priority comment at 2540-2541) and the cascade stops at the first non-empty draft (2725-2726, 2753-2755); docs/speculative.md:207 states it plainly. Lowering n_match raises ngram-mod's fire rate BY SUPPRESSING dflash calls. An arm can show ngram-mod firing twice as often and decode slower. Refuted by: speculative.cpp:2545 sitting six lines above 2551, and the `if (n_drafting == 0) break;` at 2753-2755.

(2) "n_min governs the i=0 decline." This is precisely the misreading that wasted the earlier --spec-ngram-mod-n-min sweep, and it is still wrong even after the i-counts-draft-tokens correction. Trace both branches at i=0: with n_min>0 you take `result.clear(); return;` (speculative.cpp:1996-1997) -- empty draft. With n_min=0 the guard `i < params.n_min` is false, you take `result.resize(n + 0)` and break (2000-2001), and then the trim loop `for (i = 0; n + i < result.size(); ++i)` at 2007 runs ZERO times because result.size() == n, and 2010 resizes to n - n = 0 -- empty draft again. n_min cannot change the i=0 outcome at any value. The first-successor miss is governed by n_match alone, because n_match alone determines the key at speculative.cpp:1986-1990 and 1993.

(3) "n_match tunes recency -- shorter means more recent matches." Half of this is a claim the code never offers. Recency is UNCONDITIONAL at every n_match: add() overwrites the slot on every write (ngram-mod.cpp:30-34), so the stored successor is always from the most recent occurrence, at 24 exactly as at 12. n_match changes key SPECIFICITY only. If the scan's argument is "shorter and more recent predicts better", the code says you are buying only the "shorter" half, and paying for it with key collapse -- more distinct contexts folding onto one slot, each stealing the others' successor.

(4) "--spec-draft-n-max 4 caps the ngram-mod draft to 4 tokens." False, and it would make the whole config look broken. The truncation at speculative.cpp:2728-2732 uses `dp.n_max`, which server-context.cpp:2936-2938 fills from slot.get_n_draft_max() = `n_ctx - prompt.n_tokens() - 2` (server-context.cpp:451) -- a remaining-context limit, thousands of tokens wide. `speculative.draft.n_max` is consumed only by the draft-model branch of common_speculative_n_max (2361) and by the dflash impl. ngram-mod drafts up to ngram_mod.n_max = 32, untouched by our --spec-draft-n-max 4.

(5) "The n_match<16 warning means output quality degrades, so 12 is risky." No. Verification runs through common_sampler_sample_and_accept_n (server-context.cpp:3830-3831); a wrong draft token is rejected and the target's own token is kept. At temperature 0.0 the output is bit-identical to no speculation. "Poor quality" at speculative.cpp:1925 means poor DRAFT quality -- wasted verification passes -- a throughput claim, not a correctness one.

(6) "Lookup length = how much history is searched." It is a hash key width into a keyless table. get() (ngram-mod.cpp:37-41) returns entries[idx] with no key comparison at all, so a hash collision hands back an unrelated token that looks like a legitimate prediction. Compounding it, `res % entries.size()` with entries.size() = 2^22 keeps the LOW 22 bits of an LCG, which are its weakest -- bit 0 of the hash is just the XOR of the low bits of the n tokens, regardless of n. At our context depths occupancy is well under 1 % so hash collisions stay negligible; the dominant aliasing at n_match=12 is semantic (two real passages genuinely sharing a 12-token suffix), not hash. Keep the two apart when reading results, and read the actual occupancy off the trace line at speculative.cpp:1950 rather than assuming it.

### Interactions

DISABLED BY: nothing. It is live whenever `ngram-mod` appears in --spec-type. There is no CLI knob for the 4*1024*1024 table size, so n_match is the only lever over the collision budget.

FIXED AT CONSTRUCTION (speculative.cpp:1914): it cannot be changed on a running server. Every arm needs a restart, which drags in the boot-VRAM problem.

PRE-EMPTS draft-dflash. This is the interaction that decides the sweep. add_config_if_enabled registers NGRAM_MOD at speculative.cpp:2545 and DRAFT_DFLASH at 2551, and that list -- not the command-line order -- "defines the priority of the speculators" (comment at 2540-2541). common_speculative_draft (2710-2756) walks impls in order and clears `dp.drafting` on the first non-empty result (2725-2726), breaking out at 2753-2755. docs/speculative.md:207 says it outright. So ngram-mod is tried FIRST on every token, and every extra time it fires is one time draft-dflash does not run. Lowering n_match hands more calls to the weaker speculator.

SILENTLY CHANGES the blind window on freshly generated text. draft_one only flushes new n-grams when `sinfo.i_last + 32 < cur_len` (1978) and only up to `cur_len - n` (1979). So text within roughly the last n_match+32 tokens is not yet in the table. n_match=12 shrinks that window from ~56 to ~44 tokens -- a second, smaller way a short key raises the fire rate on self-repetition. Prompt text is exempt: begin() (1943-1947) adds every prompt window at once.

DELAYS the occupancy reset. begin() re-adds the whole prompt on every request and never clears; the table is only wiped if used/size > 0.25, i.e. above 1,048,576 distinct keys (1949-1957). A shorter key yields fewer distinct keys for the same corpus, so n_match=12 reaches that threshold later than 24 -- 12 is more reset-resistant, not less.

FEEDS the low-acceptance reset. accept() (2042-2058) counts drafts with n_accepted/n_draft_last < 0.25; five in a row (with no good draft between) call mod.reset(), wiping the table for ALL sequences and setting i_last = 0. Because n_min=16 forces every firing draft to be >= 16 tokens, one wrong first token is enough to score a low fraction. Too short an n_match therefore drives a thrash loop. Note the loop is correctly insulated when dflash produced the draft: accept arrives with is_other=true and returns at 2035.

n_min / n_max GATE ONLY THE RESIDUAL. With 93.7 % of calls dying at i=0, --spec-ngram-mod-n-min 16 and --spec-ngram-mod-n-max 32 only shape the surviving ~6.3 %.

SHARED AND PERSISTENT: one table across all slots and all requests (speculative.cpp:1891, docs/speculative.md:176). Under -np 1 the cross-slot part is moot, but the cross-request warming is not -- the first request against a fresh server always under-fires.

EMITS A STARTUP WARNING at our value: speculative.cpp:1924-1927 prints "ngram_mod n_match=12 is too small - poor quality is possible" via SPC_WRN -> LOG_WRN, which is above the default threshold and will always appear. Expected, not an error.

NOT clamped by --spec-draft-n-max (see misreading_risk).

### VRAM

No -- zero VRAM, and zero host-RAM variation either. The table is `std::vector<int32_t>` sized `4*1024*1024` in the constructor (speculative.cpp:1914 -> ngram-mod.cpp:9-13), i.e. 16 MiB of host memory, allocated once and entirely independent of n_match; `size_bytes()` (ngram-mod.cpp:60-62) confirms it is entries.size()*4 with no n term. n_match only sets a loop bound in idx() (ngram-mod.cpp:18), so its whole cost is n_match multiply-adds per lookup on the CPU -- at most n_match*n_max = 12*32 = 384 vs 24*32 = 768 integer ops per draft call, unmeasurable next to a decode step.

The ngram-mod knob that DOES touch allocation is n_max, not n_match: common_speculative_n_max (speculative.cpp:2372-2373) takes ngram_mod.n_max = 32, and common_speculative_get_output_limits (speculative.cpp:2512-2521, called at server-context.cpp:49) sizes the per-seq output budget as min(n_batch, 1 + 32) = 33.

This is operationally good news for the sweep: n_match arms do not change any allocation, so --fit sees the same picture in every arm. The only VRAM variance is the boot-to-boot free-VRAM drift, which a restart between arms reintroduces.

### If it is measured

VALUES: four arms -- 24 (the real default, baseline), 16 (the author's own warning boundary, speculative.cpp:1924), 12 (current), 8. Do not go below 6: at that width the key collapses, wrong successors drive f_acc < 0.25, and the reset loop at speculative.cpp:2044-2054 wipes the shared table repeatedly -- you would be measuring the reset loop, not the flag. Do not bother with 32/48: the i=0 hit rate is monotonically non-increasing in n_match, so above the default you are only trading fire rate away for a stricter key, and the 93.7 % decline is already the binding constraint.

HOLD FIXED across all arms: --spec-ngram-mod-n-min 16, --spec-ngram-mod-n-max 32, --spec-draft-n-max 4, -ctk/-ctv q4_0, -np 1, temperature 0.0, the same prompt corpus, the same context depth. n_min in particular must not move in the same round -- it gates the residual ~6.3 % and would confound the arm.

PROTOCOL: n_match is fixed in the constructor (speculative.cpp:1914), so each arm needs its own server process. Run -lv 4 so SPC_TRC is above threshold (LOG_TRC needs verbosity >= 4, log.h:25/115) -- that is what surfaces the per-impl statistics line at speculative.cpp:2863 and the occupancy line at 1950. Those counters are CUMULATIVE over the server's life, so either diff them per request or read them once at the end of a fixed corpus. Run at least 5 identical requests per arm and DISCARD THE FIRST: the table persists across requests and begin() only re-adds (1943-1947), so a fresh server always under-fires on request one. Alternate arm order across rounds and pair within a round -- n_match itself moves no allocation, but the restart it forces re-rolls free VRAM and --fit follows, so the repo's 13.6 % noise floor applies to the tok/s column.

METRICS REQUIRED PER ARM -- all three, or the arm is not readable:
  a. ngram-mod fire rate = #gen drafts / #calls(g) from the ngram-mod statistics line (speculative.cpp:2863-2871). This is the direct measurement of the i=0 miss and is insensitive to VRAM drift.
  b. draft-dflash #gen drafts from the SAME log block. This is the pre-emption cost.
  c. end-to-end decode tok/s, plus "draft acceptance" and "mean len" from server-context.cpp:634-636.

WHAT WOULD BE UNINTERPRETABLE:
  1. Any arm whose log contains "ngram_mod occupancy ... exceeds threshold (0.25) - resetting" (speculative.cpp:1954) mid-corpus. The shared table was wiped part-way through; requests before and after are not the same experiment. Discard the arm and shorten the corpus or restart between arms.
  2. Any arm with more than an occasional "low acceptance streak (%d) - resetting ngram_mod" (speculative.cpp:2048). You measured the thrash loop, not the key width. Expect this to appear first at n_match=8, and treat its onset as the floor rather than as a data point.
  3. An ngram-mod fire-rate number reported WITHOUT dflash's draft count beside it. A fire-rate win bought by suppressing dflash (speculative.cpp:2545 above 2551) can be a throughput loss, and the ngram-mod line alone cannot tell you which happened. This single omission is what would repeat today's two wasted sweeps.
  4. Any tok/s comparison across arms that is not paired within a round with the order alternated. The restart per arm re-rolls boot VRAM.
  5. A single-request arm, or an arm that includes its first request. Cold table.
  6. Any comparison against numbers taken at a different context depth. This project has already been bitten (draft-mtp: +81 % at 16K, -71 % at 131,072); n_match's value is a function of how much repeated text is in the window, so it is depth-bound by construction.

EXPECTED SHAPE, stated as a hypothesis so the sweep can falsify it: fire rate rises monotonically 24 -> 8; mean accepted length falls; dflash's share falls; tok/s is non-monotonic with an interior optimum. If tok/s turns out FLAT across 24/16/12/8 within the 13.6 % noise floor, the honest conclusion is that ngram-mod contributes too little at this workload for n_match to matter, and the next question is whether ngram-mod belongs in --spec-type at all -- not a finer n_match grid.

### Citations

- `C:/AI/llama.cpp/common/common.h:352 -- the real default, n_match = 24 (n_min = 48, n_max = 64 alongside it)`
- `C:/AI/llama.cpp/common/arg.cpp:4183-4191 -- the flag, help string, and the 1..1024 validation`
- `C:/AI/llama.cpp/common/speculative.cpp:1914 -- the ONLY consumer of the value: mod(n_match, 4*1024*1024). Fixed at construction; cannot be changed without a server restart`
- `C:/AI/llama.cpp/common/ngram-mod.cpp:15-25 -- idx(): LCG polynomial hash over exactly n_match tokens, then % 2^22`
- `C:/AI/llama.cpp/common/ngram-mod.cpp:27-35 -- add(): stores tokens[n] (the single next token), last-writer-wins, no key stored`
- `C:/AI/llama.cpp/common/ngram-mod.cpp:37-41 -- get(): returns entries[i] with NO key comparison. A collision returns a wrong successor silently`
- `C:/AI/llama.cpp/common/speculative.cpp:1986-1990 -- the i=0 key is built here: last n_match-1 prompt tokens + id_last`
- `C:/AI/llama.cpp/common/speculative.cpp:1992-2004 -- the draft chain; i counts draft tokens produced, and i=0 is the first-successor lookup`
- `C:/AI/llama.cpp/common/speculative.cpp:1995-1998 -- the n_min gate, which at i=0 is a no-op (see misreading_risk)`
- `C:/AI/llama.cpp/common/speculative.cpp:1978-1984 -- new n-grams added only in chunks of >32, and only up to cur_len-n: the blind window is ~n_match+32 tokens`
- `C:/AI/llama.cpp/common/speculative.cpp:1943-1957 -- begin() re-adds the whole prompt WITHOUT clearing; occupancy reset only above 0.25 (1,048,576 distinct keys)`
- `C:/AI/llama.cpp/common/speculative.cpp:1924-1927 -- SPC_WRN fires at startup for n_match < 16 (LOG_WRN, always visible). Expected at our 12`
- `C:/AI/llama.cpp/common/speculative.cpp:2042-2058 -- the low-acceptance reset: 5 drafts with f_acc < 0.25 wipes the shared table`
- `C:/AI/llama.cpp/common/speculative.cpp:2542-2552 -- priority list: NGRAM_MOD (2545) is registered ABOVE DRAFT_DFLASH (2551)`
- `C:/AI/llama.cpp/common/speculative.cpp:2710-2756 -- the cascade stops at the first impl that returns a non-empty draft (2725-2726, 2753-2755)`
- `C:/AI/llama.cpp/common/speculative.cpp:2728-2732 -- the draft truncation site; dp.n_max is NOT --spec-draft-n-max`
- `C:/AI/llama.cpp/tools/server/server-context.cpp:441-459 -- get_n_draft_max() = n_ctx - prompt.n_tokens() - 2, the context-space limit that becomes dp.n_max`
- `C:/AI/llama.cpp/common/speculative.cpp:2351-2385 -- common_speculative_n_max: ngram_mod contributes n_max (32), not n_match`
- `C:/AI/llama.cpp/common/speculative.cpp:2512-2521 + tools/server/server-context.cpp:49 -- output/logits budget = min(n_batch, 1+n_max). The only ngram-mod path that touches allocation, and n_match is absent from it`
- `C:/AI/llama.cpp/common/speculative.cpp:2829-2873 + tools/server/server-context.cpp:641 -- per-impl #calls / #gen drafts / #acc tokens, printed at request end via SPC_TRC (needs -lv 4)`
- `C:/AI/llama.cpp/tools/server/server-context.cpp:3830-3831 -- common_sampler_sample_and_accept_n: verification is exact, so a wrong draft costs throughput, not output`
- `C:/AI/llama.cpp/docs/speculative.md:207 -- "If a draft model is combined with a draftless decoding the draftless decoding has higher precedence."`

---

## `-ctkd / -ctvd (--spec-draft-type-k / --spec-draft-type-v, aliases --cache-type-k-draft / --cache-type-v-draft, env LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K/_V)`

> **DO NOT SPEND A GPU ROUND.** The code shows our current value is
> the only sensible one, or that the flag cannot act here at all.

**Default.** f16 for both, hard-coded at common/common.h:340-341 (`ggml_type cache_type_k = GGML_TYPE_F16;` inside struct common_params_speculative_draft). It is NOT "same as --cache-type-k" — the help string for the neighbouring draft flags says "same as ..." for CPU params (arg.cpp:4000, 4017) but -ctkd/-ctvd say "(default: f16)" and mean it. Confirmed against the staged binary's --help.

### What it actually does

It sets ggml_type on `common_params_speculative_draft::cache_type_k/_v` (arg.cpp:4032, 4045), which `common_base_params_to_speculative` copies UNCONDITIONALLY into the draft-context params at speculative.cpp:2405-2406 (`result.cache_type_k = params_spec.cache_type_k;` — note this line sits OUTSIDE the `if (has_draft)` block at 2393-2403), from where common.cpp:1727-1728 puts it into `cparams.type_k/type_v` of the llama_context created for the drafter.

The drafter genuinely has its own cache. speculative.cpp:2464-2482: with a draft model present it calls `llama_model_load_from_file(params.model.path, mparams)` (params.model was already re-pointed at the draft GGUF by speculative.cpp:2395) and then `llama_init_from_model(model_dft, cparams)` — a second llama_context, therefore a second llama_kv_cache, therefore the second "KV buffer size" line. The target's own -ctk/-ctv live in a different struct field (common.h:577-578 vs common.h:340-341) and there is no inheritance path between them anywhere in common/ — grep for cache_type_k in common/ returns only arg.cpp, common.cpp, common.h, speculative.cpp:2405. So -ctk q4_0 does not reach the drafter, and -ctkd does not reach the target. The 45.00 MiB line is the drafter's, sized by the draft context's n_ctx (inherited from the target's n_ctx via `common_params result = params;` at speculative.cpp:2391).

Now the part that decides whether the saving is real. The value lands in the CUDA flash-attention kernel selector:

1. Quantized V force-enables FA and DISABLES the support probe. llama-context.cpp:3602-3611: if type_v is quantized and flash_attn_type is AUTO, it is promoted to ENABLED. That makes `cparams.auto_fa` false (llama-context.cpp:229-230), so `resolve_fused_ops` skips the Flash-Attention probe entirely (llama-context.cpp:554-557). Consequence: for a quantized draft V you get neither the "Flash Attention enabled" line nor the "not supported, set to disabled" warning — the graph is built with an FA node and, if CUDA cannot take it, ggml_backend_sched silently places that node on the CPU backend.

2. Which CUDA FA kernel is chosen. RTX 4070 SUPER is Ada (cc 8.9), so `turing_mma_available` is true and fattn.cu:461-483 applies. With quantized K/V the VEC kernel — the only one that reads quantised K/V natively — is selected only when `Q->ne[1] <= 2` (fattn.cu:469). Otherwise line 482 returns BEST_FATTN_KERNEL_MMA_F16.

3. What Q->ne[1] is on the DFlash draft path. speculative.cpp:1183-1188: `n_block_tokens = n_draft + 1` for draft-dflash (the `+ 0` branch is dspark-only), all pushed into one batch and one `llama_decode(ctx_dft, batch)` at speculative.cpp:1196. With --spec-draft-n-max 4 that is 5 tokens. llama-graph.cpp:2532-2534 permutes q to (0,2,1,3), so Q->ne[1] == n_tokens == 5. 5 > 2, so the drafter always takes MMA_F16, never VEC.

4. MMA_F16 does not consume quantised KV — it dequantises it first. fattn.cu:550-555: for TILE and MMA_F16, `need_f16_K = true; need_f16_V = true`. fattn-common.cuh:68-71 then appends an f16 staging area of `ggml_nelements(K)*sizeof(half)` (and the same for V) to the FA node's allocation, which ggml-cuda.cu:906-911 charges into the compute buffer. fattn-common.cuh:1022-1030 and 1056-1065 run `to_fp16(K_data, K_f16, ggml_nelements(K), stream)` over the WHOLE K view — i.e. the entire KV span up to n_kv — on every FA call, every layer, every draft step. fattn-common.cuh:68 skips this only when `K->type == GGML_TYPE_F16`.

So: with -ctkd/-ctvd f16 the drafter's MMA kernel reads the cache directly, zero conversion. With any quantised value the drafter pays a full O(n_kv) dequantisation per layer per draft step, plus a compute-buffer scratch that gives back part of the KV saving. `llama_set_causal_attn(ctx_dft, false)` (speculative.cpp:1036) does not change any of this — causal_attn only fills the mask differently (llama-kv-cache.cpp:1717-1725), the op is still GGML_OP_FLASH_ATTN_EXT.

### The plausible-but-wrong reading

The plausible-but-wrong reading is: "the drafter is a small model doing tiny single-token steps, so it will take the quantised-KV vector kernel just like the target does, and I get 34 MiB for free."

Both halves are wrong, and each has its own refuting line.

(a) "single-token steps" — speculative.cpp:1183: `const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);`. draft-dflash is not dspark, so n_block_tokens = n_max + 1 = 5, and speculative.cpp:1196 decodes all five in ONE llama_decode. After llama-graph.cpp:2534 permutes q, Q->ne[1] == 5.

(b) "so it takes the VEC kernel" — fattn.cu:469: `if (Q->ne[1] <= 2) { return BEST_FATTN_KERNEL_VEC; }`, inside the `else` branch at 467 that handles quantised K/V on Ada. 5 > 2, so control reaches fattn.cu:482 `return BEST_FATTN_KERNEL_MMA_F16;`. And fattn.cu:553-554 sets `need_f16_K = true; need_f16_V = true` for MMA_F16 — the quantised cache is dequantised in full before every kernel launch (fattn-common.cuh:1029-1030). VEC is the ONLY kernel that reads quantised KV natively (fattn.cu:556-558).

This is exactly why the flag behaves differently on the drafter than on the target. The target at -np 1 decodes one token per step, hits fattn.cu:469-470 with Q->ne[1] = 1, gets VEC, and its q4_0 KV is genuinely free. The drafter, by construction, never can. The reasoning that justified `-ctk q4_0 -ctv q4_0` on the target does not transfer across the target/draft boundary — and the boundary is `Q->ne[1] <= 2`, which `--spec-draft-n-max` sits directly on top of.

Second-order trap worth naming: `--spec-draft-n-max 1` would put Q->ne[1] at 2 and DOES reach VEC. So "q4_0 draft KV was fine when I tried it" from any run with n_max <= 1 is not evidence for n_max 4.

Third: the help lists q4_1/q5_0/q5_1/iq4_nl as "allowed values". They are allowed by kv_cache_type_from_str and rejected by the CUDA FA kernel (fattn.cu:343-354), and llama-context.cpp:3602-3605 has already thrown away the probe that would have warned you. A sweep over the full "allowed values" list would measure a CPU attention fallback for four of the nine entries and report it as a KV-type result.

### Interactions

- Disables nothing directly, but a quantised V silently rewrites the FA policy: llama-context.cpp:3602-3605 promotes flash_attn AUTO -> ENABLED for the DRAFT context, which sets auto_fa false (229-230) and makes resolve_fused_ops skip the FA probe (554-557). You lose both the "Flash Attention enabled" confirmation and the "not supported, set to disabled" warning for the drafter.
- K and V must be the SAME type. fattn.cu:442-446: with GGML_CUDA_FA_ALL_QUANTS off (ggml/CMakeLists.txt:208, and this is a stock CUDA-13 prebuilt), K->type != V->type returns BEST_FATTN_KERNEL_NONE. So `-ctkd q4_0` alone, without `-ctvd q4_0`, kills flash attention on the draft path.
- The `allowed values` list in the help is the parser's list, not the kernel's. q4_1, q5_0, q5_1 return false at fattn.cu:343-348 and iq4_nl falls through to `default: return false` at 353-354. Combined with the forced-ENABLED FA above, choosing one of those puts the drafter's attention node on the CPU backend with no warning printed.
- Clamped by nothing on the value itself; clamped indirectly by llama-context.cpp:3613-3633, which rejects a KV type whose block size does not divide n_embd_head_k/v (q4_0 blck 32 divides 64/96/128, so this will not bite here).
- Silently changes the target's fitted configuration when --fit is on and -c is unset: server-context.cpp:1074 -> fit.cpp:310-344. This is the confound, not a feature.
- No interaction with the ngram-mod half of `--spec-type draft-dflash,ngram-mod`. ngram-mod owns no context and no KV (speculative.cpp:2612-2615 constructs it from params alone).
- `-ctkd`/`-ctvd` are the only two draft flags in that block without `.set_spec()` (arg.cpp:4034 vs 4004, 4014, 4021, 4053). That is purely help-grouping (arg.cpp:982) — it does not change parsing.

### VRAM

Yes, but far less than the 34 MiB the KV line suggests, and it can go the wrong way. Two opposing effects:

SAVES: the drafter's KV buffer. q4_0 is 4.5 bit/elem vs 16, so 45.00 MiB -> ~12.7 MiB, a ~32 MiB drop in the second "KV buffer size" line. q8_0 -> ~24 MiB, a ~21 MiB drop.

COSTS: because the drafter always lands on MMA_F16 (fattn.cu:482, since Q->ne[1] = n_max+1 = 5 > 2 at fattn.cu:469), fattn.cu:553-554 sets need_f16_K/V, and fattn-common.cuh:68-71 appends `ggml_nelements(K)*2` + `ggml_nelements(V)*2` bytes of f16 staging to the FA node, charged to the compute buffer by ggml-cuda.cu:906-911. That staging is per-layer-view sized (the graph allocator reuses it across layers, so roughly max-over-layers, not sum), i.e. roughly (45 MiB / 2 / n_layer_draft) for K plus the same for V. For a small DFlash drafter that is single-digit to low-double-digit MiB clawed straight back.

Net is positive but I would predict on the order of 15-25 MiB for q4_0, not 34. This is directly checkable: at load, the draft "KV buffer size" line must fall AND the draft "compute buffer size" line must rise. If the compute line does not move, my MMA-path reading is wrong and the whole analysis should be re-opened.

Whether the net saving buys anything at all: server-context.cpp:1068-1074 adds the drafter's measured model+context+compute to `fit_params_target`, so the freed bytes become target --fit budget. fit.cpp:310 only re-fits n_ctx when `cparams->n_ctx == 0`, and fit.cpp:344 rounds to a multiple of 256. With -c pinned, the freed bytes are simply not spent — the flag then costs a dequant pass and buys literally nothing.

### If it is measured

Do not sweep this. If you sweep it anyway, here is how not to walk into the traps.

Values worth measuring — only two, and both K and V must move together:
- `-ctkd q8_0 -ctvd q8_0`
- `-ctkd q4_0 -ctvd q4_0`
Both take MMA_F16 with a full dequant per layer per draft step. The only defensible readout is the load-time VRAM pair, not throughput: the draft "KV buffer size" line must drop (~21 MiB for q8_0, ~32 MiB for q4_0) AND the draft "compute buffer size" line must rise. Record both numbers from the same log. If the compute line does not move, my MMA reading is wrong and you should stop and re-derive before drawing any conclusion.

Values that must NOT be swept, because the result is uninterpretable rather than merely bad:
- `bf16` — same 2 bytes as f16, and fattn-common.cuh:68 fires because K->type != GGML_TYPE_F16, so it buys zero VRAM and adds a conversion. Strictly dominated by the default; a null result here proves nothing.
- `q4_1`, `q5_0`, `q5_1`, `iq4_nl` — fattn.cu:343-354 returns unsupported in this build (GGML_CUDA_FA_ALL_QUANTS is OFF, ggml/CMakeLists.txt:208), and llama-context.cpp:3602-3605 has suppressed the warning. The FA node lands on the CPU backend. Any number you get measures a host round-trip of the draft KV, not a cache type.
- Any mixed pair (`-ctkd q4_0` without `-ctvd`, or vice versa) — fattn.cu:443-445 returns BEST_FATTN_KERNEL_NONE for K->type != V->type. Same CPU-fallback trap, same missing warning.

What makes a run uninterpretable regardless of value:
1. `-c` left unset while `--fit` is on. server-context.cpp:1074 hands the freed bytes to `fit_params_target`, and fit.cpp:310-344 spends them on the target's n_ctx (rounded to 256). The arms then differ in target context depth, and CLAUDE.md's own rule applies — a verdict at one depth does not transfer to another. Pin `-c` and `--fit-target` byte-identically across arms.
2. Reading the result off raw decode t/s. Even the optimistic net saving is ~20-30 MiB against a 6.77 GB weight set on a 12 GB card, which is under one offloaded layer — and the repo's own noise floor is 13.6% across boots. A t/s delta from this flag is unmeasurable by construction; only the VRAM lines and, if you want the cost side, a paired same-boot draft-step latency comparison carry signal.
3. Any run whose `--spec-draft-n-max` is not 4. n_max is what puts Q->ne[1] on the wrong side of fattn.cu:469, so a result at n_max 1 (Q->ne[1] = 2, VEC, quants free) says nothing about n_max 4.

Cheaper alternative that answers the same question with no GPU round: start the server once with `-ctkd q4_0 -ctvd q4_0` and once without, and diff the two draft buffer lines in the load log. That gives you the true net VRAM number — which is the only thing this flag can deliver — without a benchmark.

### Citations

- `C:\AI\llama.cpp\common\arg.cpp:4022-4034 (-ctkd definition; note no .set_spec()/.set_examples(), unlike every neighbouring draft flag — cosmetic, affects only help grouping via arg.cpp:982)`
- `C:\AI\llama.cpp\common\arg.cpp:4035-4047 (-ctvd definition)`
- `C:\AI\llama.cpp\common\common.h:340-341 (the f16 default lives here)`
- `C:\AI\llama.cpp\common\common.h:577-578 (the TARGET's cache_type_k/v — a separate field; no code copies one into the other)`
- `C:\AI\llama.cpp\common\speculative.cpp:2405-2406 (unconditional copy into the draft context params, outside the has_draft guard)`
- `C:\AI\llama.cpp\common\common.cpp:1727-1728 (-> cparams.type_k / type_v)`
- `C:\AI\llama.cpp\common\speculative.cpp:2464-2482 (separate model + separate llama_context => the drafter's own KV cache => the second 'KV buffer size' log line)`
- `C:\AI\llama.cpp\common\speculative.cpp:1183-1196 (DFlash draft decodes n_max+1 = 5 tokens in one batch)`
- `C:\AI\llama.cpp\common\speculative.cpp:990-996 (n_max is clamped to block_size-1 for draft-dflash — 4 survives a block_size of 16)`
- `C:\AI\llama.cpp\common\speculative.cpp:1036 (llama_set_causal_attn(ctx_dft,false) — mask only, does not disable FA)`
- `C:\AI\llama.cpp\src\llama-graph.cpp:2532-2557 (q permuted so Q->ne[1] == n_tokens; ggml_flash_attn_ext built whenever cparams.flash_attn)`
- `C:\AI\llama.cpp\src\llama-context.cpp:3602-3611 (quantised V force-promotes flash_attn AUTO -> ENABLED)`
- `C:\AI\llama.cpp\src\llama-context.cpp:229-230 and 554-557 (auto_fa becomes false, so the FA support probe and its warning are skipped)`
- `C:\AI\llama.cpp\src\llama-context.cpp:3613-3633 (block size of the KV type must divide n_embd_head_k/v — q4_0 blck 32 is fine for head dims 64/96/128)`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\fattn.cu:338-356 (FA-supported KV types: f32/f16/bf16/q4_0/q8_0 only; q4_1/q5_0/q5_1 return false and iq4_nl hits default:false unless GGML_CUDA_FA_ALL_QUANTS)`
- `C:\AI\llama.cpp\ggml\CMakeLists.txt:208 (GGML_CUDA_FA_ALL_QUANTS defaults OFF)`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\fattn.cu:442-446 (K->type != V->type => BEST_FATTN_KERNEL_NONE)`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\fattn.cu:461-483 (Ada: quantised KV takes VEC only if Q->ne[1] <= 2, else MMA_F16)`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\fattn.cu:550-559 (MMA_F16/TILE set need_f16_K = need_f16_V = true; only VEC reads quants natively)`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\fattn-common.cuh:53-71 (extra f16 staging appended to the FA node's allocation, skipped only when K->type == F16)`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\fattn-common.cuh:1022-1030, 1056-1065 (to_fp16 over the whole K and V view on every FA call)`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:906-911 (that staging is charged to the compute buffer)`
- `C:\AI\llama.cpp\tools\server\server-context.cpp:1033-1087 (--fit measures the draft's model+context+compute with THESE cparams and adds the total to fit_params_target)`
- `C:\AI\llama.cpp\common\fit.cpp:310 (n_ctx is auto-fitted only when the user left -c unset), fit.cpp:344 (rounded down to a multiple of 256), fit.cpp:377 (-ngl set by user aborts the fit)`

---

## `GGML_CUDA_GRAPH_OPT (environment variable, undocumented — no help text, no mention anywhere in docs/ or *.md)`

> **DO NOT SPEND A GPU ROUND.** The code shows our current value is
> the only sensible one, or that the flag cannot act here at all.

**Default.** Unset = disabled. Enabled only when `getenv` returns non-null AND `atoi(env) == 1` (ggml-cuda.cu:4330-4334). Read into a function-local `static`, so it is latched once per process on the first `graph_optimize` call.

### What it actually does

It is NOT a CUDA-graph tuning knob. It enables a multi-stream fork/join rewrite of the Q/K/V projection region inside `ggml_backend_cuda_graph_optimize`. When set to exactly 1, that function scans the split graph for a node whose output is consumed by exactly 3 other nodes and whose name contains "attn_norm", builds a `ggml_cuda_concurrent_event` (one `fork_event`, N `join_events`, a node→stream map), assigns each of the 3 branches to CUDA stream 1/2/3, and physically reorders `cgraph->nodes` to interleave the branches so ggml-alloc does not recycle the branch tensors (ggml-cuda.cu:4526-4529). At execute time `ggml_cuda_graph_evaluate_and_capture` records the fork event on stream 0, makes streams 1-3 wait on it, runs each branch on its own stream by flipping `cuda_ctx->curr_stream_no`, and joins at the join node (ggml-cuda.cu:4009-4026, 4102-4126). Net effect: the three QKV GEMMs run concurrently instead of serially. The only connection to CUDA graphs is a gate on `graph->is_enabled()` and the fact that a captured graph will replay the recorded concurrency.

Three gates the scan did not mention, each of which independently kills it on our model:
1. `ggml_nrows(node) <= 1` on the CONSUMER node (ggml-cuda.cu:4381). `ggml_nrows` = ne1*ne2*ne3 (ggml.c:1291-1295). For `Qcur_full = mul_mat(wq, cur)` that is exactly `n_tokens`. So `fan_out` is populated only when the ubatch is a single token.
2. fan-out must be exactly 3 — `min_fan_out = max_fan_out = 3` (ggml-cuda.cu:4396-4397, tested at 4403). In qwen35 the 48 recurrent (gated-delta-net) layers feed attn_norm into wqkv, wqkv_gate, ssm_beta and ssm_alpha = 4 consumers (qwen35.cpp:237, 241, 362, 369) → excluded even at n_tokens==1. Only the 16 full-attention layers (`full_attention_interval = 4`, log line 62) have fan-out 3 (qwen35.cpp:270, 282, 285), and only because this GGUF stores split attn_q/attn_k/attn_v rather than a fused attn_qkv for those layers (log lines 308-311 show the fused attempt falling through to the q/k/v branch of `create_tensor_qkv`, llama-model.cpp:2955-2965).
3. the fork→join span must contain the branch nodes and nothing else, or the fork is dropped (ggml-cuda.cu:4501-4508).

For our config the first gate alone is fatal. `--spec-draft-n-max 4` with `p_min` defaulting to 0.0 (common.h:329) means the DFlash2 selector never truncates, so it always emits 4 draft tokens; the draft model decodes a block of `n_draft + 1 = 5` (speculative.cpp:1178-1185) and the target verifies 5. `--spec-draft-n-min` also defaults to 0 (common.h:326), so speculation is never skipped for being too short. Every generation-time ubatch on both models has n_tokens >= 2, so `fan_out` is never populated, `concurrent_node_ranges` stays empty, and the function does nothing but `stream_context.reset()`. The scan's warning is CONFIRMED, and it is stronger than the scan states: this is not "would not benefit", it is "cannot execute a single line of the optimization".

### The plausible-but-wrong reading

Four wrong readings, the first two of which are exactly the shape that burned `--spec-ngram-mod-n-min` and `--fit-target 768`:

1. "GRAPH_OPT optimizes CUDA graphs — it should help whenever CUDA graphs are on, and our logs show CUDA graphs are on (`CUDA Graph id 57 reused`, log:4774)." REFUTED by the body of the function: ggml-cuda.cu:4339-4551 contains no `cudaGraph*` call at all. It builds `cudaEvent_t fork_event` / `join_events` and a stream map. The observation that CUDA graphs are working tells you nothing about whether this flag does anything, because the actual gate is ggml-cuda.cu:4381.

2. "Set it to `on` / `true` / `enabled` / `2`." REFUTED by ggml-cuda.cu:4331-4333: `atoi(env) == 1`. `atoi("true")` is 0, `atoi("2")` is 2 — both leave `enable_graph_optimization` false and the function returns at 4336. A sweep that mis-spells the value measures the OFF path in both arms, gets "no difference", and concludes correctly for the wrong reason — which is worse than a wrong number, because it validates a broken instrument.

3. "It is read per-decode, so I can flip it between runs of the same server / via a preset." REFUTED by the function-local `static` at ggml-cuda.cu:4330: the lambda runs once, on the first `graph_optimize` call of the process. It must be in the environment before `llama-server.exe` starts, and a server started without it can never pick it up.

4. "single-row decode-shaped nodes" reads as "it only helps at batch 1, and we are batch 1 because `-np 1`." REFUTED by speculative.cpp:1178-1185 and common.h:329: `-np 1` is the number of parallel SLOTS, not the ubatch width. With `--spec-draft-n-max 4` and `p_min = 0.0`, the draft model decodes a 5-token block and the target verifies 5 tokens, so `ggml_nrows(Qcur_full) = n_tokens = 5` on every single generation step of both models. The condition at 4381 is on the consumer node's row count, which for the QKV projections IS the ubatch width.

Bonus correction to the scan itself: "needs CUDA graphs plus exactly one CUDA device" is half right. The device-count half is exact (4344). The CUDA-graph half is only `is_enabled()` — architecture >= Volta and `GGML_CUDA_DISABLE_GRAPHS` unset (common.cuh:1257-1260) — not successful capture. Believing it needs capture would lead someone to try "isolate the effect by disabling graphs", which silently disables the flag too.

### Interactions

- `GGML_CUDA_DISABLE_GRAPHS` disables it. graph_optimize gates on `ggml_cuda_graph_set_enabled` → `is_enabled()` (common.cuh:1257-1260), which is false when that variable is set. So you cannot use DISABLE_GRAPHS as a control arm — it turns off the thing you are measuring.
- More than one CUDA device disables it outright (ggml-cuda.cu:4344, `ggml_backend_cuda_get_device_count() != 1`). A second CUDA device appearing in the machine silently zeroes the flag.
- Anything that raises the target ubatch above 1 token disables it: speculative decoding (our case), n_parallel batching, prompt processing. `-np 1` does not help — the draft block itself is 5 tokens.
- It silently REORDERS `cgraph->nodes` at split time. The compute path then tries to restore the original order inside each concurrent region so that op fusion still works (ggml-cuda.cu:4042-4101), but that restore bails out with `continue` if the nodes are not all found or not contiguous (4066-4083). A failed restore leaves the graph interleaved and fusion lost — a plausible mechanism for the flag being a net SLOWDOWN even where it fires.
- It does NOT require a CUDA graph to have been captured. The fork/join stream execution runs on the direct-eval path too (ggml-cuda.cu:4030-4126 is inside `if (!use_cuda_graph || cuda_graph_update_required)`), so a MUL_MAT_ID-driven graph-incompatibility (ggml-cuda.cu:2553-2564) does not stop it.
- Latent null-deref: `fan_out[src] += 1` at 4381 also counts `src == nullptr` slots (GGML_MAX_SRC = 10, ggml.h:224), and 4408 dereferences `root_node->name` without a null check. Safe only because the null count is normally far outside [3,3].

### VRAM

Yes if it fires, no in our config (it cannot fire). Two mechanisms, both real and both invisible to `--fit`: (1) the interleave at ggml-cuda.cu:4526-4529 exists specifically to extend branch-tensor lifetimes "so that ggml graph doesn't recycle them" — that runs before ggml-alloc (ggml-backend.cpp:1470 comment) and therefore enlarges the compute buffer by roughly the Q+K+V activation set of one layer; (2) each concurrent stream gets its OWN `ggml_cuda_pool` — `pools[device][curr_stream_no]` (common.cuh:1513-1521), instantiated via `new_pool_for_device(device, stream_no)` (ggml-cuda.cu:685-693), so 3 additional VMM pools commit their own dequant/quantized-src1 scratch. Neither is in the `--fit` budget. On a 12 GB card with the fit already tuned to the MiB, a config where this DID fire could OOM or force `--fit` to shed layers, which would itself masquerade as "the flag made it slower". Since no `ggml_cuda_concurrent_event` is ever constructed here, the cost is exactly zero: no extra stream is created, no extra pool is instantiated, no node is reordered.

### If it is measured

Do not spend a GPU round. There is only one value that is not the default (`1`), and on `draft-dflash,ngram-mod` with `--spec-draft-n-max 4` it provably executes zero optimization work — the same class of dead sweep as `--spec-ngram-mod-n-min`.

If the register needs a measured entry rather than a code argument, the cheap and honest experiment is an ACTIVATION CHECK, not a benchmark. This project's launch config already surfaces ggml-cuda `GGML_LOG_DEBUG` output (log:4774-4775 carry `CUDA Graph id 57 reused` and `CUDA graph warmup complete`). So: one boot with `GGML_CUDA_GRAPH_OPT=1` set in the environment before `llama-server.exe`, generate ~50 tokens, then `grep -c "Adding stream at node"` and `grep -c "Launching .* streams at"` in the log. Zero hits on both = the flag is dead on this config, recorded as evidence, no paired round, no timing, no VRAM comparison. Non-zero hits = the code reading above is wrong and everything here must be re-derived before any timing is trusted.

Uninterpretable results — the traps:
- ANY timing A/B with the spec stack on. Both arms execute byte-identical GPU work. A non-zero delta is boot-to-boot free-VRAM movement driving `--fit` (the repo's documented 9,326-10,732 MiB / 13.6 % noise floor), not the flag. Publishing a win from that would be the third misread sweep today.
- Setting the variable to anything other than the literal `1` and reporting "no effect" — that is the OFF path measured twice.
- Using `GGML_CUDA_DISABLE_GRAPHS=1` as the control arm. It disables GRAPH_OPT as well (common.cuh:1257-1260), so treatment and control are the same code path.
- Exporting the variable after the server is already up, or via a preset that is applied post-launch — the `static` at 4330 has already latched.
- Comparing across boots at all, per CLAUDE.md.

The only configuration in which this flag could ever be measurable on this hardware is speculation OFF, i.e. plain 1-token decode — and even there it would touch only the 16 full-attention layers out of 64, because the 48 gated-delta-net layers have fan-out 4 and are excluded by `min_fan_out == max_fan_out == 3` (ggml-cuda.cu:4396-4397 vs qwen35.cpp:237,241,362,369). That is a config this project has already rejected, and per the repo's own rule a verdict there would not transfer back to the spec stack. If someone runs it anyway: pair within one round, alternate the order, and require BOTH the timing delta and a non-zero `Adding stream at node` count before calling it a result.

### Citations

- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:4331 — the only getenv("GGML_CUDA_GRAPH_OPT") in the tree`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:4330-4337 — function-local static, atoi(env)==1, early return when off`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:4344 — `if (!use_cuda_graph || ggml_backend_cuda_get_device_count() != 1) return;``
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:4381 — THE KILL LINE: `if (node && !is_noop(node) && ggml_nrows(node) <= 1) fan_out[src] += 1;` with the author's own `//TODO: check why nrows > 1 fails` on 4380`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:4396-4397,4403 — min_fan_out = max_fan_out = 3`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:4408 — `if (!strstr(root_node->name, "attn_norm")) continue;``
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:4501-4508 — fork dropped when unaccounted (cpy) nodes sit in the span`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:4519-4522 — GGML_LOG_DEBUG("Adding stream at node %s %p") — the activation probe`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:4526-4551 — the interleave that extends tensor lifetimes (the compute-buffer cost)`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:4009-4026 — try_launch_concurrent_event, fork event record + stream waits, GGML_LOG_DEBUG("Launching %d streams at %s")`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:4035-4101 — is_valid() gate and the fusion-restoring reorder; 4094 clears events when invalid`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:4102-4126 — join, and per-node `curr_stream_no` assignment`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\common.cuh:1257-1260 — ggml_cuda_graph::is_enabled(): only cc>=Volta and GGML_CUDA_DISABLE_GRAPHS`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\common.cuh:1513-1521 — pools[device][curr_stream_no]: one VRAM pool PER STREAM`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\common.cuh:1489-1493 — lazy cudaStreamCreateWithFlags per stream index`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:685-693 — new_pool_for_device: a VMM pool per (device, stream_no)`
- `C:\AI\llama.cpp\ggml\src\ggml.c:1291-1295 — ggml_nrows = ne[1]*ne[2]*ne[3]`
- `C:\AI\llama.cpp\ggml\src\ggml-backend.cpp:1470 — graph_optimize is called once per sched split, per graph build`
- `C:\AI\llama.cpp\src\models\qwen35.cpp:164-165 — cb(cur, "attn_norm", il): the name strstr looks for`
- `C:\AI\llama.cpp\src\models\qwen35.cpp:270,282,285 — attention layers: wq/wk/wv = fan-out 3`
- `C:\AI\llama.cpp\src\models\qwen35.cpp:237,241,362,369 — recurrent layers: wqkv/wqkv_gate/ssm_beta/ssm_alpha = fan-out 4, out of range`
- `C:\AI\llama.cpp\src\llama-model.cpp:2955-2965 — create_tensor_qkv: split q/k/v only when fused attn_qkv is absent`
- `C:\AI\llama.cpp\common\common.h:325-329 — draft n_max=3, n_min=0, p_min=0.0f`
- `C:\AI\llama.cpp\common\speculative.cpp:1178-1185 — dflash builds a block of n_draft+1 tokens in ONE llama_decode`
- `C:\AI\llama.cpp\common\speculative.cpp:1274-1281 — result.clear() only when result.size() < n_min (=0), so the draft is never empty`
- `C:\AI\qwen38-tuning\logs\arena-r0-iq2xxs-nomtp.log:114 — arch = qwen35`
- `C:\AI\qwen38-tuning\logs\arena-r0-iq2xxs-nomtp.log:62 — qwen35.full_attention_interval = 4 (16 of 64 layers)`
- `C:\AI\qwen38-tuning\logs\arena-r0-iq2xxs-nomtp.log:308-311 — blk.3 loads attn_q/attn_k/attn_v, i.e. unfused QKV`
- `C:\AI\qwen38-tuning\logs\arena-r0-iq2xxs-nomtp.log:4774-4775 — ggml-cuda GGML_LOG_DEBUG lines DO reach this project's logs`

---

## `-fitt / --fit-target (env LLAMA_ARG_FIT_TARGET)`

**Default.** 1024 MiB per device. common.h:473 `std::vector<size_t>(llama_max_devices(), 1024 * 1024*1024)`; the staged binary's --help confirms "default: 1024". The project runs 768, i.e. 256 MiB BELOW the upstream default -- it has never been measured against the actual default. Note llama-bench is different: its default is 0 (llama-bench.cpp:397) and it only runs the fit at all if you set the flag (llama-bench.cpp:2267).

### What it actually does

It is a per-device SUBTRAHEND, not a policy. Every decision --fit makes is of the form `projected_use <= free - margin`, and -fitt sets `margin`.

Value path: arg.cpp:2851-2874 parses the MiB list, multiplies by 1024*1024, and broadcasts a single value across all devices into `params.fit_params_target` (common.h:473). common.cpp:1297-1302 hands `.data()` to `common_fit_params` as `margins`. fit.cpp:199-208 copies it into `margins[]`. It is consumed at exactly three places:

1. THE EARLY EXIT (fit.cpp:253, 269-274). `projected_used = mb.total() = model + context + compute`; `projected_free = dmd.free - projected_used`. On a single device, `if (projected_free >= margins[0]) return;` -- the function returns having changed NOTHING. So above a threshold the flag is a no-op and every value produces a byte-identical configuration.

2. CONTEXT (fit.cpp:310-350), gated on `cparams->n_ctx == 0`. It sets `sum_used_target = sum_free - margins[...]`, measures memory at n_ctx_min, then linearly interpolates n_ctx between n_ctx_min and n_ctx_train so projected use lands on that target, rounding down to a multiple of 256 (fit.cpp:343). If `-c N` was passed, this whole branch is skipped (fit.cpp:367-368 logs "context size set by user ... -> no change").

3. LAYER PLACEMENT (fit.cpp:562): `targets[id] = dmds_full[id].free - margins[id]`. This single line is the only input the placement search has. The false-position search at fit.cpp:585-650 finds the largest number of layers with `mem[id] <= targets[id]`; MoE expert tensors of the remainder are redirected to the CPU buffer type via tensor_buft_overrides (fit.cpp:405-445, 484-490). Step 4 (fit.cpp:645-773) converts dense-only layers back to full layers and then tries to squeeze one more PARTIAL layer on, testing overflow_type UP / GATE / ATTN -- each trial compared against the same `targets[id]`.

So: BOTH context and layer placement, but they are mutually exclusive in practice and `-c` decides which. With `-c 16384` (dflash2_arena.py:325, production-iq2xxs-ngram.ps1:49) the context branch is dead and 100 % of the margin goes into layer placement / MoE offload.

WHAT THE MARGIN IS RESERVED FOR: everything `mb.total()` structurally cannot see. `compute` is only the ggml_backend_sched arena (llama-context.cpp:3281-3293). It excludes the separate ggml-cuda temp pool, which is cudaMalloc'd at op-execution time: `ggml_cuda_mul_mat_id` takes `src1_sorted` and `dst_sorted` from `ctx.pool()` sized `ne12*n_expert_used*ne10*ts` (ggml-cuda.cu:1964-1965), and the cuBLAS path converts whole src0/src1 tensors to f16 into `ctx.pool()` (ggml-cuda.cu:1438-1439, 1494). Both scale with the batch, so their peak is at PREFILL, not decode. Also excluded: cuBLAS/cuBLASLt workspace, allocator fragmentation, and any other process growing its VRAM after the fit ran. The margin is the budget for that unseen tail.

HOW IT FAILS WHEN TOO SMALL: not by crashing. Smaller margin -> LARGER target (fit.cpp:562) -> MORE layers kept on GPU. The projection then says it fits and the server starts normally. At the first large prefill the pool allocations above land on top of it. If they merely fail, ggml-cuda.cu:495-505 flushes the pool and retries, then `CUDA_CHECK(err)` aborts; if the context buffer fails, ggml-cuda.cu:887-892 returns nullptr and startup fails loudly. But on Windows/WDDM the driver typically satisfies the allocation by paging resident VRAM to system memory instead of returning cudaErrorMemoryAllocation, so nothing errors and prefill degrades by a factor. That is the 151 s -> 825 s at -fitt 192: fit.cpp did what it was told, and the reserve it was told to skip was exactly the prefill pool.

### The plausible-but-wrong reading

FOUR, in descending order of how much GPU time they would waste.

A. "--fit-target 768 is the VRAM headroom the server leaves." It is not, whenever -md is present. tools/server/server-context.cpp:1074 does `params_base.fit_params_target[i] += bytes;` before common_fit_params is ever called, with `bytes = dmd.model + dmd.context + dmd.compute` of the draft model (measure_model_bytes = has_draft, line 1036). The DFlash2 drafter on disk is 1,143,006,752 B = 1,090 MiB, so under `--spec-type draft-dflash,ngram-mod` the value reaching fit.cpp:562 is roughly 768 + 1,150-1,300 MiB. Consequence for a sweep: the SAME `-fitt` number is a different reservation in the `none` arm, the `ngram-mod` arm (no -md, has_dft() false) and the `dflash2` arms. Any cross-arm -fitt comparison, or any value tuned on ngram-mod and carried to dflash, measures two different margins under one label. The refuting line is server-context.cpp:1074, and the enabling condition is common.h:382-384 (`has_dft()` is nothing more than "was -md given").

B. "--fit-target trades context length against speed." Refuted by fit.cpp:310, `if (cparams->n_ctx == 0)`. The harness passes `-c 16384` (dflash2_arena.py:325), so cparams->n_ctx is non-zero, the entire context branch is skipped and fit.cpp:367-368 logs "context size set by user ... -> no change". Every MiB goes through fit.cpp:562 into layer placement and MoE-expert offload, nothing else. Anyone who reasons about -fitt as a context knob here is reasoning about dead code.

C. "Lower margin = tighter = the fitter is being cautious and holding layers back, so 192 was slow because it offloaded more." Exactly backwards inside fit.cpp. targets[id] = free - margin (fit.cpp:562), so lowering the margin RAISES the target and the search at fit.cpp:585-650 keeps MORE layers on the GPU. The 825 s prefill is not a placement fit.cpp regretted; it is a placement the projection endorsed. `compute` in that projection is only the ggml_backend_sched arena (llama-context.cpp:3281-3293) and excludes the ggml-cuda temp pool, whose two largest consumers -- `src1_sorted`/`dst_sorted` in ggml_cuda_mul_mat_id (ggml-cuda.cu:1964-1965) and the cuBLAS f16 conversion buffers (ggml-cuda.cu:1438-1439) -- are cudaMalloc'd at execution time and scale with the batch. Their peak is at prefill. That is why the regression is a prefill regression and why it produced a number instead of an error.

D. "The flag is continuous, so a finer sweep will find a better value." fit.cpp:269-274 returns with zero changes the moment `projected_free >= margins[0]`. Above that threshold every -fitt value yields an identical `-ngl`/`-ot`, and any tok/s spread between them is boot noise. The production script's own note that "the 65+0 split were unchanged" says that arm was in the dead zone. Worse, the threshold is `dmd.free`-dependent and free VRAM at boot swings 9,326-10,732 MiB here, so the same -fitt can be a no-op on one boot and force an offload on the next -- which would read as a bimodal throughput result and invite exactly the wrong conclusion. Classify every run by its fitted layer split before comparing any number.

(A fifth, cheap one: "-ngl auto might disable the fit." It does not -- arg.cpp:2748 maps "auto" to -1, which is llama_model_default_params().n_gpu_layers at llama-model.cpp:2484, so the guard at fit.cpp:377 does not fire. But `-ngl 99` WOULD fire it, common.cpp:1297 would throw the FAILURE status away, and the server would still start and print a believable number with --fit-target doing nothing.)

### Interactions

DISABLED BY: `--fit off` (common.cpp:1295) -- and that also skips the draft-bytes accounting at server-context.cpp:1033. `-ngl <number>` or `-ngl all` -> fit.cpp:377-378 throws COMMON_PARAMS_FIT_STATUS_FAILURE, so the layer-placement half is dead; `-ngl auto` maps to -1 which equals the llama default (arg.cpp:2748, llama-model.cpp:2484) and keeps the fit alive. `-ts` / `--tensor-split` set by the user -> fit.cpp:387 throws. `-ot` / tensor_buft_overrides set by the user -> fit.cpp:399 throws. `-sm row` -> fit.cpp:392 throws; `-sm tensor` -> fit.cpp:183 throws.

SILENTLY CHANGES: `-c` decides which half of the flag is live. `-c N` (N != 0) kills the context branch entirely (fit.cpp:310, 367-368). `-c 0` explicitly ALSO kills it, by a different route -- arg.cpp:1643 sets fit_params_min_ctx = UINT32_MAX. Omitting `-c` is the only way the margin buys context.

SILENTLY INFLATED BY (this is the one that matters here): server-context.cpp:1033-1085. When `has_spec` is true the server measures the draft/MTP model with common_get_device_memory_data and does `params_base.fit_params_target[i] += bytes` (line 1074), where `bytes = (has_draft ? dmd.model : 0) + dmd.context + dmd.compute`. `has_draft` is `!draft.mparams.empty()` (common.h:382), i.e. simply "was -md given". The arena passes `-md ...Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1,090 MiB on disk) with `-ngld 99`, so its full weights plus context plus compute are added. `-fitt 768` is therefore an effective margin near 2 GB in this arm. `--mmproj` does the same thing at server-context.cpp:1005-1028.

CLAMPS: none. arg.cpp:2851-2874 has no minimum, no maximum, and no check that the value is achievable. `-fitt 0` is legal. More than llama_max_devices() comma-separated values throws; fewer than the device count leaves the remaining devices at 1024 MiB.

INSIDE THE PROJECTION, NOT THE MARGIN: `--spec-draft-n-max 4` reaches cparams.n_rs_seq via common.cpp:1699, so its recurrent-state cost IS in `mb.total()` and is not something -fitt has to cover. `-b 2048 -ub 256` set the reserved graph size and so are inside `compute` -- but the batch-scaled CUDA pool allocations that ride on top of them are not.

ERROR HANDLING: common.cpp:1297 ignores the return value of common_fit_params. A FAILURE logs one LOG_WRN and the server boots with whatever partial mutation happened before the throw -- notably, an n_ctx reduction written at fit.cpp:332/343 survives a later throw at fit.cpp:378. llama-fit-params, by contrast, exits 1 on non-SUCCESS (fit-params.cpp:38-40).

### VRAM

No -- it never allocates anything. It SPENDS VRAM by proxy, in the opposite direction from the intuition. It is subtracted from free at fit.cpp:562, so a LARGER -fitt gives a SMALLER target, and the false-position search (fit.cpp:585-650) puts FEWER layers on the GPU and pushes more MoE expert tensors to the CPU buffer type. A SMALLER -fitt puts MORE on the GPU. The VRAM it reserves is not held by llama.cpp at all -- it is left unallocated so that the runtime ggml-cuda pool (ggml-cuda.cu:1964-1965, 1438-1439), the cuBLAS workspace, and allocator fragmentation have somewhere to land, since llama-context.cpp:3285 counts none of them.

### If it is measured

MEASURE THE FITTED CONFIG BEFORE MEASURING ANY TOK/S. The flag is a step function with a dead zone (fit.cpp:269-274), and the step location moves with boot-time free VRAM, which this project already knows swings 9,326-10,732 MiB. Do this first, and it costs no GPU round:

  a) The harness already passes `-lv 5` (LOG_TRC needs verbosity >= 4, log.h:115). Grep the existing arena logs for `no changes needed` and for `id=0, target=... MiB`. Any arm that printed "no changes needed" had --fit-target applied as a no-op.
  b) Better: build `llama-fit-params` from C:\AI\llama.cpp (tools/fit-params) and enumerate -fitt offline -- it prints the exact `-c N -ngl M -ot "..."` and never touches port 8080. CAVEAT: it does NOT run server-context.cpp, so it will not add the draft bytes. To reproduce the server, run it once on the drafter with `-fitp on` to get the draft's model/context/compute MiB, then add that number to whatever -fitt you pass the tool.

VALUES WORTH MEASURING (12 GB, -c 16384, -ngl auto, draft-dflash,ngram-mod):
  * 1024 -- the actual upstream default. Never measured here. If it is still in the dead zone, the whole 768 setting is decoration and the sweep ends in one round.
  * 768 -- incumbent, the paired control.
  * 512 and 384 -- the only plausible window between "no-op" and "over-commit". Effective margin under this arm is roughly -fitt + 1,150-1,300 MiB of draft accounting, so 512 is already ~1.7 GB effective.
  * 256 -- the boundary probe. Expect the first configuration change here, or the first paging.
DO NOT re-measure 192 (already 5.5x prefill regression) and do not sweep above ~1,536: the direction above the dead zone is monotone and known -- more margin, fewer layers on GPU, slower.
Alternate order within each round and pair against the 768 control inside the same boot; free VRAM at boot is the confounder this flag is most sensitive to.

RESULTS THAT WOULD BE UNINTERPRETABLE -- reject them, do not publish them:
  1. Two -fitt values whose fit trace produced the SAME `-ngl` / same `-ot` / same free_after but different tok/s. That is boot noise inside the 13.6 % band, not the flag. This is the most likely outcome and the sweep must be able to say so.
  2. A -fitt value tuned under one --spec-type carried to another. server-context.cpp:1074 makes the effective margin depend on whether `-md` was given. 768 under `ngram-mod` is 768 MiB; 768 under `draft-dflash,ngram-mod` is ~1.9 GB. Different experiments, same label.
  3. Any run with `-ngl` set to a number or to `all`. fit.cpp:378 throws, common.cpp:1297 discards the status, the server starts anyway and prints a perfectly plausible tok/s with --fit-target having done nothing.
  4. Any run where `-c` was dropped or set to 0. The margin then moves n_ctx (fit.cpp:310, arg.cpp:1643) instead of layers, so the arms differ in context length as well -- and a decode verdict at one depth does not transfer to another anyway.
  5. Any single-round result at the low end. The paging failure mode has non-deterministic onset because it depends on what else holds VRAM at that moment; a low -fitt can look fine for one round and collapse the next. Require prefill AND decode AND free_after per round, and treat a prefill outlier as evidence of paging rather than as a sample to average away.
  6. Comparing raw prefill across boots. The 151 s vs 825 s figure is only usable because the gap is 5.5x; a 20 % prefill difference across boots says nothing.

### Citations

- `C:\AI\llama.cpp\common\common.h:473 -- the real default, 1024 MiB per device`
- `C:\AI\llama.cpp\common\arg.cpp:2851-2874 -- parse, *1024*1024, single value broadcast, no clamp and no validation`
- `C:\AI\llama.cpp\common\common.cpp:1295-1303 -- call site; the returned status is DISCARDED`
- `C:\AI\llama.cpp\common\fit.cpp:199-208 -- margins_s copied into margins[]`
- `C:\AI\llama.cpp\common\fit.cpp:253 -- projected_free = dmd.free - mb.total()`
- `C:\AI\llama.cpp\common\fit.cpp:269-274 -- single-device EARLY EXIT: margin met => return, zero changes`
- `C:\AI\llama.cpp\common\fit.cpp:295-298 -- global_surplus -= margins[id]`
- `C:\AI\llama.cpp\common\fit.cpp:310 -- context reduction gated on cparams->n_ctx == 0`
- `C:\AI\llama.cpp\common\fit.cpp:312-350 -- sum_used_target = sum_free - margins; linear n_ctx interpolation, rounded to 256`
- `C:\AI\llama.cpp\common\fit.cpp:367-368 -- '-c set by user -> no change'`
- `C:\AI\llama.cpp\common\fit.cpp:377-378 -- throws if n_gpu_layers != llama default (-1); kills the layer-placement half`
- `C:\AI\llama.cpp\common\fit.cpp:530-543 -- MoE-all-on-CPU probe, also margin-subtracted`
- `C:\AI\llama.cpp\common\fit.cpp:562 -- targets[id] = dmds_full[id].free - margins[id]  <-- the whole layer-placement mechanism`
- `C:\AI\llama.cpp\common\fit.cpp:585-650 -- false-position search maximising layers subject to mem <= targets[id]`
- `C:\AI\llama.cpp\common\fit.cpp:645-773 -- step 4, partial-layer squeeze (UP/GATE/ATTN), same targets[]`
- `C:\AI\llama.cpp\common\arg.cpp:1641-1644 -- '-c 0' sets fit_params_min_ctx = UINT32_MAX, disabling context reduction`
- `C:\AI\llama.cpp\common\arg.cpp:2747-2752 -- '-ngl auto' => -1 (== llama default, fit stays alive); a number or 'all' => fit throws`
- `C:\AI\llama.cpp\src\llama-model.cpp:2484 -- llama_model_default_params().n_gpu_layers = -1`
- `C:\AI\llama.cpp\src\llama-context.cpp:3281-3293 -- 'compute' is ONLY the ggml_backend_sched arena`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:1964-1965 -- mul_mat_id src1_sorted/dst_sorted from ctx.pool(), batch-scaled, invisible to the fit`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:1438-1439,1494 -- cuBLAS f16 conversion buffers from ctx.pool()`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:493-508 -- pool alloc: on failure flush + retry, then CUDA_CHECK abort`
- `C:\AI\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:883-893 -- buffer-type alloc returns nullptr on failure (startup-time failure)`
- `C:\AI\llama.cpp\tools\server\server-context.cpp:961-965 -- has_draft = speculative.has_dft(); spec_mtp; has_spec`
- `C:\AI\llama.cpp\tools\server\server-context.cpp:1033-1085 -- THE DRAFT MODEL'S BYTES ARE ADDED TO fit_params_target BEFORE THE FIT RUNS`
- `C:\AI\llama.cpp\tools\server\server-context.cpp:1074 -- params_base.fit_params_target[i] += bytes;`
- `C:\AI\llama.cpp\tools\server\server-context.cpp:1005-1028 -- mmproj bytes added the same way`
- `C:\AI\llama.cpp\common\common.h:382-384 -- has_dft() == !draft.mparams.empty(), i.e. '-md was given'`
- `C:\AI\llama.cpp\common\common.cpp:1699 -- cparams.n_rs_seq = need_n_rs_seq(), so --spec-draft-n-max IS inside the projection`
- `C:\AI\llama.cpp\ggml\src\ggml-backend-reg.cpp:121,172 -- CUDA registered first, CPU last => index 0 is CUDA0, so the += lands on the right margin here`
- `C:\AI\llama.cpp\tools\fit-params\fit-params.cpp:33-66 -- offline tool: prints the fitted '-c N -ngl M -ot "..."' without starting a server`
- `C:\AI\qwen38-tuning\bench\dflash2_arena.py:67,325-326 -- the harness passes -md, -c 16384, -ngl auto, --fit-target 768`

---

## `-bs / --backend-sampling (target-side; env LLAMA_ARG_BACKEND_SAMPLING; also a per-request JSON field "backend_sampling")`

> **DO NOT SPEND A GPU ROUND.** The code shows our current value is
> the only sensible one, or that the flag cannot act here at all.

**Default.** false (common.h:295). Distinct field from the draft-side `common_params_speculative_draft::backend_sampling = true` (common.h:331).

### What it actually does

It does NOT mean "sampling runs on the GPU". It means "register this sequence's sampler chain with the context, and offload the longest OFFLOADABLE PREFIX of that chain into the model's compute graph."

Mechanism, traced through the consumers:

1. Registration. `-bs` sets `params.sampling.backend_sampling` (arg.cpp:2299). Two consumers: common.cpp:1369-1372 attaches per-seq sampler chains to `cparams.samplers` at context creation, and the server re-registers per task at server-context.cpp:1734-1744 via `llama_set_sampler(ctx_tgt, slot.id, chain)`. Note server-context.cpp:1275 calls `slot.reset()`, which at server-context.cpp:358 calls `llama_set_sampler(..., nullptr)` — so the context-creation registration is torn down at startup and the live behaviour is entirely per-task.

2. Prefix offload, not chain offload. `llama_context::set_sampler` (llama-context.cpp:1229-1245) calls the chain's `backend_init`. `llama_sampler_chain_backend_init` (llama-sampler.cpp:733-771) walks the chain in order, keeping a boolean `backend_prefix`: the first sampler that lacks `.backend_init` or returns false sets `is_backend=false` for itself and EVERY sampler after it (llama-sampler.cpp:746-765). `llama_sampler_chain_backend_apply` then `break`s at the first non-backend entry (llama-sampler.cpp:800-808).

3. Which samplers can offload at all: greedy, dist, top_k, top_p, min_p, temp, temp_ext, penalties, logit_bias. NOT offloadable (`.backend_init = nullptr`): typical (1986), xtc (2408), mirostat/mirostat_v2 (2529/2635), grammar (2758), top_n_sigma (3292), **dry (3631)**, adaptive_p (3852), infill (4280).

4. The default chain kills it. `--samplers` defaults to `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature` (common.h:260-270, confirmed from the staged binary's --help), built unconditionally regardless of parameter values (sampling.cpp:350-406), plus `dist` appended (405) and `logit_bias` prepended when non-empty (341-343). So the prefix stops at **dry**, position 2. temp/dist — the parts that actually select the token — never reach the device under the default sampler set.

5. Speculation kills it one position earlier. `llama_sampler_penalties_backend_init` returns false whenever `n_outputs_max_per_seq > 1` (llama-sampler.cpp:3018-3021), with no is_disabled() escape. With any `--spec-type`, the target context gets `n_outputs_max_per_seq = 1 + common_speculative_n_max(...)` (server-context.cpp:42-53, speculative.cpp:2512-2521, 2351-2385). For our config that is `1 + max(draft.n_max=4, ngram_mod.n_max=32) = 33`. So penalties — the FIRST sampler in the chain — refuses, and the offloaded prefix is empty (or just `logit_bias`).

6. What is skipped on the host when it DOES work: `common_sampler_sample` (sampling.cpp:608-643) calls `llama_get_sampled_token_ith`; if the backend produced a token it returns immediately and "will not run any CPU samplers". `needs_raw_logits` (llama-context.cpp:1618-1633) then suppresses the device→host copy of the full logits row (llama-context.cpp:1863-1875), and `set_logits` (sampling.cpp:136-165) builds `cur` from the tiny sampled/candidates arrays instead of materialising ~152k `llama_token_data` per output position. At temperature 0.0 the device path is an `ggml_argmax` (llama-sampler.cpp:2041-2058) that reduces logits and candidates to ONE element before dist, so the copy-back is 1 token + 1 logit + 1 probability per row instead of ~152k floats × up to 33 rows.

7. What happens when the prefix is empty (our case): `build_sampling` still runs. `data.logits` is initialised to the row view (llama-graph.cpp:3710-3711), chain_backend_apply is a no-op, so `res->t_sampled_logits[row]` = the unmodified row (3736-3742) and it is copied to host as `sampling.logits` (llama-context.cpp:1963) — the same n_vocab volume as the raw-logits copy that `needs_raw_logits` just suppressed. `t_sampled` stays null, so `llama_get_sampled_token_ith` returns LLAMA_TOKEN_NULL and the entire CPU chain runs anyway. Zero work moved, all overhead paid.

### The plausible-but-wrong reading

Four plausible-but-wrong readings, each refuted by a specific line.

(1) "It moves sampling to the GPU." It moves the longest offloadable PREFIX. llama-sampler.cpp:746-765 sets `is_backend=false` for every sampler after the first failure, and 800-808 breaks there. With the default `--samplers`, the prefix ends at `dry` (llama-sampler.cpp:3631, `.backend_init = nullptr`) — position 2 of 10. temperature and dist, the samplers that actually pick the token, are never on the device unless you also change `--samplers`.

(2) "It works with speculative decoding — the code clearly supports multi-output." dist does (llama-sampler.cpp:1266). **penalties does not**: llama-sampler.cpp:3018-3021 returns false whenever `n_outputs_max_per_seq > 1`, unconditionally, without checking `is_disabled()`. penalties is FIRST in the default chain. Our target context has `n_outputs_max_per_seq = 1 + max(4, 32) = 33` (server-context.cpp:42-53 → speculative.cpp:2514 → 2351-2385). So on our exact config, `-bs` offloads **nothing** (or only `logit_bias` if any bias/suppress token exists, sampling.cpp:341). This is the single most important line in the report.

(3) "If nothing offloads, -bs is a harmless no-op." It is not. `needs_raw_logits` (llama-context.cpp:1618-1633) returns false as soon as the seq is *registered*, regardless of whether any sampler actually offloaded — it never inspects `is_backend`. The raw-logits copy is suppressed and replaced by an identical-volume copy of the untouched row into `sampling.logits` (llama-graph.cpp:3736-3742 → llama-context.cpp:1963). You pay `ggml_pad` of the entire logits matrix in the compute graph (llama-graph.cpp:3688), ~3× extra pinned host output buffer (llama-context.cpp:2076-2077), and a `sched_need_reserve` graph re-reserve on every task start AND every slot release (llama-context.cpp:1242/1261 via server-context.cpp:1741 and 358) — for zero work moved.

(4) "common.h:331 defaults true, so the draft side is already doing this and -bs is the matching target switch." For a **DFlash2** draft model it is doing nothing at all: speculative.cpp:1015 is `if (this->params.backend_sampling && !is_dflash2)`, and `is_dflash2` is true whenever the draft GGUF carries `dflash.selector_top_k > 0` (speculative.cpp:976-979) — which is exactly what PR #27342 artifacts carry. So on a DFlash2 draft, `--spec-draft-backend-sampling` / `--no-spec-draft-backend-sampling` changes only `n_outputs_max_per_seq` on the draft context (speculative.cpp:2424-2426), which has no consumer once no sampler is registered there. Sweeping the draft-side flag on a DFlash2 artifact measures nothing. Verify which you have before assuming: check the draft GGUF for `dflash.selector_top_k`, or read the launch log line `- block_size=..., sample_from_anchor=...` region (speculative.cpp:983-986).

Bonus trap for reading the logs: the server's `sampler chain: logits -> +penalties -> dry -> ...` trace (server-context.cpp:1746, names from llama-sampler.cpp:529-549) shows `+`/`-` for each sampler's OWN backend_init result. It does not show the prefix cut. A chain can print `+logit_bias -penalties +top_k +temperature +dist` while only `logit_bias` runs on the GPU.

### Interactions

DISABLED SILENTLY BY (each of these makes an `-bs` arm measure nothing):
- Any grammar / json_schema / tool-call grammar on the request → sampling.cpp:421-425, logs "backend sampling is not compatible with grammar, disabling" and flips the flag false. This repo ships a `grammars/` directory; if the bench corpus uses one, the sweep is void.
- Reasoning budget / reasoning_control → sampling.cpp:427-431. Populated for a thinking model on `/v1/chat/completions` whenever `reasoning_budget_tokens >= 0` or `--reasoning-budget` is set (server-common.cpp:1352-1366, sampling.cpp:317). `--reasoning-budget 0` — the usual way to turn Qwen3 thinking off — turns `-bs` off with it.
- `n_probs`/`logprobs` > 0 without `post_sampling_probs` → server-context.cpp:1732-1737, `use_backend_sampling &= !need_pre_sample_logits`.
- `LLAMA_SPLIT_MODE_TENSOR` → llama-context.cpp:1216-1227, WARNs "using CPU". Not our case (single GPU, default LAYER).
- A device op-support failure → llama-sampler.cpp:638-661 WARNs the exact op and sampler name.

NEUTERED (not disabled) BY:
- The default `--samplers` order — prefix stops at `dry` (llama-sampler.cpp:3631).
- Any `--spec-type` at all — penalties refuses at `n_outputs_max_per_seq > 1` (llama-sampler.cpp:3018), and penalties is first. This applies to `ngram-mod` alone as much as to `draft-dflash`; there is nothing dflash-specific about it. It is driven purely by `common_speculative_n_max` = max over enabled types = 32 from `--spec-ngram-mod-n-max 32`, so even dropping draft-dflash would not restore it.

WHAT IT SILENTLY CHANGES:
- Suppresses the raw-logits D2H copy for every registered seq (llama-context.cpp:1863 via 1618-1633) even when nothing offloaded, substituting an equal-volume copy through `sampling.logits`.
- Adds the `ggml_pad` logits copy to the compute graph (llama-graph.cpp:3688).
- Forces a graph re-reserve at every task start and every slot release.
- `common_sampler_init` mutates the caller's `params` when it disables the flag (sampling.cpp:424/430 write through the `common_params_sampling &`), so the `/props` and task JSON echo (server-task.cpp:86, 145) can report `backend_sampling` differently from what you passed. Read the echoed value, not the command line.

INDEPENDENT OF: `--spec-draft-backend-sampling` (common.h:331). That is a different struct on a different context, and on a DFlash2 draft it is dead code (speculative.cpp:1015). Setting or clearing it does not affect what `-bs` does on the target.

COMPATIBLE WITH: temperature 0.0 speculative verification. server-context.cpp:3828-3831 selects the non-`dists` overload when `temp <= 0`, which calls `common_sampler_sample` per position and takes the backend-token fast path cleanly. (The `dists` overload at sampling.cpp:753 needs the full candidate distribution and would be degraded by a fully-offloaded chain — that is a temp>0 concern only.)

### VRAM

Yes, on the target context, and it is invisible to `--fit`.

Mechanism: `build_sampling` opens with `ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0)` (llama-graph.cpp:3688) — a full copy of the [n_vocab × n_outputs] logits matrix plus one dummy row, allocated in the compute buffer. This node exists only when at least one sampler is registered (guard at 3660). For Qwen3's ~152k vocab and n_outputs = 33 (the spec verify batch), that is ~152k × 34 × 4 B ≈ 20 MB of extra compute buffer. Per-row sampling temporaries add on top: with temp 0.0 they are trivial (argmax → 1 element, llama-sampler.cpp:2045-2056), but with a temp>0 chain, dist's softmax+cumsum+step over the full vocab is ~3 more n_vocab f32 tensors per output row.

The graph node budget also grows: llama-context.cpp:2323-2341 adds `n_sampling_nodes` plus `(n_outputs-1) × n_sampling_nodes_max`.

Critically, `--fit` cannot see any of this: common.cpp:1294-1303 runs `common_fit_params` **before** line 1369 sets `cparams.samplers`, and fit.cpp contains no reference to `samplers` or `n_outputs_max` at all. So `-bs` adds compute-buffer VRAM after the fit has already chosen n_gpu_layers/n_ctx. On a 12 GB card at the edge this is a plausible OOM or eviction mechanism that the fit log will not warn about.

NOT VRAM: the ~3× larger output buffer (llama-context.cpp:2076-2077, 2148-2159) is allocated from `ggml_backend_dev_host_buffer_type` — CUDA pinned **host** memory (llama-context.cpp:2110-2117). Roughly 60 MB extra host-pinned at n_outputs=33; do not count it against the 12 GB.

Also: `llama_set_sampler` sets `sched_need_reserve = true` on both register and clear (llama-context.cpp:1242, 1261). With `-bs` the server does this twice per request (server-context.cpp:1741 and 358), so the sched buffers are re-reserved around every request — a per-request latency and a re-allocation event, not a steady-state cost.

### If it is measured

Do NOT run the obvious sweep. `-bs` off vs `-bs` on with everything else held at our config is a guaranteed non-result: penalties refuses (llama-sampler.cpp:3018) because per_seq = 33, so arm B offloads nothing while paying ~20 MB of compute buffer, ~60 MB pinned host, and two graph re-reserves per request. The predicted effect is a small negative, well under the 13.6 % noise floor, i.e. uninterpretable in either direction. That round is better spent elsewhere.

The one experiment that is worth GPU time, if you want this flag answered at all — a 2-arm pair inside ONE boot, alternating order:

  arm A: --samplers "temperature" (nothing else changed, -bs OFF)
  arm B: --samplers "temperature" -bs

`--samplers "temperature"` yields chain [temp_ext, dist] (sampling.cpp:380-381, 405), both backend-capable, and at temp 0.0 temp_ext collapses to `ggml_argmax` on device (llama-sampler.cpp:2041-2058) so dist selects from a 1-element candidate set. That is the only configuration in which `llama_get_sampled_token_ith` returns a real token and the ~152k-entry host candidate array (sampling.cpp:161-164) is skipped — for up to 33 verify positions per decode step. At temperature 0.0 `--samplers "temperature"` is behaviourally identical to the default chain (every other sampler is a no-op at its default value), so arm A is a valid baseline for our production config as well. If you distrust that, add `--samplers "top_k;temperature"` with top_k 1 — top_k is also backend-capable (llama-sampler.cpp:1463).

Optional third arm, only if A/B separates: `--samplers "penalties;temperature"` to confirm the penalties cliff directly — it should land on top of arm A, proving the mechanism rather than just the effect.

UNINTERPRETABLE RESULTS — do not report a number from any of these:
- Any arm whose requests carry a grammar or json_schema (sampling.cpp:421), `n_probs`/`logprobs` > 0 (server-context.cpp:1737), or a reasoning budget / `--reasoning-budget` (sampling.cpp:427). Each silently forces the flag false, so "-bs on" is actually "-bs off" and the arms are identical by construction.
- Comparing `-bs` on with `--samplers "temperature"` against `-bs` off with the DEFAULT samplers. That confounds device placement with sampler set and is not a measurement of this flag.
- Any cross-boot comparison. Standard house rule, and doubly so here: `-bs` adds compute buffer that `--fit` never modelled (common.cpp:1294 runs before 1369; fit.cpp has no `samplers` reference), so the two arms can land on different memory outcomes for reasons unrelated to sampling speed.
- Any arm run against a draft GGUF whose DFlash2 status you have not checked, if you also touch `--spec-draft-backend-sampling` in the same round.

VERIFY BEFORE TRUSTING EITHER ARM: run with trace logging and read the `sampler chain:` line (server-context.cpp:1746). Arm B must print `+` on every sampler in the chain — `logits -> +temp_ext -> +dist`. A single `-` anywhere, or the WARN "sampler '<name>' for seq_id = N, cannot be offloaded to the backend" (llama-context.cpp:1248), means the arm did not do what its name says. Remember the `+`/`-` marks each sampler's own init result, not the prefix cut (llama-sampler.cpp:529-549), so also confirm the FIRST entry is `+`.

### Citations

- `C:/AI/llama.cpp/common/arg.cpp:2295-2301 — the flag definition, .set_sampling().set_env()`
- `C:/AI/llama.cpp/common/common.h:295 — target-side default false`
- `C:/AI/llama.cpp/common/common.h:331 — DRAFT-side default true (different struct, different context)`
- `C:/AI/llama.cpp/common/common.h:260-270 — the default sampler order: penalties, dry, top_n_sigma, top_k, typ_p, top_p, min_p, xtc, temperature`
- `C:/AI/llama.cpp/common/common.cpp:1364-1372 — cparams.samplers attached only when backend_sampling`
- `C:/AI/llama.cpp/common/common.cpp:1294-1303 — --fit runs BEFORE line 1369, so the fit estimate never sees the sampling graph`
- `C:/AI/llama.cpp/common/sampling.cpp:341-343 — logit_bias prepended when non-empty`
- `C:/AI/llama.cpp/common/sampling.cpp:350-406 — chain built unconditionally from params.samplers, no pruning of no-op samplers`
- `C:/AI/llama.cpp/common/sampling.cpp:421-425 — grammar silently disables backend sampling (WARN)`
- `C:/AI/llama.cpp/common/sampling.cpp:427-431 — reasoning budget silently disables backend sampling (WARN)`
- `C:/AI/llama.cpp/common/sampling.cpp:608-643 — the fast path: backend token short-circuits all CPU samplers`
- `C:/AI/llama.cpp/common/sampling.cpp:136-165 — set_logits; full-vocab candidate array built only when no sampled logits`
- `C:/AI/llama.cpp/src/llama-sampler.cpp:733-771 — chain_backend_init, the PREFIX rule`
- `C:/AI/llama.cpp/src/llama-sampler.cpp:800-808 — chain_backend_apply breaks at first non-backend sampler`
- `C:/AI/llama.cpp/src/llama-sampler.cpp:3018-3021 — penalties refuses when n_outputs_max_per_seq > 1 (i.e. under ANY speculation)`
- `C:/AI/llama.cpp/src/llama-sampler.cpp:3631 — dry has .backend_init = nullptr`
- `C:/AI/llama.cpp/src/llama-sampler.cpp:1986, 2408, 3292 — typ_p, xtc, top_n_sigma have no backend impl either`
- `C:/AI/llama.cpp/src/llama-sampler.cpp:2036-2064 — temp<=0 collapses to ggml_argmax on device (the greedy win)`
- `C:/AI/llama.cpp/src/llama-sampler.cpp:1257-1272 — dist backend_init; sets backend_transactional when n_outputs_max_per_seq>1 (dist DOES support spec)`
- `C:/AI/llama.cpp/src/llama-sampler.cpp:529-556 — the +name/-name convention printed in the sampler-chain trace`
- `C:/AI/llama.cpp/src/llama-sampler.cpp:638-661 — llama_sampler_backend_support: per-op device probe, WARNs the unsupported op`
- `C:/AI/llama.cpp/src/llama-context.cpp:1209-1264 — set_sampler: SPLIT_MODE_TENSOR refusal, sched_need_reserve on every set/clear`
- `C:/AI/llama.cpp/src/llama-context.cpp:1618-1633 — needs_raw_logits keys off REGISTRATION, not off actual offload`
- `C:/AI/llama.cpp/src/llama-context.cpp:1664-1689 — hard error -1 if a sampler-bearing seq exceeds n_outputs_max_per_seq`
- `C:/AI/llama.cpp/src/llama-context.cpp:2073-2078, 2110-2117 — extra output buffers (~3x logits) in CUDA PINNED HOST memory, not VRAM`
- `C:/AI/llama.cpp/src/llama-context.cpp:2323-2341 — graph node budget grows by n_sampling_nodes per output row`
- `C:/AI/llama.cpp/src/llama-graph.cpp:3688 — ggml_pad of the whole logits matrix: the VRAM cost`
- `C:/AI/llama.cpp/src/llama-graph.cpp:3698-3752 — build_sampling loops per output row`
- `C:/AI/llama.cpp/tools/server/server-context.cpp:42-53 — server_output_limits: per_seq = 1 + n_draft`
- `C:/AI/llama.cpp/tools/server/server-context.cpp:1732-1744 — per-task registration; n_probs>0 silently disables it`
- `C:/AI/llama.cpp/tools/server/server-context.cpp:358 + 1275 — slot.reset() clears the context-creation registration at startup`
- `C:/AI/llama.cpp/tools/server/server-context.cpp:3828-3831 — at temp 0.0 the verify path uses the non-dists overload, which is compatible`
- `C:/AI/llama.cpp/common/speculative.cpp:2512-2521 and 2351-2385 — where per_seq=33 comes from for draft-dflash,ngram-mod`
- `C:/AI/llama.cpp/common/speculative.cpp:1015 — draft-side backend sampling is SKIPPED entirely when is_dflash2`
- `C:/AI/llama.cpp/common/speculative.cpp:976-979 — is_dflash2 = (dflash.selector_top_k > 0) read from the draft GGUF`
- `C:/AI/llama.cpp/common/speculative.cpp:2413-2427 — draft ctx n_outputs_max_per_seq = n_max+1 gated on draft.backend_sampling`
- `C:/AI/llama.cpp/common/fit.cpp — contains no reference to samplers or n_outputs_max`

---
