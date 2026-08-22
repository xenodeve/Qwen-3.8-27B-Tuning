# 05 — Runtime flags: threads, placement, sampling

Two full sweeps of this surface found **nothing above the 13.6 % drift floor**.
That is the headline of the page: the flags are not where the wins are.

## Placement and scheduling — all inert

Two rounds each at 16,384 on `v3-iq2xxs`, `--fixed-text`, order reversed:

| flag | effect | verdict |
|---|---|---|
| `pcore-mask` (thread affinity to P-cores) | +0.46 % | inert |
| `prio-high` (process priority) | −2.02 % | inert, sign flips |
| `poll-0` (polling strategy) | +0.69 % | inert |
| `backend-samp` (GPU-side sampling) | +2.27 % | inert |

Baseline decode was 38.6–38.65 and every arm landed between 36.4 and 39.5.

**A methodological note worth following up, not acting on.** Those pairs repeat
to within **0.05 percentage points across separate boots** — two orders of
magnitude tighter than the 13.6 % floor, which was derived from unpinned text.
`backend-samp` at +2.27 % with a range of 0.05 does not look like noise. But free
VRAM spanned only 2,872–3,016 MiB across these boots, a fifth of the 9,326–10,732
spread the floor came from. **A quiet night is not a smaller floor.**

*Raw: `results/kv-layers-16k.jsonl`. Reports 20 §4, 23 §4.*

## Threads and batch

| flag | tried | result |
|---|---|---|
| `-t` 8 / 12 / 18 / 24 | yes | 18 chosen; differences under the floor |
| `-b` / `-ub` | yes | see [`03-memory-and-kv.md`](03-memory-and-kv.md) — a VRAM lever, not a speed lever |
| `-fa on` / `off` | yes | `on` required; `off` loses residency |
| `-sm tensor` | yes | single GPU, no effect |
| `--no-repack`, `--no-op-offload`, `--load-mode`, `--no-host`, `--swa-full`, `--no-kv-unified` | yes | all confirmed inert, as report 16 predicted |

*Raw: `results/sweep-threads*.jsonl`, `results/sweep-batch*.jsonl`,
`results/kv-layers-16k.jsonl`, `results/kv-depth-levers.jsonl`.*

## Six flags read from source before spending GPU time — 2026-08-22

**Read, not measured.** Six agents each traced one flag into the code that
*consumes* it, after two sweeps that day were designed against a misreading and
measured nothing. Three of the six turn out to be provably inert in our
configuration, which saves three GPU rounds.

### 🔴 `-ctkd` / `-ctvd` — DO NOT SWEEP. The saving is real; the cost is larger

Default **`f16` for both** (`common/common.h:340-341`) — *not* "same as
`--cache-type-k`". Quantising the draft KV drops its buffer from 45.00 MiB to
~12.7 MiB at `q4_0`.

**But the drafter does not do single-token steps.**
`common/speculative.cpp:1183` computes `n_block_tokens = n_max + 1` = **5**, and
line 1196 decodes all five in one `llama_decode`, so `Q->ne[1] == 5`. That does
**not** take the quantised-KV vector kernel — it takes `MMA_F16` with a full
dequantisation per layer per draft step. The ~32 MiB saving is paid for in
compute on the hot path.

Also: a quantised V silently promotes flash-attention AUTO → ENABLED for the
*draft* context (`llama-context.cpp:3602-3605`), so you lose both the "Flash
Attention enabled" confirmation and the "not supported" warning for the drafter.

**If measured anyway**, the only defensible readout is the load-time VRAM pair
from one log: the draft `KV buffer size` line must fall **and** the draft
`compute buffer size` line must rise. If the compute line does not move, the
reading above is wrong and the sweep should stop.

### 🔴 `GGML_CUDA_GRAPH_OPT=1` — DO NOT SWEEP. It cannot fire here

Undocumented environment variable; unset = disabled
(`ggml/src/ggml-cuda/ggml-cuda.cu:4330-4334`). The scan called it "the cheapest
untried experiment in the whole map". It is cheaper still: **in our
configuration it provably executes zero optimisation work.**

The misreading it invites: *"our logs show `CUDA Graph id 57 reused`, so CUDA
graphs are on, so this should help."* The function body
(`ggml-cuda.cu:4339-4551`) contains **no `cudaGraph*` call at all** — it builds
`cudaEvent_t` fork/join pairs and a stream map. CUDA graphs working tells you
nothing about whether this flag does anything.

`GGML_CUDA_DISABLE_GRAPHS` cannot serve as the control arm: it turns off the
gate the flag itself depends on (`common.cuh:1257-1260`).

**The honest cheap experiment** is an activation check, not a benchmark: one
boot with the variable set, and grep the debug output for the optimisation
firing at all.

### 🔴 `-bs` / `--backend-sampling` — DO NOT SWEEP as-is. A guaranteed non-result

Default false (`common/common.h:295`), and a **different field** from the
draft-side `backend_sampling = true` at `common.h:331`.

It does not "move sampling to the GPU" — it offloads **the longest prefix** of
the sampler chain. `llama-sampler.cpp:746-765` sets `is_backend=false` for every
sampler after the first failure. With our default `--samplers`, the prefix ends
at `dry`, **position 2 of 10**; `temperature` and `dist` — the samplers that
pick the token — never reach the device.

Worse, `penalties` refuses outright (`llama-sampler.cpp:3018`), so an `-bs` arm
offloads nothing while paying ~20 MB of compute buffer and ~60 MB pinned host,
plus two graph re-reserves per request. **Predicted effect: a small negative,
under the noise floor — uninterpretable in either direction.**

And it self-disables silently on a grammar or a reasoning budget
(`sampling.cpp:421-427`). The served profile needs a grammar, so an `-bs` arm
measured with one is void by construction.

### `--spec-draft-p-min` — sweepable, but **anything ≤ 0.0625 is a no-op**

Default **0.00** (`common/common.h:329`), i.e. the confidence early-stop is off,
and the `if (params.p_min > 0.0f)` guard at `speculative.cpp:1262` makes that a
genuine zero-cost path.

**What it actually compares.** Our drafter is DFlash2
(`dflash.selector_top_k = 16`), so `speculative.cpp:978` sets `is_dflash2` and
control enters the lattice branch at 1219. The vocabulary-probability check at
`speculative.cpp:1328` is **dead code for this artifact**. At temperature 0 the
live line is 1268: `1.0f / sum < params.p_min`, where
`sum = Σ_{k=0..15} exp(scores[k] − scores[argmax])` over the **16 selector
candidates**, not the 151k vocabulary.

> **Every term is ≤ 1 and the argmax term is exactly 1, so `1/sum ∈ [0.0625,
> 1.0]`. Any `p_min ≤ 1/16` is mathematically identical to 0.00.**

A "gentle first step" of 0.01 or 0.05 would repeat the `--spec-ngram-mod-n-min`
error exactly: a ladder of arms that cannot differ.

Two more things that shrink its value: it saves **zero draft-side compute** (the
whole block is decoded at `speculative.cpp:1195` *before* any check), and
`ngram-mod` outranks `draft-dflash`, so every step ngram serves is a step where
`p_min` does nothing.

### `--spec-ngram-mod-n-match` — swept 2026-08-22. It was worth the round, and we had it backwards

Default **24** (`common/common.h:352`); we run **12**. Costs no VRAM — the table
is a fixed 16 MiB host allocation independent of `n_match`.

> ✅ **Measured.** ctx 16,384, real-code, three paired rounds, arms rotated:
> **`24` +34.6 % [+31.4, +40.8] RESOLVED**, `16` −1.5 % within the floor,
> `12` baseline, **`8` −14.5 % [−20.9, −8.0] RESOLVED**. The trap below is
> exactly what happened — at `8`, ngram fires 43 times against 29 at `24` and
> still loses, because mean accepted length falls 23.45 → 8.95.
> **The default beats the value in all four worker profiles.** Not yet a config
> change: this is a 16,384 verdict and the profiles serve 65,536–98,304.
> [`02-decoders.md`](02-decoders.md) · [`CORRECTIONS.md` §21](../reports/CORRECTIONS.md).

**The trap that would make the sweep uninterpretable:** `ngram-mod` is
registered *above* `draft-dflash` and the cascade stops at the first non-empty
draft (`speculative.cpp:2545, 2551, 2725-2726`). **Lowering `n_match` raises
ngram's fire rate by suppressing dflash calls.** An arm can show ngram firing
twice as often and decode *slower*. Any sweep must read the per-implementation
counters, not just tok/s.

Arms worth measuring: **24 (the real default), 16, 12 (ours), 8.** Not below 6 —
the key collapses, acceptance falls under the reset threshold, and the table
wipe at `speculative.cpp:2044-2054` fires repeatedly, so the measurement becomes
one of the reset loop. Not above 24 either: the `i = 0` hit rate is monotonically
non-increasing in `n_match`, and the 93.7 % decline is already the binding
constraint.

### `--fit-target` — and the discovery that changes what it means here

Default **1024 MiB per device** (`common/common.h:473`); we run 768.

> **When `-md` is present, the server ADDS the draft model's bytes to the
> target before `--fit` ever runs.** `tools/server/server-context.cpp:1074`:
> `params_base.fit_params_target[i] += bytes`, where `bytes` is the drafter's
> model + context + compute. The DFlash2 sidecar is 1,090 MiB on disk, so the
> value reaching `fit.cpp:562` under `draft-dflash,ngram-mod` is roughly
> **768 + 1,150–1,300 ≈ 1,900–2,100 MiB**, not 768.

So the header comment on every worker profile describing 768 as "the margin the
server leaves free" is wrong in exactly the configuration we now serve. It also
explains why the drafter fitted at 98,304 when arithmetic from the raw buffer
sizes said it should not.

The direction is counter-intuitive: `-fitt` is **subtracted** from free at
`fit.cpp:562`, so a *larger* value gives a *smaller* target and pushes **fewer**
layers onto the GPU. And it is a **step function with a dead zone**
(`fit.cpp:269-274`) whose step location moves with boot-time free VRAM — which
this project already knows swings 9,326–10,732 MiB.

**Measure the fitted configuration before measuring any tok/s.** Our harness
already passes `-lv 5`; grep the existing arena logs for `no changes needed` —
any arm that printed it had `--fit-target` applied as a no-op.

---

## Speculation flags — swept 2026-08-22, on the frozen corpus

Arena: `bench/dflash2_arena.py --regime real-code`, ctx 16,384,
`draft-dflash,ngram-mod`, three rounds, arms rotated, paired by round.
Raw: `results/sweep-*.jsonl`.

### `--spec-ngram-mod-n-min` — **no effect. Do not re-run it.**

| `n-min` | rounds (tok/s) | vs base |
|---:|---|---|
| 16 (ours) | 79.7, 79.6, 79.8 | baseline |
| 8 | 79.7, 79.5, 79.8 | −0.1 % |
| 4 | 79.7, 79.6, 79.8 | −0.0 % |
| 2 | 79.8, 79.8, 79.7 | +0.1 % |

Spread across all twelve runs: **0.3 %**. At that repeatability a 1 % effect
would be visible; there is none.

**The hypothesis and why it was wrong.** `ngram-mod` declines **93.7 %** of the
calls it receives on real code, and when it does fire it is worth **16.7
tokens** against `draft-dflash`'s 2.9 — so letting shorter drafts through looked
like a large free win.

It was a misreading of `common/speculative.cpp:1993`. In `draft_one`, `i` counts
**draft tokens already produced**, not matched context. `n_min` is therefore a
minimum draft *length*, and the declines happen at `i = 0` — the n-gram table
misses on the very first successor — where no value of `n_min` can help.

**What that leaves open.** The decline rate is real and large. The knob that
governs it is `--spec-ngram-mod-n-match` (default 24, ours 12): the width of the
context window the table is keyed on.

**Swept since, and it does not close the decline.** `24` is **+34.6 % RESOLVED**
over our `12`, but it gets there by drafting *less often and better* — 29 drafts
against 31, at mean accepted length 23.45 against 18.00. The decline rate barely
moves (93.9 % against 94.3 %). **Nothing swept so far makes `ngram-mod` fire
usefully more often on real code**, which is the finding, not a gap in the sweep.

## Grammar (GBNF) — what it costs, and the one thing nobody has measured

| question | answer | evidence |
|---|---|---|
| Does `--grammar-file` allocate VRAM? | **No — read from source, NOT measured.** `src/llama-grammar.cpp` contains no reference to `ggml_backend`, `cuda`, `ggml_new_tensor` or `device`. Its whole state is `std::vector` on the host: a pushdown stack and a rule table. It runs in the sampler chain after logits are copied back | source read on build 10499, `src/llama-grammar.cpp`, `src/llama-grammar.h` |
| Then what does it cost? | **CPU time per token**, which surfaces as tok/s and not as MiB. Unquantified here | — |
| Does it change anything else? | **Yes: it disables backend sampling.** `common/sampling.cpp:421` — `"backend sampling is not compatible with grammar, disabling"`. `--reasoning-budget` does the same at line 427, and `grammars/README.md` tells you to use both | `common/sampling.cpp:421,427` |
| Does that cost us anything? | **No.** `common.h:295` defaults `backend_sampling` to `false`, no worker profile enables it, and the flag was measured at **+2.27 % — inert** (table above) | `common.h:295`, `results/kv-layers-16k.jsonl` |
| 🔴 **Does a grammar work alongside a drafter?** | **Unmeasured, and the config we intend to serve needs both.** `common.h:331` is a *different* field — `backend_sampling = true` for the **draft** sampler, on by default — and the disable at `sampling.cpp:421` touches only the main one. So grammar + drafter runs in a state nothing has exercised | source read; **no run** |

**Why the last row matters.** The production profile has to carry a grammar —
without one, 41.5 % (`UD-IQ1_M`) to 58.3 % (`UD-IQ2_XXS`) of corpus attempts emit
no fenced code block at all — and it has to carry a drafter, because that is
where the speed is. Every measurement so far has one or the other.

**The check is cheap:** two boots, grammar on and off, read `nvidia-smi`. Then
the same pair with `--spec-type draft-dflash,ngram-mod` added. It has not been
run because the GPU was busy; that is a schedule, not a result.

## Sampling — two passes, nothing resolved

Arms tried across `answer-screen-sampling.jsonl` and `-sampling2.jsonl`:
`samp-base`, `samp-dry`, `samp-mirostat`, `samp-nsigma`, `samp-rep-default`,
`samp-rep4096`, `samp-grammar`, `samp-prefill`, `samp-backend`,
`samp-rbudget0`, `samp-rbudget2k`, `samp-rea-off`, `samp-dry-rb2k`, and the
second-pass repeats.

**Nothing moved a decision.** Two findings did come out of it, both negative:

- **`--reasoning-budget 0` does not end the reasoning block.** Screened alone it
  ran to **24,709 characters**. Paired with `--grammar-file` it returned
  `content_chars = 0` on 3 of 3 trials — the model reasons freely, then emits
  end-of-turn where the grammar starts to bind. **`-rea off` is the flag that
  ends the block.**
- **The screen itself is capped at 3 trials.** `answer_screen --trials 10`
  silently gives 3, because `for i in range(min(args.trials, len(PROBES)))` and
  `PROBES` has three entries. Every "n=10" screen in this project was n=3.

*Raw: `results/answer-screen-sampling*.jsonl`. Reports 20 §6, 22 §5.*

## `reasoning_effort` — swept, but not where it matters

**Corrected 2026-08-21.** This was written up as never tested; it was tested on
2026-08-18. `results/reasoning-effort-sweep.jsonl`, `low`/`medium`/`xhigh`, two
runs each, on a tool-calling probe:

| effort | wall | completion tokens | reasoning chars | reached the patch |
|---|---|---|---|---|
| `low` | 67.8 / 85.3 s | 453 / 608 | 384 / 570 | **both** |
| `medium` | 85.6 / 50.1 s | 610 / 352 | 621 / 212 | **both** |
| `xhigh` | 84.2 / 106.9 s | 588 / 741 | 632 / 1,008 | **both** |

Reasoning length rises with effort, as expected. **All six runs succeeded**, so
the probe cannot separate the levels on quality.

**What is genuinely untested:** this ran on **Q4** with n=2 and a tool probe. It
has never run on the 2-bit V3 artifacts, where the failure being chased is the
model looping inside the reasoning block until the budget runs out, and never
through the 30-task corpus. An external review of this model reports xHigh taking
**15 minutes** where medium takes 3 for *"90 % of the result"* — a difference our
probe was far too short to see.

*Raw: `results/reasoning-effort-sweep.jsonl` (note: UTF-8 BOM, read with
`utf-8-sig`).*

## Never tried

- **A system prompt that instructs how to think** rather than how to format —
  e.g. *"don't hedge, make conclusions, work forward, don't reconsider"*. Costs
  nothing, aims straight at the blocker.
- **`reasoning_effort: low` on a 2-bit artifact, through the corpus.**

## KV cache type against prefill — tested 2026-08-21

| question | answer | evidence |
|---|---|---|
| Is `q8_0` KV faster than `q4_0` for prefill? | **No reliable difference.** 714/882.5 against 984/871.1 over two reversed rounds; the within-arm spread is wider than the gap | report 27 |
| Is `f16` KV worth trying? | **Not measurable here.** 2,048 MiB of KV left 427 and 242 MiB free, both rows in the collapse regime | report 27 |
| Does `iq4_nl` KV work? | **No.** Prefill abandoned at the 737 s timeout, twice | report 27 |
| Can prefill be tuned at all? | **No.** Every setting-level lever is measured and none move it | report 27 |

Raw: `qwen38-tuning/results/prefill-kv-type.jsonl`.
