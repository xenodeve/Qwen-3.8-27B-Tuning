# 05 — Runtime flags: threads, placement, sampling

> 🔴 **Every number on this page was measured at `reasoning_effort: xhigh` with
> an unlimited thinking budget, except the `reasoning_effort` sweep below, which sets `low`/`medium`/`xhigh` explicitly.** That is the model's chat-template
> default — the client sends no effort field, and **no `worker-*.ps1` profile and
> nothing in `bench/` has ever set the flag** (established 2026-08-24 from a boot
> log: [`05-runtime-flags.md`](05-runtime-flags.md), this page).
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

Two full sweeps of this surface found **nothing above the 13.6 % drift floor**.
That is the headline of the page: the flags are not where the wins are.

## Environment knobs — the surface nobody had looked at

`grep getenv ggml/src/ggml-cuda/` at `1deefcca3` finds **twelve** runtime knobs
that are not command-line flags. Until 2026-08-24 the arena could not test one:
arms carried argv only, so trying an env var meant re-running the whole sweep
with it exported — a comparison **across boots**, which `CLAUDE.md` forbids.
Arms now carry an env mapping and every row records it.

### `GGML_CUDA_GRAPH_OPT` — MEASURED 2026-08-24, NOT RESOLVED

Off unless asked for, and never asked for here:

```c
static bool enable_graph_optimization = [] {
    const char * env = getenv("GGML_CUDA_GRAPH_OPT");
    return env != nullptr && atoi(env) == 1;      // ggml-cuda.cu:4330
}();
```

It further requires CUDA graphs in use and **exactly one device**
(`ggml-cuda.cu:4342`) — both true here. Decode at batch 1 is a long run of small
kernels, which is the case graph optimisation exists for, so this was the
runtime knob most likely to move the number that matters.

RTX 5060 Ti, ctx 98,304, corpus `real-code-deep`, three rounds, arms alternated
within each round, **identical argv on both arms** — the incumbent `ngram-mod`
window — so the variable is the only difference:

| | round 1 | round 2 | round 3 | |
|---|---:|---:|---:|---|
| `graph-opt-off` | 79.4 | 82.3 | 84.6 | spread 6.6 % |
| `graph-opt-on` | 84.0 | 76.6 | 89.3 | spread **16.6 %** |
| paired delta | **+5.8 %** | **−6.9 %** | **+5.6 %** | mean +1.4 % |

**`within noise / inconsistent`** by `harness.paired_deltas`, which resolves an
effect only when it is consistent in sign across rounds *and* above the floor.
It is neither. **And it did not reduce variance either** — the treated arm is the
wider of the two.

> ⚠️ **A null here means "no effect OR not applied".** Nothing in argv, the boot
> banner or the log echoes `GGML_CUDA_GRAPH_OPT` back, so there is no independent
> confirmation llama.cpp read it. Both readings are consistent with this data and
> the register must not collapse them.

*Raw: `results/graph-opt-98304-blackwell.jsonl`, 6 rows. Eleven knobs untried —
`GGML_CUDA_DISABLE_FUSION`, `GGML_CUDA_CUBLAS_COMPUTE_TYPE`,
`GGML_CUDA_REGISTER_HOST`, `GGML_CUDA_NO_PINNED`, `GGML_OP_OFFLOAD_MIN_BATCH`
and the rest are single- or multi-GPU knobs listed but not swept.*

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
| `-b` / `-ub` | yes, and **re-measured at depth 2026-08-23** — see below | it is **both**, and the exchange rate is bad |
| `-fa on` / `off` | yes | `on` required; `off` loses residency |
| `-sm tensor` | yes | single GPU, no effect |
| `--no-repack`, `--no-op-offload`, `--load-mode`, `--no-host`, `--swa-full`, `--no-kv-unified` | yes | all confirmed inert, as report 16 predicted |

*Raw: `results/sweep-threads*.jsonl`, `results/sweep-batch*.jsonl`,
`results/kv-layers-16k.jsonl`, `results/kv-depth-levers.jsonl`.*

### `-ub` at the served window — swept 2026-08-23. The VRAM is real and costs too much

From the RTX 3090 scan's *present-and-never-set* list, where it is described as
*"the single knob that sizes the worst-case compute buffer"* — the reserve pass
builds the prompt-processing graph at `n_tokens = min(n_ctx, n_ubatch)`. **The
mechanism is correct. The exchange rate is what kills it.**

**Step 1 — what it buys**, `bench/ubatch_preflight.py`, one boot per value at
ctx 98,304, read back from `llama_context: n_ubatch = N` rather than assumed:

| `-ub` | compute buffer | free after load |
|---:|---:|---:|
| 256 (ours) | 472.27 MiB | 825 MiB |
| 128 | 428.27 MiB | 869 MiB |
| 64 | 406.27 MiB | 891 MiB |

**A 4× cut returns 66 MiB.** That is the whole prize, and it is not enough to
matter where it was wanted: the DFlash2 arms finish with 45–376 MiB free and are
unreliable there ([`CORRECTIONS.md` §26](../reports/CORRECTIONS.md)); 66 MiB
moves them to 111–442, the same band.

**Step 2 — what it costs**, `results/ubatch-98304.jsonl`, three paired rounds
against `ngram-mod` on **`UD-IQ2_XXS` at ctx 98,304** -- `ngram-mod` is the
decoder every worker profile runs, though none serves that artifact at that
depth:

| arm | rounds | vs `-ub 256` |
|---|---|---|
| `ub-256-base` | 106.3 · 108.3 · 107.8 | baseline |
| `ub-128` | 100.6 · 101.8 · **108.0** | −3.7 % [−5.9, +0.2] — **inconsistent, see below** |
| **`ub-64`** | 91.8 · 92.3 · 93.0 | **−14.0 % [−14.8, −13.7] RESOLVED** |

**`-ub 64` loses 14 % of decode to buy 77 MiB.** Nothing on this page is worth
that trade.

**The `ub-128` third round is the interesting part, and it is not noise in the
usual sense.** All three of its boots are byte-identical in the log —
`n_ubatch = 128`, compute buffer `428.27 MiB`, `projected to use 8827 MiB vs
10919`, `will leave 2091 >= 768`, `11069 MiB free` at boot. Yet `free_after`,
sampled while the server is still running, reads **759 · 757 · 1,214 MiB**. The
457 MiB the third round had spare was not allocated differently by us; **it was
released by something else on the machine.** That round also ran 6 % faster.

This is the first direct evidence in this project that the desktop's VRAM
occupancy — the ledger's *"1,650–2,200 MiB, the largest untouched lever on this
machine"* — moves a measured rate on an otherwise identical configuration. It
does not license a conclusion about `-ub 128`; it disqualifies that round.

**Also worth noting: `-ub` moves the n-gram statistics**, which nothing
predicted. `ngram-mod`'s mean accepted length falls 20.35 → 19.0 → 17.57 and its
decline rate goes 33.8 % → 22.1 % → 54.2 % across 256/128/64. Unexplained.

## Six flags read from source before spending GPU time — 2026-08-22

**Read, not measured.** Six agents each traced one flag into the code that
*consumes* it, after two sweeps that day were designed against a misreading and
measured nothing. Three of the six turn out to be provably inert in our
configuration, which saves three GPU rounds.

### ☠️ fp16 recurrent state — DO NOT ATTEMPT. It would not fail, it would lie

**Read from source 2026-08-23, no GPU round spent.** The RTX 3090 scan lists
*"fp16 Gated-DeltaNet recurrent state (`--mamba-ssm-cache-dtype float16`)"* as a
**small-patch** whose seam is *"`recurrent_type_r` / `recurrent_type_s` are
`GGML_TYPE_F32` literals at the call site"*. The literals are real —
`src/llama-model.cpp:2316-2317` and `:2335-2336` — and the prize is real too: our
boot log records `llama_memory_recurrent: size = 748.12 MiB … S (f32): 720.00
MiB`, so halving it returns **~360 MiB**, five times what `-ub` returns.

**Do not change them.** Qwen3.8's recurrent layers run
`ggml/src/ggml-cuda/gated_delta_net.cu`, and that file:

- contains **zero** occurrences of `GGML_TYPE_F32` — there is no type check of
  any kind on the state tensor;
- casts it unconditionally — `const float * s_d = (const float *) src_state->data;`
- computes every stride as `nb / sizeof(float)`.

So flipping the literal produces a server that **boots, allocates 374 MiB
instead of 748, reports a healthy `65+0`, and decodes at a plausible rate while
the kernel reads the state array at twice its real span.** No assertion fires,
no error is logged, and the only symptom is wrong output.

**This is the project's north-star failure mode in its purest form**, and it is
worse than the other do-not-sweep entries on this page: those produce a bad
number you can see. This one produces a good number that is false.

Reaching fp16 state here means writing a templated or converting DeltaNet
kernel — **new-backend effort, not a patch** — and the scan's own effort rating
is wrong by two tiers. Recorded so the 360 MiB does not tempt the next reader
the way it tempted this one.

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

### `--spec-draft-p-min` — swept 2026-08-22. **Null, and the bound was still too generous**

> ✅ **Measured.** `results/sweep-p-min.jsonl`, 9 rows, three paired rounds:
> 0.10 → **+2.2 % [−0.3, +6.2]**, 0.25 → **+1.5 % [−1.6, +7.1]** — both
> inside the floor with the sign flipping. **The counters are the result:**
> at `0.10` every per-implementation counter is byte-identical to the
> baseline, so the early-stop **never fired once**; at `0.25` it fired on
> 2.2 % of calls. The algebraic bound below is correct **and was still too
> generous** — on this workload the selector's confidence sits above 0.10
> essentially always. Designing arms above a proven worst-case bound was
> necessary and **not sufficient**.

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
> 🔴 **And it does not survive the trip.** At ctx 65,536 the optimum moves to
> **`16` (+67.5 % RESOLVED)** and `24` becomes a null; the shipped `12` loses
> at both depths. [`CORRECTIONS.md` §22](../reports/CORRECTIONS.md).
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

### 🔴 2026-08-24 — the prediction above landed, and nothing has ever set the flag

**Every server this project has launched runs at `xhigh` with an unlimited
thinking budget**, because the template supplies both and nothing overrides them.
Read out of a boot log, not inferred:

```
init: chat template, example_format: '<|im_start|>system
Reasoning effort is set to xhigh. Please think carefully through the task,
validate key assumptions, consider plausible alternatives, and prioritize
correctness, consistency, and clarity in the final answer.
init: chat template, thinking = 1
srv eval_llama_c: reasoning budget: tokens=-1
```

The client sends no `reasoning_effort` field — the level comes from the
template's own default. And:

| | sets a reasoning flag? |
|---|---|
| `worker-5060ti.ps1` | **no** |
| `worker-iq2s-2slot.ps1` | **no** |
| `worker-iq2s-fast.ps1` | **no** |
| `worker-iq2s-quality.ps1` | **no** |
| `worker-iq2xxs-deep.ps1` | **no** |
| `bench/dflash2_arena.py` | **no — zero references** |

**So does every measurement on this page, and every real-task run.**

**The section above predicted this on 2026-08-18** — *"an external review reports
xHigh taking 15 minutes where medium takes 3"* — and the four real-task runs of
2026-08-24 all landed in that band:

| artifact + decoder | wall | outcome | files |
|---|---:|---|---:|
| `IQ2_XXS` `dflash2+ngram` | **537.7 s** | FAIL | 0 |
| `Q2_K_XL` `draft-mtp` | **855.8 s** | FAIL | 0 |
| `Q2_K_XL` `draft-mtp+ngram` | **947.2 s** | FAIL | 0 |
| `Q2_K_XL` `dflash2+ngram` | **1,019.3 s** | WINDOW_BOUND | 0 |

9 to 17 minutes each, `reasoning_content` dominating the stream, zero files
changed four times out of four.

> ⚠️ **This is a hypothesis with a named mechanism, not a result.** No run at a
> lower effort has been made on these artifacts, so "xhigh is why they failed"
> is unproven. What IS established: the flag exists
> (`--reasoning-effort minimal|low|medium|high|xhigh`, plus
> `--reasoning-budget N`), nothing here has ever set it, and the level in force
> is the most expensive one the template offers.

**The two items under *Never tried* below are now the highest-value untried
things on this page**, and the first of them costs one flag.

**And the level to try first is `medium`, not `low`.** Artificial Analysis
prices the three levels of this model on both of its indices
([`researchs/artificial-analysis`](../researchs/artificial-analysis/README.md)),
and they disagree about where the cost sits:

| Qwen3.8-27B | Intelligence Index | **Agentic Index** |
|---|---:|---:|
| `xhigh` | 52 | **51** |
| `medium` | 44 | **50** |
| `low` | 43 | 44 |

**On the agentic axis — the one this project's metric sits on — `xhigh` to
`medium` costs one point and `medium` to `low` costs six.** The general axis
is the other way round. Full-precision through an API, so the ranking may
transfer and the absolute numbers do not.

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

## The 2026-09-01 research sweep — five flag levers, all null or already right

Issue #67. Artifact `Qwen3.8-27B-NVFP4-MTP-VERY-LOW`, `-sm tensor -ts 7429,15346`,
`-ctk/-ctv q4_0`, `-ub 1024`, **ctx 16,384**, build 10499. Every arm rotated so it
takes each position in the run exactly once, one boot discarded first, four
warm-up generations before three measured. **Repeatability 0.6 %**, established by
measuring the baseline in two independent sweeps (62.72 and 63.11, identical
output hashes).

| lever | tried | result | evidence |
|---|---|---|---|
| `--spec-draft-n-max` | 2 / **3** / 4 | **3 keeps it.** 57.20 / **62.72** / 60.86; 3 wins all three rounds against both. 4 drafts 4,100 and accepts 55.2 % where 3 drafts 3,489 and accepts 65.9 % | `scratchpad/sweep_nmax.json` |
| `--spec-draft-p-min` | 0.0 / 0.7 | **Rejected.** Slower, and **it changes the emitted text** under `temperature 0, top_k 1, seed 42` | issue #67 |
| `--spec-type` order | `draft-mtp,ngram-mod` / reversed | **Exactly nothing.** Identical draft counts (9,528 / 6,512) and identical output hashes at every rep | `scratchpad/sweep_bsampling.json` |
| `--spec-draft-backend-sampling` | default (on) / `--no-` | **Costs nothing either way: 63.11 vs 63.18**, identical draft counts and identical hashes. The flag is `(default: enabled)`, so the served profile already had it on; turning it off only silences the warning. **That the fallback happens at all was already recorded** — [results README](README.md) line 112, from the split-mode work — this row adds only that the fallback is free | `scratchpad/sweep_bsampling_server.log` |
| `CUDA_SCALE_LAUNCH_QUEUES` | unset / 2x / 4x | **Nothing.** Prefill 1010.69 / 1021.33 / 1010.14, and not sign-consistent. Round 0 gives 988.61 / 988.49 / 987.64 across three different environments | `scratchpad/sweep_queues.json` |

**`GGML_CUDA_FA_ALL_QUANTS` is `OFF` in all four local builds and that is correct
for us.** `ggml/src/ggml-cuda/CMakeLists.txt:119-123` still compiles
`fattn-vec-instance-q4_0-q4_0.cu` in the off branch; our `q4_0`/`q4_0` KV is not
one of the "additional" types the flag gates.

**The verdicts above are at 16,384 and do not transfer to the served 147,456** —
`draft-mtp` is +81 % at 16K and −71 % at 131,072 on the same artifact.

## Build 10499 -> 10729, and PR #27140 — measured 2026-09-01

Issue #67. Same artifact, same argv, same corpus, three binaries rotated across
three rounds. Each binary self-identifies by commit, and the harness recorded the
executable path and size it actually launched, because this project has already
invalidated one build comparison where the harness named one binary and ran another.

| arm | build | commit | prefill mean | decode mean |
|---|---|---|---|---|
| served | 10499 | `1deefcca3` + two DFlash2 commits | 954.30 | 61.53 |
| upstream | 10729 | `458681e1d` | 943.44 | 63.12 |
| upstream_fix | 10730 | `7e8864187` (= `458681e1d` + PR #27140) | 964.07 | 63.58 |

**The newer build is +2.58 % decode, not +26 %.** The contested claim is not
reproduced. Prefill does not move. **All three binaries emit byte-identical greedy
text** (`6a632a00cc76`, `6b47d54a7dcc`, `855b386fdbea`).

**PR #27140 is null here.** `upstream_fix` against `upstream` is the only
one-variable comparison in the table — one file, 129 lines — and rounds 1 and 2 are
973.58 / 977.24 against 976.72 / 972.68. Only the cold round 0 differs. The PR
reports 74 -> 1,182 tok/s on 2x RTX 3090; **we were already at ~990 without it**,
and the patch's own comment scopes the path it bypasses to Ampere.

Upstream loads the NVFP4 artifact and drives `draft-mtp` unmodified; no arm errored.

**Build note.** Upstream master will not compile here at full parallelism: nvcc's
`cicc` died with `0xC0000005` on `fattn-mma-f16-instance-ncols1_16-ncols2_2.cu`
with 20 jobs against 26.6 GB free RAM. `--parallel 6` completed both trees.
