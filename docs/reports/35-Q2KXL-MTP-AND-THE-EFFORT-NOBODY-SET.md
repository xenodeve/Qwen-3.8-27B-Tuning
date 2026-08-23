# 35 — Q2_K_XL, the MTP head that was already there, and the effort nobody had set — 2026-08-24

**Continues [report 34](34-BLACKWELL-BOUGHT-HEADROOM-NOT-SPEED.md), which ends
with the native `sm_120a` build in place.** Everything here happened after that,
on the same night. Registers hold the numbers; this is the argument.

> **Read the banner on every page in [`../results/`](../results/README.md) before
> quoting a rate from it.** Everything below was measured at
> `reasoning_effort: xhigh` unless it says otherwise, because that was the
> template's default and nothing had ever overridden it — which is itself one of
> the findings.

---

## 1. What was asked, and the order it happened in

1. Continue with a bigger quantisation — `UD-Q2_K_XL` against the `UD-IQ2_XXS`
   the worker serves.
2. Try the MTP head, since `Q2_K_XL` turned out to carry one.
3. Try `--spec-draft-n-max 7`, the ceiling the ledger had flagged and nobody
   had set.
4. Find out what reasoning effort the model actually runs at.
5. Make `medium` the default everywhere.
6. Produce a recommended config, and serve it.

Each answer changed the next question, and two of them contradicted a
recommendation this session had already made.

---

## 2. Four configurations, one task, zero files changed

`xeno-skills#306` in a throwaway clone, ctx 98,304, native `sm_120a`, `q4_0` KV,
one run each. Later two more at `n_max 7`.

| artifact | decoder | `n_max` | outcome | ctx high-water | wall | files |
|---|---|---:|---|---:|---:|---:|
| `UD-IQ2_XXS` | `dflash2+ngram` | 4 | FAIL | 69,401 | 537.7 s | 0 |
| `UD-Q2_K_XL` | `dflash2+ngram` | 4 | **WINDOW_BOUND** | **98,303** | 1,019.3 s | 0 |
| `UD-Q2_K_XL` | `draft-mtp` | 3 | FAIL | 85,782 | 855.8 s | 0 |
| `UD-Q2_K_XL` | `draft-mtp+ngram` | 3 | FAIL | 82,696 | 947.2 s | 0 |
| `UD-Q2_K_XL` | `dflash2+ngram` | **7** | FAIL | 87,390 | **762.3 s** | 0 |
| `UD-Q2_K_XL` | `draft-mtp+ngram` | **7** | **WINDOW_BOUND** | 98,537 | 1,481.3 s | 0 |

**Six configurations. Two artifacts two bpw classes apart. Four decoders. Zero
files changed, six times out of six.** On task success this stopped being a
decoder question after the third row.

### The two artifacts failed differently, and the better one failed better

`UD-IQ2_XXS` **generated to the 8,192-token request cap twice and never emitted
an edit.** Its transcript opens with the model firing POSIX at PowerShell:

```
$ ls -la
Get-ChildItem: A parameter cannot be found that matches parameter name 'la'.
$ ls -la . 2>/dev/null || dir /B
Out-File: Could not find a part of the path 'D:\dev\null'.
```

`UD-Q2_K_XL` opened with a correct cmdlet on its first command, and **terminated
every generation** — longest 4,811 tokens. It failed by *filling the window*:
79,930 → 83,934 → 90,792 → 92,905 → **98,303 with `truncated = 1`**, the first
truncation in this project's records.

**That is the opposite of what an external ladder describes**
([`../researchs/superalesha-quant-ladder/`](../researchs/superalesha-quant-ladder/README.md)),
where the 2-bit failure is *"the model stops stopping"* and `Q2_K_XL` blew a
262,144 ceiling with a **single 255,755-token generation**. Ours never exceeded
4,811. **Same outcome name, different mechanism — and this report nearly wrote
"reproduced" before checking.**

---

## 3. The MTP head was already in the file, and nobody had run it that way

`docs/results/02-decoders.md` carries `draft-mtp` at **+81 % @16K and −71 %
@131,072**, and records on the same page why neither could have used a baked-in
head:

> Can `draft-mtp` run on `UD-IQ2_S` alone? **No.** *"model doesn't contain MTP
> layers"* — the weights are a separate 1.3 GB file passed with `-md`

`UD-Q2_K_XL` is not that artifact:

```
llama_model_loader: - kv 28: qwen35.nextn_predict_layers u32 = 1
create_tensor: loading tensor blk.64.nextn.eh_proj.weight
common_speculative_init_result: creating MTP draft context against the
    TARGET model '...Qwen3.8-27B-UD-Q2_K_XL.gguf'
spec common_specu: adding speculative implementation 'draft-mtp'
```

`n_layer_all = 65` against `UD-IQ2_XXS`'s 64; 866 tensors against 851. **So
`--spec-type draft-mtp` with no `-md` is a configuration this project had never
run**, and every earlier figure paid 564 MiB for a sidecar head.

### It returns 743 MiB, not the 1,394 removing the sidecar suggests

| | `dflash2+ngram` | `draft-mtp` |
|---|---:|---:|
| model, CUDA0 | 8,630.57 | **8,965.31** |
| target KV | 1,728.00 | 1,728.00 |
| MTP draft KV | — | 384.00 |
| RS | 748.12 *(n_max 4)* | 598.50 *(n_max 3)* |
| compute | 472.27 | 472.27 + 82.01 |
| separate drafter | 1,393.90 | **0** |
| **total on CUDA0** | **12,973** | **12,230** |
| **free of 15,172** | 2,199 | **2,942** |

**The model buffer itself grows 334.74 MiB once the head is used**, and `--fit`
raises its own target from 768 to 1,234 MiB for the 466 MiB MTP context. A first
estimate said 1,394 MiB back; it was wrong and is corrected here rather than
dropped.

---

## 4. `--spec-draft-n-max 7` helps one drafter and hurts the other

`common.h:325` defaults `n_max` to **3**. `speculative.cpp:989` caps it at
`block_size - 1`, and the boot log prints **`block_size=8`** for DFlash2 — so
**7**. Every DFlash2 figure this project holds was taken at **4**, which the
ledger records as *"chosen without knowing either number"*, with two independent
reviews calling it the largest unclaimed lever on the list.

Both arms accepted 7 with no `clamping to` warning, and the recurrent state came
out at **1,197.00 MiB** = `149.62 × (1 + 7)` — confirming the formula at the
ceiling and not only at 4.

| decoder | `n_max` | outcome | ctx high-water | wall | acceptance |
|---|---:|---|---:|---:|---|
| `dflash2+ngram` | 4 | WINDOW_BOUND | 98,303 | 1,019.3 s | 0.36–0.49 |
| `dflash2+ngram` | **7** | FAIL | 87,390 | **762.3 s** | 0.37–0.44 |
| `draft-mtp+ngram` | 3 | FAIL | 82,696 | 947.2 s | **0.48–0.61** |
| `draft-mtp+ngram` | **7** | WINDOW_BOUND | 98,537 | **1,481.3 s** | 0.38–0.44 |

**DFlash2 gets 25 % off the wall clock and stops saturating the window. MTP runs
56 % slower and its acceptance falls.** The mechanism is in the metadata:
`qwen35.nextn_predict_layers = 1` — the head predicts **one** token ahead, so
asking it for seven produces drafts that are mostly rejected while the verify
cost is paid anyway. DFlash2's `block_size = 8` makes 7 its natural maximum.

---

## 5. Decode rate across five arms, bucketed — because a single median lies

Rate tracks generation length hard on this machine (19.68 tok/s at 324 tokens,
62.85 at 8,192 on one arm), and the arms did not produce the same mix of turn
lengths. Median tok/s per bucket, sample count in brackets:

| arm | all | <500 | 500–2k | 2k–8k | ≥8k |
|---|---:|---:|---:|---:|---:|
| `dflash2+ngram` n4 | 36.01 *(29)* | 34.5 *(15)* | **38.2** *(9)* | 33.0 *(5)* | — |
| `dflash2+ngram` n7 | 36.88 *(20)* | 38.8 *(14)* | 36.0 *(3)* | **33.5** *(2)* | 33.4 *(1)* |
| `draft-mtp` n3 | **40.33** *(22)* | 43.5 *(17)* | 32.8 *(2)* | 30.2 *(2)* | 26.2 *(1)* |
| `draft-mtp+ngram` n3 | 39.03 *(20)* | **54.4** *(12)* | 36.3 *(5)* | 30.7 *(2)* | 26.6 *(1)* |
| `draft-mtp+ngram` n7 | 35.74 *(34)* | 41.0 *(19)* | 33.8 *(8)* | 29.7 *(7)* | — |

**MTP wins the short turns decisively — 54.4 against 38.8, +40 %. DFlash2 wins
the long ones.** The crossover is around 500–2,000 tokens.

**Three reasons this is not a verdict.** One unpaired session per arm, on
different prompts. The overall medians span **35.74–40.33, a 13 % range**,
against a noise floor measured up to **9.8 %** at this depth — only the
short-turn gap clears it. And **tok/s did not predict wall clock**:
`dflash2+ngram n7` finished the same task in 762.3 s against
`draft-mtp+ngram`'s 947.2 while decoding slower.

---

## 6. The finding the developer found: every server ever launched ran at `xhigh`

Asked what reasoning effort the model defaults to. Read out of a boot log:

```
init: chat template, example_format: '<|im_start|>system
Reasoning effort is set to xhigh. Please think carefully through the task,
validate key assumptions, consider plausible alternatives, ...
init: chat template, thinking = 1
srv eval_llama_c: reasoning budget: tokens=-1
```

The client sends no `reasoning_effort` field. And nothing overrode it anywhere:

```
worker-5060ti.ps1        no reasoning flag
worker-iq2s-2slot.ps1    no reasoning flag
worker-iq2s-fast.ps1     no reasoning flag
worker-iq2s-quality.ps1  no reasoning flag
worker-iq2xxs-deep.ps1   no reasoning flag
bench/dflash2_arena.py   zero references
```

**So every figure in `docs/results/` and every real-task run was taken at the
most expensive setting the template offers, and no page said so.**

### The register had predicted it six days earlier

[`../results/05-runtime-flags.md`](../results/05-runtime-flags.md), written
2026-08-18:

> *"An external review of this model reports **xHigh taking 15 minutes where
> medium takes 3** for 90 % of the result — a difference our probe was far too
> short to see."*

The four real-task runs came in at **537.7 / 855.8 / 947.2 / 1,019.3 s** — 9 to
17 minutes, all inside that band, `reasoning_content` dominating the stream,
zero files changed four times out of four.

The sweep that made this look covered was **Q4, n=2, a short tool probe, 6/6
pass**. It could not separate the levels, and the page already said so.

### Which level, and why the obvious answer was wrong

Artificial Analysis prices the three levels of this model on both its indices
([`../researchs/artificial-analysis/`](../researchs/artificial-analysis/README.md)):

| Qwen3.8-27B | Intelligence Index | **Agentic Index** |
|---|---:|---:|
| `xhigh` | 52 | **51** |
| `medium` | 44 | **50** |
| `low` | 43 | 44 |

```
Intelligence   xhigh -> medium   -8      medium -> low   -1
Agentic        xhigh -> medium   -1      medium -> low   -6
```

**The two indices disagree about where the cost is.** This project's metric sits
on the agentic axis, where `medium` is one point below `xhigh` and `low` is six
below that. A question asked an hour earlier — *"medium or low?"* — was posed as
if the two were the same direction. They are not.

**`medium` is the served default from 2026-08-24.** All five worker profiles,
the new sixth, and `dflash2_arena.server_argv` set it; the arena records `effort`
on every row; all nine results pages carry a banner naming the level their
numbers were taken at **and** stating that the default has since changed.

**Unmeasured:** no run on any of these artifacts exists at any level but `xhigh`.
The change is a decision, not a result.

---

## 7. A projection said 163,840 would fit. A boot said otherwise

The recommended config was proposed at **ctx 163,840** from buffers measured at
98,304. It does not load:

```
common_params_fit_impl: cannot meet free memory target of 1522 MiB,
                        need to reduce device memory by 154 MiB
load_tensors: offloaded 64/66 layers to GPU
```

Two CPU layers at depth are not a rounding error —
[`../results/04-context-depth.md`](../results/04-context-depth.md) measures
`AD-IQ1_M` at `65+1` decoding **6.08 tok/s** against 26.50 resident.

**Three buffers that look fixed scale with context:**

| buffer | 98,304 | 131,072 | 147,456 | 163,840 | rate |
|---|---:|---:|---:|---:|---|
| target KV | 1,728.00 | 2,304.00 | 2,592.00 | 2,880.00 | 18.00 KiB/token |
| **target compute** | **472.27** | **616.27** | **688.27** | **777.57** | ~0.0047 MiB/token |
| **MTP draft KV** | **384.00** | **512.00** | **576.00** | **640.00** | 4.00 KiB/token exactly |
| **MTP compute** | **82.01** | **98.01** | **106.01** | **114.01** | ~0.0005 MiB/token |

**Only the first is the one everybody knows.** The other three add ~290 MiB per
32,768 tokens — enough to turn a 1,790 MiB projection into a 154 MiB shortfall
two rungs later. `-ub` does not change between these boots; the compute buffer
grows with the window anyway.

> **A VRAM projection is not a residency verdict.** One boot settles it in under
> a minute, and `--fit` will silently **spill** rather than refuse — which reads
> as success in every field except the layer count.

**Measured ceiling: 147,456, 66/66 resident**, 13,526 MiB of 15,172, leaving
1,646. 131,072 is the safer rung at 2,078.

---

## 8. What is being served, and the first production data on it

`scripts/worker-q2kxl-mtp.ps1` — `UD-Q2_K_XL`, ctx **147,456**,
`draft-mtp,ngram-mod` with **no `-md`**, `n_max` at its default 3,
`--reasoning-effort medium`, and the same Blackwell-SASS guard as
`worker-5060ti.ps1`.

**33 turns of real Claude Code use through it, read from the server's own log:**

| | |
|---|---|
| decode | median **37.36 tok/s**, range 25.03–65.49 |
| generated tokens | median **95**, max 2,028, total 9,235 |
| **hit the 8,192 cap** | **0 of 33** |
| prefill | median 297.0 tok/s, largest prompt **40,264 tokens** |
| draft acceptance | median **0.5165**, range 0.3365–0.8000 |
| mean accepted length | 3.08 |
| context high-water | **75,841 of 147,456**, `truncated = 0` |
| prefix-cache hits | **25**, LCP similarity up to 0.984 |

**Three things this settles that the benchmark could not.**

**It stops.** Zero of 33 turns reached the cap, where `UD-IQ2_XXS` hit it twice
inside one task. The one 8,192 runaway seen on `draft-mtp` during benchmarking
has not recurred in production use.

**A median generation of 95 tokens is production proof of what the bucket table
implies**: an agent loop is overwhelmingly short turns, so the short-turn column
is the one that matters and the corpus figure of 87.72 tok/s describes almost
nothing an operator experiences.

**Acceptance holds on real work at 0.5165**, inside the 0.48–0.61 band the
benchmark measured — the repetitive-prompt worry that hangs over every n-gram
verdict in `02-decoders.md` does not apply to this arm.

---

## 9. What is open

- **No task-success number exists for `UD-Q2_K_XL` on this machine.** Six real-task
  runs, zero completions, across two artifacts and four decoders.
- **No measurement at `medium` exists on any of these artifacts.** The default
  changed on a decision.
- **The decoder choice rests on one unpaired session per arm.** The arena's
  paired protocol would settle it in about 25 minutes.
- **`xeno-skills#306` may simply be too hard for this model class.** Six
  configurations is enough evidence to stop treating it as a decoder question and
  start asking whether the task is a fair probe.
- **147,456 has no throughput measurement**, only a residency one.
- **The desktop still holds ~1.8 GB of VRAM** — Wallpaper Engine, Discord, Steam,
  Edge, two NVIDIA overlays. The Intel UHD 770 exists and is idle. Moving the
  display there is the only untried change that costs nothing and may also
  explain the boot-to-boot spread `CORRECTIONS.md` §27 leaves unexplained.

*Issues [#40](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/40),
[#41](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/41),
PR [#42](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/pull/42).*
