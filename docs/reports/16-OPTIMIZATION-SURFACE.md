# The Optimization Surface — Complete Catalogue of Everything That Can Be Tuned

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **Date:** 2026-08-20 UTC+7 · **exhaustive by request.** Nothing is omitted for
> being unpromising; items judged inert are listed with the prediction and the
> reason, so a wrong prediction is falsifiable rather than invisible.
>
> **Why this exists.** For two days this project tuned one speculative decoder —
> MTP — across twelve configurations, while ten other decoders sat one line away
> in the same help text (report 15 §2). That was not a gap in effort. It was a
> gap in **the map**: the search ran deep inside the one layer that was known and
> never enumerated the layers beside it. This document is the map.
>
> **Grounding.** Every flag name, default and description below is parsed from
> `llama-server --help` on build **10472** commit **`60eeeb608`**, not recalled.
> Coverage is computed by scanning every file in `C:\AI\qwen38-tuning\scripts\`
> and `bench\*.py` for flags actually passed. Predictions are labelled as
> predictions and are **not** measurements.

---

> **Correction, 2026-08-21.** `output_contract_pct` is the **pass** rate —
> `100 * (attempts_seen - contract_violations) / attempts_seen` — not the violation
> rate. Text written on 2026-08-20 read it backwards. The figures are unchanged;
> their direction is. Higher is better.

## 0. Method and headline

```text
option entries parsed from --help        248   (323 distinct long-option spellings)
ever passed by this project               38   (15 %)

  common       82 entries    19 used
  server       91 entries     6 used     <- largest group, least touched
  speculative  41 entries     8 used
  sampling     34 entries     5 used
```

**15 % is not a criticism of the work done.** The residency cliff was found, and
it is worth more than most of the remaining 85 % put together. The number says
where the *unexplored* surface is, and the answer is: not evenly spread, and not
where the search has been.

### The lens that decides every prediction below

On a 12 GB card serving a 27B model, one mechanism dominates: **a layer on the
GPU is worth roughly twice a layer on the CPU**, and the split the loader
chooses decides everything else. Measured: 33+32 → 13.1 tok/s, 61+4 → 21.8,
65+0 → 41.3. Any knob is therefore judged on two questions:

1. Does it move the split?
2. If not, is its effect larger than the **13.6 %** restart-drift floor?

Most knobs fail both. The ones that pass are named in §17.

---

## 1. The map — sixteen layers

| # | layer | tunable via | our coverage | predicted size here |
|---|---|---|---|---|
| 1 | Model / weights | artifact choice | 8 families, 20 artifacts | **very large** (13→69 tok/s) |
| 2 | Weight quantization | artifact choice, `--override-kv` | broad | **very large** (via residency) |
| 3 | Adapters & steering | `--lora`, `--control-vector` | none | small–medium, quality not speed |
| 4 | Tensor placement | `-ot`, `-cmoe`, `-ncmoe`, `-ts`, `-sm`, `-dev` | `-ncmoe` once | **large — direct on the cliff** |
| 5 | Memory & loading | `-ngl`, `-fit*`, `-lm`, `--no-host`, `--rpc` | `-ngl`, `-fit*` | medium |
| 6 | KV cache | `-ctk/-ctv`, `-nkvo`, `--swa-full`, `--kv-unified`, checkpoints, `--cache-*` | type + offload | **large at depth** |
| 7 | Context geometry | `-c`, RoPE/YaRN family, `--context-shift`, `--keep` | `-c` only | **the >128K goal lives here** |
| 8 | Attention & kernels | `-fa`, `--op-offload`, `--repack`, `--check-tensors` | none forced | unknown, probably small |
| 9 | Decoder / speculation | `--spec-type` (11) + 40 spec flags | 4 of 11 | large offloaded, ~0 resident |
| 10 | Batching & slots | `-b`, `-ub`, `-np`, `-cb`, `--threads-http` | `-b`/`-ub` swept | small, settled |
| 11 | CPU & scheduling | `-t`, `-tb`, masks, `--numa`, `--prio`, `--poll` | `-t`/`-tb` swept | small |
| 12 | Sampling | 34 flags incl. **grammar / JSON schema** | 5 flags | **large, unexplored** |
| 13 | Prompt & chat protocol | template, reasoning family, `--prefill-assistant` | `--jinja` only | **large** |
| 14 | Server & session | prompt cache, slots, save/restore, sleep | none | medium |
| 15 | Build & runtime | compile flags, version, PRs, forks, other engines | one pinned build | unknown, high variance |
| 16 | Host & agent loop | OS, driver, power, and the client above the server | partial | **largest per unit of effort** |

---

## 2. Layer 1 — the model itself

Not a flag. The artifact is chosen, and it is the single largest lever measured:
`13.12 → 69.33 tok/s` across report 15 §1.

| sub-axis | status | prediction / note |
|---|---|---|
| parameter count | 9B / 20B / 27B / 35B-A3B measured | settled — smaller is faster, quality is the constraint |
| dense vs MoE | both measured | MoE is fastest raw (+78–80 %) but **measured at 227–363 MiB free**, below the reserve — directional only |
| base vs fine-tune | Ornith is a Qwen3.5 fine-tune | **untested as a variable.** A drafter matched to the base is not automatically matched to a fine-tune |
| instruct vs coder-specialised | never varied | *Predicted medium.* A coder-tuned model of the same size may change `p1` more than any runtime flag. `Qwen3-Coder-30B-A3B` is named in the research and never fetched |
| context length as trained | 262 K claimed | untested against actual retrieval — see §8 |

---

## 3. Layer 2 — weight quantization

| sub-axis | status | prediction / note |
|---|---|---|
| bit depth Q1…Q8 | measured across 6 levels | settled: the artifact that fits at `65+0` wins |
| scheme (K / IQ / UD / ternary / binary / MXFP4) | measured | UD-IQ beats plain K at equal size |
| **who quantized it** | Unsloth vs AtomicChat vs Prism measured | AtomicChat `AD-IQ2_XXS` +4.0 %, `AD-IQ1_M` **usable where V3 IQ1_M is not** — the requantizer is a real variable |
| imatrix / calibration corpus | **never chosen** | *Predicted medium on quality, zero on speed.* We inherit whatever corpus the publisher used. Cannot be varied without re-quantizing locally |
| AWQ / GPTQ / QAT | **never tried** | *Predicted low priority* — llama.cpp support is uneven and GGUF is the pipeline |
| `--override-kv KEY=TYPE:VALUE` | **never used** | *Predicted small but diagnostic.* Rewrites model metadata at load: rope base, advertised context, template markers. It reads and writes the exact things reports 06 §0 item 7 is guessing about (did `UD-Q2_K_XL` keep its MTP head) |
| `--check-tensors` | never used | *Predicted zero performance,* non-zero safety: validates tensor data on load. Worth one run on any 1-bit artifact behaving strangely |

---

## 4. Layer 3 — adapters and steering (entirely untouched)

| flag | status | prediction / note |
|---|---|---|
| `--lora FNAME` | never used | *Predicted: not applicable now, potentially large later.* A LoRA trained on this project's own corpus would target `p1` directly — the metric — rather than tok/s. Costs training, not tuning |
| `--lora-scaled FNAME:SCALE` | never used | scale a loaded adapter |
| `--lora-init-without-apply` | never used | load without applying; apply via `POST /lora-adapters`. Enables A/B within one boot — **no restart drift** |
| `--control-vector FNAME` | never used | *Predicted small–medium and untested anywhere in this project.* A control vector is a per-layer activation offset. The failure that killed both V3 1-bit arms was **runaway reasoning** — a behavioural mode, which is the class of thing control vectors are for |
| `--control-vector-scaled FNAME:SCALE` | never used | |
| `--control-vector-layer-range START END` | never used | restrict the vector to a layer band |

**Note on `--lora-init-without-apply`:** this project's largest measurement
problem is that comparisons across boots span a 13.6 % drift floor. Anything that
allows two conditions **within one boot** is methodologically valuable out of
proportion to its raw effect. This flag and `POST /props` are the only two such
mechanisms in the whole surface.

---

## 5. Layer 4 — tensor placement (the biggest specific gap)

| flag | default | status | prediction / note |
|---|---|---|---|
| `-ot, --override-tensor <pattern>=<buffer>` | unset | **never used** | ***Predicted large.*** The only knob that controls the governing mechanism at tensor granularity. `--fit` treats layers as indivisible and identical; attention and FFN tensors have very different size-to-work ratios. On an arm at **58+7** or **61+4** the question is which *tensor* moves, not which layer |
| `-cmoe, --cpu-moe` | off | never used | all MoE weights to CPU. *Predicted bad alone* — `-ncmoe` already lost 2.6–6.2 % — but it is the extreme point of a curve we sampled once |
| `-ncmoe, --n-cpu-moe N` | 0 | **used once** | −2.6 % / −6.2 % on the two MoE arms. The research predicted a large win from a wrong size assumption (20.6 GiB vs the actual 10.02) |
| `-sm, --split-mode {none,layer,row,tensor}` | layer | never used | *Predicted zero* — single GPU. **Listed because `tensor` mode is not obviously single-GPU-inert** and one boot settles it |
| `-ts, --tensor-split N0,N1,…` | unset | never used | *Predicted zero* — one device |
| `-mg, --main-gpu INDEX` | 0 | never used | *Predicted zero* — one device |
| `-dev, --device <dev1,dev2,…>` | all | never used | *Predicted zero,* but `--device none` is a clean way to get a CPU-only control arm, which this project has never measured |

**Cheapest decisive test in this layer:** `UD-IQ2_XXS` at 128K sits at 58+7 and
7.84 tok/s. Push the last 7 layers' FFN tensors to CPU while keeping their
attention on GPU. One boot. If split *granularity* rather than byte count is the
binding constraint, this is where it shows.

---

## 6. Layer 5 — memory and loading

| flag | default | status | prediction / note |
|---|---|---|---|
| `-ngl, --gpu-layers N\|auto\|all` | auto | **used** | the number every report reads |
| `-fit, --fit [on\|off]` | on | used implicitly | *Worth one run with `-fit off` and a hand-set `-ngl`.* Every measurement inherits `--fit`'s judgement; nothing has tested against it |
| `-fitt, --fit-target MiB` | — | **swept** | settled; 512 MiB reserve adopted |
| `-fitc, --fit-ctx N` | 4096 | never set | minimum ctx `--fit` may choose. *Predicted small,* but it silently bounds every depth experiment |
| `-lm, --load-mode {auto,none,mmap}` | auto | never set | *Predicted zero on decode, real on load time.* Load time matters here: the queue does 25+ boots per session |
| `--mlock` | — | never used | **DEPRECATED** in favour of `--load-mode` |
| `--mmap / --no-mmap` | — | never used | **DEPRECATED** in favour of `--load-mode` |
| `-dio, --direct-io` | — | never used | **DEPRECATED** in favour of `--load-mode` |
| `--no-host` | off | never used | bypass host buffer, allow extra buffers. *Predicted unknown — undocumented in help beyond one line.* Cheap to try, cheap to revert |
| `--op-offload / --no-op-offload` | on | never used | offload host tensor ops to device. *Predicted small,* but it is exactly the kind of default that matters when 4 layers sit on the CPU |
| `--rpc SERVERS` | unset | never used | *Predicted not applicable* — no second machine. Listed because a second host on the LAN would change the residency problem entirely |

---

## 7. Layer 6 — KV cache

| flag | default | status | prediction / note |
|---|---|---|---|
| `-ctk / -ctv TYPE` | f16 | **measured** | only `f16 / bf16 / q8_0 / q4_0` have a fast kernel; the rest collapse 7× in prefill (report 15 §7.1) |
| **K and V at different types** | — | **never tried** | *Predicted small–medium.* `-ctk q8_0 -ctv q4_0` is legal. V is often more compressible than K; this halves cache growth at possibly lower quality cost than compressing both |
| `-nkvo, --no-kv-offload` | off | **measured** | reaches 65+0 and is still slower (5.26 vs 7.84) — PCIe per token |
| `--swa-full` | false | **never tried** | full-size sliding-window cache. *Predicted: irrelevant unless the arch uses SWA — and Qwen3.8 is hybrid Gated-DeltaNet, so this must be checked, not assumed* |
| `-kvu, --kv-unified` | auto | **never tried** | one shared KV buffer across sequences. *Predicted small at `-np 1`, larger if slots are ever used* |
| `-ctxcp, --ctx-checkpoints N` | **32** | **never tried** | max context checkpoints per slot. ***Predicted medium at depth.*** At 128K the margin is 412–503 MiB; 32 speculative checkpoints is VRAM that could instead be a resident layer |
| `-cms, --checkpoint-min-step N` | 8 | never tried | spacing between checkpoints — the other half of the same cost |
| `-cram, --cache-ram N` | 8192 MiB | never tried | host-side cache cap. *Predicted: reliability at depth, not throughput* |
| `--cache-idle-slots` | on | never tried | saves idle slots to prompt cache. *Predicted zero at one slot* |
| `--cache-prompt / --no-cache-prompt` | **on** | never varied | the panel asked for per-run prefix-cache control and it was not built (report 14 §4). **Every number in this project assumes prompt caching on and none controlled for it** |
| `--cache-reuse N` | **0** | **never tried** | ***Predicted large.*** Reuse a chunk via KV shifting instead of re-prefilling. Attacks the single most expensive thing ever measured here: a broken prefix costs **63 s at 16K and 248 s at 64K** (report 09 §1) |
| `-sps, --slot-prompt-similarity` | 0.10 | never tried | how closely a prompt must match a slot to reuse it. Directly interacts with `--cache-reuse` |
| `-dt, --defrag-thold` | — | never used | **DEPRECATED** |
| `-lcs / -lcd, --lookup-cache-static/dynamic` | unset | never used | persistent n-gram lookup caches. *Predicted small alone, but they are the state behind the three unrun `ngram-*` decoders* — a cold n-gram cache is why `ngram-simple` scored 31 % acceptance |

---

## 8. Layer 7 — context geometry (where the stated goal lives)

**The goal is usable context beyond 128 K.** It has been attacked from two
directions — smaller weights, smaller KV — and both bottom out because KV grows
linearly out of the pool the weights live in. There is a third direction and
**not one flag in it has ever been set.**

| flag | default | status | prediction / note |
|---|---|---|---|
| `-c, --ctx-size N` | from model | **used** | the ladder variable |
| `--rope-scaling {none,linear,yarn}` | from model | **never set** | *Predicted medium.* This project has asked artifacts to hold 262,144 tokens without ever recording which scaling method the loader chose |
| `--rope-scale N` | — | never set | context expansion factor |
| `--rope-freq-base N` | from model | never set | NTK-aware scaling base |
| `--rope-freq-scale N` | — | never set | 1/N expansion |
| `--yarn-orig-ctx N` | model training ctx | never set | |
| `--yarn-ext-factor N` | −1 (auto) | never set | extrapolation mix; 0.0 = full interpolation |
| `--yarn-attn-factor N` | −1 | never set | attention magnitude scaling |
| `--yarn-beta-fast / --yarn-beta-slow` | −1 | never set | interpolation ramp |
| `--context-shift / --no-context-shift` | **disabled** | **never enabled** | ***Predicted large, and shape-changing.*** Turns "hold 256 K of cache" into "keep a moving window over an unbounded conversation". For a coding agent whose turns are append-only that may be the shape actually needed, and it sidesteps the KV-vs-weights competition entirely |
| `--keep N` | 0 | never set | how many prompt tokens survive a shift — with `--context-shift`, this is what protects the system prompt and tool schemas |

**Why both halves are open at once:** RoPE/YaRN changes *retrieval quality* at
depth, and **deep-context retrieval quality has never been measured on anything
but Q4** (report 15 §8). So there is currently no instrument that would detect a
RoPE setting making depth worse. Fix the measurement first, then tune.

---

## 9. Layer 8 — attention and kernels

| flag | default | status | prediction / note |
|---|---|---|---|
| `-fa, --flash-attn [on\|off\|auto]` | **auto** | **never forced** | *Predicted small but uncontrolled.* Every cross-artifact comparison assumes FA resolved the same way on every arm. The loader logs its decision and nobody has read it. **This is a validity problem before it is an optimization** |
| `--repack / --no-repack` | on | never varied | CPU-side weight repacking. *Predicted zero at 65+0, non-zero at 33+32* — exactly the arms whose numbers are least load-bearing |
| `--op-offload` | on | never varied | see §6 |
| `--check-tensors` | off | never used | *Predicted zero speed;* a correctness gate for aggressive quants |
| `--warmup / --no-warmup` | on | never varied | *Predicted zero on steady state.* Relevant only to load-time accounting |
| `--perf / --no-perf` | off | never used | internal libllama timings. *Predicted zero effect, useful instrumentation* |
| build: `FA_ALL_QUANTS` | off | **open** | the Q8 result that closed this cannot test it — `Q8_0` compiles either way. Unlocks `q4_1`/`q5_0`/`q5_1` + asymmetric K≠V; **none run** ([`CORRECTIONS` §29](CORRECTIONS.md)) |
| build: CUDA arch list | prebuilt | never varied | binary ships `500…900`. *Predicted small* — a build targeting only `890` (Ada) could differ, unmeasured |
| backend: CUDA vs Vulkan | CUDA | never compared | *Predicted CUDA wins,* untested |

---

## 10. Layer 9 — decoder / speculation

Full treatment in **report 15 §2**. Catalogued here for completeness.

### 10.1 The decoders

`--spec-type` accepts eleven values. **Four have run.**

| value | status | note |
|---|---|---|
| `none` | ✅ | the control everywhere |
| `draft-mtp` | ✅ | **+46.8 % on Q4 (33+32), −8.8 % on Q2_K_XL (61+4)** — inverts across the cliff |
| `ngram-simple` | ✅ | +1.6 % / +0.8 %, 31 % acceptance — below the floor |
| `ngram-mod` | ~ | preboot snapshot only, no paired result |
| `draft-dspark` | ❌ | attempted, drafter path resolved empty, never launched. **Fixed and not re-run** |
| `draft-dflash` | ❌ | never attempted. **DFlash 2** drafter for Qwen3.8-27B now on disk (1.06 GiB); this binary rejects it |
| `draft-eagle3` | ❌ | never attempted; no checkpoint sought |
| `draft-simple` | ❌ | never attempted — a plain small-model drafter |
| `ngram-map-k` | ❌ | **free — no drafter file, no download** |
| `ngram-map-k4v` | ❌ | **free** |
| `ngram-cache` | ❌ | **free** |

### 10.2 Drafter placement — a whole sub-surface, entirely unused

The drafter has its **own** copy of nearly every placement knob, and this is
the reason MTP's regression happened:

| flag | status | prediction / note |
|---|---|---|
| `-ngld, --gpu-layers-draft` | used (`999`) | |
| `-otd, --spec-draft-override-tensor` | **never used** | ***Predicted medium.*** MTP lost 8.8 % because the head's VRAM displaced six target layers. This flag can put the drafter's tensors on the CPU instead — turning "drafter vs residency" back into "drafter vs nothing" |
| `-cmoed / --spec-draft-ncmoe` | never used | same idea for MoE drafters |
| `-devd, --spec-draft-device` | never used | `--device-draft none` = drafter on CPU entirely |
| `-ctkd / -ctvd` draft KV type | never used | the drafter has its own KV cache and its own quantization for it |
| `-td / -tbd` draft threads | never used | |
| `-Cd / -Crd / --cpu-strict-draft` draft affinity | never used | |
| `--prio-draft`, `--poll-draft` | never used | |
| `--spec-draft-backend-sampling` | on by default | never varied |

**This is the sharpest single omission in the report.** Report 15 §2.1 concluded
"MTP does not pay on a resident target". That conclusion was drawn without ever
trying `-otd` or `-devd`, which exist precisely to stop a drafter from evicting
the target. The conclusion may still hold — but it is currently a claim about
one placement, not about MTP.

### 10.3 Speculation tuning knobs

| flag | default | status | note |
|---|---|---|---|
| `--spec-draft-n-max` | 3 | swept 2–6 | |
| `--spec-draft-n-min` | 0 | swept | **reversed sign between two sweeps** |
| `--spec-draft-p-min` | 0.00 | swept | **reversed sign between two sweeps** |
| `--spec-draft-p-split` | 0.10 | swept | within the floor |
| `--spec-ngram-mod-n-min / -n-max / -n-match` | 48 / 64 / 24 | never set | untuned, and `ngram-mod` never properly run |
| `--spec-ngram-simple-size-n / -m / -min-hits` | 12 / 48 / 1 | never set | the 31 % acceptance was at defaults |
| `--spec-ngram-map-k-*` (3 flags) | 12 / 48 / 1 | never set | decoder itself never run |
| `--spec-ngram-map-k4v-*` (3 flags) | 12 / 48 / 1 | never set | decoder itself never run |
| `--spec-default` | — | never used | "enable default speculative decoding config" — a one-flag baseline nobody has looked at |

**Removed flags, listed so no time is wasted on them:** `--draft`, `--draft-n`,
`--draft-max`, `--draft-min`, `--draft-n-min`, `--spec-ngram-size-n`,
`--spec-ngram-size-m`, `--spec-ngram-min-hits`. The help text still lists these
and they error out.

---

## 11. Layer 10 — batching and slots

| flag | default | status | prediction / note |
|---|---|---|---|
| `-b, --batch-size` | 2048 | **swept** | settled, below the floor |
| `-ub, --ubatch-size` | 512 | **swept** | settled |
| `-np, --parallel N` | auto | used (=1) | *Predicted negative at depth* — each slot costs KV out of the same pool. But an agent that could overlap two tool calls changes the metric, not just the tok/s |
| `-cb, --cont-batching` | on | never varied | *Predicted zero at one slot* |
| `--threads-http N` | −1 | never set | *Predicted zero* — 19 threads for a single local client |
| `--sse-ping-interval` | 30 s | never set | *Predicted zero* |
| `-to, --timeout` | 3600 s | used | |
| `--sleep-idle-seconds` | −1 | never set | *Predicted zero on performance,* useful for an always-on worker that should release VRAM between sessions — which is exactly this machine's usage pattern |

---

## 12. Layer 11 — CPU and scheduling

| flag | default | status | prediction / note |
|---|---|---|---|
| `-t, --threads` | −1 | **swept** | settled at 18 |
| `-tb, --threads-batch` | = threads | **swept** | settled |
| `-C / -Cr, --cpu-mask / --cpu-range` | unset | never set | ***Predicted unknown and asymmetric.*** The i5-13500 has P-cores and E-cores. Windows processor numbering must be discovered first — **a wrong mask looks like an optimization and behaves like a regression**, which is why it has been deferred, not because it is small |
| `--cpu-strict <0\|1>` | 0 | never set | |
| `-Cb / -Crb / --cpu-strict-batch` | = above | never set | separate mask for prompt processing — plausibly *different* from the decode mask, since prefill is throughput-bound and decode is latency-bound |
| `--prio N` | 0 | never set | *Predicted small,* real under contention. The corpus runs unattended while nothing else uses the machine |
| `--prio-batch N` | 0 | never set | |
| `--poll <0..100>` | 50 | never set | busy-wait level. *Predicted small; may matter with CPU layers* |
| `--poll-batch` | = poll | never set | |
| `--numa TYPE` | off | never set | *Predicted zero* — single socket |

---

## 13. Layer 12 — sampling and constrained decoding

**34 entries. Five used.** This is the largest blind spot after the decoder axis.

| flag | default | status | prediction / note |
|---|---|---|---|
| `--temp` | 0.80 | **used** | **temperature 0.6 vs 1.0 was never run** — an open item since report 06 |
| `--top-k` | 40 | used | |
| `--top-p` | 0.95 | used | |
| `--min-p` | 0.05 | used | |
| `-s, --seed` | −1 | used | |
| `--samplers SEQ` | `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature` | **never set** | *Predicted small–medium.* **The order is itself a knob** and nobody has looked at the default chain that has been running under every number in this project |
| `--sampler-seq / --sampling-seq` | `edskypmxt` | never set | short form of the same |
| `--top-n-sigma` | −1 (off) | never set | *Predicted medium for code.* Cuts the tail by standard deviations rather than by rank — reported to help deterministic tasks |
| `--typical, --typical-p` | 1.0 (off) | never set | *Predicted small* |
| `--xtc-probability / --xtc-threshold` | 0.0 / 0.10 | never set | XTC removes top tokens to increase variety — *predicted **harmful** for code.* Listed so it is explicitly ruled out rather than forgotten |
| `--dynatemp-range / --dynatemp-exp` | 0.0 / 1.0 | never set | *Predicted small–medium.* Entropy-adaptive temperature: low temp when confident, higher when not — plausible fit for a model that loops when uncertain |
| `--mirostat / -lr / -ent` | 0 / 0.10 / 5.0 | never set | targets a fixed perplexity; **overrides top-k/top-p/typical**. *Predicted: worth exactly one run against the runaway-reasoning failure, because it directly controls output entropy over a long generation* |
| `--adaptive-target / --adaptive-decay` | off | never set | adaptive-p: select tokens near a target probability. *Predicted unknown — newest sampler here, no local evidence either way* |
| `--repeat-penalty` | 1.00 (off) | never set | ***Predicted directly relevant.*** Both V3 1-bit arms failed by **looping** in the reasoning block. This is the classical anti-loop control and it is **off by default** |
| `--repeat-last-n` | 64 | never set | window for the above — 64 tokens is far shorter than a 30,000-character loop |
| `--presence-penalty / --frequency-penalty` | 0.0 | never set | same family |
| `--dry-multiplier / -base / -allowed-length / -penalty-last-n / -sequence-breaker` | 0.0 (off) / 1.75 / 2 / 64 / defaults | never set | ***Predicted the strongest anti-loop tool here.*** DRY penalises *repeated sequences* rather than repeated tokens, which is the exact shape of the observed failure |
| `--ignore-eos` | off | never used | *Predicted harmful* — would make the truncation problem worse. Listed to rule out |
| `-l, --logit-bias TOKEN(+/-)BIAS` | unset | never used | *Predicted narrow but sharp.* Biasing the closing-think token or the fence token is a one-line fix for a format failure |
| `--grammar GRAMMAR` | unset | **never used** | ***Predicted large — see §13.1*** |
| `--grammar-file FNAME` | unset | never used | |
| `-j, --json-schema SCHEMA` | unset | **never used** | ***Predicted large*** — tool-call structure becomes a decoder guarantee |
| `-jf, --json-schema-file FILE` | unset | never used | |
| `-bs, --backend-sampling` | off (experimental) | never used | *Predicted small speed gain,* moves sampling to GPU. Experimental — check greedy equivalence before trusting |

### 13.1 Grammar is the structural fix for the failure that just killed two arms

Not a speculative suggestion. The same failure, four times:

| observation | source |
|---|---|
| V3 `IQ1_S` — no fenced block in **12 of 12** attempts | report 15 §4 |
| V3 `IQ1_M` — **31 of 53** attempts violated the output contract (**41.5 % PASS rate**), 27 of 53 truncated at 8192 | corpus 2026-08-20 03:54 |
| `Q2_K_XL` tool compliance read as 40 % — actually truncation | report 15 §5 |
| `Q2_K_XL` empty replies **55 of 100** turns | report 15 §6 |

Every one is a **format** failure, not a reasoning failure. `--grammar` and
`--json-schema` make format a property of the decoder rather than a hope about
the prompt: the sampler cannot emit a token the grammar forbids.

**If it works:** V3 `IQ1_S` — 50.55 tok/s, `65+0` at 128 K with 1,436 MiB spare,
the fastest artifact ever measured here — was rejected on format alone. That is
the difference between discarding it and adopting it.

**If it does not work:** still valuable, because it separates two explanations
currently confounded — a model that cannot solve the task, versus a model that
cannot stop talking about it.

---

## 14. Layer 13 — prompt and chat protocol

| flag | default | status | prediction / note |
|---|---|---|---|
| `--jinja / --no-jinja` | on | **used** | |
| `--chat-template JINJA` | from GGUF metadata | **never set** | ***Predicted medium, and a validity problem.*** Every cross-model comparison in report 15 used a **different template**, inherited from each artifact. That is an uncontrolled variable in a paired design |
| `--chat-template-file` | — | never set | |
| `--chat-template-kwargs JSON` | — | never set | |
| `--reasoning-budget N` | **−1 (unrestricted)** | **never set** | ***Predicted large.*** Both V3 1-bit arms produced 19,280–33,871 chars of reasoning and never closed it; 27 of 53 attempts hit the cap. This is the runtime-side control for exactly that |
| `--reasoning-budget-message` | none | never set | text injected before the forced end-of-thinking tag — shapes what the model does with the interruption |
| `-rea, --reasoning [on\|off\|auto]` | auto | never set | *Predicted large on wall-clock.* Turning thinking **off** is untested and this project's metric is tasks per hour, not thoughts per task |
| `--reasoning-effort LEVEL` | template default | **used** | |
| `--reasoning-format FORMAT` | — | never set | how thought tags are extracted. *Predicted medium diagnostic value* — the harness parses `<think>` blocks by hand today (report 04 §7: the greedy grader once counted code inside a think block) |
| `--reasoning-preserve` | template default | **never set** | ***Predicted large and under-appreciated.*** If reasoning is preserved into the next turn's prefix, an arm that thinks 30,000 characters is an arm whose prefix grows 30,000 characters per turn. That reprices every verbose low-bit arm and is invisible in a single-turn benchmark |
| `--prefill-assistant / --no-prefill-assistant` | on | never varied | *Predicted medium — the cheap cousin of `--grammar`.* Put the opening fence in the model's mouth |
| `--skip-chat-parsing` | off | never used | forces a pure content parser. *Predicted zero performance, useful for debugging what the model actually emitted* |
| `--special` | off | never used | show special tokens in output — *pure diagnostic, and this project has spent hours guessing at token-level behaviour* |
| `-r, --reverse-prompt` | unset | never used | halt at a string. *Predicted: a blunt stand-in for a stop condition* |
| `--spm-infill` | off | never used | *Predicted not applicable* — FIM ordering, not chat |
| `--tools TOOL1,…` | off (experimental) | never used | server-side built-in tools. *Predicted: out of scope* — tools are client-side by design here, and the help warns against untrusted environments |
| `--tools-runtime` | none | never used | |
| `--mcp-servers-config / --mcp-servers-json` | unset | never used | *Out of scope,* same warning |
| `-ag, --agent` | off | never used | enables CORS proxy + all built-in tools. *Explicitly not wanted* |

---

## 15. Layer 14 — server, session and observability

| flag | default | status | prediction / note |
|---|---|---|---|
| `--slot-save-path PATH` | disabled | **never used** | ***Predicted medium and novel.*** Persist and restore a slot's KV cache to disk. A warm start across boots would remove cold prefill — the 11-minute 256 K prefill — from the measurement *and* from real use |
| `--slots / --no-slots` | on | never used | slots monitoring endpoint — *free observability the harness does not read* |
| `--metrics` | off | never used | Prometheus endpoint. *Predicted zero performance;* would give per-request timings the harness currently reconstructs by hand |
| `--props` | off | never used | change global properties via `POST /props`. **One of only two mechanisms for A/B within a single boot** (see §4) — methodologically valuable against the 13.6 % drift floor |
| `--log-prompts-dir PATH` | disabled | never used | *Pure diagnostic.* Would have settled the "is it censoring or failure" question directly instead of by inference |
| `-lv, --verbosity` | — | **used** | |
| `--log-file / --log-colors / --log-prefix / --log-timestamps / --log-disable / -v` | — | partly used | housekeeping |
| `--offline` | off | never used | forces cache, blocks network. ***Predicted directly useful:*** `-hf` performing an online etag check per launch stalled an unattended queue for 11 minutes. This flag prevents that failure class outright |
| `--reuse-port` | off | never used | *Predicted zero,* and would defeat the port-lock discipline |
| `--api-key / --api-key-file / --ssl-*` | unset | never used | *Not applicable* — binds `127.0.0.1` |
| `--cors-*` (4 flags) | permissive | never used | *Not applicable* — local only |
| `--path / --api-prefix / --ui* / --webui*` (6 flags) | — | never used | *Not applicable* — no browser client |
| `--models-dir / --models-preset / --models-max / --models-autoload` | disabled | never used | router-server mode. ***Predicted worth a look***: it is the built-in mechanism for holding more than one model, which is what "fast arm + strong arm routing" (§16) would need |
| `--alias / --tags` | — | `--alias` used | labelling |
| `--embedding / --rerank / --pooling / --embd-normalize / --embd-gemma-default` | off | never used | *Not applicable* — not an embedding workload |
| `--mmproj*, --image-*, --mtmd-batch-max-tokens, --media-path, --vision-*` (9 flags) | — | `--no-mmproj-auto` used | *Not applicable* — text only. `--no-mmproj-auto` is used only to stop `-hf` pulling a projector |
| `--fim-qwen-*` / `--gpt-oss-*-default` (7 flags) | — | never used | convenience presets that download models. *Not applicable,* but `--fim-qwen-14b-spec` documents a working target+drafter pairing worth reading as an example |
| `-cl, --cache-list`, `--completion-bash`, `-h`, `--version`, `--list-devices`, `--usage` | — | some used | inspection, no effect |

---

## 16. Layers 15–16 — below and above llama.cpp

### 16.1 The runtime itself

| axis | status | prediction / note |
|---|---|---|
| llama.cpp version | pinned b10472 `60eeeb608` | correct discipline. **Changing it invalidates cross-report comparison** the way the Unsloth republish did — so the move is a *second* binary with its own re-measured control, never an upgrade in place |
| unmerged PRs | **PR #27342 (DFlash 2)** | an exact Qwen3.8-27B drafter exists (1.06 GiB, on disk); this build rejects it |
| forks — PrismML | required for Bonsai g128 `Q2_0` | never built |
| forks — `ik_llama.cpp` | never evaluated | low-bit-focused fork; *predicted plausible gain on IQ1/IQ2, entirely unmeasured* |
| other engines — vLLM / SGLang / TensorRT-LLM / ExLlamaV3 | **none evaluated** | *Predicted: different residency and batching behaviour on 12 GB.* ExLlamaV3 in particular targets exactly this class of card. No local evidence either way |
| compile flags | prebuilt binary | `FA_ALL_QUANTS` off in both builds, **reopened** (§29); CUDA arch list untested |

### 16.2 Host and OS

| axis | status | prediction / note |
|---|---|---|
| GPU driver version | uncontrolled | never recorded per run — *should be in the env snapshot and is not* |
| WDDM vs TCC | WDDM (consumer card, TCC unavailable) | the eviction hypothesis was **measured and closed** (report 14 §3) |
| Windows power plan / clock behaviour | uncontrolled | *Predicted small but real* — could contribute to the 13.6 % drift floor that everything is measured against |
| PCIe link width/state | uncontrolled | *Predicted meaningful only for CPU-offloaded arms,* which is where `--no-kv-offload` lost |
| OS page cache across boots | raised by the panel, judged narrower | affects load time and CPU-resident layers only (report 14 §3) |
| background GPU consumers | monitored via free-VRAM-at-boot | the 9,326–10,732 MiB spread is partly this |

### 16.3 The agent loop above the server

Not a llama.cpp layer, and the highest leverage per unit of effort, because the
metric is **verified accepted tasks per hour** and the loop decides how many
attempts a task gets.

| axis | status | prediction / note |
|---|---|---|
| `max_tokens` the client sends | **measured — it is a treatment** | 3072→8192 moved `bonsai-g64` 15/31→27/31 and `iq1m` 20/31→27/31 on identical weights |
| retry policy | one evidence-assisted retry | `p2` measured 0.20–0.625 against the research's assumed 0.93 |
| escalation policy | 90 s charged per escalation | never varied |
| task decomposition | never varied | corpus is ten self-contained functions; **structurally cannot see cross-file drift** (report 14 §2, still the largest untaken panel item) |
| context compaction | never varied | |
| prefix discipline in the client | the largest measured cost (63 s / 248 s), untested against a real client | **check first when OpenCode is wired up** |
| **model routing** | **never tried** | *Predicted large.* A 69 tok/s arm that fails 40 % and a 41 tok/s arm that fails 10 % may beat either alone. `--models-dir` (§15) is the runtime-side support for it |

---

## 17. Priority — everything above, ordered

Ordered by expected value **on this machine**, not by novelty. Items 1–7 are all
single boots or single runs.

| # | action | layer | cost | why |
|---|---|---|---|---|
| 1 | `--grammar` / `--json-schema` on V3 `IQ1_S` | 12 | 1 corpus | the fastest artifact ever measured here was rejected on format alone |
| 2 | `--reasoning-budget` + `--dry-multiplier` + `--repeat-penalty` on both V3 1-bit arms | 12–13 | 1 screen each | three independent anti-loop controls, all **off by default**, against a measured loop |
| 3 | `-ot, --override-tensor` on an arm at 58+7 | 4 | 1 boot | the only direct control over the governing mechanism |
| 4 | `-otd` / `-devd` — re-test MTP with the drafter off the GPU | 9 | 1 paired round | the −8.8 % verdict was measured at one drafter placement out of several |
| 5 | `--context-shift` + `--keep`, then RoPE/YaRN at depth | 7 | 2 boots | the stated >128 K goal, from the direction never tried |
| 6 | `--cache-reuse` + `-sps` | 6 | 1 boot | attacks the largest single measured cost in the project |
| 7 | `ngram-map-k`, `ngram-map-k4v`, `ngram-cache` | 9 | 1 boot each | free — no drafter file, no download |
| 8 | `-fa on` vs `off` explicitly, on two arms | 8 | 2 boots | **validity before optimization** — currently uncontrolled |
| 9 | `--chat-template` pinned across arms | 13 | rerun a paired round | removes an uncontrolled variable from every cross-model comparison |
| 10 | `--reasoning-preserve` + prefix growth on a verbose arm | 13 | 1 stability run | may reprice every low-bit arm |
| 11 | `-ctxcp` lowered at 128 K | 6 | 1 boot | 32 checkpoints of VRAM where the margin is 412 MiB |
| 12 | `--slot-save-path` warm start | 14 | 1 boot | removes cold prefill from both the benchmark and real use |
| 13 | temperature 0.6 vs 1.0, and `--top-n-sigma` | 12 | 2 corpus runs | an open item since report 06, never run |
| 14 | model routing (fast arm + strong arm) | 16 | harness work | changes the metric, not the tok/s |
| 15 | a second binary with DFlash 2 | 15 | source build | high ceiling; **unpins the baseline — do it last, in parallel, never in place** |
| 16 | `ik_llama.cpp` / ExLlamaV3 evaluation | 15 | days | unknown, and a different residency model |

**Items 1, 2 and 7 together cost less than one afternoon and touch three
different layers.** That ratio is the point of this document.

---

## 18. Explicitly judged inert — recorded so the judgement is falsifiable

These are predicted to have no useful effect **on this machine, in this
workload**. They are listed rather than dropped, because a prediction that is
written down can be proved wrong.

| flag(s) | why predicted inert |
|---|---|
| `-sm`, `-ts`, `-mg` | single GPU. *(But `-sm tensor` is not obviously single-GPU-inert — one boot would settle it)* |
| `--numa` | single socket |
| `--rpc` | no second host |
| `--threads-http`, `--sse-ping-interval`, `--reuse-port` | one local client |
| `--cors-*`, `--api-key*`, `--ssl-*`, `--path`, `--api-prefix`, `--ui*`, `--webui*` | binds `127.0.0.1`, no browser |
| `--embedding`, `--rerank`, `--pooling`, `--embd-*` | not an embedding workload |
| `--mmproj*`, `--image-*`, `--mtmd-*`, `--media-path`, `--vision-*` | text only |
| `--fim-*`, `--gpt-oss-*-default` | convenience downloaders, not knobs |
| `--xtc-*`, `--ignore-eos` | predicted **harmful** for deterministic code generation |
| `--spm-infill` | FIM ordering, not chat |
| `--tools`, `--tools-runtime`, `--mcp-*`, `-ag` | tool handling is client-side by design; help warns against untrusted use |
| `--cont-batching`, `--cache-idle-slots`, `-kvu` | single slot — revisit if `-np > 1` is ever adopted |
| `--warmup`, `--check-tensors`, `--perf` | affect load time or diagnostics, not steady-state decode |
| `--mlock`, `--mmap`, `--direct-io`, `-dt` | **deprecated** — use `--load-mode` |
| `--draft`, `--draft-n`, `--draft-max`, `--draft-min`, `--draft-n-min`, `--spec-ngram-size-n`, `--spec-ngram-size-m`, `--spec-ngram-min-hits` | **removed** — they error out |

---

## 19. The procedure that would have caught the decoder gap

Five minutes, and it should run whenever the binary changes.

```sh
# 1. what does the runtime actually offer?
llama-server --help > surface.txt

# 2. what do we actually pass?
grep -ohE '(?<![\w-])--?[a-zA-Z][a-zA-Z0-9-]*' scripts/* bench/*.py | sort -u > used.txt

# 3. the gap IS the backlog
comm -23 <(grep -oE '\-\-[a-z0-9][a-z0-9-]+' surface.txt | sort -u) used.txt
```

Three habits go with it:

1. **Enumerate the enum.** Any flag whose value is a list — `--spec-type`,
   `--rope-scaling`, `--load-mode`, `--samplers`, `--split-mode`,
   `--reasoning-format` — is a *set of experiments*, not a setting. Read the
   whole list before tuning one member of it.
2. **Read defaults as decisions.** `--reasoning-budget -1`, `--context-shift
   disabled`, `--ctx-checkpoints 32`, `--cache-reuse 0`, `--repeat-penalty 1.00`,
   `--dry-multiplier 0.00`, `-fa auto`, `--cache-prompt on` are all choices this
   project made by not making them.
3. **Sub-surfaces mirror.** The drafter has its own copy of nearly every
   placement flag (§10.2). Wherever a component is duplicated in the flag
   namespace, the experiment space is duplicated with it.
