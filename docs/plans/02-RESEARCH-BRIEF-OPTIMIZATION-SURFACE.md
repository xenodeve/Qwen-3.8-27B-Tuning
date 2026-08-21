# Research Brief — The Complete Optimization Surface of a Local Coding Worker

**For an external research agent. Self-contained: assumes no access to this machine.**
**Date of brief:** 2026-08-20 · **Author:** the local benchmarking agent

---

## 0. What I want from you

I run a local coding-agent worker on one consumer GPU. Over two days I measured
about 20 model artifacts and found one dominant mechanism. Then I discovered I
had been optimizing **one knob inside one layer** while ten sibling options sat
one line away in the same help text.

So the request is not "make my model faster." It is:

> **Enumerate and research the complete surface of things that can be changed in
> a local LLM serving stack, and for each one tell me what is known, what the
> expected size of the effect is on the specific hardware below, and what the
> exact artifact or build requirement is.**

I have already built the catalogue from my runtime's own `--help`. **§5 is that
catalogue.** I want you to (a) verify it, (b) fill in what I could not know from
a help text — published results, papers, checkpoints, known bugs — and (c) find
the layers I *still* have not thought of.

**Include the items I predicted are worthless.** I would rather be told my
prediction is wrong than have a small real effect quietly dropped because it
looked unpromising. Every item in §5 marked *predicted inert* is a claim I am
asking you to confirm or refute.

---

## 1. The machine — exact specification

| component | value |
|---|---|
| **GPU** | NVIDIA GeForce RTX 4070 SUPER, **12,282 MiB** VRAM (≈12 GB GDDR6X) |
| GPU compute capability | **8.9** (Ada Lovelace, AD104) |
| GPU driver | **610.88** |
| GPU power limit | 220 W · max SM clock 3,105 MHz · max mem clock 10,501 MHz |
| PCIe | **Gen 4 × 16** (max link) |
| **CPU** | Intel Core **i5-13500** — 14 cores (6 P + 8 E), **20 threads**, base 2.5 GHz |
| CPU features | AVX2, AVX_VNNI, F16C, FMA, BMI2 — **no AVX-512** |
| **RAM** | **48 GB** (2 × 24 GB) DDR5 @ **7000 MT/s**, dual channel |
| Board | ASUS TUF GAMING B760M-PLUS WIFI |
| **Storage** | WD_BLACK SN850X 1 TB NVMe (system, models) + Samsung MZVL2512HDJD 512 GB NVMe |
| Free disk | ~29 GB — **a real constraint on how many artifacts can be held** |
| **OS** | Windows 11 Pro, build 26200 (10.0.26200.9168) |
| Display driver model | **WDDM** — consumer card, TCC mode unavailable |

**Critical consequence of this hardware:** a 27B model at 2 bits is ~8–9 GiB and
the KV cache at 128K is ~2–4 GiB, out of **one 12 GB pool**. Weights and cache
compete directly. That single fact explains most of my results.

---

## 2. The software stack — exactly what is pinned

| component | value |
|---|---|
| Inference server | **llama.cpp `llama-server`**, build **10472**, commit **`60eeeb608`** |
| Compiler | Clang 20.1.8, Windows x86_64 |
| Backend | CUDA, `ARCHS = 500,610,700,750,800,860,890,900`, `USE_GRAPHS = 1` |
| CPU backend | `LLAMAFILE=1, OPENMP=1, REPACK=1` |
| Model format | GGUF only |
| Client chain | Claude Code → Xeno → OpenClink → OpenCode → llama-server (OpenAI-compatible HTTP) |
| Serving | single process, `127.0.0.1:8080`, one slot, single stream |

**The pin is deliberate.** Every number I have was measured on this binary.
Changing it invalidates cross-comparison, so any recommendation that requires a
different build must say so explicitly and I will treat it as a separate,
re-baselined track.

---

## 3. The workload and the metric

**Workload:** an autonomous coding agent. Multi-turn, append-only conversation,
tool calls with nested JSON schemas, reads and edits files, runs tests, repairs,
returns evidence. Prompts grow monotonically within a session.

**Metric — this is the only number that decides anything:**

> **verified accepted coding tasks per hour**

A task counts only if the generated code **executes and passes its tests**. Not
tok/s. Not benchmark scores. Throughput and capability are tracked as two
separate axes, because I have four artifacts that tie at 27/31 accepted and
differ **2.9×** in wall clock.

Secondary requirement, currently a hard goal: **usable context beyond 128K.**

---

## 4. What I have already measured — do not re-derive this

Condensed so you can skip it. Numbers are medians of paired boots.

### 4.1 The governing mechanism — the residency cliff

A layer on the GPU is worth roughly **twice** a layer on the CPU. `llama-server`
prints its split as `<gpu>+<cpu>` layers (65 total for this model).

| artifact | split | tok/s |
|---|:--:|---:|
| `UD-Q4_K_XL` (16.7 GiB) | **33+32** | 13.1 |
| `UD-Q2_K_XL` (9.9 GiB) | **61+4** | 21.8 |
| `UD-IQ2_XXS` (8.4 GiB) | **65+0** | 41.3 |
| `Bonsai-27B-Q1_0` (3.5 GiB) | **65+0** | 69.3 |

**The last 4 CPU layers cost about half the throughput.** This dominates every
flag, every KV setting and every speculation setting combined.

### 4.2 Results that will save you effort

- **Speculative decoding inverts across the cliff.** MTP is **+46.8 %** on the
  CPU-offloaded Q4 target and **−8.8 %** on a resident Q2 target, because the
  draft head's VRAM moves the split from 61+4 to 55+10. Draft acceptance was
  78–99 % in *both* cases — acceptance does not predict the outcome here.
- **Best speedup ever measured from speculation: 1.47×.** The 3.2× on this
  machine came from **residency**, not speculation.
- **Speculation does not change output.** Byte-identical greedy across every
  speculative config tested.
- **KV type buys residency, not speed.** `q4_0` vs `q8_0` is +474 % when it moves
  the split, and **−1.6 %** when the arm already fits at 65+0.
- **Only `f16`, `bf16`, `q8_0`, `q4_0` have a fast KV kernel** — prefill ~1,180
  tok/s. `q5_1`, `q5_0`, `q4_1`, `iq4_nl` collapse to **144–170** tok/s.
- **`--no-kv-offload` reaches 65+0 and is still slower** (5.26 vs 7.84 tok/s at
  128K) — it moves the whole cache across PCIe every token.
- **Restart-drift floor is 13.6 %.** Free VRAM at boot varies 9,326–10,732 MiB
  and `--fit` follows it; the same control config spans 32.4–42.5 tok/s across
  25 boots. **Any effect below 13.6 % is noise here.**
- **`-t`, `-tb`, `-b`, `-ub`, `--fit-target` are swept and settled.** Remaining
  effects are below the floor. Do not propose re-sweeping them.
- **Output token budget is a treatment, not a detail.** Raising `max_tokens`
  3072→8192 moved two different artifacts from 15/31 and 20/31 to **27/31** with
  no other change. Median reasoning length spans **59 → 2,811 characters across
  quantizations of the same model**; one artifact reached 37,000.
- **Deepest fully-resident context** (measured 2026-08-20, `q4_0` KV):
  `Bonsai-27B-Q1_0` **262,144** at 65+0 · `Ornith-9B-Q6` **262,144** ·
  V3 `IQ1_S` **196,608** · V3 `IQ1_M` **163,840**.
- **The current failure mode is format, not reasoning.** Aggressive low-bit
  artifacts loop inside their reasoning block and never emit a fenced code
  block: 12/12 attempts on one artifact, 31/53 on another, 27/53 truncated at
  the 8192-token cap.

### 4.3 Where previous research led me wrong — please avoid these failure modes

I was given a prior deep-research report. Four of its claims cost me real time:

| claim | reality on this machine |
|---|---|
| MoE CPU-expert offload will be a large win; the artifact is 20.6 GiB | The artifact is **10.02 GiB**. The proposed config was a **−46 to −48 %** loss |
| retry success probability `p2 ≈ 0.93` | measured **0.20–0.625** |
| "no exact Qwen3.8 DFlash drafter exists" | true for DFlash v1; **DFlash 2 has an exact Qwen3.8-27B GGUF drafter** |
| modern decoders promise **2–5×** speedups | best measured here **1.47×**, and negative on resident targets |

**So:** state sizes as exact byte counts, distinguish *measured on similar
hardware* from *paper claim*, and never assume a feature that exists upstream is
in my pinned build.

---

## 5. The catalogue — every layer, with my prediction, for you to verify

This is built from `llama-server --help` on my exact build: **248 option
entries** (323 distinct spellings), of which I have ever used **38 (15 %)**.

For each layer: what I want researched. **Items marked ⚠️ are ones I predicted
are worthless — I want those checked too.**

### Layer 1 — model / weights
Explored: 8 families, 20 artifacts (Qwen3.8-27B, Qwen3.6-35B-A3B MoE, Ornith 9B
and 35B-A3B, Ternary Bonsai 27B, gpt-oss-20b).
**Research:** Which currently-released open-weight models best fit ~12 GB while
being *coder-specialised*? I have never tested a coder-tuned model of this size.
Include exact HF repo, file, byte count, and whether a `base_model` matched
drafter exists. Also: does a fine-tune (e.g. an Ornith on a Qwen base) break
drafter compatibility with the base model's drafter?

### Layer 2 — weight quantization
Explored: Q1…Q8, K/IQ/UD/ternary/binary/MXFP4, three different requantizers.
**Research:** (a) does the **imatrix / calibration corpus** used by the publisher
measurably change code-task accuracy at 1–2 bits, and is there a published
comparison? (b) Is `--override-kv` the correct way to inspect/alter GGUF metadata
such as advertised context length and MTP-head presence? (c) Current state of
AWQ/GPTQ/QAT **in GGUF** — worth pursuing or a dead end for llama.cpp?

### Layer 3 — adapters and steering ⚠️ *(I predicted "not applicable now")*
Flags: `--lora`, `--lora-scaled`, `--lora-init-without-apply`,
`--control-vector`, `--control-vector-scaled`, `--control-vector-layer-range`.
**Research:** My models fail by **looping inside the reasoning block**. Control
vectors are a per-layer activation offset — is there published evidence they can
suppress a runaway-reasoning / repetition mode? Are there published control
vectors for Qwen-family models? Separately: `--lora-init-without-apply` allows
switching adapters **within a single server process** — given my 13.6 % restart
drift, is this usable as a within-boot A/B mechanism?

### Layer 4 — tensor placement (my largest suspected gap)
Flags: `-ot/--override-tensor`, `-cmoe`, `-ncmoe`, `-sm/--split-mode`, `-ts`,
`-mg`, `-dev`.
**Research:** `--fit` treats layers as indivisible and identical. Attention and
FFN tensors have very different size-to-work ratios. **What are the known-good
`--override-tensor` patterns for keeping attention tensors on GPU while moving
FFN tensors to CPU**, and is there published data on the throughput effect for a
dense model one or two layers short of full residency? Exact regex/pattern syntax
and real examples please. ⚠️ Also: is `--split-mode tensor` genuinely inert on a
single GPU, or does it change intra-device behaviour?

### Layer 5 — memory and loading
Flags: `-ngl`, `-fit`, `-fit-target`, `-fit-ctx`, `-lm/--load-mode`, `--no-host`,
`--op-offload`, `--rpc`.
**Research:** What does `--no-host` actually do (the help is one line)? Is
`--load-mode` known to change anything beyond load time? ⚠️ `--rpc` — I have a
second NVMe machine idle; is llama.cpp RPC mature enough that offloading layers
to a second host over LAN beats offloading them to local CPU, and what is the
measured penalty?

### Layer 6 — KV cache
Explored: `-ctk`/`-ctv` types, `--no-kv-offload`.
Untouched: **K and V at different types**, `--swa-full`, `--kv-unified`,
`--ctx-checkpoints` (default **32**), `--checkpoint-min-step`, `--cache-ram`,
`--cache-reuse` (default **0**), `--slot-prompt-similarity`, `--lookup-cache-static/dynamic`.
**Research:** (a) Is asymmetric KV quantization (`-ctk q8_0 -ctv q4_0`) an
established practice, and is there quality data? (b) **`--ctx-checkpoints`
defaults to 32 per slot — how much VRAM does that actually cost at 128K–256K?**
At my margins that could be an entire resident layer. (c) `--cache-reuse` with
KV shifting: what is it actually capable of reusing, and does it help when a
prefix is *edited in the middle* rather than truncated? **This is my single most
expensive measured cost: one broken prefix costs 63 s at 16K and 248 s at 64K.**
(d) Does Qwen3.8's hybrid Gated-DeltaNet architecture use sliding-window
attention at all — i.e. is `--swa-full` even meaningful for it?

### Layer 7 — context geometry (my current hard goal)
Untouched entirely: `--rope-scaling {none,linear,yarn}`, `--rope-scale`,
`--rope-freq-base`, `--rope-freq-scale`, `--yarn-orig-ctx`, `--yarn-ext-factor`,
`--yarn-attn-factor`, `--yarn-beta-fast`, `--yarn-beta-slow`,
`--context-shift` (default **disabled**), `--keep`.
**Research:** (a) For models advertising 262K natively, is any RoPE/YaRN override
ever correct, or does overriding always degrade a model that was trained long?
(b) **What is the measured retrieval-quality cost of YaRN extrapolation past the
trained length**, by depth position? (c) `--context-shift` for an append-only
agent conversation: what exactly is evicted, does `--keep` reliably protect the
system prompt and tool schemas, and what breaks? This would change my problem
from "hold 256K of cache" to "keep a moving window", which sidesteps the
weights-vs-cache competition entirely. (d) Is there published evidence on
low-bit quantization degrading *long-context retrieval specifically* more than
aggregate benchmarks show?

### Layer 8 — attention and kernels
Flags: `-fa [on|off|auto]` (I have **always** left it at `auto`), `--repack`,
`--op-offload`, `--check-tensors`, `--warmup`, plus build-time options.
**Research:** (a) On what basis does `-fa auto` decide, and can it silently
resolve differently across artifacts — which would invalidate my cross-artifact
comparisons? (b) Is building with a **single** CUDA arch (`89` for Ada) rather
than the shipped 8-arch fat binary known to change performance? (c) ⚠️ CUDA vs
Vulkan backend on an Ada consumer card — I predicted CUDA wins; is that still
true in 2026?

### Layer 9 — decoder / speculative decoding
My runtime accepts exactly these 11 values:
`none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark,
ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache`.
**I have run 4.** Also entirely unused: the drafter's **own** placement flags —
`-otd/--spec-draft-override-tensor`, `-cmoed`, `--spec-draft-ncmoe`,
`-devd/--spec-draft-device`, `-ctkd`/`-ctvd` (draft KV type), `-td`, `-Cd`,
`--prio-draft`, `--poll-draft`, `--spec-draft-backend-sampling`.
**Research:**
1. For **each** of the 11 types: what does it actually do, does it need a
   separate checkpoint, and for **Qwen3.8-27B specifically** does an exact
   drafter exist (repo + file + byte count + revision SHA)?
2. The three `ngram-*` variants need no checkpoint. What distinguishes
   `ngram-simple` / `ngram-map-k` / `ngram-map-k4v` / `ngram-cache` / `ngram-mod`,
   and which suits **code editing** (high literal repetition from the file being
   edited) rather than prose?
3. **`--spec-draft-override-tensor` / `--spec-draft-device none`** — my
   conclusion that "MTP does not pay on a resident target" was measured with the
   drafter on the GPU. Can the drafter be placed on CPU and still pay? Any
   published data?
4. **DFlash 2** — I found `z-lab/Qwen3.8-27B-DFlash2-GGUF` (Q4_K_M
   1,143,006,752 B; Q8_0 2,056,414,752 B; BF16 3,860,293,152 B) and the vendor
   says it needs llama.cpp **PR #27342**, unmerged. Is that PR merged now? Is
   there a prebuilt Windows CUDA binary anywhere? Will the **stock** `draft-dflash`
   loader read a DFlash 2 GGUF, or reject the architecture?
5. Any known **greedy divergence** bugs for speculative decoding against
   *quantized* targets, by spec type and by commit. I need bit-exactness.

### Layer 10 — batching and slots ⚠️ *(predicted small)*
`-b`, `-ub` swept and settled. `-np/--parallel`, `-cb/--cont-batching`,
`--threads-http`, `--sse-ping-interval`, `--sleep-idle-seconds` untouched.
**Research:** For a **single-stream** agent, is there any case where `-np > 1`
plus continuous batching wins — e.g. overlapping a tool call's follow-up? And
what does each extra slot cost in KV at 128K?

### Layer 11 — CPU and scheduling ⚠️ *(predicted small)*
`-t`/`-tb` swept. Untouched: `--cpu-mask`, `--cpu-range`, `--cpu-strict`, the
separate `-batch` variants, `--prio`, `--prio-batch`, `--poll`, `--poll-batch`,
`--numa`.
**Research:** My CPU is **hybrid — 6 P-cores + 8 E-cores, 20 threads.** (a) How
does Windows number P vs E cores for an affinity mask, and how do I discover it
reliably? (b) Is pinning llama.cpp to P-cores only a known win for the
CPU-resident-layer case, and by how much? (c) Should the **prompt-processing**
mask differ from the **decode** mask — prefill is throughput-bound, decode is
latency-bound, and llama.cpp exposes separate flags for exactly this.

### Layer 12 — sampling and constrained decoding (34 flags, 5 used)
Used: `--temp`, `--top-k`, `--top-p`, `--min-p`, `--seed`.
Untouched: `--samplers` (the **order** is a knob), `--sampler-seq`,
`--top-n-sigma`, `--typical-p`, `--xtc-*`, `--dynatemp-range`/`--dynatemp-exp`,
`--mirostat`/`-lr`/`-ent`, `--adaptive-target`/`--adaptive-decay`,
`--repeat-penalty` (**default 1.00 = off**), `--repeat-last-n` (64),
`--presence-penalty`, `--frequency-penalty`, the whole `--dry-*` family
(**default off**), `--logit-bias`, `--grammar`, `--grammar-file`,
`--json-schema`, `--json-schema-file`, `--backend-sampling`, `--ignore-eos`.
**Research — this is my highest-priority layer after decoders:**
1. **`--grammar` / GBNF for enforcing "exactly one fenced Python block".** Give me
   a working GBNF. What is the **throughput cost** of grammar-constrained
   sampling in llama.cpp, and does it interact badly with speculative decoding?
2. **Anti-loop.** My failure is *repeated sequences* inside a reasoning block —
   19,280–33,871 characters, never closing. Compare **DRY** vs `--repeat-penalty`
   vs `--top-n-sigma` vs Mirostat for this specific mode, with published evidence
   if it exists. Note `--repeat-last-n` defaults to **64 tokens**, which is far
   shorter than the loops I see — what window is appropriate?
3. Does a **repetition penalty degrade code quality** (code is legitimately
   repetitive: `self.`, `return`, indentation)? What do practitioners actually
   set for code generation?
4. `--samplers` default order is
   `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`. Does the
   order matter measurably, and is there a recommended order for deterministic
   code?
5. ⚠️ I predicted **XTC and `--ignore-eos` are actively harmful** for code.
   Confirm or refute.
6. `--backend-sampling` (experimental, GPU-side sampling) — speedup, and does it
   preserve greedy equivalence?

### Layer 13 — prompt and chat protocol
Used: `--jinja`, `--reasoning-effort`, `-n`.
Untouched: `--chat-template`, `--chat-template-file`, `--chat-template-kwargs`,
`--reasoning-budget` (**default −1 = unrestricted**), `--reasoning-budget-message`,
`--reasoning [on|off|auto]`, `--reasoning-format`, `--reasoning-preserve`,
`--prefill-assistant`, `--skip-chat-parsing`, `--special`, `--reverse-prompt`.
**Research:**
1. **`--reasoning-budget N`** — what exactly happens at the boundary? Is the
   thinking block force-closed with a valid tag, and does the model then produce
   a usable answer, or garbage? `--reasoning-budget-message` suggests a message
   is injected — what should it say?
2. **`--reasoning off`** — for a coding worker measured in tasks/hour, is
   disabling thinking on a reasoning model a known net win or loss?
3. **`--reasoning-preserve`** — if reasoning is kept in the history, an artifact
   that thinks 30,000 characters grows its prefix by 30,000 characters per turn.
   What is the default for Qwen-family templates, and what do agent frameworks
   normally do?
4. **`--chat-template`** — every cross-model comparison I ran used a *different*
   template inherited from each GGUF. Is normalising the template across models a
   valid way to remove that variable, or does it invalidate each model?
5. `--prefill-assistant` — is prefilling the opening fence a recognised cheaper
   alternative to a grammar?

### Layer 14 — server and session
Untouched: `--slot-save-path`, `--slots`, `--metrics`, `--props`,
`--log-prompts-dir`, `--offline`, `--models-dir`/`--models-preset`/`--models-max`.
**Research:** (a) `--slot-save-path` — can a KV cache be saved and restored
across **process restarts**, and does it survive a different `-c`? A warm start
would remove an 11-minute cold prefill at 256K from both my benchmark and real
use. (b) `--props` allows changing global properties at runtime — **which**
properties, and is it enough to A/B sampling settings inside one boot? (c)
**Router-server mode** (`--models-dir`) — can one llama-server hold two models
and route between them? I want "fast weak model + slow strong model" routing.
⚠️ (d) I judged CORS/SSL/API-key/WebUI/embedding/reranking/multimodal flags
inapplicable — confirm none of them has a performance side effect.

### Layer 15 — build and runtime
**Research:** (a) What has changed in llama.cpp **since build 10472 / commit
`60eeeb608`** that affects low-bit dense models, KV cache, or speculative
decoding on Ada? Name commits or releases. (b) **`ik_llama.cpp`** — is it still
maintained, and are its low-bit/IQ kernels measurably faster than mainline on
Ada? (c) **ExLlamaV3, vLLM, SGLang, TensorRT-LLM on a single 12 GB consumer
card** — which are actually viable for a 27B at ≤2 bits, and what is their KV
memory behaviour compared to llama.cpp? (d) Windows-specific gotchas for any of
the above.

### Layer 16 — host, OS, and the agent loop
**Research:** (a) On Windows/WDDM with a consumer card, what host-level settings
measurably affect a compute workload — power plan, "Hardware-accelerated GPU
scheduling", MPO, driver-level shader cache, `CUDA_MODULE_LOADING`? I see a
**13.6 % run-to-run drift** that I currently treat as an irreducible noise floor
and would like to reduce. (b) Does DDR5-7000 vs slower RAM measurably change the
CPU-offloaded-layer case, and is dual-channel bandwidth the binding constraint
there? (c) At the agent-loop level: is there published work on **model routing**
— a cheap fast model attempting first, escalating to a stronger one — with
measured task-completion economics rather than benchmark scores?

---

## 6. Ranked questions — if you can only answer some, answer these

1. **Grammar/GBNF for enforcing a single fenced code block**, plus its throughput
   cost and its interaction with speculative decoding. *(My fastest artifact —
   50.6 tok/s, fully resident at 128K — was rejected purely on output format.)*
2. **The anti-loop comparison**: DRY vs repeat-penalty vs top-n-sigma vs Mirostat
   vs `--reasoning-budget`, against runaway reasoning specifically, for code.
3. **All 11 `--spec-type` values explained**, with exact Qwen3.8-27B checkpoint
   availability, and the status of **DFlash 2 / llama.cpp PR #27342**.
4. **`--override-tensor` patterns** for attention-on-GPU / FFN-on-CPU, with any
   published throughput data for a nearly-resident dense model.
5. **`--context-shift` + `--keep` semantics** for an append-only agent
   conversation — what is evicted, what breaks.
6. **`--cache-reuse` capability** — what can KV shifting actually salvage when a
   prefix is edited mid-stream.
7. **`--ctx-checkpoints` VRAM cost** at 128K–256K.
8. **Hybrid P-core/E-core affinity** on Windows for llama.cpp, and whether
   prefill and decode want different masks.
9. **What changed in llama.cpp since commit `60eeeb608`** that matters here.
10. **Any layer I have not listed.** This is the question that motivated the
    brief: I missed an entire axis once already.

---

## 7. Rules for your answer

1. **Cite sources.** Papers, model cards, llama.cpp PRs/issues/commits, or
   benchmark posts. Distinguish clearly between *measured on comparable
   hardware*, *vendor claim*, and *paper claim under a different serving stack*.
2. **Artifacts by exact identity.** HF repo + exact filename + **byte count** +
   revision SHA. Repos get republished in place — one of mine was replaced
   mid-session with identical filenames and different contents, and a tag-matched
   fetch (`repo:Q2_0`) silently pulled a *different file of identical byte
   count*.
3. **State the build requirement.** If something needs a commit newer than
   `60eeeb608`, an unmerged PR, or a fork, say so up front — that changes the
   cost from "one boot" to "a source build that unpins my baseline".
4. **Size every recommendation against the 13.6 % floor.** An effect I cannot
   distinguish from restart drift is not actionable.
5. **Answer the ⚠️ items.** A confirmed "yes, inert, here is why" is a useful
   result and closes a question permanently.
6. **VRAM arithmetic explicitly.** 12,282 MiB total, and I hold ~512 MiB in
   reserve because below that I have observed instability. Any proposal that
   adds VRAM must say how much and what it displaces.
7. **Do not recommend re-tuning** `-t`, `-tb`, `-b`, `-ub`, `--fit-target`, MTP
   draft depth, or the `--spec-draft-p-*` sub-knobs. Measured, settled, below the
   floor.
8. **Flag anything that would change output determinism.** I verify greedy
   byte-equality between configurations, and I need to keep that.

---

## 8. Output format I would like

For each item, a short block:

```
LAYER / KNOB
  what it does          — one or two sentences, mechanism not marketing
  evidence              — measured / vendor claim / paper, with the citation
  expected size here    — against a 12 GB Ada card, a 27B at 1–2 bits, single stream
  requirement           — stock build 10472 | newer commit | unmerged PR | fork | other engine
  artifact              — repo / file / bytes / revision, if any
  risk                  — what it breaks: determinism, residency, quality at depth
  verdict               — test now / test later / confirmed inert (and why)
```

Then a final table ranking everything by **expected gain in verified accepted
tasks per hour**, not by tok/s.

---

## 9. One more thing

If your answer contains a number that would change what I run, tell me **the
cheapest experiment that would falsify it on this machine** — one boot, one
corpus run, or one download. I would rather run a five-minute test than adopt a
recommendation.

---

## 10. The dispatched plan — final, 2026-08-20

Recorded verbatim so the reply can be scored against what was actually asked.
Report 17 had to reconstruct this by hand for the previous round.

**Workstreams, in dispatch order (priority is the order):**

1. **GBNF grammar** for a Python fenced code block — design it, its throughput
   cost, and its interaction with speculative decoding. Plus the **anti-loop
   comparison**: DRY vs `--repeat-penalty` vs `--top-n-sigma` vs Mirostat vs
   `--reasoning-budget`, against runaway reasoning specifically.
2. **All 11 speculative decoders.** Drafter placement on CPU
   (`--spec-draft-override-tensor`). Status of **DFlash 2 / llama.cpp PR #27342**.
   Search for any drafter GGUF matching Qwen3.8-27B.
3. **imatrix / calibration corpus** effect on code accuracy at 1–2 bit.
   `--override-kv` for GGUF metadata and MTP-head inspection. AWQ/GPTQ/QAT status
   in GGUF. `--samplers` ordering. Temperature 0.6 vs 1.0. `--backend-sampling`
   greedy equivalence.
4. **`--override-tensor` patterns** for attention-on-GPU / FFN-on-CPU on a
   nearly-resident model. **Control vectors / LoRA** to change model behaviour
   *without a restart*.
5. **Asymmetric KV quantization** (`-ctk` / `-ctv`). VRAM overhead of
   `--ctx-checkpoints`. `--cache-reuse` under a mid-stream prefix edit. RoPE/YaRN
   effect on long-context retrieval. `--context-shift` + `--keep` semantics.
6. **`--props`** for within-boot sampling A/B. **Model routing** via
   `--models-dir` (fast model + accurate model for the agent loop).
   `--slot-save-path`. Chat-template normalization. `--reasoning-preserve`.
   `--prefill-assistant`.
7. **Hybrid CPU affinity** (P/E cores; separate prefill vs decode masks).
   DDR5-7000 effect on the CPU-offload case. Reducing OS/driver run-to-run drift
   on Windows 11 / WDDM. Building for CUDA arch **89** vs the shipped fat binary.
8. **Confirm or refute the ⚠️ predicted-inert list** — XTC / `--ignore-eos`,
   `--split-mode tensor`, `--rpc`, CUDA vs Vulkan, `-np > 1`, DDR5 impact,
   server/network flag side effects. Plus llama.cpp changes since `60eeeb608`,
   `ik_llama.cpp`, ExLlamaV3 / vLLM / SGLang on 12 GB, `-fa auto` behaviour
   across artifacts, **and any layer still unsurveyed**.
9. **Coder-specialised open models 9B–35B** for 12 GB — repo, file, bytes, SHA,
   drafter compatibility. *(lowest priority)*
10. **Acceptance criteria enforced across (1)–(9):** resolvable URL per claim ·
    mechanism rather than an unsourced speedup multiplier · evidence tier stated
    (measured / vendor / paper) · exact artifact identity (repo, file, bytes,
    SHA) · determinism warning · build requirement (10472 / newer commit /
    unmerged PR / fork / other engine) · VRAM arithmetic from 12,282 MiB with a
    512 MiB reserve · cheapest falsification test (1 boot / 1 corpus / 1
    download) · effects below **13.6 %** are noise · no re-proposing settled
    knobs (`-t`, `-tb`, `-b`, `-ub`, `--fit-target`, MTP draft depth,
    `--spec-draft-p-*`).

**Layer coverage against report 16:** all sixteen layers are represented.
Ordering deliberately inverts the previous draft — the model search that had been
first is now last, because if workstream 1 succeeds it may remove the reason to
search for a different model at all.

**How to score the reply:** one row per workstream, answered / partially / not
answered, plus the criteria in (10) checked per claim. That is the table report
17 had to build by hand.
