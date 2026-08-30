# `unsloth/Qwen3.8-27B-GGUF` discussion #26 — "Single 16GB RTX 5060 Ti: 53 t/s with MTP and Quantized KV Cache"

**Captured 2026-08-24** from a page saved by the developer.
Archived verbatim: [`discussion-2026-08-24.html`](discussion-2026-08-24.html).
Opened 9 days before capture by `hfmiguel`; 24 👍; five participants.

**This is the closest external match this project has ever had.** Same card
(**RTX 5060 Ti 16 GB**), same model repository (`unsloth/Qwen3.8-27B-GGUF`), same
runtime (`llama-server`), same decoder family (`draft-mtp` on the baked-in head),
and — unlike the [Reddit thread](../reddit-5060ti-quant-thread/README.md)
captured the same day — **the participants name their artifact, their context
depth, their KV type and their reasoning effort.**

> Still not evidence. Nobody here alternates arms within a round, and this
> project has measured a **48.9 % spread across boots at 65,536** on
> byte-identical counters ([`CORRECTIONS` §23](../../reports/CORRECTIONS.md)).
> What follows is *the best-conditioned outside claim set we have*, not a result.

**Outcome of checking it against our tree: nothing needed changing.** Every lever
it names, we already set or already recorded. That is the finding, and §4 lists
each check rather than asserting it.

---

## 1. The one table worth the capture

`Bellatorius01`, RTX 5060 Ti 16 GB, `UD-IQ4_XS`, ctx 32,768, **`q8_0` K and V**,
`-fa on`, `parallel 1`, MTP head fully on GPU, `--spec-draft-n-max 3`,
llama.cpp CUDA 13.3 build **b10549**. Peak VRAM ~15.4 GB, checked for Windows
shared-memory spill and found none.

| context | no MTP | MTP n=3 | ratio |
|---:|---:|---:|---:|
| ~2,500 | 26.0 tok/s | **54.1** | **2.08×** |
| ~14,200 | 24.9 | **50.8** | **2.04×** |
| ~18,100 | 24.5 | **51.3** | **2.09×** |
| ~25,400 | 23.7 | **40.8** | **1.72×** |

**Why this one matters and the rest do not:** it is the only external report on
this card that measures **both arms** — and it is therefore the only one whose
number survives the objection that killed our own *"4× slower"* headline
([`CORRECTIONS` §28](../../reports/CORRECTIONS.md)), where two correctly measured
figures were made false by being put in one table.

**The shape agrees with ours and the magnitude does not.** MTP's benefit
**decays with depth** — 2.09× at 18K falling to 1.72× at 25K, a 17-point drop
across 7,000 tokens. This project's own rule is that
*a verdict at one depth does not transfer to another*, and here is an outside
curve saying the same thing about the exact decoder we serve. Our measurement of
`draft-mtp` on `UD-Q2_K_XL` was **+40 % on short turns**, against his +104 % to
+109 % in the same range — **different artifact, different KV, one unpaired
session on our side.** The gap is not explained and should not be explained away.

**We serve at 147,456.** His curve stops at 25,400 and is already falling. We
have **no MTP-vs-no-MTP figure at our depth on our artifact**, which is now the
most obviously missing measurement this project has — issue
[#44](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/44).

---

## 2. `medium` — the third and fourth independent confirmations

This project discovered on 2026-08-24, by reading the chat template, that **every
server it had ever launched ran at `reasoning_effort: xhigh`**, and made `medium`
the served default the same day ([report 35](../../reports/35-Q2KXL-MTP-AND-THE-EFFORT-NOBODY-SET.md)).

Two participants here reached it independently, and a third did in the Reddit
thread:

> `hfmiguel`: *"Without `"reasoning_effort": "medium"`, the model overthink too
> much ( **the default is xhigh** )"* — listed as one of three notes on his
> config, above the settings themselves.

> `Bellatorius01`: *"The model template supports xhigh / medium / low, with
> **xhigh as default**. Medium looks promising for coding quality…"*

> `DrKappa` (Reddit, same card): *"be careful it **defaults to xhigh**, setting
> to **medium** is mandatory for 16gb"*

**Three operators on this card, none citing each other, all naming the same
undeclared default.** That is the strongest outside support anything in
`docs/researchs/` carries.

### And a warning we should have had before we shipped it

`Bellatorius01`, on the cost of `medium`:

> *"it absolutely needs a bigger output budget — with a **3K-token ceiling** it
> could spend the whole allowance reasoning and never emit the final answer. At
> **6K** it completed and gave a stronger engineering analysis than
> thinking-off."*

**A generation cap can be consumed entirely by reasoning, producing a truncated
turn that looks like a model failure.** Our served cap is **8,192**, and across
33 production turns **0 hit it** (median generation 95 tokens) — so we are on the
right side of this and were before reading it. Recorded because the failure mode
is silent: a `medium` run under a 3K cap returns *something*, and nothing in the
result says the budget was the cause.

---

## 3. A wrong backend, a plausible number, and nothing saying so

`Hackin085`, same card, same artifact, same flags as `hfmiguel` — **16–22 tok/s
against 50–55.** He then found it:

> *"My llama.cpp install from Winget was using the **Vulkan backend instead of
> CUDA**. After switching to the official CUDA build, performance improved a lot.
> … So if anyone has the same performance problem, first check whether llama.cpp
> is actually using CUDA and not Vulkan."*

**This is our own 2026-08-24 finding, happening to somebody else.** We had every
binary compiled for `sm_89` on an `sm_120` card, silently JIT-ing Ada PTX, and
found it only by reading `CMakeCache.txt` — prefill **146,155 → 66,582 ms** once
fixed ([report 34](../../reports/34-BLACKWELL-BOUGHT-HEADROOM-NOT-SPEED.md)).
Same fault class, one layer up: **the wrong compute backend returns a believable
number and nothing in the log volunteers it.**

`scripts/worker-q2kxl-mtp.ps1` refuses to serve unless `cuobjdump --list-elf`
finds `sm_120` in `ggml-cuda.dll`. **That guard is now externally validated** —
it is exactly the check that would have saved `Hackin085` three days.

He also notes something our own harness should hear: *"CLI agents add a large
system/tool prompt, so even a fresh session can already start with ~20k
context."* Our production high-water was **75,841 of 147,456**, and the floor
under it is not the conversation.

---

## 4. What we checked against our own profile, and did not change

Every lever this discussion names, checked against
`qwen38-tuning/scripts/worker-q2kxl-mtp.ps1` on 2026-08-24. **No profile edit
resulted.**

| their setting | ours | verdict |
|---|---|---|
| `--parallel 1` — *"makes possible up the context to 92K"* | `-np 1` | **already set**, same flag, short form |
| `--cache-type-k q4_0 --cache-type-v q4_0` | `-ctk q4_0 -ctv q4_0` | **identical** — and symmetric `q4_0` is one of the pairs our `FA_ALL_QUANTS=OFF` build can actually express ([§29](../../reports/CORRECTIONS.md)) |
| `--flash-attn on` | `-fa on` | identical |
| `--spec-draft-n-max 3` | default 3 | identical — and we measured 7 at **−56 % on MTP**, so 3 is held on our own evidence |
| `--gpu-layers-draft all` | not set | **already satisfied.** `probe-q2kxl-mtp-147456.log`: `n_layer_all = 65`, `offloaded 66/66 layers to GPU`, `blk.64.nextn.*` tensors created. `-ngl auto` reaches the head |
| `temp 1 / top_p 0.95 / top_k 20 / min_p 0.0 / presence 0.0` | `/props` on the live server: **1.0 / 0.95 / 20 / 0.05 / 0.0** | **already recorded** — [`vendor-quantization-tables.md` §2](../vendor-quantization-tables.md) had the vendor thinking preset and the `min_p` **0.05 vs 0.0** delta written down before this capture |
| `--reasoning-effort medium` | `--reasoning-effort medium` | identical, since 2026-08-24 |

**`Bellatorius01` independently confirms the metadata we read today** —
`qwen35.nextn_predict_layers = 1` and `blk.64.nextn.*`, *"so no separate draft
model is needed."* We reached that by loading `UD-Q2_K_XL` with **no `-md`** and
watching 743 MiB come back.

---

## 5. The sampling claim, and why it does not transfer to us

`Bellatorius01`'s last update is the largest single effect anyone in either
document reports:

> On one deterministic coding canary, moving from `temp 0.2 / top_p 0.95` with
> **thinking disabled** to `temp 0.7 / top_p 0.8 / top_k 20 / presence_penalty
> 1.5`: *"the same correct 8/8 implementation, but **tool-call repetition
> dropped from 30 to 0, requests from 53 to 4, and wall time from 374.9 s to
> 42.2 s**."*

**8.9× wall clock with final correctness unchanged.**

**It does not apply to our configuration.** He was in **non-thinking** mode and
moved to Qwen's **non-thinking** preset. We run **thinking at `medium`**, and
`/props` shows the live server already on the **thinking** preset. His result is
evidence that mis-matching the preset to the mode is expensive — not that our
preset is wrong.

**What it does say, and this project has no data on it:** the axis he measured is
**agentic convergence** — tool-call repetition, request count, wall time to a
correct answer. This project measures **tok/s** and **task success**, and those
two would both have looked *identical* across his two runs. A configuration can
be 8.9× worse to work with while scoring the same on everything we record.

---

## 6. Leads, none measured

- **`--load-mode none`** — set here by `hfmiguel`, and twice in the Reddit thread
  with the reason *"mmap slows down PP"*. **Two independent sources, zero
  measurements here.** We do not set it.
- **Build `b10549`** against our **10499 / `1deefcca3`**. Ours is pinned to the
  DFlash2 PR; moving is not free and the delta is unread.
- **`--device CUDA0 --spec-draft-device CUDA0`** — explicit device pinning,
  which `Hackin085` added after his Vulkan incident. One GPU here, so it is
  belt-and-braces rather than a fix.
- **`--no-warmup`, `--no-mmap`** (`hfmiguel`) — unset in every profile of ours.
- **`--load-mode dio` + `--no-mmproj-offload`** (`Stranikviv`) for vision, with
  *"mmproj load into normal memory"*. We run `--no-mmproj-auto`.

---

## Provenance

Saved page: [`discussion-2026-08-24.html`](discussion-2026-08-24.html) —
`https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/26`.
Live-server values in §4 are from `GET /props` on `127.0.0.1:8080` while
`worker-q2kxl-mtp.ps1` was serving, 2026-08-24.
Layer counts are from `qwen38-tuning/logs/probe-q2kxl-mtp-147456.log`.
Companion capture from the same day:
[`../reddit-5060ti-quant-thread/`](../reddit-5060ti-quant-thread/README.md).
