# How to make this thing faster — the guide, 2026-08-29

Everything below is either **MEASURED HERE** (with the file its number came
from), **VENDOR** (someone else's documented claim, useful and unverified), or
**UNMEASURED** (a lever nobody has pulled). Nothing is a recommendation from
memory, and the tags are not decoration — this project has published forty-three
claims it later contradicted with its own data
([`CORRECTIONS.md`](CORRECTIONS.md)).

Read [`38-NVFP4-PROFILE-REFERENCE.md`](38-NVFP4-PROFILE-REFERENCE.md) first if
you do not know what is being served.

---

## 0. The four rules that decide whether an answer is real

These cost nothing and they are the reason most of the numbers here survived.

1. **Pair within a boot, rotate the order, three rounds.** Never compare a rate
   from one boot against a rate from another. The measured drift is **13.6 %** at
   ctx 16,384 and up to **48.9 %** at 65,536 for *the same arm with byte-identical
   counters* ([`CORRECTIONS.md`](CORRECTIONS.md) §23). A well-run pair here
   spreads 1.6–3.3 %.
2. **Residency before arithmetic.** A delta between an arm that spilled and one
   that did not is a measure of the spill. `harness.residency_note` refuses it;
   look for `66+0` on both sides.
3. **A verdict does not transfer.** Not across **depth** (`draft-mtp` is +81 % at
   16K and −71 % at 131,072), not across **artifact** (`n-match 24` lost on
   `UD-Q4_K_XL` and wins by +27.1 % on NVFP4), and not across **workload**
   (`ngram-mod` is worth +27.1 % on a vendor-source corpus and fires **5 times in
   4,653 calls** on real agent traffic).
4. **Loading is not surviving, and the probe size is part of the claim.** Push a
   request **half the window** through every rung. 229,376 was published as the
   ceiling on the strength of a 65,643-token request — a *quarter* of its window —
   and dies on a half-window one ([`CORRECTIONS.md`](CORRECTIONS.md) §35).

---

## 1. Settled. Do not re-test these

| lever | verdict | evidence |
|---|---|---|
| `-sm tensor` vs `-sm layer` | **tensor, −31.0 % [−32.9, −29.6] for layer** on NVFP4 with the served decoder, both `66+0` | MEASURED, `dflash2-arena.jsonl` |
| `-ts` computed vs unset | **compute it.** Unset splits *evenly* (`llama-model.cpp:707`) and produced **0.38 tok/s** — an 85× silent spill | MEASURED, [`CORRECTIONS.md`](CORRECTIONS.md) §33 |
| `-sm row` | **cannot load.** `device CUDA0 does not support split buffers`; `ggml-cuda.cu` does not export `ggml_backend_split_buffer_type` at this commit | MEASURED |
| `--fit` under `-sm tensor` | **inert.** `llama_params_fit is not implemented for SPLIT_MODE_TENSOR`; its `abort` is the *fitting step* giving up, not the load | MEASURED |
| KV type | **`q4_0`.** 18.00 KiB/token; no other type in this build has a fast kernel | MEASURED, [results 03](../results/03-memory-and-kv.md) |
| `-ub` | **1024.** Decode flat, **prefill +10.1 %** | MEASURED, `dual-ubatch-16384.jsonl` |
| the artifact | **NVFP4 VERY-LOW + baked-in MTP, +63.1 %** over `UD-Q4_K_XL` + `ngram-mod` | MEASURED, `nvfp4-final-147456.jsonl` |
| the artifact **alone** | **−22.4 %.** NVFP4 without MTP is a *loss*; n-gram acceptance falls 55.4 → 22.1 | MEASURED, `nvfp4-vs-q4-147456.jsonl` |
| ~~DFlash2 on NVFP4~~ **RETRACTED 2026-08-30 — not settled** | The `+0.2 % and the sign flips` behind this row was a **handicapped arm**: ctx 147,456, `--spec-draft-n-max 3`, and `n-match 12` — the window the row two above records collapsing on this artifact (55.4 → 22.1) while 24 wins. Re-measured with 65,536 / `n_max` 4 / `n-match` 24 it is **+67.9 % [+65.8, +71.5] RESOLVED**. At the served 147,456: 44.48 / 44.56 / 44.23 against MTP's pooled 42.77 — **+4.0 %, under the floor and across boots, so NOT resolved**, though the ranges are disjoint. **What it buys is consistency — 0.7 % spread against 9.3 % — for ~950 MiB** | MEASURED, `nvfp4-dflash-65536.jsonl`, `nvfp4-dflash-147456-n4.jsonl`, [`CORRECTIONS.md`](CORRECTIONS.md) §42 |
| depth ceiling | **200,704** with a half-window request; 229,376 loads and dies | MEASURED, [`CORRECTIONS.md`](CORRECTIONS.md) §35 |
| vision | **works at every depth we serve**, 888 MiB, on the unpatched binary | MEASURED, [results 02](../results/02-decoders.md) |

**Why tensor wins, and where it stops winning:** *"improves tokens/sec for
**dense** models. MoE models don't benefit."* (VENDOR). Qwen3.8-27B is dense. Do
not carry the +31 % to an MoE.

---

## 2. The levers, in the order worth pulling

### Tier 0 — what the client sends, which no flag on this page can touch

**MEASURED 2026-08-30, one boot, minutes apart, nothing changed but a toggle in
the chat UI** (`qwen38-tuning/logs/serve-20260830-010653.log`, tasks 2931 and
2994):

| | tools ON | tools OFF |
|---|---|---|
| prompt | **17,843 tokens** | **334** |
| prefill | 18,618 ms | **554 ms** |
| decode | 35.20 tok/s | **45.64** |
| the whole answer | **21.5 s** | **1.5 s** |

The message both times was `สวัสดี`. **17,509 of those 17,843 tokens were tool
schemas**, sent on every request by the client, and reading them is what the
18 seconds were. Decode rose too, because decoding at depth 334 is cheaper than
at 17,843 — the prompt size moves both halves.

**14x on the wall clock, from a checkbox.** Every lever below this line is worth
single-digit or low-double-digit percentages. Before touching any of them, ask
what the client is putting in front of the user's actual words:

- **tool schemas.** Claude Code sends **17,881** tokens of preamble before the
  first character of a greeting — the same shape, a different client. A tool
  the model will not call this turn still costs its full definition every turn.
- **whole files pasted into context.** A 117 KB plan is ~29,000 tokens. Unsloth
  Studio answers questions about that same file in under 4 seconds because it
  indexes it and retrieves 5 chunks — **1,942 new tokens against our 46,998**
  (`docs/researchs/unsloth-studio-config-2026-08-29.md`). That is a client
  feature, not a server setting, and it is most of the difference in feel.
- **conversation history**, including preserved thinking. See
  `--reasoning-preserve` below.

**This is the one place a factor of ten is available on this machine.** It is
recorded in an optimisation guide because a reader who came here for flags
should meet it first.

### Tier 1 — one flag, one paired sweep, no downside if it loses

| # | lever | now | try | why |
|---|---|---|---|---|
| 1 | `--spec-type` **order** | `draft-mtp,ngram-mod` | `ngram-mod,draft-mtp` | if order decides which is asked first, it may explain 5 drafts in 4,653 calls |
| 2 | `--spec-ngram-mod-n-max` | 32 | 64 | **never swept here at all** |
| 3 | `--spec-ngram-mod-n-min` | 16 | 48 | recorded as *measured, no effect* — worth one confirmation at the new n-max |
| 4 | `--spec-draft-n-max` | 3 | 2 | 2 is the **documented default for MTP on GPU** (VENDOR); our 3 is the deviation, and acceptance per position `(0.690, 0.448, 0.284)` says it earns its place. A test we expect to win |
| 5 | `--threads` | 18 | 2 | everything is GPU-resident; 18 may be contention |
| 6 | `--kv-unified` | unset | set | may be inert at `-np 1` |

### Tier 2 — a real trade, needs a decision as well as a number

| lever | the trade |
|---|---|
| ~~`--ctx-checkpoints 0`~~ **SETTLED — do not turn them off** | **MEASURED HERE 2026-08-29.** This model is hybrid, and the recurrent half cannot rewind to a shared prefix, so with no checkpoint llama.cpp abandons the prompt: `forcing full prompt re-processing due to lack of cache data`. `serve-20260829-125227.log` printed it on **all three** requests it served — 17,881, then 46,998, then 46,997 tokens, the last two the same conversation, **51.6 s of prefill each**. The same binary with the default printed it once in a whole session (`serve-20260829-073741.log`) and prefilled 13–1,358 tokens per turn. Cost of keeping them: **150.89 MiB each, at most 32, ≥8,192 tokens apart** — about six at our depth, in host RAM |
| `--cache-ram 0` | same family, same trade, smaller. Default is 8,192 MiB of host prompt cache |
| **drop `ngram-mod`** | it fires 5 times in 4,653 calls on agent traffic and Studio's single runs put **MTP alone ahead of MTP+ngram** (54.95 vs 52.28). **Our corpus cannot answer this** — `real-code-vendor` is exactly the text an n-gram is good at. Needs an agent-like regime first |
| the sampler | we set **none**, and the served value is **not** the flag default. `GET /props` reads `temp 1.0 · top_k 20 · top_p 0.95` — llama.cpp applies `general.sampling.*` from the GGUF, and Studio sends the same three off the same file (MEASURED HERE, `/props` on port 8080). **The real gaps are `min_p` 0.05 vs their 0.0, `presence_penalty` 0.0 vs their 1.5, and `n_predict` -1 vs their 36,453.** A quality lever, and quality is unmeasured on every artifact here |

### Tier 1b — the one that needed the developer to catch a bad argument

**`--spec-draft-backend-sampling`.** Every tensor-split boot prints
`set_sampler: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU`,
one line after `draft-mtp` announces `backend_sampling=1`. This guide previously
implied the CPU fallback was harmless because *"layer has backend sampling and
is still 31 % slower"*. **That argument is invalid** — the layer/tensor pair
changed the split *and* the offload together, so −31 % bounds the offload's
benefit only from above. It could be worth 20 % while the split costs 51 %.

Two flags, from the binary's own help:

```
-bs, --backend-sampling         enable backend sampling (experimental)   default DISABLED
--spec-draft-backend-sampling   offload DRAFT sampling to the backend    default ENABLED
```

So the **main** sampler is on the CPU under *both* splits — nothing here passes
`-bs`. What the tensor split loses is only the **draft** offload.

`-sm layer` is the only split where it works, so it is the only place it can be
varied alone: `--arms draft-sampling-cost`, one flag, everything else held. The
delta is the offload's worth **X**, and **X is a tax this configuration pays and
cannot avoid** — the offload is refused under the split that wins by 31 %.
Tensor's true advantage is about **31 % + X**.

### Tier 3 — recorded, not proposed

`GGML_CUDA_ALLREDUCE` A/B · the display card moving to the UHD 770 ·
`-b`/`-ub` both at 2048 · a newer llama.cpp (Studio runs **b10672** from the
`unslothai` fork against our **10499**; swapping binaries voids every rate this
project holds).

---

## 3. What is already at its documented default, so leave it

VENDOR, from Studio's own field descriptions:

- `--batch-size` **2048** — *"rarely needs changing; the micro-batch is what
  usually matters"*. We set 2048.
- `--ubatch-size` default **512** — we deviate to 1024 **and measured why**.
- KV dtype default **f16**, *"8-bit is the safest reduction"* — we run `q4_0`
  and measured that too.
- `--load-mode` auto prefers `none` *"since a mapped read is slower"*. We do not
  set it. **UNMEASURED here.**

---

## 4. The things that will waste your day

- **Comparing raw decode across boots.** See rule 1.
- **Re-running `-sm row`.** It cannot load and the cause is the build, not the
  cards.
- **Trusting a rate from an ad-hoc script.** The harness already has
  `generation_is_measurable`, `copied_window_fraction`, `generation_is_original`
  and a frozen corpus. Four times in three days a probe written beside them was
  the thing that was wrong — including one that read a profile failure off a
  prompt with no instruction in it.
- **Launching without `QWEN38_LLAMA_EXE` pointed at a Blackwell build** — now
  the arena default, after the Ada-only binary produced fifteen published rows on
  the wrong machine.
- **Asserting on source text in a test.** Five assertions this week called a
  refactor a regression, and one was green throughout the fault it was written to
  catch. Assert on a resolved value.
- **Believing a launcher's prose.** Seven times a `.bat` has described a run it
  did not cause. `test_the_hub_launcher.py` now pins the menu against the file it
  calls.

---

## 5. How to actually run one

```powershell
cd C:\AI\qwen38-tuning\bench
python -m pytest tests\ -q                     # the gate. 1004 tests.
$env:QWEN38_LLAMA_EXE = "C:\AI\llama.cpp-blackwell\llama-server.exe"
python dflash2_arena.py --arms <set> --ctx 147456 --rounds 3 --regime real-code-vendor
```

Gate on **pytest's own exit code**, not a pipeline's — `pytest | tail && run`
returns `tail`'s status, and three red tests once launched a sweep anyway.

Read three things from the result, never one: the **paired delta**, the **spread
of each arm**, and the **split each arm ran at**. Then read the boot log for
whatever the arm was supposed to change — a flag that does nothing usually says
so there.

---

## 6. The one that is not a speed lever, and gates everything

**Quality has never been measured on any artifact this project serves.** The
current proposal changes the **model file**, and `VERY-LOW` is the cheapest tier
of nine — `Q3_K` LM head, `Q2_K` embeddings. `COMPACT-LOW` buys `Q4_K`/`Q3_K`
for **300 MB** and has never been downloaded.

The one outside statement that helps: *"MTP and ngram do not change output"*
while DFlash and DSpark *"on quantized targets … can differ from a non
speculative run"* (VENDOR). If that holds, the decoder half of this profile is
quality-neutral and only the **artifact swap** needs answering.

**Nothing on this page has changed a default.**
