# r/LocalLLM — "Qwen 3.8 27B on a 16GB 5060 Ti and 64gb DDR4"

**Captured 2026-08-24** from a page saved by the developer.
Archived verbatim: [`thread-2026-08-24.html`](thread-2026-08-24.html) (1,393,721 B).
Thread posted 2 days before capture; 74 upvotes, ~40 top-level comments.

> **Nothing on this page is evidence.** It is forum text, most of it without a
> named binary, a context depth, a KV type or a reasoning effort. What makes it
> worth a folder is the **hardware**: at least four commenters are on a
> **5060 Ti 16 GB**, and this project has never had an outside number on its own
> card. Read every figure below as *a claim someone made*, not a result.

**What it actually bought us is not a number.** It is a **compile flag our build
does not have**, which we found by checking our own `CMakeCache.txt` after a
commenter told the OP to set it. That finding is verified here, in this tree,
and it is written up in [§1](#1-the-flag-we-decided-against-and-the-question-that-decision-answered).

---

## 1. The flag we decided against, and the question that decision answered

A commenter (`tsangberg`, same card) opens with:

> **IMPORTANT: Compile llama.cpp with `GGML_CUDA_FA_ALL_QUANTS=ON`**

and then posts a config whose KV line is `cache-type-k = q5_0`,
`cache-type-v = q4_1`.

**Verified in this tree, 2026-08-24:**

| what | where | value |
|---|---|---|
| our Blackwell build | `llama.cpp/build-blackwell/CMakeCache.txt` | `GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF` |
| our Ada build | `llama.cpp/build-dflash2/CMakeCache.txt` | `GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF` |
| the default | `ggml/CMakeLists.txt:208` | `option(... OFF)` |

**What OFF actually removes** — `ggml/src/ggml-cuda/fattn.cu:340-352`:

```c
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            return true;
```

and `fattn.cu:442-446`:

```c
#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif
```

So on both of our binaries: **`q4_1`, `q5_0` and `q5_1` are not KV types at
all**, and **every asymmetric K≠V pair is refused** before the type check is
even reached. `tsangberg`'s `q5_0`/`q4_1` line is not slow on our build — it is
**not expressible** on it.

### Why this was not caught: the recorded answer is to a narrower question

Three pages carry the decision, and all three give the same reason:

| page | what it says |
|---|---|
| `reports/05-OPERATING-GUIDE.md:153` | ``FA_ALL_QUANTS`` rebuild for Q8 KV? **not needed** — Q8 is faster on the stock binary |
| `reports/06-OPEN-QUESTIONS.md:211` | Is `FA_ALL_QUANTS` needed for Q8 KV? **No** |
| `reports/16-OPTIMIZATION-SURFACE.md:228` | build: `FA_ALL_QUANTS` · off · **decided** · Q8 KV is faster on the stock binary, so it was not needed |

**That answer is correct and it cannot bear the weight the row puts on it.**
`GGML_TYPE_Q8_0` is in the *always-compiled* list above — it falls through to
`return true` whether the flag is on or off. So a Q8 measurement is
**structurally incapable** of saying anything about this flag. It was never a
test of `FA_ALL_QUANTS`; it was a test of a KV type the flag does not gate.

The caveat existed at the time and the decision row dropped it —
`Deep Research/deep-research-optimization2.md:138` says the flag is
*"Only for asymmetric/non-stock KV experiments"*, which is exactly the set that
was then never run.

Retracted as [`CORRECTIONS.md` §29](../../reports/CORRECTIONS.md).

### Half of the failure is loud and half is silent

Traced through `src/llama-context.cpp`:

- **`-fa auto` is the default** (`llama-context.cpp:3534`). It probes, and on
  failure emits `LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n")` and
  **continues** (`llama-context.cpp:547`).
- **If V is quantized** and FA ends up disabled, load **hard-fails**:
  `"quantized V cache requires flash_attn to be enabled"`, `return nullptr`
  (`llama-context.cpp:3607-3610`). `-ctv q4_1` on our build does not boot.
- **If only K is quantized** — `-ctk q5_1 -ctv f16` — nothing fails. Flash
  attention is silently switched off, the server boots, and the run reports a
  number.

**That last row is this project's north-star fault**: an instrument that returns
a believable figure instead of a failure, behind a `WARN` in a log nobody greps.

**Not measured.** Whether `FA_ALL_QUANTS=ON` is worth a rebuild is open — the
flag costs compile time and this project has never run a KV type it unlocks.
What is settled is that **the register said "decided" about a question it never
asked.**

---

## 2. The four same-card claims

None of these name a build, and only one names a reasoning effort. Recorded
because they are the only outside numbers this project has on an RTX 5060 Ti
16 GB.

| commenter | artifact | ctx | KV | decode | prefill | notes |
|---|---|---:|---|---:|---:|---|
| `tsangberg` | NVFP4-MTP-COMPACT-LOW | 62,200 | q5_0/q4_1 | 25 tok/s | 1,100–1,500 tok/s | fully resident, **MTP off** |
| `tsangberg` | same, ffn layers → CPU | 140,000 | q5_0/q4_1 | 10 tok/s | ~700–800 tok/s | with MTP `n_max` 2 |
| `tsangberg` | `UD-Q4_K_M`, ffn → CPU | 106,000 | q5_0/q4_1 | *"very similar"* | — | his point: the NVFP4 edge disappears once you offload |
| `DrKappa` | Q2 (tier unnamed) | ~90,000 | *"q4_0-ish"* | 50–60 tok/s | — | also Q3 40–50 @ 60K, Q4 30 @ 40–50K |
| `Proper-Tower2016` | `UD-IQ3_XXS` turbo | 180,000 | — | 33 tok/s | 300–800 tok/s | MTP 1 |
| `Embarrassed-Boot5193` | `IQ3_XSS` | 100,000 | — | 25–32 tok/s | — | llama.cpp, MTP off, one GPU of two |

**For scale, ours:** `UD-Q2_K_XL` + `draft-mtp,ngram-mod` at ctx **147,456**,
66/66 resident, **decode median 37.36 tok/s over 33 production turns**
(`reports/35`). That sits above `Proper-Tower2016` at a comparable depth and
below `DrKappa`'s Q2 claim at 90K — but **no comparison here is valid**: not one
of these rows names its binary, and this project has measured a **48.9 % spread
across boots at 65,536 on byte-identical counters** (`CORRECTIONS.md` §23).

### `tsangberg`'s conclusion is the one worth keeping

> *"as soon as we try to get context up to usable levels with the best dense
> model offloading trick (ffn layers in RAM) there's no longer any performance
> to be had from using NVFP4 on Blackwell"*

If that holds, **NVFP4's advantage is confined to the fully-resident regime**,
and in that regime he measures **25 tok/s at 62,200** — below our 37.36 at
147,456. **This strengthens the 2026-08-24 decision not to pursue NVFP4**
(`results/09-hardware.md`) rather than reopening it: the trade was called on
context budget, and the first outside decode number on our card says the
throughput side does not compensate.

**Where NVFP4 still wins is prefill** — 1,100–1,500 tok/s against our binary's
untested figure at the same depth. Prefill is the axis `reports/27` says cannot
be tuned. **Unmeasured on both sides; not a verdict.**

---

## 3. Sizes: our table was right, the thread's is not

The thread's top comment gives `NVFP4 COMPACT-LOW` as **14.12 GB**. It is not.
Fetched from the Hub API, 2026-08-24, exact bytes:

| file | bytes | GB | GiB |
|---|---:|---:|---:|
| `NVFP4-STARVED` | 14,593,700,288 | 14.59 | **13.59** |
| `NVFP4-BUDGET` | 14,722,826,656 | 14.72 | **13.71** |
| `NVFP4-MTP-VERY-LOW` | 14,862,277,984 | 14.86 | **13.84** |
| `NVFP4-MTP-COMPACT-LOW` | 15,160,261,920 | 15.16 | **14.12** |

`results/09-hardware.md`'s table matches every row **to the digit**. The thread
quotes COMPACT-LOW's **GiB** figure with a **GB** label, which makes it look
0.4 GB smaller than `BUDGET` when it is in fact **0.41 GiB larger** — and the
comment then argues from that inversion. Against our 14.82 GiB budget,
COMPACT-LOW leaves **~0.70 GiB**, not the comfortable fit the comment implies.

**We checked our own table because we suspected it, and it held.** Recorded so
the next reader does not re-derive it.

---

## 4. Independent confirmation of the `xhigh` finding

`DrKappa`, unprompted and on the same card:

> *"be careful it defaults to **xhigh**, setting to **medium** is mandatory for
> 16gb and even in this case it can burn 10k tokens just reasoning"*

This project found the same thing on 2026-08-24 by reading the template — that
**every server it had ever launched ran at `xhigh`** — and made `medium` the
served default the same day (`reports/35`). An outside operator on the same
hardware reaching the same setting independently is the strongest external
support anything in this folder has.

`DrKappa` adds a claim we have **not** tested and should: that quantizing the
**draft** cache *"makes the model prone to looping even if you set repetition
penalty."* Our served profile quantizes KV but the MTP head's draft cache
handling is unaudited.

---

## 5. Things named here that this project has never tried

Recorded as leads, not recommendations. None is measured.

- **`GGML_CUDA_DISABLE_GRAPHS=1`** as a runtime env var. We measured
  `GGML_CUDA_GRAPH_OPT` (results/03) but never *disabling* graphs, which is a
  different switch.
- **Selective `override-tensor` ffn offload on a stride** —
  `blk\.(0|1|2|...|52)\.ffn_.*=CPU`, with the claim that these layers are
  **165.4 MiB of VRAM each** and that *"every extra layer offloaded causes tps
  to go down linearly."* A linear cost is testable and would refute or confirm
  the residency-cliff framing in `reports/19`.
- **`--load-mode none`**, twice, with the reason *"mmap slows down PP"*.
- **`Qwen3.8-27B-Ridge-3.7bpw`** and **`jrell` IQ4_XS-smaller** — two artifacts
  absent from `reports/15`'s inventory.
- **`-t 2 -tb 2`** (`Additional-Ordinary2`) against `-t 16 -tb 16`
  (`tsangberg`) — the thread contradicts itself on thread count and neither
  side shows a measurement.

---

## 6. What the thread gets wrong, so it is not quoted later

- **The top-voted comment is machine-written and self-contradicting.** It
  addresses a **5070 Ti** throughout while the OP has a **5060 Ti**, presents
  the OP's own two numbers back to them as *"your measurements"*, and mixes them
  into one table with a third party's benchmark taken at a different prompt
  length, context and KV type. A reply reads simply *"Thanks random AI"*. Its
  size column is the unit error in [§3](#3-sizes-our-table-was-right-the-threads-is-not).
- **"NVFP4 is even better than q8"** — asserted, unsourced.
- **Nobody in the thread measured quality.** `DystopianRealist` says so
  outright: *"They're just looking at token generation number speed, while
  quantizing their cache into something that will add errors."* That is this
  project's founding thesis, arrived at independently in a comment section.

---

## Provenance

Saved page: [`thread-2026-08-24.html`](thread-2026-08-24.html).
Hub sizes: `https://huggingface.co/api/models/esatapedico/Qwen3.8-27B-NVFP4-{BUDGET,MTP}-GGUF?blobs=true`,
fetched 2026-08-24.
Source citations are against llama.cpp **build 10499, commit `1deefcca3`**, the
tree at `C:\AI\llama.cpp`.
