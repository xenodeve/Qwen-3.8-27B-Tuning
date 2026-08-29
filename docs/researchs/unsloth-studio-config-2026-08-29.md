# What Unsloth Studio runs, on this machine, on our artifact

**2026-08-29.** Unsloth Studio (Desktop) is installed here and was launched
against **the same model file we serve** — `esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF`,
variant `Qwen3.8-27B-NVFP4-MTP-VERY-LOW`, snapshot `bcd7a7d3…`. Its own logs hold
the `llama-server` command line it built, so this is **not** a recommendation
read off a web page: it is another team's configuration for our exact artifact on
our exact hardware.

**Read from** `~/.unsloth/studio/logs/backend-*.log`,
`~/.unsloth/studio/studio.db` (`app_settings`, `chat_settings`), and
`~/.unsloth/llama.cpp/UNSLOTH_PREBUILT_INFO.json`. **No credential was copied
here.** `run/desktop_backend.json` holds a session token; it is deliberately not
reproduced, and `credential_secrets` was not read.

> **Nothing in this file is a measurement.** Everything below is *what another
> program does*, which is a hypothesis about what is good, not evidence. This
> project's own numbers are in `docs/results/`.

---

## The first thing worth saying: they agree with us where it counts

`app_settings.openai_api_auto_switch_overrides`, for our artifact:

```json
{"kv_cache_dtype": "q4_0", "speculative_type": "mtp+ngram", "tensor_parallel": true}
```

**Three independent agreements** with what this project measured its way to:
`q4_0` KV, MTP **beside** an n-gram rather than either alone, and the tensor
split. And in the command line, **`--spec-ngram-mod-n-match 24`** — the exact
value this project measured as +27.1 % over the 12 every other profile serves,
and which *lost* on the other artifact. Two parties arriving at 24 separately is
the strongest outside support any decoder verdict here has.

## The command line, theirs against ours

Theirs, reformatted; ours is `worker-q4-dual.ps1 -Nvfp4 -Vision`.

| | Unsloth Studio | this project | notes |
|---|---|---|---|
| model | same file, same snapshot | same | — |
| `--mmproj` | set | set (`-Vision`) | both load the vision tower |
| `--cache-type-k/v` | `q4_0` | `q4_0` | **agree** |
| `--split-mode` | `tensor` | `tensor` | **agree** |
| `--tensor-split` | `7177,12425` | computed, e.g. `7505,15288` | both compute it |
| `--flash-attn` | `on` | `on` | **agree** |
| `--spec-ngram-mod-n-match` | **24** | **24** | **agree, and independently** |
| `--spec-type` | `ngram-mod,draft-mtp` | `draft-mtp,ngram-mod` | **order differs** |
| `--spec-draft-n-max` | **2** | **3** | differs |
| `--spec-ngram-mod-n-min` | **48** | **16** | differs |
| `--spec-ngram-mod-n-max` | **64** | **32** | differs |
| `-c` | **41,984** | 147,456 / 200,704 | they serve a short window |
| `--parallel` | **4** | `-np 1` | they share the window four ways |
| `--threads` | **2** | **18** | differs sharply |
| `-ngl` | `-1` | `auto` | — |
| `--fit` | **off** | `on --fit-target 768` | — |
| `--kv-unified` | **set** | unset (log reads `kv_unified = false`) | — |
| `--no-context-shift` | **set** | unset | — |
| `--cache-ram` | **0 — prompt cache OFF** | unset → 8,192 MiB | see below |
| `--ctx-checkpoints` | **0 — OFF** | unset → 32 | see below |
| `--load-mode` | `none` | unset | — |
| `--metrics` | set | unset | free Prometheus endpoint |
| `--slot-save-path` | set | unset | — |
| thinking | `--chat-template-kwargs {"enable_thinking": true, "preserve_thinking": true}` | `--chat-template-file …` + `--reasoning-effort medium` | different mechanism |

## The two that explain something we already noticed

**`--cache-ram 0` and `--ctx-checkpoints 0`.** They turn both off. This project
leaves both at their defaults, and a real session was measured holding **20.4 GB
working set / 34.4 GB private**, with the log showing `context checkpoints
enabled, max = 32, min spacing = 8192` and individual checkpoints reaching
**350 MiB**. That is where the host RAM goes.

**It is not free to copy.** The same session showed checkpoints being *restored*
at positions 47,940–50,091 — they were doing work, and turning them off trades
RAM for re-prefill. At ~825 tok/s a 50,000-token re-prefill is about a minute.
Studio can afford it at `-c 41,984`; at 200,704 the trade is different.

## The sampler, and an ambiguity in it

`chat_settings.inferenceParams` (their global default):

```json
{"temperature": 0.7, "topP": 0.8, "minP": 0.0, "presencePenalty": 1.5,
 "maxTokens": 41984, "topK": 20}
```

and per-model, for our artifact: `{"temperature": 0.7, "topP": 0.8}`.

**Our server sets none of these**, so llama.cpp's own defaults apply —
`--temp 0.80`, `--top-k 40`, `--top-p 0.95`, `--min-p 0.05`,
`--presence-penalty 0.00`, read from `--help` on the served binary. **Every one
differs.**

Two cautions before copying them:

- **`presencePenalty 1.5` is high, and this artifact's publisher warned about
  exactly this class of setting.** Their README dropped DRY sampling because it
  "interferes with verbatim reproduction of long strings (paths, identifiers,
  tool arguments), which matters for coding and tool use". A presence penalty
  pushes against repeating tokens already emitted; a coding agent repeats
  identifiers constantly. **Untested here, and the risk is quality, which is the
  one thing this project has never measured.**
- ~~The thinking state is ambiguous.~~ **Resolved by reading every launch:
  thinking is ALWAYS ON at the server.** All eighteen Qwen launches carry
  `--chat-template-kwargs {"enable_thinking": true, "preserve_thinking": true}`
  — including the ones where `chat_settings.reasoningEnabled` is `false`, so
  that switch governs the UI, not the model. **Their 0.7 / 0.8 / 20 was
  therefore used WITH thinking on**, which makes it directly comparable to our
  `--reasoning-effort medium`. It remains a *different* preset from the
  `0.6 / 0.95 / 20` the NVFP4 repo quotes for the same mode, and which of the
  two is right here is still unmeasured — but it is now a straight choice
  between two presets, not a confusion about which mode they belong to.

## Their build, which is not a setting but is worth knowing

```
tag            b10672   (release b10672-mix-67dfc8b)
source repo    unslothai/llama.cpp   (a FORK, not upstream)
source commit  760fb1c764ee21db541777bffefb332e4e8628f9
bundle         windows-x64-cuda13-newer
supported_sms  86, 89, 90, 100, 103, 120
installed      2026-08-29T01:03:18Z
```

Ours is **build 10499, commit `1deefcca3`**, `CMAKE_CUDA_ARCHITECTURES=89;120`.
Theirs is newer and from a fork. **Whether the fork carries anything that
matters to us is unknown**, and swapping binaries is not a settings question —
it would void every rate this project has, which is why it is not proposed here.

## What is worth testing, ranked by cost

Cheap, one paired sweep each, no downside if they lose:

1. **`--spec-type` order.** `ngram-mod,draft-mtp` against our
   `draft-mtp,ngram-mod`. One flag, and the real-use log showed `ngram-mod`
   firing **5 times in 4,653 calls** — if order decides which is asked first,
   this is the cheapest thing on the list.
2. **`n-min 48 / n-max 64`** against our `16 / 32`. `n-min` is recorded here as
   *measured, no effect*; `n-max` has **never been swept at all**.
3. **`--spec-draft-n-max 2`** against our 3. Our own counters argue for 3 —
   acceptance per position `(0.690, 0.448, 0.284)` in real use — so this is a
   test we expect to win, which is the useful kind.
4. **`--threads 2`** against 18. Everything is GPU-resident; 18 threads may be
   contention rather than help.
5. **`--kv-unified`**, which may be inert at `-np 1`.

Needs a decision, not just a sweep:

6. **`--cache-ram 0` / `--ctx-checkpoints 0`** — RAM against re-prefill, and the
   answer depends on the window we serve, which is 5× theirs.
7. **The sampler.** This is a **quality** lever and quality is unmeasured on any
   artifact here. Copying it blind would replace one unchosen default set with
   another, borrowed one.

**Nothing here has been applied.** The profile is unchanged.


---

## The eight runs the developer made in Studio, and what each ACTUALLY ran

Eight chat threads named for a decoder — `Ngram`, `MTP no Tensor Split`,
`DSpark`, `MTP+Ngram`, `MTP`, `MTP+Ngram no Tensor Split`, `MTP+Ngram`,
`DFlash` — each given the same job (read a Markdown file through the
knowledge-base tool), all on `NVFP4-MTP-VERY-LOW`.

Studio relaunches `llama-server` when the configuration changes and logs the
argv, so each run can be matched to the configuration that served it. **Match on
the message's `responseDetails.startedAt`, not on thread creation time** — the
threads are created before the relaunch finishes, and matching on creation
attributes two of them to the wrong server.

| thread | sent | `-c` | `--spec-type` | `n-max` | split | prefill t/s | decode t/s | draft acc. |
|---|---|---|---|---|---|---|---|---|
| **MTP** | 01:45:13 | 40,704 | `draft-mtp` | 2 | tensor | 1,030 | **54.95** | 68.3 % |
| MTP+Ngram | 01:46:57 | 40,960 | `ngram-mod,draft-mtp` | 2 | tensor | 1,019 | 52.28 | 60.1 % |
| MTP+Ngram | 01:40:06 | 38,912 | `ngram-mod,draft-mtp` | 2 | tensor | 981 | 49.72 | 55.0 % |
| MTP+Ngram no Tensor Split | 01:41:33 | 62,208 | `ngram-mod,draft-mtp` | 2 | **layer** | **1,275** | 41.13 | 56.4 % |
| MTP no Tensor Split | 01:49:40 | 59,392 | `draft-mtp` | 2 | **layer** | 1,256 | 40.02 | 68.0 % |
| Ngram | 01:50:56 | 63,816 | `ngram-mod` | — | tensor | 1,071 | 30.14 | **0 drafts** |
| DSpark ⚠️ | 01:48:31 | 31,469 | **`--spec-default`** | — | tensor | 1,069 | 29.77 | 6.2 % |
| DFlash ⚠️ | 01:38:20 | 38,656 | **`--spec-default`** | — | tensor | 1,075 | 29.42 | 8.3 % |

### ⚠️ Two labels do not describe what ran

**`DFlash` and `DSpark` ran neither.** Both launches carry `--spec-default` and
**no `--spec-type` and no `-md`**. `draft-dflash` and `draft-dspark` each require
a drafter model file, and none was passed. Their 8.3 % and 6.2 % acceptance is
consistent with something weak, not with DFlash — which this project has
measured at high acceptance when it does run.

Those two rows say what `--spec-default` does on this artifact. They say nothing
about DFlash2 or DSpark.

### `-c` moved on every launch, and that is the biggest confound

31,469 · 38,656 · 38,912 · 40,704 · 40,960 · 59,392 · 62,208 · 63,816 — a **2×
range**. `gpuMemoryMode: "auto"` recomputes the window from whatever VRAM is free
at that moment, and `--tensor-split` moved with it (`7009,12462` … `7195,12544`).
**No two of these eight ran the same configuration in anything but name.**

The prompts were similar in size (`cache_n` 6,310–10,270), so the effect of the
allocated window on decode is probably small — but it is unquantified, and the
two `layer` runs happen to be the two deepest, which is exactly the direction
that would flatter the tensor split.

### What survives the caveats

These are single runs, unrotated, unpaired, with tool calls in the loop and a
window that changed underneath them. **They are hypotheses.** Two point the same
way as things this project measured independently:

- **`ngram-mod` alone produced ZERO drafts** on this task. Not a low acceptance —
  none generated. That is the same story as the served profile's real-use log,
  where `ngram-mod` fired **5 times in 4,653 calls** on agent traffic.
- **Layer split is slower at decode** — 40.02 against 54.95 with `draft-mtp`,
  −27 % — near this project's own paired **−31.0 %** on the same artifact.
  And it gives something we do not have: **layer is FASTER at prefill**,
  1,256 against 1,030, about +22 %. The paired split sweep here reported decode
  only.

One points the other way and is worth taking seriously:

- **`MTP` alone beat `MTP+Ngram`**, 54.95 against 52.28 and 49.72. This project
  measured the *pairing* as the result — but on a frozen corpus of vendor source
  code, where an exact 24-token match into context is common. On agent work it
  may be dead weight. **Testing this needs a regime that resembles agent traffic,
  and the arena does not have one** — running it on `real-code-vendor` would
  answer a question nobody asked.


---

## The complete Run settings panel, and where each value actually lives

Asked for in full because the UI shows many fields as `auto`, and `auto` is not
a value — it is a promise to compute one. Four different stores hold pieces of
this, and **they do not agree with each other**, so each row below names its
source.

**S** = `studio.db`  ·  **L** = webview Local Storage  ·  **A** = resolved
`llama-server` argv from the backend log  ·  **U** = visible in the UI only.

### Load / runtime

| field | UI shows | actually | src |
|---|---|---|---|
| GPU memory mode | `auto` | recomputes `-c` **and** `--tensor-split` on every launch — `-c` spanned **31,469 → 63,816** across eight runs | S, A |
| Checkpoints | `auto` | resolved to **`--ctx-checkpoints 0`** — off | U, A |
| Cache RAM | `auto` | resolved to **`--cache-ram 0`** — prompt cache off | U, A |
| KV cache dtype | — | `q4_0` / `q4_0` | S, L, A |
| Speculative type | `auto` | **the three stores disagree**: `app_settings` says `"ngram"`, Local Storage says `"mtp+ngram"` with `specDraftNMax: 3`, and the argv that actually ran says `--spec-type ngram-mod,draft-mtp --spec-draft-n-max 2` | S, L, A |
| Tensor parallel | — | `app_settings` `true`, Local Storage `false`, argv `--split-mode tensor` | S, L, A |
| GPU layers | — | `gpuLayers: -1` → `-ngl -1` | L, A |
| `nCpuMoe` | — | `0` | L |
| Parallel / batch / ubatch | — | stored `null`; argv resolved `--parallel 4`, no `-b`/`-ub` | L, A |
| Vision | — | `disableVision: false` → `--mmproj` passed | L, A |
| Advanced settings | on | — | U |
| Preset | `Default` | — | U |
| System prompt | empty | — | U |

### Sampling — none of it reaches the server as a flag

These are per-request values the app sends, not `llama-server` arguments.

| field | value | source |
|---|---|---|
| Temperature | **0.7** | S (`inferenceParams`), U |
| Top P | **0.8** | S, U |
| Top K | **20** | S, U |
| Min P | **0** | S, U |
| Repetition penalty | **Off** | U |
| Presence penalty | **1.5** | S, U |
| Max tokens | `Max` → **63,816**, and it tracks whatever `-c` `auto` chose | S, U |
| Seed | Random | S (`null`), U |

### Tools and retrieval — no effect on the server

`Auto-Healing Tool Calls` on · `Nudge Tool Calls` on · `Confirm tool calls` off ·
permissions `Approve for me` · max **25** tool calls per message · **5 min** per
call · search mode `Hybrid` · passages top-K **5** · auto-retrieve `Auto` at
threshold **0.70** · OCR scanned pages on · describe figures and charts on.

These shape the *prompt* the eight runs were given — 25 tool calls and hybrid
retrieval is why their `cache_n` and `predicted_n` vary so much — but they are
not comparable to anything in our profile, which serves an API and no tools.

### ⚠️ `Extra Arguments: --rope-scaling yarn --yarn-orig-ctx 32768`

**It has never been applied, and it would be wrong for this model.**

*Never applied*: the string appears **zero times** in all 19 logged
`llama-server` launches. It lives in the webview's `Web Data` — Chromium's
**autofill** store, origin `tauri.localhost` — which is remembered form input,
not application state. The model has not been reloaded since it was typed.

*Wrong for this model*: our own boot log reads the GGUF's own metadata —

```
n_ctx_train      = 262144        qwen35.context_length = 262144
n_ctx_orig_yarn  = 262144        freq_base_train       = 10000000.0
rope scaling     = linear        freq_scale_train      = 1
```

The model is **trained to 262,144**. `--yarn-orig-ctx 32768` asserts it was
trained to 32,768 and asks llama.cpp to stretch from there — a rescaling of
every position, on a model that needs none. llama.cpp is explicit in the other
direction on our own runs: *"n_ctx_seq (200704) < n_ctx_train (262144) — the
full capacity of the model will not be utilized."*

**Nothing here changes what this project serves.** The value of the whole panel
is that it is another team's answers for our artifact — and, in this one field,
a reminder that a setting sitting in a text box is not a setting in force.


---

## Three things only the full command line showed

The argv was truncated at `--chat-template-kwargs {` in the first extraction —
the regex stopped at the first quote inside the JSON. The tail changes the
picture in three places.

**1. `--mmproj` is on EVERY run, including the pure decoder comparisons.** The
vision tower was loaded for all eight. It costs 888 MiB and some of the `-c`
that `auto` computed; none of those eight numbers is a text-only configuration.

**2. One launch carries `--no-mmproj-offload`** — the vision tower on the CPU
rather than a card. That is the fallback this project would have needed if the
tensor split had refused to host the projector, which was the predicted failure
before it was measured. It exists and is reachable; we have not needed it.

**3. `--chat-template-kwargs {"enable_thinking": true, "preserve_thinking": true}`
on every Qwen launch**, regardless of the UI's `reasoningEnabled`. Thinking is
always on at the server. (One launch in the nineteen reads
`{"reasoning_effort": "high"}` — that is `gpt-oss-20b`, a different model.)

**And the `--rope-scaling` field is settled.** It appears in Studio's own source
— `studio/backend/core/inference/llama_cpp.py`, a test, and the frontend — so the
app supports the flag. It appears **zero times in the launch logs**, so no run
used it. Supported, entered, never sent.


---

## Studio documents every setting, so several of ours stop being guesses

Each field carries an `InfoHint` in the UI. Nineteen were read out of
`studio/frontend/src/features/model-picker/components/model-config-page.tsx`
(and siblings) rather than from screenshots. **These are the vendor's own words
about llama.cpp's behaviour, not measurements** — but where they state a
*default*, that is checkable fact we had been inferring.

### The four that change how our own numbers should be read

**Draft Tokens** — *"Max draft tokens per step. Leave blank for the default (MTP
and DFlash: **2 on GPU**, 3 on CPU/Mac; DSpark: 3)."*
So Studio's `--spec-draft-n-max 2` is **the documented default, not a tuned
choice.** Our 3 is the deviation. Our own real-use counters give acceptance per
position `(0.690, 0.448, 0.284)` — position 3 still lands 28 % of the time — so
the deviation looks earned. It is still unmeasured, and item 3 of the test list
above is now a test against a *default* rather than against someone's verdict.

**Speculative Decoding** — *"DSpark or DFlash when the model ships a drafter
sidecar, otherwise MTP / ngram. DSpark downloads a sidecar of about 11 GB and
DFlash one of about 1.5 GB … on quantized targets their greedy output can differ
from a non speculative run. **MTP and ngram do not change output.**"*

Two things follow. It confirms the `DFlash` / `DSpark` threads: no sidecar was
downloaded, so `auto` fell back — which is why those launches carry
`--spec-default` and no `-md`. And the last sentence bears directly on **the one
question gating everything here**: the vendor states that MTP and n-gram leave
output identical, while DFlash and DSpark can change it on a quantized target.
Our served configuration is MTP + n-gram. **Their claim, not our measurement**,
but it is the first outside statement that the decoder half of our profile is
quality-neutral.

**Tensor Parallelism** — *"No effect on a single GPU. On multi-GPU setups,
improves tokens/sec for **dense models**. MoE models don't benefit."*
Qwen3.8-27B is dense. That is the mechanism behind our measured **+31.0 %**, and
it says the result should not be expected to carry to an MoE.

**Checkpoints** — *"Context checkpoints kept per slot (`--ctx-checkpoints`),
which let a **sliding-window model** rewind instead of re-processing the prompt.
Each one costs host memory, and **models without a sliding window ignore the
setting**."*
Ours does **not** ignore it: our own log shows checkpoints created *and restored*
at positions 47,940–50,091, at ~350 MiB each with `max = 32`. So the host RAM is
buying real re-prefill avoidance on this model, and `--ctx-checkpoints 0` is a
trade rather than a free saving.

### The rest, verbatim, for the record

| setting | Studio's own description |
|---|---|
| **Max Seq Length** | Maximum context window size in tokens. Applies when the model loads. |
| **Context Length** | Drag all the way left for Auto, which chooses a context that fits **while prioritizing GPU speed**. Custom values request an exact context; higher values use more memory and may move model layers to system RAM. |
| **GPU Memory** | Default: Unsloth fits the model and context to your GPUs. Manual: set GPU Layers yourself. Auto lets llama.cpp size the context and offload overflow (including MoE experts) to RAM. |
| **KV Cache Dtype** | Lower KV precision to save VRAM at the cost of some quality. **f16 is the default**; **8-bit is the safest reduction**; q8_0 through iq4_nl are quantized. |
| **Spec Decoding KV Cache Dtype** | KV precision for the *draft model's own* context (`--spec-draft-type-k/-v`), separate from the target's. A quantized draft cache saves VRAM on a drafter whose output is verified by the target anyway. |
| **Parallel Slots** | `--parallel` decode slots for concurrent requests. **More slots share the context pool** and use more VRAM. |
| **Batch Size** | `--batch-size`. Default **2048**. Rarely needs changing; the micro-batch is what usually matters. |
| **Micro-batch Size** | `--ubatch-size`. Default **512**. Larger speeds up prompt processing but costs compute-buffer VRAM; capped at the batch size. |
| **Cache RAM** | Host MiB for prompt state evicted from a slot (`--cache-ram`), so a returning conversation is not re-processed. **0 disables, −1 lifts the limit.** |
| **Mmap/Mlock** | `--load-mode`. Auto picks **None** when it can prove the model fits without paging, **since a mapped read is slower**. |
| **Vision** | Loads the projector so the model can read images. Turning it off frees that VRAM, which can leave room for more layers. **Text generation is unaffected either way.** |
| **Extra Arguments** | Passed straight to llama-server **after** the settings above, so anything set in both is taken from here. Flags Unsloth owns — model, port, API key — are refused. |
| **GPUs / Chat Template** | candidate-pool selection; Jinja override applied at load. |

### One number of theirs cannot be read from ours

`--parallel 4` with `--kv-unified` — llama.cpp's help says the unified buffer is
"shared across all sequences", and Studio's own text says slots "share the
context pool". **Whether each of their conversations saw the full `-c` or a
quarter of it is not established from anything readable here**: their logs do
not capture llama-server's own `n_ctx_slot` line. Their `contextUsage.promptTokens`
of 10,231 against a `-c` of 40,960 is consistent with either. **Not resolved, and
it matters** — it decides whether their eight decode rates were taken at ~40k or
at ~10k of context.
