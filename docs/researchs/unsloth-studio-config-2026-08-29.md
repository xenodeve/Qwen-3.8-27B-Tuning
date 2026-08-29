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
- **The thinking state is ambiguous.** `chat_settings.reasoningEnabled` is
  `false` while the server is launched with `enable_thinking: true`. Qwen's
  published presets differ between thinking and non-thinking modes, and the NVFP4
  repo quotes `temperature 0.6, top_p 0.95, top_k 20, min_p 0` — a *different*
  preset from Studio's 0.7 / 0.8. **Which one belongs beside
  `--reasoning-effort medium` is not established.**

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
