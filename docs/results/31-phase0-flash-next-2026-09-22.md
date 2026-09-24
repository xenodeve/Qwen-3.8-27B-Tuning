# 31 — Phase 0A: artifact qualification — Qwen3.8-Flash-Next GSQ-RCO Q2_0

**Status: PASS (identity + integrity).** Downloaded and verified 2026-09-23 01:58,
from a NO-GO on 2026-09-22 when the artifact was absent.

Gate: [`plans/2026-09-22-Flash-Next-Phase0.md`](../plans/2026-09-22-Flash-Next-Phase0.md) §0A.

---

## Artifact identity — MEASURED HERE

| | |
|---|---|
| repo | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` |
| quant | `Q2_0` |
| revision fetched (HEAD at download) | **`2c4721899b4382bd07dfb61ae4fbad90c09caf7d`** |
| directory commit named in the plan | `8f752f8` — **not reconciled**, recorded as a difference (the plan named the commit that added the `Q2_0` directory; HEAD has moved) |
| local path | `C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0\` |

## Integrity — MEASURED HERE

Sizes are exact byte counts; SHA256 is computed locally and compared against the
**HF LFS metadata** for this revision, not against a local-only hash.

| file | bytes | GB | GiB | SHA256 | LFS match |
|---|---:|---:|---:|---|---|
| `…-Q2_0-00001-of-00002.gguf` | 37,623,740,192 | 37.62 | 35.04 | `69820c02…` | **PASS** |
| `…-Q2_0-00002-of-00002.gguf` | 28,800,138,432 | 28.80 | 26.82 | `316b46f3…` | **PASS** |
| **total** | **66,423,878,624** | **66.42** | **61.86** | | |

Full hashes:

- shard 1 `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720`
- shard 2 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`

`tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.rco-allocation.txt`
(39,062 bytes) fetched alongside; records the per-tensor allocation.

No partial download present. The two shards are the only weight files under the path.

> ⚠️ The `37.6 / 28.8 GB` figures in the plan are **decimal GB**, and match exactly.
> In GiB they are **35.04 / 26.82**. All internal memory arithmetic should use
> bytes/GiB; GB only when quoting the model card.

## How it was fetched — the instrument matters

Three attempts, recorded so nobody repeats them:

1. **`hf download` — stalls.** `huggingface_hub` 1.24.0 (XET disabled) creates the
   `.incomplete` placeholders, then transfers **0 bytes**, indefinitely (0 CPU, 0
   connections). Not used.
2. **`curl -C -` — stalls at ~16 KiB.** The xet-bridge CDN does not serve a
   `Range: bytes=0-` request for these objects properly. Not used.
3. **`curl` plain GET — works at ~11–12 MB/s.** 61.86 GiB in about 94 minutes
   (shard 1 00:24→01:15, shard 2 01:15→01:58).

Script: `qwen38-tuning/scripts/download-flash-next.ps1`.
Log: `qwen38-tuning/logs/flash-next-download.log`.

## PASS criteria

| criterion | result |
|---|---|
| 2 shards at the pinned names | yes |
| sizes consistent with 37.6 + 28.8 GB | yes — exact |
| both hashes computed | yes |
| hashes match upstream LFS | yes — both |
| no partial download | yes |
| revision + hashes recorded | yes (this file) |
| path recorded, and it is the path later phases will use | yes |

**0A PASS.**

---

## 0B — runtime / binary — PASS

MEASURED HERE, 2026-09-23.

| | |
|---|---|
| binary | `C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe` |
| `--version` | `0.3.0-dev (build 215, commit 9f55aee)`, MSVC 19.44.35228.0, *"Compiled by the Unsloth team"* |
| `BUILD_INFO.txt` | `b10679-mix-67dfc8b`; source commit **`b8472555738ca1d8782e56717ffa4095031ad4a1`**; CUDA toolkit **13.3**; built `2026-08-29T02:57:15Z`; `build_shared_libs: ON` |
| `UNSLOTH_PREBUILT_INFO.json` | asset `app-b10679-mix-67dfc8b-windows-x64-cuda13-newer.zip` |
| devices | `CUDA0` RTX 4070 SUPER (12,281 / 11,069 free) · `CUDA1` RTX 5060 Ti (16,283 / 15,172 free) |
| driver | 616.92, cc 8.9 / 12.0 |

**Provenance mismatch, recorded not reconciled:** the exe reports commit
`9f55aee`, the source tar reports `b847255`. Both are written down.

**CLI contract, the name that is actually on this binary:**

- `--tensor-read-lazy MODE` — `on` / `auto` / `off`, **default `auto`**
  (on for tensors > 4 GiB). **`--lazy-mode` and `-lzm` do not exist here.**
- `-ncmoe, --n-cpu-moe N` · `-ncffn, --n-cpu-ffn N` (dense; different tensor)
- `-sm, --split-mode {none,layer,row,tensor}` · `-ts` · `-fit/-fitt/-fitc`
- `-lm, --load-mode MODE` · `-np` · `-fa` · `-c`

## 0C — GPU / CUDA code objects — PASS

`ggml-cuda.dll` contains **40 × `sm_120a`** and **26 × `sm_89`** code objects —
both cards covered. Read from the file, not from the manifest.

## 0D — split-mode constraints — PASS

| test | result |
|---|---|
| `-sm row` | **fails in <1 s**: `llama_model_load: error loading model: device CUDA0 does not support split buffers` — same as the 27B |
| `--fit` | **defaults to ON** in this build, and its fitting step errors under tensor (`common_fit_params: encountered an error…`). `-fit off` is passed so placement is stated, not inferred |
| chosen split | **`-sm tensor -ts 11069,15172`** — see why below |

## 0E — 16K smoke — PASS (first usable run)

`-ncmoe` sweep at ctx 16,384, both cards, `-fit off`, KV `q4_0`:

| `-sm` | `-ncmoe` | load | total VRAM | device0 free / device1 free | RAM |
|---|---:|---|---:|---|---:|
| layer | 40 | 21 s | 11.0 GiB | 8,848 / 7,933 | 34.6 GiB |
| layer | 34 | 21 s | 14.96 GiB | 8,847 / 3,881 | 38.1 GiB |
| layer | 32 | 33 s | 16.3 GiB | 8,829 / 2,531 | 39.8 GiB |
| layer | 30 | 21 s | 17.6 GiB | 8,848 / 1,183 | 40.6 GiB |
| layer | 26 | 12 s | — | **CUDA OOM** | — |
| tensor | 34 | 36 s | 16.66 GiB | 3,162 / 7,825 | 38.1 GiB |
| tensor | 26 | 39 s | 21.91 GiB | **484** / 5,125 | 43.0 GiB |
| tensor | 20 | 12 s | — | **CUDA OOM** | — |

**Under `-sm layer` the 4070 SUPER is nearly idle** — 3,150 MiB used against
8,848 free — because `-ncmoe` offloads the *first* N layers and the layer split
puts those on device 0. `-sm tensor` splits across both cards, so both work.

**Chosen for the first usable run: `-sm tensor`, `-ncmoe 34`, ctx 16,384.**
Not the fastest number in the table — it is the one with real headroom on both
cards (device0 3,162 free, device1 7,825 free), so it is the honest starting
point for optimization rather than a peak to defend.

**Working run, verified:**

```
content   : def fibonacci(n: int) -> int:
                """Return the nth Fibonacci number…"""
decode    : 6.66 tok/s
prefill   : 6.50 tok/s
VRAM      : 17,062 MiB total (4070S 8,836 / 5060 Ti 8,226)
RAM       : 38.1 GiB
model meta: arch qwen4exp, n_params 176,943,899,520, n_ctx_train 262,144,
            n_embd 2,560, n_vocab 248,320, ftype Q2_0, size 66,412,853,760
```

Higher `n_cpu_moe` does not rescue the rate (tensor 34 = 6.66, tensor 26 = 7.99),
so the ~7–8 tok/s ceiling at this configuration is the CPU-resident experts and
host RAM bandwidth, not one mis-set flag.

Launcher: `qwen38-tuning/scripts/serve-flash-next.ps1`, reachable as hub key `I`
via `launchers\serve-flash-next.bat`.

## 0F — lazy / mmap residency — NOT DONE

Shard 2 has not yet been watched to prove it stays off-heap, and no cold/warm
comparison of `--tensor-read-lazy auto` vs `on` has been run. Phase 0 is
therefore **not fully closed**; the optimization work starts from 0E's config.
