# 40 — Qwen3.8-Flash-Next GSQ-RCO Q2_0 — Phase 0 handoff for an optimization study

**Written 2026-09-23.** This is a self-contained brief: everything a fresh reader
needs to understand what is running, what was measured, and where the speed is
going — so an optimization pass can start without re-deriving the ground state.

**Read this with** [`results/09-hardware.md`](../results/09-hardware.md) (which card
produced which numbers), [`CORRECTIONS.md`](CORRECTIONS.md) (claims this project
retracted), [`agents/traps.md`](../agents/traps.md) (ways of working that failed
here), and [`39-OPTIMISATION-GUIDE.md`](39-OPTIMISATION-GUIDE.md) (levers already
settled *for the 27B* — none of those verdicts transfers to this artifact).

**State in one line:** the model loads, serves an OpenAI-compatible API, and answers
correctly at 16K in **~7 tok/s decode / ~6.5 tok/s prefill**. It works; it is not
yet tuned. That rate is a floor to move, not a result to report.

---

## 1. What is running, exactly

### Artifact

| | |
|---|---|
| repo | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` |
| quant | `Q2_0` (target whole-file **2.40 bpw**) |
| revision fetched | `2c4721899b4382bd07dfb61ae4fbad90c09caf7d` |
| directory commit the plan named | `8f752f8` — **not reconciled**, recorded as a difference |
| arch | **`qwen4exp`** |
| path | `C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0\` |
| shard 1 | `…-Q2_0-00001-of-00002.gguf` — **37,623,740,192 B** (35.04 GiB) — the resident transformer |
| shard 2 | `…-Q2_0-00002-of-00002.gguf` — **28,800,138,432 B** (26.82 GiB) — the n-gram/PLE table |
| sha256 shard 1 | `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720` |
| sha256 shard 2 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| both hashes | **match HF LFS metadata** (verified 2026-09-23) |
| tensor-allocation file | `…-Q2_0-allocation.txt` (39,062 B) — per-tensor quant map, on disk |

### What the runtime reports about it (`/v1/models`, MEASURED)

```
n_params     176,943,899,520      (176.9 B)
n_ctx_train  262,144
n_embd       2,560
n_vocab      248,320
ftype        Q2_0
size         66,412,853,760 bytes
```

### Architecture (VENDOR — Qwen/ISTA model cards; **not** measured here)

- 125 B MoE language model, **6 B activated**, plus **51 B n-gram embedding** and a **4 B MTP head**.
- **48 layers = 36 Gated DeltaNet (linear attention) + 12 Qwen Sparse Attention (QSA, compression ratio 4)**.
- MoE: **512 experts, top-10 routed**, in every layer. Shared-expert tensors (`_shexp`) exist beside routed ones (`_exps`).
- PLE table: `per_layer_token_embd.weight`, shape `[160, 320001536]` — **lookup-only, never in a matmul**. About 16 rows are gathered per generated token.
- KV cache exists **only in the 12 QSA layers**: 36 of 48 layers are GDN and carry a fixed-size recurrent state, not growing KV. **Any KV scaling assumed for a 48-layer dense transformer is wrong here.**

### Quant census (from the allocation file — MEASURED counts, VENDOR names)

1223 tensors. `Q2_0`=202, `Q3_K`=91, `Q4_K`=38, `IQ4_XS`=56, `Q4_0`=16, `Q5_K`=16, `Q6_K`=12, `IQ4_NL`=7, `Q5_0`=7, `Q8_0`=2, `F16`=1, **`BF16`=483, `F32`=292**.

> ⚠️ **484 high-precision tensors (483 BF16 + 292 F32 across that list) are not
> 2-bit.** Norms and router/gate tensors commonly live there. The researcher should
> check whether a few large BF16 tensors (e.g. `ffn_gate_inp.weight`, the `output_hc_*`
> family, `attn_gate`) are a meaningful share of the 35 GiB resident shard — the
> quant-type *counts* do not tell us their *bytes*.

Observed tensor families (from the allocation head): `attn_gate`, `attn_qkv`,
`ffn_down_exps`, `ffn_gate_exps`, `ffn_gate_inp`, `ffn_gate_inp_shexp`,
`ffn_gate_shexp`, `ffn_down_shexp`, `output.weight`, `output_hc_down/up/norm`,
`token_embd.weight`. No `ffn_up_exps` was seen in the head; whether gate/up are
fused is **unconfirmed**.

---

## 2. The machine (MEASURED HERE unless marked)

| | |
|---|---|
| CPU | Intel Core i5-13500 — 6 P-cores + 8 E-cores, 14 cores / 20 threads |
| RAM | 48 GB DDR5 (KLEVV CRAS V RGB). **~108 GB/s** is a *plan claim, not measured here* |
| OS | Windows 11 |
| GPU 0 | **RTX 4070 SUPER 12 GB**, cc 8.9, driver **616.92**. llama.cpp sees **12,281 MiB total / 11,069 MiB free**. This is the **display GPU** (idle desktop load ≈ 1,037–1,093 MiB). PCIe **gen4 x16** (measured under load, `09-hardware.md`) |
| GPU 1 | **RTX 5060 Ti 16 GB**, cc 12.0, driver 616.92. llama.cpp sees **16,283 MiB total / 15,172 MiB free**. PCIe **gen4 x4** (measured under load) — the link carrying one card is narrow, which matters for any per-layer cross-device traffic |
| topology | `PXB`, several bridges, **no NVLink / no peer link** |
| free space | C: 91.7 GiB (models + HF cache live here) · D: 58.1 · F: 106.3 |

**VRAM budget policy carried from the plan (not derived here):**
preferred total dedicated VRAM ≤ **25.5 GB**; absolute ceiling **26.0 GB**; do not
optimize toward 27–28 GB. At 16K the working config uses **17,062 MiB**, so there is
roughly **8 GiB of budget in hand** before that ceiling — the depth question is
open, not answered.

---

## 3. The runtime — and the traps in it

### Binary (MEASURED)

| | |
|---|---|
| path | `C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe` |
| `--version` | `0.3.0-dev (build 215, commit 9f55aee)`, MSVC 19.44.35228.0, *"Compiled by the Unsloth team"* |
| manifest | release `b10679-mix-67dfc8b`, **source commit `b8472555738ca1d8782e56717ffa4095031ad4a1`**, CUDA toolkit **13.3**, built `2026-08-29T02:57:15Z`, `build_shared_libs: ON` |
| code objects | `ggml-cuda.dll` → **40 × `sm_120a` + 26 × `sm_89`** (both cards covered) |
| tools in `bin/` | **only `llama-server.exe`** — no `llama-cli`, no `llama-bench`, no `gguf-dump` |
| provenance | exe reports `9f55aee`, source tar reports `b847255` — **recorded, not reconciled** |

This is a **prebuilt**, not built here. There is a matching source tree at
`C:\AI\llama.cpp-unsloth-mirror` (has `LLM_ARCH_QWEN4EXP` and `--tensor-read-lazy`).

**The older trees are stale and must not be used for this model:**
`C:\AI\llama.cpp\build-blackwell` and `build-dflash2` are commit `1deefcca3` and
**fail with `unknown model architecture: 'qwen4exp'`**. They stay pinned for the
Qwen3.8-27B / DFlash2 work.

### CLI contract — the names actually on this binary (MEASURED)

- **`--tensor-read-lazy MODE`** — `on` / `auto` / `off`, **default `auto`** (on for tensors > 4 GiB). **`--lazy-mode` and `-lzm` do not exist here** — do not pass them.
- `-ncmoe, --n-cpu-moe N` — MoE experts of the **first N layers** to CPU.
- `-ncffn, --n-cpu-ffn N` — **dense** FFN weights (a different tensor family; see §7).
- `-cmoe, --cpu-moe` — all experts to CPU.
- `-sm {none,layer,row,tensor}` · `-ts` · `-mg` · `-fit [on|off]` · `-fitt` · `-fitc`
- `-lm, --load-mode {auto,none,mmap,dio}` · `-fa` · `-c` · `-np` · `-t` · `-tb` · `-b` · `-ub` · `-ngl` · `-ctk` · `-ctv`

### Hard constraints already reproduced (MEASURED)

| constraint | evidence |
|---|---|
| **`-sm row` cannot load** | fails in **<1 s**: `llama_model_load: error loading model: device CUDA0 does not support split buffers`. Same failure as the 27B pair. |
| **`--fit` defaults to ON** | and its fitting step **errors under `-sm tensor`** (`common_fit_params: …`). Pass `-fit off` so placement is stated rather than inferred. |
| **`-sm tensor` cannot host an external drafter** | this is the **27B's** measured result (both `draft-mtp` and `draft-dflash` abort in `ggml-backend-meta.cpp`). **Not re-tested on this model** — re-test before assuming. `ngram-mod` needs no weights and is untried here. |
| mmap + CPU overrides warning | `tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance`. **Caveat:** `none` may break `--tensor-read-lazy` (it requires mmap). Untested trade. |

---

## 4. The working configuration

Launcher: **`qwen38-tuning/scripts/serve-flash-next.ps1`**, reachable as hub key
**`I`**. Exact argv used for the verified run (the split below is computed from
live VRAM at launch; the measured run used `-ts 11069,15172`):

```powershell
llama-server.exe `
  -m "C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" `
  -c 16384 `
  -ctk q4_0 -ctv q4_0 `
  -b 2048 -ub 512 `
  -ngl all -ncmoe 34 `
  -sm tensor -ts 11069,15172 `
  -fit off `
  -lm mmap --tensor-read-lazy on `
  -fa on -np 1 `
  -t 14 -tb 20 `
  --host 127.0.0.1 --port 8099
```

**Measured result of that run:**

| | |
|---|---|
| load | 36 s |
| decode | **6.66 tok/s** |
| prefill | **6.50 tok/s** (73-token prompt) |
| VRAM total | **17,062 MiB** — 4070S 8,836 used / 3,162 free · 5060 Ti 8,226 used / 7,825 free |
| RAM used | 38.1 GiB (≈23.5 GiB with no model resident) |
| output | `def fibonacci(n: int) -> int:` with a correct docstring — a smoke test, not a quality result |

---

## 5. The `-ncmoe` × `-sm` map — the whole of Phase 1's input

All at **ctx 16,384**, KV `q4_0`, `-fit off`, both cards, same artifact (MEASURED):

| `-sm` | `-ncmoe` | load | VRAM total | 4070S used/free | 5060 Ti used/free | RAM |
|---|---:|---|---:|---|---|---:|
| layer | 40 | 21 s | 11.0 GiB | 3,150 / 8,848 | 8,118 / 7,933 | 34.6 GiB |
| layer | 34 | 21 s | 14.96 GiB | 3,151 / 8,847 | 12,170 / 3,881 | 38.1 GiB |
| layer | 32 | 33 s | 16.3 GiB | 3,169 / 8,829 | 13,520 / 2,531 | 39.8 GiB |
| layer | 30 | 21 s | 17.6 GiB | 3,150 / 8,848 | 14,868 / 1,183 | 40.6 GiB |
| layer | 26 | 12 s | — | **CUDA OOM** | — | 23.4 GiB |
| tensor | 34 | 36 s | **16.66 GiB** | 8,836 / 3,162 | 8,226 / 7,825 | 38.1 GiB |
| tensor | 26 | 39 s | 21.91 GiB | 11,514 / **484** | 10,926 / 5,125 | 43.0 GiB |
| tensor | 20 | 12 s | — | **OOM** (device0 asked 12,012 MiB) | — | 23.8 GiB |
| tensor | 14 | 18 s | — | exited, `GGML_ASSERT(…) failed` | — | 23.3 GiB |

### The two findings this table contains

1. **Under `-sm layer`, the 4070 SUPER is nearly idle** — ~3,150 MiB used against
   8,848 free — because `-ncmoe` offloads the **first** N layers and the layer split
   puts those on device 0. **Any `-sm layer` result on this model is measuring one
   GPU plus host RAM.** `-sm tensor` uses both cards and is the correct base.
2. **The rate floor does not move much with `-ncmoe`.** tensor 34 → **6.66** tok/s;
   tensor 26 → **7.99** tok/s. Moving 8 layers of experts back onto the GPUs bought
   ~20 %, while the 4070 SUPER's free VRAM fell to 484 MiB. **The bottleneck is not
   the GPU/CPU split ratio; it is something the split does not change.**

---

## 6. Where the time is going — hypotheses, not conclusions

Ordered by how well the existing evidence points at each. **None of these is
measured.** Each names the experiment that would settle it.

1. **The PLE lazy read path (strongest suspect).** Shard 2 is 26.82 GiB of
   lookup-only table, read through mmap with `--tensor-read-lazy on`. Prefill is
   **6.5 tok/s**, which is *lower than decode per token* in one reading — the
   signature of a per-token disk/page-fault cost, not a compute cost. On Windows a
   ~90-byte row can cost a 4 KiB page fault. The upstream direct-read PR (#28136,
   ~20–37 % cold-prefill gains elsewhere) **is not in this binary**.
   *Experiment:* sample page-faults/sec, disk-read B/s and process working set
   cold vs warm; compare `--tensor-read-lazy on` vs `off` (off needs the table
   resident — ~27 GiB of RAM, which is available-ish at 38 GiB used of 48).
2. **Host RAM bandwidth for the CPU-resident experts.** 34 of 48 layers' experts
   stream from RAM per token. ~108 GB/s (claim) against a large working set is a
   plausible ceiling around 7–8 tok/s. *Experiment:* measure decode at `-ncmoe 48`
   (all CPU) vs 34 vs 26 to get the slope, and watch whether decode tracks
   `n_cpu_moe` linearly.
3. **The 5060 Ti is on PCIe gen4 x4.** Any per-layer (not per-tensor) cross-device
   traffic in tensor mode pays ~7.9 GB/s on that link. *Experiment:* profile PCIe /
   compare `-sm tensor` vs a `-sm layer` variant that actually uses both cards.
4. **KV is only 12 QSA layers** (§1), so KV is unlikely to be the 16K cost — but
   this also means **context may be much cheaper than expected**, and the depth
   ladder may go far beyond 16K. *Experiment:* the depth ladder (§7).
5. **484 high-precision (BF16/F32) tensors** may be a non-trivial byte share of
   shard 1. *Experiment:* sum bytes by quant type from the allocation file + tensor
   shapes.

---

## 7. Open questions, in the order they should be closed

1. **0F — PLE residency.** Prove shard 2 is not pulled resident: no ~27 GiB jump in
   the process working set; disk reads during inference and bounded; cold vs warm
   differ. Compare `--tensor-read-lazy auto` vs `on` (the table is >4 GiB, so `auto`
   should already be on — confirm rather than assume).
2. **Depth ladder.** 16K → 32K → 64K → 128K → 192K → 224K → 256K. Measure KV-per-token
   for this arch (12 QSA layers, c4 compression — expect it to be cheap). Watch the
   25.5 GB budget. **Do not assume the 27B's numbers.**
3. **`-ncffn` vs `-ncmoe`.** Read out which qwen4exp tensors llama.cpp classifies as
   dense FFN vs MoE experts before turning `-ncffn` — moving the wrong family will
   look like a memory win and be a placement bug.
4. **Decoder.** `ngram-mod` first (no weights). Whether `draft-mtp` / DFlash2 load
   under `-sm tensor` on **this** build is unmeasured. The MTP head is 4 B per the
   card — confirm whether it is inside shard 1 at all.
5. **Ubatch / batch.** `-ub 1024` beat `-ub 256` on the 27B tensor split by +10.1 %
   prefill; whether that transfers here is unknown.
6. **Threads.** i5-13500, 14C/20T. `-t 14 -tb 20` was inherited from the 27B, not
   chosen here. The P/E-core split matters for the CPU-resident experts.
7. **`-ts` ratio.** 11069,15172 was chosen from free VRAM, not swept. On the 27B
   NVFP4 artifact the ratio was a **sharp peak** (±1 pp ≈ 18 %); treat it as a
   candidate lever, not a plateau.
8. **`--load-mode none`** to drop the mmap warning — but check it does not disable
   `--tensor-read-lazy`.
9. **Quality.** Nothing about this artifact's output quality has been measured. One
   correct one-liner is not a quality result.

---

## 8. Constraints any measurement here must respect

From this repo's own record — these are not style notes, they are what produced the
thirteen documented instrument faults.

- **No verdict before evidence.** Every number names the file it came from.
- **A verdict at one depth does not transfer.** `draft-mtp` was +81 % at 16K and
  −71 % at 131,072 on the 27B.
- **Never compare raw decode across boots.** The spread is real and the cause is
  unknown. At ctx 16,384 the retired floor was **13.6 %** (Ada) — but this is a
  different machine configuration; **re-derive the floor for this artifact before
  quoting any effect**. At 65,536 the old card spanned up to **48.9 %** on the same
  arm with identical counters.
- **Pair within a round, alternate the order.** Speculative decoders make tok/s
  track acceptance, so two arms that produced different text are not comparable.
- **`--fit` reads a number `nvidia-smi` does not show** (11,069 vs 12,281 in every
  log). Ask llama.cpp, not the driver, for a VRAM figure that matters.
- **Two orchestrators cannot share port 8080.** This work used **8099**.
- **Verify every subagent report.** A delegated run in this repo once reported a task
  complete while the file sat in the wrong directory.

### Prior 27B results — **hypotheses only, do not inherit**

- `-sm tensor` beat `-sm layer` by **+65.4 %** at 147,456 on `UD-Q4_K_XL`; tensor also
  could not host a drafter on that model.
- `-ub 1024` beat `-ub 256` by **+10.1 % prefill**, decode flat, on the tensor split.
- `q8_0` KV **could not load at 147,456** on the 27B tensor split. (That row is also
  flagged as confounded in `results/README.md`.)
- The noise floor printed by the arena, `NOISE_FLOOR_PCT = 13.6`, is an **Ada figure
  at 16,384** and is stale.

---

## 9. Reproduce / operate

```powershell
# serve (first working config) -- hub key I, or directly:
C:\AI\qwen38-tuning\scripts\serve-flash-next.cmd
C:\AI\qwen38-tuning\scripts\serve-flash-next.ps1 -Ctx 16384 -NcMoE 34

# boot-test harness: loads, reports VRAM/RAM, tries a completion, stops
C:\AI\qwen38-tuning\scripts\fn-boot.ps1 -NcMoE 34 -Ctx 16384 -Sm tensor -Ts "11069,15172"

# fetch (if the artifact must be re-downloaded)
C:\AI\qwen38-tuning\scripts\download-flash-next.ps1
```

**Download gotchas (MEASURED, save an afternoon):**
`hf download` (huggingface_hub 1.24.0) creates the `.incomplete` placeholders and
then transfers **0 bytes**. `curl -C -` stalls at ~16 KiB — the xet-bridge CDN does
not serve `Range: bytes=0-` for these objects. **A plain `curl -L -f` GET works at
~11–12 MB/s** and fetched all 61.86 GiB in ~94 minutes.

**Files:**
- `docs/results/31-phase0-flash-next-2026-09-22.md` — 0A–0E, the full evidence
- `docs/plans/2026-09-22-Flash-Next-Phase0.md` — the gate definition
- `docs/plans/2026-09-22-Qwen3.8-Flash-Next-Q2_0-Optimization-Plan.md` — the main plan
- `qwen38-tuning/scripts/serve-flash-next.ps1` (the recipe), `serve-flash-next.cmd`
  (the hub entry point), `fn-boot.ps1` (boot sweep harness), `download-flash-next.ps1`
- `launchers/serve-flash-next.bat`, `launchers/serve-flash-next-lan.bat` — hub key `I`

---

## 10. What the study should deliver

A table of configurations at **16K, 64K and one deep depth**, each with load time,
per-GPU VRAM, RAM, prefill, decode and TTFT, paired and repeated enough to clear a
**floor derived on this artifact**; a named answer to which of §6's suspects owns
the 7 tok/s; and three candidate profiles — **max-context, daily, max-speed** — with
the command line for each. Quality is unmeasured and remains the gate on calling
any of them better than the 27B incumbents.
