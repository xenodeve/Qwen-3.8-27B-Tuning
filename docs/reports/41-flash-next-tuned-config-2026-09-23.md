# 41 — Qwen3.8-Flash-Next GSQ-RCO Q2_0 — tuned configuration report

**Written 2026-09-23.** One file: what config, what flags, what versions, how it is
measured, and what the numbers are. Everything below marked **MEASURED** came from
this session's own logs (`qwen38-tuning/results/flash-next-optimization/`);
anything marked **VENDOR** is an outside claim recorded, not verified.

**The tuned launch line in one copyable block:**

```powershell
$env:CUDA_VISIBLE_DEVICES = 'GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4,GPU-059b90e2-2b5c-00b8-f3ba-f6dea8de083e'
C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe `
  -m C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf `
  -c 16384 -ngl all `
  -ncmoe 24 `
  -sm tensor -ts 8500,16000 -fit off `
  -ctk q4_0 -ctv q4_0 `
  -b 2048 -ub 512 `
  -lm mmap --tensor-read-lazy on `
  -fa on -np 1 `
  -t 14 -tb 20 `
  --host 127.0.0.1 --port 8080 `
  --alias Qwen3.8-Flash-Next-Q2_0
```

Installed as the recipe `qwen38-tuning/scripts/serve-flash-next.ps1`, reachable as
hub key `I` (`serve-hub.bat` → `launchers\serve-flash-next{,-lan}.bat` →
`qwen38-tuning/scripts/serve-flash-next.cmd`).

---

## 1. Versions and provenance

| | | tag |
|---|---|---|
| binary | `C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe` — `0.3.0-dev (build 215, commit 9f55aee)`, Unsloth team, CUDA 13.3, built 2026-08-29 | MEASURED |
| source provenance | source commit `b8472555738ca1d8782e56717ffa4095031ad4a1`, manifest release `b10679-mix-67dfc8b`; **exe commit disagrees with tar commit (9f55aee vs b847255), recorded as a mismatch, not reconciled** | MEASURED |
| binary sha256 | see `run-manifests/` (written per boot) | MEASURED |
| CUDA code objects | `ggml-cuda.dll` carries 40 × `sm_120a` and 26 × `sm_89` — covers 5060 Ti (sm_12.0) and 4070 SUPER (sm_8.9) | MEASURED |
| llama.cpp device order | CUDA0 = RTX 4070 SUPER 12 GB (display), CUDA1 = RTX 5060 Ti 16 GB | MEASURED |
| CPU | i5-13500, 14C/20T, 48 GB DDR5-7200 | MEASURED |

**Binary must stay on `build-mirror`.** The old 27B trees (`llama.cpp`, `llama.cpp-mirror`, `llama.cpp-dflash2`, commit `1deefcca3`) do NOT have arch `qwen4exp` and fail with `unknown model architecture`. Do not repoint this recipe at them.

## 2. Artifact identity

| | | tag |
|---|---|---|
| repo | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | MEASURED |
| quant | `Q2_0`, target whole-file 2.40 bpw | MEASURED |
| revision fetched | `2c4721899b4382bd07dfb61ae4fbad90c09caf7d` | MEASURED |
| arch | `qwen4exp` | MEASURED |
| path | `C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0\` | MEASURED |
| shard 1 | `…-Q2_0-00001-of-00002.gguf` — 37,623,740,192 B (35.04 GiB) | MEASURED |
| shard 2 | `…-Q2_0-00002-of-00002.gguf` — 28,800,138,432 B (26.82 GiB) — the PLE/n-gram table | MEASURED |
| sha256 both shards | match HF LFS metadata (verified 2026-09-23) | MEASURED |
| architecture model | 125B MoE (512 experts, top-10, ~6 B active) + ~51B PLE/n-gram + 4 B MTP head; 48 layers = 36 Gated DeltaNet + 12 QSA; **KV exists only in the 12 QSA layers** | VENDOR |

## 3. The tuned flags — what each one does and why

| flag | value | why (from measurement) |
|---|---|---|
| `-c` | 16384 | the only currently measured context depth. Storage at 32K exceeds this build's 16K ctx limit; ladder comes later, separately |
| `-ngl all` | — | without it the attention layers fall to CPU;
 **measured: with ngl all decide/s jasmine speed triples** |
| `-ncmoe` | 24 | ▲ from baseline 34. Decode slope: 48→5.5, 40→6.5, 34→7.1, 26→8.4, **24→10.7 tok/s (3 confirmations)**. 22 and 20 are CUDA OOMs — do not replicate without re-measuring |
| `-sm` | tensor | under -sm layer the 4070 SUPER sits nearly idle, because first-N `-ncmoe` offload maps onto the first layers that the layer split assigns to device0 — only tensor split uses both GPUs |
| `-ts` | 8500,16000 | frozen, from the sweep. FROZEN beats live-computed: at ncmoe 24, 8500,16000 measured 8.72/11.71/11.83 tok/s decode; the live-computed values 8449,15<sup>452</sup>/nearby were behind |
| `-fit off` | — | `--fit` defaults ON in this build and errors under split=tensor; this recipe always states placement rather than guessing |
| `-ctk q4_0 -ctv q4_0` | — | keep KV cache at q4_0 for the 12 QSA layers; cheaper/balanced for the coding-agent use case |
| `-b 2048 -ub 512` | — | matches 0E frozen entry; do not transfer a 27B ubatch verdict to this artifact |
| `-lm mmap` | — | shards 1-2 arrive via file-backed path; run warns "consider --load-mode none" — that suggestion should be evaluated later, separately |
| `--tensor-read-lazy on` | — | keep shard 2 (26.8 GiB) off the heap; verify with 0F later |
| `-fa on` | — | flash-attention on |
| `-np 1` | — | single request at a time for the Code-Agent use case |
| `-t 14 -tb 20` | — | decode threads sweep at the tuned tuple: t8→7.5, t12→9.1, t14→**10.75**, t16→9.4 tok/s. tb=14 → 9.8, lower. Keep 14/20 |
| `--alias` | Qwen3.8-Flash-Next-Q2_0 | stable model name in `/v1/models` |

## 4. Measured result (ctx 16,384, Q4 KV, fit off, correct answer)

| config | decode mean tok/s | cold prefill tok/s | notes |
|---|---:|---:|---|
| baseline: `-ncmoe 34` `-ts 11069,15172` | 6.66 | 6.50 | the Phase 0 smoke floor |
| live-computed: `-ncmoe 34` `-ts 8449,15452` | 7.28 | 7.98 | split-computed variant |
| trend: `-ncmoe 24` `-ts 8500,16000` (tuned) | **10.75** | **20–23** (4K=20.5, 16K=22.9) | the winner; 3 confirmations |

Noise floor for decode on THIS artifact, measured in the same boot: 6.19 / 7.66 / 7.51 tok/s — range ~23% between adjacent reps at the worst config. Guideline: differences under ~10% are within noise unless you have more repeats.

**The above compares:**
- decode gain vs Phase 0 floor: **+61%**
- cold prefill gain vs Phase 0 floor: **3×** (20–23 vs 6.5)

---

## 4a. Deep memory and context record — every number, per boot

All values below are **MEASURED** from this session's request logs
(`qwen38-tuning/results/flash-next-optimization/` — one row per request, in
`results.csv` / `results-fixed.csv`; llama-server stdout in `logs/<run>.log`,
stderr in `logs/<run>-err.log`; one manifest per boot in
`run-manifests/<run>.json`). The tuned tuple = `-ncmoe 24 -sm tensor -ts
8500,16000 -t 14 -tb 20 -b 2048 -ub 512` at ctx 16,384.

### 4a.1 Context windows

| quantity | value | tag |
|---|---|---|
| allocated context (`-c`) | **16,384 tokens** — size at 32K exceeds this build's ctx ceiling; 32K-depth at ctx 16K is an exceed_context_error from the tuner (24114 > 16384). Ladder is a separate measurement, later |
| model train context (`n_ctx_train`) | **262,144** | from `/v1/models`, MEASURED at runtime |
| max prompt tested | 12,090 tokens (the "16K" file, 62 KB English) | MEASURED |
| cached-token behavior | in the SAME boot, requests 2..n of the same prompt hit KV prefix cache and finished in ~6 s: the false impression of a slow-warm prefill comes from KV cache measurements, not "slower work" | MEASURED (`cache_n` in timings) |
| KV cache type | `-ctk q4_0 -ctv q4_0`; only the **12 QSA layers** grow KV — 36 GDN layers carry fixed recurrent state (VENDOR arch model) |

### 4a.2 Drafter decode (MTP) — NOT MEASURED

- The artifact has a **4 B MTP head** (VENDOR claim; not exercised here). This session did **zero** drafter/speculative measurement: the plan (§24) places speculation *after* base inference. So **"drafter decode" has no measured value in this report** — that data does not exist yet, by honest labelling.
- The mechanism to measure it *later*, when the base is stable, is llama.cpp speculation with the MTP weights from shard 1 (`--spec-type` / draft-MTP semantics; the hub exposes a `-Mtp` choice node the docs conservatively reserve). Both flags and acceptance-rate telemetry need to be re-derived for this artifact — **do not copy draft-mtp verdicts from the 27B work** (different family, different head).

### 4a.3 VRAM status (per run — MEASURED)

Total card sizes: GPU0 = RTX 4070 SUPER **12,282 MiB total** (display card, idle ≈ 1,084); GPU1 = RTX 5060 Ti **16,283 MiB total** (idle ≈ 87).

| run (config) | GPU0 used / free | GPU1 used / free | **total dedicated** | decode mean | notes |
|---|---:|---:|---:|---:|---|
| `-ncmoe 48` (all-MoE CPU) | 3,850 / 8,432 | 3,646 / 12,637 | 7,496 MiB (7.3 GiB) | 5.54 | smallest VRAM, slowest decode |
| `-ncmoe 40` | 5,480 / 6,802 | 7,432 / 8,851 | 12,912 MiB | 6.54 | |
| `-ncmoe 34` `-ts 8449,15452` (baseline-ish) | 6,692 / 5,590 | 10,268 / 6,015 | **16,960 MiB (16.6 GiB)** | 7.12 | the Phase 0 NTuned floor |
| `-ncmoe 26` `-ts 8449,15452` | 8,313 / 3,969 | 14,048 / 2,235 | **22,361 MiB (21.8 GiB)** | 8.39 | |
| `-ncmoe 26` `-ts 9000,16000` | 8,293 / 3,989 | 14,048 / 2,235 | 22,341 MiB | 8.59 | best among the ncmoe-26 ts sweep |
| `-ncmoe 26` `-ts 10000,16000` | 8,440 / 3,842 | 13,910 / 2,373 | 22,350 MiB | 7.98 | more VRAM to GPU0 did not help — the x4 card is the gain side |
| `-ncmoe 24` `-ts 9000,16000` | 8,757 / 3,525 | 14,992 / 1,291 | 23,749 MiB | 8.53 | |
| **`-ncmoe 24` `-ts 8500,16000` (TUNED)** | **8,772 / 3,510** | **14,992 / 1,291** | **23,764 MiB = 23.21 GiB** | **10.84 (6 requests)** | the promoted daily profile, 1,291 MiB free headroom on GPU1 |
| `-ncmoe 22` and lower | — | — | — | — | CUDA OOM: GPU0 (CUDA 0) or GPU1 cannot hold the slab; the boundary is exactly between 24 (fits, fast) and 22 (OOM, both cards) |

**Policy check (tuned tuple):** 23,764 MiB total dedicated — **within the ≤ 25.5 GB preferred policy** and well under the 26.0 GB ceiling. GPU1 keeps 1,291 MiB free; GPU0 keeps 3,510 MiB free after the display reserve. No WDDM-spill warning was recorded; shared-GPU usage stayed at idle during measurements.

### 4a.4 Layer placement — what sits on VRAM vs RAM (tuned tuple)

48 layers total, in the `qwen4exp` layout (VENDOR arch model — names below are
derived from llama.cpp GPU-split semantics for MoE graphs; treat interpretation as
**INFERRED from measured VRAM deltas**, not a raw dump):

| component | where (tuned tuple) | why it lands there |
|---|---|---|
| All 48 layers' attention + GDN core (linear-attention state, QSA projections) | **VRAM** (both cards, proportional to `-ts`) | `-ngl all` places everything not overridden |
| The 12 QSA layers' KV cache | **VRAM** (q4_0/q4_0) | grows only with prompt depth |
| MoE experts of the **first 24 MoE layers** (`ffn_{down,gate,up}_exps`, fused) | **RAM** (host, mmap-backed from file) | `-ncmoe 24` overrides exactly those |
| Routed experts of remaining MoE layers | **VRAM** | the split ratios directions of the sweep: more to GPU1 was the gain side |
| Shared-expert tensors (`ffn_down_shexp` / `ffn_gate_shexp` / `ffn_gate_inp_shexp`) and router (`ffn_gate_inp`) | reside beside their layer's placement — **GPU** in this build's split; detecting whether `--cpu-moe` moves them is unverified (VENDOR, flagged open) | dense-shaped, never routed |
| Output head, embeddings, MTP head | **VRAM** | small, part of `-ngl all` |

The 1-echo measurable signature of this placement: GPU1 takes a **larger share than GPU0** (14,992 vs 8,772 MiB) because the Gen4-x4 card gets the larger half of the tensor split — the sweep showed pushing allocation back toward the x16 card (ts 10000) *loses* decode speed, consistent with the kernel-communication cost of the x4 card being worth paying for the bigger cheaper bandwidth.

### 4a.5 System RAM (MEASURED, per Task-Manager class)

| quantity | value | tag |
|---|---|---|
| total installed | 48 GB DDR5-7200 (KLEVV CRAS V RGB) — vendor plan value ~108 GB/s; **not re-measured this session, reference stays from historical evidence** | MEASURED (size), VENDOR (108 GB/s) |
| RAM used while serving the tuned tuple | **~47.8 GiB** (available down to ~300–1,300 MiB) | MEASURED |
| THROUGHPUT worth knowing: process working set | ~29–32 GiB | MEASURED |
| process private bytes | ~28.9–29.5 GiB | MEASURED |
| page file | ~28–29 GiB committed | MEASURED |
| interpretation of the numbers | these memory numbers are (a) **OS view**, and (b) a 48-GB machine that is *at the memory ceiling* while the model is resident — Task Manager's value is NOT the model residency; part of it is the OS standby/file cache holding the mmap'd shard-2 pages | flagged in the plan §2 discipline |

**Stop-condition tally from this batch:** no CUDA OOM on the tuned tuple; one expected boot-failure on `ncmoe 22` (data point, retained); no driver reset; no server crash; no pagefile thrashing beyond commit; no corrupted output (the tuner checked the sum answer each boot).

### 4a.6 Request-level record (tuned tuple, 6 requests)

| # | prompt | prompt tok | output tok | prompt TPS | decode TPS | wall s | correctness tag |
|---|---|---:|---:|---:|---:|---:|---|
| e24-confirm 1 | decode 512 | 74 | 69 | 9.2 | 8.72 | 16.13 | answer=correct-sum |
| e24-confirm 2 | decode 512 | 74 | 69 | 9.1 | 11.71 | 6.38 | answer=correct-sum |
| e24-confirm 3 | decode 512 | 74 | 69 | 9.1 | 11.83 | 6.14 | answer=correct-sum |
| A-winner pre4k 1 | 4K review probe | 3,077 | 64 | **20.5** | 10.55 | 156.12 | — |
| A-winner pre4k 2–3 (warm) | same | 3,077 | 64 | 7.22 / 9.25 | 10.83 / 11.06 | 6.41 / 6.16 | KV-cached |
| A-winner pre16k 1 | 16K review probe | 12,090 | 64 | **22.92** | 10.67 | 531.69 | — |
| A-winner pre16k 2–3 (warm) | same | 12,090 | 64 | 9.4 / 10.84 | 10.88 / 11.06 | 6.27 / 6.1 | KV-cached |
| A-winner pre32k | 32K probe | 24,114 | — | — | — | — | **exceed_context_size_error** (24114 > 16384) |


## 5. Data constraints and what is NOT yet answered

- **32K depth at ctx 16K is impossible** — the tuner sizes it 24114 tokens, above the 16384 allocation. To test 32K+ you must increase `-c` (and per the plan §25 test the ladder: 16K→32K→64K→128K→192K→224K→~256K, remembering that the KV weights exist only in the 12 QSA layers).
- **Quality is UNMEASURED beyond one smoke answer** (`sum(range(1,11))`, correct). Plan §27 coding-agent and Thai/language gates are still open.
- **0F — mmap/lazy residency verification** is still NOT done: `{auto, on}` for `--tensor-read-lazy` and the shard-2 residency behavior are unmeasured. Do not try `off` until explicitly verified — it may need ~26.8 GiB to become resident.
- **`-ncffn` classification for qwen4exp is unknown** — whether the 36 GDN layers are counted as dense FFN for `-ncffn` vs `-ncmoe` is unverified; do not assume.

## 6. Reproduce

One-shot boot from a terminal (after `Get-GpuVram.ps1` dot-source mode):

```powershell
cd C:\AI\qwen38-tuning\scripts
pwsh -NoProfile -ExecutionPolicy Bypass -File .\serve-flash-next.ps1 -WhatIf   # resolve argv only
pwsh -NoProfile -ExecutionPolicy Bypass -File .\serve-flash-next.ps1            # serve
```

Through the hub: `serve-hub.bat`, then key `I`.

Boot sweep harness:

```powershell
cd C:\AI\qwen38-tuning\results\flash-next-optimization
pwsh -NoProfile -ExecutionPolicy Bypass -File .\fn-config-run.ps1 -RunId test1 -Ts 8500,16000 -NcMoE 24 -Probes decode -DecodeRepeats 2
```

Result CSV: `qwen38-tuning/results/flash-next-optimization/results.csv` (fields run_id, decode_tps, prompt_tps, gpu0/g1 used, success/failure per request).
Boot logs: `.../logs/{run_id}.log` and `{run_id}-err.log` (llama-server stdout/stderr).

## 7. Vendors and provenance

- **Binary** — see `run-manifests/<run>.json`, one manifest per boot: argv, `CUDA_VISIBLE_DEVICES`, ctx, ncmoe, ts, t, tb, b, ub, sm, fit, lm, lazy, kv, model, exe sha256.
- **Results CSV** — `results.csv`, one row per request, fields: `run_id,timestamp,artifact_revision,runtime_commit,ctx_allocated,kv_k,kv_v,batch,ubatch,split_mode,tensor_split,placement,n_cpu_moe,threads,threads_batch,load_mode,tensor_read_lazy,prompt_tokens,output_tokens,prompt_tps,decode_tps,ttft_proxy_s,request_wall_s,gpu0_used_mb,gpu0_free_mb,gpu1_used_mb,gpu1_free_mb,total_dedicated_mb,gpu0_util,gpu1_util,ram_used_mb,ram_avail_mb,proc_ws_mb,proc_priv_mb,pagefile_mb,success,failure_reason,notes`.

## 8. Fit into the repo

- **results 31** (`docs/results/31-phase0-flash-next-2026-09-22.md`) — Phase 0 gate (0A→0E results; 0F NOT DONE).
- **plan §31** (`docs/plans/2026-09-23-Flash-Next-Performance-Improvement-Plan.md`) — the immediate next batch (experiments A–D); B and C from that list are done and recorded here; experiment D (current upstream binary) is not built in this session.
- **handoff report** (`docs/reports/40-flash-next-handoff-2026-09-23.md`) — the Phase 0 brief that pointed an optimization study here.

**Status:** Phase 0 untuned floor → **tuned floor moved to ~10.7 tok/s decode / ~23 tok/s cold prefill** at 16K with the tuple above.
