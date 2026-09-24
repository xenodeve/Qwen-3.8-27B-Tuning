# 32 — Flash-Next optimisation study: split, VNNI, expert cache, context, RAM, n-gram speculation

**Status: SHIPPED two profiles (hub keys I = 262,144, J = 131,072); MTP next.**
Dates 2026-09-23 → 2026-09-24. Artifact `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`
`Q2_0` @ `2c47218` ([results 31](31-phase0-flash-next-2026-09-22.md)). Cards: RTX 4070
SUPER 12 GB (GPU0, display, PCIe 4.0 x16) + RTX 5060 Ti 16 GB (GPU1, PCIe 4.0 x4),
UUID-pinned. 48 GB DDR5. i5-13500.

Every row is **MEASURED HERE** unless tagged **VENDOR** or **HYPOTHESIS**. Evidence lives
in `qwen38-tuning/results/flash-next-optimization/` (`results-v2.csv`, `spec-ngram.csv`,
`boot262k.csv`, `vram262k.csv`, `logs/`, `run-manifests/`). Speed quality is one to three
reps per cell, cross-boot unless stated; treat single-rep differences under ~15 % as noise
(`CORRECTIONS.md` §23: up to 48.9 % spread at ~64K depth).

---

## 1. Headline

| | before (report 41 profile) | shipped 128k (J) | shipped 262k (I) |
|---|---:|---:|---:|
| binary | unsloth mirror 9f55aee | pinned `llama.cpp-flash-vnni-cache` (1c6af3af) | same |
| split | `-sm tensor -ts 8500,16000` | `-sm layer -ts 31,17` | `-sm layer -ts 31,17` |
| `-ncmoe` / `-ub` | 24 / 512 | 26 / 1024 | 28 / 512 |
| decode, empty ctx | ~10.7 | 26.5 | 22.2 |
| prefill cold 12K prompt | 24 tok/s (507 s) | 323 tok/s | 221 tok/s |
| prefill / decode at 60K | — | 320 / 19.8 | 242 / 17.5 |
| prefill / decode at 120K | — | 341 / 14.5 | 262 / 15.3 |
| prefill / decode at 240K | — | — | 251 / 9.5 |
| peak VRAM GPU0 / GPU1 (MiB) | 8,772 / 14,992 | 11,934 / 14,722 | 11,350 / 14,929 |

Runs: `T-ctrl-ub512`, `D-C-128k-n26`, `D-D-256k-n28`. One cold fill per depth.

## 2. What moved the numbers

### 2.1 `-sm tensor` silently disables op offload — prefill ×15–23
The tensor split runs through the meta backend, whose `offload_op` is NULL
(`ggml/src/ggml-backend-meta.cpp:192`), so the scheduler (`ggml-backend.cpp:961`) can
never send a host-resident expert matmul to a GPU. Prefill was the i5 computing every
CPU expert: CPU 95–100 %, GPU util 0–10 %, no PCIe traffic
(`logs/diag-prefill-ub512-samples.csv`). Under `-sm layer`: PCIe 1.6–5.9 GB/s, same
binary, same round: 12K prefill **24.2 → 452–512 tok/s**, 3K **22.4 → 168–222**
(`T-ctrl-ub512` vs `L-ts30_18-*`, `V-M-*`). `-ub` did nothing under tensor (23.8 vs 23.3,
`ub512-r1` / `ub2048-r2`) because no ubatch ever reached the GPU.

### 2.2 x86 VNNI Q2_0 kernel (PR #26348) — decode ×1.8
A = upstream `e6ab7c1a`, B = A + VNNI. **A differs from B only in `ggml-cpu-*.dll`**
(`bin-A` = `build-b/bin` with CPU dlls built from `e6ab7c1a`). Order M A B B A M, same run
(`V-*`): decode **12.8 → 21.8** (empty ctx), **12.7 → 23.1** (after 3K/12K prompts);
prefill unchanged (A 577–589, B 554–595 at 12K). VENDOR claim was 9.63 → 19.59 at
`ncmoe 45`. Temperature-0 answers identical on short prompts (`logs/answer-A.txt`,
`answer-B.txt`).

### 2.3 GPU expert cache (PR #27861) — decode +28–36 %
`--moe-expert-cache N` on top of VNNI (`build-c`). B no-cache controls at both ends of the
run (`K-*`): 512-token cold decode **22.2 / 22.5 → 28.4** (64 slots), 31.4 (96); after a
12K prompt **20.9–23.1 → 31.3** (64). Cache 0 on the same binary = B (21.5), so the patch
is inert when off. 32 slots 20.9, 96 slots +4–10 % over 64 but leaves < 0.5 GB on the
display card. Inserts 4 vs 2: no difference. Prefill unaffected (decode-only by design).
Output coherent over 568 tokens at temp 0 (`logs/answer-C64.txt`). The cache graph is
built **only when `n_tokens == 1`** (`src/llama-graph.cpp:2166`) — see §2.7.

### 2.4 Context — the compute buffer is ~ubatch × ctx on EACH card
Boot-only matrix at 262,144 (`boot262k.csv`): prefill compute buffer **7.2 GiB at ub 2048,
3.7 at 1024, 1.85 at 512**, reserved on both cards. `-ncmoe` frees only GPU0 (the first
N layers' experts), so `-ts` must move with it. Measured VRAM (nvidia-smi 250 ms,
`vram262k.csv`) — boots at 262K: only `n26 ts31,17 ub512`, `n28 ts31,17`, `n28 ts32,16`.
At 131,072: `n24 ts30,18 ub1024`, `n26 ts31,17 ub1024`. Deep fills (`D-*`): A (64K,
n24, ub2048) collapsed to 6.9 tok/s decode at 60K — cause not isolated (HYPOTHESIS:
RAM pressure, §2.5); B (128K n24) was slower than C (128K n26) at every depth; D (262K
n28) beat E (262K n26) at 240K (251/9.5 vs 225/9.2).
Estimate that was WRONG and is corrected here: I predicted ub 512 would cost ~4× prefill;
measured 221 vs 323 tok/s at 12K (−32 %).

### 2.5 Host RAM: mmap keeps a second copy of the GPU weights — trim it
`-lm mmap` leaves every page read to upload GPU weights in the process working set
(`logs/diag-mem-128k*.csv`): after boot WS 20.6 GB of which 19.1 GB shared (file pages),
1.5 GB private. After a 60K fill 34.7 GB WS and **2.2 GB / 0.6 GB RAM available** — this is
what got a server reaped by Claude Code's low-memory guard. `EmptyWorkingSet` once after
`/health`: WS 20.6 → 0.8 GB; after a 60K fill it regrows to 18.5 GB (host experts + PLE),
not 33.5. ABBA ×3 (`trim*` / `notrim*`): **RAM free after 60K 14.5 vs 1.1 GB**; prefill 60K
237–272 vs 118–283; decode at 60K **13.1–13.9 vs 3.8–17.2** — two of three untrimmed runs
collapsed. PLE (28.8 GB, `--lazy-mode on`) is read row by row; it is not the RAM hog
(my earlier "PLE page cache" claim was unmeasured and is withdrawn).

### 2.6 Prefill speed depends on prompt length
Per-ubatch cost is streaming host experts over PCIe: 51 tok 55 tok/s, 74 tok 49,
3,077 tok 388, 12,090 tok 577 (layer-mode rows). Short incremental turns in an agent loop
read as low tok/s while costing ~seconds. Below 32 tokens the CUDA backend does not
offload at all (`GGML_OP_OFFLOAD_MIN_BATCH`, default 32, `ggml-cuda.cu:5720`). Not tuned.

### 2.7 n-gram speculation (`ngram-mod`), 128k config + trim, temp 1.0 and 0
`spec-ngram.csv`, 2 reps × 2 temps per cell. **copy** = rename a function in a 100-line
file and print the whole file; **new** = write a fresh module, 512 forced tokens.

| arm (match/min/max) | copy tok/s (median) | new tok/s (median) | accept copy / new |
|---|---|---|---|
| none (S0) | 22.0–27.1 (25.9) | 23.9–27.7 (26.2) | — |
| 12/4/12 | 29.6–37.8 (32.8) | 10.9–21.6 (16.1) | 0.98 / 0.39 |
| 12/8/24 | 39.7–47.5 (43.4) | 7.3–36.1 (15.5) | 0.94 / 0.23 |
| **12/16/32** | **40.9–55.9 (53.0)** | **21.4–29.4 (22.7)** | 0.94 / 0.09 |
| 24/48/64 (llama.cpp default) ⚠ | 15.9–22.7 (18.7) | 2.8–14.3 (8.0) | 0.92 / 0.06 |
| none, end control ⚠ | 10.5–24.5 | 10.2–26.8 | — |

⚠ From 02:51 GPU0 (display card) sat at 85 % util / 92 W during plain decode with desktop
apps open; the end control fell to 10.5 tok/s and recovered to 24–27 within the arm. The
default-arm row is **not a settled verdict**, though its failure mode is predicted by code:
verify batches of 49–65 tokens exceed the op-offload threshold of 32.
Findings: 12/16/32 roughly **doubles copy-heavy decode and costs ~13 % on fresh text**;
a high `n-min` is what limits the fresh-text loss (it drafts only on long matches).
Acceptance on copy is 92–99 % at temp 1.0 as well as 0 — copying is temperature-robust.
Every verify step runs without the expert cache (`n_tokens == 1` guard).
Caveat: `new` forces 512 tokens (`ignore_eos`); some high-acceptance `new` rows may be
the model repeating itself after finishing — content was not inspected.
**Not shipped yet** — the plan is MTP first, then MTP + n-gram together.

## 3. Instrument faults found and fixed (each produced a plausible number)

1. `build-a2` was compiled after the VNNI cherry-pick (`LLAMA_COMMIT 654e1828`) — "A" was B.
2. Prefill reps 2–3 were prompt-cache hits (4 tokens evaluated) logged as prefill speed.
   Harness now sends `cache_prompt:false` and refuses `prompt_n < 0.9 × prompt_tokens`.
3. `-Exe` collided with `$global:EXE` (PowerShell is case-insensitive): a run labelled B
   ran the mirror. Global renamed; rebinding now throws.
4. Git-bash `tail -f` holds `study.log` without write-sharing: `Add-Content` threw, the
   run died after boot and orphaned a server. `Write-Log` retries 15 s then fails loudly.
5. `build-b` / `bin-A` lacked the CUDA 13 runtime DLLs (`bin\x64`, not on PATH):
   `ggml-cuda.dll` failed to load silently and the server ran **CPU-only** at ~5 tok/s.
   Harness now refuses a boot whose log says `no usable GPU found`.
6. `powershell -File` drops an empty `-Extra ''` and passes `a,b` as one string.
7. Two llama-servers bound `0.0.0.0:8080` at once (Windows allows it) and WDDM let both
   over-commit VRAM: no error, 1.2 tok/s prefill. Not yet guarded in the launcher.
8. Upstream renamed `--tensor-read-lazy` to `--lazy-mode`; harness asks the binary.

## 3b. 2026-09-24 — speculation, VRAM placement, Claude Code fitness

All `spec-mtp.csv` rows are `spec-bench.py` (copy = rename-and-reprint 1,055-token
prompt; new = 512 forced tokens), eff tok/s, 1 rep per boot unless noted.

- **MTP + ngram-mod duo (128k).** Pinned `C:\AI\llama.cpp-flash-mtp` (7a0ac666,
  PROVENANCE.txt). MTP n=3 beat n=2/4. Prefill of a 1K prompt drops ~40–50 % with MTP.
- **262k + MTP over-committed GPU0 silently** (1,864 MiB free before the draft, 2,825
  needed): WDDM spilled, a 14.7K prefill ran at ~42 tok/s. nvidia-smi showed nothing;
  the per-process "GPU Process Memory … Shared Usage" counter did.
- **ngram-mod n-max 64 (q-ng64, ABBA C A B B A C, duo):** over 12/16/32, **24/16/64**
  copy +23 % (t0) / +13 % (t1), fresh +26 % / +6 % (fresh-t0 control noisy);
  12/16/64 won copy but lost fresh text (acceptance 0.42–0.54). Matches the 27B's
  +14.5–15.6 % for n-max 64 (02-decoders.md). Shipped 24/16/64 on both profiles.
- **The display card's free VRAM moves ~1 GB with desktop apps** (dwm, Discord,
  browser), same argv: 3,103 MiB free before the draft at 07:24 vs 2,035 at 07:49;
  decode read 18–20 tok/s at 15K. **Draft + output.weight moved to the 5060 Ti**
  (nothing else uses it), `-ncmoe 36` to make room there (q-ncmoe-hr, 2 boots each,
  desktop apps open): llama holds 7,669 MiB on CUDA0 instead of 10,609; decode equal or
  better (copy t0 59.9 vs 55.9, fresh t1 29.7 vs 20.7), prefill 1K 275–283 vs 207–240.
  `-ncmoe 34` OOMs CUDA1 at boot (1,674 free, needs 2,602). Note: under `-ts 31,17`
  raising `-ncmoe` past 31 frees CUDA1, not CUDA0.
- **262k with the draft on CUDA1 does not pay** (q-fix262): vs ngram-only at `-ncmoe 28`,
  n36 duo copy −7 / −9 %, fresh +7 / +9 % (under the floor), prefill 1K −18 %. 262k
  stays ngram-only (24/16/64); ngram-only there copies at 63.9 t0.
- **Claude Code needs** (all on both profiles, `test_flash_next_launcher.py`):
  `qwen38-late-system.jinja` (Flash-Next's stock template is byte-identical to
  `qwen38-stock.jinja`; without it every request is HTTP 500), `--sse-ping-interval 5`,
  `--reasoning-effort medium`, `--cache-ram 8192 --ctx-checkpoints 8`. At the default 32
  checkpoints one 105K entry held 5,918 MiB, so Claude Code's **auto-mode classifier**
  (30–35K-token requests to the same model, 47 of them = 40 % of server time in 83 min)
  evicted it and the next turn re-prefilled 66,890 tokens (274 s).
  cache-ram stays 8192, not 24576: host experts already hold 25–28 GB (UNMEASURED
  whether a larger cap would page experts).
- **Quality/time against GSQ:** [result 33](33-flash-next-vs-gsq-quality-time-2026-09-24.md).

## 4. Shipped

`qwen38-tuning/scripts/serve-flash-next.ps1 -Profile 262k|128k` (hub I / J, launchers
`serve-flash-next[-128k][-lan].bat`), binary `C:\AI\llama.cpp-flash-vnni-cache\bin`
(`PROVENANCE.txt`), `--moe-expert-cache 64`, `-lv 4 --log-colors on --log-file`,
one working-set trim after `/health`. Test gate 2079 passed / 2 skipped after the change.
Sampler is the GGUF default (temp 1.0, top_k 20, top_p 0.95, min_p 0.05). No speculation.

## 5. Open

- **MTP** (next): head `unsloth/.../MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`
  downloaded (1,907,151,936 B, matches HF); PR #28243 applied cleanly as branch
  `vnni-cache-mtp` (`998881bd`), not built. VENDOR: 1.34–1.67× greedy on a B200.
  VRAM to be freed by raising `-ncmoe`, never by shrinking the cache (developer, 2026-09-24).
- Expert cache during small verify batches (`n_tokens <= K`) — needed for any speculation.
- `ngram-map-k` / `k4v`; `ngram-mod` 12/16/30 (verify batch under 32); default arm re-run.
- `--sse-ping-interval 5`, `--reasoning-effort` (server runs at `xhigh`) — undecided.
- `-t 14 -tb 20` not re-tuned under layer + VNNI. `GGML_OP_OFFLOAD_MIN_BATCH` not tuned.
- Launcher guard against a second server on the same port.
- Quality — deliberately last.
