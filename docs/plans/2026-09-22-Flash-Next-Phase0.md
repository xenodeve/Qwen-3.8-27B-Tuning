# Qwen3.8-Flash-Next GSQ-RCO Q2_0 — Phase 0: Environment & Artifact Qualification (GATE)

> Companion **gate** to
> [`2026-09-22-Qwen3.8-Flash-Next-Q2_0-Optimization-Plan.md`](2026-09-22-Qwen3.8-Flash-Next-Q2_0-Optimization-Plan.md).
> Written 2026-09-22. Status: **NOT STARTED**.

## Why this file is separate

Phase 0 is no longer "the first benchmark step". It is **qualification** — of the
artifact, of the binary, of the CUDA objects, of the CLI contract, and of the
lazy/mmap residency behaviour. That is a different level of information from the
optimization plan: it answers *can this machine run this model, and did it actually
run the code we think it ran* — not *how fast*.

Keeping it separate means a binary swap two weeks from now re-runs Phase 0 without
touching the optimization plan. Conclusions only merge back into the main plan
**after** Phase 0 passes.

```
0A Artifact verification
        ↓ PASS
0B Runtime / binary verification
        ↓ PASS
0C GPU / CUDA architecture verification
        ↓ PASS
0D Split-mode / placement constraints
        ↓ PASS
0E 16K smoke test
        ↓ PASS
0F Lazy / mmap residency verification
        ↓
GO → Optimization Plan (Phase 1+)
```

**Fail anywhere → stop at that step.** Do not touch the context sweep, the
`n_cpu_moe` ladder, or any tuning. Record the failure and hand it back.

---

## Ground truth already established — MEASURED HERE, 2026-09-22

Everything in this block was read off this machine today. It is the starting
state, not a to-do.

| | value | source |
|---|---|---|
| CUDA0 | `RTX 4070 SUPER` — 12,281 MiB total, **11,069 MiB free**, cc **8.9** | `--list-devices`, `nvidia-smi` |
| CUDA1 | `RTX 5060 Ti` — 16,283 MiB total, **15,172 MiB free**, cc **12.0** | `--list-devices`, `nvidia-smi` |
| driver | **616.92** | `nvidia-smi` |
| binary | `C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe` | on disk |
| binary version | `0.3.0-dev (build 215, commit 9f55aee)`, MSVC 19.44.35228.0, *"Compiled by the Unsloth team"* | `--version` |
| `ggml-cuda.dll` code objects | **26 × `sm_89` + 40 × `sm_120a`** | string scan of the DLL |
| flags that exist | `--tensor-read-lazy`, `-lm/--load-mode`, `-ncmoe`, `-ncffn`, `-sm {none,layer,row,tensor}`, `-ts`, `-fit/-fitt` | `--help` |
| flags that do **not** exist on it | `--lazy-mode`, `-lzm` | `--help` |
| tools in `bin/` | **`llama-server.exe` only** — no `llama-cli`, `llama-bench`, `gguf-dump` | directory listing |
| helpers that exist | `qwen38-tuning/scripts/Get-GpuVram.ps1`, `qwen38-tuning/bench/gpu_device.py`, `bench/hardware_baseline.py` | on disk |

**Do not modify these trees** — they stay pinned for the Qwen3.8-27B / DFlash2 work:

```
C:\AI\llama.cpp              @ 1deefcca3   (no qwen4exp — expected)
C:\AI\llama.cpp-mirror       @ 1deefcca3   (no qwen4exp — expected)
```

The qwen4exp-capable tree is `C:\AI\llama.cpp-unsloth-mirror` and its build is
`build-mirror\`.

---

## What Phase 0 must NOT do

- No context ladder (128K / 192K / 224K / 256K).
- No `n_cpu_moe` sweep beyond what 0E/0F need to prove a load.
- No throughput comparison, no arm ranking, no decoder choice.
- No "fixing" a provenance mismatch so the numbers look tidy — record it as found.

---

## Phase 0A — Artifact verification

**Purpose:** prove the two shards are the files we think they are, and that the
GGUF metadata matches the architecture the plan is written against.

```powershell
$modelDir = "C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0"

Get-ChildItem $modelDir | Select-Object Name, Length
Get-FileHash "$modelDir\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" -Algorithm SHA256
Get-FileHash "$modelDir\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" -Algorithm SHA256
```

VENDOR claims to check against (ISTA model card — **not** our number until measured):

| | vendor |
|---|---|
| shard 1 | `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` — **37.6 GB** |
| shard 2 | `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` — **28.8 GB** |
| total | **66.4 GB** |
| directory commit | `8f752f8` |

Metadata (no `gguf-dump` in this build → read it out of the load log):

```powershell
$exe = "C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe"
$m   = "$modelDir\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
$log = "$env:TEMP\phase0-0A-metadata.log"

$p = Start-Process $exe -PassThru -RedirectStandardError $log -ArgumentList @(
  "-m", $m, "-c", "512", "-ngl", "0", "--no-warmup",
  "--host", "127.0.0.1", "--port", "8099", "-v"
)
Start-Sleep -Seconds 45
Stop-Process -Id $p.Id -Force

Select-String -Path $log -Pattern "arch|n_layer|n_expert|n_embd|n_ctx_train|per_layer_token_embd|qwen4exp|error|unknown"
```

**PASS** when all of these hold:

- exactly **2** shard files present; each size within ~1 % of the vendor figure;
- SHA256 computed and recorded for both;
- log names **`qwen4exp`** (not `unknown model architecture`);
- log shows **48 layers**, MoE expert count **512**, `n_ctx_train` **262144**;
- the PLE tensor **`per_layer_token_embd.weight`** appears in the tensor list.

**FAIL** if the log says `unknown model architecture: 'qwen4exp'` → 0B is wrong,
stop and go back to the binary.

---

## Phase 0B — Runtime / binary verification

**Purpose:** pin the binary's provenance and its CLI contract before any argv is
written. The repo's own history is the reason this is a step and not an assumption:
two builds of the same commit once differed 4× in decode and were
indistinguishable by `--version` (`results/09-hardware.md`).

```powershell
$exe = "C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe"

& $exe --version
Get-Content "C:\AI\llama.cpp-unsloth-mirror\BUILD_INFO.txt"
Get-Content "C:\AI\llama.cpp-unsloth-mirror\UNSLOTH_PREBUILT_INFO.json"

& $exe --help 2>&1 | Select-String `
  "tensor-read-lazy","load-mode","n-cpu-moe","n-cpu-ffn","split-mode","tensor-split","fit","main-gpu","warmup"

& $exe --list-devices
```

Record, **without reconciling**:

- exe-reported commit `9f55aee` **and** source-tar commit
  `b8472555738ca1d8782e56717ffa4095031ad4a1` — these differ, record both;
- release `b10679-mix-67dfc8b`, CUDA toolkit **13.3**, built `2026-08-29T02:57:15Z`,
  `build_shared_libs: ON`;
- the flag contract actually printed. On this binary the lazy flag is
  **`--tensor-read-lazy MODE`** (`on` / `auto` / `off`, default `auto`).
  **`--lazy-mode` and `-lzm` are not present.** Any argv must use the name this
  binary prints, byte-exact.

**PASS** when: `--version` and `--list-devices` both run; the flag contract is
recorded verbatim; CUDA0/CUDA1 match the ground-truth table above.

⚠️ Record any divergence between `BUILD_INFO.txt`, `UNSLOTH_PREBUILT_INFO.json`
and `--version`. Do not edit one to match another.

---

## Phase 0C — GPU / CUDA architecture verification

**Purpose:** the check the repo demands is on the **code objects**, not on a
manifest's `supported_sms` list. A `supported_sms` line is a claim; `sm_120a`
cubins in the DLL are evidence.

```powershell
$dll = "C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\ggml-cuda.dll"
$s = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($dll))
[regex]::Matches($s, 'sm_120a|sm_120|sm_89|sm_90|sm_103') |
  ForEach-Object Value | Group-Object | Select-Object Count, Name
```

Cross-check with `cuobjdump --list-elf` if the CUDA toolkit is on `PATH` (it was
not, at `v12.4`, on 2026-09-22 — locate the toolkit first; do not guess the path).

**PASS** when the DLL contains **≥ 1 `sm_89`** *and* **≥ 1 `sm_120a`**.
Measured here today: `26 × sm_89`, `40 × sm_120a` — both cards covered.

**Re-run this step every time the binary or `ggml-cuda.dll` changes.** A build with
Ada-only cubins loads fine, allocates identically, OOMs never, logs nothing — and
is 2.2× slower at prefill (`results/09-hardware.md`).

---

## Phase 0D — Split-mode / placement constraints

**Purpose:** lock the split decision before the smoke test, because split-mode
constrains how `n_cpu_moe` can be calibrated in Phase 2.

Known constraints from this machine (verify by reproduction, do not assume):

- **`-sm row` cannot load** on this pair — `error loading model: device CUDA0 does
  not support split buffers` (PXB topology, no NVLink). Reproduce once and record.
- **`--fit` is inert under `-sm tensor`** — `llama_params_fit is not implemented for
  SPLIT_MODE_TENSOR`. Worse, an over-committed tensor split **spills silently into
  host memory** and returns a working server at **0.38 tok/s**. Never rely on
  `--fit` to protect the VRAM budget under tensor.
- under `-sm tensor` the default split is **even**, capacity ignored → `-ts` is
  mandatory and must be computed from measured free VRAM
  (`qwen38-tuning/scripts/Get-GpuVram.ps1`, pinned by UUID — not by index).

**PASS** when: the chosen split-mode is recorded with its reason; if `tensor`, the
`-ts` values and the free-VRAM reading they came from are recorded; if `layer`, the
reason it was chosen over `tensor` is stated.

---

## Phase 0E — 16K smoke test

**Purpose:** prove the model loads and answers at a low-risk depth. 16K is
deliberately small — it is a *smoke* test, not a performance point.

Baseline argv (adjust only where 0B's `--help` disagrees):

```powershell
$exe = "C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe"
$m   = "$modelDir\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"

& $exe `
  -m $m `
  -c 16384 `
  -ctk q4_0 -ctv q4_0 `
  -lm mmap `
  --tensor-read-lazy on `
  -ngl all `
  -sm <from 0D> `
  -ts <from 0D, if tensor> `
  -np 1 `
  -fa on `
  --host 127.0.0.1 --port 8099
```

Then:

```powershell
curl.exe -s http://127.0.0.1:8099/health
curl.exe -s http://127.0.0.1:8099/v1/chat/completions -H "Content-Type: application/json" -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function that reverses a linked list, with a docstring.\"}],\"max_tokens\":256}'
```

**PASS** when: `/health` is ok; the completion is coherent (not repeated-prompt
filler); **≥ 128** tokens generated; no `cudaMalloc failed` / OOM; no driver reset;
**dedicated VRAM total ≤ 25.5 GB**; shared GPU memory within ~512 MB of baseline.

---

## Phase 0F — Lazy / mmap residency verification

**Purpose:** prove the PLE table is treated as a lazy lookup and not pulled
resident — and watch shard 1 too, because **every mmap'd shard is lazy in
principle**, not just shard 2.

Two runs of the *same* prompt in the *same* process:

- **cold** — fresh process, `--no-warmup`, sample from launch through first token;
- **warm** — repeat the identical prompt; the PLE path should be hot.

Sampler (run alongside):

```powershell
Get-Counter `
  "\Process(llama-server)\Working Set - Private",
  "\Process(llama-server)\Page Faults/sec",
  "\PhysicalDisk(_Total)\Disk Read Bytes/sec" `
  -SampleInterval 1 -MaxSamples 120

Get-CimInstance Win32_OperatingSystem |
  Select-Object TotalVisibleMemorySize, FreePhysicalMemory, TotalVirtualMemorySize, FreeVirtualMemory
```

Then compare `--tensor-read-lazy auto` against `on` on the same prompt (one round
each). Since the PLE is ~26.8 GiB — far above the 4 GiB `auto` threshold — the two
are expected to behave the same; confirm it rather than assume it.

**PASS** when:

- process working set does **not** jump by ~28.8 GB at load → PLE is not resident;
- disk reads occur during inference and are bounded (not a sustained thrash);
- warm run is not slower than cold on the PLE path;
- `auto` and `on` show equivalent behaviour (or the difference is explained);
- commit stays **< 90 %**; no sustained pagefile thrashing.

⚠️ On Windows a lazy row read (~90 bytes) can cost a 4 KiB page fault; the open PR
for direct PLE reads reports ~20–37 % cold-prefill gains on other systems.
**That PR is not in this binary.** Measure what is here; do not import its number.

---

## Exit criteria

Phase 0 is **GO** only when 0A–0F have each been recorded as PASS, with the raw
evidence (log path, hash, counter output) named for every one.

Any FAIL → STOP. The optimization plan is not touched. The failure is recorded in
the Phase 0 result file and handed back.

---

## Result record

Write one file, numbered in the results register, e.g.
`docs/results/31-phase0-flash-next-<YYYY-MM-DD>.md`, containing:

- 0A: filenames, sizes, SHA256, HF revision, metadata log path
- 0B: version string verbatim, BUILD_INFO / PREBUILT_INFO values, both commits, flag contract
- 0C: code-object counts, `cuobjdump` output if obtained
- 0D: split-mode chosen, `-ts` and the free-VRAM reading behind it
- 0E: argv verbatim, `/health`, sample completion, peak per-GPU VRAM, shared memory
- 0F: cold/warm working set, page-fault rate, disk-read rate, `auto` vs `on`, peak commit

Tag every figure **MEASURED HERE** with its file, or **VENDOR** if it is ISTA's /
Unsloth's / Qwen's claim. An untagged number is the one to distrust.

---

## After the gate — the order the main plan will use

Not executed in Phase 0; recorded here so §31 of the main plan can stay short.

```
Phase 1  16K placement calibration   → measure ΔVRAM / Δ(n_cpu_moe) slope
Phase 2  context memory calibration  → 16K → 64K → 128K → 192K → 224K → ~256K
Phase 3  derive max-context config   → from the measured slopes, not a guessed ladder
Phase 4  performance tuning          → threads, ubatch, split, KV type
Phase 5  long-context + stability + real coding-agent test
```

The `n_cpu_moe` range is **derived** from Phase 1's slope and Phase 2's context
slope. The old fixed ladder `28 → 23` was a guess and is retired.

---

## Open investigation item: `-ncffn` vs `-ncmoe`

`-ncffn, --n-cpu-ffn N` keeps **dense FFN** weights of the first N layers on CPU;
`-ncmoe` keeps **MoE expert** weights. They are different tensors.

**Do not sweep either until Phase 0 answers:** in the current `qwen4exp`
implementation, which tensors are classified as dense FFN, and how does that
overlap the GDN block? Until that is read out of the placement log, turning
`-ncffn` may move a different part of the model than intended.
