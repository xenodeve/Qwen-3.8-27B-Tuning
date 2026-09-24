# Qwen3.8-Flash-Next GSQ-RCO Q2_0 — Performance Improvement Plan

**Date:** 2026-09-23  
**Status:** Phase 0 smoke test passed; performance optimization has not yet been completed.  
**Primary target:** Qwen3.8-Flash-Next GSQ-RCO Q2_0 on the local Windows dual-GPU system.  
**Primary use case:** local Coding Agent / OpenAI-compatible serving.

---

# 0. Purpose

This plan starts **after** Phase 0 qualification.

The model is already known to:

- load successfully;
- expose an OpenAI-compatible endpoint;
- generate coherent output;
- run through the intended `qwen4exp` runtime;
- run on RTX 4070 SUPER + RTX 5060 Ti;
- use the current dual-architecture CUDA binary;
- work with PLE/n-gram lazy reads;
- run at 16K context.

The current observed performance is only a **working floor**, not a tuned result.

The study must answer:

1. Why is decode only ~7–8 tok/s?
2. Why is short-prompt prefill only ~6.5 tok/s?
3. Which part dominates: Q2 CPU kernels, CPU MoE, PLE/page faults, mmap, multi-GPU placement, PCIe x4, threads, batch/ubatch, or the runtime implementation?
4. Can speed be improved substantially without violating memory policy?
5. What are the final max-speed, daily and max-context profiles?

**Do not optimize 256K first.**

> First make the 16K baseline fast and explain where the time goes. Then perform the context ladder.

---

# 1. Hardware Ground Truth

## CPU

```text
Intel Core i5-13500
6 P-Cores + 8 E-Cores
14 cores / 20 threads
```

## RAM

```text
48 GB DDR5
KLEVV CRAS V RGB
7200 MT/s
```

Historical machine measurement/plan value:

```text
~108 GB/s
```

Do not silently treat that value as freshly measured during this study; reference the earlier evidence or re-measure it.

## GPU 0

```text
RTX 4070 SUPER 12 GB
Compute capability 8.9
Display GPU
PCIe Gen4 x16
```

Phase 0 llama.cpp observation:

```text
12,281 MiB total
11,069 MiB free at the baseline measurement
```

## GPU 1

```text
RTX 5060 Ti 16 GB
Compute capability 12.0
PCIe Gen4 x4
```

Observed:

```text
16,283 MiB total
15,172 MiB free
```

## Interconnect

```text
PXB
no NVLink
no peer link
```

Do not assume that a placement which balances VRAM also minimizes communication cost.

---

# 2. Memory Policy

## Dedicated VRAM

Preferred:

```text
<= 25.5 GB total dedicated VRAM
```

Absolute ceiling:

```text
<= 26.0 GB
```

Do not intentionally optimize toward 27–28 GB.

## Shared GPU Memory

If shared-GPU usage rises materially above idle, inspect for WDDM spill.

Initial warning threshold:

```text
> ~512 MiB increase from idle/baseline
```

## System RAM

Record separately:

```text
system RAM used
system RAM available
process working set
process private bytes
commit
pagefile
standby/file cache if available
hard/page faults
```

Do **not** interpret Task Manager "RAM used" as direct model residency.

Phase 0 already showed:

```text
tensor / ncmoe 34 -> RAM ~38.1 GiB
tensor / ncmoe 26 -> RAM ~43.0 GiB
```

although fewer MoE layers should reside on CPU in the second case. OS cache, mmap pages, pinned memory and staging can therefore be material.

---

# 3. Artifact Ground Truth

Repository:

```text
ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
```

Quant:

```text
Q2_0
```

Architecture:

```text
qwen4exp
```

Fetched revision:

```text
2c4721899b4382bd07dfb61ae4fbad90c09caf7d
```

Plan-named directory commit:

```text
8f752f8
```

These remain recorded as different provenance points. Do not silently reconcile them.

## Shard 1

```text
37,623,740,192 bytes
35.04 GiB
resident transformer/core weights
SHA256:
69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720
```

## Shard 2

```text
28,800,138,432 bytes
26.82 GiB
PLE / n-gram lookup table
SHA256:
316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113
```

Both hashes were verified against Hugging Face LFS metadata during Phase 0.

---

# 4. Architecture Facts Relevant to Optimization

Use this architecture model:

```text
125B MoE language model
~6B active
+ ~51B PLE / n-gram embedding
+ MTP head
```

Layers:

```text
48 total
36 Gated DeltaNet
12 Qwen Sparse Attention
```

MoE:

```text
512 routed experts
top-10 selected per token
```

PLE:

```text
per_layer_token_embd.weight
shape [160, 320001536]
lookup-only
not a dense matmul
```

KV:

```text
grows only for the 12 QSA layers
36 GDN layers use recurrent/compressed state instead of normal growing KV
```

Therefore at 16K, weight placement, CPU expert execution, PLE access and PCIe behavior are likely more important than KV.

Do not use 48-layer dense-transformer KV formulas.

---

# 5. Current Runtime Ground Truth

Binary:

```text
C:\AI\llama.cpp-unsloth-mirror\build-mirror\bin\llama-server.exe
```

Version:

```text
0.3.0-dev
build 215
exe commit 9f55aee
```

Manifest/source provenance:

```text
b8472555738ca1d8782e56717ffa4095031ad4a1
release b10679-mix-67dfc8b
CUDA 13.3
built 2026-08-29
```

CUDA code objects:

```text
40 × sm_120a
26 × sm_89
```

Only tool present in that binary directory:

```text
llama-server.exe
```

Every binary change requires a new record of:

```text
--version
--help
binary SHA256
ggml-cuda.dll SHA256
source commit / patch set
CUDA toolkit
sm_89 code-object guard
sm_120a code-object guard
```

---

# 6. Current CLI Contract

On the current binary:

```text
--tensor-read-lazy {on,auto,off}
```

exists.

Do not pass:

```text
--lazy-mode
-lzm
```

to this binary.

Relevant flags:

```text
-ncmoe / --n-cpu-moe
-ncffn / --n-cpu-ffn
-cmoe / --cpu-moe
-sm {none,layer,row,tensor}
-ts
-mg
-fit
-fitt
-fitc
-lm / --load-mode {auto,none,mmap,dio}
-fa
-c
-np
-t
-tb
-b
-ub
-ngl
-ctk
-ctv
```

---

# 7. Known Hard Constraints

## Row split

Known failure:

```text
-sm row
```

Observed error:

```text
device CUDA0 does not support split buffers
```

Do not spend optimization time on row split unless a future runtime explicitly changes this capability.

## Fit

Under tensor split, `--fit` has already behaved incorrectly/ineffectively for this workflow and can permit silent bad placement/spill.

Use:

```text
-fit off
```

for controlled experiments.

## Tensor split

Must be explicitly controlled with:

```text
-ts
```

Do not trust automatic placement.

---

# 8. Frozen Baseline

Current first working configuration:

```powershell
llama-server.exe `
  -m "<Q2_0 shard 1>" `
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

Observed:

```text
load     ~36 s
decode   6.66 tok/s
prefill  6.50 tok/s on a 73-token smoke prompt
VRAM     17,062 MiB total
RAM      ~38.1 GiB used
```

This is a smoke-test baseline, not a performance benchmark.

---

# 9. Existing Placement Evidence

At 16K, Q4 KV, fit off:

| Split | CPU MoE | VRAM total | Result |
|---|---:|---:|---|
| layer | 40 | ~11.0 GiB | loads |
| layer | 34 | ~14.96 GiB | loads |
| layer | 32 | ~16.3 GiB | loads |
| layer | 30 | ~17.6 GiB | loads |
| layer | 26 | — | CUDA OOM |
| tensor | 34 | ~16.66 GiB | 6.66 tok/s |
| tensor | 26 | ~21.91 GiB | 7.99 tok/s |
| tensor | 20 | — | OOM |
| tensor | 14 | — | assertion failure |

Important findings:

### First-N layer split is misleading

With first-N `-ncmoe`, layer split leaves the 4070 SUPER mostly unused.

Therefore:

```text
-sm layer + first-N -ncmoe
```

is not evidence against every possible layer-based placement.

### Tensor split is imbalanced

At:

```text
-sm tensor
-ncmoe 26
-ts 11069,15172
```

observed free VRAM was approximately:

```text
4070S    ~484 MiB free
5060Ti   ~5125 MiB free
```

There is combined free VRAM, but GPU0 runs out first.

---

# 10. Experimental Rule

Never combine several community optimizations in one first test.

Required progression:

```text
baseline
  ↓
change exactly one major mechanism
  ↓
paired benchmark
  ↓
record result
  ↓
retain or reject
```

A binary containing new qwen4exp + Q2 kernel + direct PLE + expert cache + placement rewrite cannot be used as the first comparison because attribution would be lost.

---

# 11. Measurement Discipline

Before optimization, derive a new noise floor for **this model and runtime**.

Do not inherit the 27B noise floor.

For every important A/B:

- same artifact;
- same prompt;
- same context allocation;
- same output cap;
- same sampler;
- same split policy unless that is the variable;
- same background-process policy;
- fresh server when required;
- alternate order when practical;
- at least 3 valid measurements for promoted candidates.

Report:

```text
mean
median
min
max
stddev
```

Decision guideline:

```text
<= 3%      effectively tied unless variance is extremely low
3–5%       weak / investigate
5–8%       meaningful candidate
> 8%       clear practical change if quality/stability are unchanged
```

---

# 12. Phase 1 — Replace the 73-Token Prefill Probe

The current 73-token prompt is too short for sustained-prefill conclusions.

Create fixed prompts of approximately:

```text
4K
16K
32K
```

tokens.

For each, in the **same server process**:

```text
request 1 = cold-ish
request 2 = warm
request 3 = warm
```

Record:

```text
prompt tokens
prompt eval time
prompt tok/s
decode tok/s
TTFT
process working set
private bytes
system commit
available RAM
hard/page faults per second
disk read bytes/sec
disk IOPS if available
disk writes
GPU VRAM
GPU utilization
CPU utilization
```

Interpretation:

- If cold prefill is far slower than warm prefill, mmap/page-cache behavior is a major suspect.
- If all three remain near ~6–8 tok/s, first-touch PLE faults alone do not explain the result.

---

# 13. Phase 2 — PLE Lazy Read Characterization

Compare:

```text
--tensor-read-lazy auto
```

against:

```text
--tensor-read-lazy on
```

The PLE tensor is >4 GiB, so they are expected to behave similarly on this runtime. Measure rather than assume.

Collect:

- process working set;
- private bytes;
- hard faults;
- SSD reads;
- warm-vs-cold prefill.

Goal:

- prove whether shard 2 remains nonresident;
- explain whether the PLE path materially owns prefill latency.

Do **not** try `off` until memory headroom is explicitly verified. Disabling lazy PLE may require roughly the full ~26.8 GiB table to become resident.

---

# 14. Phase 3 — Runtime Freshness A/B

Community reports indicate qwen4exp has changed rapidly after the current binary lineage. Treat this as a hypothesis to verify locally.

Create a **separate binary tree**. Do not overwrite the Phase 0 binary.

Suggested layout:

```text
build-flash-phase0/
build-flash-upstream/
build-flash-q2/
build-flash-direct/
build-flash-cache/
```

For current upstream:

1. pin exact commit;
2. record source commit;
3. build for both `sm_89` and `sm_120a`;
4. verify code objects;
5. capture `--help`;
6. preserve exact build flags;
7. run the frozen baseline unchanged where semantics permit.

First comparison:

```text
Phase0 binary
vs
current upstream binary
```

No extra patches.

Use the same:

```text
16K
ncmoe
tensor split
Q4 KV
threads
batch/ubatch
prompt
```

If CLI names changed, map semantics explicitly and record the mapping.

---

# 15. Phase 4 — Q2 CPU Kernel / VNNI Investigation

This is a **community-derived high-priority hypothesis** because many MoE expert weights execute on the i5-13500.

Agent tasks:

1. identify the exact Q2 x86/VNNI optimization branch/PR/implementation;
2. determine whether it is merged, open, or fork-only;
3. inspect code rather than relying on a discussion claim;
4. build it separately;
5. avoid unrelated bundled changes where possible.

Benchmark:

```text
ncmoe 48
ncmoe 40
ncmoe 34
ncmoe 26
```

at 16K.

Derive:

```text
decode vs CPU-MoE count
```

before and after the Q2 CPU kernel.

Question to answer:

> Is the i5-13500/Q2 CPU execution path the dominant ~7 tok/s decode bottleneck?

---

# 16. Phase 5 — Direct PLE Read

Community work specifically targets the PLE access pattern:

```text
row ~90 bytes
OS mmap page fault ~4 KiB
```

On Windows, sparse random lookup through mmap may create excessive page-fault cost.

Investigate current direct-read PLE work separately.

Requirements:

1. pin exact PR/fork commit;
2. confirm Windows support;
3. confirm current qwen4exp support;
4. inspect what code path changes;
5. build separately.

Compare:

```text
mmap lazy PLE
vs
direct-read PLE
```

using:

```text
cold 4K prefill
warm 4K prefill
cold 16K prefill
warm 16K prefill
decode
disk reads
page faults
```

Primary expected effect is prefill/TTFT. Do not assume decode must improve equally.

---

# 17. Phase 6 — Tensor-Split Sweep

Before redesigning placement, fix the obvious tensor imbalance.

Current:

```text
-ts 11069,15172
```

was derived from initial free VRAM, not optimized.

At `ncmoe 26`:

```text
4070S nearly full
5060Ti has ~5 GiB free
```

Sweep toward more allocation on the 5060 Ti after confirming the exact `-ts` semantics on the active runtime.

Bounded candidates may include:

```text
9500,16000
9000,16000
8500,16000
10000,15500
10000,16000
```

Do not brute-force blindly.

For each record:

```text
load success
VRAM per GPU
decode
prefill
TTFT
```

Goal:

- avoid GPU0 being the first limiter;
- bring per-device free VRAM closer together;
- determine whether `ncmoe 24` or lower becomes possible;
- stay under the 25.5/26 GB policy.

---

# 18. Phase 7 — Replace First-N CPU MoE with Manual Expert Placement

This is a major community-derived idea and may be especially relevant because:

```text
4070S = Gen4 x16
5060Ti = Gen4 x4
```

and first-N `-ncmoe` interacts badly with layer placement.

Do not copy community regexes blindly.

First enumerate the actual GSQ tensor names from the allocation file.

Known families include:

```text
ffn_down_exps
ffn_gate_exps
ffn_gate_inp
ffn_gate_inp_shexp
ffn_gate_shexp
ffn_down_shexp
```

and possibly fused variants.

The allocation file is the source of truth.

## Goal

Instead of:

```text
CPU = first N MoE layers
GPU = remaining layers
```

test deliberate placement such as:

```text
CPU expert band A
GPU region A
CPU expert band B
GPU region B
```

or another generated mapping that:

- uses both GPUs;
- avoids stranding the 4070S;
- reduces unnecessary tensor-split communication;
- keeps CPU expert traffic predictable.

Generate tensor override rules from the actual allocation map rather than hand-maintaining fragile regexes.

---

# 19. Phase 8 — Layer Split with Manual Expert Bands

After manual expert placement works, test:

```text
-sm layer
```

again.

This is a **different experiment** from the already-poor first-N layer split.

The original layer result is not evidence against all layer-based placement because first-N CPU offload left one GPU largely idle.

Question:

> Does layer split + intentional expert bands beat tensor split on the x16/x4 topology?

Measure:

```text
decode
prefill
TTFT
per-GPU utilization
per-GPU VRAM
PCIe traffic if available
```

If manual layer split is reproducibly slower by >10%, prune it.

---

# 20. Phase 9 — Shard 1 mmap / Load Strategy

The current runtime warns about CPU tensor overrides combined with mmap, while PLE lazy reading uses the file-backed path.

Investigate whether a newer runtime can implement:

```text
Shard 1:
normal resident/non-mmap model loading

Shard 2:
lazy/direct sparse PLE lookup
```

Desired topology:

```text
Shard 1 -> normal model residency
Shard 2 -> lazy sparse lookup
```

Compare current full-mmap behavior against a selective/non-mmap core strategy only if the runtime proves PLE lazy behavior remains available.

Record shard 1 disk reads and working-set behavior as well as shard 2.

---

# 21. Phase 10 — CPU Thread Topology

Current:

```text
-t 14
-tb 20
```

was inherited from 27B and is not tuned for Flash-Next.

After major kernel/placement questions are resolved, test decode thread counts:

```text
6
8
10
12
14
16
20
```

and batch-thread candidates:

```text
14
20
```

Observe:

- P-core/E-core scheduling;
- memory-bandwidth saturation;
- CPU utilization;
- decode consistency.

Do not assume all logical threads are optimal.

---

# 22. Phase 11 — Batch / Ubatch

Current:

```text
-b 2048
-ub 512
```

Test:

```text
ub 256
ub 512
ub 768
ub 1024
```

where supported.

Measure separately:

```text
prefill
TTFT
VRAM
decode
```

Do not transfer the old 27B ubatch result to this artifact.

---

# 23. Phase 12 — Expert Cache

Do not implement expert cache before:

- runtime behavior is understood;
- Q2 CPU kernel is measured;
- PLE path is measured;
- placement is stable.

Concept:

```text
RAM:
full/cold expert pool

VRAM:
hot experts
LRU / frequency-aware cache
```

Potential use case: repository/coding workloads may repeatedly route to a subset of experts.

Requirements:

1. pin implementation;
2. understand cache granularity;
3. measure cache hit rate;
4. measure PCIe traffic;
5. measure VRAM overhead;
6. ensure cache does not create WDDM spill.

Record:

```text
hit rate
miss rate
expert transfer bytes
decode
TTFT
VRAM
```

Do not promote a cache that wins only on a warm synthetic repeat.

---

# 24. Phase 13 — Speculative Decoding

Speculation comes **after** base inference is optimized.

Order:

```text
none
↓
ngram-only / ngram-mod
↓
MTP if supported
↓
DFlash only if independently justified
```

For every speculative configuration record:

```text
drafted tokens
accepted tokens
acceptance rate
output tokens
verified task time
decode rate
```

Do not rank speculative modes by tok/s alone.

---

# 25. Phase 14 — Context Ladder

Only start after a stable optimized 16K profile exists.

Test:

```text
16K
32K
64K
128K
192K
224K
~256K
```

At each context record:

```text
allocated context
actual prompt depth
KV type
VRAM
RAM
commit
prefill
decode
TTFT
page faults
disk reads
```

Remember: only 12 QSA layers grow normal KV.

Do not assume 256K is the memory limit. The system may hit a performance wall first.

---

# 26. Phase 15 — Native Linux A/B

Optional but high value if native Linux becomes available.

Do not treat WSL2 as equivalent to native Linux.

Compare Windows native vs Linux native with:

- same model;
- same quant;
- same runtime commit;
- same CUDA architecture targets;
- same placement;
- same prompts.

Purpose: isolate WDDM, Windows mmap/page-fault behavior, scheduler and PCIe/runtime effects.

---

# 27. Quality Gate

No performance profile becomes recommended until quality is tested.

Current Flash-Next quality evidence is only a correct Fibonacci smoke response.

After performance stabilizes, run at minimum:

1. small coding fixture;
2. multi-file coding task;
3. PAL-style frozen task;
4. tool-use workflow;
5. long-context retrieval;
6. Thai/language sanity;
7. reasoning sanity.

At minimum compare:

```text
Flash-Next GSQ-RCO Q2_0
vs
Qwen3.8-27B GSQ IQ3_S-MTP
```

Use verified-task outcomes, not prose impressions alone.

---

# 28. Build Lineage Policy

Never overwrite a working binary.

Suggested tree:

```text
builds/
  flash-phase0/
  flash-upstream-<commit>/
  flash-q2-vnni-<commit>/
  flash-direct-ple-<commit>/
  flash-expert-cache-<commit>/
```

For each save:

```text
llama-server.exe SHA256
ggml-cuda.dll SHA256
--version
--help
source commit
patch list
build flags
CUDA toolkit
sm_89 guard
sm_120a guard
date
```

If executable and source provenance disagree, record both and do not reconcile by assumption.

---

# 29. Result Schema

Create:

```text
flash-next-optimization-results.csv
```

Minimum columns:

```text
run_id
timestamp
artifact_revision
shard1_sha256
shard2_sha256
runtime_name
runtime_commit
patch_set
binary_sha256
os
ctx_allocated
ctx_actual
kv_k
kv_v
batch
ubatch
split_mode
tensor_split
placement_profile
n_cpu_moe
n_cpu_ffn
threads
threads_batch
load_mode
tensor_read_lazy
prompt_tokens
output_tokens
prompt_tps
decode_tps
ttft_ms
request_wall_s
gpu0_vram_used_mb
gpu0_vram_free_mb
gpu1_vram_used_mb
gpu1_vram_free_mb
total_dedicated_vram_mb
shared_gpu_mb
ram_used_mb
ram_available_mb
process_working_set_mb
process_private_mb
commit_mb
pagefile_mb
hard_faults_s
disk_read_mb_s
disk_write_mb_s
gpu0_util
gpu1_util
cpu_util
spec_mode
drafted_tokens
accepted_tokens
success
failure_reason
notes
```

---

# 30. Recommended Experiment Order

```text
0. freeze Phase 0 baseline
   ↓
1. derive current noise floor
   ↓
2. proper 4K/16K/32K cold-vs-warm prefill
   ↓
3. characterize PLE lazy path
   ↓
4. current upstream runtime A/B
   ↓
5. Q2 CPU/VNNI kernel A/B
   ↓
6. direct PLE read A/B
   ↓
7. tensor-split balance sweep
   ↓
8. manual expert-placement design
   ↓
9. layer split + expert bands A/B
   ↓
10. selective core-load / PLE-lazy strategy
   ↓
11. threads
   ↓
12. batch / ubatch
   ↓
13. expert cache
   ↓
14. speculative decoding
   ↓
15. context ladder
   ↓
16. native Linux A/B if available
   ↓
17. quality / Coding Agent gate
```

Do not skip directly from baseline to a heavily patched expert-cache build.

---

# 31. Immediate Next Batch

The next bounded batch should contain only these experiments.

## Experiment A — Sustained Prefill

Current binary:

```text
4K prompt
16K prompt
32K prompt
3 repeats in the same process
```

Collect page faults, disk reads and working set.

## Experiment B — Tensor Split

Current binary:

```text
ncmoe 26
ctx 16K
```

Move more allocation toward the 5060 Ti until VRAM headroom is better balanced.

## Experiment C — CPU-MoE Slope

Current binary:

```text
ncmoe 48
40
34
26
```

Same frozen prompt.

This establishes the existing-kernel slope.

## Experiment D — Current Upstream

Same A/B/C configurations on current upstream with no extra patches.

Only after these four should the study decide whether to prioritize:

```text
Q2 VNNI
or
direct PLE
or
placement redesign
```

---

# 32. Hypotheses to Test, Not Assume

## H1 — Q2 CPU kernel dominates decode

Supported if CPU-heavy `ncmoe` rows improve dramatically with the newer/VNNI kernel while GPU-heavy rows improve much less.

## H2 — PLE mmap dominates prefill

Supported if warm prefill greatly exceeds cold prefill, page faults/SSD reads are high, and direct PLE strongly improves prefill/TTFT.

## H3 — Tensor split is harmed by the x4 GPU path

Supported if manual layer/expert-band placement beats tensor split while reducing PCIe traffic.

## H4 — First-N MoE offload is the placement problem

Supported if expert-band placement uses both GPUs better, permits lower CPU-expert count, and improves decode without increasing total VRAM beyond policy.

## H5 — The current runtime is simply too old

Supported if current upstream with an otherwise identical configuration produces a large reproducible gain.

Multiple hypotheses may be true simultaneously.

---

# 33. Stop Conditions

Stop immediately for:

```text
CUDA OOM
driver reset
server crash
assertion failure
dedicated VRAM >26.0 GB
sustained shared-GPU spill
commit >90–92%
sustained pagefile thrashing
disk write storm
corrupted output
```

Do not let a spilled/OOM-adjacent run finish merely to collect a tok/s number.

---

# 34. Promotion Criteria

A configuration may replace the current baseline only if:

1. load is stable;
2. output is coherent;
3. no memory-policy violation occurs;
4. gain clears measured noise;
5. no severe latency regression appears;
6. no quality regression appears at the relevant gate;
7. exact binary/config provenance is recorded.

For a daily profile prefer:

```text
slightly slower + stable + headroom
```

over:

```text
fastest single run + almost no free VRAM + WDDM risk
```

---

# 35. Required Deliverables

Create:

```text
docs/results/Flash-Next-Optimization-Study.md

qwen38-tuning/results/flash-next-optimization/
  results.csv
  run-manifests/
  telemetry/
  logs/
```

Final launchers:

```text
run-flash-max-speed.ps1
run-flash-daily.ps1
run-flash-max-context.ps1
```

Final report must include for each profile:

```text
context
runtime commit
patches
placement
VRAM
RAM
prefill
decode
TTFT
```

Daily profile additionally includes:

```text
Coding Agent task time
quality gate result
headroom
```

Max-context profile additionally includes:

```text
allocated context
actual tested depth
long-context throughput
```

---

# 36. Final Questions

The study must answer:

1. Why was the Phase 0 system limited to ~6.66–7.99 tok/s?
2. How much is caused by the Q2 CPU kernel?
3. How much is caused by PLE lookup/page faults?
4. Does direct PLE read materially help Windows?
5. Does current upstream materially outperform the Phase 0 binary?
6. What tensor split best balances the 4070S and 5060 Ti?
7. Is tensor split actually appropriate with the 5060 Ti on Gen4 x4?
8. Can layer split become competitive with manual expert-band placement?
9. What CPU-MoE count is optimal after placement is fixed?
10. Is expert caching worth its complexity on coding workloads?
11. What context can be retained under the 25.5 GB VRAM policy?
12. What configuration is actually best for a real Coding Agent?
13. Is the final Flash-Next profile meaningfully better than the existing 27B GSQ setup?

---

# 37. Interpretation Rule

The current number:

```text
~7 tok/s
```

must be described as:

```text
Phase 0 untuned working floor
```

not:

```text
hardware ceiling
```

The model is proven runnable. The optimization study now needs to identify the software and placement path that turns it into a practical daily model.
