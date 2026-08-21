# Qwen3.8-27B Local Coding Worker — Optimization Plan

> **Goal:** maximize **Verified Successful Coding Tasks / Hour**, not raw tok/s.

## 1. Target Stack

```text
Claude Code / Opus 5
        │
        │ master: planning, architecture, integration,
        │ security, evidence judgment, final verification
        ▼
Xeno Agentic Engineering Framework
        │
        ▼
OpenClink
        │ CLI-to-CLI delegation
        ▼
OpenCode
        │ coding-agent harness
        ▼
llama-server / llama.cpp CUDA
        │
        ▼
Qwen3.8-27B GGUF
```

## 2. Current Machine Baseline

**MACHINE**

```text
OS      Windows 11
CPU     Intel Core i5-13500
GPU     RTX 4070 SUPER 12 GB
RAM     48 GB DDR5
Runtime C:\AI\llama.cpp-cuda
Build   llama.cpp b10472 / commit 60eeeb608
CUDA    CUDA0 recognized correctly
```

Measured Q4 baseline:

```text
Quant                  Unsloth UD-Q4_K_XL
Disk size              16.69 GiB
Context                16,384
Slots                   1
VRAM free before load  ~10,192 MiB in measured run
VRAM after load         ~505 MiB free
Host RAM free           ~11.35 / 47.69 GB
Prompt processing       518.8 tok/s @ 4,601 prompt tokens
Generation              6.29 / 6.81 / 7.56 tok/s across 3 runs
```

Important observed confounder:

```text
free VRAM before launch varied ~9.36–11.07 GiB
```

Therefore identical `--fit on` commands can produce different layer splits.

## 3. Optimization Objective

Priority order:

```text
1. verified task success / hour
2. tool-call correctness
3. task success rate
4. recovery/retry burden
5. long-context stability
6. wall-clock latency
7. generation tok/s
8. prompt processing
9. RAM/VRAM efficiency
```

## 4. Final Experiment Order

### Phase A — Runtime baseline
Status: **complete**

- CUDA works
- Q4 boots
- basic speed and memory measured

### Phase B — Protocol correctness
Do this **before** performance tuning.

Validate:

```text
plain completion
developer/system role
simple tool call
nested object arguments
tool result → continuation
repeated tool loop
reasoning separation
no unwanted <think> leakage in content
min_p corrected to 0.0
```

A faster configuration that breaks OpenAI-compatible tool round trips is rejected.

### Phase B2 — Reasoning policy

The local Qwen chat template was observed to default to:

```text
reasoning_effort = xhigh
```

and map:

```text
high → xhigh
```

Sweep:

```text
low
medium
xhigh
```

Use the same model, same prompt/task, same sampling, MTP off.

Hypothesis:

```text
low     → easy/cheap leaves
medium  → likely local-worker default candidate
xhigh   → hard-leaf escalation
```

Do not assume this; measure verified success and wall time.

### Phase B3 — Coding sampling

Vendor/general Qwen3.8 thinking baseline from the supplied Unsloth documentation:

```text
temperature       1.0
top_p             0.95
top_k             20
min_p             0.0
presence_penalty  0.0
```

**PROXY:** Qwen3.5 architecture-family docs also show a lower-temperature precise-coding profile. Therefore test:

```text
A: temperature 1.0
B: temperature 0.6
```

Keep all other sampling values identical.

This changes model behavior, so evaluate **verified success**, not only speculative acceptance.

### Phase C — Quant × Speculation matrix

This is now the major speed phase.

Run:

| Quant | none | ngram-simple | draft-mtp n=2 | draft-mtp n=3 |
|---|---:|---:|---:|---:|
| UD-Q4_K_XL | ✓ | ✓ | ✓ | ✓ |
| UD-Q3_K_XL | ✓ | ✓ | ✓ | ✓ |

Why:

- Q4 is the reliability/fidelity hypothesis.
- Q3 returns several GB of model footprint and may move significantly more layers to GPU.
- MTP can accelerate predictable decoding but consumes extra runtime memory.
- `ngram-simple` is almost memory-free and llama.cpp explicitly cites source-code rewriting as a use case.

Do **not** combine MTP + ngram in the first pass. Let them compete independently.

### Phase C2 — MTP depth sweep

Only for the best MTP candidate from Phase C:

```text
--spec-draft-n-max 2
3
4
5
6
```

Measure:

```text
accepted / drafted
acceptance rate
tok/s
VRAM
RAM
actual target layer split
tool-call correctness
output equivalence
```

Do not assume the video claim that 4–5 is optimal.

### Phase D — Optional third quant vendor

Only if the Q3/Q4 tradeoff leaves a real gap:

- Q3 too weak on verified tasks
- Q4 too slow due residency

Then test **one** smaller Q4-ish AtomicChat candidate.

Do not compare AtomicChat KL directly to Unsloth Top-1; different proxy metrics/evals.

### Phase E — VRAM residency / headroom

Inspect local b10472 support for:

```text
--fit-target
```

If present, test useful margins such as:

```text
512
1024
1536
2048 MiB
```

Production should not casually run with only ~505 MiB free if desktop apps can take VRAM afterward.

### Phase F — Context depth

Run both cold-prefill and generation-at-depth:

```text
32K
64K
128K
192K
256K
```

Use `llama-bench -d` where applicable.

Configured context maximum can remain 256K while normal active working context stays much lower.

### Phase G — Prefix-cache behavior

Test an OpenCode-like multi-turn sequence.

Measure:

```text
cold first turn
incremental second turn
incremental third turn
effect of system/tool-schema changes
effect of skill injection/reordering
```

Determine whether OpenCode preserves a stable prefix.

### Phase H — Main KV placement

A/B:

```text
default/GPU KV
vs
--no-kv-offload
```

This is not assumed to favor CPU KV. Measure.

### Phase I — Main KV precision

A/B:

```text
F16
vs
Q8_0
```

Do not automatically apply the same quant to MTP/draft KV.

### Phase J — Host prompt cache budget

Current master llama.cpp docs report a default host prompt-cache budget of 8192 MiB.

On the measured Q4 run only ~11.35 GB host RAM remained free.

Test lower budgets, e.g.:

```text
2048 MiB
4096 MiB
```

while measuring cache effectiveness and Windows paging pressure.

### Phase K — Batch / ubatch / threads

Only after the major architectural knobs are understood.

Target:

```text
single-user interactive coding
not serving throughput
```

Keep:

```text
--parallel 1
```

unless evidence strongly says otherwise.

### Phase L — OpenCode

After raw protocol/runtime stability:

```text
inspect repository
→ edit bounded file(s)
→ run test
→ observe result
→ repair if needed
→ return exact evidence
```

### Phase M — OpenClink

Then:

```text
Claude Code
→ OpenClink
→ OpenCode
→ llama-server
→ Qwen
```

Start read-only, then bounded edits.

### Phase N — Real Xeno benchmark

Use real tasks:

```text
repo exploration
small bug fix
medium bug fix
test-first implementation
refactor with tests
documentation update
bounded multi-file feature
debug failing test
code review
tool-heavy task
long-context task
```

The winner is the configuration that maximizes:

> **Verified Successful Coding Tasks / Hour**

## 5. Current Candidate Profiles

### Reliability hypothesis

```text
UD-Q4_K_XL
CUDA
parallel 1
vision off
reasoning medium candidate
precise-coding temp 0.6 candidate
ngram-simple candidate
main KV Q8 at larger contexts
prompt cache on
context checkpoints on
bounded host prompt-cache budget
safe VRAM headroom
```

### Performance hypothesis

```text
UD-Q3_K_XL
CUDA
parallel 1
reasoning medium candidate
MTP n=2/3 candidate
or ngram-simple if it wins
```

## 6. Hard Stop Conditions

Reject/stop a profile if it:

```text
breaks tool-call round trip
causes repeatable output corruption
forces Windows into heavy paging
causes driver-level VRAM eviction
is only faster in synthetic decode but worse in verified task throughput
depends on a cache behavior OpenCode does not actually preserve
```

## 7. Deliverables

Final optimization session must produce:

```text
fastest stable profile
reliability-first profile
recommended production profile
exact launch commands
Q3/Q4 comparison
MTP/ngram comparison
reasoning-effort comparison
sampling comparison
context-depth curve
GPU-vs-CPU KV result
F16-vs-Q8 KV result
cache hit/reuse data
tool-call reliability table
verified-task throughput table
unresolved risks
```
