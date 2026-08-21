# Benchmark Protocol — Qwen3.8-27B Local Coding Worker

## 1. Rule Zero

> One benchmark run is not evidence.

Measured unchanged Q4 runs varied:

```text
6.29
6.81
7.56 tok/s
```

~18% spread.

For noisy timings:

```text
N >= 3
```

Report:

```text
median
min
max
```

or mean/std where appropriate.

## 2. Capture Environment Before Every Launch

Record:

```text
timestamp
llama.cpp build
GPU driver
CUDA backend
free VRAM
used VRAM
free RAM
desktop processes
quant
context
GPU layers
KV placement/type
spec mode
reasoning profile
sampling profile
```

Why: `--fit on` can choose a different split when desktop VRAM changes.

## 3. Synthetic Metrics

Record:

```text
model load time
prompt processing tok/s
generation tok/s
TTFT
VRAM
RAM
context depth
accepted/drafted speculative tokens
acceptance rate
```

Use sufficiently long prompts; an 11-token prompt was shown to be useless for reliable pp measurement due fixed overhead.

## 4. Protocol Correctness Metrics

Record pass/fail for:

```text
plain completion
developer/system role
simple tool call
nested object args
parallel tool call if needed
tool-result continuation
repeated tool loop
reasoning parse
reasoning leakage
```

## 5. Deterministic Equivalence Test

For speculative decoding:

```text
same quant
same prompt
temperature 0 / greedy
top_k 1 if appropriate
same seed
```

Compare:

```text
baseline
ngram-simple
draft-mtp
```

Capture:

```text
raw text
token IDs if available
tool-call name
tool-call args
tool-call order
```

## 6. Core Matrix

First-pass matrix:

| Quant | none | ngram-simple | MTP n=2 | MTP n=3 |
|---|---:|---:|---:|---:|
| UD-Q4_K_XL | ✓ | ✓ | ✓ | ✓ |
| UD-Q3_K_XL | ✓ | ✓ | ✓ | ✓ |

Use same reasoning/sampling/context for all cells.

## 7. Reasoning Sweep

At fixed Q4 baseline:

```text
low
medium
xhigh
```

Task types:

```text
easy bounded edit
medium bug fix
hard but bounded debugging leaf
```

Measure task success and time, not only reasoning length.

## 8. Sampling Sweep

At fixed reasoning/spec mode:

```text
temperature 1.0
temperature 0.6
```

with:

```text
top_p .95
top_k 20
min_p 0
presence 0
```

Measure:

```text
task success
spec acceptance
tok/s
retry rate
```

## 9. MTP Depth Sweep

Only after picking the best MTP quant:

```text
n=2
3
4
5
6
```

Do not waste time deeply tuning a losing quant.

## 10. Context-Depth Benchmark

At top two configs:

```text
32K
64K
128K
192K
256K
```

Measure separately:

```text
cold prefill
generation at depth
incremental cached turn
```

## 11. Prompt-Cache Test

Run a multi-turn OpenCode-like serialization.

Then perturb one dimension at a time:

```text
tool schema order
system prompt
skill block
reasoning history
message content
```

Observe cache reuse and TTFT.

## 12. KV Matrix

Only after quant/spec winner emerges:

```text
main KV F16 GPU
main KV Q8 GPU
main KV F16 CPU
main KV Q8 CPU
```

Do not change all other knobs simultaneously.

## 13. Real Coding Tasks

A useful corpus:

```text
T1 repo mapping
T2 tiny bug fix
T3 medium bug fix
T4 add tests
T5 refactor
T6 documentation
T7 bounded multi-file feature
T8 debug from failure log
T9 code review
T10 long-context repo task
```

For each:

```text
pass/fail
tests pass?
wrong edits
unnecessary edits
tool errors
retries
wall time
tokens
final verification
```

## 14. Main Score

Possible summary metric:

```text
verified_successes / wall_clock_hours
```

Secondary:

```text
median verified task time
tool-call failure rate
retry rate
cold-prefill penalty
incremental-turn latency
```

## 15. Artifact Layout

```text
C:\AI\qwen38-tuning\
├─ EXPERIMENTS.md
├─ hardware.json
├─ scripts\
├─ logs\
└─ results\
   ├─ env-snapshots.jsonl
   ├─ llama-bench.jsonl
   ├─ protocol-results.jsonl
   ├─ task-results.csv
   └─ summary.md
```

## 16. PowerShell Caveat

On this machine, normal successful output from some native commands was written to stderr.

Do not use a global PowerShell setting that converts any stderr line into a fatal failure.

## 17. Stop Conditions

Stop a benchmark if:

```text
Windows begins paging heavily
VRAM driver eviction is visible
tool protocol is broken
server becomes unstable
results are confounded by a changing environment
```

## Sources

- MACHINE: `HANDOFF-BACK.md`
- https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
- https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
