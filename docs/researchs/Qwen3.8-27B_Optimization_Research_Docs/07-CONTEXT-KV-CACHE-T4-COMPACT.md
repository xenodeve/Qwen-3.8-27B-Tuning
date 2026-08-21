# Context, KV Cache, Prompt Reuse, and T4-Compact

## 1. 256K Means Capacity Ceiling

Use:

```text
configured maximum = 256K
```

not:

```text
normal working set = 256K
```

The measured Q4 prompt-processing rate was ~518.8 tok/s at a 4.6K prompt, but prompt processing degrades with depth.

Cold 256K prefill can therefore be minutes.

## 2. Working-Region Hypothesis

Not production thresholds:

```text
32–96K    normal
96–160K   heavy
160–224K  very heavy
224–256K  emergency headroom
```

Measure before codifying.

## 3. Main KV

Initial long-context main-KV tests:

```text
F16
vs
Q8_0
```

and:

```text
GPU/default KV
vs
CPU KV (--no-kv-offload)
```

Do not assume CPU KV is faster.

## 4. Architecture-Family KV Estimate

**PROXY / INFERENCE**

For qwen3_5 dense 27B family:

```text
64 layers
full attention every 4th layer
4 KV heads
head dim 256
```

implies 16 full-attention layers and approximately:

```text
32,768 K/V scalar values per token
```

Approximate full-attention KV:

| Context | F16 | Q8 |
|---:|---:|---:|
| 64K | ~4 GiB | ~2 GiB |
| 128K | ~8 GiB | ~4 GiB |
| 256K | ~16 GiB | ~8 GiB |

Exact Qwen3.8 recurrent-state and runtime allocations must be measured.

## 5. Prompt Cache

llama-server:

```text
--cache-prompt
```

is currently documented as enabled by default.

Common-prefix requests can reuse KV and process only changed suffixes.

This means agent-turn cost depends heavily on **prefix stability**, not just total logical conversation length.

## 6. Context Checkpoints

Current llama.cpp docs expose:

```text
--ctx-checkpoints       default 32
--checkpoint-min-step   default 8192
```

For hybrid/recurrent architectures, checkpoints are important for state restoration/reuse.

Do not disable them early.

## 7. Host Prompt Cache

Current docs:

```text
--cache-ram 8192 MiB default
```

Measured Q4 free host RAM was ~11.35 GB after load.

Therefore an 8 GB prompt-cache budget is risky on the real workstation.

Benchmark:

```text
2 GB
4 GB
```

or similar.

## 8. Persistent Slot Save/Restore

Do not treat server slot persistence as the durable semantic memory of Xeno/T4 without strong evidence that hybrid/recurrent checkpoint state is restored correctly for the exact build/model.

Keep T4's durable memory as explicit handoff text/artifacts.

## 9. T4-Compact

Intended flow:

```text
semantic boundary detected
→ request_compaction
→ /handoff
→ validate durable handoff
→ /compact
→ detect completion
→ restore instruction
→ read handoff by path
→ reload needed skills
→ continue
```

State machine:

```text
IDLE
→ REQUESTED
→ HANDOFF_RUNNING
→ HANDOFF_READY
→ COMPACTING
→ COMPACTED
→ RESTORE_REQUIRED
→ RESTORING
→ READY
```

Invariant:

```text
NO VALID HANDOFF
→ NO COMPACT
```

## 10. Why T4 Is Also a Performance Feature

Originally T4 was mainly about reliability.

Now it is also about compute economics:

```text
old context carried forever
→ more prompt/state processing
→ slower cold recovery
→ more memory pressure
```

A semantic checkpoint can deliberately replace a large stale context with a compact durable state before a new large task.

## 11. What Not to Do

Do not:

```text
compact at a fixed 80% threshold only
assume 256K should be filled
preserve every reasoning trace forever
use server cache as the only durable memory
disable recurrent-context checkpoints without evidence
```

## Sources

- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json  (architecture-family proxy)
- PROJECT: Xeno skill-compliance/review-handoff plans
- MACHINE: `HANDOFF-BACK.md`
