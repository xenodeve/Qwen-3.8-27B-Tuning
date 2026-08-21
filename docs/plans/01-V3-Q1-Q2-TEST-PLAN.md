# Test Plan — Qwen3.8-27B Dynamic V3, Q1 and Q2 Ladder

> **Date:** 2026-08-20 UTC+7
> **Supersedes** the candidate-selection sections of
> [`00-OPTIMIZATION-PLAN.md`](00-OPTIMIZATION-PLAN.md) for the low-bit lane.
> That plan's flag-tuning phases are complete and its findings are in
> `docs/reports/00`–`11`; what remains open is which **artifact** to run.
> **Runner:** `C:/AI/qwen38-tuning/scripts/afk-v3plan.sh` executes stages 1 and
> 4 unattended; later stages are gated on their results.
> **Scope:** Qwen3.8-27B only. Every other family measured on 2026-08-19 —
> Ornith 9B/35B, Ternary Bonsai, Qwen3.6-35B-A3B MoE, gpt-oss-20b — is parked
> with its pre-V3 numbers on record. It is not deleted and not re-run.
> **GOAL RESTATED 2026-08-20:** the developer's objective is a usable context
> **beyond 128K**, not peak decode at 16K. That reorders what matters here. The
> deepest fully-resident context becomes the primary number, and 16K throughput
> becomes a tiebreak among arms that reach the same depth. Added as Stage 7 and
> run by `bench/ctx_ceiling.py`; results in
> [`../reports/12-DYNAMIC-V3-FIRST-RESULTS.md`](../reports/12-DYNAMIC-V3-FIRST-RESULTS.md) §6.
>
> **Focus inside that scope:** the **1-bit and 2-bit rungs**, because that is
> where this machine's residency cliff lives and where Unsloth's Dynamic 3.0
> release concentrates its claims.

---

## 0. Why this plan exists rather than continuing the old one

Two things changed under the previous plan while it was running.

**Unsloth republished the entire repo.** `unsloth/Qwen3.8-27B-GGUF` was updated
at **2026-08-19T16:39:23Z**, mid-session, to Dynamic 3.0. Same filenames, new
contents, new byte counts, new commit `27af057e`. Everything this project
measured in nineteen hours of benchmarking is the **pre-V3** generation.

```text
                       pre-V3 (measured)      V3 (current repo)
UD-IQ2_XXS              9,010,048,064          7,266,070,528
UD-Q2_K_XL             10,676,423,744          9,828,981,664
UD-Q4_K_XL             17,923,394,624         17,559,178,144
UD-IQ2_M               10,319,907,904          deleted
UD-IQ2_S                    —                  8,371,970,048   new
UD-IQ1_M                    —                  6,729,166,848   new
UD-IQ1_S                    —                  6,192,222,208   new
MTP/mtp-…-Q4_0.gguf         —                  1,369,590,656   new, standalone
```

**The MTP head moved.** From Unsloth's own documentation:

> *"The MTP module was removed from quants Q2_K_XL and smaller to conserve
> ~500MB disk space, available separately as Q4_0."*

So a V3 Q2-or-smaller arm differs from its pre-V3 namesake in **two** ways at
once: requantized weights, and no built-in speculative head. The 1.74 GB the
IQ2_XXS file lost is far more than the ~0.5 GB the head accounts for, so most of
it is genuine requantization — but the two effects must not be reported as one.

---

## 1. What the machine already established, and which this plan takes as given

Not re-opened. Re-measured only where V3 could plausibly move it.

| finding | value |
|---|---|
| **The residency cliff** | 33→61 GPU layers buys +64 %; 61→65 buys another +95 %. Almost all of the prize is in the last few layers. |
| Restart drift floor | **13.6 %** peak-to-peak on an unchanged config. Nothing below that is an effect. |
| MTP on a resident target | a **loss** (−7 % on pre-V3 Q2_K_XL): the draft head's VRAM pushes six target layers back to the CPU. |
| KV type at depth | `q4_0` is **+52 %** over `q8_0` at 128K *when it changes the split*, and **+1.6 %** (unresolved) when the model is already 65/0. KV type buys residency, not speed. |
| `--no-kv-offload` | **−33 %** even though it reaches 65/0. Total bytes moved per token beats weight residency. |
| KV kernels in b10472 | fast for `f16, bf16, q8_0, q4_0` (~1180 tok/s pp); `q5_1, q5_0, q4_1, iq4_nl` fall back to ~150–170 and worsen with depth. |
| Probe token budget | the single most persistent source of wrong verdicts today. See §4. |
| Prefix break cost | 63 s at 16K, 248 s at 64K (10-point fit, r² 0.968). |

---

## 2. Arms

Pinned by **exact byte count**, never by filename: the cache now holds two
snapshot directories with identical names inside, and a resolver that picked
either would have produced a paired, order-counterbalanced comparison of an
artifact against itself.

| arm | artifact | GiB | MTP head | role |
|---|---|---:|---|---|
| `iq2xxs-nomtp` | pre-V3 `UD-IQ2_XXS` | 8.39 | yes | **control** — current production |
| `v3-iq1s` | `UD-IQ1_S` | **5.77** | no | most headroom available |
| `v3-iq1m` | `UD-IQ1_M` | 6.27 | no | |
| `v3-iq2xxs` | `UD-IQ2_XXS` | **6.77** | no | direct successor to the control |
| `v3-iq2s` | `UD-IQ2_S` | 7.80 | no | replaces the deleted IQ2_M rung |
| `v3-q2kxl` | `UD-Q2_K_XL` | 9.15 | no | Unsloth's own efficiency pick |
| `q4-tuned` | pre-V3 `UD-Q4_K_XL` | 16.69 | yes | quality reference / escalation |

**V3 `UD-Q4_K_XL` is deliberately not downloaded.** It is 16.35 GiB and ~35
minutes of link time for a control this project already has, and the focus is
the Q1/Q2 rungs.

---

## 3. Stages, and the gate between each

Running a 30-task corpus costs 45–90 minutes per arm. Seven arms is a night.
So each stage eliminates arms before the expensive one.

### Stage 1 — residency and speed (paired, 3 rounds, ~35 min)

`model_arena.py`, order counterbalanced. Records GPU/CPU split, VRAM before and
after load, decode, prompt processing, greedy hash.

**Gate:** an arm proceeds if it reaches **full residency** *and* keeps
**≥512 MiB free** — the reserve adopted after `--fit-target 256` produced
intermittent driver eviction at 345 MiB, recognisable as a wide spread with one
normal sample rather than a lower mean.

The interesting question is not which is fastest. Every V3 Q1/Q2 arm should be
resident at 16K. It is **how much VRAM each leaves**, because §1 says that is
what buys depth.

### Stage 2 — protocol gate (n=15, `max_tokens 4096`, ~10 min/arm)

Nested-object tool call, required fields, `tool_call_id` round-trip.
`finish_reason` recorded per trial.

**Gate:** compliance **not materially below the control** at the same settings.
Absolute thresholds are rejected — the pre-V3 Q4 control scores 80 % on this
probe at temperature 0.7, so a "100 % required" rule would reject the production
model.

### Stage 3 — corpus (30 tasks, `max_tokens 8192`, 45–90 min/arm)

`run_retry_bench.py`. The decision metric. Reports `p1`, `p2`, accepted,
attempts per accepted, escalations, tokens, worker wall, merged and verified
tasks/hour, and truncation count.

**Budget is 8192, not 3072, and this is not negotiable.** See §4.

**Gate:** accepted count not below the control, at which point throughput
decides.

### Stage 4 — depth (64K and 128K, `q4_0` KV, ~25 min/arm)

**Selection rule corrected 2026-08-20** after two independent reviewers flagged
the original one. It read *"the two arms with the most free VRAM from Stage 1"*,
which selects the **smallest and most damaged** artifacts — precisely the ones
least likely to pass the coding corpus that decides the question. Free VRAM is
not the goal; it is the budget that buys residency, and §1 says KV compression
exists so that **better weights** stay resident, not so that VRAM sits empty.

The rule is now: **the largest arm that still holds full residency once the 128K
`q4_0` KV cache is allocated**, plus the Stage-3 utility winner and the control.
An arm qualifies as a depth candidate on quality *and* headroom together, never
headroom alone.

### Stage 5 — the standalone MTP drafter

V3 Q2-and-smaller have no built-in head, but Unsloth now ships
`MTP/mtp-Qwen3.8-27B-Q4_0.gguf` at 1.28 GiB separately.

On pre-V3 this project measured MTP as a **−7 % loss** on a resident target,
because the head cost VRAM and pushed six layers to the CPU. V3 IQ1_S leaves
~4.6 GB free. **The question is whether 1.28 GiB of drafter can be added without
changing the layer split at all** — which is the only condition under which the
pre-V3 result would not carry over.

Paired, on the arm with the most headroom, with the split recorded on both sides.

### Stage 0 — identity and residency preflight (new, ~5 min/arm)

Added 2026-08-20. Two checks that gate everything after them.

**Artifact identity.** Byte count alone is not enough: the cache holds two
snapshot directories with identical filenames, and this project has already
downloaded a wrong file whose byte count matched the right one exactly. Record
path, size, SHA-256 and the architecture string the loader reports, and abort a
run whose loaded artifact does not match its manifest entry.

**Real residency, not reported residency.** `bench/residency_check.py` reads the
process's *shared* GPU memory from the Windows performance counters during a
real generation. WDDM can page a CUDA allocation out to host RAM while llama.cpp
still reports the layer as assigned to `CUDA0`; throughput then falls off
exactly the cliff a CPU-resident layer produces, with nothing in the log to say
why. That is a precise candidate explanation for the unexplained instability at
345 MiB free — `[6.70, 8.28, 11.57]`, a 73 % spread with one normal sample.

First measurement, on the pre-V3 production artifact under load:

```text
dedicated 9,417 MiB   shared 98 MiB   = 1.04 %   free VRAM 654 MiB
```

**This is a baseline, not a verdict.** 1 % is the size of ordinary pinned
staging for host-to-device copies, not evicted weights. The number that carries
information is the **ratio compared across arms with different headroom**: it
climbing as free VRAM falls is eviction; sitting flat near 1 % is staging. An
absolute `shared == 0` gate was written first and discarded — it would have
repeated the same error as a "100 % tool compliance" gate that rejects its own
control.

### Stage 6 — stability (100 turns) on the winner only

Hangs, empty replies, prefix reuse, and recovery after forced invalidation.

---

### Stage 7 — the context ceiling (added 2026-08-20, now the primary question)

Everything measured so far collapses at 256K, and always for the same reason:
KV is allocated from the pool the weights live in, so context spends exactly the
VRAM that quantization freed.

```text
IQ2_XXS  + q4_0 @ 256K    43 + 22    2.23 tok/s
AD-IQ1_M + q4_0 @ 256K    46 + 19    2.29 tok/s
```

Two properties set the ceiling, and the strongest candidates come from opposite
directions:

| lever | candidate | evidence |
|---|---|---|
| smallest **weights** | `Bonsai-27B-Q1_0`, 3.54 GiB | 2.2 GiB below `v3-iq1s`, less than half the production artifact; untested at any depth |
| smallest **cache per token** | `Ornith-9B Q6_K`, 6.85 GiB | its KV at 128K is **1,152 MiB against the 27B's 2,016** — 43 % smaller, because a 9B has fewer layers and heads |

`ctx_ceiling.py` walks 128K → 160K → 192K → 224K → 256K, reading only the layer
split from the load report. About a minute per boot rather than the ten a 256K
cold prefill costs, and it stops at the first spill because deeper contexts only
allocate more cache. The split is deterministic, so unlike decode this is safe
to measure while other work shares the machine.

**Residency is necessary, not sufficient.** A ceiling says the weights stay on
the card at that depth. It says nothing about whether the model can still find
one fact inside that window — which remains verified on Q4 alone and is the
project's largest open quality risk.

## 4. The rule that has to be stated once, loudly

**Three wrong verdicts on 2026-08-19 came from an undersized token budget, and
every one of them looked exactly like lost capability:**

| probe budget | what it reported | what was true |
|---|---|---|
| `max_tokens 1024` | pre-V3 Q2_K_XL tool compliance **40 %** | **86.7 %** — every non-call was `finish_reason: length` |
| `max_tokens 3072` | AD-IQ1_M accepted **20/30**, failing five tasks the others pass | **27/30** at 8192; the `NameError`s were truncated code |
| `max_tokens 3072` | Bonsai accepted **15/30** | **27/30** at 8192; 35 of 60 attempts had been truncated |
| `n_predict 400` | Ornith-9B "failed the rename task" | cut off mid-class; its `<think>` block used the budget |

Median reasoning per call spans **59 chars (Q4) to 2,811 (pre-V3 Q2_K_XL)** — a
factor of 48. Any budget chosen for the control truncates the others, and the
artifact that reasons longest always looks like the weakest one.

**Therefore:** budget for the most verbose arm, record `finish_reason` on every
call, and treat a truncation count as part of the result rather than a footnote.

---

## 5. What this plan will not conclude

- **Nothing about deep retrieval quality.** Reports 02/03's `30/30` at 64K and
  `10/10` at 114K tokens were measured on **Q4**. No low-bit artifact has been
  checked, and this plan measures depth *throughput* only. This remains the
  largest open quality risk in the project.
- **Nothing from the vendor's numbers.** Unsloth's "72 % top-1 %" for IQ1_S and
  "+8 % top-1 %" for Q2_K_XL come from **Divergence-300 @32** — greedy 32-token
  agreement with BF16 over 300 held-out examples. That is a fidelity proxy, not
  a coding pass rate, and it is a vendor claim about a vendor benchmark.
- **Equivalence.** 27/30 against 27/30 at n=30 cannot separate arms that differ
  by a few percent. "No difference detected" is the strongest available claim.
- **Anything about the corpus's breadth.** Ten single-function tasks run three
  times is not a repository-agent workload, and it is closer to the benchmarks
  that hide a selective low-bit collapse than to the ones that expose it.
