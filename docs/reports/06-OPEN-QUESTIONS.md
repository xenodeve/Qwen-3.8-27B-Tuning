# Open Questions — What Is Unmeasured, and What To Do Next

> **Date:** 2026-08-20 UTC+7 · rewritten again after Dynamic V3 and the panel review
>
> **Superseding note.** The list below was accurate at 2026-08-19. Since then the
> vendor republished the artifact repo mid-session (report 12 §0), a three-agent
> panel reviewed the test plan (report 14), nine further models and quants were
> measured (report 13), and the developer restated the goal as **a usable context
> beyond 128K**. §0 is the current queue; the older sections still hold except
> where §0 contradicts them.
> Ordered by value, not by effort.
>
> The previous version of this document ended with *"the measurement queue is now
> empty."* It was, for the question being asked then — which flags to set on Q4.
> The new research (report 08) changed the question to which **artifact** to run,
> and the queue refilled. This is the state after Experiment A.

---

## 0. The queue as of 2026-08-20, in priority order

**1. Deep-context retrieval quality, on anything that is not Q4.** Unchanged as
the top item and now a year older in dog-years: the `30/30` at 64K and `10/10` at
a 114K prompt belong to `UD-Q4_K_XL`. Nine artifacts have since been measured at
depth and **not one has a quality number there**. Every depth figure in reports
11 and 13 is throughput and residency. The corpus and the harness both exist;
this is two runs of work and it gates every deep recommendation.

**2. The context ceiling.** The restated goal. Everything measured collapses at
256K — `IQ2_XXS` 43+22 at 2.23 tok/s, `AD-IQ1_M` 46+19 at 2.29 — because KV is
allocated from the pool the weights live in. Two levers, from opposite
directions: **smallest weights** (`Bonsai-27B-Q1_0`, 3.54 GiB, leaves 5.1–5.8 GiB
free — more than any arm measured) and **smallest cache per token** (Ornith-9B
holds 1,152 MiB at 128K against the 27B's 2,016). `bench/ctx_ceiling.py` is
queued across both plus three Qwen arms.

**3. Quality on the fast candidates that have none.** `Bonsai-27B-Q1_0`
(+80.12 % resolved), `gptoss20b` (+68.44 % resolved), and both MoE arms have
**no corpus result at all**. The MoE figures were additionally taken at 227–339
MiB free, below the reserve, so they are directional only.

**4. Re-measure the MoE arms above the VRAM reserve.** `qwen36moe` and
`ornith35moe` are the fastest things on this machine (+99 % and +80 %) and were
measured in the regime where this project has already seen driver eviction. That
also happens to be the one place left to test Gemini's WDDM hypothesis (report
14 §3), since the ~345 MiB band is exactly where the unexplained
`[6.70, 8.28, 11.57]` spread was observed.

**5. A multi-file task.** The largest untaken panel recommendation (report 14
§2): the corpus is ten self-contained functions and structurally cannot see
cross-file interface drift. One 2–3 file repository repair with an executable
integration test.

**6. The standalone MTP drafter.** V3 removed the built-in head below Q2_K_XL and
ships a 1.28 GiB drafter separately. On a resident target this project measured
speculation at **−7 %**, because the head's VRAM displaced six layers. `v3-iq1s`
holds 128K at 65+0 with **1,436 MiB spare** — the one condition under which that
result would not carry over, because the drafter would not change the split.

**7. Whether `UD-Q2_K_XL` really kept its MTP head.** Unsloth's documentation
says the head was removed from *"Q2_K_XL and smaller"*; the layer counts say
`v3-q2kxl` loads **66** layers where every smaller V3 arm loads 65. Either the
documentation is imprecise or the artifact is. Cheap to settle by inspecting the
tensor list.

---

## 1. Deep-context retrieval quality on IQ2_XXS — **the top item**

`UD-IQ2_XXS` is now the default (report 10) and its depth throughput is measured
(report 11). Its **retrieval quality at depth is not.** The `30/30` at 64K and
`10/10` at a 114,406-token prompt in report 03 were measured on **Q4**.

This matters more for this artifact than it would for a milder one, because the
documented failure mode of conventional 2-bit builds in this family is
*selective*: aggregate scores hold while reasoning-heavy and long-span tasks
collapse (report 11 §4 quotes the vendor's own numbers).

**Everything needed already exists.** `bench/run_deep_bench.py --v2` with the
corpus scaled to the window, and `production-iq2xxs-deep.ps1`. Two runs — 64K and
128K — answer it.

Until then: **`production-q4-deep.ps1` for any task that depends on finding one
fact inside 100K tokens.**

---

## 2. The token budget is confounding every quality number we have

Low-bit artifacts reason far longer on the same task — median 2,811 chars on
`Q2_K_XL` and 1,023 on `IQ2_XXS` against **59** on Q4 (report 10 §2). At
`max_tokens 3072`, IQ2_XXS was truncated on **7 of 30** corpus attempts against
Q4's 3.

So `p1 = 73.3 %` for IQ2_XXS is a **lower bound**, and the gap to Q4's 83.3 % is
partly the probe. A re-run at a budget generous enough that truncation is rare
would move `p1` up and wall time up with it; neither has been measured, and the
trade decides how much of report 10's throughput advantage is real at the level
of accepted tasks.

**This is also an integration hazard**, not just a measurement one: a client
tuned to Q4's token appetite will truncate these models mid-reasoning and read it
as a refusal to call tools. It looked exactly like that here before
`finish_reason` was recorded.

---

## 3. Stability and prefix behaviour at depth

The 100-turn gate (report 10 §2) ran at **16K**. At 128K the free-VRAM margin is
412–503 MiB, below the 512 MiB reserve this project adopted after `--fit-target
256` produced intermittent driver eviction at 345 MiB (report 04 §5).

Untested: whether prefix reuse stays at 99 %, whether invalidation still recovers
on the next turn, and whether the thin margin produces the eviction signature —
a wide spread with occasional normal samples — rather than a lower mean.

---

## 4. Candidates not yet tried

Ordering from report 08 §4, updated by what Experiment A found.

| candidate | status | note |
|---|---|---|
| `UD-IQ2_M` (9.61 GiB) | **not tried** | sits between the two arms tested. Given the cliff, the question is only whether it reaches 65/65; if it does not, it is `Q2_K_XL` again |
| Bonsai `Q2_g64` (7.06 GiB) | **not tried** | needs verifying that mainline b10472 serves the g64 pack at all — the g128 `Q2_0` requires the PrismML fork (report 08 §1). Fetch by exact filename; `-hf …:Q2_0` resolves to `PQ2_0` |
| 35B-A3B MoE | **not tried** | `-ncmoe` exists in our build. Gate on a hostile multi-turn cache test before any quality number, per the research's warning about hybrid/MoE cache invalidation |
| Ornith-1.0-9B Q6 | **not tried** | `ornith-ai/`, not `deepreinforce-ai/` |

**Cost note:** Hugging Face throttled sustained downloads to ~1.2 MB/s for part
of 2026-08-19 and ran at 5–7 MB/s at other times, and **parallel downloads make
it worse, not better** (report 09 §4). Serialize, and pick deliberately.

---

## 5. Untuned flags

Unchanged from the previous version, except that `--no-kv-offload` has become
more interesting: at 128K and 256K, KV is what pushes layers off the GPU, so
trading PCIe latency for weight residency is now aimed at exactly the binding
constraint.

| flag | why it might matter | expected size |
|---|---|---|
| `--no-kv-offload` at 128K/256K | moves the 3.3–4.4 GB cache off the card; at 256K the split is 31+34, so the weights it would buy back are the whole advantage | unknown, possibly large |
| `--cache-ram` (default 8192 MiB) | relevant to host pressure at depth | reliability, not throughput |
| `--fit-ctx` | untouched | small |
| CPU affinity mask | Windows processor numbering must be discovered first; a wrong mask looks like an optimization and behaves like a regression | unknown |

**Do not** re-sweep `-t`, `-b`/`-ub`, `--fit-target`, draft depth, ngram, or the
speculative sub-knobs at 16K. Settled, and the remaining effects sit below the
13.6 % noise floor (report 04 §0).

---

## 6. The integration that has not started

Everything measured is synthetic: single-file coding tasks plus a synthetic
retrieval corpus and a synthetic agent loop. Untested:

- **OpenCode** — repository inspection, edit, test, observe, repair, return evidence.
- **OpenClink** → OpenCode → llama-server chain.
- **Real Xeno workload** across the task classes in the original plan.

**Check first when OpenCode is wired up**, in this order:

1. **Does its serialization respect the prefix-freeze rule?** One broken prefix
   costs **63 s at 16K and 248 s at 64K** on Q4 (report 09 §1, 10-point fit,
   r² 0.968). That is worth more than every flag in this project combined.
2. **What `max_tokens` does it send?** See §2.

---

## 7. Known limits of the current evidence

- **No accuracy difference was *detected* between Q4 and IQ2_XXS; equivalence was
  not shown.** 27/30 against 27/30 at n=30 cannot separate arms differing by a
  few percent.
- **The corpus may not probe the failure this quantization is known for.** Ten
  single-function implementations run three times is closer to the benchmarks
  that hide a selective collapse than to the ones that expose it (report 10 §5).
- **`p2` is measured and low** — 0.20 to 0.625 against the research's assumed
  0.93. Retries are not the cheap safety net the economic model assumes.
- **The `Q2_K_XL` empty-reply anomaly is unexplained.** 19/100 on Q4, 55/100 on
  `Q2_K_XL`, 1/100 on `IQ2_XXS` — not monotonic in quantization, so the obvious
  explanation is wrong and no other has been tested.
- **`bracket_matching` has never been solved** by any configuration or artifact —
  a capability ceiling, not a tuning artifact.
- **Temperature 0.6 vs 1.0 was never run**, and a partial n=30 protocol-gate
  comparison at temperature 0 vs 0.7 was cut short at n=2 to free the GPU for
  Experiment A. The n=10 pair (100 % vs 80 %) is not separable — Fisher p ≈ 0.47.
- **Per-lever attribution is not established** for the Q4 runtime flags. Only the
  stacked config is validated (+6.6 % paired / +9.6 % pooled).

---

## 8. Questions that are closed

Recorded so they are not reopened by accident. Rows marked **↺** were closed on
Q4 and **reopened by the artifact change**.

| question | answer |
|---|---|
| Does crossing the VRAM residency threshold beat Q4? | **Yes** — +220 % decode, same 27/30 accepted (report 10) |
| Is "nearly resident" good enough? | **No** — the last 4 CPU layers cost ~half the throughput |
| Should MTP be on? | **On for Q4, off for resident artifacts** — the draft head's VRAM pushes target layers to CPU |
| Does Q3 overtake Q4 at depth? | **No** — settled before Experiment A |
| ↺ Is 256K usable? | **Reachable now, not usable for a loop** — 1.71 tok/s behind an 11-minute prefill, but no host paging (report 11 §3) |
| Does the hybrid re-prefill bug affect us? | **No** — append-only turns evaluate ~40–50 tokens |
| Is `FA_ALL_QUANTS` needed **for Q8 KV**? | **No** — `Q8_0` is in the always-compiled list, so the flag never applied. **Still open for `q4_1`/`q5_0`/`q5_1` and asymmetric K≠V** ([`CORRECTIONS` §29](CORRECTIONS.md)) |
| Does MTP change output? | **No** — byte-identical greedy across every speculative config |
| Is 3× achievable *from speculation*? | **No** — 1.30–1.47×. It was achievable from **residency**: 3.2× |
| Does Bonsai run on our binary? | **Its headline `Q2_0` does not** — that is g128 and needs the PrismML fork; mainline's `Q2_0` is group-64 (report 08 §1) |
