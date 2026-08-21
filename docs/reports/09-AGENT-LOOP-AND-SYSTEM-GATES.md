# Agent-Loop Gates and System Findings — Q4 Control

> **Date:** 2026-08-19 UTC+7
> **Why this exists:** the new research (report 08 §0) makes three things
> preconditions for adopting any candidate worker, ahead of tok/s: schema-correct
> tool calling, survival across a long agent loop, and prefix reuse. This project
> had numbers for none of them. This report establishes the **control** values on
> the tuned Q4 production profile, so a candidate has something to be compared
> against rather than an absolute threshold invented in a document.
>
> Every number here is from `production-q4-tuned.ps1` on build 10472, measured
> 2026-08-19 between 06:05 and 06:35 UTC+7, while a model download saturated the
> network link but touched neither GPU nor the server.

---

## 1. The headline: a broken prefix costs more than every runtime flag combined

`bench/stability_gate.py` runs one append-only agent conversation and edits the
system block every tenth turn — exactly what injecting a skill does to a real
agent loop. Ten such invalidations, at increasing depth:

| turn | tokens re-evaluated | wall |
|---:|---:|---:|
| 10 | 2,925 | 10.64 s |
| 20 | 3,478 | 14.95 s |
| 30 | 4,008 | 15.20 s |
| 40 | 4,650 | 21.75 s |
| 50 | 5,450 | 23.64 s |
| 60 | 6,166 | 25.93 s |
| 70 | 6,850 | 26.35 s |
| 80 | 7,581 | 29.50 s |
| 90 | 8,290 | 32.30 s |
| 100 | 9,066 | 35.41 s |

Least-squares slope over all ten (`harness.marginal_rate`, r² = **0.968**):

```text
prefill 265.5 tok/s,  constant offset 1.55 s
```

Projected cost of **one** prefix break:

| context | cost |
|---|---:|
| 16K | **63.3 s** |
| 32K | 125.0 s |
| 64K | **248.4 s** |

Set against the entire runtime-tuning campaign, which bought **+6.6 % of decode**
(report 01, paired) — about 0.8 tok/s, or roughly 4 s on a 500-token reply:

> **One skill injection at 64K undoes more than five minutes' worth of every
> flag this project tuned.**

That is the quantitative form of the prefix-freeze rule in report 05. It is no
longer a warning borrowed from a research document; it is this machine's slope.

### A method note, because the first version of this number was derived wrongly

The cost was first computed by dividing one perturbed turn's wall time by its
token count. Wall time is prefill **plus** the decode of the turn's 48 tokens,
so that ratio charges the decode to the prefill. The slope across ten points is
the correct estimator — the decode component is constant at every point and
falls into the intercept.

In this case the two agree closely (274.9 vs 265.5 tok/s, 3.5 %) because decode
is small next to prefill at these depths. The correction changed the number by
almost nothing and the method by everything: a single-point ratio has no
residual to inspect, and would have gone on being wrong at a depth where decode
is not small. `marginal_rate` is tested, and refuses fewer than three points
because two always fit a line perfectly and report certainty they do not have.

---

## 2. Stability: 100 turns, no degradation

```json
{"turns_survived": 100, "hangs": 0, "p50_wall_s": 3.38, "p95_wall_s": 23.64,
 "steady_reuse_median_pct": 99.1,
 "invalidation_recovered": 9, "invalidation_not_recovered": 0}
```

- **Steady-state reuse 99.1 %** — 46 tokens evaluated per turn against a cache
  that grew to 9,066. The prefix contract holds under append-only growth.
- **Every invalidation recovered on the very next turn.** Turn 80 re-evaluated
  7,581 tokens; turn 81 was back to 46 with 7,622 cached. No stuck slot, no
  residual state. This is the failure mode the research warns about for
  hybrid/MoE models, and Q4 does not have it.
- **No latency drift with depth.** First ten turns median 4.82 s, last ten
  3.77 s. p95 is dominated by the ten deliberate invalidations, not by decay.

### One thing this probe cannot claim

19 of 100 turns returned **zero tokens**. They are not a degradation signal:
none fell on a perturbed turn, and there is no depth trend (12 in the first
half, 7 in the second). Every one reported 0.0 tok/s, i.e. an immediate stop on
the first token.

The likely cause is the probe, not the model: `stability_gate` drives
`/completion` with a hand-built `<system>/<tools>/<assistant>` framing, because
only the raw endpoint exposes `cache_n`. That framing is not Qwen3.8's chat
template, and the model sometimes reads the turn as already finished. **Stated
as a hypothesis** — it has not been tested, and the cache and stability
conclusions above do not depend on it.

---

## 3. Tool-call protocol: the research's 100 % gate would reject Q4 itself

`bench/protocol_gate.py` asks for one `apply_patch` call with a **nested array of
objects** in its arguments, then feeds the tool result back and checks that the
model continues instead of re-issuing the same call.

| | temp 0.7, n=10 | temp 0.0, n=10 |
|---|---:|---:|
| emitted a schema-correct call | 80 % | **100 %** |
| nested `edits` array well-formed | 80 % | **100 %** |
| `tool_call_id` round-trip | 70 % | **100 %** |
| required-field omissions | **0** | **0** |

Two readings, one firm and one not:

**Firm.** Whenever Q4 emits a call, the schema is right — zero required-field
omissions in twenty trials, and the nested array never degraded. Every failure
was the model answering in prose instead of calling the tool at all. So the
research's threshold — *"Required tool/schema compliance 100 %"* — cannot be
applied as an absolute: at temperature 0.7 the production model scores 80 % on
its own gate. A candidate must be compared against the control **at the same
sampling settings**, or it will be rejected for a property Q4 shares.

**Not firm.** 8/10 versus 10/10 is not a resolved difference. Fisher's exact
gives p ≈ 0.47 — nowhere near separable. That temperature drives the no-call
rate is a hypothesis; an n=30 pair is running to settle it, and this section
will be amended with the result rather than left to imply what it did not show.

---

## 4. System findings: the network, not the machine, is the current bottleneck

Recorded because the goal directive covers the machine as well as the model, and
because this is what actually limited today's throughput.

| measurement | value |
|---|---|
| Wi-Fi link | 802.11ac, ch 36, **520 Mbps** receive, signal 90 % |
| Sustained Hugging Face download | **0.66 – 1.30 MB/s** |
| Cloudflare 20 MB test, run *concurrently* | **2.39 MB/s** |
| Two HF streams | 0.53 MB/s combined — *worse* than one |
| Three HF streams (added `hf` CLI + Xet) | 1.30 MB/s — unchanged from one |

The link is not the limit and neither is the downloader: the `hf` CLI with the
Xet backend was no faster than llama.cpp's plain HTTP client, and additional
streams only split the same allowance. Hugging Face throttles the aggregate, and
the rate decayed over the session (5.2 MB/s at 05:55, 0.66 MB/s at 06:35).

**Consequence for the plan:** a 10 GiB artifact costs roughly 2.5–4 hours to
fetch. Candidate downloads must be serialized and chosen deliberately — running
two in parallel finishes both later and neither sooner. A faster uplink would be
the single highest-leverage change available to this project today, and it is
not something the machine can fix.

---

## 5. What this changes about the plan

1. **Gates now exist and have control values**, so a candidate can be judged
   against Q4 rather than against a threshold from a document.
2. **The prefix-freeze rule has a price tag.** Any integration decision that
   re-serializes or re-stamps the prompt prefix between turns costs 63 s per
   turn at 16K. That is worth more attention than the remaining runtime knobs in
   report 06 put together.
3. **Nothing here measures a candidate.** No quantization comparison has been
   run yet; the Q2 artifact was still downloading while these were collected.

### Still pending in this session

- Retry economics on Q4 (`bench/run_retry_bench.py`) — measures p1, p2, attempts
  per accepted task and escalations per 100, the values report 08 §3 lists as
  *assumed* in the research's economic table.
- Protocol gate at n=30 for both temperatures, to settle §3's open question.
- Experiment A itself, once `UD-Q2_K_XL` finishes downloading.
