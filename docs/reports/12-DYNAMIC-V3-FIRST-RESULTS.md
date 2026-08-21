# Dynamic V3 — Results, and the Instrument That Found Them

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **Date:** 2026-08-20 UTC+7 · executes
> [`docs/plans/01-V3-Q1-Q2-TEST-PLAN.md`](../plans/01-V3-Q1-Q2-TEST-PLAN.md)
> **Complete:** Stage 0 (identity, residency), Stage 1 (paired speed, 6 arms),
> Stage 2 (answer screen, 4 arms).
> **Running / queued at the time of writing:** Stage 3 corpus on the two arms
> that passed, Stage 4 depth, and a context-ceiling sweep added after the goal
> was restated as *"exceed 128K"*.
> **Read §4 first.** The fastest artifact this project has ever measured is
> unusable, and no throughput number shows it.

---

## 0. Which generation these numbers belong to

Unsloth republished `unsloth/Qwen3.8-27B-GGUF` in place at
**2026-08-19T16:39:23Z**, commit `27af057e`, mid-session. Same filenames, new
contents, new byte counts. **Reports 00–11 are the pre-V3 generation** — still
internally consistent and comparable to each other, not comparable to the
current repo.

```text
                       pre-V3 (reports 00-11)   V3 (this report)
UD-IQ2_XXS              9,010,048,064            7,266,070,528
UD-Q2_K_XL             10,676,423,744            9,828,981,664
UD-Q4_K_XL             17,923,394,624           17,559,178,144
UD-IQ2_M               10,319,907,904            deleted
UD-IQ2_S                    —                    8,371,970,048   new
UD-IQ1_M                    —                    6,729,166,848   new
UD-IQ1_S                    —                    6,192,222,208   new
MTP/mtp-…-Q4_0.gguf         —                    1,369,590,656   new, standalone
```

Identity is now verified by hash, not name or size:

```text
Qwen3.8-27B-UD-IQ1_S.gguf
  local  sha256 3895b6eaa91e705c06ad1938d16c22e86f073c6a67df86260a1da79be3d1f887
  repo   lfs oid 3895b6eaa91e705c06ad1938d16c22e86f073c6a67df86260a1da79be3d1f887
```

`model_arena.cached(repo, filename, size)` **raises on ambiguity** rather than
choosing. Without it the pre-V3 and V3 arms would both have resolved to
whichever snapshot the glob returned first, producing a paired,
order-counterbalanced comparison of a file against itself.

---

## 1. Stage 1 — speed, paired across three boots

Order counterbalanced, baseline = the pre-V3 `UD-IQ2_XXS` currently in
production. Effects are called real only above the machine's **13.6 %** restart
drift and with a consistent sign in every round.

| arm | GiB | layers | free VRAM | decode, per round | pp | vs control |
|---|---:|---:|---:|---|---:|---|
| `v3-iq1s` | 5.77 | 65 | **2,959–3,591** | 42.06 · **50.81** · 50.55 | 668–769 | **+27.94 % RESOLVED** |
| `v3-iq2xxs` | 6.77 | 65 | 1,972–2,565 | 44.78 · 45.11 · 44.84 | 740–759 | **+20.93 % RESOLVED** |
| `v3-iq1m` | 6.27 | 65 | 2,485–3,075 | 42.11 · 47.69 · 43.75 | 686–727 | **+19.54 % RESOLVED** |
| `v3-iq2s` | 7.80 | 65 | 926–1,558 | 40.71 · 40.52 · 40.68 | 713–759 | +9.46 % *(under floor)* |
| `v3-q2kxl` | 9.15 | **66** | 417–544 | 30.77 · 30.99 · 37.15 | 699–742 | −11.50 % *(under floor)* |
| pre-V3 `IQ2_XXS` | 8.39 | **66** | 505–1,113 | 33.62 · 39.31 · 39.04 | 665–728 | baseline |

**Dynamic V3 is genuinely faster at the small rungs.** Three arms clear the
drift floor with consistent sign.

**Do not read the raw decode column across rounds.** The control itself moved
33.62 → 39.31 between boots, a 17 % swing, because free VRAM at boot differed
(9,913 vs 10,530 MiB). The trustworthy column is *vs control*, which pairs
within a round.

### Stage 1 gate: residency AND ≥512 MiB free

`v3-q2kxl` fails at **417 MiB**, below the reserve adopted after `--fit-target
256` produced intermittent driver eviction at 345 MiB.

### The MTP head, counted rather than assumed

Unsloth's page says the head was *"removed from quants Q2_K_XL and smaller"*.
The layer counts say otherwise: every V3 arm below Q2_K_XL loads **65** layers
while `v3-q2kxl` and the pre-V3 control load **66**. **`UD-Q2_K_XL` still
carries its head** — the documentation is imprecise, the measurement is not.
That also explains part of why `v3-q2kxl` is the slowest arm: it pays VRAM for a
head it cannot use without a `--spec-type` flag.

---

## 2. Residency — the assumption everything rests on, finally checked

A review panel noted that llama.cpp's reported split cannot detect WDDM paging a
CUDA allocation to host RAM: the loader still says `CUDA0`, throughput falls off
exactly the residency cliff, and nothing in the log says why.
`bench/residency_check.py` reads the process's *shared* GPU memory from the
Windows performance counters during a real generation.

| arm | free VRAM | dedicated | shared | shared % |
|---|---:|---:|---:|---:|
| pre-V3 `IQ2_XXS` | 654 MiB | 9,417 MiB | 98 MiB | **1.04 %** |
| `v3-iq1s` | 3,290 MiB | — | — | **1.41 %** |

**The ratio does not climb as headroom falls** — the arm with five times the
headroom has a slightly higher ratio. That is ordinary pinned staging for
host-to-device copies, not eviction. **The project's residency conclusions
hold.** Still open: the ~345 MiB regime, where the MoE arms sat at 227–335 MiB.

An absolute `shared == 0` gate was written first and discarded; it would have
repeated the mistake of a "100 % tool compliance" threshold that rejects its own
control.

---

## 3. Residency at depth

| at 128K, `q4_0` KV | split | free VRAM | decode |
|---|---|---:|---:|
| pre-V3 `IQ2_XXS` | 58 + 7 | 589 MiB | 7.84 |
| AtomicChat `AD-IQ1_M` (pre-V3 vendor) | 65 + 0 | 557 MiB | 24.04 |
| **`v3-iq1s`** | **65 + 0** | **1,436 MiB** | *pending* |

`v3-iq1s` is the first artifact to hold **full residency at 128K with over a
gigabyte spare** — enough for a deeper window, or for the standalone 1.28 GiB
drafter, which is the one condition under which the measured "speculation costs
7 % on a resident target" would not apply.

### All three V3 arms hold 128K resident, and it buys nothing in speed

Measured 04:51–05:07 on 2026-08-20, two boots each, `q4_0` KV, speculation off,
2,304 MiB of cache in every case.

| arm | split | free VRAM (r1 / r2) | decode (r1 / r2) | prefill s |
|---|---|---:|---:|---:|
| `v3-iq1s` | 65 + 0 | 842 / 803 | 27.29 / 27.45 | 108.5 |
| `v3-iq1m` | 65 + 0 | 552 / 630 | 26.37 / 27.45 | 111.4 |
| `v3-iq2xxs` | 65 + 0 | 446 / 493 | 26.72 / 26.16 | 105.3 |

**The whole spread is 26.16–27.45 tok/s — under 5 %, well below the project's
13.6 % restart-drift floor.** Two gigabytes of weight difference between
`IQ1_S` and `IQ2_XXS` buys **no measurable decode at 128K**, because all three
have already crossed the residency cliff. This is the cliff result restated at
depth: past 65 + 0, weight size buys **headroom, not speed.**

That reframes the arm choice at 128K. It is not a throughput decision — it is a
question of how much margin you want, and only `v3-iq1s` clears the project's
512 MiB reserve on both boots. `v3-iq2xxs` sits at **446–493 MiB, below the
reserve**, which by the project's own rule makes it a stability risk rather
than a validated configuration (report 04 §5).

**None of this rehabilitates the V3 arms** — §7 rejected two of them on the
corpus. It says the *depth* lever is exhausted at 128K, and that the remaining
question is how far past 128K the headroom reaches, which is what
`afk-ceiling.sh` measures.

---

## 4. The result that matters: fastest is not usable

`bench/answer_screen.py` — three probes per arm, `max_tokens 8192`, about four
minutes, run **before** any 45–90 minute corpus.

| arm | GiB | answered | contract OK | gate | greedy hash |
|---|---:|---:|---:|---|---|
| `v3-iq1s` | 5.77 | **0/3** | 0/3 | **REJECT** | **differs** |
| `v3-iq1m` | 6.27 | **3/3** | **3/3** | **PASS** | same as pre-V3 |
| `v3-iq2xxs` | 6.77 | 2/3 | 2/3 | MIXED | **differs** |
| `v3-iq2s` | 7.80 | **3/3** | **3/3** | **PASS** | same as pre-V3 |
| `v3-q2kxl` | 9.15 | — | — | failed Stage 1 (417 MiB) | same as pre-V3 |

### V3 IQ1_S: 50.8 tok/s and nothing to show for it

The fastest 27B artifact this project has measured, fully resident at 16K *and*
128K, and it produced **no usable output in twelve corpus attempts and three
screen probes.** From the parsed message llama.cpp logs:

```text
finish_reason  stop          <- not "length"; the budget was never the constraint
content        ""            <- nothing emitted
reasoning      14,523 chars  <- and 17k, 21k, 24k, 29k, 32k, 37k on other probes
```

The reasoning loops — *"I think this is correct. Let me finalize."* → *"Actually,
I realize I should reconsider."* → *"Let me write the final version."* → back to
the start, six times over. **It cannot exit its own reasoning, so it never writes
the answer.** No budget fixes that: it stopped voluntarily at 4,038 tokens with
empty content.

The raw completion path shows the same damage in a different shape — it does
produce code there, at 50.7 tok/s, and the code carries the fingerprints:

```python
last[1] = max(last[1], interval[1])
last[1] = max(last[1], interval[1])      # emitted twice
```

plus a missing `sorted()` the prompt explicitly required, and it runs on into
unrelated functions rather than stopping. This is catastrophic repetition, which
the handoff names as a hard reject condition for 1-bit artifacts.

### The failure is bimodal, not gradual

`v3-iq2xxs` answered two probes with **378 characters of reasoning in 5 seconds**
and blew up to **23,604 characters** on the third. Not "the model got more
verbose" — a model that occasionally enters a loop and cannot leave. A mean would
hide both halves. `rotated_search` triggered the runaway on both `iq1s` and
`iq2xxs`, so part of the trigger belongs to the task.

### It is not monotonic in size

`iq1m` at 6.27 GiB passes 3/3 where `iq2xxs` at 6.77 GiB is mixed, and `iq1s` at
5.77 GiB fails outright.

### The greedy hash predicted the gate, 5 for 5

Three arms return `227749403A7404D4` — byte-identical to the pre-V3 generation
*and* to Q4 on the same mechanical rename task. Those three are exactly the three
with no quality problem. The two that diverge are exactly the two the screen
caught. A 30-second probe predicting a 4-minute gate is worth having, though
five arms is a mechanism worth trusting, not a validated rule.

---

## 5. What the instrument changes caught, on first use

Every one of these came from a three-agent review panel earlier the same day,
and every one caught something real immediately.

| change | what it caught |
|---|---|
| `check_output_contract` | `IQ1_S` emitting **no fenced block at all**, twelve times out of twelve — hidden behind a `NameError` because `extract_code` falls back to the whole reply and the subprocess then ran prose |
| censored-attempt handling | `merge_intervals` (truncated, outcome unknown) correctly separated from `lru_cache` (stopped, genuinely failed). Both would previously have counted as "1-bit is broken" |
| `residency_check` | that WDDM eviction is **not** occurring — closing a hypothesis rather than opening one |
| capability/throughput split | four arms tie at 27/30 accepted and differ only in wall clock, 2,004 s to 4,572 s. `merged_tasks_per_hour` multiplies the two and was being read as a capability ranking |
| request-failure guard | a corpus whose server was killed at 02:00:17 by a colliding queue: 26 of 30 tasks returned HTTP 503 in 0.0 s and the summary still printed *"3/29 accepted, 22.0 merged tasks/hour"* |
| ambiguity guard in `cached()` | refused to pick between two snapshots holding `Qwen3.8-27B-UD-Q2_K_XL.gguf`. It also aborted three unrelated depth steps at import time, which is the cost of a guard that raises rather than guesses |
| port lock in `swap-model.sh` | held by the calling job, not the swap — a lock released when a five-second swap ends protects nothing across the hour-long corpus that follows |

Harness: **81 tests, all passing.**

---

## 6. State of play

**Settled.** V3 is faster than pre-V3 at the 1-bit and 2-bit rungs. `v3-iq1s` is
rejected on quality despite winning on speed. `v3-iq1m` and `v3-iq2s` pass the
screen. `v3-q2kxl` fails the VRAM reserve and still carries an MTP head.

**Running or queued.** Stage 3 corpus on `v3-iq1m` (a re-run; the first attempt
lost its server) and `v3-iq2xxs`; Stage 4 depth at 128K on three arms; and a
context-ceiling sweep.

**The goal moved.** The developer restated it as *"exceed 128K"*. That reframes
which number matters: not 16K decode but the deepest context that keeps full
residency. Everything measured so far collapses at 256K —

```text
IQ2_XXS  + q4_0 @ 256K    43 + 22    2.23 tok/s
AD-IQ1_M + q4_0 @ 256K    46 + 19    2.29 tok/s
```

— because KV is allocated from the pool the weights live in. Two properties
decide the ceiling and they are not the same one:

- **weight size** — `Bonsai-27B-Q1_0` at 3.54 GiB is the smallest artifact on
  disk, 2.2 GiB below `v3-iq1s`;
- **cache per token** — a 9B holds a much smaller cache at the same depth:
  Ornith-9B measured **1,152 MiB at 128K against the 27B's 2,016**, 43 % less.

`bench/ctx_ceiling.py` walks 128K → 160K → 192K → 224K → 256K reading only the
layer split, about a minute per boot instead of the ten a 256K cold prefill
costs, and stops at the first spill because deeper contexts only allocate more.

---

## 7. The first completed V3 corpus run — `UD-IQ1_M`, 30 tasks

Finished 03:54:45 on 2026-08-20, `max_tokens 8192`, the same ten tasks over
three passes that every arm below ran, on the same machine.

| arm | generation | p1 % | attempts/accepted | worker wall s | **verified tasks/hr** |
|---|---|---|---|---|---|
| `iq2xxs-mt8192` | pre-V3 `UD-IQ2_XXS` | **83.3** | 1.30 | 2,004 | **48.5** |
| `iq1m-mt8192` | pre-V3 `AD-IQ1_M` (AtomicChat) | 76.7 | 1.37 | 2,756 | 35.3 |
| `v3-iq1m` | **V3 `UD-IQ1_M`** | **33.3** | **5.30** | **6,006** | **6.0** |

**Three times the wall clock for 40 % of the pass rate.** Against the pre-V3
`IQ2_XXS` control at an identical token budget, verified throughput falls by a
factor of eight.

### The mechanism is the same one that rejected `IQ1_S`

```text
attempts_seen                     53
attempts_truncated_by_budget      27      (51 %)
output_contract_pct             41.5      compliance, so 58.5 % violated
censored                           9
censoring_could_change_verdict  True
```

Over half of all attempts ran to the 8,192-token wall inside the reasoning
block and never emitted a fenced answer. The `error` strings the corpus records
— `NameError: name 'evaluate' is not defined`, `name 'is_balanced' is not
defined`, `name 'Node' is not defined` — are **downstream of that**, not coding
failures. Median reasoning was 19,280 characters, maximum 33,871.

`censoring_could_change_verdict` is `True`, so **33.3 % is a lower bound** on
capability. It does not rescue the arm: even granting every censored task,
the wall clock is unchanged and 6,006 s buys at most 19 accepted tasks against
the control's 2,004 s for 25.

### What this does and does not indict

**Not 1-bit.** `AD-IQ1_M` is the same bit class on the same model and reaches
76.7 % at 1.37 attempts per accepted task. A 1-bit Qwen3.8-27B is a working
coding worker. The regression belongs to **this generation's 1-bit tier**,
measured against a same-bit-class control rather than against Q4.

Whether it extends to V3's 2-bit tier is being measured now — `corpus-iq2xxs`
started at 03:54:45 on V3 `UD-IQ2_XXS`.

### `UD-IQ2_XXS` — the cleanest comparison this project has

Finished 04:51:29. Same vendor, same filename, same quant tier: **only the
generation changed.** Unsloth republished the file 19 % smaller
(9,010,048,064 → 7,266,070,528 bytes). Verified against the loader:
`snapshotsaf057e…\Qwen3.8-27B-UD-IQ2_XXS.gguf`, 851 tensors.

| | pre-V3 | **V3** | |
|---|---|---|---|
| p1 % | 83.3 | **44.4** | |
| attempts per accepted | 1.30 | **2.53** | |
| worker wall s | 2,004 | **3,384** | |
| **verified tasks / hour** | **48.5** | **20.2** | **2.4× worse** |
| attempts truncated by budget | 1 | 7 of 48 | |

`accepted_of_decided` is 19/27 and `censoring_could_change_verdict` is `True`,
so 44.4 % is again a lower bound — but the wall clock is not censored, and
3,384 s for at most 22 accepted tasks cannot reach the control's 2,004 s for 25.

### The mechanism and the regression are two different findings

The runaway-reasoning collapse **does not** extend to the 2-bit tier:

| | `v3-iq1m` | `v3-iq2xxs` |
|---|---|---|
| attempts truncated by budget | 27/53 (51 %) | 7/48 (15 %) |
| reasoning chars, median | 19,280 | 5,398 |
| error strings | `NameError: name '<the requested function>' is not defined`, every one | `AssertionError`, `SyntaxError`, `NameError` |

At 2-bit the model answers and is wrong, which is the failure the corpus was
built to measure. At 1-bit it never answers, and the `NameError` is the
harness reporting an empty extraction.

**The throughput regression extends to both.** V3 is 8× worse at 1-bit and
2.4× worse at 2-bit against same-bit-class controls. Do not read "the reasoning
loop is confined to 1-bit" as "2-bit V3 is fine".

One caveat that the 1-bit row does not carry: `output_contract_pct` is 58.3 %
compliance here against 41.5 % at 1-bit, yet only 15 % of attempts were
truncated. So most V3 `IQ2_XXS` contract violations are **not** truncation — it
stops on its own and omits the fence. That is a third failure mode, unmeasured,
and it is not what rejected either arm.

### The screen did not catch it

`answer_screen.py` passed `v3-iq1m` **3/3, `finish_reason: stop` every time**,
about ninety minutes before the corpus reached these numbers. The screen's
probes are short enough that the arm completes them; the tasks it runs away on
are `expr_eval`, `tree_codec`, `rotated_search` and `lfu_cache`.

The screen still earns its place — it rejected `IQ1_S` for four minutes of GPU
time instead of ninety. But it is a **floor, not a gate**: it rejects, it does
not promote. `SESSION-STATE.md` §3 previously said this arm was likely to
supersede the recommendation on the strength of that 3/3. It has been corrected.

---

## 8. What none of this establishes

- **No deep-context retrieval quality on any low-bit artifact.** The `30/30` at
  64K and `10/10` at a 114K prompt belong to **Q4**. This remains the project's
  largest open quality risk, and the ceiling sweep does not touch it: residency
  is necessary, not sufficient.
- **Two V3 corpus results, not a V3 verdict.** `UD-IQ1_M` and `UD-IQ2_XXS`
  completed (§7); `UD-Q2_K_XL` and `UD-IQ1_S` are unrun on the corpus, so the
  regression is measured at two tiers and extrapolated to none. Two earlier attempts were
  spoiled — one cancelled after the reasoning-loop finding, one killed by a
  colliding queue — and are marked `SPOILED` in `results/retry-bench.jsonl`
  rather than deleted.
- **The 1-bit comparison is across vendors as well as generations.** `AD-IQ1_M`
  is AtomicChat's build and `UD-IQ1_M` is Unsloth's. Generation and packer move
  together, so §7 attributes the regression to the artifact, not to a named
  cause inside it.
- **Three probes cannot rank arms**, only separate a working artifact from a
  broken one.
- **`IQ1_S` is rejected for this workload, not proven worthless.** A pipeline
  that does not use a reasoning template, or that caps reasoning explicitly, was
  never tested.
- **The vendor's numbers are untested here.** "72 % top-1 %" for IQ1_S and
  "+8 %" for Q2_K_XL come from Divergence-300 @32 — greedy 32-token agreement
  with BF16 over 300 held-out examples. A fidelity proxy, not a coding pass rate.
