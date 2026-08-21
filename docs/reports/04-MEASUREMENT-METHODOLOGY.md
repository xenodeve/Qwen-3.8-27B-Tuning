# Measurement Methodology — How to Benchmark This Machine Without Fooling Yourself

> **Date:** 2026-08-18 UTC+7
> **Why this exists:** several results in this project were reported, then
> withdrawn, because the measurement was wrong rather than the model. This
> collects what those failures taught, so the next session does not repeat them.

---

## 0. The single most important number

**Restart-to-restart spread on an unchanged configuration is 13.6 % peak-to-peak
(stdev 4.5 %).** Six restarts of an identical config:

```text
11.63   12.59   12.60   12.63   13.21     tok/s (per-restart medians)
```

**Any claimed effect below ~14 % cannot be established by a single control-first
comparison on this machine.** That floor is larger than every individual runtime
flag measured in this project.

---

## 1. How drift manufactured a result that did not exist

The speculative sub-knob sweep looked like a clear win:

| knob | vs control |
|---|---|
| `--spec-draft-n-min 2` | **+11.6 %** |
| `--spec-draft-p-min 0.10` | **+9.8 %** |
| `--spec-draft-p-split 0.25` | **+8.8 %** |

Re-run against a **fresh** control, every one reversed:

| knob | vs fresh control |
|---|---|
| `-n-min 2` | **−0.8 %** |
| `-p-min 0.10` | **−10.1 %** |
| all three | **−4.3 %** |

The first sweep measured **machine drift, not knob effects**: its control ran
first, in a slow window, and every later configuration ran as the machine
recovered. A monotonic time trend is indistinguishable from a monotonic knob
effect when the control is sampled once, at one end of it.

A correlation check against free VRAM (**+0.06**) did *not* catch it, because the
drift was not VRAM-driven. Checking the confounder you thought of is not the same
as controlling for drift.

---

## 2. Rules that follow

1. **Interleave the arms.** `A/B/A/B/A/B`, not `control, then everything else`.
   Both arms then share the drift instead of it landing on whichever ran later.
2. **Report paired differences**, never a ratio of two separately-measured medians.
3. **Never add per-sweep deltas.** Each carries its own drift; summing compounds
   it. Adding three control-first sweeps produced "+19 % cumulative" where the
   paired measurement gives **+6.6 % mean / +9.6 % pooled**.
4. **N ≥ 3, decide on ranges.** Non-overlapping ranges, not point estimates. Re-run
   at N = 5 when leaders sit within noise.
5. **An effect below ~14 % needs a paired design** or it is not measurable here.
6. **Snapshot the environment before every launch.** `--fit on` derives the layer
   split from free VRAM *at boot*, and free VRAM ranged **9 933 – 10 530 MiB**
   across 22 recorded launches.

---

## 3. Choosing the right probe

### Prompt length decides the answer

- **An 11-token prompt cannot measure prompt processing.** It returned 13.7 tok/s
  where the real figure at 4 601 tokens was 518.8 — fixed per-request overhead
  dominates.
- **An 11-token prompt cannot measure speculation either.** It stayed inside
  9.86–11.90 tok/s across *every* configuration tested. All tuning decisions in
  this project were read from a code-rewrite prompt instead.
- MTP acceptance was **78.1 %** on a short instruction and **98.0 %** on a
  code-rewrite prompt. Benchmarks using short prompts understate real agent
  throughput.

### An equivalence probe must exercise what changed

The greedy-hash check (`temperature 0, top_k 1, seed 42`, SHA-256 compared) is
**stronger** than a pass-rate comparison *for flags that do not change
arithmetic* — thread counts, batch sizes, fit margins. A pass-rate comparison can
miss a small regression; an identical hash cannot.

It is **worthless** for anything touching the cache. With a 4-token prompt it
reported Q8_0 KV as identical to F16. At 46 557 tokens the two share **one
character of 778**.

---

## 4. Instrument bugs found in this project

Every one produced a plausible wrong number rather than a crash. That is the
failure mode to design against.

| bug | symptom | consequence |
|---|---|---|
| `[int](3/2)` is **2** in PowerShell (rounds half to even) | field named `tg_median` held the **maximum** | every sweep table mislabelled |
| PowerShell 5.1 writes a **BOM** on first `Add-Content -Encoding utf8` | `json.loads` raised on line 1; `except: pass` swallowed it | **baseline row silently deleted from every table** |
| device token is `CUDA0,` with a trailing comma | `== "CPU"` matched nothing | layer split reported `32+0` instead of `32+33` |
| `0.0 tok/s` from a generation that produced no tokens | folded into the sample list | median survives one, dies on two |
| deep corpus emitted `Handler0017` **twice** | two contradictory answers in context | the task measured nothing |
| corpus size test checked only a **lower** bound | 112K-token corpus passed the test | every request HTTP 400 → **0/18 in four seconds** |

**The lesson in one line: a benchmark harness needs regression tests as much as
production code does, and every primitive should raise rather than guess.**

`bench\harness.py` now holds the four repeat offenders — `median`, `load_jsonl`,
`parse_layer_split`, `project_prefill_seconds` — each written test-first, each
raising on empty/corrupt/unaccounted input. `bench\tests\test_harness.py` has 28
tests, and each names the incident it guards.

---

## 5. Stop conditions that were actually used

From the project's own protocol, and both were exercised:

- **Host paging.** 256K loaded and began prefilling, then showed host RAM free
  0.63 GB of 47.69, pagefile 10.11 GB, working set 26.64 GB, 296 pages/sec. It was
  **stopped**, and recorded as stopped-with-reason. Any throughput measured under
  that pressure would describe Windows paging, not the model.
- **Instability rather than slowness.** At `--fit-target 256` the code prompt
  produced `[6.70, 8.28, 11.57]` — a 73 % spread with one perfectly normal sample.
  That is intermittent driver eviction at 345 MiB free, and reading its median as
  "27 % slower" was wrong; the right reading is "unstable".

---

## 6. Checklist for the next measurement

```text
[ ] environment snapshot recorded before launch
[ ] arms interleaved, not control-first
[ ] N >= 3 (5 if leaders are close), ranges reported
[ ] read from a context-bearing prompt, not a short instruction
[ ] equivalence probe exercises the thing being changed
[ ] zero/empty samples dropped and counted, not averaged in
[ ] partial runs labelled interrupted, never merged with complete ones
[ ] effect larger than 14%? if not, say "not measurable" rather than reporting it
```

---

## 7. Instrument failures found on 2026-08-19/20

The table in §4 was written after one day. A second day added seven more, and
they share the shape of the first six: **the instrument returned a believable
number instead of a failure.** Recorded here because every one cost real time
and several produced published claims that had to be withdrawn.

| failure | symptom | what it cost |
|---|---|---|
| `-hf repo:<tag>` matches by **substring** | `:Q2_0` began fetching `PQ2_0.gguf` — a different file of **identical byte count** (7,165,121,600) in the same repo | two gigabytes transferred before a hash check caught it; nothing in the size, log or cache layout would ever have reported it |
| `-hf` does an **online etag check on every launch** | a fully cached model would not boot while a download saturated the link: `common_pull_file: download failed … retrying after 2 seconds`, in a loop | an unattended queue hung for eleven minutes on a server that needed no network |
| `parse_layer_split(total=65)` hardcoded | a 41-layer MoE reported "65 + 0" by slicing the last 65 of 451 assignment lines across two reserve passes | the arms happened to be resident so the conclusion held **by luck**; the same slice would have printed 65+0 with layers on the CPU |
| `cached()` returning `hits[0]` | after the vendor republished the repo in place, two snapshot directories held the same filename | would have produced a paired, order-counterbalanced comparison of an artifact **against itself**, with nothing in the output to show for it |
| the zero-worker-time guard was too narrow | a colliding queue killed the server mid-corpus; 26 of 30 tasks returned HTTP 503 in 0.0 s | the summary still printed *"3/29 accepted, 22.0 merged tasks/hour"* — the guard only fired when **no** task recorded worker time, and four had |
| `extract_code` falls back to the whole reply | an artifact that emitted **no fenced block at all** was scored as bad code: `NameError: name 'merge_intervals' is not defined` | twelve attempts misread as a coding failure when the model had produced no code |
| `merged_tasks_per_hour` read as a ranking | four arms tie at 27/30 accepted and differ only in wall clock, 2,004 s to 4,572 s | capability and verbosity multiplied into one number and presented as a quality order |

### The one that recurred four times in a single day

**An undersized `max_tokens` looks exactly like lost capability, every time.**

| budget | what was reported | what was true |
|---|---|---|
| `max_tokens 1024` | `UD-Q2_K_XL` tool compliance **40 %** | **86.7 %** — every non-call was `finish_reason: length` |
| `max_tokens 3072` | `AD-IQ1_M` accepted **20/30**, failing five tasks the others pass | **27/30** at 8192; the `NameError`s were truncated code |
| `max_tokens 3072` | Bonsai accepted **15/30** | **27/30** at 8192; 35 of 60 attempts had been truncated |
| `n_predict 400` | Ornith-9B "failed the rename task" | cut off mid-class; its `<think>` block used the budget |

Median reasoning per call spans **59 characters (Q4) to 2,811 (Q2_K_XL)** across
quantizations *of the same model* — and one V3 artifact reached **37,000**. Any
budget chosen for the control truncates the others, and the artifact that reasons
longest always looks like the weakest one.

**The rules that follow:**

- Budget for the most verbose arm, never the control.
- Record `finish_reason` on every call and report the truncation count as part
  of the result, not a footnote.
- Treat a length-truncated attempt as **censored, not failed**: its outcome is
  unknown, and scoring it as a failure penalises exactly the artifacts that
  reason longest. `retry_economics` now flags when a censored task could flip
  the verdict.
- Report **capability and throughput apart**. `accepted_of_decided` is the first;
  `wall_per_accepted_s` is the second; their product is not a ranking.

### Two gates written as absolutes, and why both were wrong

Both were caught before they rejected anything, and the pattern is worth naming:
**a threshold with no control is a guess wearing a number.**

- *"Required tool/schema compliance 100 %"*, from the research. The production
  `Q4_K_XL` control scores **80 %** on that probe at temperature 0.7. An absolute
  gate would have rejected the model it was meant to protect.
- *"`shared` GPU memory must be 0"*, written by this project when adding
  `residency_check.py`. The production artifact measures **98 MiB against 9,417
  dedicated — 1.04 %** — which is ordinary pinned staging, not eviction. The
  informative quantity is the **ratio across arms with different headroom**: it
  climbing as free VRAM falls is eviction; sitting flat near 1 % is staging.

### A cheap predictor found by accident

The greedy-decode hash, added as an invariance check for flag sweeps, turned out
to predict a four-minute quality gate in thirty seconds. Across five V3 arms, the
three returning `227749403A7404D4` — byte-identical to the pre-V3 generation and
to Q4 on the same mechanical rename — were exactly the three with no quality
problem, and the two that diverged were exactly the two the screen caught.

Five arms is a mechanism worth trusting, not a validated rule. But the mechanism
is sound: a differing hash on a task with one correct answer means the weights
moved enough to change behaviour.

**What it does not do:** an identical hash across *different models* proves
nothing. `Q4_K_XL`, `Q2_K_XL`, `UD-IQ2_XXS` and three V3 arms all return the same
hash on that prompt because a mechanical rename has one right answer. The hash is
a divergence detector, not an equivalence proof.

---

## 8. Instrument failures found on 2026-08-21 — the two flat constants

Both are the same mistake in two places: **a hard-coded wait that does not know
what it is waiting for.** Neither returned a wrong number, which is why they had
survived every sweep so far. They did something the earlier faults did not —
**one failing arm destroyed a step that had nothing to do with it.**

| failure | symptom | what it cost |
|---|---|---|
| `post(path, payload, timeout=3600)` — a flat hour, whatever the depth | `F-ot-iq1m-128k` arm `ot-ffn-1` pushed 644 MiB of FFN weights to CPU with `-ot`; prefill collapsed 240.6 → **8.56 tok/s**, so its 93,086 tokens needed ~10,900 s and could never finish | the harness sat on an obvious failure from **01:34:36 to 02:34:36 — one hour to the second** — then raised `TimeoutError`. Fifty-nine of those sixty minutes bought nothing |
| `kill()` ending in `time.sleep(5)` | WDDM releases a 12 GB allocation in stages; the next server started 5 s after the kill, **passed `/health`**, then died on its first `/completion` | `F-place-cpu-rest` failed on its baseline arm with `ConnectionResetError` and produced **zero rows**. It is the only step of the night with no data at all. **Measured after the fix, 03:15:** an 11,501 MiB teardown took **9.87 s** to return 9,924 MiB, so the 5 s sleep handed the next arm a GPU that was still about half full |

### The chain, and why the second fault is the dangerous one

The teardown `p.kill(); fh.close()` sat at the **end of the happy path** in
`depth_sweep.run()`, with no `try/finally`. So the timeout above did not merely
lose its own arm — it skipped the teardown, left a 12 GB server resident, and
handed the next queue step a GPU that was still full thirty seconds later.

That second step is the real lesson. **`ConnectionResetError` was the lucky
outcome.** The new server passed its health check; had the machine happened to
have a few hundred MiB more free — a different desktop state, one browser window
fewer, any point on the 13.6 % boot-to-boot drift — it would have loaded, run,
and written a row of plausible numbers measured against a GPU it was sharing.
That is indistinguishable in the JSONL from a clean measurement, and it is the
same failure that destroyed a 30-task corpus at 02:00 on 08-20.

### The fix, and the principle behind it

Three changes, all in `bench/`, all covered by tests named after this incident
(`test_harness.py`, suite now 92):

- **`harness.completion_timeout_s(ctx)`** sizes the budget from the prefill the
  depth actually implies: `ctx × 0.8 / 60 tok/s + 300 s`. The floor of 60 sits
  four times below the slowest **legitimate** prefill ever measured at 131,072
  (240.6 tok/s) and seven times above the **pathological** one (8.56), so it can
  neither truncate a real measurement nor sit an hour on a dead arm. At 131,072
  the budget is 2,048 s, not 3,600.
- **`harness.vram_settled(readings, floor_mib=...)`** replaces the sleep: poll
  free VRAM until two consecutive readings agree within 64 MiB **and the
  latest one has cleared a floor**. A desktop compositor moves tens of MiB
  between polls; a model unloading moves thousands, so the tolerance separates
  them with two orders of magnitude to spare. One reading is never settled --
  that is the 5 s sleep restated.

  **The floor is not decoration, and the first version of this fix did not**
  **have it.** "Stopped moving" is ambiguous between *release finished* and
  *release has not begun*: two polls taken 3 s apart before the driver does
  anything agree perfectly, and the caller concludes the GPU is free -- which
  is the 5 s sleep with extra steps. `kill()` therefore reads free VRAM
  **before** stopping the process and demands the reading beat it by 1,024
  MiB, a tenth of the smallest artifact this project loads (7.80 GiB): a real
  teardown clears that easily and a release that never started cannot. When
  nothing was listening on 8080, `kill()` skips the wait entirely rather than
  sitting out its limit.

- **`try/finally` around the whole measurement**, and a timeout is now recorded
  as a row (`note="abandoned after 2048s: TimeoutError: …"`) rather than raised.
  **An arm too slow to finish inside its depth budget is a result.** `-ot` on
  `AD-IQ1_M` is exactly that result, and a traceback that kills the queue is not
  a better way to report it.

The principle: **a constant that does not scale with the thing it bounds is not
a bound, it is a coincidence.** `3600` was never chosen against a measured
prefill rate, and `5` was never chosen against a measured release time. Both
held for weeks because no arm had yet been slow enough to test them.
