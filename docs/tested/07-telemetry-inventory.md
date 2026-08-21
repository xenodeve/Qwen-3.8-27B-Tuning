# 07 — Every value a run can yield, and where it comes from

Written 2026-08-21 to answer one question directly: **what can we actually
measure, so that model tuning is driven by data rather than by whichever number
happened to be printed?**

Four sources. None of them is complete on its own, and the gaps are the reason
several conclusions in this project had to be withdrawn.

| source | gives | cannot give |
|---|---|---|
| **llama-server log** (`-lv N`, stderr) | prefill and decode rate per request, cache hits, slot reuse, layer split, buffer sizes | which task a request belonged to; the request body |
| **`/slots`** | sampling params, context state, `n_prompt_tokens_cache` | anything about a request that already finished |
| **`/props`** | model metadata, default sampling, build info | anything per request |
| **`--metrics`** (needs restart) | cumulative counters | attribution to a single request |
| **`bench/tap.py`** | the request body, the response, and llama's own `timings`, joined and labelled per task | GPU-side numbers |
| **the harness** | accepted / failed, wall time, files written, tool-call transcript | tokens |

---

## 1. Per request — `timings`, the richest object we get

llama-server returns this on every completion. `bench/tap.py` captures it; the
corpus harness does not, which is why it was invisible until now.

| field | what it is |
|---|---|
| `prompt_n` | tokens actually prefilled — **after** cache reuse, so this is the marginal cost of the turn |
| `prompt_ms`, `prompt_per_second` | prefill time and rate |
| `predicted_n` | tokens generated |
| `predicted_ms`, `predicted_per_second` | **decode rate — the number this project optimises** |
| `draft_n` | tokens the speculative decoder proposed |
| `draft_n_accepted` | how many the target model agreed with |
| `cache_n` | tokens served from the prompt cache |

**`draft_n_accepted / draft_n` is the single most useful ratio we have** and it
was measured wrongly for a day: the sweep computed it from the **first of five**
generations while reporting `tg_med` as the median of all five
(`CORRECTIONS.md` §11). On real work through OpenCode it has never been measured
at all — the server was started at `-lv 3`, which does not print the draft
lines, so the run in progress can report decode rate but not acceptance.

---

## 2. Per request — what the harness sends

Only `bench/tap.py` sees this, and it matters more than expected:

- **`chars_by_role`** — how big the system prefix is versus the actual task.
  Measured on OpenCode: **5,377 tokens of prefix** against a task of a few
  hundred, and 99,073 before the lean profile.
- **`n_tools`, `tools_bytes`** — 141 tools and 265 KB of schema in the default
  OpenCode config; 6 tools and 13 KB in the lean one.
- **`sampling`** — and this is a live problem. OpenCode sends **no sampling
  parameters at all**, so the server's defaults apply: `temperature 1.0`,
  `top_k 20`, `top_p 0.95`, `min_p 0.05`. Every quality number in
  `retry-bench.jsonl` was measured at `temperature 0.7` or greedy. **The two
  are not comparable until this is stated on both.**
- **`tool_choice`, `finish_reason`, `tool_call_deltas`** — whether a turn ended
  because the model was done, hit a stop, or called a tool.
- **`ttfb_s`** — time to first byte, which separates "the model is thinking"
  from "the harness is slow".

---

## 3. Cumulative — `--metrics`

**Not currently enabled.** `curl /metrics` returns 501 with
*"Start it with `--metrics`"*. It costs a restart and gives Prometheus counters:
tokens processed and generated, requests, KV cache utilisation, and queue depth
over the process lifetime.

Useful for a long unattended run where per-request rows would be unwieldy.
Useless for attribution — it cannot say which task cost what.

---

## 4. What the server log adds that nothing else does

At `-lv 5`, which the sweeps use:

- `load_tensors: offloaded N/M layers to GPU` — **the layer split**, the single
  most predictive number in this project
- `CUDA0 model buffer size`, `KV buffer size`, `RS buffer size` — where the VRAM
  went
- `slot get_availabl: ... selected slot by LCP similarity, f_sim_best = 0.996`
  — **prompt-cache hit quality.** This is how we confirmed OpenCode's fixed
  prefix is reused across turns: after the first request, later turns prefill
  only 19–49 tokens.
- `graphs reused = 1334` — CUDA graph reuse, a proxy for how stable the shape
  of the work is
- `n_ctx_train`, `rope scaling`, `n_ctx_seq` — **262,144 native**, so depth is
  bounded by VRAM alone

At `-lv 3`, which the OpenCode run uses, the draft lines are absent. **That is
the one thing the current run cannot report.**

---

## 5. What only the harness knows

- **accepted / failed**, from executing the produced code against hidden
  assertions — the only quality signal that is not a proxy
- **the failure mode**, which changed completely with the harness: under
  `run_retry_bench.py` the dominant failure was *no fenced code block*; under
  OpenCode the first observed failure was `AssertionError` — **code that ran and
  was wrong.** Those need different fixes and only the harness can tell them
  apart
- **wall time per task**, which is what `tasks_per_hour` is built from
- **files written**, including files that should not have been

---

## 6. What we still cannot measure

- **GPU-side anything per request** — clock, power, memory bandwidth. `nvidia-smi`
  samples the device, not the request, and the two cannot be joined without
  timestamps finer than we collect.
- **Where the model spent its reasoning.** `reasoning_chars` exists in the
  protocol probes but not in the corpus path.
- **Whether a drafted-and-rejected token cost the same as a normal one.** The
  acceptance ratio says how many were thrown away, not what throwing them away
  cost.

---

## 7. To collect everything on the next run

Three changes, none of which is in the run currently in progress:

1. **Restart the server with `--metrics` and `-lv 5`.** Buys draft/acceptance
   lines and cumulative counters. Costs a reload.
2. **Put `bench/tap.py` between OpenCode and the server** and point the harness
   at the tap's port. Buys the request body, per-request `timings`, and the
   task label joined to both.
3. **Have the harness `POST /_tap/mark` before each task** so every row carries
   the task id.

Only then is a row self-describing enough that a later reader can tell what was
measured without reconstructing it from four files and a memory.
