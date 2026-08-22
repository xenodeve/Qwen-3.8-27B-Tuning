# Qwen3.8-27B Local Worker — Experiment Log

> 🔴 **HISTORICAL — the E0–E13 programme, superseded.** Correct as the
> record of what was planned and run at the time. The live queue is
> [`docs/OPEN-WORK-LEDGER.md`](../docs/OPEN-WORK-LEDGER.md).

Every entry: **hypothesis → exact command → environment → result → interpretation → next**.
One major variable per experiment. A result without its `env-snapshots.jsonl` line is not evidence.

Primary metric is **verified successful coding tasks per unit time**, not raw tok/s.

---

## E0 — Verify b10472 flag names

**Hypothesis:** the flags named in the handoff doc (§9, §20) still exist under this build.

**Command**
```powershell
C:\AI\llama.cpp-cuda\llama-server.exe --help
C:\AI\llama.cpp-cuda\llama-bench.exe  --help
```

**Environment:** build 10472, commit 60eeeb608, CUDA 12.4.

**Result:** all present, none renamed.

| flag | status | detail |
|---|---|---|
| `-nkvo, --no-kv-offload` | ✅ | also `-kvo, --kv-offload` |
| `-ctk/-ctv, --cache-type-k/-v` | ✅ | `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1` — default `f16` |
| `-ngl, --n-gpu-layers` | ✅ | `N \| auto \| all`, **default already `auto`** |
| `-fit, --fit` | ✅ | `[on\|off]` |
| `-fa, --flash-attn` | ✅ | `[on\|off\|auto]`, default `auto` |
| `-c, --ctx-size` | ✅ | default 0 = from model |
| `-b / -ub` | ✅ | default 2048 / 512 |
| `-dev, --device` | ✅ | |
| `--jinja` | ✅ | **default enabled** — passing it is a no-op |
| `-np, --parallel` | ✅ | default -1 = auto |
| llama-bench `-d, --n-depth` | ✅ | plus `-p -n -pg -ctk -ctv -o csv\|json\|jsonl\|md\|sql` |

**Interpretation:** the doc's planned knobs are all reachable; `q8_0` KV (§10) is
supported, so Phase E needs no substitute. The §20 baseline command is valid as written.

**Correction to handoff [C2]:** free VRAM is **not** a fixed 11069 MiB.
`nvidia-smi` measured 9361 MiB free (2637 MiB already in use) minutes later.
Since `--fit on` derives the layer split from free VRAM at boot, this is a
confounder for Phases A/C/F. Mitigation: `scripts/collect-env.ps1` snapshots
VRAM/RAM before every launch and appends to `results/env-snapshots.jsonl`.
Ollama (pid on :11434) should be stopped before controlled runs.

**Next:** E1.

---

## E1 — Phase A: UD-Q4_K_XL boot at 16K  *(DONE — baseline established)*

**Hypothesis:** UD-Q4_K_XL loads on this machine at `-c 16384` with `--fit on`
producing a partial GPU offload, and serves basic completions. Expected: model
~16–17 GB vs ~9–11 GB free VRAM, so a meaningful share of layers stays CPU-resident
and generation is bounded by CPU/PCIe, not GPU compute.

**Command**
```powershell
C:\AI\qwen38-tuning\scripts\start-q4.ps1
```

**Environment:** build b10472-60eeeb608, CUDA 12.4, driver 610.88, 47.69 GB RAM.
Desktop was live during the run (Wallpaper Engine, Discord, Comet, Edge WebView,
NVIDIA Overlay all held VRAM). Ollama was left running. **This is a dirty baseline
by choice — it is the machine's real working state, not an isolated one.**

**Result — PASS, model serves.**

| measurement | value |
|---|---|
| model on disk | **16.69 GB** (`Q4_K - Small` ftype label; Unsloth dynamic is mixed) |
| load time | ~13 s after download (47.32 → 47.46 in server log) |
| VRAM after load | **11719 / 12282 MiB used → only 279 MiB free** |
| llama-server working set | 16.67 GB (incl. mmap) / private 11.41 GB |
| host RAM free | 11.35 / 47.69 GB |
| layers | 64 dense + `blk.64` = nextn/MTP head, reported unused → **speculative-decode head is in the GGUF but not active** |
| n_ctx / slots | 16384 / 1, `kv_unified=false` |
| **prompt processing** | **518.8 tok/s** @ prompt_n=4601 |
| **generation** | **6.8 – 7.6 tok/s** (n_predict 32 and 128) |

Short-prompt pp is not a valid number: an 11-token prompt measured 13.7 tok/s
because fixed per-request overhead dominates. Only the 4601-token figure is real.

**Interpretation**

1. `--fit on` did its job aggressively — it consumed VRAM down to 279 MiB free.
   That is *efficient* but *fragile*: any desktop app that grabs VRAM afterwards
   (browser tab, game overlay, Wallpaper Engine) can push allocation into
   driver-level eviction and silently tank speed, or OOM on the next resize.
   Headroom is a variable worth its own experiment, not a free win.

2. **pp 519 tok/s is the number that decides whether 256K (§11) is viable.**
   Straight-line projection, ignoring depth degradation:
   ```
    16K prefill  ~   32 s
    64K prefill  ~  126 s   (2.1 min)
   128K prefill  ~  253 s   (4.2 min)
   256K prefill  ~  505 s   (8.4 min)
   ```
   Prompt caching makes incremental turns cheap, but every cache miss, branch, or
   restart pays this in full. Depth degradation will make the real 256K number worse.
   Must be measured with `llama-bench -d`, not extrapolated (§24 says exactly this).

3. **tg ~7 tok/s is the throughput ceiling** and it interacts badly with finding 4.

4. **NEW — not covered anywhere in the handoff doc.** The Qwen3.8 chat template
   defaults `reasoning_effort` to **`xhigh`** when the caller does not set it:
   ```jinja
   {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
   ```
   and `'high'` is silently remapped to `'xhigh'`. Supported values are only
   `xhigh | medium | low` — anything else raises. At 7 tok/s a multi-thousand-token
   thinking block costs minutes *per agent step*. Since the primary metric is
   verified tasks per hour, **reasoning effort is a first-class tuning variable
   here, arguably ahead of KV placement.** It costs nothing to test.

5. Sampling defaults arriving from the model (`temp 1.0, top_k 20, top_p 0.95,
   min_p 0.05`) already match the benchmark methodology quoted in §15. Do not override.

6. `chat_template_caps` confirms the Phase B prerequisites are all present:
   `supports_tool_calls`, `supports_parallel_tool_calls`, `supports_object_arguments`
   (= nested JSON args), `supports_system_role`, `supports_reasoning_effort`,
   `supports_preserve_reasoning`. Developer-role messages are merged into system
   by the template.

7. Tool-call wire format is **not** JSON — it is XML-ish:
   `<tool_call><function=NAME><parameter=ARG>value</parameter></function></tool_call>`.
   llama.cpp must parse that back into OpenAI `tool_calls`. This is the single
   highest-risk integration point for OpenCode and is exactly what Phase B must prove.

8. `reasoning_format: none` in the default settings means thinking is **not**
   split out of `content`. The server log itself hinted: *"chat template supports
   preserving reasoning, consider enabling it via --reasoning-preserve"*.
   If `<think>` blocks leak into `content`, OpenCode will treat reasoning as answer text.

9. Non-issue, resolved: `failed to create symlink: A required privilege is not held`
   → *degraded mode*. It did **not** duplicate the file. `blobs/` is empty and the
   16.69 GB GGUF sits directly in `snapshots/`. No wasted disk, no action needed.
   (Cause: Windows symlink creation needs Developer Mode or admin.)

**Next:** E1b, then E2.

---

## E1b — Clean-ish baseline (Ollama stopped, desktop left running)

**Hypothesis:** stopping Ollama frees VRAM, `--fit on` puts more layers on GPU,
generation speed rises.

**Method:** stopped `ollama` + `ollama app`, restarted llama-server via
`scripts/start-q4.ps1`. Wallpaper Engine / Discord / Comet deliberately left
running — the target is the machine's real working state, not a lab condition.

**Result — hypothesis REJECTED.**

| | dirty (Ollama up) | clean (Ollama down) |
|---|---|---|
| VRAM freed by stopping Ollama | — | **58 MiB** |
| VRAM used after load | 11719 MiB | 11493 MiB |
| VRAM free after load | 279 MiB | 505 MiB |
| load time | ~13 s | ~12 s |
| **generation** | 6.81 / 7.56 tok/s | **6.29 tok/s** |

`GET /api/ps` showed Ollama had **no model loaded**, so it was holding ~nothing.
Generation came out *slower* on the clean run, which is run-to-run noise, not a
regression.

**Interpretation:** two things follow, and the second is the important one.

1. Stopping Ollama is worth doing for **risk** (it cannot wake and grab VRAM
   mid-experiment), not for capacity. Do not credit it with any speed gain.
2. **Single-shot tok/s cannot resolve differences of this size.** The spread
   across three samples of an unchanged configuration was 6.29–7.56 tok/s, ~18%.
   Any later A/B whose effect is smaller than that is unmeasurable with one run.
   From here, every speed claim needs N≥3 and a reported spread, or `llama-bench`,
   which repeats internally.

**Tooling defects found and fixed (both mine, both the same root cause):**
`llama-server --version`, `nvidia-smi`, and `llama-server` itself all write normal
output to **stderr**. Under Windows PowerShell 5.1 with `$ErrorActionPreference='Stop'`,
the first such line is raised as a terminating `NativeCommandError`. This killed
`collect-env.ps1` at the version probe, and then killed `start-q4.ps1` at the launch
line before the port was ever bound — the server appeared to "fail to start" while
nothing was actually wrong with llama.cpp. Both scripts now drop to `'Continue'`
around native calls.

---

## E1c — Vendor documentation review (user-supplied Unsloth + AtomicChat charts)

**Source:** Unsloth Qwen3.8 page [S2] and an AtomicChat quant-fidelity chart, both
provided as screenshots.

**What they confirm**

- The §15 benchmark transcription in the handoff doc is **accurate** — 73.0 / 61.7 /
  42.3 / 42.2 / 79.0 / 70.7 match the published table exactly. That column of the
  doc can be trusted.
- 27B dense, 64 layers, 256K context — matches the `blk.0..63` + `blk.64` nextn
  head observed at load.
- Developer Role and improved nested-object tool parsing are advertised features,
  consistent with the `chat_template_caps` read in E1.

**What they CORRECT**

1. **`min_p` is wrong in our current setup.** Unsloth's recommended settings:

   | param | thinking | non-thinking |
   |---|---|---|
   | temperature | 1.0 | 0.7 |
   | top_p | 0.95 | 0.80 |
   | top_k | 20 | 20 |
   | **min_p** | **0.0** | **0.0** |
   | presence_penalty | 0.0 | 1.5 |

   The server reports `min_p = 0.05`. My E1 note said the defaults already matched
   the recommended sampling and should not be overridden — **that was wrong on
   `min_p`**, and wrong on the existence of a separate non-thinking profile at all.
   Thinking and non-thinking need *different* sampling; a single default cannot
   serve both.

2. **The model has vision.** We launched with `--no-mmproj-auto`, so this instance
   is text-only. Correct for a coding worker, but it is now a *recorded choice*
   rather than an accident, and it means published numbers that exercise vision
   do not describe this instance.

**Fidelity priors (use as priors only — neither chart measures coding-agent success)**

Unsloth, top-1 agreement vs BF16: `UD-Q2_K_XL ~85.5% · UD-IQ3_XXS ~90% ·
UD-Q3_K_XL ~92.4% · UD-Q4_K_XL ~96% · UD-Q5_K_XL ~97% · Q8_0 ~98.5%`.
The Q3→Q4 step (~3.6 pp) is much larger than Q4→Q5 (~1 pp), which is the shape
the handoff doc's §4 State-4 reasoning relied on. That reasoning holds up.

AtomicChat, mean KL vs BF16 at 4096 ctx on BF16 reference: the AD line sits at or
slightly below the best non-AtomicChat file across most sizes, with the clearest
gap in the ~16–20 GB band. **Caveat:** the two charts use different metrics
(top-1 agreement vs mean KL), different eval sets, and different context lengths,
so numbers cannot be carried between them. Within the AtomicChat chart alone the
comparison is self-consistent; across charts it is not.

**Interpretation — this changes the phase ordering.**

The handoff doc (§25) sequences Q3-vs-Q4 last, as Phase H, after KV placement, KV
precision, `-ngl` tuning and batch tuning. On *this* machine that ordering is wrong,
and the E1 measurements say why:

```
model            16.69 GB
VRAM available   ~10.2 GB free before load
=> roughly 40% of weights are CPU-resident
=> generation is 6.3-7.6 tok/s, bound by that CPU fraction
```

Quant size is not only a fidelity knob here — it is **the dominant speed knob**,
because every GB removed moves ~3-4 more of the 64 layers onto the GPU. A Q3-class
file (~12.5-14 GB) would cut the CPU-resident share roughly in half. No amount of
KV placement or batch tuning can move tg the way that can.

Against that, Q3 costs ~3.6 pp of top-1 agreement, and §26's decision logic says
that only matters if it converts into verification failures.

So the two candidates should be compared **early and directly**, on the real
metric, instead of being reached last. Proposed reordering:

```
doc order:  A → B → C → D(KV place) → E(KV prec) → F(ngl) → G(batch) → H(Q3vQ4)
proposed:   A → B → H(Q3 vs Q4) → C(context) → D → E → F → G
```

with `reasoning_effort` (E1 finding 4) folded into B, since at 6-7 tok/s the
default `xhigh` thinking block plausibly costs more wall-clock per agent step
than every other variable on this list combined.

**Next:** E2 = Phase B tool-calling gate + `reasoning_effort` sweep + `min_p 0.0`
correction. Tool calling is still the hard gate — nothing downstream matters if
OpenCode cannot drive the model.

---

## E2 — Phase B: protocol correctness gate  *(PASS — all 8 items)*

**Hypothesis:** the Qwen3.8 XML tool syntax round-trips through llama.cpp's parser
into OpenAI-compatible `tool_calls`, and an OpenCode-style action/observation loop
can be driven end to end.

**Environment:** b10472, UD-Q4_K_XL, `-c 16384`, MTP off. All requests sent with
`min_p = 0.0` and `chat_template_kwargs.reasoning_effort = 'low'` to keep gate
latency bounded.

| # | gate item | result |
|---|---|---|
| 1 | plain completion | PASS |
| 2 | developer-role behaviour | PASS — accepted, merged into system by template |
| 3 | simple tool call | PASS — `finish_reason: tool_calls`, `{"city":"Bangkok"}` |
| 4 | nested object arguments | PASS — 3-level nesting + array, valid JSON |
| 5 | tool result → continuation | PASS |
| 6 | repeated tool loop | PASS — 2 tool rounds then coherent synthesis |
| 7 | reasoning separation | PASS — see correction below |
| 8 | `min_p = 0.0` | applied per-request |

**Correction to E1 finding 8 (and to §5.2 of HANDOFF-BACK).** I reported that
`reasoning_format: none` meant `<think>` would leak into `content` and that this was
an open risk for OpenCode. **That was wrong.** `/v1/chat/completions` returns:

```
message keys      : role, content, reasoning_content
content           : "HELLO"                     <- clean
reasoning_content : "The user wants me to ..."  <- separated
```

The `reasoning_format` value I read lives in `default_generation_settings`, which
governs the raw `/completion` endpoint, not the OpenAI chat endpoint. The reasoning
contract is unambiguous and no `--reasoning-format` / `--reasoning-preserve` flag is
needed for the basic case. Item 7 is closed.

**Nested-argument detail (test 4).** Requested `notify: true`; the emitted arguments
were:

```json
{"title":"Fix login bug","assignee":{"name":"Alice","email":"alice@example.com"},
 "labels":["bug","urgent"],"config":{"priority":1,"nested":{"retries":3}}}
```

`notify` was dropped. `notify` was not in `required`, so this is **not** a schema or
parser failure — it is the model omitting an instructed field. n=1, at
`reasoning_effort=low`. Not a conclusion; logged because instruction-drop on tool
arguments is a failure mode that would be invisible to a schema validator and would
show up downstream as silently wrong agent behaviour.

**Agent-loop transcript (tests 5–6), abbreviated:**

```
round 1  finish=tool_calls  read_file {"path":"src/auth.py"}
round 2  finish=tool_calls  run_tests {"pattern":"test_auth"}
round 3  finish=stop        correct diagnosis of the injected bug
```

`tool_call_id` round-tripped correctly across both rounds.

**Interpretation:** the highest-risk integration point named in §8 of the continuation
plan is **clear**. The XML-template → OpenAI-`tool_calls` conversion works, including
nested objects, arrays, multi-round loops and id correlation. OpenCode has no known
protocol blocker at this layer. Remaining tool-calling risk moves from *"can it
parse"* to *"does it keep doing it correctly over long sessions"*, which only the
Phase-K workload can answer.

**Next:** B2 reasoning-effort sweep (running), then Q4 MTP smoke test.

---

## E3 — MTP flag verification against local b10472  *(CONFIRMED)*

Per §4 of the continuation plan, checked the local binary rather than master docs.

```
--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,
            ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache
--spec-draft-n-max N   (default: 3)
--spec-draft-n-min N
--spec-draft-p-min P   (default: 0.00, greedy)
--spec-draft-p-split P (default: 0.10)
--spec-draft-ngl / -devd / -ctkd / -ctvd / -otd
--spec-draft-backend-sampling            (default: enabled)
--spec-default                           enable default speculative config
```

Both flags the continuation plan specifies (`--spec-type draft-mtp`,
`--spec-draft-n-max`) exist here with the stated default of 3. `--draft` /
`--draft-max` are **removed** in this build and error out pointing at the new names —
worth knowing, since most tutorials still use the old spelling.

**New, not in the continuation plan:** `--spec-type` takes a **comma-separated list**,
and the `ngram-*` family performs speculation from the prompt/context with **no draft
model and effectively no extra VRAM**.

That matters here specifically. §2.5 of the continuation plan flags an unverified
~2.5 GB MTP VRAM overhead, against a Q4 baseline with **505 MiB free**. If that
overhead is real, `draft-mtp` on Q4 may be unusable while `ngram-*` still works,
because it costs no weights. Coding workloads are also the favourable case for
ngram speculation — the context is full of near-repeated code the decoder is about
to reproduce.

**Proposed addition to the Phase-C matrix:** carry `ngram-mod` (or `ngram-simple`)
as a third arm alongside MTP-off / MTP-on, at least for the Q4 lane where VRAM
headroom is the binding constraint. It is cheap to test and it is the only
speculative option that cannot lose to VRAM pressure.

---

## E4 — B2 reasoning-effort sweep  *(INCONCLUSIVE — and the reason matters)*

**Hypothesis (mine, from E1 finding 5.1):** `reasoning_effort=xhigh` is the single
largest cost on this hardware; dropping to `low`/`medium` should cut wall-clock
per agent task substantially.

**Method:** one fixed agentic task (fix an LRU eviction bug via `read_file` +
`apply_patch`), vendor thinking sampling with `min_p` corrected to 0.0, MTP off,
2 runs per level.

| effort | wall_s | completion tok | reasoning chars | rounds | bad args | reached patch |
|---|---|---|---|---|---|---|
| low | 67.8 / 85.3 | 453 / 608 | 384 / 570 | 3 / 4 | 0 | yes / yes |
| medium | 85.6 / 50.1 | 610 / 352 | 621 / 212 | 4 / 3 | 0 | yes / yes |
| xhigh | 84.2 / 106.9 | 588 / 741 | 632 / 1008 | 4 / 4 | 0 | yes / yes |

**Result: the hypothesis is not supported, and the experiment cannot resolve it.**

Measured reasoning was 212–1008 characters, roughly 50–250 tokens — not the
"multi-thousand-token thinking block" I asserted in E1. The cost claim was
overstated.

Worse, the design was inadequate: within-level spread (`medium` spanned 50.1–85.6 s,
71%) swamps between-level difference, and **I ran n=2 after proposing an N≥3 rule
in E1b.** The fastest single run of the whole experiment was `medium`, and `low`
vs `medium` are indistinguishable.

The task was also too easy — all 6 runs reached a correct patch with zero
malformed tool arguments. A task every arm solves cannot discriminate quality.

**What the data does support, and it is more useful than the original question:**

```
3 rounds -> 67.8 s, 50.1 s          ~17-23 s per round
4 rounds -> 85.3, 85.6, 84.2, 106.9  ~21-27 s per round
```

Cost per agent round is near-constant; **total wall-clock is driven by the number
of tool rounds, not by reasoning verbosity.** For a verified-tasks-per-hour metric
the lever is fewer agent steps, not shorter thinking.

**Decision:** adopt `medium` as the operational profile provisionally (matching the
plan's own hypothesis), do not spend more runs here now, and revisit with a harder
task and N≥5 during the real-workload phase. Effect size here is below the noise
floor; MTP's is not.

---

## E5 — Speculative decoding sweep, Q4  *(MTP CONFIRMED; ngram-mod null on this test)*

7 configs, N=3 each, server restarted between every config.

| spec | n_max | tg min–max | VRAM free | acceptance | greedy SHA-256 |
|---|---|---|---|---|---|
| off | — | 7.71 – 8.27 | 338 MiB | — | `A2F070D5480ADEE4` |
| draft-mtp | 2 | **9.88 – 10.75** | 772 | **77.5 %** | `A2F070D5480ADEE4` |
| draft-mtp | 3 | 9.41 – **11.23** | 942 | 68.6 % | `A2F070D5480ADEE4` |
| draft-mtp | 4 | 9.37 – 11.08 | 1155 | 65.2 % | `A2F070D5480ADEE4` |
| draft-mtp | 5 | 8.59 – 9.38 | 1089 | 56.4 % | `A2F070D5480ADEE4` |
| draft-mtp | 6 | 7.18 – 10.45 | 1275 | 52.4 % | `A2F070D5480ADEE4` |
| ngram-mod | 4 | 8.03 – 8.34 | 365 | 20.8 % | `A2F070D5480ADEE4` |

**Harness defect, disclosed:** the field written as `tg_median` is actually the
**maximum** of the three runs. `[int](3/2)` is 2 in PowerShell, not 1 — `[int]`
rounds half to even, so the index landed on the last element of the sorted array.
Fixed with `[math]::Floor`. The table above is relabelled to min–max, and the
conclusions below rest on non-overlapping ranges, not on any point estimate.

**1. Output equivalence: PASS.** All seven configurations produced a
byte-identical greedy sample (`temperature 0, top_k 1, seed 42`). The divergence
reported in llama.cpp issue #25618 for quantized targets under speculative paths
**does not reproduce here**. Protocol §5 satisfied; plan §6's
`repeatable output corruption` stop condition is not triggered.

**2. MTP is real, but smaller than I first reported.** My earlier figure of 1.56×
compared runs from *different boots* with different `--fit` layer splits — the very
confounder flagged in E1b. Within this single controlled sweep it is **≈1.36×**
(max/max) to ≈1.19× (min/min). Ranges for n=2/3/4 do not overlap the baseline at all.

**3. The video's "4–5 draft-step sweet spot" does not reproduce.** Peak here is
**n=2–3**; n=5 and n=6 regress, and n=6's range (7.18–10.45) overlaps the baseline.

**4. Mechanism for the regression is visible in VRAM.** Free VRAM *rises* with
n_max (338 → 772 → 942 → 1155 → 1275 MiB), meaning `--fit` reserves progressively
more for the draft path and therefore **moves target layers off the GPU**. High
n_max is penalised twice: falling acceptance *and* falling target residency.

**5. Recommend n=2 over n=3 for production.** n=3 has the higher ceiling (11.23)
but n=2 has the higher floor (9.88 vs 9.41), the tighter range, and the best
acceptance (77.5 %). For a tasks-per-hour metric, consistency beats peak.

**6. `ngram-mod` scored no gain — but the test was unfair to it.** 20.8 %
acceptance, range fully overlapping the baseline. The benchmark prompt was 11
tokens with nothing in context to match, while research doc 04 §7 names
*source-code rewriting* as the intended use case. Re-tested properly in E6 with a
long prompt containing the source about to be rewritten, and with `ngram-simple`,
which is the variant the protocol actually specifies.

---

## E6 — Speculation matrix, both quants (protocol §6)

Identical procedure for each quant: server restarted per config, environment
snapshotted before load, N=3 generations of 160 tokens, greedy equivalence sample
captured. Two prompt types, because E5 showed the prompt decides the answer:

- **bench** — a 11-token instruction. Nothing in context to copy. This is the
  worst case for any speculative method that predicts from history.
- **code** — a long prompt containing the exact class the model is asked to
  rewrite with one attribute renamed. This is the case llama.cpp's own docs cite
  for `ngram-simple`, and it is what a coding agent actually does most of the
  time: reproduce existing code with a small change.

| quant | spec | n_max | bench tok/s | code tok/s | acceptance (bench / code) | VRAM used |
|---|---|---|---|---|---|---|
| Q4 | none | — | 8.24 | 8.22 | — | 11832 |
| Q4 | ngram-simple | 4 | 8.29 | 8.37 | — / 30.8 % | 11773 |
| **Q4** | **draft-mtp** | **2** | **10.67** | **12.10** | **78.1 / 98.0 %** | 11394 |
| Q4 | draft-mtp | 3 | 9.91 | 12.03 | 70.3 / 88.8 % | 11248 |
| Q3 | none | — | 9.01 | 9.25 | — | 11527 |
| Q3 | ngram-simple | 4 | 9.16 | 9.08 | — / 30.8 % | 11493 |
| **Q3** | **draft-mtp** | **2** | 8.88 | **10.30** | 77.5 / 96.4 % | 11275 |
| Q3 | draft-mtp | 3 | 7.27 | 9.92 | 64.1 / 99.1 % | 10935 |

Greedy samples were byte-identical across every config **within** each quant
(`6F8AAC2789…` for Q4, `0659173109…` for Q3), so speculation remains a pure
performance toggle on this stack for both quants. Issue #25618 still does not
reproduce.

### Finding 1 — prompt type changes the MTP result more than any tuning knob

Q4 + MTP n=2 reaches **98 % acceptance and 12.10 tok/s** on the code-rewrite
prompt versus 78.1 % and 10.67 on the synthetic one. Against the same-prompt
baseline that is **+47 %**, not the +36 % E5 measured on the synthetic prompt.

The synthetic prompt therefore *understates* what an agent workload will see.
Any future benchmark that measures speculation with a short instruction prompt
is measuring the wrong thing.

### Finding 2 — `ngram-simple` is not competitive here, and the fair test says so

Re-tested on the prompt type its own documentation names, with the variant the
protocol specifies: 30.8 % acceptance on Q4 and Q3 alike, converting to
8.37 vs 8.22 tok/s — inside run-to-run noise. On the short prompt it drafts
nothing at all.

E5's dismissal of ngram was based on an unfair test; this one is fair and reaches
the same verdict. Recommend dropping the ngram arm from further phases unless a
long-context test revives it.

### Finding 3 — the plan's central speed hypothesis is much weaker than stated

Research doc 02 §5 and the plan's reordering rest on "quant size is a major speed
knob" — the reason Q3-vs-Q4 was promoted ahead of KV/batch tuning. Measured:

```
Q4 -> Q3 : 16.69 -> 12.52 GiB, -4.17 GiB
speed    : 8.24 -> 9.01 tok/s bench  (+9 %)
           8.22 -> 9.25 tok/s code   (+13 %)
```

Giving up 4.17 GiB and ~3.6 points of top-1 fidelity buys about **10 %**. Over
the same baseline, MTP buys **30–47 %** and costs nothing in fidelity. The
promotion of Q3-vs-Q4 was still the right call — it just turned out to answer
"which quant" rather than "where the speed is".

### Finding 4 — MTP's benefit is inversely proportional to VRAM fit *(new mechanism)*

This one contradicts the plan directly. Continuation §3 and research doc 04 §7
predict `UD-Q3_K_XL + MTP` as the likely performance winner, reasoning that Q4 is
too VRAM-saturated to host MTP. Measured, the opposite:

| | MTP gain, bench | MTP gain, code |
|---|---|---|
| Q4 (~42 % of weights CPU-resident) | **+30 %** | **+47 %** |
| Q3 (~22 % CPU-resident) | **−1 %** | +11 % |

Acceptance is essentially identical between the quants (77.5–78.1 % bench,
96.4–98.0 % code), so the difference is not draft quality. The mechanism is that
speculative decoding amortises the cost of *one forward pass* across several
tokens. When 42 % of the weights are on the CPU, a forward pass is very
expensive and batching the verification is a large win. When only 22 % are, the
pass is already cheaper, the constant draft overhead is unchanged, and the net
gain collapses — on the short prompt it goes negative.

**MTP is a compensation mechanism for CPU offload, not an independent
accelerator.** The worse a model fits, the more it helps. That is why the
smaller quant benefits less, and it is the reason the two levers do not stack the
way the plan assumed.

### Consequence for the decision

```
Q4 + MTP n=2 : 10.67 bench / 12.10 code
Q3 + MTP n=2 :  8.88 bench / 10.30 code
```

**Best-Q4 is faster than best-Q3 on both prompts**, while also holding the higher
fidelity proxy (~96 % vs ~92.4 % top-1). If the quality benchmark agrees, this is
not a speed-versus-quality trade-off at all — Q4 wins on both axes and the
"performance lane / reliability lane" framing collapses into one answer.

Quality measurement is what decides it; speed alone cannot.


---

## E7 — Prefix-cache gate (plan Phase G)

**Hypothesis:** Qwen3.8 is hybrid recurrent/attention; llama.cpp issues report
hybrid-memory models logging *"forcing full prompt re-processing due to lack of
cache data"*. If that happens here, every agent turn pays a cold prefill and
prefill — not decode — is the real bottleneck.

**Method:** an OpenCode-shaped conversation (system + 8 tool schemas + 40-file repo
context ≈ 3.9K tokens) that only ever appends, reading `cache_n` / `prompt_n` from
`/completion` timings. Then one perturbation at a time.
Harness: `bench/prefix_cache_gate.py`.

| turn | prompt_n (evaluated) | cache_n (reused) | wall |
|---|---|---|---|
| 1 (cold) | 3878 | 0 | 12.6 s |
| 2 | **43** | 3874 | 2.8 s |
| 3 | **35** | 3913 | 3.9 s |
| 4 | **37** | 3944 | 1.3 s |

**PASS.** The hybrid full-reprocess bug does not reproduce on b10472. Append-only
agent turns evaluate ~40 tokens instead of ~3900.

**The perturbation result is the actionable part:**

| change | cache retained | cost |
|---|---|---|
| reorder tool schemas | **0 %** | full 3990-token re-prefill, 11.1 s |
| edit one system-prompt sentence | **0 %** | 11.5 s |
| prepend a skill block | **0 %** | 12.1 s |
| append only (control) | **100 %** | 2.4 s |
| `cache_prompt=false` (reference) | 0 % | 9.9 s |

The cache is **prefix-exact**. Any edit above the append point costs the same as
having no cache at all. At 4K that is 11 s; scaled to 64K it is ~2 min and to 256K
~8 min.

**Operational rule for the OpenCode/Xeno integration:** freeze everything above the
append point — stable tool-schema order, byte-stable system prompt, skills injected
once at the start and never reordered or prepended later.

---

## E8 — Runtime tuning sweeps (plan Phases E, K)

All three levers change *where* work happens and *how* it is scheduled, not what
the model computes. Rather than paying ~48 min of quality benchmark per arm, every
config emitted a greedy sample (`temperature 0, top_k 1, seed 42`) whose SHA-256
was compared to the control. **Every config in every sweep matched**, so each lever
is provably output-neutral — stronger evidence than a pass-rate comparison, which
could miss a small regression.

All decisions read from the **code-rewrite prompt**. The 11-token bench prompt
stayed inside 9.86–11.90 across every configuration; it is overhead-dominated and
cannot discriminate. Harness: `bench/sweep_runtime.py`.

### E8.1 `--fit-target` — +9.3 %

| target | GPU layers | code tok/s | range | VRAM free |
|---|---|---|---|---|
| 1024 (default) | 32 | 11.34 | [11.23, 11.50] | 867 MiB |
| 256 | 35 | 8.28 | **[6.70, 11.57]** | 345 |
| 512 | 34 | 11.89 | [11.46, 12.08] | 357 |
| **768** | **33** | **12.39** | **[12.11, 12.40]** | 584 |
| 1536 | 30 | 11.61 | [11.59, 11.62] | 1079 |
| 2048 | 28 | 11.06 | [10.87, 11.11] | 1403 |

**Not monotonic in layer count.** 1536 with 30 layers beat the 1024 default with
32; 2048 with 28 layers posted the best synthetic figure. The governing variable is
the *balance* between resident layers and the headroom left for compute buffers,
and 768 lands on it.

At 256 the code prompt did not get slower on average — it became **unstable**
([6.70, 8.28, 11.57], a 73 % spread with one perfectly normal sample). That is the
signature of intermittent driver eviction at 345 MiB free, and it is why an
optimizer that maximises resident layers is the wrong optimizer here.

### E8.2 CPU threads — +6.9 %

| `-t` | 6 | 8 | 10 | 12 | 14 (default) | 18 | 20 |
|---|---|---|---|---|---|---|---|
| code tok/s | 9.38 | 10.59 | 11.19 | 11.54 | 12.70 | **13.58** | 13.42 |
| prompt processing | — | — | — | — | 166.8 | **167.4** | 137.2 |

Confirmed at N=5 for 14 / 18 / 20; ranges for 14 ([12.64, 12.74]) and 18
([13.53, 13.63]) do not overlap.

Throughput rises monotonically from 6 to 20, which **contradicts the usual
physical-core guidance**; `-t 6` (P-cores only, on a 6P+8E i5-13500) was the worst
result measured. With 33 of 65 layers CPU-resident, E-cores are contributing, not
dragging.

`-t 20` claims every logical thread and **costs 18 % of prompt processing**
(137.2 vs 167.4) with a wider decode spread. `-t 18` leaves two threads for the OS
and wins decode, spread and pp at once.

**New mechanism:** `-tb 14` under `-t 20` dropped *decode* from 13.42 to 12.71.
`-tb` is documented as the prompt/batch thread count, but MTP verifies several
drafted tokens in a single batched pass, so the batch thread count is on the decode
path whenever speculative decoding is enabled.

### E8.3 batch / ubatch — +3.8 %

| `-b` / `-ub` | code tok/s | range | prompt processing |
|---|---|---|---|
| 2048 / 512 (default) | 13.00 | [12.46, 13.08] | 164.4 |
| 1024 / 512 | 13.08 | [11.92, 13.47] | 159.7 |
| **2048 / 256** | **13.49** | [12.99, 13.65] | **164.2** |
| 512 / 128 | 13.36 | [13.30, 13.38] | **103.0** |

Confirmed at N=5 for the top three.

`-b 512 -ub 128` had the best raw decode in the N=3 pass and was still **rejected**:
it costs 33 % of prompt processing. Using E7's numbers, that is a 59-second penalty
per cache invalidation at 16K versus a 0.8-second gain on a 500-token response —
about 74 responses to repay a single miss, and E7 showed misses are easy to trigger.
Keeping `-b` large protects pp while halving `-ub` frees compute-buffer VRAM.

### E8.4 Stacked result

```
--fit-target 768 -t 18 -b 2048 -ub 256
```

Code-rewrite decode ~11.3 → ~13.5 tok/s, **about +19 %**, with the greedy output
bit-identical at every step. Written up as `scripts/production-q4-tuned.ps1`.


---

## E9 — Speculative sub-knobs, and the measurement failure they exposed

**Hypothesis:** `--spec-draft-p-min` (default 0.00), `--spec-draft-p-split`
(default 0.10) and `--spec-draft-n-min` are unswept by every planning document.
MTP is the largest measured lever, and these control when it drafts at all, so
they might be free throughput.

**First pass, N=5, control first** — looked like a clear win:

| knob | code tok/s | vs control | acceptance |
|---|---|---|---|
| control | 11.53 | — | 88.4 % |
| `-n-min 2` | 12.87 | **+11.6 %** | 86.1 % |
| `-p-min 0.10` | 12.66 | **+9.8 %** | 87.1 % |
| `-p-split 0.25` | 12.54 | **+8.8 %** | 86.2 % |
| `-p-min 0.05` | 12.20 | +5.8 % | 84.4 % |
| `-n-min 1` | 12.07 | +4.7 % | 83.7 % |
| `-p-split 0.00` | 11.34 | −1.6 % | 84.2 % |

I checked one confounder — whether speed was just tracking free VRAM, which
varied 267–517 MiB across the rows. Correlation was **+0.06**, so I reported the
gains as real.

**Second pass, leaders re-run against a fresh control — every one reversed:**

| knob | vs fresh control |
|---|---|
| `-n-min 2` | **−0.8 %** |
| `-p-min 0.10` | **−10.1 %** |
| `-n-min 2 + -p-min 0.10` | −5.2 % |
| all three | −4.3 % |

The first sweep measured **machine drift, not knob effects.** Its control ran
first, in a slow window; every later configuration ran as the machine recovered.
A monotonic time trend is indistinguishable from a monotonic knob effect when the
control is sampled once, at one end of it. The VRAM correlation check could not
catch it because the drift was not VRAM-driven.

### E9.1 How large the drift is

Six restarts of an **identical** configuration, N=5 each:

```text
11.63   12.59   12.60   12.63   13.21     tok/s (per-restart medians)
peak-to-peak 13.6 %      stdev 4.5 % of mean
```

**This floor exceeds every per-lever claim in E8** (+9.3 %, +6.9 %, +3.8 %). None
of them is established by the control-first design that produced them.

### E9.2 Paired re-test of the stacked configuration

Interleaving makes both arms share the drift instead of assigning it all to
whichever ran later:

```text
stock  11.79   10.90   11.12
tuned  11.34   12.50   12.12
diff   -3.8%  +14.7%   +9.0%      paired mean +6.6%
pooled (15 samples per arm):  11.18 -> 12.25  =  +9.6%
```

One pair went negative. The tuned configuration still wins on pooled samples and
on the independent 45-minute quality run (+7.4 % verified tasks/hour), but the
honest magnitude is **roughly half** the "+19 % cumulative" that adding the three
E8 deltas produced.

### E9.3 Rules this changes

- **Interleave the arms.** Control-first ordering is not adequate on this machine.
- **Report paired differences**, never a ratio of two separately-measured medians.
- **Never add per-sweep deltas** — each carries its own drift and summing compounds it.
- **An effect below ~14 % requires a paired design**, or it is not measurable here.
- Speculative sub-knobs: **rejected**, keep llama.cpp defaults.

### E9.4 Why the E8 conclusions are still kept

The stacked configuration is retained despite the per-lever figures being
unseparable, on two pieces of evidence that do not share the flawed design:
the pooled 15-sample paired comparison (+9.6 %), and a 45-minute 30-sample
quality run per arm (+7.4 % tasks/hour) that averages over far more drift than a
five-sample burst. What is **not** retained is any claim about which individual
flag earned the gain.

---

## E10 — Measurement primitives under test  *(bench/harness.py)*

Three silent failures in this project's own tooling — a median that returned the
maximum, a BOM that deleted the baseline row of every table, a device token whose
trailing comma made the CPU layer count read zero — were all pure functions.
They are now extracted with regression tests written **before** the
implementations (`bench/tests/test_harness.py`, 16 tests): `median`,
`load_jsonl`, `parse_layer_split`, `project_prefill_seconds`.

Each test names the incident it guards. The shared design rule is that these
functions **raise rather than guess**: an empty median, a corrupt JSONL line, a
layer split that does not add up, and a non-positive token rate all raise, because
every one of the original failures was a plausible-looking wrong number rather
than a crash.

`sweep_runtime.py` now calls them instead of its inline copies.


---

## E11 — Context depth, and the crossover that did not happen  *(plan Phase F)*

Every planning document named this as the one phase that could overturn the
Q4 verdict. Effects here are in tens of percent — comfortably above the 13.6 %
drift floor from E9, unlike the 16K flag tuning.

Method: fill ~80 % of the window with realistic source text, measure cold prefill
once (a 256K prefill costs minutes; N=3 would cost an hour per row), then measure
decode 5× using `cache_prompt` so the deep prefill is paid once — which is also
how an agent behaves. Harness: `bench/depth_sweep.py`.

### E11.1 Q4 collapses with depth

| ctx | GPU / CPU layers | KV | cold prefill | decode | vs 16K |
|---|---|---|---|---|---|
| 16K | 33 / 32 | 512 MiB | 40 s | **9.77** | — |
| 32K | 31 / 34 | 1 024 MiB | 80 s | 7.44 | −24 % |
| 64K | 27 / 38 | 2 304 MiB | 205 s | 4.37 | −55 % |
| 128K | 20 / 45 | 5 632 MiB | 481 s | **2.10** | **−78 %** |
| 256K | — | — | — | **stopped** | see below |

KV growth is the mechanism: it evicts GPU layers, 33 → 20 across the range. At
128K decode is 2.10 tok/s — a 500-token reply takes 4 minutes and a cold prefill
8 minutes. That is not an interactive agent.

**256K loaded but was stopped under the protocol's paging condition**, not on a
throughput number: host RAM free 0.63 GB of 47.69, pagefile 10.11 GB in use,
llama-server working set 26.64 GB, 296 pages/sec. Any throughput measured under
that pressure would describe Windows paging rather than the model.

The measured KV curve (512 → 1024 → 2304 → 5632 MiB) also confirms the correction
already made to the benchmark report: 256K lands near 11 GiB, not the ~16 GiB the
research-doc proxy predicted.

### E11.2 Q3 keeps more layers, prefills faster, and still never wins

| ctx | | GPU layers | KV | prompt processing | decode |
|---|---|---|---|---|---|
| 64K | Q4 | 27 | 2 304 MiB | 227.3 | **4.37** |
| 64K | Q3 | **34** | 2 048 MiB | **299.9** | 3.68 |
| 128K | Q4 | 20 | 5 632 MiB | 193.4 | **2.10** |
| 128K | Q3 | **26** | 5 120 MiB | **244.5** | 2.09 |

Q3 does exactly what the theory predicted — 7 more resident layers at 64K, 6 more
at 128K, and consistently faster prefill — **and still loses or ties on decode.**
Q4 is 19 % faster at 64K; at 128K they are indistinguishable.

**The crossover does not exist at any measured depth.** This closes the largest
open risk in the project: the 16K verdict was flagged as vulnerable to exactly
this, and it survives.

The mechanism is consistent with the 16K result, where Q3 baseline beat Q4
baseline but Q3+MTP lost to Q4+MTP: `UD-Q3_K_XL` costs enough more per token to
dequantize that it gives back the residency advantage. Layer count is not the
only thing that matters — per-layer cost matters too, and the smaller quant is
not the cheaper one here.

### E11.3 Q8_0 KV is the one deep-context lever that pays

| ctx | | GPU layers | KV | prompt processing | decode |
|---|---|---|---|---|---|
| 64K | F16 | 27 | 2 304 MiB | 227.3 | 4.37 |
| 64K | **Q8_0** | **29** | **1 224 MiB** | **256.1** | **5.10** (+16.7 %) |
| 128K | F16 | 20 | 5 632 MiB | 193.4 | 2.10 |
| 128K | **Q8_0** | **23** | **2 720 MiB** | **212.9** | **2.48** (+18.1 %) |

KV halves as expected, buying 2–3 GPU layers and ~17 % decode at both depths —
consistently above the drift floor.

**It also settles a build question empirically.** The deep-research report warned
that requesting a quantized-KV Flash-Attention kernel that was not compiled falls
back to a catastrophically slow path, and recommended a pinned SM89 build with
`GGML_CUDA_FA_ALL_QUANTS=ON` *before* testing Q8. Measured: Q8 KV is **faster**
than F16 on the stock b10472 binary at both depths. **The custom build is not
required for q8_0 KV.**

### E11.4 But Q8 KV is a trade, not a free win — and the cheap check missed it

The sweeps verify quality with a greedy probe. For flags that do not change
arithmetic that is strictly stronger than a pass-rate comparison. **Q8 KV changes
the arithmetic**, and the probe sends a 4-token prompt — so it barely touches the
very cache Q8 quantizes. It reported "hash identical", and that was not evidence.

Re-tested properly (`bench/kv_equivalence.py`): identical greedy settings, but
with ~46.5K tokens of context so the continuation is decided by attention over a
deeply-populated cache:

```text
prompt_n = 46 557
F16  hash 1A4F7C9924198E8A
Q8   hash 05C38B387571F755
common prefix: 1 character of 778
```

**Completely different output.** Divergence is not automatically damage — a
long-context summary has many valid continuations — but it means Q8 KV cannot be
adopted on the same "provably output-neutral" grounds as the other flags. Its
effect on task success has to be measured, not asserted.

**Rule this adds:** an equivalence probe must exercise the thing being changed.
A short-prompt greedy check is valid for scheduling and placement flags and
worthless for anything touching the cache.


---

## E12 — Deep-context quality: two corpora, and Q8_0 KV survives both

E11.4 left one question open: Q8_0 KV buys ~17 % at 64K–128K, but its quality was
only ever measured at 16K, where it has no benefit and only a cost. A verdict from
that measurement would describe the wrong regime.

### E12.1 Corpus v1 — retrieval at depth

Six execution-verified tasks over a shared ~44K-token repository prefix. Each
answer depends on an arbitrary constant planted at a known depth
(`MAX_RETRIES = 7`, `TIMEOUT_MS = 8700`, `CHECKSUM_FIELD = "drain_token"`) that no
prior can supply. `cache_prompt` pays the deep prefill once; every later task
reuses it, which is also how an agent behaves.

| | F16 KV | **Q8_0 KV** |
|---|---|---|
| pass rate at 64K | 100 % (18/18) | **100 % (18/18)** |
| verified tasks / hour | 51.8 | **57.4** (+10.7 %) |
| warm turn, median | 51.2 s | **48.4 s** |
| cold prefill | 349.1 s | **321.0 s** |

Identical quality, ~11 % more throughput. But **both arms ceilinged**, which
bounds the damage rather than measuring it.

### E12.2 Corpus v2 — built specifically to break the ceiling

Ten tasks, four kinds of difficulty added, each aimed at a different way
long-context retrieval fails:

- **Confusable neighbours** — every planted shard has decoys whose IDs differ by a
  digit or a transposition (`0203` / `0230` / `2003`) with near-miss constants and
  an upper-cased field name, placed *immediately before* the real block so a
  forward scan meets the wrong one first.
- **Multi-hop** — `DEPENDS_ON` chains requiring two retrievals (2941 → 1508 → 417 → 203).
- **Aggregation** — one task sums `MAX_RETRIES` across all four authoritative
  shards; missing any single one fails it.
- **Depth** — the last planted shard sits at 95 % of the prefix.

**Result: Q8_0 KV scored 100 % (30/30)** — every task, including the 95 %-depth
retrieval, both two-hop chains, all three aggregations, and the
distractor-rejection task.

### E12.3 What this supports, and what it does not

Across two corpus designs and 48 samples, **Q8_0 KV at 64K shows no measurable
retrieval degradation**, while buying ~17 % decode and halving KV. Combined with
the 16K result (86.7 % vs 90.0 %, and slower), the recommendation is
depth-conditional and evidenced on both sides:

```text
16K-32K   F16 KV      Q8 measurably worse and no faster
64K       Q8_0 KV     identical quality, +11-17% throughput
```

**What it does not support:** a claim that Q8 is *exactly* as good. Both corpora
ceilinged on this model, so a 2–3 % regression remains below resolution. Two
independent designs failing to find a difference is meaningful evidence of
absence-of-large-effect, not proof of zero effect.

There is also a diminishing return here worth naming: past this point the
exercise becomes designing a task *the model itself* fails, so that Q8 can be
seen failing it more. That measures corpus difficulty, not KV precision. The
useful remaining axis is **depth** — 128K quality — not more difficulty at 64K.

### E12.4 Two instrument bugs the corpus tests caught first

Both would have produced a confident verdict from a broken instrument:

- **v1:** `Handler0017` was emitted twice — once as a routine block at index 17,
  once as the planted block — so "the class for shard 17" had two contradictory
  answers in context.
- **v1:** the size test asserted only a lower bound, so a **112K-token** corpus
  passed and then failed every request with HTTP 400 against a 64K window — 0/18
  in four seconds. Both bounds are asserted now, and the same test caught v2
  landing at 19.5K tokens because its blocks are shorter.


---

## E13 — 128K quality, and a same-boot redo of 64K

Closes the last unmeasured axis and repairs a comparison that the machine restart
had invalidated.

### E13.1 Why 64K had to be re-run

`v2-64k-q8` (30/30) was measured **before** a planned machine restart. Comparing a
post-restart F16 arm against it would be the cross-boot comparison E9 established
as invalid — free VRAM at the new boot was **9 326 MiB**, below the 9 933–10 530
range of every earlier launch. Both arms were therefore re-run in one boot.

| 64K, corpus v2 | F16 KV | **Q8_0 KV** |
|---|---|---|
| pass rate | **100 % (30/30)** | **100 % (30/30)** |
| warm turn, median | 55.7 s | **47.2 s** (−15.3 %) |
| total wall | 1 859.9 s | **1 706.9 s** |

Cold prefill is not comparable here — the F16 arm inherited a warm prefix from an
aborted run on the same server (80.2 s vs 246.6 s). The warm median over 29 turns
per arm is the number.

### E13.2 A stop I called wrongly, and retracted

The first re-run attempt (`v2b-64k-f16`) was stopped at 3/30 when its third sample
took **1 057.5 s** against the previous sample's 105.4 s. I read that as a decode
collapse and "confirmed" it with a probe returning **11.21 tok/s** — far above the
4.37 tok/s expected for this configuration, which I took as further evidence of
something wrong.

**Both readings were wrong.** The probe used a 4-token prompt with
`cache_prompt=false`, measuring decode over an *empty* cache; 4.37 tok/s was
measured with the window 80 % full. They are not the same measurement — the exact
error E11.4 established a rule about, committed by the person who wrote the rule.

The real cause was ordinary variance: that sample hit `max_tokens 1536` while its
predecessor answered in 212 tokens; at depth that is roughly 1 000 s of generation.
The redo completed 30/30 without incident.

Retraction recorded in-band on the `v2b-64k-f16` marker row (`retracted: true`).

### E13.3 128K — measured at real depth

Running the 64K corpus against a 128K-configured server would only have measured a
44K retrieval with a different layer split. The corpus was scaled to **1 550 blocks
→ a 114 406-token prompt**; planted shards are placed by percentage, so the deepest
still sits at ~95 % of a 114K context. Guarded by
`test_v2_deep_variant_fits_a_128k_window_with_room_to_answer` (both bounds) and
`test_v2_deep_variant_still_plants_each_shard_once`.

| 128K, 114 406-token prompt | F16 KV | **Q8_0 KV** |
|---|---|---|
| pass rate | **100 % (10/10)** | **100 % (10/10)** |
| warm turn, median | 144.5 s | **104.0 s** (−28 %) |
| total wall | 1 993.3 s | **1 555.1 s** (−22 %) |
| cold prefill | 715.5 s | 701.1 s |

`--attempts 1` (10 samples per arm) because decode at 128K is ~2.3 tok/s.

Host pressure stayed well clear of the E11 stop condition: RAM free 4.22 GB,
pagefile 1.27 GB, against 0.63 GB / 10.11 GB at 256K.

### E13.4 Conclusion

**Q8_0 KV's advantage grows with depth and never costs quality at depth:**

```text
16K    F16 KV     Q8 worse (86.7% vs 90.0%) and no faster - nothing to reclaim
64K    Q8_0 KV    identical quality (30/30 both), ~15% faster warm turns
128K   Q8_0 KV    identical quality (10/10 both), ~28% faster warm turns
```

That ordering follows from KV being a larger share of the memory budget the deeper
the context runs, so halving it returns more GPU residency the further out you go.

**The measurement queue is now empty.** What remains is integration — OpenCode,
OpenClink, a real repository workload — and the first thing to check there is
whether OpenCode's serialization respects the prefix-freeze rule from E7.

---

## E14 — Residency: does a smaller artifact beat Q4's half-CPU split?

**Date:** 2026-08-19. **Hypothesis:** a quantization small enough to become
GPU-resident beats Q4 despite lower fidelity, on verified tasks per hour rather
than tok/s. **Design:** `bench/model_arena.py` — arms alternated across boots and
paired by round, order reversed on even rounds, because two quantizations cannot
share a boot and report 04 measured a 13.6 % restart drift.

```text
arena 1:  q2kxl-nomtp   +62.12%   [+60.94, +61.78, +63.64]   RESOLVED
          q2kxl-mtp2    +50.13%   [+50.42, +47.56, +52.41]   RESOLVED
arena 2:  q2kxl-nomtp   +64.22%   [+72.10, +60.59, +59.97]   RESOLVED
          iq2xxs-nomtp +219.58%   [+237.36, +212.28, +209.10] RESOLVED
```

**Result: confirmed, and the curve is a cliff.**

```text
Q4        33 GPU / 32 CPU    12.6 - 13.7 tok/s    prefill 156
Q2_K_XL   61 GPU /  4 CPU    21.3 - 22.0 tok/s    prefill 394-486
IQ2_XXS   65 GPU /  0 CPU    42.4 - 42.5 tok/s    prefill 809-818
```

Moving 28 layers to the GPU buys +64 %. Moving the final **four** buys another
+95 %. Report 01's "more GPU layers is not monotonically better" was measured at
32–35 layers, nowhere near this. Report 10 §1.

## E15 — MTP inverts once the target is resident

**Hypothesis** (from the new research): speculative benefit collapses when the
target forward pass becomes cheap. **Result: confirmed, with a mechanism.**

```text
q2kxl-nomtp    61 + 4     21.3 - 22.0 tok/s
q2kxl-mtp2     55 + 10    19.9 tok/s          -7%
```

The draft head's VRAM pushes **six target layers back onto the CPU**. Speculation
trades residency for arithmetic; that trade pays on Q4 and loses on Q2. E5 found
the same mechanism running the other way.

## E16 — Does the quantization survive the workflow, not just the benchmark?

**Design:** `run_retry_bench.py` (attempt, then one retry with the real
traceback), `protocol_gate.py` (nested-schema tool call + `tool_call_id`
round-trip), `stability_gate.py` (100 appending turns, forced prefix
invalidation every tenth). 30 tasks per arm, identical budgets.

| | Q4 | Q2_K_XL | IQ2_XXS |
|---|---:|---:|---:|
| first attempt `p1` | 83.3 % | 83.3 % | 73.3 % |
| retry `p2` | 40.0 % | 20.0 % | **62.5 %** |
| **accepted** | **27/30** | 26/30 | **27/30** |
| worker wall | 4 008.7 s | 1 972.5 s | **1 599.0 s** |
| merged tasks/hr | 17.8 | 26.1 | **29.4** |
| tool-call compliance | 80.0 % | 86.7 % | **93.3 %** |
| 100-turn hangs | 0 | 0 | 0 |

**Result: no quality regression detected; the fast lane is IQ2_XXS.**

Three retractions this experiment produced, all from measuring rather than
reading a summary line:

1. "Q2 tool compliance 40 %" — the probe's `max_tokens 1024` truncated a model
   whose median reasoning is 2 811 chars. With `finish_reason` recorded, **every
   non-call was a truncation**, none a refusal.
2. "Q4 p1 = 70 %" — a 10-task sample. At 30 tasks it is 83.3 %, identical to Q2.
   The first number would have claimed the quantized model was *more* accurate
   than its source.
3. "24.0 merged tasks/hour" from a run where **every request returned HTTP 503**.
   Escalation is charged as a constant, so 30 tasks that never ran still produced
   a plausible number. `retry_economics` now refuses a run with no worker time.

The research's own economic assumption also failed: it models
`p2 = min(p1 + 0.10, 0.95)`, predicting 0.93 at our p1. Measured p2 is
**0.20 – 0.625**. Report 10 §3.
