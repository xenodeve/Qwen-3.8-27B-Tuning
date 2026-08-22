# Qwen3.8-27B Local Worker — Q3 vs Q4 Optimization Result

> 🔴 **SUPERSEDED 2026-08-18. Do not act on this document.** Its
> recommendation — `UD-Q4_K_XL` with MTP at `--spec-draft-n-max 2` — was
> abandoned: the project serves `UD-IQ2_XXS`/`UD-IQ2_S`, and `draft-mtp`
> measures **−71 % at 131,072**. Kept as the record of what was believed
> on 2026-08-18. Current state:
> [`docs/reports/32-BENCHMARK-STATUS-BRIEF.md`](../docs/reports/32-BENCHMARK-STATUS-BRIEF.md).

**Machine:** RTX 4070 SUPER 12 GB · i5-13500 · 48 GB DDR5 · Windows 11
**Runtime:** llama.cpp b10472 / commit `60eeeb608` · CUDA 12.4 · driver 610.88
**Date:** 2026-08-18 · **Metric:** verified successful coding tasks per hour

---

## 1. Verdict

**Use `UD-Q4_K_XL` with built-in MTP speculative decoding at `--spec-draft-n-max 2`.**

There is no speed-versus-quality trade-off to make. Q4 wins on **both** axes:

| | **UD-Q4_K_XL + MTP n=2** | UD-Q3_K_XL + MTP n=2 |
|---|---|---|
| **Verified tasks / hour** | **33.6** | 22.2 |
| **Pass rate** | **90.0 %** (27/30) | 86.7 % (26/30) |
| Median tok/s during the task suite | **10.56** | 8.73 |
| Synthetic decode | **10.67** tok/s | 8.88 |
| Code-rewrite decode | **12.10** tok/s | 10.30 |
| Wall clock for the same 30 tasks | **2 889 s** | 4 213 s |
| Tokens generated for the same work | 29 363 | 34 543 (+18 %) |
| Reasoning emitted | 82 653 chars | 103 486 (+25 %) |
| Disk | 16.69 GiB | 12.52 GiB |
| Top-1 agreement vs BF16 (vendor) | ~96 % | ~92.4 % |

**Q4 is 51 % more productive per hour.** Q3's penalty compounds: it decodes
slower *and* needs more tokens to reach an answer that is slightly worse.

Keep Q3 only as a contingency — see §6.

---

## 2. Production launch command

```powershell
C:\AI\llama.cpp-cuda\llama-server.exe `
  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL `
  --alias qwen38-q4 `
  -c 16384 `
  -ngl auto --fit on -fa on -np 1 `
  --no-mmproj-auto `
  --spec-type draft-mtp --spec-draft-n-max 2 `
  --host 127.0.0.1 --port 8080
```

Saved as `scripts\production-q4.ps1` (Q3 equivalent: `scripts\production-q3.ps1`).

**Client-side settings the server will not supply correctly:**

| setting | value | why |
|---|---|---|
| `min_p` | **0.0** | server default is 0.05; the vendor specifies 0.0 for *both* thinking and non-thinking |
| `temperature` / `top_p` | 1.0 / 0.95 thinking · 0.7 / 0.80 non-thinking | two distinct published profiles; one server default cannot serve both |
| `presence_penalty` | 0.0 thinking · 1.5 non-thinking | same |
| `chat_template_kwargs.reasoning_effort` | send explicitly | template silently defaults to `xhigh` and remaps `high` → `xhigh` |

`--jinja` is omitted deliberately — already the default in b10472, so passing it is a no-op.

---

## 3. What decided it

### 3.1 MTP is the dominant lever, not quant choice

| change | speed effect | fidelity cost |
|---|---|---|
| Q4 → Q3 (−4.17 GiB) | **+9 % / +13 %** | −3.6 pts top-1 |
| enabling MTP on Q4 | **+30 % / +47 %** | **none measured** |

Giving up 4.17 GiB and 3.6 points of fidelity buys about 10 %. Turning on a flag
that already ships inside the GGUF buys 30–47 % and costs nothing.

### 3.2 MTP compensates for CPU offload — so it helps Q4 *more*

Actual layer placement, read from the verbose load report:

| | layers on GPU | layers on CPU | MTP gain (bench / code) |
|---|---|---|---|
| **Q4** | 32 | **33 (51 %)** | **+30 % / +47 %** |
| **Q3** | 43 | 22 (34 %) | **−1 % / +11 %** |

Draft acceptance is nearly identical between the two (77.5–78.1 % synthetic,
96.4–98.0 % code-rewrite), so this is not a draft-quality difference.

Speculative decoding amortises **one forward pass** across several tokens. With
51 % of weights on the CPU a forward pass is very expensive, so batching the
verification is a large win. At 34 % the pass is already cheaper, the constant
draft overhead is unchanged, and the gain collapses — on the short prompt it goes
slightly negative.

**This inverts the plan's prediction.** The research docs expected
`UD-Q3_K_XL + MTP` to be the performance winner on the theory that Q4 was too
VRAM-saturated to host MTP. The opposite is true: the worse a model fits, the
more MTP repays.

### 3.3 Prompt type moves the MTP result more than any tuning knob

| prompt | Q4 + MTP n=2 | acceptance |
|---|---|---|
| short instruction, nothing in context | 10.67 tok/s | 78.1 % |
| **rewrite the class in the prompt** | **12.10 tok/s** | **98.0 %** |

The second case is what a coding agent actually does — reproduce existing code
with a small change. **Benchmarks that measure speculation with a short prompt
understate real agent throughput.**

### 3.4 Draft depth: 2, not the widely-quoted 4–5

| n_max | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| tok/s (synthetic) | **10.58–10.78** | 9.49–11.13 | 9.37–11.08 | 8.59–9.38 | 7.18–10.45 |
| acceptance | **77.5 %** | 68.6 % | 65.2 % | 56.4 % | 52.4 % |
| VRAM free after load | 772 MiB | 942 | 1155 | 1089 | 1275 |

Free VRAM *rises* with `n_max`, meaning `--fit` reserves more for the draft path
and **evicts target layers from the GPU**. High depth is penalised twice: falling
acceptance and falling residency. n=3 has a higher ceiling but a lower floor and
a much wider spread; n=2 is the right production choice for a throughput metric.

### 3.5 `ngram-simple` is not competitive here

30.8 % acceptance on the prompt type its own documentation names
(source-code rewriting), converting to 8.37 vs 8.22 tok/s — inside noise. On a
short prompt it drafts nothing. Tested on both quants, same verdict. Dropped.

### 3.6 Speculation is output-lossless on this stack

Greedy samples (`temperature 0, top_k 1, seed 42`) were **byte-identical across
every speculative configuration** within each quant — Q4 `6F8AAC2789…`,
Q3 `0659173109…`. The greedy divergence reported in llama.cpp issue #25618 for
quantized targets **does not reproduce here**. MTP is a pure performance toggle.

### 3.7 Where the quality difference actually shows up

Both models fail the same task and pass eight of the ten identically. The entire
measured quality gap is one task:

| task | difficulty | Q4 | Q3 | Q4 wall | Q3 wall |
|---|---|---|---|---|---|
| `lfu_cache` | hard | **3/3** | **2/3** | 514 s | 793 s |
| `bracket_matching` | easy* | 0/3 | 0/3 | 337 s | **889 s** |
| all other 8 tasks | — | 3/3 | 3/3 | — | — |

\* labelled easy, but it requires ignoring brackets inside quoted string literals
with backslash escapes — neither model ever solved it. That is a capability
ceiling, not variance.

`bracket_matching` is the clearest illustration of the compounding penalty: Q3
spent **2.6× the wall time** and still failed.

---

## 4. How it was measured

- **Speed:** N=3 per configuration, server restarted between every config,
  environment snapshotted before each load. Median plus min–max reported;
  conclusions rest on non-overlapping ranges, never on a point estimate.
- **Quality:** 10 coding tasks × 3 attempts × 2 configs = 60 samples. Each reply's
  code is extracted and **executed against assertions in a subprocess with a 20 s
  timeout** — pass/fail is machine-verified, no LLM judge, no partial credit.
  Tasks were chosen to probe what quantization damages: eviction order,
  tie-breaking, cycle detection, operator precedence, transpositions.
- **Equivalence:** greedy sample per config, SHA-256 compared.
- **Protocol correctness:** all 8 gate items passed before any performance work
  — plain completion, developer role, simple tool call, nested object arguments,
  tool-result continuation, repeated tool loop, reasoning separation, `min_p`.

Artifacts: `EXPERIMENTS.md` (full log E0–E6), `results\*.jsonl`,
`results\summary.md`, `scripts\`, `logs\`, `bench\`.

---

## 5. Corrections to the planning documents

| claim | status |
|---|---|
| `~2.5 GB` MTP VRAM overhead (video) | **wrong.** The `blk.64.*` tensors total **285.8 MB**; measured VRAM went *down* |
| `4–5` draft-step sweet spot (video) | **not reproduced.** Peak is n=2–3; n≥5 regresses |
| `3×` speedup (video) | **not reproduced.** 1.30–1.47× here |
| `Q3 + MTP` is the performance lane (plan) | **inverted.** Q4 + MTP is faster on both prompt types |
| quant size is the dominant speed knob (research doc 02 §5) | **weak.** Worth ~10 %; MTP is worth 30–47 % |
| `ngram-simple` is a serious candidate (research doc 04 §7) | **not here.** No measurable gain on either quant |
| `<think>` may leak into `content` (my own earlier report) | **wrong.** `/v1/chat/completions` returns a separate `reasoning_content` |
| `reasoning_effort=xhigh` is the largest single cost (my own earlier report) | **overstated.** Measured 50–250 reasoning tokens, and the sweep could not resolve the effect above noise |
| llama.cpp issue #25618 greedy divergence | **does not reproduce** on this build/model |
| §15 benchmark figures, §6 size proxy, §4 State-4 fidelity reasoning | **confirmed accurate** |

---

## 6. Unresolved risks and what is NOT tested

1. **Only 16K context was tested.** Every number here is at `-c 16384`. The KV
   estimate for this architecture family is ~16 GiB (F16) at 256K, which will
   change the layer split completely and could reverse the Q3/Q4 ranking, since
   Q3 leaves 4.17 GiB more room. **This is the single most likely thing to
   invalidate the verdict.**
2. **Prompt-cache and multi-turn reuse untested.** Agent economics depend on
   prefix stability, not total context length.
3. **KV placement and precision untested** (`--no-kv-offload`, `q8_0`).
   `--fit-target`, `--fit-ctx`, `--cache-ram` (default 8192 MiB against ~11 GB
   free host RAM) all exist in b10472 and are untuned.
4. **VRAM headroom is thin and variable.** Free VRAM before load ranged
   9 933–10 530 MiB across 22 launches; `--fit on` leaves 450–1 275 MiB free
   afterwards. A desktop app claiming VRAM mid-session can force driver eviction.
   Consider an explicit `--fit-target` margin in production.
5. **Temperature 1.0 only.** The planned 0.6 comparison (plan Phase B3) was not
   run; the head-to-head was prioritised.
6. **No OpenCode / OpenClink / real-repo run yet.** These are synthetic coding
   tasks, not multi-file agent work with a real tool loop.
7. **One measured instruction-drop.** A tool call omitted an instructed but
   non-required field (`notify: true`). n=1, not a conclusion, but schema
   validation would not catch it.
8. **CPU contention observed.** Session MCP servers held CPU during the run;
   two `expr_eval` samples dipped to 8.5–8.9 tok/s. Throughput otherwise matched
   the isolated matrix (10–11 tok/s), so contamination appears minor.

---

## 7. Recommended next steps, in order

1. **Context-depth sweep at 32K → 256K on both quants.** This is where Q3 could
   still win, and it is the only open question that could change the verdict.
2. **Prompt-cache / multi-turn behaviour**, since agent cost is prefix-driven.
3. **KV precision F16 vs Q8_0** at depth — directly buys context headroom.
4. `--fit-target` margins (512 / 1024 / 1536 MiB) for stability under desktop load.
5. Temperature 0.6 vs 1.0 on the winning config.
6. Then OpenCode integration, and only then the real Xeno workload.

Do **not** re-tune draft depth, re-test ngram, or revisit the quant question at
16K — those are settled by the data above.
