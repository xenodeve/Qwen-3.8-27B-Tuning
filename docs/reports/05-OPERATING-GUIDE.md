# Operating Guide — What to Actually Run

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **Date:** 2026-08-18 UTC+7
> **Scripts:** `C:\AI\qwen38-tuning\scripts\`
> Everything here is measured; the reports behind each number are cited.

---

## 1. Pick a profile by working context

**Revised 2026-08-19** — Experiment A (report 10) replaced the 16K default.

| working context | script | measured |
|---|---|---|
| **16K — everyday** | **`production-iq2xxs.ps1`** | **42.4 tok/s · 818 tok/s prefill · 27/30 accepted · 60.8 verified tasks/hr **at `max_tokens 3072`** — the comparable 8,192 figure is **48.5 verified / 26.5 merged**, same 90 % accept** |
| 16K — escalation lane | `production-q4-tuned.ps1` | 12.6–13.7 tok/s · 27/30 accepted · 24.2 verified tasks/hr |
| **64K** | `production-iq2xxs.ps1 -Ctx 65536` + `-ctk q8_0 -ctv q8_0` | **15.81 tok/s** · 64 s cold prefill · split 61+4 |
| **128K** | same, `-Ctx 131072` | **5.15 tok/s** · 196 s cold prefill · split 47+18 |
| 256K — one deep question, not a loop | same, `-Ctx 262144` | **1.71 tok/s** · **11-minute cold prefill** · split 31+34 · no host paging |

Deep-context quality (`30/30` at 64K, `10/10` at a 114K prompt) was verified on
**Q4**, in report 03. The IQ2_XXS depth figures above are throughput and
residency only — **retrieval quality at depth has not been re-verified on this
artifact.** For work that depends on finding one fact in 100K tokens, use
`production-q4-deep.ps1` until it has been.

Everyday profile:

```powershell
C:\AI\llama.cpp-cuda\llama-server.exe `
  -hf unsloth/Qwen3.8-27B-GGUF:UD-IQ2_XXS `
  --alias qwen38-iq2xxs -c 16384 `
  -ngl auto --fit on --fit-target 768 -fa on -np 1 `
  -t 18 -b 2048 -ub 256 `
  --no-mmproj-auto `
  --host 127.0.0.1 --port 8080
```

**No `--spec-type` on the IQ2 profile.** MTP is a net loss once the target is
resident: its draft head costs VRAM, which pushes target layers back onto the
CPU. Measured −7 % on `UD-Q2_K_XL` (61+4 became 55+10). Keep MTP for Q4, where
the expensive CPU-resident forward pass is exactly what speculation compensates
for (reports 01 and 10 §1).

**Budget more output tokens for the low-bit profiles.** IQ2_XXS was truncated on
7 of 30 corpus attempts at `max_tokens 3072`, against Q4's 3, and a 1024-token
budget made its tool-calling look broken when it was not. A client tuned for Q4's
token appetite will misread these models (report 10 §2).

The 64K and 128K rows are **Q4 only**. Depth has not been measured on any low-bit
artifact; a resident model has far more room for KV and the answer may well
change, but it has not been checked.

The 64K profile is identical to the Q4 everyday one plus `-c 65536 -ctk q8_0
-ctv q8_0`.

**Do not add `-ctk/-ctv q8_0` at 16K.** Measured there: 86.7 % vs 90.0 % pass and
*slower*. At 512 MiB of KV there is nothing to reclaim and only the cost remains
(report 02 §3.2).

---

## 2. The rule that matters more than every flag

**The prefix cache is exact. Freeze everything above the append point.**

| change above the append point | cache kept | cost at 4K |
|---|---|---|
| reorder tool schemas | **0 %** | 11.1 s |
| edit one sentence of the system prompt | **0 %** | 11.5 s |
| prepend a skill block | **0 %** | 12.1 s |
| append only | **100 %** | **2.4 s** |

Append-only agent turns evaluate **~40 tokens instead of ~3 900**. Any edit above
the append point costs exactly as much as having no cache at all — and that cost
scales with context: ~11 s at 4K, roughly **two minutes at 64K**.

For the OpenCode / Xeno integration this means:

- **Stable tool-schema order** — do not let the client re-serialize tools between turns.
- **Byte-stable system prompt** — one changed word invalidates everything.
- **Skills injected once, at the start** — never prepended or reordered later.

The entire runtime-flag stack is worth **+6.6 – 9.6 %**. A preserved cache is worth
**5×** on a turn. Get the serialization right first.

---

## 3. Client-side settings the server will not supply correctly

| setting | value | why |
|---|---|---|
| `min_p` | **0.0** | server default is 0.05; the vendor specifies 0.0 for **both** modes |
| `temperature` / `top_p` | 1.0 / 0.95 thinking · 0.7 / 0.80 non-thinking | two published profiles; one server default cannot serve both |
| `presence_penalty` | 0.0 thinking · 1.5 non-thinking | same split |
| `chat_template_kwargs.reasoning_effort` | send explicitly | the template defaults to **`xhigh`** and silently remaps `high` → `xhigh`; accepts only `low`, `medium`, `xhigh` |

Operational reasoning profile: **`medium`**.

`--jinja` is a no-op — already the default in b10472.

---

## 4. Protocol facts worth knowing

- **Tool calls round-trip correctly.** The wire format is XML
  (`<tool_call><function=NAME><parameter=ARG>…`), and llama.cpp converts it into
  OpenAI `tool_calls` including nested objects, arrays, multi-round loops and
  `tool_call_id` correlation. All 8 protocol gates passed.
- **Reasoning is separated.** `/v1/chat/completions` returns `reasoning_content`
  distinct from `content`. The `reasoning_format: none` visible in `/props`
  governs the raw `/completion` endpoint only. No `--reasoning-preserve` needed.
- **`/props` cannot confirm MTP is active** — it reports
  `speculative.types = none` even when MTP is running. Check the load log for
  `common_speculative_init_result: creating MTP draft context`.
- **`-tb` is on the decode path when MTP is on.** Speculative verification is a
  batched op, so lowering the batch thread count cut *generation* from 13.42 to
  12.71. Leave `-tb` unset so it follows `-t`.
- **One instruction-drop observed.** A tool call omitted an instructed but
  non-`required` field. Put semantically required fields in `required` — a schema
  validator will not catch the omission otherwise.

---

## 5. Operational hygiene

- **Stop Ollama before benchmarking** — not for VRAM (it returned 58 MiB) but so it
  cannot wake and take VRAM mid-run.
- **Snapshot the environment before every launch** (`scripts\collect-env.ps1`).
  Free VRAM ranged 9 933 – 10 530 MiB across 22 launches, and `--fit` decides the
  layer split from whatever is free at boot.
- **Do not chase VRAM headroom to zero.** `--fit-target 256` left 345 MiB free and
  produced *intermittent* collapse (`[6.70, 8.28, 11.57]`), not steady slowness —
  driver eviction. 768 is the measured balance.
- **The Vulkan winget build is still installed** and is not the tuning backend.
  CUDA at `C:\AI\llama.cpp-cuda` is.

---

## 6. Settled — do not re-litigate

| question | answer | where |
|---|---|---|
| Q4 or Q3? | **Q4**, at every depth measured | 00, 02 |
| Draft depth? | **n=2** | 00 §3.1 |
| ngram speculation? | **no** — 30.8 % acceptance, no gain | 00 §3 |
| speculative sub-knobs? | **no** — neutral to −10 % | 04 §1 |
| threads / batch / fit-target? | **settled**; further 16K flag tuning is below the noise floor | 01 §7 |
| `FA_ALL_QUANTS` rebuild for Q8 KV? | **not needed** — Q8 is faster on the stock binary | 02 §3.1 |
