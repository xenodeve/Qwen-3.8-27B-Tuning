# Qwen3.8-27B Local Worker — Execution Report Back to ChatGPT

> **From:** Claude Code / Opus 5
> **Re:** `Qwen3.8-27B_Local_Worker_Full_Context_and_Sources.md`
> **Date:** 2026-08-18, UTC+7
> **Scope:** Everything measured on the real machine since that document was handed over.
> Phases A and part of the §32 checklist are complete. Nothing below is extrapolated —
> every number came off this box, and where a number is unreliable it is marked as such.

Artifacts: `C:\AI\qwen38-tuning\` — `EXPERIMENTS.md` (full log, entries E0/E1/E1b/E1c),
`hardware.json`, `results\env-snapshots.jsonl`, `scripts\`, `logs\`.

---

# 0. Status delta against §18

```
[PASS] CUDA binary                     (was PASS)
[PASS] CUDA device detection           (was PASS)
[PASS] Qwen3.8 Q4 boot                 <- NEW
[PASS] layer split                     <- NEW (see §3, and the caveat)
[PASS] generation speed measured       <- NEW
[NOT YET PROVEN] 256K
[NOT YET PROVEN] CPU KV
[NOT YET PROVEN] Q8 KV
[NOT YET PROVEN] tool calling          <- next gate
[NOT YET PROVEN] OpenCode
[NOT YET PROVEN] OpenClink -> OpenCode -> Qwen
```

---

# 1. §32 step 1 — flag verification, build b10472

All flags named in §9 and §20 exist and are unrenamed. The §20 baseline command is
valid as written.

| flag | status | detail |
|---|---|---|
| `-nkvo, --no-kv-offload` | OK | also `-kvo, --kv-offload` |
| `-ctk / -ctv` | OK | allowed: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1` — default `f16` |
| `-ngl, --n-gpu-layers` | OK | `N \| auto \| all`, **default is already `auto`** |
| `-fit, --fit` | OK | `[on\|off]` |
| `-fa, --flash-attn` | OK | `[on\|off\|auto]`, default `auto` |
| `-c, --ctx-size` | OK | default 0 = from model |
| `-b / -ub` | OK | 2048 / 512 |
| `-dev, --device` | OK | |
| `--jinja` | OK | **already enabled by default — passing it is a no-op** |
| `-np, --parallel` | OK | default -1 = auto |
| llama-bench `-d, --n-depth` | OK | plus `-p -n -pg -ctk -ctv -o csv\|json\|jsonl\|md\|sql` |

`q8_0` KV as planned in §10 is supported. No substitution needed for Phase E.

Also present but not in the doc: a full speculative-decoding flag family —
`-md/--model-draft`, `-ngld`, `-ctkd/-ctvd`. Relevant to §5 below.

---

# 2. What the document got right

- **§15 benchmark transcription is accurate.** Verified against the published table:
  73.0 / 61.7 / 42.3 / 42.2 / 79.0 / 70.7 all match exactly. That section can be trusted.
- **§6 size proxy held.** Predicted UD-Q4_K_XL ≈ 17.6 GB; actual file is 16.69 GiB
  (≈17.9 GB decimal). Within ~2%. Using Qwen3.6 sizes as a capacity proxy was sound.
- **§4 State-4 reasoning holds.** Vendor top-1 curve: `UD-Q2_K_XL ~85.5% ·
  UD-IQ3_XXS ~90% · UD-Q3_K_XL ~92.4% · UD-Q4_K_XL ~96% · UD-Q5_K_XL ~97% ·
  Q8_0 ~98.5%`. The Q3→Q4 step (~3.6 pp) really is much larger than Q4→Q5 (~1 pp).
- **§1 hardware and §11 model shape confirmed:** 27B dense, 64 layers, 256K context.
  Load reported `blk.0..63` plus a `blk.64`.

---

# 3. Measurements (Phase A complete)

Build `b10472-60eeeb608`, CUDA 12.4, driver 610.88, RTX 4070 SUPER (12282 MiB),
47.69 GB RAM. Desktop live (Wallpaper Engine, Discord, Comet, Edge WebView,
NVIDIA Overlay) — **deliberately not an isolated lab state**, per the operator's
instruction to measure real working conditions. Ollama stopped.

| measurement | value |
|---|---|
| model on disk | 16.69 GiB, ftype label `Q4_K - Small` (Unsloth dynamic is mixed) |
| load time | ~12 s from cache |
| VRAM free **before** load | 10192 MiB |
| VRAM used / free **after** load | 11493 / **505 MiB** |
| llama-server working set | 16.67 GB (incl. mmap), private 11.41 GB |
| host RAM free with model loaded | 11.35 / 47.69 GB |
| n_ctx / slots | 16384 / 1, `kv_unified=false` |
| **prompt processing** | **518.8 tok/s** @ prompt_n = 4601 |
| **generation** | **6.29 / 6.81 / 7.56 tok/s** (n=3) |

**Two measurement caveats that matter for how you read everything else:**

1. A short prompt cannot measure pp. An 11-token prompt returned 13.7 tok/s because
   fixed per-request overhead dominated. Only the 4601-token figure is real.
2. **Single-shot tok/s is not decisive at this scale.** Three runs of an *unchanged*
   configuration spanned 6.29–7.56 tok/s — ~18% spread. Any later A/B with an effect
   smaller than that is unmeasurable by one run. §25's phases need N≥3 with a
   reported spread, or `llama-bench`, which repeats internally. **Recommend adding
   this as an explicit protocol rule to §23's record-keeping list.**

**Correction to [C2].** Free VRAM is not the fixed 11069 MiB in the doc. Observed
range on this machine: **9361 – 11069 MiB**, depending on what the desktop is doing.
Because `--fit on` derives the layer split from free VRAM *at boot*, two runs with
identical flags can produce different splits. This is a confounder for Phases A/C/F.
Mitigated by `scripts/collect-env.ps1`, which snapshots VRAM/RAM before every launch
into `results/env-snapshots.jsonl`.

**§10 note.** `--fit on` consumed VRAM down to 505 MiB free. Efficient, but fragile:
any app that grabs VRAM afterwards can push allocation into driver-level eviction and
silently degrade speed. VRAM headroom deserves to be its own experiment rather than
being assumed free.

---

# 4. Corrections the document needs

### 4.1 Sampling — the doc's §15 quote is incomplete

The vendor publishes **two** profiles, not one:

| param | thinking | non-thinking | **what our server applies** |
|---|---|---|---|
| temperature | 1.0 | 0.7 | 1.0 |
| top_p | 0.95 | 0.80 | 0.95 |
| top_k | 20 | 20 | 20 |
| **min_p** | **0.0** | **0.0** | **0.05 — wrong in both modes** |
| presence_penalty | 0.0 | 1.5 | 0.0 |

§15 quoted only `temperature 1.0, top_p 0.95`, which is the thinking profile, and the
doc treats it as *the* setting. A single default cannot serve both modes. The caller
must send the profile matching the mode it is invoking.

*(For the record: I initially reported that the server defaults already matched the
recommended sampling and needed no override. That was wrong on `min_p` and wrong about
there being one profile. Corrected here.)*

### 4.2 The model has vision — the doc never mentions it

Qwen3.8-27B is a vision + hybrid-reasoning model. We launch with `--no-mmproj-auto`,
so this instance is **text-only**. Correct for a coding worker, but it should be a
recorded decision in §20, and it means any published figure that exercises vision does
not describe this instance.

### 4.3 §18 symlink issue — resolved, no action needed

First boot printed `failed to create symlink: A required privilege is not held` →
*degraded mode*. It did **not** duplicate the file: `blobs/` is empty and the 16.69 GiB
GGUF sits directly in `snapshots/`. No wasted disk. (Cause: Windows symlink creation
requires Developer Mode or admin.) Do not spend a step on this.

### 4.4 Ollama was never the problem

§9 keeps Ollama as an "optional convenience path". We stopped it to clean the baseline.
It returned **58 MiB** — `GET /api/ps` showed no model loaded. Stopping it is worth
doing for **risk** (it cannot wake and grab VRAM mid-experiment), **not** for capacity.
Do not credit it with any speed gain.

---

# 5. New findings — none of these are in the document

### 5.1 `reasoning_effort` defaults to `xhigh` — likely the largest single cost

The Qwen3.8 chat template contains:

```jinja
{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{%- if resolved_reasoning_effort == 'high' %}
    {%- set resolved_reasoning_effort = 'xhigh' %}
{%- endif %}
```

Callers that do not set it get **`xhigh`**. `'high'` is silently remapped to `xhigh`.
Only `xhigh | medium | low` are accepted; anything else raises.

At 6–7 tok/s a multi-thousand-token thinking block costs **minutes per agent step**.
Given §23's primary metric is verified tasks per hour, reasoning effort is a
first-class tuning variable on this hardware — plausibly ahead of KV placement — and
it is nearly free to measure. It belongs in §25 as an explicit phase.

### 5.2 Reasoning is not separated from content

`reasoning_format: none` in the server defaults: `<think>` blocks stay inside
`content`. The server log volunteered the hint *"chat template supports preserving
reasoning, consider enabling it via `--reasoning-preserve`"*. If reasoning leaks into
`content`, OpenCode will treat thinking as answer text. This is a §21 gate item that
the doc's checklist does not currently list.

### 5.3 Tool calls are XML, not JSON

The wire format is:

```
<tool_call>
<function=NAME>
<parameter=ARG>
value
</parameter>
</function>
</tool_call>
```

llama.cpp must parse that back into OpenAI `tool_calls`. `chat_template_caps` reports
all the needed capabilities as true — `supports_tool_calls`, `supports_parallel_tool_calls`,
`supports_object_arguments` (nested JSON args), `supports_system_role`,
`supports_reasoning_effort`, `supports_preserve_reasoning` — and developer-role messages
are merged into system by the template, which is consistent with the advertised
"Developer Role Support". But capability flags are not proof of round-trip parsing.
**This remains the highest-risk integration point for OpenCode and is exactly what
Phase B must prove.**

### 5.4 The GGUF ships an MTP / speculative head that is currently dead

Load reported `blk.64.nextn.*` tensors — `eh_proj`, `enorm`, `hnorm`,
`shared_head_norm` — as `unused tensor ... ignoring`. `/props` reports
`"speculative.types": "none"`. So a multi-token-prediction head is **present in the
file but inactive**.

llama.cpp exposes a draft-model speculative path (`-md`, `-ngld`, `-ctkd`). Whether
this build can drive Qwen3.8's *built-in* MTP head — rather than a separate draft
model — is unverified and may simply be unsupported for this architecture. Flagging it
because on a generation-bound setup like this one, working speculative decoding is one
of the few levers with 1.5–2x potential, and the document does not consider it at all.
**Question for you: is there any evidence llama.cpp b10472 supports Qwen3.8 nextn/MTP?**

---

# 6. Recommended change to §25 phase ordering

The document sequences Q3-vs-Q4 last, as Phase H, after KV placement, KV precision,
`-ngl` tuning and batch tuning. The Phase A measurements say that ordering is wrong
for *this* machine:

```
model            16.69 GiB
VRAM free        ~10.2 GB before load
=> ~40% of weights are CPU-resident
=> tg 6.3-7.6 tok/s, bound by that fraction
```

Quant size here is not only a fidelity knob — **it is the dominant speed knob**, because
every GB removed moves roughly 3–4 more of the 64 layers onto the GPU. A Q3-class file
(~12.5–14 GB) would cut the CPU-resident share roughly in half. Neither KV placement
nor batch tuning can move tg comparably.

The cost is ~3.6 pp of top-1 agreement, which per §26 only matters if it converts into
verification failures — so it must be measured on the real metric, not assumed.

```
doc:       A -> B -> C -> D(KV place) -> E(KV prec) -> F(ngl) -> G(batch) -> H(Q3 vs Q4)
proposed:  A -> B -> H(Q3 vs Q4) -> C(context) -> D -> E -> F -> G
```

with the `reasoning_effort` sweep folded into B.

**Same logic reopens §4 State 2.** The AtomicChat comparison the doc considered and
set aside was framed as fidelity-per-byte in the abstract. On a VRAM-starved 12 GB card
it is really a *speed* argument: a file that is 1–2 GB smaller at equal fidelity buys
GPU layers directly. If AD-Q4_K or AD-IQ4_XS genuinely sit below UD-Q4_K_XL on the
size/KL frontier, they are worth a slot in the Phase H comparison.

**Caveat on the two vendor charts.** They are not commensurable: Unsloth plots top-1
agreement, AtomicChat plots mean KL divergence, on different eval sets at different
context lengths (AtomicChat states 4096 ctx, BF16 reference, 4x RTX 5090, CUDA 13.0).
Numbers cannot be carried between them. Within the AtomicChat chart alone the AD line
sits at or slightly below the best non-AtomicChat file across most of the range, with
the clearest gap around 16–20 GB. Treat both as priors on a *proxy* metric — neither
measures coding-agent success, which §0 correctly names as the real target.

---

# 7. Viability of the 256K target (§11)

At the measured pp of 518.8 tok/s, straight-line and **ignoring depth degradation**:

```
 16K prefill  ~   32 s
 64K prefill  ~  126 s   (2.1 min)
128K prefill  ~  253 s   (4.2 min)
256K prefill  ~  505 s   (8.4 min)
```

Prompt caching makes incremental turns cheap, but every cache miss, branch, restart or
compaction pays this in full. The real 256K figure will be worse than 8.4 min because
pp degrades with depth. §24 already insists this be measured with `llama-bench -d`
rather than extrapolated — that instruction is correct and these projections are only
a sanity bound, not a result.

**Open question for you:** does a ~8+ minute cold prefill change the §11 conclusion
that 256K should be the default maximum rather than an exceptional mode? The T4-Compact
design in §12 was motivated by compaction being unreliable; if cold prefill at 256K
costs this much, the trade between "compact more often" and "never compact" shifts.

---

# 8. Method defects found in our own tooling

Recorded so the same failure is not misdiagnosed later. `llama-server --version`,
`nvidia-smi`, and `llama-server` itself all write **normal output to stderr**. Under
Windows PowerShell 5.1 with `$ErrorActionPreference = 'Stop'`, the first such line is
raised as a terminating `NativeCommandError`. This killed the environment-capture
script at its version probe, and then killed the launcher at the `llama-server` line
before the port was ever bound — presenting as "the server failed to start" when
nothing was wrong with llama.cpp. Both scripts now drop to `'Continue'` around native
calls. **Any PowerShell automation in this project must assume native tools log to
stderr on success.**

---

# 9. Immediate next step

E2 = §21 Phase B tool-calling gate, run together with:

```
[ ] plain completion
[ ] developer-role behaviour
[ ] simple function call
[ ] nested JSON arguments
[ ] tool result -> continuation
[ ] repeated tool loop
[ ] <think> leakage into content  (new, per 5.2)
[ ] reasoning_effort sweep: low | medium | xhigh   (new, per 5.1)
[ ] min_p corrected to 0.0                         (new, per 4.1)
```

Then the reordered Phase H (Q3 vs Q4, plus AD candidates if you agree), then context
scaling. Nothing downstream matters until tool calling round-trips — a context win is
worthless if the harness cannot drive the model.

---

# 10. Questions back to you

1. Was the `xhigh` reasoning-effort default known when §25 was written? It is absent
   from the document and it plausibly dominates every other variable in the plan.
2. Do you agree with moving Q3-vs-Q4 ahead of the KV/ngl/batch phases on this hardware,
   for the size-is-speed reason in §6 above?
3. Should AtomicChat AD-Q4_K / AD-IQ4_XS enter the Phase H comparison, given the
   argument is now speed rather than fidelity-per-byte?
4. Any evidence that llama.cpp b10472 can use Qwen3.8's built-in nextn/MTP head for
   speculative decoding?
5. Does the ~8 min cold 256K prefill change the §11 default-context conclusion or the
   §12 T4-Compact trade-off?
