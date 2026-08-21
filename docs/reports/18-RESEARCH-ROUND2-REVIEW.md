# Review of the Second Research Reply — 10 Workstreams

> **Date:** 2026-08-20 UTC+7
> **Subject:** the reply to the final dispatched plan
> ([`../plans/02-RESEARCH-BRIEF-OPTIMIZATION-SURFACE.md`](../plans/02-RESEARCH-BRIEF-OPTIMIZATION-SURFACE.md) §10)
> **Predecessor:** [report 17](17-EXTERNAL-RESEARCH-REVIEW.md) reviewed round 1.
>
> **Everything below marked ✅ verified was checked on this machine today** —
> argument parsing against build 10472, tensor names from real loader logs,
> logical-processor performance classes, and the Hugging Face API. Nothing in
> this review is taken on trust from either side.

---

## 0. Verdict

**A large improvement over round 1, and it earns its keep.** It answers questions
that were asked, in the requested format, and several answers are concrete
enough to act on. Two failure modes survive:

| | round 1 | round 2 |
|---|---|---|
| Workstreams addressed | ~4 of 10, generically | **10 of 10** |
| Answers concrete enough to run | 0 | **6** |
| Resolvable URLs | none | **none** — `[cite: 1]` markers again |
| Unsourced speedup multipliers | throughout | **still throughout** |
| Fabricated artifacts | 1 (a model that does not exist) | **2 of 3 rows in the model table** |

**Net: adopt the mechanisms, discard every percentage, re-derive the model table
from scratch.**

---

## 1. Verified on this machine — these are now facts, not claims

### 1.1 All four proposed flags parse on build 10472

```text
--spec-draft-device none                             accepted
--override-tensor "blk\.(5[0-9]|6[0-5])\.ffn_.*=CPU" accepted
--ctx-checkpoints 8                                  accepted
--backend-sampling                                   accepted
```

Each reached model loading rather than erroring at argument parse. **No source
build is needed for workstreams 1–8.** That was the single most important thing
to establish and it is established.

### 1.2 The `-ot` regex matches real tensor names — with one gap the reply missed

Tensor names pulled from an actual loader log for this model:

```text
attn_gate  attn_k  attn_k_norm  attn_norm  attn_output  attn_q  attn_q_norm
attn_qkv   attn_v  ffn_down  ffn_gate  ffn_up  post_attention_norm
ssm_a  ssm_alpha  ssm_beta  ssm_conv1d  ssm_dt  ssm_norm  ssm_out
```

`ffn_down`, `ffn_gate`, `ffn_up` exist, so `blk\.N\.ffn_.*=CPU` is well-formed.

**But this model is hybrid Gated-DeltaNet and carries seven `ssm_*` tensors per
block, which the reply's scheme does not mention at all.** Its taxonomy is
"attention on GPU, FFN on CPU" — a dense-transformer framing. Here, a third
tensor family exists and is unassigned, so `ffn_.*=CPU` silently leaves the SSM
state path wherever `--fit` put it. That is not necessarily wrong, but it is
unexamined, and it is the part of the architecture most likely to be
latency-sensitive.

**Action:** any `-ot` experiment here must state where `ssm_*` goes, as a third
arm.

### 1.3 P-core / E-core numbering — the reply is correct

`PercentProcessorPerformance` per logical processor:

| logical processors | performance | class |
|---|---|---|
| **0–11** | 144–179 | **P-cores** (boosting above nominal) |
| **12–19** | 99–139 | **E-cores** |

The split falls exactly at index 12. **`--cpu-mask 0x0FFF` = logical processors
0–11 = the six P-cores** is correct as stated. This was one of the specific
things the brief asked for and could not determine alone. ✅

---

## 2. Fabricated — the model table is mostly invented

Workstream 9 asked for "repo, file, bytes, SHA". Checked against
`https://huggingface.co/api/models/<repo>/tree/main` today:

| reply's claim | reality |
|---|---|
| `Qwen/Qwen2.5-Coder-32B-Instruct-GGUF` → `qwen2.5-coder-32b-instruct-iq2_m.gguf`, **9,845,211,136 B** | **The file does not exist.** That repo publishes **no IQ quants at all** — only fp16, q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k. Its smallest single file is `q2_k` at **12,313,098,432 B = 11.47 GiB**, which does **not** fit beside any KV cache on a 12,282 MiB card |
| `unsloth/Qwen2.5-Coder-14B-Instruct-GGUF` → `Q4_K_M.gguf`, **8,981,123,072 B** | File exists. Byte count is **8,988,110,240** — wrong by ~7 MB. Close enough to look right, wrong enough to fail the exact-identity check that criterion 4 exists to enforce |
| `bigcode/starcoder2-15b-instruct-gguf` → `starcoder2-15b-instruct-q4_k_m.gguf`, 9,120,442,112 B | Repo returns **HTTP 401** — not publicly available. Byte count and SHA fabricated |
| SHA digests `c83f91…`, `f1a823…`, `a942b1…` | **Truncated to six hex characters.** Criterion 4 asked for a revision SHA precisely so an artifact can be pinned; a six-character stub cannot pin anything |

**One row of three survives contact, and even that one has the wrong byte
count.** The priority table then ranks "Model Swap (Qwen2.5-Coder-32B IQ2_M)" at
#8 with "+10 % ถึง +30 %" — a recommendation to adopt a file that does not exist.

There is also a judgement problem independent of the fabrication: recommending
**Qwen2.5**-Coder in 2026 when this project already runs **Qwen3.8** was not
justified anywhere in the reply.

---

## 3. Unsourced multipliers — the round-1 failure, repeated

Criterion 1 said: *mechanism rather than an unsourced speedup multiplier.* The
reply gives the mechanism **and then attaches a number anyway**, with no source:

| claim | problem |
|---|---|
| GBNF grammar → **+150 % to +300 %** tasks/hour | labelled "Tier A/C (Measured/Paper)" with no citation of either |
| Drafter on CPU → **+70 % to +85 %** | derived from "+100 % over the residency cliff" minus "10–15 % CPU draft latency". **Both inputs are invented.** Our own cliff figure (21.8 → 41.3) compares *two different artifacts*; the same-artifact 61+4 → 65+0 number does not exist in our data or theirs |
| `-ot` FFN split → **+20 % to +35 %**, "PCIe bandwidth 60 % lower" | no measurement, no model of the transfer volume |
| imatrix on code corpus → **+15–22 %** syntactic accuracy | no citation |
| HAGS + `CUDA_MODULE_LOADING=EAGER` → drift **13.6 % → 5–7 %** | our 13.6 % comes from free-VRAM variation at boot across 25 boots; nothing in the reply explains how those settings would change that distribution |
| DDR5-7000 → **+25–30 %** on CPU offload | no baseline stated — faster than what? |
| P-core affinity → **+14–18 %** | the one number placed *just above* our 13.6 % floor, with no measurement behind it |
| `ik_llama.cpp` I-Quant kernels **8–12 %** faster | **below our noise floor**, which the reply does not flag despite criterion 8 |

**Every one of these is testable on this machine in one boot.** That is the
correct disposition: keep the mechanism, delete the number, measure it.

---

## 4. Genuinely new and usable

Ranked by how much they change what we do.

### 4.1 `--override-tensor` syntax — the concrete answer round 1 refused to give

```text
-ot "blk\.(5[0-9]|6[0-5])\.ffn_.*=CPU"
```

Regex over tensor name → buffer type. Verified to parse, and verified against
real tensor names (§1.2). This is the tool for the layer report 16 called the
largest specific gap, and we now know how to invoke it.

### 4.2 `--spec-draft-device none` — the fix for our own MTP conclusion

Report 15 §2.1 concluded "MTP does not pay on a resident target". That was
measured with the drafter on the GPU. `--spec-draft-device none` puts the drafter
on CPU, so the target keeps its layers. **The mechanism is sound and the flag
parses.** The +70–85 % attached to it is not evidence, but the experiment is now
well-defined and costs one paired round.

### 4.3 `/props` cannot change sampler parameters — a useful negative

The brief hoped `--props` would allow within-boot A/B of sampling settings,
which would sidestep the 13.6 % restart drift. The reply says it exposes state
(sleeping, system-prompt defaults) but **not** live sampler changes.

If true, the only remaining within-boot mechanism is
`--lora-init-without-apply` + the `/lora-adapters` endpoint. **Worth one direct
check against the `/props` endpoint rather than accepting it**, because the
consequence — every sampling comparison must cross a boot — is expensive.

### 4.4 `--slot-save-path` semantics

Save/restore via `/slots/{id}?action=save|restore`, valid while model and `-c`
are unchanged. If it works, an 11-minute 256K cold prefill becomes an NVMe read.
**The 660 s → 3–5 s figure is unsourced**, but the mechanism is checkable in one
boot and the upside is large enough to check first.

### 4.5 Do not override RoPE/YaRN on a natively-long model

For a model trained at 262K, forcing `--rope-scaling` or `--yarn-*` distorts
positional embedding and breaks retrieval. This matches the prior in report 16
§8 and **removes a whole workstream from the queue** — which is worth as much as
adding one. Still uncited, but the direction is consistent and cheap to respect.

### 4.6 Two anti-loop specifics worth keeping

- **`--repeat-last-n 64` is structurally unable to catch our loops.** Our loops
  run 19,280–33,871 characters ≈ 4,000–8,000 tokens. A 64-token window cannot
  see them. **This is arithmetic, not a claim**, and it is correct.
- **`--repeat-penalty` is dangerous for code** — it penalises `self.`, `return`,
  indentation. **DRY penalises repeated *sequences*** and does not. This matches
  the reasoning in report 16 §13 and is the right shape of answer.

### 4.7 Asymmetric KV rationale

K matters more than V for positional precision, so `-ctk q8_0 -ctv q4_0` keeps
retrieval quality nearer f16 while saving cache. Mechanism plausible; the "~25 %"
is unsourced.

**Tested 2026-08-20 and it does not work on this build.** At 131,072 the mixed
pair spent **over 65 minutes at 2–20 % GPU** on a prefill the symmetric `q4_0`
arm completed in 105.6 s, then died on a socket timeout. No kernel exists for
K and V at different types here — the same fallback signature as `q5_1`
(144–170 tok/s prefill against 1,180). Recorded in report 15 §7.1.

The reply did not mention that the pair needs kernel support, and neither did
this project's own kernel screen, which had only ever tested K and V at the
same type. **Both missed the same assumption.**

---

## 5. Claims that contradict our own measurements

| reply | our data |
|---|---|
| "IQ1_S **0/31** accepted tasks" | The V3 `IQ1_S` corpus was aborted at **6 records** (0 accepted, 2 censored) after the answer screen rejected it. 0/31 was never run |
| `-fa auto` "falls back to standard attention for q5_1 and iq4_nl", causing 144–170 tok/s prefill | The 144–170 tok/s figure is **ours**, from report 15 §7.1. The *explanation* — that it is `-fa auto` falling back — is the reply's, and is plausible but **unverified**. It is also directly testable with `-fa on` forced, which is exactly the validity check report 16 §9 asked for |
| DFlash 2 ranked #9, "do not test yet", VRAM ~1.1 GiB risks displacing the target | Consistent with our numbers — **but it contradicts the reply's own workstream 2**, which says to put drafters on CPU with `--spec-draft-device none`. If that works for MTP it should be tried for DFlash 2 before writing it off |
| ExLlamaV3 "supports EXL2 only" | ExLlamaV3's format is **EXL3**. Minor, but it signals the engine comparison was not checked |

---

## 6. What actually changes in the queue

Nothing in this reply reorders report 16 §17's top three. It **sharpens** them.

| # | action | change from report 16 | cost |
|---|---|---|---|
| 1 | `--grammar` on V3 `IQ1_S` | unchanged — still first | 1 corpus |
| 2 | Anti-loop: **DRY first**, `--repeat-last-n ≥ 4096` if repeat-penalty is used at all | **sharpened** — the 64-token default is arithmetically useless, and repeat-penalty is now a second choice rather than a peer | 1 screen |
| 3 | `--spec-draft-device none` — re-test MTP with the drafter on CPU | **promoted** from "test later" to top-3; it directly reopens a conclusion we published | 1 paired round |
| 4 | `-ot` FFN→CPU on an arm at 58+7, **with `ssm_*` placement stated as a third arm** | **now runnable** — we have the syntax and the real tensor names | 1 boot |
| 5 | `--ctx-checkpoints 8` vs 32 at 128K/256K | unchanged; the claimed ~900 MiB saving is one boot to confirm | 1 boot |
| 6 | `--slot-save-path` warm start | **promoted** — mechanism now specified end-to-end | 1 boot |
| 7 | `--cpu-mask 0x0FFF` for decode, `0xFFFFF` for prefill, on `Q4_K_XL` at 33+32 | **now runnable** — core numbering verified | 1 boot |
| — | RoPE/YaRN override at depth | **dropped** — override is contraindicated on a natively-long model | — |
| — | Model swap to Qwen2.5-Coder | **rejected** — the recommended artifact does not exist and the repo's smallest file is 11.47 GiB | — |
| — | `ik_llama.cpp` | **deprioritised** — its own claimed 8–12 % is below our noise floor | — |

---

## 7. For the next round, if there is one

The plan and the acceptance criteria were both good. The criteria were **not
enforced by the responder**, and no wording will fix that. So:

1. **Stop asking for artifact tables.** Two of three rows were invented across
   two rounds. Artifact identity is cheap to resolve locally against the HF API
   and expensive to un-believe. Ask for *repo names to check*, not byte counts.
2. **Ask for the mechanism and explicitly forbid a number.** "State the
   mechanism. Do not state an expected percentage — I will measure it." A blank
   where a fabricated number would go is strictly more useful.
3. **Ask for one URL per section, not per claim.** Per-claim citation was
   ignored twice; a single "what did you read for this section" may survive.

What this round proves is that the format works: the six items in §4 are real
answers to real questions, and they came back because the questions were
specific and the layer map was complete. The numbers attached to them are
decoration and should be treated as absent.
