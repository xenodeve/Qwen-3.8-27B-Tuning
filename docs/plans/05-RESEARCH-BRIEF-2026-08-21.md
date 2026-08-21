# 05 — Research brief, 2026-08-21

**For an external researcher.** Everything below was measured on the machine
described in §1. Where a number is an estimate or a hypothesis it says so.

**What we want back:** mechanisms and things to try, with the reasoning attached.
**Not** confident numbers — six external claims have already been measured wrong
here (see §7), and in every case the mechanism was sound and the figure was
invented. **Keep the mechanism, delete the number, and we will measure it.**

---

## 1. The machine and the target

| | |
|---|---|
| GPU | **RTX 4070 SUPER, 12 GB**, Windows 11 WDDM, CUDA 12 |
| Desktop overhead | **1,650–2,200 MiB** of VRAM held by the OS and whatever is open — it moves boot to boot |
| CPU / RAM | 18 threads used (`-t 18`), 64 GB system RAM |
| Runtime | `llama.cpp` **build 10472**, commit `60eeeb608` |
| Model | **Qwen3.8-27B**, 26.9 B params, `n_ctx_train = 262144` |
| Serving artifact | Unsloth Dynamic V3 `UD-IQ2_XXS`, **2.16 real bits/weight**, 6.77 GiB |
| Production harness | **OpenCode** driving the local `llama-server`, tool loop, writes files |

**The goal:** the most verified accepted coding tasks per hour, at 128K context
or deeper, with everything resident in VRAM.

**Current best measured, end to end through the real harness:**

```text
131,072 context, 65+0 (all layers resident), q4_0 KV, --spec-type ngram-mod
  decode          35-61 tok/s on real code, median 45
  prefix          5,377 tokens (OpenCode, lean profile)
  corpus          6/10 accepted, 16.5 accepted tasks/hour
```

---

## 2. Problem 1 — the model reasons until the budget is gone and emits nothing

**This is the blocker.** Everything else is optimisation.

Measured two ways, and they agree:

| harness | shape of failure | rate |
|---|---|---|
| single completion, "reply with one fenced block" | **no fenced block at all** — reasoned to the token cap | 41.7 % of attempts |
| OpenCode, tool loop, "write a file" | **no output at all** — 190–248 s each, then nothing | 3 of 10 tasks |

The task shape changed completely and the failure survived it. On the tool-call
probes the same pattern shows as an unfinished round trip:

```text
artifact       calls tool OK   completes the round trip   reasoning chars
iq2xxs (2 bit)     14/16              10/16                  55-14,496
iq1m               15/16               7/16                  35-11,020
q2kxl              13/16               9/16                  29-16,341
ornith9b (9B)      15/16              14/16                 129-450
bonsai-g64         14/16              13/16                 287-616
```

**The two artifacts that finish are the two whose reasoning never exceeds ~600
characters.** The ones that fail are the ones that sometimes write 11,000–16,000.

### What we have tried, and what it did

| treatment | contract pass | accepted | note |
|---|---|---|---|
| nothing (control) | 58.3 % | 19/27 | |
| `-rea off` (disable reasoning) | 58.0 % | 15/30 | **no help** — the model relocates its reasoning into prose outside the fence |
| `--grammar-file` + `-rea off` | **84.3 %** | 16/27 | format fixed, accepted fell |
| `--grammar-file` alone | never run | — | queued |
| `--reasoning-budget 0` | — | — | **does not end the block**: 24,709 chars alone; 0 content chars when paired with a grammar |
| `max_tokens` 3,072 → 8,192 | — | 15/31 → **27/31** | an undersized budget looks exactly like lost capability |

**A grammar constrains the output and cannot stop the model spending 8,192
tokens deciding.** That is consistent with contract rising while accepted fell.

### The hypothesis we cannot test here

A commenter on a public review of this exact model, running bf16 on far larger
hardware, reported that **q8 made it "doubt itself at every step, like a
paranoia" compared with bf16.** If quantization damage presents as *self-doubt*
rather than as wrong answers, our failure is the same effect four times further
down the bit ladder — and it is not a format problem at all.

**Questions we would like answered:**

1. Is there published or anecdotal evidence that **low-bit quantization
   lengthens reasoning traces** on reasoning-tuned models, as distinct from
   degrading answer quality? Qwen3.8 specifically, or the family.
2. Is there a **sampler-level** intervention that shortens reasoning without
   disabling it? We have swept temperature, top-k, top-p, min-p, DRY, mirostat,
   n-sigma, repeat penalties — **nothing moved above our 13.6 % noise floor.**
3. Does a **system-prompt instruction about how to think** work on this model?
   The same public thread claims *"don't hedge, make conclusions, work forward,
   don't reconsider"* fixes it. We have never tried instructing the *process*,
   only the output format. Any evidence either way.
4. Is there a way to **cap reasoning that actually works** in `llama.cpp` b10472?
   `--reasoning-budget 0` does not, and `-rea off` relocates rather than stops.
5. The same thread describes **injecting budget reminders into the reasoning
   stream** ("you have 12K tokens of thinking budget", updated at 50 % and
   75 %). Is that implementable server-side, or only in a harness?

---

## 3. Problem 2 — is the artifact the variable, not the flags?

Five artifacts, two vendors, sorted by **real** bits per weight from the loader's
tensor histogram:

| artifact | bpw | accepted | contract pass |
|---|---:|---|---|
| V3 `UD-IQ1_S` | **1.84** | 0 of 12 | no fenced block, 12/12 |
| V3 `UD-IQ1_M` | ~2.0 | 10/21 | 41.5 % |
| V3 `UD-IQ2_XXS` | **2.16** | 19/27 | 58.3 % |
| `AD-IQ1_M` (AtomicChat) | **2.49** | 27/30 | — |
| pre-V3 `UD-IQ2_XXS` | **2.64** | 27/30 (90 %) | — |

**Perfectly monotone.** Nothing else in this project correlates that cleanly —
not the flags, not the sampler, not the KV type. We spent two days trying to fix
a 2.16 bpw artifact with flags.

**Note the filenames lie.** `AD-IQ1_M` at 2.49 bpw is *heavier* than
`UD-IQ2_XXS` at 2.16: only 80 of its tensors are 1-bit and 128 are full `q8_0`.
V3 `UD-IQ2_XXS` contains **zero** `q8_0` tensors.

**Questions:**

1. Is there a known **threshold** below which reasoning-tuned models start
   looping, distinct from the threshold where answers get worse?
2. **Which tensors matter most** for this failure? If the fix is "keep the
   attention or the output head at higher precision", a custom mix at ~2.4 bpw
   that fits 12 GB at 128K would be the whole answer. Is there tooling to build
   one, and prior art on which tensors to protect?
3. Unsloth Dynamic V3 vs the AtomicChat mix vs a hand-rolled `llama-quantize`
   recipe — any measured comparison on reasoning behaviour rather than
   perplexity?
4. **Perplexity does not predict this.** Is there a cheap metric that does?

---

## 4. Problem 3 — speculative decoding behaves in three ways we cannot explain

`--spec-type ngram-*` is the largest free win we have: no VRAM, no drafter file,
byte-identical output. But:

**(a) The advantage depends heavily on how repetitive the text is.** Our own
benchmark prompt turned out to be **84.5 % duplicate lines**, which is close to
the best case an n-gram drafter can get. On that prompt it reported +200 % at
131,072. **On real code through OpenCode the same configuration gives ~1.8×,
not 3×.**

**(b) Acceptance collapses in some configurations and not others**, with the
same flag, same speculative settings:

| where | split | acceptance |
|---|---|---|
| `v3-iq2xxs` @163,840 + `-ot ssm` (10 blocks) | `65+0` | **4 %**, reproduced in 4 boots |
| `v3-iq2xxs` @163,840 + `-ot ssm` (4 blocks) | `65+0` | **drafts nothing at all** |
| `v3-iq1m` @196,608 + `-ot ssm` (10 blocks) | `65+0` | **100 %** |
| `v3-iq2xxs` @163,840, `--fit-target 192` (no `-ot`) | `65+0` | **drafts nothing** |
| `v3-iq2xxs` @163,840, `-ub 128` | `63+2` | **4 %** |

**Every arm that reaches full residency at 163,840 by any means loses
speculation.** `-ot` moves weights to the CPU; `--fit-target` and `-ub` do not.
So "CPU/GPU float divergence" cannot be the whole story.

**Questions:**

1. What in `llama.cpp` b10472 makes an n-gram drafter **produce no drafts at
   all**? A buffer that fails to allocate under a tight fit, and fails silently?
2. Is speculative decoding **allocating from the same headroom** `--fit-target`
   reserves? That would explain why freeing the reserve to gain a layer costs
   the drafter.
3. What actually determines n-gram acceptance on **real source code** — how
   should we predict the ~1.8× we measured rather than the 3× our synthetic
   prompt suggested?
4. Are there speculative decoders in this build better suited to non-repetitive
   code? We have tried all eleven `--spec-type` values; the drafter-model family
   (`draft-mtp`, `draft-dflash`, eagle3, dspark) all lose because holding
   weights competes with the layers on 12 GB.

---

## 5. Problem 4 — residency versus speculation at depth

At 163,840 on the artifact worth having, we can have full residency **or**
working speculation, not both:

```text
  ngram-mod, 62+3 (three layers on CPU)   38.65 tok/s   acceptance 100 %
  fully resident 65+0, any route          21-48 tok/s   acceptance 0-50 %
```

A CPU layer at depth is expensive — measured at **22 %** of decode for three
layers, and `AD-IQ1_M` at `65+1` on 131,072 decodes at **6.08 tok/s** against a
resident 26.50. So residency normally wins by a lot. Here it does not, because
of §4(b).

**We need about 576 MiB** to hold `65+0` at 163,840. Routes tried:

| route | frees | outcome |
|---|---|---|
| `-ot ffn_` (1 block) | 644 MiB | prefill **240.6 → 8.56 tok/s**, unusable |
| `-ot ssm_` (4 or 10 blocks) | ~168 MiB | reaches `65+0`, kills speculation |
| `--fit-target` 768 → 192 | ~576 MiB | **reaches `65+0`**, kills speculation |
| `-ub` 256 → 128 | some | `63+2`, slower |
| `--ctx-checkpoints 8` | **10–16 MiB** | external research claimed ~900 MiB |
| the desktop's 1,650–2,200 MiB | — | **never tested** — largest untouched lever |

**Questions:**

1. What is `--fit-target` actually reserving, and what breaks when it is too
   small? The documentation we have found does not say.
2. Is there a supported way to run the display on an iGPU and give the whole
   12 GB to the model on Windows, or does WDDM always reserve?
3. Any other route to ~576 MiB that does not put weights on the CPU?

---

## 6. Problem 5 — the harness costs more than the model, and nobody documents it

**OpenCode's default configuration sends a 99,073-token prefix** — measured by
the gateway's own tokenizer, for the prompt *"say READY"*. On a 131,072 window
that plus its 32,000-token output reservation is 131,073: **one token over, so
the request failed before any work started.**

Breakdown and the reduction we found by reading the binary's flag table (none of
it is documented anywhere we could find):

| profile | tools | skills in prompt | prefix |
|---|---|---|---|
| as configured | 141 | 387 | **99,073** |
| MCP servers disabled | 10 | 387 | ~46,500 |
| `OPENCODE_DISABLE_CLAUDE_CODE=1` + `OPENCODE_CONFIG_DIR` | 6 | **0** | **~5,377** |

**94.5 % of the prefix was a skill catalogue and MCP tool schemas the worker
cannot use.** Verified still fully functional afterwards: it writes files, runs
bash, and its output passes our corpus tests.

**Questions:**

1. Is there a supported, documented way to configure a **minimal worker profile**
   in OpenCode? We found the env vars by searching strings in the binary.
2. OpenCode sends **no sampling parameters at all**, so `llama-server` defaults
   apply: `temperature 1.0`, `top_k 20`, `top_p 0.95`, `min_p 0.05`. Every
   quality number we hold from the older harness was measured at 0.7 or greedy.
   Is there a way to pin sampling per provider?
3. It reserves **32,000 output tokens** on every request. Can that be lowered
   per provider without breaking long edits?
4. Any harness with a materially smaller prefix that still has a real tool loop?

---

## 7. Please do not send us these — they were measured wrong here already

| external claim | what we measured |
|---|---|
| MoE CPU-offload is a large win; the artifact is 20.6 GiB | the artifact is **10.02 GiB**; the config lost **46–48 %** |
| retry success `p2 ≈ 0.93` | **0.20–0.625** |
| asymmetric KV (`q8_0`/`q4_0`) saves ~25 % VRAM | **no fast kernel in b10472** — prefill **29× slower**, cache **44 % larger** |
| `--ctx-checkpoints 8` frees ~900 MiB | frees **10–16 MiB** |
| a drafter on CPU gives +70–85 % | **−59 %** |
| a specific recommended GGUF, with a byte count | **the file does not exist** |

---

## 8. What we already know and do not need re-derived

- **`q4_0` KV is settled.** It buys residency; no other type in this build has a
  fast kernel.
- **Placement flags are inert**: thread affinity, process priority, polling
  strategy, GPU-side sampling — all within ±2.3 %, under our noise floor.
- **`-np > 1` is harmful, not inert** — it divides the context between slots.
- **Depth is bounded by VRAM, not the model.** `n_ctx_train = 262144`, no rope
  scaling engaged at 163,840.
- **Residency ceilings** (`q4_0` KV, all layers on GPU): `UD-IQ1_S` to 196,608,
  `UD-IQ1_M` to 163,840, `UD-IQ2_XXS` to **147,456**.
- **The prompt cache works and is worth a factor of five.** OpenCode's fixed
  prefix reprefills at 19–49 tokens per turn after the first. **Injecting text
  at the front of the prompt destroys it** — measured.
