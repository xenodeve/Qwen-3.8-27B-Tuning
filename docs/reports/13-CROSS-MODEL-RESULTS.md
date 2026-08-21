# Cross-Model and Cross-Quant Results — Everything Measured, One Table

> **Date:** 2026-08-20 UTC+7
> **Why this exists:** reports 00–11 cover the Qwen3.8-27B flag-tuning programme
> and report 12 covers Dynamic V3. The **cross-model sweep of 2026-08-19** — nine
> other artifacts across four model families, every one of them measured under
> the same paired design — existed only as raw JSONL and conversation. This is
> that data, read back out of `results/*.jsonl` rather than from memory.
> **Generation:** everything here except §5 is the **pre-V3** Unsloth build. See
> report 12 §0.

---

## 1. Every arm, at 16K, in one table

All figures from `results/arena-*.jsonl`. `split` is the final-pass GPU+CPU layer
assignment; `free` is VRAM remaining after load. Ranked by peak decode.

| arm | family / size | split | free VRAM | decode | prompt proc | boots |
|---|---|---|---:|---:|---:|---:|
| `ornith35moe` | Ornith-1.0-35B MoE, IQ2_XXS 10.71 GiB | 41 + 0 | **227–331** | 49.8 – **87.0** | 766–816 | 2 |
| `qwen36moe` | Qwen3.6-35B-A3B MoE, IQ2_XXS 10.02 GiB | 41 + 0 | **250–339** | 65.8 – **84.8** | 811–919 | 4 |
| `bonsai-1bit` | Ternary Bonsai 27B, Q1_0 **3.54 GiB** | 65 + 0 | **5,074–5,768** | 55.7 – **69.3** | 766–987 | 3 |
| `gptoss20b` | gpt-oss-20b, Q4_K_M 10.83 GiB | 25 + 0 | 255–413 | 33.0 – **68.7** | 516–808 | 4 |
| `ornith9b-nomtp` | Ornith-1.0-9B, Q6_K 6.85 GiB | 65 + 0 | 3,922–3,941 | **61.0–61.3** | **2,129–2,203** | 3 |
| `bonsai-g64` | Ternary Bonsai 27B, Q2_g64 7.06 GiB | 65 + 0 | 1,794–2,628 | 43.1–49.9 | 803–996 | 3 |
| `iq1m-nomtp` | AtomicChat AD-IQ1_M 7.91 GiB | 65 + 0 | 2,117 | 45.5–45.6 | 675–698 | 3 |
| `ornith9b-q8` | Ornith-1.0-9B, Q8_0 8.87 GiB | 33 + 0 | 1,120–1,814 | 41.7–45.0 | 1,746–2,016 | 4 |
| `iq2xxs-nomtp` | **control** — Qwen3.8-27B UD-IQ2_XXS 8.39 GiB | 66 + 0 | 453–1,545 | 32.4–42.5 | 560–829 | 25 |
| `adiq2xxs` | AtomicChat AD-IQ2_XXS 8.36 GiB | 65 + 0 | 700–858 | 38.7–40.1 | 750–767 | 3 |
| `qwen36moe-cpu` | same MoE, `--n-cpu-moe 34` | 41 + 0 | **7,101–7,106** | 30.4–41.1 | 255–283 | 2 |
| `ornith35-cpu` | same MoE, `--n-cpu-moe 34` | 41 + 0 | 6,912 | 36.7–37.6 | 268–280 | 2 |
| `q2kxl-nomtp` | Qwen3.8-27B UD-Q2_K_XL 9.94 GiB | **61 + 4** | 451–569 | 21.3–22.0 | 394–510 | 6 |
| `q2kxl-mtp2` | same, with MTP n=2 | **55 + 10** | 793–803 | 19.9 | 310–330 | 3 |
| `q4-tuned` | Qwen3.8-27B UD-Q4_K_XL 16.69 GiB | **33 + 32** | 236–931 | 12.6–13.7 | 142–168 | 7 |

**Raw decode is not comparable across rows** — the control alone spans 32.4 to
42.5 across 25 boots, because free VRAM at boot moved 9,326–10,732 MiB and the
`--fit` split follows it. The comparable column is §2.

---

## 2. Paired results — the column that means something

Alternating boots, order counterbalanced, paired by round. An effect is
`RESOLVED` only above the **13.6 %** restart-drift floor *and* with a consistent
sign in every round.

### Against the Qwen3.8-27B `UD-IQ2_XXS` control

| arm | mean | per round | verdict |
|---|---:|---|---|
| `qwen36moe` | **+99.01 %** | +123.69, +74.33 | RESOLVED |
| `ornith35moe` | **+79.51 %** | +129.42, +29.60 | RESOLVED |
| `bonsai-1bit` | **+80.12 %** | +79.21, +81.04 | RESOLVED |
| `gptoss20b` | **+68.44 %** | +57.50, +79.37 | RESOLVED |
| `ornith9b-nomtp` | **+44.37 %** | +43.98, +44.47, +44.65 | RESOLVED |
| `bonsai-g64` | **+17.66 %** | +17.71, +17.62 | RESOLVED |
| `ornith9b-q8` | **+16.75 %** | +16.39, +17.10 | RESOLVED |
| `iq1m-nomtp` | +7.61 % | +7.63, +7.59, +7.61 | under floor |
| `adiq2xxs` | +0.79 % | −6.39, +4.73, +4.02 | under floor |

An earlier `gptoss20b` round returned +28.28 % with the range −11.95 to +68.51 —
correctly refused as unresolved. It was measured while a download saturated the
machine; the clean re-run above is the usable one.

### Against the Qwen3.8-27B `UD-Q4_K_XL` production control

| arm | mean | per round | verdict |
|---|---:|---|---|
| `iq2xxs-nomtp` | **+219.58 %** | +237.36, +212.28, +209.10 | RESOLVED |
| `q2kxl-nomtp` | +62.12 % / +64.22 % | two independent arenas | RESOLVED |
| `q2kxl-mtp2` | +50.13 % | +50.42, +47.56, +52.41 | RESOLVED |

### MoE with CPU expert offload — the configuration the research proposed

| arm | vs the same MoE resident | verdict |
|---|---:|---|
| `qwen36moe-cpu` (`--n-cpu-moe 34`) | **−47.68 %** | RESOLVED |
| `ornith35-cpu` (`--n-cpu-moe 34`) | **−45.37 %** | RESOLVED |

The deep-research document proposed CPU expert offload because it costed the MoE
at Q4_K_M, **20.6 GiB**, which cannot fit this card. Unsloth ships the same model
at **10.02 GiB**, which can. Once the experts are resident, moving them to the
host is pure cost: **−46 to −48 % decode and −70 % prompt processing** (255–283
against 811–919 tok/s). The proposed configuration is the wrong one here, and
only because a size assumption was wrong.

---

## 3. What the table actually says

**Four families beat the Qwen3.8-27B control on raw decode**, and the two MoE
arms nearly double it. But three cautions decide how much of that is usable.

**The MoE arms ran in the eviction regime.** 227–339 MiB free, below the 512 MiB
reserve this project adopted after `--fit-target 256` destabilised at 345 MiB.
Their per-round spread shows it: `ornith35moe` returned +129.42 % in one round
and +29.60 % in the next, and its raw samples inside a single boot were
`[45.93, 49.78, 75.57]` — a 64 % spread. The harness called both arms RESOLVED
because the sign held, which is the rule working as designed and still not
enough. **Treat the MoE speed figures as directional.**

**`gptoss20b` loads only 25 layers.** Its split is `25 + 0`, not 41 or 65 — a
different architecture with far fewer blocks, so its layer count is not
comparable to anything else in the table and its 10.83 GiB leaves 255–413 MiB.

**Bonsai's 1-bit build is the outlier worth noticing.** At **3.54 GiB** it is the
smallest artifact on disk, it is fully resident, it leaves **5.1–5.8 GiB free** —
more than any other arm by 1.3 GiB — and it decodes at 55.7–69.3 tok/s,
**+80.12 % over the control, resolved.** For a goal of *"context beyond 128K"*
that headroom is the single most interesting number in this report.

---

## 4. Depth — measured at 128K and 256K

All from `results/kv-sweep*.jsonl`.

| artifact | ctx | KV | split | decode | cold prefill | KV size | free |
|---|---|---|---|---:|---:|---:|---:|
| **Ornith-9B Q6_K** | 128K | q8_0 | **65 + 0** | **46.6** | **34.2 s** | 2,176 MiB | 1,585–1,716 |
| **Ornith-9B Q6_K** | 128K | q4_0 | **65 + 0** | 45.8–46.1 | **33.8 s** | **1,152 MiB** | **2,609–2,740** |
| Bonsai `Q2_g64` | 128K | q4_0 | **65 + 0** | 28.0–28.1 | 93.0 s | 2,304 MiB | 635–667 |
| Bonsai `Q2_g64` | 128K | q8_0 | 51 + 14 | **4.89** | 148.3 s | 3,536 MiB | 483 |
| AtomicChat `AD-IQ1_M` | 128K | q4_0 | **65 + 0** | 23.8–24.0 | 125.1 s | 2,304 MiB | 557–621 |
| `UD-IQ2_XXS` control | 128K | q4_0 | 58 + 7 | 7.71–7.84 | 139.7 s | 2,016 MiB | 562–589 |
| `UD-IQ2_XXS` control | 128K | q8_0 | 47 + 18 | 4.98–5.22 | 192.0 s | 3,264 MiB | 401–624 |
| `UD-IQ2_XXS` + `--no-kv-offload` | 128K | q4_0 | 65 + 0 | 5.22–5.29 | 153.0 s | 2,304 MiB | 1,720–1,726 |
| `UD-IQ2_XXS` + `--no-kv-offload` | 128K | q8_0 | 65 + 0 | **3.23–3.59** | 175.1 s | 4,352 MiB | 1,591–1,687 |
| `UD-IQ2_XXS` | 256K | q4_0 | 43 + 22 | **2.23** | 476.9 s | 3,168 MiB | 563 |
| `UD-IQ2_XXS` | 256K | q8_0 | 31 + 34 | **1.76** | 633.7 s | 4,352 MiB | 496 |
| `AD-IQ1_M` | 256K | q4_0 | 46 + 19 | **2.29** | 471.7 s | 3,168 MiB | 555 |

### Three mechanisms, each measured twice

**KV type buys residency, not speed.** On the control at 128K, `q8_0 → q4_0` is
**+52.5 %** (paired, resolved) *because* it moves the split 47+18 → 58+7. On
Bonsai it is **+473 %** (4.89 → 28.07) because it moves 51+14 → 65+0. On
Ornith-9B, already 65+0 under `q8_0`, the same change is **+1.6 %** and
unresolved — there is nothing left to buy. Same flag, three outcomes, one
explanation.

**`--no-kv-offload` reaches full residency and still loses.** It achieves 65+0,
frees over 1.7 GiB, and costs **−33 %** (paired, resolved). Weight residency is
not the objective; **total bytes moved per token** is, and at depth the cache is
larger than the layers it displaced.

**A 9B holds a much smaller cache than a 27B at the same depth.** 1,152 MiB
against 2,016 at 128K with the same KV type — 43 % less, because fewer layers and
fewer heads. This is why Ornith-9B is the only artifact that reaches 128K without
paying anything for it: **34-second cold prefill against the control's 192 s**,
and 46.6 tok/s against 5.0.

---

## 5. Quality — the corpus, at a budget that does not lie

30 execution-verified tasks per arm (10 single-function Python tasks × 3 passes),
one evidence-assisted retry, `max_tokens 8192`. The 3072 budget used first
produced four wrong verdicts and is documented in report 12 §5.

| arm | p1 | p2 | accepted | truncated | worker wall | merged/h |
|---|---:|---:|---:|---:|---:|---:|
| `q4-matched` | 83.3 % | 40.0 % | **27/30** | 3 | 4,008.7 s | 17.8 |
| `iq2xxs` @8192 | 83.3 % | 40.0 % | **27/30** | 1 | **2,004.5 s** | 26.5 |
| `iq1m` @8192 | 76.7 % | 57.1 % | **27/30** | 4 | 2,755.9 s | 22.4 |
| `bonsai-g64` @8192 | 70.0 % | 66.7 % | **27/30** | 7 | 4,572.2 s | 16.3 |
| `ornith9b` @8192 | 70.0 % | 22.2 % | 23/30 | 4 | 1,652.9 s | 26.5 |

**Four arms tie at 27/30 and differ only in wall clock**, 2,004 s to 4,572 s.
That is verbosity, not capability, and it is why this project now reports
capability and throughput as separate numbers instead of multiplying them into
`merged_tasks_per_hour` and calling the product a ranking.

Ornith-9B is the one arm that is genuinely lower on acceptance — 23/30 at the
same budget with only 4 truncations. That is a 9B's capability ceiling, not a
probe artifact, and it is the honest cost of its speed.

### Protocol gate, `max_tokens 4096`, n=15

| arm | schema-correct call | round-trip | median reasoning | wall |
|---|---:|---:|---:|---:|
| **Ornith-9B** | **100 %** | **93.3 %** | 280 | **61.9 s** |
| Bonsai `Q2_g64` | 93.3 % | 86.7 % | 322 | 98.7 s |
| `IQ2_XXS` | 93.3 % | 66.7 % | 1,023 | 431.9 s |
| `AD-IQ1_M` | 100 % | 46.7 % | 170 | 215.4 s |
| `Q2_K_XL` | 86.7 % | 60.0 % | **2,811** | 1,121.8 s |
| `Q4_K_XL` | 80.0 % | 60.0 % | **59** | 1,367.1 s |

Ornith-9B's **93.3 % round-trip** is the figure that retires an earlier
conclusion. When only the Qwen3.8 arms had been measured they all scored 60–67 %
and it was written off as a property of the probe. One arm scoring 93.3 % on the
same probe says otherwise: **the 60–67 % is a real weakness of the Qwen3.8
quantizations at handling `tool_call_id` continuation**, not an instrument
artefact.

### Stability, 100 sequential turns with a forced prefix invalidation every tenth

| arm | survived | hangs | recovered | prefix reuse | empty replies |
|---|---|---:|---|---:|---:|
| `Q4_K_XL` | 100/100 | 0 | 9/9 | 99.1 % | 19 |
| `Q2_K_XL` | 100/100 | 0 | 9/9 | 99.0 % | **55** |
| `IQ2_XXS` | 100/100 | 0 | 9/9 | 99.2 % | 1 |
| **Ornith-9B** | 100/100 | 0 | 9/9 | **99.3 %** | **0** |
| **Bonsai `Q2_g64`** | 100/100 | 0 | 9/9 | **99.3 %** | **0** |

Every arm survives, every arm recovers its prefix on the turn after an
invalidation, and no arm hangs a slot — including both MoE candidates the
research warned about. The empty-reply counts are **not monotonic in bit-width**
(19 / 55 / 1 / 0 / 0), so the obvious explanation is wrong and no other has been
tested.

---

## 6. Where each candidate actually stands

| candidate | speed | quality | depth | verdict |
|---|---|---|---|---|
| **Ornith-9B Q6_K** | +44 % resolved | 23/30, best protocol scores of any arm | **best measured**: 65+0 at 128K, 34 s prefill, 46.6 tok/s | strongest deep-context candidate; a 9B's ceiling is the cost |
| **Bonsai `Q1_0` 3.54 GiB** | **+80 % resolved** | **untested** | **untested** | most headroom of anything measured (5.1–5.8 GiB free); the obvious next test for >128K |
| Bonsai `Q2_g64` | +17.7 % resolved | 27/30, 93.3 % schema | 65+0 at 128K with q4_0 only | ties the top on quality; slowest end-to-end (4,572 s) |
| `qwen36moe` / `ornith35moe` | +99 % / +80 % | **untested** | untested | measured in the eviction regime; figures directional only |
| `gptoss20b` | +68 % resolved | untested | untested | different architecture, 25 layers, 255–413 MiB free |
| `AD-IQ1_M` | +7.6 % under floor | 27/30 | 65+0 at 128K, 24 tok/s | matches the control; no reason to switch |
| `AD-IQ2_XXS` | +0.8 % under floor | untested | untested | **the quantizer battle is a tie** — AtomicChat and Unsloth at the same size are indistinguishable here |
| `Q2_K_XL` | +62 % over Q4 | 26/30 | 58+7 at 128K | dominated: never fully resident, 451–569 MiB free |

---

## 7. What none of this establishes

- **No deep-context retrieval quality on any artifact except Q4.** The `30/30` at
  64K and `10/10` at a 114K prompt belong to `UD-Q4_K_XL`. Every depth figure in
  §4 is throughput and residency. This is the project's largest open risk and
  nothing here reduces it.
- **The MoE arms have no quality measurement at all**, and their speed numbers
  were taken below the VRAM reserve.
- **Bonsai `Q1_0`, `gptoss20b` and `AD-IQ2_XXS` have no corpus result.**
- **27/30 against 27/30 is not equivalence** at n=30.
- **The corpus is ten single-function tasks.** It cannot see cross-file interface
  drift, and it repairs format violations before grading — `check_output_contract`
  now scores those separately, but only from 2026-08-20 onward.
