# The Context Ceiling — How Far Past 128K This Card Actually Goes

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **Correction, 2026-08-21 (report 24).** This report walks a ladder and records
> the deepest rung that loaded. That is not the same as the ceiling, and for
> `v3-iq2xxs` the difference matters: it records `65+0` at 131,072 and `62+3` at
> 163,840, and **147,456 — which holds `65+0` — was never tried.** The ladder
> steps are 32,768 apart, so every ceiling here is only as precise as that
> spacing. Read each as "at least this deep", never as "no deeper".


> **Renumbered 16 → 21 on 2026-08-20.** It was written the same night as
> `16-OPTIMIZATION-SURFACE.md` and both claimed number 16, so this one was
> invisible to the index for a day. Content unchanged.

> **Date:** 2026-08-20 UTC+7 · `bench/ctx_ceiling.py`, run 05:09:00–05:15:05
> **Question:** the developer restated the goal as **a usable context beyond
> 128K**. This is the run that answers the residency half of it.
> **Method:** five arms, ladder 128K → 160K → 192K → 224K → 256K, `q4_0` KV,
> one boot per rung reading only the layer split, stopping at the first spill.
> About a minute per rung against the ten a 256K cold prefill costs.

---

## 1. The ladder

`free` is MiB of free VRAM after the weights and the full cache are allocated.

| rung | `bonsai-1bit` | `ornith9b` | `v3-iq1s` | `v3-iq1m` | `iq2xxs` (control) |
|---|---|---|---|---|---|
| 128K | 65+0 · 3,584 | 33+0 · 1,929 | 65+0 · 1,552 | 65+0 · 943 | **58+8 · 428 spilled** |
| 160K | 65+0 · 2,606 | 33+0 · 1,497 | 65+0 · 820 | 65+0 · **365** | — |
| 192K | 65+0 · 1,924 | 33+0 · 1,065 | 65+0 · **226** | 60+5 spilled | — |
| 224K | 65+0 · 1,222 | 33+0 · 633 | 57+8 spilled | — | — |
| 256K | 65+0 · **513** | 33+0 · **444** | — | — | — |

**256K is fully resident on this card.** Two arms reach it with every layer on
the GPU. That is the headline, and it is new: the deepest 256K this project had
previously measured was `IQ2_XXS` at **43+22, 2.23 tok/s** and `AD-IQ1_M` at
**46+19, 2.29 tok/s** — a third of the model on the CPU and unusable in a loop.

Neither ladder stopped early. **513 MiB is the edge of the ladder, not the edge
of the model.** For `bonsai-1bit` that is moot — 262,144 is Qwen3.8's own
maximum. For Ornith-9B it is an open question the ladder did not ask.

---

## 2. The number the script reports is not the number to use

`ctx_ceiling.py` decides a rung on one condition: *is the split still N+0?* It
does not apply the project's own **512 MiB reserve**, adopted in report 04 §5
after `--fit-target 256` produced intermittent driver eviction at 345 MiB.

So it reports rungs the project does not accept.

| arm | script reports | **deepest rung still ≥512 MiB free** | |
|---|---|---|---|
| `bonsai-1bit` | 256K · 513 MiB | **256K** | clears by 1 MiB |
| `ornith9b` | 256K · 444 MiB | **224K** · 633 MiB | one rung down |
| `v3-iq1s` | 192K · 226 MiB | **160K** · 820 MiB | one rung down |
| `v3-iq1m` | 160K · 365 MiB | **128K** · 943 MiB | one rung down |
| `iq2xxs` | none | **none** | spills at 128K |

**Every arm loses a rung to the reserve except the one that had none to lose.**
`v3-iq1m` at 160K sits at 365 MiB — twenty megabytes above the headroom where
this project measured real eviction.

This is a defect in the instrument, not a reading error, and it is the same
class of defect the panel warned about (report 14 §3): a residency number that
is true while the claim built on it is false. **Fix `ctx_ceiling.py` to report
both columns** rather than requiring every reader to re-derive the second.

---

## 3. What the control says about the current recommendation

`iq2xxs` — pre-V3 `UD-IQ2_XXS`, what `production-iq2xxs.ps1` serves and the
project's standing recommendation — **spills at 128K**: 58+8, 428 MiB free.
That matches report 12 §3's 58+7 at 589 MiB to within boot variance.

So the arm this project recommends **cannot do 128K resident at all**, and the
arms that can are the ones whose coding quality is either rejected (both V3
arms, report 12 §7) or entirely unmeasured (`bonsai-1bit`, `ornith9b`).

That is the real state of the ">128K" goal: **the residency half is solved and
the quality half is untouched.**

---

## 4. What this run does not establish

- **No throughput at any of these depths.** The ladder reads the layer split
  and nothing else. `bonsai-1bit` at 256K is 65+0; what it decodes there is
  unmeasured. Residency predicts the cliff, it does not predict a rate.
- **No quality anywhere on the ladder.** `bonsai-1bit` has **no corpus result
  at all** — its +80.12 % is a speed figure. This project has already seen the
  fastest artifact it ever measured (V3 `IQ1_S`, 50.8 tok/s) turn out to emit
  no usable answer in twelve of twelve attempts. A 256K residency number from
  an arm that has never passed a coding task is a capacity claim, not a
  capability one.
- **One boot per rung.** Free VRAM at boot has moved 9,326–10,732 MiB across
  this project's 25 measured boots, and `--fit` follows it. A rung that clears
  by 1 MiB — which is exactly `bonsai-1bit` at 256K — is inside that variance.
  **Treat 256K on `bonsai-1bit` as unconfirmed until it is booted again.**
- **`q4_0` KV only.** No other cache type was walked.

---

## 5. What to run next, in order

1. **`answer_screen.py` on `bonsai-1bit`** — four minutes, and it decides
   whether anything else on this page matters. It is the only arm that reaches
   256K and the only one with zero quality evidence.
2. **Re-boot `bonsai-1bit` at 256K** — twice more, to see whether 513 MiB was
   the arm or the boot.
3. **Decode rate at the reserve-safe rung** for whichever arms survive (1).
4. **Deep retrieval quality**, still the top item in report 06 §0 and still
   measured on Q4 alone.
