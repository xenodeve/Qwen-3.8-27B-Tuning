# Session Record — 2026-08-20 into 2026-08-21

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

> **What this is.** One session, roughly nine hours, in which 21 optimization
> levers were measured, four published claims were retracted, and the largest
> throughput result of the project was found — in the layer that had been
> assumed settled.
>
> Reports 15, 16, 19 and 20 hold the structured findings. **This file holds the
> arc**: what was tried in what order, what broke, and what each break taught
> about the instrument. Written because the reasoning behind a number is what
> gets lost first, and re-deriving it is what a fresh agent spends its time on.

---

## 0. The headline, in one table

`v3-iq2xxs`, KV `q4_0`, all at `65+0` — nothing displaced, no VRAM added.

| context | config | tok/s | note |
|---:|---|---:|---|
| 16,384 | none | 41.81 | control |
| 16,384 | `--spec-type ngram-map-k` | **93.75** | **+135.89 %**, four rounds, +124…+146 |
| 131,072 | none | 26.50 | control |
| 131,072 | `--spec-type ngram-mod` tuned | **81.46** | **+213.08 %**, +207…+219 |

**The same model is nearly twice as fast at 128K with n-gram as it is at 16K
without it.** That was not thought possible when the session started.

---

## 1. What was retracted, and why each retraction happened

Four claims published earlier in the session did not survive. Each failed for a
different reason, and the reasons are more useful than the claims.

### 1.1 "At 128K throughput is a plateau; no lever can raise it"

**Report 19, written 22:00.** Ten boots across three artifacts all returned
24.98–28.67 tok/s at `65+0`, and the reasoning was: the cache is 2,304 MiB for
every artifact, attention over it dominates, therefore the number is fixed by
the window.

**Refuted at 01:24** by `ngram-mod` returning **81.46 tok/s** at the same depth,
same artifact, same cache size.

**Where the reasoning went wrong:** the plateau is real *across artifacts* —
changing the model does not help, because the cache is identical. The step from
"changing the model does not help" to "nothing helps" was never justified, and
**all ten boots had speculation off.** n-gram does not shrink the cache. It
reduces how many times the cache is read, by drafting several tokens per verify
pass.

### 1.2 "`ngram-map-k` is the winner at +94.69 %"

**Measured 20:08, re-run 23:08:** `+69.73 %` with one round at `+3.68 %`. And
`ngram-cache` reversed from **+80.79 % to −30.56 %** — both sweeps marked
`RESOLVED`, because the sign was consistent *within* each sweep.

**Cause:** the timed generations ran at `temperature 0.7`
(`depth_sweep.py:189, 210`). Every round wrote different text; n-gram replays
sequences already in the context, so its hit rate followed the text.

The `greedy_hash` values that matched across all five arms came from a
**separate** request at `temperature 0.0, seed 42` (line 219). Treating the two
as one thing was the mistake — the hashes prove the decoders are lossless, not
that the timing was stable.

**This was a hole in the resolution criterion, not in one lever.** The 13.6 %
floor was measured from boot-to-boot VRAM drift; it cannot see variance whose
source is the content. Any two-round result on a content-dependent lever carried
the same risk.

**Fix:** `depth_sweep.FIXED_TEXT` / `kv_sweep --fixed-text` pins temperature 0
and a fixed seed for the timed generations, and `fixed_text` is recorded on every
row so a number cannot be read without knowing which mode produced it. Default
unchanged, so no existing result was invalidated.

Under `--fixed-text`, four rounds put every n-gram arm inside a **9–22 point
band**, and `ngram-map-k` at its **default** lookup lengths is the fastest.

### 1.3 "The n-gram defaults are too long for this workload"

Written at 23:30 to explain why the tuned arms looked stable and the default ones
did not. **Also wrong** — shortening the lookup did not stabilise anything; it
landed in a quiet part of the noise. Under `--fixed-text` the defaults are stable
too, and `ngram-map-k` at defaults wins at 16K.

*A caveat on the caveat:* at 131,072 the tuned `ngram-mod` was measured and
`ngram-map-k` was **not**. Which wins at depth is still unmeasured.

### 1.4 "`output_contract_pct` is the violation rate"

```python
output_contract_pct = 100.0 * (attempts_seen - contract_violations) / attempts_seen
```

It is the **pass** rate. Every statement of the form "58.3 % of attempts violate
the contract" said the opposite of the truth; the correct reading is "58.3 % of
attempts pass".

The gate in `afk-q38-quality.sh` was written `pct <= 10.0` and is now
`pct >= 90.0`. **The error was safe in exactly one direction** — an inverted gate
passes nothing, so no arm was ever declared good that was not — but every report
quoting it was misleading. Corrected in reports 15, 16, 19, 20 and `START-HERE`.

---

## 2. Layer 9 — the decoders, in full

Eleven exist. Six had never been run at the start of the session; ten have now
been run, and the last one cannot be.

| decoder | 16K | 128K | VRAM cost |
|---|---:|---:|---:|
| `ngram-map-k` (defaults) | **+135.89 %** | not measured | **0** |
| `ngram-mod` (tuned) | +112.55 % | **+213.08 %** | **0** |
| `ngram-map-k4v` (tuned) | +114.64 % | +108.15 % | **0** |
| `ngram-cache` | +108.49 % | not measured | **0** |
| `ngram-simple` | erratic at defaults | not measured | **0** |
| `draft-mtp` on GPU | **+81.48 %** | **−71.36 %** | 1,195 MiB |
| `draft-mtp` via `-otd .*=CPU` | +18.41 % | −59.27 % | 467 MiB |
| `draft-mtp` via `--spec-draft-device none` | −5.08 % | −58.03 % | moves the *target* to CPU |
| `draft-simple` (Qwen3.8-2B distill) | +37.91 % | cannot fit | 1,452 MiB |
| `draft-simple` on CPU | −61.83 % | — | — |
| `draft-dflash` | **will not load** | — | — |
| `draft-eagle3`, `draft-dspark` | no checkpoint for Qwen3.8 | — | — |

### 2.1 MTP inverts with depth, and the old explanation was wrong

Report 15 §2.1 said MTP fails on a resident target because the head's VRAM
displaces layers. That fits the original `Q2_K_XL` measurement — it sat at
`61+4` and the drafter pushed it to `55+10`.

It does **not** fit `v3-iq2xxs`, where `mtp-gpu` reaches **`66+0` at both
depths** — nothing displaced, and at 128K the cache even *shrinks* to 1,872 MiB.
It still loses 71 %.

**The cost is prefill.** At 16K, prefill 10.4 s → 10.9 s, against decode 64 %
faster: it wins. At 131,072, prefill 120 s → **206 s**: 86 seconds lost before
the first character, which decode cannot earn back.

**MTP pays when both hold:** prefill is short enough that the added cost is
smaller than the decode gain, **and** there is VRAM slack the drafter can take
without evicting layers. At 16K both hold; at 128K neither does.

Even where it wins it is beaten — n-gram returns more, for zero VRAM.

### 2.2 `--spec-draft-device none` does not mean "drafter on CPU"

It produced `0+66` — the **entire target** moved to the CPU, prefill 541–595 s.
The flag disables offload globally. **`-otd ".*=CPU"` is the flag that does what
was intended**: target stays `66+0`, only the drafter's tensors move, 467 MiB
instead of 1,195.

### 2.3 DFlash 2 — the loader says exactly why

```text
print_info: general.name = Qwen3.8-27B-DFlash2
llama_model_load: error loading model: done_getting_tensors:
    wrong number of tensors; expected 81, got 58
```

The file reads and its metadata parses; the tensor layout does not match what
`draft-dflash` expects. **DFlash 2 is a different architecture**, so
[PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) must change the
loader — adding a name to a list would not be enough.

**Cost of that answer: one boot, four seconds.** The alternative was a source
build of an unmerged PR on an assumption.

---

## 3. The format problem, and what did and did not fix it

The blocking failure is that a large share of corpus attempts emit **no fenced
code block at all**, having looped inside the reasoning block until the token cap.

### 3.1 The sampling screen — 14 configs, `v3-iq2xxs`, n=3 each

| config | max reasoning chars | median | fenced 3/3? | wall |
|---|---:|---:|:--:|---:|
| `-rea off` | **0** | 0 | ✓ | **2.9 s** |
| `--prefill-assistant` | 5,005 | 1,486 | ✓ | 20.5 s |
| `--reasoning-budget 2048` | 6,682 | 1,586 | ✓ | 23.2 s |
| DRY + `--reasoning-budget 2048` | 7,235 | 2,917 | ✓ | 28.1 s |
| `--mirostat 2` | 8,477 | 8,362 | ✓ | 81.1 s |
| `--repeat-penalty 1.05 --repeat-last-n 4096` | 4,771 | 1,750 | 2/3 | 20.3 s |
| `--repeat-penalty 1.05` (window 64) | 4,907 | 1,689 | 2/3 | 18.8 s |
| control | 16,277 | 1,777 | 5/6 | 29.3 s |
| `--top-n-sigma 1.0` | 26,076 | 1,281 | 2/3 | 71.5 s |
| **`--reasoning-budget 0`** | **24,709** | 1,365 | 1/3 | 69.1 s |
| DRY alone | 26,451 | 14,063 | 1/3 | 93.6 s |
| **`--grammar-file`** | 695 | **content 0** | **0/3** | 3.3 s |

Two flag findings worth keeping:

- **`--reasoning-budget 0` does not end the block**, despite being documented as
  an immediate stop. Screened alone it ran to 24,709 characters. `-rea off` is
  the flag that actually disables reasoning.
- **`--repeat-last-n 64` performs the same as 4096.** The arithmetic argument
  ("a 64-token window cannot see a 4,000-token loop") predicted otherwise. The
  loops are apparently short repeats repeated often, not one long block.

### 3.2 The grammar failed in a way that refutes its own premise

`--grammar-file` with `--reasoning-budget 0` returned **`content_chars = 0` on
all three trials** — 350–700 characters of reasoning, then `finish_reason: stop`
with nothing emitted.

The premise had been: *"the sampler cannot emit a token the grammar forbids, so
'no fenced block' stops being a possible outcome."* **The model found a third
option — emitting nothing.** The grammar does not cover the reasoning block, so
the model reasons freely, and at the point the grammar starts to bind it emits
end-of-turn instead of the fence.

**Fixed by pairing it with `-rea off` instead**: 6/6 trials produced a fenced
block, content 314–539 characters, 2.4–3.7 s. With no reasoning block there is
nowhere to escape to, and the grammar binds from token one.

The three `serve-v3-*-fmt.ps1` profiles were corrected from
`--reasoning-budget 0` to `-rea off` before the corpus ran. **The 4-minute screen
caught this** — without it, three 90-minute corpus runs would have measured a
configuration that produces no answers.

### 3.3 The corpus verdicts

30 tasks, `max_tokens 8192`, `v3-iq2xxs`:

| arm | accepted | contract **pass** | wall | note |
|---|---|---:|---:|---|
| control (unconstrained) | 19/27 | 58.3 % | ~2,900 s | |
| `-rea off` | **15/30** | 58.0 % | **444 s** | worse, and the contract did not move |
| `--grammar-file` + `-rea off` | 16/27 | **84.3 %** | — | contract **+26 points** |

**`-rea off` alone does not fix the format problem.** Disabling reasoning does
not stop the model reasoning — it moves where it reasons. The violations show it
plainly:

```text
no fenced python block                                       17
prose outside the fence: 'Wait, the code above is wrong...'   1
prose outside the fence: 'The previous code had a logical...' 1
prose outside the fence: 'I need to understand why the...'    1
2 fenced blocks, expected 1                                   1
3 fenced blocks, expected 1                                   1
```

**The grammar does fix it** — contract pass from 58.3 % to 84.3 %, which is what
it was designed to do and the only thing that has moved that number.

**But accepted did not follow**, 16/27 against the control's 19/27. That run
changed **two things at once** — grammar and `-rea off` — so it cannot say which
caused what. The direction suggests the grammar helps format and `-rea off`
hurts correctness.

**Next experiment:** grammar with reasoning *enabled* and a non-zero
`--reasoning-budget`, so the model may think but must emit a fenced block.

---

## 4. Layer 4 — `-ot` reaches the mechanism `--fit` cannot

`v3-iq2xxs` at 163,840 sits at `62+3` under `--fit`.

| arm | split | tok/s | free MiB | Δ |
|---|:--:|---:|---:|---:|
| control | **62+3** | 14.93 / 16.70 | 382 / 473 | — |
| `-ot …ffn_.*=CPU` | **65+0** | 14.01 / 13.95 | **1,310 / 1,407** | −11.31 % |
| `-ot …ssm_.*=CPU` | **65+0** | 16.83 / 15.87 | 381 / 353 | +3.88 % |

**Both restore full residency**, which `--fit` cannot: it treats a layer as
indivisible and evicts the whole thing. `ssm_*` reaches `65+0` at **no
measurable cost**.

**Caveat:** both `-ot` arms produced a different `greedy_hash` from the control.
Moving tensors to the CPU **changes the output** — CPU and GPU floating point do
not agree bit-for-bit. Unlike n-gram, this is not a free speedup; it is a
different computation.

This is the live route to `AD-IQ1_M` — the artifact with the best corpus of any
1-bit-named file (27/30) — reaching 128K, where it currently misses `65+0` by
**one layer** (`65+1`, 338 MiB free, needs ~125).

---

## 5. Instrument faults found, and what each cost

Five in one session. Every one produced a plausible number or silently discarded
work.

| fault | symptom | cost | fix |
|---|---|---|---|
| timed generations at `temperature 0.7` | a lever's result followed the text it wrote; one arm reversed sign between sweeps | two retracted claims | `--fixed-text` |
| `output_contract_pct` read as a violation rate | every report stated the inverse | all narrative, no decision (the inverted gate passed nothing) | gate now `>= 90` |
| `swap-model.sh` lock keyed to `$PPID` | a sweep that swaps 14 times refused itself after the first | 13 of 14 configs died in 2 s | allow re-swap when the lock owner **is** the caller |
| `answer_screen --trials N` clamped to `len(PROBES)` = 3 | `--trials 10` silently gave 3 | the entire premise of a second pass | acknowledged; the corpus is the sample size |
| `rc=$?` after `$(date …)` | `FAIL … (rc=0)` — the real exit code lost | one undiagnosable failure | capture `rc` before anything else runs |

**A sixth, in the queue design:** editing a running bash script corrupts its
parse, because bash reads incrementally by byte offset. Every armed script that
needed a change was killed, edited, relaunched, and verified to have exactly one
instance — twice a duplicate slipped through and was caught by that check.

---

## 6. Where things stand

### The config to run today

`scripts/production-iq2xxs-ngram.ps1` — pre-V3 `UD-IQ2_XXS`, 16K, `q4_0` KV,
plus `--spec-type ngram-map-k`. The base artifact is the standing default at
**27/30 accepted, 60.8 tasks/hour**; the flag is one line.

**Two caveats recorded in the file itself:** the +135.89 % was measured on
**V3** `IQ2_XXS`, not the pre-V3 file this profile serves (the mechanism is
token-level and should carry, but that is reasoning); and it is **not verified at
depth or for quality** in this combination.

### The 128K picture

| artifact | resident to | tok/s @128K | corpus | blocked by |
|---|---:|---:|---|---|
| V3 `UD-IQ1_S` | 196,608 | 27.3–28.7 | 0 accepted | quality |
| V3 `UD-IQ1_M` | 163,840 | 26.4–27.5 | 10/21 | quality |
| V3 `UD-IQ2_XXS` | 131,072 | **26.5 → 81.5 with n-gram** | 19/27 | quality |
| `AD-IQ1_M` | `65+1` — one layer short | 18.75 | **27/30** | 125 MiB |
| pre-V3 `UD-IQ2_XXS` | `58+7` | — | **27/30 · 60.8/hr** | depth |

**Speed at 128K is no longer the blocker.** Quality is.

### The strategy, as agreed

Tune to a stable configuration on the **Q2** artifact first, then carry the same
configuration to Q1 without discarding it. This works because **most of what was
found is artifact-independent** — n-gram operates on tokens, KV type is
universal, and the confirmed-inert flags are inert everywhere. Only `-ot` needs
re-measuring per artifact, because tensor names differ and `AD-IQ1_M` carries an
extra `blk.64`.

**One trap to avoid:** choose the artifact by *quality*, then make it fast — not
the reverse. V3 `IQ1_S` is the fastest artifact ever measured here (50.55 tok/s
at 16K, resident to 196,608) and produced **zero** accepted tasks.

**A naming note that matters here:** `AD-IQ1_M` is **2.49 bits per weight** —
*heavier* than V3 `UD-IQ2_XXS` at 2.16. Only 80 of its tensors are `iq1_m`; 128
are `q8_0`. Carrying a config from "Q2" to that "Q1" is not a step down in bits.

---

## 7. Open, in priority order

1. **Grammar with reasoning enabled** and a non-zero budget — the run that
   separates "grammar helps format" from "`-rea off` hurts correctness".
2. **`-ot` on `AD-IQ1_M` at 131,072** — running at the time of writing. If a
   small slice buys the missing layer, the best-quality artifact reaches the
   target depth without touching the desktop's 2,202 MiB.
3. **`ngram-map-k` at 131,072** — it wins at 16K and was not in the depth sweep.
4. **A corpus at 128K with n-gram** — the config that would actually ship has
   never had its quality measured.
5. **`greedy_hash` under n-gram at 128K** — verified identical at 16K, unchecked
   at depth.
6. **Deep retrieval quality on anything but Q4** — nine artifacts have depth
   *throughput* numbers and not one has a depth *quality* number. Top item in
   report 06 §0 for four days.
7. **The desktop's 2,202 MiB.** 33 processes hold it; freeing it is the single
   largest untested lever for any arm within a gigabyte of its ceiling.
