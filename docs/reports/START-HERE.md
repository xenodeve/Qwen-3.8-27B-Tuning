# Start Here — What This Project Did, and What It Found

> **For an agent joining with no context.** Read this one document and you will
> know what was tried, what it cost, what was learned, and what is still open.
> Every number below was measured on this machine; the report it came from is
> named so you can check it.
>
> **Last updated:** 2026-08-21 03:10 UTC+7 -- see the two corrections below.

---

> **Correction, 2026-08-21.** `output_contract_pct` is the **pass** rate —
> `100 * (attempts_seen - contract_violations) / attempts_seen` — not the violation
> rate. Text written on 2026-08-20 read it backwards. The figures are unchanged;
> their direction is. Higher is better.

> **Correction, 2026-08-21 (report 23 section 2).** `AD-IQ1_M` does **not**
> reach 131,072, and the "one layer short" framing below was too kind. At `65+1`
> it decodes at **6.08 tok/s** with a 386.9 s prefill -- a collapse, not a near
> miss. The `-ot` route named below was tried and is dead: moving 644 MiB of FFN
> to CPU freed the layer and dropped prefill to **8.56 tok/s**, twenty-eight
> times slower. Treat `AD-IQ1_M` as a 16K artifact.

> **Correction, 2026-08-21 05:50 (report 24).** Two figures below are now wrong
> in the same direction — too pessimistic. **`v3-iq2xxs` holds `65+0` at
> 147,456**, not 131,072: report 21 measured 131,072 and 163,840 and never
> looked between them. And **`60.8 tasks/hour` is `verified_tasks_per_hour` at
> `max_tokens 3072`**; the same artifact at the standard 8,192 budget gives
> **48.5 verified / 26.5 merged**, at the same 90 % accept. Quote 48.5 when
> comparing against anything measured at 8,192.

> **Caveat, 2026-08-21 05:50 (report 24, instrument fault 8).** Every n-gram
> percentage in this document was measured on a prompt that is **84.5 %
> duplicate lines** — one class repeated with a changing index. An n-gram
> decoder drafts from context, so those are upper bounds on a synthetic best
> case. The mechanism holds (free, no VRAM, byte-identical output); the
> magnitudes are pending re-measurement at 73 % repetition.

> **Correction, 2026-08-21 (report 23 section 1).** The n-gram figures below were
> measured before `--fixed-text` existed, when the timed generations still ran at
> `temperature 0.7`. On the corrected instrument, four rounds at 16K:
> `ngram-map-k` **+135.89 %** (not +94.69 %), `ngram-mapk4v` +114.64 %,
> `ngram-mod-short` +112.55 %. **At 131,072 the order reverses**: `ngram-mod`
> **+200.22 %** against `ngram-map-k`'s +120.54 %, both hash-identical.
> **`ngram-cache` is disqualified** -- its greedy hash differs from the baseline,
> so it changes the answer.

> **The full list of superseded claims is [`CORRECTIONS.md`](CORRECTIONS.md).**
> The banners above are the two that change what you would do today; there are
> ten in total.

## 1. What this is

A local coding-agent worker. One consumer GPU, one model, serving an autonomous
agent that reads files, edits them, runs tests, repairs, and returns evidence.

**The metric is the only thing that decides anything:**

> **verified accepted coding tasks per hour** — a task counts only if the
> generated code runs and passes its tests.

Not tok/s. Not benchmark scores. Throughput and capability are tracked as two
separate numbers, because four different artifacts tie at 27/31 accepted and
differ **2.9×** in wall clock.

**The current goal:** a usable context of **128K or more**, fully GPU-resident,
at the highest tok/s achievable.

---

## 2. The machine, and the one fact that explains most results

| | |
|---|---|
| GPU | RTX 4070 SUPER, **12,282 MiB**, compute 8.9 (Ada), driver 610.88, PCIe 4×16 |
| CPU | i5-13500 — 6 P-cores + 8 E-cores, 20 threads. **Logical processors 0–11 are the P-cores** (verified via `PercentProcessorPerformance`: 144–179 vs 99–139) |
| RAM | 48 GB DDR5-7000 dual channel |
| Disk | WD_BLACK SN850X 1 TB NVMe. **~29 GB free — a real constraint** |
| OS | Windows 11 Pro build 26200, WDDM (TCC unavailable) |
| Runtime | llama.cpp `llama-server` build **10472**, commit **`60eeeb608`**, CUDA 12 |
| Serving | one process, `127.0.0.1:8080`, one slot, single stream |

**The fact:** a 27B model at 2 bits is ~7 GiB and the KV cache at 128K is
~2.3 GiB, out of **one 12 GB pool**. Weights and cache compete directly, and the
loader resolves that competition by moving whole layers to the CPU.

**A layer on the GPU is worth roughly twice a layer on the CPU:**

```text
UD-Q4_K_XL   16.69 GiB    33+32    13.1 tok/s
UD-Q2_K_XL    9.94 GiB    61+4     21.8
UD-IQ2_XXS    8.39 GiB    65+0     41.3
Bonsai-Q1_0   3.54 GiB    65+0     69.3
```

`llama-server` prints the split as `<gpu>+<cpu>`; 65 layers total. **The last
four CPU layers cost about half the throughput.** This single mechanism explains
more of this project's results than every flag combined.

**The measurement floor is 13.6 %.** Free VRAM at boot varies 9,326–10,732 MiB
and `--fit` follows it; the same control config spans 32.4–42.5 tok/s across 25
boots. Anything smaller than 13.6 % cannot be distinguished from restarting.

---

## 3. What was done, in order

### Phase 1 — flags on Q4 *(reports 00, 01, 02, 03, 09)*

Swept `-t`, `-tb`, `-b`, `-ub`, `--fit-target`, speculation depth and its
sub-knobs, KV type, and depth to 256K. Built the measurement discipline in
report 04.

**Found:** almost nothing. The whole stacked tuned config was +6.6 % paired.
`-t`, `-tb`, `-b`, `-ub`, `--fit-target` are **settled — do not re-sweep them.**

The one large result was speculation on Q4: `draft-mtp` at +46.8 %. That turned
out to be a special case (see phase 2).

### Phase 2 — crossing the residency threshold *(report 10)*

Dropped from Q4 to Q2 to IQ2 until the model fit entirely on the GPU.

**Found: +220 % decode at the same 27/30 accepted.** This is the project's
largest result, and it is what made everything after it about VRAM.

It also inverted the speculation verdict: MTP is **+46.8 % on CPU-offloaded Q4
and −8.8 % on resident Q2**, because the draft head's VRAM pushed six layers back
to the CPU.

### Phase 3 — every other model *(reports 08, 13)*

Twenty artifacts across eight families: Qwen3.8-27B, Qwen3.6-35B-A3B MoE,
Ornith 9B and 35B-A3B, Ternary Bonsai 27B, gpt-oss-20b, AtomicChat requants.

**Found:** the MoE arms are fastest raw (+78–80 %) but were measured at 227–363
MiB free, below the 512 MiB stability reserve, so those numbers are directional
only. `Bonsai-27B-Q1_0` (3.54 GiB) and `Ornith-9B` both hold **262,144** context
fully resident — but Bonsai fails the answer screen and has no corpus result.

Also found the research this phase came from was wrong in four specific ways —
recorded in the research brief §4.3 so it is not repeated.

### Phase 4 — Unsloth republished the repo mid-session *(report 12)*

On 2026-08-19T16:39:23Z, `unsloth/Qwen3.8-27B-GGUF` was replaced in place — same
filenames, different contents, different byte counts.

**Consequence: reports 00–11 and 13 are the PRE-V3 generation and are not
comparable to the current repo.** Every artifact is now pinned by exact byte
count in `bench/depth_sweep.py`, and `cached()` raises rather than choosing when
a filename is ambiguous across snapshots.

### Phase 5 — the 128K programme *(reports 15, 16, 19, 20 — this is today)*

Catalogued the entire tunable surface (248 runtime options, 16 layers), then
measured 21 levers in one session.

**Found three things that changed the strategy:**

1. **At 128K, throughput is a plateau.** Ten boots, three artifacts: 24.98–28.67
   tok/s, all `65+0`, all with a 2,304 MiB cache. The spread is inside the noise
   floor. Weight size decides *whether* you are resident, **not how fast you are
   once you are** (report 19). So at depth, prefer the **largest** artifact that
   fits — not the smallest.
2. **n-gram speculative decoding doubles decode, for free** (report 20 §1).
3. **The blocking failure is output format, not reasoning.** Only 41.5 % to 58.3 % of
   corpus attempts emit **no fenced code block at all**, having looped inside the
   reasoning block until the token cap.

---

## 4. Where things stand

**No artifact passes every requirement.** They split cleanly into two groups that
fail on opposite sides.

| artifact | 128K resident | tok/s @128K | corpus 30 tasks | fails on |
|---|:--:|---:|---|---|
| V3 `UD-IQ1_S` | ✓ to **196,608** | 27.3–28.7 | **0** — no fenced block 12/12 | quality |
| V3 `UD-IQ1_M` | ✓ to 163,840 | 26.4–27.5 | 10/21 · **41.5 % contract pass** | quality |
| V3 `UD-IQ2_XXS` | ✓ to **147,456** (report 24; 131,072 was never the ceiling, only the deepest depth tried) | 24.9–26.7 | 19/30 · **58.3 % contract pass** | quality |
| `AD-IQ1_M` (AtomicChat) | X `65+1` -- **ruled out at 128K**, report 23 s2 | 18.75 @16K / **6.08 @128K** | **27/30** | depth |
| pre-V3 `UD-IQ2_XXS` | ✗ `58+7` | — | **27/30 · 48.5 verified/hr** (8,192) | depth |
| V3 `UD-Q2_K_XL` | ✗ `54+12` | — | not measured | depth |
| pre-V3 `UD-Q2_K_XL` | ✗ `50+16` | — | 26/30 | depth |

**Two candidates are one fix away:**

- **V3 `UD-IQ2_XXS`** — passes depth and speed, fails only on format. If a GBNF
  grammar moves it from 19/30 toward 27, it qualifies outright.
- ~~**`AD-IQ1_M`** -- passes quality, needs about **125 MiB**, and `-ot` is
  the live route.~~ **Dead, 2026-08-21.** `-ot` frees the layer and destroys
  prefill (240.6 to 8.56 tok/s); the `65+1` baseline is 6.08 tok/s either way.
  Any future attempt must free ~125 MiB **without putting weights on the
  CPU**. Report 23 section 2.

**If the requirement relaxed to 64K, `pre-V3 UD-IQ2_XXS` qualifies today** at
27/30 and **48.5 verified tasks per hour at the 8,192 budget** — the best
comparable number this project has measured,
three times the best 128K arm.

---

## 5. Two things about the artifacts that are not obvious

### 5.1 The quantization names do not describe the files

From the loader's own tensor-type histogram:

| file | GiB | params | **real bits/weight** | what is actually inside |
|---|---:|---:|---:|---|
| V3 `UD-IQ1_S` | 5.77 | 26.90 B | **1.84** | iq1_s ×264, q8_0 ×96 |
| V3 `UD-IQ2_XXS` | 6.77 | 26.90 B | 2.16 | iq2_xxs ×143, **q8_0 ×0** |
| `AD-IQ1_M` | 7.91 | 27.32 B | **2.49** | iq1_m ×80, **q8_0 ×128** |
| pre-V3 `UD-IQ2_XXS` | 8.39 | 27.32 B | 2.64 | iq2_s ×208, iq1_m ×96 |

**The file named `IQ1_M` is heavier than the file named `IQ2_XXS`.** Only 80 of
`AD-IQ1_M`'s tensors are actually 1-bit; 128 are full 8-bit.

Within one publisher the ordering is correct. Across publishers it inverts. **The
old rule "pick the smallest artifact that fits" assumed the name tracked the
size, and it does not.** It also explains why `AD-IQ1_M` has the best corpus of
any 1-bit-named artifact — it is not a 1-bit model.

### 5.2 V3 removed the MTP head, but not where the documentation says

```text
v3-q2kxl        blk.64.attn_q  blk.64.attn_k  …     HEAD PRESENT   27.32 B params
v3-iq2xxs       (no blk.64)                          REMOVED        26.90 B
v3-iq1s         (no blk.64)                          REMOVED        26.90 B
pre-V3 iq2xxs   blk.64.attn_q  …                     PRESENT        27.32 B
```

Unsloth's docs say the head was removed from "Q2_K_XL and smaller". **`Q2_K_XL`
kept it; removal starts at `IQ2_XXS`.** Any MTP experiment on `IQ2_XXS` or
smaller therefore needs the standalone 1.28 GiB drafter.

---

## 6. Every lever measured, and its verdict

Full detail and per-round numbers in **report 20**.

### Works

| lever | effect | notes |
|---|---|---|
| **residency** (`65+0`) | **+220 %** | the largest result in the project |
| **`--spec-type ngram-map-k`** | **+135.89 %** at 16K (4 rounds, fixed text) | byte-identical output, zero VRAM, no drafter file |
| **`--spec-type ngram-mod`** (short window) | **+213.08 %** at 131,072 | 99 % acceptance, identical output. The deep-context arm |
| `ngram-map-k4v` / `ngram-simple` | +108 % to +115 % | same properties, more variance |
| ~~`ngram-cache`~~ | +108 % | **disqualified -- changes the greedy hash.** Fast and wrong |
| `-ctk q4_0 -ctv q4_0` | buys residency | the settled KV choice |
| **`-ot …ssm_.*=CPU`** | `62+3` → **`65+0`** at 163,840 | no measurable throughput cost. **Changes the greedy hash** — CPU and GPU floats differ |
| `-ot …ffn_.*=CPU` | frees 1,310–1,407 MiB | costs 11 % at depth, 61 % at 16K |
| `max_tokens` 3072 → 8192 | 15/31 → **27/31** | a treatment, not a detail |

### Does not work

| lever | measured | note |
|---|---|---|
| `draft-mtp`, all three placements | **−58 % to −71 %** at 128K | fails on **prefill** (120 s → 206–595 s), not on VRAM |
| `-ctk q8_0 -ctv q4_0` (mixed KV) | **−76.7 %**, prefill 29× slower | no kernel for mismatched K/V. Cache is also 44 % *larger* |
| `--ctx-checkpoints 8` | frees **10–16 MiB** | the research claimed ~900 MiB. Off by ~50× |
| `-np 2` | **harmful** | divides the context between slots — 16K becomes 8K per slot |
| `-fa off` | **will not load** | flash attention is a precondition of a quantized KV cache |
| `-sm tensor`, `--no-repack`, `--no-op-offload`, `--load-mode none`, `--no-host`, `--swa-full`, `--no-kv-unified` | all under the floor | `--swa-full` reports an identical cache size — the architecture does not use SWA |

### Measured with the wrong instrument — do not read these as settled

`--cache-reuse` (+9.62 %) and `--context-shift` (−1.29 %) both look inert, but
the probe never breaks a prefix and never fills the window, which is the only
situation either flag acts in. They need `stability_gate.py`, which forces a
prefix invalidation every tenth turn.

### Never run

`draft-simple` and `draft-dflash` — both drafters are now on disk (2B distill
1.22 GiB; DFlash 2 1.06 GiB) and queued. `draft-eagle3` and `draft-dspark` have
no checkpoint for Qwen3.8.

---

## 7. How to measure here without fooling yourself

Report 04 §7 lists thirteen instrument failures, each of which produced a
believable wrong number. These four catch a fresh agent within the first hour.

1. **An undersized `max_tokens` is indistinguishable from a stupid model.** Four
   verdicts were withdrawn in one day over this. Median reasoning spans 59 to
   2,811 characters *across quantizations of the same model*; one V3 artifact
   reached 37,000. Budget for the most verbose arm, record `finish_reason`, and
   treat a truncated attempt as **censored, not failed**.
2. **Never compare raw decode across boots.** Pair within a round. The harness
   refuses to call an effect real below 13.6 % or with an inconsistent sign.
3. **Address artifacts by exact path plus byte count.** `-hf repo:Q2_0` fetched
   `PQ2_0.gguf` — a different file of *identical* byte count. `-hf` also does an
   online etag check on every launch; a busy link once stalled an unattended
   queue for eleven minutes. All scripts use `-m <path>`.
4. **`resolved` is necessary, not sufficient.** Below ~512 MiB free VRAM an arm
   can pass the sign test and still be unstable; the signature is a wide spread
   containing one normal sample, not a lower mean.

Two more learned today:

5. **Desktop VRAM is a live variable.** 33 processes held **2,202 MiB** during
   the 2026-08-20 runs — Edge WebView2 ×4, NVIDIA Overlay, Windows Terminal,
   Snipping Tool. `AD-IQ1_M` *was* resident at 128K in an earlier session with
   10,730 MiB free at boot and is not with ~9,796. Any arm within ~1 GB of its
   ceiling is conditional on what else is open. **Read `free_before`.**
6. **Verify a flag parses before spending a boot on it.** `llama-server <flags>
   -m /nonexistent.gguf --port 18080` errors at argument parsing if the flag is
   wrong, and reaches model loading if it is right. Thirty flag groups were
   checked this way today; one (`--slot-save-path`) failed only because its
   directory did not exist.

### The test gate

```powershell
cd C:\AI\qwen38-tuning\bench ; python -m pytest tests\ -q    # 103 tests
```

If those do not pass, **fix that first.** They are the instrument, and a broken
instrument returns a number instead of a failure.

### Two orchestrators cannot share port 8080

`scripts/swap-model.sh` takes a lock keyed to the calling job's PID. That guard
exists because an armed queue killed a running corpus at 02:00:17 and the summary
still printed a plausible number — 26 of 30 tasks had returned HTTP 503 in 0.0 s.
Note that `bench/ctx_ceiling.py` kills whatever listens on 8080 directly and does
**not** take the lock, so it must never run beside another job.

---

## 8. What is running right now

Seven chained queues under `C:\AI\qwen38-tuning\scripts\`, each waiting on a
literal line in `logs/afk-driver.log`. One process per stage — check with:

```sh
ps -ef | grep "bash scripts/afk-q"
```

```text
afk-qwen38-resident   DONE
afk-q38-ckpt          DONE
afk-q38-layers        DONE
afk-q38-depth-levers  running  -- -ot and P-core mask at 163,840
afk-q38-sampling      armed    -- 14 configs, answer_screen, 4 min each
afk-q38-decoder       armed    -- all 11 decoders incl. draft-simple, DFlash 2
afk-q38-quality       armed    -- grammar corpus: iq2xxs -> iq1m -> iq1s,
                                  STOPS at the first ≥80 % accepted and ≤10 %
                                  contract violations
afk-q38-followup      armed    -- n-gram at 131,072; graded -ot on AD-IQ1_M
```

**The three questions those queues answer**, in order of what they decide:

1. Does the n-gram win survive to 131,072? Prefill there is 110–127 s and
   speculation cannot touch it, so the gain per task will be smaller than the
   tok/s figure suggests.
2. Does a GBNF grammar fix the format failure? `grammars/python-fence.gbnf` plus
   `--reasoning-budget 0`, served by `scripts/serve-v3-*-fmt.ps1`, which are
   byte-identical to their unconstrained twins except for those two flags.
3. Can a graded `-ot` buy `AD-IQ1_M` its one missing layer?

---

## 9. Where to read more

| you want | read |
|---|---|
| every model × quant × probe actually run | report **15** |
| every tunable option, including the ones judged inert | report **16** |
| why 128K speed is flat across artifacts | report **19** |
| all 21 levers with per-round numbers | report **20** |
| how to measure without fooling yourself | report **04** |
| what is unmeasured and what to do next | report **06** §0 |
| the external research replies, verified claim by claim | reports **17**, **18** |
| a self-contained version for someone off this machine | `MASTER-REPORT-2026-08-19.md` (predates V3) |

**Reports 00–11 and 13 are the pre-V3 generation** and are internally consistent
but not comparable to the current repo.

---

## 10. The largest open item

**Deep-context retrieval quality has never been measured on anything but Q4.**

Nine artifacts now have depth *throughput* numbers. Not one has a depth *quality*
number. Every recommendation this project makes about 128K therefore rests on
throughput and residency alone, and the documented failure mode of aggressive
low-bit builds in this family is *selective* — aggregate scores hold while
long-span retrieval collapses.

It has been the top item in report 06 §0 for three days. The corpus and the
harness both exist; it is two runs of work.
