# 23 — Session record, 2026-08-21 (02:30 onward)

> **Some claims below were later contradicted by this project's own
> measurements.** See [`CORRECTIONS.md`](CORRECTIONS.md) before quoting
> any number from this report.

**Read `22-SESSION-RECORD-2026-08-20.md` first.** This continues it. Where the
two disagree, this one is later.

The night's work was small and mostly corrective: two measurements landed that
settle open questions, one artifact was ruled out, two instrument faults were
found and fixed under test, and a published table turned out to carry a
hand-typed hash that contradicted its own data.

**Nothing here changes the shipping configuration** -- `production-iq2xxs-ngram.ps1`
runs `ngram-map-k`, which is confirmed, not corrected.

---

## 1. The n-gram result survives a correct instrument

Report 22 §5 recorded instrument fault 1: the **timed** generations ran at
`temperature 0.7`, so every content-dependent lever followed the text it happened
to write. `ngram-cache` returned **+80.79 %** and **−30.56 %** in two sweeps three
hours apart, *both* marked `RESOLVED`, because the 13.6 % floor was built for
boot-to-boot VRAM drift and cannot see variance that comes from content.

`kv_sweep --fixed-text` pins `temperature 0`, `top_k 1`, `seed 42` on the timed
requests. Four rounds at 16,384 on `v3-iq2xxs`, alternating boots:

| arm | per-round tok/s | vs `q4_0` | greedy hash |
|---|---|---|---|
| `q4_0` (baseline) | 40.60 · 40.11 · 40.18 · 41.81 | — | `04E5CAB1…` |
| **`ngram-map-k`** | 97.28 · 98.86 · 93.72 · 93.75 | **+135.89 %** | `04E5CAB1…` |
| `ngram-mapk4v-wide` | 91.29 · 84.42 · 86.21 · 87.25 | +114.64 % | `04E5CAB1…` |
| `ngram-mod-short` | 88.20 · 85.26 · 85.15 · 87.16 | +112.55 % | `04E5CAB1…` |
| `ngram-cache` | 83.11 · 86.60 · 84.37 · 85.02 | +108.49 % | **`3EFE9395…`** |

**The swing is gone.** `ngram-map-k` holds 93.7–98.9 across four rounds where the
same family previously produced +80 then −30. The instrument was the variance.

Two things follow.

**`ngram-map-k` is the arm at 16K, and the production script is already correct.**
`scripts/production-iq2xxs-ngram.ps1` was reverted to `ngram-map-k` on the
strength of the first `--fixed-text` data; this four-round run confirms it rather
than merely failing to contradict it.

**`ngram-cache` is disqualified, and not for speed.** Its greedy hash is
`3EFE93950A8A980E` against the baseline's `04E5CAB1D14525C0` -- it changes the
answer. Speculative decoding that verifies against the target model cannot do
that; identical output is the whole contract. Its reported acceptance is `0.0`
while it still runs at twice the baseline, which says the same thing from the
other side: whatever it is doing, it is not draft-and-verify. **Do not ship it.**
It is fast and it is wrong, which is the worst combination this project measures.

### The part that is not a new finding

This was **already in the data on 2026-08-20** and was reported backwards.
`results/kv-decoders.jsonl` recorded `3EFE93950A8A980E` for both `ngram-cache`
boots that night, and report 20 section 1.1 printed `04E5CAB1D14525C0` for it
under the heading *"Byte-identical output"*. The hash block in that report was
**typed by hand rather than read from the JSONL**, and the error certified a
decoder as safe that is not.

Tonight's four boots reproduce `3EFE9395...` exactly. Report 20 section 1.1 now
carries the correction and the arm is struck from its table.

**The lesson is not about `ngram-cache`.** Every other number in report 20 was
extracted from the data; this one block was transcribed, and it is the one block
that was wrong. A figure that a reader cannot trace to a row is a figure nobody
checked -- including the person who wrote it.

### At 131,072

Two rounds, `--fixed-text`, same artifact:

| arm | rounds | vs `q4_0` | acceptance | hash |
|---|---|---|---|---|
| `q4_0` (baseline) | 26.50 · 23.03 | — | — | `04E5CAB1…` |
| **`ngram-mod-short`** | 81.46 · 73.41 | **+213.08 %** | 99.0 % | identical |
| `ngram-mapk4v-wide` | 47.70 · 54.42 | +108.15 % | 83.2 % | identical |

`ngram-mod-short` is the deep-context arm: +207 % and +219 % in its two rounds,
99 % of drafted tokens accepted, byte-identical output, zero VRAM, no drafter
file. `mapk4v` still swings 80 %/136 % even with the text pinned — a second
variance source it has and `mod-short` does not; not worth chasing while a better
arm exists.

### The two depths want different arms

`ngram-map-k` won at 16K and had never been run at the target depth. It now has:
two rounds, `--fixed-text`, both candidates against one baseline in a single
alternating sequence.

| arm | rounds | vs `q4_0` | acceptance | hash |
|---|---|---:|---:|---|
| `q4_0` (baseline) | 24.35 / 24.26 | -- | -- | `04E5CAB1...` |
| `ngram-map-k` | 55.17 / 52.04 | +120.54 % | 96.9 % | identical |
| **`ngram-mod-short`** | **73.05 / 72.89** | **+200.22 %** | 99.0 % | identical |

**The 16K winner is not the 128K winner.** `ngram-map-k` leads by 10 points at
16,384 and loses by 80 at 131,072. `ngram-mod-short` is also the steadier of the
two by a wide margin -- +200.00 % and +200.45 % in its two rounds against
+126.57 % and +114.51 % -- which is what you want from the arm that ships.

So the recommendation splits by depth, and this is now measured rather than
assumed:

```text
ctx <= 16K      --spec-type ngram-map-k        ~2.4x decode
ctx >= 128K     --spec-type ngram-mod          ~3.0x decode
                --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16
                --spec-ngram-mod-n-max 32
```

Both are byte-identical to the unaccelerated output, cost no VRAM and need no
drafter file. This is the fourth trap in report 22 restated with a fourth
example: **a verdict at one depth does not transfer to another.**

---

## 2. `AD-IQ1_M` does not reach 128K. The one-layer story was wrong

Report 22 recorded that `AD-IQ1_M` — the artifact with the best corpus of any
1-bit-named file, 27/30 accepted — missed full residency at 131,072 by **one
layer**, 338 MiB free against ~125 needed, and that a small `-ot` slice might buy
it. The follow-up ran, and both halves of that hope are dead.

**The baseline is not a near miss, it is a collapse.** At `65+1`:

```text
AD-IQ1_M @ 131,072, q4_0 KV, 65 GPU + 1 CPU layer
  pp            240.6 tok/s
  cold prefill  386.9 s
  decode          6.08 tok/s   (6.52 · 6.36 · 6.08 · 5.11 · 4.34)
```

**6.08 tok/s.** Against `v3-iq2xxs` fully resident at the same depth — 26.50
baseline, 81.46 with n-gram — a single CPU layer costs more than a factor of
four. This is the residency cliff from report 20 restated at depth, and it is
steeper here than the 33+32 → 61+4 → 65+0 ladder suggested.

*(The handoff written at 02:28 said this prefill was ~150 s. That was wrong; it
is 386.9 s. Corrected in `HANDOFF-qwen38-2026-08-21.md` §2.)*

**`-ot` does not rescue it; it trades one collapse for a worse one.** The
`ot-ffn-1` arm moved 644 MiB of FFN weights to CPU — enough to free the layer,
which was the point — and the server's own load report confirms it:
`offloaded 66/66 layers to GPU` with `CPU_Mapped model buffer size = 644.14 MiB`.
Prefill then ran at **8.56 tok/s** against the baseline's 240.6. Twenty-eight
times slower. The 93,086-token prompt would have taken about three hours, per
round, per arm.

The mechanism is the same one that made mixed KV (`k8v4`) unusable on 08-20:
weights that live on the CPU have to be crossed once per token of prefill, and
prefill is precisely the part of deep-context work that speculation cannot help.
`-ot` buys decode headroom by spending prefill, and at 131,072 prefill is the
larger bill.

**Verdict: `AD-IQ1_M` is a 16K artifact.** Its corpus is the best measured and
that is worth something, but not at this depth. Its quality result stands; its
residency does not. Any future attempt needs a way to free ~125 MiB that does
*not* put weights on the CPU — a smaller KV, a smaller batch, or the desktop's
2,202 MiB (report 22 §7 item 7, still untested).

---

## 3. Two instrument faults, found by the queue failing

Full write-up with the fix and the reasoning: **`04-MEASUREMENT-METHODOLOGY.md`
§8.** In brief, both are flat constants that did not know what they bounded:

- **`post(..., timeout=3600)`** spent one hour to the second — 01:34:36 to
  02:34:36 — waiting on the 8.56 tok/s arm above, then raised.
- **`kill()` ending in `time.sleep(5)`** did not wait for the driver to release
  12 GB, so the *next* step started into a full GPU, passed `/health`, and died
  on its first request.

  **Measured after the fix, on this machine at 03:15:** tearing down an
  11,501 MiB server took **9.87 s** to return 9,924 MiB. The sleep it replaces
  was 5 s. The old code was not marginally early -- it handed the next arm a
  GPU that was still roughly half full.

The teardown had no `try/finally`, which is what chained them: one slow arm took
out a second step that shared nothing with it but a port.

**The second failure is the one to remember.** `ConnectionResetError` was luck.
With a few hundred MiB more free — a different desktop state, any point on the
13.6 % boot drift — the second server would have loaded and written a row of
plausible numbers measured against a GPU it was sharing. That row is
indistinguishable from a clean one in the JSONL.

Fixed in `bench/harness.py` (`completion_timeout_s`, `vram_settled`),
`bench/depth_sweep.py` (depth-sized budgets, `wait_for_vram_release`,
`try/finally`, abandonment recorded as a row) and `bench/kv_equivalence.py`.
Suite is **92 tests**, up from 81; the eleven new ones are named after this
incident.

**One of them names a bug in the fix itself.** The first version of
`vram_settled` asked only whether free VRAM had stopped moving -- and two polls
taken 3 s apart *before the driver starts releasing* agree perfectly, so it
would have declared a still-full GPU ready. That is the five-second sleep with
extra steps. `kill()` now reads free VRAM before stopping the process and
requires the reading to beat it by 1,024 MiB, a tenth of the smallest artifact
here. Caught by writing the test, not by running the code -- the live batch had
already booted ten servers past the broken version without incident, because on
this machine the driver happens to begin releasing inside 3 s.

**Not fixed, deliberately:** `model_arena.py` and `sweep_runtime.py` each define
their own `post` with a flat `timeout=1800`. Same fault class, but both use short
prompts with no deep prefill, so the constant has never been near its limit
there. Left as tracked work rather than widened into tonight's change.

---

## 4. Four placement levers, finally measured: all inert

These four had never produced a single row. The first attempt died because
`-np 2` halves the per-slot context and the 11,663-token probe came back HTTP
400; the second died on the port collision in section 3. Two rounds at 16,384
on `v3-iq2xxs`, `--fixed-text`, order reversed in round 2:

| arm | per-round vs `q4_0` | mean | verdict |
|---|---|---:|---|
| `pcore-mask` | +0.44 %, +0.49 % | +0.46 % | under the floor |
| `prio-high` | +1.84 %, -5.87 % | -2.02 % | under the floor, sign flips |
| `poll-0` | +0.75 %, +0.62 % | +0.69 % | under the floor |
| `backend-samp` | +2.25 %, +2.30 % | +2.27 % | under the floor |

**Nothing here is worth a flag.** Baseline decode was 38.6-38.65 tok/s and every
arm landed between 36.4 and 39.5. Thread affinity, process priority, the polling
strategy and GPU-side sampling all do nothing on this workload. That closes the
placement group of the sixteen-layer surface; report 20 can now say measured
rather than untested for all four.

### A methodological note worth following up, not acting on

Look at how tightly the pairs repeat: `pcore-mask` +0.44/+0.49, `poll-0`
+0.75/+0.62, `backend-samp` +2.25/+2.30. Those are **separate boots**, and they
agree to within 0.05 percentage points.

The 13.6 % floor was measured on unpinned text, where the generation content
itself varied between rounds. With `--fixed-text` the pairs are two orders of
magnitude tighter than the floor, which raises a real question: **is the floor
now too conservative, and is it hiding small true effects?** `backend-samp` at
+2.27 % with a range of 0.05 points does not look like noise.

**Do not act on this yet.** One night, one depth, and free VRAM happened to span
only 2,872-3,016 MiB across these boots -- a fifth of the 9,326-10,732 MiB
spread the floor was derived from. A quiet night is not a smaller floor. The
honest next step is to re-derive the floor from `--fixed-text` control boots
across a normal VRAM spread, and only then decide whether 13.6 % should move.

---

## 5. State at close

`scripts/afk-q38-night2.sh` finished at **03:12:59**, both steps green. Nothing
is on the GPU; port 8080 is free; the stale `.port8080.lock` from job 99082 is
still on disk and `swap-model.sh` will clear it on the next swap.

The batch doubled as the end-to-end check of tonight's harness changes:
**sixteen server boots, sixteen clean teardowns**, every VRAM handoff through
`wait_for_vram_release()` instead of a five-second guess, and no step took out
the one behind it.

### Verified, and what is not

- **Verified from data.** Every number in sections 1, 2 and 4 comes from a row
  in `results/*.jsonl`. The 92-test gate passes. The doc map has 85 relative
  links and none broken (`scripts/check-doc-links.py`).
- **Verified live.** `kill()` was exercised on both branches at 03:15: with
  nothing listening it adds no poll cycle, and against a real 11,501 MiB server
  it waited 9.87 s for a 9,924 MiB release. That is the fix working on the exact
  path that failed at 02:35.
- **Not verified end-to-end.** The abandonment path -- the branch that writes
  `note="abandoned after ..."` instead of raising -- has unit coverage of its
  inputs, and its consumer contract was checked by reading rather than running
  (`kv_sweep.py:234` guards on `row.get("loaded") and tg`, so a row without
  `tg_med` prints *unpaired* instead of crashing). **No arm has actually timed
  out since the change.**

### Open, carried forward from report 22 section 7

The grammar-with-reasoning question, a corpus at 128K under n-gram, deep
retrieval quality on anything but Q4, the desktop's 2,202 MiB, and the V3
`UD-IQ2_S` ceiling.

**Answered and removed:** `-ot` on `AD-IQ1_M` (section 2, negative);
`ngram-map-k` at 131,072 (section 1 -- it loses there); the four placement
levers (section 4, all inert).

**Newly open:**

1. **Re-derive the 13.6 % floor from `--fixed-text` control boots.** Tonight's
   paired rounds repeated to within 0.05 points across separate boots. If that
   holds across a normal VRAM spread, the floor is hiding small true effects --
   `backend-samp` at +2.27 % being the first candidate. Section 4.
2. **Give `model_arena.py` and `sweep_runtime.py` the depth-aware budget.** Both
   still carry a flat `timeout=1800`. Neither has been near it, because both use
   short prompts, but it is the same fault in the same shape. Section 3.
3. **A corpus at 128K under `ngram-mod`.** The shipping recommendation now
   differs by depth and the deep arm has no quality measurement at all.
