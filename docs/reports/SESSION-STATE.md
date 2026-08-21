# Session State — live operational picture

> **Rewritten 2026-08-21 02:30 UTC+7.** Overwrite this file; never append.
> The narrative of how these numbers were reached is
> [report 22](22-SESSION-RECORD-2026-08-20.md).

---

## 1. The config to run today

`scripts/production-iq2xxs-ngram.ps1` — pre-V3 `UD-IQ2_XXS`, 16K, `q4_0` KV,
plus **`--spec-type ngram-map-k`** -- but only at 16K. **At 131,072 use**
**`--spec-type ngram-mod`** instead: +200.22 % against the baseline where
`ngram-map-k` gives +120.54 %. Measured 2026-08-21, report 23 section 1.

The base artifact is the standing default: **27/30 accepted (90 %), 48.5 verified / 26.5 merged tasks per hour** at the standard 8,192 budget. *(The 60.8 quoted elsewhere is `verified_tasks_per_hour` at 3,072 and is not comparable — CORRECTIONS.md §6.)*,
the best this project has measured. The flag adds **+135.89 %** decode
(41.81 → 93.75 tok/s, four rounds under `--fixed-text`) for zero VRAM and
byte-identical output.

**Two caveats are written into the file itself** — the flag was measured on
**V3** `IQ2_XXS`, not the pre-V3 artifact this profile serves, and it is not yet
verified at depth or for quality in that combination.

For retrieval-critical deep work: `production-q4-deep.ps1`. Deep-context quality
is still verified on Q4 alone.

---

## 2. The headline result of 2026-08-20

```text
v3-iq2xxs, q4_0 KV, 65+0, no VRAM added
  16,384    none                        41.81 tok/s
  16,384    --spec-type ngram-map-k     93.75      +135.89 %
 131,072    none                        26.50
 131,072    --spec-type ngram-mod       81.46      +213.08 %
```

**Speed at 128K is no longer the blocker. Quality is.**

---

## 3. What is running

**As of 2026-08-21 03:15:** `scripts/afk-q38-night2.sh` -- the four placement
arms at 16K, then `ngram-map-k` vs `ngram-mod-short` at 131,072. The previous
queue (`afk-q38-followup.sh`) ended at 02:35 with its last two steps failed;
report 23 section 3 says why and what was fixed.

```sh
tail -20 logs/afk-driver.log
ps -ef | grep "bash scripts/afk-q"
```

Queues chain by waiting for a literal completion line in `afk-driver.log`.
**One process per stage — check it.** A duplicate means two orchestrators racing
for port 8080, which destroyed a corpus run at 02:00:17 on 2026-08-20.

---

## 4. The 128K picture

| artifact | resident to | corpus | blocked by |
|---|---:|---|---|
| V3 `UD-IQ1_S` | 196,608 | 0 accepted | quality |
| V3 `UD-IQ1_M` | 163,840 | 10/21 | quality |
| V3 `UD-IQ2_XXS` | **147,456** (was 131,072; report 24 found the gap unmeasured) | 19/27 · 58.3 % contract pass | quality |
| `AD-IQ1_M` | **does not reach 128K** -- 6.08 tok/s at `65+1`, report 23 s2 | **27/30** | the CPU layer itself |
| pre-V3 `UD-IQ2_XXS` | `58+7` | **27/30 · 48.5 verified/hr** (8,192 budget) | depth |

The `-ot` route to `AD-IQ1_M` is closed: it frees the layer by moving 644 MiB
of FFN to CPU and prefill drops from 240.6 to **8.56 tok/s**. Freeing ~125 MiB
some other way -- smaller KV, smaller batch, the desktop's 2,202 MiB -- is
still open; putting weights on the CPU is not.

**Strategy:** tune to a stable config on the Q2 artifact, then carry it to Q1
without discarding Q1. Most of what was found is artifact-independent — n-gram
works on tokens, KV type is universal, the inert flags are inert everywhere.
Only `-ot` needs re-measuring per artifact.

**Choose the artifact by quality, then make it fast** — not the reverse. V3
`IQ1_S` is the fastest artifact ever measured here and produced zero accepted
tasks.

---

## 5. Harness

`bench/` — **103 tests.** `cd bench ; python -m pytest tests\ -q` before trusting
anything.

New in this session:
- **`kv_sweep --fixed-text`** — pins temperature 0 and a fixed seed for the
  **timed** generations. **Required for any content-dependent lever.** Without
  it, `ngram-cache` returned +80.79 % and −30.56 % in two sweeps three hours
  apart, both passing the paired test. `fixed_text` is recorded on every row.
- `kv_sweep.ARMS` is now 48 arms; `ctx_ceiling.py` takes `--extra` / `--tag` and
  records both.
- `swap-model.sh` allows a re-swap when the lock owner **is** the caller — a
  sweep that swaps once per config was refusing itself after the first.
- `grammars/python-fence.gbnf` + `serve-v3-*-fmt.ps1`, which use **`-rea off`**,
  not `--reasoning-budget 0` (that flag does not end the block despite its docs).

---

## 6. Rules that are cheap to forget and expensive to relearn

1. **Address artifacts by exact path plus byte count**, never `-hf repo:tag`.
   `:Q2_0` fetched `PQ2_0.gguf` — a different file of identical byte count.
2. **Budget output tokens for the most verbose arm** and record `finish_reason`.
3. **Two orchestrators cannot share port 8080.** Take the lock. Note that
   `ctx_ceiling.py` kills whatever listens on 8080 **without** taking it.
4. **Never compare raw decode across boots.** Pair within a round; floor 13.6 %.
5. **`resolved` is necessary, not sufficient.** Below ~512 MiB free an arm can
   pass the sign test and still be unstable.
6. **Desktop VRAM is a live variable.** 33 processes held **2,202 MiB** during
   these runs. Any arm within ~1 GB of its ceiling is conditional on what else is
   open. Read `free_before`.
7. **Past full residency, spare VRAM is inert for speed** — it buys depth only.
8. **Never edit a running queue script.** bash reads incrementally by byte
   offset. Kill, edit, relaunch, then verify exactly one instance.
9. **`output_contract_pct` is the PASS rate.** Higher is better. It was read
   backwards for a full day.
10. **Verify a flag parses before spending a boot on it:**
    `llama-server <flags> -m /nonexistent.gguf --port 18080`.
