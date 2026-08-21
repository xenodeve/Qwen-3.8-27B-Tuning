# scripts — launch profiles and unattended queues

## Serving

`swap-model.sh <profile.ps1> <expect-substring>` is **the only correct way** to
change what serves port 8080. It takes a lock keyed to the calling job's PID,
kills the old server, starts the new one, and **proves the new model answers a
real request** before returning — `/health` alone once passed while every
inference call returned HTTP 503.

| profile | what it is |
|---|---|
| `production-iq2xxs.ps1` | the standing 16K default — pre-V3 `UD-IQ2_XXS`, 27/30 corpus, 48.5 verified tasks/hr at the 8,192 budget |
| `production-iq2xxs-ngram.ps1` | the same plus `--spec-type ngram-map-k`. **+135.89 % decode at 16K over four fixed-text rounds, byte-identical output.** **Do not use this arm at 128K** -- there `ngram-mod` gives +200.22 % and `ngram-map-k` only +120.54 % |
| `production-q4-deep.ps1` | for retrieval-critical deep work — deep quality is verified on Q4 alone |
| `serve-v3-*.ps1` | Dynamic V3 arms, 16K default, `-Ctx` parameter |
| `serve-v3-*-fmt.ps1` | the same **plus `--grammar-file` and `-rea off`** -- byte-identical to their twins except those two flags, so the pair is a controlled comparison. **Corrected 2026-08-20:** this pair originally used `--reasoning-budget 0`, which does not end the reasoning block; screened at n=3 that combination returned 0/3. See the script header |
| `serve-v3-iq2xxs-flex.ps1` | parameterised: `-Extra '<flags>'`. Used by the sampling sweep to vary one thing at a time |

## Unattended queues (`afk-*.sh`)

Each writes `START` / `DONE` / `FAIL` lines to `../logs/afk-driver.log`, and each
**waits for a literal completion line from its predecessor**, so several can be
armed at once and they run in sequence without colliding.

```sh
tail -20 ../logs/afk-driver.log          # the timeline
ps -ef | grep "bash scripts/afk-q"       # one process per armed stage
```

The 2026-08-20 chain, in order:

```text
afk-qwen38-resident  →  afk-q38-ckpt  →  afk-q38-layers  →  afk-q38-depth-levers
                     →  afk-q38-sampling  →  afk-q38-decoder  →  afk-q38-quality
                     →  afk-q38-followup
```

### Two rules for editing a queue script

**Never edit a running one.** bash reads a script incrementally by byte offset;
changing its length while it sleeps in a wait loop makes it resume at the wrong
place. Kill it, edit, relaunch, then **confirm exactly one instance is alive** —
two orchestrators racing for port 8080 is what destroyed a corpus run at
02:00:17.

**Capture the exit code before anything else runs:**

```sh
else
  rc=$?                                  # $(date) would reset $? to 0
  echo "[$(date +%T)] FAIL $n (rc=$rc)" >> "$LOG"
fi
```

A `FAIL … (rc=0)` line in the log is this bug, and it silently discards the only
diagnostic the step produced.
