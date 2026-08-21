# qwen38-tuning — the machine

Everything that runs. Findings live in
[`../docs/reports/`](../docs/reports/); this is the apparatus that produced them.

```text
bench\      the harness — 103 tests. Run them before trusting anything
scripts\    launch profiles (production-*.ps1, serve-*.ps1) and unattended queues (afk-*.sh)
results\    raw JSONL, one row per boot. The source of every number in the reports
logs\       llama-server output per boot, plus afk-driver.log — the queue timeline
grammars\   GBNF files that constrain output format
slots\      KV cache save/restore target for --slot-save-path
```

---

## Before you touch the GPU

**1. Is something already running?**

```sh
tail -5 logs/afk-driver.log
netstat -ano | grep ':8080 ' | grep LISTENING
ps -ef | grep "bash scripts/afk-q"
```

Queues chain by waiting for a literal line in `afk-driver.log`. Several can be
armed at once, each sleeping until its predecessor writes its completion line.

**2. Take the lock.** `scripts/swap-model.sh <profile.ps1> <expect-substring>` is
the only correct way to change what serves port 8080. It holds a lock keyed to
the calling job's PID and refuses to swap under a live one.

That guard exists because an armed queue killed a running corpus at 02:00:17 and
the summary still printed "22.0 merged tasks/hour" from 26 tasks that returned
HTTP 503 in 0.0 s.

**`bench/ctx_ceiling.py` does not take the lock** — it kills whatever listens on
8080 directly. Never run it beside another job.

**3. Run the test gate.**

```powershell
cd bench ; python -m pytest tests\ -q     # 103 tests
```

A broken instrument returns a number instead of a failure.

---

## What to serve

| you want | profile |
|---|---|
| 16K, best measured tasks/hour | `scripts/production-iq2xxs.ps1` — 27/30 corpus, **48.5 verified/hr (8,192 budget; the 60.8 quoted in older reports is the 3,072 measurement — see docs/reports/CORRECTIONS.md §6) tasks/hr** |
| 16K, same but ~2× decode | `scripts/production-iq2xxs-ngram.ps1` — adds `--spec-type ngram-map-k`, byte-identical output *(not yet verified at depth or on this artifact)* |
| retrieval-critical deep work | `scripts/production-q4-deep.ps1` — deep quality is verified on Q4 alone |
| 128K fully resident | `scripts/serve-v3-iq2xxs.ps1` — reaches 131,072 at `65+0`, but 58.3 % of attempts emit no fenced code block |

---

## Stale files at this level

`EXPERIMENTS.md`, `FINAL-REPORT.md` and `HANDOFF-BACK.md` date from 2026-08-18
and predate both Experiment A and the Unsloth V3 republish. They are kept for
history. **Use [`../docs/reports/START-HERE.md`](../docs/reports/START-HERE.md)
instead.**
