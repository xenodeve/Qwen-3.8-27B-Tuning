# C:\AI — Local Coding Worker

A Qwen3.8-27B coding agent running on one RTX 4070 SUPER (12 GB).
Claude Code → Xeno → OpenClink → OpenCode → `llama-server`.

**Metric:** verified accepted coding tasks per hour — a task counts only if the
generated code runs and passes its tests.
**Current goal:** a usable context of 128K or more, fully GPU-resident, at the
highest tok/s achievable.

---

## → If you are new, read [`docs/reports/START-HERE.md`](docs/reports/START-HERE.md)

One document: what was tried, what it cost, what was learned, what is still
open. Everything else is the detail behind it.

---

**Before quoting any number:** [`docs/reports/CORRECTIONS.md`](docs/reports/CORRECTIONS.md)
lists ten claims this project published and later contradicted with its own
measurements.

---

## The map

```text
C:\AI\
├── README.md                  ← you are here
├── docs\                      what we know          → docs/README.md
│   ├── reports\               findings, numbered 00-24
│   ├── tested\                the register — has X been tried, what happened
│   ├── plans\                 what we intend to run next
│   └── researchs\             external material, NOT our measurements
├── scripts\                   tools for the docs map → scripts/README.md
└── qwen38-tuning\             the machine           → qwen38-tuning/README.md
    ├── bench\                 the harness (103 tests)
    ├── scripts\               launch profiles and unattended queues
    ├── results\               raw JSONL, one row per boot
    ├── logs\                  server and driver logs
    └── grammars\              GBNF output constraints
```

**Every folder has a `README.md` that says what is in it and what to read
first.** If you land somewhere and are unsure, read that folder's README.

---

## The three rules that matter most

1. **Never compare raw decode across boots.** Free VRAM at boot moves
   9,326–10,732 MiB and `--fit` follows it; the same control config spans
   32.4–42.5 tok/s across 25 boots. **Effects below 13.6 % are noise.** Pair
   within a round.
2. **Two orchestrators cannot share port 8080.** `scripts/swap-model.sh` takes a
   lock. An armed queue once killed a running corpus and the summary still
   printed a plausible number.
3. **Run the test gate before trusting any measurement:**
   `cd qwen38-tuning\bench ; python -m pytest tests\ -q` — 103 tests.

Twelve more in [`docs/reports/04-MEASUREMENT-METHODOLOGY.md`](docs/reports/04-MEASUREMENT-METHODOLOGY.md) §7,
each of which produced a believable wrong number.
