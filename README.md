# C:\AI — Local Coding Worker

A Qwen3.8-27B coding agent running on one **RTX 5060 Ti (16 GB)**.

> **The card changed on 2026-08-23.** It was an RTX 4070 SUPER 12 GB, and
> **every measurement in this repo predating that date was taken on the old
> card.** What transfers and what does not:
> [`docs/results/09-hardware.md`](docs/results/09-hardware.md).

Claude Code → Xeno → OpenClink → OpenCode → `llama-server`.

## Just start it

**Double-click [`serve.bat`](serve.bat).** The server runs *in* that window and
its output is that window's output. `Ctrl+C` stops it, and so does closing the
window — there is one process, not a server beside a log-watcher.

**Four icons, two independent choices.**

| | one card, `UD-Q2_K_XL` | **both cards, `UD-Q4_K_XL`** | both cards **+ `draft-mtp`** |
|---|---|---|---|
| loopback only | [`serve.bat`](serve.bat) | [`serve-dual.bat`](serve-dual.bat) | [`serve-dual-mtp.bat`](serve-dual-mtp.bat) |
| reachable from other machines | [`serve-lan.bat`](serve-lan.bat) | [`serve-dual-lan.bat`](serve-dual-lan.bat) | [`serve-dual-mtp-lan.bat`](serve-dual-mtp-lan.bat) |

**The `mtp` pair has no measured speed.** `draft-mtp` does run on the two-card
split — verified 2026-08-27, after this project had wrongly recorded that it
could not — but **every paired measurement of it was voided by our own output
guard, because the generations copy the prompt instead of answering it.** Three
unpaired manual readings looked excellent and are exactly what that guard exists
to reject: a speculative decoder gets faster the more predictable the text is.
Click it to try it; `serve-dual.bat` is the one with a number behind it.

The `lan` files are separate rather than a prompt because `--host` is the **only
access control this server has** — no API key, CORS `*` — so exposure should
happen because someone clicked the thing that says `lan`, not because they
wanted the model running.

**The `dual` files are not a free upgrade.** `UD-Q4_K_XL` is 16.69 GiB and does
not fit on one 16 GB card at any depth; across both it is fully resident to
229,376 and runs at **parity** with the single-card profile — 32.4/32.6/33.1
against 32.1/32.0/32.0 at ctx 147,456. But it draws roughly **130 W more**, it
**needs both cards installed** (with one it refuses to start rather than quietly
serving something else), and its **quality has never been measured here** — every
reason to prefer that artifact comes from a bits-per-weight ladder and an
external campaign, neither of which is our number. Which icon is right is a
decision, which is why none of the four implies another.
[Issue #52](https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/52),
[`docs/results/09-hardware.md`](docs/results/09-hardware.md).

**The model announces which artifact it is.** `serve.bat` serves it as
`Qwen3.8-27B-Q2_K_XL`, the `dual` pair as `Qwen3.8-27B-Q4_K_XL`. It used to be
`qwen38` for both, which told a client nothing and left a saved transcript
unable to say afterwards which one had answered. **A client configured with the
old name needs updating** — that string is all this rename changes; no file on
disk moved and no measured row means anything different.

From a terminal, the same six with flags:

**To reach the model's full 262,144 native context**, which no launcher does by
default, call the profile directly — it costs about 3.5 % of prefill and leaves
only ~450 MiB spare against ~2,000 at the default depth:

```powershell
& .\qwen38-tuning\scripts\worker-q4-dual.ps1 -Ctx 262144 -UBatch 512
```

```powershell
.\serve.ps1                          # one card, loopback
.\serve.ps1 -Lan -AllowFirewall      # one card, exposed
.\serve.ps1 -Dual                    # both cards, loopback
.\serve.ps1 -Dual -Lan -AllowFirewall
.\serve.ps1 -Dual -Mtp               # both cards + draft-mtp, rate unmeasured
.\serve.ps1 -Dual -WhatIf            # print what it would run, touch nothing
```

That is the whole answer. `qwen38-tuning/scripts/` holds **58 `.ps1` files** and
several of them serve artifacts that stopped being the default at windows that
stopped being the answer; nothing in the tree said which was current.
[`serve.ps1`](serve.ps1) resolves the profile the evidence supports, **refuses to
start over a port already answering**, and **reads the layer split back out of
the boot log** — `--fit` spills rather than refusing, so residency is checked,
never assumed. It prints what each choice rests on and names the one still open.

`.\serve.ps1 -WhatIf` shows the resolved command without touching the GPU.

**It holds no configuration of its own.** The flags live in
`qwen38-tuning/scripts/worker-q2kxl-mtp.ps1` and only there — a launcher that
copied them would become a second source of truth and drift silently, since both
files would still run.

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
lists twenty-eight claims this project published and later contradicted with its own
measurements.

---

## The map

```text
C:\AI\
├── README.md                  ← you are here
├── docs\                      what we know          → docs/README.md
│   ├── reports\               findings, numbered 00-32
│   ├── results\               the register — has X been tried, what happened
│   ├── plans\                 what we intend to run next
│   └── researchs\             external material, NOT our measurements
├── scripts\                   tools for the docs map → scripts/README.md
└── qwen38-tuning\             the machine           → qwen38-tuning/README.md
    ├── bench\                 the harness (329 tests)
    ├── scripts\               launch profiles and unattended queues
    ├── results\               raw JSONL, one row per boot
    ├── logs\                  server and driver logs
    └── grammars\              GBNF output constraints
```

**Every folder has a `README.md` that says what is in it and what to read
first.** If you land somewhere and are unsure, read that folder's README.

---

## The three rules that matter most

1. **Never compare raw decode across boots.** The same control config spans
   **32.4–42.5 tok/s across 25 boots**, and the **cause is unknown** — the old
   explanation that `--fit` follows the boot VRAM is retracted, because
   llama.cpp has reported 11,069 MiB free in all 552 logs and 148 of 150 boots
   say *"no changes needed"* (`docs/reports/CORRECTIONS.md` §27).
   **Effects below 13.6 % are noise** — **at ctx 16,384, where that floor was measured.** At 65,536 the same arm with byte-identical counters spans up to **48.9 %** across boots, so re-derive before using it at depth (`CORRECTIONS.md` §23). Pair
   within a round.
2. **Two orchestrators cannot share port 8080.** `qwen38-tuning\scripts\swap-model.sh` takes a
   lock. An armed queue once killed a running corpus and the summary still
   printed a plausible number.
3. **Run the test gate before trusting any measurement:**
   `cd qwen38-tuning\bench ; python -m pytest tests\ -q` — 329 tests.

Twelve more in [`docs/reports/04-MEASUREMENT-METHODOLOGY.md`](docs/reports/04-MEASUREMENT-METHODOLOGY.md) §7,
each of which produced a believable wrong number.
