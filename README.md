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

**Ten icons, and the columns are the real choice.**

| | one card, `UD-Q2_K_XL` | **both cards, `UD-Q4_K_XL`** | both cards **+ `draft-mtp`** | both cards **+ DFlash2** | **both cards, NVFP4** |
|---|---|---|---|---|---|
| loopback only | [`serve.bat`](serve.bat) | [`serve-dual.bat`](serve-dual.bat) | [`serve-dual-mtp.bat`](serve-dual-mtp.bat) | [`serve-dual-dflash.bat`](serve-dual-dflash.bat) | [`serve-dual-nvfp4.bat`](serve-dual-nvfp4.bat) |
| reachable from other machines | [`serve-lan.bat`](serve-lan.bat) | [`serve-dual-lan.bat`](serve-dual-lan.bat) | [`serve-dual-mtp-lan.bat`](serve-dual-mtp-lan.bat) | [`serve-dual-dflash-lan.bat`](serve-dual-dflash-lan.bat) | [`serve-dual-nvfp4-lan.bat`](serve-dual-nvfp4-lan.bat) |

**The `nvfp4` pair is the fastest thing measured here, and it is the cheapest to
reach.** At ctx 147,456, three paired rounds rotated on real vendor code:
**39.4 / 42.6 / 42.6 tok/s against 24.9 / 25.7 / 25.7** for `serve-dual.bat`
measured in the same rounds — **+63.1 %** [+58.3, +65.6], baseline spread 3.3 %.

Unlike the `dflash` pair it costs **no patch, no second model and no unreviewed
binary**: the speculative head is inside the model file and it runs on the same
`llama.cpp-blackwell` every other icon uses. It even finishes a large request
with **more** room than the default — about 2,395 MiB against 2,010.

Two numbers in it are not preferences. The n-gram runs at **`n-match 24`, not
the `12` every other profile serves**: `12` won on `UD-Q4_K_XL` and is worth a
third less here, and `24` is the value that *lost* on the Q4 at this same depth.
The tuning belongs to the file, not to the depth. And the window is 147,456
against a measured ceiling of **229,376** — verified by pushing a 65,643-token
request through it, which finished with 846 and 526 MiB free. 262,144 does not
come up, so this pair does not ask for the deepest window that fits.

**What it changes is the model file, and that is why it is an icon and not the
default: quality has not been measured.** Not here and not on any artifact this
project serves. What *is* measured is that the n-gram decoder's acceptance falls
from **55.4 to 22.1** on this file — it writes text the predictor cannot
anticipate, which is evidence it writes *differently*. Whether differently is
worse is exactly what nobody knows.

**The `dflash` pair is more than twice as fast and gives up half the window.**
Measured 2026-08-27, three paired rounds on real vendor code at ctx 65,536:
**65.1 / 64.3 / 63.8 tok/s against 29.0 / 29.0 / 28.4** for the `ngram-mod` the
other dual launchers serve — **+123.8 %** [+121.9, +125.1].

It costs three things, which is why it is its own icon and not a default:

- **A patched llama.cpp.** Unpatched, the drafter aborts — `TOP_K` cannot read
  logits the tensor split scatters across two cards. The patch mirrors the
  output projection, costs **1,080 MiB** measured, and **has been reviewed by
  nobody outside this project**.
- **A window capped at 131,072**, against about 250,000 from `serve-dual.bat`.
  That cap is not a budget the launcher can stretch: **147,456 loads, answers a
  health check, and dies on the first real request.**
- **Almost all the headroom** — roughly 600 MiB per card after a large request,
  against about 2,210. A run here died with 336 MiB free and survived with 488.

**And its rate at 131,072 has never been measured.** The +123.8 % is at 65,536,
and a verdict at one depth does not transfer here: at 147,456 a *better* drafter
measured *slower*, because verify cost dominates at depth. Expect less.

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
`Qwen3.8-27B-Q2_K_XL`, the `dual` pair as `Qwen3.8-27B-Q4_K_XL`, and the `nvfp4`
pair as `Qwen3.8-27B-NVFP4-MTP`. It used to be
`qwen38` for both, which told a client nothing and left a saved transcript
unable to say afterwards which one had answered. **A client configured with the
old name needs updating** — that string is all this rename changes; no file on
disk moved and no measured row means anything different.

From a terminal, the same six with flags:

**The four `dual` launchers now serve the deepest window that fits**, capped at
the model's own `n_ctx_train` of 262,144. It is **computed at launch, not
fixed** — 262,144 loaded on this machine when the desktop held ~1,600 MiB and
ran out of memory at 2,575, so the number moves with what you have open. Two
real boots minutes apart settled on **249,856** and **245,760**; the window it
chose is printed when it starts.

They spend the micro-batch before the context: halving `-ub` frees about a
gigabyte across the pair for roughly **3.5 %** of prefill, where the same memory
bought with context costs tens of thousands of tokens.

**Deep is measured, not comfortable.** At full depth a large request finishes
with a few hundred MiB spare against about **2,000** at the 147,456 default — a
run with **336 MiB** free died on its first request, one with **488** survived
135,233 tokens. To pin a shallower, roomier window, call the profile directly:

```powershell
& .\qwen38-tuning\scripts\worker-q4-dual.ps1 -Ctx 147456
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
