# C:\AI — Local Coding Worker

**English** · [ภาษาไทย](README.md)

A Qwen3.8-27B coding agent running on **two consumer cards in one desktop** — an
**RTX 5060 Ti 16 GB** (Blackwell, `sm_120`) beside an **RTX 4070 SUPER 12 GB**
(Ada, `sm_89`), 28 GB of VRAM between them. Everything measured here is on that
pair, split with `-sm tensor`.

Claude Code → Xeno → OpenClink → OpenCode → `llama-server`.

---

## What it does, in one table

| | |
|---|---|
| **Primary artifact** | **`Qwen3.8-27B-NVFP4-MTP-VERY-LOW`**, 13.8 GiB, with an MTP speculative head baked into the file |
| **Sustained decode, real sessions** | **60–66 tok/s** at ctx 147,456 |
| **Best paired measurement** | **+63.1 %** over the previous default, three rounds rotated |
| **Deepest window that survives a real request** | **200,704** |
| **Images** | yes, at every depth served, +888 MiB |
| **Quality** | **never measured, on any artifact here.** This is the open question, not a footnote |

**The rate is the one number people ask for, so here is exactly where it comes
from.** In a real Claude Code session on 2026-08-30, NVFP4 VERY-LOW drafted by
DFlash2 at ctx 147,456 held **66.27 tok/s across 11,218 generated tokens**
(`qwen38-tuning/logs/serve-20260830-084604.log`), and **60.58 across 7,725** on
the separately built Unsloth 0.3.0 binary
(`serve-20260830-102405.log`). Those are long generations, not a lucky first
second — the same logs show 65.93 and 65.53 at 11,362 and 10,893 tokens.

**They are also not paired measurements, and this project does not let those two
things be confused.** Under the benchmark harness, on its frozen vendor-code
corpus at the same depth, the same pairing reads **44.5 tok/s**. The gap is the
workload: real agent traffic is far more predictable than the corpus, and a
speculative decoder gets faster the more predictable the text is. **Quote 44.5
when comparing configurations. Quote 60–66 when describing what the machine
feels like.** Never swap them.

---

## Just start it

**Double-click [`serve-hub.bat`](serve-hub.bat).** It is the only `.bat` at the
top of this repository. It asks two questions — which server, and whether to
expose it on the LAN — then hands off to one of the eighteen in
[`launchers/`](launchers/). It holds no flags of its own; it only picks a file.

Every launcher also works if you open that folder and double-click it directly.
Each one carries its own evidence in its header, and there are tests that
**refuse to let a launcher claim a speedup it does not have**.

**The server runs *in* that window.** Its output is that window's output,
`Ctrl+C` stops it, and so does closing the window. There is one process, not a
server beside a log-watcher.

### The eighteen icons

**Start here — the NVFP4 family. This is what the project recommends.**

| | 147,456, with images | **200,704**, deepest measured |
|---|---|---|
| loopback only | [`serve-dual-nvfp4.bat`](launchers/serve-dual-nvfp4.bat) | [`serve-dual-nvfp4-deep.bat`](launchers/serve-dual-nvfp4-deep.bat) |
| reachable from other machines | [`serve-dual-nvfp4-lan.bat`](launchers/serve-dual-nvfp4-lan.bat) | [`serve-dual-nvfp4-deep-lan.bat`](launchers/serve-dual-nvfp4-deep-lan.bat) |

**The `UD-Q4_K_XL` family — what was served before, kept for comparison.**

| | one card, `UD-Q2_K_XL` | both cards | both cards **+ `draft-mtp`** | both cards **+ DFlash2** |
|---|---|---|---|---|
| loopback | [`serve.bat`](launchers/serve.bat) | [`serve-dual.bat`](launchers/serve-dual.bat) | [`serve-dual-mtp.bat`](launchers/serve-dual-mtp.bat) | [`serve-dual-dflash.bat`](launchers/serve-dual-dflash.bat) |
| LAN | [`serve-lan.bat`](launchers/serve-lan.bat) | [`serve-dual-lan.bat`](launchers/serve-dual-lan.bat) | [`serve-dual-mtp-lan.bat`](launchers/serve-dual-mtp-lan.bat) | [`serve-dual-dflash-lan.bat`](launchers/serve-dual-dflash-lan.bat) |

**The experiments — each one is a question, not a recommendation.**

| icon | what it is | status |
|---|---|---|
| [`serve-dual-dflash-n4.bat`](launchers/serve-dual-dflash-n4.bat) | DFlash2 at `--spec-draft-n-max 4`, its measured best. **7 is 6.5 % worse** and 308 MiB dearer, with acceptance falling 61.8 → 51.9 | measured |
| [`serve-dual-nvfp4-dflash.bat`](launchers/serve-dual-nvfp4-dflash.bat) | NVFP4 drafted by DFlash2 instead of its own head. **+67.9 % at 65,536.** At the served 147,456 it is +4.0 %, under the noise floor — what it buys there is **steadiness, 0.7 % spread against 9.3 %**, for ~950 MiB | measured; **not** a speedup at the served depth |
| [`serve-dual-nvfp4-dflash-theirmirror.bat`](launchers/serve-dual-nvfp4-dflash-theirmirror.bat) | the same pairing on **Unsloth's 0.3.0 source** with our mirror patch applied — the fourth `llama-server` on this machine | boots and serves; unpaired |
| [`serve-dual-nvfp4-beta.bat`](launchers/serve-dual-nvfp4-beta.bat) | nine settings borrowed whole from Unsloth Studio, which runs this same model file on these same two cards | one boot per side |
| [`serve-dual-nvfp4-clone.bat`](launchers/serve-dual-nvfp4-clone.bat) | Studio's entire command line as a baseline, with **six deliberate deviations** listed in the file header | baseline |

**The EXL3 engine — the second server, hub keys 15 and 16, port 8000.**
ExLlama3 (the Mia-AiLab fork built from source here) serving turboderp's
SC 4.0bpw H5 with the model's own MTP head, integer 4-bit KV, tensor-parallel
across both cards. Since 2026-09-04 this is the daily driver for Claude Code:
it speaks the Anthropic Messages API directly (`claude-xeno-exl3`, no proxy),
holds **262,144** tokens, and decodes at ~81 % of llama.cpp in the one
same-boot pairing taken (47–55 tok/s in real 30–70K sessions). Quality is
measured in [`docs/results/11-quality-bench-2026-09-05.md`](docs/results/11-quality-bench-2026-09-05.md).

| | 163,840 | **262,144**, the served depth |
|---|---|---|
| loopback only | [`serve-exl3.bat`](launchers/serve-exl3.bat) | [`serve-exl3-max.bat`](launchers/serve-exl3-max.bat) |
| reachable from other machines | [`serve-exl3-lan.bat`](launchers/serve-exl3-lan.bat) | [`serve-exl3-max-lan.bat`](launchers/serve-exl3-max-lan.bat) |

The server is ours (`qwen38-tuning/serving/exl3/`, the fork's file plus marked
hooks) and carries three guards the fork does not, each from a fault that
happened here: it **relaunches itself** when its tensor-parallel children die
(#75; stop it on purpose only with `qwen38-tuning\scripts\stop-exl3.cmd`), it
**cuts a generation that has degenerated into repetition** (#76: one Thai
report ran 127,996 tokens on a single tone mark), and it **bans Chinese
characters** unless the prompt carries or names Chinese (#77: 14 Han characters
leaked into 3 of 43 bench streams, always mid-Thai-sentence). `/health` reports
`loops_stopped` and `cjk_chars_total`. The recipe lives in one place,
`qwen38-tuning/scripts/serve-exl3.cmd`; the four launchers pass a depth, the
split caps and the bind address, nothing else.

The `lan` files are separate rather than a prompt because `--host` is the **only
access control this server has** — no API key, CORS `*`. Exposure should happen
because someone clicked the thing that says `lan`, not because they wanted the
model running.

---

## Two changes that invalidate older numbers

> **2026-08-23 — the second card arrived.** Before that this was one RTX 4070
> SUPER 12 GB, and **every measurement predating that date was taken on the old
> single card.** What transfers and what does not:
> [`docs/results/09-hardware.md`](docs/results/09-hardware.md).
>
> **The 4070 SUPER also drives the display**, so its free VRAM moves with
> whatever is on screen. That is not a footnote. The launch profiles compute the
> tensor split from free VRAM at boot and **refuse to start rather than spill** —
> a spill once cost 85× and printed a plausible number while doing it.

---

## Why NVFP4 VERY-LOW, and what it cost to know

**Measured head to head at ctx 147,456**, three paired rounds rotated on real
vendor code: `NVFP4-MTP-VERY-LOW` + `draft-mtp,ngram-mod` at `n-match 24` decodes
**39.4 / 42.6 / 42.6** against **24.9 / 25.7 / 25.7** for the previous default —
**+63.1 %** [+58.3, +65.6], baseline spread 3.3 %.

It needs nothing extra: **the speculative head is inside the model file**, so no
sidecar drafter, no patch, and it runs on the same binary every other icon uses.
It finishes a large request with *more* room than the default — about 2,395 MiB
against 2,010.

**Three things that measurement settled on the way, each of which contradicts
something this project had already published:**

- **The artifact alone is a loss.** NVFP4 with the n-gram decoder but *without*
  MTP is **−22.4 %**, because n-gram acceptance falls **55.4 → 22.1** — that file
  writes text the predictor cannot anticipate. **The pairing is the result;
  neither half is.**
- **The n-gram tuning did not transfer.** `n-match 24` *lost* on `UD-Q4_K_XL` at
  this exact depth and wins by **+27.1 %** here. A verdict does not transfer
  across artifacts any more than across depths.
- **MTP's prompt-copying belongs to the artifact, not to MTP.** This file reports
  `copied_frac [0.0, 0.0, 0.0]` where another head at the same depth reports
  `[0.519, 0.0, 0.23]`. A question that had been open for weeks.

**`MID-HIGH` is also on disk, 15.8 GiB, and has no measured rate at all.**

---

## What is still unknown, stated first rather than buried

- **Quality has never been measured on any artifact this project serves.** The
  harness for it exists — an answer screen, an output-identity check against
  no-speculation, a hundred-turn stability soak, a code-task suite — and every
  row in those result files belongs to an older artifact on the old card.
  **This is the critical path.**
- **The two builds have never been paired.** One reading once suggested a newer
  llama.cpp was 26 % faster; the run that appeared to refute it turned out to
  have measured **one binary twice**. The status is *contested*, not settled.
- **A reproducible stall.** NVFP4 + DFlash2 on the Unsloth 0.3.0 build wedged
  twice on 2026-08-31 — one thread of 36 spinning flat out, both GPUs idle,
  the slot never released. Under investigation; the cause is not known.
- **Nothing above 147,456 has been measured with DFlash2.**

---

## The four llama-server binaries, and why telling them apart matters

| binary | build | our mirror patch | what runs it |
|---|---|---|---|
| `llama.cpp-blackwell` | 10499 | no | the served default |
| `llama.cpp-mirror` | 10499 | **yes** | every `-Dflash` icon |
| `~/.unsloth/…/bin` | 10679 | no | `-TheirBuild` |
| `llama.cpp-unsloth-mirror` | 10679 | **yes** | `-TheirMirror` |

**The mirror patch** maps `output.weight` to a mirrored split instead of a
sharded one. Without it, DFlash2 under `-sm tensor` aborts: `TOP_K` needs a whole
vocabulary row and the split scatters it across two cards. Upstream already does
exactly this for another architecture, so the patch **widens an accepted
exception** rather than inventing one — but it **has been reviewed by nobody
outside this project**.

**Read the fourth binary's banner carefully forever.** It says
`0.3.0-dev (build 215, commit …)`. The version string and the "Compiled by the
Unsloth team" line are theirs; the build number and commit are **our** git,
counted by their build system because the copied tree has no `.git`. A log from
it is not a log from 10499.

---

## The largest speed lever is not on this page

**Measured 2026-08-30, one boot, minutes apart, nothing changed but a checkbox in
the chat client.** The message both times was a two-word greeting:

| | tools ON | tools OFF |
|---|---|---|
| prompt | **17,843 tokens** | **334** |
| prefill | 18,618 ms | **554 ms** |
| decode | 35.20 tok/s | **45.64** |
| the whole answer | **21.5 s** | **1.5 s** |

**17,509 of those 17,843 tokens were tool schemas**, sent on every request by the
client. **Fourteen times on the wall clock, from a checkbox.** Every server flag
in this repository is worth single digits or low double digits by comparison.
Before touching any of them, ask what the client is putting in front of the
user's actual words.

---

## The measurement discipline

This project has published **43 claims it later contradicted with its own
data** ([`docs/reports/CORRECTIONS.md`](docs/reports/CORRECTIONS.md)), and
**thirteen documented instrument faults each produced a plausible wrong number
rather than an error.** That is why the rules below exist, and why they are
stated before any result.

1. **Never compare raw decode across boots.** The same control config spans
   **32.4–42.5 tok/s across 25 boots** and **the cause is unknown**. Effects
   below **13.6 %** are noise *at ctx 16,384, where that floor was measured*; at
   65,536 the same arm with byte-identical counters spans up to **48.9 %**. Pair
   within one round and rotate the order.
2. **Residency before arithmetic.** A delta between an arm that spilled and one
   that did not measures the spill. The harness refuses such a row rather than
   reporting it.
3. **A verdict does not transfer** — not across depth, not across artifact, not
   across workload. One decoder is +81 % at 16K and −71 % at 131,072 on the same
   file.
4. **Loading is not surviving.** Every depth claim is made with a request of
   **half the window**. A ceiling once certified with a quarter-window request
   loads with 206 MiB free and dies on a half-window one.
5. **Run the test gate before trusting any measurement:**
   `cd qwen38-tuning\bench ; python -m pytest tests\ -q` — **1,435 tests**.

**Retracting is part of the work.** When a measurement contradicts something
already published here, the retraction is not finished until the claim has an
entry in `CORRECTIONS.md` **and** a rule in `scripts/audit-stale-claims.py` that
finds every copy of it still sitting in the tree.

---

## From a terminal

```powershell
.\serve.ps1                          # one card, loopback
.\serve.ps1 -Lan -AllowFirewall      # one card, exposed
.\serve.ps1 -Dual                    # both cards, loopback
.\serve.ps1 -Dual -Nvfp4 -Vision     # the recommended pairing
.\serve.ps1 -Dual -Nvfp4 -Vision -Deep     # the same at 200,704
.\serve.ps1 -Dual -WhatIf            # print what it would run, touch nothing
```

[`serve.ps1`](serve.ps1) holds **no configuration of its own** — the flags live
in the worker profile and only there, because a launcher that copied them would
become a second source of truth and drift silently while both still ran. It
resolves the profile the evidence supports, **refuses to start over a port
already answering**, and **reads the layer split back out of the boot log**,
because `--fit` spills rather than refusing and residency must be checked rather
than assumed.

`qwen38-tuning/scripts/` holds **62 `.ps1` files**, several serving artifacts
that stopped being the default at windows that stopped being the answer. Nothing
in the tree said which was current; `serve.ps1` is the answer to that.

**The model announces which artifact it is** on `/v1/models` —
`Qwen3.8-27B-NVFP4-MTP`, `Qwen3.8-27B-Q4_K_XL`, `Qwen3.8-27B-Q2_K_XL`. It used to
be one name for all of them, which told a client nothing and left a saved
transcript unable to say afterwards which one had answered.

---

## The map

```text
C:\AI\
├── README.md                  ไทย (default)
├── README.en.md               ← you are here
├── docs\                      what we know          → docs/README.md
│   ├── OPEN-WORK-LEDGER.md    what is open, including items no issue tracks
│   ├── reports\               findings, numbered 00-39
│   ├── results\               the register — has X been tried, what happened
│   ├── plans\                 what we intend to run next
│   ├── researchs\             external material, NOT our measurements
│   └── agents\                the operating standard, and traps.md
├── scripts\                   tools for the docs map → scripts/README.md
└── qwen38-tuning\             the machine           → qwen38-tuning/README.md
    ├── bench\                 the harness (1,435 tests)
    ├── scripts\               launch profiles and unattended queues
    ├── templates\             the one patched chat template, and why
    ├── results\               raw JSONL, one row per boot
    ├── logs\                  server and driver logs
    └── grammars\              GBNF output constraints
```

**Every folder has a `README.md`** that says what is in it and what to read
first.

---

## Where to read next

| you want | read |
|---|---|
| the whole story, once | [`docs/reports/START-HERE.md`](docs/reports/START-HERE.md) |
| **before quoting any number** | [`docs/reports/CORRECTIONS.md`](docs/reports/CORRECTIONS.md) |
| before proposing any speed work | [`docs/reports/39-OPTIMISATION-GUIDE.md`](docs/reports/39-OPTIMISATION-GUIDE.md) |
| what exactly is being served | [`docs/reports/38-NVFP4-PROFILE-REFERENCE.md`](docs/reports/38-NVFP4-PROFILE-REFERENCE.md) |
| has X been tried already | [`docs/results/README.md`](docs/results/README.md) |
| what is still open | [`docs/OPEN-WORK-LEDGER.md`](docs/OPEN-WORK-LEDGER.md) |
| the ways of *working* that failed here | [`docs/agents/traps.md`](docs/agents/traps.md) |
| twelve more rules, each of which produced a believable wrong number | [`docs/reports/04-MEASUREMENT-METHODOLOGY.md`](docs/reports/04-MEASUREMENT-METHODOLOGY.md) §7 |

**Metric:** verified accepted coding tasks per hour — a task counts only if the
generated code runs and passes its tests.
**Current goal:** quality measured on the artifact now served, so the fastest
configuration can stop being provisional.
