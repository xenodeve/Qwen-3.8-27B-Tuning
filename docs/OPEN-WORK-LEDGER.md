# Open work ledger

**The discovery index over everything still open.** GitHub issues are the source
of truth for tracked work; this table also catches the **MD-only** items that
`gh issue list` cannot see — and those are the ones that vanish.

This project has already lost work that way twice in one day:
`results/reasoning-effort-sweep.jsonl` and `results/prefix-cache.jsonl` both held
answers to questions later written up as open, because a result that lives only
inside a report is a result nobody can find.

**Read this at session start**, before picking anything up.

| status | means |
|---|---|
| 🔴 **UNTRACKED** | no GitHub issue exists — highest miss-risk |
| 🟡 **TRACKED** | has an issue; the row is a pointer |
| 🔵 **BLOCKED** | needs something we do not have |

---

## Measurement — the machine

| status | item | why it is open | where |
|---|---|---|---|
| 🔴 | **`UD-IQ2_S`, the untested rung** | 8.37 GB, in the local cache since 2026-08-20 01:36, never loaded once. Sits between the artifact that fails on format (2.16 bpw) and the one that works (2.64). Registered as `v3-iq2s` | [plan 04 P2](plans/04-REVISED-PLAN-2026-08-21.md) |
| 🔴 | **Does the 160-token probe understate speculation?** | every decoder verdict — `draft-mtp`, `draft-dflash`, eagle3, dspark — was decided on 160-token generations. `scripts/afk-q38-warmup.sh` exists and was interrupted | [CORRECTIONS §8](reports/CORRECTIONS.md) |
| 🔴 | **Re-derive the 13.6 % noise floor under `--fixed-text`** | paired rounds now repeat to within 0.05 points across boots, two orders of magnitude tighter than the floor. May be hiding small true effects | [report 23 §4](reports/23-SESSION-RECORD-2026-08-21.md) |
| 🔴 | **The desktop's 1,650–2,200 MiB** | the largest untouched lever on this machine, and it needs no code | [tested 03](tested/03-memory-and-kv.md) |
| 🔴 | **Why speculation dies at `65+0` on 163,840** | four independent routes to full residency all lose the drafter. `-ot` moves weights to CPU; `--fit-target` and `-ub` do not, so float divergence cannot be the whole story | [report 24 §1b](reports/24-BEYOND-128K.md) |
| 🔵 | **`model_arena.py` / `sweep_runtime.py` flat `timeout=1800`** | same fault class as the one fixed in `depth_sweep`; neither has been near its limit, so it is real but not urgent | [CORRECTIONS §8](reports/CORRECTIONS.md) |

## Quality — the worker

| status | item | why it is open | where |
|---|---|---|---|
| 🔴 | **Grammar alone, reasoning left on** | the only arm that showed the 26-point contract jump changed two things at once. `scripts/serve-v3-iq2xxs-gram.ps1` was built 03:29 and aborted 03:30 | [tested 06](tested/06-prompt-and-quality.md) |
| 🔴 | **How a long-thinking model survives a multi-turn agent loop** | the sharpened version of the retracted "it loops" claim. `damerau` takes 62.6 s direct and 247.6 s through OpenCode with no output | [post-mortem](reports/2026-08-21-inferred-looping-from-three-numbers.md) |
| 🔴 | **`reasoning_effort: low` on a 2-bit artifact, through the corpus** | swept on Q4 with a tool probe (6/6 succeeded); never where the failure lives | [tested 05](tested/05-runtime-flags.md) |
| 🔴 | **A system prompt that instructs *how to think*** | every sampler lever is exhausted; the process has never been instructed. Costs nothing | [plan 04 P3](plans/04-REVISED-PLAN-2026-08-21.md) |
| 🔴 | **Deep-context retrieval quality on anything but Q4** | nine artifacts have depth throughput numbers and none has a depth quality number. Open since 2026-08-17 | [tested 06](tested/06-prompt-and-quality.md) |
| 🔴 | **`bench/tap.py` was built and never run** | full per-request telemetry, written 2026-08-21, zero result files. Speculative until it earns a row | [tested 07](tested/07-telemetry-inventory.md) |
| 🔴 | **`--skill` in `run_retry_bench.py` was built and never run** | injects the real `karpathy-guidelines` + `tdd` text the way `clink-subagents` §7 mandates. No script calls it | [tested 06](tested/06-prompt-and-quality.md) |

## Infrastructure

| status | item | why it is open | where |
|---|---|---|---|
| 🔴 | **Required status checks are not enforced yet** | `t4-verify.yml` is committed; `lint`/`typecheck`/`test` still need to be made required on `main` with direct pushes disallowed | `.github/workflows/t4-verify.yml` |
| 🔴 | **`.claude/t4.json` `"verify"` is empty** | the local ship gate is installed but disarmed. Arming it means pointing it at the pytest suite | `.claude/t4.json` |
| 🔴 | **`docs/agents/` conventions not written** | `domain.md`, `workflow.md`, tracker and label docs | this bootstrap |

---

## Closed since this ledger was created

Nothing yet. When an item closes, move it here with the evidence — a commit, a
test, or a measured number — rather than deleting the row. A ledger that only
ever shrinks teaches the next reader nothing about what was tried.
