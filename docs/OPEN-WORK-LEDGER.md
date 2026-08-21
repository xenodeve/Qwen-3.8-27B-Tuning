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

## Shipping — the gates

| status | item | why it is open | where |
|---|---|---|---|
| 🟡 **TRACKED** | **CI cannot run on this account** — every job fails in 2 s on a billing lock, so `lint`/`typecheck`/`test` can never go green | The `required_status_checks` rule was removed from `main-protection` and `requireGreenCI` set to `false`, both on a developer-initiated waiver dated 2026-08-21. In exchange `.claude/t4.json` `verify` now runs **all three** checks locally instead of `pytest` alone. **The web UI and other clones are unguarded until this is reverted.** | issue #15, `.claude/t4.json` `ciWaiver` |

---

## Measurement — the machine

| status | item | why it is open | where |
|---|---|---|---|
| 🔴 **UNTRACKED** | **Grammar + drafter has never been run together** | Source says a grammar costs no VRAM (`src/llama-grammar.cpp` allocates nothing on device) and that it disables backend sampling (`common/sampling.cpp:421`), which is free here because that flag defaults off and measured inert at +2.27 %. **But `common.h:331` is a second field — the drafter's own backend sampling, on by default — and the disable does not reach it.** The profile we intend to serve needs a grammar (41.5–58.3 % of attempts emit no fenced block without one) *and* a drafter. Every run so far has had one or the other. **Read from source, not measured** | [tested 05](tested/05-runtime-flags.md) |
| 🟡 **TRACKED** | **DFlash 2 is loadable but unmeasured** | Build 10499 (PR #27342, commit `1deefcca3`) is staged at `C:\AI\llama.cpp-dflash2` and the drafter loads — `scripts/probe-dflash2-load.ps1` exits 0. **No speed number exists on this card.** Two hazards found while building the instrument: `--fit` cannot measure the drafter's VRAM at all, and `--spec-draft-n-max` caps at 7, not 8. Build 10472 is untouched so the two can be paired in one round | issue #17, [tested 02](tested/02-decoders.md) |
| 🔴 **UNTRACKED** | **No published number measures the stated metric** | Every result in `docs/reports/` is tok/s. Verified accepted coding tasks per hour has never been measured, and neither has the context a real task consumes — which is why three worker profiles shipped mis-sized in one day. [Plan 06](plans/06-REAL-TASK-BENCHMARK.md) is the runbook: 19 real issues, an FP8 ceiling on the same model, skills on/off, standard vs clink. **Not started; no issue yet** | [plan 06](plans/06-REAL-TASK-BENCHMARK.md) |
| 🟡 **TRACKED** | **Was trading `UD-IQ2_S` for `UD-IQ2_XXS` + a drafter a good trade?** | The row that used to sit here said IQ2_S had "never been loaded once". **That was false when written or shortly after** — `v3-iq2s` has 38+ measured rows across six result files, dozens of logs, and four `worker-iq2s-*.ps1` profiles, one of which is the recommended one. The real open question is the **trade**: IQ2_S was given up deliberately to free VRAM for DFlash2, and DFlash2 only became loadable on 2026-08-22. The trade is finally answerable — [CORRECTIONS §19](reports/CORRECTIONS.md) | issue #18, [plan 06 §3.5](plans/06-REAL-TASK-BENCHMARK.md) |
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
