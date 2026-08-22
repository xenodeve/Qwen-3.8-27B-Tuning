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
| 🔴 **UNTRACKED** | **Grammar + drafter has never been run together** | Source says a grammar costs no VRAM (`src/llama-grammar.cpp` allocates nothing on device) and that it disables backend sampling (`common/sampling.cpp:421`), which is free here because that flag defaults off and measured inert at +2.27 %. **But `common.h:331` is a second field — the drafter's own backend sampling, on by default — and the disable does not reach it.** The profile we intend to serve needs a grammar (41.5–58.3 % of attempts emit no fenced block without one) *and* a drafter. Every run so far has had one or the other. **Read from source, not measured** | [results 05](results/05-runtime-flags.md) |
| 🔴 **UNTRACKED** | **The worker changes nothing even with room to spare** | At ctx 98,304, four of five real GitHub issues ran **1,427–2,400 s** and produced **zero file changes** with a green verify — which is a FAIL, because it passes tests that were already passing. Every baseline was green, so none is an environment failure. **No mechanism attached.** The OpenCode transcript is written beside the clone and deleted with the scratch root — capture it before the next run | [results 04](results/04-context-depth.md), [report 31 §6](reports/31-SESSION-RECORD-2026-08-22.md) |
| 🔴 **UNTRACKED** | **`--fit-target` does not mean what every worker profile says it means** | `server-context.cpp:1074` **adds the draft model's bytes** to `fit_params_target` before `--fit` runs. With the DFlash2 sidecar (1,090 MiB) our `--fit-target 768` reaches `fit.cpp` as roughly **1,900–2,100 MiB**. Every profile header describing 768 as "the margin left free" is wrong in the configuration we now serve. Also a **step function with a dead zone** whose step moves with boot VRAM | [results 05](results/05-runtime-flags.md) |
| 🟡 **TRACKED** | **What transferred from the RTX 3090 scan — the scoreboard** | 434 techniques scanned; **two measured wins** (`--spec-draft-n-max` +23.4 % RESOLVED, the `draft-dflash,ngram-mod` pair +48.5 %), one measured null, one running (`--spec-ngram-mod-n-match`), five read-and-closed without a GPU round, two already had, two impossible. **The largest untested idea left is recurrent-state prefix reuse** — their `PREFIX_CACHE=1` took turn 2 of a 24K chat from 23 s to 1.15 s, and whether llama.cpp's `--cache-reuse` restores DeltaNet state or only KV is unknown | [results 08](results/08-rtx3090-transfer.md) |
| 🟡 **TRACKED** | **Three of the six remaining scan flags are provably inert — do not sweep them** | Read from source, not measured: `-ctkd`/`-ctvd` (the drafter decodes 5 tokens per step, so quantised KV takes MMA_F16 with a full dequant, not the VEC kernel), `GGML_CUDA_GRAPH_OPT=1` (its body contains no `cudaGraph*` call and it cannot fire on one device with our shape), `-bs` (offloads only to position 2 of 10 samplers, and self-disables on a grammar). **`--spec-draft-p-min` ≤ 0.0625 is mathematically identical to 0.00** — `1/sum ∈ [1/16, 1]` by construction | issue #18, [results 05](results/05-runtime-flags.md) |
| 🔴 **UNTRACKED** | **`--spec-draft-n-max` defaults to 3 and the DFlash clamp allows 7** | Read from source: `common/common.h:325` is `int32_t n_max = 3`, and `speculative.cpp:989` clamps at `block_size - 1` = **7** for this drafter. Report 29's whole result was measured at **4**, chosen without knowing either number. Two agents independently called this the largest unclaimed lever on the list. Also never set: `--spec-draft-p-min` (default 0.0, `common.h:329`) and `-bs`/`--backend-sampling` for the main path (`arg.cpp:2296`) | [researchs/syv-rtx3090](researchs/syv-rtx3090/README.md) |
| 🔴 **UNTRACKED** | **48 flags exist that no profile here has ever set** | From an exhaustive scan of an external stack matched against a 175-capability map of our own llama.cpp. Six of the load-bearing claims are hand-verified; the other 428 verdicts are agent output and unmeasured. The scan's first result was a **false claim in our own `worker-iq2xxs-deep.ps1`** — it said `--fit-target 768` was the default when the default is 1024 | [researchs/syv-rtx3090](researchs/syv-rtx3090/README.md) |
| 🟡 **TRACKED** | **`--spec-draft-n-max` is a VRAM knob: 149.62 MiB per unit** | Measured. The Gated DeltaNet recurrent state is **flat at 149.62 MiB from 32K to 131K** — it does not scale with context — but it scales with the draft count: `common.h:390` returns `draft.n_max` from `need_n_rs_seq()`, so the buffer is `149.62 x (1 + n_max)`. At our `n-max 4` that is 748.12 MiB, confirmed in the log. Raising to the clamp of 7 costs **+449 MiB**; `ngram-mod` pays none of it. **A third of the drafter's measured 1,936 MiB is the target's state, not the drafter** | issue #18, [results 03](results/03-memory-and-kv.md) |
| 🔴 **UNTRACKED** | **`-ctkd` / `-ctvd` — corrected downward, still untried** | Recorded earlier the same day as a VRAM lever on a bad estimate. The drafter's KV buffer is **45.00 MiB**, so `q4_0` saves ~34 MiB, not hundreds. Still one flag, still untested, just small | [results 03](results/03-memory-and-kv.md) |
| 🔴 **UNTRACKED** | **Every `ngram-*` verdict was set on a repetitive prompt** | `ngram-mod` drafts by matching text already in the context, so a prompt with 66.2 % duplicate lines is its best case — and that is what every sweep used. On real source (4.7 %) it is worth ~17 % over no speculation, not the 2.7× the synthetic prompt showed. **Report 20's "+200 % at 131,072" is the largest claim owed a re-measurement** | [report 29](reports/29-DFLASH2-AND-THE-PROMPT-THAT-FLATTERED-NGRAM.md) |
| 🟡 **TRACKED** | **DFlash2 measured at 16,384 only, and never with a grammar** | **+34.7 % over `ngram-mod` on real code, RESOLVED**; the combination `draft-dflash,ngram-mod` is **+48.5 %**. Costs 1,936 MiB resident and `--fit` cannot measure it, so depth is an open question rather than an extrapolation — `draft-mtp` is +81 % at 16K and −71 % at 131,072 on the same artifact | issue #18, [report 29](reports/29-DFLASH2-AND-THE-PROMPT-THAT-FLATTERED-NGRAM.md) |
| 🔴 **UNTRACKED** | **No published number measures the stated metric** | Every result in `docs/reports/` is tok/s. Verified accepted coding tasks per hour has never been measured, and neither has the context a real task consumes — which is why three worker profiles shipped mis-sized in one day. [Plan 06](plans/06-REAL-TASK-BENCHMARK.md) is the runbook: 19 real issues, an FP8 ceiling on the same model, skills on/off, standard vs clink. **Not started; no issue yet** | [plan 06](plans/06-REAL-TASK-BENCHMARK.md) |
| 🟡 **TRACKED** | **Was trading `UD-IQ2_S` for `UD-IQ2_XXS` + a drafter a good trade?** | The row that used to sit here said IQ2_S had "never been loaded once". **That was false when written or shortly after** — `v3-iq2s` has 38+ measured rows across six result files, dozens of logs, and four `worker-iq2s-*.ps1` profiles, one of which is the recommended one. The real open question is the **trade**: IQ2_S was given up deliberately to free VRAM for DFlash2, and DFlash2 only became loadable on 2026-08-22. The trade is finally answerable — [CORRECTIONS §19](reports/CORRECTIONS.md) | issue #18, [plan 06 §3.5](plans/06-REAL-TASK-BENCHMARK.md) |
| 🔴 | **Does the 160-token probe understate speculation?** | every decoder verdict — `draft-mtp`, `draft-dflash`, eagle3, dspark — was decided on 160-token generations. `scripts/afk-q38-warmup.sh` exists and was interrupted | [CORRECTIONS §8](reports/CORRECTIONS.md) |
| 🔴 | **Re-derive the 13.6 % noise floor under `--fixed-text`** | paired rounds now repeat to within 0.05 points across boots, two orders of magnitude tighter than the floor. May be hiding small true effects | [report 23 §4](reports/23-SESSION-RECORD-2026-08-21.md) |
| 🔴 | **The desktop's 1,650–2,200 MiB** | the largest untouched lever on this machine, and it needs no code | [results 03](results/03-memory-and-kv.md) |
| 🔴 | **Why speculation dies at `65+0` on 163,840** | four independent routes to full residency all lose the drafter. `-ot` moves weights to CPU; `--fit-target` and `-ub` do not, so float divergence cannot be the whole story | [report 24 §1b](reports/24-BEYOND-128K.md) |
| 🔵 | **`model_arena.py` / `sweep_runtime.py` flat `timeout=1800`** | same fault class as the one fixed in `depth_sweep`; neither has been near its limit, so it is real but not urgent | [CORRECTIONS §8](reports/CORRECTIONS.md) |

## Quality — the worker

| status | item | why it is open | where |
|---|---|---|---|
| 🔴 | **Grammar alone, reasoning left on** | the only arm that showed the 26-point contract jump changed two things at once. `scripts/serve-v3-iq2xxs-gram.ps1` was built 03:29 and aborted 03:30 | [results 06](results/06-prompt-and-quality.md) |
| 🔴 | **How a long-thinking model survives a multi-turn agent loop** | the sharpened version of the retracted "it loops" claim. `damerau` takes 62.6 s direct and 247.6 s through OpenCode with no output | [post-mortem](reports/2026-08-21-inferred-looping-from-three-numbers.md) |
| 🔴 | **`reasoning_effort: low` on a 2-bit artifact, through the corpus** | swept on Q4 with a tool probe (6/6 succeeded); never where the failure lives | [results 05](results/05-runtime-flags.md) |
| 🔴 | **A system prompt that instructs *how to think*** | every sampler lever is exhausted; the process has never been instructed. Costs nothing | [plan 04 P3](plans/04-REVISED-PLAN-2026-08-21.md) |
| 🔴 | **Deep-context retrieval quality on anything but Q4** | nine artifacts have depth throughput numbers and none has a depth quality number. Open since 2026-08-17 | [results 06](results/06-prompt-and-quality.md) |
| 🔴 | **`bench/tap.py` was built and never run** | full per-request telemetry, written 2026-08-21, zero result files. Speculative until it earns a row | [results 07](results/07-telemetry-inventory.md) |
| 🔴 | **`--skill` in `run_retry_bench.py` was built and never run** | injects the real `karpathy-guidelines` + `tdd` text the way `clink-subagents` §7 mandates. No script calls it | [results 06](results/06-prompt-and-quality.md) |

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
