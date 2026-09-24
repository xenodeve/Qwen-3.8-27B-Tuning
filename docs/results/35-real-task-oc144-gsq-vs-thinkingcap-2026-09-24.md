# 35 — A real hard issue at 262K: GSQ vs ThinkingCap IQ4_XS, 2026-09-24

**Task:** [xenodeve/openclink #144](https://github.com/xenodeve/openclink/issues/144) — own
the whole process tree, so cancellation means nothing is still writing. Multi-layer: spawn
into a process group / Job Object, terminate the group, confirm the tree is gone before
acknowledging, cover every exit path, bring the PTY runner along, and — not stated in the
issue — do not hang when a grandchild holds the stdout pipe.

**Setup, as the developer works:** each model's served profile at **262,144**
(`serve-gsq.ps1`, ub 1024; `serve-thinkingcap.ps1`, IQ4_XS, ub 512), a fresh clone at
`200fcb92` with no remote, the installed Claude Code 2.1.281 `-p --permission-mode auto`
with the daily hooks, prompt = a Thai instruction + the issue text
(`bench/fixtures/oc144_issue.md`), no gh / push / network, **90-minute cap**, ABBA.
Runner `tools/run-real-oc144.py`; aggregate
[`summary.json`](../../qwen38-tuning/results/real-oc144-2026-09-24/summary.json).

**Oracle:** `bench/fixtures/oc144_process_tree_hidden_test.py`, 7 black-box tests with a
real fake CLI whose grandchild writes a heartbeat file. Validated before any model ran:
**original 0/7, a reference fix 7/7**. Runs in the Linux sandbox with `--init` (opt-in,
`fixture_test_tool.run_sandboxed_pytest(init=True)`): without it pytest is PID 1, killed
grandchildren stay zombies, and a correct fix reads as "group survived".

## Results

| cell | ended | hidden | requests | max ctx | prefill / decode tok/s | out / prefill tok | thinking chars | tools |
|---|---|---|---:|---:|---|---|---:|---:|
| gsq-a | **cap (90 min)** | **7/7** | 311 | 171,245 | 599 / 32.5 | 126K / 397K | 260K | 79 |
| thinkingcap-a | done, 74.7 min | 5/7 | 268 | 164,943 | 637 / 35.0 | 116K / 293K | 251K | 103 |
| thinkingcap-b | **cap (90 min)** | **7/7** | 259 | 209,178 | 634 / 33.2 | 140K / 311K | 284K | 117 |
| gsq-b | done, 79.7 min | 5/7 | 238 | 186,158 | 603 / 34.2 | 125K / 317K | 296K | 102 |

Prefill = median of requests ≥ 2K tokens; decode token-weighted; draft acceptance median
0.48–0.55; one forced full re-prefill per cell; no Han in any thinking.

- **The two models are indistinguishable here.** Each produced one complete-but-late tree
  and one on-time tree that fails the **same two tests**
  (`test_success_leaves_no_descendant`, `test_failure_leaves_no_descendant`): cleanup on
  the normal exit paths, which the issue lists ("Cleanup runs on every exit path").
  No cell was both correct and on time.
- **Speed at 165–209K is equal within noise** (ThinkingCap ~5 % faster, below the floor,
  despite ub 512 vs 1024).
- **ThinkingCap's shorter thinking (result 34, code1) does not carry to this task:**
  251–284K chars vs GSQ 260–296K.
- Thai finals only in the two cells that finished (719 and 1,130 Thai chars; not
  spelling-checked).

## Instrument notes

- The repo's own clink tests in the sandbox are **unusable**: the image lacks `openai` and
  `mcp`, and the original tree also reports 50 errors. Regressions are not measured.
- Capped cells were killed mid-session; their trees are graded as left.
- ThinkingCap Q4_K_M could not boot at 262K with MTP (four splits tried; draft KV or model
  OOM with the display card's 2,500 MiB reserve kept); IQ4_XS boots at ub 512, shared
  774 MiB. Bench logs `logs/tc262-*.out.err`.
- n = 2 per model. A cap of 90 minutes decided two of four outcomes.
