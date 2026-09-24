# Three-candidate extension: preliminary screen, 2026-09-20

Status: incomplete campaign under issue #92. No winning foundation selected and no production default changed.

## Selection rule

The binding rule is in `AGENTS.md`, engineering north star: weigh prefill, decode and quality, prioritizing time to verified task completion. Compare each artifact's own best locally validated configuration at a common viable allocated context. Read complete visible thinking alongside executable/final-answer evidence. Repairable language faults are optimization candidates, not automatic rejection grounds; measure repair costs and false changes. A fast decoder that thinks longer can lose overall.

This screen does **not** establish best configurations. It uses short prompts, one seed and one round per Sharp cell. Full thinking is retained but its complete qualitative review is outstanding. Neither raw decode nor short-prompt prefill rates support a promotion verdict.

## Evidence

Raw root: `qwen38-tuning/results/three-candidate-2026-09-20/`.

Sharp matrix root: `sharp-matrix-c65536/` within that root. All four cells use NVFP4 VERY-LOW, served speculation recipe, requested temperature 1.0, seed 17, max output 8192, conditional Han guard, allocated context 65536. Each cell is a separate server boot; these are exploratory observations, not same-boot paired speed effects. Actual rendered prompt lengths differ by template and effort.

| Run directory suffix (full timestamp identifies evidence) | Code assertions | Thai repair exact match | Math exact matches | Notable unfinished cost |
| --- | --- | --- | --- | --- |
| `20260920-083957-nvfp4-stock-medium-served-n3-c65536-r1` | 2/2 | yes | 2/2 | none in this cell |
| `20260920-084259-nvfp4-stock-xhigh-served-n3-c65536-r1` | 2/2 | yes | 2/2 | Thai prose completed but misspelled required words |
| `20260920-085009-nvfp4-sharp-medium-served-n3-c65536-r1` | 2/2 | no | 2/2 | Thai repair final: `บอกรื่องคุณภาพ` |
| `20260920-090733-nvfp4-sharp-xhigh-served-n3-c65536-r1` | 2/2 | no | 2/2 | Thai repair: 8192 tokens, 203.671 s request wall, `finish_reason=length`, empty final answer |

Each value above comes from that run's `responses.jsonl`; executable verification files are retained under its `verification/` directory. These assertions cover only the two frozen coding fixtures, not general coding quality.

Stock/xhigh Thai prose used 8012 generated tokens and 187.726 s request wall. Its final answer includes `เว็บไซ้`, `ลิงก`, and `ข้อพ้ดผิ`. Sharp/xhigh topological sorting used 7072 generated tokens and 174.104 s request wall while passing the fixture. These observations illustrate why output rate alone cannot rank the candidates. They do not establish whether the long thinking was useful without reviewing its contents.

Request wall is **not yet total time per verified task**: verification and preparation must also be accounted for, and failed attempts retained in the denominator. Do not label these request times as the final primary metric.

## Context and candidate limits

Earlier Q6 screens retain Swift load failures at 147456 and 131072 and TURBO's load failure at 98304. Swift at 98304 and both Q6 candidates at 65536 produced short responses. This establishes neither half-window large-request survival nor maximum certified context. Context 65536 is a provisional common screen allocation only.

Existing Swift screens request temperature 1.0; TURBO requests 0.6. They are exploratory configurations, not an isolated weight comparison and not proven best configs. Original-session replay, meaningful-depth prefill, configuration tuning, rotated validation, mixed-language evaluation, and TURBO long-horizon/tool-recovery counters remain outstanding.

## Verification status and procedural incident

The user ran `python -m pytest qwen38-tuning/bench/tests/test_gsq_compare.py -q`: 28 passed in 0.25 s.

A subsequent full suite ran concurrently with the Sharp server and returned 1 failed, 1755 passed, 2 skipped, 9 warnings. `test_chat_template_travels.py::test_it_differs_from_the_served_artifacts_own_template_by_one_line` read live `/props` and compared Sharp's 488-line template against the 184-line late-system template. Its documented prerequisite is a Stock-template server. This run does not satisfy that prerequisite. Preserve the failure; do not modify the test or template just to force a pass. The idle-server rerun passed: 1756 passed, 2 skipped, 9 warnings in 253.54 s. This confirms the suite passes without the experimental server; the live Stock-template check remains skipped, not independently validated. Future test gates must not overlap candidate inference.

Cleanup review found duplicate Swift digest metadata, shared-verifier refactoring potential, and two template-metadata ownership sites. The duplicate digest now references `EXPECTED_DIGESTS['swift']`; focused tests then passed 28/28 in 0.18 s. Shared-verifier extraction is deferred because it changes an older runner outside this slice. Current-template provenance is unchanged to avoid broadening this experiment's diff. Avoid modifying the runner while active inference or test evidence is being collected.
