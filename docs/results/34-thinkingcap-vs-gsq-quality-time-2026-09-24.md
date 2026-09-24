# 34 — ThinkingCap-Qwen3.8-27B Q4_K_M vs GSQ IQ3_S-MTP, 2026-09-24

**Artifact:** `bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF`, `Q4_K_M` (17,442,400,352 B,
SHA-256 `fafa890c…f299` = Hub LFS oid). An efficient-thinking finetune of Qwen3.8-27B;
the GGUF carries the MTP head. License: PolyForm Small Business 1.0.0 + Apache-2.0 (Qwen).

**Protocol:** identical to [result 33](33-flash-next-vs-gsq-quality-time-2026-09-24.md) —
code1 and PAL, context 65,536, effort medium, client 2.1.281, ABBA in one sitting, GSQ
re-run as control, overlap audit afterwards on copies. ThinkingCap argv:
`llama_argv('thinkingcap_q4km', 65536, 'mtp-ngram', '9500,14500', 24)` (the Q4 rows'
split; MTP n3 + ngram 24/16/64), `66+0` resident. Runners take `--challenger` /
`PAL_CHALLENGER`. Aggregate: [`summary.json`](../../qwen38-tuning/results/thinkingcap-vs-gsq-2026-09-24/summary.json).

## code1

| cell | hidden | own tests | task wall |
|---|---|---|---:|
| gsq-a | 7/7 | pass | 156 s |
| thinkingcap-a | 7/7 | pass | 151 s |
| thinkingcap-b | 7/7 | **1 of 12 fail** | 105 s |
| gsq-b | 7/7 | pass | 160 s |

thinkingcap-b's implementation is correct; the test it wrote is not: after `add 2,
remove 1`, it removes twice more, expects the second to raise, then asserts quantity
**1** — the first of the two removals succeeded, so the store correctly holds 0. Its
thinking has no edge-case pass at all (0 hits for edge / exact / more than available),
unlike thinkingcap-a and both GSQ cells. Counted as not passed, per the frozen rule.

## PAL

| cell | hidden | overlap audit | red → green | task wall | accepted |
|---|---|---|---|---:|---|
| gsq-a | 8/8 | 0/4 | yes | 429 s | no |
| thinkingcap-a | 8/8 | **4/4** | yes | 466 s | **yes** |
| thinkingcap-b | 8/8 | 0/4 | yes | 288 s | no |
| gsq-b | 8/8 | 4/4 | **no** | 397 s | no |

ThinkingCap 1/2 accepted, GSQ 0/2. **Across today's two sittings GSQ is 1/4 accepted**
(results 33 + 34): the overlap requirement is a coin flip for both 27B artifacts, and n = 2
per sitting cannot rank them.

## Speed and thinking

From each cell's `server.log` (median prefill of requests ≥ 2K tokens, token-weighted
decode):

| | prefill | decode | output tokens code1 |
|---|---:|---:|---:|
| GSQ | 660–700 tok/s | 45–57 tok/s | 5,477 / 4,356 |
| ThinkingCap | 678–774 tok/s | 46–52 tok/s | 3,543 / 3,036 |

Throughput is the same within noise; ThinkingCap emits ~30 % fewer tokens on code1.
Thinking chars: code1 4,225 / 4,063 vs GSQ 11,927 / 7,534 (**2–3× less**); PAL 38,815 vs
29,095 (more, but 34 turns / 16 tools vs 41 / 19). No empty blocks, no repetition, no Han;
Thai finals 779–806 chars (spelling not checked).

## Reading

On this evidence ThinkingCap is **at least as good as GSQ at the same speed**, with shorter
thinking on easy work, and one visible cost of that brevity (an unchecked self-written
test). Not measured: depth > 65,536, more attempts, Thai spelling, the second PAL task.
