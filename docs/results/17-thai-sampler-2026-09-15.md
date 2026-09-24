# 17 - the Thai sampler: a tightened triple that changes nothing

Question (#81): on profile G (turboderp SC 4.0bpw H5, ctx 262144, default
`temperature 0.6 / top-k 20 / top-p 0.95`), does serving Thai-heavy prompts
with a tightened sampler (`0.3 / 10 / 0.9`) reduce Thai script errors?

The incident behind it: a Claude Code session through G wrote the YT
Downloader `README.md` / `templates/index.html` with dropped U+0E4C
(`ดาวน์โหลด` -> `ดาวนโหลด`) and Latin substitutions inside Thai runs
(`ชื่อ` -> `ชื่o`). Same drift family as #77 (Han in Thai sentences) and #76
(a U+0E48 loop).

## Design

`qwen38-tuning/serving/exl3/thai_sampler.py` (pure function, same shape as
`cjk_guard`): Thai-heavy (>= 20 Thai-block chars) prompts get `0.3 / 0.9 /
10`; anything else passes through; a client value already tighter is never
loosened. Wired into `generate_full` next to the #77 ban, all lines marked
`# xeno`. Test-first: `bench/tests/test_exl3_thai_sampler.py`.

The pair ran in one boot on :8000 with a per-request body key distinguishing
the arms. Rows: `bench/results/thai-sampler-<stamp>.jsonl`.

## Numbers

Second run (`thai-sampler-20260915-032718.jsonl`, 48 requests: 8 prompts x 2
arms x 3 rounds, arm order alternated per prompt), recomputed after the
counter fix below:

| arm | broken script tokens | Thai chars | per 1k |
|---|---:|---:|---:|
| A default | 0 | 5,425 | 0.0 |
| B tightened | 0 | 5,496 | 0.0 |

Per the pre-registered falsifier (B equal to or worse than A), the answer is
**no**. The tightened triple does not ship: `thai_sampler` is now **opt-in
only** (body `"thai_sampler": 1`), default passthrough. First run
(`...-032226.jsonl`, 32 requests) is discarded as evidence -- its rows carry
no text, so the counts cannot be audited.

## Two instrument faults found and fixed

1. **The false-positive counter.** `เว็บไซต` is a substring of correct
   `เว็บไซต์` (the broken form is the correct word minus the final U+0E4C),
   so a plain substring counter flagged every correct `เว็บไซต์`. The first
   two summaries (0.54/0.55 and 0.55/0.73 per 1k) were counter artifacts.
   The counter now uses `เว็บไซต(?!์)`; `ดาวนโหลด` and `วดีโอ` are safe as
   substrings and were verified so.
2. **Unauditable rows.** The first run stored counts but not text. Every row
   now stores `prompt_text` and `text`, which is what made fault 1 checkable.

## What this does not say

The base rate on short single-shot Thai prose is ~0 in both arms: the
incident-level breakage (long agentic session, tool loops, thinking-heavy
context) did not reproduce here, so this pair cannot separate the arms on
that regime. A long-session probe is the follow-up, not a re-run of this one.

## Addendum 2026-09-15: the failure is bursty, not uniform

The full Claude Code transcript of the YT Downloader session (developer
provided) shows the garbling is message-bimodal, not per-token noise: the
opening greeting is clean Thai throughout, while post-tool-call summaries are
densely garbled in bursts ("เครื่่องมื้อ", "สำudio", "เสิร์ฟท่ี", "เริ่่มสร้าง").
The model even noticed its own fault once ("ผมพิมพ์ฟุตเตอร์มั่ว
(ปนตัวอักษรอังกฤษ)") and its repair attempts garbled the same way. Code and
identifiers stayed byte-correct in every message; only Thai prose broke.

Two mechanisms fit, and both point away from this pair's regime: a
context-primed seed (English tool output raises Latin-token priors) plus an
autoregressive cascade (one wrong token conditions the rest of the response).
The pair prompts here had neither English context nor length, which explains
the 0/0. The follow-up is a seeding experiment (same Thai prompt with vs
without an English tool-output prefix) and, if confirmed, a stream watcher
that cancels and retries once with the opt-in tightened sampler -- the
consumer the `thai_sampler` opt-in was kept for.

## Addendum 2, same day: temperature is the variable, and prevention fails

144 responses at temperature 0.6 across regimes 1-3 (short prose,
pasted-English seed, real tool-role + multiturn + system prompt + medium
effort) scored ZERO confirmed breaks over ~45k Thai chars: `thai-seed-*`,
`thai-toolseed-*`, `thai-multiturn-20260915-040001.jsonl` (audited with the
fixed patterns; the โฟลเดอร/แลว hits there were the same substring-class
false positives, corrected to `โฟลเดอร(?!์)` / `แลว(?![่-๋แ])`).

The transcript ran under Claude Code, whose default temperature is 1.0 --
every probe above sent 0.6. At `--temperature 1.0` with the multiturn shape,
64 responses / ~54k Thai chars (`thai-multiturn-t1-20260915-041009.jsonl`,
`thai-multiturn-t1-20260915-041813.jsonl`):

| arm | genuine breaks |
|---|---|
| A default | เว็บไซตส์ x2 (misplaced ์ -- the transcript's family), Vietnamese leak `lớn` x1 |
| B tightened 0.3/0.9/10 | Vietnamese leak `thói quen` x1 (+ ambiguous `วิดีโอยou`) |

Three readings, all negative-leaning. First, reproduction needs temp 1.0:
the failure was never about context shape alone. Second, the prevention is
**not demonstrated** -- B leaks too, at temperature 0.3/top_k 10, so the
wrong-language token sits inside the narrowed top-k: the drift is in core
probabilities, not the sampling tail, and no sampler squeeze tested here
reaches it. Third, the leaks are Vietnamese now that Han is banned (#77) --
spillover hypothesis at n=2, not a claim. (Spacing slips like `และPad Thai`
/ `ทบทวนSubscription` occur in both arms and are counted separately -- they
are a different, milder class.)

## Intervention shipped same day: cap Thai-heavy prompts at 0.6

Developer directive after the prevention failure: put Thai serving back at
0.6. `thai_sampler.params_for` now caps a Thai-heavy (>= 20 Thai chars)
prompt at temperature 0.6 no matter what the client sent -- top_p/top_k
untouched, lower client values never raised. Body `thai_sampler: 1` keeps
the full tighten triple, `0` is pure passthrough, `EXL3_THAI_SAMPLER=0`
kills everything. Unit gate 40/40 green, live on G after a restart, smoke
(2x Thai requests at client temp 1.0) clean. Open: effectiveness of the cap
at scale is unmeasured -- the temp-1.0 break rate (~3/27k chars) needs a
long run to confirm any reduction. Tracked on #82, left open.

Unit gate: `test_exl3_thai_sampler.py` + the exl3 serving tests green (37
passed). Full bench suite: 1691 passed, 3 skipped, 1 pre-existing failure
(`test_corpus_depth.py::test_the_incumbent_corpus_is_untouched_by_the_addition`
-- the pin matches the LF blob; this Windows checkout is CRLF and
`corpus_hash` reads raw disk bytes. Unrelated to this change; flagged, not
fixed here).
