# Thai repair reported FP 0 while corrupting correct Thai

## Summary

The offline Thai repair experiment reported `40 -> 0, FP 0` while its substring
dictionary changed correct Thai such as `ไฟล์` to `ไฟล์์`. Issue #89 and commit
`7d5c046` replace the self-referential metric with an expected-text oracle, keep
only a precision-first 29-rule subset, protect structured spans, withdraw the
unsafe float-token ban, and retract the shipping verdict. The repair remains
offline.

## Symptom

`thai-eval-offline.py` reported `rows_worse: 0` on both captured problem
sessions. A direct call showed the contradiction:

```text
dict_fix("ไฟล์ เว็บไซต์ โปรเจกต์ หรือ ลิงก์ โฟลเดอร์")
-> "ไฟล์์ เว็บไซต์์ โปรเจกต์์ หรืออ ลิงก์์ โฟลเดอร์์"
```

The same implementation corrupted 23 correct target occurrences in the two
source sessions and changed 172 entries in PyThaiNLP's 62,101-word
`thai_words()` corpus.

## Root cause

Two independent defects combined in
`qwen38-tuning/tools/thai-eval-offline.py`:

1. `DICT_RE` matched every malformed spelling as an unbounded substring.
   Several broken forms are prefixes of their own corrections (`ไฟล` inside
   `ไฟล์`, `หรื` inside `หรือ`), while other short forms occur inside unrelated
   valid words.
2. `broken_count()` counted only eleven incident regexes. `rows_worse` compared
   that counter before and after repair instead of comparing repaired output
   with expected text. New corruption such as `ไฟล์์` was outside the detector,
   so the counter remained zero and certified its own bug.

The proposed change to `cjk_guard.bias_for()` had the same proof gap in a
different layer: it inferred that a mark-only token was never valid from 13
whole-word examples. On the real tokenizer, 26,875 of 62,101 known Thai words
use at least one of the 21 proposed banned IDs as a continuation token.

## Why it produced the symptom

The substring regex altered both malformed and already-correct text. The
evaluator then asked only whether one of its original eleven malformed patterns
increased. The newly duplicated marks were invisible to that question, so the
more text the repair corrupted, the more confidently the summary could still
say `FP 0`.

## Fix

Commit `7d5c046` makes four changes:

- `DICT` keeps only 29 malformed forms that occur in no entry of the pinned
  clean-word corpus. The 20 ambiguous forms return raw text.
- `dict_fix()` leaves Markdown code, URLs, Windows and Unix paths, valid JSON,
  and tool-shaped payloads byte-identical.
- `evaluate_expected()` requires explicit `text` and `expected` strings. Empty
  or non-gold input raises, and the CLI emits raw/expected/repaired rows for
  every arm.
- `test_thai_mark_continuation_tokens_remain_available` pins the serving seam:
  the Han guard may ban Han IDs, but not Thai mark continuation IDs.

The old verdict is retracted in `docs/results/18-thai-repair-2026-09-16.md`
and `docs/reports/CORRECTIONS.md` section 52. The raw occurrence replay is
`qwen38-tuning/bench/results/thai-repair-safe-20260916.jsonl`; every occurrence
is marked `gold: false` so it cannot be mistaken for an independent precision
measurement.

## How it was found

The deterministic repro applied `dict_fix()` to known-correct words. Source
tracing then followed `DICT_RE.sub()` into `broken_count()` and showed that the
repair and its judge used different error vocabularies. The disproof for the
initial "only six prefix collisions" hypothesis was the full `thai_words()`
scan: even after guarding self-prefixes, 155 known words still changed. Removing
all rules whose malformed form occurs in the clean corpus reduced that count to
zero.

For the sampler hypothesis, decoding the real tokenizer vocabulary found 21
mark-only IDs. Encoding all 62,101 clean words showed 26,875 depend on at least
one of them, disproving the claim that they are globally expendable.

## Why it slipped through

The experiment had no independent oracle. Its false-positive definition was an
increase in the same small pattern list used to describe the original incident,
not a difference from correct text. The float-ban test used a mock tokenizer
with no floating-mark pieces, so it stayed green while proving nothing about
the behavior being added.

## Validation

- Original repro: `test_correct_thai_forms_are_byte_identical` now passes.
- Clean corpus: 0 changed of 62,101 `thai_words()` entries.
- Protected spans and fail-loud CLI: `test_thai_repair.py` passes all 9 tests.
- Targeted serving/Thai gate: **35 passed, 1 skipped**.
- Captured-session replay: 68 + 11 repairable occurrences, zero remaining for
  the selected rules; the historical counter moves **36 -> 22** and **4 -> 2**.
- `compileall` passed and the new stale-claim rule compiled and reported four
  intentional retraction references in three files.
- Full bench: **1640 passed, 18 failed, 3 skipped**. The 18 failures are the
  existing missing CUDA-runtime checks for `-TheirBuild` plus the pre-existing
  `real-code` corpus hash mismatch; none imports the Thai repair files.
- Documentation link check still reports two pre-existing missing targets:
  `docs/reports/2026-09-16-spark13-problem-inventory.md` and
  `qwen38-tuning/logs/bat-maxctx2.log`.

No live server was started. The repaired dictionary is not connected to
streaming output, and no claim here says users receive repaired text yet.

## Action items

- Design and test chunk-boundary-safe repair for streamed prose while keeping
  reasoning, code, JSON, and tool calls untouched. Track separately before
  enabling repair in serving.
- Obtain independently human-corrected gold text for the two source sessions
  before reporting precision or recall for full responses.
