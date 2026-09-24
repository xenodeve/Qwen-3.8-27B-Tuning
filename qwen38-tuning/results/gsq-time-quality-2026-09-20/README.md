# Time per verified task and quality: resumed comparison

Issue: https://github.com/xenodeve/Qwen-3.8-27B-Tuning/issues/92.
The earlier language-only stopping decision is superseded (CORRECTIONS 52).

Completed: nine rotated validation runs /27 responses. Canonical interpretation:
`docs/results/20-gsq-time-quality-2026-09-20.md`. Raw validation summary is
`validation/summary.json`; full visible thinking is in `thinking-and-results.md`.
No production default changed. Full agent, repair-policy and broader quality
validation remain open; the single-response campaign does not certify them.

## Frozen comparison contract

- Same allocated context: 147456 for all artifacts, subject to actual-request
  validation. Report rendered token counts separately; no padding/truncation.
- Same original-session capture as the initial study, SHA256
  1721824613d680f3c0c052b0ca72ac64bbcca3f12bab96c7994887f3123d73f9.
- Each boot begins with the original summary request. Subsequent code questions
  replace only the final ask in a copy of that same frozen history. They do not
  feed a candidate's previous response into another candidate's input.
- First request is cold; subsequent requests may reuse the common frozen prefix.
  cache_prompt=True is requested, but actual cached tokens must be observed.
- Initial reasoning effort is medium. Draft depth is not reasoning effort.
- Configuration tuning tasks: original summary, merge_intervals, toposort.
- Held-out final code tasks: lfu_cache and tree_codec. Their existing assertions
  were read before selection; the invalid bracket_matching fixture is excluded.
- Compare at least three rounds with rotated order and paired seeds for a ranking.
  A single tuning result selects a candidate, not a proven universal best.
- Config search starts from measured NVFP4/EXL3 recipes. GSQ native-MTP depths
  2/3/4 and a layer-split challenger are candidates, not inherited winners.
- A conditional Han bias is an experimental mitigation; build from each GGUF's
  actual vocabulary, not EXL3 token IDs. Chinese-containing/requesting prompts
  lift the guard. Part-token Unicode behavior and effects on other languages
  remain limitations to assess; no universal language-correctness guarantee.

## Metrics

The historical session supports emphasizing warm operation: of 101 unique usable
requests, 93 have cached input, 86 have at least 90 percent cache, and the median
cached fraction is 98.15 percent. Two zero/unknown-input records are excluded.
See historical-cache-mix.json for the deduplication rule and source hash. These
are requests, not whole tasks; cold and warm results remain separate rather than
being hidden inside an invented weighted score.

Store whole thinking and final output, timestamped stream events, actual usage,
engine timing fields, guard preparation time, and time to complete response.
Client-observed thinking duration is first thinking text to first final-answer
text; it includes transport/buffering and is not exact GPU compute time.

For code, add executable verification time to request/preparation time. Failed
and truncated tasks remain failures with their spent time; do not average them
away. Report time per successful task together with pass rate and total elapsed
time. No arbitrary combined weighted score is invented. Ranking prioritizes
lower verified-task latency at comparable quality; tradeoffs remain explicit.

Quality: executable correctness, instruction adherence, factual grounding,
language integrity, and stability. Thinking review records visible duplicated
drafts/checks, unsupported assumptions and wasted restarts. Fluent thinking is
not proof of correctness or a faithful explanation of internal computation.

## Scope boundaries

The current code fixtures are single-answer implementation tasks with external
verification, not full repository-editing agents. Summary replay is also a
single-response task. Do not label these as a full agent benchmark or use them
alone to certify a daily driver. Full agent/tool-loop validation remains in the
parent plan before promotion. No default has changed.

## Instrument correction before ranking

The first warm-task pilot (tuning/20260920-022308-gsq-mtp-n2-c147456-r1)
replaced a trailing ambient system note after conversion, leaving the original
summary question alongside the new code question. Its warm rows are invalid,
not model failures. They remain with INVALID-WARM-TASKS.md. The corrected builder
replaces the last actual Anthropic user message before conversion and preserves
the ambient note; a regression test reproduces the original fault.

The tuning-v2 campaign uses the corrected builder and info-level logging (-lv 4)
to retain architecture and single-model offload evidence without per-token debug
logging. Do not pool it with the earlier debug-level pilot as a matched pair.
The conditional guard suppresses whole vocabulary pieces containing Han; it does
not guarantee that partial byte tokens cannot combine into Han. It also inherits
the incumbent's broad Chinese-mention exception. These are guard limitations,
not reasons to reject an artifact before measuring remedies.
