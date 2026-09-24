# Visible-thinking review: observations, not an intelligence score

## What the tuning evidence establishes

1. **A faster decoder can complete the task later.** On merge_intervals, GSQ
   tensor/MTP n4 reports 43.17 tok/s against n3's 38.01, but accepted task time
   is 14.37 s against 10.88 s. Total generated tokens are 308 versus 145 and
   observed thinking spans are 4.89 versus 0.99 s. These are single-seed tuning
   observations, not a statistical claim that n4 always reasons longer.
   Sources: tuning-v2/20260920-024556-gsq-mtp-n4-c147456-r1 and
   tuning-v2/20260920-024001-gsq-mtp-n3-c147456-r1.

2. **Useful checking and redundant reconsideration must be separated.** The
   GSQ n2 topological-sort trace correctly identifies missing-key target nodes,
   indegrees, heap tie-breaking and cycle detection. It also revisits duplicate
   edges and set versus dict.fromkeys repeatedly. It passes the fixture but
   spends 44.17 s in the observed thinking span. The trace contains substantive
   reasoning; its length alone is not a low-quality verdict.

3. **Correct visible reasoning can become an incorrect final artifact.** In
   the layer/MTP n3 merge task, the thinking block sorts the whole input before
   merging. The final code initializes from the unsorted first interval and
   sorts only the remainder, failing the fixture. Both channels must be checked.
   Source: tuning-v2/20260920-025208-gsq-mtp-n3-c147456-r1.

4. **That defect is demonstrably recoverable under a verifier, but is not erased
   from the raw score.** The fenced code from that thinking block was manually
   inspected and passed the same fixture. repair-demo/ retains the alternative
   and verification. Its 20.42 s figure sums prior response/check time and a new
   check; it excludes manual inspection and is NOT a measured fully automated
   recovery latency or an independent held-out success. Raw layer score stays 1/2.

5. **Lower temperature did not automatically improve this incumbent.** NVFP4
   temperature 1.0 produced shorter thinking and lower task times than 0.6 in
   the pilot, while both code tasks passed; the 0.6 summary also had more visible
   Thai corruption. This justifies retaining the incumbent sampler for validation,
   not a general rule that higher temperature is better. Sources: nvfp4-tuning/
   20260920-033009-nvfp4-served-n3-c147456-r1 and
   20260920-033503-nvfp4-served-n3-c147456-r1.

6. **Passing fixtures is not the entirety of code quality.** NVFP4's merge answer
   sorts the same input twice even though its thinking draft stores the sorted
   list once. It is functionally correct on the fixture, with avoidable work in
   the final implementation. This is a separate engineering-quality observation,
   not a fabricated functional failure.

7. **The summary inherits claims from a flawed conversation.** Several artifacts
   repeat old claims about byte corruption or PyThaiNLP without distinguishing
   observation from hypothesis. The task asks for a history summary, so faithfully
   repeating a historical claim must not be confused with independent proof of
   that claim. A separate evidence-grounding task is needed to score that ability.

## Language mitigation

Conditional Han token bias is applied to llama.cpp requests using token IDs from
the loaded GGUF vocabulary; EXL3 uses its existing guard. This makes language
guarding part of the operating point rather than a reason to stop evaluation.
Whole thinking and final text remain available. Remaining Thai spelling defects
are recorded for later repair evaluation. No zero-regex-hit claim is used as
proof of universally correct Thai.

## Evaluation scope

Validation uses held-out LFU/tree-codec tasks and rotated rounds. Final-code
execution and contract checks are the functional evidence. Trace review adds
requirements understanding, relevant checks, repeated drafts and reasoning/final
consistency; it does not reveal unobserved internal computation. Full autonomous
repository-editing quality remains a later gate before any default promotion.

## Held-out validation: trace findings

- **NVFP4 r2 LFU, budget-limited:** the complete retained trace repeatedly
  revisits the minimum-frequency invariant after constructing a valid solution.
  The final channel is empty at 4096 output tokens. Offline extraction found
  two complete drafts that pass the frozen assertions, first available in the
  stream at 113.298 s and later at 152.154 s. The first, incomplete draft fails.
  See validation/20260920-041027-nvfp4-served-n3-c147456-r2/ and
  nvfp4-thinking-recovery-demo/. These are recovery opportunities, NOT a measured
  automatic early-exit system. The raw attempt remains budget-limited.
- **GSQ r2 tree:** repeatedly recounts a four-node BFS example incorrectly
  (six null children instead of five), then argues that an extra unused marker
  is acceptable. Its actual final algorithm passes the frozen fixture and the
  supplemental depth-1500 integer-tree probe. The observed thinking span is
  107.607 s. Incorrect hand simulation is a trace defect, not evidence that this
  final implementation fails. See validation/20260920-042713-gsq-mtp-ngram-n3-c147456-r2/.
- **GSQ r3 tree, budget-limited:** drafts a complete BFS implementation, then
  repeatedly questions whether trailing-null trimming is safe, reconstructing
  several examples and reconsidering the integer-value assumption. At 4096
  tokens the final channel is still empty. Record the spent 132.33 s, not an
  imaginary successful completion. See validation/20260920-044259-gsq-mtp-ngram-n3-c147456-r3/.
- **EXL3 r1/r3 LFU:** starts from the standard frequency-bucket design and
  checks the minimum-frequency invariant. r3 initially considers a wrong update
  rule before correcting it. Visible draft formatting is damaged in places;
  the delivered code is independently executable and passes. This is another
  reason not to execute an arbitrary thinking fragment as a repair without a
  verifier. All three delivered LFU implementations remove empty buckets.
- **EXL3 tree across three rounds:** selects iterative level-order serialization,
  checks negative values/null children, and delivers an implementation that
  passes both the fixture and the supplemental depth-1500 probe. Near-identical
  outputs across some rounds are not proof of independent seeded draws: this
  EXL3 path does not establish that the requested seed is applied.
- **NVFP4 r3 LFU:** brief thinking describes an OrderedDict-based design, but
  the final implementation creates `defaultdict(int)` and tries indexed item
  assignment into its integer value. The frozen test fails on the first put.
  This short response is not a successful speed win. The r3 tree answer passes
  with iterative BFS, although its thinking also miscounts null markers in a
  worked example; final execution remains the deciding correctness evidence.

These traces motivate bounded verifier-assisted completion and thought-budget
experiments, not blanket deletion of reasoning or an assumption that short
thinking always has higher quality. No such production change has been made.
