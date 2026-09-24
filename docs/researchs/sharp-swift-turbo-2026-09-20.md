# Sharp, Swift and TURBO — external claims and pinned inputs

Status: external material only until a claim is reproduced under
`qwen38-tuning/results/three-candidate-2026-09-20/`.

## Sharp

Source: `peculiar-ragdoll/Qwen-Sharp-Chat-Templates`, revision
`85461fc118aaf25e7319c7ecf2481f944aac3a32`.
Model card license: Apache-2.0.

CREATOR CLAIM: Sharp reduces filler/repeated reasoning while retaining essential
content. The current root template identifies itself as v22.5.0 and says it is a
portable template intervention, not changed model weights. The creator reports
SWE-bench-Live and Claw-Eval improvements, but those values are not local rows.

LOCAL PREFLIGHT: upstream `scripts/verify_template.py` passed its complete live
upstream-diff, rendering, effort, tool, terseness and minification checks under
Python UTF-8. Template SHA256:
`cdff39fb26b60dc90faa292e726655c6b21f62db497846e02e4c4bbab942a84a`.
This verifies template integrity, not coding quality or speed.

## Swift

Source: `ukisai/Swift-Qwen3.8-27B-GGUF`, revision
`eb0e3a7dc70643c2ab920f5133750d02c20849ff`.
Model card license: Swift Open License v1.0; its published terms distinguish
organizations above USD1M annual recurring revenue. This evaluation does not
reinterpret that license.

CREATOR CLAIM: Swift post-training reduces pathological overthinking. Its model
card recommends Q6_K or higher for long agentic runs and strict tool-call
formatting, sampling 1.0/0.95/top-k20/min-p0/repetition1, and at least65536
context. The card explicitly says its GGUF conversion has not run the full
benchmark suite.

PINNED INPUT: `Swift-Qwen3.8-27B-Q6_K.gguf`, 22884407456 bytes, Hub LFS SHA256
`7f4de8abd5446c08b0f975a1b38e43d02639f4ae9ecdd2ddfd5c5b0f612bda59`.
The same revision's `SHA256SUMS.quants` says `a429f636...` for that filename and
therefore does not describe the object currently selected by the revision.

## TURBO

Source:
`DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF`,
revision `c02caef111a8acf987947f35e1e288aa5450e184`.
Model card license: Apache-2.0.

CREATOR CLAIM: the tuning reduces thinking by roughly one-half to one-tenth while
retaining detail. The card recommends Q6 for tool calling and temperature0.6 for
precise coding. It also explicitly warns that reasoning was materially modified
and must be tested for the deployment.

PINNED INPUT:
`Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q6_K.gguf`,
24033703456 bytes, Hub LFS SHA256
`ac011aabe685edbdf542e49351eb6c76c0e5531408f2507f2235ab10931e23a5`.

## Local falsification targets

- Fewer tokens are useful only when verified completion and requirement retention
  do not regress.
- Sharp must be separated from effort and from the existing current template.
- Swift's known math signal is characterized separately from coding promotion.
- TURBO must be checked for premature completion, continuation dependence,
  duplicate work, tool misuse, goal abandonment and reasoning loops over a
  long-horizon agent workload. Short code answers cannot clear this gate.
