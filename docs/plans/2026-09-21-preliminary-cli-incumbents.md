# Incumbent extension to the preliminary Claude CLI screen — 2026-09-21

Completed: [result23](../results/23-preliminary-cli-quality-time-2026-09-21.md).
Both pass the functional oracle; EXL3 is faster in this sample but its Thai final
is severely malformed. This does not select a stable winner.

Issue #92. User requested the same code1 task for the two existing incumbents and
clarified the llama.cpp model is exactly
`Qwen3.8-27B-NVFP4-MTP-VERY-LOW.gguf`—the current source-default `-Nvfp4`
profile—not a literal LOW artifact.

This extends `preliminary-cli-code1-v1`; prompt, fixture, client2.1.258,
context65536, medium effort, seed request29, output8192, max32 turns,1200s,
zero retries, Read/Edit-only policy, recorder and independent verifier remain
identical. It remains one sample on one shallow task and cannot establish a winner.

Profiles:

- **NVFP4 llama.cpp default:** resolve live source via
  `gsq_compare.production_nvfp4_argv(65536)` / `worker-q4-dual.ps1 -Nvfp4
  -Ctx65536 -WhatIf`, including dynamic tensor split, `--fit off`, Q4_0 KV,
  checkpoints8, cache RAM24576, MTP3+ngram24/16/64, medium effort, repeat
  penalty1.05 and `qwen38-late-system.jinja`. This is the source-default worker,
  not `serve-dual-nvfp4-deep.bat`'s extra `-Vision -Upstream`. Record exact
  resolved argv, model/template/executable hashes and call it VERY-LOW.
- **EXL3 incumbent:** exact 4.0bpw H5 directory, MTP, fp8 KV4, tensor-parallel
  `9,15.5`, draft3, context65536, cwd `exllamav3-mia`, native server language
  guards. Recomputed shard hashes must equal
  `2633b0cb...b214f` and `23879b3c...c12`, revision
  `b4e3574d5665efb5d8031c05a578837e6700a912`. Direct owned launch; never use
  restart/stop wrappers. Set restart flag to private evidence. Native health
  corroborates model-directory name/context; absolute identity comes from owned
  argv plus hashes and is explicitly labeled—not fabricated native `/props`.

Engine-native guards differ and are part of the tested operating points: llama
conditional Han logit bias versus EXL3's built-in CJK/Thai/loop controls. Preserve
raw requests/outputs. Throughput formulas stay engine-native and are reported
with sources; no cross-engine conversion is invented.

The same user game-closure/GPU authorization and stale-lock archive permission
from the immediately preceding screen apply to this requested extension. Still
recheck processes, resources, ports, leases and final cleanup. No default changes,
commit, push or raw-evidence upload.
