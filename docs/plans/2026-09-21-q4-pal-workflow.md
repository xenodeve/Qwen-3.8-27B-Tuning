# Swift/TURBO Q4 and isolated PAL workflow — 2026-09-21

Completed bounded evaluation: [result24](../results/24-q4-real-project-workflow-2026-09-21.md).
Both Q4 and conditional Q6 controls completed the workflow, but none satisfies the
full audited PAL contract. No additional quant downloads or default promotion.

Issue #92; approved execution plan is also retained in the session plan file.
The user selected Q4 first, raising quant only if it is poor, and explicitly
required that seven active original repositories remain unaffected.

## Frozen scope

- Swift Q4_K_M at revision `eb0e3a7dc70643c2ab920f5133750d02c20849ff`:
  18,024,380,576 bytes; SHA256
  `ad5811e291431bd0de1cec0c4004a5eac98daee9850882edac69a823209e88ab`.
- TURBO MTP-Q4_K_M at revision `c02caef111a8acf987947f35e1e288aa5450e184`:
  18,498,573,856 bytes; SHA256
  `bc7a6cf2bcc78d1190aaf04d1ab1c5cb845b6ff23aa0e7d24fe0d2ea6d3a7c7c`.
- Downloads are complete and locally hash-verified. Short read-only-use hardlinks
  under `C:/AI/models/` avoid the loader's long-cache-path failure without a
  second copy of the weights. The Swift checksum text disagrees with downloaded
  bytes; actual Hub object/local hash is recorded, not the contradictory text.
- Code1 first; then a real PAL registry task on detached commit
  `200fcb9262d25e4002e30bf00c4364f55a42354e`, cloned with `--no-hardlinks`, no remote,
  no untracked files or source credentials. Original PAL files are never edited.
- Task: skip unsupported CLI definitions only when loaded as normal user overrides;
  warn with name/path, preserve valid clients/legacy warning/current precedence,
  and preserve fatal behavior for bundled and explicitly selected definitions.
  Schema errors and missing prompts remain fatal. Inspection established malformed
  JSON is already skipped by `read_json_file`, so that unrelated behavior is kept
  unchanged rather than incorrectly requiring it to become fatal.
- Only `clink/registry.py` and `tests/test_user_config_dir_rename.py` may change.
  Real Claude Code2.1.258, Read/Edit plus fixed `run_visible_tests` MCP tool,
  context65536, medium effort, requested seed29, output8192, task30min/64 turns.
  The original code1 phase uses its existing20min/32-turn policy.
- The PAL task is an actual code/test workflow in a real repository snapshot, not
  the full daily T4 skill/plugin/network/PR workflow: bare/restricted mode remains.

## Isolation amendment before scored runs

Host Python execution of model-written tests was rejected by security review.
The developer built `qwen-pal-test:20260921`, image
`sha256:3a368bcdf5c27140d15e03f9143ed3530763d0d9d937fc34c2a68e7dfd094dd1`.
Both visible and hidden PAL tests run under Docker with no network, read-only
workspace/root, no Docker socket/host profile mounts, dropped capabilities,
no-new-privileges and an unprivileged image user. Temporary data is in container
`/tmp`. Hidden tests are mounted only for parent verification; no hidden output
is returned to the model. Containers are named and force-removed on timeout.
`--noconftest` isolates the focused registry suite from unrelated provider imports.
A real CLI/MCP Docker canary and broken/gold/negative verifier controls ran.
The separate adversarial host/network canary command was permission-denied and
was not bypassed; do not claim that particular dynamic probe ran.

Each visible run records source/test hashes. The scorer requires RED with baseline
source plus changed tests, then GREEN with changed source and the same test hash
as the final submission. Raw visible logs are copied into the sealed session
inventory. Functional results and workflow adherence remain separately inspectable.

## Minecraft invalidation and fresh evidence

The user invalidated every model test since the beginning of the contaminated
hour and then explicitly confirmed Minecraft closed. The following roots are
retained with `INVALID-MINECRAFT.json`, never used for ranking or MTP selection:

- `qwen-q4-mtp-qualification`
- `qwen-q4-mtp-qualification-2`
- `qwen-q4-pal-07cp301a`
- `qwen-q4-pal-d0kim_6f`
- `qwen-q4-pal-qtao1ziw`

These are under `C:/Users/xenod/AppData/Local/Temp/`.
Fresh idle test gate after clearance:1939 passed,2 skipped,9 warnings.
Fresh qualification root: `qwen-q4-clean-qualification`. The MTP and ngram arms
use the same TURBO MTP-Q4 weights; MTP output hashes differ and its Thai-repair
probe reaches the1024 cap, so the approved conservative gate chooses ngram-only
for TURBO's task runs. This is a runtime fallback, not a regular-weight download.
Swift retains MTP+ngram. Comparisons are operating-point, not pure quant ablations.

Clean Q4 task roots:
- TURBO: `qwen-q4-pal-ckv2820y`
- Swift: `qwen-q4-pal-ki6kzney`

## Post-run oracle coverage correction

Both Q4 submissions passed the original eight hidden PAL cases and test-first
workflow checks. Primary diff/thinking review then found a missing case in the
oracle: `CLI_CLIENTS_CONFIG_PATH` can explicitly select a file/directory *inside*
a normal user directory. Both submissions incorrectly identify provenance from
physical location and downgrade the explicit error. Swift's emitted reasoning
recognized this exact distinction and then discarded it as a corner case.

A separate immutable-submission audit (`pal_registry_overlap_audit.py`) was added
without editing or giving feedback to either model. Baseline passed4/4; each Q4
submission failed4/4. The original8/8 result is preserved, but it does not mean
complete task correctness. Audit root:
`qwen-pal-overlap-audit-kr1jsfe1` under the same private temp directory.

This contract failure triggers the already-approved Q6 controls. They receive
the identical original task prompt, original baseline, same tool/test policy,
same family runtime setting and limits—no overlap-case hint or previous solution.
The audit is applied to their saved artifacts too, outside the timed original
verifier, so times retain the same boundary. No default changes or commits.
