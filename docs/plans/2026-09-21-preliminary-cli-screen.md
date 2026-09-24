# Preliminary Claude Code CLI quality/time screen — 2026-09-21

Tracking: issue #92. Developer authorized this reduced preliminary evaluation and
explicitly confirmed Minecraft closed and both GPUs available on2026-09-21.
This is a separate screen, not completion or replacement of the long-horizon PRD.

Completed: [result23](../results/23-preliminary-cli-quality-time-2026-09-21.md).
All four passed the frozen coding checks; a single sample does not establish a winner.

## Frozen protocol before model outcomes

- Actual Swift Q6_K, TURBO MTP-Q6_K, Dirk UD-Q6_K, GSQ IQ3_S-MTP, in that order.
- One fresh boot, one independent `code-task-1` workspace/session per candidate.
- Reuse result22 selected configurations:65536, embedded templates, tensor split
  9500,14500 for Swift/TURBO and8500,15468 for Dirk/GSQ, Q4_0 KV,
  draft-mtp+ngram-mod, draft3, ngram24/16/64, medium effort, sampler1.0/.95/20/0.
- Seed29 requested for every generation; this is not a claim of deterministic
  cross-model trajectories. Preserve conditional Han masking and record requests.
- Installed real Claude Code CLI2.1.258 in bare/restricted mode with only effective
  Edit/Read tools. This is the primary CLI, but **not its unrestricted daily tool
  profile**. No Bash, external MCP, custom model system prompt, or permission bypass.
- Client context explicitly65536 via CLAUDE_CODE_MAX_CONTEXT_TOKENS, output8192,
  max32 assistant turns,1200-second session deadline. No retries, repairs or nudges
  after the parent verifier; measure first-pass completion only.
- Adapt the existing Thai brief only to explain that tests are run by the parent,
  not by the model. Require implementation and new tests in the two allowed files.
  No hidden tests, expected solution, or oracle feedback enters the agent workspace.
- Parent runs visible and seven frozen hidden tests on a separate copy after CLI
  exit. Preserve raw stdout/stderr/XML and original/final agent trees. Reject
  out-of-scope modifications. Review generated tests separately: modifying a test
  file alone is not evidence that it tests the requirements meaningfully.
- Validate unchanged fixture fails and reference solution passes before GPU use.
- Quality: hidden tests, visible tests, change scope, added-test adequacy, Thai final
  explanation and full emitted-thinking/final consistency. No arbitrary weighted sum.
- Time: real elapsed time from client launch through parent verification including
  recording/analysis/copy overhead; report load/hash/guard setup separately. Retain
  failure/censored/invalid costs. Zero passes gives no finite time-per-verified-task.
- Stream evidence remains private in a unique OS-temp directory. Freeze prompt,
  fixture inventory, source snapshots, client/engine/model hashes and launch argv.
- Own model process and dedicated port18080; never call legacy global-stop routes.
  Hold the campaign OS lock and a live Git Bash supervisor's legacy port8080
  lease. Check exact lease contents before each candidate. The developer explicitly
  authorized archiving the old lease (PID185122 absent in both PID namespaces);
  it was preserved at `C:\Users\xenod\AppData\Local\Temp\qwen-legacy-lock-20260921-001.txt`.
  Never delete an unknown/stale lease or stop its claimed owner automatically.
- Record null RAM/process sampler fields where optional psutil is absent; supplement
  startup with a Windows process-name check. No claim of complete process telemetry.

## Instrument admission record

The first live attempt was invalidated, not scored as a Swift failure:
`C:\Users\xenod\AppData\Local\Temp\qwen-preliminary-cli-z41ul_pg`.
Its Swift cell consumed183.562s including setup before stopping the screen; the
other three cells are explicitly not_run. The EXL3 stream translator expected
complete tool calls, whereas llama.cpp emits index-keyed argument fragments. It
closed the first Read block after `{`, then opened nameless continuation blocks;
the CLI rejected all16 calls. This also caused early client disconnects and lost
trailing usage. No model settings or task requirements were selected from this run.

The bench-only adapter now coalesces tool argument fragments and delays the final
Anthropic stop until trailing upstream usage has arrived. Production EXL3 code
was not changed. The regression failed before the fix; the real installed CLI
then edited a fixture successfully using fragmented arguments and retained the
trailing synthetic usage in
`C:\Users\xenod\AppData\Local\Temp\qwen-fragment-canary-91p95v0y`.
Replacement measurement uses fresh sessions/workspaces and keeps the invalid
attempt in the infrastructure-cost record, not the model-quality denominator.

## Admission and limits

Real fake-upstream CLI canaries established actual in-workspace editing and an
out-of-workspace Read permission denial, with no sentinel leak. A separate context
canary observed65536 in the client's result metadata. These are instrument checks,
not model-quality evidence. Actual GPU preflight, runtime identity/context/full
residency and final evidence readback are still mandatory per candidate.

One small coding task and one sample per model supports preliminary observations,
not a statistically stable winner, long-horizon reliability, general Thai quality,
maximum-context certification or default promotion. Old direct-answer results are
not pooled with these CLI task times. A broken shared instrument stops the screen.
