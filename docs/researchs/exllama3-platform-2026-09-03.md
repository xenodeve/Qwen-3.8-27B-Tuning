# ExLlama3 / EXL3 — Alternative Engine Platform Tracker (2026-09-03)

> Separate platform from the main llama.cpp/NVFP4 work. Spike at the user's request.
> Goal: probe whether ExLlama3 gives better decode@deep on our 5060 Ti than llama.cpp NVFP4 dual (baseline ~35.9 real / 39.4-42.6 paired tok/s).

## Platform summary
- **Model:** `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` — EXL3 3.5bpw, ~14.2GB, fits ONE 16GB card (drops tensor-split). Built-in MTP head. NVFP4 KV. arch `Qwen3_5ForConditionalGeneration`, has `config.yarn-1m.json`, `quantization_config.json`, chat_template.jinja.
- **Fork for NVFP4 KV + DFlash2/DSpark:** https://github.com/MiaAI-Lab/exllamav3 (upstream x86-only → fork adds GB10/aarch64 + NVFP4-KV/DFlash). Upstream wheel gives MTP + fp16 KV only.
- **GB10 claims (NOT our cards):** MTP ~30 tok/s; DFlash2 47.5 tok/s HumanEval; code prose (greedy) 40-43 tok/s ≈ our ~40. No consumer-GPU Qwen3.8 EXL3 benchmark published. One head-to-head (Qwen3.6, default) favored llama.cpp ~12%. **Proof needed on our cards.**

## INSTALL STATUS (2026-09-03) — complete & certified
- venv `C:\AI\exllama3-venv` (Python 3.13.14)
- **torch 2.11.0+cu128** — TRAP: first install silently grabbed `2.11.0+cpu` (`torch.cuda.is_available()`=False). Fixed by force-installing wheel `https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp313-cp313-win_amd64.whl` → now True (CUDA 12.8).
- **exllamav3 1.4.6 via prebuilt wheel** `exllamav3-1.4.6+cu128.torch2.11.0-cp313-cp313-win_amd64.whl` (GitHub release v1.4.6, 265MB, kept at `C:\AI\exllamav3-1.4.6+cu128.torch2.11.0-cp313-cp313-win_amd64.whl`). TRAP: pip REQUIRES the wheel's exact build-tag filename — never rename it (uv rejects "invalid build tag").
- **Windows DLL trap:** MUST `os.add_dll_directory(r'C:\AI\exllama3-venv\Lib\site-packages\torch\lib')` before `import torch/exllamav3`, else `.pyd` DLL load fails. Verified: `exllamav3_ext`, `exllamav3`, core `Config,Model,Cache,Tokenizer,Generator,Job` all OK.

## Model downloaded
- `C:\AI\models\Mia-AiLab-Qwen3.8-27B-EXL3-3.5bpw\` (15.36GB, 17 files; 2× safetensors 8.5+6.8GB; config, yarn-1m, quant, chat_template).

## Test harness (`exllama3-test-decode.py`)
- Supports `--cache-quant` (nvfp4/4), `--gpu-split`, `--cache-size`, `--ctx`, `--gen`, `--rounds`, `--regime`, `--out`. Add `os.add_dll_directory` before any `import torch` (script already does). Benchmarks stream to `qwen38-tuning/results/exl3-decode.jsonl`.

## Round-1 result (2026-09-03) — cache-quant change, big win
> Full record in register + issue #71 (do not duplicate here). Single lever: **use upstream's quant cache (`-cq 4`, same size as NVFP4) so the decode CUDA-graph builds fully.** Before: decode@144K 8–13 tok/s, unoptimised.

| point | before (raw) | after (-cq 4 + graph) | llama.cpp served |
|---|---|---|---|
| 30K single-card decode | 21.4 (NVFP4 KV) | **41.7–45.3** | — |
| 65K single-card decode | 13.8–14.5 | **34.8–39.1** | — |
| 144K dual decode | 8–12 | **31.8–32.6** | 39.4–42.6 |
| 144K prefill | 191–280 | **480** | ~830 |

- After round 1, 144K-decode is close to llama.cpp (~33–34 vs 39–43) — NOT beaten, but the gap is now small, and single-card 30K already beats no-known llama.cpp-served figure.
- **Prefill is unchanged (480 vs ~830) — this is the big remaining gap.** Not yet moved.
- **In progress:** 144K dual ladder with `-cq 4` running 4 more arms (`-ndt 2`, `-ndt 3`, dynamic draft, chunk 4096 for prefill) ≈ 30 min. These will show whether the drafter + prefill-chunk help the gap.
- **Verdict correction:** my earlier "cut / it's not great" from the single raw run was WRONG — a one-lever change nearly closed the gap. Do not judge this platform from one run; it still has draft/split/prefill knobs untested.

## NEXT STEP (when GPU free)
- Run: `C:/AI/exllama3-venv/Scripts/python.exe C:/AI/exllama3-test-decode.py <model-dir>`
- Compare decode tok/s vs llama.cpp NVFP4 (baseline ~35.9 real / 39.4-42.6 paired) and GB10 claims (~30-47 tok/s).
- If it beats llama.cpp ≥ threshold: optional dual-GPU `use_per_device=[14.2,6.0]` split via library API (`-gs` in serve_openai.py is mislabeled int; split needs library API/TabbyAPI). Multi-GPU asymmetric IS supported on x86 with `tensor_p=True, tp_backend='native'` (no NCCL needed — matters on Windows).

## If NVFP4 KV wanted later
- Swap upstream wheel → Mia-AiLab **fork** build of exllamav3 (adds NVFP4 KV ~4.5 bit/elem + DFlash2 draft).
- Serve ref: `serve_openai.py -m <3.5bpw> -dm <DFlash2-5.0bpw> -cq nvfp4 -cs 262144`.

## INSTALL STATUS, SECOND PASS (2026-09-03, later the same day) — the fork is built and the model loads

The upstream 1.4.6 wheel builds zero modules for this VL-nested export, so the runtime is now the **Mia-AiLab fork, built from source** (issue #71). What changed in `C:\AI\exllama3-venv`:

- **torch 2.11.0+cu128 -> 2.11.0+cu130** (`download.pytorch.org/whl/cu130`, cp313/win) so the only toolkit on the machine, CUDA **13.3**, can build the extension. `torch/utils/cpp_extension.py` raises only on a MAJOR mismatch and warns on minor, read from source.
- **fork source** `C:\AI\exllamav3-mia` (clone of `MiaAI-Lab/exllamav3` @ 63b32f0, base version 1.4.2; **no releases, tags or wheels exist**; 60 C++/CUDA and 91 Python files differ from upstream 1.4.6, so it cannot be laid over the upstream extension). Built with `scratchpad/build-exl3-fork.cmd`: vcvars64 (MSVC 14.44) + `CUDA_HOME` 13.3 + `TORCH_CUDA_ARCH_LIST=8.9;12.0` + `pip install . --no-build-isolation --no-deps`. **Under 10 minutes.** Installs as `exllamav3 1.4.2` with `exllamav3_ext.cp313-win_amd64.pyd`.
- **triton-windows 3.8.0.post28** (pip). Without it nothing serves paged-cache attention for head_dim 256: the fork's torch-SDPA fallback only takes `dim >= 512`, and flash-attn/xformers are absent on Windows. With it the fork's preferred Triton paged kernels run, and so do the NVFP4/FP8 KV paths.
- **Patch (same as the handoff's upstream fix, re-applied to the fork, both in the source tree and the installed copy):** `exllamav3/modules/attention_fn/bc_dsa.py` wraps `from .dsa_triton import ...` in try/except and sets `bc_dsa_enable = bc_dsa_enable and _DSA_AVAILABLE`. DSA is DeepSeek-V4-only.
- **Harness rewritten:** `C:\AI\exllama3-test-decode.py` now loads through `model_init.add_args/init` exactly like `tools/serve_openai.py`, takes `--regime real-code-vendor --ctx N` and builds the prompt with `bench/dflash2_arena.filler()` (greedy, N_PREDICT 512), records `copied_frac` via `bench/harness.copied_window_fraction`, and appends one JSON row per round to `qwen38-tuning/results/exl3-decode.jsonl`. The previous script used a v2-style stream API and had never run.
- **Traps found on the way:** (1) `add_args(..., add_sampling_args=True)` is required before `get_arg_sampler`; (2) the generator reports `time_generate == time_prefill` on a cold first job, so the harness falls back to wall time minus prefill and tags the row `timing_source`; (3) `-gs 0,15.5` keeps the 4070 SUPER empty; (4) a single 16 GB card holds model 12.4 GB + NVFP4 KV, but **prefill OOMs at ctx 98,304** because EXL3 materialises a 170 MiB fp16 weight per matrix during prefill — the single-card ceiling with this KV is between 65,536 (fits, 13.6 GB) and 98,304.
- **KV per token, from config.json:** 16 full-attention layers of 64, 4 kv heads x 256 -> fp16 64 KiB, fp8 32 KiB, NVFP4 18 KiB per token. At 163,840 tokens: 10.0 / 5.0 / 2.8 GiB.

Measured decode/prefill rows live in `qwen38-tuning/results/exl3-decode.jsonl` and the register entry for issue #71.
## Housekeeping
- Untracked 5 files in repo root warning still applies — do not `git add -A`.

## SERVING (2026-09-04) — the fork's OpenAI server on the two-card recipe

`tools/serve_openai.py` assumes one large GPU (`-gs <grid>` is a single number
and there is no way to pass `-tp`). Two local patches, both marked
`Local patch 2026-09-04 (Qwen-3.8-27B-Tuning #71)` in the file:

1. `--extra "<raw model_init argv>"` appended after the server's own argv
   (a later `-gs` overrides the grid one);
2. `Generator(..., num_draft_tokens = getattr(args, "num_draft_tokens", None))`
   so `-ndt` from `--extra` is honoured.

Also `pip install aiohttp` into `C:\AI\exllama3-venv` (not a fork dependency
on Windows by default). Launch used:

```text
python tools\serve_openai.py -m C:\AI\models\Mia-AiLab-Qwen3.8-27B-EXL3-3.5bpw ^
  -dm mtp -cs 163840 -cq 4 --port 8000 --host 127.0.0.1 ^
  --extra "-tp -tpb native -gs 9,15.5 -ndt 3"
```

Ready in ~30 s; VRAM 7.7 GB (4070) + 9.8 GB (5060 Ti); host commit 12.4 +
10.8 GB for the two Python processes. Smoke test: first request 12.6 tok/s
(Triton JIT), second 46.8 tok/s at 63-token context; thinking on by default,
returned in `reasoning_content`. OpenAI API only — Claude Code needs a
translating proxy to reach it.

3. (2026-09-04, later) llama.cpp-style `timings` on every response — top level
   when not streaming, on the final chunk with `usage` when streaming — and one
   console line per request: `prompt_n/ms/per_second` (prefill),
   `predicted_n/ms/per_second` (decode, from `time_generate`, CORRECTIONS 47),
   `draft_accepted/rejected`, `wall_ms`. `cached_tokens` reads a key the Job
   does not seem to fill (0 on a cache hit whose prefill fell 980 → 165 ms) —
   open. Warm smoke test at 63-token context: prefill 381 tok/s, decode 54 tok/s.

## CLAUDE CODE PATH (2026-09-04) — `claude-xeno-exl3`

Claude Code speaks the Anthropic API; the fork's server speaks OpenAI only. A
LiteLLM proxy (`litellm[proxy]` 1.90.0, global Python) on **:4000** translates
`/v1/messages` -> `/v1/chat/completions` on :8000. Files, all in the home dir:
`~/.claude-xeno-exl3.json` (settings: model `qwen3.8-27b-exl3-3.5bpw-wm`,
`ANTHROPIC_BASE_URL` :4000, dummy token, auto-compact window 150,000),
`~/.claude/litellm-exl3.yaml`, `~/.claude/claude-xeno-exl3.bat` (checks :8000,
starts the proxy if :4000 is silent, same `--strict-mcp-config` policy as
`claude-xeno-direct`), `~/bin/claude-xeno-exl3` (Git Bash shim).

**Trap, measured with an echo backend:** LiteLLM's Anthropic adapter sends
`openai/<model>` targets to **`/v1/responses`** (the Responses API), which the
fork does not serve -> 404 "OpenAIException". `hosted_vllm/<model>` forces
`/v1/chat/completions`. Second trap: the proxy dies at startup with a cp1252
`UnicodeEncodeError` when stdout is redirected — `PYTHONIOENCODING=utf-8`.

Verified through the proxy: non-stream, SSE (`message_start` … `message_stop`),
and tool use (`tool_use` block with typed `input`, plus a `thinking` block from
`reasoning_content`). Requests queue behind whatever the server is generating
(batch-1); a smoke request waited 268 s behind a live session.

**Reasoning effort (2026-09-04, MEASURED HERE):** the Qwen3.8 chat template defaults `reasoning_effort` to **xhigh** and the fork server never passed the field, so every request through `serve_openai.py` before 13:17 ran at xhigh whatever the client set. Echo-backend probes through LiteLLM 1.90.0: a bare `output_config.effort` is dropped, but Claude Code always sends `thinking: {type: adaptive}` beside it, and for a non-Claude model the Anthropic->OpenAI adapter then maps the effort to `reasoning_effort` low / medium / high (xhigh and max both arrive as `high`). Fix shipped in `serve_openai.py`: read `reasoning_effort` on both the stream and non-stream paths (aliases high/max -> xhigh, minimal/none -> low; env `EXL3_REASONING_EFFORT` is the default) and print `reasoning effort = ...` on the end block. Verified end to end on the live server: `--effort low|medium|xhigh` in Claude Code prints low / medium / xhigh in the serve log. Note `medium` injects **no** instruction text in this template; only xhigh and low do. The 2026-09-04 Claude Code vs Deepseek Harness comparison (22m03s vs 12m15s, `logs/exl3-serve-20260904-120152.log`) ran both at xhigh; the developer's "medium" never reached the model.

**Anthropic Messages API (2026-09-04, issue #73):** the fork server now serves `/v1/messages` and `/v1/messages/count_tokens` itself — `tools/anthropic_compat.py` translates Claude Code's request onto the OpenAI route and the reply (JSON or SSE) back, over loopback; no engine code. `claude-xeno-exl3` points `ANTHROPIC_BASE_URL` at :8000 and starts no LiteLLM. Survey and verification: `anthropic-messages-api-on-exllamav3-2026-09-04.md`.

**Profile launcher (2026-09-04):** `~/.claude/claude-xeno-exl3.bat` starts the server itself when :8000 is silent (`C:\AI\launchers\serve-exl3.bat`, minimized; `EXL3_MAX=1` picks `serve-exl3-max.bat`), waits up to 5 min on `/health`, then reads `context_length` from `/health` and exports it as `CLAUDE_CODE_MAX_CONTEXT_TOKENS` and `CLAUDE_CODE_AUTO_COMPACT_WINDOW` — Claude Code 2.1.258's own notice for an unknown model says to set the first to the real window; the json no longer pins a window, so keys F and G both get the right meter. Verified: auto-start of G from a stopped server answered a `-p` prompt in 1 min 14 s (`exl3-serve-20260904-135556.log`).

**Our server is a module of ours (2026-09-04, later the same day):** the customisations left the fork tree. `qwen38-tuning/serving/exl3/server.py` is the fork's `tools/serve_openai.py` (pristine copy + commit in `upstream/`) plus 49 marked `# xeno` hook lines; everything else is in `live_timing.py`, `effort.py`, `anthropic_compat.py`, `anthropic_routes.py`. `serve-exl3.cmd` runs that file; `C:\AI\exllamav3-mia` is pristine again apart from the Windows `bc_dsa.py` import guard, so it can be pulled, and a fork update is a three-way merge (README in that directory). Pinned by `bench/tests/test_exl3_serving_module.py`. **Keep-alive:** `/v1/messages` now sends `message_start` at once and an Anthropic `ping` every 5 s of upstream silence — the llama.cpp lesson (`--sse-ping-interval 5`, report 05 §3); measured on a ~25K-token prompt: pings at 5, 10, 15 … s, ten before the first content block at 50.2 s, where Claude Code previously reported an API error.

**Model name (2026-09-04, 16:40):** `/v1/models`, `/health` and every response now name the loaded directory (`MODEL_NAME = basename(-m)` in `server.py`) instead of the literal `qwen3.8-27b-exl3-3.5bpw-wm` the fork hard-coded; `claude-xeno-exl3.bat` reads that id and passes `--model=`, so Claude Code shows which file is served (`turboderp-Qwen3.8-27B-EXL3-SC4.0bpw-H5` today). The json no longer pins a model.

**Live token counter (2026-09-04, 17:10):** Claude Code's counter only moved when a turn ended and then jumped (1.2k -> 10k at the next tool call). Read from the binary: it takes `message_delta.usage.output_tokens`, and the real API streams that mid-turn (the `thinking-token-count` beta) while ours came once, at the end. Now the OpenAI route puts a running `usage.completion_tokens` on every chunk (`on_tokens` hook in `server.py`) and the translator emits an interim `message_delta` with `delta: {}` and `usage.output_tokens` every 32 tokens (`StreamTranslator.USAGE_EVERY`); the final one still carries stop_reason and the full usage. Wire check: 12 interim events at 33, 66, 98 … 393 on a 396-token reply; Claude Code's `--output-format stream-json` shows it accepting them. Whether the interactive counter moves is for the developer to confirm.

**Issue #74 batch (2026-09-04 evening, three `gpt-5.6-luna`/max workers, verified here):** history thinking -> `reasoning_content`; mid-conversation `system` -> `System note:` user turn; `unknown_blocks` reported; `normalize_billing_header` (`cch=` -> `fffff`, llama.cpp PR #21793); pre-flight count with `budget()` — `max_tokens` clamped to what the window leaves after the prompt (the profile sets Claude Code's cap to the whole window, 262,144, so a long think is never cut at 16,384 and restarted: three xhigh turns hit that cap on the Google benchmark) and 400 `prompt is too long: N tokens > M maximum` when fewer than 256 remain; `logs/exl3-requests.jsonl` one line per request; `EXL3_CAPTURE_DIR`; `API_TIMEOUT_MS=3600000`; `tools/exl3-smoke.py`. The first live smoke run found two real defects the unit tests could not: the too-long probe undershot the window (4 chars/token; now sized by `count_tokens`) and `tool_choice: {type: tool}` made the pre-count append a second system message -> `System message must be at the beginning` 500 (now merged into the first, as `generate_full` does).

**Hub (2026-09-04):** `serve-hub.bat` keys **F** (163,840, measured) and **G** (262,144 native max, `-gs 10,15.5`, UNMEASURED) -> `launchers\serve-exl3*.bat` -> `qwen38-tuning\scripts\serve-exl3.cmd`, the single flag holder. Pinned by `bench/tests/test_exl3_launcher.py`; the four .bat files also pass the cmd-parse test, and `test_exl3_launcher_args_survive_cmd.py` runs their argument string through real cmd.exe after the comma-split incident (`10,15.5` bare -> `-gs 10`, one card, model failed to load).
