# serving/exl3 — our EXL3 server, kept apart from the fork

The ExLlamaV3 fork at `C:\AI\exllamav3-mia` (Mia-AiLab) is a tree we pull
updates into, so nothing of ours is edited in place there. This directory is
the server we actually run (`scripts/serve-exl3.cmd`, hub keys F and G).

| file | what | whose |
|---|---|---|
| `server.py` | the fork's `tools/serve_openai.py` plus hooks, every one marked `# xeno` | fork + hooks |
| `upstream/serve_openai.py`, `upstream/COMMIT` | the pristine copy `server.py` was cut from, and the fork commit it came from | fork |
| `live_timing.py` | llama-server-shaped live lines every 3 s and the end block (`prompt eval time`, `eval time`, `total time`, `draft acceptance`, `reasoning effort`); the `timings` dict — decode rate over `time_generate` only (CORRECTIONS §47) | ours |
| `effort.py` | `reasoning_effort` -> xhigh / medium / low for the Qwen3.8 template; default `EXL3_REASONING_EFFORT` | ours |
| `anthropic_compat.py` | Anthropic Messages <-> OpenAI chat, pure functions; `StreamTranslator`; `pump()` with a `ping` every 5 s of upstream silence | ours |
| `anthropic_routes.py` | `/v1/messages` (a pipe over the server's own `/v1/chat/completions`) and `/v1/messages/count_tokens`; pre-flight count: `max_tokens` is clamped to what the window leaves after the prompt (`budget()`, so the profile can set Claude Code's cap to the whole window and a long think is never cut), and when fewer than `MIN_GEN` = 256 tokens are left the answer is 400 `prompt is too long: N tokens > M maximum` (the text Claude Code auto-compacts on); one JSONL line per request in `EXL3_REQUEST_LOG` (default `logs/exl3-requests.jsonl`); `EXL3_CAPTURE_DIR` dumps request bodies as contract fixtures | ours |
| `watchdog.py` | issue #75, the server comes back when it is down. Three paths: (1) `check(exc)` from both generation error paths — on the fork's dead-child signatures (`CPU reduce process timeout`, `Synchronization timeout`, `Timed out waiting for worker`) it writes `logs/exl3-restart.flag` and exits 3 a second later; (2) `start_self_probe` (from `on_startup`) polls the server's own `/health` every 30 s and exits the same way after two consecutive 20 s misses — alive but deaf; (3) `scripts/serve-exl3.cmd` relaunches after ANY exit unless `logs/exl3-stop.flag` exists (`scripts/stop-exl3.cmd` writes it, then kills the tree by command line with `taskkill /T`), and gives up after three exits within 420 s of their start. Any ordinary error is left alone. Seen 2026-09-04 21:00: `/health` said ok for an hour while every completion 500'd in 0.4 s. Loop proven 2026-09-05 with a missing model directory (2 relaunches, then "giving up"); a real death has not yet been through it | ours |
| `loop_guard.py` | issue #76: stops a generation that has degenerated into repetition — the last 512 generated characters hold ≤ 2 distinct characters, or one unit of ≤ 8 characters repeated end to end. Fed every text chunk in `generate_full`; on a trip the job is cancelled, the client sees `finish_reason: length`, `timings.stop_reason = "loop"`, one log line, `/health.loops_stopped` counts it. Second rule, thinking only (`in_think`): a prose unit of 64 characters repeated 8 times in the last 4,096 characters of thinking — the 19:03 mode, a three-sentence cycle ("OK, I'm going to write the code now. Let me stop deliberating…") repeated ~1,000 times for 127,996 tokens; markup units are ignored because a healthy run drafts section headers in thought. Replayed over the 43 Claude Code streams of 2026-09-05: trips on the two runaways only (13:19 at character 1,913; 19:03 at 167,681 of thinking), nowhere else. The window-sized output cap stays | ours |
| `cjk_guard.py` | issue #77: no Han (Chinese) character in the prompt -> none in the answer. Over the 43 streams of the 2026-09-05 bench, 14 Han characters leaked into 3 streams, always mid-Thai-sentence (`โมเดล前沿…`, `协作`): sampling drift, which a prompt line cannot reach. Every vocab piece with a Han character (55,328 of 248,044 on the 4.0bpw tokenizer, scanned once) goes into `ComboSampler`'s `logit_bias` at `-inf`, thinking included. A prompt that carries Han (pasted text, a tool result) or names Chinese/China (`จีน`, `china`, `chinese`, `mandarin`, word-bounded in English) lifts the ban for that request; `EXL3_ALLOW_CJK=1` lifts it for the server. The instrument: `timings.cjk_chars` per completion, a log line when non-zero, `/health.cjk_chars_total` | ours |

Tests: `bench/tests/test_exl3_serving_module.py` (this shape),
`test_exl3_anthropic_messages.py`, `test_exl3_anthropic_routes.py`,
`test_exl3_live_timing_and_effort.py`, `test_exl3_watchdog.py`, `test_exl3_loop_guard.py`, `test_exl3_cjk_guard.py`.

**Diagnosing the live view** (thinking / token counter in Claude Code): start the server with `EXL3_TRACE_SSE=C:\AI\qwen38-tuning\logs\exl3-sse.jsonl` and render the last request with `python qwen38-tuning	ools\exl3-trace.py` (server side, every SSE event timed, silences flagged); `--claude-p "<prompt>"` renders what Claude Code itself received via `stream-json`. Interim `message_delta` usage events are off by default (`StreamTranslator(interim_usage=True)` to re-enable): on the interactive UI they froze the live view after the first one.

**After every fork or Claude Code update** run `python qwen38-tuning/tools/exl3-smoke.py`
against the live server (issue #74): health, count_tokens, stream shape incl.
`signature_delta`, effort reaching the template, a tool_use stream, the too-long
400, and a `claude -p` round-trip.

## Updating the fork

```text
cd C:\AI\exllamav3-mia && git pull                      # tree is pristine, pulls clean
git diff <old COMMIT> HEAD -- tools/serve_openai.py     # what upstream changed
```

Then a three-way merge: `upstream/serve_openai.py` (base) vs the fork's new
`tools/serve_openai.py` (theirs) vs `server.py` (ours). Every line of ours in
`server.py` carries `# xeno` or sits in the header, so the merge is mechanical.
Afterwards copy the new file over `upstream/serve_openai.py`, write the new
hash to `upstream/COMMIT`, and run the three test files. The one edit that does
live in the fork tree is `exllamav3/modules/attention_fn/bc_dsa.py` (a guarded
triton import so `import exllamav3` works on Windows); it is a library fix,
not a server customisation, and `git status` shows it.

## Hooks in `server.py`

`--extra` (raw `model_init` argv, how the two-card recipe is served) ·
`-ndt` honoured · `LiveTiming` calls in the generation loop · `effort`
passed to the chat template · `timings` in the response and the final SSE
chunk · `reasoning_effort` read from the request on both paths ·
`anthropic_routes.register(app, ...)` · `app["port"]` for the loopback pipe ·
`watchdog.check(e)` in both generation error paths and `watchdog.start_self_probe` on `on_startup` (#75).
