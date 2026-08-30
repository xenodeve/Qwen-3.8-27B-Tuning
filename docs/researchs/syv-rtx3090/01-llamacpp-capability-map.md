# 01 — llama.cpp capability map (build 10499)

**This file is the one part of this folder that is evidence.** Everything
else here describes someone else's system; this describes ours, read from
`C:\AI\llama.cpp` at commit `1deefcca3` — llama.cpp PR #27342 on top of
master, the tree that built the binary in use.

**175 capabilities across 6 areas.** Compiled by six agents, one per area,
each asked for defaults, clamps and interactions rather than a feature list —
a flag whose default surprises is worth more than one that behaves.

---

## llama-server: slots, -np, cross-request prompt caching, state endpoints (/props, /slots, /tokenize, /apply-template, /metrics), chat templates, and every log line that emits a benchmark-readable number. Source tree C:\AI\llama.cpp @ 1deefcca3 (build 10499, verified against C:\AI\llama.cpp-dflash2\llama-server.exe --version / --help).

### `-np / --parallel: number of server slots`
**Use:** -np N (N != 0). Env LLAMA_ARG_N_PARALLEL. N = -1 means auto.  ·  **default:** `-1 (auto) for the server example only; common_params default is 1 for every other example`

SURPRISE: the server default is NOT 1. common/arg.cpp:1401 overrides n_parallel to -1 for LLAMA_EXAMPLE_SERVER, and tools/server/server.cpp:151-155 turns -1 into `n_parallel = 4; kv_unified = true`. Passing -np explicitly (e.g. `-np 1`) takes the other branch, so kv_unified is NOT set and stays at its common_params default of false (common/common.h:563). `-np 0` throws at common/arg.cpp:2537-2539. Slots are created 1:1 with n_parallel at tools/server/server-context.cpp:1224-1226 and each gets slot.id = i (server-context.cpp:1252).

`common/arg.cpp:1401` · `common/arg.cpp:2533-2543` · `common/common.h:447` · `tools/server/server.cpp:151-155`

### `Per-slot context size (n_ctx_slot) derivation`
**Use:** -c/--ctx-size N sets the TOTAL context; the per-slot context is derived.  ·  **default:** `n_ctx = 0 (= model training context); n_ctx_slot = llama_n_ctx_seq()`

CLAMPS, in order: (1) src/llama-context.cpp:287 pads n_ctx up to a multiple of 256; (2) if kv_unified, n_ctx_seq = n_ctx (llama-context.cpp:290); else n_ctx_seq = n_ctx / n_seq_max padded to 256, and n_ctx is then RECOMPUTED as n_ctx_seq * n_seq_max with the warn line `n_ctx is not divisible by n_seq_max - rounding down to %u` (llama-context.cpp:292-302); n_ctx_seq == 0 throws. (3) server-context.cpp:1202-1206 caps n_ctx_slot to the model's n_ctx_train and logs `the slot context (%d) exceeds the training context of the model (%d) - capping`. cparams.n_seq_max = params.n_parallel (common/common.cpp:1698). With -np 1 (kv_unified false) n_ctx_seq == n_ctx, so the split is invisible; with -np 4 each slot gets a quarter.

`src/llama-context.cpp:287` · `src/llama-context.cpp:289-303` · `tools/server/server-context.cpp:1202-1206` · `tools/server/server-context.cpp:1112` · `tools/server/server-context.cpp:1257`

### `Startup line that reports the slot geometry`
**Use:** Read stderr for `srv init: initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'`  ·  **default:** `printed at INFO level, always visible at default verbosity 3`

This is the single authoritative line for what the server actually built, after every clamp above. Companion lines from libllama (src/llama-context.cpp:305-318) print n_seq_max, n_ctx, n_ctx_seq, n_batch, n_ubatch, kv_unified, n_outputs_max. Default verbosity is 3 = LOG_LEVEL_INFO (common/common.h:524, common/log.h:26), so SRV_INF/SLT_INF are visible and SRV_TRC (4) / SRV_DBG (5) are NOT — several prompt-cache and checkpoint diagnostics are TRC and silent unless -lv 4 / -v.

`tools/server/server-context.cpp:1220-1222` · `tools/server/server-common.h:33` · `common/common.h:524` · `common/log.h:24-32`

### `Slot selection: explicit id_slot`
**Use:** POST body field `"id_slot": N` on /completion, /v1/chat/completions, /infill etc.  ·  **default:** `-1 (let the server choose)`

SURPRISE: id_slot is read with a bare json_value at server-context.cpp:4235 — it is NOT in the request schema and has NO bounds check. get_slot_by_id (1463-1464) does `id_slot = id_slot % slots.size()`, so an out-of-range id silently WRAPS instead of erroring. Logs `selected slot by id (%d)` at INFO. If that slot is busy the whole task is deferred (2329-2333), it does not fall back to another slot.

`tools/server/server-context.cpp:4235` · `tools/server/server-context.cpp:1489-1500` · `tools/server/server-context.cpp:1462-1473`

### `Slot selection: LCP prompt similarity (-sps)`
**Use:** -sps / --slot-prompt-similarity FLOAT (0.0 disables)  ·  **default:** `0.10`

For each IDLE non-empty slot it computes f_sim = LCP(slot.prompt, request)/len(request) and picks the best strictly above the threshold (1532). Emits `selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f` at INFO (1543). f_keep = (f_sim_best*len(request))/len(slot.prompt) — if f_keep < 0.5 it sets update_cache=true (1546-1548), which triggers the RAM prompt-cache save/load below. NOTE: this runs even with a single slot, so -np 1 still exercises the prompt-cache round-trip.

`common/arg.cpp:3736-3742` · `common/common.h:677` · `tools/server/server-context.cpp:1198` · `tools/server/server-context.cpp:1503-1551`

### `Slot selection: LRU fallback`
**Use:** automatic when no slot passes the similarity threshold  ·  **default:** `n/a`

Picks the idle slot with the smallest t_last_used and logs `selected slot by LRU, t_last = %lld` at INFO. ALWAYS sets update_cache = true (1573), so any LRU selection pays a prompt_save + prompt_load against the RAM cache. If every slot is busy, ret == nullptr and the task is deferred (server-context.cpp:2322-2327, `no slot is available, defer task` at DEBUG only).

`tools/server/server-context.cpp:1553-1574`

### `Cross-request KV reuse inside a slot (prompt caching)`
**Use:** --cache-prompt / --no-cache-prompt; per-request `"cache_prompt": bool`  ·  **default:** `enabled`

When on, n_past = slot.prompt.tokens.get_common_prefix(input) (3126). When off, n_past is forced to 0 and the whole prompt is re-decoded (3197-3200). This is the in-slot (VRAM) reuse, distinct from the RAM prompt cache below.

`common/common.h:611` · `common/arg.cpp:3513-3520` · `tools/server/server-schema.cpp:31-32` · `tools/server/server-context.cpp:3125-3126` · `tools/server/server-context.cpp:3197-3200`

### `The forced 1-token re-evaluation [TAG_PROMPT_LOGITS]`
**Use:** automatic  ·  **default:** `n/a`

BENCHMARK TRAP. If the cached prefix covers the ENTIRE prompt, n_past is decremented by 1 and `need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)` + `n_past was set to %d` are logged at WARN. stats.n_prompt_cached is assigned AFTER the decrement (3320), so a fully cached 1000-token prompt reports cache_n = 999, prompt_n = 1. prompt_per_second is then 1 token over the whole slot-start-to-prompt-end wall time — a meaningless number that still looks plausible.

`tools/server/server-context.cpp:3313-3318` · `tools/server/server-context.cpp:3320-3323`

### `--cache-reuse: KV shifting of non-prefix chunks`
**Use:** --cache-reuse N (min chunk size); per-request `"n_cache_reuse": N`  ·  **default:** `0 (disabled)`

Requires prompt caching. Silently forced to 0 when a mmproj is loaded (server-context.cpp:1170-1174, WARN `cache_reuse is not supported by multimodal, it will be disabled`) and when llama_memory_can_shift() is false (1182-1186). Per-request n_cache_reuse > 0 on a context that cannot shift logs `cache reuse is not supported - ignoring n_cache_reuse = %d` at WARN (3145) and is dropped. The reuse loop (3161-3193) does seq_rm + seq_add per matched chunk and logs `reusing chunk with size %zu, shifting KV cache ...` at TRC only.

`common/arg.cpp:3522-3531` · `common/common.h:610` · `tools/server/server-schema.cpp:67-69` · `tools/server/server-context.cpp:3138-3196`

### `RAM prompt cache across slots (--cache-ram)`
**Use:** -cram / --cache-ram N (MiB). -1 = no limit, 0 = disable.  ·  **default:** `8192 MiB`

Constructed as server_prompt_cache(cache_ram_mib, n_ctx) where n_ctx is the TOTAL context (server-context.cpp:1112, 1312) — that second argument becomes limit_tokens (server-task.h:608-611). SILENT SELF-DEMOTION: on std::bad_alloc while sizing a state, limit_size is cut to 0.4*current size and `- cache size limit reduced to %.3f MiB` is logged at WARN (server-task.cpp:1762-1768) — the cache silently shrinks for the rest of the run. alloc() refuses entries larger than the limit (`- prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping`, 1724-1728) and evicts oldest-first (1745-1751). load() refuses to trash an entry whose f_keep < 0.25 (1806-1808). update() also enforces limit_tokens, dynamically raised to limit_size/bytes-per-token (1866-1875).

`common/common.h:615` · `common/arg.cpp:1703-1710` · `tools/server/server-context.cpp:1305-1316` · `tools/server/server-task.h:607-620` · `tools/server/server-task.cpp:1706-1783` · `tools/server/server-task.cpp:1790-1856` · `tools/server/server-task.cpp:1858-1878`

### `--cache-idle-slots: publish idle slots to the RAM cache on every new task`
**Use:** --cache-idle-slots / --no-cache-idle-slots  ·  **default:** `enabled`

Auto-disabled with WARN `--cache-idle-slots requires --cache-ram, disabling` when cache_ram_mib == 0 (server-context.cpp:1374-1377). [TAG_IDLE_SLOT_CLEAR]: when kv_unified is TRUE, every idle slot's KV is CLEARED after being saved (2365-2369) — under -np auto (which turns kv_unified on) an idle slot loses its VRAM KV as soon as any other request arrives, and the next hit has to reload from RAM. When kv_unified is false (i.e. any explicit -np) only the RAM copy is made and VRAM KV survives (comment at 1381-1383).

`common/common.h:612` · `common/arg.cpp:1721-1727` · `tools/server/server-context.cpp:1373-1387` · `tools/server/server-context.cpp:2355-2371`

### `try_clear_idle_slots (KV pressure relief)`
**Use:** automatic on decode failure/pressure  ·  **default:** `n/a`

Returns immediately unless params_base.kv_unified (1600-1602) — with any explicit -np this is a permanent no-op. When active it purges one idle slot per call and logs `purging slot %d with %zu tokens` at WARN (1622).

`tools/server/server-context.cpp:1596-1626`

### `Context checkpoints (-ctxcp / -cms)`
**Use:** -ctxcp / --ctx-checkpoints / --swa-checkpoints N ; -cms / --checkpoint-min-step N  ·  **default:** `n_ctx_checkpoints = 32 ; checkpoint_min_step = 8192`

Only created for SERVER_TASK_TYPE_COMPLETION (3374) and only when the context cannot roll back partially (COMMON_CONTEXT_SEQ_RM_TYPE_FULL / _RS) or the model uses SWA without --swa-full (3381-3384). Never after an mtmd chunk (3533). Eviction is two-stage: entries closer than checkpoint_min_step to an earlier one are erased (2249-2261, TRC), then oldest-first until size < n_ctx_checkpoints, logged at WARN as `erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %lld, size = %.3f MiB)` (2266-2270). Restore on a prefix mismatch logs `restored context checkpoint (...)` or `forcing full prompt re-processing due to lack of cache data` at TRC (3286-3294). Known bug tagged in-source: [TAG_CHECKPOINTS_FIX_POS_MIN] at 2278-2281 — the saved range is wrong for SWA models.

`common/common.h:613-614` · `common/arg.cpp:1686-1704` · `tools/server/server-context.cpp:3371-3384` · `tools/server/server-context.cpp:3528-3546` · `tools/server/server-context.cpp:2243-2291` · `tools/server/server-context.cpp:3260-3298`

### `Context shift (--context-shift)`
**Use:** --context-shift / --no-context-shift ; --keep N ; per-request `n_keep`, `n_discard`  ·  **default:** `DISABLED (ctx_shift = false); n_keep = 0; n_discard = 0`

SURPRISE: context shift is OFF by default, so generation simply STOPS at n_ctx with stop_type LIMIT and truncated=true, logged only at DEBUG: `stopped due to running out of context capacity, ...` (1825-1827). When enabled: n_keep = (n_keep < 0 ? whole prompt : n_keep), +1 if the model adds BOS, then CLAMPED to n_ctx - 4 (2850). n_discard defaults to n_left/2 and is clamped to [0, n_left-1] (2853-2854). Logged at WARN: `slot context shift, n_keep = %d, n_left = %d, n_discard = %d` (2857). Auto-disabled with WARN when mmproj is loaded (1165-1168) or llama_memory_can_shift() is false (1176-1180). Refused for parent/child (n_cmpl>1) tasks with a server error (2836-2840).

`common/common.h:561` · `common/common.h:445` · `common/arg.cpp:1729-1736` · `common/arg.cpp:1672-1678` · `tools/server/server-context.cpp:2818-2879` · `tools/server/server-context.cpp:1819-1828` · `tools/server/server-schema.cpp:54-60`

### `Prompt-too-long behaviour`
**Use:** n/a  ·  **default:** `n/a`

There is NO silent prompt truncation. A splittable task with n_tokens >= slot.n_ctx returns ERROR_TYPE_EXCEED_CONTEXT_SIZE `request (%d tokens) exceeds the available context size (%d tokens), try increasing it` (3118-3125). A non-splittable task (embeddings with non-LAST pooling, server-context.cpp:399-405) gets `input (%d tokens) is too large to process. increase the physical batch size` if > n_ubatch (3096-3102) or `input (%d tokens) is larger than the max context size` if > n_ctx (3105-3113). The error result carries n_prompt_tokens and n_ctx (server-task.cpp:1505-1508).

`tools/server/server-context.cpp:3116-3127` · `tools/server/server-context.cpp:3103-3115`

### `Continuous batching (-cb)`
**Use:** -cb / --cont-batching, -nocb / --no-cont-batching  ·  **default:** `enabled`

When disabled, new prompts are only admitted when the batch is empty (`if (params_base.cont_batching || batch.size() == 0)`, 3022). Slots can only share a batch if can_batch_with() holds: same task type, same input-embedding size, identical LoRA set (server-context.cpp:407-413).

`common/common.h:558` · `common/arg.cpp:2559-2565` · `tools/server/server-context.cpp:3022`

### `GET /props`
**Use:** GET /props (always available, works even while sleeping)  ·  **default:** `n/a`

Reports total_slots = params.n_parallel, default_generation_settings.n_ctx = meta->slot_n_ctx (the PER-SLOT context, taken from slots.back().n_ctx at 3939-3941 — not the total), chat_template, chat_template_caps, bos_token/eos_token, build_info, model_path, model_ftype, modalities, endpoint_slots/props/metrics flags, is_sleeping. chat_template_tool_use is added only when --jinja is on and a tool_use variant exists (4622-4626). GET /props is NOT gated by --props.

`tools/server/server-context.cpp:4580-4629` · `tools/server/server-context.cpp:3939-3941` · `tools/server/server-context.cpp:4107`

### `POST /props`
**Use:** --props to enable; POST /props  ·  **default:** `disabled`

HARD FACT: even when enabled, the handler changes nothing — the body is ignored and it returns {"success": true} (4638-4642, comment `// update any props here`). --props buys you a 200 instead of a 501, nothing else.

`common/common.h:654` · `common/arg.cpp:3540-3546` · `tools/server/server-context.cpp:4632-4643`

### `GET /slots`
**Use:** --slots / --no-slots ; GET /slots ; GET /slots?fail_on_no_slot=1  ·  **default:** `ENABLED`

Per slot: id, n_ctx, speculative, is_processing, id_task, n_prompt_tokens, n_prompt_tokens_processed, n_prompt_tokens_cache, params, next_token{has_next_token, has_new_line, n_remain, n_decoded} (643-676). Prompt text and generated text are ONLY included when the env var LLAMA_SERVER_SLOTS_DEBUG is non-zero — slot.to_json(only_metrics = slots_debug == 0) at 2430, env read at 1283-1289. `?fail_on_no_slot=` (any value) turns zero idle slots into an ERROR_TYPE_UNAVAILABLE response (4534-4540). The request is served as a high-priority METRICS task, so it reflects a consistent snapshot, and unlike /metrics it does NOT reset the rate buckets (4513 sets no metrics_reset_bucket).

`common/common.h:653` · `common/arg.cpp:3547-3554` · `tools/server/server-context.cpp:4505-4544` · `tools/server/server-context.cpp:2419-2451` · `tools/server/server-context.cpp:643-680` · `tools/server/server-context.cpp:1282-1289`

### `POST /slots?id_slot=N&action=save|restore|erase`
**Use:** --slot-save-path DIR to enable  ·  **default:** `disabled (empty path)`

--slot-save-path validates the directory at parse time and appends a trailing separator (arg.cpp:3560-3567). Without it, POST /slots is ERROR_TYPE_NOT_SUPPORTED (4549-4552). If the target slot is busy the task is DEFERRED, not rejected (2461-2466 / 2511-2516 / 2578-2583). Restore rejects a state larger than slot.n_ctx and one that fails token validation (2537-2544). Results carry n_tokens, n_bytes and t_ms — real measured numbers (2497-2501, 2557-2562).

`common/arg.cpp:3555-3569` · `tools/server/server-context.cpp:4547-4577` · `tools/server/server-context.cpp:5156-5245` · `tools/server/server-context.cpp:2453-2589`

### `POST /tokenize and /detokenize`
**Use:** POST /tokenize {content, add_special, parse_special, with_pieces}; POST /detokenize {tokens}  ·  **default:** `add_special = false, parse_special = TRUE, with_pieces = false`

SURPRISE: parse_special defaults to true (4936) — special tokens in the text ARE consumed as tokens by default, which changes token counts versus a plain tokenizer. add_special (BOS/EOS) defaults to FALSE, so /tokenize counts will not match what a completion request actually feeds the model. No chat template is applied. If `content` is absent the response is an empty token array, not an error (4932/4964). with_pieces returns byte arrays instead of strings for invalid UTF-8 (4946-4955).

`tools/server/server-context.cpp:4929-4966` · `tools/server/server-context.cpp:4970-4989`

### `POST /apply-template`
**Use:** POST /apply-template with an OAI-shaped chat body  ·  **default:** `n/a`

Runs the full oaicompat_chat_params_parse pipeline and returns ONLY {"prompt": ...} (4883). It uses the same meta->chat_params as the real chat endpoint, so it is the reliable way to see what the template produces — but it discards the grammar/parser side of the result, and files are parsed into a dummy vector (4878).

`tools/server/server-context.cpp:4876-4886` · `tools/server/server-common.cpp:1128-1330`

### `GET /metrics (Prometheus)`
**Use:** --metrics to enable; GET /metrics  ·  **default:** `disabled`

BENCHMARK TRAP: the two throughput gauges llamacpp:prompt_tokens_seconds and llamacpp:predicted_tokens_seconds are computed from buckets that are RESET on every /metrics scrape (task.metrics_reset_bucket = true at 4477, reset at server-context.cpp:2444-2446). They are an average over the window between scrapes, so a second scrape moments later returns 0. /slots does NOT reset them. Rates use `steps`, not `count` (server-common.h:445-447), and generation steps exclude the first token (server-common.h:399-402), so a 1-token completion contributes 0 steps. Counters: prompt_tokens_total (excludes cached), prompt_tokens_cached_total, prompt_seconds_total, tokens_predicted_total, tokens_predicted_seconds_total, n_decode_total, n_tokens_max, spec_decode_* . tokens_predicted_total is only added on slot RELEASE (server-context.cpp:4037-4043 via callback_on_reset at 1268-1273), so it lags an in-flight request entirely. Response also sets the Process-Start-Time-Unix header (4498).

`common/common.h:655` · `common/arg.cpp:3533-3539` · `tools/server/server-context.cpp:4466-4502` · `tools/server/server-task.cpp:1520-1614` · `tools/server/server-common.h:432-492`

### `Per-request timings block`
**Use:** present on non-stream completions; `"timings_per_token": true` puts it on every stream chunk  ·  **default:** `timings_per_token = false`

Fields: cache_n, prompt_n, prompt_ms, prompt_per_token_ms, prompt_per_second, predicted_n, predicted_ms, predicted_per_token_ms, predicted_per_second, and draft_n/draft_n_accepted when speculation ran. DEFINITIONS THAT MATTER: prompt_n = n_prompt_processed EXCLUDES cached tokens; prompt_ms = t_prompt_last - t_start where t_start is set when the slot enters PROCESSING_PROMPT (server-common.h:363-366, server-context.cpp:3053), so cache lookup, checkpoint restore and RAM-cache load are inside the numerator's wall time but outside prompt_n. predicted_per_second divides by n_gen_steps() = n_gen - 1 (server-common.h:400-402), i.e. the first token is free; a 1-token generation reports 0 t/s. t_gen_us() is clamped to a minimum of 1 us (server-common.h:388-393).

`tools/server/server-schema.cpp:20-21` · `tools/server/server-common.cpp:66-88` · `tools/server/server-common.h:346-425`

### `Slot log lines carrying benchmark numbers`
**Use:** read stderr  ·  **default:** `all at INFO, visible at default verbosity`

On completion, print_timings() emits `prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)`, ` eval time = ...`, ` total time = %10.2f ms / %5d tokens`, ` graphs reused = %10d` (from llama_perf_context(ctx_tgt).n_reused, 617-619) and, when drafting ran, `draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f` (634-637). Mid-run: print_timings_tg emits `n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s` but ONLY after 100 generated tokens and at most every 3 s (553-559, 573); print_timings_pp emits `prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second` only once total prompt time exceeds 3000 ms (577-585). Release logs `stop processing: n_tokens = %d, truncated = %d` (505). All are prefixed `slot <func>: id %2d | task %d |` (server-common.h:26).

`tools/server/server-context.cpp:600-607` · `tools/server/server-context.cpp:609-611` · `tools/server/server-context.cpp:613-615` · `tools/server/server-context.cpp:617-619` · `tools/server/server-context.cpp:634-637` · `tools/server/server-context.cpp:552-573` · `tools/server/server-context.cpp:576-585` · `tools/server/server-context.cpp:505`

### `Chat templates: source selection`
**Use:** --jinja / --no-jinja ; --chat-template NAME_OR_JINJA ; --chat-template-file PATH  ·  **default:** `use_jinja = TRUE for the server (false for CLI/completion/mtmd)`

Resolution order in common_chat_templates_init (chat.cpp:759-781): explicit override wins; else the GGUF `tokenizer.chat_template`; else the GGUF `tool_use` variant; else the built-in CHATML source. A literal source of "chatml" is treated as empty and falls through to the same fallback chain (776-781). Two templates are silently PATCHED in-place: gpt-oss `<|channel|>` guards (chat.cpp:785-793) and Mistral `[TOOL_CALLS]` guards (795-804). A template parse failure aborts server startup with `chat template parsing error`, `please consider disabling jinja via --no-jinja...`, `for example: --no-jinja --chat-template chatml` (server-context.cpp:1420-1423) and load_model returns false.

`common/common.h:621` · `common/arg.cpp:1394-1398` · `common/arg.cpp:3611-3618` · `common/arg.cpp:3690-3712` · `common/chat.cpp:753-800` · `tools/server/server-context.cpp:1402-1428`

### `Chat template validation at argument-parse time`
**Use:** automatic when --chat-template is given  ·  **default:** `n/a`

With --jinja (the server default) the template is validated by actually rendering a one-message conversation (chat.cpp:628-645); with --no-jinja it must be one of the built-in names accepted by llama_chat_apply_template (chat.cpp:646-650). Failure throws `error: the supplied chat template is not supported: ...`. ORDERING TRAP documented in the help text at arg.cpp:3692-3695: the built-in-only restriction applies unless --jinja appears BEFORE --chat-template on the command line, because arg.cpp:956 reads params.use_jinja as of the end of parsing but the help is written for the sequential case.

`common/arg.cpp:955-961` · `common/chat.cpp:627-651`

### `Reasoning / thinking controls that ride on the template`
**Use:** -rea/--reasoning on|off|auto ; --reasoning-format ; --reasoning-effort ; --reasoning-budget N ; --reasoning-preserve ; --chat-template-kwargs JSON  ·  **default:** `reasoning auto (enable_reasoning = -1); reasoning-format auto; reasoning-budget -1 (unrestricted)`

enable_thinking is the AND of `--reasoning != off` and the template actually supporting it (server-context.cpp:1416-1417). Setting enable_thinking through --chat-template-kwargs is deprecated and warns (arg.cpp:3620-3623). --reasoning-preserve logs `chat template does NOT support preserving reasoning, --reasoning-preserve has no effect` at WARN when the caps say otherwise (1453-1455), and the reverse suggestion at INFO (1450-1452).

`common/arg.cpp:3630-3648` · `common/arg.cpp:3649-3660` · `common/arg.cpp:3661-3667` · `common/arg.cpp:3676-3689` · `common/arg.cpp:3612-3624` · `tools/server/server-context.cpp:1414-1418` · `tools/server/server-context.cpp:1445-1456`

### `Request-parameter clamping vs. rejection`
**Use:** any completion-endpoint JSON field  ·  **default:** `n/a`

CRITICAL DISTINCTION. set_limits() = SOFT: the value is silently clamped with std::max/std::min and the request succeeds with a different value than asked (server-schema.cpp:594-595). set_hard_limits() = the value is validated and a 400 is thrown (596-602). Silently clamped: top_k (0..INT32_MAX), top_p, min_p, xtc_probability, xtc_threshold (0..1), temperature (0..inf), mirostat (0..2). Hard-limited (rejected): n_predict (-1..), n_keep (-1..), n_discard (0..), n_cache_reuse (0..), n_indent (0..), t_max_predict_ms, sse_ping_interval, repeat_last_n, and n_cmpl.

`tools/server/server-schema.h:47-64` · `tools/server/server-schema.cpp:588-607` · `tools/server/server-schema.cpp:63` · `tools/server/server-schema.cpp:88-118`

### `n_cmpl / n (parallel completions) is bounded by -np`
**Use:** `"n_cmpl": N` or the OpenAI alias `"n": N`  ·  **default:** `1`

HARD limit of [1, params_base.n_parallel] (server-schema.cpp:63) — with -np 1, any n > 1 is a 400. n_cmpl > 1 spawns n-1 child tasks (4243-4249) that each need their own free slot; if there are not enough the parent is deferred (2338-2344). Child tasks cannot pin a slot (server-task.h:236 forces id_slot = -1), their prompt cache is dropped on release (server-context.cpp:511-513), and context shift is refused for them (2836-2840).

`tools/server/server-schema.cpp:61-65` · `tools/server/server-context.cpp:4243-4249` · `tools/server/server-context.cpp:2335-2348` · `tools/server/server-task.h:236`

### `HTTP serving parameters`
**Use:** --host, --port, -to/--timeout, --threads-http, --sse-ping-interval, --api-key  ·  **default:** `host 127.0.0.1, port 8080, timeout 3600 s read and write, threads-http -1 (auto), sse-ping-interval 30 s`

--timeout sets read AND write together (arg.cpp:3496-3499). threads-http < 1 becomes max(n_parallel + 4, hardware_concurrency - 1) (server-http.cpp:313-314) with up to 1024 extra dynamic threads (320-322). Before the model is loaded, every non-frontend endpoint returns 503 {"message":"Loading model"} (server-http.cpp:254-273). /health, /v1/health, /models, /v1/models bypass the API-key check (server-http.cpp:196-206, 214-217). Startup prints `srv main: listening on %s` (server.cpp:490) and, on port 8080, a WARN that the default port becomes 9931 in a future release (server.cpp:493-497).

`common/common.h:604` · `common/common.h:617` · `common/common.h:606-609` · `common/arg.cpp:3493-3512` · `tools/server/server-http.cpp:311-323` · `tools/server/server-http.cpp:254-273`

### `Task queue and deferral`
**Use:** observable via /slots and /metrics requests_deferred  ·  **default:** `n/a`

Deferred tasks are woken by slot release: pop_deferred_task prefers a task whose id_slot matches the freed slot, otherwise takes the head (server-queue.cpp:80-96). /metrics exposes the depth as llamacpp:requests_deferred (server-task.cpp:1578-1580, fed from queue_tasks_deferred_size at server-context.cpp:2450). /metrics and /slots requests are posted front-of-queue (front = true) so they cannot starve behind generation.

`tools/server/server-queue.cpp:23-61` · `tools/server/server-queue.cpp:63-69` · `tools/server/server-queue.cpp:76-99` · `tools/server/server-context.cpp:2322-2333` · `tools/server/server-task.cpp:1575-1580`

### `Sleep on idle`
**Use:** --sleep-idle-seconds N (-1 disables)  ·  **default:** `-1 (disabled)`

0 or < -1 throws at parse time (arg.cpp:3757-3759). Entering/leaving is logged at INFO: `server is entering sleeping state` / `server is exiting sleeping state` (901-907). /props and /health remain answerable while sleeping (server-context.cpp:4455-4460, 4582-4587) and report is_sleeping.

`common/common.h:635` · `common/arg.cpp:3752-3762` · `tools/server/server-context.cpp:901-908`

**What this area CANNOT do.** Things this area CANNOT do, each with the line that forecloses it: 1. Cannot set slot count to 0, and cannot change it after startup. `-np 0` throws (common/arg.cpp:2537-2539); slots are built once in the init loop at tools/server/server-context.cpp:1224-1226 and there is no endpoint that resizes them. 2. Cannot get per-slot contexts that do not divide the total. Without kv_unified, n_ctx_seq = n_ctx / n_seq_max padded to 256 and n_ctx is then rewritten to n_ctx_seq * n_seq_max (src/llama-context.cpp:292-302) — the requested -c is silently reduced, with only a WARN. 3. Cannot exceed the model's training context per slot. tools/server/server-context.cpp:1202-1206 caps n_ctx_slot to n_ctx_train unconditionally. 4. Cannot truncate an over-long prompt. tools/server/server-context.cpp:3118-3125 returns ERROR_TYPE_EXCEED_CONTEXT_SIZE instead; the old truncate-the-middle behaviour is gone. 5. Cannot run past n_ctx by default. ctx_shift defaults to false (common/common.h:561) and generation is cut with stop_type LIMIT at tools/server/server-context.cpp:1820-1827, logged only at DEBUG. 6. Cannot use context shift or --cache-reuse with a multimodal model. Both are force-disabled at tools/server/server-context.cpp:1165-1174, and pre_decode() would GGML_ABORT("not supported by multimodal") at 2842-2846 if reached. 7. Cannot use context shift for an n_cmpl > 1 request. tools/server/server-context.cpp:2836-2840 sends an error and releases the slot. 8. Cannot request more parallel completions than there are slots. Hard limit [1, n_parallel] at tools/server/server-schema.cpp:63. 9. Cannot pin a child task to a slot. tools/server/server-task.h:236 forces id_slot = -1 on every cloned child. 10. Cannot get an error for an out-of-range id_slot — it wraps modulo the slot count at tools/server/server-context.cpp:1464. 11. Cannot actually change anything through POST /props. The handler body is empty and returns success unconditionally (tools/server/server-context.cpp:4638-4642). 12. Cannot read prompt or generated text from /slots in a normal run. slot.to_json(only_metrics = slots_debug == 0) at tools/server/server-context.cpp:2430 with slots_debug read from LLAMA_SERVER_SLOTS_DEBUG at 1283-1289. 13. Cannot save/restore/erase slot KV without a directory. POST /slots is ERROR_TYPE_NOT_SUPPORTED when slot_save_path is empty (tools/server/server-context.cpp:4549-4552). 14. Cannot get a per-request HTTP access log. The httplib logger registration is commented out at tools/server/server-http.cpp:127 ("this is too spamy"); only SRV_DBG("response: %s") at 47 exists, and it is DEBUG level. 15. Cannot read /metrics rate gauges twice for the same window. The buckets are cleared on every /metrics scrape (tools/server/server-context.cpp:4477 and 2444-2446). 16. Cannot see generation counted in /metrics until the request finishes. metrics.predict is only added in metrics_on_prediction, invoked from the slot's callback_on_reset on release (tools/server/server-context.cpp:1268-1273, 4037-4043). 17. Cannot get a fully-cached prompt to cost zero decode. [TAG_PROMPT_LOGITS] forces n_past-- at tools/server/server-context.cpp:3313-3318, so at least one token is always re-evaluated. 18. Cannot batch two slots with different task types, different input-embedding sizes, or different LoRA sets. tools/server/server-context.cpp:407-413. 19. Cannot rely on -dt/--defrag-thold: it is accepted and does nothing but print a deprecation warning (common/arg.cpp:2522-2531). 20. Cannot make the prompt cache survive an allocation failure at full size — limit_size is permanently cut to 0.4x on bad_alloc (tools/server/server-task.cpp:1762-1768). 21. Cannot enable --cache-idle-slots without a RAM cache; it is auto-disabled when --cache-ram 0 (tools/server/server-context.cpp:1374-1377). 22. Cannot see the prompt-cache diagnostics at default verbosity. Every server_prompt_cache line is SRV_TRC/SRV_WRN in a TRC-gated region and the default threshold is LOG_LEVEL_INFO = 3 (common/common.h:524, common/log.h:26, common/log.cpp:29).

**Exists but unused in our profile.** A profile running `--spec-type ngram-mod -ctk q4_0 -ctv q4_0 -ngl auto --fit on -np 1` never touches the following, even though they exist: - The kv_unified path in its entirety. Explicit `-np 1` takes the else branch at tools/server/server.cpp:151-155, so kv_unified stays false (common/common.h:563). Consequences: n_ctx_seq == n_ctx (src/llama-context.cpp:290 not taken, 292 taken with n_seq_max=1 so it is a no-op); try_clear_idle_slots is a permanent no-op (tools/server/server-context.cpp:1600-1602); and [TAG_IDLE_SLOT_CLEAR] never wipes an idle slot's VRAM KV (tools/server/server-context.cpp:2365-2369). Switching to `-np` auto (or omitting -np) flips all three at once — this is the single largest behavioural difference in this area between "-np 1" and "no -np". - Multi-slot scheduling. With one slot, LRU selection (tools/server/server-context.cpp:1553-1574) always returns that slot, the LCP loop (1505-1551) has one candidate, and continuous batching across slots (3022-3037) has nothing to interleave. `n_busy_slots_per_decode` (tools/server/server-task.cpp:1556-1559) is therefore always 0 or 1. NOTE that -sps is still live even at -np 1: it decides between the LCP branch and the LRU branch, and both can set update_cache, so the RAM prompt-cache save/load round-trip IS exercised. - `n_cmpl` / `n` > 1 and the whole parent/child slot machinery (tools/server/server-context.cpp:2335-2348, 4243-4249) — hard-limited to 1 by server-schema.cpp:63 at -np 1. - /metrics and its Prometheus counters and gauges: disabled by default (common/common.h:655) and not enabled by this profile. - POST /props (disabled by default, common/common.h:654) and POST /slots save/restore/erase (needs --slot-save-path, common/common.h:674). - --cache-reuse: default 0 (common/common.h:610), so the KV-shifting chunk reuse loop at tools/server/server-context.cpp:3149-3195 never runs; only the plain common-prefix path at 3126 does. - Context shift: default off (common/common.h:561), so tools/server/server-context.cpp:2818-2879 is dead and --keep / n_discard have no effect. - Multimodal (mtmd) auto-disable interactions at tools/server/server-context.cpp:1165-1174 and the mtmd chunk path at 3387-3435. - Router mode (--models-dir / --models-preset / --models-max) and every proxy route wiring at tools/server/server.cpp:180-232. - LoRA / aLoRA slot interactions (tools/server/server-context.cpp:1650-1714, 3355-3365) and the per-request `lora` field. - --sleep-idle-seconds (default -1, common/common.h:635). - Router/child process notification and the MCP proxy. Also worth flagging as active but easy to miss under this profile: --cache-ram defaults to 8192 MiB of HOST RAM for the cross-request prompt cache (common/common.h:615) and --cache-idle-slots is on (common/common.h:612), so every new task does a full llama_state_seq_get_data of the idle slot into RAM before the new prompt starts (tools/server/server-context.cpp:2355-2363) — a real, unlogged-at-INFO cost that lands inside the next request's prompt_ms window, since t_start is only set once the slot enters PROCESSING_PROMPT (server-context.cpp:3053) which happens after get_available_slot returned.

## Sampling and constrained decoding (sampler chain + order, GBNF grammar, --reasoning-budget, top-k/top-p internals, host vs device) — llama.cpp build 10499, commit 1deefcca3

### `Sampler chain construction and its fixed order`
**Use:** --samplers "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature" (';'-separated) or --sampler-seq "edsktypmxt" (one char each). Server: no per-request field for order.  ·  **default:** `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`

The user-orderable list is only the middle of the chain. logit-bias is ALWAYS prepended before the loop (common/sampling.cpp:331-344) and `dist` (or `adaptive_p`) is ALWAYS appended after it (common/sampling.cpp:400-406). So the real chain is: logit_bias -> [your order] -> dist. Grammar and reasoning-budget are NOT chain members at all — they are separate samplers applied by common_sampler_sample (common/sampling.cpp:646-652). Names accept aliases: kebab (top-k), no-dash (topk), plus nucleus/temp/typ (common/sampling.cpp:952-968). An unmatched name is silently dropped with only a LOG_WRN (common/sampling.cpp:985) — a typo'd sampler is removed from the chain, not an error.

`common/common.h:260-270` · `common/sampling.cpp:350-399` · `common/sampling.cpp:417-419` · `common/arg.cpp:1974-1996` · `common/sampling.cpp:895-1019`

### `Disabled samplers become zero-cost `empty` stubs, not omissions`
**Use:** Set the disabling value (top_k<=0, top_p>=1.0, min_p<=0, typ_p>=1.0, top_n_sigma<=0, xtc_probability<=0 or xtc_threshold>0.5, temp==1.0 && dynatemp_range<=0, dry_multiplier==0, penalty_last_n==0 or all three penalties neutral).

llama_sampler_init_empty("?name") is substituted. This matters for device sampling: the empty sampler DOES implement backend_init (src/llama-sampler.cpp:512), so a disabled sampler does not break the backend prefix. Chain printout marks them '?name'; backend-capable ones print '+name' when offloaded and '-name' when the device lacks an op (src/llama-sampler.cpp:534-549). Note temp is only 'empty' at exactly 1.0 — temp 0.8 (the default) builds a real temp-ext sampler.

`src/llama-sampler.cpp:1519-1524` · `src/llama-sampler.cpp:1719-1724` · `src/llama-sampler.cpp:1882-1887` · `src/llama-sampler.cpp:1994-1999` · `src/llama-sampler.cpp:2104-2109` · `src/llama-sampler.cpp:2307-2312` · `src/llama-sampler.cpp:2416-2421` · `src/llama-sampler.cpp:3203-3214` · `src/llama-sampler.cpp:3300-3305` · `src/llama-sampler.cpp:3639-3649` · `src/llama-sampler.cpp:4043-4051` · `src/llama-sampler.cpp:505-520`

### `top-k: partial sort with a 128-element fast path and a 128-bucket logit histogram above it`
**Use:** --top-k N (0 or negative = disabled)  ·  **default:** `40`

CLAMP: k = min(k, cur_p->size) (src/llama-sampler.cpp:330). FAST PATH: llama_token_data_array_partial_sort_inplace uses plain std::partial_sort when npartial <= 128 (src/llama-sampler.cpp:198-205); above 128 it bucket-sorts logits into 128 buckets spanning [-10, 10] with out-of-range values clamped into the end buckets (src/llama-sampler.cpp:140-159). With the default k=40 the cheap std::partial_sort path is always taken. top-k is the only filter that does NOT honour min_keep. If cur_p is already flagged sorted, top_k just truncates size with no sort at all (src/llama-sampler.cpp:333-337). Env var: LLAMA_ARG_TOP_K.

`src/llama-sampler.cpp:321-338` · `src/llama-sampler.cpp:193-215` · `src/llama-sampler.cpp:135-190` · `src/llama-sampler.cpp:1449-1452` · `common/common.h:229` · `common/arg.cpp:2013-2020`

### `top-p: adaptive two-stage sort that avoids sorting the full vocab`
**Use:** --top-p N (1.0 = disabled); server field top_p  ·  **default:** `0.95`

FAST PATH: if the candidate array is unsorted and larger than 1024, it first partial-sorts only the top 256 into a scratch buffer (ctx->buf_sort) and walks the CDF there; only if the cumulative mass has not reached p by element 255 does it re-sort the whole array (src/llama-sampler.cpp:1564-1567 then 1588-1592). Under 1024 candidates it sorts in place. Because top-k runs BEFORE top-p in the default order, by the time top-p runs there are only 40 candidates, so neither the 1024 branch nor the 256 heuristic ever fires in a default chain. min_keep is honoured (src/llama-sampler.cpp:1582). Note softmax is computed with do_sort=false (src/llama-sampler.cpp:1556), so probabilities are computed over the pre-truncation set.

`src/llama-sampler.cpp:1549-1602` · `src/llama-sampler.cpp:1552-1554` · `src/llama-sampler.cpp:1563-1571` · `src/llama-sampler.cpp:1586-1592` · `common/common.h:230` · `common/arg.cpp:2021-2028`

### `min-p: unsorted log-domain threshold first, sorted scan as fallback`
**Use:** --min-p N (0.0 = disabled); server field min_p  ·  **default:** `0.05`

FAST PATH: when the array is unsorted it computes min_logit = max_logit + logf(p) and filters in one pass with no sort at all (src/llama-sampler.cpp:1762-1772). That path is abandoned and a full sort is done only if the filtered set is empty or smaller than min_keep (src/llama-sampler.cpp:1775, 1783-1787). Since top-k/top-p have already set sorted=true in a default chain, the sorted branch is what actually runs.

`src/llama-sampler.cpp:1749-1801` · `src/llama-sampler.cpp:1759-1780` · `common/common.h:231`

### `temperature and dynamic temperature (entropy-mapped)`
**Use:** --temp N, --dynatemp-range N, --dynatemp-exp N  ·  **default:** `temp 0.80, dynatemp_range 0.00 (disabled), dynatemp_exponent 1.00`

CLAMP: --temp is floored at 0.0 by the arg parser (common/arg.cpp:2009); the server clamps to [0, +inf] silently (tools/server/server-schema.cpp:116-118). temp <= 0 is NOT a divide — it is a greedy rewrite that sets every logit except the argmax to -INFINITY (src/llama-sampler.cpp:270-286). With dynatemp_range > 0 the sampler force-sorts and computes a full softmax + entropy over all remaining candidates every token (src/llama-sampler.cpp:2149-2158), which is strictly more work than plain temp. dyn_temp = min_temp + (max_temp-min_temp) * normalized_entropy^exponent, min_temp floored at 0 (src/llama-sampler.cpp:2136-2164).

`src/llama-sampler.cpp:265-291` · `src/llama-sampler.cpp:2133-2202` · `src/llama-sampler.cpp:2307-2312` · `common/common.h:235-237` · `common/arg.cpp:2004-2012` · `common/arg.cpp:2196-2209`

### `GBNF grammar: state lives in llama_grammar, a rules vector plus a live stack set`
**Use:** --grammar GBNF | --grammar-file FNAME | -j/--json-schema SCHEMA | -jf/--json-schema-file FILE. Server: "grammar", "json_schema", "grammar_lazy", "grammar_triggers".  ·  **default:** `none (COMMON_GRAMMAR_TYPE_NONE)`

State is: the immutable parsed `rules` (vector of vectors of llama_grammar_element), a mutable `stacks` (vector of pushdown stacks, each a vector of raw pointers INTO rules), a 4-byte partial_utf8 carry, and — for lazy grammars — trigger_buffer / trigger_buffer_positions. Because stacks hold pointers into rules, rules must be moved not copied (src/llama-grammar.cpp:1298-1300) and cloning does an O(stacks x rules) pointer-fixup scan (src/llama-grammar.cpp:1337-1348) — cloning a grammar is quadratic in grammar size, and the server clones the whole sampler on every speculative verification step (tools/server/server-context.cpp:3822). llama_sampler_grammar_reset does NOT rewind state — it re-parses the grammar string from scratch and frees the old object (src/llama-sampler.cpp:2700-2718).

`src/llama-grammar.h:126-151` · `src/llama-grammar.cpp:1209-1313` · `src/llama-grammar.cpp:1470-1523` · `src/llama-sampler.cpp:2660-2667` · `common/arg.cpp:2257-2294` · `common/common.h:187-220`

### `GBNF cost model: rejection sampling by default, full-vocab mask only on a miss`
**Use:** automatic; common_sampler_sample(gsmpl, ctx, idx, grammar_first=false)  ·  **default:** `grammar_first = false`

Default path: run the whole chain, then test ONLY the sampled token against the grammar via a 1-element llama_token_data_array (common/sampling.cpp:662-667). If it passes, cost is one llama_grammar_apply_impl over one candidate. If it fails, logits are re-fetched for the FULL vocab, rbudget + grammar + the entire chain are re-run (common/sampling.cpp:675-687) — so a grammar miss costs a full-vocab grammar mask plus a second pass of every sampler. There is NO precomputed token mask and no cache: each apply decodes UTF-8 per candidate (src/llama-grammar.cpp:1385) and runs the recursive llama_grammar_reject_candidates over every stack. Token pieces themselves are cached at vocab load (src/llama-vocab.cpp:3021, 3698), so token_to_piece is O(1). grammar_first=true is passed by exactly one caller — the maximal-coupling speculative verifier (common/sampling.cpp:739); nothing in tools/ or examples/ ever sets it.

`common/sampling.cpp:608-690` · `common/sampling.cpp:648-658` · `common/sampling.cpp:660-671` · `common/sampling.cpp:673-689` · `src/llama-grammar.cpp:1353-1394` · `src/llama-grammar.cpp:1055-1124` · `src/llama-vocab.cpp:3021` · `src/llama-vocab.cpp:3697-3698`

### `Lazy grammars and triggers (tool calling)`
**Use:** grammar_lazy=true plus grammar_triggers of type TOKEN / WORD / PATTERN / PATTERN_FULL (server fields)  ·  **default:** `grammar_lazy = false`

While awaiting_trigger, llama_grammar_apply_impl returns immediately without touching logits (src/llama-grammar.cpp:1356-1358) — a lazy grammar is free until it fires. WORD triggers are regex-escaped, PATTERN_FULL is anchored with ^...$ (common/sampling.cpp:230-252). Each pattern compiles a std::regex at init (src/llama-grammar.cpp:1295) and is re-matched against the whole accumulated trigger_buffer on EVERY accepted token (src/llama-grammar.cpp:1413-1414) — cost grows with output length until the trigger fires. When a pattern fires, the buffered tokens overlapping the match are replayed into the grammar, with partial-token slicing (src/llama-grammar.cpp:1419-1429). Server throws if grammar_lazy is set with no triggers (tools/server/server-schema.cpp:375-377).

`common/sampling.cpp:226-278` · `common/common.h:143-152` · `src/llama-grammar.cpp:1396-1441` · `src/llama-grammar.cpp:1291-1296` · `tools/server/server-schema.cpp:283` · `tools/server/server-schema.cpp:375-377`

### `Grammar prefill from the generation prompt`
**Use:** server field "generation_prompt"; applied only for OUTPUT_FORMAT and TOOL_CALLS grammar types  ·  **default:** `empty`

Tokens of the generation prompt are fed to the grammar so it advances past text the chat template already emitted. Explicitly NOT applied to user-supplied --grammar (common_grammar_needs_prefill returns true only for OUTPUT_FORMAT/TOOL_CALLS, common/common.h:217-220) and not applied to lazy grammars (common/sampling.cpp:303). The same prefill tokens are also fed to the reasoning-budget sampler unconditionally (common/sampling.cpp:325-328). A leading whitespace piece produced by the tokenizer for the first special token is dropped (common/sampling.cpp:291-293).

`common/sampling.cpp:284-314` · `common/common.h:217-220` · `common/sampling.cpp:303` · `tools/server/server-schema.cpp:306`

### `--reasoning-budget: a five-state token-forcing sampler outside the chain`
**Use:** --reasoning-budget N (-1 unrestricted, 0 immediate end, N>0 budget) plus --reasoning-budget-message MESSAGE. Server per-request: reasoning_budget_tokens / thinking_budget_tokens, reasoning_budget_message, reasoning_control.  ·  **default:** `-1 (disabled)`

States IDLE -> COUNTING -> (WAITING_UTF8) -> FORCING -> DONE. Detection of the start/end tags is an Aho-Corasick automaton over TOKEN ids, not text (common/reasoning-budget.cpp:14-50). apply() is a no-op in every state except FORCING, where it sets every logit but the one forced token to -INFINITY (common/reasoning-budget.cpp:166-186). It is applied FIRST, before the grammar and before the chain (common/sampling.cpp:646). CLAMP/SURPRISE: budget -1 is stored as INT_MAX, not as 'skip' (common/sampling.cpp:323), and the sampler is still constructed when a lazy grammar is present even with an unlimited budget, because the lazy grammar depends on it for thinking-block suppression (common/sampling.cpp:317). Budget 0 forces immediately on the start tag (common/reasoning-budget.cpp:85-89). It RE-ARMS on a second start tag, giving each <think> block a fresh budget (common/reasoning-budget.cpp:146-161). Budget exhaustion waits for a UTF-8 boundary before forcing so it cannot split a multibyte char (common/reasoning-budget.cpp:104-129). --reasoning-budget-message is prepended to the FIRST end tag to form the forced sequence (tools/server/server-schema.cpp:419-427).

`common/arg.cpp:3661-3675` · `common/common.h:286-293` · `common/reasoning-budget.h:10-52` · `common/reasoning-budget.cpp:74-196` · `common/sampling.cpp:316-329` · `common/sampling.cpp:646` · `tools/server/server-common.cpp:1352-1367` · `tools/server/server-schema.cpp:380-428`

### `Runtime 'end reasoning now' control`
**Use:** POST a control task with action "reasoning_end" for a live completion id; requires reasoning_control=true on the original request  ·  **default:** `reasoning_control = false`

reasoning_control=true forces the budget sampler to be created even with budget -1 (common/sampling.cpp:317) so it can be forced later. The transition only succeeds from COUNTING — IDLE, FORCING, WAITING_UTF8 and DONE are all left untouched and it returns false (common/reasoning-budget.cpp:300-302). The server rejects the control action outright if reasoning_control was not set (tools/server/server-context.cpp:2400-2405).

`common/common.h:293` · `common/sampling.cpp:821-827` · `common/reasoning-budget.cpp:291-310` · `tools/server/server-context.cpp:2398-2408` · `tools/server/server-schema.cpp:380-381`

### `Reasoning budget gates whether a lazy grammar is applied at all`
**Use:** automatic when both grammar_lazy and a reasoning-budget sampler exist

grammar_should_apply() returns false while the budget sampler is in COUNTING / FORCING / WAITING_UTF8 — i.e. a lazy grammar is suppressed for the whole duration of the thinking block and only re-enabled at IDLE or DONE (common/sampling.cpp:469-473). When the budget transitions to DONE while the grammar was suppressed, the end sequence is replayed into the grammar so a trigger embedded in the end tag still fires (common/sampling.cpp:490-499).

`common/sampling.cpp:462-475` · `common/sampling.cpp:485-504`

### `Device (backend) sampling: only a PREFIX of the chain can be offloaded`
**Use:** -bs / --backend-sampling (env LLAMA_ARG_BACKEND_SAMPLING); server per-request field backend_sampling  ·  **default:** `false (disabled)`

llama_sampler_chain_backend_init walks the chain and stops the backend prefix at the FIRST sampler that has no backend_init or whose op-support probe fails (src/llama-sampler.cpp:746-765). Everything from that point on runs on the host. Samplers WITH a device implementation: empty, greedy, dist, top_k, top_p, min_p, temp, temp_ext, penalties, logit_bias. Samplers with NO device implementation (backend_init = nullptr): typical, xtc, mirostat, mirostat_v2, grammar, top_n_sigma, dry, adaptive_p, infill. So enabling typ_p, xtc, dry, top_n_sigma or adaptive-p pushes everything after it back to the CPU. Support is decided by building a probe graph on a 1,048,576-element logits tensor and asking the device about every op (src/llama-sampler.cpp:583-661). llama_sampler_chain_apply on the host SKIPS the samplers already marked is_backend (src/llama-sampler.cpp:686-693).

`common/arg.cpp:2295-2301` · `common/common.h:295` · `src/llama-sampler.cpp:733-771` · `src/llama-sampler.cpp:681-701` · `src/llama-context.cpp:1209-1260` · `tools/server/server-context.cpp:1734-1744` · `src/llama-sampler.cpp:638-661`

### `Device implementations of top-k / top-p / min-p as ggml graphs`
**Use:** automatic under -bs

top-k on device is a single ggml_top_k + ggml_get_rows (src/llama-sampler.cpp:1486-1499). top-p is a full ggml_argsort(DESC) over the whole vocab, then soft_max, cumsum, scale_bias, step, sum, clamp, set_rows, log, add (src/llama-sampler.cpp:1645-1698) — a full-vocab sort every token, unlike the host's 256-element heuristic. min-p is argmax + scale_bias + sub + step + log + add, no sort (src/llama-sampler.cpp:1835-1861). CUDA supports ARGSORT/TOP_K on arbitrary row lengths only when built against CUB; without CUB it is capped at ne0 <= 1024 (ggml/src/ggml-cuda/ggml-cuda.cu:5249-5253), which a 151k vocab would fail. This build has CUDA 13.3 (build-dflash2/CMakeCache.txt) and CUB is defined for CUDART >= 11070 (ggml/src/ggml-cuda/common.cuh:110-112), so the cap does not bite here.

`src/llama-sampler.cpp:1477-1502` · `src/llama-sampler.cpp:1627-1702` · `src/llama-sampler.cpp:1826-1865` · `src/llama-sampler.cpp:2036-2088` · `ggml/src/ggml-cuda/ggml-cuda.cu:5247-5253` · `ggml/src/ggml-cuda/common.cuh:110-112`

### `Reading back what the device sampled`
**Use:** llama_get_sampled_token_ith / _probs_ith / _logits_ith / _candidates_ith and their *_count_ith

common_sampler::set_logits prefers device-sampled probs, then device-sampled logits, then falls back to the full host logits array of n_vocab entries (common/sampling.cpp:146-165). If llama_get_sampled_token_ith returns a real token, common_sampler_sample returns it immediately and runs NO host sampler, asserting that neither grammar nor rbudget exists (common/sampling.cpp:628-642). Every getter calls ctx->synchronize() first (src/llama-context.cpp:3826 etc.).

`common/sampling.cpp:136-168` · `common/sampling.cpp:623-643` · `src/llama-context.cpp:3825-3865`

### `Model GGUF metadata can override your sampling defaults`
**Use:** nothing to type — happens at common_init_from_params time  ·  **default:** `model metadata wins over the built-in defaults, loses to explicitly-passed flags`

Keys general.sampling.{sequence,top_k,top_p,min_p,xtc_probability,xtc_threshold,temp,penalty_last_n,penalty_repeat,mirostat,mirostat_tau,mirostat_eta} are read from the GGUF and overwrite params.sampling unless the corresponding user_sampling_config bit was set by an explicit flag (common/common.cpp:1216-1246). BIG GOTCHA: --sampler-seq / --sampling-seq does NOT set the SAMPLERS bit (common/arg.cpp:1990-1996) while --samplers does (common/arg.cpp:1980) — so a model carrying general.sampling.sequence silently overrides an order you gave with --sampler-seq. Not overridable this way: typ_p, top_n_sigma, all dry_*, dynatemp_*, presence/frequency penalties, adaptive_*.

`common/common.cpp:1212-1270` · `common/common.cpp:1338` · `src/llama-model.cpp:2785-2796` · `common/common.h:256` · `common/arg.cpp:1978-1980` · `common/arg.cpp:1990-1996`

### `Server clamps most sampling fields SILENTLY; a few throw`
**Use:** JSON fields on /completion and /v1/chat/completions

set_limits => value is silently clamped with max(min, min(max, v)) (tools/server/server-schema.cpp:595). set_hard_limits => std::invalid_argument is thrown (tools/server/server-schema.cpp:597-601). SILENTLY CLAMPED: top_k [0, INT32_MAX], top_p [0,1], min_p [0,1], xtc_probability [0,1], xtc_threshold [0,1], temperature [0, +inf], mirostat [0,2], adaptive_target [-FLT_MAX, 1.0]. HARD (throws): repeat_last_n [0,INT32_MAX], dry_allowed_length, dry_penalty_last_n, adaptive_decay [0.0,0.99], min_keep [0,INT32_MAX], reasoning_budget_tokens [-1,INT32_MAX]. dry_base has a custom handler: anything < 1.0 is silently replaced by the server's base default rather than clamped (tools/server/server-schema.cpp:144-147). typical_p has no limits at all — the comment literally asks what the valid range is (tools/server/server-schema.cpp:113).

`tools/server/server-schema.cpp:588-607` · `tools/server/server-schema.cpp:89-191` · `tools/server/server-schema.h:48-66`

### `min_keep floor for top-p / min-p / xtc / typical`
**Use:** server JSON field "min_keep" only  ·  **default:** `0 (disabled)`

There is NO --min-keep command-line flag anywhere in common/arg.cpp. It is reachable only through the server request body. top_k and top_n_sigma ignore it entirely.

`common/common.h:228` · `common/sampling.cpp:366` · `common/sampling.cpp:372` · `common/sampling.cpp:375` · `common/sampling.cpp:378` · `tools/server/server-schema.cpp:183-185`

### `logit bias, --ignore-eos and model suppress tokens`
**Use:** -l/--logit-bias TOKEN(+/-)BIAS, --ignore-eos; server "logit_bias" (array of [token,bias] or object; bias=false means ban) and "ignore_eos"  ·  **default:** `no biases; ignore_eos = false`

--ignore-eos is implemented purely as -INFINITY biases on every EOG token, precomputed once at model load into logit_bias_eog and appended to the active bias list (common/common.cpp:1345-1358). It is silently downgraded to false if the vocab has no EOS (common/common.cpp:1340-1343). Independently, llama_vocab_get_suppress_tokens contributes -INFINITY biases that the user cannot turn off (common/sampling.cpp:335-339). The apply has a fast path that indexes cur_p->data[token] directly when the array is still identity-ordered, falling back to an O(size x n_bias) scan for the rest (src/llama-sampler.cpp:3912-3932).

`common/arg.cpp:1997-2003` · `common/arg.cpp:2235-2256` · `common/sampling.cpp:331-344` · `common/common.cpp:1340-1358` · `src/llama-sampler.cpp:3902-3933` · `tools/server/server-schema.cpp:430-480`

### `Penalties, DRY, XTC, typical, top-n-sigma, mirostat, adaptive-p (host-only detail)`
**Use:** --repeat-last-n/--repeat-penalty/--presence-penalty/--frequency-penalty; --dry-multiplier/--dry-base/--dry-allowed-length/--dry-penalty-last-n/--dry-sequence-breaker; --xtc-probability/--xtc-threshold; --typical; --top-nsigma; --mirostat/--mirostat-lr/--mirostat-ent; --adaptive-target/--adaptive-decay  ·  **default:** `penalty_last_n 64, penalty_repeat 1.00 (off), freq 0.00, present 0.00; dry_multiplier 0.00 (off), dry_base 1.75, dry_allowed_length 2, dry_penalty_last_n 64; xtc_probability 0.00 (off), xtc_threshold 0.10; typ_p 1.00 (off); top_n_sigma -1.00 (off); mirostat 0 (off), tau 5.00, eta 0.10; adaptive_target -1.0 (off), adaptive_decay 0.90`

CLAMPS: penalty_last_n floored at 0 and n_prev raised to match it (common/arg.cpp:2075, src/llama-sampler.cpp:3209); repeat/presence/frequency penalties must be finite and repeat > 0 or common_sampler_init THROWS (common/sampling.cpp:196-206) — an invalid value fails slot creation, not the request parse. --dry-base silently IGNORES any value < 1.0, leaving the old value (common/arg.cpp:2127-2130). DRY sequence breakers are truncated to 40 characters and expanded to at most 20-token sequences (src/llama-sampler.cpp:3642-3643, 3665-3670); the first --dry-sequence-breaker wipes the four defaults {"\n", ":", "\"", "*"} via a function-local static (common/arg.cpp:2162-2167) — meaning that static persists across repeated parses in one process. XTC advances its own RNG on every call before the probability roll and mutates cur_p->data by POINTER ADVANCE (src/llama-sampler.cpp:2372-2373), not by copying. penalties is the only sampler that clears cur_p->sorted (src/llama-sampler.cpp:2979). mirostat's m is hard-coded to 100 (common/sampling.cpp:409). adaptive_decay is clamped to [0.0, 0.99] inside the sampler regardless of caller (src/llama-sampler.cpp:3866).

`common/common.h:238-251` · `common/arg.cpp:2067-2234` · `src/llama-sampler.cpp:2950-2980` · `src/llama-sampler.cpp:3386-3397` · `src/llama-sampler.cpp:3639-3672` · `src/llama-sampler.cpp:2344-2375` · `src/llama-sampler.cpp:3237-3274` · `src/llama-sampler.cpp:3860-3881`

### `Final selection: dist (stochastic) or adaptive_p, and the greedy special case`
**Use:** automatic — dist is appended unless you explicitly list adaptive_p in --samplers  ·  **default:** `dist with seed = params.sampling.seed (LLAMA_DEFAULT_SEED = random)`

dist does its own softmax and inverse-CDF walk in a single fused pass and does NOT sort (src/llama-sampler.cpp:1186-1214) — so after sampling, cur_p may be unsorted; use common_sampler_get_candidates(gsmpl, true) to force a sort, which also re-locates the selected index (common/sampling.cpp:836-853). With exactly one candidate it still burns one RNG draw to stay bit-aligned with the device path (src/llama-sampler.cpp:1163-1168). Greedy generation is not a separate sampler here — temp <= 0 turns the temp sampler into an argmax mask (src/llama-sampler.cpp:270-286) and dist then trivially picks it. LLAMA_DEFAULT_SEED resolves to std::random_device, or to the system clock if random_device reports zero entropy (src/llama-sampler.cpp:340-351).

`common/sampling.cpp:400-406` · `src/llama-sampler.cpp:1150-1223` · `src/llama-sampler.cpp:340-351` · `src/llama-sampler.cpp:1399-1415` · `common/arg.cpp:1983-1989`

### `Speculative verification: greedy match, and (DFlash2 only) maximal-coupling residual sampling`
**Use:** automatic; common_sampler_sample_and_accept_n with or without a dists vector

This is the part PR #27342 (DFlash2) added to this file. Default verification is plain greedy: sample the target token, accept the draft only if identical, stop at the first mismatch (common/sampling.cpp:699-709). The stochastic path is entered ONLY when the drafter supplied per-position distributions AND temp > 0 AND the KV can be rolled back (tools/server/server-context.cpp:3828-3830); it then does maximal coupling — accept draft[i] if uniform*q <= p, else resample from the positive residual max(0, p-q) (common/sampling.cpp:763-780). It forces grammar_first=true for the target sample (common/sampling.cpp:739), i.e. the grammar mask over the full candidate set on every drafted position. Its RNG is deliberately decorrelated: seed = llama_sampler_get_seed(chain) ^ 0x9e3779b9 (common/sampling.cpp:434). Only the DFlash selector fills dists (common/speculative.cpp:1238-1258).

`common/sampling.cpp:692-720` · `common/sampling.cpp:722-793` · `common/sampling.cpp:433-434` · `common/sampling.cpp:126-127` · `common/speculative.cpp:1215-1279` · `tools/server/server-context.cpp:3828-3831`

### `n_probs / logprobs and its effect on the pipeline`
**Use:** server fields n_probs (alias logprobs) and post_sampling_probs; OAI logprobs+top_logprobs  ·  **default:** `n_probs 0, post_sampling_probs false; OAI logprobs=true defaults top_logprobs to 20`

n_probs has NO limits declared (tools/server/server-schema.cpp:179-181) — it is only min'd against the number of surviving candidates at read time (tools/server/server-context.cpp:1905, 1933). post_sampling_probs=false (the default) requires pre-sampling logits, and that combination FORCE-DISABLES backend sampling for the slot (tools/server/server-context.cpp:1732-1737). With post_sampling_probs=true, zero-probability entries truncate the list early (tools/server/server-context.cpp:1920-1922). OAI rejects logprobs together with tools+stream (tools/server/server-common.cpp:1372-1374).

`common/common.h:227` · `tools/server/server-schema.cpp:179-191` · `tools/server/server-context.cpp:1732-1737` · `tools/server/server-context.cpp:1899-1953` · `tools/server/server-common.cpp:1369-1378`

### `Sampler cloning / copying, and the ring buffer of accepted tokens`
**Use:** common_sampler_clone / common_sampler_copy / common_sampler_reset; --keep-alive-ish n_prev is internal  ·  **default:** `n_prev 64, capacity = max(32, n_prev)`

prev ring capacity is max(32, n_prev) and --repeat-last-n raises n_prev to match penalty_last_n (common/arg.cpp:2075). common_sampler_copy asserts that src and dst agree on whether grammar and rbudget exist (common/sampling.cpp:538-539) and must re-point cur_p.data into dst's own buffer (common/sampling.cpp:549) — a consequence of XTC's pointer-advance trick. common_sampler_reset resets the chain and re-seeds the speculative RNG but does NOT reset the grammar or the reasoning-budget sampler (common/sampling.cpp:129-134).

`common/sampling.cpp:519-553` · `common/sampling.cpp:440` · `common/common.h:226` · `src/llama-sampler.cpp:4299-4320` · `src/llama-sampler.cpp:838-853`

**What this area CANNOT do.** A grammar and device sampling are mutually exclusive, and the code enforces it twice: common_sampler_init prints "backend sampling is not compatible with grammar, disabling" and clears params.backend_sampling (common/sampling.cpp:421-425), and common_sampler_sample asserts the same at runtime (common/sampling.cpp:631). The reasoning-budget sampler gets identical treatment (common/sampling.cpp:427-431 and the assert at common/sampling.cpp:632). Neither has any device implementation to begin with — llama_sampler_grammar_i sets .backend_init = nullptr (src/llama-sampler.cpp:2758) and the reasoning-budget iface does the same (common/reasoning-budget.cpp:216). Only a contiguous PREFIX of the chain can ever leave the host. llama_sampler_chain_backend_init stops the prefix at the first sampler whose backend_init is null or returns false, and everything after it is host-side forever (src/llama-sampler.cpp:746-765). typical, xtc, mirostat, mirostat_v2, top_n_sigma, dry, adaptive_p and infill all have .backend_init = nullptr, so enabling any of them forecloses device sampling for every sampler downstream. Device sampling is refused outright under tensor split: "backend sampling not supported with SPLIT_MODE_TENSOR; using CPU" (src/llama-context.cpp:1216-1227). It is also refused for any request that wants pre-sampling logprobs — use_backend_sampling &= !need_pre_sample_logits (tools/server/server-context.cpp:1732-1737). There is no incremental grammar token-mask cache. Every llama_grammar_apply_impl re-decodes UTF-8 for each surviving candidate and re-runs the recursive stack rejection (src/llama-grammar.cpp:1368-1393, 1055-1124). Nothing is memoised between tokens beyond the vocab's token-piece cache (src/llama-vocab.cpp:3021). Cloning a grammar is O(stacks x rules x elements) because the stack pointers must be re-based (src/llama-grammar.cpp:1337-1348). GBNF cannot express left recursion — llama_grammar_init_impl detects it and returns nullptr, and common_sampler_init then throws "failed to parse grammar" (src/llama-grammar.cpp:1249-1261, common/sampling.cpp:280-282). LLGuidance is not compiled into this binary: LLAMA_LLGUIDANCE:BOOL=OFF in build-dflash2/CMakeCache.txt, so any grammar string beginning with "%llguidance" hits GGML_ABORT and kills the process rather than returning an error (common/sampling.cpp:219-224). --reasoning-budget is inert outside the server's chat endpoints. The sampler is only created when reasoning_budget_start and reasoning_budget_end are both non-empty (common/sampling.cpp:317), and the only code that populates them is the OAI chat path reading thinking_start_tag / thinking_end_tags off the chat template (tools/server/server-common.cpp:1360-1366, tools/server/server-schema.cpp:387-413). A template that declares no thinking end tags, /completion, and llama-cli all leave the budget unenforced no matter what N you pass. min_keep has no command-line flag at all — grep of common/arg.cpp finds no "--min-keep"; it is reachable only as a server JSON field (tools/server/server-schema.cpp:183-185), and top_k and top_n_sigma ignore it even then. mirostat != 0 discards the ENTIRE --samplers list. The whole for-loop over params.samplers is inside `if (params.mirostat == 0)` (common/sampling.cpp:346-406); the else branches build only temp + mirostat (common/sampling.cpp:407-415). The help text understates this — it claims only "Top K, Nucleus and Locally Typical" are ignored (common/arg.cpp:2212), but penalties, dry, top_n_sigma, min_p, xtc and adaptive_p are dropped too. logit_bias survives because it is added before the branch. The user cannot reorder logit_bias or the final selection sampler: logit_bias is unconditionally first (common/sampling.cpp:341-343) and dist/adaptive_p unconditionally last (common/sampling.cpp:400-406). The top-k bucket sort quantises logits into 128 buckets spanning only [-10, 10], clamping everything outside (src/llama-sampler.cpp:140-156) — that path is only taken for k > 128, and within a bucket it falls back to std::sort/std::partial_sort, so it is exact, but it is a fixed structure with no tuning knob. Without CUB, CUDA ARGSORT and TOP_K support only ne0 <= 1024 (ggml/src/ggml-cuda/ggml-cuda.cu:5249-5253) — device top-k/top-p over a real vocab would be unsupported and silently fall back to host. This build escapes that (CUDA 13.3 >= 11.07, ggml/src/ggml-cuda/common.cuh:110-112), but a non-CUB build of the same source cannot do device top-p at all.

**Exists but unused in our profile.** A profile of `--spec-type ngram-mod, -ctk q4_0 -ctv q4_0, -ngl auto --fit on, -np 1` with no sampling flags exercises only: logit_bias (empty stub), penalties (empty stub — penalty_repeat defaults to 1.00, so llama_sampler_penalties::is_disabled is true and init returns "?penalties", src/llama-sampler.cpp:2885-2896, 3211-3214), dry (empty stub, multiplier 0), top_n_sigma (empty stub, -1), top_k=40 (the cheap std::partial_sort branch, never the 128-bucket sort), typ_p (empty stub, 1.0), top_p=0.95 (over 40 candidates, so neither the >1024 nor the 256-element heuristic fires), min_p=0.05 (the already-sorted branch), xtc (empty stub), temp_ext with temp=0.80 and delta=0 (the plain temp branch, src/llama-sampler.cpp:2199-2201), and dist. NOT exercised by that profile: - Device/backend sampling entirely. It is off by default and needs -bs / --backend-sampling or a per-request backend_sampling:true (common/arg.cpp:2295-2301, common/common.h:295). Everything under src/llama-sampler.cpp's *_backend_apply functions, the ggml_argsort/ggml_cumsum/ggml_top_k graphs, llama_sampler_backend_support probing (src/llama-sampler.cpp:638-661), llama_set_sampler (src/llama-context.cpp:1209-1260) and the extra graph-node budget for sampling (src/llama-context.cpp:2323-2341) are all dead in this configuration. Consequently the SPLIT_MODE_TENSOR restriction and the CUB/ne0<=1024 argsort question are both moot here. - GBNF grammar and everything in src/llama-grammar.cpp. No --grammar, --grammar-file, -j/--json-schema or -jf, so common/sampling.cpp:270-282 builds no grmr, grammar_should_apply returns false at common/sampling.cpp:463, and the rejection-sampling / resample path at common/sampling.cpp:660-689 is never entered. json_schema_to_grammar, lazy grammars, trigger patterns and grammar prefill are all unused. - --reasoning-budget and the whole common/reasoning-budget.cpp state machine. Default -1, plus reasoning_budget_start/end are only populated by the server's OAI chat path, so rbudget is nullptr and llama_sampler_apply(rbudget, ...) at common/sampling.cpp:646 is the null-guarded no-op at src/llama-sampler.cpp:383-385. Same for reasoning_control and the "reasoning_end" control action. - The maximal-coupling speculative verifier that PR #27342 added (common/sampling.cpp:722-793). ngram-mod never fills a dists vector — only the DFlash selector does (common/speculative.cpp:1215-1279) — so the server takes the greedy accept-if-equal branch at tools/server/server-context.cpp:3831 / common/sampling.cpp:692-720. The speculative_rng and its 0x9e3779b9 seed derivation (common/sampling.cpp:434) are initialised but never drawn from. - mirostat / mirostat_v2 (src/llama-sampler.cpp:2460-2660), adaptive_p (src/llama-sampler.cpp:3750-3881; not in the default --samplers list so llama_sampler_init_adaptive_p is never called), infill (src/llama-sampler.cpp:4081-4290; COMMON_SAMPLER_TYPE_INFILL is not in the default list), dynamic temperature (dynatemp_range 0), and llguidance (compiled out). - min_keep is 0, so the min_keep guards in top_p (src/llama-sampler.cpp:1582), min_p (1775, 1793), xtc (2371) and typical are all trivially satisfied and never change a decision. - n_probs is 0, so populate_token_probs (tools/server/server-context.cpp:1899-1953) and get_token_probabilities are never called, and common_sampler_get_candidates(gsmpl, true) — the only caller that forces the post-hoc sort at common/sampling.cpp:836-853 — never runs. - With -np 1, cparams.n_outputs_max_per_seq stays 1, so llama_sampler_dist's transactional backend RNG (backend_transactional = n_outputs_max_per_seq > 1, src/llama-sampler.cpp:1266) would be false even if backend sampling were on. - -ctk/-ctv q4_0 and -ngl auto --fit on touch nothing in this area: no sampler reads cache types, and --fit only moves layer placement. The one indirect link is that --fit changes which device owns the output tensor, which would decide the buft passed to backend_init (src/llama-context.cpp:1236) — but only if backend sampling were enabled.

## KV cache, context and memory — llama.cpp build 10499 / commit 1deefcca3 (PR #27342 DFlash2 on master), as built into C:\AI\llama.cpp-dflash2\llama-server.exe (CUDA, CMAKE_CUDA_ARCHITECTURES=89, GGML_CUDA_FA_ALL_QUANTS=OFF per C:\AI\llama.cpp\build-dflash2\CMakeCache.txt:64,660)

### `-ctk / --cache-type-k and -ctv / --cache-type-v: the accepted value set`
**Use:** -ctk TYPE / -ctv TYPE, env LLAMA_ARG_CACHE_TYPE_K / LLAMA_ARG_CACHE_TYPE_V. Exactly nine values are accepted, matched by ggml_type_name string: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1. Anything else throws "Unsupported cache type: <s>" at arg-parse time.  ·  **default:** `f16 for both (common/common.h:577-578). Also f16 in the library default (src/llama-context.cpp:3545-3546).`

The nine-value list is a *parser* whitelist only. It is NOT the list of types that will actually run — see the fast-kernel entry below. The help text renders the list verbatim, so `--help` shows iq4_nl as if it were usable; nothing in arg.cpp rejects it.

`C:\AI\llama.cpp\common\arg.cpp:305-315 (const std::vector<ggml_type> kv_cache_types = { F32, F16, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1 })` · `C:\AI\llama.cpp\common\arg.cpp:317-324 (kv_cache_type_from_str, throws on miss)` · `C:\AI\llama.cpp\common\arg.cpp:2427-2451 (-ctk / -ctv option definitions)` · `C:\AI\llama.cpp\common\common.h:577-578 (defaults)` · `C:\AI\llama.cpp\common\common.cpp:1727-1728 (cparams.type_k/type_v = params.cache_type_k/v)`

### `Which -ctk/-ctv values have a CUDA FlashAttention kernel in THIS build`
**Use:** On this binary (GGML_CUDA_FA_ALL_QUANTS=OFF, sm_89) the KV types with a CUDA FA kernel are: F32 (silently upconverted to F16), F16, BF16, Q4_0, Q8_0. Q4_1, Q5_0, Q5_1 are explicitly returned as unsupported; IQ4_NL is never in any case list. Additionally K and V must be the SAME type.  ·  **default:** `f16/f16 — supported`

`-ctk q4_0 -ctv q4_0` is a first-class supported pair here. `-ctk q4_0 -ctv f16` is NOT — the K!=V guard at fattn.cu:443 kills the kernel before the type check runs. On sm_89 (Ada) with quantized K/V the VEC kernel is chosen only when Q->ne[1] <= 2 (fattn.cu:466-470), otherwise MMA_F16, which dequantises K and V to F16 into a scratch allocation sized by ggml_cuda_flash_attn_ext_get_alloc_size (fattn.cu:534-568) — so quantised KV costs extra compute-buffer VRAM during prompt processing, not just less cache VRAM.

`C:\AI\llama.cpp\ggml\src\ggml-cuda\fattn.cu:340-357 (ggml_cuda_fattn_kv_type_supported: F32/F16 true; Q4_1/Q5_0/Q5_1 `return false` under #ifndef GGML_CUDA_FA_ALL_QUANTS; Q4_0/Q8_0/BF16 true; default false)` · `C:\AI\llama.cpp\ggml\src\ggml-cuda\fattn.cu:442-446 (#ifndef GGML_CUDA_FA_ALL_QUANTS: `if (K->type != V->type) return BEST_FATTN_KERNEL_NONE;`)` · `C:\AI\llama.cpp\ggml\src\ggml-cuda\fattn.cu:321-325 (the only FATTN_VEC_CASES compiled without FA_ALL_QUANTS: F16/F16, Q4_0/Q4_0, Q8_0/Q8_0, BF16/BF16)` · `C:\AI\llama.cpp\ggml\CMakeLists.txt:208 (option GGML_CUDA_FA_ALL_QUANTS ... OFF)` · `C:\AI\llama.cpp\build-dflash2\CMakeCache.txt:660 (GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF)` · `C:\AI\llama.cpp\ggml\src\ggml-cuda\fattn.cu:586-588 (ggml_cuda_flash_attn_ext_supported == kernel != NONE)`

### `Automatic Hadamard rotation of quantised K/V (quality mitigation, silent)`
**Use:** Nothing to set. Enabled automatically when the cache type is quantised and the head dim is a multiple of 64. Disable with env LLAMA_ATTN_ROT_DISABLE=1.  ·  **default:** `on whenever ggml_is_quantized(type_k) && n_embd_head_k % 64 == 0 (same for V)`

Two log lines at startup tell you whether it engaged. If the head dim is not a multiple of 64 the rotation is silently skipped and -ctk q4_0 degrades more than the same flag on another model. n_embd_head_k_all must also be uniform across layers (set to -1 and thus disabling rotation if layers differ, llama-kv-cache.cpp:192-196).

`C:\AI\llama.cpp\src\llama-kv-cache.cpp:308-336 (attn_rot_k / attn_rot_v computation, LLAMA_ATTN_ROT_DISABLE env)` · `C:\AI\llama.cpp\src\llama-kv-cache.cpp:337-338 (logs `attn_rot_k = %d` / `attn_rot_v = %d` at load)` · `C:\AI\llama.cpp\src\llama-kv-cache.cpp:20-57 (ggml_gen_hadamard)` · `C:\AI\llama.cpp\src\llama-kv-cache.cpp:1863-1878 (K-shift path un-rotates, RoPEs, re-rotates for quantised K)`

### `-c / --ctx-size, and its padding/rounding`
**Use:** -c N, env LLAMA_ARG_CTX_SIZE. 0 = use the model's n_ctx_train.  ·  **default:** `0 (common/common.h:442) — i.e. the model's trained context. Library default if you bypass common is 512 (src/llama-context.cpp:3521).`

CLAMP/SURPRISE: n_ctx is always rounded UP to a multiple of 256 (llama-context.cpp:288). Then, when kv_unified is false, n_ctx_seq = n_ctx / n_seq_max padded to 256 and n_ctx is rewritten to n_ctx_seq * n_seq_max, logging `n_ctx is not divisible by n_seq_max - rounding down`. INTERACTION: passing `-c 0` explicitly is not the same as omitting -c — the handler at arg.cpp:1641-1644 also sets fit_params_min_ctx = UINT32_MAX, which turns OFF --fit's context-reduction step entirely (see fit entry).

`C:\AI\llama.cpp\common\arg.cpp:1636-1646` · `C:\AI\llama.cpp\common\common.h:442` · `C:\AI\llama.cpp\src\llama-context.cpp:131 (n_ctx == 0 -> hparams.n_ctx_train)` · `C:\AI\llama.cpp\src\llama-context.cpp:288 (cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256))` · `C:\AI\llama.cpp\src\llama-context.cpp:290-303 (n_ctx_seq derivation and round-down warning)`

### `--fit / -fit: automatic fitting of unset params to free device memory`
**Use:** -fit on|off, env LLAMA_ARG_FIT. Runs before the real model load, inside common_init_result, by loading the model twice or more with no_alloc=true and reading llama_get_memory_breakdown.  ·  **default:** `ON (common/common.h:468 `bool fit_params = true`)`

It is ON by default, which is the single most surprising default in this area. Order of operations: step 1 measure; step 2 reduce n_ctx (fit.cpp:290-373); step 3 assign layers per device (fit.cpp:376+). Because the wrapper catches the exception (fit.cpp:803-810), mutations already written into cparams->n_ctx BEFORE the throw survive — so `-ngl all` still gets a reduced context even though layer placement aborts with `n_gpu_layers already set by user`. Measurement is against FREE memory at that instant (fit.cpp:562), so it tracks whatever else is on the GPU at launch.

`C:\AI\llama.cpp\common\arg.cpp:2822-2835 (-fit / --fit)` · `C:\AI\llama.cpp\common\common.h:468` · `C:\AI\llama.cpp\common\common.cpp:1294-1302 (call site, before llama_model_load_from_file)` · `C:\AI\llama.cpp\common\fit.cpp:178-183 (impl entry; throws immediately for SPLIT_MODE_TENSOR)` · `C:\AI\llama.cpp\common\fit.cpp:29-70 (loads the model with mparams.no_alloc=true, load_mode=NONE, builds a context, reads the breakdown)` · `C:\AI\llama.cpp\common\fit.h:15-18 (contract: only params equal to llama_model_default_params are modified; n_ctx only if == 0)` · `C:\AI\llama.cpp\common\fit.cpp:791-814 (common_fit_params wrapper; exceptions are caught and downgraded to a status, NOT rethrown)`

### `-fitt / --fit-target: the per-device margin --fit leaves free`
**Use:** -fitt MiB or -fitt MiB0,MiB1,... (comma or slash separated). A single value is broadcast to all devices. Env LLAMA_ARG_FIT_TARGET.  ·  **default:** `1024 MiB (1 GiB) PER DEVICE — common/common.h:473 `fit_params_target = std::vector<size_t>(llama_max_devices(), 1024 * 1024*1024)``

On a 12 GB card the default silently forfeits 1 GiB. Lowering it (e.g. -fitt 384) is the direct lever for more KV. CLAMP: the parser throws if you pass as many or more values than llama_max_devices() (arg.cpp:2861-2865). The value is in MiB and multiplied by 1024*1024 at parse (arg.cpp:2868, 2872).

`C:\AI\llama.cpp\common\arg.cpp:2851-2874` · `C:\AI\llama.cpp\common\common.h:473` · `C:\AI\llama.cpp\common\fit.cpp:559-563 (`targets.push_back(dmds_full[id].free - margins[id])`)` · `C:\AI\llama.cpp\common\fit.cpp:200-207 (single value broadcast into the margins vector)`

### `-fitc / --fit-ctx: floor on the context --fit may choose`
**Use:** -fitc N, env LLAMA_ARG_FIT_CTX  ·  **default:** `4096 (common/common.h:470)`

--fit interpolates linearly between n_ctx_min and n_ctx_train on measured bytes-per-context-token, then rounds the result DOWN to a multiple of 256 (fit.cpp:344, comment `round down context for CUDA backend`). fit.cpp:361-366 is the branch that prints `user has requested full context size ... -> no change` when n_ctx_min == UINT32_MAX, i.e. when `-c 0` was passed.

`C:\AI\llama.cpp\common\arg.cpp:2876-2883` · `C:\AI\llama.cpp\common\common.h:470` · `C:\AI\llama.cpp\common\fit.cpp:310-368 (n_ctx_min used as the lower interpolation bound; final `cparams->n_ctx = std::max(cparams->n_ctx - cparams->n_ctx % 256, n_ctx_min)` at fit.cpp:344)`

### `-fitp / --fit-print: print the estimated memory instead of running`
**Use:** -fitp on|off, env LLAMA_ARG_FIT_ESTIMATE. Only registered for the LLAMA_EXAMPLE_FIT_PARAMS example, so it is not exposed on llama-server.  ·  **default:** `off (common/common.h:469)`

The equivalent human-readable table for a live context is common_memory_breakdown_print (fit.cpp:816-951), which is what the server logs; it splits model / context / compute and shows an `unaccounted` column = total - free - self.

`C:\AI\llama.cpp\common\arg.cpp:2836-2850` · `C:\AI\llama.cpp\common\fit.cpp:953-984 (common_fit_print — prints `<dev> <model_MiB> <context_MiB> <compute_MiB>` per device plus Host)`

### `-ngl / --gpu-layers: layer placement`
**Use:** -ngl N | auto | all, env LLAMA_ARG_N_GPU_LAYERS. 'auto' stores -1, 'all' stores -2, a number stores the number.  ·  **default:** `-1 = auto (common/common.h:465). NOTE arg.cpp:2742 asserts the default is negative.`

SURPRISE: at the library level -1 and -2 are identical — both become n_layer_all + 1. The ONLY thing that distinguishes `auto` from `all` is --fit: fit.cpp:377-379 throws `n_gpu_layers already set by user` when mparams->n_gpu_layers != llama_model_default_params().n_gpu_layers (which is -1, src/llama-model.cpp:2484). So `-ngl all` and `-ngl 99` both disable --fit's layer-placement pass; `-ngl auto` is the only value that lets it run. Layers are offloaded from the top down (i_gpu_start = n_layer_all + 1 - n_gpu_layers).

`C:\AI\llama.cpp\common\arg.cpp:2742-2761` · `C:\AI\llama.cpp\common\common.h:465` · `C:\AI\llama.cpp\src\llama-model.cpp:1745-1748 (`n_gpu_layers() = params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer_all + 1`)` · `C:\AI\llama.cpp\src\llama-model.cpp:1354-1366 (i_gpu_start / act_gpu_layers / get_layer_buft_list — layers are taken from the END of the stack)` · `C:\AI\llama.cpp\src\llama-model.cpp:1368-1370 (the input layer is ALWAYS kept on CPU)` · `C:\AI\llama.cpp\src\llama-model.cpp:1378 (dev_output = get_layer_buft_list(n_layer_all) — the output head counts as the +1 layer)`

### `KV cache placement follows layer placement`
**Use:** Implicit. Each KV layer's buffer type is the buffer type of the device that layer's weights were assigned to.  ·  **default:** `offload = cparams.offload_kqv = true`

A layer left on CPU by -ngl or by --fit gets its KV on CPU too. There is no flag to split KV independently of weights other than -nkvo (all-or-nothing).

`C:\AI\llama.cpp\src\llama-kv-cache.cpp:209-217 (`buft = ggml_backend_cpu_buffer_type(); if (offload) { dev = model.dev_layer(il); buft = ggml_backend_dev_buffer_type(dev); }`)` · `C:\AI\llama.cpp\src\llama-memory-recurrent.cpp:84-90 (same pattern for the recurrent state)` · `C:\AI\llama.cpp\src\llama-model.cpp:1356-1366`

### `-nkvo / --no-kv-offload`
**Use:** -nkvo (or -kvo to force on), env LLAMA_ARG_KV_OFFLOAD  ·  **default:** `KV offloading ENABLED (common/common.h:568 `no_kv_offload = false`)`

INTERACTION: -nkvo also disables pipeline parallelism (src/llama-context.cpp:428-433 includes `cparams.offload_kqv &&` in the pipeline_parallel condition). The code comment at llama-context.cpp:530-531 flags that the auto-FA device probe is `still wrong for cases like --no-kv-offload`.

`C:\AI\llama.cpp\common\arg.cpp:2404-2411` · `C:\AI\llama.cpp\common\common.cpp:1721 (`cparams.offload_kqv = !params.no_kv_offload`)` · `C:\AI\llama.cpp\src\llama-kv-cache.cpp:211-215 (buffer stays CPU when offload is false)` · `C:\AI\llama.cpp\src\llama-graph.cpp:2641-2644 (`if (!cparams.offload_kqv) ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu)` — the whole attention output node is pinned to CPU)`

### `-kvu / --kv-unified and -no-kvu`
**Use:** -kvu / --kv-unified to force on, -no-kvu / --no-kv-unified to force off. Env LLAMA_ARG_KV_UNIFIED. Registered for SERVER, PERPLEXITY, BATCHED, BENCH, PARALLEL.  ·  **default:** `false in common_params (common/common.h:563) and in the library (src/llama-context.cpp:3554). BUT llama-server flips it to TRUE whenever -np is auto.`

With -np 1 explicitly given, n_parallel is NOT negative, so the auto branch at server.cpp:151 does not fire and kv_unified stays FALSE unless you pass -kvu. With n_seq_max == 1 unified and non-unified are numerically identical (n_stream == 1 either way) — the practical difference at -np 1 is only that n_ctx_seq == n_ctx in both cases. GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max) at llama-kv-cache.cpp:132. LLAMA_MAX_SEQ is 256 (src/llama-cparams.h:8) and n_seq_max > 256 throws (llama-context.cpp:100-102).

`C:\AI\llama.cpp\common\arg.cpp:1713-1722` · `C:\AI\llama.cpp\common\common.h:563` · `C:\AI\llama.cpp\tools\server\server.cpp:151-156 (`if (params.n_parallel < 0) { params.n_parallel = 4; params.kv_unified = true; }`)` · `C:\AI\llama.cpp\src\llama-kv-cache.cpp:79 (`n_stream(unified ? 1 : n_seq_max)`)` · `C:\AI\llama.cpp\src\llama-context.cpp:290-303 (unified -> n_ctx_seq == n_ctx; else n_ctx is divided by n_seq_max)` · `C:\AI\llama.cpp\src\llama-context.cpp:1414,1661,3386 (`cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max` in the batch allocator)`

### `--swa-full: full-size sliding-window cache`
**Use:** --swa-full, env LLAMA_ARG_SWA_FULL. No negation flag.  ·  **default:** `false in common_params (common/common.h:562) — but TRUE in llama_context_default_params (src/llama-context.cpp:3553). common always overwrites it, so CLI/server effective default is false.`

The SWA cache is always padded to 256 cells (kv-cache-iswa.cpp:71-73, ref issue #17037) and its size depends on -ub: it is n_swa*(n_seq_max if unified else 1) + n_ubatch. INTERACTION: the server force-disables --swa-full on a model with no SWA layers and prints `swa_full is not supported by this model, it will be disabled` (server-context.cpp:1189-1191). Setting swa_full also sets the server's n_swa to 0, which in turn removes one of the three reasons to create context checkpoints (server-context.cpp:1195 + 3381-3384).

`C:\AI\llama.cpp\common\arg.cpp:1679-1685` · `C:\AI\llama.cpp\common\common.h:562` · `C:\AI\llama.cpp\src\llama-context.cpp:3553 (library default true)` · `C:\AI\llama.cpp\common\common.cpp:1724 (`cparams.swa_full = params.swa_full`)` · `C:\AI\llama.cpp\src\llama-kv-cache-iswa.cpp:70-79 (`size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256)`; swa_full sets size_swa = size_base)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:1188-1195 (`if (llama_model_n_swa(model_tgt) == 0) { swa_full = false; ... } n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt)`)`

### `-ctxcp / --ctx-checkpoints (alias --swa-checkpoints) and -cms / --checkpoint-min-step`
**Use:** -ctxcp N (env LLAMA_ARG_CTX_CHECKPOINTS), -cms N (env LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT). Server and CLI only.  ·  **default:** `n_ctx_checkpoints = 32 (common/common.h:613); checkpoint_min_step = 8192 tokens (common/common.h:614). 0 disables checkpoints; -cms 0 = no minimum spacing.`

CLAMP: -cms rejects negative values with `checkpoint-min-step must be non-negative` (arg.cpp:1697-1699). Checkpoints are created ONLY for SERVER_TASK_TYPE_COMPLETION (server-context.cpp:3374) and only when the memory cannot be rolled back cheaply: seq_rm type FULL, seq_rm type RS, or n_swa > 0 (server-context.cpp:3381-3384). PARTIAL_ONLY means a checkpoint for a hybrid model stores ONLY the recurrent/DeltaNet state and skips the attention KV entirely (src/llama-memory-hybrid.cpp:190-202); for a hybrid+SWA model it stores both (src/llama-memory-hybrid-iswa.cpp:195-202). A known-wrong assumption is tagged in-source at server-context.cpp:2277-2280 [TAG_CHECKPOINTS_FIX_POS_MIN]: the saved range is recorded as [pos_min,pos_max] which is not true for SWA models.

`C:\AI\llama.cpp\common\arg.cpp:1687-1701` · `C:\AI\llama.cpp\common\common.h:613-614` · `C:\AI\llama.cpp\tools\server\server-context.cpp:1318-1323 (enable/disable log)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:3371-3384 (the three conditions that make checkpoints happen at all)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:2243-2271 (eviction: too-close-together eviction, then LRU drop once size >= n_ctx_checkpoints)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:2283-2285 (`cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)`)` · `C:\AI\llama.cpp\include\llama.h:897-907 (flag values: NONE=0, PARTIAL_ONLY=1 == SWA_ONLY, ON_DEVICE=2)`

### `-cram / --cache-ram: the host-RAM prompt cache`
**Use:** -cram N (MiB), env LLAMA_ARG_CACHE_RAM. -1 = no size limit, 0 = disable the prompt cache entirely.  ·  **default:** `8192 MiB (common/common.h:615)`

SURPRISE: the second constructor argument is `limit_tokens = n_ctx` (server-context.cpp:1312), a token cap in addition to the MiB cap. It is then dynamically raised at runtime to limit_size/size_per_token (server-task.cpp:1878). SURPRISE: -1 maps to limit_size = 0, and 0 in the struct means "no limit" (server-task.h:609,615) — so -1 and an internal 0 are the same thing, while `-cram 0` never constructs the cache at all. A prompt whose state alone exceeds limit_size is skipped with a warning rather than evicting everything (server-task.cpp:1726-1731). Restore is scored by (f_keep, f_sim) and entries with f_keep < 0.25 are never trashed (server-task.cpp:1811-1813).

`C:\AI\llama.cpp\common\arg.cpp:1705-1712` · `C:\AI\llama.cpp\common\common.h:615` · `C:\AI\llama.cpp\tools\server\server-context.cpp:1304-1315 (construction; `prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx)`)` · `C:\AI\llama.cpp\tools\server\server-task.h:607-628 (`limit_size = 1MiB*(limit_size_mib < 0 ? 0 : limit_size_mib)`; 0 == no limit)` · `C:\AI\llama.cpp\tools\server\server-task.cpp:1708-1790 (alloc: skip if already cached, skip if a single state exceeds limit_size, evict oldest to make room, and on bad_alloc shrink limit_size to 0.4*size())` · `C:\AI\llama.cpp\tools\server\server-task.cpp:1866-1897 (update(): size eviction, then a token limit that is dynamically raised to limit_size/size_per_token)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:256-280 (prompt_save via llama_state_seq_get_data_ext with LLAMA_STATE_SEQ_FLAGS_NONE — full state, attention KV included)`

### `--cache-idle-slots / --no-cache-idle-slots`
**Use:** --cache-idle-slots / --no-cache-idle-slots, env LLAMA_ARG_CACHE_IDLE_SLOTS. Server only.  ·  **default:** `true (common/common.h:612)`

INTERACTION: it is force-disabled with `--cache-idle-slots requires --cache-ram, disabling` when cache_ram_mib == 0 (server-context.cpp:1374-1376). INTERACTION: with kv_unified false the idle slot's KV is NOT cleared from VRAM, only copied to RAM — tagged in source [TAG_IDLE_SLOT_CLEAR] (server-context.cpp:1381-1383).

`C:\AI\llama.cpp\common\arg.cpp:1722-1729` · `C:\AI\llama.cpp\common\common.h:612` · `C:\AI\llama.cpp\tools\server\server-context.cpp:1373-1389 (requires cache-ram; behaviour differs by kv_unified)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:2355-2363 (the save-on-new-task path)`

### `--cache-reuse N: reuse non-prefix chunks by shifting KV`
**Use:** --cache-reuse N, env LLAMA_ARG_CACHE_REUSE. Also settable per request as the JSON field `n_cache_reuse`.  ·  **default:** `0 = disabled (common/common.h:610)`

INTERACTIONS that silently zero it at startup: multimodal loaded -> `cache_reuse is not supported by multimodal, it will be disabled` (server-context.cpp:1170-1174); !llama_memory_can_shift -> `cache_reuse is not supported by this context, it will be disabled` (server-context.cpp:1182-1186). It also requires cache_prompt (the prefix must have been kept) — the whole block is inside `if (slot.task->params.cache_prompt)` at server-context.cpp:3128.

`C:\AI\llama.cpp\common\arg.cpp:3523-3532` · `C:\AI\llama.cpp\common\common.h:610` · `C:\AI\llama.cpp\tools\server\server-schema.cpp:67-69 (per-request field, hard limits 0..INT32_MAX)` · `C:\AI\llama.cpp\tools\server\server-schema.cpp:527 (per-request default inherited from --cache-reuse)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:3138-3196 (the reuse loop: matches chunks of >= n_cache_reuse tokens, then seq_rm + seq_add with kv_shift)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:3140-3147 (`can_cache_reuse = llama_memory_can_shift(...) && !slot.prompt.tokens.has_mtmd`; warns `cache reuse is not supported - ignoring n_cache_reuse = %d`)`

### `-sps / --slot-prompt-similarity: slot selection by longest common prefix`
**Use:** -sps FLOAT. Server only. 0.0 disables similarity-based selection and falls back to LRU.  ·  **default:** `0.10 (common/common.h:677)`

Only slots that are idle AND non-empty are considered (server-context.cpp:1514-1524). With -np 1 there is exactly one slot, so -sps changes almost nothing about which slot is picked — but it still changes whether update_cache is set, and therefore whether the prompt cache is written and re-loaded on each new task (server-context.cpp:1575-1596). Note the asymmetry: the LRU path always sets update_cache, the similarity path only when f_keep < 0.5.

`C:\AI\llama.cpp\common\arg.cpp:3736-3743` · `C:\AI\llama.cpp\common\common.h:677` · `C:\AI\llama.cpp\tools\server\server-context.cpp:1197-1199 (copied into the server field)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:1503-1552 (the selection loop; f_sim_cur = lcp_len / task.tokens.size(); must be strictly > the threshold)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:1546-1549 (`if (f_keep < 0.5f) update_cache = true` — losing more than half the slot triggers a prompt-cache save)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:1554-1573 (LRU fallback, which always sets update_cache = true)`

### `--context-shift / --no-context-shift`
**Use:** --context-shift to enable, --no-context-shift to disable. Env LLAMA_ARG_CONTEXT_SHIFT.  ·  **default:** `DISABLED (common/common.h:561 `bool ctx_shift = false`)`

INTERACTIONS: force-disabled for multimodal (`ctx_shift is not supported by multimodal, it will be disabled`, server-context.cpp:1165-1168) and when llama_memory_can_shift is false (server-context.cpp:1176-1180). n_discard is clamped at server-context.cpp:2855: `std::clamp(n_discard, 0, std::max(0, n_left - 1))`.

`C:\AI\llama.cpp\common\arg.cpp:1729-1736` · `C:\AI\llama.cpp\common\common.h:561` · `C:\AI\llama.cpp\tools\server\server-context.cpp:1821 (`if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx)` — the request is rejected instead of shifted)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:2822-2875 (the shift itself: n_discard defaults to n_left/2, clamped to [0, n_left-1], then seq_rm + seq_add(-n_discard))`

### `llama_memory_can_shift — the gate under both context shift and cache reuse`
**Use:** Not a flag. Computed per model/cache. For a plain llama_kv_cache it is false for LLM_ARCH_STEP35 and false for ANY model whose rope type is MROPE or IMROPE.  ·  **default:** `true`

THE decisive fact for a Qwen3.5-style model: QWEN35 uses IMROPE, so n_pos_per_embd() == 4, so get_can_shift() is false, so the server prints BOTH `ctx_shift is not supported by this context, it will be disabled` and `cache_reuse is not supported by this context, it will be disabled` at startup and zeroes n_cache_reuse. Passing --cache-reuse on such a model is a no-op you will only notice in the log. The recurrent side reports can_shift = true unconditionally (src/llama-memory-recurrent.cpp:695-698) — it is the attention half that vetoes.

`C:\AI\llama.cpp\src\llama-kv-cache.cpp:1171-1179 (`if (model.arch == LLM_ARCH_STEP35) return false; if (hparams.n_pos_per_embd() > 1) return false;`)` · `C:\AI\llama.cpp\src\llama-hparams.cpp:238-240 (`n_pos_per_embd() = rope_type == MROPE || rope_type == IMROPE ? 4 : 1`)` · `C:\AI\llama.cpp\src\llama-model.cpp:2743-2746 (LLM_ARCH_QWEN35 / QWEN35MOE -> LLAMA_ROPE_TYPE_IMROPE)` · `C:\AI\llama.cpp\src\llama-model.cpp:1268 (`hparams.rope_type = llama_model_rope_type(this)`)` · `C:\AI\llama.cpp\src\llama-memory-hybrid.cpp:133-136 (hybrid delegates can_shift to the attention cache only)` · `C:\AI\llama.cpp\src\llama-kv-cache.cpp:566-573 (seq_add itself GGML_ASSERTs n_pos_per_embd() == 1)`

### `Hybrid / recurrent memory selection for a Qwen3.5-style model`
**Use:** Automatic from the architecture. QWEN35 / QWEN35MOE / QWEN3NEXT are in llm_arch_is_hybrid, so create_memory builds llama_memory_hybrid (attention KV cache + separate recurrent state), with layer filters splitting on hparams.is_recr(il).  ·  **default:** `n/a`

qwen35.cpp declares no SWA, so swa_type stays NONE and the PLAIN llama_memory_hybrid is used — not hybrid_iswa. That is why --swa-full is inert on it. Note that `unified` is passed to mem_attn but NOT to mem_recr (llama-memory-hybrid.cpp:40 vs 54-63): the DeltaNet state is always per-sequence.

`C:\AI\llama.cpp\src\llama-arch.cpp:1008-1030 (llm_arch_is_hybrid includes QWEN35, QWEN35MOE, QWEN3NEXT)` · `C:\AI\llama.cpp\src\llama-model.cpp:2281-2303 (hybrid branch; filter_attn = `il < n_layer() && !is_recr(il)`, filter_recr = `il < n_layer() && is_recr(il)`)` · `C:\AI\llama.cpp\src\llama-model.cpp:2305-2344 (hybrid_iswa when hparams.swa_type != NONE, plain llama_memory_hybrid otherwise)` · `C:\AI\llama.cpp\src\models\qwen35.cpp:21-26 (is_recr_impl from LLM_KV_ATTENTION_RECURRENT_LAYERS, else `(i+1) % full_attn_interval != 0`)` · `C:\AI\llama.cpp\src\llama-memory-hybrid.cpp:11-64 (ctor: mem_attn = llama_kv_cache, mem_recr = llama_memory_recurrent)`

### `Recurrent / DeltaNet state sizing and type`
**Use:** Not configurable. rows = rs_size * (1 + n_rs_seq) where rs_size = max(1, n_seq_max); width = hparams.n_embd_r() for the conv state and hparams.n_embd_s() for the recurrent state.  ·  **default:** `type_r = type_s = GGML_TYPE_F32, hard-coded at the call site`

The DeltaNet state does NOT scale with context length — only with n_seq_max and n_rs_seq. So raising -c costs nothing on the recurrent half and everything on the attention half. The `RS buffer size` line in the startup log is the number to read.

`C:\AI\llama.cpp\src\llama-model.cpp:2314-2315 (`/* recurrent_type_r */ GGML_TYPE_F32, /* recurrent_type_s */ GGML_TYPE_F32` for hybrid_iswa)` · `C:\AI\llama.cpp\src\llama-model.cpp:2335-2336 (same for llama_memory_hybrid)` · `C:\AI\llama.cpp\src\llama-model.cpp:2274-2275 (same for pure recurrent models)` · `C:\AI\llama.cpp\src\llama-memory-recurrent.cpp:99-101 (`n_rows = mem_size * (1 + n_rs_seq)`; ggml_new_tensor_2d(type_r, n_embd_r(), n_rows))` · `C:\AI\llama.cpp\src\llama-memory-recurrent.cpp:118-127 (the `RS buffer size` / `size = ... R (%s) ... S (%s)` startup log)` · `C:\AI\llama.cpp\src\llama-hparams.cpp:183-205 (n_embd_r) and 207-229 (n_embd_s)`

### `n_rs_seq: bounded partial rollback of the recurrent state`
**Use:** Not a direct flag. cparams.n_rs_seq = params.speculative.need_n_rs_seq(), which is draft.n_max if any speculative type is DRAFT_MTP / DRAFT_EAGLE3 / DRAFT_DFLASH / DRAFT_DSPARK, else 0.  ·  **default:** `0`

CLAMP that is easy to miss: the clamp to 0 for an unsupported arch logs at LLAMA_LOG_DEBUG only, so a request for rollback silently becomes no rollback. With n_rs_seq == 0 any partial seq_rm on the recurrent half returns false, which is what pushes common_context_can_seq_rm to FULL and turns context checkpoints on.

`C:\AI\llama.cpp\common\common.h:386-392 (need_n_rs_seq)` · `C:\AI\llama.cpp\common\common.cpp:1697 (`cparams.n_rs_seq = params.speculative.need_n_rs_seq()`)` · `C:\AI\llama.cpp\src\llama-context.cpp:104-109 (CLAMP: `if (n_rs_seq > 0 && !llm_arch_supports_rs_rollback(arch)) n_rs_seq = 0`, logged only at DEBUG level)` · `C:\AI\llama.cpp\src\llama-arch.cpp:1044-1055 (llm_arch_supports_rs_rollback: QWEN35, QWEN35MOE, DEEPSEEK4, NEMOTRON_H, NEMOTRON_H_MOE)` · `C:\AI\llama.cpp\src\llama-memory-recurrent.cpp:180-190 (`rollback >= 1 && rollback <= n_rs_seq` else return false)` · `C:\AI\llama.cpp\src\llama-memory-recurrent.cpp:422-424 (the trailing 1+n_rs_seq tokens of a seq must stay in one ubatch)`

### `common_context_can_seq_rm: how the server discovers what the memory can undo`
**Use:** Probed once at server init by clearing memory, decoding 2 tokens, and trying llama_memory_seq_rm(mem, 0, 1, -1). Returns NO / PART / FULL / RS.  ·  **default:** `PART`

The probe runs a real decode, so it is part of your startup cost. RS short-circuits before the seq_rm attempt (`if (llama_n_rs_seq(ctx) > 0)` at common.cpp:1581-1585) — meaning with n_rs_seq > 0 the FULL/PART distinction is never even measured.

`C:\AI\llama.cpp\common\common.h:970-979 (the enum)` · `C:\AI\llama.cpp\common\common.cpp:1559-1599 (the probe; note it clears the memory and llama_synchronize on the way out)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:1209-1218 (`ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt)`; logs `speculative decoding not supported by this context` for NO, `speculative decoding will use checkpoints` for FULL)` · `C:\AI\llama.cpp\tools\server\server-context.cpp:3381-3384 (FULL or RS or n_swa>0 -> create checkpoints)`

### `-b / -ub interaction with context and SWA`
**Use:** -b N (env LLAMA_ARG_BATCH), -ub N (env LLAMA_ARG_UBATCH)  ·  **default:** `n_batch = 2048, n_ubatch = 512 (src/llama-context.cpp:3522-3523)`

CLAMP: for a causal model n_batch is silently capped at n_ctx, and n_ubatch is silently capped at n_batch. Note the capping uses cparams.n_ctx AFTER the 256-padding at llama-context.cpp:288? No — line 245 runs BEFORE line 288, so it caps against the unpadded n_ctx.

`C:\AI\llama.cpp\common\arg.cpp:1657-1670` · `C:\AI\llama.cpp\src\llama-context.cpp:245 (`cparams.n_batch = causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch`)` · `C:\AI\llama.cpp\src\llama-context.cpp:247 (`cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch)`)` · `C:\AI\llama.cpp\src\llama-kv-cache-iswa.cpp:73 (n_ubatch is added to the SWA cache size)`

### `-ctkd / -ctvd / -ngld: separate KV settings for a draft model`
**Use:** --spec-draft-type-k / -ctkd, --spec-draft-type-v / -ctvd, --spec-draft-ngl / -ngld. Same nine-value type list.  ·  **default:** `f16 / f16 (common/common.h:340-341); draft n_gpu_layers = -1 auto (common/common.h:338)`

These are entirely separate from -ctk/-ctv; setting -ctk q4_0 does NOT propagate to the draft context. Only relevant when a real draft model context exists.

`C:\AI\llama.cpp\common\arg.cpp:4022-4048` · `C:\AI\llama.cpp\common\arg.cpp:4125-4144` · `C:\AI\llama.cpp\common\common.h:338-341`

### `-ot / --override-tensor and -ncmoe / --n-cpu-moe as the manual alternative to --fit`
**Use:** -ot <pattern>=<buffer type>, -cmoe (all MoE weights to CPU), -ncmoe N (MoE weights of the first N layers to CPU)  ·  **default:** `none`

INTERACTION: any -ot / -cmoe / -ncmoe makes --fit's layer-placement pass throw `model_params::tensor_buft_overrides already set by user, abort` (fit.cpp:398-400). The context-reduction step still ran before that point, so you get fit's context shrink without fit's placement. --fit uses the same mechanism internally, splitting a boundary layer into ATTN / UP / GATE / MOE fractions (common/fit.cpp:17-23).

`C:\AI\llama.cpp\common\arg.cpp:2714-2741` · `C:\AI\llama.cpp\common\fit.cpp:396-400 (`did not provide buffer to set tensor_buft_overrides, abort` / `model_params::tensor_buft_overrides already set by user, abort`)` · `C:\AI\llama.cpp\common\fit.cpp:462-497 (how --fit itself synthesises overflow overrides per partially-offloaded layer)` · `C:\AI\llama.cpp\common\fit.cpp:485-489 (throws if llama_max_tensor_buft_overrides() is exceeded)`

### `-sm / --split-mode and -mg / --main-gpu`
**Use:** -sm none|layer|row|tensor (env LLAMA_ARG_SPLIT_MODE), -mg INDEX (env LLAMA_ARG_MAIN_GPU)  ·  **default:** `split_mode = LLAMA_SPLIT_MODE_LAYER (common/common.h:475); main_gpu = 0 (common/common.h:466)`

-sm tensor kills --fit outright at fit.cpp:183 (before any measurement), so unlike the other aborts it does not even get the context reduction. -sm row aborts only the placement pass. -mg only matters for -sm none (model device) and -sm row (KV + intermediates device), per the help string at arg.cpp:2814.

`C:\AI\llama.cpp\common\arg.cpp:2761-2784` · `C:\AI\llama.cpp\common\arg.cpp:2811-2821` · `C:\AI\llama.cpp\common\fit.cpp:182-184 (`llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort`)` · `C:\AI\llama.cpp\common\fit.cpp:390-393 (`changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort`)` · `C:\AI\llama.cpp\src\llama-context.cpp:3585-3594 (SPLIT_MODE_TENSOR requires flash_attn ENABLED and an arch in llm_arch_supports_sm_tensor)`

### `-dt / --defrag-thold: accepted and ignored`
**Use:** -dt N — parses, warns, discards.  ·  **default:** `n/a`

The flag still exists so old scripts do not die, but the value goes nowhere. llama_context_params still carries a defrag_thold field defaulted to -1.0f (src/llama-context.cpp:3542) that nothing reads — grep for `defrag` in src/llama-kv-cache.cpp returns nothing.

`C:\AI\llama.cpp\common\arg.cpp:2522-2531 (`GGML_UNUSED(params); GGML_UNUSED(value); LOG_WRN("DEPRECATED: --defrag-thold is deprecated and no longer necessary to specify")`)`

**What this area CANNOT do.** WHAT THIS AREA CANNOT DO, with the line that forecloses it: 1. The recurrent / DeltaNet state cannot be quantised, at all. Every construction site passes GGML_TYPE_F32 as a literal for both the conv state and the recurrent state: src/llama-model.cpp:2274-2275 (pure recurrent), :2314-2315 (hybrid_iswa), :2335-2336 (hybrid). -ctk/-ctv reach only the attention half — src/llama-memory-hybrid.cpp:36-37 passes type_k/type_v to mem_attn, while :55-56 passes type_r/type_s to mem_recr from separate arguments. There is no flag anywhere in common/arg.cpp that sets type_r or type_s. 2. You cannot quantise V without FlashAttention. src/llama-context.cpp:3602-3611 upgrades AUTO to ENABLED and hard-errors on DISABLED ("quantized V cache requires flash_attn to be enabled"), and src/llama-context.cpp:463-466 throws "quantized V cache was requested, but this requires Flash Attention" after the graph reserve if FA ended up off. Reason: with FA off the V cache is stored transposed (v_trans = !cparams.flash_attn, passed at src/llama-model.cpp:2124 and elsewhere; used at src/llama-kv-cache.cpp:206). 3. On this binary, K and V cache types must be identical. ggml/src/ggml-cuda/fattn.cu:442-446 — `#ifndef GGML_CUDA_FA_ALL_QUANTS if (K->type != V->type) return BEST_FATTN_KERNEL_NONE;` — and the build has GGML_CUDA_FA_ALL_QUANTS=OFF (build-dflash2/CMakeCache.txt:660, option default at ggml/CMakeLists.txt:208). Mixed pairs like q8_0/q4_0 or q4_0/f16 have no CUDA kernel. 4. q4_1, q5_0, q5_1 and iq4_nl are accepted by the parser but have no CUDA FA kernel here. ggml/src/ggml-cuda/fattn.cu:343-347 returns false for Q4_1/Q5_0/Q5_1 under `#ifndef GGML_CUDA_FA_ALL_QUANTS`; iq4_nl falls to `default: return false` at fattn.cu:355. common/arg.cpp:305-315 still lists all nine, so `--help` advertises types this build cannot run on the GPU. 5. K/V block size must divide the head dimension. src/llama-context.cpp:3613-3622 (K) and :3624-3633 (V) refuse to build the context: "K cache type %s with block size %u does not divide n_embd_head_k=%u". Q4_0/Q8_0/Q5_0/Q5_1/Q4_1 all have block size 32; iq4_nl also 32. 6. MLA models and DEEPSEEK4 cannot have different K and V types at all, regardless of backend. src/llama-context.cpp:3597-3600: "model does not support different K (%s) and V (%s) cache types". 7. Context shift and --cache-reuse are impossible on any M-RoPE / I-M-RoPE model, which includes QWEN35 and QWEN35MOE. src/llama-kv-cache.cpp:1176-1178 (`if (hparams.n_pos_per_embd() > 1) return false;`) foreclosing get_can_shift, with n_pos_per_embd defined at src/llama-hparams.cpp:238-240 and the arch mapped to IMROPE at src/llama-model.cpp:2743-2746. The server then zeroes both features at tools/server/server-context.cpp:1176-1186. The underlying seq_add would abort anyway: src/llama-kv-cache.cpp:573 `GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1")`. 8. There is no KV defragmentation any more. common/arg.cpp:2522-2531 discards --defrag-thold; grep for "defrag" in src/llama-kv-cache.cpp yields nothing. A fragmented unified cache cannot be compacted on request. 9. --fit cannot help you once you have set the thing it wants to set. common/fit.cpp:377-379 aborts if n_gpu_layers != -1; :386-388 aborts if tensor_split was set by the user; :398-400 aborts if tensor_buft_overrides were set (-ot / -cmoe / -ncmoe); :390-393 aborts for SPLIT_MODE_ROW; :182-184 aborts for SPLIT_MODE_TENSOR before measuring anything. And per common/fit.h:15-18 it modifies n_ctx "if and only if equal to 0" — a numeric -c is never adjusted (the branch that says so is fit.cpp:368-370, "context size set by user to %u -> no change"). 10. --fit cannot reduce context below -fitc, and cannot reduce it at all if you wrote `-c 0`. common/fit.cpp:311 (`if (hp_nct > n_ctx_min)`), :344 (`std::max(cparams->n_ctx - cparams->n_ctx % 256, n_ctx_min)`), :361-363 (`user has requested full context size of %u -> no change` when n_ctx_min == UINT32_MAX), set by common/arg.cpp:1641-1644. 11. n_ctx cannot be an arbitrary number. src/llama-context.cpp:288 pads it up to a multiple of 256 unconditionally; :294 pads n_ctx_seq to 256 and :300-302 rewrites n_ctx down to n_ctx_seq * n_seq_max when they disagree; :296-298 throws "n_ctx_seq == 0" if the division underflows. 12. n_seq_max cannot exceed 256. src/llama-cparams.h:8 (`#define LLAMA_MAX_SEQ 256`) and src/llama-context.cpp:100-102 ("n_seq_max must be <= 256"). 13. A context checkpoint on a non-SWA hybrid model does not contain the attention KV. src/llama-memory-hybrid.cpp:190-196 — under LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY the mem_attn->state_write call is skipped entirely; only mem_recr is written. The server always creates checkpoints with that flag (tools/server/server-context.cpp:2283). So on a Qwen3.5-style model a checkpoint restores the DeltaNet state and nothing else; the attention KV must be recomputed. 14. --swa-full is meaningless on a model with no SWA layers and will be turned off with a warning. tools/server/server-context.cpp:1188-1192. qwen35 declares no SWA (src/models/qwen35.cpp has no swa_type assignment), so hparams.swa_type stays NONE and src/llama-model.cpp:2305 routes to plain llama_memory_hybrid rather than llama_memory_hybrid_iswa. 15. The SWA cache size cannot be tuned directly — only via -ub and --swa-full. src/llama-kv-cache-iswa.cpp:73: `size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256)`. 16. --cache-idle-slots cannot work without --cache-ram. tools/server/server-context.cpp:1374-1376 disables it with a warning when cache_ram_mib == 0. 17. --cache-reuse and context shift cannot work with multimodal. tools/server/server-context.cpp:1165-1174, and the belt-and-braces `GGML_ABORT("not supported by multimodal")` at :3157. 18. The prompt cache cannot hold a single state larger than the -cram limit; it is skipped rather than making room. tools/server/server-task.cpp:1726-1731. 19. You cannot place the KV of a layer on a different device from that layer's weights, short of -nkvo which moves all of it to CPU. src/llama-kv-cache.cpp:209-217 derives buft from model.dev_layer(il) with no per-layer override.

**Exists but unused in our profile.** A profile of `--spec-type ngram-mod -ctk q4_0 -ctv q4_0 -ngl auto --fit on -np 1` leaves the following in this area unexercised: - n_rs_seq and the whole bounded-rollback path. common/common.h:386-392 (need_n_rs_seq) returns non-zero only for DRAFT_MTP / DRAFT_EAGLE3 / DRAFT_DFLASH / DRAFT_DSPARK; ngram-mod is COMMON_SPECULATIVE_TYPE_NGRAM_MOD (common/common.h:180, mapped from the string at common/speculative.cpp:44) and is self-speculative, so cparams.n_rs_seq == 0 (common/common.cpp:1697). Consequence: the recurrent state is allocated with n_rows = mem_size * 1 (src/llama-memory-recurrent.cpp:99), partial seq_rm on the recurrent half always fails (src/llama-memory-recurrent.cpp:184-188), common_context_can_seq_rm returns FULL rather than RS (common/common.cpp:1588-1591), and context checkpoints are therefore ON for this profile via the FULL branch (tools/server/server-context.cpp:3381-3384) — the RS branch is never taken. - -ctkd / -ctvd / -ngld and all draft-context memory. common/arg.cpp:4022-4048 and :4125-4144. ngram-mod builds no draft model context, so slot.prompt_save's `cur_size_dft` is always 0 (tools/server/server-context.cpp:262) and the drft half of every prompt-cache entry stays empty (tools/server/server-task.h:585-590). - kv_unified. With `-np 1` explicitly given, params.n_parallel is not negative, so the auto branch at tools/server/server.cpp:151-156 never fires and kv_unified stays at its common default of false (common/common.h:563). And with n_seq_max == 1, n_stream is 1 either way (src/llama-kv-cache.cpp:79), so the whole multi-stream code path — seq_to_stream (src/llama-kv-cache.cpp:146-153), the per-stream v_cells vector, the LLAMA_MAX_SEQ widening in the batch allocator (src/llama-context.cpp:1414, 1661, 3386) — is inert. The `--cache-idle-slots` kv_unified branch that actually frees VRAM (tools/server/server-context.cpp:1378-1380) is also not the one taken; you get the [TAG_IDLE_SLOT_CLEAR] non-unified branch at :1381-1383 instead. - -sps / --slot-prompt-similarity as a slot chooser. With one slot the loop at tools/server/server-context.cpp:1505-1536 can only ever return that slot; the flag's remaining effect is the update_cache side effect at :1546-1549. - --swa-full, hparams.n_swa, and the entire llama_kv_cache_iswa / llama_memory_hybrid_iswa path, on a Qwen3.5-style model with no SWA layers. src/llama-model.cpp:2305 selects plain llama_memory_hybrid; tools/server/server-context.cpp:1188-1195 forces swa_full false and n_swa 0. - --context-shift and --cache-reuse: unreachable on an IMROPE model (src/llama-kv-cache.cpp:1176-1178), and both default to off/0 anyway (common/common.h:561, :610). - -nkvo, -sm row/tensor/none, -mg, -ts, -ot / -cmoe / -ncmoe: all default-off, and each of the last three would additionally abort --fit's placement pass (common/fit.cpp:182-184, 386-393, 398-400). - The FA_ALL_QUANTS mixed-type kernel table (ggml/src/ggml-cuda/fattn.cu:265-319) is not even compiled into this binary; with -ctk q4_0 -ctv q4_0 the live path is the four-case table at fattn.cu:321-325. - With quantised K/V on sm_89, BEST_FATTN_KERNEL_VEC is taken only when Q->ne[1] <= 2 (ggml/src/ggml-cuda/fattn.cu:466-470); prompt processing runs BEST_FATTN_KERNEL_MMA_F16 with the dequant-to-F16 scratch buffer sized at fattn.cu:534-568. The TILE kernel (fattn.cu:522-527) is unreachable on Ada. - -dt / --defrag-thold: accepted and discarded (common/arg.cpp:2522-2531). - -fitp / --fit-print: registered only for LLAMA_EXAMPLE_FIT_PARAMS (common/arg.cpp:2850), so it is not on llama-server."

## Model loading and quantisation (llama.cpp build 10499, commit 1deefcca3, C:\AI\llama.cpp; staged binary C:\AI\llama.cpp-dflash2\llama-server.exe)

### `GGUF weight types recognised at load`
**Use:** No flag. The loader reads whatever ggml_type each tensor carries. The 40 live types are F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K..Q6_K, Q8_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, IQ4_NL, IQ4_XS, I8/I16/I32/I64, F64, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, Q2_0. Removed: Q4_2, Q4_3, Q4_0_4_4/4_8/8_8, IQ4_NL_4_4/4_8/8_8.  ·  **default:** `n/a — read from file`

The printed `file type` is only a GUESS derived from the most-common tensor type (src/llama-model-loader.cpp:723-775), then overwritten by general.file_type if that KV exists (src/llama-model-loader.cpp:780-785). A guessed ftype is OR'd with LLAMA_FTYPE_GUESSED=1024 (src/llama-model-loader.cpp:778). An unknown type_max silently reports ALL_F32 (src/llama-model-loader.cpp:770-774) — the header line can therefore be wrong without any error.

`ggml/include/ggml.h:390-433` · `src/llama-model-loader.cpp:723-775`

### `-ot / --override-tensor: per-tensor buffer placement`
**Use:** -ot "<ecmascript-regex>=<buffer-type-name>" , repeatable and comma-separated. Env LLAMA_ARG_OVERRIDE_TENSOR. Matched with std::regex_search against the FULL tensor name including the `.weight` suffix; first match in argument order wins.  ·  **default:** `none (empty pattern list)`

CLAMP ON NAMES: the accepted buffer-type strings are ONLY the per-device DEFAULT buffer types, collected from ggml_backend_dev_buffer_type() over every device (common/arg.cpp:256-263). On this machine that is exactly `CPU` and `CUDA0` (verified: `llama-server.exe -ot foo=NOSUCHBUF` prints the list). You cannot name CUDA_Host, the CPU repack buft, or a split buft. A bad name throws and prints the list. SPECIAL CASE: when the target resolves to ggml_backend_cpu_buffer_type(), the loader does NOT pin the tensor to the plain CPU buffer — it re-runs select_weight_buft over the whole cpu_buft_list (ACCEL → GPU-host → CPU-extra/repack → CPU) (src/llama-model-loader.cpp:1183-1185). So `-ot ...=CPU` can land in a pinned-host or repack buffer. A one-time warning fires if any -ot-to-CPU is combined with mmap: 'consider using --load-mode none for better performance' (src/llama-model-loader.cpp:1186-1192). A fresh std::regex is compiled per (tensor × override) inside the loop (src/llama-model-loader.cpp:1181) — cost is O(n_tensors × n_overrides) regex constructions.

`common/arg.cpp:2714-2719` · `common/arg.cpp:253-284` · `src/llama-model-loader.cpp:1177-1203`

### `-cmoe / --cpu-moe: all MoE expert weights to CPU`
**Use:** -cmoe (no value). Env LLAMA_ARG_CPU_MOE. Pushes one override: pattern `\.ffn_(up|down|gate|gate_up)_(ch|)exps` → CPU buffer type.  ·  **default:** `off`

It is literally sugar for one -ot entry, so it inherits every -ot behaviour above, including the CPU re-selection over cpu_buft_list. It matches `_exps` and `_chexps` only — shared/dense FFN tensors are untouched. Because it populates params.tensor_buft_overrides it DISABLES --fit's layer fitting (see interactions).

`common/arg.cpp:2720-2726` · `common/common.h:1113-1121`

### `-ncmoe / --n-cpu-moe N: MoE experts of the first N layers to CPU`
**Use:** -ncmoe N. Env LLAMA_ARG_N_CPU_MOE. Emits N overrides, one per layer: `blk\.<i>\.ffn_(up|down|gate|gate_up)_(ch|)exps` → CPU.  ·  **default:** `0`

CLAMP: N<0 throws 'invalid value'. No upper clamp — N larger than n_layer just emits patterns that never match. The literal `\.` after the layer index prevents blk.1 from matching blk.10 (common/common.h:1115). Patterns are kept alive in a function-static std::list, so they persist for process lifetime. Same --fit-disabling side effect as -cmoe.

`common/arg.cpp:2727-2741` · `common/common.h:1115-1117`

### `-otd / -cmoed / -ncmoed: the same three, for the draft model`
**Use:** --spec-draft-override-tensor|-otd, --spec-draft-cpu-moe|-cmoed, --spec-draft-n-cpu-moe|-ncmoed. Env LLAMA_ARG_SPEC_DRAFT_CPU_MOE / LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE.  ·  **default:** `none`

They write params.speculative.draft.tensor_buft_overrides, a SEPARATE vector that is only null-terminated when non-empty (common/arg.cpp:952-954) and is NOT padded to 4096 — so the draft model is never handed to --fit's override writer. -otd has no set_env; -cmoed/-ncmoed do.

`common/arg.cpp:4048-4074` · `common/arg.cpp:952-954`

### `-ngl / --gpu-layers: how many layers get a GPU buffer list`
**Use:** -ngl N | -ngl auto | -ngl all. Env LLAMA_ARG_N_GPU_LAYERS. 'auto' → -1, 'all' → -2, otherwise std::stoi.  ·  **default:** `auto (-1). Confirmed in the binary's --help: "(default: auto)".`

SURPRISE: inside libllama, auto (-1) and all (-2) are IDENTICAL — n_gpu_layers() returns hparams.n_layer_all+1 for any negative value (src/llama-model.cpp:1746-1747). The only place the two differ is common/fit.cpp:377, where --fit refuses to act if n_gpu_layers != the default -1. So `-ngl all` = `-ngl auto` + "--fit, hands off the layers". CLAMP: the effective count is min(n_gpu_layers, n_layer_all+1) (src/llama-model.cpp:1355), and is forced to 0 when the device list is empty. The +1 is the output layer. The count is applied from the BACK of the model: i_gpu_start = max(n_layer_all+1-n_gpu_layers, 0) (src/llama-model.cpp:1354) — the last N layers are offloaded, not the first N. If no GPU backend is available the flag prints three warnings and is ignored (common/arg.cpp:2754-2758).

`common/arg.cpp:2742-2760` · `src/llama-model.cpp:1745-1748` · `src/llama-model.cpp:1354-1355`

### `-lm / --load-mode: mmap / mlock / direct-IO as one switch`
**Use:** -lm {auto|none|mmap|mlock|mmap+mlock|dio}. Env LLAMA_ARG_LOAD_MODE.  ·  **default:** `auto (LLAMA_LOAD_MODE_AUTO = -1)`

use_mmap is true for MMAP, MMAP_MLOCK and AUTO only (src/llama-model-loader.cpp:554). use_direct_io is true for DIO only (src/llama-model-loader.cpp:555). use_mlock is true for MLOCK and MMAP_MLOCK (src/llama-model.cpp:1279). THE TRAP: `mlock` alone means mlock WITHOUT mmap. To keep both you must say `mmap+mlock`. AUTO is downgraded to no-mmap if ANY selected device reports !caps.mmap_support (src/llama-model.cpp:1288-1298) — and only under AUTO; an explicit `-lm mmap` is never downgraded there. It is separately downgraded if the platform has no mmap at all (src/llama-model-loader.cpp:817-820). The resolved mode is printed as 'loading model tensors, this can take a while... (load_mode = X)' (src/llama-model.cpp:1300-1305).

`common/arg.cpp:2662-2681` · `include/llama.h:205-212` · `src/llama-model-loader.cpp:554-555` · `src/llama-model.cpp:1279` · `src/llama-model.cpp:1288-1298`

### `--mmap / --no-mmap / --mlock / -dio (all DEPRECATED)`
**Use:** They now just assign load_mode: --mmap→MMAP, --no-mmap→NONE, --mlock→MLOCK, -dio→DIRECT_IO, -ndio→NONE. Each prints a DEPRECATED warning.  ·  **default:** `unset`

They are a single-slot assignment, so `--mlock --no-mmap` and `--no-mmap --mlock` differ, and mixing any of them with -lm makes only the LAST flag on the command line take effect — the code detects this and warns (common/arg.cpp:883-886) but does not error. `--mlock` no longer means 'mmap and also lock': it means load_mode=MLOCK, which turns mmap OFF (src/llama-model-loader.cpp:554).

`common/arg.cpp:2638-2661` · `common/arg.cpp:877-886`

### `--repack / -nr / --no-repack: CPU extra buffer types (runtime weight repacking)`
**Use:** --no-repack disables; --repack re-enables. Env LLAMA_ARG_REPACK. Maps to mparams.use_extra_bufts = !no_extra_bufts.  ·  **default:** `enabled (params.no_extra_bufts = false). Binary --help: "(default: enabled)".`

Repack is a LAYOUT change only — get_alloc_size is nullptr so allocation stays ggml_nbytes (ggml/src/ggml-cpu/repack.cpp:4829). But it forces a REAL host allocation instead of aliasing the mmap (the repack buft is not the device default, so the buffer_from_host_ptr path is skipped at src/llama-model.cpp:1605), and the repack buft reports is_host=nullptr→false (ggml/src/ggml-cpu/repack.cpp:4830), so repacked tensors are also skipped by --load-mode mlock's buffer locking (src/llama-model.cpp:1639). Built in by default: option(GGML_CPU_REPACK ... ON) (ggml/CMakeLists.txt:152). On x86 AVX2 only Q4_0, Q4_K, IQ4_NL and MXFP4 repack (plus Q2_K on AVX512); Q5_K, Q6_K and Q8_0 have NEON/RISC-V paths only (ggml/src/ggml-cpu/repack.cpp:4573-4712). Every case additionally requires ne[1] % 8 == 0.

`common/arg.cpp:2411-2418` · `common/common.h:572` · `common/common.cpp:1669` · `src/llama-model.cpp:944-963`

### `--no-host: drop the pinned host buffer from the CPU buffer list`
**Use:** --no-host (no value). Env LLAMA_ARG_NO_HOST → mparams.no_host.  ·  **default:** `false`

It removes the ggml_backend_dev_host_buffer_type() entry that otherwise sits SECOND in cpu_buft_list, ahead of the extra/repack bufts — which is why the help says it 'allows extra buffers to be used'. INTERACTION: it is a NO-OP for weights whenever mmap is on, because the loader independently demotes any host buffer type back to plain CPU under mmap (src/llama-model-loader.cpp:1212-1221).

`common/arg.cpp:2419-2425` · `src/llama-model.cpp:935-942` · `common/common.cpp:1670`

### `--check-tensors: validate every tensor's rows at load`
**Use:** --check-tensors (no value, no env).  ·  **default:** `false`

Runs ggml_validate_row_data per tensor on std::async threads; a failure throws 'found tensors with invalid data' AFTER the whole load. INTERACTION: it disables the async pinned-memory upload fast path outright — `if (use_mmap || check_tensors) return nullptr;` (src/llama-model-loader.cpp:1459-1462). Combining --check-tensors with -lm none costs you both the validation time and the fast upload.

`common/arg.cpp:2882-2888` · `src/llama-model-loader.cpp:1556-1560` · `src/llama-model-loader.cpp:1673-1684`

### `--override-kv: rewrite scalar GGUF metadata at load`
**Use:** --override-kv KEY=TYPE:VALUE, comma-separated, repeatable. TYPE ∈ {int,float,bool,str}.  ·  **default:** `none`

CLAMPS: key must be < 128 bytes before the '=' (common/common.cpp:683); str values are capped at 127 chars with an explicit error (common/common.cpp:715-719). bool accepts only the literals 'true'/'false' (common/common.cpp:701-707). It can INJECT a key that is absent from the file — try_override runs before the `k < 0` check (src/llama-model-loader.cpp:255-261). A tag/type mismatch is a WARNING only; the override is dropped and the file value wins (src/llama-model-loader.cpp:209-212). The metadata dump printed at load explicitly says overrides are NOT reflected in it (src/llama-model-loader.cpp:787).

`common/arg.cpp:2889-2900` · `common/common.cpp:681-724` · `src/llama-model-loader.cpp:414-422` · `src/llama-model-loader.cpp:255-262`

### `-fit / --fit: auto-fit ngl, tensor_split and ctx to free device memory`
**Use:** -fit on|off. Env LLAMA_ARG_FIT. Companions: -fitt/--fit-target MiB per device, -fitc/--fit-ctx minimum ctx, -fitp/--fit-print.  ·  **default:** `ON (common/common.h:468). -fitt default 1024 MiB per device (common/common.h:473 = 1024*1024*1024 bytes). -fitc default 4096 (common/common.h:470). -fitp default off.`

It probes by loading the model with no_alloc=true and load_mode=NONE, then reading ggml_backend_dev_memory() for free VRAM AT THAT MOMENT (common/fit.cpp:56-57, common/fit.cpp:111-116) — so its answer moves with whatever else is holding VRAM at launch. It writes its OWN -ot patterns to spill part of a boundary layer: `blk\.<il>\.ffn_(gate|up|gate_up|down).*` variants and `blk\.<il>\.ffn_(up|down|gate_up|gate)_(ch|)exps` (common/fit.cpp:405-441). At most 1000 layers (common/fit.cpp:407-409). CLAMP: fitted n_ctx is rounded DOWN to a multiple of 256 and floored at -fitc (common/fit.cpp:340). All failures are swallowed into a WARN + status FAILURE (common/fit.cpp:806-809) — the process still starts, with whatever partial mutations already landed.

`common/arg.cpp:2822-2881` · `common/common.h:468-473` · `common/common.cpp:1294-1303` · `common/fit.cpp:176-198` · `common/fit.cpp:404-500`

### `-sm / --split-mode, -ts / --tensor-split, -mg / --main-gpu`
**Use:** -sm {none,layer,row,tensor}; -ts N0,N1,... (comma or slash separated); -mg INDEX.  ·  **default:** `-sm layer (LLAMA_SPLIT_MODE_LAYER), -ts all zeros → even split, -mg 0`

CLAMP: -ts rejects more entries than llama_max_devices() (=16, src/llama.cpp:85-87); entries past the list are zero-filled (common/arg.cpp:2799-2805). Split proportions are normalised to a cumulative distribution (common/fit.cpp / src/llama-model.cpp:1343-1352). -sm none KEEPS ONLY devices[main_gpu] and errors if main_gpu is out of range; main_gpu<0 clears the device list entirely, i.e. CPU-only (src/llama.cpp:288-300). -sm row is the only mode that adds a split buffer type, and it THROWS if the backend has no ggml_backend_split_buffer_type (src/llama-model.cpp:976-999).

`common/arg.cpp:2761-2821` · `src/llama-model.cpp:976-1021` · `src/llama.cpp:288-300`

### `-dev / --device and --list-devices`
**Use:** -dev CUDA0,CUDA1 (or `none`). --list-devices prints and exits. Env LLAMA_ARG_DEVICE.  ·  **default:** `all available devices`

Integrated GPUs are only added when no discrete GPU was found, and RPC servers do not count as discrete (src/llama.cpp:277-285). Verified on this box: `--list-devices` → 'CUDA0: NVIDIA GeForce RTX 4070 SUPER (12281 MiB, 11069 MiB free)' — one device only.

`common/arg.cpp:2699-2713` · `src/llama.cpp:275-310`

### `-ctk / -ctv (and -ctkd / -ctvd for the draft): KV cache element type`
**Use:** -ctk TYPE / -ctv TYPE. Env LLAMA_ARG_CACHE_TYPE_K / _V.  ·  **default:** `f16 for both (llama_context_default_params type_k/type_v)`

CLAMP: the whitelist is exactly nine types — f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 (common/arg.cpp:305-315). Anything else throws 'Unsupported cache type: X'. Note q4_K/q6_K/mxfp4 are NOT offerable here even though they are valid weight types. This is a CONTEXT parameter, not a model-loading one: it never touches the weight tensors.

`common/arg.cpp:2426-2450` · `common/arg.cpp:305-332` · `common/arg.cpp:4022-4047`

### `--numa TYPE`
**Use:** --numa {distribute,isolate,numactl}. Env LLAMA_ARG_NUMA.  ·  **default:** `disabled`

The loader queries ggml_backend_cpu_is_numa and passes it into llama_mmap, where it suppresses MAP_POPULATE and switches madvise to POSIX_MADV_RANDOM — Linux only. The Windows llama_mmap ctor takes `numa` and immediately GGML_UNUSEDs it (src/llama-mmap.cpp:536-537), so --numa has zero effect on mmap here.

`common/arg.cpp:2683-2698` · `src/llama-model-loader.cpp:1354-1362` · `src/llama-mmap.cpp:451-471`

### `llama-quantize: whole-file ftype (offline, changes the file)`
**Use:** llama-quantize [flags] model-f32.gguf [model-quant.gguf] TYPE [nthreads]. TYPE accepts the name (case-insensitive) or its numeric ftype.  ·  **default:** `nthread 0 → std::thread::hardware_concurrency()`

37 named targets including Q1_0, Q2_0, MXFP4_MOE, NVFP4, TQ1_0/TQ2_0 and the IQ family. 'COPY' is deliberately placed after F32 in the table so numeric ftype 0 resolves to F32, not COPY (tools/quantize/quantize.cpp:73). NOT PRESENT in the staged directory C:\AI\llama.cpp-dflash2 — only llama-server.exe and llama-cli.exe ship there.

`tools/quantize/quantize.cpp:33-76` · `tools/quantize/quantize.cpp:90-115` · `src/llama-quant.cpp:1310-1327`

### `--token-embedding-type / --output-tensor-type: quantise embeddings and LM head separately`
**Use:** llama-quantize --token-embedding-type <ggml_type> / --output-tensor-type <ggml_type>  ·  **default:** `GGML_TYPE_COUNT = 'unset' for both (src/llama-quant.cpp:1315-1316)`

PRECEDENCE: these two are checked FIRST in llama_tensor_get_type (src/llama-quant.cpp:683-688) and return immediately — they outrank --tensor-type AND they skip tensor_type_fallback, so a block-size-incompatible choice is not clamped for these two tensors the way it is everywhere else (src/llama-quant.cpp:717). When unset, the built-in mixture forces the head to Q8_0 for Falcon or when nx % block_size != 0, Q5_K for the IQ1/IQ2/IQ3_XXS ftypes, else Q6_K (src/llama-quant.cpp:456-471). For tied-embedding models the token embedding is quantised with the OUTPUT rule, not the embedding rule (src/llama-quant.cpp:452-454).

`tools/quantize/quantize.cpp:410-427` · `src/llama-quant.cpp:683-688` · `src/llama-quant.cpp:452-471` · `src/llama-quant.cpp:487-499`

### `--tensor-type / --tensor-type-file: arbitrary per-tensor quant overrides`
**Use:** --tensor-type <name-substring-or-regex>=<ggml_type>, repeatable; --tensor-type-file reads the same tokens whitespace-separated from a file. Example from the usage text: --tensor-type attn_q=q8_0  ·  **default:** `none`

The name is LOWERCASED at parse (tools/quantize/quantize.cpp:332) and then used as a std::regex with regex_search over the tensor name — so it is a regex, not a literal, despite the docs calling it a name. Patterns are compiled once in the quantize_state ctor (src/llama-quant.cpp:190-195). First match wins. SETTING A MATCH SUPPRESSES THE MIXTURE for that tensor (`manual=true` → llama_tensor_get_type_impl is skipped, src/llama-quant.cpp:712-715), which is the same effect --pure has globally. The result still passes through tensor_type_fallback (src/llama-quant.cpp:717).

`tools/quantize/quantize.cpp:314-361` · `src/llama-quant.cpp:184-196` · `src/llama-quant.cpp:693-718`

### `--pure: no k-quant mixture`
**Use:** llama-quantize --pure  ·  **default:** `false`

Only skips llama_tensor_get_type_impl. It does NOT bypass tensor_allows_quantization (norms, ffn_gate_inp, 1-D tensors stay unquantised), does not bypass the token-embedding/output early returns, and does not bypass tensor_type_fallback.

`tools/quantize/quantize.cpp:448-449` · `src/llama-quant.cpp:712-715`

### `--leave-output-tensor`
**Use:** llama-quantize --leave-output-tensor → params.quantize_output_tensor = false  ·  **default:** `quantize_output_tensor = true (src/llama-quant.cpp:1318)`

Implemented as one clause in tensor_allows_quantization: `quantize &= params->quantize_output_tensor || name != "output.weight"` — an exact string compare on `output.weight`. It does not protect a tied token_embd.weight acting as the head, nor any arch whose head has a different tensor name.

`tools/quantize/quantize.cpp:408-409` · `src/llama-quant.cpp:304`

### `tensor_type_fallback: the silent shape clamp`
**Use:** Automatic. Applies whenever ne[0] is not divisible by the target type's block size.  ·  **default:** `n/a`

THE most surprising clamp in the quantiser: it downgrades silently-ish (a WARN line) — IQ1_S/IQ1_M/IQ2_*/IQ3_*/IQ4_XS → IQ4_NL; Q2_0/Q2_K/Q3_K/TQ1_0/TQ2_0 → Q4_0; Q4_K → Q5_0; Q5_K → Q5_1; Q6_K → Q8_0. If the fallback is still incompatible it goes to F16 (src/llama-quant.cpp:407-414). Any other target type throws 'no tensor type fallback is defined'. Every hit increments qs.n_fallback.

`src/llama-quant.cpp:374-420`

### `--allow-requantize, --only-copy (COPY), --dry-run, --keep-split, --prune-layers, --imatrix + --include/--exclude-weights`
**Use:** llama-quantize flags; COPY as the TYPE argument sets only_copy.  ·  **default:** `allow_requantize=false, only_copy=false, keep_split=false, dry_run=false, prune_layers=nullptr (src/llama-quant.cpp:1313-1326)`

Without --allow-requantize, an already-quantised source tensor raises an error rather than being dequantised (src/llama-quant.cpp:1234-1236). only_copy short-circuits tensor_allows_quantization first thing (src/llama-quant.cpp:291) and the header comment states it makes ftype, allow_requantize and quantize_output_tensor all ignored (include/llama.h:432). --prune-layers renumbers surviving blk.N via remap_layer (src/llama-quant.cpp:48-70) — a genuine way to shrink a model, offline. --include-weights and --exclude-weights are mutually exclusive (usage note, tools/quantize/quantize.cpp:159).

`tools/quantize/quantize.cpp:436-470` · `src/llama-quant.cpp:1234-1237` · `src/llama-quant.cpp:945-1003` · `src/llama-quant.cpp:291`

### `mmap / prefetch / mlock mechanics on Windows`
**Use:** Governed by --load-mode. init_mappings is always called with prefetch=true.  ·  **default:** `AUTO → mmap on, prefetch on`

init_mappings passes `prefetch ? -1 : 0` as a size_t, i.e. SIZE_MAX (src/llama-model-loader.cpp:1363). The Windows impl maps the WHOLE file (MapViewOfFile(...,0,0), src/llama-mmap.cpp:552) and PrefetchVirtualMemory's min(size, SIZE_MAX) = the entire file (src/llama-mmap.cpp:558-572). So `-lm mmap`/auto pulls the whole GGUF into the page cache at load. mlock on Windows is VirtualLock with an automatic SetProcessWorkingSetSize retry of len+1 MiB, giving up after two tries with a warning (src/llama-mmap.cpp:687-712). Lock granularity is the system page size (src/llama-mmap.cpp:682-686).

`src/llama-model-loader.cpp:1348-1373` · `src/llama-mmap.cpp:536-584` · `src/llama-mmap.cpp:681-720`

### `Async pinned-memory upload path (the fast non-mmap load)`
**Use:** No flag. Engages automatically when the tensor's buffer is a device default buft on a device advertising caps.async + host_buffer + events.  ·  **default:** `4 staging buffers; 1 MiB each when read_alignment()==1, 64 MiB + 2*alignment when direct-IO alignment applies`

Gated off by `use_mmap || check_tensors`. On Windows read_alignment() is always 1 (src/llama-mmap.cpp:387-391; the Windows llama_file::impl never assigns `alignment`), so the buffer is 1 MiB, never the 64 MiB NVMe-tuned size.

`src/llama-model-loader.cpp:1444-1533` · `src/llama-model-loader.cpp:1459-1462`

### `Tied embeddings duplicate the tensor into two buffers`
**Use:** No flag. Happens when the GGUF has no output.weight — the arch code re-creates token_embd with TENSOR_DUPLICATED.  ·  **default:** `n/a`

The duplicate is re-tagged LLM_TENSOR_OUTPUT and therefore goes to dev_output (GPU when offloaded), while the original stays on dev_input (always CPU). Both are loaded from the same file offset, so a tied embedding matrix is resident twice: once in the mmap/CPU buffer and once in VRAM. This is a real VRAM line item that no flag removes short of -ot on the output tensor.

`src/models/qwen3.cpp:21-26` · `src/models/qwen3moe.cpp:20-25` · `src/llama-model-loader.cpp:1110-1114` · `src/llama-model-loader.h:66-69`

### `select_weight_buft: silent fallback when a device cannot run a weight's op`
**Use:** Automatic. Each tensor's buffer type is the FIRST entry in its device's buft list for which ggml_backend_dev_supports_op returns true for the tensor's canonical op.  ·  **default:** `n/a`

Because each GPU's buft list has the whole cpu_buft_list appended as a fallback (src/llama-model.cpp:1310-1313), a tensor the GPU cannot handle silently lands on the CPU. The only trace is a DEBUG-level line naming the FIRST moved tensor and a count of the rest (src/llama-model-loader.cpp:1341-1345) — invisible at default verbosity. If even the CPU list fails: 'failed to find a compatible buffer type for tensor X' (src/llama-model-loader.cpp:1207).

`src/llama-model-loader.cpp:1053-1065` · `src/llama-model-loader.cpp:960-1051` · `src/llama-model-loader.cpp:1204-1210` · `src/llama-model-loader.cpp:1341-1345`

**What this area CANNOT do.** NOTHING IN THIS AREA CAN RE-QUANTISE A WEIGHT AT LOAD TIME. There is no code path from any CLI flag to a change of a tensor's ggml_type; llama_model_loader only places tensors in buffers. The single load-time transformation is CPU repack, and it is layout-only — the repack buft leaves get_alloc_size as nullptr, so allocation stays exactly ggml_nbytes (ggml/src/ggml-cpu/repack.cpp:4828-4829). Resident weight bytes therefore always equal file bytes for every tensor. Per-tensor quantisation exists only offline in llama-quantize, which writes a new file, and that binary is not in the staged set at C:\AI\llama.cpp-dflash2. token_embd can never be offloaded by -ngl. LLM_TENSOR_TOKEN_EMBD is LLM_TENSOR_LAYER_INPUT (src/llama-arch.cpp:672) and dev_input is hard-wired to the CPU device with the comment "there is very little benefit to offloading the input layer, so always keep it on the CPU" (src/llama-model.cpp:1368-1370). Only an explicit -ot pattern moves it. -ot cannot name a non-default buffer type. The name→buft map is built solely from ggml_backend_dev_buffer_type() per device (common/arg.cpp:256-263). On this machine that is exactly {CPU, CUDA0} — verified by running the staged binary with a bogus name. CUDA_Host, the repack buft and split bufts are unreachable from the command line. -ot ...=CPU does not guarantee the plain CPU buffer. src/llama-model-loader.cpp:1183-1185 re-runs select_weight_buft over the entire cpu_buft_list when the override target equals ggml_backend_cpu_buffer_type(). --no-host does nothing for weights while mmap is on. src/llama-model-loader.cpp:1212-1221 unconditionally demotes any host buffer type back to plain CPU under mmap, after --no-host has already had its say. --no-repack does not disable GPU extra buffer types. use_extra_bufts is consulted only inside make_cpu_buft_list (src/llama-model.cpp:944-963); make_gpu_buft_list appends the device's extra bufts unconditionally (src/llama-model.cpp:1006-1020). --load-mode dio is inert on Windows. The Windows llama_file::impl ctor marks use_direct_io [[maybe_unused]] and ignores it (src/llama-mmap.cpp:86-94); the O_DIRECT open is inside #ifdef __linux__ (src/llama-mmap.cpp:181-199). It still switches mmap OFF (src/llama-model-loader.cpp:554-555), so on Windows `-lm dio` == `-lm none` plus a misleading log line. Windows has_direct_io() nevertheless returns a hardcoded true (src/llama-mmap.cpp:173-175) while read_alignment() stays 1 (src/llama-mmap.cpp:387-391). mmap page release after offload does not happen on Windows. load_all_data's final "unmap offloaded tensors and metadata" step (src/llama-model-loader.cpp:1686-1697) calls llama_mmap::unmap_fragment, whose Windows implementation is an empty function with two GGML_UNUSEDs (src/llama-mmap.cpp:580-584). Whatever was pulled in by the whole-file PrefetchVirtualMemory (src/llama-mmap.cpp:558-572) stays mapped for the process's life. Resident host memory under mmap is therefore ≈ the full GGUF, offloaded or not. --override-kv cannot touch array-valued metadata. get_arr and get_arr_n never consult kv_overrides (src/llama-model-loader.cpp:276-292 and 305-410) — only get_key does (src/llama-model-loader.cpp:414-422). llama_model_kv_override itself has no array tag (include/llama.h:288-300). Key and str value are each capped at 127 chars (common/common.cpp:683, 715-719). At most 4096 tensor buffer overrides. llama_max_tensor_buft_overrides() returns a hardcoded 4096 (src/llama.cpp:89-91) and common/arg.cpp:946-950 pads the vector to exactly that; supplying ≥4096 -ot patterns leaves back().pattern non-null and trips GGML_ASSERT "Tensor buffer overrides not terminated with empty pattern" (common/common.cpp:1682). --fit throws rather than truncating if it needs more (common/fit.cpp:477-484). --fit will not fit more than 1000 layers (common/fit.cpp:407-409), is not implemented for -sm tensor (common/fit.cpp:183) or for multi-GPU -sm row (common/fit.cpp:390-392), and refuses if the user already set tensor_split (common/fit.cpp:383-389). --tensor-type and the whole k-quant mixture are skipped when the target ftype is not a quantised type. src/llama-quant.cpp:693 guards the entire block with `if (ggml_is_quantized(default_type))`, so `llama-quantize --tensor-type attn_q=q8_0 model.gguf F16` silently ignores the override. --token-embedding-type and --output-tensor-type bypass the shape clamp. They return at src/llama-quant.cpp:684 and :687, before tensor_type_fallback runs at :717 — an incompatible block size is not corrected for those two tensors the way it is for every other.

**Exists but unused in our profile.** A profile of `--spec-type ngram-mod, -ctk q4_0 -ctv q4_0, -ngl auto --fit on, -np 1` on one CUDA0 leaves most of this area cold: TENSOR PLACEMENT — -ot, -cmoe, -ncmoe are all unset, so params.tensor_buft_overrides is the 4096-entry all-null pad (common/arg.cpp:946-950) and --fit owns the whole override array. This is the ONLY configuration in which --fit's layer fitting runs at all: setting any one of the three makes common/fit.cpp:396-398 throw "model_params::tensor_buft_overrides already set by user, abort", after which fitting is abandoned — but note the context reduction at common/fit.cpp:310-345 has ALREADY mutated cparams->n_ctx in place by then, and common/fit.cpp:806-809 only downgrades this to a WARN. So adding -cmoe to this profile silently converts --fit from "fit layers and ctx" into "reduce ctx only". DRAFT-SIDE OVERRIDES — ngram-mod is a draft-model-free speculation type, so -otd/-cmoed/-ncmoed and -ctkd/-ctvd (common/arg.cpp:4022-4074) are never reached, and params.speculative.draft.tensor_buft_overrides stays empty and unterminated (common/arg.cpp:952-954). LOAD MODE — load_mode stays AUTO, which means mmap ON and the AUTO-only mmap_support downgrade at src/llama-model.cpp:1288-1298 is live. Untouched: -lm none/mmap/mlock/mmap+mlock/dio, and the deprecated --mmap/--no-mmap/--mlock/-dio last-flag-wins hazard (common/arg.cpp:877-886). Because mmap is on, three further paths are dead: the async pinned-memory upload (src/llama-model-loader.cpp:1459-1462), any effect of --no-host (src/llama-model-loader.cpp:1212-1221), and llama_mlock entirely (src/llama-model.cpp:1279). REPACK / EXTRA BUFTS — --no-repack unset means use_extra_bufts=true, so the CPU repack buft IS in cpu_buft_list. With -ngl auto + --fit succeeding, few or no weight tensors land on the CPU, so repack likely never fires; if --fit spills a partial layer to ggml_backend_cpu_buffer_type() (common/fit.cpp:490-492) those tensors do go through the CPU list and may repack — Q4_K and Q4_0 do on AVX2, Q5_K/Q6_K/Q8_0 do not (ggml/src/ggml-cpu/repack.cpp:4600-4712). MULTI-GPU — -sm (all four modes), -ts, -mg, the split buffer type (src/llama-model.cpp:976-999) and llama_max_devices()=16 are all inert with a single CUDA0. -fitt's per-device broadcast collapses to one value. INTEGRITY / METADATA — --check-tensors and --override-kv unset; ggml_validate_row_data never runs and kv_overrides is empty (mparams.kv_overrides = NULL, common/common.cpp:1672-1673). QUANTISATION — the entire llama-quantize surface (ftype table, --pure, --tensor-type, --token-embedding-type, --output-tensor-type, --leave-output-tensor, --allow-requantize, --prune-layers, --keep-split, --imatrix, --dry-run, tensor_type_fallback) is offline-only and the tool is not even staged in C:\AI\llama.cpp-dflash2. Whatever quant the GGUF already carries is what is loaded. KV CACHE — -ctk/-ctv q4_0 ARE exercised, and are within the nine-type whitelist (common/arg.cpp:305-315). -np 1 means n_seq_max=1, which is a context-sizing input to --fit's probe but not a model-loading capability. NUMA — --numa unset, and would be a no-op on Windows anyway (src/llama-mmap.cpp:536-537).

## Speculative decoding (llama.cpp build 10499, commit 1deefcca3 = PR #27342 "DFlash2" on master). Source tree C:\AI\llama.cpp; all line numbers are from that tree.

### `--spec-type <csv> — selects which speculators are built`
**Use:** --spec-type ngram-mod / --spec-type ngram-mod,draft-simple (comma separated; env LLAMA_ARG_SPEC_TYPE). Accepted values, exactly these 11 strings: none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache.  ·  **default:** `none (common.h:370 `types = { COMMON_SPECULATIVE_TYPE_NONE }`) — speculation is OFF unless you ask for it or a draft/sidecar is auto-detected`

The handler APPENDS (`types.insert(types.end(), ...)`, arg.cpp:4160) — it never replaces. Passing --spec-type twice accumulates both lists, and arg.cpp:828 / arg.cpp:1221 deliberately suppress the usual 'specified multiple times' warning for this one flag. If any name in the list is `none`, the parser returns exactly {NONE} and discards everything else in that same invocation (speculative.cpp:2286-2288) — but a later --spec-type will append to it. An unknown name throws std::invalid_argument (speculative.cpp:2292) and aborts startup.

`common/arg.cpp:4153-4162` · `common/speculative.cpp:34-46 (name->enum map)` · `common/speculative.cpp:2262-2277 (enum->name)` · `common/speculative.cpp:2279-2296 (parser)` · `common/common.h:170-183 (enum)`

### `What each --spec-type value does`
**Use:** draft-simple = separate draft model, autoregressive N-step draft. draft-eagle3 = EAGLE3 sidecar, needs exactly 3 extract layers. draft-mtp = multi-token-prediction head (target model itself or a separate head; three sub-modes: shared-memory/gemma4, chained heads/step35, single head/qwen35). draft-dflash = DFlash diffusion block denoiser. draft-dspark = DFlash + Markov head (anchor-first layout). ngram-simple / ngram-map-k / ngram-map-k4v = self-speculation over the prompt via n-gram->m-gram lookup. ngram-mod = self-speculation via a 4M-entry hash of the last n_match tokens. ngram-cache = the old lookup-decoding 3-level n-gram cache.

draft-dspark is implemented by the SAME class as draft-dflash, constructed with a type tag (speculative.cpp:2577-2581). ngram-map-k and ngram-map-k4v are the same class too, distinguished only by `key_only` (speculative.cpp:2216: key_only == (type==NGRAM_MAP_K)). DFlash/DSpark force `llama_set_causal_attn(ctx_dft,false)` (speculative.cpp:1036) — that draft context is non-causal for its whole life.

`common/speculative.cpp:179-389 (draft_simple)` · `common/speculative.cpp:426-908 (eagle3)` · `common/speculative.cpp:910-1347 (dflash/dspark)` · `common/speculative.cpp:1349-1785 (mtp)` · `common/speculative.cpp:1788-1831 (ngram_simple)` · `common/speculative.cpp:1833-1885 (ngram_map_k / k4v)` · `common/speculative.cpp:1887-2060 (ngram_mod)` · `common/speculative.cpp:2062-2199 (ngram_cache)`

### `Auto-detection of the speculative type from the draft GGUF / repo sidecar`
**Use:** Give only -md/--spec-draft-model or -hfd and leave --spec-type at its default; the type is inferred.

Sidecar precedence when several are shipped: mtp > dspark > dflash > eagle3 (arg.cpp:546-561). GGUF sniff: arch=='dflash' + tensor `markov_w1.weight` => draft-dspark, else draft-dflash; any other arch with `blk.<n_last>.nextn.eh_proj.weight` => draft-mtp; otherwise nothing is inferred. arg.cpp:565 warns in-source: it reads ONLY the first split, so a sharded draft needs an explicit --spec-type. Detection only fires while types is still exactly {NONE}.

`common/arg.cpp:358-360 (spec_types_is_default)` · `common/arg.cpp:544-562 (sidecar precedence)` · `common/arg.cpp:564-571 (GGUF sniff)` · `common/speculative.cpp:2306-2341`

### `--spec-draft-n-max — draft length for every draft-model-based type`
**Use:** --spec-draft-n-max N (env LLAMA_ARG_SPEC_DRAFT_N_MAX). Applies to draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark. Does NOT apply to any ngram-* type.  ·  **default:** `3 (common.h:325)`

Rejects N<0 with std::invalid_argument (arg.cpp:4080-4082); N=0 is accepted and means 'never draft'. Three separate clamps can silently lower it after parsing: the DFlash block-size clamp (speculative.cpp:990-996), the MTP chained-head clamp (speculative.cpp:1446), and the per-call slot budget dp.n_max (server-context.cpp:2938, applied at speculative.cpp:2728-2732).

`common/arg.cpp:4076-4085` · `common/common.h:325` · `common/speculative.cpp:351-356 (draft-simple stop)` · `common/speculative.cpp:813-817 (eagle3 stop)` · `common/speculative.cpp:1718-1722 (mtp stop)` · `common/speculative.cpp:1181 (dflash block size)`

### `--spec-draft-n-min — discard a draft that came out too short`
**Use:** --spec-draft-n-min N (env LLAMA_ARG_SPEC_DRAFT_N_MIN).  ·  **default:** `0 (common.h:326)`

NO validation at all on this flag — negative values are accepted and compared as `result.size() < (size_t) params.n_min`, i.e. a negative n_min casts to a huge size_t and would clear EVERY draft. The check is 'shorter than n_min => throw the whole draft away', not 'truncate'. Clamped down alongside n_max by the DFlash block-size clamp (speculative.cpp:995).

`common/arg.cpp:4086-4092` · `common/speculative.cpp:380-383 (draft-simple)` · `common/speculative.cpp:842-844 (eagle3)` · `common/speculative.cpp:1765-1767 (mtp)` · `common/speculative.cpp:1276-1281 and 1338-1340 (dflash/dspark)`

### `--spec-draft-p-min — confidence early-stop for the draft`
**Use:** --spec-draft-p-min P (alias --draft-p-min, env LLAMA_ARG_SPEC_DRAFT_P_MIN). Stop extending the draft once the top candidate's probability falls below P.  ·  **default:** `0.00 (common.h:329) — early-stop disabled by default`

Parsed with std::stof and NEVER range-checked or clamped — p_min > 1.0 kills every draft, negative is a no-op. Semantics differ per type: draft-simple/eagle3/mtp/dflash compare `cur_p->data[0].p`; dspark compares a confidence scalar read out of the nextn embedding; DFlash2 compares the softmax of the selector lattice at the argmax (greedy) or the sampled entry's probability (temperature>0). Ignored entirely by every ngram-* type.

`common/arg.cpp:4101-4107` · `common/speculative.cpp:337-342 (draft-simple)` · `common/speculative.cpp:797-804 (eagle3)` · `common/speculative.cpp:1704-1709 (mtp)` · `common/speculative.cpp:1328-1330 (dflash greedy)` · `common/speculative.cpp:1287-1295 (dspark, reads the confidence row from llama_get_embeddings_nextn)` · `common/speculative.cpp:1254-1256 and 1262-1271 (dflash2 selector; the p_min added by this very commit)`

### `--spec-draft-p-split`
**Use:** --spec-draft-p-split P (alias --draft-p-split, env LLAMA_ARG_SPEC_DRAFT_P_SPLIT).  ·  **default:** `0.10 (common.h:328)`

DEAD OUTSIDE ONE EXAMPLE. The only read of params.speculative.draft.p_split in the whole tree is examples/speculative/speculative.cpp:67 (the old tree-drafting demo binary). common/speculative.cpp never reads it and neither does the server. Setting it on llama-server changes nothing.

`common/arg.cpp:4094-4100` · `examples/speculative/speculative.cpp:67`

### `--spec-draft-backend-sampling / --no-spec-draft-backend-sampling — GPU-side sampling for the DRAFT path`
**Use:** --spec-draft-backend-sampling / --no-spec-draft-backend-sampling (env LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING).  ·  **default:** `ENABLED (common.h:331 `backend_sampling = true`) — note this is the opposite of the main path's default`

The chain attached to the draft context is hardcoded `top_k(10)` only (speculative.cpp:502, 1018, 1428) — it is not your sampler and not configurable. Because a top_k-only backend chain emits candidates+logits but never `data.sampled` or `data.probs` (src/llama-sampler.cpp:1477-1501, src/llama-graph.cpp:3720-3742), llama_get_sampled_token_ith returns NULL and the CPU chain still runs (common/sampling.cpp:626-643) — so this flag only pre-truncates the draft logits to top-10 on the GPU. draft-simple is the one draft type that NEVER attaches a backend chain (no such code in its ctor at speculative.cpp:186-252). DFlash2 opts out: the condition is `backend_sampling && !is_dflash2` (speculative.cpp:1015). If llama_set_sampler fails, it warns 'backend offload failed for seq_id=%d; using CPU sampler' and falls back (speculative.cpp:505, 1021, 1431). It is refused outright under -sm row / LLAMA_SPLIT_MODE_TENSOR (src/llama-context.cpp:1216-1227).

`common/arg.cpp:4108-4116` · `common/common.h:331` · `common/speculative.cpp:498-511 (eagle3)` · `common/speculative.cpp:1013-1027 (dflash/dspark)` · `common/speculative.cpp:1423-1437 (mtp)` · `src/llama-context.cpp:1209-1258 (llama_context::set_sampler)`

### `-bs / --backend-sampling — GPU-side sampling for the MAIN (target) path`
**Use:** -bs or --backend-sampling (env LLAMA_ARG_BACKEND_SAMPLING). Positive-only flag, no --no- form.  ·  **default:** `disabled (common.h:295 `backend_sampling = false`)`

Unlike the draft path this uses the request's real sampler chain (server-context.cpp:1741 passes common_sampler_get(slot.smpl)). Self-disables with a warning when a grammar is active (sampling.cpp:421-425) or a reasoning budget is active (sampling.cpp:427-431), and is silently skipped when the request needs pre-sampling probs, i.e. n_probs>0 && !post_sampling_probs (server-context.cpp:1732-1737). With speculation on, the verify batch produces n_draft+1 outputs for one sequence; llama_decode returns -1 if that exceeds cparams.n_outputs_max_per_seq (src/llama-context.cpp:1683-1689), which is exactly what the output-limit plumbing below exists to prevent.

`common/arg.cpp:2295-2301` · `common/common.h:295` · `common/common.cpp:1369-1372` · `tools/server/server-context.cpp:1732-1744` · `common/sampling.cpp:421-431` · `src/llama-context.cpp:1216-1227` · `src/llama-context.cpp:1681-1689`

### `Where the draft model's KV cache and cache type come from`
**Use:** --spec-draft-type-k / -ctkd / --cache-type-k-draft TYPE and --spec-draft-type-v / -ctvd / --cache-type-v-draft TYPE (env LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K / _V). Allowed: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1.  ·  **default:** `f16 for both (common.h:340-341)`

THE MAIN MODEL'S -ctk/-ctv DOES NOT PROPAGATE. common_base_params_to_speculative overwrites result.cache_type_k/v with the speculative struct's own fields unconditionally (speculative.cpp:2405-2406), even when there is no draft model. So `-ctk q4_0 -ctv q4_0 -md draft.gguf` gives you a q4_0 target cache and an f16 draft cache unless you also pass -ctkd/-ctvd. These two flags carry no set_examples() call, so unlike most --spec-draft-* flags they are accepted by every binary.

`common/arg.cpp:4022-4047` · `common/common.h:340-341` · `common/speculative.cpp:2405-2406` · `src/llama-context.cpp:385-396`

### `Where the draft context lives (its own llama_context, own memory)`
**Use:** Created once at load time; you do not address it directly.

The draft always gets its OWN memory module, i.e. its own KV cache, sized from the inherited n_ctx/n_parallel. cparams.ctx_other is set to ctx_tgt for every draft context (speculative.cpp:2461) but llama_context only keeps it for LLM_ARCH_GEMMA4_ASSISTANT, or for EAGLE3/DFLASH sidecars that ship without tok_embd/output (llama-context.cpp:145-161); only in those cases is the target's memory shared (llama-context.cpp:392). The draft ctx always sets n_rs_seq=0 (speculative.cpp:2460), so rollback on the draft side goes through checkpoints. With --spec-type draft-mtp and no -md, no second model is loaded — a second context on the target model with ctx_type=LLAMA_CONTEXT_TYPE_MTP is created instead (speculative.cpp:2483-2494).

`common/speculative.cpp:2432-2496 (common_speculative_init_result)` · `common/speculative.cpp:2460-2461 (n_rs_seq=0, ctx_other=ctx_tgt)` · `common/speculative.cpp:2464-2482 (draft model + ctx)` · `common/speculative.cpp:2483-2495 (MTP-on-target-model ctx)` · `src/llama-context.cpp:142-161 (when ctx_other is actually honoured)` · `src/llama-context.cpp:385-396 (memory module)`

### `Where the draft sampler lives`
**Use:** Not configurable.

One common_sampler per sequence, built on the DRAFT model's vocab, hardcoded to `samplers = {TOP_K}` with `top_k = 10` and `no_perf = false`. The only variation: DFlash2 uses `top_k = selector_top_k` read from the GGUF instead of 10 (speculative.cpp:1005). common_sampler_init always appends llama_sampler_init_dist(seed) at the end (sampling.cpp:405), which is what fills in the `.p` values that p_min then tests. There is a commented-out block at speculative.cpp:209-224 showing the intent to make this configurable; it is not.

`common/speculative.cpp:226-236 (draft-simple)` · `common/speculative.cpp:488-495 (eagle3)` · `common/speculative.cpp:1001-1008 (dflash/dspark)` · `common/speculative.cpp:1414-1421 (mtp)` · `common/sampling.cpp:346-406 (chain assembly)`

### `The DFlash/DSpark block-size clamp`
**Use:** Nothing to pass — it fires automatically from the draft GGUF metadata.  ·  **default:** `block_size = 16 when the model has no `dflash.block_size` key (speculative.cpp:967)`

n_draft_max = (is_dspark && sample_from_anchor) ? block_size : block_size - 1. If n_max or n_min exceeds it, BOTH are clamped with LOG_WRN 'requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d'. Metadata keys read: dflash.block_size (atoi, default 16), dflash.sample_from_anchor (string compare to "true", default true), dflash.selector_top_k (atoi; >0 turns on the DFlash2 path, is_dflash2). Consequence: with a stock 16-wide DFlash sidecar the largest usable --spec-draft-n-max is 15 (16 for DSpark with anchor sampling).

`common/speculative.cpp:966-980 (metadata read)` · `common/speculative.cpp:988-996 (the clamp)` · `common/speculative.cpp:1183 (block layout)` · `common/speculative.cpp:2413-2427 (batch/ubatch/output widening)`

### `The MTP chained-head clamp`
**Use:** Automatic when the draft has more than one nextn layer and does not share the target's memory.

chain_heads = n_mtp_layers > 1 && !is_mem_shared; when true, `params.n_max = std::min(params.n_max, n_mtp_layers)` — SILENT, no warning line. One trained head per draft step, so you cannot draft deeper than the head count.

`common/speculative.cpp:1396 (n_mtp_layers)` · `common/speculative.cpp:1442-1452`

### `The per-call draft budget dp.n_max (context/remaining clamp)`
**Use:** Automatic per generation step in the server.

n_draft_max = n_ctx - prompt.n_tokens() - 2, further min'd with n_remaining()-1 when n_predict is bounded. If it is <= 0 the slot simply does not draft this step. Any draft longer than dp.n_max is truncated after the fact in common_speculative_draft with SPC_DBG 'truncating draft to %d tokens' — which means an n-gram speculator can burn a full 64-token lookup and have it cut to 3 near the context edge. draft-simple is the only impl that also checks dp.n_max inside its own loop (speculative.cpp:352).

`tools/server/server-context.cpp:441-460 (get_n_draft_max)` · `tools/server/server-context.cpp:2912-2914` · `tools/server/server-context.cpp:2936-2946` · `common/speculative.h:56-58` · `common/speculative.cpp:2728-2733 (the truncation)`

### `Chaining more than one speculator (--spec-type a,b,c)`
**Use:** --spec-type ngram-mod,draft-simple etc. Every listed type is instantiated; at each step they are tried in a FIXED priority order and the first one that returns a non-empty draft wins for that sequence.

THE ORDER YOU TYPE IS IGNORED. The list is rebuilt from a bitmask (common_get_enabled_speculative_configs, speculative.cpp:2343-2349) and re-emitted in this hardcoded priority: ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark (speculative.cpp:2542-2552). So every n-gram speculator outranks every model-based one. Duplicates in the mask collapse to one instance. The four model-based types after draft-simple are only added if params.draft.ctx_dft != nullptr (speculative.cpp:2549-2552). Drafts are never merged or tree-combined — exactly one impl's flat token list is used per sequence per step, recorded in spec->impl_last[seq_id] (speculative.cpp:2741).

`common/speculative.cpp:2525-2553 (build + priority list)` · `common/speculative.cpp:2538 (static_assert COUNT==11)` · `common/speculative.cpp:2710-2756 (the fallback loop)` · `common/speculative.h:49-54 (the `drafting` chaining flag)` · `common/speculative.cpp:2201-2209 (impl_last per seq)`

### `How acceptance is counted — server-side (the numbers you see and scrape)`
**Use:** Read the per-slot log line, /metrics, or the trace lines.

Three counters: n_draft_tokens (incremented at draft time with the FULL drafted length, before truncation/verification), n_draft_accepted, n_draft_verif_steps (+1 per verify). Reported as `draft acceptance = accepted/generated` and `mean len = 1 + accepted/verif_steps`. A checkpoint-restore round (partial acceptance on a context that cannot seq_rm partially) returns EARLY at server-context.cpp:3869 without counting anything, then replays; on the replay `spec_is_replay` subtracts one from n_accepted (3886-3888) to avoid double counting. Prometheus names: llamacpp:spec_decode_num_draft_tokens_total, ..._num_accepted_tokens_total, ..._num_drafts_total, and a per-position labelled counter ..._num_accepted_tokens_per_pos_total. The per-position array is sized common_speculative_n_max(&params.speculative) (server-context.cpp:3899), so with ngram-mod it is 64 buckets wide.

`tools/server/server-context.cpp:2966 (n_draft_tokens += draft.size())` · `tools/server/server-context.cpp:3877 (common_speculative_accept with accepted.size()-1)` · `tools/server/server-context.cpp:3883-3903` · `tools/server/server-context.cpp:616-641 (the printed line)` · `tools/server/server-context.cpp:4045-4050 (aggregation)` · `tools/server/server-task.cpp:1551-1561, 1602-1608 (prometheus)`

### `How acceptance is counted — per-implementation (the SPC_TRC statistics line)`
**Use:** Visible at LOG_TRC level only (common_speculative_print_stats).

Per impl: n_call_begin/draft/accept, n_gen_drafts, n_acc_drafts, n_gen_tokens, n_acc_tokens, n_acc_tokens_per_pos, plus t_begin_us/t_draft_us/t_accept_us. Credit goes only to spec->impl_last[seq_id] — the impl that actually produced the draft; every OTHER impl gets accept(seq_id, n_accepted, is_other=true) so it can update its own state but its counters are untouched (speculative.cpp:2796-2801). Printed mean is `1.0 + n_acc_tokens/n_call_accept`. Note gen_perf is a hardcoded `const bool = true` (speculative.cpp:156) — timing is always on.

`common/speculative.cpp:139-161 (the counters)` · `common/speculative.cpp:2768-2802 (common_speculative_accept)` · `common/speculative.cpp:2829-2872 (the print)`

### `The two acceptance rules (greedy match vs. residual/rejection sampling)`
**Use:** Automatic. The rejection-sampling variant is used only when the request has temp > 0 AND the speculator supplied per-token proposal distributions AND the context supports partial rollback.

Greedy variant: sample the target at each position, accept while draft[i] == sampled id, stop at the first mismatch; always returns >= 1 token. Residual variant: accept draft[i] if uniform*q(draft[i]) <= p(draft[i]), else sample from max(0, p-q) and stop. DFlash2's selector lattice is the ONLY thing in this tree that fills dparams.dists (speculative.cpp:1239-1258), so with every other speculator spec_dists stays empty and the greedy rule is used even at temperature 1.0. The verifier RNG is deliberately decorrelated: `llama_sampler_get_seed(chain) ^ 0x9e3779b9U` (sampling.cpp:433-434); DFlash2's selector RNG uses `seed ^ 0x85ebca6bU` (speculative.cpp:1229).

`tools/server/server-context.cpp:3825-3831 (the selection)` · `common/sampling.cpp:692-720 (greedy variant)` · `common/sampling.cpp:722-793 (residual variant)` · `common/sampling.cpp:433-434 (independent verifier RNG)` · `common/speculative.cpp:1238-1258 (the only producer of dists)`

### `ngram-mod parameters (the --spec-default speculator)`
**Use:** --spec-ngram-mod-n-match N (hash window), --spec-ngram-mod-n-max N (max drafted), --spec-ngram-mod-n-min N (discard below).  ·  **default:** `n_match=24, n_max=64, n_min=48 (common.h:351-356)`

CLAMPS: n-min and n-max must be 0..1024 inclusive, n-match must be 1..1024 inclusive — outside that range the flag THROWS and startup fails (arg.cpp:4167, 4177, 4187). A warning fires if n_match < 16: 'ngram_mod n_match=%d is too small - poor quality is possible' (speculative.cpp:1924-1927). The table is a fixed 4*1024*1024-entry open-addressed-by-overwrite hash, 16 MiB, allocated in the ctor (speculative.cpp:1914) and SHARED ACROSS ALL SEQUENCES. There is no key stored and no collision check (ngram-mod.cpp:27-41) — a collision silently returns a wrong token, which speculation then merely fails to accept. Two automatic resets: occupancy > 0.25 at begin() (speculative.cpp:1952-1957, warns 'ngram_mod occupancy %.2f exceeds threshold'), and 5 consecutive accept rounds with acceptance fraction < 0.25 (speculative.cpp:2044-2054). draft_one builds exactly n_max tokens or clears the result if it got fewer than n_min (speculative.cpp:1992-2004) — with the defaults that means 48 consecutive hash hits or nothing at all.

`common/arg.cpp:4163-4192` · `common/common.h:351-356` · `common/speculative.cpp:1887-2059` · `common/ngram-mod.cpp:9-46`

### `ngram-simple / ngram-map-k / ngram-map-k4v parameters`
**Use:** --spec-ngram-{simple,map-k,map-k4v}-size-n N, -size-m N, -min-hits N. Three independent parameter blocks, one per type.  ·  **default:** `size_n=12, size_m=48, min_hits=1 for all three (common.h:358-362)`

CLAMPS: size-n and size-m must be 1..1024 inclusive (throws otherwise); min-hits must be >= 1 (throws otherwise). Fields are uint16_t so 1024 is well inside range. ngram-map keeps at most COMMON_NGRAM_MAX_VALUES = 4 m-grams per key (ngram-map.h:39) — that is what the 'k4v' name means — and a fixed 262144-entry hash index (COMMON_NGRAM_HASH_MAP_SIZE, ngram-map.h:42, allocated per sequence at ngram-map.h:76). ngram-map-k vs k4v is only the key_only flag. The replaced flags --spec-ngram-size-n/-size-m/-min-hits now hard-error telling you to use the per-type ones (arg.cpp:4305-4325).

`common/arg.cpp:4194-4285` · `common/common.h:358-362` · `common/speculative.cpp:2211-2220 (get_common_ngram_map)` · `common/ngram-map.h:39-77` · `common/ngram-map.cpp:401-404 (min_hits gate)`

### `ngram-cache parameters`
**Use:** --spec-type ngram-cache, optionally with -lcs/--lookup-cache-static FNAME and -lcd/--lookup-cache-dynamic FNAME.  ·  **default:** `both paths empty; draft length hardcoded to 8`

n_draft = 8 is a literal with a `// TODO get from config?` next to it (speculative.cpp:2227) — there is NO flag for it. save_static and save_dynamic are hardcoded false (speculative.cpp:2230-2231), so -lcd is read at startup and never written back. n-gram orders are fixed by LLAMA_NGRAM_MIN=1 / LLAMA_NGRAM_MAX=4 (ngram-cache.h:9-10). A cache file that fails to load does not degrade gracefully: it GGML_ABORTs (speculative.cpp:2111, 2124).

`common/arg.cpp:1622-1635` · `common/speculative.cpp:2222-2236 (create_state_ngram_cache)` · `common/speculative.cpp:2375-2377 (n_max reported as 8)` · `common/ngram-cache.h:9-11`

### `--spec-default — the one-flag preset`
**Use:** --spec-default

Appends ngram-mod to types and sets n_match=24, n_min=48, n_max=64 — i.e. exactly the struct defaults, so it is equivalent to `--spec-type ngram-mod` today. It APPENDS, so combining it with --spec-type gives you both. A commented-out block (arg.cpp:4636-4640) shows an intended ngram-map-k4v companion (size_n=8, size_m=24, min_hits=2) that is not enabled.

`common/arg.cpp:4627-4642`

### `Draft model loading: path, HF repo, devices, layers, tensor overrides`
**Use:** -md/--spec-draft-model FNAME; -hfd/--spec-draft-hf <user>/<model>[:quant]; -devd/--spec-draft-device; -ngld/--spec-draft-ngl N|auto|all; -otd/--spec-draft-override-tensor; -cmoed/--spec-draft-cpu-moe; -ncmoed/--spec-draft-n-cpu-moe N.  ·  **default:** `-ngld default 'auto' (n_gpu_layers = -1, common.h:338); model path unset`

-ngld encoding: 'auto' => -1, 'all' => -2, else the literal integer (arg.cpp:4131-4137); arg.cpp:4125 GGML_ASSERTs the default is negative. --fit IS NOT APPLIED TO THE DRAFT MODEL: common_speculative_init_result calls llama_model_load_from_file directly (speculative.cpp:2468) and never common_fit_params, so a draft left at 'auto' loads with n_gpu_layers<0, which llama_model::n_gpu_layers() (src/llama-model.cpp:1745-1747) resolves to ALL layers on GPU. The server compensates only by adding the measured draft footprint to fit_params_target so the TARGET shrinks (server-context.cpp:1032-1087). -md also writes the value into mparams.hf_file so it doubles as the filename when -hfd is set (arg.cpp:4150). Everything else — n_ctx, n_parallel, flash-attn, rope — is inherited wholesale from the target because common_base_params_to_speculative starts with `result = params` (speculative.cpp:2391).

`common/arg.cpp:3907-3913 (hf)` · `common/arg.cpp:4048-4074 (overrides / cpu-moe)` · `common/arg.cpp:4117-4144 (device / ngl)` · `common/arg.cpp:4145-4152 (model)` · `common/speculative.cpp:2387-2430 (common_base_params_to_speculative)` · `common/common.cpp:1289-1305 (fit is applied here, for the target only)`

### `Draft-model CPU/thread flags`
**Use:** -td/--spec-draft-threads, -tbd/--spec-draft-threads-batch, -Cd/--spec-draft-cpu-mask, -Crd/--spec-draft-cpu-range, --spec-draft-cpu-strict, --spec-draft-prio, --spec-draft-poll and the four -batch variants.  ·  **default:** `all inherit the corresponding main-model setting; prio default 0 (normal)`

--spec-draft-prio and --spec-draft-prio-batch throw for values outside 0..3 (arg.cpp:3965-3967, 4009-4011). -td/-tbd map any value <= 0 to std::thread::hardware_concurrency() (arg.cpp:3919-3921, 3929-3931). The thread counts are only copied into the draft params when cpuparams.n_threads > 0 (speculative.cpp:2399). --spec-draft-cpu-range-batch (-Crbd) has set_examples({LLAMA_EXAMPLE_SPECULATIVE}) ONLY (arg.cpp:3997) — unlike its siblings it is NOT accepted by llama-server or llama-cli, confirmed absent from the shipped binary's --help.

`common/arg.cpp:3914-4021` · `common/arg.cpp:895-896 (postprocess_cpu_params inheritance)` · `common/speculative.cpp:2399-2402`

### `Output-limit plumbing that speculation forces on the target context`
**Use:** Automatic; it is what makes a 1+n_draft verify batch legal.

per_seq = min(n_batch, 1+n_draft); total = min(n_batch, n_parallel*(1+n_draft)). common_speculative_n_max takes the MAX over all enabled types: draft.n_max for the model-based ones, size_m for ngram-simple/map-k/map-k4v, ngram_mod.n_max for ngram-mod, and a literal 8 for ngram-cache. For embedding/pooling servers the limits collapse to {n_batch, 1} (server-context.cpp:43-46). DFlash/DSpark additionally force n_batch and n_ubatch up to n_parallel*(n_max+1) (speculative.cpp:2418-2423) — the only place in this area that raises your batch behind your back.

`common/speculative.h:36-43` · `common/speculative.cpp:2351-2385 (common_speculative_n_max)` · `common/speculative.cpp:2512-2521 (common_speculative_get_output_limits)` · `tools/server/server-context.cpp:42-54, 956-958` · `common/common.cpp:1700-1701` · `src/llama-context.cpp:249-251`

### `n_rs_seq — recurrent-state snapshots reserved for rollback`
**Use:** Automatic; derived from the type list.

cparams.n_rs_seq = draft.n_max, but ONLY if the type list contains draft-mtp, draft-eagle3, draft-dflash or draft-dspark; every ngram-* type gets 0. This is applied to the TARGET context (common.cpp:1699, which is common_context_params_to_llama, used for the target too). It bounds how far a recurrent/hybrid target can roll back without a full checkpoint restore, which is why n_draft > llama_n_rs_seq(ctx) flips the code onto the slower checkpoint path.

`common/common.h:386-392 (need_n_rs_seq)` · `common/common.cpp:1699` · `common/speculative.cpp:2460` · `tools/server/server-context.cpp:2984-2987, 3825-3827, 3838-3840`

### `Speculative state save/restore across prompt checkpoints`
**Use:** Automatic (used for eagle3's deferred boundary on recurrent/hybrid targets).

The function carries an explicit source comment: '// TODO: support the case of more than one speculative implementations having a state' (speculative.cpp:2804). get_state returns on the FIRST impl that answers true, so with a chained --spec-type only one impl's state can ever be stashed. set_state broadcasts the same blob to every impl (speculative.cpp:2824-2826). Only eagle3 implements it, and only when the target is recurrent or hybrid (speculative.cpp:867-874).

`common/speculative.h:90-92` · `common/speculative.cpp:2804-2827` · `common/speculative.cpp:865-908 (eagle3 need_boundary_stash/get_state)` · `common/common.h:1148`

### `When speculation is silently switched off`
**Use:** Diagnose from the startup log.

Four distinct silent-ish paths: (1) the target context cannot seq_rm at all -> SRV_WRN 'speculative decoding not supported by this context' and common_speculative_init is never even called (server-context.cpp:1211-1213, 1229). (2) common_speculative_init returns nullptr when the config list produced zero impls, logging only at TRC level: 'no implementations specified for speculative decoding' (speculative.cpp:2630-2632); the server then also drops ctx_dft and model_dft (server-context.cpp:1243-1247). (3) any impl constructor that throws takes the whole thing down to SRV_ERR 'failed to initialize speculative decoding context' and no speculation (server-context.cpp:1230-1234). (4) --spec-type draft-eagle3/draft-mtp/draft-dflash/draft-dspark with no draft context available: the config is simply not added (speculative.cpp:2549-2552).

`tools/server/server-context.cpp:1210-1217, 1228-1247` · `common/common.cpp:1559-1584 (common_context_can_seq_rm)` · `common/speculative.cpp:2630-2633` · `common/speculative.cpp:2549-2552`

### `Draft-model / target-model vocabulary compatibility check`
**Use:** Automatic, at draft-simple construction only.

Requires identical vocab type, identical add_bos/add_eos and the BOS/EOS ids themselves, a vocab-size difference <= SPEC_VOCAB_MAX_SIZE_DIFFERENCE = 128, and byte-identical token text for every id from SPEC_VOCAB_CHECK_START_TOKEN_ID = 5 upward. On failure it throws 'draft model vocab type must match target model to use speculation'. IMPORTANT: this is called from the draft-simple constructor ONLY — eagle3, mtp, dflash and dspark never run it.

`common/speculative.cpp:31-32 (the constants)` · `common/speculative.cpp:68-131 (common_speculative_are_compatible)` · `common/speculative.cpp:238-245 (the only caller)`

### `Speculative hooks into the target decode loop`
**Use:** Automatic.

common_speculative_process is invoked on EVERY successful target decode batch, prompt processing included, and fans out to every impl. For draft-simple that means the entire prompt is also decoded through the draft model (speculative.cpp:268) — the prefill cost of a draft model is paid on every batch. For all four ngram-* impls process() is a stub that returns true with a '// TODO: implement' (speculative.cpp:1810-1813, 1858-1861, 2016-2019, 2178-2181), so they cost nothing at prefill. If process returns false the server throws 'failed to process speculative batch' and the request dies (server-context.cpp:3661-3666). A source comment marks the known waste: '[TAG_SPEC_AVOID_DRAFT_REEVAL] for now, always re-evaluate for simplicity' (server-context.cpp:3652-3654).

`common/speculative.h:79-88` · `tools/server/server-context.cpp:3655-3667 (process on every target batch)` · `tools/server/server-context.cpp:711-718 (same for mtmd image chunks)` · `tools/server/server-context.cpp:3750-3752 (begin)` · `common/speculative.cpp:2673-2685 (process fan-out)` · `common/speculative.cpp:262-277 (draft-simple process = a full draft-model llama_decode)`

### `Removed / renamed flags that now hard-error`
**Use:** Do not use; the error message names the replacement.

--draft / --draft-n / --draft-max, --draft-min / --draft-n-min, --spec-ngram-size-n, --spec-ngram-size-m, --spec-ngram-min-hits all call arg_removed() and abort startup. LLAMA_ARG_DRAFT_MAX and LLAMA_ARG_DRAFT_MIN are still bound to the removed options (arg.cpp:4297, 4304), so a stale env var in a profile will kill the process too. Old aliases that DO still work: -md/--model-draft, -hfd/--hf-repo-draft, -ngld/--n-gpu-layers-draft, -ctkd/--cache-type-k-draft, -devd/--device-draft, --draft-p-min, --draft-p-split, -td/--threads-draft and the CPU-mask family.

`common/arg.cpp:4291-4325`

### `--mtp / --dflash / --eagle3 (llama-download only)`
**Use:** llama-download --mtp / --dflash / --eagle3

These push a speculative TYPE onto params.speculative.types purely to make the downloader fetch the matching sidecar; set_examples({LLAMA_EXAMPLE_DOWNLOAD}) so they are not accepted by the server. Note there is no --dspark counterpart, though a dspark sidecar is discovered and downloaded when --spec-type is left at its default (arg.cpp:405-415).

`common/arg.cpp:3038-3058` · `common/arg.cpp:392-397 (the download opts they feed)`

### `Server runtime reporting of the active config`
**Use:** GET /props and the task-params dump show `speculative.types`.

`speculative.types` is the comma-joined name list (common_speculative_type_name_str), so it reflects the accumulated/deduplicated list actually in effect — the fastest way to confirm what a profile really enabled. The boolean at server-context.cpp:650 is simply whether common_speculative_init returned non-null.

`tools/server/server-task.cpp:83, 142` · `tools/server/server-context.cpp:650 ({"speculative", can_speculate()})`

**What this area CANNOT do.** Things this area CANNOT do, each with the line that forecloses it: 1. NO PER-REQUEST SPECULATIVE PARAMETERS. Every speculative field in the server's request schema — speculative.n_max, speculative.n_min, speculative.p_min, speculative.type, speculative.ngram_size_n/size_m/min_hits — is inside an `#if 0 ... #endif` block (tools/server/server-schema.cpp:198 and :227), with the comment 'to keep things simple, we disable speculative parameter adjustments for now' (server-schema.cpp:197). Speculation is a process-lifetime setting; changing it means restarting the server. 2. NO TREE / MULTI-BRANCH DRAFTS. The draft is a flat llama_tokens (common/speculative.h:67) laid into the batch as consecutive positions (tools/server/server-context.cpp:488-493). p_split, the parameter that would fork a draft, is read only by examples/speculative/speculative.cpp:67 and by nothing in common/ or tools/. 3. NO COMBINING TWO SPECULATORS INTO ONE DRAFT. common_speculative_draft breaks out of the impl loop for a sequence as soon as one impl returns a non-empty result (common/speculative.cpp:2725-2726 sets dp.drafting=false, 2753-2755 breaks). Multiple --spec-type values give you a fallback chain, not an ensemble. 4. YOU CANNOT CHOOSE THE ORDER OF THAT CHAIN. The priority list is hardcoded at common/speculative.cpp:2542-2552 and rebuilt from a bitmask (common/speculative.cpp:2343-2349), so command-line order is discarded and duplicates collapse. Every n-gram speculator outranks every model-based one. 5. AT MOST ONE SPECULATOR MAY CARRY STATE ACROSS CHECKPOINTS. common_speculative_get_state returns on the first impl that answers true (common/speculative.cpp:2810-2814), under an explicit '// TODO: support the case of more than one speculative implementations having a state' (common/speculative.cpp:2804). 6. THE DRAFT SAMPLER IS NOT CONFIGURABLE. It is fixed to {TOP_K} with top_k=10 at common/speculative.cpp:230-233, 492-493, 1005-1006, 1418-1419; the backend chain is likewise a bare top_k(10) at 502, 1018, 1428. The block that would have made it configurable is commented out at common/speculative.cpp:209-224. 7. BACKEND SAMPLING CANNOT COEXIST WITH A GRAMMAR OR A REASONING BUDGET. common/sampling.cpp:421-425 and 427-431 force params.backend_sampling=false with a warning; common/sampling.cpp:631-632 GGML_ASSERTs it if it somehow got through. It is also refused under -sm row / LLAMA_SPLIT_MODE_TENSOR (src/llama-context.cpp:1216-1227). 8. --fit DOES NOT SIZE THE DRAFT MODEL. common_speculative_init_result loads the draft with llama_model_load_from_file directly (common/speculative.cpp:2468) and never calls common_fit_params; the fit path exists only in common_init_from_params (common/common.cpp:1294-1303) for the target. With -ngld left at 'auto' (-1), llama_model::n_gpu_layers() resolves a negative value to all layers (src/llama-model.cpp:1745-1747). The server only reserves headroom so the TARGET gets smaller (tools/server/server-context.cpp:1032-1087). 9. -ctk / -ctv DO NOT REACH THE DRAFT CACHE. common/speculative.cpp:2405-2406 overwrites result.cache_type_k/v with the speculative struct's own fields unconditionally, whose defaults are F16 (common/common.h:340-341). Only -ctkd/-ctvd change them. 10. THE VERIFY BATCH MUST FIT IN n_batch OR THE PROCESS ABORTS. tools/server/server-context.cpp:496: `GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens")`. With backend sampling on, exceeding n_outputs_max_per_seq is a decode failure instead: src/llama-context.cpp:1683-1689 logs 'backend sampling supports at most %u outputs per sequence' and returns -1. 11. SPECULATION IS IMPOSSIBLE ON A CONTEXT THAT CANNOT seq_rm. tools/server/server-context.cpp:1229 gates common_speculative_init on ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO, which common/common.cpp:1559-1584 determines empirically by trial-decoding two tokens. 12. ngram-cache CANNOT BE TUNED OR PERSISTED. Its draft length is the literal 8 with '// TODO get from config?' (common/speculative.cpp:2227) and save_static/save_dynamic are hardcoded false (common/speculative.cpp:2230-2231), so -lcd is loaded and never written back. Its n-gram orders are fixed by LLAMA_NGRAM_MIN=1 / LLAMA_NGRAM_MAX=4 (common/ngram-cache.h:9-10). A malformed cache file GGML_ABORTs (common/speculative.cpp:2111, 2124). 13. ngram-mod CANNOT DETECT A HASH COLLISION. common_ngram_mod stores only the successor token, no key (common/ngram-mod.cpp:27-41); a collision returns a plausible wrong token that speculation then merely fails to accept. Its table size is the literal 4*1024*1024 at common/speculative.cpp:1914 with no flag, and it is shared across all sequences. 14. THE DRAFT CANNOT EXCEED THE TRAINED BLOCK / HEAD COUNT. DFlash/DSpark clamp both n_max and n_min to block_size-1 (or block_size for anchor-sampling DSpark) at common/speculative.cpp:990-996; MTP with chained heads clamps n_max to n_mtp_layers SILENTLY at common/speculative.cpp:1446. 15. THE VOCAB-COMPATIBILITY GUARD ONLY GUARDS draft-simple. common_speculative_are_compatible is called from exactly one place, the draft-simple constructor (common/speculative.cpp:238); eagle3, mtp, dflash and dspark construct without it. 16. --spec-draft-n-min HAS NO INPUT VALIDATION (common/arg.cpp:4086-4092) and --spec-draft-p-min / --spec-draft-p-split are raw std::stof with no range check (common/arg.cpp:4098, 4105), while n_min is later compared as an unsigned size_t (e.g. common/speculative.cpp:380). A negative n_min therefore discards every draft with no error and no warning — the classic 'plausible number instead of a failure' shape. 17. --spec-draft-cpu-range-batch (-Crbd) IS NOT AVAILABLE ON THE SERVER: set_examples({LLAMA_EXAMPLE_SPECULATIVE}) at common/arg.cpp:3997, confirmed absent from the shipped binary's --help.

**Exists but unused in our profile.** For a profile of `--spec-type ngram-mod`, `-ctk q4_0 -ctv q4_0`, `-ngl auto --fit on`, `-np 1`, the following exist but are never exercised: NO DRAFT CONTEXT IS CREATED AT ALL. tools/server/server-context.cpp:961-965 computes `has_spec = has_draft || spec_mtp`; with ngram-mod and no -md both are false, so the fit-time draft VRAM reservation (server-context.cpp:1032-1087) and common_speculative_init_from_params (server-context.cpp:1116-1147) are both skipped, and ctx_dft stays nullptr. Everything downstream of a draft context is therefore dead in this profile: - --spec-draft-model / -hfd / -devd / -ngld / -otd / -cmoed / -ncmoed and the whole draft CPU/thread family (common/arg.cpp:3907-4074, 4117-4152). - --spec-draft-n-max (3), --spec-draft-n-min (0), --spec-draft-p-min (0.0), --spec-draft-p-split (0.1), --spec-draft-backend-sampling (default ON) — every one of these reads common_params_speculative_draft, which only the five draft-model impls touch (common/speculative.cpp:180, 427, 911, 1350). - --spec-draft-type-k / --spec-draft-type-v: the F16 default is copied into params_dft (common/speculative.cpp:2405-2406) but no context is ever built from it, so the fact that -ctk q4_0 does not propagate is harmless HERE and only bites the moment a draft model is added. - The DFlash/DSpark block-size clamp (common/speculative.cpp:988-996) and the batch/ubatch widening it triggers (common/speculative.cpp:2413-2427), the MTP chained-head clamp (common/speculative.cpp:1446), the eagle3 3-extract-layer check (common/speculative.cpp:472-475), the vocab-compatibility guard (common/speculative.cpp:68-131, 238). - The draft-side backend sampler chains (common/speculative.cpp:498-511, 1013-1027, 1423-1437) and their 'backend offload failed' fallback. n_rs_seq IS ZERO. common_params_speculative::need_n_rs_seq() returns non-zero only for draft-mtp/eagle3/dflash/dspark (common/common.h:386-391), so the target context gets n_rs_seq=0 and every RS-bounded rollback branch (server-context.cpp:2984-2987, 3825-3827, 3838-3840) is unreachable; rollback is decided purely by PART vs FULL. THE REJECTION-SAMPLING ACCEPT PATH NEVER RUNS. common_sampler_sample_and_accept_n's dists overload (common/sampling.cpp:722-793) is selected only when spec_dists.size() == spec_draft.size() (server-context.cpp:3828-3830), and the only producer of dists in the tree is the DFlash2 selector (common/speculative.cpp:1239-1258). With ngram-mod, spec_dists is always empty, so the greedy prefix-match rule (common/sampling.cpp:692-720) is used even at temperature 1.0 — and the verifier RNG at common/sampling.cpp:433-434 is never consulted. common_speculative_process IS A NO-OP. ngram-mod's process() is the '// TODO: implement' stub returning true (common/speculative.cpp:2016-2019), so the per-batch fan-out at server-context.cpp:3655-3667 costs nothing and the prefill penalty a draft model would impose (common/speculative.cpp:268) does not exist here. OTHER SPECULATORS' PARAMETERS ARE INERT: --spec-ngram-simple-*, --spec-ngram-map-k-*, --spec-ngram-map-k4v-* (all defaulting size_n=12 / size_m=48 / min_hits=1, common/common.h:358-362) and -lcs/-lcd for ngram-cache (common/arg.cpp:1622-1635) parse and store but are only read when their own type is in the list (common/speculative.cpp:2582-2622). -np 1 COLLAPSES THE MULTI-SEQUENCE MACHINERY: n_seq=1, so the per-seq vectors (sinfos, dparams, impl_last) are length 1, the cross-sequence batching in every draft() loop degenerates, and the ngram-mod hash table — shared across sequences by design (common/speculative.cpp:1914) — has no cross-request contamination to worry about. The output limits reduce to per_seq = total = min(n_batch, 1+64) = 65 (common/speculative.cpp:2512-2521, server-context.cpp:42-53), since common_speculative_n_max returns ngram_mod.n_max = 64 for this profile (common/speculative.cpp:2372-2374). WHAT IS LIVE and worth watching in this profile: the 4 MiB-entry shared hash and its two automatic resets (occupancy > 0.25 at begin, common/speculative.cpp:1952-1957; five consecutive rounds below 25 % acceptance, common/speculative.cpp:2044-2054); the all-or-nothing n_min=48 gate inside draft_one (common/speculative.cpp:1995-1998); and the dp.n_max truncation near the context edge (server-context.cpp:451, common/speculative.cpp:2728-2732), which can silently cut a 64-token lookup down to a handful without any of it showing up as a lower acceptance rate — n_draft_tokens is counted AFTER truncation (server-context.cpp:2966 runs on the already-truncated draft), but the per-position histogram is still sized 64 (server-context.cpp:3899).

## Attention, kernels and graphs — Flash Attention selection, CUDA FA kernel/quant-type coverage, -b/-ub, CUDA graph capture, and query-tokens-per-step thresholds. Source tree C:\AI\llama.cpp @ 1deefcca3 (build 10499), as compiled into C:\AI\llama.cpp-dflash2\llama-server.exe.

### `-fa / --flash-attn [on|off|auto]`
**Use:** -fa on | -fa off | -fa auto (env LLAMA_ARG_FLASH_ATTN). Values parsed by is_truthy/is_falsey/is_autoy; anything else is a hard error.  ·  **default:** `auto (LLAMA_FLASH_ATTN_TYPE_AUTO = -1)`

cparams.flash_attn = (type != DISABLED), i.e. AUTO starts life as ENABLED and is only turned off later by the probe (src/llama-context.cpp:229). cparams.auto_fa is true ONLY for 'auto' (src/llama-context.cpp:230) — with '-fa on' no probe ever runs, so an unsupported configuration is not detected and not reported. Verified present in the staged binary: `llama-server.exe --help` prints "-fa, --flash-attn [on|off|auto] ... (default: 'auto')".

`common/arg.cpp:1744-1758` · `common/common.h:489` · `include/llama.h:190-195` · `src/llama-context.cpp:3534`

### `-fa auto resolution probe (what actually decides FA on/off)`
**Use:** Implicit. With -fa auto, llama_context::resolve_fused_ops() reserves a probe graph with n_tokens = 1 * n_seq_max, then checks whether every fused FLASH_ATTN node landed on the same device as its layer. Mismatch => cparams.flash_attn = false.  ·  **default:** `runs once, at context construction, before the pp/tg reserve`

llm_fused_op_flash_attn_probe.n_tokens_per_seq = 1 (src/llama-context.cpp:42), so the decision is made on a ONE-QUERY-TOKEN-PER-SEQ graph. Kernel eligibility is almost entirely head-dim/type driven so this rarely diverges, but the probe never sees the prompt-processing shape. On success it logs "Flash Attention enabled"; on failure "layer N is assigned to device X but Flash Attention is assigned to device Y (usually due to missing support)" then "Flash Attention not supported, set to disabled" (src/llama-context.cpp:532-548). cparams.auto_fa is cleared afterwards (556) so the probe never re-runs.

`src/llama-context.cpp:34-43` · `src/llama-context.cpp:504-551` · `src/llama-context.cpp:511` · `src/llama-context.cpp:513` · `src/llama-context.cpp:554-557` · `src/llama-context.cpp:664 (sched_reserve calls resolve_fused_ops)`

### `FA forced ON without asking`
**Use:** Triggered by configuration, not by a flag.  ·  **default:** `n/a`

Two cases silently promote AUTO -> ENABLED: (1) LLAMA_SPLIT_MODE_TENSOR requires FA (3587-3590); with -fa off it is a hard init failure (3591-3594). (2) A quantized V cache requires FA: with AUTO it logs "enabling flash_attn since it is required for quantized V cache" and forces ENABLED (3603-3606); with -fa off llama_init_from_model returns nullptr (3607-3610). So `-ctv q4_0 -fa off` cannot start.

`src/llama-context.cpp:3586-3595` · `src/llama-context.cpp:3602-3611`

### `FA forced OFF without asking`
**Use:** Triggered by model arch.  ·  **default:** `n/a`

LLM_ARCH_GROK: warns "flash_attn is not compatible with Grok - forcing off" and overwrites the requested type. This happens before the SPLIT_MODE_TENSOR / quantized-V checks, so Grok + quantized V is an unconditional init failure.

`src/llama-context.cpp:3581-3584`

### `Quantized-K block-size gate`
**Use:** -ctk <type> with FA not disabled.  ·  **default:** `n/a`

For every layer, n_embd_head_k(il) must be divisible by ggml_blck_size(type_k) (and the same for V). q4_0/q8_0 have block 32; head dims 40, 72, 112 are NOT multiples of 32 and would fail here. Head dim 128 (Qwen3-family) passes. Error text: "K cache type %s with block size %u does not divide n_embd_head_k=%u" -> returns nullptr.

`src/llama-context.cpp:3613-3622` · `src/llama-context.cpp:3624-3633`

### `-fa changes the physical V-cache layout`
**Use:** Implicit consequence of the resolved flash_attn value.  ·  **default:** `FA on => v_trans = false; FA off => v_trans = true`

Every llama_kv_cache is constructed with v_trans = !cparams.flash_attn. Turning FA off transposes the V cache, which is also why a non-FA V cache cannot be quantized (src/llama-context.cpp:463-467 throws "quantized V cache was requested, but this requires Flash Attention"). It also changes the kq_mask dtype: F16 with FA, F32 without (src/llama-graph.cpp:38, 789).

`src/llama-model.cpp:2124` · `src/llama-model.cpp:2151` · `src/llama-model.cpp:2311` · `src/llama-model.cpp:2330` · `src/llama-kv-cache.cpp:81` · `src/llama-kv-cache.cpp:208` · `src/llama-kv-cache.cpp:1282`

### `CUDA FA kernel selection (three kernels: VEC=100, TILE=200, MMA_F16=400)`
**Use:** No flag. ggml_cuda_get_best_fattn_kernel() decides per FLASH_ATTN_EXT node, per call, from cc, head dims, K/V types, gqa_ratio, and Q->ne[1] (= number of query tokens in this ubatch).  ·  **default:** `n/a`

On this GPU (RTX 4070 SUPER, cc = 890 = GGML_CUDA_CC_ADA_LOVELACE; the build compiles ONLY compute_89) turing_mma_available() is true, so the whole Ada branch at fattn.cu:461-483 applies and TILE is unreachable for any head dim except 40 and 72 (excluded at 461, and 40/72 fail `Q->ne[0] % 64 == 0` at 458 so they fall to TILE at 532). For head dim 128 the choice is VEC or MMA_F16 only.

`ggml/src/ggml-cuda/fattn.cu:330-336` · `ggml/src/ggml-cuda/fattn.cu:358-533` · `ggml/src/ggml-cuda/fattn.cu:570-583` · `ggml/src/ggml-cuda/common.cuh:344-350`

### `THE query-tokens-per-step threshold that decides VEC vs MMA_F16`
**Use:** Implicit: Q->ne[1] is the number of query tokens in the ubatch (1 for plain decode, 1+n_draft for a speculative step, n_ubatch for prompt processing).  ·  **default:** `n/a`

On Ada with can_use_vector_kernel (head dim 64/128/256, and n_kv % 256 == 0): - UNQUANTIZED K and V (F16/BF16): VEC only if Q->ne[1] == 1 AND Q->ne[3] == 1 AND NOT (gqa_ratio > 4 && K->ne[1] >= 8192) — fattn.cu:464. So with GQA ratio 8 (32 q heads / 4 kv heads) and an F16 cache, plain 1-token decode switches from VEC to MMA_F16 once the padded n_kv reaches 8192. A kernel change at a context depth, with no log line. - QUANTIZED K/V (q4_0, q8_0): VEC only if Q->ne[1] <= 2 — fattn.cu:469. Three or more query tokens in a step => MMA_F16. - Fallback: if gqa_opt does not apply and Q->ne[1] == 1, VEC — fattn.cu:476-478. - Everything else => MMA_F16 (482). There is no flag to override this and no message when it flips.

`ggml/src/ggml-cuda/fattn.cu:458` · `ggml/src/ggml-cuda/fattn.cu:461-483` · `ggml/src/ggml-cuda/fattn.cu:464` · `ggml/src/ggml-cuda/fattn.cu:469` · `ggml/src/ggml-cuda/fattn.cu:476-478` · `ggml/src/ggml-cuda/fattn.cu:482`

### `MMA_F16 and TILE dequantize the WHOLE K and V cache to F16 on every call`
**Use:** Implicit. launch_fattn is called with need_f16_K = need_f16_V = true from both the MMA and the TILE kernels; only VEC passes the real types through.  ·  **default:** `n/a`

This is the biggest hidden cost in the area. When the chosen kernel is MMA_F16 or TILE and the cache is quantized, launch_fattn runs to_fp16 over ggml_nelements(K) and ggml_nelements(V) into scratch appended to the FLASH_ATTN_EXT dst tensor (fattn-common.cuh:1024-1027, 1063-1066). K here is the full padded KV view (ne[1] = n_kv), so it is the ENTIRE cache for that layer, re-expanded per layer per decode step. The scratch is charged to the compute buffer: ggml_backend_cuda_buffer_type_get_alloc_size routes FLASH_ATTN_EXT through ggml_cuda_flash_attn_ext_get_alloc_size (ggml-cuda.cu:909-910), and ggml-alloc sizes blocks from that. Because the reserve pass runs prompt processing at n_tokens = min(n_ctx, n_ubatch) (src/llama-context.cpp:595), the MMA path is always taken during reserve, so the F16 scratch is always budgeted whenever the KV cache is quantized — 2 bytes/element against q4_0's 0.5625 bytes/element of actual cache. VEC needs no conversion unless the source is F32 (fattn.cu:557-558).

`ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963` · `ggml/src/ggml-cuda/fattn-tile.cuh:1165-1166` · `ggml/src/ggml-cuda/fattn-tile.cuh:1228-1229` · `ggml/src/ggml-cuda/fattn-vec.cuh:539-543` · `ggml/src/ggml-cuda/fattn-common.cuh:1022-1047` · `ggml/src/ggml-cuda/fattn-common.cuh:1050-1084` · `ggml/src/ggml-cuda/fattn-common.cuh:47-85` · `ggml/src/ggml-cuda/fattn.cu:536-568` · `ggml/src/ggml-cuda/ggml-cuda.cu:906-912` · `ggml/src/ggml-alloc.c:684, 701, 867`

### `Which KV quant types have CUDA FA kernels IN THIS BUILD`
**Use:** -ctk / -ctv. Nothing selects a kernel directly.  ·  **default:** `f16 for both`

GGML_CUDA_FA_ALL_QUANTS defaults OFF (ggml/CMakeLists.txt:208) and is NOT in this build — verified: `grep -c FA_ALL_QUANTS build-dflash2/compile_commands.json` = 0, and the nvcc lines carry only -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 with --generate-code=arch=compute_89. Consequences: - ggml_cuda_fattn_kv_type_supported() returns TRUE only for F32, F16, Q4_0, Q8_0, BF16. Q4_1, Q5_0, Q5_1 fall into the `#ifndef GGML_CUDA_FA_ALL_QUANTS return false;` at fattn.cu:346-348. - K->type must EQUAL V->type (fattn.cu:442-446). `-ctk q8_0 -ctv q4_0` has no CUDA kernel at all. - Only four vec instances are compiled: f16-f16, q4_0-q4_0, q8_0-q8_0, bf16-bf16 (CMakeLists.txt:120-124, matching fattn.cu:284-289). So the usable quantized KV settings on this binary are exactly: -ctk q4_0 -ctv q4_0, -ctk q8_0 -ctv q8_0 (plus f16/f16 and bf16/bf16).

`ggml/src/ggml-cuda/fattn.cu:338-356` · `ggml/src/ggml-cuda/fattn.cu:264-325` · `ggml/src/ggml-cuda/fattn.cu:284-290` · `ggml/src/ggml-cuda/fattn.cu:442-450` · `ggml/src/ggml-cuda/CMakeLists.txt:115-125` · `ggml/CMakeLists.txt:208`

### `Head-dimension support table for CUDA FA`
**Use:** Determined by the model.  ·  **default:** `n/a`

Accepted DK: 40, 64, 72, 80, 96, 112, 128, 256 (require DV == DK); 192 (DV 128, needs gqa_opt and gqa_ratio%8==0); 320 (DV 256, gqa_ratio%32==0); 512 (DV==DK, needs gqa_opt); 576 (DV 512, needs gqa_opt). Anything else => BEST_FATTN_KERNEL_NONE => op falls off the GPU. Note 40 and 72 have MMA/vec excluded (fattn.cu:461) so they only ever get TILE.

`ggml/src/ggml-cuda/fattn.cu:392-437` · `ggml/src/ggml-cuda/fattn.cu:120-236`

### `n_kv is padded to a multiple of 256 = FATTN_KQ_STRIDE`
**Use:** Implicit.  ·  **default:** `pad = max(n_pad, 256); n_pad itself is 1 for the attention caches`

get_n_kv() rounds used_max_p1 up to 256 explicitly so "the graph remains constant across batches and can be reused". This is what makes `K->ne[1] % FATTN_KQ_STRIDE == 0` true, which is a precondition for BOTH gqa_opt_applies (fattn.cu:378) and can_use_vector_kernel (fattn.cu:458). Separately, cparams.n_ctx is itself GGML_PAD'd to 256 (src/llama-context.cpp:288), and --fit rounds n_ctx down to a multiple of 256 (common/fit.cpp:344).

`src/llama-kv-cache.cpp:1233-1246` · `src/llama-kv-cache.cpp:1238` · `ggml/src/ggml-cuda/fattn-common.cuh:9` · `src/llama-model.cpp:2311` · `src/llama-context.cpp:288`

### `-b / --batch-size (logical batch)`
**Use:** -b N (env LLAMA_ARG_BATCH)  ·  **default:** `2048 (common/common.h:443 and llama_context_default_params src/llama-context.cpp:3522)`

CLAMP: with causal attention, cparams.n_batch = min(n_ctx, n_batch) (src/llama-context.cpp:245) — so `-c 4096 -b 8192` silently becomes 4096. Enforced at decode by GGML_ASSERT(n_tokens_all <= cparams.n_batch) (1711). Confirmed against the staged binary's --help: "logical maximum batch size (default: 2048)".

`common/arg.cpp:1658-1664` · `common/common.h:443` · `src/llama-context.cpp:245` · `src/llama-context.cpp:1711` · `src/llama-context.cpp:3522`

### `-ub / --ubatch-size (physical batch)`
**Use:** -ub N (env LLAMA_ARG_UBATCH)  ·  **default:** `512 (common/common.h:444, src/llama-context.cpp:3523)`

CLAMP: cparams.n_ubatch = min(n_batch, n_ubatch == 0 ? n_batch : n_ubatch) (247). Two surprises there: -ub larger than -b is silently reduced to -b, and `-ub 0` means "= n_batch", not "zero". `-b 0 -ub 0` together is a hard error (3571-3574). n_ubatch is the single knob that sizes the worst-case compute buffer: the reserve uses n_tokens = min(n_ctx, n_ubatch) for the pp graph (595, 826) and llama_decode splits the batch by cparams.n_ubatch (1739).

`common/arg.cpp:1665-1671` · `common/common.h:444` · `src/llama-context.cpp:247` · `src/llama-context.cpp:595` · `src/llama-context.cpp:826` · `src/llama-context.cpp:1739` · `src/llama-context.cpp:3523`

### `Server-side -b/-ub overrides`
**Use:** Automatic in llama-server.  ·  **default:** `n/a`

(1) With --embedding, if n_batch > n_ubatch the server sets n_batch = n_ubatch and warns (server.cpp:145-148). (2) A non-splittable task larger than n_ubatch is rejected with "input (%d tokens) is too large to process. increase the physical batch size" (3099-3105). (3) On a KV-space failure the server HALVES its working n_batch and retries (3641), so an observed effective batch can be smaller than -b with only a SRV_WRN in the log.

`tools/server/server.cpp:145-148` · `tools/server/server-context.cpp:3015-3016` · `tools/server/server-context.cpp:3096-3105` · `tools/server/server-context.cpp:3641-3646`

### `CUDA graph capture — enabled, and how it arms`
**Use:** Compiled in (-DGGML_CUDA_USE_GRAPHS present in this build). Kill switch: set GGML_CUDA_DISABLE_GRAPHS to any value.  ·  **default:** `on for cc >= GGML_CUDA_CC_VOLTA (700); Ada 890 qualifies`

WARMUP RULE (the important one): a graph is only captured after TWO consecutive calls whose node properties are unchanged. First call with a new/changed property set executes eagerly and sets nothing; the second identical call sets warmup_complete and captures (4253-4262). After warmup, ANY property change resets warmup_complete back to false and drops to eager execution (4265-4268). Properties compared are the full ggml_tensor struct plus every src's data pointer, ne and nb, for every node (2600-2617).

`ggml/src/ggml-cuda/common.cuh:1225-1260` · `ggml/src/ggml-cuda/common.cuh:1255-1259` · `ggml/src/ggml-cuda/ggml-cuda.cu:4218-4231` · `ggml/src/ggml-cuda/ggml-cuda.cu:4234-4289` · `ggml/src/ggml-cuda/ggml-cuda.cu:4249-4274` · `ggml/src/ggml-cuda/CMakeLists.txt:134-137`

### `CUDA graph key — several graphs CAN coexist, but not per decode length`
**Use:** Implicit. Key = cgraph->nodes[0] (the first node pointer of the split graph handed to the backend).  ·  **default:** `n/a`

ggml_backend_cuda_context holds std::unordered_map<const void*, ggml_cuda_graph>. The in-source comment states the purpose: "allows multiple graphs per context when the computation is split across CPU/GPU (e.g., with --n-cpu-moe)" (common.cuh:1426-1427). It keys on WHICH SPLIT, not on how many tokens the step had. Two decode lengths that produce the same first-node pointer share one ggml_cuda_graph entry and therefore contend for the same warmup state. EVICTION: every 5 s the map is swept and any graph unused for >= 10 s is destroyed (common.cuh:1435-1444), so the first decode after an idle gap pays a full re-capture.

`ggml/src/ggml-cuda/ggml-cuda.cu:2574-2576` · `ggml/src/ggml-cuda/common.cuh:1426-1455` · `ggml/src/ggml-cuda/common.cuh:1432-1444`

### `The uid fast path — how a replay avoids re-scanning node properties`
**Use:** Implicit; driven by llama.cpp's graph reuse.  ·  **default:** `n/a`

Each split graph gets a fresh uid from ggml_graph_next_uid() every time ggml_backend_sched_split_graph runs (ggml-backend.cpp:1538). If cgraph->uid matches the stored one, ggml_cuda_graph_update_required returns false immediately and logs "CUDA Graph id %zu reused" (2584-2590). That only happens when the scheduler did NOT re-split, i.e. when llama.cpp reused the previous graph and sched->is_alloc was still true (ggml-backend.cpp:1955-1959).

`ggml/src/ggml-cuda/ggml-cuda.cu:2578-2591` · `ggml/src/ggml-backend.cpp:1074` · `ggml/src/ggml-backend.cpp:1538` · `ggml/src/ggml-backend.cpp:1949-1961` · `ggml/src/ggml.c:56-62`

### `llama-level graph reuse — the gate that decides whether the CUDA graph survives`
**Use:** On by default. Disable with LLAMA_GRAPH_REUSE_DISABLE=1 (logs "graph reuse disabled").  ·  **default:** `enabled`

llm_graph_params::allow_reuse requires ubatch.n_tokens == other.ubatch.n_tokens (llama-graph.h:785), plus equal n_seq_tokens, n_seqs, n_seqs_unq, matching seq ids, and n_outputs == other.n_outputs (813-815). So the ggml graph is rebuilt from scratch the moment the number of query tokens per step changes. Chain: n_tokens changes -> no reuse -> sched reset + re-split -> new uid -> node properties differ -> CUDA graph warmup reset. A workload whose step size alternates (accepting a variable number of drafted tokens) therefore never gets past warmup and runs eagerly. Counter n_reused is incremented on each reuse (1348) and is reported by llama_perf_context_print.

`src/llama-context.cpp:278-286` · `src/llama-context.cpp:1332-1372` · `src/llama-context.cpp:1339` · `src/llama-graph.cpp:1379-1409` · `src/llama-graph.h:781-816`

### `CUDA graph compatibility check`
**Use:** Implicit.  ·  **default:** `n/a`

The ONLY thing that makes a graph incompatible is a GGML_OP_MUL_MAT_ID node whose dispatch would take the fallback path requiring a stream sync (ggml_cuda_mul_mat_id_needs_sync, ggml-cuda.cu:1870-1898). Everything else — including large prompt-processing batches — is capturable. Note that the old "batch size > 1 disables CUDA graphs" rule is gone in this tree.

`ggml/src/ggml-cuda/ggml-cuda.cu:2541-2572` · `ggml/src/ggml-cuda/ggml-cuda.cu:2553-2565` · `ggml/src/ggml-cuda/ggml-cuda.cu:1869-1898`

### `GGML_CUDA_GRAPH_OPT (multi-stream QKV concurrency)`
**Use:** export GGML_CUDA_GRAPH_OPT=1  ·  **default:** `off (env must be exactly "1")`

Interleaves the Q/K/V branches across extra CUDA streams between a fan-out node and the join node (typically flash-attn). Requires CUDA graphs enabled AND exactly one CUDA device (4342). Fan-out window is hardcoded min_fan_out = max_fan_out = 3 (4394-4395), and candidate nodes must have ggml_nrows(node) <= 1 (4380-4382) — i.e. it only fires on single-row (decode-shaped) nodes.

`ggml/src/ggml-cuda/ggml-cuda.cu:4318-4344` · `ggml/src/ggml-cuda/ggml-cuda.cu:4329-4336` · `ggml/src/ggml-cuda/ggml-cuda.cu:4342` · `ggml/src/ggml-cuda/ggml-cuda.cu:4386-4396`

### `Query-tokens-per-step thresholds in the dense matmul kernels`
**Use:** Implicit; ne11 / src1_ncols is the token count of the ubatch.  ·  **default:** `n/a`

MMVQ_MAX_BATCH_SIZE = MMVF_MAX_BATCH_SIZE = 8. On NVIDIA, ggml_cuda_should_use_mmvq returns ne11 <= 8 (mmvq.cu:337). MMF (non-quantized) bails at src1_ncols > 16 (mmf.cu:176). Dispatch order in ggml_cuda_mul_mat is MMF -> MMVQ -> MMQ -> cuBLAS (ggml-cuda.cu:1853-1864). So for a quantized model the weight matmuls flip from the vector kernel to the tiled MMQ kernel at 9 query tokens per step — the same variable that flips FA from VEC to MMA at 3. NOTE: on Ada, ggml_cuda_should_use_mmq returns true unconditionally for supported types because turing_mma_available short-circuits at mmq.cu:312-314, so MMQ_DP4A_MAX_BATCH_SIZE (64) is dead here and -ub has no effect on MMQ-vs-cuBLAS.

`ggml/src/ggml-cuda/mmvq.cuh:3` · `ggml/src/ggml-cuda/mmvf.cuh:3` · `ggml/src/ggml-cuda/mmvq.cu:289-337` · `ggml/src/ggml-cuda/mmf.cu:133-178` · `ggml/src/ggml-cuda/mmq.cu:258-326` · `ggml/src/ggml-cuda/ggml-cuda.cu:1853-1865`

### `--op-offload / --no-op-offload and its batch threshold`
**Use:** --no-op-offload to disable; env GGML_OP_OFFLOAD_MIN_BATCH=N to move the threshold  ·  **default:** `enabled; min batch 32`

ggml_backend_cuda_device_offload_op returns get_op_batch_size(op) >= 32. get_op_batch_size is op->ne[1] for MUL_MAT, op->ne[2] for MUL_MAT_ID/ROPE, ggml_nrows otherwise, and 0 for GET_ROWS (5312-5313, so get_rows never offloads). Verified in --help: "--op-offload, --no-op-offload ... (default: true)". Only matters for tensors that live in host memory.

`ggml/src/ggml-cuda/ggml-cuda.cu:5501` · `ggml/src/ggml-cuda/ggml-cuda.cu:5311-5330` · `src/llama-context.cpp:275` · `src/llama-context.cpp:604`

### `graph_max_nodes — the ceiling on graph size`
**Use:** Implicit; computed from n_tokens = min(n_ctx, n_ubatch) at reserve.  ·  **default:** `max(1024, 8 * model.n_tensors()) for ordinary archs`

Hybrid/linear archs (QWEN3NEXT, QWEN35, QWEN35MOE, DEEPSEEK4, MINIMAX, NANBEIGE, BAILINGMOE3, DFLASH-with-hc) instead get max(n_tokens*40, 32*n_tensors); KIMI_K3 gets n_tokens*160 with a comment that "the n_tokens*40 budget below is exhausted at ubatch 3840". The DFlash2 patch adds 32*selector_tokens for LLM_ARCH_DFLASH with dflash_selector_rank > 0 (2317-2321) — that is the only line the DFlash2 commit touched in this whole area.

`src/llama-context.cpp:2294-2325` · `src/llama-context.cpp:596-604`

### `Prompt-processing and token-generation reserve passes`
**Use:** Implicit at context construction and after any memory update.  ·  **default:** `pp at n_tokens = min(n_ctx, n_ubatch); tg at n_tokens = n_seq_max`

Order is: resolve_fused_ops (FA probe, 1 token/seq) -> pp reserve -> tg reserve -> pp reserve AGAIN "to avoid ggml-alloc reallocations during inference" (662-671). The compute buffer is whatever survives that sequence. The log lines "graph nodes = %d (with bs=%d), %d (with bs=1)" and "graph splits = ..." (686-697) come from these two shapes and are the cheapest way to see whether pp and tg have different topologies. If the pp reserve fails and pipeline parallelism was on, it retries once with it off (633-641).

`src/llama-context.cpp:576-664` · `src/llama-context.cpp:595` · `src/llama-context.cpp:618-649` · `src/llama-context.cpp:651-660` · `src/llama-context.cpp:662-671` · `src/llama-context.cpp:686-697`

### `Where the FLASH_ATTN_EXT node is actually built`
**Use:** Implicit.  ·  **default:** `n/a`

use_flash_attn = cparams.flash_attn && kq_b == nullptr (2540) — a model needing a KQ bias never gets FA, with the assert message "Flash attention does not support KQ bias yet". F32 K/V are cast to F16 in the graph itself (2549-2555). Precision is pinned: ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32) at 2562. The node is registered as LLM_FUSED_OP_FLASH_ATTN (2559) which is exactly what the -fa auto probe looks for.

`src/llama-graph.cpp:2540-2565` · `src/llama-graph.cpp:2549-2555` · `src/llama-graph.cpp:2562` · `src/llama-graph.cpp:34-41`

**What this area CANNOT do.** Things this area CANNOT do, each with the line that forecloses it. 1. You cannot choose the FA kernel. There is no flag, no env var, no config key. ggml_cuda_get_best_fattn_kernel (ggml/src/ggml-cuda/fattn.cu:358-533) is called fresh on every FLASH_ATTN_EXT node on every call and its inputs are cc, head dims, K/V types, gqa_ratio and Q->ne[1]. The only lever you have is Q->ne[1], i.e. how many query tokens you put in a step. 2. You cannot make several decode lengths coexist as captured CUDA graphs. The graph key is cgraph->nodes[0] (ggml-cuda.cu:2574-2576) — which split, not which shape — and the map's stated purpose is CPU/GPU splits (common.cuh:1426-1427). Upstream of that, llm_graph_params::allow_reuse requires ubatch.n_tokens == other.ubatch.n_tokens (src/llama-graph.h:785), so any change in step size rebuilds the ggml graph, which re-splits, which changes node properties, which resets warmup_complete (ggml-cuda.cu:4265-4268). Re-arming needs two consecutive identical calls (4255-4261). An alternating step size therefore never captures. 3. You cannot mix K and V cache types on CUDA in this binary. ggml/src/ggml-cuda/fattn.cu:442-446 returns BEST_FATTN_KERNEL_NONE when K->type != V->type unless GGML_CUDA_FA_ALL_QUANTS is defined, and it is not (ggml/CMakeLists.txt:208 default OFF; grep of build-dflash2/compile_commands.json returns 0 hits). 4. q4_1, q5_0 and q5_1 KV caches have no CUDA FA kernel here. ggml/src/ggml-cuda/fattn.cu:346-348 returns false for them without FA_ALL_QUANTS. Only f16, bf16, q4_0, q8_0 (and F32, converted) work. 5. You cannot keep the memory saving of a quantized KV cache on the MMA/TILE path. ggml/src/ggml-cuda/fattn-mma-f16.cuh:1962-1963 and fattn-tile.cuh:1165-1166 pass need_f16_K = need_f16_V = true unconditionally, and fattn-common.cuh:1022-1084 then expands the whole cache to F16 into scratch that ggml-cuda.cu:906-912 charges to the compute buffer. 6. `-fa on` does not fail loudly on an unsupported configuration. The device-mismatch probe runs only under `if (cparams.auto_fa)` (src/llama-context.cpp:554), and auto_fa is set only for AUTO (229-230). With `-fa on` and a combination CUDA cannot serve, ggml_backend_cuda_device_supports_op returns false (ggml-cuda.cu:5286-5289) and the FLASH_ATTN_EXT node is scheduled on the CPU backend, which accepts it (ggml/src/ggml-cpu/ggml-cpu.c:2398, 2951). The result is a working server that is enormously slower, with no error. 7. The startup log does not tell you the resolved FA state. src/llama-context.cpp:312 prints llama_flash_attn_type_name(params.flash_attn_type) — the value you asked for, not cparams.flash_attn after the probe. The only truthful lines are the probe's own "Flash Attention enabled" / "Flash Attention not supported, set to disabled" (532-548). 8. The vec (decode) kernel processes at most 2 query columns. fattn-vec.cuh:553-572: cols_per_block is 1 if Q->ne[1] == 1, otherwise the constant 2. There is no wider vec instance to fall into. 9. FA cannot be enabled per layer. cparams.flash_attn is a single bool for the whole context (src/llama-context.cpp:229) and the probe disables it globally on the first mismatching layer (src/llama-context.cpp:538-548). 10. `-ub` cannot exceed `-b`, and `-b` cannot exceed `-c` under causal attention. Both are silent min() clamps: src/llama-context.cpp:245 and 247. Neither prints a warning. 11. A quantized V cache cannot be used without FA, in either direction. src/llama-context.cpp:3607-3610 refuses at init with `-fa off`; src/llama-context.cpp:463-467 throws "quantized V cache was requested, but this requires Flash Attention" if the auto probe turns FA off after the fact. 12. FA cannot be combined with a KQ bias (ALiBi-style additive bias tensor). src/llama-graph.cpp:2540-2542: use_flash_attn requires kq_b == nullptr, with a hard assert. 13. Head dims outside {40,64,72,80,96,112,128,192,256,320,512,576} get no CUDA FA kernel at all — fattn.cu:435-437 default: return BEST_FATTN_KERNEL_NONE. 14. On this GPU the TILE kernel is unreachable for head dim 128. fattn.cu:461 excludes only Q->ne[0] == 40 and 72 from the Turing+ branch, and that branch always returns VEC or MMA_F16 (462-482). So TILE tuning is irrelevant to a 128-head-dim model on Ada. 15. CUDA graphs cannot be captured on pre-Volta hardware (ggml-cuda.cu:4222, disable_due_to_gpu_arch) and are killed outright by the mere presence of GGML_CUDA_DISABLE_GRAPHS in the environment (common.cuh:1258 — the value is not parsed, only existence).

**Exists but unused in our profile.** A profile of `--spec-type ngram-mod, -ctk q4_0 -ctv q4_0, -ngl auto --fit on, -np 1` leaves the following capability surface untouched. Listed because a later reader will otherwise waste time tuning them. - GGML_CUDA_FA_ALL_QUANTS and everything it unlocks. Not compiled in (ggml/CMakeLists.txt:208; absent from build-dflash2/compile_commands.json). The 49 mixed K/V vec instances at fattn.cu:265-323 do not exist in this binary; only the four at fattn.cu:284-289 do. Rebuilding with it ON is the only way to reach q5_0/q5_1 KV or an asymmetric -ctk/-ctv, and it costs compile time and binary size. - The TILE kernel. Ada satisfies turing_mma_available (common.cuh:348-350, cc 890 >= 750), so fattn.cu:461-483 always returns before the TILE branches at 492-532 for head dim 128. All of ggml/src/ggml-cuda/fattn-tile.cuh is dead weight for this profile. - The Volta and AMD selection branches: fattn.cu:66-88 (Volta ncols2), 490-498 (volta_mma_available), 501-511 (amd_mfma_available), 514-517 (amd_wmma_available), and the CDNA/RDNA tables in mmvq.cu:293-335 and mmq.cu:376-383. - Large-head-dim MLA paths. fattn.cu:141-155 (DK 192, MiMo-V2.5), 161-176 (DK 320, Mistral Small 4), 180-234 (DK 576, DeepSeek / GLM 4.7 Flash) with their per-architecture Q->ne[1] and K->ne[1] cutoffs. A Qwen3-family head dim 128 goes through fattn.cu:137-140 only. - Pipeline parallelism and multi-copy scheduling. cparams.pipeline_parallel requires model.n_devices() > 1 (src/llama-context.cpp:427-433); on one GPU it stays false, so GGML_SCHED_MAX_COPIES=4 in the build is inert and the pipeline-parallel synchronize at src/llama-context.cpp:1343-1345 never fires. - LLAMA_SPLIT_MODE_TENSOR's forced-FA path (src/llama-context.cpp:3586-3595) and the tensor-split fitting in common/fit.cpp — single device. - The multi-graph CUDA map beyond one entry. With -ngl auto --fit on placing every layer on the GPU there is one GPU split, hence one key in cuda_graphs (common.cuh:1428). The map exists for --n-cpu-moe-style CPU/GPU interleaving, per its own comment at common.cuh:1426-1427. Its 10 s eviction / 5 s sweep (1435-1444) still applies to that single entry after an idle gap. - Multi-stream KV. -np 1 means n_stream == 1, so kq_mask ne[3] == 1 (src/llama-graph.cpp:950) and the per-stream branches in src/llama-graph.cpp:2532-2534 and the seq_id_unq comparison in llama-graph.h:797-807 are trivially satisfied. The kv_unified vs non-unified distinction (common/arg.cpp:1714-1719, src/llama-context.cpp:290-302) collapses. - --op-offload / GGML_OP_OFFLOAD_MIN_BATCH=32 (ggml-cuda.cu:5501, 5327-5330). It only decides whether an op on a HOST-resident tensor is shipped to the GPU; with the whole model on the GPU there are no such ops. - GGML_CUDA_GRAPH_OPT multi-stream QKV concurrency (ggml-cuda.cu:4318-4344). Off unless the env var is exactly "1", and additionally requires a single CUDA device — which is satisfied, so this is the one item on this list that is cheap to try. - MMQ_DP4A_MAX_BATCH_SIZE = 64 (mmq.cuh:8) and the whole non-tensor-core MMQ heuristic (mmq.cu:325-383): unreachable because turing_mma_available short-circuits to `return true` at mmq.cu:312-314. - The non-causal / encoder paths: src/llama-context.cpp:1426 (encoder requires n_ubatch >= n_tokens) and 1713 (non-causal requires n_ubatch >= n_tokens), plus the embedding-mode n_batch = n_ubatch override at tools/server/server.cpp:145-148. - LLAMA_GRAPH_REUSE_DISABLE (src/llama-context.cpp:279-285) and GGML_CUDA_DISABLE_GRAPHS (common.cuh:1258): diagnostic kill switches, not part of a fast profile — though flipping each one and re-measuring is the cheapest way to attribute a regression to graph reuse vs graph capture. - The DFlash2 additions themselves. Across the two commits on top of master, the only line touching this area is the LLM_ARCH_DFLASH node budget at src/llama-context.cpp:2317-2321; the rest lives in common/speculative.cpp, common/sampling.cpp and src/models/dflash.cpp. With --spec-type ngram-mod the DFLASH speculator is not selected (common/speculative.cpp:2354-2360 vs 2372-2374).
