# Cut re-prefill & prefill time per task — cross-source research (2026-09-02)

Hardware: RTX 5060 Ti 16GB + RTX 4070 SUPER 12GB, Windows, llama.cpp build 10729 (458681e1d).
Profile 2 = `worker-q4-dual.ps1 -Nvfp4` — NVFP4-MTP, `-sm tensor`, ctx 200,704, `-ctk/-ctv q4_0`, `--cache-ram` default 8192 MiB, `-mm` (multimodal).
Source sweep: 4 parallel research agents (deleg_9f9d367c) over llama.cpp source, other engines, Reddit/HF, and client-side.

## The problem, in one line
Last 30 minutes of a real 160k-context session: **68% of wall (1,229s of 1,801s) was forced full re-prefill** for just 7,024 decoded tokens (prefill:decode ≈ 147:1). Cause: a single conversation's cache state grew to ~9,801 MiB, exceeding the `--cache-ram` 8192-MiB cap, so llama.cpp **skips saving state entirely** ("exceeds cache size limit ... skipping") and every sub-agent task replays ~160k tokens (200–250s each).

## Root-cause findings (trust, then levers)

### A. `--cache-ram` cap is GLOBAL, and skip ≥ evict
- Source `server-task.cpp`: `if (limit_size > 0 && state_size_new > limit_size) SKIP` — when state exceeds the cap it is NOT evicted, it is **not cached at all**. That's why every task pays full price once the conversation outgrows 8192 MiB.
- **Lever:** raise `--cache-ram` above real state, e.g. `16384` (or `-1` unlimited). On 47.7 GB RAM this is safe. `--cache-ram 0` disables cache (the profile does NOT pass 0 — verified argv; it runs at llama.cpp default 8192).
- Cross-checked: unsloth#9037, reddit 1td9stc.

### B. Prefix must be byte-identical across tasks (biggest free win)
- llama-server is already delta-based under `cache_prompt=true`: it matches the new prompt against the cached slot stream and evaluates only the unseen suffix (server README, maintainer-written).
- **The cache killer we must rule out first:** `CLAUDE_CODE_ATTRIBUTION_HEADER` — a changing attribution header (client version + prompt fingerprint) at the start of the system prompt resets the prefix. **Set `CLAUDE_CODE_ATTRIBUTION_HEADER=0`.** Real Qwen3.6-27B llama-server workload: fixed → full 160k re-prefill collapses to a ~212-token tail eval (mykolaaleksandrov, 2026-06).
- Also freeze tool-schema ordering and prior-history representation across sub-agent turns (particula.tech prompt-reprocessing breakdown: both are #1 triggers for llama.cpp giving up on cache).
- **Screen first:** open issue #18497 "cache-reuse not effective in qwen3-next" reports the family giving up on a stable prefix. Before investing in client shaping, run `-lv 4` and grep "forcing full prompt re-processing" to rule the family bug in/out on build 10729.

### C. Slot save/restore — the only cache-ram-independent warm path
- `--slot-save-path` + `POST /slots/{id}?action=save|restore` moves the full sequence (prompt+KV) to disk, independent of the RAM cap. ggerganov confirmed it as the supported way to keep N long prefixes warm, and it survives server restart.
- **Use this for the 9,801-MiB state** instead of relying on the 8192-MiB RAM cache. The only path that sidesteps the cap entirely.

### D. Keep the warm slot from being stolen
- `--parallel 1` + raise `--cache-idle-slots` / `-sps` prevents LRU from evicting the active conversation's checkpoints between sub-agent turns (the exact "15 checkpoints nuked, n_past=3" failure). ~2 GB RAM overhead, ~93% TTFT cut for cached requests.

### E. Recurrent-state root fix (Qwen3.8 hybrid)
- Qwen3.8 has 48/65 recurrent (Gated DeltaNet) blocks → checkpoint invalidation on hybrid. Fix: `recurrent_shrink`/`recurrent_expand` so the recurrent state is shrunk to 1 cell before prompt-cache save/load then re-expanded — **kills full re-processing across turns** (PR #24785 / BeeLlama fork; 38K ctx 207s/turn → 5 turns all "No re-processing" at 63–93 tok/s).

### F. Prefill throughput (for the unavoidable full replays)
- `-ub 512→2048` (+1.8× prompt eval: 1119→2001 tok/s, dzx.fr RTX 5080).
- `Q8_0` KV cache (halves memory, +0.004 ppl; Q4_0 can be 92% slower at 64K) — also directly fights the >8192-cap overflow.
- `-b 2048` for fewer forward passes on big prompts.

### G. Other engines — not transferable to in-process single-slot
- vLLM Automatic Prefix Caching / radix attention and SGLang session-aware eviction are architecturally different (block-level shared KV across requests); not stealable into a single-slot llama-server.
- ExLlama3 `sysmem_recurrent_cache` (RAM offload of recurrent state) is the closest analogue to slot-save, but lives outside llama.cpp.

## Recommended action (by leverage)
1. **`--cache-ram 16384`** in worker-q4-dual.ps1 — stops the global SKIP. (Opus 5's original proposal; verified safe on 47.7 GB RAM.)
2. **`CLAUDE_CODE_ATTRIBUTION_HEADER=0`** + byte-stable prefix + frozen tool-schema order — free client-side, attacks the 160k replay at root.
3. **`--slot-save-path`** for the big 9,801-MiB state — bypasses the RAM cap entirely, survives restart.
4. Optional: `--parallel 1`, `-ub 2048`, Q8_0 KV; then evaluate recurrent_shrink/expand (#24785) if checkpoints still invalidate.
5. **Verify first** on build 10729: `-lv 4` + grep "forcing full prompt re-processing" to rule out the qwen3-next family bug (#18497) before committing to client shaping.

## Sources
- llama.cpp server README (cache_prompt delta, slot save/restore, cache-idle-slots, n_predict=0): tools/server/README.md
- server-task.cpp (cache-ram SKIP condition)
- ggml-org/llama.cpp discussions/8860 (maintainer: shared-prefix reuse, slot pinning, slot save/restore)
- ggml-org/llama.cpp PR #24785 (recurrent shrink/expand) · issue #18497 (qwen3-next cache-reuse)
- Unsloth Studio issue #9037, PR #24190 (cache-idle-slots), r/LocalLLaMA 1td9stc
- mykolaaleksandrov.dev/posts/2026/06/claude-code-llamacpp-prompt-cache-fix (CLAUDE_CODE_ATTRIBUTION_HEADER)
- particula.tech/blog/prompt-reprocessing-swa-hybrid-models-kv-cache (prefix stability triggers)
- dzx.fr (ubatch prefill 1119→2001 tok/s) · omniforge.online/blog/your-local-llm-is-slow-because-of-five-config-flags (Q8_0 KV)
- vLLM docs automatic_prefix_caching · ExLlama3/GB10 notes
---

## Checked against this machine's source and logs — 2026-09-02

Added by the session that shipped `--cache-ram 24576`. Source read is
`F:\llama-build\up` at **458681e1d** — the tree build 10729 was compiled from,
which is what icon 2 serves. Log is `logs/serve-20260902-034815.log`.

**Three of the seven sections survive, three do not, and one is out of scope.**

| § | verdict |
|---|---|
| **A** `--cache-ram` cap is global, skip ≥ evict | **CONFIRMED and SHIPPED.** `--cache-ram 24576` is served from 2026-09-02. But **not `-1`** — see below. |
| **B** byte-identical prefix / attribution header | **REFUTED HERE, and out of scope.** |
| **C** `--slot-save-path` | **REAL but dominated.** Same mechanism, slower medium, and it needs a client that POSTs. |
| **D** raise `--cache-idle-slots` / `-sps` | **NOT A THING.** The flag is a boolean; `-sps` is unrelated and inert at `-np 1`. |
| **E** recurrent shrink/expand, PR #24785 | **NOT IN THIS BUILD.** Needs a rebuild before it is even a question. |
| **F** `-ub 2048` / `q8_0` KV | `-ub` open (task #41). **`q8_0` REFUTED HERE at −18.39 %**, and it makes A worse, not better. |
| **G** vLLM / SGLang / ExLlama3 | fine, and correctly marked non-transferable. |

### A — right, but `-1` is a trap and would have been a regression

`server-task.h:613` — `limit_size = 1024*1024*(limit_size_mib < 0 ? 0 : limit_size_mib)`,
so `-1` means **`limit_size == 0`**. Then in `update()` (`server-task.cpp:1870`):

```cpp
const size_t limit_tokens_cur = limit_size > 0
    ? std::max<size_t>(limit_tokens, limit_size/size_per_token)
    : limit_tokens;
```

**The dynamic raise is gated on `limit_size > 0`.** With `-1` the token cap stays
pinned at its constructor value, `limit_tokens = n_ctx = 200,704`
(`server-context.cpp:1359`), while the two live conversations are 167k + 46k =
**213k tokens** — so the cache evicts by tokens even with unlimited RAM.

Replaying the log's own recorded entry sizes through the same `alloc()`/`update()`
arithmetic, counting how many of the 52 forced re-prefills would have found their
prefix:

```
-cram   8192 (the default)     0 / 52      84 evictions
-cram  16384                  35 / 52      82
-cram  24576 (served now)     43 / 52      80
-cram     -1 (no size limit)  13 / 52      84
```

**`-1` is worse than half of `16384`.** *(This is a SIMULATION on recorded entry
sizes, not a measurement; the prefix test is a heuristic, not the server's
`f_keep`/`f_sim` rule.)*

**24576 was rejected on host memory, and the rejection was wrong — corrected the
same day, and it is now what we serve.** The reasoning was: commit **87.8 GB used
of a 104.3 GB limit**, so `24576` at +16 GiB "would leave under a gigabyte". Two
errors in one sentence.

**`--cache-ram` is a cap, not a reservation.** `alloc()` resizes only to the state
actually being stored (`server-task.cpp:1760-1770`), so raising the cap costs the
difference the cache really holds — about **4–6 GB**, not 16 — against 16.5 GB of
free commit. And **the commit limit is not fixed**: `AutomaticManagedPagefile` is
**True**, on a 932 GB **WD_BLACK SN850X**, measured here at **1,809 MB/s write and
5,332 MB/s read**. A 7 GiB entry faulted back costs **~1.3 s** against the
200–250 s re-prefill it replaces — about **170×** cheaper.

**And the machine was already doing this.** `llama-server` holds **34.5 GB private
against a 4.5 GB working set**, with hard faults running 1–577/s while it serves.
"It would page" described the status quo, not a new risk. What *is* a real risk is
Windows choosing to page something hot instead — the CUDA host allocations that
mirror VRAM — and that is visible as decode falling while hard faults climb.

The remaining wall is disk space, not memory: **C: has 25 GB free of 931**, and the
pagefile already occupies 56.4 GB of it. D: has 11 GB; F: has 132 GB but is USB.

**Why 24576 wins is worth knowing**, because it names the next lever: entries are
40 % checkpoint, and on a recurrent model a checkpoint carries the SSM state
whatever the prompt length — a **526-token** prompt occupies **463 MiB** of cache.
Three or four of those are what push the real pair over 16,384.
`--ctx-checkpoints` is where that goes, and it was deliberately not changed in the
same commit.

### B — our own log refutes the premise, before scope enters

The developer has ruled client-side work out of scope. Separately, **the premise
is wrong for this machine**: if a changing attribution header were resetting the
prefix, every turn would re-prefill. Ours do not.

```
srv  load: - prompt with length 35733, lcp = 35733, f_keep = 1.000, f_sim = 0.883
slot     : task 70303 | checking checkpoint with [45590, 45590] against 45594...
slot     : task 70303 | restored context checkpoint (pos_min = 45590, n_past = 45591)
```

`lcp` equal to the whole cached prompt, and `n_past` 45,591 of 45,595 — **the
prefix is byte-stable within a conversation.** 220 checkpoint restores and 37
prompt-cache restores succeeded. The re-prefills happen only on a **swap between
two conversations**, where the shared prefix is **3 tokens** and no prefix
stability could help.

### C — the same mechanism, a slower medium, and no caller

`--slot-save-path` writes via `llama_state_seq_get_data` — the call the prompt
cache already uses. It trades RAM for disk, which is the wrong direction unless
RAM is the binding constraint, and it only moves when **the client POSTs
`/slots/{id}?action=save`**. Nothing does that here.

### D — the flag does not take a number

```
{"--cache-idle-slots"}, {"--no-cache-idle-slots"},
"save idle slots to the prompt cache on new task ... (default: enabled, requires cache-ram)"
```

`common/arg.cpp:1729-1735`. It is a boolean and **already on** — the boot log says
`idle slots will be saved to prompt cache upon starting a new task`. `-sps` is
`--slot-prompt-similarity` (`arg.cpp:3804`), a slot-*selection* threshold; with
`-np 1` there is one slot and the log shows it falling through to LRU anyway.

### E — not in the binary we serve

`grep -rn "recurrent_shrink\|recurrent_expand" src/ tools/` returns nothing at
458681e1d. PR #24785 would need a rebuild before it is testable, and this project
has a rebuild procedure (`build-dflash2.ps1`) for exactly that.

### F — one open, one refuted here

`-ub 2048` is open and already tracked as task #41; the profile serves `-b 2048
-ub 1024`, and 1024 was measured at +10.1 % over 512 *here*.

**`q8_0` KV is measured and it loses**, at the served depth on this artifact
(`results/05-runtime-flags.md`, issue #46, three rounds each, every row `66+0`):

| arm | mean | vs served | `free_after` |
|---|---|---|---|
| `q4_0`/`q4_0` (served) | **46.04** | baseline | 2,600 MiB |
| `q8_0`/`q4_0` | 41.06 | −10.82 % | 1,880 |
| `q8_0`/`q8_0` | 37.58 | **−18.39 % RESOLVED** | 1,158 |

And the note has the cache argument backwards. `q8_0` is **twice** `q4_0`, not
half — it halves against `f16`. The prompt-cache entry *is* the KV, measured at
**0.0224 MiB/token** under `q4_0`, so `q8_0` would roughly double every entry and
make the 8,192-cap overflow **worse**.

## Post-restart verification (2026-09-02, session serve-20260902-094554) — what it does and does not prove

Served with `--cache-ram 24576` + `--ctx-checkpoints 4` (both shipped). Verified from the real boot log:

**Proven: re-prefill is gone.** The log shows prefix reuse working, not full re-prefill:
```
cached n_tokens = 38352, memory_seq_rm [38352, end)
prompt processing, n_tokens = 6300 -> 7324   (843 tok/s)
```
It pre-fills only the appended tail (~6–7k tokens) after restoring a 38k cached prefix — `memory_seq_rm` + partial prompt eval = the cache hit. Restores succeeded (243 in this session). No `exceeds cache size limit`, no `making room` since boot.

**NOT proven / not a decode win:** a `tg ≈ 50 tok/s` row in the same log was at **ctx 39,536** (init_sampler `text = 39536`), a shallow-to-mid context — that is the normal rate there, NOT evidence that deep-context decode improved. The known rule still holds: `decode@100k+` on this hardware is 30–40 tok/s by KV-bandwidth limit, and no cache flag changes that.

**Verification gate for decode@deep (still open):** confirm `tg` at actual `ctx ≥ 100k` on a long run. Expect 30–40 as the baseline, matching the developer's own measured rule; anything above that would be the first real evidence a change helped decode rather than only removing re-prefill. This row is pending that confirmation. (The `tg_3s`/`n_gen` intermediate rows are rolling averages, not per-step decode speed.)

## Lead (จดไว้): FreeToken engine — 2026-09-02 (Kintu Substack benchmark)
- FreeToken (FlashML): claim ~2x MoE-offload decode on same consumer GPU. llama.cpp 21 → FreeToken 40-42 tok/s, RTX 5060 Ti 16GB, Qwen 3.6-35B-A3B (~37GB offload). Mechanisms: bandwidth-adaptive execution, dynamic VRAM LRU expert cache, hybrid/direct CPU expert execution, dynamic KV↔expert rebalance.
- **Applicability: LOW for us** — win is MoE expert-offload; our Qwen3.8-27B is dense-hybrid (no `exps` experts, 48 SSM). Doesn't apply to llama.cpp/NVFP4 dense workload.
- llama.cpp already building dynamic LRU MoE caching (PR #27861, RFC #24528). Author notes FreeToken early-release (CUDA quirks, WSL2 bugs), llama.cpp wins TTFT/ecosystem. Revisit only if a real MoE target or full offload is ever needed.
