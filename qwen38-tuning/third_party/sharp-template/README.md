---
license: apache-2.0
base_model: froggeric/Qwen-Fixed-Chat-Templates
base_model_relation: finetune
tags:
  - jinja
  - chat-template
  - qwen
  - qwen3.5
  - qwen3.6
  - qwen3.8
  - mlx
  - llama.cpp
  - lm-studio
  - vllm
  - tool-calling
  - thinking
  - token-efficient
---

# Qwen Sharp Chat Templates

This is a drop-in fix for any Qwen3.5, 3.6, or 3.8 model, optimizing the models for knowledge work and coding.

<div align="center">
  <img src="plates/sharp_template_plate.png" alt="Qwen3.8-27b gets more accurate and uses fewer tokens with the template applied" width="100%">
</div>

With the *Sharp* template, Qwen3.8-27b (medium effort) gets smarter **and** uses fewer thinking tokens before it answers, and the effect is comparable for other compatible models.

<div align="center">
  <img src="plates/claweval_sharp_plate.png" alt="ThinkingCap-Qwen3.6-27B on Claw-Eval: answer score +7.4, overall +3.8, answer tokens -59% with the Sharp template" width="100%">
</div>

*Sharp* makes Qwen's models *more intelligent per token*, and makes them communicate *more information per token* by cutting filler without sacrificing correctness or substance, saving time and effort for both the model and the user in multi-turn conversations.

<div align="center">
  <img src="plates/swe_sharp_template_plate.png" alt="Qwen3.8-27b on SWE-bench-Live: 15 of 25 problems solved with the Sharp template against 16 stock, but a median 20.0 minutes to a fix against 54.6, both arms at medium effort" width="100%">
</div>

On real repository work the same effect shows up as speed: *Sharp* fixes about as many issues as the stock template, while reaching each one **2.7&times; faster** on the median problem.

## Straight to the point

This is froggeric's [Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
`v22.5` with a force-appended system prompt spliced in, plus this repo's revisions on top. The base
fixes issues, the addition makes it better, `v22.3.1` makes the fast (thinking-off) path coherent,
`v22.3.2` lets you switch the appended prompt off per request, `v22.4.1` stands the tool block down
when the runtime supplies its own tool protocol, and **`v22.5.0`** rebases onto froggeric's v22.5 —
see the changelog below.

> **`v22.5.0` (current).** Rebased from froggeric `v22.4` onto **`v22.5`**. Two conflicts, both
> resolved to upstream:
>
> 1. the version line, and
> 2. **the thinking-aware tool instructions — upstream has now fixed this itself.** It is the same
>    bug this fork fixed in `v22.3.1`: with tools offered and thinking *off*, the tool-call
>    instructions still showed a `<think>` block and said "IMMEDIATELY after thinking". Rendered
>    output is **byte-identical** between this fork's fix and upstream's across all four
>    xml/json x thinking-on/off combinations, so the local patch is retired in favour of upstream's
>    (their line structure, so the next rebase stays clean). Upstream's new test 102 passes against
>    `v22.4.1` unchanged, which is the same fact from the other direction.
>
> It picks up v22.5's two remaining changes, neither of which this fork had:
>
> - **`video_url` content items are accepted** alongside `video` — OpenAI-style clients previously
>   hit `Unexpected item type in content.`
> - **Tool-response truncation is narrowed.** `max_tool_response_chars` used to be skipped for the
>   whole of `tool_call_format='json'`; it is now skipped only when the payload actually looks like
>   JSON (starts with `{` or `[`), so a plain-text tool response truncates again in JSON mode.
>
> **Everything else renders byte-identically to `v22.4.1`** — verified over a 72-combination matrix
> (9 conversation shapes x thinking on/off x xml/json x terse on/off): the 12 that differ are the 8
> `video_url` cases and the 4 plain-text-in-json truncation cases, and nothing else moved.

> **`v22.4.1`.** Stands down when the runtime supplies its own tool protocol. LM Studio's
> MLX backend has no native tool-call parser: whenever tools are passed it prepends its own
> `[TOOL_REQUEST]` protocol into the system content. Emitting ours as well handed the model two
> contradictory tool protocols and it followed one at random — measured 2/6 tool calls parsed, the
> rest leaking raw `<tool_call>` text into the reply, which stalls agents in a retry loop. The
> template now detects that marker and omits its own tools block: **6/6 after the change**, with
> oMLX unaffected at 4/4 and llama.cpp/GGUF (which has a native parser and never injects) untouched.
>
> `suppress_tool_instructions` overrides in either direction wherever a runtime can pass template
> kwargs (llama.cpp `--chat-template-kwargs`, transformers, oMLX). LM Studio cannot, which is why the
> detection is automatic rather than a flag.
>
> **The default render is unchanged** — with no such marker present the output is byte-identical to
> `v22.4.0` across single- and multi-turn, both tool-call formats, and the no-tools path.

> **`v22.4.0`.** Rebased from froggeric `v22.3` onto **`v22.4`** (a clean three-way merge —
> only the version line conflicted; this repo's terseness block, thinking-on/off tool-call format, and
> identity handling are untouched). It takes v22.4's three functional changes:
>
> 1. **Parallel tool-call token parity.** Consecutive `<tool_call>` blocks in assistant history are now
>    separated by a single newline instead of two, restoring exact token alignment with official Qwen
>    generation and preventing prefix KV-cache divergence during multi-tool turns — the change that
>    matters most for agentic coding.
> 2. **`message.reasoning` extraction.** Reasoning history is now read from `message.reasoning` (vLLM's
>    OpenAI-compatible endpoints and Responses-API schemas) alongside `reasoning_content` and `thinking`.
> 3. **`_default_reasoning_effort` knob.** The `medium` default is now an explicit top-of-template
>    variable rather than a buried literal.
>
> **The default render is unchanged** — with no kwargs you get the same terse, thinking-off-by-default
> behaviour as `v22.3.2`, now on the v22.4 base. Validated: this repo's 74-check suite plus froggeric's
> property-based fuzz harness (all invariants over 500 generated conversations).
>
> **`v22.3.2`.** The terseness system prompt is now optional per request. Pass
> `{"terse": false}` in `chat_template_kwargs` and the block is not appended; the model runs on its
> own system prompt, or none. **The default is unchanged** — omit the kwarg and you get exactly what
> `v22.3.1` rendered, byte for byte, so every existing caller and every already-published build
> behaves as before. Nothing else changed. See [Turning terseness off](#turning-terseness-off).
>
> **`v22.3.1` (this repo's bugfix, rebased onto froggeric v22.3).** Three changes, all
> scoped to **thinking-off (fast) mode**; the thinking-**on** path is byte-identical to upstream
> v22.3 plus the terseness block, so anything already running with thinking on gets only upstream's
> fixes.
>
> 1. **Fast-mode `<think>` contradiction fixed.** With tools and thinking *off*, upstream still tells
>    the model to put its reasoning inside `<think></think>` — while the generation prompt has already
>    closed thinking. This gates every `<think>`-related tool-call instruction on thinking being on, so
>    fast mode no longer asks for a block it can't open. (Tool-call *format* rules are untouched.)
> 2. **Terseness lead split by mode.** The single lead — *"Answer directly, after thinking"* — is
>    incoherent when thinking is off, so thinking-off gets a terse-neutral *"Answer directly and
>    concisely"* lead instead. The terseness core (the never/always rules) is identical in both.
> 3. **The last thinking reference on the fast path removed.** One tool-call rule still read *"output
>    the `<tool_call>` block IMMEDIATELY **after thinking**"* in fast mode — the same contradiction as
>    (1), surviving in a line the first fix didn't cover. Only the two-word fragment is gated, so
>    thinking-off reads *"…IMMEDIATELY, with NO conversational text before it"* and thinking-on is
>    unchanged to the byte. With this, a fast-mode prompt contains no reference to thinking at all.
>
> (1) and (2) carry over unchanged from `v22.1.1`; (3) is new in `v22.3.1`. None is fixed upstream as
> of v22.3.
>
> **Version-string break.** `v22.3.1` does **not** contain `v22.1` as a substring, so any checker
> matching on the old id stops matching. This project's re-embed scripts (`publish/retemplate_dirk.py`,
> `publish/retemplate_dirk_v3.py`) now read the expected version out of the template file at run time
> instead of hardcoding it, so the next rebase won't break them again. Already-published GGUF/MLX
> builds still carry v22.1.1 and are unaffected until deliberately re-templated.

> **What upstream added in v22.2 / v22.3 (and what it closed for us).** Rebasing picked up, verified
> by upstream's own suite:
>
> - **Long tool errors escalate again.** The old `content|length < 500` gate meant a multi-line
>   traceback — the exact case where escalation matters — silently *never* escalated. v22.2+ replaces
>   it with two tiers: structural signals (`"error":`, `"status": "error"`, a nonzero exit code, a
>   real `Traceback (most recent call last):`) escalate at any payload size, while weak signals stay
>   size-gated. This is one of the two issues this repo previously listed as open and deferred —
>   **it is now fixed**, upstream, and better than the patch that regressed here.
> - **False retry loops on code search killed.** Grep hits containing `throw new Error(...)`,
>   `console.error`, `logger.error` no longer count as tool failures.
> - **Multiple leading system/developer messages merge** into one system turn joined by blank lines,
>   instead of only the first being treated as a system prompt.
> - **In-content reasoning extraction widened** to content that *starts* with `<think>`/`<thinking>`,
>   and de-duplicated when `reasoning_content`/`thinking` is supplied alongside inline tags.
> - **Preserved assistant turns always render the `<think>` wrapper**, even when the thought was
>   empty, so rendered history matches what was actually generated and the prefix cache stays valid.
> - **Tool arguments serialize correctly.** Booleans, nulls and numbers now go through `tojson`
>   instead of `| string` (which emitted Python `True`/`None`); raw string args honour
>   `max_tool_arg_chars`; JSON tool format no longer truncates tool responses.
> - **More effort aliases:** `off`, `max`, `ultracode`, `extreme`, with matching `<|think_…|>` tags.
>
> **Still open (so you don't over-trust it):** a literal `<|think_off|>` arriving in tool output still
> disables reasoning **if** your harness packs tool results into a `user` message — upstream's tag
> scanner reads `system`/`developer`/`user` roles, so a proper `tool`-role message is safe, and that is
> unchanged in v22.3. And a separately-reported *mid-answer* `<think>` tag remains unreproduced in our
> stack (2,863 corpus messages + 14 fresh llama.cpp generations, zero repros), with MTP speculative
> decoding the leading suspect; v22.3.1 does not target it.

> **v22.3 (upstream base).** Covers Qwen 3.8 alongside 3.5/3.6 and adds prompt-directed
> reasoning-effort steering (`none`/`minimal`/`low`/`medium`/`high`/`xhigh`, plus the aliases above)
> and inline `<|think_…|>` control tags. The default effort is `medium` — a neutral baseline that
> injects **no** steering line when the caller asks for nothing. (Earlier v22 forced `xhigh` by
> default; froggeric fixed that upstream, so this Sharp build no longer suppresses anything — out of
> the box you get the tuned terseness behavior and nothing else, exactly as v1.) An *explicit* effort
> still renders; pass it via `chat_template_kwargs` (a bare top-level `reasoning_effort` field is
> dropped by OpenAI-style servers before the template sees it):
>
> ```json
> {"messages": [...], "chat_template_kwargs": {"reasoning_effort": "low"}}
> ```

[Dagger-Qwen3.6-27B](https://huggingface.co/peculiar-ragdoll/Dagger-Qwen3.6-27B-MLX) and
[Nail-Qwen3.6-35B-A3B](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-MLX) shipped with
the **v1** template baked into those builds — that is the exact template embedded in those GGUF and
MLX builds (`template_version = "qwen3.6-froggeric-v21.3"`, terseness, no reasoning-effort steering),
and it lives here in [`archive/v1-qwen3.6-froggeric-v21.3/`](archive/v1-qwen3.6-froggeric-v21.3). The
`chat_template.jinja` at the root of this repo is the newest **v22.5.0** described above; drop it in to
move a model onto it. The template is published separately because it is the portable part — the
thing worth reusing is not tied to either model.

Superseded versions are kept verbatim under [`archive/`](archive): the **v1** froggeric-v21.3 build
(Dagger/Nail), the **v22.1** build in [`archive/v22.1-sharp/`](archive/v22.1-sharp), the **v22.1.1**
build in [`archive/v22.1.1-sharp/`](archive/v22.1.1-sharp) — the last one before the v22.3 rebase, and
the version embedded in the published Dirk builds — the **v22.3.1** build in
[`archive/v22.3.1-sharp/`](archive/v22.3.1-sharp), the **v22.4.0** build in
[`archive/v22.4.0-sharp/`](archive/v22.4.0-sharp), and the immediately-prior **v22.4.1** in
[`archive/v22.4.1-sharp/`](archive/v22.4.1-sharp).

## What it changes

A terseness block, force-appended after your own system prompt. The **lead** now varies by thinking
mode (the v22.3.1 fix); the **core** never/always rules are identical on both paths, and the
thinking-on path is byte-identical to upstream v22.3 + the original terseness block.

```jinja
{%- if ns_state.thinking %}
    {%- set _terse_lead = 'Answer directly, after thinking. Lead with the answer, then only what it needs to be correct and usable.' %}
{%- else %}
    {%- set _terse_lead = 'Answer directly and concisely. Give the answer with only what it needs to be correct and usable.' %}
{%- endif %}
{%- set _terse_core %}
Never: open with preamble or pleasantries; restate the question; add filler transitions; hedge with niceties; or repeat a point you've already made.
Always: keep essential steps, caveats, uncertainties, and specifics — never drop correctness or a needed warning for brevity. Keep the final answer lean. Use the least structure that conveys it (plain prose when short; lists or code only when they earn their place). If genuinely uncertain, say so and explain why — never omit uncertainty for the sake of brevity.
If a user request is genuinely ambiguous, ask a sharp question, don't guess.
{%- endset %}
{%- set _terse = _terse_lead ~ '\n' ~ (_terse_core | trim) %}
{%- if not _sc %}
    {%- set _sc = _terse | trim %}
{%- else %}
    {%- set _sc = (_sc | trim) ~ '\n\n' ~ (_terse | trim) %}
{%- endif %}
```

Two things happen here: the `if/else` on `_sc` keeps **your own system prompt** — the terseness block
is appended after it, nothing you pass in is replaced; and the lead line matches the reasoning mode so
a fast-mode model isn't told to "answer after thinking" when it isn't thinking. Separately, the
tool-calling instructions gate every `<think>` reference — and the phrase *"IMMEDIATELY after
thinking"* — on thinking being on (the other half of the v22.3.1 fix). No effort-suppression is needed: v22.3 already defaults to `medium`, which injects no
reasoning-effort line unless you ask for one (earlier v22 forced `xhigh`; see the v22.3 note above).
An explicit `reasoning_effort` still renders.

## Impact

The terseness instruction targets prose padding: preamble, restating the question, filler
transitions. Where the deliverable is mostly code or a structured artifact there is less paddingto remove, so expect less from it — and the prompt deliberately protects those 
(*"lists or code only when they earn their place"*, *"never drop correctness for brevity"*).

In addition to reducing thinking tokens while retaining or increasing accuracy, like shown in the graphs up top, the template avoids amnesia and loops by turning on thinking retention: with this template, the model remembers what it thought last turn by default, instead of discarding it. This also increases time to first token on subsequent turns by guaranteeing a cache hit, instead of invalidating the cache by ripping out previous thinking blocks.

## Use

**MLX / transformers** — drop `chat_template.jinja` into the model directory.

```bash
hf download peculiar-ragdoll/Qwen-Sharp-Chat-Templates chat_template.jinja \
  --local-dir /path/to/your-model
```

> **Two places can hold a template, and old runtimes disagree about which wins.** A model
> directory can carry it as `chat_template.jinja` *and* as a `chat_template` key inside
> `tokenizer_config.json`. Anything on transformers ≥ 4.51 — which includes current oMLX and
> LM Studio — prefers the `.jinja` file, so the drop-in just works. Older runtimes read only
> the embedded key and ignore the file, and then the drop-in silently does nothing.
>
> If the directory has both and you are unsure of your runtime, patch both — that is what
> `chat_template_oneline.txt` is for: paste it as the `chat_template` value. Or run
> `scripts/check_applied.py` (below), which reports every source and flags a mismatch.

**oMLX** — drop `chat_template.jinja` into the model directory and rescan. Verified on oMLX
(transformers 5.12.1) by loading a model with the Sharp template as `chat_template.jinja` *and*
a deliberately different template embedded in `tokenizer_config.json`: the `.jinja` file won,
and the model reported the terseness rules with the caller's own system prompt still in force.

**GGUF** — rewrite the embedded template without requantizing:

```bash
pip install gguf
gguf-new-metadata \
  --chat-template-file chat_template.jinja \
  input.gguf output.gguf
```

**tokenizer_config.json** — use `chat_template_oneline.txt`, the minified single-line form. It
renders identically to the full template (verified by `scripts/verify_template.py`).

**llama.cpp at runtime, without touching the file** — pass it per-run instead:

```bash
llama-server -m model.gguf --chat-template-file chat_template.jinja --reasoning-format deepseek -ngl 99
llama-cli    -m model.gguf --chat-template-file chat_template.jinja -ngl 99
```

Same effect, and it fully replaces whatever is embedded in the GGUF — verified against a build
whose embedded template names a specific model: with the flag, the served template is
byte-identical to this file and the model name is gone. Check it yourself with
`curl localhost:8080/props | jq -r .chat_template`, or render a prompt through
`POST /apply-template`.

Three caveats. `--jinja` is enabled by default in current llama.cpp, so you usually do not need
it — on older builds you do, and it must come *before* `--chat-template-file`. And the flag is
per-invocation: forget it once and you silently get the embedded template back. Rewriting the
GGUF with `gguf-new-metadata` is the durable version; the flag is right for trying it out or for
running one template across several models.

Third, `--reasoning-format deepseek` (shown on the server line; it is an API-response setting, so
it does nothing for `llama-cli`). It puts the model's `<think>` block in the OpenAI
`reasoning_content` field instead of leaving it inline in `content` — which is what keeps a coding
agent from stalling on raw thinking tokens mid-stream. **On current llama.cpp it is already a
no-op:** `--reasoning-format` defaults to `auto`, which the source defines as *"same as deepseek"*
— verified at build 9890 (`74976e1ae`), where `COMMON_REASONING_FORMAT_AUTO` appears in no
behavioural branch at all and every extraction site gates on `!= none`. Pass it anyway if you may
be on an older build. The setting that genuinely breaks agents is `--reasoning-format none`, which
leaves the tags inline — don't use it except to inspect raw output.

## Did it actually apply?

Point `check_applied.py` at a model directory or a `.gguf`. It finds every template source,
renders each, and tells you whether they agree — exits non-zero if the prompt is missing or the
two sources disagree.

```bash
python3 scripts/check_applied.py /path/to/model-dir
python3 scripts/check_applied.py model.gguf
```

```
  [chat_template.jinja]  28162 bytes
     terseness prompt ......... yes
     keeps your system prompt . yes
     retains thinking* ........ yes

  [tokenizer_config.json]  8952 bytes
     terseness prompt ......... NO (found 0x)
     keeps your system prompt . yes
     retains thinking* ........ yes

  *** THE TWO SOURCES DISAGREE ***
  Recent transformers uses chat_template.jinja; oMLX and others read the
  copy embedded in tokenizer_config.json. Right now those RENDER DIFFERENTLY,
  so what you get depends on your runtime. Patch both to the same template.
```

That case — a fresh `.jinja` dropped in next to a stale embedded copy — is the most common way
this silently does nothing. It also warns if the template names a specific model, which happens
when the file was taken from a model repo rather than from here.

**It compares what the sources *render*, not how they are spelled.** That matters because the
documented way to patch both places is to paste `chat_template_oneline.txt` into
`tokenizer_config.json` — the minified form of the same template, byte-different by construction.
A text comparison flags that recommended state as broken; this one reports:

```
  Both sources render the SAME prompts — whichever your runtime prefers,
  you get the same behaviour (they differ only as full vs. minified text).
```

## Setting reasoning effort

By default there is no reasoning-effort instruction — you get the tuned terseness behavior and
nothing else (that is exactly what `medium` renders). To turn steering *on* for a request, set
`reasoning_effort` to `low` or `xhigh`. (`high`, `max`, `ultracode` and `extreme` are all accepted
as aliases for `xhigh`, so there is no level between `medium` and `xhigh`.) **How you pass it depends on the runtime, and one obvious-looking channel does
not work:**

| How you pass it | oMLX | llama.cpp | transformers | Works? |
|---|:--:|:--:|:--:|:--:|
| `chat_template_kwargs: {"reasoning_effort": "low"}` (in the request body) | ✅ | ✅ | — | **yes — use this** |
| `apply_chat_template(..., reasoning_effort="low")` (Python) | — | — | ✅ | **yes** |
| top-level `reasoning_effort` field (the OpenAI API param) | ❌ | ❌ | — | **no** |

```json
{"messages": [...], "chat_template_kwargs": {"reasoning_effort": "low"}}
```

The last row is the trap. The OpenAI-style **top-level** `reasoning_effort` field is *consumed by
the server* (oMLX and llama.cpp both use it internally to pick reasoning-parse behavior for formats
like harmony/gpt-oss) and is **never handed to the chat template** — so a custom Qwen template can't
see it, and it silently has no effect here. This isn't something the template can fix: a template
only reads the variables the runtime binds at render time. If you need the literal top-level field
to work against these servers, put a thin proxy in front that copies `reasoning_effort` into
`chat_template_kwargs` before forwarding. Otherwise, use the `chat_template_kwargs` channel above —
it works everywhere and needs no code.

Verified on both runtimes: with `chat_template_kwargs` the steering line renders (oMLX prompt grows
+38 tokens for `xhigh`, +26 for `low`; llama.cpp/minja `POST /apply-template` shows the same line);
with the bare top-level field it does not.

## Turning terseness off

The appended terseness prompt is what makes this template *Sharp*, so it is on by default. When you
want the model without it — A/B-ing the effect, or a downstream prompt that conflicts with it — turn
it off for that request:

```json
{"messages": [...], "chat_template_kwargs": {"terse": false}}
```

| Value | Result |
|---|---|
| omitted | terseness appended — the default, unchanged from `v22.3.1` |
| `true` | same as omitting it |
| `false` | terseness not appended; your own system prompt, if any, is passed through untouched |

Same channel as `reasoning_effort` above, and the same caveat applies: a **top-level** `terse` field
in the request body is not handed to the template and has no effect. In Python,
`apply_chat_template(..., terse=False)`.

Turning it off does not revert to a stock Qwen template — you keep every upstream fix and this
repo's fast-mode fixes. It removes exactly one thing: the appended prompt.

## Tests

froggeric ships a test suite upstream; this repo vendors it under `scripts/` and runs it against the
Sharp template rather than a stock one, so the fork is held to upstream's own invariants.

| Script | Covers | v22.5.0 |
|---|---|:--:|
| `test_v22.py` | 105 cases — effort steering and aliases, inline `<\|think_…\|>` tags, tool-call wire formats, system merging, error escalation, vision parts; v22.5 adds non-thinking tool-prompt coherence, `video_url`, and json-mode truncation | 105 / 105 |
| `test_v21.py` | 9 cases — the v21-era retention and rendering baseline | 8 / 9 † |
| `verify_template.py` | this fork's own invariants, including 12 checks that `terse` defaults on, honours an explicit opt-out on both the full and minified sources, and never swallows the caller's system prompt | all pass |
| `fuzz_template.py` | property fuzzer, 9 invariants: render, oneline parity, tag balance, content, XML fidelity, JSON validity, warning precision, prefix stability, empty-think prefill | clean over 2,000 conversations |

† `parallel tools delimiter` fails, and **it is upstream's failure, not this fork's**: pristine
froggeric `v22.4` and `v22.5` both fail the same case (checked by running the suite against each
unmodified). Recorded rather than patched around, so the count stays honest.

```bash
pip install jinja2
python3 scripts/test_v22.py
python3 scripts/test_v21.py
python3 scripts/fuzz_template.py --cases 2000
```

`test_v22.py` carries one local change, marked in the file: a shim that strips Sharp's appended
terseness block before each assertion. Sixteen upstream tests pin the exact *end* of the system turn,
which is precisely where Sharp appends — without the shim they fail on a difference this repo makes on
purpose, and sixteen permanently-red tests would hide a real regression the next time upstream bumps.
The shim removes only the block Sharp adds; every other upstream assertion still runs against our
rendering. It keys off the terseness marker, so the same file scores a *pristine* upstream template
100/100 as well, which is how it was checked for being a genuine no-op. Run against the pre-rebase
v22.1.1 template it still reports 70/100 — it hides Sharp's intended divergence, not real breakage.

`scripts/verify_template.py` is this repo's own check and complements those: it re-fetches upstream
live, asserts the thinking-on path is still byte-identical to upstream-plus-terseness, and fails if
froggeric has moved past the base recorded in `BASE` — which is what caught the v22.1 → v22.3 drift.

## What it doesn't do

- **It is not a fine-tune**, despite the `base_model_relation: finetune` tag — that is the closest
  vocabulary HuggingFace offers for "derived from," and it exists so this repo is linked from
  froggeric's. No weights are involved. It changes what the model is asked for, not what it knows.
- **It does not fix thinking retention by itself** — that comes from froggeric's upstream template,
  which this builds on. If you splice only the terseness block into a stock Qwen template, you get
  the brevity and not the retention.
- **It is not tuned per model.** Every model responds a little differently to a terseness
  instruction; measure yours. The numbers above are from a 27B; a 4B may need firmer wording.
- **The table's figures are Qwen3.6 (the plate above is the 3.8 result).** The template covers 3.5,
  3.6, and 3.8 alike — upstream unified them into one file — but every figure in the table was
  measured on a 3.6 model.

## Credits

Everything structural here is [froggeric](https://huggingface.co/froggeric)'s work — the retention
fix, the tool-calling handling, the error-escalation tiers, the whole template. This repo adds a
system prompt, the two fast-mode fixes, and nothing else.

`scripts/test_v22.py`, `scripts/test_v21.py` and `scripts/fuzz_template.py` are froggeric's test
suite, vendored so this fork is measured against upstream's invariants; `test_v22.py` carries the
documented shim described under [Tests](#tests). `scripts/minify_jinja.py` is froggeric's with one
patch: it now preserves newlines inside `{% set %}…{% endset %}` blocks, which upstream's template
doesn't contain and this one does. `scripts/check_applied.py` and `scripts/verify_template.py` are
this repo's.

Apache-2.0, matching upstream.

## Citation

```bibtex
@misc{Qwen-Sharp-Chat-Templates,
  title  = {Qwen Sharp Chat Templates},
  author = {Saga Ishtardottir},
  year   = {2026},
  url    = {https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates},
  note   = {froggeric's fixed Qwen3.5/3.6/3.8 chat template with a default-on, switchable terseness system prompt (v22.3.2)}
}
```
