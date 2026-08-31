# templates/ - chat templates passed to `llama-server`

One file. It is the model's own Jinja chat template with a single line changed.

## `qwen38-late-system.jinja`

Qwen3.8's stock template counts the run of `system` messages at the front of the
conversation and then **raises** if a `system` or `developer` role appears after it:

```jinja
{%- if message.role == "system" or message.role == "developer" %}
    {{- raise_exception('System message must be at the beginning.') }}
```

Anthropic's API allows a trailing system message and **Claude Code sends one**:
the output of its `SessionStart` hooks arrives as a `role: "system"` message of
25-33 KB appended after the user turn. Every request then fails at sampler init
with `Failed to initialize samplers` and the harness retries forever.

The one changed line renders it as an ordinary system turn instead, in exactly
the form the `user` branch on the next line already uses:

```jinja
    {{- '<|im_start|>system\n' + content + '<|im_end|>' + '\n' }}
```

## Evidence, 2026-08-21

The captured failing request was replayed against both templates. Stock: 500,
`System message must be at the beginning.` Patched: 200, `input_tokens` 54,685.
Deleting the trailing system message, or changing its role to `user`, also returns
200 on the stock template -- so message position is the whole cause, and nothing
else in that request mattered.

End to end after the change, with the server log at zero exceptions:

| harness | config | result |
|---|---|---|
| `claude-xeno.bat` | the user's normal Claude Code settings | replies; `Bash` tool call runs and returns |
| `opencode run` | the user's normal global config, 11 MCP servers | replies; `Write` tool call creates the file |

Before the change the same two harnesses produced 50 consecutive failures.

## It must not be bundled with anything, 2026-08-31

The flag spent a week inside the `else` branch of `-Beta` in
`worker-q4-dual.ps1`, because `-Beta` borrows Unsloth Studio's thinking mechanism
and Studio passes no template file. Two unrelated concerns in one `if/else`.
`-Clone` rebuilds its command line from scratch and never had it either.

**Five hub icons -- 7, 8, 9, A and B -- then answered HTTP 500 to every Claude
Code request**, fifteen in a row in `logs/serve-20260831-023636.log`.

Studio omits the file safely because Studio's client never sends a late system
message. **Copying that omission reproduces a client incompatibility, not a
baseline** -- the same shape as CORRECTIONS 36, on the same switch. The omission
now lives on its own switch, `-StockTemplate`, and the profile refuses to launch
when the final `argv` lost the flag any other way. See CORRECTIONS 43 and
issue #58.

## `qwen38-stock.jinja` — the artifact's own template, unmodified

Added 2026-08-31, issue #65. It is the second file in this folder and it exists
so the one-line claim above can be **checked offline, every test run**.

Issue #58 asked for that difference to be *"verified in a test, not by hand"*,
and the test written for it read `/props` from a live server. **`/props` reports
the template the server is USING** — so a normally-booted profile hands back our
own patched file, and the check can only run against a boot with
`-StockTemplate`, which had never happened. **The criterion was written and never
exercised.** It passed once, against a `-Beta` boot that happened to have no
override, and then reported *zero* differing lines as if the patch had vanished.

With the stock text checked in, `test_ours_differs_from_the_vendored_stock_by_exactly_one_line`
runs with no server at all. The `/props` comparison stays as the stronger form
when a `-StockTemplate` boot is available — it is the only thing that can catch
**this file** drifting from the artifact.

**Provenance:** extracted from `GET /props` on 2026-08-31 while
`serve-20260831-023636.log` was serving `Qwen3.8-27B-NVFP4-MTP-VERY-LOW` with no
`--chat-template-file` on its command line, so `/props` was reporting the GGUF's
own text. 183 lines; it differs from `qwen38-late-system.jinja` at line 110 and
nowhere else.

## Regenerating it

The template belongs to the model, so it must be re-derived if the artifact changes.
`GET /props` returns the stock text in `.chat_template`. Take it, change the one
`raise_exception` line, and keep everything else byte-identical -- a template that
differs anywhere else is a different experiment.
