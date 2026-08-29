# llama-tap — read what a client actually sends llama-server

**Status: built and unit-tested 2026-08-29 (issue #53). Never yet run against
Unsloth Studio.** Sixteen tests cover it, including the transparency rule and
the Windows argument round trip. No capture from Studio exists yet, so nothing
in `docs/` cites it.

---

## Why it exists

Three claims this project published about Unsloth Studio were retracted the same
week, and all three were read off a command line or out of source:

| | claim | what was actually true |
|---|---|---|
| [CORRECTIONS 36](../../../docs/reports/CORRECTIONS.md) | `-Beta` needs no `--reasoning-effort` because Studio's argv has none | Studio sends it **per request**. Our server ran at `xhigh` for an afternoon |
| [CORRECTIONS 37](../../../docs/reports/CORRECTIONS.md) | we serve llama.cpp's default sampler | the artifact's own `general.sampling.*` wins |
| [CORRECTIONS 38](../../../docs/reports/CORRECTIONS.md) | `n-match 24` is independent agreement | it sits beside two more defaults |

One shape: **a command line is not a configuration.** Reading source beats
reading argv, and both lose to reading the wire.

## The rule it must not break

**It forwards bytes and observes them. It never re-serialises a request.**

A tap that parsed and rebuilt a request would hand llama-server whatever our
parser thinks JSON looks like, and every number taken through it would be a
measurement of the parser. That is the instrument this repo's north star warns
about — one that returns a believable result instead of failing. The first test
in `bench/tests/test_llama_tap.py` is not *did we capture it*, it is *is what
arrived upstream byte-identical to what was sent*.

Redaction happens on the **copy**, never on the socket: the capture must not
hold a bearer token, and upstream must still receive one.

## It is not `bench/tap.py`, and that one came first

**This was written without checking the register** — `CLAUDE.md` puts
`docs/results/README.md` at session-start item 4 for exactly this reason, and
the collision only surfaced when `import tap` resolved to the wrong module and
sixteen green tests went red inside the suite.

Both are wanted. They are different instruments:

| | `bench/tap.py` | `tools/llama-tap/relay.py` |
|---|---|---|
| level | HTTP: parses a request, re-issues it with `urllib` | TCP: forwards bytes, rebuilds nothing |
| upstream sees | this proxy's idea of the request | the client's bytes, exactly |
| output | one JSONL row per request, with `--mark` labels so a harness can attribute rows to tasks | raw capture + a separate reader |
| credentials | recorded | redacted at write time |
| reaches a server it did not launch | no | yes, via the shim |

**Use `bench/tap.py` to label our own runs** — we wrote that client, so a
rebuilt request is still our request. **Use this to audit a client we do not
control**, where a rebuilt request would turn the audit into a measurement of
our own parser.

## Use it without Studio

```powershell
python relay.py --listen 8081 --upstream 8080 --out ..\..\logs\llama-tap
# point a client at 8081 instead of 8080, then:
python read_capture.py ..\..\logs\llama-tap --summary
```

`--summary` answers the question the tool exists for: **which settings arrive
per request rather than on the command line.**

## Use it on Unsloth Studio

Studio resolves its binary from `LLAMA_SERVER_PATH` first and checks only that
the path is a file — no extension, no executable bit on Windows
(`studio/backend/core/inference/llama_cpp.py:6544`). So installation is one
variable and **their install is never touched**:

```powershell
setx LLAMA_SERVER_PATH "C:\AI\qwen38-tuning\tools\llama-tap\llama-server.cmd"
setx LLAMA_TAP_REAL    "C:\Users\xenod\.unsloth\llama.cpp\build\bin\Release\llama-server.exe"
# restart Unsloth Studio
```

Uninstall:

```powershell
setx LLAMA_SERVER_PATH ""
```

### What happens then

```
Studio  ->  llama-server.cmd  ->  shim.py
                                    |- starts the REAL binary on a free port
                                    |- starts relay.py on the port Studio asked for
                                    `- waits, and returns the child's exit code
```

A launch has `--port`; a capability probe (`--help`) does not, and probes go
straight through — Studio decides which flags to use from that output
(`supports_slot_save = _is_real("--slot-save-path")`), so it must be the
binary's own answer, not a proxy's.

## The argument that forced the design

Studio passes this as **one** argument:

```
--chat-template-kwargs {"enable_thinking": true, "preserve_thinking": true}
```

Rebuilding a command line from `sys.argv` loses it — the C runtime has already
re-split it by the time Python sees it, and llama.cpp would be launched with a
configuration the tap invented. So the shim reads `GetCommandLineW()`, rewrites
only the `--port` value textually, and hands the raw string on.

`llama-server.cmd` forwards its tail as `%*`, which is a **literal** text
substitution — quotes and braces intact. That is the opposite of `sys.argv`, and
one of this tool's tests was originally written asserting the reverse; building
it settled the question. What arrives is Windows' own encoding
(`"{\"enable_thinking\": true, ...}"`), and the test asserts the round trip with
`CommandLineToArgvW` rather than by comparing text.

## Privacy

A capture holds prompt text. It stays local:

- `Authorization`, `X-Api-Key`, `Api-Key`, `Cookie` and `Proxy-Authorization`
  values are replaced **at write time**, line by line, so a value straddling a
  chunk boundary cannot leak;
- the redactor is conservative — a body line beginning `cookie:` is redacted
  too, which is cheaper than a parser that must track chunked framing to know
  when to stop looking;
- `logs/llama-tap/` is gitignored;
- nothing is uploaded, ever. Uninstall is one environment variable.

**The captures are still someone's private conversations.** Treat a capture
directory the way you would treat the chat window it came from.

## Files

| | |
|---|---|
|  `relay.py` | the relay. Standalone, no dependencies beyond the standard library |
| `read_capture.py` | capture → JSONL rows, or `--summary` |
| `shim.py` | the Studio-facing wrapper; also importable for its pure parts |
| `llama-server.cmd` | what `LLAMA_SERVER_PATH` points at |

Tests: `qwen38-tuning/bench/tests/test_llama_tap.py`.

## What it cannot tell you

- **`first_byte_s` is not time-to-first-token.** The tap sees when bytes crossed
  it, not when llama-server started working. It is a ceiling on prefill plus
  first token and is named so it cannot be quoted as anything better.
- **A chunked request body ends the walk.** llama-server's clients do not send
  one; guessing at a frame never seen would be the reader inventing data.
- **It does not explain a rate.** It records what was asked for. Every rule in
  `docs/agents/traps.md` about pairing arms and rotating order still applies.
