"""EXL3 and Claude Code integration smoke test (issue #74).

The checks prove that health reports a ready model and window, count_tokens counts
Messages input, stream_shape has thinking SSE plus signature_delta,
effort_reaches_model reaches the template distinctly, tool_use_stream carries a
Read call and tool_use stop, too_long returns Claude Code's compaction 400, and
claude_p completes a CLI round-trip. Run this after every fork or Claude Code
update (issue #74).
"""
import argparse, json, os, subprocess, sys, time, urllib.request

HEALTH_TIMEOUT, CLAUDE_TIMEOUT = 10, 600
WORDS = ("amber", "birch", "canyon", "dawn", "ember", "fable", "garden", "harbor",
         "island", "jasmine", "kernel", "lantern", "meadow", "nectar", "ocean", "pebble",
         "quartz", "river", "silver", "thunder", "upland", "velvet", "willow", "yonder",
         "zephyr", "acorn", "breeze", "cedar", "drift", "echo", "frost", "grove",
         "horizon", "ink", "juniper", "kindle", "lilac", "maple", "north", "orchard",
         "prairie", "ripple", "sunset", "thistle", "umber", "valley", "wander", "xylophone",
         "year", "zinnia")

def clean(value, limit=240):
    text = str(value).replace("\r", " ").replace("\n", " ").encode("ascii", "replace").decode("ascii")
    return text if len(text) <= limit else text[:limit - 3] + "..."
def endpoint(base, path): return base.rstrip("/") + path
def call(url, payload, timeout, method="POST", accept="application/json"):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Accept": accept}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.getcode()), response.read().decode("utf-8", "replace")
    except Exception as exc:
        status = getattr(exc, "code", None)
        if status is None: raise
        try: body = exc.read()
        except Exception: body = ""
        return int(status), body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
def parse_sse(text):
    events, name, data = [], "", ""
    for line in text.splitlines() + [""]:
        if not line:
            if data and data != "[DONE]":
                value = json.loads(data)
                if not isinstance(value, dict): raise RuntimeError("SSE data is not an object")
                events.append((name or value.get("type") or "", value))
            name, data = "", ""
        elif line.startswith("event:"): name = line[6:].strip()
        elif line.startswith("data:"): data += ("\n" if data else "") + line[5:].lstrip()
    return events
def kind(item): return item[1].get("type") or item[0]
def stop_reason(data):
    delta, value = data.get("delta"), None
    if isinstance(delta, dict): value = delta.get("stop_reason")
    if value in (None, ""): value = data.get("stop_reason")
    return value if value not in (None, "") else None
def response_message(text):
    try: value = json.loads(text)
    except (TypeError, ValueError): return text
    if not isinstance(value, dict): return text
    error = value.get("error")
    if isinstance(error, dict) and error.get("message"): return str(error["message"])
    if isinstance(error, str): return error
    return str(value.get("message", text))

def check_health(base, state):
    status, text = call(endpoint(base, "/health"), None, HEALTH_TIMEOUT, "GET")
    if status != 200: return False, "status=%d message=%s" % (status, clean(response_message(text)))
    data = json.loads(text)
    state.update(model=data.get("model"), context_length=data.get("context_length"))
    detail = "model=%s context_length=%s" % (clean(data.get("model")), clean(data.get("context_length")))
    ready = data.get("ok") is True and data.get("busy") is False
    model_ok = isinstance(data.get("model"), str) and bool(data.get("model"))
    context = data.get("context_length")
    context_ok = isinstance(context, int) and not isinstance(context, bool) and context > 0
    return (True, detail) if ready and model_ok and context_ok else (False, detail + " ok=%s busy=%s" % (clean(data.get("ok")), clean(data.get("busy"))))

def check_count_tokens(base, state, timeout):
    payload = {"model": state.get("model") or "smoke", "messages": [{"role": "user", "content": "Count this small message."}]}
    status, text = call(endpoint(base, "/v1/messages/count_tokens"), payload, timeout)
    if status != 200: return False, "status=%d message=%s" % (status, clean(response_message(text)))
    data, tokens = json.loads(text), None
    if isinstance(data, dict): tokens = data.get("input_tokens")
    return (True, "input_tokens=%d" % tokens) if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0 else (False, "input_tokens=%s" % clean(tokens))

def check_stream_shape(base, state, timeout):
    payload = {"model": state.get("model") or "smoke", "stream": True, "max_tokens": 200,
               "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"},
               "messages": [{"role": "user", "content": "Write a paragraph of exactly 100 words."}]}
    status, text = call(endpoint(base, "/v1/messages"), payload, timeout, accept="text/event-stream")
    if status != 200: return False, "status=%d message=%s" % (status, clean(response_message(text)))
    events, issues = parse_sse(text), []
    kinds = [kind(item) for item in events]
    deltas = [item[1] for item in events if kind(item) == "message_delta"]
    finals = [data for data in deltas if stop_reason(data) is not None]
    if not kinds or kinds[0] != "message_start": issues.append("first=%s" % clean(kinds[0] if kinds else "none"))
    if not kinds or kinds[-1] != "message_stop": issues.append("last=%s" % clean(kinds[-1] if kinds else "none"))
    if len(finals) != 1: issues.append("message_delta_with_stop=%d" % len(finals))
    if not any(k == "content_block_start" for k in kinds): issues.append("content_block_start=0")
    for pos, item in enumerate(events):
        if kind(item) != "content_block_start" or (item[1].get("content_block") or {}).get("type") != "thinking": continue
        index, signature, closed = item[1].get("index"), False, False
        for later in events[pos + 1:]:
            if kind(later) == "content_block_delta" and later[1].get("index") == index:
                signature |= (later[1].get("delta") or {}).get("type") == "signature_delta"
            if kind(later) == "content_block_stop" and later[1].get("index") == index:
                closed = True; break
        if not closed: issues.append("thinking_without_stop")
        elif not signature: issues.append("thinking_without_signature_delta")
    detail = "events=%d ping=%d interim_message_delta=%d" % (len(events), kinds.count("ping"), len(deltas) - len(finals))
    return (False, detail + " " + ";".join(issues)) if issues else (True, detail)

def check_effort(base, state, timeout):
    counts = {}
    for effort in ("low", "medium", "xhigh"):
        payload = {"model": state.get("model") or "smoke", "stream": False, "max_tokens": 8,
                   "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": effort}
        status, text = call(endpoint(base, "/v1/chat/completions"), payload, timeout)
        if status != 200: return False, "effort=%s status=%d message=%s" % (effort, status, clean(response_message(text)))
        usage = (json.loads(text) or {}).get("usage", {})
        value = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if not isinstance(value, int) or isinstance(value, bool): return False, "effort=%s prompt_tokens=%s" % (effort, clean(value))
        counts[effort] = value
    detail = "low=%d medium=%d xhigh=%d" % (counts["low"], counts["medium"], counts["xhigh"])
    return (True, detail) if len(set(counts.values())) == 3 else (False, detail + " counts_not_pairwise_different")

def check_tool_use_stream(base, state, timeout):
    tool = {"name": "Read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}
    payload = {"model": state.get("model") or "smoke", "stream": True, "max_tokens": 64, "tools": [tool], "tool_choice": {"type": "tool", "name": "Read"}, "messages": [{"role": "user", "content": "Use the Read tool to read C:/AI/README.md. Do not answer without calling the tool."}]}
    status, text = call(endpoint(base, "/v1/messages"), payload, timeout, accept="text/event-stream")
    if status != 200: return False, "status=%d message=%s" % (status, clean(response_message(text)))
    events, starts, active, following, input_deltas = parse_sse(text), 0, None, False, 0
    for item in events:
        data = item[1]
        if kind(item) == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use" and block.get("name") == "Read": starts, active = starts + 1, data.get("index")
        elif kind(item) == "content_block_delta":
            delta = data.get("delta") or {}
            if delta.get("type") == "input_json_delta":
                input_deltas += 1; following |= active == data.get("index")
        elif kind(item) == "content_block_stop" and active == data.get("index"): active = None
    deltas = [item[1] for item in events if kind(item) == "message_delta"]
    final = stop_reason(deltas[-1]) if deltas else None
    detail = "tool_use_start=%d input_json_delta=%d final_stop=%s" % (starts, input_deltas, clean(final))
    return (True, detail) if starts and following and final == "tool_use" else (False, detail)

def varied_text(length):
    parts, size, index = [], 0, 0
    while size < length:
        part = (" " if parts else "") + WORDS[index % len(WORDS)]
        parts.append(part); size, index = size + len(part), index + 1
    return "".join(parts)[:length]
def too_long_numbers(message):
    marker, lower = "prompt is too long:", message.lower()
    start = lower.find(marker)
    if start < 0: return None
    fields = message[start + len(marker):].split()
    return (int(fields[0]), int(fields[3])) if len(fields) >= 4 and fields[1].lower() == "tokens" and fields[2] == ">" and fields[0].isdigit() and fields[3].isdigit() else None
def check_too_long(base, state, timeout):
    context = state.get("context_length")
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0: return False, "missing context_length from health"
    # size the prompt with the server's own count_tokens: 4 chars/token undershoots on
    # varied English (measured 2026-09-04: 1M chars = ~220K tokens, under a 262K window,
    # and the first smoke run prefilled it for ten minutes instead of getting the 400)
    text_body, chars = "", context * 4
    for _ in range(6):
        text_body = varied_text(chars)
        status, count_text = call(endpoint(base, "/v1/messages/count_tokens"),
                                  {"model": state.get("model") or "smoke", "messages": [{"role": "user", "content": text_body}]}, timeout)
        counted = json.loads(count_text).get("input_tokens") if status == 200 else None
        if isinstance(counted, int) and counted > context: break
        chars = int(chars * 1.4)
    else:
        return False, "could not build a prompt over the window (last count=%s)" % clean(counted)
    payload = {"model": state.get("model") or "smoke", "stream": False, "max_tokens": 64, "messages": [{"role": "user", "content": text_body}]}
    status, text = call(endpoint(base, "/v1/messages"), payload, timeout)
    if status == 200: return False, "status=200 (expected 400)"
    if status != 400: return False, "status=%d message=%s" % (status, clean(response_message(text)))
    numbers = too_long_numbers(response_message(text))
    return (True, "prompt_tokens=%d context_limit=%d" % numbers) if numbers else (False, "status=400 message=" + clean(response_message(text)))

def check_claude(base, state):
    model = state.get("model")
    if not isinstance(model, str) or not model: return False, "missing model from health"
    root = os.path.abspath(os.environ.get("TEMP") or os.environ.get("TMP") or os.getcwd())
    cwd = os.path.join(root, "exl3-smoke-%d-%d" % (os.getpid(), int(time.time() * 1000000)))
    os.mkdir(cwd)
    env = os.environ.copy(); env.update(ANTHROPIC_BASE_URL=base, ANTHROPIC_AUTH_TOKEN="sk-local", CLAUDE_CODE_MAX_OUTPUT_TOKENS="4096")
    # Windows: `claude` is a .cmd shim, which subprocess without a shell cannot find (WinError 2)
    import shutil
    exe = shutil.which("claude.cmd") or shutil.which("claude") or "claude"
    command = [exe, "-p", "Reply with exactly: SMOKE OK", "--model", model, "--strict-mcp-config", "--effort", "low"]
    try: result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=CLAUDE_TIMEOUT, check=False)
    except subprocess.TimeoutExpired: return False, "timeout=%ds" % CLAUDE_TIMEOUT
    except Exception as exc: return False, "error=" + clean(exc)
    finally:
        try: os.rmdir(cwd)
        except OSError: pass
    output = result.stdout or ""
    return (True, "status=0") if result.returncode == 0 and "SMOKE OK" in output else (False, "status=%d stdout=%s" % (result.returncode, clean(output)))

def main(argv=None):
    parser = argparse.ArgumentParser(description="Smoke-test the EXL3 Anthropic and Claude Code integration.")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="server base URL")
    parser.add_argument("--timeout", type=int, default=900, help="timeout in seconds for generation and API calls")
    parser.add_argument("--no-claude", action="store_true", help="skip the Claude Code check")
    args = parser.parse_args(argv)
    if args.timeout <= 0: parser.error("--timeout must be positive")
    base, state = args.base.rstrip("/"), {}
    checks = [("health", lambda: check_health(base, state)), ("count_tokens", lambda: check_count_tokens(base, state, args.timeout)),
              ("stream_shape", lambda: check_stream_shape(base, state, args.timeout)), ("effort_reaches_model", lambda: check_effort(base, state, args.timeout)),
              ("tool_use_stream", lambda: check_tool_use_stream(base, state, args.timeout)), ("too_long", lambda: check_too_long(base, state, args.timeout))]
    if not args.no_claude: checks.append(("claude_p", lambda: check_claude(base, state)))
    passed = 0
    for name, check in checks:
        try: ok, detail = check()
        except Exception as exc: ok, detail = False, "error=" + clean(exc)
        print("%s %s (%s)" % ("PASS" if ok else "FAIL", name, clean(detail))); passed += int(ok)
    print("SMOKE: %d/%d passed" % (passed, len(checks)))
    return 0 if passed == len(checks) else 1
if __name__ == "__main__": sys.exit(main())
