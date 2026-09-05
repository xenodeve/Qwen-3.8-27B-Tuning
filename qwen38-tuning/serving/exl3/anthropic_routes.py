"""Anthropic Messages API routes for the EXL3 server (issue #73, 2026-09-04).

`/v1/messages` is a pipe: the request is translated (anthropic_compat), sent
to this process's own `/v1/chat/completions` over loopback, and the JSON or
SSE reply translated back. `/v1/messages/count_tokens` tokenises through the
same translation. No engine code.

Keep-alive (the llama.cpp lesson, docs/reports/05-OPERATING-GUIDE.md §3): a
30K-token prefill is 45 s of silence on the wire, and Claude Code reports a
silent connection as an API error. llama-server fixed it with
`--sse-ping-interval 5`; here `message_start` goes out before the upstream
call and an Anthropic `ping` event every PING_S seconds until the first
delta arrives. It does not make the wait shorter; it keeps the line visibly
alive.

`register(app, server)` mounts the routes; `server` is the module that owns
`tool_choice_directive` and `normalize_messages` (the fork's), passed in so
this file imports nothing from it.
"""
import asyncio
import datetime
import json
import os
import time
import uuid

import aiohttp
from aiohttp import web

import anthropic_compat as ac

PING_S = 5.0
MIN_GEN = 256          # the fewest output tokens a request must have room for


def budget(n_prompt, requested, context_length):
    """Output-token budget for one request, or None when the prompt leaves
    fewer than MIN_GEN tokens. The developer's profile sets Claude Code's
    max_tokens to the whole window (so a long think is never cut and
    restarted), so the cap is clamped to what the window leaves after the
    prompt rather than subtracted from it."""
    left = int(context_length) - int(n_prompt)
    if left < MIN_GEN:
        return None
    return max(min(int(requested), left), MIN_GEN)
DEFAULT_REQUEST_LOG = r"C:\AI\qwen38-tuning\logs\exl3-requests.jsonl"
TRACE_PATH = os.environ.get("EXL3_TRACE_SSE")   # set to a file: every outgoing SSE event, timed


def _trace(rid, ev, obj, t0):
    """One JSON line per outgoing SSE event -- what Claude Code actually got and
    when. For diagnosing the live view (tools/exl3-trace.py renders it)."""
    if not TRACE_PATH:
        return
    try:
        d = obj.get("delta") if isinstance(obj, dict) else None
        rec = {"rid": rid, "t": round(time.perf_counter() - t0, 3), "ev": ev,
               "idx": obj.get("index") if isinstance(obj, dict) else None,
               "dt": (d or {}).get("type") if isinstance(d, dict) else None,
               "n": len((d or {}).get("thinking") or (d or {}).get("text") or (d or {}).get("partial_json") or "") if isinstance(d, dict) else None,
               "est": (d or {}).get("estimated_tokens") if isinstance(d, dict) else None,
               "out": ((obj.get("usage") or {}).get("output_tokens") if isinstance(obj, dict) else None),
               "stop": (d or {}).get("stop_reason") if isinstance(d, dict) else None}
        with open(TRACE_PATH, "a", encoding = "utf-8") as fh:
            fh.write(json.dumps(rec, separators = (",", ":")) + "\n")
    except Exception as e:
        print(f" ## anthropic: trace write failed ({TRACE_PATH}): {e!r}", flush = True)


def _error(message, status = 400, kind = "invalid_request_error"):
    return web.json_response({"type": "error", "error": {"type": kind, "message": message}},
                             status = status)


async def _read_body(request):
    try:
        body = await request.json()
    except web.HTTPRequestEntityTooLarge:
        return None, _error(f"request body exceeds {request.app['max_body_mb']} MiB limit",
                            413, "request_too_large")
    except Exception:
        return None, _error("invalid JSON")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return None, _error("`messages` (list) is required")
    if not body.get("max_tokens"):
        body["max_tokens"] = 1024
    return body, None


def _error_text(response):
    try:
        data = json.loads(response.body)
        return (data.get("error") or {}).get("message")
    except Exception:
        return None


def _capture_request(body):
    try:
        directory = os.environ.get("EXL3_CAPTURE_DIR")
        if not directory or not os.path.isdir(directory):
            return
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = os.path.join(directory, f"{stamp}-{uuid.uuid4().hex[:8]}.json")
        with open(path, "w", encoding = "utf-8") as fh:
            json.dump(body, fh, indent = 2, ensure_ascii = False)
            fh.write("\n")
    except Exception as e:
        print(f" ## anthropic: capture write failed ({directory}): {e!r}", flush = True)


def _as_int(value):
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _record_openai_usage(metrics, data):
    usage = data.get("usage") or {}
    output = _as_int(usage.get("completion_tokens"))
    cached = _as_int((data.get("timings") or {}).get("cached_tokens"))
    if output is not None:
        metrics["output_tokens"] = output
    if cached is not None:
        metrics["cached_tokens"] = cached


def _record_stream_usage(metrics, event, data):
    if event != "message_delta":
        return
    usage = data.get("usage") or {}
    output = _as_int(usage.get("output_tokens"))
    cached = _as_int(usage.get("cache_read_input_tokens", usage.get("cached_tokens")))
    if output is not None:
        metrics["output_tokens"] = output
    if cached is not None:
        metrics["cached_tokens"] = cached


def _append_request_log(metrics, started):
    try:
        record = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": int(metrics.get("status", 500)),
            "stream": bool(metrics.get("stream", False)),
            "prompt_tokens": metrics.get("prompt_tokens"),
            "output_tokens": metrics.get("output_tokens"),
            "cached_tokens": metrics.get("cached_tokens"),
            "wall_ms": float((time.perf_counter() - started) * 1000.0),
            "error": (None if metrics.get("error") is None else str(metrics.get("error"))),
            "model": str(metrics.get("model") or ""),
        }
        path = os.environ.get("EXL3_REQUEST_LOG") or DEFAULT_REQUEST_LOG
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok = True)
        with open(path, "a", encoding = "utf-8") as fh:
            json.dump(record, fh, separators = (",", ":"))
            fh.write("\n")
    except Exception as e:
        print(f" ## anthropic: request log write failed ({path}): {e!r}", flush = True)


def _prompt_tokens(request, req, server):
    tokenizer = request.app["tokenizer"]
    tools, directive = server.tool_choice_directive(req.get("tool_choice"), req.get("tools"))
    msgs = list(server.normalize_messages(req["messages"]))
    if directive:
        # as generate_full does: the Qwen template allows ONE leading system
        # message, so the directive is merged into it (or becomes it)
        if msgs and msgs[0].get("role") == "system":
            first = dict(msgs[0])
            first["content"] = (first.get("content") or "").rstrip() + "\n\n" + directive
            msgs[0] = first
        else:
            msgs = [{"role": "system", "content": directive}] + msgs
    ids = tokenizer.hf_chat_template(msgs, add_generation_prompt = True,
                                     enable_thinking = True, tools = tools)
    return int(ids.shape[-1])


async def _stream(request, req, url, session, metrics):
    resp = web.StreamResponse(headers = {
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "Connection": "keep-alive"})
    await resp.prepare(request)
    tr = ac.StreamTranslator(model = req["model"])
    rid, t0 = uuid.uuid4().hex[:8], time.perf_counter()
    try:
        for ev, obj in tr.start():
            _trace(rid, ev, obj, t0)
            await resp.write(ac.sse(ev, obj))
        async with session.post(url, json = req) as r:
            if r.status != 200:
                data = await r.json()
                error = data.get("error") or {"message": "upstream error"}
                metrics["error"] = error.get("message") if isinstance(error, dict) else str(error)
                events = tr.feed({"error": error})
                for ev, obj in events:
                    _trace(rid, ev, obj, t0)
                    await resp.write(ac.sse(ev, obj))
            else:
                async def lines():
                    async for line in r.content:
                        yield line
                async for ev, obj in ac.pump(lines(), tr, ping_s = PING_S):
                    _record_stream_usage(metrics, ev, obj)
                    _trace(rid, ev, obj, t0)
                    await resp.write(ac.sse(ev, obj))
        await resp.write_eof()
    except ConnectionResetError:
        pass
    if metrics["output_tokens"] is None:
        output = _as_int(getattr(tr, "usage_seen", None))
        if output:
            metrics["output_tokens"] = output
    return resp


def make_messages(server):
    async def messages(request):
        started = time.perf_counter()
        response = None
        metrics = {"status": 500, "stream": False, "prompt_tokens": None,
                   "output_tokens": None, "cached_tokens": None,
                   "error": None, "model": ""}
        try:
            body, err = await _read_body(request)
            if err:
                response = err
                metrics["error"] = _error_text(err)
                return response
            metrics["model"] = str(body.get("model") or "")
            metrics["stream"] = bool(body.get("stream", False))
            _capture_request(body)
            req = ac.anthropic_to_openai(body)
            metrics["model"] = str(req.get("model") or metrics["model"])
            metrics["stream"] = bool(req["stream"])
            unknown = req.get("unknown_blocks")
            if unknown:
                print(f"[anthropic] unknown content blocks dropped: {','.join(map(str, unknown))}",
                      flush = True)
            metrics["prompt_tokens"] = _prompt_tokens(request, req, server)
            context_length = server.stats["context_length"]
            if context_length is not None:
                allowed = budget(metrics["prompt_tokens"], req["max_tokens"], context_length)
                if allowed is None:
                    message = ac.too_long_message(metrics["prompt_tokens"], int(context_length) - MIN_GEN)
                    metrics["error"] = message
                    response = _error(message)
                    return response
                req["max_tokens"] = allowed
            url = f"http://127.0.0.1:{request.app['port']}/v1/chat/completions"
            async with aiohttp.ClientSession(timeout = aiohttp.ClientTimeout(total = None)) as session:
                if req["stream"]:
                    response = await _stream(request, req, url, session, metrics)
                    return response
                async with session.post(url, json = req) as r:
                    status = r.status
                    data = await r.json()
                _record_openai_usage(metrics, data)
            if status != 200 or "error" in data:
                msg = (data.get("error") or {}).get("message") or "upstream error"
                metrics["error"] = msg
                response = _error(msg, status if status != 200 else 500,
                                  "api_error" if status >= 500 else "invalid_request_error")
                return response
            response = web.json_response(ac.openai_to_anthropic(data))
            return response
        except Exception as exc:
            metrics["error"] = str(exc) or exc.__class__.__name__
            raise
        finally:
            if response is not None:
                metrics["status"] = response.status
            _append_request_log(metrics, started)
    return messages


def make_count_tokens(server):
    async def count_tokens(request):
        body, err = await _read_body(request)
        if err:
            return err
        req = ac.anthropic_to_openai(body)
        return web.json_response({"input_tokens": _prompt_tokens(request, req, server)})
    return count_tokens


def register(app, server):
    app.router.add_post("/v1/messages", make_messages(server))
    app.router.add_post("/v1/messages/count_tokens", make_count_tokens(server))
