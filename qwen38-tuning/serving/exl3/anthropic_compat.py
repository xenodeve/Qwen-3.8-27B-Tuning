"""Anthropic Messages API <-> OpenAI chat completions, as pure functions.

Issue #73, 2026-09-04. Lives in qwen38-tuning/serving/exl3 (ours, not the
fork's tree); anthropic_routes.py mounts `/v1/messages` as a pipe over the
server's own `/v1/chat/completions`: the
request is translated here, sent to the OpenAI route over loopback, and the
JSON or SSE reply is translated back. Nothing in here touches the engine, so
the whole file is tested without a model
(qwen38-tuning/bench/tests/test_exl3_anthropic_messages.py).

What the client is: Claude Code. Its request shape, captured 2026-09-04:
`thinking: {type: adaptive}` + `output_config: {effort}`, a `system` list of
text blocks with cache_control, ~30 custom tools, tool_use/tool_result
history, `?beta=true` and an `anthropic-beta` header. Thinking blocks in the
history are dropped (the Qwen template carries no prior reasoning); images
are replaced by a note (the served model has no vision tower).
"""
import asyncio
import json
import re
import uuid

# Qwen3.8's chat template accepts xhigh / medium / low; serve_openai's
# resolve_effort() aliases the rest, but we hand it the exact value.
EFFORT = {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}
STOP_REASON = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


# ----------------------------------------------------------------- request ----

def normalize_billing_header(text):
    if not text.startswith("x-anthropic-billing-header:"):
        return text
    return re.sub(r"cch=[^;]{5};", "cch=fffff;", text, count = 1)


def too_long_message(n, limit):
    return f"prompt is too long: {n} tokens > {limit} maximum"


def _text_of(content, sep = "\n", normalize = False):
    """A string, or a list of text blocks, to one string."""
    if isinstance(content, str):
        return normalize_billing_header(content) if normalize else content
    parts = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            text = b.get("text") or ""
            parts.append(normalize_billing_header(text) if normalize else text)
        elif isinstance(b, dict) and b.get("type") == "image":
            parts.append("[image omitted]")
    return sep.join(parts)


def _user_turn(content, unknown_blocks = None):
    """One Anthropic user turn -> OpenAI tool messages (first) + a user message."""
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    tools, texts = [], []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "tool_result":
            body = _text_of(b.get("content"))
            if b.get("is_error"):
                body = "Error: " + body
            tools.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": body})
        elif t == "text":
            texts.append(b.get("text") or "")
        elif t == "image":
            texts.append("[image omitted]")
        elif t not in ("tool_use", "tool_result", "thinking", "redacted_thinking"):
            if unknown_blocks is not None:
                unknown_blocks.append(t)
    out = tools
    if texts:
        out.append({"role": "user", "content": "\n".join(texts)})
    return out


def _assistant_turn(content, unknown_blocks = None):
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    texts, calls, reasoning = [], [], []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            texts.append(b.get("text") or "")
        elif t == "tool_use":
            calls.append({"id": b.get("id"), "type": "function",
                          "function": {"name": b.get("name"),
                                       "arguments": json.dumps(b.get("input") or {})}})
        elif t == "thinking":
            reasoning.append(b.get("thinking") or "")
        elif t not in ("image", "tool_result", "redacted_thinking"):
            if unknown_blocks is not None:
                unknown_blocks.append(t)
    msg = {"role": "assistant", "content": "\n".join(texts)}
    if calls:
        msg["tool_calls"] = calls
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    return msg


def _tools(tools):
    out = []
    for t in tools or []:
        if isinstance(t, dict) and t.get("name") and isinstance(t.get("input_schema"), dict):
            fn = {"name": t["name"], "parameters": t["input_schema"]}
            if t.get("description"):
                fn["description"] = t["description"]
            out.append({"type": "function", "function": fn})
    return out


def _tool_choice(choice):
    if not isinstance(choice, dict):
        return None
    t = choice.get("type")
    if t == "any":
        return "required"
    if t == "tool":
        return {"type": "function", "function": {"name": choice.get("name")}}
    if t in ("auto", "none"):
        return t
    return None


def anthropic_to_openai(body):
    """Anthropic /v1/messages request -> the OpenAI chat request serve_openai reads."""
    messages = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _text_of(system, sep = "\n\n", normalize = True)})
    unknown_blocks = []
    for m in body.get("messages") or []:
        if m.get("role") == "assistant":
            messages.append(_assistant_turn(m.get("content"), unknown_blocks))
        elif m.get("role") == "system":
            messages.append({"role": "user", "content": "System note: " +
                            _text_of(m.get("content"), normalize = True)})
        else:
            messages.extend(_user_turn(m.get("content"), unknown_blocks))
    req = {"model": body.get("model"), "messages": messages,
           "max_tokens": body.get("max_tokens"), "stream": bool(body.get("stream", False))}
    for k in ("temperature", "top_p", "top_k"):
        if k in body:
            req[k] = body[k]
    if body.get("stop_sequences"):
        req["stop"] = list(body["stop_sequences"])
    tools = _tools(body.get("tools"))
    if tools:
        req["tools"] = tools
    choice = _tool_choice(body.get("tool_choice"))
    if choice is not None:
        req["tool_choice"] = choice
    effort = EFFORT.get(str(((body.get("output_config") or {}).get("effort") or "")).lower())
    if effort:
        req["reasoning_effort"] = effort
    if unknown_blocks:
        req["unknown_blocks"] = unknown_blocks
    return req


# ---------------------------------------------------------------- response ----

def _usage(usage, timings):
    prompt = int((usage or {}).get("prompt_tokens") or 0)
    cached = int((timings or {}).get("cached_tokens") or 0)
    return {"input_tokens": max(prompt - cached, 0),
            "output_tokens": int((usage or {}).get("completion_tokens") or 0),
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0}


def _message_id(openai_id = None):
    tail = (openai_id or "").split("-", 1)[-1] or uuid.uuid4().hex[:12]
    return f"msg_{tail}"


def _tool_use_block(call):
    fn = call.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args else {}
        except ValueError:
            args = {}
    return {"type": "tool_use", "id": call.get("id"), "name": fn.get("name"), "input": args or {}}


def openai_to_anthropic(resp):
    """OpenAI chat.completion -> Anthropic message (non-stream)."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = []
    if msg.get("reasoning_content"):
        content.append({"type": "thinking", "thinking": msg["reasoning_content"], "signature": ""})
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for c in msg.get("tool_calls") or []:
        content.append(_tool_use_block(c))
    finish = choice.get("finish_reason")
    stop = "tool_use" if msg.get("tool_calls") else STOP_REASON.get(finish, "end_turn")
    return {"id": _message_id(resp.get("id")), "type": "message", "role": "assistant",
            "model": resp.get("model"), "content": content,
            "stop_reason": stop, "stop_sequence": None,
            "usage": _usage(resp.get("usage"), resp.get("timings"))}


# ------------------------------------------------------------------ stream ----

def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


class StreamTranslator:
    """Feed OpenAI chat.completion.chunk objects, get (event, data) pairs.

    Blocks open lazily on the first delta of a kind and close when the kind
    changes, so the index sequence is thinking, text, tool_use... in the
    order the server emitted them. The final chunk (finish_reason set)
    closes the open block and emits message_delta + message_stop; a chunk
    shaped {"error": ...} becomes an `error` event.
    """

    USAGE_EVERY = 32          # tokens between interim usage events

    def __init__(self, model, message_id = None, interim_usage = False):
        # interim_usage: mid-turn message_delta usage events. OFF by default:
        # on the interactive UI they froze the live view after the first one
        # (thinking shown for ~1 s, then a jump at turn end, 2026-09-04). The
        # counter during thinking is Claude Code's own estimate from the text.
        self.interim_usage = interim_usage
        self.model = model
        self.id = message_id or _message_id()
        self.started = False
        self.index = -1
        self.open = None          # "thinking" | "text" | "tool_use" | None
        self.tool_calls = False
        self.usage_reported = 0   # output tokens last sent in an interim message_delta
        self.usage_seen = 0       # last running count seen on any chunk

    def _start(self):
        self.started = True
        return ("message_start", {"type": "message_start", "message": {
            "id": self.id, "type": "message", "role": "assistant", "model": self.model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}}})

    def _close(self):
        if self.open is None:
            return []
        out = []
        if self.open == "thinking":
            # llama-server does the same (server-task.cpp:918): the Anthropic
            # protocol closes a thinking block with a signature_delta; a local
            # model has no signature, so it is empty. Without it Claude Code
            # never rendered our thinking while llama-server's showed.
            out.append(self._delta({"type": "signature_delta", "signature": ""}))
        out.append(("content_block_stop", {"type": "content_block_stop", "index": self.index}))
        self.open = None
        return out

    def _open(self, kind, block):
        out = self._close()
        self.index += 1
        self.open = kind
        out.append(("content_block_start",
                    {"type": "content_block_start", "index": self.index, "content_block": block}))
        return out

    def _delta(self, delta):
        return ("content_block_delta", {"type": "content_block_delta", "index": self.index, "delta": delta})

    def start(self):
        """Emit message_start now (before the upstream call) so keep-alive pings
        can follow it; feed() then never emits it again."""
        return [] if self.started else [self._start()]

    def feed(self, chunk):
        out = []
        if not self.started:
            out.append(self._start())
        if "error" in chunk and not chunk.get("choices"):
            out += self._close()
            err = chunk["error"] if isinstance(chunk["error"], dict) else {"message": str(chunk["error"])}
            out.append(("error", {"type": "error", "error": dict({"type": "api_error"}, **err)}))
            return out
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        running = (chunk.get("usage") or {}).get("completion_tokens")
        if delta.get("reasoning_content"):
            if self.open != "thinking":
                # no signature key on open, as llama-server (server-task.cpp:827)
                out += self._open("thinking", {"type": "thinking", "thinking": ""})
            # text only, as llama-server. No `estimated_tokens`: Claude Code
            # 2.1.258 adds that field up as an increment per delta, so a running
            # total summed to N^2/2 and the counter read 731k (2026-09-04).
            out.append(self._delta({"type": "thinking_delta", "thinking": delta["reasoning_content"]}))
        if isinstance(running, int):
            self.usage_seen = running
        if delta.get("content"):
            if self.open != "text":
                out += self._open("text", {"type": "text", "text": ""})
            out.append(self._delta({"type": "text_delta", "text": delta["content"]}))
        for call in delta.get("tool_calls") or []:
            self.tool_calls = True
            block = _tool_use_block(call)
            args = (call.get("function") or {}).get("arguments") or ""
            out += self._open("tool_use", dict(block, input = {}))
            if args:
                out.append(self._delta({"type": "input_json_delta",
                                        "partial_json": args if isinstance(args, str) else json.dumps(args)}))
        finish = choice.get("finish_reason")
        # Interim usage: Claude Code's live token counter reads
        # message_delta.usage.output_tokens, and the real API streams it mid-turn
        # (the thinking-token-count beta). The OpenAI route puts a running
        # usage.completion_tokens on every chunk; report it every USAGE_EVERY
        # tokens with no stop_reason. input_tokens is unknown until the end and
        # is omitted; the SDK only overwrites it when present.
        if self.interim_usage and finish is None and isinstance(running, int) and running - self.usage_reported >= self.USAGE_EVERY:
            self.usage_reported = running
            out.append(("message_delta", {"type": "message_delta", "delta": {},
                                          "usage": {"output_tokens": running}}))
        if finish is not None:
            out += self._close()
            stop = "tool_use" if (self.tool_calls or finish == "tool_calls") else STOP_REASON.get(finish, "end_turn")
            out.append(("message_delta", {"type": "message_delta",
                                          "delta": {"stop_reason": stop, "stop_sequence": None},
                                          "usage": _usage(chunk.get("usage"), chunk.get("timings"))}))
            out.append(("message_stop", {"type": "message_stop"}))
        return out


PING = ("ping", {"type": "ping"})


async def pump(lines, translator, ping_s = 5.0):
    """Async-iterate OpenAI SSE lines from `lines`, yield Anthropic (event, data)
    pairs from `translator`, and yield a `ping` whenever `ping_s` seconds pass
    with nothing from upstream — the prefill silence Claude Code otherwise
    reports as an API error. Stops at `data: [DONE]` or end of stream."""
    it = lines.__aiter__()
    pending = None
    while True:
        if pending is None:
            pending = asyncio.ensure_future(it.__anext__())
        done, _ = await asyncio.wait({pending}, timeout = ping_s)
        if not done:
            yield PING
            continue
        try:
            line = pending.result()
        except StopAsyncIteration:
            return
        pending = None
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            return
        for ev in translator.feed(json.loads(payload)):
            yield ev
