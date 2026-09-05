"""HTTP-seam tests for the EXL3 Anthropic routes."""
import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING = os.path.dirname(BENCH)
sys.path.insert(0, os.path.join(TUNING, "serving", "exl3"))
import anthropic_compat as ac  # noqa: E402
import anthropic_routes as routes  # noqa: E402


def _fallback_too_long_message(n, limit):
    return f"prompt is too long: {n} tokens > {limit} maximum"


TOO_LONG_MESSAGE = getattr(ac, "too_long_message", _fallback_too_long_message)


class FakeTokenIds:
    def __init__(self, n):
        self.shape = (1, n)


class FakeTokenizer:
    def __init__(self, n):
        self.n = n

    def hf_chat_template(self, messages, add_generation_prompt, enable_thinking, tools):
        return FakeTokenIds(self.n)


@pytest.mark.parametrize("stream", [False, True])
def test_too_long_messages_are_rejected_logged_and_captured(monkeypatch, tmp_path, stream):
    monkeypatch.setattr(ac, "too_long_message", TOO_LONG_MESSAGE, raising=False)
    log_path = tmp_path / "requests.jsonl"
    capture_dir = tmp_path / "captures"
    capture_dir.mkdir()
    monkeypatch.setenv("EXL3_REQUEST_LOG", str(log_path))
    monkeypatch.setenv("EXL3_CAPTURE_DIR", str(capture_dir))
    body = {
        "model": "test-model",
        "max_tokens": 100,
        "stream": stream,
        "messages": [{"role": "user", "content": "hello"}],
    }

    async def exercise():
        tokenizer = FakeTokenizer(990)
        server = SimpleNamespace(
            tool_choice_directive=lambda tool_choice, tools: (tools, None),
            normalize_messages=lambda messages: messages,
            stats={"context_length": 1000},
        )
        app = web.Application()
        app["tokenizer"] = tokenizer
        app["port"] = 1
        app["max_body_mb"] = 64
        routes.register(app, server)

        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post("/v1/messages", json=body)
            assert response.status == 400
            assert await response.json() == {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "prompt is too long: 990 tokens > 744 maximum",
                },
            }
        finally:
            await client.close()

    asyncio.run(exercise())

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == 400

    captures = list(capture_dir.glob("*.json"))
    assert len(captures) == 1
    assert json.loads(captures[0].read_text(encoding="utf-8")) == body


# --- output budget: the cap is the window, not a fixed number (2026-09-04) ---

def test_budget_clamps_max_tokens_to_what_the_window_leaves_and_rejects_only_when_nothing_is_left():
    """The developer set CLAUDE_CODE_MAX_OUTPUT_TOKENS to the whole window so a
    long think is never cut and restarted. A request then carries max_tokens
    equal to the window; the old rule (limit = window - max_tokens) would call
    every prompt too long. The budget is: clamp max_tokens to what the window
    leaves after the prompt; reject only when less than MIN_GEN is left."""
    assert routes.MIN_GEN == 256
    # plenty of room: requested cap is honoured
    assert routes.budget(n_prompt = 1000, requested = 4096, context_length = 262144) == 4096
    # window-sized request: clamped to what is left
    assert routes.budget(n_prompt = 1000, requested = 262144, context_length = 262144) == 262144 - 1000
    # exactly MIN_GEN left: allowed, at MIN_GEN
    assert routes.budget(n_prompt = 262144 - 256, requested = 262144, context_length = 262144) == 256
    # less than MIN_GEN left: too long
    assert routes.budget(n_prompt = 262144 - 255, requested = 64, context_length = 262144) is None
    assert routes.budget(n_prompt = 990, requested = 100, context_length = 1000) is None


@pytest.mark.parametrize("stream", [False, True])
def test_too_long_limit_in_the_message_is_window_minus_min_gen(monkeypatch, tmp_path, stream):
    monkeypatch.setenv("EXL3_REQUEST_LOG", str(tmp_path / "r.jsonl"))
    body = {"model": "m", "max_tokens": 262144, "stream": stream,
            "messages": [{"role": "user", "content": "hello"}]}

    async def exercise():
        server = SimpleNamespace(tool_choice_directive=lambda tc, t: (t, None),
                                 normalize_messages=lambda m: m, stats={"context_length": 1000})
        app = web.Application(); app["tokenizer"] = FakeTokenizer(900); app["port"] = 1; app["max_body_mb"] = 64
        routes.register(app, server)
        client = TestClient(TestServer(app)); await client.start_server()
        try:
            r = await client.post("/v1/messages", json=body)
            assert r.status == 400
            assert (await r.json())["error"]["message"] == "prompt is too long: 900 tokens > 744 maximum"
        finally:
            await client.close()
    asyncio.run(exercise())


def test_forced_tool_choice_directive_is_merged_into_the_leading_system_message_when_counting():
    """The first live smoke run: tool_choice {type: tool} made the pre-count
    append a SECOND system message at the end and the Qwen template raised
    `System message must be at the beginning` -> 500. generate_full merges the
    directive into the first system message; the count must do the same."""
    seen = {}

    class CapturingTokenizer(FakeTokenizer):
        def hf_chat_template(self, messages, add_generation_prompt, enable_thinking, tools):
            seen["messages"] = messages
            return FakeTokenIds(self.n)

    def directive(tool_choice, tools):
        return tools, "You must call the function `Read` now."

    server = SimpleNamespace(tool_choice_directive=directive, normalize_messages=lambda m: m,
                             stats={"context_length": 1000})
    app = web.Application(); app["tokenizer"] = CapturingTokenizer(10); app["port"] = 1; app["max_body_mb"] = 64
    routes.register(app, server)
    body = {"model": "m", "max_tokens": 10, "system": "You are terse.",
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "Read"},
            "messages": [{"role": "user", "content": "go"}]}

    async def exercise():
        client = TestClient(TestServer(app)); await client.start_server()
        try:
            r = await client.post("/v1/messages/count_tokens", json=body)
            assert r.status == 200 and (await r.json())["input_tokens"] == 10
        finally:
            await client.close()
    asyncio.run(exercise())
    roles = [m["role"] for m in seen["messages"]]
    assert roles.count("system") == 1 and roles[0] == "system"
    assert seen["messages"][0]["content"].endswith("You must call the function `Read` now.")
    assert seen["messages"][0]["content"].startswith("You are terse.")
