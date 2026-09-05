r"""The EXL3 fork server speaks the Anthropic Messages API as a pipe over its
own OpenAI route (issue #73, 2026-09-04).

WHY. `claude-xeno-exl3` needed LiteLLM between Claude Code and
`tools/serve_openai.py`; the proxy is a second process and a translation
layer we do not control -- it dropped `output_config` whenever no `thinking`
sat beside it, and its adapter version decides what reaches the model. The
translator under test is pure Python (`qwen38-tuning/serving/exl3/anthropic_compat.py`,
ours), so it is tested here without the engine, against the request shape
captured from Claude Code on 2026-09-04: `thinking{type:adaptive}` +
`output_config{effort}`, a system list with cache_control, 31 tools,
`?beta=true`.

Every assertion is on something only the translator can produce: a key the
OpenAI request must carry, an event name in the Anthropic stream, a number
derived from the server's `timings`.
"""
import json
import os
import sys

import pytest

TUNING = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TUNING, "serving", "exl3"))
import anthropic_compat as ac  # noqa: E402


def claude_code_request(effort = "medium", stream = True):
    """The shape Claude Code sent on 2026-09-04, reduced to two tools."""
    return {
        "model": "qwen3.8-27b-exl3-3.5bpw-wm",
        "max_tokens": 16384,
        "stream": stream,
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": effort},
        "system": [
            {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Second system block."},
        ],
        "tools": [
            {"name": "Read", "description": "Read a file",
             "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}},
                              "required": ["file_path"]}},
            {"name": "Bash", "description": "Run a command",
             "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
        ],
        "messages": [
            {"role": "user", "content": "read the readme"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I should read it.", "signature": "sig"},
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "toolu_01", "name": "Read",
                 "input": {"file_path": "README.md"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01",
                 "content": [{"type": "text", "text": "# Title\nbody"}]},
                {"type": "text", "text": "now summarise it"},
            ]},
        ],
        "metadata": {"user_id": "x"},
    }


# ---------------------------------------------------------------- request ----

def test_system_blocks_become_one_system_message_and_tools_become_functions():
    req = ac.anthropic_to_openai(claude_code_request())
    assert req["messages"][0] == {"role": "system",
                                  "content": "You are Claude Code.\n\nSecond system block."}
    assert [t["type"] for t in req["tools"]] == ["function", "function"]
    assert req["tools"][0]["function"]["name"] == "Read"
    assert req["tools"][0]["function"]["parameters"]["required"] == ["file_path"]
    assert req["max_tokens"] == 16384 and req["stream"] is True
    assert req["model"] == "qwen3.8-27b-exl3-3.5bpw-wm"
    assert "metadata" not in req and "thinking" not in req and "output_config" not in req


def test_history_maps_tool_use_to_tool_calls_and_tool_result_to_tool_messages():
    req = ac.anthropic_to_openai(claude_code_request())
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user"]
    asst = req["messages"][2]
    assert asst["content"] == "Reading."
    assert "thinking" not in json.dumps(asst)            # thinking blocks are not history
    call = asst["tool_calls"][0]
    assert call["id"] == "toolu_01" and call["type"] == "function"
    assert call["function"]["name"] == "Read"
    assert json.loads(call["function"]["arguments"]) == {"file_path": "README.md"}
    tool = req["messages"][3]
    assert tool["tool_call_id"] == "toolu_01" and tool["content"] == "# Title\nbody"
    assert req["messages"][4] == {"role": "user", "content": "now summarise it"}


@pytest.mark.parametrize("effort,expect", [
    ("low", "low"), ("medium", "medium"), ("high", "xhigh"), ("xhigh", "xhigh"), ("max", "xhigh")])
def test_effort_travels_as_reasoning_effort_the_template_accepts(effort, expect):
    assert ac.anthropic_to_openai(claude_code_request(effort))["reasoning_effort"] == expect


def test_no_effort_means_no_reasoning_effort_key():
    body = claude_code_request()
    del body["output_config"]
    assert "reasoning_effort" not in ac.anthropic_to_openai(body)


@pytest.mark.parametrize("choice,expect", [
    ({"type": "auto"}, "auto"),
    ({"type": "any"}, "required"),
    ({"type": "none"}, "none"),
    ({"type": "tool", "name": "Bash"}, {"type": "function", "function": {"name": "Bash"}}),
])
def test_tool_choice_maps_onto_the_openai_forms_the_server_enforces(choice, expect):
    body = claude_code_request()
    body["tool_choice"] = choice
    assert ac.anthropic_to_openai(body)["tool_choice"] == expect


def test_stop_sequences_and_sampling_pass_through():
    body = claude_code_request()
    body.update(stop_sequences = ["\n\nHuman:"], temperature = 0.2, top_p = 0.9, top_k = 40)
    req = ac.anthropic_to_openai(body)
    assert req["stop"] == ["\n\nHuman:"]
    assert (req["temperature"], req["top_p"], req["top_k"]) == (0.2, 0.9, 40)


def test_a_string_system_and_a_string_tool_result_are_accepted():
    body = claude_code_request()
    body["system"] = "plain"
    body["messages"][2]["content"][0]["content"] = "plain result"
    req = ac.anthropic_to_openai(body)
    assert req["messages"][0]["content"] == "plain"
    assert req["messages"][3]["content"] == "plain result"


def test_an_error_tool_result_says_so_and_an_image_is_replaced_by_a_note():
    body = claude_code_request()
    body["messages"][2]["content"][0]["is_error"] = True
    body["messages"][2]["content"].append(
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}})
    req = ac.anthropic_to_openai(body)
    assert req["messages"][3]["content"].startswith("Error: ")
    assert "[image omitted]" in req["messages"][4]["content"]
    assert "AAAA" not in json.dumps(req)


def test_builtin_tool_types_without_a_schema_are_skipped():
    body = claude_code_request()
    body["tools"].append({"type": "web_search_20250305", "name": "web_search"})
    assert [t["function"]["name"] for t in ac.anthropic_to_openai(body)["tools"]] == ["Read", "Bash"]


# --------------------------------------------------------------- response ----

def openai_response(**over):
    resp = {
        "id": "chatcmpl-abc123", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "Hello.",
                                 "reasoning_content": "think first"}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "total_tokens": 1020},
        "timings": {"cached_tokens": 900, "predicted_per_second": 50.0},
    }
    resp.update(over)
    return resp


def test_non_stream_response_becomes_thinking_then_text_with_cache_aware_usage():
    msg = ac.openai_to_anthropic(openai_response())
    assert msg["type"] == "message" and msg["role"] == "assistant" and msg["model"] == "m"
    assert msg["id"].startswith("msg_")
    assert msg["content"] == [{"type": "thinking", "thinking": "think first", "signature": ""},
                              {"type": "text", "text": "Hello."}]
    assert msg["stop_reason"] == "end_turn" and msg["stop_sequence"] is None
    assert msg["usage"] == {"input_tokens": 100, "output_tokens": 20,
                            "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0}


def test_tool_calls_become_tool_use_blocks_with_parsed_input():
    resp = openai_response()
    resp["choices"][0]["finish_reason"] = "tool_calls"
    resp["choices"][0]["message"] = {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "Read", "arguments": "{\"file_path\": \"a.py\"}"}}]}
    msg = ac.openai_to_anthropic(resp)
    assert msg["content"] == [{"type": "tool_use", "id": "call_1", "name": "Read",
                               "input": {"file_path": "a.py"}}]
    assert msg["stop_reason"] == "tool_use"


def test_length_finish_is_max_tokens():
    resp = openai_response()
    resp["choices"][0]["finish_reason"] = "length"
    assert ac.openai_to_anthropic(resp)["stop_reason"] == "max_tokens"


# ----------------------------------------------------------------- stream ----

def chunk(delta, finish = None, **extra):
    obj = {"id": "chatcmpl-s1", "object": "chat.completion.chunk", "model": "m",
           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    obj.update(extra)
    return obj


def drain(chunks):
    tr = ac.StreamTranslator(model = "m")
    events = []
    for c in chunks:
        events += tr.feed(c)
    return events


def test_stream_emits_the_anthropic_event_sequence_for_thinking_text_and_tool_use():
    events = drain([
        chunk({"reasoning_content": "think "}),
        chunk({"reasoning_content": "more"}),
        chunk({"content": "Hel"}),
        chunk({"content": "lo"}),
        chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                               "function": {"name": "Read", "arguments": "{\"file_path\": \"a\"}"}}]}),
        chunk({}, finish = "tool_calls",
              usage = {"prompt_tokens": 1000, "completion_tokens": 9, "total_tokens": 1009},
              timings = {"cached_tokens": 900}),
    ])
    names = [e for e, _ in events]
    assert names == [
        "message_start",
        "content_block_start", "content_block_delta", "content_block_delta", "content_block_delta", "content_block_stop",
        "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
        "content_block_start", "content_block_delta", "content_block_stop",
        "message_delta", "message_stop",
    ]
    data = [d for _, d in events]
    assert data[0]["message"]["role"] == "assistant" and data[0]["message"]["content"] == []
    # the thinking block opens WITHOUT a signature key and closes with an empty
    # signature_delta -- what llama-server's Anthropic route does (server-task.cpp
    # :827 and :918, "Anthropic API requires a signature_delta before closing
    # thinking blocks"); llama-server's thinking renders in Claude Code, ours did not
    assert data[1]["content_block"] == {"type": "thinking", "thinking": ""}
    assert data[2]["delta"] == {"type": "thinking_delta", "thinking": "think "}
    assert data[4]["delta"] == {"type": "signature_delta", "signature": ""}
    assert data[6]["content_block"] == {"type": "text", "text": ""}
    assert data[7]["delta"] == {"type": "text_delta", "text": "Hel"}
    assert data[10]["content_block"] == {"type": "tool_use", "id": "call_1", "name": "Read", "input": {}}
    assert data[11]["delta"] == {"type": "input_json_delta", "partial_json": "{\"file_path\": \"a\"}"}
    assert [d["index"] for d in (data[1], data[6], data[10])] == [0, 1, 2]
    assert data[13]["delta"]["stop_reason"] == "tool_use"
    assert data[13]["usage"] == {"input_tokens": 100, "output_tokens": 9,
                                 "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0}


def test_stream_with_no_deltas_still_closes_the_message():
    events = drain([chunk({}, finish = "stop",
                          usage = {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5})])
    assert [e for e, _ in events] == ["message_start", "message_delta", "message_stop"]
    assert events[1][1]["delta"]["stop_reason"] == "end_turn"


def test_a_server_error_chunk_becomes_an_error_event():
    events = drain([chunk({"content": "x"}), {"error": {"message": "context/cache: boom"}}])
    assert events[-1][0] == "error"
    assert events[-1][1]["error"]["message"] == "context/cache: boom"


def test_sse_encoding_is_event_then_data():
    line = ac.sse("message_stop", {"type": "message_stop"})
    assert line == b'event: message_stop\ndata: {"type": "message_stop"}\n\n'


# ------------------------------------------------------- keep-alive pings ----

def test_start_emits_message_start_once_and_feed_never_repeats_it():
    tr = ac.StreamTranslator(model = "m")
    assert [e for e, _ in tr.start()] == ["message_start"]
    assert tr.start() == []
    events = tr.feed(chunk({"content": "x"}))
    assert [e for e, _ in events] == ["content_block_start", "content_block_delta"]


def test_pump_pings_through_upstream_silence_then_translates_the_lines():
    """The llama.cpp lesson: a 45 s prefill is silence on the wire and Claude
    Code reports it as an API error. llama-server got --sse-ping-interval 5;
    here pump() yields a ping every ping_s of silence."""
    import asyncio

    async def upstream():
        await asyncio.sleep(0.25)                       # the "prefill"
        yield b"data: " + json.dumps(chunk({"content": "hi"})).encode() + b"\n"
        yield b"\n"
        yield b"data: " + json.dumps(chunk({}, finish = "stop",
                                           usage = {"prompt_tokens": 3, "completion_tokens": 1})).encode() + b"\n"
        yield b"data: [DONE]\n"
        yield b"data: {\"never\": 1}\n"                 # after DONE: must not be read

    async def run():
        tr = ac.StreamTranslator(model = "m")
        out = list(tr.start())
        async for ev in ac.pump(upstream(), tr, ping_s = 0.1):
            out.append(ev)
        return out

    names = [e for e, _ in asyncio.run(run())]
    assert names[0] == "message_start"
    assert names.count("ping") >= 2                     # 0.25 s of silence at 0.1 s
    assert names[-2:] == ["message_delta", "message_stop"]
    assert "content_block_delta" in names


# ------------------------------------------------ live output-token count ----

def test_running_usage_on_chunks_becomes_throttled_message_delta_usage_events():
    """2026-09-04: Claude Code's token counter only moved when a turn ended (it
    reads `message_delta.usage.output_tokens`, and ours came once, at the end),
    so it sat at 1.2k and jumped to 10k on the next tool call. The OpenAI route
    now puts a running `usage.completion_tokens` on every chunk; the translator
    turns that into interim `message_delta` events at most every EVERY tokens,
    with no stop_reason, and the final one still carries the real usage."""
    tr = ac.StreamTranslator(model = "m", interim_usage = True)
    events = []
    n = 0
    for i in range(1, 101):                       # 100 chunks, one token each
        n = i
        events += tr.feed(chunk({"content": "x"}, usage = {"completion_tokens": n}))
    events += tr.feed(chunk({}, finish = "stop",
                            usage = {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}))
    interim = [d for e, d in events if e == "message_delta" and "stop_reason" not in d["delta"]]
    final = [d for e, d in events if e == "message_delta" and d["delta"].get("stop_reason")]
    assert len(final) == 1 and final[0]["usage"]["output_tokens"] == 100
    assert 2 <= len(interim) <= 100 // ac.StreamTranslator.USAGE_EVERY + 1
    counts = [d["usage"]["output_tokens"] for d in interim]
    assert counts == sorted(counts) and counts[-1] <= 100
    assert all("input_tokens" not in d["usage"] for d in interim)   # unknown mid-stream: omitted
    # ordering: an interim usage event never splits a content block open/delta pair wrongly
    names = [e for e, _ in events]
    assert names[0] == "message_start" and names[-1] == "message_stop"
    assert names.index("content_block_start") < names.index("message_delta")


def test_chunks_without_usage_emit_no_interim_message_delta():
    tr = ac.StreamTranslator(model = "m")
    events = tr.feed(chunk({"content": "x"})) + tr.feed(chunk({"content": "y"}))
    assert [e for e, _ in events].count("message_delta") == 0


def test_thinking_deltas_carry_no_estimated_tokens_because_claude_code_adds_the_field_as_an_increment():
    """Observed 2026-09-04 ~20:10 on the interactive UI: the output counter read
    731.0k after four minutes at ~25 tok/s. Claude Code 2.1.258's thinking_delta
    handler (read from the binary) is

        if ("estimated_tokens" in delta) thinking_progress{estimatedTokensDelta: delta.estimated_tokens}
        else if (delta.thinking)         thinking_progress{estimatedTokensDelta: chars/4}
        ...  thinkingTokenEstimate += estimatedTokensDelta

    -- the field is consumed as an INCREMENT, whatever the SDK schema comment
    says about a running total. We sent the block's running total on every
    delta, so N deltas summed to N^2/2 (N ~ 1,200 -> 720k). llama-server never
    sends the field and Claude Code estimates from the thinking text itself, so
    neither do we: a thinking_delta carries only its text."""
    tr = ac.StreamTranslator(model = "m")
    events = tr.feed(chunk({"content": "Sure."}, usage = {"completion_tokens": 3}))
    events += tr.feed(chunk({"reasoning_content": "hmm"}, usage = {"completion_tokens": 7}))
    events += tr.feed(chunk({"reasoning_content": " more"}, usage = {"completion_tokens": 9}))
    events += tr.feed(chunk({"content": "Hi"}, usage = {"completion_tokens": 10}))
    deltas = [d["delta"] for e, d in events if e == "content_block_delta"]
    assert deltas[0] == {"type": "text_delta", "text": "Sure."}
    assert deltas[1] == {"type": "thinking_delta", "thinking": "hmm"}
    assert deltas[2] == {"type": "thinking_delta", "thinking": " more"}
    assert deltas[3] == {"type": "signature_delta", "signature": ""}
    assert deltas[4] == {"type": "text_delta", "text": "Hi"}
    # without a count on the chunk, the delta stays plain
    assert ac.StreamTranslator(model = "m").feed(chunk({"reasoning_content": "x"}))[-1][1]["delta"] == \
        {"type": "thinking_delta", "thinking": "x"}


# ------------------------------------ the llama-server gaps, closed (issue #74) ----

def test_history_thinking_becomes_reasoning_content_so_a_tool_loop_stays_byte_identical():
    """llama-server keeps prior thinking as `reasoning_content`; the Qwen3.8
    template renders reasoning for turns after the last user query, so inside
    a tool loop the model's own reasoning re-enters the prompt exactly as it
    was generated and the page cache can reuse it. Dropping it (what we did)
    forced a re-prefill of every generated token on the next tool turn."""
    req = ac.anthropic_to_openai(claude_code_request())
    asst = req["messages"][2]
    assert asst["reasoning_content"] == "I should read it."
    assert asst["content"] == "Reading."
    assert "signature" not in json.dumps(asst)


def test_thinking_only_assistant_turn_keeps_empty_content_and_its_reasoning():
    body = claude_code_request()
    body["messages"][1]["content"] = [{"type": "thinking", "thinking": "just thought", "signature": "s"}]
    asst = ac.anthropic_to_openai(body)["messages"][2]
    assert asst == {"role": "assistant", "content": "", "reasoning_content": "just thought"}


def test_mid_conversation_system_message_becomes_a_marked_user_turn():
    """Claude Code sends the `mid-conversation-system` beta; the Qwen template
    accepts one leading system message only (the fork merges directives into
    it), so an operator note in the middle is carried as a user turn with a
    marker rather than dropped or hoisted (hoisting would move the prefix)."""
    body = claude_code_request()
    body["messages"].append({"role": "system", "content": "Terse mode."})
    body["messages"].append({"role": "user", "content": "go on"})
    msgs = ac.anthropic_to_openai(body)["messages"]
    assert msgs[-2] == {"role": "user", "content": "System note: Terse mode."}
    assert msgs[-1] == {"role": "user", "content": "go on"}


def test_unknown_block_types_are_reported_not_silently_dropped():
    body = claude_code_request()
    body["messages"][2]["content"].append({"type": "tool_reference", "tool_name": "Grep"})
    body["messages"][2]["content"].append({"type": "document", "source": {"type": "text", "data": "x"}})
    req = ac.anthropic_to_openai(body)
    assert sorted(req["unknown_blocks"]) == ["document", "tool_reference"]
    # a clean request carries no such key at all
    assert "unknown_blocks" not in ac.anthropic_to_openai(claude_code_request())


def test_billing_header_cch_stamp_is_neutralised_like_llamacpp_pr_21793():
    s = "x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;You are Claude Code."
    assert ac.normalize_billing_header(s) == \
        "x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=fffff;You are Claude Code."
    plain = "x-anthropic-billing-header: cc_version=2.1.258.1e2; cc_entrypoint=cli;You are Claude Code."
    assert ac.normalize_billing_header(plain) == plain            # no stamp: untouched
    assert ac.normalize_billing_header("hello") == "hello"          # not the header: untouched
    body = claude_code_request()
    body["system"][0]["text"] = s
    assert "cch=fffff;" in ac.anthropic_to_openai(body)["messages"][0]["content"]


def test_too_long_message_is_the_exact_text_claude_code_compacts_on():
    import re
    msg = ac.too_long_message(270123, 245760)
    assert msg == "prompt is too long: 270123 tokens > 245760 maximum"
    # the regex Claude Code's binary applies, copied from it
    m = re.search(r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)", msg, re.I)
    assert m and (m.group(1), m.group(2)) == ("270123", "245760")


def test_interim_usage_events_are_off_by_default_because_they_freeze_claude_codes_live_view():
    """Observed 2026-09-04 evening on the interactive UI: thinking rendered for
    about one second, then the view froze until the turn ended and the counter
    jumped (1.4k -> 4.4k). One second at ~45 tok/s is the first interim
    message_delta at 32 tokens. The real API never sends a mid-turn
    message_delta; Claude Code reads a message_delta as the turn's end-of-stream
    bookkeeping. So the interim events are opt-in, and the thinking deltas carry
    text only (Claude Code estimates the count from it; see the test above)."""
    tr = ac.StreamTranslator(model = "m")
    events = []
    for i in range(1, 101):
        events += tr.feed(chunk({"reasoning_content": "x"}, usage = {"completion_tokens": i}))
    assert [e for e, _ in events].count("message_delta") == 0
    last = [d for e, d in events if e == "content_block_delta"][-1]["delta"]
    assert last == {"type": "thinking_delta", "thinking": "x"}
