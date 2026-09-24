"""Regression tests for lossless offline Claude session analysis."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from session_analysis import analyze_events, render_history


def assistant(message_id, content, model="claude-test"):
    return {
        "type": "assistant",
        "message": {"id": message_id, "model": model, "content": content},
    }


def test_analyze_complete_assistant_messages_not_stream_partials_or_duplicate_ids():
    events = [
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"text": "partial"}}},
        assistant("msg-1", [{"type": "thinking", "thinking": "plan"}, {"type": "text", "text": "answer"}]),
        assistant("msg-1", [{"type": "thinking", "thinking": "plan"}, {"type": "text", "text": "answer"}]),
    ]

    summary = analyze_events(events)

    assert summary["assistant_messages"] == 1
    assert summary["thinking_chars"] == 4
    assert summary["answer_chars"] == 6
    assert summary["thinking_states"] == {"unavailable": 0, "present-empty": 0, "present-text": 1}
    assert summary["errors"] == []


def test_analyze_thinking_states_are_per_complete_message():
    summary = analyze_events([
        assistant("no-thinking", [{"type": "text", "text": "a"}]),
        assistant("empty-thinking", [{"type": "thinking", "thinking": ""}]),
        assistant("visible-thinking", [{"type": "thinking", "thinking": "reason"}]),
    ])

    assert summary["thinking_states"] == {"unavailable": 1, "present-empty": 1, "present-text": 1}
    assert summary["thinking_chars"] == len("reason")


def test_analyze_keeps_full_text_and_detects_conflicting_message_id():
    long_text = "x" * 12_000
    summary = analyze_events([
        assistant("same", [{"type": "text", "text": long_text}]),
        assistant("same", [{"type": "text", "text": "different"}]),
    ])

    assert summary["answer_chars"] == len(long_text)
    assert summary["assistant_messages"] == 1
    assert any("conflicting assistant message id same" in error for error in summary["errors"])


def test_analyze_joins_tool_results_without_executing_malformed_input():
    malformed_input = '{"path": broken'
    summary = analyze_events([
        assistant("tools", [{"type": "tool_use", "id": "tool-1", "name": "Read", "input": malformed_input}]),
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "tool-1", "content": "permission denied", "is_error": True},
            {"type": "tool_result", "tool_use_id": "orphan", "content": "no call"},
        ]}},
    ])

    assert summary["tool_calls"] == {"Read": 1}
    assert summary["tool_uses"] == {"tool-1": {"name": "Read", "input": malformed_input}}
    assert summary["tool_errors"] == 1
    assert summary["unresolved_tool_ids"] == []
    assert summary["orphan_tool_result_ids"] == ["orphan"]
    assert summary["repeat_tool_calls"] == {}


def test_analyze_reports_unresolved_repeats_models_compaction_and_raw_result():
    repeated = {"path": "C:/AI/a.py", "line": 1}
    result = {"type": "result", "usage": {"output_tokens": 9}, "duration_ms": 20}
    summary = analyze_events([
        assistant("one", [{"type": "tool_use", "id": "a", "name": "Read", "input": repeated}]),
        assistant("two", [{"type": "tool_use", "id": "b", "name": "Read", "input": {"line": 1, "path": "C:/AI/a.py"}}], model="claude-other"),
        {"type": "system", "subtype": "compact_boundary", "compact_metadata": {"trigger": "auto", "pre_tokens": 99}},
        result,
    ])

    assert summary["unresolved_tool_ids"] == ["a", "b"]
    assert summary["tool_calls"] == {"Read": 2}
    assert summary["model_names"] == ["claude-other", "claude-test"]
    assert summary["repeat_tool_calls"] == {'{"input":{"line":1,"path":"C:/AI/a.py"},"name":"Read"}': 2}
    assert summary["compactions"] == [{"trigger": "auto", "pre_tokens": 99}]
    assert summary["result"] == result


def test_real_cli_emits_distinct_blocks_with_same_message_id_and_unique_event_uuids():
    # Installed CLI2.1.258 emits one assistant event per completed content block.
    first = dict(assistant('shared', [{'type':'thinking','thinking':'plan'}]), uuid='event-1')
    second = dict(assistant('shared', [{'type':'tool_use','id':'call-1','name':'Edit','input':{}}]), uuid='event-2')
    summary = analyze_events([first, second, first,
        {'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':'call-1','content':'ok'}]}}])
    assert summary['errors'] == []
    assert summary['assistant_messages'] == 1
    assert summary['thinking_chars'] == 4
    assert summary['tool_calls'] == {'Edit':1}
    assert summary['thinking_states'] == {'unavailable':0,'present-empty':0,'present-text':1}
    assert summary['orphan_tool_result_ids'] == []


def test_render_history_keeps_all_content_and_uses_safe_fences():
    events = [
        {"type": "user", "message": {"content": [
            {"type": "text", "text": "user ``` snippet"},
            {"type": "tool_result", "tool_use_id": "call", "content": "result ```` value", "is_error": True},
        ]}},
        assistant("render", [
            {"type": "thinking", "thinking": "thinking ``` value"},
            {"type": "text", "text": "assistant text"},
            {"type": "tool_use", "id": "call", "name": "Bash", "input": {"command": "echo ```"}},
        ]),
        {"type": "system", "subtype": "compact_boundary", "compact_metadata": {"reason": "limit"}},
    ]

    history = render_history(events)

    for value in ("user ``` snippet", "result ```` value", "thinking ``` value", "assistant text", '"command":"echo ```"', '"reason":"limit"'):
        assert value in history
    assert "# Session history (private)" in history
    assert "`````user tool result: call" in history
