"""Lossless offline analysis for raw Claude session-history events.

Unlike ``quality-bench.parse_stream``, this module receives already-decoded raw
stream-json dictionaries and never silently skips malformed relevant data:
structural problems are recorded in ``errors``.  It analyzes only; callers keep
and control access to the raw event stream.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


_COMPLETE_ASSISTANT = "assistant"


def _canonical(value: Any) -> str:
    """Return a deterministic JSON representation without interpreting inputs."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return json.dumps(repr(value), ensure_ascii=False)


def _content_blocks(message: dict[str, Any], errors: list[str], event_index: int) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        errors.append(f"event {event_index}: message content is not a list")
        return []
    blocks: list[dict[str, Any]] = []
    for block_index, block in enumerate(content):
        if isinstance(block, dict):
            blocks.append(block)
        else:
            errors.append(f"event {event_index}: content block {block_index} is not an object")
    return blocks


def _assistant_fingerprint(message: dict[str, Any]) -> str:
    return _canonical(message)


def analyze_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize complete raw Claude stream-json events without executing data.

    ``events`` must contain raw event dictionaries, not session-history journal
    envelopes.  Only complete ``type == 'assistant'`` messages contribute to
    output counts; partial ``stream_event`` deltas intentionally do not.
    """
    errors: list[str] = []
    tool_calls: Counter[str] = Counter()
    fingerprints: Counter[str] = Counter()
    tool_uses: dict[str, tuple[str, str, Any]] = {}
    tool_results: set[str] = set()
    seen_assistants: dict[tuple[str, str], str] = {}
    message_thinking: dict[str, int] = {}
    model_names: set[str] = set()
    thinking_states: Counter[str] = Counter({"unavailable": 0, "present-empty": 0, "present-text": 0})
    compactions: list[Any] = []
    result: dict[str, Any] | None = None
    assistant_messages = thinking_chars = answer_chars = tool_errors = 0

    if not isinstance(events, list):
        raise TypeError("events must be a list of raw event dictionaries")

    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"event {event_index}: event is not an object")
            continue
        event_type = event.get("type")

        if event_type == _COMPLETE_ASSISTANT:
            message = event.get("message")
            if not isinstance(message, dict):
                errors.append(f"event {event_index}: assistant message is not an object")
                continue
            message_id = message.get("id")
            if not isinstance(message_id, str) or not message_id:
                errors.append(f"event {event_index}: assistant message has no id")
                continue
            fingerprint = _assistant_fingerprint(message)
            # Real CLI emits separately completed blocks with one API message ID.
            event_uuid = event.get('uuid')
            identity = (('event', event_uuid) if isinstance(event_uuid, str) and event_uuid
                        else ('message', message_id))
            prior = seen_assistants.get(identity)
            if prior is not None:
                if prior != fingerprint:
                    errors.append(f"event {event_index}: conflicting assistant message id {message_id}")
                continue
            seen_assistants[identity] = fingerprint
            if message_id not in message_thinking:
                assistant_messages += 1
                message_thinking[message_id] = 0

            model = message.get("model")
            if isinstance(model, str) and model and model != "<synthetic>":
                model_names.add(model)
            blocks = _content_blocks(message, errors, event_index)
            has_thinking = False
            has_thinking_text = False
            for block in blocks:
                block_type = block.get("type")
                if block_type == "text":
                    value = block.get("text", "")
                    if not isinstance(value, str):
                        errors.append(f"event {event_index}: text block is not a string")
                    else:
                        answer_chars += len(value)
                elif block_type == "thinking":
                    value = block.get("thinking", "")
                    if not isinstance(value, str):
                        errors.append(f"event {event_index}: thinking block is not a string")
                    else:
                        has_thinking = True
                        has_thinking_text = has_thinking_text or bool(value)
                        thinking_chars += len(value)
                elif block_type == "tool_use":
                    name = block.get("name")
                    tool_id = block.get("id")
                    if not isinstance(name, str) or not name:
                        errors.append(f"event {event_index}: tool_use has no name")
                        continue
                    tool_calls[name] += 1
                    input_data = block.get("input")
                    canonical_input = _canonical(input_data)
                    fingerprints[f'{{"input":{canonical_input},"name":{_canonical(name)}}}'] += 1
                    if not isinstance(tool_id, str) or not tool_id:
                        errors.append(f"event {event_index}: tool_use {name} has no id")
                        continue
                    prior_tool = tool_uses.get(tool_id)
                    current_tool = (name, canonical_input, input_data)
                    if prior_tool is not None:
                        if prior_tool != current_tool:
                            errors.append(f"event {event_index}: conflicting tool_use id {tool_id}")
                        else:
                            errors.append(f"event {event_index}: duplicate tool_use id {tool_id}")
                        continue
                    tool_uses[tool_id] = current_tool
            message_thinking[message_id] = max(message_thinking[message_id],
                                                2 if has_thinking_text else 1 if has_thinking else 0)

        elif event_type == "user":
            message = event.get("message")
            if not isinstance(message, dict):
                errors.append(f"event {event_index}: user message is not an object")
                continue
            for block in _content_blocks(message, errors, event_index):
                if block.get("type") != "tool_result":
                    continue
                tool_id = block.get("tool_use_id")
                if not isinstance(tool_id, str) or not tool_id:
                    errors.append(f"event {event_index}: tool_result has no tool_use_id")
                    continue
                tool_results.add(tool_id)
                if block.get("is_error") is True:
                    tool_errors += 1

        elif event_type == "system" and event.get("subtype") == "compact_boundary":
            if "compact_metadata" not in event:
                errors.append(f"event {event_index}: compact_boundary has no compact_metadata")
            compactions.append(event.get("compact_metadata"))

        elif event_type == "result":
            if result is not None:
                errors.append(f"event {event_index}: multiple result events")
            result = event

    for state in message_thinking.values():
        thinking_states[('unavailable', 'present-empty', 'present-text')[state]] += 1
    orphan_ids = sorted({tool_id for tool_id in tool_results if tool_id not in tool_uses})
    resolved_ids = tool_results
    unresolved_ids = sorted(tool_id for tool_id in tool_uses if tool_id not in resolved_ids)
    repeat_tool_calls = {fingerprint: count for fingerprint, count in sorted(fingerprints.items()) if count > 1}

    return {
        "assistant_messages": assistant_messages,
        "tool_calls": dict(sorted(tool_calls.items())),
        "tool_uses": {
            tool_id: {"name": name, "input": input_data}
            for tool_id, (name, _canonical_input, input_data) in sorted(tool_uses.items())
        },
        "tool_errors": tool_errors,
        "unresolved_tool_ids": unresolved_ids,
        "orphan_tool_result_ids": orphan_ids,
        "model_names": sorted(model_names),
        "thinking_chars": thinking_chars,
        "answer_chars": answer_chars,
        "thinking_states": dict(thinking_states),
        "repeat_tool_calls": repeat_tool_calls,
        "compactions": compactions,
        "errors": errors,
        "result": result,
    }


def _fence(text: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _block(label: str, value: Any) -> str:
    text = value if isinstance(value, str) else _canonical(value)
    fence = _fence(text)
    return f"### {label}\n{fence}{label}\n{text}\n{fence}\n"


def render_history(events: list[dict[str, Any]]) -> str:
    """Render a private, lossless Markdown view of relevant raw event content."""
    if not isinstance(events, list):
        raise TypeError("events must be a list of raw event dictionaries")
    sections = ["# Session history (private)\n"]
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            sections.append(_block(f"event {event_index} malformed", event))
            continue
        event_type = event.get("type")
        if event_type in {"assistant", "user"}:
            message = event.get("message")
            if not isinstance(message, dict):
                sections.append(_block(f"{event_type} event {event_index} malformed", event))
                continue
            content = message.get("content")
            if isinstance(content, str):
                sections.append(_block(f"{event_type} text", content))
                continue
            if not isinstance(content, list):
                sections.append(_block(f"{event_type} content", content))
                continue
            for block in content:
                if not isinstance(block, dict):
                    sections.append(_block(f"{event_type} malformed content", block))
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    sections.append(_block(f"{event_type} text", block.get("text", "")))
                elif block_type == "thinking":
                    sections.append(_block("assistant thinking", block.get("thinking", "")))
                elif block_type == "tool_use":
                    name = block.get("name", "unknown")
                    tool_id = block.get("id", "unknown")
                    sections.append(_block(f"assistant tool input: {name} ({tool_id})", block.get("input")))
                elif block_type == "tool_result":
                    tool_id = block.get("tool_use_id", "unknown")
                    sections.append(_block(f"user tool result: {tool_id}", block.get("content")))
                else:
                    sections.append(_block(f"{event_type} content: {block_type or 'unknown'}", block))
        elif event_type == "system" and event.get("subtype") == "compact_boundary":
            sections.append(_block("system compact_boundary metadata", event.get("compact_metadata")))
    return "\n".join(sections)
