"""Contract tests for append-only benchmark session history."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import session_history
from session_history import SessionHistory, classify_thinking, read_events


def test_append_persists_unicode_full_payload_and_order(tmp_path: Path) -> None:
    history = SessionHistory(tmp_path / "private-run", session_id="run-ไทย")
    payload = {"message": "ไทย 日本語 😀", "nested": ["control-safe", {"all": "kept"}]}

    first = history.append("request", payload)
    second = history.append("response", {"answer": "ครบถ้วน"})
    restored = read_events(tmp_path / "private-run" / "events.jsonl")

    assert first["schema_version"] == 1
    assert first["session_id"] == "run-ไทย"
    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert first["payload"] == payload
    assert restored == {"events": [first, second], "interrupted": False}
    assert first["elapsed_seconds"] >= 0
    assert datetime.fromisoformat(first["timestamp"].replace("Z", "+00:00")).tzinfo is not None


def test_manifest_starts_and_finishes_in_explicit_terminal_state(tmp_path: Path) -> None:
    directory = tmp_path / "private-run"
    history = SessionHistory(directory, session_id="run-1")

    assert json.loads((directory / "manifest.json").read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "session_id": "run-1",
        "state": "started",
    }

    manifest = history.finish("verified", {"evidence": "pytest"})

    assert manifest == {
        "schema_version": 1,
        "session_id": "run-1",
        "state": "complete",
        "outcome": "verified",
        "metadata": {"evidence": "pytest"},
        "event_count": 0,
        "events_sha256": __import__('hashlib').sha256(b'').hexdigest(),
    }
    assert json.loads((directory / "manifest.json").read_text(encoding="utf-8")) == manifest


@pytest.mark.parametrize(
    ("outcome", "state"),
    [("verified", "complete"), ("failed", "failed"), ("censored", "complete"), ("interrupted", "interrupted")],
)
def test_finish_records_each_allowed_terminal_outcome(tmp_path: Path, outcome: str, state: str) -> None:
    manifest = SessionHistory(tmp_path / outcome, session_id="run-1").finish(outcome)

    assert manifest["state"] == state
    assert manifest["outcome"] == outcome


def test_finish_rejects_unlisted_outcome(tmp_path: Path) -> None:
    history = SessionHistory(tmp_path / "private-run", session_id="run-1")

    with pytest.raises(ValueError, match="unsupported outcome"):
        history.finish("complete")


def test_refuses_existing_session_directory_and_post_finish_append(tmp_path: Path) -> None:
    directory = tmp_path / "private-run"
    history = SessionHistory(directory, session_id="run-1")

    with pytest.raises(FileExistsError):
        SessionHistory(directory, session_id="run-1")

    history.finish("interrupted")
    with pytest.raises(RuntimeError, match="finished"):
        history.append("late", {})


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ({}, "unavailable"),
        ({"reasoning_content": None}, "unavailable"),
        ({"reasoning_content": ""}, "present-empty"),
        ({"reasoning_content": "visible reasoning"}, "present-text"),
    ],
)
def test_classify_thinking_does_not_infer_hidden_content(message: dict[str, object], expected: str) -> None:
    assert classify_thinking(message) == expected


def test_read_events_marks_only_truncated_final_line_as_interrupted(tmp_path: Path) -> None:
    directory = tmp_path / "private-run"
    history = SessionHistory(directory, session_id="run-1")
    event = history.append("request", {"ok": True})
    events_path = directory / "events.jsonl"
    with events_path.open("ab") as handle:
        handle.write(b'{"schema_version":1')

    assert read_events(events_path) == {"events": [event], "interrupted": True}


def test_read_events_rejects_malformed_internal_record(tmp_path: Path) -> None:
    directory = tmp_path / "private-run"
    history = SessionHistory(directory, session_id="run-1")
    history.append("request", {})
    events_path = directory / "events.jsonl"
    with events_path.open("ab") as handle:
        handle.write(b"not json\n")

    with pytest.raises(ValueError, match="line 2"):
        read_events(events_path)


def test_fsync_failure_cannot_be_followed_by_verified_completion(tmp_path, monkeypatch):
    """An uncertain persisted event must not produce a believable success manifest."""
    history = SessionHistory(tmp_path / 'private-run')
    def fail_sync(fd):
        raise OSError('flush failed')
    with monkeypatch.context() as patch:
        patch.setattr(session_history.os, 'fsync', fail_sync)
        with pytest.raises(OSError):
            history.append('response', {'content': 'done'})
    with pytest.raises(RuntimeError):
        history.finish('verified')
    with pytest.raises(RuntimeError):
        history.append('response', {'content': 'retry'})


def test_inspection_distinguishes_unfinished_session_from_complete_jsonl(tmp_path):
    directory = tmp_path / 'unfinished'
    history = SessionHistory(directory)
    history.append('response', {'text': 'partial task'})
    result = session_history.inspect_session(directory)
    assert result['complete'] is False
    assert result['state'] == 'started'


def test_inspection_detects_tampering_after_finalization(tmp_path):
    directory = tmp_path / 'tampered'
    history = SessionHistory(directory)
    history.append('response', {'text': 'original'})
    history.finish('verified', {'verification': 'passed'})
    assert session_history.inspect_session(directory)['complete'] is True
    events = directory / 'events.jsonl'
    events.write_text(events.read_text().replace('original', 'modified'), encoding='utf-8')
    assert session_history.inspect_session(directory)['complete'] is False


def test_writer_rejects_invalid_identity_before_creating_directory(tmp_path):
    directory = tmp_path / 'bad-id'
    with pytest.raises(ValueError):
        SessionHistory(directory, session_id=12)
    assert not directory.exists()


@pytest.mark.parametrize('field,value', [('elapsed_seconds', float('nan')),
    ('elapsed_seconds', -1), ('timestamp', 'not-a-timestamp'), ('kind', ''), ('session_id', '')])
def test_reader_rejects_invalid_event_metadata(tmp_path, field, value):
    history = SessionHistory(tmp_path / 'bad-event')
    event = history.append('response', {})
    event[field] = value
    path = history.directory / 'events.jsonl'
    path.write_text(json.dumps(event) + '\n', encoding='utf-8')
    with pytest.raises(ValueError):
        read_events(path)


def test_append_rejects_empty_kind_before_writing(tmp_path):
    history = SessionHistory(tmp_path / 'empty-kind')
    with pytest.raises(ValueError):
        history.append('', {})
    assert not (history.directory / 'events.jsonl').exists()


def test_request_record_separates_test_shape_prompt_depth_and_window(tmp_path):
    history = SessionHistory(tmp_path / 'context-record')
    test = {'format': 'long-horizon-agent', 'case_id': 'repair-inventory',
            'stage': 'regression-recovery', 'round': 2, 'attempt': 1}
    context = {'requested_window_tokens': 131072, 'runtime_window_tokens': 131072,
               'input_tokens': 10000, 'cached_input_tokens': 8000,
               'max_output_tokens': 4096, 'token_count_source': 'server_usage',
               'history_policy': 'original', 'source': 'fixture-session'}
    request = {'messages': [{'role': 'user', 'content': 'แก้ข้อผิดพลาด'}]}
    event = history.record_request('req-1', test, context, request)
    restored = read_events(history.directory / 'events.jsonl')['events'][0]['payload']
    assert event['kind'] == 'request'
    assert restored['test'] == test
    assert restored['context']['input_tokens'] == 10000
    assert restored['context']['runtime_window_tokens'] == 131072
    assert restored['context']['cached_input_tokens'] == 8000
    assert restored['request'] == request
    assert restored['request_id'] == 'req-1'


def test_request_record_keeps_unknown_depth_unknown(tmp_path):
    history = SessionHistory(tmp_path / 'unknown-depth')
    event = history.record_request('req-2', {'format': 'context-certification',
        'case_id': 'retention', 'stage': 'prefill', 'round': 1, 'attempt': 1},
        {'requested_window_tokens': 65536, 'runtime_window_tokens': 65536,
         'input_tokens': None, 'cached_input_tokens': None, 'max_output_tokens': 4096,
         'token_count_source': None, 'history_policy': 'compacted', 'source': 'fixture-session'}, {})
    assert event['payload']['context']['input_tokens'] is None
    assert event['payload']['context']['history_policy'] == 'compacted'


def test_request_record_rejects_depth_without_measurement_source(tmp_path):
    history = SessionHistory(tmp_path / 'unsourced-depth')
    with pytest.raises(ValueError):
        history.record_request('req-3', {'format': 'single-answer', 'case_id': 'code',
            'stage': 'answer', 'round': 1, 'attempt': 1},
            {'requested_window_tokens': 65536, 'runtime_window_tokens': 65536,
             'input_tokens': 9000, 'cached_input_tokens': None, 'max_output_tokens': 4096,
             'token_count_source': None, 'history_policy': 'original', 'source': 'fixture'}, {})


def test_records_journal_write_overhead_separately(tmp_path, monkeypatch):
    history = SessionHistory(tmp_path / 'overhead')
    ticks = iter((10.0, 10.25))
    monkeypatch.setattr(session_history.time, 'perf_counter', lambda: next(ticks))
    history.append('request', {'text': 'prompt'})
    assert history.recording_write_s == 0.25


def test_append_write_failure_does_not_consume_sequence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    history = SessionHistory(tmp_path / "private-run", session_id="run-1")

    def fail_open(*args: object, **kwargs: object) -> object:
        raise OSError("disk full")

    monkeypatch.setattr(session_history, "open", fail_open, raising=False)
    with pytest.raises(OSError, match="disk full"):
        history.append("request", {})
    monkeypatch.undo()

    assert history.append("request", {})["sequence"] == 1
