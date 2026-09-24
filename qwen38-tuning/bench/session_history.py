"""Append-only JSONL history for one private benchmark session."""

from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

_SCHEMA_VERSION = 1
_OUTCOME_STATES = {
    "verified": "complete",
    "failed": "failed",
    "censored": "complete",
    "interrupted": "interrupted",
}


class SessionHistory:
    """Write one new session directory without recovery or replay support."""

    def __init__(self, directory: str | Path, session_id: str | None = None) -> None:
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise ValueError("session_id must be a nonempty string")
        self.directory = Path(directory)
        if self.directory.exists():
            raise FileExistsError(f"session directory already exists: {self.directory}")
        self.directory.mkdir()
        self.session_id = session_id or str(uuid.uuid4())
        self._events_path = self.directory / "events.jsonl"
        self._sequence = 0
        self.recording_write_s = 0.0
        self._started_at = time.monotonic()
        self._finished = False
        self._write_failed = False
        self._write_manifest({
            "schema_version": _SCHEMA_VERSION,
            "session_id": self.session_id,
            "state": "started",
        })

    def append(self, kind: str, payload: Any) -> dict[str, Any]:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("event kind must be nonempty")
        if self._write_failed:
            raise RuntimeError("journal persistence failed; session evidence is incomplete")
        if self._finished:
            raise RuntimeError("session has finished")
        sequence = self._sequence + 1
        event = {
            "schema_version": _SCHEMA_VERSION,
            "session_id": self.session_id,
            "sequence": sequence,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": time.monotonic() - self._started_at,
            "kind": kind,
            "payload": payload,
        }
        write_started = time.perf_counter()
        line = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with open(self._events_path, "a", encoding="utf-8", newline="\n") as handle:
            self._write_failed = True
            handle.write(line)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._write_failed = False
        self._sequence = sequence
        self.recording_write_s += time.perf_counter() - write_started
        return event

    def record_request(self, request_id: str, test: Mapping[str, Any],
                       context: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
        """Keep actual prompt depth distinct from allocation and declared capacity.

        input_tokens is total rendered input, including cached input. A caller
        must normalize backend usage semantics and name its measurement source.
        None means unmeasured, never zero or the configured window.
        """
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be nonempty")
        for key in ('format', 'case_id', 'stage'):
            if not isinstance(test.get(key), str) or not test[key].strip():
                raise ValueError(f"missing test {key}")
        for key in ('round', 'attempt'):
            if type(test.get(key)) is not int or test[key] < 1:
                raise ValueError(f"invalid test {key}")
        required = ('requested_window_tokens', 'runtime_window_tokens', 'input_tokens',
                    'cached_input_tokens', 'max_output_tokens', 'token_count_source',
                    'history_policy', 'source')
        if any(key not in context for key in required):
            raise ValueError("incomplete context provenance")
        for key in required[:5]:
            value = context[key]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"invalid token count: {key}")
        if any(context[key] is not None for key in ('input_tokens', 'cached_input_tokens')):
            if not context['token_count_source']:
                raise ValueError("measured depth requires token_count_source")
        if not context['history_policy'] or not context['source']:
            raise ValueError("history policy and source must be explicit")
        # Retain even overflow/mismatch evidence; downstream eligibility checks
        # decide whether it can enter a comparison, rather than losing the row.
        return self.append('request', {'request_id': request_id, 'test': dict(test),
                                      'context': dict(context), 'request': dict(request)})

    def finish(self, outcome: str, metadata: Any = None) -> dict[str, Any]:
        if self._write_failed:
            raise RuntimeError("journal persistence failed; session evidence is incomplete")
        if self._finished:
            raise RuntimeError("session has finished")
        try:
            state = _OUTCOME_STATES[outcome]
        except KeyError as error:
            raise ValueError(f"unsupported outcome: {outcome}") from error
        manifest: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "session_id": self.session_id,
            "state": state,
            "outcome": outcome,
        }
        raw = self._events_path.read_bytes() if self._events_path.exists() else b""
        manifest["event_count"] = self._sequence
        manifest["events_sha256"] = hashlib.sha256(raw).hexdigest()
        if metadata is not None:
            manifest["metadata"] = metadata
        self._write_manifest(manifest)
        self._finished = True
        return manifest

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".manifest-", suffix=".json", dir=self.directory, text=True
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(manifest, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.directory / "manifest.json")
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


def inspect_session(directory: str | Path) -> dict[str, object]:
    """Check evidence finalization, not whether the model's answer is correct."""
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        path = directory / "events.jsonl"
        raw = path.read_bytes() if path.exists() else b""
        restored = _parse_events(raw)
        events = restored["events"]
        terminal = manifest.get("outcome") in _OUTCOME_STATES
        complete = (terminal and not restored["interrupted"]
                    and manifest.get("schema_version") == _SCHEMA_VERSION
                    and manifest.get("state") == _OUTCOME_STATES[manifest["outcome"]]
                    and manifest.get("event_count") == len(events)
                    and manifest.get("events_sha256") == hashlib.sha256(raw).hexdigest()
                    and all(event["session_id"] == manifest.get("session_id") for event in events))
        return {"complete": complete, "state": manifest.get("state"),
                "outcome": manifest.get("outcome"), "events": events,
                "note": None if complete else "session unfinished or evidence integrity mismatch"}
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
        return {"complete": False, "state": "unreadable", "events": [],
                "note": type(error).__name__}


def classify_thinking(message: Mapping[str, object]) -> str:
    """Classify only supplied reasoning content; never infer hidden thinking."""
    value = message.get("reasoning_content")
    if value is None:
        return "unavailable"
    if not isinstance(value, str):
        raise TypeError("reasoning_content must be a string or None")
    return "present-empty" if value == "" else "present-text"


def read_events(path: str | Path) -> dict[str, object]:
    """Read complete events and report an incomplete final JSON fragment."""
    return _parse_events(Path(path).read_bytes())


def _parse_events(raw: bytes) -> dict[str, object]:
    lines = raw.split(b"\n")
    has_final_terminator = raw.endswith(b"\n")
    complete_lines = lines[:-1]
    tail = b"" if has_final_terminator else lines[-1]

    events: list[dict[str, Any]] = []
    session_id: str | None = None
    for line_number, line in enumerate(complete_lines, start=1):
        if not line:
            raise ValueError(f"malformed event at line {line_number}")
        try:
            event = _parse_and_validate_event(line, line_number, len(events) + 1, session_id)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"malformed event at line {line_number}") from error
        if session_id is None:
            session_id = event["session_id"]
        events.append(event)

    if not tail:
        return {"events": events, "interrupted": False}
    try:
        event = _parse_and_validate_event(tail, len(complete_lines) + 1, len(events) + 1, session_id)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"events": events, "interrupted": True}
    if session_id is not None and event["session_id"] != session_id:
        raise ValueError(f"session_id mismatch at line {len(complete_lines) + 1}")
    events.append(event)
    return {"events": events, "interrupted": False}


def _parse_and_validate_event(
    line: bytes, line_number: int, expected_sequence: int, expected_session_id: str | None
) -> dict[str, Any]:
    decoded = line.decode("utf-8")
    parsed = json.loads(decoded)
    if not isinstance(parsed, dict):
        raise ValueError(f"malformed event at line {line_number}")
    required = {
        "schema_version",
        "session_id",
        "sequence",
        "timestamp",
        "elapsed_seconds",
        "kind",
        "payload",
    }
    if set(parsed) != required:
        raise ValueError(f"malformed event at line {line_number}")
    if parsed["schema_version"] != _SCHEMA_VERSION:
        raise ValueError(f"unsupported schema at line {line_number}")
    if not isinstance(parsed["session_id"], str) or not parsed["session_id"].strip():
        raise ValueError(f"malformed session_id at line {line_number}")
    if expected_session_id is not None and parsed["session_id"] != expected_session_id:
        raise ValueError(f"session_id mismatch at line {line_number}")
    sequence = parsed["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != expected_sequence:
        raise ValueError(f"sequence mismatch at line {line_number}")
    if not isinstance(parsed["timestamp"], str):
        raise ValueError(f"malformed timestamp at line {line_number}")
    stamp = datetime.fromisoformat(parsed["timestamp"].replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError(f"timestamp has no timezone at line {line_number}")
    elapsed = parsed["elapsed_seconds"]
    if (not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
            or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError(f"malformed elapsed_seconds at line {line_number}")
    if not isinstance(parsed["kind"], str) or not parsed["kind"].strip():
        raise ValueError(f"malformed kind at line {line_number}")
    return parsed
