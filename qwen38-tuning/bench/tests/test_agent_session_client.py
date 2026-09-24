"""Behavioral contracts for the isolated local Claude CLI session client."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from agent_session_client import build_client_argv, build_client_env, run_client
from session_history import SessionHistory, read_events


TOOLS = "Read,Write,Edit,Glob,Grep"


def _history(tmp_path: Path, name: str) -> SessionHistory:
    return SessionHistory(tmp_path / name, session_id=name)


def _child(source: str) -> list[str]:
    return [sys.executable, "-u", "-c", source]


def test_build_client_argv_is_restricted_and_has_no_bash() -> None:
    argv = build_client_argv("claude", "local-model", "session-uuid", "C:/scratch/settings.json")

    assert argv == [
        "claude",
        "--bare",
        "--restricted",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--settings",
        "C:/scratch/settings.json",
        "--permission-mode",
        "dontAsk",
        "--tools",
        TOOLS,
        "--allowedTools",
        TOOLS,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--session-id",
        "session-uuid",
        "--model",
        "local-model",
    ]


def test_build_client_argv_can_expose_only_file_and_fixed_test_tools(tmp_path: Path) -> None:
    config = tmp_path / 'mcp.json'
    config.write_text('{"mcpServers":{"visible-tests":{}}}')
    argv = build_client_argv('claude', 'local', '00000000-0000-0000-0000-000000000001',
                             'settings.json', mcp_config=config,
                             allowed_tools='Read,Write,Edit,Glob,Grep,mcp__visible-tests__run_visible_tests')
    assert argv[argv.index('--mcp-config') + 1] == str(config)
    tools = argv[argv.index('--tools') + 1]
    assert tools == 'Read,Write,Edit,Glob,Grep,mcp__visible-tests__run_visible_tests'
    assert argv[argv.index('--allowedTools') + 1] == tools
    assert 'Bash' not in tools and 'PowerShell' not in tools


def test_default_history_id_is_accepted_by_claude_session_id_contract(tmp_path: Path) -> None:
    # 2026-09-21: the canary passed uuid4().hex to --session-id. Claude's
    # installed parser requires hyphens and exits before emitting any events.
    from uuid import UUID

    history = SessionHistory(tmp_path / "default-id")
    argv = build_client_argv("claude", "local-model", history.session_id, "settings.json")
    client_id = argv[argv.index("--session-id") + 1]
    assert client_id == str(UUID(client_id))
    event = history.append("user_prompt", "canary")
    manifest = history.finish("interrupted")
    assert client_id == event["session_id"] == manifest["session_id"]


def test_build_client_env_scrubs_credentials_and_routes_only_to_loopback(tmp_path: Path) -> None:
    env = build_client_env(
        {
            "PATH": "C:/Windows/System32",
            "SystemRoot": "C:/Windows",
            "ANTHROPIC_API_KEY": "real-key",
            "ANTHROPIC_AUTH_TOKEN": "oauth",
            "CLAUDE_CONFIG_DIR": "inherited",
            "CLAUDECODE": "nested",
            "AWS_ACCESS_KEY_ID": "aws-key",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "GOOGLE_APPLICATION_CREDENTIALS": "google-creds",
            "AZURE_CLIENT_SECRET": "azure-secret",
        },
        tmp_path / "claude-config",
        "http://127.0.0.1:8080",
    )

    assert env["PATH"] == "C:/Windows/System32"
    assert env["SystemRoot"] == "C:/Windows"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
    assert env["ANTHROPIC_API_KEY"] == "local-benchmark-placeholder"
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "claude-config")
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    for name in (
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDECODE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AZURE_CLIENT_SECRET",
    ):
        assert name not in env


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:8080",
        "http://127.0.0.1",
        "http://127.0.0.1:8080/path",
        "http://user@127.0.0.1:8080",
        "http://127.0.0.1:8080?x=1",
        "http://127.0.0.1:8080#fragment",
        "http://192.168.1.2:8080",
    ],
)
def test_build_client_env_denies_non_local_or_ambiguous_endpoints(tmp_path: Path, endpoint: str) -> None:
    with pytest.raises(ValueError):
        build_client_env({}, tmp_path / "config", endpoint)


def test_run_client_records_prompt_events_result_and_private_stderr(tmp_path: Path) -> None:
    history = _history(tmp_path, "complete")
    metadata = run_client(
        _child(
            "import json, sys; "
            "prompt = sys.stdin.buffer.read().decode('utf-8'); "
            "print(json.dumps({'type': 'assistant', 'message': prompt}), flush=True); "
            "print(json.dumps({'type': 'result', 'is_error': False, 'result': 'done'}), flush=True); "
            "print('private diagnostic', file=sys.stderr, flush=True)"
        ),
        os.environ.copy(),
        tmp_path,
        "edit only the canary file",
        history,
    )

    assert metadata == {"status": "completed", "returncode": 0}
    events = read_events(history.directory / "events.jsonl")["events"]
    assert [event["kind"] for event in events] == ["user_prompt", "client_event", "client_event"]
    assert events[0]["payload"] == "edit only the canary file"
    assert events[1]["payload"] == {"type": "assistant", "message": "edit only the canary file"}
    assert (history.directory / "stderr.log").read_text(encoding="utf-8") == "private diagnostic\n"
    assert json.loads((history.directory / "manifest.json").read_text(encoding="utf-8"))["state"] == "started"


def test_run_client_censors_timeout_and_does_not_publish_stderr(tmp_path: Path) -> None:
    history = _history(tmp_path, "timeout")
    metadata = run_client(
        _child("import sys, time; print('private timeout detail', file=sys.stderr, flush=True); time.sleep(30)"),
        os.environ.copy(),
        tmp_path,
        "prompt",
        history,
        timeout=0.1,
    )

    assert metadata["status"] == "censored"
    assert "stderr" not in metadata
    assert (history.directory / "stderr.log").exists()


def test_run_client_marks_malformed_or_missing_result_as_evidence_failure(tmp_path: Path) -> None:
    malformed_history = _history(tmp_path, "malformed")
    malformed = run_client(
        _child("print('not json', flush=True)"),
        os.environ.copy(),
        tmp_path,
        "prompt",
        malformed_history,
    )
    missing_history = _history(tmp_path, "missing")
    missing = run_client(
        _child("import json; print(json.dumps({'type': 'assistant'}), flush=True)"),
        os.environ.copy(),
        tmp_path,
        "prompt",
        missing_history,
    )

    assert malformed["status"] == "evidence_failure"
    assert missing["status"] == "evidence_failure"


def test_stdout_eof_before_exit_does_not_wait_until_deadline(tmp_path):
    import time
    history = _history(tmp_path, 'eof-before-exit')
    started = time.monotonic()
    result = run_client(_child(
        "import os,json,time; print(json.dumps({'type':'result','is_error':False}),flush=True); os.close(1); time.sleep(0.2)"),
        os.environ.copy(), tmp_path, 'prompt', history, timeout=3)
    assert result['status'] == 'completed'
    assert time.monotonic() - started < 2


def test_journal_error_terminates_owned_child_and_reports_evidence_failure(tmp_path, monkeypatch):
    import agent_session_client as client
    history = _history(tmp_path, 'disk-error')
    original_append = history.append
    def fail_event(kind, payload):
        if kind == 'client_event':
            raise OSError('disk full')
        return original_append(kind, payload)
    monkeypatch.setattr(history, 'append', fail_event)
    children = []
    original_start = client._start_process
    def start(*args):
        process = original_start(*args)
        children.append(process)
        return process
    monkeypatch.setattr(client, '_start_process', start)
    try:
        result = run_client(_child("import json,time; print(json.dumps({'type':'assistant'}),flush=True); time.sleep(30)"),
                            os.environ.copy(), tmp_path, 'prompt', history, timeout=2)
        assert result['status'] == 'evidence_failure'
        assert children[0].poll() is not None
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_stderr_capture_failure_invalidates_completed_result(tmp_path, monkeypatch):
    import agent_session_client as client
    history = _history(tmp_path, 'stderr-error')
    def fail_stderr(*args):
        raise OSError('capture volume unavailable')
    monkeypatch.setattr(client, '_write_stderr', fail_stderr)
    result = run_client(_child("import json; print(json.dumps({'type':'result','is_error':False}),flush=True)"),
                        os.environ.copy(), tmp_path, 'prompt', history, timeout=2)
    assert result['status'] == 'evidence_failure'


def test_malformed_stdout_is_retained_as_raw_evidence(tmp_path):
    history = _history(tmp_path, 'raw-malformed')
    result = run_client(_child("print('invalid-json-evidence',flush=True)"),
                        os.environ.copy(), tmp_path, 'prompt', history, timeout=2)
    assert result['status'] == 'evidence_failure'
    assert b'invalid-json-evidence' in (history.directory / 'stdout.jsonl').read_bytes()
    summary = json.loads((history.directory / 'client-run.json').read_text())
    assert summary['status'] == 'evidence_failure'
    assert summary['elapsed_s'] >= 0
    assert summary['returncode'] is not None


def test_low_capture_space_prevents_child_launch(tmp_path, monkeypatch):
    import agent_session_client as client
    from collections import namedtuple
    history = _history(tmp_path, 'low-space')
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr(client.shutil, 'disk_usage', lambda path: usage(1024, 1024, 0))
    launched = []
    monkeypatch.setattr(client, '_start_process', lambda *args: launched.append(True))
    with pytest.raises(OSError):
        run_client(['unused'], {}, tmp_path, 'prompt', history)
    assert not launched


def test_run_client_marks_error_result_failed(tmp_path: Path) -> None:
    history = _history(tmp_path, "error")

    metadata = run_client(
        _child("import json; print(json.dumps({'type': 'result', 'is_error': True}), flush=True)"),
        os.environ.copy(),
        tmp_path,
        "prompt",
        history,
    )

    assert metadata == {"status": "failed", "returncode": 0}
