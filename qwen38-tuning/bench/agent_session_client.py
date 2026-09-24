"""Restricted local Claude CLI invocation with append-only evidence capture."""

from __future__ import annotations

import json
import os
import queue
import signal
import shutil
import tempfile
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from session_history import SessionHistory


_ALLOWED_TOOLS = "Read,Write,Edit,Glob,Grep"
_EMPTY_MCP_CONFIG = '{"mcpServers":{}}'
_CLOUD_ENV_PREFIXES = (
    "ANTHROPIC_",
    "CLAUDE_",
    "AWS_",
    "AZURE_",
    "BEDROCK_",
    "GCP_",
    "GOOGLE_",
    "VERTEX_",
    "VERTEXAI_",
)
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def build_client_argv(
    executable: str | Path, model: str, session_id: str, settings_path: str | Path,
    *, mcp_config: str | Path = _EMPTY_MCP_CONFIG, allowed_tools: str = _ALLOWED_TOOLS,
) -> list[str]:
    """Build the fixed least-privilege argv for one local benchmark client."""
    return [
        str(executable),
        "--bare",
        "--restricted",
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp_config),
        "--settings",
        str(settings_path),
        "--permission-mode",
        "dontAsk",
        "--tools",
        allowed_tools,
        "--allowedTools",
        allowed_tools,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--session-id",
        session_id,
        "--model",
        model,
    ]


def build_client_env(
    base_env: Mapping[str, str], config_dir: str | Path, endpoint: str
) -> dict[str, str]:
    """Create an isolated environment which can route only to local inference."""
    _validate_loopback_endpoint(endpoint)
    env = {
        key: value
        for key, value in base_env.items()
        if not key.startswith(_CLOUD_ENV_PREFIXES) and key != "CLAUDECODE"
    }
    env.update(
        {
            "ANTHROPIC_BASE_URL": endpoint,
            "ANTHROPIC_API_KEY": "local-benchmark-placeholder",
            "CLAUDE_CONFIG_DIR": str(config_dir),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    )
    return env


def run_client(
    argv: Sequence[str],
    env: Mapping[str, str],
    workdir: str | Path,
    prompt: str,
    history: SessionHistory,
    timeout: float = 120,
) -> dict[str, object]:
    """Run one owned client process and journal public stream evidence only."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    started = time.monotonic()
    if shutil.disk_usage(history.directory).free < 1024 ** 3:
        raise OSError("less than 1 GiB available for session evidence")
    with tempfile.TemporaryFile(dir=history.directory) as probe:
        probe.write(b"capture-preflight")
        probe.flush()
        os.fsync(probe.fileno())
        probe.seek(0)
        if probe.read() != b"capture-preflight":
            raise OSError("capture storage round-trip failed")
    for name in ("stdout.jsonl", "stderr.log", "client-run.json", "client-run.json.tmp"):
        if (history.directory / name).exists():
            raise FileExistsError(f"refusing to overwrite session evidence: {name}")
    history.append("user_prompt", prompt)
    stderr_path = history.directory / "stderr.log"
    capture_errors: list[str] = []
    process = _start_process(argv, env, workdir)
    prompt_thread = threading.Thread(target=_write_prompt, args=(process, prompt), daemon=True)
    prompt_thread.start()
    stdout_events: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=64)
    stdout_thread = threading.Thread(
        target=_read_stdout, args=(process, stdout_events, history.directory / "stdout.jsonl"), daemon=True
    )
    def capture_stderr():
        try:
            _write_stderr(process, stderr_path)
        except Exception as error:
            capture_errors.append(type(error).__name__)
    stderr_thread = threading.Thread(target=capture_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    deadline = time.monotonic() + timeout
    stdout_closed = False
    result_is_error: bool | None = None
    evidence_failure = False
    timed_out = False

    while not stdout_closed or process.poll() is None:
        if capture_errors:
            evidence_failure = True
            _terminate_owned_process_tree(process)
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            _terminate_owned_process_tree(process)
            break
        try:
            kind, payload = stdout_events.get(timeout=min(remaining, 0.1))
        except queue.Empty:
            continue
        if kind == "closed":
            stdout_closed = True
            continue
        if kind == "reader_error":
            evidence_failure = True
            _terminate_owned_process_tree(process)
            break

        try:
            event = json.loads(bytes(payload).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            evidence_failure = True
            _terminate_owned_process_tree(process)
            break
        if not isinstance(event, dict):
            evidence_failure = True
            _terminate_owned_process_tree(process)
            break
        try:
            history.append("client_event", event)
        except Exception as error:
            capture_errors.append(type(error).__name__)
            evidence_failure = True
            _terminate_owned_process_tree(process)
            break
        if event.get("type") == "result":
            if not isinstance(event.get("is_error"), bool) or result_is_error is not None:
                evidence_failure = True
                _terminate_owned_process_tree(process)
                break
            result_is_error = event["is_error"]

    if process.poll() is None:
        _terminate_owned_process_tree(process)
    returncode = process.wait()
    while stdout_thread.is_alive():
        try:
            stdout_events.get(timeout=0.1)
        except queue.Empty:
            pass
    stdout_thread.join()
    stderr_thread.join()
    prompt_thread.join()

    if evidence_failure or capture_errors:
        status = "evidence_failure"
    elif timed_out:
        status = "censored"
    elif result_is_error is None:
        status = "evidence_failure"
    elif result_is_error or returncode != 0:
        status = "failed"
    else:
        status = "completed"
    result = {"status": status, "returncode": returncode}
    summary = dict(result, elapsed_s=time.monotonic() - started,
                   session_id=history.session_id, capture_errors=capture_errors,
                   timed_out=timed_out)
    temporary = history.directory / "client-run.json.tmp"
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, history.directory / "client-run.json")
    return result


def _validate_loopback_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("endpoint must have a valid explicit port") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname not in _LOOPBACK_HOSTS
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("endpoint must be an explicit loopback http URL")


def _start_process(
    argv: Sequence[str], env: Mapping[str, str], workdir: str | Path
) -> subprocess.Popen[bytes]:
    options: dict[str, Any] = {
        "cwd": str(workdir),
        "env": dict(env),
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    return subprocess.Popen(list(argv), **options)


def _write_prompt(process: subprocess.Popen[bytes], prompt: str) -> None:
    assert process.stdin is not None
    try:
        process.stdin.write(prompt.encode("utf-8"))
        process.stdin.close()
    except OSError:
        pass


def _read_stdout(
    process: subprocess.Popen[bytes], events: queue.Queue[tuple[str, object]], path: Path
) -> None:
    assert process.stdout is not None
    try:
        with path.open("xb") as raw:
            while line := process.stdout.readline():
                raw.write(line)
                raw.flush()
                events.put(("line", line))
            os.fsync(raw.fileno())
    except BaseException as error:
        events.put(("reader_error", error))
    finally:
        events.put(("closed", None))


def _write_stderr(process: subprocess.Popen[bytes], path: Path) -> None:
    assert process.stderr is not None
    with path.open("wb") as handle:
        while chunk := process.stderr.read(65536):
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())


def _terminate_owned_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    os.killpg(process.pid, signal.SIGKILL)
