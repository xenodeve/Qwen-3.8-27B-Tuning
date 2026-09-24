"""Owned local model-process lifecycle with health and port evidence."""
from __future__ import annotations

import datetime as dt
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence


class SessionStartError(RuntimeError):
    """The owned model process did not become a verified healthy server."""


class ModelSession:
    def __init__(self, argv: Sequence[str], cwd: str | Path, port: int,
                 env: Mapping[str, str] | None = None, health_timeout: float = 600,
                 poll_interval: float = 1.0, log_path: str | Path | None = None) -> None:
        if not argv or any(not isinstance(value, str) or not value for value in argv):
            raise ValueError("argv must be a nonempty string sequence")
        if not Path(cwd).is_dir():
            raise ValueError("cwd must be an existing directory")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("port must be a valid TCP port")
        if health_timeout <= 0 or poll_interval <= 0:
            raise ValueError("timeouts must be positive")
        self.argv = list(argv)
        self.cwd = Path(cwd)
        self.port = port
        self.env = dict(env) if env is not None else None
        self.health_timeout = health_timeout
        self.poll_interval = poll_interval
        self.log_path = Path(log_path) if log_path is not None else None
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle = None
        self.started_utc: str | None = None

    def start(self) -> dict[str, object]:
        if self.process is not None:
            raise RuntimeError("model session already started")
        if self._port_occupied():
            raise SessionStartError(f"port {self.port} is occupied")
        options: dict[str, object] = {
            "cwd": str(self.cwd), "env": self.env, "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
        }
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("xb")
            options["stdout"] = self._log_handle
            options["stderr"] = subprocess.STDOUT
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        self.process = subprocess.Popen(self.argv, **options)
        self.started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        deadline = time.monotonic() + self.health_timeout
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise SessionStartError(f"model exited before health: {self.process.returncode}")
                try:
                    health = self._get_json("/health")
                    props = self._get_json("/props")
                    if health.get("status") == "ok":
                        return {"pid": self.process.pid, "port": self.port,
                                "started_utc": self.started_utc, "health": health,
                                "props": props}
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                time.sleep(self.poll_interval)
            raise SessionStartError("model health timeout")
        except Exception:
            self.stop()
            raise

    def stop(self) -> dict[str, object]:
        process = self.process
        if process is None:
            return {"owned_pid": None, "returncode": None}
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               check=False, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("owned model process did not stop") from error
        finally:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
        return {"owned_pid": process.pid, "returncode": process.returncode}

    def _port_occupied(self) -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.2)
            return sock.connect_ex(("127.0.0.1", self.port)) == 0

    def _get_json(self, route: str) -> dict[str, object]:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{self.port}{route}", timeout=2) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise ValueError(f"{route} did not return an object")
        return value
