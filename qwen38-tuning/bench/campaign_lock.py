"""Cross-process campaign lease; never deletes stale lock files."""
from __future__ import annotations

import json
import os
import socket
import threading
from datetime import UTC, datetime
from pathlib import Path


class LockUnavailable(RuntimeError):
    """Another owner holds the campaign lease."""


class CampaignLock:
    def __init__(self, path: str | Path, campaign_id: str, profile_id: str) -> None:
        if not isinstance(campaign_id, str) or not campaign_id.strip():
            raise ValueError("campaign_id and profile_id must be nonempty")
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("campaign_id and profile_id must be nonempty")
        self.path = Path(path)
        self.owner_path = self.path.with_suffix(self.path.suffix + ".owner.json")
        self.owner = {"pid": os.getpid(), "hostname": socket.gethostname(),
                      "campaign_id": campaign_id, "profile_id": profile_id,
                      "started_utc": datetime.now(UTC).isoformat()}
        self._handle = None
        self._thread_lock = threading.Lock()

    def acquire(self) -> "CampaignLock":
        with self._thread_lock:
            if self._handle is not None:
                raise RuntimeError("campaign lock already acquired")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, IOError) as error:
                handle.close()
                owner = self.read_owner(self.path)
                raise LockUnavailable(owner or "existing campaign lock") from error
            handle.seek(0)
            handle.truncate()
            handle.write((json.dumps(self.owner, ensure_ascii=False) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
            self.owner_path.write_text(json.dumps(self.owner, ensure_ascii=False) + "\n", encoding="utf-8")
            self._handle = handle
            return self

    def release(self) -> None:
        with self._thread_lock:
            if self._handle is None:
                return
            handle, self._handle = self._handle, None
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
            try:
                owner = self.read_owner(self.owner_path)
                if owner and owner.get("pid") == self.owner["pid"] and owner.get("started_utc") == self.owner["started_utc"]:
                    self.owner_path.unlink(missing_ok=True)
            except OSError:
                pass

    def __enter__(self) -> "CampaignLock":
        return self.acquire()

    def __exit__(self, *_args: object) -> None:
        self.release()

    @staticmethod
    def read_owner(path: str | Path) -> dict[str, object] | None:
        path = Path(path)
        candidates = [path] if path.name.endswith('.owner.json') else [path.with_suffix(path.suffix + '.owner.json'), path]
        for candidate in candidates:
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            return value if isinstance(value, dict) else None
        return None
