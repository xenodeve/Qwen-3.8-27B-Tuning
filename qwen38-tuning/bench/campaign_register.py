"""Crash-safe, single-process campaign attempt register."""
import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path


_STATUSES = {"planned", "running", "verified", "failed", "invalid", "censored", "not_run"}
_TERMINAL = _STATUSES - {"planned", "running"}
_REASON_REQUIRED = {"failed", "invalid", "censored", "not_run"}
_REQUIRED_ATTEMPT_FIELDS = {"attempt_id", "model", "case_id", "round", "config_id"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


class CampaignRegister:
    """A thread-safe register; callers provide any inter-process locking."""

    def __init__(self, path, specification):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._attempt_ids = self._validate_specification(specification)
        self._specification = specification
        self._specification_sha256 = _digest(specification)
        self.path.mkdir()
        self._write_json(self.path / "specification.json", specification, replace=False)
        self._progress = {
            "specification_sha256": self._specification_sha256,
            "attempts": {attempt_id: {"status": "planned"} for attempt_id in self._attempt_ids},
        }
        self._write_progress(self._progress)

    @classmethod
    def load(cls, path):
        register = cls.__new__(cls)
        register.path = Path(path)
        register._lock = threading.RLock()
        register._specification = register._read_json(register.path / "specification.json", "specification")
        register._specification_sha256 = _digest(register._specification)
        register._progress = register._read_json(register.path / "progress.json", "progress")
        if not isinstance(register._progress, Mapping) or (
            register._progress.get("specification_sha256") != register._specification_sha256
        ):
            raise ValueError("specification digest mismatch")
        register._attempt_ids = register._validate_specification(register._specification)
        register._validate_progress(register._progress)
        return register

    def mark(self, attempt_id, status, evidence=None, reason=None):
        with self._lock:
            self._check_specification()
            self._validate_progress(self._progress)
            if attempt_id not in self._progress["attempts"]:
                raise ValueError("unknown attempt_id")
            if status not in _STATUSES:
                raise ValueError("unknown status")
            current = self._progress["attempts"][attempt_id]["status"]
            if current == "planned" and status not in {"running", "not_run"}:
                raise ValueError("planned attempts may only become running or not_run")
            if current == "running" and status not in _TERMINAL:
                raise ValueError("running attempts may only become terminal")
            if current in _TERMINAL:
                raise ValueError("terminal attempts cannot be overwritten")
            if status in _REASON_REQUIRED and not isinstance(reason, str):
                raise ValueError("terminal status requires a nonempty reason")
            if status in _REASON_REQUIRED and not reason.strip():
                raise ValueError("terminal status requires a nonempty reason")
            if status == "verified":
                if not isinstance(evidence, Mapping):
                    raise ValueError("verified status requires evidence mapping")
                if not all(isinstance(evidence.get(key), str) and evidence[key].strip()
                           for key in ("session_id", "verifier_evidence_ref")):
                    raise ValueError("verified evidence needs session_id and verifier_evidence_ref")
            candidate = json.loads(_canonical(self._progress))
            entry = candidate["attempts"][attempt_id]
            entry["status"] = status
            if evidence is not None:
                entry["evidence"] = evidence
            if reason is not None:
                entry["reason"] = reason
            self._write_progress(candidate)
            self._progress = candidate

    def summary(self):
        with self._lock:
            self._check_specification()
            self._validate_progress(self._progress)
            attempts = self._progress["attempts"]
            counts = {status: sum(entry["status"] == status for entry in attempts.values())
                      for status in sorted(_STATUSES)}
            missing = [attempt_id for attempt_id in self._attempt_ids
                       if attempts[attempt_id]["status"] in {"planned", "running"}]
            return {
                "expected": len(self._attempt_ids),
                "terminal": sum(counts[status] for status in _TERMINAL),
                "counts": counts,
                "missing_attempt_ids": missing,
                "complete": not missing,
                "failed_attempt_ids": [key for key in self._attempt_ids if attempts[key]["status"] == "failed"],
                "censored_attempt_ids": [key for key in self._attempt_ids if attempts[key]["status"] == "censored"],
            }

    @staticmethod
    def _validate_specification(specification):
        if not isinstance(specification, Mapping) or not isinstance(specification.get("attempts"), list):
            raise ValueError("specification must contain attempts list")
        ids = []
        for attempt in specification["attempts"]:
            if not isinstance(attempt, Mapping) or not _REQUIRED_ATTEMPT_FIELDS <= set(attempt):
                raise ValueError("each attempt needs required fields")
            attempt_id = attempt["attempt_id"]
            if not isinstance(attempt_id, str) or not attempt_id.strip() or attempt_id in ids:
                raise ValueError("attempt_id must be unique and nonempty")
            ids.append(attempt_id)
        try:
            _canonical(specification)
        except (TypeError, ValueError) as error:
            raise ValueError("specification must be JSON serializable") from error
        return ids

    def _validate_progress(self, progress):
        if not isinstance(progress, Mapping) or set(progress) != {"specification_sha256", "attempts"}:
            raise ValueError("corrupt progress")
        if progress["specification_sha256"] != self._specification_sha256:
            raise ValueError("specification digest mismatch")
        attempts = progress["attempts"]
        if not isinstance(attempts, Mapping) or set(attempts) != set(self._attempt_ids):
            raise ValueError("progress attempt IDs do not match specification")
        for entry in attempts.values():
            if not isinstance(entry, Mapping) or entry.get("status") not in _STATUSES:
                raise ValueError("corrupt progress attempt")
            status = entry["status"]
            if status in _REASON_REQUIRED and not isinstance(entry.get("reason"), str):
                raise ValueError("corrupt progress reason")
            if status == "verified":
                evidence = entry.get("evidence")
                if not isinstance(evidence, Mapping) or not all(
                    isinstance(evidence.get(key), str) and evidence[key].strip()
                    for key in ("session_id", "verifier_evidence_ref")
                ):
                    raise ValueError("corrupt verified evidence")

    def _check_specification(self):
        current = self._read_json(self.path / "specification.json", "specification")
        if _digest(current) != self._specification_sha256:
            raise ValueError("specification digest mismatch")

    @staticmethod
    def _read_json(path, label):
        try:
            with Path(path).open(encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"corrupt {label}") from error

    def _write_progress(self, progress):
        self._write_json(self.path / "progress.json", progress, replace=True)

    @staticmethod
    def _write_json(path, value, replace):
        data = _canonical(value)
        if not replace:
            with Path(path).open("xb") as handle:
                handle.write(data)
            return
        descriptor, temporary = tempfile.mkstemp(dir=Path(path).parent, prefix=".progress-", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
