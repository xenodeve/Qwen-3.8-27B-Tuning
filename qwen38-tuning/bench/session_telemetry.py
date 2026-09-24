"""Small, non-invasive resource snapshots for benchmark sessions."""
import csv
import datetime as dt
import json
import os
from pathlib import Path
import threading
import time
import gpu_device

try:
    import psutil
except ImportError:  # Optional: telemetry must not add a dependency.
    psutil = None
GPU_FIELDS = (
    "uuid", "name", "memory.total", "memory.used", "memory.free",
    "utilization.gpu", "clocks.sm", "temperature.gpu", "power.draw", "driver_version",
)
_FIELDS = (
    ("uuid", str), ("name", str), ("memory_total_mib", int),
    ("memory_used_mib", int), ("memory_free_mib", int),
    ("utilization_gpu_pct", float), ("clocks_sm_mhz", float),
    ("temperature_gpu_c", float), ("power_draw_w", float), ("driver_version", str),
)
def parse_gpu_csv(text):
    """Parse the exact NVIDIA field order requested through ``gpu_device.query``."""
    rows = []
    try:
        parsed = csv.reader(text.splitlines())
        for line_number, row in enumerate(parsed, 1):
            if not row:
                continue
            if len(row) != len(_FIELDS):
                raise ValueError(f"nvidia-smi row {line_number} has {len(row)} fields, expected {len(_FIELDS)}")
            item, unknown = {}, []
            for (key, converter), value in zip(_FIELDS, row):
                value = value.strip()
                if value == "N/A":
                    item[key] = None
                    unknown.append(key)
                    continue
                if not value:
                    raise ValueError(f"nvidia-smi row {line_number} has empty {key}")
                try:
                    item[key] = converter(value)
                except ValueError as exc:
                    raise ValueError(f"nvidia-smi row {line_number} has invalid {key}: {value!r}") from exc
            item["unknown"] = unknown
            rows.append(item)
    except csv.Error as exc:
        raise ValueError(f"malformed nvidia-smi CSV: {exc}") from exc
    return rows

def _error(errors, label, exc):
    errors.append(f"{label}: {type(exc).__name__}: {exc}")


def _safe_process_info(errors):
    if psutil is None:
        return None, None, None, []
    total = available = process = None
    games = []
    try:
        memory = psutil.virtual_memory()
        total, available = memory.total, memory.available
    except Exception as exc:  # psutil can fail while the host is shutting down.
        _error(errors, "psutil virtual_memory", exc)
    try:
        current = psutil.Process()
        process = {"pid": current.pid, "name": current.name()}
    except Exception as exc:
        _error(errors, "psutil process", exc)
    try:
        for candidate in psutil.process_iter(["name"]):
            name = (candidate.info.get("name") or "").strip()
            stem = name.lower().removesuffix(".exe")
            if stem in {"minecraft", "java", "javaw"}:
                games.append(name)
    except Exception as exc:
        _error(errors, "psutil process_iter", exc)
    return total, available, process, games


def _sample_gpus(errors, device_uuids=None):
    """Read pinned GPUs; a driver timeout is recorded as unknown, never zero."""
    if device_uuids is not None:
        if not isinstance(device_uuids, (list, tuple)):
            errors.append("invalid explicit GPU UUID list")
            return None, True, []
        uuids = list(device_uuids)
        invalid = [uuid for uuid in uuids if not isinstance(uuid, str) or len(uuid) == 4 or not uuid.startswith("GPU-") or uuid != uuid.strip()]
        if invalid:
            errors.append(f"invalid explicit GPU UUID(s): {invalid!r}")
            return None, True, invalid
    else:
        try:
            uuids = gpu_device.visible_uuids()
        except Exception as exc:
            _error(errors, "gpu UUID discovery", exc)
            return None, True, []
    rows, unknown = [], []
    for uuid in uuids:
        try:
            values = gpu_device.query(GPU_FIELDS, uuid)
            row = parse_gpu_csv(",".join(values))[0]
            if row["uuid"] != uuid:
                raise ValueError(f"driver returned {row['uuid']!r} for pinned {uuid!r}")
            rows.append(row)
        except Exception as exc:
            _error(errors, f"GPU {uuid}", exc)
            unknown.append(uuid)
    return rows or None, bool(unknown), unknown

def sample_resources(device_uuids=None):
    """Take one snapshot without reading command lines, environment, or credentials."""
    errors = []
    snapshot = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "monotonic_s": time.monotonic(),
        "gpus": None,
        "gpu_unknown": True,
        "unknown_gpu_uuids": [],
        "ram_total_bytes": None,
        "ram_available_bytes": None,
        "process": None,
        "suspected_game_processes": [],
        "errors": errors,
    }
    gpus, gpu_unknown, unknown_uuids = _sample_gpus(errors, device_uuids)
    snapshot.update({
        "gpus": gpus, "gpu_unknown": gpu_unknown, "unknown_gpu_uuids": unknown_uuids,
    })
    total, available, process, games = _safe_process_info(errors)
    snapshot.update({
        "ram_total_bytes": total, "ram_available_bytes": available,
        "process": process, "suspected_game_processes": games,
    })
    return snapshot


class TelemetrySampler:
    """Bounded sampler that optionally persists snapshots before retaining them."""

    def __init__(self, interval_s=2, sample=sample_resources, sink_path=None):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.interval_s, self.sample = interval_s, sample
        self._sink_path = Path(sink_path) if sink_path is not None else None
        self._sink = None
        self._samples, self._errors = [], []
        self._lock, self._stop, self._thread = threading.Lock(), threading.Event(), None
        self._ready = threading.Event()

    def start(self):
        if self._thread is not None:
            raise RuntimeError("TelemetrySampler already started")
        if self._sink_path is not None:
            fd = os.open(self._sink_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            self._sink = os.fdopen(fd, "w", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="session-telemetry", daemon=True)
        self._thread.start()
        return self

    def wait_ready(self, timeout=7):
        if not self._ready.wait(timeout):
            raise TimeoutError('initial resource snapshot was not persisted')
        with self._lock:
            return dict(self._samples[0])

    def _write_sink(self, snapshot):
        self._sink.write(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._sink.flush()
        os.fsync(self._sink.fileno())

    def _run(self):
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    snapshot = self.sample()
                    if not isinstance(snapshot, dict):
                        raise TypeError("sample must return a dict")
                    snapshot["collection_s"] = time.monotonic() - started
                    errors = snapshot.get("errors", [])
                except Exception as exc:
                    snapshot = {"collection_s": time.monotonic() - started, "errors": []}
                    errors = [f"sampler sample: {type(exc).__name__}: {exc}"]
                    snapshot["errors"] = errors
                with self._lock:
                    if len(self._samples) >= 10000:
                        self._errors.append("sampler: maximum 10000 samples reached")
                        return
                try:
                    if self._sink is not None:
                        self._write_sink(snapshot)
                except Exception as exc:
                    with self._lock:
                        self._errors.append(f"sampler sink: {type(exc).__name__}: {exc}")
                    return
                with self._lock:
                    self._samples.append(snapshot)
                    self._errors.extend(errors)
                self._ready.set()
                deadline += self.interval_s
                self._stop.wait(max(0, deadline - time.monotonic()))
        finally:
            if self._sink is not None:
                self._sink.close()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(7)
            if self._thread.is_alive():
                raise RuntimeError("TelemetrySampler thread did not stop within 7 seconds")
        with self._lock:
            samples, errors = list(self._samples), list(self._errors)
        peaks = {}
        for snapshot in samples:
            for gpu in snapshot.get("gpus") or []:
                uuid, used = gpu.get("uuid"), gpu.get("memory_used_mib")
                if uuid and used is not None:
                    peaks[uuid] = max(peaks.get(uuid, used), used)
        return {"samples": samples, "collection_s": sum(s.get("collection_s", 0) for s in samples),
                "peak_gpu_memory_used_mib": peaks, "errors": errors}
