"""Tests for session telemetry's deliberately non-invasive collection boundary."""
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import session_telemetry


CSV = "GPU-abc, RTX 5060 Ti, 16384, 4096, 12288, 42, 2500, 61, 123.45, 576.02\n"


def test_parse_gpu_csv_converts_nvidia_values_and_marks_na_unknown():
    rows = session_telemetry.parse_gpu_csv(
        "GPU-abc, RTX 5060 Ti, N/A, 4096, N/A, N/A, 2500, 61, N/A, 576.02\n"
    )
    assert rows == [{
        "uuid": "GPU-abc", "name": "RTX 5060 Ti", "memory_total_mib": None,
        "memory_used_mib": 4096, "memory_free_mib": None, "utilization_gpu_pct": None,
        "clocks_sm_mhz": 2500, "temperature_gpu_c": 61, "power_draw_w": None,
        "driver_version": "576.02",
        "unknown": ["memory_total_mib", "memory_free_mib", "utilization_gpu_pct", "power_draw_w"],
    }]


@pytest.mark.parametrize("text", [
    "GPU-abc, RTX, 1, 2\n",
    "GPU-abc, RTX, not-a-number, 2, 3, 4, 5, 6, 7, 576.02\n",
    "GPU-abc, RTX, 1, 2, 3, 4, 5, 6, 7, 576.02, surplus\n",
])
def test_parse_gpu_csv_rejects_malformed_rows(text):
    with pytest.raises(ValueError):
        session_telemetry.parse_gpu_csv(text)


def test_sample_resources_marks_failed_gpu_query_unknown(monkeypatch):
    monkeypatch.setattr(session_telemetry.gpu_device, "visible_uuids", lambda: ["GPU-abc"])
    monkeypatch.setattr(
        session_telemetry.gpu_device, "query",
        lambda *args: (_ for _ in ()).throw(OSError("nvidia-smi unavailable")),
    )
    monkeypatch.setattr(session_telemetry, "psutil", None)
    sample = session_telemetry.sample_resources()
    assert sample["gpus"] is None
    assert sample["gpu_unknown"] is True
    assert sample["unknown_gpu_uuids"] == ["GPU-abc"]
    assert sample["ram_total_bytes"] is None
    assert sample["ram_available_bytes"] is None
    assert any("GPU-abc" in error for error in sample["errors"])


def test_sample_resources_queries_every_visible_gpu_by_uuid(monkeypatch):
    calls = []

    def query(fields, uuid):
        calls.append((fields, uuid))
        return [uuid] + CSV.strip().split(", ")[1:]

    monkeypatch.setattr(session_telemetry.gpu_device, "visible_uuids", lambda: ["GPU-abc", "GPU-def"])
    monkeypatch.setattr(session_telemetry.gpu_device, "query", query)
    monkeypatch.setattr(session_telemetry, "psutil", None)
    sample = session_telemetry.sample_resources()
    assert [uuid for _, uuid in calls] == ["GPU-abc", "GPU-def"]
    assert all(fields == session_telemetry.GPU_FIELDS for fields, _ in calls)
    assert [gpu["uuid"] for gpu in sample["gpus"]] == ["GPU-abc", "GPU-def"]


def test_sample_resources_collects_only_safe_psutil_process_fields(monkeypatch):
    class VirtualMemory:
        total = 48
        available = 24

    class Process:
        pid = 123

        def name(self):
            return "python.exe"

    class Psutil:
        @staticmethod
        def virtual_memory():
            return VirtualMemory()

        @staticmethod
        def Process():
            return Process()

        @staticmethod
        def process_iter(attrs):
            assert attrs == ["name"]
            return [type("Game", (), {"info": {"name": "javaw.exe"}})()]

    monkeypatch.setattr(session_telemetry, "psutil", Psutil())
    monkeypatch.setattr(session_telemetry.gpu_device, "visible_uuids", lambda: ["GPU-abc"])
    monkeypatch.setattr(session_telemetry.gpu_device, "query", lambda *args: CSV.strip().split(", "))
    sample = session_telemetry.sample_resources()
    assert sample["ram_total_bytes"] == 48
    assert sample["ram_available_bytes"] == 24
    assert sample["process"] == {"pid": 123, "name": "python.exe"}
    assert sample["suspected_game_processes"] == ["javaw.exe"]
    assert sample["gpus"][0]["memory_used_mib"] == 4096


def test_sampler_starts_immediately_records_duration_and_gpu_peak():
    called = threading.Event()

    def fake_sample():
        called.set()
        return {"gpus": [{"uuid": "GPU-abc", "memory_used_mib": 4096}], "errors": []}

    sampler = session_telemetry.TelemetrySampler(interval_s=60, sample=fake_sample)
    sampler.start()
    assert called.wait(1), "start() must collect the first sample immediately"
    result = sampler.stop()
    assert result["samples"]
    assert result["samples"][0]["collection_s"] >= 0
    assert result["peak_gpu_memory_used_mib"] == {"GPU-abc": 4096}
    assert result["errors"] == []


def test_gpu_device_smi_passes_two_second_timeout(monkeypatch):
    seen = {}

    def run(*args, **kwargs):
        seen.update(kwargs)
        return type("Result", (), {"stdout": "", "returncode": 0})()

    monkeypatch.setattr(session_telemetry.gpu_device.subprocess, "run", run)
    session_telemetry.gpu_device._smi(["--help"])
    assert seen["timeout"] == 2


def test_gpu_device_query_propagates_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(session_telemetry.gpu_device.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        session_telemetry.gpu_device.query(["uuid"], "GPU-abc")


def test_gpu_timeout_is_recorded_as_unknown(monkeypatch):
    monkeypatch.setattr(session_telemetry.gpu_device, "visible_uuids", lambda: ["GPU-abc"])
    monkeypatch.setattr(
        session_telemetry.gpu_device, "query",
        lambda *args: (_ for _ in ()).throw(subprocess.TimeoutExpired("nvidia-smi", 2)),
    )
    monkeypatch.setattr(session_telemetry, "psutil", None)
    sample = session_telemetry.sample_resources()
    assert sample["gpus"] is None
    assert sample["unknown_gpu_uuids"] == ["GPU-abc"]
    assert any("TimeoutExpired" in error for error in sample["errors"])


def test_explicit_device_uuids_override_visible_devices(monkeypatch):
    calls = []
    monkeypatch.setattr(
        session_telemetry.gpu_device, "visible_uuids",
        lambda: (_ for _ in ()).throw(AssertionError("must not read parent CUDA_VISIBLE_DEVICES")),
    )
    monkeypatch.setattr(
        session_telemetry.gpu_device, "query",
        lambda fields, uuid: calls.append(uuid) or [uuid] + CSV.strip().split(", ")[1:],
    )
    monkeypatch.setattr(session_telemetry, "psutil", None)
    sample = session_telemetry.sample_resources(["GPU-first", "GPU-second"])
    assert calls == ["GPU-first", "GPU-second"]
    assert [row["uuid"] for row in sample["gpus"]] == calls


@pytest.mark.parametrize("bad", ["", "not-a-uuid", " GPU-abc", "GPU-abc "])
def test_invalid_explicit_device_uuid_is_unknown_not_queried(monkeypatch, bad):
    monkeypatch.setattr(
        session_telemetry.gpu_device, "query",
        lambda *args: (_ for _ in ()).throw(AssertionError("invalid UUID must not query driver")),
    )
    monkeypatch.setattr(session_telemetry, "psutil", None)
    sample = session_telemetry.sample_resources([bad])
    assert sample["gpus"] is None
    assert sample["gpu_unknown"] is True
    assert sample["unknown_gpu_uuids"] == [bad]
    assert any("invalid explicit GPU UUID" in error for error in sample["errors"])


def test_sampler_persists_sample_before_stop(tmp_path):
    called = threading.Event()

    def fake_sample():
        called.set()
        return {"gpus": [], "errors": []}

    sink = tmp_path / "resource-live.jsonl"
    sampler = session_telemetry.TelemetrySampler(interval_s=60, sample=fake_sample, sink_path=sink).start()
    try:
        assert called.wait(1)
        assert sampler.wait_ready(1)['errors'] == []
        assert sink.read_text(encoding="utf-8").count("\n") == 1
    finally:
        sampler.stop()


def test_sampler_rejects_existing_sink(tmp_path):
    sink = tmp_path / "resource-live.jsonl"
    sink.write_text("old\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        session_telemetry.TelemetrySampler(sink_path=sink).start()


def test_sampler_reports_sink_write_failure_and_stops(monkeypatch, tmp_path):
    failed = threading.Event()

    def fail_fsync(_):
        failed.set()
        raise OSError("disk full")

    monkeypatch.setattr(session_telemetry.os, "fsync", fail_fsync)
    sampler = session_telemetry.TelemetrySampler(
        interval_s=60, sample=lambda: {"gpus": [], "errors": []}, sink_path=tmp_path / "live.jsonl"
    ).start()
    assert failed.wait(1)
    result = sampler.stop()
    assert result["samples"] == []
    assert any("sink" in error and "disk full" in error for error in result["errors"])
