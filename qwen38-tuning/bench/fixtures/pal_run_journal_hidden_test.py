"""Parent-only hidden oracle for PAL #149 run journal.

This file is mounted read-only by the parent verifier and is never visible to
an evaluated client. It imports only the candidate module from the submitted
clone and the existing RecordStore implementation. The journal must not use
JSONL or O_APPEND, and tests must not write ~/.openclink/store.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.fixture
def workspace():
    return Path('/workspace')
def _journal(workspace: Path, tmp_path: Path):
    module_path = workspace / "tools" / "run_journal.py"
    spec = importlib.util.spec_from_file_location("candidate_run_journal", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"candidate module missing: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from utils.record_store import RecordStore
    return module.RunJournal(RecordStore(tmp_path / "store")), RecordStore, module


def test_single_return_is_a_self_describing_atomic_record(workspace, tmp_path):
    journal, _, _ = _journal(workspace, tmp_path)
    result = {"status": "ok", "unicode": "ภาษาไทย🚀"}
    original = copy.deepcopy(result)
    identity = journal.append("run-a", "agent-a", result)
    assert result == original
    assert journal.read("run-a") == [{"run_id": "run-a", "agent_id": "agent-a", "result": original}]
    assert identity.startswith("run-5-run-a-7-agent-a")
    files = list((tmp_path / "store").iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".json"
    assert not list((tmp_path / "store").glob("*.jsonl"))


def test_multiple_returns_are_sorted_and_restartable(workspace, tmp_path):
    journal, RecordStore, _ = _journal(workspace, tmp_path)
    journal.append("run-a", "agent-z", {"n": 2})
    journal.append("run-a", "agent-a", {"n": 1})
    journal.append("run-b", "agent-a", {"n": 3})
    restarted = journal.__class__(RecordStore(tmp_path / "store"))
    assert [row["agent_id"] for row in restarted.read("run-a")] == ["agent-a", "agent-z"]
    assert restarted.read("missing") == []
    assert [row["agent_id"] for row in restarted.read("run-b")] == ["agent-a"]


def test_run_prefixes_and_compound_delimiters_do_not_collide(workspace, tmp_path):
    journal, _, _ = _journal(workspace, tmp_path)
    first = journal.append("run", "a-b", {"value": 1})
    second = journal.append("run-a", "b", {"value": 2})
    assert first != second
    assert journal.read("run") == [{"run_id": "run", "agent_id": "a-b", "result": {"value": 1}}]
    assert journal.read("run-a") == [{"run_id": "run-a", "agent_id": "b", "result": {"value": 2}}]


def test_duplicate_identity_is_rejected(workspace, tmp_path):
    journal, _, _ = _journal(workspace, tmp_path)
    journal.append("run-a", "agent-a", {"n": 1})
    with pytest.raises(ValueError):
        journal.append("run-a", "agent-a", {"n": 2})


def test_invalid_components_are_rejected(workspace, tmp_path):
    journal, _, _ = _journal(workspace, tmp_path)
    for run_id, agent_id in (("", "agent"), ("RUN", "agent"), ("run", "Agent"),
                             ("run/a", "agent"), ("run", "agent\\x"), ("run\n", "agent"),
                             ("run", "agent:"), ("a" * 201, "agent")):
        with pytest.raises(ValueError):
            journal.append(run_id, agent_id, {})


def test_corruption_is_loud_for_matching_run_but_isolated_between_runs(workspace, tmp_path):
    journal, RecordStore, _ = _journal(workspace, tmp_path)
    matching = journal.append("run-a", "agent-a", {"n": 1})
    unrelated = journal.append("run-b", "agent-a", {"n": 2})
    store = RecordStore(tmp_path / "store")
    store.path_for(unrelated).write_bytes(b"not-json")
    assert journal.read("run-a") == [{"run_id": "run-a", "agent_id": "agent-a", "result": {"n": 1}}]
    from utils.record_store import StoreCorruptError
    with pytest.raises(StoreCorruptError):
        journal.read("run-b")
    assert matching != unrelated


def test_empty_and_invalid_utf8_matching_records_are_corrupt(workspace, tmp_path):
    journal, RecordStore, _ = _journal(workspace, tmp_path)
    identity = journal.append("run-a", "agent-a", {"n": 1})
    store = RecordStore(tmp_path / "store")
    store.path_for(identity).write_bytes(b"\x80")
    from utils.record_store import StoreCorruptError
    with pytest.raises(StoreCorruptError):
        journal.read("run-a")


def test_concurrent_unicode_returns_are_all_complete(workspace, tmp_path):
    journal, _, _ = _journal(workspace, tmp_path)
    payload = {"text": "ภาษาไทย🚀" * 20000}
    def write(index):
        return journal.append("run-concurrent", f"agent-{index:02d}", {**payload, "n": index})
    with ThreadPoolExecutor(max_workers=8) as pool:
        identities = list(pool.map(write, range(16)))
    assert len(set(identities)) == 16
    records = journal.read("run-concurrent")
    assert len(records) == 16
    assert {row["result"]["n"] for row in records} == set(range(16))


def test_candidate_uses_the_existing_store_not_jsonl(workspace, tmp_path):
    journal, _, _ = _journal(workspace, tmp_path)
    journal.append("run-a", "agent-a", {"n": 1})
    store = tmp_path / "store"
    assert all(path.suffix == ".json" for path in store.iterdir())
    assert not (workspace / "run.jsonl").exists()
