"""Offline policy/contract tests for the second PAL task runner."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "tools" / "run-pal-journal.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("pal_journal_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_task_has_exact_scope_and_candidate_order():
    runner = load_runner()
    policy = runner.task_policy()
    assert policy["allowed_files"] == [
        "tools/run_journal.py", "tests/test_run_journal.py"
    ]
    assert policy["candidates"] == [
        "swift_q4km", "turbo_mtp_q4km", "swift", "turbo",
        "dirk", "gsq", "nvfp4", "exl3"
    ]
    assert policy["baseline_sha"] == "200fcb9262d25e4002e30bf00c4364f55a42354e"
    assert policy["context"] == 65536
    assert policy["output"] == 8192
    assert policy["max_turns"] == 64
    assert policy["timeout_s"] == 1800


def test_duplicate_identity_policy_is_explicit():
    runner = load_runner()
    assert runner.task_policy()["duplicate_identity"] == "reject"
    assert runner.compound_identity("run-a", "agent-b") == "run-5-run-a-7-agent-b"


def test_compound_identity_is_collision_resistant_for_delimiters():
    runner = load_runner()
    left = runner.compound_identity("a-b", "c")
    right = runner.compound_identity("a", "b-c")
    assert left != right
    assert runner.validate_component("abc-123") == "abc-123"
    for bad in ("", "ABC", "a/b", "a\\b", "a:b", "a\n", "../a", "a" * 201):
        with pytest.raises(ValueError):
            runner.validate_component(bad)


def test_runner_dry_policy_never_starts_a_model():
    runner = load_runner()
    policy = runner.task_policy()
    assert policy["model_launch"] == "parent-controlled only"
    assert policy["hidden_oracle_to_model"] is False
    assert policy["retries"] == 0


def test_hidden_fixture_is_a_parent_only_pytest_module():
    fixture = ROOT / "bench" / "fixtures" / "pal_run_journal_hidden_test.py"
    assert fixture.is_file()
    source = fixture.read_text(encoding="utf-8")
    assert "StoreCorruptError" in source
    assert "O_APPEND" in source
    assert "~/.openclink/store" in source
    assert "test_duplicate" in source
