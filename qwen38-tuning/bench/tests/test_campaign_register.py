"""Campaign register state-machine tests."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
try:
    from campaign_register import CampaignRegister
except ImportError:
    CampaignRegister = None


def spec():
    return {
        "campaign": "trial",
        "extra": {"preserved": [1, 2]},
        "attempts": [
            {"attempt_id": "a", "model": "m1", "case_id": "c1", "round": 1, "config_id": "x"},
            {"attempt_id": "b", "model": "m2", "case_id": "c2", "round": 1, "config_id": "y"},
            {"attempt_id": "c", "model": "m3", "case_id": "c3", "round": 2, "config_id": "z"},
        ],
    }


def test_register_exports_public_class():
    assert CampaignRegister is not None


def test_create_preserves_canonical_spec_and_rejects_malformed_attempt_ids(tmp_path):
    register = CampaignRegister(tmp_path / "campaign", spec())
    saved = (tmp_path / "campaign" / "specification.json").read_text(encoding="utf-8")
    assert json.loads(saved) == spec()
    for attempts in ([{**spec()["attempts"][0], "attempt_id": ""}],
                     [spec()["attempts"][0], spec()["attempts"][0]]):
        with pytest.raises(ValueError):
            CampaignRegister(tmp_path / str(len(attempts)), {"attempts": attempts})
    with pytest.raises(FileExistsError):
        CampaignRegister(tmp_path / "campaign", spec())
    assert register.summary()["expected"] == 3


def test_state_transitions_and_required_evidence(tmp_path):
    register = CampaignRegister(tmp_path / "campaign", spec())
    with pytest.raises(ValueError):
        register.mark("unknown", "running")
    with pytest.raises(ValueError):
        register.mark("a", "verified", {})
    with pytest.raises(ValueError):
        register.mark("a", "not_run")
    register.mark("a", "not_run", reason="hardware unavailable")
    with pytest.raises(ValueError):
        register.mark("a", "running")
    register.mark("b", "running")
    with pytest.raises(ValueError):
        register.mark("b", "planned")
    with pytest.raises(ValueError):
        register.mark("b", "failed")
    register.mark("b", "verified", {"session_id": "s-1", "verifier_evidence_ref": "evidence.json"})
    with pytest.raises(ValueError):
        register.mark("b", "failed", reason="cannot overwrite")


def test_summary_counts_three_outcomes_and_missing_attempts(tmp_path):
    register = CampaignRegister(tmp_path / "campaign", spec())
    register.mark("a", "running")
    register.mark("a", "failed", reason="timeout")
    register.mark("b", "running")
    register.mark("b", "censored", reason="interrupted")
    summary = register.summary()
    assert summary["expected"] == 3
    assert summary["terminal"] == 2
    assert summary["counts"]["failed"] == 1
    assert summary["counts"]["censored"] == 1
    assert summary["missing_attempt_ids"] == ["c"]
    assert summary["failed_attempt_ids"] == ["a"]
    assert summary["censored_attempt_ids"] == ["b"]
    assert not summary["complete"]


def test_tampered_spec_and_corrupt_progress_fail_loudly(tmp_path):
    path = tmp_path / "campaign"
    register = CampaignRegister(path, spec())
    (path / "specification.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        register.mark("a", "running")
    with pytest.raises(ValueError, match="digest"):
        CampaignRegister.load(path)
    path = tmp_path / "corrupt"
    CampaignRegister(path, spec())
    (path / "progress.json").write_text('{"attempts": {}}', encoding="utf-8")
    with pytest.raises(ValueError):
        CampaignRegister.load(path)


def test_reopen_retains_running_state_and_replace_failure_keeps_progress(tmp_path):
    path = tmp_path / "campaign"
    register = CampaignRegister(path, spec())
    register.mark("a", "running")
    reopened = CampaignRegister.load(path)
    assert reopened.summary()["missing_attempt_ids"] == ["a", "b", "c"]
    before = (path / "progress.json").read_bytes()
    with patch("campaign_register.os.replace", side_effect=OSError("injected")):
        with pytest.raises(OSError, match="injected"):
            reopened.mark("a", "failed", reason="test failure")
    assert (path / "progress.json").read_bytes() == before
    assert reopened.summary()["counts"]["running"] == 1
