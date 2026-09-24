"""Behavioral contracts for the cross-process campaign lease."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).parents[1]))
from campaign_lock import CampaignLock, LockUnavailable


def test_lock_records_owner_and_blocks_second_owner(tmp_path: Path) -> None:
    path = tmp_path / "campaign.lock"
    first = CampaignLock(path, "campaign-1", "profile-a").acquire()
    try:
        owner = CampaignLock.read_owner(path)
        assert owner["campaign_id"] == "campaign-1"
        assert owner["profile_id"] == "profile-a"
        with pytest.raises(LockUnavailable):
            CampaignLock(path, "campaign-1", "profile-b").acquire()
    finally:
        first.release()


def test_release_allows_next_owner_without_deleting_lock_file(tmp_path: Path) -> None:
    path = tmp_path / "campaign.lock"
    first = CampaignLock(path, "campaign-1", "profile-a").acquire()
    first.release()
    assert path.exists()
    with CampaignLock(path, "campaign-1", "profile-b") as second:
        assert CampaignLock.read_owner(path)["profile_id"] == "profile-b"


def test_lock_rejects_empty_identity_and_double_acquire(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CampaignLock(tmp_path / "x", "", "profile")
    lock = CampaignLock(tmp_path / "x", "campaign", "profile").acquire()
    try:
        with pytest.raises(RuntimeError):
            lock.acquire()
    finally:
        lock.release()


def test_corrupt_old_metadata_does_not_authorize_deleting_or_bypassing_lock(tmp_path: Path) -> None:
    path = tmp_path / "campaign.lock"
    path.write_text("not-json\n", encoding="utf-8")
    with CampaignLock(path, "campaign", "profile") as lock:
        assert lock.path == path
        assert CampaignLock.read_owner(path)["pid"] > 0
