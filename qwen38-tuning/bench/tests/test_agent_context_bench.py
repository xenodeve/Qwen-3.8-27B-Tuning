"""Red-first contracts for the frozen long-horizon campaign manifest."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import agent_context_bench as bench


CANDIDATES = ("swift", "turbo", "dirk", "gsq")
TASKS = ("inventory", "ledger", "mixed-language")
SEEDS = (29, 43, 71)


def _manifest(tmp_path: Path) -> dict:
    artifacts = []
    profiles = []
    attempts = []
    for candidate in CANDIDATES:
        model = tmp_path / f"{candidate}.gguf"
        model.write_bytes(candidate.encode())
        artifacts.append({
            "key": candidate, "path": str(model), "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "quant": ("IQ3_S-MTP" if candidate == "gsq" else
                      "MTP-Q6_K" if candidate == "turbo" else
                      "UD-Q6_K" if candidate == "dirk" else "Q6_K"),
            "revision": f"rev-{candidate}", "template_sha256": f"template-{candidate}",
        })
        profile = {
            "profile_id": f"{candidate}-p1", "artifact": candidate,
            "requested_context_tokens": 65536, "runtime_context_tokens": 65536,
            "tensor_split": "8500,15468", "effort": "medium", "speculation": "mtp-ngram",
            "kv_cache": "q4_0", "seed_policy": "fixed-per-round",
        }
        profiles.append(profile)
        for round_no, seed in enumerate(SEEDS, 1):
            for task in TASKS:
                attempts.append({
                    "attempt_id": f"{candidate}-r{round_no}-{task}", "model": candidate,
                    "case_id": task, "round": round_no, "attempt": 1,
                    "config_id": profile["profile_id"], "seed": seed,
                })
    return {
        "schema_version": 1, "campaign_id": "long-horizon-2026-09-21",
        "engine": {"binary": "llama-server.exe", "version": "b10499-1deefcca3"},
        "client": {"executable": "claude.exe", "version": "unfrozen-test-client"},
        "gpu_uuids": ["GPU-4070", "GPU-5060"], "artifacts": artifacts,
        "profiles": profiles,
        "tasks": {"episodes": list(TASKS), "rounds": 3, "seeds": list(SEEDS),
                   "max_minutes": 60, "max_assistant_turns": 128,
                   "continuation_policy": "one-frozen-nudge"},
        "attempts": attempts,
    }


def write_manifest(tmp_path, value):
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def test_validate_requires_all_four_pinned_candidates_and_full_rotated_grid(tmp_path):
    manifest = _manifest(tmp_path)
    result = bench.validate_manifest(write_manifest(tmp_path, manifest))
    assert result["campaign_id"] == manifest["campaign_id"]
    assert result["artifact_keys"] == list(CANDIDATES)
    assert result["attempt_count"] == 36
    assert result["rounds"] == 3
    assert result["gpu_uuids"] == manifest["gpu_uuids"]
    assert result["writes_performed"] is False


def test_validate_rejects_wrong_artifact_hash_before_any_plan(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        bench.validate_manifest(write_manifest(tmp_path, manifest))


def test_validate_rejects_profile_context_mismatch(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["profiles"][0]["runtime_context_tokens"] = 32768
    with pytest.raises(ValueError, match="context"):
        bench.validate_manifest(write_manifest(tmp_path, manifest))


def test_validate_rejects_missing_attempt_without_writing_register(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["attempts"].pop()
    with pytest.raises(ValueError, match="attempt"):
        bench.validate_manifest(write_manifest(tmp_path, manifest))
    assert not (tmp_path / "register").exists()


def test_dry_run_is_json_only_and_does_not_start_process_or_create_register(tmp_path, capsys):
    path = write_manifest(tmp_path, _manifest(tmp_path))
    exit_code = bench.main(["dry-run", "--spec", str(path)])
    assert exit_code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["attempt_count"] == 36
    assert plan["profiles"]["swift"]["requested_context_tokens"] == 65536
    assert list(tmp_path.glob("register*")) == []


def test_cli_rejects_gpu_actions_in_validate_dry_run(tmp_path):
    path = write_manifest(tmp_path, _manifest(tmp_path))
    with pytest.raises(SystemExit):
        bench.main(["run", "--spec", str(path)])
