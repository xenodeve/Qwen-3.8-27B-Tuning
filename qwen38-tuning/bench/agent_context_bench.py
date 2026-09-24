"""Validate and preview the frozen long-horizon campaign specification.

W1 deliberately has no model/client/process side effects. It verifies the complete
cell grid and local artifact bytes before W2 adds owned lifecycle execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CANDIDATES = ("swift", "turbo", "dirk", "gsq")
EXPECTED_QUANTS = {"swift": "Q6_K", "turbo": "MTP-Q6_K",
                   "dirk": "UD-Q6_K", "gsq": "IQ3_S-MTP"}
REQUIRED_ARTIFACT_FIELDS = {"key", "path", "sha256", "quant", "revision", "template_sha256"}
REQUIRED_PROFILE_FIELDS = {
    "profile_id", "artifact", "requested_context_tokens", "runtime_context_tokens",
    "tensor_split", "effort", "speculation", "kv_cache", "seed_policy",
}
REQUIRED_TEST_FIELDS = {"format", "case_id", "stage", "round", "attempt", "config_id", "seed"}


def _load(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read campaign specification: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("campaign specification must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _nonempty(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _positive_int(value: Any, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _validate_artifacts(raw: Any) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if not isinstance(raw, list) or {item.get("key") for item in raw if isinstance(item, dict)} != set(CANDIDATES):
        raise ValueError(f"artifacts must contain exactly: {', '.join(CANDIDATES)}")
    by_key: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or set(item) < REQUIRED_ARTIFACT_FIELDS:
            raise ValueError("artifact is missing frozen identity fields")
        key = item["key"]
        if key in by_key:
            raise ValueError(f"duplicate artifact: {key}")
        if item["quant"] != EXPECTED_QUANTS[key]:
            raise ValueError(f"{key} quant must be {EXPECTED_QUANTS[key]}")
        path = Path(item["path"])
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"artifact path is not a regular file: {key}")
        expected = item["sha256"]
        if not isinstance(expected, str) or len(expected) != 64 or _sha256(path).lower() != expected.lower():
            raise ValueError(f"artifact checksum mismatch: {key}")
        for field in ("revision", "template_sha256"):
            _nonempty(item[field], f"{key}.{field}")
        by_key[key] = item
    return by_key, {key: str(item["path"]) for key, item in by_key.items()}


def _validate_profiles(raw: Any, artifacts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) != len(CANDIDATES):
        raise ValueError("profiles must contain one profile for each candidate")
    profiles: dict[str, dict[str, Any]] = {}
    for profile in raw:
        if not isinstance(profile, dict) or set(profile) < REQUIRED_PROFILE_FIELDS:
            raise ValueError("profile is missing frozen configuration fields")
        profile_id = profile["profile_id"]
        _nonempty(profile_id, "profile_id")
        if profile_id in profiles or profile["artifact"] not in artifacts:
            raise ValueError("profile ID or artifact reference is invalid")
        _positive_int(profile["requested_context_tokens"], f"{profile_id}.requested_context_tokens")
        _positive_int(profile["runtime_context_tokens"], f"{profile_id}.runtime_context_tokens")
        if profile["requested_context_tokens"] != profile["runtime_context_tokens"]:
            raise ValueError(f"{profile_id} context allocation/runtime mismatch")
        for field in ("tensor_split", "effort", "speculation", "kv_cache", "seed_policy"):
            _nonempty(profile[field], f"{profile_id}.{field}")
        profiles[profile_id] = profile
    if {profile["artifact"] for profile in profiles.values()} != set(CANDIDATES):
        raise ValueError("profiles must cover all candidates exactly once")
    return profiles


def _validate_attempts(raw: Any, profiles: dict[str, dict[str, Any]], tasks: dict[str, Any]) -> int:
    if not isinstance(raw, list):
        raise ValueError("attempts must be a list")
    episodes, rounds, seeds = tasks["episodes"], tasks["rounds"], tasks["seeds"]
    _positive_int(rounds, "tasks.rounds")
    if not isinstance(episodes, list) or not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("tasks.episodes must be unique and nonempty")
    if not isinstance(seeds, list) or len(seeds) != rounds or any(type(seed) is not int for seed in seeds):
        raise ValueError("tasks.seeds must contain one integer seed per round")
    expected = {(candidate, round_no, case_id) for candidate in CANDIDATES
                for round_no in range(1, rounds + 1) for case_id in episodes}
    seen: set[tuple[str, int, str]] = set()
    ids: set[str] = set()
    for attempt in raw:
        if not isinstance(attempt, dict) or set(attempt) < REQUIRED_TEST_FIELDS:
            raise ValueError("attempt is missing frozen test fields")
        if attempt["attempt_id"] in ids:
            raise ValueError("attempt IDs must be unique")
        ids.add(attempt["attempt_id"])
        key = (attempt["model"], attempt["round"], attempt["case_id"])
        if key in seen or key not in expected:
            raise ValueError(f"unexpected or duplicate attempt cell: {key}")
        seen.add(key)
        if attempt["config_id"] not in profiles or profiles[attempt["config_id"]]["artifact"] != attempt["model"]:
            raise ValueError(f"attempt profile does not match model: {attempt['attempt_id']}")
        if attempt["attempt"] != 1 or attempt["seed"] != seeds[attempt["round"] - 1]:
            raise ValueError(f"attempt rotation/seed mismatch: {attempt['attempt_id']}")
    if seen != expected:
        raise ValueError(f"attempt grid incomplete: missing {len(expected - seen)} cells")
    return len(raw)


def validate_manifest(path: str | Path) -> dict[str, Any]:
    """Validate all frozen cells with no filesystem writes or process launches."""
    manifest = _load(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {manifest.get('schema_version')!r}")
    _nonempty(manifest.get("campaign_id"), "campaign_id")
    for section in ("engine", "client"):
        if not isinstance(manifest.get(section), dict):
            raise ValueError(f"missing {section} provenance")
        for value in manifest[section].values():
            _nonempty(value, f"{section} provenance")
    gpu_uuids = manifest.get("gpu_uuids")
    if not isinstance(gpu_uuids, list) or not gpu_uuids or any(
            not isinstance(uuid, str) or not uuid.startswith("GPU-") or not uuid.strip() == uuid
            for uuid in gpu_uuids):
        raise ValueError("gpu_uuids must be explicit UUIDs")
    artifacts, paths = _validate_artifacts(manifest.get("artifacts"))
    profiles = _validate_profiles(manifest.get("profiles"), artifacts)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("missing tasks specification")
    for key in ("max_minutes", "max_assistant_turns"):
        _positive_int(tasks.get(key), f"tasks.{key}")
    attempt_count = _validate_attempts(manifest.get("attempts"), profiles, tasks)
    return {
        "schema_version": SCHEMA_VERSION, "campaign_id": manifest["campaign_id"],
        "artifact_keys": list(CANDIDATES), "artifact_paths": paths,
        "profiles": {candidate: next(profile for profile in profiles.values()
                                     if profile["artifact"] == candidate)
                     for candidate in CANDIDATES},
        "gpu_uuids": list(gpu_uuids), "episodes": list(tasks["episodes"]),
        "rounds": tasks["rounds"], "seeds": list(tasks["seeds"]),
        "attempt_count": attempt_count, "budget_minutes": tasks["max_minutes"],
        "budget_assistant_turns": tasks["max_assistant_turns"], "writes_performed": False,
    }


def _json_print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "dry-run"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--spec", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        parser.error("run is not implemented until W2-W8 gates are complete")
    result = validate_manifest(args.spec)
    if args.command == "dry-run":
        result["dry_run"] = True
        result["actions"] = ["validate hashes", "validate runtime later", "run no processes"]
    _json_print(result)
    return 0


if __name__ == "__main__":
    main()
