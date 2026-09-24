"""Run the isolated PAL #149 journal task on frozen local model arms."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "qwen38-tuning" / "bench"
sys.path.insert(0, str(BENCH))

def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

previous = _load(Path(__file__).with_name("run-pal-remaining.py"), "pal_remaining_journal")
q4 = _load(Path(__file__).with_name("run-q4-pal-workflow.py"), "pal_q4_journal")
journal_screen = _load(BENCH / "pal_workflow_screen.py", "pal_journal_screen")
gsq = previous.gsq
PAL_SHA = previous.PAL_SHA
PAL_ORIGINAL = previous.pal.PAL_ORIGINAL
PAL_HIDDEN = previous.PAL_HIDDEN
CLIENT = previous.CLIENT
SANDBOX_IMAGE = previous.pal.SANDBOX_IMAGE
CONTEXT, OUTPUT, TURNS, TIMEOUT, EFFORT = 65536, 8192, 64, 1800, "medium"
CANDIDATES = ("swift_q4km", "turbo_mtp_q4km", "swift", "turbo", "dirk", "gsq", "nvfp4", "exl3")
ALLOWED = {"tools/run_journal.py", "tests/test_run_journal.py"}
TASK_SOURCE_FILE, TASK_TEST_FILE = "tools/run_journal.py", "tests/test_run_journal.py"
JOURNAL_HIDDEN = ROOT / "qwen38-tuning" / "bench" / "fixtures" / "pal_run_journal_hidden_test.py"
_COMPONENT = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


def validate_component(value: str) -> str:
    if not isinstance(value, str) or not _COMPONENT.fullmatch(value) or len(value) > 200:
        raise ValueError("invalid journal identity component")
    return value


def compound_identity(run_id: str, agent_id: str) -> str:
    run_id, agent_id = validate_component(run_id), validate_component(agent_id)
    return f"run-{len(run_id)}-{run_id}-{len(agent_id)}-{agent_id}"


def task_policy() -> dict[str, object]:
    return {"id": "pal-journal-v1", "baseline_sha": PAL_SHA,
        "prompt": JOURNAL_PROMPT, "prompt_sha256": hashlib.sha256(JOURNAL_PROMPT.encode()).hexdigest(),
        "allowed_files": [TASK_SOURCE_FILE, TASK_TEST_FILE], "candidates": list(CANDIDATES),
        "context": CONTEXT, "output": OUTPUT, "max_turns": TURNS, "timeout_s": TIMEOUT,
        "effort": EFFORT, "duplicate_identity": "reject", "hidden_oracle_to_model": False,
        "model_launch": "parent-controlled only", "retries": 0, "sandbox_image": SANDBOX_IMAGE,
        "client_path": str(CLIENT)}

JOURNAL_PROMPT = """You are working from scratch in an offline disposable clone of pal-mcp-server. Follow TDD using only the fixed run_visible_tests tool with suite_id "user-config-focused"; do not claim tests ran unless that tool returned the result.

Implement the standalone run-journal storage layer for issue #149. Modify only tools/run_journal.py and tests/test_run_journal.py. Do not modify utils/record_store.py, tools/clink.py, existing tests, config or any other file. Do not wire CLinkTool yet.

Required API:
- RunJournal(store: RecordStore)
- append(run_id: str, agent_id: str, result: dict[str, Any]) -> str
- read(run_id: str) -> list[dict[str, Any]]

Required behavior:
- append one self-describing JSON record per agent return through exactly one atomic RecordStore.put; no JSONL, O_APPEND, aggregate file or in-memory-only journal.
- Payload contains run_id, agent_id and the unmodified result mapping. Tests inject RecordStore(tmp_path), never ~/.openclink/store.
- read returns exactly one record for each matching agent, deterministically sorted by compound identity; missing run returns []. A newly constructed RecordStore/RunJournal reads after restart.
- Matching StoreCorruptError propagates; corruption in another run does not make this run appear valid.
- Reject invalid run/agent components using the existing safe identity rules: empty, uppercase/case-collision, separators, colon, newline, traversal and overlong values.
- Compound identity is exactly length-prefixed: run-{len(run_id)}-{run_id}-{len(agent_id)}-{agent_id}; delimiter-containing components must not collide.
- Duplicate (run_id, agent_id) is rejected with ValueError. Concurrent writes/read-during-writes must not expose torn records; snapshot ordering during concurrent writes is not promised.

First observe the appropriate baseline test failure, then add focused tests, implement the smallest layer, and rerun to green. No network, installs, remotes, hidden-test access or changes outside the two allowed files."""


def _write_json(path, value):
    previous.pal._write_json(path, value)


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_inventory(directory):
    return previous.pal.source_inventory(directory)


def clone_snapshot(work):
    return previous.clone_snapshot(work)


def candidate_argv(key):
    if key in ("swift_q4km", "turbo_mtp_q4km"):
        mode = "mtp-ngram" if key == "swift_q4km" else "ngram"
        argv = gsq.apply_template(gsq.llama_argv(key, CONTEXT, mode, "9500,14500", 24), "stock")
        return gsq.replace_flag(gsq.replace_flag(argv, "--port", 18080), "-lv", 4)
    if key in ("swift", "turbo"):
        mode = "mtp-ngram" if key == "swift" else "ngram"
        argv = gsq.apply_template(gsq.llama_argv(key, CONTEXT, mode, "8500,15468", 24), "stock")
        return gsq.replace_flag(gsq.replace_flag(argv, "--port", 18080), "-lv", 4)
    return previous.candidate_argv(key)


def candidate_files(key, argv):
    if key == "exl3":
        return previous.exl3_files()
    primary = Path(argv[argv.index("-m") + 1])
    return [{"path": str(primary), "bytes": primary.stat().st_size, "sha256": gsq.digest(primary)}]


def _copy_evidence(source, target):
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def _verify_journal_workspace(workspace, baseline, output, visible_test_journal=None):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    before = source_inventory(baseline)
    after = source_inventory(workspace)
    changes = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    violations = sorted(set(changes) - ALLOWED)
    hidden = journal_screen._run(workspace, SANDBOX_IMAGE, [str(JOURNAL_HIDDEN), "-q"], output, "hidden", JOURNAL_HIDDEN)
    visible = journal_screen._run(workspace, SANDBOX_IMAGE,
        [TASK_TEST_FILE, "-q"], output, "visible")
    workflow = {"tests_file_changed": TASK_TEST_FILE in changes,
                "red_then_green": False, "test_runs": 0, "evidence": {}}
    if visible_test_journal is not None and Path(visible_test_journal).is_file():
        journal = Path(visible_test_journal)
        runs = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        workflow["test_runs"] = len(runs)
        baseline_source = before.get(TASK_SOURCE_FILE)
        baseline_tests = before.get(TASK_TEST_FILE)
        final_source = after.get(TASK_SOURCE_FILE)
        final_tests = after.get(TASK_TEST_FILE)
        red_tests = None
        red_seen = False
        for run in runs:
            if run.get("suite_id") != "user-config-focused" or run.get("timed_out") is True:
                continue
            observed = run.get("workspace_files") or {}
            source_before = observed.get(TASK_SOURCE_FILE) == baseline_source
            tests_changed = observed.get(TASK_TEST_FILE) not in (None, baseline_tests)
            if run.get("passed") is False and source_before and tests_changed:
                red_seen, red_tests = True, observed.get(TASK_TEST_FILE)
            elif (run.get("passed") is True and red_seen and not source_before and tests_changed
                  and observed.get(TASK_TEST_FILE) == red_tests == final_tests
                  and observed.get(TASK_SOURCE_FILE) == final_source):
                workflow["red_then_green"] = True
            for field in ("stdout_ref", "stderr_ref"):
                ref = run.get(field)
                path = journal.parent / ref if isinstance(ref, str) else None
                if path is not None and path.is_file() and path.parent == journal.parent:
                    workflow["evidence"][ref] = {"bytes": path.stat().st_size, "sha256": _digest(path)}
        workflow["evidence"][journal.name] = {"bytes": journal.stat().st_size, "sha256": _digest(journal)}
    passed = (not violations and hidden["returncode"] == 0 and visible["returncode"] == 0
              and not hidden["timed_out"] and not visible["timed_out"]
              and workflow["tests_file_changed"] and workflow["red_then_green"])
    result = {"passed": passed, "returncode": 0 if passed else 1,
              "hidden": hidden, "visible": visible, "changed_files": changes,
              "scope_violations": violations, "workflow": workflow}
    _write_json(output / "result.json", result)
    return result


def check_lease(lease):
    path = ROOT / "qwen38-tuning/.port8080.lock"
    if not path.exists() or path.read_text() != lease:
        raise RuntimeError("legacy lease changed")
    subprocess.run(["C:/Program Files/Git/usr/bin/bash.exe", "-c", "kill -0 \"$1\"",
                    "lease", lease.split()[0]], check=True)


def assert_ports_free(ports=(8000, 8080, 18080)):
    for port in ports:
        with socket.socket() as sock:
            sock.settimeout(.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"occupied port {port}")


def prepare_workspace(template, task_root):
    work = task_root / "work"
    shutil.copytree(template, work)
    # Edit cannot create a missing file under the restricted candidate policy;
    # these empty scaffolds are harness setup and are recorded outside the prompt.
    (work / TASK_SOURCE_FILE).parent.mkdir(parents=True, exist_ok=True)
    (work / TASK_TEST_FILE).parent.mkdir(parents=True, exist_ok=True)
    (work / TASK_SOURCE_FILE).write_text("# harness scaffold; candidate must replace\n", encoding="utf-8")
    (work / TASK_TEST_FILE).write_text("# harness scaffold; candidate must replace\n", encoding="utf-8")
    return work


def run_cell(key, root, template, lease, uuids):
    cell = root / key
    cell.mkdir()
    work = prepare_workspace(template, cell)
    baseline = cell / "baseline"
    shutil.copytree(work, baseline)
    model = adapter = tap = None
    times = {}
    started = time.monotonic()
    result = {"status": "evidence_failure", "times": times}
    try:
        check_lease(lease)
        assert_ports_free()
        pre = previous.pal.sample_resources(uuids)
        _write_json(cell / "gpu-preflight.json", pre)
        if pre["errors"] or pre["suspected_game_processes"] or not pre["gpus"]:
            raise RuntimeError("resource preflight failed")
        argv = candidate_argv(key)
        is_exl3 = key == "exl3"
        if is_exl3:
            files = candidate_files(key, argv)
            primary = Path(files[0]["path"])
            port, cwd, model_name = 8000, ROOT / "exllamav3-mia", previous.incumbents.EXL3.name
            metadata = {"quant": "SC4.0bpw-H5", "upstream_revision": "b4e3574d5665efb5d8031c05a578837e6700a912"}
        else:
            if key == "nvfp4":
                previous.check_nvfp4_budget(pre)
            primary = Path(argv[argv.index("-m") + 1])
            files = candidate_files(key, argv)
            gsq.verify_artifact_digest(key, files[0]["sha256"])
            port, cwd, model_name = 18080, ROOT, argv[argv.index("--alias") + 1]
            metadata = gsq.artifact_metadata(key)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(uuids), PYTHONIOENCODING="utf-8")
        if is_exl3:
            env["EXL3_RESTART_FLAG"] = str(cell / "restart-flag.json")
        _write_json(cell / "launch.json", {"artifact": key, "artifact_metadata": metadata,
            "files": files, "argv": argv, "engine_sha256": gsq.digest(argv[0]),
            "cwd": str(cwd), "profile": "PAL #149 frozen operating point"})
        model = previous.pal.ModelSession(argv, cwd, port, env=env, log_path=cell / "server.log")
        if is_exl3:
            observed = previous.start_exl3(model, argv, cwd, env, cell / "server.log", primary)
        else:
            observed = model.start()
            observed.update(gsq.layer_evidence((cell / "server.log").read_text(encoding="utf-8", errors="replace"), argv))
            if observed["layers"] != [66, 0]:
                raise RuntimeError("CPU layer spill")
        observed["listener"] = gsq.listener_evidence(model.process.pid, port)
        _write_json(cell / "boot.json", observed)
        tap = previous.pal._load_tap_module("relay").Tap(0, port, str(cell / "backend-wire"))
        tap.start()
        if is_exl3:
            transform = lambda body: q4.frozen_request(body, None)
        else:
            pieces = gsq.gguf_pieces(str(primary))
            transform = lambda body: q4.frozen_request(body, gsq.conditional_han_bias(pieces, body["messages"]))
        adapter = previous.pal.AdapterServer(f"http://127.0.0.1:{tap.listen_port}", request_transform=transform,
            observation_directory=cell / "backend-observations").start()
        times["setup_s"] = time.monotonic() - started

        def client_runner(argv2, env2, wd, prompt, history, timeout):
            times["start"] = time.monotonic()
            env2 = dict(env2, CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(CONTEXT),
                        CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(OUTPUT), CLAUDE_CODE_MAX_TURNS=str(TURNS))
            argv2 = list(argv2) + ["--effort", EFFORT]
            history.append("effective_client_policy", {"argv": argv2, "context": CONTEXT,
                "output_cap": OUTPUT, "turn_cap": TURNS})
            try:
                return previous.pal.run_client(argv2, env2, wd, JOURNAL_PROMPT, history, timeout=timeout)
            finally:
                times["client_end"] = time.monotonic()

        def verifier(wd):
            try:
                return _verify_journal_workspace(wd, baseline, cell / "verification",
                    visible_test_journal=cell / "visible-test-evidence/visible-tests.jsonl")
            finally:
                times["end"] = time.monotonic()
                if (cell / "visible-test-evidence").is_dir():
                    shutil.copytree(cell / "visible-test-evidence", cell / "session/visible-test-evidence")
                if (cell / "backend-observations").is_dir():
                    shutil.copytree(cell / "backend-observations", cell / "session/upstream-observations")

        suites = cell / "suites.json"
        _write_json(suites, {"user-config-focused": {"args": [TASK_TEST_FILE, "-q"],
            "evidence_files": [TASK_SOURCE_FILE, TASK_TEST_FILE]}})
        mcp = cell / "mcp.json"
        _write_json(mcp, q4.mcp_config(ROOT / "qwen38-tuning/bench/fixture_test_tool.py", work,
            SANDBOX_IMAGE, cell / "visible-test-evidence", suites))
        spec = {"artifacts": files, "gpu_uuids": uuids,
            "test": {"format": "claude-cli", "case_id": "pal-journal", "stage": "first-pass", "round": 1, "attempt": 1},
            "context": {"requested_window_tokens": CONTEXT, "runtime_window_tokens": CONTEXT,
                "input_tokens": None, "cached_input_tokens": None, "max_output_tokens": OUTPUT,
                "token_count_source": None, "history_policy": "client-default-compaction", "source": "pal-journal-v1"}}
        summary = previous.pal.run_recorded_session(cell / "session", work, CLIENT, model_name,
            adapter.port, JOURNAL_PROMPT, spec, verifier, timeout=TIMEOUT, client_runner=client_runner,
            client_mcp_config=mcp, client_allowed_tools=q4.MCP_TOOL,
            server_probe=lambda _port: observed,
            recorder_source_paths=[Path(__file__), Path(previous.__file__), Path(q4.__file__),
                ROOT / "qwen38-tuning/bench/fixture_test_tool.py", ROOT / "qwen38-tuning/bench/pal_workflow_screen.py",
                ROOT / "qwen38-tuning/bench/stream_observation.py"])
        result.update(summary=summary, integrity=previous.pal.verify_evidence(cell / "session"))
        result["status"] = "original_verified_audit_pending" if summary["outcome"] == "verified" else summary["outcome"]
        result["task_wall_s"] = times.get("end", times.get("client_end", time.monotonic())) - times.get("start", started)
        result["clone_integrity"] = {"baseline": str(baseline), "final": str(cell / "session/workspace-final")}
        if not result["integrity"]["complete"]:
            result["status"] = "evidence_failure"
    except Exception as error:
        (cell / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        result.update(status="evidence_failure", reason=str(error), error_type=type(error).__name__)
    finally:
        cleanup = previous.cleanup_owned(adapter, tap, model, [8000, 8080, 18080])
        result["cleanup_errors"] = cleanup
        if cleanup:
            result.update(status="evidence_failure", reason="owned cleanup fault")
        _write_json(cell / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--legacy-owner", type=int)
    parser.add_argument("--pal-template", type=Path, required=True)
    args = parser.parse_args(argv)
    policy = json.loads((ROOT / "qwen38-tuning/tools/run-pal-journal.py").read_text(encoding="utf-8")) if False else {
        "id": "pal-journal-v1", "baseline_sha": PAL_SHA, "candidates": list(CANDIDATES),
        "context": CONTEXT, "output": OUTPUT, "turns": TURNS, "timeout_s": TIMEOUT,
        "prompt_sha256": hashlib.sha256(JOURNAL_PROMPT.encode()).hexdigest(), "allowed_files": ["tools/run_journal.py", "tests/test_run_journal.py"]}
    candidates = [args.candidate] if args.candidate else list(CANDIDATES)
    policy["candidates"] = candidates
    if not args.run:
        print(json.dumps(policy, indent=2)); return
    if args.legacy_owner is None:
        raise RuntimeError("matching live legacy lease required")
    lock_path = ROOT / "qwen38-tuning/.port8080.lock"
    if not lock_path.exists():
        raise RuntimeError("matching live legacy lease required")
    lease = lock_path.read_text()
    if lease.split()[0] != str(args.legacy_owner):
        raise RuntimeError("matching live legacy lease required")
    template = args.pal_template.resolve()
    if template == PAL_ORIGINAL.resolve() or template.is_relative_to(PAL_ORIGINAL.resolve()):
        raise RuntimeError("template cannot be original PAL repository")
    if clone_snapshot(template)["git_head"].strip() != PAL_SHA:
        raise RuntimeError("template baseline mismatch")
    if subprocess.check_output(["git", "-C", str(template), "status", "--porcelain=v1", "--untracked-files=all"], text=True).strip():
        raise RuntimeError("template dirty")
    root = Path(tempfile.mkdtemp(prefix="qwen-pal-journal-")); print("PRIVATE_EVIDENCE " + str(root), flush=True)
    with previous.original_guard(root):
        _write_json(root / "frozen-protocol.json", policy)
        progress = {key: {"status": "planned"} for key in candidates}; _write_json(root / "progress.json", progress)
        uuids = gsq.arena.BOTH_CARDS.split(",")
        with previous.pal.CampaignLock(ROOT / "qwen38-tuning/.agent-campaign.lock", root.name, policy["id"]):
            for index, key in enumerate(candidates):
                progress[key] = {"status": "running"}; _write_json(root / "progress.json", progress)
                progress[key] = run_cell(key, root, template, lease, uuids)
                _write_json(root / "progress.json", progress)
                if progress[key]["status"] == "evidence_failure":
                    for later in candidates[index + 1:]: progress[later] = {"status": "not_run", "reason": "prior evidence failure"}
                    break
    print("CAMPAIGN_FINISHED " + str(root), flush=True)

if __name__ == "__main__": main()
