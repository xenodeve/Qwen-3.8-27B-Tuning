"""Frozen PAL remote runner policy; never starts a real client or gateway."""
import importlib.util
from pathlib import Path


def load_runner():
    path = Path(__file__).resolve().parents[2] / 'tools/run-pal-fp8.py'
    spec = importlib.util.spec_from_file_location('pal_fp8', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_policy_reuses_exact_frozen_pal_task_and_tools():
    runner = load_runner()
    protocol = runner.protocol()
    assert protocol['pal_sha'] == '200fcb9262d25e4002e30bf00c4364f55a42354e'
    assert protocol['prompt'] == runner.previous.PAL_PROMPT
    assert protocol['tools'] == runner.previous.MCP_TOOL
    assert protocol['model'] == 'qwen3.8-27b-fp8'
    assert protocol['context'] == 65536
    assert protocol['output'] == 8192
    assert protocol['timeout_s'] == 1800
    assert protocol['max_turns'] == 64
    assert protocol['retries'] == 0


def test_openai_measurement_repeat_is_explicit_without_changing_task():
    runner = load_runner()
    old, new = runner.protocol(), runner.protocol('openai')
    assert old['gateway_api'] == 'anthropic'
    assert new['gateway_api'] == 'openai'
    assert new['id'] != old['id']
    assert new['measurement_repeat_of'] == old['id']
    assert new['prompt'] == old['prompt']
    assert new['pal_sha'] == old['pal_sha']


def test_attempt_timer_uses_client_start_through_original_verifier(tmp_path, monkeypatch):
    # Remote transport setup/teardown must not silently replace result25's timer.
    import json
    runner = load_runner()
    template = tmp_path / 'template'; template.mkdir()
    root = tmp_path / 'attempt'; root.mkdir()
    ticks = iter([10.0, 15.0, 20.0])
    monkeypatch.setattr(runner, 'clock', lambda: next(ticks), raising=False)
    monkeypatch.setattr(runner.previous.pal, 'run_client', lambda *a, **k: {'status': 'completed'})
    monkeypatch.setattr(runner.previous.pal, 'verify_pal_workspace', lambda *a, **k: {'passed': True, 'returncode': 0})
    monkeypatch.setattr(runner.previous.pal, 'client_context_valid', lambda *a: True)
    monkeypatch.setattr(runner, 'verify_evidence', lambda *a: {'complete': True})
    def record(session, work, executable, prompt, credential, verifier, **kwargs):
        session.mkdir()
        (session / 'stdout.jsonl').write_text(json.dumps({'type': 'result'}) + '\n')
        kwargs['client_runner']([], {}, work, prompt, None, 1800)
        verifier(work)
        return {'outcome': 'verified', 'client': {'status': 'completed'}, 'workspace_changes': []}
    monkeypatch.setattr(runner, 'run_remote_recorded_session', record)
    result = runner.run_task(root, template, 'fake-secret')
    assert result['task_wall_s'] == 10.0
    assert result['client_execution_s'] == 5.0


def test_dry_run_never_reads_profile_or_runs_client(monkeypatch, capsys):
    runner = load_runner()
    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run side effect')
    monkeypatch.setattr(runner, 'read_credential', forbidden)
    monkeypatch.setattr(runner, 'run_task', forbidden)
    runner.main(['--pal-template', 'unused'])
    assert 'qwen3.8-27b-fp8' in capsys.readouterr().out
